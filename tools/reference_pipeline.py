#!/usr/bin/env python3
"""Reference ISP pipeline for GC2607 (self-contained Python golden source).

This mirrors the validated PoC pipeline and is the numeric reference the Rust
implementation must reproduce. Reads tuning from ../data (gc2607_ccms.json,
gc2607_lsc.npz); takes a raw SGRBG10 frame.

Pipeline: BLC(64) -> LSC(per Bayer channel, light source by scene chroma) ->
robust-neutral AWB -> half-res debayer -> CCM(interp by CCT) -> sRGB gamma.
AWB / CCT / LSC source are estimated on the pre-LSC (BLC-only) planes.

Raw geometry: SGRBG10, 1928x1088, stride 1952 u16/line (width padded), 10-bit
in 16-bit LE, Bayer GRBG (top-left Gr).
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

W, H = 1928, 1088
STRIDE_SAMPLES = 1952
BLACK = 64
WHITE = 1023
MAXLIN = float(WHITE - BLACK)        # 959


def load_raw(path):
    buf = np.fromfile(path, dtype="<u2")
    return buf.reshape(-1, STRIDE_SAMPLES)[:H, :W].astype(np.float32)


def black_level(raw):
    return np.clip(raw - BLACK, 0.0, None)


def bayer_planes(raw):
    """GRBG -> (Gr, R, B, Gb), each half-res."""
    return raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]


def planes_to_rgb(gr, r, b, gb):
    return np.stack([r, 0.5 * (gr + gb), b], axis=-1)


def m_robust_neutral(rgb):
    """Median chroma over bright, non-clipped pixels, with graceful fallbacks."""
    flat = rgb.reshape(-1, 3)
    g = flat[:, 1]
    not_clipped = flat.max(axis=1) < 0.95 * MAXLIN
    valid_g = g > 1.0
    for mask in (not_clipped & (g >= np.percentile(g, 60.0)) & valid_g,
                 not_clipped & valid_g,
                 (g >= np.percentile(g, 60.0)) & valid_g,
                 valid_g):
        if int(mask.sum()) >= 100:
            sel = flat[mask]
            return (float(np.median(sel[:, 0] / sel[:, 1])),
                    float(np.median(sel[:, 2] / sel[:, 1])))
    m = flat[valid_g].mean(axis=0)
    return m[0] / m[1], m[2] / m[1]


def gains_from_chroma(rg, bg):
    return np.array([1.0 / rg, 1.0, 1.0 / bg], dtype=np.float64)


def load_ccm_tuning():
    data = json.load(open(os.path.join(DATA, "gc2607_ccms.json")))
    data.sort(key=lambda c: c["cct"])
    ccts = np.array([c["cct"] for c in data], dtype=np.float64)
    locus = np.array([c["chroma"] for c in data], dtype=np.float64)
    ccms = np.array([c["ccm"] for c in data], dtype=np.float64).reshape(-1, 3, 3)
    return ccts, locus, ccms


def load_lsc():
    d = np.load(os.path.join(DATA, "gc2607_lsc.npz"), allow_pickle=True)
    return d["grids"].astype(np.float64), d["chroma"].astype(np.float64)


def project_to_locus(chroma, locus):
    best = (1e18, locus[0], 0, 0.0)
    p = np.array(chroma)
    for i in range(len(locus) - 1):
        a, b = locus[i], locus[i + 1]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
        proj = a + t * ab
        dd = np.linalg.norm(p - proj)
        if dd < best[0]:
            best = (dd, proj, i, t)
    return best


def estimate_cct(rg, bg, ccts, locus):
    _, _, i, t = project_to_locus((rg, bg), locus)
    return ccts[i] * (1 - t) + ccts[i + 1] * t


def interp_ccm(cct, ccts, ccms):
    cct = float(np.clip(cct, ccts[0], ccts[-1]))
    j = int(np.searchsorted(ccts, cct))
    if j <= 0:
        return ccms[0]
    if j >= len(ccts):
        return ccms[-1]
    t = (cct - ccts[j - 1]) / (ccts[j] - ccts[j - 1])
    return ccms[j - 1] * (1 - t) + ccms[j] * t


def select_ls(chroma, lsc_chroma):
    d = np.linalg.norm(lsc_chroma - np.array(chroma)[None, :], axis=1)
    return int(np.argmin(d))


def bilinear_resize(grid, out_h, out_w):
    gh, gw = grid.shape
    ys = np.linspace(0, gh - 1, out_h)
    xs = np.linspace(0, gw - 1, out_w)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, gh - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, gw - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    top = grid[y0][:, x0] * (1 - wx) + grid[y0][:, x1] * wx
    bot = grid[y1][:, x0] * (1 - wx) + grid[y1][:, x1] * wx
    return top * (1 - wy) + bot * wy


def apply_ccm(rgb01, ccm):
    out = rgb01.reshape(-1, 3) @ ccm.T
    return out.reshape(rgb01.shape)


def srgb_gamma(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x,
                    1.055 * np.power(x, 1 / 2.4) - 0.055)


def process(raw_path):
    """Run the full pipeline; return (rgb8, meta dict)."""
    ccts, locus, ccms = load_ccm_tuning()
    grids, lsc_chroma = load_lsc()

    raw = black_level(load_raw(raw_path))
    gr, r, b, gb = bayer_planes(raw)
    hh, ww = gr.shape

    rgb_lin = planes_to_rgb(gr, r, b, gb)               # pre-LSC, for estimation
    rg, bg = m_robust_neutral(rgb_lin)
    gains = gains_from_chroma(rg, bg)
    cct = estimate_cct(rg, bg, ccts, locus)
    ccm = interp_ccm(cct, ccts, ccms)
    ls = select_ls((rg, bg), lsc_chroma)

    gmaps = [bilinear_resize(grids[ls, c], hh, ww) for c in range(4)]  # Gr,R,B,Gb
    grc, rc, bc, gbc = gr * gmaps[0], r * gmaps[1], b * gmaps[2], gb * gmaps[3]

    rgb = planes_to_rgb(grc, rc, bc, gbc) * gains[None, None, :]
    out = srgb_gamma(apply_ccm(rgb / MAXLIN, ccm))
    rgb8 = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)

    meta = {
        "width": ww, "height": hh,
        "scene_chroma": [rg, bg],
        "gains": gains.tolist(),
        "cct": float(cct),
        "lsc_ls": ls,
        "ccm": ccm.flatten().tolist(),
    }
    return rgb8, meta


if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(HERE, "..", "tests", "data", "sample-raw.bin")
    rgb8, meta = process(raw)
    print(json.dumps(meta, indent=2))
    print("rgb8 shape:", rgb8.shape, "mean:", float(rgb8.mean()))
