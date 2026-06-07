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


# Hue-sectored colour correction (ACM). Constants mirror pipeline.rs /
# tuning_data.rs; ACM_HUE0/STEP are derived from the sector edges in the npz.
ACM_SAT_KNEE = 0.10


def load_acm():
    """Per-sector matrices sorted by CCT (to align with the CCMs), plus the
    sector-centre hue origin/step. Returns (ccts, advanced, hue0, hue_step) or
    None if the file is absent."""
    path = os.path.join(DATA, "gc2607_acm.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    order = np.argsort(z["cct"].astype(np.int64))
    ccts = z["cct"].astype(np.float64)[order]
    advanced = z["advanced"].astype(np.float64)[order]      # (L,S,3,3)
    edges = np.concatenate([[0], z["hues"].astype(np.float64)])
    centres = (edges[:-1] + edges[1:]) / 2.0
    return ccts, advanced, float(centres[0]), float(centres[1] - centres[0])


def interp_acm(cct, ccts, advanced):
    """Interpolate the per-sector matrices by CCT (same bracket as interp_ccm).
    Returns (S,3,3)."""
    cct = float(np.clip(cct, ccts[0], ccts[-1]))
    j = int(np.searchsorted(ccts, cct))
    if j <= 0:
        return advanced[0]
    if j >= len(ccts):
        return advanced[-1]
    t = (cct - ccts[j - 1]) / (ccts[j] - ccts[j - 1])
    return advanced[j - 1] * (1 - t) + advanced[j] * t


def apply_acm(rgb01, ccm, sectors, hue0, hue_step):
    """Hue-sectored colour correction (mirrors pipeline::acm_color). `rgb01` is
    white-balanced linear RGB (..,3); `ccm` the global 3x3; `sectors` the (S,3,3)
    per-sector matrices for this CCT. The sector is chosen by the hue of the
    globally-corrected colour; the result fades to the global CCM at low
    saturation."""
    flat = rgb01.reshape(-1, 3)
    glob = flat @ ccm.T                                     # (P,3) global-corrected
    linc = np.clip(glob, 0.0, None)
    r, g, b = linc[:, 0], linc[:, 1], linc[:, 2]
    mx = linc.max(axis=1)
    mn = linc.min(axis=1)
    d = mx - mn

    safe = d > 0.0
    dd = np.where(safe, d, 1.0)
    is_r = (r >= g) & (r >= b)
    is_g = (~is_r) & (g >= b)
    hr = (g - b) / dd
    hg = 2.0 + (b - r) / dd
    hb = 4.0 + (r - g) / dd
    h = np.where(is_r, hr, np.where(is_g, hg, hb)) * 60.0
    h = np.where(h < 0.0, h + 360.0, h)
    h = np.where(safe, h, 0.0)

    sat = np.where(mx > 0.0, d / np.where(mx > 0.0, mx, 1.0), 0.0)
    w = np.clip(sat / ACM_SAT_KNEE, 0.0, 1.0)
    w = np.where(safe & (mx > 0.0), w, 0.0)                 # achromatic -> global

    nsec = sectors.shape[0]
    pos = (h - hue0) / hue_step
    fi = np.floor(pos)
    frac = pos - fi
    ia = (fi.astype(np.int64) % nsec + nsec) % nsec
    ib = (ia + 1) % nsec
    sflat = sectors.reshape(nsec, 9)
    ms = sflat[ia] * (1.0 - frac)[:, None] + sflat[ib] * frac[:, None]   # (P,9)
    ccmf = ccm.reshape(9)
    meff = ccmf[None, :] * (1.0 - w)[:, None] + ms * w[:, None]          # (P,9)
    m = meff.reshape(-1, 3, 3)
    out = np.einsum("pij,pj->pi", m, flat)
    return out.reshape(rgb01.shape)


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


# Highlight-desaturation knee; must match pipeline::HIGHLIGHT_KNEE (Rust/WGSL).
HIGHLIGHT_KNEE = 0.95


def desaturate_highlight(lin):
    """Blend each pixel toward its max channel as the max approaches full scale,
    so blown highlights converge to neutral white instead of taking a colour
    cast (green clips first -> otherwise bright whites tint magenta/purple).
    `lin` is linear post-white-balance RGB in 0..1 scale."""
    m = lin.max(axis=-1, keepdims=True)
    t = np.clip((m - HIGHLIGHT_KNEE) / (1.0 - HIGHLIGHT_KNEE), 0.0, 1.0)
    return lin + (m - lin) * t


def process(raw_path):
    """Run the full pipeline; return (rgb8, meta dict)."""
    ccts, locus, ccms = load_ccm_tuning()
    grids, lsc_chroma = load_lsc()
    acm = load_acm()

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
    lin = desaturate_highlight(rgb / MAXLIN)
    if acm is not None:
        a_ccts, advanced, hue0, hue_step = acm
        sectors = interp_acm(cct, a_ccts, advanced)
        corrected = apply_acm(lin, ccm, sectors, hue0, hue_step)
    else:
        corrected = apply_ccm(lin, ccm)
    out = srgb_gamma(corrected)
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
