#!/usr/bin/env python3
"""Linear-space channel-ratio probe for two fixed boxes (neutral wall, forehead
skin) on a rendered PNG. Exposure-invariant R/G and B/G are computed AFTER
inverse-sRGB (gamma-space ratios are exposure-confounded). Usage:
  measure_linear.py INT.png [USB.png]
"""
import sys
import numpy as np
from PIL import Image

# Neutral reference: content-based bright near-white pixels (white t-shirt /
# headphones / pillow), robust to the differing INT/USB framing. This is the
# legitimate white-balance reference (a fixed corner box lands on the warm
# bedding, not on neutral white). NEUTRAL_THR is a gamma-domain threshold.
NEUTRAL_THR = 140
# Skin patch as (x0, y0, x1, y1) frame fractions — the same face box in both
# cameras (verified visually during diagnosis). Keep identical across every
# measurement for comparability.
FACE = (0.44, 0.46, 0.56, 0.55)


def to_linear(rgb8):
    c = rgb8 / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def box_ratio(lin, box):
    h, w = lin.shape[:2]
    x0, y0, x1, y1 = box
    patch = lin[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].reshape(-1, 3)
    m = patch.mean(axis=0)
    return m[0] / m[1], m[2] / m[1]


def neutral_ratio(rgb8, lin):
    # Pixels near-white in all three gamma channels, averaged in linear.
    m = (rgb8[..., 0] > NEUTRAL_THR) & (rgb8[..., 1] > NEUTRAL_THR) & (rgb8[..., 2] > NEUTRAL_THR)
    px = lin[m]
    if px.shape[0] < 500:
        return float("nan"), float("nan"), 0
    mean = px.mean(axis=0)
    return mean[0] / mean[1], mean[2] / mean[1], int(px.shape[0])


def probe(path):
    rgb8 = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    lin = to_linear(rgb8)
    nrg, nbg, n = neutral_ratio(rgb8, lin)
    frg, fbg = box_ratio(lin, FACE)
    return dict(wall_rg=nrg, wall_bg=nbg, wall_n=n, fore_rg=frg, fore_bg=fbg)


def main():
    for tag, path in zip(("INT", "USB"), sys.argv[1:3]):
        r = probe(path)
        print(f"[{tag}] {path}")
        print(f"   neutral R/G={r['wall_rg']:.3f} B/G={r['wall_bg']:.3f}  (n={r['wall_n']})")
        print(f"   face    R/G={r['fore_rg']:.3f} B/G={r['fore_bg']:.3f}")


if __name__ == "__main__":
    main()
