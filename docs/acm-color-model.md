# Hue-sectored colour correction (ACM)

## Contents

- [Overview](#overview)
- [Data model](#data-model)
- [Per-pixel algorithm](#per-pixel-algorithm)
  - [1. Global correction and hue](#1-global-correction-and-hue)
  - [2. Sector selection](#2-sector-selection)
  - [3. Saturation fade](#3-saturation-fade)
  - [4. Effective matrix](#4-effective-matrix)
- [CCT interpolation](#cct-interpolation)
- [Constants](#constants)
- [Where it runs](#where-it-runs)
- [Notes and limits](#notes-and-limits)

## Overview

The colour step applies a colour-correction matrix (CCM) to the
white-balanced, linear RGB pixel. A single global 3x3 matrix corrects the
average colour response, but it cannot fit every hue at once: tightening reds
loosens blues, and so on.

The advanced colour matrices (ACM) refine this. In addition to the global
matrix, the tuning carries 24 per-hue-sector 3x3 matrices, each a full,
luminance-preserving correction tuned for colours whose hue falls in its
15-degree sector. Per pixel the renderer picks the matrix by the hue of the
colour, blends the two neighbouring sectors for a smooth result, and fades back
to the global matrix as the colour approaches neutral grey (where hue is
ill-defined and dominated by noise).

The result is more accurate colour across the hue circle than a single matrix,
with no hard sector boundaries and no instability on near-grey pixels.

## Data model

Generated into [`src/tuning_data.rs`](../src/tuning_data.rs) from
[`data/gc2607_acm.npz`](../data/gc2607_acm.npz) by
[`tools/gen_tuning.py`](../tools/gen_tuning.py):

| Constant        | Meaning                                                        |
|-----------------|---------------------------------------------------------------|
| `ACM_NSEC`      | number of hue sectors (24)                                     |
| `ACM_HUE0`      | centre hue of sector 0, degrees (7.5)                          |
| `ACM_HUE_STEP`  | hue step between adjacent sector centres, degrees (15.0)       |
| `ACM`           | `[NUM_CCM][ACM_NSEC]` matrices: per CCT, per sector, row-major 3x3 |

Each matrix is luminance-preserving (its rows sum to 1, so a neutral input maps
to a neutral output). The sectors are evenly spaced: sector `s` is centred at
`ACM_HUE0 + s * ACM_HUE_STEP` degrees, covering the full 360-degree hue circle.
The per-CCT sets are ordered by CCT to match the global CCMs (`CCM`, `CCM_CT`),
so both are interpolated by the same scene CCT.

## Per-pixel algorithm

Implemented in `pipeline::acm_color` ([`src/pipeline.rs`](../src/pipeline.rs))
and mirrored in the WGSL shader ([`src/gpu.rs`](../src/gpu.rs)) and the Python
reference ([`tools/reference_pipeline.py`](../tools/reference_pipeline.py)). The
input `rgb_wb` is the white-balanced linear pixel in 0..1 scale, after highlight
desaturation; `M_ccm` is the global CCM for the scene CCT; `M[s]` are the 24
sector matrices for the same CCT.

### 1. Global correction and hue

Compute the globally-corrected colour and take its hue and saturation:

```
g     = M_ccm * rgb_wb                 # linear, may be slightly out of [0,1]
c     = max(g, 0)                      # clamp negatives for the hue decision only
mx    = max(c.r, c.g, c.b)
mn    = min(c.r, c.g, c.b)
d     = mx - mn
if mx <= 0 or d == 0:  return g        # achromatic: global CCM only
hue   = HSV hue of c, in [0, 360)      # standard six-segment formula
sat   = d / mx                          # HSV saturation in [0, 1]
```

The hue is taken from the *globally-corrected* colour so the sectors line up
with the corrected output, not the raw sensor response. Negative components
(from out-of-gamut correction) are clamped before deciding the hue so a small
overshoot cannot flip the dominant channel.

### 2. Sector selection

Sector centres are evenly spaced, so the bracketing pair and the interpolation
fraction follow directly from the hue, wrapping around the circle:

```
pos   = (hue - ACM_HUE0) / ACM_HUE_STEP
frac  = pos - floor(pos)
a     = floor(pos) mod ACM_NSEC        # lower neighbour (circular)
b     = (a + 1) mod ACM_NSEC           # upper neighbour
M_sec = (1 - frac) * M[a] + frac * M[b]
```

Blending the two neighbouring sectors removes any visible seam at sector
boundaries.

### 3. Saturation fade

Near grey the hue is meaningless, so the correction fades to the stable global
matrix:

```
w = clamp(sat / ACM_SAT_KNEE, 0, 1)
```

`w = 0` at zero saturation (use the global CCM) and ramps to `w = 1` at and above
the saturation knee (full per-sector correction).

### 4. Effective matrix

The per-pixel matrix mixes the global and sector matrices by `w`, then corrects
the colour:

```
M_eff = (1 - w) * M_ccm + w * M_sec
out   = M_eff * rgb_wb
```

`out` is the corrected linear RGB; the sRGB gamma is applied afterwards as usual.

## CCT interpolation

The 24 sector matrices are interpolated by the scene CCT exactly as the global
CCM is (`pipeline::interp_acm` mirrors `interp_ccm`): the same calibration CCTs,
the same `searchsorted` bracket, and a linear blend between the two bracketing
CCT sets. This produces one set of 24 matrices per frame, built once and applied
to every pixel.

## Constants

| Name            | Value | Location                | Meaning                                  |
|-----------------|-------|-------------------------|------------------------------------------|
| `ACM_SAT_KNEE`  | 0.10  | `src/pipeline.rs`       | saturation at which the per-sector correction reaches full strength |
| `ACM_HUE0`      | 7.5   | `src/tuning_data.rs`    | centre hue of sector 0 (degrees)         |
| `ACM_HUE_STEP`  | 15.0  | `src/tuning_data.rs`    | hue step between sector centres (degrees) |
| `ACM_NSEC`      | 24    | `src/tuning_data.rs`    | number of sectors                        |

`ACM_SAT_KNEE` is a tuning parameter: lower values apply the per-sector
correction to fainter colours, higher values keep more near-grey pixels on the
global matrix.

## Where it runs

The same arithmetic runs on all three render paths, which the test suite keeps in
agreement:

- **CPU** — `pipeline::acm_color`, used by the half-res render, the full-res MHC
  render, and the lateral-CA render.
- **GPU** — `acm_apply` in the WGSL compute shader; the 24 CCT-interpolated
  matrices are uploaded per re-estimate, the rest read from the uniform block.
- **Reference** — `apply_acm` in the Python reference, the source of the golden
  image the CPU path is checked against.

`tests/golden.rs` pins the CPU render to the Python reference; `tests/gpu.rs`
checks the GPU path matches the CPU path within rounding.

## Notes and limits

- Hue and saturation use the standard HSV definitions on the linear,
  globally-corrected RGB. This is a deliberate, documented choice of hue space;
  if a different space proves more accurate against a colour target, only
  `acm_color` (and its two mirrors) need to change.
- The matrices preserve luminance (rows sum to 1), so the fade and the
  neighbour blend never shift overall brightness, only chroma.
- The full 24-matrix correction is the default on every render path. The cost is
  one extra 3x3 multiply and a hue computation per pixel.
