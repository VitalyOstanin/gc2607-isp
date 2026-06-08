//! ISP processing core (CPU). Mirrors the validated Python reference
//! (`tools/reference_pipeline.py`) closely enough to pass the golden test.

use crate::raw::{RawFrame, H, MAXLIN, W};
use crate::tuning;
use crate::tuning_data::{
    ACM, ACM_HUE0, ACM_HUE_STEP, ACM_NSEC, CCM, CCM_CT, CCM_LOCUS, LCA_CELL_X, LCA_CELL_Y, LCA_GH,
    LCA_GW, LSC_CHROMA, LSC_GH, LSC_GW, NUM_CCM,
};

#[cfg(feature = "video")]
use rayon::prelude::*;

/// Apply `body` to each row of `buf` (`row` elements per row), data-parallel
/// with the `video` feature, serial otherwise. Rows are independent, so the
/// result is identical either way (the golden test still holds). Parallelism is
/// row-granular (not per-pixel) to keep each task substantial.
fn for_each_row_mut<T: Send>(buf: &mut [T], row: usize, body: impl Fn(usize, &mut [T]) + Sync + Send) {
    #[cfg(feature = "video")]
    buf.par_chunks_mut(row).enumerate().for_each(|(y, r)| body(y, r));
    #[cfg(not(feature = "video"))]
    buf.chunks_mut(row).enumerate().for_each(|(y, r)| body(y, r));
}

/// Half-resolution Bayer channel planes (GRBG). Each is `hh*ww`, row-major.
pub struct Planes {
    pub hh: usize,
    pub ww: usize,
    pub gr: Vec<f32>,
    pub r: Vec<f32>,
    pub b: Vec<f32>,
    pub gb: Vec<f32>,
}

/// Scene estimation result: white-balance gains, CCT, chosen LSC light source,
/// and the interpolated colour-correction matrix.
#[derive(Debug, Clone)]
pub struct Estimate {
    pub chroma: (f32, f32), // r/g, b/g
    pub gains: [f64; 3],    // R, G(=1), B
    pub cct: f64,
    pub ls: usize,
    pub ccm: [f64; 9],
}

/// Split a black-level-corrected frame into half-res GRBG planes.
pub fn bayer_planes(raw: &RawFrame) -> Planes {
    debug_assert_eq!((raw.w, raw.h), (W, H));
    let ww = raw.w / 2;
    let hh = raw.h / 2;
    let mut gr = vec![0f32; hh * ww];
    let mut r = vec![0f32; hh * ww];
    let mut b = vec![0f32; hh * ww];
    let mut gb = vec![0f32; hh * ww];
    for y in 0..hh {
        let r0 = (2 * y) * raw.w;
        let r1 = (2 * y + 1) * raw.w;
        let o = y * ww;
        for x in 0..ww {
            gr[o + x] = raw.data[r0 + 2 * x];
            r[o + x] = raw.data[r0 + 2 * x + 1];
            b[o + x] = raw.data[r1 + 2 * x];
            gb[o + x] = raw.data[r1 + 2 * x + 1];
        }
    }
    Planes { hh, ww, gr, r, b, gb }
}

/// numpy-compatible linear-interpolation percentile over a sorted slice.
fn percentile_sorted(sorted: &[f32], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0] as f64;
    }
    let rank = (n as f64 - 1.0) * q / 100.0;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    let a = sorted[lo] as f64;
    let b = sorted[hi] as f64;
    a + (b - a) * frac
}

/// Median of a slice (numpy-compatible: average of the two middles when even).
fn median(values: &mut [f32]) -> f64 {
    let n = values.len();
    // total_cmp orders NaN deterministically instead of panicking on partial_cmp.
    values.sort_by(f32::total_cmp);
    if n % 2 == 1 {
        values[n / 2] as f64
    } else {
        (values[n / 2 - 1] as f64 + values[n / 2] as f64) / 2.0
    }
}

/// Temporal AWB smoothing factor (exponential moving average on the scene
/// chroma). The live runtimes re-estimate every few frames; without smoothing
/// each estimate is applied as a hard step, which reads as the white balance
/// "resetting" periodically. `new = old + ALPHA * (measured - old)`; smaller is
/// smoother (slower to follow a real lighting change). Stateless callers
/// (`estimate`, the golden path) do not smooth.
pub(crate) const AWB_SMOOTH_ALPHA: f32 = 0.3;

/// LSC light-source switch hysteresis. The chosen source only changes when a
/// different one is at least `1/LS_HYST` times closer in chroma than the current
/// one; otherwise the current source is kept. Prevents the whole lens-shading
/// grid (and its field-wide brightness/colour) from flipping back and forth when
/// the scene chroma sits between two calibrated sources.
pub(crate) const LS_HYST: f64 = 0.8;

/// Highlight-desaturation knee in linear post-white-balance scale (0..1). Above
/// this level a pixel is blended toward its own max channel, so blown highlights
/// converge to neutral white instead of taking a colour cast — green clips first
/// (highest sensitivity), so without this, bright whites pick up a magenta/purple
/// tint once R and B are gained past the clipped green.
pub(crate) const HIGHLIGHT_KNEE: f64 = 0.95;

/// Blend a linear (post-WB, 0..1-scale) RGB triple toward its max channel as the
/// max approaches full scale, removing the colour cast on near-clipped
/// highlights. Below [`HIGHLIGHT_KNEE`] it is a no-op. Mirrored in the WGSL
/// shader (`gpu.rs`) and the Python reference so all render paths agree.
#[inline(always)]
pub(crate) fn desaturate_highlight(r: &mut f64, g: &mut f64, b: &mut f64) {
    let m = r.max(*g).max(*b);
    if m > HIGHLIGHT_KNEE {
        let t = ((m - HIGHLIGHT_KNEE) / (1.0 - HIGHLIGHT_KNEE)).clamp(0.0, 1.0);
        *r += (m - *r) * t;
        *g += (m - *g) * t;
        *b += (m - *b) * t;
    }
}

/// Fraction of full scale above which a pixel is treated as clipped and excluded
/// from the AWB statistics. Numerically equal to [`HIGHLIGHT_KNEE`] but a
/// distinct concept (clip rejection vs highlight desaturation).
const AWB_CLIP_FRACTION: f32 = 0.95;

/// Minimum number of pixels a mask must select before its median chroma is
/// trusted; below this the next, looser mask is tried.
const AWB_MIN_SAMPLES: usize = 100;

/// Robust-neutral AWB: median chroma over bright, non-clipped pixels, with
/// graceful fallbacks so the selection never collapses to empty.
pub(crate) fn robust_neutral(p: &Planes) -> (f32, f32) {
    let n = p.hh * p.ww;
    let green: Vec<f32> = (0..n).map(|i| 0.5 * (p.gr[i] + p.gb[i])).collect();
    let clip = AWB_CLIP_FRACTION * MAXLIN;

    let mut sorted = green.clone();
    sorted.sort_by(f32::total_cmp);
    let p60 = percentile_sorted(&sorted, 60.0) as f32;

    let not_clipped: Vec<bool> = (0..n)
        .map(|i| p.r[i].max(green[i]).max(p.b[i]) < clip)
        .collect();
    let valid: Vec<bool> = green.iter().map(|&g| g > 1.0).collect();

    let masks: [Box<dyn Fn(usize) -> bool>; 4] = [
        Box::new(|i| not_clipped[i] && green[i] >= p60 && valid[i]),
        Box::new(|i| not_clipped[i] && valid[i]),
        Box::new(|i| green[i] >= p60 && valid[i]),
        Box::new(|i| valid[i]),
    ];

    for mask in masks.iter() {
        let idx: Vec<usize> = (0..n).filter(|&i| mask(i)).collect();
        if idx.len() >= AWB_MIN_SAMPLES {
            let mut rg: Vec<f32> = idx.iter().map(|&i| p.r[i] / green[i]).collect();
            let mut bg: Vec<f32> = idx.iter().map(|&i| p.b[i] / green[i]).collect();
            return (median(&mut rg) as f32, median(&mut bg) as f32);
        }
    }

    // last resort: mean over valid green
    let idx: Vec<usize> = (0..n).filter(|&i| valid[i]).collect();
    let (mut sr, mut sg, mut sb) = (0f64, 0f64, 0f64);
    for &i in &idx {
        sr += p.r[i] as f64;
        sg += green[i] as f64;
        sb += p.b[i] as f64;
    }
    // Degenerate (near-black) frame with no valid green: assume neutral rather
    // than dividing by zero and propagating NaN through the gains.
    if sg <= 0.0 {
        return (1.0, 1.0);
    }
    ((sr / sg) as f32, (sb / sg) as f32)
}

/// Project chromaticity onto the locus polyline; return (segment index, t).
fn project_to_locus(rg: f64, bg: f64) -> (usize, f64) {
    // Distance from (rg, bg) to segment i, with the clamped projection parameter.
    let segment = |i: usize| {
        let ax = CCM_LOCUS[i][0] as f64;
        let ay = CCM_LOCUS[i][1] as f64;
        let bx = CCM_LOCUS[i + 1][0] as f64;
        let by = CCM_LOCUS[i + 1][1] as f64;
        let abx = bx - ax;
        let aby = by - ay;
        let denom = abx * abx + aby * aby;
        let t = (((rg - ax) * abx + (bg - ay) * aby) / denom).clamp(0.0, 1.0);
        let d = (rg - (ax + t * abx)).hypot(bg - (ay + t * aby));
        (d, i, t)
    };
    (0..NUM_CCM - 1)
        .map(segment)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, i, t)| (i, t))
        .unwrap_or((0, 0.0))
}

fn estimate_cct(rg: f64, bg: f64) -> f64 {
    let (i, t) = project_to_locus(rg, bg);
    CCM_CT[i] as f64 * (1.0 - t) + CCM_CT[i + 1] as f64 * t
}

fn interp_ccm(cct: f64) -> [f64; 9] {
    let lo = CCM_CT[0] as f64;
    let hi = CCM_CT[NUM_CCM - 1] as f64;
    let c = cct.clamp(lo, hi);
    // searchsorted: first index j with CCM_CT[j] >= c (numpy default 'left')
    let mut j = NUM_CCM;
    for (k, &ct) in CCM_CT.iter().enumerate() {
        if (ct as f64) >= c {
            j = k;
            break;
        }
    }
    if j == 0 {
        return f9(&CCM[0]);
    }
    if j >= NUM_CCM {
        return f9(&CCM[NUM_CCM - 1]);
    }
    let t = (c - CCM_CT[j - 1] as f64) / (CCM_CT[j] as f64 - CCM_CT[j - 1] as f64);
    let (lo, hi) = (&CCM[j - 1], &CCM[j]);
    let mut out = [0f64; 9];
    for (o, (&a, &b)) in out.iter_mut().zip(lo.iter().zip(hi.iter())) {
        *o = a as f64 * (1.0 - t) + b as f64 * t;
    }
    out
}

fn f9(m: &[f32; 9]) -> [f64; 9] {
    let mut o = [0f64; 9];
    for k in 0..9 {
        o[k] = m[k] as f64;
    }
    o
}

// ---------------------------------------------------------------------------
// Advanced colour matrices (ACM): hue-sectored colour correction.
//
// Beyond the single global CCM, the tuning carries 24 per-hue-sector 3x3
// matrices (tuning_data::ACM, one set per calibration CCT). Each is a full,
// luminance-preserving colour matrix that refines the correction for colours in
// its hue range; the global CCM is the achromatic/neutral fallback. Per pixel we
// pick the matrix by the hue of the globally-corrected colour, blend the two
// neighbouring sectors, and fade toward the global CCM as saturation drops (the
// hue of a near-grey pixel is ill-defined and noisy). The exact device hue model
// is fixed-function in the camera's image processor and is not reproduced
// bit-for-bit; this is a faithful, fully-parameterised software equivalent. The
// arithmetic here is mirrored in the WGSL shader (`gpu.rs`) and the Python
// reference (`tools/reference_pipeline.py`) so all render paths agree.
// ---------------------------------------------------------------------------

/// Saturation knee for ACM: below this HSV saturation the global CCM is used,
/// ramping linearly to full per-sector correction at/above it. Keeps near-grey
/// pixels (whose hue is dominated by noise) on the stable global matrix.
pub(crate) const ACM_SAT_KNEE: f64 = 0.10;

/// The 24 per-sector colour matrices interpolated to one scene CCT, ready to
/// apply per pixel. Built once per frame by [`interp_acm`].
pub struct AcmFrame {
    pub mats: [[f64; 9]; ACM_NSEC],
}

/// Interpolate the per-sector matrices by CCT, identically to [`interp_ccm`]
/// (same calibration CCTs, same `searchsorted` bracket and linear blend).
pub fn interp_acm(cct: f64) -> AcmFrame {
    let lo = CCM_CT[0] as f64;
    let hi = CCM_CT[NUM_CCM - 1] as f64;
    let c = cct.clamp(lo, hi);
    let mut j = NUM_CCM;
    for (k, &ct) in CCM_CT.iter().enumerate() {
        if (ct as f64) >= c {
            j = k;
            break;
        }
    }
    let mut mats = [[0f64; 9]; ACM_NSEC];
    if j == 0 {
        for (s, m) in mats.iter_mut().enumerate() {
            *m = f9(&ACM[0][s]);
        }
        return AcmFrame { mats };
    }
    if j >= NUM_CCM {
        for (s, m) in mats.iter_mut().enumerate() {
            *m = f9(&ACM[NUM_CCM - 1][s]);
        }
        return AcmFrame { mats };
    }
    let t = (c - CCM_CT[j - 1] as f64) / (CCM_CT[j] as f64 - CCM_CT[j - 1] as f64);
    for (s, m) in mats.iter_mut().enumerate() {
        let (a, b) = (&ACM[j - 1][s], &ACM[j][s]);
        for k in 0..9 {
            m[k] = a[k] as f64 * (1.0 - t) + b[k] as f64 * t;
        }
    }
    AcmFrame { mats }
}

/// Apply the hue-sectored colour correction to one white-balanced linear pixel
/// `(rl, gl, bl)` (0..1 scale, post-highlight-desaturation). `ccm` is the global
/// matrix; `acm` the per-sector matrices for this frame's CCT. Returns the
/// corrected linear RGB (before sRGB gamma). The hue/saturation that select the
/// sector are taken from the globally-corrected colour `ccm * rgb`.
#[inline(always)]
pub(crate) fn acm_color(rl: f64, gl: f64, bl: f64, ccm: &[f64; 9], acm: &AcmFrame) -> (f64, f64, f64) {
    let g0 = ccm[0] * rl + ccm[1] * gl + ccm[2] * bl;
    let g1 = ccm[3] * rl + ccm[4] * gl + ccm[5] * bl;
    let g2 = ccm[6] * rl + ccm[7] * gl + ccm[8] * bl;

    // Hue/saturation of the globally-corrected colour (negatives clamped: a
    // slightly out-of-gamut corrected value should not flip the dominant hue).
    let lr = g0.max(0.0);
    let lg = g1.max(0.0);
    let lb = g2.max(0.0);
    let mx = lr.max(lg).max(lb);
    let mn = lr.min(lg).min(lb);
    let d = mx - mn;
    if mx <= 0.0 || d <= 0.0 {
        return (g0, g1, g2); // achromatic: global CCM only
    }

    let mut h = if lr >= lg && lr >= lb {
        (lg - lb) / d
    } else if lg >= lb {
        2.0 + (lb - lr) / d
    } else {
        4.0 + (lr - lg) / d
    };
    h *= 60.0;
    if h < 0.0 {
        h += 360.0;
    }
    let sat = d / mx;
    let w = (sat / ACM_SAT_KNEE).clamp(0.0, 1.0);

    // Sector bracket (centres evenly spaced; circular). `pos` is the fractional
    // sector position; the floor's modulo gives the lower neighbour, wrapping
    // across the hue circle.
    let pos = (h - ACM_HUE0 as f64) / ACM_HUE_STEP as f64;
    let fi = pos.floor();
    let frac = pos - fi;
    let ia = (fi as i64).rem_euclid(ACM_NSEC as i64) as usize;
    let ib = (ia + 1) % ACM_NSEC;
    let (ma, mb) = (&acm.mats[ia], &acm.mats[ib]);

    // Effective matrix: blend the two neighbouring sectors, then fade between the
    // global CCM (low saturation) and the sector matrix (high saturation).
    let mut m = [0f64; 9];
    for k in 0..9 {
        let ms = ma[k] * (1.0 - frac) + mb[k] * frac;
        m[k] = ccm[k] * (1.0 - w) + ms * w;
    }
    (
        m[0] * rl + m[1] * gl + m[2] * bl,
        m[3] * rl + m[4] * gl + m[5] * bl,
        m[6] * rl + m[7] * gl + m[8] * bl,
    )
}

/// Chroma distance from scene `(rg, bg)` to LSC light source `k`.
fn ls_dist(k: usize, rg: f64, bg: f64) -> f64 {
    (LSC_CHROMA[k][0] as f64 - rg).hypot(LSC_CHROMA[k][1] as f64 - bg)
}

/// Choose the LSC light source nearest the scene chroma, with hysteresis: when
/// `prev` is given, keep it unless another source is at least `1/LS_HYST` times
/// closer (see [`LS_HYST`]). `prev = None` (stateless callers) picks the plain
/// nearest source, matching the original `select_ls`.
fn select_ls_hyst(rg: f64, bg: f64, prev: Option<usize>) -> usize {
    let best = (0..LSC_CHROMA.len())
        .min_by(|&a, &b| ls_dist(a, rg, bg).total_cmp(&ls_dist(b, rg, bg)))
        .unwrap_or(0);
    match prev {
        Some(p) if ls_dist(p, rg, bg) <= ls_dist(best, rg, bg) / LS_HYST => p,
        _ => best,
    }
}

/// Exponentially smooth a freshly measured scene chroma against the previous
/// (smoothed) value; `None` returns the measurement unchanged (first frame).
pub(crate) fn smooth_chroma(prev: Option<(f32, f32)>, rg: f32, bg: f32) -> (f32, f32) {
    match prev {
        Some((p0, p1)) => (
            p0 + AWB_SMOOTH_ALPHA * (rg - p0),
            p1 + AWB_SMOOTH_ALPHA * (bg - p1),
        ),
        None => (rg, bg),
    }
}

/// Build a full scene estimate (gains, CCT, LSC source, CCM) from an already
/// chosen scene chroma. `prev_ls` enables LSC switch hysteresis for the live
/// runtimes; stateless callers pass `None`.
pub(crate) fn estimate_from_chroma(rg: f32, bg: f32, prev_ls: Option<usize>) -> Estimate {
    // Guard against a degenerate chroma (zero/NaN from a black frame): fall back
    // to neutral so the gains stay finite instead of becoming inf/NaN.
    let rg = if rg.is_finite() && rg > 0.0 { rg } else { 1.0 };
    let bg = if bg.is_finite() && bg > 0.0 { bg } else { 1.0 };
    let gains = [1.0 / rg as f64, 1.0, 1.0 / bg as f64];
    let cct = estimate_cct(rg as f64, bg as f64);
    let ls = select_ls_hyst(rg as f64, bg as f64, prev_ls);
    let ccm = interp_ccm(cct);
    Estimate { chroma: (rg, bg), gains, cct, ls, ccm }
}

/// Estimate WB gains, CCT, LSC light source and CCM from the (pre-LSC) planes.
/// Stateless (no temporal smoothing, plain nearest LSC source) — this is the
/// reference path the golden test pins; the live runtimes smooth on top of it.
pub fn estimate(p: &Planes) -> Estimate {
    let (rg, bg) = robust_neutral(p);
    estimate_from_chroma(rg, bg, None)
}

/// Bilinear resize of a `gh*gw` grid to `out_h*out_w` (numpy linspace semantics).
fn resize_grid(grid: &[f32], gh: usize, gw: usize, out_h: usize, out_w: usize) -> Vec<f64> {
    let sy = if out_h > 1 { (gh as f64 - 1.0) / (out_h as f64 - 1.0) } else { 0.0 };
    let sx = if out_w > 1 { (gw as f64 - 1.0) / (out_w as f64 - 1.0) } else { 0.0 };
    let mut out = vec![0f64; out_h * out_w];
    for oy in 0..out_h {
        let fy = oy as f64 * sy;
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(gh - 1);
        let wy = fy - y0 as f64;
        for ox in 0..out_w {
            let fx = ox as f64 * sx;
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(gw - 1);
            let wx = fx - x0 as f64;
            let g00 = grid[y0 * gw + x0] as f64;
            let g01 = grid[y0 * gw + x1] as f64;
            let g10 = grid[y1 * gw + x0] as f64;
            let g11 = grid[y1 * gw + x1] as f64;
            let top = g00 * (1.0 - wx) + g01 * wx;
            let bot = g10 * (1.0 - wx) + g11 * wx;
            out[oy * out_w + ox] = top * (1.0 - wy) + bot * wy;
        }
    }
    out
}

/// sRGB OETF (linear -> gamma) breakpoint and coefficients. The WGSL shader
/// (`gpu.rs` `srgb`) and the Python reference mirror these literals; keep all
/// three in sync.
const SRGB_LIN_THRESH: f64 = 0.003_130_8;
const SRGB_LIN_SLOPE: f64 = 12.92;
const SRGB_SCALE: f64 = 1.055;
const SRGB_GAMMA: f64 = 1.0 / 2.4;
const SRGB_OFFSET: f64 = 0.055;

fn srgb(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    if x <= SRGB_LIN_THRESH {
        SRGB_LIN_SLOPE * x
    } else {
        SRGB_SCALE * x.powf(SRGB_GAMMA) - SRGB_OFFSET
    }
}

/// Inverse sRGB EOTF (gamma-encoded `0..1` -> linear `0..1`): the exact inverse
/// of [`srgb`], reusing the same breakpoint and coefficients. The live AE loop
/// uses this to meter the produced frame's mean luminance in linear light, so
/// the exposure model (which assumes brightness is proportional to the
/// exposure-gain product) operates on a linear quantity rather than the
/// gamma-encoded luma.
pub fn srgb_to_linear(y: f64) -> f64 {
    let y = y.clamp(0.0, 1.0);
    if y <= SRGB_LIN_SLOPE * SRGB_LIN_THRESH {
        y / SRGB_LIN_SLOPE
    } else {
        ((y + SRGB_OFFSET) / SRGB_SCALE).powf(1.0 / SRGB_GAMMA)
    }
}

/// Pre-resized per-Bayer-channel LSC gain grids (`hh*ww` each) for one light
/// source. Building these is a fixed per-frame cost that depends only on the
/// chosen light source, so [`Processor`] caches a `Grids` and rebuilds it only
/// when the selected source changes.
pub struct Grids {
    pub ls: usize,
    pub hh: usize,
    pub ww: usize,
    pub g_gr: Vec<f64>,
    pub g_r: Vec<f64>,
    pub g_b: Vec<f64>,
    pub g_gb: Vec<f64>,
}

/// Resize the four LSC grids for light source `ls` to `hh*ww`.
pub fn build_grids(ls: usize, hh: usize, ww: usize) -> Grids {
    Grids {
        ls,
        hh,
        ww,
        g_gr: resize_grid(&tuning::lsc_grid(ls, 0), LSC_GH, LSC_GW, hh, ww),
        g_r: resize_grid(&tuning::lsc_grid(ls, 1), LSC_GH, LSC_GW, hh, ww),
        g_b: resize_grid(&tuning::lsc_grid(ls, 2), LSC_GH, LSC_GW, hh, ww),
        g_gb: resize_grid(&tuning::lsc_grid(ls, 3), LSC_GH, LSC_GW, hh, ww),
    }
}

/// Core half-res render into a caller-owned buffer using pre-resized grids.
/// `out` must be `p.hh*p.ww*3` bytes. Row-parallel with the `video` feature.
fn render_half_into(out: &mut [u8], p: &Planes, gains: [f64; 3], ccm: [f64; 9], grids: &Grids, acm: &AcmFrame) {
    let inv = 1.0 / MAXLIN as f64;
    let ww = p.ww;
    let (g_gr, g_r, g_b, g_gb) = (&grids.g_gr, &grids.g_r, &grids.g_b, &grids.g_gb);
    for_each_row_mut(out, ww * 3, |y, orow| {
        let base = y * ww;
        for x in 0..ww {
            let i = base + x;
            let rc = p.r[i] as f64 * g_r[i];
            let bc = p.b[i] as f64 * g_b[i];
            let grc = p.gr[i] as f64 * g_gr[i];
            let gbc = p.gb[i] as f64 * g_gb[i];
            let gc = 0.5 * (grc + gbc);

            let mut rl = rc * gains[0] * inv;
            let mut gl = gc * gains[1] * inv;
            let mut bl = bc * gains[2] * inv;
            desaturate_highlight(&mut rl, &mut gl, &mut bl);

            let (cr, cg, cb) = acm_color(rl, gl, bl, &ccm, acm);

            orow[x * 3] = (srgb(cr) * 255.0 + 0.5) as u8;
            orow[x * 3 + 1] = (srgb(cg) * 255.0 + 0.5) as u8;
            orow[x * 3 + 2] = (srgb(cb) * 255.0 + 0.5) as u8;
        }
    });
}

/// Render RGB8 (row-major `hh*ww*3`) from planes using given gains/CCM/LSC src.
/// Pipeline: LSC -> half-res debayer -> WB -> hue-sectored CCM -> sRGB gamma.
/// `cct` selects the per-sector matrices (and must be the same CCT the `ccm` was
/// interpolated for).
pub fn render(p: &Planes, gains: [f64; 3], ccm: [f64; 9], cct: f64, ls: usize) -> Vec<u8> {
    let grids = build_grids(ls, p.hh, p.ww);
    let acm = interp_acm(cct);
    let mut out = vec![0u8; p.hh * p.ww * 3];
    render_half_into(&mut out, p, gains, ccm, &grids, &acm);
    out
}

/// Debayer strategy. `HalfRes` collapses each 2x2 GRBG tile to one pixel
/// (half resolution, no interpolation; this is the golden-checked path).
/// `Mhc` is full-resolution Malvar-He-Cutler (our own row-parallel
/// implementation; see [`mhc_rgb_at`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebayerMode {
    HalfRes,
    Mhc,
}

/// Clamped CFA sample access (edge replication at borders).
#[inline(always)]
fn cfa_at(cfa: &[f32], w: usize, h: usize, y: isize, x: isize) -> f32 {
    let yy = y.clamp(0, h as isize - 1) as usize;
    let xx = x.clamp(0, w as isize - 1) as usize;
    cfa[yy * w + xx]
}

/// Malvar-He-Cutler "high-quality linear interpolation" of the full RGB triple
/// at `(y, x)` on a GRBG-mosaic CFA (`cfa` is `w*h`, row-major). Implements the
/// five 5x5 gradient-corrected filters from Malvar, He & Cutler (2004), all with
/// the canonical 1/8 normalisation; edges use replication via [`cfa_at`]. This
/// replaces the external `demosaic` crate so the debayer can run row-parallel
/// and fuse with the colour stage. Returns linear `(R, G, B)` in CFA scale.
#[inline]
fn mhc_rgb_at(cfa: &[f32], w: usize, h: usize, y: usize, x: usize) -> (f32, f32, f32) {
    let (yi, xi) = (y as isize, x as isize);
    let c = cfa[y * w + x];
    let p = |dy: isize, dx: isize| cfa_at(cfa, w, h, yi + dy, xi + dx);

    let up = p(-1, 0);
    let dn = p(1, 0);
    let lf = p(0, -1);
    let rt = p(0, 1);
    let uu = p(-2, 0);
    let dd = p(2, 0);
    let ll = p(0, -2);
    let rr = p(0, 2);
    let ul = p(-1, -1);
    let ur = p(-1, 1);
    let dl = p(1, -1);
    let dr = p(1, 1);
    let diag = ul + ur + dl + dr;

    // Filter A: G at a R/B centre.
    let g_at_rb = (4.0 * c + 2.0 * (up + dn + lf + rt) - (uu + dd + ll + rr)) / 8.0;
    // Filter B: colour at G, the wanted colour lying on the same row (horizontal).
    let at_g_hrow = (5.0 * c + 4.0 * (lf + rt) - (ll + rr) - diag + 0.5 * (uu + dd)) / 8.0;
    // Filter C: colour at G, the wanted colour lying on the same column (vertical).
    let at_g_vcol = (5.0 * c + 4.0 * (up + dn) - (uu + dd) - diag + 0.5 * (ll + rr)) / 8.0;
    // Filter D: colour at the opposite primary (diagonal neighbours).
    let at_diag = (6.0 * c + 2.0 * diag - 1.5 * (uu + dd + ll + rr)) / 8.0;

    // GRBG: (even,even)=Gr, (even,odd)=R, (odd,even)=B, (odd,odd)=Gb.
    match (y & 1, x & 1) {
        (0, 0) => (at_g_hrow, c, at_g_vcol), // Gr: R is horizontal, B is vertical
        (0, 1) => (c, g_at_rb, at_diag),     // R:  G via A, B via diagonal
        (1, 0) => (at_diag, g_at_rb, c),     // B:  R via diagonal, G via A
        _ => (at_g_vcol, c, at_g_hrow),      // Gb: R is vertical, B is horizontal
    }
}

/// MHC interpolation for an interior pixel (>=2 from every edge), using direct
/// offsets from the centre index `i = y*w+x` with no bounds clamping. This is
/// the hot path; `(yp, xp)` are `(y & 1, x & 1)`.
#[inline(always)]
fn mhc_rgb_interior(cfa: &[f32], w: usize, i: usize, yp: usize, xp: usize) -> (f32, f32, f32) {
    let c = cfa[i];
    let up = cfa[i - w];
    let dn = cfa[i + w];
    let lf = cfa[i - 1];
    let rt = cfa[i + 1];
    let uu = cfa[i - 2 * w];
    let dd = cfa[i + 2 * w];
    let ll = cfa[i - 2];
    let rr = cfa[i + 2];
    let ul = cfa[i - w - 1];
    let ur = cfa[i - w + 1];
    let dl = cfa[i + w - 1];
    let dr = cfa[i + w + 1];
    let diag = ul + ur + dl + dr;

    let g_at_rb = (4.0 * c + 2.0 * (up + dn + lf + rt) - (uu + dd + ll + rr)) / 8.0;
    let at_g_hrow = (5.0 * c + 4.0 * (lf + rt) - (ll + rr) - diag + 0.5 * (uu + dd)) / 8.0;
    let at_g_vcol = (5.0 * c + 4.0 * (up + dn) - (uu + dd) - diag + 0.5 * (ll + rr)) / 8.0;
    let at_diag = (6.0 * c + 2.0 * diag - 1.5 * (uu + dd + ll + rr)) / 8.0;

    match (yp, xp) {
        (0, 0) => (at_g_hrow, c, at_g_vcol),
        (0, 1) => (c, g_at_rb, at_diag),
        (1, 0) => (at_diag, g_at_rb, c),
        _ => (at_g_vcol, c, at_g_hrow),
    }
}

/// Number of segments in the sRGB-gamma lookup table (linear-interpolated).
const SRGB_LUT_N: usize = 4096;

/// sRGB OETF scaled to 0..255, sampled on `[0,1]` (one extra endpoint sample so
/// interpolation at v=1 is in range). Built once. Used by the full-res MHC path
/// to avoid three `powf` calls per pixel; the half-res path keeps exact `srgb`
/// so the golden render stays bit-stable.
fn srgb255_lut() -> &'static [f32; SRGB_LUT_N + 1] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f32; SRGB_LUT_N + 1]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; SRGB_LUT_N + 1];
        for (k, e) in t.iter_mut().enumerate() {
            *e = (srgb(k as f64 / SRGB_LUT_N as f64) * 255.0) as f32;
        }
        t
    })
}

/// Linear-interpolated sRGB-to-byte via the LUT. Max error vs exact `srgb` is
/// well under 0.1 LSB for `SRGB_LUT_N = 4096`.
#[inline(always)]
fn srgb255(lut: &[f32; SRGB_LUT_N + 1], v: f64) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let f = v * SRGB_LUT_N as f64;
    let i = f as usize;
    let frac = (f - i as f64) as f32;
    let a = lut[i];
    let b = lut[(i + 1).min(SRGB_LUT_N)];
    (a + (b - a) * frac + 0.5) as u8
}

/// MHC debayer of a CFA into a planar `[R | G | B]` buffer (`3*w*h`). Used by the
/// validation test that compares our implementation against the reference crate;
/// the runtime paths use the fused [`render_mhc_fused_into`] instead.
pub fn mhc_debayer(cfa: &[f32], w: usize, h: usize) -> Vec<f32> {
    let plane = w * h;
    let mut planar = vec![0f32; 3 * plane];
    for y in 0..h {
        for x in 0..w {
            let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
            let i = y * w + x;
            planar[i] = r;
            planar[plane + i] = g;
            planar[2 * plane + i] = b;
        }
    }
    planar
}

/// Fused full-res render: MHC debayer + CCM + sRGB gamma straight from an
/// (LSC+WB-applied) CFA into RGB8, row-parallel, with no planar intermediate.
/// Interior pixels take the unclamped fast path; the 2-pixel border uses the
/// clamped [`mhc_rgb_at`]. sRGB uses the LUT (see [`srgb255`]).
fn render_mhc_fused_into(out: &mut [u8], cfa: &[f32], w: usize, h: usize, ccm: [f64; 9], acm: &AcmFrame) {
    let inv = 1.0 / MAXLIN as f64;
    let lut = srgb255_lut();
    let color = |r: f32, g: f32, b: f32, orow: &mut [u8], x: usize| {
        let mut rl = r as f64 * inv;
        let mut gl = g as f64 * inv;
        let mut bl = b as f64 * inv;
        desaturate_highlight(&mut rl, &mut gl, &mut bl);
        let (cr, cg, cb) = acm_color(rl, gl, bl, &ccm, acm);
        orow[x * 3] = srgb255(lut, cr);
        orow[x * 3 + 1] = srgb255(lut, cg);
        orow[x * 3 + 2] = srgb255(lut, cb);
    };
    for_each_row_mut(out, w * 3, |y, orow| {
        if y >= 2 && y + 2 < h {
            let yp = y & 1;
            let row = y * w;
            for x in 0..2 {
                let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
                color(r, g, b, orow, x);
            }
            for x in 2..w - 2 {
                let (r, g, b) = mhc_rgb_interior(cfa, w, row + x, yp, x & 1);
                color(r, g, b, orow, x);
            }
            for x in w - 2..w {
                let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
                color(r, g, b, orow, x);
            }
        } else {
            for x in 0..w {
                let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
                color(r, g, b, orow, x);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Lateral chromatic aberration correction (full-res MHC path only).
//
// The red and blue channels are shifted sub-pixel relative to green by the lens;
// the .aiqb ships per-location shift grids (see tuning_data::LCA_*). We correct
// by resampling the demosaiced R and B planes at the green-aligned position
// `(x + shift)`, which requires the demosaiced neighbours — hence the full-res
// MHC path renders into a planar RGB buffer first, then resamples + colours it.
// The shifts are well under one pixel, so the effect is a small reduction of
// coloured fringing toward the corners. Half-res keeps the fused path (no LCA).
// ---------------------------------------------------------------------------

/// The four embedded lateral-chromatic-aberration shift grids (native px),
/// order blue_x, blue_y, red_x, red_y.
pub struct Lca {
    bx: Vec<f32>,
    by: Vec<f32>,
    rx: Vec<f32>,
    ry: Vec<f32>,
}

impl Lca {
    /// Load the grids from the embedded tuning blob.
    pub fn load() -> Self {
        Lca {
            bx: tuning::lca_grid(0),
            by: tuning::lca_grid(1),
            rx: tuning::lca_grid(2),
            ry: tuning::lca_grid(3),
        }
    }

    /// The four grids as slices in (blue_x, blue_y, red_x, red_y) order, for
    /// callers that upload them elsewhere (the GPU backend).
    pub fn grids(&self) -> [&[f32]; 4] {
        [&self.bx, &self.by, &self.rx, &self.ry]
    }
}

/// Bilinear sample of an `LCA_GH*LCA_GW` shift grid at native pixel `(px, py)`.
#[inline]
fn lca_sample(grid: &[f32], px: f32, py: f32) -> f32 {
    let gx = (px / LCA_CELL_X).clamp(0.0, (LCA_GW - 1) as f32);
    let gy = (py / LCA_CELL_Y).clamp(0.0, (LCA_GH - 1) as f32);
    let x0 = gx.floor() as usize;
    let x1 = (x0 + 1).min(LCA_GW - 1);
    let y0 = gy.floor() as usize;
    let y1 = (y0 + 1).min(LCA_GH - 1);
    let fx = gx - x0 as f32;
    let fy = gy - y0 as f32;
    let top = grid[y0 * LCA_GW + x0] * (1.0 - fx) + grid[y0 * LCA_GW + x1] * fx;
    let bot = grid[y1 * LCA_GW + x0] * (1.0 - fx) + grid[y1 * LCA_GW + x1] * fx;
    top * (1.0 - fy) + bot * fy
}

/// Bilinear sample of a `w*h` channel plane at float `(sx, sy)`, edge-clamped.
#[inline]
fn plane_sample(p: &[f32], w: usize, h: usize, sx: f32, sy: f32) -> f32 {
    let x = sx.clamp(0.0, (w - 1) as f32);
    let y = sy.clamp(0.0, (h - 1) as f32);
    let x0 = x.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y0 = y.floor() as usize;
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let top = p[y0 * w + x0] * (1.0 - fx) + p[y0 * w + x1] * fx;
    let bot = p[y1 * w + x0] * (1.0 - fx) + p[y1 * w + x1] * fx;
    top * (1.0 - fy) + bot * fy
}

/// MHC debayer of `cfa` (LSC+WB-applied, GRBG) into a planar `[R | G | B]`
/// buffer (`3*w*h`), row-parallel with the `video` feature. Interior pixels use
/// the unclamped fast path; the 2-pixel border uses the clamped accessor.
fn mhc_to_planar_into(planar: &mut [f32], cfa: &[f32], w: usize, h: usize) {
    let plane = w * h;
    let (rp, rest) = planar.split_at_mut(plane);
    let (gp, bp) = rest.split_at_mut(plane);
    let fill = |y: usize, rr: &mut [f32], gr: &mut [f32], br: &mut [f32]| {
        if y >= 2 && y + 2 < h {
            let yp = y & 1;
            let row = y * w;
            for x in 0..2 {
                let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
                (rr[x], gr[x], br[x]) = (r, g, b);
            }
            for x in 2..w - 2 {
                let (r, g, b) = mhc_rgb_interior(cfa, w, row + x, yp, x & 1);
                (rr[x], gr[x], br[x]) = (r, g, b);
            }
            for x in w - 2..w {
                let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
                (rr[x], gr[x], br[x]) = (r, g, b);
            }
        } else {
            for x in 0..w {
                let (r, g, b) = mhc_rgb_at(cfa, w, h, y, x);
                (rr[x], gr[x], br[x]) = (r, g, b);
            }
        }
    };
    #[cfg(feature = "video")]
    rp.par_chunks_mut(w)
        .zip(gp.par_chunks_mut(w))
        .zip(bp.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((rr, gr), br))| fill(y, rr, gr, br));
    #[cfg(not(feature = "video"))]
    rp.chunks_mut(w)
        .zip(gp.chunks_mut(w))
        .zip(bp.chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((rr, gr), br))| fill(y, rr, gr, br));
}

/// Stage 2 of the full-res LCA path: per output pixel, take green from the
/// planar buffer, resample red/blue at their green-aligned (LCA-shifted)
/// positions, then highlight-desaturate, CCM and sRGB gamma into RGB8.
fn render_lca_color_into(out: &mut [u8], planar: &[f32], w: usize, h: usize, ccm: [f64; 9], acm: &AcmFrame, lca: &Lca) {
    let plane = w * h;
    let inv = 1.0 / MAXLIN as f64;
    let lut = srgb255_lut();
    let (rp, rest) = planar.split_at(plane);
    let (gp, bp) = rest.split_at(plane);
    for_each_row_mut(out, w * 3, |y, orow| {
        let fy = y as f32;
        for x in 0..w {
            let fx = x as f32;
            let g = gp[y * w + x];
            let r = plane_sample(rp, w, h, fx + lca_sample(&lca.rx, fx, fy), fy + lca_sample(&lca.ry, fx, fy));
            let b = plane_sample(bp, w, h, fx + lca_sample(&lca.bx, fx, fy), fy + lca_sample(&lca.by, fx, fy));
            let mut rl = r as f64 * inv;
            let mut gl = g as f64 * inv;
            let mut bl = b as f64 * inv;
            desaturate_highlight(&mut rl, &mut gl, &mut bl);
            let (cr, cg, cb) = acm_color(rl, gl, bl, &ccm, acm);
            orow[x * 3] = srgb255(lut, cr);
            orow[x * 3 + 1] = srgb255(lut, cg);
            orow[x * 3 + 2] = srgb255(lut, cb);
        }
    });
}

/// Full-resolution render: LSC + per-channel WB applied to the Bayer frame,
/// then MHC debayer, then lateral-CA correction, CCM and sRGB gamma. Returns
/// (W, H, RGB8).
pub fn render_mhc(raw: &RawFrame, gains: [f64; 3], ccm: [f64; 9], cct: f64, ls: usize) -> (usize, usize, Vec<u8>) {
    let (w, h) = (raw.w, raw.h);
    let (ww, hh) = (w / 2, h / 2);
    let grids = build_grids(ls, hh, ww);
    let acm = interp_acm(cct);

    // Apply LSC and white balance on the Bayer frame, per channel (GRBG).
    let mut cfa = vec![0f32; w * h];
    for_each_row_mut(&mut cfa, w, |y, crow| {
        let iy = y / 2;
        let src = &raw.data[y * w..(y + 1) * w];
        for x in 0..w {
            let ix = x / 2;
            let (lsc, wb) = match (y & 1, x & 1) {
                (0, 0) => (grids.g_gr[iy * ww + ix], gains[1]), // Gr
                (0, 1) => (grids.g_r[iy * ww + ix], gains[0]),  // R
                (1, 0) => (grids.g_b[iy * ww + ix], gains[2]),  // B
                _ => (grids.g_gb[iy * ww + ix], gains[1]),      // Gb
            };
            crow[x] = (src[x] as f64 * lsc * wb) as f32;
        }
    });

    let mut planar = vec![0f32; 3 * w * h];
    mhc_to_planar_into(&mut planar, &cfa, w, h);
    let lca = Lca::load();
    let mut rgb = vec![0u8; w * h * 3];
    render_lca_color_into(&mut rgb, &planar, w, h, ccm, &acm, &lca);
    (w, h, rgb)
}

/// Full pipeline (half-res): returns (width, height, RGB8, estimate).
pub fn process(raw: &RawFrame) -> (usize, usize, Vec<u8>, Estimate) {
    let planes = bayer_planes(raw);
    let est = estimate(&planes);
    let rgb = render(&planes, est.gains, est.ccm, est.cct, est.ls);
    (planes.ww, planes.hh, rgb, est)
}

/// Full pipeline with a selectable debayer mode.
pub fn process_with(raw: &RawFrame, mode: DebayerMode) -> (usize, usize, Vec<u8>, Estimate) {
    match mode {
        DebayerMode::HalfRes => process(raw),
        DebayerMode::Mhc => {
            let planes = bayer_planes(raw);
            let est = estimate(&planes);
            let (w, h, rgb) = render_mhc(raw, est.gains, est.ccm, est.cct, est.ls);
            (w, h, rgb, est)
        }
    }
}

// ---------------------------------------------------------------------------
// Live runtime: a reusable, data-parallel processor.
//
// The offline path above allocates fresh buffers and re-estimates per call,
// which is fine for a one-shot CLI. The live webcam runs the same pipeline tens
// of times a second, where per-frame allocation and the AWB sort dominate. The
// `Processor` below keeps all working buffers across frames, fuses the raw
// unpacking with the Bayer split (one parallel pass, no `RawFrame`), caches the
// resized LSC grids per light source, and re-estimates white balance only every
// `awb_interval` frames (the scene's colour is stable between estimates).
// ---------------------------------------------------------------------------

#[cfg(feature = "video")]
use crate::raw::{BLACK, STRIDE_SAMPLES};

/// Fused raw unpack + black level + GRBG split, straight from the capture bytes
/// into the four half-res planes (`hh*ww` each), row-parallel. Equivalent to
/// `bayer_planes(&RawFrame::from_bytes(bytes))` but with no intermediate
/// full-res `RawFrame` and no second pass. `bytes` must be at least
/// `H*STRIDE_SAMPLES*2` long (checked by the caller).
#[cfg(feature = "video")]
fn fill_planes_blc_from_bytes(bytes: &[u8], p: &mut Planes) {
    let ww = p.ww;
    let Planes { gr, r, b, gb, .. } = p;
    gr.par_chunks_mut(ww)
        .zip(r.par_chunks_mut(ww))
        .zip(b.par_chunks_mut(ww))
        .zip(gb.par_chunks_mut(ww))
        .enumerate()
        .for_each(|(y, (((grr, rr), br), gbr))| {
            let r0 = (2 * y) * STRIDE_SAMPLES;
            let r1 = (2 * y + 1) * STRIDE_SAMPLES;
            let sample = |s: usize| -> f32 {
                let i = s * 2;
                (u16::from_le_bytes([bytes[i], bytes[i + 1]]) as f32 - BLACK).max(0.0)
            };
            for x in 0..ww {
                grr[x] = sample(r0 + 2 * x);
                rr[x] = sample(r0 + 2 * x + 1);
                br[x] = sample(r1 + 2 * x);
                gbr[x] = sample(r1 + 2 * x + 1);
            }
        });
}

/// Fused raw unpack + black level + LSC + per-channel white balance, straight
/// from the capture bytes into a full-res CFA (`W*H`, GRBG), row-parallel. This
/// is the MHC front-end: it produces exactly the `cfa` that `render_mhc` builds
/// internally, ready for debayering.
#[cfg(feature = "video")]
fn fill_cfa_lscwb_from_bytes(bytes: &[u8], cfa: &mut [f32], grids: &Grids, gains: [f64; 3]) {
    let ww = grids.ww;
    let w = W;
    cfa.par_chunks_mut(w).enumerate().for_each(|(y, crow)| {
        let iy = y / 2;
        let r0 = y * STRIDE_SAMPLES;
        for (x, cell) in crow.iter_mut().enumerate() {
            let ix = x / 2;
            let i = (r0 + x) * 2;
            let v = (u16::from_le_bytes([bytes[i], bytes[i + 1]]) as f64 - BLACK as f64).max(0.0);
            let (lsc, wb) = match (y & 1, x & 1) {
                (0, 0) => (grids.g_gr[iy * ww + ix], gains[1]), // Gr
                (0, 1) => (grids.g_r[iy * ww + ix], gains[0]),  // R
                (1, 0) => (grids.g_b[iy * ww + ix], gains[2]),  // B
                _ => (grids.g_gb[iy * ww + ix], gains[1]),      // Gb
            };
            *cell = (v * lsc * wb) as f32;
        }
    });
}

/// Reusable, data-parallel processor for the live webcam path. Holds all
/// working buffers, caches the LSC grids per light source, and re-estimates
/// white balance only every `awb_interval` frames.
#[cfg(feature = "video")]
pub struct Processor {
    mode: DebayerMode,
    interval: u64,
    frame: u64,
    planes: Planes,
    cfa: Vec<f32>,
    out: Vec<u8>,
    grids: Option<Grids>,
    est: Option<Estimate>,
    acm: Option<AcmFrame>,
    chroma: Option<(f32, f32)>,
    planar: Vec<f32>,
    lca: Option<Lca>,
}

#[cfg(feature = "video")]
impl Processor {
    /// Create a processor for `mode`, re-estimating AWB/CCM every
    /// `awb_interval` frames (clamped to >= 1; the first frame always estimates).
    /// `lca` enables lateral-chromatic-aberration correction; it only applies to
    /// the full-res MHC mode (half-res ignores it).
    pub fn new(mode: DebayerMode, awb_interval: u64, lca: bool) -> Self {
        let (ww, hh) = (W / 2, H / 2);
        let planes = Planes {
            hh,
            ww,
            gr: vec![0f32; hh * ww],
            r: vec![0f32; hh * ww],
            b: vec![0f32; hh * ww],
            gb: vec![0f32; hh * ww],
        };
        let (cfa, out) = match mode {
            DebayerMode::HalfRes => (Vec::new(), vec![0u8; hh * ww * 3]),
            DebayerMode::Mhc => (vec![0f32; W * H], vec![0u8; W * H * 3]),
        };
        let (planar, lca) = match (mode, lca) {
            (DebayerMode::Mhc, true) => (vec![0f32; 3 * W * H], Some(Lca::load())),
            _ => (Vec::new(), None),
        };
        Processor {
            mode,
            interval: awb_interval.max(1),
            frame: 0,
            planes,
            cfa,
            out,
            grids: None,
            est: None,
            acm: None,
            chroma: None,
            planar,
            lca,
        }
    }

    /// The most recent scene estimate, if any frame has been processed.
    pub fn estimate(&self) -> Option<&Estimate> {
        self.est.as_ref()
    }

    /// Re-estimate from the current (BLC) planes and rebuild the LSC grids if
    /// the chosen light source changed. The scene chroma is exponentially
    /// smoothed across estimates (see [`AWB_SMOOTH_ALPHA`]) and the LSC source
    /// switch is hysteretic (see [`LS_HYST`]), so white balance follows lighting
    /// changes gradually instead of stepping every interval. The first frame has
    /// no history, so it reproduces the stateless [`estimate`] exactly.
    fn reestimate(&mut self) {
        let (rg, bg) = robust_neutral(&self.planes);
        let (srg, sbg) = smooth_chroma(self.chroma, rg, bg);
        self.chroma = Some((srg, sbg));
        let prev_ls = self.est.as_ref().map(|e| e.ls);
        let est = estimate_from_chroma(srg, sbg, prev_ls);
        if self.grids.as_ref().map(|g| g.ls) != Some(est.ls) {
            self.grids = Some(build_grids(est.ls, self.planes.hh, self.planes.ww));
        }
        // The per-sector matrices depend only on CCT; rebuilding them every
        // estimate is cheap (24 matrices x 9 lerps) and keeps them in step.
        self.acm = Some(interp_acm(est.cct));
        self.est = Some(est);
    }

    /// Process one raw SGRBG10 frame; returns `(out_w, out_h, rgb)`, the RGB8
    /// borrowing the processor's reused output buffer (valid until the next call).
    pub fn process(&mut self, bytes: &[u8]) -> std::io::Result<(usize, usize, &[u8])> {
        let needed = H * STRIDE_SAMPLES * 2;
        if bytes.len() < needed {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("raw too small: {} bytes, need >= {needed}", bytes.len()),
            ));
        }
        let estimating = self.est.is_none() || self.frame % self.interval == 0;

        match self.mode {
            DebayerMode::HalfRes => {
                fill_planes_blc_from_bytes(bytes, &mut self.planes);
                if estimating {
                    self.reestimate();
                }
                let gains = self.est.as_ref().expect("estimate present after first frame").gains;
                let ccm = self.est.as_ref().expect("estimate present after first frame").ccm;
                let Self { out, planes, grids, acm, .. } = &mut *self;
                render_half_into(out, planes, gains, ccm, grids.as_ref().expect("grids built after first estimate"), acm.as_ref().expect("acm built after first estimate"));
            }
            DebayerMode::Mhc => {
                if estimating {
                    fill_planes_blc_from_bytes(bytes, &mut self.planes);
                    self.reestimate();
                }
                let gains = self.est.as_ref().expect("estimate present after first frame").gains;
                let ccm = self.est.as_ref().expect("estimate present after first frame").ccm;
                {
                    let Self { cfa, grids, .. } = &mut *self;
                    fill_cfa_lscwb_from_bytes(bytes, cfa, grids.as_ref().expect("grids built after first estimate"), gains);
                }
                if self.lca.is_some() {
                    {
                        let Self { planar, cfa, .. } = &mut *self;
                        mhc_to_planar_into(planar, cfa, W, H);
                    }
                    let Self { out, planar, lca, acm, .. } = &mut *self;
                    render_lca_color_into(out, planar, W, H, ccm, acm.as_ref().expect("acm built after first estimate"), lca.as_ref().unwrap());
                } else {
                    let Self { out, cfa, acm, .. } = &mut *self;
                    render_mhc_fused_into(out, cfa, W, H, ccm, acm.as_ref().expect("acm built after first estimate"));
                }
            }
        }

        self.frame = self.frame.wrapping_add(1);
        let (w, h) = match self.mode {
            DebayerMode::HalfRes => (self.planes.ww, self.planes.hh),
            DebayerMode::Mhc => (W, H),
        };
        Ok((w, h, &self.out))
    }
}
