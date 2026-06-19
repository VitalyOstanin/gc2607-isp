//! GPU backend (wgpu / Vulkan): the per-pixel ISP stages run as compute shaders
//! on the integrated GPU, while AWB/CCM estimation stays on the CPU (it needs a
//! sort over downsampled pixels and runs only every `awb_interval` frames).
//!
//! Division of labour, per frame:
//!   - CPU: nothing on most frames; every `awb_interval` frames it builds the
//!     half-res Bayer planes and runs `pipeline::estimate` (gains, CCM, LSC src).
//!     When the chosen light source changes, the four resized LSC grids are
//!     rebuilt on the CPU and uploaded once.
//!   - GPU pass 1 (`build_cfa`): unpack SGRBG10 -> black level -> LSC -> per
//!     channel white balance, into a full-res CFA buffer.
//!   - GPU pass 2 (`render_pack`): Malvar-He-Cutler debayer -> hue-sectored CCM
//!     -> sRGB gamma -> full-range BT.601 YCbCr -> packed YUYV, with the centre
//!     crop applied, straight into the output buffer (no CPU `rgb_to_yuyv_crop`).
//!     The 24 per-sector matrices for the scene CCT are uploaded per re-estimate;
//!     see `docs/acm-color-model.md`.
//!
//! Only the full-res MHC path is implemented on the GPU (the heavy case the GPU
//! is meant to offload); half-res stays on the CPU. The arithmetic mirrors
//! `pipeline::mhc_rgb_interior` and `output::rgb_to_yuyv_crop`; results match the
//! CPU MHC path within rounding (validated by `tests/gpu.rs`), not bit-for-bit
//! (the GPU uses `pow` where the CPU MHC path uses a LUT).

use std::io;

use bytemuck::Zeroable;

use crate::pipeline::{self, interp_acm, Estimate, Lca, ACM_SAT_KNEE};
use crate::raw::{RawFrame, BLACK, H, MAXLIN, STRIDE_SAMPLES, W};
use crate::tuning_data::{
    ACM_HUE0, ACM_HUE_STEP, ACM_NSEC, LCA_CELL_X, LCA_CELL_Y, LCA_GH, LCA_GW,
};

/// Output (centre-cropped) size for the GPU MHC path: 1920x1080 from 1928x1088.
pub const OUT_W: usize = 1920;
pub const OUT_H: usize = 1080;

const WW: usize = W / 2; // 964
const HH: usize = H / 2; // 544

/// Uniform block shared by both compute passes. Laid out as `vec4`-aligned
/// groups to match the WGSL `Params` struct (std140 / 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 4],     // w, h, ww, hh
    out_dims: [u32; 4], // dst_w, dst_h, off_x, off_y
    misc: [u32; 4],     // stride, groups_per_row, _, _
    consts: [f32; 4],   // black, inv_maxlin, _, _
    gains: [f32; 4],    // R, G, B, _
    ccm0: [f32; 4],     // ccm[0..3]
    ccm1: [f32; 4],     // ccm[3..6]
    ccm2: [f32; 4],     // ccm[6..9]
    lca: [f32; 4],      // lca grid_w, grid_h, cell_x, cell_y
    acm_cfg: [f32; 4],  // hue0, hue_step, sat_knee, nsec
    dn_i: [u32; 4],     // chroma_radius, chroma_on, temporal_on, temporal_reset
    dn_f: [f32; 4],     // chroma_strength, temporal_alpha, temporal_motion, _
}

const SHADER: &str = r#"
struct Params {
    dims: vec4<u32>,
    out_dims: vec4<u32>,
    misc: vec4<u32>,
    consts: vec4<f32>,
    gains: vec4<f32>,
    ccm0: vec4<f32>,
    ccm1: vec4<f32>,
    ccm2: vec4<f32>,
    lca: vec4<f32>,
    acm_cfg: vec4<f32>,
    dn_i: vec4<u32>,
    dn_f: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read> g_gr: array<f32>;
@group(0) @binding(3) var<storage, read> g_r: array<f32>;
@group(0) @binding(4) var<storage, read> g_b: array<f32>;
@group(0) @binding(5) var<storage, read> g_gb: array<f32>;
@group(0) @binding(6) var<storage, read_write> cfa: array<f32>;
@group(0) @binding(7) var<storage, read_write> yuyv: array<u32>;
@group(0) @binding(8) var<storage, read_write> rgb: array<f32>;
@group(0) @binding(9) var<storage, read> lca: array<f32>;
// Per-sector colour matrices for this frame's CCT: 9 floats per sector
// (row-major 3x3), nsec sectors, concatenated. Uploaded on each re-estimate.
@group(0) @binding(10) var<storage, read> acm: array<f32>;
// Denoise scratch: horizontally-blurred chroma (one word per YUYV word, Cb in
// the low 16 bits and Cr in the high 16) and the temporal luma history (one Y
// value per output pixel). Both are written by the denoise passes only.
@group(0) @binding(11) var<storage, read_write> ctmp: array<u32>;
@group(0) @binding(12) var<storage, read_write> prev_y: array<u32>;

// Pass 1: unpack + black level + LSC + white balance into the full-res CFA.
@compute @workgroup_size(64)
fn build_cfa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.dims.x;
    let h = P.dims.y;
    let ww = P.dims.z;
    let idx = gid.x;
    if (idx >= w * h) { return; }
    let y = idx / w;
    let x = idx % w;
    let stride = P.misc.x;
    let s = y * stride + x;
    let word = raw[s >> 1u];
    var sample: u32;
    if ((s & 1u) == 0u) { sample = word & 0xFFFFu; } else { sample = word >> 16u; }
    let v = max(f32(sample) - P.consts.x, 0.0);
    let iy = y / 2u;
    let ix = x / 2u;
    let gi = iy * ww + ix;
    let yp = y & 1u;
    let xp = x & 1u;
    var lsc: f32;
    var gain: f32;
    if (yp == 0u && xp == 0u) {        // Gr
        lsc = g_gr[gi]; gain = P.gains.y;
    } else if (yp == 0u && xp == 1u) { // R
        lsc = g_r[gi]; gain = P.gains.x;
    } else if (yp == 1u && xp == 0u) { // B
        lsc = g_b[gi]; gain = P.gains.z;
    } else {                           // Gb
        lsc = g_gb[gi]; gain = P.gains.y;
    }
    cfa[idx] = v * lsc * gain;
}

// Highlight-desaturation knee; must match pipeline::HIGHLIGHT_KNEE.
const HIGHLIGHT_KNEE: f32 = 0.95;

// Mirrors pipeline::srgb (constants SRGB_LIN_THRESH/SLOPE/SCALE/GAMMA/OFFSET);
// keep the literals below in sync with that single Rust source.
fn srgb(x: f32) -> f32 {
    let v = clamp(x, 0.0, 1.0);
    if (v <= 0.0031308) { return 12.92 * v; }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

// Malvar-He-Cutler interpolation at an interior pixel (GRBG). Mirrors
// pipeline::mhc_rgb_interior; the centre crop guarantees >=4 px from each edge,
// so no bounds clamping is needed. Returns linear (R, G, B) in CFA scale.
fn mhc(y: u32, x: u32) -> vec3<f32> {
    let w = P.dims.x;
    let i = y * w + x;
    let c = cfa[i];
    let up = cfa[i - w];
    let dn = cfa[i + w];
    let lf = cfa[i - 1u];
    let rt = cfa[i + 1u];
    let uu = cfa[i - 2u * w];
    let dd = cfa[i + 2u * w];
    let ll = cfa[i - 2u];
    let rr = cfa[i + 2u];
    let diag = cfa[i - w - 1u] + cfa[i - w + 1u] + cfa[i + w - 1u] + cfa[i + w + 1u];

    let g_at_rb = (4.0 * c + 2.0 * (up + dn + lf + rt) - (uu + dd + ll + rr)) / 8.0;
    let at_g_hrow = (5.0 * c + 4.0 * (lf + rt) - (ll + rr) - diag + 0.5 * (uu + dd)) / 8.0;
    let at_g_vcol = (5.0 * c + 4.0 * (up + dn) - (uu + dd) - diag + 0.5 * (ll + rr)) / 8.0;
    let at_diag = (6.0 * c + 2.0 * diag - 1.5 * (uu + dd + ll + rr)) / 8.0;

    let yp = y & 1u;
    let xp = x & 1u;
    if (yp == 0u && xp == 0u) { return vec3<f32>(at_g_hrow, c, at_g_vcol); }      // Gr
    else if (yp == 0u && xp == 1u) { return vec3<f32>(c, g_at_rb, at_diag); }     // R
    else if (yp == 1u && xp == 0u) { return vec3<f32>(at_diag, g_at_rb, c); }     // B
    else { return vec3<f32>(at_g_vcol, c, at_g_hrow); }                            // Gb
}

// Yellow-desaturation operator (mirrors pipeline::desaturate_yellow). Scales the
// chroma of yellow / orange-yellow hues toward their luminance gray by
// YELLOW_DESAT_K, preserving luma and hue; other hues and neutrals pass through.
const YELLOW_DESAT_K: f32 = 0.70;
const YELLOW_HUE_LO: f32 = 35.0;
const YELLOW_HUE_HI: f32 = 80.0;
const YELLOW_HUE_SOFT: f32 = 12.0;

// Axis 2 skin-red desaturation band (mirrors pipeline.rs SKIN_* constants).
const SKIN_DESAT_K: f32 = 0.80;
const SKIN_HUE_LO: f32 = 0.0;
const SKIN_HUE_HI: f32 = 28.0;
const SKIN_HUE_SOFT: f32 = 10.0;

// Trapezoidal-band chroma desaturation toward luma (mirrors
// pipeline::desaturate_band). Pixels outside [lo,hi] and neutrals pass through.
fn desaturate_band(c: vec3<f32>, lo: f32, hi: f32, soft: f32, k: f32) -> vec3<f32> {
    let lr = max(c.x, 0.0);
    let lg = max(c.y, 0.0);
    let lb = max(c.z, 0.0);
    let mx = max(lr, max(lg, lb));
    let mn = min(lr, min(lg, lb));
    let d = mx - mn;
    if (mx <= 0.0 || d <= 0.0) { return c; }
    var h: f32;
    if (lr >= lg && lr >= lb) { h = (lg - lb) / d; }
    else if (lg >= lb) { h = 2.0 + (lb - lr) / d; }
    else { h = 4.0 + (lr - lg) / d; }
    h = h * 60.0;
    if (h < 0.0) { h = h + 360.0; }
    let rise = clamp((h - lo) / soft, 0.0, 1.0);
    let fall = clamp((hi - h) / soft, 0.0, 1.0);
    let win = min(rise, fall);
    if (win <= 0.0) { return c; }
    let keff = 1.0 - win * (1.0 - k);
    let y = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z;
    return vec3<f32>(y) + keff * (c - vec3<f32>(y));
}

fn desaturate_yellow(c: vec3<f32>) -> vec3<f32> {
    return desaturate_band(c, YELLOW_HUE_LO, YELLOW_HUE_HI, YELLOW_HUE_SOFT, YELLOW_DESAT_K);
}

// Hue-sectored colour correction (mirrors pipeline::acm_color). `s` is the
// white-balanced linear pixel (0..1). The hue/saturation that select the sector
// come from the globally-corrected colour `ccm * s`; the result fades between the
// global CCM (low saturation) and the per-sector matrix (high saturation), then
// the yellow-desaturation operator is applied.
fn acm_apply(s: vec3<f32>) -> vec3<f32> {
    let g0 = P.ccm0.x * s.x + P.ccm0.y * s.y + P.ccm0.z * s.z;
    let g1 = P.ccm1.x * s.x + P.ccm1.y * s.y + P.ccm1.z * s.z;
    let g2 = P.ccm2.x * s.x + P.ccm2.y * s.y + P.ccm2.z * s.z;

    let lr = max(g0, 0.0);
    let lg = max(g1, 0.0);
    let lb = max(g2, 0.0);
    let mx = max(lr, max(lg, lb));
    let mn = min(lr, min(lg, lb));
    let d = mx - mn;
    if (mx <= 0.0 || d <= 0.0) { return vec3<f32>(g0, g1, g2); }

    var h: f32;
    if (lr >= lg && lr >= lb) { h = (lg - lb) / d; }
    else if (lg >= lb) { h = 2.0 + (lb - lr) / d; }
    else { h = 4.0 + (lr - lg) / d; }
    h = h * 60.0;
    if (h < 0.0) { h = h + 360.0; }
    let sat = d / mx;
    let w = clamp(sat / P.acm_cfg.z, 0.0, 1.0);

    let nsec = i32(P.acm_cfg.w);
    let pos = (h - P.acm_cfg.x) / P.acm_cfg.y;
    let fi = floor(pos);
    let frac = pos - fi;
    let ia = ((i32(fi) % nsec) + nsec) % nsec;
    let ib = (ia + 1) % nsec;
    let ba = u32(ia) * 9u;
    let bb = u32(ib) * 9u;

    let cc = array<f32, 9>(
        P.ccm0.x, P.ccm0.y, P.ccm0.z,
        P.ccm1.x, P.ccm1.y, P.ccm1.z,
        P.ccm2.x, P.ccm2.y, P.ccm2.z,
    );
    var m: array<f32, 9>;
    for (var k = 0u; k < 9u; k = k + 1u) {
        let ms = acm[ba + k] * (1.0 - frac) + acm[bb + k] * frac;
        m[k] = cc[k] * (1.0 - w) + ms * w;
    }
    let corrected = vec3<f32>(
        m[0] * s.x + m[1] * s.y + m[2] * s.z,
        m[3] * s.x + m[4] * s.y + m[5] * s.z,
        m[6] * s.x + m[7] * s.y + m[8] * s.z,
    );
    let skin = desaturate_band(corrected, SKIN_HUE_LO, SKIN_HUE_HI, SKIN_HUE_SOFT, SKIN_DESAT_K);
    return desaturate_yellow(skin);
}

// Linear CFA-scale RGB -> hue-sectored CCM -> sRGB gamma -> 0..255 (rounded).
fn to_rgb8(lin: vec3<f32>) -> vec3<f32> {
    var s = lin * P.consts.y; // * inv_maxlin
    // Highlight desaturation: blend toward the max channel near full scale so
    // blown highlights converge to neutral white (mirrors desaturate_highlight).
    let m = max(s.x, max(s.y, s.z));
    if (m > HIGHLIGHT_KNEE) {
        let t = clamp((m - HIGHLIGHT_KNEE) / (1.0 - HIGHLIGHT_KNEE), 0.0, 1.0);
        s = s + (vec3<f32>(m) - s) * t;
    }
    let c = acm_apply(s);
    return vec3<f32>(
        floor(srgb(c.x) * 255.0 + 0.5),
        floor(srgb(c.y) * 255.0 + 0.5),
        floor(srgb(c.z) * 255.0 + 0.5),
    );
}

// Full-range BT.601 (JFIF) RGB(0..255) -> YCbCr(0..255), rounded and clamped.
// Mirrors output::rgb_to_ycbcr (128.0 == output::CHROMA_NEUTRAL); keep in sync.
fn ycbcr(c: vec3<f32>) -> vec3<f32> {
    let yy = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let cb = 128.0 - 0.168736 * c.x - 0.331264 * c.y + 0.5 * c.z;
    let cr = 128.0 + 0.5 * c.x - 0.418688 * c.y - 0.081312 * c.z;
    return vec3<f32>(
        clamp(floor(yy + 0.5), 0.0, 255.0),
        clamp(floor(cb + 0.5), 0.0, 255.0),
        clamp(floor(cr + 0.5), 0.0, 255.0),
    );
}

// Pack two adjacent YCbCr pixels into one YUYV word; the shared chroma is the
// average of the pair (4:2:2 subsampling).
fn pack_yuyv(yc0: vec3<f32>, yc1: vec3<f32>) -> u32 {
    let y0 = u32(yc0.x);
    let y1 = u32(yc1.x);
    let cb = (u32(yc0.y) + u32(yc1.y)) / 2u;
    let cr = (u32(yc0.z) + u32(yc1.z)) / 2u;
    return y0 | (cb << 8u) | (y1 << 16u) | (cr << 24u);
}

// Pass 2: one thread per output pixel pair -> MHC + colour + YUYV pack.
@compute @workgroup_size(64)
fn render_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_h = P.out_dims.y;
    let off_x = P.out_dims.z;
    let off_y = P.out_dims.w;
    let groups_per_row = P.misc.y; // dst_w / 2
    let gidx = gid.x;
    if (gidx >= groups_per_row * dst_h) { return; }
    let gy = gidx / groups_per_row;
    let gx = gidx % groups_per_row;
    let sy = off_y + gy;
    let sx0 = off_x + gx * 2u;
    let sx1 = sx0 + 1u;

    let yc0 = ycbcr(to_rgb8(mhc(sy, sx0)));
    let yc1 = ycbcr(to_rgb8(mhc(sy, sx1)));
    yuyv[gidx] = pack_yuyv(yc0, yc1);
}

// --- Lateral chromatic aberration path (three passes) ---

// Pass 2a: MHC debayer into a planar linear-RGB buffer ([R | G | B], w*h each).
// Only the interior is filled; the centre crop guarantees the sampled region is
// well inside, so edge pixels are left zero.
@compute @workgroup_size(64)
fn mhc_planar(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.dims.x;
    let h = P.dims.y;
    let idx = gid.x;
    if (idx >= w * h) { return; }
    let plane = w * h;
    let y = idx / w;
    let x = idx % w;
    if (y < 2u || y + 2u >= h || x < 2u || x + 2u >= w) {
        rgb[idx] = 0.0; rgb[plane + idx] = 0.0; rgb[2u * plane + idx] = 0.0;
        return;
    }
    let c = mhc(y, x);
    rgb[idx] = c.x;
    rgb[plane + idx] = c.y;
    rgb[2u * plane + idx] = c.z;
}

// Bilinear sample of LCA shift grid channel `ch` (0=bx,1=by,2=rx,3=ry) at native
// pixel (px, py). The four grids are concatenated in the `lca` buffer.
fn lca_sample(ch: u32, px: f32, py: f32) -> f32 {
    let gw = u32(P.lca.x);
    let gh = u32(P.lca.y);
    let gx = clamp(px / P.lca.z, 0.0, f32(gw - 1u));
    let gy = clamp(py / P.lca.w, 0.0, f32(gh - 1u));
    let x0 = u32(floor(gx));
    let x1 = min(x0 + 1u, gw - 1u);
    let y0 = u32(floor(gy));
    let y1 = min(y0 + 1u, gh - 1u);
    let fx = gx - f32(x0);
    let fy = gy - f32(y0);
    let base = ch * gw * gh;
    let top = lca[base + y0 * gw + x0] * (1.0 - fx) + lca[base + y0 * gw + x1] * fx;
    let bot = lca[base + y1 * gw + x0] * (1.0 - fx) + lca[base + y1 * gw + x1] * fx;
    return top * (1.0 - fy) + bot * fy;
}

// Bilinear sample of one RGB plane (offset `chbase` into `rgb`) at float (sx, sy).
fn plane_sample(chbase: u32, w: u32, h: u32, sx: f32, sy: f32) -> f32 {
    let x = clamp(sx, 0.0, f32(w - 1u));
    let y = clamp(sy, 0.0, f32(h - 1u));
    let x0 = u32(floor(x));
    let x1 = min(x0 + 1u, w - 1u);
    let y0 = u32(floor(y));
    let y1 = min(y0 + 1u, h - 1u);
    let fx = x - f32(x0);
    let fy = y - f32(y0);
    let top = rgb[chbase + y0 * w + x0] * (1.0 - fx) + rgb[chbase + y0 * w + x1] * fx;
    let bot = rgb[chbase + y1 * w + x0] * (1.0 - fx) + rgb[chbase + y1 * w + x1] * fx;
    return top * (1.0 - fy) + bot * fy;
}

// Green from the planar buffer, red/blue resampled at their LCA-shifted (green-
// aligned) positions, then coloured. Returns RGB8 (0..255).
fn lca_color(sy: u32, sx: u32) -> vec3<f32> {
    let w = P.dims.x;
    let h = P.dims.y;
    let plane = w * h;
    let fx = f32(sx);
    let fy = f32(sy);
    let g = rgb[plane + sy * w + sx];
    let r = plane_sample(0u, w, h, fx + lca_sample(2u, fx, fy), fy + lca_sample(3u, fx, fy));
    let b = plane_sample(2u * plane, w, h, fx + lca_sample(0u, fx, fy), fy + lca_sample(1u, fx, fy));
    return to_rgb8(vec3<f32>(r, g, b));
}

// Pass 2b: per output pixel pair, LCA-correct colour + YUYV pack (centre crop).
@compute @workgroup_size(64)
fn render_pack_lca(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_h = P.out_dims.y;
    let off_x = P.out_dims.z;
    let off_y = P.out_dims.w;
    let groups_per_row = P.misc.y;
    let gidx = gid.x;
    if (gidx >= groups_per_row * dst_h) { return; }
    let gy = gidx / groups_per_row;
    let gx = gidx % groups_per_row;
    let sy = off_y + gy;
    let sx0 = off_x + gx * 2u;
    let sx1 = sx0 + 1u;

    let yc0 = ycbcr(lca_color(sy, sx0));
    let yc1 = ycbcr(lca_color(sy, sx1));
    yuyv[gidx] = pack_yuyv(yc0, yc1);
}

// --- Gain-adaptive denoise (mirrors output::denoise_chroma_yuyv and
// output::temporal_denoise_luma_yuyv). One YUYV word holds two pixels and one
// shared chroma sample, so the chroma grid is exactly the word grid: cw words
// per row (cw == groups_per_row == P.misc.y), dst_h rows. The box blur is
// separable (horizontal then vertical); each border window shrinks to the
// available samples and the mean uses integer truncation, matching the CPU
// running-sum result bit-for-bit. The blend and the temporal IIR use f32 (the
// CPU uses f64), so the final bytes match within rounding, not bit-for-bit.

// Chroma pass 1: horizontal integer box blur of Cb/Cr into `ctmp`.
@compute @workgroup_size(64)
fn chroma_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = P.misc.y;
    let h = P.out_dims.y;
    let idx = gid.x;
    if (idx >= cw * h) { return; }
    let r = P.dn_i.x;
    let y = idx / cw;
    let c = idx % cw;
    let lo = max(c, r) - r;
    let hi = min(c + r, cw - 1u);
    var sb: u32 = 0u;
    var sr: u32 = 0u;
    for (var cc = lo; cc <= hi; cc = cc + 1u) {
        let w = yuyv[y * cw + cc];
        sb = sb + ((w >> 8u) & 0xFFu);
        sr = sr + ((w >> 24u) & 0xFFu);
    }
    let n = hi - lo + 1u;
    ctmp[idx] = (sb / n) | ((sr / n) << 16u);
}

// Chroma pass 2: vertical integer box blur of `ctmp`, then blend the blurred
// chroma back over the original at `strength`, writing into `yuyv` (luma kept).
@compute @workgroup_size(64)
fn chroma_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = P.misc.y;
    let h = P.out_dims.y;
    let idx = gid.x;
    if (idx >= cw * h) { return; }
    let r = P.dn_i.x;
    let y = idx / cw;
    let lo = max(y, r) - r;
    let hi = min(y + r, h - 1u);
    var sb: u32 = 0u;
    var sr: u32 = 0u;
    for (var yy = lo; yy <= hi; yy = yy + 1u) {
        let t = ctmp[yy * cw + (idx % cw)];
        sb = sb + (t & 0xFFFFu);
        sr = sr + (t >> 16u);
    }
    let n = hi - lo + 1u;
    let bb = f32(sb / n);
    let br = f32(sr / n);

    let s = clamp(P.dn_f.x, 0.0, 1.0);
    let inv = 1.0 - s;
    let w = yuyv[idx];
    let ocb = f32((w >> 8u) & 0xFFu);
    let ocr = f32((w >> 24u) & 0xFFu);
    let ncb = u32(clamp(floor(ocb * inv + bb * s + 0.5), 0.0, 255.0));
    let ncr = u32(clamp(floor(ocr * inv + br * s + 0.5), 0.0, 255.0));
    yuyv[idx] = (w & 0x00FF00FFu) | (ncb << 8u) | (ncr << 24u);
}

// Temporal luma IIR with a per-pixel motion gate; updates the history in place.
// One thread per YUYV word handles both packed luma samples.
fn temporal_one(cur: u32, prev: u32) -> u32 {
    if (P.dn_i.w == 1u) { return cur; } // reset: seed history, no blend
    let a = clamp(P.dn_f.y, 0.0, 1.0);
    let mthr = max(P.dn_f.z, 1.0);
    let fc = f32(cur);
    let fp = f32(prev);
    let diff = abs(fc - fp);
    let gate = clamp(1.0 - diff / mthr, 0.0, 1.0);
    let eff = a * gate;
    return u32(clamp(floor(fc * (1.0 - eff) + fp * eff + 0.5), 0.0, 255.0));
}

@compute @workgroup_size(64)
fn temporal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cw = P.misc.y;
    let h = P.out_dims.y;
    let idx = gid.x;
    if (idx >= cw * h) { return; }
    let dst_w = P.out_dims.x;
    let y = idx / cw;
    let c = idx % cw;
    let p0 = y * dst_w + 2u * c;
    let p1 = p0 + 1u;
    let w = yuyv[idx];
    let y0 = temporal_one(w & 0xFFu, prev_y[p0]);
    let y1 = temporal_one((w >> 16u) & 0xFFu, prev_y[p1]);
    prev_y[p0] = y0;
    prev_y[p1] = y1;
    yuyv[idx] = (w & 0xFF00FF00u) | y0 | (y1 << 16u);
}
"#;

/// GPU-resident ISP for the live webcam path. Holds the device, both compute
/// pipelines, and all persistent buffers; per frame it uploads the raw bytes,
/// dispatches the two passes, and reads the packed YUYV back into a reused host
/// buffer. AWB/CCM estimation runs on the CPU every `awb_interval` frames.
/// Gain-adaptive denoise strengths for one frame, computed by the caller from
/// the frame's analogue gain (see `chroma_denoise_for_gain` /
/// `temporal_luma_for_gain` in the video binary). The processor decides on/off
/// (radius and strength > 0 for chroma; alpha > 0 for temporal) and tracks the
/// temporal history reset internally.
#[derive(Clone, Copy, Default)]
pub struct DenoiseParams {
    pub chroma_radius: u32,
    pub chroma_strength: f32,
    pub temporal_alpha: f32,
    pub temporal_motion: f32,
}

pub struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cfa_pipeline: wgpu::ComputePipeline,
    pack_pipeline: wgpu::ComputePipeline,
    planar_pipeline: wgpu::ComputePipeline,
    pack_lca_pipeline: wgpu::ComputePipeline,
    chroma_h_pipeline: wgpu::ComputePipeline,
    chroma_v_pipeline: wgpu::ComputePipeline,
    temporal_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    raw_buf: wgpu::Buffer,
    grid_bufs: [wgpu::Buffer; 4],
    acm_buf: wgpu::Buffer,
    yuyv_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,

    // Cached uniform: the scene-derived fields are refreshed on each re-estimate
    // (`upload_params`); the per-frame denoise fields are set in `process`, which
    // writes the whole block each frame.
    params: Params,
    interval: u64,
    frame: u64,
    lca_on: bool,
    // Whether temporal denoise ran on the previous frame, to seed the history
    // (reset) when it (re)starts after being off.
    temporal_was_on: bool,
    est: Option<Estimate>,
    chroma: Option<(f32, f32)>,
    grids_ls: Option<usize>,

    yuyv_host: Vec<u8>,
    raw_needed: usize,
    yuyv_bytes: usize,

    // Reused CPU-side scratch: AWB working buffers, the flattened per-sector ACM
    // upload buffer, and the f64->f32 grid conversion buffer. Kept here so the
    // periodic re-estimate path does not reallocate them.
    awb: pipeline::AwbScratch,
    acm_flat: Vec<f32>,
    grid_scratch: Vec<f32>,
}

/// All persistent GPU buffers, produced by `GpuProcessor::create_buffers`. `cfa`,
/// `rgb`, and `lca` are referenced only through the bind group (which keeps them
/// alive) and are not retained on the [`GpuProcessor`] itself.
struct GpuBuffers {
    params: wgpu::Buffer,
    raw: wgpu::Buffer,
    grids: [wgpu::Buffer; 4],
    cfa: wgpu::Buffer,
    rgb: wgpu::Buffer,
    lca: wgpu::Buffer,
    acm: wgpu::Buffer,
    yuyv: wgpu::Buffer,
    staging: wgpu::Buffer,
    ctmp: wgpu::Buffer,
    prev_y: wgpu::Buffer,
}

/// The compute pipelines, one per shader entry point.
struct Pipelines {
    cfa: wgpu::ComputePipeline,
    pack: wgpu::ComputePipeline,
    planar: wgpu::ComputePipeline,
    pack_lca: wgpu::ComputePipeline,
    chroma_h: wgpu::ComputePipeline,
    chroma_v: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
}

impl GpuProcessor {
    /// Initialise the GPU backend (Vulkan / mesa ANV). Re-estimates AWB/CCM every
    /// `awb_interval` frames (clamped to >= 1; the first frame always estimates).
    /// `lca` enables lateral-chromatic-aberration correction (an extra MHC->planar
    /// pass plus a resampling colour pass).
    pub fn new(awb_interval: u64, lca: bool) -> Result<Self, String> {
        pollster::block_on(Self::new_async(awb_interval, lca))
    }

    async fn new_async(awb_interval: u64, lca: bool) -> Result<Self, String> {
        let (device, queue) = Self::init_device().await?;
        let bufs = Self::create_buffers(&device, &queue);
        let (bind_group, pipelines) = Self::build_pipelines(&device, &bufs);

        Ok(GpuProcessor {
            device,
            queue,
            cfa_pipeline: pipelines.cfa,
            pack_pipeline: pipelines.pack,
            planar_pipeline: pipelines.planar,
            pack_lca_pipeline: pipelines.pack_lca,
            chroma_h_pipeline: pipelines.chroma_h,
            chroma_v_pipeline: pipelines.chroma_v,
            temporal_pipeline: pipelines.temporal,
            bind_group,
            params_buf: bufs.params,
            raw_buf: bufs.raw,
            grid_bufs: bufs.grids,
            acm_buf: bufs.acm,
            yuyv_buf: bufs.yuyv,
            staging_buf: bufs.staging,
            params: Params::zeroed(),
            interval: awb_interval.max(1),
            frame: 0,
            lca_on: lca,
            temporal_was_on: false,
            est: None,
            chroma: None,
            grids_ls: None,
            yuyv_host: vec![0u8; OUT_W * OUT_H * 2],
            raw_needed: H * STRIDE_SAMPLES * 2,
            yuyv_bytes: OUT_W * OUT_H * 2,
            awb: pipeline::AwbScratch::default(),
            acm_flat: Vec::with_capacity(ACM_NSEC * 9),
            grid_scratch: Vec::with_capacity(HH * WW),
        })
    }

    /// Acquire a Vulkan device and queue. Uses the adapter's own limits because
    /// the pipeline binds 7 storage buffers in one compute stage and wgpu's
    /// downlevel defaults cap storage buffers at 4 (too few on ANV).
    async fn init_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no Vulkan adapter: {e}"))?;
        let info = adapter.get_info();
        eprintln!(
            "gpu: {} ({:?}, {:?})",
            info.name, info.device_type, info.backend
        );

        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gc2607-isp"),
                required_features: wgpu::Features::empty(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e}"))
    }

    /// Create every persistent GPU buffer and upload the one-shot data (the four
    /// LCA shift grids). The `cfa`, `rgb`, and `lca` buffers are referenced only
    /// through the bind group, which keeps them alive, so they are not stored on
    /// the [`GpuProcessor`].
    fn create_buffers(device: &wgpu::Device, queue: &wgpu::Queue) -> GpuBuffers {
        let raw_needed = H * STRIDE_SAMPLES * 2;
        let cfa_bytes = (W * H * 4) as u64;
        let grid_bytes = (HH * WW * 4) as u64;
        let yuyv_bytes = OUT_W * OUT_H * 2;

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raw = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw"),
            size: raw_needed as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make_grid = |name: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: grid_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let grids = [
            make_grid("g_gr"),
            make_grid("g_r"),
            make_grid("g_b"),
            make_grid("g_gb"),
        ];
        let cfa = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfa"),
            size: cfa_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // Full-res planar linear RGB buffer ([R | G | B]) for the LCA path.
        let rgb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rgb"),
            size: (W * H * 3 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // The four LCA shift grids concatenated (blue_x, blue_y, red_x, red_y),
        // uploaded once.
        let lca = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lca"),
            size: (4 * LCA_GH * LCA_GW * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let l = Lca::load();
            let mut flat: Vec<f32> = Vec::with_capacity(4 * LCA_GH * LCA_GW);
            for g in l.grids() {
                flat.extend_from_slice(g);
            }
            queue.write_buffer(&lca, 0, bytemuck::cast_slice(&flat));
        }
        // Per-sector colour matrices for the current CCT (9 floats per sector);
        // rewritten on each re-estimate (see `upload_params`).
        let acm = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("acm"),
            size: (ACM_NSEC * 9 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let yuyv = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuyv"),
            size: yuyv_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: yuyv_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Denoise scratch: horizontally-blurred chroma (one u32 per YUYV word)
        // and the temporal luma history (one u32 per output pixel).
        let ctmp = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ctmp"),
            size: ((OUT_W / 2) * OUT_H * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let prev_y = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prev_y"),
            size: (OUT_W * OUT_H * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        GpuBuffers {
            params,
            raw,
            grids,
            cfa,
            rgb,
            lca,
            acm,
            yuyv,
            staging,
            ctmp,
            prev_y,
        }
    }

    /// Compile the shader, build the bind group over `bufs`, and create the four
    /// compute pipelines (one per shader entry point).
    fn build_pipelines(device: &wgpu::Device, bufs: &GpuBuffers) -> (wgpu::BindGroup, Pipelines) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("isp"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("isp-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_rw(6),
                storage_rw(7),
                storage_rw(8),
                storage_ro(9),
                storage_ro(10),
                storage_rw(11),
                storage_rw(12),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("isp-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bufs.params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.raw.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.grids[0].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: bufs.grids[1].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bufs.grids[2].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bufs.grids[3].as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bufs.cfa.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: bufs.yuyv.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bufs.rgb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: bufs.lca.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: bufs.acm.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: bufs.ctmp.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: bufs.prev_y.as_entire_binding(),
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("isp-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipelines = Pipelines {
            cfa: make_pipeline("build_cfa"),
            pack: make_pipeline("render_pack"),
            planar: make_pipeline("mhc_planar"),
            pack_lca: make_pipeline("render_pack_lca"),
            chroma_h: make_pipeline("chroma_h"),
            chroma_v: make_pipeline("chroma_v"),
            temporal: make_pipeline("temporal"),
        };
        (bind_group, pipelines)
    }

    /// Output frame dimensions (YUYV).
    pub fn out_dims(&self) -> (usize, usize) {
        (OUT_W, OUT_H)
    }

    /// The most recent scene estimate, if any frame has been processed.
    pub fn estimate(&self) -> Option<&Estimate> {
        self.est.as_ref()
    }

    /// Re-estimate AWB/CCM on the CPU from `bytes`, and upload fresh LSC grids if
    /// the chosen light source changed.
    fn reestimate(&mut self, bytes: &[u8]) -> io::Result<()> {
        let frame = RawFrame::from_bytes(bytes)?;
        let planes = pipeline::bayer_planes(&frame);
        // Same temporal smoothing / LSC hysteresis as the CPU `Processor`: the
        // first frame (no history) reproduces the stateless estimate exactly.
        let (rg, bg) = pipeline::robust_neutral_into(&planes, &mut self.awb);
        let (srg, sbg) = pipeline::smooth_chroma(self.chroma, rg, bg);
        self.chroma = Some((srg, sbg));
        let prev_ls = self.est.as_ref().map(|e| e.ls);
        let est = pipeline::estimate_from_chroma(srg, sbg, prev_ls);
        if self.grids_ls != Some(est.ls) {
            let grids = pipeline::build_grids(est.ls, HH, WW);
            // Reuse one f32 scratch buffer for the f64->f32 conversion of all
            // four grids instead of allocating a fresh Vec per grid.
            let queue = &self.queue;
            let scratch = &mut self.grid_scratch;
            let mut upload = |buf: &wgpu::Buffer, g: &[f64]| {
                scratch.clear();
                scratch.extend(g.iter().map(|&v| v as f32));
                queue.write_buffer(buf, 0, bytemuck::cast_slice(scratch));
            };
            upload(&self.grid_bufs[0], &grids.g_gr);
            upload(&self.grid_bufs[1], &grids.g_r);
            upload(&self.grid_bufs[2], &grids.g_b);
            upload(&self.grid_bufs[3], &grids.g_gb);
            self.grids_ls = Some(est.ls);
        }
        self.est = Some(est);
        Ok(())
    }

    /// Pack the current estimate into the uniform block and upload it, along with
    /// this frame's per-sector ACM matrices (CCT-interpolated on the CPU).
    fn upload_params(&mut self) {
        let est = self
            .est
            .as_ref()
            .expect("estimate present (upload_params runs right after reestimate)");
        let g = est.gains;
        let c = est.ccm;
        let cct = est.cct;
        // Refresh the scene-derived fields; the per-frame denoise fields
        // (`dn_i`/`dn_f`) are set and uploaded in `process`, so leave them.
        self.params.dims = [W as u32, H as u32, WW as u32, HH as u32];
        self.params.out_dims = [
            OUT_W as u32,
            OUT_H as u32,
            ((W - OUT_W) / 2) as u32,
            ((H - OUT_H) / 2) as u32,
        ];
        self.params.misc = [STRIDE_SAMPLES as u32, (OUT_W / 2) as u32, 0, 0];
        self.params.consts = [BLACK, 1.0 / MAXLIN, 0.0, 0.0];
        self.params.gains = [g[0] as f32, g[1] as f32, g[2] as f32, 0.0];
        self.params.ccm0 = [c[0] as f32, c[1] as f32, c[2] as f32, 0.0];
        self.params.ccm1 = [c[3] as f32, c[4] as f32, c[5] as f32, 0.0];
        self.params.ccm2 = [c[6] as f32, c[7] as f32, c[8] as f32, 0.0];
        self.params.lca = [LCA_GW as f32, LCA_GH as f32, LCA_CELL_X, LCA_CELL_Y];
        self.params.acm_cfg = [ACM_HUE0, ACM_HUE_STEP, ACM_SAT_KNEE as f32, ACM_NSEC as f32];

        // Per-sector matrices for this CCT, flattened sector-major row-major into
        // the reused scratch buffer.
        let acm = interp_acm(cct);
        self.acm_flat.clear();
        self.acm_flat.extend(
            acm.mats
                .iter()
                .flat_map(|mat| mat.iter().map(|&v| v as f32)),
        );
        self.queue
            .write_buffer(&self.acm_buf, 0, bytemuck::cast_slice(&self.acm_flat));
    }

    /// Process one raw SGRBG10 frame, applying gain-adaptive denoise on the GPU
    /// (`dn`); returns the packed YUYV bytes (length `OUT_W*OUT_H*2`), borrowing a
    /// reused host buffer (valid until the next call). The returned frame is
    /// post-denoise, so the caller's AE luma metric is metered on the denoised
    /// output (chroma denoise leaves luma untouched and temporal denoise barely
    /// shifts the mean, so this matches the pre-denoise metric in practice).
    pub fn process(&mut self, bytes: &[u8], dn: DenoiseParams) -> io::Result<&[u8]> {
        if bytes.len() < self.raw_needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "raw too small: {} bytes, need >= {}",
                    bytes.len(),
                    self.raw_needed
                ),
            ));
        }

        // AWB/CCM on the CPU, occasionally.
        if self.est.is_none() || self.frame % self.interval == 0 {
            self.reestimate(bytes)?;
            self.upload_params();
        }

        // Resolve this frame's denoise state and refresh the uniform. Chroma
        // needs a positive radius and strength; temporal needs a positive alpha.
        // The history is reset (seeded, no blend) the frame temporal (re)starts.
        let chroma_on = dn.chroma_radius > 0 && dn.chroma_strength > 0.0;
        let temporal_on = dn.temporal_alpha > 0.0;
        let reset = temporal_on && !self.temporal_was_on;
        self.temporal_was_on = temporal_on;
        self.params.dn_i = [
            dn.chroma_radius,
            chroma_on as u32,
            temporal_on as u32,
            reset as u32,
        ];
        self.params.dn_f = [
            dn.chroma_strength,
            dn.temporal_alpha,
            dn.temporal_motion,
            0.0,
        ];
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));

        // Upload this frame's raw and dispatch the two passes.
        self.queue
            .write_buffer(&self.raw_buf, 0, &bytes[..self.raw_needed]);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_cfa"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cfa_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = ((W * H) as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        if self.lca_on {
            // Pass 2a: MHC into the planar RGB buffer (full frame interior).
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("mhc_planar"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.planar_pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                let groups = ((W * H) as u32).div_ceil(64);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            // Pass 2b: LCA-correct colour + YUYV pack.
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render_pack_lca"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_lca_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (((OUT_W / 2) * OUT_H) as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render_pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (((OUT_W / 2) * OUT_H) as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // Gain-adaptive denoise on the packed YUYV (one word == one chroma
        // sample == two luma pixels): separable chroma box blur, then temporal
        // luma IIR. Each pass covers the (OUT_W/2)*OUT_H words.
        let dn_groups = (((OUT_W / 2) * OUT_H) as u32).div_ceil(64);
        if self.params.dn_i[1] == 1 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_h"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.chroma_h_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(dn_groups, 1, 1);
            drop(pass);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("chroma_v"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.chroma_v_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(dn_groups, 1, 1);
        }
        if self.params.dn_i[2] == 1 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(dn_groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &self.yuyv_buf,
            0,
            &self.staging_buf,
            0,
            self.yuyv_bytes as u64,
        );
        self.queue.submit(Some(encoder.finish()));

        // Read the packed YUYV back.
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| io::Error::other(format!("gpu poll: {e}")))?;
        rx.recv()
            .map_err(|e| io::Error::other(format!("gpu map channel: {e}")))?
            .map_err(|e| io::Error::other(format!("gpu map: {e}")))?;
        // Copy the readback out, guarding against a size mismatch (which would
        // make copy_from_slice panic) by surfacing it as an Err instead. `data`
        // is dropped before unmap so the buffer is no longer borrowed.
        let mismatch = {
            let data = slice.get_mapped_range();
            if data.len() == self.yuyv_host.len() {
                self.yuyv_host.copy_from_slice(&data);
                None
            } else {
                Some((data.len(), self.yuyv_host.len()))
            }
        };
        self.staging_buf.unmap();
        if let Some((got, want)) = mismatch {
            return Err(io::Error::other(format!(
                "gpu readback size {got} != expected {want}"
            )));
        }

        self.frame = self.frame.wrapping_add(1);
        Ok(&self.yuyv_host)
    }
}
