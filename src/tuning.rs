//! Access to the embedded tuning data.
//!
//! Scalar tables (CCT, locus, CCM, LSC chroma, dimensions) live in the
//! generated `tuning_data` module. The LSC gain grids are embedded as a flat
//! f32 little-endian blob, laid out `[ls][ch][gh][gw]`, values already gain
//! (raw / scale). Bayer channel order is Gr, R, B, Gb.

use crate::tuning_data::{LSC_GH, LSC_GW, LSC_NCH};

static LSC_BYTES: &[u8] = include_bytes!("../data/lsc_grids.bin");

/// Return the `gh*gw` gain grid for light source `ls`, Bayer channel `ch`
/// (0=Gr, 1=R, 2=B, 3=Gb), row-major.
pub fn lsc_grid(ls: usize, ch: usize) -> Vec<f32> {
    let n = LSC_GH * LSC_GW;
    let start = (ls * LSC_NCH + ch) * n * 4;
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let i = start + k * 4;
        out.push(f32::from_le_bytes([
            LSC_BYTES[i],
            LSC_BYTES[i + 1],
            LSC_BYTES[i + 2],
            LSC_BYTES[i + 3],
        ]));
    }
    out
}
