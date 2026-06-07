//! Raw frame loading and black-level subtraction.
//!
//! Raw geometry: SGRBG10, 1928x1088, stride 1952 u16/line (width padded
//! 1928->1952), 10-bit values in 16-bit little-endian, Bayer GRBG (top-left Gr).

use std::io;
use std::path::Path;

pub const W: usize = 1928;
pub const H: usize = 1088;
pub const STRIDE_SAMPLES: usize = 1952;
pub const BLACK: f32 = 64.0;
pub const WHITE: f32 = 1023.0;
pub const MAXLIN: f32 = WHITE - BLACK; // 959.0

/// One frame after black-level subtraction; `data` is row-major `w*h`, clamped
/// to be non-negative, still in linear 10-bit scale (0..MAXLIN).
pub struct RawFrame {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

/// Mean linear signal (black-level subtracted) as a fraction of full scale,
/// metered directly from a raw SGRBG10 byte buffer.
///
/// Used by the live AE loop, which has the mapped capture buffer in hand and
/// does not need to build a [`RawFrame`]. Returns `None` if the buffer is too
/// small for the expected geometry.
pub fn mean_norm_from_bytes(bytes: &[u8]) -> Option<f64> {
    let needed = H * STRIDE_SAMPLES * 2;
    if bytes.len() < needed {
        return None;
    }
    let mut sum = 0f64;
    for y in 0..H {
        let row = y * STRIDE_SAMPLES;
        for x in 0..W {
            let i = (row + x) * 2;
            let v = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as f32;
            sum += (v - BLACK).max(0.0) as f64;
        }
    }
    Some(sum / (W * H) as f64 / MAXLIN as f64)
}

impl RawFrame {
    /// Build a frame from a raw SGRBG10 byte buffer, applying black level.
    pub fn from_bytes(bytes: &[u8]) -> io::Result<RawFrame> {
        let needed = H * STRIDE_SAMPLES * 2;
        if bytes.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!(
                    "raw too small: {} bytes, need >= {} ({}x{} stride {})",
                    bytes.len(),
                    needed,
                    W,
                    H,
                    STRIDE_SAMPLES
                ),
            ));
        }
        let mut data = vec![0f32; W * H];
        for y in 0..H {
            let row = y * STRIDE_SAMPLES;
            let out = y * W;
            for x in 0..W {
                let i = (row + x) * 2;
                let v = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as f32;
                data[out + x] = (v - BLACK).max(0.0);
            }
        }
        Ok(RawFrame { w: W, h: H, data })
    }
}

/// Load a raw SGRBG10 frame from a flat binary and apply black level.
pub fn load_blc<P: AsRef<Path>>(path: P) -> io::Result<RawFrame> {
    RawFrame::from_bytes(&std::fs::read(path)?)
}
