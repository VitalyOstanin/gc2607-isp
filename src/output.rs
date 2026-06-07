//! Minimal V4L2 output to a v4l2loopback device.
//!
//! The processed frames are pushed to a loopback node (`/dev/videoN`, the
//! "Virtual Camera") as packed YUYV (YUV 4:2:2), the format applications expect
//! from a webcam. v4l2loopback accepts the simple `write()` producer method, so
//! after one `VIDIOC_S_FMT` the frames are written sequentially.
//!
//! As decided for this project, this is a hand-rolled V4L2 layer over raw
//! ioctls (the `v4l` crate is stale and its output path is undocumented). The
//! one fragile point is the `struct v4l2_format` layout: its size and field
//! offsets must match the kernel exactly or the ioctl is rejected. The struct
//! is laid out to the 64-bit kernel ABI (total 208 bytes, the `fmt` union at
//! offset 8) and this is asserted at construction.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::path::Path;

use rayon::prelude::*;

/// `V4L2_PIX_FMT_YUYV` four-character code.
const V4L2_PIX_FMT_YUYV: u32 =
    (b'Y' as u32) | ((b'U' as u32) << 8) | ((b'Y' as u32) << 16) | ((b'V' as u32) << 24);
/// `V4L2_BUF_TYPE_VIDEO_OUTPUT`.
const V4L2_BUF_TYPE_VIDEO_OUTPUT: u32 = 2;
/// `V4L2_FIELD_NONE`.
const V4L2_FIELD_NONE: u32 = 1;
/// `V4L2_COLORSPACE_SRGB`.
const V4L2_COLORSPACE_SRGB: u32 = 8;
/// `V4L2_QUANTIZATION_FULL_RANGE` (full-range YCbCr, matches our sRGB output).
const V4L2_QUANTIZATION_FULL_RANGE: u32 = 1;

/// `struct v4l2_pix_format` (uapi/linux/videodev2.h), 48 bytes on all ABIs.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    private: u32,
    flags: u32,
    enc: u32, // union { ycbcr_enc; hsv_enc }
    quantization: u32,
    xfer_func: u32,
}

/// `struct v4l2_format`. The kernel's `fmt` union is 8-aligned (it contains
/// `v4l2_window`, which has pointers) and starts at offset 8; the whole struct
/// is 208 bytes on 64-bit. `pix` is the first union member.
#[repr(C, align(8))]
struct V4l2Format {
    type_: u32,
    _pad: u32,
    pix: V4l2PixFormat,
    _tail: [u8; 152],
}

// VIDIOC_S_FMT = _IOWR('V', 5, struct v4l2_format)
nix::ioctl_readwrite!(vidioc_s_fmt, b'V', 5, V4l2Format);

/// A loopback output node configured for YUYV frames of a fixed size.
pub struct LoopbackOutput {
    file: File,
    width: u32,
    height: u32,
}

impl LoopbackOutput {
    /// Open `path` and set its output format to YUYV `width`x`height`.
    pub fn open<P: AsRef<Path>>(path: P, width: u32, height: u32) -> io::Result<LoopbackOutput> {
        // Guard the hand-laid ABI: a wrong size would make the kernel reject
        // (or mis-copy) the ioctl.
        const _: () = assert!(core::mem::size_of::<V4l2Format>() == 208);
        const _: () = assert!(core::mem::size_of::<V4l2PixFormat>() == 48);

        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let pix = V4l2PixFormat {
            width,
            height,
            pixelformat: V4L2_PIX_FMT_YUYV,
            field: V4L2_FIELD_NONE,
            bytesperline: width * 2,
            sizeimage: width * height * 2,
            colorspace: V4L2_COLORSPACE_SRGB,
            quantization: V4L2_QUANTIZATION_FULL_RANGE,
            ..Default::default()
        };
        let mut fmt = V4l2Format {
            type_: V4L2_BUF_TYPE_VIDEO_OUTPUT,
            _pad: 0,
            pix,
            _tail: [0u8; 152],
        };
        // SAFETY: `fmt` is a correctly-sized v4l2_format; the fd is open RW.
        unsafe { vidioc_s_fmt(file.as_raw_fd(), &mut fmt) }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;

        Ok(LoopbackOutput {
            file,
            width,
            height,
        })
    }

    /// Frame size in bytes (YUYV: 2 bytes per pixel).
    pub fn frame_size(&self) -> usize {
        self.width as usize * self.height as usize * 2
    }

    /// Write one YUYV frame. The buffer must be exactly [`Self::frame_size`].
    pub fn write_frame(&mut self, yuyv: &[u8]) -> io::Result<()> {
        if yuyv.len() != self.frame_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "frame is {} bytes, expected {} ({}x{} YUYV)",
                    yuyv.len(),
                    self.frame_size(),
                    self.width,
                    self.height
                ),
            ));
        }
        self.file.write_all(yuyv)
    }
}

/// Convert an interleaved RGB8 image to packed YUYV (YUY2), optionally
/// centre-cropping from `src_w`x`src_h` to `dst_w`x`dst_h`.
///
/// Uses full-range BT.601 (JFIF) coefficients to match the full-range sRGB
/// output declared on the loopback format. `dst_w` must be even (YUYV pairs
/// two horizontal pixels); `dst` must hold `dst_w*dst_h*2` bytes.
pub fn rgb_to_yuyv_crop(
    rgb: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    debug_assert!(dst_w % 2 == 0);
    debug_assert!(dst_w <= src_w && dst_h <= src_h);
    debug_assert_eq!(dst.len(), dst_w * dst_h * 2);

    let off_x = (src_w - dst_w) / 2;
    let off_y = (src_h - dst_h) / 2;

    // Row-parallel: each output row depends only on its own source row.
    dst.par_chunks_mut(dst_w * 2).enumerate().for_each(|(y, drow)| {
        let src_row = (off_y + y) * src_w + off_x;
        let mut x = 0;
        while x < dst_w {
            let i0 = (src_row + x) * 3;
            let i1 = (src_row + x + 1) * 3;
            let (y0, cb0, cr0) = rgb_to_ycbcr(rgb[i0], rgb[i0 + 1], rgb[i0 + 2]);
            let (y1, cb1, cr1) = rgb_to_ycbcr(rgb[i1], rgb[i1 + 1], rgb[i1 + 2]);
            // Subsample chroma by averaging the pixel pair.
            let cb = ((cb0 as u16 + cb1 as u16) / 2) as u8;
            let cr = ((cr0 as u16 + cr1 as u16) / 2) as u8;
            let o = x * 2;
            drow[o] = y0;
            drow[o + 1] = cb;
            drow[o + 2] = y1;
            drow[o + 3] = cr;
            x += 2;
        }
    });
}

/// Full-range BT.601 (JFIF) RGB -> YCbCr.
#[inline]
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = 128.0 - 0.168_736 * rf - 0.331_264 * gf + 0.5 * bf;
    let cr = 128.0 + 0.5 * rf - 0.418_688 * gf - 0.081_312 * bf;
    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}
