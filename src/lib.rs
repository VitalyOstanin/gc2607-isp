//! Software ISP for the GalaxyCore GC2607 colour camera.
//!
//! Pipeline: black-level -> lens-shading correction -> robust-neutral AWB ->
//! debayer -> colour-correction matrix (interpolated by CCT) -> sRGB gamma.
//! Tuning (CCM, white locus, LSC grids) is parsed from the camera's `.aiqb`
//! tuning file; see `tools/` and `data/`.

pub mod tuning_data;
pub mod tuning;
pub mod raw;
pub mod pipeline;
pub mod ae;

// Live V4L2 sensor control (AE) and loopback output need raw ioctls; gated to
// keep the offline ISP std-only.
#[cfg(feature = "video")]
pub mod sensor;
#[cfg(feature = "video")]
pub mod output;

// GPU backend (wgpu / Vulkan): per-pixel ISP stages as compute shaders. Gated to
// keep the host build dependency-light when not used.
#[cfg(feature = "gpu")]
pub mod gpu;

pub use pipeline::Estimate;
pub use raw::{RawFrame, H, MAXLIN, W};
