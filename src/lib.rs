//! Software ISP for the GalaxyCore GC2607 colour camera.
//!
//! Pipeline: black-level -> lens-shading correction -> robust-neutral AWB ->
//! debayer -> hue-sectored colour correction (interpolated by CCT) -> sRGB
//! gamma. Tuning (CCM, ACM sectors, white locus, LSC/LCA grids) is parsed from
//! the camera's `.aiqb` tuning file; see `tools/` and `data/`.

pub mod ae;
pub mod pipeline;
pub mod raw;
pub mod tuning;
pub mod tuning_data;

// Live V4L2 sensor control (AE) and loopback output need raw ioctls; gated to
// keep the offline ISP std-only.
#[cfg(feature = "video")]
pub mod output;
#[cfg(feature = "video")]
pub mod sensor;

// GPU backend (wgpu / Vulkan): per-pixel ISP stages as compute shaders. Gated to
// keep the host build dependency-light when not used.
#[cfg(feature = "gpu")]
pub mod gpu;

// Shared libcamera capture setup for the live and raw-dump binaries. Needs the
// libcamera crate, so it is gated behind `capture` (same as the binaries).
#[cfg(feature = "capture")]
pub mod camera;

pub use pipeline::Estimate;
pub use raw::{RawFrame, H, MAXLIN, W};
