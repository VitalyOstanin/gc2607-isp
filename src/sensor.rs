//! Minimal V4L2 sub-device control for the GC2607 sensor.
//!
//! The live AE loop drives the sensor directly through its V4L2 sub-device
//! (`/dev/v4l-subdevN`): this is the standard kernel sensor-control interface —
//! the same one libcamera uses internally — and the only way to run a custom AE,
//! because libcamera's soft IPA exposes no manual exposure control. Capturing
//! raw frames through libcamera and setting exposure/gain here concurrently was
//! verified to work: the values persist and scale as expected.
//!
//! Only the three controls the AE needs are wired up (exposure, analogue gain,
//! vertical blanking), via raw `VIDIOC_G/S_CTRL` ioctls. The `v4l` crate is
//! deliberately not used (stale, and we need only a few ioctls).

use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

use crate::ae::AeState;

/// `V4L2_CID_EXPOSURE`.
pub const V4L2_CID_EXPOSURE: u32 = 0x0098_0911;
/// `V4L2_CID_ANALOGUE_GAIN`.
pub const V4L2_CID_ANALOGUE_GAIN: u32 = 0x009e_0903;
/// `V4L2_CID_VBLANK`.
pub const V4L2_CID_VBLANK: u32 = 0x009e_0901;

/// `struct v4l2_control` (uapi/linux/videodev2.h).
#[repr(C)]
struct V4l2Control {
    id: u32,
    value: i32,
}

// VIDIOC_G_CTRL = _IOWR('V', 27, struct v4l2_control)
// VIDIOC_S_CTRL = _IOWR('V', 28, struct v4l2_control)
nix::ioctl_readwrite!(vidioc_g_ctrl, b'V', 27, V4l2Control);
nix::ioctl_readwrite!(vidioc_s_ctrl, b'V', 28, V4l2Control);

/// An opened GC2607 sensor sub-device.
pub struct Sensor {
    file: File,
    path: PathBuf,
}

impl Sensor {
    /// Locate the GC2607 sub-device by its media-entity name and open it.
    ///
    /// The sub-device node number is not stable across boots, so it is resolved
    /// from `/sys/class/video4linux/v4l-subdev*/name` (the entity is named
    /// `gc2607 <bus>-<addr>`).
    pub fn open_gc2607() -> io::Result<Sensor> {
        let path = find_gc2607_subdev()?;
        Sensor::open(&path)
    }

    /// Open a specific sub-device node.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Sensor> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        Ok(Sensor { file, path })
    }

    /// The opened sub-device path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn get_ctrl(&self, id: u32) -> io::Result<i32> {
        let mut ctrl = V4l2Control { id, value: 0 };
        // SAFETY: `ctrl` is a valid, correctly-sized v4l2_control; the fd is open.
        unsafe { vidioc_g_ctrl(self.file.as_raw_fd(), &mut ctrl) }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
        Ok(ctrl.value)
    }

    fn set_ctrl(&self, id: u32, value: i32) -> io::Result<()> {
        let mut ctrl = V4l2Control { id, value };
        // SAFETY: `ctrl` is a valid, correctly-sized v4l2_control; the fd is open.
        unsafe { vidioc_s_ctrl(self.file.as_raw_fd(), &mut ctrl) }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
        Ok(())
    }

    /// Current integration time in lines.
    pub fn exposure(&self) -> io::Result<i32> {
        self.get_ctrl(V4L2_CID_EXPOSURE)
    }

    /// Current analogue gain LUT index (0..=16).
    pub fn analogue_gain(&self) -> io::Result<i32> {
        self.get_ctrl(V4L2_CID_ANALOGUE_GAIN)
    }

    /// Current vertical blanking in lines.
    pub fn vblank(&self) -> io::Result<i32> {
        self.get_ctrl(V4L2_CID_VBLANK)
    }

    /// Apply an AE state to the sensor.
    ///
    /// Vertical blanking is written first: the driver couples the exposure
    /// control's maximum to the frame length, so setting `vblank` before
    /// `exposure` keeps the requested exposure within range in both directions
    /// (when shrinking the frame the driver clamps exposure, which the
    /// subsequent write then sets to its new, smaller value).
    pub fn apply(&self, state: AeState) -> io::Result<()> {
        self.set_ctrl(V4L2_CID_VBLANK, state.vblank)?;
        self.set_ctrl(V4L2_CID_EXPOSURE, state.exposure)?;
        self.set_ctrl(V4L2_CID_ANALOGUE_GAIN, state.gain_index as i32)?;
        Ok(())
    }
}

/// Resolve the GC2607 sub-device node from sysfs entity names.
fn find_gc2607_subdev() -> io::Result<PathBuf> {
    let sys = Path::new("/sys/class/video4linux");
    for entry in std::fs::read_dir(sys)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("v4l-subdev") {
            continue;
        }
        let label = std::fs::read_to_string(entry.path().join("name")).unwrap_or_default();
        if label.trim_start().starts_with("gc2607") {
            return Ok(PathBuf::from("/dev").join(name.as_ref()));
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        "gc2607 v4l-subdev not found (is the sensor driver loaded?)",
    ))
}
