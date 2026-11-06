//! Frame snapshots for debugging a live session.
//!
//! While a video call holds the loopback node no second reader can open it, so
//! the frame the other side sees cannot be captured from outside. The daemon
//! writes it itself on request: the raw sensor frame (the exact bytes the
//! offline `gc2607-isp` CLI and the golden tools read), the output frame as PNG,
//! and a text file with the tuning, the scene estimate and the AWB statistics.

use std::io;
use std::path::{Path, PathBuf};

use crate::output::yuyv_to_rgb;

/// One captured moment of a live session.
pub struct Snapshot {
    /// Raw SGRBG10 capture bytes, as the camera delivered them.
    pub raw: Vec<u8>,
    /// Output YUYV frame (`w*h*2`), as written to the loopback.
    pub yuyv: Vec<u8>,
    pub w: usize,
    pub h: usize,
    /// Free-form state: tuning, estimate, AWB statistics, exposure.
    pub info: String,
}

/// Write `snap` to `dir` as `<name>.raw`, `<name>.png` and `<name>.txt`,
/// creating `dir` if needed. Returns the path stem.
pub fn write(dir: &Path, name: &str, snap: &Snapshot) -> io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let stem = dir.join(name);
    std::fs::write(stem.with_extension("raw"), &snap.raw)?;
    let rgb = yuyv_to_rgb(&snap.yuyv, snap.w, snap.h);
    image::save_buffer(
        stem.with_extension("png"),
        &rgb,
        snap.w as u32,
        snap.h as u32,
        image::ExtendedColorType::Rgb8,
    )
    .map_err(io::Error::other)?;
    std::fs::write(stem.with_extension("txt"), &snap.info)?;
    Ok(stem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn writes_three_files() {
        let dir = std::env::temp_dir().join(format!("gc2607-snap-{}", std::process::id()));
        let snap = Snapshot {
            raw: vec![1, 2, 3, 4],
            yuyv: vec![128; 4 * 2 * 2],
            w: 4,
            h: 2,
            info: "tuning: test\n".into(),
        };
        let stem = write(&dir, "snap-1", &snap).unwrap();
        assert_eq!(std::fs::read(stem.with_extension("raw")).unwrap(), snap.raw);
        assert_eq!(
            std::fs::read_to_string(stem.with_extension("txt")).unwrap(),
            snap.info
        );
        let png = image::open(stem.with_extension("png")).unwrap();
        assert_eq!((png.width(), png.height()), (4, 2));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
