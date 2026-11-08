//! Asking the running service for a snapshot (bump the `snapshot` key) and
//! picking up the PNG it writes.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use crate::tune_file::{get_value, set_value, write_atomic, WriteError};

/// Where the service writes snapshots by default.
pub const SNAPSHOT_DIR: &str = "/var/lib/gc2607-isp/snapshots";

/// A `snapshot` value different from `current`, within the parser's range.
pub fn next_snapshot_number(current: Option<u64>) -> u64 {
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut n = now % 1_000_000_000;
    if n == 0 || Some(n) == current {
        n = n % 999_999_999 + 1;
    }
    n
}

/// The most recently modified `*.png` in `dir`.
pub fn newest_png(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("png"))
        .max_by_key(|p| {
            std::fs::metadata(p)
                .and_then(|m| m.modified())
                .unwrap_or(SystemTime::UNIX_EPOCH)
        })
}

/// Why a snapshot request failed.
#[derive(Debug)]
pub enum SnapshotError {
    Write(WriteError),
    Read(std::io::Error),
    /// No new PNG appeared in time: the service is not running or the camera
    /// is off (snapshots are taken from live frames only).
    Timeout,
    Decode(String),
}

/// Request a snapshot through `conf` and wait up to `timeout` for the PNG;
/// returns `(width, height, rgba)`.
pub fn request(
    conf: &Path,
    dir: &Path,
    timeout: Duration,
) -> Result<(usize, usize, Vec<u8>), SnapshotError> {
    let text = std::fs::read_to_string(conf).map_err(SnapshotError::Read)?;
    let before = newest_png(dir);
    let n = next_snapshot_number(get_value(&text, "snapshot").map(|v| v as u64));
    write_atomic(conf, &set_value(&text, "snapshot", n as f64)).map_err(SnapshotError::Write)?;
    let start = Instant::now();
    while start.elapsed() < timeout {
        std::thread::sleep(Duration::from_millis(200));
        let Some(now) = newest_png(dir) else { continue };
        if Some(&now) != before.as_ref() {
            // The service writes the PNG in one go; give it a moment to finish.
            std::thread::sleep(Duration::from_millis(300));
            let img = image::open(&now)
                .map_err(|e| SnapshotError::Decode(e.to_string()))?
                .to_rgba8();
            let (w, h) = img.dimensions();
            return Ok((w as usize, h as usize, img.into_raw()));
        }
    }
    Err(SnapshotError::Timeout)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_number_changes_and_stays_in_range() {
        let a = next_snapshot_number(Some(5));
        assert_ne!(a, 5);
        assert!((1..=1_000_000_000).contains(&a));
    }

    #[test]
    fn newest_png_picks_the_latest_snapshot() {
        let dir = std::env::temp_dir().join(format!("gc2607-snapreq-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("snap-1.png"), b"x").unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join("snap-2.png"), b"x").unwrap();
        std::fs::write(dir.join("snap-2.raw"), b"x").unwrap();
        assert_eq!(newest_png(&dir).unwrap().file_name().unwrap(), "snap-2.png");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
