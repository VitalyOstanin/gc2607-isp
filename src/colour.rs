//! Run-time colour tuning.
//!
//! Every knob here used to be a compile-time constant, so re-tuning skin
//! rendition for a new scene meant a rebuild, a reinstall and a service restart
//! per try. [`ColourTuning`] carries the same values at run time; its `Default`
//! equals the built-in constants, so an absent file renders exactly as before.
//!
//! The file is plain `key = value` lines; `#` starts a comment and keys left out
//! keep their defaults. [`TuningWatcher`] re-reads it whenever its modification
//! time or size changes, so a running call can be corrected live. A file that
//! fails to parse or validate is reported and ignored: the previous tuning stays
//! in effect.

use std::fmt;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::pipeline::{
    AWB_LOCUS_MAX_DIST, AWB_LSC, SKIN_DESAT_K, SKIN_HUE_HI, SKIN_HUE_LO, SKIN_HUE_SHIFT,
    SKIN_HUE_SOFT, WB_BLUE_TRIM, WB_RED_TRIM, YELLOW_DESAT_K, YELLOW_HUE_HI, YELLOW_HUE_LO,
    YELLOW_HUE_SOFT,
};

/// Default location of the run-time tuning file.
pub const DEFAULT_PATH: &str = "/etc/gc2607-isp/colour.conf";

/// A hue band: trapezoidal window `[lo, hi]` (degrees, signed, so a band may
/// straddle 0) with linear ramps of width `soft`, and the chroma factor `k`
/// applied on its flat top.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub lo: f64,
    pub hi: f64,
    pub soft: f64,
    pub k: f64,
}

/// All run-time colour knobs. See the module docs and `colour.conf.example`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColourTuning {
    /// Multiplier on the red white-balance gain (1 = as estimated).
    pub wb_red_trim: f64,
    /// Multiplier on the blue white-balance gain (Axis-1 cooling trim).
    pub wb_blue_trim: f64,
    /// Fixed scene chroma `(r/g, b/g)` replacing the AWB measurement, or `None`
    /// for automatic white balance. Drives gains, CCT and matrix choice alike.
    pub wb_fixed: Option<(f64, f64)>,
    /// Weight of the per-hue-sector matrices against the global CCM
    /// (1 = calibrated ACM, 0 = global CCM only).
    pub acm_strength: f64,
    /// Skin band: desaturation toward luma.
    pub skin: Band,
    /// Hue rotation applied inside the skin band, degrees; positive turns red
    /// toward orange. Weighted by the band window, so its edges blend.
    pub skin_hue_shift: f64,
    /// Yellow band: desaturation toward luma.
    pub yellow: Band,
    /// Global chroma factor applied last (1 = unchanged).
    pub saturation: f64,
    /// AE target in linear light, or `None` to keep the command-line value.
    pub ae_target: Option<f64>,
    /// Measure the AWB on lens-shading corrected planes (off: on the raw
    /// planes, where the vignetting lets the frame centre decide alone).
    pub awb_lsc: bool,
    /// Maximum distance of an AWB sample's chroma to the white locus, in raw
    /// `(r/g, b/g)` space; smaller rejects skin and other off-white colours.
    pub awb_locus_dist: f64,
    /// Chroma-denoise box-blur radius on the chroma grid at every gain, or
    /// `None` for the gain table (0 below ~2x gain).
    pub denoise_chroma_radius: Option<usize>,
    /// Chroma-denoise blend strength (0..1) at every gain, or `None` for the
    /// gain table scaled by `--denoise`.
    pub denoise_chroma_strength: Option<f64>,
    /// Debugging switches: stage bypasses and diagnostics.
    pub debug: Debug,
    /// Snapshot request counter: each change to a new non-zero value asks the
    /// daemon for one snapshot of the current frame.
    pub snapshot: u64,
}

/// Stage bypasses and diagnostics, for finding which stage spoils the colour
/// without a rebuild. Every stage is on and every diagnostic off by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Debug {
    /// Lens-shading correction (off: flat unit gain grids).
    pub lsc: bool,
    /// Colour matrices, global and per hue sector (off: identity, the picture
    /// is white-balanced sensor RGB).
    pub ccm: bool,
    /// Skin and yellow hue bands and the global saturation.
    pub bands: bool,
    /// Chroma and temporal denoise.
    pub denoise: bool,
    /// Lateral chromatic aberration correction (only where it was enabled at
    /// start-up).
    pub lca: bool,
    /// AWB statistics in the periodic log line.
    pub awb: bool,
    /// Tint the pixels the AWB measured on, in the output video.
    pub awb_map: bool,
    /// Per-stage frame timings in the periodic log line.
    pub timing: bool,
}

impl Default for Debug {
    fn default() -> Self {
        Debug {
            lsc: true,
            ccm: true,
            bands: true,
            denoise: true,
            lca: true,
            awb: false,
            awb_map: false,
            timing: false,
        }
    }
}

impl Default for ColourTuning {
    fn default() -> Self {
        ColourTuning {
            wb_red_trim: WB_RED_TRIM,
            wb_blue_trim: WB_BLUE_TRIM,
            wb_fixed: None,
            acm_strength: 1.0,
            skin: Band {
                lo: SKIN_HUE_LO,
                hi: SKIN_HUE_HI,
                soft: SKIN_HUE_SOFT,
                k: SKIN_DESAT_K,
            },
            skin_hue_shift: SKIN_HUE_SHIFT,
            yellow: Band {
                lo: YELLOW_HUE_LO,
                hi: YELLOW_HUE_HI,
                soft: YELLOW_HUE_SOFT,
                k: YELLOW_DESAT_K,
            },
            saturation: 1.0,
            ae_target: None,
            awb_lsc: AWB_LSC,
            awb_locus_dist: AWB_LOCUS_MAX_DIST,
            denoise_chroma_radius: None,
            denoise_chroma_strength: None,
            debug: Debug::default(),
            snapshot: 0,
        }
    }
}

/// Largest accepted chroma-denoise radius (the blur cost grows with it).
pub const DENOISE_CHROMA_RADIUS_MAX: usize = 8;

/// Accepted range per key: `(key, min, max)`. Keys whose value 0 means "off"
/// or "automatic" (`ZERO_MEANS_OFF`) accept 0 in addition to the range.
const RANGES: &[(&str, f64, f64)] = &[
    ("wb_red_trim", 0.5, 2.0),
    ("wb_blue_trim", 0.5, 2.0),
    ("wb_fixed_rg", 0.1, 4.0),
    ("wb_fixed_bg", 0.1, 4.0),
    ("acm_strength", 0.0, 1.0),
    ("skin_hue_lo", -180.0, 180.0),
    ("skin_hue_hi", -180.0, 180.0),
    ("skin_hue_soft", 0.5, 90.0),
    ("skin_desat_k", 0.0, 1.5),
    ("skin_hue_shift", -60.0, 60.0),
    ("yellow_hue_lo", -180.0, 180.0),
    ("yellow_hue_hi", -180.0, 180.0),
    ("yellow_hue_soft", 0.5, 90.0),
    ("yellow_desat_k", 0.0, 1.5),
    ("saturation", 0.0, 2.0),
    ("ae_target", 0.02, 0.6),
    ("awb_lsc", 0.0, 1.0),
    ("awb_locus_dist", 0.02, 0.5),
    (
        "denoise_chroma_radius",
        1.0,
        DENOISE_CHROMA_RADIUS_MAX as f64,
    ),
    ("denoise_chroma_strength", 0.01, 1.0),
    ("debug_lsc", 0.0, 1.0),
    ("debug_ccm", 0.0, 1.0),
    ("debug_bands", 0.0, 1.0),
    ("debug_denoise", 0.0, 1.0),
    ("debug_lca", 0.0, 1.0),
    ("debug_awb", 0.0, 1.0),
    ("debug_awb_map", 0.0, 1.0),
    ("debug_timing", 0.0, 1.0),
    ("snapshot", 0.0, 1e9),
];

/// Keys that take only whole numbers (switches are 0 or 1).
const INTEGER_KEYS: &[&str] = &[
    "awb_lsc",
    "denoise_chroma_radius",
    "debug_lsc",
    "debug_ccm",
    "debug_bands",
    "debug_denoise",
    "debug_lca",
    "debug_awb",
    "debug_awb_map",
    "debug_timing",
    "snapshot",
];

const ZERO_MEANS_OFF: &[&str] = &[
    "wb_fixed_rg",
    "wb_fixed_bg",
    "ae_target",
    "denoise_chroma_radius",
    "denoise_chroma_strength",
];

impl ColourTuning {
    /// Parse a tuning file's text over the defaults. Unknown keys, malformed
    /// lines and out-of-range values are errors (the whole file is rejected).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut t = ColourTuning::default();
        let (mut fixed_rg, mut fixed_bg) = (0.0, 0.0);
        for (n, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let lineno = n + 1;
            let (key, value) = line
                .split_once('=')
                .ok_or_else(|| format!("line {lineno}: expected `key = value`"))?;
            let key = key.trim();
            let value: f64 = value
                .trim()
                .parse()
                .map_err(|_| format!("line {lineno}: {key}: not a number: {:?}", value.trim()))?;
            let &(_, min, max) = RANGES
                .iter()
                .find(|(k, _, _)| *k == key)
                .ok_or_else(|| format!("line {lineno}: unknown key {key:?}"))?;
            let off = value == 0.0 && ZERO_MEANS_OFF.contains(&key);
            if !off && !(min..=max).contains(&value) {
                return Err(format!(
                    "line {lineno}: {key} = {value} is outside {min}..{max}"
                ));
            }
            if INTEGER_KEYS.contains(&key) && value.fract() != 0.0 {
                return Err(format!(
                    "line {lineno}: {key} = {value} must be a whole number"
                ));
            }
            let on = value != 0.0;
            match key {
                "wb_red_trim" => t.wb_red_trim = value,
                "wb_blue_trim" => t.wb_blue_trim = value,
                "wb_fixed_rg" => fixed_rg = value,
                "wb_fixed_bg" => fixed_bg = value,
                "acm_strength" => t.acm_strength = value,
                "skin_hue_lo" => t.skin.lo = value,
                "skin_hue_hi" => t.skin.hi = value,
                "skin_hue_soft" => t.skin.soft = value,
                "skin_desat_k" => t.skin.k = value,
                "skin_hue_shift" => t.skin_hue_shift = value,
                "yellow_hue_lo" => t.yellow.lo = value,
                "yellow_hue_hi" => t.yellow.hi = value,
                "yellow_hue_soft" => t.yellow.soft = value,
                "yellow_desat_k" => t.yellow.k = value,
                "saturation" => t.saturation = value,
                "ae_target" => t.ae_target = (!off).then_some(value),
                "awb_lsc" => t.awb_lsc = on,
                "awb_locus_dist" => t.awb_locus_dist = value,
                "denoise_chroma_radius" => {
                    t.denoise_chroma_radius = (!off).then_some(value as usize)
                }
                "denoise_chroma_strength" => t.denoise_chroma_strength = (!off).then_some(value),
                "debug_lsc" => t.debug.lsc = on,
                "debug_ccm" => t.debug.ccm = on,
                "debug_bands" => t.debug.bands = on,
                "debug_denoise" => t.debug.denoise = on,
                "debug_lca" => t.debug.lca = on,
                "debug_awb" => t.debug.awb = on,
                "debug_awb_map" => t.debug.awb_map = on,
                "debug_timing" => t.debug.timing = on,
                "snapshot" => t.snapshot = value as u64,
                _ => unreachable!("every key in RANGES is matched"),
            }
        }
        t.wb_fixed = match (fixed_rg > 0.0, fixed_bg > 0.0) {
            (true, true) => Some((fixed_rg, fixed_bg)),
            (false, false) => None,
            _ => return Err("wb_fixed_rg and wb_fixed_bg must be set together".into()),
        };
        for (name, band) in [("skin", &t.skin), ("yellow", &t.yellow)] {
            if band.lo >= band.hi {
                return Err(format!(
                    "{name}_hue_lo ({}) must be below {name}_hue_hi ({})",
                    band.lo, band.hi
                ));
            }
        }
        Ok(t)
    }
}

impl fmt::Display for ColourTuning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fixed = match self.wb_fixed {
            Some((rg, bg)) => format!("{rg:.3}/{bg:.3}"),
            None => "auto".into(),
        };
        let ae = match self.ae_target {
            Some(v) => format!("{v:.3}"),
            None => "cli".into(),
        };
        write!(
            f,
            "wb_trim r={:.3} b={:.3} wb_fixed={fixed} acm={:.2} \
             skin={}..{}/{} k={:.2} shift={:+.1} yellow={}..{}/{} k={:.2} sat={:.2} ae_target={ae}",
            self.wb_red_trim,
            self.wb_blue_trim,
            self.acm_strength,
            self.skin.lo,
            self.skin.hi,
            self.skin.soft,
            self.skin.k,
            self.skin_hue_shift,
            self.yellow.lo,
            self.yellow.hi,
            self.yellow.soft,
            self.yellow.k,
            self.saturation,
        )?;
        write!(
            f,
            " awb_lsc={} awb_locus_dist={:.3}",
            self.awb_lsc as u8, self.awb_locus_dist
        )?;
        let auto = |v: Option<String>| v.unwrap_or_else(|| "auto".into());
        write!(
            f,
            " denoise_chroma r={} s={}",
            auto(self.denoise_chroma_radius.map(|r| r.to_string())),
            auto(self.denoise_chroma_strength.map(|v| format!("{v:.2}"))),
        )?;
        let d = &self.debug;
        let off: Vec<&str> = [
            (d.lsc, "lsc"),
            (d.ccm, "ccm"),
            (d.bands, "bands"),
            (d.denoise, "denoise"),
            (d.lca, "lca"),
        ]
        .iter()
        .filter(|(on, _)| !on)
        .map(|(_, n)| *n)
        .collect();
        if !off.is_empty() {
            write!(f, " DEBUG off={}", off.join(","))?;
        }
        if d.awb {
            write!(f, " debug_awb")?;
        }
        if d.awb_map {
            write!(f, " debug_awb_map")?;
        }
        if d.timing {
            write!(f, " debug_timing")?;
        }
        Ok(())
    }
}

/// Watches the tuning file and re-reads it when it changes.
pub struct TuningWatcher {
    path: PathBuf,
    // Modification time and size at the last read; `None` while the file is
    // absent. The size catches a rewrite within the mtime granularity.
    seen: Option<Option<(SystemTime, u64)>>,
}

/// Outcome of one [`TuningWatcher::poll`].
pub enum Poll {
    /// Nothing changed since the last poll.
    Unchanged,
    /// The file changed (or appeared, or disappeared) and this is the tuning to
    /// apply now; the string says where it came from, for the log.
    Changed(Box<ColourTuning>, String),
    /// The file changed but was rejected; the previous tuning stays in effect.
    Rejected(String),
}

impl TuningWatcher {
    pub fn new(path: impl AsRef<Path>) -> Self {
        TuningWatcher {
            path: path.as_ref().to_path_buf(),
            seen: None,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check the file; the first call always reports its current state.
    pub fn poll(&mut self) -> Poll {
        let stamp = std::fs::metadata(&self.path)
            .ok()
            .map(|m| (m.modified().unwrap_or(SystemTime::UNIX_EPOCH), m.len()));
        if self.seen == Some(stamp) {
            return Poll::Unchanged;
        }
        self.seen = Some(stamp);
        if stamp.is_none() {
            return Poll::Changed(
                Box::default(),
                format!("{} absent, built-in defaults", self.path.display()),
            );
        }
        match std::fs::read_to_string(&self.path) {
            Ok(text) => match ColourTuning::parse(&text) {
                Ok(t) => Poll::Changed(Box::new(t), format!("{}", self.path.display())),
                Err(e) => Poll::Rejected(format!("{}: {e}", self.path.display())),
            },
            Err(e) => Poll::Rejected(format!("{}: {e}", self.path.display())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The defaults must equal the values the constants hold, so an absent file
    /// renders exactly like the reference pipeline and the golden images.
    #[test]
    fn defaults_equal_constants() {
        let t = ColourTuning::default();
        assert_eq!((t.wb_red_trim, t.wb_blue_trim), (WB_RED_TRIM, WB_BLUE_TRIM));
        assert_eq!(
            (t.skin.lo, t.skin.hi, t.skin.soft, t.skin.k),
            (SKIN_HUE_LO, SKIN_HUE_HI, SKIN_HUE_SOFT, SKIN_DESAT_K)
        );
        assert_eq!(
            (t.yellow.lo, t.yellow.hi, t.yellow.soft, t.yellow.k),
            (
                YELLOW_HUE_LO,
                YELLOW_HUE_HI,
                YELLOW_HUE_SOFT,
                YELLOW_DESAT_K
            )
        );
        assert_eq!(
            (t.acm_strength, t.skin_hue_shift, t.saturation),
            (1.0, SKIN_HUE_SHIFT, 1.0)
        );
        assert_eq!((t.wb_fixed, t.ae_target), (None, None));
        assert_eq!((t.awb_lsc, t.awb_locus_dist), (AWB_LSC, AWB_LOCUS_MAX_DIST));
    }

    /// The shipped example lists every key at its default value.
    #[test]
    fn example_file_is_the_defaults() {
        let text = include_str!("../packaging/colour.conf.example");
        assert_eq!(ColourTuning::parse(text).unwrap(), ColourTuning::default());
        for (key, _, _) in RANGES {
            assert!(
                text.lines().any(|l| l.trim_start().starts_with(key)),
                "example lacks {key}"
            );
        }
    }

    #[test]
    fn denoise_chroma_keys_zero_is_auto() {
        let t = ColourTuning::parse("denoise_chroma_radius = 2\ndenoise_chroma_strength = 0.6\n")
            .unwrap();
        assert_eq!(t.denoise_chroma_radius, Some(2));
        assert_eq!(t.denoise_chroma_strength, Some(0.6));
        let t = ColourTuning::parse("denoise_chroma_radius = 0\ndenoise_chroma_strength = 0\n")
            .unwrap();
        assert_eq!(
            (t.denoise_chroma_radius, t.denoise_chroma_strength),
            (None, None)
        );
        assert!(ColourTuning::parse("denoise_chroma_radius = 9\n").is_err());
        assert!(ColourTuning::parse("denoise_chroma_radius = 1.5\n").is_err());
    }

    #[test]
    fn empty_and_comment_only_text_gives_defaults() {
        let t = ColourTuning::parse("# nothing\n\n   \n").unwrap();
        assert_eq!(t, ColourTuning::default());
    }

    #[test]
    fn parses_keys_with_comments() {
        let t = ColourTuning::parse(
            "wb_blue_trim = 0.95  # less blue\nskin_hue_shift=12\nwb_fixed_rg = 0.6\nwb_fixed_bg = 0.45\nae_target = 0.18\n",
        )
        .unwrap();
        assert_eq!(t.wb_blue_trim, 0.95);
        assert_eq!(t.skin_hue_shift, 12.0);
        assert_eq!(t.wb_fixed, Some((0.6, 0.45)));
        assert_eq!(t.ae_target, Some(0.18));
    }

    #[test]
    fn parses_debug_switches() {
        let t = ColourTuning::parse("debug_lsc = 0\ndebug_awb = 1\nsnapshot = 7\n").unwrap();
        assert!(!t.debug.lsc && t.debug.ccm && t.debug.awb && !t.debug.awb_map);
        assert_eq!(t.snapshot, 7);
    }

    #[test]
    fn zero_switches_optional_keys_off() {
        let t = ColourTuning::parse("wb_fixed_rg = 0\nwb_fixed_bg = 0\nae_target = 0\n").unwrap();
        assert_eq!((t.wb_fixed, t.ae_target), (None, None));
    }

    #[test]
    fn rejects_bad_input() {
        for bad in [
            "nonsense",
            "unknown_key = 1",
            "saturation = lots",
            "saturation = 5",
            "wb_fixed_rg = 0.6",
            "skin_hue_lo = 50\nskin_hue_hi = 40",
            "skin_hue_soft = 0",
            "debug_lsc = 0.5",
            "debug_awb = 2",
            "snapshot = 1.5",
        ] {
            assert!(ColourTuning::parse(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn watcher_reports_changes_only() {
        let dir = std::env::temp_dir().join(format!("gc2607-colour-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("colour.conf");
        let _ = std::fs::remove_file(&path);
        let mut w = TuningWatcher::new(&path);
        assert!(matches!(w.poll(), Poll::Changed(t, _) if *t == ColourTuning::default()));
        assert!(matches!(w.poll(), Poll::Unchanged));
        std::fs::write(&path, "saturation = 0.8\n").unwrap();
        assert!(matches!(w.poll(), Poll::Changed(t, _) if t.saturation == 0.8));
        assert!(matches!(w.poll(), Poll::Unchanged));
        std::fs::write(&path, "saturation = 9\n").unwrap();
        assert!(matches!(w.poll(), Poll::Rejected(_)));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
