//! Editing the run-time tuning file in place: change one value without
//! touching comments or layout, and replace the file atomically only when the
//! service would accept it.

use std::fmt;
use std::io::Write;
use std::path::Path;

use crate::colour::ColourTuning;

/// Split a line into `(key, value_start, value_end)` when it assigns a key.
fn assignment(line: &str) -> Option<(&str, usize, usize)> {
    let code_end = line.find('#').unwrap_or(line.len());
    let eq = line[..code_end].find('=')?;
    let key = line[..eq].trim();
    let rest = &line[eq + 1..code_end];
    let start = eq + 1 + (rest.len() - rest.trim_start().len());
    let end = start + rest.trim().len();
    (!key.is_empty()).then_some((key, start, end))
}

/// Format a value the way the example file does: whole numbers without a
/// fraction, others with at most four decimals and no trailing zeros.
fn fmt_value(v: f64) -> String {
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{v:.4}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// The text with `key` set to `value`. The parser lets a later assignment
/// override an earlier one, so the last assignment of `key` is the one
/// changed. The value token is replaced in place, padded to its old width so
/// aligned comments stay aligned; a key absent from the text is appended as
/// `key = value`.
pub fn set_value(text: &str, key: &str, value: f64) -> String {
    let new = fmt_value(value);
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let target = lines
        .iter()
        .rposition(|l| assignment(l).is_some_and(|(k, _, _)| k == key));
    let mut out = String::with_capacity(text.len() + 32);
    for (i, line) in lines.iter().enumerate() {
        if Some(i) != target {
            out.push_str(line);
            continue;
        }
        let (_, s, e) = assignment(line).expect("target line assigns the key");
        let old_w = e - s;
        let tail = &line[e..];
        let commented = tail.trim_start().starts_with('#');
        out.push_str(&line[..s]);
        out.push_str(&new);
        if commented && new.len() < old_w {
            // Pad so the trailing comment keeps its column.
            out.push_str(&" ".repeat(old_w - new.len()));
            out.push_str(tail);
        } else if commented && new.len() > old_w {
            // Take the extra width from the spaces before the comment, but
            // keep at least one.
            let spaces = tail.len() - tail.trim_start().len();
            let keep = spaces.saturating_sub(new.len() - old_w).max(1);
            out.push_str(&" ".repeat(keep));
            out.push_str(tail.trim_start());
        } else {
            out.push_str(tail);
        }
    }
    if target.is_none() {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&format!("{key} = {new}\n"));
    }
    out
}

/// The value assigned to `key` in the text (the last assignment, as the
/// parser reads it), if any.
pub fn get_value(text: &str, key: &str) -> Option<f64> {
    text.lines()
        .rev()
        .filter_map(|l| assignment(l).map(|(k, s, e)| (k, &l[s..e])))
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.parse().ok())
}

/// Why a write did not happen.
#[derive(Debug)]
pub enum WriteError {
    /// The service would reject the text; the file was left as it was.
    Invalid(String),
    /// The file system refused (no permission, missing directory, ...).
    Io(std::io::Error),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WriteError::Invalid(e) => write!(f, "rejected: {e}"),
            WriteError::Io(e) => write!(f, "{e}"),
        }
    }
}

/// Replace the file with `text` if it parses: write a temporary file in the
/// same directory, flush it to disk, then rename it over the target so the
/// service never reads a half-written file.
pub fn write_atomic(path: &Path, text: &str) -> Result<(), WriteError> {
    ColourTuning::parse(text).map_err(WriteError::Invalid)?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("colour.conf");
    let tmp = dir.join(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
        if let Ok(meta) = std::fs::metadata(path) {
            std::fs::set_permissions(&tmp, meta.permissions())?;
        }
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.map_err(WriteError::Io)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str =
        "# head\nskin_desat_k = 0.35     # 0..1.5\nsaturation = 1.0        # 0..2\n";

    #[test]
    fn set_value_keeps_comment_and_alignment() {
        let out = set_value(SAMPLE, "skin_desat_k", 0.55);
        assert_eq!(
            out,
            "# head\nskin_desat_k = 0.55     # 0..1.5\nsaturation = 1.0        # 0..2\n"
        );
        // Shorter and longer values keep the comment column where they can.
        let out = set_value(SAMPLE, "skin_desat_k", 1.0);
        assert!(out.contains("skin_desat_k = 1        # 0..1.5\n"), "{out}");
        let out = set_value(SAMPLE, "skin_desat_k", 0.1234);
        assert!(out.contains("skin_desat_k = 0.1234   # 0..1.5\n"), "{out}");
    }

    #[test]
    fn set_value_appends_a_missing_key() {
        let out = set_value(SAMPLE, "ae_target", 0.36);
        assert!(
            out.ends_with("saturation = 1.0        # 0..2\nae_target = 0.36\n"),
            "{out}"
        );
        let out = set_value("saturation = 1.0", "ae_target", 0.36);
        assert_eq!(out, "saturation = 1.0\nae_target = 0.36\n");
    }

    #[test]
    fn a_repeated_key_is_edited_where_the_parser_reads_it() {
        let text = "saturation = 1.0\nsaturation = 1.1\n";
        let out = set_value(text, "saturation", 1.3);
        assert_eq!(out, "saturation = 1.0\nsaturation = 1.3\n");
        assert_eq!(get_value(&out, "saturation"), Some(1.3));
        assert_eq!(ColourTuning::parse(&out).unwrap().saturation, 1.3);
    }

    #[test]
    fn get_value_reads_the_effective_value() {
        assert_eq!(get_value(SAMPLE, "saturation"), Some(1.0));
        assert_eq!(get_value(SAMPLE, "ae_target"), None);
        assert_eq!(get_value("# saturation = 2\n", "saturation"), None);
    }

    #[test]
    fn every_key_round_trips_through_the_parser() {
        let base = include_str!("../packaging/colour.conf.example");
        for k in crate::colour::KEYS {
            // hue_lo at its max would exceed hue_hi: that pair is checked by
            // the parser on purpose.
            if k.name.ends_with("_hue_lo") {
                continue;
            }
            let v = if k.default == 0.0 && k.min > 0.0 {
                k.min
            } else {
                k.max
            };
            let mut out = set_value(base, k.name, v);
            // The fixed white-balance chroma is valid only as a pair.
            let other = match k.name {
                "wb_fixed_rg" => Some("wb_fixed_bg"),
                "wb_fixed_bg" => Some("wb_fixed_rg"),
                _ => None,
            };
            if let Some(other) = other {
                out = set_value(&out, other, 1.0);
            }
            ColourTuning::parse(&out).unwrap_or_else(|e| panic!("{}: {e}", k.name));
            assert_eq!(get_value(&out, k.name), Some(v), "{}", k.name);
            assert_eq!(out.lines().count(), base.lines().count(), "{}", k.name);
        }
    }

    #[test]
    fn write_atomic_refuses_an_invalid_file() {
        let dir = std::env::temp_dir().join(format!("gc2607-tune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("colour.conf");
        std::fs::write(&path, SAMPLE).unwrap();
        let bad = set_value(SAMPLE, "saturation", 5.0);
        assert!(matches!(
            write_atomic(&path, &bad),
            Err(WriteError::Invalid(_))
        ));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), SAMPLE);
        let good = set_value(SAMPLE, "saturation", 1.2);
        write_atomic(&path, &good).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), good);
        let leftovers = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(leftovers, 1, "temporary file left behind");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
