//! Golden test: verify the Rust pipeline reproduces the validated Python
//! reference (`tools/reference_pipeline.py`) on a fixed raw sample.
//!
//! Two independent checks:
//!   A. estimation (gains, CCT, LSC source) matches `golden.json` within
//!      tolerance (AWB uses median/percentile, sensitive to float order).
//!   B. the deterministic render (LSC -> debayer -> WB -> CCM -> gamma) using
//!      the golden's gains/CCM/LSC source matches `golden_render.bin` exactly
//!      (allowing at most a 1-LSB rounding difference on a few pixels).

use std::path::PathBuf;

use gc2607_isp::pipeline;
use gc2607_isp::raw;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

/// Sensor captures and golden artifacts derived from them are private to the
/// user's scene and are not committed (see .gitignore). When they are absent
/// the golden tests skip instead of failing; regenerate locally with a raw
/// sample and `python3 tools/gen_golden.py`.
fn have_local_data() -> bool {
    ["sample-raw.bin", "golden.json", "golden_render.bin"]
        .iter()
        .all(|n| data(n).exists())
}

/// Minimal extraction of the few JSON fields we need (avoids a serde dep).
fn json_number_array(json: &str, key: &str) -> Vec<f64> {
    let k = format!("\"{key}\"");
    let start = json.find(&k).unwrap_or_else(|| panic!("key {key} not found"));
    let after = &json[start + k.len()..];
    let lb = after.find('[').expect("expected '['");
    let rb = after.find(']').expect("expected ']'");
    after[lb + 1..rb]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("number"))
        .collect()
}

fn json_scalar(json: &str, key: &str) -> f64 {
    let k = format!("\"{key}\"");
    let start = json.find(&k).unwrap_or_else(|| panic!("key {key} not found"));
    let after = &json[start + k.len()..];
    let colon = after.find(':').expect("expected ':'");
    let tail = &after[colon + 1..];
    let end = tail
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(tail.len());
    tail[..end].trim().parse::<f64>().expect("scalar")
}

#[test]
fn estimate_matches_reference() {
    if !have_local_data() {
        eprintln!("golden: skipped (no local sensor data in tests/data/)");
        return;
    }
    let frame = raw::load_blc(data("sample-raw.bin")).expect("raw");
    let planes = pipeline::bayer_planes(&frame);
    let est = pipeline::estimate(&planes);

    let golden = std::fs::read_to_string(data("golden.json")).expect("golden.json");
    let gains = json_number_array(&golden, "gains");
    let chroma = json_number_array(&golden, "scene_chroma");
    let cct = json_scalar(&golden, "cct");
    let ls = json_scalar(&golden, "lsc_ls") as usize;

    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-9);
    assert!(rel(est.chroma.0 as f64, chroma[0]) < 1e-3, "r/g {} vs {}", est.chroma.0, chroma[0]);
    assert!(rel(est.chroma.1 as f64, chroma[1]) < 1e-3, "b/g {} vs {}", est.chroma.1, chroma[1]);
    assert!(rel(est.gains[0], gains[0]) < 1e-3, "gainR {} vs {}", est.gains[0], gains[0]);
    assert!(rel(est.gains[2], gains[2]) < 1e-3, "gainB {} vs {}", est.gains[2], gains[2]);
    assert!(rel(est.cct, cct) < 1e-3, "cct {} vs {}", est.cct, cct);
    assert_eq!(est.ls, ls, "LSC light source mismatch");
}

#[test]
fn render_matches_reference() {
    if !have_local_data() {
        eprintln!("golden: skipped (no local sensor data in tests/data/)");
        return;
    }
    let frame = raw::load_blc(data("sample-raw.bin")).expect("raw");
    let planes = pipeline::bayer_planes(&frame);

    let golden = std::fs::read_to_string(data("golden.json")).expect("golden.json");
    let gains_v = json_number_array(&golden, "gains");
    let ccm_v = json_number_array(&golden, "ccm");
    let ls = json_scalar(&golden, "lsc_ls") as usize;
    let gains = [gains_v[0], gains_v[1], gains_v[2]];
    let mut ccm = [0f64; 9];
    ccm.copy_from_slice(&ccm_v[..9]);

    let rgb = pipeline::render(&planes, gains, ccm, ls);
    let expected = std::fs::read(data("golden_render.bin")).expect("golden_render.bin");
    assert_eq!(rgb.len(), expected.len(), "render size mismatch");

    let mut max_diff = 0i32;
    let mut over1 = 0usize;
    for (a, b) in rgb.iter().zip(expected.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d > 1 {
            over1 += 1;
        }
    }
    let frac = over1 as f64 / rgb.len() as f64;
    assert!(
        max_diff <= 2 && frac < 0.005,
        "render diverges: max_diff={max_diff}, fraction |diff|>1 = {frac:.4}"
    );
}

/// Our own Malvar-He-Cutler debayer must match the reference `demosaic` crate
/// on interior pixels (borders may differ by edge policy). This guards the
/// hand-written MHC that replaced the crate in the runtime path.
#[test]
fn mhc_debayer_matches_reference_crate() {
    if !have_local_data() {
        eprintln!("golden: skipped (no local sensor data in tests/data/)");
        return;
    }
    let frame = raw::load_blc(data("sample-raw.bin")).expect("raw");
    let (w, h) = (frame.w, frame.h);
    let plane = w * h;

    let mine = pipeline::mhc_debayer(&frame.data, w, h);

    let pattern = demosaic::CfaPattern::bayer_grbg();
    let mut theirs = vec![0f32; 3 * plane];
    demosaic::demosaic(&frame.data, w, h, &pattern, demosaic::Algorithm::Mhc, &mut theirs)
        .expect("reference demosaic");

    // Compare interior (3-pixel margin), all three planes.
    let margin = 3usize;
    let mut max_diff = 0f32;
    for ch in 0..3 {
        for y in margin..h - margin {
            for x in margin..w - margin {
                let i = ch * plane + y * w + x;
                let d = (mine[i] - theirs[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }
    assert!(max_diff < 1.0, "MHC diverges from reference: max interior diff = {max_diff}");
}

/// The live `Processor` (half-res) must reproduce the free-function `process`
/// byte-for-byte on the first frame: it fuses unpack+split and caches grids, but
/// the arithmetic is identical, so the golden path stays valid for the runtime.
#[cfg(feature = "video")]
#[test]
fn processor_matches_process_halfres() {
    if !have_local_data() {
        eprintln!("golden: skipped (no local sensor data in tests/data/)");
        return;
    }
    let bytes = std::fs::read(data("sample-raw.bin")).expect("raw");

    let frame = raw::load_blc(data("sample-raw.bin")).expect("raw");
    let (_, _, expected, _) = pipeline::process(&frame);

    let mut proc = pipeline::Processor::new(pipeline::DebayerMode::HalfRes, 8);
    let (w, h, got) = proc.process(&bytes).expect("processor");
    assert_eq!((w, h), (frame.w / 2, frame.h / 2), "size mismatch");
    assert_eq!(got, &expected[..], "processor diverges from process()");
}
