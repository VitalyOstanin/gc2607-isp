//! The CPU live processor measures AWB on a background thread (see
//! `AwbWorker`); the result lands a few frames later. On a repeated frame the
//! estimate it brings is the one the first frame waited for, so every later
//! output must equal the first exactly. Skips without the private sample raw.

#![cfg(feature = "video")]

use std::path::PathBuf;
use std::time::{Duration, Instant};

use gc2607_isp::pipeline::{DebayerMode, Processor};

#[test]
fn cpu_background_awb_keeps_a_steady_frame_steady() {
    let raw_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("background awb: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");
    let mut proc = Processor::new(DebayerMode::HalfRes, 4, false);
    let first = proc.process(&bytes).expect("process").2.to_vec();
    assert_eq!(
        proc.awb_applied(),
        1,
        "the first frame waits for its estimate"
    );
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut n = 0u32;
    while proc.awb_applied() < 4 {
        assert!(
            Instant::now() < deadline,
            "background AWB applied only {} times in {n} frames",
            proc.awb_applied()
        );
        n += 1;
        let out = proc.process(&bytes).expect("process").2;
        assert!(out == &first[..], "frame {n} differs from the first");
        std::thread::sleep(Duration::from_millis(5));
    }
    eprintln!(
        "background AWB: {} measurements over {n} frames",
        proc.awb_applied()
    );
}

/// Frame-time probe for the CPU live path (run with `--release --ignored
/// --nocapture`): with the AWB measurement off the frame loop, the slowest
/// frame should stay close to the mean instead of spiking every interval.
#[test]
#[ignore]
fn cpu_frame_time_probe() {
    let raw_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-raw.bin");
    let bytes = std::fs::read(&raw_path).expect("raw");
    let mut proc = Processor::new(DebayerMode::Mhc, 8, true);
    proc.process(&bytes).expect("process");
    let mut times = Vec::new();
    for _ in 0..48 {
        let t = Instant::now();
        proc.process(&bytes).expect("process");
        times.push(t.elapsed());
        std::thread::sleep(Duration::from_millis(10));
    }
    let mean = times.iter().sum::<Duration>() / times.len() as u32;
    let max = times.iter().max().unwrap();
    eprintln!(
        "cpu mhc frame: mean {mean:?}, max {max:?}, AWB applied {}",
        proc.awb_applied()
    );
}
