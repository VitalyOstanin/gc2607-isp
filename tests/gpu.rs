//! GPU backend validation: the wgpu/Vulkan pipeline must reproduce the CPU MHC
//! path (debayer -> CCM -> sRGB -> YUYV) within rounding on a fixed raw sample.
//!
//! Not bit-exact: the GPU uses `pow` for the sRGB gamma where the CPU MHC path
//! uses a LUT, and float accumulation order differs. The tolerance below checks
//! the two agree to within a couple of LSB on essentially every byte.
//!
//! Requires both `gpu` (the backend) and `video` (the CPU `rgb_to_yuyv_crop`
//! reference) features, plus a working Vulkan device; skips if the sample raw is
//! absent (it is private scene data, not committed).

#![cfg(all(feature = "gpu", feature = "video"))]

use std::path::PathBuf;

use gc2607_isp::gpu::{GpuProcessor, OUT_H, OUT_W};
use gc2607_isp::output::rgb_to_yuyv_crop;
use gc2607_isp::pipeline;
use gc2607_isp::raw;

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

#[test]
fn gpu_yuyv_matches_cpu_mhc() {
    let raw_path = data("sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("gpu: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");

    // CPU reference: estimate, full-res MHC render, then pack to YUYV with the
    // same centre crop the GPU applies.
    let frame = raw::load_blc(&raw_path).expect("raw");
    let planes = pipeline::bayer_planes(&frame);
    let est = pipeline::estimate(&planes);
    let (w, h, rgb) = pipeline::render_mhc(&frame, est.gains, est.ccm, est.cct, est.ls);
    let mut cpu = vec![0u8; OUT_W * OUT_H * 2];
    rgb_to_yuyv_crop(&rgb, w, h, &mut cpu, OUT_W, OUT_H);

    // GPU path.
    let mut proc = match GpuProcessor::new(8, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gpu: skipped (no usable Vulkan device: {e})");
            return;
        }
    };
    let gpu = proc.process(&bytes).expect("gpu process").to_vec();

    assert_eq!(gpu.len(), cpu.len(), "YUYV size mismatch");

    let mut max_diff = 0i32;
    let mut over2 = 0usize;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        let d = (*a as i32 - *b as i32).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d > 2 {
            over2 += 1;
        }
    }
    let frac = over2 as f64 / gpu.len() as f64;
    eprintln!("gpu vs cpu: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}");
    assert!(
        max_diff <= 4 && frac < 0.01,
        "GPU diverges from CPU MHC: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}"
    );
}
