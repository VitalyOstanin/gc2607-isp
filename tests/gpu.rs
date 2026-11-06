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

use std::path::{Path, PathBuf};

use gc2607_isp::colour::ColourTuning;
use gc2607_isp::gpu::{DenoiseParams, GpuProcessor, OUT_H, OUT_W};
use gc2607_isp::output::{denoise_chroma_yuyv, rgb_to_yuyv_crop, temporal_denoise_luma_yuyv};
use gc2607_isp::pipeline::{self, DebayerMode, Processor};
use gc2607_isp::raw;

/// Serialises the tests of this file. Each builds its own Vulkan instance, and
/// in a debug build wgpu names objects through the loader's debug-utils
/// trampoline; when one test drops its instance while another is creating
/// buffers, the loader unloads a library the other is calling into and the
/// process crashes (seen intermittently with several tests in flight). The
/// daemon has a single instance for its lifetime and is not affected.
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// Largest absolute byte difference and the fraction of bytes differing by more
/// than 2, between two equal-length YUYV frames.
fn diff_stats(a: &[u8], b: &[u8]) -> (i32, f64) {
    let mut max_diff = 0i32;
    let mut over2 = 0usize;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as i32 - *y as i32).abs();
        max_diff = max_diff.max(d);
        if d > 2 {
            over2 += 1;
        }
    }
    (max_diff, over2 as f64 / a.len() as f64)
}

/// CPU reference YUYV for the sample raw: estimate, full-res MHC, centre-crop pack.
fn cpu_reference(raw_path: &Path) -> Vec<u8> {
    let frame = raw::load_blc(raw_path).expect("raw");
    let planes = pipeline::bayer_planes(&frame);
    let est = pipeline::estimate(&planes);
    let (w, h, rgb) = pipeline::render_mhc(&frame, est.gains, est.ccm, est.cct, est.ls);
    let mut cpu = vec![0u8; OUT_W * OUT_H * 2];
    rgb_to_yuyv_crop(&rgb, w, h, &mut cpu, OUT_W, OUT_H);
    cpu
}

#[test]
fn gpu_yuyv_matches_cpu_mhc() {
    let _gpu = gpu_lock();
    let raw_path = data("sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("gpu: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");

    // CPU reference: estimate, full-res MHC render, then pack to YUYV with the
    // same centre crop the GPU applies.
    let cpu = cpu_reference(&raw_path);

    // GPU path, no denoise (default params) so this compares the pure MHC path.
    let mut proc = match GpuProcessor::new(8, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gpu: skipped (no usable Vulkan device: {e})");
            return;
        }
    };
    let gpu = proc
        .process(&bytes, DenoiseParams::default())
        .expect("gpu process")
        .to_vec();

    assert_eq!(gpu.len(), cpu.len(), "YUYV size mismatch");

    let (max_diff, frac) = diff_stats(&gpu, &cpu);
    eprintln!("gpu vs cpu: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}");
    assert!(
        max_diff <= 4 && frac < 0.01,
        "GPU diverges from CPU MHC: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}"
    );
}

/// With every run-time colour knob away from its default, the GPU shader and
/// the CPU live processor must still agree: both receive the same
/// `ColourTuning`, and the shader's band, hue-rotation, saturation and ACM
/// weighting mirror the CPU arithmetic.
#[test]
fn gpu_matches_cpu_with_runtime_tuning() {
    let _gpu = gpu_lock();
    let raw_path = data("sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("gpu: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");
    // Two tunings: every colour knob moved, then every stage switch off.
    for text in [
        "wb_red_trim = 1.05\nwb_blue_trim = 0.97\nacm_strength = 0.7\n\
         skin_hue_lo = -40\nskin_hue_hi = 40\nskin_hue_soft = 12\nskin_desat_k = 0.5\n\
         skin_hue_shift = 15\nyellow_desat_k = 0.8\nsaturation = 1.2\n",
        "debug_lsc = 0\ndebug_ccm = 0\ndebug_bands = 0\ndebug_lca = 0\n",
    ] {
        compare_tuned(&raw_path, &bytes, text);
    }
}

fn compare_tuned(raw_path: &Path, bytes: &[u8], text: &str) {
    let tuning = ColourTuning::parse(text).expect("tuning");

    let mut cpu_proc = Processor::new(DebayerMode::Mhc, 8, true);
    cpu_proc.set_tuning(tuning);
    let (w, h, rgb) = cpu_proc.process(bytes).expect("cpu process");
    let mut cpu = vec![0u8; OUT_W * OUT_H * 2];
    rgb_to_yuyv_crop(rgb, w, h, &mut cpu, OUT_W, OUT_H);

    let mut proc = match GpuProcessor::new(8, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gpu: skipped (no usable Vulkan device: {e})");
            return;
        }
    };
    proc.set_tuning(tuning);
    let gpu = proc
        .process(bytes, DenoiseParams::default())
        .expect("gpu process")
        .to_vec();

    // The tuning must actually change the picture, or the comparison proves
    // nothing about the tuned arithmetic.
    let (changed, _) = diff_stats(&cpu, &cpu_reference(raw_path));
    assert!(changed > 10, "tuning left the CPU render unchanged");

    let (max_diff, frac) = diff_stats(&gpu, &cpu);
    eprintln!("gpu vs cpu ({text:?}): max_diff={max_diff}, fraction |diff|>2 = {frac:.5}");
    assert!(
        max_diff <= 4 && frac < 0.01,
        "tuned GPU diverges from CPU: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}"
    );
}

/// GPU chroma box-blur denoise must reproduce the CPU `denoise_chroma_yuyv`
/// within rounding. The GPU box blur is integer (bit-exact to the CPU running
/// sum); only the final f32 blend can differ from the CPU f64 blend by ~1 LSB,
/// on top of the existing MHC rounding gap. A radius-2 / strength-0.7 setting
/// matches the mid gain table entry.
#[test]
fn gpu_chroma_denoise_matches_cpu() {
    let _gpu = gpu_lock();
    let raw_path = data("sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("gpu: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");

    let mut cpu = cpu_reference(&raw_path);
    denoise_chroma_yuyv(&mut cpu, OUT_W, OUT_H, 2, 0.7);

    let mut proc = match GpuProcessor::new(8, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gpu: skipped (no usable Vulkan device: {e})");
            return;
        }
    };
    let gpu = proc
        .process(
            &bytes,
            DenoiseParams {
                chroma_radius: 2,
                chroma_strength: 0.7,
                temporal_alpha: 0.0,
                temporal_motion: 0.0,
            },
        )
        .expect("gpu process")
        .to_vec();

    let (max_diff, frac) = diff_stats(&gpu, &cpu);
    eprintln!("gpu chroma vs cpu: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}");
    assert!(
        max_diff <= 5 && frac < 0.01,
        "GPU chroma denoise diverges from CPU: max_diff={max_diff}, frac>2 = {frac:.5}"
    );
}

/// GPU temporal luma denoise: on the first frame the history is seeded (reset),
/// so the output Y must equal the no-denoise output (chroma untouched too). This
/// pins the temporal pass's word packing and per-pixel history indexing; the
/// f32 blend itself shares its structure with the chroma blend validated above.
#[test]
fn gpu_temporal_denoise_seeds_first_frame() {
    let _gpu = gpu_lock();
    let raw_path = data("sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("gpu: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");

    // CPU reference: MHC YUYV, then a seeding temporal pass (empty history ->
    // output unchanged, history filled). Output must equal the plain MHC YUYV.
    let cpu = cpu_reference(&raw_path);
    let mut cpu_seeded = cpu.clone();
    let mut prev_y: Vec<u8> = Vec::new();
    temporal_denoise_luma_yuyv(&mut cpu_seeded, &mut prev_y, OUT_W, OUT_H, 0.6, 12.0);
    assert_eq!(cpu_seeded, cpu, "CPU temporal seed frame must be a no-op");

    let mut proc = match GpuProcessor::new(8, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gpu: skipped (no usable Vulkan device: {e})");
            return;
        }
    };
    // First GPU frame with temporal on: reset path -> Y left as captured.
    let gpu = proc
        .process(
            &bytes,
            DenoiseParams {
                chroma_radius: 0,
                chroma_strength: 0.0,
                temporal_alpha: 0.6,
                temporal_motion: 12.0,
            },
        )
        .expect("gpu process")
        .to_vec();

    let (max_diff, frac) = diff_stats(&gpu, &cpu);
    eprintln!("gpu temporal-seed vs cpu MHC: max_diff={max_diff}, fraction |diff|>2 = {frac:.5}");
    assert!(
        max_diff <= 4 && frac < 0.01,
        "GPU temporal seed frame altered luma: max_diff={max_diff}, frac>2 = {frac:.5}"
    );
}

/// The AWB measurement runs on a background thread and lands a few frames
/// later. On a repeated frame the estimate it brings is the one the first frame
/// waited for, so every later output must equal the first exactly; a stale,
/// lost or mis-applied measurement would change the colour.
#[test]
fn gpu_background_awb_keeps_a_steady_frame_steady() {
    let _gpu = gpu_lock();
    let raw_path = data("sample-raw.bin");
    if !raw_path.exists() {
        eprintln!("gpu: skipped (no local sample-raw.bin in tests/data/)");
        return;
    }
    let bytes = std::fs::read(&raw_path).expect("raw");
    let mut proc = match GpuProcessor::new(4, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("gpu: skipped (no usable Vulkan device: {e})");
            return;
        }
    };
    let first = proc
        .process(&bytes, DenoiseParams::default())
        .expect("gpu process")
        .to_vec();
    // Feed frames until three background measurements have been applied; the
    // worker's speed depends on the build profile, so bound by time, not count.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let (mut applied, mut n) = (0u32, 0u32);
    while applied < 3 {
        assert!(
            std::time::Instant::now() < deadline,
            "background AWB applied only {applied} times in {n} frames"
        );
        n += 1;
        let out = proc
            .process(&bytes, DenoiseParams::default())
            .expect("gpu process");
        assert!(out == &first[..], "frame {n} differs from the first");
        applied += proc.timing().estimate.is_some() as u32;
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    eprintln!("background AWB: {applied} measurements applied over {n} frames");
}
