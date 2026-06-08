//! Live colour webcam: capture raw via libcamera, run the tuned ISP, drive
//! auto-exposure on the sensor, and publish processed YUYV frames to a
//! v4l2loopback device that any application can open.
//!
//! Auto-exposure meters the produced frame's mean luminance in linear light
//! (the output luma with the sRGB gamma inverted) toward a linear mid-grey
//! target, so the exposure model — which treats brightness as proportional to
//! the exposure-gain product — operates on a linear quantity. This tracks the
//! brightness the viewer sees while keeping the control loop's per-frame
//! correction the right magnitude (see [`mean_linear_luma`]).
//!
//! Defaults to full-res MHC on 8 threads (sustains the sensor's 30 fps); use
//! `--debayer half` and/or fewer `--threads` to trade quality for lower CPU use.
//! Stale frames are dropped so latency stays low when the ISP falls behind.
//!
//! With `--backend gpu` (requires the `gpu` build feature) the per-pixel stages
//! run as Vulkan compute shaders on the iGPU and AWB stays on the CPU; this
//! offloads the CPU and always produces full-res 1920x1080 (the `--debayer`
//! option is ignored on the GPU path).
//!
//! With `--measure-delay` the binary runs a one-off calibration instead of the
//! live loop: on a static scene it steps exposure and gain and reports how many
//! frames the sensor takes to apply each change (used to tune the AE settle
//! count and to fill libcamera's sensor-delays database). No loopback is opened.
//!
//! By default the camera and ISP run on demand: the binary keeps the loopback
//! open (so the virtual webcam is visible) but opens the sensor and processes
//! frames only while an application is capturing, driven by the v4l2loopback
//! client-usage V4L2 event. Where that event is unavailable it streams always-on.
//! `--on-demand off` forces always-on; `--on-demand on` requires the event.
//!
//! Usage:
//!   gc2607-video [--device /dev/videoN] [--backend cpu|gpu] [--debayer half|mhc]
//!                [--no-ae] [--target <0..1>] [--max-gain <idx>] [--threads <n>]
//!                [--on-demand auto|on|off] [--measure-delay]

use std::io;
use std::time::{Duration, Instant};

use libcamera::{
    camera::ActiveCamera,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::FrameBuffer,
    framebuffer_map::MemoryMappedFrameBuffer,
    request::{Request, ReuseFlag},
    stream::Stream,
};

use gc2607_isp::ae::{self, AeConfig, AeState};
use gc2607_isp::camera as cam_setup;
use gc2607_isp::output::{
    denoise_chroma_yuyv, rgb_to_yuyv_crop, temporal_denoise_luma_yuyv, LoopbackOutput,
};
use gc2607_isp::pipeline::{DebayerMode, Processor};
use gc2607_isp::sensor::Sensor;

#[cfg(feature = "gpu")]
use gc2607_isp::gpu::GpuProcessor;

/// Requested ISP backend. `Auto` (the default) prefers the GPU (Vulkan compute,
/// full-res MHC) and falls back to the CPU if no GPU is available; `Cpu` forces
/// the CPU path (rayon, half|mhc); `Gpu` forces the GPU and fails if it cannot
/// be initialised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Backend {
    Auto,
    Cpu,
    Gpu,
}

/// On-demand camera gating. `Auto` (the default) subscribes to the v4l2loopback
/// client-usage event and runs the camera + ISP only while a consumer is
/// capturing, falling back silently to always-on when the event is unavailable
/// (a v4l2loopback without the patch); `On` is the same but logs the fallback;
/// `Off` always streams from start-up (the original behaviour).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OnDemandMode {
    Auto,
    On,
    Off,
}

/// Either ISP engine, producing a packed YUYV frame from raw capture bytes. The
/// CPU engine renders RGB then packs (with the centre crop); the GPU engine
/// packs YUYV directly in the shader.
// Both engine states are large (the `Processor` holds the CFA/RGB scratch
// buffers; the `GpuProcessor` owns the wgpu device, queue, pipelines and
// buffers), so both are boxed to keep `Engine` itself pointer-sized.
enum Engine {
    Cpu {
        proc: Box<Processor>,
        yuyv: Vec<u8>,
        dst_w: usize,
        dst_h: usize,
    },
    #[cfg(feature = "gpu")]
    Gpu(Box<GpuProcessor>),
}

impl Engine {
    fn process(&mut self, buf: &[u8]) -> io::Result<&[u8]> {
        match self {
            Engine::Cpu { proc, yuyv, dst_w, dst_h } => {
                let (w, h, rgb) = proc.process(buf)?;
                rgb_to_yuyv_crop(rgb, w, h, yuyv, *dst_w, *dst_h);
                Ok(yuyv)
            }
            #[cfg(feature = "gpu")]
            Engine::Gpu(p) => p.process(buf),
        }
    }
}

// Native sensor geometry (single source of truth in `raw`).
const WIDTH: u32 = gc2607_isp::raw::W as u32;
const HEIGHT: u32 = gc2607_isp::raw::H as u32;

/// Re-estimate white balance / CCM every N frames. The scene's colour
/// temperature is stable between estimates, so the AWB sort (the heaviest serial
/// step) runs only occasionally; AE brightness is still metered every frame.
const AWB_INTERVAL: u64 = 8;

/// Default AE target, in *linear* light (0..1). AE meters the produced frame's
/// mean luminance with the sRGB gamma inverted (see [`mean_linear_luma`]), so the
/// target is a linear level, not a gamma-encoded one. 0.15 corresponds to a
/// gamma-encoded mid-grey of `srgb(0.15) ≈ 0.43` on the output scale (near the
/// photographic 18%-grey reference, `srgb(0.18) ≈ 0.46`); it preserves the
/// previously tuned output brightness while letting the linear exposure model
/// drive the loop. Override with `--target` (also interpreted as linear).
const AE_TARGET_LINEAR: f64 = 0.15;

/// Interval between standby frames written while on-demand idle (~10 fps). Just
/// frequent enough to keep the loopback a negotiable capture device for a
/// connecting consumer; the hardware camera stays off, so this is a memset+write
/// only. Lower is more responsive to a new consumer; higher means fewer wake-ups.
const STANDBY_PERIOD: Duration = Duration::from_millis(100);

struct Args {
    device: String,
    backend: Backend,
    mode: DebayerMode,
    ae: bool,
    target: f64,
    max_gain: u8,
    threads: usize,
    lca: bool,
    measure: bool,
    denoise: f64,
    temporal: f64,
    on_demand: OnDemandMode,
}

fn parse_args() -> Args {
    let mut device = "/dev/video0".to_string();
    // Default backend is Auto: prefer the GPU (full-res MHC, ~4x lower CPU) and
    // fall back to the CPU path if no GPU is available. The CPU defaults (#24)
    // are full-res MHC at 8 threads, overridable via --debayer / --threads.
    let mut backend = Backend::Auto;
    let mut mode = DebayerMode::Mhc;
    let mut ae = true;
    let mut target = AE_TARGET_LINEAR;
    let mut max_gain = AeConfig::default().max_gain_index;
    let mut threads = 8usize;
    // Lateral chromatic aberration correction (full-res MHC only); on by default.
    let mut lca = true;
    // Sensor apply-delay measurement mode (no loopback, no ISP output).
    let mut measure = false;
    // Gain-adaptive chroma-denoise strength scaler (0 disables; 1 is the tuned
    // default; >1 strengthens). See `chroma_denoise_for_gain`.
    let mut denoise = 1.0f64;
    // Gain-adaptive temporal luma-denoise scaler (0 disables; 1 is the default;
    // >1 strengthens). See `temporal_luma_for_gain`.
    let mut temporal = 1.0f64;
    // On-demand camera gating: open the camera only while a consumer captures.
    // Auto by default (uses the v4l2loopback client-usage event when available).
    let mut on_demand = OnDemandMode::Auto;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--device" => {
                if let Some(v) = it.next() {
                    device = v;
                }
            }
            "--backend" => match it.next().as_deref() {
                Some("auto") => backend = Backend::Auto,
                Some("cpu") => backend = Backend::Cpu,
                Some("gpu") => backend = Backend::Gpu,
                other => {
                    eprintln!("unknown backend: {other:?} (use auto|cpu|gpu)");
                    std::process::exit(1);
                }
            },
            "--debayer" => match it.next().as_deref() {
                Some("half") => mode = DebayerMode::HalfRes,
                Some("mhc") => mode = DebayerMode::Mhc,
                other => {
                    eprintln!("unknown debayer mode: {other:?} (use half|mhc)");
                    std::process::exit(1);
                }
            },
            "--no-ae" => ae = false,
            "--no-lca" => lca = false,
            "--no-denoise" => denoise = 0.0,
            "--denoise" => denoise = it.next().and_then(|s| s.parse().ok()).unwrap_or(denoise).max(0.0),
            "--no-temporal" => temporal = 0.0,
            "--temporal" => temporal = it.next().and_then(|s| s.parse().ok()).unwrap_or(temporal).max(0.0),
            "--measure-delay" => measure = true,
            "--on-demand" => match it.next().as_deref() {
                Some("auto") => on_demand = OnDemandMode::Auto,
                Some("on") => on_demand = OnDemandMode::On,
                Some("off") => on_demand = OnDemandMode::Off,
                other => {
                    eprintln!("unknown on-demand mode: {other:?} (use auto|on|off)");
                    std::process::exit(1);
                }
            },
            "--target" => target = it.next().and_then(|s| s.parse().ok()).unwrap_or(target),
            "--max-gain" => max_gain = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_gain),
            "--threads" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(threads).max(1),
            "-h" | "--help" => {
                eprintln!(
                    "usage: gc2607-video [--device /dev/videoN] [--backend auto|cpu|gpu] \
                     [--debayer half|mhc] [--no-ae] [--no-lca] [--no-denoise] \
                     [--denoise <scale>] [--no-temporal] [--temporal <scale>] \
                     [--target <0..1>] [--max-gain <idx>] [--threads <n>] \
                     [--on-demand auto|on|off] [--measure-delay]"
                );
                std::process::exit(0);
            }
            _ => {
                eprintln!("unexpected argument: {a}");
                std::process::exit(1);
            }
        }
    }
    Args {
        device,
        backend,
        mode,
        ae,
        target,
        max_gain,
        threads,
        lca,
        measure,
        denoise,
        temporal,
        on_demand,
    }
}

/// Chroma-denoise parameters (box-blur radius on the chroma grid, blend strength
/// 0..1) for an analogue-gain LUT index, multiplied by the `--denoise` scaler.
///
/// Noise grows with analogue gain, so denoise only kicks in as the AE raises
/// gain in dim light and stays off at low gain to preserve fidelity. The table
/// is keyed by the gain LUT index (0..16, which doubles every four steps).
fn chroma_denoise_for_gain(gain_index: u8, scale: f64) -> (usize, f64) {
    let (radius, base) = match gain_index {
        0..=3 => (0usize, 0.0f64), // ~1x..1.7x gain: clean enough, no denoise
        4..=7 => (1, 0.5),         // ~2x..3x
        8..=11 => (2, 0.7),        // ~4x..7x
        _ => (3, 0.85),            // ~8x..16x
    };
    (radius, (base * scale).clamp(0.0, 1.0))
}

/// Maximum temporal-blend weight (alpha, 0..1) for an analogue-gain LUT index,
/// multiplied by the `--temporal` scaler. Off at low gain; ramps up with gain.
fn temporal_luma_for_gain(gain_index: u8, scale: f64) -> f64 {
    if scale <= 0.0 {
        return 0.0;
    }
    let base = match gain_index {
        0..=3 => 0.0f64,
        4..=7 => 0.40,
        8..=11 => 0.55,
        _ => 0.70,
    };
    (base * scale).clamp(0.0, 0.95)
}

/// Motion gate threshold (luma codes) for temporal denoise: above this per-pixel
/// frame-to-frame change the blend fades out to avoid ghosting on motion.
const TEMPORAL_MOTION: f64 = 12.0;

/// Apply gain-adaptive denoise to a freshly produced YUYV frame.
///
/// Chroma denoise is spatial, luma denoise is temporal; both ramp with analogue
/// gain and are no-ops at low gain (the well-lit common case). When neither is
/// active the temporal history is dropped so it re-seeds cleanly (no ghosting)
/// when gain rises again, and the caller writes the original `yuyv` straight
/// through. When at least one is active the result is written into `scratch`.
///
/// Returns `(chroma_radius, temporal_alpha, used_scratch)`: the first two are for
/// telemetry, and `used_scratch` is `true` when the denoised frame is in
/// `scratch` (else the caller uses `yuyv` unchanged).
fn apply_denoise(
    yuyv: &[u8],
    dst_w: usize,
    dst_h: usize,
    gain_index: u8,
    args: &Args,
    scratch: &mut Vec<u8>,
    prev_y: &mut Vec<u8>,
) -> (usize, f64, bool) {
    let (dr, ds) = chroma_denoise_for_gain(gain_index, args.denoise);
    let ta = temporal_luma_for_gain(gain_index, args.temporal);
    let chroma_on = dr > 0 && ds > 0.0;
    let temporal_on = ta > 0.0;
    if !chroma_on && !temporal_on {
        prev_y.clear();
        return (dr, ta, false);
    }
    scratch.clear();
    scratch.extend_from_slice(yuyv);
    if chroma_on {
        denoise_chroma_yuyv(scratch, dst_w, dst_h, dr, ds);
    }
    if temporal_on {
        temporal_denoise_luma_yuyv(scratch, prev_y, dst_w, dst_h, ta, TEMPORAL_MOTION);
    }
    (dr, ta, true)
}

/// Output (cropped) size for a debayer mode: a standard 16:9 size centred in
/// the slightly larger ISP output.
fn output_size(mode: DebayerMode) -> (usize, usize) {
    match mode {
        DebayerMode::Mhc => (1920, 1080),  // from 1928x1088
        DebayerMode::HalfRes => (960, 540), // from 964x544
    }
}

/// Inverse-sRGB lookup table: gamma-encoded luma byte (0..=255) -> linear light
/// (0..1). Built once. The AE metric maps each sampled Y byte through this table
/// (a load, no `powf`) so metering stays as cheap as a plain mean while yielding
/// a linear quantity.
fn srgb_to_linear_lut() -> &'static [f64; 256] {
    use std::sync::OnceLock;
    static LUT: OnceLock<[f64; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f64; 256];
        for (b, e) in t.iter_mut().enumerate() {
            *e = gc2607_isp::pipeline::srgb_to_linear(b as f64 / 255.0);
        }
        t
    })
}

/// Mean *linear* luminance of a packed YUYV frame, as a fraction of full scale
/// (0..1). Each sampled Y byte is mapped back through the inverse sRGB EOTF
/// before averaging, so the result is mean scene luminance in linear light, not
/// the gamma-encoded mean.
///
/// This is the AE control variable. The exposure model in [`ae::step`] assumes
/// brightness is proportional to the exposure-gain product, which holds for a
/// linear metric but not for gamma-encoded luma; metering in linear light makes
/// each damped correction the right magnitude and keeps convergence independent
/// of the scene. Averaging the gamma-encoded byte then inverting (as opposed to
/// inverting per pixel) would bias the metric toward shadows, so the inversion
/// is applied per sample via the LUT.
///
/// The Y bytes sit at even offsets; this subsamples every fourth pixel, which is
/// ample for an exposure metric and keeps the per-frame cost negligible. Note
/// the inverse is applied to the BT.601 luma byte rather than to R, G, B
/// separately (the YUYV frame carries only luma); for a scalar AE metric this is
/// a standard, monotonic approximation of scene luminance.
fn mean_linear_luma(yuyv: &[u8]) -> f64 {
    let lut = srgb_to_linear_lut();
    let mut sum = 0f64;
    let mut n = 0u64;
    let mut i = 0;
    while i < yuyv.len() {
        sum += lut[yuyv[i] as usize]; // Y of every fourth pixel (2 bytes/px, step 8)
        n += 1;
        i += 8;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

/// Resolve the requested backend into a concrete engine, returning it with its
/// output size. `Auto` tries the GPU first and falls back to the CPU; `Gpu`
/// exits if the GPU cannot be initialised (or the feature is not compiled).
fn build_engine(args: &Args) -> (Engine, usize, usize) {
    #[cfg(feature = "gpu")]
    if matches!(args.backend, Backend::Auto | Backend::Gpu) {
        match GpuProcessor::new(AWB_INTERVAL, args.lca) {
            Ok(p) => {
                if args.mode == DebayerMode::HalfRes {
                    eprintln!("note: GPU backend is always full-res MHC (--debayer half ignored)");
                }
                let (w, h) = p.out_dims();
                println!("isp backend: gpu (full-res MHC, lca={})", args.lca);
                return (Engine::Gpu(Box::new(p)), w, h);
            }
            Err(e) => {
                if args.backend == Backend::Gpu {
                    eprintln!("GPU backend requested but unavailable: {e}");
                    std::process::exit(1);
                }
                eprintln!("note: GPU backend unavailable ({e}); falling back to CPU");
            }
        }
    }

    #[cfg(not(feature = "gpu"))]
    if args.backend == Backend::Gpu {
        eprintln!("this binary was built without the `gpu` feature; rebuild with --features capture,gpu");
        std::process::exit(1);
    }

    // CPU backend (forced, or the Auto fallback).
    let (dst_w, dst_h) = output_size(args.mode);
    println!(
        "isp backend: cpu ({:?}, {} threads, lca={})",
        args.mode, args.threads, args.lca
    );
    (
        Engine::Cpu {
            proc: Box::new(Processor::new(args.mode, AWB_INTERVAL, args.lca)),
            yuyv: vec![0u8; dst_w * dst_h * 2],
            dst_w,
            dst_h,
        },
        dst_w,
        dst_h,
    )
}

/// Borrow the raw capture bytes from a completed request's first plane, or
/// `None` if the buffer, metadata, or plane is missing or `bytes_used` exceeds
/// the mapped plane. A malformed or short frame is then skipped rather than
/// panicking, so the long-running daemon keeps streaming.
fn frame_bytes<'a>(req: &'a Request, stream: &Stream) -> Option<&'a [u8]> {
    let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(stream)?;
    // The plane reference borrows the mmap (tied to `req`), so copy it out of the
    // temporary Vec returned by `data()` before that Vec is dropped.
    let plane: &[u8] = fb.data().first()?;
    // `planes()` returns a libcamera wrapper (not a slice), so index with get(0).
    let bytes_used = fb.metadata()?.planes().get(0)?.bytes_used as usize;
    Some(&plane[..bytes_used.min(plane.len())])
}

/// Read the next completed request in strict capture order (no frame dropping)
/// and return its raw-mean brightness metric, requeuing the buffer. Returns
/// `None` only on capture timeout. `--measure-delay` must observe every sensor
/// frame in sequence, so unlike the live loop it never drains to the freshest.
fn next_in_order_mean(
    cam: &mut ActiveCamera,
    stream: &Stream,
    rx: &std::sync::mpsc::Receiver<Request>,
) -> Option<f64> {
    let mut req = rx.recv_timeout(Duration::from_secs(5)).ok()?;
    let mean = frame_bytes(&req, stream).and_then(gc2607_isp::raw::mean_norm_from_bytes);
    req.reuse(ReuseFlag::REUSE_BUFFERS);
    cam.queue_request(req).map_err(|(_, e)| e).ok()?;
    Some(mean.unwrap_or(f64::NAN))
}

/// Drop every request currently waiting in the channel (requeuing its buffer)
/// to shrink the in-flight pipeline before arming a step change.
fn flush_pending(cam: &mut ActiveCamera, rx: &std::sync::mpsc::Receiver<Request>) {
    while let Ok(mut req) = rx.try_recv() {
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        let _ = cam.queue_request(req).map_err(|(_, e)| e);
    }
}

/// Read `n` frames in order and return the last frame's mean (None on timeout).
fn settle_mean(
    cam: &mut ActiveCamera,
    stream: &Stream,
    rx: &std::sync::mpsc::Receiver<Request>,
    n: u32,
) -> Option<f64> {
    let mut last = f64::NAN;
    for _ in 0..n {
        last = next_in_order_mean(cam, stream, rx)?;
    }
    Some(last)
}

/// Record the means of the next `n` frames in order (stops early on timeout).
fn record_means(
    cam: &mut ActiveCamera,
    stream: &Stream,
    rx: &std::sync::mpsc::Receiver<Request>,
    n: u32,
) -> Vec<f64> {
    let mut v = Vec::with_capacity(n as usize);
    for _ in 0..n {
        match next_in_order_mean(cam, stream, rx) {
            Some(m) => v.push(m),
            None => break,
        }
    }
    v
}

/// Settled level after a step: the mean of the last few recorded frames.
fn plateau_of(means: &[f64]) -> f64 {
    let tail: Vec<f64> = means.iter().rev().take(4).copied().filter(|m| m.is_finite()).collect();
    if tail.is_empty() {
        f64::NAN
    } else {
        tail.iter().sum::<f64>() / tail.len() as f64
    }
}

/// First frame (1-based) whose mean crosses halfway between `base` and
/// `plateau` — the observed apply delay for a rising step. None if never seen.
fn detect_delay(base: f64, plateau: f64, means: &[f64]) -> Option<usize> {
    if !base.is_finite() || !plateau.is_finite() || plateau <= base {
        return None;
    }
    let thr = base + 0.5 * (plateau - base);
    means
        .iter()
        .position(|&m| m.is_finite() && m >= thr)
        .map(|i| i + 1)
}

/// Print a step's per-frame means, marking the detected delay frame.
fn report_step(header: &str, base: f64, plateau: f64, means: &[f64], delay: Option<usize>) {
    println!("{header}: base mean {base:.3}, settled {plateau:.3}");
    for (i, m) in means.iter().enumerate() {
        let marker = if delay == Some(i + 1) { "  <- delay" } else { "" };
        println!("  +{:>2}: mean {m:.3}{marker}", i + 1);
    }
    println!();
}

/// Print one named delay result.
fn print_delay(name: &str, delay: Option<usize>) {
    match delay {
        Some(d) => println!("{name:>9} delay: {d} frames"),
        None => println!("{name:>9} delay: not detected"),
    }
}

/// Measure the sensor's exposure and gain apply latency on a static scene.
///
/// AE is off. The routine calibrates a unity-gain exposure for a mid-range
/// image, then applies a step change to exposure (and separately to gain) and
/// records the per-frame mean, in capture order with no frame dropping. The
/// delay is the number of frames between the register write and the first frame
/// that reflects it. The figure is *observed* end-to-end: it includes the
/// capture pipeline depth, so it is the right basis for the live loop's
/// `AE_SETTLE` (which reads frames the same way). The sensor-internal latch
/// delay stored in libcamera's database is usually smaller.
fn run_measure_delay(
    cam: &mut ActiveCamera,
    stream: &Stream,
    rx: &std::sync::mpsc::Receiver<Request>,
    sensor: &Sensor,
    mut state: AeState,
) {
    const CAL_TARGET: f64 = 0.30;
    const CAL_ITERS: u32 = 12;
    const SETTLE: u32 = 24; // ~0.8 s at 30 fps: well past any apply delay
    const WINDOW: u32 = 16; // frames observed after each step

    println!("measure-delay: keep the camera on a static, evenly-lit scene\n");

    let vb = ae::VBLANK_MIN;
    state.vblank = vb;
    state.gain_index = 0;
    state.exposure = state.exposure.clamp(ae::EXPOSURE_MIN, ae::exposure_max(vb));
    let _ = sensor.apply(state);

    // Calibrate unity-gain exposure to ~CAL_TARGET so neither step clips.
    let mut cal_mean = f64::NAN;
    for _ in 0..CAL_ITERS {
        cal_mean = match settle_mean(cam, stream, rx, SETTLE) {
            Some(m) => m,
            None => {
                eprintln!("capture timed out during calibration");
                return;
            }
        };
        if (0.27..=0.34).contains(&cal_mean) {
            break;
        }
        let factor = (CAL_TARGET / cal_mean.max(1e-4)).clamp(0.25, 4.0);
        state.exposure = ((state.exposure as f64 * factor).round() as i32)
            .clamp(ae::EXPOSURE_MIN, ae::exposure_max(vb));
        let _ = sensor.apply(state);
    }
    let e_star = state.exposure;
    println!("calibrated: exposure={e_star} lines, gain_idx=0, mean~{cal_mean:.3}\n");

    // --- Exposure step: e_lo -> e_hi (~3x) at unity gain. ---
    let e_lo = (e_star / 3).max(ae::EXPOSURE_MIN);
    let e_hi = e_star;
    state.exposure = e_lo;
    state.gain_index = 0;
    let _ = sensor.apply(state);
    let exp_base = match settle_mean(cam, stream, rx, SETTLE) {
        Some(m) => m,
        None => return,
    };
    flush_pending(cam, rx);
    state.exposure = e_hi;
    let _ = sensor.apply(state);
    let exp_means = record_means(cam, stream, rx, WINDOW);
    let exp_plateau = plateau_of(&exp_means);
    let exp_delay = detect_delay(exp_base, exp_plateau, &exp_means);
    report_step(
        &format!("exposure step {e_lo}->{e_hi} lines"),
        exp_base,
        exp_plateau,
        &exp_means,
        exp_delay,
    );

    // --- Gain step: index 0 -> 4 (~2x) at exposure e_star/2. ---
    let e_g = (e_star / 2).max(ae::EXPOSURE_MIN);
    state.exposure = e_g;
    state.gain_index = 0;
    let _ = sensor.apply(state);
    let gain_base = match settle_mean(cam, stream, rx, SETTLE) {
        Some(m) => m,
        None => return,
    };
    flush_pending(cam, rx);
    state.gain_index = 4;
    let _ = sensor.apply(state);
    let gain_means = record_means(cam, stream, rx, WINDOW);
    let gain_plateau = plateau_of(&gain_means);
    let gain_delay = detect_delay(gain_base, gain_plateau, &gain_means);
    report_step(
        "gain step idx 0->4 (~2x)",
        gain_base,
        gain_plateau,
        &gain_means,
        gain_delay,
    );

    // --- Summary. ---
    println!("=== apply-delay summary (observed, includes capture pipeline depth) ===");
    print_delay("exposure", exp_delay);
    print_delay("gain", gain_delay);
    match (exp_delay, gain_delay) {
        (Some(a), Some(b)) => println!("suggested AE_SETTLE = {} frames (max of the two)", a.max(b)),
        _ => println!("could not auto-detect one or both delays; read the per-frame tables above"),
    }
    println!(
        "note: libcamera's sensor-delays DB wants the sensor-internal latch delay\n      \
         (commonly 2 frames for exposure and gain); the figure above is end-to-end."
    );
}

/// Open the sensor sub-device (if present) and seed the AE state from whatever
/// exposure/gain/vblank it currently holds. AE is disabled if no sensor control
/// is available (the ISP still runs on the current sensor state).
fn init_sensor() -> (Option<Sensor>, AeState) {
    let sensor = match Sensor::open_gc2607() {
        Ok(s) => {
            println!("sensor: {}", s.path().display());
            Some(s)
        }
        Err(e) => {
            eprintln!("warning: no sensor control, AE disabled ({e})");
            None
        }
    };

    let mut state = AeState::default();
    if let Some(s) = &sensor {
        state.exposure = s.exposure().unwrap_or(state.exposure);
        state.gain_index = s.analogue_gain().unwrap_or(0).clamp(0, ae::MAX_GAIN_INDEX as i32) as u8;
        state.vblank = s.vblank().unwrap_or(state.vblank);
    }
    (sensor, state)
}

fn main() {
    let args = parse_args();

    // Bound the ISP thread pool to the requested count (the CPU budget is the
    // user's to set). The GPU backend does its pixel work on the iGPU, so the
    // pool is only used by the occasional CPU-side AWB estimate.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .expect("build rayon pool");

    let (sensor, mut state) = init_sensor();
    let mgr = CameraManager::new().expect("CameraManager");

    // --- Sensor apply-delay measurement mode (no loopback, no ISP output). ---
    if args.measure {
        let mut session = cam_setup::open_raw(&mgr, WIDTH, HEIGHT)
            .unwrap_or_else(|e| panic!("open raw camera: {e}"));
        session.cam.start(None).expect("start");
        if let Some(s) = &sensor {
            let _ = s.apply(state);
        }
        queue_initial(&mut session);
        match &sensor {
            Some(s) => run_measure_delay(&mut session.cam, &session.stream, &session.rx, s, state),
            None => eprintln!("--measure-delay needs sensor control, but none is available"),
        }
        return;
    }

    // Build the ISP engine once (GPU init etc.) and open the loopback up front,
    // before choosing on-demand vs always-on: the loopback must stay open so the
    // virtual webcam is visible to applications while the camera is idle.
    let (mut engine, dst_w, dst_h) = build_engine(&args);
    let mut out = LoopbackOutput::open(&args.device, dst_w as u32, dst_h as u32)
        .unwrap_or_else(|e| panic!("open loopback {}: {e}", args.device));
    println!("output: {} {dst_w}x{dst_h} YUYV", args.device);

    // On-demand gating: subscribe to the v4l2loopback client-usage event so the
    // camera runs only while a consumer captures. Unavailable (un-patched
    // v4l2loopback) falls back to always-on.
    let on_demand = match args.on_demand {
        OnDemandMode::Off => false,
        mode => match out.subscribe_client_usage() {
            Ok(()) => {
                println!("on-demand: camera runs only while a consumer captures (client-usage event)");
                true
            }
            Err(e) => {
                if mode == OnDemandMode::On {
                    eprintln!(
                        "on-demand requested but the v4l2loopback client-usage event is \
                         unavailable ({e}); falling back to always-on"
                    );
                } else {
                    eprintln!("note: v4l2loopback client-usage event unavailable ({e}); running always-on");
                }
                false
            }
        },
    };

    if on_demand {
        run_on_demand(&mgr, &mut out, dst_w, dst_h, &mut engine, &sensor, &mut state, &args);
    } else {
        // Always-on: open and start the camera once, stream until a fatal error.
        let mut session = cam_setup::open_raw(&mgr, WIDTH, HEIGHT)
            .unwrap_or_else(|e| panic!("open raw camera: {e}"));
        session.cam.start(None).expect("start");
        if let Some(s) = &sensor {
            let _ = s.apply(state);
        }
        queue_initial(&mut session);
        run_live(
            &mut engine, &mut out, &mut session.cam, &session.stream, &session.rx,
            dst_w, dst_h, &sensor, &mut state, &args, false,
        );
    }
}

/// Queue a freshly started session's initial requests, panicking on the first
/// failure (a new session must accept its own buffers; a failure here means a
/// broken setup, not a transient runtime condition).
fn queue_initial(session: &mut cam_setup::Session<'_>) {
    for req in std::mem::take(&mut session.requests) {
        session
            .cam
            .queue_request(req)
            .map_err(|(_, e)| e)
            .expect("queue initial request");
    }
}

/// Outcome of processing and writing one frame (see `process_and_write`).
enum FrameOutcome {
    /// Written to the loopback. Carries the output mean linear luminance (the AE
    /// control variable; see [`mean_linear_luma`]) and the applied denoise
    /// strengths (chroma radius, temporal alpha).
    Written { luma: f64, chroma_radius: usize, temporal_alpha: f64 },
    /// The ISP/GPU returned an error or the frame work panicked; the frame is
    /// skipped. Carries a human-readable reason for the throttled log.
    Skipped(String),
    /// Writing to the loopback failed; the caller should stop the stream.
    WriteFailed(io::Error),
}

/// Run the ISP on one raw frame, apply gain-adaptive denoise, and write the
/// result to the loopback. The whole frame's work runs inside `catch_unwind` so
/// an unexpected panic (e.g. an out-of-range index in a pixel stage) skips that
/// one frame instead of killing the long-running daemon; the panic message is
/// still printed by the default hook. ISP errors and write failures are returned
/// as variants, not panics.
#[allow(clippy::too_many_arguments)]
fn process_and_write(
    engine: &mut Engine,
    out: &mut LoopbackOutput,
    buf: &[u8],
    dst_w: usize,
    dst_h: usize,
    gain_index: u8,
    args: &Args,
    denoise_buf: &mut Vec<u8>,
    prev_y: &mut Vec<u8>,
) -> FrameOutcome {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let yuyv = match engine.process(buf) {
            Ok(y) => y,
            Err(e) => return FrameOutcome::Skipped(e.to_string()),
        };
        let luma = mean_linear_luma(yuyv);
        let (chroma_radius, temporal_alpha, used) =
            apply_denoise(yuyv, dst_w, dst_h, gain_index, args, denoise_buf, prev_y);
        let frame: &[u8] = if used { denoise_buf.as_slice() } else { yuyv };
        match out.write_frame(frame) {
            Ok(()) => FrameOutcome::Written { luma, chroma_radius, temporal_alpha },
            Err(e) => FrameOutcome::WriteFailed(e),
        }
    }));
    result.unwrap_or_else(|_| FrameOutcome::Skipped("panic in frame processing".to_string()))
}

/// Why the live loop returned (see `run_live`).
enum LiveExit {
    /// On-demand: the last consumer disconnected. The caller should stop the
    /// camera and wait for the next consumer.
    Idle,
    /// A fatal condition (camera timeout or loopback write failure). The caller
    /// should stop the daemon.
    Stop,
}

/// Live path: run the capture/process/AE loop on an already-started `cam`,
/// writing to an already-open `out`. Drops stale frames so latency stays low
/// when the ISP falls behind, meters output luma for AE, and applies
/// gain-adaptive denoise to the produced frame.
///
/// When `watch_consumer` is set (on-demand mode) the loop also drains the
/// v4l2loopback client-usage event each iteration and returns [`LiveExit::Idle`]
/// as soon as the last consumer disconnects; otherwise it runs until a fatal
/// condition and returns [`LiveExit::Stop`]. The engine, loopback and AE state
/// are owned by the caller so they persist across on-demand activations.
#[allow(clippy::too_many_arguments)]
fn run_live(
    engine: &mut Engine,
    out: &mut LoopbackOutput,
    cam: &mut ActiveCamera,
    stream: &Stream,
    rx: &std::sync::mpsc::Receiver<Request>,
    dst_w: usize,
    dst_h: usize,
    sensor: &Option<Sensor>,
    state: &mut AeState,
    args: &Args,
    watch_consumer: bool,
) -> LiveExit {
    let ae_cfg = AeConfig {
        target: args.target,
        max_gain_index: args.max_gain,
        ..AeConfig::default()
    };

    let mut frames = 0u64;
    let mut dropped = 0u64;
    // Frames skipped because the ISP/GPU returned an error or the buffer was
    // malformed, and sensor-apply failures: counted so a systematic fault is
    // visible in the periodic report instead of being silently swallowed.
    let mut errors = 0u64;
    let mut ae_errors = 0u64;
    let mut last_report = Instant::now();
    let mut report_frames = 0u64;
    // Last metered output mean linear luminance (0..1), for the telemetry line.
    let mut last_luma = 0f64;
    // Scratch for the denoise post-pass (only filled when a denoise stage runs),
    // the temporal-denoise luma history, and the last applied strengths (for
    // telemetry).
    let mut denoise_buf: Vec<u8> = Vec::new();
    let mut prev_y: Vec<u8> = Vec::new();
    let mut last_dn = 0usize;
    let mut last_ta = 0f64;

    // AE settling: the sensor applies a new exposure/gain with a delay. If we
    // re-meter every frame we issue several corrections before the first takes
    // effect, so the loop double-corrects and oscillates bright/dark at a
    // brightness boundary. After committing a change, hold metering for
    // AE_SETTLE frames so the change is reflected before the next decision.
    //
    // `--measure-delay` measured the observed end-to-end apply delay (capture
    // pipeline included) at 2 frames for both exposure and gain on this part;
    // AE_SETTLE keeps one frame of margin on top.
    const AE_SETTLE: u32 = 3;
    let mut ae_hold = 0u32;

    let exit = loop {
        // On-demand: stop as soon as the last consumer disconnects. The event
        // queue is drained non-blocking each iteration (~per frame), so the
        // shutdown latency is at most one frame.
        if watch_consumer {
            let mut left = false;
            loop {
                match out.poll_client_usage() {
                    Ok(Some(active)) => left |= !active,
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("client-usage event poll failed: {e}");
                        break;
                    }
                }
            }
            if left {
                println!("on-demand: consumer disconnected");
                break LiveExit::Idle;
            }
        }

        // Block for one completed request, then drain any extra ready ones,
        // requeuing all but the freshest so we always process the newest frame.
        let mut req = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("camera timed out, stopping");
                break LiveExit::Stop;
            }
        };
        while let Ok(newer) = rx.try_recv() {
            req.reuse(ReuseFlag::REUSE_BUFFERS);
            if let Err(e) = cam.queue_request(req).map_err(|(_, e)| e) {
                eprintln!("requeue (drain) failed: {e}");
            }
            req = newer;
            dropped += 1;
        }

        let buf = match frame_bytes(&req, stream) {
            Some(b) => b,
            None => {
                // Malformed/short frame: skip it, requeue the buffer, keep going.
                errors += 1;
                req.reuse(ReuseFlag::REUSE_BUFFERS);
                if let Err(e) = cam.queue_request(req).map_err(|(_, e)| e) {
                    eprintln!("requeue after malformed frame failed: {e}; stopping");
                    break LiveExit::Stop;
                }
                continue;
            }
        };

        // ISP + denoise + loopback write for this frame, isolated against panics
        // (see `process_and_write`). The output luma is metered inside as the AE
        // control variable; denoise is a no-op at low gain (the common case).
        let mut luma = None;
        let processed = match process_and_write(
            engine,
            out,
            buf,
            dst_w,
            dst_h,
            state.gain_index,
            args,
            &mut denoise_buf,
            &mut prev_y,
        ) {
            FrameOutcome::Written { luma: m, chroma_radius, temporal_alpha } => {
                luma = Some(m);
                last_dn = chroma_radius;
                last_ta = temporal_alpha;
                true
            }
            FrameOutcome::Skipped(reason) => {
                // Make the failure observable (throttled) so a systematic ISP/GPU
                // fault does not look like a silently idle daemon.
                errors += 1;
                if errors <= 5 || errors % 100 == 0 {
                    eprintln!("frame {frames}: skipped ({errors} total): {reason}");
                }
                false
            }
            FrameOutcome::WriteFailed(e) => {
                eprintln!("loopback write failed: {e}");
                break LiveExit::Stop;
            }
        };
        if processed {
            // AE from this frame's output luma, with apply-delay settling.
            if let Some(m) = luma {
                last_luma = m;
            }
            if args.ae {
                if ae_hold > 0 {
                    ae_hold -= 1;
                } else if let (Some(m), Some(s)) = (luma, sensor) {
                    let next = ae::step(&ae_cfg, *state, m);
                    if next != *state {
                        if let Err(e) = s.apply(next) {
                            // Don't kill the stream on a transient subdev write
                            // failure; count it so a persistent fault is visible.
                            ae_errors += 1;
                            if ae_errors <= 5 {
                                eprintln!("sensor apply failed ({ae_errors}): {e}");
                            }
                        }
                        *state = next;
                        ae_hold = AE_SETTLE;
                    }
                }
            }
            frames += 1;
            report_frames += 1;
        }

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        if let Err(e) = cam.queue_request(req).map_err(|(_, e)| e) {
            eprintln!("requeue failed: {e}; stopping");
            break LiveExit::Stop;
        }

        // Periodic throughput report.
        if last_report.elapsed() >= Duration::from_secs(2) {
            let fps = report_frames as f64 / last_report.elapsed().as_secs_f64();
            println!(
                "{frames} frames, {fps:.1} fps processed, {dropped} dropped, {errors} errors, \
                 exposure={} gain_idx={} ({:.1} fps sensor), Ylin={last_luma:.3}, denoise_r={last_dn}, temporal_a={last_ta:.2}",
                state.exposure,
                state.gain_index,
                ae::frame_rate(state.vblank),
            );
            last_report = Instant::now();
            report_frames = 0;
        }
    };

    println!("stopped after {frames} frames ({dropped} dropped, {errors} errors, {ae_errors} sensor-apply errors)");
    exit
}

/// On-demand path: keep the loopback open so the virtual webcam stays visible,
/// and open/start the GC2607 only while a consumer is capturing. Blocks on the
/// client-usage event when idle; on connect, opens and starts the camera and
/// runs [`run_live`] until the consumer disconnects ([`LiveExit::Idle`], back to
/// idle) or a fatal error ([`LiveExit::Stop`], stop the daemon). Dropping the
/// session stops and releases the camera, so the sensor powers down between uses.
///
/// The engine and AE `state` persist across activations: the GPU device is built
/// once, and exposure/gain resume near their last values rather than re-converging
/// from default on every reconnect.
#[allow(clippy::too_many_arguments)]
fn run_on_demand(
    mgr: &CameraManager,
    out: &mut LoopbackOutput,
    dst_w: usize,
    dst_h: usize,
    engine: &mut Engine,
    sensor: &Option<Sensor>,
    state: &mut AeState,
    args: &Args,
) {
    // Standby frame (black YUYV: Y=0, Cb/Cr=128) written while idle. Under
    // exclusive_caps=1 the loopback only presents a usable CAPTURE stream while a
    // producer is actively streaming, so a consumer cannot negotiate (let alone
    // STREAMON to fire the activation event) against an idle, silent producer.
    // Writing a cheap standby frame keeps the device live and negotiable with the
    // hardware camera and ISP off -- the same approach v4l2-relayd takes with its
    // test/splash source. (A single priming write is not enough: consumers need a
    // continuously streaming producer to negotiate and pre-roll.)
    let mut standby = vec![0u8; out.frame_size()];
    for px in standby.chunks_exact_mut(2) {
        px[1] = 128;
    }

    loop {
        println!("on-demand: camera idle (standby, camera off), waiting for a consumer on {}", args.device);
        // Idle keepalive: write standby frames so the device stays negotiable,
        // polling the client-usage event between frames. The hardware camera and
        // ISP stay off; this is a plain memset+write, no per-pixel work.
        loop {
            let mut active = None;
            loop {
                match out.poll_client_usage() {
                    Ok(Some(a)) => active = Some(a),
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("client-usage event poll failed: {e}");
                        break;
                    }
                }
            }
            if active == Some(true) {
                break;
            }
            if let Err(e) = out.write_frame(&standby) {
                eprintln!("standby write failed: {e}; stopping");
                return;
            }
            std::thread::sleep(STANDBY_PERIOD);
        }

        println!("on-demand: consumer connected, starting camera");
        let mut session = match cam_setup::open_raw(mgr, WIDTH, HEIGHT) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("open raw camera failed: {e}; back to idle");
                continue;
            }
        };
        if let Err(e) = session.cam.start(None) {
            eprintln!("camera start failed: {e}; back to idle");
            continue;
        }
        if let Some(s) = sensor {
            let _ = s.apply(*state);
        }
        let mut queued = true;
        for req in std::mem::take(&mut session.requests) {
            if let Err(e) = session.cam.queue_request(req).map_err(|(_, e)| e) {
                eprintln!("queue initial request failed: {e}; back to idle");
                queued = false;
                break;
            }
        }

        let exit = if queued {
            run_live(
                engine, out, &mut session.cam, &session.stream, &session.rx,
                dst_w, dst_h, sensor, state, args, true,
            )
        } else {
            LiveExit::Idle
        };

        // Dropping the session stops and releases the camera (libcamera's Drop
        // calls stop + release), powering the sensor down via runtime PM.
        drop(session);
        println!("on-demand: camera stopped");

        match exit {
            LiveExit::Idle => continue,
            LiveExit::Stop => return,
        }
    }
}
