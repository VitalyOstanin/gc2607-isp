//! Live colour webcam: capture raw via libcamera, run the tuned ISP, drive
//! auto-exposure on the sensor, and publish processed YUYV frames to a
//! v4l2loopback device that any application can open.
//!
//! Auto-exposure meters the mean luma of the produced frame (post white
//! balance, lens-shading, CCM and gamma) toward a perceptual mid-grey, so the
//! exposure tracks the brightness the viewer sees rather than a raw level.
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
//! Usage:
//!   gc2607-video [--device /dev/videoN] [--backend cpu|gpu] [--debayer half|mhc]
//!                [--no-ae] [--target <0..1>] [--max-gain <idx>] [--threads <n>]
//!                [--measure-delay]

use std::io;
use std::time::{Duration, Instant};

use libcamera::{
    camera::{ActiveCamera, CameraConfigurationStatus},
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    request::{Request, ReuseFlag},
    stream::{Stream, StreamRole},
};

use gc2607_isp::ae::{self, AeConfig, AeState};
use gc2607_isp::output::{rgb_to_yuyv_crop, LoopbackOutput};
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

/// Either ISP engine, producing a packed YUYV frame from raw capture bytes. The
/// CPU engine renders RGB then packs (with the centre crop); the GPU engine
/// packs YUYV directly in the shader.
enum Engine {
    Cpu {
        proc: Processor,
        yuyv: Vec<u8>,
        dst_w: usize,
        dst_h: usize,
    },
    #[cfg(feature = "gpu")]
    Gpu(GpuProcessor),
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

const WIDTH: u32 = 1928;
const HEIGHT: u32 = 1088;

/// Re-estimate white balance / CCM every N frames. The scene's colour
/// temperature is stable between estimates, so the AWB sort (the heaviest serial
/// step) runs only occasionally; AE brightness is still metered every frame.
const AWB_INTERVAL: u64 = 8;

/// Default AE target. AE meters the mean luma of the *produced* frame (after
/// white balance, lens-shading, CCM and sRGB gamma), so the target is a
/// perceptual mid-grey on the 0..1 output scale — not a linear raw level. This
/// makes exposure converge on the brightness the viewer actually sees and
/// self-corrects for the pipeline's brightness gain. Override with `--target`.
const AE_TARGET_LUMA: f64 = 0.42;

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
}

fn parse_args() -> Args {
    let mut device = "/dev/video0".to_string();
    // Default backend is Auto: prefer the GPU (full-res MHC, ~4x lower CPU) and
    // fall back to the CPU path if no GPU is available. The CPU defaults (#24)
    // are full-res MHC at 8 threads, overridable via --debayer / --threads.
    let mut backend = Backend::Auto;
    let mut mode = DebayerMode::Mhc;
    let mut ae = true;
    let mut target = AE_TARGET_LUMA;
    let mut max_gain = AeConfig::default().max_gain_index;
    let mut threads = 8usize;
    // Lateral chromatic aberration correction (full-res MHC only); on by default.
    let mut lca = true;
    // Sensor apply-delay measurement mode (no loopback, no ISP output).
    let mut measure = false;

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
            "--measure-delay" => measure = true,
            "--target" => target = it.next().and_then(|s| s.parse().ok()).unwrap_or(target),
            "--max-gain" => max_gain = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_gain),
            "--threads" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(threads).max(1),
            "-h" | "--help" => {
                eprintln!(
                    "usage: gc2607-video [--device /dev/videoN] [--backend auto|cpu|gpu] \
                     [--debayer half|mhc] [--no-ae] [--no-lca] [--target <0..1>] \
                     [--max-gain <idx>] [--threads <n>] [--measure-delay]"
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
    }
}

/// Output (cropped) size for a debayer mode: a standard 16:9 size centred in
/// the slightly larger ISP output.
fn output_size(mode: DebayerMode) -> (usize, usize) {
    match mode {
        DebayerMode::Mhc => (1920, 1080),  // from 1928x1088
        DebayerMode::HalfRes => (960, 540), // from 964x544
    }
}

/// Mean luma of a packed YUYV frame, as a fraction of full scale (0..1).
///
/// The Y bytes sit at even offsets; this subsamples every fourth pixel, which is
/// ample for an exposure metric and keeps the per-frame cost negligible. This is
/// the AE control variable: metering the produced frame's luma makes exposure
/// converge on the brightness the viewer actually sees, accounting for white
/// balance, lens-shading, CCM and gamma in a single measurement.
fn mean_luma_norm(yuyv: &[u8]) -> f64 {
    let mut sum = 0u64;
    let mut n = 0u64;
    let mut i = 0;
    while i < yuyv.len() {
        sum += yuyv[i] as u64; // Y of every fourth pixel (2 bytes/px, step 8)
        n += 1;
        i += 8;
    }
    if n == 0 {
        0.0
    } else {
        sum as f64 / n as f64 / 255.0
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
                return (Engine::Gpu(p), w, h);
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
            proc: Processor::new(args.mode, AWB_INTERVAL, args.lca),
            yuyv: vec![0u8; dst_w * dst_h * 2],
            dst_w,
            dst_h,
        },
        dst_w,
        dst_h,
    )
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
    let mean = {
        let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(stream).unwrap();
        let plane = fb.data();
        let plane = plane.first().unwrap();
        let bytes_used = fb.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize;
        gc2607_isp::raw::mean_norm_from_bytes(&plane[..bytes_used])
    };
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

fn main() {
    let args = parse_args();

    // Bound the ISP thread pool to the requested count (the CPU budget is the
    // user's to set). The GPU backend does its pixel work on the iGPU, so the
    // pool is only used by the occasional CPU-side AWB estimate.
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .expect("build rayon pool");

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

    // Seed the AE state from whatever the sensor currently holds.
    let mut state = AeState::default();
    if let Some(s) = &sensor {
        state.exposure = s.exposure().unwrap_or(state.exposure);
        state.gain_index = s.analogue_gain().unwrap_or(0).clamp(0, 16) as u8;
        state.vblank = s.vblank().unwrap_or(state.vblank);
    }

    // --- Camera setup (shared by the live path and --measure-delay). ---
    let mgr = CameraManager::new().expect("CameraManager");
    let cameras = mgr.cameras();
    let cam = cameras.get(0).expect("no cameras found");
    let mut cam = cam.acquire().expect("acquire camera");

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::Raw])
        .expect("generate raw configuration");
    cfgs.get_mut(0).unwrap().set_size(Size {
        width: WIDTH,
        height: HEIGHT,
    });
    if matches!(cfgs.validate(), CameraConfigurationStatus::Invalid) {
        panic!("invalid camera configuration");
    }
    cam.configure(&mut cfgs).expect("configure");

    let cfg = cfgs.get(0).unwrap();
    let stream = cfg.stream().unwrap();

    let mut alloc = FrameBufferAllocator::new(&cam);
    let buffers = alloc.alloc(&stream).expect("alloc buffers");
    let buffers = buffers
        .into_iter()
        .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
        .collect::<Vec<_>>();

    let reqs: Vec<_> = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| {
            let mut req = cam.create_request(Some(i as u64)).unwrap();
            req.add_buffer(&stream, buf).unwrap();
            req
        })
        .collect();

    let (tx, rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        tx.send(req).unwrap();
    });

    cam.start(None).expect("start");
    if let Some(s) = &sensor {
        let _ = s.apply(state);
    }
    for req in reqs {
        cam.queue_request(req).map_err(|(_, e)| e).unwrap();
    }

    // --- Sensor apply-delay measurement mode (no loopback, no ISP output). ---
    if args.measure {
        match &sensor {
            Some(s) => run_measure_delay(&mut cam, &stream, &rx, s, state),
            None => eprintln!("--measure-delay needs sensor control, but none is available"),
        }
        return;
    }

    // --- Live path: ISP engine + loopback output. ---
    // Resolve the backend now (Auto may fall back to CPU), so the output size is
    // known before the loopback is opened.
    let (mut engine, dst_w, dst_h) = build_engine(&args);

    let mut out = LoopbackOutput::open(&args.device, dst_w as u32, dst_h as u32)
        .unwrap_or_else(|e| panic!("open loopback {}: {e}", args.device));
    println!("output: {} {dst_w}x{dst_h} YUYV", args.device);

    let ae_cfg = AeConfig {
        target: args.target,
        max_gain_index: args.max_gain,
        ..AeConfig::default()
    };

    let mut frames = 0u64;
    let mut dropped = 0u64;
    let mut last_report = Instant::now();
    let mut report_frames = 0u64;
    // Last metered output luma (0..1), for the periodic telemetry line.
    let mut last_luma = 0f64;

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

    loop {
        // Block for one completed request, then drain any extra ready ones,
        // requeuing all but the freshest so we always process the newest frame.
        let mut req = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("camera timed out, stopping");
                break;
            }
        };
        while let Ok(newer) = rx.try_recv() {
            req.reuse(ReuseFlag::REUSE_BUFFERS);
            cam.queue_request(req).map_err(|(_, e)| e).unwrap();
            req = newer;
            dropped += 1;
        }

        let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
        let plane = fb.data();
        let plane = plane.first().unwrap();
        let bytes_used = fb.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize;
        let buf = &plane[..bytes_used];

        // ISP (reuses buffers, caches AWB/CCM across frames). The engine returns
        // a packed YUYV frame (CPU renders RGB then packs; GPU packs in-shader).
        // The output luma is metered here (while the YUYV is in hand) as the AE
        // control variable: see `mean_luma_norm`.
        let mut luma = None;
        let processed = match engine.process(buf) {
            Ok(yuyv) => {
                luma = Some(mean_luma_norm(yuyv));
                match out.write_frame(yuyv) {
                    Ok(()) => true,
                    Err(e) => {
                        eprintln!("loopback write failed: {e}");
                        break;
                    }
                }
            }
            Err(_) => false,
        };
        if processed {
            // AE from this frame's output luma, with apply-delay settling.
            if let Some(m) = luma {
                last_luma = m;
            }
            if args.ae {
                if ae_hold > 0 {
                    ae_hold -= 1;
                } else if let (Some(m), Some(s)) = (luma, &sensor) {
                    let next = ae::step(&ae_cfg, state, m);
                    if next != state {
                        let _ = s.apply(next);
                        state = next;
                        ae_hold = AE_SETTLE;
                    }
                }
            }
            frames += 1;
            report_frames += 1;
        }

        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e).unwrap();

        // Periodic throughput report.
        if last_report.elapsed() >= Duration::from_secs(2) {
            let fps = report_frames as f64 / last_report.elapsed().as_secs_f64();
            println!(
                "{frames} frames, {fps:.1} fps processed, {dropped} dropped, \
                 exposure={} gain_idx={} ({:.1} fps sensor), Y={last_luma:.3}",
                state.exposure,
                state.gain_index,
                ae::frame_rate(state.vblank),
            );
            last_report = Instant::now();
            report_frames = 0;
        }
    }

    println!("stopped after {frames} frames ({dropped} dropped)");
}
