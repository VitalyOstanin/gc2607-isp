//! Live colour webcam: capture raw via libcamera, run the tuned ISP, drive
//! auto-exposure on the sensor, and publish processed YUYV frames to a
//! v4l2loopback device that any application can open.
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
//! Usage:
//!   gc2607-video [--device /dev/videoN] [--backend cpu|gpu] [--debayer half|mhc]
//!                [--no-ae] [--target <0..1>] [--max-gain <idx>] [--threads <n>]

use std::io;
use std::time::{Duration, Instant};

use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    request::ReuseFlag,
    stream::StreamRole,
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

struct Args {
    device: String,
    backend: Backend,
    mode: DebayerMode,
    ae: bool,
    target: f64,
    max_gain: u8,
    threads: usize,
}

fn parse_args() -> Args {
    let mut device = "/dev/video0".to_string();
    // Default backend is Auto: prefer the GPU (full-res MHC, ~4x lower CPU) and
    // fall back to the CPU path if no GPU is available. The CPU defaults (#24)
    // are full-res MHC at 8 threads, overridable via --debayer / --threads.
    let mut backend = Backend::Auto;
    let mut mode = DebayerMode::Mhc;
    let mut ae = true;
    let mut target = AeConfig::default().target;
    let mut max_gain = AeConfig::default().max_gain_index;
    let mut threads = 8usize;

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
            "--target" => target = it.next().and_then(|s| s.parse().ok()).unwrap_or(target),
            "--max-gain" => max_gain = it.next().and_then(|s| s.parse().ok()).unwrap_or(max_gain),
            "--threads" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(threads).max(1),
            "-h" | "--help" => {
                eprintln!(
                    "usage: gc2607-video [--device /dev/videoN] [--backend auto|cpu|gpu] \
                     [--debayer half|mhc] [--no-ae] [--target <0..1>] [--max-gain <idx>] \
                     [--threads <n>]"
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

/// Resolve the requested backend into a concrete engine, returning it with its
/// output size. `Auto` tries the GPU first and falls back to the CPU; `Gpu`
/// exits if the GPU cannot be initialised (or the feature is not compiled).
fn build_engine(args: &Args) -> (Engine, usize, usize) {
    #[cfg(feature = "gpu")]
    if matches!(args.backend, Backend::Auto | Backend::Gpu) {
        match GpuProcessor::new(AWB_INTERVAL) {
            Ok(p) => {
                if args.mode == DebayerMode::HalfRes {
                    eprintln!("note: GPU backend is always full-res MHC (--debayer half ignored)");
                }
                let (w, h) = p.out_dims();
                println!("isp backend: gpu (full-res MHC)");
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
    println!("isp backend: cpu ({:?}, {} threads)", args.mode, args.threads);
    (
        Engine::Cpu {
            proc: Processor::new(args.mode, AWB_INTERVAL),
            yuyv: vec![0u8; dst_w * dst_h * 2],
            dst_w,
            dst_h,
        },
        dst_w,
        dst_h,
    )
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

    // Resolve the backend now (Auto may fall back to CPU), so the output size is
    // known before the loopback is opened.
    let (mut engine, dst_w, dst_h) = build_engine(&args);

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

    let mut out = LoopbackOutput::open(&args.device, dst_w as u32, dst_h as u32)
        .unwrap_or_else(|e| panic!("open loopback {}: {e}", args.device));
    println!("output: {} {dst_w}x{dst_h} YUYV", args.device);

    let ae_cfg = AeConfig {
        target: args.target,
        max_gain_index: args.max_gain,
        ..AeConfig::default()
    };
    let mut state = AeState::default();
    if let Some(s) = &sensor {
        state.exposure = s.exposure().unwrap_or(state.exposure);
        state.gain_index = s.analogue_gain().unwrap_or(0).clamp(0, 16) as u8;
        state.vblank = s.vblank().unwrap_or(state.vblank);
    }

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

    let mut frames = 0u64;
    let mut dropped = 0u64;
    let mut last_report = Instant::now();
    let mut report_frames = 0u64;

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
        let processed = match engine.process(buf) {
            Ok(yuyv) => match out.write_frame(yuyv) {
                Ok(()) => true,
                Err(e) => {
                    eprintln!("loopback write failed: {e}");
                    break;
                }
            },
            Err(_) => false,
        };
        if processed {
            // AE from the same frame's brightness.
            if args.ae {
                if let (Some(m), Some(s)) =
                    (gc2607_isp::raw::mean_norm_from_bytes(buf), &sensor)
                {
                    let next = ae::step(&ae_cfg, state, m);
                    if next != state {
                        let _ = s.apply(next);
                        state = next;
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
                 exposure={} gain_idx={} ({:.1} fps sensor)",
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
