//! Raw capture via libcamera, with direct V4L2 sensor control (`capture`
//! feature; built in a container).
//!
//! Captures raw frames from the GC2607 into flat `.bin` files compatible with
//! the offline ISP (SGRBG10, 1928x1088).
//!
//! libcamera's simple pipeline drops exposure/gain controls in raw-only
//! streaming (verified by reading the source: `start()` ignores its control
//! list and `queueRequestDevice` only forwards controls when conversion runs),
//! so exposure and gain are set directly on the sensor sub-device instead. With
//! `--ae`, brightness is metered from each frame and a closed AE loop drives the
//! sensor; otherwise fixed `--exposure`/`--gain`/`--vblank` are applied once.
//!
//! Usage:
//!   gc2607-capture <out-prefix> [frames]
//!                  [--ae] [--exposure <lines>] [--gain <idx>] [--vblank <lines>]
//!   out-prefix "-" disables writing frames (useful with --ae for tuning).

use std::time::Duration;

use libcamera::{
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::FrameBuffer,
    framebuffer_map::MemoryMappedFrameBuffer,
    properties,
    request::ReuseFlag,
};

use gc2607_isp::ae::{self, AeConfig, AeState};
use gc2607_isp::camera as cam_setup;
use gc2607_isp::raw;
use gc2607_isp::sensor::Sensor;

const WIDTH: u32 = 1928;
const HEIGHT: u32 = 1088;

struct Args {
    prefix: String,
    frames: usize,
    ae: bool,
    exposure: Option<i32>,
    gain: Option<u8>,
    vblank: Option<i32>,
}

fn parse_args() -> Args {
    let mut prefix: Option<String> = None;
    let mut frames: Option<usize> = None;
    let mut ae = false;
    let mut exposure = None;
    let mut gain = None;
    let mut vblank = None;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--ae" => ae = true,
            "--exposure" => exposure = it.next().and_then(|s| s.parse().ok()),
            "--gain" => gain = it.next().and_then(|s| s.parse().ok()),
            "--vblank" => vblank = it.next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => {
                usage();
                std::process::exit(0);
            }
            _ if prefix.is_none() => prefix = Some(a),
            _ if frames.is_none() => frames = a.parse().ok(),
            _ => {}
        }
    }

    let prefix = prefix.unwrap_or_else(|| {
        usage();
        std::process::exit(1);
    });
    Args {
        prefix,
        frames: frames.unwrap_or(if ae { 30 } else { 5 }),
        ae,
        exposure,
        gain,
        vblank,
    }
}

fn usage() {
    eprintln!(
        "usage: gc2607-capture <out-prefix> [frames] \
         [--ae] [--exposure <lines>] [--gain <idx>] [--vblank <lines>]"
    );
}

fn main() {
    let args = parse_args();
    let save = args.prefix != "-";

    // Sensor sub-device for exposure/gain control. Required for --ae and for any
    // fixed exposure/gain; without it we can still dump frames at the current
    // sensor state.
    let sensor = match Sensor::open_gc2607() {
        Ok(s) => {
            println!("sensor: {}", s.path().display());
            Some(s)
        }
        Err(e) => {
            eprintln!("warning: no sensor control ({e})");
            None
        }
    };

    // Initial AE state: explicit flags override, otherwise read the sensor.
    let ae_cfg = AeConfig::default();
    let mut state = AeState::default();
    if let Some(s) = &sensor {
        state.exposure = s.exposure().unwrap_or(state.exposure);
        state.gain_index = s.analogue_gain().unwrap_or(0).clamp(0, ae::MAX_GAIN_INDEX as i32) as u8;
        state.vblank = s.vblank().unwrap_or(state.vblank);
    }
    if let Some(e) = args.exposure {
        state.exposure = e;
    }
    if let Some(g) = args.gain {
        state.gain_index = g.min(ae::MAX_GAIN_INDEX);
    }
    if let Some(v) = args.vblank {
        state.vblank = v;
    }
    if let Some(s) = &sensor {
        if let Err(e) = s.apply(state) {
            eprintln!("warning: failed to apply initial sensor state: {e}");
        }
        println!(
            "initial: exposure={} gain_idx={} ({:.2}x) vblank={} ({:.1} fps)",
            state.exposure,
            state.gain_index,
            ae::GAIN_TABLE[state.gain_index as usize],
            state.vblank,
            ae::frame_rate(state.vblank),
        );
    }

    let mgr = CameraManager::new().expect("CameraManager");
    let session = cam_setup::open_raw(&mgr, WIDTH, HEIGHT).expect("open raw camera");
    let cam_setup::Session { mut cam, stream, rx, requests, adjusted, .. } = session;
    println!(
        "camera: {}",
        *cam.properties().get::<properties::Model>().unwrap()
    );
    if adjusted {
        println!("config adjusted");
    }
    if let Some(cfg) = stream.configuration() {
        println!(
            "stream: {:?} {}x{} stride={}",
            cfg.get_pixel_format(),
            cfg.get_size().width,
            cfg.get_size().height,
            cfg.get_stride()
        );
    }

    cam.start(None).expect("start");

    // Re-apply our state after start: libcamera may touch the sensor at
    // configure/start time.
    if let Some(s) = &sensor {
        let _ = s.apply(state);
    }

    for req in requests {
        cam.queue_request(req).map_err(|(_, e)| e).unwrap();
    }

    let mut done = 0;
    while done < args.frames {
        let mut req = match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(r) => r,
            Err(_) => {
                eprintln!("camera request timed out, stopping");
                break;
            }
        };

        let fb: &MemoryMappedFrameBuffer<FrameBuffer> = req.buffer(&stream).unwrap();
        let planes = fb.data();
        let plane = planes.first().unwrap();
        let bytes_used = fb.metadata().unwrap().planes().get(0).unwrap().bytes_used as usize;
        let buf = &plane[..bytes_used];

        let metric = raw::mean_norm_from_bytes(buf);

        if save {
            let path = format!("{}-{done:03}.bin", args.prefix);
            std::fs::write(&path, buf).expect("write frame");
        }

        if args.ae {
            if let (Some(m), Some(s)) = (metric, &sensor) {
                let next = ae::step(&ae_cfg, state, m);
                if next != state {
                    if let Err(e) = s.apply(next) {
                        eprintln!("warning: AE apply failed: {e}");
                    }
                    state = next;
                }
                println!(
                    "frame {done:03}: mean={:.1}% -> exposure={} gain_idx={} ({:.2}x) vblank={} ({:.1} fps)",
                    m * 100.0,
                    state.exposure,
                    state.gain_index,
                    ae::GAIN_TABLE[state.gain_index as usize],
                    state.vblank,
                    ae::frame_rate(state.vblank),
                );
            }
        } else {
            println!(
                "frame {done:03}: mean={}",
                metric.map(|m| format!("{:.1}%", m * 100.0)).unwrap_or_else(|| "n/a".into())
            );
        }

        done += 1;
        req.reuse(ReuseFlag::REUSE_BUFFERS);
        cam.queue_request(req).map_err(|(_, e)| e).unwrap();
    }

    println!("done: {done} frames");
}
