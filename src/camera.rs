//! Shared libcamera capture setup for the live (`gc2607-video`) and raw-dump
//! (`gc2607-capture`) binaries.
//!
//! Both binaries open camera 0, configure a single raw stream at a fixed size,
//! allocate and map its buffers, build one request per buffer, and wire a
//! completion channel before starting. `open_raw` performs that whole sequence
//! and hands back a [`Session`] the caller starts, seeds with sensor state, and
//! queues.
//!
//! Lifetimes: `ActiveCamera` holds its own libcamera `shared_ptr` copy (taken in
//! `acquire`) plus a tracker token, so it survives dropping the temporary
//! `CameraList` this function uses internally; it only needs the [`CameraManager`]
//! borrow (`'d`) to stay alive. `Stream` is a `Copy` key that stays valid while
//! the camera configuration is unchanged (not tied to the `CameraConfiguration`
//! object). Mapped buffers each hold an `Arc` to the allocator instance, so the
//! local allocator can be dropped here while the buffers (now inside `requests`)
//! keep it alive.

use std::io;
use std::sync::mpsc::Receiver;

use libcamera::{
    camera::{ActiveCamera, CameraConfiguration, CameraConfigurationStatus},
    camera_manager::CameraManager,
    framebuffer_allocator::FrameBufferAllocator,
    framebuffer_map::MemoryMappedFrameBuffer,
    geometry::Size,
    properties,
    request::Request,
    stream::{Stream, StreamRole},
};

/// libcamera `Model` property of the GC2607 (the internal front camera). The
/// camera is selected by this model rather than by enumeration index, because
/// the index is not stable: when another camera is present (e.g. a USB webcam)
/// it may enumerate first, and `cameras.get(0)` would then pick the wrong
/// device and fail to configure a raw stream on it.
const GC2607_MODEL: &str = "gc2607";

/// A configured, allocated raw-capture session that has **not** been started.
///
/// The caller starts the camera (`session.cam.start(None)`), applies any initial
/// sensor state, then queues `requests` with `session.cam.queue_request`.
/// Completed requests arrive on `rx` in completion order; the libcamera callback
/// forwards them and a send after `rx` is dropped is ignored, so shutting down by
/// dropping the receiver does not panic in libcamera's thread.
pub struct Session<'d> {
    /// Acquired camera, configured but not started.
    pub cam: ActiveCamera<'d>,
    /// Raw stream handle (use to fetch buffers from completed requests).
    pub stream: Stream,
    /// Completed requests, in completion order.
    pub rx: Receiver<Request>,
    /// One request per allocated buffer, ready to queue after `start`.
    pub requests: Vec<Request>,
    /// `true` if validation adjusted the requested configuration.
    pub adjusted: bool,
    /// Held only to keep the configuration object alive for the session; the
    /// `Stream` handle remains valid as long as the camera is not reconfigured.
    _config: CameraConfiguration,
}

/// Select the GC2607 camera from `mgr` (by `Model`, see [`GC2607_MODEL`]),
/// configure a single raw stream at `width`x`height`, allocate and map its
/// buffers, build one request per buffer, and wire the completion channel.
/// Returns a [`Session`] that is configured but not started.
pub fn open_raw(mgr: &CameraManager, width: u32, height: u32) -> io::Result<Session<'_>> {
    let cameras = mgr.cameras();
    let mut chosen = None;
    let mut seen = Vec::new();
    for i in 0..cameras.len() {
        let Some(cam) = cameras.get(i) else { continue };
        let model = cam
            .properties()
            .get::<properties::Model>()
            .ok()
            .map(|m| m.to_string());
        seen.push(format!("{} ({})", model.as_deref().unwrap_or("?"), cam.id()));
        if model.as_deref() == Some(GC2607_MODEL) {
            chosen = Some(cam);
            break;
        }
    }
    let cam = chosen.ok_or_else(|| {
        io::Error::other(format!(
            "GC2607 camera (Model {GC2607_MODEL:?}) not found; available cameras: [{}]",
            seen.join(", ")
        ))
    })?;
    let mut cam = cam.acquire()?;

    let mut cfgs = cam
        .generate_configuration(&[StreamRole::Raw])
        .ok_or_else(|| io::Error::other("generate raw configuration failed"))?;
    cfgs.get_mut(0)
        .ok_or_else(|| io::Error::other("raw stream configuration missing"))?
        .set_size(Size { width, height });

    let adjusted = match cfgs.validate() {
        CameraConfigurationStatus::Valid => false,
        CameraConfigurationStatus::Adjusted => true,
        CameraConfigurationStatus::Invalid => {
            return Err(io::Error::other("invalid camera configuration"));
        }
    };
    cam.configure(&mut cfgs)?;

    let stream = cfgs
        .get(0)
        .ok_or_else(|| io::Error::other("configured stream 0 missing"))?
        .stream()
        .ok_or_else(|| io::Error::other("stream handle unavailable"))?;

    let mut alloc = FrameBufferAllocator::new(&cam);
    let buffers = alloc.alloc(&stream)?;
    let buffers = buffers
        .into_iter()
        .map(MemoryMappedFrameBuffer::new)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io::Error::other(format!("map capture buffer: {e:?}")))?;

    let requests = buffers
        .into_iter()
        .enumerate()
        .map(|(i, buf)| -> io::Result<Request> {
            let mut req = cam
                .create_request(Some(i as u64))
                .ok_or_else(|| io::Error::other("create request failed"))?;
            req.add_buffer(&stream, buf)?;
            Ok(req)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let (tx, rx) = std::sync::mpsc::channel();
    cam.on_request_completed(move |req| {
        // Runs on the libcamera thread. If the consumer dropped `rx` (shutting
        // down), the send fails; ignore it rather than panicking in that thread.
        let _ = tx.send(req);
    });

    Ok(Session {
        cam,
        stream,
        rx,
        requests,
        adjusted,
        _config: cfgs,
    })
}
