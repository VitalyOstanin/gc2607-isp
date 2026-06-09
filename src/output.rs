//! Minimal V4L2 output to a v4l2loopback device.
//!
//! The processed frames are pushed to a loopback node (`/dev/videoN`, the
//! "Virtual Camera") as packed YUYV (YUV 4:2:2), the format applications expect
//! from a webcam. v4l2loopback accepts the simple `write()` producer method, so
//! after one `VIDIOC_S_FMT` the frames are written sequentially.
//!
//! As decided for this project, this is a hand-rolled V4L2 layer over raw
//! ioctls (the `v4l` crate is stale and its output path is undocumented). The
//! one fragile point is the `struct v4l2_format` layout: its size and field
//! offsets must match the kernel exactly or the ioctl is rejected. The struct
//! is laid out to the 64-bit kernel ABI (total 208 bytes, the `fmt` union at
//! offset 8) and this is asserted at construction.

use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io;
use std::num::NonZeroUsize;
use std::os::fd::{AsFd, AsRawFd};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};
use rayon::prelude::*;

/// `V4L2_PIX_FMT_YUYV` four-character code.
const V4L2_PIX_FMT_YUYV: u32 =
    (b'Y' as u32) | ((b'U' as u32) << 8) | ((b'Y' as u32) << 16) | ((b'V' as u32) << 24);
/// `V4L2_BUF_TYPE_VIDEO_OUTPUT`.
const V4L2_BUF_TYPE_VIDEO_OUTPUT: u32 = 2;
/// `V4L2_FIELD_NONE`.
const V4L2_FIELD_NONE: u32 = 1;
/// `V4L2_COLORSPACE_SRGB`.
const V4L2_COLORSPACE_SRGB: u32 = 8;
/// `V4L2_QUANTIZATION_FULL_RANGE` (full-range YCbCr, matches our sRGB output).
const V4L2_QUANTIZATION_FULL_RANGE: u32 = 1;

/// `struct v4l2_pix_format` (uapi/linux/videodev2.h), 48 bytes on all ABIs.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2PixFormat {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    private: u32,
    flags: u32,
    enc: u32, // union { ycbcr_enc; hsv_enc }
    quantization: u32,
    xfer_func: u32,
}

/// `struct v4l2_format`. The kernel's `fmt` union is 8-aligned (it contains
/// `v4l2_window`, which has pointers) and starts at offset 8; the whole struct
/// is 208 bytes on 64-bit. `pix` is the first union member.
#[repr(C, align(8))]
struct V4l2Format {
    type_: u32,
    _pad: u32,
    pix: V4l2PixFormat,
    _tail: [u8; 152],
}

// VIDIOC_S_FMT = _IOWR('V', 5, struct v4l2_format)
nix::ioctl_readwrite!(vidioc_s_fmt, b'V', 5, V4l2Format);

// --- streaming output (MMAP) ---
//
// v4l2loopback only forwards a producer-set frame timestamp when the producer
// uses the streaming API (VIDIOC_QBUF); the plain `write()` path always stamps
// the frame with the write time (verified in v4l2loopback's `vidioc_qbuf` /
// `v4l2l_get_timestamp`). To let a consumer (a browser's WebRTC stack) sync
// audio against the true capture instant, we drive the loopback as an MMAP
// output device and set `v4l2_buffer.timestamp` to libcamera's capture time.
// v4l2loopback supports only V4L2_MEMORY_MMAP for buffers (USERPTR is rejected).

/// `V4L2_MEMORY_MMAP`.
const V4L2_MEMORY_MMAP: u32 = 1;

/// `struct v4l2_requestbuffers` (uapi videodev2.h), 20 bytes on 64-bit.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2RequestBuffers {
    count: u32,
    type_: u32,
    memory: u32,
    capabilities: u32,
    flags: u8,
    reserved: [u8; 3],
}

/// `struct timeval` as embedded in `v4l2_buffer` (64-bit kernel ABI: both fields
/// are `__kernel_long_t`/`__kernel_suseconds_t`, 8 bytes each).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2Timeval {
    tv_sec: i64,
    tv_usec: i64,
}

/// `struct v4l2_timecode` (uapi videodev2.h), 16 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2Timecode {
    type_: u32,
    flags: u32,
    frames: u8,
    seconds: u8,
    minutes: u8,
    hours: u8,
    userbits: [u8; 4],
}

/// `struct v4l2_buffer` (uapi videodev2.h), 88 bytes on 64-bit. `m` is the union
/// (we only use `offset` for MMAP query and never read it back as a pointer).
/// `repr(C)` inserts the 4-byte pad before `timestamp` (8-aligned) and the
/// trailing pad; the offsets are asserted below against the kernel layout.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2Buffer {
    index: u32,
    type_: u32,
    bytesused: u32,
    flags: u32,
    field: u32,
    timestamp: V4l2Timeval,
    timecode: V4l2Timecode,
    sequence: u32,
    memory: u32,
    m: u64,
    length: u32,
    reserved2: u32,
    request_fd: u32,
}

const _: () = assert!(core::mem::size_of::<V4l2RequestBuffers>() == 20);
const _: () = assert!(core::mem::size_of::<V4l2Buffer>() == 88);
const _: () = assert!(core::mem::offset_of!(V4l2Buffer, timestamp) == 24);
const _: () = assert!(core::mem::offset_of!(V4l2Buffer, m) == 64);
const _: () = assert!(core::mem::offset_of!(V4l2Buffer, length) == 72);

// VIDIOC_REQBUFS  = _IOWR('V',  8, struct v4l2_requestbuffers)
nix::ioctl_readwrite!(vidioc_reqbufs, b'V', 8, V4l2RequestBuffers);
// VIDIOC_QUERYBUF = _IOWR('V',  9, struct v4l2_buffer)
nix::ioctl_readwrite!(vidioc_querybuf, b'V', 9, V4l2Buffer);
// VIDIOC_QBUF     = _IOWR('V', 15, struct v4l2_buffer)
nix::ioctl_readwrite!(vidioc_qbuf, b'V', 15, V4l2Buffer);
// VIDIOC_DQBUF    = _IOWR('V', 17, struct v4l2_buffer)
nix::ioctl_readwrite!(vidioc_dqbuf, b'V', 17, V4l2Buffer);
// VIDIOC_STREAMON  = _IOW('V', 18, int)
nix::ioctl_write_ptr!(vidioc_streamon, b'V', 18, i32);
// VIDIOC_STREAMOFF = _IOW('V', 19, int)
nix::ioctl_write_ptr!(vidioc_streamoff, b'V', 19, i32);

// --- v4l2loopback client-usage event (on-demand camera gating) ---
//
// The Ubuntu/IPU6 build of v4l2loopback queues a private "client usage" V4L2
// event whenever a capture client starts or stops streaming on the loopback;
// the event payload's leading `count` is non-zero while at least one consumer
// is capturing. A producer (this binary) subscribes to it to run the hardware
// camera and ISP only while the virtual webcam is actually in use. This is the
// same mechanism v4l2-relayd uses. A v4l2loopback without the patch rejects the
// subscription with EINVAL, which the caller treats as "always-on".

/// `V4L2_EVENT_PRIVATE_START` (uapi videodev2.h).
const V4L2_EVENT_PRIVATE_START: u32 = 0x0800_0000;
/// v4l2loopback's `V4L2_EVENT_PRI_CLIENT_USAGE` (module base + 0x08E00000 + 1).
const V4L2_EVENT_PRI_CLIENT_USAGE: u32 = V4L2_EVENT_PRIVATE_START + 0x08E0_0000 + 1;
/// `V4L2_EVENT_SUB_FL_SEND_INITIAL`: deliver the current state right after
/// subscribing, so a consumer already capturing at startup is noticed at once.
const V4L2_EVENT_SUB_FL_SEND_INITIAL: u32 = 1;

/// `struct v4l2_event_subscription` (uapi videodev2.h), 32 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct V4l2EventSubscription {
    type_: u32,
    id: u32,
    flags: u32,
    reserved: [u32; 5],
}

/// `struct v4l2_event` (uapi videodev2.h), 136 bytes on 64-bit. Only `type_`
/// (offset 0) and the payload's first `u32` (`u`, offset 8 -- the client-usage
/// `count`) are read; the remaining fields are reserved to match the kernel ABI
/// size exactly, so their internal layout (the `timespec` etc.) is irrelevant.
/// The layout is asserted below against the size/offsets the kernel headers give.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct V4l2Event {
    type_: u32,
    _pad0: u32,
    u: [u8; 64],
    pending: u32,
    sequence: u32,
    timestamp: [u8; 16],
    id: u32,
    reserved: [u32; 8],
}

// Guard the hand-laid ABI: a wrong size/offset would make the kernel reject or
// mis-copy the ioctl (matches the verified layout: size 136, type@0, u@8).
const _: () = assert!(core::mem::size_of::<V4l2EventSubscription>() == 32);
const _: () = assert!(core::mem::size_of::<V4l2Event>() == 136);
const _: () = assert!(core::mem::offset_of!(V4l2Event, type_) == 0);
const _: () = assert!(core::mem::offset_of!(V4l2Event, u) == 8);

// VIDIOC_SUBSCRIBE_EVENT = _IOW('V', 90, struct v4l2_event_subscription)
nix::ioctl_write_ptr!(vidioc_subscribe_event, b'V', 90, V4l2EventSubscription);
// VIDIOC_DQEVENT = _IOR('V', 89, struct v4l2_event)
nix::ioctl_read!(vidioc_dqevent, b'V', 89, V4l2Event);

/// Resolve a v4l2loopback node by its `card_label` (the string set with the
/// `card_label=` module option, exposed at `/sys/class/video4linux/videoN/name`).
///
/// This makes the output device addressable by a stable label instead of a
/// `/dev/videoN` number that depends on probe order: on an Intel IPU6 machine the
/// ISYS alone registers dozens of `/dev/video*` nodes, so the loopback's number is
/// not fixed. The lookup mirrors how the GC2607 sub-device is found by entity name
/// in `sensor.rs`. The label match also works regardless of `exclusive_caps`: the
/// sysfs `name` does not change when the device flips between output and capture
/// caps as a producer attaches.
///
/// Note: V4L2 stores the card string in a 32-byte field (31 chars + NUL), so a
/// label longer than 31 characters is truncated by the kernel; the comparison is
/// against whatever sysfs reports, so the requested label must fit.
pub fn find_loopback_by_label(label: &str) -> io::Result<PathBuf> {
    let want = label.trim();
    let sys = Path::new("/sys/class/video4linux");
    for entry in std::fs::read_dir(sys)? {
        let entry = entry?;
        let node = entry.file_name();
        let node = node.to_string_lossy();
        if !node.starts_with("video") {
            continue;
        }
        let name = std::fs::read_to_string(entry.path().join("name")).unwrap_or_default();
        if name.trim() == want {
            return Ok(PathBuf::from("/dev").join(node.as_ref()));
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "no v4l2loopback device labelled {want:?} found; create it with \
             `modprobe v4l2loopback card_label=\"{want}\" exclusive_caps=1`"
        ),
    ))
}

/// One MMAP-ed output buffer shared with the kernel/v4l2loopback.
struct MappedBuffer {
    ptr: NonNull<c_void>,
    len: usize,
}

/// A loopback output node configured for YUYV frames of a fixed size, driven as
/// an MMAP streaming device so each frame carries the capture timestamp.
pub struct LoopbackOutput {
    file: File,
    width: u32,
    height: u32,
    buffers: Vec<MappedBuffer>,
    /// Buffer indices we currently own (dequeued / not yet queued). When empty,
    /// every buffer is in flight and the next write reclaims one via `DQBUF`.
    free: Vec<u32>,
}

impl LoopbackOutput {
    /// Number of output buffers to request. v4l2loopback clamps this to its
    /// `max_buffers` module option (2 by default); the returned count is used.
    const REQUEST_BUFFERS: u32 = 4;

    /// Open `path`, set its output format to YUYV `width`x`height`, allocate and
    /// map the MMAP output buffers, and start streaming.
    pub fn open<P: AsRef<Path>>(path: P, width: u32, height: u32) -> io::Result<LoopbackOutput> {
        // Guard the hand-laid ABI: a wrong size would make the kernel reject
        // (or mis-copy) the ioctl.
        const _: () = assert!(core::mem::size_of::<V4l2Format>() == 208);
        const _: () = assert!(core::mem::size_of::<V4l2PixFormat>() == 48);

        let file = OpenOptions::new().read(true).write(true).open(path)?;

        let pix = V4l2PixFormat {
            width,
            height,
            pixelformat: V4L2_PIX_FMT_YUYV,
            field: V4L2_FIELD_NONE,
            bytesperline: width * 2,
            sizeimage: width * height * 2,
            colorspace: V4L2_COLORSPACE_SRGB,
            quantization: V4L2_QUANTIZATION_FULL_RANGE,
            ..Default::default()
        };
        let mut fmt = V4l2Format {
            type_: V4L2_BUF_TYPE_VIDEO_OUTPUT,
            _pad: 0,
            pix,
            _tail: [0u8; 152],
        };
        // SAFETY: `fmt` is a correctly-sized v4l2_format; the fd is open RW.
        unsafe { vidioc_s_fmt(file.as_raw_fd(), &mut fmt) }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;

        // Allocate MMAP output buffers.
        let mut req = V4l2RequestBuffers {
            count: Self::REQUEST_BUFFERS,
            type_: V4L2_BUF_TYPE_VIDEO_OUTPUT,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        // SAFETY: `req` is a correctly-sized v4l2_requestbuffers; fd is open RW.
        unsafe { vidioc_reqbufs(file.as_raw_fd(), &mut req) }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
        if req.count == 0 {
            return Err(io::Error::other("v4l2loopback allocated 0 output buffers"));
        }

        // Query and map each buffer.
        let mut buffers = Vec::with_capacity(req.count as usize);
        let mut free = Vec::with_capacity(req.count as usize);
        for i in 0..req.count {
            let mut qb = V4l2Buffer {
                index: i,
                type_: V4L2_BUF_TYPE_VIDEO_OUTPUT,
                memory: V4L2_MEMORY_MMAP,
                ..Default::default()
            };
            // SAFETY: correctly-sized v4l2_buffer; fd open RW.
            unsafe { vidioc_querybuf(file.as_raw_fd(), &mut qb) }
                .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
            let len = NonZeroUsize::new(qb.length as usize)
                .ok_or_else(|| io::Error::other("v4l2loopback buffer has zero length"))?;
            // SAFETY: map the driver buffer at its reported `m.offset`; `file`
            // stays open for the lifetime of `LoopbackOutput`, keeping the
            // mapping valid. Unmapped in `Drop`.
            let ptr = unsafe {
                mmap(
                    None,
                    len,
                    ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                    MapFlags::MAP_SHARED,
                    &file,
                    qb.m as i64,
                )
            }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
            buffers.push(MappedBuffer {
                ptr,
                len: len.get(),
            });
            free.push(i);
        }

        // Start streaming.
        let type_ = V4L2_BUF_TYPE_VIDEO_OUTPUT as i32;
        // SAFETY: `type_` is a valid buffer-type int; fd open RW.
        unsafe { vidioc_streamon(file.as_raw_fd(), &type_) }
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;

        Ok(LoopbackOutput {
            file,
            width,
            height,
            buffers,
            free,
        })
    }

    /// Frame size in bytes (YUYV: 2 bytes per pixel).
    pub fn frame_size(&self) -> usize {
        self.width as usize * self.height as usize * 2
    }

    /// Queue one YUYV frame for output, stamped with `timestamp_ns` (a
    /// `CLOCK_MONOTONIC` nanosecond count — libcamera's capture time). A zero
    /// `timestamp_ns` leaves the timestamp unset, so v4l2loopback stamps the
    /// frame with the current time (used for standby frames). The buffer must be
    /// exactly [`Self::frame_size`].
    ///
    /// Reclaims a previously queued buffer (blocking `DQBUF`) when all buffers
    /// are in flight; with >= 2 buffers one is always available right after a
    /// queue, so this does not stall a steadily producing loop.
    pub fn write_frame(&mut self, yuyv: &[u8], timestamp_ns: u64) -> io::Result<()> {
        if yuyv.len() != self.frame_size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "frame is {} bytes, expected {} ({}x{} YUYV)",
                    yuyv.len(),
                    self.frame_size(),
                    self.width,
                    self.height
                ),
            ));
        }

        // Reclaim a buffer if none are free.
        if self.free.is_empty() {
            let mut dq = V4l2Buffer {
                type_: V4L2_BUF_TYPE_VIDEO_OUTPUT,
                memory: V4L2_MEMORY_MMAP,
                ..Default::default()
            };
            // SAFETY: correctly-sized v4l2_buffer; fd open RW.
            match unsafe { vidioc_dqbuf(self.file.as_raw_fd(), &mut dq) } {
                Ok(_) => self.free.push(dq.index),
                // v4l2loopback returns EFAULT from DQBUF(OUTPUT) when its output
                // queue is empty. That queue is cleared whenever a consumer
                // attaches: the consumer's VIDIOC_REQBUFS runs
                // prepare_buffer_queue(), which drops every buffer from the
                // output queue and resets their flags (see vidioc_reqbufs ->
                // prepare_buffer_queue and the DQBUF OUTPUT branch in
                // v4l2loopback.c). Our buffers are no longer in flight and the
                // producer keeps its format token and buffer_count, so reclaim
                // them all and carry on; the next QBUF re-enqueues a buffer.
                Err(nix::errno::Errno::EFAULT) => {
                    self.free.clear();
                    self.free.extend(0..self.buffers.len() as u32);
                }
                Err(e) => return Err(io::Error::from_raw_os_error(e as i32)),
            }
        }

        let idx = self.free.pop().expect("a free buffer after reclaim");
        let mb = &self.buffers[idx as usize];
        if yuyv.len() > mb.len {
            // Re-add the index so state stays consistent on this error path.
            self.free.push(idx);
            return Err(io::Error::other(format!(
                "frame {} bytes exceeds mapped buffer {}",
                yuyv.len(),
                mb.len
            )));
        }
        // SAFETY: `mb.ptr` maps `mb.len` writable bytes; we copy at most that many.
        unsafe {
            std::slice::from_raw_parts_mut(mb.ptr.as_ptr() as *mut u8, mb.len)[..yuyv.len()]
                .copy_from_slice(yuyv);
        }

        let mut qb = V4l2Buffer {
            index: idx,
            type_: V4L2_BUF_TYPE_VIDEO_OUTPUT,
            memory: V4L2_MEMORY_MMAP,
            bytesused: yuyv.len() as u32,
            field: V4L2_FIELD_NONE,
            length: mb.len as u32,
            ..Default::default()
        };
        if timestamp_ns != 0 {
            qb.timestamp = V4l2Timeval {
                tv_sec: (timestamp_ns / 1_000_000_000) as i64,
                tv_usec: ((timestamp_ns % 1_000_000_000) / 1_000) as i64,
            };
        }
        // SAFETY: correctly-sized v4l2_buffer; fd open RW.
        unsafe { vidioc_qbuf(self.file.as_raw_fd(), &mut qb) }.map_err(|e| {
            self.free.push(idx);
            io::Error::from_raw_os_error(e as i32)
        })?;
        Ok(())
    }

    /// Subscribe to the v4l2loopback client-usage event so the producer can run
    /// the camera only while a consumer is capturing. `SEND_INITIAL` delivers the
    /// current state immediately. Returns an error (typically `EINVAL`) on a
    /// v4l2loopback build without this event, letting the caller fall back to
    /// always-on streaming.
    pub fn subscribe_client_usage(&self) -> io::Result<()> {
        let sub = V4l2EventSubscription {
            type_: V4L2_EVENT_PRI_CLIENT_USAGE,
            flags: V4L2_EVENT_SUB_FL_SEND_INITIAL,
            ..Default::default()
        };
        // SAFETY: `sub` is a correctly-sized v4l2_event_subscription; fd is open.
        unsafe { vidioc_subscribe_event(self.file.as_raw_fd(), &sub) }
            .map(|_| ())
            .map_err(|e| io::Error::from_raw_os_error(e as i32))
    }

    /// Dequeue one pending client-usage event without blocking: `Some(active)`
    /// where `active` is true while at least one consumer is capturing, or `None`
    /// when the event queue is empty. Requires a prior [`Self::subscribe_client_usage`].
    ///
    /// The loopback fd is opened blocking (so `write_frame` is reliable), and on a
    /// blocking fd `VIDIOC_DQEVENT` *waits* for an event instead of returning
    /// `ENOENT` -- which would deadlock the idle keepalive loop. So gate the
    /// dequeue on a zero-timeout `poll(POLLPRI)`: only call `VIDIOC_DQEVENT` when
    /// the kernel reports a pending event, in which case it returns at once.
    pub fn poll_client_usage(&self) -> io::Result<Option<bool>> {
        use nix::poll::{poll, PollFd, PollFlags, PollTimeout};

        let mut fds = [PollFd::new(self.file.as_fd(), PollFlags::POLLPRI)];
        let ready = poll(&mut fds, PollTimeout::ZERO)
            .map_err(|e| io::Error::from_raw_os_error(e as i32))?;
        if ready == 0
            || !fds[0]
                .revents()
                .is_some_and(|r| r.contains(PollFlags::POLLPRI))
        {
            return Ok(None);
        }

        let mut ev = V4l2Event {
            type_: 0,
            _pad0: 0,
            u: [0u8; 64],
            pending: 0,
            sequence: 0,
            timestamp: [0u8; 16],
            id: 0,
            reserved: [0u32; 8],
        };
        // SAFETY: `ev` is a correctly-sized v4l2_event; the kernel fills it. An
        // event is pending (POLLPRI), so this does not block.
        match unsafe { vidioc_dqevent(self.file.as_raw_fd(), &mut ev) } {
            Ok(_) => {
                // The client-usage payload is a single u32 `count` at the start
                // of the event union; non-zero means a consumer is capturing.
                let count = u32::from_ne_bytes([ev.u[0], ev.u[1], ev.u[2], ev.u[3]]);
                Ok(Some(count != 0))
            }
            // Raced away between poll and dequeue: treat as no event.
            Err(nix::errno::Errno::ENOENT) => Ok(None),
            Err(e) => Err(io::Error::from_raw_os_error(e as i32)),
        }
    }
}

impl Drop for LoopbackOutput {
    fn drop(&mut self) {
        // Stop streaming, then unmap every buffer. Errors are ignored: there is
        // nothing to recover to in a destructor, and the fd close that follows
        // releases the kernel-side resources regardless.
        let type_ = V4L2_BUF_TYPE_VIDEO_OUTPUT as i32;
        // SAFETY: valid buffer-type int; fd still open.
        let _ = unsafe { vidioc_streamoff(self.file.as_raw_fd(), &type_) };
        for b in &self.buffers {
            // SAFETY: `b.ptr`/`b.len` came from `mmap` in `open` and are unmapped
            // exactly once here.
            let _ = unsafe { munmap(b.ptr, b.len) };
        }
    }
}

/// Convert an interleaved RGB8 image to packed YUYV (YUY2), optionally
/// centre-cropping from `src_w`x`src_h` to `dst_w`x`dst_h`.
///
/// Uses full-range BT.601 (JFIF) coefficients to match the full-range sRGB
/// output declared on the loopback format. `dst_w` must be even (YUYV pairs
/// two horizontal pixels); `dst` must hold `dst_w*dst_h*2` bytes.
pub fn rgb_to_yuyv_crop(
    rgb: &[u8],
    src_w: usize,
    src_h: usize,
    dst: &mut [u8],
    dst_w: usize,
    dst_h: usize,
) {
    debug_assert!(dst_w % 2 == 0);
    debug_assert!(dst_w <= src_w && dst_h <= src_h);
    debug_assert_eq!(dst.len(), dst_w * dst_h * 2);

    let off_x = (src_w - dst_w) / 2;
    let off_y = (src_h - dst_h) / 2;

    // Row-parallel: each output row depends only on its own source row.
    dst.par_chunks_mut(dst_w * 2)
        .enumerate()
        .for_each(|(y, drow)| {
            let src_row = (off_y + y) * src_w + off_x;
            let mut x = 0;
            while x < dst_w {
                let i0 = (src_row + x) * 3;
                let i1 = (src_row + x + 1) * 3;
                let (y0, cb0, cr0) = rgb_to_ycbcr(rgb[i0], rgb[i0 + 1], rgb[i0 + 2]);
                let (y1, cb1, cr1) = rgb_to_ycbcr(rgb[i1], rgb[i1 + 1], rgb[i1 + 2]);
                // Subsample chroma by averaging the pixel pair.
                let cb = ((cb0 as u16 + cb1 as u16) / 2) as u8;
                let cr = ((cr0 as u16 + cr1 as u16) / 2) as u8;
                let o = x * 2;
                drow[o] = y0;
                drow[o + 1] = cb;
                drow[o + 2] = y1;
                drow[o + 3] = cr;
                x += 2;
            }
        });
}

/// Spatially denoise the chroma of a packed YUYV frame in place, leaving luma
/// untouched.
///
/// Low-light sensor noise shows up most objectionably as coloured speckle
/// (chroma noise), amplified by the CCM's off-diagonal terms. Smoothing only the
/// Cb/Cr samples removes the speckle while keeping luma — hence perceived
/// sharpness — intact. `radius` is the box-blur half-width on the chroma grid
/// and `strength` (0..1) blends the blurred chroma back over the original.
///
/// The chroma is already horizontally subsampled by the 4:2:2 packing, so this
/// operates on the `(w/2) x h` chroma grid: each `[Y0 Cb Y1 Cr]` group is one
/// chroma sample. `radius == 0` or `strength <= 0` is a no-op.
pub fn denoise_chroma_yuyv(buf: &mut [u8], w: usize, h: usize, radius: usize, strength: f64) {
    debug_assert_eq!(buf.len(), w * h * 2);
    debug_assert!(w % 2 == 0);
    if radius == 0 || strength <= 0.0 {
        return;
    }
    let cw = w / 2;
    let mut cb = vec![0u16; cw * h];
    let mut cr = vec![0u16; cw * h];
    for y in 0..h {
        let row = y * w * 2;
        for c in 0..cw {
            cb[y * cw + c] = buf[row + 4 * c + 1] as u16;
            cr[y * cw + c] = buf[row + 4 * c + 3] as u16;
        }
    }
    let cb_b = box_blur_plane(&cb, cw, h, radius);
    let cr_b = box_blur_plane(&cr, cw, h, radius);
    let s = strength.clamp(0.0, 1.0);
    let inv = 1.0 - s;
    for y in 0..h {
        let row = y * w * 2;
        for c in 0..cw {
            let i = y * cw + c;
            let nb = (cb[i] as f64 * inv + cb_b[i] as f64 * s)
                .round()
                .clamp(0.0, 255.0) as u8;
            let nr = (cr[i] as f64 * inv + cr_b[i] as f64 * s)
                .round()
                .clamp(0.0, 255.0) as u8;
            buf[row + 4 * c + 1] = nb;
            buf[row + 4 * c + 3] = nr;
        }
    }
}

/// Separable box blur of a single-channel `w x h` plane (values 0..=255 in
/// `u16`). At the borders the window shrinks (each border pixel averages only the
/// available neighbours), and the mean uses integer truncation.
///
/// Both passes use a sliding-window running sum, so the cost is O(w*h)
/// independent of the radius `r` (a naive window sum is O(w*h*r)). The horizontal
/// pass runs in parallel per row; the vertical pass carries one running sum per
/// column down the rows in a single pass — sequential but still O(w*h), and it
/// keeps CPU use low rather than fanning a tiny workload across all cores.
fn box_blur_plane(src: &[u16], w: usize, h: usize, r: usize) -> Vec<u16> {
    if w == 0 || h == 0 {
        return vec![0u16; w * h];
    }
    // Horizontal pass: each row is an independent sliding window.
    let mut tmp = vec![0u16; w * h];
    tmp.par_chunks_mut(w)
        .zip(src.par_chunks(w))
        .for_each(|(trow, srow)| blur_line_runsum(srow, trow, w, r));

    // Vertical pass: maintain `colsum[x]`, the sum of `tmp[yy*w + x]` over the
    // current vertical window, advancing the window one row at a time.
    let mut out = vec![0u16; w * h];
    let mut colsum = vec![0u32; w];
    let mut hi = r.min(h - 1);
    for yy in 0..=hi {
        let base = yy * w;
        for (x, c) in colsum.iter_mut().enumerate() {
            *c += tmp[base + x] as u32;
        }
    }
    let mut lo = 0usize;
    for y in 0..h {
        let want_hi = (y + r).min(h - 1);
        while hi < want_hi {
            hi += 1;
            let base = hi * w;
            for (x, c) in colsum.iter_mut().enumerate() {
                *c += tmp[base + x] as u32;
            }
        }
        let want_lo = y.saturating_sub(r);
        while lo < want_lo {
            let base = lo * w;
            for (x, c) in colsum.iter_mut().enumerate() {
                *c -= tmp[base + x] as u32;
            }
            lo += 1;
        }
        let n = (want_hi - want_lo + 1) as u32;
        let orow = &mut out[y * w..y * w + w];
        for (o, &c) in orow.iter_mut().zip(colsum.iter()) {
            *o = (c / n) as u16;
        }
    }
    out
}

/// One row (or any contiguous line) box-blurred with a sliding-window running
/// sum: `out[x]` is the integer mean of `src` over `[x-r, x+r]` clamped to
/// `[0, w-1]`. Border windows shrink, matching the divisor to the actual count.
fn blur_line_runsum(src: &[u16], out: &mut [u16], w: usize, r: usize) {
    let mut hi = r.min(w - 1);
    let mut sum: u32 = src[0..=hi].iter().map(|&v| v as u32).sum();
    let mut lo = 0usize;
    for (x, o) in out.iter_mut().enumerate() {
        let want_hi = (x + r).min(w - 1);
        while hi < want_hi {
            hi += 1;
            sum += src[hi] as u32;
        }
        let want_lo = x.saturating_sub(r);
        while lo < want_lo {
            sum -= src[lo] as u32;
            lo += 1;
        }
        *o = (sum / (want_hi - want_lo + 1) as u32) as u16;
    }
}

/// Temporally denoise the luma of a packed YUYV frame in place, blending each
/// pixel with the running history in `prev_y`, leaving chroma untouched.
///
/// Spatial blur cannot remove temporal luma grain at high gain (per-frame random
/// noise), so this averages each Y sample against the previous filtered frame
/// (an IIR low-pass over time), which suppresses grain on a static scene without
/// softening spatial detail. A per-pixel motion gate fades the blend out where
/// the frame changes (`|cur - prev| > motion`), so moving regions take the live
/// pixel and do not ghost.
///
/// `prev_y` holds the filtered luma of the previous frame (`w*h` bytes); it is
/// (re)initialised from the current frame when empty or mismatched, and updated
/// in place every call so the history stays current even when `alpha == 0`.
/// `alpha` (0..1) is the maximum temporal weight of the history.
pub fn temporal_denoise_luma_yuyv(
    buf: &mut [u8],
    prev_y: &mut Vec<u8>,
    w: usize,
    h: usize,
    alpha: f64,
    motion: f64,
) {
    debug_assert_eq!(buf.len(), w * h * 2);
    let n = w * h;

    // First frame (or a size change): seed history, no blend this frame.
    if prev_y.len() != n {
        prev_y.clear();
        prev_y.resize(n, 0);
        prev_y
            .par_chunks_mut(w)
            .zip(buf.par_chunks(w * 2))
            .for_each(|(prow, brow)| {
                for (x, p) in prow.iter_mut().enumerate() {
                    *p = brow[2 * x];
                }
            });
        return;
    }

    let a = alpha.clamp(0.0, 1.0);
    let mthr = motion.max(1.0);
    buf.par_chunks_mut(w * 2)
        .zip(prev_y.par_chunks_mut(w))
        .for_each(|(brow, prow)| {
            for x in 0..w {
                let cur = brow[2 * x] as f64;
                if a <= 0.0 {
                    // Disabled this frame: keep history fresh, leave luma as-is.
                    prow[x] = brow[2 * x];
                    continue;
                }
                let prev = prow[x] as f64;
                let diff = (cur - prev).abs();
                // Motion gate: full weight when still, fading to 0 past `motion`.
                let gate = (1.0 - diff / mthr).clamp(0.0, 1.0);
                let eff = a * gate;
                let y = (cur * (1.0 - eff) + prev * eff).round().clamp(0.0, 255.0) as u8;
                brow[2 * x] = y;
                prow[x] = y; // IIR: history is the filtered output
            }
        });
}

/// Neutral chroma level for 8-bit full-range YCbCr (the Cb/Cr zero point).
const CHROMA_NEUTRAL: f32 = 128.0;

/// Full-range BT.601 (JFIF) RGB -> YCbCr. The WGSL shader (`gpu.rs` `ycbcr`)
/// mirrors these coefficients; keep both in sync.
#[inline]
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cb = CHROMA_NEUTRAL - 0.168_736 * rf - 0.331_264 * gf + 0.5 * bf;
    let cr = CHROMA_NEUTRAL + 0.5 * rf - 0.418_688 * gf - 0.081_312 * bf;
    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A YUYV frame with a per-pixel luma ramp and a checkerboard chroma. After
    /// chroma denoise the luma must be byte-identical and the chroma must move
    /// toward its local mean (128), without touching the frame's byte length.
    #[test]
    fn chroma_denoise_smooths_chroma_keeps_luma() {
        let (w, h) = (8usize, 4usize);
        let mut buf = vec![0u8; w * h * 2];
        for y in 0..h {
            let row = y * w * 2;
            for c in 0..w / 2 {
                buf[row + 4 * c] = (10 * c) as u8; // Y0
                buf[row + 4 * c + 2] = (10 * c + 5) as u8; // Y1
                                                           // Checkerboard chroma around the 128 neutral: 100 / 156.
                let v = if (c + y) % 2 == 0 { 100u8 } else { 156u8 };
                buf[row + 4 * c + 1] = v; // Cb
                buf[row + 4 * c + 3] = 255 - v; // Cr (also 155 / 99)
            }
        }
        let luma_before: Vec<u8> = (0..w * h).map(|p| buf[p * 2]).collect();

        denoise_chroma_yuyv(&mut buf, w, h, 1, 1.0);

        // Luma (every even byte) untouched.
        let luma_after: Vec<u8> = (0..w * h).map(|p| buf[p * 2]).collect();
        assert_eq!(luma_before, luma_after, "luma must be preserved");

        // An interior chroma sample must be pulled toward the 128 neutral.
        let c = 2usize; // interior column
        let yq = 1usize; // interior row
        let cb = buf[yq * w * 2 + 4 * c + 1];
        assert!(
            (100..=156).contains(&cb) && (cb as i32 - 128).abs() < (100i32 - 128).abs(),
            "chroma {cb} should move toward 128 from the 100/156 extremes"
        );
    }

    /// Radius 0 (or zero strength) is a no-op.
    #[test]
    fn chroma_denoise_zero_radius_is_noop() {
        let (w, h) = (4usize, 2usize);
        let mut buf: Vec<u8> = (0..(w * h * 2) as u8).collect();
        let orig = buf.clone();
        denoise_chroma_yuyv(&mut buf, w, h, 0, 1.0);
        assert_eq!(buf, orig);
        denoise_chroma_yuyv(&mut buf, w, h, 2, 0.0);
        assert_eq!(buf, orig);
    }

    /// Naive O(w*h*r) reference: the box blur written as an explicit window sum,
    /// kept only to pin the optimised running-sum [`box_blur_plane`] to identical
    /// output (same border-shrinking and integer-truncated mean).
    fn box_blur_naive(src: &[u16], w: usize, h: usize, r: usize) -> Vec<u16> {
        let mut tmp = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w - 1);
                let sum: u32 = (x0..=x1).map(|xx| src[y * w + xx] as u32).sum();
                tmp[y * w + x] = (sum / (x1 - x0 + 1) as u32) as u16;
            }
        }
        let mut out = vec![0u16; w * h];
        for y in 0..h {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let n = (y1 - y0 + 1) as u32;
            for x in 0..w {
                let sum: u32 = (y0..=y1).map(|yy| tmp[yy * w + x] as u32).sum();
                out[y * w + x] = (sum / n) as u16;
            }
        }
        out
    }

    /// The optimised running-sum box blur must be byte-identical to the naive
    /// reference across radii, sizes, and the degenerate single-row/column cases.
    #[test]
    fn box_blur_runsum_matches_naive() {
        // Deterministic pseudo-random plane (no rand dependency): a simple LCG.
        let make = |w: usize, h: usize| -> Vec<u16> {
            let mut s = 0x1234_5678u32;
            (0..w * h)
                .map(|_| {
                    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((s >> 16) % 256) as u16
                })
                .collect()
        };
        for &(w, h) in &[(1, 1), (1, 7), (7, 1), (8, 4), (17, 13), (32, 32)] {
            let src = make(w, h);
            for r in 0..=6 {
                assert_eq!(
                    box_blur_plane(&src, w, h, r),
                    box_blur_naive(&src, w, h, r),
                    "box blur mismatch at {w}x{h} r={r}"
                );
            }
        }
    }

    /// Build a single-row YUYV frame with the given per-pixel luma; chroma is set
    /// to a constant marker so the test can confirm it is never touched.
    fn yuyv_row(luma: &[u8]) -> Vec<u8> {
        let mut buf = vec![0u8; luma.len() * 2];
        for (p, &y) in luma.iter().enumerate() {
            buf[2 * p] = y;
            buf[2 * p + 1] = 64; // Cb/Cr marker (also covers Cr at odd pairs)
        }
        buf
    }

    /// The first call seeds history without blending; a later static frame is
    /// pulled toward the history; chroma is preserved throughout.
    #[test]
    fn temporal_denoise_blends_luma_keeps_chroma() {
        let (w, h) = (2usize, 1usize);
        let mut prev = Vec::new();

        // Seed: history := [200, 200], frame unchanged.
        let mut buf = yuyv_row(&[200, 200]);
        temporal_denoise_luma_yuyv(&mut buf, &mut prev, w, h, 0.5, 1000.0);
        assert_eq!(prev, vec![200, 200]);
        assert_eq!(buf, yuyv_row(&[200, 200]));

        // New frame luma 100; alpha 0.5, large motion threshold -> gate ~1.
        // diff=100, gate=1-100/1000=0.9, eff=0.45 -> 100*0.55 + 200*0.45 = 145.
        let mut buf = yuyv_row(&[100, 100]);
        temporal_denoise_luma_yuyv(&mut buf, &mut prev, w, h, 0.5, 1000.0);
        assert_eq!(buf[0], 145);
        assert_eq!(buf[2], 145);
        assert_eq!(prev, vec![145, 145]);
        // Chroma markers intact.
        assert_eq!(buf[1], 64);
        assert_eq!(buf[3], 64);
    }

    /// Motion beyond the threshold disables the blend (gate -> 0), so a fast
    /// change passes through as the live pixel.
    #[test]
    fn temporal_denoise_motion_gate_passes_live_pixel() {
        let (w, h) = (1usize, 1usize);
        let mut prev = Vec::new();
        let mut buf = yuyv_row(&[10]);
        temporal_denoise_luma_yuyv(&mut buf, &mut prev, w, h, 0.8, 8.0); // seed -> 10
        let mut buf = yuyv_row(&[200]); // diff 190 >> motion 8 -> gate 0
        temporal_denoise_luma_yuyv(&mut buf, &mut prev, w, h, 0.8, 8.0);
        assert_eq!(buf[0], 200);
        assert_eq!(prev, vec![200]);
    }

    /// alpha 0 keeps history current but leaves luma unchanged.
    #[test]
    fn temporal_denoise_alpha_zero_refreshes_history_only() {
        let (w, h) = (2usize, 1usize);
        let mut prev = Vec::new();
        let mut buf = yuyv_row(&[50, 60]);
        temporal_denoise_luma_yuyv(&mut buf, &mut prev, w, h, 0.0, 8.0); // seed
        let mut buf = yuyv_row(&[80, 90]);
        temporal_denoise_luma_yuyv(&mut buf, &mut prev, w, h, 0.0, 8.0);
        assert_eq!(buf, yuyv_row(&[80, 90])); // luma untouched
        assert_eq!(prev, vec![80, 90]); // history advanced
    }
}
