//! Minimal V4L2 capture from the loopback node, for the tuning window's live
//! view. MMAP streaming with three buffers; frames are copied out so the
//! buffers go straight back to the queue.

use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::num::NonZeroUsize;
use std::os::fd::{AsFd, AsRawFd};
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use nix::errno::Errno;
use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};

use crate::output::{
    V4l2Buffer, V4l2Format, V4l2RequestBuffers, V4L2_MEMORY_MMAP, V4L2_QUANTIZATION_FULL_RANGE,
};

const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;

nix::ioctl_readwrite!(vidioc_g_fmt, b'V', 4, V4l2Format);
nix::ioctl_readwrite!(vidioc_reqbufs, b'V', 8, V4l2RequestBuffers);
nix::ioctl_readwrite!(vidioc_querybuf, b'V', 9, V4l2Buffer);
nix::ioctl_readwrite!(vidioc_qbuf, b'V', 15, V4l2Buffer);
nix::ioctl_readwrite!(vidioc_dqbuf, b'V', 17, V4l2Buffer);
nix::ioctl_write_ptr!(vidioc_streamon, b'V', 18, i32);
nix::ioctl_write_ptr!(vidioc_streamoff, b'V', 19, i32);

/// Why the camera could not be opened for preview.
#[derive(Debug)]
pub enum CaptureError {
    /// Another program (a call) holds the camera.
    Busy,
    Other(String),
}

fn map_err(e: Errno, what: &str) -> CaptureError {
    if e == Errno::EBUSY {
        CaptureError::Busy
    } else {
        CaptureError::Other(format!("{what}: {e}"))
    }
}

/// One captured frame (YUYV) and how to read it.
#[derive(Clone)]
pub struct Frame {
    pub yuyv: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub limited: bool,
}

struct Stream {
    file: File,
    maps: Vec<(NonNull<c_void>, usize)>,
    width: usize,
    height: usize,
    limited: bool,
}

// SAFETY: the mapped pointers are only touched by the owning capture thread.
unsafe impl Send for Stream {}

impl Stream {
    fn open(path: &Path) -> Result<Self, CaptureError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| {
                if e.raw_os_error() == Some(Errno::EBUSY as i32) {
                    CaptureError::Busy
                } else {
                    CaptureError::Other(format!("open {}: {e}", path.display()))
                }
            })?;
        let fd = file.as_raw_fd();
        let mut fmt = V4l2Format::zeroed(V4L2_BUF_TYPE_VIDEO_CAPTURE);
        // SAFETY: correctly-sized v4l2_format; fd is open.
        unsafe { vidioc_g_fmt(fd, &mut fmt) }.map_err(|e| map_err(e, "G_FMT"))?;
        let (width, height) = (fmt.pix.width as usize, fmt.pix.height as usize);
        let limited = fmt.pix.quantization != V4L2_QUANTIZATION_FULL_RANGE;
        let mut req = V4l2RequestBuffers {
            count: 3,
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        // SAFETY: correctly-sized v4l2_requestbuffers.
        unsafe { vidioc_reqbufs(fd, &mut req) }.map_err(|e| map_err(e, "REQBUFS"))?;
        let mut stream = Stream {
            file,
            maps: Vec::new(),
            width,
            height,
            limited,
        };
        for index in 0..req.count {
            let mut buf = V4l2Buffer {
                index,
                type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
                memory: V4L2_MEMORY_MMAP,
                ..Default::default()
            };
            // SAFETY: correctly-sized v4l2_buffer.
            unsafe { vidioc_querybuf(fd, &mut buf) }.map_err(|e| map_err(e, "QUERYBUF"))?;
            let len = NonZeroUsize::new(buf.length as usize)
                .ok_or_else(|| CaptureError::Other("zero-length buffer".into()))?;
            // SAFETY: map the driver buffer at its reported offset; the file
            // stays open while `Stream` lives, and `Drop` unmaps it.
            let ptr = unsafe {
                mmap(
                    None,
                    len,
                    ProtFlags::PROT_READ,
                    MapFlags::MAP_SHARED,
                    stream.file.as_fd(),
                    buf.m as i64,
                )
            }
            .map_err(|e| map_err(e, "mmap"))?;
            stream.maps.push((ptr, len.get()));
            // SAFETY: as above.
            unsafe { vidioc_qbuf(fd, &mut buf) }.map_err(|e| map_err(e, "QBUF"))?;
        }
        let ty = V4L2_BUF_TYPE_VIDEO_CAPTURE as i32;
        // SAFETY: pointer to the buffer type, as the ioctl expects.
        unsafe { vidioc_streamon(fd, &ty) }.map_err(|e| map_err(e, "STREAMON"))?;
        Ok(stream)
    }

    /// Wait up to `timeout_ms` for a frame; `Ok(None)` on timeout.
    fn next(&mut self, timeout_ms: u16) -> Result<Option<Frame>, CaptureError> {
        use nix::poll::{poll, PollFd, PollFlags, PollTimeout};
        let mut fds = [PollFd::new(self.file.as_fd(), PollFlags::POLLIN)];
        let n = poll(&mut fds, PollTimeout::from(timeout_ms)).map_err(|e| map_err(e, "poll"))?;
        if n == 0 {
            return Ok(None);
        }
        let fd = self.file.as_raw_fd();
        let mut buf = V4l2Buffer {
            type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory: V4L2_MEMORY_MMAP,
            ..Default::default()
        };
        // SAFETY: correctly-sized v4l2_buffer.
        unsafe { vidioc_dqbuf(fd, &mut buf) }.map_err(|e| map_err(e, "DQBUF"))?;
        let (ptr, len) = *self
            .maps
            .get(buf.index as usize)
            .ok_or_else(|| CaptureError::Other(format!("buffer index {}", buf.index)))?;
        let used = (buf.bytesused as usize).min(len);
        // SAFETY: the mapping is `len` bytes long and the driver handed the
        // buffer to us until the QBUF below.
        let yuyv = unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const u8, used) }.to_vec();
        // SAFETY: returning the same buffer to the queue.
        unsafe { vidioc_qbuf(fd, &mut buf) }.map_err(|e| map_err(e, "QBUF"))?;
        Ok(Some(Frame {
            yuyv,
            width: self.width,
            height: self.height,
            limited: self.limited,
        }))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let fd = self.file.as_raw_fd();
        let ty = V4L2_BUF_TYPE_VIDEO_CAPTURE as i32;
        // SAFETY: pointer to the buffer type; errors are irrelevant on close.
        let _ = unsafe { vidioc_streamoff(fd, &ty) };
        for &(ptr, len) in &self.maps {
            // SAFETY: each pointer came from `mmap` with this length.
            let _ = unsafe { munmap(ptr, len) };
        }
    }
}

/// A running preview: a thread that keeps the newest frame. Dropping it stops
/// the thread and closes the device within one poll interval (100 ms).
pub struct Preview {
    stop: Arc<AtomicBool>,
    latest: Arc<Mutex<Option<Frame>>>,
    error: Arc<Mutex<Option<String>>>,
    thread: Option<JoinHandle<()>>,
}

impl Preview {
    pub fn start(path: &Path) -> Result<Preview, CaptureError> {
        let mut stream = Stream::open(path)?;
        let stop = Arc::new(AtomicBool::new(false));
        let latest = Arc::new(Mutex::new(None));
        let error = Arc::new(Mutex::new(None));
        let (s, l, e) = (stop.clone(), latest.clone(), error.clone());
        let thread = std::thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                match stream.next(100) {
                    Ok(Some(f)) => *l.lock().unwrap() = Some(f),
                    Ok(None) => {}
                    Err(err) => {
                        *e.lock().unwrap() = Some(format!("{err:?}"));
                        break;
                    }
                }
            }
        });
        Ok(Preview {
            stop,
            latest,
            error,
            thread: Some(thread),
        })
    }

    /// The newest frame since the last call, if any.
    pub fn take(&self) -> Option<Frame> {
        self.latest.lock().unwrap().take()
    }

    /// A capture error that ended the thread, if any.
    pub fn error(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }
}

impl Drop for Preview {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}
