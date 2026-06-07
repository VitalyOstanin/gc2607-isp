//! GPU backend (wgpu / Vulkan): the per-pixel ISP stages run as compute shaders
//! on the integrated GPU, while AWB/CCM estimation stays on the CPU (it needs a
//! sort over downsampled pixels and runs only every `awb_interval` frames).
//!
//! Division of labour, per frame:
//!   - CPU: nothing on most frames; every `awb_interval` frames it builds the
//!     half-res Bayer planes and runs `pipeline::estimate` (gains, CCM, LSC src).
//!     When the chosen light source changes, the four resized LSC grids are
//!     rebuilt on the CPU and uploaded once.
//!   - GPU pass 1 (`build_cfa`): unpack SGRBG10 -> black level -> LSC -> per
//!     channel white balance, into a full-res CFA buffer.
//!   - GPU pass 2 (`render_pack`): Malvar-He-Cutler debayer -> CCM -> sRGB gamma
//!     -> full-range BT.601 YCbCr -> packed YUYV, with the centre crop applied,
//!     straight into the output buffer (no CPU `rgb_to_yuyv_crop`).
//!
//! Only the full-res MHC path is implemented on the GPU (the heavy case the GPU
//! is meant to offload); half-res stays on the CPU. The arithmetic mirrors
//! `pipeline::mhc_rgb_interior` and `output::rgb_to_yuyv_crop`; results match the
//! CPU MHC path within rounding (validated by `tests/gpu.rs`), not bit-for-bit
//! (the GPU uses `pow` where the CPU MHC path uses a LUT).

use std::io;

use crate::pipeline::{self, Estimate};
use crate::raw::{BLACK, H, MAXLIN, RawFrame, STRIDE_SAMPLES, W};

/// Output (centre-cropped) size for the GPU MHC path: 1920x1080 from 1928x1088.
pub const OUT_W: usize = 1920;
pub const OUT_H: usize = 1080;

const WW: usize = W / 2; // 964
const HH: usize = H / 2; // 544

/// Uniform block shared by both compute passes. Laid out as `vec4`-aligned
/// groups to match the WGSL `Params` struct (std140 / 16-byte alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 4],     // w, h, ww, hh
    out_dims: [u32; 4], // dst_w, dst_h, off_x, off_y
    misc: [u32; 4],     // stride, groups_per_row, _, _
    consts: [f32; 4],   // black, inv_maxlin, _, _
    gains: [f32; 4],    // R, G, B, _
    ccm0: [f32; 4],     // ccm[0..3]
    ccm1: [f32; 4],     // ccm[3..6]
    ccm2: [f32; 4],     // ccm[6..9]
}

const SHADER: &str = r#"
struct Params {
    dims: vec4<u32>,
    out_dims: vec4<u32>,
    misc: vec4<u32>,
    consts: vec4<f32>,
    gains: vec4<f32>,
    ccm0: vec4<f32>,
    ccm1: vec4<f32>,
    ccm2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> raw: array<u32>;
@group(0) @binding(2) var<storage, read> g_gr: array<f32>;
@group(0) @binding(3) var<storage, read> g_r: array<f32>;
@group(0) @binding(4) var<storage, read> g_b: array<f32>;
@group(0) @binding(5) var<storage, read> g_gb: array<f32>;
@group(0) @binding(6) var<storage, read_write> cfa: array<f32>;
@group(0) @binding(7) var<storage, read_write> yuyv: array<u32>;

// Pass 1: unpack + black level + LSC + white balance into the full-res CFA.
@compute @workgroup_size(64)
fn build_cfa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = P.dims.x;
    let h = P.dims.y;
    let ww = P.dims.z;
    let idx = gid.x;
    if (idx >= w * h) { return; }
    let y = idx / w;
    let x = idx % w;
    let stride = P.misc.x;
    let s = y * stride + x;
    let word = raw[s >> 1u];
    var sample: u32;
    if ((s & 1u) == 0u) { sample = word & 0xFFFFu; } else { sample = word >> 16u; }
    let v = max(f32(sample) - P.consts.x, 0.0);
    let iy = y / 2u;
    let ix = x / 2u;
    let gi = iy * ww + ix;
    let yp = y & 1u;
    let xp = x & 1u;
    var lsc: f32;
    var gain: f32;
    if (yp == 0u && xp == 0u) {        // Gr
        lsc = g_gr[gi]; gain = P.gains.y;
    } else if (yp == 0u && xp == 1u) { // R
        lsc = g_r[gi]; gain = P.gains.x;
    } else if (yp == 1u && xp == 0u) { // B
        lsc = g_b[gi]; gain = P.gains.z;
    } else {                           // Gb
        lsc = g_gb[gi]; gain = P.gains.y;
    }
    cfa[idx] = v * lsc * gain;
}

fn srgb(x: f32) -> f32 {
    let v = clamp(x, 0.0, 1.0);
    if (v <= 0.0031308) { return 12.92 * v; }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

// Malvar-He-Cutler interpolation at an interior pixel (GRBG). Mirrors
// pipeline::mhc_rgb_interior; the centre crop guarantees >=4 px from each edge,
// so no bounds clamping is needed. Returns linear (R, G, B) in CFA scale.
fn mhc(y: u32, x: u32) -> vec3<f32> {
    let w = P.dims.x;
    let i = y * w + x;
    let c = cfa[i];
    let up = cfa[i - w];
    let dn = cfa[i + w];
    let lf = cfa[i - 1u];
    let rt = cfa[i + 1u];
    let uu = cfa[i - 2u * w];
    let dd = cfa[i + 2u * w];
    let ll = cfa[i - 2u];
    let rr = cfa[i + 2u];
    let diag = cfa[i - w - 1u] + cfa[i - w + 1u] + cfa[i + w - 1u] + cfa[i + w + 1u];

    let g_at_rb = (4.0 * c + 2.0 * (up + dn + lf + rt) - (uu + dd + ll + rr)) / 8.0;
    let at_g_hrow = (5.0 * c + 4.0 * (lf + rt) - (ll + rr) - diag + 0.5 * (uu + dd)) / 8.0;
    let at_g_vcol = (5.0 * c + 4.0 * (up + dn) - (uu + dd) - diag + 0.5 * (ll + rr)) / 8.0;
    let at_diag = (6.0 * c + 2.0 * diag - 1.5 * (uu + dd + ll + rr)) / 8.0;

    let yp = y & 1u;
    let xp = x & 1u;
    if (yp == 0u && xp == 0u) { return vec3<f32>(at_g_hrow, c, at_g_vcol); }      // Gr
    else if (yp == 0u && xp == 1u) { return vec3<f32>(c, g_at_rb, at_diag); }     // R
    else if (yp == 1u && xp == 0u) { return vec3<f32>(at_diag, g_at_rb, c); }     // B
    else { return vec3<f32>(at_g_vcol, c, at_g_hrow); }                            // Gb
}

// Linear CFA-scale RGB -> CCM -> sRGB gamma -> 0..255 (rounded).
fn to_rgb8(lin: vec3<f32>) -> vec3<f32> {
    let s = lin * P.consts.y; // * inv_maxlin
    let cr = P.ccm0.x * s.x + P.ccm0.y * s.y + P.ccm0.z * s.z;
    let cg = P.ccm1.x * s.x + P.ccm1.y * s.y + P.ccm1.z * s.z;
    let cb = P.ccm2.x * s.x + P.ccm2.y * s.y + P.ccm2.z * s.z;
    return vec3<f32>(
        floor(srgb(cr) * 255.0 + 0.5),
        floor(srgb(cg) * 255.0 + 0.5),
        floor(srgb(cb) * 255.0 + 0.5),
    );
}

// Full-range BT.601 (JFIF) RGB(0..255) -> YCbCr(0..255), rounded and clamped.
fn ycbcr(c: vec3<f32>) -> vec3<f32> {
    let yy = 0.299 * c.x + 0.587 * c.y + 0.114 * c.z;
    let cb = 128.0 - 0.168736 * c.x - 0.331264 * c.y + 0.5 * c.z;
    let cr = 128.0 + 0.5 * c.x - 0.418688 * c.y - 0.081312 * c.z;
    return vec3<f32>(
        clamp(floor(yy + 0.5), 0.0, 255.0),
        clamp(floor(cb + 0.5), 0.0, 255.0),
        clamp(floor(cr + 0.5), 0.0, 255.0),
    );
}

// Pass 2: one thread per output pixel pair -> MHC + colour + YUYV pack.
@compute @workgroup_size(64)
fn render_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dst_h = P.out_dims.y;
    let off_x = P.out_dims.z;
    let off_y = P.out_dims.w;
    let groups_per_row = P.misc.y; // dst_w / 2
    let gidx = gid.x;
    if (gidx >= groups_per_row * dst_h) { return; }
    let gy = gidx / groups_per_row;
    let gx = gidx % groups_per_row;
    let sy = off_y + gy;
    let sx0 = off_x + gx * 2u;
    let sx1 = sx0 + 1u;

    let yc0 = ycbcr(to_rgb8(mhc(sy, sx0)));
    let yc1 = ycbcr(to_rgb8(mhc(sy, sx1)));
    let y0 = u32(yc0.x);
    let y1 = u32(yc1.x);
    let cb = (u32(yc0.y) + u32(yc1.y)) / 2u;
    let cr = (u32(yc0.z) + u32(yc1.z)) / 2u;
    yuyv[gidx] = y0 | (cb << 8u) | (y1 << 16u) | (cr << 24u);
}
"#;

/// GPU-resident ISP for the live webcam path. Holds the device, both compute
/// pipelines, and all persistent buffers; per frame it uploads the raw bytes,
/// dispatches the two passes, and reads the packed YUYV back into a reused host
/// buffer. AWB/CCM estimation runs on the CPU every `awb_interval` frames.
pub struct GpuProcessor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cfa_pipeline: wgpu::ComputePipeline,
    pack_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    raw_buf: wgpu::Buffer,
    grid_bufs: [wgpu::Buffer; 4],
    yuyv_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,

    interval: u64,
    frame: u64,
    est: Option<Estimate>,
    grids_ls: Option<usize>,

    yuyv_host: Vec<u8>,
    raw_needed: usize,
    yuyv_bytes: usize,
}

impl GpuProcessor {
    /// Initialise the GPU backend (Vulkan / mesa ANV). Re-estimates AWB/CCM every
    /// `awb_interval` frames (clamped to >= 1; the first frame always estimates).
    pub fn new(awb_interval: u64) -> Result<Self, String> {
        pollster::block_on(Self::new_async(awb_interval))
    }

    async fn new_async(awb_interval: u64) -> Result<Self, String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .map_err(|e| format!("no Vulkan adapter: {e}"))?;
        let info = adapter.get_info();
        eprintln!("gpu: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("gc2607-isp"),
                required_features: wgpu::Features::empty(),
                // The pipeline binds 7 storage buffers in one compute stage; the
                // adapter's own limits cover that on ANV (downlevel defaults cap
                // storage buffers at 4, which is too few).
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("request_device: {e}"))?;

        let raw_needed = H * STRIDE_SAMPLES * 2;
        let cfa_bytes = (W * H * 4) as u64;
        let grid_bytes = (HH * WW * 4) as u64;
        let yuyv_bytes = OUT_W * OUT_H * 2;

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let raw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw"),
            size: raw_needed as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let make_grid = |name: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(name),
                size: grid_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let grid_bufs = [
            make_grid("g_gr"),
            make_grid("g_r"),
            make_grid("g_b"),
            make_grid("g_gb"),
        ];
        let cfa_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cfa"),
            size: cfa_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let yuyv_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("yuyv"),
            size: yuyv_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: yuyv_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("isp"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let storage_ro = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let storage_rw = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("isp-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_ro(1),
                storage_ro(2),
                storage_ro(3),
                storage_ro(4),
                storage_ro(5),
                storage_rw(6),
                storage_rw(7),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("isp-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: raw_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: grid_bufs[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: grid_bufs[1].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: grid_bufs[2].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: grid_bufs[3].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: cfa_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: yuyv_buf.as_entire_binding() },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("isp-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let cfa_pipeline = make_pipeline("build_cfa");
        let pack_pipeline = make_pipeline("render_pack");

        Ok(GpuProcessor {
            device,
            queue,
            cfa_pipeline,
            pack_pipeline,
            bind_group,
            params_buf,
            raw_buf,
            grid_bufs,
            yuyv_buf,
            staging_buf,
            interval: awb_interval.max(1),
            frame: 0,
            est: None,
            grids_ls: None,
            yuyv_host: vec![0u8; yuyv_bytes],
            raw_needed,
            yuyv_bytes,
        })
    }

    /// Output frame dimensions (YUYV).
    pub fn out_dims(&self) -> (usize, usize) {
        (OUT_W, OUT_H)
    }

    /// The most recent scene estimate, if any frame has been processed.
    pub fn estimate(&self) -> Option<&Estimate> {
        self.est.as_ref()
    }

    /// Re-estimate AWB/CCM on the CPU from `bytes`, and upload fresh LSC grids if
    /// the chosen light source changed.
    fn reestimate(&mut self, bytes: &[u8]) -> io::Result<()> {
        let frame = RawFrame::from_bytes(bytes)?;
        let planes = pipeline::bayer_planes(&frame);
        let est = pipeline::estimate(&planes);
        if self.grids_ls != Some(est.ls) {
            let grids = pipeline::build_grids(est.ls, HH, WW);
            let upload = |buf: &wgpu::Buffer, g: &[f64]| {
                let f: Vec<f32> = g.iter().map(|&v| v as f32).collect();
                self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&f));
            };
            upload(&self.grid_bufs[0], &grids.g_gr);
            upload(&self.grid_bufs[1], &grids.g_r);
            upload(&self.grid_bufs[2], &grids.g_b);
            upload(&self.grid_bufs[3], &grids.g_gb);
            self.grids_ls = Some(est.ls);
        }
        self.est = Some(est);
        Ok(())
    }

    /// Pack the current estimate into the uniform block and upload it.
    fn upload_params(&self) {
        let est = self.est.as_ref().unwrap();
        let g = est.gains;
        let c = est.ccm;
        let params = Params {
            dims: [W as u32, H as u32, WW as u32, HH as u32],
            out_dims: [OUT_W as u32, OUT_H as u32, ((W - OUT_W) / 2) as u32, ((H - OUT_H) / 2) as u32],
            misc: [STRIDE_SAMPLES as u32, (OUT_W / 2) as u32, 0, 0],
            consts: [BLACK, 1.0 / MAXLIN, 0.0, 0.0],
            gains: [g[0] as f32, g[1] as f32, g[2] as f32, 0.0],
            ccm0: [c[0] as f32, c[1] as f32, c[2] as f32, 0.0],
            ccm1: [c[3] as f32, c[4] as f32, c[5] as f32, 0.0],
            ccm2: [c[6] as f32, c[7] as f32, c[8] as f32, 0.0],
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));
    }

    /// Process one raw SGRBG10 frame; returns the packed YUYV bytes (length
    /// `OUT_W*OUT_H*2`), borrowing a reused host buffer (valid until the next call).
    pub fn process(&mut self, bytes: &[u8]) -> io::Result<&[u8]> {
        if bytes.len() < self.raw_needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("raw too small: {} bytes, need >= {}", bytes.len(), self.raw_needed),
            ));
        }

        // AWB/CCM on the CPU, occasionally.
        if self.est.is_none() || self.frame % self.interval == 0 {
            self.reestimate(bytes)?;
            self.upload_params();
        }

        // Upload this frame's raw and dispatch the two passes.
        self.queue.write_buffer(&self.raw_buf, 0, &bytes[..self.raw_needed]);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("build_cfa"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cfa_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = ((W * H) as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("render_pack"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pack_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            let groups = (((OUT_W / 2) * OUT_H) as u32).div_ceil(64);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.yuyv_buf, 0, &self.staging_buf, 0, self.yuyv_bytes as u64);
        self.queue.submit(Some(encoder.finish()));

        // Read the packed YUYV back.
        let slice = self.staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| io::Error::other(format!("gpu poll: {e}")))?;
        rx.recv()
            .map_err(|e| io::Error::other(format!("gpu map channel: {e}")))?
            .map_err(|e| io::Error::other(format!("gpu map: {e}")))?;
        {
            let data = slice.get_mapped_range();
            self.yuyv_host.copy_from_slice(&data);
        }
        self.staging_buf.unmap();

        self.frame = self.frame.wrapping_add(1);
        Ok(&self.yuyv_host)
    }
}
