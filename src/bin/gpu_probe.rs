//! Minimal wgpu/Vulkan probe: pick a compute adapter, print its info, and run a
//! trivial compute shader (multiply an array by 2) to confirm the GPU path works
//! headless on this machine before building the real ISP shaders.

fn main() {
    pollster::block_on(run());
}

async fn run() {
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
        .expect("no Vulkan compute adapter found");

    let info = adapter.get_info();
    println!(
        "adapter: {} ({:?}, {:?}) driver: {} {}",
        info.name, info.device_type, info.backend, info.driver, info.driver_info
    );

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("gpu-probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })
        .await
        .expect("request_device");

    let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let bytes = bytemuck::cast_slice(&input);
    let size = bytes.len() as u64;

    use wgpu::util::DeviceExt;
    let storage = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("storage"),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("double"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < arrayLength(&data)) { data[i] = data[i] * 2.0; }
}
"#
            .into(),
        ),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("double"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage.as_entire_binding(),
        }],
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage, 0, &readback, 0, size);
    queue.submit(Some(encoder.finish()));

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("poll");

    let data = slice.get_mapped_range();
    let out: &[f32] = bytemuck::cast_slice(&data);
    println!("compute result: {out:?} (expected [2.0, 4.0, 6.0, 8.0])");
    assert_eq!(out, &[2.0, 4.0, 6.0, 8.0]);
    println!("OK: GPU compute works");
}
