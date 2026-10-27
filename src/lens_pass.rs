//! Fullscreen lens mud. The soil solver is not involved: a puddle is just
//! "tires in water, moving", and a landing is the impact spray counter.

use silt::lens::LensMud;
use silt::world::{CameraPose, World};

pub struct LensPass {
    mud: LensMud,
    pipe: wgpu::RenderPipeline,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
    last_time: Option<f32>,
}

impl LensPass {
    pub fn new(
        device: &wgpu::Device,
        color: wgpu::TextureFormat,
        depth: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lens.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<silt::lens::GpuLens>() as u64,
                    ),
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("lens-pipeline"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens-uniform"),
            size: std::mem::size_of::<silt::lens::GpuLens>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens-bg"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            }],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("lens"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_lens"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_lens"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        Self {
            mud: LensMud::new(),
            pipe,
            buf,
            bg,
            last_time: None,
        }
    }

    /// Step from the world clock and, on the chase camera, draw drops and streaks.
    pub fn composite(
        &mut self,
        pass: &mut wgpu::RenderPass<'_>,
        queue: &wgpu::Queue,
        world: &World,
        cam: &CameraPose,
        aspect: f32,
        show: bool,
    ) {
        let dt = match self.last_time {
            Some(t) if world.time + 1.0e-4 >= t => (world.time - t).min(0.5),
            _ => 0.0,
        };
        self.last_time = Some(world.time);
        let (wet, speed) = wet_and_speed(world);
        self.mud.sync(dt, wet, speed, world.telemetry.impact_sprays);
        // The overhead view stays clean. L hides the coat; C stays the camera cycle.
        if !show || cam.ortho_width.is_some() || self.mud.mass() < 0.05 {
            return;
        }
        let gpu = self.mud.pack(aspect);
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&gpu));
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn wet_and_speed(world: &World) -> (u32, f32) {
    let mut wet = 0u32;
    let mut speed = 0.0f32;
    if world.vehicle.wheels.iter().any(|w| w.in_water) {
        wet += world.vehicle.wheels.iter().filter(|w| w.in_water).count() as u32;
        speed = speed.max(world.vehicle.speed());
    }
    if world.bike.wheels.iter().any(|w| w.in_water) {
        wet += world.bike.wheels.iter().filter(|w| w.in_water).count() as u32;
        speed = speed.max(world.bike.speed());
    }
    (wet, speed)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use glam::Vec3;
    use silt::world::{demo_drive, demo_drive_bike, CameraPose, World};

    fn lavapipe() -> bool {
        std::path::Path::new("/usr/share/vulkan/icd.d/lvp_icd.json").exists()
    }

    fn device() -> (wgpu::Device, wgpu::Queue, wgpu::TextureFormat) {
        crate::render::prepare_software_vulkan();
        let (_instance, adapter, device, queue) = crate::render::open_adapter(true, None);
        let depth = crate::render::gpu_formats(&adapter).depth;
        (device, queue, depth)
    }

    #[test]
    fn lens_paints_tan_mud_and_leaves_the_rest_of_the_glass() {
        if !lavapipe() {
            eprintln!("skip: lavapipe is not installed");
            return;
        }
        let (device, queue, depth) = device();
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut lens = LensPass::new(&device, wgpu::TextureFormat::Rgba8Unorm, depth);
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("lens pipeline: {err}");
        }
        lens.mud.sync(1.0 / 60.0, 2, 4.0, 0);
        assert!(lens.mud.mud_runs() >= 4);

        let width = 320u32;
        let height = 180u32;
        let color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lens-test-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lens-test-depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&Default::default());
        let padded = (width * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens-test-read"),
            size: padded as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lens-test"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("lens-test"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 30.0 / 255.0,
                            g: 70.0 / 255.0,
                            b: 140.0 / 255.0,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let gpu = lens.mud.pack(width as f32 / height as f32);
            queue.write_buffer(&lens.buf, 0, bytemuck::bytes_of(&gpu));
            pass.set_pipeline(&lens.pipe);
            pass.set_bind_group(0, &lens.bg, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));
        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map").expect("map buffer");
        let data = slice.get_mapped_range();
        let mut tan = 0u32;
        let mut blue = 0u32;
        let mut spec = 0u32;
        let row = (width * 4) as usize;
        let stride = padded as usize;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let i = y * stride + x * 4;
                let (r, g, b) = (data[i], data[i + 1], data[i + 2]);
                if r > 120 && g > 55 && r > b.saturating_add(25) && g > b {
                    tan += 1;
                } else if r < 45 && b > 100 && b > r {
                    blue += 1;
                }
                if r > 200 && g > 190 && b > 180 {
                    spec += 1;
                }
            }
        }
        let png = std::env::temp_dir().join("silt-lens-blue.png");
        let mut tight = Vec::with_capacity(row * height as usize);
        for y in 0..height as usize {
            tight.extend_from_slice(&data[y * stride..y * stride + row]);
        }
        drop(data);
        readback.unmap();
        crate::render::write_png(&png, width, height, &tight).expect("png");
        assert!(tan > 40, "tan mud pixels {tan} (see {})", png.display());
        assert!(
            blue > width * height / 3,
            "clear glass pixels {blue} (see {})",
            png.display()
        );
        assert!(spec >= 1, "catchlights {spec} (see {})", png.display());
    }

    #[test]
    fn chase_camera_after_a_puddle_shows_lens_mud() {
        if !lavapipe() {
            eprintln!("skip: lavapipe is not installed");
            return;
        }
        crate::render::prepare_software_vulkan();
        let (_instance, adapter, device, queue) = crate::render::open_adapter(true, None);
        let formats = crate::render::gpu_formats(&adapter);
        let mut renderer =
            crate::render::Renderer::new(device, queue, adapter.get_info().name, 640, 360, formats);
        let mut world = World::demo();
        let mut hit = false;
        while world.time < 8.0 {
            let bike = demo_drive_bike(&world);
            world.step_bike(1.0 / 60.0, &bike);
            let input = demo_drive(&world);
            world.step(1.0 / 60.0, &input);
            let wet =
                world.vehicle.wheels.iter().any(|w| w.in_water) && world.vehicle.speed() > 0.4;
            if wet && world.telemetry.splash_bursts > 0 {
                hit = true;
                break;
            }
        }
        assert!(hit, "the lap should drive into a puddle, t={}", world.time);
        let rgba = renderer
            .draw(&world, chase_pose(&world), None, true)
            .expect("puddle readback");
        assert_eq!(rgba.len(), 640 * 360 * 4);
        let path = std::env::temp_dir().join("silt-lens-chase.png");
        crate::render::write_png(&path, 640, 360, &rgba).expect("png");
        eprintln!(
            "lens chase t={:.2} splashes {} {}",
            world.time,
            world.telemetry.splash_bursts,
            path.display()
        );
    }

    fn chase_pose(world: &World) -> CameraPose {
        let yaw = world.vehicle.yaw;
        let pos = world.vehicle.pos;
        let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        CameraPose {
            eye: pos - forward * 7.4 + Vec3::Y * 2.7 + right * 0.85,
            target: pos + forward * 1.8 + Vec3::Y * 0.55,
            ortho_width: None,
        }
    }
}
