//! Fullscreen lens water. The soil solver is not involved: a puddle is just
//! "tires in water, moving", and a landing is the impact spray counter.
//!
//! Drops refract a copy of the color target, so this pass runs after the
//! main scene pass has closed.

use silt::lens::LensMud;
use silt::world::{CameraPose, World};

pub struct LensPass {
    mud: LensMud,
    pipe: wgpu::RenderPipeline,
    buf: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    samp: wgpu::Sampler,
    bg: wgpu::BindGroup,
    scene: wgpu::Texture,
    scene_w: u32,
    scene_h: u32,
    format: wgpu::TextureFormat,
    last_time: Option<f32>,
}

impl LensPass {
    pub fn new(device: &wgpu::Device, color: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lens"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lens.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<
                            silt::lens::GpuLens,
                        >() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
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
        let samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("lens-scene"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let (scene, bg) = scene_binding(device, &layout, &buf, &samp, color, 1, 1);
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
            depth_stencil: None,
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
            layout,
            samp,
            bg,
            scene,
            scene_w: 1,
            scene_h: 1,
            format: color,
            last_time: None,
        }
    }

    /// Step from the world clock and, on the chase camera, refract drops over the frame.
    pub fn composite(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color: &wgpu::Texture,
        color_view: &wgpu::TextureView,
        world: &World,
        cam: &CameraPose,
        aspect: f32,
    ) {
        let dt = match self.last_time {
            Some(t) if world.time + 1.0e-4 >= t => (world.time - t).min(0.5),
            _ => 0.0,
        };
        self.last_time = Some(world.time);
        let (wet, speed) = wet_and_speed(world);
        self.mud.sync(dt, wet, speed, world.telemetry.impact_sprays);
        // The overhead view stays clean. The lens is the chase camera.
        if cam.ortho_width.is_some() || self.mud.mass() < 0.05 {
            return;
        }
        let gpu = self.mud.pack(aspect);
        self.refract(device, encoder, queue, color, color_view, &gpu);
    }

    fn refract(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color: &wgpu::Texture,
        color_view: &wgpu::TextureView,
        gpu: &silt::lens::GpuLens,
    ) {
        let size = color.size();
        self.ensure_scene(device, size.width, size.height);
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.scene,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(gpu));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("lens"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipe);
        pass.set_bind_group(0, &self.bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn ensure_scene(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.scene_w == width && self.scene_h == height {
            return;
        }
        let (scene, bg) = scene_binding(
            device,
            &self.layout,
            &self.buf,
            &self.samp,
            self.format,
            width,
            height,
        );
        self.scene = scene;
        self.bg = bg;
        self.scene_w = width;
        self.scene_h = height;
    }
}

fn scene_binding(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buf: &wgpu::Buffer,
    samp: &wgpu::Sampler,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let scene = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("lens-scene"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = scene.create_view(&Default::default());
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("lens-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(samp),
            },
        ],
    });
    (scene, bg)
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

    fn device() -> (wgpu::Device, wgpu::Queue) {
        crate::render::prepare_software_vulkan();
        let (_instance, _adapter, device, queue) = crate::render::open_adapter(true, None);
        (device, queue)
    }

    fn landscape(width: u32, height: u32) -> Vec<u8> {
        let mut px = vec![0u8; (width * height * 4) as usize];
        for y in 0..height {
            let v = y as f32 / height as f32;
            for x in 0..width {
                let u = x as f32 / width as f32;
                let sky = v < 0.46;
                let mut rgb = if sky {
                    let t = v / 0.46;
                    [
                        (126.0 + 70.0 * t) as u8,
                        (168.0 + 36.0 * t) as u8,
                        (214.0 - 18.0 * t) as u8,
                    ]
                } else {
                    let t = (v - 0.46) / 0.54;
                    [
                        (96.0 + 48.0 * t) as u8,
                        (68.0 + 22.0 * t) as u8,
                        (38.0 + 10.0 * t) as u8,
                    ]
                };
                let road_half = 0.07 + (v - 0.46).max(0.0) * 0.62;
                if !sky && (u - 0.5).abs() < road_half {
                    let edge = (u - 0.5).abs() / road_half;
                    rgb = [
                        (176.0 - edge * 36.0) as u8,
                        (148.0 - edge * 28.0) as u8,
                        (108.0 - edge * 18.0) as u8,
                    ];
                    if (u - 0.5).abs() < 0.005 && (v * 16.0).fract() > 0.55 {
                        rgb = [232, 224, 196];
                    }
                }
                let du = u - 0.52;
                let dv = v - 0.64;
                if du * du * 2.2 + dv * dv * 10.0 < 0.018 {
                    rgb = [186, 168, 112];
                }
                if (u - 0.5).abs() < 0.04 && (v - 0.545).abs() < 0.028 {
                    rgb = [26, 30, 34];
                }
                if (u - 0.18).abs() < 0.0035 && (0.18..0.72).contains(&v) {
                    rgb = [240, 236, 224];
                }
                let i = ((y * width + x) * 4) as usize;
                px[i] = rgb[0];
                px[i + 1] = rgb[1];
                px[i + 2] = rgb[2];
                px[i + 3] = 255;
            }
        }
        px
    }

    fn render_lens(
        lens: &mut LensPass,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        plate: &[u8],
        gpu: &silt::lens::GpuLens,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let color_view = color.create_view(&Default::default());
        let row = width * 4;
        let stride = row.div_ceil(256) * 256;
        let mut staged = vec![0u8; stride as usize * height as usize];
        for y in 0..height as usize {
            let src = y * row as usize;
            let dst = y * stride as usize;
            staged[dst..dst + row as usize].copy_from_slice(&plate[src..src + row as usize]);
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &staged,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(stride),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens-test-read"),
            size: stride as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("lens-test"),
        });
        lens.refract(device, &mut encoder, queue, &color, &color_view, gpu);
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
                    bytes_per_row: Some(stride),
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
        let mut tight = Vec::with_capacity(row as usize * height as usize);
        for y in 0..height as usize {
            let src = y * stride as usize;
            tight.extend_from_slice(&data[src..src + row as usize]);
        }
        drop(data);
        readback.unmap();
        tight
    }

    fn save_capture(name: &str, width: u32, height: u32, rgba: &[u8]) -> std::path::PathBuf {
        let dir = std::env::var("LENS_CAPTURE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir());
        let path = dir.join(name);
        crate::render::write_png(&path, width, height, rgba).expect("png");
        path
    }

    #[test]
    fn lens_refracts_round_drops_and_a_few_streaks() {
        if !lavapipe() {
            eprintln!("skip: lavapipe is not installed");
            return;
        }
        let (device, queue) = device();
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut lens = LensPass::new(&device, wgpu::TextureFormat::Rgba8Unorm);
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("lens pipeline: {err}");
        }
        lens.mud.sync(1.0 / 60.0, 2, 4.0, 0);
        assert!(lens.mud.mud_runs() >= 2 && lens.mud.mud_runs() <= 6);
        assert!(lens.mud.clear_beads() >= 4);

        let width = 1280u32;
        let height = 720u32;
        let aspect = width as f32 / height as f32;
        let plate = landscape(width, height);
        let packed = lens.mud.pack(aspect);

        let mut drops = packed;
        for run in &mut drops.runs {
            run.params[0] = 0.0;
        }
        let drop_px = render_lens(&mut lens, &device, &queue, &plate, &drops, width, height);
        let drop_path = save_capture("lens-pbr-drops.png", width, height, &drop_px);

        let mut streaks = packed;
        for bead in &mut streaks.beads {
            bead.params[0] = 0.0;
        }
        streaks.head[0] = 0.0;
        let streak_px = render_lens(&mut lens, &device, &queue, &plate, &streaks, width, height);
        let streak_path = save_capture("lens-pbr-streak.png", width, height, &streak_px);
        assert_round_beads(&plate, &drop_px, &drops, width, height, &drop_path);
        assert_streaks(&plate, &streak_px, &streaks, width, height, &streak_path);
    }

    fn assert_round_beads(
        plate: &[u8],
        image: &[u8],
        gpu: &silt::lens::GpuLens,
        width: u32,
        height: u32,
        path: &std::path::Path,
    ) {
        let (same, changed, spikes, spec) = coverage(plate, image, width, height);
        let pixels = (width * height) as u32;
        assert!(
            same > pixels / 2,
            "the plate should stay readable, unchanged {same}/{pixels} ({})",
            path.display()
        );
        assert!(
            changed > pixels / 200,
            "drops should cover part of the glass, changed {changed} ({})",
            path.display()
        );
        assert!(
            spikes < 40,
            "opaque brown spikes on the sky: {spikes} ({})",
            path.display()
        );
        assert!(spec >= 4, "water catchlights {spec} ({})", path.display());
        let mut round = 0u32;
        for bead in &gpu.beads {
            if bead.params[0] < 0.4 || bead.radii[1] < 1.0e-4 {
                continue;
            }
            let radius = bead.radii[1] * height as f32;
            if radius < 8.0 {
                continue;
            }
            let cx = bead.xy[0] * width as f32;
            let cy = bead.xy[1] * height as f32;
            // Stay inside this bead so a neighbor cannot stretch the box into a spike.
            let (bw, bh, n) = blob(image, plate, width, height, cx, cy, radius * 1.15);
            if n < 20 {
                continue;
            }
            // The changed pixels follow whatever the drop refracts, so a bead on a
            // vertical edge is a tall box. Reject only a very stretched mark.
            let tall = bh / bw.max(1.0);
            assert!(
                (0.45..2.4).contains(&tall) && n >= 12,
                "bead at ({cx:.0},{cy:.0}) is not a drop {bw:.0}x{bh:.0} radius {radius:.1} ({})",
                path.display()
            );
            let tail = connected_tail_rows(image, plate, width, height, cx, cy, radius);
            assert!(
                tail < 16,
                "bead at ({cx:.0},{cy:.0}) drips into a spike, tail rows {tail} ({})",
                path.display()
            );
            round += 1;
        }
        assert!(
            round >= 2,
            "expected round refractive beads, found {round} ({})",
            path.display()
        );
    }

    fn assert_streaks(
        plate: &[u8],
        image: &[u8],
        gpu: &silt::lens::GpuLens,
        width: u32,
        height: u32,
        path: &std::path::Path,
    ) {
        let (same, changed, spikes, _spec) = coverage(plate, image, width, height);
        let pixels = (width * height) as u32;
        assert!(
            same > pixels * 2 / 3,
            "streaks should leave most of the glass clear, unchanged {same}/{pixels} ({})",
            path.display()
        );
        assert!(
            changed > 80,
            "streaks should be visible, changed {changed} ({})",
            path.display()
        );
        assert!(
            spikes < 30,
            "opaque brown spikes on the sky: {spikes} ({})",
            path.display()
        );
        let mut thin = 0u32;
        for run in &gpu.runs {
            if run.params[0] < 0.4 || run.len_w[0] < 0.05 {
                continue;
            }
            let x0 = ((run.xy[0] - 0.1) * width as f32).max(0.0) as u32;
            let x1 = ((run.xy[0] + 0.1) * width as f32).min(width as f32 - 1.0) as u32;
            let y0 = ((run.xy[1] - 0.02) * height as f32).max(0.0) as u32;
            let y1 =
                ((run.xy[1] + run.len_w[0] + 0.05) * height as f32).min(height as f32 - 1.0) as u32;
            let (min_x, max_x, min_y, max_y, n) = rect_blob(image, plate, width, x0, x1, y0, y1);
            if n < 8 {
                continue;
            }
            let bw = (max_x - min_x + 1) as f32;
            let bh = (max_y - min_y + 1) as f32;
            if bh > bw * 1.8 {
                thin += 1;
            }
        }
        assert!(
            thin >= 2,
            "expected a few vertical streaks, found {thin} ({})",
            path.display()
        );
    }

    fn coverage(plate: &[u8], image: &[u8], width: u32, height: u32) -> (u32, u32, u32, u32) {
        let mut same = 0u32;
        let mut changed = 0u32;
        let mut spikes = 0u32;
        let mut spec = 0u32;
        for y in 0..height as usize {
            for x in 0..width as usize {
                let i = (y * width as usize + x) * 4;
                let (pr, pg, pb) = (plate[i], plate[i + 1], plate[i + 2]);
                let (r, g, b) = (image[i], image[i + 1], image[i + 2]);
                let d = pr.abs_diff(r) as u32 + pg.abs_diff(g) as u32 + pb.abs_diff(b) as u32;
                if d <= 12 {
                    same += 1;
                } else {
                    changed += 1;
                }
                if pb > 160 && pr < 170 && r > 90 && b < 55 && r > b.saturating_add(40) {
                    spikes += 1;
                }
                if r > 230 && g > 228 && b > 220 {
                    spec += 1;
                }
            }
        }
        (same, changed, spikes, spec)
    }

    /// Rows of a drip that stays attached under the bead. A second drop below a gap does not count.
    fn connected_tail_rows(
        image: &[u8],
        plate: &[u8],
        width: u32,
        height: u32,
        cx: f32,
        cy: f32,
        radius: f32,
    ) -> u32 {
        let y0 = (cy + radius * 1.25).floor().max(0.0) as u32;
        let y1 = (cy + radius * 4.0).ceil().min(height as f32 - 1.0) as u32;
        let x0 = (cx - radius * 0.4).floor().max(0.0) as u32;
        let x1 = (cx + radius * 0.4).ceil().min(width as f32 - 1.0) as u32;
        if y1 < y0 || x1 < x0 {
            return 0;
        }
        let mut rows = 0u32;
        let mut gap = 0u32;
        let mut attached = false;
        for y in y0..=y1 {
            let mut hit = false;
            for x in x0..=x1 {
                let i = ((y * width + x) * 4) as usize;
                let d = plate[i].abs_diff(image[i]) as u32
                    + plate[i + 1].abs_diff(image[i + 1]) as u32
                    + plate[i + 2].abs_diff(image[i + 2]) as u32;
                if d > 28 {
                    hit = true;
                    break;
                }
            }
            if hit {
                // A separate bead below a gap is not a tail on this one.
                if !attached && gap > 3 {
                    return 0;
                }
                attached = true;
                rows += 1;
                gap = 0;
            } else {
                gap += 1;
                if attached && gap > 2 {
                    break;
                }
                if !attached && gap > 4 {
                    return 0;
                }
            }
        }
        rows
    }

    fn blob(
        image: &[u8],
        plate: &[u8],
        width: u32,
        height: u32,
        cx: f32,
        cy: f32,
        window: f32,
    ) -> (f32, f32, u32) {
        let x0 = (cx - window).floor().max(0.0) as u32;
        let y0 = (cy - window).floor().max(0.0) as u32;
        let x1 = (cx + window).ceil().min(width as f32 - 1.0) as u32;
        let y1 = (cy + window).ceil().min(height as f32 - 1.0) as u32;
        let (min_x, max_x, min_y, max_y, n) = rect_blob(image, plate, width, x0, x1, y0, y1);
        if n == 0 {
            return (0.0, 0.0, 0);
        }
        ((max_x - min_x + 1) as f32, (max_y - min_y + 1) as f32, n)
    }

    fn rect_blob(
        image: &[u8],
        plate: &[u8],
        width: u32,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
    ) -> (u32, u32, u32, u32, u32) {
        let mut min_x = u32::MAX;
        let mut max_x = 0u32;
        let mut min_y = u32::MAX;
        let mut max_y = 0u32;
        let mut n = 0u32;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let i = ((y * width + x) * 4) as usize;
                let d = plate[i].abs_diff(image[i]) as u32
                    + plate[i + 1].abs_diff(image[i + 1]) as u32
                    + plate[i + 2].abs_diff(image[i + 2]) as u32;
                if d > 28 {
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                    n += 1;
                }
            }
        }
        (min_x, max_x, min_y, max_y, n)
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
