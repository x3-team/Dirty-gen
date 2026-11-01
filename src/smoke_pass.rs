//! Smoke and fire in the picture: steps `silt::smoke::GpuSmoke` and marches
//! its texture over the opaque scene (`smoke_volume.wgsl`, pass "smoke-volume").
//!
//! Off unless a scene asks for smoke. Needs compute, so never on WebGL2.

use glam::Vec3;
use silt::smoke::{blackbody, GpuSmoke, SmokeConfig, AMBIENT_K};

const SHADER: &str = include_str!("smoke_volume.wgsl");
const LUT: usize = 24;
/// Kelvin over ambient per LUT entry: 0..1610 K over 293 K ambient.
const LUT_STEP_K: f32 = 70.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmokeDraw {
    origin: [f32; 4],
    size: [f32; 4],
    albedo: [f32; 4],
    light: [f32; 4],
    lut: [[f32; 4]; LUT],
    fire_pos: [[f32; 4]; 4],
    fire_col: [[f32; 4]; 4],
    detail: [f32; 4],
}

/// Look of the smoke. Defaults: light grey smoke, forward scattering.
#[derive(Clone, Copy, Debug)]
pub struct SmokeLook {
    pub albedo: Vec3,
    pub phase_g: f32,
    pub emission: f32,
    pub ambient: f32,
    /// Sub-grid erosion strength, 0 off.
    pub detail: f32,
    /// Detail noise frequency, 1/m.
    pub detail_scale: f32,
    /// Detail noise rise speed, m/s.
    pub detail_rise: f32,
    /// Light a fire throws on its smoke per unit of fuel fed per second.
    pub fire_light: f32,
}

impl Default for SmokeLook {
    fn default() -> Self {
        Self {
            albedo: Vec3::new(0.82, 0.82, 0.84),
            phase_g: 0.35,
            emission: 3.0,
            ambient: 1.6,
            detail: 1.0,
            detail_scale: 8.0,
            detail_rise: 0.9,
            fire_light: 0.5,
        }
    }
}

pub struct SmokePass {
    pub sim: GpuSmoke,
    pub look: SmokeLook,
    pipe: wgpu::RenderPipeline,
    frame_bgl: wgpu::BindGroupLayout,
    bgl: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl SmokePass {
    pub fn new(device: &wgpu::Device, format: wgpu::TextureFormat, config: SmokeConfig) -> Self {
        let sim = GpuSmoke::new(device, config);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoke-volume"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke-frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tex = |binding: u32, sample_type, dim| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: dim,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke-volume"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                tex(
                    1,
                    wgpu::TextureSampleType::Float { filterable: true },
                    wgpu::TextureViewDimension::D3,
                ),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                tex(
                    3,
                    wgpu::TextureSampleType::Depth,
                    wgpu::TextureViewDimension::D2,
                ),
                tex(
                    4,
                    wgpu::TextureSampleType::Float { filterable: false },
                    wgpu::TextureViewDimension::D2,
                ),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smoke-volume"),
            bind_group_layouts: &[&frame_bgl, &bgl],
            push_constant_ranges: &[],
        });
        let pipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("smoke-volume"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_smoke"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_smoke"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke-draw"),
            size: std::mem::size_of::<SmokeDraw>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("smoke"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        Self {
            sim,
            look: SmokeLook::default(),
            pipe,
            frame_bgl,
            bgl,
            uniform,
            sampler,
        }
    }

    fn draw_uniform(&self) -> SmokeDraw {
        let c = &self.sim.config;
        let size = c.size();
        let mut lut = [[0.0f32; 4]; LUT];
        for (k, e) in lut.iter_mut().enumerate() {
            let (rgb, power) = blackbody(AMBIENT_K + k as f32 * LUT_STEP_K);
            // Below ~800 K the gas does not glow visibly.
            let vis = ((AMBIENT_K + k as f32 * LUT_STEP_K - 750.0) / 250.0).clamp(0.0, 1.0);
            *e = [rgb[0], rgb[1], rgb[2], power * vis];
        }
        let l = &self.look;
        let mut fire_pos = [[0.0f32; 4]; 4];
        let mut fire_col = [[0.0f32; 4]; 4];
        let mut fires = 0usize;
        for src in self.sim.sources.iter().filter(|s| s.fuel_s > 0.0).take(4) {
            // The light sits a little above the fuel bed, warm like 1300 K.
            let (rgb, _) = blackbody(1300.0);
            let p = src.pos + Vec3::Y * 0.35;
            fire_pos[fires] = [p.x, p.y, p.z, src.radius];
            let w = src.fuel_s * l.fire_light;
            fire_col[fires] = [rgb[0] * w, rgb[1] * w, rgb[2] * w, 0.0];
            fires += 1;
        }
        SmokeDraw {
            fire_pos,
            fire_col,
            detail: [fires as f32, l.detail, l.detail_scale, l.detail_rise],
            origin: [c.origin.x, c.origin.y, c.origin.z, c.cell_m],
            size: [size.x, size.y, size.z, c.extinction_m],
            albedo: [l.albedo.x, l.albedo.y, l.albedo.z, l.phase_g],
            light: [l.emission, l.ambient, LUT_STEP_K, LUT as f32],
            lut,
        }
    }

    /// Record the march over `color`. `depth` is a depth-only view of the scene depth.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame_buf: &wgpu::Buffer,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        shadow: &wgpu::TextureView,
    ) {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.draw_uniform()));
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smoke-frame"),
            layout: &self.frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buf.as_entire_binding(),
            }],
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smoke-volume"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.sim.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(shadow),
                },
            ],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("smoke-volume"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
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
        pass.set_bind_group(0, &frame_bg, &[]);
        pass.set_bind_group(1, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}
