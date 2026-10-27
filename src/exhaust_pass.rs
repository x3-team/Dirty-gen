//! Draws tailpipe smoke into the color target, then a heat-haze pass that
//! only shifts texels. The scene itself stays in `render`.

use glam::{Mat4, Vec3};
use silt::exhaust::{Exhaust, SmokePuff};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmokeVertex {
    pos: [f32; 4],
    vel: [f32; 4],
    geom: [f32; 4],
    color_seed: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SmokeFrame {
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PlumeGpu {
    origin_s: [f32; 4],
    axis_len: [f32; 4],
    radii: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HazeFrame {
    view_proj: [[f32; 4]; 4],
    cam_right: [f32; 4],
    params: [f32; 4],
    plumes: [PlumeGpu; 3],
}

const _: () = assert!(std::mem::size_of::<SmokeVertex>() == 64);
const _: () = assert!(std::mem::size_of::<SmokeFrame>() == 112);
const _: () = assert!(std::mem::size_of::<HazeFrame>() == 240);

const CORNERS: [[f32; 2]; 6] = [
    [-0.5, -0.5],
    [0.5, -0.5],
    [0.5, 0.5],
    [-0.5, -0.5],
    [0.5, 0.5],
    [-0.5, 0.5],
];

pub struct ExhaustPass {
    smoke_pipe: wgpu::RenderPipeline,
    haze_pipe: wgpu::RenderPipeline,
    smoke_bg: wgpu::BindGroup,
    haze_bgl: wgpu::BindGroupLayout,
    haze_bg: wgpu::BindGroup,
    frame_buf: wgpu::Buffer,
    haze_buf: wgpu::Buffer,
    vbuf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    scratch: wgpu::Texture,
    scratch_view: wgpu::TextureView,
    width: u32,
    height: u32,
    puff_scratch: Vec<SmokePuff>,
    verts: Vec<SmokeVertex>,
}

impl ExhaustPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, depth: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("exhaust"),
            source: wgpu::ShaderSource::Wgsl(include_str!("exhaust.wgsl").into()),
        });
        let smoke_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exhaust-smoke"),
            entries: &[uniform_entry(0, std::mem::size_of::<SmokeFrame>() as u64)],
        });
        let haze_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("exhaust-haze"),
            entries: &[
                uniform_entry(0, std::mem::size_of::<HazeFrame>() as u64),
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
        let smoke_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("exhaust-smoke-layout"),
            bind_group_layouts: &[&smoke_bgl],
            push_constant_ranges: &[],
        });
        let haze_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("exhaust-haze-layout"),
            bind_group_layouts: &[&haze_bgl],
            push_constant_ranges: &[],
        });
        let smoke_pipe = pipeline(
            device,
            &shader,
            &smoke_layout,
            "vs_smoke",
            "fs_smoke",
            true,
            Some(depth),
        );
        let haze_pipe = pipeline(device, &shader, &haze_layout, "vs_haze", "fs_haze", false, None);

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exhaust-frame"),
            size: std::mem::size_of::<SmokeFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let haze_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exhaust-haze-buf"),
            size: std::mem::size_of::<HazeFrame>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("exhaust-verts"),
            size: (64 * 6 * std::mem::size_of::<SmokeVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("exhaust-samp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let smoke_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exhaust-smoke-bg"),
            layout: &smoke_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buf.as_entire_binding(),
            }],
        });
        let (scratch, scratch_view) = scratch_target(device, width, height);
        let haze_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exhaust-haze-bg"),
            layout: &haze_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: haze_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&scratch_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        Self {
            smoke_pipe,
            haze_pipe,
            smoke_bg,
            haze_bgl,
            haze_bg,
            frame_buf,
            haze_buf,
            vbuf,
            sampler,
            scratch,
            scratch_view,
            width,
            height,
            puff_scratch: Vec::with_capacity(64),
            verts: Vec::with_capacity(64 * 6),
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        let (scratch, scratch_view) = scratch_target(device, width, height);
        self.scratch = scratch;
        self.scratch_view = scratch_view;
        self.haze_bg = self.make_haze_bg(device);
    }

    pub fn draw(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        color: &wgpu::Texture,
        color_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        exhaust: &Exhaust,
        view_proj: Mat4,
        eye: Vec3,
        target: Vec3,
        width: u32,
        height: u32,
    ) {
        if width != self.width || height != self.height || width == 0 || height == 0 {
            return;
        }
        exhaust.fill_smoke(&mut self.puff_scratch);
        let count = self.puff_scratch.len().min(64);
        let hot = exhaust.plumes().iter().any(|p| p.strength > 0.02);
        if count == 0 && !hot {
            return;
        }
        let (right, up) = camera_basis(eye, target);
        if count > 0 {
            let frame = SmokeFrame {
                view_proj: view_proj.to_cols_array_2d(),
                cam_right: [right.x, right.y, right.z, 0.0],
                cam_up: [up.x, up.y, up.z, 0.0],
                params: [exhaust.clock(), width as f32, height as f32, 0.0],
            };
            queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
            self.verts.clear();
            for puff in self.puff_scratch.iter().take(count) {
                for corner in CORNERS {
                    self.verts.push(SmokeVertex {
                        pos: [puff.pos.x, puff.pos.y, puff.pos.z, 1.0],
                        vel: [puff.vel.x, puff.vel.y, puff.vel.z, 0.0],
                        geom: [puff.size, puff.alpha, corner[0], corner[1]],
                        color_seed: [puff.color.x, puff.color.y, puff.color.z, puff.seed],
                    });
                }
            }
            queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&self.verts));
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("exhaust-smoke"),
                color_attachments: &[Some(color_attachment(color_view))],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.smoke_pipe);
            pass.set_bind_group(0, &self.smoke_bg, &[]);
            pass.set_vertex_buffer(0, self.vbuf.slice(..));
            pass.draw(0..(count as u32 * 6), 0..1);
        }
        if !hot {
            return;
        }
        let mut haze = HazeFrame {
            view_proj: view_proj.to_cols_array_2d(),
            cam_right: [right.x, right.y, right.z, 0.0],
            params: [exhaust.clock(), width as f32, height as f32, 0.0],
            plumes: [PlumeGpu {
                origin_s: [0.0; 4],
                axis_len: [0.0, 0.2, -1.0, 0.0],
                radii: [0.05, 0.12, 0.0, 0.0],
            }; 3],
        };
        for (i, plume) in exhaust.plumes().iter().enumerate() {
            haze.plumes[i] = PlumeGpu {
                origin_s: [plume.origin.x, plume.origin.y, plume.origin.z, plume.strength],
                axis_len: [plume.axis.x, plume.axis.y, plume.axis.z, plume.length],
                radii: [plume.radius0, plume.radius1, 0.0, 0.0],
            };
        }
        queue.write_buffer(&self.haze_buf, 0, bytemuck::bytes_of(&haze));
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.scratch,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("exhaust-haze"),
            color_attachments: &[Some(color_attachment(color_view))],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.haze_pipe);
        pass.set_bind_group(0, &self.haze_bg, &[]);
        pass.draw(0..3, 0..1);
    }

    fn make_haze_bg(&self, device: &wgpu::Device) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("exhaust-haze-bg"),
            layout: &self.haze_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.haze_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.scratch_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }
}

fn camera_basis(eye: Vec3, target: Vec3) -> (Vec3, Vec3) {
    let mut forward = target - eye;
    if forward.length_squared() < 1.0e-6 {
        forward = Vec3::Z;
    }
    forward = forward.normalize();
    let mut right = forward.cross(Vec3::Y);
    if right.length_squared() < 1.0e-6 {
        right = Vec3::X;
    }
    right = right.normalize();
    let up = right.cross(forward).normalize();
    (right, up)
}

fn scratch_target(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let scratch = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("exhaust-scratch"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = scratch.create_view(&Default::default());
    (scratch, view)
}

fn color_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    }
}

fn uniform_entry(binding: u32, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    vs: &'static str,
    fs: &'static str,
    smoke: bool,
    depth_format: Option<wgpu::TextureFormat>,
) -> wgpu::RenderPipeline {
    let blend = if smoke {
        Some(wgpu::BlendState::ALPHA_BLENDING)
    } else {
        Some(wgpu::BlendState::REPLACE)
    };
    let buffers: &[wgpu::VertexBufferLayout] = if smoke {
        &[wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SmokeVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 3,
                },
            ],
        }]
    } else {
        &[]
    };
    let depth = depth_format.map(|format| wgpu::DepthStencilState {
        format,
        depth_write_enabled: false,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(vs),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
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
        depth_stencil: depth,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some(fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview: None,
        cache: None,
    })
}
