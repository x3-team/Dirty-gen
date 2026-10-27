//! Draws the corner map into the color target. Geometry comes from `silt::ui`.

use silt::world::World;

const MAX_VERTS: usize = 4096;
const MAX_INDICES: usize = 8192;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuVert {
    pos: [f32; 2],
    color: [f32; 4],
}

pub struct Minimap {
    pipeline: wgpu::RenderPipeline,
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
}

impl Minimap {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("minimap"),
            source: wgpu::ShaderSource::Wgsl(MAP_WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("minimap"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("minimap"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_map"),
                compilation_options: Default::default(),
                buffers: vertex_layout(),
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
                entry_point: Some("fs_map"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minimap-v"),
            size: (MAX_VERTS * std::mem::size_of::<GpuVert>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("minimap-i"),
            size: (MAX_INDICES * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            pipeline,
            vbuf,
            ibuf,
        }
    }

    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        width: u32,
        height: u32,
        world: &World,
    ) {
        if width < 8 || height < 8 {
            return;
        }
        let vehicles = silt::ui::read_positions(world);
        let mesh = silt::ui::widget_mesh(width as f32, height as f32, &vehicles);
        if mesh.indices.is_empty()
            || mesh.verts.len() > MAX_VERTS
            || mesh.indices.len() > MAX_INDICES
        {
            return;
        }
        let w = width as f32;
        let h = height as f32;
        let gpu: Vec<GpuVert> = mesh
            .verts
            .iter()
            .map(|v| GpuVert {
                pos: [(v.x / w) * 2.0 - 1.0, 1.0 - (v.y / h) * 2.0],
                color: v.rgba,
            })
            .collect();
        queue.write_buffer(&self.vbuf, 0, bytemuck::cast_slice(&gpu));
        queue.write_buffer(&self.ibuf, 0, bytemuck::cast_slice(&mesh.indices));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("minimap"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, self.vbuf.slice(..));
        pass.set_index_buffer(self.ibuf.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
    }
}

fn vertex_layout() -> &'static [wgpu::VertexBufferLayout<'static>] {
    &[wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<GpuVert>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 8,
                shader_location: 1,
            },
        ],
    }]
}

const MAP_WGSL: &str = r#"
struct V {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex
fn vs_map(v: V) -> Out {
    var o: Out;
    o.clip = vec4<f32>(v.pos, 0.0, 1.0);
    o.color = v.color;
    return o;
}

@fragment
fn fs_map(v: Out) -> @location(0) vec4<f32> {
    return v.color;
}
"#;
