//! Thin wgpu runtime. The soil and the buggy are drawn from the solver state:
//! height and material live in textures, mud mass lives in a per-draw uniform.

use std::path::Path;

use glam::{Mat4, Vec3};
use silt::particles::Particle;
use silt::soil::{SoilField, MAP_RES, MAP_SIZE};
use silt::vehicle::{Vehicle, PANEL_COUNT, PANEL_MUD_CAP};
use silt::world::{CameraPose, World};

use crate::mesh::{
    buggy_body, environment_mesh, suspension_mesh, terrain_mesh, wheel_mesh, CpuMesh, MeshVertex,
    ParticleVertex, TerrainVertex,
};

const SHADOW: u32 = 2048;
const MODEL_SLOTS: u64 = 8;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameData {
    view_proj: [[f32; 4]; 4],
    light_vp: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    sun_dir: [f32; 4],
    sun_color: [f32; 4],
    sky_zenith: [f32; 4],
    sky_horizon: [f32; 4],
    amb: [f32; 4],
    params: [f32; 4],
    misc: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ModelData {
    world: [[f32; 4]; 4],
    mud: [[f32; 4]; 12],
    extra: [f32; 4],
}

struct GpuMesh {
    vbuf: wgpu::Buffer,
    ibuf: wgpu::Buffer,
    count: u32,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    #[allow(dead_code)]
    pub adapter_name: String,
    width: u32,
    height: u32,
    model_stride: u32,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    model_buf: wgpu::Buffer,
    model_bg: wgpu::BindGroup,
    tex_bg: wgpu::BindGroup,
    height_bg: wgpu::BindGroup,
    height_tex: wgpu::Texture,
    mat_tex: wgpu::Texture,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    #[allow(dead_code)]
    shadow: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
    terrain: GpuMesh,
    body: GpuMesh,
    wheel: GpuMesh,
    env: GpuMesh,
    susp_v: wgpu::Buffer,
    susp_i: wgpu::Buffer,
    particles: wgpu::Buffer,
    pipe_sky: wgpu::RenderPipeline,
    pipe_terrain: wgpu::RenderPipeline,
    pipe_water: wgpu::RenderPipeline,
    pipe_mesh: wgpu::RenderPipeline,
    pipe_particle: wgpu::RenderPipeline,
    pipe_shadow_terrain: wgpu::RenderPipeline,
    pipe_shadow_mesh: wgpu::RenderPipeline,
    blit_bg: wgpu::BindGroup,
    pipe_blit: wgpu::RenderPipeline,
    pipe_blit_srgb: wgpu::RenderPipeline,
    blit_shader: wgpu::ShaderModule,
    blit_layout: wgpu::PipelineLayout,
    mat_scratch: Vec<u8>,
}

pub fn prepare_software_vulkan() {
    let icd = "/usr/share/vulkan/icd.d/lvp_icd.json";
    if Path::new(icd).exists() {
        std::env::set_var("VK_DRIVER_FILES", icd);
        std::env::set_var("VK_ICD_FILENAMES", icd);
    }
    std::env::remove_var("DISPLAY");
    if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
        let dir = "/tmp/runtime-ubuntu";
        let _ = std::fs::create_dir_all(dir);
        std::env::set_var("XDG_RUNTIME_DIR", dir);
    }
}

pub fn open_adapter(
    software: bool,
    surface: Option<&wgpu::Surface<'_>>,
) -> (wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if software {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::all()
        },
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: software,
        compatible_surface: surface,
    }))
    .unwrap_or_else(|| {
        panic!(
            "no wgpu adapter (software={software}). For lavapipe set VK_DRIVER_FILES to lvp_icd.json and unset DISPLAY."
        )
    });
    let info = adapter.get_info();
    println!(
        "wgpu adapter: {} {:?} {:?}",
        info.name, info.backend, info.device_type
    );
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("silt"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .expect("request_device");
    (instance, adapter, device, queue)
}

impl Renderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, adapter_name: String, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silt"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[uniform_entry(0, false, std::mem::size_of::<FrameData>() as u64)],
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("terrain-tex"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let height_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("height-only"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let model_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("model"),
            entries: &[uniform_entry(
                0,
                true,
                std::mem::size_of::<ModelData>() as u64,
            )],
        });
        let blit_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let align = device.limits().min_uniform_buffer_offset_alignment.max(256);
        let model_stride = (std::mem::size_of::<ModelData>() as u32).div_ceil(align) * align;

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let model_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("models"),
            size: model_stride as u64 * MODEL_SLOTS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-bg"),
            layout: &frame_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buf.as_entire_binding(),
            }],
        });
        let model_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("model-bg"),
            layout: &model_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &model_buf,
                    offset: 0,
                    size: wgpu::BufferSize::new(std::mem::size_of::<ModelData>() as u64),
                }),
            }],
        });

        let height_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("height"),
            size: wgpu::Extent3d {
                width: MAP_RES as u32,
                height: MAP_RES as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let mat_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material"),
            size: wgpu::Extent3d {
                width: MAP_RES as u32,
                height: MAP_RES as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let shadow = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow"),
            size: wgpu::Extent3d {
                width: SHADOW,
                height: SHADOW,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let nearest = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex-bg"),
            layout: &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&height_tex.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mat_tex.create_view(&Default::default())),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
            ],
        });
        let height_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height-bg"),
            layout: &height_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &height_tex.create_view(&Default::default()),
                ),
            }],
        });

        let (color, color_view, depth, depth_view, readback, padded_row) =
            make_targets(&device, width, height);
        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit-bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
            ],
        });

        let terrain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-layout"),
            bind_group_layouts: &[&frame_bgl, &tex_bgl],
            push_constant_ranges: &[],
        });
        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh-layout"),
            bind_group_layouts: &[&frame_bgl, &tex_bgl, &model_bgl],
            push_constant_ranges: &[],
        });
        let solo_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("solo-layout"),
            bind_group_layouts: &[&frame_bgl],
            push_constant_ranges: &[],
        });
        let blit_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blit-layout"),
            bind_group_layouts: &[&blit_bgl],
            push_constant_ranges: &[],
        });

        let color_format = wgpu::TextureFormat::Rgba8Unorm;
        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow-layout"),
            bind_group_layouts: &[&frame_bgl, &height_bgl, &model_bgl],
            push_constant_ranges: &[],
        });
        let terrain_layout_ref = &terrain_layout;
        let mesh_layout_ref = &mesh_layout;

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipe_terrain = pipeline(
            &device,
            &shader,
            terrain_layout_ref,
            "vs_terrain",
            Some("fs_terrain"),
            terrain_buffers(),
            color_format,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_water = pipeline(
            &device,
            &shader,
            terrain_layout_ref,
            "vs_water",
            Some("fs_water"),
            terrain_buffers(),
            color_format,
            Some(depth_state(false, wgpu::CompareFunction::Less)),
            true,
            Some(wgpu::Face::Back),
        );
        let pipe_mesh = pipeline(
            &device,
            &shader,
            mesh_layout_ref,
            "vs_mesh",
            Some("fs_mesh"),
            mesh_buffers(),
            color_format,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_sky = pipeline(
            &device,
            &shader,
            &solo_layout,
            "vs_sky",
            Some("fs_sky"),
            &[],
            color_format,
            Some(depth_state(false, wgpu::CompareFunction::Always)),
            false,
            None,
        );
        let pipe_particle = pipeline(
            &device,
            &shader,
            &solo_layout,
            "vs_particle",
            Some("fs_particle"),
            particle_buffers(),
            color_format,
            Some(depth_state(false, wgpu::CompareFunction::Less)),
            true,
            None,
        );
        let pipe_shadow_terrain = pipeline(
            &device,
            &shader,
            &shadow_layout,
            "vs_shadow_terrain",
            None,
            terrain_buffers(),
            color_format,
            Some(shadow_depth_state()),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_shadow_mesh = pipeline(
            &device,
            &shader,
            &shadow_layout,
            "vs_shadow_mesh",
            None,
            mesh_buffers(),
            color_format,
            Some(shadow_depth_state()),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_blit = pipeline(
            &device,
            &blit_shader,
            &blit_layout,
            "vs_blit",
            Some("fs_blit"),
            &[],
            color_format,
            None,
            false,
            None,
        );
        let pipe_blit_srgb = pipeline(
            &device,
            &blit_shader,
            &blit_layout,
            "vs_blit",
            Some("fs_blit_tosrgb"),
            &[],
            color_format,
            None,
            false,
            None,
        );
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("wgpu pipeline validation: {err}");
        }

        let (tverts, tidx) = terrain_mesh(MAP_RES, MAP_SIZE);
        let terrain = upload_mesh(&device, &tverts, &tidx, "terrain");
        let body_cpu = buggy_body();
        let body = upload_cpu(&device, &body_cpu, "body");
        let wheel = upload_cpu(&device, &wheel_mesh(), "wheel");
        let env = upload_cpu(&device, &environment_mesh(), "env");
        let susp_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("susp-v"),
            size: (4096 * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let susp_i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("susp-i"),
            size: (8192 * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: (6144 * std::mem::size_of::<ParticleVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            adapter_name,
            width,
            height,
            model_stride,
            frame_buf,
            frame_bg,
            model_buf,
            model_bg,
            tex_bg,
            height_bg,
            height_tex,
            mat_tex,
            color,
            color_view,
            depth,
            depth_view,
            shadow,
            shadow_view,
            readback,
            padded_row,
            terrain,
            body,
            wheel,
            env,
            susp_v,
            susp_i,
            particles,
            pipe_sky,
            pipe_terrain,
            pipe_water,
            pipe_mesh,
            pipe_particle,
            pipe_shadow_terrain,
            pipe_shadow_mesh,
            blit_bg,
            pipe_blit,
            pipe_blit_srgb,
            blit_shader,
            blit_layout,
            mat_scratch: vec![0; MAP_RES * MAP_RES * 4],
        }
    }

    /// Point the window blit at the swapchain format. Offscreen frames stay linear-in-a-Unorm-buffer
    /// (gamma already applied); an sRGB surface must undo that before it encodes again.
    pub fn set_present_format(&mut self, format: wgpu::TextureFormat) {
        self.pipe_blit = pipeline(
            &self.device,
            &self.blit_shader,
            &self.blit_layout,
            "vs_blit",
            Some("fs_blit"),
            &[],
            format,
            None,
            false,
            None,
        );
        self.pipe_blit_srgb = pipeline(
            &self.device,
            &self.blit_shader,
            &self.blit_layout,
            "vs_blit",
            Some("fs_blit_tosrgb"),
            &[],
            format,
            None,
            false,
            None,
        );
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return;
        }
        self.width = width;
        self.height = height;
        let (color, color_view, depth, depth_view, readback, padded_row) =
            make_targets(&self.device, width, height);
        self.color = color;
        self.color_view = color_view;
        self.depth = depth;
        self.depth_view = depth_view;
        self.readback = readback;
        self.padded_row = padded_row;
        let nearest = self.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        // Rebuild the blit bind group against the new color view. Layout is recovered from the pipeline.
        let blit_bgl = self.pipe_blit.get_bind_group_layout(0);
        self.blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit-bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
            ],
        });
    }

    pub fn draw(
        &mut self,
        world: &World,
        cam: CameraPose,
        present: Option<(&wgpu::TextureView, bool)>,
        readback: bool,
    ) -> Option<Vec<u8>> {
        self.upload_soil(&world.soil);
        let susp = suspension_mesh(&world.vehicle);
        let susp_index_count = susp.indices.len() as u32;
        self.queue
            .write_buffer(&self.susp_v, 0, bytemuck::cast_slice(&susp.vertices));
        self.queue
            .write_buffer(&self.susp_i, 0, bytemuck::cast_slice(&susp.indices));

        let view = Mat4::look_at_rh(cam.eye, cam.target, Vec3::Y);
        let proj = Mat4::perspective_rh(
            46f32.to_radians(),
            self.width as f32 / self.height as f32,
            0.15,
            180.0,
        );
        let view_proj = proj * view;
        // Low sun so rut walls and the buggy contact shadow separate from the ground.
        let sun = Vec3::new(0.88, 0.32, 0.24).normalize();
        let light_target = Vec3::new(0.0, 0.0, -1.5);
        let light_view = Mat4::look_at_rh(light_target + sun * 48.0, light_target, Vec3::Y);
        let light_proj = Mat4::orthographic_rh(-28.0, 28.0, -28.0, 28.0, 1.0, 110.0);
        let light_vp = light_proj * light_view;
        let frame = FrameData {
            view_proj: view_proj.to_cols_array_2d(),
            light_vp: light_vp.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            cam_pos: vec4(cam.eye, 1.0),
            sun_dir: vec4(sun, 0.0),
            sun_color: [1.15, 0.74, 0.42, 1.0],
            sky_zenith: [0.13, 0.26, 0.52, 1.0],
            sky_horizon: [0.50, 0.28, 0.18, 1.0],
            amb: [0.10, 0.062, 0.038, 1.0],
            params: [
                self.width as f32,
                self.height as f32,
                MAP_SIZE / MAP_RES as f32,
                MAP_RES as f32,
            ],
            misc: [world.time, SHADOW as f32, 0.0, 0.0],
        };
        self.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        self.write_models(&world.vehicle);

        let mut quads = Vec::<ParticleVertex>::new();
        build_particles(&mut quads, &world.particles.items, cam.eye, cam.target);
        if !quads.is_empty() {
            self.queue
                .write_buffer(&self.particles, 0, bytemuck::cast_slice(&quads));
        }
        let particle_count = quads.len() as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_viewport(0.0, 0.0, SHADOW as f32, SHADOW as f32, 0.0, 1.0);
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(1, &self.height_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[0]);
            pass.set_pipeline(&self.pipe_shadow_terrain);
            draw_indexed(&mut pass, &self.terrain);
            pass.set_pipeline(&self.pipe_shadow_mesh);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            draw_indexed(&mut pass, &self.env);
            self.draw_vehicle(&mut pass, susp_index_count);
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_pipeline(&self.pipe_sky);
            pass.draw(0..3, 0..1);

            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_pipeline(&self.pipe_terrain);
            draw_indexed(&mut pass, &self.terrain);

            pass.set_pipeline(&self.pipe_mesh);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            draw_indexed(&mut pass, &self.env);
            self.draw_vehicle(&mut pass, susp_index_count);

            pass.set_pipeline(&self.pipe_water);
            draw_indexed(&mut pass, &self.terrain);

            if particle_count > 0 {
                pass.set_pipeline(&self.pipe_particle);
                pass.set_vertex_buffer(0, self.particles.slice(..));
                pass.draw(0..particle_count, 0..1);
            }
        }

        if let Some((view, srgb)) = present {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(if srgb {
                &self.pipe_blit_srgb
            } else {
                &self.pipe_blit
            });
            pass.set_bind_group(0, &self.blit_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        if readback {
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &self.color,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &self.readback,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(self.padded_row),
                        rows_per_image: Some(self.height),
                    },
                },
                wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                },
            );
        }

        self.queue.submit(Some(encoder.finish()));

        if !readback {
            return None;
        }
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().expect("map").expect("map buffer");
        let padded = slice.get_mapped_range();
        let row = (self.width * 4) as usize;
        let stride = self.padded_row as usize;
        let mut out = Vec::with_capacity(row * self.height as usize);
        for y in 0..self.height as usize {
            out.extend_from_slice(&padded[y * stride..y * stride + row]);
        }
        drop(padded);
        self.readback.unmap();
        Some(out)
    }

    fn upload_soil(&mut self, soil: &SoilField) {
        let res = soil.res as u32;
        let extent = wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: 1,
        };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.height_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&soil.height),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(res * 4),
                rows_per_image: Some(res),
            },
            extent,
        );
        for (i, px) in self.mat_scratch.chunks_exact_mut(4).enumerate() {
            px[0] = (soil.wetness[i].clamp(0.0, 1.0) * 255.0) as u8;
            px[1] = ((soil.water[i] / 0.34).clamp(0.0, 1.0) * 255.0) as u8;
            px[2] = (soil.darken[i].clamp(0.0, 1.0) * 255.0) as u8;
            px[3] = (soil.roughness[i].clamp(0.0, 1.0) * 255.0) as u8;
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.mat_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.mat_scratch,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(res * 4),
                rows_per_image: Some(res),
            },
            extent,
        );
    }

    fn write_models(&self, vehicle: &Vehicle) {
        let body = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
        let mut mud = [[0.0; 4]; PANEL_COUNT];
        for (i, panel) in vehicle.panels.iter().enumerate() {
            mud[i] = [
                panel.mass / PANEL_MUD_CAP,
                panel.drip,
                panel.streak,
                panel.wet,
            ];
        }
        let slots = [
            model(body, mud, [0.0, 0.0, 0.0, 0.0]),
            wheel_model(vehicle, body, mud, 0),
            wheel_model(vehicle, body, mud, 1),
            wheel_model(vehicle, body, mud, 2),
            wheel_model(vehicle, body, mud, 3),
            model(Mat4::IDENTITY, [[0.0; 4]; 12], [0.0, 0.0, 0.0, 0.0]),
            model(body, mud, [0.0, 0.0, 0.0, 0.0]),
        ];
        for (i, slot) in slots.iter().enumerate() {
            self.queue.write_buffer(
                &self.model_buf,
                i as u64 * self.model_stride as u64,
                bytemuck::bytes_of(slot),
            );
        }
    }

    fn draw_vehicle(&self, pass: &mut wgpu::RenderPass<'_>, susp_indices: u32) {
        pass.set_bind_group(2, &self.model_bg, &[0]);
        draw_indexed(pass, &self.body);
        for i in 0..4 {
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * (i as u32 + 1)]);
            draw_indexed(pass, &self.wheel);
        }
        pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 6]);
        pass.set_vertex_buffer(0, self.susp_v.slice(..));
        pass.set_index_buffer(self.susp_i.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..susp_indices, 0, 0..1);
    }
}

fn wheel_model(vehicle: &Vehicle, body: Mat4, mud: [[f32; 4]; 12], index: usize) -> ModelData {
    let local = vehicle.wheel_center_local(index);
    let wheel = vehicle.wheels[index];
    let world = body
        * Mat4::from_translation(local)
        * Mat4::from_rotation_y(wheel.steer)
        * Mat4::from_rotation_x(wheel.spin);
    model(
        world,
        mud,
        [1.0, (wheel.mud / 1.5).clamp(0.0, 1.4), 0.0, 0.0],
    )
}

fn model(world: Mat4, mud: [[f32; 4]; 12], extra: [f32; 4]) -> ModelData {
    ModelData {
        world: world.to_cols_array_2d(),
        mud,
        extra,
    }
}

fn vec4(v: Vec3, w: f32) -> [f32; 4] {
    [v.x, v.y, v.z, w]
}

fn build_particles(out: &mut Vec<ParticleVertex>, items: &[Particle], eye: Vec3, target: Vec3) {
    let forward = (target - eye).normalize_or_zero();
    let inv = Mat4::look_at_rh(eye, target, Vec3::Y).inverse();
    let right = Vec3::new(inv.x_axis.x, inv.x_axis.y, inv.x_axis.z).normalize_or_zero();
    let up = Vec3::new(inv.y_axis.x, inv.y_axis.y, inv.y_axis.z).normalize_or_zero();
    let _ = forward;
    for p in items {
        if !p.alive || p.life <= 0.0 {
            continue;
        }
        let life = (p.life / p.max_life).clamp(0.0, 1.0);
        let (color, scale) = match p.kind {
            // Dirty brown-gray. Pale spray stacked into white blobs.
            0 => ([0.15, 0.09, 0.05, life * 0.7], 0.28),
            1 => ([0.10, 0.05, 0.025, life * 0.9], 0.42),
            _ => ([0.07, 0.035, 0.018, life * 0.85], 0.32),
        };
        let h = p.size * scale;
        let c = p.pos;
        let pts = [
            c - right * h - up * h,
            c + right * h - up * h,
            c + right * h + up * h,
            c - right * h + up * h,
        ];
        let uvs = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let tri = [0, 1, 2, 0, 2, 3];
        for k in tri {
            out.push(ParticleVertex {
                pos: pts[k].to_array(),
                color,
                uv: uvs[k],
            });
        }
    }
}

fn draw_indexed(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh) {
    pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
    pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.count, 0, 0..1);
}

fn upload_cpu(device: &wgpu::Device, mesh: &CpuMesh, label: &str) -> GpuMesh {
    upload_mesh(device, &mesh.vertices, &mesh.indices, label)
}

fn upload_mesh<V: bytemuck::Pod>(
    device: &wgpu::Device,
    vertices: &[V],
    indices: &[u32],
    label: &str,
) -> GpuMesh {
    use wgpu::util::DeviceExt;
    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    GpuMesh {
        vbuf,
        ibuf,
        count: indices.len() as u32,
    }
}

fn make_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Buffer,
    u32,
) {
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color"),
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
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());
    let padded_row = (width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: padded_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    (color, color_view, depth, depth_view, readback, padded_row)
}

fn uniform_entry(binding: u32, dynamic: bool, size: u64) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: write,
        depth_compare: compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn shadow_depth_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState {
            constant: 2,
            slope_scale: 2.0,
            clamp: 0.0,
        },
    }
}

fn terrain_buffers() -> &'static [wgpu::VertexBufferLayout<'static>] {
    &[wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<TerrainVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 8,
                shader_location: 1,
            },
        ],
    }]
}

fn mesh_buffers() -> &'static [wgpu::VertexBufferLayout<'static>] {
    &[wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<MeshVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 24,
                shader_location: 2,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32,
                offset: 32,
                shader_location: 3,
            },
        ],
    }]
}

fn particle_buffers() -> &'static [wgpu::VertexBufferLayout<'static>] {
    &[wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ParticleVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 28,
                shader_location: 2,
            },
        ],
    }]
}

fn pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    vs: &'static str,
    fs: Option<&'static str>,
    buffers: &'static [wgpu::VertexBufferLayout<'static>],
    format: wgpu::TextureFormat,
    depth: Option<wgpu::DepthStencilState>,
    alpha: bool,
    cull: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    let blend = if alpha {
        Some(wgpu::BlendState::ALPHA_BLENDING)
    } else {
        Some(wgpu::BlendState::REPLACE)
    };
    // Depth-only shadow passes have no color target. A color target is still
    // declared only when a fragment entry exists.
    let targets: Vec<Option<wgpu::ColorTargetState>> = if fs.is_some() {
        vec![Some(wgpu::ColorTargetState {
            format,
            blend,
            write_mask: wgpu::ColorWrites::ALL,
        })]
    } else {
        Vec::new()
    };
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
            cull_mode: cull,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: depth,
        multisample: wgpu::MultisampleState::default(),
        fragment: fs.map(|entry| wgpu::FragmentState {
            module: shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview: None,
        cache: None,
    })
}

pub fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))?
        .write_image_data(rgba)
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
}

const BLIT_WGSL: &str = r#"
struct V {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var blit_tex: texture_2d<f32>;
@group(0) @binding(1) var blit_samp: sampler;

@vertex
fn vs_blit(@builtin(vertex_index) vid: u32) -> V {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    var o: V;
    o.clip = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}

@fragment
fn fs_blit(in: V) -> @location(0) vec4<f32> {
    return textureSample(blit_tex, blit_samp, in.uv);
}

@fragment
fn fs_blit_tosrgb(in: V) -> @location(0) vec4<f32> {
    let c = textureSample(blit_tex, blit_samp, in.uv);
    return vec4<f32>(pow(c.rgb, vec3<f32>(2.2)), c.a);
}
"#;
