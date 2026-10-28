//! Thin wgpu runtime. The soil and the buggy are drawn from the solver state:
//! height and material live in textures, mud mass lives in a per-draw uniform.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use glam::{Mat4, Vec3};
use silt::soil::{SoilField, MAP_RES, MAP_SIZE};
use silt::bike::{Bike, BIKE_RADIUS};
use silt::vehicle::{Vehicle, PANEL_COUNT, PANEL_MUD_CAP, WHEEL_RADIUS};
use silt::world::{CameraPose, World};

use crate::cc0::{buggy_fit, RgbaTex};
use crate::mesh::{
    bike_body, bike_suspension, environment_mesh, mud_chunks, terrain_mesh, wheel_mesh, CpuMesh,
    MeshVertex, ParticleVertex, TerrainVertex,
};

const SHADOW: u32 = 2048;
const MODEL_SLOTS: u64 = 12;

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
    /// Four tires, two vec4s each: center.xyz + radius, then forward.xz, sink, active.
    tires: [[f32; 4]; 8],
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
    imprint_tex: wgpu::Texture,
    mat_tex: wgpu::Texture,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    #[allow(dead_code)]
    shadow: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    #[allow(dead_code)]
    shadow_depth: wgpu::Texture,
    shadow_depth_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
    terrain: GpuMesh,
    body: GpuMesh,
    glass: GpuMesh,
    wheels: [GpuMesh; 4],
    wheel: GpuMesh,
    bike_body: GpuMesh,
    env: GpuMesh,
    props: GpuMesh,
    #[allow(dead_code)]
    resident: Vec<wgpu::Texture>,
    paint_bg: wgpu::BindGroup,
    bike_susp_v: wgpu::Buffer,
    bike_susp_i: wgpu::Buffer,
    #[allow(dead_code)]
    particles: wgpu::Buffer,
    chunk_v: wgpu::Buffer,
    chunk_i: wgpu::Buffer,
    pipe_sky: wgpu::RenderPipeline,
    pipe_terrain: wgpu::RenderPipeline,
    pipe_water: wgpu::RenderPipeline,
    pipe_mesh: wgpu::RenderPipeline,
    pipe_mesh_blend: wgpu::RenderPipeline,
    pipe_mesh_double: wgpu::RenderPipeline,
    #[allow(dead_code)]
    pipe_particle: wgpu::RenderPipeline,
    pipe_shadow_terrain: wgpu::RenderPipeline,
    pipe_shadow_mesh: wgpu::RenderPipeline,
    blit_bg: wgpu::BindGroup,
    pipe_blit: wgpu::RenderPipeline,
    pipe_blit_srgb: wgpu::RenderPipeline,
    blit_shader: wgpu::ShaderModule,
    blit_layout: wgpu::PipelineLayout,
    mat_scratch: Vec<u8>,
    imprint_scratch: Vec<u8>,
    height_scratch: Vec<f32>,
    height_bytes: Vec<u8>,
    height_packed: bool,
    depth_format: wgpu::TextureFormat,
    /// Midday sun and sky. The window keeps the low sun unless a still asks.
    pub daylight: bool,
    pub show_props: bool,
    pub show_bike: bool,
    /// Draw the fitted Lada Niva instead of the dune buggy mesh.
    pub niva_picture: bool,
    niva_body: GpuMesh,
    niva_glass: GpuMesh,
    niva_wheels: [GpuMesh; 4],
}

pub struct GpuFormats {
    pub depth: wgpu::TextureFormat,
    pub height: wgpu::TextureFormat,
    pub height_packed: bool,
}

pub fn gpu_formats(adapter: &wgpu::Adapter) -> GpuFormats {
    let depth_usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
    let depth = [
        wgpu::TextureFormat::Depth32Float,
        wgpu::TextureFormat::Depth24Plus,
    ]
    .into_iter()
    .find(|fmt| {
        adapter
            .get_texture_format_features(*fmt)
            .allowed_usages
            .contains(depth_usage)
    })
    .unwrap_or(wgpu::TextureFormat::Depth24Plus);
    let sample = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    let rg = wgpu::TextureFormat::Rg32Float;
    if adapter
        .get_texture_format_features(rg)
        .allowed_usages
        .contains(sample)
    {
        GpuFormats {
            depth,
            height: rg,
            height_packed: false,
        }
    } else {
        GpuFormats {
            depth,
            height: wgpu::TextureFormat::Rgba8Unorm,
            height_packed: true,
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

#[cfg(not(target_arch = "wasm32"))]
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
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_name: String,
        width: u32,
        height: u32,
        formats: GpuFormats,
        cc0: crate::cc0::Cc0Scene,
    ) -> Self {
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
            entries: &[uniform_entry(
                0,
                false,
                std::mem::size_of::<FrameData>() as u64,
            )],
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                filter_array(4, wgpu::ShaderStages::FRAGMENT),
                filter_array(5, wgpu::ShaderStages::FRAGMENT),
                filter_array(6, wgpu::ShaderStages::VERTEX_FRAGMENT),
                filter_samp(7, wgpu::ShaderStages::VERTEX_FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let paint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paint"),
            entries: &[
                filter_tex(0, wgpu::ShaderStages::FRAGMENT),
                filter_tex(1, wgpu::ShaderStages::FRAGMENT),
                filter_tex(2, wgpu::ShaderStages::FRAGMENT),
                filter_tex(3, wgpu::ShaderStages::FRAGMENT),
                filter_tex(4, wgpu::ShaderStages::FRAGMENT),
                filter_tex(5, wgpu::ShaderStages::FRAGMENT),
                filter_samp(6, wgpu::ShaderStages::FRAGMENT),
                array_tex(7),
                array_tex(8),
                array_tex(9),
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
            format: formats.height,
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
        let imprint_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("imprint"),
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let shadow_view = shadow.create_view(&Default::default());
        let shadow_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow-depth"),
            size: wgpu::Extent3d {
                width: SHADOW,
                height: SHADOW,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: formats.depth,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let shadow_depth_view = shadow_depth.create_view(&Default::default());
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
        let ground_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ground"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let paint_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("paint"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let (ground_diff, ground_diff_view) =
            upload_array(&device, &queue, "ground-diff", &cc0.ground_diff);
        let (ground_nor, ground_nor_view) =
            upload_array(&device, &queue, "ground-nor", &cc0.ground_nor);
        let (ground_pack, ground_pack_view) =
            upload_array(&device, &queue, "ground-pack", &cc0.ground_pack);
        let (buggy_base, buggy_base_view) = upload_rgba(&device, &queue, "buggy-base", &cc0.buggy_base);
        let (buggy_nor, buggy_nor_view) = upload_rgba(&device, &queue, "buggy-nor", &cc0.buggy_nor);
        let (buggy_mr, buggy_mr_view) = upload_rgba(&device, &queue, "buggy-mr", &cc0.buggy_mr);
        let (tire_base, tire_base_view) = upload_rgba(&device, &queue, "tire-base", &cc0.tire_base);
        let (tire_nor, tire_nor_view) = upload_rgba(&device, &queue, "tire-nor", &cc0.tire_nor);
        let (prop_tex, prop_view) = upload_rgba(&device, &queue, "prop", &cc0.prop_tex);
        eprintln!("loading Lada Niva");
        let mut niva = crate::niva::load();
        let (niva_base, niva_base_view) =
            upload_rgba_array(&device, &queue, "niva-base", &niva.base);
        niva.base = Vec::new();
        let (niva_nor, niva_nor_view) = upload_rgba_array(&device, &queue, "niva-nor", &niva.nor);
        niva.nor = Vec::new();
        let (niva_mr, niva_mr_view) = upload_rgba_array(&device, &queue, "niva-mr", &niva.mr);
        niva.mr = Vec::new();
        let resident = vec![
            ground_diff,
            ground_nor,
            ground_pack,
            buggy_base,
            buggy_nor,
            buggy_mr,
            tire_base,
            tire_nor,
            prop_tex,
            niva_base,
            niva_nor,
            niva_mr,
        ];
        let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tex-bg"),
            layout: &tex_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &height_tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &mat_tex.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&nearest),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&ground_diff_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&ground_nor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&ground_pack_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(&ground_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(
                        &imprint_tex.create_view(&Default::default()),
                    ),
                },
            ],
        });
        let paint_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paint-bg"),
            layout: &paint_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&buggy_base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&buggy_nor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&buggy_mr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&tire_base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&tire_nor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&prop_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&paint_samp),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&niva_base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&niva_nor_view),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&niva_mr_view),
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
            make_targets(&device, width, height, formats.depth);
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
            bind_group_layouts: &[&frame_bgl, &tex_bgl, &model_bgl, &paint_bgl],
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
            Some(depth_state(
                formats.depth,
                true,
                wgpu::CompareFunction::Less,
            )),
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
            Some(depth_state(
                formats.depth,
                false,
                wgpu::CompareFunction::Less,
            )),
            true,
            None,
        );
        let pipe_mesh = pipeline(
            &device,
            &shader,
            mesh_layout_ref,
            "vs_mesh",
            Some("fs_mesh"),
            mesh_buffers(),
            color_format,
            Some(depth_state(
                formats.depth,
                true,
                wgpu::CompareFunction::Less,
            )),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_mesh_blend = pipeline(
            &device,
            &shader,
            mesh_layout_ref,
            "vs_mesh",
            Some("fs_mesh"),
            mesh_buffers(),
            color_format,
            Some(depth_state(
                formats.depth,
                true,
                wgpu::CompareFunction::Less,
            )),
            true,
            None,
        );
        let pipe_mesh_double = pipeline(
            &device,
            &shader,
            mesh_layout_ref,
            "vs_mesh",
            Some("fs_mesh"),
            mesh_buffers(),
            color_format,
            Some(depth_state(
                formats.depth,
                true,
                wgpu::CompareFunction::Less,
            )),
            false,
            None,
        );
        let pipe_sky = pipeline(
            &device,
            &shader,
            &solo_layout,
            "vs_sky",
            Some("fs_sky"),
            &[],
            color_format,
            Some(depth_state(
                formats.depth,
                false,
                wgpu::CompareFunction::Always,
            )),
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
            Some(depth_state(
                formats.depth,
                false,
                wgpu::CompareFunction::Less,
            )),
            true,
            None,
        );
        let pipe_shadow_terrain = pipeline(
            &device,
            &shader,
            &shadow_layout,
            "vs_shadow_terrain",
            Some("fs_shadow"),
            terrain_buffers(),
            color_format,
            Some(shadow_depth_state(formats.depth)),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_shadow_mesh = pipeline(
            &device,
            &shader,
            &shadow_layout,
            "vs_shadow_mesh",
            Some("fs_shadow"),
            mesh_buffers(),
            color_format,
            Some(shadow_depth_state(formats.depth)),
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
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(err) = pollster::block_on(device.pop_error_scope()) {
            panic!("wgpu pipeline validation: {err}");
        }
        #[cfg(target_arch = "wasm32")]
        {
            let fut = device.pop_error_scope();
            wasm_bindgen_futures::spawn_local(async move {
                if let Some(err) = fut.await {
                    web_sys::console::error_1(&format!("wgpu pipeline: {err}").into());
                }
            });
        }

        let (tverts, tidx) = terrain_mesh(MAP_RES, MAP_SIZE);
        let terrain = upload_mesh(&device, &tverts, &tidx, "terrain");
        let body = upload_cpu(&device, &cc0.body, "body");
        let glass = upload_cpu(&device, &cc0.glass, "glass");
        let wheels = [
            upload_cpu(&device, &cc0.wheels[0], "wheel-fl"),
            upload_cpu(&device, &cc0.wheels[1], "wheel-fr"),
            upload_cpu(&device, &cc0.wheels[2], "wheel-rl"),
            upload_cpu(&device, &cc0.wheels[3], "wheel-rr"),
        ];
        let wheel = upload_cpu(&device, &wheel_mesh(), "wheel");
        let bike_body = upload_cpu(&device, &bike_body(), "bike");
        let env = upload_cpu(&device, &environment_mesh(), "env");
        let props = upload_cpu(&device, &cc0.props, "props");
        let niva_body = upload_cpu(&device, &niva.body, "niva-body");
        let niva_glass = upload_cpu(&device, &niva.glass, "niva-glass");
        let niva_wheels = [
            upload_cpu(&device, &niva.wheels[0], "niva-fl"),
            upload_cpu(&device, &niva.wheels[1], "niva-fr"),
            upload_cpu(&device, &niva.wheels[2], "niva-rl"),
            upload_cpu(&device, &niva.wheels[3], "niva-rr"),
        ];
        let bike_susp_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bike-susp-v"),
            size: (8192 * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bike_susp_i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bike-susp-i"),
            size: (16384 * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let particles = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles"),
            size: (6144 * std::mem::size_of::<ParticleVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunks-v"),
            size: (16384 * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let chunk_i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunks-i"),
            size: (48000 * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
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
            imprint_tex,
            mat_tex,
            color,
            color_view,
            depth,
            depth_view,
            shadow,
            shadow_view,
            shadow_depth,
            shadow_depth_view,
            readback,
            padded_row,
            terrain,
            body,
            glass,
            wheels,
            wheel,
            bike_body,
            env,
            props,
            resident,
            paint_bg,
            bike_susp_v,
            bike_susp_i,
            particles,
            chunk_v,
            chunk_i,
            pipe_sky,
            pipe_terrain,
            pipe_water,
            pipe_mesh,
            pipe_mesh_blend,
            pipe_mesh_double,
            pipe_particle,
            pipe_shadow_terrain,
            pipe_shadow_mesh,
            blit_bg,
            pipe_blit,
            pipe_blit_srgb,
            blit_shader,
            blit_layout,
            mat_scratch: vec![0; MAP_RES * MAP_RES * 4],
            imprint_scratch: vec![0; MAP_RES * MAP_RES * 4],
            height_scratch: vec![0.0; MAP_RES * MAP_RES * 2],
            height_bytes: vec![0; MAP_RES * MAP_RES * 4],
            height_packed: formats.height_packed,
            depth_format: formats.depth,
            daylight: false,
            show_props: true,
            show_bike: true,
            niva_picture: false,
            niva_body,
            niva_glass,
            niva_wheels,
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
            make_targets(&self.device, width, height, self.depth_format);
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
        let bike_susp = bike_suspension(&world.bike);
        let bike_susp_count = bike_susp.indices.len() as u32;
        self.queue
            .write_buffer(&self.bike_susp_v, 0, bytemuck::cast_slice(&bike_susp.vertices));
        self.queue
            .write_buffer(&self.bike_susp_i, 0, bytemuck::cast_slice(&bike_susp.indices));

        let view = Mat4::look_at_rh(cam.eye, cam.target, Vec3::Y);
        let aspect = self.width as f32 / self.height as f32;
        let proj = if let Some(width) = cam.ortho_width {
            let half_w = width * 0.5;
            let half_h = half_w / aspect.max(0.1);
            Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, 0.2, 260.0)
        } else {
            Mat4::perspective_rh(46f32.to_radians(), aspect, 0.15, 180.0)
        };
        let view_proj = proj * view;
        // The window keeps the low sun. Persist stills ask for daylight.
        let sun = if self.daylight {
            Vec3::new(0.38, 0.88, 0.18).normalize()
        } else {
            Vec3::new(0.88, 0.32, 0.24).normalize()
        };
        let light_target = if self.daylight {
            Vec3::new(0.4, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 2.0)
        };
        let extent = if self.daylight { 14.0 } else { 36.0 };
        let light_view = Mat4::look_at_rh(light_target + sun * 48.0, light_target, Vec3::Y);
        let light_proj = Mat4::orthographic_rh(-extent, extent, -extent, extent, 1.0, 130.0);
        let light_vp = light_proj * light_view;
        let (sun_color, sky_zenith, sky_horizon, amb) = if self.daylight {
            (
                [1.70, 1.78, 1.90, 1.0],
                [0.42, 0.64, 0.98, 1.0],
                [0.76, 0.84, 0.96, 1.0],
                [0.18, 0.20, 0.22, 1.0],
            )
        } else {
            (
                [1.15, 0.74, 0.42, 1.0],
                [0.13, 0.26, 0.52, 1.0],
                [0.50, 0.28, 0.18, 1.0],
                [0.10, 0.062, 0.038, 1.0],
            )
        };
        let frame = FrameData {
            view_proj: view_proj.to_cols_array_2d(),
            light_vp: light_vp.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            cam_pos: vec4(cam.eye, 1.0),
            sun_dir: vec4(sun, 0.0),
            sun_color,
            sky_zenith,
            sky_horizon,
            amb,
            params: [
                self.width as f32,
                self.height as f32,
                MAP_SIZE / MAP_RES as f32,
                MAP_RES as f32,
            ],
            misc: [
                world.time,
                SHADOW as f32,
                if self.height_packed { 1.0 } else { 0.0 },
                0.0,
            ],
            tires: tire_frame(&world.vehicle),
        };
        self.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        self.write_models(&world.vehicle);
        self.write_bike_models(&world.bike);

        let chunks = mud_chunks(&world.particles.items);
        let chunk_count = chunks.indices.len() as u32;
        if chunk_count > 0 {
            self.queue
                .write_buffer(&self.chunk_v, 0, bytemuck::cast_slice(&chunks.vertices));
            self.queue
                .write_buffer(&self.chunk_i, 0, bytemuck::cast_slice(&chunks.indices));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.shadow_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_depth_view,
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
            if self.show_props {
                draw_indexed(&mut pass, &self.props);
            }
            self.draw_vehicle(&mut pass);
            if self.show_bike {
                self.draw_bike(&mut pass, bike_susp_count);
            }
            pass.set_pipeline(&self.pipe_shadow_mesh);
            self.draw_chunks(&mut pass, chunk_count);
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
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            draw_indexed(&mut pass, &self.env);
            self.draw_vehicle(&mut pass);
            if self.show_bike {
                self.draw_bike(&mut pass, bike_susp_count);
            }

            pass.set_pipeline(&self.pipe_mesh_double);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            if self.show_props {
                draw_indexed(&mut pass, &self.props);
            }

            pass.set_pipeline(&self.pipe_water);
            draw_indexed(&mut pass, &self.terrain);

            pass.set_pipeline(&self.pipe_mesh);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            self.draw_chunks(&mut pass, chunk_count);

            pass.set_pipeline(&self.pipe_mesh_blend);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            self.draw_glass(&mut pass);
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
        for (i, px) in self.height_scratch.chunks_exact_mut(2).enumerate() {
            px[0] = soil.height[i];
            // Current height minus the surface the stamp started from.
            px[1] = soil.height[i] - soil.initial_height[i];
        }
        if self.height_packed {
            for (i, px) in self.height_bytes.chunks_exact_mut(4).enumerate() {
                let h = ((soil.height[i] + 1.0) / 3.5).clamp(0.0, 1.0);
                let stamp = soil.height[i] - soil.initial_height[i];
                let s = ((stamp + 0.75) / 1.5).clamp(0.0, 1.0);
                let hq = (h * 65535.0).round() as u32;
                let sq = (s * 65535.0).round() as u32;
                px[0] = (hq >> 8) as u8;
                px[1] = (hq & 255) as u8;
                px[2] = (sq >> 8) as u8;
                px[3] = (sq & 255) as u8;
            }
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.height_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.height_bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(res * 4),
                    rows_per_image: Some(res),
                },
                extent,
            );
        } else {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.height_tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&self.height_scratch),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(res * 8),
                    rows_per_image: Some(res),
                },
                extent,
            );
        }
        for (i, px) in self.mat_scratch.chunks_exact_mut(4).enumerate() {
            px[0] = (soil.wetness[i].clamp(0.0, 1.0) * 255.0) as u8;
            // Free water and the mixed body share this channel. The mesh
            // already sits on the slurry, so the tint is that one surface.
            let liquid = soil.water[i] + soil.slurry[i];
            px[1] = ((liquid / 0.34).clamp(0.0, 1.0) * 255.0) as u8;
            px[2] = (soil.darken[i].clamp(0.0, 1.0) * 255.0) as u8;
            px[3] = soil.kind[i];
        }
        for (i, px) in self.imprint_scratch.chunks_exact_mut(4).enumerate() {
            let amp = soil.tread_amp[i];
            if amp < 0.001 {
                px[0] = 0;
                px[1] = 0;
                px[2] = 0;
                px[3] = 0;
                continue;
            }
            let ang = soil.tread_ang[i].rem_euclid(std::f32::consts::TAU)
                / std::f32::consts::TAU;
            let lat = ((soil.tread_lat[i] + 0.4) / 0.8).clamp(0.0, 1.0);
            px[0] = ((amp / 0.06).clamp(0.0, 1.0) * 255.0) as u8;
            px[1] = (ang.clamp(0.0, 1.0) * 255.0) as u8;
            px[2] = (lat * 255.0) as u8;
            px[3] = 255;
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.imprint_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.imprint_scratch,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(res * 4),
                rows_per_image: Some(res),
            },
            extent,
        );
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
        let chassis = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
        // The Niva is already fitted onto the hardpoints. The buggy mesh is not.
        let body = if self.niva_picture {
            chassis
        } else {
            chassis * buggy_fit()
        };
        let mut mud = [[0.0; 4]; PANEL_COUNT];
        for (i, panel) in vehicle.panels.iter().enumerate() {
            mud[i] = [
                panel.mass / PANEL_MUD_CAP,
                panel.drip,
                panel.streak,
                panel.wet,
            ];
        }
        let niva = self.niva_picture;
        let slots = [
            model(body, mud, [0.0, 0.0, 0.0, 0.0]),
            wheel_model(vehicle, chassis, mud, 0, niva),
            wheel_model(vehicle, chassis, mud, 1, niva),
            wheel_model(vehicle, chassis, mud, 2, niva),
            wheel_model(vehicle, chassis, mud, 3, niva),
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

    fn draw_vehicle(&self, pass: &mut wgpu::RenderPass<'_>) {
        let (body, wheels) = if self.niva_picture {
            (&self.niva_body, &self.niva_wheels)
        } else {
            (&self.body, &self.wheels)
        };
        pass.set_bind_group(2, &self.model_bg, &[0]);
        draw_indexed(pass, body);
        for i in 0..4 {
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * (i as u32 + 1)]);
            draw_indexed(pass, &wheels[i]);
        }
    }

    fn draw_glass(&self, pass: &mut wgpu::RenderPass<'_>) {
        let glass = if self.niva_picture {
            &self.niva_glass
        } else {
            &self.glass
        };
        if glass.count == 0 {
            return;
        }
        pass.set_bind_group(2, &self.model_bg, &[0]);
        draw_indexed(pass, glass);
    }

    fn write_bike_models(&self, bike: &Bike) {
        let body = Mat4::from_translation(bike.pos) * Mat4::from_quat(bike.orientation());
        let mut mud = [[0.0; 4]; PANEL_COUNT];
        for (i, panel) in bike.panels.iter().enumerate() {
            mud[i] = [
                panel.mass / PANEL_MUD_CAP,
                panel.drip,
                panel.streak,
                panel.wet,
            ];
        }
        let slots = [
            (7u64, model(body, mud, [0.0, 0.0, 0.0, 0.0])),
            (8, bike_wheel_model(bike, body, mud, 0)),
            (9, bike_wheel_model(bike, body, mud, 1)),
            (10, model(body, mud, [0.0, 0.0, 0.0, 0.0])),
        ];
        for (i, slot) in slots {
            self.queue.write_buffer(
                &self.model_buf,
                i * self.model_stride as u64,
                bytemuck::bytes_of(&slot),
            );
        }
    }

    fn draw_bike(&self, pass: &mut wgpu::RenderPass<'_>, susp_indices: u32) {
        pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 7]);
        draw_indexed(pass, &self.bike_body);
        for i in 0..2 {
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * (8 + i)]);
            draw_indexed(pass, &self.wheel);
        }
        pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 10]);
        pass.set_vertex_buffer(0, self.bike_susp_v.slice(..));
        pass.set_index_buffer(self.bike_susp_i.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..susp_indices, 0, 0..1);
    }

    fn draw_chunks(&self, pass: &mut wgpu::RenderPass<'_>, count: u32) {
        if count == 0 {
            return;
        }
        // Slot 5 is the identity transform. Chunk vertices are already in world space.
        pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
        pass.set_vertex_buffer(0, self.chunk_v.slice(..));
        pass.set_index_buffer(self.chunk_i.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..count, 0, 0..1);
    }
}

fn tire_frame(vehicle: &Vehicle) -> [[f32; 4]; 8] {
    use glam::Quat;
    let mut out = [[0.0; 4]; 8];
    let ori = vehicle.orientation();
    for i in 0..4 {
        let center = vehicle.pos + ori * vehicle.wheel_center_local(i);
        let wheel = vehicle.wheels[i];
        let mut fwd = ori * (Quat::from_rotation_y(wheel.steer) * Vec3::Z);
        fwd.y = 0.0;
        if fwd.length_squared() < 1.0e-6 {
            fwd = Vec3::Z;
        } else {
            fwd = fwd.normalize();
        }
        let live = wheel.sinkage > 0.003 && wheel.normal > 1.0;
        out[i * 2] = [center.x, center.y, center.z, WHEEL_RADIUS];
        out[i * 2 + 1] = [fwd.x, fwd.z, wheel.sinkage, if live { 1.0 } else { 0.0 }];
    }
    out
}

fn bike_wheel_model(bike: &Bike, body: Mat4, mud: [[f32; 4]; 12], index: usize) -> ModelData {
    let local = bike.wheel_center_local(index);
    let wheel = bike.wheels[index];
    let radius = BIKE_RADIUS / WHEEL_RADIUS;
    let world = body
        * Mat4::from_translation(local)
        * Mat4::from_rotation_y(wheel.steer)
        * Mat4::from_rotation_x(wheel.spin)
        * Mat4::from_scale(Vec3::new(0.62, radius, radius));
    model(
        world,
        mud,
        [
            1.0,
            (wheel.mud / 1.5).clamp(0.0, 1.4),
            wheel.groove.clamp(0.0, 1.0),
            wheel.smear.clamp(0.0, 1.0),
        ],
    )
}

fn wheel_model(
    vehicle: &Vehicle,
    body: Mat4,
    mud: [[f32; 4]; 12],
    index: usize,
    niva: bool,
) -> ModelData {
    let local = vehicle.wheel_center_local(index);
    let wheel = vehicle.wheels[index];
    // The Niva tire is already the physics radius. The buggy mesh is not:
    // authored front radius is 0.30 and rear is 0.36.
    let world = if niva {
        body * Mat4::from_translation(local)
            * Mat4::from_rotation_y(wheel.steer)
            * Mat4::from_rotation_x(wheel.spin)
    } else {
        let authored = if index < 2 { 0.30 } else { 0.36 };
        let fit = WHEEL_RADIUS / authored;
        body * Mat4::from_translation(local)
            * Mat4::from_rotation_y(wheel.steer)
            * Mat4::from_rotation_x(wheel.spin)
            * Mat4::from_scale(Vec3::splat(fit))
    };
    let mud_n = (wheel.mud / 1.5).clamp(0.0, 1.4);
    // Niva kind 2 carries the contact patch: sidewall, groove fill, lug smear.
    // The procedural tire stays kind 1. Its grooves are the mesh, not this carcass.
    let extra = if niva {
        [
            2.0,
            wheel.sidewall.clamp(0.0, 1.0),
            wheel.groove.clamp(0.0, 1.0),
            wheel.smear.clamp(0.0, 1.0),
        ]
    } else {
        [
            1.0,
            mud_n,
            wheel.groove.clamp(0.0, 1.0),
            wheel.smear.clamp(0.0, 1.0),
        ]
    };
    model(world, mud, extra)
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

fn draw_indexed(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh) {
    pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
    pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.count, 0, 0..1);
}

fn filter_array(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

fn filter_tex(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn array_tex(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2Array,
            multisampled: false,
        },
        count: None,
    }
}

fn filter_samp(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn upload_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    layers: &[crate::cc0::RgbaTex; 5],
) -> (wgpu::Texture, wgpu::TextureView) {
    let width = layers[0].width;
    let height = layers[0].height;
    for layer in layers {
        assert_eq!(layer.width, width);
        assert_eq!(layer.height, height);
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 5,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let stride = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = stride.div_ceil(align) * align;
    for (i, layer) in layers.iter().enumerate() {
        let data: Vec<u8> = if padded == stride {
            layer.rgba.clone()
        } else {
            let mut out = vec![0u8; (padded * height) as usize];
            for y in 0..height as usize {
                let src = y * stride as usize;
                let dst = y * padded as usize;
                out[dst..dst + stride as usize]
                    .copy_from_slice(&layer.rgba[src..src + stride as usize]);
            }
            out
        };
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: i as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

fn upload_rgba_array(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    layers: &[RgbaTex],
) -> (wgpu::Texture, wgpu::TextureView) {
    let width = layers[0].width;
    let height = layers[0].height;
    for layer in layers {
        assert_eq!(layer.width, width);
        assert_eq!(layer.height, height);
    }
    let mips = mip_count(width.max(height));
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: layers.len() as u32,
        },
        mip_level_count: mips,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (i, layer) in layers.iter().enumerate() {
        let mut rgba = layer.rgba.clone();
        let mut mw = width;
        let mut mh = height;
        for level in 0..mips {
            let (padded, stride) = pad_rows(&rgba, mw, mh);
            queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: level,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: i as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(mh),
                },
                wgpu::Extent3d {
                    width: mw,
                    height: mh,
                    depth_or_array_layers: 1,
                },
            );
            if level + 1 == mips {
                break;
            }
            rgba = box_down(&rgba, mw, mh);
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
        }
    }
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some(label),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

fn mip_count(size: u32) -> u32 {
    let mut n = 1u32;
    let mut s = size.max(1);
    while s > 1 {
        s /= 2;
        n += 1;
    }
    n
}

fn box_down(src: &[u8], w: u32, h: u32) -> Vec<u8> {
    let nw = (w / 2).max(1);
    let nh = (h / 2).max(1);
    let mut dst = vec![0u8; (nw * nh * 4) as usize];
    for y in 0..nh {
        for x in 0..nw {
            let mut acc = [0u32; 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    let sx = (x * 2 + dx).min(w - 1);
                    let sy = (y * 2 + dy).min(h - 1);
                    let i = ((sy * w + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += src[i + c] as u32;
                    }
                }
            }
            let o = ((y * nw + x) * 4) as usize;
            for c in 0..4 {
                dst[o + c] = (acc[c] / 4) as u8;
            }
        }
    }
    dst
}

fn pad_rows(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32) {
    let tight = w * 4;
    let stride = tight.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    if stride == tight {
        return (src.to_vec(), stride);
    }
    let mut dst = vec![0u8; (stride * h) as usize];
    for y in 0..h as usize {
        let from = y * tight as usize;
        let to = y * stride as usize;
        dst[to..to + tight as usize].copy_from_slice(&src[from..from + tight as usize]);
    }
    (dst, stride)
}

fn upload_rgba(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    tex: &RgbaTex,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let stride = tex.width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded = stride.div_ceil(align) * align;
    let data: Vec<u8> = if padded == stride {
        tex.rgba.clone()
    } else {
        let mut out = vec![0u8; (padded * tex.height) as usize];
        for y in 0..tex.height as usize {
            let src = y * stride as usize;
            let dst = y * padded as usize;
            out[dst..dst + stride as usize].copy_from_slice(&tex.rgba[src..src + stride as usize]);
        }
        out
    };
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded),
            rows_per_image: Some(tex.height),
        },
        wgpu::Extent3d {
            width: tex.width,
            height: tex.height,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());
    (texture, view)
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
    depth_format: wgpu::TextureFormat,
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
        format: depth_format,
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

fn depth_state(
    format: wgpu::TextureFormat,
    write: bool,
    compare: wgpu::CompareFunction,
) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format,
        depth_write_enabled: write,
        depth_compare: compare,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

fn shadow_depth_state(format: wgpu::TextureFormat) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format,
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

#[cfg(not(target_arch = "wasm32"))]
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
