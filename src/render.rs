//! Thin wgpu runtime. The soil and the buggy are drawn from the solver state:
//! height and material live in textures, mud mass lives in a per-draw uniform.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use glam::{Mat4, Vec3};
use silt::soil::{SoilField, MAP_RES, MAP_SIZE};
use silt::bike::{Bike, BIKE_RADIUS};
use silt::patch::groove_slurry_capacity;
use silt::vehicle::{Vehicle, PANEL_COUNT, PANEL_MUD_CAP, WHEEL_RADIUS, WHEEL_WIDTH};
use silt::world::{CameraPose, World};

use crate::cc0::{buggy_fit, RgbaTex};
use crate::mesh::{
    bike_body, bike_suspension, dust_cloud, environment_mesh, horizon_mesh, mud_chunks_split,
    spray_from_contract, terrain_mesh,
    wheel_mesh, CpuMesh,
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
    /// Irradiance / pi of the environment map, nine RGB SH terms.
    sh: [[f32; 4]; 9],
    /// x: env mip count, y: IBL gain, z: 1 when the sky is an HDRI, w: exposure.
    env: [f32; 4],
    /// Niva headlights: left lamp xyz + intensity, right lamp xyz, beam
    /// direction xyz + cosine of the outer cone. Intensity 0 is off.
    lamps: [[f32; 4]; 3],
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
    env_tex: wgpu::Texture,
    env_sh: [[f32; 4]; 9],
    env_from_file: bool,
    /// Which sky the env map was baked for. None until the first frame.
    env_baked: Option<bool>,
    hdri: Option<Vec<u8>>,
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
    hills: GpuMesh,
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
    dust_v: wgpu::Buffer,
    dust_i: wgpu::Buffer,
    spray_v: wgpu::Buffer,
    spray_i: wgpu::Buffer,
    pipe_sky: wgpu::RenderPipeline,
    pipe_terrain: wgpu::RenderPipeline,
    pipe_water: wgpu::RenderPipeline,
    pipe_mesh: wgpu::RenderPipeline,
    pipe_mesh_blend: wgpu::RenderPipeline,
    pipe_mesh_double: wgpu::RenderPipeline,
    pipe_dust: wgpu::RenderPipeline,
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
    /// Horizon skirt and filler outside the heightfield. Off for solver proof stills.
    pub show_environment: bool,
    /// Distance hills. Off when the horizon must stay a flat field.
    pub show_hills: bool,
    pub show_bike: bool,
    /// Draw the fitted Lada Niva instead of the dune buggy mesh.
    pub niva_picture: bool,
    /// Draw the solved water column, including a bow above the film range.
    pub water_column: bool,
    niva_body: GpuMesh,
    niva_glass: GpuMesh,
    niva_wheels: [GpuMesh; 4],
    /// Versioned physics maps. The picture shaders do not read these.
    contract: Option<silt::render_contract::ContractGpu>,
    contract_cpu: silt::render_contract::ContractFrame,
    pipe_debug: Option<wgpu::RenderPipeline>,
    pipe_fog: Option<wgpu::RenderPipeline>,
    fog_bgl: wgpu::BindGroupLayout,
    fog_bg: wgpu::BindGroup,
    /// Top-down debug view of the render contract. Off for the Niva picture.
    pub debug_contract: bool,
    /// Field shown by the debug view. See `docs/RENDER_CONTRACT.md`.
    pub debug_mode: u32,
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
        // The picture reads the render contract at group 4 of the terrain and
        // water pipelines. A device without storage buffers or a fifth bind
        // group (WebGL2) gets the stand-in readers and keeps its soil cache.
        let contract = silt::render_contract::ContractGpu::try_new(&device)
            .filter(|_| device.limits().max_bind_groups >= 5);
        let readers = if contract.is_some() {
            include_str!("beauty_contract.wgsl")
        } else {
            include_str!("beauty_nocontract.wgsl")
        };
        let source = format!("{}\n{}", include_str!("shader.wgsl"), readers);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silt"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let frame_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[
                uniform_entry(0, false, std::mem::size_of::<FrameData>() as u64),
                filter_tex(1, wgpu::ShaderStages::FRAGMENT),
                filter_samp(2, wgpu::ShaderStages::FRAGMENT),
            ],
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
        let env_levels = crate::env::ENV_W.max(crate::env::ENV_H).ilog2() + 1;
        let env_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("env"),
            size: wgpu::Extent3d {
                width: crate::env::ENV_W,
                height: crate::env::ENV_H,
                depth_or_array_layers: 1,
            },
            mip_level_count: env_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let env_view = env_tex.create_view(&Default::default());
        // Wraps around the horizon, clamps at the poles. Trilinear so a rough
        // reflection blends between the box-filtered levels.
        let env_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("env"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let frame_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&env_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&env_samp),
                },
            ],
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

        let terrain_groups: Vec<&wgpu::BindGroupLayout> = match &contract {
            Some(gpu) => vec![&frame_bgl, &tex_bgl, &model_bgl, &paint_bgl, &gpu.layout],
            None => vec![&frame_bgl, &tex_bgl],
        };
        let terrain_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("terrain-layout"),
            bind_group_layouts: &terrain_groups,
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
        let pipe_dust = pipeline(
            &device,
            &shader,
            mesh_layout_ref,
            "vs_mesh",
            Some("fs_mesh"),
            mesh_buffers(),
            color_format,
            Some(depth_state(
                formats.depth,
                false,
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
        // Dust volume. Only with the contract: it marches the dust_fog grid.
        let fog_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-depth"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let pipe_fog = contract.as_ref().map(|contract_gpu| {
            let fog_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fog-layout"),
                bind_group_layouts: &[&frame_bgl, &tex_bgl, &fog_bgl, &paint_bgl, &contract_gpu.layout],
                push_constant_ranges: &[],
            });
            pipeline(
                &device,
                &shader,
                &fog_layout,
                "vs_fog",
                Some("fs_fog"),
                &[],
                color_format,
                None,
                true,
                None,
            )
        });
        let fog_bg = make_fog_bg(&device, &fog_bgl, &depth_view);

        // Debug view. It reads only the render contract. Compiling it here
        // keeps a contract edit from landing with a shader that does not match.
        let debug_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("render-contract-debug"),
            source: wgpu::ShaderSource::Wgsl(include_str!("render_contract.wgsl").into()),
        });
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let pipe_debug_try = contract.as_ref().map(|contract_gpu| {
            let debug_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("contract-debug"),
                bind_group_layouts: &[&contract_gpu.layout],
                push_constant_ranges: &[],
            });
            pipeline(
                &device,
                &debug_shader,
                &debug_layout,
                "vs_debug",
                Some("fs_debug"),
                &[],
                color_format,
                None,
                false,
                None,
            )
        });
        // A device that cannot sample the contract maps keeps the picture.
        // The shader module above still has to compile.
        #[cfg(not(target_arch = "wasm32"))]
        let pipe_debug = if pollster::block_on(device.pop_error_scope()).is_some() {
            None
        } else {
            pipe_debug_try
        };
        #[cfg(target_arch = "wasm32")]
        let pipe_debug = {
            let fut = device.pop_error_scope();
            wasm_bindgen_futures::spawn_local(async move {
                let _ = fut.await;
            });
            pipe_debug_try
        };
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
        let hills = upload_cpu(&device, &horizon_mesh(), "hills");
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
        let dust_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust-v"),
            size: (16384 * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dust_i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust-i"),
            size: (48000 * 4) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spray_v = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spray-v"),
            size: (24000 * std::mem::size_of::<MeshVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spray_i = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("spray-i"),
            size: (72000 * 4) as u64,
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
            env_tex,
            env_sh: [[0.0; 4]; 9],
            env_from_file: false,
            env_baked: None,
            hdri: crate::env::find_hdri(),
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
            hills,
            props,
            resident,
            paint_bg,
            bike_susp_v,
            bike_susp_i,
            particles,
            chunk_v,
            chunk_i,
            dust_v,
            dust_i,
            spray_v,
            spray_i,
            pipe_sky,
            pipe_terrain,
            pipe_water,
            pipe_mesh,
            pipe_mesh_blend,
            pipe_mesh_double,
            pipe_dust,
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
            show_environment: true,
            show_hills: true,
            show_bike: true,
            niva_picture: false,
            water_column: false,
            niva_body,
            niva_glass,
            niva_wheels,
            contract,
            contract_cpu: silt::render_contract::ContractFrame::new(),
            pipe_debug,
            pipe_fog,
            fog_bgl,
            fog_bg,
            debug_contract: false,
            debug_mode: 0,
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
        self.fog_bg = make_fog_bg(&self.device, &self.fog_bgl, &self.depth_view);
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

    /// Fill the env map for this sky: the HDRI when one was found, else the
    /// analytic sky. Every mip is written; the shader picks one by roughness.
    fn bake_env(&mut self, sky: &crate::env::SkyLight) {
        let map = self
            .hdri
            .as_deref()
            .and_then(|bytes| crate::env::from_hdr(bytes, sky))
            .unwrap_or_else(|| crate::env::procedural(sky));
        for (level, (w, h, px)) in map.levels.iter().enumerate() {
            if level as u32 >= self.env_tex.mip_level_count() {
                break;
            }
            let stride = w * 8;
            let padded = stride.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
                * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            let mut bytes = vec![0u8; (padded * h) as usize];
            for y in 0..*h {
                for x in 0..*w {
                    let c = px[(y * w + x) as usize];
                    let at = (y * padded + x * 8) as usize;
                    for k in 0..4 {
                        let b = crate::env::f16_bits(c[k]).to_le_bytes();
                        bytes[at + k * 2] = b[0];
                        bytes[at + k * 2 + 1] = b[1];
                    }
                }
            }
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.env_tex,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(*h),
                },
                wgpu::Extent3d {
                    width: *w,
                    height: *h,
                    depth_or_array_layers: 1,
                },
            );
        }
        self.env_sh = map.sh;
        self.env_from_file = map.from_file;
    }

    pub fn draw(
        &mut self,
        world: &World,
        cam: CameraPose,
        present: Option<(&wgpu::TextureView, bool)>,
        readback: bool,
    ) -> Option<Vec<u8>> {
        self.draw_with_sheet(world, cam, present, readback, None, None)
    }

    /// Draw the Niva picture. `fluid` and `dust`, when present, are the sheet
    /// the contract publishes. The picture shaders do not read that sheet.
    pub fn draw_with_sheet(
        &mut self,
        world: &World,
        cam: CameraPose,
        present: Option<(&wgpu::TextureView, bool)>,
        readback: bool,
        fluid: Option<&silt::fluid_core::FluidCore>,
        dust: Option<&silt::dust::Dust>,
    ) -> Option<Vec<u8>> {
        silt::render_contract::publish(
            &mut self.contract_cpu,
            &world.soil,
            &world.particles.items,
            [world.vehicle.pos.x, world.vehicle.pos.z],
            world.time,
            self.debug_mode,
            fluid,
            dust,
        );
        if let Some(gpu) = &self.contract {
            gpu.upload(&self.queue, &self.contract_cpu);
        }
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
        if self.env_baked != Some(self.daylight) {
            self.bake_env(&crate::env::SkyLight {
                sun,
                sun_color: Vec3::from_slice(&sun_color[..3]),
                zenith: Vec3::from_slice(&sky_zenith[..3]),
                horizon: Vec3::from_slice(&sky_horizon[..3]),
            });
            self.env_baked = Some(self.daylight);
        }
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
                if self.water_column { 1.0 } else { 0.0 },
            ],
            tires: tire_frame(&world.vehicle),
            sh: self.env_sh,
            env: [
                self.env_tex.mip_level_count() as f32,
                if self.daylight { 0.42 } else { 0.55 },
                if self.env_from_file { 1.0 } else { 0.0 },
                if self.daylight { 0.86 } else { 1.0 },
            ],
            lamps: headlamps(&world.vehicle, if self.daylight || !self.niva_picture { 0.0 } else { 1.0 }),
        };
        self.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        self.write_models(&world.vehicle);
        self.write_bike_models(&world.bike);

        // Liquid in flight comes from the contract's spray slots; the chunk
        // mesh keeps clods and what lies on the ground.
        let spray_valid = self.contract_cpu.header.version == 1
            && self.contract_cpu.header.valid & silt::render_contract::VALID_SPRAY != 0;
        let chunks = mud_chunks_split(&world.particles.items, Some(&world.soil), spray_valid);
        let spray = if spray_valid {
            spray_from_contract(
                &self.contract_cpu.spray,
                self.contract_cpu.header.spray_count as usize,
                cam.eye,
            )
        } else {
            CpuMesh::new()
        };
        let spray_count = spray.indices.len() as u32;
        if spray_count > 0 {
            self.queue
                .write_buffer(&self.spray_v, 0, bytemuck::cast_slice(&spray.vertices));
            self.queue
                .write_buffer(&self.spray_i, 0, bytemuck::cast_slice(&spray.indices));
        }
        let chunk_count = chunks.indices.len() as u32;
        if chunk_count > 0 {
            self.queue
                .write_buffer(&self.chunk_v, 0, bytemuck::cast_slice(&chunks.vertices));
            self.queue
                .write_buffer(&self.chunk_i, 0, bytemuck::cast_slice(&chunks.indices));
        }
        // With the volume pass the airborne dust is the published grid; the
        // picture cards would draw the same dust twice.
        let fog_volume = self.pipe_fog.is_some()
            && self.contract_cpu.header.valid & silt::render_contract::VALID_DUST_FOG != 0;
        let dust = if fog_volume {
            CpuMesh::new()
        } else {
            dust_cloud(&world.particles.items)
        };
        let dust_count = dust.indices.len() as u32;
        if dust_count > 0 {
            self.queue
                .write_buffer(&self.dust_v, 0, bytemuck::cast_slice(&dust.vertices));
            self.queue
                .write_buffer(&self.dust_i, 0, bytemuck::cast_slice(&dust.indices));
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
            if self.show_environment {
                draw_indexed(&mut pass, &self.env);
                if self.show_hills {
                    draw_indexed(&mut pass, &self.hills);
                }
            }
            if self.show_props {
                draw_indexed(&mut pass, &self.props);
            }
            self.draw_vehicle(&mut pass);
            if self.show_bike {
                self.draw_bike(&mut pass, bike_susp_count);
            }
            pass.set_pipeline(&self.pipe_shadow_mesh);
            if chunk_count > 0 {
                self.draw_chunks(&mut pass, chunk_count);
            }
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

            // Floor outside the heightfield, under the terrain so the rim stays the field.
            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_pipeline(&self.pipe_mesh);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            if self.show_environment {
                draw_indexed(&mut pass, &self.env);
            }

            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            if let Some(gpu) = &self.contract {
                pass.set_bind_group(4, &gpu.bind_group, &[]);
            }
            pass.set_pipeline(&self.pipe_terrain);
            draw_indexed(&mut pass, &self.terrain);

            pass.set_pipeline(&self.pipe_mesh);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            if self.show_environment && self.show_hills {
                draw_indexed(&mut pass, &self.hills);
            }
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

            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            if let Some(gpu) = &self.contract {
                pass.set_bind_group(4, &gpu.bind_group, &[]);
            }
            pass.set_pipeline(&self.pipe_water);
            draw_indexed(&mut pass, &self.terrain);

            // Sheets are thin. Back-face cull hides the side the camera sees.
            pass.set_pipeline(&self.pipe_mesh_double);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            if chunk_count > 0 {
                self.draw_chunks(&mut pass, chunk_count);
            }

            pass.set_pipeline(&self.pipe_dust);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            if dust_count > 0 {
                self.draw_dust(&mut pass, dust_count);
            }
            if spray_count > 0 {
                pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
                pass.set_vertex_buffer(0, self.spray_v.slice(..));
                pass.set_index_buffer(self.spray_i.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..spray_count, 0, 0..1);
            }

            pass.set_pipeline(&self.pipe_mesh_blend);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            self.draw_glass(&mut pass);
        }

        if let (true, Some(pipe), Some(gpu)) = (fog_volume, &self.pipe_fog, &self.contract) {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("dust-volume"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
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
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_bind_group(2, &self.fog_bg, &[]);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(4, &gpu.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        if self.debug_contract {
            if let (Some(pipe), Some(gpu)) = (&self.pipe_debug, &self.contract) {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("contract-debug"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.05,
                                g: 0.05,
                                b: 0.05,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                let n = silt::render_contract::WINDOW_CELLS
                    * silt::render_contract::WINDOW_CELLS
                    * 6;
                pass.draw(0..n, 0..1);
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
            let liquid_scale = if self.water_column { 0.90 } else { 0.34 };
            px[1] = ((liquid / liquid_scale).clamp(0.0, 1.0) * 255.0) as u8;
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

    fn draw_dust(&self, pass: &mut wgpu::RenderPass<'_>, count: u32) {
        if count == 0 {
            return;
        }
        pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
        pass.set_vertex_buffer(0, self.dust_v.slice(..));
        pass.set_index_buffer(self.dust_i.slice(..), wgpu::IndexFormat::Uint32);
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
    let slurry_fill = (wheel.slurry_groove
        / groove_slurry_capacity(WHEEL_RADIUS, WHEEL_WIDTH).max(1.0e-6))
    .clamp(0.0, 1.0);
    let fill = wheel.groove.max(slurry_fill).clamp(0.0, 1.0);
    // Niva kind 2 carries the contact patch: sidewall, groove fill, lug smear.
    // The procedural tire stays kind 1. Its grooves are the mesh, not this carcass.
    // Slurry held after a pull-out uses the same void fill as packed clay.
    let extra = if niva {
        [
            2.0,
            wheel.sidewall.clamp(0.0, 1.0),
            fill,
            wheel.smear.clamp(0.0, 1.0),
        ]
    } else {
        [
            1.0,
            mud_n,
            fill,
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
    layers: &[crate::cc0::RgbaTex],
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
            depth_or_array_layers: layers.len() as u32,
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
        // The dust volume pass reads scene depth to stop its march.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
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

#[cfg(all(test, not(target_arch = "wasm32")))]
#[test]
fn debug_contract_view_reads_the_published_maps() {
    prepare_software_vulkan();
    let (_instance, _adapter, device, queue) = open_adapter(true, None);
    let gpu = silt::render_contract::ContractGpu::try_new(&device).expect("contract maps");
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render-contract-debug"),
        source: wgpu::ShaderSource::Wgsl(include_str!("render_contract.wgsl").into()),
    });
    let debug_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("contract-debug-test"),
        bind_group_layouts: &[&gpu.layout],
        push_constant_ranges: &[],
    });
    let pipe = pipeline(
        &device,
        &shader,
        &debug_layout,
        "vs_debug",
        Some("fs_debug"),
        &[],
        wgpu::TextureFormat::Rgba8Unorm,
        None,
        false,
        None,
    );
    let _ = device.poll(wgpu::Maintain::Wait);
    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        panic!("debug contract shader: {err}");
    }

    let soil = silt::soil::SoilField::flat(0.35);
    let mut frame = silt::render_contract::ContractFrame::new();
    silt::render_contract::publish(
        &mut frame,
        &soil,
        &[],
        [0.0, 0.0],
        1.0,
        0,
        None,
        None,
    );
    gpu.upload(&queue, &frame);

    let width = 64u32;
    let height = 64u32;
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("contract-debug-color"),
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
    let view = color.create_view(&Default::default());
    let padded = width * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("contract-debug-readback"),
        size: (padded * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("contract-debug-test"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
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
        pass.set_pipeline(&pipe);
        pass.set_bind_group(0, &gpu.bind_group, &[]);
        let n = silt::render_contract::WINDOW_CELLS * silt::render_contract::WINDOW_CELLS * 6;
        pass.draw(0..n, 0..1);
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
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    queue.submit(Some(encoder.finish()));
    let _ = device.poll(wgpu::Maintain::Wait);
    if let Some(err) = pollster::block_on(device.pop_error_scope()) {
        panic!("debug contract draw: {err}");
    }
    let slice = readback.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    let _ = device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map").expect("map buffer");
    let bytes = slice.get_mapped_range();
    let mid = ((height / 2) * padded + (width / 2) * 4) as usize;
    let px = [bytes[mid], bytes[mid + 1], bytes[mid + 2], bytes[mid + 3]];
    drop(bytes);
    readback.unmap();
    assert_ne!(px, [255, 0, 255, 255], "version mismatch magenta");
    assert!(px[3] == 255, "debug view did not cover the centre {px:?}");
    assert!(px[0] > 0, "ground ramp was empty {px:?}");
}

/// Niva headlamps in world space. The body is fitted with the hubs at
/// y -0.20 and the front axle at z 1.08 (niva.rs); the lamps sit 0.40 m above
/// the hubs and 0.74 m ahead of the axle, 0.56 m either side.
fn headlamps(vehicle: &Vehicle, on: f32) -> [[f32; 4]; 3] {
    let ori = vehicle.orientation();
    let left = vehicle.pos + ori * Vec3::new(-0.56, 0.20, 1.82);
    let right = vehicle.pos + ori * Vec3::new(0.56, 0.20, 1.82);
    // Dipped beam: a few degrees down the road.
    let dir = (ori * Vec3::new(0.0, -0.05, 1.0)).normalize_or_zero();
    [
        [left.x, left.y, left.z, on],
        [right.x, right.y, right.z, 0.0],
        [dir.x, dir.y, dir.z, 0.86],
    ]
}

fn make_fog_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog-depth"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(depth_view),
        }],
    })
}
