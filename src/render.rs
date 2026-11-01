//! Thin wgpu runtime. The soil and the buggy are drawn from the solver state:
//! height and material live in textures, mud mass lives in a per-draw uniform.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use glam::{Mat4, Vec3};
use silt::soil::{crust_shader_texel, SoilField, MAP_RES, MAP_SIZE};
use silt::bike::{Bike, BIKE_RADIUS};
use silt::patch::groove_slurry_capacity;
use silt::vehicle::{Vehicle, PANEL_COUNT, PANEL_MUD_CAP, WHEEL_RADIUS, WHEEL_WIDTH};
use silt::world::{CameraPose, World};

use crate::cc0::{buggy_fit, RgbaTex};
use crate::gpu_time::{pass_timestamps, GpuSample, GpuTime};
use crate::overlay::Overlay;
use crate::terrain_draw::TerrainGeometry;
use crate::mesh::{
    bike_body, bike_suspension, dust_cloud, environment_mesh, horizon_mesh, mud_chunks_with_ground,
    terrain_mesh,
    wheel_mesh, CpuMesh,
    MeshVertex, ParticleVertex, TerrainVertex,
};

/// Sun shadow map side on the `high` preset. The renderer builds with
/// [`silt::quality::active`]`().shadow_res`.
#[allow(dead_code)]
const SHADOW: u32 = 2048;

thread_local! {
    /// Draw calls and triangles encoded since the last [`take_draw_stats`].
    static DRAW_STATS: std::cell::Cell<(u32, u64)> = const { std::cell::Cell::new((0, 0)) };
}

pub(crate) fn count_draw(vertices: u32) {
    DRAW_STATS.with(|c| {
        let (d, t) = c.get();
        c.set((d + 1, t + vertices as u64 / 3));
    });
}

/// Draw calls and triangles since the previous call.
pub fn take_draw_stats() -> (u32, u64) {
    DRAW_STATS.with(|c| c.replace((0, 0)))
}
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
    /// Turn crest. snow_arc2[1] is the bin count; 0 leaves the heightfield alone.
    snow_arc: [f32; 4],
    snow_arc2: [f32; 4],
    snow_profile: [[f32; 4]; 8],
    reflect_view_proj: [[f32; 4]; 4],
    reflect_flags: [f32; 4],
    /// x: water detail (0 low, 1 medium, 2 high). y: 1 when the water_fx
    /// texture holds a fluid sheet. z, w: 0.
    water_fx: [f32; 4],
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
    /// Window size the blit fills. `width × height` is this times the render scale.
    out_width: u32,
    out_height: u32,
    quality: silt::quality::Quality,
    shadow_res: u32,
    overlay: Option<Overlay>,
    model_stride: u32,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    model_buf: wgpu::Buffer,
    model_bg: wgpu::BindGroup,
    tex_bg: wgpu::BindGroup,
    /// Win-demo reflect encode must not bind `reflect_color` while rendering into it.
    tex_bg_pool_encode: wgpu::BindGroup,
    height_bg: wgpu::BindGroup,
    height_tex: wgpu::Texture,
    imprint_tex: wgpu::Texture,
    mat_tex: wgpu::Texture,
    crust_tex: wgpu::Texture,
    /// Turbidity, foam and flow from the render contract, resampled on the
    /// soil grid. Zero without a fluid sheet.
    water_fx_tex: wgpu::Texture,
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    scene_color: wgpu::Texture,
    scene_color_view: wgpu::TextureView,
    reflect_color: wgpu::Texture,
    reflect_color_view: wgpu::TextureView,
    /// Sample-only copy of the pool reflect pass (never a render target).
    reflect_sample: wgpu::Texture,
    reflect_sample_view: wgpu::TextureView,
    reflect_rt: wgpu::Texture,
    reflect_rt_view: wgpu::TextureView,
    scene_depth: wgpu::Texture,
    scene_depth_view: wgpu::TextureView,
    reflect_depth: wgpu::Texture,
    reflect_depth_view: wgpu::TextureView,
    scene_samp: wgpu::Sampler,
    nearest_samp: wgpu::Sampler,
    ground_samp: wgpu::Sampler,
    ground_diff_view: wgpu::TextureView,
    ground_nor_view: wgpu::TextureView,
    ground_pack_view: wgpu::TextureView,
    tex_bgl: wgpu::BindGroupLayout,
    depth: wgpu::Texture,
    depth_view: wgpu::TextureView,
    #[allow(dead_code)]
    shadow: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    #[allow(dead_code)]
    shadow_depth: wgpu::Texture,
    shadow_depth_view: wgpu::TextureView,
    readback: wgpu::Buffer,
    reflect_readback: wgpu::Buffer,
    padded_row: u32,
    terrain: Option<GpuMesh>,
    pub terrain_geom: TerrainGeometry,
    pipe_terrain_lod: wgpu::RenderPipeline,
    pipe_terrain_lod_debug: wgpu::RenderPipeline,
    pipe_water_lod: wgpu::RenderPipeline,
    pipe_shadow_terrain_lod: wgpu::RenderPipeline,
    body: GpuMesh,
    glass: GpuMesh,
    wheels: [GpuMesh; 4],
    wheel: GpuMesh,
    bike_body: GpuMesh,
    env: GpuMesh,
    hills: GpuMesh,
    props: GpuMesh,
    /// Free bodies (balls, crates, walkers), world space. Rebuilt per frame.
    bodies: Option<GpuMesh>,
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
    present_format: Option<wgpu::TextureFormat>,
    mat_scratch: Vec<u8>,
    crust_scratch: Vec<u8>,
    water_fx_scratch: Vec<u8>,
    /// True while the last upload carried a fluid sheet.
    water_fx_live: bool,
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
    /// Win-demo standing puddles: readable depth, sky reflection, tire ripples.
    pub puddle_beauty: bool,
    /// Win-demo PNG stills: skip tire seat on the heightfield (avoids spikes/holes).
    pub win_demo_still: bool,
    /// Latest pool reflect RT readback (win-demo still capture, same resolution as the frame).
    pub win_demo_reflect_rgba: Vec<u8>,
    /// Win-demo still: water shows raw `pool_planar_reflect_sample` (no fresnel/foam).
    pub win_demo_pool_reflect_debug: bool,
    /// Slurry drops and dust cards. Off for win-demo stills (water carries splash).
    pub show_mud_particles: bool,
    /// Round slurry drops. Off when solver sheets carry the splash read.
    pub show_mud_drops: bool,
    /// River or sea shore: the solved column is drawn as the free surface
    /// (bed + depth) with sky reflection, instead of a film on the bed.
    /// Implies the `water_column` pack.
    pub open_water: bool,
    /// Paint the snow layer. Compaction in the darken channel tints the blanket.
    pub snow: bool,
    /// Radial profile of a turn, added on top of `snow_bed` inside the arc.
    pub snow_arc: Option<silt::snow::SnowArc>,
    pub snow_bed: f32,
    niva_body: GpuMesh,
    niva_glass: GpuMesh,
    niva_wheels: [GpuMesh; 4],
    /// Versioned physics maps. The picture shaders do not read these.
    contract: Option<silt::render_contract::ContractGpu>,
    contract_cpu: silt::render_contract::ContractFrame,
    pipe_debug: Option<wgpu::RenderPipeline>,
    /// Top-down debug view of the render contract. Off for the Niva picture.
    pub debug_contract: bool,
    /// Field shown by the debug view. See `docs/RENDER_CONTRACT.md`.
    pub debug_mode: u32,
    /// Set by [`Renderer::enable_gpu_time`]. Absent for the window and for captures.
    gpu_time: Option<GpuTime>,
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

/// Open an adapter for `capture --gpu-time`. Missing adapter is an error the
/// caller prints, not a panic. Timestamp queries are requested when the
/// adapter offers them.
#[cfg(not(target_arch = "wasm32"))]
pub fn try_open_adapter(
    software: bool,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue), String> {
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
    .ok_or_else(|| {
        format!(
            "no wgpu adapter (software={software}). For lavapipe set VK_DRIVER_FILES to lvp_icd.json and unset DISPLAY."
        )
    })?;
    let info = adapter.get_info();
    println!(
        "wgpu adapter: {} {:?} {:?}",
        info.name, info.backend, info.device_type
    );
    let (device, queue) = open_timed_device(&adapter)?;
    Ok((instance, adapter, device, queue))
}

/// Adapter and device on the instance that made `surface`. Timestamp
/// queries are requested when the adapter offers them.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_for_surface(
    instance: &wgpu::Instance,
    surface: &wgpu::Surface<'_>,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), String> {
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: Some(surface),
    }))
    .ok_or_else(|| "no wgpu adapter for the window surface".to_string())?;
    let info = adapter.get_info();
    println!(
        "wgpu adapter: {} {:?} {:?}",
        info.name, info.backend, info.device_type
    );
    let (device, queue) = open_timed_device(&adapter)?;
    Ok((adapter, device, queue))
}

#[cfg(not(target_arch = "wasm32"))]
fn open_timed_device(adapter: &wgpu::Adapter) -> Result<(wgpu::Device, wgpu::Queue), String> {
    if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        let timed = wgpu::DeviceDescriptor {
            label: Some("silt"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        };
        if let Ok(pair) = pollster::block_on(adapter.request_device(&timed, None)) {
            return Ok(pair);
        }
    }
    pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("silt"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|err| format!("request_device: {err}"))
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
        let quality = silt::quality::active();
        let shadow_res = quality.shadow_res;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("silt"),
            source: wgpu::ShaderSource::Wgsl(
                format!("{}\n{}", include_str!("shader.wgsl"), include_str!("terrain_geom.wgsl"))
                    .into(),
            ),
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
        let tg_layout = TerrainGeometry::layout_entries();
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
                // Crust thickness (R) and crack opening (G). The soil shader
                // samples this so a material can read it. It does not retint.
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                tg_layout[0],
                tg_layout[1],
                filter_tex(12, wgpu::ShaderStages::FRAGMENT),
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                filter_samp(14, wgpu::ShaderStages::FRAGMENT),
                filter_tex(15, wgpu::ShaderStages::FRAGMENT),
                // Contract turbidity (R), foam (G) and flow (B, A) on the
                // soil grid. Water passes read it; the solver never does.
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                tg_layout[0],
                tg_layout[1],
            ],
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
        let crust_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("crust"),
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
        let water_fx_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("water-fx"),
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
                width: shadow_res,
                height: shadow_res,
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
                width: shadow_res,
                height: shadow_res,
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
        let terrain_geom = TerrainGeometry::new(&device, !formats.height_packed);
        let tg_tex = terrain_geom.bind_entries();
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
        let [tg_fine, tg_uniform] = terrain_geom.bind_entries();
        let height_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("height-bg"),
            layout: &height_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &height_tex.create_view(&Default::default()),
                    ),
                },
                tg_fine.clone(),
                tg_uniform.clone(),
            ],
        });

        let (color, color_view, depth, depth_view, readback, padded_row) =
            make_targets(&device, width, height, formats.depth);
        let reflect_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflect-readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let (
            scene_color,
            scene_color_view,
            reflect_color,
            reflect_color_view,
            reflect_sample,
            reflect_sample_view,
            reflect_rt,
            reflect_rt_view,
            scene_depth,
            scene_depth_view,
            reflect_depth,
            reflect_depth_view,
        ) = make_scene_targets(&device, width, height, formats.depth);
        let scene_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scene-samp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let tex_bg = build_tex_bg(
            &device,
            &tex_bgl,
            &height_tex,
            &mat_tex,
            &nearest,
            &shadow_view,
            &ground_diff_view,
            &ground_nor_view,
            &ground_pack_view,
            &ground_samp,
            &imprint_tex,
            &crust_tex,
            &scene_color_view,
            &scene_depth_view,
            &scene_samp,
            &reflect_sample_view,
            &water_fx_tex,
            &tg_tex,
        );
        let tex_bg_pool_encode = build_tex_bg(
            &device,
            &tex_bgl,
            &height_tex,
            &mat_tex,
            &nearest,
            &shadow_view,
            &ground_diff_view,
            &ground_nor_view,
            &ground_pack_view,
            &ground_samp,
            &imprint_tex,
            &crust_tex,
            &scene_color_view,
            &scene_depth_view,
            &scene_samp,
            &scene_color_view,
            &water_fx_tex,
            &tg_tex,
        );
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

        let contract = silt::render_contract::ContractGpu::try_new(&device);

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
        let pipe_terrain_lod = pipeline(
            &device,
            &shader,
            terrain_layout_ref,
            "vs_terrain_lod",
            Some("fs_terrain"),
            TerrainGeometry::buffers(),
            color_format,
            Some(depth_state(formats.depth, true, wgpu::CompareFunction::Less)),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_terrain_lod_debug = pipeline(
            &device,
            &shader,
            terrain_layout_ref,
            "vs_terrain_lod_debug",
            Some("fs_terrain_lod_debug"),
            TerrainGeometry::buffers(),
            color_format,
            Some(depth_state(formats.depth, true, wgpu::CompareFunction::Less)),
            false,
            Some(wgpu::Face::Back),
        );
        let pipe_water_lod = pipeline(
            &device,
            &shader,
            terrain_layout_ref,
            "vs_water_lod",
            Some("fs_water"),
            TerrainGeometry::buffers(),
            color_format,
            Some(depth_state(formats.depth, false, wgpu::CompareFunction::Less)),
            true,
            None,
        );
        let pipe_shadow_terrain_lod = pipeline(
            &device,
            &shader,
            &shadow_layout,
            "vs_shadow_terrain_lod",
            Some("fs_shadow"),
            TerrainGeometry::buffers(),
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

        // The old uniform grid is 2.5 M vertices. Only build it when asked for.
        let terrain = if terrain_geom.lod {
            None
        } else {
            Some(legacy_terrain(&device))
        };
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
            tex_bg_pool_encode,
            height_bg,
            height_tex,
            imprint_tex,
            mat_tex,
            crust_tex,
            water_fx_tex,
            color,
            color_view,
            scene_color,
            scene_color_view,
            reflect_color,
            reflect_color_view,
            reflect_sample,
            reflect_sample_view,
            reflect_rt,
            reflect_rt_view,
            scene_depth,
            scene_depth_view,
            reflect_depth,
            reflect_depth_view,
            scene_samp,
            nearest_samp: nearest,
            ground_samp,
            ground_diff_view,
            ground_nor_view,
            ground_pack_view,
            tex_bgl,
            depth,
            depth_view,
            shadow,
            shadow_view,
            shadow_depth,
            shadow_depth_view,
            readback,
            reflect_readback,
            padded_row,
            terrain,
            terrain_geom,
            pipe_terrain_lod,
            pipe_terrain_lod_debug,
            pipe_water_lod,
            pipe_shadow_terrain_lod,
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
            present_format: None,
            mat_scratch: vec![0; MAP_RES * MAP_RES * 4],
            crust_scratch: vec![0; MAP_RES * MAP_RES * 4],
            // Flow 0 m/s is the byte 128.
            water_fx_scratch: [0u8, 0, 128, 128].repeat(MAP_RES * MAP_RES),
            water_fx_live: true,
            imprint_scratch: vec![0; MAP_RES * MAP_RES * 4],
            height_scratch: vec![0.0; MAP_RES * MAP_RES * 2],
            height_bytes: vec![0; MAP_RES * MAP_RES * 4],
            height_packed: formats.height_packed,
            depth_format: formats.depth,
            daylight: false,
            show_props: true,
            bodies: None,
            show_environment: true,
            show_hills: true,
            show_bike: true,
            niva_picture: false,
            water_column: false,
            puddle_beauty: false,
            win_demo_still: false,
            win_demo_reflect_rgba: Vec::new(),
            win_demo_pool_reflect_debug: false,
            show_mud_particles: true,
            show_mud_drops: true,
            open_water: false,
            snow: false,
            snow_arc: None,
            snow_bed: 0.0,
            niva_body,
            niva_glass,
            niva_wheels,
            contract,
            contract_cpu: silt::render_contract::ContractFrame::new(),
            pipe_debug,
            debug_contract: false,
            debug_mode: 0,
            gpu_time: None,
            out_width: width,
            out_height: height,
            quality,
            shadow_res,
            overlay: None,
        }
    }

    pub fn quality(&self) -> silt::quality::Quality {
        self.quality
    }

    /// Switch preset live. Render scale applies now. Shadow resolution is
    /// fixed when the renderer is built; the window prints that it applies
    /// on the next start when it differs.
    pub fn set_quality(&mut self, quality: silt::quality::Quality) -> bool {
        self.quality = quality;
        self.terrain_geom.apply_quality(quality.rut_tess);
        let (w, h) = (self.out_width, self.out_height);
        self.resize_internal(w, h);
        quality.shadow_res == self.shadow_res
    }

    /// Internal target size (after render scale).
    pub fn internal_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Text drawn over the top-left corner of the presented frame. `None` hides it.
    pub fn set_overlay_text(&mut self, lines: Option<&[String]>) {
        let Some(lines) = lines else {
            if let Some(o) = self.overlay.as_mut() {
                o.visible = false;
            }
            return;
        };
        if self.overlay.is_none() {
            let format = self.present_format.unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
            self.overlay = Some(Overlay::new(&self.device, format));
        }
        let o = self.overlay.as_mut().unwrap();
        o.visible = true;
        o.write(&self.queue, lines);
    }

    /// Arm timestamp queries. The window leaves this off.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn enable_gpu_time(&mut self) -> bool {
        if self.gpu_time.is_none() {
            self.gpu_time = GpuTime::try_new(&self.device, &self.queue);
        }
        self.gpu_time.is_some()
    }

    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub fn take_gpu_time(&mut self) -> GpuSample {
        let Self {
            gpu_time, device, ..
        } = self;
        match gpu_time {
            Some(clock) => clock.read(device),
            None => GpuSample::empty(),
        }
    }

    /// Newest finished GPU frame without waiting.
    pub fn poll_gpu_time(&mut self) -> Option<GpuSample> {
        let clock = self.gpu_time.as_ref()?;
        clock.poll(&self.device)
    }

    /// Point the window blit at the swapchain format. Offscreen frames stay linear-in-a-Unorm-buffer
    /// (gamma already applied); an sRGB surface must undo that before it encodes again.
    pub fn set_present_format(&mut self, format: wgpu::TextureFormat) {
        self.present_format = Some(format);
        self.overlay = None;
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

    /// Window size. Internal targets follow at the preset's render scale.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.resize_internal(width, height);
    }

    fn resize_internal(&mut self, out_w: u32, out_h: u32) {
        if out_w == 0 || out_h == 0 {
            return;
        }
        self.out_width = out_w;
        self.out_height = out_h;
        let (width, height) = self.quality.scaled(out_w, out_h);
        if width == self.width && height == self.height {
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
        self.reflect_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflect-readback"),
            size: padded_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
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

    /// The old uniform grid, when the LOD tree is off.
    fn legacy_terrain(&self) -> Option<&GpuMesh> {
        if self.terrain_geom.lod {
            None
        } else {
            self.terrain.as_ref()
        }
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
    /// World-space mesh of the free bodies for the next frames. `None` hides them.
    pub fn set_bodies_mesh(&mut self, mesh: Option<&CpuMesh>) {
        self.bodies = mesh
            .filter(|m| !m.indices.is_empty())
            .map(|m| upload_cpu(&self.device, m, "bodies"));
    }

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
        self.upload_water_fx(fluid.is_some());
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
        let reflect_view_proj = if self.win_demo_still && !self.snow {
            Some(win_demo_reflect_view_proj(cam, proj))
        } else {
            None
        };
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
            if self.win_demo_still {
                (
                    [1.70, 1.78, 1.90, 1.0],
                    [0.10, 0.32, 0.82, 1.0],
                    [0.78, 0.84, 0.92, 1.0],
                    [0.18, 0.20, 0.22, 1.0],
                )
            } else {
                (
                    [1.70, 1.78, 1.90, 1.0],
                    [0.42, 0.64, 0.98, 1.0],
                    [0.76, 0.84, 0.96, 1.0],
                    [0.18, 0.20, 0.22, 1.0],
                )
            }
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
            sun_dir: vec4(sun, if self.snow { 1.0 } else { 0.0 }),
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
                self.shadow_res as f32,
                if self.height_packed { 1.0 } else { 0.0 },
                (if self.water_column || self.open_water { 1.0 } else { 0.0 })
                    + (if self.puddle_beauty { 2.0 } else { 0.0 })
                    + (if self.win_demo_still { 4.0 } else { 0.0 })
                    + (if reflect_view_proj.is_some() { 8.0 } else { 0.0 })
                    + (if self.open_water { 32.0 } else { 0.0 }),
            ],
            tires: tire_frame(&world.vehicle),
            snow_arc: [0.0; 4],
            snow_arc2: [1.0, 0.0, 0.0, 0.0],
            snow_profile: [[0.0; 4]; 8],
            reflect_view_proj: [[0.0; 4]; 4],
            reflect_flags: [0.0; 4],
            water_fx: [
                self.quality.water_detail as f32,
                if self.water_fx_live { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
        };
        let mut frame = frame;
        if let Some(reflect_vp) = reflect_view_proj {
            frame.reflect_view_proj = reflect_vp.to_cols_array_2d();
            frame.reflect_flags = [1.0, 0.0, 0.0, 0.0];
        }
        if self.win_demo_pool_reflect_debug {
            frame.reflect_flags[1] = 1.0;
        }
        if self.snow {
            if let Some(arc) = self.snow_arc {
                frame.snow_arc = [arc.center.x, arc.center.y, arc.sweep, arc.r_lo];
                frame.snow_arc2 = [arc.dr, silt::snow::ARC_BINS as f32, 0.08, arc.sweep - 0.06];
                for i in 0..silt::snow::ARC_BINS {
                    frame.snow_profile[i / 4][i % 4] = arc.depth[i] + self.snow_bed;
                }
            }
        }
        self.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&frame));
        if !self.terrain_geom.lod && self.terrain.is_none() {
            self.terrain = Some(legacy_terrain(&self.device));
        }
        self.terrain_geom
            .prepare(&self.queue, &world.soil, cam.eye, &view_proj, &light_vp);
        self.write_models(&world.vehicle);
        self.write_bike_models(&world.bike);

        let (chunk_count, dust_count) = if self.show_mud_particles {
            let chunks = mud_chunks_with_ground(
                &world.particles.items,
                Some(&world.soil),
                self.show_mud_drops,
            );
            let chunk_count = chunks.indices.len() as u32;
            if chunk_count > 0 {
                self.queue.write_buffer(
                    &self.chunk_v,
                    0,
                    bytemuck::cast_slice(&chunks.vertices),
                );
                self.queue
                    .write_buffer(&self.chunk_i, 0, bytemuck::cast_slice(&chunks.indices));
            }
            let dust = dust_cloud(&world.particles.items);
            let dust_count = dust.indices.len() as u32;
            if dust_count > 0 {
                self.queue
                    .write_buffer(&self.dust_v, 0, bytemuck::cast_slice(&dust.vertices));
                self.queue
                    .write_buffer(&self.dust_i, 0, bytemuck::cast_slice(&dust.indices));
            }
            (chunk_count, dust_count)
        } else {
            (0, 0)
        };

        // Each pass takes one begin/end timestamp pair, labelled with the pass
        // label (`silt::quality::pass_budgets` names the same labels). The
        // clock is out of `self` for the frame.
        let clock = self.gpu_time.take();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });

        {
            let shadow_time = pass_timestamps(clock.as_ref(), "shadow");
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
                timestamp_writes: shadow_time,
                occlusion_query_set: None,
            });
            pass.set_viewport(0.0, 0.0, self.shadow_res as f32, self.shadow_res as f32, 0.0, 1.0);
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(1, &self.height_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[0]);
            if let Some(legacy) = self.legacy_terrain() {
                pass.set_pipeline(&self.pipe_shadow_terrain);
                draw_indexed(&mut pass, legacy);
            } else {
                pass.set_pipeline(&self.pipe_shadow_terrain_lod);
                self.terrain_geom.draw_all(&mut pass);
            }
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
            if let Some(b) = &self.bodies {
                pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
                draw_indexed(&mut pass, b);
            }
        }

        let water_ssr = self.puddle_beauty || self.win_demo_still;
        {
            let main_time = pass_timestamps(clock.as_ref(), "main");
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
                timestamp_writes: main_time,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_pipeline(&self.pipe_sky);
            pass.draw(0..3, 0..1);
            count_draw(3);

            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_pipeline(&self.pipe_mesh);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            if self.show_environment {
                draw_indexed(&mut pass, &self.env);
            }

            pass.set_bind_group(1, &self.tex_bg, &[]);
            if let Some(legacy) = self.legacy_terrain() {
                pass.set_pipeline(&self.pipe_terrain);
                draw_indexed(&mut pass, legacy);
            } else {
                pass.set_pipeline(if self.terrain_geom.debug {
                    &self.pipe_terrain_lod_debug
                } else {
                    &self.pipe_terrain_lod
                });
                self.terrain_geom.draw_view(&mut pass);
            }

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
            // Bodies before the water, so the sheet covers what is under it.
            if let Some(b) = &self.bodies {
                draw_indexed(&mut pass, b);
            }

            if !water_ssr {
                if let Some(legacy) = self.legacy_terrain() {
                    pass.set_pipeline(&self.pipe_water);
                    draw_indexed(&mut pass, legacy);
                } else {
                    pass.set_pipeline(&self.pipe_water_lod);
                    self.terrain_geom.draw_view(&mut pass);
                }
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
                pass.set_pipeline(&self.pipe_mesh_blend);
                pass.set_bind_group(3, &self.paint_bg, &[]);
                self.draw_glass(&mut pass);
            }
        }
        if water_ssr {
            let main_frame = frame;
            if self.win_demo_still {
                self.encode_win_demo_pool_reflect(
                    &mut encoder,
                    reflect_view_proj.expect("win demo reflect vp"),
                    main_frame,
                );
                let extent = wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                };
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.reflect_rt,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &self.reflect_sample,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    extent,
                );
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.reflect_rt,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &self.scene_color,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    extent,
                );
                self.queue.write_buffer(
                    &self.frame_buf,
                    0,
                    bytemuck::bytes_of(&main_frame),
                );
                // Still capture: submit reflect RT before water samples pool_reflect_tex.
                self.queue.submit(Some(encoder.finish()));
                self.device.poll(wgpu::Maintain::Wait);
                self.rebuild_tex_bg_pool_reflect();
                self.queue.write_buffer(
                    &self.frame_buf,
                    0,
                    bytemuck::bytes_of(&main_frame),
                );
                encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("frame-pool-water"),
                    });
            } else {
                let extent = wgpu::Extent3d {
                    width: self.width,
                    height: self.height,
                    depth_or_array_layers: 1,
                };
                encoder.copy_texture_to_texture(
                    wgpu::ImageCopyTexture {
                        texture: &self.color,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::ImageCopyTexture {
                        texture: &self.scene_color,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    extent,
                );
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main-water"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(1, &self.tex_bg, &[]);
            if let Some(legacy) = self.legacy_terrain() {
                pass.set_pipeline(&self.pipe_water);
                draw_indexed(&mut pass, legacy);
            } else {
                pass.set_pipeline(&self.pipe_water_lod);
                self.terrain_geom.draw_view(&mut pass);
            }
            pass.set_pipeline(&self.pipe_mesh_double);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            if chunk_count > 0 {
                self.draw_chunks(&mut pass, chunk_count);
            }
            pass.set_pipeline(&self.pipe_dust);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            if dust_count > 0 {
                self.draw_dust(&mut pass, dust_count);
            }
            pass.set_pipeline(&self.pipe_mesh_blend);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            self.draw_glass(&mut pass);
        }

        if self.debug_contract {
            if let (Some(pipe), Some(gpu)) = (&self.pipe_debug, &self.contract) {
                let debug_time = pass_timestamps(clock.as_ref(), "contract-debug");
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
                    timestamp_writes: debug_time,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, &gpu.bind_group, &[]);
                let n = silt::render_contract::WINDOW_CELLS
                    * silt::render_contract::WINDOW_CELLS
                    * 6;
                pass.draw(0..n, 0..1);
                count_draw(n);
            }
        }

        if let Some((view, srgb)) = present {
            let blit_time = pass_timestamps(clock.as_ref(), "blit");
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
                timestamp_writes: blit_time,
                occlusion_query_set: None,
            });
            pass.set_pipeline(if srgb {
                &self.pipe_blit_srgb
            } else {
                &self.pipe_blit
            });
            pass.set_bind_group(0, &self.blit_bg, &[]);
            pass.draw(0..3, 0..1);
            count_draw(3);
        }

        if let (Some((view, _)), Some(overlay)) = (present, self.overlay.as_ref()) {
            if overlay.visible {
                let overlay_time = pass_timestamps(clock.as_ref(), "overlay");
                overlay.draw(
                    &mut encoder,
                    &self.queue,
                    view,
                    self.out_width,
                    self.out_height,
                    overlay_time,
                );
            }
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

        if let Some(clock) = clock.as_ref() {
            clock.resolve(&mut encoder);
        }
        self.queue.submit(Some(encoder.finish()));
        if let Some(clock) = clock.as_ref() {
            clock.after_submit();
        }
        self.gpu_time = clock;

        if readback && self.win_demo_still && self.win_demo_pool_reflect_debug {
            self.stash_win_demo_reflect_readback();
        }

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
            let liquid_scale = if self.puddle_beauty {
                0.28
            } else if self.win_demo_still {
                0.90
            } else if self.water_column || self.open_water {
                0.90
            } else {
                0.34
            };
            px[1] = ((liquid / liquid_scale).clamp(0.0, 1.0) * 255.0) as u8;
            px[2] = (soil.darken[i].clamp(0.0, 1.0) * 255.0) as u8;
            px[3] = soil.kind[i];
        }
        for (i, px) in self.crust_scratch.chunks_exact_mut(4).enumerate() {
            let texel = crust_shader_texel(soil.crust[i], soil.crack[i]);
            px.copy_from_slice(&texel);
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.crust_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.crust_scratch,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(res * 4),
                rows_per_image: Some(res),
            },
            extent,
        );
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

    /// Resample the published turbidity, foam and flow onto the soil grid.
    /// Reads `contract_cpu` only, after `publish`. Nearest texel: the contract
    /// cell is 0.20 m and the soil cell 0.208 m, so the two grids are close
    /// to one to one and the shader filters.
    fn upload_water_fx(&mut self, sheet: bool) {
        use silt::render_contract::{FLOW_CHANNELS, VALID_FOAM, VALID_TURBIDITY};
        let header = &self.contract_cpu.header;
        let live = sheet && header.valid & (VALID_TURBIDITY | VALID_FOAM) != 0;
        if !live && !self.water_fx_live {
            return;
        }
        self.water_fx_live = live;
        let res = MAP_RES;
        let cell = MAP_SIZE / MAP_RES as f32;
        let half = MAP_SIZE * 0.5;
        let win = silt::render_contract::WINDOW_CELLS as usize;
        let wcell = silt::render_contract::WINDOW_CELL_M;
        let (ox, oz) = (header.origin_x, header.origin_z);
        let turb_ok = header.valid & VALID_TURBIDITY != 0;
        let flow = &self.contract_cpu.flow;
        let foam = &self.contract_cpu.foam_crest;
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        for j in 0..res {
            let z = (j as f32 + 0.5) * cell - half;
            let cj = ((z - oz) / wcell).floor() as i32;
            for i in 0..res {
                let o = (j * res + i) * 4;
                let x = (i as f32 + 0.5) * cell - half;
                let ci = ((x - ox) / wcell).floor() as i32;
                let px = &mut self.water_fx_scratch[o..o + 4];
                if !live || ci < 0 || cj < 0 || ci >= win as i32 || cj >= win as i32 {
                    px.copy_from_slice(&[0, 0, 128, 128]);
                    continue;
                }
                let idx = cj as usize * win + ci as usize;
                let f = idx * FLOW_CHANNELS;
                // sqrt keeps the faint tail of a settling cloud above one byte.
                let t = if turb_ok { flow[f + 3].max(0.0).sqrt() } else { 0.0 };
                px[0] = q(t);
                px[1] = q(foam[idx * 2]);
                // ±2 m/s around the byte 128.
                px[2] = q(flow[f] * 0.25 + 0.5);
                px[3] = q(flow[f + 1] * 0.25 + 0.5);
            }
        }
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.water_fx_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.water_fx_scratch,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(res as u32 * 4),
                rows_per_image: Some(res as u32),
            },
            wgpu::Extent3d {
                width: res as u32,
                height: res as u32,
                depth_or_array_layers: 1,
            },
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

    fn stash_win_demo_reflect_readback(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reflect-readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.reflect_sample,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.reflect_readback,
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
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        let slice = self.reflect_readback.slice(..);
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
        self.reflect_readback.unmap();
        self.win_demo_reflect_rgba = out;
    }

    fn rebuild_tex_bg_pool_reflect(&mut self) {
        self.tex_bg = build_tex_bg(
            &self.device,
            &self.tex_bgl,
            &self.height_tex,
            &self.mat_tex,
            &self.nearest_samp,
            &self.shadow_view,
            &self.ground_diff_view,
            &self.ground_nor_view,
            &self.ground_pack_view,
            &self.ground_samp,
            &self.imprint_tex,
            &self.crust_tex,
            &self.scene_color_view,
            &self.scene_depth_view,
            &self.scene_samp,
            &self.reflect_sample_view,
            &self.water_fx_tex,
            &self.terrain_geom.bind_entries(),
        );
    }

    fn encode_win_demo_pool_reflect(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        reflect_view_proj: Mat4,
        mut reflect_frame: FrameData,
    ) {
        reflect_frame.view_proj = reflect_view_proj.to_cols_array_2d();
        self.queue
            .write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&reflect_frame));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("win-demo-pool-reflect"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.reflect_rt_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.reflect_depth_view,
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
        pass.set_bind_group(1, &self.tex_bg_pool_encode, &[]);
        pass.set_pipeline(&self.pipe_mesh);
        pass.set_bind_group(3, &self.paint_bg, &[]);
        pass.set_bind_group(2, &self.model_bg, &[0]);
        self.draw_vehicle(&mut pass);
        pass.set_pipeline(&self.pipe_mesh_blend);
        pass.set_bind_group(3, &self.paint_bg, &[]);
        pass.set_bind_group(2, &self.model_bg, &[0]);
        self.draw_glass(&mut pass);
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
        count_draw(susp_indices);
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
        count_draw(count);
    }

    fn draw_dust(&self, pass: &mut wgpu::RenderPass<'_>, count: u32) {
        if count == 0 {
            return;
        }
        pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
        pass.set_vertex_buffer(0, self.dust_v.slice(..));
        pass.set_index_buffer(self.dust_i.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..count, 0, 0..1);
        count_draw(count);
    }
}

fn win_demo_reflect_view_proj(cam: CameraPose, proj: Mat4) -> Mat4 {
    let plane = 0.22_f32;
    let flip = |p: Vec3| Vec3::new(p.x, 2.0 * plane - p.y, p.z);
    let eye = flip(cam.eye);
    let target = flip(cam.target);
    let view = Mat4::look_at_rh(eye, target, Vec3::new(0.0, -1.0, 0.0));
    proj * view
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
        let spin = wheel.omega.abs();
        // Odd slot w carries wheel spin (rad/s) when driving; otherwise 1/0 live flag.
        let active = if spin > 0.35 { spin } else if live { 1.0 } else { 0.0 };
        out[i * 2] = [center.x, center.y, center.z, WHEEL_RADIUS];
        out[i * 2 + 1] = [fwd.x, fwd.z, wheel.sinkage, active];
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

fn legacy_terrain(device: &wgpu::Device) -> GpuMesh {
    let (tverts, tidx) = terrain_mesh(MAP_RES, MAP_SIZE);
    upload_mesh(device, &tverts, &tidx, "terrain")
}

fn draw_indexed(pass: &mut wgpu::RenderPass<'_>, mesh: &GpuMesh) {
    pass.set_vertex_buffer(0, mesh.vbuf.slice(..));
    pass.set_index_buffer(mesh.ibuf.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..mesh.count, 0, 0..1);
    count_draw(mesh.count);
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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

fn make_scene_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    depth_format: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
) {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let scene_color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-color"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let scene_color_view = scene_color.create_view(&Default::default());
    let reflect_color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflect-color"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let reflect_color_view = reflect_color.create_view(&Default::default());
    let reflect_sample = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflect-sample"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let reflect_sample_view = reflect_sample.create_view(&Default::default());
    let reflect_rt = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflect-rt"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let reflect_rt_view = reflect_rt.create_view(&Default::default());
    let scene_depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let scene_depth_view = scene_depth.create_view(&Default::default());
    let reflect_depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("reflect-depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: depth_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let reflect_depth_view = reflect_depth.create_view(&Default::default());
    (
        scene_color,
        scene_color_view,
        reflect_color,
        reflect_color_view,
        reflect_sample,
        reflect_sample_view,
        reflect_rt,
        reflect_rt_view,
        scene_depth,
        scene_depth_view,
        reflect_depth,
        reflect_depth_view,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_tex_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    height_tex: &wgpu::Texture,
    mat_tex: &wgpu::Texture,
    nearest: &wgpu::Sampler,
    shadow_view: &wgpu::TextureView,
    ground_diff_view: &wgpu::TextureView,
    ground_nor_view: &wgpu::TextureView,
    ground_pack_view: &wgpu::TextureView,
    ground_samp: &wgpu::Sampler,
    imprint_tex: &wgpu::Texture,
    crust_tex: &wgpu::Texture,
    scene_color_view: &wgpu::TextureView,
    scene_depth_view: &wgpu::TextureView,
    scene_samp: &wgpu::Sampler,
    pool_reflect_view: &wgpu::TextureView,
    water_fx_tex: &wgpu::Texture,
    tg: &[wgpu::BindGroupEntry<'_>; 2],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tex-bg"),
        layout,
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
                resource: wgpu::BindingResource::Sampler(nearest),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(ground_diff_view),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(ground_nor_view),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(ground_pack_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::Sampler(ground_samp),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: wgpu::BindingResource::TextureView(
                    &imprint_tex.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: wgpu::BindingResource::TextureView(
                    &crust_tex.create_view(&Default::default()),
                ),
            },
            tg[0].clone(),
            tg[1].clone(),
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(scene_color_view),
            },
            wgpu::BindGroupEntry {
                binding: 13,
                resource: wgpu::BindingResource::TextureView(scene_depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 14,
                resource: wgpu::BindingResource::Sampler(scene_samp),
            },
            wgpu::BindGroupEntry {
                binding: 15,
                resource: wgpu::BindingResource::TextureView(pool_reflect_view),
            },
            wgpu::BindGroupEntry {
                binding: 16,
                resource: wgpu::BindingResource::TextureView(
                    &water_fx_tex.create_view(&Default::default()),
                ),
            },
        ],
    })
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
