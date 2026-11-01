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
    bike_body, bike_suspension, dust_cloud, environment_mesh, horizon_mesh, mud_chunks_split,
    spray_from_contract, terrain_mesh,
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
    /// Irradiance / pi of the environment map, nine RGB SH terms.
    sh: [[f32; 4]; 9],
    /// x: env mip count, y: IBL gain, z: 1 when the sky is an HDRI, w: exposure.
    env: [f32; 4],
    /// Niva headlights: left lamp xyz + intensity, right lamp xyz, beam
    /// direction xyz + cosine of the outer cone. Intensity 0 is off.
    lamps: [[f32; 4]; 3],
    /// Air for the dust and fog volume, picture only. 0: wind xyz (m/s),
    /// ground fog extinction (1/m, 0 off). 1: Niva position xyz, fog scale
    /// height (m). 2: Niva velocity xyz, yaw.
    air: [[f32; 4]; 3],
    /// Near shadow cascade: a tight ortho round the Niva, drawn into the
    /// right half of the shadow atlas. The far cascade is `light_vp`.
    light_vp_near: [[f32; 4]; 4],
    /// Water surface height at each tire, metres: soil plus the solved film
    /// round the carcass. Below -100 when no sheet was given.
    tire_water: [f32; 4],
    /// x: 1 draws the depth debug (water by depth, tire bottoms marked).
    debug: [f32; 4],
    /// Last frame's view-projection, for reprojecting the SSR history.
    prev_view_proj: [[f32; 4]; 4],
    /// x: 1 when the SSR history holds a valid previous frame. y, z: the
    /// half-resolution SSR target size in pixels.
    ssr_info: [f32; 4],
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
    ssr: SsrTargets,
    post_bg_ssr: wgpu::BindGroup,
    pipe_water_ssr: wgpu::RenderPipeline,
    ssr_prev_vp: [[f32; 4]; 4],
    ssr_hist_valid: f32,
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
    /// The frame with `light_vp` swapped for the near cascade, for its
    /// shadow pass only.
    frame_near_buf: wgpu::Buffer,
    frame_near_bg: wgpu::BindGroup,
    /// The frame mirrored about the pool's water level, for the planar
    /// reflection of the Niva (wgpu water example / Ultra-Water).
    frame_refl_buf: wgpu::Buffer,
    frame_refl_bg: wgpu::BindGroup,
    planar: Planar,
    env_tex: wgpu::Texture,
    env_sh: [[f32; 4]; 9],
    env_from_file: bool,
    /// Which sky the env map was baked for. None until the first frame.
    env_baked: Option<(bool, [u32; 3])>,
    hdri: Option<Vec<u8>>,
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
    /// Format of `color` and of the scene copies (see [`GpuFormats::scene`]).
    scene_format: wgpu::TextureFormat,
    /// Tone-mapped 8-bit copy of `color`. `None` when `color` is already 8-bit.
    ldr: Option<(wgpu::Texture, wgpu::TextureView)>,
    tonemap_bg: Option<wgpu::BindGroup>,
    pipe_tonemap: wgpu::RenderPipeline,
    /// Post-processing prototype (`SILT_POST=1`, docs/LOOK_GAP.md). When on,
    /// the blit and the capture readback read its output instead of `color`.
    post: Option<crate::post::Post>,
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
    /// Blit that tone-maps the HDR target on the way to the window
    /// (plain, to-sRGB). Saves the separate tonemap pass when nothing
    /// else needs the 8-bit copy.
    pipe_blit_hdr: [wgpu::RenderPipeline; 2],
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
    /// Lay the undisturbed field (kind 0) and the skirt as sand. Picture only.
    pub sand_field: bool,
    /// Ground fog for the picture: extinction at the ground (1/m, 0 off) and
    /// scale height (m). It drifts with `wind` and parts round the Niva.
    pub fog: f32,
    pub fog_height: f32,
    /// A still can place the sun (unit vector toward it); None keeps the
    /// day or dusk sun.
    pub sun_override: Option<Vec3>,
    /// Depth debug: water coloured by depth, tire bottoms marked.
    pub debug_depth: f32,
    tire_water: [f32; 4],
    pub wind: Vec3,
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
    pipe_fog: Option<wgpu::RenderPipeline>,
    post_bgl: wgpu::BindGroupLayout,
    post_bg: wgpu::BindGroup,
    air_noise_view: wgpu::TextureView,
    air_samp: wgpu::Sampler,
    scene_copy: wgpu::Texture,
    scene_copy_view: wgpu::TextureView,
    /// The dust and fog march runs at half resolution into this target and
    /// is upsampled over the frame (`pipe_fog_up`): a quarter of the pixels.
    fog_half_view: wgpu::TextureView,
    fog_up_bgl: wgpu::BindGroupLayout,
    fog_up_bg: wgpu::BindGroup,
    pipe_fog_up: Option<wgpu::RenderPipeline>,
    /// `pipe_fog_up` into the HDR target, for `SILT_POST`. Without post the
    /// cloud goes over the tone-mapped 8-bit picture, as it was tuned.
    pipe_fog_up_hdr: Option<wgpu::RenderPipeline>,
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
    /// Scene colour target. `Rgba16Float` (linear light, tone curve in the
    /// tonemap pass) when the adapter can render, blend and filter it;
    /// otherwise `Rgba8Unorm` with the curve applied in the scene shaders.
    pub scene: wgpu::TextureFormat,
}

/// Final 8-bit picture: what the blit, the overlay and the capture read.
const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// `SILT_HDR=0` forces the old 8-bit scene target (A/B checks, odd drivers).
fn hdr_allowed() -> bool {
    std::env::var("SILT_HDR").map(|v| v.trim() != "0").unwrap_or(true)
}

fn pick_scene_format(adapter: &wgpu::Adapter) -> wgpu::TextureFormat {
    let hdr = wgpu::TextureFormat::Rgba16Float;
    let f = adapter.get_texture_format_features(hdr);
    let usage = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC
        | wgpu::TextureUsages::COPY_DST;
    let flags = wgpu::TextureFormatFeatureFlags::BLENDABLE
        | wgpu::TextureFormatFeatureFlags::FILTERABLE;
    if hdr_allowed() && f.allowed_usages.contains(usage) && f.flags.contains(flags) {
        hdr
    } else {
        LDR_FORMAT
    }
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
            scene: pick_scene_format(adapter),
        }
    } else {
        GpuFormats {
            depth,
            height: wgpu::TextureFormat::Rgba8Unorm,
            height_packed: true,
            scene: pick_scene_format(adapter),
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
        let source = format!(
            "{}\n{}\n{}",
            include_str!("shader.wgsl"),
            include_str!("terrain_geom.wgsl"),
            readers
        );
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
        let frame_near_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-near"),
            size: std::mem::size_of::<FrameData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_near_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-near-bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_near_buf.as_entire_binding(),
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
        let frame_refl_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame-refl"),
            size: std::mem::size_of::<FrameData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_refl_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("frame-refl-bg"),
            layout: &frame_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: frame_refl_buf.as_entire_binding(),
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
                // Two cascades side by side: far left, near (round the Niva) right.
                width: shadow_res * 2,
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
                // Two cascades side by side: far left, near (round the Niva) right.
                width: shadow_res * 2,
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

        let scene_format = formats.scene;
        let (color, color_view, depth, depth_view, readback, padded_row, ldr) =
            make_targets(&device, width, height, formats.depth, scene_format);
        let post = crate::post::enabled().then(|| {
            crate::post::Post::new(&device, &quality, width, height, &color_view, ldr.is_some())
        });
        let reflect_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflect-readback"),
            size: reflect_row(width, scene_format) as u64 * height as u64,
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
        ) = make_scene_targets(&device, width, height, formats.depth, scene_format);
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
                    resource: wgpu::BindingResource::TextureView(final_view(
                        post.as_ref(),
                        ldr.as_ref(),
                        &color_view,
                    )),
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
        // Scene depth and a copy of the opaque frame, for the water's screen
        // reflections and the dust volume's march.
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Read as unfilterable float: GLSL ES has no texelFetch on
                        // depth samplers, so WebGL2 needs this for the SSR pass.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Half-resolution SSR: reflection (rgb premultiplied, a weight)
                // and the view distance it was traced from.
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Planar reflection of the Niva about the water level.
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let water_groups: Vec<&wgpu::BindGroupLayout> = match &contract {
            Some(gpu) => vec![&frame_bgl, &tex_bgl, &post_bgl, &paint_bgl, &gpu.layout],
            None => vec![&frame_bgl, &tex_bgl, &post_bgl],
        };
        let water_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water-layout"),
            bind_group_layouts: &water_groups,
            push_constant_ranges: &[],
        });
        let (scene_copy, scene_copy_view) = make_scene_copy(&device, width, height, scene_format);
        let fog_up_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fog-up"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 20,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 21,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let (fog_half_view, fog_up_bg) = make_fog_half(&device, &fog_up_bgl, width, height);
        let (air_noise_view, air_samp) = make_air_noise(&device, &queue);
        let ssr = make_ssr(&device, width, height);
        let planar = make_planar(&device, width, height, formats.depth, scene_format);
        let post_bg = make_post_bg(
            &device,
            &post_bgl,
            &depth_view,
            &scene_copy_view,
            &air_noise_view,
            &air_samp,
            &ssr.a_view,
            &ssr.b_view,
            &planar.view,
        );
        let post_bg_ssr = make_post_bg(
            &device,
            &post_bgl,
            &depth_view,
            &scene_copy_view,
            &air_noise_view,
            &air_samp,
            &ssr.hist_a_view,
            &ssr.hist_b_view,
            &planar.view,
        );
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

        let color_format = scene_format;
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
            &water_layout,
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
        // Half-resolution SSR trace of the water surface: two targets (the
        // reflection and its view distance) with its own depth so the
        // nearest sheet wins.
        let ssr_target = Some(wgpu::ColorTargetState {
            format: SSR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        });
        let pipe_water_ssr = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("water-ssr"),
            layout: Some(&water_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_water"),
                compilation_options: Default::default(),
                buffers: terrain_buffers(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_water_ssr"),
                compilation_options: Default::default(),
                targets: &[ssr_target.clone(), ssr_target],
            }),
            multiview: None,
            cache: None,
        });
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
            LDR_FORMAT,
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
            &water_layout,
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
            LDR_FORMAT,
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
            LDR_FORMAT,
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
            LDR_FORMAT,
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
            LDR_FORMAT,
            None,
            false,
            None,
        );
        // Dust volume. Only with the contract: it marches the dust_fog grid.
        let pipe_fog = contract.as_ref().map(|contract_gpu| {
            let fog_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fog-layout"),
                bind_group_layouts: &[&frame_bgl, &tex_bgl, &post_bgl, &paint_bgl, &contract_gpu.layout],
                push_constant_ranges: &[],
            });
            pipeline(
                &device,
                &shader,
                &fog_layout,
                "vs_fog",
                Some("fs_fog"),
                &[],
                LDR_FORMAT,
                None,
                false,
                None,
            )
        });
        let pipe_fog_up = contract.as_ref().map(|_| {
            let up_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fog-up-layout"),
                bind_group_layouts: &[&fog_up_bgl],
                push_constant_ranges: &[],
            });
            pipeline(
                &device,
                &shader,
                &up_layout,
                "vs_fog",
                Some("fs_fog_up"),
                &[],
                LDR_FORMAT,
                None,
                true,
                None,
            )
        });
        // Over the linear HDR target (only when SILT_POST reads it).
        let pipe_fog_up_hdr = contract.as_ref().map(|_| {
            let up_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fog-up-layout"),
                bind_group_layouts: &[&fog_up_bgl],
                push_constant_ranges: &[],
            });
            pipeline(
                &device,
                &shader,
                &up_layout,
                "vs_fog",
                Some("fs_fog_up_hdr"),
                &[],
                color_format,
                None,
                true,
                None,
            )
        });

        let pipe_blit_hdr = blit_hdr_pipes(&device, &blit_shader, &blit_layout, LDR_FORMAT);
        // HDR scene -> 8-bit picture. Same curve as `aces_fit` in shader.wgsl.
        let pipe_tonemap = pipeline(
            &device,
            &blit_shader,
            &blit_layout,
            "vs_blit",
            Some("fs_tonemap"),
            &[],
            LDR_FORMAT,
            None,
            false,
            None,
        );
        let tonemap_bg = ldr
            .as_ref()
            .map(|_| tonemap_bind_group(&device, &blit_bgl, &color_view, &nearest));
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
                LDR_FORMAT,
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
            frame_near_buf,
            frame_near_bg,
            frame_refl_buf,
            frame_refl_bg,
            planar,
            env_tex,
            env_sh: [[0.0; 4]; 9],
            env_from_file: false,
            env_baked: None,
            hdri: crate::env::find_hdri(),
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
            scene_format,
            ldr,
            tonemap_bg,
            pipe_tonemap,
            post,
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
            pipe_blit_hdr,
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
            sand_field: false,
            fog: 0.0,
            fog_height: 1.4,
            sun_override: None,
            debug_depth: 0.0,
            tire_water: [-1000.0; 4],
            wind: Vec3::new(1.1, 0.0, -0.65),
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
            pipe_fog,
            pipe_fog_up,
            pipe_fog_up_hdr,
            fog_half_view,
            fog_up_bgl,
            fog_up_bg,
            post_bgl,
            post_bg,
            air_noise_view,
            air_samp,
            scene_copy,
            scene_copy_view,
            ssr,
            post_bg_ssr,
            pipe_water_ssr,
            ssr_prev_vp: Mat4::IDENTITY.to_cols_array_2d(),
            ssr_hist_valid: 0.0,
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
        self.pipe_blit_hdr = blit_hdr_pipes(&self.device, &self.blit_shader, &self.blit_layout, format);
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
        let (color, color_view, depth, depth_view, readback, padded_row, ldr) =
            make_targets(&self.device, width, height, self.depth_format, self.scene_format);
        self.color = color;
        self.color_view = color_view;
        self.ldr = ldr;
        if let Some(post) = self.post.as_mut() {
            post.resize(&self.device, width, height, &self.color_view);
        }
        self.depth = depth;
        self.depth_view = depth_view;
        self.readback = readback;
        self.reflect_readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reflect-readback"),
            size: reflect_row(width, self.scene_format) as u64 * height as u64,
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
        let (scene_copy, scene_copy_view) = make_scene_copy(&self.device, width, height, self.scene_format);
        let (fog_half_view, fog_up_bg) = make_fog_half(&self.device, &self.fog_up_bgl, width, height);
        self.fog_half_view = fog_half_view;
        self.fog_up_bg = fog_up_bg;
        self.scene_copy = scene_copy;
        self.scene_copy_view = scene_copy_view;
        self.ssr = make_ssr(&self.device, width, height);
        self.planar = make_planar(&self.device, width, height, self.depth_format, self.scene_format);
        self.ssr_hist_valid = 0.0;
        self.post_bg = make_post_bg(
            &self.device,
            &self.post_bgl,
            &self.depth_view,
            &self.scene_copy_view,
            &self.air_noise_view,
            &self.air_samp,
            &self.ssr.a_view,
            &self.ssr.b_view,
            &self.planar.view,
        );
        self.post_bg_ssr = make_post_bg(
            &self.device,
            &self.post_bgl,
            &self.depth_view,
            &self.scene_copy_view,
            &self.air_noise_view,
            &self.air_samp,
            &self.ssr.hist_a_view,
            &self.ssr.hist_b_view,
            &self.planar.view,
        );
        // Rebuild the blit bind group against the new color view. Layout is recovered from the pipeline.
        let blit_bgl = self.pipe_blit.get_bind_group_layout(0);
        self.tonemap_bg = self.ldr.as_ref().map(|_| {
            let layout = self.pipe_tonemap.get_bind_group_layout(0);
            tonemap_bind_group(&self.device, &layout, &self.color_view, &nearest)
        });
        self.blit_bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit-bg"),
            layout: &blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(final_view(
                        self.post.as_ref(),
                        self.ldr.as_ref(),
                        &self.color_view,
                    )),
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
        // The HDRI is a midday sky; the low dusk sun keeps the analytic sky
        // it was tuned under.
        let map = self
            .hdri
            .as_deref()
            .filter(|_| sky.physical)
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
        self.tire_water = match fluid {
            Some(fluid) => tire_waterlines(world, fluid),
            None => [-1000.0; 4],
        };
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
        let sun = if let Some(s) = self.sun_override {
            s.normalize()
        } else if self.daylight {
            // 49 degrees up, the sun's height in assets/hdri/sky.hdr.
            Vec3::new(0.593, 0.755, 0.281).normalize()
        } else {
            Vec3::new(0.88, 0.32, 0.24).normalize()
        };
        let light_target = if self.daylight {
            Vec3::new(0.4, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 0.0, 2.0)
        };
        let mut extent = if self.daylight { 14.0 } else { 36.0 };
        // The beauty drive keeps the shadow map on the Niva: a 36 m low-sun
        // map saws the body's shadow edge into teeth.
        let light_target = if self.sand_field {
            extent = 16.0;
            Vec3::new(world.vehicle.pos.x, 0.0, world.vehicle.pos.z)
        } else {
            light_target
        };
        let light_view = Mat4::look_at_rh(light_target + sun * 48.0, light_target, Vec3::Y);
        let light_proj = Mat4::orthographic_rh(-extent, extent, -extent, extent, 1.0, 130.0);
        let light_vp = light_proj * light_view;
        // Near cascade: 7 m round the Niva at ~3 mm a texel, so the tires
        // and sills print a sharp contact shadow where they meet the ground.
        let near_target = Vec3::new(
            world.vehicle.pos.x,
            world.vehicle.pos.y - 0.4,
            world.vehicle.pos.z,
        );
        let light_vp_near = Mat4::orthographic_rh(-3.5, 3.5, -3.5, 3.5, 1.0, 24.0)
            * Mat4::look_at_rh(near_target + sun * 12.0, near_target, Vec3::Y);
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
        // The HDRI is turned to the sun, so a moved sun rebakes it.
        let env_key = (self.daylight, sun.to_array().map(f32::to_bits));
        if self.env_baked != Some(env_key) {
            self.bake_env(&crate::env::SkyLight {
                sun,
                sun_color: Vec3::from_slice(&sun_color[..3]),
                zenith: Vec3::from_slice(&sky_zenith[..3]),
                horizon: Vec3::from_slice(&sky_horizon[..3]),
                physical: self.daylight,
            });
            self.env_baked = Some(env_key);
        }
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
            sh: self.env_sh,
            env: [
                self.env_tex.mip_level_count() as f32,
                if self.daylight { 0.42 } else { 0.55 },
                if self.env_from_file { 1.0 } else { 0.0 },
                if self.daylight { 0.86 } else { 1.0 },
            ],
            lamps: headlamps(
                &world.vehicle,
                if !self.niva_picture { 0.0 } else if self.daylight { 0.6 } else { 1.0 },
                self.sand_field,
            ),
            air: [
                [self.wind.x, self.wind.y, self.wind.z, self.fog],
                [
                    world.vehicle.pos.x,
                    world.vehicle.pos.y,
                    world.vehicle.pos.z,
                    self.fog_height,
                ],
                [
                    world.vehicle.vel.x,
                    world.vehicle.vel.y,
                    world.vehicle.vel.z,
                    world.vehicle.yaw,
                ],
            ],
            light_vp_near: if self.niva_picture {
                light_vp_near.to_cols_array_2d()
            } else {
                // No near cascade: a matrix that maps everything off the map.
                Mat4::from_translation(Vec3::new(9.0, 9.0, 0.0)).to_cols_array_2d()
            },
            tire_water: self.tire_water,
            debug: [self.debug_depth, 0.0, 0.0, 0.0],
            prev_view_proj: self.ssr_prev_vp,
            ssr_info: [
                self.ssr_hist_valid,
                self.ssr.width as f32,
                self.ssr.height as f32,
                0.0,
            ],
            reflect_view_proj: [[0.0; 4]; 4],
            reflect_flags: [0.0; 4],
            water_fx: [
                self.quality.water_detail as f32,
                if self.water_fx_live { 1.0 } else { 0.0 },
                0.0,
                // 1: linear HDR scene target, tone curve in the tonemap pass.
                if self.scene_format != LDR_FORMAT { 1.0 } else { 0.0 },
            ],
        };
        self.ssr_prev_vp = view_proj.to_cols_array_2d();
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
        let mut frame_near = frame;
        frame_near.light_vp = frame.light_vp_near;
        self.queue
            .write_buffer(&self.frame_near_buf, 0, bytemuck::bytes_of(&frame_near));
        // Planar reflection about the water level the tires stand in (the
        // solved sheet's level, tire_waterlines). No standing water: no pass.
        let levels: Vec<f32> = self.tire_water.iter().copied().filter(|w| *w > -100.0).collect();
        let refl_level = (!levels.is_empty() && std::env::var_os("SILT_NO_PLANAR").is_none())
            .then(|| levels.iter().sum::<f32>() / levels.len() as f32);
        if let Some(h) = refl_level {
            let mirror = Mat4::from_translation(Vec3::new(0.0, h, 0.0))
                * Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0))
                * Mat4::from_translation(Vec3::new(0.0, -h, 0.0));
            let mut frame_refl = frame;
            let vp_r = view_proj * mirror;
            frame_refl.view_proj = vp_r.to_cols_array_2d();
            frame_refl.inv_view_proj = vp_r.inverse().to_cols_array_2d();
            frame_refl.cam_pos = vec4(Vec3::new(cam.eye.x, 2.0 * h - cam.eye.y, cam.eye.z), 1.0);
            frame_refl.debug = [frame.debug[0], 1.0, h, 0.0];
            self.queue
                .write_buffer(&self.frame_refl_buf, 0, bytemuck::bytes_of(&frame_refl));
        }
        if !self.terrain_geom.lod && self.terrain.is_none() {
            self.terrain = Some(legacy_terrain(&self.device));
        }
        self.terrain_geom
            .prepare(&self.queue, &world.soil, cam.eye, &view_proj, &light_vp);
        self.write_models(&world.vehicle);
        self.write_bike_models(&world.bike);

        // Liquid in flight comes from the contract's spray slots; the chunk
        // mesh keeps clods and what lies on the ground.
        let spray_valid = self.contract_cpu.header.version == 1
            && self.contract_cpu.header.valid & silt::render_contract::VALID_SPRAY != 0;
        let chunks = mud_chunks_split(&world.particles.items, Some(&world.soil), spray_valid, self.show_mud_drops);
        let spray = if spray_valid {
            spray_from_contract(
                &self.contract_cpu.spray,
                self.contract_cpu.header.spray_count as usize,
                cam.eye,
            )
        } else {
            CpuMesh::new()
        };
        let spray_count = if self.show_mud_particles { spray.indices.len() as u32 } else { 0 };
        if spray_count > 0 {
            self.queue
                .write_buffer(&self.spray_v, 0, bytemuck::cast_slice(&spray.vertices));
            self.queue
                .write_buffer(&self.spray_i, 0, bytemuck::cast_slice(&spray.indices));
        }
        let chunk_count = if self.show_mud_particles { chunks.indices.len() as u32 } else { 0 };
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
        let dust_count = if self.show_mud_particles { dust.indices.len() as u32 } else { 0 };
        if dust_count > 0 {
            self.queue
                .write_buffer(&self.dust_v, 0, bytemuck::cast_slice(&dust.vertices));
            self.queue
                .write_buffer(&self.dust_i, 0, bytemuck::cast_slice(&dust.indices));
        }

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

        if self.niva_picture {
            let near_time = pass_timestamps(clock.as_ref(), "shadow-near");
            // Near cascade, right half of the atlas: the ground under the
            // Niva and the Niva itself.
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow-near"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.shadow_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: near_time,
                occlusion_query_set: None,
            });
            let sr = self.shadow_res as f32;
            pass.set_viewport(sr, 0.0, sr, sr, 0.0, 1.0);
            pass.set_bind_group(0, &self.frame_near_bg, &[]);
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
            self.draw_vehicle(&mut pass);
        }

        let refl_level = refl_level.filter(|_| self.quality.planar_reflection);
        if refl_level.is_some() {
            let planar_time = pass_timestamps(clock.as_ref(), "planar");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("planar"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.planar.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.planar.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: planar_time,
                occlusion_query_set: None,
            });
            {
                pass.set_bind_group(0, &self.frame_refl_bg, &[]);
                pass.set_bind_group(1, &self.tex_bg, &[]);
                pass.set_bind_group(3, &self.paint_bg, &[]);
                // The mirror flips the winding: draw both faces.
                pass.set_pipeline(&self.pipe_mesh_double);
                self.draw_vehicle(&mut pass);
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
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);
            pass.set_bind_group(3, &self.paint_bg, &[]);
            if let Some(gpu) = &self.contract {
                pass.set_bind_group(4, &gpu.bind_group, &[]);
            }
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
        } else {

        // The water reflects the opaque frame: copy it, then draw the water
        // with the depth read-only so the same depth can be sampled.
        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &self.color,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &self.scene_copy,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        {
            let ssr_time = pass_timestamps(clock.as_ref(), "water-ssr");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water-ssr"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr.a_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &self.ssr.b_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.ssr.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_bind_group(2, &self.post_bg_ssr, &[]);
            if let Some(gpu) = &self.contract {
                pass.set_bind_group(3, &self.paint_bg, &[]);
                pass.set_bind_group(4, &gpu.bind_group, &[]);
            }
            pass.set_pipeline(&self.pipe_water_ssr);
            // The SSR trace runs on the uniform grid only.
            if let (true, Some(legacy)) = (std::env::var("SILT_NO_SSR").is_err(), self.legacy_terrain()) {
                draw_indexed(&mut pass, legacy);
            }
        }
        {
            let water_time = pass_timestamps(clock.as_ref(), "water");
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water"),
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
                    depth_ops: None,
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.frame_bg, &[]);
            pass.set_bind_group(1, &self.tex_bg, &[]);
            pass.set_bind_group(2, &self.post_bg, &[]);
            if let Some(gpu) = &self.contract {
                pass.set_bind_group(3, &self.paint_bg, &[]);
                pass.set_bind_group(4, &gpu.bind_group, &[]);
            }
            if let Some(legacy) = self.legacy_terrain() {
                pass.set_pipeline(&self.pipe_water);
                draw_indexed(&mut pass, legacy);
            } else {
                pass.set_pipeline(&self.pipe_water_lod);
                self.terrain_geom.draw_view(&mut pass);
            }
        }
        // This frame's SSR becomes next frame's history.
        let ssr_extent = wgpu::Extent3d {
            width: self.ssr.width,
            height: self.ssr.height,
            depth_or_array_layers: 1,
        };
        for (src, dst) in [(&self.ssr.a, &self.ssr.hist_a), (&self.ssr.b, &self.ssr.hist_b)] {
            encoder.copy_texture_to_texture(
                wgpu::ImageCopyTexture {
                    texture: src,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyTexture {
                    texture: dst,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                ssr_extent,
            );
        }
        self.ssr_hist_valid = 1.0;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("after-water"),
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
            if water_ssr {
                // Main's puddle demo draws its sheet here.
                pass.set_bind_group(2, &self.post_bg, &[]);
                if let Some(gpu) = &self.contract {
                    pass.set_bind_group(3, &self.paint_bg, &[]);
                    pass.set_bind_group(4, &gpu.bind_group, &[]);
                }
                if let Some(legacy) = self.legacy_terrain() {
                    pass.set_pipeline(&self.pipe_water);
                    draw_indexed(&mut pass, legacy);
                } else {
                    pass.set_pipeline(&self.pipe_water_lod);
                    self.terrain_geom.draw_view(&mut pass);
                }
            }
            pass.set_bind_group(2, &self.model_bg, &[self.model_stride * 5]);

            // Sheets are thin. Back-face cull hides the side the camera sees.
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

        let air_pass = fog_volume || self.fog > 0.0;
        let mut fog_over_ldr = false;
        if let (true, Some(pipe), Some(up), Some(gpu)) =
            (air_pass, &self.pipe_fog, &self.pipe_fog_up, &self.contract)
        {
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("dust-volume"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.fog_half_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
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
                pass.set_bind_group(2, &self.post_bg, &[]);
                pass.set_bind_group(3, &self.paint_bg, &[]);
                pass.set_bind_group(4, &gpu.bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
            // Display-space composite over the 8-bit picture after the
            // tonemap pass, unless post needs the cloud in the HDR target.
            if self.ldr.is_none() || self.post.is_some() {
                let up_pipe = if self.ldr.is_some() {
                    self.pipe_fog_up_hdr.as_ref().unwrap_or(up)
                } else {
                    up
                };
                self.encode_fog_up(&mut encoder, &self.color_view, up_pipe);
            } else {
                fog_over_ldr = true;
            }
        }

        // The window alone needs no 8-bit copy: the blit tone-maps on the way.
        let blit_tonemaps = self.ldr.is_some()
            && self.post.is_none()
            && !fog_over_ldr
            && !self.debug_contract
            && !readback
            && present.is_some();
        if let (Some((_, ldr_view)), Some(bg)) = (self.ldr.as_ref(), self.tonemap_bg.as_ref()) {
            if !self.debug_contract && !blit_tonemaps {
                let tonemap_time = pass_timestamps(clock.as_ref(), "tonemap");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("tonemap"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: ldr_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: tonemap_time,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&self.pipe_tonemap);
                pass.set_bind_group(0, bg, &[]);
                pass.draw(0..3, 0..1);
                count_draw(3);
            }
        }
        if fog_over_ldr && !self.debug_contract {
            if let (Some((_, ldr_view)), Some(up)) = (self.ldr.as_ref(), self.pipe_fog_up.as_ref()) {
                self.encode_fog_up(&mut encoder, ldr_view, up);
            }
        }

        if self.debug_contract {
            if let (Some(pipe), Some(gpu)) = (&self.pipe_debug, &self.contract) {
                let debug_time = pass_timestamps(clock.as_ref(), "contract-debug");
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("contract-debug"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: self.ldr.as_ref().map_or(&self.color_view, |l| &l.1),
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

        // The debug view writes the final picture itself; post would grade it.
        if let Some(post) = self.post.as_ref().filter(|_| !self.debug_contract) {
            post.encode(&mut encoder, clock.as_ref());
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
            match (blit_tonemaps, self.tonemap_bg.as_ref()) {
                (true, Some(bg)) => {
                    pass.set_pipeline(&self.pipe_blit_hdr[srgb as usize]);
                    pass.set_bind_group(0, bg, &[]);
                }
                _ => {
                    pass.set_pipeline(if srgb {
                        &self.pipe_blit_srgb
                    } else {
                        &self.pipe_blit
                    });
                    pass.set_bind_group(0, &self.blit_bg, &[]);
                }
            }
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
                    texture: match (self.post.as_ref(), self.ldr.as_ref()) {
                        (Some(p), _) => p.output(),
                        (None, Some(l)) => &l.0,
                        (None, None) => &self.color,
                    },
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

    /// Upsample of the half-resolution dust and fog march over `view`.
    fn encode_fog_up(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        pipe: &wgpu::RenderPipeline,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("dust-volume-up"),
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
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, &self.fog_up_bg, &[]);
        pass.draw(0..3, 0..1);
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
                    bytes_per_row: Some(reflect_row(self.width, self.scene_format)),
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
        let stride = reflect_row(self.width, self.scene_format) as usize;
        let mut out = Vec::with_capacity(row * self.height as usize);
        let hdr = self.scene_format != LDR_FORMAT;
        for y in 0..self.height as usize {
            if hdr {
                // Rgba16Float: tone-map to the same 8-bit value the frame shows.
                let line = &padded[y * stride..y * stride + row * 2];
                for (i, px) in line.chunks_exact(2).enumerate() {
                    let v = half_to_f32(u16::from_le_bytes([px[0], px[1]]));
                    out.push(if i % 4 == 3 { 255 } else { tonemap_u8(v) });
                }
            } else {
                out.extend_from_slice(&padded[y * stride..y * stride + row]);
            }
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

/// Mean water surface on a ring just outside each carcass: soil height plus
/// the solved water and slurry. Read only; the sheet is the solver's.
pub fn tire_waterlines(world: &World, fluid: &silt::fluid_core::FluidCore) -> [f32; 4] {
    let v = &world.vehicle;
    let ori = v.orientation();
    let mut out = [-1000.0; 4];
    for (i, o) in out.iter_mut().enumerate() {
        let c = v.pos + ori * v.wheel_center_local(i);
        let mut sum = 0.0;
        let mut n = 0.0;
        for k in 0..8 {
            let a = k as f32 * std::f32::consts::FRAC_PI_4;
            let (x, z) = (c.x + a.cos() * 0.32, c.z + a.sin() * 0.45);
            let g = fluid.sample(x, z);
            let film = g.water_m + g.slurry_m;
            if film > 0.004 {
                sum += world.soil.sample(x, z).height + film;
                n += 1.0;
            }
        }
        if n > 0.0 {
            *o = sum / n;
        }
    }
    out
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
    scene_format: wgpu::TextureFormat,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Buffer,
    u32,
    Option<(wgpu::Texture, wgpu::TextureView)>,
) {
    let ldr = (scene_format != LDR_FORMAT).then(|| {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color-ldr"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LDR_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        (tex, view)
    });
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
        format: scene_format,
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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
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
    (color, color_view, depth, depth_view, readback, padded_row, ldr)
}

fn make_scene_targets(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    depth_format: wgpu::TextureFormat,
    scene_format: wgpu::TextureFormat,
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
        format: scene_format,
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
        format: scene_format,
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
        format: scene_format,
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
        format: scene_format,
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

// Exposed linear HDR scene -> display value. Must match `aces_fit` in
// shader.wgsl (Narkowicz ACES fit; exposure is applied by `present`), so a
// frame looks the same with and without the HDR target.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let m = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return pow(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_tonemap(in: V) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(blit_tex));
    let c = textureLoad(blit_tex, vec2<i32>(in.uv * dims), 0).rgb;
    return vec4<f32>(tonemap(c), 1.0);
}

// Window blit straight from the HDR target (no 8-bit copy in between).
@fragment
fn fs_blit_tonemap(in: V) -> @location(0) vec4<f32> {
    return vec4<f32>(tonemap(textureSample(blit_tex, blit_samp, in.uv).rgb), 1.0);
}

@fragment
fn fs_blit_tonemap_tosrgb(in: V) -> @location(0) vec4<f32> {
    let c = tonemap(textureSample(blit_tex, blit_samp, in.uv).rgb);
    return vec4<f32>(pow(c, vec3<f32>(2.2)), 1.0);
}
"#;

fn blit_hdr_pipes(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
) -> [wgpu::RenderPipeline; 2] {
    ["fs_blit_tonemap", "fs_blit_tonemap_tosrgb"].map(|fs| {
        pipeline(device, shader, layout, "vs_blit", Some(fs), &[], format, None, false, None)
    })
}

/// What the blit and the capture read: post output, else the tone-mapped
/// copy, else the 8-bit scene target itself.
fn final_view<'a>(
    post: Option<&'a crate::post::Post>,
    ldr: Option<&'a (wgpu::Texture, wgpu::TextureView)>,
    color_view: &'a wgpu::TextureView,
) -> &'a wgpu::TextureView {
    post.map(|p| p.output_view())
        .or(ldr.map(|l| &l.1))
        .unwrap_or(color_view)
}

fn tonemap_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    src: &wgpu::TextureView,
    samp: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tonemap-bg"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(src),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(samp),
            },
        ],
    })
}

/// Padded bytes per row for a readback of a scene-format texture.
fn reflect_row(width: u32, scene_format: wgpu::TextureFormat) -> u32 {
    let bpp = scene_format.block_copy_size(None).unwrap_or(4);
    (width * bpp).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT
}

/// IEEE half float to f32 (readback of the HDR reflect copy).
fn half_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = (h & 0x3ff) as f32;
    match exp {
        0 => sign * man * 2f32.powi(-24),
        31 => {
            if man == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Same curve as `fs_tonemap`, on the CPU, to 8 bits.
fn tonemap_u8(c: f32) -> u8 {
    let x = if c.is_finite() { c.max(0.0) } else { 0.0 };
    let m = ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0);
    (m.powf(1.0 / 2.2) * 255.0).round().clamp(0.0, 255.0) as u8
}

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
/// The right lamp's w carries the sand-field flag for the ground shader.
fn headlamps(vehicle: &Vehicle, on: f32, sand: bool) -> [[f32; 4]; 3] {
    let ori = vehicle.orientation();
    let left = vehicle.pos + ori * Vec3::new(-0.56, 0.20, 1.82);
    let right = vehicle.pos + ori * Vec3::new(0.56, 0.20, 1.82);
    // Dipped beam: a few degrees down the road.
    let dir = (ori * Vec3::new(0.0, -0.05, 1.0)).normalize_or_zero();
    [
        [left.x, left.y, left.z, on],
        [right.x, right.y, right.z, if sand { 1.0 } else { 0.0 }],
        [dir.x, dir.y, dir.z, 0.86],
    ]
}

fn make_scene_copy(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    scene_format: wgpu::TextureFormat,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("scene-copy"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: scene_format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    (texture, view)
}

/// Half-resolution target for the dust and fog march, and the bind group the
/// upsample reads it through. RGB is premultiplied by A.
fn make_fog_half(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    width: u32,
    height: u32,
) -> (wgpu::TextureView, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fog-half"),
        size: wgpu::Extent3d {
            width: width.div_ceil(2),
            height: height.div_ceil(2),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let samp = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("fog-half"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fog-up"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 20,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 21,
                resource: wgpu::BindingResource::Sampler(&samp),
            },
        ],
    });
    (view, bg)
}

/// The curl and billow volume the dust and fog march warps its detail with.
fn make_air_noise(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::TextureView, wgpu::Sampler) {
    let n = crate::air_noise::AIR_NOISE;
    let size = wgpu::Extent3d {
        width: n,
        height: n,
        depth_or_array_layers: n,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("air-noise"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba8Snorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texels = crate::air_noise::curl_volume();
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(n * 4),
            rows_per_image: Some(n),
        },
        size,
    );
    let samp = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("air-noise"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    (tex.create_view(&wgpu::TextureViewDescriptor::default()), samp)
}

fn make_post_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    depth_view: &wgpu::TextureView,
    scene_view: &wgpu::TextureView,
    air_noise: &wgpu::TextureView,
    air_samp: &wgpu::Sampler,
    ssr_a: &wgpu::TextureView,
    ssr_b: &wgpu::TextureView,
    planar: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("post"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(scene_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(air_noise),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::Sampler(air_samp),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(ssr_a),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(ssr_b),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(planar),
            },
        ],
    })
}

/// Target for the planar reflection: the Niva drawn mirrored about the
/// water level, same format as the frame so the mesh pipelines draw into it.
struct Planar {
    view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

fn make_planar(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    depth_format: wgpu::TextureFormat,
    scene_format: wgpu::TextureFormat,
) -> Planar {
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let color = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planar"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: scene_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let depth = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("planar-depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: depth_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    Planar {
        view: color.create_view(&Default::default()),
        depth_view: depth.create_view(&Default::default()),
    }
}

/// Half-resolution screen-space reflection of the water: the traced
/// result (a, b) and last frame's (history a, b), plus a depth buffer so
/// the nearest water surface wins.
struct SsrTargets {
    width: u32,
    height: u32,
    a: wgpu::Texture,
    a_view: wgpu::TextureView,
    b: wgpu::Texture,
    b_view: wgpu::TextureView,
    hist_a: wgpu::Texture,
    hist_a_view: wgpu::TextureView,
    hist_b: wgpu::Texture,
    hist_b_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
}

const SSR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn make_ssr(device: &wgpu::Device, width: u32, height: u32) -> SsrTargets {
    let w = (width / 2).max(1);
    let h = (height / 2).max(1);
    let tex = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        let v = t.create_view(&Default::default());
        (t, v)
    };
    let rt = wgpu::TextureUsages::RENDER_ATTACHMENT
        | wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::COPY_SRC;
    let hist = wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST;
    let (a, a_view) = tex("ssr-a", SSR_FORMAT, rt);
    let (b, b_view) = tex("ssr-b", SSR_FORMAT, rt);
    let (hist_a, hist_a_view) = tex("ssr-hist-a", SSR_FORMAT, hist);
    let (hist_b, hist_b_view) = tex("ssr-hist-b", SSR_FORMAT, hist);
    let (_depth, depth_view) = tex("ssr-depth", wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::RENDER_ATTACHMENT);
    SsrTargets { width: w, height: h, a, a_view, b, b_view, hist_a, hist_a_view, hist_b, hist_b_view, depth_view }
}
