//! GPU side of the ground geometry: the LOD patch, its per-view instances,
//! the geometry uniform, and the fine track texture.
//!
//! `render.rs` owns the passes. This module owns what the ground vertex
//! shaders in `terrain_geom.wgsl` read (group 1, bindings 10 and 11) and
//! the instance lists those shaders are drawn with.

use glam::{Mat4, Vec3};
use silt::soil::{SoilField, MAP_RES, MAP_SIZE};
use silt::terrain_lod::{self, LodNode, LodParams};

/// Fine track texels the texture holds. The solver's proposed window is
/// 640 × 640 at 2.5 cm (16 m around the Niva).
pub const TRACK_FINE_RES: u32 = 640;
/// Default shadow ring: 1.56 cm to 1.5 m, 25 cm past 24 m.
pub const SHADOW_RANGE_STEPS: f32 = 96.0;
/// Most nodes one view can draw. Two views share the buffer.
const MAX_NODES: usize = 4096;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TerrainGeomData {
    lod: [f32; 4],
    ext: [f32; 4],
    recon: [f32; 4],
    fine: [f32; 4],
}

/// Fine track window as the solver publishes it.
#[derive(Clone, Copy, Debug)]
pub struct TrackFineWindow {
    /// Minimum corner, world X and Z.
    pub origin: [f32; 2],
    /// Texel size, metres.
    pub cell: f32,
    /// Texels per side. At most [`TRACK_FINE_RES`].
    pub res: u32,
}

pub struct TerrainGeometry {
    patch_v: wgpu::Buffer,
    patch_i: wgpu::Buffer,
    patch_count: u32,
    instances: wgpu::Buffer,
    uniform: wgpu::Buffer,
    fine_tex: wgpu::Texture,
    fine_view: wgpu::TextureView,
    fine_float: bool,
    fine_window: Option<TrackFineWindow>,
    main_nodes: u32,
    shadow_nodes: u32,
    /// Draw the LOD tree. Off draws the old uniform grid (`SILT_TERRAIN_LOD=0`).
    pub lod: bool,
    /// Monotone cubic between solver samples. Off is bilinear (`SILT_TERRAIN_CUBIC=0`).
    pub cubic: bool,
    /// Tint by level and draw the triangle grid (`SILT_TERRAIN_DEBUG=1`).
    pub debug: bool,
    /// Ring radius in vertex steps (`SILT_TERRAIN_LOD_RANGE`).
    pub range_steps: f32,
    /// Ring radius for the shadow pass (`SILT_TERRAIN_SHADOW_RANGE`). The
    /// shadow map texel is 1.4 cm at best, so its ground can be coarser.
    pub shadow_range_steps: f32,
}

fn env_flag(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => !matches!(v.as_str(), "0" | "false" | "off" | ""),
        Err(_) => default,
    }
}

impl TerrainGeometry {
    pub fn new(device: &wgpu::Device, float_textures: bool) -> Self {
        use wgpu::util::DeviceExt;
        let (verts, idx) = terrain_lod::patch_mesh();
        let patch_v = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-lod-patch"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let patch_i = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("terrain-lod-patch"),
            contents: bytemuck::cast_slice(&idx),
            usage: wgpu::BufferUsages::INDEX,
        });
        let instances = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-lod-nodes"),
            size: (std::mem::size_of::<LodNode>() * MAX_NODES * 2) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("terrain-geom"),
            size: std::mem::size_of::<TerrainGeomData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The packed (WebGL) path has no float maps. It keeps a 1 × 1
        // placeholder and never turns the fine track on.
        let side = if float_textures { TRACK_FINE_RES } else { 1 };
        let fine_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("track-fine"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if float_textures {
                wgpu::TextureFormat::Rgba32Float
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let fine_view = fine_tex.create_view(&Default::default());
        let range_steps = std::env::var("SILT_TERRAIN_LOD_RANGE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v >= 64.0)
            .unwrap_or(terrain_lod::RANGE_STEPS);
        let shadow_range_steps = std::env::var("SILT_TERRAIN_SHADOW_RANGE")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|v| *v >= 32.0)
            .unwrap_or(SHADOW_RANGE_STEPS);
        Self {
            patch_v,
            patch_i,
            patch_count: idx.len() as u32,
            instances,
            uniform,
            fine_tex,
            fine_view,
            fine_float: float_textures,
            fine_window: None,
            main_nodes: 0,
            shadow_nodes: 0,
            lod: env_flag("SILT_TERRAIN_LOD", true),
            cubic: env_flag("SILT_TERRAIN_CUBIC", true),
            debug: env_flag("SILT_TERRAIN_DEBUG", false),
            range_steps,
            shadow_range_steps,
        }
    }

    /// Layout entries for group 1. The ground and the shadow layouts both
    /// carry them, since both vertex paths read the height.
    pub fn layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
        [
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<TerrainGeomData>() as u64,
                    ),
                },
                count: None,
            },
        ]
    }

    pub fn bind_entries(&self) -> [wgpu::BindGroupEntry<'_>; 2] {
        [
            wgpu::BindGroupEntry {
                binding: 10,
                resource: wgpu::BindingResource::TextureView(&self.fine_view),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: self.uniform.as_entire_binding(),
            },
        ]
    }

    pub fn buffers() -> &'static [wgpu::VertexBufferLayout<'static>] {
        &[
            wgpu::VertexBufferLayout {
                array_stride: 8,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 0,
                    shader_location: 0,
                }],
            },
            wgpu::VertexBufferLayout {
                array_stride: 16,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &[wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 1,
                }],
            },
        ]
    }

    /// Publish the solver's fine track field. `texels` is rgba per texel,
    /// row-major with i along +X and j along +Z, `res * res * 4` floats.
    /// R is the surface offset over contract ground (bed + placed, binding 1
    /// R) at the texel centre. `ground(x, z)` must return that contract
    /// ground, bilinear. The texture stores the absolute surface (ground + R),
    /// so the drawn height does not depend on what the 20 cm height map holds
    /// (the snow capture keeps the snow top there).
    /// Ignored on the packed path, which has no float textures.
    pub fn set_track_fine(
        &mut self,
        queue: &wgpu::Queue,
        window: TrackFineWindow,
        texels: &[f32],
        ground: impl Fn(f32, f32) -> f32,
    ) {
        if !self.fine_float || window.res == 0 || window.res > TRACK_FINE_RES {
            return;
        }
        let n = (window.res * window.res * 4) as usize;
        if texels.len() < n {
            return;
        }
        let res = window.res as usize;
        let mut clean: Vec<f32> = texels[..n]
            .iter()
            .map(|v| if v.is_finite() { *v } else { 0.0 })
            .collect();
        for j in 0..res {
            let z = window.origin[1] + (j as f32 + 0.5) * window.cell;
            for i in 0..res {
                let x = window.origin[0] + (i as f32 + 0.5) * window.cell;
                let g = ground(x, z);
                let k = (j * res + i) * 4;
                clean[k] += if g.is_finite() { g } else { 0.0 };
            }
        }
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.fine_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&clean),
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(window.res * 16),
                rows_per_image: Some(window.res),
            },
            wgpu::Extent3d {
                width: window.res,
                height: window.res,
                depth_or_array_layers: 1,
            },
        );
        self.fine_window = Some(window);
    }

    /// The solver did not publish a fine field this frame.
    pub fn clear_track_fine(&mut self) {
        self.fine_window = None;
    }

    /// Drawn ground square: the cell centres of the soil map, as the old
    /// grid had them.
    fn extent() -> (f32, f32) {
        let cell = MAP_SIZE / MAP_RES as f32;
        let half = MAP_SIZE * 0.5;
        (-half + cell * 0.5, half - cell * 0.5)
    }

    fn params(&self, soil: &SoilField) -> LodParams {
        let (lo, hi) = Self::extent();
        let mut y_min = f32::MAX;
        let mut y_max = f32::MIN;
        for h in &soil.height {
            y_min = y_min.min(*h);
            y_max = y_max.max(*h);
        }
        if !(y_min.is_finite() && y_max.is_finite()) || y_min > y_max {
            y_min = -1.0;
            y_max = 3.0;
        }
        LodParams {
            range_steps: self.range_steps,
            y_ref: 0.0,
            extent_min: [lo, lo],
            extent_max: [hi, hi],
            // Seats, lugs, and the fine track move the drawn ground a little
            // past the stored height.
            y_min: y_min - 0.5,
            y_max: y_max + 0.5,
        }
    }

    /// Pick this frame's patches and write the uniform. Call before the
    /// passes are encoded.
    /// `light_vp` culls the shadow list: a patch outside the light's box
    /// never lands in the shadow map.
    pub fn prepare(
        &mut self,
        queue: &wgpu::Queue,
        soil: &SoilField,
        eye: Vec3,
        view_proj: &Mat4,
        light_vp: &Mat4,
    ) {
        let params = self.params(soil);
        let (lo, hi) = Self::extent();
        let fine = self.fine_window;
        let data = TerrainGeomData {
            lod: [
                terrain_lod::FINEST_STEP,
                params.range_steps,
                params.y_ref,
                terrain_lod::MORPH_START,
            ],
            ext: [
                terrain_lod::ROOT_ORIGIN,
                lo,
                hi,
                if self.debug { 1.0 } else { 0.0 },
            ],
            recon: [
                if self.cubic { 1.0 } else { 0.0 },
                if fine.is_some() { 1.0 } else { 0.0 },
                0.25,
                self.shadow_range_steps,
            ],
            fine: match fine {
                Some(w) => [w.origin[0], w.origin[1], w.cell, w.res as f32],
                None => [0.0, 0.0, 1.0, 1.0],
            },
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&data));
        if !self.lod {
            self.main_nodes = 0;
            self.shadow_nodes = 0;
            return;
        }
        let mut main = terrain_lod::select(eye, &params, Some(view_proj));
        let shadow_params = LodParams {
            range_steps: self.shadow_range_steps,
            ..params
        };
        let mut shadow = terrain_lod::select(eye, &shadow_params, Some(light_vp));
        main.truncate(MAX_NODES);
        shadow.truncate(MAX_NODES);
        queue.write_buffer(&self.instances, 0, bytemuck::cast_slice(&main));
        queue.write_buffer(
            &self.instances,
            (std::mem::size_of::<LodNode>() * MAX_NODES) as u64,
            bytemuck::cast_slice(&shadow),
        );
        self.main_nodes = main.len() as u32;
        self.shadow_nodes = shadow.len() as u32;
    }

    fn draw_range(&self, pass: &mut wgpu::RenderPass<'_>, first: usize, count: u32) {
        if count == 0 {
            return;
        }
        let stride = std::mem::size_of::<LodNode>();
        let start = (first * stride) as u64;
        let end = start + count as u64 * stride as u64;
        pass.set_vertex_buffer(0, self.patch_v.slice(..));
        pass.set_vertex_buffer(1, self.instances.slice(start..end));
        pass.set_index_buffer(self.patch_i.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..self.patch_count, 0, 0..count);
    }

    /// The patches in the camera's view.
    pub fn draw_view(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_range(pass, 0, self.main_nodes);
    }

    /// The patches inside the light's box, for the shadow map.
    pub fn draw_all(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_range(pass, MAX_NODES, self.shadow_nodes);
    }

    /// Triangles the shadow pass draws, for the log.
    pub fn shadow_triangles(&self) -> u64 {
        self.shadow_nodes as u64 * self.patch_count as u64 / 3
    }

    /// Triangles one view draws, for the log.
    pub fn view_triangles(&self) -> u64 {
        self.main_nodes as u64 * self.patch_count as u64 / 3
    }
}
