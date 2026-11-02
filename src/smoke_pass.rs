//! Smoke and fire in the picture: steps `silt::smoke::GpuSmoke` and hands its
//! textures to the dust volume march (`fs_fog` in `beauty_contract.wgsl`,
//! group 5, pass "dust-volume"), so smoke, dust and mist share one march.
//!
//! Off unless a scene asks for smoke. Needs compute, so never on WebGL2.

use glam::Vec3;
use silt::smoke::{blackbody, GpuSmoke, SmokeConfig, AMBIENT_K};

const LUT: usize = 24;
/// Kelvin over ambient per LUT entry: 0..1610 K over 293 K ambient.
const LUT_STEP_K: f32 = 70.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SmokeDraw {
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
    /// Sub-grid curl-noise detail gain, 0 off.
    pub detail: f32,
    /// Base noise frequency, cycles per grid cell.
    pub detail_scale: f32,
    /// Unresolved eddy speed (|curl v| * h) that gets the full gain, m/s.
    pub detail_ref_m_s: f32,
    /// Light a fire throws on its smoke per unit of fuel fed per second.
    pub fire_light: f32,
}

impl Default for SmokeLook {
    fn default() -> Self {
        Self {
            albedo: Vec3::new(0.82, 0.82, 0.84),
            phase_g: 0.35,
            emission: 3.0,
            ambient: 0.6,
            detail: 0.3,
            detail_scale: 0.22,
            detail_ref_m_s: 0.25,
            fire_light: 0.5,
        }
    }
}

pub struct SmokePass {
    pub sim: GpuSmoke,
    pub look: SmokeLook,
    uniform: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl SmokePass {
    pub fn new(device: &wgpu::Device, config: SmokeConfig) -> Self {
        let sim = GpuSmoke::new(device, config);
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
            detail: [fires as f32, l.detail, l.detail_scale, l.detail_ref_m_s],
            origin: [c.origin.x, c.origin.y, c.origin.z, c.cell_m],
            size: [size.x, size.y, size.z, c.extinction_m],
            albedo: [l.albedo.x, l.albedo.y, l.albedo.z, l.phase_g],
            light: [l.emission, l.ambient, LUT_STEP_K, LUT as f32],
            lut,
        }
    }

    /// The group-5 bind group of the dust march for this frame's smoke:
    /// draw constants, soot/temperature/burn/sun-transmittance volume, the
    /// filtering sampler, and the flow volume that drives sub-grid detail.
    pub fn volume_group(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
    ) -> wgpu::BindGroup {
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.draw_uniform()));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("smoke-volume"),
            layout,
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
                    resource: wgpu::BindingResource::TextureView(&self.sim.flow_view),
                },
            ],
        })
    }
}

// The dust march declares `SmokeDraw` as 37 vec4 (render.rs SMOKE_DRAW_BYTES).
const _: () = assert!(std::mem::size_of::<SmokeDraw>() == 37 * 16);
