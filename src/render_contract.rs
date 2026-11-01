//! Physics → render contract. The version is [`RENDER_CONTRACT_VERSION`].
//!
//! The solver publishes these maps at the end of a frame. The debug view
//! reads only this contract. Picture shading (materials, light, HDRI, fog,
//! headlights, wet sheen) is not part of it.
//!
//! Change this contract only by bumping [`RENDER_CONTRACT_VERSION`] and
//! recording the change in `docs/RENDER_CONTRACT.md`.

use bytemuck::Zeroable;

use crate::dust::{DensityTexture, Dust, AIR_DY_M, SETTLED_CELL_M};
use crate::fluid_core::{FluidCore, AIR_LAYERS, AIR_RES, COARSE_CELL_M, WINDOW_RES};
use crate::particles::Particle;
use crate::snow::{SnowField, RHO_FRESH, SNOW_EMPTY_M};
use crate::soil::{SoilField, SurfaceKind};

/// Contract version. Bump it when a map, channel, unit, or formula below changes.
pub const RENDER_CONTRACT_VERSION: u32 = 8;

/// Floats per flow texel: vx, vz, speed, turbidity.
pub const FLOW_CHANNELS: usize = 4;

/// Coarse ground window. 200 × 0.20 m = 40 m. The 2 cm tire patch is restricted
/// back into this window before publish. It is not a second map.
pub const WINDOW_CELLS: u32 = WINDOW_RES as u32;
pub const WINDOW_CELL_M: f32 = COARSE_CELL_M;
pub const WINDOW_EXTENT_M: f32 = WINDOW_RES as f32 * COARSE_CELL_M;

pub const AIR_CELLS: u32 = AIR_RES as u32;
pub const AIR_SLABS: u32 = AIR_LAYERS as u32;

/// Live spray slots in the storage buffer.
pub const SPRAY_CAPACITY: u32 = 4096;

/// Pore moisture at or above this, or a film at or above [`WET_FILM_M`], is wet.
pub const WET_MOISTURE: f32 = 0.45;
/// Standing water plus slurry, metres, that counts as wet.
pub const WET_FILM_M: f32 = 0.004;
/// Film thinner than this does not make foam.
pub const FOAM_MIN_FILM_M: f32 = 0.004;
/// Froude number where foam starts. `speed / sqrt(g * thickness)`.
pub const FOAM_FROUDE_0: f32 = 0.85;
/// Froude number where foam is 1.
pub const FOAM_FROUDE_1: f32 = 1.45;

const GRAVITY: f32 = 9.81;

/// Picture cards spawned from the dust grid. Not spray mass.
const KIND_DUST_MARK: u8 = 8;

/// Soil kind id for a dry crust. It is not the crack channel. Opening is metres in `crust.g`.
pub const SOIL_KIND_DRY_CRUST: u8 = SurfaceKind::DryCrust as u8;

pub const FLOW_NONE: u32 = 0;
pub const FLOW_FLUID_SHEET: u32 = 1;

pub const DUST_NONE: u32 = 0;
pub const DUST_LEDGER: u32 = 1;
pub const DUST_FLUID_AIR: u32 = 2;

pub const VALID_GROUND: u32 = 1 << 0;
pub const VALID_MOISTURE: u32 = 1 << 1;
pub const VALID_FILM: u32 = 1 << 2;
pub const VALID_FLOW: u32 = 1 << 3;
pub const VALID_DUST_SETTLED: u32 = 1 << 4;
pub const VALID_DUST_FOG: u32 = 1 << 5;
pub const VALID_CRUST: u32 = 1 << 6;
pub const VALID_WET_EDGE: u32 = 1 << 7;
pub const VALID_FOAM: u32 = 1 << 8;
pub const VALID_SPRAY: u32 = 1 << 9;
pub const VALID_AIR: u32 = 1 << 10;
pub const VALID_TURBIDITY: u32 = 1 << 11;
pub const VALID_MIX: u32 = 1 << 12;
pub const VALID_SNOW: u32 = 1 << 14;
pub const VALID_FOG: u32 = 1 << 13;

pub const BIND_HEADER: u32 = 0;
pub const BIND_GROUND: u32 = 1;
pub const BIND_MOISTURE: u32 = 2;
pub const BIND_FILM: u32 = 3;
pub const BIND_FLOW: u32 = 4;
pub const BIND_DUST_SETTLED: u32 = 5;
pub const BIND_DUST_FOG: u32 = 6;
pub const BIND_CRUST: u32 = 7;
pub const BIND_WET_EDGE: u32 = 8;
pub const BIND_FOAM: u32 = 9;
pub const BIND_SPRAY: u32 = 10;
pub const BIND_AIR: u32 = 11;
pub const BIND_FOG: u32 = 12;
pub const BIND_MIX: u32 = 13;
pub const BIND_SNOW: u32 = 14;

/// Uniform header. 96 bytes. Layout matches `ContractHeader` in the WGSL.
///
/// `debug_mode` is a viewer selector written by the renderer. It is not a
/// solver map. Adding or retuning a solver map still bumps the version.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ContractHeader {
    pub version: u32,
    pub window_res: u32,
    pub air_res: u32,
    pub air_layers: u32,
    pub cell_m: f32,
    pub extent_m: f32,
    pub origin_x: f32,
    pub origin_z: f32,
    pub air_origin_x: f32,
    pub air_origin_y: f32,
    pub air_origin_z: f32,
    pub air_cell_x: f32,
    pub air_cell_y: f32,
    pub air_cell_z: f32,
    pub spray_count: u32,
    pub spray_capacity: u32,
    pub frame: u32,
    pub valid: u32,
    pub time_s: f32,
    pub center_x: f32,
    pub center_z: f32,
    pub flow_source: u32,
    pub dust_source: u32,
    pub debug_mode: u32,
}

/// One spray slot. 48 bytes. `pos_mass.w` is kilograms. `vel_age.w` is age in seconds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SprayGpu {
    pub pos_mass: [f32; 4],
    pub vel_age: [f32; 4],
    pub material: u32,
    pub alive: u32,
    pub _pad: [u32; 2],
}

/// CPU copy of every map the solver publishes for one frame.
pub struct ContractFrame {
    pub header: ContractHeader,
    pub ground_rut: Vec<f32>,
    pub moisture: Vec<f32>,
    pub film: Vec<f32>,
    pub flow: Vec<f32>,
    pub dust_settled: Vec<f32>,
    pub crust: Vec<f32>,
    pub wet_edge: Vec<f32>,
    pub foam_crest: Vec<f32>,
    pub dust_fog: Vec<f32>,
    /// World air velocity, rgba, same index as `dust_fog` times 4.
    pub air_velocity: Vec<f32>,
    pub fog: Vec<f32>,
    /// Mud, sand, gravel, grass shares. Four floats per texel.
    pub soil_mix: Vec<f32>,
    /// Snow column. RGBA: height m, density kg/m³, compaction 0..1, 0.
    pub snow: Vec<f32>,
    pub spray: Vec<SprayGpu>,
}

impl ContractFrame {
    pub fn new() -> Self {
        let n = WINDOW_RES * WINDOW_RES;
        let air = AIR_RES * AIR_RES * AIR_LAYERS;
        Self {
            header: ContractHeader::zeroed(),
            ground_rut: vec![0.0; n * 4],
            moisture: vec![0.0; n],
            film: vec![0.0; n * 2],
            flow: vec![0.0; n * FLOW_CHANNELS],
            dust_settled: vec![0.0; n],
            crust: vec![0.0; n * 2],
            wet_edge: vec![0.0; n],
            foam_crest: vec![0.0; n * 2],
            dust_fog: vec![0.0; air],
            air_velocity: vec![0.0; air * 4],
            fog: vec![0.0; air],
            soil_mix: vec![0.0; n * 4],
            snow: vec![0.0; n * 4],
            spray: vec![SprayGpu::zeroed(); SPRAY_CAPACITY as usize],
        }
    }
}

impl Default for ContractFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// Kilograms per cubic metre for a spray material id.
/// Water droplets are 1000. Wet clods are 1600. Slurry pieces are 1450.
/// An unknown id, including a dust picture mark, is 0.
pub fn spray_density_kg_m3(kind: u8) -> f32 {
    match kind {
        0 | 4 | 9 => 1000.0,
        1 | 2 | 3 => 1600.0,
        5 | 6 | 7 => 1450.0,
        _ => 0.0,
    }
}

/// Foam fraction, 0 to 1, from the Froude number of the film.
/// Zero when the film is thinner than [`FOAM_MIN_FILM_M`], the speed is 0,
/// or either input is not finite.
pub fn foam_fraction(speed_m_s: f32, thickness_m: f32) -> f32 {
    if !speed_m_s.is_finite()
        || !thickness_m.is_finite()
        || thickness_m < FOAM_MIN_FILM_M
        || speed_m_s <= 0.0
    {
        return 0.0;
    }
    let froude = speed_m_s / (GRAVITY * thickness_m).sqrt();
    finite0(smooth01(FOAM_FROUDE_0, FOAM_FROUDE_1, froude))
}

fn finite0(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn scrub(values: &mut [f32]) {
    for v in values {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
}

fn smooth01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn is_wet(moisture: f32, water_m: f32, slurry_m: f32) -> bool {
    moisture >= WET_MOISTURE || water_m + slurry_m >= WET_FILM_M
}

/// Copy the snow cell that contains each texel centre.
///
/// The untouched blanket is one constant, so only a column the solver has
/// changed is read. Height is the geometric thickness. Density is that
/// column's bulk density. Compaction is [`SnowField::compaction`]. A column
/// thinner than [`SNOW_EMPTY_M`] is left at 0, density included. Off the snow
/// square the texel stays 0. A is 0.
fn write_snow_binding(dst: &mut [f32], snow: &SnowField, origin_x: f32, origin_z: f32) {
    let n = WINDOW_RES;
    let half = snow.size * 0.5;
    let blanket = snow.blanket_m();
    for j in 0..n {
        let z = origin_z + (j as f32 + 0.5) * WINDOW_CELL_M;
        if z < -half || z >= half {
            continue;
        }
        let row = j * n;
        for i in 0..n {
            let x = origin_x + (i as f32 + 0.5) * WINDOW_CELL_M;
            if x < -half || x >= half {
                continue;
            }
            let o = (row + i) * 4;
            dst[o] = blanket;
            dst[o + 1] = RHO_FRESH;
            dst[o + 2] = 0.0;
            dst[o + 3] = 0.0;
        }
    }
    let margin = snow.cell + WINDOW_CELL_M;
    for src in 0..snow.depth.len() {
        if !snow.is_active(src) {
            continue;
        }
        let (x0, x1, z0, z1) = snow.cell_bounds(src);
        let i0 = ((x0 - margin - origin_x) / WINDOW_CELL_M).floor() as i32;
        let i1 = ((x1 + margin - origin_x) / WINDOW_CELL_M).ceil() as i32;
        let j0 = ((z0 - margin - origin_z) / WINDOW_CELL_M).floor() as i32;
        let j1 = ((z1 + margin - origin_z) / WINDOW_CELL_M).ceil() as i32;
        let height = snow.depth[src];
        let sample = if !height.is_finite() || height < SNOW_EMPTY_M {
            [0.0_f32; 4]
        } else {
            [
                height,
                finite0(snow.density[src]),
                finite0(snow.compaction(src)),
                0.0,
            ]
        };
        let j_lo = j0.max(0) as usize;
        let j_hi = j1.min(n as i32 - 1).max(0) as usize;
        let i_lo = i0.max(0) as usize;
        let i_hi = i1.min(n as i32 - 1).max(0) as usize;
        if j_lo > j_hi || i_lo > i_hi {
            continue;
        }
        for j in j_lo..=j_hi {
            let z = origin_z + (j as f32 + 0.5) * WINDOW_CELL_M;
            for i in i_lo..=i_hi {
                let x = origin_x + (i as f32 + 0.5) * WINDOW_CELL_M;
                if snow.idx_at(x, z) != Some(src) {
                    continue;
                }
                let o = (j * n + i) * 4;
                dst[o..o + 4].copy_from_slice(&sample);
            }
        }
    }
}

/// Fill `out` from the soil heightfield, the live particles, and, when the
/// through-drive sheet is attached, the fluid window and the dust ledger.
///
/// Does not step the solver and does not write back into it. Snow stays empty:
/// pass the field to [`publish_with_snow`] to fill binding 14.
pub fn publish(
    out: &mut ContractFrame,
    soil: &SoilField,
    particles: &[Particle],
    center: [f32; 2],
    time_s: f32,
    debug_mode: u32,
    fluid: Option<&FluidCore>,
    dust: Option<&Dust>,
) {
    publish_with_snow(
        out, soil, particles, center, time_s, debug_mode, fluid, dust, None,
    );
}

/// [`publish`], plus the snow column when `snow` is attached.
///
/// Does not step the solver and does not write back into it.
pub fn publish_with_snow(
    out: &mut ContractFrame,
    soil: &SoilField,
    particles: &[Particle],
    center: [f32; 2],
    time_s: f32,
    debug_mode: u32,
    fluid: Option<&FluidCore>,
    dust: Option<&Dust>,
    snow: Option<&SnowField>,
) {
    out.ground_rut.fill(0.0);
    out.moisture.fill(0.0);
    out.film.fill(0.0);
    out.flow.fill(0.0);
    out.dust_settled.fill(0.0);
    out.crust.fill(0.0);
    out.wet_edge.fill(0.0);
    out.foam_crest.fill(0.0);
    out.dust_fog.fill(0.0);
    out.air_velocity.fill(0.0);
    out.fog.fill(0.0);
    out.soil_mix.fill(0.0);
    out.snow.fill(0.0);
    for slot in &mut out.spray {
        *slot = SprayGpu::zeroed();
    }

    let (origin_x, origin_z) = if let Some(fluid) = fluid {
        let (x0, z0, _) = fluid.window_bounds();
        (x0, z0)
    } else {
        let origin = FluidCore::window_origin(center);
        (origin[0], origin[1])
    };

    let mut wet = vec![false; WINDOW_RES * WINDOW_RES];
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let idx = j * WINDOW_RES + i;
            let x = origin_x + (i as f32 + 0.5) * COARSE_CELL_M;
            let z = origin_z + (j as f32 + 0.5) * COARSE_CELL_M;
            let sheet = fluid.and_then(|fluid| fluid.coarse_cell(i, j));
            let soil_idx = soil.idx_at(x, z);

            let (bed, placed, rut) = if let Some(sidx) = soil_idx {
                (
                    soil.bed[sidx],
                    soil.placed[sidx],
                    (soil.initial_height[sidx] - soil.bed[sidx]).max(0.0),
                )
            } else if let Some(cell) = sheet {
                (cell.bed_m, 0.0, 0.0)
            } else {
                (0.0, 0.0, 0.0)
            };
            // Thickness and opening are the soil arrays, in metres. They are
            // not added to ground. A dry-crust kind is not a stand-in, and the
            // sheet has no arrays of its own, so a texel off the soil stays 0.
            let (thickness, opening) = if let Some(sidx) = soil_idx {
                (
                    finite0(soil.crust[sidx].max(0.0)),
                    finite0(soil.crack[sidx].max(0.0)),
                )
            } else {
                (0.0, 0.0)
            };
            out.crust[idx * 2] = thickness;
            out.crust[idx * 2 + 1] = opening;
            out.ground_rut[idx * 4] = finite0(bed + placed);
            out.ground_rut[idx * 4 + 1] = finite0(rut);
            out.ground_rut[idx * 4 + 2] = finite0(bed);
            out.ground_rut[idx * 4 + 3] = finite0(placed);

            let (sheet_moisture, water_m, slurry_m, u, v, settled, turbidity) =
                if let Some(cell) = sheet {
                    (
                        cell.moisture,
                        cell.water_m,
                        cell.slurry_m,
                        cell.u,
                        cell.v,
                        cell.settled_dust_m,
                        cell.turbidity,
                    )
                } else if let Some(sidx) = soil_idx {
                    (
                        soil.wetness[sidx],
                        soil.water[sidx],
                        soil.slurry[sidx],
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                } else {
                    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                };
            // Wet tire track: soil wetness under this texel is part of the same
            // moisture channel. The drained film stays in the water channel.
            // Neither fill adds a map, a unit, or a resolution.
            let moisture = if let Some(sidx) = soil_idx {
                sheet_moisture.max(soil.wetness[sidx])
            } else {
                sheet_moisture
            };
            let dry = !water_m.is_finite() || !slurry_m.is_finite() || water_m + slurry_m < 1.0e-5;
            let (u, v) = if dry {
                (0.0, 0.0)
            } else {
                (finite0(u), finite0(v))
            };
            let speed = if dry {
                0.0
            } else {
                finite0((u * u + v * v).sqrt())
            };
            // Turbidity is solids in the water column. No water, no sample: 0, not NaN.
            let turb = if !water_m.is_finite() || water_m < 1.0e-5 {
                0.0
            } else {
                finite0(turbidity).clamp(0.0, 1.0)
            };
            out.moisture[idx] = finite0(moisture);
            out.film[idx * 2] = finite0(water_m);
            out.film[idx * 2 + 1] = finite0(slurry_m);
            let mix = soil_idx
                .map(|sidx| soil.mix_texel(sidx))
                .unwrap_or([0.0; 4]);
            out.soil_mix[idx * 4] = finite0(mix[0]);
            out.soil_mix[idx * 4 + 1] = finite0(mix[1]);
            out.soil_mix[idx * 4 + 2] = finite0(mix[2]);
            out.soil_mix[idx * 4 + 3] = finite0(mix[3]);
            let flow = idx * FLOW_CHANNELS;
            out.flow[flow] = u;
            out.flow[flow + 1] = v;
            out.flow[flow + 2] = speed;
            out.flow[flow + 3] = turb;
            if dust.is_none() {
                out.dust_settled[idx] = settled;
            }
            wet[idx] = is_wet(moisture, water_m, slurry_m);
        }
    }

    if let Some(dust) = dust {
        let area = SETTLED_CELL_M * SETTLED_CELL_M;
        for (x, z, volume) in dust.settled_marks() {
            if volume <= 0.0 || area <= 0.0 {
                continue;
            }
            let depth = volume / area;
            let half = SETTLED_CELL_M * 0.5;
            let i0 = ((x - half - origin_x) / COARSE_CELL_M).floor() as i32;
            let i1 = ((x + half - origin_x) / COARSE_CELL_M).floor() as i32;
            let j0 = ((z - half - origin_z) / COARSE_CELL_M).floor() as i32;
            let j1 = ((z + half - origin_z) / COARSE_CELL_M).floor() as i32;
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if i < 0 || j < 0 || i >= WINDOW_RES as i32 || j >= WINDOW_RES as i32 {
                        continue;
                    }
                    let idx = j as usize * WINDOW_RES + i as usize;
                    out.dust_settled[idx] += depth;
                }
            }
        }
    }

    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let idx = j * WINDOW_RES + i;
            let mut edge = 0.0;
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= WINDOW_RES as i32 || nj >= WINDOW_RES as i32 {
                    continue;
                }
                let nidx = nj as usize * WINDOW_RES + ni as usize;
                if wet[idx] != wet[nidx] {
                    edge = 1.0;
                    break;
                }
            }
            out.wet_edge[idx] = edge;
            let water_m = out.film[idx * 2];
            let slurry_m = out.film[idx * 2 + 1];
            let h = water_m + slurry_m;
            let mut sum = 0.0;
            let mut count = 0.0;
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= WINDOW_RES as i32 || nj >= WINDOW_RES as i32 {
                    continue;
                }
                let nidx = nj as usize * WINDOW_RES + ni as usize;
                sum += out.film[nidx * 2] + out.film[nidx * 2 + 1];
                count += 1.0;
            }
            let mean = if count > 0.0 { sum / count } else { h };
            let excess = finite0((h - mean).max(0.0));
            let placed = finite0(out.ground_rut[idx * 4 + 3].max(0.0));
            let foam = match fluid {
                Some(fluid) if h >= FOAM_MIN_FILM_M => finite0(fluid.foam_grid()[idx]).clamp(0.0, 1.0),
                _ => 0.0,
            };
            out.foam_crest[idx * 2] = foam;
            out.foam_crest[idx * 2 + 1] = finite0(placed + excess);
        }
    }

    let (dust_source, air_origin, air_cell) = if let Some(dust) = dust {
        let mut tex = DensityTexture::empty();
        dust.publish(&mut tex);
        if tex.samples.len() == out.dust_fog.len() {
            out.dust_fog.copy_from_slice(&tex.samples);
        }
        (
            DUST_LEDGER,
            [tex.origin.x, tex.origin.y, tex.origin.z],
            [tex.cell.x, tex.cell.y, tex.cell.z],
        )
    } else if let Some(fluid) = fluid {
        let (origin, cell) = fluid.air_window();
        for iz in 0..AIR_RES {
            for layer in 0..AIR_LAYERS {
                for ix in 0..AIR_RES {
                    let depth = fluid.air_depth(ix, layer, iz);
                    let frac = if AIR_DY_M > 0.0 {
                        depth / AIR_DY_M
                    } else {
                        0.0
                    };
                    let dst = ix + layer * AIR_RES + iz * AIR_RES * AIR_LAYERS;
                    out.dust_fog[dst] = frac;
                }
            }
        }
        (
            DUST_FLUID_AIR,
            [origin[0], 0.0, origin[1]],
            [cell, AIR_DY_M, cell],
        )
    } else {
        (DUST_NONE, [0.0; 3], [0.0; 3])
    };
    let mut fog_valid = false;
    if let Some(dust) = dust {
        dust.write_air_velocity(&mut out.air_velocity);
        if dust.fog_air_m3() > 1.0e-12 {
            dust.write_fog(&mut out.fog);
            fog_valid = true;
        }
    } else if let Some(fluid) = fluid {
        fluid.write_air_velocity(&mut out.air_velocity);
        for iz in 0..AIR_RES {
            for layer in 0..AIR_LAYERS {
                for ix in 0..AIR_RES {
                    let depth = fluid.fog_depth(ix, layer, iz);
                    let frac = if AIR_DY_M > 0.0 { depth / AIR_DY_M } else { 0.0 };
                    if frac > 1.0e-12 { fog_valid = true; }
                    let dst = ix + layer * AIR_RES + iz * AIR_RES * AIR_LAYERS;
                    out.fog[dst] = frac;
                }
            }
        }
    }

    let mut spray_count = 0u32;
    for piece in particles {
        if spray_count >= SPRAY_CAPACITY {
            break;
        }
        if !piece.alive || piece.life <= 0.0 || piece.kind == KIND_DUST_MARK {
            continue;
        }
        let density = spray_density_kg_m3(piece.kind);
        let mass = if piece.volume > 0.0 {
            finite0(piece.volume * density)
        } else {
            0.0
        };
        let age = finite0((piece.max_life - piece.life).max(0.0));
        out.spray[spray_count as usize] = SprayGpu {
            pos_mass: [
                finite0(piece.pos.x),
                finite0(piece.pos.y),
                finite0(piece.pos.z),
                mass,
            ],
            vel_age: [
                finite0(piece.vel.x),
                finite0(piece.vel.y),
                finite0(piece.vel.z),
                age,
            ],
            material: piece.kind as u32,
            alive: 1,
            _pad: [0, 0],
        };
        spray_count += 1;
    }
    if let Some(fluid) = fluid {
        for drop in fluid.spray_drops() {
            if spray_count >= SPRAY_CAPACITY {
                break;
            }
            if drop.kind == KIND_DUST_MARK || !drop.volume.is_finite() || drop.volume <= 0.0 {
                continue;
            }
            let mass = finite0(drop.volume * spray_density_kg_m3(drop.kind));
            out.spray[spray_count as usize] = SprayGpu {
                pos_mass: [
                    finite0(drop.pos[0]),
                    finite0(drop.pos[1]),
                    finite0(drop.pos[2]),
                    mass,
                ],
                vel_age: [
                    finite0(drop.vel[0]),
                    finite0(drop.vel[1]),
                    finite0(drop.vel[2]),
                    finite0(drop.age),
                ],
                material: drop.kind as u32,
                alive: 1,
                _pad: [0, 0],
            };
            spray_count += 1;
        }
    }

    let flow_source = if fluid.is_some() {
        FLOW_FLUID_SHEET
    } else {
        FLOW_NONE
    };
    let mut valid = VALID_GROUND
        | VALID_MOISTURE
        | VALID_FILM
        | VALID_CRUST
        | VALID_WET_EDGE
        | VALID_FOAM
        | VALID_SPRAY
        | VALID_MIX;
    if flow_source == FLOW_FLUID_SHEET {
        valid |= VALID_FLOW | VALID_TURBIDITY;
    }
    if fog_valid {
        valid |= VALID_FOG;
    }
    if dust_source != DUST_NONE {
        valid |= VALID_DUST_SETTLED | VALID_DUST_FOG | VALID_AIR;
    }
    if let Some(snow) = snow {
        write_snow_binding(&mut out.snow, snow, origin_x, origin_z);
        valid |= VALID_SNOW;
    }

    scrub(&mut out.ground_rut);
    scrub(&mut out.moisture);
    scrub(&mut out.film);
    scrub(&mut out.flow);
    scrub(&mut out.dust_settled);
    scrub(&mut out.crust);
    scrub(&mut out.wet_edge);
    scrub(&mut out.foam_crest);
    scrub(&mut out.dust_fog);
    scrub(&mut out.air_velocity);
    scrub(&mut out.fog);
    scrub(&mut out.soil_mix);
    scrub(&mut out.snow);
    for slot in &mut out.spray {
        scrub(&mut slot.pos_mass);
        scrub(&mut slot.vel_age);
    }

    out.header = ContractHeader {
        version: RENDER_CONTRACT_VERSION,
        window_res: WINDOW_CELLS,
        air_res: AIR_CELLS,
        air_layers: AIR_SLABS,
        cell_m: WINDOW_CELL_M,
        extent_m: WINDOW_EXTENT_M,
        origin_x,
        origin_z,
        air_origin_x: air_origin[0],
        air_origin_y: air_origin[1],
        air_origin_z: air_origin[2],
        air_cell_x: air_cell[0],
        air_cell_y: air_cell[1],
        air_cell_z: air_cell[2],
        spray_count,
        spray_capacity: SPRAY_CAPACITY,
        frame: fluid.map(|fluid| fluid.frame).unwrap_or(0),
        valid,
        time_s,
        center_x: center[0],
        center_z: center[1],
        flow_source,
        dust_source,
        debug_mode,
    };
}

/// Row bytes padded to the copy alignment. `channels` is floats per texel.
pub fn padded_row_bytes(width: u32, channels: u32) -> u32 {
    let row = width * channels * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    row.div_ceil(align) * align
}

pub fn pack_f32_rows(src: &[f32], width: usize, channels: usize) -> (Vec<u8>, u32) {
    let row_floats = width * channels;
    assert!(row_floats > 0 && src.len() % row_floats == 0);
    let padded = padded_row_bytes(width as u32, channels as u32);
    let rows = src.len() / row_floats;
    let mut out = vec![0u8; padded as usize * rows];
    for row in 0..rows {
        let start = row * row_floats;
        let bytes = bytemuck::cast_slice(&src[start..start + row_floats]);
        let dst = row * padded as usize;
        out[dst..dst + bytes.len()].copy_from_slice(bytes);
    }
    (out, padded)
}

fn float_entry(
    binding: u32,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

/// Bind group 0 of the debug view. Bindings match `src/render_contract.wgsl`.
pub fn contract_bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render-contract"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: BIND_HEADER,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(
                        std::mem::size_of::<ContractHeader>() as u64
                    ),
                },
                count: None,
            },
            float_entry(BIND_GROUND, wgpu::TextureViewDimension::D2),
            float_entry(BIND_MOISTURE, wgpu::TextureViewDimension::D2),
            float_entry(BIND_FILM, wgpu::TextureViewDimension::D2),
            float_entry(BIND_FLOW, wgpu::TextureViewDimension::D2),
            float_entry(BIND_DUST_SETTLED, wgpu::TextureViewDimension::D2),
            float_entry(BIND_DUST_FOG, wgpu::TextureViewDimension::D3),
            float_entry(BIND_CRUST, wgpu::TextureViewDimension::D2),
            float_entry(BIND_WET_EDGE, wgpu::TextureViewDimension::D2),
            float_entry(BIND_FOAM, wgpu::TextureViewDimension::D2),
            wgpu::BindGroupLayoutEntry {
                binding: BIND_SPRAY,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            float_entry(BIND_AIR, wgpu::TextureViewDimension::D3),
            float_entry(BIND_FOG, wgpu::TextureViewDimension::D3),
            float_entry(BIND_MIX, wgpu::TextureViewDimension::D2),
            float_entry(BIND_SNOW, wgpu::TextureViewDimension::D2),
        ],
    })
}

struct Map2 {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    channels: usize,
}

struct Map3 {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    channels: usize,
}

/// GPU copy of the contract. Absent when the device cannot store the float maps.
pub struct ContractGpu {
    pub layout: wgpu::BindGroupLayout,
    pub bind_group: wgpu::BindGroup,
    header: wgpu::Buffer,
    spray: wgpu::Buffer,
    ground: Map2,
    moisture: Map2,
    film: Map2,
    flow: Map2,
    dust_settled: Map2,
    dust_fog: Map3,
    air_velocity: Map3,
    crust: Map2,
    wet_edge: Map2,
    foam: Map2,
    soil_mix: Map2,
    snow: Map2,
}

impl ContractGpu {
    pub fn try_new(device: &wgpu::Device) -> Option<Self> {
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let created = Self::create(device);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = device.poll(wgpu::Maintain::Wait);
            let err = pollster::block_on(device.pop_error_scope());
            if let Some(err) = err {
                eprintln!("render contract textures unavailable: {err:?}");
                return None;
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            let fut = device.pop_error_scope();
            wasm_bindgen_futures::spawn_local(async move {
                let _ = fut.await;
            });
        }
        created
    }

    fn create(device: &wgpu::Device) -> Option<Self> {
        let layout = contract_bind_group_layout(device);
        let header = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-contract-header"),
            size: std::mem::size_of::<ContractHeader>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spray = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render-contract-spray"),
            size: SPRAY_CAPACITY as u64 * std::mem::size_of::<SprayGpu>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let res = WINDOW_CELLS;
        let ground = map2(
            device,
            "contract-ground-rut",
            res,
            wgpu::TextureFormat::Rgba32Float,
            4,
        )?;
        let moisture = map2(
            device,
            "contract-moisture",
            res,
            wgpu::TextureFormat::R32Float,
            1,
        )?;
        let film = map2(
            device,
            "contract-film",
            res,
            wgpu::TextureFormat::Rg32Float,
            2,
        )?;
        let flow = map2(
            device,
            "contract-flow",
            res,
            wgpu::TextureFormat::Rgba32Float,
            FLOW_CHANNELS,
        )?;
        let dust_settled = map2(
            device,
            "contract-dust-settled",
            res,
            wgpu::TextureFormat::R32Float,
            1,
        )?;
        let dust_fog = map3(
            device,
            "contract-dust-fog",
            wgpu::TextureFormat::R32Float,
            1,
        )?;
        let air_velocity = map3(
            device,
            "contract-air-velocity",
            wgpu::TextureFormat::Rgba32Float,
            4,
        )?;
        let crust = map2(
            device,
            "contract-crust",
            res,
            wgpu::TextureFormat::Rg32Float,
            2,
        )?;
        let wet_edge = map2(
            device,
            "contract-wet-edge",
            res,
            wgpu::TextureFormat::R32Float,
            1,
        )?;
        let foam = map2(
            device,
            "contract-foam-crest",
            res,
            wgpu::TextureFormat::Rg32Float,
            2,
        )?;
        let fog = map3(
            device,
            "contract-fog",
            wgpu::TextureFormat::R32Float,
            1,
        )?;
        let soil_mix = map2(
            device,
            "contract-soil-mix",
            res,
            wgpu::TextureFormat::Rgba32Float,
            4,
        )?;
        let snow = map2(
            device,
            "contract-snow",
            res,
            wgpu::TextureFormat::Rgba32Float,
            4,
        )?;
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render-contract-bg"),
            layout: &layout,
            entries: &[
                bind_buf(BIND_HEADER, header.as_entire_binding()),
                bind_tex(BIND_GROUND, &ground.view),
                bind_tex(BIND_MOISTURE, &moisture.view),
                bind_tex(BIND_FILM, &film.view),
                bind_tex(BIND_FLOW, &flow.view),
                bind_tex(BIND_DUST_SETTLED, &dust_settled.view),
                bind_tex(BIND_DUST_FOG, &dust_fog.view),
                bind_tex(BIND_CRUST, &crust.view),
                bind_tex(BIND_WET_EDGE, &wet_edge.view),
                bind_tex(BIND_FOAM, &foam.view),
                bind_buf(BIND_SPRAY, spray.as_entire_binding()),
                bind_tex(BIND_AIR, &air_velocity.view),
                bind_tex(BIND_FOG, &fog.view),
                bind_tex(BIND_MIX, &soil_mix.view),
                bind_tex(BIND_SNOW, &snow.view),
            ],
        });
        Some(Self {
            layout,
            bind_group,
            header,
            spray,
            ground,
            moisture,
            film,
            flow,
            dust_settled,
            dust_fog,
            air_velocity,
            crust,
            wet_edge,
            foam,
            soil_mix,
            snow,
        })
    }

    pub fn upload(&self, queue: &wgpu::Queue, frame: &ContractFrame) {
        queue.write_buffer(&self.header, 0, bytemuck::bytes_of(&frame.header));
        queue.write_buffer(&self.spray, 0, bytemuck::cast_slice(&frame.spray));
        write_map2(queue, &self.ground, &frame.ground_rut, WINDOW_CELLS);
        write_map2(queue, &self.moisture, &frame.moisture, WINDOW_CELLS);
        write_map2(queue, &self.film, &frame.film, WINDOW_CELLS);
        write_map2(queue, &self.flow, &frame.flow, WINDOW_CELLS);
        write_map2(queue, &self.dust_settled, &frame.dust_settled, WINDOW_CELLS);
        write_map2(queue, &self.crust, &frame.crust, WINDOW_CELLS);
        write_map2(queue, &self.wet_edge, &frame.wet_edge, WINDOW_CELLS);
        write_map2(queue, &self.foam, &frame.foam_crest, WINDOW_CELLS);
        write_map2(queue, &self.soil_mix, &frame.soil_mix, WINDOW_CELLS);
        write_map3(queue, &self.dust_fog, &frame.dust_fog);
        write_map3(queue, &self.air_velocity, &frame.air_velocity);
        write_map2(queue, &self.snow, &frame.snow, WINDOW_CELLS);
    }
}

fn map2(
    device: &wgpu::Device,
    label: &str,
    res: u32,
    format: wgpu::TextureFormat,
    channels: usize,
) -> Option<Map2> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Some(Map2 {
        texture,
        view,
        channels,
    })
}

fn map3(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    channels: usize,
) -> Option<Map3> {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: AIR_CELLS,
            height: AIR_SLABS,
            depth_or_array_layers: AIR_CELLS,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Some(Map3 {
        texture,
        view,
        channels,
    })
}

fn write_map3(queue: &wgpu::Queue, map: &Map3, src: &[f32]) {
    let (bytes, padded) = pack_f32_rows(src, AIR_CELLS as usize, map.channels);
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &map.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded),
            rows_per_image: Some(AIR_SLABS),
        },
        wgpu::Extent3d {
            width: AIR_CELLS,
            height: AIR_SLABS,
            depth_or_array_layers: AIR_CELLS,
        },
    );
}

fn bind_buf(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

fn bind_tex(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn write_map2(queue: &wgpu::Queue, map: &Map2, src: &[f32], res: u32) {
    let (bytes, padded) = pack_f32_rows(src, res as usize, map.channels);
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &map.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded),
            rows_per_image: Some(res),
        },
        wgpu::Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: 1,
        },
    );
}

/// Foam peak and water-drop count on the scripted Niva crossing.
///
/// The contract test prints these. The through-drive ledger prints them
/// from this sample so both tests read the same render contract.
#[cfg(test)]
pub(crate) fn through_drive_contract_sample() -> (f32, u32) {
    let mut drive = crate::world::ThroughDrive::new();
    let dt = 1.0 / 30.0;
    let mut frame = ContractFrame::new();
    let mut foam_peak = 0.0f32;
    let mut water_drops = 0u32;
    // The ford is the puddle ending at z = -11.2. Keep going until the
    // Niva is on the bank and the peeled drops are still in the air.
    while drive.world.vehicle.pos.z < -4.0 {
        drive.step(dt);
        publish(
            &mut frame,
            &drive.world.soil,
            &drive.world.particles.items,
            [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
            drive.world.vehicle.pos.z,
            0,
            Some(&drive.fluid),
            Some(drive.dust()),
        );
        assert!(
            contract_frame_is_finite(&frame),
            "non-finite contract sample at z {:.2}",
            drive.world.vehicle.pos.z
        );
        for foam in frame.foam_crest.iter().step_by(2) {
            foam_peak = foam_peak.max(*foam);
        }
        if drive.world.vehicle.pos.z > -11.2 {
            water_drops = water_drops.max(contract_water_drops(&frame));
        }
    }
    (foam_peak, water_drops)
}

#[cfg(test)]
fn contract_water_drops(frame: &ContractFrame) -> u32 {
    frame.spray[..frame.header.spray_count as usize]
        .iter()
        .filter(|slot| {
            slot.alive == 1
                && slot.material == 0
                && slot.pos_mass[3] > 0.0
                && slot.pos_mass.iter().all(|c| c.is_finite())
                && slot.vel_age.iter().all(|c| c.is_finite())
        })
        .count() as u32
}

#[cfg(test)]
fn contract_frame_is_finite(frame: &ContractFrame) -> bool {
    let maps = [
        frame.ground_rut.as_slice(),
        frame.moisture.as_slice(),
        frame.film.as_slice(),
        frame.flow.as_slice(),
        frame.dust_settled.as_slice(),
        frame.crust.as_slice(),
        frame.wet_edge.as_slice(),
        frame.foam_crest.as_slice(),
        frame.dust_fog.as_slice(),
        frame.air_velocity.as_slice(),
        frame.fog.as_slice(),
        frame.soil_mix.as_slice(),
        frame.snow.as_slice(),
    ];
    if maps.iter().any(|map| map.iter().any(|v| !v.is_finite())) {
        return false;
    }
    frame.spray.iter().all(|slot| {
        slot.pos_mass.iter().all(|c| c.is_finite()) && slot.vel_age.iter().all(|c| c.is_finite())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fluid_core::TirePush;
    use crate::particles::Particle;
    use glam::Vec3;

    #[test]
    fn version_and_layout_match_the_wgsl() {
        let wgsl = include_str!("render_contract.wgsl");
        assert!(
            wgsl.contains(&format!(
                "const RENDER_CONTRACT_VERSION: u32 = {RENDER_CONTRACT_VERSION}u;"
            )),
            "bump the WGSL constant with the Rust version"
        );
        assert!(wgsl.contains("docs/RENDER_CONTRACT.md"));
        assert_eq!(std::mem::size_of::<ContractHeader>(), 96);
        assert_eq!(std::mem::size_of::<SprayGpu>(), 48);
        assert!(wgsl.contains("@group(0) @binding(0)"));
        assert!(wgsl.contains("@group(0) @binding(10)"));
        assert!(wgsl.contains("@group(0) @binding(11)"));
        assert!(wgsl.contains("@group(0) @binding(12)"));
        assert!(wgsl.contains("var fog_mist: texture_3d<f32>"));
        assert_eq!(VALID_FOG, 1 << 13);
        assert_eq!(BIND_FOG, 12);
        assert_ne!(VALID_FOG, 1 << 11);
        assert_ne!(VALID_FOG, 1 << 12);
        assert!(wgsl.contains("@group(0) @binding(13) var soil_mix"));
        assert!(wgsl.contains("var soil_mix: texture_2d<f32>"));
        assert_eq!(VALID_MIX, 1 << 12);
        assert_eq!(BIND_MIX, 13);
        assert!(wgsl.contains("@group(0) @binding(14)"));
        assert!(wgsl.contains("var air_velocity: texture_3d<f32>"));
        assert!(wgsl.contains("var snow_tex: texture_2d<f32>"));
        assert_eq!(VALID_SNOW, 1 << 14);
        assert_eq!(BIND_SNOW, 14);
        assert!(wgsl.contains("flow.a"));
        assert!(wgsl.contains("flow.b"));
        let air_row = padded_row_bytes(AIR_CELLS, 4);
        assert_eq!(air_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
        assert!(!wgsl.contains("height_tex"));
        assert!(!wgsl.contains("sun_dir"));
        assert!(!wgsl.contains("ground_diff"));
        let row = padded_row_bytes(WINDOW_CELLS, 4);
        assert_eq!(row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT, 0);
    }

    #[test]
    fn foam_follows_froude_and_a_thin_film_stays_clear() {
        assert_eq!(foam_fraction(0.0, 0.2), 0.0);
        assert_eq!(foam_fraction(3.0, 0.001), 0.0);
        let quiet = foam_fraction(0.3, 0.05);
        assert!(quiet < 0.01, "slow film foamed {quiet}");
        let breaking = foam_fraction(2.0, 0.05);
        assert!(breaking > 0.99, "breaking film stayed clear {breaking}");
    }

    #[test]
    fn wet_edge_crest_and_spray_mass() {
        let mut soil = SoilField::flat(0.1);
        let half = soil.size * 0.5;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                let idx = j * soil.res + i;
                if x.abs() < 1.2 && z.abs() < 1.2 {
                    soil.water[idx] = 0.05;
                    soil.wetness[idx] = 0.9;
                }
                if x.abs() < 0.3 && z.abs() < 0.3 {
                    soil.placed[idx] = 0.04;
                }
            }
        }
        let mut piece = Particle::default();
        piece.alive = true;
        piece.life = 0.25;
        piece.max_life = 1.0;
        piece.volume = 0.002;
        piece.kind = 0;
        piece.pos = Vec3::new(0.2, 0.4, 0.2);
        piece.vel = Vec3::new(0.1, 2.0, -0.4);
        let mut mark = piece;
        mark.kind = KIND_DUST_MARK;
        let mut frame = ContractFrame::new();
        publish(
            &mut frame,
            &soil,
            &[piece, mark],
            [0.0, 0.0],
            1.5,
            0,
            None,
            None,
        );
        assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
        assert_eq!(frame.header.flow_source, FLOW_NONE);
        assert_eq!(frame.header.valid & VALID_FLOW, 0);
        assert_eq!(frame.header.valid & VALID_FOAM, VALID_FOAM);
        assert!(frame.wet_edge.iter().any(|e| *e > 0.5));
        assert!(frame.wet_edge.iter().any(|e| *e == 0.0));
        assert!(
            frame
                .foam_crest
                .iter()
                .skip(1)
                .step_by(2)
                .any(|c| *c >= 0.04),
            "berm crest missing"
        );
        assert!(frame.foam_crest.iter().step_by(2).all(|f| *f == 0.0));
        assert_eq!(frame.header.spray_count, 1);
        let spray = &frame.spray[0];
        assert!(
            (spray.pos_mass[3] - 2.0).abs() < 1.0e-4,
            "mass {}",
            spray.pos_mass[3]
        );
        assert!((spray.vel_age[3] - 0.75).abs() < 1.0e-4);
        assert_eq!(spray.material, 0);
        assert!(
            frame.crust.iter().all(|c| *c == 0.0),
            "a dry-crust kind with empty arrays is not the field"
        );
        assert_eq!(SOIL_KIND_DRY_CRUST, 3);
    }

    #[test]
    fn crust_arrays_publish_in_metres_and_stay_off_the_ground() {
        let mut soil = SoilField::flat(0.1);
        assert_eq!(soil.kind[0], SOIL_KIND_DRY_CRUST);
        let idx = soil.idx_at(0.25, -0.15).expect("cell");
        soil.bed[idx] = 0.40;
        soil.placed[idx] = 0.03;
        soil.crust[idx] = 0.021;
        soil.crack[idx] = 0.007;
        let mut frame = ContractFrame::new();
        publish(&mut frame, &soil, &[], [0.0, 0.0], 2.0, 7, None, None);
        assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
        assert_eq!(frame.header.valid & VALID_CRUST, VALID_CRUST);
        assert_published_crust(&soil, &frame);

        let mut fluid = crate::fluid_core::FluidCore::new([0.0, 0.0]);
        let _ = fluid.step(1.0 / 60.0);
        let mut with_sheet = ContractFrame::new();
        publish(
            &mut with_sheet,
            &soil,
            &[],
            [0.0, 0.0],
            2.0,
            8,
            Some(&fluid),
            None,
        );
        assert_eq!(with_sheet.header.version, RENDER_CONTRACT_VERSION);
        assert_published_crust(&soil, &with_sheet);
    }

    fn assert_published_crust(soil: &SoilField, frame: &ContractFrame) {
        let res = frame.header.window_res as usize;
        let mut hit = false;
        let mut kind_only = 0u32;
        let mut ground_err = 0.0f32;
        for j in 0..res {
            for i in 0..res {
                let x = frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m;
                let z = frame.header.origin_z + (j as f32 + 0.5) * frame.header.cell_m;
                let Some(sidx) = soil.idx_at(x, z) else {
                    continue;
                };
                let cidx = j * res + i;
                let thickness = frame.crust[cidx * 2];
                let opening = frame.crust[cidx * 2 + 1];
                assert!(
                    (thickness - soil.crust[sidx]).abs() < 1.0e-6,
                    "thickness {thickness} soil {}",
                    soil.crust[sidx]
                );
                assert!(
                    (opening - soil.crack[sidx]).abs() < 1.0e-6,
                    "opening {opening} soil {}",
                    soil.crack[sidx]
                );
                let ground = frame.ground_rut[cidx * 4];
                let expect = soil.bed[sidx] + soil.placed[sidx];
                ground_err = ground_err.max((ground - expect).abs());
                if (soil.crust[sidx] - 0.021).abs() < 1.0e-6
                    && (soil.crack[sidx] - 0.007).abs() < 1.0e-6
                {
                    hit = true;
                    assert!((thickness - 0.021).abs() < 1.0e-6, "thickness {thickness}");
                    assert!((opening - 0.007).abs() < 1.0e-6, "opening {opening}");
                    assert!(
                        (ground - (expect + thickness)).abs() > 0.01,
                        "ground absorbed the skin"
                    );
                } else if soil.crust[sidx] == 0.0 && soil.crack[sidx] == 0.0 {
                    kind_only += 1;
                    assert_eq!(thickness, 0.0);
                    assert_eq!(opening, 0.0);
                }
            }
        }
        assert!(hit, "the painted cell missed the window");
        assert!(kind_only > 100, "kind-only cells {kind_only}");
        assert!(ground_err < 1.0e-5, "ground moved by {ground_err}");
    }

    #[test]
    fn fluid_sheet_publishes_film_and_flow() {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        fluid.pour_water(-1.5, 1.5, -1.5, 1.5, 0.06);
        fluid.push_tire(TirePush {
            x: 0.0,
            z: 0.0,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.36,
            width: 0.22,
            speed: 6.0,
            slip: 0.8,
        });
        let _ = fluid.step(1.0 / 60.0);
        let soil = SoilField::flat(0.2);
        let mut frame = ContractFrame::new();
        publish(
            &mut frame,
            &soil,
            &[],
            [0.0, 0.0],
            0.0,
            3,
            Some(&fluid),
            None,
        );
        assert_eq!(frame.header.flow_source, FLOW_FLUID_SHEET);
        assert_eq!(frame.header.dust_source, DUST_FLUID_AIR);
        assert_eq!(frame.header.valid & VALID_FLOW, VALID_FLOW);
        assert_eq!(frame.header.debug_mode, 3);
        let wet = frame.film.chunks(2).filter(|c| c[0] > 0.01).count();
        assert!(wet > 10, "poured water missing, cells {wet}");
        let (x0, z0, extent) = fluid.window_bounds();
        assert!((frame.header.origin_x - x0).abs() < 1.0e-4);
        assert!((frame.header.origin_z - z0).abs() < 1.0e-4);
        assert!((frame.header.extent_m - extent).abs() < 1.0e-4);
    }

    #[test]
    fn air_velocity_is_published_with_the_dust_ledger() {
        let mut dust = Dust::new();
        dust.set_wind(1.5, -0.9);
        dust.step(&crate::dust::tires_at(0.0, 8.0, 1.6), 0.04, 0.0, 1.0 / 60.0);
        let soil = SoilField::flat(0.2);
        let mut frame = ContractFrame::new();
        publish(
            &mut frame,
            &soil,
            &[],
            [0.0, 0.0],
            0.0,
            14,
            None,
            Some(&dust),
        );
        assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
        assert_eq!(frame.header.dust_source, DUST_LEDGER);
        assert_eq!(frame.header.valid & VALID_AIR, VALID_AIR);
        assert_eq!(frame.air_velocity.len(), frame.dust_fog.len() * 4);
        assert!(frame_is_finite(&frame));
        assert!(
            frame.air_velocity.chunks(4).any(|texel| {
                texel[3] == 0.0 && texel[0].abs() + texel[1].abs() + texel[2].abs() > 0.2
            }),
            "air velocity stayed at rest"
        );
        assert!(frame.air_velocity.chunks(4).all(|texel| texel[3] == 0.0));

        let mut sheet = FluidCore::new([0.0, 0.0]);
        let _ = sheet.step(1.0 / 60.0);
        let mut fluid_frame = ContractFrame::new();
        publish(
            &mut fluid_frame,
            &soil,
            &[],
            [0.0, 0.0],
            0.0,
            0,
            Some(&sheet),
            None,
        );
        assert_eq!(fluid_frame.header.dust_source, DUST_FLUID_AIR);
        assert_eq!(fluid_frame.header.valid & VALID_AIR, VALID_AIR);
        assert!(fluid_frame
            .air_velocity
            .chunks(4)
            .all(|texel| texel[1] == 0.0 && texel[3] == 0.0));
        assert!(frame_is_finite(&fluid_frame));

        let mut bare = ContractFrame::new();
        publish(&mut bare, &soil, &[], [0.0, 0.0], 0.0, 0, None, None);
        assert_eq!(bare.header.valid & VALID_AIR, 0);
        assert!(bare.air_velocity.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn wet_track_moisture_and_film_are_in_the_contract() {
        let mut soil = SoilField::flat(0.08);
        let half = soil.size * 0.5;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                if (x - 0.74).abs() < 0.5 && z.abs() < 1.2 {
                    soil.wetness[j * soil.res + i] = 0.78;
                }
            }
        }
        let mut fluid = FluidCore::new([0.0, 0.0]);
        let poured = 0.004_f32;
        fluid.add_water_at(0.74, 0.0, poured);
        fluid.raise_moisture_disk(0.74, 0.0, 0.45, 0.78);
        let mut frame = ContractFrame::new();
        publish(
            &mut frame,
            &soil,
            &[],
            [0.0, 0.0],
            1.0,
            2,
            Some(&fluid),
            None,
        );
        assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
        let res = frame.header.window_res as usize;
        let mut moist = 0.0_f32;
        let mut n = 0.0_f32;
        let mut film = 0.0_f32;
        let area = frame.header.cell_m * frame.header.cell_m;
        for j in 0..res {
            for i in 0..res {
                let x = frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m;
                let z = frame.header.origin_z + (j as f32 + 0.5) * frame.header.cell_m;
                let idx = j * res + i;
                if (x - 0.74).abs() < 0.5 && z.abs() < 1.2 {
                    moist += frame.moisture[idx];
                    n += 1.0;
                }
                film += frame.film[idx * 2] * area;
            }
        }
        let mean = moist / n.max(1.0);
        assert!(
            mean > 0.70,
            "wet track moisture {mean:.3} over {n:.0} cells"
        );
        assert!(
            (film - poured).abs() < 1.0e-4,
            "drained film {film:.6} poured {poured:.6}"
        );
        assert!(frame.moisture.iter().all(|v| v.is_finite()));
        assert!(frame.film.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn through_drive_contract_is_finite_and_carries_foam_and_water() {
        let (foam_peak, water_drops) = super::through_drive_contract_sample();
        println!("foam peak {foam_peak} water-drop count {water_drops}");
        assert!(foam_peak > 0.0, "foam peak {foam_peak}");
        assert!(water_drops > 0, "water-drop count {water_drops}");
    }

    fn frame_is_finite(frame: &ContractFrame) -> bool {
        let maps = [
            frame.ground_rut.as_slice(),
            frame.moisture.as_slice(),
            frame.film.as_slice(),
            frame.flow.as_slice(),
            frame.dust_settled.as_slice(),
            frame.crust.as_slice(),
            frame.wet_edge.as_slice(),
            frame.foam_crest.as_slice(),
            frame.dust_fog.as_slice(),
            frame.air_velocity.as_slice(),
            frame.fog.as_slice(),
            frame.soil_mix.as_slice(),
            frame.snow.as_slice(),
        ];
        if maps.iter().any(|map| map.iter().any(|v| !v.is_finite())) {
            return false;
        }
        frame.spray.iter().all(|slot| {
            slot.pos_mass.iter().all(|c| c.is_finite())
                && slot.vel_age.iter().all(|c| c.is_finite())
        })
    }
}
