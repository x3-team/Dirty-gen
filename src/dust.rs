//! Dust caller for `silt::fluid_core`.
//!
//! This is the dust-trail API (`Dust`, `inject_dust`, `step_pads`, `set_wind`).
//! The cubic metres live here, on a 128 × 16 × 128 air window 4 m tall.
//! Dust is a passive scalar. Each step it rides the relative wind
//! ([`shift_layers`]: the lee hold up to the roof, whole cells over it), the
//! wake mixes it up and down ([`mix_vertical`]), and it falls on the slow
//! stride. The same grid carries air velocity. `step_vehicle_air` injects the
//! Niva body and wheels, confines the wake, and `advect_density_by_air` moves
//! dust on that residual. Fog is a second scalar on this grid, moved the same
//! way. There is no second air grid and no second ledger.

use std::collections::HashMap;

use glam::Vec3;

/// Moisture where loft stops. Same cutoff as `DUST_WET_MOISTURE` in fluid core.
pub const DUST_WET_MOISTURE: f32 = 0.55;
/// Standing water where loft stops, metres.
pub const DUST_WET_WATER_M: f32 = 0.008;
/// Horizontal cells in the air window. Dust plan.
pub const AIR_RES: u32 = 128;
/// Vertical layers. The bottom layer settles. 16 × 0.25 m = 4 m of air.
pub const AIR_LAYERS: u32 = 16;
/// Eddy diffusivity of the open air, m²/s. 0: the ledger keeps still air
/// still, as before the 16 layers; only the resolved wake mixes.
pub const MIX_K_AIR: f32 = 0.0;
/// Eddy diffusivity per m/s of air speed the Niva induces (|v − wind|), m²/s.
/// Mixing-length model: K = c · |u'| · ℓ with c ≈ 0.3 and ℓ = 0.5 m.
pub const MIX_K_WAKE: f32 = 0.15;
/// Wake residual below this mixes nothing, m/s. Keeps the far, settled
/// plume from smearing a film of dust into every layer.
pub const MIX_WAKE_MIN_M_S: f32 = 0.3;
/// The wind shift runs on every `SHIFT_STRIDE`-th dust step, `dt` scaled:
/// one shift per 1/30 s frame at the 1/60 dust step.
pub const SHIFT_STRIDE: u32 = 2;
/// The vertical mix runs on every `MIX_STRIDE`-th dust step with `dt` scaled.
pub const MIX_STRIDE: u32 = 4;
/// Settle cadence, frames. Inside the 4–8 band.
pub const SLOW_STRIDE: u32 = 6;
/// Cubic metres per second offered at 1 m/s slip and dryness 1.
pub const DUST_OFFER_K: f32 = 1.4e-3;
/// Fall speed applied on the slow stride, metres per second.
pub const SETTLE_M_S: f32 = 0.24;
pub use crate::fluid_core::{fog_may_lift, FogInject, FOG_MIN_WATER_M, FOG_SETTLE_M_S};
/// Dust frame budget, milliseconds. Offer, advection, settle, and texture publish.
pub const FRAME_BUDGET_MS: f32 = 1.5;

/// Vehicle-frame air window, metres. +X right, +Y up, +Z forward.
pub const AIR_X0_M: f32 = -8.0;
pub const AIR_X1_M: f32 = 8.0;
pub const AIR_Y0_M: f32 = 0.0;
pub const AIR_Y1_M: f32 = 4.0;
pub const AIR_Z0_M: f32 = -12.0;
pub const AIR_Z1_M: f32 = 4.0;
pub const AIR_DX_M: f32 = (AIR_X1_M - AIR_X0_M) / AIR_RES as f32;
pub const AIR_DZ_M: f32 = (AIR_Z1_M - AIR_Z0_M) / AIR_RES as f32;
pub const AIR_DY_M: f32 = (AIR_Y1_M - AIR_Y0_M) / AIR_LAYERS as f32;

const CELL_VOL_M3: f32 = AIR_DX_M * AIR_DY_M * AIR_DZ_M;
/// Ground cells for dust that has left the air window, metres.
/// The render contract splats settled volume onto the 40 m window with this cell.
/// Changing it changes `dust_settled` and requires a render-contract version bump.
pub const SETTLED_CELL_M: f32 = 0.5;

const THROW_AFT_0: f32 = 0.12;
const THROW_AFT_1: f32 = 1.85;
const THROW_SIDE: f32 = 0.36;
const THROW_UP_1: f32 = 1.75;
const THROW_ALONG_C: f32 = 0.90;
const THROW_UP_C: f32 = 0.88;
const THROW_ALONG_R: f32 = 0.62;
const THROW_SIDE_R: f32 = 0.36;
const THROW_UP_R: f32 = 0.78;

/// Solver tire radius. Matches the Niva contact used by the rest of the engine.
pub const WHEEL_RADIUS_M: f32 = 0.38;
/// Hub offsets in chassis space, FL FR RL RR. +Z is forward, +X is right.
pub const HUBS: [[f32; 3]; 4] = [
    [-0.74, -0.20, 1.08],
    [0.74, -0.20, 1.08],
    [-0.74, -0.20, -1.18],
    [0.74, -0.20, -1.18],
];
/// Chassis origin height that puts the contacts on y = 0.
pub const CHASSIS_Y: f32 = 0.20 + WHEEL_RADIUS_M;

pub const PROOF_SPEED_M_S: f32 = 8.0;
pub const PROOF_SLIP_M_S: f32 = 1.6;
pub const PROOF_MOISTURE: f32 = 0.04;
pub const PROOF_WATER_M: f32 = 0.0;
pub const PROOF_SECONDS: f32 = 4.0;
pub const PROOF_DT_S: f32 = 1.0 / 60.0;

pub fn proof_wind() -> Vec3 {
    Vec3::new(1.5, 0.0, -0.9)
}

/// False when moisture or standing water has closed the dust gate.
pub fn dust_may_lift(moisture: f32, water_m: f32) -> bool {
    moisture < DUST_WET_MOISTURE && water_m < DUST_WET_WATER_M
}

/// 1 on dry ground, 0 when [`dust_may_lift`] is false.
pub fn dryness(moisture: f32, water_m: f32) -> f32 {
    if !dust_may_lift(moisture, water_m) {
        return 0.0;
    }
    let m = moisture.max(0.0);
    ((DUST_WET_MOISTURE - m) / DUST_WET_MOISTURE).clamp(0.0, 1.0)
}

/// Cubic metres per second one tire offers the air grid.
pub fn offer_m3_s(slip: f32, moisture: f32, water_m: f32) -> f32 {
    let dry = dryness(moisture, water_m);
    let slip = slip.max(0.0);
    if dry == 0.0 || slip == 0.0 {
        0.0
    } else {
        DUST_OFFER_K * slip * dry
    }
}

/// Offer at one contact. `inject_dust` decides what actually enters.
#[derive(Clone, Copy, Debug)]
pub struct DustInject {
    pub x: f32,
    pub z: f32,
    pub slip: f32,
    pub moisture: f32,
    pub water_m: f32,
    pub offer_m3_s: f32,
}

impl DustInject {
    pub fn from_contact(x: f32, z: f32, slip: f32, moisture: f32, water_m: f32) -> Self {
        Self {
            x,
            z,
            slip,
            moisture,
            water_m,
            offer_m3_s: offer_m3_s(slip, moisture, water_m),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tire {
    pub index: u8,
    pub contact: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub speed: f32,
    /// Longitudinal slip speed at the contact, metres per second.
    pub slip: f32,
}

/// Straight drive on y = 0, nose along +Z, chassis x = 0.
pub fn tires_at(chassis_z: f32, speed: f32, slip: f32) -> [Tire; 4] {
    let forward = Vec3::new(0.0, 0.0, 1.0);
    let right = Vec3::new(1.0, 0.0, 0.0);
    std::array::from_fn(|i| {
        let h = HUBS[i];
        Tire {
            index: i as u8,
            contact: Vec3::new(h[0], 0.0, chassis_z + h[2]),
            forward,
            right,
            speed,
            slip,
        }
    })
}

pub fn chassis_at(chassis_z: f32) -> Vec3 {
    Vec3::new(0.0, CHASSIS_Y, chassis_z)
}

/// Airborne dust as a 3D density texture. Samples are volume fraction,
/// indexed `x + y * nx + z * nx * ny`, with `ny` = [`AIR_LAYERS`].
#[derive(Clone, Debug)]
pub struct DensityTexture {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub origin: Vec3,
    pub cell: Vec3,
    pub samples: Vec<f32>,
}

impl DensityTexture {
    pub fn empty() -> Self {
        Self {
            nx: 0,
            ny: 0,
            nz: 0,
            origin: Vec3::ZERO,
            cell: Vec3::ZERO,
            samples: Vec::new(),
        }
    }

    pub fn airborne_m3(&self) -> f32 {
        let vol = (self.cell.x * self.cell.y * self.cell.z) as f64;
        let mut sum = 0.0f64;
        for fraction in &self.samples {
            sum += *fraction as f64 * vol;
        }
        sum as f32
    }

    /// Cell centres and dust volumes above `min_fraction`.
    pub fn splats(&self, min_fraction: f32) -> Vec<(Vec3, f32)> {
        let vol = self.cell.x * self.cell.y * self.cell.z;
        let mut out = Vec::new();
        if self.nx == 0 || vol <= 0.0 {
            return out;
        }
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                for ix in 0..self.nx {
                    let i = (ix + iy * self.nx + iz * self.nx * self.ny) as usize;
                    let fraction = self.samples[i];
                    if fraction <= min_fraction {
                        continue;
                    }
                    let pos = self.origin
                        + Vec3::new(
                            (ix as f32 + 0.5) * self.cell.x,
                            (iy as f32 + 0.5) * self.cell.y,
                            (iz as f32 + 0.5) * self.cell.z,
                        );
                    out.push((pos, fraction * vol));
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DustLedger {
    pub emitted_m3: f32,
    pub dust_air_m3: f32,
    pub dust_settled_m3: f32,
}

/// Dust channel on the air window. Wind is stored for `FluidCore::set_wind`.
pub struct Dust {
    vol: Vec<f32>,
    emitted: f64,
    air: f64,
    settled: f64,
    /// Dry soil still in the ground. `None` until [`Dust::set_soil_stock`].
    /// Emission draws this stock down; the ledger reads it back.
    soil: Option<f64>,
    per_tire: [f64; 4],
    wind_x: f32,
    wind_z: f32,
    speed: f32,
    forward: Vec3,
    right: Vec3,
    chassis_x: f32,
    chassis_z: f32,
    /// Longitudinal slip at each hub, metres per second. Feeds wheel spin.
    wheel_slip: [f32; 4],
    /// World air velocity, xyz, same index as `vol`.
    vel: Vec<f32>,
    vel_scratch: Vec<f32>,
    /// Two 128² planes for [`shift_layers`]: the shifted layer and the
    /// volume that left the window. The second plane is all zero between calls.
    shift_scratch: Vec<f32>,
    /// Running sub-cell offset of the layers over the roof, cells (x, z), for
    /// dust and for fog. Whole cells are moved, the remainder carried.
    roof_carry: [(f32, f32); 2],
    /// Coarse air velocity for the active window. Same layout, `AIR_COARSE` times fewer X/Z cells.
    coarse_vel: Vec<f32>,
    coarse_scratch: Vec<f32>,
    /// Bits of wind and heading. A change restamps the free stream onto the fine grid.
    air_stamp_key: [u32; 6],
    /// Occupied dust columns, `ix | (iz << 16)`, rebuilt with the row spans.
    occupied: Vec<u32>,
    air_ready: bool,
    wake_vorticity: f32,
    wake_cells: u32,
    air_nan: u32,
    last_air_ms: f32,
    frame: u32,
    settle_dt: f32,
    x0: u32,
    x1: u32,
    z0: u32,
    z1: u32,
    /// Inclusive x span that holds dust on each z row. Empty when lo > hi.
    row_lo: Vec<u32>,
    row_hi: Vec<u32>,
    /// Settled dust on the ground, 0.5 m cells, world metres. Not a second sheet.
    settled_cells: HashMap<(i32, i32), f64>,
    /// Fog cubic metres. Same index as `vol`. Not dust, and not a second velocity.
    fog: Vec<f32>,
    fog_emitted: f64,
    fog_air: f64,
    fog_settled: f64,
    fog_occupied: Vec<u32>,
    /// Fog that fell back out of the air, 0.5 m cells. The caller owns the water.
    fog_settled_cells: HashMap<(i32, i32), f64>,
}

impl Dust {
    pub fn new() -> Self {
        let n = (AIR_RES * AIR_LAYERS * AIR_RES) as usize;
        let c = (AIR_RES as usize) / crate::fluid_core::AIR_COARSE;
        let cy = (AIR_LAYERS as usize) / crate::fluid_core::AIR_COARSE_Y;
        let coarse_n = c * cy * c;
        Self {
            vol: vec![0.0; n],
            emitted: 0.0,
            air: 0.0,
            settled: 0.0,
            soil: None,
            per_tire: [0.0; 4],
            wind_x: 0.0,
            wind_z: 0.0,
            speed: 0.0,
            forward: Vec3::new(0.0, 0.0, 1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            chassis_x: 0.0,
            chassis_z: 0.0,
            wheel_slip: [0.0; 4],
            vel: vec![0.0; n * 3],
            vel_scratch: vec![0.0; n * 3],
            shift_scratch: vec![0.0; (AIR_RES * AIR_RES * 2) as usize],
            roof_carry: [(0.0, 0.0); 2],
            coarse_vel: vec![0.0; coarse_n * 3],
            coarse_scratch: vec![0.0; coarse_n * 3],
            air_stamp_key: [0; 6],
            occupied: Vec::new(),
            air_ready: false,
            wake_vorticity: 0.0,
            wake_cells: 0,
            air_nan: 0,
            last_air_ms: 0.0,
            frame: 0,
            settle_dt: 0.0,
            x0: 1,
            x1: 0,
            z0: 1,
            z1: 0,
            row_lo: vec![1; AIR_RES as usize],
            row_hi: vec![0; AIR_RES as usize],
            settled_cells: HashMap::new(),
            fog: Vec::new(),
            fog_emitted: 0.0,
            fog_air: 0.0,
            fog_settled: 0.0,
            fog_occupied: Vec::new(),
            fog_settled_cells: HashMap::new(),
        }
    }

    /// Horizontal wind boundary, metres per second. Same shape as `FluidCore::set_wind`.
    pub fn set_wind(&mut self, x: f32, z: f32) {
        self.wind_x = x;
        self.wind_z = z;
    }

    pub fn wind(&self) -> (f32, f32) {
        (self.wind_x, self.wind_z)
    }

    pub fn chassis_z(&self) -> f32 {
        self.chassis_z
    }

    pub fn emitted_m3(&self) -> f32 {
        self.emitted as f32
    }

    pub fn dust_air_m3(&self) -> f32 {
        self.air as f32
    }

    pub fn dust_settled_m3(&self) -> f32 {
        self.settled as f32
    }

    /// Open the dry-soil stock, cubic metres. Later emission draws it down.
    pub fn set_soil_stock(&mut self, m3: f64) {
        self.soil = Some(m3);
    }

    /// Dry soil still in the ground, cubic metres. Zero until a stock is opened.
    pub fn soil_m3(&self) -> f64 {
        self.soil.unwrap_or(0.0)
    }

    /// Settled dust as world `(x, z, m³)` for the picture. The ledger is unchanged.
    pub fn settled_marks(&self) -> Vec<(f32, f32, f32)> {
        let mut out = Vec::new();
        for (&(ix, iz), &volume) in &self.settled_cells {
            if volume <= 1.0e-8 {
                continue;
            }
            let x = (ix as f32 + 0.5) * SETTLED_CELL_M;
            let z = (iz as f32 + 0.5) * SETTLED_CELL_M;
            out.push((x, z, volume as f32));
        }
        out
    }

    /// Settled dust whose 0.5 m cell center matches `pred` (world x, world z).
    pub fn settled_m3_where(&self, mut pred: impl FnMut(f32, f32) -> bool) -> f32 {
        let mut sum = 0.0f64;
        for (&(ix, iz), &volume) in &self.settled_cells {
            let x = (ix as f32 + 0.5) * SETTLED_CELL_M;
            let z = (iz as f32 + 0.5) * SETTLED_CELL_M;
            if pred(x, z) {
                sum += volume;
            }
        }
        sum as f32
    }

    #[cfg(test)]
    fn settled_map_m3(&self) -> f64 {
        self.settled_cells.values().sum()
    }

    #[cfg(test)]
    fn settled_centroid(&self) -> Option<(f32, f32)> {
        let mut mass = 0.0f64;
        let mut sx = 0.0f64;
        let mut sz = 0.0f64;
        for (&(ix, iz), &volume) in &self.settled_cells {
            mass += volume;
            sx += (ix as f64 + 0.5) * (SETTLED_CELL_M as f64) * volume;
            sz += (iz as f64 + 0.5) * (SETTLED_CELL_M as f64) * volume;
        }
        if mass <= 1.0e-14 {
            None
        } else {
            Some(((sx / mass) as f32, (sz / mass) as f32))
        }
    }

    fn note_settled(&mut self, world_x: f32, world_z: f32, volume: f64) {
        if volume <= 0.0 {
            return;
        }
        let ix = (world_x / SETTLED_CELL_M).floor() as i32;
        let iz = (world_z / SETTLED_CELL_M).floor() as i32;
        *self.settled_cells.entry((ix, iz)).or_insert(0.0) += volume;
    }

    pub fn ledger(&self) -> DustLedger {
        DustLedger {
            emitted_m3: self.emitted_m3(),
            dust_air_m3: self.dust_air_m3(),
            dust_settled_m3: self.dust_settled_m3(),
        }
    }

    pub fn emitted_by_tire(&self) -> [f32; 4] {
        self.per_tire.map(|m| m as f32)
    }

    pub fn step(&mut self, tires: &[Tire], moisture: f32, water_m: f32, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.sync_frame(tires);
        for tire in tires {
            let inject = DustInject::from_contact(
                tire.contact.x,
                tire.contact.z,
                tire.slip,
                moisture,
                water_m,
            );
            let entered = inject_dust(self, inject, dt);
            let ti = tire.index as usize;
            if ti < 4 {
                self.per_tire[ti] += entered as f64;
            }
        }
        self.finish_frame(dt);
    }

    pub fn step_pads(&mut self, tires: &[Tire], moisture: &[f32], water_m: &[f32], dt: f32) {
        if dt <= 0.0 {
            return;
        }
        self.sync_frame(tires);
        for (i, tire) in tires.iter().enumerate() {
            let m = moisture.get(i).copied().unwrap_or(0.0);
            let w = water_m.get(i).copied().unwrap_or(0.0);
            let inject = DustInject::from_contact(tire.contact.x, tire.contact.z, tire.slip, m, w);
            let entered = inject_dust(self, inject, dt);
            let ti = tire.index as usize;
            if ti < 4 {
                self.per_tire[ti] += entered as f64;
            }
        }
        self.finish_frame(dt);
    }

    pub fn publish(&self, out: &mut DensityTexture) {
        let n = (AIR_RES * AIR_LAYERS * AIR_RES) as usize;
        if out.samples.len() != n {
            out.samples.resize(n, 0.0);
        }
        out.nx = AIR_RES;
        out.ny = AIR_LAYERS;
        out.nz = AIR_RES;
        out.origin = Vec3::new(AIR_X0_M, AIR_Y0_M, self.chassis_z + AIR_Z0_M);
        out.cell = Vec3::new(AIR_DX_M, AIR_DY_M, AIR_DZ_M);
        let inv = 1.0 / CELL_VOL_M3;
        for (dst, src) in out.samples.iter_mut().zip(self.vol.iter()) {
            *dst = *src * inv;
        }
    }

    pub fn max_fraction(&self) -> f32 {
        let inv = 1.0 / CELL_VOL_M3;
        self.vol.iter().fold(0.0f32, |m, c| m.max(*c * inv))
    }

    /// Positive when that tire's airborne dust sits aft of its contact.
    pub fn aft_of_tire(&self, index: usize) -> Option<f32> {
        let h = HUBS.get(index)?;
        let contact_z = self.chassis_z + h[2];
        let contact_x = h[0];
        let mut volume = 0.0f64;
        let mut z = 0.0f64;
        for iz in 0..AIR_RES {
            for iy in 0..AIR_LAYERS {
                for ix in 0..AIR_RES {
                    let v = self.vol[at(ix, iy, iz)];
                    if v <= 0.0 {
                        continue;
                    }
                    let p = cell_center(self.chassis_z, ix, iy, iz);
                    if (p.x - contact_x).abs() > 0.9
                        || p.z > contact_z + 0.05
                        || p.z < contact_z - 2.4
                    {
                        continue;
                    }
                    let vd = v as f64;
                    volume += vd;
                    z += p.z as f64 * vd;
                }
            }
        }
        if volume <= 1.0e-14 {
            None
        } else {
            Some(contact_z - (z / volume) as f32)
        }
    }

    pub fn volume_ahead_of_front_m3(&self) -> f32 {
        let front_z = self.chassis_z + HUBS[0][2];
        let mut volume = 0.0f64;
        for iz in 0..AIR_RES {
            for iy in 0..AIR_LAYERS {
                for ix in 0..AIR_RES {
                    let v = self.vol[at(ix, iy, iz)];
                    if v <= 0.0 {
                        continue;
                    }
                    let p = cell_center(self.chassis_z, ix, iy, iz);
                    if p.z > front_z + 0.05 {
                        volume += v as f64;
                    }
                }
            }
        }
        volume as f32
    }

    /// `(metres behind the rear axle, metres to the right of the centerline)`.
    pub fn plume_offset(&self) -> (f32, f32) {
        if self.x0 > self.x1 || self.z0 > self.z1 {
            return (0.0, 0.0);
        }
        let rear_z = self.chassis_z + (HUBS[2][2] + HUBS[3][2]) * 0.5;
        let mut volume = 0.0f64;
        let mut sx = 0.0f64;
        let mut sz = 0.0f64;
        for iz in self.z0..=self.z1 {
            for iy in 0..AIR_LAYERS {
                for ix in self.x0..=self.x1 {
                    let v = self.vol[at(ix, iy, iz)];
                    if v <= 0.0 {
                        continue;
                    }
                    let p = cell_center(self.chassis_z, ix, iy, iz);
                    let vd = v as f64;
                    volume += vd;
                    sx += p.x as f64 * vd;
                    sz += p.z as f64 * vd;
                }
            }
        }
        if volume <= 1.0e-14 {
            return (0.0, 0.0);
        }
        let x = (sx / volume) as f32;
        let z = (sz / volume) as f32;
        (rear_z - z, x)
    }

    fn sync_frame(&mut self, tires: &[Tire]) {
        let Some(tire) = tires.first() else {
            return;
        };
        if tire.forward.length_squared() > 1.0e-8 {
            self.forward = tire.forward.normalize();
        }
        if tire.right.length_squared() > 1.0e-8 {
            self.right = tire.right.normalize();
        }
        let i = tire.index as usize;
        if i < HUBS.len() {
            let h = HUBS[i];
            self.chassis_x = tire.contact.x - self.right.x * h[0] - self.forward.x * h[2];
            self.chassis_z = tire.contact.z - self.right.z * h[0] - self.forward.z * h[2];
        }
        self.speed = tire.speed.max(0.0);
        self.wheel_slip = [0.0; 4];
        for tire in tires {
            let k = tire.index as usize;
            if k < 4 {
                self.wheel_slip[k] = tire.slip;
            }
        }
    }

    /// Peak smoothed wake vorticity from the last air step, 1/s.
    pub fn wake_vorticity(&self) -> f32 {
        self.wake_vorticity
    }

    /// Wake cells on the last step whose smoothed |ω| cleared [`crate::fluid_core::WAKE_VORTICITY_MIN`].
    pub fn wake_cell_count(&self) -> u32 {
        self.wake_cells
    }

    /// Non-finite air samples replaced during air steps so far.
    pub fn air_nan_count(&self) -> u32 {
        self.air_nan
    }

    /// Milliseconds spent in the last air step. Zero on wasm.
    pub fn last_air_ms(&self) -> f32 {
        self.last_air_ms
    }

    /// World velocity, xyz, dust-grid index `x + y * nx + z * nx * ny`.
    pub fn air_velocity_xyz(&self) -> &[f32] {
        &self.vel
    }

    /// Airborne fog, cubic metres, same index as the dust volume.
    pub fn fog_volume(&self) -> &[f32] {
        &self.fog
    }

    pub fn fog_emitted_m3(&self) -> f32 {
        self.fog_emitted as f32
    }

    pub fn fog_air_m3(&self) -> f32 {
        self.fog_air as f32
    }

    pub fn fog_settled_m3(&self) -> f32 {
        self.fog_settled as f32
    }

    /// Settled fog, cubic metres, on 0.5 m ground cells. Does not move mass.
    pub fn fog_settled_cells(&self) -> &HashMap<(i32, i32), f64> {
        &self.fog_settled_cells
    }

    /// Pack fog as a volume fraction, cubic metres of water per cubic metre of air.
    /// Same index and cell as `publish` of the dust volume. A is not used.
    pub fn write_fog(&self, dst: &mut [f32]) {
        let n = self.fog.len();
        if dst.len() < n {
            return;
        }
        if self.fog_air <= 1.0e-12 {
            dst[..n].fill(0.0);
            for v in dst.iter_mut().skip(n) {
                *v = 0.0;
            }
            return;
        }
        let inv = 1.0 / CELL_VOL_M3;
        for i in 0..n {
            let v = self.fog[i] * inv;
            dst[i] = if v.is_finite() { v } else { 0.0 };
        }
        for v in dst.iter_mut().skip(n) {
            *v = 0.0;
        }
    }

    /// Pack world velocity into rgba: R +X, G +Y, B +Z, A 0. Metres per second.
    pub fn write_air_velocity(&self, dst: &mut [f32]) {
        let n = self.vol.len();
        if dst.len() < n * 4 || self.vel.len() < n * 3 {
            return;
        }
        for i in 0..n {
            dst[i * 4] = finite_sample(self.vel[i * 3]);
            dst[i * 4 + 1] = finite_sample(self.vel[i * 3 + 1]);
            dst[i * 4 + 2] = finite_sample(self.vel[i * 3 + 2]);
            dst[i * 4 + 3] = 0.0;
        }
    }

    fn solve_air(&mut self, dt: f32) {
        let (drive, solids) = crate::fluid_core::niva_air(
            [self.chassis_x, CHASSIS_Y, self.chassis_z],
            [self.forward.x, self.forward.y, self.forward.z],
            [self.right.x, self.right.y, self.right.z],
            self.speed,
            self.wheel_slip,
            self.wind_x,
            self.wind_z,
        );
        let origin = [AIR_X0_M, AIR_Y0_M, self.chassis_z + AIR_Z0_M];
        let cell = [AIR_DX_M, AIR_DY_M, AIR_DZ_M];
        let res = [AIR_RES as usize, AIR_LAYERS as usize, AIR_RES as usize];
        let key = [
            self.wind_x.to_bits(),
            self.wind_z.to_bits(),
            self.speed.to_bits(),
            self.forward.x.to_bits(),
            self.forward.z.to_bits(),
            self.right.x.to_bits(),
        ];
        let full_stamp = !self.air_ready || key != self.air_stamp_key;
        self.air_stamp_key = key;
        let solve_stride = crate::fluid_core::AIR_SOLVE_STRIDE.max(1);
        let solve_due = full_stamp || self.frame % solve_stride == 0;
        #[cfg(not(target_arch = "wasm32"))]
        let t0 = std::time::Instant::now();
        let report = if solve_due {
            crate::fluid_core::step_dust_air(
                &mut self.vel,
                &mut self.coarse_vel,
                &mut self.coarse_scratch,
                origin,
                cell,
                res,
                &drive,
                &solids,
                dt * solve_stride as f32,
                self.air_ready,
                full_stamp,
            )
        } else {
            crate::fluid_core::AirReport {
                peak_vorticity: self.wake_vorticity,
                wake_cells: self.wake_cells,
                nan_count: 0,
            }
        };
        self.air_ready = true;
        self.wake_vorticity = report.peak_vorticity;
        self.wake_cells = report.wake_cells;
        self.air_nan = self.air_nan.saturating_add(report.nan_count);
        let n = self.vol.len();
        let stride = crate::fluid_core::AIR_FLUX_STRIDE.max(1);
        if !self.occupied.is_empty() && self.frame % stride == 0 {
            let _gap = crate::fluid_core::advect_density_by_air(
                &mut self.vol,
                &mut self.vel_scratch[..n],
                &self.vel,
                origin,
                cell,
                res,
                &drive,
                &solids,
                dt * stride as f32,
                &self.occupied,
            );
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.last_air_ms = t0.elapsed().as_secs_f64() as f32 * 1000.0;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.last_air_ms = 0.0;
        }
    }

    fn finish_frame(&mut self, dt: f32) {
        self.transport_dust(dt);
        self.solve_air(dt);
        if self.fog_air > 1.0e-12 {
            self.transport_fog(dt);
        }
        self.frame = self.frame.wrapping_add(1);
        self.settle_dt += dt;
        if self.frame.is_multiple_of(SLOW_STRIDE) {
            let elapsed = self.settle_dt;
            self.settle_dt = 0.0;
            self.settle_bottom(elapsed);
            if self.fog_air > 1.0e-12 {
                self.settle_fog(elapsed);
            }
        }
    }

    /// Horizontal density step, then the caller settles.
    ///
    /// With `src/fluid_core.rs` on the crate path this calls
    /// `fluid_core::advect_dust_density(vol, wind_x, wind_z, speed, forward, right, dt)`
    /// and books the returned cubic metres as settled. Otherwise the private
    /// wind-and-wake donor cell runs.
    fn transport_dust(&mut self, dt: f32) {
        if dt <= 0.0 || self.x0 > self.x1 {
            return;
        }
        let (gu, gv) = self.relative_wind();
        let shift_dt = self.shift_dt(dt);
        let whole = self.roof_whole(0, gu, gv, shift_dt);
        if shift_layers(&mut self.vol, &mut self.shift_scratch, gu, gv, shift_dt, whole) > 0.0 {
            self.book_outflow(false);
        }
        if self.frame.is_multiple_of(MIX_STRIDE) {
            let drive = self.air_drive();
            mix_vertical(&mut self.vol, &self.vel, &drive, dt * MIX_STRIDE as f32);
        }
        self.rebuild_rows_from_volume();
    }

    /// Book what [`shift_layers`] carried off the window (its `lost`
    /// plane, the second plane of `shift_scratch`) on the ground under the edge.
    fn book_outflow(&mut self, fog: bool) {
        let res = AIR_RES as usize;
        let plane = res * res;
        for iz in 0..AIR_RES {
            for ix in 0..AIR_RES {
                let i = plane + ix as usize + iz as usize * res;
                let m = self.shift_scratch[i] as f64;
                if m <= 0.0 {
                    continue;
                }
                self.shift_scratch[i] = 0.0;
                let p = cell_center(self.chassis_z, ix, 0, iz);
                if fog {
                    self.fog_air -= m;
                    self.fog_settled += m;
                    self.note_fog(p.x, p.z, m);
                } else {
                    self.air -= m;
                    self.settled += m;
                    self.note_settled(p.x, p.z, m);
                }
            }
        }
    }

    fn air_drive(&self) -> crate::fluid_core::AirDrive {
        crate::fluid_core::niva_air(
            [self.chassis_x, CHASSIS_Y, self.chassis_z],
            [self.forward.x, self.forward.y, self.forward.z],
            [self.right.x, self.right.y, self.right.z],
            self.speed,
            self.wheel_slip,
            self.wind_x,
            self.wind_z,
        )
        .0
    }

    /// Whole cells the layers over the roof move this step, with the remainder
    /// carried to the next. `which` 0 is dust, 1 fog.
    fn roof_whole(&mut self, which: usize, gu: f32, gv: f32, shift_dt: f32) -> (i32, i32) {
        let c = &mut self.roof_carry[which];
        c.0 += gu * shift_dt / AIR_DX_M;
        c.1 += gv * shift_dt / AIR_DZ_M;
        let w = (c.0.round() as i32, c.1.round() as i32);
        c.0 -= w.0 as f32;
        c.1 -= w.1 as f32;
        w
    }

    /// `dt` for [`shift_layers`] on this step: the shift runs on every
    /// [`SHIFT_STRIDE`]-th dust step for that many steps, 0 otherwise.
    fn shift_dt(&self, dt: f32) -> f32 {
        if self.frame.is_multiple_of(SHIFT_STRIDE) {
            dt * SHIFT_STRIDE as f32
        } else {
            0.0
        }
    }

    /// Wind relative to the window along `right` and `forward`, m/s.
    fn relative_wind(&self) -> (f32, f32) {
        let forward = if self.forward.length_squared() > 1.0e-8 {
            self.forward.normalize()
        } else {
            Vec3::Z
        };
        let right = if self.right.length_squared() > 1.0e-8 {
            self.right.normalize()
        } else {
            Vec3::X
        };
        let rel = Vec3::new(self.wind_x, 0.0, self.wind_z) - forward * self.speed.max(0.0);
        (rel.dot(right), rel.dot(forward))
    }



    /// `fluid_core` writes `vol` directly. Recover the row spans settle scans.
    fn rebuild_rows_from_volume(&mut self) {
        self.x0 = 1;
        self.x1 = 0;
        self.z0 = 1;
        self.z1 = 0;
        self.occupied.clear();
        for iz in 0..AIR_RES as usize {
            self.row_lo[iz] = 1;
            self.row_hi[iz] = 0;
        }
        let mut live = [false; AIR_RES as usize];
        for iz in 0..AIR_RES {
            occupied_columns(&self.vol, iz, &mut live);
            for ix in 0..AIR_RES {
                if live[ix as usize] {
                    self.touch(ix, iz);
                    self.occupied.push(ix | (iz << 16));
                }
            }
        }
    }

    /// On the slow stride, dust falls and the bottom of each column settles.
    fn settle_bottom(&mut self, elapsed: f32) {
        if self.x0 > self.x1 || self.z0 > self.z1 || elapsed <= 0.0 {
            return;
        }
        let mut remaining = SETTLE_M_S * elapsed;
        while remaining > 1.0e-8 {
            let step = remaining.min(AIR_DY_M);
            self.shift_down(step / AIR_DY_M);
            remaining -= step;
        }
    }

    fn shift_down(&mut self, f: f32) {
        for iz in self.z0..=self.z1 {
            let lo = self.row_lo[iz as usize];
            let hi = self.row_hi[iz as usize];
            if lo > hi {
                continue;
            }
            for ix in lo..=hi {
                if column_empty(self, ix, iz) {
                    continue;
                }
                let mut col = [0.0f32; AIR_LAYERS as usize];
                for iy in 0..AIR_LAYERS {
                    col[iy as usize] = self.vol[at(ix, iy, iz)];
                }
                let before: f64 = col.iter().map(|v| *v as f64).sum();
                let mut leave = [0.0f32; AIR_LAYERS as usize];
                for iy in 0..AIR_LAYERS as usize {
                    let mut dropped = col[iy] * f;
                    if dropped > col[iy] {
                        dropped = col[iy];
                    }
                    leave[iy] = dropped;
                }
                for iy in 0..AIR_LAYERS as usize {
                    let from_above = if iy + 1 < AIR_LAYERS as usize {
                        leave[iy + 1]
                    } else {
                        0.0
                    };
                    col[iy] = col[iy] - leave[iy] + from_above;
                }
                let after: f64 = col.iter().map(|v| *v as f64).sum();
                let settled = before - after;
                self.settled += settled;
                self.air -= settled;
                if settled > 0.0 {
                    let p = cell_center(self.chassis_z, ix, 0, iz);
                    self.note_settled(p.x, p.z, settled);
                }
                for iy in 0..AIR_LAYERS {
                    self.vol[at(ix, iy, iz)] = col[iy as usize];
                }
            }
        }
    }

    fn ensure_fog_grid(&mut self) {
        let n = self.vol.len();
        if self.fog.len() != n {
            self.fog.resize(n, 0.0);
        }
    }

    /// Same wind step and the same wake residual as dust. No second velocity solve.
    fn transport_fog(&mut self, dt: f32) {
        if dt <= 0.0 || self.fog_air <= 1.0e-12 {
            return;
        }
        self.ensure_fog_grid();
        let (gu, gv) = self.relative_wind();
        let shift_dt = self.shift_dt(dt);
        let whole = self.roof_whole(1, gu, gv, shift_dt);
        if shift_layers(&mut self.fog, &mut self.shift_scratch, gu, gv, shift_dt, whole) > 0.0 {
            self.book_outflow(true);
        }
        if self.frame.is_multiple_of(MIX_STRIDE) {
            let drive = self.air_drive();
            mix_vertical(&mut self.fog, &self.vel, &drive, dt * MIX_STRIDE as f32);
        }
        self.rebuild_fog_columns();
        let stride = crate::fluid_core::AIR_FLUX_STRIDE.max(1);
        if !self.fog_occupied.is_empty() && self.frame % stride == 0 {
            let n = self.fog.len();
            let origin = [AIR_X0_M, AIR_Y0_M, self.chassis_z + AIR_Z0_M];
            let cell = [AIR_DX_M, AIR_DY_M, AIR_DZ_M];
            let res = [AIR_RES as usize, AIR_LAYERS as usize, AIR_RES as usize];
            let (drive, solids) = crate::fluid_core::niva_air(
                [self.chassis_x, CHASSIS_Y, self.chassis_z],
                [self.forward.x, self.forward.y, self.forward.z],
                [self.right.x, self.right.y, self.right.z],
                self.speed,
                self.wheel_slip,
                self.wind_x,
                self.wind_z,
            );
            let _gap = crate::fluid_core::advect_density_by_air(
                &mut self.fog,
                &mut self.vel_scratch[..n],
                &self.vel,
                origin,
                cell,
                res,
                &drive,
                &solids,
                dt * stride as f32,
                &self.fog_occupied,
            );
            self.rebuild_fog_columns();
        }
        self.sync_fog_air();
    }




    fn note_fog(&mut self, world_x: f32, world_z: f32, volume: f64) {
        if volume <= 0.0 {
            return;
        }
        let ix = (world_x / SETTLED_CELL_M).floor() as i32;
        let iz = (world_z / SETTLED_CELL_M).floor() as i32;
        *self.fog_settled_cells.entry((ix, iz)).or_insert(0.0) += volume;
    }

    fn rebuild_fog_columns(&mut self) {
        self.fog_occupied.clear();
        let mut live = [false; AIR_RES as usize];
        for iz in 0..AIR_RES {
            occupied_columns(&self.fog, iz, &mut live);
            for ix in 0..AIR_RES {
                if live[ix as usize] {
                    self.fog_occupied.push(ix | (iz << 16));
                }
            }
        }
    }

    fn sync_fog_air(&mut self) {
        let grid: f64 = self.fog.iter().map(|v| *v as f64).sum();
        let residual = self.fog_air - grid;
        if residual.abs() > 1.0e-9 {
            self.fog_settled += residual;
        }
        self.fog_air = grid.max(0.0);
        if self.fog_settled < 0.0 {
            self.fog_settled = 0.0;
        }
    }

    fn settle_fog(&mut self, elapsed: f32) {
        if elapsed <= 0.0 || self.fog_air <= 1.0e-12 {
            return;
        }
        self.rebuild_fog_columns();
        let mut remaining = FOG_SETTLE_M_S * elapsed;
        while remaining > 1.0e-8 {
            let step = remaining.min(AIR_DY_M);
            self.shift_fog_down(step / AIR_DY_M);
            remaining -= step;
        }
        self.rebuild_fog_columns();
        self.sync_fog_air();
    }

    fn shift_fog_down(&mut self, f: f32) {
        let cols = self.fog_occupied.clone();
        for col in cols {
            let ix = col & 0xffff;
            let iz = col >> 16;
            let mut column = [0.0f32; AIR_LAYERS as usize];
            for iy in 0..AIR_LAYERS {
                column[iy as usize] = self.fog[at(ix, iy, iz)];
            }
            let before: f64 = column.iter().map(|v| *v as f64).sum();
            let mut leave = [0.0f32; AIR_LAYERS as usize];
            for iy in 0..AIR_LAYERS as usize {
                let mut dropped = column[iy] * f;
                if dropped > column[iy] {
                    dropped = column[iy];
                }
                leave[iy] = dropped;
            }
            for iy in 0..AIR_LAYERS as usize {
                let from_above = if iy + 1 < AIR_LAYERS as usize {
                    leave[iy + 1]
                } else {
                    0.0
                };
                column[iy] = column[iy] - leave[iy] + from_above;
            }
            let after: f64 = column.iter().map(|v| *v as f64).sum();
            let settled = before - after;
            if settled > 0.0 {
                self.fog_settled += settled;
                self.fog_air -= settled;
                let p = cell_center(self.chassis_z, ix, 0, iz);
                self.note_fog(p.x, p.z, settled);
            }
            for iy in 0..AIR_LAYERS {
                self.fog[at(ix, iy, iz)] = column[iy as usize];
            }
        }
    }

    fn touch(&mut self, ix: u32, iz: u32) {
        let row = iz as usize;
        if self.row_lo[row] > self.row_hi[row] {
            self.row_lo[row] = ix;
            self.row_hi[row] = ix;
        } else {
            self.row_lo[row] = self.row_lo[row].min(ix);
            self.row_hi[row] = self.row_hi[row].max(ix);
        }
        if self.x0 > self.x1 {
            self.x0 = ix;
            self.x1 = ix;
            self.z0 = iz;
            self.z1 = iz;
            return;
        }
        self.x0 = self.x0.min(ix);
        self.x1 = self.x1.max(ix);
        self.z0 = self.z0.min(iz);
        self.z1 = self.z1.max(iz);
    }
}

impl Default for Dust {
    fn default() -> Self {
        Self::new()
    }
}

/// Which columns of z slab `iz` hold anything, reading the slab row by row.
fn occupied_columns(field: &[f32], iz: u32, live: &mut [bool; AIR_RES as usize]) {
    live.fill(false);
    let res = AIR_RES as usize;
    let base = at(0, 0, iz);
    for iy in 0..AIR_LAYERS as usize {
        let row = &field[base + iy * res..base + iy * res + res];
        for (l, v) in live.iter_mut().zip(row) {
            *l |= *v > 0.0;
        }
    }
}

fn column_empty(dust: &Dust, ix: u32, iz: u32) -> bool {
    (0..AIR_LAYERS).all(|iy| dust.vol[at(ix, iy, iz)] == 0.0)
}

/// Deposit `inject.offer_m3_s * dt` behind the contact.
///
/// Returns the cubic metres that entered the air. A wet cell returns 0,
/// and that volume is not taken from the offer into the grid.
pub fn inject_dust(air: &mut Dust, inject: DustInject, dt: f32) -> f32 {
    if dt <= 0.0 || inject.offer_m3_s <= 0.0 || !dust_may_lift(inject.moisture, inject.water_m) {
        return 0.0;
    }
    let Some((ix0, ix1, iy0, iy1, iz0, iz1)) = kernel_bounds(air, inject.x, inject.z) else {
        return 0.0;
    };
    let mut wsum = 0.0f64;
    for iz in iz0..=iz1 {
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                wsum += weight_at(air, inject.x, inject.z, ix, iy, iz) as f64;
            }
        }
    }
    if wsum <= 0.0 {
        return 0.0;
    }
    let budget = (inject.offer_m3_s * dt) as f64;
    let mut added = 0.0f64;
    for iz in iz0..=iz1 {
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                let w = weight_at(air, inject.x, inject.z, ix, iy, iz) as f64;
                if w <= 0.0 {
                    continue;
                }
                let dm = (budget * w / wsum) as f32;
                let i = at(ix, iy, iz);
                let before = air.vol[i];
                let after = before + dm;
                air.vol[i] = after;
                added += (after - before) as f64;
                air.touch(ix, iz);
            }
        }
    }
    air.emitted += added;
    air.air += added;
    if let Some(stock) = air.soil.as_mut() {
        *stock -= added;
    }
    added as f32
}

/// Lift `inject.offer_m3_s * dt` of standing water into the fog scalar.
///
/// Returns the cubic metres that entered the air. A film under
/// [`FOG_MIN_WATER_M`] returns 0. Dust volume, the dust ledger, and the air
/// velocity are left alone. The caller removes the returned volume from the
/// water sheet.
pub fn inject_fog(air: &mut Dust, inject: FogInject, dt: f32) -> f32 {
    if dt <= 0.0 || inject.offer_m3_s <= 0.0 || !fog_may_lift(inject.water_m) {
        return 0.0;
    }
    let Some((ix0, ix1, mut iy0, mut iy1, iz0, iz1)) = kernel_bounds(air, inject.x, inject.z)
    else {
        return 0.0;
    };
    iy0 = iy0.max(1);
    iy1 = iy1.min(3);
    if iy0 > iy1 {
        return 0.0;
    }
    let mut wsum = 0.0f64;
    for iz in iz0..=iz1 {
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                wsum += weight_at(air, inject.x, inject.z, ix, iy, iz) as f64;
            }
        }
    }
    if wsum <= 0.0 {
        return 0.0;
    }
    air.ensure_fog_grid();
    let budget = (inject.offer_m3_s * dt) as f64;
    let mut added = 0.0f64;
    for iz in iz0..=iz1 {
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                let w = weight_at(air, inject.x, inject.z, ix, iy, iz) as f64;
                if w <= 0.0 {
                    continue;
                }
                let dm = (budget * w / wsum) as f32;
                let i = at(ix, iy, iz);
                air.fog[i] += dm;
                added += dm as f64;
            }
        }
    }
    air.fog_emitted += added;
    air.fog_air += added;
    added as f32
}

/// Drive four tires in a straight line. Returns the dust channel and chassis z.
pub fn drive(
    speed: f32,
    slip: f32,
    moisture: f32,
    water_m: f32,
    wind: Vec3,
    seconds: f32,
    dt: f32,
) -> (Dust, f32) {
    let mut dust = Dust::new();
    dust.set_wind(wind.x, wind.z);
    let steps = ((seconds / dt).round() as i32).max(0);
    let mut z = 0.0;
    for _ in 0..steps {
        let tires = tires_at(z, speed, slip);
        dust.step(&tires, moisture, water_m, dt);
        z += speed * dt;
    }
    let chassis_z = dust.chassis_z();
    (dust, chassis_z)
}

/// Straight drive where each tire can sit on its own pad.
pub fn drive_mixed(
    speed: f32,
    slip: f32,
    moisture: [f32; 4],
    water_m: [f32; 4],
    wind: Vec3,
    seconds: f32,
    dt: f32,
) -> (Dust, f32) {
    let mut dust = Dust::new();
    dust.set_wind(wind.x, wind.z);
    let steps = ((seconds / dt).round() as i32).max(0);
    let mut z = 0.0;
    for _ in 0..steps {
        let tires = tires_at(z, speed, slip);
        dust.step_pads(&tires, &moisture, &water_m, dt);
        z += speed * dt;
    }
    let chassis_z = dust.chassis_z();
    (dust, chassis_z)
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub moisture: f32,
    pub water_m: f32,
    pub slip: f32,
    pub dryness: f32,
    pub offer_m3_s: f32,
}

/// Reference offer table. Wet rows and the zero-slip row are zeros.
pub fn reference_samples() -> [Sample; 8] {
    let rows = [
        (0.00, 0.0, 0.0),
        (0.00, 0.0, 1.6),
        (0.04, 0.0, 1.6),
        (0.30, 0.0, 1.6),
        (0.48, 0.0, 1.6),
        (0.55, 0.0, 1.6),
        (0.00, 0.008, 1.6),
        (0.00, 0.02, 1.6),
    ];
    rows.map(|(moisture, water_m, slip)| Sample {
        moisture,
        water_m,
        slip,
        dryness: dryness(moisture, water_m),
        offer_m3_s: offer_m3_s(slip, moisture, water_m),
    })
}

/// Move every layer by the relative wind `(gu, gv)` m/s along +X and +Z for
/// `dt`, one layer per worker. Below the roof each column keeps the lee hold
/// (`fluid_core::dust_stream_hold`) and moves by a linear split between the
/// two cells it lands on; over the roof the layer moves by `whole` cells.
/// Nothing is made or lost inside the window. What crosses its edge is added
/// to the second plane of `scratch` at the edge cell it left from
/// (`x + z * 128`); the caller books it on the ground there. Returns that
/// volume.
fn shift_layers(
    field: &mut [f32],
    scratch: &mut [f32],
    gu: f32,
    gv: f32,
    dt: f32,
    whole: (i32, i32),
) -> f64 {
    let res = AIR_RES as usize;
    let plane = res * res;
    if dt <= 0.0 || field.len() < plane * AIR_LAYERS as usize || scratch.len() < plane * 2 {
        return 0.0;
    }
    if (gu * dt / AIR_DX_M).abs() < 1.0e-6 && (gv * dt / AIR_DZ_M).abs() < 1.0e-6 && whole == (0, 0) {
        return 0.0;
    }
    let lost = &mut scratch[plane..plane * 2];
    let layers: Vec<u32> = (0..AIR_LAYERS).collect();
    #[cfg(not(target_arch = "wasm32"))]
    let outs: Vec<Vec<(usize, f32)>> = {
        use rayon::prelude::*;
        let n = field.len();
        let ptr = LayerPtr(field.as_mut_ptr());
        layers
            .par_iter()
            .map(|&iy| {
                let ptr = &ptr;
                // SAFETY: the worker for layer `iy` reads and writes only the
                // indices `at(_, iy, _)`. Layers are disjoint index sets and the
                // parallel map ends before `field` is used again.
                let field = unsafe { std::slice::from_raw_parts_mut(ptr.get(), n) };
                shift_layer(field, iy, gu, gv, dt, whole)
            })
            .collect()
    };
    #[cfg(target_arch = "wasm32")]
    let outs: Vec<Vec<(usize, f32)>> = layers
        .iter()
        .map(|&iy| shift_layer(field, iy, gu, gv, dt, whole))
        .collect();
    let mut gone = 0.0f64;
    for out in outs {
        for (i, m) in out {
            lost[i] += m;
            gone += m as f64;
        }
    }
    gone
}

/// Lee layers (below the roof) take the fractional per-column shift. Layers
/// over the roof all ride one uniform wind, so they move by whole cells
/// (`whole`, from the caller's running offset): a row copy, no smearing.
fn shift_layer(field: &mut [f32], iy: u32, gu: f32, gv: f32, dt: f32, whole: (i32, i32)) -> Vec<(usize, f32)> {
    if layer_in_lee(iy) {
        SHIFT_PLANE.with(|tmp| shift_one_layer(field, &mut tmp.borrow_mut(), iy, gu, gv, dt))
    } else {
        shift_layer_whole(field, iy, whole.0, whole.1)
    }
}

fn layer_in_lee(iy: u32) -> bool {
    AIR_Y0_M + (iy as f32 + 0.5) * AIR_DY_M < BODY_TOP_M
}

/// Move layer `iy` by `nx` cells along x and `nz` along z. What crosses the
/// window edge is returned at the edge cell it left from.
fn shift_layer_whole(field: &mut [f32], iy: u32, nx: i32, nz: i32) -> Vec<(usize, f32)> {
    let res = AIR_RES as usize;
    let mut lost = Vec::new();
    if nx == 0 && nz == 0 {
        return lost;
    }
    let r = res as i32;
    let edge_x = if nx > 0 { res - 1 } else { 0 };
    let edge_z = if nz > 0 { res - 1 } else { 0 };
    // What leaves: cells whose destination is outside the window. A cell
    // that leaves through both faces is booked once, at its z edge.
    let mut gone_z = [0.0f64; AIR_RES as usize];
    let mut any = false;
    for iz in 0..r {
        let row = at(0, iy, iz as u32);
        let rr = &field[row..row + res];
        let dz_out = !(0..r).contains(&(iz + nz));
        for (ix, v) in rr.iter().enumerate() {
            if *v <= 0.0 {
                continue;
            }
            any = true;
            if dz_out {
                gone_z[ix] += *v as f64;
            } else if !(0..r).contains(&(ix as i32 + nx)) {
                lost.push((edge_x + iz as usize * res, *v));
            }
        }
    }
    if !any {
        return lost;
    }
    for (ix, g) in gone_z.iter().enumerate() {
        if *g > 0.0 {
            lost.push((ix + edge_z * res, *g as f32));
        }
    }
    // One pass of row copies, walking against the motion so no source row is
    // overwritten before it is read.
    let k = nx.unsigned_abs() as usize;
    let walk: Vec<i32> = if nz > 0 { (0..r).rev().collect() } else { (0..r).collect() };
    for dz in walk {
        let sz = dz - nz;
        let dst = at(0, iy, dz as u32);
        if !(0..r).contains(&sz) || k >= res {
            field[dst..dst + res].fill(0.0);
            continue;
        }
        let src = at(0, iy, sz as u32);
        if nx >= 0 {
            field.copy_within(src..src + res - k, dst + k);
            field[dst..dst + k].fill(0.0);
        } else {
            field.copy_within(src + k..src + res, dst);
            field[dst + res - k..dst + res].fill(0.0);
        }
    }
    lost
}

struct LayerPtr(*mut f32);

impl LayerPtr {
    fn get(&self) -> *mut f32 {
        self.0
    }
}
// SAFETY: see `shift_upper_layers`; each worker touches its own layer only.
unsafe impl Send for LayerPtr {}
// SAFETY: as above.
unsafe impl Sync for LayerPtr {}

thread_local! {
    static SHIFT_PLANE: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Top of the Niva body above the ground, metres. Below it, shifted layers
/// keep the sheet's lee hold: the slipstream drags that air along.
const BODY_TOP_M: f32 = CHASSIS_Y
    + crate::fluid_core::NIVA_BODY_OFFSET[1]
    + crate::fluid_core::NIVA_BODY_HALF[1];

/// One layer of [`shift_layers`]. Returns the volume that left the
/// window as `(x + z * 128, volume)` at the edge cell it left from.
///
/// Separable: the whole layer moves `sx` cells along x (linear split between
/// the two cells it lands on), then each column moves its own `sz` along z.
/// Both passes are gathers. What a row or column loses is booked at its
/// downstream edge.
fn shift_one_layer(field: &mut [f32], tmp: &mut Vec<f32>, iy: u32, gu: f32, gv: f32, dt: f32) -> Vec<(usize, f32)> {
    let res = AIR_RES as usize;
    let mut lost = Vec::new();
    // Bounding box of the layer's dust. Most of the upper air is empty.
    let (mut x0, mut x1, mut z0, mut z1) = (res, 0usize, res, 0usize);
    for iz in 0..res {
        let row = at(0, iy, iz as u32);
        let r = &field[row..row + res];
        let Some(a) = r.iter().position(|v| *v > 0.0) else {
            continue;
        };
        let b = r.iter().rposition(|v| *v > 0.0).unwrap_or(a);
        x0 = x0.min(a);
        x1 = x1.max(b);
        z0 = z0.min(iz);
        z1 = z1.max(iz);
    }
    if x0 > x1 {
        return lost;
    }
    if tmp.len() < res * res {
        tmp.resize(res * res, 0.0);
    }
    let last = res as i32 - 1;
    // Pass 1, x: rows z0..=z1 into `tmp`.
    let sx = gu * dt / AIR_DX_M;
    let (nx, fx) = (sx.floor() as i32, sx - sx.floor());
    let edge_x = if sx >= 0.0 { res - 1 } else { 0 };
    let cx0 = (x0 as i32 + nx).clamp(0, last) as usize;
    let cx1 = (x1 as i32 + nx + 1).clamp(0, last) as usize;
    for iz in z0..=z1 {
        let row = at(0, iy, iz as u32);
        let src = &field[row..row + res];
        let dst = &mut tmp[iz * res..iz * res + res];
        dst.fill(0.0);
        // Shares whose landing cell is outside the row.
        let mut gone = 0.0f64;
        for (x, v) in src.iter().enumerate().take(x1 + 1).skip(x0) {
            if *v <= 0.0 {
                continue;
            }
            let a = x as i32 + nx;
            if !(0..=last).contains(&a) {
                gone += (*v * (1.0 - fx)) as f64;
            }
            if !(0..=last).contains(&(a + 1)) {
                gone += (*v * fx) as f64;
            }
        }
        for (x, d) in dst.iter_mut().enumerate().take(cx1 + 1).skip(cx0) {
            let a = x as i32 - nx;
            let b = a - 1;
            let va = if (0..=last).contains(&a) { src[a as usize] } else { 0.0 };
            let vb = if (0..=last).contains(&b) { src[b as usize] } else { 0.0 };
            *d = va * (1.0 - fx) + vb * fx;
        }
        if gone > 0.0 {
            lost.push((edge_x + iz * res, gone as f32));
        }
    }
    // Pass 2, z: each column moves its own `sz` (the lee hold up to the
    // roof). Gathered row by row so the field is written in whole rows.
    let lee = layer_in_lee(iy);
    let edge_z = if gv >= 0.0 { res - 1 } else { 0 };
    let mut nzs = [0i32; AIR_RES as usize];
    let mut fzs = [0.0f32; AIR_RES as usize];
    let (mut nz_lo, mut nz_hi) = (i32::MAX, i32::MIN);
    for ix in cx0..=cx1 {
        let hold = if lee {
            crate::fluid_core::dust_stream_hold(AIR_X0_M + (ix as f32 + 0.5) * AIR_DX_M)
        } else {
            1.0
        };
        let sz = gv * hold * dt / AIR_DZ_M;
        nzs[ix] = sz.floor() as i32;
        fzs[ix] = sz - sz.floor();
        nz_lo = nz_lo.min(nzs[ix]);
        nz_hi = nz_hi.max(nzs[ix]);
    }
    // Shares whose landing cell is outside the column.
    let mut col_gone = [0.0f64; AIR_RES as usize];
    for iz in z0..=z1 {
        let r = &tmp[iz * res..iz * res + res];
        for ix in cx0..=cx1 {
            let v = r[ix];
            if v <= 0.0 {
                continue;
            }
            let a = iz as i32 + nzs[ix];
            if !(0..=last).contains(&a) {
                col_gone[ix] += (v * (1.0 - fzs[ix])) as f64;
            }
            if !(0..=last).contains(&(a + 1)) {
                col_gone[ix] += (v * fzs[ix]) as f64;
            }
        }
        // The x pass already holds this row; clear the field row for the write.
        let row = at(0, iy, iz as u32);
        field[row..row + res].fill(0.0);
    }
    let lo = (z0 as i32 + nz_lo).clamp(0, last) as usize;
    let hi = (z1 as i32 + nz_hi + 1).clamp(0, last) as usize;
    let (zl, zh) = (z0 as i32, z1 as i32);
    for z in lo..=hi {
        let row = at(0, iy, z as u32);
        let dst = &mut field[row..row + res];
        for ix in cx0..=cx1 {
            let a = z as i32 - nzs[ix];
            let b = a - 1;
            let va = if a >= zl && a <= zh { tmp[ix + a as usize * res] } else { 0.0 };
            let vb = if b >= zl && b <= zh { tmp[ix + b as usize * res] } else { 0.0 };
            dst[ix] = va * (1.0 - fzs[ix]) + vb * fzs[ix];
        }
    }
    for ix in cx0..=cx1 {
        let gone = col_gone[ix];
        if gone > 0.0 {
            lost.push((ix + edge_z * res, gone as f32));
        }
    }
    lost
}

/// Vertical exchange in each occupied column: the air's own vertical speed
/// (binding 11 G, upwind) and eddy diffusion. K is [`MIX_K_AIR`] plus
/// [`MIX_K_WAKE`] times the wake residual (air velocity minus the free stream
/// and slipstream, the same residual that moves dust sideways), so the
/// resolved wake churns dust up and the air around it barely moves it. Fluxes run between
/// neighbouring layers of one column: the column sum holds. The top and the
/// ground are closed; settling is the slow stride's job.
fn mix_vertical(field: &mut [f32], vel: &[f32], drive: &crate::fluid_core::AirDrive, dt: f32) {
    let res = AIR_RES as usize;
    let slab = res * AIR_LAYERS as usize;
    let n = slab * res;
    if dt <= 0.0 || field.len() < n || vel.len() < n * 3 {
        return;
    }
    let mut stream = [[0.0f32; 3]; AIR_RES as usize];
    for (ix, s) in stream.iter_mut().enumerate() {
        *s = crate::fluid_core::stream_at(AIR_X0_M + (ix as f32 + 0.5) * AIR_DX_M, drive);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        field[..n]
            .par_chunks_mut(slab)
            .zip(vel[..n * 3].par_chunks(slab * 3))
            .for_each(|(f, v)| mix_slab(f, v, &stream, dt));
    }
    #[cfg(target_arch = "wasm32")]
    for (f, v) in field[..n].chunks_mut(slab).zip(vel[..n * 3].chunks(slab * 3)) {
        mix_slab(f, v, &stream, dt);
    }
}

/// [`mix_vertical`] on one z slab, `x + layer * 128`.
fn mix_slab(field: &mut [f32], vel: &[f32], stream: &[[f32; 3]], dt: f32) {
    let res = AIR_RES as usize;
    let layers = AIR_LAYERS as usize;
    // Explicit diffusion is stable to K dt / dy² = 0.5. Substep to stay at 0.4.
    let k_cap = 0.4 * AIR_DY_M * AIR_DY_M;
    for ix in 0..res {
        let mut c = [0.0f32; AIR_LAYERS as usize];
        let mut sum = 0.0f32;
        for iy in 0..layers {
            c[iy] = field[ix + iy * res];
            sum += c[iy];
        }
        if sum <= 0.0 {
            continue;
        }
        let mut k = [0.0f32; AIR_LAYERS as usize];
        let mut w = [0.0f32; AIR_LAYERS as usize];
        let mut k_max = 0.0f32;
        let mut w_max = 0.0f32;
        for iy in 0..layers {
            let i = (ix + iy * res) * 3;
            let vx = vel[i] - stream[ix][0];
            let vz = vel[i + 2] - stream[ix][2];
            let vy = finite_sample(vel[i + 1]);
            let induced = (vx * vx + vz * vz + vy * vy).sqrt();
            let induced = if induced.is_finite() && induced >= MIX_WAKE_MIN_M_S { induced } else { 0.0 };
            k[iy] = MIX_K_AIR + MIX_K_WAKE * induced;
            w[iy] = vy;
            k_max = k_max.max(k[iy]);
            w_max = w_max.max(vy.abs());
        }
        if k_max <= 0.0 && w_max <= 0.0 {
            continue;
        }
        let rate = (k_max / (AIR_DY_M * AIR_DY_M)).max(w_max / AIR_DY_M);
        let steps = ((rate * dt / 0.4).ceil() as usize).clamp(1, 8);
        let h = dt / steps as f32;
        // Past the cap, K is clipped rather than substepped further.
        let k_lim = k_cap / h;
        let mut flux = [0.0f32; AIR_LAYERS as usize];
        for _ in 0..steps {
            for f in 0..layers - 1 {
                let kf = (0.5 * (k[f] + k[f + 1])).min(k_lim);
                let wf = 0.5 * (w[f] + w[f + 1]);
                let donor = if wf > 0.0 { c[f] } else { c[f + 1] };
                // Upward volume through the face between layer f and f + 1.
                flux[f] = (wf * donor / AIR_DY_M - kf * (c[f + 1] - c[f]) / (AIR_DY_M * AIR_DY_M)) * h;
            }
            // A layer never gives more than it holds.
            for iy in 0..layers {
                let up = if iy + 1 < layers { flux[iy].max(0.0) } else { 0.0 };
                let down = if iy > 0 { (-flux[iy - 1]).max(0.0) } else { 0.0 };
                let out = up + down;
                if out > c[iy] && out > 0.0 {
                    let s = c[iy] / out;
                    if up > 0.0 {
                        flux[iy] *= s;
                    }
                    if down > 0.0 {
                        flux[iy - 1] *= s;
                    }
                }
            }
            for f in 0..layers - 1 {
                c[f] -= flux[f];
                c[f + 1] += flux[f];
            }
        }
        for iy in 0..layers {
            field[ix + iy * res] = c[iy].max(0.0);
        }
    }
}

fn at(ix: u32, iy: u32, iz: u32) -> usize {
    (ix + iy * AIR_RES + iz * AIR_RES * AIR_LAYERS) as usize
}

fn finite_sample(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn cell_center(chassis_z: f32, ix: u32, iy: u32, iz: u32) -> Vec3 {
    Vec3::new(
        AIR_X0_M + (ix as f32 + 0.5) * AIR_DX_M,
        AIR_Y0_M + (iy as f32 + 0.5) * AIR_DY_M,
        chassis_z + AIR_Z0_M + (iz as f32 + 0.5) * AIR_DZ_M,
    )
}

fn index_range(min_w: f32, max_w: f32, origin: f32, d: f32, n: u32) -> Option<(u32, u32)> {
    let a = ((min_w - origin) / d).floor() as i32;
    let b = ((max_w - origin) / d).floor() as i32;
    let a = a.max(0);
    let b = b.min(n as i32 - 1);
    if a > b {
        None
    } else {
        Some((a as u32, b as u32))
    }
}

fn kernel_bounds(air: &Dust, x: f32, z: f32) -> Option<(u32, u32, u32, u32, u32, u32)> {
    let contact = Vec3::new(x, 0.0, z);
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for &along in &[THROW_AFT_0, THROW_AFT_1] {
        for &side in &[-THROW_SIDE, THROW_SIDE] {
            for &up in &[0.0f32, THROW_UP_1] {
                let p = contact - air.forward * along + air.right * side + Vec3::Y * up;
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
    }
    let z_origin = air.chassis_z + AIR_Z0_M;
    let (ix0, ix1) = index_range(lo.x, hi.x, AIR_X0_M, AIR_DX_M, AIR_RES)?;
    let (iy0, iy1) = index_range(lo.y, hi.y, AIR_Y0_M, AIR_DY_M, AIR_LAYERS)?;
    let (iz0, iz1) = index_range(lo.z, hi.z, z_origin, AIR_DZ_M, AIR_RES)?;
    Some((ix0, ix1, iy0, iy1, iz0, iz1))
}

fn weight_at(air: &Dust, x: f32, z: f32, ix: u32, iy: u32, iz: u32) -> f32 {
    let rel = cell_center(air.chassis_z, ix, iy, iz) - Vec3::new(x, 0.0, z);
    let along = -rel.dot(air.forward);
    let side = rel.dot(air.right);
    throw_weight(along, side, rel.y)
}

/// One donor-cell substep. Mass that leaves the window is returned as cubic metres.
fn throw_weight(along: f32, side: f32, up: f32) -> f32 {
    if !(THROW_AFT_0..=THROW_AFT_1).contains(&along)
        || side.abs() > THROW_SIDE
        || !(0.0..=THROW_UP_1).contains(&up)
    {
        return 0.0;
    }
    let a = (along - THROW_ALONG_C) / THROW_ALONG_R;
    let s = side / THROW_SIDE_R;
    let h = (up - THROW_UP_C) / THROW_UP_R;
    let q = 1.0 - (a * a + s * s + h * h);
    let mut w = if q <= 0.0 { 0.0 } else { q * q };
    // A thin skirt on the bottom layer so dust that has fallen can leave the grid.
    if up < AIR_DY_M {
        let ha = 1.0 - a * a;
        let hs = 1.0 - s * s;
        if ha > 0.0 && hs > 0.0 {
            w += 0.08 * ha * hs;
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} != {b} (±{eps})");
    }

    fn closes(dust: &Dust) {
        let err = (dust.emitted - dust.air - dust.settled).abs();
        assert!(err < 1.0e-9, "ledger err {err}");
        let books = dust.ledger();
        approx(
            books.emitted_m3,
            books.dust_air_m3 + books.dust_settled_m3,
            1.0e-6,
        );
        let map_err = (dust.settled_map_m3() - dust.settled).abs();
        assert!(
            map_err < 1.0e-9,
            "settled map {map_err} scalar {}",
            dust.settled
        );
    }

    #[test]
    fn wet_gate_and_zero_slip_offer_nothing() {
        assert!(!dust_may_lift(0.55, 0.0));
        assert!(!dust_may_lift(0.9, 0.0));
        assert!(!dust_may_lift(0.0, 0.008));
        assert!(!dust_may_lift(0.0, 0.05));
        assert!(dust_may_lift(0.54, 0.0));
        assert!(dust_may_lift(0.0, 0.007));
        assert_eq!(dryness(0.55, 0.0), 0.0);
        assert_eq!(offer_m3_s(2.0, 0.8, 0.0), 0.0);
        assert_eq!(offer_m3_s(2.0, 0.0, 0.02), 0.0);
        assert_eq!(offer_m3_s(0.0, 0.0, 0.0), 0.0);

        let mut wet = Dust::new();
        let entered = inject_dust(
            &mut wet,
            DustInject {
                x: 0.0,
                z: 0.0,
                slip: 2.0,
                moisture: 0.9,
                water_m: 0.0,
                offer_m3_s: 1.0,
            },
            0.1,
        );
        assert_eq!(entered, 0.0);
        assert_eq!(wet.dust_air_m3(), 0.0);

        let (pond, _) = drive(8.0, 1.6, 0.02, 0.03, Vec3::ZERO, 0.5, PROOF_DT_S);
        assert_eq!(pond.emitted_m3(), 0.0);
        let (roll, _) = drive(8.0, 0.0, 0.0, 0.0, Vec3::ZERO, 0.5, PROOF_DT_S);
        assert_eq!(roll.emitted_m3(), 0.0);
        let (spin, _) = drive(0.0, 1.2, 0.0, 0.0, Vec3::ZERO, 0.5, PROOF_DT_S);
        assert!(
            spin.emitted_m3() > 0.0,
            "slip on a stopped tire still offers dust"
        );
    }

    #[test]
    fn offer_follows_slip_and_dryness() {
        let full = offer_m3_s(1.6, 0.0, 0.0);
        approx(full, DUST_OFFER_K * 1.6, 1.0e-8);
        let d30 = dryness(0.30, 0.0);
        approx(d30, (0.55 - 0.30) / 0.55, 1.0e-6);
        approx(offer_m3_s(1.6, 0.30, 0.0), full * d30, 1.0e-8);
        approx(offer_m3_s(3.2, 0.0, 0.0), full * 2.0, 1.0e-8);
    }

    #[test]
    fn bottom_layer_settles_on_the_slow_stride() {
        let mut dust = Dust::new();
        dust.set_wind(1.5, -0.9);
        for frame in 1..SLOW_STRIDE {
            dust.step(&tires_at(0.0, 8.0, 1.6), 0.0, 0.0, PROOF_DT_S);
            assert_eq!(dust.dust_settled_m3(), 0.0, "frame {frame}");
            assert_eq!(dust.wind(), (1.5, -0.9));
        }
        dust.step(&tires_at(0.0, 8.0, 1.6), 0.0, 0.0, PROOF_DT_S);
        assert!(dust.dust_settled_m3() > 0.0);
        closes(&dust);
    }

    #[test]
    fn ledger_closes_and_dust_sits_behind_each_tire() {
        let (dust, z) = drive(
            PROOF_SPEED_M_S,
            PROOF_SLIP_M_S,
            PROOF_MOISTURE,
            0.0,
            proof_wind(),
            2.0,
            PROOF_DT_S,
        );
        closes(&dust);
        assert!((dust.chassis_z() - z).abs() < 1.0e-4);
        assert!(dust.emitted_m3() > 1.0e-3, "{}", dust.emitted_m3());
        assert!(dust.dust_air_m3() > 0.0);
        assert!(dust.dust_settled_m3() > 0.0);
        let rate = offer_m3_s(PROOF_SLIP_M_S, PROOF_MOISTURE, 0.0);
        approx(
            dust.emitted_m3(),
            rate * 4.0 * 2.0,
            rate * 4.0 * 2.0 * 0.002,
        );
        for i in 0..4 {
            let aft = dust.aft_of_tire(i).unwrap();
            assert!(aft > 0.2, "tire {i} aft {aft}");
            assert!(dust.emitted_by_tire()[i] > 0.0);
        }
        let ahead = dust.volume_ahead_of_front_m3();
        assert!(ahead <= dust.dust_air_m3() * 0.02 + 1.0e-8, "ahead {ahead}");
    }

    #[test]
    fn plume_trails_behind_the_niva_and_leans_with_the_wind() {
        let (wind, _) = drive(
            PROOF_SPEED_M_S,
            PROOF_SLIP_M_S,
            PROOF_MOISTURE,
            0.0,
            proof_wind(),
            2.0,
            PROOF_DT_S,
        );
        let (calm, _) = drive(
            PROOF_SPEED_M_S,
            PROOF_SLIP_M_S,
            PROOF_MOISTURE,
            0.0,
            Vec3::ZERO,
            2.0,
            PROOF_DT_S,
        );
        let (behind, lateral) = wind.plume_offset();
        let (calm_behind, calm_lateral) = calm.plume_offset();
        assert!(
            behind > 2.5,
            "wind plume is {behind:.2} m behind the rear axle"
        );
        assert!(
            calm_behind > 2.5,
            "calm plume is {calm_behind:.2} m behind the rear axle"
        );
        assert!(
            lateral > calm_lateral + 0.45,
            "wind lateral {lateral:.2} calm {calm_lateral:.2}"
        );
        assert_eq!(wind.wind(), (1.5, -0.9));
        closes(&wind);
        closes(&calm);
    }

    #[test]
    fn a_wet_patch_under_one_tire_does_not_inject_there() {
        let (dust, _) = drive_mixed(
            8.0,
            1.6,
            [0.04, 0.04, 0.9, 0.04],
            [0.0, 0.05, 0.0, 0.0],
            Vec3::ZERO,
            1.0,
            PROOF_DT_S,
        );
        let per = dust.emitted_by_tire();
        assert_eq!(per[1], 0.0, "standing water");
        assert_eq!(per[2], 0.0, "wet moisture");
        assert!(per[0] > 0.0 && per[3] > 0.0);
        closes(&dust);
    }

    #[test]
    fn dust_settles_on_wet_and_does_not_lift() {
        for (wet_m, wet_w) in [(0.90_f32, 0.0_f32), (0.0_f32, 0.02_f32)] {
            let mut dust = Dust::new();
            let mut z = 0.0;
            let dt = PROOF_DT_S;
            let dry_steps = (1.5 / dt).round() as i32;
            for _ in 0..dry_steps {
                dust.step(&tires_at(z, 8.0, 1.6), 0.02, 0.0, dt);
                z += 8.0 * dt;
            }
            let emitted = dust.emitted_m3();
            let air_before = dust.dust_air_m3();
            assert!(
                air_before > 1.0e-6,
                "air {air_before} wet ({wet_m},{wet_w})"
            );
            let mut peak = air_before;
            let wet_steps = (7.0 / dt).round() as i32;
            for _ in 0..wet_steps {
                dust.step(&tires_at(z, 8.0, 1.6), wet_m, wet_w, dt);
                z += 8.0 * dt;
                let air = dust.dust_air_m3();
                assert!(
                    air <= peak + 1.0e-5,
                    "dust lifted on wet ({wet_m},{wet_w}): {air} > {peak}"
                );
                peak = air;
            }
            approx(dust.emitted_m3(), emitted, 1.0e-6);
            // One upwind step at 1/60 leaves more of the plume inside the 16 m
            // window. The open face used to book that smear as settled. The
            // fall still clears it. With 16 layers the wake mixes the dry
            // plume up to 4 m, and dust at 3 m takes 12 s to fall at
            // SETTLE_M_S: 15.5% stayed aloft after 7 s (8 layers: under 8%).
            assert!(
                dust.dust_air_m3() < air_before * 0.20,
                "still airborne {} after wet ({wet_m},{wet_w})",
                dust.dust_air_m3()
            );
            assert!(dust.dust_settled_m3() > emitted * 0.80);
            closes(&dust);
            let settled = dust.dust_settled_m3();
            dust.step(&tires_at(z, 8.0, 2.0), wet_m, wet_w, dt);
            assert!(dust.dust_settled_m3() + 1.0e-6 >= settled);
            assert!(dust.dust_air_m3() <= peak + 1.0e-5);
        }
    }

    #[test]
    fn density_texture_is_128_by_sixteen_layers() {
        let (dust, _) = drive(8.0, 1.6, 0.04, 0.0, proof_wind(), 1.0, PROOF_DT_S);
        let mut tex = DensityTexture::empty();
        dust.publish(&mut tex);
        assert_eq!(tex.nx, AIR_RES);
        assert_eq!(tex.nz, AIR_RES);
        assert_eq!(tex.ny, AIR_LAYERS);
        assert_eq!(tex.samples.len(), (AIR_RES * AIR_LAYERS * AIR_RES) as usize);
        let err = (tex.airborne_m3() - dust.dust_air_m3()).abs();
        assert!(err < 1.0e-5, "texture volume err {err}");
        assert!(dust.max_fraction() > 0.0);
        assert!(!tex.splats(1.0e-6).is_empty());
    }

    /// Wet track, slurry, then dry ground. Dust uses the air grid already in this crate.
    /// Water, slurry, and soil are read back from the solver on one ledger.
    #[test]
    fn through_drive() {
        // Puddle x −6..6, z −24..0, 20 mm. Slurry x −2..2, z −16..−8, 40 mm.
        assert_eq!(AIR_RES, 128);
        assert_eq!(AIR_LAYERS, 16);

        let mut fluid = crate::fluid_core::FluidCore::new([0.0, -12.0]);
        fluid.set_wind(1.5, -0.9);
        let poured_water = fluid.pour_water(-6.0, 6.0, -24.0, 0.0, 0.02) as f64;
        let poured_slurry = fluid.pour_slurry(
            -2.0,
            2.0,
            -16.0,
            -8.0,
            0.04,
            crate::fluid_core::Grain::Clay,
            0.55,
        ) as f64;

        let mut dust = Dust::new();
        dust.set_wind(1.5, -0.9);
        dust.set_soil_stock(0.25);
        let soil0 = dust.soil_m3();
        let opened_books = fluid.ledger();
        let water_ref = opened_books.water_m3 as f64;
        let slurry_ref = opened_books.slurry_m3 as f64;
        let sum_ref = water_ref + slurry_ref + soil0;
        let opened = sum_ref;
        assert_volume_converged("water", poured_water, water_ref);
        assert_volume_converged("slurry", poured_slurry, slurry_ref);
        assert_volume_converged("sum", poured_water + poured_slurry + soil0, sum_ref);
        let dt = PROOF_DT_S;
        let speed = PROOF_SPEED_M_S;
        let slip = PROOF_SLIP_M_S;
        let mut chassis_z = -20.0_f32;
        let steps = (5.0_f32 / dt).round() as i32;
        let mut saw_dry_air = false;
        let mut max_lateral = f32::MIN;
        let mut crossed_slurry = false;

        for _ in 0..steps {
            let tires = tires_at(chassis_z, speed, slip);
            let mut moisture = [0.0_f32; 4];
            let mut water = [0.0_f32; 4];
            let mut any_dry = false;
            let mut any_slurry = false;
            for (i, tire) in tires.iter().enumerate() {
                let z = tire.contact.z;
                if z <= 0.0 {
                    moisture[i] = 0.70;
                    water[i] = 0.02;
                } else {
                    moisture[i] = PROOF_MOISTURE;
                    water[i] = 0.0;
                    any_dry = true;
                }
                if tire.contact.x.abs() <= 2.0 && (-16.0..=-8.0).contains(&z) {
                    any_slurry = true;
                }
            }
            let before = dust.emitted;
            dust.step_pads(&tires, &moisture, &water, dt);
            let entered = dust.emitted - before;
            assert!(entered >= -1.0e-12, "dust returned to the soil");
            if !any_dry {
                assert_eq!(entered, 0.0, "wet track lofted at chassis {chassis_z}");
            }
            if any_slurry {
                crossed_slurry = true;
                assert_eq!(entered, 0.0, "slurry lofted at chassis {chassis_z}");
            }
            let (water, slurry, soil, sum, gap) = solver_ledger(&fluid, &dust, opened);
            assert!(
                soil >= -1.0e-9,
                "soil stock went negative at chassis {chassis_z}: {soil}"
            );
            assert_ledger_closed(water, slurry, soil, sum, gap, chassis_z);
            if any_dry && dust.air > 1.0e-6 {
                saw_dry_air = true;
                max_lateral = max_lateral.max(dust.plume_offset().1);
            }
            chassis_z += speed * dt;
        }

        assert!(crossed_slurry, "the Niva never met the slurry patch");
        assert!(
            saw_dry_air,
            "dry ground did not lift dust into the air grid"
        );
        assert!(dust.emitted > 1.0e-3, "emitted {}", dust.emitted);
        closes(&dust);

        let on_wet = dust.settled_m3_where(|_x, z| z <= 0.0);
        assert!(
            on_wet > 1.0e-5,
            "settled on the wet track {on_wet} air {} settled {}",
            dust.dust_air_m3(),
            dust.dust_settled_m3()
        );
        assert!(
            max_lateral > 0.30,
            "wind did not carry the plume, lateral {max_lateral}"
        );
        let (cx, cz) = dust.settled_centroid().expect("settled film");
        assert!(
            cx > 0.05,
            "settled film did not lean with the wind ({cx}, {cz})"
        );
        approx(dust.emitted as f32, 0.02070, 5.0e-5);
        // One donor-cell step at Courant 0.85. The previous pair of half-steps
        // smeared more volume out of the window, so the airborne share was 0.01154.
        // 16 layers moved by the conservative shift (lee hold up to the roof,
        // full relative wind over it) instead of the sheet step, and the wake
        // mixes the plume up to 4 m. More of it rides off the back of the
        // window and lands on the wet track behind. The sheet's pressure
        // filter no longer spreads the plume sideways. The 8-layer sheet read
        // air 0.01373, settled 0.00697, wet track 0.000115, lean 1.25 m.
        approx(dust.air as f32, 0.01085, 2.5e-4);
        approx(dust.settled as f32, 0.00985, 2.5e-4);
        assert!(
            (on_wet - 0.000787).abs() < 1.0e-4,
            "wet track settled {on_wet}"
        );
        assert!(
            (max_lateral - 1.0).abs() < 0.15,
            "plume lean {max_lateral} m"
        );

        let emitted = dust.emitted;
        let air = dust.air;
        let wet_before = on_wet;
        let back = tires_at(-5.0, 0.0, 2.0);
        for _ in 0..(SLOW_STRIDE * 4) {
            dust.step_pads(&back, &[0.70; 4], &[0.02; 4], dt);
            assert_eq!(dust.emitted, emitted, "wet ground lofted");
            assert!(dust.air <= air + 1.0e-8, "air rose on the wet track");
        }
        let wet_after = dust.settled_m3_where(|_x, z| z <= 0.0);
        assert!(
            wet_after + 1.0e-6 >= wet_before,
            "settled dust left the wet track {wet_before} -> {wet_after}"
        );
        let (water, slurry, soil, sum, gap) = solver_ledger(&fluid, &dust, opened);
        let (foam_peak, water_drops) = crate::render_contract::through_drive_contract_sample();
        println!(
            "water {water} slurry {slurry} soil {soil} sum {sum} gap {gap} foam peak {foam_peak} water-drop count {water_drops}"
        );
        assert_volume_converged("water", water, water_ref);
        assert_volume_converged("slurry", slurry, slurry_ref);
        assert_volume_converged("sum", sum, sum_ref);
        assert_ledger_closed(water, slurry, soil, sum, gap, "wet return");
        assert!(
            (dust.emitted - (soil0 - soil)).abs() < 1.0e-9,
            "soil stock {} does not match emitted {}",
            soil0 - soil,
            dust.emitted
        );
        closes(&dust);
    }

    /// Pour and the closed books against the solver ledger.
    /// Same band as the sheet's pour-to-ledger check.
    const VOLUME_CONVERGENCE: f64 = 1.0e-3;

    fn assert_volume_converged(name: &str, got: f64, reference: f64) {
        let err = (got - reference).abs();
        assert!(
            got.is_finite() && reference.is_finite() && err < VOLUME_CONVERGENCE,
            "{name} {got} ledger {reference} err {err} tol {VOLUME_CONVERGENCE}"
        );
    }

    /// Water and slurry from the sheet, soil from the dust stock.
    fn solver_ledger(
        fluid: &crate::fluid_core::FluidCore,
        dust: &Dust,
        opened: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let books = fluid.ledger();
        let water = books.water_m3 as f64;
        let slurry = books.slurry_m3 as f64;
        let soil = dust.soil_m3();
        let sum = water + slurry + soil + dust.air + dust.settled;
        let gap = (sum - opened).abs();
        (water, slurry, soil, sum, gap)
    }

    fn assert_ledger_closed(
        water: f64,
        slurry: f64,
        soil: f64,
        sum: f64,
        gap: f64,
        at: impl std::fmt::Display,
    ) {
        assert!(
            water.is_finite()
                && slurry.is_finite()
                && soil.is_finite()
                && sum.is_finite()
                && gap.is_finite(),
            "non-finite ledger at {at}: water {water} slurry {slurry} soil {soil} sum {sum} gap {gap}"
        );
        assert!(
            gap < 1.0e-8,
            "shared ledger gap {gap} at {at}: water {water} slurry {slurry} soil {soil} sum {sum}"
        );
    }

    /// Body and wheels push the shared air grid. Dust mass stays on the ledger,
    /// the wake vorticity clears the bar, and the step stays inside its budget.
    #[test]
    fn drive_leaves_a_wake_and_conserves_dust() {
        for (i, hub) in HUBS.iter().enumerate() {
            assert!(
                (hub[0] - crate::fluid_core::NIVA_WHEEL_HUB[i][0]).abs() < 1.0e-6
                    && (hub[2] - crate::fluid_core::NIVA_WHEEL_HUB[i][2]).abs() < 1.0e-6,
                "air hubs drifted from the dust hubs"
            );
        }
        let mut dust = Dust::new();
        dust.set_wind(proof_wind().x, proof_wind().z);
        let dt = PROOF_DT_S;
        let steps = (2.0 / dt).round() as i32;
        let mut z = 0.0f32;
        let mut air_ms = 0.0f64;
        let mut step_ms = 0.0f64;
        let mut peak = 0.0f32;
        let mut wake_cells = 0u32;
        for i in 0..steps {
            let tires = tires_at(z, PROOF_SPEED_M_S, PROOF_SLIP_M_S);
            let t0 = std::time::Instant::now();
            dust.step(&tires, PROOF_MOISTURE, PROOF_WATER_M, dt);
            let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
            if i >= 20 {
                step_ms += elapsed;
                air_ms += dust.last_air_ms() as f64;
            }
            peak = peak.max(dust.wake_vorticity());
            wake_cells = wake_cells.max(dust.wake_cell_count());
            z += PROOF_SPEED_M_S * dt;
        }
        let timed = (steps - 20).max(1) as f64;
        let air_mean = air_ms / timed;
        let step_mean = step_ms / timed;
        closes(&dust);
        let gap = (dust.emitted - dust.air - dust.settled).abs();
        let mut nan = dust.air_nan_count();
        for v in dust.air_velocity_xyz() {
            if !v.is_finite() {
                nan = nan.saturating_add(1);
            }
        }
        for v in &dust.vol {
            if !v.is_finite() {
                nan = nan.saturating_add(1);
            }
        }
        println!(
            "dust emitted {:.6} air {:.6} settled {:.6} gap {:.8}",
            dust.emitted, dust.air, dust.settled, gap
        );
        println!(
            "wake vorticity {:.3} threshold {:.3} cells {wake_cells}",
            peak,
            crate::fluid_core::WAKE_VORTICITY_MIN
        );
        println!("nan {nan}");
        println!("air step ms {air_mean:.3} dust step ms {step_mean:.3}");
        assert!(gap < 1.0e-9, "ledger gap {gap}");
        assert!(
            peak > crate::fluid_core::WAKE_VORTICITY_MIN,
            "wake vorticity {peak} below {}",
            crate::fluid_core::WAKE_VORTICITY_MIN
        );
        assert!(wake_cells > 8, "wake cells {wake_cells}");
        assert_eq!(nan, 0, "nan {nan}");
        assert!(
            air_mean < crate::fluid_core::AIR_STEP_BUDGET_MS as f64,
            "air step {air_mean:.3} ms over {}",
            crate::fluid_core::AIR_STEP_BUDGET_MS
        );
        assert!(dust.emitted > 1.0e-3, "emitted {}", dust.emitted);
        assert!(dust.air > 0.0 && dust.settled > 0.0);
    }
}
