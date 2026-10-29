//! Dust caller for `silt::fluid_core`.
//!
//! Names match the fluid-core API: [`dust_may_lift`], [`DustInject`],
//! [`inject_dust`], [`AIR_RES`], [`AIR_LAYERS`], [`SLOW_STRIDE`],
//! [`DUST_WET_MOISTURE`], [`DUST_WET_WATER_M`], and [`Dust::set_wind`].
//!
//! `src/fluid_core.rs` is not on this branch, so density is advected here by
//! the wind plus a kinematic wake, then settled. That step is not a pressure
//! projection and not vorticity confinement. When `src/fluid_core.rs` is
//! present, `build.rs` sets `has_fluid_core` and the same step calls
//! `fluid_core::advect_dust_density` instead of this copy.
//!
//! The offer is `DUST_OFFER_K * slip * dryness` cubic metres per second.
//! `inject_dust` returns the cubic metres that entered the air, or 0 when
//! [`dust_may_lift`] is false. Every [`SLOW_STRIDE`] frames the column falls
//! and the bottom settles. Settled dust is not lofted. Emitted volume =
//! airborne + settled, and dust that leaves the window is settled.

use glam::Vec3;

/// Moisture where loft stops. Same cutoff as `DUST_WET_MOISTURE` in fluid core.
pub const DUST_WET_MOISTURE: f32 = 0.55;
/// Standing water where loft stops, metres.
pub const DUST_WET_WATER_M: f32 = 0.008;
/// Horizontal cells in the air window. Dust plan.
pub const AIR_RES: u32 = 128;
/// Vertical layers. The bottom layer settles.
pub const AIR_LAYERS: u32 = 8;
/// Settle cadence, frames. Inside the 4–8 band.
pub const SLOW_STRIDE: u32 = 6;
/// Cubic metres per second offered at 1 m/s slip and dryness 1.
pub const DUST_OFFER_K: f32 = 1.4e-3;
/// Fall speed applied on the slow stride, metres per second.
pub const SETTLE_M_S: f32 = 0.24;
/// Dust frame budget, milliseconds. Offer, advection, settle, and texture publish.
pub const FRAME_BUDGET_MS: f32 = 1.5;

/// Vehicle-frame air window, metres. +X right, +Y up, +Z forward.
pub const AIR_X0_M: f32 = -8.0;
pub const AIR_X1_M: f32 = 8.0;
pub const AIR_Y0_M: f32 = 0.0;
pub const AIR_Y1_M: f32 = 2.0;
pub const AIR_Z0_M: f32 = -12.0;
pub const AIR_Z1_M: f32 = 4.0;
pub const AIR_DX_M: f32 = (AIR_X1_M - AIR_X0_M) / AIR_RES as f32;
pub const AIR_DZ_M: f32 = (AIR_Z1_M - AIR_Z0_M) / AIR_RES as f32;
pub const AIR_DY_M: f32 = (AIR_Y1_M - AIR_Y0_M) / AIR_LAYERS as f32;

const CELL_VOL_M3: f32 = AIR_DX_M * AIR_DY_M * AIR_DZ_M;
/// Cells thinner than this are settled. They are below the density texture's visible floor.
const CELL_CULL_M3: f32 = 1.0e-8;

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
    /// Scratch for the donor-cell step. Same layout as `vol`.
    next: Vec<f32>,
    emitted: f64,
    air: f64,
    settled: f64,
    per_tire: [f64; 4],
    wind_x: f32,
    wind_z: f32,
    speed: f32,
    forward: Vec3,
    right: Vec3,
    chassis_z: f32,
    frame: u32,
    settle_dt: f32,
    x0: u32,
    x1: u32,
    z0: u32,
    z1: u32,
    /// Inclusive x span that holds dust on each z row. Empty when lo > hi.
    row_lo: Vec<u32>,
    row_hi: Vec<u32>,
}

impl Dust {
    pub fn new() -> Self {
        let n = (AIR_RES * AIR_LAYERS * AIR_RES) as usize;
        Self {
            vol: vec![0.0; n],
            next: vec![0.0; n],
            emitted: 0.0,
            air: 0.0,
            settled: 0.0,
            per_tire: [0.0; 4],
            wind_x: 0.0,
            wind_z: 0.0,
            speed: 0.0,
            forward: Vec3::new(0.0, 0.0, 1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            chassis_z: 0.0,
            frame: 0,
            settle_dt: 0.0,
            x0: 1,
            x1: 0,
            z0: 1,
            z1: 0,
            row_lo: vec![1; AIR_RES as usize],
            row_hi: vec![0; AIR_RES as usize],
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
        let i = tire.index as usize;
        if i < HUBS.len() {
            self.chassis_z = tire.contact.z - HUBS[i][2];
        }
        if tire.forward.length_squared() > 0.0 {
            self.forward = tire.forward.normalize();
            self.right = tire.right.normalize();
        }
        self.speed = tire.speed.max(0.0);
    }

    fn finish_frame(&mut self, dt: f32) {
        self.transport_dust(dt);
        self.frame = self.frame.wrapping_add(1);
        self.settle_dt += dt;
        if self.frame.is_multiple_of(SLOW_STRIDE) {
            let elapsed = self.settle_dt;
            self.settle_dt = 0.0;
            self.settle_bottom(elapsed);
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
        #[cfg(has_fluid_core)]
        {
            let outflow = crate::fluid_core::advect_dust_density(
                &mut self.vol,
                self.wind_x,
                self.wind_z,
                self.speed,
                self.forward,
                self.right,
                dt,
            );
            if outflow > 0.0 {
                self.air -= outflow;
                self.settled += outflow;
            }
            self.rebuild_rows_from_volume();
        }
        #[cfg(not(has_fluid_core))]
        {
            self.advect_by_wind_and_wake(dt);
        }
    }

    /// `fluid_core` writes `vol` directly. Recover the row spans settle scans.
    #[cfg(has_fluid_core)]
    fn rebuild_rows_from_volume(&mut self) {
        self.x0 = 1;
        self.x1 = 0;
        self.z0 = 1;
        self.z1 = 0;
        for iz in 0..AIR_RES as usize {
            self.row_lo[iz] = 1;
            self.row_hi[iz] = 0;
        }
        for iz in 0..AIR_RES {
            for ix in 0..AIR_RES {
                if (0..AIR_LAYERS).any(|iy| self.vol[at(ix, iy, iz)] > 0.0) {
                    self.touch(ix, iz);
                }
            }
        }
    }

    #[cfg(not(has_fluid_core))]
    fn advect_by_wind_and_wake(&mut self, dt: f32) {
        let rel = Vec3::new(self.wind_x, 0.0, self.wind_z) - self.forward * self.speed;
        let span = rel.x.abs() + rel.z.abs();
        if span < 1.0e-5 {
            return;
        }
        let cfl = span * dt / AIR_DX_M.min(AIR_DZ_M);
        let steps = (cfl / 0.70).ceil().clamp(1.0, 6.0) as u32;
        let sub = dt / steps as f32;
        for _ in 0..steps {
            let (x0, x1, z0, z1, live, outflow) = advect_once(
                &mut self.vol,
                &mut self.next,
                &mut self.row_lo,
                &mut self.row_hi,
                self.z0,
                self.z1,
                self.forward,
                self.right,
                rel,
                sub,
            );
            if live {
                self.x0 = x0;
                self.x1 = x1;
                self.z0 = z0;
                self.z1 = z1;
            } else {
                self.x0 = 1;
                self.x1 = 0;
                self.z0 = 1;
                self.z1 = 0;
                for iz in 0..AIR_RES as usize {
                    self.row_lo[iz] = 1;
                    self.row_hi[iz] = 0;
                }
            }
            if outflow > 0.0 {
                self.air -= outflow;
                self.settled += outflow;
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
                for iy in 0..AIR_LAYERS {
                    self.vol[at(ix, iy, iz)] = col[iy as usize];
                }
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

fn at(ix: u32, iy: u32, iz: u32) -> usize {
    (ix + iy * AIR_RES + iz * AIR_RES * AIR_LAYERS) as usize
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
#[cfg(not(has_fluid_core))]
fn advect_once(
    vol: &mut [f32],
    next: &mut [f32],
    row_lo: &mut [u32],
    row_hi: &mut [u32],
    z0: u32,
    z1: u32,
    forward: Vec3,
    right: Vec3,
    rel: Vec3,
    dt: f32,
) -> (u32, u32, u32, u32, bool, f64) {
    let mut read_lo = [1u32; AIR_RES as usize];
    let mut read_hi = [0u32; AIR_RES as usize];
    if z0 <= z1 {
        for iz in z0..=z1 {
            read_lo[iz as usize] = row_lo[iz as usize];
            read_hi[iz as usize] = row_hi[iz as usize];
        }
    }
    let wz0 = z0.saturating_sub(1);
    let wz1 = (z1 + 1).min(AIR_RES - 1);
    let mut write_lo = [1u32; AIR_RES as usize];
    let mut write_hi = [0u32; AIR_RES as usize];
    for iz in wz0..=wz1 {
        let mut lo = 1u32;
        let mut hi = 0u32;
        let mut any = false;
        let src0 = iz.saturating_sub(1);
        let src1 = (iz + 1).min(AIR_RES - 1);
        for src in src0..=src1 {
            if read_lo[src as usize] > read_hi[src as usize] {
                continue;
            }
            let (a, b) = if src == iz {
                (
                    read_lo[src as usize].saturating_sub(1),
                    (read_hi[src as usize] + 1).min(AIR_RES - 1),
                )
            } else {
                (read_lo[src as usize], read_hi[src as usize])
            };
            if !any {
                lo = a;
                hi = b;
                any = true;
            } else {
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        if !any {
            continue;
        }
        write_lo[iz as usize] = lo;
        write_hi[iz as usize] = hi;
        for iy in 0..AIR_LAYERS {
            for ix in lo..=hi {
                next[at(ix, iy, iz)] = 0.0;
            }
        }
        row_lo[iz as usize] = 1;
        row_hi[iz as usize] = 0;
    }

    let mut outflow = 0.0f64;
    let res = AIR_RES as i32;
    let stream = rel.dot(forward);

    for iz in z0..=z1 {
        if read_lo[iz as usize] > read_hi[iz as usize] {
            continue;
        }
        let gz = AIR_Z0_M + (iz as f32 + 0.5) * AIR_DZ_M;
        for ix in read_lo[iz as usize]..=read_hi[iz as usize] {
            if (0..AIR_LAYERS).all(|iy| vol[at(ix, iy, iz)] == 0.0) {
                continue;
            }
            let gx = AIR_X0_M + (ix as f32 + 0.5) * AIR_DX_M;
            let (lx, lz) = vehicle_local(gx, gz, right, forward);
            let deficit = wake_deficit(lx, lz);
            let vel = rel - forward * (stream * deficit);
            let mut fx = vel.x * dt / AIR_DX_M;
            let mut fz = vel.z * dt / AIR_DZ_M;
            let mag = fx.abs() + fz.abs();
            if mag > 0.92 {
                let s = 0.92 / mag;
                fx *= s;
                fz *= s;
            }
            let ax = fx.abs();
            let az = fz.abs();
            let nix = if fx > 0.0 {
                ix as i32 + 1
            } else if fx < 0.0 {
                ix as i32 - 1
            } else {
                ix as i32
            };
            let niz = if fz > 0.0 {
                iz as i32 + 1
            } else if fz < 0.0 {
                iz as i32 - 1
            } else {
                iz as i32
            };
            let x_moves = fx != 0.0;
            let z_moves = fz != 0.0;
            for iy in 0..AIR_LAYERS {
                let v = vol[at(ix, iy, iz)];
                if v == 0.0 {
                    continue;
                }
                let leave_x = v * ax;
                let mut leave_z = v * az;
                let mut stay = v - leave_x;
                if stay < leave_z {
                    leave_z = stay;
                    stay = 0.0;
                } else {
                    stay -= leave_z;
                }
                deposit(next, ix as i32, iy, iz as i32, stay, res, &mut outflow);
                if x_moves {
                    deposit(next, nix, iy, iz as i32, leave_x, res, &mut outflow);
                }
                if z_moves {
                    deposit(next, ix as i32, iy, niz, leave_z, res, &mut outflow);
                }
            }
        }
    }

    // Drop sub-visible specks back into the fullest surviving cell so the
    // active rows stay on the plume. That mass is still airborne.
    let mut bx0 = 0u32;
    let mut bx1 = 0u32;
    let mut bz0 = 0u32;
    let mut bz1 = 0u32;
    let mut live = false;
    let mut speck = 0.0f64;
    let mut sink = 0usize;
    let mut sink_v = 0.0f32;
    for iz in wz0..=wz1 {
        row_lo[iz as usize] = 1;
        row_hi[iz as usize] = 0;
        if write_lo[iz as usize] > write_hi[iz as usize] {
            continue;
        }
        for ix in write_lo[iz as usize]..=write_hi[iz as usize] {
            let mut any = false;
            for iy in 0..AIR_LAYERS {
                let i = at(ix, iy, iz);
                let v = next[i];
                if v < CELL_CULL_M3 {
                    if v > 0.0 {
                        speck += v as f64;
                    }
                    vol[i] = 0.0;
                } else {
                    vol[i] = v;
                    any = true;
                    if v > sink_v {
                        sink_v = v;
                        sink = i;
                    }
                }
            }
            if !any {
                continue;
            }
            let row = iz as usize;
            if row_lo[row] > row_hi[row] {
                row_lo[row] = ix;
                row_hi[row] = ix;
            } else {
                row_lo[row] = row_lo[row].min(ix);
                row_hi[row] = row_hi[row].max(ix);
            }
            if !live {
                bx0 = ix;
                bx1 = ix;
                bz0 = iz;
                bz1 = iz;
                live = true;
            } else {
                bx0 = bx0.min(ix);
                bx1 = bx1.max(ix);
                bz0 = bz0.min(iz);
                bz1 = bz1.max(iz);
            }
        }
    }
    if speck > 0.0 {
        if live {
            vol[sink] += speck as f32;
        } else {
            outflow += speck;
        }
    }
    (bx0, bx1, bz0, bz1, live, outflow)
}

#[cfg(not(has_fluid_core))]
fn deposit(next: &mut [f32], ix: i32, iy: u32, iz: i32, amount: f32, res: i32, outflow: &mut f64) {
    if amount == 0.0 {
        return;
    }
    if ix < 0 || iz < 0 || ix >= res || iz >= res {
        *outflow += amount as f64;
        return;
    }
    next[at(ix as u32, iy, iz as u32)] += amount;
}

/// Chassis-relative grid offset to vehicle right / forward.
#[cfg(not(has_fluid_core))]
fn vehicle_local(offset_x: f32, offset_z: f32, right: Vec3, forward: Vec3) -> (f32, f32) {
    let lx = offset_x * right.x + offset_z * right.z;
    let lz = offset_x * forward.x + offset_z * forward.z;
    (lx, lz)
}

/// Fraction of the oncoming stream held with the car. No pressure solve.
#[cfg(not(has_fluid_core))]
fn wake_deficit(x: f32, z: f32) -> f32 {
    let mut d = body_wake(x, z);
    d = d.max(tire_wake(x, z, HUBS[0][0], HUBS[0][2]));
    d = d.max(tire_wake(x, z, HUBS[1][0], HUBS[1][2]));
    d = d.max(tire_wake(x, z, HUBS[2][0], HUBS[2][2]));
    d = d.max(tire_wake(x, z, HUBS[3][0], HUBS[3][2]));
    d.clamp(0.0, 0.70)
}

#[cfg(not(has_fluid_core))]
fn body_wake(x: f32, z: f32) -> f32 {
    let aft = -1.90 - z;
    if aft <= 0.0 || aft >= 8.5 {
        return 0.0;
    }
    let ax = x.abs();
    if ax >= 1.15 {
        return 0.0;
    }
    let along = 1.0 - aft / 8.5;
    let side = 1.0 - (ax * ax) / (1.15 * 1.15);
    0.62 * along * side
}

#[cfg(not(has_fluid_core))]
fn tire_wake(x: f32, z: f32, hx: f32, hz: f32) -> f32 {
    let aft = hz - z;
    if aft <= 0.0 || aft >= 3.8 {
        return 0.0;
    }
    let dx = x - hx;
    if dx.abs() >= 0.55 {
        return 0.0;
    }
    let along = 1.0 - aft / 3.8;
    let side = 1.0 - (dx * dx) / (0.55 * 0.55);
    0.36 * along * side
}

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
            assert!(
                dust.dust_air_m3() < air_before * 0.05,
                "still airborne {} after wet ({wet_m},{wet_w})",
                dust.dust_air_m3()
            );
            assert!(dust.dust_settled_m3() > emitted * 0.95);
            closes(&dust);
            let settled = dust.dust_settled_m3();
            dust.step(&tires_at(z, 8.0, 2.0), wet_m, wet_w, dt);
            assert!(dust.dust_settled_m3() + 1.0e-6 >= settled);
            assert!(dust.dust_air_m3() <= peak + 1.0e-5);
        }
    }

    #[test]
    fn density_texture_is_128_by_eight_layers() {
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
}
