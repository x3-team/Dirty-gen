//! Dust caller for `silt::fluid_core`.
//!
//! Names match the fluid-core API: [`dust_may_lift`], [`DustInject`],
//! [`inject_dust`], [`AIR_RES`], [`AIR_LAYERS`], [`SLOW_STRIDE`],
//! [`DUST_WET_MOISTURE`], [`DUST_WET_WATER_M`], and [`Dust::set_wind`].
//! `FluidCore::step` with `SheetDrive::Wind` is the horizontal transport.
//! That module is not on this branch, so this file does not define
//! `fluid_core` and does not keep a second sheet solver.
//!
//! The offer is `DUST_OFFER_K * slip * dryness` cubic metres per second.
//! `inject_dust` returns the cubic metres that entered the air, or 0 when
//! [`dust_may_lift`] is false. Every [`SLOW_STRIDE`] frames the bottom layer
//! settles onto the ground. Settled dust is not lofted. Emitted volume =
//! airborne + settled.

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
pub const SETTLE_M_S: f32 = 0.42;
/// Dust frame budget, milliseconds. Offer, settle, and texture publish.
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

const THROW_AFT_0: f32 = 0.10;
const THROW_AFT_1: f32 = 1.55;
const THROW_SIDE: f32 = 0.34;
const THROW_UP_1: f32 = 1.15;
const THROW_ALONG_C: f32 = 0.78;
const THROW_UP_C: f32 = 0.48;
const THROW_ALONG_R: f32 = 0.55;
const THROW_SIDE_R: f32 = 0.34;
const THROW_UP_R: f32 = 0.42;

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
    per_tire: [f64; 4],
    wind_x: f32,
    wind_z: f32,
    forward: Vec3,
    right: Vec3,
    chassis_z: f32,
    frame: u32,
    settle_dt: f32,
    x0: u32,
    x1: u32,
    z0: u32,
    z1: u32,
}

impl Dust {
    pub fn new() -> Self {
        let n = (AIR_RES * AIR_LAYERS * AIR_RES) as usize;
        Self {
            vol: vec![0.0; n],
            emitted: 0.0,
            air: 0.0,
            settled: 0.0,
            per_tire: [0.0; 4],
            wind_x: 0.0,
            wind_z: 0.0,
            forward: Vec3::new(0.0, 0.0, 1.0),
            right: Vec3::new(1.0, 0.0, 0.0),
            chassis_z: 0.0,
            frame: 0,
            settle_dt: 0.0,
            x0: 1,
            x1: 0,
            z0: 1,
            z1: 0,
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
    }

    fn finish_frame(&mut self, dt: f32) {
        self.frame = self.frame.wrapping_add(1);
        self.settle_dt += dt;
        if self.frame.is_multiple_of(SLOW_STRIDE) {
            let elapsed = self.settle_dt;
            self.settle_dt = 0.0;
            self.settle_bottom(elapsed);
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
            for ix in self.x0..=self.x1 {
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
    if q <= 0.0 {
        0.0
    } else {
        q * q
    }
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
