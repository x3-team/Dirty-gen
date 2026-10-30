//! One solver for the ground sheets and the dust air.
//!
//! Water and slurry are height–velocity sheets. Pressure is a two-pass
//! separable filter, a wide gaussian plus a steeper kernel. Viscosity is
//! the mix, and it enters by overloading the divergence the projection
//! removes. Dust is a 2.5D air grid on the same advection, projection, and
//! vorticity confinement. The window moves with the Niva: a coarse sheet
//! about 20 cm across the near field, a 2 cm sheet under the tires, and
//! cells that leave the window are parked so the volume comes back.
//! Slow exchange runs every few frames. Spray is a volume handed to the
//! particle system, not a third grid.
//!
//! Terramechanics, support, and μ do not live here. Dust trail and the
//! solver LOD call this module. They do not keep a second solver.

use std::cell::RefCell;
use std::collections::HashMap;

/// Coarse cell, metres. About 20 cm.
pub const COARSE_CELL_M: f32 = 0.20;
/// Cells on a side of the moving window. 200 × 0.20 m = 40 m, inside 30–50 m.
pub const WINDOW_RES: usize = 200;
/// Fine cell under the tires, metres.
pub const FINE_CELL_M: f32 = 0.02;
/// Fine patch on a side. 160 × 0.02 m = 3.2 m, the axle and the track.
pub const FINE_RES: usize = 160;
/// Air plan. Columns, not a solid cube.
pub const AIR_RES: usize = 128;
/// Layers in each dust column. The bottom layer is the one that can settle.
pub const AIR_LAYERS: usize = 8;
/// Fraction of the free-stream headwind that remains on the centerline.
/// The crosswind is not sheltered. Past the lee the plume is in the free stream.
pub const DUST_STREAM_HOLD: f32 = 0.66;
/// Half-width of that lee, metres.
pub const DUST_LEE_HALF_M: f32 = 1.52;

/// 1 in the free stream, [`DUST_STREAM_HOLD`] on the centerline.
pub fn dust_stream_hold(local_x: f32) -> f32 {
    let t = (local_x.abs() / DUST_LEE_HALF_M).min(1.0);
    let lee = 1.0 - t * t;
    1.0 - (1.0 - DUST_STREAM_HOLD) * lee
}
/// Slow exchange (settle, park compact) runs on this stride, inside 4–8 frames.
pub const SLOW_STRIDE: u32 = 6;
/// Settled dust does not lift at or above this pore moisture.
pub const DUST_WET_MOISTURE: f32 = 0.55;
/// Settled dust does not lift at or above this standing water, metres.
pub const DUST_WET_WATER_M: f32 = 0.008;

const GAUSS: [f32; 5] = [0.06, 0.24, 0.40, 0.24, 0.06];
const STEEP: [f32; 5] = [0.02, 0.08, 0.80, 0.08, 0.02];
const GRAVITY: f32 = 9.81;

/// What the solids in one cell are. Water with no solids uses [`Grain::Water`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grain {
    Clay = 0,
    Silt = 1,
    Sand = 2,
    Water = 3,
    Other = 4,
}

impl Grain {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Silt,
            2 => Self::Sand,
            3 => Self::Water,
            4 => Self::Other,
            _ => Self::Clay,
        }
    }
}

/// Dynamic viscosity, Pa·s. Clay stays thick. Silt sits under it. Sand does
/// not slurry like clay. A thin mix is the fluid end of the same curve.
pub fn dynamic_viscosity(grain: Grain, moisture: f32) -> f32 {
    let x = ((moisture.clamp(0.0, 1.0) - 0.22) / 0.70).clamp(0.0, 1.0);
    let thin = x * x * (3.0 - 2.0 * x);
    let (stiff, fluid) = match grain {
        Grain::Clay => (420.0, 0.35),
        Grain::Silt => (220.0, 4.0),
        Grain::Sand => (28.0, 0.55),
        Grain::Water => (0.9, 0.9),
        Grain::Other => (1.5, 0.08),
    };
    stiff + (fluid - stiff) * thin
}

/// How hard a free-surface slope accelerates the sheet. A very thin mix,
/// under about 12 Pa·s, moves more freely. Clay and silt do not.
pub fn sheet_mobility_for(nu: f32) -> f32 {
    let boost = if nu < 12.0 {
        let t = ((12.0 - nu.max(0.0)) / 12.0).clamp(0.0, 1.0);
        t * t * 1.6
    } else {
        0.0
    };
    0.042 * (1.0 + boost)
}

/// Per-second damping. Clay and silt stay in one band. Below that, a thin
/// mix sheds damping quickly and pools.
pub fn sheet_damp(nu: f32) -> f32 {
    let rel = (nu.max(0.05) / 231.0).clamp(0.002, 12.0);
    if rel >= 0.42 {
        return 1.70 * rel.powf(0.30);
    }
    let at_silt = 1.70 * 0.42_f32.powf(0.30);
    let t = (rel / 0.42).clamp(0.0, 1.0);
    at_silt * t.powf(1.15)
}

/// True when slip may loft dust. Wet pore moisture and a standing film may not.
pub fn dust_may_lift(moisture: f32, water_m: f32) -> bool {
    moisture < DUST_WET_MOISTURE && water_m < DUST_WET_WATER_M
}

/// What drives a sheet for one step.
#[derive(Clone, Copy, Debug)]
pub enum SheetDrive {
    /// Down the free surface.
    Gravity { vorticity: f32 },
    /// Toward a horizontal wind. Used by the dust layers.
    Wind { x: f32, z: f32, vorticity: f32 },
}

/// Impermeable footprint of a wheel or a belly panel in the water grid.
///
/// The sheet treats every cell inside this box, and below the free surface,
/// as a wall. The head, the downstream hole, and the wake are whatever the
/// height–velocity step does against that wall. This is not a wave generator.
#[derive(Clone, Copy, Debug)]
pub struct HydroSolid {
    pub x: f32,
    pub z: f32,
    /// Yaw of the long axis, radians. 0 points the long axis along +Z.
    pub yaw: f32,
    pub half_length: f32,
    pub half_width: f32,
    /// World Y of the underside. A cell is a wall only when the free surface
    /// is above this.
    pub bottom_y: f32,
    pub top_y: f32,
    /// Tire contact. Scour reads the shear on the fluid cells touching these.
    pub contact: bool,
    /// Cylinder radius, metres. Zero is a box (belly), and buoyancy is the
    /// submerged box. A positive radius uses the submerged circular segment.
    pub radius: f32,
    /// Cylinder width, metres. Unused for a box.
    pub width: f32,
}

/// Pressure force and displaced weight read back from the sheet after a step.
#[derive(Clone, Copy, Debug, Default)]
pub struct HydroLoad {
    pub force_x: f32,
    pub force_z: f32,
    pub buoyancy_n: f32,
    pub hull_buoyancy_n: f32,
    pub wheel_buoyancy_n: f32,
    pub displaced_m3: f32,
    pub eroded_m3: f32,
    pub hole_m: f32,
    pub wheel_blocks: u32,
    pub belly_blocks: u32,
}

/// Periodic lips of an open channel. Water that piles at the downstream lip
/// is averaged with the upstream lip so a long reach can keep a current.
/// Off unless the caller sets one. Wheels and the belly do not use this.
#[derive(Clone, Copy, Debug)]
pub struct StreamCycle {
    pub x0: f32,
    pub x1: f32,
    pub z0: f32,
    pub z1: f32,
    pub dir_x: f32,
    pub dir_z: f32,
}

/// One tire telling the sheet which way it is shoving.
#[derive(Clone, Copy, Debug)]
pub struct TirePush {
    pub x: f32,
    pub z: f32,
    pub forward_x: f32,
    pub forward_z: f32,
    pub radius: f32,
    pub width: f32,
    pub speed: f32,
    pub slip: f32,
}

/// Volume that left a sheet and must be spawned as particles by the caller.
#[derive(Clone, Copy, Debug)]
pub struct SprayEvent {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub volume: f32,
    /// 0 spray drop, 1 ribbon. The mesh is the particle system's, not a sheet.
    pub kind: u8,
}

/// A water drop the sheet has already peeled off. It is not a picture particle.
/// The render contract reads these while they are still in the air.
#[derive(Clone, Copy, Debug)]
pub struct LiveSpray {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    /// Cubic metres peeled from the water sheet.
    pub volume: f32,
    /// 0 is a water drop.
    pub kind: u8,
    /// Seconds since the sheet let it go.
    pub age: f32,
}

const SPRAY_LIFE_S: f32 = 1.1;

/// A request to move soil into the air column. The returned cubic metres are
/// what actually entered. The caller takes that out of the ground.
#[derive(Clone, Copy, Debug)]
pub struct DustInject {
    pub x: f32,
    pub z: f32,
    pub slip: f32,
    pub moisture: f32,
    pub water_m: f32,
    /// Cubic metres per second offered by a dry spinning tire.
    pub offer_m3_s: f32,
}

/// One visible sample of the air column or the settled film. Not a volume.
/// The cubic metres stay on the grid. Callers draw a mark and discard it.
#[derive(Clone, Copy, Debug)]
pub struct DustMark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub u: f32,
    pub v: f32,
    /// Sitting on the ground. A lofted layer is false.
    pub settled: bool,
}

/// Ground state at one point. Heights are metres of column, not a second mesh.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroundSample {
    pub water_m: f32,
    pub slurry_m: f32,
    pub u: f32,
    pub v: f32,
    pub settled_dust_m: f32,
    pub moisture: f32,
    pub bed_m: f32,
}

/// One coarse-window cell. Heights are metres. `u` and `v` are metres per second.
#[derive(Clone, Copy, Debug)]
pub struct CoarseCell {
    pub x: f32,
    pub z: f32,
    pub bed_m: f32,
    pub water_m: f32,
    pub slurry_m: f32,
    pub moisture: f32,
    pub u: f32,
    pub v: f32,
    pub settled_dust_m: f32,
}

/// Maps a beauty pass can read. Coarse window, after the fine patch is put back.
#[derive(Clone, Debug)]
pub struct SurfaceMaps {
    pub res: usize,
    pub cell: f32,
    pub origin: [f32; 2],
    pub height: Vec<f32>,
    pub vel_u: Vec<f32>,
    pub vel_v: Vec<f32>,
    pub moisture: Vec<f32>,
    pub dust: Vec<f32>,
}

/// Closed account inside this module. Spray is volume already given to particles.
#[derive(Clone, Copy, Debug, Default)]
pub struct FluidLedger {
    pub water_m3: f32,
    pub slurry_m3: f32,
    pub dust_air_m3: f32,
    pub dust_settled_m3: f32,
    pub frozen_water_m3: f32,
    pub frozen_slurry_m3: f32,
    pub frozen_dust_m3: f32,
    pub spray_m3: f32,
    pub total_m3: f32,
}

/// How a rounded rut in one mix levels.
#[derive(Clone, Copy, Debug)]
pub struct RutHeal {
    pub level_s: f32,
    pub relief0: f32,
    pub relief_8s: f32,
    pub still_deep_at_8s: bool,
}

struct Scratch {
    n: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    e: Vec<f32>,
    removed: Vec<f32>,
    added: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            n: 0,
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            e: Vec::new(),
            removed: Vec::new(),
            added: Vec::new(),
        }
    }

    fn fit(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.n = n;
        self.a.resize(n, 0.0);
        self.b.resize(n, 0.0);
        self.c.resize(n, 0.0);
        self.d.resize(n, 0.0);
        self.e.resize(n, 0.0);
        self.removed.resize(n, 0.0);
        self.added.resize(n, 0.0);
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

/// Step one height–velocity sheet. `bed` is the solid under the film, metres.
/// `gate` is 1 where a dry neighbour may receive fluid. Mass only moves in the
/// upwind flux, so the filter cannot mint or delete a column.
pub fn step_sheet(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
) {
    step_sheet_eps(h, u, v, bed, nu, gate, res, cell, dt, drive, 1.0e-6, None);
}

fn step_sheet_eps(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    solid: Option<&[u8]>,
) {
    if dt <= 0.0 || res == 0 || h.len() < res * res {
        return;
    }
    let mut sum = 0.0f32;
    let mut any = false;
    for s in h.iter() {
        if *s > eps {
            any = true;
            sum += *s;
        }
    }
    if !any || sum <= eps {
        return;
    }
    SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        sc.fit(res * res);
        step_sheet_scratch(
            h, u, v, bed, nu, gate, res, cell, dt, drive, eps, solid, &mut sc,
        );
    });
}

/// Advect one vehicle-frame dust volume, `x + layer * 128 + z * 128 * 8`.
///
/// The dust caller owns the cubic metres. This is the shared wind step:
/// the same upwind flux, pressure filter, and vorticity confinement as the
/// ground sheets. Mass that stays in `vol` is still airborne. The return is
/// the cubic metres that left the window, which the caller books as settled.
pub fn advect_dust_density(
    vol: &mut [f32],
    wind_x: f32,
    wind_z: f32,
    speed: f32,
    forward: glam::Vec3,
    right: glam::Vec3,
    dt: f32,
) -> f32 {
    const RES: usize = 128;
    const LAYERS: usize = 8;
    if dt <= 0.0 || vol.len() < RES * LAYERS * RES {
        return 0.0;
    }
    let forward = if forward.length_squared() > 1.0e-8 {
        forward.normalize()
    } else {
        glam::Vec3::Z
    };
    let right = if right.length_squared() > 1.0e-8 {
        right.normalize()
    } else {
        glam::Vec3::X
    };
    let rel = glam::Vec3::new(wind_x, 0.0, wind_z) - forward * speed.max(0.0);
    let gu = rel.dot(right);
    let gv = rel.dot(forward);
    let cell = 16.0 / RES as f32;
    let n = RES * RES;
    let mut h = vec![0.0f32; n];
    let mut u = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    let bed = vec![0.0f32; n];
    let nu = vec![0.4f32; n];
    let gate = vec![1u8; n];
    for layer in 0..LAYERS {
        let mut before = 0.0f64;
        for iz in 0..RES {
            for ix in 0..RES {
                let src = ix + layer * RES + iz * RES * LAYERS;
                let cell_v = vol[src];
                let local_x = -8.0 + (ix as f32 + 0.5) * cell;
                h[iz * RES + ix] = cell_v;
                u[iz * RES + ix] = gu;
                v[iz * RES + ix] = gv * dust_stream_hold(local_x);
                before += cell_v as f64;
            }
        }
        if before <= 0.0 {
            continue;
        }
        step_sheet_eps(
            &mut h,
            &mut u,
            &mut v,
            &bed,
            &nu,
            &gate,
            RES,
            cell,
            dt,
            SheetDrive::Wind {
                x: gu,
                z: gv,
                vorticity: 0.35,
            },
            0.0,
            None,
        );
        let after = h.iter().map(|c| *c as f64).sum::<f64>();
        let scale = if after > 0.0 {
            (before / after) as f32
        } else {
            1.0
        };
        for iz in 0..RES {
            for ix in 0..RES {
                let src = ix + layer * RES + iz * RES * LAYERS;
                vol[src] = h[iz * RES + ix] * scale;
            }
        }
    }
    // The caller books a positive return as settled without a ground cell.
    // Mass stays in `vol` and their slow stride settles it onto the map.
    0.0
}

fn step_sheet_scratch(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    solid: Option<&[u8]>,
    sc: &mut Scratch,
) {
    let (mut i0, mut i1, mut j0, mut j1) = (res, 0usize, res, 0usize);
    for j in 0..res {
        for i in 0..res {
            if h[j * res + i] > eps {
                i0 = i0.min(i);
                i1 = i1.max(i);
                j0 = j0.min(j);
                j1 = j1.max(j);
            }
        }
    }
    if i0 > i1 {
        return;
    }
    i0 = i0.saturating_sub(3);
    j0 = j0.saturating_sub(3);
    i1 = (i1 + 3).min(res - 1);
    j1 = (j1 + 3).min(res - 1);

    let mut max_s = 0.15f32;
    if let SheetDrive::Wind { x, z, .. } = drive {
        max_s = max_s.max(x.hypot(z));
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            max_s = max_s.max(u[idx].hypot(v[idx]));
        }
    }
    let nsub = ((dt * max_s) / (0.42 * cell.max(1.0e-3)))
        .ceil()
        .clamp(1.0, 8.0) as u32;
    let dts = dt / nsub as f32;
    for _ in 0..nsub {
        substep(
            h, u, v, bed, nu, gate, res, cell, dts, drive, eps, i0, i1, j0, j1, solid, sc,
        );
    }
}

fn substep(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    solid: Option<&[u8]>,
    sc: &mut Scratch,
) {
    let n = res * res;
    for idx in 0..n {
        sc.a[idx] = bed[idx] + h[idx];
    }
    // A wall is not a hole in the free surface. Copy the neighbouring water
    // line onto solid cells so the pressure gradient does not fall into the body.
    if let Some(mask) = solid {
        lift_solid_surface(&mut sc.a, h, mask, res, i0, i1, j0, j1, eps, &mut sc.e);
    }
    blur_box(&sc.a, &mut sc.b, &mut sc.e, res, i0, i1, j0, j1, &GAUSS);
    blur_box(&sc.a, &mut sc.c, &mut sc.e, res, i0, i1, j0, j1, &STEEP);
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            let filtered = 0.62 * sc.b[idx] + 0.38 * sc.c[idx];
            // The filter is the pressure. A share of the raw surface stays in
            // the force so a broad rut still runs until the free surface is flat.
            sc.d[idx] = 0.45 * filtered + 0.55 * sc.a[idx];
        }
    }
    let p = &sc.d;
    let eta = &sc.a;
    if let Some(mask) = solid {
        write_noslip_ghost(u, v, h, mask, res, i0, i1, j0, j1, eps);
    }
    match drive {
        SheetDrive::Gravity { vorticity } => {
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let idx = j * res + i;
                    if h[idx] <= eps {
                        continue;
                    }
                    let (px, pz) = grad(p, res, i, j, cell);
                    // Mud films keep their reduced mobility. A rigid body is a
                    // wall, so the water against it feels the full free-surface
                    // gradient and can pass around the geometry.
                    let mut mob = sheet_mobility_for(nu[idx]);
                    if let Some(mask) = solid {
                        // Thin films on the bank already sit on a steep bed. Full
                        // gravity there runs them up to the speed cap and back.
                        if h[idx] > 0.08 && solid_near(mask, res, i, j, 1) {
                            mob = mob.max(1.0);
                        }
                    }
                    u[idx] += -GRAVITY * px * mob * dt;
                    v[idx] += -GRAVITY * pz * mob * dt;
                    if vorticity > 0.0 {
                        confine_one(u, v, res, i, j, cell, dt, vorticity);
                    }
                }
            }
        }
        SheetDrive::Wind { x, z, vorticity } => {
            let blend = 1.0 - (-dt * 1.6).exp();
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let idx = j * res + i;
                    if h[idx] <= eps {
                        continue;
                    }
                    u[idx] += (x - u[idx]) * blend;
                    v[idx] += (z - v[idx]) * blend;
                    if vorticity > 0.0 {
                        confine_one(u, v, res, i, j, cell, dt, vorticity);
                    }
                }
            }
        }
    }

    // Viscosity overloads the divergence the projection removes. A thick mix
    // asks the filter for a stronger correction, so it resists spreading.
    // The gain stays small: this sheet is a free surface, and the slope force
    // above is what moves the column.
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            let (ux, vz) = grad_vel(u, v, res, i, j, cell);
            let div = ux + vz;
            let lap_h = lap(eta, res, i, j, cell);
            let overload = 1.0 + (nu[idx] / 231.0).clamp(0.0, 6.0);
            sc.b[idx] = div * overload - 0.015 * overload * lap_h;
        }
    }
    blur_box(&sc.b, &mut sc.c, &mut sc.e, res, i0, i1, j0, j1, &GAUSS);
    blur_box(&sc.b, &mut sc.a, &mut sc.e, res, i0, i1, j0, j1, &STEEP);
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            sc.d[idx] = 0.62 * sc.c[idx] + 0.38 * sc.a[idx];
        }
    }
    let q = &sc.d;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            // q has the sign of the divergence. Adding its gradient undoes
            // that divergence. Subtracting it was a positive feedback.
            let (qx, qz) = grad(q, res, i, j, cell);
            u[idx] += 0.012 * dt * qx;
            v[idx] += 0.012 * dt * qz;
            let decay = (-sheet_damp(nu[idx]) * dt).exp();
            let lim = match drive {
                SheetDrive::Wind { x, z, .. } => x.hypot(z).max(1.6),
                SheetDrive::Gravity { .. } => 1.6,
            };
            u[idx] = finite_speed(u[idx] * decay, lim);
            v[idx] = finite_speed(v[idx] * decay, lim);
        }
    }
    if let Some(mask) = solid {
        // Water that is aimed at the body goes around it. Leaving it in the
        // upstream cell is what builds a reflected bore instead of a bow.
        redirect_into_walls(h, u, v, mask, bed, res, cell, dt, i0, i1, j0, j1, sc);
        // No penetration: drop the velocity that points into the wall.
        // Tangential flow stays, which is what shears the wake.
        block_into_solid(u, v, h, mask, res, i0, i1, j0, j1, eps);
        write_noslip_ghost(u, v, h, mask, res, i0, i1, j0, j1, eps);
    }
    let banks = if solid.is_some() { Some(bed) } else { None };
    transport(
        h, u, v, gate, res, cell, dt, i0, i1, j0, j1, true, solid, banks, sc,
    );
    transport(
        h, u, v, gate, res, cell, dt, i0, i1, j0, j1, false, solid, banks, sc,
    );
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if eps > 0.0 && h[idx] < eps {
                h[idx] = 0.0;
                u[idx] = 0.0;
                v[idx] = 0.0;
            }
        }
    }
}

fn confine_one(
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    i: usize,
    j: usize,
    cell: f32,
    dt: f32,
    eps: f32,
) {
    let idx = j * res + i;
    let w = vorticity_at(u, v, res, i, j, cell);
    if !w.is_finite() || w.abs() < 1.0e-4 {
        return;
    }
    let wxp = vorticity_at(u, v, res, (i + 1).min(res - 1), j, cell).abs();
    let wxm = vorticity_at(u, v, res, i.saturating_sub(1), j, cell).abs();
    let wzp = vorticity_at(u, v, res, i, (j + 1).min(res - 1), cell).abs();
    let wzm = vorticity_at(u, v, res, i, j.saturating_sub(1), cell).abs();
    let nx = (wxp - wxm) / (2.0 * cell);
    let nz = (wzp - wzm) / (2.0 * cell);
    // A 2 cm cell under a tire makes |w| enormous. Skipping that kick keeps
    // the sheet velocity finite. A small confinement still applies.
    let du = eps * dt * (-nz * w);
    let dv = eps * dt * (nx * w);
    if du.is_finite() && dv.is_finite() && du.hypot(dv) <= 0.25 {
        u[idx] += du;
        v[idx] += dv;
    }
}

fn finite_speed(v: f32, lim: f32) -> f32 {
    if !v.is_finite() {
        0.0
    } else {
        v.clamp(-lim, lim)
    }
}

fn finite0(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn vorticity_at(u: &[f32], v: &[f32], res: usize, i: usize, j: usize, cell: f32) -> f32 {
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let dvdx = (v[j * res + ip] - v[j * res + im]) / (2.0 * cell);
    let dudz = (u[jp * res + i] - u[jm * res + i]) / (2.0 * cell);
    dvdx - dudz
}

fn grad(p: &[f32], res: usize, i: usize, j: usize, cell: f32) -> (f32, f32) {
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let dx = (p[j * res + ip] - p[j * res + im]) / (2.0 * cell);
    let dz = (p[jp * res + i] - p[jm * res + i]) / (2.0 * cell);
    (dx, dz)
}

fn grad_vel(u: &[f32], v: &[f32], res: usize, i: usize, j: usize, cell: f32) -> (f32, f32) {
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let dux = (u[j * res + ip] - u[j * res + im]) / (2.0 * cell);
    let dvz = (v[jp * res + i] - v[jm * res + i]) / (2.0 * cell);
    (dux, dvz)
}

fn lap(p: &[f32], res: usize, i: usize, j: usize, cell: f32) -> f32 {
    let idx = j * res + i;
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let s = p[j * res + ip] + p[j * res + im] + p[jp * res + i] + p[jm * res + i] - 4.0 * p[idx];
    s / (cell * cell)
}

fn blur_box(
    src: &[f32],
    dst: &mut [f32],
    tmp: &mut [f32],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k: &[f32; 5],
) {
    let r = res as i32;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let mut a = 0.0f32;
            for t in -2i32..=2 {
                let ii = (i as i32 + t).clamp(0, r - 1) as usize;
                a += src[j * res + ii] * k[(t + 2) as usize];
            }
            tmp[j * res + i] = a;
        }
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let mut a = 0.0f32;
            for t in -2i32..=2 {
                let jj = (j as i32 + t).clamp(0, r - 1) as usize;
                a += tmp[jj * res + i] * k[(t + 2) as usize];
            }
            dst[j * res + i] = a;
        }
    }
}

fn transport(
    h: &mut [f32],
    u: &[f32],
    v: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    solid: Option<&[u8]>,
    bed: Option<&[f32]>,
    sc: &mut Scratch,
) {
    let n = res * res;
    for idx in 0..n {
        sc.removed[idx] = 0.0;
        sc.added[idx] = 0.0;
    }
    if along_x {
        for j in j0..=j1 {
            for i in i0..i1 {
                let a = j * res + i;
                let b = a + 1;
                move_edge(
                    h,
                    u,
                    gate,
                    a,
                    b,
                    0.5 * (u[a] + u[b]),
                    cell,
                    dt,
                    solid,
                    bed,
                    sc,
                );
            }
        }
    } else {
        for j in j0..j1 {
            for i in i0..=i1 {
                let a = j * res + i;
                let b = a + res;
                move_edge(
                    h,
                    v,
                    gate,
                    a,
                    b,
                    0.5 * (v[a] + v[b]),
                    cell,
                    dt,
                    solid,
                    bed,
                    sc,
                );
            }
        }
    }
    for idx in 0..n {
        if sc.removed[idx] > 0.0 || sc.added[idx] > 0.0 {
            h[idx] = (h[idx] - sc.removed[idx] + sc.added[idx]).max(0.0);
        }
    }
}

fn move_edge(
    h: &[f32],
    _speed: &[f32],
    gate: &[u8],
    a: usize,
    b: usize,
    vel: f32,
    cell: f32,
    dt: f32,
    solid: Option<&[u8]>,
    bed: Option<&[f32]>,
    sc: &mut Scratch,
) {
    if let Some(solid) = solid {
        if solid[a] != 0 || solid[b] != 0 {
            return;
        }
    }
    if vel.abs() < 1.0e-5 {
        return;
    }
    let (donor, recv) = if vel > 0.0 { (a, b) } else { (b, a) };
    // A dry bank above the donor's free surface does not take water.
    // Only the solid path passes a bed; a bare sheet spreads as before.
    if let Some(bed) = bed {
        if h[recv] <= 1.0e-6 && bed[recv] > bed[donor] + h[donor] + 1.0e-3 {
            return;
        }
    }
    let room = h[donor] - sc.removed[donor];
    if room <= 1.0e-8 {
        return;
    }
    if gate[recv] == 0 && h[recv] <= 1.0e-6 {
        return;
    }
    let frac = (vel.abs() * dt / cell).min(0.45);
    let dh = (h[donor] * frac).min(room);
    if dh <= 1.0e-8 {
        return;
    }
    sc.removed[donor] += dh;
    sc.added[recv] += dh;
}

/// Free-surface stand-in for a wall. Each solid cell takes the mean of the
/// wet cells within two cells, so the body is not a hole in η.
fn lift_solid_surface(
    eta: &mut [f32],
    h: &[f32],
    solid: &[u8],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    eps: f32,
    tmp: &mut [f32],
) {
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] == 0 {
                continue;
            }
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for dj in -2i32..=2 {
                for di in -2i32..=2 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                        continue;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    if solid[nidx] != 0 || h[nidx] <= eps {
                        continue;
                    }
                    acc += eta[nidx];
                    n += 1.0;
                }
            }
            tmp[idx] = if n > 0.0 { acc / n } else { eta[idx] };
        }
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] != 0 {
                eta[idx] = tmp[idx];
            }
        }
    }
}

/// No-slip ghost: the wall velocity is the negation of the neighbouring fluid.
/// Vorticity on the fluid side is then the shear of that wall, not a drawn vortex.
fn write_noslip_ghost(
    u: &mut [f32],
    v: &mut [f32],
    h: &[f32],
    solid: &[u8],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    eps: f32,
) {
    let mut ghosts = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] == 0 {
                continue;
            }
            let mut au = 0.0f32;
            let mut av = 0.0f32;
            let mut n = 0.0f32;
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] != 0 || h[nidx] <= eps {
                    continue;
                }
                au += u[nidx];
                av += v[nidx];
                n += 1.0;
            }
            if n > 0.0 {
                ghosts.push((idx, -au / n, -av / n));
            } else {
                ghosts.push((idx, 0.0, 0.0));
            }
        }
    }
    for (idx, gu, gv) in ghosts {
        u[idx] = gu;
        v[idx] = gv;
    }
}

/// Remove the into-wall part of the fluid velocity. The tangential part stays.
fn block_into_solid(
    u: &mut [f32],
    v: &mut [f32],
    h: &[f32],
    solid: &[u8],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    eps: f32,
) {
    let mut next = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] != 0 || h[idx] <= eps {
                continue;
            }
            let mut ux = u[idx];
            let mut vx = v[idx];
            let mut hit = false;
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] == 0 {
                    continue;
                }
                let nx = di as f32;
                let nz = dj as f32;
                let vn = ux * nx + vx * nz;
                if vn > 0.0 {
                    ux -= vn * nx;
                    vx -= vn * nz;
                    hit = true;
                }
            }
            if hit {
                next.push((idx, ux, vx));
            }
        }
    }
    for (idx, ux, vx) in next {
        u[idx] = ux;
        v[idx] = vx;
    }
}

/// Push water (or slurry) out of cells the body now occupies. The volume is
/// split across the open neighbours. A cell with nowhere to go keeps its column.
fn displace_occupied(h: &mut [f32], u: &mut [f32], v: &mut [f32], solid: &[u8], res: usize) {
    let n = res * res;
    let mut give = vec![0.0f32; n];
    for j in 0..res {
        for i in 0..res {
            let idx = j * res + i;
            if solid[idx] == 0 {
                continue;
            }
            let vol = h[idx];
            if vol <= 1.0e-8 {
                h[idx] = 0.0;
                u[idx] = 0.0;
                v[idx] = 0.0;
                continue;
            }
            let mut neigh = Vec::with_capacity(4);
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] == 0 {
                    neigh.push(nidx);
                }
            }
            if neigh.is_empty() {
                for dj in -2i32..=2 {
                    for di in -2i32..=2 {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                            continue;
                        }
                        let nidx = nj as usize * res + ni as usize;
                        if solid[nidx] == 0 {
                            neigh.push(nidx);
                        }
                    }
                }
            }
            if neigh.is_empty() {
                continue;
            }
            h[idx] = 0.0;
            u[idx] = 0.0;
            v[idx] = 0.0;
            let share = vol / neigh.len() as f32;
            for nidx in neigh {
                give[nidx] += share;
            }
        }
    }
    for idx in 0..n {
        if give[idx] > 0.0 {
            h[idx] += give[idx];
        }
    }
}

/// Move the discharge that points into a wall out to the open flanks.
/// The wall stays dry. The volume stays on the sheet.
fn redirect_into_walls(
    h: &mut [f32],
    u: &[f32],
    v: &[f32],
    solid: &[u8],
    bed: &[f32],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    sc: &mut Scratch,
) {
    let n = res * res;
    for idx in 0..n {
        sc.removed[idx] = 0.0;
        sc.added[idx] = 0.0;
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] != 0 || h[idx] < 0.02 {
                continue;
            }
            for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] == 0 {
                    continue;
                }
                // Own speed into the wall, or the stream feeding this cell if the
                // wall has already stopped it. Either way the discharge turns aside.
                let mut vn = u[idx] * di as f32 + v[idx] * dj as f32;
                let ui = i as i32 - di;
                let uj = j as i32 - dj;
                if ui >= 0 && uj >= 0 && ui < res as i32 && uj < res as i32 {
                    let uidx = uj as usize * res + ui as usize;
                    if solid[uidx] == 0 {
                        let feed = u[uidx] * di as f32 + v[uidx] * dj as f32;
                        if feed > vn {
                            vn = feed;
                        }
                    }
                }
                if vn < 0.05 {
                    continue;
                }
                let perps: [(i32, i32); 2] = if di != 0 {
                    [(0, -1), (0, 1)]
                } else {
                    [(-1, 0), (1, 0)]
                };
                let mut dest = [0usize; 2];
                let mut n_dest = 0usize;
                let mut flank = 0.0f32;
                for (pi, pj) in perps {
                    let qi = i as i32 + pi;
                    let qj = j as i32 + pj;
                    if qi < 0 || qj < 0 || qi >= res as i32 || qj >= res as i32 {
                        continue;
                    }
                    let q = qj as usize * res + qi as usize;
                    if solid[q] != 0 {
                        continue;
                    }
                    if h[q] <= 1.0e-6 && bed[q] > bed[idx] + h[idx] + 1.0e-3 {
                        continue;
                    }
                    dest[n_dest] = q;
                    flank += h[q];
                    n_dest += 1;
                }
                if n_dest == 0 {
                    continue;
                }
                // Only the water standing above the path around the body turns
                // aside. The rest is the bow.
                let excess = h[idx] - flank / n_dest as f32;
                if excess < 2.0e-3 {
                    continue;
                }
                let room = h[idx] - sc.removed[idx];
                if room < 1.0e-6 {
                    continue;
                }
                let frac = (vn * dt / cell).min(0.45);
                let dh = excess.min(h[idx] * frac).min(room);
                if dh < 1.0e-8 {
                    continue;
                }
                sc.removed[idx] += dh;
                let share = dh / n_dest as f32;
                for q in dest.iter().take(n_dest) {
                    sc.added[*q] += share;
                }
            }
        }
    }
    for idx in 0..n {
        if sc.removed[idx] > 0.0 || sc.added[idx] > 0.0 {
            h[idx] = (h[idx] - sc.removed[idx] + sc.added[idx]).max(0.0);
        }
    }
}

fn solid_near(mask: &[u8], res: usize, i: usize, j: usize, radius: i32) -> bool {
    for dj in -radius..=radius {
        for di in -radius..=radius {
            if di == 0 && dj == 0 {
                continue;
            }
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                continue;
            }
            if mask[nj as usize * res + ni as usize] != 0 {
                return true;
            }
        }
    }
    false
}

fn solid_contains(solid: &HydroSolid, x: f32, z: f32) -> bool {
    let dx = x - solid.x;
    let dz = z - solid.z;
    let fwd_x = solid.yaw.sin();
    let fwd_z = solid.yaw.cos();
    let right_x = fwd_z;
    let right_z = -fwd_x;
    let along = dx * fwd_x + dz * fwd_z;
    let lat = dx * right_x + dz * right_z;
    along.abs() <= solid.half_length.max(0.0) && lat.abs() <= solid.half_width.max(0.0)
}

fn solid_overlaps(solid: &HydroSolid, x: f32, z: f32, eta: f32, bed: f32) -> bool {
    if solid.half_length <= 1.0e-4 || solid.half_width <= 1.0e-4 {
        return false;
    }
    if !solid_contains(solid, x, z) {
        return false;
    }
    eta > solid.bottom_y + 1.0e-4 && bed < solid.top_y
}

fn cylinder_segment_volume(depth: f32, radius: f32, width: f32) -> f32 {
    let d = depth.clamp(0.0, radius * 2.0);
    if d <= 1.0e-4 || width <= 0.0 || radius <= 1.0e-3 {
        return 0.0;
    }
    let c = ((radius - d) / radius).clamp(-1.0, 1.0);
    let area =
        radius * radius * c.acos() - (radius - d) * (2.0 * radius * d - d * d).max(0.0).sqrt();
    area.max(0.0) * width
}

fn column_sum(h: &[f32], cell: f32) -> f32 {
    let area = cell * cell;
    h.iter().map(|s| s.max(0.0)).sum::<f32>() * area
}

/// A rounded pool. `level_s` is when the relief has fallen under 28% of the
/// start. Clay and silt stay deep through 8 s and level between 10 and 15 s.
/// A thin clay mix levels between 1 and 3 s.
pub fn rut_heal(grain: Grain, moisture: f32) -> RutHeal {
    let res = 48usize;
    let cell = 0.06f32;
    let n = res * res;
    let mut h = vec![0.0f32; n];
    let mut u = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    let mut bed = vec![0.0f32; n];
    let nu_v = dynamic_viscosity(grain, moisture);
    let nu = vec![nu_v; n];
    let gate = vec![1u8; n];
    let c = (res / 2) as f32;
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5 - c) * cell;
            let z = (j as f32 + 0.5 - c) * cell;
            let r = (x * x + z * z).sqrt();
            let idx = j * res + i;
            let bowl = (-(r * r) / (2.0 * 0.42 * 0.42)).exp();
            bed[idx] = -0.07 * bowl;
            h[idx] = 0.012 + 0.028 * (1.0 - bowl);
        }
    }
    let _ = nu_v;
    let relief0 = relief(&h, &bed, res, cell);
    let mut level_s = 20.0f32;
    let mut relief_8s = relief0;
    let mut t = 0.0f32;
    let dt = 0.05f32;
    while t < 20.0 - 1.0e-4 {
        step_sheet(
            &mut h,
            &mut u,
            &mut v,
            &bed,
            &nu,
            &gate,
            res,
            cell,
            dt,
            SheetDrive::Gravity { vorticity: 0.0 },
        );
        t += dt;
        let r = relief(&h, &bed, res, cell);
        if (t - 8.0).abs() <= dt * 0.51 {
            relief_8s = r;
        }
        if level_s > 19.0 && r < relief0 * 0.28 {
            level_s = t;
        }
        if t >= 8.0 && level_s < 19.0 {
            break;
        }
    }
    RutHeal {
        level_s,
        relief0,
        relief_8s,
        still_deep_at_8s: relief_8s > relief0 * 0.38,
    }
}

fn relief(h: &[f32], bed: &[f32], res: usize, cell: f32) -> f32 {
    let c = (res / 2) as f32;
    let mut center = 0.0f32;
    let mut cn = 0.0f32;
    let mut ring = 0.0f32;
    let mut rn = 0.0f32;
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5 - c) * cell;
            let z = (j as f32 + 0.5 - c) * cell;
            let r = (x * x + z * z).sqrt();
            let eta = bed[j * res + i] + h[j * res + i];
            if r < 0.12 {
                center += eta;
                cn += 1.0;
            } else if (0.48..0.68).contains(&r) {
                ring += eta;
                rn += 1.0;
            }
        }
    }
    if cn < 1.0 || rn < 1.0 {
        return 0.0;
    }
    (ring / rn) - (center / cn)
}

#[derive(Clone, Copy, Debug, Default)]
struct Parked {
    water: f32,
    slurry: f32,
    uw: f32,
    vw: f32,
    us: f32,
    vs: f32,
    dust: f32,
}

/// The moving window. Coarse sheet, fine sheet under the tires, dust column.
pub struct FluidCore {
    pub center: [f32; 2],
    origin: [f32; 2],
    air_origin: [f32; 2],
    water: Vec<f32>,
    slurry: Vec<f32>,
    uw: Vec<f32>,
    vw: Vec<f32>,
    us: Vec<f32>,
    vs: Vec<f32>,
    bed: Vec<f32>,
    moisture: Vec<f32>,
    grain: Vec<u8>,
    settled: Vec<f32>,
    fine_origin: [f32; 2],
    fine_water: Vec<f32>,
    fine_slurry: Vec<f32>,
    fine_uw: Vec<f32>,
    fine_vw: Vec<f32>,
    fine_us: Vec<f32>,
    fine_vs: Vec<f32>,
    fine_bed: Vec<f32>,
    fine_moisture: Vec<f32>,
    fine_grain: Vec<u8>,
    fine_on: bool,
    dust: Vec<f32>,
    dust_u: Vec<f32>,
    dust_v: Vec<f32>,
    parked: HashMap<(i32, i32), Parked>,
    parked_dust: HashMap<(i32, i32), f32>,
    spray_m3: f32,
    /// Drops peeled this pass that are still in the air.
    drops: Vec<LiveSpray>,
    /// Tire shoves from the step that just ran. Foam around the tires reads these.
    last_pushes: Vec<TirePush>,
    pub frame: u32,
    pushes: Vec<TirePush>,
    wind: [f32; 2],
    /// Wheel and belly walls. Empty means the sheet has no body in it.
    solids: Vec<HydroSolid>,
    /// Metres of bed removed by contact shear, per coarse cell.
    bed_drop: Vec<f32>,
    eroded_m3: f32,
    scour_hole_m: f32,
    cycle: Option<StreamCycle>,
}

impl FluidCore {
    pub fn new(center: [f32; 2]) -> Self {
        let n = WINDOW_RES * WINDOW_RES;
        let fine_n = FINE_RES * FINE_RES;
        let air_n = AIR_RES * AIR_RES * AIR_LAYERS;
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        let origin = snap_origin(center, extent, COARSE_CELL_M);
        let air_extent = AIR_RES as f32 * (extent / AIR_RES as f32);
        let air_origin = snap_origin(center, air_extent, extent / AIR_RES as f32);
        Self {
            center,
            origin,
            air_origin,
            water: vec![0.0; n],
            slurry: vec![0.0; n],
            uw: vec![0.0; n],
            vw: vec![0.0; n],
            us: vec![0.0; n],
            vs: vec![0.0; n],
            bed: vec![0.0; n],
            moisture: vec![0.0; n],
            grain: vec![Grain::Other as u8; n],
            settled: vec![0.0; n],
            fine_origin: [0.0, 0.0],
            fine_water: vec![0.0; fine_n],
            fine_slurry: vec![0.0; fine_n],
            fine_uw: vec![0.0; fine_n],
            fine_vw: vec![0.0; fine_n],
            fine_us: vec![0.0; fine_n],
            fine_vs: vec![0.0; fine_n],
            fine_bed: vec![0.0; fine_n],
            fine_moisture: vec![0.0; fine_n],
            fine_grain: vec![Grain::Other as u8; fine_n],
            fine_on: false,
            dust: vec![0.0; air_n],
            dust_u: vec![0.0; air_n],
            dust_v: vec![0.0; air_n],
            parked: HashMap::new(),
            parked_dust: HashMap::new(),
            spray_m3: 0.0,
            drops: Vec::new(),
            last_pushes: Vec::new(),
            frame: 0,
            pushes: Vec::new(),
            wind: [1.1, 0.35],
            solids: Vec::new(),
            bed_drop: vec![0.0; n],
            eroded_m3: 0.0,
            scour_hole_m: 0.0,
            cycle: None,
        }
    }

    /// Replace the walls for the next step. An empty slice is open water.
    pub fn set_hydro_solids(&mut self, solids: &[HydroSolid]) {
        self.solids.clear();
        self.solids.extend_from_slice(solids);
    }

    pub fn hydro_solids(&self) -> &[HydroSolid] {
        &self.solids
    }

    /// Open-channel lips. `None` leaves the sheet closed at its own banks.
    pub fn set_stream_cycle(&mut self, cycle: Option<StreamCycle>) {
        self.cycle = cycle;
    }

    /// Initial bed and water thickness. Returns the water volume now on the sheet.
    pub fn map_water_bed(&mut self, mut f: impl FnMut(f32, f32) -> (f32, f32)) -> f32 {
        self.restrict_fine();
        let mut vol = 0.0f32;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                let (bed, water) = f(x, z);
                let idx = j * WINDOW_RES + i;
                self.bed[idx] = if bed.is_finite() { bed } else { 0.0 };
                let w = if water.is_finite() {
                    water.max(0.0)
                } else {
                    0.0
                };
                self.water[idx] = w;
                if w > 0.0 {
                    self.moisture[idx] = self.moisture[idx].max(0.8);
                    if self.grain[idx] == Grain::Other as u8 {
                        self.grain[idx] = Grain::Silt as u8;
                    }
                }
                vol += w * area;
            }
        }
        vol
    }

    /// Initial horizontal velocity. Call once. Later steps are the sheet's.
    pub fn map_water_velocity(&mut self, mut f: impl FnMut(f32, f32, f32) -> (f32, f32)) {
        self.restrict_fine();
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let idx = j * WINDOW_RES + i;
                if self.water[idx] <= 1.0e-6 {
                    continue;
                }
                let (x, z) = self.coarse_xz(i, j);
                let (u, v) = f(x, z, self.water[idx]);
                self.uw[idx] = if u.is_finite() { u } else { 0.0 };
                self.vw[idx] = if v.is_finite() { v } else { 0.0 };
            }
        }
    }

    pub fn non_finite_count(&self) -> u32 {
        let mut n = 0u32;
        for arr in [
            &self.water,
            &self.slurry,
            &self.uw,
            &self.vw,
            &self.us,
            &self.vs,
            &self.bed,
            &self.moisture,
            &self.bed_drop,
        ] {
            for v in arr {
                if !v.is_finite() {
                    n += 1;
                }
            }
        }
        n
    }

    pub fn set_wind(&mut self, x: f32, z: f32) {
        self.wind = [x, z];
    }

    /// Move the window. Cells that slide off are parked. Cells that slide on
    /// are restored. The sum does not change.
    pub fn recenter(&mut self, center: [f32; 2]) {
        self.restrict_fine();
        self.center = center;
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        let new_origin = snap_origin(center, extent, COARSE_CELL_M);
        self.shift_coarse(new_origin);
        let air_cell = (WINDOW_RES as f32 * COARSE_CELL_M) / AIR_RES as f32;
        let air_extent = AIR_RES as f32 * air_cell;
        let new_air = snap_origin(center, air_extent, air_cell);
        self.shift_air(new_air, air_cell);
    }

    pub fn pour_water(&mut self, x0: f32, x1: f32, z0: f32, z1: f32, depth: f32) -> f32 {
        self.paint(x0, x1, z0, z1, depth, true, Grain::Water, 1.0)
    }

    pub fn pour_slurry(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        depth: f32,
        grain: Grain,
        moisture: f32,
    ) -> f32 {
        self.paint(x0, x1, z0, z1, depth, false, grain, moisture)
    }

    /// Remove slurry cubic metres at a world point. The caller holds that volume.
    pub fn take_slurry_at(&mut self, x: f32, z: f32, volume: f32) -> f32 {
        self.take_column(x, z, volume, false)
    }

    /// Remove water cubic metres at a world point. The caller holds that volume.
    pub fn take_water_at(&mut self, x: f32, z: f32, volume: f32) -> f32 {
        self.take_column(x, z, volume, true)
    }

    /// Put water cubic metres back onto the sheet. Used when an arch drips.
    pub fn add_water_at(&mut self, x: f32, z: f32, volume: f32) {
        if volume <= 0.0 {
            return;
        }
        if self.fine_on {
            self.restrict_fine();
        }
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return;
        };
        let idx = j * WINDOW_RES + i;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        self.water[idx] += volume / area;
    }

    fn take_column(&mut self, x: f32, z: f32, volume: f32, water: bool) -> f32 {
        if volume <= 0.0 {
            return 0.0;
        }
        if self.fine_on {
            self.restrict_fine();
        }
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return 0.0;
        };
        let idx = j * WINDOW_RES + i;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let col = if water {
            &mut self.water
        } else {
            &mut self.slurry
        };
        let have = col[idx] * area;
        let take = volume.min(have).max(0.0);
        if take <= 0.0 {
            return 0.0;
        }
        col[idx] -= take / area;
        take
    }

    fn paint(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        depth: f32,
        water: bool,
        grain: Grain,
        moisture: f32,
    ) -> f32 {
        let depth = depth.max(0.0);
        let mut vol = 0.0f32;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * WINDOW_RES + i;
                if water {
                    self.water[idx] = depth;
                } else {
                    self.slurry[idx] = depth;
                    self.grain[idx] = grain as u8;
                    self.moisture[idx] = moisture.clamp(0.0, 1.0);
                }
                vol += depth * area;
            }
        }
        vol
    }

    /// Multiply the coarse water and slurry by masks in 0..1. A pour can
    /// start as a box and leave as an oval or a pair of ruts.
    pub fn scale_columns(&mut self, mut mask: impl FnMut(f32, f32) -> (f32, f32)) {
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                let (mw, ms) = mask(x, z);
                let idx = j * WINDOW_RES + i;
                self.water[idx] *= mw.clamp(0.0, 1.0);
                self.slurry[idx] *= ms.clamp(0.0, 1.0);
            }
        }
    }

    /// Pull the rim of each sheet down to the bed over `feather_m`. The
    /// interior depth stays. A hard rectangle does not stay a wall of fluid.
    pub fn soften_rims(&mut self, feather_m: f32) {
        let feather = feather_m.max(COARSE_CELL_M);
        let rad = (feather / COARSE_CELL_M).ceil() as i32;
        let n = WINDOW_RES;
        let mut water = self.water.clone();
        let mut slurry = self.slurry.clone();
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                if self.water[idx] > 1.0e-6 {
                    water[idx] *= rim_scale(&self.water, i, j, rad, feather);
                }
                if self.slurry[idx] > 1.0e-6 {
                    slurry[idx] *= rim_scale(&self.slurry, i, j, rad, feather);
                }
            }
        }
        self.water = water;
        self.slurry = slurry;
    }

    /// Copy the solid under the film from the terramechanics field. Does not
    /// touch water, slurry, or dust volumes.
    pub fn read_substrate(&mut self, mut sample: impl FnMut(f32, f32) -> (f32, f32, Grain)) {
        self.restrict_fine();
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                let (bed, moisture, grain) = sample(x, z);
                let idx = j * WINDOW_RES + i;
                self.bed[idx] = bed;
                self.moisture[idx] = moisture.clamp(0.0, 1.0);
                if self.slurry[idx] > 1.0e-5 {
                    self.grain[idx] = grain as u8;
                }
            }
        }
    }

    pub fn push_tire(&mut self, push: TirePush) {
        self.pushes.push(push);
    }

    /// Cubic metres that entered the air. Zero on a wet cell.
    pub fn inject_dust(&mut self, inj: DustInject, dt: f32) -> f32 {
        if !dust_may_lift(inj.moisture, inj.water_m) || dt <= 0.0 || inj.offer_m3_s <= 0.0 {
            return 0.0;
        }
        let dryness = (1.0 - inj.moisture.clamp(0.0, 1.0)).max(0.0);
        let slip = inj.slip.abs().clamp(0.0, 1.5);
        let vol = inj.offer_m3_s * dt * dryness * (0.25 + slip);
        if vol <= 1.0e-8 {
            return 0.0;
        }
        let air_cell = self.air_cell();
        let Some((i, j)) = self.air_ij(inj.x, inj.z) else {
            return 0.0;
        };
        let layer = AIR_LAYERS / 2;
        let idx = (layer * AIR_RES + j) * AIR_RES + i;
        let area = air_cell * air_cell;
        self.dust[idx] += vol / area;
        let lift = 0.6 + slip;
        self.dust_u[idx] += self.wind[0] * 0.25;
        self.dust_v[idx] += self.wind[1] * 0.25 + lift * 0.05;
        vol
    }

    pub fn step(&mut self, dt: f32) -> Vec<SprayEvent> {
        let slow = self.frame % SLOW_STRIDE == 0;
        self.frame = self.frame.wrapping_add(1);
        let pushes = std::mem::take(&mut self.pushes);
        self.last_pushes = pushes.clone();
        self.refine(&pushes);
        let fine_rect = self.fine_rect();
        self.apply_pushes(&pushes, true);
        self.step_pair(dt, true);
        let mut spray = self.peel_spray(&pushes, true, None);
        self.restrict_fine();
        self.apply_pushes_outside(&pushes, fine_rect);
        self.step_pair(dt, false);
        spray.extend(self.peel_spray(&pushes, false, fine_rect));
        if !self.solids.is_empty() {
            self.scour_contacts(dt);
        }
        self.cycle_stream();
        self.step_dust(dt, slow);
        if slow {
            self.settle_dust();
        }
        for s in &spray {
            self.spray_m3 += s.volume;
        }
        self.keep_spray(&spray, dt);
        spray
    }

    /// Tire shoves applied on the step that just finished.
    pub fn tire_pushes(&self) -> &[TirePush] {
        &self.last_pushes
    }

    /// Water and other peeled drops still in the air. The picture mesh does
    /// not draw these. The render contract does.
    pub fn spray_drops(&self) -> &[LiveSpray] {
        &self.drops
    }

    fn keep_spray(&mut self, fresh: &[SprayEvent], dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        for drop in &mut self.drops {
            drop.vel[1] -= GRAVITY * dt;
            drop.pos[0] += drop.vel[0] * dt;
            drop.pos[1] += drop.vel[1] * dt;
            drop.pos[2] += drop.vel[2] * dt;
            drop.age += dt;
        }
        for event in fresh {
            if event.kind != 0 || !event.volume.is_finite() || event.volume <= 0.0 {
                continue;
            }
            self.drops.push(LiveSpray {
                pos: [event.x, event.y, event.z],
                vel: [event.vx, event.vy, event.vz],
                volume: event.volume,
                kind: event.kind,
                age: 0.0,
            });
        }
        self.drops.retain(|drop| {
            drop.age < SPRAY_LIFE_S
                && drop.pos[1] > 0.02
                && drop.volume.is_finite()
                && drop.pos.iter().all(|c| c.is_finite())
                && drop.vel.iter().all(|c| c.is_finite())
        });
        const SPRAY_KEEP: usize = 8192;
        if self.drops.len() > SPRAY_KEEP {
            let skip = self.drops.len() - SPRAY_KEEP;
            self.drops.drain(0..skip);
        }
    }

    pub fn sample(&self, x: f32, z: f32) -> GroundSample {
        if self.fine_on {
            if let Some(s) = self.sample_fine(x, z) {
                return s;
            }
        }
        self.sample_coarse(x, z)
    }

    pub fn ledger(&self) -> FluidLedger {
        let mut water = column_sum(&self.water, COARSE_CELL_M);
        let mut slurry = column_sum(&self.slurry, COARSE_CELL_M);
        if self.fine_on {
            water += column_sum(&self.fine_water, FINE_CELL_M);
            slurry += column_sum(&self.fine_slurry, FINE_CELL_M);
        }
        let dust_air = column_sum(&self.dust, self.air_cell());
        let dust_settled = column_sum(&self.settled, COARSE_CELL_M);
        let mut frozen_water = 0.0f32;
        let mut frozen_slurry = 0.0f32;
        let mut frozen_dust = 0.0f32;
        for p in self.parked.values() {
            let area = COARSE_CELL_M * COARSE_CELL_M;
            frozen_water += p.water * area;
            frozen_slurry += p.slurry * area;
            frozen_dust += p.dust * area;
        }
        for d in self.parked_dust.values() {
            frozen_dust += *d;
        }
        let total = water
            + slurry
            + dust_air
            + dust_settled
            + frozen_water
            + frozen_slurry
            + frozen_dust
            + self.spray_m3;
        FluidLedger {
            water_m3: water,
            slurry_m3: slurry,
            dust_air_m3: dust_air,
            dust_settled_m3: dust_settled,
            frozen_water_m3: frozen_water,
            frozen_slurry_m3: frozen_slurry,
            frozen_dust_m3: frozen_dust,
            spray_m3: self.spray_m3,
            total_m3: total,
        }
    }

    pub fn export_maps(&mut self) -> SurfaceMaps {
        self.restrict_fine();
        let n = WINDOW_RES * WINDOW_RES;
        let mut height = vec![0.0f32; n];
        let mut vel_u = vec![0.0f32; n];
        let mut vel_v = vec![0.0f32; n];
        let mut moisture = vec![0.0f32; n];
        let mut dust = vec![0.0f32; n];
        for idx in 0..n {
            height[idx] = self.bed[idx] + self.water[idx] + self.slurry[idx];
            let hw = finite0(self.water[idx]);
            let hs = finite0(self.slurry[idx]);
            let h = hw + hs;
            if h < 1.0e-5 {
                vel_u[idx] = 0.0;
                vel_v[idx] = 0.0;
            } else {
                vel_u[idx] = finite0((self.uw[idx] * hw + self.us[idx] * hs) / h);
                vel_v[idx] = finite0((self.vw[idx] * hw + self.vs[idx] * hs) / h);
            }
            moisture[idx] = self.moisture[idx];
            dust[idx] = self.settled[idx];
        }
        SurfaceMaps {
            res: WINDOW_RES,
            cell: COARSE_CELL_M,
            origin: self.origin,
            height,
            vel_u,
            vel_v,
            moisture,
            dust,
        }
    }

    /// Axis-aligned window on the ground, metres: origin x, origin z, extent.
    pub fn window_bounds(&self) -> (f32, f32, f32) {
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        (self.origin[0], self.origin[1], extent)
    }

    /// Minimum corner of the coarse window around `center`. Same snap as [`Self::new`].
    ///
    /// The render contract uses this corner. Change it only by bumping
    /// `render_contract::RENDER_CONTRACT_VERSION` and recording the change
    /// in `docs/RENDER_CONTRACT.md`.
    pub fn window_origin(center: [f32; 2]) -> [f32; 2] {
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        snap_origin(center, extent, COARSE_CELL_M)
    }

    /// One coarse cell after the fine patch has been restricted back.
    ///
    /// `u` and `v` are the thickness-weighted water and slurry velocity.
    /// A dry cell reports zero. A non-finite sample reports zero. This is the
    /// flow the render contract publishes. Change the blend only with a
    /// render-contract version bump.
    pub fn coarse_cell(&self, i: usize, j: usize) -> Option<CoarseCell> {
        if i >= WINDOW_RES || j >= WINDOW_RES {
            return None;
        }
        let idx = j * WINDOW_RES + i;
        let water_m = finite0(self.water[idx]);
        let slurry_m = finite0(self.slurry[idx]);
        let film = water_m + slurry_m;
        let (u, v) = if film < 1.0e-5 {
            (0.0, 0.0)
        } else {
            (
                finite0((self.uw[idx] * water_m + self.us[idx] * slurry_m) / film),
                finite0((self.vw[idx] * water_m + self.vs[idx] * slurry_m) / film),
            )
        };
        Some(CoarseCell {
            x: self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M,
            z: self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M,
            bed_m: finite0(self.bed[idx]),
            water_m,
            slurry_m,
            moisture: finite0(self.moisture[idx]),
            u,
            v,
            settled_dust_m: finite0(self.settled[idx]),
        })
    }

    /// Air-grid corner and horizontal cell, metres. Corner of texel (0, 0), not the centre.
    pub fn air_window(&self) -> ([f32; 2], f32) {
        (self.air_origin, self.air_cell())
    }

    /// Dust depth in one air slab, metres of equivalent column in that layer.
    /// Index is `ix + layer * AIR_RES * AIR_RES + iz * AIR_RES` in solver storage.
    /// The render contract republishes this as a volume fraction.
    pub fn air_depth(&self, ix: usize, layer: usize, iz: usize) -> f32 {
        if ix >= AIR_RES || iz >= AIR_RES || layer >= AIR_LAYERS {
            return 0.0;
        }
        let idx = (layer * AIR_RES + iz) * AIR_RES + ix;
        self.dust[idx]
    }

    /// Loudest airborne columns, then the settled film, up to `limit` marks.
    /// Heights are metres above the bed. This does not move mass.
    pub fn dust_marks(&self, limit: usize) -> Vec<DustMark> {
        if limit == 0 {
            return Vec::new();
        }
        let cell = self.air_cell();
        let mut air: Vec<(f32, usize, usize, usize)> = Vec::new();
        for layer in 0..AIR_LAYERS {
            for j in 0..AIR_RES {
                for i in 0..AIR_RES {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    let h = self.dust[idx];
                    if h > 2.0e-5 {
                        air.push((h, i, j, layer));
                    }
                }
            }
        }
        air.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let air_n = (limit * 2 / 3).max(1).min(limit);
        let mut marks = Vec::with_capacity(limit);
        for (_, i, j, layer) in air.into_iter().take(air_n) {
            let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
            let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
            let ground = self.sample(x, z);
            let idx = (layer * AIR_RES + j) * AIR_RES + i;
            marks.push(DustMark {
                x,
                y: ground.bed_m + ground.water_m + 0.16 + layer as f32 * 0.26,
                z,
                u: self.dust_u[idx],
                v: self.dust_v[idx],
                settled: false,
            });
        }
        let mut film: Vec<(f32, usize)> = Vec::new();
        for (idx, h) in self.settled.iter().enumerate() {
            if *h > 2.0e-5 {
                film.push((*h, idx));
            }
        }
        film.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, idx) in film.into_iter().take(limit.saturating_sub(marks.len())) {
            let i = idx % WINDOW_RES;
            let j = idx / WINDOW_RES;
            let (x, z) = self.coarse_xz(i, j);
            let ground = self.sample(x, z);
            marks.push(DustMark {
                x,
                y: ground.bed_m + ground.water_m + ground.slurry_m + 0.025,
                z,
                u: 0.0,
                v: 0.0,
                settled: true,
            });
        }
        marks
    }

    /// Write water and slurry back onto a coarser or matching ground grid.
    /// `deposit(x, z, water, slurry)` replaces the columns at that point.
    pub fn write_columns(&self, mut deposit: impl FnMut(f32, f32, f32, f32)) {
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let idx = j * WINDOW_RES + i;
                if self.water[idx] <= 1.0e-6 && self.slurry[idx] <= 1.0e-6 {
                    continue;
                }
                let (x, z) = self.coarse_xz(i, j);
                deposit(x, z, self.water[idx], self.slurry[idx]);
            }
        }
    }

    /// Force, buoyancy, and the scour account, read from the coarse sheet.
    pub fn hydro_load(&self) -> HydroLoad {
        let mask = if self.solids.is_empty() {
            vec![0u8; WINDOW_RES * WINDOW_RES]
        } else {
            self.solid_mask(false)
        };
        let mut force_x = 0.0f32;
        let mut force_z = 0.0f32;
        let mut wheel_blocks = 0u32;
        let mut belly_blocks = 0u32;
        let cell = COARSE_CELL_M;
        let res = WINDOW_RES;
        for m in &mask {
            if *m == 1 {
                wheel_blocks += 1;
            } else if *m == 2 {
                belly_blocks += 1;
            }
        }
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                if mask[idx] != 0 {
                    continue;
                }
                let h = self.water[idx];
                if h < 1.0e-4 || !h.is_finite() {
                    continue;
                }
                for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                        continue;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    if mask[nidx] == 0 {
                        continue;
                    }
                    let nx = di as f32;
                    let nz = dj as f32;
                    let hydro = 1000.0 * GRAVITY * h * h * 0.5 * cell;
                    force_x += nx * hydro;
                    force_z += nz * hydro;
                    let vn = self.uw[idx] * nx + self.vw[idx] * nz;
                    if vn > 0.0 {
                        let dyn_f = 0.5 * 1000.0 * vn * vn * h * cell;
                        force_x += nx * dyn_f;
                        force_z += nz * dyn_f;
                    }
                }
            }
        }
        let mut wheel_b = 0.0f32;
        let mut hull_b = 0.0f32;
        let mut displaced = 0.0f32;
        for solid in &self.solids {
            let Some((bed, water)) = self.ring_surface(solid) else {
                continue;
            };
            if solid.radius > 0.0 {
                let vol = cylinder_segment_volume(water, solid.radius, solid.width);
                displaced += vol;
                wheel_b += 1000.0 * GRAVITY * vol;
            } else {
                let eta = bed + water;
                let height = (solid.top_y - solid.bottom_y).max(0.0);
                let sub = (eta - solid.bottom_y).clamp(0.0, height);
                let vol = (solid.half_length * 2.0) * (solid.half_width * 2.0) * sub * 0.36;
                displaced += vol;
                hull_b += 1000.0 * GRAVITY * vol;
            }
        }
        HydroLoad {
            force_x,
            force_z,
            buoyancy_n: wheel_b + hull_b,
            hull_buoyancy_n: hull_b,
            wheel_buoyancy_n: wheel_b,
            displaced_m3: displaced,
            eroded_m3: self.eroded_m3,
            hole_m: self.scour_hole_m,
            wheel_blocks,
            belly_blocks,
        }
    }

    fn ring_surface(&self, solid: &HydroSolid) -> Option<(f32, f32)> {
        let reach = solid.half_length.max(solid.half_width) + 0.55;
        let i0 = ((solid.x - reach - self.origin[0]) / COARSE_CELL_M).floor() as i32;
        let i1 = ((solid.x + reach - self.origin[0]) / COARSE_CELL_M).floor() as i32;
        let j0 = ((solid.z - reach - self.origin[1]) / COARSE_CELL_M).floor() as i32;
        let j1 = ((solid.z + reach - self.origin[1]) / COARSE_CELL_M).floor() as i32;
        let mut bed = 0.0f32;
        let mut water = 0.0f32;
        let mut n = 0.0f32;
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= WINDOW_RES as i32 || j >= WINDOW_RES as i32 {
                    continue;
                }
                let (x, z) = self.coarse_xz(i as usize, j as usize);
                if solid_contains(solid, x, z) {
                    continue;
                }
                if (x - solid.x).hypot(z - solid.z) > reach {
                    continue;
                }
                let idx = j as usize * WINDOW_RES + i as usize;
                if self.water[idx] < 0.01 {
                    continue;
                }
                bed += self.bed[idx];
                water += self.water[idx];
                n += 1.0;
            }
        }
        if n < 1.0 {
            None
        } else {
            Some((bed / n, water / n))
        }
    }

    fn solid_mask(&self, fine: bool) -> Vec<u8> {
        let (res, cell, origin, water, bed) = if fine {
            (
                FINE_RES,
                FINE_CELL_M,
                self.fine_origin,
                self.fine_water.as_slice(),
                self.fine_bed.as_slice(),
            )
        } else {
            (
                WINDOW_RES,
                COARSE_CELL_M,
                self.origin,
                self.water.as_slice(),
                self.bed.as_slice(),
            )
        };
        let mut mask = vec![0u8; res * res];
        for j in 0..res {
            for i in 0..res {
                let x = origin[0] + (i as f32 + 0.5) * cell;
                let z = origin[1] + (j as f32 + 0.5) * cell;
                let idx = j * res + i;
                // The free surface beside a cell that the body has already
                // emptied. The wall stays up for as long as that water stands
                // above the solid's underside.
                let mut eta = bed[idx] + water[idx];
                for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                        continue;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    eta = eta.max(bed[nidx] + water[nidx]);
                }
                for solid in self.solids.iter().filter(|s| !s.contact) {
                    if solid_overlaps(solid, x, z, eta, bed[idx]) {
                        mask[idx] = 2;
                    }
                }
                for solid in self.solids.iter().filter(|s| s.contact) {
                    if solid_overlaps(solid, x, z, eta, bed[idx]) {
                        mask[idx] = 1;
                    }
                }
            }
        }
        mask
    }

    /// Excess shear beside a tire contact entrains bed into the slurry sheet.
    fn scour_contacts(&mut self, dt: f32) {
        if dt <= 0.0 || self.solids.is_empty() {
            return;
        }
        let mask = self.solid_mask(false);
        let res = WINDOW_RES;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let mut cut = vec![0.0f32; res * res];
        for j in 1..res - 1 {
            for i in 1..res - 1 {
                let idx = j * res + i;
                if mask[idx] != 0 || self.water[idx] < 0.02 {
                    continue;
                }
                let mut touch = false;
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nidx = (j as i32 + dj) as usize * res + (i as i32 + di) as usize;
                    if mask[nidx] == 1 {
                        touch = true;
                        break;
                    }
                }
                if !touch {
                    continue;
                }
                let speed = self.uw[idx].hypot(self.vw[idx]);
                if !speed.is_finite() {
                    continue;
                }
                let tau = 1000.0 * 0.03 * speed * speed;
                let tau_c = 1.5f32;
                if tau <= tau_c {
                    continue;
                }
                let rate = (0.0035 * (tau / tau_c - 1.0)).min(0.012);
                let room = (0.08 - self.bed_drop[idx]).max(0.0);
                cut[idx] = (rate * dt).min(room);
            }
        }
        for idx in 0..res * res {
            let dh = cut[idx];
            if dh <= 0.0 {
                continue;
            }
            self.bed[idx] -= dh;
            self.slurry[idx] += dh;
            self.us[idx] = self.uw[idx];
            self.vs[idx] = self.vw[idx];
            self.grain[idx] = Grain::Silt as u8;
            self.moisture[idx] = self.moisture[idx].max(0.9);
            self.bed_drop[idx] += dh;
            self.eroded_m3 += dh * area;
            self.scour_hole_m = self.scour_hole_m.max(self.bed_drop[idx]);
        }
    }

    /// Recycle a reach. The downstream lip copies the flowing interior so the
    /// bank does not reflect, and the volume that left is put back on the
    /// upstream lip. No lip, no change.
    fn cycle_stream(&mut self) {
        let Some(cyc) = self.cycle else {
            return;
        };
        let along_z = cyc.dir_z.abs() >= cyc.dir_x.abs();
        let cell = COARSE_CELL_M;
        let res = WINDOW_RES;
        let lip = |p: f32, origin: f32| -> i32 { ((p - origin) / cell).floor() as i32 };
        if along_z {
            let j_up = lip(cyc.z0, self.origin[1]);
            let j_dn = lip(cyc.z1, self.origin[1]);
            // Reference is a few cells inside the upstream lip, away from the
            // car and away from the bank. The last four rows copy it.
            let (j_ref, j_out0, j_out1, j_inlet) = if cyc.dir_z >= 0.0 {
                (j_up + 6, j_dn - 3, j_dn, j_up)
            } else {
                (j_dn - 6, j_up, j_up + 3, j_dn)
            };
            if j_up < 1
                || j_dn >= res as i32 - 1
                || j_ref <= j_out1 && j_ref >= j_out0
                || j_out0 <= j_up
            {
                return;
            }
            let i0 = lip(cyc.x0, self.origin[0]);
            let i1 = lip(cyc.x1, self.origin[0]);
            for i in i0..i1 {
                if i < 0 || i >= res as i32 {
                    continue;
                }
                self.recycle_column(
                    i as usize,
                    j_ref as usize,
                    j_out0 as usize,
                    j_out1 as usize,
                    j_inlet as usize,
                    true,
                );
            }
        } else {
            let i_up = lip(cyc.x0, self.origin[0]);
            let i_dn = lip(cyc.x1, self.origin[0]);
            let (i_ref, i_out0, i_out1, i_inlet) = if cyc.dir_x >= 0.0 {
                (i_up + 6, i_dn - 3, i_dn, i_up)
            } else {
                (i_dn - 6, i_up, i_up + 3, i_dn)
            };
            if i_up < 1 || i_dn >= res as i32 - 1 || i_out0 <= i_up {
                return;
            }
            let j0 = lip(cyc.z0, self.origin[1]);
            let j1 = lip(cyc.z1, self.origin[1]);
            for j in j0..j1 {
                if j < 0 || j >= res as i32 {
                    continue;
                }
                self.recycle_column(
                    j as usize,
                    i_ref as usize,
                    i_out0 as usize,
                    i_out1 as usize,
                    i_inlet as usize,
                    false,
                );
            }
        }
    }

    fn recycle_column(
        &mut self,
        cross: usize,
        interior: usize,
        out0: usize,
        out1: usize,
        inlet: usize,
        along_z: bool,
    ) {
        let res = WINDOW_RES;
        let at = |a: usize, b: usize| -> usize {
            if along_z {
                a * res + b
            } else {
                b * res + a
            }
        };
        let src = at(interior, cross);
        if self.water[src] < 0.02 || !self.water[src].is_finite() {
            return;
        }
        let mut surplus = 0.0f32;
        for out in out0..=out1 {
            let idx = at(out, cross);
            if self.water[idx].is_finite() {
                surplus += self.water[idx] - self.water[src];
            }
            self.water[idx] = self.water[src];
            self.uw[idx] = self.uw[src];
            self.vw[idx] = self.vw[src];
        }
        let inn = at(inlet, cross);
        let next = (self.water[inn] + surplus).max(0.0);
        if next.is_finite() {
            self.water[inn] = next;
        }
        if self.vw[src].abs() + self.uw[src].abs() > 0.05 {
            self.uw[inn] = self.uw[src];
            self.vw[inn] = self.vw[src];
        }
    }

    fn step_pair(&mut self, dt: f32, fine: bool) {
        let (res, cell) = if fine {
            (FINE_RES, FINE_CELL_M)
        } else {
            (WINDOW_RES, COARSE_CELL_M)
        };
        let mask = if fine && !self.fine_on {
            None
        } else if self.solids.is_empty() && self.cycle.is_none() {
            None
        } else if self.solids.is_empty() {
            let n = if fine {
                FINE_RES * FINE_RES
            } else {
                WINDOW_RES * WINDOW_RES
            };
            Some(vec![0u8; n])
        } else {
            Some(self.solid_mask(fine))
        };
        let n = res * res;
        let nu_w = vec![dynamic_viscosity(Grain::Water, 1.0); n];
        let mut nu_s = vec![0.0f32; n];
        let gate = vec![1u8; n];
        let mut bed = vec![0.0f32; n];
        let (water, slurry, uw, vw, us, vs, bed_s, moisture, grain) = if fine {
            (
                &mut self.fine_water,
                &mut self.fine_slurry,
                &mut self.fine_uw,
                &mut self.fine_vw,
                &mut self.fine_us,
                &mut self.fine_vs,
                &self.fine_bed,
                &self.fine_moisture,
                &self.fine_grain,
            )
        } else {
            (
                &mut self.water,
                &mut self.slurry,
                &mut self.uw,
                &mut self.vw,
                &mut self.us,
                &mut self.vs,
                &self.bed,
                &self.moisture,
                &self.grain,
            )
        };
        for idx in 0..n {
            bed[idx] = bed_s[idx];
            nu_s[idx] = dynamic_viscosity(Grain::from_u8(grain[idx]), moisture[idx]);
            let _ = nu_w[idx];
        }
        // Soft edge of the window: slow the velocity, keep the mass.
        if !fine {
            soften_border(uw, vw, res);
            soften_border(us, vs, res);
        }
        if let Some(mask) = &mask {
            displace_occupied(water, uw, vw, mask, res);
            displace_occupied(slurry, us, vs, mask, res);
            step_sheet_eps(
                water,
                uw,
                vw,
                &bed,
                &nu_w,
                &gate,
                res,
                cell,
                dt,
                SheetDrive::Gravity { vorticity: 0.08 },
                1.0e-6,
                Some(mask),
            );
            step_sheet_eps(
                slurry,
                us,
                vs,
                &bed,
                &nu_s,
                &gate,
                res,
                cell,
                dt,
                SheetDrive::Gravity { vorticity: 0.05 },
                1.0e-6,
                Some(mask),
            );
        } else {
            step_sheet(
                water,
                uw,
                vw,
                &bed,
                &nu_w,
                &gate,
                res,
                cell,
                dt,
                SheetDrive::Gravity { vorticity: 0.08 },
            );
            step_sheet(
                slurry,
                us,
                vs,
                &bed,
                &nu_s,
                &gate,
                res,
                cell,
                dt,
                SheetDrive::Gravity { vorticity: 0.05 },
            );
        }
    }

    fn apply_pushes(&mut self, pushes: &[TirePush], fine: bool) {
        for p in pushes {
            if fine && self.cell_is_refined(p.x, p.z) {
                self.push_on(p, true);
            }
        }
    }

    fn apply_pushes_outside(&mut self, pushes: &[TirePush], skip: Option<[f32; 4]>) {
        for p in pushes {
            if let Some(rect) = skip {
                if p.x >= rect[0] && p.x < rect[1] && p.z >= rect[2] && p.z < rect[3] {
                    continue;
                }
            }
            self.push_on(p, false);
        }
    }

    fn push_on(&mut self, p: &TirePush, fine: bool) {
        let (res, cell, origin, water, slurry, uw, vw, us, vs) = if fine {
            (
                FINE_RES,
                FINE_CELL_M,
                self.fine_origin,
                &mut self.fine_water,
                &mut self.fine_slurry,
                &mut self.fine_uw,
                &mut self.fine_vw,
                &mut self.fine_us,
                &mut self.fine_vs,
            )
        } else {
            (
                WINDOW_RES,
                COARSE_CELL_M,
                self.origin,
                &mut self.water,
                &mut self.slurry,
                &mut self.uw,
                &mut self.vw,
                &mut self.us,
                &mut self.vs,
            )
        };
        let fx = p.forward_x;
        let fz = p.forward_z;
        let len = (fx * fx + fz * fz).sqrt().max(1.0e-4);
        let fx = fx / len;
        let fz = fz / len;
        let rx = fz;
        let rz = -fx;
        let speed = p.speed.max(0.0);
        for j in 0..res {
            for i in 0..res {
                let x = origin[0] + (i as f32 + 0.5) * cell;
                let z = origin[1] + (j as f32 + 0.5) * cell;
                let dx = x - p.x;
                let dz = z - p.z;
                let along = dx * fx + dz * fz;
                let lat = dx * rx + dz * rz;
                if lat.abs() > p.width * 0.65 + cell {
                    continue;
                }
                if along < -p.radius * 1.4 || along > p.radius * 1.1 {
                    continue;
                }
                let idx = j * res + i;
                let flank = (lat.abs() / (p.width * 0.65 + cell)).clamp(0.0, 1.0);
                let shove = speed.max(0.4) * (1.0 - flank * flank);
                if water[idx] > 0.004 {
                    uw[idx] += fx * shove * 0.85;
                    vw[idx] += fz * shove * 0.85;
                    uw[idx] += rx * lat.signum() * shove * 0.18 * flank;
                    vw[idx] += rz * lat.signum() * shove * 0.18 * flank;
                }
                if slurry[idx] > 0.004 {
                    // Stretch behind the tire, shove ahead, part the shoulders.
                    let fore = if along > 0.0 { 1.0 } else { 0.72 };
                    us[idx] += fx * shove * fore;
                    vs[idx] += fz * shove * fore;
                    us[idx] += rx * lat.signum() * shove * 0.28 * flank;
                    vs[idx] += rz * lat.signum() * shove * 0.28 * flank;
                }
            }
        }
    }

    fn peel_spray(
        &mut self,
        pushes: &[TirePush],
        fine: bool,
        skip: Option<[f32; 4]>,
    ) -> Vec<SprayEvent> {
        let mut out = Vec::new();
        let (res, cell, origin, water, uw, vw) = if fine {
            (
                FINE_RES,
                FINE_CELL_M,
                self.fine_origin,
                &mut self.fine_water,
                &self.fine_uw,
                &self.fine_vw,
            )
        } else {
            (
                WINDOW_RES,
                COARSE_CELL_M,
                self.origin,
                &mut self.water,
                &self.uw,
                &self.vw,
            )
        };
        if pushes.is_empty() {
            return out;
        }
        let area = cell * cell;
        for p in pushes {
            if p.speed < 0.8 {
                continue;
            }
            for j in 0..res {
                for i in 0..res {
                    let x = origin[0] + (i as f32 + 0.5) * cell;
                    let z = origin[1] + (j as f32 + 0.5) * cell;
                    if let Some(rect) = skip {
                        if x >= rect[0] && x < rect[1] && z >= rect[2] && z < rect[3] {
                            continue;
                        }
                    }
                    let dx = x - p.x;
                    let dz = z - p.z;
                    if dx * dx + dz * dz > (p.radius * 1.3) * (p.radius * 1.3) {
                        continue;
                    }
                    let idx = j * res + i;
                    if water[idx] < 0.01 {
                        continue;
                    }
                    let take = (water[idx] * 0.04).min(0.004);
                    if take * area < 1.0e-6 {
                        continue;
                    }
                    water[idx] -= take;
                    out.push(SprayEvent {
                        x,
                        y: 0.35,
                        z,
                        vx: uw[idx] * 0.3,
                        vy: 1.4 + p.speed * 0.25,
                        vz: vw[idx] * 0.3,
                        volume: take * area,
                        kind: 0,
                    });
                }
            }
        }
        out
    }

    fn fine_rect(&self) -> Option<[f32; 4]> {
        if !self.fine_on {
            return None;
        }
        let extent = FINE_RES as f32 * FINE_CELL_M;
        Some([
            self.fine_origin[0],
            self.fine_origin[0] + extent,
            self.fine_origin[1],
            self.fine_origin[1] + extent,
        ])
    }

    /// A coarse cell is refined only when the fine patch covers it completely,
    /// so the hand-off does not drop the uncovered part of the cell.
    fn cell_is_refined(&self, x: f32, z: f32) -> bool {
        if !self.fine_on {
            return false;
        }
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return false;
        };
        let x0 = self.origin[0] + i as f32 * COARSE_CELL_M;
        let z0 = self.origin[1] + j as f32 * COARSE_CELL_M;
        let x1 = x0 + COARSE_CELL_M;
        let z1 = z0 + COARSE_CELL_M;
        let Some(rect) = self.fine_rect() else {
            return false;
        };
        x0 >= rect[0] - 1.0e-4
            && z0 >= rect[2] - 1.0e-4
            && x1 <= rect[1] + 1.0e-4
            && z1 <= rect[3] + 1.0e-4
    }

    fn refine(&mut self, pushes: &[TirePush]) {
        self.restrict_fine();
        if pushes.is_empty() {
            return;
        }
        let mut cx = 0.0f32;
        let mut cz = 0.0f32;
        for p in pushes {
            cx += p.x;
            cz += p.z;
        }
        let n = pushes.len() as f32;
        cx /= n;
        cz /= n;
        let extent = FINE_RES as f32 * FINE_CELL_M;
        self.fine_origin = snap_origin([cx, cz], extent, COARSE_CELL_M);
        self.fine_water.fill(0.0);
        self.fine_slurry.fill(0.0);
        self.fine_uw.fill(0.0);
        self.fine_vw.fill(0.0);
        self.fine_us.fill(0.0);
        self.fine_vs.fill(0.0);
        let full = ((COARSE_CELL_M / FINE_CELL_M).round() as u32).pow(2);
        let mut count = vec![0u32; WINDOW_RES * WINDOW_RES];
        let mut hits: Vec<(usize, usize)> = Vec::new();
        for j in 0..FINE_RES {
            for i in 0..FINE_RES {
                let x = self.fine_origin[0] + (i as f32 + 0.5) * FINE_CELL_M;
                let z = self.fine_origin[1] + (j as f32 + 0.5) * FINE_CELL_M;
                let Some((ci, cj)) = self.coarse_ij(x, z) else {
                    continue;
                };
                let cidx = cj * WINDOW_RES + ci;
                count[cidx] += 1;
                hits.push((j * FINE_RES + i, cidx));
            }
        }
        for (fidx, cidx) in &hits {
            if count[*cidx] != full {
                continue;
            }
            self.fine_water[*fidx] = self.water[*cidx];
            self.fine_slurry[*fidx] = self.slurry[*cidx];
            self.fine_uw[*fidx] = self.uw[*cidx];
            self.fine_vw[*fidx] = self.vw[*cidx];
            self.fine_us[*fidx] = self.us[*cidx];
            self.fine_vs[*fidx] = self.vs[*cidx];
            self.fine_bed[*fidx] = self.bed[*cidx];
            self.fine_moisture[*fidx] = self.moisture[*cidx];
            self.fine_grain[*fidx] = self.grain[*cidx];
        }
        for (cidx, n) in count.iter().enumerate() {
            if *n == full {
                self.water[cidx] = 0.0;
                self.slurry[cidx] = 0.0;
                self.uw[cidx] = 0.0;
                self.vw[cidx] = 0.0;
                self.us[cidx] = 0.0;
                self.vs[cidx] = 0.0;
            }
        }
        self.fine_on = true;
    }

    fn restrict_fine(&mut self) {
        if !self.fine_on {
            return;
        }
        let mut acc_w = vec![0.0f32; WINDOW_RES * WINDOW_RES];
        let mut acc_s = vec![0.0f32; WINDOW_RES * WINDOW_RES];
        let mut acc_uw = vec![0.0f32; WINDOW_RES * WINDOW_RES];
        let mut acc_vw = vec![0.0f32; WINDOW_RES * WINDOW_RES];
        let mut acc_us = vec![0.0f32; WINDOW_RES * WINDOW_RES];
        let mut acc_vs = vec![0.0f32; WINDOW_RES * WINDOW_RES];
        let mut cnt = vec![0u32; WINDOW_RES * WINDOW_RES];
        let fine_area = FINE_CELL_M * FINE_CELL_M;
        let coarse_area = COARSE_CELL_M * COARSE_CELL_M;
        for j in 0..FINE_RES {
            for i in 0..FINE_RES {
                let x = self.fine_origin[0] + (i as f32 + 0.5) * FINE_CELL_M;
                let z = self.fine_origin[1] + (j as f32 + 0.5) * FINE_CELL_M;
                let Some((ci, cj)) = self.coarse_ij(x, z) else {
                    let fidx = j * FINE_RES + i;
                    // Outside the coarse window: park the fine column's volume.
                    let key = world_key(x, z, COARSE_CELL_M);
                    let slot = self.parked.entry(key).or_default();
                    slot.water += self.fine_water[fidx] * fine_area / coarse_area;
                    slot.slurry += self.fine_slurry[fidx] * fine_area / coarse_area;
                    continue;
                };
                let cidx = cj * WINDOW_RES + ci;
                let fidx = j * FINE_RES + i;
                acc_w[cidx] += self.fine_water[fidx] * fine_area;
                acc_s[cidx] += self.fine_slurry[fidx] * fine_area;
                acc_uw[cidx] += self.fine_uw[fidx] * self.fine_water[fidx] * fine_area;
                acc_vw[cidx] += self.fine_vw[fidx] * self.fine_water[fidx] * fine_area;
                acc_us[cidx] += self.fine_us[fidx] * self.fine_slurry[fidx] * fine_area;
                acc_vs[cidx] += self.fine_vs[fidx] * self.fine_slurry[fidx] * fine_area;
                cnt[cidx] += 1;
            }
        }
        for idx in 0..WINDOW_RES * WINDOW_RES {
            if cnt[idx] == 0 {
                continue;
            }
            let w = acc_w[idx] / coarse_area;
            let s = acc_s[idx] / coarse_area;
            self.water[idx] = w;
            self.slurry[idx] = s;
            if w > 1.0e-6 {
                self.uw[idx] = acc_uw[idx] / (w * coarse_area);
                self.vw[idx] = acc_vw[idx] / (w * coarse_area);
            }
            if s > 1.0e-6 {
                self.us[idx] = acc_us[idx] / (s * coarse_area);
                self.vs[idx] = acc_vs[idx] / (s * coarse_area);
            }
        }
        self.fine_water.fill(0.0);
        self.fine_slurry.fill(0.0);
        self.fine_on = false;
    }

    fn covers_fine(&self, x: f32, z: f32) -> bool {
        if !self.fine_on {
            return false;
        }
        let extent = FINE_RES as f32 * FINE_CELL_M;
        x >= self.fine_origin[0]
            && x < self.fine_origin[0] + extent
            && z >= self.fine_origin[1]
            && z < self.fine_origin[1] + extent
    }

    fn step_dust(&mut self, dt: f32, _slow: bool) {
        let sum: f32 = self.dust.iter().sum();
        if sum <= 1.0e-6 {
            return;
        }
        let cell = self.air_cell();
        let n = AIR_RES * AIR_RES;
        for layer in 0..AIR_LAYERS {
            let start = layer * n;
            let h = &mut self.dust[start..start + n];
            let u = &mut self.dust_u[start..start + n];
            let v = &mut self.dust_v[start..start + n];
            let bed = vec![0.0f32; n];
            let nu = vec![0.4f32; n];
            let gate = vec![1u8; n];
            soften_border(u, v, AIR_RES);
            step_sheet(
                h,
                u,
                v,
                &bed,
                &nu,
                &gate,
                AIR_RES,
                cell,
                dt,
                SheetDrive::Wind {
                    x: self.wind[0],
                    z: self.wind[1],
                    vorticity: 0.35,
                },
            );
        }
    }

    fn settle_dust(&mut self) {
        let air_cell = self.air_cell();
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * air_cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * air_cell;
                let ground = self.sample_coarse(x, z);
                let wet = !dust_may_lift(ground.moisture, ground.water_m);
                let fall = if wet { 0.62 } else { 0.05 };
                for layer in 0..AIR_LAYERS {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    let drop = self.dust[idx] * fall;
                    if drop <= 0.0 {
                        continue;
                    }
                    self.dust[idx] -= drop;
                    if layer == 0 {
                        if let Some((ci, cj)) = self.coarse_ij(x, z) {
                            let cidx = cj * WINDOW_RES + ci;
                            // Air cell and coarse cell differ. Convert thickness
                            // so the cubic metres match.
                            let air_area = air_cell * air_cell;
                            let coarse_area = COARSE_CELL_M * COARSE_CELL_M;
                            self.settled[cidx] += drop * air_area / coarse_area;
                        }
                    } else {
                        let below = ((layer - 1) * AIR_RES + j) * AIR_RES + i;
                        self.dust[below] += drop;
                    }
                }
                if wet {
                    // Do not loft what just landed.
                    let _ = ground;
                }
            }
        }
    }

    fn shift_coarse(&mut self, new_origin: [f32; 2]) {
        if (new_origin[0] - self.origin[0]).abs() < 1.0e-4
            && (new_origin[1] - self.origin[1]).abs() < 1.0e-4
        {
            return;
        }
        let n = WINDOW_RES * WINDOW_RES;
        let mut water = vec![0.0f32; n];
        let mut slurry = vec![0.0f32; n];
        let mut uw = vec![0.0f32; n];
        let mut vw = vec![0.0f32; n];
        let mut us = vec![0.0f32; n];
        let mut vs = vec![0.0f32; n];
        let mut bed = vec![0.0f32; n];
        let mut moisture = vec![0.0f32; n];
        let mut grain = vec![Grain::Other as u8; n];
        let mut settled = vec![0.0f32; n];
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let x = self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M;
                let z = self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M;
                let idx = j * WINDOW_RES + i;
                let key = world_key(x, z, COARSE_CELL_M);
                let slot = Parked {
                    water: self.water[idx],
                    slurry: self.slurry[idx],
                    uw: self.uw[idx],
                    vw: self.vw[idx],
                    us: self.us[idx],
                    vs: self.vs[idx],
                    dust: self.settled[idx],
                };
                if slot.water > 1.0e-8 || slot.slurry > 1.0e-8 || slot.dust > 1.0e-8 {
                    self.parked.insert(key, slot);
                } else {
                    self.parked.remove(&key);
                }
            }
        }
        self.origin = new_origin;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let x = self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M;
                let z = self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M;
                let key = world_key(x, z, COARSE_CELL_M);
                let idx = j * WINDOW_RES + i;
                if let Some(slot) = self.parked.remove(&key) {
                    water[idx] = slot.water;
                    slurry[idx] = slot.slurry;
                    uw[idx] = slot.uw;
                    vw[idx] = slot.vw;
                    us[idx] = slot.us;
                    vs[idx] = slot.vs;
                    settled[idx] = slot.dust;
                }
                bed[idx] = self.bed[idx];
                moisture[idx] = self.moisture[idx];
                grain[idx] = self.grain[idx];
            }
        }
        self.water = water;
        self.slurry = slurry;
        self.uw = uw;
        self.vw = vw;
        self.us = us;
        self.vs = vs;
        self.settled = settled;
        self.bed = bed;
        self.moisture = moisture;
        self.grain = grain;
    }

    fn shift_air(&mut self, new_origin: [f32; 2], cell: f32) {
        if (new_origin[0] - self.air_origin[0]).abs() < 1.0e-4
            && (new_origin[1] - self.air_origin[1]).abs() < 1.0e-4
        {
            return;
        }
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
                let mut vol = 0.0f32;
                for layer in 0..AIR_LAYERS {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    vol += self.dust[idx];
                    self.dust[idx] = 0.0;
                    self.dust_u[idx] = 0.0;
                    self.dust_v[idx] = 0.0;
                }
                let key = world_key(x, z, cell);
                if vol > 1.0e-8 {
                    *self.parked_dust.entry(key).or_insert(0.0) += vol * cell * cell;
                }
            }
        }
        self.air_origin = new_origin;
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
                let key = world_key(x, z, cell);
                if let Some(vol) = self.parked_dust.remove(&key) {
                    let idx = j * AIR_RES + i;
                    self.dust[idx] = vol / (cell * cell);
                }
            }
        }
    }

    fn sample_coarse(&self, x: f32, z: f32) -> GroundSample {
        let Some((i, j)) = self.coarse_ij(x, z) else {
            let key = world_key(x, z, COARSE_CELL_M);
            if let Some(p) = self.parked.get(&key) {
                return GroundSample {
                    water_m: p.water,
                    slurry_m: p.slurry,
                    u: p.us,
                    v: p.vs,
                    settled_dust_m: p.dust,
                    moisture: 0.0,
                    bed_m: 0.0,
                };
            }
            return GroundSample::default();
        };
        let idx = j * WINDOW_RES + i;
        GroundSample {
            water_m: self.water[idx],
            slurry_m: self.slurry[idx],
            u: self.us[idx],
            v: self.vs[idx],
            settled_dust_m: self.settled[idx],
            moisture: self.moisture[idx],
            bed_m: self.bed[idx],
        }
    }

    fn sample_fine(&self, x: f32, z: f32) -> Option<GroundSample> {
        if !self.covers_fine(x, z) {
            return None;
        }
        let i = ((x - self.fine_origin[0]) / FINE_CELL_M).floor() as usize;
        let j = ((z - self.fine_origin[1]) / FINE_CELL_M).floor() as usize;
        if i >= FINE_RES || j >= FINE_RES {
            return None;
        }
        let idx = j * FINE_RES + i;
        Some(GroundSample {
            water_m: self.fine_water[idx],
            slurry_m: self.fine_slurry[idx],
            u: self.fine_us[idx],
            v: self.fine_vs[idx],
            settled_dust_m: 0.0,
            moisture: self.fine_moisture[idx],
            bed_m: self.fine_bed[idx],
        })
    }

    fn coarse_xz(&self, i: usize, j: usize) -> (f32, f32) {
        (
            self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M,
            self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M,
        )
    }

    fn coarse_ij(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        let i = ((x - self.origin[0]) / COARSE_CELL_M).floor();
        let j = ((z - self.origin[1]) / COARSE_CELL_M).floor();
        if i < 0.0 || j < 0.0 {
            return None;
        }
        let i = i as usize;
        let j = j as usize;
        if i >= WINDOW_RES || j >= WINDOW_RES {
            None
        } else {
            Some((i, j))
        }
    }

    fn air_cell(&self) -> f32 {
        (WINDOW_RES as f32 * COARSE_CELL_M) / AIR_RES as f32
    }

    fn air_ij(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        let cell = self.air_cell();
        let i = ((x - self.air_origin[0]) / cell).floor();
        let j = ((z - self.air_origin[1]) / cell).floor();
        if i < 0.0 || j < 0.0 {
            return None;
        }
        let i = i as usize;
        let j = j as usize;
        if i >= AIR_RES || j >= AIR_RES {
            None
        } else {
            Some((i, j))
        }
    }
}

fn rim_scale(h: &[f32], i: usize, j: usize, rad: i32, feather: f32) -> f32 {
    let n = WINDOW_RES as i32;
    let mut nearest = feather;
    for dj in -rad..=rad {
        for di in -rad..=rad {
            let ii = i as i32 + di;
            let jj = j as i32 + dj;
            let dry = ii < 0
                || jj < 0
                || ii >= n
                || jj >= n
                || h[(jj as usize) * WINDOW_RES + ii as usize] <= 1.0e-6;
            if dry {
                let d = (di as f32).hypot(dj as f32) * COARSE_CELL_M;
                nearest = nearest.min(d);
            }
        }
    }
    let t = (nearest / feather).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn soften_border(u: &mut [f32], v: &mut [f32], res: usize) {
    let edge = (res / 12).max(2);
    for j in 0..res {
        for i in 0..res {
            let di = i.min(res - 1 - i);
            let dj = j.min(res - 1 - j);
            let d = di.min(dj);
            if d >= edge {
                continue;
            }
            let t = d as f32 / edge as f32;
            let w = t * t * (3.0 - 2.0 * t);
            let idx = j * res + i;
            u[idx] *= w;
            v[idx] *= w;
        }
    }
}

fn snap_origin(center: [f32; 2], extent: f32, cell: f32) -> [f32; 2] {
    let x = center[0] - extent * 0.5;
    let z = center[1] - extent * 0.5;
    [(x / cell).floor() * cell, (z / cell).floor() * cell]
}

fn world_key(x: f32, z: f32, cell: f32) -> (i32, i32) {
    ((x / cell).floor() as i32, (z / cell).floor() as i32)
}

/// Bow height in metres. `k * speed * sqrt(depth / g)`, not a speed×depth flux.
pub fn bow_height_m(speed: f32, depth: f32) -> f32 {
    let v = speed.max(0.0);
    let d = depth.max(0.0);
    if v < 0.05 || d < 0.02 {
        return 0.0;
    }
    (0.42 * v * (d / GRAVITY).sqrt()).min(0.24).min(d * 0.85)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slope_accelerates_downhill() {
        let res = 16usize;
        let cell = 0.10f32;
        let n = res * res;
        let mut h = vec![0.05f32; n];
        let mut u = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let mut bed = vec![0.0f32; n];
        for j in 0..res {
            for i in 0..res {
                bed[j * res + i] = -0.04 * j as f32;
            }
        }
        let nu = vec![40.0f32; n];
        let gate = vec![1u8; n];
        step_sheet(
            &mut h,
            &mut u,
            &mut v,
            &bed,
            &nu,
            &gate,
            res,
            cell,
            0.05,
            SheetDrive::Gravity { vorticity: 0.0 },
        );
        let idx = 8 * res + 8;
        assert!(v[idx] > 0.002, "downhill v {}", v[idx]);
    }

    #[test]
    fn viscosity_orders_clay_silt_sand_and_thins() {
        let clay = dynamic_viscosity(Grain::Clay, 0.55);
        let silt = dynamic_viscosity(Grain::Silt, 0.55);
        let sand = dynamic_viscosity(Grain::Sand, 0.55);
        assert!(clay > silt * 1.8, "clay {clay} silt {silt}");
        assert!(silt > sand * 1.5, "silt {silt} sand {sand}");
        let stiff = dynamic_viscosity(Grain::Clay, 0.30);
        let mid = dynamic_viscosity(Grain::Clay, 0.55);
        let thin = dynamic_viscosity(Grain::Clay, 0.85);
        assert!(stiff > mid && mid > thin, "{stiff} {mid} {thin}");
    }

    #[test]
    fn rut_heal_times_match_the_mix() {
        let clay = rut_heal(Grain::Clay, 0.55);
        let silt = rut_heal(Grain::Silt, 0.55);
        let thin = rut_heal(Grain::Clay, 0.90);
        eprintln!(
            "heal clay {:.2}s silt {:.2}s thin {:.2}s  relief8 clay {:.4}/{:.4} silt {:.4}",
            clay.level_s, silt.level_s, thin.level_s, clay.relief_8s, clay.relief0, silt.relief_8s
        );
        assert!(
            (10.0..=15.0).contains(&clay.level_s),
            "clay heal {:.2}",
            clay.level_s
        );
        assert!(
            (10.0..=15.0).contains(&silt.level_s),
            "silt heal {:.2}",
            silt.level_s
        );
        assert!(
            (1.0..=3.0).contains(&thin.level_s),
            "thin heal {:.2}",
            thin.level_s
        );
        assert!(clay.still_deep_at_8s, "clay relief fell early");
        assert!(silt.still_deep_at_8s, "silt relief fell early");
    }

    #[test]
    fn sheet_conserves_and_runs_downhill() {
        let res = 24usize;
        let cell = 0.2f32;
        let n = res * res;
        let mut h = vec![0.0f32; n];
        let mut u = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let mut bed = vec![0.0f32; n];
        let nu = vec![dynamic_viscosity(Grain::Clay, 0.55); n];
        let gate = vec![1u8; n];
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                bed[idx] = -0.16 * j as f32;
                if (4..9).contains(&j) && (8..16).contains(&i) {
                    h[idx] = 0.10;
                }
            }
        }
        let before = column_sum(&h, cell);
        let centroid = |h: &[f32]| -> f32 {
            let mut m = 0.0f32;
            let mut mj = 0.0f32;
            for j in 0..res {
                for i in 0..res {
                    let s = h[j * res + i];
                    m += s;
                    mj += s * j as f32;
                }
            }
            if m <= 1.0e-6 {
                0.0
            } else {
                mj / m
            }
        };
        let c0 = centroid(&h);
        for _ in 0..60 {
            step_sheet(
                &mut h,
                &mut u,
                &mut v,
                &bed,
                &nu,
                &gate,
                res,
                cell,
                0.05,
                SheetDrive::Gravity { vorticity: 0.02 },
            );
        }
        let after = column_sum(&h, cell);
        let c1 = centroid(&h);
        assert!(
            (after - before).abs() < 1.0e-3,
            "volume {before} -> {after}"
        );
        assert!(
            c1 > c0 + 1.5,
            "slurry did not run downhill, centroid {c0:.2} -> {c1:.2}"
        );
    }

    #[test]
    fn dust_does_not_lift_when_wet_and_window_keeps_volume() {
        assert!(!dust_may_lift(0.55, 0.0));
        assert!(!dust_may_lift(0.2, 0.008));
        assert!(dust_may_lift(0.4, 0.004));
        let mut fluid = FluidCore::new([0.0, 0.0]);
        let poured = fluid.pour_water(-1.0, 1.0, -1.0, 1.0, 0.12);
        let before = fluid.ledger().total_m3;
        assert!(
            (before - poured).abs() < 1.0e-3,
            "pour {poured} ledger {before}"
        );
        fluid.recenter([55.0, 0.0]);
        fluid.recenter([0.0, 0.0]);
        let after = fluid.ledger();
        assert!(
            (after.total_m3 - before).abs() < 1.0e-3,
            "recenter {} -> {}",
            before,
            after.total_m3
        );
        assert!(after.water_m3 > poured * 0.5, "water did not return");
        let dry = fluid.inject_dust(
            DustInject {
                x: 2.0,
                z: 2.0,
                slip: 0.4,
                moisture: 0.1,
                water_m: 0.0,
                offer_m3_s: 0.01,
            },
            0.1,
        );
        let wet = fluid.inject_dust(
            DustInject {
                x: 0.0,
                z: 0.0,
                slip: 0.8,
                moisture: 0.7,
                water_m: 0.02,
                offer_m3_s: 0.01,
            },
            0.1,
        );
        assert!(dry > 0.0, "dry ground took no dust");
        assert!(wet == 0.0, "wet ground lofted dust");
    }

    #[test]
    fn bow_height_is_metres() {
        let deep = bow_height_m(1.8, 0.40);
        let slow = bow_height_m(0.4, 0.40);
        let shallow = bow_height_m(1.8, 0.14);
        assert!(deep > 0.05, "deep {deep}");
        assert!(deep > slow + 0.04, "fast {deep} slow {slow}");
        assert!(deep > shallow + 0.03, "deep {deep} shallow {shallow}");
        assert_eq!(bow_height_m(0.0, 0.4), 0.0);
    }
}
