//! Free bodies in the sheet: a ball, a box, a rock, the legs of a walker.
//!
//! A body is a volume that shares the water column with the fluid. On each
//! step the part of a column the body fills below the free surface is added
//! to the bed the sheet steps on, and to nothing else. The water does not
//! change: the step sees a raised floor under the body and moves water off
//! it, outward. That outflow is the displacement ring when a ball drops in,
//! the bow ahead of a walking leg and the trough behind it. Volume is exact,
//! because the lift only ever changes the floor the step reads.
//!
//! The body does its own buoyancy and drag from the surface and current
//! read back here (`WetLoad`). The drag it feels is handed back as an
//! impulse (`WetBody::impulse_*`), which goes into the water around it with
//! the opposite sign, so momentum is exchanged and not made up.

use super::*;

/// Supersamples per coarse cell side when a body fills a column.
const COARSE_SAMPLES: usize = 4;
/// Ring around a body where its surface and current are read, metres.
const RING_M: f32 = 0.30;
/// Water lighter than this does not take part in the ring average.
const RING_MIN_WATER_M: f32 = 2.0e-3;
/// Slurry bulk density used for a body in mud, kg/m³.
pub const SLURRY_RHO: f32 = 1600.0;
pub const WATER_RHO: f32 = 1000.0;

/// What a body looks like to the sheet: the vertical span it fills at a point.
#[derive(Clone, Copy, Debug)]
pub enum WetShape {
    /// Ball centred at `cy`.
    Sphere { cy: f32, r: f32 },
    /// Box with a flat bottom and top, long axis along `yaw` (0 is +Z).
    Box {
        yaw: f32,
        half_l: f32,
        half_w: f32,
        bottom: f32,
        top: f32,
    },
    /// Upright cylinder: a leg, a post.
    Column { r: f32, bottom: f32, top: f32 },
}

impl WetShape {
    /// Lowest and highest point of the body over `(dx, dz)` from its centre.
    #[inline]
    pub fn span(&self, dx: f32, dz: f32) -> Option<(f32, f32)> {
        match *self {
            WetShape::Sphere { cy, r } => {
                let d2 = dx * dx + dz * dz;
                if d2 >= r * r {
                    return None;
                }
                let h = (r * r - d2).sqrt();
                Some((cy - h, cy + h))
            }
            WetShape::Box {
                yaw,
                half_l,
                half_w,
                bottom,
                top,
            } => {
                let (s, c) = yaw.sin_cos();
                let along = dx * s + dz * c;
                let lat = dx * c - dz * s;
                if along.abs() > half_l || lat.abs() > half_w {
                    return None;
                }
                Some((bottom, top))
            }
            WetShape::Column { r, bottom, top } => {
                if dx * dx + dz * dz >= r * r {
                    return None;
                }
                Some((bottom, top))
            }
        }
    }

    /// Radius of a circle around the centre that holds the footprint.
    pub fn reach(&self) -> f32 {
        match *self {
            WetShape::Sphere { r, .. } => r,
            WetShape::Box { half_l, half_w, .. } => half_l.hypot(half_w),
            WetShape::Column { r, .. } => r,
        }
    }

    /// Plan area of the footprint, m².
    pub fn plan_area(&self) -> f32 {
        match *self {
            WetShape::Sphere { r, .. } | WetShape::Column { r, .. } => std::f32::consts::PI * r * r,
            WetShape::Box { half_l, half_w, .. } => 4.0 * half_l * half_w,
        }
    }
}

/// One body handed to the sheet for the next step.
#[derive(Clone, Copy, Debug)]
pub struct WetBody {
    /// Caller's id, echoed in the load.
    pub id: u32,
    pub x: f32,
    pub z: f32,
    pub shape: WetShape,
    /// Drag impulse the body took from the water since the last step, N·s.
    /// The water gets the opposite.
    pub impulse_x: f32,
    pub impulse_z: f32,
}

/// What the sheet reports for one body after a step.
#[derive(Clone, Copy, Debug, Default)]
pub struct WetLoad {
    pub id: u32,
    /// Free surface around the body, world Y. Bed where the ring is dry.
    pub surface_y: f32,
    /// Bed under the ring, world Y.
    pub bed_y: f32,
    /// Mean fluid depth on the ring, metres (water plus slurry).
    pub depth_m: f32,
    /// Share of that depth that is slurry, 0..1.
    pub slurry_share: f32,
    /// Depth-weighted current on the ring, m/s.
    pub flow_u: f32,
    pub flow_v: f32,
    /// Fluid volume the body held out of its columns on this step, m³.
    pub displaced_m3: f32,
    /// True when any wet cell was found on the ring.
    pub wet: bool,
}

impl WetLoad {
    /// Bulk density of the fluid around the body, kg/m³.
    pub fn rho(&self) -> f32 {
        WATER_RHO + (SLURRY_RHO - WATER_RHO) * self.slurry_share.clamp(0.0, 1.0)
    }
}

#[derive(Default)]
pub(super) struct WetState {
    bodies: Vec<WetBody>,
    /// Column each body filled on the last step, per coarse cell, metres.
    /// The free surface over a body is bed + fluid + this.
    occ: Vec<f32>,
    occ_cells: Vec<u32>,
    pub(super) lift_c: Vec<(u32, f32)>,
    pub(super) lift_f: Vec<(u32, f32)>,
    displaced: Vec<f32>,
    loads: Vec<WetLoad>,
    /// Where each body was when it last moved, and how fast: the sheet
    /// derives body speed from the positions it is handed.
    track: Vec<Track>,
}

#[derive(Clone, Copy, Debug)]
struct Track {
    id: u32,
    pos: [f32; 3],
    vel: [f32; 3],
    /// Seconds since `pos` last changed.
    since: f32,
}

impl WetState {
    pub(super) fn active(&self) -> bool {
        !self.bodies.is_empty() || !self.occ_cells.is_empty()
    }

    pub(super) fn forget_cells(&mut self) {
        for &i in &self.occ_cells {
            if let Some(o) = self.occ.get_mut(i as usize) {
                *o = 0.0;
            }
        }
        self.occ_cells.clear();
        self.lift_c.clear();
        self.lift_f.clear();
        self.track.clear();
    }
}

impl FluidCore {
    /// Bodies for the next step. An empty slice takes them all out.
    pub fn set_wet_bodies(&mut self, bodies: &[WetBody]) {
        self.wet.bodies.clear();
        self.wet.bodies.extend_from_slice(bodies);
    }

    /// Loads from the step that just ran, one per body, in the order given.
    pub fn wet_loads(&self) -> &[WetLoad] {
        &self.wet.loads
    }

    /// Free surface and current at a point without a body: what a body
    /// that is not in the sheet yet would meet.
    pub fn surface_at(&self, x: f32, z: f32) -> (f32, f32, f32, f32) {
        let s = self.sample(x, z);
        let occ = self
            .coarse_ij(x, z)
            .and_then(|(i, j)| self.wet.occ.get(j * WINDOW_RES + i).copied())
            .unwrap_or(0.0);
        (s.bed_m + s.water_m + s.slurry_m + occ, s.bed_m, s.u, s.v)
    }

    /// Fill the columns, before `refine`. Writes the coarse lift and hands
    /// each body's drag impulse to the water around it.
    pub(super) fn wet_prepare_coarse(&mut self) {
        let n = WINDOW_RES * WINDOW_RES;
        if self.wet.occ.len() != n {
            self.wet.occ = vec![0.0; n];
        }
        let nb = self.wet.bodies.len();
        self.wet.displaced.clear();
        self.wet.displaced.resize(nb, 0.0);
        let mut lift: Vec<(u32, f32)> = Vec::new();
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let sub = COARSE_SAMPLES as f32;
        for (bi, b) in self.wet.bodies.iter().enumerate() {
            let reach = b.shape.reach();
            let (i0, i1) = index_span(self.origin[0], COARSE_CELL_M, WINDOW_RES, b.x, reach);
            let (j0, j1) = index_span(self.origin[1], COARSE_CELL_M, WINDOW_RES, b.z, reach);
            // The free surface the body sits in is the one around it, read
            // on the last step. A column's own depth under the body is what
            // the lift is changing, so reading it back would ring.
            let ring = self
                .wet
                .loads
                .iter()
                .find(|l| l.id == b.id && l.wet)
                .map(|l| l.surface_y);
            let mut vol = 0.0f32;
            for j in j0..j1 {
                for i in i0..i1 {
                    let idx = j * WINDOW_RES + i;
                    let bed = self.bed[idx];
                    let eta = ring.unwrap_or(bed + self.water[idx] + self.slurry[idx] + self.wet.occ[idx]);
                    let x0 = self.origin[0] + i as f32 * COARSE_CELL_M;
                    let z0 = self.origin[1] + j as f32 * COARSE_CELL_M;
                    let mut o = 0.0f32;
                    for sj in 0..COARSE_SAMPLES {
                        for si in 0..COARSE_SAMPLES {
                            let x = x0 + (si as f32 + 0.5) / sub * COARSE_CELL_M;
                            let z = z0 + (sj as f32 + 0.5) / sub * COARSE_CELL_M;
                            if let Some((lo, hi)) = b.shape.span(x - b.x, z - b.z) {
                                o += (hi.min(eta) - lo.max(bed)).max(0.0);
                            }
                        }
                    }
                    o /= sub * sub;
                    if o > 1.0e-6 {
                        lift.push((idx as u32, o));
                        vol += o * area;
                    }
                }
            }
            self.wet.displaced[bi] = vol;
        }
        // Two bodies in one column add up.
        lift.sort_unstable_by_key(|e| e.0);
        let mut merged: Vec<(u32, f32)> = Vec::with_capacity(lift.len());
        for (i, o) in lift {
            match merged.last_mut() {
                Some(last) if last.0 == i => last.1 += o,
                _ => merged.push((i, o)),
            }
        }
        for &i in &self.wet.occ_cells {
            self.wet.occ[i as usize] = 0.0;
        }
        self.wet.occ_cells.clear();
        for &(i, o) in &merged {
            self.wet.occ[i as usize] = o;
            self.wet.occ_cells.push(i);
        }
        self.wet.lift_c = merged;

        // Drag impulse into the water: the opposite of what the body took,
        // spread over the wet cells around it by mass.
        for b in &self.wet.bodies {
            let jx = -b.impulse_x;
            let jz = -b.impulse_z;
            if !(jx.is_finite() && jz.is_finite()) || jx.abs() + jz.abs() < 1.0e-9 {
                continue;
            }
            let reach = b.shape.reach() + COARSE_CELL_M;
            let (i0, i1) = index_span(self.origin[0], COARSE_CELL_M, WINDOW_RES, b.x, reach);
            let (j0, j1) = index_span(self.origin[1], COARSE_CELL_M, WINDOW_RES, b.z, reach);
            let mut mass = 0.0f32;
            for j in j0..j1 {
                for i in i0..i1 {
                    let (x, z) = self.coarse_xz(i, j);
                    if (x - b.x).hypot(z - b.z) > reach {
                        continue;
                    }
                    let h = self.water[j * WINDOW_RES + i];
                    if h > RING_MIN_WATER_M {
                        mass += WATER_RHO * h * area;
                    }
                }
            }
            if mass <= 1.0e-6 {
                continue;
            }
            let du = jx / mass;
            let dv = jz / mass;
            for j in j0..j1 {
                for i in i0..i1 {
                    let (x, z) = self.coarse_xz(i, j);
                    if (x - b.x).hypot(z - b.z) > reach {
                        continue;
                    }
                    let idx = j * WINDOW_RES + i;
                    if self.water[idx] > RING_MIN_WATER_M {
                        self.uw[idx] += du;
                        self.vw[idx] += dv;
                    }
                }
            }
        }
    }

    /// Fill the 2 cm columns, after `refine`, on cells the patch holds. A
    /// body does not pull the patch over itself: the patch's rim is open to
    /// dry cells, which drains deep water into the ring on every step. Only
    /// a body that is under a tire's patch anyway is filled here.
    pub(super) fn wet_prepare_fine(&mut self) {
        self.wet.lift_f.clear();
        if !self.fine_on {
            return;
        }
        let (col, row) = self.fine_coarse_axes();
        for b in &self.wet.bodies {
            let reach = b.shape.reach();
            let (i0, i1) = index_span(self.fine_origin[0], FINE_CELL_M, FINE_RES, b.x, reach);
            let (j0, j1) = index_span(self.fine_origin[1], FINE_CELL_M, FINE_RES, b.z, reach);
            for j in j0..j1 {
                let cj = row[j];
                if cj == u16::MAX {
                    continue;
                }
                for i in i0..i1 {
                    let ci = col[i];
                    if ci == u16::MAX {
                        continue;
                    }
                    let cidx = cj as usize * WINDOW_RES + ci as usize;
                    if self.fine_hold[cidx] != 1 {
                        continue;
                    }
                    let x = self.fine_origin[0] + (i as f32 + 0.5) * FINE_CELL_M;
                    let z = self.fine_origin[1] + (j as f32 + 0.5) * FINE_CELL_M;
                    let Some((lo, hi)) = b.shape.span(x - b.x, z - b.z) else {
                        continue;
                    };
                    let fidx = j * FINE_RES + i;
                    let bed = self.fine_bed[fidx];
                    let eta = self
                        .wet
                        .loads
                        .iter()
                        .find(|l| l.id == b.id && l.wet)
                        .map(|l| l.surface_y)
                        .unwrap_or(bed + self.fine_water[fidx] + self.fine_slurry[fidx] + self.wet.occ[cidx]);
                    let o = (hi.min(eta) - lo.max(bed)).max(0.0);
                    if o > 1.0e-6 {
                        self.wet.lift_f.push((fidx as u32, o));
                    }
                }
            }
        }
    }

    /// Read the ring around each body, after the step.
    pub(super) fn wet_finish(&mut self) {
        let mut loads = std::mem::take(&mut self.wet.loads);
        loads.clear();
        for (bi, b) in self.wet.bodies.iter().enumerate() {
            let inner = b.shape.reach();
            let outer = inner + RING_M;
            let (i0, i1) = index_span(self.origin[0], COARSE_CELL_M, WINDOW_RES, b.x, outer);
            let (j0, j1) = index_span(self.origin[1], COARSE_CELL_M, WINDOW_RES, b.z, outer);
            let mut eta = 0.0f32;
            let mut bed = 0.0f32;
            let mut nb = 0.0f32;
            let mut depth = 0.0f32;
            let mut slurry = 0.0f32;
            let mut mu = 0.0f32;
            let mut mv = 0.0f32;
            let mut nw = 0.0f32;
            let mut dry_bed = 0.0f32;
            let mut nd = 0.0f32;
            for j in j0..j1 {
                for i in i0..i1 {
                    let (x, z) = self.coarse_xz(i, j);
                    let d = (x - b.x).hypot(z - b.z);
                    if d > outer {
                        continue;
                    }
                    let idx = j * WINDOW_RES + i;
                    let h = self.water[idx];
                    let s = self.slurry[idx];
                    let inside = d < inner;
                    if h + s > RING_MIN_WATER_M && !inside {
                        eta += self.bed[idx] + h + s + self.wet.occ[idx];
                        bed += self.bed[idx];
                        depth += h + s;
                        slurry += s;
                        mu += self.uw[idx] * h + self.us[idx] * s;
                        mv += self.vw[idx] * h + self.vs[idx] * s;
                        nw += 1.0;
                        nb += 1.0;
                    } else if !inside {
                        dry_bed += self.bed[idx];
                        nd += 1.0;
                    }
                }
            }
            let mut load = WetLoad {
                id: b.id,
                displaced_m3: self.wet.displaced.get(bi).copied().unwrap_or(0.0),
                ..Default::default()
            };
            if nw > 0.0 {
                load.wet = true;
                load.surface_y = eta / nw;
                load.bed_y = bed / nb;
                load.depth_m = depth / nw;
                load.slurry_share = if depth > 0.0 { slurry / depth } else { 0.0 };
                load.flow_u = mu / depth.max(1.0e-6);
                load.flow_v = mv / depth.max(1.0e-6);
            } else if nd > 0.0 {
                load.bed_y = dry_bed / nd;
                load.surface_y = load.bed_y;
            } else {
                let s = self.sample(b.x, b.z);
                load.bed_y = s.bed_m;
                load.surface_y = s.bed_m + s.water_m + s.slurry_m;
                load.depth_m = s.water_m + s.slurry_m;
                load.wet = load.depth_m > RING_MIN_WATER_M;
            }
            loads.push(load);
        }
        self.wet.loads = loads;
        if self.wet.bodies.is_empty() {
            self.wet.forget_cells();
        }
    }
}

/// Slowest wave on water (capillary-gravity minimum, 0.23 m/s). A body moving
/// slower than this through the surface makes no breaking bow and no foam.
const WAVE_MIN_M_S: f32 = 0.23;
/// Body Froude number `U / sqrt(g D)` where the bow wave at the waterline
/// starts to break, and where it is fully broken and ventilated (surface
/// piercing cylinders, Chaplin & Teigen 2003: run-up sheet breaks near 0.5,
/// the lee cavity opens by about 1.4).
const BOW_FR_START: f32 = 0.5;
const BOW_FR_FULL: f32 = 1.4;
/// Entry Froude number `w / sqrt(g D)` where an impact throws a crown.
const ENTRY_FR_START: f32 = 0.3;
const ENTRY_FR_FULL: f32 = 1.2;
/// Speed past which the waterline sheds drops, m/s.
const SPLASH_M_S: f32 = 0.6;
/// Drop flux off a broken waterline, `Q = C · D² · (v − SPLASH_M_S)` m³/s:
/// a bow sheet about a tenth of the body wide and thick, a fifth of it
/// leaving as drops. A 13 cm leg swinging at 2 m/s sheds about 0.5 l/s.
const SPLASH_C: f32 = 0.02;
/// A position held longer than this means the body stopped.
const TRACK_STALE_S: f32 = 0.25;

/// Lowest point of a body and its centre height.
fn shape_heights(shape: &WetShape) -> (f32, f32) {
    match *shape {
        WetShape::Sphere { cy, r } => (cy - r, cy),
        WetShape::Box { bottom, top, .. } | WetShape::Column { bottom, top, .. } => (bottom, 0.5 * (bottom + top)),
    }
}

/// Radius of the body's cut through a free surface at `eta`; 0 when it does
/// not pierce it.
fn waterline_radius(shape: &WetShape, eta: f32) -> f32 {
    match *shape {
        WetShape::Sphere { cy, r } => {
            let d = eta - cy;
            if d.abs() >= r {
                0.0
            } else {
                (r * r - d * d).sqrt()
            }
        }
        WetShape::Box { half_l, half_w, bottom, top, .. } => {
            if eta <= bottom || eta >= top {
                0.0
            } else {
                (4.0 * half_l * half_w / std::f32::consts::PI).sqrt()
            }
        }
        WetShape::Column { r, bottom, top } => {
            if eta <= bottom || eta >= top {
                0.0
            } else {
                r
            }
        }
    }
}

impl FluidCore {
    /// Foam and drops where a body cuts the free surface, after
    /// [`Self::wet_finish`]. The bow of a moving body breaks once its Froude
    /// number `U / sqrt(g D)` passes the breaking range, so a slow buoy
    /// bobbing in place makes none and a striding leg or a rolling ball does.
    /// An entry through the surface throws a crown the same way. Foam goes
    /// into the tracer field from #104 and is carried by its pass. Drops are
    /// water taken from the ring cells, so the book closes on `spray_m3`.
    pub(super) fn wet_splash(&mut self, dt: f32) -> Vec<SprayEvent> {
        let mut out = Vec::new();
        if !(dt > 0.0) || !dt.is_finite() {
            return out;
        }
        let mut track = std::mem::take(&mut self.wet.track);
        let mut next: Vec<Track> = Vec::with_capacity(self.wet.bodies.len());
        let bodies = std::mem::take(&mut self.wet.bodies);
        for b in &bodies {
            let (lo, cy) = shape_heights(&b.shape);
            let pos = [b.x, cy, b.z];
            let t = match track.iter().position(|t| t.id == b.id) {
                Some(k) => {
                    let mut t = track.swap_remove(k);
                    t.since += dt;
                    if pos != t.pos {
                        for a in 0..3 {
                            t.vel[a] = (pos[a] - t.pos[a]) / t.since;
                        }
                        t.pos = pos;
                        t.since = 0.0;
                    } else if t.since > TRACK_STALE_S {
                        t.vel = [0.0; 3];
                    }
                    t
                }
                None => Track { id: b.id, pos, vel: [0.0; 3], since: 0.0 },
            };
            next.push(t);
            let Some(load) = self.wet.loads.iter().find(|l| l.id == b.id).copied() else {
                continue;
            };
            if !load.wet || load.depth_m < 0.004 {
                continue;
            }
            let eta = load.surface_y;
            let r_wl = waterline_radius(&b.shape, eta);
            if r_wl <= 1.0e-3 || !t.vel.iter().all(|v| v.is_finite()) {
                continue;
            }
            let d = 2.0 * r_wl;
            let ux = t.vel[0] - load.flow_u;
            let uz = t.vel[2] - load.flow_v;
            let u = ux.hypot(uz);
            // Entry: the lowest point just went under, moving down.
            let w = if t.vel[1] < 0.0 && lo < eta && lo > eta - d { -t.vel[1] } else { 0.0 };
            let gd = (GRAVITY * d).sqrt();
            let bow = if u > WAVE_MIN_M_S { smoothstep(BOW_FR_START, BOW_FR_FULL, u / gd) } else { 0.0 };
            let crown = if w > WAVE_MIN_M_S { smoothstep(ENTRY_FR_START, ENTRY_FR_FULL, w / gd) } else { 0.0 };
            let prod = bow.max(crown);
            if prod <= 0.0 {
                continue;
            }
            // The ring the waterline sits in, one cell either side.
            let inner = (r_wl - 0.5 * COARSE_CELL_M).max(0.0);
            let outer = r_wl + 1.35 * COARSE_CELL_M;
            let (i0, i1) = index_span(self.origin[0], COARSE_CELL_M, WINDOW_RES, b.x, outer);
            let (j0, j1) = index_span(self.origin[1], COARSE_CELL_M, WINDOW_RES, b.z, outer);
            let mut ring: Vec<(usize, f32, f32, f32)> = Vec::new();
            for j in j0..j1 {
                for i in i0..i1 {
                    let (x, z) = self.coarse_xz(i, j);
                    let dist = (x - b.x).hypot(z - b.z);
                    if dist < inner || dist > outer {
                        continue;
                    }
                    let idx = j * WINDOW_RES + i;
                    if self.water[idx] + self.slurry[idx] < 0.004 {
                        continue;
                    }
                    let band = 1.0 - ((dist - r_wl).abs() / (1.35 * COARSE_CELL_M)).clamp(0.0, 1.0);
                    self.add_foam(idx, prod * band.max(0.35), dt);
                    if self.foam_live_mark.get(idx) == Some(&0) && self.foam[idx] > 1.0e-4 {
                        self.foam_live_mark[idx] = 1;
                        self.foam_live.push(idx as u32);
                    }
                    ring.push((idx, x, z, dist));
                }
            }
            // Drops: the part of the broken bow sheet that leaves the water.
            let v = u.max(w);
            if v <= SPLASH_M_S || ring.is_empty() {
                continue;
            }
            let runup = (v * v / (2.0 * GRAVITY)).min(0.5);
            let total = SPLASH_C * prod * d * d * (v - SPLASH_M_S) * dt;
            let per = total / ring.len() as f32;
            let area = COARSE_CELL_M * COARSE_CELL_M;
            for &(idx, x, z, dist) in &ring {
                let take = (per / area).min(self.water[idx] * 0.04).min(0.004);
                if take * area < 1.0e-7 {
                    continue;
                }
                self.water[idx] -= take;
                let (nx, nz) = if dist > 1.0e-4 { ((x - b.x) / dist, (z - b.z) / dist) } else { (0.0, 0.0) };
                // Ahead of a moving body the sheet rides up its face; an entry
                // throws the crown straight out.
                let fwd = if u > 1.0e-3 { (nx * ux + nz * uz) / u } else { 0.0 };
                let out_s = 0.5 * v;
                out.push(SprayEvent {
                    x: b.x + nx * r_wl,
                    y: eta + 0.02,
                    z: b.z + nz * r_wl,
                    vx: t.vel[0] * 0.6 + nx * out_s,
                    vy: (2.0 * GRAVITY * runup).sqrt() * (0.6 + 0.4 * fwd.max(0.0)) + 0.4 * w,
                    vz: t.vel[2] * 0.6 + nz * out_s,
                    volume: take * area,
                    kind: 0,
                });
            }
        }
        self.wet.bodies = bodies;
        self.wet.track = next;
        out
    }
}

impl FluidCore {
    /// What the bed is made of at a point.
    pub fn grain_at(&self, x: f32, z: f32) -> Grain {
        match self.coarse_ij(x, z) {
            Some((i, j)) => Grain::from_u8(self.grain[j * WINDOW_RES + i]),
            None => Grain::Other,
        }
    }
}
