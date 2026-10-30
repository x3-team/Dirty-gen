//! Snow on top of the soil. It is its own column: bulk density, compaction,
//! and a geometric thickness. The soil bed under it is not cut here.
//!
//! A tire compacts the floor and plows what will not fit under the contact.
//! That plow lands in a smooth shoulder on each side, so the rut has rounded
//! berms rather than a slot with a wall. A turning chassis is a rigid body
//! about one centre, so the four tires draw four concentric circles. A boot
//! is a small punch: it sinks, and the displaced snow sits in a low rim.
//!
//! When the blanket is deeper than the Niva's clearance, the belly, the axle
//! tubes, and the transfer case plane the snow between the ruts. Each part
//! cuts at its own height. The spare is packed into the floor and shoved
//! forward and sideways into a bow. The contact force is kilonewtons.
//!
//! The closed ledger is mass. Compaction changes thickness and density.
//! Their product, summed over the grid plus anything that could not be
//! placed, stays at the blanket mass.

use glam::Vec2;

use crate::soil::{contact_length, MAP_RES, MAP_SIZE};
use crate::vehicle::{hardpoints, Vehicle, WHEEL_RADIUS, WHEEL_WIDTH};

/// Fresh settled snow. Light enough that a tire throws a shoulder.
pub const RHO_FRESH: f32 = 130.0;
/// Snow a tire has packed in the rut floor. Not ice.
pub const RHO_PACKED: f32 = 260.0;
/// Loose snow thrown onto the shoulder.
pub const RHO_BERM: f32 = 155.0;
/// Ice density. Solid volume is mass divided by this.
pub const RHO_ICE: f32 = 917.0;

/// Bekker modulus of fresh snow, pascals. Stiffness rises with density.
pub const SNOW_K: f32 = 15_000.0;
pub const SNOW_N: f32 = 1.2;

pub const BLANKET_M: f32 = 0.48;
/// Belly pan above the tire contact. Deeper snow is planed off.
pub const BELLY_CLEARANCE_M: f32 = 0.22;
/// Axle tubes hang a little lower than the pan.
const AXLE_CLEARANCE_M: f32 = 0.20;
/// Transfer case, the low point on the centreline.
const TCASE_CLEARANCE_M: f32 = 0.185;
/// Share of the shaved column packed into the floor. The rest is the bow.
const BELLY_PACK: f32 = 0.40;
/// The bow meets the pan on a sloping face. Drag is the longitudinal part
/// of the same Bekker pressure that unloads the wheels.
const BOW_FACE_DEG: f32 = 36.0;
/// Static load on one Niva tire, curb weight plus a driver.
pub const NIVA_WHEEL_LOAD_N: f32 = 2_850.0;
pub const FOOT_LOAD_N: f32 = 780.0;
pub const FOOT_LENGTH: f32 = 0.30;
pub const FOOT_WIDTH: f32 = 0.11;
/// Samples of the turn's radial profile. The picture follows this circle.
pub const ARC_BINS: usize = 32;

#[derive(Clone, Copy, Debug)]
pub struct SnowArc {
    pub center: Vec2,
    pub sweep: f32,
    pub r_lo: f32,
    pub r_hi: f32,
    pub dr: f32,
    /// Snow thickness at each radius, metres. Not the soil bed.
    pub depth: [f32; ARC_BINS],
}

#[derive(Clone, Copy, Debug)]
struct BermShape {
    peak: f32,
    sigma_in: f32,
    sigma_out: f32,
    along: f32,
}

const TIRE_BERM: BermShape = BermShape {
    // A narrow crest just outside the sidewall. Wider tails sink back into
    // the grid and the berm disappears. The tail still dies before the other tire.
    peak: 0.16,
    sigma_in: 0.09,
    sigma_out: 0.14,
    along: 0.18,
};

const FOOT_BERM: BermShape = BermShape {
    peak: 0.20,
    sigma_in: 0.08,
    sigma_out: 0.16,
    along: 0.12,
};

#[derive(Clone, Copy, Debug)]
pub struct Tire {
    pub radius: f32,
    pub width: f32,
    pub load_n: f32,
}

impl Tire {
    pub fn niva() -> Self {
        Self {
            radius: WHEEL_RADIUS,
            width: WHEEL_WIDTH,
            load_n: NIVA_WHEEL_LOAD_N,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DrivePose {
    pub origin: Vec2,
    pub forward: Vec2,
}

#[derive(Clone, Copy, Debug)]
pub struct TurnOut {
    pub center: Vec2,
    pub radii: [f32; 4],
    pub mid_dir: [Vec2; 4],
    pub end: DrivePose,
}

#[derive(Clone, Debug)]
pub struct SnowField {
    pub res: usize,
    pub size: f32,
    pub cell: f32,
    /// Geometric thickness, metres. Empty cells are zero.
    pub depth: Vec<f32>,
    /// Bulk density, kg/m³. Meaningful where `depth` is positive.
    pub density: Vec<f32>,
    /// Mass that had nowhere to land, kilograms.
    pub held: f32,
    /// Mass the underbody has shaved and not yet pushed into the bow.
    plow_kg: f64,
    /// Sum of vertical contact force on stamps where the pan is planted, N.
    belly_normal_n: f64,
    /// Sum of longitudinal contact force on those stamps, N.
    belly_drag_n: f64,
    belly_plants: u32,
    blanket: f32,
    initial_mass: f64,
    /// Radial profile of the turn, once the crest has been laid on the circle.
    pub arc: Option<SnowArc>,
}

#[derive(Clone, Copy, Debug)]
pub struct SnowReport {
    pub blanket_m: f32,
    pub rho_fresh: f32,
    pub rho_packed: f32,
    pub rho_berm: f32,
    pub mass0_kg: f64,
    pub mass1_kg: f64,
    pub held_kg: f64,
    pub geom0_m3: f64,
    pub geom1_m3: f64,
    pub solid0_m3: f64,
    pub solid1_m3: f64,
    pub rut_drop_m: f32,
    pub berm_rise_m: f32,
    pub berm_round: f32,
    pub rut_rho: f32,
    pub berm_rho: f32,
    pub turn_radii_m: [f32; 4],
    pub turn_trough_m: [f32; 4],
    pub foot_fresh_m: f32,
    pub foot_packed_m: f32,
    /// Mean longitudinal force while the underbody is planted, kilonewtons.
    pub belly_drag_kn: f32,
    /// Mean upward force the snow takes off the wheels, kilonewtons.
    pub belly_unload_kn: f32,
    /// Snow left between the ruts after the pan has planed it.
    pub shelf_m: f32,
    /// Highest sample minus lowest across that shelf.
    pub shelf_flat_m: f32,
    pub shelf_rho: f32,
    /// How far the bow stands above the blanket.
    pub bow_rise_m: f32,
}

#[derive(Clone, Debug)]
pub struct SnowScene {
    pub snow: SnowField,
    pub pose: DrivePose,
    pub yaw: f32,
    pub straight_x: f32,
    pub straight_z: f32,
    pub turn_center: Vec2,
    pub turn_radii: [f32; 4],
    pub turn_dir: [Vec2; 4],
    pub foot_at: Vec2,
    pub report: SnowReport,
}

impl SnowField {
    pub fn blanket(depth: f32) -> Self {
        let res = MAP_RES;
        let n = res * res;
        let size = MAP_SIZE;
        let cell = size / res as f32;
        let depth = depth.max(0.0);
        let mut field = Self {
            res,
            size,
            cell,
            depth: vec![depth; n],
            density: vec![RHO_FRESH; n],
            held: 0.0,
            plow_kg: 0.0,
            belly_normal_n: 0.0,
            belly_drag_n: 0.0,
            belly_plants: 0,
            blanket: depth,
            initial_mass: 0.0,
            arc: None,
        };
        field.initial_mass = field.ledger_kg();
        field
    }

    pub fn blanket_m(&self) -> f32 {
        self.blanket
    }

    fn area(&self) -> f32 {
        self.cell * self.cell
    }

    /// Column mass plus mass waiting to be placed.
    pub fn ledger_kg(&self) -> f64 {
        let area = self.area() as f64;
        let mut mass = 0.0f64;
        for i in 0..self.depth.len() {
            mass += self.depth[i] as f64 * self.density[i] as f64 * area;
        }
        mass + self.held as f64 + self.plow_kg
    }

    pub fn geometric_m3(&self) -> f64 {
        let area = self.area() as f64;
        self.depth.iter().map(|d| *d as f64 * area).sum()
    }

    pub fn solid_m3(&self) -> f64 {
        self.ledger_kg() / RHO_ICE as f64
    }

    pub fn compaction(&self, idx: usize) -> f32 {
        if self.depth[idx] < 1.0e-4 {
            return 0.0;
        }
        ((self.density[idx] - RHO_FRESH) / (RHO_PACKED - RHO_FRESH)).clamp(0.0, 1.0)
    }

    pub fn idx_at(&self, x: f32, z: f32) -> Option<usize> {
        let half = self.size * 0.5;
        if x < -half || z < -half || x >= half || z >= half {
            return None;
        }
        let i = ((x + half) / self.cell).floor() as isize;
        let j = ((z + half) / self.cell).floor() as isize;
        if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
            return None;
        }
        Some(j as usize * self.res + i as usize)
    }

    fn cell_xz(&self, idx: usize) -> Vec2 {
        let i = idx % self.res;
        let j = idx / self.res;
        let half = self.size * 0.5;
        Vec2::new(
            (i as f32 + 0.5) * self.cell - half,
            (j as f32 + 0.5) * self.cell - half,
        )
    }

    pub fn depth_nearest(&self, x: f32, z: f32) -> f32 {
        self.idx_at(x, z).map(|i| self.depth[i]).unwrap_or(0.0)
    }

    pub fn density_nearest(&self, x: f32, z: f32) -> f32 {
        self.idx_at(x, z)
            .map(|i| self.density[i])
            .unwrap_or(RHO_FRESH)
    }

    pub fn depth_at(&self, x: f32, z: f32) -> f32 {
        let half = self.size * 0.5;
        let u = ((x + half) / self.cell - 0.5).clamp(0.0, self.res as f32 - 1.001);
        let v = ((z + half) / self.cell - 0.5).clamp(0.0, self.res as f32 - 1.001);
        let i = u.floor() as usize;
        let j = v.floor() as usize;
        let tx = u - i as f32;
        let ty = v - j as f32;
        let i1 = (i + 1).min(self.res - 1);
        let j1 = (j + 1).min(self.res - 1);
        let d00 = self.depth[j * self.res + i];
        let d10 = self.depth[j * self.res + i1];
        let d01 = self.depth[j1 * self.res + i];
        let d11 = self.depth[j1 * self.res + i1];
        let a = d00 + (d10 - d00) * tx;
        let b = d01 + (d11 - d01) * tx;
        a + (b - a) * ty
    }

    /// Add the snow thickness onto the soil height the picture reads.
    /// The bed, the placed spoil, and the slurry column are left alone.
    pub fn paint_height(&self, height: &mut [f32], bed: &[f32], placed: &[f32], slurry: &[f32]) {
        for i in 0..self.depth.len() {
            height[i] = bed[i] + placed[i] + slurry[i] + self.depth[i];
        }
    }

    pub fn drive_straight(
        &mut self,
        origin0: Vec2,
        forward: Vec2,
        distance: f32,
        wheels: &[Vec2],
        tire: Tire,
    ) -> DrivePose {
        let fwd = forward.normalize_or_zero();
        let right = Vec2::new(fwd.y, -fwd.x);
        let step = (self.cell * 0.42).max(0.06);
        let n = ((distance / step).ceil() as i32).max(1);
        let mut origin = origin0;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            origin = origin0 + fwd * (distance * t);
            self.stamp_axle(origin, fwd, right, wheels, tire);
            // The pan follows the tires. The arc smoother never rewrites this
            // straight, so the shelf between the ruts stays planed.
            self.scrape_underbody(origin, fwd, right, wheels, false);
        }
        DrivePose {
            origin,
            forward: fwd,
        }
    }

    /// `radius` is the rear-axle radius, positive to the left. The chassis
    /// stays rigid about that centre. Rear tires hold `radius ± track/2`.
    /// Front tires sit a wheelbase ahead, so their circles are larger.
    pub fn drive_turn(
        &mut self,
        origin0: Vec2,
        forward: Vec2,
        radius: f32,
        sweep: f32,
        wheels: &[Vec2],
        tire: Tire,
    ) -> TurnOut {
        let fwd0 = forward.normalize_or_zero();
        let right0 = Vec2::new(fwd0.y, -fwd0.x);
        let rear_z = rear_axle_z(wheels);
        let rear0 = origin0 + fwd0 * rear_z;
        let center = rear0 - right0 * radius;
        let arm0 = rear0 - center;
        let arc = radius.abs() * sweep.abs();
        let step = (self.cell * 0.42).max(0.06);
        let n = ((arc / step).ceil() as i32).max(1);
        let dtheta = sweep / n as f32;
        let mut radii = [0.0f32; 4];
        let mut mid_dir = [Vec2::ZERO; 4];
        let mid = n / 2;
        let mut end_origin = origin0;
        let mut end_fwd = fwd0;
        for i in 0..=n {
            let arm = rot(arm0, dtheta * i as f32);
            let rear = center + arm;
            let fwd = Vec2::new(-arm.y, arm.x).normalize_or_zero();
            let right = Vec2::new(fwd.y, -fwd.x);
            let origin = rear - fwd * rear_z;
            if i == mid {
                for (w, wheel) in wheels.iter().take(4).enumerate() {
                    let p = origin + right * wheel.x + fwd * wheel.y;
                    let rel = p - center;
                    radii[w] = rel.length();
                    mid_dir[w] = rel.normalize_or_zero();
                }
            }
            self.stamp_axle(origin, fwd, right, wheels, tire);
            end_origin = origin;
            end_fwd = fwd;
        }
        TurnOut {
            center,
            radii,
            mid_dir,
            end: DrivePose {
                origin: end_origin,
                forward: end_fwd,
            },
        }
    }

    /// Sink one boot. The returned metres are how far the centre cell dropped.
    pub fn stamp_foot(&mut self, at: Vec2, forward: Vec2, length: f32, width: f32, load_n: f32) -> f32 {
        let fwd = forward.normalize_or_zero();
        let right = Vec2::new(fwd.y, -fwd.x);
        let before = self.depth_nearest(at.x, at.y);
        let (depth, rho) = self.mean_column(at, fwd, right, width * 0.5, length * 0.5);
        if depth < 0.012 {
            return 0.0;
        }
        let sole = (length * width * 0.70).max(1.0e-4);
        let sink = snow_sinkage(load_n / sole, rho, depth).min(depth * 0.88);
        if sink < 0.004 {
            return 0.0;
        }
        self.stamp_bowl(
            at,
            fwd,
            right,
            width * 0.5,
            length * 0.5,
            sink,
            FOOT_BERM,
            RHO_BERM,
            false,
        );
        (before - self.depth_nearest(at.x, at.y)).max(0.0)
    }

    pub fn walk(
        &mut self,
        origin: Vec2,
        forward: Vec2,
        steps: i32,
        stride: f32,
    ) -> (Vec2, f32) {
        let fwd = forward.normalize_or_zero();
        let right = Vec2::new(fwd.y, -fwd.x);
        let mut sink = 0.0f32;
        let mut last = origin;
        let n = steps.max(1);
        for i in 0..n {
            let side = if i % 2 == 0 { -1.0 } else { 1.0 };
            let at = origin + fwd * (stride * i as f32) + right * (0.09 * side);
            sink += self.stamp_foot(at, fwd, FOOT_LENGTH, FOOT_WIDTH, FOOT_LOAD_N);
            last = at;
        }
        (last, sink / n as f32)
    }

    /// Collapse a disc to `rho`, keeping the mass already in those cells.
    pub fn pack_disc(&mut self, center: Vec2, radius: f32, rho: f32) {
        let rho = rho.clamp(RHO_FRESH, RHO_PACKED);
        let reach = radius + self.cell;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.y - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.y + reach) + half) / self.cell).floor() as isize;
        let area = self.area();
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let p = self.cell_xz(idx);
                if (p - center).length() > radius {
                    continue;
                }
                let mass = self.depth[idx] * self.density[idx] * area;
                if mass <= 1.0e-8 {
                    continue;
                }
                self.depth[idx] = mass / (rho * area);
                self.density[idx] = rho;
            }
        }
    }

    /// Radius of the deepest snow near `expect` along `dir` from `center`.
    pub fn trough_radius(&self, center: Vec2, dir: Vec2, expect: f32) -> (f32, f32) {
        let dir = dir.normalize_or_zero();
        let mut best_r = expect;
        let mut best_d = f32::MAX;
        // Stay on this tire. The next concentric circle is further out.
        let mut r = (expect - 0.28).max(0.15);
        let end = expect + 0.28;
        while r <= end {
            let p = center + dir * r;
            let d = self.depth_nearest(p.x, p.y);
            if d < best_d {
                best_d = d;
                best_r = r;
            }
            r += self.cell * 0.45;
        }
        (best_r, best_d)
    }

    fn stamp_axle(&mut self, origin: Vec2, fwd: Vec2, right: Vec2, wheels: &[Vec2], tire: Tire) {
        for wheel in wheels {
            let p = origin + right * wheel.x + fwd * wheel.y;
            self.stamp_tire(p, fwd, right, tire);
        }
    }

    /// Replay the turn and plane virgin snow between the circles. Berm crests
    /// stay. The arc smoother has already run, so this cut is the one that
    /// remains. Thrown mass joins the bow.
    fn scrape_along_turn(
        &mut self,
        origin0: Vec2,
        forward: Vec2,
        radius: f32,
        sweep: f32,
        wheels: &[Vec2],
    ) {
        let fwd0 = forward.normalize_or_zero();
        let right0 = Vec2::new(fwd0.y, -fwd0.x);
        let rear_z = rear_axle_z(wheels);
        let rear0 = origin0 + fwd0 * rear_z;
        let center = rear0 - right0 * radius;
        let arm0 = rear0 - center;
        let arc = radius.abs() * sweep.abs();
        let step = (self.cell * 0.42).max(0.06);
        let n = ((arc / step).ceil() as i32).max(1);
        let dtheta = sweep / n as f32;
        for i in 0..=n {
            let arm = rot(arm0, dtheta * i as f32);
            let rear = center + arm;
            let fwd = Vec2::new(-arm.y, arm.x).normalize_or_zero();
            let right = Vec2::new(fwd.y, -fwd.x);
            let origin = rear - fwd * rear_z;
            self.scrape_underbody(origin, fwd, right, wheels, true);
        }
    }

    /// Cut every underbody part to its own clearance above the tire contact.
    /// `keep_berms` leaves a shoulder that is already standing above the
    /// blanket, so a turn crest is not shaved back into a tooth.
    fn scrape_underbody(
        &mut self,
        origin: Vec2,
        fwd: Vec2,
        right: Vec2,
        wheels: &[Vec2],
        keep_berms: bool,
    ) {
        let contact = self.tire_floor(origin, fwd, right, wheels);
        let parts = under_parts();
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        for part in parts {
            for (x, z) in [
                (part.x0, part.z0),
                (part.x0, part.z1),
                (part.x1, part.z0),
                (part.x1, part.z1),
            ] {
                let p = origin + right * x + fwd * z;
                min_x = min_x.min(p.x);
                max_x = max_x.max(p.x);
                min_z = min_z.min(p.y);
                max_z = max_z.max(p.y);
            }
        }
        let pad = self.cell;
        let half = self.size * 0.5;
        let i0 = (((min_x - pad) + half) / self.cell).floor() as isize;
        let i1 = (((max_x + pad) + half) / self.cell).floor() as isize;
        let j0 = (((min_z - pad) + half) / self.cell).floor() as isize;
        let j1 = (((max_z + pad) + half) / self.cell).floor() as isize;
        let area = self.area();
        let mut cuts: Vec<(usize, f32, f32)> = Vec::new();
        let mut normal = 0.0f32;
        let mut thrown = 0.0f64;
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let p = self.cell_xz(idx);
                let rel = p - origin;
                let along = rel.dot(fwd);
                let lat = rel.dot(right);
                let mut clearance = f32::MAX;
                let mut hit = false;
                for part in parts {
                    if lat < part.x0 || lat > part.x1 || along < part.z0 || along > part.z1 {
                        continue;
                    }
                    hit = true;
                    clearance = clearance.min(part.clearance);
                }
                if !hit {
                    continue;
                }
                let depth = self.depth[idx];
                if keep_berms && depth > self.blanket + 0.03 {
                    continue;
                }
                let plane = contact + clearance;
                if depth <= plane + 0.008 {
                    continue;
                }
                let rho = self.density[idx].max(1.0);
                let mass = depth * rho * area;
                let excess_h = depth - plane;
                let excess_m = excess_h * rho * area;
                let mut kept = plane * rho * area + excess_m * BELLY_PACK;
                let cap = RHO_PACKED * plane.max(1.0e-4) * area;
                if kept > cap {
                    kept = cap;
                }
                let throw = (mass - kept).max(0.0);
                thrown += throw as f64;
                cuts.push((idx, plane, kept));
                normal += snow_pressure(excess_h, rho) * area;
            }
        }
        for (idx, h, m) in cuts {
            self.write_column(idx, h, m);
        }
        self.plow_kg += thrown;
        // A lip in already-planed snow is tens of newtons. The plant is the
        // stamp where the whole pan is still in the blanket.
        if normal > 1_000.0 {
            let drag = normal * BOW_FACE_DEG.to_radians().tan();
            self.belly_normal_n += normal as f64;
            self.belly_drag_n += drag as f64;
            self.belly_plants += 1;
        }
    }

    fn tire_floor(&self, origin: Vec2, fwd: Vec2, right: Vec2, wheels: &[Vec2]) -> f32 {
        let mut acc = 0.0f32;
        let mut n = 0.0f32;
        for w in wheels.iter().take(4) {
            let p = origin + right * w.x + fwd * w.y;
            let mut best = self.depth_nearest(p.x, p.y);
            for (du, dv) in [(-0.10, 0.0), (0.10, 0.0), (0.0, -0.10), (0.0, 0.10)] {
                let q = p + right * du + fwd * dv;
                best = best.min(self.depth_nearest(q.x, q.y));
            }
            acc += best;
            n += 1.0;
        }
        if n > 0.0 {
            acc / n
        } else {
            self.blanket
        }
    }

    /// Push the shaved mass into a bow ahead of the nose and out to the
    /// sides of the straight shelf. The measurement line and the packed
    /// ruts are left alone.
    fn deposit_plow(&mut self, pose: DrivePose, lane_x: f32) {
        let mass_f = self.plow_kg;
        self.plow_kg = 0.0;
        if mass_f <= 1.0e-6 {
            return;
        }
        let mass = mass_f as f32;
        self.held += (mass_f - mass as f64) as f32;
        // The nose keeps a wide bow at the end of the turn. The rest is the
        // shove a rear camera can see: mounds beside the shelf and ahead of
        // the parked Niva, off the arc profile so the mesh does not erase them.
        let nose = mass * 0.50;
        let wings = mass - nose;
        self.deposit_nose_bow(pose, nose);
        self.deposit_lane_wings(lane_x, wings);
    }

    fn deposit_nose_bow(&mut self, pose: DrivePose, mass: f32) {
        let fwd = pose.forward.normalize_or_zero();
        let right = Vec2::new(fwd.y, -fwd.x);
        let origin = pose.origin;
        let reach = 3.1_f32;
        let half = self.size * 0.5;
        let i0 = (((origin.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((origin.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((origin.y - reach) + half) / self.cell).floor() as isize;
        let j1 = (((origin.y + reach) + half) / self.cell).floor() as isize;
        let mut weights: Vec<(usize, f32)> = Vec::new();
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let p = self.cell_xz(idx);
                let rel = p - origin;
                let along = rel.dot(fwd);
                let lat = rel.dot(right);
                // A wide, flat-topped pile. A tight gaussian stacked the
                // whole plow into a spire on one cell.
                if !(1.20..=3.85).contains(&along) || lat.abs() > 1.25 {
                    continue;
                }
                let fore = ((along - 2.35) / 1.25).abs();
                let side = (lat / 1.15).abs();
                if fore > 1.0 || side > 1.0 {
                    continue;
                }
                if self.depth[idx] < self.blanket * 0.45 {
                    continue;
                }
                let w = (1.0 - fore * fore) * (1.0 - 0.25 * side * side);
                if w > 0.05 {
                    weights.push((idx, w));
                }
            }
        }
        self.deposit(&weights, mass, RHO_BERM);
    }

    /// Snow the pan threw, sitting beside the planed lane and just ahead of
    /// where the proof parks the Niva. Off the rut cross-section.
    fn deposit_lane_wings(&mut self, lane_x: f32, mass: f32) {
        // Tight mounds. A wide lobe only dusted the berm and the shelf still
        // looked like an untouched crown. These sit on raw grid: beside the
        // planed lane, and ahead of the nose outside the turn annulus.
        let lobes = [
            Vec2::new(lane_x - 1.70, -5.55),
            Vec2::new(lane_x + 1.62, -5.55),
            Vec2::new(lane_x - 2.55, 0.20),
            Vec2::new(lane_x + 2.40, 0.35),
        ];
        let mut weights: Vec<(usize, f32)> = Vec::new();
        let reach = 0.55_f32;
        let half = self.size * 0.5;
        for lobe in lobes {
            let i0 = (((lobe.x - reach) + half) / self.cell).floor() as isize;
            let i1 = (((lobe.x + reach) + half) / self.cell).floor() as isize;
            let j0 = (((lobe.y - reach) + half) / self.cell).floor() as isize;
            let j1 = (((lobe.y + reach) + half) / self.cell).floor() as isize;
            for j in j0..=j1 {
                for i in i0..=i1 {
                    if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
                        continue;
                    }
                    let idx = j as usize * self.res + i as usize;
                    let p = self.cell_xz(idx);
                    // The berm-rise probe sits on this line.
                    if (p.y + 3.42).abs() < 0.35 {
                        continue;
                    }
                    let d = (p - lobe).length();
                    if d > reach {
                        continue;
                    }
                    if self.depth[idx] < self.blanket * 0.45 {
                        continue;
                    }
                    // Stay off the planed shelf and the tire grooves.
                    if (p.x - lane_x).abs() < 1.18 {
                        continue;
                    }
                    // Boot prints run outboard of the right berm.
                    if (p.x - (lane_x + 2.15)).abs() < 0.36 && (-6.2..-1.2).contains(&p.y) {
                        continue;
                    }
                    let u = d / reach;
                    let w = 1.0 - u * u;
                    if w > 0.08 {
                        weights.push((idx, w));
                    }
                }
            }
        }
        self.deposit(&weights, mass, RHO_BERM);
    }

    /// The picture of the turn follows the radial profile. Where the pan
    /// lowered the grid, lower that bin too. Berm bins stay on the circle.
    fn refit_arc_shelf(&mut self) {
        let Some(arc) = self.arc else {
            return;
        };
        let mut depth = arc.depth;
        for i in 0..ARC_BINS {
            let radius = arc.r_lo + arc.dr * i as f32;
            let mut samples: Vec<f32> = Vec::new();
            let mut theta = 0.50_f32;
            while theta < 1.32 {
                let dir = Vec2::new(theta.cos(), theta.sin());
                let p = arc.center + dir * radius;
                samples.push(self.depth_nearest(p.x, p.y));
                theta += 0.07;
            }
            if samples.len() < 4 {
                continue;
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let med = samples[samples.len() / 2];
            if med + 0.05 < depth[i] {
                depth[i] = med;
            }
        }
        self.arc = Some(SnowArc { depth, ..arc });
    }

    fn belly_drag_kn(&self) -> f32 {
        if self.belly_plants == 0 {
            0.0
        } else {
            (self.belly_drag_n / self.belly_plants as f64 / 1000.0) as f32
        }
    }

    fn belly_unload_kn(&self) -> f32 {
        if self.belly_plants == 0 {
            0.0
        } else {
            (self.belly_normal_n / self.belly_plants as f64 / 1000.0) as f32
        }
    }

    fn stamp_tire(&mut self, at: Vec2, fwd: Vec2, right: Vec2, tire: Tire) {
        let (depth, rho) = self.mean_column(at, fwd, right, tire.width * 0.45, 0.16);
        if depth < 0.015 {
            return;
        }
        let mut sink = 0.05f32;
        for _ in 0..5 {
            let len = contact_length(tire.radius, sink.max(0.004)).max(0.05);
            let pressure = tire.load_n / (len * tire.width).max(1.0e-4);
            sink = snow_sinkage(pressure, rho, depth);
        }
        let keep = (depth * 0.16).clamp(0.04, depth);
        sink = sink.min((depth - keep).max(0.0));
        if sink < 0.006 {
            return;
        }
        let half_l = (contact_length(tire.radius, sink) * 0.5).max(0.08);
        self.stamp_bowl(
            at,
            fwd,
            right,
            tire.width * 0.5,
            half_l,
            sink,
            TIRE_BERM,
            RHO_BERM,
            true,
        );
    }

    fn mean_column(&self, at: Vec2, fwd: Vec2, right: Vec2, half_w: f32, half_l: f32) -> (f32, f32) {
        let mut thickness = 0.0f32;
        let mut mass_per_area = 0.0f32;
        let mut n = 0.0f32;
        let lat_n = 3;
        let lon_n = 3;
        for j in 0..lat_n {
            for i in 0..lon_n {
                let u = (i as f32 + 0.5) / lon_n as f32 - 0.5;
                let v = (j as f32 + 0.5) / lat_n as f32 - 0.5;
                let p = at + fwd * (half_l * 2.0 * u) + right * (half_w * 2.0 * v);
                let d = self.depth_nearest(p.x, p.y);
                let rho = self.density_nearest(p.x, p.y);
                thickness += d;
                mass_per_area += d * rho;
                n += 1.0;
            }
        }
        if n < 1.0 || thickness <= 1.0e-5 {
            return (self.depth_nearest(at.x, at.y), RHO_FRESH);
        }
        (thickness / n, mass_per_area / thickness)
    }

    fn stamp_bowl(
        &mut self,
        at: Vec2,
        fwd: Vec2,
        right: Vec2,
        half_w: f32,
        half_l: f32,
        sink: f32,
        berm: BermShape,
        rho_deposit: f32,
        lock_packed: bool,
    ) {
        let reach_lat = half_w + berm.peak + berm.sigma_out * 2.6 + self.cell;
        let reach_along = half_l + berm.along * 2.6 + self.cell;
        let half = self.size * 0.5;
        let i0 = (((at.x - reach_lat) + half) / self.cell).floor() as isize;
        let i1 = (((at.x + reach_lat) + half) / self.cell).floor() as isize;
        let j0 = (((at.y - reach_along) + half) / self.cell).floor() as isize;
        let j1 = (((at.y + reach_along) + half) / self.cell).floor() as isize;
        let area = self.area();
        let mut cuts: Vec<(usize, f32, f32)> = Vec::new();
        // The floor stays packed. Berm snow lands outside this set.
        let mut blocked: Vec<usize> = Vec::new();
        let mut plowed = 0.0f32;
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let p = self.cell_xz(idx);
                let rel = p - at;
                let along = rel.dot(fwd);
                let lat = rel.dot(right);
                let bite = bowl_bite(lat, along, half_w, half_l);
                if bite <= 0.02 {
                    continue;
                }
                let depth = self.depth[idx];
                if depth <= 1.0e-5 {
                    continue;
                }
                // One pass packs the floor. A later stamp on that floor
                // rides it instead of shaving the column down to the soil.
                let packed_frac = ((self.density[idx] - RHO_FRESH) / (RHO_PACKED - RHO_FRESH))
                    .clamp(0.0, 1.0);
                if lock_packed && packed_frac > 0.55 {
                    continue;
                }
                let strain = ((sink * bite) / depth).clamp(0.0, 0.88);
                if strain < 0.04 {
                    continue;
                }
                // The leading edge throws most of the sunk column. What stays
                // is compacted into the floor, and anything past packed density
                // is thrown as well.
                let mass = depth * self.density[idx] * area;
                let h_stay = depth * (1.0 - strain);
                let sunk = mass * strain;
                let mut thrown = sunk * 0.70;
                let mut kept = mass - thrown;
                let cap = RHO_PACKED * h_stay.max(1.0e-5) * area;
                if kept > cap {
                    thrown += kept - cap;
                    kept = cap;
                }
                plowed += thrown;
                cuts.push((idx, h_stay, kept));
                if bite > 0.45 {
                    blocked.push(idx);
                }
            }
        }
        for (idx, h_stay, m_stay) in cuts {
            self.write_column(idx, h_stay, m_stay);
        }
        if plowed <= 1.0e-8 {
            return;
        }
        let mut weights: Vec<(usize, f32)> = Vec::new();
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let p = self.cell_xz(idx);
                let rel = p - at;
                let along = rel.dot(fwd);
                let lat = rel.dot(right);
                if blocked.contains(&idx) {
                    continue;
                }
                let outward = lat.abs() - half_w;
                let w = berm_weight(outward, berm) * along_weight(along, berm.along);
                if w > 1.0e-4 {
                    weights.push((idx, w));
                }
            }
        }
        self.deposit(&weights, plowed, rho_deposit);
    }

    fn write_column(&mut self, idx: usize, depth: f32, mass: f32) {
        let area = self.area();
        if depth <= 1.0e-6 || mass <= 1.0e-8 {
            self.depth[idx] = 0.0;
            self.density[idx] = RHO_FRESH;
            return;
        }
        self.depth[idx] = depth;
        self.density[idx] = mass / (depth * area);
    }

    fn deposit(&mut self, weights: &[(usize, f32)], mass: f32, density: f32) {
        if mass <= 0.0 {
            return;
        }
        let items: Vec<(usize, f32)> = weights
            .iter()
            .copied()
            .filter(|(_, w)| *w > 0.0)
            .collect();
        if items.is_empty() {
            self.held += mass;
            return;
        }
        let sum: f32 = items.iter().map(|(_, w)| *w).sum();
        if sum <= 1.0e-8 {
            self.held += mass;
            return;
        }
        let n = items.len();
        let mut placed = 0.0f32;
        for k in 0..n {
            let (idx, w) = items[k];
            let give = if k + 1 == n {
                mass - placed
            } else {
                let g = mass * (w / sum);
                placed += g;
                g
            };
            if give > 0.0 {
                self.add_mass(idx, give, density);
            } else if give < 0.0 {
                // Rounding ran ahead. The ledger keeps the extra in `held`
                // as a negative correction so the sum still closes.
                self.held += give;
            }
        }
    }

    /// A curved crest snaps from cell to cell, so the berm is a row of teeth.
    /// Snow moves along the circle, from a pile into the next gap, and does
    /// not cross into the packed floor or the straight track behind the arc.
    fn smooth_turn_berms(&mut self, center: Vec2, _radii: [f32; 4], sweep: f32) {
        let area = self.area() as f64;
        let step = self.cell;
        // One cell only reaches the next pile. The gap sits about two cells
        // along the circle, so the stencil has to step past the tooth.
        let hops = [1.15_f32, 2.05, 2.9];
        for _ in 0..8 {
            let depth = self.depth.clone();
            let density = self.density.clone();
            let mut delta = vec![0.0f64; depth.len()];
            for idx in 0..depth.len() {
                let h = depth[idx];
                if h < self.blanket + 0.02 {
                    continue;
                }
                let p = self.cell_xz(idx);
                let rel = p - center;
                let radius = rel.length();
                if radius < 0.8 {
                    continue;
                }
                let theta = rel.y.atan2(rel.x);
                if theta < 0.02 || theta > sweep - 0.02 {
                    continue;
                }
                let tangent = Vec2::new(-rel.y, rel.x) / radius;
                let mut outflow = 0.0f64;
                let mut sends: Vec<(usize, f64)> = Vec::new();
                for sign in [-1.0_f32, 1.0] {
                    for hop in hops {
                        let q = p + tangent * (step * hop * sign);
                        let Some(jdx) = self.idx_at(q.x, q.y) else {
                            continue;
                        };
                        if jdx == idx {
                            continue;
                        }
                        let hj = depth[jdx];
                        // The packed floor stays a rut. Only the shoulder and the
                        // gaps in the crest take snow.
                        if hj < 0.22 {
                            continue;
                        }
                        let dh = h - hj;
                        if dh < 0.018 {
                            continue;
                        }
                        let move_h = (dh * 0.34).min(0.09);
                        let mass = move_h as f64 * density[idx] as f64 * area / hops.len() as f64;
                        if mass <= 0.0 {
                            continue;
                        }
                        outflow += mass;
                        sends.push((jdx, mass));
                    }
                }
                if sends.is_empty() {
                    continue;
                }
                let have = depth[idx] as f64 * density[idx] as f64 * area;
                let cap = have * 0.45;
                let scale = if outflow > cap { cap / outflow } else { 1.0 };
                for (jdx, mass) in sends {
                    let m = mass * scale;
                    delta[idx] -= m;
                    delta[jdx] += m;
                }
            }
            for idx in 0..delta.len() {
                let m = delta[idx];
                if m > 1.0e-8 {
                    self.add_mass(idx, m as f32, RHO_BERM);
                } else if m < -1.0e-8 {
                    self.remove_mass(idx, (-m) as f32);
                }
            }
        }
        // A one-cell crest is a pyramid on the grid. Spread it radially so the
        // shoulder is a few cells wide and the mesh can round it.
        for _ in 0..5 {
            let depth = self.depth.clone();
            let density = self.density.clone();
            let mut delta = vec![0.0f64; depth.len()];
            for idx in 0..depth.len() {
                let h = depth[idx];
                if h < self.blanket + 0.03 {
                    continue;
                }
                let p = self.cell_xz(idx);
                let rel = p - center;
                let radius = rel.length();
                if radius < 0.8 {
                    continue;
                }
                let theta = rel.y.atan2(rel.x);
                if theta < 0.02 || theta > sweep - 0.02 {
                    continue;
                }
                let radial = rel / radius;
                let mut outflow = 0.0f64;
                let mut sends: Vec<(usize, f64)> = Vec::new();
                for sign in [-1.0_f32, 1.0] {
                    let q = p + radial * (step * sign);
                    let Some(jdx) = self.idx_at(q.x, q.y) else {
                        continue;
                    };
                    if jdx == idx {
                        continue;
                    }
                    let hj = depth[jdx];
                    if hj < 0.20 {
                        continue;
                    }
                    let dh = h - hj;
                    if dh < 0.012 {
                        continue;
                    }
                    let move_h = (dh * 0.38).min(0.06);
                    let mass = move_h as f64 * density[idx] as f64 * area;
                    outflow += mass;
                    sends.push((jdx, mass));
                }
                if sends.is_empty() {
                    continue;
                }
                let have = h as f64 * density[idx] as f64 * area;
                let cap = have * 0.35;
                let scale = if outflow > cap { cap / outflow } else { 1.0 };
                for (jdx, mass) in sends {
                    let m = mass * scale;
                    delta[idx] -= m;
                    delta[jdx] += m;
                }
            }
            for idx in 0..delta.len() {
                let m = delta[idx];
                if m > 1.0e-8 {
                    self.add_mass(idx, m as f32, RHO_BERM);
                } else if m < -1.0e-8 {
                    self.remove_mass(idx, (-m) as f32);
                }
            }
        }
        // Diffusion along the circle evens the height, but the crest still
        // hops from cell to cell. Lay that same snow on the median circle.
        self.round_turn_ridges(center, sweep);
    }

    /// The plow is a circle. On the grid its crest is a staircase. Replace the
    /// arc with the radial profile the circle actually has, and keep the mass.
    fn round_turn_ridges(&mut self, center: Vec2, sweep: f32) {
        let r_lo = 2.15_f32;
        let r_hi = 5.40_f32;
        let dr = self.cell * 0.5;
        let n_r = ((r_hi - r_lo) / dr).round().max(1.0) as usize + 1;
        let th0 = 0.40_f32;
        let th1 = (sweep - 0.28).max(th0 + 0.2);
        let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); n_r];
        for idx in 0..self.depth.len() {
            let rel = self.cell_xz(idx) - center;
            let radius = rel.length();
            if !(r_lo..=r_hi).contains(&radius) {
                continue;
            }
            let theta = rel.y.atan2(rel.x);
            if theta < th0 || theta > th1 {
                continue;
            }
            let bin = ((radius - r_lo) / dr).round() as isize;
            if bin < 0 || bin >= n_r as isize {
                continue;
            }
            buckets[bin as usize].push(self.depth[idx]);
        }
        let blanket = self.blanket;
        let mut profile = vec![blanket; n_r];
        for (bin, samples) in buckets.iter_mut().enumerate() {
            if samples.len() < 6 {
                continue;
            }
            samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            profile[bin] = samples[samples.len() / 2];
        }
        for bin in 1..n_r - 1 {
            if buckets[bin].len() < 6 {
                profile[bin] = 0.5 * (profile[bin - 1] + profile[bin + 1]);
            }
        }
        // A flat ring of cells still reads as a curb. Dome each shoulder so the
        // crest is one soft ridge and the lip eases down toward the rut.
        let mut rounded = profile.clone();
        let mut i = 0usize;
        while i < n_r {
            if profile[i] < blanket + 0.04 {
                i += 1;
                continue;
            }
            let start = i;
            while i < n_r && profile[i] >= blanket + 0.04 {
                i += 1;
            }
            let end = i;
            if end - start < 2 {
                continue;
            }
            let n = (end - start) as f32;
            let mean = profile[start..end].iter().sum::<f32>() / n;
            let excess = mean - blanket;
            if excess < 0.025 {
                continue;
            }
            let mid = 0.5 * (start + end - 1) as f32;
            let half = (0.5 * (end - start) as f32).max(0.5);
            let mut weights = vec![0.0f32; end - start];
            let mut wsum = 0.0f32;
            for k in start..end {
                let t = (k as f32 - mid) / half;
                let lobe = 0.5 * (1.0 + (t * std::f32::consts::PI).cos());
                let w = 0.35 + 0.65 * lobe.clamp(0.0, 1.0);
                weights[k - start] = w;
                wsum += w;
            }
            let amp = excess * n / wsum.max(1.0e-4);
            for k in start..end {
                rounded[k] = blanket + amp * weights[k - start];
            }
        }

        let area = self.area();
        let mut touched: Vec<(usize, f32, f32)> = Vec::new();
        let mut old_mass = 0.0f64;
        let edge = 0.22_f32;
        let th_in = 0.08_f32;
        let th_out = sweep - 0.06;
        for idx in 0..self.depth.len() {
            let rel = self.cell_xz(idx) - center;
            let radius = rel.length();
            if !(r_lo..=r_hi).contains(&radius) {
                continue;
            }
            let theta = rel.y.atan2(rel.x);
            if theta < th_in || theta > th_out {
                continue;
            }
            let mut fade = 1.0f32;
            if theta < th_in + edge {
                fade = (theta - th_in) / edge;
            } else if theta > th_out - edge {
                fade = (th_out - theta) / edge;
            }
            fade = fade.clamp(0.0, 1.0);
            fade = fade * fade * (3.0 - 2.0 * fade);
            if fade < 0.002 {
                continue;
            }
            let bin = ((radius - r_lo) / dr).round() as isize;
            let bin = bin.clamp(0, n_r as isize - 1) as usize;
            // Nearest bin. A blend into the rut bin next door was flattening the lip.
            let tgt_h = rounded[bin];
            let old_h = self.depth[idx];
            let old_rho = self.density[idx].max(1.0);
            let old_m = old_h * old_rho;
            let tgt_rho = if tgt_h < blanket - 0.08 {
                RHO_PACKED
            } else if tgt_h > blanket + 0.015 {
                RHO_BERM
            } else {
                RHO_FRESH
            };
            let h = old_h + (tgt_h - old_h) * fade;
            let tgt_m = tgt_h * tgt_rho;
            let m = old_m + (tgt_m - old_m) * fade;
            let rho = if h > 1.0e-5 { m / h } else { RHO_FRESH };
            old_mass += old_m as f64 * area as f64;
            touched.push((idx, h, rho));
        }
        if touched.is_empty() {
            return;
        }
        let mut new_mass = 0.0f64;
        for &(_, h, rho) in &touched {
            new_mass += h as f64 * rho.max(1.0) as f64 * area as f64;
        }
        let scale = if new_mass > 1.0e-6 {
            (old_mass / new_mass) as f32
        } else {
            1.0
        };
        // Keep the ridge height. Density carries the mass difference, so a
        // crest is not shrunk to pay for the circle.
        let mut placed = 0.0f64;
        let n = touched.len();
        for (k, (idx, h, rho)) in touched.iter().copied().enumerate() {
            let mut rho = (rho.max(1.0) * scale).max(1.0);
            let h = h.max(0.0);
            if k + 1 == n && h > 1.0e-5 {
                let need = old_mass - placed;
                let want = need / (h as f64 * area as f64);
                if want > 1.0 {
                    rho = want.clamp(rho as f64 * 0.5, rho as f64 * 1.5) as f32;
                }
            }
            placed += h as f64 * rho as f64 * area as f64;
            self.depth[idx] = h;
            self.density[idx] = rho;
        }
        let residual = old_mass - placed;
        if residual.abs() > 1.0e-6 {
            self.held += residual as f32;
        }
        self.arc = Some(self.pack_arc(center, sweep, &rounded, r_lo, dr));
    }

    /// Continuous radial profile for the picture. The grid still holds the
    /// ledger; this is the same ridge sampled so the mesh can follow the circle.
    fn pack_arc(
        &self,
        center: Vec2,
        sweep: f32,
        rounded: &[f32],
        r_lo: f32,
        dr: f32,
    ) -> SnowArc {
        let r_hi = r_lo + dr * (rounded.len().saturating_sub(1) as f32);
        let mut depth = [self.blanket; ARC_BINS];
        for i in 0..ARC_BINS {
            let radius = r_lo + (r_hi - r_lo) * (i as f32) / (ARC_BINS - 1) as f32;
            let x = ((radius - r_lo) / dr).clamp(0.0, (rounded.len() - 1) as f32);
            let bin = x.round() as usize;
            let bin = bin.min(rounded.len() - 1);
            depth[i] = rounded[bin];
        }
        // Soften the lip. A one-bin cliff is a smooth circle in the shader,
        // but a wider shoulder reads as a ridge instead of a curb.
        let src = depth;
        let sigma = 1.15_f32;
        for i in 0..ARC_BINS {
            if src[i] < self.blanket - 0.22 {
                continue;
            }
            let mut acc = 0.0f32;
            let mut wsum = 0.0f32;
            for k in -2isize..=2 {
                let j = i as isize + k;
                if j < 0 || j >= ARC_BINS as isize {
                    continue;
                }
                if src[j as usize] < self.blanket - 0.22 {
                    continue;
                }
                let w = (-0.5 * (k as f32 / sigma).powi(2)).exp();
                acc += src[j as usize] * w;
                wsum += w;
            }
            if wsum > 1.0e-6 {
                depth[i] = acc / wsum;
            }
        }
        SnowArc {
            center,
            sweep,
            r_lo,
            r_hi,
            dr: (r_hi - r_lo) / (ARC_BINS - 1) as f32,
            depth,
        }
    }


    fn remove_mass(&mut self, idx: usize, mass: f32) {
        let area = self.area();
        let old = self.depth[idx] * self.density[idx] * area;
        let new_m = (old - mass).max(0.0);
        if new_m <= 1.0e-8 {
            self.held += old - mass;
            self.depth[idx] = 0.0;
            self.density[idx] = RHO_FRESH;
            return;
        }
        let rho = self.density[idx].max(1.0);
        self.depth[idx] = new_m / (rho * area);
        if (old - mass) < 0.0 {
            self.held += old - mass;
        }
    }

    fn add_mass(&mut self, idx: usize, mass: f32, density: f32) {
        if mass <= 0.0 {
            return;
        }
        let area = self.area();
        let density = density.max(1.0);
        let old = self.depth[idx] * self.density[idx] * area;
        let add_h = mass / (density * area);
        let new_h = self.depth[idx] + add_h;
        let new_m = old + mass;
        self.depth[idx] = new_h;
        self.density[idx] = new_m / (new_h * area);
    }
}

/// Bekker pressure under a plate that has sunk `pen` metres.
fn snow_pressure(pen: f32, density: f32) -> f32 {
    if pen <= 0.0 {
        return 0.0;
    }
    let stiff = (density / RHO_FRESH).clamp(1.0, 8.0).powf(1.7);
    SNOW_K * stiff * pen.powf(SNOW_N)
}

#[derive(Clone, Copy)]
struct UnderPart {
    x0: f32,
    x1: f32,
    z0: f32,
    z1: f32,
    clearance: f32,
}

/// Belly pan across the gap between the inner sidewalls, axle tubes a little
/// narrower and lower, transfer case the low keel on the centreline.
fn under_parts() -> [UnderPart; 4] {
    [
        UnderPart {
            x0: -0.54,
            x1: 0.54,
            z0: -1.05,
            z1: 0.95,
            clearance: BELLY_CLEARANCE_M,
        },
        UnderPart {
            x0: -0.48,
            x1: 0.48,
            z0: 0.98,
            z1: 1.22,
            clearance: AXLE_CLEARANCE_M,
        },
        UnderPart {
            x0: -0.48,
            x1: 0.48,
            z0: -1.34,
            z1: -1.06,
            clearance: AXLE_CLEARANCE_M,
        },
        UnderPart {
            x0: -0.20,
            x1: 0.20,
            z0: -0.06,
            z1: 0.44,
            clearance: TCASE_CLEARANCE_M,
        },
    ]
}

/// Fresh-snow sinkage under a uniform pressure. Density stiffens the modulus.
pub fn snow_sinkage(pressure: f32, density: f32, depth: f32) -> f32 {
    if depth <= 1.0e-5 || pressure <= 0.0 {
        return 0.0;
    }
    let stiff = (density / RHO_FRESH).clamp(1.0, 8.0).powf(1.7);
    let k = SNOW_K * stiff;
    let z = (pressure / k).powf(1.0 / SNOW_N);
    z.min(depth)
}

pub fn niva_wheels() -> [Vec2; 4] {
    let hp = hardpoints();
    [
        Vec2::new(hp[0].x, hp[0].z),
        Vec2::new(hp[1].x, hp[1].z),
        Vec2::new(hp[2].x, hp[2].z),
        Vec2::new(hp[3].x, hp[3].z),
    ]
}

/// Yaw whose flat forward matches `forward`, using the chassis quaternion.
pub fn yaw_for_forward(forward: Vec2) -> f32 {
    let fwd = forward.normalize_or_zero();
    let yaw = fwd.x.atan2(fwd.y);
    let mut probe = Vehicle::new();
    probe.pitch = 0.0;
    probe.roll = 0.0;
    probe.yaw = yaw;
    let got = probe.forward();
    let err = Vec2::new(got.x, got.z) - fwd;
    if err.length() < 1.0e-3 {
        return yaw;
    }
    let yaw = (-fwd.x).atan2(fwd.y);
    probe.yaw = yaw;
    let got = probe.forward();
    let err = Vec2::new(got.x, got.z) - fwd;
    if err.length() < 1.0e-3 {
        return yaw;
    }
    fwd.x.atan2(fwd.y)
}

/// Straight ruts, then a left-hand arc, then a line of boot prints beside the ruts.
pub fn niva_snow_scene() -> SnowScene {
    let mut snow = SnowField::blanket(BLANKET_M);
    let wheels = niva_wheels();
    let tire = Tire::niva();
    let straight_x = 0.15;
    let straight_from = Vec2::new(straight_x, -6.4);
    let straight_len = 6.2;
    let straight = snow.drive_straight(straight_from, Vec2::new(0.0, 1.0), straight_len, &wheels, tire);
    let turn = snow.drive_turn(straight.origin, straight.forward, 3.55, 1.7, &wheels, tire);
    // The crest of a curve snaps between grid cells. Spread each arc slice
    // back onto a smooth shoulder so the ridge follows the circle.
    snow.smooth_turn_berms(turn.center, turn.radii, 1.7);
    // The smoother rewrites the arc, so the pan has to plane the circles
    // after it. Berms above the blanket are left on the circle.
    snow.scrape_along_turn(straight.origin, straight.forward, 3.55, 1.7, &wheels);
    snow.refit_arc_shelf();
    let foot_origin = Vec2::new(straight_x + 2.15, -5.6);
    let (foot_at, foot_fresh) = snow.walk(foot_origin, Vec2::new(0.0, 1.0), 8, 0.58);
    snow.deposit_plow(turn.end, straight_x);

    let mut packed = SnowField::blanket(BLANKET_M);
    packed.pack_disc(Vec2::ZERO, 1.4, RHO_PACKED);
    let foot_packed = packed.stamp_foot(Vec2::ZERO, Vec2::new(0.0, 1.0), FOOT_LENGTH, FOOT_WIDTH, FOOT_LOAD_N);

    let straight_z = straight_from.y + straight_len * 0.48;
    let (rut_drop, berm_rise, berm_round, rut_rho, berm_rho) =
        cross_section(&snow, straight_x, straight_z);
    let (shelf_m, shelf_flat_m, shelf_rho) = shelf_between(&snow, straight_x, straight_z);
    let bow_rise_m = bow_rise(&snow, turn.end);

    let mut troughs = [0.0f32; 4];
    for i in 0..4 {
        let (r, _) = snow.trough_radius(turn.center, turn.mid_dir[i], turn.radii[i]);
        troughs[i] = r;
    }

    let mass0 = snow.initial_mass;
    let mass1 = snow.ledger_kg();
    let geom0 = {
        let cells = (snow.res * snow.res) as f64;
        snow.blanket as f64 * snow.area() as f64 * cells
    };
    let report = SnowReport {
        blanket_m: snow.blanket,
        rho_fresh: RHO_FRESH,
        rho_packed: RHO_PACKED,
        rho_berm: RHO_BERM,
        mass0_kg: mass0,
        mass1_kg: mass1,
        held_kg: snow.held as f64,
        geom0_m3: geom0,
        geom1_m3: snow.geometric_m3(),
        solid0_m3: mass0 / RHO_ICE as f64,
        solid1_m3: mass1 / RHO_ICE as f64,
        rut_drop_m: rut_drop,
        berm_rise_m: berm_rise,
        berm_round: berm_round,
        rut_rho,
        berm_rho,
        turn_radii_m: turn.radii,
        turn_trough_m: troughs,
        foot_fresh_m: foot_fresh,
        foot_packed_m: foot_packed,
        belly_drag_kn: snow.belly_drag_kn(),
        belly_unload_kn: snow.belly_unload_kn(),
        shelf_m,
        shelf_flat_m,
        shelf_rho,
        bow_rise_m,
    };
    SnowScene {
        snow,
        pose: turn.end,
        yaw: yaw_for_forward(turn.end.forward),
        straight_x,
        straight_z,
        turn_center: turn.center,
        turn_radii: turn.radii,
        turn_dir: turn.mid_dir,
        foot_at,
        report,
    }
}

fn shelf_between(snow: &SnowField, track_x: f32, z: f32) -> (f32, f32, f32) {
    let samples = [
        snow.depth_nearest(track_x, z),
        snow.depth_nearest(track_x - 0.30, z),
        snow.depth_nearest(track_x + 0.30, z),
    ];
    let mean = (samples[0] + samples[1] + samples[2]) / 3.0;
    let mut hi = samples[0];
    let mut lo = samples[0];
    for s in samples {
        hi = hi.max(s);
        lo = lo.min(s);
    }
    (mean, hi - lo, snow.density_nearest(track_x, z))
}

fn bow_rise(snow: &SnowField, pose: DrivePose) -> f32 {
    let fwd = pose.forward.normalize_or_zero();
    let right = Vec2::new(fwd.y, -fwd.x);
    let mut best = 0.0f32;
    let mut along = 1.15_f32;
    while along <= 4.0 {
        let mut lat = -1.30_f32;
        while lat <= 1.30 {
            let p = pose.origin + fwd * along + right * lat;
            best = best.max(snow.depth_nearest(p.x, p.y));
            lat += snow.cell * 0.5;
        }
        along += snow.cell * 0.5;
    }
    (best - snow.blanket_m()).max(0.0)
}

fn cross_section(snow: &SnowField, track_x: f32, z: f32) -> (f32, f32, f32, f32, f32) {
    let wheels = niva_wheels();
    let mut drop = 0.0f32;
    let mut rise = 0.0f32;
    let mut round = 1.0f32;
    let mut rut_rho = RHO_FRESH;
    let mut berm_rho = RHO_FRESH;
    let blanket = snow.blanket_m();
    for wheel in wheels {
        let x = track_x + wheel.x;
        let (floor, rho) = deepest_near(snow, x, z, 0.22);
        let this_drop = blanket - floor;
        if this_drop > drop {
            drop = this_drop;
            rut_rho = rho;
        }
        let sign = if wheel.x >= 0.0 { 1.0 } else { -1.0 };
        let inner = x + sign * (WHEEL_WIDTH * 0.5);
        let outer = x + sign * (WHEEL_WIDTH * 0.5 + 1.15);
        if let Some(shoulder) = shoulder(snow, z, inner, outer, sign) {
            if shoulder.rise > rise {
                rise = shoulder.rise;
                berm_rho = shoulder.rho;
                round = round.min(shoulder.round);
            }
        }
    }
    (drop, rise, round, rut_rho, berm_rho)
}

struct Shoulder {
    rise: f32,
    round: f32,
    rho: f32,
}

fn deepest_near(snow: &SnowField, x: f32, z: f32, reach: f32) -> (f32, f32) {
    let mut best = f32::MAX;
    let mut rho = RHO_FRESH;
    let mut s = x - reach;
    while s <= x + reach {
        let d = snow.depth_nearest(s, z);
        if d < best {
            best = d;
            rho = snow.density_nearest(s, z);
        }
        s += snow.cell * 0.45;
    }
    if best == f32::MAX {
        (snow.blanket_m(), RHO_FRESH)
    } else {
        (best, rho)
    }
}

fn shoulder(snow: &SnowField, z: f32, inner: f32, outer: f32, sign: f32) -> Option<Shoulder> {
    let blanket = snow.blanket_m();
    let mut x = inner.min(outer);
    let end = inner.max(outer);
    let mut peak_x = x;
    let mut peak = f32::MIN;
    while x <= end {
        let d = snow.depth_nearest(x, z);
        if d > peak {
            peak = d;
            peak_x = x;
        }
        x += snow.cell * 0.5;
    }
    if peak <= blanket {
        return None;
    }
    let step = snow.cell;
    let inward = snow.depth_nearest(peak_x - sign * step, z);
    let outward = snow.depth_nearest(peak_x + sign * step, z);
    let further = snow.depth_nearest(peak_x + sign * step * 2.0, z);
    let excess = (peak - blanket).max(1.0e-4);
    // The lip drops into the rut. The outer face falls over more than one
    // cell, so the crest is a shoulder and not a one-cell needle.
    let round = if outward < peak - 0.008 && inward < peak - 0.008 && further <= outward + 0.004
    {
        ((outward - blanket).max(0.0) / excess).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some(Shoulder {
        rise: peak - blanket,
        round,
        rho: snow.density_nearest(peak_x, z),
    })
}

fn bowl_bite(lat: f32, along: f32, half_w: f32, half_l: f32) -> f32 {
    let reach_w = (half_w + half_w * 0.55).max(0.04);
    let reach_l = (half_l + 0.05).max(0.04);
    let u = lat.abs() / reach_w;
    let v = along.abs() / reach_l;
    if u >= 1.0 || v >= 1.0 {
        return 0.0;
    }
    let a = 1.0 - u;
    let b = 1.0 - v;
    let across = a * a * (3.0 - 2.0 * a);
    let fore = b * b * (3.0 - 2.0 * b);
    across * fore
}

fn berm_weight(outward: f32, shape: BermShape) -> f32 {
    // Inside the rubber the floor stays put. The shoulder starts at the sidewall.
    if outward < 0.02 {
        return 0.0;
    }
    let d = outward - shape.peak;
    let sigma = if d < 0.0 { shape.sigma_in } else { shape.sigma_out };
    let x = d / sigma.max(1.0e-4);
    (-0.5 * x * x).exp()
}

fn along_weight(along: f32, sigma: f32) -> f32 {
    let x = along / sigma.max(1.0e-4);
    (-0.5 * x * x).exp()
}

fn rear_axle_z(wheels: &[Vec2]) -> f32 {
    let mut acc = 0.0f32;
    let mut n = 0.0f32;
    for w in wheels {
        if w.y < 0.0 {
            acc += w.y;
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc / n
    } else {
        0.0
    }
}

fn rot(v: Vec2, theta: f32) -> Vec2 {
    let (s, c) = theta.sin_cos();
    Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> SnowScene {
        niva_snow_scene()
    }

    #[test]
    fn heading_matches_niva_yaw() {
        let mut v = Vehicle::new();
        v.pitch = 0.0;
        v.roll = 0.0;
        for yaw in [-1.2_f32, -0.4, 0.0, 0.7, 2.1] {
            v.yaw = yaw;
            let f = v.forward();
            let expect = Vec2::new(yaw.sin(), yaw.cos());
            let got = Vec2::new(f.x, f.z);
            assert!(
                (got - expect).length() < 1.0e-4,
                "yaw {yaw} forward {got:?} expected {expect:?}"
            );
        }
    }

    #[test]
    fn volume_ledger_closes() {
        let scene = scene();
        let r = scene.report;
        let err = (r.mass1_kg - r.mass0_kg).abs();
        assert!(
            err / r.mass0_kg < 2.0e-5,
            "mass {} -> {} held {} err {}",
            r.mass0_kg,
            r.mass1_kg,
            r.held_kg,
            err
        );
        assert!(r.held_kg.abs() < 0.2, "held {}", r.held_kg);
        let shrunk = r.geom0_m3 - r.geom1_m3;
        assert!(
            shrunk > 0.35,
            "compaction should shrink geometric volume {} -> {} ({shrunk})",
            r.geom0_m3,
            r.geom1_m3
        );
        let solid_err = (r.solid1_m3 - r.solid0_m3).abs();
        assert!(solid_err < 1.0e-3, "solid volume {solid_err}");
    }

    #[test]
    fn pass_leaves_a_rut_with_rounded_berms() {
        let scene = scene();
        let r = scene.report;
        assert!(r.rut_drop_m > 0.12, "rut drop {}", r.rut_drop_m);
        assert!(r.berm_rise_m > 0.045, "berm rise {}", r.berm_rise_m);
        assert!(
            r.berm_round > 0.15,
            "berm round {} rise {}",
            r.berm_round, r.berm_rise_m
        );
        assert!(r.rut_rho > RHO_FRESH + 40.0, "rut density {}", r.rut_rho);
        assert!(
            r.berm_rho + 30.0 < r.rut_rho,
            "berm density {} should stay looser than the floor {}",
            r.berm_rho,
            r.rut_rho
        );
        let snow = &scene.snow;
        let blanket = snow.blanket_m();
        let x = scene.straight_x;
        let z = scene.straight_z;
        let right = x + 0.74;
        assert!(
            snow.depth_nearest(right, z) < blanket - 0.10,
            "right rut {}",
            snow.depth_nearest(right, z)
        );
        let mut outer = 0.0f32;
        let mut s = right + 0.2;
        while s < right + 1.2 {
            outer = outer.max(snow.depth_nearest(s, z));
            s += snow.cell * 0.5;
        }
        assert!(outer > blanket + 0.03, "outer berm {outer} blanket {blanket}");
    }

    #[test]
    fn turn_leaves_concentric_tracks() {
        let scene = scene();
        let snow = &scene.snow;
        let blanket = snow.blanket_m();
        let mut radii = scene.turn_radii;
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            radii[3] - radii[0] > 1.3,
            "radii spread {:?}",
            scene.turn_radii
        );
        for i in 0..4 {
            let expect = scene.turn_radii[i];
            let (trough, depth) = snow.trough_radius(scene.turn_center, scene.turn_dir[i], expect);
            assert!(
                (trough - expect).abs() < 0.30,
                "wheel {i} trough {trough} expected {expect}"
            );
            assert!(
                depth < blanket - 0.08,
                "wheel {i} floor {depth} blanket {blanket}"
            );
            let dir = scene.turn_dir[i];
            // The berm is the crest beside this circle, not the next tire.
            let mut shoulder = 0.0f32;
            for delta in [-0.34_f32, -0.24, 0.24, 0.36] {
                let p = scene.turn_center + dir * (expect + delta);
                shoulder = shoulder.max(snow.depth_nearest(p.x, p.y));
            }
            assert!(
                shoulder > blanket + 0.03,
                "wheel {i} berm {shoulder} blanket {blanket}"
            );
            assert!(
                shoulder > depth + 0.04,
                "wheel {i} shoulder {shoulder} floor {depth}"
            );
        }
        // Distinct circles, not one groove sampled four times.
        let mut distinct = 0;
        for i in 1..4 {
            if radii[i] - radii[i - 1] > 0.35 {
                distinct += 1;
            }
        }
        assert!(distinct >= 3, "radii {radii:?}");
    }

    #[test]
    fn footprints_sink_and_packed_snow_resists() {
        let scene = scene();
        let r = scene.report;
        assert!(r.foot_fresh_m > 0.12, "fresh sink {}", r.foot_fresh_m);
        assert!(
            r.foot_packed_m + 0.05 < r.foot_fresh_m,
            "packed {} fresh {}",
            r.foot_packed_m,
            r.foot_fresh_m
        );
        assert!(
            r.foot_packed_m < r.foot_fresh_m * 0.72,
            "packed {} fresh {}",
            r.foot_packed_m,
            r.foot_fresh_m
        );
        let snow = &scene.snow;
        let center = snow.depth_nearest(scene.foot_at.x, scene.foot_at.y);
        let blanket = snow.blanket_m();
        assert!(center < blanket - 0.10, "print {center} blanket {blanket}");
        let mut rim = 0.0f32;
        for (dx, dz) in [(0.22, 0.0), (-0.22, 0.0), (0.0, 0.28), (0.0, -0.28)] {
            rim = rim.max(snow.depth_nearest(scene.foot_at.x + dx, scene.foot_at.y + dz));
        }
        assert!(rim > center + 0.04, "rim {rim} print {center}");
    }

    #[test]
    fn turn_berm_crest_follows_the_circle() {
        let scene = scene();
        let r = scene.report;
        assert!(
            (r.rut_drop_m - 0.4376).abs() < 0.001,
            "rut drop {}",
            r.rut_drop_m
        );
        assert!(
            (r.berm_rise_m - 0.1863).abs() < 0.001,
            "berm rise {}",
            r.berm_rise_m
        );
        assert!(
            (r.mass1_kg - r.mass0_kg).abs() < 0.05,
            "mass {} -> {}",
            r.mass0_kg,
            r.mass1_kg
        );
        let snow = &scene.snow;
        let center = scene.turn_center;
        let blanket = snow.blanket_m();
        // The crest used to hop between grid cells, so one radius was a pile
        // and the next angle was a hole. On the circle it stays a ridge.
        let crest_r = 4.55_f32;
        let mut prev: Option<f32> = None;
        let mut theta = 0.45_f32;
        while theta < 1.40 {
            let dir = Vec2::new(theta.cos(), theta.sin());
            let mut crest = 0.0f32;
            let mut s = crest_r - 0.12;
            while s <= crest_r + 0.12 {
                let p = center + dir * s;
                crest = crest.max(snow.depth_nearest(p.x, p.y));
                s += 0.04;
            }
            let rut_p = center + dir * 4.30;
            let rut = snow.depth_nearest(rut_p.x, rut_p.y);
            assert!(
                crest > blanket + 0.04,
                "theta {theta} crest {crest} blanket {blanket}"
            );
            assert!(
                crest > rut + 0.20,
                "theta {theta} crest {crest} rut {rut}"
            );
            if let Some(ph) = prev {
                assert!(
                    (crest - ph).abs() < 0.08,
                    "tooth {ph} -> {crest} at theta {theta}"
                );
            }
            prev = Some(crest);
            theta += 0.06;
        }
    }

    #[test]
    fn belly_planes_between_the_ruts() {
        let scene = scene();
        let r = scene.report;
        assert!(
            (r.mass1_kg - r.mass0_kg).abs() < 0.05,
            "mass {} -> {} held {}",
            r.mass0_kg,
            r.mass1_kg,
            r.held_kg
        );
        assert!(
            (r.rut_drop_m - 0.4376).abs() < 0.001,
            "rut drop {}",
            r.rut_drop_m
        );
        assert!(
            (r.berm_rise_m - 0.1863).abs() < 0.001,
            "berm rise {}",
            r.berm_rise_m
        );
        let snow = &scene.snow;
        let blanket = snow.blanket_m();
        let x = scene.straight_x;
        let z = scene.straight_z;
        let samples = [
            snow.depth_nearest(x, z),
            snow.depth_nearest(x - 0.30, z),
            snow.depth_nearest(x + 0.30, z),
        ];
        let mut hi = samples[0];
        let mut lo = samples[0];
        for d in samples {
            hi = hi.max(d);
            lo = lo.min(d);
            assert!(
                d < blanket - 0.18,
                "shelf {d} is still a crest, blanket {blanket}"
            );
            let floor = blanket - r.rut_drop_m;
            assert!(
                d > floor + 0.10,
                "shelf {d} fell into the rut floor {floor}"
            );
        }
        assert!(hi - lo < 0.05, "shelf not flat {lo}..{hi}");
        assert!(
            r.shelf_rho > RHO_FRESH + 15.0,
            "shelf density {}",
            r.shelf_rho
        );
        assert!(
            r.bow_rise_m > 0.08,
            "bow rise {}",
            r.bow_rise_m
        );
        assert!(
            r.bow_rise_m < 0.75,
            "bow is a spire {}",
            r.bow_rise_m
        );
        assert!(
            r.belly_drag_kn > 1.0,
            "drag {:.3} kN",
            r.belly_drag_kn
        );
        assert!(
            r.belly_unload_kn > 1.0,
            "unload {:.3} kN",
            r.belly_unload_kn
        );
    }
}
