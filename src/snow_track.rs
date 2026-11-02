//! Fine snow track. A 2.5 cm column field on world-anchored tiles.
//!
//! The coarse snow column in `snow.rs` is 20 cm per cell, so a Niva tire is
//! one cell wide and a rut cannot have a wall, a floor, or a tread print.
//! This field is the same physics at the size of a lug:
//!
//! * The tire is a cylinder with a rounded shoulder and a lug band. Every
//!   column whose top stands above that surface is pressed down to it.
//! * Pressing keeps mass. The column gets denser, up to the density the
//!   contact pressure can sinter (`density_under`). What will not fit is
//!   bulldozed: most of it goes out past the sidewall into the berm, the rest
//!   is shoved ahead of the contact, where the tire meets it again.
//! * Bulldozed snow breaks. It lands as loose snow: part as a ridge along the
//!   sidewall, part as discrete clumps whose size follows their mass. Loose
//!   snow slides until it stands at its angle of repose. Undisturbed snow
//!   and the pressed floor are cohesive and keep their walls.
//! * A spinning tire shears the floor. The lugs dig out snow and throw it
//!   backwards as clods; each clod flies, lands, and is laid back as a clump.
//!
//! Tiles are allocated the first time anything touches them and never
//! dropped, so a track stays where it was cut for the whole drive. Untouched
//! ground is the virgin blanket, computed, not stored.
//!
//! The closed ledger is mass: snow in allocated tiles plus clods in the air
//! equals what those tiles held when they were allocated.

use std::collections::HashMap;

use glam::{Vec2, Vec3};

use crate::soil::Terramech;

/// Fine cell, metres. One Niva lug is about two cells.
pub const TRACK_CELL_M: f32 = 0.025;
/// Cells on a tile edge. 64 × 2.5 cm = 1.6 m.
pub const TILE_N: usize = 64;
/// Published window, cells per side. 640 × 2.5 cm = 16 m around the Niva.
pub const TRACK_RES: usize = 640;

/// Fresh settled snow, kg/m³.
pub const RHO_FRESH: f32 = 130.0;
/// A blanket that has settled a few days, kg/m³.
pub const RHO_SETTLED: f32 = 190.0;
/// Snow that has been broken and thrown, kg/m³. Denser than fresh: the
/// bonds are gone and the grains pack.
pub const RHO_LOOSE: f32 = 230.0;
/// Pressure sintering ceiling for a tire, kg/m³. Short of firn and ice.
pub const RHO_MAX: f32 = 560.0;
/// Contact pressure that closes 63 % of the gap to `RHO_MAX`, pascals.
pub const P_COMPACT: f32 = 110_000.0;
/// Angle of repose of loose snow. tan(48°).
pub const REPOSE_TAN: f32 = 1.11;
/// Bekker modulus of fresh snow, Pa/m^n. 2.85 kN on a Niva tire sinks about 15 cm.
pub const SNOW_K0: f32 = 600_000.0;
pub const SNOW_N: f32 = 1.6;
/// Stiffening with density: k = k0 · (ρ/ρ_fresh)^this.
pub const SNOW_K_EXP: f32 = 2.5;

/// Lug height on the Niva tire, metres.
pub const LUG_DEPTH: f32 = 0.011;
/// Circumferential lug pitch, metres.
pub const LUG_PITCH: f32 = 0.055;
/// Shoulder radius of the carcass, metres.
const SHOULDER_R: f32 = 0.045;

/// One flat underbody part in body space: half width, rear and front edge,
/// bottom height below `pos`, and a front skid that ramps up.
#[derive(Clone, Copy, Debug)]
pub struct UnderPlate {
    pub half_w: f32,
    pub rear_z: f32,
    pub front_z: f32,
    pub y: f32,
    pub skid_len: f32,
    pub skid_rise: f32,
}

/// The Niva underside that meets deep snow. Loaded, the wheel centre sits
/// about 0.23 m below `pos`. The real car has 0.22 m clearance under the axle
/// housings, so they hang 0.16 m below the hubs on the 0.38 m model tire.
/// The floor pan and sills are 0.32 m up, between the axles.
pub const NIVA_UNDERSIDE: [UnderPlate; 3] = [
    // Rear axle housing, rigid, nearly track wide, diff in the middle.
    UnderPlate { half_w: 0.60, rear_z: -1.26, front_z: -1.10, y: -0.39, skid_len: 0.08, skid_rise: 0.04 },
    // Front diff and crossmember.
    UnderPlate { half_w: 0.24, rear_z: 0.98, front_z: 1.18, y: -0.38, skid_len: 0.10, skid_rise: 0.05 },
    // Floor pan, transfer case and sills between the axles.
    UnderPlate { half_w: 0.55, rear_z: -1.00, front_z: 0.95, y: -0.29, skid_len: 0.25, skid_rise: 0.07 },
];
/// Bekker plate constants of fresh snow for a wide plate (Wong, Harrison
/// snow): k = kc/b + kφ with b the plate width, n as for the tire.
const BELLY_KC: f32 = 4_370.0;
const BELLY_KPHI: f32 = 196_700.0;
/// Above this the snow fails in shear ahead of the plate instead of packing
/// under it: the belly ploughs, Pa.
const BELLY_P_FAIL: f32 = 25_000.0;
/// Sliding friction of snow on the underbody.
const BELLY_MU: f32 = 0.18;

/// Pushed snow ends this much denser than its source, capped at `RHO_LOOSE`.
/// Disturbed fresh snow settles 15–30 % on breaking (grain rearrangement).
const PUSHED_DENSIFY: f32 = 1.2;
/// Share of bulldozed mass that goes out past the sidewall. The rest is
/// pushed ahead of the contact.
const SIDE_SHARE: f32 = 0.72;
/// Share of the sideways mass that breaks into clumps. The rest is a ridge.
const CLUMP_SHARE: f32 = 0.45;
/// Share of the packing the free edge of the tread still gets. The rest of
/// that snow shears out past the sidewall instead of densifying.
const EDGE_PACK: f32 = 0.25;
/// Outer share of the half tread that is not confined, 0..1.
const CONFINE_BAND: f32 = 0.45;
/// Slip above which the lugs shear the floor.
const DIG_SLIP: f32 = 0.18;

#[derive(Clone, Copy, Debug)]
pub struct TrackCell {
    pub bed: f32,
    pub depth: f32,
    pub rho: f32,
    /// Metres of the column that are loose snow. Only that part slides.
    pub loose: f32,
    /// How far the lug band pushed this floor below the carcass, metres.
    pub lug: f32,
    /// Density when the current pass began. One pass packs once.
    pub rho_pass: f32,
    /// Step on which a tire last pressed this column.
    pub pressed_step: u32,
}

impl TrackCell {
    pub fn surface(&self) -> f32 {
        self.bed + self.depth
    }
    pub fn mass_per_m2(&self) -> f32 {
        self.depth * self.rho
    }
    pub fn compaction(&self) -> f32 {
        if self.depth < 1.0e-4 {
            return 0.0;
        }
        ((self.rho - RHO_FRESH) / (RHO_MAX - RHO_FRESH)).clamp(0.0, 1.0)
    }
}

struct Tile {
    cells: Vec<TrackCell>,
    mass0: f64,
}

/// A clod in flight. Lands as a clump.
#[derive(Clone, Copy, Debug)]
pub struct Clod {
    pub pos: Vec3,
    pub vel: Vec3,
    pub mass: f32,
}

/// What the belly did to the snow and what the snow does back.
#[derive(Clone, Copy, Debug, Default)]
pub struct BellyReport {
    pub cells: u32,
    /// Force on the car, world frame, newtons.
    pub force: Vec3,
    /// Torque about the car's `pos`, world frame, N·m.
    pub torque: Vec3,
    /// Upward support, newtons.
    pub lift_n: f32,
    pub ploughed_kg: f32,
    pub packed_kg: f32,
}

/// What one tire did to the snow on this step.
#[derive(Clone, Copy, Debug, Default)]
pub struct PressReport {
    pub cells: u32,
    pub pressure_pa: f32,
    pub compacted_kg: f32,
    pub bulldozed_kg: f32,
    pub dug_kg: f32,
}

/// Inputs for one tire on one step. All world space, metres, newtons.
#[derive(Clone, Copy, Debug)]
pub struct TirePress {
    pub center: Vec3,
    /// Flat forward of the tire, unit.
    pub forward: Vec2,
    pub radius: f32,
    pub width: f32,
    pub normal_n: f32,
    pub slip: f32,
    pub omega: f32,
    /// Wheel spin angle, radians. Places the lugs.
    pub spin: f32,
    /// Ground speed of the hub along `forward`, m/s.
    pub v_long: f32,
    /// Ground speed of the contact across `forward`, toward `right`, m/s.
    /// A tire crabbing through a turn sweeps snow out on the side it moves to.
    pub v_lat: f32,
    pub dt: f32,
    /// Which tire. Keeps each tire's clump accumulator apart.
    pub index: usize,
}

/// The ground a tire meets, for the vehicle's suspension and traction.
#[derive(Clone, Copy, Debug)]
pub struct SnowGround {
    /// Top of the snow the tire is rolling into, metres.
    pub surface: f32,
    pub density: f32,
    pub mech: Terramech,
}

#[derive(Clone, Copy, Debug, Default)]
struct Bucket {
    mass: f32,
    /// Mass-weighted mean point where the mass was lifted.
    at: Vec2,
    out: Vec2,
    edge: f32,
}

pub struct SnowTrack {
    pub blanket: f32,
    /// Density of the undisturbed blanket, kg/m³. Settled powder by default.
    pub rho_blanket: f32,
    pub bed_y: f32,
    /// Amplitude of the wind-laid relief on the blanket, metres.
    pub drift: f32,
    tiles: HashMap<(i32, i32), Tile>,
    pub clods: Vec<Clod>,
    buckets: [[Bucket; 2]; 8],
    rng: u64,
    /// Mass that could not be placed (off any tile). Stays in the ledger.
    pub held_kg: f64,
    /// Step counter. A column pressed on consecutive steps is one pass.
    step_no: u32,
}

/// Global cell index of a world coordinate.
#[inline]
pub fn cell_of(x: f32) -> i32 {
    (x / TRACK_CELL_M).floor() as i32
}

#[inline]
fn cell_center(g: i32) -> f32 {
    (g as f32 + 0.5) * TRACK_CELL_M
}

/// Density a column of density `rho0` reaches under `pressure`, kg/m³.
/// Each load closes part of the gap to the sintering ceiling, so a second
/// pass packs the floor further, by less.
pub fn density_under(pressure: f32, rho0: f32) -> f32 {
    let p = pressure.max(0.0);
    let r0 = rho0.clamp(RHO_FRESH, RHO_MAX);
    r0 + (RHO_MAX - r0) * (1.0 - (-p / P_COMPACT).exp())
}

/// Bekker modulus for snow of this density, Pa/m^n.
pub fn snow_modulus(rho: f32) -> f32 {
    SNOW_K0 * (rho / RHO_FRESH).clamp(1.0, 6.0).powf(SNOW_K_EXP)
}

/// Grip and sinkage of snow for the vehicle's traction model.
pub fn snow_mech(rho: f32) -> Terramech {
    let c = ((rho - RHO_FRESH) / (RHO_MAX - RHO_FRESH)).clamp(0.0, 1.0);
    Terramech {
        k_eq: snow_modulus(rho),
        n: SNOW_N,
        // Fresh snow is weak and frictional. Packed snow sinters and holds.
        cohesion: 900.0 + 5_500.0 * c,
        phi: 0.33 + 0.12 * c,
        shear_k: 0.030 - 0.012 * c,
        yield_stress: 0.0,
        viscosity: 0.0,
    }
}

/// Lug band at a point of the tire surface. 1 on a lug, 0 in a void.
/// `s` is the circumferential coordinate in metres, `lat` the offset from
/// the tread centre. Directional chevrons, a centre groove, staggered shoulders.
pub fn lug_mask(s: f32, lat: f32, half_width: f32) -> f32 {
    let a = lat.abs();
    if a < 0.010 || a > half_width - 0.006 {
        return 0.0;
    }
    let shoulder = a > half_width * 0.55;
    let phase = if shoulder { 0.5 } else { 0.0 };
    let p = (s + a * 0.55) / LUG_PITCH + phase;
    let f = p - p.floor();
    if f < 0.48 {
        1.0
    } else {
        0.0
    }
}

impl SnowTrack {
    pub fn new(blanket: f32, bed_y: f32) -> Self {
        Self {
            blanket,
            rho_blanket: RHO_SETTLED,
            bed_y,
            drift: 0.012,
            tiles: HashMap::new(),
            clods: Vec::new(),
            buckets: [[Bucket::default(); 2]; 8],
            rng: 0x9e37_79b9_7f4a_7c15,
            held_kg: 0.0,
            step_no: 1,
        }
    }

    pub fn cell_area(&self) -> f32 {
        TRACK_CELL_M * TRACK_CELL_M
    }

    /// Virgin blanket at a global cell. Wind lays a long low swell on it.
    pub fn virgin(&self, gi: i32, gj: i32) -> TrackCell {
        let x = cell_center(gi);
        let z = cell_center(gj);
        let swell = value_noise(x * 0.35, z * 0.35) * 0.65 + value_noise(x * 1.3, z * 1.1) * 0.35;
        TrackCell {
            bed: self.bed_y,
            depth: (self.blanket + (swell - 0.5) * 2.0 * self.drift).max(0.0),
            rho: self.rho_blanket,
            loose: 0.0,
            lug: 0.0,
            rho_pass: self.rho_blanket,
            pressed_step: 0,
        }
    }

    fn tile_key(gi: i32, gj: i32) -> ((i32, i32), usize) {
        let n = TILE_N as i32;
        let ti = gi.div_euclid(n);
        let tj = gj.div_euclid(n);
        let li = gi.rem_euclid(n) as usize;
        let lj = gj.rem_euclid(n) as usize;
        ((ti, tj), lj * TILE_N + li)
    }

    pub fn get(&self, gi: i32, gj: i32) -> TrackCell {
        let (key, idx) = Self::tile_key(gi, gj);
        match self.tiles.get(&key) {
            Some(t) => t.cells[idx],
            None => self.virgin(gi, gj),
        }
    }

    fn ensure_tile(&mut self, key: (i32, i32)) {
        if self.tiles.contains_key(&key) {
            return;
        }
        let n = TILE_N as i32;
        let mut cells = Vec::with_capacity(TILE_N * TILE_N);
        let mut mass = 0.0f64;
        let area = self.cell_area() as f64;
        for lj in 0..n {
            for li in 0..n {
                let c = self.virgin(key.0 * n + li, key.1 * n + lj);
                mass += c.mass_per_m2() as f64 * area;
                cells.push(c);
            }
        }
        self.tiles.insert(key, Tile { cells, mass0: mass });
    }

    pub fn cell_mut(&mut self, gi: i32, gj: i32) -> &mut TrackCell {
        let (key, idx) = Self::tile_key(gi, gj);
        self.ensure_tile(key);
        &mut self.tiles.get_mut(&key).unwrap().cells[idx]
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    pub fn surface_at(&self, x: f32, z: f32) -> f32 {
        self.get(cell_of(x), cell_of(z)).surface()
    }

    /// Bilinear surface height, metres.
    pub fn surface_bilinear(&self, x: f32, z: f32) -> f32 {
        let u = x / TRACK_CELL_M - 0.5;
        let v = z / TRACK_CELL_M - 0.5;
        let i = u.floor() as i32;
        let j = v.floor() as i32;
        let tx = u - i as f32;
        let ty = v - j as f32;
        let a = self.get(i, j).surface();
        let b = self.get(i + 1, j).surface();
        let c = self.get(i, j + 1).surface();
        let d = self.get(i + 1, j + 1).surface();
        let ab = a + (b - a) * tx;
        let cd = c + (d - c) * tx;
        ab + (cd - ab) * ty
    }

    /// Snow mass in allocated tiles plus clods in flight plus held, kg.
    pub fn ledger_kg(&self) -> (f64, f64) {
        let area = self.cell_area() as f64;
        let mut now = 0.0f64;
        let mut start = 0.0f64;
        for t in self.tiles.values() {
            start += t.mass0;
            for c in &t.cells {
                now += c.mass_per_m2() as f64 * area;
            }
        }
        for c in &self.clods {
            now += c.mass as f64;
        }
        for side in &self.buckets {
            for b in side {
                now += b.mass as f64;
            }
        }
        (start, now + self.held_kg)
    }

    fn rand(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        ((x.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / (1u64 << 24) as f32
    }

    /// The snow a tire is rolling into: the band just ahead of the axle,
    /// across the tread. The vehicle's Bekker solve sinks from that top.
    pub fn ground_for(
        &self,
        center: Vec3,
        forward: Vec2,
        radius: f32,
        width: f32,
        v_long: f32,
    ) -> SnowGround {
        let fwd = forward.normalize_or_zero();
        let right = Vec2::new(fwd.y, -fwd.x);
        let dir = if v_long < -0.05 { -1.0 } else { 1.0 };
        // Each strip across the tread meets the highest snow on the leading
        // half of the chord. In virgin snow that is the blanket; in an old
        // rut it is the packed floor.
        let mut top = 0.0f32;
        let mut rho = 0.0f32;
        let mut n = 0.0f32;
        let lat_n = 7;
        let long_n = 10;
        for b in 0..lat_n {
            let lat = ((b as f32 + 0.5) / lat_n as f32 - 0.5) * (width * 0.85);
            let mut best = f32::MIN;
            let mut best_rho = RHO_FRESH;
            for a in 0..long_n {
                let along = dir * radius * 0.9 * (a as f32 / (long_n - 1) as f32);
                let p = Vec2::new(center.x, center.z) + fwd * along + right * lat;
                let c = self.get(cell_of(p.x), cell_of(p.y));
                if c.surface() > best {
                    best = c.surface();
                    best_rho = if c.depth > 1.0e-4 { c.rho } else { RHO_MAX };
                }
            }
            top += best;
            rho += best_rho;
            n += 1.0;
        }
        let rho = (rho / n).clamp(RHO_FRESH, RHO_MAX);
        SnowGround {
            surface: top / n,
            density: rho,
            mech: snow_mech(rho),
        }
    }

    /// Start a solver step. Call once per step before pressing the tires.
    pub fn begin_step(&mut self) {
        self.step_no = self.step_no.wrapping_add(1).max(2);
    }

    /// Press one tire into the snow. Returns what it did.
    pub fn press(&mut self, t: &TirePress) -> PressReport {
        let mut rep = PressReport::default();
        let fwd = t.forward.normalize_or_zero();
        if fwd.length_squared() < 0.5 {
            return rep;
        }
        let right = Vec2::new(fwd.y, -fwd.x);
        let c2 = Vec2::new(t.center.x, t.center.z);
        let half_w = t.width * 0.5;
        let r = t.radius;
        let reach_lat = half_w + 0.03;
        let reach = Vec2::new(r.abs() + reach_lat, r.abs() + reach_lat);
        let i0 = cell_of(c2.x - reach.x);
        let i1 = cell_of(c2.x + reach.x);
        let j0 = cell_of(c2.y - reach.y);
        let j1 = cell_of(c2.y + reach.y);
        let area = self.cell_area();
        let dir = if t.v_long < -0.05 { -1.0 } else { 1.0 };

        // Tire surface under a ground point. None off the cylinder.
        let surface_y = |p: Vec2| -> Option<(f32, f32, f32)> {
            let d = p - c2;
            let long = d.dot(fwd);
            let lat = d.dot(right);
            if lat.abs() > half_w || long.abs() >= r * 0.995 {
                return None;
            }
            let mut y = t.center.y - (r * r - long * long).sqrt();
            // Rounded shoulder: the carcass lifts toward the sidewall.
            let inner = half_w - SHOULDER_R;
            if lat.abs() > inner {
                let e = (lat.abs() - inner).min(SHOULDER_R);
                y += SHOULDER_R - (SHOULDER_R * SHOULDER_R - e * e).max(0.0).sqrt();
            }
            let s = t.spin * r + long;
            let lug = lug_mask(s, lat, half_w);
            Some((y - lug * LUG_DEPTH, long, lat))
        };

        // Pass 1: contact area at the current state.
        let mut contact = 0u32;
        for gj in j0..=j1 {
            for gi in i0..=i1 {
                let p = Vec2::new(cell_center(gi), cell_center(gj));
                if let Some((y, _, _)) = surface_y(p) {
                    if self.get(gi, gj).surface() > y + 1.0e-4 {
                        contact += 1;
                    }
                }
            }
        }
        if contact == 0 || t.normal_n <= 1.0 {
            return rep;
        }
        let pressure = (t.normal_n / (contact as f32 * area).max(0.012)).min(350_000.0);
        rep.pressure_pa = pressure;
        // Sideways sweep as a share: 0 rolling straight, ±1 sliding across.
        let crab = (t.v_lat / (t.v_long.abs() + 0.4) * 2.5).clamp(-1.0, 1.0);

        // Pass 2: press every column above the tire down onto it.
        let mut side_mass = [0.0f32; 2];
        let mut side_at = [Vec2::ZERO; 2];
        let mut ahead_mass = [0.0f32; 2];
        let mut contact_cells: Vec<(i32, i32)> = Vec::with_capacity(contact as usize);
        let mut front_long = 0.0f32;
        // Bulldozed mass and the volume it had where it came from.
        let (mut moved_kg, mut moved_m3) = (0.0f32, 0.0f32);
        for gj in j0..=j1 {
            for gi in i0..=i1 {
                let p = Vec2::new(cell_center(gi), cell_center(gj));
                let Some((y, long, lat)) = surface_y(p) else {
                    continue;
                };
                let cell = *self.cell_mut(gi, gj);
                if cell.surface() <= y + 1.0e-4 {
                    continue;
                }
                rep.cells += 1;
                contact_cells.push((gi, gj));
                if long * dir > front_long {
                    front_long = long * dir;
                }
                let mass = cell.mass_per_m2();
                let d_new = (y - cell.bed).max(0.0);
                // Confinement. Under the middle of the tread the snow has
                // nowhere to go and densifies. Near the sidewall it has a free
                // face beside it and flows out instead.
                let kappa = ((half_w - lat.abs()) / (CONFINE_BAND * half_w)).clamp(0.0, 1.0);
                let same_pass = cell.pressed_step + 2 >= self.step_no && cell.pressed_step != 0;
                let rho0 = if same_pass { cell.rho_pass } else { cell.rho };
                let rho_cap = density_under(pressure, rho0);
                let reach_rho = rho0 + (rho_cap - rho0).max(0.0) * (EDGE_PACK + (1.0 - EDGE_PACK) * kappa);
                let target = reach_rho.max(cell.rho);
                let mut excess = 0.0;
                let (depth, rho) = if d_new <= 1.0e-5 {
                    excess = mass;
                    (0.0, cell.rho)
                } else if mass <= d_new * target {
                    (d_new, (mass / d_new).max(cell.rho))
                } else {
                    excess = mass - d_new * target;
                    (d_new, target)
                };
                let step_no = self.step_no;
                let c = self.cell_mut(gi, gj);
                c.depth = depth;
                c.rho = rho;
                c.rho_pass = rho0;
                c.pressed_step = step_no;
                c.loose = 0.0;
                // Lug print: how far below the carcass the band pressed.
                let carcass = y + lug_mask(t.spin * r + long, lat, half_w) * LUG_DEPTH;
                c.lug = (carcass - y).max(0.0);
                rep.compacted_kg += (mass - excess) * area;
                if excess > 0.0 {
                    let m = excess * area;
                    rep.bulldozed_kg += m;
                    moved_kg += m;
                    moved_m3 += m / rho0.max(1.0);
                    let s = if lat > 0.0 { 1 } else { 0 };
                    let side = m * SIDE_SHARE;
                    // Crabbing: the sidewall that leads sideways is a blade.
                    let to_right =
                        (0.5 + 0.5 * (crab * lat.signum()).clamp(-1.0, 1.0)).clamp(0.0, 1.0);
                    let keep = if lat > 0.0 { to_right } else { 1.0 - to_right };
                    let w_own = keep.max(0.0) * 2.0_f32.min(1.0 / keep.max(1e-3)).min(1.0);
                    let own = side * w_own;
                    let other = side - own;
                    side_mass[s] += own;
                    side_at[s] += p * own;
                    side_mass[1 - s] += other;
                    side_at[1 - s] += (p - right * (2.0 * lat)) * other;
                    ahead_mass[s] += m - side;
                }
            }
        }

        // A spinning tire digs. The lugs shear a layer off the floor and throw it.
        let slip = t.slip.abs();
        if slip > DIG_SLIP && !contact_cells.is_empty() {
            let v_surf = (t.omega * r).abs();
            let rate = (slip - DIG_SLIP) * v_surf * t.width * LUG_DEPTH * 1.6; // m³/s
            let per_cell_m = rate * t.dt / (contact_cells.len() as f32 * area);
            let mut dug = 0.0f32;
            for &(gi, gj) in &contact_cells {
                let c = self.cell_mut(gi, gj);
                let take = per_cell_m.min(c.depth);
                dug += take * c.rho * area;
                c.depth -= take;
            }
            rep.dug_kg = dug;
            self.throw_clods(t, dug, fwd, right);
        }

        // Pushed snow is broken but not pressed: its grains rearrange and it
        // ends a little denser than where it came from, never denser than
        // thrown snow.
        let rho_src = if moved_m3 > 0.0 { moved_kg / moved_m3 } else { RHO_LOOSE };
        let rho_dep = (rho_src * PUSHED_DENSIFY).clamp(rho_src.min(RHO_LOOSE), RHO_LOOSE);

        // Ahead of the contact: the bow the tire meets next step.
        for s in 0..2 {
            if ahead_mass[s] <= 0.0 {
                continue;
            }
            let lat = if s == 1 { half_w * 0.5 } else { -half_w * 0.5 };
            let at = c2 + fwd * (dir * (front_long + 0.04)) + right * lat;
            self.lay_band(at, fwd, right, half_w * 0.9, 0.05, ahead_mass[s], rho_dep);
        }

        // Out past the sidewall: a ridge plus clumps.
        for s in 0..2 {
            if side_mass[s] <= 0.0 {
                continue;
            }
            let at = side_at[s] / side_mass[s];
            let out = if s == 1 { right } else { -right };
            let ridge = side_mass[s] * (1.0 - CLUMP_SHARE);
            let edge = c2 + out * half_w;
            // Project the lift point onto the sidewall line.
            let along = (at - c2).dot(fwd);
            let base = edge + fwd * along;
            self.lay_ridge(base, out, fwd, ridge, rho_dep);
            let b = &mut self.buckets[t.index.min(3) * 2 + s][0];
            let m = side_mass[s] - ridge;
            let tot = b.mass + m;
            if tot > 0.0 {
                b.at = (b.at * b.mass + base * m) / tot;
            }
            b.mass = tot;
            b.out = out;
            b.edge = 0.0;
            self.drop_clumps(t.index.min(3) * 2 + s, fwd);
        }

        // Loose snow slides to its angle of repose around this tire.
        let pad = 10;
        self.relax(i0 - pad, i1 + pad, j0 - pad, j1 + pad, 2);
        rep
    }

    /// Press the belly plate of a car at `pos` with orientation `ori` and
    /// velocity `vel` into the snow. Snow above the plate is packed under it
    /// as a Bekker plate; what would need more than `BELLY_P_FAIL` fails in
    /// shear and is ploughed ahead. Returns the support, the friction and the
    /// plough drag on the car. One step of `dt` seconds.
    pub fn press_belly(&mut self, pos: Vec3, ori: glam::Quat, vel: Vec3, mass_car: f32, dt: f32) -> BellyReport {
        let mut rep = BellyReport::default();
        let mut force = Vec3::ZERO;
        let mut torque = Vec3::ZERO;
        for plate in NIVA_UNDERSIDE {
            let r = self.press_plate(&plate, pos, ori, vel, dt);
            rep.cells += r.cells;
            rep.lift_n += r.lift_n;
            rep.ploughed_kg += r.ploughed_kg;
            rep.packed_kg += r.packed_kg;
            force += r.force;
            torque += r.torque;
        }
        // Never more horizontal drag than stops the car this step.
        let hv = Vec3::new(vel.x, 0.0, vel.z);
        let hf = Vec3::new(force.x, 0.0, force.z);
        let cap = mass_car * hv.length() / dt.max(1.0e-4);
        if hf.length() > cap && hf.length() > 0.0 {
            let k = cap / hf.length();
            force.x *= k;
            force.z *= k;
        }
        rep.force = force;
        rep.torque = torque;
        rep
    }

    fn press_plate(&mut self, plate: &UnderPlate, pos: Vec3, ori: glam::Quat, vel: Vec3, dt: f32) -> BellyReport {
        let mut rep = BellyReport::default();
        let inv = ori.inverse();
        let corners = [
            Vec3::new(-plate.half_w, plate.y, plate.rear_z),
            Vec3::new(plate.half_w, plate.y, plate.rear_z),
            Vec3::new(-plate.half_w, plate.y, plate.front_z),
            Vec3::new(plate.half_w, plate.y, plate.front_z),
        ]
        .map(|c| pos + ori * c);
        let lowest = corners.iter().fold(f32::MAX, |a, c| a.min(c.y));
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for c in corners {
            lo = lo.min(Vec2::new(c.x, c.z));
            hi = hi.max(Vec2::new(c.x, c.z));
        }
        // Quick out: nothing in the box reaches the lowest corner.
        let (i0, i1, j0, j1) = (cell_of(lo.x), cell_of(hi.x), cell_of(lo.y), cell_of(hi.y));
        let up = ori * Vec3::Y;
        if up.y < 0.3 {
            return rep;
        }
        let area = self.cell_area();
        let k = BELLY_KC / (2.0 * plate.half_w).min(plate.front_z - plate.rear_z) + BELLY_KPHI;
        let rho_ref_blanket = self.rho_blanket;
        let hv = Vec2::new(vel.x, vel.z);
        let speed = hv.length();
        let mut ahead = 0.0f32;
        let mut ahead_kg_m3 = (0.0f32, 0.0f32);
        let mut lift_at = Vec3::ZERO;
        let mut front_at = Vec2::ZERO;
        let mut front_w = 0.0f32;
        for gj in j0..=j1 {
            for gi in i0..=i1 {
                let p2 = Vec2::new(cell_center(gi), cell_center(gj));
                let cell = self.get(gi, gj);
                if cell.surface() <= lowest - 0.10 {
                    continue;
                }
                // Plate height over this point: the plane through `pos`.
                let l = inv * (Vec3::new(p2.x, pos.y, p2.y) - pos);
                if l.x.abs() > plate.half_w || l.z < plate.rear_z || l.z > plate.front_z {
                    continue;
                }
                let skid = ((l.z - (plate.front_z - plate.skid_len)) / plate.skid_len.max(1.0e-3)).clamp(0.0, 1.0);
                let ly = plate.y + plate.skid_rise * skid * skid;
                // Plate height straight above or below this point: the plane
                // through the body-space point, along the body's up.
                let q0 = pos + ori * Vec3::new(l.x, ly, l.z);
                let yb = q0.y - (up.x * (p2.x - q0.x) + up.z * (p2.y - q0.z)) / up.y;
                if cell.surface() <= yb + 1.0e-4 {
                    continue;
                }
                rep.cells += 1;
                let m = cell.mass_per_m2();
                let d_new = (yb - cell.bed).max(0.0);
                let rho_ref = rho_ref_blanket.min(cell.rho).max(1.0);
                // Plate sinkage into this column: how far it is pressed below
                // the height its mass would stand at reference density.
                let delta_max = (BELLY_P_FAIL / k).powf(1.0 / SNOW_N);
                let keep_max = (rho_ref * (d_new + delta_max)).min(RHO_MAX * d_new);
                let kept = m.min(keep_max);
                let excess = m - kept;
                let delta = (kept / rho_ref - d_new).max(0.0);
                let pressure = (k * delta.powf(SNOW_N)).min(BELLY_P_FAIL);
                let c = self.cell_mut(gi, gj);
                c.depth = d_new;
                c.rho = if d_new > 1.0e-6 { (kept / d_new).max(c.rho.min(RHO_MAX)) } else { c.rho };
                if d_new <= 1.0e-6 {
                    c.depth = 0.0;
                }
                c.loose = 0.0;
                c.lug = 0.0;
                let kept_real = c.depth * c.rho;
                let excess = excess + (kept - kept_real).max(0.0);
                rep.packed_kg += kept_real.min(m) * area;
                if excess > 0.0 {
                    let kg = excess * area;
                    ahead += kg;
                    ahead_kg_m3.0 += kg;
                    ahead_kg_m3.1 += kg / rho_ref;
                }
                let f = pressure * area;
                rep.lift_n += f;
                lift_at += Vec3::new(p2.x, yb, p2.y) * f;
                let lead = if speed > 0.05 { p2.dot(hv / speed) } else { 0.0 };
                let w = (lead * 20.0).exp().min(1.0e30);
                front_at += p2 * w;
                front_w += w;
            }
        }
        if rep.cells == 0 {
            return rep;
        }
        // Support along the plate normal, at the pressure centroid.
        let n = up;
        let f_n = n * rep.lift_n;
        let mut force = f_n;
        let at = if rep.lift_n > 1.0e-3 { lift_at / rep.lift_n } else { pos };
        // Friction and plough drag against the horizontal motion.
        if speed > 0.02 {
            let dir = Vec3::new(hv.x / speed, 0.0, hv.y / speed);
            let plough = ahead / dt.max(1.0e-4) * speed;
            let drag = BELLY_MU * rep.lift_n + plough;
            force -= dir * drag;
        }
        rep.force = force;
        rep.torque = (at - pos).cross(force);
        // The ploughed snow piles at the leading edge, a little ahead.
        if ahead > 0.0 {
            rep.ploughed_kg = ahead;
            let rho_src = ahead_kg_m3.0 / ahead_kg_m3.1.max(1.0e-9);
            let rho_dep = (rho_src * PUSHED_DENSIFY).clamp(rho_src.min(RHO_LOOSE), RHO_LOOSE);
            let fwd3 = if speed > 0.05 {
                Vec2::new(hv.x, hv.y) / speed
            } else {
                let f = ori * Vec3::Z;
                Vec2::new(f.x, f.z).normalize_or_zero()
            };
            let right = Vec2::new(fwd3.y, -fwd3.x);
            let front = if front_w > 0.0 { front_at / front_w } else { Vec2::new(pos.x, pos.z) };
            let base = Vec2::new(pos.x, pos.z) + fwd3 * ((front - Vec2::new(pos.x, pos.z)).dot(fwd3) + 0.08);
            self.lay_band(base, fwd3, right, plate.half_w, 0.08, ahead, rho_dep);
            let pad = 8;
            self.relax(i0 - pad, i1 + pad, j0 - pad, j1 + pad, 2);
        }
        rep
    }

    /// Lay `mass` kg as loose snow over a rectangle, evenly.
    fn lay_band(
        &mut self,
        center: Vec2,
        fwd: Vec2,
        right: Vec2,
        half_lat: f32,
        half_long: f32,
        mass: f32,
        rho_dep: f32,
    ) {
        let mut cells: Vec<(i32, i32, f32)> = Vec::new();
        let reach = half_lat.max(half_long) + TRACK_CELL_M;
        let mut wsum = 0.0;
        for gj in cell_of(center.y - reach)..=cell_of(center.y + reach) {
            for gi in cell_of(center.x - reach)..=cell_of(center.x + reach) {
                let p = Vec2::new(cell_center(gi), cell_center(gj)) - center;
                let a = p.dot(right).abs() / half_lat.max(1e-3);
                let b = p.dot(fwd).abs() / half_long.max(1e-3);
                if a > 1.0 || b > 1.0 {
                    continue;
                }
                let w = (1.0 - a * a) * (1.0 - b * b);
                cells.push((gi, gj, w));
                wsum += w;
            }
        }
        self.deposit(&cells, wsum, mass, rho_dep);
    }

    /// A ridge just outside the sidewall: steep inner face, longer outer tail.
    fn lay_ridge(&mut self, base: Vec2, out: Vec2, fwd: Vec2, mass: f32, rho_dep: f32) {
        if mass <= 0.0 {
            return;
        }
        let mut cells: Vec<(i32, i32, f32)> = Vec::new();
        let width = 0.16;
        let half_long = 0.05;
        let mut wsum = 0.0;
        let far = base + out * width;
        let lo = base.min(far) - Vec2::splat(half_long + TRACK_CELL_M);
        let hi = base.max(far) + Vec2::splat(half_long + TRACK_CELL_M);
        for gj in cell_of(lo.y)..=cell_of(hi.y) {
            for gi in cell_of(lo.x)..=cell_of(hi.x) {
                let p = Vec2::new(cell_center(gi), cell_center(gj)) - base;
                let o = p.dot(out);
                let l = p.dot(fwd).abs() / half_long;
                if o < 0.0 || o > width || l > 1.0 {
                    continue;
                }
                // Crest 3.5 cm out from the sidewall.
                let x = o / width;
                let w = if x < 0.22 {
                    x / 0.22
                } else {
                    ((1.0 - x) / 0.78).powf(1.6)
                } * (1.0 - l * l);
                cells.push((gi, gj, w));
                wsum += w;
            }
        }
        self.deposit(&cells, wsum, mass, rho_dep);
    }

    fn deposit(&mut self, cells: &[(i32, i32, f32)], wsum: f32, mass: f32, rho_dep: f32) {
        if cells.is_empty() || wsum <= 1.0e-6 {
            self.held_kg += mass as f64;
            return;
        }
        let area = self.cell_area();
        for &(gi, gj, w) in cells {
            let m = mass * w / wsum / area; // kg/m²
            let c = self.cell_mut(gi, gj);
            let add = m / rho_dep;
            let total = c.depth * c.rho + m;
            c.depth += add;
            c.rho = if c.depth > 1.0e-6 {
                total / c.depth
            } else {
                rho_dep
            };
            c.loose += add;
            c.lug = 0.0;
        }
    }

    /// One clump: a broken lump of loose snow whose volume is its mass.
    /// Lumps are not round: each gets its own elongation and heading.
    fn lay_clump(&mut self, at: Vec2, mass: f32) {
        let vol = mass / RHO_LOOSE;
        let aspect = 1.0 + 0.9 * self.rand();
        let tall = 0.45 + 0.4 * self.rand();
        let ang = self.rand() * std::f32::consts::TAU;
        let (sn, cs) = ang.sin_cos();
        let ax = Vec2::new(cs, sn);
        let ay = Vec2::new(-sn, cs);
        // Paraboloid lump: V = π a b h / 2 with a = aspect·b, h = tall·b.
        let b = (2.0 * vol / (std::f32::consts::PI * aspect * tall))
            .powf(1.0 / 3.0)
            .max(TRACK_CELL_M * 0.7);
        let a = b * aspect;
        let mut cells: Vec<(i32, i32, f32)> = Vec::new();
        let mut wsum = 0.0;
        for gj in cell_of(at.y - a)..=cell_of(at.y + a) {
            for gi in cell_of(at.x - a)..=cell_of(at.x + a) {
                let p = Vec2::new(cell_center(gi), cell_center(gj)) - at;
                let u = p.dot(ax) / a;
                let v = p.dot(ay) / b;
                let q = u * u + v * v;
                if q >= 1.0 {
                    continue;
                }
                let w = 1.0 - q;
                cells.push((gi, gj, w));
                wsum += w;
            }
        }
        if cells.is_empty() {
            cells.push((cell_of(at.x), cell_of(at.y), 1.0));
            wsum = 1.0;
        }
        self.deposit(&cells, wsum, mass, RHO_LOOSE);
    }

    fn drop_clumps(&mut self, slot: usize, fwd: Vec2) {
        loop {
            // Broken snow has a heavy-tailed lump size: many crumbs, a few slabs.
            let u = self.rand().max(0.02);
            let want = (0.018 * u.powf(-0.85)).min(0.45);
            let b = self.buckets[slot][0];
            if b.mass < want {
                return;
            }
            let jitter_out = 0.03 + 0.11 * self.rand();
            let jitter_long = (self.rand() - 0.5) * 0.10;
            let at = b.at + b.out * jitter_out + fwd * jitter_long;
            self.buckets[slot][0].mass -= want;
            self.lay_clump(at, want);
        }
    }

    fn throw_clods(&mut self, t: &TirePress, mass: f32, fwd: Vec2, right: Vec2) {
        if mass <= 0.0 {
            return;
        }
        let spin_dir = t.omega.signum();
        let v_surf = (t.omega * t.radius).abs();
        // Lugs leave the floor at the trailing edge of the contact.
        let back = Vec2::new(fwd.x, fwd.y) * (-spin_dir);
        let mut left = mass;
        while left > 1.0e-5 {
            let m = (0.01 + 0.06 * self.rand()).min(left);
            left -= m;
            let lat = (self.rand() - 0.5) * t.width;
            let p2 = Vec2::new(t.center.x, t.center.z) + back * (t.radius * 0.55) + right * lat;
            let surf = self.surface_at(p2.x, p2.y);
            let speed = v_surf * (0.35 + 0.4 * self.rand());
            let up = speed * (0.45 + 0.5 * self.rand());
            let side = (self.rand() - 0.5) * speed * 0.35;
            let h = back * speed + right * side;
            self.clods.push(Clod {
                pos: Vec3::new(p2.x, surf + 0.03, p2.y),
                vel: Vec3::new(h.x, up, h.y),
                mass: m,
            });
        }
    }

    /// Fly the clods. A clod that meets the snow is laid as a clump.
    pub fn step_clods(&mut self, dt: f32) {
        let mut landed: Vec<Clod> = Vec::new();
        let mut i = 0;
        while i < self.clods.len() {
            let mut c = self.clods[i];
            c.vel.y -= 9.81 * dt;
            // Quadratic drag on a 5 cm lump of snow.
            let v = c.vel.length();
            c.vel *= 1.0 / (1.0 + 0.35 * v * dt);
            c.pos += c.vel * dt;
            let surf = self.surface_at(c.pos.x, c.pos.z);
            if c.pos.y <= surf {
                landed.push(c);
                self.clods.swap_remove(i);
                continue;
            }
            self.clods[i] = c;
            i += 1;
        }
        for c in landed {
            self.lay_clump(Vec2::new(c.pos.x, c.pos.z), c.mass);
        }
    }

    /// Loose snow slides to its angle of repose. Cohesive snow does not move.
    pub fn relax(&mut self, i0: i32, i1: i32, j0: i32, j1: i32, iters: usize) {
        let max_step = REPOSE_TAN * TRACK_CELL_M;
        for _ in 0..iters {
            let mut moved = false;
            for gj in j0..=j1 {
                for gi in i0..=i1 {
                    let c = self.get(gi, gj);
                    if c.loose <= 1.0e-5 {
                        continue;
                    }
                    let h = c.surface();
                    // Steepest neighbour.
                    let mut best = (0, 0, 0.0f32);
                    for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let n = self.get(gi + di, gj + dj);
                        let drop = h - n.surface() - max_step;
                        if drop > best.2 {
                            best = (di, dj, drop);
                        }
                    }
                    if best.2 <= 1.0e-5 {
                        continue;
                    }
                    let take = (best.2 * 0.5).min(c.loose);
                    let mass = take * c.rho;
                    {
                        let a = self.cell_mut(gi, gj);
                        a.depth -= take;
                        a.loose -= take;
                    }
                    let b = self.cell_mut(gi + best.0, gj + best.1);
                    let total = b.depth * b.rho + mass;
                    b.depth += take;
                    b.rho = if b.depth > 1.0e-6 {
                        total / b.depth
                    } else {
                        RHO_LOOSE
                    };
                    b.loose += take;
                    moved = true;
                }
            }
            if !moved {
                break;
            }
        }
    }

    /// Window of `TRACK_RES`² cells centred on `center`, snapped to the cell.
    /// Texel: R surface above the bed, G snow depth, B compaction, A lug print.
    /// Untouched cells are the virgin blanket, so every texel is valid.
    pub fn window(&self, center: Vec2) -> TrackWindow {
        let half = TRACK_RES as i32 / 2;
        let gi0 = cell_of(center.x) - half;
        let gj0 = cell_of(center.y) - half;
        let mut texels = vec![[0.0f32; 4]; TRACK_RES * TRACK_RES];
        for j in 0..TRACK_RES {
            for i in 0..TRACK_RES {
                let c = self.get(gi0 + i as i32, gj0 + j as i32);
                let sanitize = |v: f32| if v.is_finite() { v } else { 0.0 };
                texels[j * TRACK_RES + i] = [
                    sanitize(c.surface() - c.bed),
                    sanitize(c.depth),
                    sanitize(c.compaction()),
                    sanitize(c.lug),
                ];
            }
        }
        TrackWindow {
            origin: Vec2::new(gi0 as f32 * TRACK_CELL_M, gj0 as f32 * TRACK_CELL_M),
            cell: TRACK_CELL_M,
            res: TRACK_RES,
            texels,
        }
    }
}

/// One scripted control segment for a snow drive. The driver holds a
/// target speed with the throttle and brake; steering is the wheel input.
#[derive(Clone, Copy, Debug)]
pub struct DriveSegment {
    pub seconds: f32,
    /// Target ground speed, m/s. Zero parks with the brake.
    pub speed: f32,
    pub steer: f32,
}

impl DriveSegment {
    fn pedals(&self, speed: f32) -> (f32, f32) {
        if self.speed <= 0.01 {
            return (0.0, 1.0);
        }
        let err = self.speed - speed;
        if err < -0.6 {
            return (0.0, (-err * 0.3).min(0.6));
        }
        ((0.18 + err * 0.5).clamp(0.0, 0.8), 0.0)
    }
}

/// Step the real Niva on the snow track for `h` seconds: the tires read the
/// snow they roll into, the suspension and traction solve on it, then each
/// tire presses the snow it now stands in.
pub fn step_vehicle(
    track: &mut SnowTrack,
    veh: &mut crate::vehicle::Vehicle,
    soil: &crate::soil::SoilField,
    seg: &DriveSegment,
    h: f32,
) -> [PressReport; 4] {
    use crate::vehicle::{hardpoints, REST_LEN, WHEEL_RADIUS, WHEEL_WIDTH};
    let ori = veh.orientation();
    let hps = hardpoints();
    for i in 0..4 {
        let hp = hps[i];
        let wc0 = veh.pos + ori * Vec3::new(hp.x, hp.y - REST_LEN, hp.z);
        let mut f = ori * (glam::Quat::from_rotation_y(veh.wheels[i].steer) * Vec3::Z);
        f.y = 0.0;
        let f2 = Vec2::new(f.x, f.z).normalize_or_zero();
        let v_long = veh.vel.x * f2.x + veh.vel.z * f2.y;
        veh.snow_ground[i] = Some(track.ground_for(wc0, f2, WHEEL_RADIUS, WHEEL_WIDTH, v_long));
    }
    veh.update_steer(seg.steer, h);
    let (throttle, brake) = seg.pedals(veh.speed());
    let contacts = veh.compute_contacts(soil, throttle, brake, h);
    veh.integrate(&contacts, soil, h);
    let ori = veh.orientation();
    let ang = ori * Vec3::new(veh.pitch_rate, veh.yaw_rate, veh.roll_rate);
    track.begin_step();
    let mut out = [PressReport::default(); 4];
    for c in &contacts {
        if c.normal <= 1.0 {
            continue;
        }
        let w = veh.wheels[c.index];
        let f2 = Vec2::new(c.forward.x, c.forward.z).normalize_or_zero();
        let r2 = Vec2::new(c.right.x, c.right.z).normalize_or_zero();
        let v = veh.vel + ang.cross(c.contact_point - veh.pos);
        let v2 = Vec2::new(v.x, v.z);
        out[c.index] = track.press(&TirePress {
            center: c.wheel_center,
            forward: f2,
            radius: WHEEL_RADIUS,
            width: WHEEL_WIDTH,
            normal_n: c.normal,
            slip: c.slip,
            omega: w.omega,
            spin: w.spin,
            v_long: v2.dot(f2),
            v_lat: v2.dot(r2),
            dt: h,
            index: c.index,
        });
    }
    let belly = track.press_belly(veh.pos, ori, veh.vel, veh.total_mass(), h);
    veh.snow_belly_force = belly.force;
    veh.snow_belly_torque = belly.torque;
    track.step_clods(h);
    out
}

/// Totals over a drive, for the proof log and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct DriveLog {
    pub seconds: f32,
    pub path_m: f32,
    pub sink_mean_m: f32,
    pub sink_max_m: f32,
    pub pressure_mean_kpa: f32,
    pub bulldozed_kg: f32,
    pub dug_kg: f32,
    pub clods_in_air: usize,
}

/// Drive a script at 120 Hz.
pub fn drive_script(
    track: &mut SnowTrack,
    veh: &mut crate::vehicle::Vehicle,
    soil: &crate::soil::SoilField,
    script: &[DriveSegment],
) -> DriveLog {
    let h = 1.0 / 120.0;
    let mut log = DriveLog::default();
    let mut sink_n = 0.0f32;
    let mut p_n = 0.0f32;
    for seg in script {
        let n = (seg.seconds / h).round() as i32;
        for _ in 0..n {
            let before = veh.pos;
            let reps = step_vehicle(track, veh, soil, seg, h);
            log.path_m += Vec2::new(veh.pos.x - before.x, veh.pos.z - before.z).length();
            for (i, r) in reps.iter().enumerate() {
                log.bulldozed_kg += r.bulldozed_kg;
                log.dug_kg += r.dug_kg;
                if r.cells > 0 {
                    log.pressure_mean_kpa += r.pressure_pa * 1.0e-3;
                    p_n += 1.0;
                }
                let s = veh.wheels[i].sinkage;
                if s > 0.0 {
                    log.sink_mean_m += s;
                    sink_n += 1.0;
                    log.sink_max_m = log.sink_max_m.max(s);
                }
            }
            log.seconds += h;
        }
    }
    log.sink_mean_m /= sink_n.max(1.0);
    log.pressure_mean_kpa /= p_n.max(1.0);
    log.clods_in_air = track.clods.len();
    log
}

/// The reference drive: straight in, a tight left circle and a bit more,
/// then out. Parked at the end so the frame is still.
pub fn circle_script() -> Vec<DriveSegment> {
    vec![
        DriveSegment {
            seconds: 3.0,
            speed: 2.4,
            steer: 0.0,
        },
        DriveSegment {
            seconds: 13.0,
            speed: 2.2,
            steer: 1.0,
        },
        DriveSegment {
            seconds: 2.5,
            speed: 2.4,
            steer: 0.0,
        },
        DriveSegment {
            seconds: 1.5,
            speed: 0.0,
            steer: 0.0,
        },
    ]
}

/// The published fine window.
#[derive(Clone, Debug)]
pub struct TrackWindow {
    /// Minimum corner, metres, snapped to the cell.
    pub origin: Vec2,
    pub cell: f32,
    pub res: usize,
    pub texels: Vec<[f32; 4]>,
}

fn hash2(i: i32, j: i32) -> f32 {
    let mut h = (i as u32).wrapping_mul(0x8da6_b343) ^ (j as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 13;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 16;
    (h & 0x00ff_ffff) as f32 / 16_777_216.0
}

fn value_noise(x: f32, z: f32) -> f32 {
    let i = x.floor() as i32;
    let j = z.floor() as i32;
    let fx = x - i as f32;
    let fz = z - j as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let a = hash2(i, j);
    let b = hash2(i + 1, j);
    let c = hash2(i, j + 1);
    let d = hash2(i + 1, j + 1);
    let ab = a + (b - a) * sx;
    let cd = c + (d - c) * sx;
    ab + (cd - ab) * sz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tire(center: Vec3, fwd: Vec2, n: f32, slip: f32, spin: f32) -> TirePress {
        TirePress {
            center,
            forward: fwd,
            radius: 0.38,
            width: 0.28,
            normal_n: n,
            slip,
            omega: 8.0,
            spin,
            v_long: 3.0,
            v_lat: 0.0,
            dt: 1.0 / 120.0,
            index: 0,
        }
    }

    #[test]
    fn pressing_keeps_mass_and_packs_the_floor() {
        let mut s = SnowTrack::new(0.30, 0.0);
        s.drift = 0.0;
        let fwd = Vec2::new(0.0, 1.0);
        for k in 0..200 {
            let z = k as f32 * 0.01;
            let c = Vec3::new(0.0, 0.30 - 0.14 + 0.38, z);
            s.begin_step();
            s.press(&tire(c, fwd, 2850.0, 0.0, z / 0.38));
        }
        let (start, now) = s.ledger_kg();
        assert!(
            (start - now).abs() / start < 1.0e-5,
            "ledger {start} -> {now}"
        );
        let floor = s.get(cell_of(0.0), cell_of(1.0));
        assert!(floor.depth < 0.20 && floor.rho > 200.0, "floor {floor:?}");
        // Berm stands above the blanket just past the sidewall.
        let mut crest = 0.0f32;
        for k in 0..10 {
            let x = 0.15 + k as f32 * TRACK_CELL_M;
            crest = crest.max(s.surface_at(x, 1.0));
        }
        eprintln!("press: crest {crest:.4} floor {floor:?}");
        assert!(crest > 0.31, "crest {crest}");
    }

    #[test]
    fn spinning_throws_clods_that_land() {
        let mut s = SnowTrack::new(0.30, 0.0);
        let fwd = Vec2::new(1.0, 0.0);
        for k in 0..60 {
            let c = Vec3::new(0.0, 0.30 - 0.14 + 0.38, 0.0);
            s.begin_step();
            s.press(&tire(c, fwd, 2850.0, 0.9, k as f32 * 0.1));
            s.step_clods(1.0 / 120.0);
        }
        assert!(!s.clods.is_empty() || s.tile_count() > 0);
        for _ in 0..600 {
            s.step_clods(1.0 / 120.0);
        }
        assert!(s.clods.is_empty());
        let (start, now) = s.ledger_kg();
        assert!(
            (start - now).abs() / start < 1.0e-5,
            "ledger {start} -> {now}"
        );
    }

    #[test]
    fn niva_circle_leaves_a_persistent_rut_with_berms() {
        use crate::soil::SoilField;
        use crate::vehicle::Vehicle;
        let mut s = SnowTrack::new(0.25, 0.0);
        s.rho_blanket = 160.0;
        let soil = SoilField::flat(0.1);
        let mut v = Vehicle::new();
        v.pos = Vec3::new(0.0, 0.25 + 0.38 + 0.45, -6.0);
        let script = [
            DriveSegment {
                seconds: 3.0,
                speed: 2.4,
                steer: 0.0,
            },
            DriveSegment {
                seconds: 1.0,
                speed: 0.0,
                steer: 0.0,
            },
        ];
        let log = drive_script(&mut s, &mut v, &soil, &script);
        for _ in 0..360 {
            s.step_clods(1.0 / 120.0);
        }
        assert!(log.path_m > 4.0, "{log:?}");
        assert!(log.sink_mean_m > 0.04 && log.sink_mean_m < 0.16, "{log:?}");
        let (m0, m1) = s.ledger_kg();
        assert!(((m1 - m0) / m0).abs() < 1.0e-6, "ledger {m0} -> {m1}");
        assert!(s.clods.is_empty());
        // Cross-section behind the car, across the left rear track.
        let z = -4.5;
        let mut floor = f32::MAX;
        let mut crest = f32::MIN;
        let mut x = -1.4;
        while x < 1.4 {
            let h = s.surface_at(x, z);
            floor = floor.min(h);
            crest = crest.max(h);
            x += TRACK_CELL_M;
        }
        eprintln!("niva section: floor {floor:.4} crest {crest:.4} sink {:.4}", log.sink_mean_m);
        assert!(floor < 0.25 - 0.04, "rut floor {floor}");
        assert!(crest > 0.25 + 0.012, "berm crest {crest}");
        let packed = s.get(cell_of(-0.74), cell_of(z));
        assert!(packed.rho > 200.0, "floor density {}", packed.rho);
        // The cut is kept: reading the window far away and back changes nothing.
        let before = s.surface_at(-0.74, z);
        let _ = s.window(Vec2::new(200.0, 200.0));
        assert_eq!(before, s.surface_at(-0.74, z));
    }
    #[test]
    fn old_ruts_in_deep_snow_drag_the_belly_over_the_hump() {
        use crate::soil::SoilField;
        use crate::vehicle::Vehicle;
        // 75 cm of snow with two old packed ruts at the Niva's track, floor
        // at 30 cm. The tires run in the ruts; the hump between them stands
        // above the axles.
        let blanket = 0.75;
        let mut s = SnowTrack::new(blanket, 0.0);
        s.drift = 0.0;
        for gj in cell_of(-8.0)..cell_of(8.0) {
            for x in [-0.74f32, 0.74] {
                for gi in cell_of(x - 0.17)..=cell_of(x + 0.17) {
                    let c = s.cell_mut(gi, gj);
                    let m = c.depth * c.rho;
                    c.depth = 0.30;
                    c.rho = m / 0.30;
                }
            }
        }
        let (a0, b0) = s.ledger_kg();
        let shift = b0 - a0;
        let soil = SoilField::flat(0.1);
        let mut v = Vehicle::new();
        v.pos = Vec3::new(0.0, 0.30 + 0.38 + 0.45, -6.0);
        let seg = DriveSegment { seconds: 6.0, speed: 2.0, steer: 0.0 };
        let h = 1.0 / 120.0;
        let (mut lift_sum, mut drag_sum, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for k in 0..(seg.seconds / h).round() as i32 {
            step_vehicle(&mut s, &mut v, &soil, &seg, h);
            if k > 120 {
                lift_sum += v.snow_belly_force.y;
                drag_sum += Vec2::new(v.snow_belly_force.x, v.snow_belly_force.z).length();
                n += 1.0;
            }
        }
        for _ in 0..360 {
            s.step_clods(h);
        }
        let (m0, m1) = s.ledger_kg();
        let hump = s.get(cell_of(0.0), cell_of(-3.0));
        let virgin = s.get(cell_of(0.0), cell_of(7.5));
        eprintln!(
            "ruts: lift {:.0} N drag {:.0} N z {:.2} speed {:.2} hump {:.3} rho {:.0} wheels {:.3}",
            lift_sum / n,
            drag_sum / n,
            v.pos.z,
            v.speed(),
            hump.surface(),
            hump.rho,
            v.wheels[2].sinkage
        );
        assert!(((m1 - m0 - shift) / m0).abs() < 1.0e-5, "ledger {m0} -> {m1}");
        assert!(lift_sum / n > 300.0, "the belly does not bear on the hump: {}", lift_sum / n);
        assert!(drag_sum / n > 50.0, "no belly drag: {}", drag_sum / n);
        // The underside planed the hump flat and packed it.
        assert!(hump.surface() < virgin.surface() - 0.10, "hump {} vs {}", hump.surface(), virgin.surface());
        assert!(hump.rho > virgin.rho + 15.0, "hump density {}", hump.rho);
    }
}
