//! Free bodies in the world: balls, crates and rocks, and a walking person.
//!
//! These are the props of the reference: a person wading through the swash,
//! buoys riding it, a ball rolling a groove into mud, boot prints in snow.
//! Nothing here draws a wake or a print. Each body hands the sheet the
//! volume it fills (`fluid_core::WetBody`) and the drag impulse it took, and
//! presses the ground under it through [`Ground`]. The wake, the ring, the
//! print and its rim come out of the sheet step and the ground's own law.
//!
//! The vehicle keeps its own wheel model. Bodies use the same pieces it does:
//! the shared sheet, the Bekker plate law with sinkage memory, and the fine
//! snow track's compaction law.

use glam::{Quat, Vec2, Vec3};

use crate::fluid_core::{FluidCore, Grain, WetBody, WetLoad, WetShape, WATER_RHO};
use crate::imprint::{cell_center, cell_of, imprint_mat, Imprint, ImprintMat, PressOut, IMPRINT_CELL_M};
use crate::snow_track::{SnowTrack, P_COMPACT, RHO_LOOSE, RHO_MAX, TRACK_CELL_M};
use crate::soil::{SoilField, SurfaceKind};

pub const GRAVITY: f32 = 9.81;

/// Ground a body can stand on and press.
pub trait Ground {
    /// Top of the ground, world Y.
    fn surface(&self, x: f32, z: f32) -> f32;
    /// Bearing pressure if the ground at the point were pushed to `y`, Pa.
    /// Zero when `y` is above the ground there.
    fn pressure(&self, x: f32, z: f32, y: f32) -> f32;
    /// Coulomb friction against a hard body.
    fn mu(&self, x: f32, z: f32) -> f32;
    /// Conform to the underside of the shape where it is below the ground.
    fn press(&mut self, center: Vec2, shape: &WetShape) -> PressOut;
    /// Cell size the footprints are integrated on, metres.
    fn cell(&self) -> f32;
}

/// Plate-law ground over a coarse height: soil grid, the sheet's bed, or a
/// plain function. `base` returns the coarse height and the plate law.
pub struct PrintGround<'a, F: Fn(f32, f32) -> (f32, ImprintMat)> {
    pub imprint: &'a mut Imprint,
    pub base: F,
}

impl<'a, F: Fn(f32, f32) -> (f32, ImprintMat)> Ground for PrintGround<'a, F> {
    fn surface(&self, x: f32, z: f32) -> f32 {
        (self.base)(cell_center(cell_of(x)), cell_center(cell_of(z))).0 + self.imprint.offset_at(x, z)
    }

    fn pressure(&self, x: f32, z: f32, y: f32) -> f32 {
        let (gi, gj) = (cell_of(x), cell_of(z));
        let (h, mat) = (self.base)(cell_center(gi), cell_center(gj));
        let g = h + self.imprint.get(gi, gj).dz;
        self.imprint.pressure(mat, gi, gj, g, y)
    }

    fn mu(&self, x: f32, z: f32) -> f32 {
        (self.base)(x, z).1.mu
    }

    fn press(&mut self, center: Vec2, shape: &WetShape) -> PressOut {
        let base = |x: f32, z: f32| (self.base)(x, z).0;
        let mat = |x: f32, z: f32| (self.base)(x, z).1;
        self.imprint.press(center, shape, &base, &mat)
    }

    fn cell(&self) -> f32 {
        IMPRINT_CELL_M
    }
}

/// Coarse height and plate law from the soil grid.
pub fn soil_base(soil: &SoilField) -> impl Fn(f32, f32) -> (f32, ImprintMat) + '_ {
    move |x, z| {
        let s = soil.sample(x, z);
        (s.height, imprint_mat(s.kind(), s.wetness, s.water + s.slurry))
    }
}

/// Coarse height and plate law from the sheet's bed and pore water.
pub fn bed_base(fluid: &FluidCore) -> impl Fn(f32, f32) -> (f32, ImprintMat) + '_ {
    move |x, z| {
        let s = fluid.sample(x, z);
        let kind = match fluid.grain_at(x, z) {
            Grain::Sand => SurfaceKind::Sand,
            Grain::Silt => SurfaceKind::Silt,
            Grain::Clay => SurfaceKind::WetMud,
            Grain::Other => SurfaceKind::Gravel,
            Grain::Water => SurfaceKind::Sand,
        };
        (s.bed_m, imprint_mat(kind, s.moisture, s.water_m + s.slurry_m))
    }
}

/// Share of the mass a boot or a crate pushes down in snow that goes out
/// sideways into the low rim instead of packing.
const SNOW_RIM_SHARE: f32 = 0.12;

/// The fine snow track as ground. The bearing pressure is the snow track's
/// own compaction law turned around: the pressure that packs the column from
/// the blanket density to the density it would have at `y`.
impl Ground for SnowTrack {
    fn surface(&self, x: f32, z: f32) -> f32 {
        self.surface_at(x, z)
    }

    fn pressure(&self, x: f32, z: f32, y: f32) -> f32 {
        let c = self.get(crate::snow_track::cell_of(x), crate::snow_track::cell_of(z));
        if y > c.surface() + 1.0e-4 {
            return 0.0;
        }
        let d = y - c.bed;
        if d <= 2.0e-3 || c.depth <= 2.0e-3 {
            // On the bed: hard ground.
            return 2.0e7;
        }
        let rho = c.mass_per_m2() / d;
        let rho0 = self.rho_blanket.min(c.rho);
        let frac = ((rho - rho0) / (RHO_MAX - rho0)).clamp(0.0, 0.999);
        if rho >= RHO_MAX {
            return 2.0e7;
        }
        // Loose snow carries a little before it packs.
        1_500.0 - P_COMPACT * (1.0 - frac).ln()
    }

    fn mu(&self, x: f32, z: f32) -> f32 {
        let c = self.get(crate::snow_track::cell_of(x), crate::snow_track::cell_of(z));
        0.30 + 0.12 * c.compaction()
    }

    fn press(&mut self, center: Vec2, shape: &WetShape) -> PressOut {
        use crate::snow_track::{cell_of as scell, TrackCell};
        let mut out = PressOut::default();
        let cell = TRACK_CELL_M;
        let area = cell * cell;
        let reach = shape.reach() + cell;
        let (i0, i1) = (scell(center.x - reach), scell(center.x + reach));
        let (j0, j1) = (scell(center.y - reach), scell(center.y + reach));
        let mut rim_kg = 0.0f32;
        let mut covered: Vec<(i32, i32)> = Vec::new();
        let mut contact: Vec<(i32, i32)> = Vec::new();
        for gj in j0..=j1 {
            for gi in i0..=i1 {
                let x = (gi as f32 + 0.5) * cell;
                let z = (gj as f32 + 0.5) * cell;
                let Some((lo, _)) = shape.span(x - center.x, z - center.y) else {
                    continue;
                };
                covered.push((gi, gj));
                let c: TrackCell = self.get(gi, gj);
                if lo >= c.surface() || c.depth <= 1.0e-4 {
                    continue;
                }
                let d_new = (lo - c.bed).max(0.0);
                let pushed = c.depth - d_new;
                let m = c.mass_per_m2();
                // A thin share of the column above the sole shears out at the
                // sole's edge; the rest packs under it.
                let mut take = (pushed * c.rho * SNOW_RIM_SHARE).min(m);
                let keep = m - take;
                let (depth, rho) = if d_new <= 1.0e-4 {
                    take = m;
                    (0.0, c.rho)
                } else if keep <= d_new * RHO_MAX {
                    (d_new, (keep / d_new).max(c.rho))
                } else {
                    take += keep - d_new * RHO_MAX;
                    (d_new, RHO_MAX)
                };
                let cm = self.cell_mut(gi, gj);
                cm.depth = depth;
                cm.rho = rho;
                cm.loose = 0.0;
                cm.lug = 0.0;
                rim_kg += take * area;
                out.cells += 1;
                out.deepest_m = out.deepest_m.max(pushed);
                out.pressed_m3 += pushed * area;
                contact.push((gi, gj));
            }
        }
        if out.cells == 0 {
            return out;
        }
        covered.sort_unstable();
        contact.sort_unstable();
        let band = 2i32;
        let mut ring: Vec<(i32, i32, f32)> = Vec::new();
        let mut wsum = 0.0f32;
        for gj in (j0 - band)..=(j1 + band) {
            for gi in (i0 - band)..=(i1 + band) {
                if covered.binary_search(&(gi, gj)).is_ok() {
                    continue;
                }
                let mut best = i32::MAX;
                for dj in -band..=band {
                    for di in -band..=band {
                        if contact.binary_search(&(gi + di, gj + dj)).is_ok() {
                            best = best.min(di.abs().max(dj.abs()));
                        }
                    }
                }
                if best > band {
                    continue;
                }
                let w = (band + 1 - best) as f32;
                ring.push((gi, gj, w));
                wsum += w;
            }
        }
        if wsum > 0.0 && rim_kg > 0.0 {
            for &(gi, gj, w) in &ring {
                let m = rim_kg * w / wsum / area;
                let c = self.cell_mut(gi, gj);
                let add = m / RHO_LOOSE;
                let total = c.depth * c.rho + m;
                c.depth += add;
                c.rho = total / c.depth.max(1.0e-6);
                c.loose += add;
            }
            out.rim_m3 = rim_kg / RHO_LOOSE;
        } else if rim_kg > 0.0 {
            self.held_kg += rim_kg as f64;
        }
        self.relax(i0 - 4, i1 + 4, j0 - 4, j1 + 4, 2);
        out
    }

    fn cell(&self) -> f32 {
        TRACK_CELL_M
    }
}

/// The fluid around a body for one step.
#[derive(Clone, Copy, Debug)]
pub struct Water {
    pub surface: f32,
    pub flow: Vec2,
    pub rho: f32,
}

impl Water {
    pub const DRY: Water = Water {
        surface: f32::NEG_INFINITY,
        flow: Vec2::ZERO,
        rho: WATER_RHO,
    };

    fn from_load(l: &WetLoad) -> Self {
        if !l.wet {
            return Self::DRY;
        }
        Water {
            surface: l.surface_y,
            flow: Vec2::new(l.flow_u, l.flow_v),
            rho: l.rho(),
        }
    }
}

fn water_at(fluid: Option<&FluidCore>, id: u32, x: f32, z: f32) -> Water {
    let Some(f) = fluid else {
        return Water::DRY;
    };
    if let Some(l) = f.wet_loads().iter().find(|l| l.id == id) {
        return Water::from_load(l);
    }
    let (eta, bed, u, v) = f.surface_at(x, z);
    if eta - bed < 2.0e-3 {
        return Water::DRY;
    }
    Water {
        surface: eta,
        flow: Vec2::new(u, v),
        rho: WATER_RHO,
    }
}

/// Shape of a rigid body.
#[derive(Clone, Copy, Debug)]
pub enum BodyShape {
    /// A ball. `hollow` is a shell (buoy, play ball), otherwise solid.
    Ball { r: f32, hollow: bool },
    /// A box that keeps level and turns about Y only: a crate, a rock.
    Crate { half: Vec3 },
}

#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub shape: BodyShape,
    pub mass: f32,
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: Quat,
    pub omega: Vec3,
    /// Drag the water took off this body since the sheet last stepped, N·s.
    pub drag_impulse: Vec2,
    /// Submerged volume on the last substep, m³.
    pub submerged_m3: f32,
    /// Deepest the ground has been pushed under it on the last press, m.
    pub print_m: f32,
    pub grounded: bool,
}

impl Body {
    pub fn ball(pos: Vec3, r: f32, mass: f32, hollow: bool) -> Self {
        Self::new(BodyShape::Ball { r, hollow }, pos, mass)
    }

    pub fn crate_box(pos: Vec3, half: Vec3, mass: f32, yaw: f32) -> Self {
        let mut b = Self::new(BodyShape::Crate { half }, pos, mass);
        b.rot = Quat::from_rotation_y(yaw);
        b
    }

    fn new(shape: BodyShape, pos: Vec3, mass: f32) -> Self {
        Self {
            shape,
            mass,
            pos,
            vel: Vec3::ZERO,
            rot: Quat::IDENTITY,
            omega: Vec3::ZERO,
            drag_impulse: Vec2::ZERO,
            submerged_m3: 0.0,
            print_m: 0.0,
            grounded: false,
        }
    }

    pub fn volume(&self) -> f32 {
        match self.shape {
            BodyShape::Ball { r, .. } => 4.0 / 3.0 * std::f32::consts::PI * r * r * r,
            BodyShape::Crate { half } => 8.0 * half.x * half.y * half.z,
        }
    }

    pub fn yaw(&self) -> f32 {
        let f = self.rot * Vec3::Z;
        f.x.atan2(f.z)
    }

    /// Lowest point of the body, world Y.
    pub fn bottom(&self) -> f32 {
        match self.shape {
            BodyShape::Ball { r, .. } => self.pos.y - r,
            BodyShape::Crate { half } => self.pos.y - half.y,
        }
    }

    /// What the sheet and the ground see.
    pub fn wet_shape(&self) -> WetShape {
        match self.shape {
            BodyShape::Ball { r, .. } => WetShape::Sphere { cy: self.pos.y, r },
            BodyShape::Crate { half } => WetShape::Box {
                yaw: self.yaw(),
                half_l: half.z,
                half_w: half.x,
                bottom: self.pos.y - half.y,
                top: self.pos.y + half.y,
            },
        }
    }

    /// Submerged volume, horizontal frontal area and waterplane area under a
    /// free surface at `eta`.
    fn immersion(&self, eta: f32) -> (f32, f32, f32) {
        match self.shape {
            BodyShape::Ball { r, .. } => {
                let h = (eta - (self.pos.y - r)).clamp(0.0, 2.0 * r);
                let vol = std::f32::consts::PI * h * h * (3.0 * r - h) / 3.0;
                let a = r - h;
                let seg = r * r * (a / r).clamp(-1.0, 1.0).acos() - a * (2.0 * r * h - h * h).max(0.0).sqrt();
                let wp = std::f32::consts::PI * (2.0 * r * h - h * h).max(0.0);
                (vol, seg, wp)
            }
            BodyShape::Crate { half } => {
                let h = (eta - (self.pos.y - half.y)).clamp(0.0, 2.0 * half.y);
                let plan = 4.0 * half.x * half.z;
                let wp = if h > 0.0 && h < 2.0 * half.y { plan } else { 0.0 };
                (plan * h, 2.0 * half.x.max(half.z) * h, wp)
            }
        }
    }

    fn inertia(&self) -> f32 {
        match self.shape {
            BodyShape::Ball { r, hollow } => {
                if hollow {
                    2.0 / 3.0 * self.mass * r * r
                } else {
                    0.4 * self.mass * r * r
                }
            }
            BodyShape::Crate { half } => self.mass * (half.x * half.x + half.z * half.z) / 3.0,
        }
    }

    /// One substep: water, gravity, ground.
    fn substep(&mut self, dt: f32, water: Water, ground: &dyn Ground) {
        let (sub, front, wplane) = self.immersion(water.surface);
        self.submerged_m3 = sub;
        let rho = water.rho;
        let m_eff = self.mass + 0.5 * rho * sub;
        let mut force = Vec3::new(0.0, -self.mass * GRAVITY + rho * GRAVITY * sub, 0.0);
        if sub > 0.0 {
            let (cd, cd_v) = match self.shape {
                BodyShape::Ball { .. } => (0.47, 0.8),
                BodyShape::Crate { .. } => (1.05, 1.1),
            };
            let rel = Vec2::new(self.vel.x, self.vel.z) - water.flow;
            let drag = -0.5 * rho * cd * front * rel.length() * rel;
            // Heave: drag on the waterplane plus a radiation term, so a
            // dropped buoy rings down instead of bobbing for ever.
            let heave = -0.5 * rho * cd_v * wplane.max(front * 0.5) * self.vel.y.abs() * self.vel.y
                - 2.0 * rho * wplane * (GRAVITY * 0.2).sqrt() * self.vel.y * 0.25;
            force += Vec3::new(drag.x, heave, drag.y);
            self.drag_impulse += drag * dt;
            // Spin decays in water.
            self.omega *= 1.0 / (1.0 + 2.5 * (sub / self.volume().max(1.0e-6)) * dt);
        }
        self.vel += force / m_eff * dt;
        self.ground_contact(dt, m_eff, ground);
        self.pos += self.vel * dt;
        if self.omega.length_squared() > 1.0e-10 {
            let w = self.omega.length();
            self.rot = (Quat::from_axis_angle(self.omega / w, w * dt) * self.rot).normalize();
        }
    }

    /// Bearing, friction and rolling drag against the ground. The ground
    /// carries at most what its plate law gives at the body's underside;
    /// past that the body sinks and the press digs the print deeper.
    fn ground_contact(&mut self, dt: f32, m_eff: f32, ground: &dyn Ground) {
        let shape = self.wet_shape();
        let c = ground.cell();
        let reach = shape.reach();
        let area = c * c;
        let drop = (self.vel.y * dt).min(0.0);
        let mut cap = 0.0f32;
        let mut n = 0u32;
        let mut sink = 0.0f32;
        let mut mu = 0.0f32;
        let x0 = ((self.pos.x - reach) / c).floor() as i32;
        let x1 = ((self.pos.x + reach) / c).floor() as i32;
        let z0 = ((self.pos.z - reach) / c).floor() as i32;
        let z1 = ((self.pos.z + reach) / c).floor() as i32;
        for gj in z0..=z1 {
            for gi in x0..=x1 {
                let x = (gi as f32 + 0.5) * c;
                let z = (gj as f32 + 0.5) * c;
                let Some((lo, _)) = shape.span(x - self.pos.x, z - self.pos.z) else {
                    continue;
                };
                let g = ground.surface(x, z);
                let y = lo + drop;
                if y > g + 1.0e-3 {
                    continue;
                }
                cap += ground.pressure(x, z, y.min(g)) * area;
                sink = sink.max(g - lo);
                mu += ground.mu(x, z);
                n += 1;
            }
        }
        self.grounded = n > 0;
        if n == 0 {
            return;
        }
        mu /= n as f32;
        // Normal: stop the fall, up to what the ground carries this substep.
        let need = (-self.vel.y * m_eff).max(0.0);
        let jn = need.min(cap * dt);
        self.vel.y += jn / m_eff;
        // Anything left over pushes back up slowly when the body is buried
        // below a ground that can carry it.
        if cap > self.mass * GRAVITY * 1.5 && sink > 0.004 && self.vel.y <= 0.0 {
            self.vel.y += (sink * 2.0).min(0.2) * 0.5;
        }
        let normal = jn / dt;
        let vh = Vec2::new(self.vel.x, self.vel.z);
        match self.shape {
            BodyShape::Ball { r, .. } => {
                let inertia = self.inertia();
                // Contact point velocity: v + ω × (0, −r, 0).
                let wc = Vec3::new(self.omega.z * r, 0.0, -self.omega.x * r);
                let vc = vh + Vec2::new(wc.x, wc.z);
                let k = 1.0 / m_eff + r * r / inertia;
                let mut p = -vc / k;
                let lim = mu * jn.max(self.mass * GRAVITY * dt * 0.2);
                if p.length() > lim {
                    p = p.normalize_or_zero() * lim;
                }
                self.vel.x += p.x / m_eff;
                self.vel.z += p.y / m_eff;
                // r_c × P with r_c = (0, −r, 0).
                let torque = Vec3::new(-r * p.y, 0.0, r * p.x);
                self.omega += torque / inertia;
                // Rolling drag from the print's front wall. The horizontal
                // share of the bearing grows with the depth of the groove.
                let crr = 0.01 + 0.5 * (sink.max(0.0) / r).sqrt();
                let speed = Vec2::new(self.vel.x, self.vel.z).length();
                if speed > 1.0e-4 {
                    let dv = (crr * normal.max(self.mass * GRAVITY * 0.5) / m_eff * dt).min(speed);
                    let f = 1.0 - dv / speed;
                    self.vel.x *= f;
                    self.vel.z *= f;
                    self.omega.x *= f;
                    self.omega.z *= f;
                }
            }
            BodyShape::Crate { .. } => {
                let lim = mu * normal.max(0.0) * dt + 0.6 * sink.min(0.1) * 9.81 * dt;
                let dv = vh.length().min(lim);
                if vh.length() > 1.0e-6 {
                    let f = 1.0 - dv / vh.length();
                    self.vel.x *= f;
                    self.vel.z *= f;
                }
                self.omega.y *= (1.0 - mu * 6.0 * dt).max(0.0);
                self.omega.x = 0.0;
                self.omega.z = 0.0;
            }
        }
    }
}

/// One foot of a walker.
#[derive(Clone, Copy, Debug)]
pub struct Foot {
    /// Centre of the sole, world. Y is the underside.
    pub pos: Vec3,
    pub yaw: f32,
    pub planted: bool,
    from: Vec3,
    to: Vec3,
    /// Depth of the last print this foot made, m.
    pub print_m: f32,
}

pub const SOLE_LENGTH: f32 = 0.27;
pub const SOLE_WIDTH: f32 = 0.10;
pub const HIP_HEIGHT: f32 = 0.92;
pub const LEG_RADIUS: f32 = 0.065;
pub const WALKER_MASS: f32 = 80.0;
/// Horizontal force the walker can put into the ground to hold its pace, N.
const WALK_FORCE_N: f32 = 300.0;
const STEP_LENGTH: f32 = 0.72;
const FOOT_TRACK: f32 = 0.095;

/// A person walking a list of waypoints. Not a ragdoll: the pelvis follows a
/// pace controller and the feet a step cycle. What the world sees of it is
/// physical: each planted sole carries the body's weight into the ground,
/// and both legs stand in the sheet as columns that push water.
#[derive(Clone, Debug)]
pub struct Walker {
    /// Pelvis over the ground, world. Y is the hip.
    pub pos: Vec3,
    pub heading: f32,
    pub vel: Vec2,
    pub pace: f32,
    pub mass: f32,
    pub feet: [Foot; 2],
    /// 0..1 through the current step.
    pub phase: f32,
    /// Index of the swinging foot.
    pub swing: usize,
    pub path: Vec<Vec2>,
    pub next: usize,
    pub steps: u32,
    /// Drag on each leg since the sheet last stepped, N·s.
    leg_impulse: [Vec2; 2],
    /// Free surface at each leg on the last step.
    pub leg_water: [f32; 2],
    pub done: bool,
}

impl Walker {
    pub fn new(at: Vec2, heading: f32, pace: f32, ground: &dyn Ground) -> Self {
        let fwd = Vec2::new(heading.sin(), heading.cos());
        let right = Vec2::new(fwd.y, -fwd.x);
        let mk = |side: f32, ahead: f32| {
            let p = at + right * (FOOT_TRACK * side) + fwd * ahead;
            let y = ground.surface(p.x, p.y);
            let v = Vec3::new(p.x, y, p.y);
            Foot {
                pos: v,
                yaw: heading,
                planted: true,
                from: v,
                to: v,
                print_m: 0.0,
            }
        };
        let feet = [mk(-1.0, 0.0), mk(1.0, 0.0)];
        let y = feet[0].pos.y.max(feet[1].pos.y) + HIP_HEIGHT;
        Self {
            pos: Vec3::new(at.x, y, at.y),
            heading,
            vel: Vec2::ZERO,
            pace,
            mass: WALKER_MASS,
            feet,
            phase: 0.0,
            swing: 0,
            path: Vec::new(),
            next: 0,
            steps: 0,
            leg_impulse: [Vec2::ZERO; 2],
            leg_water: [f32::NEG_INFINITY; 2],
            done: false,
        }
    }

    pub fn forward(&self) -> Vec2 {
        Vec2::new(self.heading.sin(), self.heading.cos())
    }

    fn sole_shape(foot: &Foot, bottom: f32) -> WetShape {
        WetShape::Box {
            yaw: foot.yaw,
            half_l: SOLE_LENGTH * 0.5,
            half_w: SOLE_WIDTH * 0.5,
            bottom,
            top: bottom + 0.12,
        }
    }

    /// Sole height where the ground carries `load_n`: bisection on the plate law.
    fn bearing_y(foot: &Foot, load_n: f32, ground: &dyn Ground) -> f32 {
        let c = ground.cell();
        let area = c * c;
        let p = Vec2::new(foot.pos.x, foot.pos.z);
        let probe = Self::sole_shape(foot, 0.0);
        let reach = probe.reach();
        let mut cells: Vec<(f32, f32, f32)> = Vec::new();
        let mut top = f32::MIN;
        let x0 = ((p.x - reach) / c).floor() as i32;
        let x1 = ((p.x + reach) / c).floor() as i32;
        let z0 = ((p.y - reach) / c).floor() as i32;
        let z1 = ((p.y + reach) / c).floor() as i32;
        for gj in z0..=z1 {
            for gi in x0..=x1 {
                let x = (gi as f32 + 0.5) * c;
                let z = (gj as f32 + 0.5) * c;
                if probe.span(x - p.x, z - p.y).is_none() {
                    continue;
                }
                let g = ground.surface(x, z);
                top = top.max(g);
                cells.push((x, z, g));
            }
        }
        if cells.is_empty() {
            return ground.surface(p.x, p.y);
        }
        let carry = |y: f32| -> f32 {
            cells
                .iter()
                .map(|&(x, z, g)| if y <= g { ground.pressure(x, z, y) * area } else { 0.0 })
                .sum()
        };
        let mut hi = top;
        let mut lo = top - 0.35;
        if carry(hi) >= load_n {
            return hi;
        }
        for _ in 0..22 {
            let mid = 0.5 * (lo + hi);
            if carry(mid) >= load_n {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Legs as the sheet sees them, with the drag they took.
    fn legs(&self) -> [(Vec3, f32); 2] {
        [0, 1].map(|k| (self.feet[k].pos, self.pos.y))
    }

    fn step(&mut self, dt: f32, ground: &mut dyn Ground, water: [Water; 2]) {
        if self.done {
            self.vel = Vec2::ZERO;
        }
        // Steer for the next waypoint.
        if let Some(&target) = self.path.get(self.next) {
            let here = Vec2::new(self.pos.x, self.pos.z);
            let to = target - here;
            if to.length() < 0.35 {
                self.next += 1;
                if self.next >= self.path.len() {
                    self.done = true;
                }
            } else {
                let want = to.x.atan2(to.y);
                let mut d = want - self.heading;
                while d > std::f32::consts::PI {
                    d -= std::f32::consts::TAU;
                }
                while d < -std::f32::consts::PI {
                    d += std::f32::consts::TAU;
                }
                self.heading += d.clamp(-2.0 * dt, 2.0 * dt);
            }
        }
        let fwd = self.forward();
        let right = Vec2::new(fwd.y, -fwd.x);
        let target_v = if self.done { Vec2::ZERO } else { fwd * self.pace };
        // Pace: legs push up to WALK_FORCE_N; water drags each leg column.
        let mut f = (target_v - self.vel) * (self.mass / 0.45);
        if f.length() > WALK_FORCE_N {
            f = f.normalize() * WALK_FORCE_N;
        }
        let mut buoy = 0.0f32;
        let swing_v = self.vel * 2.0;
        for k in 0..2 {
            let w = water[k];
            self.leg_water[k] = w.surface;
            let foot = self.feet[k];
            let h = (w.surface - foot.pos.y).clamp(0.0, HIP_HEIGHT);
            if h <= 0.0 {
                continue;
            }
            let v_leg = if foot.planted { self.vel } else { swing_v };
            let rel = v_leg - w.flow;
            let area = 2.0 * LEG_RADIUS * h;
            let drag = -0.5 * w.rho * 1.0 * area * rel.length() * rel;
            f += drag;
            self.leg_impulse[k] += drag * dt;
            buoy += w.rho * GRAVITY * std::f32::consts::PI * LEG_RADIUS * LEG_RADIUS * h;
        }
        self.vel += f / self.mass * dt;
        let p2 = Vec2::new(self.pos.x, self.pos.z) + self.vel * dt;

        // Step cycle. A step takes as long as covering a step length at the
        // pace held, inside a human cadence.
        let speed = self.vel.length();
        let period = (STEP_LENGTH / speed.max(0.25)).clamp(0.38, 0.95);
        let moving = speed > 0.08 || !self.done;
        if moving {
            self.phase += dt / period;
        }
        let load = (self.mass * GRAVITY - buoy).max(self.mass * GRAVITY * 0.2);
        let s = self.swing;
        if self.phase >= 1.0 {
            // Heel strike: plant the swing foot where it was going.
            self.phase -= 1.0;
            let mut foot = self.feet[s];
            foot.pos = foot.to;
            foot.planted = true;
            self.feet[s] = foot;
            self.steps += 1;
            self.swing = 1 - s;
            let o = self.swing;
            let side = if o == 0 { -1.0 } else { 1.0 };
            let ahead = p2 + self.vel * period + fwd * (STEP_LENGTH * 0.25) * if speed > 0.08 { 1.0 } else { 0.0 };
            let land = ahead + right * (FOOT_TRACK * side);
            let mut f2 = self.feet[o];
            f2.from = f2.pos;
            f2.to = Vec3::new(land.x, ground.surface(land.x, land.y), land.y);
            f2.yaw = self.heading;
            f2.planted = false;
            self.feet[o] = f2;
        }
        // Swing foot follows an arc.
        let o = self.swing;
        if !self.feet[o].planted {
            let t = self.phase.clamp(0.0, 1.0);
            let e = t * t * (3.0 - 2.0 * t);
            let mut foot = self.feet[o];
            let lift = 0.09 * (std::f32::consts::PI * t).sin();
            foot.pos = foot.from.lerp(foot.to, e) + Vec3::Y * lift;
            self.feet[o] = foot;
        }
        // Stance: the planted soles carry the weight. Heel strike adds a
        // fifth for the first part of the step.
        let planted: Vec<usize> = (0..2).filter(|&k| self.feet[k].planted).collect();
        let strike = if self.phase < 0.2 { 1.2 } else { 1.0 };
        let share = load * strike / planted.len().max(1) as f32;
        for &k in &planted {
            let mut foot = self.feet[k];
            let y = Self::bearing_y(&foot, share, ground);
            foot.pos.y = y;
            let shape = Self::sole_shape(&foot, y);
            let out = ground.press(Vec2::new(foot.pos.x, foot.pos.z), &shape);
            if out.cells > 0 {
                foot.print_m = foot.print_m.max(out.deepest_m);
            }
            self.feet[k] = foot;
        }
        let base_y = planted
            .iter()
            .map(|&k| self.feet[k].pos.y)
            .fold(f32::NEG_INFINITY, f32::max);
        let base_y = if base_y.is_finite() { base_y } else { ground.surface(p2.x, p2.y) };
        let bob = 0.03 * (std::f32::consts::TAU * self.phase).cos();
        self.pos = Vec3::new(p2.x, base_y + HIP_HEIGHT - 0.03 + bob, p2.y);
    }
}

/// Every free body in a scene.
#[derive(Clone, Debug, Default)]
pub struct Bodies {
    pub items: Vec<Body>,
    pub walkers: Vec<Walker>,
    /// Substeps per call for the rigid bodies.
    pub substeps: u32,
}

/// Id of leg `k` of walker `w` in the sheet's load list.
pub fn leg_id(w: usize, k: usize) -> u32 {
    10_000 + (w as u32) * 2 + k as u32
}

impl Bodies {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            walkers: Vec::new(),
            substeps: 8,
        }
    }

    /// Advance every body by `dt` against the ground and the sheet as it
    /// stands. Then call [`Bodies::hand_off`] and step the sheet.
    pub fn step(&mut self, dt: f32, ground: &mut dyn Ground, fluid: Option<&FluidCore>) {
        let n = self.substeps.max(1);
        let h = dt / n as f32;
        for (i, b) in self.items.iter_mut().enumerate() {
            let water = water_at(fluid, i as u32, b.pos.x, b.pos.z);
            for _ in 0..n {
                b.substep(h, water, ground);
            }
            let out = ground.press(Vec2::new(b.pos.x, b.pos.z), &b.wet_shape());
            b.print_m = out.deepest_m;
        }
        for (w, wk) in self.walkers.iter_mut().enumerate() {
            let water = [0, 1].map(|k| {
                let f = wk.feet[k].pos;
                water_at(fluid, leg_id(w, k), f.x, f.z)
            });
            wk.step(dt, ground, water);
        }
    }

    /// Give the sheet the volumes and the drag for its next step.
    pub fn hand_off(&mut self, fluid: &mut FluidCore) {
        fluid.set_wet_bodies(&self.wet_bodies());
        for b in &mut self.items {
            b.drag_impulse = Vec2::ZERO;
        }
        for wk in &mut self.walkers {
            wk.leg_impulse = [Vec2::ZERO; 2];
        }
    }

    /// The volumes the sheet should see on its next step.
    pub fn wet_bodies(&self) -> Vec<WetBody> {
        let mut out = Vec::with_capacity(self.items.len() + self.walkers.len() * 2);
        for (i, b) in self.items.iter().enumerate() {
            out.push(WetBody {
                id: i as u32,
                x: b.pos.x,
                z: b.pos.z,
                shape: b.wet_shape(),
                impulse_x: b.drag_impulse.x,
                impulse_z: b.drag_impulse.y,
            });
        }
        for (w, wk) in self.walkers.iter().enumerate() {
            for (k, (foot, hip)) in wk.legs().into_iter().enumerate() {
                out.push(WetBody {
                    id: leg_id(w, k),
                    x: foot.x,
                    z: foot.z,
                    shape: WetShape::Column {
                        r: LEG_RADIUS,
                        bottom: foot.y,
                        top: hip,
                    },
                    impulse_x: wk.leg_impulse[k].x,
                    impulse_z: wk.leg_impulse[k].y,
                });
            }
        }
        out
    }
}
