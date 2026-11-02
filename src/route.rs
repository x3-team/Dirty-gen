//! The Niva route: one run through every surface the slice has, in the order a
//! driver meets them, with the car's mud coat and the route clock carried from
//! leg to leg.
//!
//! 1. `drive`: [`ThroughDrive`]: puddle, clay slurry ruts, dry exit with dust.
//! 2. `crust`: the drying pad: mud, sand, gravel and grass seams with a
//!    cracked crust ([`World::drying_crust_frame`]).
//! 3. `shore`: [`ShoreLeg`]: the ford with its swell. The Niva rolls off the
//!    sand beach through the swash and crosses the river among the rocks while
//!    a person wades the shallows beside it.
//! 4. `snow`: [`SnowLeg`]: deep snow on the 2.5 cm track, ruts, berms, belly.
//!
//! The legs are separate solver scenes. The soil map and the sheet window are
//! 40 m, and the route is longer than that; there is no streaming between
//! scenes yet. What carries over is the car: its panels and wheels keep the
//! mud they picked up. Nothing here draws a surface. Every leg is the same
//! solver the bench and the tests run.
//!
//! [`RouteClock`] times each solver step on the CPU, per leg, so the route
//! reports what a frame costs on each surface.

use glam::{Vec2, Vec3};

use crate::bodies::{bed_base, Bodies, Body, PrintGround, Walker};
use crate::ford::{still_waterline_z, FordDrive, Swell};
use crate::imprint::Imprint;
use crate::snow_track::{step_vehicle, DriveSegment, SnowTrack};
use crate::vehicle::Vehicle;
use crate::world::World;
use crate::ThroughDrive;

/// Legs in route order.
pub const LEGS: [&str; 4] = ["drive", "crust", "shore", "snow"];

/// Copy the mud the car carries: panel coats and the tires' mud, smear,
/// sidewall and tread fill. Pose, speed and suspension stay with the leg.
pub fn carry_coat(from: &Vehicle, to: &mut Vehicle) {
    to.panels = from.panels;
    for (a, b) in from.wheels.iter().zip(to.wheels.iter_mut()) {
        b.mud = a.mud;
        b.smear = a.smear;
        b.sidewall = a.sidewall;
        b.groove = a.groove;
    }
}

/// CPU time of solver steps, per leg.
#[derive(Clone, Debug, Default)]
pub struct LegTimes {
    pub leg: String,
    /// Simulated seconds per step.
    pub dt: f32,
    pub step_ms: Vec<f64>,
    /// One-off set-up work (settle, drying), not per frame.
    pub setup_ms: f64,
}

impl LegTimes {
    pub fn new(leg: &str, dt: f32) -> Self {
        Self {
            leg: leg.to_string(),
            dt,
            ..Default::default()
        }
    }

    pub fn mean(&self) -> f64 {
        if self.step_ms.is_empty() {
            return 0.0;
        }
        self.step_ms.iter().sum::<f64>() / self.step_ms.len() as f64
    }

    pub fn pct(&self, p: f64) -> f64 {
        if self.step_ms.is_empty() {
            return 0.0;
        }
        let mut v = self.step_ms.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        let i = ((v.len() - 1) as f64 * p).round() as usize;
        v[i.min(v.len() - 1)]
    }

    pub fn max(&self) -> f64 {
        self.step_ms.iter().cloned().fold(0.0, f64::max)
    }

    /// CPU milliseconds per 1/60 s of simulated time.
    pub fn ms_per_60hz_frame(&self) -> f64 {
        if self.dt <= 0.0 {
            return 0.0;
        }
        self.mean() * (1.0 / 60.0) / self.dt as f64
    }
}

/// Timer around one step.
pub struct StepTimer(std::time::Instant);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leg_times_stats() {
        let mut t = LegTimes::new("x", 1.0 / 30.0);
        t.step_ms = (1..=100).map(|v| v as f64).collect();
        assert!((t.mean() - 50.5).abs() < 1e-9);
        assert_eq!(t.pct(0.95), 95.0);
        assert_eq!(t.max(), 100.0);
        assert!((t.ms_per_60hz_frame() - 25.25).abs() < 1e-3);
    }
}

impl StepTimer {
    pub fn start() -> Self {
        Self(std::time::Instant::now())
    }
    pub fn ms(&self) -> f64 {
        self.0.elapsed().as_secs_f64() * 1000.0
    }
}

/// Leg 3: the ford with its swell, the Niva crossing from the beach, and a
/// person wading the shallows to the left of the car's line. The walker's
/// legs are volumes in the sheet and the car's wheels and belly are walls in
/// it, both in the same step.
pub struct ShoreLeg {
    pub ford: FordDrive,
    pub bodies: Bodies,
    pub imprint: Imprint,
}

/// Beach depth grows 1 m in 11 north of the still waterline.
fn beach_z(depth_m: f32) -> f32 {
    still_waterline_z() + depth_m * 11.0
}

impl ShoreLeg {
    pub fn new(settle_s: f32) -> Self {
        let ford = FordDrive::new(Some(Swell::default()), settle_s);
        let mut imprint = Imprint::new();
        let mut bodies = Bodies::new();
        {
            let ground = PrintGround {
                imprint: &mut imprint,
                base: bed_base(&ford.fluid),
            };
            // Off the dry sand into the swash, then out to knee depth, always
            // left of the car's line (x = 0, the car is 1.7 m wide).
            let mut w = Walker::new(Vec2::new(-5.0, beach_z(-0.15)), 0.8, 1.35, &ground);
            w.path = vec![
                Vec2::new(-3.6, beach_z(0.10)),
                Vec2::new(-2.6, beach_z(0.28)),
                Vec2::new(-2.4, beach_z(0.40)),
            ];
            bodies.walkers.push(w);
        }
        for (x, d) in [(-4.2f32, 0.30f32), (3.2, 0.26)] {
            let z = beach_z(d);
            let y = ford.fluid.sample(x, z).bed_m + d + 0.05;
            bodies.items.push(Body::ball(Vec3::new(x, y, z), 0.2, 6.0, true));
        }
        Self {
            ford,
            bodies,
            imprint,
        }
    }

    pub fn time_s(&self) -> f32 {
        self.ford.time_s
    }

    pub fn step(&mut self, dt: f32) {
        {
            let mut ground = PrintGround {
                imprint: &mut self.imprint,
                base: bed_base(&self.ford.fluid),
            };
            self.bodies.step(dt, &mut ground, Some(&self.ford.fluid));
        }
        self.bodies.hand_off(&mut self.ford.fluid);
        self.ford.step(dt);
        self.ford.world.time += dt;
    }
}

/// Leg 4: the real Niva on the 2.5 cm snow track: a straight run into a
/// left-hand turn and out. The coarse soil copies the track surface so the
/// ground outside the fine window matches it.
pub struct SnowLeg {
    pub track: SnowTrack,
    pub world: World,
    pub blanket_m: f32,
    pub time_s: f32,
    script: Vec<DriveSegment>,
    flat: crate::soil::SoilField,
}

/// Substep of the snow track, seconds. Same as `snow_track::drive_script`.
pub const SNOW_H: f32 = 1.0 / 120.0;

impl SnowLeg {
    pub fn new(blanket_m: f32) -> Self {
        let mut track = SnowTrack::new(blanket_m, 0.0);
        track.rho_blanket = 160.0;
        let mut world = World::flat(0.1);
        world.soil = crate::soil::SoilField::flat(0.1);
        world.particles = crate::particles::Particles::new();
        world.vehicle.clear_loose_mud();
        world.vehicle.vel = Vec3::ZERO;
        world.vehicle.yaw = 0.0;
        world.vehicle.pos = Vec3::new(-10.0, blanket_m + 0.38 + 0.45, -10.0);
        world.bike.pos = Vec3::new(80.0, 0.0, 80.0);
        let flat = world.soil.clone();
        let seg = |seconds: f32, speed: f32, steer: f32| DriveSegment {
            seconds,
            speed,
            steer,
        };
        Self {
            track,
            world,
            blanket_m,
            time_s: 0.0,
            script: vec![
                seg(5.0, 2.6, 0.0),
                seg(6.0, 2.3, 0.7),
                seg(4.0, 2.6, 0.0),
                seg(30.0, 0.0, 0.0),
            ],
            flat,
        }
    }

    fn segment(&self) -> DriveSegment {
        let mut t = self.time_s;
        for s in &self.script {
            if t < s.seconds {
                return *s;
            }
            t -= s.seconds;
        }
        *self.script.last().expect("script")
    }

    /// Two 120 Hz substeps: one 1/60 s frame.
    pub fn step_frame(&mut self) {
        for _ in 0..2 {
            let seg = self.segment();
            step_vehicle(&mut self.track, &mut self.world.vehicle, &self.flat, &seg, SNOW_H);
            self.time_s += SNOW_H;
        }
    }

    /// Copy the track surface onto the coarse soil, for the picture.
    pub fn paint_soil(&mut self) {
        let soil = &mut self.world.soil;
        let half = soil.size * 0.5;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                let idx = j * soil.res + i;
                soil.height[idx] = self.track.surface_at(x, z);
                soil.darken[idx] = 0.0;
                soil.water[idx] = 0.0;
                soil.slurry[idx] = 0.0;
                soil.wetness[idx] = 0.0;
            }
        }
    }
}

/// Leg 1 helper: a fresh through-drive, and the stop past which it is done.
pub fn drive_leg() -> ThroughDrive {
    ThroughDrive::new()
}

/// Last chassis z the through-drive reaches.
pub const DRIVE_END_Z: f32 = 19.6;

/// Leg 2: the drying pad. Returns the world and the one-off drying cost.
pub fn crust_leg() -> (World, f64) {
    let t = StepTimer::start();
    let mut world = World::wet_clay();
    let _ = world.drying_crust_frame();
    seam_band(&mut world.soil);
    world.soil.blend_material_joints();
    (world, t.ms())
}

/// Grass, sand and gravel strips beyond the rolled lanes (z 3.4..8), next to
/// the clay pad that stays east of x = 3. The joints between them come from
/// [`crate::soil::SoilField::blend_material_joints`]. The drying spell only
/// cracks clay, so the painted strips carry no crust.
fn seam_band(soil: &mut crate::soil::SoilField) {
    use crate::soil::SurfaceKind;
    let half = soil.size * 0.5;
    for j in 0..soil.res {
        for i in 0..soil.res {
            let x = (i as f32 + 0.5) * soil.cell - half;
            let z = (j as f32 + 0.5) * soil.cell - half;
            if !(3.4..8.0).contains(&z) || !(-9.0..3.0).contains(&x) {
                continue;
            }
            let (kind, wet) = if x < -3.0 {
                (SurfaceKind::Grass, 0.42)
            } else if x < 0.0 {
                (SurfaceKind::Sand, 0.30)
            } else {
                (SurfaceKind::Gravel, 0.08)
            };
            let idx = j * soil.res + i;
            soil.kind[idx] = kind as u8;
            soil.wetness[idx] = wet;
            soil.base_wetness[idx] = wet;
            soil.crust[idx] = 0.0;
            soil.crack[idx] = 0.0;
        }
    }
}
