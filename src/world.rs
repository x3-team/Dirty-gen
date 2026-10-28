//! Couples the buggy, the heightfield and splash.

use glam::Vec3;

use crate::bike::{Bike, BIKE_RADIUS, BIKE_WIDTH};
use crate::particles::{Particles, TreadThrow};
use crate::soil::{on_launch_crest, track_offset, SoilField, TRACK_R, TRACK_ZN, TRACK_ZS};
use crate::vehicle::{
    hardpoints, Contact, Vehicle, PANEL_BELLY, PANEL_FENDER_FL, PANEL_FENDER_FR, PANEL_FENDER_RL,
    PANEL_FENDER_RR, PANEL_LEFT, PANEL_MUD_CAP, PANEL_NOSE, PANEL_REAR, PANEL_RIGHT, WHEEL_RADIUS,
    WHEEL_WIDTH,
};

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub throttle: f32,
    pub brake: f32,
    pub steer: f32,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            throttle: 0.0,
            brake: 0.0,
            steer: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Telemetry {
    pub max_speed: f32,
    pub min_speed_in_soft: f32,
    pub bogged: bool,
    pub escaped: bool,
    pub max_sink: f32,
    pub splash_bursts: u32,
    pub bog_time: f32,
    pub distance: f32,
    pub left_soft: bool,
    pub impact_depth: f32,
    pub impact_sprays: u32,
    pub impact_time: f32,
    pub max_air_pitch: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub eye: Vec3,
    pub target: Vec3,
    /// Full width of an orthographic frame, in metres. Empty means a perspective lens.
    pub ortho_width: Option<f32>,
}

pub struct World {
    pub soil: SoilField,
    pub vehicle: Vehicle,
    pub bike: Bike,
    pub particles: Particles,
    pub time: f32,
    pub telemetry: Telemetry,
    drip_acc: f32,
    crest_armed: bool,
    air_time: f32,
    impact_latched: bool,
    did_launch: bool,
}

/// Wheels on one side of the axle, averaged into a single groove.
#[derive(Clone, Copy)]
struct LaneAccum {
    center: Vec3,
    forward: Vec3,
    right: Vec3,
    sink: f32,
    slip: f32,
    n: u32,
    emit_index: usize,
    speed: f32,
    contact: Vec3,
}

impl Default for LaneAccum {
    fn default() -> Self {
        Self {
            center: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
            sink: 0.0,
            slip: 0.0,
            n: 0,
            emit_index: 0,
            speed: 0.0,
            contact: Vec3::ZERO,
        }
    }
}

impl LaneAccum {
    fn add(&mut self, c: &Contact, slip: f32) {
        if self.n == 0 || c.sinkage >= self.sink {
            self.forward = c.forward;
            self.right = c.right;
            self.sink = c.sinkage;
            self.slip = slip.abs();
            self.emit_index = c.index;
            self.speed = c.speed;
            self.contact = c.contact_point;
        }
        self.center += c.wheel_center;
        self.n += 1;
    }

    fn stamp(&self, soil: &mut SoilField, radius: f32, width: f32, dt: f32) -> f32 {
        if self.n == 0 || self.sink < 0.004 {
            return 0.0;
        }
        soil.stamp_wheel(
            self.center / self.n as f32,
            radius,
            width,
            self.forward,
            self.right,
            self.sink,
            self.slip,
            dt,
            true,
        )
    }
}

impl World {
    pub fn demo() -> Self {
        let mut world = Self {
            soil: SoilField::demo(),
            vehicle: Vehicle::new(),
            bike: Bike::new(),
            particles: Particles::new(),
            time: 0.0,
            telemetry: Telemetry {
                min_speed_in_soft: 99.0,
                ..Telemetry::default()
            },
            drip_acc: 0.0,
            crest_armed: false,
            air_time: 0.0,
            impact_latched: false,
            did_launch: false,
        };
        world.place_on_soil();
        for _ in 0..120 {
            world.step(1.0 / 60.0, &Input::default());
        }
        world.vehicle.clear_loose_mud();
        world.particles.clear();
        world.time = 0.0;
        world.telemetry = Telemetry {
            min_speed_in_soft: 99.0,
            ..Telemetry::default()
        };
        world.drip_acc = 0.0;
        world.bike.place_on(&world.soil);
        world
    }

    /// Uniform flat pad. `wetness` 0 is firm, 1 is deep mud. No puddle.
    pub fn flat(wetness: f32) -> Self {
        let mut world = Self {
            soil: SoilField::flat(wetness),
            vehicle: Vehicle::new(),
            bike: Bike::new(),
            particles: Particles::new(),
            time: 0.0,
            telemetry: Telemetry {
                min_speed_in_soft: 99.0,
                ..Telemetry::default()
            },
            drip_acc: 0.0,
            crest_armed: false,
            air_time: 0.0,
            impact_latched: false,
            did_launch: false,
        };
        world.vehicle.pos = Vec3::new(0.0, 0.8, 0.0);
        world.vehicle.yaw = 0.0;
        world.place_on_soil();
        for _ in 0..90 {
            world.step(1.0 / 60.0, &Input::default());
        }
        world.vehicle.clear_loose_mud();
        world.particles.clear();
        world.time = 0.0;
        world.telemetry = Telemetry {
            min_speed_in_soft: 99.0,
            ..Telemetry::default()
        };
        world
    }

    fn place_on_soil(&mut self) {
        let sample = self.soil.sample(self.vehicle.pos.x, self.vehicle.pos.z);
        // CG about half a metre above undisturbed ground before the springs settle.
        self.vehicle.pos.y = sample.height + 0.62;
    }

    pub fn step(&mut self, dt: f32, input: &Input) {
        let substeps = 2;
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.substep(h, input);
        }
        self.time += dt;
    }

    /// Integrate the bike only. Does not advance `time`, water, or particles.
    /// Call it before [`step`](Self::step) so the buggy path tests stay put
    /// and the shared particle step still runs once.
    pub fn step_bike(&mut self, dt: f32, input: &Input) {
        let substeps = 2;
        let h = dt / substeps as f32;
        for _ in 0..substeps {
            self.bike.update_steer(input.steer, h);
            let contacts = self
                .bike
                .contacts(&self.soil, input.throttle, input.brake, h);
            if self.bike.jump_hold <= 0.0 {
                // The bike stays on the track, but its groove is not a third
                // slot beside the buggy's pair. A rolling scuff is all it cuts.
                // Spinning in place still passes the real slip and can dig.
                let crawling = self.bike.speed() < 1.1;
                for c in &contacts {
                    if c.sinkage < 0.004 {
                        continue;
                    }
                    let slip = if crawling { c.slip } else { 0.0 };
                    let sample = self.soil.sample(c.contact_point.x, c.contact_point.z);
                    let ejected = self.soil.stamp_wheel(
                        c.wheel_center,
                        BIKE_RADIUS,
                        BIKE_WIDTH,
                        c.forward,
                        c.right,
                        c.sinkage,
                        slip,
                        h,
                        false,
                    );
                    if ejected > 1.0e-6 {
                        self.particles.throw_tread(&TreadThrow {
                            origin: c.contact_point,
                            forward: c.forward,
                            right: c.right,
                            speed: self.bike.speed(),
                            slip: c.slip,
                            sample,
                            volume: ejected,
                            wheel_radius: BIKE_RADIUS,
                            wheel_width: BIKE_WIDTH,
                            outward: if c.index % 2 == 0 { -1.0 } else { 1.0 },
                        });
                    }
                }
            }
            let planted = contacts.iter().filter(|c| c.active).count();
            self.bike
                .note_flight(&mut self.soil, &mut self.particles, planted, h);
            self.bike.integrate(&contacts, h);
        }
    }

    fn substep(&mut self, dt: f32, input: &Input) {
        self.vehicle.update_steer(input.steer, dt);
        let contacts = self
            .vehicle
            .compute_contacts(&self.soil, input.throttle, input.brake, dt);

        let mut sides = [LaneAccum::default(), LaneAccum::default()];
        let mut scuff = [0.0_f32; 4];
        // At speed the visible cut is the lane ribbon. The per-tire slip dig
        // is for a tire that is spinning in place, not a second slot per wheel.
        let crawling = self.vehicle.speed() < 1.1;
        for c in &contacts {
            if !c.active {
                continue;
            }
            let slip_stamp = if crawling { c.slip } else { 0.0 };
            scuff[c.index] = self.soil.stamp_wheel(
                c.wheel_center,
                WHEEL_RADIUS,
                WHEEL_WIDTH,
                c.forward,
                c.right,
                c.sinkage,
                slip_stamp,
                dt,
                false,
            );
            // One groove per side of the axle. Front and rear share it, so a
            // turn does not chew a second slot into the berm.
            sides[c.index % 2].add(c, slip_stamp);
            if c.sample.water > 0.008 {
                let speed = c.speed.max(0.2);
                self.soil.displace_water(
                    c.wheel_center.x,
                    c.wheel_center.z,
                    c.forward,
                    c.sample.water.min(0.05) * 0.25 * (0.4 + speed * 0.15),
                );
            }
        }
        let mut side_ej = [0.0_f32; 2];
        for (i, side) in sides.iter().enumerate() {
            side_ej[i] = side.stamp(&mut self.soil, WHEEL_RADIUS, WHEEL_WIDTH, dt);
        }

        let planted = contacts.iter().filter(|c| c.active).count();
        self.note_flight(planted, dt);
        self.vehicle.integrate(&contacts, &self.soil, dt);
        self.vehicle.step_adhesion(&contacts, dt);

        let speed = self.vehicle.speed();
        for c in &contacts {
            if !c.active || c.index >= scuff.len() {
                continue;
            }
            let mut volume = scuff[c.index];
            let side = c.index % 2;
            if sides[side].n > 0 && sides[side].emit_index == c.index {
                volume += side_ej[side];
            }
            if volume <= 1.0e-6 {
                continue;
            }
            let before = self.particles.alive_count();
            self.particles.throw_tread(&TreadThrow {
                origin: c.contact_point,
                forward: c.forward,
                right: c.right,
                speed: c.speed,
                slip: c.slip,
                sample: c.sample,
                volume,
                wheel_radius: WHEEL_RADIUS,
                wheel_width: WHEEL_WIDTH,
                outward: if c.index % 2 == 0 { -1.0 } else { 1.0 },
            });
            let spawned = self.particles.alive_count().saturating_sub(before) as u32;
            if spawned > 0 {
                self.telemetry.splash_bursts = self.telemetry.splash_bursts.saturating_add(spawned);
            }
        }

        self.drip_acc += dt;
        if self.drip_acc > 0.08 {
            self.drip_acc = 0.0;
            let ori = self.vehicle.orientation();
            for (i, panel) in self.vehicle.panels.iter().enumerate() {
                if panel.mass > 0.12 && panel.drip > 0.25 {
                    let local = panel_drip_point(i);
                    let origin = self.vehicle.pos + ori * local;
                    self.particles.drip(origin);
                }
            }
        }

        self.particles.update(&mut self.soil, dt);
        self.soil.step_water(dt);

        let softness = self.vehicle.mean_softness(&contacts);
        let sink = contacts.iter().map(|c| c.sinkage).fold(0.0, f32::max);
        self.telemetry.max_sink = self.telemetry.max_sink.max(sink);
        self.telemetry.max_speed = self.telemetry.max_speed.max(speed);
        self.telemetry.distance += speed * dt;
        if softness > 0.62 {
            self.telemetry.min_speed_in_soft = self.telemetry.min_speed_in_soft.min(speed);
            if input.throttle > 0.55 && speed < 0.7 {
                self.telemetry.bog_time += dt;
                if self.telemetry.bog_time > 0.35 {
                    self.telemetry.bogged = true;
                }
            }
        } else if speed > 1.15 {
            self.telemetry.bog_time = (self.telemetry.bog_time - dt * 0.5).max(0.0);
            if self.telemetry.bogged {
                self.telemetry.escaped = true;
                self.telemetry.left_soft = true;
            }
        }
    }

    fn note_flight(&mut self, planted: usize, dt: f32) {
        let x = self.vehicle.pos.x;
        let z = self.vehicle.pos.z;
        let on_crest = on_launch_crest(x, z);
        // The face is short. Arm on the firm approach, not only in the lip box,
        // or a fast car is already rotating when the box starts and never arms.
        let approaching = (x + 6.6).abs() < 2.4
            && (-2.0..1.65).contains(&z)
            && planted >= 2
            && self.vehicle.speed() > 2.0;
        if (on_crest && planted >= 1) || approaching {
            self.crest_armed = true;
        }
        let ground = self.soil.sample(x, z).height;
        let clearance = self.vehicle.pos.y - ground;
        let flying = planted == 0 || self.vehicle.jump_hold > 0.0;
        if flying && (clearance > 0.22 || self.vehicle.jump_hold > 0.0) {
            self.air_time += dt;
            self.telemetry.max_air_pitch =
                self.telemetry.max_air_pitch.max(self.vehicle.pitch.abs());
        }
        // The lip, plus speed, is the whole trigger. No key.
        // Rear tires are still on the face when the nose clears it, so this
        // does not wait for every wheel to unload. jump_hold keeps the damper
        // from gluing the car back onto the ramp.
        let on_lip = on_crest && z > 1.55;
        if !self.did_launch && self.crest_armed && on_lip && self.vehicle.speed() > 3.15 {
            let spd = self.vehicle.speed();
            // Enough to clear the lip and tumble, short enough to land in the puddle.
            let vy = (spd * 0.85).clamp(4.2, 5.4);
            if self.vehicle.vel.y < vy {
                self.vehicle.vel.y = vy;
            }
            // Positive pitch is nose-down. This rate is nose-up, enough to tumble.
            let rate = (4.8 + spd * 0.28).clamp(5.2, 6.6);
            if self.vehicle.pitch_rate > -rate {
                self.vehicle.pitch_rate = -rate;
            }
            self.vehicle.roll_rate += 0.85;
            self.vehicle.jump_hold = 0.78;
            self.crest_armed = false;
            self.did_launch = true;
        }
        if self.did_launch
            && self.air_time > 0.40
            && !self.impact_latched
            && self.vehicle.vel.y < -2.0
            && clearance < 1.55
        {
            self.apply_impact();
            self.impact_latched = true;
        }
        if self.vehicle.jump_hold <= 0.0
            && planted >= 3
            && self.vehicle.vel.y > -0.7
            && clearance < 0.9
        {
            // Wheels ate the last of the fall. Still stamp the landing.
            if self.did_launch && !self.impact_latched && self.air_time > 0.18 {
                self.apply_impact();
                self.impact_latched = true;
            }
            self.air_time = 0.0;
            self.did_launch = false;
            if !on_crest {
                self.crest_armed = false;
            }
        }
    }

    fn apply_impact(&mut self) {
        let x = self.vehicle.pos.x;
        let z = self.vehicle.pos.z;
        let vy = self.vehicle.vel.y.abs();
        let depth = (vy * 0.024).clamp(0.10, 0.22);
        let sample = self.soil.sample(x, z);
        let dug = self.soil.stamp_impact(x, z, 1.35, depth);
        self.soil.shock_ring(x, z);
        let h = self.soil.sample(x, z).height;
        let spawned = self
            .particles
            .impact_from_cut(Vec3::new(x, h + 0.15, z), sample, dug.ejected);
        stick_impact_mud(&mut self.vehicle, &self.soil);
        self.telemetry.impact_depth = self.telemetry.impact_depth.max(dug.depth);
        self.telemetry.impact_sprays = self.telemetry.impact_sprays.saturating_add(spawned);
        self.telemetry.splash_bursts = self.telemetry.splash_bursts.saturating_add(spawned);
        if self.telemetry.impact_time <= 0.0 {
            self.telemetry.impact_time = self.time;
        }
        if self.vehicle.vel.y < 0.0 {
            self.vehicle.vel.y *= 0.22;
        }
    }

    pub fn run_seconds(&mut self, seconds: f32, input_fn: impl Fn(f32) -> Input) {
        let dt = 1.0 / 60.0;
        let steps = (seconds / dt).round() as u32;
        for _ in 0..steps {
            let input = input_fn(self.time);
            self.step(dt, &input);
        }
    }

    pub fn run_seconds_with(&mut self, seconds: f32, input_fn: impl Fn(&Self) -> Input) {
        let dt = 1.0 / 60.0;
        let steps = (seconds / dt).round() as u32;
        for _ in 0..steps {
            let input = input_fn(self);
            self.step(dt, &input);
        }
    }
}

fn stick_impact_mud(vehicle: &mut Vehicle, soil: &SoilField) {
    let ori = vehicle.orientation();
    let spots = [
        (PANEL_BELLY, Vec3::new(0.0, -0.16, 0.05)),
        (PANEL_NOSE, Vec3::new(0.0, 0.02, 1.35)),
        (PANEL_REAR, Vec3::new(0.0, 0.08, -1.15)),
        (PANEL_LEFT, Vec3::new(-0.38, 0.02, 0.2)),
        (PANEL_RIGHT, Vec3::new(0.38, 0.02, 0.2)),
        (PANEL_FENDER_FL, Vec3::new(-0.74, 0.15, 1.08)),
        (PANEL_FENDER_FR, Vec3::new(0.74, 0.15, 1.08)),
        (PANEL_FENDER_RL, Vec3::new(-0.74, 0.15, -1.18)),
        (PANEL_FENDER_RR, Vec3::new(0.74, 0.15, -1.18)),
    ];
    for (panel, local) in spots {
        let world = vehicle.pos + ori * local;
        let sample = soil.sample(world.x, world.z);
        if world.y < vehicle.pos.y + 0.05 || world.y < sample.height + 0.55 {
            let add = 0.16 + 0.18 * sample.wetness + 0.12 * sample.water.min(1.0);
            let p = &mut vehicle.panels[panel];
            p.mass = (p.mass + add).min(PANEL_MUD_CAP);
            p.wet = (p.wet + 0.7).min(1.0);
            p.streak = (p.streak + 0.45).min(1.0);
            p.drip = (p.mass / PANEL_MUD_CAP).clamp(0.0, 1.0);
        }
    }
}

fn panel_drip_point(index: usize) -> Vec3 {
    match index {
        1 => Vec3::new(0.0, 0.15, 1.55),
        2 => Vec3::new(-0.55, 0.05, 0.2),
        3 => Vec3::new(0.55, 0.05, 0.2),
        4 => Vec3::new(0.0, 0.2, -1.45),
        5 => Vec3::new(0.0, -0.28, 0.0),
        6 => Vec3::new(-0.74, 0.15, 1.08),
        7 => Vec3::new(0.74, 0.15, 1.08),
        8 => Vec3::new(-0.74, 0.15, -1.18),
        9 => Vec3::new(0.74, 0.15, -1.18),
        _ => hardpoints()[0] * 0.0 + Vec3::new(0.0, 0.4, 0.0),
    }
}

/// One lap of the mud loop. Pure pursuit on the stadium centerline.
/// Deterministic: same pose always produces the same pedal and wheel.
pub fn demo_drive(world: &World) -> Input {
    chase_line(
        world.time,
        world.vehicle.pos,
        world.vehicle.yaw,
        world.vehicle.speed(),
        &world.soil,
        0.0,
    )
}

/// Same loop as the buggy, one tyre-width inside it, so the two stay side by side.
pub fn demo_drive_bike(world: &World) -> Input {
    chase_line(
        world.time,
        world.bike.pos,
        world.bike.yaw,
        world.bike.speed(),
        &world.soil,
        -0.55,
    )
}

fn chase_line(
    time: f32,
    pos: Vec3,
    yaw: f32,
    speed: f32,
    soil: &SoilField,
    lane_bias: f32,
) -> Input {
    if time < 0.50 {
        return Input::default();
    }
    let s = closest_s(pos);
    let off = track_offset(pos.x, pos.z);
    // Short lookahead when the buggy drifts, so it stays on the ribbon
    // instead of riding the berm or cutting the infield pond.
    let look = if off.abs() > 0.65 { 1.8 } else { 3.0 };
    let ahead = point_on_track(s + look);
    // The scripted lap was riding the outside berm and missing the lip.
    // Aim at the lip until the car is on it. The player gets the same crest.
    let aim_lip = pos.x < -1.0 && (-6.3..2.4).contains(&pos.z);
    let desired = if pos.x < -2.2 && (-4.2..2.4).contains(&pos.z) {
        // Due north on the straight. A point ahead and to the west kept the
        // nose pointed off the puddle, so the landing missed the water.
        ((-6.35 - pos.x) * 0.28).clamp(-0.42, 0.42)
    } else if aim_lip {
        (-6.35 - pos.x).atan2(-0.4 - pos.z)
    } else {
        (ahead.x - pos.x).atan2(ahead.z - pos.z)
    };
    let mut err = wrap_pi(desired - yaw);
    // Positive offset is outside the loop (left of the clockwise heading).
    if !aim_lip {
        err += ((off - lane_bias) * 0.22).clamp(-0.55, 0.55);
    }
    let steer = (err / if aim_lip { 0.36 } else { 0.30 }).clamp(-1.0, 1.0);
    let under = soil.sample(pos.x, pos.z);
    // Full throttle on the firm run-up. The lip, not a key, is what launches.
    let on_runup = aim_lip || (pos.x < -4.0 && (-3.6..3.0).contains(&pos.z));
    // Ease off on the straight, and brake in standing water so the tires
    // slip, throw spray, and cut a rut before the buggy climbs back out.
    let (throttle, brake) = if on_runup && under.water < 0.04 {
        if speed > 4.25 {
            (0.08, 0.0)
        } else {
            (1.0, 0.0)
        }
    } else if under.water > 0.05 && speed > 0.62 {
        (0.0, 1.0)
    } else if speed > 3.05 {
        (0.1, 0.0)
    } else {
        (1.0, 0.0)
    };
    Input {
        throttle,
        brake,
        steer,
    }
}

/// Slow entry so Bingham drag bites, then a full-throttle dig, then reverse.
/// Brake to a stop in the water, then spin. The rut should lock the buggy.
pub fn dig_input(t: f32) -> Input {
    if t < 1.15 {
        Input::default()
    } else if t < 3.15 {
        Input {
            throttle: 1.0,
            brake: 0.0,
            steer: 0.0,
        }
    } else if t < 4.3 {
        Input {
            throttle: 0.0,
            brake: 1.0,
            steer: 0.0,
        }
    } else {
        Input {
            throttle: 1.0,
            brake: 0.0,
            steer: 0.0,
        }
    }
}

/// Fixed oblique orthographic view of the whole loop. The chase camera is a toggle.
pub fn demo_camera(_t: f32, _pos: Vec3, _yaw: f32) -> CameraPose {
    let target = Vec3::new(0.0, 0.4, 1.4);
    let eye = target + Vec3::new(-14.5, 12.2, -13.0);
    CameraPose {
        eye,
        target,
        ortho_width: Some(25.5),
    }
}

/// A short pass for the tread-spray stills. The left tires are in slurry and
/// the right tires are on dry broken ground, so the throw is not one brown.
pub struct SprayFrame {
    pub world: World,
    pub tire: CameraPose,
    pub side: CameraPose,
}

pub fn spray_frame() -> SprayFrame {
    let mut world = World::flat(0.82);
    let half = world.soil.size * 0.5;
    let res = world.soil.res;
    let cell = world.soil.cell;
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5) * cell - half;
            if x > 0.15 {
                let idx = j * res + i;
                world.soil.wetness[idx] = 0.14;
                world.soil.base_wetness[idx] = 0.14;
                world.soil.water[idx] = 0.0;
                world.soil.roughness[idx] = 0.92;
                world.soil.loose[idx] = 0.0;
                world.soil.darken[idx] = 0.04;
            }
        }
    }
    world.particles.clear();
    // The settle pass parks a second vehicle in the shot. Keep it out of frame.
    world.bike.pos = Vec3::new(0.0, -40.0, 0.0);
    world.vehicle.pos = Vec3::new(0.0, 1.05, -2.6);
    world.vehicle.vel = Vec3::new(0.0, 0.0, 3.5);
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.yaw_rate = 0.0;
    world.vehicle.pitch_rate = 0.0;
    world.vehicle.roll_rate = 0.0;
    for w in &mut world.vehicle.wheels {
        w.omega = 16.0;
    }
    world.place_on_soil();
    let input = Input {
        throttle: 1.0,
        brake: 0.28,
        steer: 0.0,
    };
    for _ in 0..22 {
        world.step(1.0 / 60.0, &input);
    }
    let v = &world.vehicle;
    let ori = v.orientation();
    let fwd = ori * Vec3::Z;
    let right = ori * Vec3::X;
    let fl = v.pos + ori * v.wheel_center_local(0);
    // Rear sidewall of the front-left tire, where the slurry leaves.
    let exit = fl - fwd * (WHEEL_RADIUS * 1.05) - right * (WHEEL_WIDTH * 0.55) + Vec3::Y * 0.02;
    let tire = CameraPose {
        eye: exit - right * 1.2 - fwd * 0.42 + Vec3::Y * 0.28,
        target: exit - fwd * 0.05 + Vec3::Y * 0.22,
        ortho_width: None,
    };
    let side = CameraPose {
        eye: v.pos - right * 4.8 + Vec3::Y * 1.2 + fwd * 0.35,
        target: v.pos + Vec3::Y * 0.85 - fwd * 0.15,
        ortho_width: None,
    };
    SprayFrame { world, tire, side }
}

fn track_perimeter() -> f32 {
    let straight = TRACK_ZN - TRACK_ZS;
    let half = std::f32::consts::PI * TRACK_R;
    2.0 * straight + 2.0 * half
}

/// Clockwise from the south point: west along the south cap, north up the west straight.
fn point_on_track(s: f32) -> Vec3 {
    let straight = TRACK_ZN - TRACK_ZS;
    let q = std::f32::consts::FRAC_PI_2 * TRACK_R;
    let half = std::f32::consts::PI * TRACK_R;
    let total = track_perimeter();
    let s = s.rem_euclid(total);
    if s < q {
        let a = -std::f32::consts::FRAC_PI_2 - (s / q) * std::f32::consts::FRAC_PI_2;
        Vec3::new(TRACK_R * a.cos(), 0.0, TRACK_ZS + TRACK_R * a.sin())
    } else if s < q + straight {
        let t = (s - q) / straight;
        Vec3::new(-TRACK_R, 0.0, TRACK_ZS + t * straight)
    } else if s < q + straight + half {
        let u = (s - q - straight) / half;
        let a = std::f32::consts::PI - u * std::f32::consts::PI;
        Vec3::new(TRACK_R * a.cos(), 0.0, TRACK_ZN + TRACK_R * a.sin())
    } else if s < q + 2.0 * straight + half {
        let t = (s - q - straight - half) / straight;
        Vec3::new(TRACK_R, 0.0, TRACK_ZN - t * straight)
    } else {
        let u = (s - q - 2.0 * straight - half) / q;
        let a = -u * std::f32::consts::FRAC_PI_2;
        Vec3::new(TRACK_R * a.cos(), 0.0, TRACK_ZS + TRACK_R * a.sin())
    }
}

fn closest_s(pos: Vec3) -> f32 {
    let total = track_perimeter();
    let n = 96;
    let mut best_s = 0.0;
    let mut best_d = f32::MAX;
    for i in 0..n {
        let s = total * (i as f32) / n as f32;
        let p = point_on_track(s);
        let d = (p.x - pos.x).powi(2) + (p.z - pos.z).powi(2);
        if d < best_d {
            best_d = d;
            best_s = s;
        }
    }
    let mut s = best_s;
    let mut step = total / n as f32;
    for _ in 0..5 {
        step *= 0.5;
        let dl = {
            let p = point_on_track(s - step);
            (p.x - pos.x).powi(2) + (p.z - pos.z).powi(2)
        };
        let dr = {
            let p = point_on_track(s + step);
            (p.x - pos.x).powi(2) + (p.z - pos.z).powi(2)
        };
        if dl < dr && dl < best_d {
            s -= step;
            best_d = dl;
        } else if dr < best_d {
            s += step;
            best_d = dr;
        }
    }
    s.rem_euclid(total)
}

fn wrap_pi(mut a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    while a > pi {
        a -= pi * 2.0;
    }
    while a < -pi {
        a += pi * 2.0;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::{FACE_GRAVEL, FACE_MUD, SHAPE_CLOD, SHAPE_DROP, SHAPE_DUST, SHAPE_RIBBON};

    #[test]
    fn spray_pass_throws_slurry_and_gravel() {
        let frame = spray_frame();
        let mut mud_ribbon = 0;
        let mut mud_drop = 0;
        let mut gravel_clod = 0;
        let mut gravel_dust = 0;
        let mut air = 0.0_f32;
        for p in frame.world.particles.items.iter().filter(|p| p.alive) {
            if !p.grounded {
                air += p.volume;
            }
            if p.face == FACE_MUD && p.shape == SHAPE_RIBBON {
                mud_ribbon += 1;
            }
            if p.face == FACE_MUD && p.shape == SHAPE_DROP {
                mud_drop += 1;
            }
            if p.face == FACE_GRAVEL && p.shape == SHAPE_CLOD {
                gravel_clod += 1;
            }
            if p.face == FACE_GRAVEL && p.shape == SHAPE_DUST {
                gravel_dust += 1;
            }
        }
        eprintln!(
            "speed {:.2} pos {:?} ribbons {mud_ribbon} drops {mud_drop} clods {gravel_clod} dust {gravel_dust} air {air:.5}",
            frame.world.vehicle.speed(),
            frame.world.vehicle.pos
        );
        assert!(mud_ribbon > 4, "slurry ribbons {mud_ribbon}");
        assert!(mud_drop > 2, "slurry drops {mud_drop}");
        assert!(gravel_clod > 2, "gravel clods {gravel_clod}");
        assert!(gravel_dust > 2, "gravel dust {gravel_dust}");
        assert!(air > 1.0e-4, "nothing in the air");
    }
}
