//! Couples the buggy, the heightfield and splash.

use glam::Vec3;

use crate::bike::{Bike, BIKE_RADIUS, BIKE_WIDTH};
use crate::particles::Particles;
use crate::patch::SIDEWALL_VOLUME;
use crate::soil::{
    on_launch_crest, stick_scale, track_offset, SoilField, TRACK_R, TRACK_ZN, TRACK_ZS,
};
use crate::vehicle::{
    hardpoints, Vehicle, PANEL_BELLY, PANEL_FENDER_FL, PANEL_FENDER_FR, PANEL_FENDER_RL,
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
    spawn_pos: Vec3,
    spawn_yaw: f32,
    bike_spawn: Vec3,
    bike_spawn_yaw: f32,
    /// Cubic metres of sidewall coat that landed back in a rut.
    pub coat_in_rut: f32,
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
        }
    }
}

impl LaneAccum {
    fn add(&mut self, center: Vec3, forward: Vec3, right: Vec3, sink: f32, slip: f32) {
        if self.n == 0 || sink >= self.sink {
            self.forward = forward;
            self.right = right;
            self.sink = sink;
            self.slip = slip.abs();
        }
        self.center += center;
        self.n += 1;
    }

    fn stamp(&self, soil: &mut SoilField, radius: f32, width: f32, dt: f32) {
        if self.n == 0 || self.sink < 0.004 {
            return;
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
        );
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
            spawn_pos: Vec3::ZERO,
            spawn_yaw: 0.0,
            bike_spawn: Vec3::ZERO,
            bike_spawn_yaw: 0.0,
            coat_in_rut: 0.0,
        };
        world.place_on_soil();
        for _ in 0..120 {
            world.step(1.0 / 60.0, &Input::default());
        }
        world.restore_sidewall_skim();
        world.vehicle.clear_loose_mud();
        world.particles.clear();
        world.time = 0.0;
        world.telemetry = Telemetry {
            min_speed_in_soft: 99.0,
            ..Telemetry::default()
        };
        world.drip_acc = 0.0;
        world.coat_in_rut = 0.0;
        world.bike.place_on(&world.soil);
        world.remember_spawn();
        world
    }

    /// Uniform flat pad. `wetness` 0 is firm, 1 is deep mud. No puddle.
    pub fn flat(wetness: f32) -> Self {
        Self::rest(SoilField::flat(wetness))
    }

    /// Niva settled across wet clay and gravel. The bed starts flat.
    pub fn surface_split() -> Self {
        Self::rest(SoilField::surface_split())
    }

    fn rest(soil: SoilField) -> Self {
        let mut world = Self {
            soil,
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
            spawn_pos: Vec3::ZERO,
            spawn_yaw: 0.0,
            bike_spawn: Vec3::ZERO,
            bike_spawn_yaw: 0.0,
            coat_in_rut: 0.0,
        };
        world.vehicle.pos = Vec3::new(0.0, 0.8, 0.0);
        world.vehicle.yaw = 0.0;
        world.place_on_soil();
        for _ in 0..90 {
            world.step(1.0 / 60.0, &Input::default());
        }
        world.restore_sidewall_skim();
        world.vehicle.clear_loose_mud();
        world.particles.clear();
        world.coat_in_rut = 0.0;
        world.time = 0.0;
        world.telemetry = Telemetry {
            min_speed_in_soft: 99.0,
            ..Telemetry::default()
        };
        world.remember_spawn();
        world
    }

    fn remember_spawn(&mut self) {
        self.spawn_pos = self.vehicle.pos;
        self.spawn_yaw = self.vehicle.yaw;
        self.bike_spawn = self.bike.pos;
        self.bike_spawn_yaw = self.bike.yaw;
    }

    /// Put both vehicles back on their start poses. The heightfield, the
    /// berm, and any mud still in the air stay where they are.
    pub fn reset_vehicle(&mut self) {
        self.vehicle = Vehicle::new();
        self.vehicle.pos = self.spawn_pos;
        self.vehicle.yaw = self.spawn_yaw;
        self.place_on_soil();
        self.bike = Bike::new();
        self.bike.pos.x = self.bike_spawn.x;
        self.bike.pos.z = self.bike_spawn.z;
        self.bike.yaw = self.bike_spawn_yaw;
        self.bike.place_on(&self.soil);
        self.crest_armed = false;
        self.air_time = 0.0;
        self.impact_latched = false;
        self.did_launch = false;
        self.drip_acc = 0.0;
    }

    /// The spawn settle can pack a coat. That is not a shed. Put the skim
    /// back into the loose smear so the parking step does not delete it
    /// and does not leave a pile that the cutter did not make.
    fn restore_sidewall_skim(&mut self) {
        for i in 0..4 {
            let vol = self.vehicle.wheels[i].sidewall.clamp(0.0, 1.0) * SIDEWALL_VOLUME;
            if vol > 1.0e-8 {
                let center = self.vehicle.pos
                    + self.vehicle.orientation() * self.vehicle.wheel_center_local(i);
                self.soil.give_loose(center, vol);
            }
            self.vehicle.wheels[i].sidewall = 0.0;
        }
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
                    self.soil.stamp_wheel(
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
        // At speed the visible cut is the lane ribbon. The per-tire slip dig
        // is for a tire that is spinning in place, not a second slot per wheel.
        let crawling = self.vehicle.speed() < 1.1;
        for c in &contacts {
            if !c.active {
                continue;
            }
            let slip_stamp = if crawling { c.slip } else { 0.0 };
            self.soil.stamp_wheel(
                c.wheel_center,
                WHEEL_RADIUS,
                crate::vehicle::WHEEL_WIDTH,
                c.forward,
                c.right,
                c.sinkage,
                slip_stamp,
                dt,
                false,
            );
            // The lug negative is a record of this contact. It does not cut.
            self.soil.imprint_tread(
                c.wheel_center,
                c.forward,
                c.right,
                WHEEL_RADIUS,
                crate::vehicle::WHEEL_WIDTH,
                c.sinkage,
            );
            // One groove per side of the axle. Front and rear share it, so a
            // turn does not chew a second slot into the berm.
            sides[c.index % 2].add(c.wheel_center, c.forward, c.right, c.sinkage, slip_stamp);
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
        let width = crate::vehicle::WHEEL_WIDTH;
        for side in &sides {
            side.stamp(&mut self.soil, WHEEL_RADIUS, width, dt);
        }

        let planted = contacts.iter().filter(|c| c.active).count();
        self.note_flight(planted, dt);
        self.vehicle.integrate(&contacts, &self.soil, dt);
        let tread = self.vehicle.step_adhesion(&contacts, dt);
        for c in &contacts {
            if !c.active {
                continue;
            }
            let step = tread[c.index];
            if step.picked > 1.0e-7 {
                let want = step.picked * SIDEWALL_VOLUME;
                let got = self.soil.take_loose(
                    c.wheel_center,
                    c.forward,
                    c.right,
                    c.patch_length.max(0.08),
                    WHEEL_WIDTH,
                    want,
                );
                if want - got > 1.0e-8 {
                    let short = (want - got) / SIDEWALL_VOLUME;
                    let w = &mut self.vehicle.wheels[c.index];
                    w.sidewall = (w.sidewall - short).max(0.0);
                }
            }
            if step.shed > 1.0e-7 {
                let vol = step.shed * SIDEWALL_VOLUME;
                let land = self.soil.return_coat(c.wheel_center, c.forward, c.right, vol);
                self.coat_in_rut += land.in_rut;
            }
        }
        for c in &contacts {
            if c.active && c.squeeze > 0.02 {
                // Loose smear only. The cut still lands on the berm.
                self.soil.skim_loose(
                    c.wheel_center,
                    c.forward,
                    c.right,
                    c.patch_length.max(0.08),
                    crate::vehicle::WHEEL_WIDTH,
                    c.squeeze * 0.0025 * dt,
                );
            }
        }

        let speed = self.vehicle.speed();
        for c in &contacts {
            if !c.active {
                continue;
            }
            if c.sample.water > 0.012 && speed > 0.35 {
                self.particles
                    .spray(c.contact_point, c.forward, c.right, speed, c.sample.water);
                self.telemetry.splash_bursts = self.telemetry.splash_bursts.saturating_add(1);
            } else if c.sample.wetness > 0.62 && speed > 1.1 && c.sinkage > 0.02 {
                if (self.particles.alive_count() as u32 + self.time.to_bits()) % 3 == 0 {
                    self.particles
                        .clod(c.contact_point, c.forward, c.right, speed);
                }
            }
            if c.fling > 0.15 && c.slip.abs() > 0.25 {
                self.particles.clod(
                    c.wheel_center + Vec3::Y * 0.05,
                    c.forward,
                    c.right,
                    speed.max(1.0),
                );
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
        let dug = self.soil.stamp_impact(x, z, 1.35, depth);
        self.soil.shock_ring(x, z);
        let h = self.soil.sample(x, z).height;
        let spawned = self.particles.impact_burst(Vec3::new(x, h + 0.15, z));
        stick_impact_mud(&mut self.vehicle, &self.soil);
        self.telemetry.impact_depth = self.telemetry.impact_depth.max(dug);
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

    /// Wet clay pad with the Niva sitting on the surface. No settle rut.
    pub fn wet_clay() -> Self {
        let mut world = Self {
            soil: SoilField::mud_pad(),
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
            spawn_pos: Vec3::ZERO,
            spawn_yaw: 0.0,
            bike_spawn: Vec3::new(14.0, 0.5, 14.0),
            bike_spawn_yaw: 0.0,
            coat_in_rut: 0.0,
        };
        world.bike.pos = world.bike_spawn;
        world.vehicle.pos = Vec3::new(0.0, 0.7, 0.0);
        world.vehicle.yaw = 0.0;
        world.vehicle.roll = 0.0;
        world
    }

    /// Roll the Niva onto its side while it yaws in the clay, then settle it
    /// in a second pose beside the furrow. Cut, berm, and thrown mass come
    /// from the body and the tires. The bed under the roll is not filled.
    pub fn roll_in_clay(&mut self) -> RollCut {
        let before = self.soil.volumes();
        let mut body = 0.0_f32;
        let mut tires = 0.0_f32;
        let mut berm = 0.0_f32;
        let mut thrown = 0.0_f32;
        let steps = 32;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let s = t * t * (3.0 - 2.0 * t);
            let moving = (s * (1.0 - s) * 4.0).clamp(0.0, 1.0);
            let roll = s * 1.48;
            let yaw = s * 1.22;
            let embed = 0.035 + 0.12 * s;
            let frac = 0.16 + 0.46 * moving;
            let frame = self.plow_pose(roll, yaw, embed, Vec3::ZERO, frac);
            body += frame.body;
            tires += frame.tires;
            berm += frame.berm;
            thrown += frame.thrown;
        }
        let mut scar: Vec<(usize, f32)> = Vec::new();
        for idx in 0..self.soil.bed.len() {
            let depth = self.soil.initial_height[idx] - self.soil.bed[idx];
            if depth > 0.03 {
                scar.push((idx, self.soil.bed[idx]));
            }
        }
        let bed = scar
            .iter()
            .map(|(idx, _)| self.soil.initial_height[*idx] - self.soil.bed[*idx])
            .fold(0.0_f32, f32::max);
        let yaw2 = 1.70_f32;
        let fwd = Vec3::new(yaw2.sin(), 0.0, yaw2.cos());
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let shift = right * 1.9 + fwd * 0.25;
        let frame = self.plow_pose(1.22, yaw2, 0.055, shift, 0.22);
        body += frame.body;
        tires += frame.tires;
        berm += frame.berm;
        thrown += frame.thrown;
        for _ in 0..48 {
            self.particles.update(&mut self.soil, 1.0 / 30.0);
        }
        self.particles.land_ledger(&mut self.soil);
        let mut bed_after = 0.0_f32;
        let mut raised = false;
        for (idx, was) in &scar {
            if self.soil.bed[*idx] > was + 1.0e-4 {
                raised = true;
            }
            bed_after = bed_after.max(self.soil.initial_height[*idx] - self.soil.bed[*idx]);
        }
        let after = self.soil.volumes();
        let roll = RollCut {
            displaced: after.cut - before.cut,
            berm,
            thrown,
            body,
            tires,
            bed,
            bed_after,
            raised,
            airborne: self.soil.airborne,
            placed: after.placed - before.placed,
        };
        // Numbers above are the cut. The picture is the same volume, laid as
        // a trench and a berm instead of cell towers.
        self.soil.lay_debris_field();
        roll
    }

    fn plow_pose(
        &mut self,
        roll: f32,
        yaw: f32,
        embed: f32,
        shift: Vec3,
        throw_frac: f32,
    ) -> PoseCut {
        self.seat_skin(roll, yaw, embed, shift);
        let ori = self.vehicle.orientation();
        let yaw_dir = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let fwd = yaw_dir;
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let mut pose = PoseCut::default();
        let mut origin = Vec3::ZERO;
        let mut dir_acc = Vec3::ZERO;
        let mut fling = 0.0_f32;
        for i in 0..4 {
            let center = self.vehicle.pos + ori * self.vehicle.wheel_center_local(i);
            let bottom = tire_bottom(&self.vehicle, i);
            let h = self.soil.sample(bottom.x, bottom.z).height;
            let pen = h - bottom.y;
            if pen < 0.012 {
                continue;
            }
            let throw_dir = throw_dir(&self.vehicle, bottom);
            let cut = self.soil.plow_contact(
                bottom,
                fwd,
                right,
                0.46,
                WHEEL_WIDTH,
                bottom.y,
                throw_dir,
                throw_frac,
            );
            credit_bed(cut, false, &mut pose);
            if pen > 0.008 {
                self.soil.imprint_tread(
                    center,
                    fwd,
                    right,
                    WHEEL_RADIUS,
                    WHEEL_WIDTH,
                    pen.min(0.09),
                );
            }
            if cut.thrown > 1.0e-6 {
                origin += bottom * cut.thrown;
                dir_acc += throw_dir * cut.thrown;
                fling += cut.thrown;
            }
        }
        for p in body_skin() {
            let w = self.vehicle.pos + ori * p;
            if wheel_owns(&self.vehicle, w) {
                continue;
            }
            let h = self.soil.sample(w.x, w.z).height;
            if h - w.y < 0.012 {
                continue;
            }
            let throw_dir = throw_dir(&self.vehicle, w);
            let cut = self.soil.plow_contact(
                w,
                fwd,
                right,
                0.50,
                0.40,
                w.y,
                throw_dir,
                throw_frac,
            );
            credit_bed(cut, true, &mut pose);
            if cut.thrown > 1.0e-6 {
                origin += w * cut.thrown;
                dir_acc += throw_dir * cut.thrown;
                fling += cut.thrown;
            }
        }
        if fling > 1.0e-6 {
            let origin = origin / fling + Vec3::Y * 0.16;
            let dir = if dir_acc.length_squared() < 1.0e-8 {
                right
            } else {
                dir_acc.normalize()
            };
            self.fling_mass(origin, dir, fling);
        }
        pose
    }

    fn fling_mass(&mut self, origin: Vec3, dir: Vec3, volume: f32) {
        let piece = 0.0032_f32;
        let n = ((volume / piece).round() as i32).clamp(1, 3);
        let each = volume / n as f32;
        let mut left = volume;
        for k in 0..n {
            let side = (k as f32 - (n as f32 - 1.0) * 0.5) * 0.35;
            let spread = Vec3::new(-dir.z, 0.0, dir.x) * side;
            let vel = dir * 2.4 + spread + Vec3::Y * (2.1 + k as f32 * 0.25);
            let took = self.particles.throw_soil(origin + spread * 0.2, vel, each);
            left -= took;
        }
        if left > 1.0e-6 {
            let land = origin + dir * 0.85;
            let _ = self.soil.receive_thrown(land.x, land.z, left);
        }
    }

    fn seat_skin(&mut self, roll: f32, yaw: f32, embed: f32, shift: Vec3) {
        self.vehicle.roll = roll;
        self.vehicle.pitch = 0.0;
        self.vehicle.yaw = yaw;
        self.vehicle.roll_rate = 0.0;
        self.vehicle.yaw_rate = 0.0;
        self.vehicle.pitch_rate = 0.0;
        self.vehicle.vel = Vec3::ZERO;
        self.vehicle.pos.x = -0.50 * (1.0 - yaw.cos()) + shift.x;
        self.vehicle.pos.z = 0.90 * yaw.sin() + shift.z;
        self.vehicle.pos.y = 0.0;
        let ori = self.vehicle.orientation();
        let mut low_y = f32::MAX;
        let mut low_x = 0.0_f32;
        let mut low_z = 0.0_f32;
        for p in body_skin() {
            let w = self.vehicle.pos + ori * p;
            if w.y < low_y {
                low_y = w.y;
                low_x = w.x;
                low_z = w.z;
            }
        }
        for i in 0..4 {
            let w = tire_bottom(&self.vehicle, i);
            if w.y < low_y {
                low_y = w.y;
                low_x = w.x;
                low_z = w.z;
            }
        }
        let ground = match self.soil.idx_at(low_x, low_z) {
            Some(idx) => self.soil.initial_height[idx],
            None => 0.0,
        };
        self.vehicle.pos.y += (ground - embed) - low_y;
    }
}

/// Volumes from a roll. Cubic metres, except `bed`, which is metres of cut
/// left under the first pose after the body has settled in the second.
#[derive(Clone, Copy, Debug, Default)]
pub struct RollCut {
    pub displaced: f32,
    pub berm: f32,
    pub thrown: f32,
    pub body: f32,
    pub tires: f32,
    pub bed: f32,
    pub bed_after: f32,
    pub raised: bool,
    pub airborne: f32,
    pub placed: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct PoseCut {
    body: f32,
    tires: f32,
    berm: f32,
    thrown: f32,
}

fn body_skin() -> Vec<Vec3> {
    let mut pts = Vec::with_capacity(84);
    let mut z = -1.55_f32;
    while z <= 1.56 {
        for y in [0.0_f32, 0.30, 0.60, 0.92] {
            pts.push(Vec3::new(-0.90, y, z));
            pts.push(Vec3::new(0.90, y, z));
        }
        z += 0.42;
    }
    for z in [-1.15_f32, -0.35, 0.45, 1.25] {
        for x in [-0.42_f32, 0.0, 0.42] {
            pts.push(Vec3::new(x, -0.24, z));
        }
    }
    pts
}

fn tire_bottom(vehicle: &Vehicle, index: usize) -> Vec3 {
    let ori = vehicle.orientation();
    let center = vehicle.pos + ori * vehicle.wheel_center_local(index);
    let axle_y = (ori * Vec3::X).y.clamp(-1.0, 1.0);
    let radial = WHEEL_RADIUS * (1.0 - axle_y * axle_y).max(0.0).sqrt();
    let flank = WHEEL_WIDTH * 0.5 * axle_y.abs();
    Vec3::new(center.x, center.y - radial - flank, center.z)
}

fn wheel_owns(vehicle: &Vehicle, point: Vec3) -> bool {
    let ori = vehicle.orientation();
    for i in 0..4 {
        let center = vehicle.pos + ori * vehicle.wheel_center_local(i);
        if (center - point).length() < WHEEL_RADIUS * 0.92 {
            return true;
        }
    }
    false
}

/// Berm and thrown figures count only the bed that was cut, so they add up
/// to the displaced soil. Spoil that was picked up and put down again is not
/// a second cut.
fn credit_bed(cut: crate::soil::PlowCut, body: bool, pose: &mut PoseCut) {
    let vol = cut.bed + cut.moved;
    let share = if vol > 1.0e-8 { cut.bed / vol } else { 0.0 };
    if body {
        pose.body += cut.bed;
    } else {
        pose.tires += cut.bed;
    }
    pose.berm += cut.berm * share;
    pose.thrown += cut.thrown * share;
}

fn throw_dir(vehicle: &Vehicle, at: Vec3) -> Vec3 {
    let yaw = vehicle.yaw;
    let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let tangent = Vec3::new(fwd.z, 0.0, -fwd.x);
    let out = Vec3::new(at.x - vehicle.pos.x, 0.0, at.z - vehicle.pos.z);
    let out = if out.length_squared() < 1.0e-6 {
        tangent
    } else {
        out.normalize()
    };
    let dir = out * 0.55 + tangent * 0.85;
    if dir.length_squared() < 1.0e-8 {
        tangent
    } else {
        dir.normalize()
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
            let add = (0.16 + 0.18 * sample.wetness + 0.12 * sample.water.min(1.0))
                * stick_scale(sample.kind(), sample.wetness);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::{MASS, WHEEL_RADIUS};

    #[test]
    fn hubs_sit_on_the_solver_sink() {
        let world = World::flat(0.72);
        let mut support = 0.0_f32;
        for i in 0..4 {
            let w = world.vehicle.wheels[i];
            let center =
                world.vehicle.pos + world.vehicle.orientation() * world.vehicle.wheel_center_local(i);
            let h = world.soil.sample(center.x, center.z).height;
            let bottom = center.y - WHEEL_RADIUS;
            let seat = h - w.sinkage;
            assert!(w.normal > 200.0, "wheel {i} is not supporting the body");
            assert!(w.sinkage > 0.02 && w.sinkage < 0.09, "wheel {i} sink {}", w.sinkage);
            assert!(
                (bottom - seat).abs() < 0.002,
                "wheel {i} hub is not on soil-sink, bottom {bottom:.4} soil-sink {seat:.4} sink {:.4}",
                w.sinkage
            );
            assert!(
                (center.y - (seat + WHEEL_RADIUS)).abs() < 0.002,
                "wheel {i} hub {:.4} is off the seat {:.4}",
                center.y,
                seat + WHEEL_RADIUS
            );
            support += w.normal;
        }
        let weight = MASS * 9.81;
        assert!(
            (support - weight).abs() < weight * 0.04,
            "support {support:.0} N does not match weight {weight:.0} N"
        );
        let (n, amp) = world.soil.tread_imprint_stats();
        assert!(n > 0 && amp > 0.01, "settle did not leave a tread negative");
        let saved = world.soil.save_state();
        let loaded = crate::soil::SoilField::load_state(&saved).expect("reload");
        let (n2, amp2) = loaded.tread_imprint_stats();
        assert_eq!(n2, n);
        assert!((amp2 - amp).abs() < 1.0e-5, "the imprint did not survive reload");
    }

    #[test]
    fn niva_sinks_on_two_surfaces() {
        use crate::soil::SurfaceKind;

        let world = World::surface_split();
        let mut clay_sink = 0.0_f32;
        let mut clay_n = 0.0_f32;
        let mut gravel_sink = 0.0_f32;
        let mut gravel_n = 0.0_f32;
        for i in 0..4 {
            let w = world.vehicle.wheels[i];
            let center = world.vehicle.pos
                + world.vehicle.orientation() * world.vehicle.wheel_center_local(i);
            let sample = world.soil.sample(center.x, center.z);
            let bottom = center.y - WHEEL_RADIUS;
            let seat = sample.height - w.sinkage;
            assert!(w.normal > 200.0, "wheel {i} is not supporting the body");
            assert!(
                (bottom - seat).abs() < 0.002,
                "wheel {i} hub is not on soil-sink, bottom {bottom:.4} soil-sink {seat:.4}"
            );
            match sample.kind() {
                SurfaceKind::WetMud | SurfaceKind::FreshCut | SurfaceKind::PackedClay => {
                    clay_sink += w.sinkage;
                    clay_n += 1.0;
                }
                SurfaceKind::Gravel => {
                    gravel_sink += w.sinkage;
                    gravel_n += 1.0;
                }
                other => panic!("wheel {i} left its surface, kind {other:?} at {center:?}"),
            }
            eprintln!(
                "split wheel {i} kind {:?} sink {:.4} m x {:.2}",
                sample.kind(),
                w.sinkage,
                center.x
            );
        }
        assert_eq!(clay_n, 2.0, "the clay side lost a tire");
        assert_eq!(gravel_n, 2.0, "the gravel side lost a tire");
        let clay = clay_sink / clay_n;
        let gravel = gravel_sink / gravel_n;
        assert!(
            clay > gravel + 0.015,
            "the two surfaces sank the same, clay {clay:.4} gravel {gravel:.4}"
        );
        let v = world.soil.volumes();
        assert!(
            (v.cut - v.placed).abs() < v.cut * 0.02 + 1.0e-5,
            "split settle broke cut = placed, cut {} placed {}",
            v.cut,
            v.placed
        );
        let (n, amp) = world.soil.tread_imprint_stats();
        assert!(n > 0 && amp > 0.01, "the clay contact left no tread height");
    }

    #[test]
    fn roll_in_wet_clay_cuts_throws_and_keeps_the_bed() {
        use crate::soil::SurfaceKind;

        let mut world = World::wet_clay();
        assert!(
            world.soil.volumes().cut < 1.0e-6,
            "the pad was already cut"
        );
        let roll = world.roll_in_clay();
        eprintln!(
            "roll displaced {:.4} m3  thrown {:.4} m3  berm {:.4} m3  body {:.4} m3  tires {:.4} m3  \
             bed {:.4} m  bed_after {:.4} m  placed {:.4} m3  airborne {:.4} m3",
            roll.displaced,
            roll.thrown,
            roll.berm,
            roll.body,
            roll.tires,
            roll.bed,
            roll.bed_after,
            roll.placed,
            roll.airborne
        );
        assert!(
            roll.displaced > 0.02,
            "soil displaced {:.4} m3",
            roll.displaced
        );
        assert!(roll.thrown > 0.004, "mass thrown {:.4} m3", roll.thrown);
        assert!(roll.berm > 0.008, "berm {:.4} m3", roll.berm);
        assert!(
            roll.body > 0.004,
            "the body did not cut, {:.4} m3",
            roll.body
        );
        assert!(
            roll.tires > 0.004,
            "the tires did not cut, {:.4} m3",
            roll.tires
        );
        assert!(roll.bed > 0.06, "the roll did not cut a bed, {:.4} m", roll.bed);
        assert!(
            !roll.raised,
            "the second pose filled the bed"
        );
        assert!(
            roll.bed_after > 0.06,
            "bed after the body settled {:.4} m",
            roll.bed_after
        );
        let gap = (roll.displaced - roll.placed - roll.airborne).abs();
        assert!(
            gap < roll.displaced * 0.02 + 1.0e-4,
            "volume deleted displaced {:.4} placed {:.4} airborne {:.4}",
            roll.displaced,
            roll.placed,
            roll.airborne
        );
        assert_eq!(
            world.soil.sample(8.0, 8.0).kind(),
            SurfaceKind::WetMud,
            "the roll retagged clay that it never touched"
        );
        let (n, amp) = world.soil.tread_imprint_stats();
        assert!(n > 0 && amp > 0.01, "the tires left no tread imprint");
        assert!(
            world.vehicle.roll.abs() > 1.0,
            "the settled pose is not on its side, roll {}",
            world.vehicle.roll
        );
        let shaped = world.soil.volumes();
        assert!(
            (shaped.cut - roll.displaced).abs() < 1.0e-4,
            "debris reshape moved the cut {:.4} vs {:.4}",
            shaped.cut,
            roll.displaced
        );
        assert!(
            (shaped.placed - roll.placed).abs() < 1.0e-4,
            "debris reshape moved the spoil {:.4} vs {:.4}",
            shaped.placed,
            roll.placed
        );
        let max_spoil = world.soil.placed.iter().copied().fold(0.0_f32, f32::max);
        let max_depth = world
            .soil
            .initial_height
            .iter()
            .zip(world.soil.bed.iter())
            .map(|(h0, bed)| (h0 - bed).max(0.0))
            .fold(0.0_f32, f32::max);
        assert!(
            max_spoil > 0.14 && max_spoil < 0.28,
            "berm is not a ridge, height {:.3} m",
            max_spoil
        );
        assert!(
            max_depth > 0.10 && max_depth < 0.18,
            "cut is not a trench, depth {:.3} m",
            max_depth
        );
        let res = world.soil.res;
        let mut towers = 0;
        let mut deep = 0;
        let mut deep_holes = 0;
        for idx in 0..world.soil.placed.len() {
            let h = world.soil.placed[idx];
            let dug = (world.soil.initial_height[idx] - world.soil.bed[idx]).max(0.0);
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            let mut best_p = 0.0_f32;
            let mut best_d = 0.0_f32;
            for dj in -1isize..=1 {
                for di in -1isize..=1 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    let nidx = jj as usize * res + ii as usize;
                    best_p = best_p.max(world.soil.placed[nidx]);
                    best_d = best_d.max(
                        (world.soil.initial_height[nidx] - world.soil.bed[nidx]).max(0.0),
                    );
                }
            }
            if h > 0.04 && best_p < h * 0.55 {
                towers += 1;
            }
            if dug > 0.04 {
                deep += 1;
                if best_d < 0.02 {
                    deep_holes += 1;
                }
            }
        }
        assert_eq!(towers, 0, "spoil still stands in towers");
        assert!(deep > 12, "the cut disappeared, deep cells {deep}");
        assert_eq!(deep_holes, 0, "the cut is still a scatter of pits");
        let lumps = world
            .particles
            .items
            .iter()
            .filter(|p| p.alive && p.ledger && !p.deposited && p.volume > 1.0e-8)
            .count();
        assert_eq!(lumps, 0, "thrown soil is still a loose lump");
    }
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
