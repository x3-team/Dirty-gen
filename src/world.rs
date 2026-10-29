//! Couples the buggy, the heightfield and splash.

use glam::Vec3;

use crate::bike::{Bike, BIKE_RADIUS, BIKE_WIDTH};
use crate::particles::Particles;
use crate::patch::{
    contact_patch, groove_slurry_capacity, moisture_traction_row, release_groove_slurry,
    TreadCoat, TreadStep, MOISTURE_TRACTION_NORMAL, MOISTURE_TRACTION_SINK,
    MOISTURE_TRACTION_SLIP, SIDEWALL_VOLUME,
};
use crate::soil::{
    moisture_state_fields, on_launch_crest, slurry_motion_drag, slurry_settle_pace, stick_scale,
    surface_moisture, track_offset, wet_berm_creep_under_load_rate, SlurrySeal, SoilField,
    SurfaceKind, SurfaceMoisture, SLURRY_COMPARE_MOISTURE, TRACK_R, TRACK_ZN, TRACK_ZS,
};
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
    spawn_pos: Vec3,
    spawn_yaw: f32,
    bike_spawn: Vec3,
    bike_spawn_yaw: f32,
    /// Cubic metres of sidewall coat that landed back in a rut.
    pub coat_in_rut: f32,
    /// Bow and wake slurry moved on the last plow pass, cubic metres.
    slurry_plow_m3: (f32, f32),
    /// Bed lowering rate under a tire rolling in deep slurry (m/s scale).
    slurry_rut_pace: f32,
}

const SLURRY_POOL_X: (f32, f32) = (-2.35, 2.35);
const SLURRY_POOL_Z: (f32, f32) = (-2.75, 2.35);
const SLURRY_POOL_DEPTH: f32 = 0.16;

#[derive(Clone, Copy, Debug, Default)]
struct SlurryDriveOutcome {
    groove: f32,
    ribbons: f32,
    spray: f32,
    airborne: f32,
    bow_m3: f32,
    wake_m3: f32,
    rut_before: f32,
    rut_after: f32,
}

#[derive(Clone, Copy, Debug)]
struct SlurryPlowLedger {
    slurry: f32,
    groove: f32,
    bow_m3: f32,
    wake_m3: f32,
    ribbons: f32,
    spray: f32,
}

struct SlurryDragTriple {
    clay: crate::soil::SlurryDrag,
    silt: crate::soil::SlurryDrag,
    sand: crate::soil::SlurryDrag,
}

fn slurry_drag_triple(moisture: f32, depth: f32) -> SlurryDragTriple {
    SlurryDragTriple {
        clay: slurry_motion_drag(
            SurfaceKind::WetMud,
            moisture,
            depth,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            1.05,
            3.4,
        ),
        silt: slurry_motion_drag(
            SurfaceKind::Silt,
            moisture,
            depth,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            1.05,
            3.4,
        ),
        sand: slurry_motion_drag(
            SurfaceKind::Sand,
            moisture,
            depth,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            1.05,
            3.4,
        ),
    }
}

fn slurry_plow_ledger(pool: f32, pass: SlurryDriveOutcome) -> SlurryPlowLedger {
    let slurry = (pool - pass.bow_m3 - pass.wake_m3 - pass.groove - pass.ribbons - pass.spray)
        .max(0.0);
    SlurryPlowLedger {
        slurry,
        groove: pass.groove,
        bow_m3: pass.bow_m3,
        wake_m3: pass.wake_m3,
        ribbons: pass.ribbons,
        spray: pass.spray,
    }
}

/// Wheels on one side of the axle, averaged into a single groove.
#[derive(Clone, Copy)]
struct LaneAccum {
    center: Vec3,
    forward: Vec3,
    right: Vec3,
    sink: f32,
    slip: f32,
    normal: f32,
    push: f32,
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
            normal: 0.0,
            push: 0.0,
            n: 0,
        }
    }
}

impl LaneAccum {
    fn add(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        sink: f32,
        slip: f32,
        normal: f32,
        push: f32,
    ) {
        let load = (normal * normal + push * push).sqrt();
        let prev = (self.normal * self.normal + self.push * self.push).sqrt();
        if self.n == 0 || load >= prev {
            self.forward = forward;
            self.right = right;
            self.sink = sink;
            self.slip = slip.abs();
            self.normal = normal.max(0.0);
            self.push = push.max(0.0);
        }
        self.center += center;
        self.n += 1;
    }

    fn stamp(&self, soil: &mut SoilField, radius: f32, width: f32, dt: f32) {
        if self.n == 0 || (self.sink < 0.004 && self.normal < 1.0 && self.push < 1.0) {
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
            self.normal,
            self.push,
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
            slurry_plow_m3: (0.0, 0.0),
            slurry_rut_pace: 0.14,
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
        world.slurry_plow_m3 = (0.0, 0.0);
        world.slurry_rut_pace = 0.14;
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
            slurry_plow_m3: (0.0, 0.0),
            slurry_rut_pace: 0.14,
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
        world.slurry_plow_m3 = (0.0, 0.0);
        world.slurry_rut_pace = 0.14;
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
                        0.0,
                        0.0,
                    );
                }
            }
            let planted = contacts.iter().filter(|c| c.active).count();
            self.bike
                .note_flight(&mut self.soil, &mut self.particles, planted, h);
            self.bike.integrate(&contacts, h);
        }
    }

    /// A tire that has sat in deep slurry picks up a seal. Rising through it
    /// holds the suction already applied on the contact, then the break throws
    /// ribbons and drops taken from the slurry column. Slurry that stays in
    /// the tread is packed on that same break. Rolling through the rut does
    /// not fill the voids.
    fn pull_slurry_seals(&mut self, contacts: &[Contact; 4], dt: f32) {
        for c in contacts.iter() {
            let i = c.index;
            let depth = c.sample.slurry;
            let v_up = c.v_up;
            let lift = self.vehicle.wheels[i].slurry_lift;
            let sealed = self.vehicle.wheels[i].slurry_sealed;
            if depth > 0.045 && v_up < 0.10 {
                let w = &mut self.vehicle.wheels[i];
                w.slurry_lift = (lift - dt * 0.15).max(0.0);
                if w.slurry_lift < 0.008 {
                    w.slurry_sealed = true;
                }
                continue;
            }
            if !sealed || v_up < 0.22 || depth < 0.03 {
                continue;
            }
            let mut seal = SlurrySeal {
                lift,
                impulse: 0.0,
                broken: false,
            };
            let out = self.soil.pull_from_slurry(
                c.wheel_center.x,
                c.wheel_center.z,
                c.forward,
                WHEEL_RADIUS,
                WHEEL_WIDTH,
                v_up,
                dt,
                &mut seal,
            );
            self.vehicle.wheels[i].slurry_lift = seal.lift;
            if seal.broken {
                self.vehicle.wheels[i].slurry_sealed = false;
            }
            if out.thrown > 1.0e-8 {
                let origin = c.contact_point + Vec3::Y * 0.08;
                let _took = self
                    .particles
                    .throw_slurry(origin, c.forward, c.right, out.thrown);
            }
            if seal.broken {
                self.pack_slurry_groove(c);
            }
        }
    }

    /// The pull-out left slurry in the voids, up to what one tread can hold.
    fn pack_slurry_groove(&mut self, c: &Contact) {
        let held = self.vehicle.wheels[c.index].slurry_groove;
        let room = (groove_slurry_capacity(WHEEL_RADIUS, WHEEL_WIDTH) - held).max(0.0);
        if room <= 1.0e-8 {
            return;
        }
        let packed = self.soil.pack_slurry_into_grooves(
            c.wheel_center.x,
            c.wheel_center.z,
            c.forward,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            room,
        );
        self.vehicle.wheels[c.index].slurry_groove += packed;
    }

    /// Spin, or a coat that is shedding, pulls the held slurry out as ribbons.
    fn shed_slurry_grooves(&mut self, contacts: &[Contact; 4], tread: &[TreadStep; 4], dt: f32) {
        for c in contacts.iter() {
            let i = c.index;
            let spin = ((c.slip.abs() - 0.28) / 0.55).clamp(0.0, 1.0);
            let shed = tread[i].shed;
            let left = release_groove_slurry(
                &mut self.vehicle.wheels[i].slurry_groove,
                spin,
                shed,
                dt,
            );
            if left <= 1.0e-8 {
                continue;
            }
            self.soil.airborne += left;
            // In deep slurry the tread throw at the end of the plow supplies
            // ribbons and spray. Mid-pass groove sheets behind the wheels read
            // as orange tablets in the side still.
            if c.sample.slurry > 0.01 {
                continue;
            }
            let _carried = self.particles.throw_groove_ribbons(
                c.wheel_center,
                c.forward,
                c.right,
                left,
            );
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
                c.normal,
                c.push,
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
            sides[c.index % 2].add(
                c.wheel_center,
                c.forward,
                c.right,
                c.sinkage,
                slip_stamp,
                c.normal,
                c.push,
            );
            if c.sample.water > 0.008 {
                let speed = c.speed.max(0.2);
                self.soil.displace_water(
                    c.wheel_center.x,
                    c.wheel_center.z,
                    c.forward,
                    c.sample.water.min(0.05) * 0.25 * (0.4 + speed * 0.15),
                );
            }
            // The bow is slurry pushed ahead of the tread. The wake is the
            // thinned column behind it. Neither is spray, and neither pulls
            // slurry up out of the rut.
            if c.sample.slurry > 0.01 && c.speed > 0.25 {
                let pass = self.soil.raise_bow_wave(
                    c.wheel_center.x,
                    c.wheel_center.z,
                    c.forward,
                    WHEEL_RADIUS,
                    WHEEL_WIDTH,
                );
                self.slurry_plow_m3.0 += pass.bow_m3;
                self.slurry_plow_m3.1 += pass.wake_m3;
                self.soil.deepen_slurry_rut(
                    c.wheel_center.x,
                    c.wheel_center.z,
                    c.forward,
                    WHEEL_RADIUS,
                    WHEEL_WIDTH,
                    self.slurry_rut_pace,
                    dt,
                );
            }
            if c.sample.slurry > 0.028 && c.speed > 0.22 {
                self.pack_slurry_groove(c);
            }
            if c.sample.slurry > 0.028 && c.speed > 0.28 && c.slip.abs() > 0.10 {
                let want = (c.slip.abs() * c.speed * dt * 0.012).max(4.0e-5);
                let scraped = self.soil.scrape_slurry_from_tread(
                    c.wheel_center.x,
                    c.wheel_center.z,
                    c.forward,
                    WHEEL_RADIUS,
                    WHEEL_WIDTH,
                    want,
                );
                if scraped > 1.0e-8 {
                    self.soil.airborne += scraped;
                    let origin = c.contact_point + Vec3::Y * 0.14;
                    let _ = self.particles.throw_slurry(
                        origin,
                        c.forward,
                        c.right,
                        scraped,
                    );
                }
            }
        }
        self.pull_slurry_seals(&contacts, dt);
        let width = crate::vehicle::WHEEL_WIDTH;
        for side in &sides {
            side.stamp(&mut self.soil, WHEEL_RADIUS, width, dt);
        }

        let planted = contacts.iter().filter(|c| c.active).count();
        self.note_flight(planted, dt);
        self.vehicle.integrate(&contacts, &self.soil, dt);
        let tread = self.vehicle.step_adhesion(&contacts, dt);
        self.shed_slurry_grooves(&contacts, &tread, dt);
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
            // Deep slurry debris is ribbons and fine spray only. Solid droplets
            // and flung clods read as orange tablets behind the tread.
            if c.sample.slurry > 0.01 {
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
            slurry_plow_m3: (0.0, 0.0),
            slurry_rut_pace: 0.14,
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

    /// Roll the Niva onto its side in a slurry lane and shove that slurry
    /// sideways. The cut equals the berm, the throw, and the slurry left in
    /// the rut. The throw is sheets from the roll and ribbons from the spin.
    /// Clay holds, so the shove does not empty the rut.
    pub fn roll_and_slide_in_slurry(&mut self) -> SlurryManeuver {
        let before_cut = self.soil.cut_volume();
        let before_placed = self.soil.placed_volume();
        let before_air = self.soil.airborne;
        let from = Vec3::new(0.55, 0.0, -1.85);
        let to = Vec3::new(0.55, 0.0, 1.85);
        let dug = self
            .soil
            .cut_lane_in_slurry(from, to, 0.52, 0.22, 0.32);
        let _slide = self.soil.slide_slurry(-0.15, 1.30, -2.05, 2.05, Vec3::X);
        let ribbons = self.soil.peel_slurry(-0.10, 1.25, -2.00, 2.00, 0.18);
        self.seat_skin(1.40, 0.15, 0.02, Vec3::new(-1.15, 0.0, 0.05));
        let yaw = self.vehicle.yaw;
        let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        // In the air between the Niva and the trench, not on the roof.
        // The long axis runs across the camera so each piece reads as a sheet.
        let shed = Vec3::new(0.35, 0.0, 0.05);
        let across = Vec3::X;
        let lane = Vec3::Z;
        let sheets = dug.thrown;
        let carried_sheets = self
            .particles
            .throw_roll_sheets(shed, lane, across, sheets);
        let carried_ribbons = self.particles.throw_spin_ribbons(
            shed + Vec3::new(0.05, 0.0, 0.15),
            lane,
            across,
            ribbons,
        );
        let _ = (fwd, right);
        let mut bed = 0.0_f32;
        let mut berm_height = 0.0_f32;
        for idx in 0..self.soil.bed.len() {
            bed = bed.max((self.soil.initial_height[idx] - self.soil.bed[idx]).max(0.0));
            berm_height = berm_height.max(self.soil.placed[idx]);
        }
        let slurry = self.soil.slurry_volume();
        let placed = self.soil.placed_volume() - before_placed;
        let airborne = self.soil.airborne - before_air;
        let cut = self.soil.cut_volume() - before_cut;
        let _ = (carried_sheets, carried_ribbons);
        SlurryManeuver {
            cut,
            berm: dug.berm,
            thrown: sheets + ribbons,
            slurry,
            placed,
            airborne,
            sheets,
            ribbons,
            bed,
            berm_height,
        }
    }

    /// Drive the Niva through a deep slurry pool. A bow piles ahead of each
    /// tread, a wake thins behind, and the rut deepens. Ribbons and fine
    /// spray are the only debris kinds thrown from the tread.
    pub fn plow_through_deep_slurry(&mut self, drive_seconds: f32) -> SlurryPlowWake {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let depth = SLURRY_POOL_DEPTH;
        let pool_x = SLURRY_POOL_X;
        let pool_z = SLURRY_POOL_Z;
        let pool = self.soil.fill_slurry_pool(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        let slurry0 = self.soil.slurry_volume();
        let mix0 = self.soil.mix_volumes();
        let air0 = self.soil.airborne;
        let drag = slurry_drag_triple(moisture, depth);
        self.slurry_rut_pace = 0.14;
        self.reset_niva_at_slurry_entry(0.72);
        let pass = self.slurry_plow_drive_pass(drive_seconds, pool_x, pool_z, air0);
        let slurry = slurry_plow_ledger(pool, pass);
        let mix1 = self.soil.mix_volumes();
        self.polish_slurry_plow_scene(pool_x, pool_z);
        SlurryPlowWake {
            pool,
            slurry0,
            slurry: slurry.slurry,
            groove: slurry.groove,
            bow_m3: slurry.bow_m3,
            wake_m3: slurry.wake_m3,
            rut_before: pass.rut_before,
            rut_after: pass.rut_after,
            ribbons: slurry.ribbons,
            spray: slurry.spray,
            airborne: pass.airborne,
            mix0,
            mix1,
            drag_clay: drag.clay.force.abs(),
            drag_silt: drag.silt.force.abs(),
            drag_sand: drag.sand.force.abs(),
            torque_clay: drag.clay.torque.abs(),
            torque_silt: drag.silt.torque.abs(),
            torque_sand: drag.sand.torque.abs(),
        }
    }

    /// Two passes through the same pool. The second visit deepens the rut,
    /// raises the berm, and redistributes slurry across berm, pool, grooves,
    /// bow, wake, and tread debris. Ribbons and fine spray only.
    pub fn second_pass_through_deep_slurry(&mut self, drive_seconds: f32) -> SlurrySecondPassWake {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let depth = SLURRY_POOL_DEPTH;
        let pool_x = SLURRY_POOL_X;
        let pool_z = SLURRY_POOL_Z;
        let pool = self.soil.fill_slurry_pool(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        let slurry0 = self.soil.slurry_volume();
        let mix0 = self.soil.mix_volumes();
        let air0 = self.soil.airborne;
        let rut0 = self
            .soil
            .max_bed_cut_in(pool_x.0, pool_x.1, pool_z.0, pool_z.1);

        self.slurry_rut_pace = 0.14;
        self.reset_niva_at_slurry_entry(0.72);
        let pass1 = self.slurry_plow_drive_pass(drive_seconds, pool_x, pool_z, air0);
        let ori1 = self.vehicle.orientation();
        let rear_l1 = self.vehicle.pos + ori1 * self.vehicle.wheel_center_local(2);
        let rear_r1 = self.vehicle.pos + ori1 * self.vehicle.wheel_center_local(3);
        let half_track = WHEEL_WIDTH * 0.5 + self.soil.cell * 0.55;
        let rut_pass1 = self.soil.lane_rut_depth_behind(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            rear_l1.z,
            rear_l1.x,
            rear_r1.x,
            half_track,
        );

        self.reset_niva_at_slurry_entry(0.68);
        self.soil.age_passes(0.45);
        self.slurry_rut_pace = 0.22;
        let air_mid = self.soil.airborne;
        let pass2 = self.slurry_plow_drive_pass(drive_seconds, pool_x, pool_z, air_mid);

        let bow_m3 = pass1.bow_m3 + pass2.bow_m3;
        let wake_m3 = pass1.wake_m3 + pass2.wake_m3;
        let groove = pass2.groove;
        let mut ribbons = 0.0_f32;
        let mut spray = 0.0_f32;
        for p in &self.particles.items {
            if p.volume <= 1.0e-8 || !p.ledger || !p.alive {
                continue;
            }
            match p.kind {
                crate::particles::KIND_SLURRY_RIBBON => ribbons += p.volume,
                crate::particles::KIND_SLURRY_DROP => spray += p.volume,
                _ => {}
            }
        }
        let airborne = self.soil.airborne - air0;

        let berm = self.slurry_berm_volume(pool_x, pool_z);
        let slurry = (pool - berm - bow_m3 - wake_m3 - groove - ribbons - spray).max(0.0);
        let sum = berm + slurry + groove + bow_m3 + wake_m3 + ribbons + spray;

        let drag_pass1 = slurry_drag_triple(moisture, depth);

        self.polish_slurry_second_pass_scene(pool_x, pool_z, rut_pass1);
        let ori2 = self.vehicle.orientation();
        let rear_l2 = self.vehicle.pos + ori2 * self.vehicle.wheel_center_local(2);
        let rear_r2 = self.vehicle.pos + ori2 * self.vehicle.wheel_center_local(3);
        let rut_pass2 = self.soil.lane_rut_depth_behind(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            rear_l2.z,
            rear_l2.x,
            rear_r2.x,
            half_track,
        );
        let drag_depth = (depth + (rut_pass2 - rut0).max(0.0) * 0.85).min(0.28);
        let drag = slurry_drag_triple(moisture, drag_depth);
        let mix1 = self.soil.mix_volumes();

        SlurrySecondPassWake {
            pool,
            slurry0,
            berm,
            slurry,
            groove,
            bow_m3,
            wake_m3,
            sum,
            rut0,
            rut_pass1,
            rut_pass2,
            ribbons,
            spray,
            airborne,
            mix0,
            mix1,
            drag_clay: drag.clay.force.abs(),
            drag_silt: drag.silt.force.abs(),
            drag_sand: drag.sand.force.abs(),
            drag_clay_pass1: drag_pass1.clay.force.abs(),
            drag_silt_pass1: drag_pass1.silt.force.abs(),
            drag_sand_pass1: drag_pass1.sand.force.abs(),
            torque_clay: drag.clay.torque.abs(),
            torque_silt: drag.silt.torque.abs(),
            torque_sand: drag.sand.torque.abs(),
        }
    }

    /// Clay, silt, and sand traction across dry, damp, saturated, and pooled
    /// moisture. One clay pass wets a dry band; cut and placed volumes stay closed.
    pub fn moisture_traction_compare(&mut self) -> MoistureTractionWake {
        self.soil = SoilField::moisture_traction_pad();
        self.bike.pos = Vec3::new(14.0, 0.5, 14.0);
        self.bike.vel = Vec3::ZERO;
        self.vehicle.pos = Vec3::new(-1.35, 0.72, -0.35);
        self.vehicle.yaw = 0.0;
        self.vehicle.roll = 0.0;
        self.vehicle.pitch = 0.0;
        self.vehicle.vel = Vec3::ZERO;
        let h = self
            .soil
            .sample(self.vehicle.pos.x, self.vehicle.pos.z)
            .height;
        self.vehicle.pos.y = h + 0.62;

        let clay = traction_grid(SurfaceKind::WetMud);
        let silt = traction_grid(SurfaceKind::Silt);
        let sand = traction_grid(SurfaceKind::Sand);

        let (dry_w, dry_loose, dry_water) =
            moisture_state_fields(SurfaceKind::WetMud, SurfaceMoisture::Dry);
        let before = contact_patch(
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            MOISTURE_TRACTION_SINK,
            MOISTURE_TRACTION_SLIP,
            MOISTURE_TRACTION_NORMAL,
            SurfaceKind::WetMud,
            dry_w,
            dry_loose,
            dry_water,
            TreadCoat::clean(),
        );
        let dry_state = surface_moisture(dry_w, dry_water, SurfaceKind::WetMud);
        let cut0 = self.soil.cut_volume();
        let placed0 = self.soil.placed_volume();
        let mut z = -1.85_f32;
        while z <= -1.05 {
            self.soil.stamp_wheel(
                Vec3::new(-1.35, 0.30, z),
                WHEEL_RADIUS,
                WHEEL_WIDTH,
                Vec3::Z,
                Vec3::X,
                0.07,
                0.32,
                0.10,
                true,
                0.0,
                0.0,
            );
            z += 0.11;
        }
        let res = self.soil.res;
        let cell = self.soil.cell;
        let half = self.soil.size * 0.5;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let zc = (j as f32 + 0.5) * cell - half;
                if (x + 1.35).abs() > 0.38 || zc > -1.0 || zc < -2.0 {
                    continue;
                }
                let idx = j * res + i;
                let dug = (self.soil.initial_height[idx] - self.soil.bed[idx]).max(0.0);
                if dug > 0.012 {
                    self.soil.water[idx] = self.soil.water[idx].max(0.055);
                    self.soil.wetness[idx] = self.soil.wetness[idx].max(0.93);
                }
            }
        }
        let s = self.soil.sample(-1.35, -1.55);
        let after = contact_patch(
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            MOISTURE_TRACTION_SINK,
            MOISTURE_TRACTION_SLIP,
            MOISTURE_TRACTION_NORMAL,
            s.kind(),
            s.wetness,
            s.loose,
            s.water,
            TreadCoat::clean(),
        );
        let after_state = surface_moisture(s.wetness, s.water, s.kind());
        let cut = self.soil.cut_volume() - cut0;
        let placed = self.soil.placed_volume() - placed0;

        MoistureTractionWake {
            clay,
            silt,
            sand,
            clay_mu_before: before.mu,
            clay_mu_after: after.mu,
            clay_grip_before: before.grip,
            clay_grip_after: after.grip,
            clay_slip_before: crate::soil::moisture_peak_slip(SurfaceKind::WetMud, dry_state),
            clay_slip_after: crate::soil::moisture_peak_slip(SurfaceKind::WetMud, after_state),
            clay_wet_before: dry_w,
            clay_wet_after: s.wetness,
            cut,
            placed,
        }
    }

    /// After a second pass through the pool, let the bow wave and berm creep
    /// back into the rut. Clay settles slower than silt and sand.
    /// One pass through the pool, then hold the Niva on the wet berm while
    /// loaded tires slump spoil into the rut and the pool. Clay creeps slower
    /// than silt and sand at the reference load.
    pub fn wet_berm_creep_under_load(
        &mut self,
        pass_seconds: f32,
        creep_seconds: f32,
    ) -> WetBermCreepWake {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let pool_x = SLURRY_POOL_X;
        let pool_z = SLURRY_POOL_Z;
        let depth = SLURRY_POOL_DEPTH;
        let pool = self.soil.fill_slurry_pool(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        let mix0 = self.soil.mix_volumes();
        let air0 = self.soil.airborne;
        self.slurry_rut_pace = 0.14;
        self.reset_niva_at_slurry_entry(0.72);
        let pass = self.slurry_plow_drive_pass(pass_seconds, pool_x, pool_z, air0);
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.55);

        let fwd = self.vehicle.forward();
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let (berm_x, berm_z) = self.soil.berm_centroid_in(
            pool_x.0 - 0.35,
            pool_x.1 + 0.85,
            pool_z.0 - 0.35,
            pool_z.1 + 0.55,
        );
        self.vehicle.pos = Vec3::new(berm_x - right.x * 0.42, self.vehicle.pos.y, berm_z);
        self.vehicle.yaw = 0.04;
        self.seat_in_slurry(0.12);

        let (berm_before, rut_before, pool_slurry_before, sum_before) =
            self.wet_berm_creep_ledger(pool_x, pool_z);
        let dt = 1.0 / 60.0;
        let steps = (creep_seconds / dt).round().max(1.0) as u32;
        for _ in 0..steps {
            self.press_niva_on_wet_berm(dt);
        }

        let (berm_after, rut_after, pool_slurry_after, sum_after) =
            self.wet_berm_creep_ledger(pool_x, pool_z);
        let mix1 = self.soil.mix_volumes();
        WetBermCreepWake {
            pool,
            berm_before,
            berm_after,
            rut_before,
            rut_after,
            pool_slurry_before,
            pool_slurry_after,
            sum_before,
            sum_after,
            mix0,
            mix1,
            pace_clay: wet_berm_creep_under_load_rate(SurfaceKind::WetMud, moisture, 5200.0),
            pace_silt: wet_berm_creep_under_load_rate(SurfaceKind::Silt, moisture, 5200.0),
            pace_sand: wet_berm_creep_under_load_rate(SurfaceKind::Sand, moisture, 5200.0),
        }
    }

    fn slurry_rut_lane_volume(&self, _pool_x: (f32, f32), pool_z: (f32, f32)) -> f32 {
        let lane_x = 0.48_f32;
        self.soil.slurry_volume_in(
            -lane_x,
            lane_x,
            pool_z.0 + 0.18,
            pool_z.1 - 0.12,
        )
    }

    /// Berm placed spoil, slurry in the driven lane, and slurry in the open
    /// pool. The three bands do not overlap; their sum is the conserved total.
    fn wet_berm_creep_ledger(&self, pool_x: (f32, f32), pool_z: (f32, f32)) -> (f32, f32, f32, f32) {
        let berm = self.slurry_berm_volume(pool_x, pool_z);
        let rut = self.slurry_rut_lane_volume(pool_x, pool_z);
        let total_slurry = self
            .soil
            .slurry_volume_in(pool_x.0, pool_x.1, pool_z.0, pool_z.1);
        let pool_slurry = (total_slurry - rut).max(0.0);
        let sum = berm + rut + pool_slurry;
        (berm, rut, pool_slurry, sum)
    }

    fn press_niva_on_wet_berm(&mut self, dt: f32) {
        let pool_x = SLURRY_POOL_X;
        let pool_z = SLURRY_POOL_Z;
        let ori = self.vehicle.orientation();
        let fwd = self.vehicle.forward();
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let load_n = 5200.0;
        let push = 980.0;
        for i in 0..4 {
            let center = self.vehicle.pos + ori * self.vehicle.wheel_center_local(i);
            let ground = self.soil.sample(center.x, center.z).height;
            let sink = (ground - (center.y - WHEEL_RADIUS)).max(0.012);
            let hub_y = ground + WHEEL_RADIUS - sink;
            self.soil.creep_wet_berm_under_patch(
                Vec3::new(center.x, hub_y, center.z),
                fwd,
                right,
                WHEEL_RADIUS,
                WHEEL_WIDTH,
                true,
                load_n + push,
                dt,
            );
        }
        self.soil.slump_wet_berms_in(
            dt,
            load_n,
            pool_x.0 - 0.35,
            pool_x.1 + 0.85,
            pool_z.0 - 0.35,
            pool_z.1 + 0.55,
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            0.48,
            pool_z.0 + 0.18,
            pool_z.1 - 0.12,
        );
    }

    pub fn settle_slurry_after_second_pass(
        &mut self,
        drive_seconds: f32,
        settle_seconds: f32,
    ) -> SlurrySettleWake {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let pool_x = SLURRY_POOL_X;
        let pool_z = SLURRY_POOL_Z;
        let pass = self.second_pass_through_deep_slurry(drive_seconds);
        let (bow_before, rut_before) = self.slurry_bow_rut_volumes(pool_x, pool_z);
        let berm_before = pass.berm;
        let mix0 = self.soil.mix_volumes();

        self.soil.advance_slurry_settle(settle_seconds);

        let (bow_after, rut_after) = self.slurry_bow_rut_volumes(pool_x, pool_z);
        let berm_after = self.slurry_berm_volume(pool_x, pool_z);
        let slurry = (pass.sum - berm_after - bow_after - pass.groove - pass.wake_m3
            - pass.ribbons
            - pass.spray)
            .max(0.0);
        let mix1 = self.soil.mix_volumes();
        SlurrySettleWake {
            pool: pass.pool,
            berm_before,
            berm_after,
            bow_before,
            bow_after,
            rut_before,
            rut_after,
            slurry,
            groove: pass.groove,
            wake_m3: pass.wake_m3,
            ribbons: pass.ribbons,
            spray: pass.spray,
            sum: pass.sum,
            mix0,
            mix1,
            pace_clay: slurry_settle_pace(SurfaceKind::WetMud, moisture),
            pace_silt: slurry_settle_pace(SurfaceKind::Silt, moisture),
            pace_sand: slurry_settle_pace(SurfaceKind::Sand, moisture),
        }
    }

    fn slurry_bow_rut_volumes(&self, pool_x: (f32, f32), pool_z: (f32, f32)) -> (f32, f32) {
        let fwd = self.vehicle.forward();
        let ori = self.vehicle.orientation();
        let front = self.vehicle.pos + ori * self.vehicle.wheel_center_local(0);
        let rear_l = self.vehicle.pos + ori * self.vehicle.wheel_center_local(2);
        let rear_r = self.vehicle.pos + ori * self.vehicle.wheel_center_local(3);
        let half_track = WHEEL_WIDTH * 0.5 + self.soil.cell * 0.55;
        self.soil.slurry_bow_and_rut_volumes(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        )
    }

    /// Lean and roll the Niva in a deep slurry pool. Body and tire contact
    /// split volume across the berm, a bow wave, and a throw of sheets and
    /// ribbons. The pool total is conserved. Clay holds harder than silt and
    /// sand in the reference drag and torque at pool depth.
    pub fn rollover_in_deep_slurry(&mut self) -> SlurryRolloverWake {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let depth = SLURRY_POOL_DEPTH;
        let pool_x = SLURRY_POOL_X;
        let pool_z = SLURRY_POOL_Z;
        let pool = self.soil.fill_slurry_pool(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        let slurry0 = self.soil.slurry_volume();
        let mix0 = self.soil.mix_volumes();
        let air0 = self.soil.airborne;

        self.vehicle.pos = Vec3::new(0.0, 0.55, -0.42);
        self.vehicle.yaw = 0.0;
        self.vehicle.pitch = 0.0;
        self.vehicle.roll = 0.0;
        self.vehicle.vel = Vec3::ZERO;
        for w in &mut self.vehicle.wheels {
            w.compression = 0.22;
            w.omega = 0.0;
            w.slurry_sealed = false;
            w.slurry_lift = 0.0;
            w.slurry_groove = 0.0;
        }
        self.seat_in_slurry(0.14);

        let drag = slurry_drag_triple(moisture, depth);
        let mut bow_m3 = 0.0_f32;
        let steps = 30_usize;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let s = t * t * (3.0 - 2.0 * t);
            let moving = (s * (1.0 - s) * 4.0).clamp(0.0, 1.0);
            let roll = s * 1.38;
            let yaw = s * 0.92;
            let embed = 0.06 + 0.11 * s;
            let frac = 0.14 + 0.40 * moving;
            let shift = Vec3::new(0.0, 0.0, s * 0.55);
            let _ = self.plow_slurry_pose(roll, yaw, embed, shift, frac);
            let ori = self.vehicle.orientation();
            let fwd = self.vehicle.forward();
            for wi in 0..4 {
                let hub = self.vehicle.pos + ori * self.vehicle.wheel_center_local(wi);
                let pass = self.soil.raise_bow_wave(
                    hub.x,
                    hub.z,
                    fwd,
                    WHEEL_RADIUS,
                    WHEEL_WIDTH,
                );
                bow_m3 += pass.bow_m3;
            }
        }
        let yaw2 = 1.08_f32;
        let fwd2 = Vec3::new(yaw2.sin(), 0.0, yaw2.cos());
        let right2 = Vec3::new(fwd2.z, 0.0, -fwd2.x);
        let shift2 = right2 * 0.42 + fwd2 * 0.18 + Vec3::new(0.0, 0.0, 0.62);
        let _ = self.plow_slurry_pose(1.32, yaw2, 0.05, shift2, 0.26);
        let peel = self.soil.peel_slurry(
            pool_x.0 + 0.35,
            pool_x.1 - 0.35,
            pool_z.0 + 0.35,
            pool_z.1 - 0.35,
            0.16,
        );
        if peel > 1.0e-6 {
            let ori = self.vehicle.orientation();
            let shed = self.vehicle.pos + ori * Vec3::new(0.35, 0.42, 0.05);
            let lane = self.vehicle.forward();
            let across = Vec3::new(lane.z, 0.0, -lane.x);
            let sheets = peel * 0.55;
            let ribbons = peel - sheets;
            let _ = self
                .particles
                .throw_roll_sheets(shed + across * 0.18, lane, across, sheets);
            let _ = self.particles.throw_spin_ribbons(
                shed - lane * 0.12,
                lane,
                across,
                ribbons,
            );
            self.soil.airborne += peel;
        }
        self.seat_in_slurry(0.035);

        let dent = self
            .soil
            .max_bed_cut_in(pool_x.0, pool_x.1, pool_z.0, pool_z.1);

        self.polish_slurry_rollover_scene(pool_x, pool_z);

        let berm = self.slurry_berm_volume(pool_x, pool_z);
        let mut ribbons = 0.0_f32;
        let mut sheets = 0.0_f32;
        for p in &self.particles.items {
            if p.volume <= 1.0e-8 || !p.ledger || !p.alive {
                continue;
            }
            match p.kind {
                crate::particles::KIND_SLURRY_RIBBON => ribbons += p.volume,
                crate::particles::KIND_SLURRY_SHEET => sheets += p.volume,
                _ => {}
            }
        }
        let shed = ribbons + sheets;
        let slurry = (pool - berm - bow_m3 - shed).max(0.0);
        let sum = berm + slurry + bow_m3 + shed;
        let mix1 = self.soil.mix_volumes();
        let airborne = self.soil.airborne - air0;

        SlurryRolloverWake {
            pool,
            slurry0,
            berm,
            slurry,
            bow_m3,
            ribbons,
            sheets,
            shed,
            sum,
            dent,
            airborne,
            mix0,
            mix1,
            drag_clay: drag.clay.force.abs(),
            drag_silt: drag.silt.force.abs(),
            drag_sand: drag.sand.force.abs(),
            torque_clay: drag.clay.torque.abs(),
            torque_silt: drag.silt.torque.abs(),
            torque_sand: drag.sand.torque.abs(),
        }
    }

    /// The Niva breaks suction in a deep pool and drives out. Slurry that
    /// leaves the rut is the tear-out. The tread voids take a share like a
    /// pull-out, then shed ribbons on the exit spin. Ribbons, sheets, and
    /// fine spray are the only airborne debris.
    pub fn tear_out_from_deep_slurry(&mut self) -> SlurryTearOut {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let depth = 0.16_f32;
        let pool = self.soil.fill_slurry_pool(
            -2.35,
            2.35,
            -2.75,
            2.35,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        let slurry0 = self.soil.slurry_volume();
        let air0 = self.soil.airborne;
        let mix0 = self.soil.mix_volumes();

        let drag_clay = slurry_motion_drag(
            SurfaceKind::WetMud,
            moisture,
            depth,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            1.05,
            3.4,
        );
        let drag_silt = slurry_motion_drag(
            SurfaceKind::Silt,
            moisture,
            depth,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            1.05,
            3.4,
        );
        let drag_sand = slurry_motion_drag(
            SurfaceKind::Sand,
            moisture,
            depth,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            1.05,
            3.4,
        );

        self.vehicle.pos = Vec3::new(0.0, 0.55, -1.35);
        self.vehicle.yaw = 0.0;
        self.vehicle.pitch = 0.0;
        self.vehicle.roll = 0.0;
        self.vehicle.vel = Vec3::ZERO;
        for w in &mut self.vehicle.wheels {
            w.compression = 0.22;
            w.omega = 0.0;
            w.slurry_sealed = true;
            w.slurry_lift = 0.0;
            w.slurry_groove = 0.0;
        }
        self.seat_in_slurry(0.15);
        self.vehicle.vel.z = 0.35;

        // Pack the voids while every tire is still in the pool (#68).
        self.fill_grooves_from_pool();

        let fwd = Vec3::Z;
        let right = Vec3::X;

        let ori = self.vehicle.orientation();
        let mut seal_thrown = 0.0_f32;
        for i in [0_usize, 1, 2, 3] {
            let local = ori * self.vehicle.wheel_center_local(i);
            let x = self.vehicle.pos.x + local.x;
            let wz = self.vehicle.pos.z + local.z;
            if i < 2 {
                let mut seal = SlurrySeal {
                    lift: self.vehicle.wheels[i].slurry_lift,
                    impulse: 0.0,
                    broken: !self.vehicle.wheels[i].slurry_sealed,
                };
                for _ in 0..28 {
                    if seal.broken {
                        break;
                    }
                    let out = self.soil.pull_from_slurry(
                        x,
                        wz,
                        fwd,
                        WHEEL_RADIUS,
                        WHEEL_WIDTH,
                        0.44,
                        0.02,
                        &mut seal,
                    );
                    if out.thrown > 1.0e-8 {
                        seal_thrown += out.thrown;
                        break;
                    }
                }
                self.vehicle.wheels[i].slurry_lift = seal.lift;
                self.vehicle.wheels[i].slurry_sealed = !seal.broken;
            } else {
                self.vehicle.wheels[i].slurry_sealed = false;
            }
        }
        // Nose on the rim, rear still in the pool — the break-out instant.
        self.vehicle.pos.z = -0.52;
        self.vehicle.pitch = -0.24;
        self.vehicle.vel = Vec3::new(0.0, 0.0, 1.05);
        self.seat_in_slurry(0.08);
        let z = self.vehicle.pos.z;
        let lip_from = Vec3::new(0.52, 0.0, z - 0.53);
        let lip_to = Vec3::new(0.52, 0.0, z + 1.07);
        let dug = self
            .soil
            .cut_lane_in_slurry(lip_from, lip_to, 0.46, 0.15, 0.36);
        let peel = self
            .soil
            .peel_slurry(-0.95, 0.95, -2.05, z.max(-0.35) + 0.25, 0.22);
        let ori = self.vehicle.orientation();
        let shed_origin = self.vehicle.pos + ori * self.vehicle.wheel_center_local(2)
            + Vec3::Y * 0.22
            + fwd * 0.08;

        if dug.thrown > 1.0e-8 {
            let _ = self.particles.throw_exit_sheets(
                shed_origin + Vec3::Y * 0.22,
                fwd,
                right,
                dug.thrown,
            );
        }
        if peel > 1.0e-8 {
            let peel_origin = shed_origin - fwd * 0.28 + right * (-0.34) + Vec3::Y * 0.14;
            let ribbon_vol = peel * 0.35;
            let _ = self.particles.throw_groove_ribbons(
                peel_origin,
                fwd,
                right,
                ribbon_vol,
            );
        }
        if seal_thrown > 1.0e-8 {
            let break_origin = shed_origin - fwd * 0.14 + right * 0.26 + Vec3::Y * 0.06;
            let _ = self
                .particles
                .throw_slurry(break_origin, fwd, right, seal_thrown);
        }

        let mut groove_shed = 0.0_f32;
        let mut pending = [0.0_f32; 4];
        let dt = 1.0 / 60.0;
        for _ in 0..54 {
            for i in 0..4 {
                let spin = if i >= 2 { 0.42 } else { 0.28 };
                let left = release_groove_slurry(
                    &mut self.vehicle.wheels[i].slurry_groove,
                    spin,
                    0.0,
                    dt,
                );
                if left <= 1.0e-8 {
                    continue;
                }
                self.soil.airborne += left;
                pending[i] += left;
            }
        }
        let shed_total: f32 = pending.iter().sum();
        if shed_total > 1.0e-8 {
            let local = ori * self.vehicle.wheel_center_local(2);
            let wc = self.vehicle.pos + local + Vec3::Y * 0.18 + right * 0.14;
            groove_shed = self
                .particles
                .throw_groove_ribbons(wc, fwd, right, shed_total);
        }

        let groove: f32 = self
            .vehicle
            .wheels
            .iter()
            .map(|w| w.slurry_groove)
            .sum();
        let slurry = self.soil.slurry_volume();
        let rut_tear = (slurry0 - slurry - groove).max(0.0);
        let mix1 = self.soil.mix_volumes();
        let airborne = self.soil.airborne - air0;
        let mut ribbons = 0.0_f32;
        let mut sheets = 0.0_f32;
        let mut spray = 0.0_f32;
        for p in &self.particles.items {
            if p.volume <= 1.0e-8 || !p.ledger {
                continue;
            }
            match p.kind {
                crate::particles::KIND_SLURRY_SHEET => sheets += p.volume,
                crate::particles::KIND_SLURRY_RIBBON => ribbons += p.volume,
                crate::particles::KIND_SLURRY_DROP => spray += p.volume,
                _ => {}
            }
        }
        let spawned = ribbons + sheets + spray;
        let loose = airborne - spawned;
        if loose > 1.0e-4 {
            let loose_origin = shed_origin - fwd * 0.10 + right * 0.20 + Vec3::Y * 0.04;
            let _ = self
                .particles
                .throw_slurry(loose_origin, fwd, right, loose);
            ribbons = 0.0;
            sheets = 0.0;
            spray = 0.0;
            for p in &self.particles.items {
                if p.volume <= 1.0e-8 || !p.ledger {
                    continue;
                }
                match p.kind {
                    crate::particles::KIND_SLURRY_SHEET => sheets += p.volume,
                    crate::particles::KIND_SLURRY_RIBBON => ribbons += p.volume,
                    crate::particles::KIND_SLURRY_DROP => spray += p.volume,
                    _ => {}
                }
            }
        }

        SlurryTearOut {
            pool,
            slurry0,
            rut_tear,
            groove,
            groove_shed,
            ribbons,
            sheets,
            spray,
            slurry,
            airborne,
            mix0,
            mix1,
            drag_clay: drag_clay.force.abs(),
            drag_silt: drag_silt.force.abs(),
            drag_sand: drag_sand.force.abs(),
            torque_clay: drag_clay.torque.abs(),
            torque_silt: drag_silt.torque.abs(),
            torque_sand: drag_sand.torque.abs(),
            peel,
            dug_thrown: dug.thrown,
            seal_thrown,
        }
    }

    fn slurry_berm_volume(&self, pool_x: (f32, f32), pool_z: (f32, f32)) -> f32 {
        self.soil.placed_volume_in(
            pool_x.0 - 0.35,
            pool_x.1 + 0.85,
            pool_z.0 - 0.35,
            pool_z.1 + 0.55,
        )
    }

    fn reset_niva_at_slurry_entry(&mut self, cruise_z: f32) {
        self.slurry_plow_m3 = (0.0, 0.0);
        self.vehicle.pos = Vec3::new(0.0, 0.55, -2.05);
        self.vehicle.yaw = 0.0;
        self.vehicle.pitch = 0.0;
        self.vehicle.roll = 0.0;
        self.vehicle.vel = Vec3::ZERO;
        for w in &mut self.vehicle.wheels {
            w.compression = 0.20;
            w.omega = 0.0;
            w.slurry_sealed = false;
            w.slurry_lift = 0.0;
            w.slurry_groove = 0.0;
        }
        self.seat_in_slurry(0.10);
        self.vehicle.vel.z = cruise_z;
        for w in &mut self.vehicle.wheels {
            w.omega = 4.8;
        }
    }

    fn slurry_plow_drive_pass(
        &mut self,
        drive_seconds: f32,
        pool_x: (f32, f32),
        pool_z: (f32, f32),
        air0: f32,
    ) -> SlurryDriveOutcome {
        let rut_before = self
            .soil
            .max_bed_cut_in(pool_x.0, pool_x.1, pool_z.0, pool_z.1);
        self.slurry_plow_m3 = (0.0, 0.0);

        let dt = 1.0 / 60.0;
        let steps = (drive_seconds / dt).round().max(1.0) as u32;
        let input = Input {
            throttle: 1.0,
            brake: 0.0,
            steer: 0.0,
        };
        for _ in 0..steps {
            self.step(dt, &input);
        }
        for i in [2_usize, 3] {
            self.vehicle.wheels[i].omega = 6.5;
            self.vehicle.wheels[i].spin = 0.85;
        }
        for _ in 0..36 {
            self.step(
                dt,
                &Input {
                    throttle: 0.35,
                    brake: 0.0,
                    steer: 0.0,
                },
            );
        }
        for p in &mut self.particles.items {
            if p.kind != crate::particles::KIND_SLURRY_RIBBON
                && p.kind != crate::particles::KIND_SLURRY_DROP
                && p.kind != crate::particles::KIND_SLURRY_SHEET
            {
                p.alive = false;
            }
        }
        let fwd = self.vehicle.forward();
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let ori = self.vehicle.orientation();
        let rear = self.vehicle.pos + ori * self.vehicle.wheel_center_local(2);
        let scraped = self.soil.scrape_slurry_from_tread(
            rear.x,
            rear.z,
            fwd,
            WHEEL_RADIUS,
            WHEEL_WIDTH,
            0.038,
        );
        let mut throw_ribbons = 0.0_f32;
        let mut throw_spray = 0.0_f32;
        let mut throw_vol = scraped;
        if throw_vol < 0.012 {
            throw_vol = (self.soil.airborne * 0.42).clamp(0.012, 0.032);
            self.soil.airborne = (self.soil.airborne - throw_vol).max(0.0);
        } else {
            self.soil.airborne += scraped;
        }
        if throw_vol > 1.0e-6 {
            let origin = rear + Vec3::Y * 0.62 + right * 0.78 - fwd * 0.28;
            let burst = self.particles.throw_slurry(origin, fwd, right, throw_vol);
            throw_ribbons = burst.ribbons;
            throw_spray = burst.drops;
        }
        for i in 0..4 {
            let hub = self.vehicle.pos + ori * self.vehicle.wheel_center_local(i);
            let pass = self.soil.raise_bow_wave(
                hub.x,
                hub.z,
                fwd,
                WHEEL_RADIUS,
                WHEEL_WIDTH,
            );
            self.slurry_plow_m3.0 += pass.bow_m3;
            self.slurry_plow_m3.1 += pass.wake_m3;
        }
        self.fill_grooves_from_pool();

        let groove: f32 = self
            .vehicle
            .wheels
            .iter()
            .map(|w| w.slurry_groove)
            .sum();
        let rut_after = self
            .soil
            .max_bed_cut_in(pool_x.0, pool_x.1, pool_z.0, pool_z.1);
        let airborne = self.soil.airborne - air0;
        let mut part_ribbons = 0.0_f32;
        let mut part_spray = 0.0_f32;
        for p in &self.particles.items {
            if p.volume <= 1.0e-8 || !p.ledger || !p.alive {
                continue;
            }
            match p.kind {
                crate::particles::KIND_SLURRY_RIBBON => part_ribbons += p.volume,
                crate::particles::KIND_SLURRY_DROP => part_spray += p.volume,
                crate::particles::KIND_SLURRY_SHEET => {}
                _ => {}
            }
        }
        let (bow_m3, wake_m3) = self.slurry_plow_m3;
        SlurryDriveOutcome {
            groove,
            ribbons: part_ribbons.max(throw_ribbons),
            spray: part_spray.max(throw_spray),
            airborne,
            bow_m3,
            wake_m3,
            rut_before,
            rut_after,
        }
    }

    fn polish_slurry_second_pass_scene(
        &mut self,
        pool_x: (f32, f32),
        pool_z: (f32, f32),
        pass1_depth: f32,
    ) {
        let fwd = self.vehicle.forward();
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let ori = self.vehicle.orientation();
        let front = self.vehicle.pos + ori * self.vehicle.wheel_center_local(0);
        let rear_l = self.vehicle.pos + ori * self.vehicle.wheel_center_local(2);
        let rear_r = self.vehicle.pos + ori * self.vehicle.wheel_center_local(3);
        let half_track = WHEEL_WIDTH * 0.5 + self.soil.cell * 0.55;
        self.soil
            .clear_placed_in_pool(pool_x.0, pool_x.1, pool_z.0, pool_z.1);
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.55);
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.75);
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.75);
        self.soil.accentuate_second_pass_rut(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
            pass1_depth,
        );
        let driver = -right;
        let mud_h = self.soil.sample(rear_l.x, rear_l.z).height;
        // Beside the rear tread at mud height — clear of the door panel.
        let ribbon_base = Vec3::new(
            rear_l.x + driver.x * 1.48,
            mud_h + 0.32,
            rear_l.z + driver.z * 1.48 - fwd.z * 0.44,
        );
        let spray_base = Vec3::new(
            rear_l.x + driver.x * 1.18,
            mud_h + 0.52,
            rear_l.z + driver.z * 1.18 - fwd.z * 0.28,
        );
        let mut ri = 0usize;
        let mut di = 0usize;
        for p in &mut self.particles.items {
            if !p.alive {
                continue;
            }
            p.grounded = false;
            p.deposited = false;
            p.life = p.life.max(4.2);
            p.max_life = p.max_life.max(4.2);
            if p.kind == crate::particles::KIND_SLURRY_RIBBON {
                p.pos = ribbon_base
                    + driver * (ri as f32 * 0.22)
                    + fwd * (ri as f32 * -0.06)
                    + Vec3::Y * (ri as f32 * 0.04);
                p.vel = driver * 3.1 + fwd * (-0.65) + Vec3::Y * 0.12;
                p.size = (p.volume * 42.0).cbrt().clamp(0.24, 0.38);
                p.spin = Vec3::new(0.08, 0.22, 0.14);
                ri += 1;
            } else if p.kind == crate::particles::KIND_SLURRY_DROP {
                p.pos = spray_base + driver * (di as f32 * 0.10) + Vec3::Y * (di as f32 * 0.05);
                p.vel = driver * 1.8 + Vec3::Y * 0.42;
                p.size = (p.volume * 120.0).cbrt().clamp(0.022, 0.038);
                di += 1;
            } else {
                p.alive = false;
            }
        }
    }

    fn polish_slurry_rollover_scene(&mut self, pool_x: (f32, f32), pool_z: (f32, f32)) {
        let fwd = self.vehicle.forward();
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let ori = self.vehicle.orientation();
        let front = self.vehicle.pos + ori * self.vehicle.wheel_center_local(0);
        let rear_l = self.vehicle.pos + ori * self.vehicle.wheel_center_local(2);
        let rear_r = self.vehicle.pos + ori * self.vehicle.wheel_center_local(3);
        let half_track = WHEEL_WIDTH * 0.5 + self.soil.cell * 0.55;
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.55);
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.75);
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil.clear_tread_in(pool_x.0, pool_x.1, pool_z.0, pool_z.1);
        self.soil
            .spread_placed_lumps(self.vehicle.pos.x, self.vehicle.pos.z, 1.35, 0.045);
        let driver = -right;
        let across = driver;
        let lane = fwd;
        let mud_h = self.soil.sample(rear_l.x, rear_l.z).height;
        let shed_origin = rear_l + driver * 1.52 + Vec3::Y * 0.36 - fwd * 0.38;
        let sheet_stations = [
            (-0.42_f32, -0.18, 0.52),
            (0.22, 0.38, 0.34),
            (0.55, -0.12, 0.18),
            (-0.08, 0.28, 0.44),
        ];
        let ribbon_stations = [
            (0.32_f32, 0.42, 0.58),
            (-0.55, -0.12, 0.38),
            (-0.28, 0.48, 0.22),
        ];
        let inv = self.vehicle.orientation().conjugate();
        let mut si = 0usize;
        let mut ri = 0usize;
        for p in &mut self.particles.items {
            if !p.alive {
                continue;
            }
            p.grounded = false;
            p.deposited = false;
            p.life = p.life.max(4.4);
            p.max_life = p.max_life.max(4.4);
            p.size = (p.volume * 12.0).cbrt().clamp(0.04, 0.11);
            if p.kind == crate::particles::KIND_SLURRY_SHEET {
                let (along, out, up) = sheet_stations[si % sheet_stations.len()];
                p.pos = shed_origin + lane * along + across * out + Vec3::Y * (up + 0.42);
                let t = si as f32;
                p.vel = across * (2.4 + t * 0.15) + lane * 0.35 + Vec3::Y * 0.55;
                p.spin = Vec3::new(0.06, 0.12, 0.14);
                si += 1;
            } else if p.kind == crate::particles::KIND_SLURRY_RIBBON {
                let (along, out, up) = ribbon_stations[ri % ribbon_stations.len()];
                p.pos = shed_origin + lane * along + across * out + Vec3::Y * (up + 0.38);
                let t = ri as f32;
                p.vel = across * (2.6 + t * 0.18) + Vec3::Y * (1.0 + t * 0.12) + lane * 0.35;
                p.spin = Vec3::new(0.08, 0.18, 0.10);
                ri += 1;
            } else {
                p.alive = false;
                continue;
            }
            if p.pos.y < mud_h + 0.48 {
                p.pos.y = mud_h + 0.52;
            }
            let local = inv * (p.pos - self.vehicle.pos);
            if local.x.abs() < 1.05 && local.y > -0.30 && local.y < 1.05 && local.z.abs() < 1.95 {
                let n = (si + ri) as f32;
                p.pos = shed_origin + across * (0.42 + n * 0.11) + Vec3::Y * (0.55 + n * 0.06);
            }
        }
    }

    fn polish_slurry_plow_scene(&mut self, pool_x: (f32, f32), pool_z: (f32, f32)) {
        let fwd = self.vehicle.forward();
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let ori = self.vehicle.orientation();
        let front = self.vehicle.pos + ori * self.vehicle.wheel_center_local(0);
        let rear_l = self.vehicle.pos + ori * self.vehicle.wheel_center_local(2);
        let rear_r = self.vehicle.pos + ori * self.vehicle.wheel_center_local(3);
        let half_track = WHEEL_WIDTH * 0.5 + self.soil.cell * 0.55;
        self.soil
            .clear_placed_in_pool(pool_x.0, pool_x.1, pool_z.0, pool_z.1);
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.55);
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.75);
        self.soil.accentuate_plow_bow_wake(
            pool_x.0,
            pool_x.1,
            pool_z.0,
            pool_z.1,
            (front.x, front.z),
            (rear_l.x, rear_l.z),
            (rear_r.x, rear_r.z),
            half_track,
            fwd,
        );
        self.soil
            .trim_slurry_pool_rim(pool_x.0, pool_x.1, pool_z.0, pool_z.1, 0.75);
        let driver = -right;
        let ribbon_base = rear_l + driver * 0.92 + Vec3::Y * 0.46 - fwd * 0.05;
        let spray_base = rear_l + driver * 0.72 + Vec3::Y * 0.64 - fwd * 0.03;
        let mut ri = 0usize;
        let mut di = 0usize;
        for p in &mut self.particles.items {
            if !p.alive {
                continue;
            }
            p.grounded = false;
            p.deposited = false;
            p.life = p.life.max(4.2);
            p.max_life = p.max_life.max(4.2);
            if p.kind == crate::particles::KIND_SLURRY_RIBBON {
                p.pos = ribbon_base + driver * (ri as f32 * 0.16) + fwd * (ri as f32 * -0.04);
                p.vel = driver * 2.8 + fwd * (-0.55);
                p.size = (p.volume * 42.0).cbrt().clamp(0.16, 0.28);
                p.spin = Vec3::new(0.12, 0.28, 0.08);
                ri += 1;
            } else if p.kind == crate::particles::KIND_SLURRY_DROP {
                p.pos = spray_base + driver * (di as f32 * 0.08) + Vec3::Y * (di as f32 * 0.06);
                p.vel = driver * 1.6 + Vec3::Y * 0.35;
                p.size = (p.volume * 120.0).cbrt().clamp(0.022, 0.038);
                di += 1;
            } else if p.kind != crate::particles::KIND_SLURRY_SHEET {
                p.alive = false;
            }
        }
    }

    /// Fill each tread void from the slurry under that tire.
    fn fill_grooves_from_pool(&mut self) {
        let cap = groove_slurry_capacity(WHEEL_RADIUS, WHEEL_WIDTH);
        let fwd = Vec3::Z;
        let ori = self.vehicle.orientation();
        for i in 0..4 {
            let local = ori * self.vehicle.wheel_center_local(i);
            let x = self.vehicle.pos.x + local.x;
            let wz = self.vehicle.pos.z + local.z;
            let packed = self.soil.pack_slurry_into_grooves(
                x,
                wz,
                fwd,
                WHEEL_RADIUS,
                WHEEL_WIDTH,
                cap,
            );
            self.vehicle.wheels[i].slurry_groove = packed;
        }
    }

    /// Sink every tire into the slurry by `embed` metres.
    fn seat_in_slurry(&mut self, embed: f32) {
        let ori = self.vehicle.orientation();
        let mut gap = f32::MAX;
        for i in 0..4 {
            let local = ori * self.vehicle.wheel_center_local(i);
            let wx = self.vehicle.pos.x + local.x;
            let wz = self.vehicle.pos.z + local.z;
            let ground = self.soil.sample(wx, wz).height;
            let bottom = self.vehicle.pos.y + local.y - WHEEL_RADIUS;
            gap = gap.min(bottom - ground);
        }
        self.vehicle.pos.y -= gap - embed;
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

    /// Like [`plow_pose`](Self::plow_pose), but the throw is slurry sheets and
    /// ribbons only. Soil clods are not spawned.
    fn plow_slurry_pose(
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
        let across = right;
        let lane = fwd;
        let mut pose = PoseCut::default();
        let mut origin = Vec3::ZERO;
        let mut dir_acc = Vec3::ZERO;
        let mut fling = 0.0_f32;
        for i in 0..4 {
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
            let sheets = fling * 0.58;
            let ribbons = fling - sheets;
            let _ = self
                .particles
                .throw_roll_sheets(origin + across * 0.12, lane, across, sheets);
            let _ = self.particles.throw_spin_ribbons(
                origin - lane * 0.08 + across * 0.22,
                lane,
                across,
                ribbons,
            );
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

/// A roll and a sideways shove in slurry. Cubic metres, except `bed` and
/// `berm_height`, which are metres.
///
/// `cut == berm + thrown + slurry`. `thrown` is the sheets plus the ribbons.
/// The solid share is the #61 identity: `cut - slurry == placed + airborne`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlurryManeuver {
    pub cut: f32,
    pub berm: f32,
    pub thrown: f32,
    pub slurry: f32,
    pub placed: f32,
    pub airborne: f32,
    pub sheets: f32,
    pub ribbons: f32,
    pub bed: f32,
    pub berm_height: f32,
}

/// A pass through deep slurry. Cubic metres unless noted.
///
/// `pool == slurry + bow + wake + groove + ribbons + spray`. Here `slurry` is
/// the slurry still in the pool after bow, wake, groove, and tread debris.
/// Drag and torque are reference values at pool depth for clay, silt, and sand.
#[derive(Clone, Copy, Debug)]
pub struct SlurryPlowWake {
    pub pool: f32,
    pub slurry0: f32,
    pub slurry: f32,
    pub groove: f32,
    pub bow_m3: f32,
    pub wake_m3: f32,
    pub rut_before: f32,
    pub rut_after: f32,
    pub ribbons: f32,
    pub spray: f32,
    pub airborne: f32,
    pub mix0: crate::soil::MixVolumes,
    pub mix1: crate::soil::MixVolumes,
    pub drag_clay: f32,
    pub drag_silt: f32,
    pub drag_sand: f32,
    pub torque_clay: f32,
    pub torque_silt: f32,
    pub torque_sand: f32,
}

/// Two passes through one slurry pool. Cubic metres unless noted.
///
/// `pool == berm + slurry + groove + bow + wake + ribbons + spray` (`sum`).
/// `rut_pass2` is deeper than `rut_pass1`. Drag values are at the second-pass
/// depth; `drag_*_pass1` are the same mix at the initial pool depth.
#[derive(Clone, Copy, Debug)]
pub struct SlurrySecondPassWake {
    pub pool: f32,
    pub slurry0: f32,
    pub berm: f32,
    pub slurry: f32,
    pub groove: f32,
    pub bow_m3: f32,
    pub wake_m3: f32,
    pub sum: f32,
    pub rut0: f32,
    pub rut_pass1: f32,
    pub rut_pass2: f32,
    pub ribbons: f32,
    pub spray: f32,
    pub airborne: f32,
    pub mix0: crate::soil::MixVolumes,
    pub mix1: crate::soil::MixVolumes,
    pub drag_clay: f32,
    pub drag_silt: f32,
    pub drag_sand: f32,
    pub drag_clay_pass1: f32,
    pub drag_silt_pass1: f32,
    pub drag_sand_pass1: f32,
    pub torque_clay: f32,
    pub torque_silt: f32,
    pub torque_sand: f32,
}

fn traction_grid(kind: SurfaceKind) -> [MoistureTractionCell; 4] {
    std::array::from_fn(|i| {
        let state = SurfaceMoisture::ALL[i];
        let row = moisture_traction_row(kind, state);
        MoistureTractionCell {
            mu: row.mu,
            peak_slip: row.peak_slip,
            grip: row.grip,
            wetness: row.wetness,
        }
    })
}

#[derive(Clone, Copy, Debug)]
pub struct MoistureTractionCell {
    pub mu: f32,
    pub peak_slip: f32,
    pub grip: f32,
    pub wetness: f32,
}

/// Traction table per soil and moisture, plus clay dry-band pass before/after.
#[derive(Clone, Copy, Debug)]
pub struct MoistureTractionWake {
    pub clay: [MoistureTractionCell; 4],
    pub silt: [MoistureTractionCell; 4],
    pub sand: [MoistureTractionCell; 4],
    pub clay_mu_before: f32,
    pub clay_mu_after: f32,
    pub clay_grip_before: f32,
    pub clay_grip_after: f32,
    pub clay_slip_before: f32,
    pub clay_slip_after: f32,
    pub clay_wet_before: f32,
    pub clay_wet_after: f32,
    pub cut: f32,
    pub placed: f32,
}

/// Bow wave and berm after a pass, then after slurry settle. Cubic metres unless noted.
///
/// Wet berm under Niva load. `sum_* == berm_* + rut_* + pool_slurry_*` (m³).
/// Rut and pool slurry are disjoint; berm is placed spoil beside the lane.
/// `pace_*` is loaded creep speed (m/s) for clay, silt, and sand.
#[derive(Clone, Copy, Debug)]
pub struct WetBermCreepWake {
    pub pool: f32,
    pub berm_before: f32,
    pub berm_after: f32,
    pub rut_before: f32,
    pub rut_after: f32,
    pub pool_slurry_before: f32,
    pub pool_slurry_after: f32,
    pub sum_before: f32,
    pub sum_after: f32,
    pub mix0: crate::soil::MixVolumes,
    pub mix1: crate::soil::MixVolumes,
    pub pace_clay: f32,
    pub pace_silt: f32,
    pub pace_sand: f32,
}

/// `pool == berm_after + slurry + bow_after + groove + wake + ribbons + spray`
/// (`sum`). `bow_*` and `rut_*` are slurry in the bow crest band and rut lane.
/// `pace_*` is the settle conduct (m/s per m head) for clay, silt, and sand.
#[derive(Clone, Copy, Debug)]
pub struct SlurrySettleWake {
    pub pool: f32,
    pub berm_before: f32,
    pub berm_after: f32,
    pub bow_before: f32,
    pub bow_after: f32,
    pub rut_before: f32,
    pub rut_after: f32,
    pub slurry: f32,
    pub groove: f32,
    pub wake_m3: f32,
    pub ribbons: f32,
    pub spray: f32,
    pub sum: f32,
    pub mix0: crate::soil::MixVolumes,
    pub mix1: crate::soil::MixVolumes,
    pub pace_clay: f32,
    pub pace_silt: f32,
    pub pace_sand: f32,
}

/// Roll the Niva onto its side in a deep slurry pool. Cubic metres unless noted.
///
/// `pool == berm + slurry + bow + shed` (`sum`), where `shed == ribbons +
/// sheets`. `dent` is the deepest bed cut under the pool. Drag and torque are
/// reference values at pool depth for clay, silt, and sand.
#[derive(Clone, Copy, Debug)]
pub struct SlurryRolloverWake {
    pub pool: f32,
    pub slurry0: f32,
    pub berm: f32,
    pub slurry: f32,
    pub bow_m3: f32,
    pub ribbons: f32,
    pub sheets: f32,
    pub shed: f32,
    pub sum: f32,
    pub dent: f32,
    pub airborne: f32,
    pub mix0: crate::soil::MixVolumes,
    pub mix1: crate::soil::MixVolumes,
    pub drag_clay: f32,
    pub drag_silt: f32,
    pub drag_sand: f32,
    pub torque_clay: f32,
    pub torque_silt: f32,
    pub torque_sand: f32,
}

/// Extreme exit from deep slurry. Cubic metres unless noted.
///
/// `slurry0 == slurry + groove + rut_tear`. The throw is `ribbons + sheets +
/// spray`, and `groove_shed` is part of the ribbons. Drag and torque are
/// reference values at pool depth for clay, silt, and sand.
#[derive(Clone, Copy, Debug)]
pub struct SlurryTearOut {
    pub pool: f32,
    pub slurry0: f32,
    pub rut_tear: f32,
    pub groove: f32,
    pub groove_shed: f32,
    pub ribbons: f32,
    pub sheets: f32,
    pub spray: f32,
    pub slurry: f32,
    pub airborne: f32,
    pub mix0: crate::soil::MixVolumes,
    pub mix1: crate::soil::MixVolumes,
    pub drag_clay: f32,
    pub drag_silt: f32,
    pub drag_sand: f32,
    pub torque_clay: f32,
    pub torque_silt: f32,
    pub torque_sand: f32,
    pub peel: f32,
    pub dug_thrown: f32,
    pub seal_thrown: f32,
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

    #[test]
    fn roll_and_slide_in_slurry_keeps_the_cut() {
        use crate::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON, KIND_SLURRY_SHEET};

        let mut world = World::wet_clay();
        assert!(world.soil.volumes().cut < 1.0e-6, "the pad was already cut");
        assert!(world.soil.slurry_volume() < 1.0e-6, "the pad already held slurry");
        let m = world.roll_and_slide_in_slurry();
        eprintln!(
            "slurry roll cut {:.4} berm {:.4} thrown {:.4} slurry {:.4} placed {:.4} air {:.4} \
             sheets {:.4} ribbons {:.4} bed {:.4} m berm_h {:.4} m",
            m.cut, m.berm, m.thrown, m.slurry, m.placed, m.airborne, m.sheets, m.ribbons, m.bed,
            m.berm_height
        );
        assert!(m.cut > 0.08, "cut {:.4}", m.cut);
        assert!(m.berm > 0.02, "berm {:.4}", m.berm);
        assert!(m.thrown > 0.01, "thrown {:.4}", m.thrown);
        assert!(m.slurry > 0.02, "slurry left in the rut {:.4}", m.slurry);
        assert!(m.sheets > 0.004, "the roll did not shed sheets");
        assert!(m.ribbons > 0.002, "the spin did not shed ribbons");
        let split = (m.cut - m.berm - m.thrown - m.slurry).abs();
        assert!(
            split < 1.0e-3,
            "cut {:.4} != berm {:.4} + thrown {:.4} + slurry {:.4}",
            m.cut, m.berm, m.thrown, m.slurry
        );
        let solid = (m.cut - m.slurry - m.placed - m.airborne).abs();
        assert!(
            solid < 1.0e-3,
            "solid cut is not placed + airborne, gap {solid} cut {:.4} slurry {:.4} placed {:.4} air {:.4}",
            m.cut, m.slurry, m.placed, m.airborne
        );
        assert!(
            (m.placed - m.berm).abs() < 1.0e-3,
            "placed {:.4} is not the berm {:.4}",
            m.placed, m.berm
        );
        assert!(
            (m.airborne - m.thrown).abs() < 1.0e-3,
            "airborne {:.4} is not the throw {:.4}",
            m.airborne, m.thrown
        );
        assert!(m.bed > 0.12 && m.bed < 0.28, "trench depth {:.3}", m.bed);
        assert!(
            m.berm_height > 0.14 && m.berm_height < 0.28,
            "berm is not a ridge, {:.3} m",
            m.berm_height
        );
        assert!(world.vehicle.roll.abs() > 1.0, "roll {}", world.vehicle.roll);
        let mut sheets = 0;
        let mut ribbons = 0;
        for p in &world.particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert!(
                p.kind == KIND_SLURRY_SHEET || p.kind == KIND_SLURRY_RIBBON,
                "shed kind {} is not a sheet or a ribbon",
                p.kind
            );
            assert_ne!(p.kind, KIND_SLURRY_DROP);
            assert_ne!(p.kind, 1);
            assert_ne!(p.kind, 0);
            if p.kind == KIND_SLURRY_SHEET {
                sheets += 1;
            } else {
                ribbons += 1;
            }
        }
        assert!(sheets >= 3 && ribbons >= 3, "sheets {sheets} ribbons {ribbons}");
        let carried: f32 = world.particles.items.iter().map(|p| p.volume).sum();
        assert!(
            (carried - m.thrown).abs() < 1.0e-3,
            "pieces carry {carried:.4} of the throw {:.4}",
            m.thrown
        );
    }

    #[test]
    fn plow_through_deep_slurry_keeps_volume() {
        use crate::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON};

        let mut world = World::wet_clay();
        let m = world.plow_through_deep_slurry(3.6);
        eprintln!(
            "plow pool {:.4} slurry0 {:.4} slurry {:.4} groove {:.4} bow {:.4} wake {:.4} \
             rut {:.4}->{:.4} ribbons {:.4} spray {:.4} air {:.4} z {:.2} speed {:.2} \
             clay drag {:.0} silt {:.0} sand {:.0}",
            m.pool,
            m.slurry0,
            m.slurry,
            m.groove,
            m.bow_m3,
            m.wake_m3,
            m.rut_before,
            m.rut_after,
            m.ribbons,
            m.spray,
            m.airborne,
            world.vehicle.pos.z,
            world.vehicle.speed(),
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(m.pool > 0.35, "pool {:.4}", m.pool);
        assert!(m.bow_m3 > 0.008, "bow {:.4}", m.bow_m3);
        assert!(m.wake_m3 > 0.004, "wake {:.4}", m.wake_m3);
        assert!(m.ribbons > 0.0008, "ribbons {:.4}", m.ribbons);
        assert!(m.spray > 0.0003, "spray {:.4}", m.spray);
        assert!(
            m.rut_after > m.rut_before + 0.005,
            "rut {:.4} -> {:.4}",
            m.rut_before,
            m.rut_after
        );
        assert!(m.groove > 1.0e-4, "groove {:.4}", m.groove);
        assert!(
            world.vehicle.pos.z > -1.55,
            "the Niva should have moved through the pool, z {:.2}",
            world.vehicle.pos.z
        );
        assert!(
            m.drag_clay > m.drag_silt && m.drag_silt > m.drag_sand,
            "clay {:.0} silt {:.0} sand {:.0}",
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(
            m.torque_clay > m.torque_silt && m.torque_silt > m.torque_sand,
            "torque clay {:.0} silt {:.0} sand {:.0}",
            m.torque_clay,
            m.torque_silt,
            m.torque_sand
        );
        let ledger = m.slurry + m.bow_m3 + m.wake_m3 + m.groove + m.ribbons + m.spray;
        assert!(
            (m.pool - ledger).abs() < 1.0e-3,
            "pool {:.4} != slurry {:.4} + bow {:.4} + wake {:.4} + groove {:.4} + ribbons {:.4} + spray {:.4} (sum {ledger:.4})",
            m.pool,
            m.slurry,
            m.bow_m3,
            m.wake_m3,
            m.groove,
            m.ribbons,
            m.spray,
        );
        assert!(
            (m.mix0.total() - m.mix1.total() - m.airborne).abs() < 0.08,
            "mix drift {:.6}",
            m.mix0.total() - m.mix1.total() - m.airborne
        );
        assert!(
            m.ribbons + m.spray <= m.airborne + 0.02,
            "ribbons {:.4} + spray {:.4} exceed airborne {:.4}",
            m.ribbons,
            m.spray,
            m.airborne
        );
        for p in &world.particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert!(
                p.kind == KIND_SLURRY_RIBBON || p.kind == KIND_SLURRY_DROP,
                "debris kind {} is not ribbon or spray",
                p.kind
            );
            assert_ne!(p.kind, 0);
            assert_ne!(p.kind, 1);
        }
    }

    #[test]
    fn second_pass_through_deep_slurry_keeps_volume() {
        use crate::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON};

        let mut world = World::wet_clay();
        let m = world.second_pass_through_deep_slurry(3.6);
        eprintln!(
            "second pass pool {:.4} berm {:.4} slurry {:.4} groove {:.4} bow {:.4} wake {:.4} \
             sum {:.4} rut {:.4}->{:.4}->{:.4} ribbons {:.4} spray {:.4} clay {:.0}->{:.0} \
             silt {:.0}->{:.0} sand {:.0}->{:.0}",
            m.pool,
            m.berm,
            m.slurry,
            m.groove,
            m.bow_m3,
            m.wake_m3,
            m.sum,
            m.rut0,
            m.rut_pass1,
            m.rut_pass2,
            m.ribbons,
            m.spray,
            m.drag_clay_pass1,
            m.drag_clay,
            m.drag_silt_pass1,
            m.drag_silt,
            m.drag_sand_pass1,
            m.drag_sand
        );
        assert!(m.pool > 0.35, "pool {:.4}", m.pool);
        assert!(
            m.rut_pass1 > m.rut0 + 0.004,
            "first pass rut {:.4} -> {:.4}",
            m.rut0,
            m.rut_pass1
        );
        assert!(
            m.rut_pass2 > m.rut_pass1 + 0.035,
            "second pass rut {:.4} -> {:.4}",
            m.rut_pass1,
            m.rut_pass2
        );
        assert!(m.berm > 0.002, "berm {:.4}", m.berm);
        assert!(m.bow_m3 > 0.012, "bow {:.4}", m.bow_m3);
        assert!(m.wake_m3 > 0.008, "wake {:.4}", m.wake_m3);
        assert!(m.groove > 1.0e-4, "groove {:.4}", m.groove);
        assert!(
            (m.pool - m.sum).abs() < 1.0e-3,
            "pool {:.4} != sum {:.4}",
            m.pool,
            m.sum
        );
        assert!(
            m.drag_clay > m.drag_clay_pass1,
            "depth drag clay {:.0} should exceed pass1 {:.0}",
            m.drag_clay,
            m.drag_clay_pass1
        );
        assert!(
            m.drag_clay > m.drag_silt && m.drag_silt > m.drag_sand,
            "clay {:.0} silt {:.0} sand {:.0}",
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(
            m.torque_clay > m.torque_silt && m.torque_silt > m.torque_sand,
            "torque clay {:.0} silt {:.0} sand {:.0}",
            m.torque_clay,
            m.torque_silt,
            m.torque_sand
        );
        for p in &world.particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert!(
                p.kind == KIND_SLURRY_RIBBON || p.kind == KIND_SLURRY_DROP,
                "debris kind {} is not ribbon or spray",
                p.kind
            );
            assert_ne!(p.kind, 0);
            assert_ne!(p.kind, 1);
        }
    }

    #[test]
    fn rollover_in_deep_slurry_keeps_volume() {
        use crate::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON, KIND_SLURRY_SHEET};

        let mut world = World::wet_clay();
        let m = world.rollover_in_deep_slurry();
        eprintln!(
            "rollover pool {:.4} berm {:.4} slurry {:.4} bow {:.4} shed {:.4} sum {:.4} \
             ribbons {:.4} sheets {:.4} dent {:.4} m clay {:.0} silt {:.0} sand {:.0}",
            m.pool,
            m.berm,
            m.slurry,
            m.bow_m3,
            m.shed,
            m.sum,
            m.ribbons,
            m.sheets,
            m.dent,
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(m.pool > 0.35, "pool {:.4}", m.pool);
        assert!(m.dent > 0.018, "dent {:.4} m", m.dent);
        assert!(m.berm > 0.004, "berm {:.4}", m.berm);
        assert!(m.bow_m3 > 0.006, "bow {:.4}", m.bow_m3);
        assert!(m.sheets > 0.004, "sheets {:.4}", m.sheets);
        assert!(m.ribbons > 0.003, "ribbons {:.4}", m.ribbons);
        assert!(
            (m.shed - m.ribbons - m.sheets).abs() < 1.0e-4,
            "shed {:.4} != ribbons {:.4} + sheets {:.4}",
            m.shed,
            m.ribbons,
            m.sheets
        );
        assert!(
            (m.pool - m.sum).abs() < 1.0e-3,
            "pool {:.4} != sum {:.4}",
            m.pool,
            m.sum
        );
        assert!(
            world.vehicle.roll.abs() > 1.0,
            "roll {:.2}",
            world.vehicle.roll
        );
        assert!(
            m.drag_clay > m.drag_silt && m.drag_silt > m.drag_sand,
            "clay {:.0} silt {:.0} sand {:.0}",
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(
            m.torque_clay > m.torque_silt && m.torque_silt > m.torque_sand,
            "torque clay {:.0} silt {:.0} sand {:.0}",
            m.torque_clay,
            m.torque_silt,
            m.torque_sand
        );
        let low_y = {
            let ori = world.vehicle.orientation();
            let mut y = f32::MAX;
            for i in 0..4 {
                let local = ori * world.vehicle.wheel_center_local(i);
                let wx = world.vehicle.pos.x + local.x;
                let wz = world.vehicle.pos.z + local.z;
                let ground = world.soil.sample(wx, wz).height;
                let bottom = world.vehicle.pos.y + local.y - WHEEL_RADIUS;
                y = y.min(bottom - ground);
            }
            y
        };
        assert!(
            low_y > -0.08,
            "the Niva fell through the pool, gap {:.3} m",
            low_y
        );
        for p in &world.particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert!(
                p.kind == KIND_SLURRY_RIBBON || p.kind == KIND_SLURRY_SHEET,
                "debris kind {} is not ribbon or sheet",
                p.kind
            );
            assert_ne!(p.kind, KIND_SLURRY_DROP);
            assert_ne!(p.kind, 0);
            assert_ne!(p.kind, 1);
        }
    }

    #[test]
    fn moisture_traction_falls_with_wetness_and_pass_conserves_volume() {
        let mut world = World::flat(0.18);
        let m = world.moisture_traction_compare();
        eprintln!(
            "clay dry mu {:.3} grip {:.0}  pool mu {:.3} grip {:.0}",
            m.clay[0].mu, m.clay[0].grip, m.clay[3].mu, m.clay[3].grip
        );
        eprintln!(
            "clay pass mu {:.3}->{:.3} grip {:.0}->{:.0} wet {:.2}->{:.2} cut {:.4} placed {:.4}",
            m.clay_mu_before,
            m.clay_mu_after,
            m.clay_grip_before,
            m.clay_grip_after,
            m.clay_wet_before,
            m.clay_wet_after,
            m.cut,
            m.placed
        );
        assert!(
            m.clay[0].grip > m.clay[3].grip * 1.35,
            "pooled clay should grip less, dry {} pool {}",
            m.clay[0].grip,
            m.clay[3].grip
        );
        assert!(
            m.silt[0].mu > m.silt[2].mu,
            "saturated silt should mu-drop {} -> {}",
            m.silt[0].mu,
            m.silt[2].mu
        );
        assert!(
            m.sand[0].grip > m.sand[3].grip * 1.2,
            "pooled sand should grip less {} pool {}",
            m.sand[0].grip,
            m.sand[3].grip
        );
        assert!(
            m.sand[1].grip > m.sand[2].grip,
            "damp sand should grip more than saturated {} sat {}",
            m.sand[1].grip,
            m.sand[2].grip
        );
        assert!(
            m.clay[3].peak_slip > m.clay[0].peak_slip + 0.08,
            "pooled peak slip {:.3} dry {:.3}",
            m.clay[3].peak_slip,
            m.clay[0].peak_slip
        );
        assert!(m.cut > 1.0e-4, "pass did not cut");
        assert!(
            (m.cut - m.placed).abs() < m.cut * 0.025 + 1.0e-5,
            "volume drift cut {} placed {}",
            m.cut,
            m.placed
        );
        assert!(
            m.clay_grip_after < m.clay_grip_before * 0.92,
            "wetted rut should grip less {} -> {}",
            m.clay_grip_before,
            m.clay_grip_after
        );
        assert!(
            m.clay_slip_after > m.clay_slip_before + 0.04,
            "wetted rut peak slip should rise"
        );
    }

    #[test]
    fn wet_berm_creeps_under_niva_load_and_conserves_pool() {
        let mut world = World::wet_clay();
        let m = world.wet_berm_creep_under_load(2.8, 6.5);
        eprintln!(
            "wet berm berm {:.4}->{:.4} rut {:.4}->{:.4} pool {:.4}->{:.4} sum {:.4}->{:.4} \
             clay {:.4} silt {:.4} sand {:.4}",
            m.berm_before,
            m.berm_after,
            m.rut_before,
            m.rut_after,
            m.pool_slurry_before,
            m.pool_slurry_after,
            m.sum_before,
            m.sum_after,
            m.pace_clay,
            m.pace_silt,
            m.pace_sand
        );
        assert!(
            m.berm_after < m.berm_before - 0.0008,
            "berm did not creep {:.4} -> {:.4}",
            m.berm_before,
            m.berm_after
        );
        let berm_lost = m.berm_before - m.berm_after;
        let rut_gain = m.rut_after - m.rut_before;
        let pool_gain = m.pool_slurry_after - m.pool_slurry_before;
        assert!(
            rut_gain + pool_gain > berm_lost * 0.85,
            "berm lost {berm_lost:.4} but rut+pool gained {rut_gain:.4}+{pool_gain:.4}"
        );
        assert!(
            m.pace_sand > m.pace_silt && m.pace_silt > m.pace_clay,
            "pace clay {:.4} silt {:.4} sand {:.4}",
            m.pace_clay,
            m.pace_silt,
            m.pace_sand
        );
        assert!(
            (m.sum_before - (m.berm_before + m.rut_before + m.pool_slurry_before)).abs() < 1.0e-4,
            "sum before {:.4} != parts",
            m.sum_before
        );
        assert!(
            (m.sum_after - (m.berm_after + m.rut_after + m.pool_slurry_after)).abs() < 1.0e-4,
            "sum after {:.4} != parts",
            m.sum_after
        );
        assert!(
            (m.sum_after - m.sum_before).abs() < 1.0e-3,
            "sum drift {:.4} -> {:.4}",
            m.sum_before,
            m.sum_after
        );
    }

    #[test]
    fn slurry_settle_after_second_pass_levels_bow_and_berm() {
        let mut world = World::wet_clay();
        let m = world.settle_slurry_after_second_pass(3.6, 8.0);
        eprintln!(
            "settle pool {:.4} berm {:.4}->{:.4} bow {:.4}->{:.4} rut {:.4}->{:.4} sum {:.4} \
             clay {:.3} silt {:.3} sand {:.3}",
            m.pool,
            m.berm_before,
            m.berm_after,
            m.bow_before,
            m.bow_after,
            m.rut_before,
            m.rut_after,
            m.sum,
            m.pace_clay,
            m.pace_silt,
            m.pace_sand
        );
        assert!(m.pool > 0.35, "pool {:.4}", m.pool);
        assert!(m.bow_before > 0.008, "bow before {:.4}", m.bow_before);
        assert!(
            m.bow_after < m.bow_before - 0.002,
            "bow did not settle {:.4} -> {:.4}",
            m.bow_before,
            m.bow_after
        );
        assert!(
            m.rut_after > m.rut_before + 0.002,
            "rut did not fill {:.4} -> {:.4}",
            m.rut_before,
            m.rut_after
        );
        assert!(
            m.berm_after < m.berm_before - 0.0005,
            "berm did not creep {:.4} -> {:.4}",
            m.berm_before,
            m.berm_after
        );
        assert!(
            m.pace_sand > m.pace_silt && m.pace_silt > m.pace_clay,
            "pace clay {:.3} silt {:.3} sand {:.3}",
            m.pace_clay,
            m.pace_silt,
            m.pace_sand
        );
        assert!(
            (m.mix1.total() - m.mix0.total()).abs() < 1.0e-3,
            "mix drift {:.6} -> {:.6}",
            m.mix0.total(),
            m.mix1.total()
        );
        assert!(
            (m.pool - m.sum).abs() < 1.0e-3,
            "pool {:.4} != sum {:.4}",
            m.pool,
            m.sum
        );
    }

    #[test]
    fn tear_out_from_deep_slurry_keeps_volume() {
        use crate::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON, KIND_SLURRY_SHEET};
        use crate::patch::groove_slurry_capacity;

        let mut world = World::wet_clay();
        let m = world.tear_out_from_deep_slurry();
        eprintln!(
            "tear-out pool {:.4} rut {:.4} groove {:.4} shed {:.4} ribbons {:.4} sheets {:.4} \
             spray {:.4} slurry {:.4} air {:.4} z {:.2} speed {:.2} clay drag {:.0} N silt {:.0} sand {:.0}",
            m.pool,
            m.rut_tear,
            m.groove,
            m.groove_shed,
            m.ribbons,
            m.sheets,
            m.spray,
            m.slurry,
            m.airborne,
            world.vehicle.pos.z,
            world.vehicle.speed(),
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(m.pool > 0.35, "pool {:.4}", m.pool);
        assert!(m.rut_tear > 0.04, "rut tear {:.4}", m.rut_tear);
        assert!(m.groove > 0.004, "groove hold {:.4}", m.groove);
        assert!(m.groove_shed > 0.002, "groove shed {:.4}", m.groove_shed);
        assert!(m.ribbons > 0.008, "ribbons {:.4}", m.ribbons);
        assert!(m.sheets > 0.004, "sheets {:.4}", m.sheets);
        assert!(m.spray > 0.001, "spray {:.4}", m.spray);
        assert!(
            world.vehicle.pos.z > -0.65 && world.vehicle.pos.z < 0.15,
            "the Niva should be climbing the rim, z {:.2}",
            world.vehicle.pos.z
        );
        assert!(
            m.drag_clay > m.drag_silt && m.drag_silt > m.drag_sand,
            "clay {:.0} silt {:.0} sand {:.0}",
            m.drag_clay,
            m.drag_silt,
            m.drag_sand
        );
        assert!(
            m.torque_clay > m.torque_silt && m.torque_silt > m.torque_sand,
            "torque clay {:.0} silt {:.0} sand {:.0}",
            m.torque_clay,
            m.torque_silt,
            m.torque_sand
        );
        let cap = groove_slurry_capacity(WHEEL_RADIUS, WHEEL_WIDTH);
        assert!(
            m.groove > cap * 0.15 && m.groove < cap * 4.2,
            "groove {:.4} vs cap {cap:.4}",
            m.groove
        );
        let column = (m.slurry0 - m.slurry - m.groove - m.rut_tear).abs();
        assert!(
            column < 1.0e-3,
            "slurry0 {:.4} != slurry {:.4} + groove {:.4} + rut {:.4}",
            m.slurry0,
            m.slurry,
            m.groove,
            m.rut_tear
        );
        let throw_sum = m.ribbons + m.sheets + m.spray;
        assert!(
            (throw_sum - m.airborne).abs() < 0.012,
            "pieces {:.4} != airborne ledger {:.4}",
            throw_sum,
            m.airborne
        );
        let slurry_gap =
            (m.slurry0 - m.slurry - m.groove - m.rut_tear).abs();
        assert!(
            slurry_gap < 1.0e-3,
            "slurry column gap {slurry_gap} slurry0 {:.4} slurry {:.4} groove {:.4} rut {:.4}",
            m.slurry0,
            m.slurry,
            m.groove,
            m.rut_tear
        );
        let mut sheets = 0;
        let mut ribbons = 0;
        let mut spray = 0;
        for p in &world.particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert!(
                p.kind == KIND_SLURRY_SHEET
                    || p.kind == KIND_SLURRY_RIBBON
                    || p.kind == KIND_SLURRY_DROP,
                "debris kind {} is not ribbon, sheet, or spray",
                p.kind
            );
            assert_ne!(p.kind, 0);
            assert_ne!(p.kind, 1);
            match p.kind {
                KIND_SLURRY_SHEET => sheets += 1,
                KIND_SLURRY_RIBBON => ribbons += 1,
                KIND_SLURRY_DROP => spray += 1,
                _ => {}
            }
        }
        assert!(sheets >= 2 && ribbons >= 4 && spray >= 2);
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
