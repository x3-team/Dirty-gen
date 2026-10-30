//! Motocross bike. Two wheels, own spring, same soil functions as the buggy.
//!
//! Bekker, Janosi–Hanamoto and the wheel stamp are the buggy's. The chassis
//! numbers are not: a bike is lighter, narrower, and has longer travel so the
//! fork and shock actually shorten. The crest launch uses the same impulse
//! as the buggy, counted on two wheels instead of four.

use glam::{EulerRot, Quat, Vec3};

use crate::particles::Particles;
use crate::patch::{contact_patch, step_tread, TreadCoat};
use crate::soil::{
    bekker_load, contact_length, janosi_hanamoto, mech_for_sample, on_launch_crest,
    rolling_resistance, stick_scale, wheel_buoyancy, SoilField, Terramech,
};
use crate::vehicle::{
    PanelMud, Wheel, PANEL_BELLY, PANEL_COUNT, PANEL_MUD_CAP, PANEL_NOSE, PANEL_SEAT,
};

pub const BIKE_RADIUS: f32 = 0.34;
pub const BIKE_WIDTH: f32 = 0.13;
const MASS: f32 = 155.0;
const REST_LEN: f32 = 0.46;
const TRAVEL: f32 = 0.32;
const SUSP_K: f32 = 3_600.0;
const SUSP_PRELOAD: f32 = 220.0;
const I_YAW: f32 = 22.0;
const I_PITCH: f32 = 18.0;
const I_ROLL: f32 = 6.5;
const WHEEL_INERTIA: f32 = 0.35;
const MAX_STEER: f32 = 0.55;
const ENGINE_TORQUE: f32 = 210.0;
const MAX_OMEGA: f32 = 46.0;

/// Front, then rear. Local space, Y up, Z forward.
pub fn hardpoints() -> [Vec3; 2] {
    [Vec3::new(0.0, 0.52, 0.74), Vec3::new(0.0, 0.46, -0.62)]
}

#[derive(Clone, Debug)]
pub struct Bike {
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub yaw_rate: f32,
    pub pitch_rate: f32,
    pub roll_rate: f32,
    pub wheels: [Wheel; 2],
    pub panels: [PanelMud; PANEL_COUNT],
    pub jump_hold: f32,
    crest_armed: bool,
    air_time: f32,
    impact_latched: bool,
    did_launch: bool,
}

impl Bike {
    /// Parked on the outer shoulder of the south cap, clear of the buggy in the
    /// isometric view. The lap pulls it onto a line just inside the buggy's.
    pub fn new() -> Self {
        Self {
            pos: Vec3::new(3.05, 0.7, -7.25),
            vel: Vec3::ZERO,
            // Clockwise tangent on the south-east cap, so it joins the loop
            // instead of nosing across the buggy's start gate.
            yaw: -2.03,
            pitch: 0.0,
            roll: 0.0,
            yaw_rate: 0.0,
            pitch_rate: 0.0,
            roll_rate: 0.0,
            wheels: [Wheel::default(); 2],
            panels: [PanelMud::default(); PANEL_COUNT],
            jump_hold: 0.0,
            crest_armed: false,
            air_time: 0.0,
            impact_latched: false,
            did_launch: false,
        }
    }

    pub fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, self.roll)
    }

    pub fn forward(&self) -> Vec3 {
        let f = self.orientation() * Vec3::Z;
        Vec3::new(f.x, 0.0, f.z).normalize_or_zero()
    }

    pub fn speed(&self) -> f32 {
        Vec3::new(self.vel.x, 0.0, self.vel.z).length()
    }

    pub fn wheel_center_local(&self, index: usize) -> Vec3 {
        let hp = hardpoints()[index];
        let c = self.wheels[index].compression;
        Vec3::new(hp.x, hp.y - REST_LEN + c, hp.z)
    }

    pub fn place_on(&mut self, soil: &SoilField) {
        let sample = soil.sample(self.pos.x, self.pos.z);
        self.pos.y = sample.height + 0.58;
        self.wheels[0].compression = 0.10;
        self.wheels[1].compression = 0.12;
        self.vel = Vec3::ZERO;
    }

    pub fn update_steer(&mut self, steer: f32, dt: f32) {
        let target = steer.clamp(-1.0, 1.0) * MAX_STEER;
        let blend = 1.0 - (-dt * 8.0).exp();
        self.wheels[0].steer += (target - self.wheels[0].steer) * blend;
        self.wheels[1].steer += (0.0 - self.wheels[1].steer) * blend;
    }

    pub fn contacts(
        &mut self,
        soil: &SoilField,
        throttle: f32,
        brake: f32,
        dt: f32,
    ) -> [BikeContact; 2] {
        let ori = self.orientation();
        let ang = ori * Vec3::new(self.pitch_rate, self.yaw_rate, self.roll_rate);
        let hps = hardpoints();
        let throttle = throttle.clamp(-1.0, 1.0);
        let brake = brake.clamp(0.0, 1.0);
        let mut out = [BikeContact::default(); 2];

        for i in 0..2 {
            let hp = hps[i];
            let local_c0 = Vec3::new(hp.x, hp.y - REST_LEN, hp.z);
            let wc0 = self.pos + ori * local_c0;
            let up = ori * Vec3::Y;
            let up_y = up.y.max(0.35);
            let sample = soil.sample(wc0.x, wc0.z);
            let z0 = sample.height - wc0.y + BIKE_RADIUS;
            let (compression, sinkage, normal) = if z0 <= 0.004 {
                ((self.wheels[i].compression - dt * 3.4).max(0.0), 0.0, 0.0)
            } else {
                solve_suspension(
                    z0,
                    up_y,
                    self.wheels[i].slip,
                    mech_for_sample(&sample),
                    sample.water,
                )
            };
            let wheel_center = wc0 + up * compression;
            let mut fwd = ori * (Quat::from_rotation_y(self.wheels[i].steer) * Vec3::Z);
            fwd.y = 0.0;
            if fwd.length_squared() < 1.0e-6 {
                fwd = Vec3::Z;
            }
            fwd = fwd.normalize();
            let right = Vec3::new(fwd.z, 0.0, -fwd.x);
            let contact_point = Vec3::new(
                wheel_center.x,
                sample.height - sinkage * 0.5,
                wheel_center.z,
            );
            let r = contact_point - self.pos;
            let v_point = self.vel + ang.cross(r);
            let v_long = v_point.x * fwd.x + v_point.z * fwd.z;
            let v_lat = v_point.x * right.x + v_point.z * right.z;
            let omega = self.wheels[i].omega;
            let slip_vel = omega * BIKE_RADIUS - v_long;
            let denom = v_long.abs().max((omega * BIKE_RADIUS).abs()).max(0.2);
            let slip = if v_long.abs() < 0.1 && (omega * BIKE_RADIUS).abs() < 0.1 {
                0.0
            } else {
                (slip_vel / denom).clamp(-1.5, 1.5)
            };
            let active = sinkage > 0.003 && normal > 1.0;
            let mut drive = if i == 1 {
                engine_torque(throttle, omega)
            } else {
                0.0
            };
            drive += brake_torque(brake, omega, v_long);
            let mut long_f = 0.0;
            let mut lat_f = 0.0;
            if active {
                let tm = mech_for_sample(&sample);
                let length = contact_length(BIKE_RADIUS, sinkage).max(0.05);
                let area = (BIKE_WIDTH * length).max(0.008);
                let sigma = (normal / area).clamp(0.0, 80_000.0);
                let j_long = slip.abs() * length;
                let slip_angle = if v_long.abs() < 0.2 && v_lat.abs() < 0.2 {
                    0.0
                } else {
                    v_lat.atan2(v_long.abs().max(0.2))
                };
                let j_lat = slip_angle.abs() * length * 0.8;
                let tau_long = janosi_hanamoto(tm.cohesion, sigma, tm.phi, j_long, tm.shear_k);
                let tau_lat = janosi_hanamoto(tm.cohesion, sigma, tm.phi, j_lat, tm.shear_k);
                let prop_long = tau_long * area * slip_vel.signum();
                let prop_lat = -tau_lat * area * v_lat.signum();
                let soil_w = smoothstep(0.025, 0.08, sinkage);
                let pace_long = pacejka_long(slip, normal);
                let pace_lat = -normal * 0.9 * slip_angle.clamp(-0.6, 0.6).signum()
                    * (1.0 - (-slip_angle.abs() / 0.14).exp());
                long_f = lerp(pace_long, prop_long, soil_w);
                lat_f = lerp(pace_lat, prop_lat, soil_w);
                let patch_len = length;
                let foot = soil.sample_patch(wheel_center, fwd, right, patch_len, BIKE_WIDTH);
                let coat = TreadCoat {
                    groove: self.wheels[i].groove,
                    smear: self.wheels[i].smear,
                    sidewall: self.wheels[i].sidewall,
                };
                let patch = contact_patch(
                    BIKE_RADIUS,
                    BIKE_WIDTH,
                    sinkage,
                    slip,
                    normal,
                    sample.kind(),
                    foot.wetness,
                    foot.loose,
                    foot.water,
                    coat,
                );
                let spun = ((slip.abs() - 0.75) / 0.20).clamp(0.0, 1.0);
                let coated = ((coat.smear - 0.72) / 0.20).clamp(0.0, 1.0);
                let patch_scale = lerp(1.0, patch.scale, spun.max(coated) * soil_w);
                long_f *= patch_scale;
                lat_f *= patch_scale;
                if v_long.abs() > 0.12 {
                    long_f -= v_long.signum() * rolling_resistance(normal, sample.water);
                }
                drive -= long_f * BIKE_RADIUS;
                let mut next = coat;
                let _step = step_tread(&mut next, slip, patch.squeeze, patch.pull, dt);
                self.wheels[i].groove = next.groove;
                self.wheels[i].smear = next.smear;
                self.wheels[i].sidewall = next.sidewall;
                let _ = tm;
            }
            let held = self.jump_hold > 0.0;
            out[i] = BikeContact {
                active: active && !held,
                compression: compression.clamp(0.0, TRAVEL),
                sinkage,
                normal: if active && !held { normal } else { 0.0 },
                long_f: if held { 0.0 } else { long_f },
                lat_f: if held { 0.0 } else { lat_f },
                wheel_torque: drive,
                slip,
                forward: fwd,
                right,
                wheel_center,
                contact_point,
                index: i,
            };
            let w = &mut self.wheels[i];
            w.compression = compression.clamp(0.0, TRAVEL);
            w.slip = slip;
            w.sinkage = sinkage;
            w.normal = normal;
            w.in_water = sample.water > 0.01 && active;
        }
        out
    }

    pub fn integrate(&mut self, contacts: &[BikeContact; 2], dt: f32) {
        let ori = self.orientation();
        let mut force = Vec3::new(0.0, -MASS * 9.81, 0.0);
        let mut torque_local = Vec3::ZERO;
        if self.jump_hold > 0.0 {
            self.jump_hold = (self.jump_hold - dt).max(0.0);
        }
        let mut planted = 0;
        for c in contacts {
            let f_world = Vec3::new(
                c.forward.x * c.long_f + c.right.x * c.lat_f,
                c.normal,
                c.forward.z * c.long_f + c.right.z * c.lat_f,
            );
            if c.active {
                planted += 1;
                force += f_world;
                let r = c.contact_point - self.pos;
                torque_local += ori.inverse() * r.cross(f_world);
            }
            let alpha = c.wheel_torque / WHEEL_INERTIA;
            let w = &mut self.wheels[c.index];
            w.omega = (w.omega + alpha * dt).clamp(-MAX_OMEGA, MAX_OMEGA);
            w.spin += w.omega * dt;
        }
        let airborne = planted == 0 || self.jump_hold > 0.0;
        if airborne {
            force.y += -40.0 * self.vel.y;
            torque_local.x -= 12.0 * self.pitch_rate;
            torque_local.z -= 18.0 * self.roll_rate;
        } else {
            force.y += -900.0 * self.vel.y;
            // A bike without a rider balance falls over. The spring is the rider.
            torque_local.z -= self.roll * 4_200.0;
            torque_local.z -= self.roll_rate * 180.0;
            torque_local.x -= self.pitch_rate * 90.0;
            if self.pitch.abs() > 0.45 {
                torque_local.x -= self.pitch.signum() * (self.pitch.abs() - 0.45) * 2_400.0;
            }
        }
        torque_local.y -= 40.0 * self.yaw_rate;

        self.vel += (force / MASS) * dt;
        if airborne {
            self.vel.y = self.vel.y.clamp(-14.0, 9.0);
        } else {
            self.vel.y = self.vel.y.clamp(-6.0, 6.0);
        }
        let hv = Vec3::new(self.vel.x, 0.0, self.vel.z);
        if hv.length() > 16.0 {
            let s = 16.0 / hv.length();
            self.vel.x *= s;
            self.vel.z *= s;
        }
        if !self.vel.is_finite() {
            self.vel = Vec3::ZERO;
        }
        self.pos += self.vel * dt;
        self.yaw_rate = (self.yaw_rate + torque_local.y / I_YAW * dt).clamp(-2.2, 2.2);
        self.yaw += self.yaw_rate * dt;
        if airborne {
            self.pitch_rate = (self.pitch_rate + torque_local.x / I_PITCH * dt).clamp(-8.0, 8.0);
            self.roll_rate = (self.roll_rate + torque_local.z / I_ROLL * dt).clamp(-6.0, 6.0);
            self.pitch = wrap_angle(self.pitch + self.pitch_rate * dt);
            self.roll = wrap_angle(self.roll + self.roll_rate * dt);
        } else {
            self.pitch_rate = (self.pitch_rate + torque_local.x / I_PITCH * dt).clamp(-2.4, 2.4);
            self.roll_rate = (self.roll_rate + torque_local.z / I_ROLL * dt).clamp(-2.0, 2.0);
            self.pitch = (self.pitch + self.pitch_rate * dt).clamp(-0.55, 0.7);
            self.roll = (self.roll + self.roll_rate * dt).clamp(-0.4, 0.4);
        }
        if !self.pos.is_finite() {
            self.pos.y = 0.6;
            self.vel = Vec3::ZERO;
        }
    }

    /// Same crest impulse as the buggy. Two planted wheels count as down.
    pub fn note_flight(
        &mut self,
        soil: &mut SoilField,
        particles: &mut Particles,
        planted: usize,
        dt: f32,
    ) {
        let x = self.pos.x;
        let z = self.pos.z;
        let on_crest = on_launch_crest(x, z);
        let approaching = (x + 6.6).abs() < 2.4
            && (-2.0..1.65).contains(&z)
            && planted >= 1
            && self.speed() > 2.0;
        if (on_crest && planted >= 1) || approaching {
            self.crest_armed = true;
        }
        let ground = soil.sample(x, z).height;
        let clearance = self.pos.y - ground;
        let flying = planted == 0 || self.jump_hold > 0.0;
        if flying && (clearance > 0.18 || self.jump_hold > 0.0) {
            self.air_time += dt;
        }
        let on_lip = on_crest && z > 1.55;
        if !self.did_launch && self.crest_armed && on_lip && self.speed() > 3.15 {
            let spd = self.speed();
            let vy = (spd * 0.85).clamp(4.2, 5.4);
            if self.vel.y < vy {
                self.vel.y = vy;
            }
            let rate = (4.8 + spd * 0.28).clamp(5.2, 6.6);
            if self.pitch_rate > -rate {
                self.pitch_rate = -rate;
            }
            self.roll_rate += 0.85;
            self.jump_hold = 0.78;
            self.crest_armed = false;
            self.did_launch = true;
        }
        let land = self.did_launch
            && !self.impact_latched
            && ((self.air_time > 0.40 && self.vel.y < -2.0 && clearance < 1.55)
                || (self.jump_hold <= 0.0 && planted >= 2 && self.air_time > 0.18 && clearance < 0.9));
        if land {
            let vy = self.vel.y.abs();
            let depth = (vy * 0.024).clamp(0.10, 0.22);
            let _dug = soil.stamp_impact(x, z, 1.15, depth);
            soil.shock_ring(x, z);
            let h = soil.sample(x, z).height;
            let _n = particles.impact_burst(Vec3::new(x, h + 0.12, z));
            stick_mud(self, soil);
            if self.vel.y < 0.0 {
                self.vel.y *= 0.22;
            }
            self.impact_latched = true;
        }
        if self.jump_hold <= 0.0 && planted >= 2 && self.vel.y > -0.7 && clearance < 0.9 {
            self.air_time = 0.0;
            self.did_launch = false;
            self.impact_latched = false;
            if !on_crest {
                self.crest_armed = false;
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct BikeContact {
    pub active: bool,
    pub compression: f32,
    pub sinkage: f32,
    pub normal: f32,
    pub long_f: f32,
    pub lat_f: f32,
    pub wheel_torque: f32,
    pub slip: f32,
    pub forward: Vec3,
    pub right: Vec3,
    pub wheel_center: Vec3,
    pub contact_point: Vec3,
    pub index: usize,
}

impl Default for BikeContact {
    fn default() -> Self {
        Self {
            active: false,
            compression: 0.0,
            sinkage: 0.0,
            normal: 0.0,
            long_f: 0.0,
            lat_f: 0.0,
            wheel_torque: 0.0,
            slip: 0.0,
            forward: Vec3::Z,
            right: Vec3::X,
            wheel_center: Vec3::ZERO,
            contact_point: Vec3::ZERO,
            index: 0,
        }
    }
}

fn stick_mud(bike: &mut Bike, soil: &SoilField) {
    let spots = [PANEL_BELLY, PANEL_NOSE, PANEL_SEAT];
    for panel in spots {
        let sample = soil.sample(bike.pos.x, bike.pos.z);
        let p = &mut bike.panels[panel];
        let add = (0.2 + 0.16 * sample.wetness) * stick_scale(sample.kind(), sample.wetness);
        p.mass = (p.mass + add).min(PANEL_MUD_CAP);
        p.wet = (p.wet + 0.7).min(1.0);
        p.streak = (p.streak + 0.4).min(1.0);
        p.drip = (p.mass / PANEL_MUD_CAP).clamp(0.0, 1.0);
    }
}

fn solve_suspension(z0: f32, up_y: f32, slip: f32, tm: Terramech, water: f32) -> (f32, f32, f32) {
    let k_eq = tm.k_eq / (1.0 + 0.18 * slip.abs());
    let spring = |c: f32| SUSP_PRELOAD + SUSP_K * c;
    let z_of = |c: f32| (z0 - up_y * c).max(0.0);
    let soil_n = |c: f32| bekker_load(z_of(c), BIKE_RADIUS, BIKE_WIDTH, k_eq, tm.n);
    let buoy = wheel_buoyancy(water, BIKE_RADIUS, BIKE_WIDTH);
    let support = |c: f32| soil_n(c) + buoy;
    let at_droop = soil_n(0.0);
    if at_droop <= 1.0 && buoy <= 1.0 {
        return (0.0, 0.0, 0.0);
    }
    if support(0.0) < spring(0.0) {
        return (0.0, z_of(0.0), support(0.0).min(8_000.0));
    }
    if support(TRAVEL) > spring(TRAVEL) {
        return (TRAVEL, z_of(TRAVEL), support(TRAVEL).min(8_000.0));
    }
    let mut lo = 0.0;
    let mut hi = TRAVEL;
    for _ in 0..12 {
        let mid = 0.5 * (lo + hi);
        if spring(mid) > support(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let c = 0.5 * (lo + hi);
    (c, z_of(c), spring(c).min(8_000.0))
}

fn engine_torque(throttle: f32, omega: f32) -> f32 {
    let fade = (1.0 - (omega.abs() / MAX_OMEGA).clamp(0.0, 1.0)).max(0.0);
    throttle * ENGINE_TORQUE * fade
}

fn brake_torque(brake: f32, omega: f32, v_long: f32) -> f32 {
    if brake < 0.01 {
        return 0.0;
    }
    let mag = brake * 180.0;
    if omega.abs() > 0.5 {
        -omega.signum() * mag
    } else if v_long.abs() > 0.25 {
        -v_long.signum() * mag * 0.3
    } else {
        -omega * 40.0 * brake
    }
}

fn pacejka_long(slip: f32, normal: f32) -> f32 {
    let s = slip.clamp(-1.0, 1.0);
    let peak = 1.05 * normal;
    let x = s.abs();
    let mag = if x < 0.16 {
        peak * (x / 0.16)
    } else {
        peak * 0.8
    };
    mag * s.signum()
}

fn wrap_angle(mut a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    while a > pi {
        a -= pi * 2.0;
    }
    while a < -pi {
        a += pi * 2.0;
    }
    a
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
