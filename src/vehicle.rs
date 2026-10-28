//! Off-road buggy: suspension, AWD torque, and a tire model that hands off
//! to terramechanics once the rut is deep.

use glam::{EulerRot, Quat, Vec3};

use crate::soil::{
    bekker_load, contact_length, contact_sink, grip_from_sink, janosi_hanamoto, on_launch_crest,
    sink_drag_coeff, sink_hold, terramech_at, SoilField, SoilSample,
};

pub const WHEEL_RADIUS: f32 = 0.38;
pub const WHEEL_WIDTH: f32 = 0.28;
pub const MASS: f32 = 640.0;
pub const REST_LEN: f32 = 0.50;
pub const SUSP_TRAVEL: f32 = 0.30;
pub const SUSP_K: f32 = 17_500.0;
pub const SUSP_PRELOAD: f32 = 280.0;
pub const HARD_Y: f32 = 0.22;
/// Lowest belly skin in body space. Frame-deep sink is this skin against the soil.
pub const BELLY_Y: f32 = -0.18;
pub const PANEL_COUNT: usize = 12;

pub const PANEL_CAGE: usize = 0;
pub const PANEL_NOSE: usize = 1;
pub const PANEL_LEFT: usize = 2;
pub const PANEL_RIGHT: usize = 3;
pub const PANEL_REAR: usize = 4;
pub const PANEL_BELLY: usize = 5;
pub const PANEL_FENDER_FL: usize = 6;
pub const PANEL_FENDER_FR: usize = 7;
pub const PANEL_FENDER_RL: usize = 8;
pub const PANEL_FENDER_RR: usize = 9;
pub const PANEL_ENGINE: usize = 10;
pub const PANEL_SEAT: usize = 11;

const I_YAW: f32 = 720.0;
const I_PITCH: f32 = 480.0;
const I_ROLL: f32 = 260.0;
const WHEEL_INERTIA: f32 = 2.6;
const MAX_STEER: f32 = 0.50;
const ENGINE_TORQUE: f32 = 500.0;
const MAX_OMEGA: f32 = 34.0;
const WHEEL_MUD_CAP: f32 = 1.5;
pub const PANEL_MUD_CAP: f32 = 0.85;

/// Hardpoints: FL, FR, RL, RR. Local space, Y up, Z forward.
pub fn hardpoints() -> [Vec3; 4] {
    [
        Vec3::new(-0.74, HARD_Y, 1.08),
        Vec3::new(0.74, HARD_Y, 1.08),
        Vec3::new(-0.74, HARD_Y, -1.18),
        Vec3::new(0.74, HARD_Y, -1.18),
    ]
}

#[derive(Clone, Copy, Debug)]
pub struct Wheel {
    pub omega: f32,
    pub spin: f32,
    pub steer: f32,
    pub compression: f32,
    pub mud: f32,
    pub slip: f32,
    pub sinkage: f32,
    pub normal: f32,
    pub in_water: bool,
    pub drive_force: f32,
}

impl Default for Wheel {
    fn default() -> Self {
        Self {
            omega: 0.0,
            spin: 0.0,
            steer: 0.0,
            compression: 0.08,
            mud: 0.0,
            slip: 0.0,
            sinkage: 0.0,
            normal: 0.0,
            in_water: false,
            drive_force: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PanelMud {
    pub mass: f32,
    pub drip: f32,
    pub streak: f32,
    pub wet: f32,
}

impl Default for PanelMud {
    fn default() -> Self {
        Self {
            mass: 0.0,
            drip: 0.0,
            streak: 0.0,
            wet: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Vehicle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub yaw_rate: f32,
    pub pitch_rate: f32,
    pub roll_rate: f32,
    pub wheels: [Wheel; 4],
    pub panels: [PanelMud; PANEL_COUNT],
    /// Seconds of free flight after the crest. A clipping tire must not erase the launch.
    pub jump_hold: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Contact {
    pub active: bool,
    pub index: usize,
    pub stuck: bool,
    pub compression: f32,
    pub sinkage: f32,
    pub normal: f32,
    pub chassis_long: f32,
    pub chassis_lat: f32,
    pub wheel_torque: f32,
    pub slip: f32,
    pub forward: Vec3,
    pub right: Vec3,
    pub wheel_center: Vec3,
    pub contact_point: Vec3,
    pub sample: SoilSample,
    pub speed: f32,
    pub fling: f32,
}

impl Vehicle {
    pub fn new() -> Self {
        Self {
            pos: Vec3::new(0.0, 0.6, -8.2),
            vel: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            yaw_rate: 0.0,
            pitch_rate: 0.0,
            roll_rate: 0.0,
            wheels: [Wheel::default(); 4],
            panels: [PanelMud::default(); PANEL_COUNT],
            jump_hold: 0.0,
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

    pub fn body_mud_mass(&self) -> f32 {
        self.panels.iter().map(|p| p.mass).sum()
    }

    pub fn wheel_mud_mass(&self) -> f32 {
        self.wheels.iter().map(|w| w.mud).sum()
    }

    pub fn clear_loose_mud(&mut self) {
        for w in &mut self.wheels {
            w.mud = 0.0;
        }
        self.panels = [PanelMud::default(); PANEL_COUNT];
    }

    pub fn wheel_center_local(&self, index: usize) -> Vec3 {
        let hp = hardpoints()[index];
        let c = self.wheels[index].compression;
        Vec3::new(hp.x, hp.y - REST_LEN + c, hp.z)
    }

    pub fn update_steer(&mut self, steer: f32, dt: f32) {
        let target = steer.clamp(-1.0, 1.0) * MAX_STEER;
        let blend = 1.0 - (-dt * 6.0).exp();
        self.wheels[0].steer += (target - self.wheels[0].steer) * blend;
        self.wheels[1].steer += (target - self.wheels[1].steer) * blend;
        self.wheels[2].steer += (0.0 - self.wheels[2].steer) * blend;
        self.wheels[3].steer += (0.0 - self.wheels[3].steer) * blend;
    }

    pub fn compute_contacts(
        &mut self,
        soil: &SoilField,
        throttle: f32,
        brake: f32,
        dt: f32,
    ) -> [Contact; 4] {
        let ori = self.orientation();
        let ang_world = ori * Vec3::new(self.pitch_rate, self.yaw_rate, self.roll_rate);
        let mut out = [Contact::default(); 4];
        let hps = hardpoints();
        let throttle = throttle.clamp(-1.0, 1.0);
        let brake = brake.clamp(0.0, 1.0);

        for i in 0..4 {
            let hp = hps[i];
            let local_c0 = Vec3::new(hp.x, hp.y - REST_LEN, hp.z);
            let wc0 = self.pos + ori * local_c0;
            let up = ori * Vec3::Y;
            let up_y = up.y.max(0.4);
            let sample = soil.sample(wc0.x, wc0.z);
            let z0 = sample.height - wc0.y + WHEEL_RADIUS;

            let (compression, sinkage, normal) = if z0 <= 0.004 {
                // Unloaded arm falls to droop, so the tire hangs once the buggy is airborne.
                ((self.wheels[i].compression - dt * 2.8).max(0.0), 0.0, 0.0)
            } else {
                solve_suspension(
                    z0,
                    up_y,
                    self.wheels[i].slip,
                    sample.softness,
                    sample.rut,
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
            let v_point = self.vel + ang_world.cross(r);
            let v_long = v_point.x * fwd.x + v_point.z * fwd.z;
            let v_lat = v_point.x * right.x + v_point.z * right.z;
            let omega = self.wheels[i].omega;
            let slip_vel = omega * WHEEL_RADIUS - v_long;
            let denom = v_long.abs().max((omega * WHEEL_RADIUS).abs()).max(0.25);
            let slip = if v_long.abs() < 0.12 && (omega * WHEEL_RADIUS).abs() < 0.12 {
                0.0
            } else {
                (slip_vel / denom).clamp(-1.5, 1.5)
            };

            let active = sinkage > 0.003 && normal > 1.0;
            let mut chassis_long = 0.0;
            let mut chassis_lat = 0.0;
            let mut wheel_torque = engine_torque(throttle, omega);
            wheel_torque += brake_torque(brake, omega, v_long);
            let mut stuck = false;
            let mut fling = 0.0;

            if active {
                let tm = terramech_at(sample.softness, sample.rut, sample.water);
                let length = contact_length(WHEEL_RADIUS, sinkage).max(0.05);
                let area = (WHEEL_WIDTH * length).max(0.01);
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
                let tau_max = (tm.cohesion + sigma * tm.phi.tan()).max(1.0);
                let mag = (tau_long * tau_long + tau_lat * tau_lat).sqrt();
                let scale = if mag > tau_max { tau_max / mag } else { 1.0 };
                let prop_long = tau_long * scale * area * slip_vel.signum();
                let prop_lat = -tau_lat * scale * area * v_lat.signum();

                // Firm ground still uses a light Pacejka-style blend so the buggy
                // steers when sinkage is only a few centimetres. Deep ruts ignore it.
                let soil_w = smoothstep(0.03, 0.09, sinkage);
                let pace_long = pacejka_long(slip, normal);
                let pace_lat = pacejka_lat(slip_angle, normal);
                // Burial, not a wet/dry flag. Ankle-deep keeps the shear.
                // Hub-deep cuts it so the tread spins. Frame-deep cannot pull.
                let depth = contact_sink(sinkage, sample.rut);
                let grip = grip_from_sink(depth);
                let mut blended_long = lerp(pace_long, prop_long, soil_w) * grip;
                let blended_lat = lerp(pace_lat, prop_lat, soil_w) * grip;
                // Climbing the lip of a rut. A hole is uphill in both directions,
                // so the centred slope cancels and has to be measured along the push.
                if blended_long.abs() > 1.0 {
                    let dir = blended_long.signum();
                    let reach = 0.36;
                    let ahead = soil.sample(
                        wheel_center.x + fwd.x * reach * dir,
                        wheel_center.z + fwd.z * reach * dir,
                    );
                    // A gentle bowl is not a lip. Only a sharp step (the rut wall)
                    // adds climb resistance, and it is capped so a rolling exit
                    // can still leave.
                    let rise = (ahead.height - sample.height - 0.03).max(0.0);
                    let climb = normal * (rise / reach).min(0.35);
                    blended_long -= dir * climb;
                }

                let speed_h = (v_long * v_long + v_lat * v_lat).sqrt();
                // A carved rut enlarges the plowing section. Fresh mud only drags;
                // a spun-in hole can exceed peak shear and hold the tire.
                let rut_w = (sample.rut / 0.055).clamp(0.0, 2.4);
                let f_yield = tm.yield_stress * area * (0.85 + 0.85 * rut_w);
                let f_visc = tm.viscosity * area * speed_h / sinkage.max(0.03);
                let prop = Vec3::new(blended_long, blended_lat, 0.0);
                let prop_mag = prop.x.hypot(prop.y);
                // At speed the full yield bogs the buggy. From rest only the
                // breakaway fraction has to be beaten, unless the rut is deep.
                let speed_gate = smoothstep(0.25, 1.2, speed_h);
                let breakaway = f_yield * (0.50 + 0.50 * speed_gate);
                let f_sink = sink_drag_coeff(depth) * speed_h;
                let hold = sink_hold(depth, normal);

                if speed_h < 0.12 {
                    if prop_mag <= breakaway + hold {
                        stuck = true;
                        chassis_long = 0.0;
                        chassis_lat = 0.0;
                    } else {
                        let excess = prop_mag - breakaway - hold;
                        let inv = excess / prop_mag.max(1.0);
                        chassis_long = prop.x * inv;
                        chassis_lat = prop.y * inv;
                    }
                } else {
                    let dir_l = v_long / speed_h;
                    let dir_a = v_lat / speed_h;
                    chassis_long = blended_long - dir_l * (breakaway + f_visc + f_sink);
                    chassis_lat = blended_lat - dir_a * (breakaway + f_visc + f_sink);
                    let rr = 0.025 * normal;
                    chassis_long -= v_long.signum() * rr;
                }

                // Shear reacts on the wheel even if Bingham holds the chassis still.
                wheel_torque -= blended_long * WHEEL_RADIUS;

                let peripheral = omega.abs() * WHEEL_RADIUS;
                fling = 0.62
                    * self.wheels[i].mud
                    * (peripheral / 5.5).clamp(0.0, 2.5)
                    * (0.2 + slip.abs());
            }

            out[i] = Contact {
                active,
                index: i,
                stuck,
                compression: compression.clamp(0.0, SUSP_TRAVEL),
                sinkage,
                normal: if active { normal } else { 0.0 },
                chassis_long,
                chassis_lat,
                wheel_torque,
                slip,
                forward: fwd,
                right,
                wheel_center,
                contact_point,
                sample,
                speed: self.speed(),
                fling,
            };
        }
        out
    }

    pub fn integrate(&mut self, contacts: &[Contact; 4], soil: &SoilField, dt: f32) {
        let ori = self.orientation();
        let mut force = Vec3::new(0.0, -MASS * 9.81, 0.0);
        let mut torque_local = Vec3::ZERO;
        let held = self.jump_hold > 0.0;
        if self.jump_hold > 0.0 {
            self.jump_hold = (self.jump_hold - dt).max(0.0);
        }

        for c in contacts.iter() {
            let f_world = Vec3::new(
                c.forward.x * c.chassis_long + c.right.x * c.chassis_lat,
                c.normal,
                c.forward.z * c.chassis_long + c.right.z * c.chassis_lat,
            );
            // The crest already set the flight. Contact springs would glue the car back down.
            if c.active && !held {
                force += f_world;
                let r = c.contact_point - self.pos;
                torque_local += ori.inverse() * r.cross(f_world);
            }
            let alpha = c.wheel_torque / WHEEL_INERTIA;
            let w = &mut self.wheels[c.index];
            w.omega = (w.omega + alpha * dt).clamp(-MAX_OMEGA, MAX_OMEGA);
            w.spin += w.omega * dt;
            w.compression = c.compression;
            w.slip = c.slip;
            w.sinkage = c.sinkage;
            w.normal = c.normal;
            w.in_water = c.sample.water > 0.01 && c.active;
            w.drive_force = c.chassis_long;
        }

        // Suspension damping from hardpoint vertical speed.
        let planted = contacts.iter().filter(|c| c.active).count();
        let fwd = self.forward();
        let here_h = soil.sample(self.pos.x, self.pos.z).height;
        let ahead_h = soil
            .sample(self.pos.x + fwd.x * 0.75, self.pos.z + fwd.z * 0.75)
            .height;
        let slope = (ahead_h - here_h) / 0.75;
        let climbing =
            !held && on_launch_crest(self.pos.x, self.pos.z) && slope > 0.16 && planted >= 2;
        let airborne = held || planted == 0;
        let tumbling = self.pitch.abs() > 0.72 || self.roll.abs() > 0.78;
        let ang_world = ori * Vec3::new(self.pitch_rate, self.yaw_rate, self.roll_rate);
        if !climbing && !airborne {
            for (i, hp) in hardpoints().into_iter().enumerate() {
                if !contacts[i].active {
                    continue;
                }
                let r = (self.pos + ori * hp) - self.pos;
                let v_y = (self.vel + ang_world.cross(r)).y;
                force.y += -2_600.0 * v_y;
            }
        }

        if airborne {
            force.y += -70.0 * self.vel.y;
            torque_local.x -= 28.0 * self.pitch_rate;
            torque_local.z -= 28.0 * self.roll_rate;
        } else if climbing {
            let want = self.speed() * slope / (1.0 + slope * slope).sqrt();
            let lift = (want - self.vel.y).max(0.0);
            force.y += lift * MASS * 7.0;
            force.y += -320.0 * self.vel.y;
            // Positive pitch is nose-down. The face wants the nose up.
            let target = -slope.atan();
            torque_local.x += (target - self.pitch) * 2_800.0;
            torque_local.x -= 70.0 * self.pitch_rate;
            torque_local.z -= 360.0 * self.roll_rate;
        } else if tumbling {
            force.y += -800.0 * self.vel.y;
            torque_local.x -= wrap_angle(self.pitch) * 3_400.0;
            torque_local.z -= wrap_angle(self.roll) * 3_400.0;
            torque_local.x -= 260.0 * self.pitch_rate;
            torque_local.z -= 260.0 * self.roll_rate;
        } else {
            force.y += -4_500.0 * self.vel.y;
            if self.roll.abs() > 0.42 {
                torque_local.z -= self.roll.signum() * (self.roll.abs() - 0.42) * 9_000.0;
            }
            if self.pitch.abs() > 0.38 {
                torque_local.x -= self.pitch.signum() * (self.pitch.abs() - 0.38) * 7_000.0;
            }
            torque_local.x -= 900.0 * self.pitch_rate;
            torque_local.z -= 750.0 * self.roll_rate;
        }
        torque_local.y -= 180.0 * self.yaw_rate;

        self.vel += (force / MASS) * dt;
        if airborne || climbing {
            self.vel.y = self.vel.y.clamp(-14.0, 9.0);
        } else {
            self.vel.y = self.vel.y.clamp(-7.0, 7.0);
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

        self.yaw_rate = (self.yaw_rate + torque_local.y / I_YAW * dt).clamp(-1.6, 1.6);
        self.yaw += self.yaw_rate * dt;
        if airborne {
            self.pitch_rate = (self.pitch_rate + torque_local.x / I_PITCH * dt).clamp(-8.0, 8.0);
            self.roll_rate = (self.roll_rate + torque_local.z / I_ROLL * dt).clamp(-6.0, 6.0);
            self.pitch = wrap_angle(self.pitch + self.pitch_rate * dt);
            self.roll = wrap_angle(self.roll + self.roll_rate * dt);
        } else if climbing || tumbling {
            self.pitch_rate = (self.pitch_rate + torque_local.x / I_PITCH * dt).clamp(-6.0, 6.0);
            self.roll_rate = (self.roll_rate + torque_local.z / I_ROLL * dt).clamp(-4.0, 4.0);
            self.pitch = wrap_angle(self.pitch + self.pitch_rate * dt);
            self.roll = wrap_angle(self.roll + self.roll_rate * dt);
        } else {
            self.pitch_rate = (self.pitch_rate + torque_local.x / I_PITCH * dt).clamp(-2.0, 2.0);
            self.roll_rate = (self.roll_rate + torque_local.z / I_ROLL * dt).clamp(-2.0, 2.0);
            self.pitch = (self.pitch + self.pitch_rate * dt).clamp(-0.5, 0.5);
            self.roll = (self.roll + self.roll_rate * dt).clamp(-0.55, 0.55);
        }

        if !held {
            self.collide_belly(soil, dt);
        }
        if !self.pos.is_finite() {
            self.pos.y = 0.5;
            self.vel = Vec3::ZERO;
        }
    }

    fn collide_belly(&mut self, soil: &SoilField, dt: f32) {
        let ori = self.orientation();
        // Clearance stays above a normal rut. Deep high-centering still drags.
        let points = [
            Vec3::new(-0.30, BELLY_Y, 0.9),
            Vec3::new(0.30, BELLY_Y, 0.9),
            Vec3::new(-0.30, BELLY_Y, -0.8),
            Vec3::new(0.30, BELLY_Y, -0.8),
        ];
        let mut pen_max = 0.0_f32;
        for p in points {
            let w = self.pos + ori * p;
            let h = soil.sample(w.x, w.z).height;
            pen_max = pen_max.max(h + 0.01 - w.y);
        }
        if pen_max > 0.0 {
            self.pos.y += pen_max.min(0.015);
            if self.vel.y < 0.0 {
                self.vel.y *= 0.5;
            }
            let decel = (pen_max * 16.0).min(4.0);
            let hv = Vec3::new(self.vel.x, 0.0, self.vel.z);
            let speed = hv.length();
            if speed > 0.05 {
                let scale = ((speed - decel * dt) / speed).max(0.0);
                self.vel.x *= scale;
                self.vel.z *= scale;
            }
        }
    }

    /// Adhesion ODE. Pickup grows with wet contact and agitation; spin flings
    /// mud onto panels; gravity drips it back off.
    pub fn step_adhesion(&mut self, contacts: &[Contact; 4], dt: f32) {
        for c in contacts.iter() {
            let w = &mut self.wheels[c.index];
            if c.active {
                let wet = c.sample.wetness;
                let loose = c.sample.loose;
                let pressure = (c.normal / 1_500.0).clamp(0.0, 2.5);
                let room = (1.0 - w.mud / WHEEL_MUD_CAP).clamp(0.0, 1.0);
                let agitate = (c.slip.abs() * 0.85 + (c.speed / 3.0).clamp(0.0, 1.0) * 0.55 + 0.05)
                    .clamp(0.0, 2.2);
                let pick = 1.35 * wet * (0.30 + loose * 8.0) * pressure * room * agitate;
                w.mud += pick * dt;
                if c.sample.water > 0.02 {
                    w.mud = (w.mud + 0.25 * c.sample.water * agitate * dt).min(WHEEL_MUD_CAP);
                }
            }
            let drip = 0.07 * w.mud * dt * (1.0 + c.speed * 0.08);
            w.mud = (w.mud - c.fling * dt - drip).clamp(0.0, WHEEL_MUD_CAP);
        }

        let mut incoming = [0.0_f32; PANEL_COUNT];
        for c in contacts.iter() {
            if c.fling <= 0.0 {
                continue;
            }
            let m = c.fling * dt;
            // Front wheels plaster the nose and front fenders; rears the deck.
            let weights: &[(usize, f32)] = match c.index {
                0 => &[
                    (PANEL_NOSE, 0.26),
                    (PANEL_LEFT, 0.24),
                    (PANEL_FENDER_FL, 0.32),
                    (PANEL_BELLY, 0.10),
                    (PANEL_SEAT, 0.08),
                ],
                1 => &[
                    (PANEL_NOSE, 0.26),
                    (PANEL_RIGHT, 0.24),
                    (PANEL_FENDER_FR, 0.32),
                    (PANEL_BELLY, 0.10),
                    (PANEL_SEAT, 0.08),
                ],
                2 => &[
                    (PANEL_REAR, 0.26),
                    (PANEL_LEFT, 0.18),
                    (PANEL_FENDER_RL, 0.30),
                    (PANEL_BELLY, 0.16),
                    (PANEL_ENGINE, 0.10),
                ],
                _ => &[
                    (PANEL_REAR, 0.26),
                    (PANEL_RIGHT, 0.18),
                    (PANEL_FENDER_RR, 0.30),
                    (PANEL_BELLY, 0.16),
                    (PANEL_ENGINE, 0.10),
                ],
            };
            for (panel, wgt) in weights {
                incoming[*panel] += m * wgt;
            }
            incoming[PANEL_CAGE] += m * 0.04;
        }

        let speed = self.speed();
        for (panel, add) in incoming.iter().enumerate() {
            let p = &mut self.panels[panel];
            let room = (1.0 - p.mass / PANEL_MUD_CAP).clamp(0.0, 1.0);
            p.mass = (p.mass + add * (0.35 + 0.65 * room)).min(PANEL_MUD_CAP * 1.05);
            let drip = 0.045 * p.mass * dt * (1.0 + p.wet);
            p.mass = (p.mass - drip).max(0.0);
            p.wet = (p.wet + add * 3.0).clamp(0.0, 1.0);
            p.wet += (0.15 - p.wet) * dt * 0.02;
            p.streak =
                (p.streak * (1.0 - dt * 0.15) + add * 6.0 + speed * dt * 0.15).clamp(0.0, 1.0);
            p.drip = (p.mass / PANEL_MUD_CAP).clamp(0.0, 1.0);
        }
    }
}

impl Default for Contact {
    fn default() -> Self {
        Self {
            active: false,
            index: 0,
            stuck: false,
            compression: 0.0,
            sinkage: 0.0,
            normal: 0.0,
            chassis_long: 0.0,
            chassis_lat: 0.0,
            wheel_torque: 0.0,
            slip: 0.0,
            forward: Vec3::Z,
            right: Vec3::X,
            wheel_center: Vec3::ZERO,
            contact_point: Vec3::ZERO,
            sample: SoilSample {
                height: 0.0,
                wetness: 0.0,
                water: 0.0,
                loose: 0.0,
                darken: 0.0,
                roughness: 0.8,
                softness: 0.0,
                rut: 0.0,
            },
            speed: 0.0,
            fling: 0.0,
        }
    }
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

fn solve_suspension(
    z0: f32,
    up_y: f32,
    slip: f32,
    softness: f32,
    rut: f32,
    water: f32,
) -> (f32, f32, f32) {
    let tm = terramech_at(softness, rut, water);
    // Slip-sinkage: a spinning tire sees a softer modulus and sinks further.
    let k_eq = tm.k_eq / (1.0 + 0.18 * slip.abs());
    let spring = |c: f32| SUSP_PRELOAD + SUSP_K * c;
    let z_of = |c: f32| (z0 - up_y * c).max(0.0);
    let soil_n = |c: f32| bekker_load(z_of(c), WHEEL_RADIUS, WHEEL_WIDTH, k_eq, tm.n);

    let at_droop = soil_n(0.0);
    if at_droop <= 1.0 {
        return (0.0, 0.0, 0.0);
    }
    // Droop stop: the ground is softer than the preloaded spring, so the arm
    // stays extended and only the soil force is passed into the chassis.
    if at_droop < spring(0.0) {
        return (0.0, z_of(0.0), at_droop.min(22_000.0));
    }
    // Bump stop: full compression and the soil is still harder than the spring.
    if soil_n(SUSP_TRAVEL) > spring(SUSP_TRAVEL) {
        let z = z_of(SUSP_TRAVEL);
        return (SUSP_TRAVEL, z, soil_n(SUSP_TRAVEL).min(22_000.0));
    }

    let mut lo = 0.0;
    let mut hi = SUSP_TRAVEL;
    for _ in 0..14 {
        let mid = 0.5 * (lo + hi);
        if spring(mid) > soil_n(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let c = 0.5 * (lo + hi);
    (c, z_of(c), spring(c).min(22_000.0))
}

fn engine_torque(throttle: f32, omega: f32) -> f32 {
    if throttle >= 0.0 {
        let fade = (1.0 - (omega / MAX_OMEGA).clamp(0.0, 1.0)).max(0.0);
        throttle * ENGINE_TORQUE * fade
    } else {
        let fade = (1.0 - ((-omega) / MAX_OMEGA).clamp(0.0, 1.0)).max(0.0);
        throttle * ENGINE_TORQUE * fade
    }
}

fn brake_torque(brake: f32, omega: f32, v_long: f32) -> f32 {
    if brake < 0.01 {
        return 0.0;
    }
    let mag = brake * 900.0;
    if omega.abs() > 0.6 {
        -omega.signum() * mag
    } else if v_long.abs() > 0.3 {
        -v_long.signum() * mag * 0.35
    } else {
        -omega * 80.0 * brake
    }
}

fn pacejka_long(slip: f32, normal: f32) -> f32 {
    let s = slip.clamp(-1.0, 1.0);
    // Smooth peak near |slip| = 0.15, then a sliding plateau.
    let peak = 1.15 * normal;
    let slide = 0.85 * normal;
    let x = s.abs();
    let mag = if x < 0.15 {
        peak * (x / 0.15)
    } else {
        let t = ((x - 0.15) / 0.5).clamp(0.0, 1.0);
        lerp(peak, slide, t)
    };
    mag * s.signum()
}

fn pacejka_lat(slip_angle: f32, normal: f32) -> f32 {
    let a = slip_angle.clamp(-0.7, 0.7);
    let peak = 1.05 * normal;
    let x = a.abs();
    let mag = peak * (1.0 - (-x / 0.12).exp());
    -mag * a.signum()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Vehicle {
    pub fn mean_softness(&self, contacts: &[Contact; 4]) -> f32 {
        let mut acc = 0.0;
        let mut n = 0.0;
        for c in contacts {
            if c.active {
                acc += c.sample.softness;
                n += 1.0;
            }
        }
        if n > 0.0 {
            acc / n
        } else {
            0.0
        }
    }
}
