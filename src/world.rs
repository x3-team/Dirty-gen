//! Couples the buggy, the heightfield and splash.

use glam::Vec3;

use crate::particles::Particles;
use crate::soil::SoilField;
use crate::vehicle::{hardpoints, Vehicle, WHEEL_RADIUS};

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
}

#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    pub eye: Vec3,
    pub target: Vec3,
}

pub struct World {
    pub soil: SoilField,
    pub vehicle: Vehicle,
    pub particles: Particles,
    pub time: f32,
    pub telemetry: Telemetry,
    drip_acc: f32,
}

impl World {
    pub fn demo() -> Self {
        let mut world = Self {
            soil: SoilField::demo(),
            vehicle: Vehicle::new(),
            particles: Particles::new(),
            time: 0.0,
            telemetry: Telemetry {
                min_speed_in_soft: 99.0,
                ..Telemetry::default()
            },
            drip_acc: 0.0,
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
        world
    }

    /// Uniform flat pad. `wetness` 0 is firm, 1 is deep mud. No puddle.
    pub fn flat(wetness: f32) -> Self {
        let mut world = Self {
            soil: SoilField::flat(wetness),
            vehicle: Vehicle::new(),
            particles: Particles::new(),
            time: 0.0,
            telemetry: Telemetry {
                min_speed_in_soft: 99.0,
                ..Telemetry::default()
            },
            drip_acc: 0.0,
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

    fn substep(&mut self, dt: f32, input: &Input) {
        self.vehicle.update_steer(input.steer, dt);
        let contacts = self
            .vehicle
            .compute_contacts(&self.soil, input.throttle, input.brake, dt);

        for c in &contacts {
            if !c.active {
                continue;
            }
            let slip_stamp = c.slip;
            self.soil.stamp_wheel(
                c.wheel_center,
                WHEEL_RADIUS,
                crate::vehicle::WHEEL_WIDTH,
                c.forward,
                c.right,
                c.sinkage,
                slip_stamp,
                dt,
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
        }

        self.vehicle.integrate(&contacts, &self.soil, dt);
        self.vehicle.step_adhesion(&contacts, dt);

        let speed = self.vehicle.speed();
        for c in &contacts {
            if !c.active {
                continue;
            }
            if c.sample.water > 0.012 && speed > 0.35 {
                self.particles.spray(
                    c.contact_point,
                    c.forward,
                    c.right,
                    speed,
                    c.sample.water,
                );
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

    pub fn run_seconds(&mut self, seconds: f32, input_fn: impl Fn(f32) -> Input) {
        let dt = 1.0 / 60.0;
        let steps = (seconds / dt).round() as u32;
        for _ in 0..steps {
            let input = input_fn(self.time);
            self.step(dt, &input);
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

/// Idle, drive into the puddle, spin in the bog, reverse out.
pub fn demo_input(t: f32) -> Input {
    if t < 1.15 {
        Input::default()
    } else if t < 3.28 {
        // Into the puddle. Speed falls as the water deepens.
        Input {
            throttle: 1.0,
            brake: 0.0,
            steer: 0.0,
        }
    } else if t < 8.0 {
        Input {
            throttle: -1.0,
            brake: 0.0,
            steer: 0.0,
        }
    } else {
        Input {
            throttle: 0.0,
            brake: 1.0,
            steer: 0.0,
        }
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

pub fn demo_camera(t: f32, pos: Vec3, yaw: f32) -> CameraPose {
    let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = Vec3::new(forward.z, 0.0, -forward.x);
    if t >= 9.0 {
        // After the buggy has left the puddle. Low, behind the rear tires,
        // aimed at the dry cut beside the contact so the groove is the subject.
        let axle = pos - forward * 1.18;
        // Low and behind the outer bank, far enough that the tire is not the frame.
        let lip = axle + right * 1.00 - forward * 0.14;
        let eye = lip - forward * 1.7 + right * 0.55;
        return CameraPose {
            eye: Vec3::new(eye.x, 0.36, eye.z),
            target: lip + Vec3::Y * 0.01,
        };
    }
    let (dist, height, side, look_ahead, lift) = if t < 1.15 {
        (7.2, 2.15, 1.35, 2.6, 1.05)
    } else if t < 3.6 {
        // Low and off to the side so the shore and the sky glint read.
        (7.8, 1.02, -2.7, 2.8, 0.16)
    } else if t < 6.5 {
        (6.4, 1.65, 1.7, 1.3, 0.42)
    } else {
        (7.5, 1.95, -1.5, 1.5, 0.48)
    };
    let eye = pos - forward * dist + Vec3::Y * height + right * side;
    let target = pos + forward * look_ahead + Vec3::Y * lift;
    CameraPose { eye, target }
}
