//! Grass, twigs and wet clods stuck to the buggy. Pickup reads the tire and
//! the soil sample the adhesion step already produced. This module does not
//! change that mass, the soil solver, or the body mesh. Bits stay put, then
//! drop off one at a time.

use glam::{Mat4, Quat, Vec3};

use crate::soil::SoilField;
use crate::vehicle::{Contact, Vehicle};

pub const KIND_GRASS: u8 = 0;
pub const KIND_TWIG: u8 = 1;
pub const KIND_CLOD: u8 = 2;

pub const MOUNT_BODY: u8 = 254;
pub const MOUNT_FREE: u8 = 255;

const CAP: usize = 96;
const MAX_WHEEL: usize = 18;
const MAX_BODY: usize = 36;

#[derive(Clone, Copy, Debug)]
pub struct Bit {
    pub kind: u8,
    pub mount: u8,
    pub local: Vec3,
    pub axis: Vec3,
    pub side: Vec3,
    pub length: f32,
    pub twist: f32,
    pub hold: f32,
    pub alive: bool,
    pub seed: u32,
    pub pos: Vec3,
    pub vel: Vec3,
    pub grounded: bool,
}

impl Bit {
    /// World root, growth axis and width axis. Wheel bits spin with the tire.
    pub fn frame(&self, vehicle: &Vehicle) -> (Vec3, Vec3, Vec3) {
        if self.mount == MOUNT_FREE {
            let axis = self.axis.normalize_or_zero();
            let side = reorth(self.side, axis);
            return (self.pos, axis, side);
        }
        let body = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
        if self.mount == MOUNT_BODY {
            let pos = body.transform_point3(self.local);
            let axis = body.transform_vector3(self.axis).normalize_or_zero();
            let side = reorth(body.transform_vector3(self.side), axis);
            return (pos, axis, side);
        }
        let i = self.mount as usize;
        let center = vehicle.wheel_center_local(i);
        let wheel = vehicle.wheels[i];
        let spun = body
            * Mat4::from_translation(center)
            * Mat4::from_rotation_y(wheel.steer)
            * Mat4::from_rotation_x(wheel.spin);
        let pos = spun.transform_point3(self.local);
        let axis = spun.transform_vector3(self.axis).normalize_or_zero();
        let side = reorth(spun.transform_vector3(self.side), axis);
        (pos, axis, side)
    }
}

fn reorth(side: Vec3, axis: Vec3) -> Vec3 {
    let s = side - axis * side.dot(axis);
    if s.length_squared() < 1.0e-6 {
        let helper = if axis.y.abs() < 0.85 {
            Vec3::Y
        } else {
            Vec3::X
        };
        axis.cross(helper).normalize_or_zero()
    } else {
        s.normalize_or_zero()
    }
}

#[derive(Clone, Debug)]
pub struct Debris {
    bits: Vec<Bit>,
    rng: u32,
}

impl Debris {
    pub fn new() -> Self {
        Self {
            bits: Vec::with_capacity(CAP),
            rng: 0x00D3_B115,
        }
    }

    pub fn clear(&mut self) {
        for bit in &mut self.bits {
            bit.alive = false;
        }
    }

    pub fn bits(&self) -> &[Bit] {
        &self.bits
    }

    pub fn stuck_count(&self) -> usize {
        self.bits
            .iter()
            .filter(|b| b.alive && b.mount != MOUNT_FREE)
            .count()
    }

    pub fn falling_count(&self) -> usize {
        self.bits
            .iter()
            .filter(|b| b.alive && b.mount == MOUNT_FREE)
            .count()
    }

    pub fn kind_count(&self, kind: u8) -> usize {
        self.bits
            .iter()
            .filter(|b| b.alive && b.kind == kind)
            .count()
    }

    pub fn step(&mut self, vehicle: &Vehicle, contacts: &[Contact; 4], soil: &SoilField, dt: f32) {
        let n = self.bits.len();
        for i in 0..n {
            if !self.bits[i].alive {
                continue;
            }
            if self.bits[i].mount == MOUNT_FREE {
                self.fall(i, soil, dt);
            } else {
                self.age(i, vehicle, dt);
            }
        }
        for c in contacts.iter() {
            if !muddy(c, vehicle) {
                continue;
            }
            let mount = c.index as u8;
            if self.room() >= 7 && self.mounted(mount) + 7 <= MAX_WHEEL {
                let p = if self.mounted(mount) == 0 {
                    dt * 6.0
                } else {
                    dt * 2.2
                };
                if self.rnd() < p.min(0.85) {
                    let (origin, axis, side) = self.tire_anchor(c.index);
                    self.spawn_cluster(mount, origin, axis, side, false);
                }
            }
            if self.room() >= 7 && self.mounted(MOUNT_BODY) + 7 <= MAX_BODY {
                let p = if self.mounted(MOUNT_BODY) < 6 {
                    dt * 4.0
                } else {
                    dt * 0.85
                };
                if self.rnd() < p.min(0.75) {
                    let (origin, axis, side, hang) = self.body_anchor(c.index);
                    self.spawn_cluster(MOUNT_BODY, origin, axis, side, hang);
                }
            }
        }
    }

    fn room(&self) -> usize {
        CAP.saturating_sub(self.bits.iter().filter(|b| b.alive).count())
    }

    fn mounted(&self, mount: u8) -> usize {
        self.bits
            .iter()
            .filter(|b| b.alive && b.mount == mount)
            .count()
    }

    fn rnd(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }

    fn tire_anchor(&mut self, index: usize) -> (Vec3, Vec3, Vec3) {
        let outward = if index % 2 == 0 { -1.0 } else { 1.0 };
        let ang = self.rnd() * std::f32::consts::TAU;
        let radial = Vec3::new(0.0, ang.sin(), ang.cos());
        let tangent = Vec3::new(0.0, ang.cos(), -ang.sin());
        let x = if self.rnd() < 0.58 {
            outward * self.rnd() * 0.045
        } else {
            outward * (0.05 + self.rnd() * 0.09)
        };
        let r = 0.392;
        let local = Vec3::new(x, radial.y * r, radial.z * r) - radial * 0.012;
        // Most of the reference blades lie in the tread. A few stand out of it.
        let out = if self.rnd() < 0.64 {
            0.16 + self.rnd() * 0.12
        } else {
            0.55 + self.rnd() * 0.4
        };
        let along = if self.rnd() > 0.5 { 1.0 } else { -1.0 };
        let mut axis = radial * out + tangent * (1.0 - out) * along;
        axis += Vec3::X * outward * self.rnd() * 0.28;
        let axis = axis.normalize_or_zero();
        let side = reorth(tangent, axis);
        (local, axis, side)
    }

    fn body_anchor(&mut self, wheel: usize) -> (Vec3, Vec3, Vec3, bool) {
        let sign = if wheel % 2 == 0 { -1.0 } else { 1.0 };
        let pick = self.rnd();
        let (local, normal, hang) = if pick < 0.42 {
            let z = if wheel < 2 {
                0.05 + self.rnd() * 0.85
            } else {
                -0.05 - self.rnd() * 0.9
            };
            let x = sign * (0.05 + self.rnd() * 0.2);
            (
                Vec3::new(x, -0.175, z),
                Vec3::new(sign * 0.22, -1.0, 0.0).normalize_or_zero(),
                true,
            )
        } else if pick < 0.68 {
            let z = if wheel < 2 { 1.08 } else { -1.18 };
            (
                Vec3::new(
                    sign * (0.40 + self.rnd() * 0.06),
                    0.16,
                    z + (self.rnd() - 0.5) * 0.12,
                ),
                Vec3::new(sign * 0.55, 0.85, 0.0).normalize_or_zero(),
                false,
            )
        } else if pick < 0.86 {
            let z = if wheel < 2 { 1.02 } else { -1.12 };
            (
                Vec3::new(
                    sign * (0.48 + self.rnd() * 0.06),
                    -0.04,
                    z + (self.rnd() - 0.5) * 0.22,
                ),
                Vec3::new(sign * 0.4, -0.9, 0.0).normalize_or_zero(),
                true,
            )
        } else {
            let front = wheel < 2;
            let z = if front { 1.46 } else { -1.30 };
            (
                Vec3::new(sign * self.rnd() * 0.18, -0.04, z),
                Vec3::new(sign * 0.15, -0.25, if front { 1.0 } else { -1.0 }).normalize_or_zero(),
                true,
            )
        };
        let tangent = reorth(Vec3::X, normal);
        let out = if hang {
            0.62 + self.rnd() * 0.38
        } else {
            if self.rnd() < 0.5 {
                0.2
            } else {
                0.7
            }
        };
        let along = if self.rnd() > 0.5 { 1.0 } else { -1.0 };
        let axis = (normal * out + tangent * (1.0 - out) * along).normalize_or_zero();
        let side = reorth(tangent, axis);
        (local, axis, side, hang)
    }

    fn spawn_cluster(&mut self, mount: u8, origin: Vec3, axis: Vec3, side: Vec3, hang: bool) {
        let axis = axis.normalize_or_zero();
        let side = reorth(side, axis);
        let clod = 0.11 + self.rnd() * 0.07;
        self.push(KIND_CLOD, mount, origin, axis, side, clod);
        let clod2 = clod * (0.55 + self.rnd() * 0.25);
        let nudge = 0.04 + self.rnd() * 0.03;
        self.push(KIND_CLOD, mount, origin + side * nudge, axis, side, clod2);
        for i in 0..3 {
            let spread = (i as f32 - 1.0) * (0.035 + self.rnd() * 0.02);
            let tilt = (self.rnd() - 0.5) * 0.85;
            let a = (axis + side * tilt).normalize_or_zero();
            let len = if hang {
                0.36 + self.rnd() * 0.28
            } else {
                0.28 + self.rnd() * 0.22
            };
            let root = origin + side * spread + axis * (0.01 + self.rnd() * 0.02);
            self.push(KIND_GRASS, mount, root, a, side, len);
        }
        let twig_len = if hang {
            0.48 + self.rnd() * 0.36
        } else {
            0.36 + self.rnd() * 0.30
        };
        let ta = (axis + side * (self.rnd() - 0.5) * 0.6).normalize_or_zero();
        self.push(
            KIND_TWIG,
            mount,
            origin + side * 0.02 + axis * 0.015,
            ta,
            side,
            twig_len,
        );
        if self.rnd() > 0.4 {
            let tb = (axis + side * (self.rnd() - 0.5) * 1.1).normalize_or_zero();
            let short = twig_len * (0.55 + self.rnd() * 0.3);
            self.push(KIND_TWIG, mount, origin - side * 0.03, tb, side, short);
        }
    }

    fn push(&mut self, kind: u8, mount: u8, local: Vec3, axis: Vec3, side: Vec3, length: f32) {
        if self.room() == 0 {
            return;
        }
        let seed = self.rng;
        let _ = self.rnd();
        let bit = Bit {
            kind,
            mount,
            local,
            axis,
            side,
            length,
            twist: self.rnd() * std::f32::consts::TAU,
            hold: 0.40 + self.rnd() * 0.60,
            alive: true,
            seed,
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            grounded: false,
        };
        if let Some(slot) = self.bits.iter_mut().find(|b| !b.alive) {
            *slot = bit;
            return;
        }
        if self.bits.len() < CAP {
            self.bits.push(bit);
        }
    }

    fn age(&mut self, index: usize, vehicle: &Vehicle, dt: f32) {
        let mount = self.bits[index].mount;
        let kind = self.bits[index].kind;
        let speed = vehicle.speed();
        let spin = if mount < 4 {
            vehicle.wheels[mount as usize].omega.abs()
        } else {
            0.0
        };
        let kind_k = match kind {
            KIND_CLOD => 0.62,
            KIND_TWIG => 0.9,
            _ => 1.0,
        };
        let rate = (0.016 + speed * 0.0065 + spin * 0.0022) * kind_k;
        self.bits[index].hold -= rate * dt;
        if self.bits[index].hold <= 0.0 {
            self.release(index, vehicle);
        }
    }

    fn release(&mut self, index: usize, vehicle: &Vehicle) {
        let (pos, axis, side) = self.bits[index].frame(vehicle);
        let jx = self.rnd() - 0.5;
        let jz = self.rnd() - 0.5;
        let b = &mut self.bits[index];
        b.pos = pos;
        b.axis = axis;
        b.side = side;
        b.vel = vehicle.vel * 0.22 + Vec3::new(jx * 0.35, -0.12, jz * 0.35);
        b.mount = MOUNT_FREE;
        b.hold = 1.7;
        b.grounded = false;
    }

    fn fall(&mut self, index: usize, soil: &SoilField, dt: f32) {
        self.bits[index].hold -= dt;
        if self.bits[index].hold <= 0.0 {
            self.bits[index].alive = false;
            return;
        }
        if self.bits[index].grounded {
            return;
        }
        self.bits[index].vel.y -= 9.81 * dt;
        let drag = (1.0 - dt * 0.55).max(0.0);
        self.bits[index].vel *= drag;
        let vel = self.bits[index].vel;
        self.bits[index].pos += vel * dt;
        let q = Quat::from_axis_angle(Vec3::new(0.25, 1.0, 0.15).normalize(), dt * 1.5);
        let axis = q * self.bits[index].axis;
        let side = q * self.bits[index].side;
        self.bits[index].axis = axis;
        self.bits[index].side = side;
        let pos = self.bits[index].pos;
        let h = soil.sample(pos.x, pos.z).height;
        if pos.y <= h + 0.03 {
            self.bits[index].pos.y = h + 0.03;
            self.bits[index].vel = Vec3::ZERO;
            self.bits[index].grounded = true;
            let flat = Vec3::new(axis.x, 0.0, axis.z);
            let lie = if flat.length_squared() < 1.0e-4 {
                Vec3::X
            } else {
                flat.normalize()
            };
            self.bits[index].axis = lie;
            self.bits[index].side = Vec3::Y.cross(lie).normalize_or_zero();
        }
    }
}

fn muddy(c: &Contact, vehicle: &Vehicle) -> bool {
    if !c.active {
        return false;
    }
    let mud = vehicle.wheels[c.index].mud;
    c.sample.wetness > 0.50 || c.sample.water > 0.008 || mud > 0.22
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soil::SoilSample;

    fn wet_contact(index: usize) -> Contact {
        let mut c = Contact::default();
        c.active = true;
        c.index = index;
        c.sample = SoilSample {
            height: 0.0,
            wetness: 0.92,
            water: 0.04,
            loose: 0.04,
            darken: 0.2,
            roughness: 0.4,
            softness: 0.8,
            rut: 0.02,
        };
        c
    }

    #[test]
    fn wet_contact_sticks_grass_twigs_and_clods() {
        let mut debris = Debris::new();
        let mut vehicle = Vehicle::new();
        vehicle.wheels[0].mud = 0.4;
        let mut contacts = [Contact::default(); 4];
        contacts[0] = wet_contact(0);
        let soil = SoilField::flat(0.9);
        for _ in 0..(3.0 * 60.0) as i32 {
            debris.step(&vehicle, &contacts, &soil, 1.0 / 60.0);
        }
        assert!(
            debris.kind_count(KIND_GRASS) >= 3,
            "grass {}",
            debris.kind_count(KIND_GRASS)
        );
        assert!(
            debris.kind_count(KIND_TWIG) >= 1,
            "twigs {}",
            debris.kind_count(KIND_TWIG)
        );
        assert!(
            debris.kind_count(KIND_CLOD) >= 1,
            "clods {}",
            debris.kind_count(KIND_CLOD)
        );
        assert!(debris.mounted(0) > 0, "nothing on the tire");
        assert!(debris.mounted(MOUNT_BODY) > 0, "nothing on the chassis");
    }

    #[test]
    fn dry_contact_stays_clean() {
        let mut debris = Debris::new();
        let vehicle = Vehicle::new();
        let mut contacts = [Contact::default(); 4];
        contacts[0] = Contact::default();
        contacts[0].active = true;
        contacts[0].index = 0;
        contacts[0].sample.wetness = 0.12;
        let soil = SoilField::flat(0.12);
        for _ in 0..(2.0 * 60.0) as i32 {
            debris.step(&vehicle, &contacts, &soil, 1.0 / 60.0);
        }
        assert_eq!(debris.stuck_count(), 0);
    }

    #[test]
    fn shedding_is_slow() {
        let mut debris = Debris::new();
        let mut vehicle = Vehicle::new();
        vehicle.wheels[0].mud = 0.6;
        let mut contacts = [Contact::default(); 4];
        contacts[0] = wet_contact(0);
        let soil = SoilField::flat(0.2);
        for _ in 0..(2.5 * 60.0) as i32 {
            debris.step(&vehicle, &contacts, &soil, 1.0 / 60.0);
        }
        let stuck = debris.stuck_count();
        assert!(stuck >= 8, "setup stuck {stuck}");
        contacts[0].active = false;
        vehicle.vel.z = 6.0;
        vehicle.wheels[0].omega = 12.0;
        for _ in 0..(1.5 * 60.0) as i32 {
            debris.step(&vehicle, &contacts, &soil, 1.0 / 60.0);
        }
        let early = debris.stuck_count();
        assert!(
            early as f32 > stuck as f32 * 0.8,
            "shed too fast in 1.5s: {stuck} -> {early}"
        );
        for _ in 0..(24.0 * 60.0) as i32 {
            debris.step(&vehicle, &contacts, &soil, 1.0 / 60.0);
        }
        let late = debris.stuck_count();
        assert!(
            late * 2 < stuck,
            "still caked after a long roll: {stuck} -> {late}"
        );
    }
}
