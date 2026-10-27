//! Volumetric exhaust smoke and the heat shimmer that rides with it.
//!
//! The body mesh is left alone. Each machine has a tailpipe point in body
//! space, and [`Exhaust::hook_tailpipe`] is the only place smoke is born.
//! Wisps are soft gray-blue volumes that curl up off the lip. The heat is a
//! refraction of the scene, not a flame and not a white streak.

use glam::{Quat, Vec3};

use crate::bike::Bike;
use crate::vehicle::Vehicle;

pub const PORT_BUGGY: u8 = 0;
pub const PORT_BIKE: u8 = 1;

const CAP: usize = 64;
const PLUMES: usize = 3;

/// Buggy outlets, body space. Just aft of the rear tires, above the tail lamps.
const BUGGY_TAILPIPE: [Vec3; 2] = [
    Vec3::new(-0.30, 0.36, -1.68),
    Vec3::new(0.30, 0.36, -1.68),
];

/// Bike outlet, body space. Right side, just clear of the rear tire.
const BIKE_TAILPIPE: Vec3 = Vec3::new(0.18, 0.28, -1.08);

#[derive(Clone, Copy, Debug)]
pub struct Plume {
    pub origin: Vec3,
    pub axis: Vec3,
    pub length: f32,
    pub radius0: f32,
    pub radius1: f32,
    pub strength: f32,
}

impl Plume {
    fn quiet() -> Self {
        Self {
            origin: Vec3::ZERO,
            axis: Vec3::new(0.0, 0.2, -1.0),
            length: 0.0,
            radius0: 0.05,
            radius1: 0.12,
            strength: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Puff {
    pos: Vec3,
    vel: Vec3,
    side: Vec3,
    /// Horizontal direction aft of the pipe. Keeps the volume behind the lip.
    drift: Vec3,
    /// World velocity of the pipe when the wisp was born, so the plume
    /// stays with the machine instead of dropping a trail of dashes.
    carry: Vec3,
    life: f32,
    max_life: f32,
    size: f32,
    seed: f32,
    port: u8,
    alive: bool,
}

/// One frame of a puff, already shaded. The renderer turns these into wisps.
#[derive(Clone, Copy, Debug)]
pub struct SmokePuff {
    pub pos: Vec3,
    pub vel: Vec3,
    pub size: f32,
    pub alpha: f32,
    pub color: Vec3,
    pub seed: f32,
}

/// Where a pipe dumps its exhaust, in world space, plus the way the gas leaves.
pub struct Tailpipe {
    port: u8,
    outlets: [Vec3; 2],
    n: u8,
    back: Vec3,
    side: Vec3,
    rpm: f32,
}

impl Tailpipe {
    pub fn buggy(vehicle: &Vehicle) -> Self {
        let rpm = vehicle.wheels.iter().map(|w| w.omega.abs()).sum::<f32>() * 0.25;
        let (back, side) = basis(vehicle.orientation());
        let ori = vehicle.orientation();
        Self {
            port: PORT_BUGGY,
            outlets: [
                vehicle.pos + ori * BUGGY_TAILPIPE[0],
                vehicle.pos + ori * BUGGY_TAILPIPE[1],
            ],
            n: 2,
            back,
            side,
            rpm,
        }
    }

    pub fn bike(bike: &Bike) -> Self {
        let rpm = bike.wheels.iter().map(|w| w.omega.abs()).sum::<f32>() * 0.5;
        let (back, side) = basis(bike.orientation());
        Self {
            port: PORT_BIKE,
            outlets: [bike.pos + bike.orientation() * BIKE_TAILPIPE, Vec3::ZERO],
            n: 1,
            back,
            side,
            rpm,
        }
    }

    pub fn back(&self) -> Vec3 {
        self.back
    }

    pub fn outlets(&self) -> &[Vec3] {
        &self.outlets[..self.n as usize]
    }
}

pub struct Exhaust {
    puffs: Vec<Puff>,
    plumes: [Plume; PLUMES],
    acc: [f32; PLUMES],
    prev_outlet: [Vec3; PLUMES],
    primed: [bool; PLUMES],
    rng: u32,
    time: f32,
}

impl Exhaust {
    pub fn new() -> Self {
        Self {
            puffs: Vec::with_capacity(CAP),
            plumes: [Plume::quiet(); PLUMES],
            acc: [0.0; PLUMES],
            prev_outlet: [Vec3::ZERO; PLUMES],
            primed: [false; PLUMES],
            rng: 0xE4A1_5C37,
            time: 0.0,
        }
    }

    pub fn clear(&mut self) {
        self.puffs.clear();
        self.plumes = [Plume::quiet(); PLUMES];
        self.acc = [0.0; PLUMES];
        self.prev_outlet = [Vec3::ZERO; PLUMES];
        self.primed = [false; PLUMES];
        self.time = 0.0;
    }

    pub fn alive_count(&self) -> usize {
        self.puffs.iter().filter(|p| p.alive && p.life > 0.0).count()
    }

    pub fn alive_on(&self, port: u8) -> usize {
        self.puffs
            .iter()
            .filter(|p| p.alive && p.life > 0.0 && p.port == port)
            .count()
    }

    pub fn plumes(&self) -> [Plume; PLUMES] {
        self.plumes
    }

    pub fn clock(&self) -> f32 {
        self.time
    }

    /// Spawn from this pipe. Call it once per vehicle step; [`sim`](Self::sim)
    /// is what carries the wisps. The buggy owns plume slots 0 and 1, the bike slot 2.
    pub fn hook_tailpipe(&mut self, pipe: Tailpipe, throttle: f32, dt: f32) {
        let throttle = throttle.abs().clamp(0.0, 1.0);
        let rpm_n = (pipe.rpm / 40.0).clamp(0.0, 1.0);
        let base = if pipe.port == PORT_BIKE { 2 } else { 0 };
        for i in 0..pipe.n as usize {
            let slot = base + i;
            self.write_plume(slot, pipe.outlets[i], pipe.back, throttle, rpm_n, pipe.port);
            self.emit(
                slot,
                pipe.outlets[i],
                pipe.back,
                pipe.side,
                throttle,
                rpm_n,
                pipe.port,
                dt,
            );
        }
    }

    pub fn sim(&mut self, dt: f32) {
        let dt = dt.clamp(0.0, 0.05);
        if dt <= 0.0 {
            return;
        }
        self.time += dt;
        for p in &mut self.puffs {
            if !p.alive {
                continue;
            }
            p.life -= dt;
            if p.life <= 0.0 {
                p.alive = false;
                continue;
            }
            let age = (1.0 - p.life / p.max_life).clamp(0.0, 1.0);
            let phase = self.time * 1.35 + p.seed * 12.0;
            let curl = phase.sin();
            let curl2 = (phase * 1.6 + 1.2).cos();
            // Stay with the pipe, then curl and climb off it.
            let relax = 1.0 - (-2.4 * dt).exp();
            p.vel.x += (p.carry.x - p.vel.x) * relax;
            p.vel.z += (p.carry.z - p.vel.z) * relax;
            p.vel += p.side * (curl * (0.30 + age * 0.55) + curl2 * 0.08) * dt;
            p.vel += Vec3::Y * (0.42 + age * 0.28) * dt;
            p.vel += p.drift * (0.28 + (1.0 - age) * 0.12) * dt;
            p.vel.y *= 1.0 - 0.28 * dt;
            p.pos += p.vel * dt;
        }
    }

    pub fn fill_smoke(&self, dst: &mut Vec<SmokePuff>) {
        dst.clear();
        for p in &self.puffs {
            if !p.alive || p.life <= 0.0 {
                continue;
            }
            let (color, alpha, size) = shade(p);
            dst.push(SmokePuff {
                pos: p.pos,
                vel: p.vel,
                size,
                alpha,
                color,
                seed: p.seed,
            });
        }
    }

    /// Share of living puffs on `port` that still sit aft of `origin`.
    pub fn mostly_aft(&self, port: u8, origin: Vec3, aft: Vec3, fraction: f32) -> bool {
        let mut n = 0u32;
        let mut ok = 0u32;
        for p in &self.puffs {
            if !p.alive || p.life <= 0.0 || p.port != port {
                continue;
            }
            n += 1;
            if (p.pos - origin).dot(aft) > -0.12 {
                ok += 1;
            }
        }
        n > 0 && ok as f32 / n as f32 >= fraction
    }

    fn write_plume(
        &mut self,
        slot: usize,
        outlet: Vec3,
        back: Vec3,
        throttle: f32,
        rpm_n: f32,
        port: u8,
    ) {
        let bike = port == PORT_BIKE;
        // Hot air leans out over the dirt and climbs with the wisps, so the
        // refraction crosses the lip, the bumper edge, and the ground.
        let raw = back * 1.05 + Vec3::Y * 0.58;
        let axis = if raw.length_squared() < 1.0e-6 {
            Vec3::Y
        } else {
            raw.normalize()
        };
        let length = if bike {
            0.40 + throttle * 0.20
        } else {
            0.58 + throttle * 0.30
        };
        let radius0 = if bike { 0.06 } else { 0.085 };
        self.plumes[slot] = Plume {
            origin: outlet,
            axis,
            length,
            radius0,
            radius1: radius0 + length * 0.48,
            strength: (0.5 + throttle * 0.36 + rpm_n * 0.16).clamp(0.0, 1.0),
        };
    }

    fn emit(
        &mut self,
        slot: usize,
        outlet: Vec3,
        back: Vec3,
        side: Vec3,
        throttle: f32,
        rpm_n: f32,
        port: u8,
        dt: f32,
    ) {
        let bike = port == PORT_BIKE;
        let rate = if bike { 13.0 } else { 16.0 } + throttle * 5.0 + rpm_n * 2.0;
        self.acc[slot] += dt.max(0.0) * rate;
        let mut drift = Vec3::new(back.x, 0.0, back.z);
        if drift.length_squared() < 1.0e-6 {
            drift = Vec3::new(0.0, 0.0, -1.0);
        } else {
            drift = drift.normalize();
        }
        let carry = self.pipe_carry(slot, outlet, dt);
        let mut born = 0;
        while self.acc[slot] >= 1.0 && born < 4 {
            self.acc[slot] -= 1.0;
            born += 1;
            let ring = self.rnd() * std::f32::consts::TAU;
            let rad = 0.015 + self.rnd() * if bike { 0.03 } else { 0.045 };
            let base = if bike { 0.085 } else { 0.12 };
            let life = 0.85 + self.rnd() * 0.30;
            let puff = Puff {
                pos: outlet
                    + back * (0.06 + self.rnd() * 0.08)
                    + side * ring.cos() * rad
                    + Vec3::Y * (0.05 + self.rnd() * 0.08),
                vel: carry
                    + back * (0.10 + throttle * 0.06 + self.rnd() * 0.04)
                    + side * (self.rnd() - 0.5) * 0.08
                    + Vec3::Y * (0.18 + self.rnd() * 0.14),
                side,
                drift,
                carry,
                life,
                max_life: life,
                size: base * (0.82 + self.rnd() * 0.55),
                seed: self.rnd(),
                port,
                alive: true,
            };
            self.insert(puff);
        }
    }

    fn pipe_carry(&mut self, slot: usize, outlet: Vec3, dt: f32) -> Vec3 {
        let prev = self.prev_outlet[slot];
        let primed = self.primed[slot];
        self.prev_outlet[slot] = outlet;
        self.primed[slot] = true;
        if !primed || dt <= 1.0e-4 {
            return Vec3::ZERO;
        }
        let mut v = (outlet - prev) / dt;
        let speed = v.length();
        if speed > 18.0 {
            v *= 18.0 / speed;
        }
        v.y *= 0.25;
        v
    }

    fn insert(&mut self, puff: Puff) {
        if let Some(slot) = self.puffs.iter_mut().find(|p| !p.alive || p.life <= 0.0) {
            *slot = puff;
            return;
        }
        if self.puffs.len() < CAP {
            self.puffs.push(puff);
            return;
        }
        if let Some(slot) = self
            .puffs
            .iter_mut()
            .min_by(|a, b| a.life.total_cmp(&b.life))
        {
            *slot = puff;
        }
    }

    fn rnd(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }
}

fn basis(ori: Quat) -> (Vec3, Vec3) {
    let back = (ori * Vec3::new(0.0, 0.28, -1.0)).normalize();
    let mut side = Vec3::new(back.z, 0.0, -back.x);
    if side.length_squared() < 1.0e-4 {
        side = Vec3::X;
    } else {
        side = side.normalize();
    }
    (back, side)
}

fn shade(p: &Puff) -> (Vec3, f32, f32) {
    let age = (1.0 - p.life / p.max_life).clamp(0.0, 1.0);
    // Light gray-blue. Fresh gas is bluer; it stays pale as it thins, and it
    // does not climb to white.
    let cool = (p.seed - 0.5) * 0.04;
    let fresh = Vec3::new(0.64, 0.71, 0.82 + cool);
    let pale = Vec3::new(0.73, 0.79, 0.88);
    let color = fresh.lerp(pale, age);
    let thin = if p.port == PORT_BIKE { 0.88 } else { 1.0 };
    // Low enough that a few overlapping wisps stay translucent.
    let alpha = (0.18 + 0.10 * (1.0 - age)) * thin;
    let size = p.size * (1.0 + age * 1.25);
    (color, alpha, size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{Input, World};

    fn straight() -> Input {
        Input {
            throttle: 1.0,
            brake: 0.0,
            steer: 0.0,
        }
    }

    #[test]
    fn buggy_tailpipe_smokes_aft_and_stays_cool() {
        let mut world = World::flat(0.12);
        world.run_seconds(1.1, |_| straight());
        assert!(
            world.exhaust.alive_on(PORT_BUGGY) > 8,
            "buggy tailpipes should be putting out wisps"
        );
        assert_eq!(world.exhaust.alive_on(PORT_BIKE), 0);
        let pipe = Tailpipe::buggy(&world.vehicle);
        let aft = Vec3::new(pipe.back().x, 0.0, pipe.back().z).normalize();
        let mid = (pipe.outlets()[0] + pipe.outlets()[1]) * 0.5;
        assert!(world.exhaust.mostly_aft(PORT_BUGGY, mid, aft, 0.8));
        let plumes = world.exhaust.plumes();
        assert!((plumes[0].origin - pipe.outlets()[0]).length() < 1.0e-3);
        assert!((plumes[1].origin - pipe.outlets()[1]).length() < 1.0e-3);
        assert!(plumes[0].strength > 0.45 && plumes[1].strength > 0.45);
        assert!(plumes[2].strength == 0.0);
        let mut smoke = Vec::new();
        world.exhaust.fill_smoke(&mut smoke);
        assert!(!smoke.is_empty());
        let mut rise = 0.0f32;
        let mut lateral = 0.0f32;
        for puff in &smoke {
            assert!(
                puff.color.x <= puff.color.z + 0.04,
                "smoke went hot: {:?}",
                puff.color
            );
            assert!(
                puff.color.z > puff.color.x,
                "smoke should stay gray-blue: {:?}",
                puff.color
            );
            assert!(puff.color.x < 0.75, "smoke went white: {:?}", puff.color);
            assert!(puff.alpha < 0.4, "wisp is too thick");
            assert!(puff.alpha > 0.05, "wisp faded out");
            assert!(puff.color.x > 0.45 && puff.color.y > 0.5);
            assert!(puff.size > 0.08, "wisp is too small to read as a volume");
            let rel = puff.pos - mid;
            rise = rise.max(rel.y);
            let along = rel.dot(aft);
            lateral = lateral.max((rel - aft * along).length());
        }
        assert!(rise > 0.08, "smoke should climb off the pipe, rose {rise}");
        assert!(
            lateral > 0.03,
            "smoke should fill a volume, lateral {lateral}"
        );
        assert!(plumes[0].axis.y > 0.35, "heat column should rise");
    }

    #[test]
    fn bike_tailpipe_smokes_too() {
        let mut world = World::flat(0.12);
        let idle = Input::default();
        for _ in 0..70 {
            world.step_bike(1.0 / 60.0, &straight());
            world.step(1.0 / 60.0, &idle);
        }
        assert!(world.exhaust.alive_on(PORT_BIKE) > 4);
        let pipe = Tailpipe::bike(&world.bike);
        let plumes = world.exhaust.plumes();
        assert!((plumes[2].origin - pipe.outlets()[0]).length() < 1.0e-3);
        assert!(plumes[2].strength > 0.45);
        let aft = Vec3::new(pipe.back().x, 0.0, pipe.back().z).normalize();
        assert!(world.exhaust.mostly_aft(PORT_BIKE, pipe.outlets()[0], aft, 0.75));
    }
}
