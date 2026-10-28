//! Short-lived splash. Pieces are ballistic solids: they arc, then rest in the
//! soil. The renderer turns them into soft wet clods, not stones or cards.

use glam::{EulerRot, Quat, Vec3};

use crate::soil::SoilField;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    /// 0 tire droplet, 1 flung clod, 2 drip, 3 impact chunk, 4 impact droplet.
    pub kind: u8,
    pub alive: bool,
    pub seed: u32,
    pub spin: Vec3,
    pub rot: Quat,
    /// Sitting on the soil after the arc. The mesh stays until `life` ends.
    pub grounded: bool,
    pub deposited: bool,
}

impl Default for Particle {
    fn default() -> Self {
        Self {
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            life: 0.0,
            max_life: 1.0,
            size: 0.08,
            kind: 0,
            alive: false,
            seed: 0,
            spin: Vec3::ZERO,
            rot: Quat::IDENTITY,
            grounded: false,
            deposited: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Particles {
    pub items: Vec<Particle>,
    rng: u32,
}

impl Particles {
    pub fn new() -> Self {
        Self {
            items: Vec::with_capacity(1024),
            rng: 0x51_17_u32,
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    pub fn alive_count(&self) -> usize {
        self.items
            .iter()
            .filter(|p| p.alive && p.life > 0.0)
            .count()
    }

    fn rnd(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }

    pub fn spawn(&mut self, pos: Vec3, vel: Vec3, life: f32, size: f32, kind: u8) {
        let seed = self.rng;
        let _ = self.rnd();
        let rate = if kind == 3 { 3.4 } else { 6.5 };
        let spin = Vec3::new(
            (self.rnd() - 0.5) * rate,
            (self.rnd() - 0.5) * rate,
            (self.rnd() - 0.5) * rate,
        );
        let rot = Quat::from_euler(
            EulerRot::YXZ,
            self.rnd() * std::f32::consts::TAU,
            (self.rnd() - 0.5) * 2.4,
            (self.rnd() - 0.5) * 2.4,
        );
        let piece = Particle {
            pos,
            vel,
            life,
            max_life: life,
            size,
            kind,
            alive: true,
            seed,
            spin,
            rot,
            grounded: false,
            deposited: false,
        };
        if self.items.len() > 900 {
            if let Some(slot) = self.items.iter_mut().find(|p| !p.alive || p.life <= 0.0) {
                *slot = piece;
            }
            return;
        }
        self.items.push(piece);
    }

    pub fn spray(&mut self, origin: Vec3, forward: Vec3, right: Vec3, speed: f32, water: f32) {
        let _ = water;
        // One droplet, and not on top of one that is already in the air.
        // A spawn every step became a pale cloud from the isometric camera.
        let side = if self.rnd() > 0.5 { 1.0 } else { -1.0 };
        let pos = origin - forward * (0.1 + self.rnd() * 0.16)
            + right * side * (0.3 + self.rnd() * 0.14)
            + Vec3::Y * (0.48 + self.rnd() * 0.18);
        let crowded = self
            .items
            .iter()
            .any(|p| p.alive && p.kind == 0 && (p.pos - pos).length() < 0.36);
        if crowded {
            return;
        }
        let kick = (0.4 + speed * 0.12).clamp(0.3, 1.15);
        let vel = -forward * kick + right * side * 0.3 + Vec3::Y * (0.9 + self.rnd() * 0.4);
        // Small solid droplet. A camera-facing quad read as an orange coin.
        self.spawn(pos, vel, 0.55, 0.09, 0);
    }

    pub fn clod(&mut self, origin: Vec3, forward: Vec3, right: Vec3, speed: f32) {
        let jx = (self.rnd() - 0.5) * 1.8;
        let up = 1.4 + self.rnd() * 1.6;
        let vel = forward * (0.4 + speed * 0.15) + right * jx + Vec3::Y * up;
        let life = 0.55 + self.rnd() * 0.4;
        let size = 0.10 + self.rnd() * 0.08;
        self.spawn(origin + Vec3::Y * 0.08, vel, life, size, 1);
    }

    /// A few large clods and a ring of smaller solid droplets. They leave the
    /// contact in an arc and are still alive when they meet the heightfield.
    pub fn impact_burst(&mut self, origin: Vec3) -> u32 {
        let mut n = 0u32;
        for i in 0..8 {
            let ang = i as f32 / 8.0 * std::f32::consts::TAU + (self.rnd() - 0.5) * 0.4;
            // A short arc that falls back into the crater, not a stone tossed across the lane.
            let speed = 0.55 + self.rnd() * 1.05;
            let up = 2.5 + self.rnd() * 1.5;
            let vel = Vec3::new(ang.cos() * speed, up, ang.sin() * speed);
            let size = 0.40 + self.rnd() * 0.22;
            let life = 2.5 + self.rnd() * 0.7;
            let pos = origin
                + Vec3::new(ang.cos(), 0.25 + self.rnd() * 0.15, ang.sin())
                    * (0.2 + self.rnd() * 0.35);
            self.spawn(pos, vel, life, size, 3);
            n += 1;
        }
        for _ in 0..16 {
            let ang = self.rnd() * std::f32::consts::TAU;
            let speed = 1.1 + self.rnd() * 2.4;
            let up = 2.0 + self.rnd() * 2.2;
            let vel = Vec3::new(ang.cos() * speed, up, ang.sin() * speed);
            let size = 0.11 + self.rnd() * 0.12;
            let life = 1.6 + self.rnd() * 0.8;
            let pos = origin
                + Vec3::new(
                    (self.rnd() - 0.5) * 0.5,
                    0.15 + self.rnd() * 0.2,
                    (self.rnd() - 0.5) * 0.5,
                );
            self.spawn(pos, vel, life, size, 4);
            n += 1;
        }
        n
    }

    pub fn drip(&mut self, origin: Vec3) {
        let vel = Vec3::new(
            (self.rnd() - 0.5) * 0.15,
            -0.2 - self.rnd() * 0.4,
            (self.rnd() - 0.5) * 0.15,
        );
        let size = 0.03 + self.rnd() * 0.02;
        self.spawn(origin, vel, 0.7, size, 2);
    }

    pub fn update(&mut self, soil: &mut SoilField, dt: f32) {
        for p in &mut self.items {
            if !p.alive {
                continue;
            }
            p.life -= dt;
            if p.life <= 0.0 {
                p.alive = false;
                continue;
            }
            if p.grounded {
                continue;
            }
            let drag = match p.kind {
                0 => 1.6,
                1 => 0.8,
                3 => 0.22,
                4 => 0.55,
                _ => 0.4,
            };
            p.vel.y -= 9.81 * dt;
            p.vel *= (1.0 - dt * drag).max(0.0);
            p.pos += p.vel * dt;
            p.rot = Quat::from_scaled_axis(p.spin * dt) * p.rot;
            let h = soil.sample(p.pos.x, p.pos.z).height;
            // Belly in the crater, the wet cap still above the soil.
            let belly = match p.kind {
                3 => 0.12,
                1 => 0.22,
                _ => 0.32,
            };
            let sit = h + p.size * belly;
            if p.pos.y <= sit {
                if !p.deposited {
                    if let Some(idx) = soil.idx_at(p.pos.x, p.pos.z) {
                        match p.kind {
                            0 | 4 => {
                                soil.wetness[idx] = (soil.wetness[idx] + 0.03).min(1.0);
                                soil.darken[idx] = (soil.darken[idx] + 0.02).min(1.0);
                            }
                            1 | 3 => {
                                let volume = p.size * p.size * p.size * 0.35;
                                soil.land_clod(p.pos.x, p.pos.z, volume);
                            }
                            2 => {
                                soil.darken[idx] = (soil.darken[idx] + 0.05).min(1.0);
                                soil.wetness[idx] = (soil.wetness[idx] + 0.1).min(1.0);
                            }
                            _ => {}
                        }
                    }
                    p.deposited = true;
                }
                p.grounded = true;
                p.pos.y = sit;
                p.vel = Vec3::ZERO;
                p.spin = Vec3::ZERO;
            }
        }
    }
}
