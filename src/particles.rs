//! Short-lived splash, flung clods and drips. Ballistic, not a fluid solve.

use glam::Vec3;

use crate::soil::SoilField;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    /// 0 spray, 1 clod, 2 drip.
    pub kind: u8,
    pub alive: bool,
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
        if self.items.len() > 900 {
            if let Some(slot) = self.items.iter_mut().find(|p| !p.alive || p.life <= 0.0) {
                *slot = Particle {
                    pos,
                    vel,
                    life,
                    max_life: life,
                    size,
                    kind,
                    alive: true,
                };
                return;
            }
            return;
        }
        self.items.push(Particle {
            pos,
            vel,
            life,
            max_life: life,
            size,
            kind,
            alive: true,
        });
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
        self.spawn(pos, vel, 0.42, 0.2, 0);
    }

    pub fn clod(&mut self, origin: Vec3, forward: Vec3, right: Vec3, speed: f32) {
        let jx = (self.rnd() - 0.5) * 1.8;
        let up = 1.4 + self.rnd() * 1.6;
        let vel = forward * (0.4 + speed * 0.15) + right * jx + Vec3::Y * up;
        let life = 0.55 + self.rnd() * 0.4;
        let size = 0.10 + self.rnd() * 0.08;
        self.spawn(origin + Vec3::Y * 0.08, vel, life, size, 1);
    }

    /// Clumps and droplets born at the impact, in world space. They fall
    /// under the same gravity as the rest of the spray.
    pub fn impact_burst(&mut self, origin: Vec3) -> u32 {
        let mut n = 0u32;
        for i in 0..48 {
            let ang = self.rnd() * std::f32::consts::TAU;
            let speed = 3.4 + self.rnd() * 5.2;
            let up = 3.0 + self.rnd() * 4.6;
            let vel = Vec3::new(ang.cos() * speed, up, ang.sin() * speed);
            let clod = i < 30;
            let life = if clod {
                1.15 + self.rnd() * 0.8
            } else {
                0.85 + self.rnd() * 0.65
            };
            let size = if clod {
                0.22 + self.rnd() * 0.2
            } else {
                0.10 + self.rnd() * 0.08
            };
            let kind = if clod { 3 } else { 4 };
            let pos = origin
                + Vec3::new(
                    (self.rnd() - 0.5) * 0.35,
                    self.rnd() * 0.1,
                    (self.rnd() - 0.5) * 0.35,
                );
            self.spawn(pos, vel, life, size, kind);
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
            let drag = match p.kind {
                0 => 1.6,
                1 => 0.8,
                3 => 0.45,
                4 => 0.7,
                _ => 0.4,
            };
            p.vel.y -= 9.81 * dt;
            p.vel *= (1.0 - dt * drag).max(0.0);
            p.pos += p.vel * dt;
            let h = soil.sample(p.pos.x, p.pos.z).height;
            if p.pos.y < h + 0.02 {
                if let Some(idx) = soil.idx_at(p.pos.x, p.pos.z) {
                    match p.kind {
                        0 | 4 => {
                            soil.wetness[idx] = (soil.wetness[idx] + 0.03).min(1.0);
                            soil.darken[idx] = (soil.darken[idx] + 0.02).min(1.0);
                        }
                        1 | 2 | 3 => {
                            soil.darken[idx] = (soil.darken[idx] + 0.05).min(1.0);
                            soil.loose[idx] = (soil.loose[idx] + 0.004).min(0.16);
                        }
                        _ => {}
                    }
                }
                p.alive = false;
            }
        }
    }
}
