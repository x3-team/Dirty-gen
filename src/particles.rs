//! Short-lived splash. Pieces are ballistic solids: they arc, then rest in the
//! soil. The renderer turns them into soft wet clods, not stones or cards.

use glam::{EulerRot, Mat3, Quat, Vec3};

use crate::soil::SoilField;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    /// 0 tire droplet, 1 flung clod, 2 drip, 3 impact chunk, 4 impact droplet,
    /// 5 mud sheet. Kind 5 does not write the soil.
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
        let fwd = forward.normalize_or_zero();
        let side = right.normalize_or_zero();
        if fwd.length_squared() < 0.5 || side.length_squared() < 0.5 {
            return;
        }
        // One curtain per puddle patch. Four tires must not fill the particle list.
        let crowded = self
            .items
            .iter()
            .any(|p| p.alive && p.kind == 5 && (p.pos - origin).length() < 0.9);
        if crowded {
            return;
        }
        let kick = (0.55 + speed * 0.08).clamp(0.4, 1.4);
        let scale = (0.75 + water * 3.0).clamp(0.75, 1.25);
        for i in 0..5 {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            let pos = origin + side * s * (0.38 + self.rnd() * 0.16)
                - fwd * (0.12 + i as f32 * 0.07)
                + Vec3::Y * (0.16 + self.rnd() * 0.1);
            let vel = side * s * (0.45 + self.rnd() * 0.25) - fwd * kick * 0.35
                + Vec3::Y * (0.12 + self.rnd() * 0.16);
            let size = (0.22 + self.rnd() * 0.08) * scale;
            let life = 0.48 + self.rnd() * 0.07;
            self.spawn_sheet(pos, vel, life, size, sheet_rot(side * s));
        }
        for i in 0..4 {
            let pos = origin - fwd * (0.32 + i as f32 * 0.09)
                + side * (self.rnd() - 0.5) * 0.45
                + Vec3::Y * (0.14 + self.rnd() * 0.12);
            let vel = -fwd * kick
                + Vec3::Y * (0.08 + self.rnd() * 0.12)
                + side * (self.rnd() - 0.5) * 0.2;
            let size = (0.24 + self.rnd() * 0.08) * scale;
            let life = 0.5 + self.rnd() * 0.06;
            self.spawn_sheet(pos, vel, life, size, sheet_rot(-fwd));
        }
    }

    fn spawn_sheet(&mut self, pos: Vec3, vel: Vec3, life: f32, size: f32, rot: Quat) {
        self.spawn(pos, vel, life, size, 5);
        if let Some(p) = self.items.iter_mut().rev().find(|p| p.alive && p.kind == 5) {
            p.rot = rot;
            p.spin = Vec3::new(0.0, 0.4, 0.0);
        }
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
                5 => 0.7,
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
                5 => 0.20,
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
                            1 | 2 | 3 => {
                                soil.darken[idx] = (soil.darken[idx] + 0.05).min(1.0);
                                soil.loose[idx] = (soil.loose[idx] + 0.008).min(0.16);
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

fn sheet_rot(normal: Vec3) -> Quat {
    let z = normal.normalize_or_zero();
    let mut x = Vec3::Y.cross(z);
    if x.length_squared() < 1.0e-6 {
        x = Vec3::X;
    }
    x = x.normalize();
    let y = z.cross(x).normalize();
    Quat::from_mat3(&Mat3::from_cols(x, y, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sheet_spray_faces_side_and_rear_without_wetting_soil() {
        let mut soil = SoilField::flat(0.2);
        let before: f32 = soil.wetness.iter().copied().sum();
        let mut parts = Particles::new();
        let forward = Vec3::Z;
        let right = Vec3::X;
        parts.spray(Vec3::new(1.0, 0.4, 2.0), forward, right, 2.0, 0.2);
        let sheets: Vec<_> = parts.items.iter().filter(|p| p.kind == 5).collect();
        assert!(sheets.len() >= 8, "sheets {}", sheets.len());
        assert!(
            sheets.iter().any(|p| p.vel.dot(right).abs() > 0.2),
            "a side sheet should leave sideways"
        );
        assert!(
            sheets.iter().any(|p| p.vel.dot(forward) < -0.2),
            "a rear sheet should leave backward"
        );
        for _ in 0..200 {
            parts.update(&mut soil, 1.0 / 60.0);
        }
        let after: f32 = soil.wetness.iter().copied().sum();
        assert!(
            (after - before).abs() < 1.0e-4,
            "sheet spray wrote the soil"
        );
    }
}
