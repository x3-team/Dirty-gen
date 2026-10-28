//! Short-lived splash. Pieces are ballistic solids: they arc, then rest in the
//! soil. The renderer turns them into soft wet clods, not stones or cards.
//!
//! Rut clods are the spoil the stamp withheld. They use the same chunk flight
//! as an impact lump, and the volume they carry is added to the heightfield
//! where they land.

use glam::{EulerRot, Quat, Vec3};

use crate::soil::{SoilField, ThrownClod};

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
    /// Cubic metres of rut spoil still in this chunk. Zero for spray.
    pub volume: f32,
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
            volume: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Particles {
    pub items: Vec<Particle>,
    rng: u32,
    /// Volume whose chunk was cleared before it landed. The next step puts it back.
    orphans: Vec<(Vec3, f32)>,
}

impl Particles {
    pub fn new() -> Self {
        Self {
            items: Vec::with_capacity(1024),
            rng: 0x51_17_u32,
            orphans: Vec::new(),
        }
    }

    pub fn clear(&mut self) {
        for p in self.items.drain(..) {
            if p.volume > 0.0 && !p.deposited {
                self.orphans.push((p.pos, p.volume));
            }
        }
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
        let _ = self.spawn_piece(pos, vel, life, size, kind, 0.0);
    }

    fn spawn_piece(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        life: f32,
        size: f32,
        kind: u8,
        volume: f32,
    ) -> bool {
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
            volume,
        };
        if self.items.len() > 900 {
            if let Some(idx) = self.items.iter().position(|p| !p.alive || p.life <= 0.0) {
                let old = self.items[idx];
                if old.volume > 0.0 && !old.deposited {
                    self.orphans.push((old.pos, old.volume));
                }
                self.items[idx] = piece;
                return true;
            }
            return false;
        }
        self.items.push(piece);
        true
    }

    /// Spoil queued by the rut stamp. Kind 3 is the existing chunk flight:
    /// low drag, a long arc, and the chunk mesh.
    fn launch_thrown(&mut self, soil: &mut SoilField) {
        let mut n = 0;
        while n < 4 {
            let Some(clod) = soil.pop_thrown() else {
                break;
            };
            if clod.volume <= 1.0e-8 {
                continue;
            }
            if !self.spawn_thrown(soil, &clod) {
                soil.return_thrown(clod);
                break;
            }
            n += 1;
        }
    }

    fn spawn_thrown(&mut self, soil: &SoilField, clod: &ThrownClod) -> bool {
        let mut right = clod.right.normalize_or_zero();
        let forward = clod.forward.normalize_or_zero();
        if right.length_squared() < 0.5 {
            right = Vec3::new(forward.z, 0.0, -forward.x).normalize_or_zero();
        }
        if right.length_squared() < 0.5 {
            right = Vec3::Z;
        }
        let side = if self.rnd() > 0.5 { 1.0 } else { -1.0 };
        let h = soil.sample(clod.origin.x, clod.origin.z).height;
        let pos = Vec3::new(clod.origin.x, h + 0.32, clod.origin.z)
            + right * side * (0.05 + self.rnd() * 0.06)
            + forward * (self.rnd() - 0.5) * 0.1;
        // Short of the next groove. The shoulder is about a metre out;
        // the other tyre sits further than that.
        let speed = 1.25 + self.rnd() * 0.30;
        let up = 2.55 + self.rnd() * 0.40;
        let vel = right * side * speed + forward * (0.15 + self.rnd() * 0.25) + Vec3::Y * up;
        let life = 1.75 + self.rnd() * 0.35;
        let radius = (clod.volume * 0.75 / std::f32::consts::PI).cbrt();
        let size = radius.clamp(0.18, 0.32);
        self.spawn_piece(pos, vel, life, size, 3, clod.volume)
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
        if !self.orphans.is_empty() {
            let pending = std::mem::take(&mut self.orphans);
            for (pos, volume) in pending {
                if volume > 1.0e-8 {
                    soil.land_volume(pos.x, pos.z, volume);
                }
            }
        }
        self.launch_thrown(soil);
        for p in &mut self.items {
            if !p.alive {
                continue;
            }
            p.life -= dt;
            if p.life <= 0.0 {
                if p.volume > 0.0 && !p.deposited {
                    soil.land_volume(p.pos.x, p.pos.z, p.volume);
                    p.volume = 0.0;
                    p.deposited = true;
                }
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
                    if p.volume > 0.0 {
                        soil.land_volume(p.pos.x, p.pos.z, p.volume);
                        p.volume = 0.0;
                    }
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
                let h = soil.sample(p.pos.x, p.pos.z).height;
                let sit = h + p.size * belly;
                p.grounded = true;
                p.pos.y = sit;
                p.vel = Vec3::ZERO;
                p.spin = Vec3::ZERO;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn height_sum(soil: &SoilField) -> f32 {
        soil.height.iter().sum()
    }

    #[test]
    fn rut_clods_fly_from_the_cut_and_raise_where_they_land() {
        let mut soil = SoilField::flat(0.35);
        let before = height_sum(&soil);
        let area = soil.cell * soil.cell;
        let depth = soil.roll_pass(Vec3::new(-1.2, 0.0, 0.0), Vec3::X, 4.2, 0.10, 0.05);
        let thrown = soil.thrown_volume();
        let cut = (before - height_sum(&soil)) * area;
        eprintln!("rut {depth:.4}  cut {cut:.4} m3  thrown {thrown:.4} m3");
        assert!(depth > 0.03, "rut did not cut, depth {depth:.4}");
        assert!(
            thrown > 0.008,
            "displaced volume did not become clods, thrown {thrown:.4}"
        );
        assert!(
            thrown > cut * 0.55,
            "clods are not the displaced volume: thrown {thrown:.4} cut {cut:.4}"
        );
        let pre = soil.height.clone();
        let mut particles = Particles::new();
        let mut saw_air = false;
        let mut grounded = 0u32;
        for _ in 0..220 {
            particles.update(&mut soil, 1.0 / 60.0);
            for p in &particles.items {
                if p.alive && p.volume > 0.0 && !p.grounded && p.pos.y > 0.45 {
                    saw_air = true;
                }
                if p.grounded && p.deposited {
                    grounded += 1;
                }
            }
        }
        let air: f32 = particles
            .items
            .iter()
            .map(|p| if p.alive { p.volume } else { 0.0 })
            .sum();
        let accounted = height_sum(&soil) * area + soil.thrown_volume() + air;
        let original = before * area;
        let mut peak_gain = 0.0_f32;
        let mut raised = 0usize;
        for (h, h0) in soil.height.iter().zip(pre.iter()) {
            let gain = h - h0;
            peak_gain = peak_gain.max(gain);
            if gain > 0.015 {
                raised += 1;
            }
        }
        eprintln!(
            "airborne seen {saw_air} grounded-frames {grounded}  peak gain {peak_gain:.4}  cells {raised}  volume {original:.4} -> {accounted:.4}  still air {air:.5} queued {:.5}",
            soil.thrown_volume()
        );
        assert!(saw_air, "clods never left the ground");
        assert!(grounded > 0, "clods never landed");
        assert!(
            peak_gain > 0.02,
            "landed clods did not raise the heightfield, peak gain {peak_gain:.4}"
        );
        assert!(raised >= 3, "landing raised only {raised} cells");
        assert!(
            (accounted - original).abs() < 2.0e-3,
            "volume was not conserved: {original:.5} -> {accounted:.5}"
        );
        assert!(
            air < 1.0e-4 && soil.thrown_volume() < 1.0e-4,
            "spoil still in the air after landing"
        );
        let floor = soil.rut_near(0.8, 0.0);
        assert!(
            floor > 0.03,
            "landed clods filled the rut back in, depth {floor:.4}"
        );
    }
}
