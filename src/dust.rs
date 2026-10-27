//! Tan dust kicked up when a tire rolls off mud onto drier ground.
//!
//! The plume is a low, wide cloud that stays behind the tread: ochre and
//! dense at the tire, pale sand as it thins. It does not wet the soil and it
//! does not change the mud spray.

use glam::Vec3;

/// One soft puff. `wide` stretches it along the camera's horizon so the cloud
/// reads as a trail from the side and a fan from the rear.
#[derive(Clone, Copy, Debug)]
pub struct DustMote {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    pub grow: f32,
    pub alpha: f32,
    pub rgb: [f32; 3],
    pub wide: f32,
    pub floor: f32,
    pub alive: bool,
}

#[derive(Clone, Debug)]
pub struct Dust {
    pub motes: Vec<DustMote>,
    recent: [f32; 4],
    trail: [f32; 4],
    cool: [f32; 4],
    armed: [bool; 4],
    rng: u32,
}

impl Dust {
    pub fn new() -> Self {
        Self {
            motes: Vec::with_capacity(256),
            recent: [0.0; 4],
            trail: [0.0; 4],
            cool: [0.0; 4],
            armed: [false; 4],
            rng: 0xD0_57_u32,
        }
    }

    pub fn clear(&mut self) {
        self.motes.clear();
        self.recent = [0.0; 4];
        self.trail = [0.0; 4];
        self.cool = [0.0; 4];
        self.armed = [false; 4];
    }

    pub fn alive_count(&self) -> usize {
        self.motes
            .iter()
            .filter(|m| m.alive && m.life > 0.0)
            .count()
    }

    fn rnd(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }

    /// One active wheel. `wetness` and `water` are the soil under the tread.
    /// A muddy tire that reaches drier ground throws the cloud; standing water
    /// stays with the mud spray.
    pub fn note_wheel(
        &mut self,
        index: usize,
        contact: Vec3,
        forward: Vec3,
        right: Vec3,
        speed: f32,
        wetness: f32,
        water: f32,
    ) {
        if index > 3 {
            return;
        }
        let rear = index >= 2;
        let outward = if index % 2 == 0 { -1.0 } else { 1.0 };
        let dry = wetness < 0.40 && water < 0.018;
        if water > 0.02 {
            self.recent[index] = self.recent[index].max(wetness.max(0.9));
            self.trail[index] = 0.0;
        } else if wetness > 0.58 {
            self.recent[index] = self.recent[index].max(wetness);
            self.trail[index] = 0.0;
        }

        let leaving = self.armed[index] && self.recent[index] > 0.62 && dry && speed > 1.15;
        if leaving {
            self.recent[index] = wetness;
            self.trail[index] = if rear { 1.65 } else { 0.58 };
            self.burst(contact, forward, right, outward, speed, rear);
        }
        if self.trail[index] > 0.0 && dry && speed > 1.05 && self.cool[index] <= 0.0 {
            self.puff(contact, forward, right, outward, speed, rear);
            self.cool[index] = if rear { 0.040 } else { 0.085 };
        }
        self.armed[index] = true;
    }

    fn burst(
        &mut self,
        contact: Vec3,
        forward: Vec3,
        right: Vec3,
        outward: f32,
        speed: f32,
        rear: bool,
    ) {
        let n = if rear { 18 } else { 7 };
        for k in 0..n {
            let grit = k % 4 == 0;
            let along = k as f32 / n as f32;
            self.mote(contact, forward, right, outward, speed, rear, grit, along);
        }
    }

    fn puff(
        &mut self,
        contact: Vec3,
        forward: Vec3,
        right: Vec3,
        outward: f32,
        speed: f32,
        rear: bool,
    ) {
        let n = if rear { 3 } else { 1 };
        for k in 0..n {
            let along = self.rnd() * 0.22;
            self.mote(
                contact,
                forward,
                right,
                outward,
                speed,
                rear,
                rear && k == 2,
                along,
            );
        }
    }

    fn mote(
        &mut self,
        origin: Vec3,
        forward: Vec3,
        right: Vec3,
        outward: f32,
        speed: f32,
        rear: bool,
        grit: bool,
        along: f32,
    ) {
        let along = if grit {
            0.05 + self.rnd() * 0.35
        } else {
            along * if rear { 1.25 } else { 0.55 } + self.rnd() * 0.12
        };
        let side = outward * (0.06 + self.rnd() * if rear { 0.78 } else { 0.36 })
            + (self.rnd() - 0.5) * 0.28;
        let lift = if grit {
            0.10 + self.rnd() * 0.22
        } else {
            0.04 + along * 0.16 + self.rnd() * 0.08
        };
        let pos = origin - forward * along + right * side + Vec3::Y * lift;
        let slip = (0.12 + self.rnd() * 0.40) * (0.30 + speed * 0.05).min(1.15);
        let vel = -forward * slip
            + right * outward * (0.20 + self.rnd() * if rear { 0.95 } else { 0.45 })
            + Vec3::Y
                * if grit {
                    0.55 + self.rnd() * 0.90
                } else {
                    0.12 + self.rnd() * 0.38 + along * 0.15
                };
        let (life, size, grow, alpha, wide, rgb) = if grit {
            (
                0.32 + self.rnd() * 0.22,
                0.05 + self.rnd() * 0.04,
                0.05,
                0.82,
                0.9,
                [0.52, 0.32, 0.14],
            )
        } else if self.rnd() > 0.42 {
            // Pale sand at the thin end of the plume.
            (
                0.95 + self.rnd() * 0.55,
                0.28 + self.rnd() * 0.18,
                0.50 + self.rnd() * 0.30,
                0.46 + self.rnd() * 0.10,
                1.9 + self.rnd() * 0.65,
                [0.91, 0.73, 0.46],
            )
        } else {
            // Ochre body, denser next to the tread.
            (
                0.75 + self.rnd() * 0.40,
                0.20 + self.rnd() * 0.14,
                0.36 + self.rnd() * 0.20,
                0.64 + self.rnd() * 0.12,
                1.4 + self.rnd() * 0.4,
                [0.80, 0.54, 0.26],
            )
        };
        let mote = DustMote {
            pos,
            vel,
            life,
            max_life: life,
            size,
            grow,
            alpha,
            rgb,
            wide,
            floor: origin.y + 0.03,
            alive: true,
        };
        if self.motes.len() > 460 {
            if let Some(slot) = self.motes.iter_mut().find(|m| !m.alive || m.life <= 0.0) {
                *slot = mote;
            }
            return;
        }
        self.motes.push(mote);
    }

    pub fn update(&mut self, dt: f32) {
        for t in &mut self.trail {
            *t = (*t - dt).max(0.0);
        }
        for c in &mut self.cool {
            *c = (*c - dt).max(0.0);
        }
        for m in &mut self.motes {
            if !m.alive {
                continue;
            }
            m.life -= dt;
            if m.life <= 0.0 {
                m.alive = false;
                continue;
            }
            m.vel *= (1.0 - dt * 1.35).max(0.0);
            m.vel.y -= dt * 0.22;
            m.pos += m.vel * dt;
            if m.pos.y < m.floor {
                m.pos.y = m.floor;
                m.vel.y = m.vel.y.max(0.12);
            }
            m.size = (m.size + m.grow * dt).min(1.35);
        }
    }
}

impl Default for Dust {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wheel(dust: &mut Dust, index: usize, speed: f32, wet: f32, water: f32, steps: u32) {
        let dt = 1.0 / 120.0;
        for _ in 0..steps {
            dust.note_wheel(
                index,
                Vec3::new(0.0, 0.1, 0.0),
                Vec3::Z,
                Vec3::X,
                speed,
                wet,
                water,
            );
            dust.update(dt);
        }
    }

    #[test]
    fn dry_dirt_after_mud_throws_a_cloud() {
        let mut dust = Dust::new();
        wheel(&mut dust, 2, 3.2, 0.94, 0.0, 40);
        assert_eq!(dust.alive_count(), 0, "still in the mud");
        wheel(&mut dust, 2, 3.2, 0.20, 0.0, 8);
        assert!(
            dust.alive_count() > 16,
            "rear tire should throw a tan cloud, alive {}",
            dust.alive_count()
        );
    }

    #[test]
    fn firm_ground_and_standing_water_stay_clear() {
        let mut firm = Dust::new();
        wheel(&mut firm, 0, 4.0, 0.12, 0.0, 80);
        assert_eq!(firm.alive_count(), 0);

        let mut bog = Dust::new();
        wheel(&mut bog, 3, 2.4, 0.98, 0.08, 40);
        wheel(&mut bog, 3, 2.4, 0.22, 0.06, 20);
        assert_eq!(bog.alive_count(), 0, "water is spray, not dust");
    }

    #[test]
    fn a_slow_tire_waits_until_it_is_rolling() {
        let mut dust = Dust::new();
        wheel(&mut dust, 2, 0.4, 0.9, 0.0, 20);
        wheel(&mut dust, 2, 0.4, 0.18, 0.0, 20);
        assert_eq!(dust.alive_count(), 0);
        wheel(&mut dust, 2, 2.4, 0.18, 0.0, 6);
        assert!(dust.alive_count() > 8);
    }
}
