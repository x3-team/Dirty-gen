//! Tan dust kicked up when a tire rolls off mud onto drier ground.
//!
//! The plume is a volume: dense ochre next to the tread, softer and paler as
//! the billows rise and drift behind. It is not a flat trail. It does not wet
//! the soil and it does not change the mud spray.

use glam::Vec3;

/// Soft ball of silt. `wide` stays near 1 — a horizon stretch reads as a ribbon.
/// `kind` is [`DUST_BILLOW`], [`DUST_CORE`], or [`DUST_GRIT`].
#[derive(Clone, Copy, Debug)]
pub struct DustMote {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    pub grow: f32,
    /// Optical density of the ball, not a flat card opacity.
    pub alpha: f32,
    pub rgb: [f32; 3],
    pub wide: f32,
    pub floor: f32,
    pub kind: u8,
    pub alive: bool,
}

pub const DUST_BILLOW: u8 = 0;
pub const DUST_CORE: u8 = 1;
pub const DUST_GRIT: u8 = 2;

#[derive(Clone, Copy)]
enum Role {
    Billow,
    Core,
    Grit,
}

const MOTE_CAP: usize = 480;

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
            self.cool[index] = if rear { 0.045 } else { 0.085 };
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
        let n = if rear { 24 } else { 8 };
        for k in 0..n {
            let role = match k % 6 {
                0 => Role::Grit,
                1 | 2 => Role::Core,
                _ => Role::Billow,
            };
            self.mote(contact, forward, right, outward, speed, rear, role, k, n);
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
        let n = if rear { 5 } else { 2 };
        for k in 0..n {
            let role = match k % 4 {
                0 => Role::Core,
                3 => Role::Grit,
                _ => Role::Billow,
            };
            self.mote(contact, forward, right, outward, speed, rear, role, k, n);
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
        role: Role,
        k: usize,
        n: usize,
    ) {
        let n = n.max(2);
        let along_t = k as f32 / (n - 1) as f32;
        let lift_t = ((k * 3 + 1) % n) as f32 / (n - 1) as f32;
        let side_t = ((k * 2 + 5) % n) as f32 / (n - 1) as f32;
        let fan = side_t - 0.5;
        let rear_scale = if rear { 1.0 } else { 0.62 };

        let (along, lift, spread) = match role {
            Role::Core => (
                0.04 + along_t * 0.42,
                0.12 + lift_t * 0.50,
                0.05 + side_t * 0.36,
            ),
            Role::Billow => (
                0.08 + along_t * 1.05,
                0.18 + lift_t * 0.78,
                0.08 + side_t * 0.82,
            ),
            Role::Grit => (
                0.02 + along_t * 0.28,
                0.16 + lift_t * 0.34,
                0.03 + side_t * 0.20,
            ),
        };
        // Some of the outer silt folds back toward the axle so the two
        // tires read as one cloud from behind, not two separate streaks.
        let inward = if matches!(role, Role::Billow) && self.rnd() > 0.64 {
            -0.55
        } else {
            1.0
        };
        let along = (along + (self.rnd() - 0.5) * 0.10) * rear_scale;
        let lift = (lift + (self.rnd() - 0.5) * 0.08) * if rear { 1.0 } else { 0.7 };
        let side = outward * inward * spread * rear_scale
            + fan * if rear { 0.78 } else { 0.30 }
            + (self.rnd() - 0.5) * 0.14;
        let pos = origin - forward * along.max(0.0) + right * side + Vec3::Y * lift.max(0.04);

        let kick = (0.55 + speed * 0.06).clamp(0.55, 1.25);
        let (back, out, up) = match role {
            Role::Core => (
                0.30 + self.rnd() * 0.50,
                0.22 + self.rnd() * 0.48,
                0.75 + self.rnd() * 0.85,
            ),
            Role::Billow => (
                0.12 + self.rnd() * 0.40,
                0.30 + self.rnd() * 0.80,
                0.50 + self.rnd() * 0.95,
            ),
            Role::Grit => (
                0.45 + self.rnd() * 0.65,
                0.35 + self.rnd() * 0.55,
                1.55 + self.rnd() * 1.40,
            ),
        };
        let side_dir = outward * inward.signum();
        let vel = (-forward * back + right * side_dir * out + Vec3::Y * up) * kick;

        let (life, size, grow, alpha, wide, rgb, kind) = match role {
            Role::Core => (
                0.70 + self.rnd() * 0.30,
                (0.24 + self.rnd() * 0.12) * rear_scale.max(0.8),
                0.12 + self.rnd() * 0.08,
                0.62 + self.rnd() * 0.16,
                0.92 + self.rnd() * 0.16,
                tan([0.58, 0.36, 0.16], [0.72, 0.46, 0.22], lift_t, self.rnd()),
                DUST_CORE,
            ),
            Role::Billow => (
                1.05 + self.rnd() * 0.50,
                (0.50 + self.rnd() * 0.26) * if rear { 1.1 } else { 0.82 },
                0.14 + self.rnd() * 0.10,
                0.28 + self.rnd() * 0.10,
                0.90 + self.rnd() * 0.18,
                tan([0.66, 0.46, 0.24], [0.82, 0.60, 0.34], lift_t, self.rnd()),
                DUST_BILLOW,
            ),
            Role::Grit => (
                0.26 + self.rnd() * 0.18,
                0.07 + self.rnd() * 0.045,
                0.02,
                0.95,
                0.94 + self.rnd() * 0.10,
                tan(
                    [0.34, 0.18, 0.08],
                    [0.46, 0.26, 0.12],
                    self.rnd(),
                    self.rnd(),
                ),
                DUST_GRIT,
            ),
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
            floor: origin.y + 0.02,
            kind,
            alive: true,
        };
        self.spawn(mote);
    }

    fn spawn(&mut self, mote: DustMote) {
        if let Some(slot) = self.motes.iter_mut().find(|m| !m.alive || m.life <= 0.0) {
            *slot = mote;
            return;
        }
        if self.motes.len() >= MOTE_CAP {
            if let Some(slot) = self
                .motes
                .iter_mut()
                .min_by(|a, b| a.life.total_cmp(&b.life))
            {
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
            let youth = (m.life / m.max_life).clamp(0.0, 1.0);
            m.vel *= (1.0 - dt * 0.7).max(0.0);
            // Young silt lifts into a billow, then settles. A hard floor pin
            // was pressing the whole plume into a sheet.
            if youth > 0.62 {
                m.vel.y += dt * 0.55;
            }
            m.vel.y -= dt * 1.05;
            let phase = m.pos.x * 1.8 + m.pos.z * 1.4 + (1.0 - youth) * 7.0;
            m.vel.x += phase.sin() * dt * 0.7;
            m.vel.z += (phase * 1.3).cos() * dt * 0.5;
            m.pos += m.vel * dt;
            if m.pos.y < m.floor {
                m.pos.y = m.floor;
                m.vel.y = 0.0;
            }
            let cap = if m.kind == DUST_GRIT { 0.16 } else { 0.95 };
            m.size = (m.size + m.grow * dt).min(cap);
        }
    }
}

fn tan(dark: [f32; 3], light: [f32; 3], t: f32, jitter: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    let j = (jitter - 0.5) * 0.05;
    [
        (dark[0] + (light[0] - dark[0]) * t + j).clamp(0.15, 0.88),
        (dark[1] + (light[1] - dark[1]) * t + j * 0.8).clamp(0.10, 0.70),
        (dark[2] + (light[2] - dark[2]) * t + j * 0.5).clamp(0.05, 0.48),
    ]
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

    #[test]
    fn plume_is_a_volume_of_round_billows() {
        let mut dust = Dust::new();
        wheel(&mut dust, 2, 3.4, 0.92, 0.0, 30);
        wheel(&mut dust, 2, 3.4, 0.18, 0.0, 36);
        let alive: Vec<_> = dust
            .motes
            .iter()
            .filter(|m| m.alive && m.life > 0.0)
            .collect();
        assert!(alive.len() > 24, "thin plume: {}", alive.len());
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut high = 0usize;
        let mut dense = 0usize;
        for m in &alive {
            assert!(
                (0.8..1.25).contains(&m.wide),
                "wide {} would stretch a ribbon",
                m.wide
            );
            assert!(
                m.rgb[0] > m.rgb[2] + 0.08,
                "dust left the tan range {:?}",
                m.rgb
            );
            assert!(m.rgb[0] < 0.90, "dust is paper-white {:?}", m.rgb);
            min = min.min(m.pos);
            max = max.max(m.pos);
            if m.pos.y > 0.42 {
                high += 1;
            }
            if m.kind == DUST_CORE && m.alpha > 0.55 {
                dense += 1;
            }
        }
        let span = max - min;
        assert!(
            span.y > 0.55,
            "plume is flat, height {:.2} span {:?}",
            span.y,
            span
        );
        assert!(span.x > 0.45, "plume has no width {:.2}", span.x);
        assert!(span.z > 0.35, "plume has no depth {:.2}", span.z);
        assert!(
            high * 4 > alive.len(),
            "only {high} of {} motes left the ground",
            alive.len()
        );
        assert!(dense > 3, "no dense core in the plume");
    }
}
