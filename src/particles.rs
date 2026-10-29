//! Short-lived splash. Pieces are ballistic solids: they arc, then rest in the
//! soil. The renderer turns them into soft wet clods, not stones or cards.
//! Slurry that breaks away from a tire is a ribbon or a drop, and its volume
//! is already on the soil ledger.

use glam::{EulerRot, Quat, Vec3};

use crate::soil::SoilField;

/// Liquid mud torn off a slurry seal. A curled sheet, not a block.
pub const KIND_SLURRY_RIBBON: u8 = 5;
/// Liquid mud torn off a slurry seal. A small fleck, not a teardrop blob.
pub const KIND_SLURRY_DROP: u8 = 6;
/// A wider sheet from a roll in slurry. Still a ribbon, not a tablet.
pub const KIND_SLURRY_SHEET: u8 = 7;

/// Volume accepted into ribbon pieces and drop pieces.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlurryThrow {
    pub ribbons: f32,
    pub drops: f32,
}

impl SlurryThrow {
    pub fn total(self) -> f32 {
        self.ribbons + self.drops
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    /// 0 tire droplet, 1 flung clod, 2 drip, 3 impact chunk, 4 impact droplet,
    /// 5 slurry ribbon, 6 slurry drop, 7 slurry sheet. The last three are
    /// liquid mud. A roll sheds sheets. A sideways spin sheds ribbons.
    pub kind: u8,
    pub alive: bool,
    pub seed: u32,
    pub spin: Vec3,
    pub rot: Quat,
    /// Sitting on the soil after the arc. The mesh stays until `life` ends.
    pub grounded: bool,
    pub deposited: bool,
    /// Cubic metres this piece is carrying for the soil ledger.
    pub volume: f32,
    /// The volume came out of a contact. Landing places it; it is not minted.
    pub ledger: bool,
    /// Landing puts the volume back into the slurry rut, not on the shoulder.
    pub returns_to_rut: bool,
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
            ledger: false,
            returns_to_rut: false,
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
            volume: 0.0,
            ledger: false,
            returns_to_rut: false,
        };
        let _ = self.insert(piece);
    }

    /// One clod whose volume is already on the soil ledger. Returns the
    /// cubic metres the particle actually took. Zero means the caller still
    /// holds that mass in `airborne` and must place it.
    pub fn throw_soil(&mut self, origin: Vec3, vel: Vec3, volume: f32) -> f32 {
        if volume <= 1.0e-8 || !volume.is_finite() {
            return 0.0;
        }
        let seed = self.rng;
        let _ = self.rnd();
        let size = (volume / 0.35).cbrt().clamp(0.10, 0.28);
        let spin = Vec3::new(
            (self.rnd() - 0.5) * 4.0,
            (self.rnd() - 0.5) * 4.0,
            (self.rnd() - 0.5) * 4.0,
        );
        let rot = Quat::from_euler(
            EulerRot::YXZ,
            self.rnd() * std::f32::consts::TAU,
            (self.rnd() - 0.5) * 1.4,
            (self.rnd() - 0.5) * 1.4,
        );
        let piece = Particle {
            pos: origin,
            vel,
            life: 4.0,
            max_life: 4.0,
            size,
            kind: 1,
            alive: true,
            seed,
            spin,
            rot,
            grounded: false,
            deposited: false,
            volume,
            ledger: true,
            returns_to_rut: false,
        };
        if self.insert(piece) {
            volume
        } else {
            0.0
        }
    }

    /// Throw liquid slurry as ribbons and drops. `volume` is already in the
    /// soil's thrown account. The return is the share the pieces actually
    /// carry; anything left over stays in that account without a mesh.
    pub fn throw_slurry(
        &mut self,
        origin: Vec3,
        forward: Vec3,
        right: Vec3,
        volume: f32,
    ) -> SlurryThrow {
        let forward = horizontal_axis(forward, Vec3::Z);
        let right = horizontal_axis(right, Vec3::X);
        if volume <= 1.0e-8 || !volume.is_finite() {
            return SlurryThrow::default();
        }
        let ribbon_vol = volume * 0.62;
        let drop_vol = volume - ribbon_vol;
        let n_r = 3usize;
        let n_d = 5usize;
        let mut ribbons = 0.0_f32;
        let mut drops = 0.0_f32;
        for i in 0..n_r {
            let t = i as f32;
            let share = ribbon_vol / n_r as f32;
            // Outboard of the tread, so the sheet is not hidden inside the tire.
            let pos = origin
                + forward * (0.02 + t * 0.05)
                + right * (0.22 + t * 0.06)
                + Vec3::Y * (0.20 + t * 0.16);
            let vel = -forward * (2.4 + t * 0.45) + Vec3::Y * (3.1 + t * 0.25) + right * (1.8 + t * 0.35);
            ribbons += self.spawn_slurry(pos, vel, share, KIND_SLURRY_RIBBON, false);
        }
        // Beside the ribbons, in the air. A stack on the hub read as teardrop blobs.
        let drop_at = [
            (0.26_f32, 0.32, 0.22),
            (-0.20, 0.30, 0.26),
            (0.30, 0.34, 0.38),
            (-0.22, 0.32, 0.44),
            (0.18, 0.36, 0.50),
        ];
        for i in 0..n_d {
            let t = i as f32;
            let share = drop_vol / n_d as f32;
            let (along, out, up) = drop_at[i];
            let pos = origin + forward * along + right * out + Vec3::Y * up;
            // Mostly rearward, so the fleck's long axis lies across the side view.
            let vel = -forward * (2.4 + t * 0.25) + Vec3::Y * 0.35 + right * 0.12;
            drops += self.spawn_slurry(pos, vel, share, KIND_SLURRY_DROP, false);
        }
        SlurryThrow { ribbons, drops }
    }

    /// Slurry that was held in the tread voids. It leaves as ribbons only.
    /// The volume is already in the thrown account. Landing returns it to the rut.
    pub fn throw_groove_ribbons(
        &mut self,
        origin: Vec3,
        forward: Vec3,
        right: Vec3,
        volume: f32,
    ) -> f32 {
        let forward = horizontal_axis(forward, Vec3::Z);
        let right = horizontal_axis(right, Vec3::X);
        if volume <= 1.0e-8 || !volume.is_finite() {
            return 0.0;
        }
        // Three sheets just behind the rear crown, so they peel off the
        // voids instead of crossing the rim.
        // Inboard of the crown, so the sheet leaves the groove into the
        // open and does not cross the rim.
        let stations = [
            (-0.48_f32, -0.16, -0.08),
            (-0.42, -0.22, -0.18),
            (-0.50, -0.12, -0.02),
        ];
        let n = stations.len();
        let mut carried = 0.0_f32;
        for (i, (along, out, up)) in stations.into_iter().enumerate() {
            let share = volume / n as f32;
            let pos = origin + forward * along + right * out + Vec3::Y * up;
            let t = i as f32;
            // Rearward and outboard. The long axis crosses a camera behind
            // the tire, and the curl reads as a ribbon leaving the tread.
            let vel = -forward * (1.6 + t * 0.2) - right * (2.8 + t * 0.25) + Vec3::Y * (0.85 + t * 0.12);
            carried += self.spawn_slurry(pos, vel, share, KIND_SLURRY_RIBBON, true);
        }
        carried
    }

    /// Slurry torn off by a roll. Wide sheets, not spheres, drops, or tablets.
    /// The volume is already in the thrown account.
    pub fn throw_roll_sheets(
        &mut self,
        origin: Vec3,
        forward: Vec3,
        right: Vec3,
        volume: f32,
    ) -> f32 {
        let forward = horizontal_axis(forward, Vec3::Z);
        let right = horizontal_axis(right, Vec3::X);
        if volume <= 1.0e-8 || !volume.is_finite() {
            return 0.0;
        }
        // Spread through the air beside the rut. Stacked stations read as one lump.
        let stations = [
            (-1.05_f32, -0.15, 1.15),
            (0.45, 0.65, 0.72),
            (1.20, -0.15, 0.32),
            (-0.25, 0.55, 0.98),
        ];
        let n = stations.len();
        let mut carried = 0.0_f32;
        for (i, (along, out, up)) in stations.into_iter().enumerate() {
            let share = volume / n as f32;
            let pos = origin + forward * along + right * out + Vec3::Y * up;
            let t = i as f32;
            // Along the sheet, slightly upward, so the long axis is a ribbon
            // in the air beside the berm and not a lump on the tire.
            let vel = right * (2.4 + t * 0.15) + forward * 0.35 + Vec3::Y * 0.55;
            carried += self.spawn_slurry(pos, vel, share, KIND_SLURRY_SHEET, false);
        }
        carried
    }

    /// Slurry torn off by a sideways spin. Ribbons only. No drops.
    /// The volume is already in the thrown account. Landing returns it to the rut.
    pub fn throw_spin_ribbons(
        &mut self,
        origin: Vec3,
        forward: Vec3,
        right: Vec3,
        volume: f32,
    ) -> f32 {
        let forward = horizontal_axis(forward, Vec3::Z);
        let right = horizontal_axis(right, Vec3::X);
        if volume <= 1.0e-8 || !volume.is_finite() {
            return 0.0;
        }
        let stations = [
            (0.75_f32, 0.20, 0.52),
            (-0.90, 0.95, 1.15),
            (-1.10, -0.45, 0.48),
        ];
        let n = stations.len();
        let mut carried = 0.0_f32;
        for (i, (along, out, up)) in stations.into_iter().enumerate() {
            let share = volume / n as f32;
            let pos = origin + forward * along + right * out + Vec3::Y * up;
            let t = i as f32;
            let vel = right * (8.5 + t * 0.4) + Vec3::Y * (1.2 + t * 0.15) + forward * 0.35;
            carried += self.spawn_slurry(pos, vel, share, KIND_SLURRY_RIBBON, true);
        }
        carried
    }

    fn spawn_slurry(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        volume: f32,
        kind: u8,
        returns_to_rut: bool,
    ) -> f32 {
        if volume <= 1.0e-8 || !volume.is_finite() {
            return 0.0;
        }
        let seed = self.rng;
        let _ = self.rnd();
        let spin = Vec3::new(
            (self.rnd() - 0.5) * 2.2,
            (self.rnd() - 0.5) * 2.2,
            (self.rnd() - 0.5) * 2.2,
        );
        let rot = Quat::from_euler(
            EulerRot::YXZ,
            self.rnd() * std::f32::consts::TAU,
            (self.rnd() - 0.5) * 0.6,
            (self.rnd() - 0.5) * 0.6,
        );
        let size = if kind == KIND_SLURRY_DROP {
            (volume * 40.0).cbrt().clamp(0.028, 0.07)
        } else {
            (volume * 12.0).cbrt().clamp(0.04, 0.12)
        };
        let piece = Particle {
            pos,
            vel,
            life: 3.2,
            max_life: 3.2,
            size,
            kind,
            alive: true,
            seed,
            spin,
            rot,
            grounded: false,
            deposited: false,
            volume,
            ledger: true,
            returns_to_rut,
        };
        if self.insert(piece) {
            volume
        } else {
            0.0
        }
    }

    fn insert(&mut self, piece: Particle) -> bool {
        if self.items.len() > 900 {
            if let Some(slot) = self.items.iter_mut().find(|p| !p.alive || p.life <= 0.0) {
                *slot = piece;
                return true;
            }
            return false;
        }
        self.items.push(piece);
        true
    }

    /// Drop any ledger mass that is still in the air so a settled pose cannot
    /// lose it. Grounded pieces stay visible.
    pub fn land_ledger(&mut self, soil: &mut SoilField) {
        for p in &mut self.items {
            if !p.ledger || p.volume <= 1.0e-8 {
                continue;
            }
            if let Some(rest) = deposit_ledger(soil, p) {
                p.pos = rest + Vec3::Y * p.size * 0.22;
                p.volume = 0.0;
                p.deposited = true;
                p.grounded = true;
                p.vel = Vec3::ZERO;
                p.spin = Vec3::ZERO;
                p.alive = true;
                p.life = p.life.max(2.0);
            }
        }
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
                if p.ledger && p.volume > 1.0e-8 {
                    if let Some(rest) = deposit_ledger(soil, p) {
                        p.pos = rest + Vec3::Y * p.size * 0.22;
                        p.volume = 0.0;
                    }
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
                KIND_SLURRY_RIBBON => 0.45,
                KIND_SLURRY_DROP => 0.7,
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
                1 | KIND_SLURRY_RIBBON => 0.22,
                KIND_SLURRY_DROP => 0.28,
                _ => 0.32,
            };
            let sit = h + p.size * belly;
            if p.pos.y <= sit {
                if !p.deposited {
                    if p.ledger {
                        if let Some(rest) = deposit_ledger(soil, p) {
                            p.pos.x = rest.x;
                            p.pos.z = rest.z;
                            p.volume = 0.0;
                            p.deposited = true;
                        }
                    } else if let Some(idx) = soil.idx_at(p.pos.x, p.pos.z) {
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
                        p.deposited = true;
                    } else {
                        p.deposited = true;
                    }
                }
                if p.deposited {
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
}

fn deposit_ledger(soil: &mut SoilField, p: &Particle) -> Option<Vec3> {
    if p.returns_to_rut {
        soil.return_slurry_to_rut(p.pos.x, p.pos.z, p.volume)
    } else {
        soil.receive_thrown(p.pos.x, p.pos.z, p.volume)
    }
}

fn horizontal_axis(v: Vec3, fallback: Vec3) -> Vec3 {
    let h = Vec3::new(v.x, 0.0, v.z);
    if h.length_squared() < 1.0e-8 {
        fallback
    } else {
        h.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slurry_breakaway_splits_ribbons_and_drops() {
        let mut particles = Particles::new();
        let volume = 0.012_f32;
        let threw = particles.throw_slurry(Vec3::new(0.2, 0.4, -0.3), Vec3::Z, Vec3::X, volume);
        assert!(threw.ribbons > 0.004, "ribbons {}", threw.ribbons);
        assert!(threw.drops > 0.002, "drops {}", threw.drops);
        assert!(
            (threw.total() - volume).abs() < 1.0e-5,
            "threw {} of {volume}",
            threw.total()
        );
        let ribbons = particles
            .items
            .iter()
            .filter(|p| p.alive && p.ledger && p.kind == KIND_SLURRY_RIBBON)
            .count();
        let drops = particles
            .items
            .iter()
            .filter(|p| p.alive && p.ledger && p.kind == KIND_SLURRY_DROP)
            .count();
        assert!(ribbons >= 2, "ribbon pieces {ribbons}");
        assert!(drops >= 3, "drop pieces {drops}");
        let carried: f32 = particles.items.iter().map(|p| p.volume).sum();
        assert!((carried - volume).abs() < 1.0e-5);
    }

    /// Pull-out packs the voids. The pack stays while the tire is still, then
    /// a spin throws it as ribbons and the landing puts it back in the rut.
    #[test]
    fn groove_hold_then_spin_shed_keeps_volume() {
        use crate::patch::{groove_slurry_capacity, release_groove_slurry};
        use crate::soil::{SurfaceKind, SoilField, SLURRY_COMPARE_MOISTURE};

        let moisture = SLURRY_COMPARE_MOISTURE;
        let mut soil = SoilField::mud_pad();
        let depth = 0.16_f32;
        let poured = soil.fill_slurry_pool(
            -1.4,
            1.4,
            -1.4,
            1.4,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        assert!(poured > 0.2, "the pool did not fill, {poured}");
        let idx = soil.idx_at(0.04, 0.08).expect("pool cell");
        let half = soil.size * 0.5;
        let i = idx % soil.res;
        let j = idx / soil.res;
        let x = (i as f32 + 0.5) * soil.cell - half;
        let z = (j as f32 + 0.5) * soil.cell - half;
        let bed0: f32 = soil.bed.iter().sum();
        let placed0 = soil.placed_volume();
        let before = soil.mix_volumes();
        let cap = groove_slurry_capacity(0.38, 0.28);
        let packed = soil.pack_slurry_into_grooves(x, z, Vec3::Z, 0.38, 0.28, cap);
        assert!(
            packed > cap * 0.85,
            "pull-out should fill the voids, packed {packed:.5} cap {cap:.5}"
        );
        assert!((soil.bed.iter().sum::<f32>() - bed0).abs() < 1.0e-4, "packing cut the bed");
        assert!(
            (before.slurry - soil.mix_volumes().slurry - packed).abs() < 1.0e-4,
            "the grooves were not taken from the slurry"
        );
        assert!(soil.airborne < 1.0e-6, "packing threw slurry");

        let mut held = packed;
        let dt = 1.0 / 60.0;
        for _ in 0..24 {
            let out = release_groove_slurry(&mut held, 0.0, 0.0, dt);
            assert!(out < 1.0e-8, "a still tire shed {out}");
        }
        assert!((held - packed).abs() < 1.0e-6, "the hold drifted");

        let mut particles = Particles::new();
        let mut thrown = 0.0_f32;
        for _ in 0..90 {
            let out = release_groove_slurry(&mut held, 1.0, 0.0, dt);
            if out <= 1.0e-8 {
                continue;
            }
            soil.airborne += out;
            let got = particles.throw_groove_ribbons(Vec3::new(x, 0.45, z), Vec3::Z, Vec3::X, out);
            assert!(
                (got - out).abs() < 1.0e-5,
                "ribbons carried {got:.6} of {out:.6}"
            );
            thrown += out;
        }
        assert!(
            held < packed * 0.08,
            "spin left {:.5} of {:.5} in the grooves",
            held,
            packed
        );
        assert!(thrown > packed * 0.9, "spin threw {thrown:.5} of {packed:.5}");
        let carried: f32 = particles.items.iter().map(|p| p.volume).sum();
        assert!((carried - thrown).abs() < 1.0e-4, "ribbon ledger {carried:.5} thrown {thrown:.5}");
        for p in &particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert_eq!(p.kind, KIND_SLURRY_RIBBON, "shed piece is not a ribbon");
            assert!(p.returns_to_rut, "the ribbon does not return to the rut");
            assert!(p.ledger, "the ribbon is off the ledger");
        }
        assert!(
            particles
                .items
                .iter()
                .all(|p| p.kind != KIND_SLURRY_DROP && p.kind != 0 && p.kind != 1),
            "spin shed a drop, a sphere, or a clod"
        );
        eprintln!(
            "groove-hold before land  packed {packed:.5} m3  groove {held:.5} m3  thrown {thrown:.5} m3  returned 0"
        );

        particles.land_ledger(&mut soil);
        let after = soil.mix_volumes();
        let returned = after.slurry - (before.slurry - packed);
        eprintln!(
            "groove-hold spin-shed  packed {packed:.5} m3  groove {held:.5} m3  thrown {:.5} m3  returned {returned:.5} m3  sum {:.5} m3",
            soil.airborne,
            held + soil.airborne + returned
        );
        assert!(soil.airborne < 1.0e-5, "ribbons still airborne {}", soil.airborne);
        assert!(
            (held + returned - packed).abs() < 1.0e-4,
            "volume drifted groove {held:.5} returned {returned:.5} packed {packed:.5}"
        );
        assert!(
            returned > packed * 0.9,
            "the rut did not get the shed slurry, returned {returned:.5}"
        );
        let gap = (before.total() - (after.total() + soil.airborne + held)).abs();
        assert!(
            gap < 1.0e-3,
            "column drifted before {:.5} after {:.5} groove {held:.5}",
            before.total(),
            after.total()
        );
        assert!(
            (soil.placed_volume() - placed0).abs() < 1.0e-4,
            "the ribbons landed as placed spoil"
        );
        assert!((soil.bed.iter().sum::<f32>() - bed0).abs() < 1.0e-3, "the shed cut the bed");

        // A shedding coat, with the wheel not spinning, opens the same voids.
        let packed_coat = soil.pack_slurry_into_grooves(x, z, Vec3::Z, 0.38, 0.28, cap);
        assert!(packed_coat > cap * 0.85, "the rut had nothing left to pack");
        let slurry_mid = soil.mix_volumes().slurry;
        let mut coat_held = packed_coat;
        let idle = release_groove_slurry(&mut coat_held, 0.02, 0.0, dt);
        assert!(idle < 1.0e-8, "a slow roll shed {idle}");
        let coat_out = release_groove_slurry(&mut coat_held, 0.0, 0.08, dt);
        assert!(
            coat_out > packed_coat * 0.7,
            "the coat shed left the slurry in the grooves, out {coat_out:.5} held {coat_held:.5}"
        );
        soil.airborne += coat_out;
        let got = particles.throw_groove_ribbons(Vec3::new(x, 0.45, z), Vec3::Z, Vec3::X, coat_out);
        assert!((got - coat_out).abs() < 1.0e-5);
        for p in &particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert_eq!(p.kind, KIND_SLURRY_RIBBON, "coat shed is not a ribbon");
        }
        particles.land_ledger(&mut soil);
        let coat_after = soil.mix_volumes();
        let coat_returned = coat_after.slurry - slurry_mid;
        eprintln!(
            "groove-hold coat-shed  packed {packed_coat:.5} m3  groove {coat_held:.5} m3  thrown {:.5} m3  returned {coat_returned:.5} m3  sum {:.5} m3",
            soil.airborne,
            coat_held + soil.airborne + coat_returned
        );
        assert!(
            (coat_held + soil.airborne + coat_returned - packed_coat).abs() < 1.0e-4,
            "coat shed deleted volume"
        );
    }

    /// A roll sheds sheets. A sideways spin sheds ribbons. Neither is a drop,
    /// a sphere, or a clod, and both carry the volume they were given.
    #[test]
    fn roll_sheets_and_spin_ribbons_carry_the_volume() {
        let mut particles = Particles::new();
        let sheets = particles.throw_roll_sheets(Vec3::new(0.2, 0.4, 0.0), Vec3::Z, Vec3::X, 0.018);
        let ribbons = particles.throw_spin_ribbons(Vec3::new(0.2, 0.3, 0.1), Vec3::Z, Vec3::X, 0.006);
        assert!((sheets - 0.018).abs() < 1.0e-5, "sheets {sheets}");
        assert!((ribbons - 0.006).abs() < 1.0e-5, "ribbons {ribbons}");
        let mut sheet_n = 0;
        let mut ribbon_n = 0;
        for p in &particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert!(
                p.kind == KIND_SLURRY_SHEET || p.kind == KIND_SLURRY_RIBBON,
                "shed kind {} is not a sheet or a ribbon",
                p.kind
            );
            assert_ne!(p.kind, KIND_SLURRY_DROP);
            assert_ne!(p.kind, 1, "shed a clod");
            assert_ne!(p.kind, 0, "shed a sphere droplet");
            if p.kind == KIND_SLURRY_SHEET {
                sheet_n += 1;
                assert!(!p.returns_to_rut);
            } else {
                ribbon_n += 1;
                assert!(p.returns_to_rut);
            }
        }
        assert!(sheet_n >= 3, "sheet count {sheet_n}");
        assert!(ribbon_n >= 3, "ribbon count {ribbon_n}");
    }
}
