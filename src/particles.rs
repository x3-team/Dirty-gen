//! Soil torn off the tread. Each piece's volume is a share of the cut that
//! the heightfield did not pile back. The mesh is a chunk, a ribbon, a drop,
//! or a dust flake — not a sphere.

use glam::{Mat3, Quat, Vec3};

use crate::soil::SoilSample;

pub const SHAPE_CLOD: u8 = 0;
pub const SHAPE_RIBBON: u8 = 1;
pub const SHAPE_DROP: u8 = 2;
pub const SHAPE_DUST: u8 = 3;

pub const FACE_CLAY: u8 = 0;
pub const FACE_MUD: u8 = 1;
pub const FACE_GRAVEL: u8 = 2;

#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: Vec3,
    pub vel: Vec3,
    pub life: f32,
    pub max_life: f32,
    pub size: f32,
    /// 0 drop, 1 clod, 2 body drip, 3 impact chunk, 4 impact droplet,
    /// 5 ribbon, 6 dust.
    pub kind: u8,
    pub alive: bool,
    pub seed: u32,
    pub spin: Vec3,
    pub rot: Quat,
    /// Sitting on the soil after the arc. The mesh stays until `life` ends.
    pub grounded: bool,
    pub deposited: bool,
    /// Cubic metres this piece stands for.
    pub volume: f32,
    /// Mesh extents. Clods and drops store ellipsoid radii. Ribbons and dust
    /// store full length, width, and thickness.
    pub span: Vec3,
    pub shape: u8,
    pub face: u8,
    pub wet: f32,
    pub dark: f32,
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
            span: Vec3::splat(0.02),
            shape: SHAPE_CLOD,
            face: FACE_MUD,
            wet: 0.5,
            dark: 0.2,
        }
    }
}

/// One wheel stamp's unpiled soil, and the tread it left from.
#[derive(Clone, Copy, Debug)]
pub struct TreadThrow {
    pub origin: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub speed: f32,
    pub slip: f32,
    pub sample: SoilSample,
    pub volume: f32,
    pub wheel_radius: f32,
    pub wheel_width: f32,
    /// +1 throws to the tire's right, -1 to its left. One shoulder, so the
    /// pieces of a stamp can join into a ribbon instead of a scatter.
    pub outward: f32,
}

/// Clay holds together, mud is slurry, gravel is dry and broken.
pub fn soil_face(sample: &SoilSample) -> u8 {
    if sample.water > 0.01 || sample.wetness > 0.78 {
        FACE_MUD
    } else if sample.wetness < 0.34 && sample.roughness > 0.62 {
        FACE_GRAVEL
    } else if sample.wetness < 0.28 {
        FACE_GRAVEL
    } else {
        FACE_CLAY
    }
}

pub fn moisture_of(sample: &SoilSample) -> f32 {
    (sample.wetness * 0.65 + (sample.water / 0.12).clamp(0.0, 1.0) * 0.35).clamp(0.0, 1.0)
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

    /// Tear `throw.volume` into chunks, ribbons, drops, or dust. The sum of
    /// the piece volumes is the volume that left the ground.
    pub fn throw_tread(&mut self, throw: &TreadThrow) {
        let volume = throw.volume.max(0.0);
        if volume < 1.0e-6 {
            return;
        }
        let face = soil_face(&throw.sample);
        let moisture = moisture_of(&throw.sample);
        let budget = shape_budget(face, moisture);
        let mut specs: Vec<(u8, f32)> = Vec::new();
        for (shape, share) in budget.into_iter().enumerate() {
            if share <= 0.0 {
                continue;
            }
            let part = volume * share;
            let n = piece_count(part, throw.slip, throw.speed, shape as u8);
            let each = part / n as f32;
            for _ in 0..n {
                specs.push((shape as u8, each));
            }
        }
        if specs.is_empty() {
            specs.push((SHAPE_CLOD, volume));
        }
        let sum: f32 = specs.iter().map(|s| s.1).sum();
        if sum > 0.0 {
            let scale = volume / sum;
            for spec in &mut specs {
                spec.1 *= scale;
            }
        }
        let side = if throw.outward == 0.0 {
            1.0
        } else {
            throw.outward.signum()
        };
        for (shape, vol) in specs {
            if vol <= 1.0e-8 {
                continue;
            }
            self.spawn_tread(throw, shape, vol, face, moisture, side);
        }
    }

    /// The crater's unpiled soil, in torn chunks. Returns how many pieces left.
    pub fn impact_from_cut(&mut self, origin: Vec3, sample: SoilSample, ejected: f32) -> u32 {
        let ejected = ejected.max(0.0);
        if ejected < 1.0e-6 {
            return 0;
        }
        let face = soil_face(&sample);
        let moisture = moisture_of(&sample);
        let n_big = 8u32;
        let n_small = 8u32;
        let big_each = ejected * 0.72 / n_big as f32;
        let small_each = ejected * 0.28 / n_small as f32;
        let mut n = 0u32;
        for i in 0..n_big {
            let ang = i as f32 / n_big as f32 * std::f32::consts::TAU + (self.rnd() - 0.5) * 0.35;
            let speed = 0.55 + self.rnd() * 1.05;
            let up = 2.5 + self.rnd() * 1.5;
            let vel = Vec3::new(ang.cos() * speed, up, ang.sin() * speed);
            let pos = origin
                + Vec3::new(ang.cos(), 0.25 + self.rnd() * 0.15, ang.sin()) * (0.2 + self.rnd() * 0.35);
            let shape = if moisture > 0.55 && i % 3 == 0 {
                SHAPE_RIBBON
            } else {
                SHAPE_CLOD
            };
            let life = 2.5 + self.rnd() * 0.7;
            self.push_shaped(
                pos,
                vel,
                life,
                big_each,
                shape,
                face,
                moisture,
                sample.darken,
                3,
                0.8,
                2.2,
                3.4,
            );
            n += 1;
        }
        for _ in 0..n_small {
            let ang = self.rnd() * std::f32::consts::TAU;
            let speed = 1.1 + self.rnd() * 2.4;
            let up = 2.0 + self.rnd() * 2.2;
            let vel = Vec3::new(ang.cos() * speed, up, ang.sin() * speed);
            let pos = origin
                + Vec3::new(
                    (self.rnd() - 0.5) * 0.5,
                    0.15 + self.rnd() * 0.2,
                    (self.rnd() - 0.5) * 0.5,
                );
            let shape = if moisture > 0.45 {
                if self.rnd() > 0.5 {
                    SHAPE_DROP
                } else {
                    SHAPE_RIBBON
                }
            } else if self.rnd() > 0.45 {
                SHAPE_DUST
            } else {
                SHAPE_CLOD
            };
            let life = 1.6 + self.rnd() * 0.8;
            self.push_shaped(
                pos,
                vel,
                life,
                small_each,
                shape,
                face,
                moisture,
                sample.darken,
                4,
                0.4,
                3.0,
                6.5,
            );
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
        let volume = 1.4e-6 + self.rnd() * 1.1e-6;
        self.push_shaped(
            origin,
            vel,
            0.7,
            volume,
            SHAPE_DROP,
            FACE_MUD,
            0.85,
            0.4,
            2,
            0.0,
            0.4,
            4.0,
        );
    }

    /// Fold fresh volume into a piece that just left this spot. The sum of
    /// volumes stays the cut. Once a piece is big enough to read, or it has
    /// left the tread, the next bit of the cut starts a new chunk.
    fn absorb(&mut self, pos: Vec3, shape: u8, face: u8, volume: f32) -> bool {
        let cap = absorb_cap(shape);
        let mut best: Option<usize> = None;
        let mut best_d = match shape {
            SHAPE_RIBBON => 0.85,
            SHAPE_CLOD => 0.7,
            _ => 0.35,
        };
        let age_limit = if shape == SHAPE_RIBBON { 0.28 } else { 0.3 };
        for (i, p) in self.items.iter().enumerate() {
            if !p.alive || p.grounded || p.shape != shape || p.face != face {
                continue;
            }
            if p.volume >= cap || p.volume + volume > cap * 1.08 {
                continue;
            }
            if p.max_life - p.life > age_limit {
                continue;
            }
            let d = (p.pos - pos).length();
            if d < best_d {
                best_d = d;
                best = Some(i);
            }
        }
        let Some(i) = best else {
            return false;
        };
        let p = &mut self.items[i];
        let new_vol = p.volume + volume;
        let scale = (new_vol / p.volume.max(1.0e-12)).cbrt();
        p.span *= scale;
        p.volume = new_vol;
        p.size = p.span.max_element().max(0.01);
        true
    }

    fn spawn_tread(
        &mut self,
        throw: &TreadThrow,
        shape: u8,
        volume: f32,
        face: u8,
        moisture: f32,
        side: f32,
    ) {
        let mut fwd = throw.forward;
        if fwd.length_squared() < 1.0e-8 {
            fwd = Vec3::Z;
        }
        fwd = fwd.normalize();
        let mut right = throw.right;
        if right.length_squared() < 1.0e-8 {
            right = Vec3::X;
        }
        right = right.normalize();
        let radius = throw.wheel_radius.max(0.18);
        let width = throw.wheel_width.max(0.08);
        // Just off the rear of the tread. The contact patch is under the hub;
        // the rubber there has already turned up, so the tear starts on that
        // shoulder instead of inside the footprint.
        let clear = match shape {
            SHAPE_RIBBON => 0.88,
            SHAPE_DROP => 0.7,
            _ => 0.48,
        };
        let along = radius * (clear + self.rnd() * 0.06);
        let lift = radius
            * match shape {
                SHAPE_RIBBON => 0.52 + self.rnd() * 0.08,
                SHAPE_DUST => 0.42 + self.rnd() * 0.16,
                _ => 0.2 + self.rnd() * 0.08,
            };
        // A ribbon stays on the tread it tore from. Dust and clods can leave
        // the shoulder. A spawn outside the sidewall reads as a strap beside
        // the tire.
        let lateral = match shape {
            SHAPE_RIBBON => side * width * (0.16 + self.rnd() * 0.14),
            SHAPE_DROP => side * width * (0.3 + self.rnd() * 0.22),
            _ => side * (width * 0.62 + 0.02 + self.rnd() * width * 0.12),
        };
        let pos = throw.origin - fwd * along + right * lateral + Vec3::Y * lift;
        if self.absorb(pos, shape, face, volume) {
            return;
        }

        let slip_s = throw.slip.clamp(-1.5, 1.5);
        let drive = slip_s.max(0.0);
        let brake = (-slip_s).max(0.0);
        let speed = throw.speed.max(0.0);
        // Speed and slip set how hard the tread throws. Moisture keeps wet
        // slurry in a low peel; dry dust is the part that rises.
        let rate = 0.85 + speed * 0.48 + drive * 1.25 + brake * 0.7;
        let rearward = (0.82 + drive * 0.3 - brake * 0.42).clamp(0.3, 1.25);
        let speed_n = (speed / 6.0).clamp(0.0, 1.0);
        let dry = (1.0 - moisture).clamp(0.0, 1.0);
        let shape_loft = match shape {
            // A wet ribbon arcs off the tread. Wet clods tumble out low.
            // Dry clods may hop; dust is the part that rises.
            SHAPE_RIBBON => 0.62 + moisture * 0.08,
            SHAPE_DROP => 0.7 + moisture * 0.06,
            SHAPE_DUST => 1.2 + dry * 0.28,
            _ => 0.14 + dry * 0.5,
        };
        let loft = shape_loft + drive * 0.08 * dry + speed_n * 0.05;
        let out = match shape {
            SHAPE_RIBBON => side * (0.03 + brake * 0.05),
            SHAPE_DROP => side * (0.08 + (drive + brake) * 0.1),
            _ => side * (0.12 + (drive + brake) * 0.16 + dry * 0.18),
        };
        let jitter = Vec3::new(
            (self.rnd() - 0.5) * 0.06,
            (self.rnd() - 0.5) * 0.04,
            (self.rnd() - 0.5) * 0.06,
        );
        // The tread throws backward. The scale keeps a still of the pass
        // from showing the strap a car-length behind the tire.
        let vel = -fwd * rate * rearward * 0.68
            + fwd * rate * brake * 0.55
            + Vec3::Y * rate * loft
            + right * rate * out
            + jitter;

        let life = match shape {
            SHAPE_RIBBON => 1.05 + self.rnd() * 0.25,
            SHAPE_DROP => 0.55 + self.rnd() * 0.22,
            SHAPE_DUST => 0.9 + self.rnd() * 0.35,
            _ => 0.85 + self.rnd() * 0.25,
        };
        let spin_rate = match shape {
            SHAPE_RIBBON => 0.28,
            SHAPE_DUST => 0.9,
            SHAPE_DROP => 0.4,
            _ => 1.4 + drive * 0.6,
        };
        self.push_shaped(
            pos,
            vel,
            life,
            volume,
            shape,
            face,
            moisture,
            throw.sample.darken,
            kind_of(shape),
            throw.slip,
            throw.speed,
            spin_rate,
        );
    }

    fn push_shaped(
        &mut self,
        pos: Vec3,
        vel: Vec3,
        life: f32,
        volume: f32,
        shape: u8,
        face: u8,
        wet: f32,
        dark: f32,
        kind: u8,
        slip: f32,
        speed: f32,
        spin_rate: f32,
    ) {
        let seed = self.rng;
        let _ = self.rnd();
        let span = span_for(shape, volume.max(1.0e-9), slip, speed, wet);
        let roll = match shape {
            SHAPE_RIBBON => (self.rnd() - 0.5) * 0.2,
            SHAPE_DUST => (self.rnd() - 0.5) * 0.7,
            _ => (self.rnd() - 0.5) * 0.7,
        };
        // A ribbon stays along the peel. Falling velocity would stand it on end.
        let mut aim = vel;
        if shape == SHAPE_RIBBON {
            aim.y *= 0.12;
        } else if shape == SHAPE_DROP {
            aim.y *= 0.22;
        } else if shape == SHAPE_DUST {
            // The puff is thin along local Y. Keep that axis upright so the
            // dust lies in the wind instead of standing up as cards.
            aim.y = 0.0;
            if aim.length_squared() < 1.0e-6 {
                aim = Vec3::Z;
            }
        }
        let rot = flight_frame(aim, roll);
        let spin = Vec3::new(
            (self.rnd() - 0.5) * spin_rate,
            (self.rnd() - 0.5) * spin_rate,
            (self.rnd() - 0.5) * spin_rate,
        );
        let piece = Particle {
            pos,
            vel,
            life,
            max_life: life,
            size: span.max_element().max(0.01),
            kind,
            alive: true,
            seed,
            spin,
            rot,
            grounded: false,
            deposited: false,
            volume,
            span,
            shape,
            face,
            wet: wet.clamp(0.0, 1.0),
            dark: dark.clamp(0.0, 1.0),
        };
        if self.items.len() > 900 {
            if let Some(slot) = self.items.iter_mut().find(|p| !p.alive || p.life <= 0.0) {
                *slot = piece;
            }
            return;
        }
        self.items.push(piece);
    }

    pub fn update(&mut self, soil: &mut crate::soil::SoilField, dt: f32) {
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
                0 => 0.9,
                1 => 0.45,
                3 => 0.22,
                4 => 0.55,
                5 => 0.4,
                6 => 1.3,
                _ => 0.4,
            };
            p.vel.y -= 9.81 * dt;
            p.vel *= (1.0 - dt * drag).max(0.0);
            p.pos += p.vel * dt;
            p.rot = Quat::from_scaled_axis(p.spin * dt) * p.rot;
            let h = soil.sample(p.pos.x, p.pos.z).height;
            let reach = p.span.x.min(p.span.y).min(p.span.z).max(0.004);
            let belly = match p.shape {
                SHAPE_RIBBON => 0.45,
                SHAPE_DUST => 0.25,
                SHAPE_DROP => 0.7,
                _ => 0.9,
            };
            let sit = h + reach * belly;
            if p.pos.y <= sit {
                // Tread pieces are the cut itself. Landing them must not feed
                // another loose or wetness term back into the stamp.
                if !p.deposited && matches!(p.kind, 2 | 3 | 4) {
                    if let Some(idx) = soil.idx_at(p.pos.x, p.pos.z) {
                        match p.kind {
                            4 => {
                                soil.wetness[idx] = (soil.wetness[idx] + 0.03).min(1.0);
                                soil.darken[idx] = (soil.darken[idx] + 0.02).min(1.0);
                            }
                            2 | 3 => {
                                soil.darken[idx] = (soil.darken[idx] + 0.05).min(1.0);
                                soil.loose[idx] = (soil.loose[idx] + 0.008).min(0.16);
                                soil.wetness[idx] = (soil.wetness[idx] + 0.1).min(1.0);
                            }
                            _ => {}
                        }
                    }
                }
                p.deposited = true;
                p.grounded = true;
                p.pos.y = sit;
                p.vel = Vec3::ZERO;
                p.spin = Vec3::ZERO;
            }
        }
    }
}

/// Wet ground throws ribbons and drops. Dry ground stays dust and clods.
fn shape_budget(face: u8, moisture: f32) -> [f32; 4] {
    // clod, ribbon, drop, dust
    if face == FACE_MUD || moisture > 0.55 {
        let drop = if moisture > 0.7 { 0.34 } else { 0.26 };
        [0.0, 1.0 - drop, drop, 0.0]
    } else {
        let dust = if face == FACE_GRAVEL || moisture < 0.28 {
            0.38
        } else {
            0.22
        };
        [1.0 - dust, 0.0, 0.0, dust]
    }
}

/// A piece stops taking new cut once it is large enough to read as one chunk.
fn absorb_cap(shape: u8) -> f32 {
    match shape {
        SHAPE_RIBBON => 2.4e-2,
        SHAPE_DROP => 7.0e-4,
        SHAPE_DUST => 3.2e-4,
        _ => 1.4e-2,
    }
}

fn piece_count(part: f32, slip: f32, speed: f32, shape: u8) -> u32 {
    if shape == SHAPE_RIBBON || shape == SHAPE_CLOD {
        return 1;
    }
    let target = match shape {
        SHAPE_DROP => 1.6e-4 * (1.0 + slip.abs() * 0.25),
        SHAPE_DUST => 5.5e-5,
        _ => 2.2e-4 * (1.0 + slip.abs() * 0.35),
    };
    let mut n = ((part / target).round() as u32).clamp(1, 2);
    if speed > 4.5 && shape == SHAPE_DUST {
        n = n.max(2);
    }
    n
}

fn kind_of(shape: u8) -> u8 {
    match shape {
        SHAPE_RIBBON => 5,
        SHAPE_DROP => 0,
        SHAPE_DUST => 6,
        _ => 1,
    }
}

fn span_for(shape: u8, volume: f32, slip: f32, speed: f32, moisture: f32) -> Vec3 {
    let slip_n = slip.abs().min(1.5);
    let speed_n = (speed / 6.0).clamp(0.0, 1.0);
    match shape {
        SHAPE_RIBBON => {
            // A torn strap. Length follows speed; slip widens it; moisture
            // keeps the section thick enough that it is not a card.
            let elong = (4.8 + speed_n * 1.2 + moisture * 0.5).clamp(4.6, 6.2);
            let wide = (2.4 + slip_n * 0.45).clamp(2.3, 3.1);
            let thick = (1.05 + moisture * 0.15).clamp(1.0, 1.25);
            fit_box(volume, Vec3::new(elong, wide, thick), 0.5)
        }
        SHAPE_DROP => {
            let elong = (2.5 + moisture * 0.25 + speed_n * 0.2).clamp(2.4, 3.1);
            fit_ellipsoid(volume, Vec3::new(elong, 1.05, 0.95), 0.55)
        }
        SHAPE_DUST => {
            let wide = 2.1 + speed_n * 0.9 + (1.0 - moisture) * 0.4;
            let tall = 0.62 + (1.0 - moisture) * 0.2;
            fit_box(volume, Vec3::new(wide, tall, 1.05), 0.85)
        }
        _ => {
            let long = 2.2 + slip_n * 0.35;
            let tall = 1.22 + moisture * 0.08;
            let thick = 0.92;
            fit_ellipsoid(volume, Vec3::new(long, tall, thick), 0.62)
        }
    }
}

fn fit_box(volume: f32, aspect: Vec3, fill: f32) -> Vec3 {
    let unit = (aspect.x * aspect.y * aspect.z * fill).max(1.0e-6);
    aspect * (volume.max(1.0e-12) / unit).cbrt()
}

fn fit_ellipsoid(volume: f32, radii: Vec3, fill: f32) -> Vec3 {
    let unit = ((4.0 / 3.0) * std::f32::consts::PI * radii.x * radii.y * radii.z * fill).max(1.0e-6);
    radii * (volume.max(1.0e-12) / unit).cbrt()
}

fn flight_frame(dir: Vec3, roll: f32) -> Quat {
    let mut x = dir.normalize_or_zero();
    if x.length_squared() < 1.0e-6 {
        x = Vec3::X;
    }
    let up = if x.y.abs() > 0.92 { Vec3::Z } else { Vec3::Y };
    // Right-handed. local X is the throw. z = X × up, y = Z × X, so X × Y = Z.
    // Finishing with Y = X × Z flips the basis into a reflection, and the
    // quat extracted from that matrix does not point the mesh along the throw.
    let z0 = {
        let z = x.cross(up).normalize_or_zero();
        if z.length_squared() < 1.0e-6 {
            Vec3::Z
        } else {
            z
        }
    };
    let y0 = z0.cross(x).normalize_or_zero();
    let (s, c) = roll.sin_cos();
    let y = (y0 * c + z0 * s).normalize_or_zero();
    let z = (z0 * c - y0 * s).normalize_or_zero();
    Quat::from_mat3(&Mat3::from_cols(x, y, z)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soil::SoilField;

    fn loss_of(before: &[f32], after: &[f32], area: f32) -> f32 {
        before
            .iter()
            .zip(after)
            .map(|(a, b)| (a - b) * area)
            .sum()
    }

    #[test]
    fn ribbon_frame_follows_the_throw() {
        let dir = Vec3::new(-0.25, 1.4, -2.2);
        let aim = Vec3::new(dir.x, dir.y * 0.12, dir.z);
        let q = flight_frame(aim, 0.08);
        assert!(
            (q.length() - 1.0).abs() < 1.0e-4,
            "frame quat length {}",
            q.length()
        );
        let axis = q * Vec3::X;
        let y = q * Vec3::Y;
        let z = q * Vec3::Z;
        let aim_n = aim.normalize();
        assert!(
            axis.dot(aim_n) > 0.98,
            "long axis {axis:?} left the throw {aim_n:?}"
        );
        let det = axis.dot(y.cross(z));
        assert!((det - 1.0).abs() < 1.0e-3, "frame det {det}");
        assert!(axis.y.abs() < 0.25, "ribbon stood up {axis:?}");
    }

    #[test]
    fn drawn_volume_matches_the_wheel_cut() {
        let mut soil = SoilField::flat(0.75);
        let before = soil.height.clone();
        let area = soil.cell * soil.cell;
        let ejected = soil.stamp_wheel(
            Vec3::ZERO,
            0.38,
            0.28,
            Vec3::Z,
            Vec3::X,
            0.08,
            0.6,
            0.05,
            true,
        );
        let loss = loss_of(&before, &soil.height, area);
        assert!(ejected > 1.0e-5, "cut ejected {ejected}");
        assert!(
            (loss - ejected).abs() < 1.0e-5,
            "heightfield loss {loss} vs ejected {ejected}"
        );
        let sample = soil.sample(0.0, 0.0);
        let mut particles = Particles::new();
        particles.throw_tread(&TreadThrow {
            origin: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
            speed: 2.4,
            slip: 0.55,
            sample,
            volume: ejected,
            wheel_radius: 0.38,
            wheel_width: 0.28,
            outward: -1.0,
        });
        let drawn: f32 = particles
            .items
            .iter()
            .filter(|p| p.alive)
            .map(|p| p.volume)
            .sum();
        assert!(
            (drawn - ejected).abs() <= ejected * 1.0e-3 + 1.0e-8,
            "drawn {drawn} ejected {ejected}"
        );
        assert!(particles.alive_count() >= 2);
        for p in particles.items.iter().filter(|p| p.alive) {
            let mut e = [p.span.x, p.span.y, p.span.z];
            e.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(
                e[2] > e[0] * 1.7,
                "piece shape {} span {:?} is round",
                p.shape,
                p.span
            );
            assert!(
                p.shape == SHAPE_RIBBON || p.shape == SHAPE_DROP,
                "wet piece shape {}",
                p.shape
            );
            if p.shape == SHAPE_RIBBON {
                assert!(
                    p.span.z > p.span.x * 0.08,
                    "ribbon is a card {:?}",
                    p.span
                );
                let axis = (p.rot * Vec3::X).normalize_or_zero();
                assert!(
                    axis.z < -0.75 && axis.y.abs() < 0.35,
                    "ribbon axis {axis:?} is not the rearward peel"
                );
            }
            assert_eq!(p.face, FACE_MUD);
        }
        let shapes: Vec<u8> = particles
            .items
            .iter()
            .filter(|p| p.alive)
            .map(|p| p.shape)
            .collect();
        assert!(
            shapes.contains(&SHAPE_RIBBON) && shapes.contains(&SHAPE_DROP),
            "wet throw needs a ribbon and a drop, got {shapes:?}"
        );
        assert!(
            !shapes.contains(&SHAPE_CLOD) && !shapes.contains(&SHAPE_DUST),
            "wet throw is ribbons and drops, got {shapes:?}"
        );
    }

    #[test]
    fn dry_ground_is_dust_and_clods() {
        let soil = SoilField::flat(0.12);
        let sample = soil.sample(0.0, 0.0);
        assert_eq!(soil_face(&sample), FACE_GRAVEL);
        let mut particles = Particles::new();
        particles.throw_tread(&TreadThrow {
            origin: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
            speed: 4.0,
            slip: 0.15,
            sample,
            volume: 4.0e-4,
            wheel_radius: 0.38,
            wheel_width: 0.28,
            outward: 1.0,
        });
        let mut clod = false;
        let mut dust = false;
        for p in particles.items.iter().filter(|p| p.alive) {
            assert!(p.shape == SHAPE_CLOD || p.shape == SHAPE_DUST);
            assert_eq!(p.face, FACE_GRAVEL);
            clod |= p.shape == SHAPE_CLOD;
            dust |= p.shape == SHAPE_DUST;
        }
        assert!(clod && dust, "dry throw needs both a clod and dust");
    }

    #[test]
    fn clay_is_not_gravel_or_slurry() {
        let soil = SoilField::flat(0.5);
        let sample = soil.sample(0.0, 0.0);
        assert_eq!(soil_face(&sample), FACE_CLAY);
        assert!(moisture_of(&sample) > 0.2 && moisture_of(&sample) < 0.7);
    }

    #[test]
    fn impact_chunks_use_the_ejected_volume() {
        let mut soil = SoilField::flat(0.8);
        let before = soil.height.clone();
        let sample = soil.sample(0.0, 0.0);
        let dug = soil.stamp_impact(0.0, 0.0, 1.35, 0.16);
        let loss = loss_of(&before, &soil.height, soil.cell * soil.cell);
        assert!(
            (loss - dug.ejected).abs() < 1.0e-4,
            "loss {loss} ejected {}",
            dug.ejected
        );
        assert!(dug.depth > 0.05);
        let mut particles = Particles::new();
        let n = particles.impact_from_cut(Vec3::Y, sample, dug.ejected);
        assert!(n > 12, "spawned {n}");
        let drawn: f32 = particles.items.iter().map(|p| p.volume).sum();
        assert!((drawn - dug.ejected).abs() <= dug.ejected * 1.0e-3);
        let big = particles.items.iter().filter(|p| p.kind == 3).count();
        assert!(big >= 8, "impact clods {big}");
    }
}
