//! Water and mud on the camera glass after a puddle hit.
//!
//! Round beads and a few gravity streaks. The shader refracts the frame through
//! them, so the scene stays readable instead of disappearing under brown spikes.
//! The isometric frame never draws this; the chase camera does.

pub const BEAD_CAP: usize = 28;
pub const RUN_CAP: usize = 16;

const MUD: f32 = 0.0;
const CLEAR: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
pub struct Bead {
    pub x: f32,
    pub y: f32,
    pub rx: f32,
    pub ry: f32,
    pub alpha: f32,
    pub seed: f32,
    pub gloss: f32,
    pub kind: f32,
    pub vy: f32,
}

impl Default for Bead {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            rx: 0.0,
            ry: 0.0,
            alpha: 0.0,
            seed: 0.0,
            gloss: 0.0,
            kind: 0.0,
            vy: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub x: f32,
    pub y: f32,
    pub len: f32,
    pub width: f32,
    pub alpha: f32,
    pub seed: f32,
    pub slant: f32,
    pub kind: f32,
    pub vy: f32,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            len: 0.0,
            width: 0.0,
            alpha: 0.0,
            seed: 0.0,
            slant: 0.0,
            kind: 0.0,
            vy: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuBead {
    pub xy: [f32; 2],
    pub radii: [f32; 2],
    /// alpha, seed, gloss, kind (0 mud, 1 clear)
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuRun {
    pub xy: [f32; 2],
    /// length down the glass, half-width of the tail
    pub len_w: [f32; 2],
    /// alpha, seed, slant, kind
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLens {
    /// film, scroll, aspect, age
    pub head: [f32; 4],
    pub beads: [GpuBead; BEAD_CAP],
    pub runs: [GpuRun; RUN_CAP],
}

pub struct LensMud {
    beads: [Bead; BEAD_CAP],
    runs: [Run; RUN_CAP],
    film: f32,
    scroll: f32,
    age: f32,
    rng: u32,
    in_puddle: bool,
    dry: f32,
    trickle: f32,
    seen_impact: u32,
}

impl LensMud {
    pub fn new() -> Self {
        Self {
            beads: [Bead::default(); BEAD_CAP],
            runs: [Run::default(); RUN_CAP],
            film: 0.0,
            scroll: 0.0,
            age: 0.0,
            rng: 0x6E6_E731,
            in_puddle: false,
            dry: 1.0,
            trickle: 0.0,
            seen_impact: 0,
        }
    }

    /// `wet_wheels` is how many tires are in standing water. `speed` is the
    /// fastest of those vehicles. `impact_sprays` jumps when a landing throws mud.
    pub fn sync(&mut self, dt: f32, wet_wheels: u32, speed: f32, impact_sprays: u32) {
        self.apply(dt, wet_wheels, speed, impact_sprays);
        let mut left = dt.clamp(0.0, 0.5);
        while left > 1.0e-5 {
            let h = left.min(1.0 / 30.0);
            self.integrate(h);
            left -= h;
        }
    }

    pub fn pack(&self, aspect: f32) -> GpuLens {
        let mut gpu = GpuLens {
            head: [self.film, self.scroll, aspect.max(0.1), self.age],
            beads: [GpuBead {
                xy: [0.0; 2],
                radii: [0.0; 2],
                params: [0.0; 4],
            }; BEAD_CAP],
            runs: [GpuRun {
                xy: [0.0; 2],
                len_w: [0.0; 2],
                params: [0.0; 4],
            }; RUN_CAP],
        };
        for (i, bead) in self.beads.iter().enumerate() {
            gpu.beads[i] = GpuBead {
                xy: [bead.x, bead.y],
                radii: [bead.rx, bead.ry],
                params: [bead.alpha, bead.seed, bead.gloss, bead.kind],
            };
        }
        for (i, run) in self.runs.iter().enumerate() {
            gpu.runs[i] = GpuRun {
                xy: [run.x, run.y],
                len_w: [run.len, run.width],
                params: [run.alpha, run.seed, run.slant, run.kind],
            };
        }
        gpu
    }

    pub fn film(&self) -> f32 {
        self.film
    }

    pub fn bead_count(&self) -> usize {
        self.beads.iter().filter(|b| b.alpha > 0.03).count()
    }

    pub fn run_count(&self) -> usize {
        self.runs.iter().filter(|r| r.alpha > 0.03).count()
    }

    pub fn mud_beads(&self) -> usize {
        self.beads
            .iter()
            .filter(|b| b.alpha > 0.03 && b.kind < 0.5)
            .count()
    }

    pub fn clear_beads(&self) -> usize {
        self.beads
            .iter()
            .filter(|b| b.alpha > 0.03 && b.kind >= 0.5)
            .count()
    }

    pub fn mud_runs(&self) -> usize {
        self.runs
            .iter()
            .filter(|r| r.alpha > 0.03 && r.kind < 0.5)
            .count()
    }

    pub fn mean_run_y(&self) -> f32 {
        let mut n = 0.0;
        let mut y = 0.0;
        for run in &self.runs {
            if run.alpha > 0.03 {
                y += run.y;
                n += 1.0;
            }
        }
        if n == 0.0 {
            0.0
        } else {
            y / n
        }
    }

    pub fn mean_abs_slant(&self) -> f32 {
        let mut n = 0.0;
        let mut s = 0.0;
        for run in &self.runs {
            if run.alpha > 0.03 {
                s += run.slant.abs();
                n += 1.0;
            }
        }
        if n == 0.0 {
            0.0
        } else {
            s / n
        }
    }

    /// Sum of live alphas. A clean lens is near zero.
    pub fn mass(&self) -> f32 {
        let beads: f32 = self.beads.iter().map(|b| b.alpha.max(0.0)).sum();
        let runs: f32 = self.runs.iter().map(|r| r.alpha.max(0.0)).sum();
        beads + runs + self.film
    }

    fn apply(&mut self, dt: f32, wet_wheels: u32, speed: f32, impact_sprays: u32) {
        let wet_move = wet_wheels > 0 && speed > 0.4;
        if wet_move {
            self.dry = 0.0;
            if !self.in_puddle {
                self.in_puddle = true;
                self.splat(0.95 + (speed * 0.035).min(0.2), true);
                self.trickle = 0.42;
            } else {
                self.trickle -= dt;
                if self.trickle <= 0.0 {
                    self.splat(0.34, false);
                    self.trickle = 0.48;
                }
            }
        } else {
            self.dry += dt;
            if self.dry > 0.28 {
                self.in_puddle = false;
            }
        }
        if impact_sprays > self.seen_impact {
            self.splat(1.05, true);
            self.seen_impact = impact_sprays;
        }
    }

    fn splat(&mut self, power: f32, force: bool) {
        let p = power.clamp(0.2, 1.25);
        self.film = (self.film + 0.5 * p.min(1.0)).min(1.0);
        // A hard hit leaves a few streaks and a scatter of round drops.
        // Trickles while still in the water only add a bead or two.
        let heavy = p > 0.7;
        let mud_runs = if heavy { 4 } else { 1 };
        let clear_runs = if heavy { 3 } else { 1 };
        let mud_beads = if heavy {
            (5.0 + p * 2.0).round() as usize
        } else {
            2
        };
        let clear_beads = if heavy {
            (8.0 + p * 3.0).round() as usize
        } else {
            3
        };
        for i in 0..mud_runs {
            let slot = (i as f32 + 0.5) / mud_runs as f32;
            let x = (slot + (self.next_f() - 0.5) * 0.14).clamp(0.08, 0.92);
            let y = 0.03 + self.next_f() * 0.08;
            let len = 0.36 + self.next_f() * 0.12;
            let width = 0.010 + self.next_f() * 0.008;
            let slant = (self.next_f() - 0.5) * 0.5;
            let alpha = 0.9 + self.next_f() * 0.1;
            self.write_run(
                x,
                y,
                len,
                width,
                alpha,
                slant,
                MUD,
                0.07 + width * 1.6,
                force,
            );
        }
        for i in 0..clear_runs {
            let slot = (i as f32 + 0.2) / clear_runs as f32;
            let x = (slot + (self.next_f() - 0.5) * 0.12).clamp(0.06, 0.94);
            let y = 0.02 + self.next_f() * 0.06;
            let len = 0.34 + self.next_f() * 0.12;
            let width = 0.0065 + self.next_f() * 0.0045;
            let slant = (self.next_f() - 0.5) * 0.22;
            let alpha = 0.72 + self.next_f() * 0.25;
            let vy = 0.08 + self.next_f() * 0.04;
            self.write_run(x, y, len, width, alpha, slant, CLEAR, vy, force);
        }
        for _ in 0..mud_beads {
            let x = 0.05 + self.next_f() * 0.9;
            let y = 0.08 + self.next_f() * 0.82;
            let mut size = 0.02 + self.next_f() * 0.03;
            if self.next_f() < 0.28 {
                size *= 1.45;
            }
            let ry = size * (0.92 + self.next_f() * 0.16);
            let alpha = 0.9 + self.next_f() * 0.1;
            let gloss = 0.55 + self.next_f() * 0.4;
            self.write_bead(x, y, size, ry, alpha, gloss, MUD, 0.02 + size * 0.45, force);
        }
        for _ in 0..clear_beads {
            let x = 0.04 + self.next_f() * 0.92;
            let y = 0.04 + self.next_f() * 0.9;
            let mut size = 0.012 + self.next_f() * 0.024;
            if self.next_f() < 0.22 {
                size *= 1.65;
            }
            let ry = size * (0.94 + self.next_f() * 0.12);
            let alpha = 0.88 + self.next_f() * 0.12;
            let gloss = 0.78 + self.next_f() * 0.22;
            self.write_bead(
                x,
                y,
                size,
                ry,
                alpha,
                gloss,
                CLEAR,
                0.015 + size * 0.4,
                force,
            );
        }
    }

    fn write_bead(
        &mut self,
        x: f32,
        y: f32,
        rx: f32,
        ry: f32,
        alpha: f32,
        gloss: f32,
        kind: f32,
        vy: f32,
        force: bool,
    ) {
        let Some(i) = self.slot_bead(force) else {
            return;
        };
        self.beads[i] = Bead {
            x,
            y,
            rx,
            ry,
            alpha,
            seed: self.next_f(),
            gloss,
            kind,
            vy,
        };
    }

    fn write_run(
        &mut self,
        x: f32,
        y: f32,
        len: f32,
        width: f32,
        alpha: f32,
        slant: f32,
        kind: f32,
        vy: f32,
        force: bool,
    ) {
        let Some(i) = self.slot_run(force) else {
            return;
        };
        self.runs[i] = Run {
            x,
            y,
            len,
            width,
            alpha,
            seed: self.next_f(),
            slant,
            kind,
            vy,
        };
    }

    fn slot_bead(&self, force: bool) -> Option<usize> {
        if let Some(i) = self.beads.iter().position(|b| b.alpha < 0.03) {
            return Some(i);
        }
        if !force {
            return None;
        }
        self.beads
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.alpha
                    .partial_cmp(&b.1.alpha)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    fn slot_run(&self, force: bool) -> Option<usize> {
        if let Some(i) = self.runs.iter().position(|r| r.alpha < 0.03) {
            return Some(i);
        }
        if !force {
            return None;
        }
        self.runs
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.alpha
                    .partial_cmp(&b.1.alpha)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    fn integrate(&mut self, h: f32) {
        self.age += h;
        self.scroll = (self.scroll + h * 0.05).rem_euclid(1.0);
        self.film = (self.film - h * 0.09).max(0.0);
        for bead in &mut self.beads {
            if bead.alpha < 0.03 {
                continue;
            }
            // Gravity slides a bead. It does not stretch it into a spike.
            bead.y += bead.vy * h;
            bead.x = (bead.x + (bead.seed * 12.0).sin() * 0.008 * h).clamp(0.02, 0.98);
            let fade = if bead.kind < 0.5 { 0.1 } else { 0.14 };
            bead.alpha = (bead.alpha - h * fade).max(0.0);
            if bead.y > 1.06 || bead.alpha < 0.03 {
                bead.alpha = 0.0;
            }
        }
        for run in &mut self.runs {
            if run.alpha < 0.03 {
                continue;
            }
            run.y += run.vy * h;
            run.slant *= 1.0 - (h * 0.45).min(0.25);
            run.len = (run.len + h * 0.01).min(0.58);
            let fade = if run.kind < 0.5 { 0.11 } else { 0.14 };
            run.alpha = (run.alpha - h * fade).max(0.0);
            if run.y > 1.02 || run.alpha < 0.03 {
                run.alpha = 0.0;
            }
        }
    }

    fn next_f(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        if x == 0 {
            x = 0xA5A5_1357;
        }
        self.rng = x;
        (x >> 8) as f32 / 16_777_216.0
    }
}

impl Default for LensMud {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_puddle_hit_puts_mud_streaks_and_clear_beads_on_the_glass() {
        let mut lens = LensMud::new();
        lens.sync(1.0 / 60.0, 2, 3.4, 0);
        assert!(lens.mud_runs() >= 4, "mud streaks {}", lens.mud_runs());
        assert!(lens.mud_beads() >= 3, "mud drops {}", lens.mud_beads());
        assert!(
            lens.clear_beads() >= 4,
            "clear beads {}",
            lens.clear_beads()
        );
        assert!(lens.run_count() > lens.mud_runs());
        assert!(lens.film() > 0.3, "film {}", lens.film());
        let gpu = lens.pack(16.0 / 9.0);
        assert!(gpu.head[0] > 0.3);
        assert!((gpu.head[2] - 16.0 / 9.0).abs() < 1.0e-4);
        assert!(gpu
            .runs
            .iter()
            .any(|r| r.params[0] > 0.5 && r.params[3] < 0.5));
        assert!(gpu
            .beads
            .iter()
            .any(|b| b.params[3] > 0.5 && b.params[0] > 0.3));
        assert!(
            lens.mud_runs() <= 6,
            "a few mud streaks, got {}",
            lens.mud_runs()
        );
        for bead in &gpu.beads {
            if bead.params[0] > 0.03 && bead.radii[1] > 1.0e-4 {
                let ratio = bead.radii[0] / bead.radii[1];
                assert!(
                    (0.8..1.25).contains(&ratio),
                    "bead should stay round, rx/ry {ratio}"
                );
            }
        }
        for run in &gpu.runs {
            if run.params[0] > 0.03 {
                assert!(
                    run.len_w[0] > run.len_w[1] * 6.0,
                    "streak should be thin, len {} width {}",
                    run.len_w[0],
                    run.len_w[1]
                );
                assert!(run.len_w[0] < 0.62, "streak too long {}", run.len_w[0]);
            }
        }
    }

    #[test]
    fn sitting_in_the_water_does_not_coat_the_lens() {
        let mut lens = LensMud::new();
        for _ in 0..40 {
            lens.sync(1.0 / 60.0, 3, 0.08, 0);
        }
        assert_eq!(lens.bead_count(), 0);
        assert_eq!(lens.run_count(), 0);
        assert!(lens.mass() < 0.02, "mass {}", lens.mass());
    }

    #[test]
    fn streaks_drip_down_then_leave_the_glass() {
        let mut lens = LensMud::new();
        lens.sync(1.0 / 60.0, 2, 4.0, 0);
        let y0 = lens.mean_run_y();
        let slant0 = lens.mean_abs_slant();
        assert!(
            slant0 > 0.02,
            "expected a few diagonal streaks, slant {slant0}"
        );
        for _ in 0..120 {
            lens.sync(1.0 / 60.0, 0, 0.0, 0);
        }
        let y1 = lens.mean_run_y();
        let slant1 = lens.mean_abs_slant();
        assert!(y1 > y0 + 0.04, "streaks should fall {y0} -> {y1}");
        assert!(slant1 < slant0, "gravity straightens {slant0} -> {slant1}");
        for _ in 0..60 * 16 {
            lens.sync(1.0 / 60.0, 0, 0.0, 0);
        }
        assert!(
            lens.mass() < 0.08,
            "glass should clear, mass {}",
            lens.mass()
        );
    }

    #[test]
    fn a_landing_coats_the_lens_again() {
        let mut lens = LensMud::new();
        lens.sync(0.02, 0, 2.0, 0);
        assert_eq!(lens.run_count(), 0);
        lens.sync(0.02, 0, 2.0, 3);
        let mass = lens.mass();
        assert!(mass > 1.0, "impact mass {mass}");
        for _ in 0..90 {
            lens.sync(1.0 / 60.0, 0, 0.0, 3);
        }
        let faded = lens.mass();
        lens.sync(1.0 / 60.0, 0, 1.0, 4);
        assert!(lens.mass() > faded + 0.4, "{} -> {}", faded, lens.mass());
    }
}
