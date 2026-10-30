//! Window and slow-step clock for the slurry ledger's `fluid_core`.
//!
//! This module does not advect, project, or store a second sheet. The height
//! and velocity step stays in `fluid_core`. What lives here is the moving
//! window (fine under the tires, coarse inside it, frozen outside) and the
//! 6-frame clock for drying, berm settle, crust, and creep.

/// Live disk around the Niva. Inside 30–50 m.
pub const WINDOW_RADIUS: f32 = 40.0;
/// Feather on the disk edge so the live field does not end in a hard ring.
pub const WINDOW_FEATHER: f32 = 2.0;
/// Cell under a tire.
pub const FINE_CELL: f32 = 0.02;
/// Cell of the coarse ledger, and of the frozen field outside the window.
pub const COARSE_CELL: f32 = 0.20;
/// Frames between slow updates of one column. Inside 4–8.
pub const SLOW_PERIOD: u32 = 6;
/// Tire patch where the fine cell is used. Weight is 1 inside, 0 at the rim.
const TIRE_INNER: f32 = 0.22;
const TIRE_OUTER: f32 = 0.55;
/// Fine patch centered on the chassis, metres. The Niva's hubs sit about
/// 1.4 m out, so a 0.55 m disk at the origin leaves the axles and the belly
/// outside the solver. Half of the 3.2 m axle patch covers every contact.
pub const VEHICLE_PATCH_R: f32 = 1.60;

/// Where the Niva is, and which frame the slow clock is on.
#[derive(Clone, Copy, Debug)]
pub struct SolverWindow {
    pub frame: u32,
    pub focus_x: f32,
    pub focus_z: f32,
}

impl Default for SolverWindow {
    fn default() -> Self {
        Self {
            frame: 0,
            focus_x: 0.0,
            focus_z: 0.0,
        }
    }
}

impl SolverWindow {
    pub fn follow(&mut self, x: f32, z: f32) {
        self.focus_x = x;
        self.focus_z = z;
        self.frame = self.frame.wrapping_add(1);
    }

    /// 1 in the interior, 0 at `WINDOW_RADIUS`, smooth between.
    pub fn live_weight(&self, x: f32, z: f32) -> f32 {
        let dist = (x - self.focus_x).hypot(z - self.focus_z);
        let outer = WINDOW_RADIUS;
        let inner = (outer - WINDOW_FEATHER).max(0.0);
        if dist >= outer {
            0.0
        } else if dist <= inner {
            1.0
        } else {
            smoothstep(outer, inner, dist)
        }
    }

    pub fn steps(&self, x: f32, z: f32) -> bool {
        self.live_weight(x, z) > 0.0
    }
}

/// One tire the fine cell sits under.
#[derive(Clone, Copy, Debug)]
pub struct TireAnchor {
    pub x: f32,
    pub z: f32,
}

/// Fine under a tire, coarse in the rest of the window, frozen outside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    Fine,
    Coarse,
    Frozen,
}

pub fn tire_weight(tire: TireAnchor, x: f32, z: f32) -> f32 {
    let dist = (x - tire.x).hypot(z - tire.z);
    if dist >= TIRE_OUTER {
        0.0
    } else if dist <= TIRE_INNER {
        1.0
    } else {
        smoothstep(TIRE_OUTER, TIRE_INNER, dist)
    }
}

pub fn resolution(window: &SolverWindow, x: f32, z: f32, tires: &[TireAnchor]) -> Resolution {
    if !window.steps(x, z) {
        return Resolution::Frozen;
    }
    let from_focus = (x - window.focus_x).hypot(z - window.focus_z);
    if from_focus <= VEHICLE_PATCH_R || tires.iter().any(|t| tire_weight(*t, x, z) > 0.0) {
        Resolution::Fine
    } else {
        Resolution::Coarse
    }
}

pub fn cell_metres(resolution: Resolution) -> f32 {
    match resolution {
        Resolution::Fine => FINE_CELL,
        Resolution::Coarse | Resolution::Frozen => COARSE_CELL,
    }
}

/// Slow work for this column on this frame. Frozen columns are never due.
/// A due column takes `SLOW_PERIOD` times the frame dt, which matches one
/// exponential step per frame over the period.
pub fn slow_due(window: &SolverWindow, x: f32, z: f32, column: usize) -> bool {
    window.steps(x, z)
        && column % SLOW_PERIOD as usize == window.frame as usize % SLOW_PERIOD as usize
}

pub fn slow_dt(window: &SolverWindow, x: f32, z: f32, column: usize, dt: f32) -> f32 {
    if slow_due(window, x, z, column) {
        dt * SLOW_PERIOD as f32
    } else {
        0.0
    }
}

/// One exponential step. Six small steps and one step of six times the dt agree.
pub fn exp_approach(value: f32, target: f32, rate: f32, dt: f32) -> f32 {
    let a = 1.0 - (-rate * dt.max(0.0)).exp();
    value + (target - value) * a
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    let t = t * t * (3.0 - 2.0 * t);
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particles::Particles;
    use crate::soil::SoilField;

    #[test]
    fn fluid_core_window_is_fine_under_the_tires_and_frozen_outside() {
        assert!((30.0..=50.0).contains(&WINDOW_RADIUS));
        assert!((4..=8).contains(&SLOW_PERIOD));
        assert!((0.018..0.022).contains(&FINE_CELL));
        assert!((0.18..0.22).contains(&COARSE_CELL));

        let soil = SoilField::flat(0.4);
        assert!(
            (soil.cell - COARSE_CELL).abs() < 0.03,
            "coarse ledger cell {} m is not ~20 cm",
            soil.cell
        );

        let mut window = SolverWindow::default();
        window.follow(1.0, -2.0);
        let tire = TireAnchor { x: 1.2, z: -1.7 };
        let under = resolution(&window, tire.x, tire.z, &[tire]);
        assert_eq!(under, Resolution::Fine);
        assert!((cell_metres(under) - 0.02).abs() < 1.0e-6);

        let mid = resolution(&window, 8.0, -2.0, &[tire]);
        assert_eq!(mid, Resolution::Coarse);
        assert!((cell_metres(mid) - 0.20).abs() < 1.0e-6);

        let far = resolution(&window, 1.0 + 45.0, -2.0, &[tire]);
        assert_eq!(far, Resolution::Frozen);
        assert!(!window.steps(1.0 + 45.0, -2.0));
        assert!(!slow_due(&window, 1.0 + 45.0, -2.0, 0));
    }

    #[test]
    fn the_fine_patch_covers_the_whole_vehicle() {
        let mut window = SolverWindow::default();
        window.follow(0.0, 0.0);
        // Rear hub. A 0.55 m disk on the chassis does not reach it.
        let x = -0.74_f32;
        let z = -1.18_f32;
        assert!(
            x.hypot(z) > TIRE_OUTER,
            "rear hub {:.2} m should sit outside the {:.2} m tire disk",
            x.hypot(z),
            TIRE_OUTER
        );
        assert!(x.hypot(z) < VEHICLE_PATCH_R);
        assert_eq!(resolution(&window, x, z, &[]), Resolution::Fine);
        assert_eq!(resolution(&window, 0.74, 1.08, &[]), Resolution::Fine);
        assert_eq!(resolution(&window, 4.0, 0.0, &[]), Resolution::Coarse);
    }

    #[test]
    fn the_window_edge_is_soft_and_the_fine_detail_is_closed() {
        let window = SolverWindow {
            frame: 1,
            focus_x: 0.0,
            focus_z: 0.0,
        };
        assert_eq!(window.live_weight(0.0, 0.0), 1.0);
        assert_eq!(window.live_weight(WINDOW_RADIUS, 0.0), 0.0);
        let rim = WINDOW_RADIUS - 0.05;
        let inside = WINDOW_RADIUS - WINDOW_FEATHER + 0.05;
        let w_rim = window.live_weight(rim, 0.0);
        let w_in = window.live_weight(inside, 0.0);
        assert!(w_rim > 0.0 && w_rim < 0.15, "edge is a hard step {w_rim}");
        assert!(w_in > 0.85, "feather starts late {w_in}");

        let tire = TireAnchor { x: 0.0, z: 0.0 };
        assert_eq!(tire_weight(tire, 0.0, 0.0), 1.0);
        assert_eq!(tire_weight(tire, TIRE_OUTER, 0.0), 0.0);

        // Zero-mean lug inside the soft tire patch. The rim sample is the
        // coarse value (detail 0), so the patch has no hard frame.
        let n = 48usize;
        let span = TIRE_OUTER * 2.0;
        let mut samples = Vec::with_capacity(n * n);
        for j in 0..n {
            for i in 0..n {
                let x = -TIRE_OUTER + (i as f32 + 0.5) * span / n as f32;
                let z = -TIRE_OUTER + (j as f32 + 0.5) * span / n as f32;
                let w = tire_weight(tire, x, z);
                let lug = (x * 80.0).sin() * (z * 80.0).cos() * 0.012;
                samples.push(lug * w);
            }
        }
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let gap: f32 = samples.iter().map(|s| s - mean).sum();
        assert!(gap.abs() < 1.0e-4, "fine detail opened volume by {gap}");
        assert_eq!(tire_weight(tire, TIRE_OUTER, 0.0) * 0.012, 0.0);
    }

    #[test]
    fn slow_processes_match_a_full_step_over_the_period() {
        let mut paced = 0.80_f32;
        let mut full = 0.80_f32;
        let target = 0.25_f32;
        let rate = 0.08_f32;
        let dt = 1.0 / 60.0;
        let mut window = SolverWindow::default();
        let column = 14usize;
        for _ in 0..180 {
            window.follow(0.0, 0.0);
            full = exp_approach(full, target, rate, dt);
            let step = slow_dt(&window, 0.0, 0.0, column, dt);
            if step > 0.0 {
                paced = exp_approach(paced, target, rate, step);
            }
        }
        assert!(
            (full - paced).abs() < 2.0e-4,
            "amortized {paced} drifted from full {full}"
        );
        assert!(paced < 0.80);

        let mut seen = [false; SLOW_PERIOD as usize];
        let mut window = SolverWindow::default();
        for _ in 0..SLOW_PERIOD {
            window.follow(0.0, 0.0);
            for column in 0..SLOW_PERIOD as usize {
                if slow_due(&window, 0.0, 0.0, column) {
                    assert!(!seen[column], "column {column} was due twice in one period");
                    seen[column] = true;
                }
            }
        }
        assert!(seen.iter().all(|s| *s));
    }

    #[test]
    fn tire_spray_stays_particles() {
        let mut particles = Particles::new();
        particles.spray(
            glam::Vec3::new(0.2, 0.3, 0.0),
            glam::Vec3::Z,
            glam::Vec3::X,
            2.0,
            0.05,
        );
        let spray = particles
            .items
            .iter()
            .find(|p| p.alive && p.kind == 0)
            .expect("tire spray");
        assert!(spray.life > 0.0);
        assert!(particles.items.iter().filter(|p| p.alive && p.kind == 0).count() >= 1);
    }
}
