//! Solver frame cost. Release: `cargo test --release --test frame_bench -- --nocapture`.

use std::time::Instant;

use silt::{demo_drive, World};

/// Whole solver frame. Do not raise this.
const SOLVER_FRAME_BUDGET_MS: f64 = 7.0;

#[test]
fn baseline_solver_frame() {
    let mut world = World::demo();
    for _ in 0..45 {
        let input = demo_drive(&world);
        world.step(1.0 / 60.0, &input);
    }
    let frames = 180u32;
    let t0 = Instant::now();
    for _ in 0..frames {
        let input = demo_drive(&world);
        world.step(1.0 / 60.0, &input);
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);
    eprintln!(
        "SOLVER_FRAME ms={ms:.3} budget={SOLVER_FRAME_BUDGET_MS:.0} frames={frames} t={:.2} pos=({:.1},{:.1})",
        world.time, world.vehicle.pos.x, world.vehicle.pos.z
    );
    #[cfg(not(debug_assertions))]
    assert!(
        ms <= SOLVER_FRAME_BUDGET_MS,
        "SOLVER_FRAME {ms:.3} ms over {SOLVER_FRAME_BUDGET_MS}"
    );

    let mut soil = world.soil.clone();
    let frames = 120u32;
    let t0 = Instant::now();
    for _ in 0..frames {
        soil.step_water(1.0 / 60.0);
    }
    let water_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);
    eprintln!("SOLVER_WATER ms={water_ms:.3}");
}
