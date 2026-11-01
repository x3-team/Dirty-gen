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
    stretch_hold(t0);
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);
    eprintln!(
        "SOLVER_FRAME ms={ms:.3} budget={SOLVER_FRAME_BUDGET_MS:.0} frames={frames} t={:.2} pos=({:.1},{:.1})",
        world.time, world.vehicle.pos.x, world.vehicle.pos.z
    );
    let _ = std::io::Write::flush(&mut std::io::stderr());
    #[cfg(not(debug_assertions))]
    {
        let limit = release_limit(SOLVER_FRAME_BUDGET_MS, "PERF_LIMIT_SOLVER_FRAME");
        assert!(
            ms <= limit,
            "SOLVER_FRAME {ms:.3} ms over {limit}"
        );
    }

    let mut soil = world.soil.clone();
    let frames = 120u32;
    let t0 = Instant::now();
    for _ in 0..frames {
        soil.step_water(1.0 / 60.0);
    }
    let water_ms = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);
    eprintln!("SOLVER_WATER ms={water_ms:.3}");
}

fn stretch_hold(started: Instant) {
    let Ok(raw) = std::env::var("PERF_STRETCH") else {
        return;
    };
    let Ok(factor) = raw.parse::<f64>() else {
        return;
    };
    if !factor.is_finite() || factor <= 1.0 {
        return;
    }
    let extra = started.elapsed().mul_f64(factor - 1.0);
    let spin = Instant::now();
    while spin.elapsed() < extra {
        std::hint::spin_loop();
    }
}

#[cfg(not(debug_assertions))]
fn release_limit(cap: f64, key: &str) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|raw| raw.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(cap)
}
