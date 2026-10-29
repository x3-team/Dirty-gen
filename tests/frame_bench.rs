//! Solver frame cost. Release: `cargo test --release --test frame_bench -- --nocapture`.

use std::time::Instant;

use silt::{demo_drive, World};

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
        "SOLVER_FRAME ms={ms:.3} frames={frames} t={:.2} pos=({:.1},{:.1})",
        world.time, world.vehicle.pos.x, world.vehicle.pos.z
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
