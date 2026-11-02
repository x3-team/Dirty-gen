//! H1 of docs/HYBRID_SIM.md: the 2 cm sheet under the tires keeps its detail
//! between steps. Same drive twice, with the near field on and off.
//!
//! Set `NEAR_DUMP=dir` to write the 2 cm depth of both runs as PGM images.

use silt::fluid_core::{FluidCore, TirePush, FINE_CELL_M, FINE_RES};

const DT: f32 = 1.0 / 30.0;
const SPEED: f32 = 2.0;

fn tires(z: f32) -> Vec<TirePush> {
    let mut out = Vec::new();
    for &(x, dz) in &[(-0.7f32, -1.2f32), (0.7, -1.2), (-0.7, 1.2), (0.7, 1.2)] {
        out.push(TirePush {
            x,
            z: z + dz,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.38,
            width: 0.2,
            speed: SPEED,
            slip: 0.15,
        });
    }
    out
}

/// Water on the sheets, parked off the window, and handed to spray, cubic metres.
fn water_book(fluid: &FluidCore) -> f32 {
    let b = fluid.ledger();
    b.water_m3 + b.frozen_water_m3 + b.spray_m3
}

struct Run {
    fine: Vec<f32>,
    /// Mean over wet 20 cm columns of the spread (max - min) of 2 cm depth, metres,
    /// in the patch the next step starts from.
    sub_spread: f32,
    water_m3: f32,
    start_m3: f32,
}

fn drive(near: bool) -> Run {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.set_near_field(near);
    fluid.pour_water(-3.0, 3.0, -2.0, 8.0, 0.08);
    let start_m3 = water_book(&fluid);
    let mut z = -1.0f32;
    for _ in 0..60 {
        fluid.recenter([0.0, z]);
        for t in tires(z) {
            fluid.push_tire(t);
        }
        fluid.step(DT);
        z += SPEED * DT;
    }
    let (fine, _) = fluid.fine_water_debug(&tires(z));
    let sub = (0.2 / FINE_CELL_M).round() as usize;
    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    for cj in 0..FINE_RES / sub {
        for ci in 0..FINE_RES / sub {
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for b in 0..sub {
                for a in 0..sub {
                    let w = fine[(cj * sub + b) * FINE_RES + ci * sub + a];
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
            }
            if hi > 1.0e-4 {
                sum += hi - lo;
                n += 1.0;
            }
        }
    }
    Run {
        fine,
        sub_spread: if n > 0.0 { sum / n } else { 0.0 },
        water_m3: water_book(&fluid),
        start_m3,
    }
}

fn dump(dir: &str, name: &str, fine: &[f32]) {
    let mut out = format!("P5\n{FINE_RES} {FINE_RES}\n255\n").into_bytes();
    for j in (0..FINE_RES).rev() {
        for i in 0..FINE_RES {
            let w = fine[j * FINE_RES + i];
            out.push((w / 0.16 * 255.0).clamp(0.0, 255.0) as u8);
        }
    }
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(format!("{dir}/{name}.pgm"), out).unwrap();
}

#[test]
fn near_field_keeps_sub_cell_detail() {
    let on = drive(true);
    let off = drive(false);
    println!(
        "NEAR_FIELD sub-cell spread on {:.5} m off {:.5} m water book on {:.6} off {:.6} start {:.6}",
        on.sub_spread, off.sub_spread, on.water_m3, off.water_m3, on.start_m3
    );
    if let Ok(dir) = std::env::var("NEAR_DUMP") {
        dump(&dir, "near-on", &on.fine);
        dump(&dir, "near-off", &off.fine);
    }
    assert!(
        on.sub_spread > 1.5 * off.sub_spread,
        "detail on {:.5} off {:.5}",
        on.sub_spread,
        off.sub_spread
    );
    for r in [&on, &off] {
        assert!(
            (r.water_m3 - r.start_m3).abs() < 1.0e-3 * r.start_m3,
            "water {} -> {}",
            r.start_m3,
            r.water_m3
        );
    }
}
