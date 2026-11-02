//! The 2 cm water window of the render contract (v12, binding 16): the
//! near-field residual published around the tires keeps each 20 cm column's
//! volume and carries sub-cell detail; without the near field it is flat,
//! and without tires there is no window.

use silt::fluid_core::{FluidCore, TirePush, WINDOW_RES};
use silt::render_contract::{
    publish, ContractFrame, VALID_WATER_FINE, WATER_FINE_CELLS, WATER_FINE_CELL_M, WINDOW_CELL_M,
};
use silt::soil::SoilField;

fn drive_through_pool(near: bool) -> (FluidCore, ContractFrame) {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.set_near_field(near);
    fluid.pour_water(-2.0, 2.0, -2.0, 2.0, 0.05);
    let mut z = -1.0;
    for _ in 0..40 {
        fluid.push_tire(TirePush {
            x: 0.0,
            z,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.36,
            width: 0.22,
            speed: 3.0,
            slip: 0.3,
        });
        let _ = fluid.step(1.0 / 60.0);
        z += 3.0 / 60.0;
    }
    let soil = SoilField::flat(0.2);
    let mut frame = ContractFrame::new();
    publish(&mut frame, &soil, &[], [0.0, 0.0], 0.0, 0, Some(&fluid), None);
    (fluid, frame)
}

/// Texels of one 20 cm column inside the 2 cm window, and its film index.
fn fine_column_stats(frame: &ContractFrame) -> Vec<(usize, f64, f64)> {
    let h = &frame.header;
    let n = h.water_fine_res as usize;
    let per = (WINDOW_CELL_M / h.water_fine_cell_m).round() as usize;
    let mut out = Vec::new();
    for bj in 0..n / per {
        for bi in 0..n / per {
            let x = h.water_fine_origin_x + (bi * per) as f32 * h.water_fine_cell_m + 0.01;
            let z = h.water_fine_origin_z + (bj * per) as f32 * h.water_fine_cell_m + 0.01;
            let ci = ((x - h.origin_x) / h.cell_m).floor() as usize;
            let cj = ((z - h.origin_z) / h.cell_m).floor() as usize;
            let cidx = cj * WINDOW_RES + ci;
            let (mut sum, mut sq) = (0.0f64, 0.0f64);
            for b in 0..per {
                for a in 0..per {
                    let k = ((bj * per + b) * n + bi * per + a) * 2;
                    let v = f64::from(frame.water_fine[k]);
                    sum += v;
                    sq += v * v;
                }
            }
            let m = sum / (per * per) as f64;
            out.push((cidx, m, (sq / (per * per) as f64 - m * m).max(0.0).sqrt()));
        }
    }
    out
}

#[test]
fn water_fine_window_keeps_the_column_volume_and_carries_detail() {
    let (_, frame) = drive_through_pool(true);
    let h = &frame.header;
    assert_eq!(h.valid & VALID_WATER_FINE, VALID_WATER_FINE);
    assert_eq!(h.water_fine_res, WATER_FINE_CELLS);
    assert!((h.water_fine_cell_m - WATER_FINE_CELL_M).abs() < 1.0e-6);
    assert!(frame.water_fine.iter().all(|v| v.is_finite() && *v >= 0.0));
    let stats = fine_column_stats(&frame);
    let mut detailed = 0;
    for &(cidx, mean, sd) in &stats {
        let coarse = f64::from(frame.film[cidx * 2]);
        // The residual is zero-sum: 10 x 10 texels average to the column.
        // A column whose floor clips (a dry tread line) can sit a hair over.
        assert!(
            (mean - coarse).abs() <= 2.0e-4 + coarse * 0.02,
            "column {cidx}: fine mean {mean} vs film {coarse}"
        );
        if sd > 1.0e-3 {
            detailed += 1;
        }
    }
    assert!(detailed >= 4, "no 2 cm detail kept: {detailed} columns");
}

#[test]
fn water_fine_is_flat_without_the_near_field_and_off_without_tires() {
    let (_, frame) = drive_through_pool(false);
    assert_eq!(frame.header.valid & VALID_WATER_FINE, VALID_WATER_FINE);
    for (_, _, sd) in fine_column_stats(&frame) {
        assert!(sd < 1.0e-6, "flat patch has detail {sd}");
    }
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.pour_water(-1.0, 1.0, -1.0, 1.0, 0.05);
    let _ = fluid.step(1.0 / 60.0);
    let soil = SoilField::flat(0.2);
    let mut frame = ContractFrame::new();
    publish(&mut frame, &soil, &[], [0.0, 0.0], 0.0, 0, Some(&fluid), None);
    let h = &frame.header;
    assert_eq!(h.valid & VALID_WATER_FINE, 0);
    assert_eq!((h.water_fine_res, h.water_fine_cell_m), (0, 0.0));
    assert!(frame.water_fine.iter().all(|v| *v == 0.0));
}

