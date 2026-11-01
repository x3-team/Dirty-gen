//! A wheel clouds a still pond, then the cloud settles.
//!
//! Contract binding 4 A is `(susp + turbid) / water`. Both columns settle in
//! still water, and both ledgers close while they do.

use silt::fluid_core::{FluidCore, TirePush};
use silt::render_contract::{publish, ContractFrame, FLOW_CHANNELS, VALID_TURBIDITY};
use silt::soil::SoilField;

const DT: f32 = 1.0 / 30.0;

fn turbidity_stats(fluid: &FluidCore, soil: &SoilField, t: f32) -> (f32, f32) {
    let mut frame = ContractFrame::new();
    publish(&mut frame, soil, &[], [0.0, 0.0], t, 16, Some(fluid), None);
    assert!(frame.header.valid & VALID_TURBIDITY != 0);
    let mut max = 0.0f32;
    let mut sum = 0.0f32;
    for px in frame.flow.chunks_exact(FLOW_CHANNELS) {
        let a = px[3];
        assert!(a.is_finite() && (0.0..=1.0).contains(&a), "turbidity {a}");
        max = max.max(a);
        sum += a;
    }
    (max, sum)
}

#[test]
fn wheel_cloud_settles_and_ledgers_close() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    let poured = fluid.pour_water(-2.4, 2.4, -3.2, 3.4, 0.10);
    assert!(poured > 0.1, "pond {poured}");
    let soil = SoilField::flat(0.05);

    let mut z = -3.0_f32;
    let mut t = 0.0_f32;
    while z < 2.4 {
        fluid.push_tire(TirePush {
            x: 0.0,
            z,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.36,
            width: 0.22,
            speed: 3.0,
            slip: 0.8,
        });
        fluid.step(DT);
        z += 3.0 * DT;
        t += DT;
    }
    let (peak_max, peak_sum) = turbidity_stats(&fluid, &soil, t);
    assert!(peak_max > 0.05, "the wheel left no cloud: max {peak_max}");
    let stirred_peak = fluid.stirred_m3();

    // 40 s with no wheel.
    for _ in 0..(40.0 / DT) as usize {
        fluid.step(DT);
        t += DT;
    }
    let (late_max, late_sum) = turbidity_stats(&fluid, &soil, t);
    eprintln!(
        "turbidity max {peak_max:.3} -> {late_max:.3}, sum {peak_sum:.2} -> {late_sum:.2}, stirred {stirred_peak:.5} m3"
    );
    assert!(
        late_sum < peak_sum * 0.35,
        "still pond kept its cloud: {peak_sum} -> {late_sum}"
    );
    assert!(fluid.suspended_gap() <= 1.0e-6, "suspended gap {}", fluid.suspended_gap());
    assert!(fluid.silt_book_gap() <= 1.0e-6, "silt gap {}", fluid.silt_book_gap());
}

/// Mean published turbidity over the river channel, away from the inlet rows.
fn channel_turbidity(fluid: &FluidCore, soil: &SoilField, t: f32) -> f32 {
    let mut frame = ContractFrame::new();
    publish(&mut frame, soil, &[], [0.0, 0.0], t, 16, Some(fluid), None);
    let ox = frame.header.origin_x;
    let oz = frame.header.origin_z;
    let cell = silt::render_contract::WINDOW_CELL_M;
    let res = silt::render_contract::WINDOW_CELLS as usize;
    let (mut sum, mut n) = (0.0f64, 0u32);
    for j in 0..res {
        let z = oz + (j as f32 + 0.5) * cell;
        if !(silt::ford::CHANNEL_Z0 + 1.0..silt::ford::CHANNEL_Z1 - 1.0).contains(&z) {
            continue;
        }
        for i in 10..res - 10 {
            let x = ox + (i as f32 + 0.5) * cell;
            let _ = x;
            let idx = j * res + i;
            if frame.film[idx * 2] < 0.05 {
                continue;
            }
            sum += f64::from(frame.flow[idx * FLOW_CHANNELS + 3]);
            n += 1;
        }
    }
    assert!(n > 100, "channel cells {n}");
    (sum / f64::from(n)) as f32
}

#[test]
fn river_inlet_turbidity_is_a_parameter_and_closes() {
    let soil = SoilField::flat(0.0);
    for c in [0.0f32, 0.08] {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        silt::ford::seed_ford_turbid(&mut fluid, None, c);
        let mut t = 0.0;
        for _ in 0..(20.0 / DT) as usize {
            fluid.step(DT);
            t += DT;
        }
        let mean = channel_turbidity(&fluid, &soil, t);
        eprintln!("inlet {c}: channel turbidity {mean:.4}, gap {}", fluid.suspended_gap());
        if c == 0.0 {
            assert!(mean < 1.0e-4, "clear river published {mean}");
        } else {
            // Settling at 4 mm/s takes some out on the way down a 0.5 m channel.
            assert!(mean > 0.5 * c && mean <= c * 1.01, "brown river published {mean}");
        }
        let books = fluid.ledger();
        assert!(
            fluid.suspended_gap() <= 1.0e-4 * books.entrained_m3.max(1.0),
            "suspended gap {} of {}",
            fluid.suspended_gap(),
            books.entrained_m3
        );
    }
}
