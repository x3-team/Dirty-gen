//! Flow vector, flow speed, and a turbidity plume behind one wheel.
//!
//! The wheel crosses a pond of water and clay. Solids lift in the water aft of
//! the contact and advect with the film. Dry cells publish 0. The suspended
//! volume stays in the ledger. [`SOLVER_STEP_BUDGET_MS`] stays 7.0. A release
//! run asserts the wall time of that [`FluidCore::step`] — the clay pond is
//! inside the same call — and the water sheet inside it at 2 ms. The pond is
//! not a separate bench and not a known miss.

use std::time::Instant;

use silt::fluid_core::{
    FluidCore, Grain, TirePush, FILM_SPEED_CAP_MPS, SOLVER_STEP_BUDGET_MS,
};
use silt::render_contract::{
    publish, ContractFrame, FLOW_CHANNELS, RENDER_CONTRACT_VERSION, VALID_FLOW, VALID_TURBIDITY,
};
use silt::soil::SoilField;

const DT: f32 = 1.0 / 30.0;
const SPEED: f32 = 4.0;
/// Water sheet inside the timed [`FluidCore::step`]. Do not raise.
const WATER_STEP_BUDGET_MS: f64 = 2.0;

#[test]
fn turbidity_plume_closes() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.set_step_parts(true);
    let poured = fluid.pour_water(-2.4, 2.4, -3.2, 3.4, 0.08);
    assert!(poured > 0.1, "pond {poured}");
    // Same footprint, same step. The clay sheet is not a second bench.
    let clay = fluid.pour_slurry(-2.4, 2.4, -3.2, 3.4, 0.08, Grain::Clay, 0.55);
    assert!(clay > 0.1, "clay pond {clay}");
    assert!(fluid.suspended_gap() <= 1.0e-6, "gap before the wheel");

    let mut z = -4.4_f32;
    let mut warm = 0u32;
    let mut timed = 0.0_f64;
    let mut step_ms = 0.0_f64;
    let mut water_ms = 0.0_f64;
    let mut slurry_ms = 0.0_f64;
    let mut fluid_dust_ms = 0.0_f64;
    let mut suspended_ms = 0.0_f64;
    let mut prep_ms = 0.0_f64;
    let mut restrict_ms = 0.0_f64;
    let mut window_ms = 0.0_f64;
    let mut vx_min = f32::MAX;
    let mut vx_max = f32::MIN;
    let mut vz_min = f32::MAX;
    let mut vz_max = f32::MIN;
    let mut turb_min = f32::MAX;
    let mut turb_max = f32::MIN;
    let mut wet_cells = 0u64;
    let mut clamp_cells = 0u64;
    let mut capture: Option<ContractFrame> = None;
    let mut capture_z = 0.0_f32;
    let soil = SoilField::flat(0.05);

    while z < 2.2 {
        fluid.push_tire(TirePush {
            x: 0.0,
            z,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.36,
            width: 0.22,
            speed: SPEED,
            slip: 0.9,
        });
        let t0 = Instant::now();
        fluid.step(DT);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        if warm >= 8 {
            let parts = fluid.step_parts();
            step_ms += elapsed;
            water_ms += f64::from(parts.water_ms);
            slurry_ms += f64::from(parts.slurry_ms);
            fluid_dust_ms += f64::from(parts.dust_ms);
            suspended_ms += f64::from(parts.suspended_ms);
            prep_ms += f64::from(parts.prep_ms);
            restrict_ms += f64::from(parts.restrict_ms);
            window_ms += f64::from(parts.window_ms);
            timed += 1.0;
        }
        warm += 1;
        z += SPEED * DT;

        let mut frame = ContractFrame::new();
        publish(
            &mut frame,
            &soil,
            &[],
            [0.0, z],
            z,
            15,
            Some(&fluid),
            None,
        );
        let (vx0, vx1, vz0, vz1, t0, t1) = flow_bounds(&frame);
        if warm >= 8 {
            let (wet, clamped) = clamp_cells_of(&frame);
            wet_cells += wet;
            clamp_cells += clamped;
        }
        vx_min = vx_min.min(vx0);
        vx_max = vx_max.max(vx1);
        vz_min = vz_min.min(vz0);
        vz_max = vz_max.max(vz1);
        turb_min = turb_min.min(t0);
        turb_max = turb_max.max(t1);
        assert!(
            frame.header.valid & VALID_FLOW != 0 && frame.header.valid & VALID_TURBIDITY != 0,
            "flow bits {:#x}",
            frame.header.valid
        );
        assert_dry_zeros(&frame);
        if (0.15..0.55).contains(&z) {
            capture = Some(frame);
            capture_z = z;
        }
    }

    let gap = fluid.suspended_gap();
    let books = fluid.ledger();
    let step_mean = if timed > 0.0 { step_ms / timed } else { 0.0 };
    let water_mean = if timed > 0.0 { water_ms / timed } else { 0.0 };
    let slurry_mean = if timed > 0.0 { slurry_ms / timed } else { 0.0 };
    let fluid_dust_mean = if timed > 0.0 { fluid_dust_ms / timed } else { 0.0 };
    let suspended_mean = if timed > 0.0 { suspended_ms / timed } else { 0.0 };
    let prep_mean = if timed > 0.0 { prep_ms / timed } else { 0.0 };
    let restrict_mean = if timed > 0.0 { restrict_ms / timed } else { 0.0 };
    let window_mean = if timed > 0.0 { window_ms / timed } else { 0.0 };
    let budget_note = if step_mean <= f64::from(SOLVER_STEP_BUDGET_MS) {
        "met"
    } else {
        "NOT met"
    };
    let clamp_frac = if wet_cells > 0 {
        clamp_cells as f64 / wet_cells as f64
    } else {
        0.0
    };
    let frame = capture.expect("the wheel never crossed the middle of the pond");
    let (aft, ahead) = plume_means(&frame, capture_z);
    let (trail_w, trail_vx, trail_vz) = plume_trail(&frame, capture_z);
    write_plume_png(&frame, capture_z);

    println!(
        "vx {vx_min:.5}..{vx_max:.5} m/s  tire {SPEED:.1}\n\
         vz {vz_min:.5}..{vz_max:.5} m/s\n\
         clamp fraction {clamp_frac:.4} ({clamp_cells}/{wet_cells}) limiter ±{FILM_SPEED_CAP_MPS:.1}\n\
         turbidity {turb_min:.5}..{turb_max:.5}\n\
         suspended gap {gap:.8} m3  column {:.6} settled {:.6} parked {:.6} lifted {:.6}\n\
         fluid step {step_mean:.3} ms  budget {SOLVER_STEP_BUDGET_MS:.1} {budget_note}  samples {timed:.0}\n\
         water {water_mean:.3} budget {WATER_STEP_BUDGET_MS:.1}\n\
         clay pond {slurry_mean:.3}\n\
         fluid dust {fluid_dust_mean:.3} suspended {suspended_mean:.3} prep {prep_mean:.3} restrict {restrict_mean:.3} window {window_mean:.3}\n\
         hardware {}\n\
         plume aft {aft:.4} ahead {ahead:.4} at z {capture_z:.3}\n\
         trail width {trail_w:.3} m  mean |vx| {trail_vx:.4} |vz| {trail_vz:.4}",
        books.suspended_m3,
        books.settled_suspended_m3,
        books.frozen_suspended_m3,
        books.entrained_m3,
        hardware_line()
    );

    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(RENDER_CONTRACT_VERSION, 6);
    assert!(turb_max > 0.2, "turbidity max {turb_max}");
    assert!(turb_min >= 0.0 && turb_max <= 1.0, "turbidity {turb_min}..{turb_max}");
    // The tire at SPEED shoves along +Z at 0.85 of that speed and across the
    // track at 0.18. The wake has to follow the tire, and it cannot outrun it.
    // Cells sitting on the 1.6 m/s limiter are counted; that limiter is not the expected speed.
    assert!(
        trail_vz > trail_vx,
        "wake |vz| {trail_vz:.4} should beat lateral |vx| {trail_vx:.4}"
    );
    assert!(
        trail_vz > 0.4 && trail_vz < SPEED,
        "along-track film {trail_vz:.4} is not a {SPEED} m/s tire wake"
    );
    assert!(
        vx_min < -0.05 && vx_max > 0.05,
        "shoulders did not part, vx {vx_min}..{vx_max}"
    );
    assert!(
        vz_max > vx_max.abs(),
        "forward shove {vz_max} did not beat the lateral one {vx_max}"
    );
    assert!(
        vx_min > -SPEED && vx_max < SPEED && vz_min > -SPEED && vz_max < SPEED,
        "film outruns the {SPEED} m/s tire, vx {vx_min}..{vx_max} vz {vz_min}..{vz_max}"
    );
    assert!(wet_cells > 0, "no wet cells");
    assert!(
        clamp_frac < 0.5,
        "clamp fraction {clamp_frac:.4} owns the field ({clamp_cells}/{wet_cells})"
    );
    // Centerline trail on this sheet sits near 0.20 and the water ahead of the
    // tire stays clear. The contrast is the check; the floor keeps a speck from passing.
    assert!(
        aft > ahead + 0.15 && aft > 0.15,
        "plume aft {aft:.4} is not behind the wheel, ahead {ahead:.4}"
    );
    assert!(gap <= 1.0e-6, "suspended gap {gap}");
    assert!(
        (books.suspended_m3 + books.settled_suspended_m3 + books.frozen_suspended_m3
            - books.entrained_m3)
            .abs()
            <= 1.0e-6
    );
    assert!(books.entrained_m3 > 1.0e-4, "the wheel lifted nothing");

    fluid.recenter([40.0, 0.0]);
    let parked = fluid.suspended_gap();
    fluid.recenter([0.0, 0.0]);
    let back = fluid.suspended_gap();
    assert!(parked <= 1.0e-6, "gap after park {parked}");
    assert!(back <= 1.0e-6, "gap after restore {back}");

    assert_eq!(SOLVER_STEP_BUDGET_MS, 7.0, "do not raise the solver step budget");
    assert_eq!(WATER_STEP_BUDGET_MS, 2.0, "do not raise the water step budget");
    #[cfg(not(debug_assertions))]
    {
        assert!(
            step_mean > 0.0 && step_mean <= f64::from(SOLVER_STEP_BUDGET_MS),
            "fluid step {step_mean:.3} ms exceeds {SOLVER_STEP_BUDGET_MS}"
        );
        assert!(
            water_mean <= WATER_STEP_BUDGET_MS,
            "water {water_mean:.3} ms exceeds {WATER_STEP_BUDGET_MS}"
        );
        assert!(
            slurry_mean > 0.0 && slurry_mean <= f64::from(SOLVER_STEP_BUDGET_MS),
            "clay pond {slurry_mean:.3} ms exceeds {SOLVER_STEP_BUDGET_MS}"
        );
    }
}

fn flow_bounds(frame: &ContractFrame) -> (f32, f32, f32, f32, f32, f32) {
    let mut vx_min = f32::MAX;
    let mut vx_max = f32::MIN;
    let mut vz_min = f32::MAX;
    let mut vz_max = f32::MIN;
    let mut t_min = f32::MAX;
    let mut t_max = f32::MIN;
    let n = frame.header.window_res as usize * frame.header.window_res as usize;
    for idx in 0..n {
        let base = idx * FLOW_CHANNELS;
        let vx = frame.flow[base];
        let vz = frame.flow[base + 1];
        let speed = frame.flow[base + 2];
        let turb = frame.flow[base + 3];
        assert!(vx.is_finite() && vz.is_finite() && speed.is_finite() && turb.is_finite());
        assert!((0.0..=1.0).contains(&turb), "turbidity {turb}");
        let expect = if frame.film[idx * 2] + frame.film[idx * 2 + 1] < 1.0e-5 {
            0.0
        } else {
            vx.hypot(vz)
        };
        assert!(
            (speed - expect).abs() < 1.0e-4,
            "speed {speed} hypot {expect}"
        );
        vx_min = vx_min.min(vx);
        vx_max = vx_max.max(vx);
        vz_min = vz_min.min(vz);
        vz_max = vz_max.max(vz);
        t_min = t_min.min(turb);
        t_max = t_max.max(turb);
    }
    (vx_min, vx_max, vz_min, vz_max, t_min, t_max)
}

/// Wet cells whose velocity sits on the limiter. The limiter is not the expected speed.
fn clamp_cells_of(frame: &ContractFrame) -> (u64, u64) {
    let n = frame.header.window_res as usize * frame.header.window_res as usize;
    let edge = FILM_SPEED_CAP_MPS - 1.0e-3;
    let mut wet = 0u64;
    let mut clamped = 0u64;
    for idx in 0..n {
        let film = frame.film[idx * 2] + frame.film[idx * 2 + 1];
        if film < 1.0e-5 {
            continue;
        }
        wet += 1;
        let base = idx * FLOW_CHANNELS;
        let vx = frame.flow[base].abs();
        let vz = frame.flow[base + 1].abs();
        if vx >= edge || vz >= edge {
            clamped += 1;
        }
    }
    (wet, clamped)
}

fn assert_dry_zeros(frame: &ContractFrame) {
    let n = frame.header.window_res as usize * frame.header.window_res as usize;
    for idx in 0..n {
        let water = frame.film[idx * 2];
        let slurry = frame.film[idx * 2 + 1];
        let base = idx * FLOW_CHANNELS;
        if water + slurry < 1.0e-5 {
            assert_eq!(frame.flow[base], 0.0, "dry vx");
            assert_eq!(frame.flow[base + 1], 0.0, "dry vz");
            assert_eq!(frame.flow[base + 2], 0.0, "dry speed");
        }
        if water < 1.0e-5 {
            assert_eq!(frame.flow[base + 3], 0.0, "dry turbidity");
        }
    }
}

/// Width of the aft plume and the film speed inside it.
///
/// Turbidity is a donor-cell upwind of the coarse sheet. There is no
/// cross-wind diffusion, and the tire only lifts solids inside about one
/// coarse cell of the track. A small lateral speed therefore stays a stripe.
fn plume_trail(frame: &ContractFrame, wheel_z: f32) -> (f32, f32, f32) {
    let res = frame.header.window_res as usize;
    let cell = frame.header.cell_m;
    let mut cols = vec![false; res];
    let mut vx = 0.0f32;
    let mut vz = 0.0f32;
    let mut n = 0.0f32;
    for j in 0..res {
        for i in 0..res {
            let z = frame.header.origin_z + (j as f32 + 0.5) * cell;
            let dz = z - wheel_z;
            if !(-1.6..-0.25).contains(&dz) {
                continue;
            }
            let idx = j * res + i;
            let turb = frame.flow[idx * FLOW_CHANNELS + 3];
            if turb <= 0.05 {
                continue;
            }
            cols[i] = true;
            vx += frame.flow[idx * FLOW_CHANNELS].abs();
            vz += frame.flow[idx * FLOW_CHANNELS + 1].abs();
            n += 1.0;
        }
    }
    let mut run = 0usize;
    let mut best = 0usize;
    for on in cols {
        if on {
            run += 1;
            best = best.max(run);
        } else {
            run = 0;
        }
    }
    let width = best as f32 * cell;
    if n > 0.0 {
        (width, vx / n, vz / n)
    } else {
        (0.0, 0.0, 0.0)
    }
}

fn plume_means(frame: &ContractFrame, wheel_z: f32) -> (f32, f32) {
    let res = frame.header.window_res as usize;
    let cell = frame.header.cell_m;
    let mut aft = 0.0_f32;
    let mut aft_n = 0.0_f32;
    let mut ahead = 0.0_f32;
    let mut ahead_n = 0.0_f32;
    for j in 0..res {
        for i in 0..res {
            let x = frame.header.origin_x + (i as f32 + 0.5) * cell;
            let z = frame.header.origin_z + (j as f32 + 0.5) * cell;
            // The trail is one coarse cell wide, on the tire track.
            if x.abs() > 0.16 {
                continue;
            }
            let water = frame.film[(j * res + i) * 2];
            if water < 0.02 {
                continue;
            }
            let turb = frame.flow[(j * res + i) * FLOW_CHANNELS + 3];
            let dz = z - wheel_z;
            if (-1.6..-0.25).contains(&dz) {
                aft += turb;
                aft_n += 1.0;
            } else if (0.40..1.6).contains(&dz) {
                ahead += turb;
                ahead_n += 1.0;
            }
        }
    }
    (
        if aft_n > 0.0 { aft / aft_n } else { 0.0 },
        if ahead_n > 0.0 { ahead / ahead_n } else { 0.0 },
    )
}

fn write_plume_png(frame: &ContractFrame, wheel_z: f32) {
    let res = frame.header.window_res as i32;
    let cell = frame.header.cell_m;
    let mut i0 = res;
    let mut i1 = 0i32;
    let mut j0 = res;
    let mut j1 = 0i32;
    for j in 0..res {
        for i in 0..res {
            let idx = j as usize * res as usize + i as usize;
            let water = frame.film[idx * 2];
            let turb = frame.flow[idx * FLOW_CHANNELS + 3];
            let x = frame.header.origin_x + (i as f32 + 0.5) * cell;
            let z = frame.header.origin_z + (j as f32 + 0.5) * cell;
            let near = (x - 0.0).hypot(z - wheel_z) < 2.6;
            if turb > 0.02 || (water > 0.02 && near) {
                i0 = i0.min(i);
                i1 = i1.max(i);
                j0 = j0.min(j);
                j1 = j1.max(j);
            }
        }
    }
    if i0 > i1 {
        i0 = 0;
        i1 = res - 1;
        j0 = 0;
        j1 = res - 1;
    }
    i0 = (i0 - 2).max(0);
    j0 = (j0 - 2).max(0);
    i1 = (i1 + 2).min(res - 1);
    j1 = (j1 + 2).min(res - 1);
    let w_cells = (i1 - i0 + 1) as u32;
    let h_cells = (j1 - j0 + 1) as u32;
    let scale = (640 / w_cells.max(h_cells)).max(8);
    let width = w_cells * scale;
    let height = h_cells * scale;
    let mut img = image::RgbImage::new(width, height);
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j as usize * res as usize + i as usize;
            let water = frame.film[idx * 2];
            let turb = frame.flow[idx * FLOW_CHANNELS + 3].clamp(0.0, 1.0);
            let rgb = plume_color(water, turb);
            let px0 = (i - i0) as u32 * scale;
            // +Z is up in the image. Behind the wheel, which travels +Z, sits lower.
            let py0 = (j1 - j) as u32 * scale;
            for dy in 0..scale {
                for dx in 0..scale {
                    img.put_pixel(px0 + dx, py0 + dy, image::Rgb(rgb));
                }
            }
        }
    }
    stamp_wheel(&mut img, frame, wheel_z, i0, j1, scale);
    let path = std::env::var("TURBIDITY_PNG")
        .unwrap_or_else(|_| "target/turbidity-plume.png".to_string());
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    img.save(&path).unwrap_or_else(|err| panic!("png {path}: {err}"));
    println!("turbidity png {path}");
}

fn stamp_wheel(
    img: &mut image::RgbImage,
    frame: &ContractFrame,
    wheel_z: f32,
    i0: i32,
    j1: i32,
    scale: u32,
) {
    let cell = frame.header.cell_m.max(1.0e-4);
    let wx = (0.0 - frame.header.origin_x) / cell;
    let wz = (wheel_z - frame.header.origin_z) / cell;
    let cx = (wx - i0 as f32) * scale as f32;
    let cy = ((j1 as f32 + 1.0) - wz) * scale as f32;
    let radius = 0.36 / cell * scale as f32;
    let (w, h) = img.dimensions();
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f32 + 0.5) - cx).hypot((y as f32 + 0.5) - cy);
            if (d - radius).abs() < scale as f32 * 0.35 {
                img.put_pixel(x, y, image::Rgb([236, 236, 236]));
            }
        }
    }
}

fn hardware_line() -> String {
    let cpu = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let model = cpu
        .lines()
        .find(|line| line.starts_with("model name"))
        .map(|line| line.split_once(':').map(|(_, v)| v.trim()).unwrap_or(line))
        .unwrap_or("unknown cpu");
    let mhz = cpu
        .lines()
        .find(|line| line.starts_with("cpu MHz"))
        .map(|line| line.split_once(':').map(|(_, v)| v.trim()).unwrap_or(""))
        .unwrap_or("");
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let kind = if cfg!(debug_assertions) { "debug" } else { "release" };
    if mhz.is_empty() {
        format!("{model}, {threads} threads, {kind}")
    } else {
        format!("{model} @ {mhz} MHz, {threads} threads, {kind}")
    }
}

fn plume_color(water: f32, turb: f32) -> [u8; 3] {
    if water < 1.0e-4 && turb <= 0.0 {
        return [16, 20, 24];
    }
    let t = turb.clamp(0.0, 1.0);
    let depth = (water / 0.09).clamp(0.0, 1.0);
    let clear_r = 36.0 + 18.0 * depth;
    let clear_g = 78.0 + 24.0 * depth;
    let clear_b = 86.0 + 8.0 * depth;
    let mud_r = 186.0 - 70.0 * t;
    let mud_g = 104.0 - 48.0 * t;
    let mud_b = 48.0 - 20.0 * t;
    let r = clear_r * (1.0 - t) + mud_r * t;
    let g = clear_g * (1.0 - t) + mud_g * t;
    let b = clear_b * (1.0 - t) + mud_b * t;
    [r as u8, g as u8, b as u8]
}
