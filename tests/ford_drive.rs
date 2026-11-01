//! The Niva drives through the river ford of `silt::ford`, from the sand
//! beach on the south bank to the gravel bank on the north. The wheels and
//! the belly are walls in the shared sheet (`Vehicle::hydro_solids`), the
//! same as in the creek. What is checked is read back from the sheet: the
//! bow wave ahead of the front wheels, the wake behind, the waves the
//! crossing sends back onto the beach, and the water ledger.

use silt::fluid_core::{CoarseCell, FluidCore, COARSE_CELL_M, WINDOW_RES};
use silt::ford::{still_waterline_z, FordDrive, CHANNEL_Z0, CHANNEL_Z1};
use silt::particles::Particle;
use silt::render_contract::{publish, ContractFrame};

const DT: f32 = 1.0 / 60.0;

fn cell_at(fluid: &FluidCore, x: f32, z: f32) -> CoarseCell {
    let (ox, oz, _) = fluid.window_bounds();
    let i = ((x - ox) / COARSE_CELL_M + 1.0e-4)
        .floor()
        .clamp(0.0, (WINDOW_RES - 1) as f32) as usize;
    let j = ((z - oz) / COARSE_CELL_M + 1.0e-4)
        .floor()
        .clamp(0.0, (WINDOW_RES - 1) as f32) as usize;
    fluid.coarse_cell(i, j).expect("cell")
}

fn eta(c: &CoarseCell) -> f32 {
    c.bed_m + c.water_m
}

/// Southernmost wet cell along column `x`.
fn shoreline_z(fluid: &FluidCore, x: f32) -> f32 {
    let mut z = CHANNEL_Z0;
    while z > CHANNEL_Z0 - 9.0 {
        if cell_at(fluid, x, z).water_m < 0.003 {
            return z + COARSE_CELL_M;
        }
        z -= COARSE_CELL_M;
    }
    z
}

/// Free-surface range on the beach foot, z in [still + 0.4, still + 1.4],
/// for x within 4 m of the crossing lane.
fn beach_stir(fluid: &FluidCore, lo: &mut [f32], hi: &mut [f32]) {
    let still = still_waterline_z();
    for (k, x) in [-3.0f32, -1.5, 0.0, 1.5, 3.0].iter().enumerate() {
        let c = cell_at(fluid, *x, still + 0.9);
        if c.water_m > 0.003 {
            lo[k] = lo[k].min(eta(&c));
            hi[k] = hi[k].max(eta(&c));
        }
        let _ = shoreline_z(fluid, *x);
    }
}

struct Run {
    end_z: f32,
    max_speed: f32,
    bow: f32,
    bow_mean: f32,
    wake: f32,
    stir: f32,
    gap: f32,
    nans: u32,
    tire_foam: f32,
}

fn run(drive: bool, seconds: f32) -> Run {
    // Let the river settle before the car arrives.
    let mut ford = FordDrive::new(None, 4.0);
    if let Some(p) = std::env::var("PACE").ok().and_then(|v| v.parse().ok()) {
        ford.pace_m_s = p;
    }
    let held = |f: &FluidCore| {
        let l = f.ledger();
        // Slurry the wheels scoured is bed soil, booked as eroded volume.
        l.water_m3 + l.reach_m3 + l.slurry_m3 + l.spray_m3 + l.fog_air_m3 + l.frozen_water_m3
            - f.hydro_load().eroded_m3
    };
    let water0 = held(&ford.fluid);
    let mut lo = [f32::MAX; 5];
    let mut hi = [f32::MIN; 5];
    let mut bow = 0.0f32;
    let mut bow_sum = 0.0f32;
    let mut bow_n = 0.0f32;
    let mut wake = 0.0f32;
    let mut max_speed = 0.0f32;
    let mut tire_foam = 0.0f32;
    let mut frame = ContractFrame::new();
    let particles: &[Particle] = &[];
    let steps = (seconds / DT) as u32;
    for s in 0..steps {
        if !drive {
            ford.fluid.step(DT);
            beach_stir(&ford.fluid, &mut lo, &mut hi);
            continue;
        }
        ford.step(DT);
        let fluid = &ford.fluid;
        let vehicle = &ford.world.vehicle;
        let soil = &ford.world.soil;
        {
            let load = fluid.hydro_load();
            max_speed = max_speed.max(vehicle.speed());
            if std::env::var("TRACE").is_ok() && s % 30 == 0 {
                println!(
                    "TRACE t {:.1} z {:.2} y {:.2} v {:.2} fz {:.0} fx {:.0} hull_lift {:.0} wheel_lift {:.0} omega {:.1} pitch {:.2}",
                    s as f32 * DT, vehicle.pos.z, vehicle.pos.y, vehicle.speed(), load.force_z, load.force_x,
                    load.hull_buoyancy_n, load.wheel_buoyancy_n, vehicle.wheels[0].omega, vehicle.pitch
                );
            }
            // Bow and wake only while the car is in the channel.
            let ori = vehicle.orientation();
            if (CHANNEL_Z0 + 1.0..CHANNEL_Z1 - 1.0).contains(&vehicle.pos.z) {
                let amb = eta(&cell_at(&fluid, vehicle.pos.x - 4.0, vehicle.pos.z));
                for i in 0..2 {
                    let w = vehicle.pos + ori * vehicle.wheel_center_local(i);
                    let mut ahead = f32::MIN;
                    for k in 1..=4 {
                        let c = cell_at(&fluid, w.x, w.z + 0.35 + k as f32 * COARSE_CELL_M);
                        if c.water_m > 0.02 {
                            ahead = ahead.max(eta(&c));
                        }
                    }
                    if ahead > f32::MIN {
                        bow = bow.max(ahead - amb);
                        bow_sum += ahead - amb;
                        bow_n += 1.0;
                    }
                }
                for i in 2..4 {
                    let w = vehicle.pos + ori * vehicle.wheel_center_local(i);
                    let mut behind = f32::MAX;
                    for k in 2..=6 {
                        let c = cell_at(&fluid, w.x, w.z - 0.35 - k as f32 * COARSE_CELL_M);
                        if c.water_m > 0.02 {
                            behind = behind.min(eta(&c));
                        }
                    }
                    if behind < f32::MAX {
                        wake = wake.max(amb - behind);
                    }
                }
            }
            if s % 20 == 0 && (CHANNEL_Z0..CHANNEL_Z1).contains(&vehicle.pos.z) {
                publish(
                    &mut frame,
                    &soil,
                    particles,
                    [0.0, 0.0],
                    s as f32 * DT,
                    0,
                    Some(&fluid),
                    None,
                );
                if let Ok(dir) = std::env::var("SILT_FORD_DUMP") {
                    dump(&frame, &format!("{dir}/drive-z{:+05.1}.ppm", vehicle.pos.z));
                }
                let (ox, oz) = (frame.header.origin_x, frame.header.origin_z);
                for j in 0..WINDOW_RES {
                    let z = oz + (j as f32 + 0.5) * COARSE_CELL_M;
                    if (z - vehicle.pos.z).abs() > 2.5 {
                        continue;
                    }
                    for i in 0..WINDOW_RES {
                        let x = ox + (i as f32 + 0.5) * COARSE_CELL_M;
                        if (x - vehicle.pos.x).abs() > 2.0 {
                            continue;
                        }
                        tire_foam = tire_foam.max(frame.foam_crest[(j * WINDOW_RES + i) * 2]);
                    }
                }
            }
        }
        beach_stir(&fluid, &mut lo, &mut hi);
    }
    let mut stir = 0.0f32;
    for k in 0..5 {
        if hi[k] > lo[k] {
            stir = stir.max(hi[k] - lo[k]);
        }
    }
    Run {
        end_z: ford.world.vehicle.pos.z,
        max_speed,
        bow,
        bow_mean: if bow_n > 0.0 { bow_sum / bow_n } else { 0.0 },
        wake,
        stir,
        gap: held(&ford.fluid) - water0,
        nans: ford.fluid.non_finite_count(),
        tire_foam,
    }
}

#[test]
fn niva_fords_the_river_and_its_waves_reach_the_beach() {
    let seconds: f32 = std::env::var("SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(24.0);
    let quiet = run(false, seconds);
    let drive = run(true, seconds);
    println!(
        "FORD_DRIVE end_z {:.2} max_speed {:.2} bow peak {:.4} mean {:.4} wake {:.4} beach_stir {:.4} (no car {:.4}) tire_foam {:.2} gap {:.5} nan {}",
        drive.end_z, drive.max_speed, drive.bow, drive.bow_mean, drive.wake, drive.stir, quiet.stir, drive.tire_foam, drive.gap, drive.nans
    );
    assert_eq!(drive.nans, 0);
    assert_eq!(quiet.nans, 0);
    assert!(
        drive.gap.abs() < 0.02,
        "ford drive water gap {} m3",
        drive.gap
    );
    assert!(
        drive.end_z > CHANNEL_Z1,
        "Niva stuck in the ford at z {}",
        drive.end_z
    );
    assert!(drive.bow > 0.02, "no bow wave: {}", drive.bow);
    assert!(
        drive.bow_mean > 0.005,
        "front wheels do not pile water: {}",
        drive.bow_mean
    );
    assert!(
        drive.wake > 0.01,
        "no trough behind the rear wheels: {}",
        drive.wake
    );
    assert!(
        drive.stir > 2.0 * quiet.stir + 0.01,
        "crossing waves did not reach the beach: {} vs {}",
        drive.stir,
        quiet.stir
    );
    assert!(
        drive.tire_foam > 0.5,
        "no foam at the tires: {}",
        drive.tire_foam
    );
}

/// Top-down picture of the published maps (same colours as `ford_dump_maps`).
fn dump(frame: &ContractFrame, path: &str) {
    let n = WINDOW_RES;
    let scale = 4usize;
    let w = n * scale;
    let mut img = vec![0u8; w * w * 3];
    for j in 0..n {
        for i in 0..n {
            let idx = j * n + i;
            let ground = frame.ground_rut[idx * 4];
            let east = frame.ground_rut[(j * n + (i + 1).min(n - 1)) * 4];
            let north = frame.ground_rut[((j + 1).min(n - 1) * n + i) * 4];
            let shade = (0.8 + (east - ground) * 6.0 + (north - ground) * 6.0).clamp(0.3, 1.3);
            let m = frame.moisture[idx];
            let sand = [0.82 - 0.40 * m, 0.72 - 0.38 * m, 0.55 - 0.32 * m];
            let h = frame.film[idx * 2];
            let u = frame.flow[idx * 2];
            let v = frame.flow[idx * 2 + 1];
            let sp = (u * u + v * v).sqrt().min(1.5);
            let foam = frame.foam_crest[idx * 2];
            let a = if h > 0.003 {
                (h / 0.12).clamp(0.15, 0.9)
            } else {
                0.0
            };
            let deep = (h / 0.6).clamp(0.0, 1.0);
            let water = [
                0.08 + 0.25 * sp,
                0.30 + 0.25 * sp - 0.12 * deep,
                0.50 - 0.15 * deep + 0.1 * sp,
            ];
            for dy in 0..scale {
                for dx in 0..scale {
                    let o = (((n - 1 - j) * scale + dy) * w + i * scale + dx) * 3;
                    for k in 0..3 {
                        let mut c = sand[k] * shade * (1.0 - a) + water[k] * a;
                        c = c * (1.0 - foam * 0.8) + 0.95 * foam * 0.8;
                        img[o + k] = (c.clamp(0.0, 1.0) * 255.0) as u8;
                    }
                }
            }
        }
    }
    let mut out = format!("P6 {w} {w} 255\n").into_bytes();
    out.extend_from_slice(&img);
    std::fs::write(path, out).expect("write");
}
