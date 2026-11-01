//! Niva fog on the shared air grid. Dust and air velocity stay the channels
//! they already were. This file checks the fog scalar with numbers.

use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use silt::dust::{
    inject_fog, tires_at, Dust, FogInject, AIR_DX_M, AIR_DY_M, AIR_DZ_M, AIR_LAYERS, AIR_RES,
    AIR_X0_M, AIR_Y0_M, AIR_Z0_M, HUBS,
};
use silt::fluid_core::FluidCore;
use silt::render_contract::{
    publish, ContractFrame, DUST_FLUID_AIR, DUST_LEDGER, RENDER_CONTRACT_VERSION, VALID_AIR,
    VALID_DUST_FOG, VALID_FOG,
};
use silt::soil::SoilField;

const DT: f32 = 1.0 / 60.0;

/// Release fog step. Do not raise this. Every build prints it.
const FOG_STEP_BUDGET_MS: f64 = 7.0;
const SPEED: f32 = 8.0;
const OFFER_M3_S: f32 = 0.0012;
const DUMP_DIR: &str = "/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media/fog-air";

fn at(ix: u32, iy: u32, iz: u32) -> usize {
    (ix + iy * AIR_RES + iz * AIR_RES * AIR_LAYERS) as usize
}

fn dump_dir() -> Option<PathBuf> {
    if let Ok(path) = std::env::var("FOG_AIR_DUMP") {
        return Some(PathBuf::from(path));
    }
    let dir = PathBuf::from(DUMP_DIR);
    if dir.parent().is_some_and(|parent| parent.is_dir()) {
        Some(dir)
    } else {
        None
    }
}

fn write_dump(name: &str, body: &str) {
    let Some(dir) = dump_dir() else {
        return;
    };
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(name);
    if let Ok(mut file) = fs::File::create(&path) {
        let _ = file.write_all(body.as_bytes());
    }
}

struct FogRead {
    air_m3: f32,
    peak: f32,
    z_cm: f32,
    behind_rear_m3: f32,
    speed_at_peak: f32,
}

fn read_fog(dust: &Dust) -> FogRead {
    let chassis = dust.chassis_z();
    let rear = chassis + (HUBS[2][2] + HUBS[3][2]) * 0.5;
    let vol = dust.fog_volume();
    let vel = dust.air_velocity_xyz();
    let cell = AIR_DX_M * AIR_DY_M * AIR_DZ_M;
    let mut air = 0.0f64;
    let mut z_sum = 0.0f64;
    let mut behind = 0.0f64;
    let mut peak = 0.0f32;
    let mut peak_at = 0usize;
    for iz in 0..AIR_RES {
        for iy in 0..AIR_LAYERS {
            for ix in 0..AIR_RES {
                let i = at(ix, iy, iz);
                let m3 = vol[i];
                if m3 <= 0.0 {
                    continue;
                }
                let z = chassis + AIR_Z0_M + (iz as f32 + 0.5) * AIR_DZ_M;
                let md = m3 as f64;
                air += md;
                z_sum += z as f64 * md;
                if z < rear {
                    behind += md;
                }
                let fraction = m3 / cell;
                if fraction > peak {
                    peak = fraction;
                    peak_at = i;
                }
            }
        }
    }
    let z_cm = if air > 0.0 {
        (z_sum / air) as f32
    } else {
        chassis
    };
    let speed = if vel.len() >= (peak_at + 1) * 3 {
        let vx = vel[peak_at * 3];
        let vy = vel[peak_at * 3 + 1];
        let vz = vel[peak_at * 3 + 2];
        (vx * vx + vy * vy + vz * vz).sqrt()
    } else {
        0.0
    };
    FogRead {
        air_m3: air as f32,
        peak,
        z_cm,
        behind_rear_m3: behind as f32,
        speed_at_peak: speed,
    }
}

#[test]
fn niva_fog_rides_the_dust_air_and_keeps_its_water() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.set_wind(1.5, -0.9);
    let _poured = fluid.pour_water(-2.2, 2.2, -3.0, 16.0, 0.08);
    let water0 = fluid.ledger().water_m3 as f64;
    let mut dust = Dust::new();
    dust.set_wind(1.5, -0.9);

    let dry = inject_fog(
        &mut dust,
        FogInject {
            x: 0.0,
            z: 0.0,
            water_m: 0.001,
            offer_m3_s: 1.0,
        },
        DT,
    );
    assert_eq!(dry, 0.0, "thin film lofted {dry}");

    let mut z = 0.0f32;
    let mut taken = 0.0f64;
    let mut step_ms = 0.0f64;
    let mut step_max = 0.0f64;
    let mut timed = 0.0f64;
    for step in 0..90 {
        let tires = tires_at(z, SPEED, 1.6);
        let t0 = Instant::now();
        dust.step_pads(&tires, &[0.92; 4], &[0.08; 4], DT);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        if step >= 12 {
            step_ms += ms;
            step_max = step_max.max(ms);
            timed += 1.0;
        }
        for tire in &tires {
            let water_m = fluid.sample(tire.contact.x, tire.contact.z).water_m;
            let got = inject_fog(
                &mut dust,
                FogInject {
                    x: tire.contact.x,
                    z: tire.contact.z,
                    water_m,
                    offer_m3_s: OFFER_M3_S,
                },
                DT,
            );
            let removed = fluid.take_water_at(tire.contact.x, tire.contact.z, got);
            assert!(
                (removed - got).abs() < 1.0e-5,
                "water removed {removed:.6} fog {got:.6}"
            );
            taken += got as f64;
        }
        z += SPEED * DT;
    }
    dust.step_pads(&tires_at(z, SPEED, 1.6), &[0.92; 4], &[0.08; 4], DT);

    let water1 = fluid.ledger().water_m3 as f64;
    let emitted = dust.fog_emitted_m3() as f64;
    let air = dust.fog_air_m3() as f64;
    let settled = dust.fog_settled_m3() as f64;
    let read = read_fog(&dust);
    let step_mean = if timed > 0.0 { step_ms / timed } else { 0.0 };
    let chassis = dust.chassis_z();
    let rear = chassis + (HUBS[2][2] + HUBS[3][2]) * 0.5;
    let books_gap = (emitted - air - settled).abs();
    let water_gap = (water0 - water1 - taken).abs();

    let soil = SoilField::flat(0.08);
    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &soil,
        &[],
        [0.0, z],
        z / SPEED,
        15,
        Some(&fluid),
        Some(&dust),
    );
    let dust_sum: f32 = frame.dust_fog.iter().sum();
    let fog_sum: f32 = frame.fog.iter().sum();

    let mut columns = String::from("ix,iy,iz,x,y,z,fraction,vx,vy,vz\n");
    let vol = dust.fog_volume();
    let vel = dust.air_velocity_xyz();
    let cell = AIR_DX_M * AIR_DY_M * AIR_DZ_M;
    for iz in 0..AIR_RES {
        for iy in 0..AIR_LAYERS {
            for ix in 0..AIR_RES {
                let i = at(ix, iy, iz);
                let fraction = vol[i] / cell;
                if fraction < 1.0e-5 {
                    continue;
                }
                let x = AIR_X0_M + (ix as f32 + 0.5) * AIR_DX_M;
                let y = AIR_Y0_M + (iy as f32 + 0.5) * AIR_DY_M;
                let wz = dust.chassis_z() + AIR_Z0_M + (iz as f32 + 0.5) * AIR_DZ_M;
                let (vx, vy, vz) = if vel.len() >= (i + 1) * 3 {
                    (vel[i * 3], vel[i * 3 + 1], vel[i * 3 + 2])
                } else {
                    (0.0, 0.0, 0.0)
                };
                columns.push_str(&format!(
                    "{ix},{iy},{iz},{x:.4},{y:.4},{wz:.4},{fraction:.8},{vx:.4},{vy:.4},{vz:.4}\n"
                ));
            }
        }
    }
    let mut settled_txt = String::from("ix,iz,m3\n");
    for ((ix, iz), m3) in dust.fog_settled_cells() {
        settled_txt.push_str(&format!("{ix},{iz},{m3:.8}\n"));
    }
    let numbers = format!(
        "water0_m3 {water0:.6}\n\
         water1_m3 {water1:.6}\n\
         taken_m3 {taken:.6}\n\
         water_gap_m3 {water_gap:.8}\n\
         fog_emitted_m3 {emitted:.6}\n\
         fog_air_m3 {air:.6}\n\
         fog_settled_m3 {settled:.6}\n\
         fog_books_gap_m3 {books_gap:.8}\n\
         fog_peak_fraction {peak:.6}\n\
         fog_z_cm {z_cm:.4}\n\
         chassis_z {chassis:.4}\n\
         rear_axle_z {rear:.4}\n\
         fog_behind_rear_m3 {behind:.6}\n\
         air_speed_at_peak_m_s {speed:.4}\n\
         dust_fog_sum {dust_sum:.8}\n\
         fog_sum {fog_sum:.6}\n\
         dust_air_m3 {dust_air:.8}\n\
         step_mean_ms {step_mean:.3}\n\
         step_max_ms {step_max:.3}\n\
         step_n {timed:.0}\n\
         FOG_STEP_BUDGET ms={FOG_STEP_BUDGET_MS:.3}\n\
         contract_version {version}\n",
        peak = read.peak,
        z_cm = read.z_cm,
        behind = read.behind_rear_m3,
        speed = read.speed_at_peak,
        dust_air = dust.dust_air_m3(),
        version = frame.header.version,
    );
    write_dump("niva-numbers.txt", &numbers);
    write_dump("niva-columns.csv", &columns);
    write_dump("niva-settled.csv", &settled_txt);
    println!("{numbers}");
    let _ = std::io::stdout().flush();

    assert!(taken > 0.004, "taken {taken:.6} m3");
    assert!(water_gap < 1.0e-3, "water gap {water_gap:.6}");
    assert!(books_gap < 1.0e-4, "fog books {books_gap:.8}");
    assert!((emitted - taken).abs() < 1.0e-4, "emitted {emitted:.6} taken {taken:.6}");
    assert!(air + settled > 0.004, "air {air:.6} settled {settled:.6}");
    assert!(
        (read.air_m3 as f64 - air).abs() < 1.0e-4,
        "grid {} counter {air:.6}",
        read.air_m3
    );
    assert!(read.peak > 1.0e-4, "peak fraction {}", read.peak);
    assert!(
        read.z_cm < chassis,
        "centroid {z:.3} is not behind the chassis {chassis:.3}",
        z = read.z_cm
    );
    assert!(
        read.behind_rear_m3 > 1.0e-4,
        "nothing behind the rear axle {}",
        read.behind_rear_m3
    );
    assert!(
        read.speed_at_peak > 0.5,
        "air at the loudest fog cell {:.3} m/s",
        read.speed_at_peak
    );
    assert!(
        dust.dust_air_m3() == 0.0 && dust.emitted_m3() == 0.0,
        "wet ground lofted dust air {} emitted {}",
        dust.dust_air_m3(),
        dust.emitted_m3()
    );
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(frame.header.dust_source, DUST_LEDGER);
    assert_eq!(frame.header.valid & VALID_FOG, VALID_FOG);
    assert_eq!(frame.header.valid & VALID_DUST_FOG, VALID_DUST_FOG);
    assert_eq!(frame.header.valid & VALID_AIR, VALID_AIR);
    assert_eq!(frame.fog.len(), frame.dust_fog.len());
    assert_eq!(frame.air_velocity.len(), frame.dust_fog.len() * 4);
    assert!(dust_sum.abs() < 1.0e-5, "dust_fog sum {dust_sum}");
    assert!(fog_sum > 0.0, "published fog is empty");
    assert!(frame.air_velocity.chunks(4).all(|texel| texel[3] == 0.0));
    assert!(frame.fog.iter().all(|v| v.is_finite()));

    #[cfg(not(debug_assertions))]
    assert!(
        step_mean <= FOG_STEP_BUDGET_MS,
        "FOG_STEP {step_mean:.3} ms over {FOG_STEP_BUDGET_MS}, max {step_max:.3}"
    );
}

#[test]
fn fluid_sheet_fog_is_not_dust_and_closes_the_ledger() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.set_wind(1.5, -0.9);
    let poured = fluid.pour_water(-1.2, 1.2, -1.2, 1.2, 0.05);
    let before = fluid.ledger();
    assert!((before.water_m3 - poured).abs() < 1.0e-3, "pour {}", before.water_m3);
    assert_eq!(before.fog_air_m3, 0.0);

    let got = fluid.inject_fog(
        FogInject {
            x: 0.15,
            z: 0.25,
            water_m: 0.05,
            offer_m3_s: 0.04,
        },
        0.05,
    );
    assert!(got > 1.0e-4, "sheet loft {got:.6}");
    let mid = fluid.ledger();
    assert!(
        (mid.total_m3 - before.total_m3).abs() < 1.0e-3,
        "total {} -> {}",
        before.total_m3,
        mid.total_m3
    );
    assert!(mid.fog_air_m3 > 1.0e-4, "fog {}", mid.fog_air_m3);
    assert!(mid.water_m3 < before.water_m3 - 1.0e-4);
    assert_eq!(
        fluid.inject_fog(
            FogInject {
                x: 0.15,
                z: 0.25,
                water_m: 0.001,
                offer_m3_s: 1.0,
            },
            0.05,
        ),
        0.0
    );

    for _ in 0..30 {
        let _ = fluid.step(DT);
    }
    let end = fluid.ledger();
    let soil = SoilField::flat(0.2);
    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &soil,
        &[],
        [0.0, 0.0],
        0.5,
        15,
        Some(&fluid),
        None,
    );
    let fog_sum: f32 = frame.fog.iter().copied().filter(|v| *v > 0.0).sum();
    let text = format!(
        "poured_m3 {poured:.6}\n\
         got_m3 {got:.6}\n\
         mid_fog_m3 {:.6}\n\
         mid_water_m3 {:.6}\n\
         end_fog_m3 {:.6}\n\
         end_water_m3 {:.6}\n\
         end_total_m3 {:.6}\n\
         before_total_m3 {:.6}\n\
         total_gap_m3 {:.8}\n\
         published_fog_sum {fog_sum:.6}\n\
         version {}\n",
        mid.fog_air_m3,
        mid.water_m3,
        end.fog_air_m3,
        end.water_m3,
        end.total_m3,
        before.total_m3,
        (end.total_m3 - before.total_m3).abs(),
        frame.header.version,
    );
    write_dump("fluid-numbers.txt", &text);
    println!("{text}");

    assert!(
        (end.total_m3 - before.total_m3).abs() < 2.0e-3,
        "ledger {} -> {}",
        before.total_m3,
        end.total_m3
    );
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(frame.header.dust_source, DUST_FLUID_AIR);
    assert_eq!(frame.header.valid & VALID_FOG, VALID_FOG);
    assert_eq!(frame.header.valid & VALID_AIR, VALID_AIR);
    assert_eq!(frame.fog.len(), frame.dust_fog.len());
    assert!(frame.dust_fog.iter().all(|v| v.abs() < 1.0e-6));
    assert!(
        fog_sum > 0.0 || end.water_m3 > mid.water_m3,
        "fog left the sheet without returning, sum {fog_sum} water {} -> {}",
        mid.water_m3,
        end.water_m3
    );
    assert!(frame
        .air_velocity
        .chunks(4)
        .all(|texel| texel[1] == 0.0 && texel[3] == 0.0));
    assert!(frame.fog.iter().all(|v| v.is_finite()));
}
