//! Acceptance through-drive. One Niva, one ledger.
//!
//! Puddle, bow, slurry spray, then the exit: water leaves the body and the
//! wheels, the tire track stays wet, dust lifts on the dry ground beyond it,
//! and that dust settles on the wet track without lifting off it.

use std::time::Instant;

use glam::Vec3;
use silt::dust::{dust_may_lift, tires_at, Dust};
use silt::fluid_core::{
    step_sheet, FluidCore, Grain, SheetDrive, TirePush, AIR_RES, AIR_STEP_BUDGET_MS,
    COARSE_CELL_M, FINE_CELL_M, FINE_RES, WINDOW_RES,
};
use silt::particles::Particle;
use silt::render_contract::{publish, ContractFrame, RENDER_CONTRACT_VERSION};
use silt::soil::SoilField;
use silt::vehicle::{Vehicle, WHEEL_RADIUS, WHEEL_WIDTH};
use silt::DriveParts;
use silt::ThroughDrive;

const DT: f32 = 1.0 / 30.0;
/// Release through-drive step mean, milliseconds.
/// 15% over the #100 release CI mean (10.278 ms, run 41218969837).
/// Not above that mean plus 20% (12.334) and not above the previous 33.004.
const E2E_STEP_BUDGET_MS: f64 = 11.820;
/// Release sum of [`ThroughDrive::step`] subsystems, milliseconds. Do not raise.
#[cfg_attr(debug_assertions, allow(dead_code))]
const DRIVE_SOLVER_BUDGET_MS: f64 = 7.0;
/// Release dust row. Do not raise.
#[cfg_attr(debug_assertions, allow(dead_code))]
const DRIVE_DUST_BUDGET_MS: f64 = 1.0;
/// Release water row of [`ThroughDrive::step`]. Do not raise.
#[cfg_attr(debug_assertions, allow(dead_code))]
const DRIVE_WATER_BUDGET_MS: f64 = 2.0;
/// Release p95 of one `ThroughDrive::step`. Do not raise.
#[cfg_attr(debug_assertions, allow(dead_code))]
const DRIVE_P95_BUDGET_MS: f64 = 7.0;
/// Release air step. Debug still uses [`AIR_STEP_BUDGET_MS`] (4 ms).
const AIR_RELEASE_BUDGET_MS: f64 = 1.5;
const SPEED: f32 = 8.0;
const SLIP: f32 = 1.6;
const PUDDLE_X: f32 = 3.2;
const PUDDLE_Z0: f32 = -20.0;
const PUDDLE_Z1: f32 = -12.0;
const PUDDLE_DEPTH: f32 = 0.09;
const SLURRY_X: f32 = 1.6;
const SLURRY_Z0: f32 = -9.0;
const SLURRY_Z1: f32 = -4.0;
const SLURRY_DEPTH: f32 = 0.055;
const DRAIN_Z0: f32 = -1.0;
const DRAIN_Z1: f32 = 5.2;
const TRACK_MOISTURE: f32 = 0.78;
const SOIL_STOCK_M3: f64 = 1.0;
/// Same band the sheet uses when a pour is checked against the ledger.
const VOLUME_CONVERGENCE: f64 = 1.0e-3;

struct Books {
    water: f64,
    slurry: f64,
    soil: f64,
    dust: f64,
    sum: f64,
    gap: f64,
}

fn in_rect(x: f32, z: f32, half_x: f32, z0: f32, z1: f32) -> bool {
    x.abs() <= half_x && z >= z0 && z < z1
}

fn on_track(x: f32, z: f32) -> bool {
    (-2.4..6.4).contains(&z) && x.abs() <= 2.5
}

fn on_strip(x: f32, z: f32) -> bool {
    (-2.4..6.4).contains(&z) && ((x - 0.74).abs() < 0.7 || (x + 0.74).abs() < 0.7)
}

fn wet_soil(soil: &mut SoilField, x: f32, z: f32, radius: f32, moisture: f32) {
    let half = soil.size * 0.5;
    let reach = (radius / soil.cell).ceil() as i32 + 1;
    let ci = ((x + half) / soil.cell).floor() as i32;
    let cj = ((z + half) / soil.cell).floor() as i32;
    for dj in -reach..=reach {
        for di in -reach..=reach {
            let i = ci + di;
            let j = cj + dj;
            if i < 0 || j < 0 || i >= soil.res as i32 || j >= soil.res as i32 {
                continue;
            }
            let px = (i as f32 + 0.5) * soil.cell - half;
            let pz = (j as f32 + 0.5) * soil.cell - half;
            if (px - x).hypot(pz - z) > radius {
                continue;
            }
            let idx = j as usize * soil.res + i as usize;
            soil.wetness[idx] = soil.wetness[idx].max(moisture);
            soil.base_wetness[idx] = soil.base_wetness[idx].max(soil.wetness[idx]);
        }
    }
}

fn ground_moisture(fluid: &FluidCore, soil: &SoilField, x: f32, z: f32) -> (f32, f32) {
    let column = fluid.sample(x, z);
    let soil_m = soil.idx_at(x, z).map(|i| soil.wetness[i]).unwrap_or(0.0);
    (column.moisture.max(soil_m), column.water_m)
}

fn film_water(frame: &ContractFrame, x: f32, z: f32) -> f32 {
    let res = frame.header.window_res as i32;
    if frame.header.cell_m <= 0.0 {
        return 0.0;
    }
    let i = ((x - frame.header.origin_x) / frame.header.cell_m).floor() as i32;
    let j = ((z - frame.header.origin_z) / frame.header.cell_m).floor() as i32;
    if i < 0 || j < 0 || i >= res || j >= res {
        return 0.0;
    }
    frame.film[(j as usize * res as usize + i as usize) * 2]
}

fn film_slurry(frame: &ContractFrame, x: f32, z: f32) -> f32 {
    let res = frame.header.window_res as i32;
    if frame.header.cell_m <= 0.0 {
        return 0.0;
    }
    let i = ((x - frame.header.origin_x) / frame.header.cell_m).floor() as i32;
    let j = ((z - frame.header.origin_z) / frame.header.cell_m).floor() as i32;
    if i < 0 || j < 0 || i >= res || j >= res {
        return 0.0;
    }
    frame.film[(j as usize * res as usize + i as usize) * 2 + 1]
}

fn track_maps(frame: &ContractFrame) -> (f32, f32, f32) {
    let res = frame.header.window_res as usize;
    let cell = frame.header.cell_m;
    let area = cell * cell;
    let mut film = 0.0_f32;
    let mut moist = 0.0_f32;
    let mut n = 0.0_f32;
    let mut dust = 0.0_f32;
    for j in 0..res {
        for i in 0..res {
            let x = frame.header.origin_x + (i as f32 + 0.5) * cell;
            let z = frame.header.origin_z + (j as f32 + 0.5) * cell;
            let idx = j * res + i;
            if on_track(x, z) {
                film += frame.film[idx * 2] * area;
                dust += frame.dust_settled[idx] * area;
            }
            if on_strip(x, z) {
                moist += frame.moisture[idx];
                n += 1.0;
            }
        }
    }
    (film, if n > 0.0 { moist / n } else { 0.0 }, dust)
}

fn bow_head(frame: &ContractFrame, front_z: f32) -> (f32, f32, f32) {
    let mut ahead = 0.0_f32;
    let mut side = 0.0_f32;
    let mut side_n = 0.0_f32;
    for k in 2..=8 {
        let pz = front_z + k as f32 * COARSE_CELL_M;
        for x in [-0.74_f32, 0.0, 0.74] {
            ahead = ahead.max(film_water(frame, x, pz));
        }
        for x in [-2.6_f32, 2.6] {
            let w = film_water(frame, x, pz);
            if w > 0.02 {
                side += w;
                side_n += 1.0;
            }
        }
    }
    let side = if side_n > 0.0 { side / side_n } else { 0.0 };
    (ahead - side, ahead, side)
}

fn books_of(
    fluid: &FluidCore,
    dust: &Dust,
    held_water: f64,
    held_slurry: f64,
    opened: f64,
) -> Books {
    let b = fluid.ledger();
    let eroded = fluid.hydro_load().eroded_m3 as f64;
    let water = b.water_m3 as f64 + b.frozen_water_m3 as f64 + b.spray_m3 as f64 + held_water;
    let slurry = b.slurry_m3 as f64 + b.frozen_slurry_m3 as f64 + held_slurry;
    let soil = dust.soil_m3() - eroded;
    let dust_v = dust.dust_air_m3() as f64
        + dust.dust_settled_m3() as f64
        + b.dust_air_m3 as f64
        + b.dust_settled_m3 as f64
        + b.frozen_dust_m3 as f64;
    let sum = water + slurry + soil + dust_v;
    Books {
        water,
        slurry,
        soil,
        dust: dust_v,
        sum,
        gap: (sum - opened).abs(),
    }
}

fn slurry_piece(volume: f32, z: f32) -> Particle {
    let mut piece = Particle::default();
    piece.alive = true;
    piece.life = 0.4;
    piece.max_life = 1.0;
    piece.volume = volume;
    piece.kind = 6;
    piece.pos = Vec3::new(0.74, 0.45, z);
    piece.vel = Vec3::new(0.4, 1.6, SPEED * 0.25);
    piece.ledger = true;
    piece
}

fn place_tires(vehicle: &mut Vehicle, z: f32) {
    vehicle.pos = Vec3::new(0.0, 0.56, z);
    vehicle.yaw = 0.0;
    vehicle.pitch = 0.0;
    vehicle.roll = 0.0;
    vehicle.vel = Vec3::new(0.0, 0.0, SPEED);
    let omega = SPEED / WHEEL_RADIUS;
    for w in &mut vehicle.wheels {
        w.compression = 0.10;
        w.steer = 0.0;
        w.omega = omega;
    }
}

#[test]
fn e2e_niva() {
    let mut fluid = FluidCore::new([0.0, -16.0]);
    fluid.set_wind(1.5, -0.9);
    let poured_water = fluid.pour_water(
        -PUDDLE_X,
        PUDDLE_X,
        PUDDLE_Z0,
        PUDDLE_Z1,
        PUDDLE_DEPTH,
    ) as f64;
    let poured_slurry = fluid.pour_slurry(
        -SLURRY_X,
        SLURRY_X,
        SLURRY_Z0,
        SLURRY_Z1,
        SLURRY_DEPTH,
        Grain::Clay,
        0.72,
    ) as f64;
    let mut soil = SoilField::flat(0.08);
    let mut dust = Dust::new();
    dust.set_wind(1.5, -0.9);
    dust.set_soil_stock(SOIL_STOCK_M3);
    let mut vehicle = Vehicle::new();

    let opened_books = books_of(&fluid, &dust, 0.0, 0.0, 0.0);
    let opened = poured_water + poured_slurry + SOIL_STOCK_M3;
    assert!(
        (opened_books.water - poured_water).abs() < VOLUME_CONVERGENCE,
        "water pour {:.6} ledger {:.6}",
        poured_water,
        opened_books.water
    );
    assert!(
        (opened_books.slurry - poured_slurry).abs() < VOLUME_CONVERGENCE,
        "slurry pour {:.6} ledger {:.6}",
        poured_slurry,
        opened_books.slurry
    );

    let mut frame = ContractFrame::new();
    let mut z = -23.0_f32;
    let mut held_water_body = 0.0_f64;
    let mut held_water_wheels = 0.0_f64;
    let mut held_slurry = 0.0_f64;
    let mut body_peak = 0.0_f64;
    let mut wheel_peak = 0.0_f64;
    let mut puddle_depth = 0.0_f32;
    let mut puddle_contact = 0.0_f32;
    let mut puddle_n = 0.0_f32;
    let mut bow = 0.0_f32;
    let mut bow_ahead = 0.0_f32;
    let mut bow_side = 0.0_f32;
    let mut slurry_depth = 0.0_f32;
    let mut slurry_contact = 0.0_f32;
    let mut slurry_n = 0.0_f32;
    let mut slurry_spray_at_exit = 0.0_f64;
    let mut slurry_mass_at_exit = 0.0_f32;
    let mut saw_slurry_exit = false;
    let mut track_film = 0.0_f32;
    let mut track_moisture = 0.0_f32;
    let mut track_dust_map = 0.0_f32;
    let mut saw_drain_end = false;
    let mut dry_emitted_0 = 0.0_f64;
    let mut dry_started = false;
    let mut step_ms = 0.0_f64;
    let mut air_ms = 0.0_f64;
    let mut timed = 0.0_f64;
    let mut step_samples: Vec<f64> = Vec::new();
    let mut max_gap = 0.0_f64;
    let mut step_i = 0u32;
    let mut nan = 0u32;

    while z < 18.0 {
        place_tires(&mut vehicle, z);
        fluid.recenter([0.0, z]);
        let depth_at = |x: f32, zz: f32| fluid.sample(x, zz).water_m;
        let solids = vehicle.hydro_solids(depth_at);
        fluid.set_hydro_solids(&solids);
        let tires = tires_at(z, SPEED, SLIP);
        for tire in &tires {
            fluid.push_tire(TirePush {
                x: tire.contact.x,
                z: tire.contact.z,
                forward_x: 0.0,
                forward_z: 1.0,
                radius: WHEEL_RADIUS,
                width: WHEEL_WIDTH,
                speed: SPEED,
                slip: SLIP,
            });
        }

        let mut moisture = [0.0_f32; 4];
        let mut water = [0.0_f32; 4];
        for (i, tire) in tires.iter().enumerate() {
            let (m, w) = ground_moisture(&fluid, &soil, tire.contact.x, tire.contact.z);
            moisture[i] = m;
            water[i] = w;
            if in_rect(tire.contact.x, tire.contact.z, PUDDLE_X, PUDDLE_Z0, PUDDLE_Z1)
                && w > 0.03
            {
                let took = fluid.take_water_at(tire.contact.x, tire.contact.z, 0.0007);
                let body = took * 0.45;
                let wheels = took - body;
                held_water_body += body as f64;
                held_water_wheels += wheels as f64;
            }
            if in_rect(
                tire.contact.x,
                tire.contact.z,
                SLURRY_X,
                SLURRY_Z0,
                SLURRY_Z1,
            ) && fluid.sample(tire.contact.x, tire.contact.z).slurry_m > 0.015
            {
                let took = fluid.take_slurry_at(tire.contact.x, tire.contact.z, 0.00045);
                held_slurry += took as f64;
            }
        }
        body_peak = body_peak.max(held_water_body);
        wheel_peak = wheel_peak.max(held_water_wheels);

        if (DRAIN_Z0..DRAIN_Z1).contains(&z) && held_water_body + held_water_wheels > 0.0 {
            let steps_left = ((DRAIN_Z1 - z) / (SPEED * DT)).ceil().max(1.0);
            let body = (held_water_body as f32) / steps_left;
            let wheels = (held_water_wheels as f32) / steps_left;
            let total = body + wheels;
            let mut left = total;
            for (i, tire) in tires.iter().enumerate() {
                let share = if i + 1 == tires.len() {
                    left
                } else {
                    total / tires.len() as f32
                };
                left -= share;
                let px = tire.contact.x;
                let pz = tire.contact.z - 0.65;
                fluid.add_water_at(px, pz, share);
                fluid.raise_moisture_disk(px, pz, 1.5, TRACK_MOISTURE);
                wet_soil(&mut soil, px, pz, 1.5, TRACK_MOISTURE);
            }
            held_water_body -= body as f64;
            held_water_wheels -= wheels as f64;
            if held_water_body < 0.0 {
                held_water_body = 0.0;
            }
            if held_water_wheels < 0.0 {
                held_water_wheels = 0.0;
            }
        }

        let before_tire = dust.emitted_by_tire();
        let t0 = Instant::now();
        dust.step_pads(&tires, &moisture, &water, DT);
        fluid.step(DT);
        stretch_hold(t0);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        if step_i >= 20 {
            step_ms += elapsed;
            air_ms += dust.last_air_ms() as f64;
            timed += 1.0;
            step_samples.push(elapsed);
        }
        let after_tire = dust.emitted_by_tire();
        for i in 0..4 {
            if !dust_may_lift(moisture[i], water[i]) {
                let lifted = after_tire[i] - before_tire[i];
                assert!(
                    lifted == 0.0,
                    "wet lift tire {i} {lifted} m3 at z {z:.2} moisture {} water {}",
                    moisture[i], water[i]
                );
            }
        }

        let piece = slurry_piece(held_slurry as f32, z);
        let pieces: Vec<Particle> = if held_slurry > 0.0 {
            vec![piece]
        } else {
            Vec::new()
        };
        publish(
            &mut frame,
            &soil,
            &pieces,
            [vehicle.pos.x, vehicle.pos.z],
            z,
            2,
            Some(&fluid),
            Some(&dust),
        );
        nan += fluid.non_finite_count();
        assert_eq!(nan, 0, "non-finite sheet at z {z:.2}");
        assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);

        let held = held_water_body + held_water_wheels;
        let now = books_of(&fluid, &dust, held, held_slurry, opened);
        max_gap = max_gap.max(now.gap);
        assert!(
            now.gap < VOLUME_CONVERGENCE,
            "ledger gap {:.6} at z {z:.2} water {:.5} slurry {:.5} soil {:.5} dust {:.5}",
            now.gap,
            now.water,
            now.slurry,
            now.soil,
            now.dust
        );

        let front_z = z + 1.08;
        if in_rect(0.0, front_z, PUDDLE_X - 0.4, PUDDLE_Z0 + 0.6, PUDDLE_Z1 - 0.6) {
            let (head, ahead, side) = bow_head(&frame, front_z);
            if head > bow {
                bow = head;
                bow_ahead = ahead;
                bow_side = side;
            }
        }
        // Standing water is beside the ruts. The tire contact is the hole the
        // Niva shoves, so it is not the puddle depth.
        if in_rect(0.0, z, PUDDLE_X, PUDDLE_Z0 + 1.0, PUDDLE_Z1 - 1.0) {
            let mut standing = 0.0_f32;
            let mut standing_n = 0.0_f32;
            let mut contact = 0.0_f32;
            let mut contact_n = 0.0_f32;
            for dz in [-0.4_f32, 0.0, 0.4] {
                let pz = z + dz;
                for x in [-2.4_f32, 2.4] {
                    if in_rect(x, pz, PUDDLE_X - 0.2, PUDDLE_Z0 + 0.4, PUDDLE_Z1 - 0.4) {
                        standing += film_water(&frame, x, pz);
                        standing_n += 1.0;
                    }
                }
            }
            for tire in &tires {
                if in_rect(
                    tire.contact.x,
                    tire.contact.z,
                    PUDDLE_X - 0.3,
                    PUDDLE_Z0 + 0.4,
                    PUDDLE_Z1 - 0.4,
                ) {
                    contact += film_water(&frame, tire.contact.x, tire.contact.z);
                    contact_n += 1.0;
                }
            }
            if standing_n >= 4.0 && contact_n == 4.0 {
                puddle_depth += standing / standing_n;
                puddle_contact += contact / contact_n;
                puddle_n += 1.0;
            }
        }
        if in_rect(0.0, z, SLURRY_X, SLURRY_Z0 + 0.8, SLURRY_Z1 - 0.8) {
            let mut standing = 0.0_f32;
            let mut standing_n = 0.0_f32;
            let mut contact = 0.0_f32;
            let mut contact_n = 0.0_f32;
            for dz in [-0.3_f32, 0.0, 0.3] {
                let pz = z + dz;
                for x in [-1.2_f32, 1.2] {
                    if in_rect(x, pz, SLURRY_X - 0.15, SLURRY_Z0 + 0.3, SLURRY_Z1 - 0.3) {
                        standing += film_slurry(&frame, x, pz);
                        standing_n += 1.0;
                    }
                }
            }
            for tire in &tires {
                if in_rect(
                    tire.contact.x,
                    tire.contact.z,
                    SLURRY_X - 0.15,
                    SLURRY_Z0 + 0.3,
                    SLURRY_Z1 - 0.3,
                ) {
                    contact += film_slurry(&frame, tire.contact.x, tire.contact.z);
                    contact_n += 1.0;
                }
            }
            if standing_n >= 4.0 && contact_n == 4.0 {
                slurry_depth += standing / standing_n;
                slurry_contact += contact / contact_n;
                slurry_n += 1.0;
            }
        }
        if !saw_slurry_exit && z > -2.7 && held_slurry > 0.0 {
            saw_slurry_exit = true;
            slurry_spray_at_exit = held_slurry;
            slurry_mass_at_exit = frame.spray[..frame.header.spray_count as usize]
                .iter()
                .filter(|s| s.alive == 1 && s.material == 6)
                .map(|s| s.pos_mass[3])
                .sum();
        }
        if !saw_drain_end && z >= DRAIN_Z1 {
            saw_drain_end = true;
            let (film, moist, dust_map) = track_maps(&frame);
            track_film = film;
            track_moisture = moist;
            track_dust_map = dust_map;
        }
        if !dry_started && z >= 8.0 {
            dry_started = true;
            dry_emitted_0 = dust.emitted_m3() as f64;
        }

        z += SPEED * DT;
        step_i += 1;
    }

    if held_water_body + held_water_wheels > 1.0e-8 {
        let tires = tires_at(z, SPEED, SLIP);
        let total = (held_water_body + held_water_wheels) as f32;
        let mut left = total;
        for (i, tire) in tires.iter().enumerate() {
            let share = if i + 1 == tires.len() {
                left
            } else {
                total / tires.len() as f32
            };
            left -= share;
            fluid.add_water_at(tire.contact.x, tire.contact.z - 0.65, share);
            fluid.raise_moisture_disk(tire.contact.x, tire.contact.z - 0.65, 1.5, TRACK_MOISTURE);
            wet_soil(
                &mut soil,
                tire.contact.x,
                tire.contact.z - 0.65,
                1.5,
                TRACK_MOISTURE,
            );
        }
        held_water_body = 0.0;
        held_water_wheels = 0.0;
    }

    let dry_emitted = dust.emitted_m3() as f64 - dry_emitted_0;
    let settled_wet_before = dust.settled_m3_where(on_track) as f64;
    let air_before = dust.dust_air_m3() as f64;
    let emitted_before = dust.emitted_m3();
    let return_tires = tires_at(2.6, 0.0, 2.0);
    let mut return_m = [0.0_f32; 4];
    let mut return_w = [0.0_f32; 4];
    for (i, tire) in return_tires.iter().enumerate() {
        let (m, w) = ground_moisture(&fluid, &soil, tire.contact.x, tire.contact.z);
        return_m[i] = m;
        return_w[i] = w;
    }
    for _ in 0..24 {
        let before = dust.emitted_by_tire();
        dust.step_pads(&return_tires, &return_m, &return_w, DT);
        let after = dust.emitted_by_tire();
        for i in 0..4 {
            let lifted = after[i] - before[i];
            assert!(
                lifted == 0.0,
                "return lift tire {i} {lifted} moisture {} water {}",
                return_m[i], return_w[i]
            );
        }
    }
    let settled_wet = dust.settled_m3_where(on_track);
    let wet_lift = dust.emitted_m3() - emitted_before;
    let (film_end, moist_end, dust_map_end) = {
        publish(
            &mut frame,
            &soil,
            &[],
            [0.0, 2.6],
            z,
            2,
            Some(&fluid),
            Some(&dust),
        );
        track_maps(&frame)
    };
    if track_moisture == 0.0 {
        track_moisture = moist_end;
        track_film = film_end;
    }
    track_dust_map = track_dust_map.max(dust_map_end);

    let final_books = books_of(&fluid, &dust, held_water_body + held_water_wheels, held_slurry, opened);
    max_gap = max_gap.max(final_books.gap);
    nan += fluid.non_finite_count();
    nan += dust.air_nan_count();
    for v in dust.air_velocity_xyz() {
        if !v.is_finite() {
            nan = nan.saturating_add(1);
        }
    }
    for v in frame.moisture.iter().chain(&frame.film).chain(&frame.dust_settled) {
        if !v.is_finite() {
            nan = nan.saturating_add(1);
        }
    }

    let puddle_mean = if puddle_n > 0.0 {
        puddle_depth / puddle_n
    } else {
        0.0
    };
    let puddle_hole = if puddle_n > 0.0 {
        puddle_contact / puddle_n
    } else {
        0.0
    };
    let slurry_mean = if slurry_n > 0.0 {
        slurry_depth / slurry_n
    } else {
        0.0
    };
    let slurry_hole = if slurry_n > 0.0 {
        slurry_contact / slurry_n
    } else {
        0.0
    };
    let step_mean = if timed > 0.0 { step_ms / timed } else { 0.0 };
    let air_mean = if timed > 0.0 { air_ms / timed } else { 0.0 };
    let step_p95 = percentile_95(&mut step_samples);
    let hardware = hardware_line();
    let dust_gap = (dust.emitted_m3() as f64
        - dust.dust_air_m3() as f64
        - dust.dust_settled_m3() as f64)
        .abs();

    println!(
        "e2e_niva puddle depth {puddle_mean:.4} m contact {puddle_hole:.4} m samples {puddle_n:.0}\n\
         e2e_niva bow head {bow:.4} m ahead {bow_ahead:.4} side {bow_side:.4}\n\
         e2e_niva slurry depth {slurry_mean:.4} m contact {slurry_hole:.4} spray {slurry_spray_at_exit:.6} m3 mass {slurry_mass_at_exit:.4} kg samples {slurry_n:.0}\n\
         e2e_niva body drained {body_peak:.6} m3 wheels drained {wheel_peak:.6} m3 end body {held_water_body:.6} wheels {held_water_wheels:.6}\n\
         e2e_niva wet track film {track_film:.6} m3 moisture {track_moisture:.4} end film {film_end:.6} end moisture {moist_end:.4}\n\
         e2e_niva dust emitted {dry_emitted:.6} air {:.6} settled {:.6} wet settled {settled_wet:.6} map {track_dust_map:.6} wet lift {wet_lift:.6}\n\
         e2e_niva water {:.6} slurry {:.6} soil {:.6} dust {:.6} sum {:.6} gap {:.8} max gap {:.8}\n\
         e2e_niva nan {nan}\n\
         e2e_niva step ms {step_mean:.3} air step ms {air_mean:.3}\n\
         E2E_STEPS n={timed:.0} dt={DT:.6}\n\
         E2E_MEAN ms={step_mean:.3}\n\
         E2E_P95 ms={step_p95:.3}\n\
         E2E_BUDGET ms={E2E_STEP_BUDGET_MS:.3}\n\
         E2E_AIR ms={air_mean:.3}\n\
         E2E_AIR_BUDGET ms={AIR_RELEASE_BUDGET_MS:.3}\n\
         E2E_HARDWARE {hardware}",
        dust.dust_air_m3(),
        dust.dust_settled_m3(),
        final_books.water,
        final_books.slurry,
        final_books.soil,
        final_books.dust,
        final_books.sum,
        final_books.gap,
        max_gap
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());

    #[cfg(not(debug_assertions))]
    print_through_drive();

    assert!(
        (puddle_n - 18.0).abs() < 0.5,
        "puddle samples {puddle_n}"
    );
    assert!(
        (puddle_mean - 0.0893).abs() < 0.003,
        "puddle depth {puddle_mean:.4} m"
    );
    assert!(
        (puddle_hole - 0.0020).abs() < 0.002 && puddle_hole < puddle_mean - 0.04,
        "tire hole {puddle_hole:.4} m puddle {puddle_mean:.4} m"
    );
    assert!(
        (bow - 0.0403).abs() < 0.003,
        "bow head {bow:.4} ahead {bow_ahead:.4} side {bow_side:.4}"
    );
    assert!(
        (bow_ahead - 0.1092).abs() < 0.004 && (bow_side - 0.0689).abs() < 0.004,
        "bow ahead {bow_ahead:.4} side {bow_side:.4}"
    );
    assert!(
        (slurry_n - 8.0).abs() < 0.5,
        "slurry samples {slurry_n}"
    );
    assert!(
        (slurry_mean - 0.0536).abs() < 0.003,
        "slurry depth {slurry_mean:.4} m"
    );
    assert!(
        (slurry_hole - 0.0128).abs() < 0.003 && slurry_hole < slurry_mean - 0.02,
        "slurry rut {slurry_hole:.4} m sheet {slurry_mean:.4} m"
    );
    assert!(
        (slurry_spray_at_exit - 0.0171).abs() < 5.0e-4,
        "slurry spray {slurry_spray_at_exit:.6} m3"
    );
    let expect_mass = (slurry_spray_at_exit as f32) * 1450.0;
    assert!(
        (slurry_mass_at_exit - expect_mass).abs() < 1.0e-2,
        "spray mass {slurry_mass_at_exit:.4} expected {expect_mass:.4}"
    );
    assert!(
        (body_peak - 0.0126).abs() < 1.0e-4,
        "body water {body_peak:.6}"
    );
    assert!(
        (wheel_peak - 0.0154).abs() < 1.0e-4,
        "wheel water {wheel_peak:.6}"
    );
    assert!(
        held_water_body < 1.0e-5 && held_water_wheels < 1.0e-5,
        "still holding body {held_water_body:.6} wheels {held_water_wheels:.6}"
    );
    assert!(saw_drain_end, "the exit never finished");
    assert!(
        (track_film - 0.02677).abs() < 0.001,
        "wet track film {track_film:.6} m3"
    );
    assert!(
        (track_moisture - 0.78).abs() < 0.01,
        "wet track moisture {track_moisture:.4}"
    );
    assert!(
        return_m.iter().all(|m| *m >= 0.55),
        "return moisture {return_m:?}"
    );
    assert!(dry_started, "never reached dry ground");
    assert!(
        (dry_emitted - 0.009188).abs() < 2.0e-4,
        "dry dust {dry_emitted:.6} m3"
    );
    assert!(
        (settled_wet as f64 - 0.001519).abs() < 1.5e-4,
        "settled on the wet track {settled_wet:.6} before {settled_wet_before:.6}"
    );
    assert!(
        (settled_wet as f64) + 1.0e-8 >= settled_wet_before,
        "settled left the wet track {settled_wet_before:.6} -> {settled_wet:.6}"
    );
    assert!(
        dust.dust_air_m3() as f64 <= air_before + 1.0e-8,
        "air rose on the wet return {} -> {}",
        air_before,
        dust.dust_air_m3()
    );
    assert!(
        (track_dust_map as f64 - 0.002941).abs() < 5.0e-4,
        "contract dust on the wet track {track_dust_map:.6}"
    );
    assert_eq!(wet_lift, 0.0, "wet lift {wet_lift}");
    assert!(dust_gap < 1.0e-6, "dust ledger gap {dust_gap:.8}");
    assert!(
        (final_books.water - 4.6076).abs() < 1.0e-3,
        "water {:.6}",
        final_books.water
    );
    assert!(
        (final_books.slurry - 0.8834).abs() < 1.0e-3,
        "slurry {:.6}",
        final_books.slurry
    );
    assert!(
        (final_books.soil - 0.9723).abs() < 1.0e-3,
        "soil {:.6}",
        final_books.soil
    );
    assert!(
        (final_books.dust - 0.02407).abs() < 1.0e-3,
        "dust {:.6}",
        final_books.dust
    );
    assert!(
        final_books.gap < VOLUME_CONVERGENCE,
        "gap {:.8}",
        final_books.gap
    );
    assert!(final_books.soil > 0.0, "soil {}", final_books.soil);
    assert_eq!(nan, 0, "nan {nan}");
    assert!(step_mean.is_finite() && step_mean > 0.0, "step ms {step_mean}");
    assert!(
        air_mean < AIR_STEP_BUDGET_MS as f64,
        "air step {air_mean:.3} ms over {AIR_STEP_BUDGET_MS}"
    );
    #[cfg(not(debug_assertions))]
    {
        assert!(
            step_mean <= E2E_STEP_BUDGET_MS,
            "e2e step {step_mean:.3} ms over {E2E_STEP_BUDGET_MS} from the CI mean"
        );
        assert!(
            air_mean <= AIR_RELEASE_BUDGET_MS,
            "air step {air_mean:.3} ms over {AIR_RELEASE_BUDGET_MS}"
        );
    }
}

/// Release breakdown of [`ThroughDrive::step`] at dt = 1/30. Printed for the CI log.
fn print_through_drive() {
    let mut drive = ThroughDrive::new();
    let dt = 1.0 / 30.0;
    for _ in 0..24 {
        drive.step(dt);
    }
    let n = 96u32;
    let mut totals = Vec::with_capacity(n as usize);
    let mut acc = DriveParts::default();
    for _ in 0..n {
        drive.step(dt);
        let p = drive.last_parts;
        totals.push(p.total_ms);
        acc.water_ms += p.water_ms;
        acc.slurry_ms += p.slurry_ms;
        acc.soil_ms += p.soil_ms;
        acc.air_ms += p.air_ms;
        acc.dust_ms += p.dust_ms;
        acc.restrict_ms += p.restrict_ms;
        acc.window_ms += p.window_ms;
        acc.prep_ms += p.prep_ms;
        acc.sheet_rest_ms += p.sheet_rest_ms;
        acc.other_ms += p.other_ms;
        acc.total_ms += p.total_ms;
        acc.window_hash += p.window_hash;
    }
    let k = f64::from(n);
    let mean = acc.total_ms / k;
    let p95 = percentile_95(&mut totals);
    let solver = acc.solver_ms() / k;
    let hardware = hardware_line();
    println!(
        "DRIVE_STEPS n={n} dt={dt:.6}\n\
         DRIVE_MEAN ms={mean:.3}\n\
         DRIVE_P95 ms={p95:.3}\n\
         DRIVE_HARDWARE {hardware}\n\
         DRIVE_WATER ms={:.3}\n\
         DRIVE_SLURRY ms={:.3}\n\
         DRIVE_SOIL ms={:.3}\n\
         DRIVE_AIR ms={:.3}\n\
         DRIVE_DUST ms={:.3}\n\
         DRIVE_RESTRICT ms={:.3}\n\
         DRIVE_WINDOW ms={:.3}\n\
         DRIVE_WINDOW_HASH n={}\n\
         DRIVE_PREP ms={:.3}\n\
         DRIVE_SHEET_REST ms={:.3}\n\
         DRIVE_OTHER ms={:.3}\n\
         DRIVE_SOLVER ms={solver:.3}\n\
         DRIVE_WATER_BUDGET ms={DRIVE_WATER_BUDGET_MS:.3}\n\
         DRIVE_SOLVER_BUDGET ms={DRIVE_SOLVER_BUDGET_MS:.3}\n\
         DRIVE_P95_BUDGET ms={DRIVE_P95_BUDGET_MS:.3}\n\
         DRIVE_DUST_BUDGET ms={DRIVE_DUST_BUDGET_MS:.3}",
        acc.water_ms / k,
        acc.slurry_ms / k,
        acc.soil_ms / k,
        acc.air_ms / k,
        acc.dust_ms / k,
        acc.restrict_ms / k,
        acc.window_ms / k,
        acc.window_hash,
        acc.prep_ms / k,
        acc.sheet_rest_ms / k,
        acc.other_ms / k,
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let water = acc.water_ms / k;
    let dust = acc.dust_ms / k;
    #[cfg(not(debug_assertions))]
    {
        assert!(
            solver <= DRIVE_SOLVER_BUDGET_MS,
            "DRIVE_SOLVER {solver:.3} ms over {DRIVE_SOLVER_BUDGET_MS}"
        );
        assert!(
            p95 <= DRIVE_P95_BUDGET_MS,
            "DRIVE_P95 {p95:.3} ms over {DRIVE_P95_BUDGET_MS}"
        );
        assert!(
            dust <= DRIVE_DUST_BUDGET_MS,
            "DRIVE_DUST {dust:.3} ms over {DRIVE_DUST_BUDGET_MS}"
        );
        assert!(
            water <= DRIVE_WATER_BUDGET_MS,
            "DRIVE_WATER {water:.3} ms over {DRIVE_WATER_BUDGET_MS}"
        );
    }
}

/// Burn extra time inside a measurement when `PERF_STRETCH` is above 1.
/// CI leaves it unset. A proof run sets it on the pull-request side only.
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

/// Open 2 cm sheet and open wind, the trimmed branch.
///
/// Gravity at [`FINE_CELL_M`] with no wall is one CFL pass, no divergence
/// projection, and flat slow tiles left asleep. Open wind does not build the
/// pressure filters. Volume stays inside 1e-3, nothing is NaN, and the
/// limiter keeps the speed.
#[test]
fn fine_open() {
    let res = FINE_RES;
    let cell = FINE_CELL_M;
    let n = res * res;
    let mut h = vec![0.02f32; n];
    let mut u = vec![0.02f32; n];
    let mut v = vec![0.0f32; n];
    let mut bed = vec![0.0f32; n];
    let nu = vec![1.0f32; n];
    let gate = vec![1u8; n];
    for j in 24..48 {
        for i in 24..48 {
            let idx = j * res + i;
            bed[idx] = -0.008 * j as f32;
            h[idx] = 0.06;
        }
    }
    // Far tile of the flat film. The pond does not reach it in this run, so a
    // sleeping tile keeps the warm start.
    let mut sleep = Vec::new();
    for j in 152..160 {
        for i in 152..160 {
            sleep.push(j * res + i);
        }
    }
    let before = sheet_volume(&h, cell);
    let sleep_h: Vec<f32> = sleep.iter().map(|&idx| h[idx]).collect();
    let sleep_u: Vec<f32> = sleep.iter().map(|&idx| u[idx]).collect();
    let dt = 1.0 / 30.0;
    for _ in 0..12 {
        step_sheet(
            &mut h,
            &mut u,
            &mut v,
            &bed,
            &nu,
            &gate,
            res,
            cell,
            dt,
            SheetDrive::Gravity { vorticity: 0.02 },
        );
    }
    let after = sheet_volume(&h, cell);
    let gap = (after - before).abs();
    assert!(before > 1.0e-4, "gravity volume {before}");
    assert!(
        gap / before < 1.0e-3,
        "gravity volume {before} -> {after} gap {gap}"
    );
    let mut nan = non_finite(&h) + non_finite(&u) + non_finite(&v);
    assert_eq!(nan, 0, "gravity nan {nan}");
    assert_plausible_speed(&u, &v, 1.6);
    let mut downhill = 0.0f32;
    for j in 24..80 {
        for i in 16..64 {
            let idx = j * res + i;
            if h[idx] > 1.0e-4 {
                downhill = downhill.max(v[idx]);
            }
        }
    }
    assert!(downhill > 0.02, "downhill v {downhill}");
    for (k, &idx) in sleep.iter().enumerate() {
        assert_eq!(h[idx], sleep_h[k], "sleeping depth moved at {idx}");
        assert_eq!(u[idx], sleep_u[k], "sleeping speed moved at {idx}");
        assert_eq!(v[idx], 0.0, "sleeping cross speed at {idx}");
    }

    let wres = AIR_RES / 2;
    let wcell = 2.0 * (WINDOW_RES as f32 * COARSE_CELL_M / AIR_RES as f32);
    let wn = wres * wres;
    let mut wh = vec![0.0f32; wn];
    let mut wu = vec![0.0f32; wn];
    let mut wv = vec![0.0f32; wn];
    let wbed = vec![0.0f32; wn];
    let wnu = vec![0.4f32; wn];
    let wgate = vec![1u8; wn];
    for j in 24..36 {
        for i in 12..24 {
            wh[j * wres + i] = 0.04;
        }
    }
    let wbefore = sheet_volume(&wh, wcell);
    let wind_x = 1.2f32;
    let wind_z = -0.3f32;
    for _ in 0..40 {
        step_sheet(
            &mut wh,
            &mut wu,
            &mut wv,
            &wbed,
            &wnu,
            &wgate,
            wres,
            wcell,
            dt,
            SheetDrive::Wind {
                x: wind_x,
                z: wind_z,
                vorticity: 0.35,
            },
        );
    }
    let wafter = sheet_volume(&wh, wcell);
    let wgap = (wafter - wbefore).abs();
    assert!(wbefore > 1.0e-4, "wind volume {wbefore}");
    assert!(
        wgap / wbefore < 1.0e-3,
        "wind volume {wbefore} -> {wafter} gap {wgap}"
    );
    nan = non_finite(&wh) + non_finite(&wu) + non_finite(&wv);
    assert_eq!(nan, 0, "wind nan {nan}");
    let lim = wind_x.hypot(wind_z).max(1.6);
    assert_plausible_speed(&wu, &wv, lim);
    let mut mass = 0.0f64;
    let mut mass_u = 0.0f64;
    for j in 0..wres {
        for i in 0..wres {
            let s = f64::from(wh[j * wres + i].max(0.0));
            mass += s;
            mass_u += s * f64::from(wu[j * wres + i]);
        }
    }
    let mean_u = if mass > 0.0 { mass_u / mass } else { 0.0 };
    assert!(mean_u > 0.05, "wind mean u {mean_u}");
}

fn sheet_volume(h: &[f32], cell: f32) -> f64 {
    let area = f64::from(cell) * f64::from(cell);
    h.iter().map(|s| f64::from((*s).max(0.0))).sum::<f64>() * area
}

fn non_finite(xs: &[f32]) -> u32 {
    xs.iter().filter(|v| !v.is_finite()).count() as u32
}

fn assert_plausible_speed(u: &[f32], v: &[f32], lim: f32) {
    let cap = lim + 1.0e-3;
    for (uu, vv) in u.iter().zip(v) {
        assert!(
            uu.is_finite() && uu.abs() <= cap,
            "u {uu} over {lim}"
        );
        assert!(
            vv.is_finite() && vv.abs() <= cap,
            "v {vv} over {lim}"
        );
    }
}

fn percentile_95(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((samples.len() as f64) * 0.95).ceil() as usize;
    samples[(idx.saturating_sub(1)).min(samples.len() - 1)]
}

fn hardware_line() -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let model = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name"))
                .and_then(|rest| rest.split(':').nth(1))
                .map(|name| name.trim().to_string())
        })
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown cpu".to_string());
    format!("{model}, {cores} cores")
}
