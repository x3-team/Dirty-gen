//! Niva wheels in the puddle. The rut is the tire force on wet sand, the
//! spoil is a berm, and the tread sits 10–15 cm under the free surface.

use silt::render_contract::{publish, ContractFrame, RENDER_CONTRACT_VERSION};
use silt::soil::rut_depth_from_force;
use silt::vehicle::{Vehicle, WHEEL_RADIUS};
use silt::world::ThroughDrive;

fn tire_bottom(vehicle: &Vehicle, index: usize) -> glam::Vec3 {
    let ori = vehicle.orientation();
    let center = vehicle.pos + ori * vehicle.wheel_center_local(index);
    glam::Vec3::new(center.x, center.y - WHEEL_RADIUS, center.z)
}

fn nan_count(frame: &ContractFrame) -> u32 {
    let maps = [
        frame.ground_rut.as_slice(),
        frame.moisture.as_slice(),
        frame.film.as_slice(),
        frame.flow.as_slice(),
        frame.dust_settled.as_slice(),
        frame.crust.as_slice(),
        frame.wet_edge.as_slice(),
        frame.foam_crest.as_slice(),
        frame.dust_fog.as_slice(),
        frame.air_velocity.as_slice(),
    ];
    let mut n = 0u32;
    for map in maps {
        for v in map {
            if !v.is_finite() {
                n += 1;
            }
        }
    }
    for slot in &frame.spray {
        if slot.pos_mass.iter().any(|c| !c.is_finite())
            || slot.vel_age.iter().any(|c| !c.is_finite())
        {
            n += 1;
        }
    }
    n
}

#[test]
fn puddle_wheels_sit_in_a_force_rut() {
    let mut drive = ThroughDrive::new();
    let dt = 1.0 / 30.0;
    // South of the cosmetic slurry groove (z > -13.5). The fronts are in the
    // sheet the bug measured at about 7 cm, with the rut still under water.
    while drive.world.vehicle.pos.z < -14.05 {
        drive.step(dt);
    }

    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &drive.world.soil,
        &drive.world.particles.items,
        [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
        drive.world.vehicle.pos.z,
        1,
        Some(&drive.fluid),
        Some(drive.dust()),
    );

    let res = frame.header.window_res.max(1) as usize;
    let cell = frame.header.cell_m;
    let origin_x = frame.header.origin_x;
    let origin_z = frame.header.origin_z;
    let mut surface_sum = 0.0_f32;
    let mut seat_sum = 0.0_f32;
    let mut n = 0.0_f32;
    let mut rut_max = 0.0_f32;
    for i in 0..4 {
        let bottom = tire_bottom(&drive.world.vehicle, i);
        let ii = ((bottom.x - origin_x) / cell).floor() as isize;
        let jj = ((bottom.z - origin_z) / cell).floor() as isize;
        if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
            continue;
        }
        let idx = jj as usize * res + ii as usize;
        let ground = frame.ground_rut[idx * 4];
        let rut = frame.ground_rut[idx * 4 + 1];
        let placed = frame.ground_rut[idx * 4 + 3];
        let water = frame.film[idx * 2];
        let slurry = frame.film[idx * 2 + 1];
        // Visible water surface: solid ground, then slurry, then the sheet.
        let surface = ground + slurry + water;
        let seat = surface - bottom.y;
        eprintln!(
            "wheel {i} surface {surface:.4} seat {seat:.4} rut {rut:.4} water {water:.4} bottom {:.4}",
            bottom.y
        );
        if water > 0.02 && rut > 0.02 {
            assert!(
                (0.10..=0.15).contains(&seat),
                "wheel {i} seat {seat:.4} m is outside 0.10–0.15 m"
            );
            surface_sum += surface;
            seat_sum += seat;
            n += 1.0;
            rut_max = rut_max.max(rut);
        }
    }

    let vols = drive.world.soil.volumes();
    let gap = (vols.cut - vols.placed).abs();
    let nans = nan_count(&frame);
    let surface = if n > 0.0 { surface_sum / n } else { 0.0 };
    let seat = if n > 0.0 { seat_sum / n } else { 0.0 };
    println!(
        "water surface {surface:.4} m  tire seat {seat:.4} m  rut depth {rut_max:.4} m  berm {:.4} m3  soil volume gap {gap:.6} m3  NaN count {nans}",
        vols.berm
    );

    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(RENDER_CONTRACT_VERSION, 2);
    assert!(
        n >= 4.0,
        "wheels in a wet rut: {n}, chassis z {:.2}",
        drive.world.vehicle.pos.z
    );
    assert!(
        (0.10..=0.15).contains(&seat),
        "tire seat {seat:.4} m is outside 0.10–0.15 m"
    );
    // Sheet beside the tracks. The Bekker sample uses this, not the column ponded in the cut.
    let sheet = drive.fluid.sample(2.2, -15.0).water_m;
    let force_rut = rut_depth_from_force(
        drive.puddle_normal_n,
        drive.puddle_push_n,
        WHEEL_RADIUS,
        silt::vehicle::WHEEL_WIDTH,
        silt::soil::SurfaceKind::Sand,
        0.90,
        0.012,
        sheet,
    );
    eprintln!(
        "force normal {:.1} N  push {:.1} N  bekker {:.4} m  rut max {:.4} m  cut {:.5} placed {:.5}",
        drive.puddle_normal_n, drive.puddle_push_n, force_rut, rut_max, vols.cut, vols.placed
    );
    assert!(
        (force_rut - rut_max).abs() <= 0.012,
        "rut {rut_max:.4} is not the force depth {force_rut:.4} at sheet {sheet:.4}"
    );
    assert!(vols.berm > 0.0, "berm {}", vols.berm);
    assert!(vols.max_berm > 0.0, "berm height {}", vols.max_berm);
    let slack = vols.cut * 0.02 + 1.0e-4;
    assert!(
        gap < slack,
        "soil volume gap {gap:.6} cut {:.5} placed {:.5}",
        vols.cut,
        vols.placed
    );
    assert_eq!(nans, 0, "NaN count {nans}");
}
