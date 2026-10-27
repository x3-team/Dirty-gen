use silt::{dig_input, Input, World};

#[test]
fn firm_ground_accelerates() {
    let mut world = World::flat(0.12);
    let start = world.vehicle.pos;
    world.run_seconds(3.0, |_| Input {
        throttle: 1.0,
        brake: 0.0,
        steer: 0.0,
    });
    let travelled = (world.vehicle.pos - start).length();
    eprintln!(
        "firm speed {:.2} m/s  travelled {:.2} m  sink {:.3}",
        world.vehicle.speed(),
        travelled,
        world.telemetry.max_sink
    );
    assert!(
        world.telemetry.max_speed > 4.0,
        "firm max speed {}",
        world.telemetry.max_speed
    );
    assert!(travelled > 6.0, "firm travel {travelled}");
}

#[test]
fn soft_mud_is_slower_than_firm() {
    let mut firm = World::flat(0.12);
    let mut soft = World::flat(0.98);
    let drive = |_| Input {
        throttle: 1.0,
        brake: 0.0,
        steer: 0.0,
    };
    firm.run_seconds(3.5, drive);
    soft.run_seconds(3.5, drive);
    eprintln!(
        "compare firm {:.2} m / {:.2} m/s   soft {:.2} m / {:.2} m/s  soft sink {:.3} rut {:.3}",
        firm.telemetry.distance,
        firm.telemetry.max_speed,
        soft.telemetry.distance,
        soft.telemetry.max_speed,
        soft.telemetry.max_sink,
        soft.soil.max_rut_depth()
    );
    assert!(
        soft.telemetry.distance < firm.telemetry.distance * 0.65,
        "soft should bog relative to firm"
    );
}

#[test]
fn scripted_drive_cuts_ruts_and_picks_up_mud() {
    let mut world = World::demo();
    let rut0 = world.soil.max_rut_depth();
    let mud0 = world.vehicle.body_mud_mass();
    let span0 = world.soil.rut_track_span();
    world.run_seconds_with(12.0, silt::demo_drive);
    let rut1 = world.soil.max_rut_depth();
    let mud1 = world.vehicle.body_mud_mass();
    let wheels = world.vehicle.wheel_mud_mass();
    let span1 = world.soil.rut_track_span();
    eprintln!(
        "demo rut {:.3}->{:.3}  body mud {:.3}->{:.3}  wheel mud {:.3}  span {:.2}->{:.2}  \
         max speed {:.2}  min soft {:.2}  splashes {}  pos {:?} sink {:.3}",
        rut0,
        rut1,
        mud0,
        mud1,
        wheels,
        span0,
        span1,
        world.telemetry.max_speed,
        world.telemetry.min_speed_in_soft,
        world.telemetry.splash_bursts,
        world.vehicle.pos,
        world.telemetry.max_sink
    );
    assert!(rut1 > rut0 + 0.04, "rut depth did not grow");
    assert!(mud1 > mud0 + 0.12, "body mud did not accumulate");
    assert!(wheels > 0.05, "tires stayed clean");
    assert!(span1 > span0 + 3.0, "track did not lengthen");
    assert!(world.telemetry.splash_bursts > 5, "puddle did not splash");
    assert!(
        world.telemetry.min_speed_in_soft < 0.8,
        "never bogged in the puddle"
    );
    assert!(
        world.telemetry.distance > 16.0,
        "did not drive the loop, distance {}",
        world.telemetry.distance
    );
    let left_start = world.vehicle.pos.x.abs() > 2.0 || (world.vehicle.pos.z + 8.2).abs() > 4.0;
    assert!(
        left_start,
        "buggy never left the gate, pos {:?}",
        world.vehicle.pos
    );
}

#[test]
fn holding_throttle_digs_in() {
    let mut escaped = World::demo();
    escaped.run_seconds_with(7.5, silt::demo_drive);
    let mut dug = World::demo();
    dug.run_seconds(7.5, dig_input);
    eprintln!(
        "escape z {:.2} speed {:.2} rut {:.3}   dug z {:.2} speed {:.2} rut {:.3} mud {:.2}",
        escaped.vehicle.pos.z,
        escaped.vehicle.speed(),
        escaped.soil.max_rut_depth(),
        dug.vehicle.pos.z,
        dug.vehicle.speed(),
        dug.soil.max_rut_depth(),
        dug.vehicle.body_mud_mass()
    );
    assert!(
        dug.soil.max_rut_depth() > 0.06,
        "spinning in the puddle should cut a rut, got {}",
        dug.soil.max_rut_depth()
    );
    assert!(
        dug.vehicle.speed() < 0.75,
        "spinning in the dug rut should stay bogged, speed {}",
        dug.vehicle.speed()
    );
    assert!(
        dug.vehicle.pos.z > -4.0,
        "bogged buggy should still be in the mud, z {}",
        dug.vehicle.pos.z
    );
    assert!(dug.vehicle.body_mud_mass() > 0.1);
}

#[test]
fn crest_impact_cuts_rut_and_sprays() {
    let mut world = World::demo();
    let rut0 = world.soil.max_rut_depth();
    world.run_seconds_with(12.0, silt::demo_drive);
    eprintln!(
        "impact depth {:.3} sprays {} pitch {:.2} t {:.2} pos {:?} rut {:.3}->{:.3} mud {:.2}",
        world.telemetry.impact_depth,
        world.telemetry.impact_sprays,
        world.telemetry.max_air_pitch,
        world.telemetry.impact_time,
        world.vehicle.pos,
        rut0,
        world.soil.max_rut_depth(),
        world.vehicle.body_mud_mass()
    );
    assert!(
        world.telemetry.impact_depth > 0.06,
        "impact did not dig the heightfield, depth {}",
        world.telemetry.impact_depth
    );
    assert!(
        world.soil.max_rut_depth() > rut0 + 0.06,
        "rut did not grow from the landing"
    );
    assert!(
        world.telemetry.impact_sprays > 12,
        "impact did not throw spray, {}",
        world.telemetry.impact_sprays
    );
    assert!(
        world.telemetry.max_air_pitch > 1.2,
        "airborne pitch {:.2} is a hop, not a tumble",
        world.telemetry.max_air_pitch
    );
}

#[test]
fn bike_rides_and_compresses_beside_the_buggy() {
    let mut world = World::demo();
    let start = world.bike.pos;
    let mut min_c = 1.0_f32;
    let mut max_c = 0.0_f32;
    let mut launched = false;
    for _ in 0..(8.0 * 60.0) as u32 {
        let bike = silt::demo_drive_bike(&world);
        world.step_bike(1.0 / 60.0, &bike);
        let buggy = silt::demo_drive(&world);
        world.step(1.0 / 60.0, &buggy);
        for w in &world.bike.wheels {
            min_c = min_c.min(w.compression);
            max_c = max_c.max(w.compression);
        }
        if world.bike.jump_hold > 0.0 || world.bike.pitch.abs() > 0.8 {
            launched = true;
        }
    }
    let travelled = (world.bike.pos - start).length();
    assert!(travelled > 8.0, "bike did not ride the loop, {travelled}");
    assert!(
        max_c > min_c + 0.04,
        "suspension did not move, {min_c:.3}..{max_c:.3}"
    );
    assert!(launched, "bike did not leave the ground on the crest");
}

#[test]
fn rolling_tires_leave_a_pair_of_grooves() {
    let mut world = World::demo();
    world.run_seconds_with(4.0, silt::demo_drive);
    // The south cap is already behind the buggy. A constant-z slice cuts both tires.
    let z = -4.0;
    let mut prev2 = 0.0_f32;
    let mut prev = 0.0_f32;
    let mut peaks: Vec<(f32, f32)> = Vec::new();
    for k in -50..20 {
        let x = k as f32 * 0.15;
        let rut = world.soil.sample(x, z).rut;
        if prev >= prev2 && prev >= rut && prev > 0.03 {
            let px = (k as f32 - 1.0) * 0.15;
            if peaks.last().is_none_or(|(lx, _)| px - *lx > 0.45) {
                peaks.push((px, prev));
            }
        }
        prev2 = prev;
        prev = rut;
    }
    eprintln!("groove peaks at z={z}: {peaks:?} pos {:?}", world.vehicle.pos);
    assert!(
        peaks.len() >= 2,
        "expected a pair of tire grooves behind the buggy, got {peaks:?}"
    );
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let sep = (peaks[0].0 - peaks[1].0).abs();
    assert!(
        (0.9..2.4).contains(&sep),
        "grooves {:?} are not one axle apart",
        peaks
    );
    assert!(
        (world.vehicle.pos.z - z).abs() > 2.0 || (world.vehicle.pos.x - peaks[0].0).abs() > 2.0,
        "buggy is still sitting on the grooves"
    );
    for (x, depth) in [peaks[0], peaks[1]] {
        let floor = world.soil.sample(x, z).height;
        let left = world.soil.sample(x - 0.42, z).height;
        let right = world.soil.sample(x + 0.42, z).height;
        let shoulder = left.max(right);
        assert!(
            shoulder > floor + 0.02,
            "groove {depth:.3} at x={x:.2} has no shoulder: floor {floor:.3} shoulder {shoulder:.3}"
        );
    }
}

#[test]
fn impact_chunks_land_on_the_heightfield() {
    let mut world = World::demo();
    world.run_seconds_with(6.4, silt::demo_drive);
    let grounded = world
        .particles
        .items
        .iter()
        .filter(|p| p.alive && p.grounded && p.kind == 3)
        .count();
    let airborne = world
        .particles
        .items
        .iter()
        .filter(|p| p.alive && !p.grounded && (p.kind == 3 || p.kind == 4))
        .count();
    eprintln!(
        "chunks grounded {grounded} still airborne {airborne} impact t {:.2}",
        world.telemetry.impact_time
    );
    assert!(
        grounded >= 3,
        "large clumps should be sitting on the soil, grounded {grounded} airborne {airborne}"
    );
}
