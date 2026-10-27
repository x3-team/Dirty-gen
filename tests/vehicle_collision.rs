use silt::{demo_drive, demo_drive_bike, Input, World};

#[test]
fn buggy_hits_bike_and_they_separate() {
    let mut world = World::flat(0.12);
    let ground = world.soil.sample(0.0, 0.0).height;
    world.vehicle.pos.x = 0.0;
    world.vehicle.pos.y = ground + 0.72;
    world.vehicle.pos.z = -2.5;
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.vel.x = 0.0;
    world.vehicle.vel.y = 0.0;
    world.vehicle.vel.z = 8.0;
    world.vehicle.yaw_rate = 0.0;
    world.vehicle.pitch_rate = 0.0;
    world.vehicle.roll_rate = 0.0;
    world.vehicle.collision_loose = 0.0;
    world.bike.pos.x = 0.48;
    world.bike.pos.y = ground + 0.7;
    world.bike.pos.z = 0.55;
    world.bike.yaw = 0.35;
    world.bike.pitch = 0.0;
    world.bike.roll = 0.0;
    world.bike.vel.x = 0.0;
    world.bike.vel.y = 0.0;
    world.bike.vel.z = 0.0;
    world.bike.yaw_rate = 0.0;
    world.bike.pitch_rate = 0.0;
    world.bike.roll_rate = 0.0;
    world.bike.collision_loose = 0.0;

    let mut max_yaw = 0.0_f32;
    let mut max_bike_speed = 0.0_f32;
    let mut min_buggy_speed = 99.0_f32;
    for _ in 0..80 {
        world.step_bike(1.0 / 60.0, &Input::default());
        world.step(1.0 / 60.0, &Input::default());
        max_yaw = max_yaw
            .max(world.vehicle.yaw_rate.abs())
            .max(world.bike.yaw_rate.abs());
        max_bike_speed = max_bike_speed.max(world.bike.speed());
        min_buggy_speed = min_buggy_speed.min(world.vehicle.speed());
    }
    let bodies = [world.vehicle.contact_body(), world.bike.contact_body()];
    let pen = silt::contact::overlap(&bodies);
    eprintln!(
        "bike speed {max_bike_speed:.2} buggy min {min_buggy_speed:.2} yaw {max_yaw:.2} pen {pen:.3} \
         loose {:.2}/{:.2} pos {:?} {:?}",
        world.vehicle.collision_loose,
        world.bike.collision_loose,
        world.vehicle.pos,
        world.bike.pos
    );
    assert!(
        max_bike_speed > 1.0,
        "bike did not take momentum, peak speed {max_bike_speed:.2}"
    );
    assert!(
        min_buggy_speed < 7.2,
        "buggy did not slow, slowest {min_buggy_speed:.2}"
    );
    assert!(
        max_yaw > 0.25,
        "impact did not spin either chassis, {max_yaw:.2}"
    );
    assert!(pen < 0.04, "chassis left overlapping by {pen:.3} m");
}

#[test]
fn drop_onto_the_cage_tips_the_buggy() {
    let mut world = World::flat(0.12);
    let ground = world.soil.sample(0.0, 0.0).height;
    world.vehicle.pos.x = 0.0;
    world.vehicle.pos.y = ground + 0.7;
    world.vehicle.pos.z = 0.0;
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.vel.x = 0.0;
    world.vehicle.vel.y = 0.0;
    world.vehicle.vel.z = 0.0;
    world.vehicle.yaw_rate = 0.0;
    world.vehicle.pitch_rate = 0.0;
    world.vehicle.roll_rate = 0.0;
    world.bike.pos.x = 0.42;
    world.bike.pos.y = ground + 1.85;
    world.bike.pos.z = 0.15;
    world.bike.yaw = 0.0;
    world.bike.pitch = 0.2;
    world.bike.roll = 0.0;
    world.bike.vel.x = 0.2;
    world.bike.vel.y = -4.5;
    world.bike.vel.z = 0.4;
    world.bike.yaw_rate = 0.0;
    world.bike.pitch_rate = 0.0;
    world.bike.roll_rate = 0.0;

    let mut tipped = 0.0_f32;
    for _ in 0..50 {
        world.step_bike(1.0 / 60.0, &Input::default());
        world.step(1.0 / 60.0, &Input::default());
        tipped = tipped
            .max(world.vehicle.roll_rate.abs())
            .max(world.vehicle.pitch_rate.abs())
            .max(world.bike.roll_rate.abs());
    }
    let pen = silt::contact::overlap(&[world.vehicle.contact_body(), world.bike.contact_body()]);
    eprintln!(
        "tip rate {tipped:.2} roll {:.2} pitch {:.2} pen {pen:.3}",
        world.vehicle.roll, world.vehicle.pitch
    );
    assert!(
        tipped > 0.35,
        "landing on the cage did not tip, rate {tipped:.2}"
    );
    assert!(pen < 0.04, "still inside each other, pen {pen:.3}");
}

#[test]
fn hard_landing_still_stamps_with_both_vehicles() {
    let mut world = World::demo();
    let rut0 = world.soil.max_rut_depth();
    for _ in 0..(8.0 * 60.0) as u32 {
        let bike = demo_drive_bike(&world);
        world.step_bike(1.0 / 60.0, &bike);
        let buggy = demo_drive(&world);
        world.step(1.0 / 60.0, &buggy);
    }
    eprintln!(
        "impact depth {:.3} sprays {} pitch {:.2} rut {:.3}->{:.3}",
        world.telemetry.impact_depth,
        world.telemetry.impact_sprays,
        world.telemetry.max_air_pitch,
        rut0,
        world.soil.max_rut_depth()
    );
    assert!(
        world.telemetry.impact_depth > 0.06,
        "landing no longer stamps the heightfield, depth {}",
        world.telemetry.impact_depth
    );
    assert!(
        world.telemetry.impact_sprays > 8,
        "landing no longer throws mud, sprays {}",
        world.telemetry.impact_sprays
    );
    assert!(
        world.telemetry.max_air_pitch > 1.0,
        "buggy no longer tumbles off the crest"
    );
}
