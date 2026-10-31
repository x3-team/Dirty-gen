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
            "wheel {i} x {:.2} z {:.2} surface {surface:.4} seat {seat:.4} ground {ground:.4} placed {placed:.4} rut {rut:.4} water {water:.4} bottom {:.4}",
            bottom.x, bottom.z, bottom.y
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

struct MapSample {
    x: f32,
    water: f32,
    moist: f32,
    edge: f32,
    bed: f32,
    rut: f32,
    berm: f32,
}

fn sample_at(frame: &ContractFrame, x: f32, z: f32) -> Option<MapSample> {
    let res = frame.header.window_res.max(1) as i32;
    let i = ((x - frame.header.origin_x) / frame.header.cell_m).floor() as i32;
    let j = ((z - frame.header.origin_z) / frame.header.cell_m).floor() as i32;
    if i < 0 || j < 0 || i >= res || j >= res {
        return None;
    }
    let idx = j as usize * res as usize + i as usize;
    Some(MapSample {
        x,
        water: frame.film[idx * 2],
        moist: frame.moisture[idx],
        edge: frame.wet_edge[idx],
        bed: frame.ground_rut[idx * 4 + 2],
        rut: frame.ground_rut[idx * 4 + 1],
        berm: frame.ground_rut[idx * 4 + 3],
    })
}

fn ray_east(frame: &ContractFrame, z: f32, x0: f32) -> Vec<MapSample> {
    let mut out = Vec::new();
    let mut x = x0;
    while x <= 8.01 {
        if let Some(s) = sample_at(frame, x, z) {
            out.push(s);
        }
        x += 0.2;
    }
    out
}

/// The east shore, off the tire tracks. The floor stays near the poured bed.
/// Water falls to 0 as the bank rises, the wet edge sits past that water, and
/// pore moisture stays wet for a band beyond the shoreline.
fn assert_puddle_shore(frame: &ContractFrame, label: &str, z: f32, x0: f32) {
    let ray = ray_east(frame, z, x0);
    assert!(ray.len() > 8, "{label}: ray is short");
    let floor = ray
        .iter()
        .find(|s| (s.x - x0).abs() < 0.25)
        .expect("floor");
    assert!(
        floor.water > 0.035,
        "{label}: floor water {:.4} m is not a puddle",
        floor.water
    );
    assert!(
        floor.bed.abs() < 0.03,
        "{label}: floor bed {:.4} m is not the poured floor",
        floor.bed
    );

    // Last cell that still holds the floor column. The bank is the fall from
    // there to a dry cell, and the bed rises over that same run.
    let deep = ray
        .iter()
        .rposition(|s| s.water > floor.water * 0.88);
    let Some(deep) = deep else {
        panic!("{label}: no deep water on the east shore");
    };
    let dry = ray
        .iter()
        .skip(deep)
        .position(|s| s.water < 0.004)
        .map(|k| deep + k);
    let Some(dry) = dry else {
        panic!("{label}: water never falls to 0");
    };
    let span = ray[dry].x - ray[deep].x;
    assert!(
        span >= 0.8,
        "{label}: shore span {span:.2} m is a step, not a bank"
    );
    let crest = ray.iter().map(|s| s.bed).fold(f32::MIN, f32::max);
    assert!(
        crest > floor.bed + 0.08,
        "{label}: bank crest {:.4} did not rise off the floor {:.4}",
        crest,
        floor.bed
    );
    assert!(
        ray[dry].bed > floor.bed + 0.04,
        "{label}: bed did not rise under the falling water, floor {:.4} shore {:.4}",
        floor.bed,
        ray[dry].bed
    );
    let mut max_drop = 0.0_f32;
    for w in ray[deep..=dry].windows(2) {
        let drop = w[0].water - w[1].water;
        max_drop = max_drop.max(drop);
        assert!(
            drop > -0.008,
            "{label}: water rose on the way out of the puddle, {:.4} to {:.4} at x {:.2}",
            w[0].water,
            w[1].water,
            w[1].x
        );
    }
    assert!(
        max_drop < 0.04,
        "{label}: shore drops {max_drop:.4} m in one cell"
    );
    // The wet flag flips where moisture crosses the wet threshold, past the
    // water line. A halo puts that edge outside the water ramp.
    let mut saw_edge = false;
    for s in ray.iter().skip(dry) {
        if s.x > ray[dry].x + 2.4 {
            break;
        }
        if s.edge > 0.5 {
            saw_edge = true;
        }
    }
    assert!(saw_edge, "{label}: wet edge is missing past the shoreline");

    // Wet band: past the standing water, still above the wet threshold, then
    // back toward the pad. A one-cell jump from saturated to the pad is not a band.
    let shore_x = ray[dry].x;
    let mut band = 0.0_f32;
    let mut prev_moist = ray[dry].moist;
    let mut max_moist_drop = 0.0_f32;
    for s in ray.iter().skip(dry) {
        let drop = prev_moist - s.moist;
        max_moist_drop = max_moist_drop.max(drop);
        prev_moist = s.moist;
        if s.water < 0.004 && s.moist >= 0.45 {
            band = band.max(s.x - shore_x);
        }
    }
    assert!(
        ray[dry].moist >= 0.50,
        "{label}: shoreline moisture {:.3} is not a wet edge",
        ray[dry].moist
    );
    assert!(
        band >= 0.6,
        "{label}: wet band is {band:.2} m, want at least 0.6 m past the water"
    );
    assert!(
        max_moist_drop < 0.28,
        "{label}: moisture steps by {max_moist_drop:.3} in one cell"
    );
    let far = ray.iter().rev().find(|s| s.x > shore_x + 1.6);
    if let Some(far) = far {
        assert!(
            far.moist < 0.45,
            "{label}: moisture {:.3} at x {:.2} never leaves the wet band",
            far.moist,
            far.x
        );
    }
    println!(
        "{label}: floor water {:.4} bed {:.4}  shore {:.2} m  max drop {:.4}  wet band {:.2} m  moist drop {:.3}",
        floor.water, floor.bed, span, max_drop, band, max_moist_drop
    );
}

fn channel_span(frame: &ContractFrame, offset: usize) -> (f32, f32, f32, f32) {
    let n = frame.header.window_res as usize * frame.header.window_res as usize;
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut pos_min = f32::MAX;
    let mut pos_max = f32::MIN;
    for i in 0..n {
        let v = frame.ground_rut[i * 4 + offset];
        min = min.min(v);
        max = max.max(v);
        if v > 0.005 {
            pos_min = pos_min.min(v);
            pos_max = pos_max.max(v);
        }
    }
    let pos_min = if pos_min > 1.0 { 0.0 } else { pos_min };
    let pos_max = if pos_max < 0.0 { 0.0 } else { pos_max };
    (min, max, pos_min, pos_max)
}

#[test]
fn puddle_shore_is_a_bank_with_a_wet_band() {
    let drive = ThroughDrive::new();
    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &drive.world.soil,
        &drive.world.particles.items,
        [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
        drive.world.vehicle.pos.z,
        0,
        Some(&drive.fluid),
        Some(drive.dust()),
    );
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_puddle_shore(&frame, "still", -14.0, 0.0);

    let mut drive = drive;
    let dt = 1.0 / 30.0;
    while drive.world.vehicle.pos.z < -14.05 {
        drive.step(dt);
    }
    publish(
        &mut frame,
        &drive.world.soil,
        &drive.world.particles.items,
        [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
        drive.world.vehicle.pos.z,
        0,
        Some(&drive.fluid),
        Some(drive.dust()),
    );
    // East of the tracks. The centre after the drive is the rut, not the floor.
    assert_puddle_shore(&frame, "driven", -14.0, 1.6);

    let (g_min, g_max, g_pos_min, g_pos_max) = channel_span(&frame, 1);
    let (a_min, a_max, a_pos_min, a_pos_max) = channel_span(&frame, 3);
    let (b_min, b_max, _, _) = channel_span(&frame, 2);
    println!(
        "binding1 G rut min {g_min:.5} max {g_max:.5}  positive {g_pos_min:.5}..{g_pos_max:.5} span {:.5}",
        g_pos_max - g_pos_min
    );
    println!(
        "binding1 A berm min {a_min:.5} max {a_max:.5}  positive {a_pos_min:.5}..{a_pos_max:.5} span {:.5}",
        a_pos_max - a_pos_min
    );
    println!("binding1 B bed min {b_min:.5} max {b_max:.5}");
    // Metres, not a 0..1 texture. A berm crushed into a few millimetres
    // cannot be the spoil of this cut. The rut floor is the Bekker depth;
    // the force check owns that number.
    assert!(
        g_max > 0.03,
        "rut amplitude {g_max:.4} m is missing"
    );
    assert!(
        a_max > 0.04,
        "berm amplitude {a_max:.4} m is too weak"
    );
    assert!(
        a_pos_max - a_pos_min > 0.025,
        "berm positive samples {:.4}..{:.4} are crushed into a narrow interval",
        a_pos_min,
        a_pos_max
    );
    assert!(
        b_max > 0.08,
        "bed max {b_max:.4} does not carry the bank"
    );
    let _ = b_min;
}

fn cross_track(frame: &ContractFrame, z: f32) -> Vec<MapSample> {
    // One sample per contract cell. A 0.20 m stride from an arbitrary origin
    // hits the same cell twice and skips the soft edge on one side.
    let res = frame.header.window_res.max(1) as i32;
    let j = ((z - frame.header.origin_z) / frame.header.cell_m).floor() as i32;
    let mut out = Vec::new();
    if j < 0 || j >= res {
        return out;
    }
    for i in 0..res {
        let cx = frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m;
        if !(-2.4..=2.4).contains(&cx) {
            continue;
        }
        if let Some(mut s) = sample_at(frame, cx, z) {
            s.x = cx;
            out.push(s);
        }
    }
    out
}

fn print_track(frame: &ContractFrame, label: &str, z: f32, row: &[MapSample]) {
    println!(
        "-- cross-track {label} z={z:.2} origin {:.3},{:.3} cell {:.3} --",
        frame.header.origin_x, frame.header.origin_z, frame.header.cell_m
    );
    println!(
        "{:>7} {:>8} {:>8} {:>8} {:>8}",
        "x", "cx", "rut G", "berm A", "bed B"
    );
    for s in row {
        let i = ((s.x - frame.header.origin_x) / frame.header.cell_m).floor();
        let cx = frame.header.origin_x + (i + 0.5) * frame.header.cell_m;
        println!(
            "{:>7.2} {:>8.3} {:>8.4} {:>8.4} {:>8.4}",
            s.x, cx, s.rut, s.berm, s.bed
        );
    }
}

/// G and A follow the two treads. The centreline is empty, each groove is
/// about a tread wide, and the edge falls off instead of stepping to zero.
fn assert_tread_profile(frame: &ContractFrame, label: &str, z: f32) {
    let row = cross_track(frame, z);
    print_track(frame, label, z, &row);
    assert!(row.len() > 10, "{label}: cross-track is short");
    for s in row.iter().filter(|s| s.x.abs() <= 0.15) {
        assert!(
            s.rut < 0.008,
            "{label}: rut {0:.4} at x {1:.2} fills the gap between the wheels",
            s.rut,
            s.x
        );
        assert!(
            s.berm < 0.012,
            "{label}: berm {0:.4} at x {1:.2} is a band between the wheels",
            s.berm,
            s.x
        );
    }
    for side in [1.0_f32, -1.0] {
        let track = 0.74 * side;
        let peak = row
            .iter()
            .filter(|s| (s.x - track).abs() <= 0.35)
            .map(|s| s.rut)
            .fold(0.0_f32, f32::max);
        assert!(
            peak > 0.02,
            "{label}: tread at x {track:.2} rut peak {peak:.4} is missing"
        );
        let mut half = Vec::new();
        for s in &row {
            if (s.x - track).abs() <= 0.9 && s.rut > peak * 0.5 {
                half.push(s.x);
            }
        }
        let width = if half.len() <= 1 {
            0.20
        } else {
            half.iter().copied().fold(f32::MIN, f32::max)
                - half.iter().copied().fold(f32::MAX, f32::min)
        };
        assert!(
            !half.is_empty() && width < 0.65,
            "{label}: tread at {track:.2} half-max width {width:.2} m is a slab"
        );
        assert!(
            width >= 0.15,
            "{label}: tread at {track:.2} half-max width {width:.2} m is narrower than a tread"
        );
        // Outward from the track centre, the rut must pass through a partial
        // sample. A jump from the floor to nothing is the straight edge.
        let mut outward: Vec<&MapSample> = row
            .iter()
            .filter(|s| {
                let out = (s.x - track) * side;
                out >= -0.05 && out <= 1.1
            })
            .collect();
        if side < 0.0 {
            outward.reverse();
        }
        let mut saw_mid = false;
        let mut prev = peak;
        let mut max_frac = 0.0_f32;
        for s in outward {
            let frac = (prev - s.rut).max(0.0) / peak;
            max_frac = max_frac.max(frac);
            if s.rut > peak * 0.12 && s.rut < peak * 0.72 {
                saw_mid = true;
            }
            prev = s.rut;
        }
        assert!(
            saw_mid,
            "{label}: tread at {track:.2} has no soft edge, max step {max_frac:.2} of the peak"
        );
        assert!(
            max_frac < 0.85,
            "{label}: tread at {track:.2} drops {max_frac:.2} of the peak in one cell"
        );
        let berm_peak = row
            .iter()
            .filter(|s| (s.x - track).abs() <= 0.85)
            .map(|s| s.berm)
            .fold(0.0_f32, f32::max);
        assert!(
            berm_peak > 0.015,
            "{label}: berm beside tread {track:.2} is missing, peak {berm_peak:.4}"
        );
    }
}

#[test]
fn puddle_ruts_follow_the_treads() {
    let mut drive = ThroughDrive::new();
    let dt = 1.0 / 30.0;
    for _ in 0..25 {
        drive.step(dt);
    }
    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &drive.world.soil,
        &drive.world.particles.items,
        [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
        drive.world.vehicle.pos.z,
        0,
        Some(&drive.fluid),
        Some(drive.dust()),
    );
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(RENDER_CONTRACT_VERSION, 2);
    let z = drive.world.vehicle.pos.z;
    println!("frame 25 chassis z {z:.3}");
    // Under the body, where a belly-wide slab would sit, and on the track
    // the rear tires have already left.
    assert_tread_profile(&frame, "under", z);
    assert_tread_profile(&frame, "behind", z - 1.6);
}

struct WetCell {
    x: f32,
    z: f32,
    pressed: f32,
    original: f32,
    water: f32,
    rut: f32,
}

fn wet_cell(frame: &ContractFrame, i: i32, j: i32) -> WetCell {
    let res = frame.header.window_res.max(1) as i32;
    let idx = j as usize * res as usize + i as usize;
    let pressed = frame.ground_rut[idx * 4];
    let rut = frame.ground_rut[idx * 4 + 1];
    let bed = frame.ground_rut[idx * 4 + 2];
    WetCell {
        x: frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m,
        z: frame.header.origin_z + (j as f32 + 0.5) * frame.header.cell_m,
        pressed,
        original: bed + rut,
        water: frame.film[idx * 2],
        rut,
    }
}

/// After the sheet settles, water in a rut that opens into the puddle sits on
/// the pressed ground and does not stand above the puddle. The original height
/// is not the datum: adding the column to it would lift the rut by the cut.
#[test]
fn puddle_rut_water_stays_with_the_pond() {
    let mut drive = ThroughDrive::new();
    let dt = 1.0 / 30.0;
    // Clear of the floor, so the bow is not the sample. The ruts stay in the window.
    while drive.world.vehicle.pos.z < -6.0 {
        drive.step(dt);
    }
    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &drive.world.soil,
        &drive.world.particles.items,
        [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
        drive.world.vehicle.pos.z,
        0,
        Some(&drive.fluid),
        Some(drive.dust()),
    );
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(RENDER_CONTRACT_VERSION, 2);

    let res = frame.header.window_res.max(1) as i32;
    let mut floor_eta = Vec::new();
    let mut ruts = Vec::new();
    for j in 0..res {
        for i in 0..res {
            let cell = wet_cell(&frame, i, j);
            if !(-16.8..-11.2).contains(&cell.z) || !(-4.0..4.0).contains(&cell.x) {
                continue;
            }
            let on_floor = cell.rut < 0.006
                && cell.water > 0.008
                && (-0.02..0.04).contains(&cell.pressed)
                && (1.4..3.6).contains(&cell.x.abs());
            if on_floor {
                floor_eta.push(cell.pressed + cell.water);
            }
            if cell.rut > 0.02 && cell.water > 0.015 && cell.x.abs() < 1.6 {
                ruts.push((i, j, cell));
            }
        }
    }
    assert!(
        floor_eta.len() >= 12,
        "puddle floor samples {}",
        floor_eta.len()
    );
    floor_eta.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let puddle = floor_eta[floor_eta.len() / 2];

    let mut connected = Vec::new();
    for (i, j, cell) in &ruts {
        let mut opens = false;
        for dj in -3..=3 {
            for di in -3..=3 {
                let ii = *i + di;
                let jj = *j + dj;
                if ii < 0 || jj < 0 || ii >= res || jj >= res {
                    continue;
                }
                let n = wet_cell(&frame, ii, jj);
                if n.rut < 0.008 && n.water > 0.02 {
                    opens = true;
                }
            }
        }
        if opens {
            connected.push(cell);
        }
    }
    assert!(
        !connected.is_empty(),
        "no rut still connected to the puddle"
    );

    let mut worst = connected[0];
    let mut worst_eta = f32::MIN;
    println!(
        "-- settled rut water, datum = pressed ground (binding 1 R = bed + placed) --"
    );
    println!(
        "{:>7} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "x", "z", "pressed", "original", "water", "surface", "orig+w"
    );
    for cell in &connected {
        let surface = cell.pressed + cell.water;
        if surface > worst_eta {
            worst_eta = surface;
            worst = cell;
        }
    }
    // One transverse slice through the deepest connected rut.
    let slice_z = worst.z;
    let mut slice = Vec::new();
    let mut x = -2.4_f32;
    while x <= 2.41 {
        let i = ((x - frame.header.origin_x) / frame.header.cell_m).floor() as i32;
        let j = ((slice_z - frame.header.origin_z) / frame.header.cell_m).floor() as i32;
        if i >= 0 && j >= 0 && i < res && j < res {
            slice.push(wet_cell(&frame, i, j));
        }
        x += 0.2;
    }
    for cell in &slice {
        println!(
            "{:>7.2} {:>7.2} {:>8.4} {:>8.4} {:>8.4} {:>8.4} {:>8.4}",
            cell.x,
            cell.z,
            cell.pressed,
            cell.original,
            cell.water,
            cell.pressed + cell.water,
            cell.original + cell.water
        );
    }
    println!(
        "puddle surface {puddle:.4} m  rut water surface {worst_eta:.4} m  ground datum pressed {pressed:.4} m (original {original:.4} m) at x {x:.2} z {z:.2}",
        pressed = worst.pressed,
        original = worst.original,
        x = worst.x,
        z = worst.z
    );
    assert!(
        (worst.pressed - (worst.original - worst.rut)).abs() < 1.0e-3,
        "pressed ground {:.4} is not bed+placed under the rut",
        worst.pressed
    );
    assert!(
        worst.original > worst.pressed + 0.015,
        "rut datum did not drop below the original height"
    );
    assert!(
        worst_eta <= puddle + 0.006,
        "rut water surface {worst_eta:.4} m is above the puddle {puddle:.4} m. Datum is pressed ground {pressed:.4} m; original ground {original:.4} m plus the column would be {orig_eta:.4} m",
        pressed = worst.pressed,
        original = worst.original,
        orig_eta = worst.original + worst.water
    );
}

fn channel_minmax(values: &[f32], stride: usize, offset: usize) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let mut i = offset;
    while i < values.len() {
        let v = values[i];
        min = min.min(v);
        max = max.max(v);
        i += stride;
    }
    (min, max)
}

/// Dry flow and foam publish 0, never NaN. While the Niva is in the puddle,
/// a standing wave has foam and the peeled sheet has spray drops.
#[test]
fn puddle_foam_and_spray_are_finite_on_the_wave() {
    let mut drive = ThroughDrive::new();
    let dt = 1.0 / 30.0;
    let mut frame = ContractFrame::new();
    let mut saw_wave = false;
    let mut saw_spray = false;
    let mut foam_min = f32::MAX;
    let mut foam_max = f32::MIN;
    let mut crest_min = f32::MAX;
    let mut crest_max = f32::MIN;
    let mut flow_min = f32::MAX;
    let mut flow_max = f32::MIN;
    let mut spray_max = 0u32;
    let mut dry_bad = 0u32;
    while drive.world.vehicle.pos.z < -8.0 {
        drive.step(dt);
        publish(
            &mut frame,
            &drive.world.soil,
            &drive.world.particles.items,
            [drive.world.vehicle.pos.x, drive.world.vehicle.pos.z],
            drive.world.vehicle.pos.z,
            0,
            Some(&drive.fluid),
            Some(drive.dust()),
        );
        assert_eq!(nan_count(&frame), 0, "NaN in the contract");
        let n = frame.header.window_res as usize * frame.header.window_res as usize;
        let mut wave = false;
        for idx in 0..n {
            let water = frame.film[idx * 2];
            let slurry = frame.film[idx * 2 + 1];
            let film = water + slurry;
            let u = frame.flow[idx * 2];
            let v = frame.flow[idx * 2 + 1];
            let foam = frame.foam_crest[idx * 2];
            let crest = frame.foam_crest[idx * 2 + 1];
            // A dry cell is film under 0.01 mm. Velocity and foam stay 0.
            if film < 1.0e-5 && (u != 0.0 || v != 0.0 || foam != 0.0) {
                dry_bad += 1;
            }
            if !(-18.0..-10.0).contains(&drive.world.vehicle.pos.z) {
                continue;
            }
            foam_min = foam_min.min(foam);
            foam_max = foam_max.max(foam);
            crest_min = crest_min.min(crest);
            crest_max = crest_max.max(crest);
            let speed = u.hypot(v);
            flow_min = flow_min.min(speed);
            flow_max = flow_max.max(speed);
            if film >= 0.004 {
                let mut sum = 0.0f32;
                let mut count = 0.0f32;
                let res = frame.header.window_res as i32;
                let i = (idx % frame.header.window_res as usize) as i32;
                let j = (idx / frame.header.window_res as usize) as i32;
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let ni = i + di;
                    let nj = j + dj;
                    if ni < 0 || nj < 0 || ni >= res || nj >= res {
                        continue;
                    }
                    let nidx = nj as usize * res as usize + ni as usize;
                    sum += frame.film[nidx * 2] + frame.film[nidx * 2 + 1];
                    count += 1.0;
                }
                let mean = if count > 0.0 { sum / count } else { film };
                let excess = film - mean;
                // Past the toe of the crest ramp, or fast enough for Froude foam.
                if excess >= 0.010 || (speed >= 1.0 && film >= 0.02) {
                    wave = true;
                    assert!(
                        foam > 0.0,
                        "wave cell film {film:.4} excess {excess:.4} speed {speed:.3} has no foam"
                    );
                }
            }
        }
        let drops = frame.spray[..frame.header.spray_count as usize]
            .iter()
            .filter(|slot| slot.alive == 1 && slot.material == 0 && slot.pos_mass[3] > 0.0)
            .count() as u32;
        spray_max = spray_max.max(drops);
        if (-18.0..-10.0).contains(&drive.world.vehicle.pos.z) {
            if wave {
                saw_wave = true;
            }
            if drops > 0 {
                saw_spray = true;
            }
        }
    }
    let (flow_u_min, flow_u_max) = channel_minmax(&frame.flow, 2, 0);
    let (flow_v_min, flow_v_max) = channel_minmax(&frame.flow, 2, 1);
    println!(
        "puddle foam {foam_min:.4}..{foam_max:.4}  crest {crest_min:.4}..{crest_max:.4}  flow speed 0..{flow_max:.4}  last-frame flow u {flow_u_min:.4}..{flow_u_max:.4} v {flow_v_min:.4}..{flow_v_max:.4}  spray drops {spray_max}  dry bad {dry_bad}"
    );
    assert_eq!(dry_bad, 0, "dry cells published a velocity or foam");
    assert!(saw_wave, "no wave in the puddle");
    assert!(foam_max > 0.0, "foam max {foam_max}");
    assert!(saw_spray, "no spray drops while the wave is up");
    assert!(spray_max > 0, "spray count {spray_max}");
    let _ = flow_min;
}
