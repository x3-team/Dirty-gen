//! Niva in a creek. The creek is only the bench: depth, slope, and the
//! incoming current are the initial column. Heads, holes, the wake, scour,
//! and the load are read back from the shared sheet after the wheels and the
//! belly have been walls in that sheet.

use glam::Vec3;
use silt::fluid_core::{CoarseCell, FluidCore, StreamCycle, COARSE_CELL_M, WINDOW_RES};
use silt::particles::Particle;
use silt::render_contract::{
    publish, ContractFrame, FLOW_CHANNELS, FLOW_FLUID_SHEET, RENDER_CONTRACT_VERSION, VALID_FLOW,
};
use silt::soil::{SoilField, SurfaceKind};
use silt::vehicle::Vehicle;

const HALF_W: f32 = 1.5;
const Z_LO: f32 = -8.0;
const Z_HI: f32 = 8.0;
const SLOPE: f32 = 0.012;
const DT: f32 = 1.0 / 60.0;

fn column(x: f32, z: f32, center_depth: f32, crown: f32) -> (f32, f32) {
    if x.abs() >= HALF_W || !(Z_LO..Z_HI).contains(&z) {
        return (0.55, 0.0);
    }
    let nx = (x.abs() / HALF_W).clamp(0.0, 1.0);
    let depth = (center_depth - crown * nx * nx).max(0.04);
    let bed = -SLOPE * z - depth;
    (bed, depth)
}

fn seed(speed: f32, center_depth: f32, crown: f32) -> FluidCore {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    fluid.map_water_bed(|x, z| column(x, z, center_depth, crown));
    fluid.map_water_velocity(|x, z, _| {
        if x.abs() < HALF_W - 0.05 && z > Z_LO + 0.3 && z < Z_HI - 0.3 {
            (0.0, speed)
        } else {
            (0.0, 0.0)
        }
    });
    // Lips sit on the last wet rows. A lip short of the bank leaves a pocket
    // that fills because the bank will not take the water.
    fluid.set_stream_cycle(Some(StreamCycle {
        x0: -1.35,
        x1: 1.35,
        z0: -7.9,
        z1: 7.9,
        dir_x: 0.0,
        dir_z: 1.0,
    }));
    fluid
}

fn depth_near(fluid: &FluidCore, x: f32, z: f32) -> f32 {
    let mut acc = 0.0f32;
    let mut n = 0.0f32;
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let h = fluid.sample(x + a.cos() * 0.48, z + a.sin() * 0.48).water_m;
        if h > 0.02 {
            acc += h;
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc / n
    } else {
        fluid.sample(x, z).water_m
    }
}

fn cell_at(fluid: &FluidCore, x: f32, z: f32) -> CoarseCell {
    let (ox, oz, _) = fluid.window_bounds();
    // Nudge off an exact edge so two probes one cell apart cannot share a row.
    let i = ((x - ox) / COARSE_CELL_M + 1.0e-4)
        .floor()
        .clamp(0.0, (WINDOW_RES - 1) as f32) as usize;
    let j = ((z - oz) / COARSE_CELL_M + 1.0e-4)
        .floor()
        .clamp(0.0, (WINDOW_RES - 1) as f32) as usize;
    fluid.coarse_cell(i, j).expect("cell")
}

fn curl_at(fluid: &FluidCore, x: f32, z: f32) -> f32 {
    let c = COARSE_CELL_M;
    let vp = cell_at(fluid, x + c, z).v;
    let vm = cell_at(fluid, x - c, z).v;
    let up = cell_at(fluid, x, z + c).u;
    let um = cell_at(fluid, x, z - c).u;
    (vp - vm) / (2.0 * c) - (up - um) / (2.0 * c)
}

struct Wake {
    head: f32,
    hole: f32,
    stream: f32,
    wake: f32,
    vort_l: f32,
    vort_r: f32,
    up_h: f32,
    down_h: f32,
    amb_h: f32,
}

fn wake_at(fluid: &FluidCore, x: f32, z: f32) -> Wake {
    // Free stream on the centerline, upstream of a vehicle parked at the origin.
    let amb = cell_at(fluid, 0.0, -4.0);
    let mut up = amb;
    for k in 1..=6 {
        let c = cell_at(fluid, x, z - k as f32 * COARSE_CELL_M);
        if c.water_m > up.water_m {
            up = c;
        }
    }
    let mut down = amb;
    let mut down_z = z + 4.0 * COARSE_CELL_M;
    let mut have_down = false;
    for k in 2..=8 {
        let pz = z + k as f32 * COARSE_CELL_M;
        let c = cell_at(fluid, x, pz);
        // The tire itself is dry. The hole is the shallow water behind it.
        if c.water_m < 0.015 {
            continue;
        }
        if !have_down || c.water_m < down.water_m {
            down = c;
            down_z = pz;
            have_down = true;
        }
    }
    let mut wake_s = 0.0f32;
    let mut wake_n = 0.0f32;
    let mut vl = 0.0f32;
    let mut nl = 0.0f32;
    let mut vr = 0.0f32;
    let mut nr = 0.0f32;
    for i in 1..=3 {
        for side in [-1.0f32, 1.0f32] {
            let px = x + side * i as f32 * COARSE_CELL_M;
            let s = cell_at(fluid, px, down_z);
            let sp = if s.water_m < 0.015 {
                0.0
            } else {
                s.u.hypot(s.v)
            };
            wake_s += sp;
            wake_n += 1.0;
            if s.water_m < 0.015 {
                continue;
            }
            let w = curl_at(fluid, px, down_z);
            if !w.is_finite() {
                continue;
            }
            if side < 0.0 {
                vl += w;
                nl += 1.0;
            } else {
                vr += w;
                nr += 1.0;
            }
        }
    }
    // Separated cells directly behind the tire count as still water.
    for k in 2..=5 {
        let s = cell_at(fluid, x, z + k as f32 * COARSE_CELL_M);
        if s.water_m < 0.015 {
            wake_s += 0.0;
            wake_n += 1.0;
        }
    }
    Wake {
        head: up.water_m - amb.water_m,
        hole: amb.water_m - down.water_m,
        stream: amb.u.hypot(amb.v),
        wake: if wake_n > 0.0 { wake_s / wake_n } else { 0.0 },
        vort_l: if nl > 0.0 { vl / nl } else { 0.0 },
        vort_r: if nr > 0.0 { vr / nr } else { 0.0 },
        up_h: up.water_m,
        down_h: down.water_m,
        amb_h: amb.water_m,
    }
}

fn strongest_wake(fluid: &FluidCore, vehicle: &Vehicle) -> Wake {
    let ori = vehicle.orientation();
    let mut best: Option<Wake> = None;
    for i in 0..4 {
        let w = vehicle.pos + ori * vehicle.wheel_center_local(i);
        let wake = wake_at(fluid, w.x, w.z);
        let score = wake.head + wake.hole;
        let replace = best
            .as_ref()
            .map(|b| score > b.head + b.hole)
            .unwrap_or(true);
        if replace {
            best = Some(wake);
        }
    }
    best.expect("wheel")
}

fn sync_soil(fluid: &FluidCore, soil: &mut SoilField, initial: bool) {
    let half = soil.size * 0.5;
    for j in 0..soil.res {
        for i in 0..soil.res {
            let x = (i as f32 + 0.5) * soil.cell - half;
            let z = (j as f32 + 0.5) * soil.cell - half;
            let s = fluid.sample(x, z);
            let idx = j * soil.res + i;
            if initial {
                soil.initial_height[idx] = s.bed_m;
            }
            soil.bed[idx] = s.bed_m;
            soil.water[idx] = s.water_m;
            soil.slurry[idx] = s.slurry_m;
            soil.placed[idx] = 0.0;
            soil.height[idx] = s.bed_m + s.slurry_m;
            if s.water_m > 0.02 {
                // Firm ford bottom. The sheet still entrains the bed; silt here
                // bogs the tire and the drive never leaves the reach.
                soil.wetness[idx] = 0.35;
                soil.kind[idx] = SurfaceKind::Gravel as u8;
                soil.loose[idx] = 0.004;
            }
        }
    }
}

fn plant(vehicle: &mut Vehicle, soil: &SoilField, x: f32, z: f32, yaw: f32, bed: f32) {
    vehicle.pos = Vec3::new(x, bed + 0.58, z);
    vehicle.yaw = yaw;
    vehicle.pitch = 0.0;
    vehicle.roll = 0.0;
    vehicle.vel = Vec3::ZERO;
    vehicle.yaw_rate = 0.0;
    vehicle.pitch_rate = 0.0;
    vehicle.roll_rate = 0.0;
    vehicle.water_lift_n = 0.0;
    vehicle.water_drag_n = 0.0;
    vehicle.water_push_n = Vec3::ZERO;
    vehicle.engine_stalled = false;
    for w in &mut vehicle.wheels {
        w.compression = 0.08;
        w.omega = 0.0;
        w.steer = 0.0;
    }
    for _ in 0..24 {
        let c = vehicle.compute_contacts(soil, 0.0, 1.0, DT);
        for ct in &c {
            vehicle.wheels[ct.index].compression = ct.compression;
        }
        if c.iter().filter(|ct| ct.active).count() >= 3 {
            break;
        }
        vehicle.pos.y -= 0.015;
    }
    vehicle.vel = Vec3::ZERO;
    vehicle.yaw_rate = 0.0;
}

struct Hold {
    wake: Wake,
    force: f32,
    force_x: f32,
    force_z: f32,
    buoy: f32,
    hull: f32,
    drift_z: f32,
    drift_x: f32,
    eroded: f32,
    hole: f32,
    hole_mid: f32,
    blocks_w: u32,
    blocks_b: u32,
    gap: f32,
    nans: u32,
    end_x: f32,
    end_speed: f32,
}

fn hold(
    fluid: &mut FluidCore,
    soil: &mut SoilField,
    vehicle: &mut Vehicle,
    seconds: f32,
    brake: f32,
    throttle: f32,
    integrate: bool,
) -> Hold {
    let water0 = fluid.ledger().water_m3;
    let z0 = vehicle.pos.z;
    let x0 = vehicle.pos.x;
    let steps = (seconds / DT).round().max(1.0) as u32;
    let mid = steps / 3;
    let mut hole_mid = 0.0f32;
    let mut load = fluid.hydro_load();
    for s in 0..steps {
        let solids = vehicle.hydro_solids(|x, z| depth_near(fluid, x, z));
        fluid.set_hydro_solids(&solids);
        fluid.step(DT);
        load = fluid.hydro_load();
        if s + 1 == mid {
            hole_mid = load.hole_m;
        }
        if !integrate {
            continue;
        }
        vehicle.water_push_n = Vec3::new(load.force_x, 0.0, load.force_z);
        vehicle.water_lift_n = load.hull_buoyancy_n;
        vehicle.water_drag_n = 0.0;
        sync_soil(fluid, soil, false);
        vehicle.update_steer(0.0, DT);
        let contacts = vehicle.compute_contacts(soil, throttle, brake, DT);
        vehicle.integrate(&contacts, soil, DT);
        if brake > 0.5 {
            for w in &mut vehicle.wheels {
                w.omega = 0.0;
            }
        }
    }
    let wake = strongest_wake(fluid, vehicle);
    Hold {
        wake,
        force: load.force_x.hypot(load.force_z),
        force_x: load.force_x,
        force_z: load.force_z,
        buoy: load.buoyancy_n,
        hull: load.hull_buoyancy_n,
        drift_z: vehicle.pos.z - z0,
        drift_x: vehicle.pos.x - x0,
        eroded: load.eroded_m3,
        hole: load.hole_m,
        hole_mid,
        blocks_w: load.wheel_blocks,
        blocks_b: load.belly_blocks,
        gap: fluid.ledger().water_m3 - water0,
        nans: fluid.non_finite_count(),
        end_x: vehicle.pos.x,
        end_speed: vehicle.speed(),
    }
}

fn contract_head(frame: &ContractFrame, x: f32, z: f32) -> (f32, f32, f32) {
    let i = ((x - frame.header.origin_x) / frame.header.cell_m).floor() as isize;
    let j = ((z - frame.header.origin_z) / frame.header.cell_m).floor() as isize;
    if i < 0 || j < 0 || i >= WINDOW_RES as isize || j >= WINDOW_RES as isize {
        return (0.0, 0.0, 0.0);
    }
    let idx = j as usize * WINDOW_RES + i as usize;
    let h = frame.film[idx * 2];
    let u = frame.flow[idx * FLOW_CHANNELS];
    let v = frame.flow[idx * FLOW_CHANNELS + 1];
    (h, u, v)
}

#[test]
fn creek_crossing_prints_solver_fields() {
    // Open water, same slope and current, no body. The sheet must not grow a head by itself.
    let mut open = seed(1.05, 0.24, 0.0);
    let open_water = open.ledger().water_m3;
    for _ in 0..(0.8 / DT) as u32 {
        open.set_hydro_solids(&[]);
        open.step(DT);
    }
    let open_wake = wake_at(&open, 0.74, 0.0);
    let open_gap = open.ledger().water_m3 - open_water;

    // Head against flow speed. Same walls, two incoming currents.
    // Nose downstream: each tire is a narrow wall with water on both sides.
    let (slow, _, _, _) = fixed_park(0.55, 0.24, 0.9);
    let (fast, bow_fluid, bow_vehicle, mut bow_soil) = fixed_park(1.35, 0.24, 0.9);
    sync_soil(&bow_fluid, &mut bow_soil, false);
    let mut frame = ContractFrame::new();
    let particles: &[Particle] = &[];
    publish(
        &mut frame,
        &bow_soil,
        particles,
        [bow_vehicle.pos.x, bow_vehicle.pos.z],
        0.9,
        0,
        Some(&bow_fluid),
        None,
    );
    let ori = bow_vehicle.orientation();
    let (amb_h, amb_u, amb_v) = contract_head(&frame, 0.0, -4.0);
    let mut up_h = amb_h;
    let mut up_s = amb_u.hypot(amb_v);
    let mut down_h = amb_h;
    for i in 0..4 {
        let wheel = bow_vehicle.pos + ori * bow_vehicle.wheel_center_local(i);
        for k in 1..=6 {
            let (h, u, v) = contract_head(&frame, wheel.x, wheel.z - k as f32 * COARSE_CELL_M);
            if h > up_h {
                up_h = h;
                up_s = u.hypot(v);
            }
        }
        for k in 2..=8 {
            let (h, _, _) = contract_head(&frame, wheel.x, wheel.z + k as f32 * COARSE_CELL_M);
            if h < 0.015 {
                continue;
            }
            if h < down_h {
                down_h = h;
            }
        }
    }
    let mut contract_nans = 0u32;
    for v in frame
        .film
        .iter()
        .chain(&frame.flow)
        .chain(&frame.ground_rut)
        .chain(&frame.foam_crest)
    {
        if !v.is_finite() {
            contract_nans += 1;
        }
    }
    let mut rut_max = 0.0f32;
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let idx = j * WINDOW_RES + i;
            rut_max = rut_max.max(frame.ground_rut[idx * 4 + 1]);
        }
    }

    // Depth sweep on the same walls. Shallow and deep are both inside 0.15–0.30 m.
    let shallow = drift_park(1.0, 0.16, 0.0, 0.0, 1.0);
    let deep = drift_park(1.0, 0.28, 0.0, 0.0, 1.0);

    // The bench: cross-section 0.15 m at the bank to 0.30 m in the thalweg,
    // current along the slope. Across, then along, then a slow ford.
    let mut fluid = seed(1.0, 0.30, 0.15);
    let mut soil = SoilField::flat(0.35);
    sync_soil(&fluid, &mut soil, true);
    let bed0 = fluid.sample(0.0, 0.0).bed_m;
    let mut vehicle = Vehicle::new();
    plant(
        &mut vehicle,
        &soil,
        0.0,
        0.0,
        std::f32::consts::FRAC_PI_2,
        bed0,
    );
    let across = hold(&mut fluid, &mut soil, &mut vehicle, 1.1, 1.0, 0.0, true);

    vehicle.vel = Vec3::ZERO;
    vehicle.yaw = 0.0;
    vehicle.yaw_rate = 0.0;
    let along = hold(&mut fluid, &mut soil, &mut vehicle, 0.8, 1.0, 0.0, true);

    vehicle.vel = Vec3::ZERO;
    vehicle.yaw = std::f32::consts::FRAC_PI_2;
    vehicle.yaw_rate = 0.0;
    let x_before = vehicle.pos.x;
    let drive = hold(&mut fluid, &mut soil, &mut vehicle, 2.4, 0.0, 0.55, true);

    let gap = across.gap + along.gap + drive.gap;
    let nans = across.nans
        + along.nans
        + drive.nans
        + shallow.nans
        + deep.nans
        + slow.nans
        + fast.nans
        + open.non_finite_count()
        + contract_nans;

    eprintln!(
        "creek head slow {:.4} m at {:.3} m/s | fast {:.4} m at {:.3} m/s\n\
         vortex L {:.3} R {:.3} 1/s  wake {:.3} stream {:.3} m/s  hole {:.4} m\n\
         eroded {:.5} m3  hole_mid {:.4} m  hole_end {:.4} m\n\
         force across {:.1} N ({:.1}, {:.1})  along {:.1} N ({:.1}, {:.1})\n\
         buoy deep {:.1} N shallow {:.1} N  drift deep {:.4} m shallow {:.4} m\n\
         water gap {:.6} m3  nan {}\n\
         open head {:.4} hole {:.4} gap {:.6}\n\
         blocks wheels {} belly {}\n\
         contract up {:.4} down {:.4} amb {:.4} bow {:.3} stream {:.3} rut {:.4} ver {}\n\
         drive dx {:.3} m speed {:.3} m/s",
        slow.wake.head,
        slow.wake.stream,
        fast.wake.head,
        fast.wake.stream,
        fast.wake.vort_l,
        fast.wake.vort_r,
        fast.wake.wake,
        fast.wake.stream,
        fast.wake.hole,
        across.eroded,
        across.hole_mid,
        across.hole,
        across.force,
        across.force_x,
        across.force_z,
        along.force,
        along.force_x,
        along.force_z,
        deep.buoy,
        shallow.buoy,
        deep.drift_z,
        shallow.drift_z,
        gap,
        nans,
        open_wake.head,
        open_wake.hole,
        open_gap,
        across.blocks_w,
        across.blocks_b,
        up_h,
        down_h,
        amb_h,
        up_s,
        amb_u.hypot(amb_v),
        rut_max,
        frame.header.version,
        drive.end_x - x_before,
        drive.end_speed,
    );

    assert!(
        fast.wake.stream > slow.wake.stream + 0.25,
        "measured stream slow {:.3} fast {:.3}",
        slow.wake.stream,
        fast.wake.stream
    );
    assert!(
        fast.wake.head > slow.wake.head + 0.008,
        "head should grow with speed {:.4} -> {:.4}",
        slow.wake.head,
        fast.wake.head
    );
    assert!(
        fast.wake.head > open_wake.head + 0.01,
        "head {:.4} is not above the empty creek {:.4}",
        fast.wake.head,
        open_wake.head
    );
    assert!(
        across.wake.head > 0.008,
        "across head {:.4}",
        across.wake.head
    );
    assert!(
        fast.wake.hole > 0.008,
        "downstream hole {:.4}",
        fast.wake.hole
    );
    assert!(
        fast.wake.wake < fast.wake.stream * 0.85,
        "wake {:.3} is not quieter than the stream {:.3}",
        fast.wake.wake,
        fast.wake.stream
    );
    assert!(
        fast.wake.vort_l * fast.wake.vort_r < 0.0,
        "wake lobes should oppose L {:.3} R {:.3}",
        fast.wake.vort_l,
        fast.wake.vort_r
    );
    assert!(
        fast.wake.vort_l.abs().max(fast.wake.vort_r.abs()) > 0.35,
        "vorticity {:.3} {:.3}",
        fast.wake.vort_l,
        fast.wake.vort_r
    );
    assert!(across.eroded > 5.0e-4, "eroded soil {:.6}", across.eroded);
    assert!(
        across.hole > across.hole_mid + 0.001,
        "scour hole should grow {:.4} -> {:.4}",
        across.hole_mid,
        across.hole
    );
    assert!(across.force > 40.0, "flow force {:.1}", across.force);
    assert!(
        deep.buoy > shallow.buoy + 80.0,
        "buoyancy deep {:.1} shallow {:.1}",
        deep.buoy,
        shallow.buoy
    );
    assert!(
        deep.drift_z.abs() > shallow.drift_z.abs(),
        "drift deep {:.4} shallow {:.4}",
        deep.drift_z,
        shallow.drift_z
    );
    assert!(across.blocks_w > 0, "wheels did not enter the sheet");
    assert!(
        drive.end_x - x_before > 0.45,
        "ford dx {:.3}",
        drive.end_x - x_before
    );
    assert!(gap.abs() < 1.0e-3, "water gap {:.6}", gap);
    assert!(open_gap.abs() < 1.0e-3, "open gap {:.6}", open_gap);
    assert_eq!(nans, 0, "non-finite samples {nans}");
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(frame.header.flow_source, FLOW_FLUID_SHEET);
    assert!(frame.header.valid & VALID_FLOW != 0, "flow map not valid");
    assert!(
        up_h > down_h + 0.008,
        "contract head {:.4} hole {:.4}",
        up_h,
        down_h
    );
    assert!(
        amb_u.hypot(amb_v) > up_s + 0.05,
        "contract stream {:.3} bow {:.3}",
        amb_u.hypot(amb_v),
        up_s
    );
    assert!(rut_max > 0.001, "contract rut {:.4}", rut_max);
    let _ = amb_h;
}

fn fixed_park(speed: f32, depth: f32, seconds: f32) -> (Hold, FluidCore, Vehicle, SoilField) {
    let mut fluid = seed(speed, depth, 0.0);
    let mut soil = SoilField::flat(0.35);
    sync_soil(&fluid, &mut soil, true);
    let bed = fluid.sample(0.0, 0.0).bed_m;
    let mut vehicle = Vehicle::new();
    // Nose downstream, so each tire is a narrow obstacle with open water on both flanks.
    plant(&mut vehicle, &soil, 0.0, 0.0, 0.0, bed);
    let held = hold(
        &mut fluid,
        &mut soil,
        &mut vehicle,
        seconds,
        1.0,
        0.0,
        false,
    );
    (held, fluid, vehicle, soil)
}

fn drift_park(speed: f32, depth: f32, crown: f32, x: f32, seconds: f32) -> Hold {
    let mut fluid = seed(speed, depth, crown);
    let mut soil = SoilField::flat(0.35);
    sync_soil(&fluid, &mut soil, true);
    let bed = fluid.sample(x, 0.0).bed_m;
    let mut vehicle = Vehicle::new();
    plant(
        &mut vehicle,
        &soil,
        x,
        0.0,
        std::f32::consts::FRAC_PI_2,
        bed,
    );
    hold(&mut fluid, &mut soil, &mut vehicle, seconds, 1.0, 0.0, true)
}
