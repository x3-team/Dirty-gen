//! Free bodies against the sheet, sand, mud and snow. Every number checked
//! here is read back from the solver: the draft a buoy settles at, the ring a
//! dropped ball sends out, the bow and the trough at a wading leg, the depth
//! and rim of a print, and the ledgers that must not move.

use glam::{Vec2, Vec3};
use silt::bodies::{bed_base, leg_id, Bodies, Body, Ground, PrintGround, Walker};
use silt::fluid_core::{FluidCore, Grain, OpenWater};
use silt::imprint::{imprint_mat, Imprint, ImprintMat, IMPRINT_CELL_M};
use silt::snow_track::SnowTrack;
use silt::soil::SurfaceKind;

const DT: f32 = 1.0 / 60.0;

/// Flat sand floor at y = 0 inside a 12 m square with banks, water to `level`.
fn basin_bed(x: f32, z: f32) -> f32 {
    let d = (x.abs() - 6.0).max(z.abs() - 6.0).max(0.0);
    d * 0.6
}

fn pool(level: f32) -> FluidCore {
    let mut f = FluidCore::new([0.0, 0.0]);
    f.read_substrate(|x, z| (basin_bed(x, z), 0.9, Grain::Sand));
    f.map_grain(|_, _| Grain::Sand);
    f.map_water_bed(|x, z| {
        let b = basin_bed(x, z);
        (b, (level - b).max(0.0))
    });
    f.set_open_water(Some(OpenWater {
        mobility: 1.0,
        manning_n: 0.03,
        eddy_m2_s: 0.03,
    }));
    f
}

/// Water in the sheet plus water thrown into the air as drops.
fn water_m3(f: &FluidCore) -> f64 {
    let l = f.ledger();
    l.water_m3 as f64 + f.spray_m3() as f64
}

fn eta_at(f: &FluidCore, x: f32, z: f32) -> f32 {
    f.surface_at(x, z).0
}

fn sand_mat(_: f32, _: f32) -> ImprintMat {
    imprint_mat(SurfaceKind::Sand, 0.4, 0.0)
}

/// Step bodies and sheet together for `seconds`.
fn run(
    bodies: &mut Bodies,
    fluid: &mut FluidCore,
    imprint: &mut Imprint,
    seconds: f32,
    mut each: impl FnMut(&Bodies, &FluidCore, f32),
) {
    let steps = (seconds / DT).round() as usize;
    for k in 0..steps {
        {
            let mut ground = PrintGround {
                imprint: &mut *imprint,
                base: bed_base(fluid),
            };
            bodies.step(DT, &mut ground, Some(fluid));
        }
        bodies.hand_off(fluid);
        fluid.step(DT);
        each(bodies, fluid, k as f32 * DT);
    }
}

#[test]
fn buoy_floats_at_its_archimedes_draft() {
    let mut fluid = pool(0.6);
    for _ in 0..30 {
        fluid.step(DT);
    }
    let v0 = water_m3(&fluid);
    let mut bodies = Bodies::new();
    // A 36 cm buoy of 6 kg: half its volume under at rest would be 12.2 l.
    bodies.items.push(Body::ball(Vec3::new(0.3, 0.6 + 0.10, -0.2), 0.18, 6.0, true));
    let mut imprint = Imprint::new();
    let mut ys = Vec::new();
    run(&mut bodies, &mut fluid, &mut imprint, 8.0, |b, _, t| {
        if t > 6.0 {
            ys.push(b.items[0].pos.y);
        }
    });
    let b = bodies.items[0];
    let want = b.mass / 1000.0;
    let load = fluid.wet_loads()[0];
    let spread = ys.iter().cloned().fold(f32::MIN, f32::max) - ys.iter().cloned().fold(f32::MAX, f32::min);
    println!(
        "buoy y {:.3} submerged {:.4} m3 (want {:.4}) sheet displaced {:.4} surface {:.3} bob {:.4} water {:.4} -> {:.4}",
        b.pos.y, b.submerged_m3, want, load.displaced_m3, load.surface_y, spread, v0, water_m3(&fluid)
    );
    assert!((b.submerged_m3 - want).abs() < 0.12 * want, "draft off Archimedes");
    assert!((load.displaced_m3 - b.submerged_m3).abs() < 0.25 * want, "sheet and body disagree on displacement");
    assert!(spread < 0.01, "buoy still bobbing {spread}");
    assert!(((water_m3(&fluid) - v0) / v0).abs() < 1.0e-4, "water volume moved");
}

#[test]
fn dropped_ball_sends_a_ring_and_keeps_the_volume() {
    let level = 0.5;
    let mut fluid = pool(level);
    for _ in 0..30 {
        fluid.step(DT);
    }
    let v0 = water_m3(&fluid);
    let still = eta_at(&fluid, 1.2, 0.0);
    let mut bodies = Bodies::new();
    // A heavy ball dropped from 40 cm: it plunges to the floor.
    bodies.items.push(Body::ball(Vec3::new(0.0, level + 0.6, 0.0), 0.22, 70.0, false));
    let mut imprint = Imprint::new();
    let mut crest = f32::MIN;
    let mut trough = f32::MAX;
    run(&mut bodies, &mut fluid, &mut imprint, 2.5, |_, f, _| {
        let e = eta_at(f, 1.2, 0.0);
        crest = crest.max(e - still);
        trough = trough.min(e - still);
    });
    let b = bodies.items[0];
    println!(
        "drop: ring crest {:+.4} trough {:+.4} at 1.2 m, ball bottom {:.3} print {:.3}, water {:.5} -> {:.5}",
        crest,
        trough,
        b.bottom(),
        b.print_m,
        v0,
        water_m3(&fluid)
    );
    assert!(crest > 0.004, "no ring reached 1.2 m");
    assert!(trough < -0.001, "ring has no trough");
    assert!(b.bottom() < 0.02, "ball did not reach the floor");
    assert!(((water_m3(&fluid) - v0) / v0).abs() < 1.0e-4, "water volume moved");
}

#[test]
fn rock_sinks_and_sits_in_the_sand() {
    let mut fluid = pool(0.5);
    let mut bodies = Bodies::new();
    let half = Vec3::new(0.2, 0.12, 0.25);
    let mass = 2600.0 * 8.0 * half.x * half.y * half.z;
    bodies.items.push(Body::crate_box(Vec3::new(-1.0, 0.8, 0.5), half, mass, 0.4));
    let mut imprint = Imprint::new();
    run(&mut bodies, &mut fluid, &mut imprint, 4.0, |_, _, _| {});
    let b = bodies.items[0];
    let net = imprint.net_m3() + imprint.compacted_m3;
    println!(
        "rock bottom {:.4} print {:.4} submerged {:.4} vel {:.3} net {:+.2e} compacted {:.5}",
        b.bottom(),
        b.print_m,
        b.submerged_m3,
        b.vel.length(),
        net,
        imprint.compacted_m3
    );
    assert!(b.bottom() < 0.0 && b.bottom() > -0.08, "rock should sit a little into the sand");
    assert!(b.vel.length() < 0.02, "rock still moving");
    assert!((b.submerged_m3 - b.volume()).abs() < 0.02 * b.volume(), "rock not fully under");
    assert!(net.abs() < 1.0e-6, "imprint ledger {net}");
}

#[test]
fn walker_wades_and_pushes_a_bow_ahead_of_each_leg() {
    let level = 0.4;
    let mut fluid = pool(level);
    for _ in 0..30 {
        fluid.step(DT);
    }
    let v0 = water_m3(&fluid);
    let mut imprint = Imprint::new();
    let mut bodies = Bodies::new();
    {
        let ground = PrintGround {
            imprint: &mut imprint,
            base: bed_base(&fluid),
        };
        let mut w = Walker::new(Vec2::new(-4.0, 0.0), std::f32::consts::FRAC_PI_2, 1.4, &ground);
        w.path = vec![Vec2::new(4.5, 0.0)];
        bodies.walkers.push(w);
    }
    let mut bow = 0.0f32;
    let mut wake = 0.0f32;
    let mut n = 0.0f32;
    let mut speeds = Vec::new();
    run(&mut bodies, &mut fluid, &mut imprint, 5.0, |b, f, t| {
        let w = &b.walkers[0];
        if t > 1.5 {
            speeds.push(w.vel.length());
            // Ahead of and behind the pelvis along the walk, a leg's
            // length out so the columns themselves are not sampled.
            let p = Vec2::new(w.pos.x, w.pos.z);
            let fwd = w.forward();
            let ahead = p + fwd * 0.35;
            let behind = p - fwd * 0.55;
            bow += eta_at(f, ahead.x, ahead.y) - level;
            wake += eta_at(f, behind.x, behind.y) - level;
            n += 1.0;
        }
    });
    let w = &bodies.walkers[0];
    let pace = speeds.iter().sum::<f32>() / speeds.len().max(1) as f32;
    let loads: Vec<_> = (0..2)
        .filter_map(|k| fluid.wet_loads().iter().find(|l| l.id == leg_id(0, k)).copied())
        .collect();
    println!(
        "wade: pace {:.2} m/s (asked 1.4) steps {} bow {:+.4} wake {:+.4} legs displaced {:.4} {:.4} water {:.5} -> {:.5} prints max {:.3}",
        pace,
        w.steps,
        bow / n,
        wake / n,
        loads.first().map(|l| l.displaced_m3).unwrap_or(0.0),
        loads.get(1).map(|l| l.displaced_m3).unwrap_or(0.0),
        v0,
        water_m3(&fluid),
        w.feet[0].print_m.max(w.feet[1].print_m)
    );
    assert!(pace < 1.3 && pace > 0.6, "water should slow the walk, not stop it");
    assert!(w.steps >= 6);
    assert!(bow / n > wake / n + 0.002, "no bow over trough at the legs");
    assert!(((water_m3(&fluid) - v0) / v0).abs() < 1.0e-4, "water volume moved");
}

#[test]
fn walker_leaves_boot_prints_in_damp_sand() {
    let mut imprint = Imprint::new();
    let mut bodies = Bodies::new();
    {
        let ground = PrintGround {
            imprint: &mut imprint,
            base: |x: f32, z: f32| (0.0 * x * z, sand_mat(x, z)),
        };
        let mut w = Walker::new(Vec2::new(0.0, 0.0), 0.0, 1.3, &ground);
        w.path = vec![Vec2::new(0.0, 6.0)];
        bodies.walkers.push(w);
    }
    for _ in 0..(5.0 / DT) as usize {
        let mut ground = PrintGround {
            imprint: &mut imprint,
            base: |x: f32, z: f32| (0.0 * x * z, sand_mat(x, z)),
        };
        bodies.step(DT, &mut ground, None);
    }
    let w = &bodies.walkers[0];
    // Deepest and highest offset along the walk.
    let mut deep = 0.0f32;
    let mut rim = 0.0f32;
    let mut prints = 0;
    let mut in_print = false;
    let mut z = 0.5f32;
    while z < 5.5 {
        let mut row_min = 0.0f32;
        for side in [-1.0f32, 1.0] {
            let x = side * 0.095;
            row_min = row_min.min(imprint.offset_at(x, z));
            rim = rim.max(imprint.offset_at(x + side * 0.08, z));
        }
        deep = deep.min(row_min);
        let now = row_min < -0.005;
        if now && !in_print {
            prints += 1;
        }
        in_print = now;
        z += IMPRINT_CELL_M;
    }
    let net = imprint.net_m3() + imprint.compacted_m3;
    println!(
        "sand: steps {} prints along the walk {} deepest {:.4} rim {:.4} net {:+.2e} tiles {}",
        w.steps,
        prints,
        -deep,
        rim,
        net,
        imprint.tile_count()
    );
    assert!(w.steps >= 6);
    assert!(-deep > 0.008 && -deep < 0.03, "damp sand print depth");
    assert!(rim > 0.0015, "no rim beside the print");
    assert!(prints >= 5, "prints are not separate");
    assert!(net.abs() < 1.0e-6, "imprint ledger {net}");
}

#[test]
fn ball_rolls_a_groove_into_soft_mud_and_stops() {
    let mut imprint = Imprint::new();
    let mud = |x: f32, z: f32| (0.0 * x * z, imprint_mat(SurfaceKind::WetMud, 0.85, 0.0));
    let mut bodies = Bodies::new();
    let mut b = Body::ball(Vec3::new(0.0, 0.2, 0.0), 0.2, 25.0, false);
    b.vel = Vec3::new(0.0, 0.0, 3.0);
    b.omega = Vec3::new(3.0 / 0.2, 0.0, 0.0);
    bodies.items.push(b);
    for _ in 0..(6.0 / DT) as usize {
        let mut ground = PrintGround {
            imprint: &mut imprint,
            base: mud,
        };
        bodies.step(DT, &mut ground, None);
    }
    let b = bodies.items[0];
    let mid = b.pos.z * 0.5;
    let groove = -imprint.offset_at(0.0, mid);
    let berm = imprint.offset_at(0.2, mid).max(imprint.offset_at(-0.2, mid)).max(imprint.offset_at(0.17, mid));
    let net = imprint.net_m3() + imprint.compacted_m3;
    println!(
        "mud ball rolled {:.2} m speed {:.3} groove {:.4} berm {:.4} net {:+.2e}",
        b.pos.z,
        b.vel.length(),
        groove,
        berm,
        net
    );
    assert!(b.pos.z > 0.5 && b.pos.z < 15.0);
    assert!(b.vel.length() < 0.05, "ball should stop in mud");
    assert!(groove > 0.01, "no groove");
    assert!(berm > 0.002, "no berm");
    assert!(net.abs() < 1.0e-6);
}

#[test]
fn boot_prints_in_fresh_snow_keep_the_mass() {
    let mut track = SnowTrack::new(0.25, 0.0);
    track.rho_blanket = 160.0;
    let mut bodies = Bodies::new();
    let mut w = Walker::new(Vec2::new(0.0, 0.0), 0.0, 1.2, &track);
    w.path = vec![Vec2::new(0.0, 6.0)];
    bodies.walkers.push(w);
    for _ in 0..(5.0 / DT) as usize {
        bodies.step(DT, &mut track, None);
    }
    let w = &bodies.walkers[0];
    let (m0, m1) = track.ledger_kg();
    let mut deep = 0.0f32;
    let mut z = 0.5f32;
    let top = 0.25;
    while z < 5.0 {
        for side in [-1.0f32, 1.0] {
            deep = deep.max(top - track.surface_at(side * 0.095, z));
        }
        z += 0.025;
    }
    println!(
        "snow: steps {} print depth {:.3} hip {:.3} mass {:.4} -> {:.4} ({:+.2e})",
        w.steps,
        deep,
        w.pos.y,
        m0,
        m1,
        (m1 - m0) / m0.max(1.0)
    );
    assert!(w.steps >= 6);
    assert!(deep > 0.04 && deep < 0.2, "boot print depth in fresh snow");
    assert!(((m1 - m0) / m0.max(1.0)).abs() < 1.0e-6, "snow mass moved");
}

/// Highest foam on the coarse cells within `r` of a point.
fn foam_near(f: &FluidCore, x: f32, z: f32, r: f32) -> f32 {
    let mut m = 0.0f32;
    let n = (r / 0.1).ceil() as i32;
    for j in -n..=n {
        for i in -n..=n {
            let (px, pz) = (x + i as f32 * 0.1, z + j as f32 * 0.1);
            if (px - x).hypot(pz - z) <= r {
                m = m.max(f.foam_turbid_at(px, pz).0);
            }
        }
    }
    m
}

#[test]
fn legs_and_impacts_foam_and_splash_but_a_resting_buoy_does_not() {
    let level = 0.4;
    // Wading: foam along the path behind the legs, a few litres of drops.
    let mut fluid = pool(level);
    for _ in 0..30 {
        fluid.step(DT);
    }
    let v0 = water_m3(&fluid);
    let mut imprint = Imprint::new();
    let mut bodies = Bodies::new();
    {
        let ground = PrintGround {
            imprint: &mut imprint,
            base: bed_base(&fluid),
        };
        let mut w = Walker::new(Vec2::new(-4.0, 0.0), std::f32::consts::FRAC_PI_2, 1.4, &ground);
        w.path = vec![Vec2::new(4.5, 0.0)];
        bodies.walkers.push(w);
    }
    let mut at_legs = 0.0f32;
    let mut n = 0.0f32;
    run(&mut bodies, &mut fluid, &mut imprint, 4.0, |b, f, t| {
        if t > 1.0 {
            for (foot, _) in b.walkers[0].legs() {
                at_legs += foam_near(f, foot.x, foot.z, 0.3);
                n += 1.0;
            }
        }
    });
    let w = &bodies.walkers[0];
    let behind = Vec2::new(w.pos.x, w.pos.z) - w.forward() * 0.8;
    let trail = foam_near(&fluid, behind.x, behind.y, 0.4);
    let far = foam_near(&fluid, 0.0, 3.5, 0.4);
    let wade_spray = fluid.spray_m3();
    println!(
        "wade foam: at legs {:.3} trail {:.3} away {:.3} spray {:.2} l water+spray {:.5} -> {:.5}",
        at_legs / n,
        trail,
        far,
        wade_spray * 1000.0,
        v0,
        water_m3(&fluid)
    );
    assert!(at_legs / n > 0.15, "no foam at the wading legs");
    assert!(trail > 0.05, "no foam trail behind the walker");
    assert!(far < 0.01, "foam away from any body");
    assert!(wade_spray > 2.0e-4 && wade_spray < 0.01, "wading drops {wade_spray} m3");
    assert!(((water_m3(&fluid) - v0) / v0).abs() < 1.0e-4, "water book moved");

    // A ball dropped in: a crown of drops and a foam ring at the impact.
    let mut fluid = pool(level);
    for _ in 0..30 {
        fluid.step(DT);
    }
    let mut bodies = Bodies::new();
    bodies.items.push(Body::ball(Vec3::new(0.0, level + 0.6, 0.0), 0.22, 70.0, false));
    let mut imprint = Imprint::new();
    let mut ring = 0.0f32;
    let mut drops_up = 0usize;
    run(&mut bodies, &mut fluid, &mut imprint, 1.0, |_, f, _| {
        ring = ring.max(foam_near(f, 0.0, 0.0, 0.5));
        drops_up = drops_up.max(f.spray_drops().iter().filter(|d| d.vel[1] > 0.5).count());
    });
    let crown = fluid.spray_m3();
    println!("impact: foam {:.3} crown {:.2} l rising drops {}", ring, crown * 1000.0, drops_up);
    assert!(ring > 0.15, "no foam ring at the impact");
    assert!(crown > 2.0e-4 && crown < 0.01, "impact drops {crown} m3");
    assert!(drops_up >= 4, "crown drops should rise");

    // A buoy at rest: no foam, no drops once it has settled.
    let mut fluid = pool(0.6);
    for _ in 0..30 {
        fluid.step(DT);
    }
    let mut bodies = Bodies::new();
    bodies.items.push(Body::ball(Vec3::new(0.3, 0.6 + 0.10, -0.2), 0.18, 6.0, true));
    let mut imprint = Imprint::new();
    run(&mut bodies, &mut fluid, &mut imprint, 4.0, |_, _, _| {});
    let s0 = fluid.spray_m3();
    let mut calm = 0.0f32;
    run(&mut bodies, &mut fluid, &mut imprint, 3.0, |_, f, _| {
        calm = calm.max(foam_near(f, 0.3, -0.2, 0.5));
    });
    println!("buoy at rest: foam {:.4} drops {:.2e} m3", calm, fluid.spray_m3() - s0);
    assert!(calm < 0.02, "resting buoy foams");
    assert!(fluid.spray_m3() - s0 < 1.0e-7, "resting buoy splashes");
}

#[test]
fn no_bodies_leave_the_sheet_alone() {
    let mut fluid = pool(0.3);
    let mut bodies = Bodies::new();
    bodies.items.push(Body::ball(Vec3::new(0.0, 0.5, 0.0), 0.2, 3.0, true));
    let mut imprint = Imprint::new();
    run(&mut bodies, &mut fluid, &mut imprint, 1.0, |_, _, _| {});
    // Take the body out: the lift is gone on the next step.
    bodies.items.clear();
    bodies.hand_off(&mut fluid);
    for _ in 0..240 {
        fluid.step(DT);
    }
    // The column the ball sat in is level with the water around it again.
    let centre = eta_at(&fluid, 0.0, 0.0);
    let mut ring = 0.0f32;
    for k in 0..16 {
        let a = k as f32 / 16.0 * std::f32::consts::TAU;
        ring += eta_at(&fluid, a.cos(), a.sin()) / 16.0;
    }
    let spread = (centre - ring).abs();
    println!("after removal: centre {:.4} ring {:.4}", centre, ring);
    assert!(fluid.wet_loads().is_empty());
    assert!(spread < 0.005, "a hole stayed where the body was");
}

#[allow(dead_code)]
fn _ground_is_object_safe(g: &dyn Ground) -> f32 {
    g.surface(0.0, 0.0)
}


/// Top-down maps for the PR: the free surface around the wading walker and
/// the 2.5 cm prints on the beach. `SILT_BODIES_DUMP=dir cargo test --release
/// --test bodies dump_maps -- --ignored`.
#[test]
#[ignore]
fn dump_maps() {
    let Ok(dir) = std::env::var("SILT_BODIES_DUMP") else {
        return;
    };
    std::fs::create_dir_all(&dir).unwrap();
    let mut scene = silt::wade::WadeScene::new(Some(silt::ford::Swell::default()), 4.0);
    for _ in 0..(7.0 / DT) as usize {
        scene.step(DT);
    }
    let w = scene.bodies.walkers[0].clone();
    let c = Vec2::new(w.pos.x, w.pos.z);
    // Surface around the walker, 6 m square, 2.5 cm pixels, diverging blue/red
    // for ±3 cm against the mean.
    let n = 240usize;
    let side = 6.0f32;
    let mut etas = vec![0.0f32; n * n];
    let mut wet = vec![false; n * n];
    let mut mean = 0.0f32;
    let mut cnt = 0.0f32;
    for j in 0..n {
        for i in 0..n {
            let x = c.x - side * 0.5 + (i as f32 + 0.5) * side / n as f32;
            let z = c.y - side * 0.5 + (j as f32 + 0.5) * side / n as f32;
            let (e, b, _, _) = scene.fluid.surface_at(x, z);
            etas[j * n + i] = e;
            wet[j * n + i] = e - b > 0.01;
            if wet[j * n + i] {
                mean += e;
                cnt += 1.0;
            }
        }
    }
    mean /= cnt.max(1.0);
    // Take the swell and the beach out: what is left against a 1.2 m box
    // mean of the wet cells is what the legs do.
    let r = 24isize;
    let mut local = vec![0.0f32; n * n];
    for j in 0..n as isize {
        for i in 0..n as isize {
            let (mut s, mut k) = (0.0f32, 0.0f32);
            for dj in (-r..=r).step_by(4) {
                for di in (-r..=r).step_by(4) {
                    let (a, b) = (i + di, j + dj);
                    if a < 0 || b < 0 || a >= n as isize || b >= n as isize {
                        continue;
                    }
                    let id = b as usize * n + a as usize;
                    if wet[id] {
                        s += etas[id];
                        k += 1.0;
                    }
                }
            }
            local[j as usize * n + i as usize] = if k > 0.0 { s / k } else { mean };
        }
    }
    let mut img = image::RgbImage::new(n as u32, n as u32);
    for j in 0..n {
        for i in 0..n {
            let px = if !wet[j * n + i] {
                image::Rgb([196u8, 170, 120])
            } else {
                let t = ((etas[j * n + i] - local[j * n + i]) / 0.015).clamp(-1.0, 1.0);
                if t > 0.0 {
                    image::Rgb([255, (255.0 * (1.0 - t)) as u8, (255.0 * (1.0 - t)) as u8])
                } else {
                    image::Rgb([(255.0 * (1.0 + t)) as u8, (255.0 * (1.0 + t)) as u8, 255])
                }
            };
            img.put_pixel(i as u32, (n - 1 - j) as u32, px);
        }
    }
    img.save(format!("{dir}/wade-surface.png")).unwrap();
    // Foam field (#104) around the walker and the buoys, 8 m square: water
    // blue, foam white, dry tan, bodies red.
    let nf = 320usize;
    let sf = 8.0f32;
    let mut imgf = image::RgbImage::new(nf as u32, nf as u32);
    for j in 0..nf {
        for i in 0..nf {
            let x = c.x - sf * 0.5 + (i as f32 + 0.5) * sf / nf as f32;
            let z = c.y - sf * 0.5 + (j as f32 + 0.5) * sf / nf as f32;
            let (e, b, _, _) = scene.fluid.surface_at(x, z);
            let px = if e - b < 0.004 {
                image::Rgb([196u8, 170, 120])
            } else {
                let f = scene.fluid.foam_turbid_at(x, z).0.clamp(0.0, 1.0);
                let lerp = |a: f32, b: f32| (a + (b - a) * f) as u8;
                image::Rgb([lerp(40.0, 250.0), lerp(80.0, 250.0), lerp(130.0, 250.0)])
            };
            imgf.put_pixel(i as u32, (nf - 1 - j) as u32, px);
        }
    }
    let mut mark = |x: f32, z: f32| {
        let i = ((x - (c.x - sf * 0.5)) / sf * nf as f32) as i32;
        let j = ((z - (c.y - sf * 0.5)) / sf * nf as f32) as i32;
        for d in -2..=2 {
            for (a, b) in [(i + d, j), (i, j + d)] {
                if a >= 0 && b >= 0 && (a as usize) < nf && (b as usize) < nf {
                    imgf.put_pixel(a as u32, (nf - 1 - b as usize) as u32, image::Rgb([220, 30, 30]));
                }
            }
        }
    };
    for (foot, _) in w.legs() {
        mark(foot.x, foot.z);
    }
    for b in &scene.bodies.items {
        mark(b.pos.x, b.pos.z);
    }
    imgf.save(format!("{dir}/wade-foam.png")).unwrap();
    // Prints along the walk: offset ±2 cm, grey.
    let n2 = 480usize;
    let side2 = 12.0f32;
    let c2 = Vec2::new(-1.0, silt::wade::beach_z(0.1));
    let mut img2 = image::RgbImage::new(n2 as u32, n2 as u32);
    for j in 0..n2 {
        for i in 0..n2 {
            let x = c2.x - side2 * 0.5 + (i as f32 + 0.5) * side2 / n2 as f32;
            let z = c2.y - side2 * 0.5 + (j as f32 + 0.5) * side2 / n2 as f32;
            let d = scene.imprint.offset_at(x, z);
            let v = (128.0 + d / 0.02 * 127.0).clamp(0.0, 255.0) as u8;
            img2.put_pixel(i as u32, (n2 - 1 - j) as u32, image::Rgb([v, v, v]));
        }
    }
    img2.save(format!("{dir}/wade-prints.png")).unwrap();
    // Boot prints in snow.
    let mut track = SnowTrack::new(0.25, 0.0);
    track.rho_blanket = 160.0;
    let mut bodies = Bodies::new();
    let mut wk = Walker::new(Vec2::new(0.0, 0.0), 0.0, 1.2, &track);
    wk.path = vec![Vec2::new(0.0, 2.5), Vec2::new(1.5, 4.5), Vec2::new(1.5, 7.0)];
    bodies.walkers.push(wk);
    for _ in 0..(7.0 / DT) as usize {
        bodies.step(DT, &mut track, None);
    }
    let n3 = 320usize;
    let side3 = 8.0f32;
    let mut img3 = image::RgbImage::new(n3 as u32, n3 as u32);
    for j in 0..n3 {
        for i in 0..n3 {
            let x = 0.75 - side3 * 0.5 + (i as f32 + 0.5) * side3 / n3 as f32;
            let z = 3.5 - side3 * 0.5 + (j as f32 + 0.5) * side3 / n3 as f32;
            let d = track.surface_at(x, z) - 0.25;
            let v = (200.0 + d / 0.12 * 200.0).clamp(0.0, 255.0) as u8;
            img3.put_pixel(i as u32, (n3 - 1 - j) as u32, image::Rgb([v, v, (v as f32 * 0.9 + 25.0).min(255.0) as u8]));
        }
    }
    img3.save(format!("{dir}/snow-boots.png")).unwrap();
    println!("walker at ({:.2},{:.2}) steps {} mean surface {:.3}", c.x, c.y, w.steps, mean);
}
