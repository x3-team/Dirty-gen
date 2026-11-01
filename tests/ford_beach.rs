//! River ford and sand beach. The layout is only the initial bed, the still
//! water, the current, and a paddle strip. Everything checked here is read
//! back from the sheet after it has stepped: the current that survives the
//! rocks, the head and the wake around a boulder, the swash on the sand, the
//! wet band the swash leaves, and the contract maps the renderer reads.

use silt::fluid_core::{CoarseCell, FluidCore, COARSE_CELL_M, WINDOW_RES};
use silt::ford::{
    seed_ford, still_surface, still_waterline_z, Swell, CHANNEL_Z0, CHANNEL_Z1, DRY_SAND_MOISTURE,
    ROCKS,
};
use silt::particles::Particle;
use silt::render_contract::{
    publish, ContractFrame, FLOW_FLUID_SHEET, VALID_FILM, VALID_FLOW, VALID_MOISTURE,
};
use silt::soil::SoilField;
use std::time::Instant;

const DT: f32 = 1.0 / 30.0;
const RIVER_DROP: f32 = silt::ford::RIVER_SLOPE;

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

/// Southernmost wet cell on the beach along column `x`, metres of `z`.
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

/// Mean speed and mean +X velocity in the rock-free crossing lane and in the
/// reach just below the inlet, on the channel floor.
fn channel_current(fluid: &FluidCore) -> (f32, f32) {
    let mut s = 0.0f32;
    let mut u = 0.0f32;
    let mut n = 0.0f32;
    for x in [-17.5f32, -16.5, -1.2, -0.4, 0.4, 1.2] {
        let mut z = CHANNEL_Z0 + 1.0;
        while z < CHANNEL_Z1 - 1.0 {
            let c = cell_at(fluid, x, z);
            if c.water_m > 0.1 {
                s += c.u.hypot(c.v);
                u += c.u;
                n += 1.0;
            }
            z += 0.4;
        }
    }
    if n > 0.0 {
        (s / n, u / n)
    } else {
        (0.0, 0.0)
    }
}

fn soil_for(fluid: &FluidCore) -> SoilField {
    let mut soil = SoilField::flat(0.0);
    let half = soil.size * 0.5;
    for j in 0..soil.res {
        for i in 0..soil.res {
            let x = (i as f32 + 0.5) * soil.cell - half;
            let z = (j as f32 + 0.5) * soil.cell - half;
            let s = fluid.sample(x, z);
            let idx = j * soil.res + i;
            soil.initial_height[idx] = s.bed_m;
            soil.bed[idx] = s.bed_m;
            soil.height[idx] = s.bed_m;
            soil.placed[idx] = 0.0;
            soil.water[idx] = 0.0;
            soil.slurry[idx] = 0.0;
            soil.wetness[idx] = 0.0;
        }
    }
    soil
}

#[test]
fn ford_river_keeps_its_current_and_parts_around_the_rocks() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    let vol0 = seed_ford(&mut fluid, None);
    let water0 = fluid.ledger().water_m3 + fluid.ledger().reach_m3;
    assert!((vol0 - water0).abs() < 1.0e-2 * vol0.max(1.0));
    let steps = (14.0 / DT) as u32;
    let t = Instant::now();
    for _ in 0..steps {
        fluid.step(DT);
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / steps as f64;
    let (speed, u_mean) = channel_current(&fluid);
    let gap = fluid.ledger().water_m3 + fluid.ledger().reach_m3 - water0;
    println!(
        "FORD_RIVER steps {steps} step_ms {ms:.3} current {speed:.3} u {u_mean:.3} water {water0:.3} gap {gap:.5} nan {}",
        fluid.non_finite_count()
    );
    assert_eq!(fluid.non_finite_count(), 0);
    assert!(
        gap.abs() < 0.005 * water0,
        "ford water drifted {gap} m3 of {water0}"
    );
    assert!(u_mean > 0.3, "river lost its current: u {u_mean}");
    assert!(speed < 2.0, "river ran away: {speed}");

    // Each standing boulder, averaged over two more seconds: water piles on
    // its upstream face and drops in its lee, and the stream runs faster past
    // its flanks than in the sheltered water behind it.
    let tall: Vec<_> = ROCKS
        .iter()
        .filter(|r| r.top > 0.0 && r.z > CHANNEL_Z0 + 1.0)
        .copied()
        .collect();
    let mut acc = vec![[0.0f32; 6]; tall.len()];
    let avg_steps = (2.0 / DT) as u32;
    for _ in 0..avg_steps {
        fluid.step(DT);
        for (k, rock) in tall.iter().enumerate() {
            let r = rock_ring(&fluid, rock);
            let (south, north) = parting(&fluid, rock);
            let all = [r[0], r[1], r[2], r[3], south, north];
            for q in 0..6 {
                acc[k][q] += all[q] / avg_steps as f32;
            }
        }
    }
    let mut parted = 0;
    for (k, rock) in tall.iter().enumerate() {
        let [up, lee, flank, lee_s, v_s, v_n] = acc[k];
        let reach = 2.0 * 1.35 * rock.rx.max(rock.rz);
        let head = up - lee - RIVER_DROP * reach;
        println!(
            "FORD_ROCK x {:.1} z {:.1} top {:+.2} head {:.4} flank {:.3} lee {:.3} v_south {:+.3} v_north {:+.3}",
            rock.x, rock.z, rock.top, head, flank, lee_s, v_s, v_n
        );
        // The stream piles against the upstream face and parts around it:
        // water south of the centre line turns south, water north turns north,
        // and the water sheltered in its lee runs slower than past its flanks.
        if head > 0.005 && v_s < -0.03 && v_n > 0.03 && flank > 1.3 * lee_s {
            parted += 1;
        }
    }
    assert!(
        parted * 3 >= tall.len() * 2,
        "only {parted} of {} boulders part the stream",
        tall.len()
    );
}

/// Mean cross-stream velocity, m/s, just upstream of the boulder on its
/// south and north halves: the water turning aside to pass it.
fn parting(fluid: &FluidCore, rock: &silt::ford::Rock) -> (f32, f32) {
    let r = 1.2 * rock.rx.max(rock.rz) + 0.2;
    let mut s = 0.0f32;
    let mut ns = 0.0f32;
    let mut n = 0.0f32;
    let mut nn = 0.0f32;
    for k in 0..24 {
        // 100° to 170° and 190° to 260°: the upstream face, off the centre line.
        let deg = 100.0 + 70.0 * k as f32 / 23.0;
        for side in [1.0f32, -1.0] {
            let a = (180.0 + side * (180.0 - deg)).to_radians();
            let c = cell_at(fluid, rock.x + a.cos() * r, rock.z + a.sin() * r);
            if c.water_m < 0.02 {
                continue;
            }
            // side +1 maps the arc to 190°..260°, the south half.
            if side > 0.0 {
                s += c.v;
                ns += 1.0;
            } else {
                n += c.v;
                nn += 1.0;
            }
        }
    }
    (
        if ns > 0.0 { s / ns } else { 0.0 },
        if nn > 0.0 { n / nn } else { 0.0 },
    )
}

/// Free surface on the upstream and lee arcs of a boulder, and the speed on
/// its flank arcs and in its lee, sampled on an ellipse 1.35 times the rock.
/// Arcs are ±40° around each direction; dry samples are skipped.
fn rock_ring(fluid: &FluidCore, rock: &silt::ford::Rock) -> [f32; 4] {
    let mut sum = [0.0f32; 4];
    let mut n = [0.0f32; 4];
    let r = 1.35 * rock.rx.max(rock.rz) + 0.15;
    for k in 0..48 {
        let a = k as f32 * std::f32::consts::TAU / 48.0;
        let (sa, ca) = a.sin_cos();
        let c = cell_at(fluid, rock.x + ca * r, rock.z + sa * r);
        if c.water_m < 0.02 {
            continue;
        }
        let deg = a.to_degrees();
        let speed = c.u.hypot(c.v);
        // Upstream is -X (180°), lee is +X (0°), flanks are ±90°.
        if (deg - 180.0).abs() < 40.0 {
            sum[0] += eta(&c);
            n[0] += 1.0;
        }
        if deg < 40.0 || deg > 320.0 {
            sum[1] += eta(&c);
            n[1] += 1.0;
            sum[3] += speed;
            n[3] += 1.0;
        }
        if (deg - 90.0).abs() < 40.0 || (deg - 270.0).abs() < 40.0 {
            sum[2] += speed;
            n[2] += 1.0;
        }
    }
    let mut out = [0.0f32; 4];
    for q in 0..4 {
        out[q] = if n[q] > 0.0 { sum[q] / n[q] } else { 0.0 };
    }
    out
}

#[test]
fn beach_swash_runs_up_the_sand_and_leaves_a_wet_band() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    seed_ford(&mut fluid, Some(Swell::default()));
    let held = |f: &FluidCore| {
        let l = f.ledger();
        l.water_m3 + l.reach_m3
    };
    let water0 = held(&fluid);
    let probe_x = [-14.0f32, -2.0, 0.0, 2.0, 16.0];
    let still = still_waterline_z();
    let mut lo = [f32::MAX; 5];
    let mut hi = [f32::MIN; 5];
    let steps = (24.0 / DT) as u32;
    let mut swash_cells = 0u32;
    let mut swash_foam = 0u32;
    let soil = soil_for(&fluid);
    let mut frame = ContractFrame::new();
    let particles: &[Particle] = &[];
    for s in 0..steps {
        fluid.step(DT);
        if s as f32 * DT < 8.0 {
            continue;
        }
        for (k, x) in probe_x.iter().enumerate() {
            let z = shoreline_z(&fluid, *x);
            lo[k] = lo[k].min(z);
            hi[k] = hi[k].max(z);
        }
        if s % 15 == 0 {
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
            // Foam on the swash itself: wet cells within a metre of the still
            // waterline, away from the boulder on the beach.
            let (ox, oz) = (frame.header.origin_x, frame.header.origin_z);
            for j in 0..WINDOW_RES {
                let z = oz + (j as f32 + 0.5) * COARSE_CELL_M;
                if (z - still).abs() > 1.0 {
                    continue;
                }
                for i in 0..WINDOW_RES {
                    let x = ox + (i as f32 + 0.5) * COARSE_CELL_M;
                    let idx = j * WINDOW_RES + i;
                    if silt::ford::on_rock(x, z) || (x - ROCKS[10].x).hypot(z - ROCKS[10].z) < 1.5 {
                        continue;
                    }
                    if frame.film[idx * 2] < 0.004 {
                        continue;
                    }
                    swash_cells += 1;
                    if frame.foam_crest[idx * 2] > 0.3 {
                        swash_foam += 1;
                    }
                }
            }
        }
    }
    let gap = held(&fluid) - water0;
    let mut excursion = 0.0f32;
    for k in 0..probe_x.len() {
        println!(
            "FORD_SWASH x {:.1} still {:.2} rundown {:.2} runup {:.2}",
            probe_x[k], still, hi[k], lo[k]
        );
        excursion += hi[k] - lo[k];
    }
    excursion /= probe_x.len() as f32;
    // Run-up is how far south of the still line the wet front reaches.
    let runup = probe_x
        .iter()
        .enumerate()
        .map(|(k, _)| still - lo[k])
        .fold(0.0f32, f32::max);
    // Moisture: the sand each probe's swash reached and left again is still
    // wet; the terrace above the highest run-up is dry.
    let mut wet_above = 0.0f32;
    let mut n = 0.0f32;
    for (k, x) in probe_x.iter().enumerate() {
        let mut z = still - COARSE_CELL_M;
        while z > lo[k] + 0.5 * COARSE_CELL_M {
            let c = cell_at(&fluid, *x, z);
            if c.water_m < 0.003 {
                wet_above += c.moisture;
                n += 1.0;
            }
            z -= COARSE_CELL_M;
        }
    }
    let wet_above = if n > 0.0 { wet_above / n } else { 1.0 };
    let dry = cell_at(&fluid, 2.0, still - runup - 2.5).moisture;
    let foam_swash = swash_foam as f32 / swash_cells.max(1) as f32;
    println!(
        "FORD_BEACH excursion {excursion:.2} runup {runup:.2} wet_band {wet_above:.3} dry {dry:.3} swash_foam_share {foam_swash:.3} gap {gap:.5} nan {}",
        fluid.non_finite_count()
    );
    assert_eq!(fluid.non_finite_count(), 0);
    assert!(gap.abs() < 0.005 * water0, "beach water drifted {gap}");
    assert!(excursion > 0.4, "swash barely moves: {excursion} m");
    assert!(runup > 0.6, "swash does not climb the sand: {runup} m");
    assert!(wet_above > 0.6, "swash zone dried to {wet_above}");
    assert!(dry < DRY_SAND_MOISTURE + 0.05, "terrace got wet: {dry}");
    // Foam is the contract's Froude/crest foam on the thin, fast swash. Some
    // of the swash foams, not all of it.
    assert!(
        foam_swash > 0.05 && foam_swash < 0.9,
        "swash foam share {foam_swash}"
    );
}

#[test]
fn ford_contract_carries_flow_film_and_wet_sand() {
    let mut fluid = FluidCore::new([0.0, 0.0]);
    seed_ford(&mut fluid, Some(Swell::default()));
    for _ in 0..(10.0 / DT) as u32 {
        fluid.step(DT);
    }
    let soil = soil_for(&fluid);
    let mut frame = ContractFrame::new();
    let particles: &[Particle] = &[];
    publish(
        &mut frame,
        &soil,
        particles,
        [0.0, 0.0],
        10.0,
        0,
        Some(&fluid),
        None,
    );
    let h = frame.header;
    assert_eq!(h.flow_source, FLOW_FLUID_SHEET);
    for bit in [VALID_FLOW, VALID_FILM, VALID_MOISTURE] {
        assert!(h.valid & bit != 0, "valid bit {bit} clear");
    }
    let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
    assert!(finite(&frame.ground_rut));
    assert!(finite(&frame.film));
    assert!(finite(&frame.flow));
    assert!(finite(&frame.moisture));
    assert!(finite(&frame.foam_crest));
    // A boulder crown is ground in binding 1 that stands above the surface.
    let rock = ROCKS.iter().find(|r| r.top > 0.2).expect("tall rock");
    let i = ((rock.x - h.origin_x) / h.cell_m).floor() as usize;
    let j = ((rock.z - h.origin_z) / h.cell_m).floor() as usize;
    let idx = j * WINDOW_RES + i;
    let ground = frame.ground_rut[idx * 4];
    assert!(
        ground > still_surface(rock.x) + 0.1,
        "rock crown {ground} is not out of the water"
    );
    assert!(frame.film[idx * 2] < 0.01, "water stands on the crown");
    // Channel carries an eastward flow in binding 4.
    let j = ((5.0 - h.origin_z) / h.cell_m).floor() as usize;
    let i = ((-1.0 - h.origin_x) / h.cell_m).floor() as usize;
    let idx = j * WINDOW_RES + i;
    assert!(frame.flow[idx * 2] > 0.2, "flow {}", frame.flow[idx * 2]);
    let _ = CHANNEL_Z1;
}

/// Top-down picture of the published contract for eyeballing: bed shade,
/// wet sand, water depth, speed, and foam. Not a beauty frame.
/// `SILT_FORD_DUMP=dir cargo test --release --test ford_beach ford_dump_maps -- --ignored`.
#[test]
#[ignore]
fn ford_dump_maps() {
    let Ok(dir) = std::env::var("SILT_FORD_DUMP") else {
        return;
    };
    let mut fluid = FluidCore::new([0.0, 0.0]);
    seed_ford(&mut fluid, Some(Swell::default()));
    let soil = soil_for(&fluid);
    let mut frame = ContractFrame::new();
    let particles: &[Particle] = &[];
    let secs: f32 = std::env::var("SILT_FORD_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12.0);
    let steps = (secs / DT).round() as u32;
    for s in 0..steps {
        fluid.step(DT);
        // Six frames, half a second apart, at the end of the run.
        let left = steps - 1 - s;
        if left % 15 != 0 || left > 75 {
            continue;
        }
        let t = (s + 1) as f32 * DT;
        publish(
            &mut frame,
            &soil,
            particles,
            [0.0, 0.0],
            t,
            0,
            Some(&fluid),
            None,
        );
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
                        let py = (n - 1 - j) * scale + dy;
                        let px = i * scale + dx;
                        let o = (py * w + px) * 3;
                        for k in 0..3 {
                            let mut c = sand[k] * shade * (1.0 - a) + water[k] * a;
                            c = c * (1.0 - foam * 0.8) + 0.95 * foam * 0.8;
                            img[o + k] = (c.clamp(0.0, 1.0) * 255.0) as u8;
                        }
                    }
                }
            }
        }
        let path = format!("{dir}/ford-{:02}.ppm", 5 - left / 15);
        let mut out = format!("P6 {w} {w} 255\n").into_bytes();
        out.extend_from_slice(&img);
        std::fs::write(&path, out).expect("write");
    }
}
