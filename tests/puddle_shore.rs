//! The puddle shore is the sheet at rest on centimetre relief.
//!
//! The pour starts as a compass oval. After the gravity sheets drain, the
//! 4 mm contour of water + slurry is a run of bays and tongues
//! (perimeter² / (4π·area) ≥ 1.3), not the oval and not two painted strips.
//! The volume book stays within 1e-3, dry cells are 0, and a release step
//! stays inside 7 ms with the water sheet inside 6 ms.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use silt::fluid_core::{FluidCore, COARSE_CELL_M, FINE_CELL_M, WINDOW_RES};
use silt::world::ThroughDrive;

const FILM_M: f32 = 0.004;
const STEP_BUDGET_MS: f32 = 7.0;
const WATER_BUDGET_MS: f32 = 6.0;
/// Relative water, slurry, and spray. Half a percent of the larger volume.
const VOL_REL: f32 = 5.0e-3;
/// L1 of |Δh| on the coarse film, cubic metres. 0.10 m³ on the drained
/// sheet is about 3 mm average. The uniform eight-substep solve is the reference.
const L1_DEPTH_M3: f64 = 0.10;
/// Peak cell difference on that same film. A tongue that only moved one
/// coarse cell stays under this. A different bay does not.
const MAX_DH_M: f32 = 0.05;
const STORE_DIR: &str =
    "/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media/puddle-shore";

struct MaskStats {
    q: f32,
    perimeter_m: f32,
    area_m2: f32,
    cells: u32,
}

fn mask_stats(depth: &[f32], cell: f32) -> MaskStats {
    let res = WINDOW_RES;
    let mut cells = 0u32;
    let mut edges = 0u32;
    for j in 0..res {
        for i in 0..res {
            let idx = j * res + i;
            if depth[idx] < FILM_M {
                continue;
            }
            cells += 1;
            for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let ii = i as i32 + di;
                let jj = j as i32 + dj;
                let dry = ii < 0
                    || jj < 0
                    || ii >= res as i32
                    || jj >= res as i32
                    || depth[(jj as usize) * res + (ii as usize)] < FILM_M;
                if dry {
                    edges += 1;
                }
            }
        }
    }
    let area = cells as f32 * cell * cell;
    let perimeter = edges as f32 * cell;
    let q = if area > 1.0e-8 {
        perimeter * perimeter / (4.0 * std::f32::consts::PI * area)
    } else {
        0.0
    };
    MaskStats {
        q,
        perimeter_m: perimeter,
        area_m2: area,
        cells,
    }
}

fn live_films(fluid: &FluidCore) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = WINDOW_RES * WINDOW_RES;
    let mut water = vec![0.0f32; n];
    let mut slurry = vec![0.0f32; n];
    let mut film = vec![0.0f32; n];
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let cell = fluid.coarse_cell(i, j).expect("cell");
            let idx = j * WINDOW_RES + i;
            water[idx] = cell.water_m;
            slurry[idx] = cell.slurry_m;
            film[idx] = cell.water_m + cell.slurry_m;
        }
    }
    (water, slurry, film)
}

fn color_of(depth: f32, edge: bool) -> [u8; 3] {
    if edge {
        return [255, 196, 48];
    }
    if depth < FILM_M {
        return [16, 22, 18];
    }
    let t = ((depth - FILM_M) / 0.09).clamp(0.0, 1.0);
    let r = (24.0 + 30.0 * t) as u8;
    let g = (70.0 + 150.0 * t) as u8;
    let b = (120.0 + 90.0 * (1.0 - t)) as u8;
    [r, g, b]
}

fn write_depth_png(path: &Path, depth: &[f32], x0: usize, x1: usize, z0: usize, z1: usize) {
    let scale = 6u32;
    let w = ((x1 - x0) as u32) * scale;
    let h = ((z1 - z0) as u32) * scale;
    let mut rgb = vec![0u8; (w * h * 3) as usize];
    for j in z0..z1 {
        for i in x0..x1 {
            let idx = j * WINDOW_RES + i;
            let d = depth[idx];
            let wet = d >= FILM_M;
            let mut edge = false;
            if wet {
                for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let ii = i as i32 + di;
                    let jj = j as i32 + dj;
                    let dry = ii < 0
                        || jj < 0
                        || ii >= WINDOW_RES as i32
                        || jj >= WINDOW_RES as i32
                        || depth[(jj as usize) * WINDOW_RES + ii as usize] < FILM_M;
                    if dry {
                        edge = true;
                        break;
                    }
                }
            }
            let px = color_of(d, edge);
            for sy in 0..scale {
                for sx in 0..scale {
                    let x = (i - x0) as u32 * scale + sx;
                    let y = (j - z0) as u32 * scale + sy;
                    let o = ((y * w + x) * 3) as usize;
                    rgb[o] = px[0];
                    rgb[o + 1] = px[1];
                    rgb[o + 2] = px[2];
                }
            }
        }
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("depth dir");
    }
    let file = File::create(path).unwrap_or_else(|err| panic!("create {}: {err}", path.display()));
    let mut encoder = png::Encoder::new(BufWriter::new(file), w, h);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(&rgb).expect("png data");
}

/// How far the 4 mm contour sticks out, by angle around the puddle.
///
/// A compass oval and a beach lens have two broad ends. A shore that has
/// drained into the relief has several tongues, each standing out from the
/// bays beside it.
fn shore_tongues(depth: &[f32], origin_x: f32, origin_z: f32) -> (u32, f32) {
    const BINS: usize = 72;
    let mut radius = vec![0.0f32; BINS];
    let mut any = vec![false; BINS];
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let idx = j * WINDOW_RES + i;
            if depth[idx] < FILM_M {
                continue;
            }
            let x = origin_x + (i as f32 + 0.5) * COARSE_CELL_M;
            let z = origin_z + (j as f32 + 0.5) * COARSE_CELL_M;
            let dx = x;
            let dz = z + 14.0;
            let r = dx.hypot(dz);
            let t = (dz.atan2(dx) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
            let bin = ((t * BINS as f32).floor() as usize).min(BINS - 1);
            if !any[bin] || r > radius[bin] {
                radius[bin] = r;
                any[bin] = true;
            }
        }
    }
    let mut lo = f32::MAX;
    let mut hi = 0.0f32;
    let mut maxima = Vec::new();
    for bin in 0..BINS {
        if !any[bin] {
            continue;
        }
        lo = lo.min(radius[bin]);
        hi = hi.max(radius[bin]);
        let prev_i = (bin + BINS - 1) % BINS;
        let next_i = (bin + 1) % BINS;
        if any[prev_i]
            && any[next_i]
            && radius[bin] >= radius[prev_i]
            && radius[bin] > radius[next_i]
        {
            maxima.push(bin);
        }
    }
    let mut tongues = 0u32;
    let n = maxima.len();
    for (k, &bin) in maxima.iter().enumerate() {
        if n < 2 {
            break;
        }
        let prev_m = maxima[(k + n - 1) % n];
        let next_m = maxima[(k + 1) % n];
        let mut valley = radius[bin];
        let mut b = (prev_m + 1) % BINS;
        let mut guard = 0;
        while b != next_m && guard < BINS {
            if any[b] {
                valley = valley.min(radius[b]);
            }
            b = (b + 1) % BINS;
            guard += 1;
        }
        // A bay is the low ground between this tongue and its neighbours.
        if radius[bin] > valley + 0.55 {
            tongues += 1;
        }
    }
    let span = if hi > lo { hi - lo } else { 0.0 };
    (tongues, span)
}

fn crop_of(a: &[f32], b: &[f32]) -> (usize, usize, usize, usize) {
    let mut i0 = WINDOW_RES;
    let mut i1 = 0usize;
    let mut j0 = WINDOW_RES;
    let mut j1 = 0usize;
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let idx = j * WINDOW_RES + i;
            if a[idx] < FILM_M && b[idx] < FILM_M {
                continue;
            }
            i0 = i0.min(i);
            i1 = i1.max(i);
            j0 = j0.min(j);
            j1 = j1.max(j);
        }
    }
    let pad = 3usize;
    (
        i0.saturating_sub(pad),
        (i1 + 1 + pad).min(WINDOW_RES),
        j0.saturating_sub(pad),
        (j1 + 1 + pad).min(WINDOW_RES),
    )
}

#[test]
fn puddle_shore_follows_relief() {
    let mut drive = ThroughDrive::new();
    let (water, slurry, film) = live_films(&drive.fluid);
    let before = drive.shore_before();
    assert_eq!(before.len(), film.len());

    for (name, grid) in [("water", &water), ("slurry", &slurry)] {
        for (k, v) in grid.iter().enumerate() {
            assert!(v.is_finite(), "{name} nan at {k}");
            if *v < 1.0e-6 {
                assert_eq!(*v, 0.0, "{name} dry cell {k} is {v}");
            }
        }
    }
    assert_eq!(drive.fluid.non_finite_count(), 0, "non-finite sheet");

    let books = drive.fluid.ledger();
    let held = (books.water_m3
        + books.slurry_m3
        + books.spray_m3
        + books.frozen_water_m3
        + books.frozen_slurry_m3) as f64;
    let gap = (held - drive.sheet_opened_m3()).abs();
    let before_stats = mask_stats(before, COARSE_CELL_M);
    let before_water = mask_stats(drive.shore_before_water(), COARSE_CELL_M);
    let film_stats = mask_stats(&film, COARSE_CELL_M);
    let water_stats = mask_stats(&water, COARSE_CELL_M);

    let (ox, oz, _) = drive.fluid.window_bounds();
    let mut escaped = 0u32;
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let idx = j * WINDOW_RES + i;
            if water[idx] < FILM_M {
                continue;
            }
            let x = ox + (i as f32 + 0.5) * COARSE_CELL_M;
            let z = oz + (j as f32 + 0.5) * COARSE_CELL_M;
            let e = (x / 5.6).powi(2) + ((z + 14.0) / 4.8).powi(2);
            if e > 1.0 {
                escaped += 1;
            }
        }
    }

    let (tongues, tongue_span) = shore_tongues(&film, ox, oz);
    println!(
        "puddle shore before film q {:.3} water q {:.3} area {:.2} m2 peri {:.2} m\n\
         puddle shore film q {:.3} area {:.2} m2 peri {:.2} m cells {}\n\
         puddle shore water q {:.3} area {:.2} m2 peri {:.2} m cells {}\n\
         puddle shore gap {:.6} opened {:.4} held {:.4} water {:.4} slurry {:.4} spray {:.4} \
         frozen {:.4} escaped {escaped}\n\
         puddle shore tongues {tongues} span {:.2} m",
        before_stats.q,
        before_water.q,
        before_water.area_m2,
        before_water.perimeter_m,
        film_stats.q,
        film_stats.area_m2,
        film_stats.perimeter_m,
        film_stats.cells,
        water_stats.q,
        water_stats.area_m2,
        water_stats.perimeter_m,
        water_stats.cells,
        gap,
        drive.sheet_opened_m3(),
        held,
        books.water_m3,
        books.slurry_m3,
        books.spray_m3,
        books.frozen_water_m3 + books.frozen_slurry_m3,
        tongue_span
    );

    let crop = crop_of(before, &film);
    let store = Path::new(STORE_DIR);
    if Path::new("/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e").exists() {
        write_depth_png(&store.join("depth-before.png"), before, crop.0, crop.1, crop.2, crop.3);
        write_depth_png(&store.join("depth-after.png"), &film, crop.0, crop.1, crop.2, crop.3);
        for name in ["depth-before.png", "depth-after.png"] {
            let path = store.join(name);
            let len = std::fs::metadata(&path)
                .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
                .len();
            assert!(len > 500, "{} is {len} bytes", path.display());
        }
    }

    assert!(
        tongues >= 4 && tongue_span >= 1.20,
        "shore is still a lens: {tongues} tongues, radius span {tongue_span:.2} m"
    );
    assert!(
        film_stats.q >= 1.3,
        "film contour q {:.3} area {:.2} peri {:.2}",
        film_stats.q,
        film_stats.area_m2,
        film_stats.perimeter_m
    );
    assert!(
        water_stats.q >= 1.3,
        "water contour q {:.3} is still a compass oval",
        water_stats.q
    );
    // A raster oval already clears 1.3. The tongue count is what rejects it.
    assert!(
        (12.0..90.0).contains(&water_stats.area_m2),
        "water area {:.2} m2",
        water_stats.area_m2
    );
    assert!(
        escaped * 20 < water_stats.cells.max(1),
        "escaped cells {escaped} of {}",
        water_stats.cells
    );
    assert!(gap <= 1.0e-3, "volume gap {gap:.6}");

    let dt = 1.0 / 30.0;
    let mut sum = silt::world::DriveParts::default();
    let mut n = 0.0f64;
    for _ in 0..48 {
        drive.step(dt);
        let z = drive.world.vehicle.pos.z;
        if !(-17.2..-12.2).contains(&z) {
            continue;
        }
        let step = drive.last_parts;
        sum.water_ms += step.water_ms;
        sum.slurry_ms += step.slurry_ms;
        sum.soil_ms += step.soil_ms;
        sum.air_ms += step.air_ms;
        sum.dust_ms += step.dust_ms;
        sum.restrict_ms += step.restrict_ms;
        sum.window_ms += step.window_ms;
        sum.prep_ms += step.prep_ms;
        sum.sheet_rest_ms += step.sheet_rest_ms;
        n += 1.0;
    }
    assert!(n >= 8.0, "timed steps {n}");
    let mean = sum.solver_ms() / n;
    println!(
        "puddle shore step ms {:.3} water {:.3} slurry {:.3} soil {:.3} air {:.3} dust {:.3} \
         restrict {:.3} window {:.3} prep {:.3} rest {:.3} samples {n:.0}",
        mean,
        sum.water_ms / n,
        sum.slurry_ms / n,
        sum.soil_ms / n,
        sum.air_ms / n,
        sum.dust_ms / n,
        sum.restrict_ms / n,
        sum.window_ms / n,
        sum.prep_ms / n,
        sum.sheet_rest_ms / n
    );
    if !cfg!(debug_assertions) {
        let water = sum.water_ms / n;
        assert!(
            mean <= STEP_BUDGET_MS as f64,
            "solver step {mean:.3} ms over {STEP_BUDGET_MS}"
        );
        assert!(
            water <= WATER_BUDGET_MS as f64,
            "water {water:.3} ms over {WATER_BUDGET_MS}"
        );
    }
}

struct ShoreSnap {
    q: f32,
    tongues: u32,
    span: f32,
    water_m3: f32,
    slurry_m3: f32,
    spray_m3: f32,
    gap: f64,
    l1_m3: f64,
    max_dh: f32,
}

fn ledger_gap(drive: &ThroughDrive) -> (f32, f32, f32, f64) {
    let books = drive.fluid.ledger();
    let held = (books.water_m3
        + books.slurry_m3
        + books.spray_m3
        + books.frozen_water_m3
        + books.frozen_slurry_m3) as f64;
    (
        books.water_m3,
        books.slurry_m3,
        books.spray_m3,
        (held - drive.sheet_opened_m3()).abs(),
    )
}

fn peak_sheet_mps(fluid: &FluidCore) -> f32 {
    let mut peak = 0.0f32;
    for j in 0..WINDOW_RES {
        for i in 0..WINDOW_RES {
            let cell = fluid.coarse_cell(i, j).expect("cell");
            if cell.water_m + cell.slurry_m <= 1.0e-4 {
                continue;
            }
            peak = peak.max(cell.u.abs()).max(cell.v.abs());
        }
    }
    peak
}

fn film_delta(a: &[f32], b: &[f32]) -> (f64, f32) {
    let area = (COARSE_CELL_M as f64) * (COARSE_CELL_M as f64);
    let mut l1 = 0.0f64;
    let mut max_dh = 0.0f32;
    for (left, right) in a.iter().zip(b.iter()) {
        let d = (left - right).abs();
        max_dh = max_dh.max(d);
        l1 += d as f64 * area;
    }
    (l1, max_dh)
}

fn shore_snap(drive: &ThroughDrive, other_film: Option<&[f32]>) -> (ShoreSnap, Vec<f32>) {
    let (_water, _slurry, film) = live_films(&drive.fluid);
    let stats = mask_stats(&film, COARSE_CELL_M);
    let (ox, oz, _) = drive.fluid.window_bounds();
    let (tongues, span) = shore_tongues(&film, ox, oz);
    let (water_m3, slurry_m3, _in_air_m3, gap) = ledger_gap(drive);
    // Spray in the air is a moment; what the two paces must agree on is how
    // much the sheet threw. Landed drops are back in `water_m3`.
    let spray_m3 = drive.fluid.spray_thrown_m3();
    let (l1_m3, max_dh) = match other_film {
        Some(other) => film_delta(&film, other),
        None => (0.0, 0.0),
    };
    (
        ShoreSnap {
            q: stats.q,
            tongues,
            span,
            water_m3,
            slurry_m3,
            spray_m3,
            gap,
            l1_m3,
            max_dh,
        },
        film,
    )
}

/// Substeps the height–velocity sheet would take at `speed`, and the cap it
/// is actually allowed. The Courant number is `speed · dt / cell`.
fn cfl_split(dt: f32, speed: f32, cell: f32, cap: f32) -> (f32, u32) {
    let courant = dt * speed / cell.max(1.0e-3);
    let nsub = ((dt * speed.max(0.15)) / (0.42 * cell.max(1.0e-3)))
        .ceil()
        .clamp(1.0, cap) as u32;
    (courant, nsub)
}

struct DriveFrame {
    soil: silt::soil::SoilField,
    vehicle: silt::vehicle::Vehicle,
}

fn snap_drive_frame(drive: &ThroughDrive) -> DriveFrame {
    DriveFrame {
        soil: drive.world.soil.clone(),
        vehicle: drive.world.vehicle.clone(),
    }
}

fn restore_drive_frame(drive: &mut ThroughDrive, frame: &DriveFrame) {
    drive.world.soil = frame.soil.clone();
    drive.world.vehicle = frame.vehicle.clone();
}

fn rel_vol(a: f32, b: f32) -> f32 {
    let scale = a.abs().max(b.abs()).max(1.0e-6);
    (a - b).abs() / scale
}

/// The drive's open 2 cm sheet against a uniform CFL split capped at eight.
///
/// The tolerances are the equality. A cap of eight with a different film,
/// or a formal pass while the volumes diverge, fails this test.
///
/// Both runs lay the 2 cm patch down flat each step (`set_near_field(false)`).
/// This test checks how one step splits its substeps. With the near field on,
/// sub-cell detail is carried from step to step, so any difference between the
/// two splits is carried too and the two runs drift apart over the drive. That
/// drift is not a pacing error; `tests/near_field.rs` covers the near field.
#[test]
fn one_fine_substep_matches_eight() {
    let mut prod = ThroughDrive::new();
    prod.fluid.set_near_field(false);
    let production_cap = prod.fluid.fine_sub_cap();
    let production_pace = prod.fluid.fine_pace();
    assert!(
        (production_cap - 8.0).abs() < 1.0e-3,
        "fine cap {production_cap} left eight"
    );
    let mut full = ThroughDrive::new();
    full.fluid.set_near_field(false);
    full.fluid.set_fine_pace(2);

    let (rest_prod, film_rest) = shore_snap(&prod, None);
    let (rest_full, _) = shore_snap(&full, Some(&film_rest));
    println!(
        "fine rest pace {production_pace} vs uniform 8: q {:.4} {:.4} tongues {} {} \
         span {:.4} {:.4} water {:.6} {:.6} slurry {:.6} {:.6} gap {:.6} {:.6} \
         l1 {:.6} max dh {:.5}",
        rest_prod.q,
        rest_full.q,
        rest_prod.tongues,
        rest_full.tongues,
        rest_prod.span,
        rest_full.span,
        rest_prod.water_m3,
        rest_full.water_m3,
        rest_prod.slurry_m3,
        rest_full.slurry_m3,
        rest_prod.gap,
        rest_full.gap,
        rest_full.l1_m3,
        rest_full.max_dh
    );
    assert!(
        rest_full.l1_m3 < 1.0e-6
            && rel_vol(rest_prod.water_m3, rest_full.water_m3) <= VOL_REL
            && rel_vol(rest_prod.slurry_m3, rest_full.slurry_m3) <= VOL_REL,
        "drain L1 {:.6} m3 water rel {:.4} slurry rel {:.4}",
        rest_full.l1_m3,
        rel_vol(rest_prod.water_m3, rest_full.water_m3),
        rel_vol(rest_prod.slurry_m3, rest_full.slurry_m3)
    );

    let dt = 1.0 / 30.0;
    let mut water_ms_p = 0.0f64;
    let mut water_ms_f = 0.0f64;
    let mut step_ms_p = 0.0f64;
    let mut step_ms_f = 0.0f64;
    let mut timed = 0.0f64;
    let mut peak_p = 0.0f32;
    let mut peak_f = 0.0f32;
    for _ in 0..48 {
        // Same soil and chassis at the start of each step. Only the fine
        // sheet integrator differs; divergent rut state was flipping tongues
        // on CI when only the vehicle was copied.
        let frame = snap_drive_frame(&prod);
        prod.step(dt);
        restore_drive_frame(&mut full, &frame);
        full.step(dt);
        peak_p = peak_p.max(peak_sheet_mps(&prod.fluid));
        peak_f = peak_f.max(peak_sheet_mps(&full.fluid));
        let z = prod.world.vehicle.pos.z;
        if !(-17.2..-12.2).contains(&z) {
            continue;
        }
        let a = prod.last_parts;
        let b = full.last_parts;
        water_ms_p += a.water_ms;
        water_ms_f += b.water_ms;
        step_ms_p += a.solver_ms();
        step_ms_f += b.solver_ms();
        timed += 1.0;
    }
    assert!(timed >= 8.0, "timed steps {timed}");
    let (run_prod, film_run) = shore_snap(&prod, None);
    let (run_full, _) = shore_snap(&full, Some(&film_run));
    let water_rel = rel_vol(run_prod.water_m3, run_full.water_m3);
    let slurry_rel = rel_vol(run_prod.slurry_m3, run_full.slurry_m3);
    let spray_rel = rel_vol(run_prod.spray_m3, run_full.spray_m3);
    let (cfl_c, n_c) = cfl_split(dt, peak_p, COARSE_CELL_M, production_cap);
    let (cfl_f, n_f) = cfl_split(dt, peak_p, FINE_CELL_M, production_cap);
    println!(
        "fine pace {production_pace} vs uniform 8 after pass: q {:.4} {:.4} tongues {} {} \
         span {:.4} {:.4} water {:.6} {:.6} slurry {:.6} {:.6} spray {:.6} {:.6} \
         gap {:.6} {:.6} l1 {:.6} max dh {:.5} peak {:.3} {:.3} \
         coarse cfl {:.3} nsub {} fine cfl {:.3} nsub {} \
         step ms {:.3} {:.3} water ms {:.3} {:.3} samples {timed:.0} \
         vol rel water {:.4} slurry {:.4} spray {:.4}",
        run_prod.q,
        run_full.q,
        run_prod.tongues,
        run_full.tongues,
        run_prod.span,
        run_full.span,
        run_prod.water_m3,
        run_full.water_m3,
        run_prod.slurry_m3,
        run_full.slurry_m3,
        run_prod.spray_m3,
        run_full.spray_m3,
        run_prod.gap,
        run_full.gap,
        run_full.l1_m3,
        run_full.max_dh,
        peak_p,
        peak_f,
        cfl_c,
        n_c,
        cfl_f,
        n_f,
        step_ms_p / timed,
        step_ms_f / timed,
        water_ms_p / timed,
        water_ms_f / timed,
        water_rel,
        slurry_rel,
        spray_rel
    );
    assert!(
        water_rel <= VOL_REL,
        "water volume rel {water_rel:.4} over {VOL_REL}"
    );
    assert!(
        slurry_rel <= VOL_REL,
        "slurry volume rel {slurry_rel:.4} over {VOL_REL}"
    );
    assert!(
        spray_rel <= VOL_REL,
        "spray volume rel {spray_rel:.4} over {VOL_REL}"
    );
    assert!(
        run_full.l1_m3 <= L1_DEPTH_M3,
        "L1 depth {:.6} m3 over {L1_DEPTH_M3} (max |dh| {:.5} m)",
        run_full.l1_m3,
        run_full.max_dh
    );
    assert!(
        run_full.max_dh <= MAX_DH_M,
        "max |dh| {:.5} m over {MAX_DH_M} (L1 {:.6} m3)",
        run_full.max_dh,
        run_full.l1_m3
    );
    // The tongue count is a discrete contour feature: a last-bit change in
    // the sheet (for example sqrt instead of hypot in #112) moves a 4 mm
    // tongue across the count while L1, max |dh|, and the books stay inside
    // their bounds above. Both runs must still be a fingered shore, and agree
    // to within two tongues.
    assert!(
        run_prod.tongues >= 4 && run_full.tongues >= 4,
        "tongues {} vs {}: a run went back to a lens",
        run_prod.tongues, run_full.tongues
    );
    assert!(
        run_prod.tongues.abs_diff(run_full.tongues) <= 2,
        "tongues {} vs {}",
        run_prod.tongues, run_full.tongues
    );
    assert!(
        (run_prod.gap - run_full.gap).abs() <= 1.5e-3,
        "ledger gap {:.6} vs {:.6}",
        run_prod.gap,
        run_full.gap
    );
}
