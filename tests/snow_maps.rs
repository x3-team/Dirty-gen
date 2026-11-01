//! Snow column on the render contract. The solver already stores height,
//! density, and compaction. This publishes that column and checks the mass
//! ledger at the thresholds the snow tests already use.
//!
//! Height is metres. Density is kg/m³. Each feature has its own bound.
//! The two channels are not one ceiling over the whole map.

use std::path::PathBuf;
use std::time::Instant;

use silt::render_contract::{
    publish, publish_with_snow, ContractFrame, RENDER_CONTRACT_VERSION, VALID_SNOW,
    VALID_TURBIDITY,
};
use silt::snow::{niva_snow_scene, SnowField, RHO_FRESH, RHO_PACKED, SNOW_EMPTY_M};
use silt::soil::SoilField;

const SOLVER_FRAME_BUDGET_MS: f64 = 7.0;
/// Rut floor, metres below the blanket. Not a density.
const RUT_HEIGHT_DROP_M: f32 = 0.10;
/// Berm shoulder, metres above the blanket. Not a density.
const BERM_HEIGHT_RISE_M: f32 = 0.03;
/// Packed rut, kg/m³ above fresh snow. Not a height.
const RUT_DENSITY_ABOVE_FRESH: f32 = 40.0;
/// Shelf, kg/m³ above fresh snow. Not a height.
const SHELF_DENSITY_ABOVE_FRESH: f32 = 15.0;

#[test]
fn snow_maps_keep_mass_and_match_the_column() {
    let wgsl = include_str!("../src/render_contract.wgsl");
    assert!(
        wgsl.contains(&format!(
            "const RENDER_CONTRACT_VERSION: u32 = {RENDER_CONTRACT_VERSION}u;"
        )),
        "WGSL version drifted from {RENDER_CONTRACT_VERSION}"
    );
    assert!(wgsl.contains("@group(0) @binding(14) var snow_tex"));
    assert!(wgsl.contains("var snow_tex: texture_2d<f32>"));
    assert!(wgsl.contains("if (mode == 16u) { return ramp(flow.a, 0.0, 1.0); }"));
    assert!(wgsl.contains("if (mode == 17u) { return ramp(snow.r, 0.0, 0.70); }"));
    assert!(wgsl.contains("if (mode == 18u) { return ramp(snow.g, 0.0, 320.0); }"));
    assert!(wgsl.contains("if (mode == 19u) { return ramp(snow.b, 0.0, 1.0); }"));
    assert!(RENDER_CONTRACT_VERSION >= 7);
    assert_eq!(VALID_TURBIDITY, 1 << 11);
    assert_eq!(VALID_SNOW, 1 << 14);
    assert_ne!(VALID_SNOW, VALID_TURBIDITY);

    let scene_t = Instant::now();
    let scene = niva_snow_scene();
    let scene_ms = scene_t.elapsed().as_secs_f64() * 1000.0;
    let report = scene.report;
    let snow = &scene.snow;
    let mass_err = (report.mass1_kg - report.mass0_kg).abs();
    let blanket = snow.blanket_m();
    let active = (0..snow.depth.len()).filter(|&i| snow.is_active(i)).count();
    assert!(active > 0, "the scene left the blanket untouched");
    assert!(
        active < snow.depth.len(),
        "snow walked every cell ({active})"
    );

    let soil = SoilField::flat(0.2);
    let mut frame = ContractFrame::new();
    for _ in 0..4 {
        publish_with_snow(
            &mut frame,
            &soil,
            &[],
            [0.0, 0.0],
            0.0,
            15,
            None,
            None,
            Some(snow),
        );
    }
    let frames = 24u32;
    let step_t = Instant::now();
    for _ in 0..frames {
        publish_with_snow(
            &mut frame,
            &soil,
            &[],
            [0.0, 0.0],
            0.0,
            15,
            None,
            None,
            Some(snow),
        );
    }
    let step_ms = step_t.elapsed().as_secs_f64() * 1000.0 / f64::from(frames);

    let mut bare = ContractFrame::new();
    publish(&mut bare, &soil, &[], [0.0, 0.0], 0.0, 0, None, None);
    let mut far = ContractFrame::new();
    publish_with_snow(
        &mut far,
        &soil,
        &[],
        [80.0, 80.0],
        0.0,
        0,
        None,
        None,
        Some(snow),
    );

    let (rut_h, rut_rho, rut_c) = column_at(&frame, snow, scene.straight_x + 0.74, scene.straight_z);
    let (shelf_h, shelf_rho_map, shelf_c) =
        column_at(&frame, snow, scene.straight_x, scene.straight_z);
    let (shelf_l, _, _) = column_at(&frame, snow, scene.straight_x - 0.30, scene.straight_z);
    let (shelf_r, _, _) = column_at(&frame, snow, scene.straight_x + 0.30, scene.straight_z);
    let mut berm_h = 0.0f32;
    let mut berm_rho = 0.0f32;
    let mut berm_c = 0.0f32;
    let mut s = scene.straight_x + 0.74 + 0.20;
    while s < scene.straight_x + 0.74 + 1.20 {
        let (h, rho, c) = column_at(&frame, snow, s, scene.straight_z);
        if h > berm_h {
            berm_h = h;
            berm_rho = rho;
            berm_c = c;
        }
        s += snow.cell * 0.5;
    }

    let mut rut_cells = 0u32;
    let mut berm_cells = 0u32;
    let mut shelf_cells = 0u32;
    let mut dense_cells = 0u32;
    let mut packed_cells = 0u32;
    let mut height_peak = 0.0f32;
    let mut density_peak = 0.0f32;
    let n = frame.header.window_res as usize;
    let floor = blanket - report.rut_drop_m;
    for idx in 0..n * n {
        let h = frame.snow[idx * 4];
        let rho = frame.snow[idx * 4 + 1];
        let c = frame.snow[idx * 4 + 2];
        height_peak = height_peak.max(h);
        density_peak = density_peak.max(rho);
        if h < blanket - 0.12 {
            rut_cells += 1;
        }
        if h > blanket + BERM_HEIGHT_RISE_M {
            berm_cells += 1;
        }
        if h < blanket - 0.18 && h > floor + 0.10 {
            shelf_cells += 1;
        }
        if rho > RHO_FRESH + RUT_DENSITY_ABOVE_FRESH {
            dense_cells += 1;
        }
        if c > 0.55 {
            packed_cells += 1;
        }
    }

    let line = format!(
        "SNOW_MAP version={RENDER_CONTRACT_VERSION} bit={VALID_SNOW} mass0={:.3} mass1={:.3} err={mass_err:.6} held={:.6} \
         rut_drop={:.4} berm_rise={:.4} shelf={:.4} shelf_flat={:.4} shelf_rho={:.1} \
         rut_h={rut_h:.4} berm_h={berm_h:.4} shelf_h={shelf_h:.4} \
         rut_rho={rut_rho:.1} berm_rho={berm_rho:.1} rut_c={rut_c:.3} berm_c={berm_c:.3} shelf_c={shelf_c:.3} \
         height_peak={height_peak:.4} density_peak={density_peak:.1} \
         cells rut={rut_cells} berm={berm_cells} shelf={shelf_cells} dense={dense_cells} packed={packed_cells} active={active} \
         SNOW_SCENE ms={scene_ms:.3} SNOW_MAP_STEP ms={step_ms:.3} budget={SOLVER_FRAME_BUDGET_MS:.0} frames={frames}",
        report.mass0_kg,
        report.mass1_kg,
        report.held_kg,
        report.rut_drop_m,
        report.berm_rise_m,
        report.shelf_m,
        report.shelf_flat_m,
        report.shelf_rho,
    );
    eprintln!("{line}");
    dump_maps(&frame, blanket, &line);

    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(frame.header.valid & VALID_SNOW, VALID_SNOW);
    assert_eq!(frame.header.valid & VALID_TURBIDITY, 0);
    assert_eq!(frame.header.window_res, 200);
    assert!((frame.header.cell_m - 0.20).abs() < 1.0e-6);
    assert!(frame.snow.iter().all(|v| v.is_finite()), "NaN in the snow map");
    assert!(frame.snow.chunks(4).all(|t| t[3] == 0.0), "snow A is not 0");
    assert!(frame.snow.chunks(4).all(|t| (0.0..=1.0).contains(&t[2])));

    assert!(
        mass_err / report.mass0_kg < 2.0e-5,
        "mass {} -> {} err {mass_err}",
        report.mass0_kg,
        report.mass1_kg
    );
    assert!(report.held_kg.abs() < 0.2, "held {}", report.held_kg);
    assert!(
        (report.mass1_kg - report.mass0_kg).abs() < 0.05,
        "mass {} -> {}",
        report.mass0_kg,
        report.mass1_kg
    );
    let shrunk = report.geom0_m3 - report.geom1_m3;
    assert!(
        shrunk > 0.35,
        "compaction should shrink geometric volume {} -> {} ({shrunk})",
        report.geom0_m3,
        report.geom1_m3
    );
    let solid_err = (report.solid1_m3 - report.solid0_m3).abs();
    assert!(solid_err < 1.0e-3, "solid volume {solid_err}");
    assert!(
        (report.rut_drop_m - 0.4376).abs() < 0.001,
        "rut drop {}",
        report.rut_drop_m
    );
    assert!(
        (report.berm_rise_m - 0.1863).abs() < 0.001,
        "berm rise {}",
        report.berm_rise_m
    );
    assert!(
        report.rut_rho > RHO_FRESH + RUT_DENSITY_ABOVE_FRESH,
        "rut density {}",
        report.rut_rho
    );
    assert!(
        report.berm_rho + 30.0 < report.rut_rho,
        "berm density {} floor {}",
        report.berm_rho,
        report.rut_rho
    );
    assert!(
        report.shelf_rho > RHO_FRESH + SHELF_DENSITY_ABOVE_FRESH,
        "shelf density {}",
        report.shelf_rho
    );

    assert_column_matches(&frame, snow);
    assert!(
        rut_h < blanket - RUT_HEIGHT_DROP_M,
        "right rut height {rut_h} m, blanket {blanket} m"
    );
    assert!(
        rut_rho > RHO_FRESH + RUT_DENSITY_ABOVE_FRESH,
        "right rut density {rut_rho} kg/m3"
    );
    assert!(
        rut_c > RUT_DENSITY_ABOVE_FRESH / (RHO_PACKED - RHO_FRESH),
        "map rut compaction {rut_c}"
    );
    assert!(
        berm_h > blanket + BERM_HEIGHT_RISE_M,
        "outer berm height {berm_h} m, blanket {blanket} m"
    );
    assert!(
        berm_rho + 30.0 < rut_rho,
        "outer berm density {berm_rho} kg/m3, rut density {rut_rho} kg/m3"
    );
    assert!(
        density_peak > height_peak + 50.0,
        "height {height_peak} m and density {density_peak} kg/m3 share a ceiling"
    );
    for (name, h) in [("mid", shelf_h), ("left", shelf_l), ("right", shelf_r)] {
        assert!(
            h < blanket - 0.18,
            "shelf {name} {h} is still a crest, blanket {blanket}"
        );
        assert!(
            h > floor + 0.10,
            "shelf {name} {h} fell into the rut floor {floor}"
        );
    }
    let hi = shelf_h.max(shelf_l).max(shelf_r);
    let lo = shelf_h.min(shelf_l).min(shelf_r);
    assert!(hi - lo < 0.05, "shelf not flat {lo}..{hi}");
    assert!(
        shelf_rho_map > RHO_FRESH + SHELF_DENSITY_ABOVE_FRESH,
        "map shelf density {shelf_rho_map} kg/m3"
    );

    assert!(rut_cells > 0, "no rut cells in the height map");
    assert!(berm_cells > 0, "no berm cells in the height map");
    assert!(shelf_cells > 0, "no underbody strip in the height map");
    assert!(dense_cells > 0, "no packed cells in the density map");
    assert!(packed_cells > 0, "no packed cells in the compaction map");

    assert_eq!(bare.header.valid & VALID_SNOW, 0);
    assert!(bare.snow.iter().all(|v| *v == 0.0));
    assert_eq!(far.header.valid & VALID_SNOW, VALID_SNOW);
    assert!(
        far.snow.iter().all(|v| *v == 0.0),
        "a window off the snow square published a blanket"
    );

    #[cfg(not(debug_assertions))]
    assert!(
        step_ms <= SOLVER_FRAME_BUDGET_MS,
        "SNOW_MAP_STEP {step_ms:.3} ms over {SOLVER_FRAME_BUDGET_MS}"
    );
}

fn column_at(frame: &ContractFrame, snow: &SnowField, x: f32, z: f32) -> (f32, f32, f32) {
    let (idx, cx, cz) = texel_of(frame, x, z).expect("probe outside the window");
    let got = &frame.snow[idx * 4..idx * 4 + 4];
    let expect = expected(snow, cx, cz);
    assert!(
        (got[0] - expect[0]).abs() < 1.0e-5
            && (got[1] - expect[1]).abs() < 1.0e-4
            && (got[2] - expect[2]).abs() < 1.0e-5
            && got[3] == 0.0,
        "map at ({cx:.3},{cz:.3}) {got:?} solver {expect:?}"
    );
    (got[0], got[1], got[2])
}

fn assert_column_matches(frame: &ContractFrame, snow: &SnowField) {
    let n = frame.header.window_res as usize;
    let cell = frame.header.cell_m;
    for j in 0..n {
        for i in 0..n {
            let x = frame.header.origin_x + (i as f32 + 0.5) * cell;
            let z = frame.header.origin_z + (j as f32 + 0.5) * cell;
            let idx = j * n + i;
            let got = &frame.snow[idx * 4..idx * 4 + 4];
            let expect = expected(snow, x, z);
            assert!(
                (got[0] - expect[0]).abs() < 1.0e-5
                    && (got[1] - expect[1]).abs() < 1.0e-4
                    && (got[2] - expect[2]).abs() < 1.0e-5
                    && got[3] == 0.0,
                "texel ({i},{j}) {got:?} solver {expect:?}"
            );
        }
    }
}

fn expected(snow: &SnowField, x: f32, z: f32) -> [f32; 4] {
    let Some(src) = snow.idx_at(x, z) else {
        return [0.0; 4];
    };
    let height = snow.depth[src];
    if !height.is_finite() || height < SNOW_EMPTY_M {
        return [0.0; 4];
    }
    [
        height,
        finite0(snow.density[src]),
        finite0(snow.compaction(src)),
        0.0,
    ]
}

fn finite0(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn texel_of(frame: &ContractFrame, x: f32, z: f32) -> Option<(usize, f32, f32)> {
    let cell = frame.header.cell_m;
    let i = ((x - frame.header.origin_x) / cell).floor() as isize;
    let j = ((z - frame.header.origin_z) / cell).floor() as isize;
    let n = frame.header.window_res as isize;
    if i < 0 || j < 0 || i >= n || j >= n {
        return None;
    }
    let cx = frame.header.origin_x + (i as f32 + 0.5) * cell;
    let cz = frame.header.origin_z + (j as f32 + 0.5) * cell;
    Some((j as usize * n as usize + i as usize, cx, cz))
}

fn dump_maps(frame: &ContractFrame, blanket: f32, line: &str) {
    let dir = dump_dir();
    if let Err(err) = std::fs::create_dir_all(&dir) {
        eprintln!("snow map dump skipped: {err}");
        return;
    }
    let n = frame.header.window_res;
    let height = plane(frame, 0);
    let density = plane(frame, 1);
    let compaction = plane(frame, 2);
    write_gray(&dir.join("height.png"), n, n, &height, 0.0, 0.80);
    write_gray(&dir.join("density.png"), n, n, &density, 0.0, 320.0);
    write_gray(&dir.join("compaction.png"), n, n, &compaction, 0.0, 1.0);
    write_png(&dir.join("relief.png"), n, n, &relief(frame, blanket));
    let note = format!(
        "{line}\n\
         PNG row 0 is north. height.png is thickness 0..0.80 m.\n\
         density.png is kg/m³ 0..320. compaction.png is 0..1.\n\
         relief.png: R berm above the blanket, G compaction, B rut drop.\n"
    );
    std::fs::write(dir.join("readings.txt"), note).expect("readings");
}

fn dump_dir() -> PathBuf {
    if let Ok(over) = std::env::var("SNOW_MAPS_DIR") {
        return PathBuf::from(over);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/snow-maps")
}

fn plane(frame: &ContractFrame, channel: usize) -> Vec<f32> {
    let n = frame.header.window_res as usize;
    let mut out = Vec::with_capacity(n * n);
    for j in (0..n).rev() {
        for i in 0..n {
            out.push(frame.snow[(j * n + i) * 4 + channel]);
        }
    }
    out
}

fn relief(frame: &ContractFrame, blanket: f32) -> Vec<u8> {
    let n = frame.header.window_res as usize;
    let mut rgb = Vec::with_capacity(n * n * 3);
    for j in (0..n).rev() {
        for i in 0..n {
            let o = (j * n + i) * 4;
            let h = frame.snow[o];
            let c = frame.snow[o + 2];
            let drop = ((blanket - h) / 0.50).clamp(0.0, 1.0);
            let rise = ((h - blanket) / 0.25).clamp(0.0, 1.0);
            rgb.push((rise * 255.0).round() as u8);
            rgb.push((c.clamp(0.0, 1.0) * 255.0).round() as u8);
            rgb.push((drop * 255.0).round() as u8);
        }
    }
    rgb
}

fn write_gray(path: &std::path::Path, w: u32, h: u32, samples: &[f32], lo: f32, hi: f32) {
    let span = (hi - lo).max(1.0e-6);
    let mut rgb = Vec::with_capacity(samples.len() * 3);
    for s in samples {
        let t = ((*s - lo) / span).clamp(0.0, 1.0);
        let g = (t * 255.0).round() as u8;
        rgb.extend_from_slice(&[g, g, g]);
    }
    write_png(path, w, h, &rgb);
}

fn write_png(path: &std::path::Path, width: u32, height: u32, rgb: &[u8]) {
    image::save_buffer(path, rgb, width, height, image::ExtendedColorType::Rgb8).expect("png");
}
