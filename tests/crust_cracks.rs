//! Drying on the Niva pad. Binding 7 carries crust thickness and crack opening
//! in metres. The skin is not solid mass and a fissure is not a cut.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use silt::fluid_core::FluidCore;
use silt::render_contract::{publish, ContractFrame, RENDER_CONTRACT_VERSION, VALID_CRUST};
use silt::soil::{CRACK_SHADER_METRES, CRUST_CAP_M, CRUST_SHADER_METRES};
use silt::World;

const DUMP: &str = "/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media/crust-cracks";

#[test]
fn drying_publishes_crust_and_cracks_without_moving_mass() {
    let mut world = World::flat(0.45);
    let stats = world.drying_crust_frame();
    let solid_before = world.soil.mix_volumes();
    let book = world.soil.crust_book();
    for idx in 0..world.soil.crust.len() {
        let thickness = world.soil.crust[idx];
        let opening = world.soil.crack[idx];
        assert!(
            thickness <= CRUST_CAP_M,
            "thickness {thickness} over {CRUST_CAP_M}"
        );
        assert!(
            opening <= thickness,
            "opening {opening} over thickness {thickness}"
        );
    }

    let center = [world.vehicle.pos.x, world.vehicle.pos.z];
    let mut frame = ContractFrame::new();
    publish(
        &mut frame,
        &world.soil,
        &[],
        center,
        world.time,
        7,
        None,
        None,
    );
    let solid_after = world.soil.mix_volumes();

    let fluid = FluidCore::new(center);
    let mut sheet = ContractFrame::new();
    publish(
        &mut sheet,
        &world.soil,
        &[],
        center,
        world.time,
        8,
        Some(&fluid),
        None,
    );

    let report = map_report(&world.soil, &frame);
    let sheet_report = map_report(&world.soil, &sheet);
    let (win_t, map_t, win_o, map_o) = window_crust(&world.soil, &frame);
    let (sheet_win_t, sheet_map_t, sheet_win_o, sheet_map_o) = window_crust(&world.soil, &sheet);
    let map_gap = (win_t - map_t).abs();
    let open_gap = (win_o - map_o).abs();
    let sheet_gap = (sheet_win_t - sheet_map_t).abs();
    let sheet_open_gap = (sheet_win_o - sheet_map_o).abs();

    eprintln!(
        "CRUST_BOOK version={} thickness_m3={:.6} opening_m3={:.6} grown_m3={:.6} \
         lost_m3={:.6} gap={:.8} window_m3={:.6} map_m3={:.6} map_gap={:.8} \
         open_window_m3={:.6} open_map_m3={:.6} open_gap={:.8} sheet_gap={:.8}",
        frame.header.version,
        book.thickness_m3,
        book.opening_m3,
        book.grown_m3,
        book.lost_m3,
        book.gap(),
        win_t,
        map_t,
        map_gap,
        win_o,
        map_o,
        open_gap,
        sheet_gap
    );
    eprintln!(
        "CRUST_MASS solid_m3={:.6} cut_m3={:.4} placed_m3={:.4} water_m3={:.6} slurry_m3={:.6} \
         bed_gap={:.8} placed_gap={:.8} cloud_gap={:.8} slurry_gap={:.8}",
        solid_before.soil,
        stats.cut,
        stats.placed,
        solid_before.water,
        solid_before.slurry,
        stats.bed_gap,
        stats.placed_gap,
        stats.cloud_gap,
        stats.slurry_gap
    );
    eprintln!(
        "CRUST_MAP version={} thickness_max={:.5} opening_max={:.5} \
         thick_cells={} open_cells={} copied={} ground_err={:.8} \
         east_thick={:.5} east_open={:.5} west_thick={:.5} sand_thick={:.6} silt_thick={:.6} \
         lane_thick={:.5} lane_open={:.5} lane_open_max={:.5}",
        frame.header.version,
        report.thickness_max,
        report.opening_max,
        report.thick_cells,
        report.open_cells,
        report.copied,
        report.ground_err,
        report.east_thick,
        report.east_open,
        report.west_thick,
        report.sand_thick,
        report.silt_thick,
        stats.crust_m,
        stats.crack_m,
        stats.crack_max
    );
    eprintln!(
        "CRUST_SHEET thickness_max={:.5} opening_max={:.5} mismatch={}",
        sheet_report.thickness_max, sheet_report.opening_max, sheet_report.mismatch
    );

    dump_maps(
        &frame,
        &format!(
            "version {RENDER_CONTRACT_VERSION}\n\
         thickness_m3 {:.6}\nopening_m3 {:.6}\ngrown_m3 {:.6}\nlost_m3 {:.6}\n\
         gap {:.8}\nwindow_m3 {:.6}\nmap_m3 {:.6}\nmap_gap {:.8}\n\
         open_window_m3 {:.6}\nopen_map_m3 {:.6}\nopen_gap {:.8}\n\
         solid_m3 {:.6}\ncut_m3 {:.4}\nplaced_m3 {:.4}\nwater_m3 {:.6}\nslurry_m3 {:.6}\n\
         bed_gap {:.8}\nplaced_gap {:.8}\ncloud_gap {:.8}\nslurry_gap {:.8}\n\
         thickness_max {:.5}\nopening_max {:.5}\n\
         thick_cells {}\nopen_cells {}\ncopied {}\nground_err {:.8}\n\
         east_thick {:.5}\neast_open {:.5}\nwest_thick {:.5}\n\
         sand_thick {:.6}\nsilt_thick {:.6}\n\
         lane_thick {:.5}\nlane_open {:.5}\nlane_open_max {:.5}\n\
         sheet_thickness_max {:.5}\nsheet_opening_max {:.5}\n",
            book.thickness_m3,
            book.opening_m3,
            book.grown_m3,
            book.lost_m3,
            book.gap(),
            win_t,
            map_t,
            map_gap,
            win_o,
            map_o,
            open_gap,
            solid_before.soil,
            stats.cut,
            stats.placed,
            solid_before.water,
            solid_before.slurry,
            stats.bed_gap,
            stats.placed_gap,
            stats.cloud_gap,
            stats.slurry_gap,
            report.thickness_max,
            report.opening_max,
            report.thick_cells,
            report.open_cells,
            report.copied,
            report.ground_err,
            report.east_thick,
            report.east_open,
            report.west_thick,
            report.sand_thick,
            report.silt_thick,
            stats.crust_m,
            stats.crack_m,
            stats.crack_max,
            sheet_report.thickness_max,
            sheet_report.opening_max,
        ),
    );

    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(frame.header.valid & VALID_CRUST, VALID_CRUST);
    assert!(
        book.thickness_m3 > 1.0,
        "crust book stayed empty, {:.6}",
        book.thickness_m3
    );
    assert!(
        book.opening_m3 > 0.05,
        "opening book stayed empty, {:.6}",
        book.opening_m3
    );
    assert!(
        book.gap() < 1.0e-3,
        "crust book gap {:.6} grown {:.6} lost {:.6} still {:.6}",
        book.gap(),
        book.grown_m3,
        book.lost_m3,
        book.thickness_m3
    );
    assert!(win_t > 1.0, "window crust book stayed empty, {win_t:.6}");
    assert!(map_gap < 1.0e-3, "thickness map gap {map_gap:.6}");
    assert!(open_gap < 1.0e-3, "opening map gap {open_gap:.6}");
    assert!(sheet_gap < 1.0e-3, "sheet thickness gap {sheet_gap:.6}");
    assert!(
        sheet_open_gap < 1.0e-3,
        "sheet opening gap {sheet_open_gap:.6}"
    );
    assert!(
        (solid_after.soil - solid_before.soil).abs() < 1.0e-6,
        "publish moved solid mass"
    );
    assert!(
        (f64::from(solid_before.soil) - book.thickness_m3).abs() > 1.0,
        "the skin was folded into the solid ledger, solid {:.6} skin {:.6}",
        solid_before.soil,
        book.thickness_m3
    );
    assert!(
        stats.bed_gap < 1.0e-5,
        "drying moved the bed, {}",
        stats.bed_gap
    );
    assert!(
        stats.placed_gap < 1.0e-5,
        "drying moved the pile, {}",
        stats.placed_gap
    );
    assert!(
        stats.cloud_gap < 1.0e-5,
        "drying moved the cloud, {}",
        stats.cloud_gap
    );
    assert!(
        stats.slurry_gap < 1.0e-5,
        "drying moved the slurry, {}",
        stats.slurry_gap
    );
    assert_eq!(report.mismatch, 0, "map diverged from the soil arrays");
    assert!(report.ground_err < 1.0e-5, "ground absorbed the skin");
    assert!(
        report.thickness_max > 0.015,
        "thickness stayed a zero, {:.5}",
        report.thickness_max
    );
    assert!(
        report.opening_max > 0.004,
        "opening stayed a zero, {:.5}",
        report.opening_max
    );
    assert!(report.thick_cells > 20, "too few skinned cells");
    assert!(report.open_cells > 8, "too few fissures");
    assert!(
        report.copied * 4 < report.thick_cells,
        "opening copied thickness, {} of {}",
        report.copied,
        report.thick_cells
    );
    assert!(
        report.east_thick > 0.015,
        "east track thickness {:.5}",
        report.east_thick
    );
    assert!(
        report.east_open > 0.003,
        "east track opening {:.5}",
        report.east_open
    );
    assert!(report.west_cells > 0, "the wet track missed the window");
    assert!(
        report.west_thick < 1.0e-4,
        "the wet track kept a skin, {:.5}",
        report.west_thick
    );
    assert!(report.sand_cells > 0, "the sand patch missed the window");
    assert!(
        report.sand_thick < 1.0e-5,
        "sand crusted, {:.6}",
        report.sand_thick
    );
    assert!(report.silt_cells > 0, "the silt patch missed the window");
    assert!(
        report.silt_thick < 1.0e-5,
        "silt crusted, {:.6}",
        report.silt_thick
    );
    assert_eq!(sheet.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(sheet_report.mismatch, 0);
    assert!(sheet_report.thickness_max > 0.015);
    assert!(sheet_report.opening_max > 0.004);
    assert!(frame.crust.iter().all(|v| v.is_finite()));
    assert!(sheet.crust.iter().all(|v| v.is_finite()));
}

struct MapReport {
    mismatch: u32,
    ground_err: f32,
    thickness_max: f32,
    opening_max: f32,
    thick_cells: u32,
    open_cells: u32,
    copied: u32,
    east_thick: f32,
    east_open: f32,
    west_thick: f32,
    west_cells: u32,
    sand_thick: f32,
    sand_cells: u32,
    silt_thick: f32,
    silt_cells: u32,
}

/// Soil cells the window hits, each once, against the published sample.
/// Both sides use the soil cell area, so a copied channel balances.
fn window_crust(soil: &silt::soil::SoilField, frame: &ContractFrame) -> (f64, f64, f64, f64) {
    let res = frame.header.window_res as usize;
    let area = f64::from(soil.cell) * f64::from(soil.cell);
    let mut seen = vec![false; soil.crust.len()];
    let mut soil_t = 0.0f64;
    let mut soil_o = 0.0f64;
    let mut map_t = 0.0f64;
    let mut map_o = 0.0f64;
    for j in 0..res {
        for i in 0..res {
            let x = frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m;
            let z = frame.header.origin_z + (j as f32 + 0.5) * frame.header.cell_m;
            let cidx = j * res + i;
            let thickness = frame.crust[cidx * 2];
            let opening = frame.crust[cidx * 2 + 1];
            assert!(
                thickness <= CRUST_CAP_M,
                "map thickness {thickness} over {CRUST_CAP_M}"
            );
            assert!(
                opening <= thickness,
                "map opening {opening} over thickness {thickness}"
            );
            let Some(sidx) = soil.idx_at(x, z) else {
                continue;
            };
            if seen[sidx] {
                continue;
            }
            seen[sidx] = true;
            let st = soil.crust[sidx];
            let so = soil.crack[sidx];
            let st = if st.is_finite() { st.max(0.0) } else { 0.0 };
            let so = if so.is_finite() { so.max(0.0) } else { 0.0 };
            let mt = if thickness.is_finite() {
                thickness.max(0.0)
            } else {
                0.0
            };
            let mo = if opening.is_finite() {
                opening.max(0.0)
            } else {
                0.0
            };
            soil_t += f64::from(st);
            soil_o += f64::from(so);
            map_t += f64::from(mt);
            map_o += f64::from(mo);
        }
    }
    (soil_t * area, map_t * area, soil_o * area, map_o * area)
}

fn map_report(soil: &silt::soil::SoilField, frame: &ContractFrame) -> MapReport {
    let res = frame.header.window_res as usize;
    let mut report = MapReport {
        mismatch: 0,
        ground_err: 0.0,
        thickness_max: 0.0,
        opening_max: 0.0,
        thick_cells: 0,
        open_cells: 0,
        copied: 0,
        east_thick: 0.0,
        east_open: 0.0,
        west_thick: 0.0,
        west_cells: 0,
        sand_thick: 0.0,
        sand_cells: 0,
        silt_thick: 0.0,
        silt_cells: 0,
    };
    for j in 0..res {
        for i in 0..res {
            let x = frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m;
            let z = frame.header.origin_z + (j as f32 + 0.5) * frame.header.cell_m;
            let cidx = j * res + i;
            let thickness = frame.crust[cidx * 2];
            let opening = frame.crust[cidx * 2 + 1];
            let Some(sidx) = soil.idx_at(x, z) else {
                if thickness != 0.0 || opening != 0.0 {
                    report.mismatch += 1;
                }
                continue;
            };
            if (thickness - soil.crust[sidx].max(0.0)).abs() > 1.0e-5
                || (opening - soil.crack[sidx].max(0.0)).abs() > 1.0e-5
            {
                report.mismatch += 1;
            }
            let expect = soil.bed[sidx] + soil.placed[sidx];
            report.ground_err = report
                .ground_err
                .max((frame.ground_rut[cidx * 4] - expect).abs());
            report.thickness_max = report.thickness_max.max(thickness);
            report.opening_max = report.opening_max.max(opening);
            if thickness > 0.01 {
                report.thick_cells += 1;
                if (thickness - opening).abs() < 1.0e-4 {
                    report.copied += 1;
                }
            }
            if opening > 0.003 {
                report.open_cells += 1;
            }
            if x > 0.5 && x < 2.2 && z.abs() < 2.3 {
                report.east_thick = report.east_thick.max(thickness);
                report.east_open = report.east_open.max(opening);
            }
            if (x + 0.96).abs() < 0.25 && z.abs() < 0.25 {
                report.west_thick = report.west_thick.max(thickness);
                report.west_cells += 1;
            }
            if (x - 6.2).abs() < 0.2 && (z - 6.0).abs() < 0.2 {
                report.sand_thick = report.sand_thick.max(thickness);
                report.sand_cells += 1;
            }
            if (x - 7.2).abs() < 0.2 && (z - 6.0).abs() < 0.2 {
                report.silt_thick = report.silt_thick.max(thickness);
                report.silt_cells += 1;
            }
        }
    }
    report
}

fn dump_maps(frame: &ContractFrame, numbers: &str) {
    let dir = Path::new(DUMP);
    if dir.parent().is_none_or(|parent| !parent.exists()) {
        return;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let res = frame.header.window_res as usize;
    let mut field = vec![0u8; res * res * 4];
    let mut thickness = vec![0u8; res * res * 4];
    let mut opening = vec![0u8; res * res * 4];
    for j in 0..res {
        for i in 0..res {
            let src = j * res + i;
            let dst = (res - 1 - j) * res + i;
            let thick = (frame.crust[src * 2] / CRUST_SHADER_METRES).clamp(0.0, 1.0);
            let open = (frame.crust[src * 2 + 1] / CRACK_SHADER_METRES).clamp(0.0, 1.0);
            let tr = (thick * 255.0).round() as u8;
            let og = (open * 255.0).round() as u8;
            let base = dst * 4;
            field[base] = tr;
            field[base + 1] = og;
            field[base + 2] = 0;
            field[base + 3] = 255;
            thickness[base] = tr;
            thickness[base + 1] = tr;
            thickness[base + 2] = tr;
            thickness[base + 3] = 255;
            opening[base] = og;
            opening[base + 1] = og;
            opening[base + 2] = og;
            opening[base + 3] = 255;
        }
    }
    let side = res as u32;
    write_png(&dir.join("field.png"), side, side, &field);
    write_png(&dir.join("thickness.png"), side, side, &thickness);
    write_png(&dir.join("opening.png"), side, side, &opening);
    let _ = File::create(dir.join("numbers.txt"))
        .and_then(|mut file| file.write_all(numbers.as_bytes()));
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    let Ok(file) = File::create(path) else {
        return;
    };
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    if let Ok(mut writer) = encoder.write_header() {
        let _ = writer.write_image_data(rgba);
    }
}
