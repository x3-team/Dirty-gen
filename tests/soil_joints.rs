//! Material joints. Mud, sand, gravel, and grass exchange share and pore
//! moisture across the boundary. The amount of each material stays put.
//!
//! `cargo test --release --test soil_joints -- --nocapture`

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use silt::render_contract::{publish, ContractFrame, RENDER_CONTRACT_VERSION, VALID_MIX};
use silt::soil::{SoilField, SurfaceKind, MIX_CHANNELS, MIX_GRASS, MIX_GRAVEL, MIX_MUD, MIX_SAND};

const DUMP_DIR: &str = "/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media/soil-joints";

const MUD_WET: f32 = 0.92;
const SAND_WET: f32 = 0.30;
const GRAVEL_WET: f32 = 0.08;
const GRASS_WET: f32 = 0.42;

#[test]
fn joint_shares_keep_mass_and_blend() {
    assert_eq!(RENDER_CONTRACT_VERSION, 6);

    let mut soil = SoilField::flat(0.5);
    paint_strips(&mut soil);
    let before_share = kind_shares(&soil);
    let before_wet = sum_f32(&soil.wetness);
    let before_cell = cell_sums_from_kind(&soil);

    soil.blend_material_joints();

    let after_share = channel_shares(&soil.mix);
    let after_wet = sum_f32(&soil.wetness);
    let area = (soil.cell * soil.cell) as f64;
    eprintln!(
        "SOIL_JOINTS version={RENDER_CONTRACT_VERSION} cells={} cell_m={:.5}",
        soil.res * soil.res,
        soil.cell
    );
    eprintln!(
        "SOIL_JOINTS mass_m2 mud={:.6} sand={:.6} gravel={:.6} grass={:.6} wet={:.6}",
        after_share[MIX_MUD] * area,
        after_share[MIX_SAND] * area,
        after_share[MIX_GRAVEL] * area,
        after_share[MIX_GRASS] * area,
        after_wet * area
    );
    for (name, before, after) in [
        ("mud", before_share[MIX_MUD], after_share[MIX_MUD]),
        ("sand", before_share[MIX_SAND], after_share[MIX_SAND]),
        ("gravel", before_share[MIX_GRAVEL], after_share[MIX_GRAVEL]),
        ("grass", before_share[MIX_GRASS], after_share[MIX_GRASS]),
    ] {
        let err = (after - before).abs();
        eprintln!("SOIL_JOINTS share {name} before={before:.6} after={after:.6} err={err:.3e}");
        assert!(
            err < 1.0e-3,
            "{name} mass drifted before {before:.6} after {after:.6}"
        );
    }
    let wet_err = (after_wet - before_wet).abs();
    eprintln!("SOIL_JOINTS wet before={before_wet:.6} after={after_wet:.6} err={wet_err:.3e}");
    assert!(
        wet_err < 1.0e-3,
        "pore moisture drifted before {before_wet:.6} after {after_wet:.6}"
    );

    for idx in 0..soil.kind.len() {
        let sum = (0..MIX_CHANNELS)
            .map(|c| soil.mix[idx * MIX_CHANNELS + c] as f64)
            .sum::<f64>();
        let expect = before_cell[idx];
        assert!(
            (sum - expect).abs() < 1.0e-4,
            "cell {idx} share sum {sum:.6} expected {expect:.1}"
        );
        for c in 0..MIX_CHANNELS {
            let v = soil.mix[idx * MIX_CHANNELS + c];
            assert!(
                (-1.0e-5..=1.0 + 1.0e-5).contains(&v),
                "channel {c} at {idx} is {v}"
            );
        }
    }

    let (mud_l, mud_r) = straddle(&soil, -4.0, 0.0);
    let (sand_l, sand_r) = straddle(&soil, 0.0, 0.0);
    let (gravel_l, gravel_r) = straddle(&soil, 4.0, 0.0);
    assert_mixed(&soil, mud_l, mud_r, MIX_MUD, MIX_SAND, "mud-sand");
    assert_mixed(&soil, sand_l, sand_r, MIX_SAND, MIX_GRAVEL, "sand-gravel");
    assert_mixed(
        &soil,
        gravel_l,
        gravel_r,
        MIX_GRAVEL,
        MIX_GRASS,
        "gravel-grass",
    );

    let mud_heart = nearest(&soil, -6.0, 0.0);
    let sand_heart = nearest(&soil, -2.0, 0.0);
    let gravel_heart = nearest(&soil, 2.0, 0.0);
    let grass_heart = nearest(&soil, 6.0, 0.0);
    assert!(
        share(&soil, mud_heart, MIX_MUD) > 0.98,
        "mud interior mixed"
    );
    assert!(
        share(&soil, sand_heart, MIX_SAND) > 0.98,
        "sand interior mixed"
    );
    assert!(
        share(&soil, gravel_heart, MIX_GRAVEL) > 0.98,
        "gravel interior mixed"
    );
    assert!(
        share(&soil, grass_heart, MIX_GRASS) > 0.98,
        "grass interior mixed"
    );
    assert!((soil.wetness[mud_heart] - MUD_WET).abs() < 0.01);
    assert!((soil.wetness[grass_heart] - GRASS_WET).abs() < 0.01);

    let seam_wet = 0.5 * (soil.wetness[mud_l] + soil.wetness[mud_r]);
    eprintln!(
        "SOIL_JOINTS seam mud/sand shares {:.4}/{:.4} wet {:.4} and {:.4}",
        share(&soil, mud_l, MIX_MUD),
        share(&soil, mud_r, MIX_MUD),
        soil.wetness[mud_l],
        soil.wetness[mud_r]
    );
    assert!(
        seam_wet > SAND_WET + 0.05 && seam_wet < MUD_WET - 0.05,
        "moisture did not sit between the pads: {seam_wet:.3}"
    );
    assert!(soil.wetness[mud_l] < MUD_WET - 0.05);
    assert!(soil.wetness[mud_r] > SAND_WET + 0.05);

    let pure = terramech_gap();
    eprintln!(
        "SOIL_JOINTS terramech k_eq mud={:.1} joint={:.1} sand={:.1}",
        pure.0, pure.1, pure.2
    );
    let lo = pure.0.min(pure.2);
    let hi = pure.0.max(pure.2);
    assert!(
        pure.1 > lo && pure.1 < hi,
        "joint modulus {:.1} should sit between mud {:.1} and sand {:.1}",
        pure.1,
        pure.0,
        pure.2
    );

    let mut frame = ContractFrame::new();
    publish(&mut frame, &soil, &[], [0.0, 0.0], 0.0, 15, None, None);
    assert_eq!(frame.header.version, RENDER_CONTRACT_VERSION);
    assert_eq!(frame.header.valid & VALID_MIX, VALID_MIX);
    assert_eq!(frame.soil_mix.len(), frame.moisture.len() * 4);
    let mut checked = 0u32;
    let res = frame.header.window_res as usize;
    for j in 0..res {
        for i in 0..res {
            let x = frame.header.origin_x + (i as f32 + 0.5) * frame.header.cell_m;
            let z = frame.header.origin_z + (j as f32 + 0.5) * frame.header.cell_m;
            let Some(sidx) = soil.idx_at(x, z) else {
                continue;
            };
            let idx = j * res + i;
            for c in 0..4 {
                let got = frame.soil_mix[idx * 4 + c];
                let expect = soil.mix[sidx * 4 + c];
                assert!(
                    (got - expect).abs() < 1.0e-5,
                    "contract mix {c} at ({x:.2},{z:.2}) {got} soil {expect}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > 1000, "contract window missed the field");
    eprintln!("SOIL_JOINTS contract_texels={checked}");

    dump_maps(&soil, &frame);
}

fn paint_strips(field: &mut SoilField) {
    let half = field.size * 0.5;
    for j in 0..field.res {
        for i in 0..field.res {
            let x = (i as f32 + 0.5) * field.cell - half;
            let z = (j as f32 + 0.5) * field.cell - half;
            if z.abs() >= 8.0 || x.abs() >= 8.0 {
                continue;
            }
            let idx = j * field.res + i;
            let (kind, wet) = if x < -4.0 {
                (SurfaceKind::WetMud, MUD_WET)
            } else if x < 0.0 {
                (SurfaceKind::Sand, SAND_WET)
            } else if x < 4.0 {
                (SurfaceKind::Gravel, GRAVEL_WET)
            } else {
                (SurfaceKind::Grass, GRASS_WET)
            };
            field.kind[idx] = kind as u8;
            field.wetness[idx] = wet;
            field.base_wetness[idx] = wet;
        }
    }
}

fn kind_shares(field: &SoilField) -> [f64; 4] {
    let mut shares = [0.0; 4];
    for kind in &field.kind {
        if let Some(channel) = silt::soil::mix_channel(SurfaceKind::from_u8(*kind)) {
            shares[channel] += 1.0;
        }
    }
    shares
}

fn channel_shares(mix: &[f32]) -> [f64; 4] {
    let mut shares = [0.0; 4];
    for (i, v) in mix.iter().enumerate() {
        shares[i % 4] += *v as f64;
    }
    shares
}

fn cell_sums_from_kind(field: &SoilField) -> Vec<f64> {
    field
        .kind
        .iter()
        .map(|kind| {
            if silt::soil::mix_channel(SurfaceKind::from_u8(*kind)).is_some() {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn sum_f32(values: &[f32]) -> f64 {
    values.iter().map(|v| *v as f64).sum()
}

fn nearest(field: &SoilField, x: f32, z: f32) -> usize {
    field
        .idx_at(x, z)
        .unwrap_or_else(|| panic!("no cell at {x},{z}"))
}

fn straddle(field: &SoilField, x: f32, z: f32) -> (usize, usize) {
    let left = nearest(field, x - field.cell * 0.5, z);
    let right = nearest(field, x + field.cell * 0.5, z);
    assert_ne!(left, right, "seam {x} landed in one cell");
    (left, right)
}

fn share(field: &SoilField, idx: usize, channel: usize) -> f32 {
    field.mix[idx * MIX_CHANNELS + channel]
}

fn assert_mixed(field: &SoilField, left: usize, right: usize, a: usize, b: usize, name: &str) {
    let la = share(field, left, a);
    let lb = share(field, left, b);
    let ra = share(field, right, a);
    let rb = share(field, right, b);
    eprintln!("SOIL_JOINTS seam {name} left {la:.4}/{lb:.4} right {ra:.4}/{rb:.4}");
    for v in [la, lb, ra, rb] {
        assert!(
            (0.25..0.75).contains(&v),
            "{name} still has a hard share {v:.3}"
        );
    }
    assert!((la - ra).abs() < 0.45, "{name} step {la:.3} to {ra:.3}");
    assert!((la + lb - 1.0).abs() < 1.0e-3);
    assert!((ra + rb - 1.0).abs() < 1.0e-3);
}

fn terramech_gap() -> (f32, f32, f32) {
    let mud = silt::soil::terramech_for_mix(SurfaceKind::WetMud, [1.0, 0.0, 0.0, 0.0], 0.7, 0.0);
    let joint = silt::soil::terramech_for_mix(SurfaceKind::WetMud, [0.5, 0.5, 0.0, 0.0], 0.7, 0.0);
    let sand = silt::soil::terramech_for_mix(SurfaceKind::Sand, [0.0, 1.0, 0.0, 0.0], 0.7, 0.0);
    (mud.k_eq, joint.k_eq, sand.k_eq)
}

fn dump_maps(soil: &SoilField, frame: &ContractFrame) {
    if std::fs::create_dir_all(DUMP_DIR).is_err() {
        eprintln!("SOIL_JOINTS dump skipped, {DUMP_DIR} is not writable");
        return;
    }
    let res = soil.res;
    let mut mix = vec![0u8; res * res * 3];
    let mut moisture = vec![0u8; res * res * 3];
    let mut mud = vec![0u8; res * res * 3];
    let mut sand = vec![0u8; res * res * 3];
    let mut gravel = vec![0u8; res * res * 3];
    let mut grass = vec![0u8; res * res * 3];
    for idx in 0..res * res {
        let m = share(soil, idx, MIX_MUD);
        let s = share(soil, idx, MIX_SAND);
        let g = share(soil, idx, MIX_GRAVEL);
        let a = share(soil, idx, MIX_GRASS);
        let sum = m + s + g + a;
        let (r, gch, b) = if sum < 1.0e-3 {
            (32u8, 32u8, 30u8)
        } else {
            let r = 92.0 * m + 196.0 * s + 120.0 * g + 46.0 * a;
            let gc = 48.0 * m + 154.0 * s + 116.0 * g + 118.0 * a;
            let b = 24.0 * m + 82.0 * s + 108.0 * g + 40.0 * a;
            (r.round() as u8, gc.round() as u8, b.round() as u8)
        };
        mix[idx * 3] = r;
        mix[idx * 3 + 1] = gch;
        mix[idx * 3 + 2] = b;
        let tone = (soil.wetness[idx].clamp(0.0, 1.0) * 255.0).round() as u8;
        moisture[idx * 3] = tone;
        moisture[idx * 3 + 1] = tone;
        moisture[idx * 3 + 2] = tone;
        paint_share(&mut mud, idx, m, [140, 72, 36]);
        paint_share(&mut sand, idx, s, [210, 176, 96]);
        paint_share(&mut gravel, idx, g, [150, 148, 140]);
        paint_share(&mut grass, idx, a, [52, 130, 48]);
    }
    write_png(
        &Path::new(DUMP_DIR).join("mix.png"),
        res as u32,
        res as u32,
        &mix,
    );
    write_png(
        &Path::new(DUMP_DIR).join("moisture.png"),
        res as u32,
        res as u32,
        &moisture,
    );
    write_png(
        &Path::new(DUMP_DIR).join("mud.png"),
        res as u32,
        res as u32,
        &mud,
    );
    write_png(
        &Path::new(DUMP_DIR).join("sand.png"),
        res as u32,
        res as u32,
        &sand,
    );
    write_png(
        &Path::new(DUMP_DIR).join("gravel.png"),
        res as u32,
        res as u32,
        &gravel,
    );
    write_png(
        &Path::new(DUMP_DIR).join("grass.png"),
        res as u32,
        res as u32,
        &grass,
    );

    let n = frame.header.window_res as usize;
    let mut contract = vec![0u8; n * n * 3];
    for idx in 0..n * n {
        let m = frame.soil_mix[idx * 4];
        let s = frame.soil_mix[idx * 4 + 1];
        let g = frame.soil_mix[idx * 4 + 2];
        let a = frame.soil_mix[idx * 4 + 3];
        let sum = m + s + g + a;
        let (r, gc, b) = if sum < 1.0e-3 {
            (16u8, 16u8, 18u8)
        } else {
            (
                (92.0 * m + 196.0 * s + 120.0 * g + 46.0 * a).round() as u8,
                (48.0 * m + 154.0 * s + 116.0 * g + 118.0 * a).round() as u8,
                (24.0 * m + 82.0 * s + 108.0 * g + 40.0 * a).round() as u8,
            )
        };
        contract[idx * 3] = r;
        contract[idx * 3 + 1] = gc;
        contract[idx * 3 + 2] = b;
    }
    write_png(
        &Path::new(DUMP_DIR).join("contract-mix.png"),
        n as u32,
        n as u32,
        &contract,
    );
    eprintln!("SOIL_JOINTS dumped {DUMP_DIR}");
}

fn paint_share(rgb: &mut [u8], idx: usize, share: f32, color: [u8; 3]) {
    let t = share.clamp(0.0, 1.0);
    for c in 0..3 {
        rgb[idx * 3 + c] = (18.0 + (color[c] as f32 - 18.0) * t).round() as u8;
    }
}

fn write_png(path: &Path, width: u32, height: u32, rgb: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = match File::create(path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("SOIL_JOINTS dump {} failed: {err}", path.display());
            return;
        }
    };
    let mut enc = png::Encoder::new(BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    match enc.write_header().and_then(|mut w| w.write_image_data(rgb)) {
        Ok(()) => {}
        Err(err) => eprintln!("SOIL_JOINTS dump {} failed: {err}", path.display()),
    }
}
