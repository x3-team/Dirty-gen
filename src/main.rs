//! Proof frame for the dust-trail solver. Niva only, no text plate.

mod draw;
mod niva_mesh;

use std::path::PathBuf;

use glam::Vec3;

use draw::{write_png, Camera};
use dust_trail::{
    chassis_at, drive, dryness, loft_rate_kg_s, proof_wind, reference_samples, wake_density_kg_m3,
    DustTrail, PROOF_DT_S, PROOF_MOISTURE, PROOF_SECONDS, PROOF_SPEED_M_S, PROOF_WATER_M,
};

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() {
    let out = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(
                "/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media/dust-trail-proof.png",
            )
        });
    let (dust, z) = drive(
        PROOF_SPEED_M_S,
        PROOF_MOISTURE,
        PROOF_WATER_M,
        proof_wind(),
        PROOF_SECONDS,
        PROOF_DT_S,
    );
    let (calm, zc) = drive(
        PROOF_SPEED_M_S,
        PROOF_MOISTURE,
        PROOF_WATER_M,
        Vec3::ZERO,
        PROOF_SECONDS,
        PROOF_DT_S,
    );
    print_report(&dust, z, &calm, zc);
    let tris = niva_mesh::load_chassis();
    let chassis = chassis_at(z);
    // Left side, a little aft, so the plume is a cloud behind the tires
    // rather than a column looked at end-on.
    // Straight from the left so the ray crosses the plume instead of
    // looking down its length. Nose to the right, cloud behind the tires.
    let cam = Camera {
        eye: chassis + Vec3::new(-8.4, 1.42, -1.35),
        target: chassis + Vec3::new(0.1, 0.58, -1.35),
        fov_y: 0.58,
    };
    let rgb = draw::render(&tris, chassis, &dust.parcels, &cam, WIDTH, HEIGHT);
    write_png(&out, WIDTH, HEIGHT, &rgb).expect("write proof");
    eprintln!(
        "proof {}x{} tris {} parcels {} -> {}",
        WIDTH,
        HEIGHT,
        tris.len(),
        dust.parcels.len(),
        out.display()
    );
}

fn print_report(dust: &DustTrail, z: f32, calm: &DustTrail, zc: f32) {
    let dry = dryness(PROOF_MOISTURE, PROOF_WATER_M);
    let rate = loft_rate_kg_s(PROOF_MOISTURE, PROOF_WATER_M, PROOF_SPEED_M_S);
    let rho = wake_density_kg_m3(PROOF_MOISTURE, PROOF_WATER_M, PROOF_SPEED_M_S);
    let (behind, lateral) = dust.rear_offset(z).unwrap();
    let (behind_c, lateral_c) = calm.rear_offset(zc).unwrap();
    let wind = proof_wind();
    let max_y = dust.parcels.iter().map(|p| p.pos.y).fold(0.0_f32, f32::max);
    println!("dust trail");
    println!("speed_m_s {:.3}", PROOF_SPEED_M_S);
    println!("moisture {:.3}", PROOF_MOISTURE);
    println!("water_m {:.3}", PROOF_WATER_M);
    println!("dryness {:.4}", dry);
    println!("wind_m_s {:.3} {:.3} {:.3}", wind.x, wind.y, wind.z);
    println!("loft_per_tire_kg_s {:.6}", rate);
    println!("loft_total_kg_s {:.6}", rate * 4.0);
    println!("wake_density_kg_m3 {:.6}", rho);
    println!("emitted_kg {:.6}", dust.emitted_kg);
    println!("airborne_kg {:.6}", dust.airborne_kg());
    println!("settled_kg {:.6}", dust.settled_kg);
    println!("near_wake_bulk_kg_m3 {:.6}", dust.near_wake_bulk_kg_m3(z));
    println!("behind_rear_axle_m {:.3}", behind);
    println!("lateral_m {:.3}", lateral);
    println!("calm_behind_m {:.3}", behind_c);
    println!("calm_lateral_m {:.3}", lateral_c);
    println!("cloud_top_m {:.3}", max_y);
    println!("parcels {}", dust.parcels.len());
    println!("reference");
    println!("moisture water_m speed dryness rate_kg_s density_kg_m3");
    for s in reference_samples() {
        println!(
            "{:.2} {:.3} {:.1} {:.4} {:.6} {:.6}",
            s.moisture, s.water_m, s.speed, s.dryness, s.rate_kg_s, s.density_kg_m3
        );
    }
    println!(
        "simulated near-wake bulk, calm wind, {:.1} s",
        PROOF_SECONDS
    );
    println!("case bulk_kg_m3 emitted_kg");
    for (name, m, w, u) in [
        ("dry_8", 0.0_f32, 0.0_f32, 8.0_f32),
        ("proof_pad_8", PROOF_MOISTURE, 0.0, 8.0),
        ("damp_8", 0.30, 0.0, 8.0),
        ("slow_3", 0.0, 0.0, 3.0),
        ("fast_12", 0.0, 0.0, 12.0),
        ("wet_8", 0.90, 0.0, 8.0),
        ("pond_8", 0.0, 0.02, 8.0),
    ] {
        let (d, zz) = drive(u, m, w, Vec3::ZERO, PROOF_SECONDS, PROOF_DT_S);
        println!(
            "{name} {:.6} {:.6}",
            d.near_wake_bulk_kg_m3(zz),
            d.emitted_kg
        );
    }
}
