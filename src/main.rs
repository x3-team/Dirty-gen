//! Proof frame for the dust channel. Niva only, no text plate.

mod draw;
mod niva_mesh;

use std::path::PathBuf;
use std::time::Instant;

use glam::Vec3;

use draw::{write_png, Camera};
use dust_trail::{
    chassis_at, offer_m3_s, proof_wind, reference_samples, upload_density_texture, DensityTexture,
    Dust, FRAME_BUDGET_MS, PROOF_DT_S, PROOF_MOISTURE, PROOF_SECONDS, PROOF_SLIP_M_S,
    PROOF_SPEED_M_S, PROOF_WATER_M,
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
    let (dust, tex, times) = drive_proof();
    let (calm, calm_z) = dust_trail::drive(
        PROOF_SPEED_M_S,
        PROOF_SLIP_M_S,
        PROOF_MOISTURE,
        PROOF_WATER_M,
        Vec3::ZERO,
        PROOF_SECONDS,
        PROOF_DT_S,
    );
    let gpu = upload_timed(&tex);
    print_report(&dust, &tex, &times, &calm, calm_z, &gpu);
    let shown = match &gpu {
        Ok((g, _)) => g.read_density().unwrap_or_else(|_| tex.samples.clone()),
        Err(_) => tex.samples.clone(),
    };
    let mut shown_tex = tex.clone();
    if shown.len() == shown_tex.samples.len() {
        shown_tex.samples = shown;
    }
    let tris = niva_mesh::load_chassis();
    let chassis = chassis_at(dust.chassis_z());
    // Outside the plume, left-rear. Nose to the right, trail behind the bumper, +X wind to the left.
    let cam = Camera {
        eye: chassis + Vec3::new(-7.6, 1.28, -6.4),
        target: chassis + Vec3::new(0.25, 0.52, -1.6),
        fov_y: 0.60,
    };
    let rgb = draw::render(&tris, chassis, &shown_tex, &cam, WIDTH, HEIGHT);
    write_png(&out, WIDTH, HEIGHT, &rgb).expect("write proof");
    eprintln!(
        "proof {}x{} tris {} tex {}x{}x{} -> {}",
        WIDTH,
        HEIGHT,
        tris.len(),
        shown_tex.nx,
        shown_tex.ny,
        shown_tex.nz,
        out.display()
    );
}

fn drive_proof() -> (Dust, DensityTexture, Vec<f64>) {
    let mut dust = Dust::new();
    let wind = proof_wind();
    dust.set_wind(wind.x, wind.z);
    let mut tex = DensityTexture::empty();
    let steps = ((PROOF_SECONDS / PROOF_DT_S).round() as i32).max(0);
    let mut z = 0.0;
    let mut times = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let tires = dust_trail::tires_at(z, PROOF_SPEED_M_S, PROOF_SLIP_M_S);
        let t = Instant::now();
        dust.step(&tires, PROOF_MOISTURE, PROOF_WATER_M, PROOF_DT_S);
        dust.publish(&mut tex);
        times.push(t.elapsed().as_secs_f64() * 1000.0);
        z += PROOF_SPEED_M_S * PROOF_DT_S;
    }
    (dust, tex, times)
}

fn plume_height(tex: &DensityTexture) -> (f32, f32) {
    let mut volume = 0.0f64;
    let mut sy = 0.0f64;
    let mut above = 0.0f64;
    if tex.nx == 0 {
        return (0.0, 0.0);
    }
    for iz in 0..tex.nz {
        for iy in 0..tex.ny {
            for ix in 0..tex.nx {
                let i = (ix + iy * tex.nx + iz * tex.nx * tex.ny) as usize;
                let fraction = tex.samples[i];
                if fraction <= 0.0 {
                    continue;
                }
                let y = tex.origin.y + (iy as f32 + 0.5) * tex.cell.y;
                let v = fraction as f64;
                volume += v;
                sy += y as f64 * v;
                if y >= 0.45 {
                    above += v;
                }
            }
        }
    }
    if volume <= 0.0 {
        (0.0, 0.0)
    } else {
        ((sy / volume) as f32, (above / volume) as f32)
    }
}

fn upload_timed(tex: &DensityTexture) -> Result<(dust_trail::GpuDensity, f64), String> {
    let t = Instant::now();
    let gpu = upload_density_texture(tex)?;
    let ms = t.elapsed().as_secs_f64() * 1000.0;
    Ok((gpu, ms))
}

fn print_report(
    dust: &Dust,
    tex: &DensityTexture,
    times: &[f64],
    calm: &Dust,
    calm_z: f32,
    gpu: &Result<(dust_trail::GpuDensity, f64), String>,
) {
    let dry = dust_trail::dryness(PROOF_MOISTURE, PROOF_WATER_M);
    let rate = offer_m3_s(PROOF_SLIP_M_S, PROOF_MOISTURE, PROOF_WATER_M);
    let wind = proof_wind();
    let (wind_x, wind_z) = dust.wind();
    let steady: Vec<f64> = times.iter().copied().skip(15).collect();
    let mean = if steady.is_empty() {
        0.0
    } else {
        steady.iter().sum::<f64>() / steady.len() as f64
    };
    let max = steady.iter().copied().fold(0.0f64, f64::max);
    println!("dust channel");
    println!("speed_m_s {:.3}", PROOF_SPEED_M_S);
    println!("slip_m_s {:.3}", PROOF_SLIP_M_S);
    println!("moisture {:.3}", PROOF_MOISTURE);
    println!("water_m {:.3}", PROOF_WATER_M);
    println!("dryness {:.4}", dry);
    println!("wind_m_s {:.3} {:.3} {:.3}", wind.x, wind.y, wind.z);
    println!("set_wind {:.3} {:.3}", wind_x, wind_z);
    println!("offer_per_tire_m3_s {:.6}", rate);
    println!("offer_total_m3_s {:.6}", rate * 4.0);
    let books = dust.ledger();
    println!("emitted_m3 {:.6}", books.emitted_m3);
    println!("dust_air_m3 {:.6}", books.dust_air_m3);
    println!("dust_settled_m3 {:.6}", books.dust_settled_m3);
    println!(
        "ledger_err_m3 {:.8}",
        (books.emitted_m3 - books.dust_air_m3 - books.dust_settled_m3).abs()
    );
    println!("max_fraction {:.6}", dust.max_fraction());
    println!("texture {}x{}x{}", tex.nx, tex.ny, tex.nz);
    println!("texture_air_m3 {:.6}", tex.airborne_m3());
    for i in 0..4 {
        let aft = dust.aft_of_tire(i).unwrap_or(f32::NAN);
        println!(
            "tire{i}_emitted_m3 {:.6} aft_m {:.3}",
            dust.emitted_by_tire()[i],
            aft
        );
    }
    let _ = calm_z;
    let (behind, lateral) = dust.plume_offset();
    let (calm_behind, calm_lateral) = calm.plume_offset();
    let (height, above) = plume_height(tex);
    println!("plume_behind_rear_m {:.3}", behind);
    println!("plume_lateral_m {:.3}", lateral);
    println!("plume_height_m {:.3}", height);
    println!("plume_mass_above_0.45 {:.3}", above);
    println!("calm_plume_behind_rear_m {:.3}", calm_behind);
    println!("calm_plume_lateral_m {:.3}", calm_lateral);
    for i in 2..4 {
        let aft = calm.aft_of_tire(i).unwrap_or(f32::NAN);
        println!("calm_tire{i}_aft_m {:.3}", aft);
    }
    println!("frame_budget_ms {:.3}", FRAME_BUDGET_MS);
    println!("frame_mean_ms {:.4}", mean);
    println!("frame_max_ms {:.4}", max);
    println!("frames_timed {}", steady.len());
    match gpu {
        Ok((g, ms)) => {
            let size = g.extent();
            println!("gpu_adapter {}", g.adapter_name);
            println!(
                "gpu_texture {}x{}x{} {:?}",
                size.width, size.height, size.depth_or_array_layers, g.format
            );
            println!("gpu_upload_ms {:.3}", ms);
        }
        Err(err) => println!("gpu_texture_error {err}"),
    }
    println!("reference");
    println!("moisture water_m slip dryness offer_m3_s");
    for s in reference_samples() {
        println!(
            "{:.2} {:.3} {:.1} {:.4} {:.6}",
            s.moisture, s.water_m, s.slip, s.dryness, s.offer_m3_s
        );
    }
    println!("simulated {:.1} s, calm wind, speed 8", PROOF_SECONDS);
    println!("case emitted_m3 dust_air_m3 dust_settled_m3");
    for (name, m, w, slip) in [
        ("dry_slip_1.6", 0.0_f32, 0.0_f32, 1.6_f32),
        ("proof_pad", PROOF_MOISTURE, 0.0, PROOF_SLIP_M_S),
        ("damp_0.30", 0.30, 0.0, 1.6),
        ("slip_0.4", 0.0, 0.0, 0.4),
        ("slip_3", 0.0, 0.0, 3.0),
        ("zero_slip", 0.0, 0.0, 0.0),
        ("wet", 0.90, 0.0, 1.6),
        ("pond", 0.0, 0.02, 1.6),
    ] {
        let (d, _) = dust_trail::drive(8.0, slip, m, w, Vec3::ZERO, PROOF_SECONDS, PROOF_DT_S);
        let books = d.ledger();
        println!(
            "{name} {:.6} {:.6} {:.6}",
            books.emitted_m3, books.dust_air_m3, books.dust_settled_m3
        );
    }
}
