//! Smoke and fire grid on wgpu compute.
//!
//! Skips (prints and passes) when no adapter is present. On the cloud box it
//! picks lavapipe by itself when `/usr/share/vulkan/icd.d/lvp_icd.json` exists.
#![cfg(not(target_arch = "wasm32"))]

use glam::Vec3;
use silt::smoke::{FireSource, GpuSmoke, SmokeConfig};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let icd = "/usr/share/vulkan/icd.d/lvp_icd.json";
    let software = std::env::var_os("SILT_GPU").is_none() && std::path::Path::new(icd).exists();
    if software {
        std::env::set_var("VK_DRIVER_FILES", icd);
        std::env::set_var("VK_ICD_FILENAMES", icd);
        std::env::remove_var("DISPLAY");
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: if software {
            wgpu::Backends::VULKAN
        } else {
            wgpu::Backends::PRIMARY
        },
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: software,
        compatible_surface: None,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("smoke-test"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .ok()?;
    if !silt::smoke::supported(&device) {
        return None;
    }
    Some((device, queue))
}

const DT: f32 = 1.0 / 30.0;

fn grid(wind: Vec3, source: FireSource) -> Option<(wgpu::Device, wgpu::Queue, GpuSmoke)> {
    let Some((device, queue)) = device() else {
        println!("no wgpu adapter with compute; skipped");
        return None;
    };
    // Coarse grid so the test runs in seconds on lavapipe.
    let cfg = SmokeConfig::around(Vec3::ZERO, 3.2, 4.8, 0.1);
    let mut smoke = GpuSmoke::new(&device, cfg);
    smoke.wind = wind;
    smoke.sources = vec![source];
    smoke.reset(&queue);
    Some((device, queue, smoke))
}

#[test]
fn smoke_column_rises_in_still_air() {
    let Some((device, queue, mut smoke)) = grid(
        Vec3::ZERO,
        FireSource::smoke_vent(Vec3::new(0.0, 0.15, 0.0)),
    ) else {
        return;
    };
    let mut tops = Vec::new();
    for k in 0..90 {
        smoke.step(&device, &queue, DT);
        if k % 30 == 29 {
            let st = smoke.stats(&device, &queue);
            println!("t={:.1}s {st:?}", (k + 1) as f32 * DT);
            assert_eq!(st.nan, 0);
            tops.push(st.top_m);
        }
    }
    let st = smoke.stats(&device, &queue);
    // Rises: 3 s of a 1.4 m/s vent with buoyancy clears 2.5 m.
    assert!(st.top_m > 2.5, "top {:.2} m", st.top_m);
    assert!(tops.windows(2).all(|w| w[1] >= w[0]), "{tops:?}");
    // Still air: the column stays over the vent.
    let off = Vec3::new(st.soot_centre.x, 0.0, st.soot_centre.z).length();
    assert!(
        off < 0.35,
        "centre drifted {off:.2} m: {:?}",
        st.soot_centre
    );
    assert!(st.soot_m3 > 0.05, "soot {:.3}", st.soot_m3);
    assert!(st.max_speed_m_s < 6.01);
    // The projection cuts the divergence the forces add. 24 compact Jacobi
    // sweeps against the wide central-difference divergence remove about half;
    // the rest is the long-wavelength part Jacobi converges on slowly.
    let after = smoke.divergence_after(&device, &queue);
    println!(
        "div before projection {:.3}, after {:.3}",
        st.div_rms, after
    );
    assert!(after < 0.6 * st.div_rms, "{after} vs {}", st.div_rms);
}

#[test]
fn wind_leans_the_plume_downwind() {
    let Some((device, queue, mut smoke)) = grid(
        Vec3::new(1.2, 0.0, 0.0),
        FireSource::smoke_vent(Vec3::new(-0.8, 0.15, 0.0)),
    ) else {
        return;
    };
    for _ in 0..60 {
        smoke.step(&device, &queue, DT);
    }
    let st = smoke.stats(&device, &queue);
    println!("{st:?}");
    assert_eq!(st.nan, 0);
    assert!(st.soot_centre.x > -0.4, "centre x {:.2}", st.soot_centre.x);
}

#[test]
fn fire_burns_hot_and_makes_soot() {
    let Some((device, queue, mut smoke)) =
        grid(Vec3::ZERO, FireSource::campfire(Vec3::new(0.0, 0.1, 0.0)))
    else {
        return;
    };
    for _ in 0..45 {
        smoke.step(&device, &queue, DT);
    }
    let st = smoke.stats(&device, &queue);
    println!("{st:?}");
    assert_eq!(st.nan, 0);
    // A flame burns 600 K and more over ambient.
    assert!(st.max_temp_k > 600.0, "{:.0} K", st.max_temp_k);
    assert!(st.soot_m3 > 0.01, "soot {:.3}", st.soot_m3);
    assert!(st.top_m > 1.5, "top {:.2}", st.top_m);
}
