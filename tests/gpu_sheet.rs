//! Spike: the gravity sheet on wgpu compute against the CPU step.
//!
//! Skips (prints and passes) when no adapter is present. On the cloud box use
//! lavapipe: `SILT_SHEET_GPU_SOFTWARE=1 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`.
#![cfg(not(target_arch = "wasm32"))]

use std::time::Instant;

use silt::fluid_core::{
    step_sheet, step_sheet_walled, with_gpu_sheet, COARSE_CELL_M, FINE_CELL_M, FINE_RES,
    WINDOW_RES,
};
use silt::gpu_sheet::SheetJob;

const DT: f32 = 1.0 / 30.0;

#[derive(Clone)]
struct Scene {
    res: usize,
    cell: f32,
    h: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    bed: Vec<f32>,
    nu: Vec<f32>,
    gate: Vec<u8>,
    solid: Option<Vec<u8>>,
    vorticity: f32,
}

/// 40 m window: a pond on a gentle slope runs at a wheel-sized wall, with a
/// rut along the track. Water viscosity.
fn walled_pond(nu: f32) -> Scene {
    let res = WINDOW_RES;
    let n = res * res;
    let mut s = Scene {
        res,
        cell: COARSE_CELL_M,
        h: vec![0.0; n],
        u: vec![0.0; n],
        v: vec![0.0; n],
        bed: vec![0.0; n],
        nu: vec![nu; n],
        gate: vec![1; n],
        solid: Some(vec![0; n]),
        vorticity: 0.08,
    };
    for j in 0..res {
        for i in 0..res {
            let idx = j * res + i;
            let x = i as f32 * s.cell;
            let z = j as f32 * s.cell;
            let rut = if (i as i32 - 96).abs() <= 1 || (i as i32 - 104).abs() <= 1 {
                -0.06
            } else {
                0.0
            };
            s.bed[idx] = -0.004 * z + 0.02 * (0.7 * x).sin() * (0.5 * z).cos() + rut;
            if (60..140).contains(&i) && (40..120).contains(&j) {
                s.h[idx] = 0.10 + 0.03 * ((i + j) % 7) as f32 / 7.0;
                s.v[idx] = 0.6;
            }
        }
    }
    let solid = s.solid.as_mut().unwrap();
    for j in 128..134 {
        for i in 94..106 {
            solid[j * res + i] = 1;
        }
    }
    s
}

/// 2 cm sheet under the tires, open: a tilted pond spreading on a film.
fn fine_open() -> Scene {
    let res = FINE_RES;
    let n = res * res;
    let mut s = Scene {
        res,
        cell: FINE_CELL_M,
        h: vec![0.004; n],
        u: vec![0.02; n],
        v: vec![0.0; n],
        bed: vec![0.0; n],
        nu: vec![1.0; n],
        gate: vec![1; n],
        solid: None,
        vorticity: 0.08,
    };
    for j in 0..res {
        for i in 0..res {
            let idx = j * res + i;
            s.bed[idx] = -0.002 * j as f32 + 0.01 * ((i as f32) * 0.3).sin();
            if (40..100).contains(&i) && (30..90).contains(&j) {
                s.h[idx] = 0.06;
            }
        }
    }
    s
}

fn cpu_step(s: &mut Scene) {
    match &s.solid {
        Some(m) => step_sheet_walled(
            &mut s.h, &mut s.u, &mut s.v, &s.bed, &s.nu, &s.gate, m, s.res, s.cell, DT,
            s.vorticity,
        ),
        None => step_sheet(
            &mut s.h,
            &mut s.u,
            &mut s.v,
            &s.bed,
            &s.nu,
            &s.gate,
            s.res,
            s.cell,
            DT,
            silt::fluid_core::SheetDrive::Gravity {
                vorticity: s.vorticity,
            },
        ),
    }
}

fn gpu_step(s: &mut Scene) -> bool {
    silt::fluid_core::step_sheet_gpu(
        &mut s.h,
        &mut s.u,
        &mut s.v,
        &s.bed,
        &s.nu,
        &s.gate,
        s.res,
        s.cell,
        DT,
        s.vorticity,
        1.0e-6,
        s.solid.as_deref(),
    )
}

fn volume(h: &[f32], cell: f32) -> f64 {
    h.iter().map(|x| f64::from(*x)).sum::<f64>() * f64::from(cell * cell)
}

fn has_gpu() -> bool {
    let _ = env_logger::try_init();
    match with_gpu_sheet(|g| g.adapter_name.clone()) {
        Some(name) => {
            println!("gpu_sheet adapter {name}");
            true
        }
        None => {
            println!("gpu_sheet: no adapter, skipped");
            false
        }
    }
}

struct Diff {
    max_dh: f32,
    mean_dh: f64,
    wet_cpu: usize,
    wet_gpu: usize,
}

fn diff(a: &Scene, b: &Scene) -> Diff {
    let mut max_dh = 0.0f32;
    let mut sum = 0.0f64;
    let mut wet = 0usize;
    let mut wet_b = 0usize;
    for k in 0..a.h.len() {
        if a.h[k] > 1.0e-4 || b.h[k] > 1.0e-4 {
            let d = (a.h[k] - b.h[k]).abs();
            max_dh = max_dh.max(d);
            sum += f64::from(d);
        }
        if a.h[k] > 1.0e-4 {
            wet += 1;
        }
        if b.h[k] > 1.0e-4 {
            wet_b += 1;
        }
    }
    Diff {
        max_dh,
        mean_dh: sum / wet.max(1) as f64,
        wet_cpu: wet,
        wet_gpu: wet_b,
    }
}

fn compare(label: &str, start: Scene, steps: usize) -> (f64, Diff) {
    let mut cpu = start.clone();
    let mut gpu = start.clone();
    let v0 = volume(&start.h, start.cell);
    for _ in 0..steps {
        cpu_step(&mut cpu);
        assert!(gpu_step(&mut gpu));
    }
    let vc = volume(&cpu.h, cpu.cell);
    let vg = volume(&gpu.h, gpu.cell);
    let d = diff(&cpu, &gpu);
    let nan = gpu
        .h
        .iter()
        .chain(gpu.u.iter())
        .chain(gpu.v.iter())
        .filter(|x| !x.is_finite())
        .count();
    let vmax = gpu.u.iter().chain(gpu.v.iter()).fold(0.0f32, |m, x| m.max(x.abs()));
    let cmax = cpu.u.iter().chain(cpu.v.iter()).fold(0.0f32, |m, x| m.max(x.abs()));
    println!(
        "GPU_SHEET {label}: steps {steps} volume start {v0:.6} m3 cpu {vc:.6} gpu {vg:.6} \
         (gpu rel gap {:.2e}, cpu rel gap {:.2e}) depth diff max {:.5} m mean {:.6} m \
         wet cells cpu {} gpu {} nan {nan} max speed gpu {vmax:.3} cpu {cmax:.3}",
        (vg - v0).abs() / v0,
        (vc - v0).abs() / v0,
        d.max_dh,
        d.mean_dh,
        d.wet_cpu,
        d.wet_gpu,
    );
    assert_eq!(nan, 0, "{label}: non-finite on GPU");
    assert!(vmax <= 1.6 + 1e-4, "{label}: speed {vmax}");
    ((vg - v0).abs() / v0, d)
}

#[test]
fn gpu_sheet_conserves_volume_and_tracks_cpu() {
    if !has_gpu() {
        return;
    }
    let (gap, d) = compare("walled water 40 m", walled_pond(0.9), 90);
    assert!(gap < 1.0e-4, "walled water volume gap {gap:.2e}");
    assert!(d.mean_dh < 2.0e-3, "walled water mean depth diff {}", d.mean_dh);

    let (gap, d) = compare("walled slurry 40 m", walled_pond(60.0), 90);
    assert!(gap < 1.0e-4, "walled slurry volume gap {gap:.2e}");
    assert!(d.mean_dh < 2.0e-3, "walled slurry mean depth diff {}", d.mean_dh);

    let (gap, d) = compare("open 2 cm", fine_open(), 90);
    assert!(gap < 1.0e-4, "open fine volume gap {gap:.2e}");
    assert!(d.mean_dh < 2.0e-3, "open fine mean depth diff {}", d.mean_dh);
}

fn bench(label: &str, start: Scene, steps: usize) {
    let warm = 10;
    // CPU, the current path.
    let mut cpu = start.clone();
    for _ in 0..warm {
        cpu_step(&mut cpu);
    }
    let t = Instant::now();
    for _ in 0..steps {
        cpu_step(&mut cpu);
    }
    let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;

    // GPU as a drop-in: upload, substeps, read back on every call.
    let mut gpu = start.clone();
    for _ in 0..warm {
        gpu_step(&mut gpu);
    }
    let (mut up, mut run, mut down) = (0.0, 0.0, 0.0);
    let t = Instant::now();
    for _ in 0..steps {
        gpu_step(&mut gpu);
        let last = with_gpu_sheet(|g| g.last).unwrap();
        up += last.upload_ms;
        run += last.compute_ms;
        down += last.readback_ms;
    }
    let drop_ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;

    // GPU resident: the sheet never leaves the device; one readback at the end.
    let mut res = start.clone();
    // Same CFL count `step_sheet_gpu` takes for this start.
    let nsub = if res.solid.is_none() && res.cell <= FINE_CELL_M + 1e-6 {
        1
    } else {
        let s = res.u.iter().zip(&res.v).fold(0.15f32, |m, (a, b)| m.max(a.hypot(*b)));
        ((DT * s) / (0.42 * res.cell)).ceil().clamp(1.0, 8.0) as u32
    };
    let project = res.solid.is_some() || res.cell > FINE_CELL_M + 1e-6;
    let resident_ms = with_gpu_sheet(|g| {
        let mut job = SheetJob {
            h: &mut res.h,
            u: &mut res.u,
            v: &mut res.v,
            bed: &res.bed,
            nu: &res.nu,
            gate: &res.gate,
            solid: res.solid.as_deref(),
            res: res.res,
            cell: res.cell,
            dt: DT,
            vorticity: res.vorticity,
            eps: 1.0e-6,
            nsub,
            project,
        };
        g.upload(&job);
        for _ in 0..warm {
            g.run(&job);
        }
        let t = Instant::now();
        for _ in 0..steps {
            g.run(&job);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3 / steps as f64;
        g.download(&mut job);
        ms
    })
    .unwrap();
    let k = steps as f64;
    println!(
        "GPU_SHEET_BENCH {label}: cpu {cpu_ms:.3} ms/step | gpu drop-in {drop_ms:.3} ms/step \
         (upload {:.3} compute {:.3} readback {:.3}) | gpu resident {resident_ms:.3} ms/step (nsub {nsub})",
        up / k,
        run / k,
        down / k,
    );
}

/// Timing only. `cargo test --release --test gpu_sheet -- --ignored --nocapture`
#[test]
#[ignore]
fn gpu_sheet_bench() {
    if !has_gpu() {
        return;
    }
    bench("walled water 40 m 200x200", walled_pond(0.9), 60);
    bench("walled slurry 40 m 200x200", walled_pond(60.0), 60);
    bench("open 2 cm 160x160", fine_open(), 60);
}
