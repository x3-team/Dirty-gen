//! `capture --route <out>`: the Niva route (`silt::route`) run start to end,
//! one still per line of `bench/route.txt` taken on the way, and the CPU time
//! of every solver step written per leg to `<out>/route.json`.
//!
//! The scene list uses the bench format (`bench::parse_scenes`), so
//! `tools/bench_sbs.py <out> --scenes bench/route.txt` builds the SBS page.
//! Presets: `route-drive` (stop = chassis z), `route-crust` (stop unused),
//! `route-shore` (stop = seconds since the car started), `route-snow` (stop =
//! seconds on the snow).

use std::fmt::Write as _;
use std::path::Path;

use silt::route::{self, LegTimes, ShoreLeg, SnowLeg, StepTimer};

use crate::bench::{parse_scenes, pose, save, walker_pose, BenchScene};
use crate::render::Renderer;
use crate::terrain_draw;

/// Seconds of the snow script that are timed, shots or not.
const SNOW_TIMED_S: f32 = 15.0;

fn shots<'a>(scenes: &'a [BenchScene], preset: &str) -> Vec<&'a BenchScene> {
    let mut v: Vec<&BenchScene> = scenes.iter().filter(|s| s.preset == preset).collect();
    v.sort_by(|a, b| a.stop.total_cmp(&b.stop));
    v
}

fn quiet_particles(world: &mut silt::World) {
    for p in &mut world.particles.items {
        p.alive = false;
    }
}

pub fn run(renderer: &mut Renderer, width: u32, height: u32, out: &Path, scenes_file: &Path) {
    let text = std::fs::read_to_string(scenes_file)
        .unwrap_or_else(|e| panic!("read {}: {e}", scenes_file.display()));
    let scenes = parse_scenes(&text);
    let dir = out.join("ours");
    std::fs::create_dir_all(&dir).expect("route dir");
    renderer.show_bike = false;
    renderer.show_props = false;
    let mut legs: Vec<LegTimes> = Vec::new();
    let mut route_s = 0.0f32;

    // Leg 1: puddle, slurry ruts, dry exit with dust. The whole drive is timed.
    renderer.show_hills = false;
    renderer.daylight = false;
    renderer.niva_picture = true;
    renderer.water_column = true;
    renderer.open_water = false;
    let dt = 1.0 / 30.0;
    let mut times = LegTimes::new("drive", dt);
    let t = StepTimer::start();
    let mut drive = route::drive_leg();
    times.setup_ms = t.ms();
    let mut pending = shots(&scenes, "route-drive").into_iter().peekable();
    while drive.world.vehicle.pos.z < route::DRIVE_END_Z {
        if let Some(scene) = pending.peek() {
            if drive.world.vehicle.pos.z >= scene.stop {
                let cam = pose(scene, &drive.world);
                let rgba = renderer
                    .draw_with_sheet(&drive.world, cam, None, true, Some(&drive.fluid), Some(drive.dust()))
                    .expect("readback");
                save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
                pending.next();
                continue;
            }
        }
        let t = StepTimer::start();
        drive.step(dt);
        times.step_ms.push(t.ms());
        route_s += dt;
    }
    for scene in pending {
        println!("route {} past the end of the drive (z {:.1})", scene.id, scene.stop);
    }
    let coat = drive.world.vehicle.clone();
    println!(
        "route leg drive {:.1} s steps {} mean {:.3} p95 {:.3} max {:.3} ms",
        times.step_ms.len() as f32 * dt,
        times.step_ms.len(),
        times.mean(),
        times.pct(0.95),
        times.max()
    );
    legs.push(times);
    drop(drive);

    // Leg 2: the drying pad with seams. Static; the drying spell is one-off work.
    renderer.daylight = true;
    renderer.show_environment = false;
    renderer.water_column = false;
    let (mut world, setup) = route::crust_leg();
    route::carry_coat(&coat, &mut world.vehicle);
    quiet_particles(&mut world);
    let mut times = LegTimes::new("crust", 0.0);
    times.setup_ms = setup;
    for scene in shots(&scenes, "route-crust") {
        let cam = pose(scene, &world);
        let rgba = renderer.draw(&world, cam, None, true).expect("readback");
        save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
    }
    println!("route leg crust drying {:.1} ms (one-off)", setup);
    legs.push(times);
    drop(world);

    // Leg 3: beach, swash, the walker in the shallows, the ford.
    renderer.water_column = true;
    renderer.open_water = true;
    let dt = 1.0 / 60.0;
    let mut times = LegTimes::new("shore", dt);
    let t = StepTimer::start();
    let mut shore = ShoreLeg::new(4.0);
    times.setup_ms = t.ms();
    route::carry_coat(&coat, &mut shore.ford.world.vehicle);
    let list = shots(&scenes, "route-shore");
    let end_s = list.last().map(|s| s.stop).unwrap_or(0.0).max(18.0);
    let mut pending = list.into_iter().peekable();
    while shore.time_s() < end_s {
        if let Some(scene) = pending.peek() {
            if shore.time_s() >= scene.stop {
                let cam = if scene.walker_frame {
                    walker_pose(scene, &shore.bodies)
                } else {
                    pose(scene, &shore.ford.world)
                };
                renderer.set_bodies_mesh(Some(&crate::mesh::bodies_mesh(&shore.bodies)));
                let rgba = renderer
                    .draw_with_sheet(&shore.ford.world, cam, None, true, Some(&shore.ford.fluid), None)
                    .expect("readback");
                save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
                pending.next();
                continue;
            }
        }
        let t = StepTimer::start();
        shore.step(dt);
        times.step_ms.push(t.ms());
        route_s += dt;
    }
    renderer.set_bodies_mesh(None);
    renderer.open_water = false;
    let v = shore.ford.world.vehicle.pos;
    println!(
        "route leg shore {:.1} s steps {} mean {:.3} p95 {:.3} max {:.3} ms niva=({:.2},{:.2},{:.2})",
        shore.time_s(),
        times.step_ms.len(),
        times.mean(),
        times.pct(0.95),
        times.max(),
        v.x,
        v.y,
        v.z
    );
    let coat = shore.ford.world.vehicle.clone();
    legs.push(times);
    drop(shore);

    // Leg 4: deep snow.
    renderer.water_column = false;
    renderer.snow = true;
    renderer.snow_arc = None;
    renderer.snow_bed = 0.0;
    let dt = 1.0 / 60.0;
    let mut times = LegTimes::new("snow", dt);
    let mut snow = SnowLeg::new(std::env::var("SILT_ROUTE_SNOW").ok().and_then(|v| v.parse().ok()).unwrap_or(0.30));
    route::carry_coat(&coat, &mut snow.world.vehicle);
    for scene in shots(&scenes, "route-snow") {
        while snow.time_s < scene.stop {
            let t = StepTimer::start();
            snow.step_frame();
            times.step_ms.push(t.ms());
            route_s += dt;
        }
        snow.paint_soil();
        let p = snow.world.vehicle.pos;
        let win = snow.track.window(glam::Vec2::new(p.x, p.z));
        let texels: &[f32] = bytemuck::cast_slice(&win.texels);
        renderer.terrain_geom.set_track_fine(
            &renderer.queue,
            terrain_draw::TrackFineWindow {
                origin: [win.origin.x, win.origin.y],
                cell: win.cell,
                res: win.res as u32,
            },
            texels,
            |_, _| 0.0,
        );
        let cam = pose(scene, &snow.world);
        let rgba = renderer.draw(&snow.world, cam, None, true).expect("readback");
        save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
    }
    // The rest of the script is timed too: the turn exit and the stop.
    while snow.time_s < SNOW_TIMED_S {
        let t = StepTimer::start();
        snow.step_frame();
        times.step_ms.push(t.ms());
        route_s += dt;
    }
    renderer.snow = false;
    let p = snow.world.vehicle.pos;
    println!(
        "route leg snow {:.1} s steps {} mean {:.3} p95 {:.3} max {:.3} ms niva=({:.2},{:.2},{:.2})",
        snow.time_s,
        times.step_ms.len(),
        times.mean(),
        times.pct(0.95),
        times.max(),
        p.x,
        p.y,
        p.z
    );
    legs.push(times);

    write_report(out, &legs, route_s);
}

fn write_report(out: &Path, legs: &[LegTimes], route_s: f32) {
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let mut json = String::new();
    let _ = write!(
        json,
        "{{\n  \"cpu\": \"{}\",\n  \"cores\": {cores},\n  \"route_s\": {route_s:.2},\n  \"legs\": [\n",
        cpu.replace('"', "'")
    );
    for (i, l) in legs.iter().enumerate() {
        let _ = write!(
            json,
            "    {{\"leg\": \"{}\", \"dt\": {:.5}, \"steps\": {}, \"mean_ms\": {:.3}, \"p95_ms\": {:.3}, \"max_ms\": {:.3}, \"ms_per_60hz_frame\": {:.3}, \"setup_ms\": {:.1}}}{}\n",
            l.leg,
            l.dt,
            l.step_ms.len(),
            l.mean(),
            l.pct(0.95),
            l.max(),
            l.ms_per_60hz_frame(),
            l.setup_ms,
            if i + 1 < legs.len() { "," } else { "" }
        );
    }
    json.push_str("  ]\n}\n");
    std::fs::write(out.join("route.json"), &json).expect("route.json");
    println!("route cpu {cpu} x{cores} route {route_s:.1} s -> {}", out.join("route.json").display());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_list_parses_in_leg_order() {
        let scenes = parse_scenes(include_str!("../bench/route.txt"));
        assert!(scenes.len() >= 8);
        let order = ["route-drive", "route-crust", "route-shore", "route-snow"];
        let mut last = 0;
        for s in &scenes {
            let leg = order
                .iter()
                .position(|p| *p == s.preset)
                .unwrap_or_else(|| panic!("{}: preset {}", s.id, s.preset));
            assert!(leg >= last, "{} out of route order", s.id);
            last = leg;
        }
    }
}
