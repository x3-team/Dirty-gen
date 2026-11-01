//! UE5 comparison bench. `capture --bench <out>` renders one still per line
//! of `bench/scenes.txt` into `<out>/ours/<id>.png`. `tools/bench_sbs.py`
//! then puts each still next to its UE5 reference.
//!
//! A scene line is
//!
//! ```text
//! id  preset  stop  frame  eye_x eye_y eye_z  tgt_x tgt_y tgt_z  ref  title...
//! ```
//!
//! * `preset` is `drive` (the Niva through-drive: puddle, slurry ruts, dry
//!   exit with dust), `snow` (snow blanket with ruts and turn circles) or
//!   `missing` (no solver scene yet; nothing is rendered and the SBS shows
//!   the title as the reason).
//! * `stop` is the chassis z where the drive still is taken. Unused for snow.
//! * `frame` is `car` (x forward, y up, z right of the Niva) or `world`.
//!
//! Cameras are data so a shot can be matched to a reference without a
//! rebuild: `--bench-scenes <file>` reads another list.

use std::path::{Path, PathBuf};

use glam::Vec3;
use silt::world::{CameraPose, World};

use crate::render::{write_png, Renderer};

#[derive(Clone, Debug)]
pub struct BenchScene {
    pub id: String,
    pub preset: String,
    pub stop: f32,
    pub car_frame: bool,
    pub eye: Vec3,
    pub target: Vec3,
}

pub fn parse_scenes(text: &str) -> Vec<BenchScene> {
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 11 {
            panic!("bench/scenes.txt line {}: need 11+ fields", n + 1);
        }
        let num = |i: usize| -> f32 {
            f[i].parse()
                .unwrap_or_else(|_| panic!("bench/scenes.txt line {}: bad number {}", n + 1, f[i]))
        };
        let car_frame = match f[3] {
            "car" => true,
            "world" => false,
            other => panic!("bench/scenes.txt line {}: frame {other}", n + 1),
        };
        out.push(BenchScene {
            id: f[0].to_string(),
            preset: f[1].to_string(),
            stop: num(2),
            car_frame,
            eye: Vec3::new(num(4), num(5), num(6)),
            target: Vec3::new(num(7), num(8), num(9)),
        });
    }
    out
}

fn pose(scene: &BenchScene, world: &World) -> CameraPose {
    if !scene.car_frame {
        return CameraPose {
            eye: scene.eye,
            target: scene.target,
            ortho_width: None,
        };
    }
    let fwd = world.vehicle.forward();
    let fwd = Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
    let right = Vec3::new(-fwd.z, 0.0, fwd.x);
    let at = |v: Vec3| world.vehicle.pos + fwd * v.x + Vec3::Y * v.y + right * v.z;
    CameraPose {
        eye: at(scene.eye),
        target: at(scene.target),
        ortho_width: None,
    }
}

fn save(path: &Path, width: u32, height: u32, rgba: &[u8], scene: &BenchScene, cam: CameraPose) {
    write_png(path, width, height, rgba).expect("bench png");
    println!(
        "bench {} {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        scene.id,
        path.display(),
        cam.eye.x,
        cam.eye.y,
        cam.eye.z,
        cam.target.x,
        cam.target.y,
        cam.target.z
    );
}

pub fn run(renderer: &mut Renderer, width: u32, height: u32, out: &Path, scenes_file: &Path) {
    let text = std::fs::read_to_string(scenes_file)
        .unwrap_or_else(|e| panic!("read {}: {e}", scenes_file.display()));
    let scenes = parse_scenes(&text);
    let dir: PathBuf = out.join("ours");
    std::fs::create_dir_all(&dir).expect("bench dir");

    renderer.show_bike = false;
    renderer.show_props = false;

    // Snow first: it is one static still and needs no simulation.
    for scene in scenes.iter().filter(|s| s.preset == "snow") {
        renderer.show_hills = true;
        renderer.show_environment = false;
        renderer.daylight = true;
        renderer.niva_picture = true;
        renderer.water_column = false;
        renderer.snow = true;
        let (world, snow) = crate::snow_ruts_world();
        renderer.snow_bed = world.soil.bed[0];
        renderer.snow_arc = snow.snow.arc;
        let cam = pose(scene, &world);
        let rgba = renderer.draw(&world, cam, None, true).expect("readback");
        save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
    }
    renderer.snow = false;
    renderer.snow_arc = None;

    // One through-drive serves every `drive` still, in order of stop.
    let mut drives: Vec<&BenchScene> = scenes.iter().filter(|s| s.preset == "drive").collect();
    drives.sort_by(|a, b| a.stop.total_cmp(&b.stop));
    if !drives.is_empty() {
        renderer.show_hills = false;
        renderer.daylight = false;
        renderer.niva_picture = true;
        renderer.water_column = true;
        let mut drive = silt::ThroughDrive::new();
        let dt = 1.0 / 30.0;
        for scene in drives {
            while drive.world.vehicle.pos.z < scene.stop && drive.world.vehicle.pos.z < 19.6 {
                drive.step(dt);
            }
            let cam = pose(scene, &drive.world);
            let rgba = renderer
                .draw_with_sheet(
                    &drive.world,
                    cam,
                    None,
                    true,
                    Some(&drive.fluid),
                    Some(drive.dust()),
                )
                .expect("readback");
            save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
        }
    }

    // River ford with rocks (`river`) and the same ford with a swell on the
    // beach (`beach`). `stop` is seconds since the Niva started.
    for (preset, swell) in [("river", None), ("beach", Some(silt::ford::Swell::default()))] {
        let mut shots: Vec<&BenchScene> = scenes.iter().filter(|s| s.preset == preset).collect();
        if shots.is_empty() {
            continue;
        }
        shots.sort_by(|a, b| a.stop.total_cmp(&b.stop));
        renderer.show_hills = false;
        renderer.daylight = true;
        renderer.niva_picture = true;
        renderer.water_column = true;
        renderer.open_water = true;
        let mut ford = silt::ford::FordDrive::new(swell, 4.0);
        let dt = 1.0 / 60.0;
        for scene in shots {
            while ford.time_s < scene.stop {
                ford.step(dt);
            }
            let cam = pose(scene, &ford.world);
            let rgba = renderer
                .draw_with_sheet(&ford.world, cam, None, true, Some(&ford.fluid), None)
                .expect("readback");
            save(&dir.join(format!("{}.png", scene.id)), width, height, &rgba, scene, cam);
        }
    }

    renderer.open_water = false;

    for scene in scenes.iter().filter(|s| s.preset == "missing") {
        println!("bench {} missing: no solver scene yet", scene.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_scene_list_parses() {
        let text = include_str!("../bench/scenes.txt");
        let scenes = parse_scenes(text);
        assert!(scenes.len() >= 5);
        for s in &scenes {
            assert!(["drive", "snow", "river", "beach", "missing"].contains(&s.preset.as_str()), "{}", s.id);
        }
    }
}
