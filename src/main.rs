//! Silt window and offscreen capture.
//!
//! `silt` drives the buggy. `silt capture` steps the same solver and writes frames.

mod cc0;
mod mesh;
mod niva;
mod render;

#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use glam::Vec3;
use silt::world::{demo_camera, demo_drive, demo_drive_bike, CameraPose, Input, World};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use render::Renderer;
#[cfg(not(target_arch = "wasm32"))]
use render::{open_adapter, prepare_software_vulkan, write_png};

#[cfg(not(target_arch = "wasm32"))]
const MEDIA: &str = "/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media";

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        run_window();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
        let mut args = std::env::args().skip(1);
        if args.next().as_deref() == Some("capture") {
            let rest: Vec<String> = args.collect();
            capture(&rest);
        } else {
            run_window();
        }
    }
}

struct Keys {
    throttle: bool,
    reverse: bool,
    left: bool,
    right: bool,
    orbit_left: bool,
    orbit_right: bool,
    orbit_up: bool,
    orbit_down: bool,
}

impl Keys {
    fn input(&self, speed: f32) -> Input {
        let mut throttle = if self.throttle { 1.0 } else { 0.0 };
        let mut brake = 0.0;
        if self.reverse {
            if speed < 0.55 && throttle == 0.0 {
                throttle = -1.0;
            } else {
                brake = 1.0;
            }
        }
        let steer = if self.left { 1.0 } else { 0.0 } - if self.right { 1.0 } else { 0.0 };
        Input {
            throttle,
            brake,
            steer: -steer,
        }
    }
}

struct App {
    window: Option<std::sync::Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    renderer: Option<Renderer>,
    world: World,
    keys: Keys,
    orbit_yaw: f32,
    orbit_pitch: f32,
    chase_cam: bool,
    drive_bike: bool,
    /// Set once the player presses a drive key. Until then both vehicles run
    /// the scripted lap. After that, the selected one takes the keys and the
    /// other keeps the lap.
    player_drove: bool,
    acc: f32,
    last: Instant,
    srgb: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        self.start_web(event_loop);
        #[cfg(not(target_arch = "wasm32"))]
        self.start_native(event_loop);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let (Some(surface), Some(config), Some(renderer)) =
                    (&self.surface, self.config.as_mut(), self.renderer.as_mut())
                {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(&renderer.device, config);
                    renderer.resize(config.width, config.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if event.state == ElementState::Pressed
                    && event.physical_key == PhysicalKey::Code(KeyCode::Escape)
                {
                    event_loop.exit();
                    return;
                }
                if event.state == ElementState::Pressed && !event.repeat {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyC) => self.chase_cam = !self.chase_cam,
                        PhysicalKey::Code(KeyCode::KeyV) => self.drive_bike = !self.drive_bike,
                        PhysicalKey::Code(KeyCode::Backspace) => self.world.reset_vehicle(),
                        _ => {}
                    }
                }
                let down = event.state == ElementState::Pressed;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                        self.keys.throttle = down;
                        if down {
                            self.player_drove = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                        self.keys.reverse = down;
                        if down {
                            self.player_drove = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                        self.keys.left = down;
                        if down {
                            self.player_drove = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                        self.keys.right = down;
                        if down {
                            self.player_drove = true;
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyQ) => self.keys.orbit_left = down,
                    PhysicalKey::Code(KeyCode::KeyE) => self.keys.orbit_right = down,
                    PhysicalKey::Code(KeyCode::KeyR) => self.keys.orbit_up = down,
                    PhysicalKey::Code(KeyCode::KeyF) => self.keys.orbit_down = down,
                    _ => {}
                }
            }
            WindowEvent::RedrawRequested => self.frame(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl App {
    #[cfg(not(target_arch = "wasm32"))]
    fn start_native(&mut self, event_loop: &ActiveEventLoop) {
        let window = std::sync::Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Silt")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("window"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let (_instance, adapter, device, queue) = open_adapter(false, Some(&surface));
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| {
                matches!(
                    f,
                    wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
                )
            })
            .unwrap_or(caps.formats[0]);
        let srgb = format.is_srgb();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let name = adapter.get_info().name;
        let formats = render::gpu_formats(&adapter);
        eprintln!("loading cc0 ground and the dune buggy");
        let scene = cc0::load();
        let mut renderer = Renderer::new(
            device,
            queue,
            name,
            config.width,
            config.height,
            formats,
            scene,
        );
        renderer.set_present_format(format);
        self.window = Some(window);
        self.surface = Some(surface);
        self.config = Some(config);
        self.renderer = Some(renderer);
        self.srgb = srgb;
    }

    fn frame(&mut self) {
        #[cfg(target_arch = "wasm32")]
        if self.renderer.is_none() {
            if !self.take_web_gpu() {
                return;
            }
            self.last = Instant::now();
        }
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32().min(0.05);
        self.last = now;
        if self.keys.orbit_left {
            self.orbit_yaw -= dt * 0.9;
        }
        if self.keys.orbit_right {
            self.orbit_yaw += dt * 0.9;
        }
        if self.keys.orbit_up {
            self.orbit_pitch = (self.orbit_pitch + dt * 0.7).min(1.1);
        }
        if self.keys.orbit_down {
            self.orbit_pitch = (self.orbit_pitch - dt * 0.7).max(-0.35);
        }
        self.acc += dt;
        let step = 1.0 / 60.0;
        while self.acc >= step {
            // The page opens on the lap: buggy and bike both chase the loop.
            // A drive key takes the selected vehicle. The other one stays on
            // the script, so the track never drops to a single machine.
            let bike_input = if self.player_drove && self.drive_bike {
                self.keys.input(self.world.bike.speed())
            } else {
                demo_drive_bike(&self.world)
            };
            let buggy_input = if self.player_drove && !self.drive_bike {
                self.keys.input(self.world.vehicle.speed())
            } else {
                demo_drive(&self.world)
            };
            self.world.step_bike(step, &bike_input);
            self.world.step(step, &buggy_input);
            self.acc -= step;
        }
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(surface) = self.surface.as_ref() else {
            return;
        };
        let (cam_pos, cam_yaw) = if self.drive_bike {
            (self.world.bike.pos, self.world.bike.yaw)
        } else {
            (self.world.vehicle.pos, self.world.vehicle.yaw)
        };
        let pose = if self.chase_cam {
            chase(cam_pos, cam_yaw, self.orbit_yaw, self.orbit_pitch)
        } else {
            demo_camera(
                self.world.time,
                self.world.vehicle.pos,
                self.world.vehicle.yaw,
            )
        };
        let pose = lift_camera(&self.world, pose);
        let frame = match surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost) | Err(wgpu::SurfaceError::Outdated) => {
                if let Some(config) = &self.config {
                    surface.configure(&renderer.device, config);
                }
                return;
            }
            Err(err) => {
                eprintln!("surface: {err}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        renderer.draw(&self.world, pose, Some((&view, self.srgb)), false);
        frame.present();
    }

    #[cfg(target_arch = "wasm32")]
    fn start_web(&mut self, event_loop: &ActiveEventLoop) {
        use winit::platform::web::WindowAttributesExtWebSys;
        let canvas = web_canvas();
        let window = std::sync::Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Silt")
                        .with_canvas(Some(canvas))
                        .with_append(false)
                        .with_prevent_default(true)
                        .with_focusable(true),
                )
                .expect("window"),
        );
        window.focus_window();
        set_web_status("загрузка…");
        let gpu_window = window.clone();
        self.window = Some(window);
        wasm_bindgen_futures::spawn_local(async move {
            match build_web_gpu(gpu_window).await {
                Ok(ready) => WEB_GPU.with(|slot| *slot.borrow_mut() = Some(ready)),
                Err(err) => {
                    web_sys::console::error_1(&err.into());
                    set_web_status("ошибка GPU");
                }
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    fn take_web_gpu(&mut self) -> bool {
        let Some(ready) = WEB_GPU.with(|slot| slot.borrow_mut().take()) else {
            return false;
        };
        set_web_status(&ready.label);
        self.srgb = ready.srgb;
        self.config = Some(ready.config);
        self.surface = Some(ready.surface);
        self.renderer = Some(ready.renderer);
        true
    }
}

fn chase(pos: Vec3, yaw: f32, orbit_yaw: f32, orbit_pitch: f32) -> CameraPose {
    let yaw = yaw + orbit_yaw;
    let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = Vec3::new(forward.z, 0.0, -forward.x);
    let dist = 7.4;
    let height = 2.7 + orbit_pitch * 2.4;
    let eye = pos - forward * dist + Vec3::Y * height + right * 0.85;
    let target = pos + forward * 1.8 + Vec3::Y * 0.55;
    CameraPose {
        eye,
        target,
        ortho_width: None,
    }
}

fn lift_camera(world: &World, mut pose: CameraPose) -> CameraPose {
    let h = world.soil.sample(pose.eye.x, pose.eye.z).height;
    // Clearance stays small so the rut still can sit just above the soil.
    if pose.eye.y < h + 0.05 {
        pose.eye.y = h + 0.05;
    }
    pose
}

/// Keep the bottom of an orthographic frame above the heightfield.
/// A low eye looks through the back of the mesh and the clear color shows.
#[cfg(not(target_arch = "wasm32"))]
fn clear_ortho_eye(world: &World, mut eye: Vec3, target: Vec3, ortho: f32) -> Vec3 {
    let aspect = 1280.0_f32 / 720.0;
    let half_h = (ortho * 0.5) / aspect;
    // Only the near plane. The far end of the frame is supposed to meet the ground.
    for _ in 0..3 {
        let delta = target - eye;
        let dist = delta.length().max(0.25);
        let fwd = delta / dist;
        let up = Vec3::Y - fwd * fwd.y;
        if up.length_squared() < 1.0e-6 {
            eye.y += 0.15;
            continue;
        }
        let up = up.normalize();
        let bottom = eye + fwd * 0.28 - up * half_h;
        let g = world.soil.sample(bottom.x, bottom.z).height;
        let g_eye = world.soil.sample(eye.x, eye.z).height;
        let raise = ((g + 0.08) - bottom.y).max((g_eye + 0.22) - eye.y);
        if raise <= 0.001 {
            break;
        }
        eye.y += raise;
    }
    eye
}

fn run_window() {
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let app = App {
        window: None,
        surface: None,
        config: None,
        renderer: None,
        world: World::demo(),
        keys: Keys {
            throttle: false,
            reverse: false,
            left: false,
            right: false,
            orbit_left: false,
            orbit_right: false,
            orbit_up: false,
            orbit_down: false,
        },
        orbit_yaw: 0.0,
        orbit_pitch: 0.0,
        chase_cam: false,
        drive_bike: false,
        player_drove: false,
        acc: 0.0,
        last: Instant::now(),
        srgb: false,
    };
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).expect("run");
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct CaptureArgs {
    software: bool,
    width: u32,
    height: u32,
    fps: u32,
    seconds: f32,
    stills_only: bool,
    cc0: bool,
    states: bool,
    volume: bool,
    moisture: bool,
    video: PathBuf,
    still_start: PathBuf,
    still_puddle: PathBuf,
    still_ruts: PathBuf,
    cc0_iso: PathBuf,
    cc0_tire: PathBuf,
    cc0_puddle: PathBuf,
    states_edge: PathBuf,
    states_grip: PathBuf,
    states_water: PathBuf,
    volume_berm: PathBuf,
    volume_slope: PathBuf,
    volume_second: PathBuf,
    moisture_wet: PathBuf,
    moisture_dry: PathBuf,
    patch: bool,
    patch_roll: PathBuf,
    patch_spin: PathBuf,
    persist: bool,
    persist_first: PathBuf,
    persist_second: PathBuf,
    persist_reload: PathBuf,
    niva_tread: bool,
    niva_shed_only: bool,
    niva_realism: bool,
    niva_groove_roll: PathBuf,
    niva_groove_spin: PathBuf,
    niva_sidewall_shed: PathBuf,
    niva_sink_side: PathBuf,
    niva_sink_front: PathBuf,
    niva_tread_top: PathBuf,
    niva_tread_side: PathBuf,
    surfaces: bool,
    surfaces_proof: PathBuf,
    roll: bool,
    roll_proof: PathBuf,
    force: bool,
    force_proof: PathBuf,
    slurry: bool,
    slurry_proof: PathBuf,
    slurry_flow: bool,
    slurry_flow_proof: PathBuf,
    slurry_drag: bool,
    slurry_drag_proof: PathBuf,
    slurry_suction: bool,
    slurry_suction_proof: PathBuf,
    slurry_groove: bool,
    slurry_groove_proof: PathBuf,
    slurry_roll: bool,
    slurry_roll_proof: PathBuf,
    slurry_tear_out: bool,
    slurry_tear_out_proof: PathBuf,
    slurry_plow: bool,
    slurry_plow_proof: PathBuf,
    slurry_second_pass: bool,
    slurry_second_pass_proof: PathBuf,
    slurry_rollover: bool,
    slurry_rollover_proof: PathBuf,
    slurry_settle: bool,
    slurry_settle_proof: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
fn capture(args: &[String]) {
    let parsed = parse_capture(args);
    if parsed.software {
        prepare_software_vulkan();
    }
    let (_instance, adapter, device, queue) = open_adapter(parsed.software, None);
    let formats = render::gpu_formats(&adapter);
    eprintln!("loading cc0 ground and the dune buggy");
    let scene = cc0::load();
    let mut renderer = Renderer::new(
        device,
        queue,
        adapter.get_info().name,
        parsed.width,
        parsed.height,
        formats,
        scene,
    );
    if parsed.cc0 {
        capture_cc0(&mut renderer, &parsed);
        return;
    }
    if parsed.states {
        capture_states(&mut renderer, &parsed);
        return;
    }
    if parsed.volume {
        capture_volume(&mut renderer, &parsed);
        return;
    }
    if parsed.moisture {
        capture_moisture(&mut renderer, &parsed);
        return;
    }
    if parsed.patch {
        capture_patch(&mut renderer, &parsed);
        return;
    }
    if parsed.persist {
        capture_persist(&mut renderer, &parsed);
        return;
    }
    if parsed.niva_realism {
        capture_niva_realism(&mut renderer, &parsed);
        capture_niva_tread(&mut renderer, &parsed, true);
        return;
    }
    if parsed.niva_tread {
        capture_niva_tread(&mut renderer, &parsed, parsed.niva_shed_only);
        return;
    }
    if parsed.surfaces {
        capture_surfaces(&mut renderer, &parsed);
        return;
    }
    if parsed.roll {
        capture_roll(&mut renderer, &parsed);
        return;
    }
    if parsed.force {
        capture_force(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry {
        capture_slurry(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_flow {
        capture_slurry_flow(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_drag {
        capture_slurry_drag(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_suction {
        capture_slurry_suction(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_groove {
        capture_slurry_groove(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_roll {
        capture_slurry_roll(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_tear_out {
        capture_slurry_tear_out(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_plow {
        capture_slurry_plow(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_second_pass {
        capture_slurry_second_pass(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_rollover {
        capture_slurry_rollover(&mut renderer, &parsed);
        return;
    }
    if parsed.slurry_settle {
        capture_slurry_settle(&mut renderer, &parsed);
        return;
    }
    let mut world = World::demo();
    let frames = (parsed.seconds * parsed.fps as f32).round() as u32;
    let steps = (60.0 / parsed.fps as f32).round().max(1.0) as u32;
    let dir = std::env::temp_dir().join("silt-mud-frames");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("frame dir");
    let mut saved = [false; 3];
    let stills = [
        (0.35_f32, &parsed.still_start),
        (4.68, &parsed.still_puddle),
        (5.25, &parsed.still_ruts),
    ];
    println!(
        "capture {frames} frames at {}x{} ({} physics steps/frame)",
        parsed.width, parsed.height, steps
    );
    let t0 = Instant::now();
    if parsed.stills_only {
        let mut next = 0usize;
        while next < stills.len() && world.time < parsed.seconds + 0.05 {
            let (when, dest) = stills[next];
            if world.time + 1.0e-3 >= when {
                let pose = lift_camera(
                    &world,
                    demo_camera(world.time, world.vehicle.pos, world.vehicle.yaw),
                );
                let rgba = renderer.draw(&world, pose, None, true).expect("readback");
                write_png(dest, parsed.width, parsed.height, &rgba).expect("still");
                println!(
                    "still {} at t={:.2} pos=({:.1},{:.1}) mud={:.2} rut={:.3}",
                    dest.display(),
                    world.time,
                    world.vehicle.pos.x,
                    world.vehicle.pos.z,
                    world.vehicle.body_mud_mass(),
                    world.soil.max_rut_depth()
                );
                next += 1;
                continue;
            }
            let bike = demo_drive_bike(&world);
            world.step_bike(1.0 / 60.0, &bike);
            let input = demo_drive(&world);
            world.step(1.0 / 60.0, &input);
        }
        if next != stills.len() {
            panic!(
                "stills-only ended at t={:.2} after {next} images",
                world.time
            );
        }
        return;
    }
    for frame in 0..frames {
        let pose = lift_camera(
            &world,
            demo_camera(world.time, world.vehicle.pos, world.vehicle.yaw),
        );
        let rgba = renderer.draw(&world, pose, None, true).expect("readback");
        let path = dir.join(format!("frame_{frame:04}.png"));
        write_png(&path, parsed.width, parsed.height, &rgba).expect("png");
        for (i, (when, dest)) in stills.iter().enumerate() {
            if !saved[i] && world.time + 1.0e-4 >= *when {
                write_png(dest, parsed.width, parsed.height, &rgba).expect("still");
                println!("still {} at t={:.2}", dest.display(), world.time);
                saved[i] = true;
            }
        }
        for _ in 0..steps {
            let bike = demo_drive_bike(&world);
            world.step_bike(1.0 / 60.0, &bike);
            let input = demo_drive(&world);
            world.step(1.0 / 60.0, &input);
        }
        if frame % 30 == 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            println!(
                "frame {frame}/{frames} t={:.2}s  sim {:.1}s  mud {:.2} kg  rut {:.3} m",
                world.time,
                elapsed,
                world.vehicle.body_mud_mass(),
                world.soil.max_rut_depth()
            );
        }
    }
    if saved.iter().any(|s| !s) {
        panic!(
            "a still was not written; sim time ended at {:.2}",
            world.time
        );
    }
    let video = &parsed.video;
    if let Some(parent) = video.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let status = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-framerate",
            &parsed.fps.to_string(),
            "-i",
            &dir.join("frame_%04d.png").to_string_lossy(),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
        ])
        .arg(video)
        .status()
        .expect("ffmpeg");
    if !status.success() {
        panic!("ffmpeg failed: {status}");
    }
    println!(
        "video {}  ({:.1}s wall)",
        video.display(),
        t0.elapsed().as_secs_f32()
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_capture(args: &[String]) -> CaptureArgs {
    let mut software = false;
    let mut width = 1280u32;
    let mut height = 720u32;
    let mut fps = 30u32;
    let mut seconds = 12.0f32;
    let mut stills_only = false;
    let mut cc0 = false;
    let mut states = false;
    let mut volume = false;
    let mut moisture = false;
    let mut patch = false;
    let mut persist = false;
    let mut niva_tread = false;
    let mut niva_shed_only = false;
    let mut niva_realism = false;
    let mut surfaces = false;
    let mut roll = false;
    let mut video = PathBuf::from(format!("{MEDIA}/mud-demo.mp4"));
    let mut still_start = PathBuf::from(format!("{MEDIA}/mud-demo-start.png"));
    let mut still_puddle = PathBuf::from(format!("{MEDIA}/mud-demo-puddle.png"));
    let mut still_ruts = PathBuf::from(format!("{MEDIA}/mud-demo-ruts.png"));
    let mut cc0_iso = PathBuf::from(format!("{MEDIA}/cc0-iso.png"));
    let mut cc0_tire = PathBuf::from(format!("{MEDIA}/cc0-tire.png"));
    let mut cc0_puddle = PathBuf::from(format!("{MEDIA}/cc0-puddle.png"));
    let mut states_edge = PathBuf::from(format!("{MEDIA}/states-edge.png"));
    let mut states_grip = PathBuf::from(format!("{MEDIA}/states-grip.png"));
    let mut states_water = PathBuf::from(format!("{MEDIA}/states-water.png"));
    let mut volume_berm = PathBuf::from(format!("{MEDIA}/volume-berm.png"));
    let mut volume_slope = PathBuf::from(format!("{MEDIA}/volume-slope.png"));
    let mut volume_second = PathBuf::from(format!("{MEDIA}/volume-second.png"));
    let mut moisture_wet = PathBuf::from(format!("{MEDIA}/moisture-wet.png"));
    let mut moisture_dry = PathBuf::from(format!("{MEDIA}/moisture-dry.png"));
    let mut patch_roll = PathBuf::from(format!("{MEDIA}/patch-roll.png"));
    let mut patch_spin = PathBuf::from(format!("{MEDIA}/patch-spin.png"));
    let mut persist_first = PathBuf::from(format!("{MEDIA}/persist-first.png"));
    let mut persist_second = PathBuf::from(format!("{MEDIA}/persist-second.png"));
    let mut persist_reload = PathBuf::from(format!("{MEDIA}/persist-reload.png"));
    let mut niva_groove_roll = PathBuf::from(format!("{MEDIA}/niva-groove-roll.png"));
    let mut niva_groove_spin = PathBuf::from(format!("{MEDIA}/niva-groove-spin.png"));
    let mut niva_sidewall_shed = PathBuf::from(format!("{MEDIA}/niva-sidewall-shed.png"));
    let niva_sink_side = PathBuf::from(format!("{MEDIA}/niva-sink-side.png"));
    let niva_sink_front = PathBuf::from(format!("{MEDIA}/niva-sink-front.png"));
    let niva_tread_top = PathBuf::from(format!("{MEDIA}/niva-tread-top.png"));
    let niva_tread_side = PathBuf::from(format!("{MEDIA}/niva-tread-side.png"));
    let mut surfaces_proof = PathBuf::from(format!("{MEDIA}/surfaces-proof.png"));
    let mut roll_proof = PathBuf::from(format!("{MEDIA}/roll-proof.png"));
    let mut force = false;
    let mut force_proof = PathBuf::from(format!("{MEDIA}/force-proof.png"));
    let mut slurry = false;
    let mut slurry_proof = PathBuf::from(format!("{MEDIA}/slurry-mix-proof.png"));
    let mut slurry_flow = false;
    let mut slurry_flow_proof = PathBuf::from(format!("{MEDIA}/slurry-flow-proof.png"));
    let mut slurry_drag = false;
    let mut slurry_drag_proof = PathBuf::from(format!("{MEDIA}/slurry-drag-proof.png"));
    let mut slurry_suction = false;
    let mut slurry_suction_proof = PathBuf::from(format!("{MEDIA}/slurry-suction-proof.png"));
    let mut slurry_groove = false;
    let mut slurry_groove_proof = PathBuf::from(format!("{MEDIA}/slurry-groove-proof.png"));
    let mut slurry_roll = false;
    let mut slurry_roll_proof = PathBuf::from(format!("{MEDIA}/slurry-roll-proof.png"));
    let mut slurry_tear_out = false;
    let mut slurry_tear_out_proof =
        PathBuf::from(format!("{MEDIA}/slurry-tear-out-proof.png"));
    let mut slurry_plow = false;
    let mut slurry_plow_proof = PathBuf::from(format!("{MEDIA}/slurry-plow-proof.png"));
    let mut slurry_second_pass = false;
    let mut slurry_second_pass_proof =
        PathBuf::from(format!("{MEDIA}/slurry-second-pass-proof.png"));
    let mut slurry_rollover = false;
    let mut slurry_rollover_proof =
        PathBuf::from(format!("{MEDIA}/slurry-rollover-proof.png"));
    let mut slurry_settle = false;
    let mut slurry_settle_proof =
        PathBuf::from(format!("{MEDIA}/slurry-settle-proof.png"));
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let val = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_else(|| {
                panic!("missing value after {flag}");
            })
        };
        match flag {
            "--software" => software = true,
            "--width" => width = val(&mut i).parse().expect("width"),
            "--height" => height = val(&mut i).parse().expect("height"),
            "--fps" => fps = val(&mut i).parse().expect("fps"),
            "--seconds" => seconds = val(&mut i).parse().expect("seconds"),
            "--stills-only" => stills_only = true,
            "--cc0" => cc0 = true,
            "--states" => states = true,
            "--volume" => volume = true,
            "--moisture" => moisture = true,
            "--patch" => patch = true,
            "--persist" => persist = true,
            "--niva-tread" => niva_tread = true,
            "--niva-shed-only" => {
                niva_tread = true;
                niva_shed_only = true;
            }
            "--niva-realism" => niva_realism = true,
            "--surfaces" => surfaces = true,
            "--surfaces-proof" => surfaces_proof = PathBuf::from(val(&mut i)),
            "--roll-proof" => {
                roll = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    roll_proof = PathBuf::from(val(&mut i));
                }
            }
            "--force-proof" => {
                force = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    force_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-proof" => {
                slurry = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-flow-proof" => {
                slurry_flow = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_flow_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-drag-proof" => {
                slurry_drag = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_drag_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-suction-proof" => {
                slurry_suction = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_suction_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-groove-proof" => {
                slurry_groove = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_groove_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-roll-proof" => {
                slurry_roll = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_roll_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-tear-out-proof" => {
                slurry_tear_out = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_tear_out_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-plow-proof" => {
                slurry_plow = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_plow_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-second-pass-proof" => {
                slurry_second_pass = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_second_pass_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-rollover-proof" => {
                slurry_rollover = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_rollover_proof = PathBuf::from(val(&mut i));
                }
            }
            "--slurry-settle-proof" => {
                slurry_settle = true;
                if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                    slurry_settle_proof = PathBuf::from(val(&mut i));
                }
            }
            "--niva-groove-roll" => niva_groove_roll = PathBuf::from(val(&mut i)),
            "--niva-groove-spin" => niva_groove_spin = PathBuf::from(val(&mut i)),
            "--niva-sidewall-shed" => niva_sidewall_shed = PathBuf::from(val(&mut i)),
            "--persist-first" => persist_first = PathBuf::from(val(&mut i)),
            "--persist-second" => persist_second = PathBuf::from(val(&mut i)),
            "--persist-reload" => persist_reload = PathBuf::from(val(&mut i)),
            "--patch-roll" => patch_roll = PathBuf::from(val(&mut i)),
            "--patch-spin" => patch_spin = PathBuf::from(val(&mut i)),
            "--moisture-wet" => moisture_wet = PathBuf::from(val(&mut i)),
            "--moisture-dry" => moisture_dry = PathBuf::from(val(&mut i)),
            "--volume-berm" => volume_berm = PathBuf::from(val(&mut i)),
            "--volume-slope" => volume_slope = PathBuf::from(val(&mut i)),
            "--volume-second" => volume_second = PathBuf::from(val(&mut i)),
            "--states-edge" => states_edge = PathBuf::from(val(&mut i)),
            "--states-grip" => states_grip = PathBuf::from(val(&mut i)),
            "--states-water" => states_water = PathBuf::from(val(&mut i)),
            "--cc0-iso" => cc0_iso = PathBuf::from(val(&mut i)),
            "--cc0-tire" => cc0_tire = PathBuf::from(val(&mut i)),
            "--cc0-puddle" => cc0_puddle = PathBuf::from(val(&mut i)),
            "--video" => video = PathBuf::from(val(&mut i)),
            "--still-start" => still_start = PathBuf::from(val(&mut i)),
            "--still-puddle" => still_puddle = PathBuf::from(val(&mut i)),
            "--still-ruts" => still_ruts = PathBuf::from(val(&mut i)),
            other => panic!("unknown capture flag {other}"),
        }
        i += 1;
    }
    let _ = Path::new(".");
    CaptureArgs {
        software,
        width,
        height,
        fps,
        seconds,
        stills_only,
        cc0,
        states,
        volume,
        moisture,
        patch,
        video,
        still_start,
        still_puddle,
        still_ruts,
        cc0_iso,
        cc0_tire,
        cc0_puddle,
        states_edge,
        states_grip,
        states_water,
        volume_berm,
        volume_slope,
        volume_second,
        moisture_wet,
        moisture_dry,
        patch_roll,
        patch_spin,
        persist,
        persist_first,
        persist_second,
        persist_reload,
        niva_tread,
        niva_shed_only,
        niva_realism,
        niva_groove_roll,
        niva_groove_spin,
        niva_sidewall_shed,
        niva_sink_side,
        niva_sink_front,
        niva_tread_top,
        niva_tread_side,
        surfaces,
        surfaces_proof,
        roll,
        roll_proof,
        force,
        force_proof,
        slurry,
        slurry_proof,
        slurry_flow,
        slurry_flow_proof,
        slurry_drag,
        slurry_drag_proof,
        slurry_suction,
        slurry_suction_proof,
        slurry_groove,
        slurry_groove_proof,
        slurry_roll,
        slurry_roll_proof,
        slurry_tear_out,
        slurry_tear_out_proof,
        slurry_plow,
        slurry_plow_proof,
        slurry_second_pass,
        slurry_second_pass_proof,
        slurry_rollover,
        slurry_rollover_proof,
        slurry_settle,
        slurry_settle_proof,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_states(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::SoilField;
    let mut world = World::flat(0.16);
    world.soil = SoilField::gallery();
    world.vehicle.pos = Vec3::new(14.0, 0.4, 14.0);
    write_cc0_still(
        renderer,
        &world,
        "edge",
        &parsed.states_edge,
        parsed,
        states_edge_camera(),
    );
    // Right-rear tire straddles gravel and wet mud.
    world.vehicle.pos = Vec3::new(3.16, 0.62, 1.18);
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    let h = world.soil.sample(world.vehicle.pos.x, world.vehicle.pos.z).height;
    world.vehicle.pos.y = h + 0.62;
    write_cc0_still(
        renderer,
        &world,
        "grip",
        &parsed.states_grip,
        parsed,
        states_grip_camera(&world),
    );
    world.vehicle.pos = Vec3::new(14.0, 0.4, 14.0);
    write_cc0_still(
        renderer,
        &world,
        "water",
        &parsed.states_water,
        parsed,
        states_water_camera(),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn states_edge_camera() -> CameraPose {
    // Across the rut: near berm, torn lip, cut floor, far berm.
    CameraPose {
        eye: Vec3::new(-1.35, 0.55, 1.25),
        target: Vec3::new(0.12, -0.01, 0.05),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn states_grip_camera(world: &World) -> CameraPose {
    let local = world.vehicle.wheel_center_local(3);
    let center = world.vehicle.pos + world.vehicle.orientation() * local;
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    CameraPose {
        eye: center + right * 1.35 + Vec3::Y * 0.48 - fwd * 0.42,
        target: center - Vec3::Y * 0.16,
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn states_water_camera() -> CameraPose {
    // Down into the groove, so the film and the slurry are the floor mesh
    // and not an edge-on plane. Clay film nearer, fresh-cut slurry beyond.
    CameraPose {
        eye: Vec3::new(-0.42, 2.05, -1.72),
        target: Vec3::new(0.02, -0.08, -1.22),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_volume(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::SoilField;

    let mut world = World::flat(0.45);
    world.bike.pos = Vec3::new(14.0, 0.4, -14.0);
    park_off(&mut world);

    world.soil = SoilField::berm_scene(false);
    print_volumes("berm", &world.soil);
    write_cc0_still(
        renderer,
        &world,
        "berm",
        &parsed.volume_berm,
        parsed,
        volume_berm_camera(),
    );

    world.soil = SoilField::berm_scene(true);
    print_volumes("second", &world.soil);
    write_cc0_still(
        renderer,
        &world,
        "second",
        &parsed.volume_second,
        parsed,
        volume_second_camera(),
    );

    world.soil = SoilField::slope_scene();
    print_volumes("slope", &world.soil);
    // The grooves stay on the lower face. The buggy is up the ramp, so the
    // rut is not under the tires.
    let grade = 0.42_f32;
    let x = 1.55;
    let z = 0.0;
    let h = world.soil.sample(x, z).height;
    world.vehicle.pos = Vec3::new(x, h + 0.62, z);
    world.vehicle.yaw = std::f32::consts::FRAC_PI_2;
    world.vehicle.pitch = -grade.atan();
    world.vehicle.roll = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    write_cc0_still(
        renderer,
        &world,
        "slope",
        &parsed.volume_slope,
        parsed,
        volume_slope_camera(),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_persist(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::SoilField;

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::flat(0.45);
    world.soil = SoilField::mud_pad();
    let from = Vec3::new(0.0, 0.0, -1.15);
    let to = Vec3::new(0.0, 0.0, 1.15);
    world.soil.roll_lane(from, to, 0.09, 0.0);
    let saved = world.soil.save_state();
    park_beside(&mut world);
    print_persist("pass1", &world.soil);
    write_cc0_still(
        renderer,
        &world,
        "persist-first",
        &parsed.persist_first,
        parsed,
        persist_camera(),
    );

    world.soil.age_passes(0.55);
    world.soil.roll_lane(from, to, 0.09, 0.0);
    park_beside(&mut world);
    print_persist("pass2", &world.soil);
    write_cc0_still(
        renderer,
        &world,
        "persist-second",
        &parsed.persist_second,
        parsed,
        persist_camera(),
    );

    // Fresh world, then the pass-1 snapshot. Reset restores the spawn pose
    // and leaves the cut. The Niva is parked beside that surviving rut.
    let mut reloaded = World::flat(0.45);
    reloaded.soil = SoilField::load_state(&saved).expect("soil snapshot");
    reloaded.reset_vehicle();
    park_beside(&mut reloaded);
    print_persist("reload", &reloaded.soil);
    write_cc0_still(
        renderer,
        &reloaded,
        "persist-reload",
        &parsed.persist_reload,
        parsed,
        persist_camera(),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn print_persist(label: &str, soil: &silt::soil::SoilField) {
    let v = soil.volumes();
    let depth = soil.rut_near(0.0, 0.0);
    println!(
        "{label} depth {:.4} m  cut {:.4} m3  placed {:.4} m3  berm {:.4} m3  clods {:.4} m3  \
         max rut {:.4} m  berm height {:.4} m",
        depth, v.cut, v.placed, v.berm, v.clods, v.max_rut, v.max_berm
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn persist_camera() -> CameraPose {
    // Same three-quarter for every still, so the second groove reads as deeper.
    // Low enough to see the torn lip, high enough to keep the Niva in frame.
    CameraPose {
        eye: Vec3::new(-3.15, 1.78, 2.55),
        target: Vec3::new(0.05, 0.72, -0.05),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn park_beside(world: &mut World) {
    // Off the lane. The tires are not in the cut, so the picture does not
    // restamp the rut the solver just wrote.
    world.bike.pos = Vec3::new(18.0, 0.4, 18.0);
    world.bike.vel = Vec3::ZERO;
    let x = 2.15;
    let z = -0.2;
    let h = world.soil.sample(x, z).height;
    world.vehicle.pos = Vec3::new(x, h + 0.66, z);
    world.vehicle.yaw = -0.55;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    world.vehicle.yaw_rate = 0.0;
    world.vehicle.pitch_rate = 0.0;
    world.vehicle.roll_rate = 0.0;
}

#[cfg(not(target_arch = "wasm32"))]
fn park_off(world: &mut World) {
    world.vehicle.pos = Vec3::new(14.0, 0.4, 14.0);
    world.vehicle.vel = Vec3::ZERO;
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.yaw_rate = 0.0;
    world.vehicle.pitch_rate = 0.0;
    world.vehicle.roll_rate = 0.0;
}

#[cfg(not(target_arch = "wasm32"))]
fn print_volumes(label: &str, soil: &silt::soil::SoilField) {
    let v = soil.volumes();
    println!(
        "{label} cut {:.4} m3  placed {:.4} m3  berm {:.4} m3  clods {:.4} m3  rut {:.4} m  berm height {:.4} m",
        v.cut, v.placed, v.berm, v.clods, v.max_rut, v.max_berm
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_patch(renderer: &mut Renderer, parsed: &CaptureArgs) {
    let mut world = World::flat(0.72);
    world.bike.pos = Vec3::new(18.0, 0.5, 18.0);
    world.bike.vel = Vec3::ZERO;
    // Standing water hides the tread. The clay stays wet, so it still sticks.
    for w in &mut world.soil.water {
        *w = 0.0;
    }
    for loose in &mut world.soil.loose {
        *loose = loose.max(0.04);
    }
    let dt = 1.0 / 60.0;
    let roll = Input {
        throttle: 0.36,
        brake: 0.0,
        steer: 0.0,
    };
    for _ in 0..360 {
        world.step(dt, &roll);
    }
    print_patch("roll", &world);
    // The roll still is the tread. Flying chunks belong to the spin frame.
    let hidden: Vec<bool> = world.particles.items.iter().map(|p| p.alive).collect();
    for p in &mut world.particles.items {
        p.alive = false;
    }
    write_cc0_still(
        renderer,
        &world,
        "roll",
        &parsed.patch_roll,
        parsed,
        patch_camera(&world, 3),
    );
    for (p, was) in world.particles.items.iter_mut().zip(hidden) {
        p.alive = was;
    }

    // Hold the chassis and let the tread spin. The patch, not a scripted
    // force, is what drops the grip and pulls the mud back out.
    for _ in 0..30 {
        world.step(
            dt,
            &Input {
                throttle: 0.0,
                brake: 1.0,
                steer: 0.0,
            },
        );
        world.vehicle.vel.x = 0.0;
        world.vehicle.vel.z = 0.0;
    }
    let spin = Input {
        throttle: 1.0,
        brake: 0.0,
        steer: 0.0,
    };
    for _ in 0..70 {
        world.step(dt, &spin);
        world.vehicle.vel.x = 0.0;
        world.vehicle.vel.z = 0.0;
        world.vehicle.yaw_rate *= 0.2;
    }
    print_patch("spin", &world);
    write_cc0_still(
        renderer,
        &world,
        "spin",
        &parsed.patch_spin,
        parsed,
        patch_camera(&world, 3),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_niva_tread(renderer: &mut Renderer, parsed: &CaptureArgs, shed_only: bool) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::flat(0.72);
    world.bike.pos = Vec3::new(18.0, 0.5, 18.0);
    world.bike.vel = Vec3::ZERO;
    for w in &mut world.soil.water {
        *w = 0.0;
    }
    for loose in &mut world.soil.loose {
        *loose = loose.max(0.06);
    }
    let before = world.soil.volumes();
    let dt = 1.0 / 60.0;
    // The mud bog either drags the carcass or lights the tires up. Prescribe a
    // matched roll so the contact patch, not a throttle hunt, packs the tread.
    // The soil stamp and the tread ODE are unchanged.
    let mut steps = 0;
    while steps < 220 {
        prescribe_roll(&mut world, 1.15);
        world.step(dt, &Input::default());
        steps += 1;
        if steps % 30 == 0 {
            let w = world.vehicle.wheels[3];
            println!(
                "roll t={:.2} slip {:.2} speed {:.2} groove {:.2} sidewall {:.2} coat {:.5}",
                world.time, w.slip, world.vehicle.speed(), w.groove, w.sidewall, world.coat_in_rut
            );
        }
        let w = world.vehicle.wheels[3];
        if steps > 80 && w.groove > 0.75 && w.sidewall > 0.60 && w.slip.abs() < 0.22 {
            break;
        }
    }
    print_patch("niva-roll", &world);
    if !shed_only {
        silence_chunks(&mut world);
        write_cc0_still(
            renderer,
            &world,
            "niva-roll",
            &parsed.niva_groove_roll,
            parsed,
            niva_tread_camera(&world, 3),
        );
    }

    for _ in 0..36 {
        world.step(
            dt,
            &Input {
                throttle: 0.0,
                brake: 1.0,
                steer: 0.0,
            },
        );
        hold_chassis(&mut world);
    }
    let spin = Input {
        throttle: 1.0,
        brake: 0.0,
        steer: 0.0,
    };
    let mut spun = 0;
    while spun < 200 {
        world.step(dt, &spin);
        hold_chassis(&mut world);
        spun += 1;
        if spun > 24 && world.vehicle.wheels[3].groove < 0.22 {
            break;
        }
    }
    print_patch("niva-spin", &world);
    if !shed_only {
        silence_chunks(&mut world);
        write_cc0_still(
            renderer,
            &world,
            "niva-spin",
            &parsed.niva_groove_spin,
            parsed,
            niva_tread_camera(&world, 3),
        );
    }

    while spun < 420 {
        let w = world.vehicle.wheels[3];
        if w.sidewall < 0.18 && world.coat_in_rut > 1.0e-4 {
            break;
        }
        world.step(dt, &spin);
        hold_chassis(&mut world);
        spun += 1;
    }
    print_patch("niva-shed", &world);
    let after = world.soil.volumes();
    let pile = rut_pile(&world, 3);
    let rear = world.vehicle.pos
        + world.vehicle.orientation() * world.vehicle.wheel_center_local(3);
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let focus = spoil_focus(&world, 3, rear, fwd, right);
    let focus_i = world.soil.idx_at(focus.x, focus.z);
    let (placed_h, depth_h) = focus_i
        .map(|i| {
            (
                world.soil.placed[i],
                (world.soil.initial_height[i] - world.soil.bed[i]).max(0.0),
            )
        })
        .unwrap_or((0.0, 0.0));
    println!(
        "niva shed coat_in_rut {:.5} m3  placed {:.5} -> {:.5}  cut {:.5} -> {:.5}  placed-cut {:.5} -> {:.5}  \
         pile {:.3} m on {} rut cells  ridge {:.3} m in a {:.3} m rut",
        world.coat_in_rut,
        before.placed,
        after.placed,
        before.cut,
        after.cut,
        before.placed - before.cut,
        after.placed - after.cut,
        pile.0,
        pile.1,
        placed_h,
        depth_h
    );
    silence_chunks(&mut world);
    // The photo is the spoil. Lug displacement on the same cells hides the ridge.
    reveal_spoil(&mut world, 3);
    write_cc0_still(
        renderer,
        &world,
        "niva-shed",
        &parsed.niva_sidewall_shed,
        parsed,
        niva_shed_camera(&world, 3),
    );
}

/// The coat is placed height in the rut. Drop the tread mask on those cells
/// and mark them as spoil so the mesh shows a ridge, not another lug print.
#[cfg(not(target_arch = "wasm32"))]
fn reveal_spoil(world: &mut World, wheel: usize) {
    use silt::soil::SurfaceKind;
    let center = world.vehicle.pos
        + world.vehicle.orientation() * world.vehicle.wheel_center_local(wheel);
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let soil = &mut world.soil;
    let half = soil.size * 0.5;
    let res = soil.res as isize;
    let reach = 4.2_f32;
    let i0 = (((center.x - reach) + half) / soil.cell).floor() as isize;
    let i1 = (((center.x + reach) + half) / soil.cell).floor() as isize;
    let j0 = (((center.z - reach) + half) / soil.cell).floor() as isize;
    let j1 = (((center.z + reach) + half) / soil.cell).floor() as isize;
    for j in j0.max(0)..j1.min(res - 1) {
        for i in i0.max(0)..i1.min(res - 1) {
            let cx = (i as f32 + 0.5) * soil.cell - half;
            let cz = (j as f32 + 0.5) * soil.cell - half;
            let rel_x = cx - center.x;
            let rel_z = cz - center.z;
            let along = rel_x * fwd.x + rel_z * fwd.z;
            let lat = rel_x * right.x + rel_z * right.z;
            let behind = -along;
            if !(1.05..=3.8).contains(&behind) || lat.abs() > 0.16 {
                continue;
            }
            let idx = j as usize * soil.res + i as usize;
            let depth = (soil.initial_height[idx] - soil.bed[idx]).max(0.0);
            // Only the thick ridge. A thin smear of placed soil was the flat sheet.
            if depth > 0.02 && soil.placed[idx] > 0.025 {
                soil.tread_amp[idx] = 0.0;
                soil.kind[idx] = SurfaceKind::PlacedClod as u8;
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn prescribe_roll(world: &mut World, speed: f32) {
    use silt::vehicle::WHEEL_RADIUS;
    world.vehicle.yaw = 0.0;
    world.vehicle.yaw_rate = 0.0;
    world.vehicle.vel.x = 0.0;
    world.vehicle.vel.z = speed;
    let omega = speed * 1.08 / WHEEL_RADIUS;
    for w in &mut world.vehicle.wheels {
        w.omega = omega;
    }
}

/// Tallest placed thickness on rut cells in the sidewall deposit window.
fn rut_pile(world: &World, wheel: usize) -> (f32, u32) {
    let vehicle = &world.vehicle;
    let center = vehicle.pos + vehicle.orientation() * vehicle.wheel_center_local(wheel);
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let soil = &world.soil;
    let half = soil.size * 0.5;
    let mut best = 0.0_f32;
    let mut n = 0u32;
    let res = soil.res as isize;
    let reach = 1.2_f32;
    let i0 = (((center.x - reach) + half) / soil.cell).floor() as isize;
    let i1 = (((center.x + reach) + half) / soil.cell).floor() as isize;
    let j0 = (((center.z - reach) + half) / soil.cell).floor() as isize;
    let j1 = (((center.z + reach) + half) / soil.cell).floor() as isize;
    for j in j0.max(0)..j1.min(res - 1) {
        for i in i0.max(0)..i1.min(res - 1) {
            let cx = (i as f32 + 0.5) * soil.cell - half;
            let cz = (j as f32 + 0.5) * soil.cell - half;
            let rel_x = cx - center.x;
            let rel_z = cz - center.z;
            let along = rel_x * fwd.x + rel_z * fwd.z;
            let lat = rel_x * right.x + rel_z * right.z;
            let behind = -along;
            if !(1.05..=3.6).contains(&behind) || lat.abs() > 0.14 {
                continue;
            }
            let idx = j as usize * soil.res + i as usize;
            let depth = (soil.initial_height[idx] - soil.bed[idx]).max(0.0);
            if depth > 0.01 {
                n += 1;
                best = best.max(soil.placed[idx]);
            }
        }
    }
    (best, n)
}

#[cfg(not(target_arch = "wasm32"))]
fn hold_chassis(world: &mut World) {
    world.vehicle.vel.x = 0.0;
    world.vehicle.vel.z = 0.0;
    world.vehicle.yaw_rate *= 0.2;
}

#[cfg(not(target_arch = "wasm32"))]
fn silence_chunks(world: &mut World) {
    for p in &mut world.particles.items {
        p.alive = false;
    }
}

/// Drop solid clods from the still. Slurry ribbons, sheets, and spray stay.
#[cfg(not(target_arch = "wasm32"))]
fn silence_soil_clods(world: &mut World) {
    use silt::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON, KIND_SLURRY_SHEET};
    for p in &mut world.particles.items {
        if p.kind == KIND_SLURRY_RIBBON || p.kind == KIND_SLURRY_SHEET || p.kind == KIND_SLURRY_DROP
        {
            continue;
        }
        p.alive = false;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn niva_tread_camera(world: &World, wheel: usize) -> CameraPose {
    let vehicle = &world.vehicle;
    let local = vehicle.wheel_center_local(wheel);
    let center = vehicle.pos + vehicle.orientation() * local;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let side = if local.x >= 0.0 { 1.0 } else { -1.0 };
    // Behind the shoulder, so the crown grooves face the camera and the
    // sidewall coat is the near edge rather than the whole frame.
    let eye = center - fwd * 1.02 + right * side * 0.46 + Vec3::Y * 0.30;
    CameraPose {
        eye,
        target: center - Vec3::Y * 0.08 + right * side * 0.02,
        ortho_width: Some(0.88),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn niva_shed_camera(world: &World, wheel: usize) -> CameraPose {
    let vehicle = &world.vehicle;
    let local = vehicle.wheel_center_local(wheel);
    let center = vehicle.pos + vehicle.orientation() * local;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let side = if local.x >= 0.0 { 1.0 } else { -1.0 };
    let pile = spoil_focus(world, wheel, center, fwd, right);
    let lip = world
        .soil
        .sample(pile.x + right.x * side * 0.55, pile.z + right.z * side * 0.55)
        .height;
    // Onto the mound, steep enough to stay sharp, wide enough to show it
    // sitting in the rut under the lip.
    let look = Vec3::new(pile.x, (pile.y + lip) * 0.5, pile.z);
    let eye = pile - fwd * 0.22 + right * side * 0.40;
    let eye = Vec3::new(eye.x, lip + 0.58, eye.z);
    let ortho = 0.70;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

/// Middle of the shed ridge, not the single cell at the tail.
/// The tail bilinear-ramps to empty rut and reads as a flat card.
#[cfg(not(target_arch = "wasm32"))]
fn spoil_focus(world: &World, _wheel: usize, center: Vec3, fwd: Vec3, right: Vec3) -> Vec3 {
    let soil = &world.soil;
    let half = soil.size * 0.5;
    let res = soil.res as isize;
    let reach = 4.2_f32;
    let i0 = (((center.x - reach) + half) / soil.cell).floor() as isize;
    let i1 = (((center.x + reach) + half) / soil.cell).floor() as isize;
    let j0 = (((center.z - reach) + half) / soil.cell).floor() as isize;
    let j1 = (((center.z + reach) + half) / soil.cell).floor() as isize;
    let mut piled: Vec<(f32, f32, Vec3)> = Vec::new();
    for j in j0.max(0)..j1.min(res - 1) {
        for i in i0.max(0)..i1.min(res - 1) {
            let cx = (i as f32 + 0.5) * soil.cell - half;
            let cz = (j as f32 + 0.5) * soil.cell - half;
            let rel_x = cx - center.x;
            let rel_z = cz - center.z;
            let along = rel_x * fwd.x + rel_z * fwd.z;
            let lat = rel_x * right.x + rel_z * right.z;
            let behind = -along;
            if !(1.2..=3.4).contains(&behind) || lat.abs() > 0.16 {
                continue;
            }
            let idx = j as usize * soil.res + i as usize;
            let depth = (soil.initial_height[idx] - soil.bed[idx]).max(0.0);
            let placed = soil.placed[idx];
            if depth > 0.02 && placed > 0.02 {
                piled.push((behind, placed, Vec3::new(cx, soil.height[idx], cz)));
            }
        }
    }
    if piled.is_empty() {
        return center - fwd * 1.6;
    }
    let min_b = piled.iter().map(|c| c.0).fold(f32::MAX, f32::min);
    let max_b = piled.iter().map(|c| c.0).fold(0.0_f32, f32::max);
    let mid = (min_b + max_b) * 0.5;
    let mut best = piled[0].2;
    let mut best_s = f32::MAX;
    for (behind, placed, pos) in &piled {
        let score = (behind - mid).abs() + (placed - 0.033).abs() * 8.0;
        if score < best_s {
            best_s = score;
            best = *pos;
        }
    }
    best
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_roll(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let roll = world.roll_in_clay();
    println!(
        "roll displaced {:.4} m3  thrown {:.4} m3  berm {:.4} m3  body {:.4} m3  tires {:.4} m3  \
         bed {:.4} m  bed_after {:.4} m  placed {:.4} m3  airborne {:.4} m3  pose roll {:.2} yaw {:.2}",
        roll.displaced,
        roll.thrown,
        roll.berm,
        roll.body,
        roll.tires,
        roll.bed,
        roll.bed_after,
        roll.placed,
        roll.airborne,
        world.vehicle.roll,
        world.vehicle.yaw
    );
    write_cc0_still(
        renderer,
        &world,
        "roll",
        &parsed.roll_proof,
        parsed,
        roll_camera(&world),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn roll_camera(world: &World) -> CameraPose {
    let yaw = world.vehicle.yaw;
    let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    // The settled Niva sits beside the furrow. A wide shot on the body
    // leaves the trench and the ridge as a wrinkle in bare ground.
    let soil = &world.soil;
    let half = soil.size * 0.5;
    let mut sx = 0.0_f32;
    let mut sz = 0.0_f32;
    let mut sw = 0.0_f32;
    for idx in 0..soil.placed.len() {
        let depth = (soil.initial_height[idx] - soil.bed[idx]).max(0.0);
        let placed = soil.placed[idx];
        let w = depth.max(placed);
        if w < 0.05 {
            continue;
        }
        let i = idx % soil.res;
        let j = idx / soil.res;
        sx += ((i as f32 + 0.5) * soil.cell - half) * w;
        sz += ((j as f32 + 0.5) * soil.cell - half) * w;
        sw += w;
    }
    let p = world.vehicle.pos;
    let focus = if sw > 1.0e-4 {
        Vec3::new(sx / sw, 0.0, sz / sw)
    } else {
        p
    };
    let look = Vec3::new(
        focus.x * 0.78 + p.x * 0.22,
        0.08,
        focus.z * 0.78 + p.z * 0.22,
    );
    // From the nose, low and close, so the trench and the ridge fill the
    // ground beside the rolled Niva instead of sitting in a wide empty frame.
    let eye = look + fwd * 2.85 - right * 0.55 + Vec3::Y * 0.92;
    let ortho = 4.15;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

/// The Niva on its side beside a slurry trench. The berm is the ridge on
/// the far flank. The numbers are the cut, the berm, the throw, and the
/// slurry left in the rut.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_roll(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let roll = world.roll_and_slide_in_slurry();
    println!(
        "slurry-roll cut {:.4} m3  berm {:.4} m3  thrown {:.4} m3  slurry {:.4} m3  \
         placed {:.4} m3  airborne {:.4} m3  sheets {:.4} m3  ribbons {:.4} m3  \
         bed {:.4} m  berm_h {:.4} m  sum {:.4} m3",
        roll.cut,
        roll.berm,
        roll.thrown,
        roll.slurry,
        roll.placed,
        roll.airborne,
        roll.sheets,
        roll.ribbons,
        roll.bed,
        roll.berm_height,
        roll.berm + roll.thrown + roll.slurry
    );
    let lines = [
        format!("cut {:.4}", roll.cut),
        format!("berm {:.4}", roll.berm),
        format!("thrown {:.4}", roll.thrown),
        format!("slurry {:.4}", roll.slurry),
        format!("placed {:.4}", roll.placed),
    ];
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let pose = lift_camera(&world, slurry_roll_camera(&world));
    let mut rgba = renderer.draw(&world, pose, None, true).expect("readback");
    stamp_proof_lines(&mut rgba, parsed.width, parsed.height, &line_refs);
    if let Some(parent) = parsed.slurry_roll_proof.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(
        &parsed.slurry_roll_proof,
        parsed.width,
        parsed.height,
        &rgba,
    )
    .expect("slurry roll still");
    println!(
        "cc0 slurry-roll {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        parsed.slurry_roll_proof.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

/// The Niva leaned onto its side in a deep slurry pool. The dent, bow, berm,
/// and throw are visible. Overlay numbers match the PR body.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_rollover(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let out = world.rollover_in_deep_slurry();
    silence_soil_clods(&mut world);
    println!(
        "slurry-rollover pool {:.4} berm {:.4} slurry {:.4} bow {:.4} shed {:.4} sum {:.4} \
         ribbons {:.4} sheets {:.4} dent {:.4} clay {:.0} silt {:.0} sand {:.0}",
        out.pool,
        out.berm,
        out.slurry,
        out.bow_m3,
        out.shed,
        out.sum,
        out.ribbons,
        out.sheets,
        out.dent,
        out.drag_clay,
        out.drag_silt,
        out.drag_sand,
    );
    let lines = [
        format!("pool {:.4} m3", out.pool),
        format!("berm {:.4} m3", out.berm),
        format!("slurry {:.4} m3", out.slurry),
        format!("bow {:.4} m3", out.bow_m3),
        format!("shed {:.4} m3", out.shed),
        format!("sum {:.4} m3", out.sum),
        format!("ribbons {:.4} m3", out.ribbons),
        format!("sheets {:.4} m3", out.sheets),
        format!("dent {:.4} m", out.dent),
        format!("clay {:.0} N", out.drag_clay),
        format!("silt {:.0} N", out.drag_silt),
        format!("sand {:.0} N", out.drag_sand),
    ];
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let pose = slurry_rollover_camera(&world);
    let mut rgba = renderer.draw(&world, pose, None, true).expect("readback");
    stamp_proof_lines(&mut rgba, parsed.width, parsed.height, &line_refs);
    if let Some(parent) = parsed.slurry_rollover_proof.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(
        &parsed.slurry_rollover_proof,
        parsed.width,
        parsed.height,
        &rgba,
    )
    .expect("slurry rollover still");
    println!(
        "cc0 slurry-rollover {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        parsed.slurry_rollover_proof.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

#[cfg(not(target_arch = "wasm32"))]
/// Second pass through the pool, then slurry settle. Bow and berm numbers are
/// just-after versus settled; sum is the conserved pool total.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_settle(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let out = world.settle_slurry_after_second_pass(3.6, 8.0);
    silence_soil_clods(&mut world);
    println!(
        "slurry-settle pool {:.4} berm {:.4}->{:.4} bow {:.4}->{:.4} rut {:.4}->{:.4} \
         sum {:.4} groove {:.4} wake {:.4} clay {:.3} silt {:.3} sand {:.3}",
        out.pool,
        out.berm_before,
        out.berm_after,
        out.bow_before,
        out.bow_after,
        out.rut_before,
        out.rut_after,
        out.sum,
        out.groove,
        out.wake_m3,
        out.pace_clay,
        out.pace_silt,
        out.pace_sand,
    );
    let lines = [
        format!("pool {:.4} m3", out.pool),
        format!("berm {:.4}->{:.4} m3", out.berm_before, out.berm_after),
        format!("bow {:.4}->{:.4} m3", out.bow_before, out.bow_after),
        format!("rut {:.4}->{:.4} m3", out.rut_before, out.rut_after),
        format!("sum {:.4} m3", out.sum),
        format!("groove {:.4} m3", out.groove),
        format!("wake {:.4} m3", out.wake_m3),
        format!("clay {:.3} m/s", out.pace_clay),
        format!("silt {:.3} m/s", out.pace_silt),
        format!("sand {:.3} m/s", out.pace_sand),
    ];
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let pose = slurry_plow_camera(&world);
    let mut rgba = renderer.draw(&world, pose, None, true).expect("readback");
    stamp_proof_lines(&mut rgba, parsed.width, parsed.height, &line_refs);
    if let Some(parent) = parsed.slurry_settle_proof.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(
        &parsed.slurry_settle_proof,
        parsed.width,
        parsed.height,
        &rgba,
    )
    .expect("slurry settle still");
    println!(
        "cc0 slurry-settle {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        parsed.slurry_settle_proof.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_rollover_camera(world: &World) -> CameraPose {
    let vehicle = &world.vehicle;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let driver = -right;
    let ori = vehicle.orientation();
    let rear = vehicle.pos + ori * vehicle.wheel_center_local(2);
    let front = vehicle.pos + ori * vehicle.wheel_center_local(0);
    let bow_x = front.x + fwd.x * 0.48;
    let bow_z = front.z + fwd.z * 0.48;
    let mud_h = world.soil.sample(rear.x, rear.z).height;
    let bow_h = world.soil.sample(bow_x, bow_z).height;
    let look = Vec3::new(
        rear.x * 0.35 + front.x * 0.25 + bow_x * 0.40,
        mud_h + 0.04,
        rear.z * 0.35 + front.z * 0.25 + bow_z * 0.40,
    );
    let eye = Vec3::new(
        rear.x + driver.x * 3.55 + fwd.x * (-1.05),
        (mud_h + bow_h) * 0.5 + 0.22,
        rear.z + driver.z * 3.55 + fwd.z * (-1.05),
    );
    let ortho = 7.2;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

/// The Niva on its second pass through the same slurry pool. The rut is deeper
/// than after the first pass. Overlay numbers match the PR body.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_second_pass(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let out = world.second_pass_through_deep_slurry(3.6);
    silence_soil_clods(&mut world);
    println!(
        "slurry-second-pass pool {:.4} berm {:.4} slurry {:.4} bow {:.4} wake {:.4} groove {:.4} \
         sum {:.4} rut {:.4}->{:.4}->{:.4} ribbons {:.4} spray {:.4} clay {:.0} silt {:.0} sand {:.0}",
        out.pool,
        out.berm,
        out.slurry,
        out.bow_m3,
        out.wake_m3,
        out.groove,
        out.sum,
        out.rut0,
        out.rut_pass1,
        out.rut_pass2,
        out.ribbons,
        out.spray,
        out.drag_clay,
        out.drag_silt,
        out.drag_sand,
    );
    let lines = [
        format!("pool {:.4} m3", out.pool),
        format!("berm {:.4} m3", out.berm),
        format!("slurry {:.4} m3", out.slurry),
        format!("bow {:.4} m3", out.bow_m3),
        format!("wake {:.4} m3", out.wake_m3),
        format!("groove {:.4} m3", out.groove),
        format!("sum {:.4} m3", out.sum),
        format!("rut1 {:.4} m", out.rut_pass1),
        format!("rut2 {:.4} m", out.rut_pass2),
        format!("ribbons {:.4} m3", out.ribbons),
        format!("spray {:.4} m3", out.spray),
        format!("clay {:.0} N", out.drag_clay),
        format!("silt {:.0} N", out.drag_silt),
        format!("sand {:.0} N", out.drag_sand),
    ];
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let pose = slurry_plow_camera(&world);
    let mut rgba = renderer.draw(&world, pose, None, true).expect("readback");
    stamp_proof_lines(&mut rgba, parsed.width, parsed.height, &line_refs);
    if let Some(parent) = parsed.slurry_second_pass_proof.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(
        &parsed.slurry_second_pass_proof,
        parsed.width,
        parsed.height,
        &rgba,
    )
    .expect("slurry second pass still");
    println!(
        "cc0 slurry-second-pass {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        parsed.slurry_second_pass_proof.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

/// The Niva mid-pass through deep slurry. The bow is ahead of the tread and
/// the wake thins behind it. Overlay numbers match the PR body.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_plow(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let out = world.plow_through_deep_slurry(3.6);
    silence_soil_clods(&mut world);
    let sum = out.slurry + out.bow_m3 + out.wake_m3 + out.groove + out.ribbons + out.spray;
    println!(
        "slurry-plow pool {:.4} m3  slurry {:.4}  bow {:.4}  wake {:.4}  groove {:.4}  \
         sum {:.4}  ribbons {:.4}  spray {:.4}  clay {:.0}  silt {:.0}  sand {:.0}",
        out.pool,
        out.slurry,
        out.bow_m3,
        out.wake_m3,
        out.groove,
        sum,
        out.ribbons,
        out.spray,
        out.drag_clay,
        out.drag_silt,
        out.drag_sand,
    );
    let lines = [
        format!("pool {:.4} m3", out.pool),
        format!("slurry {:.4} m3", out.slurry),
        format!("bow {:.4} m3", out.bow_m3),
        format!("wake {:.4} m3", out.wake_m3),
        format!("groove {:.4} m3", out.groove),
        format!("sum {:.4} m3", sum),
        format!("ribbons {:.4} m3", out.ribbons),
        format!("spray {:.4} m3", out.spray),
        format!("clay {:.0} N", out.drag_clay),
        format!("silt {:.0} N", out.drag_silt),
        format!("sand {:.0} N", out.drag_sand),
    ];
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let pose = slurry_plow_camera(&world);
    let mut rgba = renderer.draw(&world, pose, None, true).expect("readback");
    stamp_proof_lines(&mut rgba, parsed.width, parsed.height, &line_refs);
    if let Some(parent) = parsed.slurry_plow_proof.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(
        &parsed.slurry_plow_proof,
        parsed.width,
        parsed.height,
        &rgba,
    )
    .expect("slurry plow still");
    println!(
        "cc0 slurry-plow {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        parsed.slurry_plow_proof.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_plow_camera(world: &World) -> CameraPose {
    let vehicle = &world.vehicle;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let driver = -right;
    let ori = vehicle.orientation();
    let front = vehicle.pos + ori * vehicle.wheel_center_local(0);
    let rear = vehicle.pos + ori * vehicle.wheel_center_local(2);
    let bow_x = front.x + fwd.x * 0.62;
    let bow_z = front.z + fwd.z * 0.62;
    let wake_x = rear.x - fwd.x * 0.52;
    let wake_z = rear.z - fwd.z * 0.52;
    let bow_h = world.soil.sample(bow_x, bow_z).height;
    let wake_h = world.soil.sample(wake_x, wake_z).height;
    // Low on the driver’s side, sight line along the pool so the bow crest
    // and the wake trough share one grazing profile.
    let target = Vec3::new(bow_x, bow_h + 0.025, bow_z);
    let eye = Vec3::new(
        rear.x + driver.x * 3.4 + fwd.x * (-1.25),
        wake_h + 0.14,
        rear.z + driver.z * 3.4 + fwd.z * (-1.25),
    );
    let ortho = 7.6;
    // Do not lift the eye: clear_ortho_eye pushes this back to a high isometric.
    CameraPose {
        eye,
        target,
        ortho_width: Some(ortho),
    }
}

/// The Niva climbing out of a deep pool. Ribbons, sheets, and fine spray
/// hang beside the rut. The numbers are the rut tear-out, the groove hold,
/// and the debris split.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_tear_out(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let out = world.tear_out_from_deep_slurry();
    silence_soil_clods(&mut world);
    println!(
        "slurry-tear-out pool {:.4} m3  rut {:.4} m3  groove {:.4} m3  shed {:.4} m3  \
         ribbons {:.4} m3  sheets {:.4} m3  spray {:.4} m3  slurry {:.4} m3  airborne {:.4} m3  \
         clay drag {:.0} N  silt {:.0} N  sand {:.0} N  clay torque {:.0} N·m  silt {:.0}  sand {:.0}",
        out.pool,
        out.rut_tear,
        out.groove,
        out.groove_shed,
        out.ribbons,
        out.sheets,
        out.spray,
        out.slurry,
        out.airborne,
        out.drag_clay,
        out.drag_silt,
        out.drag_sand,
        out.torque_clay,
        out.torque_silt,
        out.torque_sand,
    );
    let lines = [
        format!("rut {:.4}", out.rut_tear),
        format!("groove {:.4}", out.groove),
        format!("ribbons {:.4}", out.ribbons),
        format!("sheets {:.4}", out.sheets),
        format!("spray {:.4}", out.spray),
        format!("slurry {:.4}", out.slurry),
        format!("clay {:.0} N", out.drag_clay),
        format!("silt {:.0} N", out.drag_silt),
        format!("sand {:.0} N", out.drag_sand),
    ];
    let line_refs: Vec<&str> = lines.iter().map(|s| s.as_str()).collect();
    let pose = lift_camera(&world, slurry_tear_out_camera(&world));
    let mut rgba = renderer.draw(&world, pose, None, true).expect("readback");
    stamp_proof_lines(&mut rgba, parsed.width, parsed.height, &line_refs);
    if let Some(parent) = parsed.slurry_tear_out_proof.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(
        &parsed.slurry_tear_out_proof,
        parsed.width,
        parsed.height,
        &rgba,
    )
    .expect("slurry tear-out still");
    println!(
        "cc0 slurry-tear-out {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        parsed.slurry_tear_out_proof.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_tear_out_camera(world: &World) -> CameraPose {
    use silt::particles::{KIND_SLURRY_DROP, KIND_SLURRY_RIBBON, KIND_SLURRY_SHEET};

    let vehicle = &world.vehicle;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let ori = vehicle.orientation();
    let rear_l = vehicle.pos + ori * vehicle.wheel_center_local(2);
    let rear_r = vehicle.pos + ori * vehicle.wheel_center_local(3);
    let rear = (rear_l + rear_r) * 0.5;

    let (mut sx, mut sy, mut sz, mut sw) = (0.0_f32, 0.0, 0.0, 0.0);
    for p in &world.particles.items {
        if !p.alive || !p.ledger || p.volume <= 1.0e-8 {
            continue;
        }
        if p.kind != KIND_SLURRY_RIBBON && p.kind != KIND_SLURRY_SHEET && p.kind != KIND_SLURRY_DROP {
            continue;
        }
        let w = p.volume;
        sx += p.pos.x * w;
        sy += p.pos.y * w;
        sz += p.pos.z * w;
        sw += w;
    }
    let debris = if sw > 1.0e-6 {
        Vec3::new(sx / sw, sy / sw, sz / sw)
    } else {
        rear - fwd * 0.42 - right * 0.30 + Vec3::Y * 0.75
    };

    let tread_lip = Vec3::new(
        rear_l.x - fwd.x * 0.12,
        world.soil.sample(rear_l.x, rear_l.z).height + 0.04,
        rear_l.z - fwd.z * 0.12,
    );
    let pool_back = rear - fwd * 0.95;
    let pool_back = Vec3::new(
        pool_back.x,
        world.soil.sample(pool_back.x, pool_back.z).height + 0.05,
        pool_back.z,
    );

    // Beside the rear treads: debris, pool lip, left rear leaving the rut.
    let look = debris * 0.36 + rear_l * 0.22 + tread_lip * 0.20 + pool_back * 0.22;
    let look = Vec3::new(
        look.x,
        look.y.max(world.soil.sample(look.x, look.z).height + 0.12),
        look.z,
    );
    // Same rear-quarter axis, pulled back for lip + tire + spray in one frame.
    let eye = rear - right * 2.75 - fwd * 3.45 + Vec3::Y * 0.82;
    let ortho = 4.85;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_roll_camera(world: &World) -> CameraPose {
    let soil = &world.soil;
    let half = soil.size * 0.5;
    let mut tx = 0.0_f32;
    let mut tz = 0.0_f32;
    let mut tw = 0.0_f32;
    let mut bx = 0.0_f32;
    let mut bz = 0.0_f32;
    let mut bw = 0.0_f32;
    for idx in 0..soil.bed.len() {
        let depth = (soil.initial_height[idx] - soil.bed[idx]).max(0.0);
        let placed = soil.placed[idx];
        let i = idx % soil.res;
        let j = idx / soil.res;
        let x = (i as f32 + 0.5) * soil.cell - half;
        let z = (j as f32 + 0.5) * soil.cell - half;
        if depth > 0.05 {
            tx += x * depth;
            tz += z * depth;
            tw += depth;
        }
        if placed > 0.04 {
            bx += x * placed;
            bz += z * placed;
            bw += placed;
        }
    }
    let trench = if tw > 1.0e-4 {
        Vec3::new(tx / tw, 0.0, tz / tw)
    } else {
        Vec3::new(0.55, 0.0, 0.0)
    };
    let berm = if bw > 1.0e-4 {
        Vec3::new(bx / bw, 0.0, bz / bw)
    } else {
        trench + Vec3::X
    };
    // Midway between the trench floor and the berm crest, with the Niva
    // still in the frame on the near side.
    let p = world.vehicle.pos;
    let look = Vec3::new(
        trench.x * 0.42 + berm.x * 0.38 + p.x * 0.20,
        0.02,
        trench.z * 0.50 + berm.z * 0.30 + p.z * 0.20,
    );
    let along = Vec3::new(0.0, 0.0, 1.0);
    let eye = look + along * 3.15 + Vec3::new(0.35, 0.0, 0.0) + Vec3::Y * 1.05;
    let ortho = 6.4;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

/// Hard-pass ruts with the standing water mixed into the wet clay.
/// The floor is one slurry. The water column is gone.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::{
        mixes_as_slurry, rut_depth_from_force, slurry_viscosity, SurfaceKind, HARD_PASS_NORMAL,
        HARD_PASS_PUSH, SLURRY_COMPARE_MOISTURE,
    };

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let moisture = SLURRY_COMPARE_MOISTURE;
    for (name, kind) in [
        ("clay", SurfaceKind::WetMud),
        ("silt", SurfaceKind::Silt),
        ("sand", SurfaceKind::Sand),
    ] {
        let visc = slurry_viscosity(kind, moisture);
        println!("slurry viscosity {name} {visc:.2} Pa·s at moisture {moisture:.2}");
    }

    let hard_bed = rut_depth_from_force(
        HARD_PASS_NORMAL,
        HARD_PASS_PUSH,
        0.38,
        0.28,
        SurfaceKind::WetMud,
        0.82,
        0.02,
        0.0,
    );
    let mut world = World::wet_clay();
    let z0 = -2.6_f32;
    let z1 = 1.8_f32;
    for side in [-0.74_f32, 0.74] {
        world.soil.roll_with_force(
            Vec3::new(side, 0.0, z0),
            Vec3::new(side, 0.0, z1),
            hard_bed.max(0.012),
            0.0,
            HARD_PASS_NORMAL,
            HARD_PASS_PUSH,
        );
    }
    let mut poured = 0u32;
    for idx in 0..world.soil.bed.len() {
        let dug = world.soil.initial_height[idx] - world.soil.bed[idx];
        let kind = SurfaceKind::from_u8(world.soil.kind[idx]);
        if dug < 0.04 || !mixes_as_slurry(kind) {
            continue;
        }
        if !matches!(
            kind,
            SurfaceKind::WetMud | SurfaceKind::FreshCut | SurfaceKind::PackedClay
        ) {
            continue;
        }
        world.soil.water[idx] = (dug * 0.42).clamp(0.05, 0.10);
        world.soil.wetness[idx] = moisture;
        poured += 1;
    }
    let before = world.soil.mix_volumes();
    println!(
        "slurry before  soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3  cells {poured}",
        before.soil, before.water, before.slurry, before.total()
    );
    world.soil.mix_slurry();
    let after = world.soil.mix_volumes();
    println!(
        "slurry after   soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
        after.soil, after.water, after.slurry, after.total()
    );
    world.vehicle.pos = Vec3::new(0.0, 0.8, 0.15);
    world.vehicle.yaw = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    let ori = world.vehicle.orientation();
    let mut seated = 0.0_f32;
    for i in 0..4 {
        let local = ori * world.vehicle.wheel_center_local(i);
        let ground = world
            .soil
            .sample(world.vehicle.pos.x + local.x, world.vehicle.pos.z + local.z)
            .height;
        seated += ground + 0.38 - local.y;
    }
    world.vehicle.pos.y = seated / 4.0;
    silence_chunks(&mut world);
    write_cc0_still(
        renderer,
        &world,
        "slurry",
        &parsed.slurry_proof,
        parsed,
        slurry_camera(&world),
    );
}

#[cfg(not(target_arch = "wasm32"))]
/// Slurry that has run into the low rut, with a bow in front of the Niva.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_flow(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::vehicle::{hardpoints, REST_LEN, WHEEL_RADIUS, WHEEL_WIDTH};

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::wet_clay();
    let bed = world.soil.paint_slurry_flow();
    let before = world.soil.mix_volumes();
    let low_before = world
        .soil
        .slurry_volume_in(bed.x0, bed.x1, bed.low_z0, bed.low_z1);
    let berm_before = world
        .soil
        .placed_volume_in(bed.x1, bed.x1 + 0.6, bed.high_z0, bed.high_z1);
    println!(
        "slurry flow before  soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
        before.soil,
        before.water,
        before.slurry,
        before.total()
    );
    world.soil.advance_slurry(5.0);
    let after = world.soil.mix_volumes();
    let low_after = world
        .soil
        .slurry_volume_in(bed.x0, bed.x1, bed.low_z0, bed.low_z1);
    let berm_after = world
        .soil
        .placed_volume_in(bed.x1, bed.x1 + 0.6, bed.high_z0, bed.high_z1);
    println!(
        "slurry flow after   soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
        after.soil, after.water, after.slurry, after.total()
    );
    println!(
        "slurry low {:.4} -> {:.4} m3   berm {:.4} -> {:.4} m3",
        low_before, low_after, berm_before, berm_after
    );

    let pool = deepest_slurry_row(&world.soil, &bed);
    // Front axle on the pooled slurry. The nose sits in the low bed.
    world.vehicle.pos = Vec3::new(0.0, 0.8, pool - 1.08);
    world.vehicle.yaw = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    println!("slurry pool z {pool:.2}  niva z {:.2}", world.vehicle.pos.z);
    let ori = world.vehicle.orientation();
    let mut bow_max = 0.0_f32;
    for i in [0_usize, 1] {
        let local = ori * world.vehicle.wheel_center_local(i);
        let x = world.vehicle.pos.x + local.x;
        let z = world.vehicle.pos.z + local.z;
        let depth = world.soil.sample(x, z).slurry;
        let pass = world
            .soil
            .raise_bow_wave(x, z, Vec3::Z, WHEEL_RADIUS, WHEEL_WIDTH);
        println!("wheel {i} slurry {depth:.4} m");
        bow_max = bow_max.max(pass.bow_m3);
        println!(
            "bow wave wheel {i} {:.4} m3  wake {:.4} m3 at ({x:.2}, {z:.2})",
            pass.bow_m3, pass.wake_m3
        );
    }
    println!("bow wave height {bow_max:.4} m");

    // Each tire sits on the slurry under it. The front axle drops into the
    // low bed; the rear stays on the higher rut.
    let hps = hardpoints();
    let mut ground = [0.0_f32; 4];
    for i in 0..4 {
        let x = world.vehicle.pos.x + hps[i].x;
        let z = world.vehicle.pos.z + hps[i].z;
        ground[i] = world.soil.sample(x, z).height;
    }
    let min_g = ground.into_iter().fold(f32::MAX, f32::min);
    for i in 0..4 {
        world.vehicle.wheels[i].compression = (ground[i] - min_g + 0.04).clamp(0.0, 0.30);
    }
    let c0 = world.vehicle.wheels[0].compression;
    world.vehicle.pos.y = ground[0] + WHEEL_RADIUS - hps[0].y + REST_LEN - c0;
    let bottom = world.vehicle.pos.y + hps[0].y - REST_LEN + c0 - WHEEL_RADIUS;
    println!(
        "front tire on slurry  ground {:.4} m  bottom {:.4} m  compression {:.3}",
        ground[0], bottom, c0
    );
    silence_chunks(&mut world);
    write_cc0_still(
        renderer,
        &world,
        "slurry-flow",
        &parsed.slurry_flow_proof,
        parsed,
        slurry_flow_camera(&world),
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(not(target_arch = "wasm32"))]
fn deepest_slurry_row(soil: &silt::soil::SoilField, bed: &silt::soil::SlurryFlowBed) -> f32 {
    let half = soil.size * 0.5;
    let mut best_z = bed.low_z0 + 0.55;
    let mut best = 0.0_f32;
    for j in 0..soil.res {
        let z = (j as f32 + 0.5) * soil.cell - half;
        if z < bed.low_z0 + 0.25 || z >= bed.low_z1 - 0.55 {
            continue;
        }
        let mut row = 0.0_f32;
        let mut n = 0.0_f32;
        for i in 0..soil.res {
            let x = (i as f32 + 0.5) * soil.cell - half;
            if x < bed.x0 + 0.2 || x >= bed.x1 - 0.2 {
                continue;
            }
            row += soil.slurry[j * soil.res + i];
            n += 1.0;
        }
        if n > 0.0 && row / n > best {
            best = row / n;
            best_z = z;
        }
    }
    best_z
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_flow_camera(world: &World) -> CameraPose {
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let p = world.vehicle.pos;
    // Right front tire, then the bow piled in the slurry ahead of it.
    let wheel = p + fwd * 1.08 + right * 0.74;
    let bow = wheel + fwd * 0.55;
    let floor = world.soil.sample(bow.x, bow.z).height;
    let look = Vec3::new(
        wheel.x * 0.35 + bow.x * 0.65,
        floor + 0.01,
        wheel.z * 0.30 + bow.z * 0.70,
    );
    // Low, beside the wave, so the mound in front of the tread and the low bed read together.
    let eye = wheel + right * 1.05 + fwd * 0.85 + Vec3::Y * 0.28;
    let ortho = 1.35;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

/// Niva sunk in a deep clay slurry. The bow in front of the tread is the
/// slurry the tire is shoving, and the mix viscosity is what resists the roll.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_drag(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::{slurry_motion_drag, SurfaceKind, SLURRY_COMPARE_MOISTURE};
    use silt::vehicle::{WHEEL_RADIUS, WHEEL_WIDTH};

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let moisture = SLURRY_COMPARE_MOISTURE;
    let depth = 0.16_f32;
    let mut world = World::wet_clay();
    let poured = world.soil.fill_slurry_pool(
        -2.4,
        2.4,
        -2.6,
        2.4,
        depth,
        SurfaceKind::WetMud,
        moisture,
    );
    println!("slurry pool {poured:.4} m3  depth {depth:.3} m");
    for (name, kind) in [
        ("clay", SurfaceKind::WetMud),
        ("silt", SurfaceKind::Silt),
        ("sand", SurfaceKind::Sand),
    ] {
        let drag = slurry_motion_drag(kind, moisture, depth, WHEEL_RADIUS, WHEEL_WIDTH, 1.15, 3.2);
        println!(
            "slurry drag {name}  force {:.1} N  torque {:.1} N·m  visc {:.2} Pa·s",
            drag.force.abs(),
            drag.torque.abs(),
            drag.viscosity
        );
    }

    world.vehicle.pos = Vec3::new(0.0, 0.55, -0.15);
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.vel = Vec3::new(0.0, 0.0, 0.85);
    for w in &mut world.vehicle.wheels {
        w.compression = 0.22;
        w.omega = 4.2;
        w.slurry_sealed = true;
    }
    seat_axle(&mut world, 0, 0.18);
    let ori = world.vehicle.orientation();
    let mut bow_max = 0.0_f32;
    for i in 0..4 {
        let local = ori * world.vehicle.wheel_center_local(i);
        let x = world.vehicle.pos.x + local.x;
        let z = world.vehicle.pos.z + local.z;
        let sample = world.soil.sample(x, z);
        let pass = world
            .soil
            .raise_bow_wave(x, z, Vec3::Z, WHEEL_RADIUS, WHEEL_WIDTH);
        bow_max = bow_max.max(pass.bow_m3);
        println!(
            "wheel {i} slurry {:.4} m  bow {:.4} m3  wake {:.4} m3  at ({x:.2}, {z:.2})",
            sample.slurry, pass.bow_m3, pass.wake_m3
        );
    }
    println!("bow wave height {bow_max:.4} m");
    silence_chunks(&mut world);
    write_cc0_still(
        renderer,
        &world,
        "slurry-drag",
        &parsed.slurry_drag_proof,
        parsed,
        slurry_drag_camera(&world),
    );
}

/// The front tire has just broken a slurry seal. Ribbons and drops in the air
/// are the volume taken out of the pool.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_suction(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::{SlurrySeal, SurfaceKind, SLURRY_COMPARE_MOISTURE};
    use silt::vehicle::{WHEEL_RADIUS, WHEEL_WIDTH};

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let moisture = SLURRY_COMPARE_MOISTURE;
    let depth = 0.16_f32;
    let mut world = World::wet_clay();
    let poured = world.soil.fill_slurry_pool(
        -2.4,
        2.4,
        -2.8,
        2.6,
        depth,
        SurfaceKind::WetMud,
        moisture,
    );
    let before = world.soil.mix_volumes();
    println!(
        "suction pool {poured:.4} m3  volume before {:.4} m3",
        before.total()
    );

    world.vehicle.pos = Vec3::new(0.0, 0.7, -0.05);
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.10;
    world.vehicle.roll = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    for (i, w) in world.vehicle.wheels.iter_mut().enumerate() {
        w.compression = if i < 2 { 0.05 } else { 0.22 };
        w.omega = if i < 2 { 1.4 } else { 0.4 };
        w.slurry_sealed = false;
        w.slurry_lift = 0.05;
    }
    // Nose up, front tread just clear of the seal it broke.
    seat_axle(&mut world, 0, -0.04);

    let ori = world.vehicle.orientation();
    let mut thrown = 0.0_f32;
    let mut impulse = 0.0_f32;
    // The camera is outside the right front tire. That is the seal that breaks.
    for i in [1_usize] {
        let local = ori * world.vehicle.wheel_center_local(i);
        let x = world.vehicle.pos.x + local.x;
        let z = world.vehicle.pos.z + local.z;
        let mut seal = SlurrySeal::seated();
        let dt = 0.02_f32;
        let lift_speed = 0.40_f32;
        for _ in 0..24 {
            let out = world.soil.pull_from_slurry(
                x,
                z,
                Vec3::Z,
                WHEEL_RADIUS,
                WHEEL_WIDTH,
                lift_speed,
                dt,
                &mut seal,
            );
            impulse += out.impulse;
            if out.thrown > 1.0e-8 {
                let right = Vec3::X;
                let origin = Vec3::new(x, world.soil.sample(x, z).height + 0.04, z);
                let took = world.particles.throw_slurry(origin, Vec3::Z, right, out.thrown);
                thrown += out.thrown;
                println!(
                    "wheel {i} breakaway thrown {:.4} m3  ribbons {:.4}  drops {:.4}  impulse {:.2} N·s  pieces {}",
                    out.thrown,
                    took.ribbons,
                    took.drops,
                    seal.impulse,
                    world.particles.alive_count()
                );
                break;
            }
        }
    }
    let after = world.soil.mix_volumes();
    println!(
        "suction impulse {impulse:.2} N·s  thrown {thrown:.4} m3  volume before {:.4} m3  after {:.4} m3  airborne {:.4} m3",
        before.total(),
        after.total(),
        world.soil.airborne
    );
    write_cc0_still(
        renderer,
        &world,
        "slurry-suction",
        &parsed.slurry_suction_proof,
        parsed,
        slurry_suction_camera(&world),
    );
}

/// The right front tire has been pulled out of a slurry rut. The voids still
/// hold slurry, and a spin is throwing the rest off as ribbons.
#[cfg(not(target_arch = "wasm32"))]
fn capture_slurry_groove(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::patch::{groove_slurry_capacity, release_groove_slurry};
    use silt::soil::{SurfaceKind, SLURRY_COMPARE_MOISTURE};
    use silt::vehicle::{WHEEL_RADIUS, WHEEL_WIDTH};

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let moisture = SLURRY_COMPARE_MOISTURE;
    let depth = 0.16_f32;
    let mut world = World::wet_clay();
    let poured = world.soil.fill_slurry_pool(
        -2.4,
        2.4,
        -2.8,
        2.6,
        depth,
        SurfaceKind::WetMud,
        moisture,
    );
    let before = world.soil.mix_volumes();
    println!("groove pool {poured:.4} m3  volume before {:.4} m3", before.total());

    world.vehicle.pos = Vec3::new(0.0, 0.72, 0.15);
    world.vehicle.yaw = 0.0;
    world.vehicle.pitch = 0.0;
    world.vehicle.roll = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    for (i, w) in world.vehicle.wheels.iter_mut().enumerate() {
        w.compression = 0.12;
        w.omega = if i == 3 { 14.0 } else { 0.2 };
        w.spin = if i == 3 { 1.1 } else { 0.0 };
        w.slurry_sealed = false;
        w.slurry_lift = 0.05;
        w.slurry_groove = 0.0;
        w.groove = 0.0;
        w.smear = 0.0;
        w.sidewall = 0.0;
    }
    // Rear tire, clear of the rut, so the crown faces a camera behind the Niva.
    seat_axle(&mut world, 3, -0.04);

    let ori = world.vehicle.orientation();
    let local = ori * world.vehicle.wheel_center_local(3);
    let x = world.vehicle.pos.x + local.x;
    let z = world.vehicle.pos.z + local.z;
    let cap = groove_slurry_capacity(WHEEL_RADIUS, WHEEL_WIDTH);
    let packed = world
        .soil
        .pack_slurry_into_grooves(x, z, Vec3::Z, WHEEL_RADIUS, WHEEL_WIDTH, cap);
    let mut held = packed;
    let dt = 1.0 / 60.0;
    let mut thrown = 0.0_f32;
    // A short spin. Ribbons leave, and most of the slurry is still in the voids.
    for _ in 0..5 {
        thrown += release_groove_slurry(&mut held, 1.0, 0.0, dt);
    }
    world.soil.airborne += thrown;
    world.vehicle.wheels[3].slurry_groove = held;
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let origin = world.vehicle.pos + local;
    let carried = world
        .particles
        .throw_groove_ribbons(origin, fwd, right, thrown);
    let after = world.soil.mix_volumes();
    let returned = 0.0_f32;
    println!(
        "groove-hold packed {packed:.5} m3  groove {held:.5} m3  thrown {thrown:.5} m3  carried {carried:.5} m3  returned {returned:.5} m3  sum {:.5} m3",
        held + thrown + returned
    );
    println!(
        "groove column before slurry {:.4} m3  after {:.4} m3  cap {cap:.5} m3",
        before.slurry, after.slurry
    );
    write_cc0_still(
        renderer,
        &world,
        "slurry-groove",
        &parsed.slurry_groove_proof,
        parsed,
        slurry_groove_camera(&world),
    );
}

/// Put one axle's tire bottoms `sink` metres under the slurry surface.
/// Negative sink lifts the tread clear.
#[cfg(not(target_arch = "wasm32"))]
fn seat_axle(world: &mut World, wheel: usize, sink: f32) {
    use silt::vehicle::{hardpoints, WHEEL_RADIUS};
    let hps = hardpoints();
    let ori = world.vehicle.orientation();
    let local = ori * world.vehicle.wheel_center_local(wheel);
    let x = world.vehicle.pos.x + hps[wheel].x;
    let z = world.vehicle.pos.z + hps[wheel].z;
    // Pitch moves the contact off the hardpoint. Sample where the tire is.
    let wx = world.vehicle.pos.x + local.x;
    let wz = world.vehicle.pos.z + local.z;
    let ground = world.soil.sample(wx, wz).height;
    world.vehicle.pos.y = ground - sink - local.y + WHEEL_RADIUS;
    let bottom = world.vehicle.pos.y + local.y - WHEEL_RADIUS;
    println!(
        "seat wheel {wheel}  ground {ground:.4} m  bottom {bottom:.4} m  sink {:.3} m  hardpoint ({x:.2},{z:.2})",
        ground - bottom
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_drag_camera(world: &World) -> CameraPose {
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let p = world.vehicle.pos;
    let wheel = p + fwd * 1.05 + right * 0.78;
    let bow = wheel + fwd * 0.42;
    let floor = world.soil.sample(wheel.x, wheel.z).height;
    let look = Vec3::new(
        wheel.x * 0.72 + p.x * 0.28,
        floor - 0.04,
        wheel.z * 0.62 + bow.z * 0.38,
    );
    let eye = wheel + right * 1.72 - fwd * 0.05 + Vec3::Y * 0.34;
    let ortho = 1.85;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_suction_camera(world: &World) -> CameraPose {
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let ori = world.vehicle.orientation();
    let local = ori * world.vehicle.wheel_center_local(1);
    let wheel = world.vehicle.pos + local;
    let floor = world.soil.sample(wheel.x, wheel.z).height;
    // Beside the tread, where the ribbons and drops leave the broken seal.
    let look = Vec3::new(wheel.x + right.x * 0.38, floor + 0.36, wheel.z + fwd.z * 0.06);
    let eye = wheel + right * 1.85 + fwd * 0.22 + Vec3::Y * 0.58;
    let ortho = 1.65;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_groove_camera(world: &World) -> CameraPose {
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let ori = world.vehicle.orientation();
    let wheel = world.vehicle.pos + ori * world.vehicle.wheel_center_local(3);
    // Behind the rear crown, far enough out that the rim is an edge and
    // the ribbons have air beside the tread.
    let look = wheel - fwd * 0.20 + right * 0.02 - Vec3::Y * 0.08;
    let eye = wheel - fwd * 1.08 + right * 0.26 + Vec3::Y * 0.10;
    CameraPose {
        eye,
        target: look,
        ortho_width: Some(1.00),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn slurry_camera(world: &World) -> CameraPose {
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let p = world.vehicle.pos;
    let bed = p - fwd * 1.35 + right * 0.74;
    let floor = world.soil.sample(bed.x, bed.z);
    let look = Vec3::new(
        bed.x * 0.62 + p.x * 0.38,
        floor.height + 0.02,
        bed.z * 0.55 + p.z * 0.45,
    );
    let eye = look - fwd * 2.15 + right * 2.05 + Vec3::Y * 0.92;
    let ortho = 3.8;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

/// Hard-pass ruts under the Niva. The bed and the berm are the force field.
#[cfg(not(target_arch = "wasm32"))]
fn capture_force(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::{
        rut_depth_from_force, SoilField, SurfaceKind, HARD_PASS_NORMAL, HARD_PASS_PUSH,
        LIGHT_PASS_NORMAL, LIGHT_PASS_PUSH,
    };

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut hard_bed = 0.0_f32;
    for (label, normal, push) in [
        ("light", LIGHT_PASS_NORMAL, LIGHT_PASS_PUSH),
        ("hard", HARD_PASS_NORMAL, HARD_PASS_PUSH),
    ] {
        let mut soil = SoilField::mud_pad();
        let want = rut_depth_from_force(
            normal,
            push,
            0.38,
            0.28,
            SurfaceKind::WetMud,
            0.82,
            0.02,
            0.0,
        );
        let before = soil.placed_volume();
        soil.roll_with_force(
            Vec3::new(0.0, 0.0, -1.2),
            Vec3::new(0.0, 0.0, 1.2),
            want.max(0.012),
            0.0,
            normal,
            push,
        );
        let depth = soil.max_rut_depth();
        let placed = soil.placed_volume() - before;
        println!("{label} depth {depth:.4} m  placed {placed:.4} m3");
        if label == "hard" {
            hard_bed = depth;
        }
    }

    let mut world = World::wet_clay();
    let z0 = -2.6_f32;
    let z1 = 1.8_f32;
    for side in [-0.74_f32, 0.74] {
        world.soil.roll_with_force(
            Vec3::new(side, 0.0, z0),
            Vec3::new(side, 0.0, z1),
            hard_bed.max(0.012),
            0.0,
            HARD_PASS_NORMAL,
            HARD_PASS_PUSH,
        );
    }
    world.vehicle.pos = Vec3::new(0.0, 0.8, 0.15);
    world.vehicle.yaw = 0.0;
    world.vehicle.vel = Vec3::ZERO;
    let ori = world.vehicle.orientation();
    let mut seated = 0.0_f32;
    for i in 0..4 {
        let local = ori * world.vehicle.wheel_center_local(i);
        let ground = world
            .soil
            .sample(world.vehicle.pos.x + local.x, world.vehicle.pos.z + local.z)
            .height;
        seated += ground + 0.38 - local.y;
    }
    world.vehicle.pos.y = seated / 4.0;
    silence_chunks(&mut world);
    let v = world.soil.volumes();
    println!(
        "picture depth {:.4} m  placed {:.4} m3  cut {:.4} m3",
        world.soil.max_rut_depth(),
        v.placed,
        v.cut
    );
    write_cc0_still(
        renderer,
        &world,
        "force",
        &parsed.force_proof,
        parsed,
        force_camera(&world),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn force_camera(world: &World) -> CameraPose {
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let p = world.vehicle.pos;
    let bed = p - fwd * 1.7 + right * 0.74;
    let berm = bed + right * 0.46;
    let bed_h = world.soil.sample(bed.x, bed.z).height;
    let berm_h = world.soil.sample(berm.x, berm.z).height;
    let look = Vec3::new(
        bed.x * 0.55 + p.x * 0.45,
        (bed_h + berm_h) * 0.5,
        bed.z * 0.45 + p.z * 0.55,
    );
    // Behind the shoulder, wide enough that the trench and the ridge read
    // as one rut beside the Niva.
    let eye = look - fwd * 2.5 + right * 2.35 + Vec3::Y * 1.15;
    let ortho = 4.6;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_surfaces(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::{surface_grip_sink, SurfaceKind};

    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let pairs = [
        ("gravel", SurfaceKind::Gravel),
        ("dry clay", SurfaceKind::DryCrust),
        ("wet clay", SurfaceKind::WetMud),
        ("sand", SurfaceKind::Sand),
        ("silt", SurfaceKind::Silt),
    ];
    for (name, kind) in pairs {
        let (grip, sink) = surface_grip_sink(kind);
        println!("surface {name}  grip {grip:.3}  sink {sink:.4} m");
    }

    let mut world = World::surface_split();
    world.bike.pos = Vec3::new(16.0, 0.5, 16.0);
    world.bike.vel = Vec3::ZERO;
    silence_chunks(&mut world);
    print_sink("surfaces", &world);
    write_cc0_still(
        renderer,
        &world,
        "surfaces",
        &parsed.surfaces_proof,
        parsed,
        surfaces_camera(&world),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn surfaces_camera(world: &World) -> CameraPose {
    let p = world.vehicle.pos;
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let axle = p + fwd * 0.72;
    let ground = world.soil.sample(axle.x, axle.z).height;
    // Down the nose, close enough that the clay hubs sit lower than the gravel.
    let look = Vec3::new(axle.x, ground + 0.46, axle.z);
    let eye = look + fwd * 3.3 + right * 0.08 + Vec3::Y * 2.35;
    let ortho = 3.55;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_niva_realism(renderer: &mut Renderer, parsed: &CaptureArgs) {
    renderer.daylight = true;
    renderer.show_props = false;
    renderer.show_bike = false;
    renderer.niva_picture = true;

    let mut world = World::flat(0.72);
    world.bike.pos = Vec3::new(18.0, 0.5, 18.0);
    world.bike.vel = Vec3::ZERO;
    for w in &mut world.soil.water {
        *w = 0.0;
    }
    let dt = 1.0 / 60.0;
    for _ in 0..24 {
        world.step(dt, &Input::default());
    }
    silence_chunks(&mut world);
    print_sink("niva-sink", &world);
    write_cc0_still(
        renderer,
        &world,
        "niva-sink-side",
        &parsed.niva_sink_side,
        parsed,
        niva_sink_side_camera(&world),
    );
    write_cc0_still(
        renderer,
        &world,
        "niva-sink-front",
        &parsed.niva_sink_front,
        parsed,
        niva_sink_front_camera(&world),
    );

    let origin = world.vehicle.pos;
    for _ in 0..140 {
        prescribe_roll(&mut world, 1.05);
        world.step(dt, &Input::default());
    }
    let (n, amp) = world.soil.tread_imprint_stats();
    println!("imprint after pass  cells {n}  deepest {amp:.4} m");
    for _ in 0..50 {
        prescribe_roll(&mut world, 1.05);
        world.step(dt, &Input::default());
    }
    let (n2, amp2) = world.soil.tread_imprint_stats();
    println!("imprint second look cells {n2}  deepest {amp2:.4} m");
    silence_chunks(&mut world);
    // The rut the car has left. A spot under the bumper hides the lugs behind the body.
    let spot = imprint_behind(&world, origin.z);
    println!(
        "imprint spot ({:.2}, {:.2}) amp {:.4}",
        spot.x,
        spot.z,
        world
            .soil
            .idx_at(spot.x, spot.z)
            .map(|i| world.soil.tread_amp[i])
            .unwrap_or(0.0)
    );
    write_cc0_still(
        renderer,
        &world,
        "niva-tread-top",
        &parsed.niva_tread_top,
        parsed,
        niva_imprint_top_camera(&world, spot),
    );
    write_cc0_still(
        renderer,
        &world,
        "niva-tread-side",
        &parsed.niva_tread_side,
        parsed,
        niva_imprint_side_camera(&world, spot),
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn print_sink(label: &str, world: &World) {
    use silt::vehicle::{MASS, WHEEL_RADIUS};
    let mut support = 0.0_f32;
    for i in 0..4 {
        let w = world.vehicle.wheels[i];
        let center =
            world.vehicle.pos + world.vehicle.orientation() * world.vehicle.wheel_center_local(i);
        let h = world.soil.sample(center.x, center.z).height;
        let bottom = center.y - WHEEL_RADIUS;
        let seat = h - w.sinkage;
        support += w.normal;
        println!(
            "{label} wheel {i}  sink {:.4} m  hub {:.4}  soil {:.4}  bottom {:.4}  soil-sink {:.4}  \
             gap {:.4}  comp {:.3}  normal {:.0} N",
            w.sinkage,
            center.y,
            h,
            bottom,
            seat,
            bottom - seat,
            w.compression,
            w.normal
        );
    }
    println!(
        "{label} support {:.0} N  weight {:.0} N",
        support,
        MASS * 9.81
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn imprint_spot(world: &World, z: f32) -> Vec3 {
    let mut best = Vec3::new(0.74, 0.0, z);
    let mut best_a = -1.0_f32;
    let mut x = -1.6_f32;
    while x <= 1.6 {
        if let Some(idx) = world.soil.idx_at(x, z) {
            let a = world.soil.tread_amp[idx];
            if a > best_a {
                best_a = a;
                let half = world.soil.size * 0.5;
                let i = idx % world.soil.res;
                let j = idx / world.soil.res;
                best = Vec3::new(
                    (i as f32 + 0.5) * world.soil.cell - half,
                    0.0,
                    (j as f32 + 0.5) * world.soil.cell - half,
                );
            }
        }
        x += world.soil.cell * 0.5;
    }
    best
}

/// Strongest tread negative at least a car-length behind the Niva.
#[cfg(not(target_arch = "wasm32"))]
fn imprint_behind(world: &World, origin_z: f32) -> Vec3 {
    let car_z = world.vehicle.pos.z;
    let mut best = imprint_spot(world, origin_z + 0.4);
    let mut best_a = -1.0_f32;
    let mut z = origin_z - 0.15;
    // Past the rear bumper, so the body does not cover the print.
    let end = car_z - 2.55;
    while z < end {
        let spot = imprint_spot(world, z);
        let amp = world
            .soil
            .idx_at(spot.x, spot.z)
            .map(|i| world.soil.tread_amp[i])
            .unwrap_or(0.0);
        if amp > best_a {
            best_a = amp;
            best = spot;
        }
        z += world.soil.cell;
    }
    best
}

#[cfg(not(target_arch = "wasm32"))]
fn niva_sink_side_camera(world: &World) -> CameraPose {
    let p = world.vehicle.pos;
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let ground = world.soil.sample(p.x + right.x * 1.6, p.z + right.z * 1.6).height;
    let look = Vec3::new(p.x, ground + 0.16, p.z);
    let eye = look + right * 2.75 + fwd * 0.28 + Vec3::Y * 0.48;
    let ortho = 3.05;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn niva_sink_front_camera(world: &World) -> CameraPose {
    let p = world.vehicle.pos;
    let fwd = world.vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let axle = p + fwd * 1.08;
    let ground = world
        .soil
        .sample(axle.x + right.x * 1.4, axle.z + fwd.z * 0.4)
        .height;
    // In front and above, looking down onto the tread. The old eye sat in
    // the ortho slice, so the bottom of the frame was the clear color.
    // Above the bumper, so the buried tread is in frame and the carcass
    // is not cropped on the mud.
    // The contact sits in the frame, with ground in front of both tires.
    let look = Vec3::new(axle.x, ground - 0.08, axle.z - 0.12);
    let eye = look + fwd * 2.15 + right * 0.06 + Vec3::Y * 0.72;
    let ortho = 2.15;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn niva_imprint_top_camera(world: &World, spot: Vec3) -> CameraPose {
    let h = world.soil.sample(spot.x, spot.z).height;
    let look = Vec3::new(spot.x, h + 0.01, spot.z);
    // Along one rut, off vertical. A straight-down look is parallel to up.
    CameraPose {
        eye: look + Vec3::new(0.34, 2.55, -0.46),
        target: look,
        ortho_width: Some(1.18),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn niva_imprint_side_camera(world: &World, spot: Vec3) -> CameraPose {
    let h = world.soil.sample(spot.x, spot.z).height;
    let lip = world.soil.sample(spot.x + 0.85, spot.z).height.max(h);
    // Beside the rut and above the lip, along the lug row. The tire stays
    // out of frame, and the eye stays off the underside of the mesh.
    let look = Vec3::new(spot.x, h + 0.01, spot.z + 0.12);
    let eye = Vec3::new(spot.x + 0.58, lip + 0.92, spot.z - 0.40);
    let ortho = 1.02;
    CameraPose {
        eye: clear_ortho_eye(world, eye, look, ortho),
        target: look,
        ortho_width: Some(ortho),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn print_patch(label: &str, world: &World) {
    use silt::patch::{contact_patch, TreadCoat};
    use silt::vehicle::{WHEEL_RADIUS, WHEEL_WIDTH};
    let w = world.vehicle.wheels[3];
    let center = world.vehicle.pos + world.vehicle.orientation() * world.vehicle.wheel_center_local(3);
    let sample = world.soil.sample(center.x, center.z);
    let patch = contact_patch(
        WHEEL_RADIUS,
        WHEEL_WIDTH,
        w.sinkage.max(0.02),
        w.slip,
        w.normal.max(800.0),
        sample.kind(),
        sample.wetness,
        sample.loose,
        sample.water,
        TreadCoat {
            groove: w.groove,
            smear: w.smear,
            sidewall: w.sidewall,
        },
    );
    let mut kinds = [0u32; 5];
    for p in &world.particles.items {
        if p.alive && (p.kind as usize) < kinds.len() {
            kinds[p.kind as usize] += 1;
        }
    }
    let thrown = (kinds[0] + kinds[1]) as usize;
    println!(
        "{label} slip {:.2}  speed {:.2}  sink {:.3}  groove {:.2}  sidewall {:.2}  smear {:.2}  mud {:.2}  \
         grip {:.0} N  mu {:.3}  scale {:.2}  len {:.3} m  pull {:.2}  coat_in_rut {:.5} m3  \
         clods {thrown}  kinds {kinds:?}  soil {:?}",
        w.slip,
        world.vehicle.speed(),
        w.sinkage,
        w.groove,
        w.sidewall,
        w.smear,
        w.mud,
        patch.grip,
        patch.mu,
        patch.scale,
        patch.length,
        patch.pull,
        world.coat_in_rut,
        sample.kind()
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn patch_camera(world: &World, wheel: usize) -> CameraPose {
    let vehicle = &world.vehicle;
    let local = vehicle.wheel_center_local(wheel);
    let center = vehicle.pos + vehicle.orientation() * local;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    let side = if local.x >= 0.0 { 1.0 } else { -1.0 };
    // Rear quarter on the crown, low enough that the fender stays off the grooves.
    let eye = center - fwd * 0.90 + right * side * 0.50 + Vec3::Y * 0.14;
    CameraPose {
        eye,
        target: center - Vec3::Y * 0.02 + right * side * 0.08,
        ortho_width: Some(1.18),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_moisture(renderer: &mut Renderer, parsed: &CaptureArgs) {
    use silt::soil::SoilField;

    let mut world = World::flat(0.2);
    world.bike.pos = Vec3::new(14.0, 0.4, -14.0);
    park_off(&mut world);
    world.soil = SoilField::moisture_scene();
    let wet = moisture_clay(&world.soil);
    print_moisture("wet", &wet);
    print_volumes("wet", &world.soil);
    write_cc0_still(
        renderer,
        &world,
        "wet",
        &parsed.moisture_wet,
        parsed,
        moisture_camera(),
    );

    let cut = world.soil.cut_volume();
    let placed = world.soil.placed_volume();
    world.soil.advance_weather(150.0, 0.0);
    let dry = moisture_clay(&world.soil);
    print_moisture("dry", &dry);
    print_volumes("dry", &world.soil);
    println!(
        "grip mu {:+.3}  ({:.3} -> {:.3})  phi {:.3} -> {:.3}  sink {:+.4} m  stick {:+.0} Pa",
        dry.mu - wet.mu,
        wet.mu,
        dry.mu,
        wet.phi,
        dry.phi,
        dry.sink - wet.sink,
        dry.stick - wet.stick
    );
    let cut_after = world.soil.cut_volume();
    let placed_after = world.soil.placed_volume();
    println!(
        "volume cut {cut:.4} -> {cut_after:.4}  placed {placed:.4} -> {placed_after:.4}"
    );
    write_cc0_still(
        renderer,
        &world,
        "dry",
        &parsed.moisture_dry,
        parsed,
        moisture_camera(),
    );
}

struct ClayRead {
    mu: f32,
    phi: f32,
    sink: f32,
    stick: f32,
    kind: silt::soil::SurfaceKind,
    wetness: f32,
}

fn moisture_clay(soil: &silt::soil::SoilField) -> ClayRead {
    use silt::soil::{grip_mu, sinkage_on, stickiness, terramech_for};
    let s = soil.sample(0.78, 0.0);
    let kind = s.kind();
    ClayRead {
        mu: grip_mu(kind, s.wetness, s.loose, s.water),
        phi: terramech_for(kind, s.softness, s.water).phi,
        sink: sinkage_on(1_600.0, kind, s.wetness, s.loose, s.water, 0.38, 0.28),
        stick: stickiness(kind, s.wetness, s.loose, s.water),
        kind,
        wetness: s.wetness,
    }
}

fn print_moisture(label: &str, c: &ClayRead) {
    println!(
        "{label} kind {:?}  wetness {:.3}  grip mu {:.3}  phi {:.3}  sink {:.4} m  stick {:.0} Pa",
        c.kind, c.wetness, c.mu, c.phi, c.sink, c.stick
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn moisture_camera() -> CameraPose {
    // Silt in the left foreground, the clay rut in the middle, sand and gravel to the right.
    CameraPose {
        eye: Vec3::new(-2.15, 1.02, 2.25),
        target: Vec3::new(0.28, 0.0, 0.05),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn volume_berm_camera() -> CameraPose {
    // Three-quarter view so the pile beside the cut reads as a berm.
    CameraPose {
        eye: Vec3::new(-1.35, 0.78, 1.7),
        target: Vec3::new(0.18, 0.02, -0.15),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn volume_second_camera() -> CameraPose {
    CameraPose {
        eye: Vec3::new(-1.45, 0.95, 1.8),
        target: Vec3::new(0.16, 0.08, -0.15),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn volume_slope_camera() -> CameraPose {
    // Down the ramp. Grooves sit on the lower face and stop; the buggy is above them.
    CameraPose {
        eye: Vec3::new(-2.85, 2.25, 2.15),
        target: Vec3::new(-0.2, 0.32, 0.05),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn capture_cc0(renderer: &mut Renderer, parsed: &CaptureArgs) {
    let mut world = World::demo();
    let dt = 1.0 / 60.0;
    let steps = (12.0f32 / dt).round() as u32;
    println!("cc0 drive {steps} steps");
    let mut iso_done = false;
    for _ in 0..steps {
        let bike = demo_drive_bike(&world);
        world.step_bike(dt, &bike);
        let input = demo_drive(&world);
        world.step(dt, &input);
        if !iso_done && world.time >= 3.2 {
            write_cc0_still(renderer, &world, "iso", &parsed.cc0_iso, parsed, cc0_iso_camera());
            // Same moment: the tires are on the lane, not buried in the north pond.
            write_cc0_still(
                renderer,
                &world,
                "tire",
                &parsed.cc0_tire,
                parsed,
                cc0_tire_camera(&world),
            );
            iso_done = true;
        }
    }
    if !iso_done {
        write_cc0_still(renderer, &world, "iso", &parsed.cc0_iso, parsed, cc0_iso_camera());
        write_cc0_still(
            renderer,
            &world,
            "tire",
            &parsed.cc0_tire,
            parsed,
            cc0_tire_camera(&world),
        );
    }
    println!(
        "cc0 t={:.2} pos=({:.2},{:.2}) yaw={:.2} rut={:.3}",
        world.time,
        world.vehicle.pos.x,
        world.vehicle.pos.z,
        world.vehicle.yaw,
        world.soil.max_rut_depth()
    );
    write_cc0_still(
        renderer,
        &world,
        "puddle",
        &parsed.cc0_puddle,
        parsed,
        cc0_puddle_camera(&world),
    );
}

/// Volume numbers on the proof still. A dark plate, so the figures stay
/// readable on the mud without a shader change.
#[cfg(not(target_arch = "wasm32"))]
fn stamp_proof_lines(rgba: &mut [u8], width: u32, height: u32, lines: &[&str]) {
    let scale = 3u32;
    let gap = scale;
    let cw = 5 * scale + gap;
    let ch = 7 * scale + gap * 2;
    let x0 = 24u32;
    let y0 = 16u32;
    let mut max_w = 0u32;
    for line in lines {
        max_w = max_w.max(line.chars().count() as u32 * cw);
    }
    let plate_w = max_w + 28;
    let plate_h = lines.len() as u32 * ch + 18;
    for y in y0..y0.saturating_add(plate_h).min(height) {
        for x in x0..x0.saturating_add(plate_w).min(width) {
            let i = ((y * width + x) * 4) as usize;
            rgba[i] = 18;
            rgba[i + 1] = 14;
            rgba[i + 2] = 10;
            rgba[i + 3] = 255;
        }
    }
    for (li, line) in lines.iter().enumerate() {
        let y = y0 + 8 + li as u32 * ch;
        let mut x = x0 + 14;
        for ch in line.chars() {
            blit_glyph(rgba, width, height, x, y, ch, scale);
            x += cw;
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn blit_glyph(rgba: &mut [u8], width: u32, height: u32, x: u32, y: u32, ch: char, scale: u32) {
    let Some(rows) = glyph_rows(ch) else {
        return;
    };
    for (row, bits) in rows.iter().enumerate() {
        for col in 0..5u32 {
            if bits & (1 << (4 - col)) == 0 {
                continue;
            }
            for dy in 0..scale {
                for dx in 0..scale {
                    let px = x + col * scale + dx;
                    let py = y + row as u32 * scale + dy;
                    if px >= width || py >= height {
                        continue;
                    }
                    let i = ((py * width + px) * 4) as usize;
                    rgba[i] = 236;
                    rgba[i + 1] = 228;
                    rgba[i + 2] = 210;
                    rgba[i + 3] = 255;
                }
            }
        }
    }
}

/// 5×7 glyphs, five low bits, left to right. Missing letters are skipped.
#[cfg(not(target_arch = "wasm32"))]
fn glyph_rows(ch: char) -> Option<[u8; 7]> {
    let rows: [u8; 7] = match ch {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        '.' => [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100],
        ' ' => [0, 0, 0, 0, 0, 0, 0],
        'a' => [0b00000, 0b00000, 0b01110, 0b00001, 0b01111, 0b10001, 0b01111],
        'b' => [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
        'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10000, 0b01110],
        'd' => [0b00001, 0b00001, 0b01111, 0b10001, 0b10001, 0b10001, 0b01111],
        'e' => [0b00000, 0b00000, 0b01110, 0b10001, 0b11111, 0b10000, 0b01110],
        'h' => [0b10000, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
        'l' => [0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        'm' => [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001],
        'n' => [0b00000, 0b00000, 0b11110, 0b10001, 0b10001, 0b10001, 0b10001],
        'o' => [0b00000, 0b00000, 0b01110, 0b10001, 0b10001, 0b10001, 0b01110],
        'p' => [0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000],
        'g' => [0b00000, 0b00000, 0b01111, 0b10000, 0b10110, 0b10001, 0b01110],
        'i' => [0b00100, 0b00000, 0b01100, 0b00100, 0b00100, 0b00100, 0b01110],
        'r' => [0b00000, 0b00000, 0b10110, 0b11001, 0b10000, 0b10000, 0b10000],
        's' => [0b00000, 0b00000, 0b01111, 0b10000, 0b01110, 0b00001, 0b11110],
        'v' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100],
        't' => [0b00100, 0b00100, 0b11111, 0b00100, 0b00100, 0b00100, 0b00011],
        'u' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10001, 0b10011, 0b01101],
        'w' => [0b00000, 0b00000, 0b10001, 0b10001, 0b10101, 0b10101, 0b01010],
        'k' => [0b00000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10100, 0b10010],
        'y' => [0b00000, 0b00000, 0b10001, 0b10001, 0b01111, 0b00001, 0b01110],
        _ => return None,
    };
    Some(rows)
}

#[cfg(not(target_arch = "wasm32"))]
fn write_cc0_still(
    renderer: &mut Renderer,
    world: &World,
    name: &str,
    dest: &Path,
    parsed: &CaptureArgs,
    pose: CameraPose,
) {
    let pose = lift_camera(world, pose);
    let rgba = renderer.draw(world, pose, None, true).expect("readback");
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    write_png(dest, parsed.width, parsed.height, &rgba).expect("cc0 still");
    println!(
        "cc0 {name} {} eye=({:.2},{:.2},{:.2}) target=({:.2},{:.2},{:.2})",
        dest.display(),
        pose.eye.x,
        pose.eye.y,
        pose.eye.z,
        pose.target.x,
        pose.target.y,
        pose.target.z
    );
}

#[cfg(not(target_arch = "wasm32"))]
fn cc0_iso_camera() -> CameraPose {
    let target = Vec3::new(0.0, 0.55, 1.4);
    let eye = target + Vec3::new(-16.8, 13.6, -15.0);
    CameraPose {
        eye,
        target,
        ortho_width: Some(36.0),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cc0_tire_camera(world: &World) -> CameraPose {
    let vehicle = &world.vehicle;
    // Rear tire: the tread blocks are larger. Sit outside the body, above the berm.
    let local = vehicle.wheel_center_local(2);
    let center = vehicle.pos + vehicle.orientation() * local;
    let fwd = vehicle.forward();
    let right = Vec3::new(fwd.z, 0.0, -fwd.x);
    // Side-on, near hub height, so the tread blocks face the lens.
    let mut eye = center - right * 0.92 - fwd * 0.16 + Vec3::Y * 0.05;
    let h = world.soil.sample(eye.x, eye.z).height;
    if eye.y < h + 0.06 {
        eye.y = h + 0.06;
    }
    CameraPose {
        eye,
        target: center,
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn cc0_puddle_camera(world: &World) -> CameraPose {
    let (x, z, surface) = standing_rut(&world.soil, &world.vehicle, &world.bike, &world.particles);
    // Clear the rut lips. A camera inside the groove looks at the underside of the mesh.
    let crown = |x: f32, z: f32| -> f32 {
        let mut h = world.soil.sample(x, z).height;
        for (dx, dz) in [
            (0.8, 0.0),
            (-0.8, 0.0),
            (0.0, 0.8),
            (0.0, -0.8),
            (1.4, 0.0),
            (-1.4, 0.0),
            (0.0, 1.4),
            (0.0, -1.4),
        ] {
            h = h.max(world.soil.sample(x + dx, z + dz).height);
        }
        h
    };
    // Steep look down into the groove. A low angle hits the rut lip and the buggy.
    let mut eye = Vec3::new(x + 0.42, 0.0, z - 0.62);
    eye.y = crown(x, z).max(crown(eye.x, eye.z)) + 1.85;
    CameraPose {
        eye,
        target: Vec3::new(x, surface + 0.02, z),
        ortho_width: None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn standing_rut(
    soil: &silt::soil::SoilField,
    vehicle: &silt::vehicle::Vehicle,
    bike: &silt::bike::Bike,
    particles: &silt::particles::Particles,
) -> (f32, f32, f32) {
    let res = soil.res;
    let clear_of = |x: f32, z: f32, min_d: f32| -> bool {
        let dv = (x - vehicle.pos.x).hypot(z - vehicle.pos.z);
        let db = (x - bike.pos.x).hypot(z - bike.pos.z);
        dv >= min_d && db >= min_d
    };
    let pick = |min_d: f32, min_water: f32, min_rut: f32| -> Option<(f32, f32, f32, f32)> {
        let mut best_score = -1.0f32;
        let mut best = None;
        for j in 1..res - 1 {
            for i in 1..res - 1 {
                let idx = j * res + i;
                let water = soil.water[idx];
                let rut = (soil.initial_height[idx] - soil.height[idx]).max(0.0);
                if water < min_water || rut < min_rut {
                    continue;
                }
                let x = (i as f32 + 0.5) * soil.cell - soil.size * 0.5;
                let z = (j as f32 + 0.5) * soil.cell - soil.size * 0.5;
                if !clear_of(x, z, min_d) {
                    continue;
                }
                let mut pond = 0.0f32;
                for dj in -2isize..=2 {
                    for di in -2isize..=2 {
                        let n = ((j as isize + dj) as usize) * res + (i as isize + di) as usize;
                        if soil.water[n] > 0.015 {
                            pond += 1.0;
                        }
                    }
                }
                let mut clods = 0.0f32;
                for p in &particles.items {
                    if p.alive && (p.pos.x - x).hypot(p.pos.z - z) < 1.2 {
                        clods += 1.0;
                    }
                }
                let score = water * 3.2 + rut * 1.6 + pond * 0.045 - clods * 0.55;
                if score > best_score {
                    best_score = score;
                    best = Some((x, z, soil.height[idx] + water.min(0.06) * 0.2, score));
                }
            }
        }
        best
    };
    if let Some((x, z, y, score)) = pick(5.5, 0.018, 0.01)
        .or_else(|| pick(3.6, 0.015, 0.008))
        .or_else(|| pick(2.2, 0.012, 0.006))
    {
        println!("cc0 puddle score={score:.3} at ({x:.2},{z:.2})");
        (x, z, y)
    } else {
        println!("cc0 puddle fallback");
        (1.15, 9.25, 0.04)
    }
}

#[cfg(target_arch = "wasm32")]
struct WebGpu {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    srgb: bool,
    label: String,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_GPU: std::cell::RefCell<Option<WebGpu>> = std::cell::RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
fn web_canvas() -> web_sys::HtmlCanvasElement {
    use wasm_bindgen::JsCast;
    let window = web_sys::window().expect("window");
    let document = window.document().expect("document");
    let canvas = document
        .get_element_by_id("silt")
        .expect("canvas #silt")
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .expect("canvas");
    let mut width = canvas.client_width();
    let mut height = canvas.client_height();
    if width < 32 || height < 32 {
        width = 1280;
        height = 720;
    }
    canvas.set_width(width as u32);
    canvas.set_height(height as u32);
    canvas
}

#[cfg(target_arch = "wasm32")]
fn set_web_status(text: &str) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Some(document) = window.document() else {
        return;
    };
    if let Some(node) = document.get_element_by_id("backend") {
        node.set_text_content(Some(text));
    }
}

#[cfg(target_arch = "wasm32")]
fn force_webgl() -> bool {
    web_sys::window()
        .and_then(|window| window.location().search().ok())
        .map(|query| {
            query.split(['?', '&']).any(|part| {
                let part = part.trim();
                part == "gl" || part.starts_with("gl=")
            })
        })
        .unwrap_or(false)
}

#[cfg(target_arch = "wasm32")]
fn backend_label(info: &wgpu::AdapterInfo) -> String {
    let kind = match info.backend {
        wgpu::Backend::BrowserWebGpu => "WebGPU",
        wgpu::Backend::Gl => "WebGL2",
        other => return format!("{other:?} · {}", info.name),
    };
    format!("{kind} · {}", info.name)
}

#[cfg(target_arch = "wasm32")]
async fn build_web_gpu(window: std::sync::Arc<Window>) -> Result<WebGpu, String> {
    let force_gl = force_webgl();
    let mut desc = wgpu::InstanceDescriptor::default();
    let instance = if force_gl {
        desc.backends = wgpu::Backends::GL;
        wgpu::Instance::new(desc)
    } else {
        wgpu::util::new_instance_with_webgpu_detection(desc).await
    };
    let surface = instance
        .create_surface(window.clone())
        .map_err(|err| format!("surface: {err}"))?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        })
        .await
        .ok_or_else(|| {
            if force_gl {
                "нет адаптера WebGL2".to_string()
            } else {
                "нет адаптера WebGPU или WebGL2".to_string()
            }
        })?;
    let info = adapter.get_info();
    let label = backend_label(&info);
    web_sys::console::log_1(&format!("silt {label}").into());
    let limits = wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
    let limits = if limits.check_limits(&adapter.limits()) {
        limits
    } else {
        adapter.limits()
    };
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("silt"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|err| format!("device: {err}"))?;
    let size = window.inner_size();
    let max_dim = device.limits().max_texture_dimension_2d;
    let width = size.width.max(1).min(max_dim);
    let height = size.height.max(1).min(max_dim);
    let caps = surface.get_capabilities(&adapter);
    if caps.formats.is_empty() {
        return Err("поверхность без формата".into());
    }
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|fmt| {
            matches!(
                fmt,
                wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm
            )
        })
        .unwrap_or(caps.formats[0]);
    let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Fifo) {
        wgpu::PresentMode::Fifo
    } else {
        caps.present_modes[0]
    };
    let config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width,
        height,
        present_mode,
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&device, &config);
    let formats = render::gpu_formats(&adapter);
    set_web_status("грунт…");
    let scene = cc0::load().await;
    let mut renderer = Renderer::new(
        device,
        queue,
        info.name,
        config.width,
        config.height,
        formats,
        scene,
    );
    renderer.set_present_format(format);
    Ok(WebGpu {
        surface,
        config,
        renderer,
        srgb: format.is_srgb(),
        label,
    })
}
