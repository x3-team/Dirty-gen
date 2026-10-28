//! Silt window and offscreen capture.
//!
//! `silt` drives the buggy. `silt capture` steps the same solver and writes frames.

mod mesh;
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
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

/// Chase-camera distance. Scroll multiplies this; the wheel cannot pass the ends.
const ORBIT_DIST_MIN: f32 = 2.6;
const ORBIT_DIST_MAX: f32 = 24.0;
const ORBIT_DIST_DEFAULT: f32 = 8.4;
/// Same pitch stops R and F already used.
const ORBIT_PITCH_MIN: f32 = -0.35;
const ORBIT_PITCH_MAX: f32 = 1.1;
/// CSS pixels, so a high-DPI canvas does not orbit faster.
const ORBIT_YAW_PER_PX: f32 = 0.0045;
const ORBIT_PITCH_PER_PX: f32 = 0.0038;
/// One mouse notch. Positive is scroll-up, which moves the camera closer.
const ORBIT_ZOOM_PER_LINE: f32 = 0.16;

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
    orbit_dist: f32,
    chase_cam: bool,
    /// Left button is down. A click with no travel only focuses the canvas.
    dragging: bool,
    /// Set once the drag is long enough to count as an orbit, not a click.
    drag_orbit: bool,
    last_cursor: Option<(f64, f64)>,
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
            WindowEvent::MouseInput { state, button, .. } => self.on_mouse_button(state, button),
            WindowEvent::CursorMoved { position, .. } => self.on_cursor_moved(position),
            WindowEvent::MouseWheel { delta, .. } => self.on_scroll(delta),
            WindowEvent::CursorLeft { .. } => self.last_cursor = None,
            WindowEvent::Focused(false) => {
                self.dragging = false;
                self.drag_orbit = false;
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
        let mut renderer = Renderer::new(device, queue, name, config.width, config.height, formats);
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
            self.orbit_pitch =
                (self.orbit_pitch + dt * 0.7).clamp(ORBIT_PITCH_MIN, ORBIT_PITCH_MAX);
        }
        if self.keys.orbit_down {
            self.orbit_pitch =
                (self.orbit_pitch - dt * 0.7).clamp(ORBIT_PITCH_MIN, ORBIT_PITCH_MAX);
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
            chase(
                cam_pos,
                cam_yaw,
                self.orbit_yaw,
                self.orbit_pitch,
                self.orbit_dist,
            )
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

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if button != MouseButton::Left {
            return;
        }
        self.dragging = state == ElementState::Pressed;
        if !self.dragging {
            self.drag_orbit = false;
        }
    }

    fn on_cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        let scale = self
            .window
            .as_ref()
            .map(|window| window.scale_factor())
            .unwrap_or(1.0)
            .max(0.1);
        let x = position.x / scale;
        let y = position.y / scale;
        if self.dragging {
            if let Some((lx, ly)) = self.last_cursor {
                let dx = (x - lx) as f32;
                let dy = (y - ly) as f32;
                // A plain click focuses the page. Orbit starts once the pointer travels.
                if !self.drag_orbit && (dx.abs() > 2.0 || dy.abs() > 2.0) {
                    self.drag_orbit = true;
                    self.chase_cam = true;
                }
                if self.drag_orbit {
                    let (yaw, pitch) = orbit_from_drag(self.orbit_yaw, self.orbit_pitch, dx, dy);
                    self.orbit_yaw = yaw;
                    self.orbit_pitch = pitch;
                }
            }
        }
        self.last_cursor = Some((x, y));
    }

    fn on_scroll(&mut self, delta: MouseScrollDelta) {
        let scale = self
            .window
            .as_ref()
            .map(|window| window.scale_factor())
            .unwrap_or(1.0)
            .max(0.1);
        let lines = scroll_lines(delta, scale);
        if lines.abs() < 1.0e-4 {
            return;
        }
        self.chase_cam = true;
        self.orbit_dist = orbit_distance(self.orbit_dist, lines);
    }
}

/// Third-person orbit around the selected vehicle. `yaw` is the vehicle heading,
/// so the camera stays on the same side of the machine as it steers. `orbit_yaw`
/// and `orbit_pitch` are the Q / E / R / F offsets. `distance` is the scroll zoom.
fn chase(pos: Vec3, yaw: f32, orbit_yaw: f32, orbit_pitch: f32, distance: f32) -> CameraPose {
    let yaw = yaw + orbit_yaw;
    let elevation = orbit_elevation(orbit_pitch);
    let distance = distance.clamp(ORBIT_DIST_MIN, ORBIT_DIST_MAX);
    let horizontal = elevation.cos() * distance;
    let height = elevation.sin() * distance;
    let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let target = pos + Vec3::Y * 0.55;
    let eye = target - forward * horizontal + Vec3::Y * height;
    CameraPose {
        eye,
        target,
        ortho_width: None,
    }
}

fn orbit_elevation(pitch: f32) -> f32 {
    (0.35 + pitch * 0.72).clamp(0.08, 1.20)
}

/// Drag right looks right. Drag up raises the camera. Pitch stays on the key stops.
fn orbit_from_drag(yaw: f32, pitch: f32, dx: f32, dy: f32) -> (f32, f32) {
    let yaw = yaw + dx * ORBIT_YAW_PER_PX;
    let pitch = (pitch - dy * ORBIT_PITCH_PER_PX).clamp(ORBIT_PITCH_MIN, ORBIT_PITCH_MAX);
    (yaw, pitch)
}

/// `lines` is positive when the wheel scrolls up. That shortens the radius.
fn orbit_distance(current: f32, lines: f32) -> f32 {
    (current * (-lines * ORBIT_ZOOM_PER_LINE).exp()).clamp(ORBIT_DIST_MIN, ORBIT_DIST_MAX)
}

fn scroll_lines(delta: MouseScrollDelta, scale: f64) -> f32 {
    match delta {
        MouseScrollDelta::LineDelta(_, y) => y,
        // winit reports wheel pixels in physical pixels. A notch is about a hundred CSS pixels.
        MouseScrollDelta::PixelDelta(pos) => (pos.y / scale / 100.0) as f32,
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
        orbit_dist: ORBIT_DIST_DEFAULT,
        chase_cam: false,
        dragging: false,
        drag_orbit: false,
        last_cursor: None,
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
    video: PathBuf,
    still_start: PathBuf,
    still_puddle: PathBuf,
    still_ruts: PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
fn capture(args: &[String]) {
    let parsed = parse_capture(args);
    if parsed.software {
        prepare_software_vulkan();
    }
    let (_instance, adapter, device, queue) = open_adapter(parsed.software, None);
    let formats = render::gpu_formats(&adapter);
    let mut renderer = Renderer::new(
        device,
        queue,
        adapter.get_info().name,
        parsed.width,
        parsed.height,
        formats,
    );
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
    let mut video = PathBuf::from(format!("{MEDIA}/mud-demo.mp4"));
    let mut still_start = PathBuf::from(format!("{MEDIA}/mud-demo-start.png"));
    let mut still_puddle = PathBuf::from(format!("{MEDIA}/mud-demo-puddle.png"));
    let mut still_ruts = PathBuf::from(format!("{MEDIA}/mud-demo-ruts.png"));
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
        video,
        still_start,
        still_puddle,
        still_ruts,
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
    let mut renderer = Renderer::new(
        device,
        queue,
        info.name,
        config.width,
        config.height,
        formats,
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

#[cfg(test)]
mod orbit_tests {
    use super::*;

    #[test]
    fn orbit_stays_on_the_selected_vehicle() {
        let pos = Vec3::new(3.0, 0.7, -4.0);
        let pose = chase(pos, 0.4, 0.0, 0.0, ORBIT_DIST_DEFAULT);
        assert!((pose.target.x - pos.x).abs() < 1.0e-4);
        assert!((pose.target.z - pos.z).abs() < 1.0e-4);
        assert!(pose.target.y > pos.y);
        assert!(pose.ortho_width.is_none());

        let forward = Vec3::new(0.4_f32.sin(), 0.0, 0.4_f32.cos());
        let behind = Vec3::new(pose.eye.x - pose.target.x, 0.0, pose.eye.z - pose.target.z);
        assert!(
            behind.dot(forward) < -1.0,
            "the default eye sits behind the nose"
        );
        assert!(pose.eye.y > pose.target.y);

        let swung = chase(
            pos,
            0.4,
            std::f32::consts::FRAC_PI_2,
            0.0,
            ORBIT_DIST_DEFAULT,
        );
        assert!((swung.target - pose.target).length() < 1.0e-4);
        assert!(swung.eye.distance(pose.eye) > 2.0);

        let far = chase(pos, 0.4, 0.0, 0.0, 16.0);
        assert!(far.eye.distance(far.target) > pose.eye.distance(pose.target) + 4.0);
        assert!((far.target - pose.target).length() < 1.0e-4);
    }

    #[test]
    fn drag_and_wheel_share_the_chase_orbit() {
        let (yaw, pitch) = orbit_from_drag(0.0, 0.0, 40.0, 0.0);
        assert!(yaw > 0.1, "drag right looks right");
        assert!(pitch.abs() < 1.0e-5);

        let (_yaw, raised) = orbit_from_drag(0.0, 0.0, 0.0, -30.0);
        assert!(raised > 0.05, "drag up raises the camera");
        let (_yaw, capped) = orbit_from_drag(0.0, ORBIT_PITCH_MAX, 0.0, -80.0);
        assert!((capped - ORBIT_PITCH_MAX).abs() < 1.0e-5);

        let closer = orbit_distance(ORBIT_DIST_DEFAULT, 1.0);
        let farther = orbit_distance(ORBIT_DIST_DEFAULT, -1.0);
        assert!(closer < ORBIT_DIST_DEFAULT);
        assert!(farther > ORBIT_DIST_DEFAULT);
        assert!((orbit_distance(ORBIT_DIST_MIN, 4.0) - ORBIT_DIST_MIN).abs() < 1.0e-4);
        assert!((orbit_distance(ORBIT_DIST_MAX, -4.0) - ORBIT_DIST_MAX).abs() < 1.0e-4);

        let lines = scroll_lines(MouseScrollDelta::LineDelta(0.0, 1.0), 1.0);
        assert!((lines - 1.0).abs() < 1.0e-5);
        let pixels = scroll_lines(
            MouseScrollDelta::PixelDelta(PhysicalPosition::new(0.0, 200.0)),
            2.0,
        );
        assert!((pixels - 1.0).abs() < 1.0e-5);
    }
}
