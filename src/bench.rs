//! `silt bench`: the frame benchmark Vitalii runs on his own GPU, and
//! `silt budget`: the static GPU budget check CI runs on lavapipe.
//!
//! The bench opens a real window (what a player sees), turns vsync off,
//! steps a fixed scene with a fixed dt per frame (so every machine draws the
//! same frames), and prints CPU and per-pass GPU milliseconds against
//! `silt::quality::pass_budgets`. `--out` writes the same numbers as JSON.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use silt::quality::{self, Preset, GPU_FRAME_BUDGET_MS};
use silt::world::{demo_camera, demo_drive, demo_drive_bike, World};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use crate::gpu_time::{self, GpuSample};
use crate::render::{self, Renderer};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scene {
    /// The window's loop: buggy + bike on the demo track.
    Lap,
    /// Niva through-drive with the fluid sheet and dust published to the
    /// render contract. The heavier scene, closest to the vertical slice.
    Niva,
}

impl Scene {
    fn name(self) -> &'static str {
        match self {
            Scene::Lap => "lap",
            Scene::Niva => "niva",
        }
    }
}

pub struct BenchArgs {
    pub preset: Preset,
    pub scene: Scene,
    pub width: u32,
    pub height: u32,
    pub warmup: u32,
    pub frames: u32,
    pub vsync: bool,
    pub out: Option<PathBuf>,
}

pub fn parse_bench(args: &[String]) -> BenchArgs {
    let mut b = BenchArgs {
        preset: Preset::Medium,
        scene: Scene::Lap,
        width: 1920,
        height: 1080,
        warmup: 120,
        frames: 1200,
        vsync: false,
        out: None,
    };
    if let Some(p) = std::env::var("SILT_QUALITY")
        .ok()
        .and_then(|v| Preset::parse(&v))
    {
        b.preset = p;
    }
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().unwrap_or_default();
        match a.as_str() {
            "--quality" => {
                let v = val();
                b.preset =
                    Preset::parse(&v).unwrap_or_else(|| panic!("--quality {v}: low|medium|high"));
            }
            "--scene" => {
                b.scene = match val().as_str() {
                    "niva" => Scene::Niva,
                    _ => Scene::Lap,
                }
            }
            "--width" => b.width = val().parse().expect("--width"),
            "--height" => b.height = val().parse().expect("--height"),
            "--warmup" => b.warmup = val().parse().expect("--warmup"),
            "--frames" => b.frames = val().parse().expect("--frames"),
            "--vsync" => b.vsync = true,
            "--out" => b.out = Some(PathBuf::from(val())),
            other => eprintln!("bench: ignored argument {other}"),
        }
    }
    b
}

/// Rolling per-frame numbers, shared by the bench and the window overlay.
#[derive(Default)]
pub struct FrameStats {
    pub wall_ms: Vec<f64>,
    pub sim_ms: Vec<f64>,
    pub encode_ms: Vec<f64>,
    pub gpu_frame_ms: Vec<f64>,
    pub passes: Vec<(&'static str, Vec<f64>)>,
    pub draws: Vec<u32>,
    pub tris: Vec<u64>,
}

impl FrameStats {
    pub fn push_gpu(&mut self, s: &GpuSample) {
        if let Some(ms) = s.frame_ms {
            self.gpu_frame_ms.push(ms);
        }
        for (label, ms) in &s.passes {
            match self.passes.iter_mut().find(|(l, _)| l == label) {
                Some((_, v)) => v.push(*ms),
                None => self.passes.push((label, vec![*ms])),
            }
        }
    }
}

pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

pub fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((s.len() as f64 - 1.0) * p).round() as usize;
    s[i.min(s.len() - 1)]
}

enum SceneState {
    Lap(Box<World>),
    Niva(Box<silt::ThroughDrive>, u32),
}

impl SceneState {
    fn new(scene: Scene) -> Self {
        match scene {
            Scene::Lap => SceneState::Lap(Box::new(World::demo())),
            Scene::Niva => SceneState::Niva(Box::new(silt::ThroughDrive::new()), 0),
        }
    }

    /// One frame of simulation at a fixed dt, so every machine draws the same frames.
    fn step(&mut self) {
        match self {
            SceneState::Lap(world) => {
                let dt = 1.0 / 60.0;
                let bike = demo_drive_bike(world);
                let buggy = demo_drive(world);
                world.step_bike(dt, &bike);
                world.step(dt, &buggy);
            }
            SceneState::Niva(drive, frame) => {
                // The drive is tuned at dt = 1/30 (docs/PERF_BUDGET.md): one
                // step every second rendered frame.
                if *frame % 2 == 0 {
                    drive.step(1.0 / 30.0);
                }
                *frame += 1;
                if drive.world.vehicle.pos.z > 19.6 {
                    **drive = silt::ThroughDrive::new();
                }
            }
        }
    }

    fn draw(&self, r: &mut Renderer, present: Option<(&wgpu::TextureView, bool)>) {
        match self {
            SceneState::Lap(world) => {
                let cam = demo_camera(world.time, world.vehicle.pos, world.vehicle.yaw);
                r.draw(world, cam, present, false);
            }
            SceneState::Niva(drive, _) => {
                let cam = crate::through_chase(&drive.world);
                r.draw_with_sheet(
                    &drive.world,
                    cam,
                    present,
                    false,
                    Some(&drive.fluid),
                    Some(drive.dust()),
                );
            }
        }
    }

    fn configure(&self, r: &mut Renderer) {
        if let SceneState::Niva(..) = self {
            r.show_bike = false;
            r.show_props = false;
            r.show_hills = false;
            r.niva_picture = true;
            r.water_column = true;
        }
    }
}

struct BenchApp {
    args: BenchArgs,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    renderer: Option<Renderer>,
    srgb: bool,
    adapter: String,
    backend: String,
    scene: SceneState,
    frame: u32,
    stats: FrameStats,
    last: Option<Instant>,
    timed: bool,
}

pub fn run_bench(args: BenchArgs) {
    quality::set_active(args.preset);
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let scene = SceneState::new(args.scene);
    let mut app = BenchApp {
        args,
        window: None,
        surface: None,
        renderer: None,
        srgb: false,
        adapter: String::new(),
        backend: String::new(),
        scene,
        frame: 0,
        stats: FrameStats::default(),
        last: None,
        timed: false,
    };
    event_loop.run_app(&mut app).expect("run");
}

impl ApplicationHandler for BenchApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Silt bench")
                        .with_resizable(false)
                        .with_inner_size(winit::dpi::PhysicalSize::new(
                            self.args.width,
                            self.args.height,
                        )),
                )
                .expect("window"),
        );
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let (adapter, device, queue) =
            render::open_for_surface(&instance, &surface).unwrap_or_else(|e| panic!("{e}"));
        let info = adapter.get_info();
        self.adapter = info.name.clone();
        self.backend = format!("{:?}", info.backend);
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
        self.srgb = format.is_srgb();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if self.args.vsync {
                wgpu::PresentMode::Fifo
            } else {
                wgpu::PresentMode::AutoNoVsync
            },
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        let formats = render::gpu_formats(&adapter);
        eprintln!("bench: loading scene");
        let cc0 = crate::cc0::load();
        let mut r = Renderer::new(
            device,
            queue,
            info.name.clone(),
            config.width,
            config.height,
            formats,
            cc0,
        );
        r.set_present_format(format);
        r.resize(config.width, config.height);
        self.timed = r.enable_gpu_time();
        self.scene.configure(&mut r);
        if (config.width, config.height) != (self.args.width, self.args.height) {
            eprintln!(
                "bench: window is {}x{}, asked {}x{} (display scaling?)",
                config.width, config.height, self.args.width, self.args.height
            );
        }
        self.window = Some(window);
        self.surface = Some(surface);
        self.renderer = Some(r);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                if self.frame_once() {
                    self.report();
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

impl BenchApp {
    /// Returns true when the measured frames are done.
    fn frame_once(&mut self) -> bool {
        let (Some(r), Some(surface)) = (self.renderer.as_mut(), self.surface.as_ref()) else {
            return false;
        };
        let measuring = self.frame >= self.args.warmup;
        let t0 = Instant::now();
        self.scene.step();
        let t1 = Instant::now();
        let Ok(tex) = surface.get_current_texture() else {
            return false;
        };
        let view = tex.texture.create_view(&Default::default());
        render::take_draw_stats();
        self.scene.draw(r, Some((&view, self.srgb)));
        let (draws, tris) = render::take_draw_stats();
        let t2 = Instant::now();
        tex.present();
        if let Some(s) = r.poll_gpu_time() {
            if measuring {
                self.stats.push_gpu(&s);
            }
        }
        let now = Instant::now();
        if measuring {
            if let Some(last) = self.last {
                self.stats.wall_ms.push((now - last).as_secs_f64() * 1e3);
            }
            self.stats.sim_ms.push((t1 - t0).as_secs_f64() * 1e3);
            self.stats.encode_ms.push((t2 - t1).as_secs_f64() * 1e3);
            self.stats.draws.push(draws);
            self.stats.tris.push(tris);
        }
        self.last = Some(now);
        self.frame += 1;
        if self.frame % 240 == 0 {
            eprintln!(
                "bench: frame {}/{}",
                self.frame,
                self.args.warmup + self.args.frames
            );
        }
        self.frame >= self.args.warmup + self.args.frames
    }

    fn report(&mut self) {
        let r = self.renderer.as_mut().unwrap();
        // Frames still in flight.
        let tail = r.take_gpu_time();
        self.stats.push_gpu(&tail);
        let q = r.quality();
        let (iw, ih) = r.internal_size();
        let s = &self.stats;
        let wall = mean(&s.wall_ms);
        let fps = if wall > 0.0 { 1000.0 / wall } else { 0.0 };
        let p99 = pct(&s.wall_ms, 0.99);
        let gpu = mean(&s.gpu_frame_ms);
        let gpu95 = pct(&s.gpu_frame_ms, 0.95);
        let gpu_ok = !s.gpu_frame_ms.is_empty() && gpu95 <= GPU_FRAME_BUDGET_MS as f64;
        let mut out = Vec::new();
        out.push(format!(
            "BENCH adapter=\"{}\" backend={} cpu=\"{}\"",
            self.adapter,
            self.backend,
            gpu_time::host_cpu_line()
        ));
        out.push(format!(
            "BENCH preset={} scene={} size={}x{} internal={iw}x{ih} vsync={} frames={}",
            q.preset.name(),
            self.args.scene.name(),
            self.args.width,
            self.args.height,
            self.args.vsync,
            s.wall_ms.len()
        ));
        out.push(format!(
            "BENCH_FPS mean={fps:.1} p99_frame_ms={p99:.2} (1% low {:.1} FPS)",
            if p99 > 0.0 { 1000.0 / p99 } else { 0.0 }
        ));
        out.push(format!(
            "BENCH_CPU frame_mean={wall:.3} p95={:.3} sim_mean={:.3} sim_p95={:.3} encode_mean={:.3}",
            pct(&s.wall_ms, 0.95),
            mean(&s.sim_ms),
            pct(&s.sim_ms, 0.95),
            mean(&s.encode_ms)
        ));
        if s.gpu_frame_ms.is_empty() {
            out.push(gpu_time::frame_uncollected(if self.timed {
                "timestamp-unread"
            } else {
                "no-timestamp-query"
            }));
        } else {
            out.push(format!(
                "BENCH_GPU frame_mean={gpu:.3} p95={gpu95:.3} budget={GPU_FRAME_BUDGET_MS:.1} {}",
                if gpu_ok { "PASS" } else { "OVER" }
            ));
            for (label, v) in &s.passes {
                let m = mean(v);
                let b = quality::pass_budget(q.preset, label);
                let verdict = match b {
                    Some(b) if b > 0.0 && m > b as f64 => "OVER",
                    Some(_) => "ok",
                    None => "no-budget",
                };
                out.push(format!(
                    "BENCH_PASS {label} mean={m:.3} p95={:.3} budget={} {verdict}",
                    pct(v, 0.95),
                    b.map(|b| format!("{b:.2}")).unwrap_or_else(|| "-".into())
                ));
            }
        }
        out.push(format!(
            "BENCH_DRAWS calls_max={} tris_max={}",
            s.draws.iter().max().copied().unwrap_or(0),
            s.tris.iter().max().copied().unwrap_or(0)
        ));
        for line in &out {
            println!("{line}");
        }
        if let Some(path) = &self.args.out {
            let mut passes = String::new();
            for (i, (label, v)) in s.passes.iter().enumerate() {
                let b = quality::pass_budget(q.preset, label).unwrap_or(-1.0);
                passes.push_str(&format!(
                    "{}\n    {{\"pass\": \"{label}\", \"mean_ms\": {:.4}, \"p95_ms\": {:.4}, \"budget_ms\": {b}}}",
                    if i == 0 { "" } else { "," },
                    mean(v),
                    pct(v, 0.95)
                ));
            }
            let json = format!(
                "{{\n  \"adapter\": {:?},\n  \"backend\": {:?},\n  \"cpu\": {:?},\n  \"preset\": {:?},\n  \"scene\": {:?},\n  \"width\": {},\n  \"height\": {},\n  \"internal_width\": {iw},\n  \"internal_height\": {ih},\n  \"vsync\": {},\n  \"frames\": {},\n  \"fps_mean\": {fps:.2},\n  \"frame_ms_mean\": {wall:.4},\n  \"frame_ms_p95\": {:.4},\n  \"frame_ms_p99\": {p99:.4},\n  \"sim_ms_mean\": {:.4},\n  \"encode_ms_mean\": {:.4},\n  \"gpu_frame_ms_mean\": {},\n  \"gpu_frame_ms_p95\": {},\n  \"gpu_budget_ms\": {GPU_FRAME_BUDGET_MS},\n  \"draw_calls_max\": {},\n  \"triangles_max\": {},\n  \"passes\": [{passes}\n  ]\n}}\n",
                self.adapter,
                self.backend,
                gpu_time::host_cpu_line(),
                q.preset.name(),
                self.args.scene.name(),
                self.args.width,
                self.args.height,
                self.args.vsync,
                s.wall_ms.len(),
                pct(&s.wall_ms, 0.95),
                mean(&s.sim_ms),
                mean(&s.encode_ms),
                if s.gpu_frame_ms.is_empty() { "null".to_string() } else { format!("{gpu:.4}") },
                if s.gpu_frame_ms.is_empty() { "null".to_string() } else { format!("{gpu95:.4}") },
                s.draws.iter().max().copied().unwrap_or(0),
                s.tris.iter().max().copied().unwrap_or(0),
            );
            if let Some(dir) = path.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            std::fs::write(path, json).expect("write bench json");
            println!("BENCH_OUT {}", path.display());
        }
    }
}

/// `silt budget [--software] [--scene lap|niva]`: offscreen frames at
/// 1920×1080 per preset; asserts draw calls, triangles, texture and buffer
/// memory against `silt::quality::static_budget`. Exits 1 on a breach.
pub fn run_budget(args: &[String]) {
    let software = args.iter().any(|a| a == "--software");
    let scene = if args.iter().any(|a| a == "niva") {
        Scene::Niva
    } else {
        Scene::Lap
    };
    if software {
        render::prepare_software_vulkan();
    }
    let (w, h) = (1920u32, 1080u32);
    // `GPU_BUDGET_ALLOW=triangles,texture_mib` reports those rows without
    // failing. CI allows `triangles` until the CDLOD terrain (#120) lands.
    let allow: Vec<String> = std::env::var("GPU_BUDGET_ALLOW")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let mut failed = false;
    for preset in Preset::ALL {
        quality::set_active(preset);
        let (_i, adapter, device, queue) =
            render::try_open_adapter(software, None).unwrap_or_else(|e| panic!("{e}"));
        let formats = render::gpu_formats(&adapter);
        let cc0 = crate::cc0::load();
        let mut r = Renderer::new(device, queue, adapter.get_info().name, w, h, formats, cc0);
        r.resize(w, h);
        let mut st = SceneState::new(scene);
        st.configure(&mut r);
        let (mut draws, mut tris) = (0u32, 0u64);
        for _ in 0..3 {
            st.step();
            render::take_draw_stats();
            st.draw(&mut r, None);
            let (d, t) = render::take_draw_stats();
            draws = draws.max(d);
            tris = tris.max(t);
        }
        let _ = r.device.poll(wgpu::Maintain::Wait);
        let c = r.device.get_internal_counters();
        let tex_mib = c.hal.texture_memory.read() as f64 / (1024.0 * 1024.0);
        let buf_mib = c.hal.buffer_memory.read() as f64 / (1024.0 * 1024.0);
        let b = quality::static_budget(preset);
        let mut line = |name: &str, v: f64, cap: f64, unit: &str| {
            let ok = v <= cap;
            let allowed = allow.iter().any(|a| a == name);
            failed |= !ok && !allowed;
            println!(
                "GPU_BUDGET preset={} scene={} {name}={v:.1}{unit} cap={cap:.1}{unit} {}",
                preset.name(),
                scene.name(),
                match (ok, allowed) {
                    (true, _) => "ok",
                    (false, true) => "OVER (allowed)",
                    (false, false) => "OVER",
                }
            );
        };
        line("draw_calls", draws as f64, b.draw_calls as f64, "");
        line("triangles", tris as f64, b.triangles as f64, "");
        if c.hal.texture_memory.read() == 0 && c.hal.textures.read() == 0 {
            println!(
                "GPU_BUDGET preset={} memory=unavailable (backend without counters)",
                preset.name()
            );
        } else {
            line("texture_mib", tex_mib, b.texture_mib, "");
            line("buffer_mib", buf_mib, b.buffer_mib, "");
        }
        let (iw, ih) = r.internal_size();
        println!(
            "GPU_BUDGET preset={} internal={iw}x{ih} shadow={} textures={} buffers={} pipelines={}",
            preset.name(),
            preset.quality().shadow_res,
            c.hal.textures.read(),
            c.hal.buffers.read(),
            c.hal.render_pipelines.read()
        );
    }
    if failed {
        eprintln!("GPU budget exceeded; see docs/GPU_BUDGET.md");
        std::process::exit(1);
    }
}
