//! Real-time audio for the mud demo. The engine, exhaust, and mud are
//! recorded clips (see `assets/audio/CREDITS.md`), crossfaded by throttle
//! and wheel slip. The module still compiles if the frame loop never calls it.
//!
//! Wire-up is one call after the fixed step:
//! `silt::audio::tick(&world, dt, drive_bike, throttle)`.

use std::cell::Cell;
#[cfg(target_arch = "wasm32")]
use std::cell::RefCell;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::LazyLock;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

use crate::particles::Particle;
use crate::vehicle::Wheel;
use crate::world::World;

const KIND_CLOD: u8 = 1;
const KIND_CHUNK: u8 = 3;
const CLIP_RATE: f32 = 22_050.0;
const LOOP_FADE: f32 = 240.0;

/// Hear `world` for this frame.
///
/// `dt` is the frame delta in seconds. `drive_bike` selects which vehicle is
/// heard. `throttle` is that vehicle's throttle this frame, in `-1..=1`.
/// On the web the context stays silent until a click or a key. Native
/// playback starts on the first call when a device is present.
pub fn tick(world: &World, dt: f32, drive_bike: bool, throttle: f32) {
    let dt = if dt.is_finite() {
        dt.clamp(0.0, 0.1)
    } else {
        0.0
    };
    let throttle = if throttle.is_finite() {
        throttle.clamp(-1.0, 1.0)
    } else {
        0.0
    };
    let ear = Ear::from_world(world, drive_bike, throttle);
    with_session(|session| {
        let cues = session.audition.hear(&ear, dt);
        session.mixer.aim(DriveState::from_ear(&ear));
        session.mixer.note_throttle(ear.throttle, ear.rpm);
        for cue in cues {
            session.mixer.push(cue);
        }
    });
    host::ensure_output();
}

#[derive(Clone, Copy)]
struct Bit {
    seed: u32,
    kind: u8,
    alive: bool,
    grounded: bool,
    size: f32,
}

struct Ear {
    drive_bike: bool,
    speed: f32,
    rpm: f32,
    slip: f32,
    mud: f32,
    moving: f32,
    throttle: f32,
    water_wheels: u32,
    buggy_vy: f32,
    bike_vy: f32,
    buggy_muddy: bool,
    bike_muddy: bool,
    impact_sprays: u32,
    parts: Vec<Bit>,
}

impl Ear {
    fn from_world(world: &World, drive_bike: bool, throttle: f32) -> Self {
        let buggy_muddy = muddy_at(world, world.vehicle.pos.x, world.vehicle.pos.z);
        let bike_muddy = muddy_at(world, world.bike.pos.x, world.bike.pos.z);
        let (speed, wheels) = if drive_bike {
            (world.bike.speed(), world.bike.wheels.as_slice())
        } else {
            (world.vehicle.speed(), world.vehicle.wheels.as_slice())
        };
        let spin = max_spin(wheels);
        let max_omega = if drive_bike { 46.0 } else { 34.0 };
        let parts = world
            .particles
            .items
            .iter()
            .map(Bit::from_particle)
            .collect();
        Self {
            drive_bike,
            speed,
            rpm: (spin / max_omega).clamp(0.0, 1.0),
            slip: peak_slip(wheels),
            mud: mud_amount(world, wheels, drive_bike),
            moving: smoothstep((speed.abs() - 0.15) / 1.8),
            throttle,
            water_wheels: wheels.iter().filter(|wheel| wheel.in_water).count() as u32,
            buggy_vy: world.vehicle.vel.y,
            bike_vy: world.bike.vel.y,
            buggy_muddy,
            bike_muddy,
            impact_sprays: world.telemetry.impact_sprays,
            parts,
        }
    }
}

impl Bit {
    fn from_particle(p: &Particle) -> Self {
        Self {
            seed: p.seed,
            kind: p.kind,
            alive: p.alive && p.life > 0.0,
            grounded: p.grounded,
            size: p.size,
        }
    }
}

fn muddy_at(world: &World, x: f32, z: f32) -> bool {
    let sample = world.soil.sample(x, z);
    sample.wetness > 0.45 || sample.water > 0.012 || sample.loose > 0.06
}

fn max_spin(wheels: &[Wheel]) -> f32 {
    wheels
        .iter()
        .fold(0.0_f32, |max, wheel| max.max(wheel.omega.abs()))
}

fn peak_slip(wheels: &[Wheel]) -> f32 {
    wheels
        .iter()
        .fold(0.0_f32, |max, wheel| max.max(wheel.slip.abs()))
        .clamp(0.0, 1.0)
}

fn mud_amount(world: &World, wheels: &[Wheel], drive_bike: bool) -> f32 {
    let pos = if drive_bike {
        world.bike.pos
    } else {
        world.vehicle.pos
    };
    let sample = world.soil.sample(pos.x, pos.z);
    let wet = sample.wetness.clamp(0.0, 1.0);
    let water = (sample.water / 0.04).clamp(0.0, 1.0);
    let carried = if wheels.is_empty() {
        0.0
    } else {
        let sum = wheels.iter().fold(0.0_f32, |sum, wheel| sum + wheel.mud);
        (sum / (wheels.len() as f32 * 0.8)).clamp(0.0, 1.0)
    };
    let standing =
        wheels.iter().filter(|wheel| wheel.in_water).count() as f32 / (wheels.len().max(1) as f32);
    wet.max(water).max(carried).max(standing)
}

struct Audition {
    primed: bool,
    prev_sprays: u32,
    prev_buggy_vy: f32,
    prev_bike_vy: f32,
    prev_water: u32,
    prev_parts: Vec<Bit>,
    impact_cd: f32,
    water_cd: f32,
    splat_budget: f32,
    impacts: u32,
}

impl Audition {
    const fn new() -> Self {
        Self {
            primed: false,
            prev_sprays: 0,
            prev_buggy_vy: 0.0,
            prev_bike_vy: 0.0,
            prev_water: 0,
            prev_parts: Vec::new(),
            impact_cd: 0.0,
            water_cd: 0.0,
            splat_budget: 3.0,
            impacts: 0,
        }
    }

    fn hear(&mut self, ear: &Ear, dt: f32) -> Vec<Cue> {
        if !self.primed {
            self.snapshot(ear);
            self.primed = true;
            return Vec::new();
        }
        self.impact_cd = (self.impact_cd - dt).max(0.0);
        self.water_cd = (self.water_cd - dt).max(0.0);
        self.splat_budget = (self.splat_budget + dt * 5.0).min(4.0);
        let mut cues = Vec::new();
        let mut splashed = false;
        if self.impact_cd <= 0.0 && self.impact_hit(ear) {
            self.impacts = self.impacts.wrapping_add(1);
            let fall = self.prev_buggy_vy.min(self.prev_bike_vy).min(0.0).abs();
            let intensity = (0.62 + fall / 10.0).clamp(0.62, 1.0);
            cues.push(Cue::Impact {
                intensity,
                seed: self.impacts.wrapping_mul(0x9E37_79B1),
            });
            if ear.water_wheels > 0 {
                cues.push(Cue::Splash { gain: 0.8 });
                splashed = true;
            }
            self.impact_cd = 0.55;
        }
        if !splashed && self.water_cd <= 0.0 && ear.water_wheels > self.prev_water {
            if ear.speed.abs() > 1.5 {
                cues.push(Cue::Splash { gain: 0.72 });
            } else {
                cues.push(Cue::Puddle { gain: 0.55 });
            }
            self.water_cd = 0.4;
        }
        let mut landings = Vec::new();
        for (index, part) in ear.parts.iter().enumerate() {
            if just_landed(&self.prev_parts, index, part) {
                landings.push((part.kind, part.size, part.seed));
            }
        }
        landings.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)));
        for (kind, size, seed) in landings {
            let cost = if kind == KIND_CHUNK { 0.55 } else { 1.0 };
            if self.splat_budget < cost {
                continue;
            }
            self.splat_budget -= cost;
            cues.push(Cue::Splat { size, seed });
        }
        self.snapshot(ear);
        cues
    }

    fn impact_hit(&self, ear: &Ear) -> bool {
        if ear.impact_sprays > self.prev_sprays {
            return true;
        }
        let fresh = ear
            .parts
            .iter()
            .enumerate()
            .filter(|(index, part)| new_chunk(&self.prev_parts, *index, part))
            .count();
        if fresh >= 4 {
            return true;
        }
        landed_in_mud(self.prev_buggy_vy, ear.buggy_vy, ear.buggy_muddy)
            || landed_in_mud(self.prev_bike_vy, ear.bike_vy, ear.bike_muddy)
    }

    fn snapshot(&mut self, ear: &Ear) {
        self.prev_sprays = ear.impact_sprays;
        self.prev_buggy_vy = ear.buggy_vy;
        self.prev_bike_vy = ear.bike_vy;
        self.prev_water = ear.water_wheels;
        self.prev_parts.clone_from(&ear.parts);
    }
}

fn new_chunk(prev: &[Bit], index: usize, part: &Bit) -> bool {
    if !(part.alive && part.kind == KIND_CHUNK && !part.grounded) {
        return false;
    }
    !matches!(
        prev.get(index),
        Some(old) if old.alive && old.seed == part.seed
    )
}

fn just_landed(prev: &[Bit], index: usize, part: &Bit) -> bool {
    if !(part.alive && part.grounded && (part.kind == KIND_CLOD || part.kind == KIND_CHUNK)) {
        return false;
    }
    match prev.get(index) {
        Some(old) => old.alive && !old.grounded && old.seed == part.seed,
        None => false,
    }
}

fn landed_in_mud(prev_vy: f32, vy: f32, muddy: bool) -> bool {
    muddy && prev_vy < -2.0 && vy > prev_vy + 1.4 && vy > -1.3
}

/// What the mixer is aiming at for this frame. Throttle is already `abs`.
struct DriveState {
    throttle: f32,
    slip: f32,
    rpm: f32,
    mud: f32,
    moving: f32,
    bike: bool,
}

impl DriveState {
    fn from_ear(ear: &Ear) -> Self {
        Self {
            throttle: ear.throttle.abs(),
            slip: ear.slip,
            rpm: ear.rpm,
            mud: ear.mud,
            moving: ear.moving,
            bike: ear.drive_bike,
        }
    }

    const fn parked() -> Self {
        Self {
            throttle: 0.0,
            slip: 0.0,
            rpm: 0.0,
            mud: 0.0,
            moving: 0.0,
            bike: false,
        }
    }
}

struct Layers {
    idle: f32,
    low: f32,
    high: f32,
    load: f32,
    burble: f32,
    tire: f32,
    master: f32,
    bike: bool,
    rpm: f32,
}

/// Crossfade the recorded loops.
///
/// Throttle and wheel rpm move energy from idle to low rpm to high rpm.
/// Wheel slip, deepened by mud, moves the spinning layers onto the loaded
/// exhaust and brings up the wet-tire loop.
const fn layer_gains(drive: &DriveState) -> Layers {
    let throttle = drive.throttle.clamp(0.0, 1.0);
    let slip = drive.slip.clamp(0.0, 1.0);
    let rpm = drive.rpm.clamp(0.0, 1.0);
    let mud = drive.mud.clamp(0.0, 1.0);
    let moving = drive.moving.clamp(0.0, 1.0);
    let rev = (rpm * 0.72 + throttle * 0.28).clamp(0.0, 1.0);
    let mut idle = smoothstep(1.0 - rev / 0.36);
    let mut low = bell(rev, 0.40, 0.34);
    let mut high = smoothstep((rev - 0.46) / 0.40);
    let sum = (idle + low + high).max(1.0e-4);
    idle /= sum;
    low /= sum;
    high /= sum;
    let slip_pull = slip * (0.40 + 0.60 * mud);
    let open = throttle.max(rev);
    let load = (slip_pull * (0.35 + 0.65 * open) + mud * open * 0.18).clamp(0.0, 1.0);
    let keep = 1.0 - 0.72 * load;
    low *= keep;
    high *= keep;
    let burble = (idle * 0.85 + low * 0.35) * (1.0 - 0.65 * high);
    let tire = mud * (0.30 + 0.70 * moving) * (0.30 + 0.70 * slip);
    let master = 0.20 + 0.50 * moving.max(throttle);
    Layers {
        idle,
        low,
        high,
        load,
        burble,
        tire,
        master,
        bike: drive.bike,
        rpm,
    }
}

const fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

const fn bell(x: f32, center: f32, half: f32) -> f32 {
    if half <= 1.0e-4 {
        return 0.0;
    }
    smoothstep(1.0 - ((x - center) / half).abs())
}

struct Bank {
    idle: Vec<f32>,
    low: Vec<f32>,
    high: Vec<f32>,
    load: Vec<f32>,
    burble: Vec<f32>,
    tire: Vec<f32>,
    rise: Vec<f32>,
    splash: Vec<f32>,
    clump: Vec<f32>,
    puddle: Vec<f32>,
    crackle: Vec<f32>,
}

impl Bank {
    fn load() -> Self {
        Self {
            idle: decode_wav(include_bytes!("../assets/audio/engine_idle.wav")),
            low: decode_wav(include_bytes!("../assets/audio/engine_low.wav")),
            high: decode_wav(include_bytes!("../assets/audio/engine_high.wav")),
            load: decode_wav(include_bytes!("../assets/audio/engine_load.wav")),
            burble: decode_wav(include_bytes!("../assets/audio/exhaust_burble.wav")),
            tire: decode_wav(include_bytes!("../assets/audio/tire_mud.wav")),
            rise: decode_wav(include_bytes!("../assets/audio/throttle_rise.wav")),
            splash: decode_wav(include_bytes!("../assets/audio/splash.wav")),
            clump: decode_wav(include_bytes!("../assets/audio/clump_impact.wav")),
            puddle: decode_wav(include_bytes!("../assets/audio/puddle.wav")),
            crackle: decode_wav(include_bytes!("../assets/audio/exhaust_crackle.wav")),
        }
    }
}

fn with_bank<R>(body: impl FnOnce(&Bank) -> R) -> R {
    #[cfg(target_arch = "wasm32")]
    {
        thread_local! {
            static BANK: RefCell<Option<Bank>> = const { RefCell::new(None) };
        }
        BANK.with(|slot| {
            let mut guard = slot.borrow_mut();
            if guard.is_none() {
                *guard = Some(Bank::load());
            }
            body(guard.as_ref().expect("audio bank"))
        })
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        static BANK: LazyLock<Bank> = LazyLock::new(Bank::load);
        body(&BANK)
    }
}

fn decode_wav(bytes: &[u8]) -> Vec<f32> {
    let Some(pcm) = find_wav_data(bytes) else {
        return Vec::new();
    };
    pcm.as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| i16::from_le_bytes(*chunk) as f32 / 32_768.0)
        .collect()
}

fn find_wav_data(bytes: &[u8]) -> Option<&[u8]> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return None;
    }
    let mut offset = 12;
    let mut channels = 0_u16;
    let mut rate = 0_u32;
    let mut bits = 0_u16;
    let mut data: Option<&[u8]> = None;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().ok()?) as usize;
        let start = offset + 8;
        let end = start.checked_add(size)?;
        if end > bytes.len() {
            return None;
        }
        let body = &bytes[start..end];
        if id == b"fmt " && body.len() >= 16 {
            channels = u16::from_le_bytes(body[2..4].try_into().ok()?);
            rate = u32::from_le_bytes(body[4..8].try_into().ok()?);
            bits = u16::from_le_bytes(body[14..16].try_into().ok()?);
        } else if id == b"data" {
            data = Some(body);
        }
        offset = end + (size & 1);
    }
    if channels != 1 || rate != 22_050 || bits != 16 {
        return None;
    }
    data
}

enum Cue {
    Impact { intensity: f32, seed: u32 },
    Splat { size: f32, seed: u32 },
    Splash { gain: f32 },
    Puddle { gain: f32 },
    Rise,
    Crackle,
}

#[derive(Clone, Copy)]
enum ShotKind {
    Clump,
    Splash,
    Puddle,
    Rise,
    Crackle,
}

struct Shot {
    kind: ShotKind,
    cursor: f32,
    step: f32,
    gain: f32,
}

impl Shot {
    fn from_cue(cue: Cue) -> Self {
        match cue {
            Cue::Impact { intensity, seed } => Self {
                kind: ShotKind::Clump,
                cursor: 0.0,
                step: 0.97 + (seed % 5) as f32 * 0.012,
                gain: 0.55 + intensity.clamp(0.0, 1.0) * 0.4,
            },
            Cue::Splat { size, seed } => Self {
                kind: ShotKind::Clump,
                cursor: 0.0,
                step: 0.96 + (seed % 6) as f32 * 0.015,
                gain: 0.28 + size.clamp(0.08, 0.7) * 0.55,
            },
            Cue::Splash { gain } => Self {
                kind: ShotKind::Splash,
                cursor: 0.0,
                step: 1.0,
                gain,
            },
            Cue::Puddle { gain } => Self {
                kind: ShotKind::Puddle,
                cursor: 0.0,
                step: 1.0,
                gain,
            },
            Cue::Rise => Self {
                kind: ShotKind::Rise,
                cursor: 0.0,
                step: 1.0,
                gain: 0.62,
            },
            Cue::Crackle => Self {
                kind: ShotKind::Crackle,
                cursor: 0.0,
                step: 1.0,
                gain: 0.48,
            },
        }
    }
}

struct Mixer {
    sr: f32,
    target: Layers,
    idle_g: f32,
    low_g: f32,
    high_g: f32,
    load_g: f32,
    burble_g: f32,
    tire_g: f32,
    master: f32,
    bike: f32,
    rpm: f32,
    cursor: [f32; 6],
    shots: Vec<Shot>,
    throttle_open: bool,
}

impl Mixer {
    const fn new() -> Self {
        Self {
            sr: 0.0,
            target: layer_gains(&DriveState::parked()),
            idle_g: 0.0,
            low_g: 0.0,
            high_g: 0.0,
            load_g: 0.0,
            burble_g: 0.0,
            tire_g: 0.0,
            master: 0.0,
            bike: 0.0,
            rpm: 0.0,
            cursor: [0.0; 6],
            shots: Vec::new(),
            throttle_open: false,
        }
    }

    fn aim(&mut self, drive: DriveState) {
        self.target = layer_gains(&drive);
    }

    fn note_throttle(&mut self, throttle: f32, rpm: f32) {
        let open = throttle.abs() > 0.5;
        if open && !self.throttle_open {
            self.push(Cue::Rise);
        }
        if !open && self.throttle_open && rpm > 0.22 {
            self.push(Cue::Crackle);
        }
        self.throttle_open = open;
    }

    fn push(&mut self, cue: Cue) {
        if self.shots.len() >= 8 {
            self.shots.remove(0);
        }
        self.shots.push(Shot::from_cue(cue));
    }

    fn render(&mut self, out: &mut [f32]) {
        if self.sr < 1_000.0 || out.is_empty() {
            out.fill(0.0);
            return;
        }
        let sr = self.sr;
        let follow = 1.0 - (-18.0 / sr).exp();
        let rate_scale = CLIP_RATE / sr;
        with_bank(|bank| {
            let loops = [
                bank.idle.as_slice(),
                bank.low.as_slice(),
                bank.high.as_slice(),
                bank.load.as_slice(),
                bank.burble.as_slice(),
                bank.tire.as_slice(),
            ];
            for sample in out.iter_mut() {
                self.idle_g += (self.target.idle - self.idle_g) * follow;
                self.low_g += (self.target.low - self.low_g) * follow;
                self.high_g += (self.target.high - self.high_g) * follow;
                self.load_g += (self.target.load - self.load_g) * follow;
                self.burble_g += (self.target.burble - self.burble_g) * follow;
                self.tire_g += (self.target.tire - self.tire_g) * follow;
                self.master += (self.target.master - self.master) * follow;
                let bike_target = if self.target.bike { 1.0 } else { 0.0 };
                self.bike += (bike_target - self.bike) * follow;
                self.rpm += (self.target.rpm - self.rpm) * follow;
                let bike_rate = 1.0 + self.bike * 0.055;
                let gains = [
                    self.idle_g,
                    self.low_g,
                    self.high_g,
                    self.load_g * 0.85,
                    self.burble_g * 0.55,
                    self.tire_g * 0.7,
                ];
                let centers = [0.05_f32, 0.40, 0.85, 0.45, 0.10, 0.5];
                let mut mixed = 0.0_f32;
                for index in 0..6 {
                    let bend = 1.0 + ((self.rpm - centers[index]) * 0.07).clamp(-0.05, 0.05);
                    let step = rate_scale * bike_rate * bend;
                    let voice = read_loop(loops[index], &mut self.cursor[index], step);
                    let level = if index == 5 {
                        gains[index] * (0.45 + 0.55 * self.master)
                    } else {
                        gains[index] * self.master
                    };
                    mixed += voice * level;
                }
                *sample = mixed;
            }
            for shot in &mut self.shots {
                let clip = match shot.kind {
                    ShotKind::Clump => bank.clump.as_slice(),
                    ShotKind::Splash => bank.splash.as_slice(),
                    ShotKind::Puddle => bank.puddle.as_slice(),
                    ShotKind::Rise => bank.rise.as_slice(),
                    ShotKind::Crackle => bank.crackle.as_slice(),
                };
                let step = rate_scale * shot.step;
                for sample in out.iter_mut() {
                    let Some(voice) = read_shot(clip, &mut shot.cursor, step) else {
                        break;
                    };
                    *sample += voice * shot.gain;
                }
            }
        });
        self.shots.retain(|shot| {
            let len = with_bank(|bank| match shot.kind {
                ShotKind::Clump => bank.clump.len(),
                ShotKind::Splash => bank.splash.len(),
                ShotKind::Puddle => bank.puddle.len(),
                ShotKind::Rise => bank.rise.len(),
                ShotKind::Crackle => bank.crackle.len(),
            });
            (shot.cursor as usize) + 1 < len
        });
        for sample in out {
            *sample = soft(*sample);
        }
    }
}

fn read_loop(clip: &[f32], cursor: &mut f32, step: f32) -> f32 {
    let n = clip.len();
    if n < 8 {
        return 0.0;
    }
    let nf = n as f32;
    let c = cursor.rem_euclid(nf);
    let fade = LOOP_FADE.min(nf * 0.2);
    let sample = if c >= nf - fade {
        let t = ((nf - c) / fade).clamp(0.0, 1.0);
        let head = sample_at(clip, c + fade - nf);
        let tail = sample_at(clip, c);
        tail * t.sqrt() + head * (1.0 - t).sqrt()
    } else {
        sample_at(clip, c)
    };
    *cursor = (c + step).rem_euclid(nf);
    sample
}

fn read_shot(clip: &[f32], cursor: &mut f32, step: f32) -> Option<f32> {
    if clip.len() < 2 || *cursor >= (clip.len() - 1) as f32 {
        return None;
    }
    let sample = sample_once(clip, *cursor);
    *cursor += step;
    Some(sample)
}

fn sample_at(clip: &[f32], pos: f32) -> f32 {
    let n = clip.len();
    if n == 0 {
        return 0.0;
    }
    let pos = pos.rem_euclid(n as f32);
    let i = pos as usize;
    let frac = pos - i as f32;
    let a = clip[i];
    let b = clip[(i + 1) % n];
    a + (b - a) * frac
}

fn sample_once(clip: &[f32], pos: f32) -> f32 {
    if clip.len() < 2 || pos >= (clip.len() - 1) as f32 {
        return 0.0;
    }
    let i = pos as usize;
    let frac = pos - i as f32;
    clip[i] + (clip[i + 1] - clip[i]) * frac
}

fn soft(x: f32) -> f32 {
    x.clamp(-2.5, 2.5).tanh()
}

struct Session {
    audition: Audition,
    mixer: Mixer,
}

impl Session {
    const fn new() -> Self {
        Self {
            audition: Audition::new(),
            mixer: Mixer::new(),
        }
    }
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static SESSION: RefCell<Session> = const { RefCell::new(Session::new()) };
}

#[cfg(not(target_arch = "wasm32"))]
static SESSION: LazyLock<Mutex<Session>> = LazyLock::new(|| Mutex::new(Session::new()));

fn with_session(body: impl FnOnce(&mut Session)) {
    #[cfg(target_arch = "wasm32")]
    SESSION.with(|slot| body(&mut slot.borrow_mut()));
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut guard = SESSION.lock().unwrap_or_else(|err| err.into_inner());
        body(&mut guard);
    }
}

#[cfg_attr(test, allow(dead_code))]
fn log_once(message: &str) {
    thread_local! {
        static DONE: Cell<bool> = const { Cell::new(false) };
    }
    let done = DONE.with(|flag| {
        if flag.get() {
            true
        } else {
            flag.set(true);
            false
        }
    });
    if done {
        return;
    }
    #[cfg(target_arch = "wasm32")]
    web_sys::console::log_1(&wasm_bindgen::JsValue::from_str(message));
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{message}");
}

#[cfg(target_arch = "wasm32")]
mod host {
    use super::{log_once, with_session};
    use std::cell::{Cell, RefCell};
    use wasm_bindgen::closure::Closure;
    use wasm_bindgen::JsCast;
    use web_sys::{AudioContext, AudioProcessingEvent};

    thread_local! {
        static LISTENING: Cell<bool> = const { Cell::new(false) };
        static STARTED: Cell<bool> = const { Cell::new(false) };
        static TRIED: Cell<bool> = const { Cell::new(false) };
        static CONTEXT: RefCell<Option<AudioContext>> = const { RefCell::new(None) };
        static NODES: RefCell<Option<web_sys::ScriptProcessorNode>> = const { RefCell::new(None) };
    }

    pub fn ensure_output() {
        install_listener();
        let tried = TRIED.with(|flag| {
            if flag.get() {
                true
            } else {
                flag.set(true);
                false
            }
        });
        if !tried {
            start_graph();
        }
    }

    fn install_listener() {
        let already = LISTENING.with(|flag| {
            if flag.get() {
                true
            } else {
                flag.set(true);
                false
            }
        });
        if already {
            return;
        }
        let Some(window) = web_sys::window() else {
            return;
        };
        let closure = Closure::wrap(Box::new(|| {
            start_graph();
        }) as Box<dyn FnMut()>);
        let function: &js_sys::Function = closure.as_ref().unchecked_ref();
        let target: &web_sys::EventTarget = window.as_ref();
        let _ = target.add_event_listener_with_callback("pointerdown", function);
        let _ = target.add_event_listener_with_callback("keydown", function);
        closure.forget();
    }

    fn start_graph() {
        if STARTED.with(|flag| flag.get()) {
            resume();
            return;
        }
        let ctx = match AudioContext::new() {
            Ok(ctx) => ctx,
            Err(err) => {
                web_sys::console::warn_1(&err);
                return;
            }
        };
        let rate = ctx.sample_rate();
        if !(8_000.0..=192_000.0).contains(&rate) {
            return;
        }
        let processor = match ctx
            .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
                2048, 1, 2,
            ) {
            Ok(node) => node,
            Err(err) => {
                web_sys::console::warn_1(&err);
                return;
            }
        };
        with_session(|session| session.mixer.sr = rate);
        let callback = Closure::wrap(Box::new(move |event: AudioProcessingEvent| {
            render_event(&event);
        }) as Box<dyn FnMut(AudioProcessingEvent)>);
        processor.set_onaudioprocess(Some(callback.as_ref().unchecked_ref()));
        callback.forget();
        let destination = ctx.destination();
        let destination: &web_sys::AudioNode = destination.as_ref();
        let node: &web_sys::AudioNode = processor.as_ref();
        if node.connect_with_audio_node(destination).is_err() {
            return;
        }
        CONTEXT.with(|slot| *slot.borrow_mut() = Some(ctx));
        NODES.with(|slot| *slot.borrow_mut() = Some(processor));
        STARTED.with(|flag| flag.set(true));
        log_once(&format!("silt audio {rate:.0} Hz"));
        resume();
    }

    fn resume() {
        CONTEXT.with(|slot| {
            if let Some(ctx) = slot.borrow().as_ref() {
                let _ = ctx.resume();
            }
        });
    }

    fn render_event(event: &AudioProcessingEvent) {
        let Ok(buffer) = event.output_buffer() else {
            return;
        };
        let frames = buffer.length() as usize;
        let channels = buffer.number_of_channels().max(1) as usize;
        thread_local! {
            static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
        }
        SCRATCH.with(|scratch| {
            let mut mono = scratch.borrow_mut();
            mono.resize(frames, 0.0);
            with_session(|session| session.mixer.render(&mut mono));
            for channel in 0..channels {
                let _ = buffer.copy_to_channel(&mono, channel as i32);
            }
        });
    }
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "linux"))]
mod host {
    #![cfg_attr(test, allow(dead_code))]

    use super::{log_once, with_session};
    use std::ffi::{CStr, CString};
    use std::os::raw::{c_char, c_int, c_long, c_ulong, c_void};
    use std::panic::{catch_unwind, AssertUnwindSafe};
    #[cfg(not(test))]
    use std::sync::Once;

    const RTLD_NOW: c_int = 2;
    const PLAYBACK: c_int = 0;
    const RW_INTERLEAVED: c_int = 3;
    const S16_LE: c_int = 2;

    type OpenFn = unsafe extern "C" fn(*mut *mut c_void, *const c_char, c_int, c_int) -> c_int;
    type SetParamsFn =
        unsafe extern "C" fn(*mut c_void, c_int, c_int, u32, u32, c_int, u32) -> c_int;
    type WriteFn = unsafe extern "C" fn(*mut c_void, *const c_void, c_ulong) -> c_long;
    type RecoverFn = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;
    type CloseFn = unsafe extern "C" fn(*mut c_void) -> c_int;
    type SetHandler = unsafe extern "C" fn(
        unsafe extern "C" fn(*const c_char, c_int, *const c_char, c_int, *const c_char, _: ...),
    ) -> c_int;

    unsafe extern "C" fn quiet_alsa(
        _file: *const c_char,
        _line: c_int,
        _function: *const c_char,
        _err: c_int,
        _fmt: *const c_char,
        _: ...
    ) {
    }

    struct Alsa {
        open: OpenFn,
        set_params: SetParamsFn,
        writei: WriteFn,
        recover: RecoverFn,
        close: CloseFn,
    }

    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    impl Alsa {
        fn load() -> Option<Self> {
            unsafe {
                let mut handle = std::ptr::null_mut();
                for name in [c"libasound.so.2".as_ptr(), c"libasound.so".as_ptr()] {
                    handle = dlopen(name, RTLD_NOW);
                    if !handle.is_null() {
                        break;
                    }
                }
                if handle.is_null() {
                    return None;
                }
                if let Some(set_handler) =
                    symbol::<SetHandler>(handle, c"snd_lib_error_set_handler")
                {
                    set_handler(quiet_alsa);
                }
                Some(Self {
                    open: symbol(handle, c"snd_pcm_open")?,
                    set_params: symbol(handle, c"snd_pcm_set_params")?,
                    writei: symbol(handle, c"snd_pcm_writei")?,
                    recover: symbol(handle, c"snd_pcm_recover")?,
                    close: symbol(handle, c"snd_pcm_close")?,
                })
            }
        }
    }

    unsafe fn symbol<T>(handle: *mut c_void, name: &CStr) -> Option<T> {
        let ptr = dlsym(handle, name.as_ptr());
        if ptr.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&ptr))
        }
    }

    pub fn ensure_output() {
        #[cfg(test)]
        {
            return;
        }
        #[cfg(not(test))]
        {
            static START: Once = Once::new();
            START.call_once(|| {
                let _ = std::thread::Builder::new()
                    .name("silt-audio".into())
                    .spawn(audio_thread);
            });
        }
    }

    fn audio_thread() {
        let Some(alsa) = Alsa::load() else {
            log_once("silt audio: libasound is not available");
            return;
        };
        let Some((pcm, rate)) = open_playback(&alsa) else {
            log_once("silt audio: no playback device");
            return;
        };
        with_session(|session| session.mixer.sr = rate);
        log_once(&format!("silt audio {rate:.0} Hz"));
        let mut mono = vec![0.0_f32; 512];
        let mut pcm_buf = vec![0_i16; mono.len()];
        let mut faults = 0_u32;
        loop {
            mono.fill(0.0);
            let rendered = catch_unwind(AssertUnwindSafe(|| {
                with_session(|session| session.mixer.render(&mut mono));
            }));
            if rendered.is_err() {
                mono.fill(0.0);
            }
            for (dst, src) in pcm_buf.iter_mut().zip(mono.iter()) {
                *dst = (src.clamp(-1.0, 1.0) * 32_767.0) as i16;
            }
            if !write_all(&alsa, pcm, &pcm_buf) {
                faults += 1;
                if faults > 8 {
                    unsafe { (alsa.close)(pcm) };
                    log_once("silt audio: playback stopped");
                    return;
                }
            } else {
                faults = 0;
            }
        }
    }

    fn open_playback(alsa: &Alsa) -> Option<(*mut c_void, f32)> {
        for name in ["default", "pulse", "sysdefault"] {
            let Ok(c_name) = CString::new(name) else {
                continue;
            };
            let mut pcm = std::ptr::null_mut();
            let opened = unsafe { (alsa.open)(&mut pcm, c_name.as_ptr(), PLAYBACK, 0) };
            if opened < 0 || pcm.is_null() {
                continue;
            }
            let params =
                unsafe { (alsa.set_params)(pcm, S16_LE, RW_INTERLEAVED, 1, 48_000, 1, 60_000) };
            if params == 0 {
                return Some((pcm, 48_000.0));
            }
            unsafe { (alsa.close)(pcm) };
        }
        None
    }

    fn write_all(alsa: &Alsa, pcm: *mut c_void, samples: &[i16]) -> bool {
        let mut wrote = 0_usize;
        let mut tries = 0_u32;
        while wrote < samples.len() && tries < 6 {
            tries += 1;
            let result = unsafe {
                (alsa.writei)(
                    pcm,
                    samples[wrote..].as_ptr().cast(),
                    (samples.len() - wrote) as c_ulong,
                )
            };
            if result > 0 {
                wrote += result as usize;
                continue;
            }
            if result == 0 {
                return false;
            }
            let recovered = unsafe { (alsa.recover)(pcm, result as c_int, 1) };
            if recovered < 0 {
                return false;
            }
        }
        wrote == samples.len()
    }
}

#[cfg(not(any(target_arch = "wasm32", target_os = "linux")))]
mod host {
    pub fn ensure_output() {}
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bit(kind: u8, seed: u32, size: f32, grounded: bool) -> Bit {
        Bit {
            seed,
            kind,
            alive: true,
            grounded,
            size,
        }
    }

    fn ear_with(parts: Vec<Bit>) -> Ear {
        Ear {
            drive_bike: false,
            speed: 0.0,
            rpm: 0.0,
            slip: 0.0,
            mud: 0.0,
            moving: 0.0,
            throttle: 0.0,
            water_wheels: 0,
            buggy_vy: 0.0,
            bike_vy: 0.0,
            buggy_muddy: false,
            bike_muddy: false,
            impact_sprays: 0,
            parts,
        }
    }

    fn drive(throttle: f32, slip: f32, rpm: f32, mud: f32, moving: f32) -> DriveState {
        DriveState {
            throttle,
            slip,
            rpm,
            mud,
            moving,
            bike: false,
        }
    }

    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        let energy = samples.iter().map(|sample| sample * sample).sum::<f32>();
        (energy / samples.len() as f32).sqrt()
    }

    fn low_share(samples: &[f32], sr: f32, cutoff: f32) -> f32 {
        let a = 1.0 - (-2.0 * std::f32::consts::PI * cutoff / sr).exp();
        let mut state = 0.0_f32;
        let mut low = 0.0_f32;
        let mut total = 0.0_f32;
        for sample in samples {
            state += (sample - state) * a;
            low += state * state;
            total += sample * sample;
        }
        low / (total + 1.0)
    }

    #[test]
    fn throttle_and_slip_crossfade_the_recorded_layers() {
        let parked = layer_gains(&DriveState::parked());
        assert!(parked.idle > 0.9, "idle {}", parked.idle);
        assert!(parked.high < 0.02);
        assert!(parked.load < 0.02);
        assert!(parked.tire < 0.02);
        assert!(parked.burble > 0.5);
        assert!(parked.master < 0.25);

        let opened = layer_gains(&drive(1.0, 0.0, 0.0, 0.0, 0.0));
        assert!(
            opened.low > opened.idle,
            "low {} idle {}",
            opened.low,
            opened.idle
        );
        assert!(opened.high < 0.05);

        let wide = layer_gains(&drive(1.0, 0.0, 1.0, 0.0, 1.0));
        assert!(wide.high > 0.9, "high {}", wide.high);
        assert!(wide.idle < 0.05);
        assert!(wide.load < 0.02);
        assert!(wide.master > parked.master + 0.3);

        let dry = layer_gains(&drive(0.5, 0.0, 0.5, 0.0, 1.0));
        let spinning = layer_gains(&drive(0.5, 1.0, 0.5, 0.0, 1.0));
        let mud = layer_gains(&drive(0.8, 1.0, 0.55, 1.0, 1.0));
        assert!(spinning.load > dry.load + 0.15);
        assert!(mud.load > mud.low && mud.load > mud.high);
        assert!(mud.tire > 0.9);
        assert!(dry.tire < 0.02);
        let dry_spin = dry.low + dry.high;
        let mud_spin = mud.low + mud.high;
        assert!(mud_spin < dry_spin, "mud spin {mud_spin} dry {dry_spin}");
    }

    #[test]
    fn recorded_clips_are_deep_pcm_and_long_enough() {
        with_bank(|bank| {
            assert!(bank.idle.len() > 22_050, "idle {}", bank.idle.len());
            assert!(bank.low.len() > 22_050);
            assert!(bank.high.len() > 10_000);
            assert!(bank.load.len() > 22_050);
            assert!(bank.burble.len() > 10_000);
            assert!(bank.tire.len() > 10_000);
            assert!(bank.rise.len() > 10_000);
            assert!(bank.splash.len() > 2_000);
            assert!(bank.clump.len() > 2_000);
            assert!(bank.puddle.len() > 1_000 && bank.puddle.len() < bank.splash.len());
            assert!(bank.crackle.len() > 1_000 && bank.crackle.len() < 22_050);
            for clip in [
                &bank.idle,
                &bank.low,
                &bank.high,
                &bank.load,
                &bank.burble,
                &bank.tire,
                &bank.rise,
                &bank.splash,
                &bank.clump,
                &bank.puddle,
                &bank.crackle,
            ] {
                assert!(clip.iter().all(|sample| sample.is_finite()));
                assert!(rms(clip) > 0.04, "rms {}", rms(clip));
            }
            let idle_low = low_share(&bank.idle, CLIP_RATE, 180.0);
            let high_low = low_share(&bank.high, CLIP_RATE, 180.0);
            assert!(
                idle_low > high_low + 0.2,
                "idle low-share {idle_low:.2} high {high_low:.2}"
            );
            assert!(low_share(&bank.burble, CLIP_RATE, 180.0) > 0.5);
            assert!(low_share(&bank.load, CLIP_RATE, 180.0) > 0.5);
        });
    }

    #[test]
    fn throttle_edges_play_the_rise_and_the_crackle() {
        let mut mixer = Mixer::new();
        mixer.note_throttle(0.0, 0.0);
        assert!(mixer.shots.is_empty());
        mixer.note_throttle(1.0, 0.1);
        assert_eq!(mixer.shots.len(), 1);
        assert!(matches!(mixer.shots[0].kind, ShotKind::Rise));
        mixer.note_throttle(1.0, 0.8);
        assert_eq!(mixer.shots.len(), 1);
        mixer.note_throttle(0.0, 0.1);
        assert_eq!(mixer.shots.len(), 1, "low rpm lifts off without a crackle");
        mixer.note_throttle(1.0, 0.4);
        mixer.note_throttle(0.0, 0.4);
        assert!(mixer
            .shots
            .iter()
            .any(|shot| matches!(shot.kind, ShotKind::Crackle)));
    }

    #[test]
    fn landing_and_chunk_splat_each_fire_once() {
        let mut audition = Audition::new();
        let mut ear = ear_with(Vec::new());
        assert!(audition.hear(&ear, 0.016).is_empty());

        ear.parts = (0..8)
            .map(|i| bit(KIND_CHUNK, 100 + i, 0.5, false))
            .collect();
        let cues = audition.hear(&ear, 0.016);
        assert_eq!(count_impacts(&cues), 1);
        assert_eq!(count_impacts(&audition.hear(&ear, 0.016)), 0);

        for _ in 0..40 {
            audition.hear(&ear, 0.016);
        }
        ear.parts[3] = bit(KIND_CHUNK, 103, 0.5, true);
        ear.parts.push(bit(0, 9, 0.08, true));
        let cues = audition.hear(&ear, 0.016);
        assert_eq!(count_splats(&cues), 1);
        assert_eq!(count_splats(&audition.hear(&ear, 0.016)), 0);
    }

    #[test]
    fn bike_burst_and_muddy_stop_are_impacts_dry_stops_are_not() {
        let mut audition = Audition::new();
        let mut ear = ear_with(Vec::new());
        ear.bike_vy = -4.5;
        ear.bike_muddy = false;
        audition.hear(&ear, 0.016);
        ear.bike_vy = -0.2;
        assert_eq!(count_impacts(&audition.hear(&ear, 0.016)), 0);

        let mut audition = Audition::new();
        ear.bike_vy = -4.2;
        ear.bike_muddy = true;
        audition.hear(&ear, 0.016);
        ear.bike_vy = -0.3;
        assert_eq!(count_impacts(&audition.hear(&ear, 0.016)), 1);

        let mut audition = Audition::new();
        ear.impact_sprays = 0;
        ear.bike_muddy = false;
        ear.bike_vy = 0.0;
        audition.hear(&ear, 0.016);
        ear.impact_sprays = 24;
        assert_eq!(count_impacts(&audition.hear(&ear, 0.016)), 1);
    }

    #[test]
    fn entering_water_picks_splash_or_puddle() {
        let mut audition = Audition::new();
        let mut ear = ear_with(Vec::new());
        ear.water_wheels = 1;
        assert!(audition.hear(&ear, 0.016).is_empty(), "prime");

        let mut audition = Audition::new();
        audition.hear(&ear, 0.016);
        ear.water_wheels = 2;
        ear.speed = 0.4;
        let cues = audition.hear(&ear, 0.016);
        assert_eq!(count_kind(&cues, false), 1);
        assert_eq!(count_kind(&audition.hear(&ear, 0.016), false), 0);

        let mut audition = Audition::new();
        ear.water_wheels = 0;
        ear.speed = 4.0;
        audition.hear(&ear, 0.016);
        ear.water_wheels = 1;
        let cues = audition.hear(&ear, 0.016);
        assert_eq!(count_kind(&cues, true), 1);
    }

    #[test]
    fn driven_vehicle_sets_the_engine_speed() {
        let mut world = World::flat(0.2);
        world.vehicle.vel.z = 7.5;
        world.bike.vel = glam::Vec3::ZERO;
        let buggy = Ear::from_world(&world, false, 0.0);
        let bike = Ear::from_world(&world, true, 1.0);
        assert!(buggy.speed > 7.0);
        assert!(bike.speed < 0.05);
        assert!(!buggy.drive_bike && bike.drive_bike);
        assert!((bike.throttle - 1.0).abs() < 1.0e-4);
        assert!(buggy.moving > 0.8);
        assert!(bike.moving < 0.05);
    }

    #[test]
    fn mixer_plays_idle_and_a_decaying_clump() {
        let mut quiet = Mixer::new();
        quiet.sr = CLIP_RATE;
        quiet.aim(DriveState::parked());
        let mut idle = vec![0.0; CLIP_RATE as usize];
        quiet.render(&mut idle);
        let tail = &idle[CLIP_RATE as usize / 2..];
        assert!(rms(tail) > 0.02, "parked idle rms {}", rms(tail));
        assert!(idle
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= 1.0));

        let mut hit = Mixer::new();
        hit.sr = CLIP_RATE;
        hit.aim(DriveState::parked());
        hit.push(Cue::Impact {
            intensity: 1.0,
            seed: 3,
        });
        let mut buf = vec![0.0; CLIP_RATE as usize];
        hit.render(&mut buf);
        assert!(buf
            .iter()
            .all(|sample| sample.is_finite() && sample.abs() <= 1.0));
        assert!(
            rms(&buf[..4_000]) > rms(&buf[18_000..]) * 1.4,
            "attack {} tail {}",
            rms(&buf[..4_000]),
            rms(&buf[18_000..])
        );
    }

    fn count_impacts(cues: &[Cue]) -> usize {
        cues.iter()
            .filter(|cue| matches!(cue, Cue::Impact { .. }))
            .count()
    }

    fn count_splats(cues: &[Cue]) -> usize {
        cues.iter()
            .filter(|cue| matches!(cue, Cue::Splat { .. }))
            .count()
    }

    fn count_kind(cues: &[Cue], splash: bool) -> usize {
        cues.iter()
            .filter(|cue| {
                if splash {
                    matches!(cue, Cue::Splash { .. })
                } else {
                    matches!(cue, Cue::Puddle { .. })
                }
            })
            .count()
    }
}
