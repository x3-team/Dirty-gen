//! One solver for the ground sheets and the dust air.
//!
//! Water and slurry are height–velocity sheets. Pressure is a two-pass
//! separable filter, a wide gaussian plus a steeper kernel. Viscosity is
//! the mix, and it enters by overloading the divergence the projection
//! removes. Dust is a 2.5D air grid on the same advection, projection, and
//! vorticity confinement. Fog is a second scalar on that grid, moved by the
//! same wind step and the same air velocity. It is not a second grid.
//! The Niva body and wheels inject velocity on that
//! grid, and the wake is confined there. The window moves with the Niva: a coarse sheet
//! about 20 cm across the near field, a 2 cm sheet under the tires, and
//! cells that leave the window are parked so the volume comes back.
//! Slow exchange runs every few frames. Spray is a volume handed to the
//! particle system, not a third grid.
//!
//! Terramechanics, support, and μ do not live here. Dust trail and the
//! solver LOD call this module. They do not keep a second solver.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;

mod near;
mod wet;
pub use near::NEAR_RESIDUAL_LIFE_S;
pub use wet::{WetBody, WetLoad, WetShape, SLURRY_RHO, WATER_RHO};

/// Wall time of one timed region. Absent on wasm so the solver does not read a clock there.
pub(crate) struct ProfTick {
    #[cfg(not(target_arch = "wasm32"))]
    t0: std::time::Instant,
}

impl ProfTick {
    pub(crate) fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            t0: std::time::Instant::now(),
        }
    }

    pub(crate) fn ms(&self) -> f64 {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.t0.elapsed().as_secs_f64() * 1000.0
        }
        #[cfg(target_arch = "wasm32")]
        {
            0.0
        }
    }
}

/// Exclusive milliseconds inside one `FluidCore` step and the recenter that precedes it.
#[derive(Clone, Copy, Default, Debug)]
pub struct SheetTimes {
    pub water_ms: f64,
    pub slurry_ms: f64,
    /// Grid viscosity, bed, and gate filled before the two sheets.
    pub prep_ms: f64,
    pub restrict_ms: f64,
    pub window_ms: f64,
    pub dust_ms: f64,
    /// Remainder of `FluidCore::step` after the rows above.
    pub rest_ms: f64,
    /// Window moves that could not scroll by a whole cell and used the hash path.
    pub window_hash: u64,
}

/// Coarse cell, metres. About 20 cm.
pub const COARSE_CELL_M: f32 = 0.20;
/// Cells on a side of the moving window. 200 × 0.20 m = 40 m, inside 30–50 m.
pub const WINDOW_RES: usize = 200;
/// Fine cell under the tires, metres.
pub const FINE_CELL_M: f32 = 0.02;
/// Fine patch on a side. 160 × 0.02 m = 3.2 m, the axle and the track.
pub const FINE_RES: usize = 160;
/// Coarse cells whose centres lie inside this radius of a tire, or of the
/// segment between the two tires on one side, are copied onto the 2 cm sheet.
/// The rest of the standing water stays on the 20 cm sheet.
const FINE_TIRE_R_M: f32 = 0.56;
/// Air plan. Columns, not a solid cube.
pub const AIR_RES: usize = 128;
/// Layers in each dust column. The bottom layer is the one that can settle.
pub const AIR_LAYERS: usize = 8;
/// Fraction of the free-stream headwind that remains on the centerline.
/// The crosswind is not sheltered. Past the lee the plume is in the free stream.
pub const DUST_STREAM_HOLD: f32 = 0.66;
/// Half-width of that lee, metres.
pub const DUST_LEE_HALF_M: f32 = 1.52;

/// 1 in the free stream, [`DUST_STREAM_HOLD`] on the centerline.
pub fn dust_stream_hold(local_x: f32) -> f32 {
    let t = (local_x.abs() / DUST_LEE_HALF_M).min(1.0);
    let lee = 1.0 - t * t;
    1.0 - (1.0 - DUST_STREAM_HOLD) * lee
}
/// Slow exchange (settle, park compact) runs on this stride, inside 4–8 frames.
pub const SLOW_STRIDE: u32 = 6;
/// Settled dust does not lift at or above this pore moisture.
pub const DUST_WET_MOISTURE: f32 = 0.55;
/// Settled dust does not lift at or above this standing water, metres.
pub const DUST_WET_WATER_M: f32 = 0.008;
/// Standing water, metres, at or above which a tire may loft fog.
pub const FOG_MIN_WATER_M: f32 = 0.004;
/// Fog droplets fall at this speed on the slow stride, metres per second.
pub const FOG_SETTLE_M_S: f32 = 0.55;

const GAUSS: [f32; 5] = [0.06, 0.24, 0.40, 0.24, 0.06];
const STEEP: [f32; 5] = [0.02, 0.08, 0.80, 0.08, 0.02];
const GRAVITY: f32 = 9.81;
/// Foam e-folding time. A few seconds of wake, then it is gone.
const FOAM_LIFE_S: f32 = 1.8;
/// Production shuts off as a cell approaches this fraction.
/// A sustained breaker therefore stays under 0.99 instead of filling to 1.
const FOAM_EQ: f32 = 0.85;
/// Time scale of that approach at full shear. One step does not paint the cell full.
const FOAM_TAU_S: f32 = 0.35;
/// Silt e-folding time in the water before it settles out of the column.
const SILT_SETTLE_S: f32 = 4.0;
/// Fall speed of the wheel-lifted solids (`susp`, contract turbidity), m/s.
/// Coarse silt: a 10 cm pool clears in about 25 s, a 2 cm film in 5 s.
const SUSP_FALL_M_S: f32 = 0.004;
/// Metres of stirred soil one cell will hold.
const SILT_COLUMN_CAP_M: f32 = 0.028;

/// What the solids in one cell are. Water with no solids uses [`Grain::Water`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grain {
    Clay = 0,
    Silt = 1,
    Sand = 2,
    Water = 3,
    Other = 4,
}

impl Grain {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Silt,
            2 => Self::Sand,
            3 => Self::Water,
            4 => Self::Other,
            _ => Self::Clay,
        }
    }
}

/// Dynamic viscosity, Pa·s. Clay stays thick. Silt sits under it. Sand does
/// not slurry like clay. A thin mix is the fluid end of the same curve.
pub fn dynamic_viscosity(grain: Grain, moisture: f32) -> f32 {
    let x = ((moisture.clamp(0.0, 1.0) - 0.22) / 0.70).clamp(0.0, 1.0);
    let thin = x * x * (3.0 - 2.0 * x);
    let (stiff, fluid) = match grain {
        Grain::Clay => (420.0, 0.35),
        Grain::Silt => (220.0, 4.0),
        Grain::Sand => (28.0, 0.55),
        Grain::Water => (0.9, 0.9),
        Grain::Other => (1.5, 0.08),
    };
    stiff + (fluid - stiff) * thin
}

/// How hard a free-surface slope accelerates the sheet. A very thin mix,
/// under about 12 Pa·s, moves more freely. Clay and silt do not.
///
/// A negative entry is not a viscosity. It is the mobility itself, for an
/// open-water reach that runs at full gravity (see
/// [`FluidCore::set_open_water`]).
pub fn sheet_mobility_for(nu: f32) -> f32 {
    if nu < 0.0 {
        return -nu;
    }
    let boost = if nu < 12.0 {
        let t = ((12.0 - nu.max(0.0)) / 12.0).clamp(0.0, 1.0);
        t * t * 1.6
    } else {
        0.0
    };
    0.042 * (1.0 + boost)
}

/// Per-second damping. Clay and silt stay in one band. Below that, a thin
/// mix sheds damping quickly and pools.
pub fn sheet_damp(nu: f32) -> f32 {
    let rel = (nu.max(0.05) / 231.0).clamp(0.002, 12.0);
    if rel >= 0.42 {
        return 1.70 * rel.powf(0.30);
    }
    let at_silt = 1.70 * 0.42_f32.powf(0.30);
    let t = (rel / 0.42).clamp(0.0, 1.0);
    at_silt * t.powf(1.15)
}

/// True when slip may loft dust. Wet pore moisture and a standing film may not.
pub fn dust_may_lift(moisture: f32, water_m: f32) -> bool {
    moisture < DUST_WET_MOISTURE && water_m < DUST_WET_WATER_M
}

/// True when standing water is deep enough to loft fog. Dry ground returns false.
pub fn fog_may_lift(water_m: f32) -> bool {
    water_m >= FOG_MIN_WATER_M
}

/// What drives a sheet for one step.
#[derive(Clone, Copy, Debug)]
pub enum SheetDrive {
    /// Down the free surface.
    Gravity { vorticity: f32 },
    /// Toward a horizontal wind. Used by the dust layers.
    Wind { x: f32, z: f32, vorticity: f32 },
}

/// Impermeable footprint of a wheel or a belly panel in the water grid.
///
/// The sheet treats every cell inside this box, and below the free surface,
/// as a wall. The head, the downstream hole, and the wake are whatever the
/// height–velocity step does against that wall. This is not a wave generator.
#[derive(Clone, Copy, Debug)]
pub struct HydroSolid {
    pub x: f32,
    pub z: f32,
    /// Yaw of the long axis, radians. 0 points the long axis along +Z.
    pub yaw: f32,
    pub half_length: f32,
    pub half_width: f32,
    /// World Y of the underside. A cell is a wall only when the free surface
    /// is above this.
    pub bottom_y: f32,
    pub top_y: f32,
    /// Tire contact. Scour reads the shear on the fluid cells touching these.
    pub contact: bool,
    /// Cylinder radius, metres. Zero is a box (belly), and buoyancy is the
    /// submerged box. A positive radius uses the submerged circular segment.
    pub radius: f32,
    /// Cylinder width, metres. Unused for a box.
    pub width: f32,
}

/// Pressure force and displaced weight read back from the sheet after a step.
#[derive(Clone, Copy, Debug, Default)]
pub struct HydroLoad {
    pub force_x: f32,
    pub force_z: f32,
    pub buoyancy_n: f32,
    pub hull_buoyancy_n: f32,
    pub wheel_buoyancy_n: f32,
    pub displaced_m3: f32,
    pub eroded_m3: f32,
    pub hole_m: f32,
    pub wheel_blocks: u32,
    pub belly_blocks: u32,
}

/// Periodic lips of an open channel. Water that piles at the downstream lip
/// is averaged with the upstream lip so a long reach can keep a current.
/// Off unless the caller sets one. Wheels and the belly do not use this.
#[derive(Clone, Copy, Debug)]
pub struct StreamCycle {
    pub x0: f32,
    pub x1: f32,
    pub z0: f32,
    pub z1: f32,
    pub dir_x: f32,
    pub dir_z: f32,
}

/// A paddle strip in the coarse water sheet. Inside the strip the water
/// velocity along `dir` is pulled toward `amplitude * sin(2π t / period)`.
/// Only velocity is driven, never depth, so the strip cannot mint or delete
/// water. The waves it sends out, their shoaling, and their run-up on a bank
/// are whatever the height–velocity step does with that push.
/// Off unless the caller sets one.
#[derive(Clone, Copy, Debug)]
pub struct WaveMaker {
    pub x0: f32,
    pub x1: f32,
    pub z0: f32,
    pub z1: f32,
    /// Push direction in the ground plane. Normalised on use.
    pub dir_x: f32,
    pub dir_z: f32,
    /// Peak paddle velocity, metres per second.
    pub amplitude_m_s: f32,
    pub period_s: f32,
    /// Seconds over which the paddle ramps from 0 to full amplitude.
    pub ramp_s: f32,
}

/// Open river reach through the window: an inlet at the upstream window
/// edge and an outlet at the downstream edge, as in a recirculating flume.
///
/// The outlet rows hold the free surface at `outlet_eta` and copy the
/// velocity of the first interior row, so the stream leaves without a wall
/// to pile against. Water the outlet takes off goes into the reach bank. The
/// inlet rows are filled from that bank toward `inlet_eta` and push the water
/// in at `inlet_speed_m_s`, slower on the shallow margins. The bank is part of
/// the ledger, so the reach neither mints nor deletes water: what arrives at
/// the inlet is what left at the outlet. Off unless the caller sets one.
#[derive(Clone, Copy, Debug)]
pub struct RiverReach {
    /// True for a river along Z, false for one along X.
    pub along_z: bool,
    /// True when the water runs toward +X (or +Z).
    pub positive: bool,
    /// Cross-stream band, metres of Z for a river along X (X for one along Z).
    pub cross0: f32,
    pub cross1: f32,
    /// Rows at each window end that belong to the inlet and the outlet.
    pub rows: usize,
    /// World Y of the free surface the inlet and the outlet hold.
    pub inlet_eta: f32,
    pub outlet_eta: f32,
    pub inlet_speed_m_s: f32,
    /// Suspended solids the inflow carries, volume of solids over volume of
    /// water (0 a clear river, 0.05–0.15 a brown flood river). The inlet
    /// rows are held at this concentration; the solids are booked as
    /// `entrained_m3` like the wheel's, settle at the same fall speed and
    /// leave through the outlet with the water.
    pub inlet_turbidity: f32,
}

/// Water as a river, not a film in a rut.
///
/// `mobility` is the share of gravity the free-surface slope applies to the
/// water sheet (1 is full gravity). `manning_n` is the Manning roughness of
/// the bed, s/m^(1/3), applied as bed friction on the coarse water sheet.
/// Sand and gravel river beds sit around 0.025 to 0.04. Uniform flow on a
/// slope S then runs at `h^(2/3) S^(1/2) / n`, and the thin water on a
/// boulder's flank or in a swash is slowed far more than the deep thalweg.
/// Typical depth-averaged eddy viscosity in a shallow gravel river is a few
/// hundredths of m²/s.
#[derive(Clone, Copy, Debug)]
pub struct OpenWater {
    pub mobility: f32,
    pub manning_n: f32,
    /// Depth-averaged horizontal eddy viscosity, m²/s. Shear between wet
    /// cells, and no slip against dry ground: a boulder or the bank holds the
    /// water beside it back, which is the wake and the slow margin.
    pub eddy_m2_s: f32,
}

/// Pore water that a passing film leaves in the ground.
///
/// A coarse cell under at least `wet_film_m` of water or slurry is saturated.
/// A cell the film has left drains toward the pad moisture with an e-folding
/// time of `drain_s` scaled by its grain: sand drains at `drain_s`, silt three
/// times slower, clay eight times slower. This is the wet band a swash leaves
/// on a beach and the dark sand behind a receding wave. Volume stays in the
/// water sheet; pore water is a state of the ground, not a second column.
/// Off unless the caller sets one.
#[derive(Clone, Copy, Debug)]
pub struct ShoreSoak {
    pub wet_film_m: f32,
    pub drain_s: f32,
}

/// One tire telling the sheet which way it is shoving.
#[derive(Clone, Copy, Debug)]
pub struct TirePush {
    pub x: f32,
    pub z: f32,
    pub forward_x: f32,
    pub forward_z: f32,
    pub radius: f32,
    pub width: f32,
    pub speed: f32,
    pub slip: f32,
}

/// Volume that left a sheet and must be spawned as particles by the caller.
#[derive(Clone, Copy, Debug)]
pub struct SprayEvent {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
    pub volume: f32,
    /// 0 spray drop, 1 ribbon. The mesh is the particle system's, not a sheet.
    pub kind: u8,
    /// 0 clear water .. 1 the drop is mostly fines (slurry share of the
    /// column it left). Colour only; the volume returns as water.
    pub murk: f32,
}

/// A water drop the sheet has already peeled off. It is not a picture particle.
/// The render contract reads these while they are still in the air.
#[derive(Clone, Copy, Debug)]
pub struct LiveSpray {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    /// Cubic metres peeled from the water sheet.
    pub volume: f32,
    /// 0 is a water drop.
    pub kind: u8,
    /// Seconds since the sheet let it go.
    pub age: f32,
    /// Copied from `SprayEvent::murk`.
    pub murk: f32,
}

const SPRAY_LIFE_S: f32 = 1.1;
/// Horizontal air drag on a spray drop, 1/s.
const SPRAY_AIR_DRAG: f32 = 0.10;
/// Share of a cell's water a fast tire throws per step, and the most in metres.
const SPRAY_PEEL_SHARE: f32 = 0.04;
const SPRAY_PEEL_MAX_M: f32 = 0.004;
/// A drop landing where the film is thinner than this soaks into the ground.
const SPRAY_SOAK_FILM_M: f32 = 0.001;
/// Water depth that saturates the pores of one cell's top layer, metres.
const SPRAY_SOAK_PORE_M: f32 = 0.02;
/// Share of a landing drop's horizontal momentum the column keeps.
const SPRAY_MOMENTUM_SHARE: f32 = 0.3;
/// A landing never pushes a column faster than this, m/s, unless it already was.
const SPRAY_LANDING_DRIFT_M_S: f32 = 0.5;

/// A request to move soil into the air column. The returned cubic metres are
/// what actually entered. The caller takes that out of the ground.
#[derive(Clone, Copy, Debug)]
pub struct DustInject {
    pub x: f32,
    pub z: f32,
    pub slip: f32,
    pub moisture: f32,
    pub water_m: f32,
    /// Cubic metres per second offered by a dry spinning tire.
    pub offer_m3_s: f32,
}

/// Water a tire offers the air as fog. The sheet path removes that volume
/// from the water column. The dust-ledger path returns it and the caller
/// removes it. A thin film returns nothing.
#[derive(Clone, Copy, Debug)]
pub struct FogInject {
    pub x: f32,
    pub z: f32,
    pub water_m: f32,
    /// Cubic metres per second offered by a tire in standing water.
    pub offer_m3_s: f32,
}

/// One visible sample of the air column or the settled film. Not a volume.
/// The cubic metres stay on the grid. Callers draw a mark and discard it.
#[derive(Clone, Copy, Debug)]
pub struct DustMark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub u: f32,
    pub v: f32,
    /// Sitting on the ground. A lofted layer is false.
    pub settled: bool,
}

/// Ground state at one point. Heights are metres of column, not a second mesh.
#[derive(Clone, Copy, Debug, Default)]
pub struct GroundSample {
    pub water_m: f32,
    pub slurry_m: f32,
    pub u: f32,
    pub v: f32,
    pub settled_dust_m: f32,
    pub moisture: f32,
    pub bed_m: f32,
}

/// One coarse-window cell. Heights are metres. `u` and `v` are metres per second.
#[derive(Clone, Copy, Debug)]
pub struct CoarseCell {
    pub x: f32,
    pub z: f32,
    pub bed_m: f32,
    pub water_m: f32,
    pub slurry_m: f32,
    pub moisture: f32,
    pub u: f32,
    pub v: f32,
    pub settled_dust_m: f32,
    /// Suspended-solids concentration in the water column, 0..1.
    /// Zero when the water column is under 0.01 mm.
    pub turbidity: f32,
}

/// Maps a beauty pass can read. Coarse window, after the fine patch is put back.
#[derive(Clone, Debug)]
pub struct SurfaceMaps {
    pub res: usize,
    pub cell: f32,
    pub origin: [f32; 2],
    pub height: Vec<f32>,
    pub vel_u: Vec<f32>,
    pub vel_v: Vec<f32>,
    pub moisture: Vec<f32>,
    pub dust: Vec<f32>,
}

/// Closed account inside this module. Spray is volume already given to particles.
#[derive(Clone, Copy, Debug, Default)]
pub struct FluidLedger {
    pub water_m3: f32,
    pub slurry_m3: f32,
    pub dust_air_m3: f32,
    pub dust_settled_m3: f32,
    pub frozen_water_m3: f32,
    pub frozen_slurry_m3: f32,
    pub frozen_dust_m3: f32,
    pub spray_m3: f32,
    /// Spray that landed on dry ground and soaked into it.
    pub spray_soaked_m3: f32,
    /// Airborne fog plus fog parked off the air window, cubic metres.
    /// Settled fog is already back in `water_m3`.
    pub fog_air_m3: f32,
    /// Suspended solids still in the water column, cubic metres.
    pub suspended_m3: f32,
    /// Suspended solids that left the column: a dry cell, or the part above the water.
    pub settled_suspended_m3: f32,
    /// Suspended solids parked with a coarse cell that slid off the window.
    pub frozen_suspended_m3: f32,
    /// Volume the wheels have lifted into suspension. The column, the settled
    /// part, and the parked part sum to this.
    pub entrained_m3: f32,
    /// Water between the outlet and the inlet of a [`RiverReach`].
    pub reach_m3: f32,
    pub total_m3: f32,
}

/// Suspended fraction on the coarse sheet after the step.
///
/// `bins` counts cells that hold solids: `[0, 0.1)`, …, `[0.9, 1)`, then
/// `ratio >= 1`. Clear water is `clear`, not bin 0. `above_water` is the
/// count the publish clamp would have painted as 1 because `susp > water`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TurbiditySpread {
    pub bins: [u32; 11],
    pub clear: u32,
    pub above_water: u32,
    pub saturated: u32,
    pub max_ratio: f32,
    /// Mean water depth of the `ratio >= 1` bin. Zero when that bin is empty.
    pub full_water: f32,
}

/// How a rounded rut in one mix levels.
#[derive(Clone, Copy, Debug)]
pub struct RutHeal {
    pub level_s: f32,
    pub relief0: f32,
    pub relief_8s: f32,
    pub still_deep_at_8s: bool,
}

struct Scratch {
    n: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    c: Vec<f32>,
    d: Vec<f32>,
    e: Vec<f32>,
    removed: Vec<f32>,
    added: Vec<f32>,
    /// One row of left-going and right-going open fluxes for `transport_open`.
    flux_l: Vec<f32>,
    flux_r: Vec<f32>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            n: 0,
            a: Vec::new(),
            b: Vec::new(),
            c: Vec::new(),
            d: Vec::new(),
            e: Vec::new(),
            removed: Vec::new(),
            added: Vec::new(),
            flux_l: Vec::new(),
            flux_r: Vec::new(),
        }
    }

    fn fit(&mut self, n: usize) {
        if self.n == n {
            return;
        }
        self.n = n;
        self.a.resize(n, 0.0);
        self.b.resize(n, 0.0);
        self.c.resize(n, 0.0);
        self.d.resize(n, 0.0);
        self.e.resize(n, 0.0);
        self.removed.resize(n, 0.0);
        self.added.resize(n, 0.0);
    }
}

thread_local! {
    /// Set while a water or slurry sheet runs beside the other one. The two
    /// sheets already fill the workers, so the row splits inside stay serial.
    static INNER_SERIAL: Cell<bool> = const { Cell::new(false) };
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
    /// Dust layers run together. Each one starts from a zero scratch so the
    /// pressure halo does not depend on which layer ran last.
    static ISOLATE_SCRATCH: Cell<bool> = const { Cell::new(false) };
}

/// Row and column splits are allowed on this thread.
fn inner_par() -> bool {
    !INNER_SERIAL.with(|flag| flag.get())
}

/// Runs `f` with the inner row splits off, then puts the flag back.
fn serial_inner<R>(f: impl FnOnce() -> R) -> R {
    let prev = INNER_SERIAL.with(|flag| flag.replace(true));
    let out = f();
    INNER_SERIAL.with(|flag| flag.set(prev));
    out
}

fn park_sheet_scratch() -> Scratch {
    SCRATCH.with(|sc| sc.replace(Scratch::new()))
}

fn unpark_sheet_scratch(snap: Scratch) {
    SCRATCH.with(|sc| {
        *sc.borrow_mut() = snap;
    });
}

struct DustScratchGuard;

impl Drop for DustScratchGuard {
    fn drop(&mut self) {
        ISOLATE_SCRATCH.with(|flag| flag.set(false));
    }
}

fn dust_scratch_guard() -> DustScratchGuard {
    ISOLATE_SCRATCH.with(|flag| flag.set(true));
    DustScratchGuard
}

/// Viscosity, gate, and bed for one sheet step. Sized to the coarse window
/// so the fine sheet and the coarse sheet share the allocation.
struct PairScratch {
    nu_w: Vec<f32>,
    nu_s: Vec<f32>,
    gate: Vec<u8>,
    bed: Vec<f32>,
}

impl PairScratch {
    fn ensure(&mut self) {
        let n = WINDOW_RES * WINDOW_RES;
        if self.nu_w.len() == n {
            return;
        }
        let nu = dynamic_viscosity(Grain::Water, 1.0);
        self.nu_w = vec![nu; n];
        self.nu_s = vec![0.0; n];
        self.gate = vec![1; n];
        self.bed = vec![0.0; n];
    }
}

thread_local! {
    static PAIR_SCRATCH: RefCell<PairScratch> = RefCell::new(PairScratch {
        nu_w: Vec::new(),
        nu_s: Vec::new(),
        gate: Vec::new(),
        bed: Vec::new(),
    });
}

/// Coarse accumulators for the fine hand-off. Reused so a step does not
/// allocate the 40 m window, and only the coarse cells the fine patch
/// touched are cleared.
struct ColumnAcc {
    w: Vec<f32>,
    s: Vec<f32>,
    uw: Vec<f32>,
    vw: Vec<f32>,
    us: Vec<f32>,
    vs: Vec<f32>,
    cnt: Vec<u32>,
    touch: Vec<usize>,
    hits: Vec<(usize, usize)>,
    ready: bool,
}

impl ColumnAcc {
    fn new() -> Self {
        Self {
            w: Vec::new(),
            s: Vec::new(),
            uw: Vec::new(),
            vw: Vec::new(),
            us: Vec::new(),
            vs: Vec::new(),
            cnt: Vec::new(),
            touch: Vec::new(),
            hits: Vec::new(),
            ready: false,
        }
    }

    fn begin(&mut self) {
        if !self.ready {
            let n = WINDOW_RES * WINDOW_RES;
            self.w = vec![0.0; n];
            self.s = vec![0.0; n];
            self.uw = vec![0.0; n];
            self.vw = vec![0.0; n];
            self.us = vec![0.0; n];
            self.vs = vec![0.0; n];
            self.cnt = vec![0; n];
            self.ready = true;
        }
        self.touch.clear();
        self.hits.clear();
    }

    fn mark(&mut self, idx: usize) {
        if self.cnt[idx] == 0 {
            self.touch.push(idx);
        }
        self.cnt[idx] += 1;
    }

    fn end_clear(&mut self) {
        for &idx in &self.touch {
            self.w[idx] = 0.0;
            self.s[idx] = 0.0;
            self.uw[idx] = 0.0;
            self.vw[idx] = 0.0;
            self.us[idx] = 0.0;
            self.vs[idx] = 0.0;
            self.cnt[idx] = 0;
        }
        self.touch.clear();
        self.hits.clear();
    }
}

thread_local! {
    static COLUMN_ACC: RefCell<ColumnAcc> = RefCell::new(ColumnAcc::new());
}

struct DustPlane {
    h: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    bed: Vec<f32>,
    nu: Vec<f32>,
    gate: Vec<u8>,
}

impl DustPlane {
    fn fit(&mut self, n: usize) {
        if self.h.len() == n {
            return;
        }
        self.h.resize(n, 0.0);
        self.u.resize(n, 0.0);
        self.v.resize(n, 0.0);
        self.bed.resize(n, 0.0);
        self.nu.resize(n, 0.4);
        self.gate.resize(n, 1);
    }
}

thread_local! {
    static DUST_PLANE: RefCell<DustPlane> = RefCell::new(DustPlane {
        h: Vec::new(),
        u: Vec::new(),
        v: Vec::new(),
        bed: Vec::new(),
        nu: Vec::new(),
        gate: Vec::new(),
    });
}

/// Step one height–velocity sheet. `bed` is the solid under the film, metres.
/// `gate` is 1 where a dry neighbour may receive fluid. Mass only moves in the
/// upwind flux, so the filter cannot mint or delete a column.
pub fn step_sheet(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
) {
    step_sheet_eps(
        h, u, v, bed, nu, gate, res, cell, dt, drive, 1.0e-6, None, false, 8.0, 0,
    );
}

/// [`step_sheet_eps`] on a caller-owned scratch instead of the thread-local one.
fn step_sheet_eps_in(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    solid: Option<&[u8]>,
    banks: bool,
    sub_cap: f32,
    pace: u8,
    sc: &mut Scratch,
    sc2: &mut Scratch,
) {
    if dt <= 0.0 || res == 0 || h.len() < res * res {
        return;
    }
    // Branch-free wet sum; it only gates the early return.
    let sum = wet_sum(h, eps);
    if sum <= eps {
        return;
    }
    sc.fit(res * res);
    sc2.fit(res * res);
    step_sheet_scratch(
        h, u, v, bed, nu, gate, res, cell, dt, drive, eps, solid, banks, sub_cap, pace, sc,
        Some(sc2),
    );
}

fn step_sheet_eps(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    solid: Option<&[u8]>,
    banks: bool,
    sub_cap: f32,
    pace: u8,
) {
    if dt <= 0.0 || res == 0 || h.len() < res * res {
        return;
    }
    // Branch-free wet sum; it only gates the early return.
    let sum = wet_sum(h, eps);
    if sum <= eps {
        return;
    }
    SCRATCH.with(|sc| {
        let mut sc = sc.borrow_mut();
        sc.fit(res * res);
        step_sheet_scratch(
            h, u, v, bed, nu, gate, res, cell, dt, drive, eps, solid, banks, sub_cap, pace,
            &mut sc, None,
        );
    });
}

/// Advect one vehicle-frame dust volume, `x + layer * 128 + z * 128 * 8`.
///
/// Donor-cell upwind on cells that hold volume. Empty layers are not visited.
/// Mass that stays in `vol` is still airborne. The return is the cubic metres
/// that left the window, which the caller books as settled.
pub fn advect_dust_density(
    vol: &mut [f32],
    wind_x: f32,
    wind_z: f32,
    speed: f32,
    forward: glam::Vec3,
    right: glam::Vec3,
    dt: f32,
) -> f32 {
    const RES: usize = 128;
    const LAYERS: usize = 8;
    if dt <= 0.0 || vol.len() < RES * LAYERS * RES {
        return 0.0;
    }
    let forward = if forward.length_squared() > 1.0e-8 {
        forward.normalize()
    } else {
        glam::Vec3::Z
    };
    let right = if right.length_squared() > 1.0e-8 {
        right.normalize()
    } else {
        glam::Vec3::X
    };
    let rel = glam::Vec3::new(wind_x, 0.0, wind_z) - forward * speed.max(0.0);
    let gu = rel.dot(right);
    let gv = rel.dot(forward);
    let cell = 16.0 / RES as f32;
    advect_dust_volume(vol, RES, LAYERS, cell, gu, gv, dt);
    // The caller books a positive return as settled without a ground cell.
    // Mass stays in `vol` and their slow stride settles it onto the map.
    0.0
}

struct Down2 {
    h: Vec<f32>,
    u: Vec<f32>,
    v: Vec<f32>,
    bed: Vec<f32>,
    nu: Vec<f32>,
    gate: Vec<u8>,
    w: Vec<f32>,
}

impl Down2 {
    fn fit(&mut self, n: usize) {
        if self.h.len() == n {
            return;
        }
        self.h.resize(n, 0.0);
        self.u.resize(n, 0.0);
        self.v.resize(n, 0.0);
        self.bed.resize(n, 0.0);
        self.nu.resize(n, 0.4);
        self.gate.resize(n, 1);
        self.w.resize(n * 4, 0.0);
    }
}

thread_local! {
    static DOWN2: RefCell<Down2> = RefCell::new(Down2 {
        h: Vec::new(),
        u: Vec::new(),
        v: Vec::new(),
        bed: Vec::new(),
        nu: Vec::new(),
        gate: Vec::new(),
        w: Vec::new(),
    });
}

/// One sheet step on a grid twice as coarse, then scatter back.
/// `as_volume` sums the 2×2 block (dust cubic metres). A depth sheet averages it.
fn step_down2(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    as_volume: bool,
) {
    if res < 4 || res % 2 != 0 || h.len() < res * res {
        step_sheet_eps(
            h, u, v, bed, nu, gate, res, cell, dt, drive, eps, None, false, 8.0, 0,
        );
        return;
    }
    let c = res / 2;
    let cn = c * c;
    DOWN2.with(|plane| {
        let mut plane = plane.borrow_mut();
        plane.fit(cn);
        let Down2 {
            h: ch,
            u: cu,
            v: cv,
            bed: cbed,
            nu: cnu,
            gate: cgate,
            w,
        } = &mut *plane;
        let (ch, cu, cv) = (&mut ch[..cn], &mut cu[..cn], &mut cv[..cn]);
        let (cbed, cnu, cgate) = (&mut cbed[..cn], &mut cnu[..cn], &mut cgate[..cn]);
        let w = &mut w[..cn * 4];
        let (fh, fu, fv) = (&h[..res * res], &u[..res * res], &v[..res * res]);
        // Same 2x2 order as before: row dj, then column di.
        for j in 0..c {
            let r0 = (j * 2) * res;
            let r1 = r0 + res;
            for i in 0..c {
                let id = j * c + i;
                let f = [r0 + i * 2, r0 + i * 2 + 1, r1 + i * 2, r1 + i * 2 + 1];
                let mut sum = 0.0f32;
                let mut su = 0.0f32;
                let mut sv = 0.0f32;
                for k in 0..4 {
                    let hv = fh[f[k]];
                    w[id * 4 + k] = hv;
                    sum += hv;
                    let m = hv.max(0.0);
                    su += fu[f[k]] * m;
                    sv += fv[f[k]] * m;
                }
                ch[id] = if as_volume { sum } else { sum * 0.25 };
                if sum > 1.0e-12 {
                    cu[id] = su / sum;
                    cv[id] = sv / sum;
                } else {
                    cu[id] = 0.0;
                    cv[id] = 0.0;
                }
                let fidx = f[0];
                cbed[id] = bed.get(fidx).copied().unwrap_or(0.0);
                cnu[id] = nu.get(fidx).copied().unwrap_or(0.4);
                cgate[id] = gate.get(fidx).copied().unwrap_or(1);
            }
        }
        step_sheet_eps(ch, cu, cv, cbed, cnu, cgate, c, cell * 2.0, dt, drive, eps, None, false, 8.0, 0);
        let (fh, fu, fv) = (&mut h[..res * res], &mut u[..res * res], &mut v[..res * res]);
        for j in 0..c {
            let r0 = (j * 2) * res;
            let r1 = r0 + res;
            for i in 0..c {
                let id = j * c + i;
                let new_h = ch[id];
                let ws = [w[id * 4], w[id * 4 + 1], w[id * 4 + 2], w[id * 4 + 3]];
                let mut s = 0.0f32;
                for wk in ws {
                    s += wk.max(0.0);
                }
                let f = [r0 + i * 2, r0 + i * 2 + 1, r1 + i * 2, r1 + i * 2 + 1];
                for k in 0..4 {
                    let share = if s > 1.0e-12 {
                        if as_volume {
                            ws[k].max(0.0) / s * new_h
                        } else {
                            ws[k].max(0.0) / s * new_h * 4.0
                        }
                    } else if as_volume {
                        new_h * 0.25
                    } else {
                        new_h
                    };
                    fh[f[k]] = share.max(0.0);
                    fu[f[k]] = cu[id];
                    fv[f[k]] = cv[id];
                }
            }
        }
    });
}

/// [`step_down2`] for a dust volume whose fine velocity is `gu` everywhere
/// and `col_v[i]` down each column, with `eps` 0. The 2x2 sums use those
/// values in the same order, and only the volume is scattered back: the
/// caller rebuilds the fine wind on the next call.
fn step_down2_wind(
    h: &mut [f32],
    gu: f32,
    col_v: &[f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
) {
    let c = res / 2;
    let cn = c * c;
    DOWN2.with(|plane| {
        let mut plane = plane.borrow_mut();
        plane.fit(cn);
        let Down2 {
            h: ch,
            u: cu,
            v: cv,
            bed: cbed,
            nu: cnu,
            gate: cgate,
            w,
        } = &mut *plane;
        let (ch, cu, cv) = (&mut ch[..cn], &mut cu[..cn], &mut cv[..cn]);
        let (cbed, cnu, cgate) = (&mut cbed[..cn], &mut cnu[..cn], &mut cgate[..cn]);
        let w = &mut w[..cn * 4];
        let fh = &h[..res * res];
        for j in 0..c {
            let r0 = (j * 2) * res;
            let r1 = r0 + res;
            for i in 0..c {
                let id = j * c + i;
                let f = [r0 + i * 2, r0 + i * 2 + 1, r1 + i * 2, r1 + i * 2 + 1];
                let fv = [col_v[i * 2], col_v[i * 2 + 1], col_v[i * 2], col_v[i * 2 + 1]];
                let mut sum = 0.0f32;
                let mut su = 0.0f32;
                let mut sv = 0.0f32;
                for k in 0..4 {
                    let hv = fh[f[k]];
                    w[id * 4 + k] = hv;
                    sum += hv;
                    let m = hv.max(0.0);
                    su += gu * m;
                    sv += fv[k] * m;
                }
                ch[id] = sum;
                if sum > 1.0e-12 {
                    cu[id] = su / sum;
                    cv[id] = sv / sum;
                } else {
                    cu[id] = 0.0;
                    cv[id] = 0.0;
                }
                let fidx = f[0];
                cbed[id] = bed.get(fidx).copied().unwrap_or(0.0);
                cnu[id] = nu.get(fidx).copied().unwrap_or(0.4);
                cgate[id] = gate.get(fidx).copied().unwrap_or(1);
            }
        }
        step_sheet_eps(ch, cu, cv, cbed, cnu, cgate, c, cell * 2.0, dt, drive, 0.0, None, false, 8.0, 0);
        let fh = &mut h[..res * res];
        for j in 0..c {
            let r0 = (j * 2) * res;
            let r1 = r0 + res;
            for i in 0..c {
                let id = j * c + i;
                let new_h = ch[id];
                let ws = [w[id * 4], w[id * 4 + 1], w[id * 4 + 2], w[id * 4 + 3]];
                let mut s = 0.0f32;
                for wk in ws {
                    s += wk.max(0.0);
                }
                let f = [r0 + i * 2, r0 + i * 2 + 1, r1 + i * 2, r1 + i * 2 + 1];
                for k in 0..4 {
                    let share = if s > 1.0e-12 {
                        ws[k].max(0.0) / s * new_h
                    } else {
                        new_h * 0.25
                    };
                    fh[f[k]] = share.max(0.0);
                }
            }
        }
    });
}

/// Seven live dust layers share the workers. The inner 64² sheet stays on the
/// worker that owns the layer. The ground-sheet scratch is put back afterwards
/// so water still sees the buffer it left.
fn advect_dust_volume(
    vol: &mut [f32],
    res: usize,
    layers: usize,
    cell: f32,
    gu: f32,
    gv: f32,
    dt: f32,
) {
    let parked = park_sheet_scratch();
    #[cfg(not(target_arch = "wasm32"))]
    if rayon::current_num_threads() > 1 {
        advect_dust_layers_par(vol, res, layers, cell, gu, gv, dt);
        unpark_sheet_scratch(parked);
        return;
    }
    for layer in 0..layers {
        advect_one_dust_layer(vol, layer, res, layers, cell, gu, gv, dt);
    }
    unpark_sheet_scratch(parked);
}

#[cfg(not(target_arch = "wasm32"))]
fn advect_dust_layers_par(
    vol: &mut [f32],
    res: usize,
    layers: usize,
    cell: f32,
    gu: f32,
    gv: f32,
    dt: f32,
) {
    let n = vol.len();
    let ptr = ShareMut(vol.as_mut_ptr());
    rayon::scope(|s| {
        for layer in 0..layers {
            s.spawn(move |_| {
                // Layer `L` writes `ix + L * res + iz * res * layers` only.
                // Those sets are disjoint, and the scope joins before `vol` ends.
                let vol = unsafe { ptr.slice(n) };
                advect_one_dust_layer(vol, layer, res, layers, cell, gu, gv, dt);
            });
        }
    });
}

fn advect_one_dust_layer(
    vol: &mut [f32],
    layer: usize,
    res: usize,
    layers: usize,
    cell: f32,
    gu: f32,
    gv: f32,
    dt: f32,
) {
    let _isolate = dust_scratch_guard();
    let n = res * res;
    DUST_PLANE.with(|plane| {
        let mut plane = plane.borrow_mut();
        plane.fit(n);
        let DustPlane {
            h,
            u,
            v,
            bed,
            nu,
            gate,
        } = &mut *plane;
        let (h, u, v) = (&mut h[..n], &mut u[..n], &mut v[..n]);
        // The stream hold depends on the column only.
        let mut hold = [0.0f32; 128];
        let hold = if res <= hold.len() {
            for (ix, slot) in hold[..res].iter_mut().enumerate() {
                let local_x = -8.0 + (ix as f32 + 0.5) * cell;
                *slot = gv * dust_stream_hold(local_x);
            }
            &hold[..res]
        } else {
            &[][..]
        };
        let stride = res * layers;
        let drive = SheetDrive::Wind {
            x: gu,
            z: gv,
            vorticity: 0.35,
        };
        // The wind is one value per column, so the halved plane reads it
        // straight from `hold` and the full-size u/v copies are skipped.
        let direct = !hold.is_empty() && res >= 4 && res % 2 == 0;
        let mut before = 0.0f64;
        for iz in 0..res {
            let src = &vol[layer * res + iz * stride..][..res];
            let row = iz * res;
            let hr = &mut h[row..row + res];
            for ix in 0..res {
                let cell_v = src[ix];
                hr[ix] = cell_v;
                before += cell_v as f64;
            }
            if !direct {
                for ix in 0..res {
                    u[row + ix] = gu;
                    v[row + ix] = if hold.is_empty() {
                        gv * dust_stream_hold(-8.0 + (ix as f32 + 0.5) * cell)
                    } else {
                        hold[ix]
                    };
                }
            }
        }
        if before <= 0.0 {
            return;
        }
        if direct {
            step_down2_wind(h, gu, hold, bed, nu, gate, res, cell, dt, drive);
        } else {
            step_down2(h, u, v, bed, nu, gate, res, cell, dt, drive, 0.0, true);
        }
        let after = h.iter().map(|c| *c as f64).sum::<f64>();
        let scale = if after > 0.0 {
            (before / after) as f32
        } else {
            1.0
        };
        for iz in 0..res {
            let dst = &mut vol[layer * res + iz * stride..][..res];
            let hr = &h[iz * res..iz * res + res];
            for ix in 0..res {
                dst[ix] = hr[ix] * scale;
            }
        }
    });
}

/// Sum of the cells above `eps`, in eight lanes so it vectorises.
fn wet_sum(h: &[f32], eps: f32) -> f32 {
    let mut lanes = [0.0f32; 8];
    let mut chunks = h.chunks_exact(8);
    for c in &mut chunks {
        for k in 0..8 {
            lanes[k] += if c[k] > eps { c[k] } else { 0.0 };
        }
    }
    let mut sum: f32 = lanes.iter().sum();
    for &x in chunks.remainder() {
        sum += if x > eps { x } else { 0.0 };
    }
    sum
}

/// Work boxes are `(i0, i1, j0, j1)` inclusive. A solid wet rectangle stays
/// one box with the historical halo of 3, so that path is unchanged. Separated
/// blobs (the tire disks) each carry their own halo and are not stepped across
/// the dry gap.
fn sheet_work_boxes(h: &[f32], res: usize, eps: f32, allow_split: bool) -> Vec<(usize, usize, usize, usize)> {
    let mut i0 = res;
    let mut i1 = 0usize;
    let mut j0 = res;
    let mut j1 = 0usize;
    let mut wet = 0usize;
    for j in 0..res {
        let row = &h[j * res..j * res + res];
        // Count without a branch per cell; only a wet row pays for its ends.
        let c = row.iter().map(|&x| (x > eps) as usize).sum::<usize>();
        if c == 0 {
            continue;
        }
        wet += c;
        let first = row.iter().position(|&x| x > eps).unwrap_or(0);
        let last = row.iter().rposition(|&x| x > eps).unwrap_or(0);
        i0 = i0.min(first);
        i1 = i1.max(last);
        j0 = j0.min(j);
        j1 = j1.max(j);
    }
    if wet == 0 {
        return Vec::new();
    }
    let halo3 = |a0: usize, a1: usize| (a0.saturating_sub(3), (a1 + 3).min(res - 1));
    let (x0, x1) = halo3(i0, i1);
    let (y0, y1) = halo3(j0, j1);
    let solid = wet == (i1 - i0 + 1) * (j1 - j0 + 1);
    if !allow_split || solid {
        return vec![(x0, x1, y0, y1)];
    }
    // Wet bodies by 4-connected runs. Each row's wet runs join the runs
    // above that share a column. A body's box is the union of its runs and
    // bodies come out in the raster order of their first cell, the order a
    // raster flood fill finds them.
    RUNS.with(|runs| {
        let mut runs = runs.borrow_mut();
        let RunScratch { runs: all, parent, bbox, first } = &mut *runs;
        all.clear();
        parent.clear();
        bbox.clear();
        first.clear();
        fn root(parent: &mut [u32], mut x: u32) -> u32 {
            while parent[x as usize] != x {
                let up = parent[parent[x as usize] as usize];
                parent[x as usize] = up;
                x = up;
            }
            x
        }
        let mut prev = 0..0usize;
        for j in j0..=j1 {
            let row = &h[j * res..j * res + res];
            let start = all.len();
            let mut i = i0;
            while i <= i1 {
                if !(row[i] > eps) {
                    i += 1;
                    continue;
                }
                let a = i;
                while i <= i1 && row[i] > eps {
                    i += 1;
                }
                let b = i - 1;
                let id = all.len() as u32;
                all.push((a, b));
                parent.push(id);
                bbox.push((a, b, j, j));
                first.push(j * res + a);
                for k in prev.clone() {
                    let (pa, pb) = all[k];
                    if pa <= b && a <= pb {
                        let ra = root(parent, k as u32);
                        let rb = root(parent, id);
                        if ra != rb {
                            // The older run keeps the root, so its first cell
                            // (the smaller raster index) stays the body's.
                            let (keep, gone) = if ra < rb { (ra, rb) } else { (rb, ra) };
                            parent[gone as usize] = keep;
                            let g = bbox[gone as usize];
                            let kb = &mut bbox[keep as usize];
                            kb.0 = kb.0.min(g.0);
                            kb.1 = kb.1.max(g.1);
                            kb.2 = kb.2.min(g.2);
                            kb.3 = kb.3.max(g.3);
                            first[keep as usize] = first[keep as usize].min(first[gone as usize]);
                        }
                    }
                }
            }
            prev = start..all.len();
        }
        let mut bodies: Vec<(usize, (usize, usize, usize, usize))> = Vec::new();
        for k in 0..all.len() {
            if parent[k] == k as u32 {
                bodies.push((first[k], bbox[k]));
            }
        }
        bodies.sort_unstable_by_key(|b| b.0);
        // Two blur passes of radius 2, plus the gradient, sit inside this halo.
        const SPLIT_HALO: usize = 6;
        let mut boxes: Vec<(usize, usize, usize, usize)> = bodies
            .iter()
            .map(|&(_, (ci0, ci1, cj0, cj1))| {
                (
                    ci0.saturating_sub(SPLIT_HALO),
                    (ci1 + SPLIT_HALO).min(res - 1),
                    cj0.saturating_sub(SPLIT_HALO),
                    (cj1 + SPLIT_HALO).min(res - 1),
                )
            })
            .collect();
        merge_boxes(&mut boxes);
        // One body keeps the halo of 3, including a puddle with holes. Only
        // separated disks pay for their own boxes.
        if boxes.len() == 1 || boxes.len() > 8 {
            vec![(x0, x1, y0, y1)]
        } else {
            boxes
        }
    })
}

fn boxes_touch(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> bool {
    let (ai0, ai1, aj0, aj1) = a;
    let (bi0, bi1, bj0, bj1) = b;
    ai0.saturating_sub(1) <= bi1.saturating_add(1)
        && bi0.saturating_sub(1) <= ai1.saturating_add(1)
        && aj0.saturating_sub(1) <= bj1.saturating_add(1)
        && bj0.saturating_sub(1) <= aj1.saturating_add(1)
}

fn merge_boxes(boxes: &mut Vec<(usize, usize, usize, usize)>) {
    loop {
        let mut did = false;
        let mut i = 0;
        while i < boxes.len() {
            let mut j = i + 1;
            while j < boxes.len() {
                if boxes_touch(boxes[i], boxes[j]) {
                    let a = boxes[i];
                    let b = boxes[j];
                    boxes[i] = (
                        a.0.min(b.0),
                        a.1.max(b.1),
                        a.2.min(b.2),
                        a.3.max(b.3),
                    );
                    boxes.swap_remove(j);
                    did = true;
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
        if !did {
            break;
        }
    }
}

#[derive(Default)]
struct RunScratch {
    runs: Vec<(usize, usize)>,
    parent: Vec<u32>,
    bbox: Vec<(usize, usize, usize, usize)>,
    first: Vec<usize>,
}

thread_local! {
    static RUNS: RefCell<RunScratch> = RefCell::new(RunScratch::default());
}

/// Inclusive 2D prefix. `dst` is `(n + 1)²` and starts at zero, so the
/// border stays zero. `plane_window` then sums any inclusive rectangle.
fn prefix_plane(src: &[f32], n: usize, dst: &mut [f32]) {
    let stride = n + 1;
    debug_assert!(src.len() >= n * n);
    debug_assert!(dst.len() >= stride * stride);
    for j in 0..n {
        let src_row = j * n;
        let above = j * stride;
        let row = above + stride;
        let mut run = 0.0f32;
        for i in 0..n {
            run += src[src_row + i];
            dst[row + i + 1] = run + dst[above + i + 1];
        }
    }
}

fn plane_window(p: &[f32], n: usize, i0: usize, i1: usize, j0: usize, j1: usize) -> f32 {
    let stride = n + 1;
    let a = p[j0 * stride + i0];
    let b = p[j0 * stride + (i1 + 1)];
    let c = p[(j1 + 1) * stride + i0];
    let d = p[(j1 + 1) * stride + (i1 + 1)];
    d - b - c + a
}

/// Tile edge for a sheet that is standing still. Eight fine cells is 16 cm.
const SLEEP_TILE: usize = 8;
/// Below this speed a flat tile does not need another solve. The next awake
/// neighbour is stepped, so a bore still walks into the tile.
const SLEEP_SPEED: f32 = 0.05;
/// Free-surface grade that still has to run. Flatter water stays put.
const SLEEP_SLOPE: f32 = 0.02;

/// Boxes around tiles that are moving or sloped, plus one tile of halo.
/// A flat, slow tile keeps its depth and its velocity (the warm start).
fn awake_work_boxes(
    h: &[f32],
    u: &[f32],
    v: &[f32],
    bed: &[f32],
    res: usize,
    cell: f32,
    eps: f32,
) -> Vec<(usize, usize, usize, usize)> {
    let tiles = res.div_ceil(SLEEP_TILE);
    let ntile = tiles * tiles;
    let mut hot = vec![0u8; ntile];
    let mut grown = vec![0u8; ntile];
    for tj in 0..tiles {
        for ti in 0..tiles {
            let i0 = ti * SLEEP_TILE;
            let j0 = tj * SLEEP_TILE;
            let i1 = (i0 + SLEEP_TILE).min(res);
            let j1 = (j0 + SLEEP_TILE).min(res);
            let mut awake = false;
            'scan: for j in j0..j1 {
                let row = j * res;
                for i in i0..i1 {
                    let idx = row + i;
                    if h[idx] <= eps {
                        continue;
                    }
                    if u[idx].hypot(v[idx]) > SLEEP_SPEED {
                        awake = true;
                        break 'scan;
                    }
                    let eta = bed[idx] + h[idx];
                    let ip = if i + 1 < res { row + i + 1 } else { idx };
                    let jp = if j + 1 < res { (j + 1) * res + i } else { idx };
                    let dex = (bed[ip] + h[ip] - eta).abs();
                    let dez = (bed[jp] + h[jp] - eta).abs();
                    if dex.max(dez) > SLEEP_SLOPE * cell {
                        awake = true;
                        break 'scan;
                    }
                }
            }
            if awake {
                hot[tj * tiles + ti] = 1;
            }
        }
    }
    for tj in 0..tiles {
        for ti in 0..tiles {
            if hot[tj * tiles + ti] == 0 {
                continue;
            }
            for dj in -1i32..=1 {
                for di in -1i32..=1 {
                    let ni = ti as i32 + di;
                    let nj = tj as i32 + dj;
                    if ni >= 0 && nj >= 0 && (ni as usize) < tiles && (nj as usize) < tiles {
                        grown[nj as usize * tiles + ni as usize] = 1;
                    }
                }
            }
        }
    }
    let mut seen = vec![0u8; ntile];
    let mut stack = Vec::new();
    let mut boxes = Vec::new();
    for s in 0..ntile {
        if grown[s] == 0 || seen[s] != 0 {
            continue;
        }
        seen[s] = 1;
        stack.clear();
        stack.push(s);
        let mut ti0 = s % tiles;
        let mut ti1 = ti0;
        let mut tj0 = s / tiles;
        let mut tj1 = tj0;
        while let Some(cur) = stack.pop() {
            let tj = cur / tiles;
            let ti = cur - tj * tiles;
            ti0 = ti0.min(ti);
            ti1 = ti1.max(ti);
            tj0 = tj0.min(tj);
            tj1 = tj1.max(tj);
            if ti > 0 {
                let nidx = tj * tiles + ti - 1;
                if grown[nidx] != 0 && seen[nidx] == 0 {
                    seen[nidx] = 1;
                    stack.push(nidx);
                }
            }
            if ti + 1 < tiles {
                let nidx = tj * tiles + ti + 1;
                if grown[nidx] != 0 && seen[nidx] == 0 {
                    seen[nidx] = 1;
                    stack.push(nidx);
                }
            }
            if tj > 0 {
                let nidx = (tj - 1) * tiles + ti;
                if grown[nidx] != 0 && seen[nidx] == 0 {
                    seen[nidx] = 1;
                    stack.push(nidx);
                }
            }
            if tj + 1 < tiles {
                let nidx = (tj + 1) * tiles + ti;
                if grown[nidx] != 0 && seen[nidx] == 0 {
                    seen[nidx] = 1;
                    stack.push(nidx);
                }
            }
        }
        let i0 = ti0 * SLEEP_TILE;
        let j0 = tj0 * SLEEP_TILE;
        let i1 = ((ti1 + 1) * SLEEP_TILE).min(res) - 1;
        let j1 = ((tj1 + 1) * SLEEP_TILE).min(res) - 1;
        boxes.push((i0, i1, j0, j1));
    }
    boxes
}

fn step_sheet_scratch(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    solid: Option<&[u8]>,
    banks: bool,
    sub_cap: f32,
    pace: u8,
    sc: &mut Scratch,
    sc2: Option<&mut Scratch>,
) {
    if ISOLATE_SCRATCH.with(|flag| flag.replace(false)) {
        sc.fit(res * res);
        sc.a.fill(0.0);
        sc.b.fill(0.0);
        sc.c.fill(0.0);
        sc.d.fill(0.0);
        sc.e.fill(0.0);
        sc.removed.fill(0.0);
        sc.added.fill(0.0);
    }
    // A mask does not glue separate tire disks into one rectangle. A single
    // connected sheet still returns the historical halo-3 box.
    // Pace 0 is the open 2 cm sheet: one CFL pass, flat tiles asleep. Pace 1
    // is adaptive CFL on awake tiles, capped at eight. Pace 2 and 3 split the
    // whole wet rectangle the same way. A wall keeps the substep loop and the
    // forward Gauss–Seidel sweep.
    let open_fine = solid.is_none()
        && cell <= FINE_CELL_M + 1.0e-6
        && matches!(drive, SheetDrive::Gravity { .. });
    let fine_open = open_fine && pace == 0;
    // Pace 2 and 3 split the whole wet rectangle by adaptive CFL, capped at
    // eight. Pace 0 keeps flat tiles asleep. Pace 1 uses awake tiles only.
    let whole_wet = open_fine && (pace == 2 || pace == 3);
    let awake = open_fine && !whole_wet && !fine_open;
    let boxes = if fine_open || awake {
        awake_work_boxes(h, u, v, bed, res, cell, eps)
    } else {
        sheet_work_boxes(h, res, eps, true)
    };
    if boxes.is_empty() {
        return;
    }
    let cap = if sub_cap.is_finite() {
        sub_cap.clamp(1.0, 8.0)
    } else {
        8.0
    };
    let nsub = if fine_open {
        1
    } else {
        let mut max_s = 0.15f32;
        if let SheetDrive::Wind { x, z, .. } = drive {
            max_s = max_s.max(x.hypot(z));
        }
        // Largest squared speed first, one square root after.
        let mut max_s2 = 0.0f32;
        for &(i0, i1, j0, j1) in &boxes {
            for j in j0..=j1 {
                let row = j * res;
                for (uu, vv) in u[row + i0..=row + i1].iter().zip(&v[row + i0..=row + i1]) {
                    let s2 = uu * uu + vv * vv;
                    if s2 > max_s2 {
                        max_s2 = s2;
                    }
                }
            }
        }
        max_s = max_s.max(max_s2.sqrt());
        // Open wind on the halved dust plane is an upwind donor cell. Courant
        // 0.85 is still under 1 for the Niva headwind, so that plane takes one
        // substep at 1/60 and two at 1/30. A wall and gravity stay on 0.42.
        let open_wind = solid.is_none() && matches!(drive, SheetDrive::Wind { .. });
        let courant = if open_wind { 0.85 } else { 0.42 };
        ((dt * max_s) / (courant * cell.max(1.0e-3)))
            .ceil()
            .clamp(1.0, cap) as u32
    };
    let dts = dt / nsub as f32;
    // Pace 3: one pressure solve on the whole wet sheet, then one flux per
    // Courant slice. Pace 2 repeats the Jacobi sweep every slice. The shore
    // equality test keeps pace 2 as the reference.
    if pace == 3 && whole_wet && nsub > 1 {
        for &(i0, i1, j0, j1) in &boxes {
            substep(
                h, u, v, bed, nu, gate, res, cell, dt, drive, eps, i0, i1, j0, j1, solid, banks,
                false, sc,
            );
        }
        let frac_cap = 0.45f32;
        for _ in 0..nsub {
            for &(i0, i1, j0, j1) in &boxes {
                transport_open(
                    h, u, gate, res, cell, dts, i0, i1, j0, j1, true, None, frac_cap, sc,
                );
                transport_open(
                    h, v, gate, res, cell, dts, i0, i1, j0, j1, false, None, frac_cap, sc,
                );
            }
        }
        if eps > 0.0 {
            for &(i0, i1, j0, j1) in &boxes {
                for j in j0..=j1 {
                    let row = j * res;
                    for i in i0..=i1 {
                        let idx = row + i;
                        if h[idx] < eps {
                            h[idx] = 0.0;
                            u[idx] = 0.0;
                            v[idx] = 0.0;
                        }
                    }
                }
            }
        }
        return;
    }
    // Two blobs far enough apart (the left and right tire tracks) touch
    // disjoint cells: a substep writes its box and a 1-cell rim and reads a
    // 2-cell rim. Those two boxes step on two workers, each on its own scratch.
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(sc2) = sc2 {
        if boxes.len() == 2 && boxes_apart(boxes[0], boxes[1], BOX_PAR_GAP) {
            let (b0, b1) = (boxes[0], boxes[1]);
            let n = res * res;
            let hp = ShareMut(h.as_mut_ptr());
            let up = ShareMut(u.as_mut_ptr());
            let vp = ShareMut(v.as_mut_ptr());
            for _ in 0..nsub {
                let run = |b: (usize, usize, usize, usize), sc: &mut Scratch| {
                    // SAFETY: the two boxes are `BOX_PAR_GAP` cells apart, so
                    // the cells one worker writes are never read or written by
                    // the other. The join below ends before `h`, `u`, `v` are
                    // used again.
                    let (h, u, v) = unsafe { (hp.slice(n), up.slice(n), vp.slice(n)) };
                    substep(
                        h, u, v, bed, nu, gate, res, cell, dts, drive, eps, b.0, b.1, b.2, b.3,
                        solid, banks, true, sc,
                    );
                };
                rayon::join(
                    || serial_inner(|| run(b0, &mut *sc)),
                    || serial_inner(|| run(b1, &mut *sc2)),
                );
            }
            return;
        }
    }
    for _ in 0..nsub {
        for &(i0, i1, j0, j1) in &boxes {
            substep(
                h, u, v, bed, nu, gate, res, cell, dts, drive, eps, i0, i1, j0, j1, solid, banks,
                true, sc,
            );
        }
    }
}

/// Cells two work boxes must keep between them to step at the same time.
/// A substep writes up to 1 cell outside its box and reads up to 2.
const BOX_PAR_GAP: usize = 6;

/// True when the boxes are at least `gap` cells apart along x or along z.
fn boxes_apart(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize), gap: usize) -> bool {
    let apart = |a0: usize, a1: usize, b0: usize, b1: usize| a1 + gap < b0 || b1 + gap < a0;
    apart(a.0, a.1, b.0, b.1) || apart(a.2, a.3, b.2, b.3)
}

fn substep(
    h: &mut [f32],
    u: &mut [f32],
    v: &mut [f32],
    bed: &[f32],
    nu: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    drive: SheetDrive,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    solid: Option<&[u8]>,
    banks: bool,
    do_flux: bool,
    sc: &mut Scratch,
) {
    // Only the box and its 2-cell halo are read below: the 5-tap blurs, the
    // 5x5 wall lift, and the 3-point laplacian. Refreshing the whole grid for
    // every box of every substep was the largest memory pass on the sheet.
    let hi0 = i0.saturating_sub(2);
    let hi1 = (i1 + 2).min(res - 1);
    let hj0 = j0.saturating_sub(2);
    let hj1 = (j1 + 2).min(res - 1);
    for j in hj0..=hj1 {
        let row = j * res;
        let a = &mut sc.a[row + hi0..=row + hi1];
        let b = &bed[row + hi0..=row + hi1];
        let hh = &h[row + hi0..=row + hi1];
        for ((a, b), hh) in a.iter_mut().zip(b).zip(hh) {
            *a = *b + *hh;
        }
    }
    // A wall is not a hole in the free surface. Copy the neighbouring water
    // line onto solid cells so the pressure gradient does not fall into the body.
    if let Some(mask) = solid {
        lift_solid_surface(&mut sc.a, h, mask, res, i0, i1, j0, j1, eps, &mut sc.e);
    }
    // Open wind never reads this pressure. Gravity and a walled sheet do.
    let wind_open = solid.is_none() && matches!(drive, SheetDrive::Wind { .. });
    if !wind_open {
        blur_box(&sc.a, &mut sc.b, &mut sc.e, res, i0, i1, j0, j1, &GAUSS);
        blur_box(&sc.a, &mut sc.c, &mut sc.e, res, i0, i1, j0, j1, &STEEP);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                let filtered = 0.62 * sc.b[idx] + 0.38 * sc.c[idx];
                // The filter is the pressure. A share of the raw surface stays in
                // the force so a broad rut still runs until the free surface is flat.
                sc.d[idx] = 0.45 * filtered + 0.55 * sc.a[idx];
            }
        }
    }
    let p = &sc.d;
    let eta = &sc.a;
    if let Some(mask) = solid {
        write_noslip_ghost(u, v, h, mask, res, i0, i1, j0, j1, eps);
    }
    if solid.is_none() {
        match drive {
            SheetDrive::Gravity { vorticity } => {
                force_gravity_open(
                    h, u, v, nu, p, res, cell, dt, vorticity, eps, i0, i1, j0, j1,
                );
            }
            SheetDrive::Wind { x, z, vorticity } => {
                force_wind_open(h, u, v, res, cell, dt, x, z, vorticity, eps, i0, i1, j0, j1);
            }
        }
    } else {
    match drive {
        SheetDrive::Gravity { vorticity } => {
            let mask = solid.expect("solid sheet");
            force_gravity_solid(
                h, u, v, nu, p, mask, res, cell, dt, vorticity, eps, i0, i1, j0, j1,
            );
        }
        SheetDrive::Wind { x, z, vorticity } => {
            let blend = 1.0 - (-dt * 1.6).exp();
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let idx = j * res + i;
                    if h[idx] <= eps {
                        continue;
                    }
                    u[idx] += (x - u[idx]) * blend;
                    v[idx] += (z - v[idx]) * blend;
                    if vorticity > 0.0 {
                        confine_one(u, v, res, i, j, cell, dt, vorticity);
                    }
                }
            }
        }
    }
    }

    // Viscosity overloads the divergence the projection removes. A thick mix
    // asks the filter for a stronger correction, so it resists spreading.
    // The gain stays small: this sheet is a free surface, and the slope force
    // above is what moves the column. The open 2 cm sheet skips that pair of
    // filters: one CFL pass already paid for the pressure blurs, and the
    // correction is 0.012 of dt. A wall, the coarse sheet, and gravity keep it.
    // Open wind does not. The dust caller rescales each layer to the volume it
    // passed in, and that 0.012*dt tweak is not what sets the plume.
    let open_wind = solid.is_none() && matches!(drive, SheetDrive::Wind { .. });
    let project = !open_wind
        && (solid.is_some()
            || cell > FINE_CELL_M + 1.0e-6
            || !matches!(drive, SheetDrive::Gravity { .. }));
    if project {
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                let (ux, vz) = grad_vel(u, v, res, i, j, cell);
                let div = ux + vz;
                let lap_h = lap(eta, res, i, j, cell);
                let overload = 1.0 + (nu[idx] / 231.0).clamp(0.0, 6.0);
                sc.b[idx] = div * overload - 0.015 * overload * lap_h;
            }
        }
        blur_box(&sc.b, &mut sc.c, &mut sc.e, res, i0, i1, j0, j1, &GAUSS);
        blur_box(&sc.b, &mut sc.a, &mut sc.e, res, i0, i1, j0, j1, &STEEP);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                sc.d[idx] = 0.62 * sc.c[idx] + 0.38 * sc.a[idx];
            }
        }
    }
    // Neighbouring cells nearly always share one viscosity (the water sheet
    // has a single value), so the damping exponent is reused until it changes.
    let mut memo_nu = f32::NAN;
    let mut memo_decay = 1.0f32;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            if project {
                // q has the sign of the divergence. Adding its gradient undoes
                // that divergence. Subtracting it was a positive feedback.
                let (qx, qz) = grad(&sc.d, res, i, j, cell);
                u[idx] += 0.012 * dt * qx;
                v[idx] += 0.012 * dt * qz;
            }
            let nu_here = nu[idx];
            if nu_here.to_bits() != memo_nu.to_bits() {
                memo_nu = nu_here;
                memo_decay = (-sheet_damp(nu_here) * dt).exp();
            }
            let decay = memo_decay;
            let lim = match drive {
                SheetDrive::Wind { x, z, .. } => x.hypot(z).max(FILM_SPEED_CAP_MPS),
                SheetDrive::Gravity { .. } => FILM_SPEED_CAP_MPS,
            };
            u[idx] = finite_speed(u[idx] * decay, lim);
            v[idx] = finite_speed(v[idx] * decay, lim);
        }
    }
    if let Some(mask) = solid {
        // Water that is aimed at the body goes around it. Leaving it in the
        // upstream cell is what builds a reflected bore instead of a bow.
        redirect_into_walls(h, u, v, mask, bed, res, cell, dt, i0, i1, j0, j1, sc);
        // No penetration: drop the velocity that points into the wall.
        // Tangential flow stays, which is what shears the wake.
        block_into_solid(u, v, h, mask, res, i0, i1, j0, j1, eps);
        write_noslip_ghost(u, v, h, mask, res, i0, i1, j0, j1, eps);
    }
    if !do_flux {
        return;
    }
    let wall_bed = if solid.is_some() { Some(bed) } else { None };
    let open_bed = if banks { Some(bed) } else { None };
    if solid.is_none() {
        transport_open(
            h, u, gate, res, cell, dt, i0, i1, j0, j1, true, open_bed, 0.45, sc,
        );
        transport_open(
            h, v, gate, res, cell, dt, i0, i1, j0, j1, false, open_bed, 0.45, sc,
        );
    } else {
        transport(
            h, u, v, gate, res, cell, dt, i0, i1, j0, j1, true, solid, wall_bed, sc,
        );
        transport(
            h, u, v, gate, res, cell, dt, i0, i1, j0, j1, false, solid, wall_bed, sc,
        );
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if eps > 0.0 && h[idx] < eps {
                h[idx] = 0.0;
                u[idx] = 0.0;
                v[idx] = 0.0;
            }
        }
    }
}

fn force_gravity_solid(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    nu: &[f32],
    p: &[f32],
    mask: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    #[cfg(not(target_arch = "wasm32"))]
    if j1 - j0 >= 47 && i1 - i0 >= 31 && inner_par() {
        force_gravity_solid_par(
            h, u, v, nu, p, mask, res, cell, dt, vorticity, eps, i0, i1, j0, j1,
        );
        return;
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            let (px, pz) = grad(p, res, i, j, cell);
            let mut mob = sheet_mobility_for(nu[idx]);
            if h[idx] > 0.08 && solid_near(mask, res, i, j, 1) {
                mob = mob.max(1.0);
            }
            u[idx] += -GRAVITY * px * mob * dt;
            v[idx] += -GRAVITY * pz * mob * dt;
            if vorticity > 0.0 {
                confine_one(u, v, res, i, j, cell, dt, vorticity);
            }
        }
    }
}

/// Gravity is bit-exact across rows. Confinement stays a forward Gauss–Seidel
/// sweep: a Jacobi sweep moves the standing slurry out of the golden band.
#[cfg(not(target_arch = "wasm32"))]
fn force_gravity_solid_par(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    nu: &[f32],
    p: &[f32],
    mask: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let up = ShareMut(u.as_mut_ptr());
    let vp = ShareMut(v.as_mut_ptr());
    let un = u.len();
    let vn = v.len();
    let mask_p = ShareU8(mask.as_ptr());
    let mn = mask.len();
    let threads = rayon::current_num_threads().clamp(1, 4);
    let rows = j1 + 1 - j0;
    let chunk = rows.div_ceil(threads);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = j0 + t * chunk;
            if a > j1 {
                break;
            }
            let b = (a + chunk - 1).min(j1);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
            s.spawn(move |_| {
                let u = unsafe { up.slice(un) };
                let v = unsafe { vp.slice(vn) };
                let mask = unsafe { mask_p.slice(mn) };
                for j in a..=b {
                    for i in i0..=i1 {
                        let idx = j * res + i;
                        if h[idx] <= eps {
                            continue;
                        }
                        let (px, pz) = grad(p, res, i, j, cell);
                        let mut mob = sheet_mobility_for(nu[idx]);
                        if h[idx] > 0.08 && solid_near(mask, res, i, j, 1) {
                            mob = mob.max(1.0);
                        }
                        u[idx] += -GRAVITY * px * mob * dt;
                        v[idx] += -GRAVITY * pz * mob * dt;
                    }
                }
            });
        }
    });
    if vorticity <= 0.0 {
        return;
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            confine_one(u, v, res, i, j, cell, dt, vorticity);
        }
    }
}

fn force_gravity_open(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    nu: &[f32],
    p: &[f32],
    res: usize,
    cell: f32,
    dt: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    // The 2 cm sheet stays on a snapshot Jacobi sweep even when the awake box
    // is small. The forward sweep below is the solid-sized path.
    #[cfg(not(target_arch = "wasm32"))]
    if cell <= FINE_CELL_M + 1.0e-6 {
        gravity_jacobi_serial(
            h, u, v, nu, p, res, cell, dt, vorticity, eps, i0, i1, j0, j1,
        );
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    if j1 - j0 >= 47 && i1 - i0 >= 31 && inner_par() {
        force_gravity_par(h, u, v, nu, p, res, cell, dt, vorticity, eps, i0, i1, j0, j1);
        return;
    }
    let two = 2.0 * cell;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            let (px, pz) = if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                (
                    (p[idx + 1] - p[idx - 1]) / two,
                    (p[idx + res] - p[idx - res]) / two,
                )
            } else {
                grad(p, res, i, j, cell)
            };
            let mob = sheet_mobility_for(nu[idx]);
            u[idx] += -GRAVITY * px * mob * dt;
            v[idx] += -GRAVITY * pz * mob * dt;
            if vorticity > 0.0 {
                if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                    confine_interior(u, v, res, i, j, dt, vorticity, two);
                } else {
                    confine_one(u, v, res, i, j, cell, dt, vorticity);
                }
            }
        }
    }
}

/// Gravity, then one Jacobi confinement sweep from the post-gravity velocity.
/// Rows do not read each other's confinement update, so the sweep is parallel.
#[cfg(not(target_arch = "wasm32"))]
fn force_gravity_par(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    nu: &[f32],
    p: &[f32],
    res: usize,
    cell: f32,
    dt: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let two = 2.0 * cell;
    let up = ShareMut(u.as_mut_ptr());
    let vp = ShareMut(v.as_mut_ptr());
    let un = u.len();
    let vn = v.len();
    let threads = rayon::current_num_threads().clamp(1, 4);
    let rows = j1 + 1 - j0;
    let chunk = rows.div_ceil(threads);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = j0 + t * chunk;
            if a > j1 {
                break;
            }
            let b = (a + chunk - 1).min(j1);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
            s.spawn(move |_| {
                let u = unsafe { up.slice(un) };
                let v = unsafe { vp.slice(vn) };
                for j in a..=b {
                    for i in i0..=i1 {
                        let idx = j * res + i;
                        if h[idx] <= eps {
                            continue;
                        }
                        let (px, pz) = if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                            (
                                (p[idx + 1] - p[idx - 1]) / two,
                                (p[idx + res] - p[idx - res]) / two,
                            )
                        } else {
                            grad(p, res, i, j, cell)
                        };
                        let mob = sheet_mobility_for(nu[idx]);
                        u[idx] += -GRAVITY * px * mob * dt;
                        v[idx] += -GRAVITY * pz * mob * dt;
                    }
                }
            });
        }
    });
    if vorticity <= 0.0 {
        return;
    }
    let n = res * res;
    VEL_SNAP.with(|snap| {
        let mut snap = snap.borrow_mut();
        if snap.u.len() < n {
            snap.u.resize(n, 0.0);
            snap.v.resize(n, 0.0);
        }
        snap.u[..n].copy_from_slice(&u[..n]);
        snap.v[..n].copy_from_slice(&v[..n]);
    });
    VEL_SNAP.with(|snap| {
        let snap = snap.borrow();
        let su = ShareConst(snap.u.as_ptr());
        let sv = ShareConst(snap.v.as_ptr());
        rayon::scope(|s| {
            #[cfg(debug_assertions)]
            let mut stripe = 0usize;
            for t in 0..threads {
                let a = j0 + t * chunk;
                if a > j1 {
                    break;
                }
                let b = (a + chunk - 1).min(j1);
                #[cfg(debug_assertions)]
                ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
                s.spawn(move |_| {
                    let u = unsafe { up.slice(un) };
                    let v = unsafe { vp.slice(vn) };
                    let su = unsafe { su.slice(n) };
                    let sv = unsafe { sv.slice(n) };
                    for j in a..=b {
                        for i in i0..=i1 {
                            let idx = j * res + i;
                            if h[idx] <= eps {
                                continue;
                            }
                            if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                                confine_jacobi_interior(
                                    su, sv, u, v, res, i, j, dt, vorticity, two,
                                );
                            } else {
                                confine_jacobi(su, sv, u, v, res, i, j, cell, dt, vorticity);
                            }
                        }
                    }
                });
            }
        });
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn confine_jacobi(
    su: &[f32],
    sv: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    i: usize,
    j: usize,
    cell: f32,
    dt: f32,
    eps: f32,
) {
    let w = vorticity_at(su, sv, res, i, j, cell);
    if !w.is_finite() || w.abs() < 1.0e-4 {
        return;
    }
    let wxp = vorticity_at(su, sv, res, (i + 1).min(res - 1), j, cell).abs();
    let wxm = vorticity_at(su, sv, res, i.saturating_sub(1), j, cell).abs();
    let wzp = vorticity_at(su, sv, res, i, (j + 1).min(res - 1), cell).abs();
    let wzm = vorticity_at(su, sv, res, i, j.saturating_sub(1), cell).abs();
    let nx = (wxp - wxm) / (2.0 * cell);
    let nz = (wzp - wzm) / (2.0 * cell);
    let du = eps * dt * (-nz * w);
    let dv = eps * dt * (nx * w);
    if du.is_finite() && dv.is_finite() && du * du + dv * dv <= 0.0625 {
        let idx = j * res + i;
        u[idx] += du;
        v[idx] += dv;
    }
}

/// Interior Jacobi confinement from a vorticity plane. `w[idx]` is
/// `vort_interior` of the snapshot, so the five-stencil repeat is one load.
#[cfg(not(target_arch = "wasm32"))]
fn jacobi_confine_box(
    su: &[f32],
    sv: &[f32],
    w: &mut [f32],
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    cell: f32,
    dt: f32,
    vorticity: f32,
    eps: f32,
    two: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let i_lo = i0.saturating_sub(1).max(1);
    let j_lo = j0.saturating_sub(1).max(1);
    let i_hi = (i1 + 1).min(res.saturating_sub(2));
    let j_hi = (j1 + 1).min(res.saturating_sub(2));
    if i_lo <= i_hi && j_lo <= j_hi {
        for j in j_lo..=j_hi {
            let row = j * res;
            for i in i_lo..=i_hi {
                w[row + i] = vort_interior(su, sv, res, i, j, two);
            }
        }
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                let omega = w[idx];
                if !omega.is_finite() || omega.abs() < 1.0e-4 {
                    continue;
                }
                let wxp = w[idx + 1].abs();
                let wxm = w[idx - 1].abs();
                let wzp = w[idx + res].abs();
                let wzm = w[idx - res].abs();
                let nx = (wxp - wxm) / two;
                let nz = (wzp - wzm) / two;
                let du = vorticity * dt * (-nz * omega);
                let dv = vorticity * dt * (nx * omega);
                if du.is_finite() && dv.is_finite() && du * du + dv * dv <= 0.0625 {
                    u[idx] += du;
                    v[idx] += dv;
                }
            } else {
                confine_jacobi(su, sv, u, v, res, i, j, cell, dt, vorticity);
            }
        }
    }
}

fn confine_jacobi_interior(
    su: &[f32],
    sv: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    i: usize,
    j: usize,
    dt: f32,
    eps: f32,
    two: f32,
) {
    let w = vort_interior(su, sv, res, i, j, two);
    if !w.is_finite() || w.abs() < 1.0e-4 {
        return;
    }
    let wxp = vort_interior(su, sv, res, i + 1, j, two).abs();
    let wxm = vort_interior(su, sv, res, i - 1, j, two).abs();
    let wzp = vort_interior(su, sv, res, i, j + 1, two).abs();
    let wzm = vort_interior(su, sv, res, i, j - 1, two).abs();
    let nx = (wxp - wxm) / two;
    let nz = (wzp - wzm) / two;
    let du = eps * dt * (-nz * w);
    let dv = eps * dt * (nx * w);
    if du.is_finite() && dv.is_finite() && du * du + dv * dv <= 0.0625 {
        let idx = j * res + i;
        u[idx] += du;
        v[idx] += dv;
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct VelSnap {
    u: Vec<f32>,
    v: Vec<f32>,
    /// Vorticity of the snapshot, one value per cell. Interior confinement
    /// reads five of these instead of recomputing each stencil.
    w: Vec<f32>,
}

#[cfg(not(target_arch = "wasm32"))]
thread_local! {
    static VEL_SNAP: RefCell<VelSnap> = RefCell::new(VelSnap {
        u: Vec::new(),
        v: Vec::new(),
        w: Vec::new(),
    });
}

/// A 64² dust plane spawns more rayon scopes than the arithmetic is worth.
/// Above this, the coarse window still uses the workers. The serial sweep
/// writes the same Jacobi update those workers write.
#[cfg(not(target_arch = "wasm32"))]
const SERIAL_SHEET_MAX: usize = 96;

/// Snapshot Jacobi on the calling thread. Same sweep as the row workers:
/// gravity, then confinement from the velocity those adds just wrote.
#[cfg(not(target_arch = "wasm32"))]
fn gravity_jacobi_serial(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    nu: &[f32],
    p: &[f32],
    res: usize,
    cell: f32,
    dt: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let two = 2.0 * cell;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            let (px, pz) = if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                (
                    (p[idx + 1] - p[idx - 1]) / two,
                    (p[idx + res] - p[idx - res]) / two,
                )
            } else {
                grad(p, res, i, j, cell)
            };
            let mob = sheet_mobility_for(nu[idx]);
            u[idx] += -GRAVITY * px * mob * dt;
            v[idx] += -GRAVITY * pz * mob * dt;
        }
    }
    if vorticity <= 0.0 {
        return;
    }
    let n = res * res;
    VEL_SNAP.with(|snap| {
        let mut snap = snap.borrow_mut();
        if snap.u.len() < n || snap.w.len() < n {
            snap.u.resize(n, 0.0);
            snap.v.resize(n, 0.0);
            snap.w.resize(n, 0.0);
        }
        let y0 = j0.saturating_sub(2);
        let y1 = (j1 + 2).min(res - 1);
        let x0 = i0.saturating_sub(2);
        let x1 = (i1 + 2).min(res - 1);
        for j in y0..=y1 {
            let row = j * res;
            let end = row + x1 + 1;
            snap.u[row + x0..end].copy_from_slice(&u[row + x0..end]);
            snap.v[row + x0..end].copy_from_slice(&v[row + x0..end]);
        }
        let VelSnap { u: su, v: sv, w } = &mut *snap;
        jacobi_confine_box(
            su, sv, w, h, u, v, res, cell, dt, vorticity, eps, two, i0, i1, j0, j1,
        );
    });
}

/// Wind blend, then one Jacobi confinement from that velocity. Same cells as
/// `force_wind_par`, on the calling thread.
#[cfg(not(target_arch = "wasm32"))]
fn force_wind_jacobi_serial(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    cell: f32,
    dt: f32,
    x: f32,
    z: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let blend = 1.0 - (-dt * 1.6).exp();
    let two = 2.0 * cell;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            u[idx] += (x - u[idx]) * blend;
            v[idx] += (z - v[idx]) * blend;
        }
    }
    if vorticity <= 0.0 {
        return;
    }
    let n = res * res;
    VEL_SNAP.with(|snap| {
        let mut snap = snap.borrow_mut();
        if snap.u.len() < n || snap.w.len() < n {
            snap.u.resize(n, 0.0);
            snap.v.resize(n, 0.0);
            snap.w.resize(n, 0.0);
        }
        snap.u[..n].copy_from_slice(&u[..n]);
        snap.v[..n].copy_from_slice(&v[..n]);
        let VelSnap { u: su, v: sv, w } = &mut *snap;
        jacobi_confine_box(
            su, sv, w, h, u, v, res, cell, dt, vorticity, eps, two, i0, i1, j0, j1,
        );
    });
}

fn force_wind_open(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    cell: f32,
    dt: f32,
    x: f32,
    z: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    #[cfg(not(target_arch = "wasm32"))]
    if j1 - j0 >= 47 && i1 - i0 >= 31 && inner_par() {
        // Dust is stepped on the halved air grid. Same Jacobi confinement as
        // the row workers, without a scope per blur and per layer.
        if res <= SERIAL_SHEET_MAX {
            force_wind_jacobi_serial(
                h, u, v, res, cell, dt, x, z, vorticity, eps, i0, i1, j0, j1,
            );
        } else {
            force_wind_par(h, u, v, res, cell, dt, x, z, vorticity, eps, i0, i1, j0, j1);
        }
        return;
    }
    let blend = 1.0 - (-dt * 1.6).exp();
    let two = 2.0 * cell;
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if h[idx] <= eps {
                continue;
            }
            u[idx] += (x - u[idx]) * blend;
            v[idx] += (z - v[idx]) * blend;
            if vorticity > 0.0 {
                if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                    confine_interior(u, v, res, i, j, dt, vorticity, two);
                } else {
                    confine_one(u, v, res, i, j, cell, dt, vorticity);
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn force_wind_par(
    h: &[f32],
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    cell: f32,
    dt: f32,
    x: f32,
    z: f32,
    vorticity: f32,
    eps: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
) {
    let blend = 1.0 - (-dt * 1.6).exp();
    let two = 2.0 * cell;
    let up = ShareMut(u.as_mut_ptr());
    let vp = ShareMut(v.as_mut_ptr());
    let un = u.len();
    let vn = v.len();
    let threads = rayon::current_num_threads().clamp(1, 4);
    let rows = j1 + 1 - j0;
    let chunk = rows.div_ceil(threads);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = j0 + t * chunk;
            if a > j1 {
                break;
            }
            let b = (a + chunk - 1).min(j1);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
            s.spawn(move |_| {
                let u = unsafe { up.slice(un) };
                let v = unsafe { vp.slice(vn) };
                for j in a..=b {
                    for i in i0..=i1 {
                        let idx = j * res + i;
                        if h[idx] <= eps {
                            continue;
                        }
                        u[idx] += (x - u[idx]) * blend;
                        v[idx] += (z - v[idx]) * blend;
                    }
                }
            });
        }
    });
    if vorticity <= 0.0 {
        return;
    }
    let n = res * res;
    VEL_SNAP.with(|snap| {
        let mut snap = snap.borrow_mut();
        if snap.u.len() < n {
            snap.u.resize(n, 0.0);
            snap.v.resize(n, 0.0);
        }
        snap.u[..n].copy_from_slice(&u[..n]);
        snap.v[..n].copy_from_slice(&v[..n]);
    });
    VEL_SNAP.with(|snap| {
        let snap = snap.borrow();
        let su = ShareConst(snap.u.as_ptr());
        let sv = ShareConst(snap.v.as_ptr());
        rayon::scope(|s| {
            #[cfg(debug_assertions)]
            let mut stripe = 0usize;
            for t in 0..threads {
                let a = j0 + t * chunk;
                if a > j1 {
                    break;
                }
                let b = (a + chunk - 1).min(j1);
                #[cfg(debug_assertions)]
                ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
                s.spawn(move |_| {
                    let u = unsafe { up.slice(un) };
                    let v = unsafe { vp.slice(vn) };
                    let su = unsafe { su.slice(n) };
                    let sv = unsafe { sv.slice(n) };
                    for j in a..=b {
                        for i in i0..=i1 {
                            let idx = j * res + i;
                            if h[idx] <= eps {
                                continue;
                            }
                            if i >= 2 && j >= 2 && i + 2 < res && j + 2 < res {
                                confine_jacobi_interior(
                                    su, sv, u, v, res, i, j, dt, vorticity, two,
                                );
                            } else {
                                confine_jacobi(su, sv, u, v, res, i, j, cell, dt, vorticity);
                            }
                        }
                    }
                });
            }
        });
    });
}

fn confine_interior(
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    i: usize,
    j: usize,
    dt: f32,
    eps: f32,
    two: f32,
) {
    let w = vort_interior(u, v, res, i, j, two);
    if !w.is_finite() || w.abs() < 1.0e-4 {
        return;
    }
    let wxp = vort_interior(u, v, res, i + 1, j, two).abs();
    let wxm = vort_interior(u, v, res, i - 1, j, two).abs();
    let wzp = vort_interior(u, v, res, i, j + 1, two).abs();
    let wzm = vort_interior(u, v, res, i, j - 1, two).abs();
    let nx = (wxp - wxm) / two;
    let nz = (wzp - wzm) / two;
    let du = eps * dt * (-nz * w);
    let dv = eps * dt * (nx * w);
    if du.is_finite() && dv.is_finite() && du * du + dv * dv <= 0.0625 {
        let idx = j * res + i;
        u[idx] += du;
        v[idx] += dv;
    }
}

fn vort_interior(u: &[f32], v: &[f32], res: usize, i: usize, j: usize, two: f32) -> f32 {
    let dvdx = (v[j * res + i + 1] - v[j * res + i - 1]) / two;
    let dudz = (u[(j + 1) * res + i] - u[(j - 1) * res + i]) / two;
    dvdx - dudz
}

/// A dry cell whose bed stands above the donor's free surface takes no water.
/// A flat bed never trips this, so a pour on a plane still spreads.
#[inline(always)]
fn dry_bank(h: &[f32], bed: &[f32], donor: usize, recv: usize) -> bool {
    h[recv] <= 1.0e-6 && bed[recv] > bed[donor] + h[donor] + 1.0e-3
}

fn transport_open(
    h: &mut [f32],
    vel: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    bed: Option<&[f32]>,
    frac_cap: f32,
    sc: &mut Scratch,
) {
    #[cfg(not(target_arch = "wasm32"))]
    // The 2 cm sheet and the halved dust plane stay on this thread. Each row
    // or column still writes the same edges. The coarse window stays split.
    if i1 - i0 >= 31 && j1 - j0 >= 47 && res != FINE_RES && res > SERIAL_SHEET_MAX && inner_par() {
        transport_open_par(
            h, vel, gate, res, cell, dt, i0, i1, j0, j1, along_x, bed, frac_cap, sc,
        );
        return;
    }
    match bed {
        Some(bed) => transport_open_rows::<true>(
            h, vel, gate, res, cell, dt, i0, i1, j0, j1, along_x, bed, frac_cap, sc,
        ),
        None => transport_open_rows::<false>(
            h, vel, gate, res, cell, dt, i0, i1, j0, j1, along_x, &[], frac_cap, sc,
        ),
    }
}

/// One open-flux edge, the same tests as `flux_open` after the speed and
/// dry-bank checks. Returns 0.0 where the serial loop would skip the edge.
/// Adding 0.0 to a removed or added sum leaves it unchanged (those sums are
/// never -0.0), so the sums match the branchy loop bit for bit.
#[inline(always)]
fn open_edge_dh<const BED: bool>(
    hd: f32,
    hr: f32,
    remd: f32,
    gate_r: u8,
    bed_d: f32,
    bed_r: f32,
    speed: f32,
    dt: f32,
    cell: f32,
    frac_cap: f32,
) -> f32 {
    let live = !(speed.abs() < 1.0e-5);
    let bank = BED && hr <= 1.0e-6 && bed_r > bed_d + hd + 1.0e-3;
    let room = hd - remd;
    let gate_shut = gate_r == 0 && hr <= 1.0e-6;
    let frac = (speed.abs() * dt / cell).min(frac_cap);
    let dh = (hd * frac).min(room);
    let ok = live & !bank & !(room <= 1.0e-8) & !gate_shut & !(dh <= 1.0e-8);
    if ok {
        dh
    } else {
        0.0
    }
}

/// One row pair of the column pass. Separate slices, so the compiler knows
/// the removed and added rows of the two cells do not overlap.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn open_row_pair<const BED: bool>(
    ha: &[f32],
    hb: &[f32],
    va: &[f32],
    vb: &[f32],
    ga: &[u8],
    gb: &[u8],
    ba: &[f32],
    bb: &[f32],
    ra: &mut [f32],
    rb: &mut [f32],
    aa: &mut [f32],
    ab: &mut [f32],
    dt: f32,
    cell: f32,
    frac_cap: f32,
) {
    let m = ha.len();
    let (hb, va, vb, ga, gb) = (&hb[..m], &va[..m], &vb[..m], &ga[..m], &gb[..m]);
    let (ra, rb, aa, ab) = (&mut ra[..m], &mut rb[..m], &mut aa[..m], &mut ab[..m]);
    // Up-going first: donor b, whose removed sum is still 0.0. Then
    // down-going: donor a, whose removed sum holds the up-going flux of the
    // row pair above. Each cell gets its two edges in the serial order.
    for i in 0..m {
        let speed = 0.5 * (va[i] + vb[i]);
        let (bd, bc) = if BED { (bb[i], ba[i]) } else { (0.0, 0.0) };
        let dh = open_edge_dh::<BED>(hb[i], ha[i], 0.0, ga[i], bd, bc, speed, dt, cell, frac_cap);
        let up = if speed > 0.0 { 0.0 } else { dh };
        rb[i] += up;
        aa[i] += up;
    }
    for i in 0..m {
        let speed = 0.5 * (va[i] + vb[i]);
        let (bd, bc) = if BED { (ba[i], bb[i]) } else { (0.0, 0.0) };
        let dh = open_edge_dh::<BED>(ha[i], hb[i], ra[i], gb[i], bd, bc, speed, dt, cell, frac_cap);
        let down = if speed > 0.0 { dh } else { 0.0 };
        ra[i] += down;
        ab[i] += down;
    }
}

/// `transport_open` without branches in the edge loop.
///
/// Along a row the serial loop is ordered by edge, but an edge only reads the
/// removed sum of its donor, and before edge k that sum holds at most the
/// left-going flux of edge k - 1. So the left-going fluxes come first (each
/// one only reads depths), then the right-going ones, then each cell adds its
/// two edges in the order the serial loop did. Down a column the rows stay in
/// order and the cells of one row pair are independent.
#[allow(clippy::too_many_arguments)]
fn transport_open_rows<const BED: bool>(
    h: &mut [f32],
    vel: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    bed: &[f32],
    frac_cap: f32,
    sc: &mut Scratch,
) {
    // The same scalar IEEE operations either way; AVX2 only widens the lanes.
    // No FMA is enabled, so no multiply and add are fused.
    #[cfg(target_arch = "x86_64")]
    if avx2_cached() {
        unsafe {
            transport_open_rows_avx2::<BED>(
                h, vel, gate, res, cell, dt, i0, i1, j0, j1, along_x, bed, frac_cap, sc,
            )
        };
        return;
    }
    transport_open_rows_body::<BED>(
        h, vel, gate, res, cell, dt, i0, i1, j0, j1, along_x, bed, frac_cap, sc,
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
unsafe fn transport_open_rows_avx2<const BED: bool>(
    h: &mut [f32],
    vel: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    bed: &[f32],
    frac_cap: f32,
    sc: &mut Scratch,
) {
    transport_open_rows_body::<BED>(
        h, vel, gate, res, cell, dt, i0, i1, j0, j1, along_x, bed, frac_cap, sc,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn transport_open_rows_body<const BED: bool>(
    h: &mut [f32],
    vel: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    bed: &[f32],
    frac_cap: f32,
    sc: &mut Scratch,
) {
    let m = i1 + 1 - i0;
    if along_x {
        if sc.flux_l.len() < m {
            sc.flux_l.resize(m, 0.0);
            sc.flux_r.resize(m, 0.0);
        }
        for j in j0..=j1 {
            let lo = j * res + i0;
            let hi = lo + m;
            let hr = &mut h[lo..hi];
            let vr = &vel[lo..hi];
            let gr = &gate[lo..hi];
            let br: &[f32] = if BED { &bed[lo..hi] } else { &[] };
            let fl = &mut sc.flux_l[..m];
            let fr = &mut sc.flux_r[..m];
            let edges = m - 1;
            // Left-going: donor k + 1, whose removed sum is still 0.0.
            for k in 0..edges {
                let speed = 0.5 * (vr[k] + vr[k + 1]);
                let (bd, bc) = if BED { (br[k + 1], br[k]) } else { (0.0, 0.0) };
                let dh = open_edge_dh::<BED>(
                    hr[k + 1], hr[k], 0.0, gr[k], bd, bc, speed, dt, cell, frac_cap,
                );
                fl[k] = if speed > 0.0 { 0.0 } else { dh };
            }
            // Right-going: donor k, whose removed sum is 0.0 + the left-going
            // flux of edge k - 1.
            for k in 0..edges {
                let speed = 0.5 * (vr[k] + vr[k + 1]);
                let prev = if k > 0 { fl[k - 1] } else { 0.0 };
                let (bd, bc) = if BED { (br[k], br[k + 1]) } else { (0.0, 0.0) };
                let dh = open_edge_dh::<BED>(
                    hr[k], hr[k + 1], 0.0 + prev, gr[k + 1], bd, bc, speed, dt, cell, frac_cap,
                );
                fr[k] = if speed > 0.0 { dh } else { 0.0 };
            }
            fl[edges] = 0.0;
            fr[edges] = 0.0;
            let rem = &mut sc.removed[lo..hi];
            let add = &mut sc.added[lo..hi];
            // Cell c: removed = 0.0 + left-going of edge c - 1, then + right-going
            // of edge c. Added = 0.0 + right-going of edge c - 1, then + left-going
            // of edge c. Cell 0 has no edge on its left in this pass.
            rem[0] = (0.0 + 0.0) + fr[0];
            add[0] = (0.0 + 0.0) + fl[0];
            for c in 1..m {
                rem[c] = (0.0 + fl[c - 1]) + fr[c];
                add[c] = (0.0 + fr[c - 1]) + fl[c];
            }
            // A select, not a branch, so the row vectorises. Same result per cell.
            for ((hv, &r), &a) in hr.iter_mut().zip(rem.iter()).zip(add.iter()) {
                let moved = (*hv - r + a).max(0.0);
                *hv = if r > 0.0 || a > 0.0 { moved } else { *hv };
            }
        }
        return;
    }
    for j in j0..=j1 {
        let lo = j * res + i0;
        sc.removed[lo..lo + m].fill(0.0);
        sc.added[lo..lo + m].fill(0.0);
    }
    for j in j0..j1 {
        let lo = j * res + i0;
        let (top, low) = h.split_at(lo + res);
        let ha = &top[lo..lo + m];
        let hb = &low[..m];
        let va = &vel[lo..lo + m];
        let vb = &vel[lo + res..lo + res + m];
        let ga = &gate[lo..lo + m];
        let gb = &gate[lo + res..lo + res + m];
        let (rem_top, rem_low) = sc.removed.split_at_mut(lo + res);
        let ra = &mut rem_top[lo..lo + m];
        let rb = &mut rem_low[..m];
        let (add_top, add_low) = sc.added.split_at_mut(lo + res);
        let aa = &mut add_top[lo..lo + m];
        let ab = &mut add_low[..m];
        let (ba, bb): (&[f32], &[f32]) = if BED {
            (&bed[lo..lo + m], &bed[lo + res..lo + res + m])
        } else {
            (&[], &[])
        };
        open_row_pair::<BED>(ha, hb, va, vb, ga, gb, ba, bb, ra, rb, aa, ab, dt, cell, frac_cap);
    }
    for j in j0..=j1 {
        let lo = j * res + i0;
        let hi = lo + m;
        for ((hv, &r), &a) in h[lo..hi]
            .iter_mut()
            .zip(&sc.removed[lo..hi])
            .zip(&sc.added[lo..hi])
        {
            let moved = (*hv - r + a).max(0.0);
            *hv = if r > 0.0 || a > 0.0 { moved } else { *hv };
        }
    }
}

#[cfg(test)]
fn transport_open_ref(
    h: &mut [f32],
    vel: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    bed: Option<&[f32]>,
    frac_cap: f32,
    sc: &mut Scratch,
) {
    for j in j0..=j1 {
        let row = j * res;
        for i in i0..=i1 {
            let idx = row + i;
            sc.removed[idx] = 0.0;
            sc.added[idx] = 0.0;
        }
    }
    if along_x {
        for j in j0..=j1 {
            let row = j * res;
            for i in i0..i1 {
                let a = row + i;
                let b = a + 1;
                let speed = 0.5 * (vel[a] + vel[b]);
                if speed.abs() < 1.0e-5 {
                    continue;
                }
                if let Some(bed) = bed {
                    let (donor, recv) = if speed > 0.0 { (a, b) } else { (b, a) };
                    if dry_bank(h, bed, donor, recv) {
                        continue;
                    }
                }
                flux_open(h, gate, a, b, speed, cell, dt, frac_cap, sc);
            }
        }
    } else {
        for j in j0..j1 {
            for i in i0..=i1 {
                let a = j * res + i;
                let b = a + res;
                let speed = 0.5 * (vel[a] + vel[b]);
                if speed.abs() < 1.0e-5 {
                    continue;
                }
                if let Some(bed) = bed {
                    let (donor, recv) = if speed > 0.0 { (a, b) } else { (b, a) };
                    if dry_bank(h, bed, donor, recv) {
                        continue;
                    }
                }
                flux_open(h, gate, a, b, speed, cell, dt, frac_cap, sc);
            }
        }
    }
    for j in j0..=j1 {
        let lo = j * res + i0;
        let hi = j * res + i1 + 1;
        // A select, not a branch, so the row vectorises. Same result per cell.
        for ((hv, &r), &a) in h[lo..hi]
            .iter_mut()
            .zip(&sc.removed[lo..hi])
            .zip(&sc.added[lo..hi])
        {
            let moved = (*hv - r + a).max(0.0);
            *hv = if r > 0.0 || a > 0.0 { moved } else { *hv };
        }
    }
}

/// Horizontal flux stays on one row. Vertical flux stays on one column.
/// Workers do not share a written index.
#[cfg(not(target_arch = "wasm32"))]
fn transport_open_par(
    h: &mut [f32],
    vel: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    bed: Option<&[f32]>,
    frac_cap: f32,
    sc: &mut Scratch,
) {
    let threads = rayon::current_num_threads().clamp(1, 4);
    let bed_ptr = bed.map(|b| ShareConst(b.as_ptr()));
    let bed_len = bed.map(|b| b.len()).unwrap_or(0);
    let removed = ShareMut(sc.removed.as_mut_ptr());
    let added = ShareMut(sc.added.as_mut_ptr());
    let n = sc.removed.len().min(sc.added.len());
    let depth = ShareConst(h.as_ptr());
    let hn = h.len();
    let rows = j1 + 1 - j0;
    let chunk = rows.div_ceil(threads);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = j0 + t * chunk;
            if a > j1 {
                break;
            }
            let b = (a + chunk - 1).min(j1);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
            s.spawn(move |_| {
                let removed = unsafe { removed.slice(n) };
                let added = unsafe { added.slice(n) };
                for j in a..=b {
                    let row = j * res;
                    for i in i0..=i1 {
                        removed[row + i] = 0.0;
                        added[row + i] = 0.0;
                    }
                }
            });
        }
    });
    if along_x {
        rayon::scope(|s| {
            #[cfg(debug_assertions)]
            let mut stripe = 0usize;
            for t in 0..threads {
                let a = j0 + t * chunk;
                if a > j1 {
                    break;
                }
                let b = (a + chunk - 1).min(j1);
                #[cfg(debug_assertions)]
                ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
                s.spawn(move |_| {
                    let h = unsafe { depth.slice(hn) };
                    let removed = unsafe { removed.slice(n) };
                    let added = unsafe { added.slice(n) };
                    for j in a..=b {
                        let row = j * res;
                        for i in i0..i1 {
                            let ia = row + i;
                            let ib = ia + 1;
                            let speed = 0.5 * (vel[ia] + vel[ib]);
                            if speed.abs() < 1.0e-5 {
                                continue;
                            }
                            if let Some(ptr) = bed_ptr {
                                let bed = unsafe { ptr.slice(bed_len) };
                                let (donor, recv) = if speed > 0.0 { (ia, ib) } else { (ib, ia) };
                                if dry_bank(h, bed, donor, recv) {
                                    continue;
                                }
                            }
                            flux_open_bufs(h, gate, ia, ib, speed, cell, dt, frac_cap, removed, added);
                        }
                    }
                });
            }
        });
    } else {
        let cols = i1 + 1 - i0;
        let cthreads = threads.min(cols.max(1));
        let cchunk = cols.div_ceil(cthreads);
        rayon::scope(|s| {
            #[cfg(debug_assertions)]
            let mut stripe = 0usize;
            for t in 0..cthreads {
                let a = i0 + t * cchunk;
                if a > i1 {
                    break;
                }
                let b = (a + cchunk - 1).min(i1);
                #[cfg(debug_assertions)]
                ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
                s.spawn(move |_| {
                    let h = unsafe { depth.slice(hn) };
                    let removed = unsafe { removed.slice(n) };
                    let added = unsafe { added.slice(n) };
                    for j in j0..j1 {
                        for i in a..=b {
                            let ia = j * res + i;
                            let ib = ia + res;
                            let speed = 0.5 * (vel[ia] + vel[ib]);
                            if speed.abs() < 1.0e-5 {
                                continue;
                            }
                            if let Some(ptr) = bed_ptr {
                                let bed = unsafe { ptr.slice(bed_len) };
                                let (donor, recv) = if speed > 0.0 { (ia, ib) } else { (ib, ia) };
                                if dry_bank(h, bed, donor, recv) {
                                    continue;
                                }
                            }
                            flux_open_bufs(h, gate, ia, ib, speed, cell, dt, frac_cap, removed, added);
                        }
                    }
                });
            }
        });
    }
    let hp = ShareMut(h.as_mut_ptr());
    let hn = h.len();
    let removed_r = ShareConst(removed.0);
    let added_r = ShareConst(added.0);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = j0 + t * chunk;
            if a > j1 {
                break;
            }
            let b = (a + chunk - 1).min(j1);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
            s.spawn(move |_| {
                let h = unsafe { hp.slice(hn) };
                let removed = unsafe { removed_r.slice(n) };
                let added = unsafe { added_r.slice(n) };
                for j in a..=b {
                    let row = j * res;
                    for i in i0..=i1 {
                        let idx = row + i;
                        if removed[idx] > 0.0 || added[idx] > 0.0 {
                            h[idx] = (h[idx] - removed[idx] + added[idx]).max(0.0);
                        }
                    }
                }
            });
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn flux_open_bufs(
    h: &[f32],
    gate: &[u8],
    a: usize,
    b: usize,
    vel: f32,
    cell: f32,
    dt: f32,
    frac_cap: f32,
    removed: &mut [f32],
    added: &mut [f32],
) {
    let (donor, recv) = if vel > 0.0 { (a, b) } else { (b, a) };
    let room = h[donor] - removed[donor];
    if room <= 1.0e-8 {
        return;
    }
    if gate[recv] == 0 && h[recv] <= 1.0e-6 {
        return;
    }
    let frac = (vel.abs() * dt / cell).min(frac_cap);
    let dh = (h[donor] * frac).min(room);
    if dh <= 1.0e-8 {
        return;
    }
    removed[donor] += dh;
    added[recv] += dh;
}

#[cfg(test)]
#[inline(always)]
fn flux_open(
    h: &[f32],
    gate: &[u8],
    a: usize,
    b: usize,
    vel: f32,
    cell: f32,
    dt: f32,
    frac_cap: f32,
    sc: &mut Scratch,
) {
    let (donor, recv) = if vel > 0.0 { (a, b) } else { (b, a) };
    let room = h[donor] - sc.removed[donor];
    if room <= 1.0e-8 {
        return;
    }
    if gate[recv] == 0 && h[recv] <= 1.0e-6 {
        return;
    }
    let frac = (vel.abs() * dt / cell).min(frac_cap);
    let dh = (h[donor] * frac).min(room);
    if dh <= 1.0e-8 {
        return;
    }
    sc.removed[donor] += dh;
    sc.added[recv] += dh;
}

fn confine_one(
    u: &mut [f32],
    v: &mut [f32],
    res: usize,
    i: usize,
    j: usize,
    cell: f32,
    dt: f32,
    eps: f32,
) {
    let idx = j * res + i;
    let w = vorticity_at(u, v, res, i, j, cell);
    if !w.is_finite() || w.abs() < 1.0e-4 {
        return;
    }
    let wxp = vorticity_at(u, v, res, (i + 1).min(res - 1), j, cell).abs();
    let wxm = vorticity_at(u, v, res, i.saturating_sub(1), j, cell).abs();
    let wzp = vorticity_at(u, v, res, i, (j + 1).min(res - 1), cell).abs();
    let wzm = vorticity_at(u, v, res, i, j.saturating_sub(1), cell).abs();
    let nx = (wxp - wxm) / (2.0 * cell);
    let nz = (wzp - wzm) / (2.0 * cell);
    // A 2 cm cell under a tire makes |w| enormous. Skipping that kick keeps
    // the sheet velocity finite. A small confinement still applies.
    let du = eps * dt * (-nz * w);
    let dv = eps * dt * (nx * w);
    if du.is_finite() && dv.is_finite() && du * du + dv * dv <= 0.0625 {
        u[idx] += du;
        v[idx] += dv;
    }
}

fn finite_speed(v: f32, lim: f32) -> f32 {
    if !v.is_finite() {
        0.0
    } else {
        v.clamp(-lim, lim)
    }
}

fn finite0(v: f32) -> f32 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

/// Thickness-weighted film velocity. The same blend the render contract publishes.
fn film_uv(water: f32, slurry: f32, uw: f32, vw: f32, us: f32, vs: f32) -> (f32, f32) {
    let h = water + slurry;
    if !h.is_finite() || h < 1.0e-5 {
        return (0.0, 0.0);
    }
    (
        finite0((uw * water + us * slurry) / h),
        finite0((vw * water + vs * slurry) / h),
    )
}

fn vorticity_at(u: &[f32], v: &[f32], res: usize, i: usize, j: usize, cell: f32) -> f32 {
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let dvdx = (v[j * res + ip] - v[j * res + im]) / (2.0 * cell);
    let dudz = (u[jp * res + i] - u[jm * res + i]) / (2.0 * cell);
    dvdx - dudz
}

fn grad(p: &[f32], res: usize, i: usize, j: usize, cell: f32) -> (f32, f32) {
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let dx = (p[j * res + ip] - p[j * res + im]) / (2.0 * cell);
    let dz = (p[jp * res + i] - p[jm * res + i]) / (2.0 * cell);
    (dx, dz)
}

fn grad_vel(u: &[f32], v: &[f32], res: usize, i: usize, j: usize, cell: f32) -> (f32, f32) {
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let dux = (u[j * res + ip] - u[j * res + im]) / (2.0 * cell);
    let dvz = (v[jp * res + i] - v[jm * res + i]) / (2.0 * cell);
    (dux, dvz)
}

fn lap(p: &[f32], res: usize, i: usize, j: usize, cell: f32) -> f32 {
    let idx = j * res + i;
    let ip = (i + 1).min(res - 1);
    let im = i.saturating_sub(1);
    let jp = (j + 1).min(res - 1);
    let jm = j.saturating_sub(1);
    let s = p[j * res + ip] + p[j * res + im] + p[jp * res + i] + p[jm * res + i] - 4.0 * p[idx];
    s / (cell * cell)
}

fn blur_box(
    src: &[f32],
    dst: &mut [f32],
    tmp: &mut [f32],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k: &[f32; 5],
) {
    if i0 > i1 || j0 > j1 || res == 0 {
        return;
    }
    // A tire-sized box is slower with a rayon scope than on one thread.
    // The full window still splits.
    #[cfg(not(target_arch = "wasm32"))]
    if j1 - j0 >= 47 && i1 - i0 >= 31 && res != FINE_RES && res > SERIAL_SHEET_MAX && inner_par() {
        par_blur_axis(src, tmp, res, i0, i1, j0, j1, k, true);
        par_blur_axis(tmp, dst, res, i0, i1, j0, j1, k, false);
        return;
    }
    for j in j0..=j1 {
        blur_axis_x(src, tmp, res, j, i0, i1, k);
    }
    for j in j0..=j1 {
        blur_axis_z(tmp, dst, res, j, i0, i1, k);
    }
}

/// Raw pointer shared by row or column workers. Each worker writes one stripe.
/// The scope joins before the buffer is freed.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Copy, Clone)]
struct ShareMut(*mut f32);
// SAFETY: workers share this pointer and write the same allocation. That is
// sound only while their stripes do not overlap. Debug builds check each
// stripe against the previous one with `ShareMut::debug_assert_stripe` before
// the spawn. The rayon scope joins before the buffer is dropped.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for ShareMut {}
// SAFETY: same disjoint-stripe invariant as `Send`.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for ShareMut {}
#[cfg(not(target_arch = "wasm32"))]
impl ShareMut {
    /// # Safety
    ///
    /// `n` is the length of the live allocation. The slice aliases every other
    /// slice from this pointer; the caller writes only its own stripe.
    unsafe fn slice<'a>(self, n: usize) -> &'a mut [f32] {
        // SAFETY: `self.0` covers `n` elements and outlives `'a`.
        unsafe { std::slice::from_raw_parts_mut(self.0, n) }
    }

    /// `[start, end_excl)` does not overlap the stripe that already ended at
    /// `prev_excl`. Touching bounds (`prev_excl == start`) are disjoint.
    #[cfg(debug_assertions)]
    fn debug_assert_stripe(prev_excl: &mut usize, start: usize, end_excl: usize) {
        debug_assert!(
            *prev_excl <= start,
            "ShareMut thread stripes overlap"
        );
        debug_assert!(start <= end_excl, "ShareMut stripe is inverted");
        *prev_excl = end_excl;
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Copy, Clone)]
struct ShareConst(*const f32);
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for ShareConst {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for ShareConst {}
#[cfg(not(target_arch = "wasm32"))]
impl ShareConst {
    unsafe fn slice<'a>(self, n: usize) -> &'a [f32] {
        unsafe { std::slice::from_raw_parts(self.0, n) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Copy, Clone)]
struct ShareU8(*const u8);
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for ShareU8 {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for ShareU8 {}
#[cfg(not(target_arch = "wasm32"))]
impl ShareU8 {
    unsafe fn slice<'a>(self, n: usize) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.0, n) }
    }
}

/// Each row is written by one worker. `blur_axis_*` only stores `j * res + i`.
#[cfg(not(target_arch = "wasm32"))]
fn par_blur_axis(
    src: &[f32],
    dst: &mut [f32],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    k: &[f32; 5],
    axis_x: bool,
) {
    let ptr = ShareMut(dst.as_mut_ptr());
    let len = dst.len();
    let threads = rayon::current_num_threads().clamp(1, 4);
    let rows = j1 + 1 - j0;
    let chunk = rows.div_ceil(threads);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = j0 + t * chunk;
            if a > j1 {
                break;
            }
            let b = (a + chunk - 1).min(j1);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
            s.spawn(move |_| {
                let dst = unsafe { ptr.slice(len) };
                for j in a..=b {
                    if axis_x {
                        blur_axis_x(src, dst, res, j, i0, i1, k);
                    } else {
                        blur_axis_z(src, dst, res, j, i0, i1, k);
                    }
                }
            });
        }
    });
}

/// Five taps, left to right. A leading zero add matches the clamped loop (`a = 0; a += …`).
fn accum5(a0: f32, a1: f32, a2: f32, a3: f32, a4: f32, k: &[f32; 5]) -> f32 {
    let mut a = a0 * k[0];
    a += a1 * k[1];
    a += a2 * k[2];
    a += a3 * k[3];
    a += a4 * k[4];
    a
}

fn accum5_clamped(row: &[f32], i: usize, k: &[f32; 5]) -> f32 {
    let r = row.len() as i32;
    let mut a = 0.0f32;
    for t in -2i32..=2 {
        let ii = (i as i32 + t).clamp(0, r - 1) as usize;
        a += row[ii] * k[(t + 2) as usize];
    }
    a
}

fn blur_axis_x(src: &[f32], tmp: &mut [f32], res: usize, j: usize, i0: usize, i1: usize, k: &[f32; 5]) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_cached() {
            unsafe { blur_axis_x_avx2(src, tmp, res, j, i0, i1, k) }
            return;
        }
    }
    let row = &src[j * res..(j + 1) * res];
    let dst = &mut tmp[j * res..(j + 1) * res];
    let lo = i0.max(2);
    let hi = i1.min(res.saturating_sub(3));
    for i in i0..lo.min(i1 + 1) {
        dst[i] = accum5_clamped(row, i, k);
    }
    if lo <= hi {
        for i in lo..=hi {
            dst[i] = accum5(row[i - 2], row[i - 1], row[i], row[i + 1], row[i + 2], k);
        }
    }
    let tail = hi.saturating_add(1).max(lo);
    if tail <= i1 {
        for i in tail..=i1 {
            dst[i] = accum5_clamped(row, i, k);
        }
    }
}

fn blur_axis_z(src: &[f32], dst: &mut [f32], res: usize, j: usize, i0: usize, i1: usize, k: &[f32; 5]) {
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_cached() {
            unsafe { blur_axis_z_avx2(src, dst, res, j, i0, i1, k) }
            return;
        }
    }
    let r = res as i32;
    let lo_j = 2usize;
    let hi_j = res.saturating_sub(3);
    if j >= lo_j && j <= hi_j {
        let r0 = (j - 2) * res;
        let r1 = (j - 1) * res;
        let r2 = j * res;
        let r3 = (j + 1) * res;
        let r4 = (j + 2) * res;
        for i in i0..=i1 {
            dst[r2 + i] = accum5(src[r0 + i], src[r1 + i], src[r2 + i], src[r3 + i], src[r4 + i], k);
        }
    } else {
        for i in i0..=i1 {
            let mut a = 0.0f32;
            for t in -2i32..=2 {
                let jj = (j as i32 + t).clamp(0, r - 1) as usize;
                a += src[jj * res + i] * k[(t + 2) as usize];
            }
            dst[j * res + i] = a;
        }
    }
}

fn avx2_cached() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        use std::sync::OnceLock;
        static OK: OnceLock<bool> = OnceLock::new();
        *OK.get_or_init(|| std::is_x86_feature_detected!("avx2"))
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blur_axis_x_avx2(
    src: &[f32],
    tmp: &mut [f32],
    res: usize,
    j: usize,
    i0: usize,
    i1: usize,
    k: &[f32; 5],
) {
    use std::arch::x86_64::*;
    let row = j * res;
    let k0 = _mm256_set1_ps(k[0]);
    let k1 = _mm256_set1_ps(k[1]);
    let k2 = _mm256_set1_ps(k[2]);
    let k3 = _mm256_set1_ps(k[3]);
    let k4 = _mm256_set1_ps(k[4]);
    let lo = i0.max(2);
    let hi = i1.min(res.saturating_sub(3));
    let mut i = i0;
    while i < lo && i <= i1 {
        tmp[row + i] = accum5_clamped(&src[row..row + res], i, k);
        i += 1;
    }
    if lo <= hi {
        i = lo;
        let src_p = src.as_ptr().add(row);
        let dst_p = tmp.as_mut_ptr().add(row);
        while i + 7 <= hi {
            let s0 = _mm256_loadu_ps(src_p.add(i - 2));
            let s1 = _mm256_loadu_ps(src_p.add(i - 1));
            let s2 = _mm256_loadu_ps(src_p.add(i));
            let s3 = _mm256_loadu_ps(src_p.add(i + 1));
            let s4 = _mm256_loadu_ps(src_p.add(i + 2));
            let mut a = _mm256_mul_ps(s0, k0);
            a = _mm256_add_ps(a, _mm256_mul_ps(s1, k1));
            a = _mm256_add_ps(a, _mm256_mul_ps(s2, k2));
            a = _mm256_add_ps(a, _mm256_mul_ps(s3, k3));
            a = _mm256_add_ps(a, _mm256_mul_ps(s4, k4));
            _mm256_storeu_ps(dst_p.add(i), a);
            i += 8;
        }
        while i <= hi {
            tmp[row + i] = accum5(
                src[row + i - 2],
                src[row + i - 1],
                src[row + i],
                src[row + i + 1],
                src[row + i + 2],
                k,
            );
            i += 1;
        }
    }
    let tail = if lo <= hi { hi + 1 } else { i0 };
    let mut i = tail.max(i0);
    while i <= i1 {
        tmp[row + i] = accum5_clamped(&src[row..row + res], i, k);
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blur_axis_z_avx2(
    src: &[f32],
    dst: &mut [f32],
    res: usize,
    j: usize,
    i0: usize,
    i1: usize,
    k: &[f32; 5],
) {
    use std::arch::x86_64::*;
    let r = res as i32;
    if j >= 2 && j + 2 < res {
        let k0 = _mm256_set1_ps(k[0]);
        let k1 = _mm256_set1_ps(k[1]);
        let k2 = _mm256_set1_ps(k[2]);
        let k3 = _mm256_set1_ps(k[3]);
        let k4 = _mm256_set1_ps(k[4]);
        let p0 = src.as_ptr().add((j - 2) * res);
        let p1 = src.as_ptr().add((j - 1) * res);
        let p2 = src.as_ptr().add(j * res);
        let p3 = src.as_ptr().add((j + 1) * res);
        let p4 = src.as_ptr().add((j + 2) * res);
        let out = dst.as_mut_ptr().add(j * res);
        let mut i = i0;
        while i + 7 <= i1 {
            let mut a = _mm256_mul_ps(_mm256_loadu_ps(p0.add(i)), k0);
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p1.add(i)), k1));
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p2.add(i)), k2));
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p3.add(i)), k3));
            a = _mm256_add_ps(a, _mm256_mul_ps(_mm256_loadu_ps(p4.add(i)), k4));
            _mm256_storeu_ps(out.add(i), a);
            i += 8;
        }
        while i <= i1 {
            dst[j * res + i] = accum5(
                src[(j - 2) * res + i],
                src[(j - 1) * res + i],
                src[j * res + i],
                src[(j + 1) * res + i],
                src[(j + 2) * res + i],
                k,
            );
            i += 1;
        }
    } else {
        for i in i0..=i1 {
            let mut a = 0.0f32;
            for t in -2i32..=2 {
                let jj = (j as i32 + t).clamp(0, r - 1) as usize;
                a += src[jj * res + i] * k[(t + 2) as usize];
            }
            dst[j * res + i] = a;
        }
    }
}

fn transport(
    h: &mut [f32],
    u: &[f32],
    v: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    solid: Option<&[u8]>,
    bed: Option<&[f32]>,
    sc: &mut Scratch,
) {
    #[cfg(not(target_arch = "wasm32"))]
    if i1 - i0 >= 31 && j1 - j0 >= 47 && inner_par() {
        let speed = if along_x { u } else { v };
        transport_par(
            h, speed, gate, res, cell, dt, i0, i1, j0, j1, along_x, solid, bed, sc,
        );
        return;
    }
    // Every edge below has both cells inside the box, so the books only
    // need the box, not the whole grid.
    for j in j0..=j1 {
        let row = j * res;
        sc.removed[row + i0..=row + i1].fill(0.0);
        sc.added[row + i0..=row + i1].fill(0.0);
    }
    if along_x {
        for j in j0..=j1 {
            for i in i0..i1 {
                let a = j * res + i;
                let b = a + 1;
                move_edge(
                    h,
                    u,
                    gate,
                    a,
                    b,
                    0.5 * (u[a] + u[b]),
                    cell,
                    dt,
                    solid,
                    bed,
                    sc,
                );
            }
        }
    } else {
        for j in j0..j1 {
            for i in i0..=i1 {
                let a = j * res + i;
                let b = a + res;
                move_edge(
                    h,
                    v,
                    gate,
                    a,
                    b,
                    0.5 * (v[a] + v[b]),
                    cell,
                    dt,
                    solid,
                    bed,
                    sc,
                );
            }
        }
    }
    for j in j0..=j1 {
        for idx in j * res + i0..=j * res + i1 {
            if sc.removed[idx] > 0.0 || sc.added[idx] > 0.0 {
                h[idx] = (h[idx] - sc.removed[idx] + sc.added[idx]).max(0.0);
            }
        }
    }
}

/// Same fluxes as [`transport`], split across rows (x) or columns (z).
/// A row does not share a written index with another row on a horizontal edge.
#[cfg(not(target_arch = "wasm32"))]
fn transport_par(
    h: &mut [f32],
    speed: &[f32],
    gate: &[u8],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    along_x: bool,
    solid: Option<&[u8]>,
    bed: Option<&[f32]>,
    sc: &mut Scratch,
) {
    let removed = ShareMut(sc.removed.as_mut_ptr());
    let added = ShareMut(sc.added.as_mut_ptr());
    let n = res * res;
    let depth = ShareConst(h.as_ptr());
    let hn = h.len();
    let sp = ShareConst(speed.as_ptr());
    let sn = speed.len();
    let gate_p = ShareU8(gate.as_ptr());
    let gn = gate.len();
    let solid_p = solid.map(|s| ShareU8(s.as_ptr()));
    let solid_n = solid.map(|s| s.len()).unwrap_or(0);
    let bed_p = bed.map(|s| ShareConst(s.as_ptr()));
    let bed_n = bed.map(|s| s.len()).unwrap_or(0);
    let threads = rayon::current_num_threads().clamp(1, 4);
    let chunk_n = n.div_ceil(threads);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = t * chunk_n;
            if a >= n {
                break;
            }
            let b = (a + chunk_n).min(n);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b);
            s.spawn(move |_| {
                let removed = unsafe { removed.slice(n) };
                let added = unsafe { added.slice(n) };
                for idx in a..b {
                    removed[idx] = 0.0;
                    added[idx] = 0.0;
                }
            });
        }
    });
    if along_x {
        let rows = j1 + 1 - j0;
        let chunk = rows.div_ceil(threads);
        rayon::scope(|s| {
            #[cfg(debug_assertions)]
            let mut stripe = 0usize;
            for t in 0..threads {
                let a = j0 + t * chunk;
                if a > j1 {
                    break;
                }
                let b = (a + chunk - 1).min(j1);
                #[cfg(debug_assertions)]
                ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
                s.spawn(move |_| {
                    let h = unsafe { depth.slice(hn) };
                    let speed = unsafe { sp.slice(sn) };
                    let gate = unsafe { gate_p.slice(gn) };
                    let solid = solid_p.map(|p| unsafe { p.slice(solid_n) });
                    let bed = bed_p.map(|p| unsafe { p.slice(bed_n) });
                    let removed = unsafe { removed.slice(n) };
                    let added = unsafe { added.slice(n) };
                    for j in a..=b {
                        let row = j * res;
                        for i in i0..i1 {
                            let ia = row + i;
                            let ib = ia + 1;
                            move_edge_bufs(
                                h,
                                gate,
                                ia,
                                ib,
                                0.5 * (speed[ia] + speed[ib]),
                                cell,
                                dt,
                                solid,
                                bed,
                                removed,
                                added,
                            );
                        }
                    }
                });
            }
        });
    } else {
        let cols = i1 + 1 - i0;
        let cthreads = threads.min(cols.max(1));
        let cchunk = cols.div_ceil(cthreads);
        rayon::scope(|s| {
            #[cfg(debug_assertions)]
            let mut stripe = 0usize;
            for t in 0..cthreads {
                let a = i0 + t * cchunk;
                if a > i1 {
                    break;
                }
                let b = (a + cchunk - 1).min(i1);
                #[cfg(debug_assertions)]
                ShareMut::debug_assert_stripe(&mut stripe, a, b + 1);
                s.spawn(move |_| {
                    let h = unsafe { depth.slice(hn) };
                    let speed = unsafe { sp.slice(sn) };
                    let gate = unsafe { gate_p.slice(gn) };
                    let solid = solid_p.map(|p| unsafe { p.slice(solid_n) });
                    let bed = bed_p.map(|p| unsafe { p.slice(bed_n) });
                    let removed = unsafe { removed.slice(n) };
                    let added = unsafe { added.slice(n) };
                    for j in j0..j1 {
                        for i in a..=b {
                            let ia = j * res + i;
                            let ib = ia + res;
                            move_edge_bufs(
                                h,
                                gate,
                                ia,
                                ib,
                                0.5 * (speed[ia] + speed[ib]),
                                cell,
                                dt,
                                solid,
                                bed,
                                removed,
                                added,
                            );
                        }
                    }
                });
            }
        });
    }
    let hp = ShareMut(h.as_mut_ptr());
    let removed_r = ShareConst(removed.0);
    let added_r = ShareConst(added.0);
    rayon::scope(|s| {
        #[cfg(debug_assertions)]
        let mut stripe = 0usize;
        for t in 0..threads {
            let a = t * chunk_n;
            if a >= n {
                break;
            }
            let b = (a + chunk_n).min(n);
            #[cfg(debug_assertions)]
            ShareMut::debug_assert_stripe(&mut stripe, a, b);
            s.spawn(move |_| {
                let h = unsafe { hp.slice(hn) };
                let removed = unsafe { removed_r.slice(n) };
                let added = unsafe { added_r.slice(n) };
                for idx in a..b {
                    if removed[idx] > 0.0 || added[idx] > 0.0 {
                        h[idx] = (h[idx] - removed[idx] + added[idx]).max(0.0);
                    }
                }
            });
        }
    });
}

fn move_edge(
    h: &[f32],
    _speed: &[f32],
    gate: &[u8],
    a: usize,
    b: usize,
    vel: f32,
    cell: f32,
    dt: f32,
    solid: Option<&[u8]>,
    bed: Option<&[f32]>,
    sc: &mut Scratch,
) {
    move_edge_bufs(
        h,
        gate,
        a,
        b,
        vel,
        cell,
        dt,
        solid,
        bed,
        &mut sc.removed,
        &mut sc.added,
    );
}

fn move_edge_bufs(
    h: &[f32],
    gate: &[u8],
    a: usize,
    b: usize,
    vel: f32,
    cell: f32,
    dt: f32,
    solid: Option<&[u8]>,
    bed: Option<&[f32]>,
    removed: &mut [f32],
    added: &mut [f32],
) {
    if let Some(solid) = solid {
        if solid[a] != 0 || solid[b] != 0 {
            return;
        }
    }
    if vel.abs() < 1.0e-5 {
        return;
    }
    let (donor, recv) = if vel > 0.0 { (a, b) } else { (b, a) };
    // A dry bank above the donor's free surface does not take water.
    // Only the solid path passes a bed; a bare sheet spreads as before.
    if let Some(bed) = bed {
        if h[recv] <= 1.0e-6 && bed[recv] > bed[donor] + h[donor] + 1.0e-3 {
            return;
        }
    }
    let room = h[donor] - removed[donor];
    if room <= 1.0e-8 {
        return;
    }
    if gate[recv] == 0 && h[recv] <= 1.0e-6 {
        return;
    }
    let frac = (vel.abs() * dt / cell).min(0.45);
    let dh = (h[donor] * frac).min(room);
    if dh <= 1.0e-8 {
        return;
    }
    removed[donor] += dh;
    added[recv] += dh;
}

/// Free-surface stand-in for a wall. Each solid cell takes the mean of the
/// wet cells within two cells, so the body is not a hole in η.
fn lift_solid_surface(
    eta: &mut [f32],
    h: &[f32],
    solid: &[u8],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    eps: f32,
    tmp: &mut [f32],
) {
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] == 0 {
                continue;
            }
            let mut acc = 0.0f32;
            let mut n = 0.0f32;
            for dj in -2i32..=2 {
                for di in -2i32..=2 {
                    if di == 0 && dj == 0 {
                        continue;
                    }
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                        continue;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    if solid[nidx] != 0 || h[nidx] <= eps {
                        continue;
                    }
                    acc += eta[nidx];
                    n += 1.0;
                }
            }
            tmp[idx] = if n > 0.0 { acc / n } else { eta[idx] };
        }
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] != 0 {
                eta[idx] = tmp[idx];
            }
        }
    }
}

/// No-slip ghost: the wall velocity is the negation of the neighbouring fluid.
/// Vorticity on the fluid side is then the shear of that wall, not a drawn vortex.
fn write_noslip_ghost(
    u: &mut [f32],
    v: &mut [f32],
    h: &[f32],
    solid: &[u8],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    eps: f32,
) {
    let mut ghosts = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] == 0 {
                continue;
            }
            let mut au = 0.0f32;
            let mut av = 0.0f32;
            let mut n = 0.0f32;
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] != 0 || h[nidx] <= eps {
                    continue;
                }
                au += u[nidx];
                av += v[nidx];
                n += 1.0;
            }
            if n > 0.0 {
                ghosts.push((idx, -au / n, -av / n));
            } else {
                ghosts.push((idx, 0.0, 0.0));
            }
        }
    }
    for (idx, gu, gv) in ghosts {
        u[idx] = gu;
        v[idx] = gv;
    }
}

/// Remove the into-wall part of the fluid velocity. The tangential part stays.
fn block_into_solid(
    u: &mut [f32],
    v: &mut [f32],
    h: &[f32],
    solid: &[u8],
    res: usize,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    eps: f32,
) {
    let mut next = Vec::new();
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] != 0 || h[idx] <= eps {
                continue;
            }
            let mut ux = u[idx];
            let mut vx = v[idx];
            let mut hit = false;
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] == 0 {
                    continue;
                }
                let nx = di as f32;
                let nz = dj as f32;
                let vn = ux * nx + vx * nz;
                if vn > 0.0 {
                    ux -= vn * nx;
                    vx -= vn * nz;
                    hit = true;
                }
            }
            if hit {
                next.push((idx, ux, vx));
            }
        }
    }
    for (idx, ux, vx) in next {
        u[idx] = ux;
        v[idx] = vx;
    }
}

/// Push water (or slurry) out of cells the body now occupies. The volume is
/// split across the open neighbours. A cell with nowhere to go keeps its column.
fn displace_occupied(h: &mut [f32], u: &mut [f32], v: &mut [f32], solid: &[u8], res: usize) {
    let n = res * res;
    let mut give = vec![0.0f32; n];
    for j in 0..res {
        for i in 0..res {
            let idx = j * res + i;
            if solid[idx] == 0 {
                continue;
            }
            let vol = h[idx];
            if vol <= 1.0e-8 {
                h[idx] = 0.0;
                u[idx] = 0.0;
                v[idx] = 0.0;
                continue;
            }
            let mut neigh = Vec::with_capacity(4);
            for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] == 0 {
                    neigh.push(nidx);
                }
            }
            if neigh.is_empty() {
                for dj in -2i32..=2 {
                    for di in -2i32..=2 {
                        let ni = i as i32 + di;
                        let nj = j as i32 + dj;
                        if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                            continue;
                        }
                        let nidx = nj as usize * res + ni as usize;
                        if solid[nidx] == 0 {
                            neigh.push(nidx);
                        }
                    }
                }
            }
            if neigh.is_empty() {
                continue;
            }
            h[idx] = 0.0;
            u[idx] = 0.0;
            v[idx] = 0.0;
            let share = vol / neigh.len() as f32;
            for nidx in neigh {
                give[nidx] += share;
            }
        }
    }
    for idx in 0..n {
        if give[idx] > 0.0 {
            h[idx] += give[idx];
        }
    }
}

/// Move the discharge that points into a wall out to the open flanks.
/// The wall stays dry. The volume stays on the sheet.
fn redirect_into_walls(
    h: &mut [f32],
    u: &[f32],
    v: &[f32],
    solid: &[u8],
    bed: &[f32],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    sc: &mut Scratch,
) {
    // Removals sit in the box and additions on its 4-neighbours, so the
    // books cover the box and a 1-cell rim.
    let ri0 = i0.saturating_sub(1);
    let ri1 = (i1 + 1).min(res - 1);
    let rj0 = j0.saturating_sub(1);
    let rj1 = (j1 + 1).min(res - 1);
    for j in rj0..=rj1 {
        let row = j * res;
        sc.removed[row + ri0..=row + ri1].fill(0.0);
        sc.added[row + ri0..=row + ri1].fill(0.0);
    }
    for j in j0..=j1 {
        for i in i0..=i1 {
            let idx = j * res + i;
            if solid[idx] != 0 || h[idx] < 0.02 {
                continue;
            }
            for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                if solid[nidx] == 0 {
                    continue;
                }
                // Own speed into the wall, or the stream feeding this cell if the
                // wall has already stopped it. Either way the discharge turns aside.
                let mut vn = u[idx] * di as f32 + v[idx] * dj as f32;
                let ui = i as i32 - di;
                let uj = j as i32 - dj;
                if ui >= 0 && uj >= 0 && ui < res as i32 && uj < res as i32 {
                    let uidx = uj as usize * res + ui as usize;
                    if solid[uidx] == 0 {
                        let feed = u[uidx] * di as f32 + v[uidx] * dj as f32;
                        if feed > vn {
                            vn = feed;
                        }
                    }
                }
                if vn < 0.05 {
                    continue;
                }
                let perps: [(i32, i32); 2] = if di != 0 {
                    [(0, -1), (0, 1)]
                } else {
                    [(-1, 0), (1, 0)]
                };
                let mut dest = [0usize; 2];
                let mut n_dest = 0usize;
                let mut flank = 0.0f32;
                for (pi, pj) in perps {
                    let qi = i as i32 + pi;
                    let qj = j as i32 + pj;
                    if qi < 0 || qj < 0 || qi >= res as i32 || qj >= res as i32 {
                        continue;
                    }
                    let q = qj as usize * res + qi as usize;
                    if solid[q] != 0 {
                        continue;
                    }
                    if h[q] <= 1.0e-6 && bed[q] > bed[idx] + h[idx] + 1.0e-3 {
                        continue;
                    }
                    dest[n_dest] = q;
                    flank += h[q];
                    n_dest += 1;
                }
                if n_dest == 0 {
                    continue;
                }
                // Only the water standing above the path around the body turns
                // aside. The rest is the bow.
                let excess = h[idx] - flank / n_dest as f32;
                if excess < 2.0e-3 {
                    continue;
                }
                let room = h[idx] - sc.removed[idx];
                if room < 1.0e-6 {
                    continue;
                }
                let frac = (vn * dt / cell).min(0.45);
                let dh = excess.min(h[idx] * frac).min(room);
                if dh < 1.0e-8 {
                    continue;
                }
                sc.removed[idx] += dh;
                let share = dh / n_dest as f32;
                for q in dest.iter().take(n_dest) {
                    sc.added[*q] += share;
                }
            }
        }
    }
    for j in rj0..=rj1 {
        for idx in j * res + ri0..=j * res + ri1 {
            if sc.removed[idx] > 0.0 || sc.added[idx] > 0.0 {
                h[idx] = (h[idx] - sc.removed[idx] + sc.added[idx]).max(0.0);
            }
        }
    }
}

fn solid_near(mask: &[u8], res: usize, i: usize, j: usize, radius: i32) -> bool {
    for dj in -radius..=radius {
        for di in -radius..=radius {
            if di == 0 && dj == 0 {
                continue;
            }
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                continue;
            }
            if mask[nj as usize * res + ni as usize] != 0 {
                return true;
            }
        }
    }
    false
}

fn solid_contains(solid: &HydroSolid, x: f32, z: f32) -> bool {
    let dx = x - solid.x;
    let dz = z - solid.z;
    let fwd_x = solid.yaw.sin();
    let fwd_z = solid.yaw.cos();
    let right_x = fwd_z;
    let right_z = -fwd_x;
    let along = dx * fwd_x + dz * fwd_z;
    let lat = dx * right_x + dz * right_z;
    along.abs() <= solid.half_length.max(0.0) && lat.abs() <= solid.half_width.max(0.0)
}

fn solid_overlaps(solid: &HydroSolid, x: f32, z: f32, eta: f32, bed: f32) -> bool {
    if solid.half_length <= 1.0e-4 || solid.half_width <= 1.0e-4 {
        return false;
    }
    if !solid_contains(solid, x, z) {
        return false;
    }
    eta > solid.bottom_y + 1.0e-4 && bed < solid.top_y
}

fn cylinder_segment_volume(depth: f32, radius: f32, width: f32) -> f32 {
    let d = depth.clamp(0.0, radius * 2.0);
    if d <= 1.0e-4 || width <= 0.0 || radius <= 1.0e-3 {
        return 0.0;
    }
    let c = ((radius - d) / radius).clamp(-1.0, 1.0);
    let area =
        radius * radius * c.acos() - (radius - d) * (2.0 * radius * d - d * d).max(0.0).sqrt();
    area.max(0.0) * width
}

fn column_sum(h: &[f32], cell: f32) -> f32 {
    let area = cell * cell;
    h.iter().map(|s| s.max(0.0)).sum::<f32>() * area
}

fn restore_wiped(h: &mut [f32], before: f32, cell: f32) {
    let lost = before - column_sum(h, cell);
    if lost <= 1.0e-8 {
        return;
    }
    let Some(idx) = h
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
    else {
        return;
    };
    h[idx] += lost / (cell * cell);
}

fn hash01(ix: i32, iz: i32, seed: u32) -> f32 {
    let mut n = (ix as u32)
        .wrapping_mul(0x27BB_2EE6)
        .wrapping_add((iz as u32).wrapping_mul(0x1656_67B1))
        ^ seed;
    n = n.wrapping_mul(0x9E37_79B1);
    n ^= n >> 16;
    n = n.wrapping_mul(0x85EB_CA6B);
    n ^= n >> 13;
    (n >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Value noise in −1..1. The puddle relief samples this around the shore.
pub fn relief_value(x: f32, z: f32, seed: u32) -> f32 {
    let x0 = x.floor();
    let z0 = z.floor();
    let fx = x - x0;
    let fz = z - z0;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sz = fz * fz * (3.0 - 2.0 * fz);
    let ix = x0 as i32;
    let iz = z0 as i32;
    let v00 = hash01(ix, iz, seed);
    let v10 = hash01(ix + 1, iz, seed);
    let v01 = hash01(ix, iz + 1, seed);
    let v11 = hash01(ix + 1, iz + 1, seed);
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    (a + (b - a) * sz) * 2.0 - 1.0
}

/// A rounded pool. `level_s` is when the relief has fallen under 28% of the
/// start. Clay and silt stay deep through 8 s and level between 10 and 15 s.
/// A thin clay mix levels between 1 and 3 s.
pub fn rut_heal(grain: Grain, moisture: f32) -> RutHeal {
    let res = 48usize;
    let cell = 0.06f32;
    let n = res * res;
    let mut h = vec![0.0f32; n];
    let mut u = vec![0.0f32; n];
    let mut v = vec![0.0f32; n];
    let mut bed = vec![0.0f32; n];
    let nu_v = dynamic_viscosity(grain, moisture);
    let nu = vec![nu_v; n];
    let gate = vec![1u8; n];
    let c = (res / 2) as f32;
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5 - c) * cell;
            let z = (j as f32 + 0.5 - c) * cell;
            let r = (x * x + z * z).sqrt();
            let idx = j * res + i;
            let bowl = (-(r * r) / (2.0 * 0.42 * 0.42)).exp();
            bed[idx] = -0.07 * bowl;
            h[idx] = 0.012 + 0.028 * (1.0 - bowl);
        }
    }
    let _ = nu_v;
    let relief0 = relief(&h, &bed, res, cell);
    let mut level_s = 20.0f32;
    let mut relief_8s = relief0;
    let mut t = 0.0f32;
    let dt = 0.05f32;
    while t < 20.0 - 1.0e-4 {
        step_sheet(
            &mut h,
            &mut u,
            &mut v,
            &bed,
            &nu,
            &gate,
            res,
            cell,
            dt,
            SheetDrive::Gravity { vorticity: 0.0 },
        );
        t += dt;
        let r = relief(&h, &bed, res, cell);
        if (t - 8.0).abs() <= dt * 0.51 {
            relief_8s = r;
        }
        if level_s > 19.0 && r < relief0 * 0.28 {
            level_s = t;
        }
        if t >= 8.0 && level_s < 19.0 {
            break;
        }
    }
    RutHeal {
        level_s,
        relief0,
        relief_8s,
        still_deep_at_8s: relief_8s > relief0 * 0.38,
    }
}

fn relief(h: &[f32], bed: &[f32], res: usize, cell: f32) -> f32 {
    let c = (res / 2) as f32;
    let mut center = 0.0f32;
    let mut cn = 0.0f32;
    let mut ring = 0.0f32;
    let mut rn = 0.0f32;
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5 - c) * cell;
            let z = (j as f32 + 0.5 - c) * cell;
            let r = (x * x + z * z).sqrt();
            let eta = bed[j * res + i] + h[j * res + i];
            if r < 0.12 {
                center += eta;
                cn += 1.0;
            } else if (0.48..0.68).contains(&r) {
                ring += eta;
                rn += 1.0;
            }
        }
    }
    if cn < 1.0 || rn < 1.0 {
        return 0.0;
    }
    (ring / rn) - (center / cn)
}

#[derive(Clone, Copy, Debug, Default)]
struct Parked {
    water: f32,
    slurry: f32,
    uw: f32,
    vw: f32,
    us: f32,
    vs: f32,
    dust: f32,
    /// Pore moisture, 0..1. Stays with the ground cell when the window moves,
    /// so a wet tire track does not slide with the Niva.
    moisture: f32,
    /// Foam fraction that left the window with this cell.
    foam: f32,
    /// Metres of soil held in the water that left the window with this cell.
    turbid: f32,
    /// Suspended-solids depth in the water column, metres. Not a second sheet.
    susp: f32,
}

/// Sheet times inside one [`FluidCore::step`], milliseconds.
///
/// Water and slurry are the fine sheet plus the coarse sheet. Dust is the
/// column carried in this step. Suspended solids are the plume advection.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepParts {
    pub water_ms: f32,
    pub slurry_ms: f32,
    pub dust_ms: f32,
    pub suspended_ms: f32,
    pub prep_ms: f32,
    pub restrict_ms: f32,
    pub window_ms: f32,
}

fn park_slot(slot: &Parked, ambient: f32) -> bool {
    slot.water > 1.0e-8
        || slot.slurry > 1.0e-8
        || slot.dust > 1.0e-8
        || slot.susp > 1.0e-8
        || (slot.moisture - ambient).abs() > 6.0e-3
        || slot.foam > 1.0e-4
        || slot.turbid > 1.0e-8
}

/// The moving window. Coarse sheet, fine sheet under the tires, dust column.
pub struct FluidCore {
    pub center: [f32; 2],
    origin: [f32; 2],
    air_origin: [f32; 2],
    water: Vec<f32>,
    slurry: Vec<f32>,
    uw: Vec<f32>,
    vw: Vec<f32>,
    us: Vec<f32>,
    vs: Vec<f32>,
    bed: Vec<f32>,
    moisture: Vec<f32>,
    /// Pore moisture of the pad, the value most cells share after a substrate
    /// read. A leaving cell that only holds this pad is not parked: the next
    /// window writes it back. A wet track, which differs, still parks.
    moisture_ambient: f32,
    grain: Vec<u8>,
    settled: Vec<f32>,
    /// Foam fraction, 0 to 1. Born on the tire waterline and the shore, then
    /// carried with the film and allowed to die. A dry cell stays 0.
    foam: Vec<f32>,
    /// Metres of bed soil held in the water. This is the stirred silt. It is
    /// not added to the standing slurry column: that column is the pour, and
    /// this one is only what shear and slip have lifted.
    turbid: Vec<f32>,
    /// Cubic metres of that silt that have left the water and settled.
    turbid_settled_m3: f32,
    /// Cubic metres lifted into `turbid` and `turbid_settled_m3`.
    stirred_m3: f32,
    /// Scratch for the tracer pass. Not a second sheet.
    tracer_scratch: Vec<f32>,
    /// Coarse cells with at least 4 mm of film. Foam is computed only here.
    wet_idx: Vec<u32>,
    /// Wet cells on the film edge. Shore foam visits only these.
    shore_idx: Vec<u32>,
    /// Cells that currently hold foam above the decay floor.
    foam_live: Vec<u32>,
    foam_live_mark: Vec<u8>,
    /// Suspended-solids depth, metres, on the coarse water column.
    susp: Vec<f32>,
    /// Scratch for the suspended advection. Sized once.
    susp_next: Vec<f32>,
    /// Solids a neighbour already accepted this pass.
    susp_taken: Vec<f32>,
    /// Cells this step where advection booked the column down to the water.
    last_susp_capped: u32,
    /// When false, skip wake entrainment and suspended advection. The through-drive
    /// acceptance test keeps this off so the solver budget is not charged twice.
    turbidity_enabled: bool,
    /// Cubic metres lifted into suspension. Not spent twice.
    entrained_m3: f32,
    /// Cubic metres that left the live column and stayed in the account.
    settled_suspended_m3: f32,
    last_parts: StepParts,
    fine_origin: [f32; 2],
    fine_water: Vec<f32>,
    fine_slurry: Vec<f32>,
    fine_uw: Vec<f32>,
    fine_vw: Vec<f32>,
    fine_us: Vec<f32>,
    fine_vs: Vec<f32>,
    fine_bed: Vec<f32>,
    /// The fine bed the slurry sheet reads. It differs from `fine_bed` only on
    /// closed cells, where each sheet meets a wall at its own coarse surface.
    fine_bed_slurry: Vec<f32>,
    fine_moisture: Vec<f32>,
    fine_grain: Vec<u8>,
    fine_on: bool,
    /// 1 when this coarse cell's sheet was moved onto the 2 cm patch.
    /// Survives `restrict_fine` so the coarse shove does not hit it twice.
    fine_hold: Vec<u8>,
    /// Sub-cell detail of the 2 cm sheet, kept between steps (`near.rs`).
    near: near::NearField,
    dust: Vec<f32>,
    dust_u: Vec<f32>,
    dust_v: Vec<f32>,
    /// Fog depth, metres, same index as `dust`. Not a second air grid.
    fog: Vec<f32>,
    fog_u: Vec<f32>,
    fog_v: Vec<f32>,
    /// False while every fog slab is empty, so a step does not scan them.
    fog_live: bool,
    parked: HashMap<(i32, i32), Parked>,
    parked_dust: HashMap<(i32, i32), f32>,
    parked_fog: HashMap<(i32, i32), f32>,
    spray_m3: f32,
    /// Cumulative spray volume, see `spray_thrown_m3`.
    spray_thrown_m3: f32,
    /// Spray that came down outside the window. Still booked as spray.
    spray_stranded_m3: f32,
    /// Spray that landed on dry ground and soaked in.
    spray_soaked_m3: f32,
    /// Drops peeled this pass that are still in the air.
    drops: Vec<LiveSpray>,
    /// Tire shoves from the step that just ran. Foam around the tires reads these.
    last_pushes: Vec<TirePush>,
    pub frame: u32,
    pushes: Vec<TirePush>,
    /// Tires already in the sheet as hydro solids. They lift silt and foam
    /// the surface; they do not displace water or peel spray a second time.
    stirs: Vec<TirePush>,
    wind: [f32; 2],
    /// Wheel and belly walls. Empty means the sheet has no body in it.
    solids: Vec<HydroSolid>,
    /// Metres of bed removed by contact shear, per coarse cell.
    bed_drop: Vec<f32>,
    eroded_m3: f32,
    scour_hole_m: f32,
    cycle: Option<StreamCycle>,
    reach: Option<RiverReach>,
    /// Cubic metres between the outlet and the inlet of the reach.
    reach_bank_m3: f32,
    wave: Option<WaveMaker>,
    /// Seconds since the wave maker was set.
    wave_t: f32,
    soak: Option<ShoreSoak>,
    /// Free-surface diffusion on the coarse water sheet, m²/s. 0 is off.
    surface_diffusion: f32,
    /// Full-gravity water with bed friction for an open reach. `None` keeps
    /// the viscous mobility of the water grain.
    open_water: Option<OpenWater>,
    /// Scratch for [`Self::seat_rut_water`]. Sized once so a step does not allocate.
    seat_pressed: Vec<f32>,
    seat_is_rut: Vec<bool>,
    seat_floor: Vec<bool>,
    seat_etas: Vec<f32>,
    /// CFL substep cap on the 2 cm sheet. Eight is the full split. The coarse
    /// sheet always uses eight.
    fine_sub_cap: f32,
    /// Open 2 cm sheet. 0: one pass. 1: adaptive CFL on awake tiles. 2: the
    /// same split on the whole wet rectangle. 3: one pressure, then the
    /// transports. A wall ignores this and keeps the body sweep.
    fine_pace: u8,
    sheet_times: SheetTimes,
    /// Reused by the integer window scroll. Not a second sheet.
    shift_scratch: Vec<f32>,
    /// Scratch for [`Self::pond_ruts`]. Sized once so a step does not allocate.
    pond_still: Vec<u8>,
    pond_eta: Vec<f32>,
    pond_mark: Vec<f32>,
    pond_count_p: Vec<f32>,
    pond_eta_p: Vec<f32>,
    pond_needs: Vec<(usize, f32)>,
    pond_donors: Vec<(usize, f32)>,
    /// Two scratches per walled sheet: fine water, fine slurry, coarse water,
    /// coarse slurry. Water and slurry step on two workers, and two far-apart
    /// boxes of one sheet may too, so each keeps its own buffers instead of
    /// whichever thread-local the worker holds.
    sheet_scratch: [[Scratch; 2]; 4],
    /// Balls, boxes and legs in the sheet. Empty costs nothing.
    wet: wet::WetState,
}

impl FluidCore {
    pub fn new(center: [f32; 2]) -> Self {
        let n = WINDOW_RES * WINDOW_RES;
        let fine_n = FINE_RES * FINE_RES;
        let air_n = AIR_RES * AIR_RES * AIR_LAYERS;
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        let origin = snap_origin(center, extent, COARSE_CELL_M);
        let air_extent = AIR_RES as f32 * (extent / AIR_RES as f32);
        let air_origin = snap_origin(center, air_extent, extent / AIR_RES as f32);
        Self {
            center,
            origin,
            air_origin,
            water: vec![0.0; n],
            slurry: vec![0.0; n],
            uw: vec![0.0; n],
            vw: vec![0.0; n],
            us: vec![0.0; n],
            vs: vec![0.0; n],
            bed: vec![0.0; n],
            moisture: vec![0.0; n],
            moisture_ambient: 0.0,
            grain: vec![Grain::Other as u8; n],
            settled: vec![0.0; n],
            foam: vec![0.0; n],
            turbid: vec![0.0; n],
            turbid_settled_m3: 0.0,
            stirred_m3: 0.0,
            tracer_scratch: Vec::new(),
            wet_idx: Vec::new(),
            shore_idx: Vec::new(),
            foam_live: Vec::new(),
            foam_live_mark: vec![0; n],
            susp: vec![0.0; n],
            susp_next: vec![0.0; n],
            susp_taken: vec![0.0; n],
            last_susp_capped: 0,
            turbidity_enabled: true,
            entrained_m3: 0.0,
            settled_suspended_m3: 0.0,
            last_parts: StepParts::default(),
            fine_origin: [0.0, 0.0],
            fine_water: vec![0.0; fine_n],
            fine_slurry: vec![0.0; fine_n],
            fine_uw: vec![0.0; fine_n],
            fine_vw: vec![0.0; fine_n],
            fine_us: vec![0.0; fine_n],
            fine_vs: vec![0.0; fine_n],
            fine_bed: vec![0.0; fine_n],
            fine_bed_slurry: vec![0.0; fine_n],
            fine_moisture: vec![0.0; fine_n],
            fine_grain: vec![Grain::Other as u8; fine_n],
            fine_on: false,
            fine_hold: vec![0; n],
            near: near::NearField::default(),
            dust: vec![0.0; air_n],
            dust_u: vec![0.0; air_n],
            dust_v: vec![0.0; air_n],
            fog: vec![0.0; air_n],
            fog_u: vec![0.0; air_n],
            fog_v: vec![0.0; air_n],
            fog_live: false,
            parked: HashMap::with_capacity(8192),
            parked_dust: HashMap::with_capacity(1024),
            parked_fog: HashMap::new(),
            spray_m3: 0.0,
            spray_thrown_m3: 0.0,
            spray_stranded_m3: 0.0,
            spray_soaked_m3: 0.0,
            drops: Vec::new(),
            last_pushes: Vec::new(),
            frame: 0,
            pushes: Vec::new(),
            stirs: Vec::new(),
            wind: [1.1, 0.35],
            solids: Vec::new(),
            bed_drop: vec![0.0; n],
            eroded_m3: 0.0,
            scour_hole_m: 0.0,
            cycle: None,
            reach: None,
            reach_bank_m3: 0.0,
            wave: None,
            wave_t: 0.0,
            soak: None,
            surface_diffusion: 0.0,
            open_water: None,
            seat_pressed: vec![0.0; n],
            seat_is_rut: vec![false; n],
            seat_floor: vec![false; n],
            seat_etas: Vec::with_capacity(n),
            fine_sub_cap: 8.0,
            fine_pace: 0,
            sheet_times: SheetTimes::default(),
            shift_scratch: Vec::new(),
            pond_still: vec![0; n],
            pond_eta: vec![0.0; n],
            pond_mark: vec![0.0; n],
            pond_count_p: vec![0.0; (WINDOW_RES + 1) * (WINDOW_RES + 1)],
            pond_eta_p: vec![0.0; (WINDOW_RES + 1) * (WINDOW_RES + 1)],
            pond_needs: Vec::new(),
            pond_donors: Vec::new(),
            sheet_scratch: std::array::from_fn(|_| [Scratch::new(), Scratch::new()]),
            wet: wet::WetState::default(),
        }
    }

    pub fn reset_sheet_times(&mut self) {
        self.sheet_times = SheetTimes::default();
    }

    pub fn sheet_times(&self) -> SheetTimes {
        self.sheet_times
    }

    /// Cap CFL substeps on the 2 cm sheet. The coarse sheet always splits up
    /// to eight. The default is eight.
    pub fn set_fine_sub_cap(&mut self, cap: f32) {
        self.fine_sub_cap = if cap.is_finite() {
            cap.clamp(1.0, 8.0)
        } else {
            8.0
        };
    }

    pub fn fine_sub_cap(&self) -> f32 {
        self.fine_sub_cap
    }

    /// How the open 2 cm sheet splits a frame. See [`Self::fine_pace`].
    pub fn set_fine_pace(&mut self, pace: u8) {
        self.fine_pace = pace.min(3);
    }

    pub fn fine_pace(&self) -> u8 {
        self.fine_pace
    }

    /// Kept so the plume bench can ask for per-step times. They are always recorded.
    pub fn set_step_parts(&mut self, _on: bool) {}

    /// Turn wake entrainment and suspended advection on or off.
    pub fn set_turbidity_enabled(&mut self, on: bool) {
        self.turbidity_enabled = on;
    }

    /// Times from the last [`Self::step`].
    pub fn step_parts(&self) -> StepParts {
        self.last_parts
    }

    /// Coarse water and slurry are both under a dry film, and the fine patch is off.
    ///
    /// Does not restrict. A live fine patch can still hold water the coarse
    /// grid has not seen, so that case is not dry.
    pub fn coarse_sheet_is_dry(&self) -> bool {
        if self.fine_on {
            return false;
        }
        self.water.iter().all(|h| *h < 1.0e-5) && self.slurry.iter().all(|h| *h < 1.0e-5)
    }

    /// Let the water and slurry sheets pool on the bed for `steps` frames of `dt`.
    ///
    /// Same gravity step as a frame, with no tire and no dust. A dry cell whose
    /// bed stands above the donor's free surface stays dry, so the pour finds
    /// the shoreline of the relief. Returns the water volume left on the coarse
    /// sheet.
    pub fn drain_over_bed(&mut self, dt: f32, steps: u32) -> f32 {
        if !(dt.is_finite() && dt > 0.0) || steps == 0 {
            return column_sum(&self.water, COARSE_CELL_M);
        }
        self.restrict_fine();
        let mut calm = 0u32;
        let mut moved = false;
        for _ in 0..steps {
            let water_before = column_sum(&self.water, COARSE_CELL_M);
            let slurry_before = column_sum(&self.slurry, COARSE_CELL_M);
            self.step_water_drain(dt);
            self.step_slurry_drain(dt);
            // Init only. The gravity step itself is unchanged. The extra
            // friction lets the pour find the bank without sloshing for minutes.
            let keep = (-0.55 * dt).exp();
            for slot in [&mut self.uw, &mut self.vw, &mut self.us, &mut self.vs] {
                for v in slot.iter_mut() {
                    *v *= keep;
                }
            }
            restore_wiped(&mut self.water, water_before, COARSE_CELL_M);
            restore_wiped(&mut self.slurry, slurry_before, COARSE_CELL_M);
            let speed = self.peak_sheet_speed().max(self.peak_slurry_speed());
            if speed > 0.05 {
                moved = true;
            }
            if moved && speed < 0.008 {
                calm += 1;
                if calm >= 24 {
                    break;
                }
            } else {
                calm = 0;
            }
        }
        column_sum(&self.water, COARSE_CELL_M)
    }

    fn step_water_drain(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let n = res * res;
        let nu = vec![dynamic_viscosity(Grain::Water, 1.0); n];
        let gate = vec![1u8; n];
        let bed = self.bed.clone();
        soften_border(&mut self.uw, &mut self.vw, res, None);
        step_sheet_eps(
            &mut self.water,
            &mut self.uw,
            &mut self.vw,
            &bed,
            &nu,
            &gate,
            res,
            COARSE_CELL_M,
            dt,
            SheetDrive::Gravity { vorticity: 0.0 },
            1.0e-6,
            None,
            true,
            8.0,
            2,
        );
    }

    fn step_slurry_drain(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let n = res * res;
        let mut nu = vec![0.0f32; n];
        for idx in 0..n {
            nu[idx] = dynamic_viscosity(Grain::from_u8(self.grain[idx]), self.moisture[idx]);
        }
        let gate = vec![1u8; n];
        let bed = self.bed.clone();
        soften_border(&mut self.us, &mut self.vs, res, None);
        step_sheet_eps(
            &mut self.slurry,
            &mut self.us,
            &mut self.vs,
            &bed,
            &nu,
            &gate,
            res,
            COARSE_CELL_M,
            dt,
            SheetDrive::Gravity { vorticity: 0.0 },
            1.0e-6,
            None,
            true,
            8.0,
            2,
        );
    }

    fn peak_sheet_speed(&self) -> f32 {
        let mut peak = 0.0f32;
        for (h, (u, v)) in self.water.iter().zip(self.uw.iter().zip(self.vw.iter())) {
            if *h > 1.0e-4 {
                peak = peak.max(u.abs()).max(v.abs());
            }
        }
        peak
    }

    fn peak_slurry_speed(&self) -> f32 {
        let mut peak = 0.0f32;
        for (h, (u, v)) in self.slurry.iter().zip(self.us.iter().zip(self.vs.iter())) {
            if *h > 1.0e-4 {
                peak = peak.max(u.abs()).max(v.abs());
            }
        }
        peak
    }

    /// Slurry thickness on cells that already hold water. The drain moves it.
    pub fn lay_slurry_on_water(&mut self, depth: f32, grain: Grain) -> f32 {
        if !(depth.is_finite() && depth > 0.0) {
            return 0.0;
        }
        self.restrict_fine();
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let mut vol = 0.0f32;
        for idx in 0..self.water.len() {
            if self.water[idx] <= 0.012 {
                continue;
            }
            self.slurry[idx] = depth;
            self.grain[idx] = grain as u8;
            self.moisture[idx] = self.moisture[idx].max(0.90);
            vol += depth * area;
        }
        vol
    }

    /// Replace the walls for the next step. An empty slice is open water.
    pub fn set_hydro_solids(&mut self, solids: &[HydroSolid]) {
        self.solids.clear();
        self.solids.extend_from_slice(solids);
    }

    pub fn hydro_solids(&self) -> &[HydroSolid] {
        &self.solids
    }

    /// Open-channel lips. `None` leaves the sheet closed at its own banks.
    pub fn set_stream_cycle(&mut self, cycle: Option<StreamCycle>) {
        self.cycle = cycle;
    }

    /// Open inlet and outlet at the window ends. `None` closes them. Water
    /// still in the bank stays on the ledger.
    pub fn set_river_reach(&mut self, reach: Option<RiverReach>) {
        self.reach = reach;
    }

    /// Free-surface diffusion between wet coarse cells, m²/s. 0 turns it off.
    ///
    /// The collocated height–velocity step leaves a two-cell ripple in the
    /// free surface of a long open reach. This moves water between wet
    /// neighbours down the free-surface difference, donor-limited, so the
    /// column is only moved, never minted. The two-cell mode decays in a few
    /// steps; a swell a few metres long loses a few percent per second.
    pub fn set_surface_diffusion(&mut self, m2_s: f32) {
        self.surface_diffusion = if m2_s.is_finite() { m2_s.max(0.0) } else { 0.0 };
    }

    /// Open-water mode for the water sheet. `None` restores the default,
    /// where water shares the viscous mobility curve of the mud sheets (about
    /// 0.1 of gravity), which is right for a film in a rut and too slow for a
    /// river: a swell then crawls and never runs up a beach. Slurry keeps its
    /// own curve either way. See [`OpenWater`].
    pub fn set_open_water(&mut self, open: Option<OpenWater>) {
        self.open_water = open
            .filter(|o| {
                o.mobility.is_finite() && o.manning_n.is_finite() && o.eddy_m2_s.is_finite()
            })
            .map(|o| OpenWater {
                mobility: o.mobility.clamp(0.0, 1.0),
                manning_n: o.manning_n.max(0.0),
                eddy_m2_s: o.eddy_m2_s.max(0.0),
            });
    }

    /// Paddle strip in the coarse sheet. `None` turns it off. Restarts the paddle clock.
    pub fn set_wave_maker(&mut self, wave: Option<WaveMaker>) {
        self.wave = wave;
        self.wave_t = 0.0;
    }

    /// Pore water left behind by the film. `None` leaves moisture to the substrate read.
    pub fn set_shore_soak(&mut self, soak: Option<ShoreSoak>) {
        self.soak = soak;
    }

    /// Grain of the solids per coarse cell. Viscosity and drainage read it.
    /// Does not move any volume.
    pub fn map_grain(&mut self, mut f: impl FnMut(f32, f32) -> Grain) {
        self.restrict_fine();
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                self.grain[j * WINDOW_RES + i] = f(x, z) as u8;
            }
        }
    }

    /// Initial bed and water thickness. Returns the water volume now on the sheet.
    pub fn map_water_bed(&mut self, mut f: impl FnMut(f32, f32) -> (f32, f32)) -> f32 {
        self.restrict_fine();
        let mut vol = 0.0f32;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                let (bed, water) = f(x, z);
                let idx = j * WINDOW_RES + i;
                self.bed[idx] = if bed.is_finite() { bed } else { 0.0 };
                let w = if water.is_finite() {
                    water.max(0.0)
                } else {
                    0.0
                };
                self.water[idx] = w;
                if w > 0.0 {
                    self.moisture[idx] = self.moisture[idx].max(0.8);
                    if self.grain[idx] == Grain::Other as u8 {
                        self.grain[idx] = Grain::Silt as u8;
                    }
                }
                vol += w * area;
            }
        }
        vol
    }

    /// Initial horizontal velocity. Call once. Later steps are the sheet's.
    pub fn map_water_velocity(&mut self, mut f: impl FnMut(f32, f32, f32) -> (f32, f32)) {
        self.restrict_fine();
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let idx = j * WINDOW_RES + i;
                if self.water[idx] <= 1.0e-6 {
                    continue;
                }
                let (x, z) = self.coarse_xz(i, j);
                let (u, v) = f(x, z, self.water[idx]);
                self.uw[idx] = if u.is_finite() { u } else { 0.0 };
                self.vw[idx] = if v.is_finite() { v } else { 0.0 };
            }
        }
    }

    pub fn non_finite_count(&self) -> u32 {
        let mut n = 0u32;
        for arr in [
            &self.water,
            &self.slurry,
            &self.uw,
            &self.vw,
            &self.us,
            &self.vs,
            &self.bed,
            &self.moisture,
            &self.bed_drop,
            &self.foam,
            &self.turbid,
        ] {
            for v in arr {
                if !v.is_finite() {
                    n += 1;
                }
            }
        }
        n
    }

    pub fn set_wind(&mut self, x: f32, z: f32) {
        self.wind = [x, z];
    }

    /// Track the chase point without sliding the coarse or air windows.
    pub fn set_center(&mut self, center: [f32; 2]) {
        self.center = center;
    }

    /// Move the window. Cells that slide off are parked. Cells that slide on
    /// are restored. The sum does not change.
    pub fn recenter(&mut self, center: [f32; 2]) {
        if self.fine_on {
            self.restrict_fine();
        }
        self.wet.forget_cells();
        self.center = center;
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        let new_origin = snap_origin(center, extent, COARSE_CELL_M);
        self.shift_coarse(new_origin);
        let air_cell = (WINDOW_RES as f32 * COARSE_CELL_M) / AIR_RES as f32;
        let air_extent = AIR_RES as f32 * air_cell;
        let new_air = snap_origin(center, air_extent, air_cell);
        self.shift_air(new_air, air_cell);
    }

    pub fn pour_water(&mut self, x0: f32, x1: f32, z0: f32, z1: f32, depth: f32) -> f32 {
        self.paint(x0, x1, z0, z1, depth, true, Grain::Water, 1.0)
    }

    pub fn pour_slurry(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        depth: f32,
        grain: Grain,
        moisture: f32,
    ) -> f32 {
        self.paint(x0, x1, z0, z1, depth, false, grain, moisture)
    }

    /// Remove slurry cubic metres at a world point. The caller holds that volume.
    pub fn take_slurry_at(&mut self, x: f32, z: f32, volume: f32) -> f32 {
        self.take_column(x, z, volume, false)
    }

    /// Remove water cubic metres at a world point. The caller holds that volume.
    pub fn take_water_at(&mut self, x: f32, z: f32, volume: f32) -> f32 {
        self.take_column(x, z, volume, true)
    }

    /// Put water cubic metres back onto the sheet. Used when an arch drips.
    pub fn add_water_at(&mut self, x: f32, z: f32, volume: f32) {
        if volume <= 0.0 {
            return;
        }
        if self.fine_on {
            self.restrict_fine();
        }
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return;
        };
        let idx = j * WINDOW_RES + i;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        self.water[idx] += volume / area;
    }

    /// Raise pore moisture on the coarse cells whose centres lie in the disk.
    /// Does not add or remove water, slurry, or dust. A wet tire track uses this
    /// so the moisture stays on the ground the Niva just left.
    pub fn raise_moisture_disk(&mut self, x: f32, z: f32, radius: f32, moisture: f32) {
        if !moisture.is_finite() || !x.is_finite() || !z.is_finite() {
            return;
        }
        if self.fine_on {
            self.restrict_fine();
        }
        let moisture = moisture.clamp(0.0, 1.0);
        let radius = radius.max(0.0);
        let reach = (radius / COARSE_CELL_M).ceil() as i32 + 1;
        let Some((ci, cj)) = self.coarse_ij(x, z) else {
            return;
        };
        for dj in -reach..=reach {
            for di in -reach..=reach {
                let i = ci as i32 + di;
                let j = cj as i32 + dj;
                if i < 0 || j < 0 || i >= WINDOW_RES as i32 || j >= WINDOW_RES as i32 {
                    continue;
                }
                let (px, pz) = self.coarse_xz(i as usize, j as usize);
                if (px - x).hypot(pz - z) > radius {
                    continue;
                }
                let idx = j as usize * WINDOW_RES + i as usize;
                if moisture > self.moisture[idx] {
                    self.moisture[idx] = moisture;
                }
            }
        }
    }

    /// Fill tire cuts up to the free surface of the pond beside them.
    ///
    /// A cut is a bed sitting more than 12 mm below the plane the sheet was
    /// poured on. Water thickness in that cut grows until bed + water matches
    /// the surrounding surface. The same cubic metres leave the still sheet,
    /// so the pond is not refilled from nowhere. Berms (a raised bed) are
    /// left alone. Returns the cubic metres moved.
    pub fn pond_ruts(&mut self) -> f32 {
        if self.fine_on {
            self.restrict_fine();
        }
        let n = WINDOW_RES;
        let nn = n * n;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let mut any_rut = false;
        for idx in 0..nn {
            let bed = self.bed[idx];
            if bed.is_finite() && bed < -0.012 {
                any_rut = true;
                break;
            }
        }
        if !any_rut {
            return 0.0;
        }
        let still = &mut self.pond_still[..nn];
        let eta_of = &mut self.pond_eta[..nn];
        still.fill(0);
        eta_of.fill(0.0);
        for idx in 0..nn {
            let bed = self.bed[idx];
            let water = self.water[idx];
            if bed.is_finite() && water.is_finite() && bed > -0.008 && bed < 0.03 && water > 0.018 {
                still[idx] = 1;
                eta_of[idx] = bed + water;
            }
        }
        let rad = 10i32;
        let mark = &mut self.pond_mark[..nn];
        mark.fill(0.0);
        for idx in 0..nn {
            if still[idx] != 0 {
                mark[idx] = 1.0;
            }
        }
        let stride = n + 1;
        let plane = stride * stride;
        let count_p = &mut self.pond_count_p[..plane];
        let eta_p = &mut self.pond_eta_p[..plane];
        count_p.fill(0.0);
        eta_p.fill(0.0);
        prefix_plane(mark, n, count_p);
        prefix_plane(eta_of, n, eta_p);
        self.pond_needs.clear();
        let needs = &mut self.pond_needs;
        let mut total_need = 0.0_f32;
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let bed = self.bed[idx];
                let water = self.water[idx];
                if !bed.is_finite() || !water.is_finite() || bed >= -0.012 {
                    continue;
                }
                let j0 = (j as i32 - rad).max(0) as usize;
                let j1 = ((j as i32 + rad).min(n as i32 - 1)) as usize;
                let i0 = (i as i32 - rad).max(0) as usize;
                let i1 = ((i as i32 + rad).min(n as i32 - 1)) as usize;
                let count = plane_window(&count_p, n, i0, i1, j0, j1);
                let sum = plane_window(&eta_p, n, i0, i1, j0, j1);
                if count < 1.0 {
                    continue;
                }
                let extra = ((sum / count) - bed - water).max(0.0);
                if extra > 1.0e-4 {
                    needs.push((idx, extra));
                    total_need += extra * area;
                }
            }
        }
        if total_need <= 1.0e-8 {
            return 0.0;
        }
        self.pond_donors.clear();
        let donors = &mut self.pond_donors;
        let mut total_room = 0.0_f32;
        for idx in 0..nn {
            if still[idx] == 0 {
                continue;
            }
            let room = (self.water[idx] - 0.012).max(0.0);
            if room > 1.0e-4 {
                donors.push((idx, room));
                total_room += room * area;
            }
        }
        if total_room <= 1.0e-8 {
            return 0.0;
        }
        let scale = (total_room / total_need).min(1.0);
        let moved = total_need * scale;
        for (idx, extra) in needs.iter() {
            self.water[*idx] += extra * scale;
        }
        for (idx, room) in donors.iter() {
            let give = moved * (*room * area) / total_room;
            self.water[*idx] = (self.water[*idx] - give / area).max(0.0);
        }
        moved
    }

    /// Seat the water column on the pressed ground of a rut that opens into the pond.
    ///
    /// `classify` returns that ground (bed + placed, not the height the cell
    /// started at) and whether the cell is a tire cut. The free surface is
    /// pressed ground plus the water column. Hanging the same column from the
    /// original height stands the rut above the pond by the depth of the cut.
    /// Water above the pond surface is poured back onto the still floor.
    /// Slurry left standing above that surface is scraped back to it. Returns
    /// the pond surface, metres, when the floor has a sheet.
    pub fn seat_rut_water(
        &mut self,
        mut classify: impl FnMut(usize, usize, f32, f32) -> (f32, bool),
    ) -> Option<f32> {
        self.restrict_fine();
        let n = WINDOW_RES;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        // Every cell is overwritten. A floor flag left from the previous step
        // would pour water onto a cell that is no longer the pond.
        self.seat_etas.clear();
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                let (x, z) = self.coarse_xz(i, j);
                let (ground, rut) = classify(i, j, x, z);
                let ground = if ground.is_finite() { ground } else { 0.0 };
                let water = self.water[idx];
                self.seat_pressed[idx] = ground;
                self.seat_is_rut[idx] = rut;
                // A floor the tires have drained still counts. Otherwise the
                // pond disappears, the rut is no longer "connected", and the
                // column left in the cut stands above a dry bed.
                let on_floor = !rut
                    && (-0.02..0.04).contains(&ground)
                    && (water > 0.008 || self.moisture[idx] > 0.55);
                self.seat_floor[idx] = on_floor;
                if on_floor {
                    self.seat_etas.push(ground + water);
                }
            }
        }
        if self.seat_etas.len() < 8 {
            return None;
        }
        // The median is the same element a full sort would put there.
        let mid = self.seat_etas.len() / 2;
        let (_, &mut eta, _) = self
            .seat_etas
            .select_nth_unstable_by(mid, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mut excess = 0.0_f32;
        let rad = 8i32;
        let cut_wet = (0..n * n).any(|idx| self.seat_is_rut[idx] && self.water[idx] >= 0.012);
        // Reuse the pond scratch planes: a fresh 160 KB pair per step is a
        // calloc plus page faults. `pond_ruts` refills both before it reads them.
        let floor_src = &mut self.pond_mark[..n * n];
        let floor_p = &mut self.pond_count_p[..(n + 1) * (n + 1)];
        if cut_wet {
            for (dst, &on) in floor_src.iter_mut().zip(&self.seat_floor[..n * n]) {
                *dst = if on { 1.0 } else { 0.0 };
            }
            floor_p.fill(0.0);
            prefix_plane(floor_src, n, floor_p);
        }
        for idx in 0..n * n {
            if !cut_wet {
                break;
            }
            if !self.seat_is_rut[idx] || self.water[idx] < 0.012 {
                continue;
            }
            let i = idx % n;
            let j = idx / n;
            let j0 = (j as i32 - rad).max(0) as usize;
            let j1 = ((j as i32 + rad).min(n as i32 - 1)) as usize;
            let i0 = (i as i32 - rad).max(0) as usize;
            let i1 = ((i as i32 + rad).min(n as i32 - 1)) as usize;
            // A floor flag is 0 or 1, and the window holds fewer than 2^23 of them,
            // so the prefix count is exact.
            if plane_window(&self.pond_count_p, n, i0, i1, j0, j1) <= 0.0 {
                continue;
            }
            // The column is measured from the pressed ground. The original
            // height is that ground plus the rut, and it is not the datum.
            let limit = (eta - self.seat_pressed[idx]).max(0.0);
            if self.water[idx] > limit + 0.004 {
                let drop = self.water[idx] - limit;
                self.water[idx] = limit;
                excess += drop * area;
            }
            let slurry_room = (eta - self.seat_pressed[idx] - self.water[idx]).max(0.0);
            if self.slurry[idx] > slurry_room + 0.004 {
                self.slurry[idx] = slurry_room;
            }
        }
        // Water the tires pushed up the bank. The free surface there is the
        // same pond: a column that stands above it is poured back onto the floor.
        // Leaving it makes one cell on the shore a wall of water.
        for idx in 0..n * n {
            if self.seat_is_rut[idx] || self.seat_pressed[idx] < 0.02 {
                continue;
            }
            let limit = (eta - self.seat_pressed[idx]).max(0.0);
            if self.water[idx] > limit + 0.004 {
                let drop = self.water[idx] - limit;
                self.water[idx] = limit;
                excess += drop * area;
            }
            let slurry_room = (eta - self.seat_pressed[idx] - self.water[idx]).max(0.0);
            if self.slurry[idx] > slurry_room + 0.004 {
                self.slurry[idx] = slurry_room;
            }
        }
        if excess > 1.0e-8 {
            let mut room = 0.0_f32;
            for idx in 0..n * n {
                if self.seat_floor[idx] {
                    room += self.water[idx].max(0.0);
                }
            }
            if room > 1.0e-6 {
                for idx in 0..n * n {
                    if self.seat_floor[idx] && self.water[idx] > 0.0 {
                        self.water[idx] += excess / area * (self.water[idx] / room);
                    }
                }
            }
        }
        Some(eta)
    }

    fn take_column(&mut self, x: f32, z: f32, volume: f32, water: bool) -> f32 {
        if volume <= 0.0 {
            return 0.0;
        }
        if self.fine_on {
            self.restrict_fine();
        }
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return 0.0;
        };
        let idx = j * WINDOW_RES + i;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let col = if water {
            &mut self.water
        } else {
            &mut self.slurry
        };
        let have = col[idx] * area;
        let take = volume.min(have).max(0.0);
        if take <= 0.0 {
            return 0.0;
        }
        col[idx] -= take / area;
        take
    }

    fn paint(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        depth: f32,
        water: bool,
        grain: Grain,
        moisture: f32,
    ) -> f32 {
        let depth = depth.max(0.0);
        let mut vol = 0.0f32;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * WINDOW_RES + i;
                if water {
                    self.water[idx] = depth;
                } else {
                    self.slurry[idx] = depth;
                    self.grain[idx] = grain as u8;
                    self.moisture[idx] = moisture.clamp(0.0, 1.0);
                }
                vol += depth * area;
            }
        }
        vol
    }

    /// Multiply the coarse water and slurry by masks in 0..1. A pour can
    /// start as a box and leave as an oval or a pair of ruts.
    pub fn scale_columns(&mut self, mut mask: impl FnMut(f32, f32) -> (f32, f32)) {
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                let (mw, ms) = mask(x, z);
                let idx = j * WINDOW_RES + i;
                self.water[idx] *= mw.clamp(0.0, 1.0);
                self.slurry[idx] *= ms.clamp(0.0, 1.0);
            }
        }
    }

    /// Pull the rim of each sheet down to the bed over `feather_m`. The
    /// interior depth stays. A hard rectangle does not stay a wall of fluid.
    pub fn soften_rims(&mut self, feather_m: f32) {
        let feather = feather_m.max(COARSE_CELL_M);
        let rad = (feather / COARSE_CELL_M).ceil() as i32;
        let n = WINDOW_RES;
        let mut water = self.water.clone();
        let mut slurry = self.slurry.clone();
        for j in 0..n {
            for i in 0..n {
                let idx = j * n + i;
                if self.water[idx] > 1.0e-6 {
                    water[idx] *= rim_scale(&self.water, i, j, rad, feather);
                }
                if self.slurry[idx] > 1.0e-6 {
                    slurry[idx] *= rim_scale(&self.slurry, i, j, rad, feather);
                }
            }
        }
        self.water = water;
        self.slurry = slurry;
    }

    /// Copy the solid under the film from the terramechanics field. Does not
    /// touch water, slurry, or dust volumes.
    pub fn read_substrate(&mut self, mut sample: impl FnMut(f32, f32) -> (f32, f32, Grain)) {
        self.read_substrate_grid(|_i, _j, x, z| sample(x, z));
    }

    /// Same as [`Self::read_substrate`], plus the coarse cell index so the
    /// caller can reuse one soil lookup per row and per column.
    pub(crate) fn read_substrate_grid(
        &mut self,
        mut sample: impl FnMut(usize, usize, f32, f32) -> (f32, f32, Grain),
    ) {
        self.restrict_fine();
        let mut hist = [0u32; 101];
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let (x, z) = self.coarse_xz(i, j);
                let (bed, moisture, grain) = sample(i, j, x, z);
                let idx = j * WINDOW_RES + i;
                let moisture = moisture.clamp(0.0, 1.0);
                self.bed[idx] = bed;
                self.moisture[idx] = moisture;
                // Round half away from zero without the libm call: for
                // 0..=100 the fraction after truncation is exact.
                let pct = moisture * 100.0;
                let whole = pct as usize;
                let bucket = if pct - whole as f32 >= 0.5 { whole + 1 } else { whole };
                hist[bucket.min(100)] += 1;
                if self.slurry[idx] > 1.0e-5 {
                    self.grain[idx] = grain as u8;
                }
            }
        }
        let mut best = 0usize;
        let mut best_n = 0u32;
        for (i, n) in hist.iter().enumerate() {
            if *n > best_n {
                best = i;
                best_n = *n;
            }
        }
        self.moisture_ambient = best as f32 * 0.01;
    }

    pub fn push_tire(&mut self, push: TirePush) {
        self.pushes.push(push);
    }

    /// A tire whose volume the sheet already sees through
    /// [`FluidCore::set_hydro_solids`]. Next step it lifts silt behind the
    /// contact and foams the surface it cuts, like [`FluidCore::push_tire`],
    /// but it does not shove water or throw spray, so the water book is the
    /// hydro solid's alone.
    pub fn push_stir(&mut self, push: TirePush) {
        self.stirs.push(push);
    }

    /// Cubic metres that entered the air. Zero on a wet cell.
    pub fn inject_dust(&mut self, inj: DustInject, dt: f32) -> f32 {
        if !dust_may_lift(inj.moisture, inj.water_m) || dt <= 0.0 || inj.offer_m3_s <= 0.0 {
            return 0.0;
        }
        let dryness = (1.0 - inj.moisture.clamp(0.0, 1.0)).max(0.0);
        let slip = inj.slip.abs().clamp(0.0, 1.5);
        let vol = inj.offer_m3_s * dt * dryness * (0.25 + slip);
        if vol <= 1.0e-8 {
            return 0.0;
        }
        let air_cell = self.air_cell();
        let Some((i, j)) = self.air_ij(inj.x, inj.z) else {
            return 0.0;
        };
        let layer = AIR_LAYERS / 2;
        let idx = (layer * AIR_RES + j) * AIR_RES + i;
        let area = air_cell * air_cell;
        self.dust[idx] += vol / area;
        let lift = 0.6 + slip;
        self.dust_u[idx] += self.wind[0] * 0.25;
        self.dust_v[idx] += self.wind[1] * 0.25 + lift * 0.05;
        vol
    }

    /// Cubic metres of water that left the sheet and entered the fog slab.
    ///
    /// Zero when the film is thinner than [`FOG_MIN_WATER_M`], or when the
    /// offer misses the air window. The missed offer is put back on the sheet.
    /// This does not touch the dust column or `air_velocity`.
    pub fn inject_fog(&mut self, inj: FogInject, dt: f32) -> f32 {
        if !fog_may_lift(inj.water_m) || dt <= 0.0 || !dt.is_finite() || inj.offer_m3_s <= 0.0 {
            return 0.0;
        }
        let want = inj.offer_m3_s * dt;
        if !want.is_finite() || want <= 1.0e-8 {
            return 0.0;
        }
        let taken = self.take_water_at(inj.x, inj.z, want);
        if taken <= 1.0e-8 {
            return 0.0;
        }
        let air_cell = self.air_cell();
        let Some((i, j)) = self.air_ij(inj.x, inj.z) else {
            self.add_water_at(inj.x, inj.z, taken);
            return 0.0;
        };
        let layer = 1usize.min(AIR_LAYERS - 1);
        let idx = (layer * AIR_RES + j) * AIR_RES + i;
        let area = air_cell * air_cell;
        self.fog[idx] += taken / area;
        self.fog_u[idx] = self.wind[0];
        self.fog_v[idx] = self.wind[1];
        self.fog_live = true;
        taken
    }

    pub fn step(&mut self, dt: f32) -> Vec<SprayEvent> {
        let wall = ProfTick::start();
        let w0 = self.sheet_times.water_ms;
        let s0 = self.sheet_times.slurry_ms;
        let p0 = self.sheet_times.prep_ms;
        let r0 = self.sheet_times.restrict_ms;
        let n0 = self.sheet_times.window_ms;
        let d0 = self.sheet_times.dust_ms;
        let slow = self.frame % SLOW_STRIDE == 0;
        self.frame = self.frame.wrapping_add(1);
        let pushes = std::mem::take(&mut self.pushes);
        let mut stirs = std::mem::take(&mut self.stirs);
        if !stirs.is_empty() {
            stirs.extend_from_slice(&pushes);
        }
        let wake: &[TirePush] = if stirs.is_empty() { &pushes } else { &stirs };
        // Lift from the column the tire rolls into, before the sheet shoves it.
        if self.turbidity_enabled && dt > 0.0 && dt.is_finite() && !wake.is_empty() {
            self.entrain_wake(wake, dt);
        }
        let wet = self.wet.active();
        if wet {
            self.wet_prepare_coarse();
        }
        self.refine(&pushes, dt);
        if wet {
            self.wet_prepare_fine();
        }
        self.apply_pushes(&pushes, true);
        self.step_pair(dt, true);
        let mut spray = self.peel_spray(&pushes, true, false);
        self.restrict_fine();
        self.apply_pushes_outside(&pushes);
        self.drive_waves(dt);
        self.step_pair(dt, false);
        if wet {
            self.wet_finish();
            spray.extend(self.wet_splash(dt));
        }
        self.bed_friction(dt);
        self.diffuse_surface(dt);
        spray.extend(self.peel_spray(&pushes, false, true));
        if !self.solids.is_empty() {
            self.scour_contacts(dt);
        }
        self.cycle_stream();
        self.step_tracers(dt, wake);
        stirs.clear();
        self.stirs = stirs;
        self.last_pushes = pushes;
        self.feed_reach();
        self.soak_shore(dt);
        let sus_tick = ProfTick::start();
        if self.turbidity_enabled {
            self.step_suspended(dt);
        }
        let suspended_ms = sus_tick.ms() as f32;
        self.step_dust(dt, slow);
        self.step_fog(dt);
        if slow {
            if self.dust.iter().sum::<f32>() > 1.0e-6 {
                self.settle_dust();
            }
            if self.fog_live {
                self.settle_fog();
            }
        }
        for s in &spray {
            self.spray_m3 += s.volume;
            self.spray_thrown_m3 += s.volume;
        }
        self.keep_spray(&spray, dt);
        let water_ms = (self.sheet_times.water_ms - w0) as f32;
        let slurry_ms = (self.sheet_times.slurry_ms - s0) as f32;
        let prep_ms = (self.sheet_times.prep_ms - p0) as f32;
        let restrict_ms = (self.sheet_times.restrict_ms - r0) as f32;
        let window_ms = (self.sheet_times.window_ms - n0) as f32;
        let dust_ms = (self.sheet_times.dust_ms - d0) as f32;
        self.last_parts = StepParts {
            water_ms,
            slurry_ms,
            dust_ms,
            suspended_ms,
            prep_ms,
            restrict_ms,
            window_ms,
        };
        let accounted = f64::from(water_ms)
            + f64::from(slurry_ms)
            + f64::from(prep_ms)
            + f64::from(restrict_ms)
            + f64::from(window_ms)
            + f64::from(dust_ms);
        self.sheet_times.rest_ms += (wall.ms() - accounted).max(0.0);
        spray
    }

    /// Tire shoves applied on the step that just finished.
    pub fn tire_pushes(&self) -> &[TirePush] {
        &self.last_pushes
    }

    /// Foam fraction on the coarse window, 0 to 1. Index is `j * WINDOW_RES + i`.
    pub fn foam_grid(&self) -> &[f32] {
        &self.foam
    }

    /// Metres of soil stirred into the water on the coarse window.
    pub fn turbid_grid(&self) -> &[f32] {
        &self.turbid
    }

    /// Foam fraction and stirred-silt thickness at a world point.
    /// A dry cell, film under 0.01 mm, is `(0, 0)`.
    pub fn foam_turbid_at(&self, x: f32, z: f32) -> (f32, f32) {
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return (0.0, 0.0);
        };
        let idx = j * WINDOW_RES + i;
        let film = self.water[idx] + self.slurry[idx];
        if !film.is_finite() || film < 1.0e-5 {
            return (0.0, 0.0);
        }
        let foam = if film < 0.004 || !self.foam[idx].is_finite() {
            0.0
        } else {
            self.foam[idx].clamp(0.0, 1.0)
        };
        let turbid = if self.turbid[idx].is_finite() {
            self.turbid[idx].max(0.0)
        } else {
            0.0
        };
        (foam, turbid)
    }

    /// `|lifted − (still in the water + settled)|` in cubic metres.
    /// The standing slurry pour is not in this book.
    pub fn silt_book_gap(&self) -> f32 {
        (self.stirred_m3 - self.silt_accounted_m3()).abs()
    }

    /// Cubic metres lifted by tire and shore shear.
    pub fn stirred_m3(&self) -> f32 {
        self.stirred_m3
    }

    fn silt_accounted_m3(&self) -> f32 {
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let mut volume = self.turbid.iter().fold(0.0f32, |acc, t| {
            acc + if t.is_finite() { t.max(0.0) } else { 0.0 }
        }) * area;
        volume += if self.turbid_settled_m3.is_finite() {
            self.turbid_settled_m3.max(0.0)
        } else {
            0.0
        };
        for parked in self.parked.values() {
            if parked.turbid.is_finite() {
                volume += parked.turbid.max(0.0) * area;
            }
        }
        volume
    }

    /// Foam and stirred silt for one step.
    ///
    /// Foam is born where the free surface cuts a moving tire, where the film
    /// runs into the shore, where the film itself is breaking, and where a
    /// crest stands above the surrounding film. The amount is the shear and
    /// the impact. The field is then carried with the film and decays. It is
    /// not a disc painted under the tire, and the publisher does not draw it.
    ///
    /// Silt is bed soil lifted into the water by the same shear. Slip under
    /// the contact does it where the tire is submerged. A wave hitting the
    /// shore does it on that edge. The cubic metres sit in `turbid` until
    /// they settle. They are not poured into the standing slurry sheet.
    fn step_tracers(&mut self, dt: f32, pushes: &[TirePush]) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        // A paddle swell or an open river reach breaks and foams with no tire.
        let wake = self.surface_driven()
            || pushes.iter().any(|p| {
                (p.speed.is_finite() && p.speed >= 0.25)
                    || (p.slip.is_finite() && p.slip.abs() >= 0.05)
            });
        let live_silt = self.turbid.iter().any(|t| t.is_finite() && *t > 1.0e-8);
        let live_foam = !self.foam_live.is_empty();
        if !wake && !live_silt && !live_foam {
            return;
        }
        self.collect_tracer_wet(pushes);
        if self.wet_idx.is_empty() && !live_silt {
            return;
        }
        if live_foam {
            self.advect_foam(dt);
        }
        self.advect_turbid(dt);
        self.settle_turbid(dt);
        self.spawn_foam_and_silt(dt, pushes);
        self.clear_dry_tracers();
        self.sync_foam_live();
    }

    /// True when a swell or a river reach drives the sheet, so the surface
    /// can break and foam without a tire in it.
    fn surface_driven(&self) -> bool {
        self.wave.is_some() || self.reach.is_some()
    }

    /// Wet and shore cells. Wake steps scan the window; decay steps visit a box.
    fn collect_tracer_wet(&mut self, pushes: &[TirePush]) {
        self.wet_idx.clear();
        self.shore_idx.clear();
        let res = WINDOW_RES;
        let wake = self.surface_driven()
            || pushes.iter().any(|p| {
                (p.speed.is_finite() && p.speed >= 0.25)
                    || (p.slip.is_finite() && p.slip.abs() >= 0.05)
            });
        let (i0, i1, j0, j1) = if wake {
            (0, res - 1, 0, res - 1)
        } else if let Some((mut a, mut b, mut c, mut d)) = self.tracer_visit_box(pushes) {
            const MARGIN: usize = 2;
            a = a.saturating_sub(MARGIN);
            c = c.saturating_sub(MARGIN);
            b = (b + MARGIN).min(res - 1);
            d = (d + MARGIN).min(res - 1);
            (a, b, c, d)
        } else {
            return;
        };
        for j in j0..=j1 {
            let row = j * res;
            // Skip a dry row with one branch-free pass. NaN never counts as wet.
            let peak = self.water[row + i0..=row + i1]
                .iter()
                .zip(&self.slurry[row + i0..=row + i1])
                .fold(0.0f32, |m, (w, s)| m.max(w + s));
            if peak < 0.004 {
                continue;
            }
            for i in i0..=i1 {
                let idx = row + i;
                let h = self.water[idx] + self.slurry[idx];
                if !h.is_finite() || h < 0.004 {
                    continue;
                }
                self.wet_idx.push(idx as u32);
                if shore_wet_cell(&self.water, res, i, j) {
                    self.shore_idx.push(idx as u32);
                }
            }
        }
    }

    /// Inclusive coarse index box covering foam, silt, and tire reach.
    fn tracer_visit_box(&self, pushes: &[TirePush]) -> Option<(usize, usize, usize, usize)> {
        let res = WINDOW_RES;
        let mut i0 = res;
        let mut i1 = 0usize;
        let mut j0 = res;
        let mut j1 = 0usize;
        let mut any = false;
        for &id in &self.foam_live {
            let idx = id as usize;
            let i = idx % res;
            let j = idx / res;
            i0 = i0.min(i);
            i1 = i1.max(i);
            j0 = j0.min(j);
            j1 = j1.max(j);
            any = true;
        }
        if let Some((ti0, ti1, tj0, tj1)) = turbid_span(&self.turbid, res) {
            i0 = i0.min(ti0);
            i1 = i1.max(ti1);
            j0 = j0.min(tj0);
            j1 = j1.max(tj1);
            any = true;
        }
        let reach = (8.0 / COARSE_CELL_M).ceil() as i32;
        for push in pushes {
            let Some((ci, cj)) = self.coarse_ij(push.x, push.z) else {
                continue;
            };
            let ci = ci as i32;
            let cj = cj as i32;
            let i_lo = (ci - reach).max(0) as usize;
            let i_hi = (ci + reach).min(res as i32 - 1) as usize;
            let j_lo = (cj - reach).max(0) as usize;
            let j_hi = (cj + reach).min(res as i32 - 1) as usize;
            i0 = i0.min(i_lo);
            i1 = i1.max(i_hi);
            j0 = j0.min(j_lo);
            j1 = j1.max(j_hi);
            any = true;
        }
        if !any {
            return None;
        }
        Some((i0, i1, j0, j1))
    }

    fn sync_foam_live(&mut self) {
        self.foam_live.clear();
        self.foam_live_mark.fill(0);
        for &id in &self.wet_idx {
            let idx = id as usize;
            if self.foam[idx].is_finite() && self.foam[idx] > 1.0e-4 {
                self.foam_live_mark[idx] = 1;
                self.foam_live.push(id);
            }
        }
    }

    /// Semi-Lagrangian foam on the wet cells only. A dry cell is written 0.
    fn advect_foam(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let cell = COARSE_CELL_M;
        let n = res * res;
        if self.foam.len() != n {
            self.foam.resize(n, 0.0);
        }
        let src = std::mem::take(&mut self.foam);
        let ids = std::mem::take(&mut self.wet_idx);
        if self.tracer_scratch.len() < ids.len() {
            self.tracer_scratch.resize(ids.len().max(1), 0.0);
        }
        for (k, &idx) in ids.iter().enumerate() {
            let idx = idx as usize;
            let u = self.uw[idx];
            let v = self.vw[idx];
            let sample = if u.is_finite() && v.is_finite() {
                let i = idx % res;
                let j = idx / res;
                // Index space: integer `i` is the centre of that cell.
                let x = i as f32 - u * dt / cell;
                let z = j as f32 - v * dt / cell;
                sample_foam(&src, res, x, z)
            } else {
                0.0
            };
            self.tracer_scratch[k] = sample;
        }
        let mut next = src;
        for &id in &self.foam_live {
            next[id as usize] = 0.0;
        }
        for &id in &ids {
            next[id as usize] = 0.0;
        }
        let decay = (-dt / FOAM_LIFE_S).exp();
        for (k, &idx) in ids.iter().enumerate() {
            let v = self.tracer_scratch[k] * decay;
            if v > 1.0e-4 && v.is_finite() {
                next[idx as usize] = v.min(FOAM_EQ);
            }
        }
        self.foam = next;
        self.wet_idx = ids;
    }

    /// Approach `FOAM_EQ` under sustained shear. The source is zero at that
    /// ceiling, so the cell does not run away to 1.
    fn add_foam(&mut self, idx: usize, prod: f32, dt: f32) {
        if prod <= 0.0 || !prod.is_finite() || !(dt > 0.0) {
            return;
        }
        let cur = if self.foam[idx].is_finite() {
            self.foam[idx].clamp(0.0, FOAM_EQ)
        } else {
            0.0
        };
        let room = FOAM_EQ - cur;
        if room <= 1.0e-5 {
            return;
        }
        let grown = cur + room * prod * (dt / FOAM_TAU_S).min(1.0);
        self.foam[idx] = if grown.is_finite() {
            grown.min(FOAM_EQ)
        } else {
            0.0
        };
    }

    fn advect_turbid(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let cell = COARSE_CELL_M;
        let n = res * res;
        let Some((i0, i1, j0, j1)) = turbid_span(&self.turbid, res) else {
            return;
        };
        if self.tracer_scratch.len() < n * 2 {
            self.tracer_scratch.resize(n * 2, 0.0);
        }
        let scratch = &mut self.tracer_scratch;
        // Only the live box is nonzero. Clearing that box is enough: the rest
        // of scratch stays 0 from the previous step's box, which we also clear.
        for j in j0..=j1 {
            let row = j * res;
            for i in i0..=i1 {
                let idx = row + i;
                scratch[idx] = 0.0;
                scratch[n + idx] = 0.0;
            }
        }
        flux_thickness(
            &self.turbid,
            &self.uw,
            &self.vw,
            res,
            cell,
            dt,
            i0,
            i1,
            j0,
            j1,
            scratch,
        );
        for j in j0..=j1 {
            let row = j * res;
            for i in i0..=i1 {
                let idx = row + i;
                let removed = scratch[idx];
                let added = scratch[n + idx];
                if removed == 0.0 && added == 0.0 {
                    if !self.turbid[idx].is_finite() {
                        self.turbid[idx] = 0.0;
                    }
                    continue;
                }
                let next = self.turbid[idx] - removed + added;
                self.turbid[idx] = if next.is_finite() { next.max(0.0) } else { 0.0 };
            }
        }
    }

    fn settle_turbid(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let keep = (-dt / SILT_SETTLE_S).exp();
        let mut settled = 0.0f32;
        let Some((i0, i1, j0, j1)) = turbid_span(&self.turbid, res) else {
            return;
        };
        for j in j0..=j1 {
            let row = j * res;
            for i in i0..=i1 {
                let idx = row + i;
                let h = self.water[idx] + self.slurry[idx];
                let column = if self.turbid[idx].is_finite() {
                    self.turbid[idx].max(0.0)
                } else {
                    0.0
                };
                if column <= 0.0 {
                    self.turbid[idx] = 0.0;
                    continue;
                }
                // A dry cell cannot hold silt. What was in it has landed.
                if !h.is_finite() || h < 1.0e-5 {
                    settled += column * area;
                    self.turbid[idx] = 0.0;
                    continue;
                }
                let stay = column * keep;
                settled += (column - stay) * area;
                self.turbid[idx] = stay;
            }
        }
        if settled.is_finite() {
            self.turbid_settled_m3 += settled;
        }
    }

    fn spawn_foam_and_silt(&mut self, dt: f32, pushes: &[TirePush]) {
        let res = WINDOW_RES;
        let cell = COARSE_CELL_M;
        let area = cell * cell;
        for push in pushes {
            self.spawn_tire_tracers(push, dt, area);
        }
        self.spawn_shore_tracers(dt, area, res);
        self.spawn_breaking_speed(dt);
    }

    /// Crest left by seating the sheet into the rut. That seat runs after
    /// the sheet step, so the step's foam pass has not seen it yet. This
    /// writes the impact into the foam field and clears anything the seat
    /// left on a dry cell. The next step carries the foam.
    pub(crate) fn note_surface_foam(&mut self, dt: f32) {
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        let pushes = self.last_pushes.clone();
        self.collect_tracer_wet(&pushes);
        if self.wet_idx.is_empty() {
            return;
        }
        self.spawn_breaking_crest(dt);
        self.clear_dry_tracers();
        self.sync_foam_live();
    }

    /// Froude foam on the wet cells. Crest from the seated sheet runs in
    /// [`Self::note_surface_foam`], after `seat_rut_sheet`.
    fn spawn_breaking_speed(&mut self, dt: f32) {
        let ids = std::mem::take(&mut self.wet_idx);
        for &id in &ids {
            let idx = id as usize;
            let h = self.water[idx] + self.slurry[idx];
            if !h.is_finite() || h < 0.004 {
                continue;
            }
            let u = self.uw[idx];
            let v = self.vw[idx];
            if !u.is_finite() || !v.is_finite() || u.hypot(v) <= 0.05 {
                continue;
            }
            let speed = u.hypot(v);
            self.add_foam(idx, breaking_foam(speed, h), dt);
        }
        self.wet_idx = ids;
    }

    /// Crest left when the seated sheet stands above its neighbours. Froude
    /// foam for the same step already ran in [`Self::spawn_breaking_speed`].
    fn spawn_breaking_crest(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let ids = std::mem::take(&mut self.wet_idx);
        for &id in &ids {
            let idx = id as usize;
            let i = idx % res;
            let j = idx / res;
            let h = self.water[idx] + self.slurry[idx];
            if !h.is_finite() || h < 0.004 {
                continue;
            }
            let mut sum = 0.0f32;
            let mut count = 0.0f32;
            for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                let ni = i as i32 + di;
                let nj = j as i32 + dj;
                if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                    continue;
                }
                let nidx = nj as usize * res + ni as usize;
                let nh = self.water[nidx] + self.slurry[nidx];
                if nh.is_finite() {
                    sum += nh.max(0.0);
                    count += 1.0;
                }
            }
            if count <= 0.0 {
                continue;
            }
            let excess = h - sum / count;
            if excess <= 0.0 {
                continue;
            }
            let prod = smoothstep(0.004, 0.018, excess);
            self.add_foam(idx, prod, dt);
        }
        self.wet_idx = ids;
    }

    fn spawn_tire_tracers(&mut self, push: &TirePush, dt: f32, area: f32) {
        let speed = if push.speed.is_finite() {
            push.speed.max(0.0)
        } else {
            0.0
        };
        let slip = if push.slip.is_finite() {
            push.slip.abs().min(1.5)
        } else {
            0.0
        };
        if speed < 0.25 && slip < 0.05 {
            return;
        }
        let radius = push.radius.max(0.05);
        let fx = push.forward_x;
        let fz = push.forward_z;
        let len = (fx * fx + fz * fz).sqrt().max(1.0e-4);
        let fx = fx / len;
        let fz = fz / len;
        let rx = fz;
        let rz = -fx;
        let reach = radius * 1.15 + COARSE_CELL_M;
        let reach2 = reach * reach;
        let slip_v = speed * slip;
        let cell = COARSE_CELL_M;
        let wet = std::mem::take(&mut self.wet_idx);
        for &id in &wet {
            let idx = id as usize;
            let h = self.water[idx];
            if !h.is_finite() || h < 0.004 {
                continue;
            }
            let i = idx % WINDOW_RES;
            let j = idx / WINDOW_RES;
            let x = self.origin[0] + (i as f32 + 0.5) * cell;
            let z = self.origin[1] + (j as f32 + 0.5) * cell;
            let dx = x - push.x;
            let dz = z - push.z;
            if dx * dx + dz * dz > reach2 {
                continue;
            }
            let along = dx * fx + dz * fz;
            let lat = dx * rx + dz * rz;
            let dist = dx.hypot(dz);
            let film = self.uw[idx].hypot(self.vw[idx]);
            let film = if film.is_finite() { film } else { 0.0 };
            // The free surface cuts the tire on this ring. Inside the
            // ring the tread is under water; outside it is in the air.
            let immerse = h.min(radius * 0.9);
            let r_line = (2.0 * radius * immerse - immerse * immerse).max(0.0).sqrt();
            let band = 1.0 - ((dist - r_line).abs() / (cell * 1.35)).clamp(0.0, 1.0);
            if band > 0.0 {
                let motion = slip_v.max(speed * 0.45).max(film);
                let prod = breaking_foam(motion, h) * band;
                self.add_foam(idx, prod, dt);
            }
            // Submerged contact. Slip shears the bed and lifts it.
            let on_patch = lat.abs() <= push.width * 0.55 + cell * 0.35
                && along >= -radius * 0.55
                && along <= radius * 0.35;
            if on_patch && h > 0.02 {
                let shear_v = slip_v.max(film * 0.35);
                self.lift_silt(idx, shear_v, dt, area, 0.015, 6.0);
            }
        }
        self.wet_idx = wet;
    }

    fn spawn_shore_tracers(&mut self, dt: f32, area: f32, res: usize) {
        let ids = std::mem::take(&mut self.shore_idx);
        for &id in &ids {
            let idx = id as usize;
            let (prod, silt_add) = shore_cell(
                &self.water,
                &self.uw,
                &self.vw,
                res,
                idx,
            );
            self.add_foam(idx, prod, dt);
            if silt_add > 0.0 {
                let dh = (silt_add * dt).min(0.0012);
                self.add_silt(idx, dh, area);
            }
        }
        self.shore_idx = ids;
    }

    fn lift_silt(&mut self, idx: usize, shear_v: f32, dt: f32, area: f32, rate_cap: f32, tau_c: f32) {
        if !shear_v.is_finite() || shear_v <= 0.0 {
            return;
        }
        let tau = 1000.0 * 0.035 * shear_v * shear_v;
        if tau <= tau_c {
            return;
        }
        let rate = (0.008 * (tau / tau_c - 1.0)).min(rate_cap);
        let dh = (rate * dt).min(0.0025);
        self.add_silt(idx, dh, area);
    }

    fn add_silt(&mut self, idx: usize, dh: f32, area: f32) {
        if dh <= 0.0 || !dh.is_finite() {
            return;
        }
        let room = (SILT_COLUMN_CAP_M - self.turbid[idx].max(0.0)).max(0.0);
        let add = dh.min(room);
        if add <= 0.0 {
            return;
        }
        self.turbid[idx] += add;
        self.stirred_m3 += add * area;
    }

    fn clear_dry_tracers(&mut self) {
        let res = WINDOW_RES;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        for &id in &self.wet_idx {
            let idx = id as usize;
            let h = self.water[idx] + self.slurry[idx];
            if !h.is_finite() || h < 0.004 {
                self.foam[idx] = 0.0;
            } else if !self.foam[idx].is_finite() {
                self.foam[idx] = 0.0;
            }
        }
        let Some((i0, i1, j0, j1)) = turbid_span(&self.turbid, res) else {
            return;
        };
        for j in j0..=j1 {
            let row = j * res;
            for i in i0..=i1 {
                let idx = row + i;
                let h = self.water[idx] + self.slurry[idx];
                if !h.is_finite() || h < 1.0e-5 {
                    if self.turbid[idx] > 0.0 {
                        self.turbid_settled_m3 += self.turbid[idx] * area;
                        self.turbid[idx] = 0.0;
                    }
                } else if !self.turbid[idx].is_finite() {
                    self.turbid[idx] = 0.0;
                }
            }
        }
    }

    /// Water and other peeled drops still in the air. The picture mesh does
    /// not draw these. The render contract does.
    pub fn spray_drops(&self) -> &[LiveSpray] {
        &self.drops
    }

    fn keep_spray(&mut self, fresh: &[SprayEvent], dt: f32) {
        let dt = if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        // Ballistic flight with a little air drag on the horizontal (drops of
        // a few mm lose ~10 %/s), so a long throw does not carry tire speed
        // across the whole window.
        let drag = (1.0 - SPRAY_AIR_DRAG * dt).clamp(0.0, 1.0);
        for drop in &mut self.drops {
            drop.vel[0] *= drag;
            drop.vel[2] *= drag;
            drop.vel[1] -= GRAVITY * dt;
            drop.pos[0] += drop.vel[0] * dt;
            drop.pos[1] += drop.vel[1] * dt;
            drop.pos[2] += drop.vel[2] * dt;
            drop.age += dt;
        }
        // One parcel per 20 cm cell per step. The 2 cm peel makes thousands
        // of millilitre events; flying and landing them one by one cost the
        // drive step ~0.5 ms. Volume is summed; position, velocity and murk
        // are volume-weighted, so the throw and the ledger are unchanged.
        let mut cell_of: Vec<(u32, usize)> = Vec::with_capacity(fresh.len().min(1024));
        for event in fresh {
            if event.kind != 0 || !event.volume.is_finite() || event.volume <= 0.0 {
                continue;
            }
            let key = match self.coarse_ij(event.x, event.z) {
                Some((i, j)) => (j * WINDOW_RES + i) as u32,
                None => u32::MAX,
            };
            let w = event.volume;
            let hit = if key == u32::MAX {
                None
            } else {
                cell_of.iter().find(|(k, _)| *k == key).map(|(_, at)| *at)
            };
            match hit {
                Some(at) => {
                    let d = &mut self.drops[at];
                    let t = d.volume + w;
                    let a = d.volume / t;
                    let b = w / t;
                    d.pos = [
                        d.pos[0] * a + event.x * b,
                        d.pos[1] * a + event.y * b,
                        d.pos[2] * a + event.z * b,
                    ];
                    d.vel = [
                        d.vel[0] * a + event.vx * b,
                        d.vel[1] * a + event.vy * b,
                        d.vel[2] * a + event.vz * b,
                    ];
                    d.murk = d.murk * a + event.murk * b;
                    d.volume = t;
                }
                None => {
                    if key != u32::MAX {
                        cell_of.push((key, self.drops.len()));
                    }
                    self.drops.push(LiveSpray {
                        pos: [event.x, event.y, event.z],
                        vel: [event.vx, event.vy, event.vz],
                        volume: w,
                        kind: event.kind,
                        age: 0.0,
                        murk: event.murk,
                    });
                }
            }
        }

        self.land_spray();
        const SPRAY_KEEP: usize = 8192;
        if self.drops.len() > SPRAY_KEEP {
            // The oldest drops go back to the sheet where they are, not out
            // of the ledger.
            let skip = self.drops.len() - SPRAY_KEEP;
            let old: Vec<LiveSpray> = self.drops.drain(0..skip).collect();
            for drop in old {
                self.deposit_drop(&drop);
            }
        }
        // The account is what is up now, summed, so it cannot drift away
        // from the drops by rounding.
        let up: f64 = self.drops.iter().map(|d| d.volume as f64).sum();
        self.spray_m3 = (up + self.spray_stranded_m3 as f64) as f32;
    }

    /// Hybrid H2, 3D -> 2D: a drop that meets the sheet surface (bed plus
    /// water) gives its volume back to that coarse column and its horizontal
    /// momentum to the column velocity. A drop that outlives `SPRAY_LIFE_S`
    /// lands where it is. `spray_m3` is the volume still in the air, so the
    /// ledger closes when the air is empty. A drop outside the window stays
    /// booked as spray (there is no column to hold it).
    fn land_spray(&mut self) {
        if self.drops.is_empty() {
            return;
        }
        let mut landed: Vec<LiveSpray> = Vec::new();
        let mut keep: Vec<LiveSpray> = Vec::with_capacity(self.drops.len());
        let drops = std::mem::take(&mut self.drops);
        for drop in drops {
            let finite = drop.volume.is_finite()
                && drop.pos.iter().all(|c| c.is_finite())
                && drop.vel.iter().all(|c| c.is_finite());
            if !finite {
                // Nothing sane to put back. Keep it booked as spray.
                continue;
            }
            let surface = self.coarse_ij(drop.pos[0], drop.pos[2]).map(|(i, j)| {
                let idx = j * WINDOW_RES + i;
                // A drop well above any film needs no fine lookup.
                let coarse_top = self.bed[idx] + self.water[idx] + self.slurry[idx];
                if drop.pos[1] > coarse_top + 0.5 || !self.fine_on {
                    return coarse_top;
                }
                let s = self.sample(drop.pos[0], drop.pos[2]);
                self.bed[idx] + s.water_m + s.slurry_m
            });
            let down = drop.vel[1] <= 0.0;
            let hit = match surface {
                Some(top) => down && drop.pos[1] <= top,
                None => drop.pos[1] <= 0.02,
            };
            if hit || drop.age >= SPRAY_LIFE_S {
                landed.push(drop);
            } else {
                keep.push(drop);
            }
        }
        self.drops = keep;
        for drop in &landed {
            self.deposit_drop(&drop);
        }
    }

    fn deposit_drop(&mut self, drop: &LiveSpray) {
        let Some((ci, cj)) = self.coarse_ij(drop.pos[0], drop.pos[2]) else {
            // No column to hold it: it stays booked as spray.
            self.spray_stranded_m3 += drop.volume;
            return;
        };
        let cidx = cj * WINDOW_RES + ci;
        // Where the 2 cm sheet holds the column it is the truth: the drop
        // goes there. Elsewhere into the 20 cm column. No restrict, so the
        // fine detail is not flattened by a landing.
        let held = self.fine_on && self.fine_hold.get(cidx).copied() == Some(1);
        let fine_at = if held && self.covers_fine(drop.pos[0], drop.pos[2]) {
            let i = ((drop.pos[0] - self.fine_origin[0]) / FINE_CELL_M).floor() as usize;
            let j = ((drop.pos[2] - self.fine_origin[1]) / FINE_CELL_M).floor() as usize;
            (i < FINE_RES && j < FINE_RES).then_some(j * FINE_RES + i)
        } else {
            None
        };
        // A drop on dry ground (no film in the 20 cm cell) soaks in: it
        // wets the pore water there and is booked in `spray_soaked_m3`.
        // Drops never start a new film on a dry shoulder; that spread a
        // haze of millimetre films the sheet then had to drain (+0.5 ms a
        // drive step on CI).
        if fine_at.is_none()
            && self.water[cidx] + self.slurry[cidx] < SPRAY_SOAK_FILM_M
        {
            let dh = drop.volume / (COARSE_CELL_M * COARSE_CELL_M);
            let m = self.moisture[cidx];
            if m.is_finite() {
                self.moisture[cidx] = (m + dh / SPRAY_SOAK_PORE_M).min(1.0);
            }
            self.spray_soaked_m3 += drop.volume;
            return;
        }
        let (water, uw, vw, area, idx) = match fine_at {
            Some(f) => (
                &mut self.fine_water,
                &mut self.fine_uw,
                &mut self.fine_vw,
                FINE_CELL_M * FINE_CELL_M,
                f,
            ),
            None => (
                &mut self.water,
                &mut self.uw,
                &mut self.vw,
                COARSE_CELL_M * COARSE_CELL_M,
                cidx,
            ),
        };
        let dh = drop.volume / area;
        let h = water[idx].max(0.0);
        let total = h + dh;
        if total > 0.0 {
            // A drop hitting a film loses most of its horizontal momentum
            // in the splash. A share goes into the column, and the result
            // never runs faster than the column did or a slow drift.
            let (u0, v0) = (uw[idx], vw[idx]);
            let mut u = (u0 * h + SPRAY_MOMENTUM_SHARE * drop.vel[0] * dh) / total;
            let mut v = (v0 * h + SPRAY_MOMENTUM_SHARE * drop.vel[2] * dh) / total;
            let cap = u0.hypot(v0).max(SPRAY_LANDING_DRIFT_M_S);
            let sp = u.hypot(v);
            if sp > cap {
                u *= cap / sp;
                v *= cap / sp;
            }
            uw[idx] = u;
            vw[idx] = v;
        }
        water[idx] = total;
    }

    /// Water and slurry thickness, metres. The fine sheet is packed before
    /// `ThroughDrive` mirrors, so the coarse arrays are the film.
    pub(crate) fn film_depth(&self, x: f32, z: f32) -> (f32, f32) {
        if self.fine_on {
            let s = self.sample(x, z);
            return (s.water_m, s.slurry_m);
        }
        if let Some((i, j)) = self.coarse_ij(x, z) {
            let idx = j * WINDOW_RES + i;
            (self.water[idx], self.slurry[idx])
        } else {
            let s = self.sample_coarse(x, z);
            (s.water_m, s.slurry_m)
        }
    }

    pub fn sample(&self, x: f32, z: f32) -> GroundSample {
        if self.fine_on {
            if let Some(s) = self.sample_fine(x, z) {
                return s;
            }
        }
        self.sample_coarse(x, z)
    }

    /// Cubic metres of spray in the air now. The same value as
    /// `ledger().spray_m3`, without summing every sheet. Drops that land go
    /// back to the sheet and leave this account.
    pub fn spray_m3(&self) -> f32 {
        self.spray_m3
    }

    /// Cubic metres the sheet has thrown as spray since it was made,
    /// landed or not. A measure of splash, not a ledger account.
    pub fn spray_thrown_m3(&self) -> f32 {
        self.spray_thrown_m3
    }

    pub fn ledger(&self) -> FluidLedger {
        let mut water = column_sum(&self.water, COARSE_CELL_M);
        let mut slurry = column_sum(&self.slurry, COARSE_CELL_M);
        if self.fine_on {
            water += column_sum(&self.fine_water, FINE_CELL_M);
            slurry += column_sum(&self.fine_slurry, FINE_CELL_M);
        }
        let dust_air = column_sum(&self.dust, self.air_cell());
        let dust_settled = column_sum(&self.settled, COARSE_CELL_M);
        let suspended = column_sum(&self.susp, COARSE_CELL_M);
        let mut frozen_water = 0.0f32;
        let mut frozen_slurry = 0.0f32;
        let mut frozen_dust = 0.0f32;
        let mut frozen_suspended = 0.0f32;
        for p in self.parked.values() {
            let area = COARSE_CELL_M * COARSE_CELL_M;
            frozen_water += p.water * area;
            frozen_slurry += p.slurry * area;
            frozen_dust += p.dust * area;
            frozen_suspended += p.susp * area;
        }
        for d in self.parked_dust.values() {
            frozen_dust += *d;
        }
        let mut fog_air = if self.fog_live {
            column_sum(&self.fog, self.air_cell())
        } else {
            0.0
        };
        for d in self.parked_fog.values() {
            fog_air += *d;
        }
        let total = water
            + slurry
            + dust_air
            + dust_settled
            + frozen_water
            + frozen_slurry
            + frozen_dust
            + fog_air
            + self.reach_bank_m3
            + self.spray_m3
            + self.spray_soaked_m3
            + suspended
            + self.settled_suspended_m3
            + frozen_suspended;
        FluidLedger {
            water_m3: water,
            slurry_m3: slurry,
            dust_air_m3: dust_air,
            dust_settled_m3: dust_settled,
            frozen_water_m3: frozen_water,
            frozen_slurry_m3: frozen_slurry,
            frozen_dust_m3: frozen_dust,
            spray_m3: self.spray_m3,
            spray_soaked_m3: self.spray_soaked_m3,
            fog_air_m3: fog_air,
            suspended_m3: suspended,
            settled_suspended_m3: self.settled_suspended_m3,
            frozen_suspended_m3: frozen_suspended,
            entrained_m3: self.entrained_m3,
            reach_m3: self.reach_bank_m3,
            total_m3: total,
        }
    }

    /// |column + settled + parked − lifted|. Zero when the suspended account closes.
    pub fn suspended_gap(&self) -> f32 {
        let b = self.ledger();
        (b.suspended_m3 + b.settled_suspended_m3 + b.frozen_suspended_m3 - b.entrained_m3).abs()
    }

    /// Cells whose suspended column was booked down onto the water this step.
    ///
    /// A published 1 that did not come from here is a full column (`susp == water`),
    /// not the publish clamp of a deeper one.
    pub fn suspended_capped_cells(&self) -> u32 {
        self.last_susp_capped
    }

    /// Raw `susp/water` on the coarse sheet. This is the value before the
    /// publish clamp, so [`TurbiditySpread::above_water`] is the clamp count.
    pub fn turbidity_spread(&self) -> TurbiditySpread {
        let mut out = TurbiditySpread::default();
        let mut full_water = 0.0f32;
        let n = self.water.len().min(self.susp.len());
        for idx in 0..n {
            let w = self.water[idx];
            if !w.is_finite() || w < 1.0e-5 {
                continue;
            }
            let s = self.susp[idx];
            if !s.is_finite() || s <= 1.0e-8 {
                out.clear += 1;
                continue;
            }
            if s > w + 1.0e-5 {
                out.above_water += 1;
            }
            let ratio = s / w;
            if ratio.is_finite() {
                out.max_ratio = out.max_ratio.max(ratio);
            }
            if ratio >= 1.0 - 1.0e-4 && s <= w + 1.0e-5 {
                out.saturated += 1;
                out.bins[10] += 1;
                full_water += w;
            } else if ratio >= 1.0 {
                out.bins[10] += 1;
                full_water += w;
            } else {
                let b = (ratio.max(0.0) * 10.0).floor() as usize;
                out.bins[b.min(9)] += 1;
            }
        }
        let full_n = out.bins[10];
        if full_n > 0 {
            out.full_water = full_water / full_n as f32;
        }
        out
    }

    pub fn export_maps(&mut self) -> SurfaceMaps {
        self.restrict_fine();
        let n = WINDOW_RES * WINDOW_RES;
        let mut height = vec![0.0f32; n];
        let mut vel_u = vec![0.0f32; n];
        let mut vel_v = vec![0.0f32; n];
        let mut moisture = vec![0.0f32; n];
        let mut dust = vec![0.0f32; n];
        for idx in 0..n {
            height[idx] = self.bed[idx] + self.water[idx] + self.slurry[idx];
            let hw = finite0(self.water[idx]);
            let hs = finite0(self.slurry[idx]);
            let h = hw + hs;
            if h < 1.0e-5 {
                vel_u[idx] = 0.0;
                vel_v[idx] = 0.0;
            } else {
                vel_u[idx] = finite0((self.uw[idx] * hw + self.us[idx] * hs) / h);
                vel_v[idx] = finite0((self.vw[idx] * hw + self.vs[idx] * hs) / h);
            }
            moisture[idx] = self.moisture[idx];
            dust[idx] = self.settled[idx];
        }
        SurfaceMaps {
            res: WINDOW_RES,
            cell: COARSE_CELL_M,
            origin: self.origin,
            height,
            vel_u,
            vel_v,
            moisture,
            dust,
        }
    }

    /// Axis-aligned window on the ground, metres: origin x, origin z, extent.
    pub fn window_bounds(&self) -> (f32, f32, f32) {
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        (self.origin[0], self.origin[1], extent)
    }

    /// Minimum corner of the coarse window around `center`. Same snap as [`Self::new`].
    ///
    /// The render contract uses this corner. Change it only by bumping
    /// `render_contract::RENDER_CONTRACT_VERSION` and recording the change
    /// in `docs/RENDER_CONTRACT.md`.
    pub fn window_origin(center: [f32; 2]) -> [f32; 2] {
        let extent = WINDOW_RES as f32 * COARSE_CELL_M;
        snap_origin(center, extent, COARSE_CELL_M)
    }

    /// One coarse cell after the fine patch has been restricted back.
    ///
    /// `u` and `v` are the thickness-weighted water and slurry velocity.
    /// A dry cell reports zero. A non-finite sample reports zero. This is the
    /// flow the render contract publishes. Change the blend only with a
    /// render-contract version bump.
    pub fn coarse_cell(&self, i: usize, j: usize) -> Option<CoarseCell> {
        if i >= WINDOW_RES || j >= WINDOW_RES {
            return None;
        }
        let idx = j * WINDOW_RES + i;
        let water_m = finite0(self.water[idx]);
        let slurry_m = finite0(self.slurry[idx]);
        let film = water_m + slurry_m;
        let (u, v) = if film < 1.0e-5 {
            (0.0, 0.0)
        } else {
            (
                finite0((self.uw[idx] * water_m + self.us[idx] * slurry_m) / film),
                finite0((self.vw[idx] * water_m + self.vs[idx] * slurry_m) / film),
            )
        };
        Some(CoarseCell {
            x: self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M,
            z: self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M,
            bed_m: finite0(self.bed[idx]),
            water_m,
            slurry_m,
            moisture: finite0(self.moisture[idx]),
            u,
            v,
            settled_dust_m: finite0(self.settled[idx]),
            turbidity: if water_m < 1.0e-5 {
                0.0
            } else {
                // Wheel-lifted solids (#99) plus silt stirred by tire and
                // shore shear (#104): one suspended load in the water.
                ((finite0(self.susp[idx]).max(0.0) + finite0(self.turbid[idx]).max(0.0))
                    / water_m)
                    .clamp(0.0, 1.0)
            },
        })
    }

    /// Air-grid corner and horizontal cell, metres. Corner of texel (0, 0), not the centre.
    pub fn air_window(&self) -> ([f32; 2], f32) {
        (self.air_origin, self.air_cell())
    }

    /// Dust depth in one air slab, metres of equivalent column in that layer.
    /// Index is `ix + layer * AIR_RES * AIR_RES + iz * AIR_RES` in solver storage.
    /// The render contract republishes this as a volume fraction.
    pub fn air_depth(&self, ix: usize, layer: usize, iz: usize) -> f32 {
        if ix >= AIR_RES || iz >= AIR_RES || layer >= AIR_LAYERS {
            return 0.0;
        }
        let idx = (layer * AIR_RES + iz) * AIR_RES + ix;
        self.dust[idx]
    }

    /// Fog depth in one air slab, metres. Same index as [`Self::air_depth`].
    /// The render contract republishes this as a volume fraction. It is not `dust`.
    pub fn fog_depth(&self, ix: usize, layer: usize, iz: usize) -> f32 {
        if ix >= AIR_RES || iz >= AIR_RES || layer >= AIR_LAYERS {
            return 0.0;
        }
        let idx = (layer * AIR_RES + iz) * AIR_RES + ix;
        self.fog[idx]
    }

    /// Horizontal dust-sheet velocity in contract order, vertical channel 0.
    ///
    /// This is not the body wake. The ledger path publishes `Dust::write_air_velocity`
    /// instead. Index `(x + layer * 128 + z * 128 * dst_layers) * 4`, R +X, G +Y,
    /// B +Z, A 0. The sheet's slabs fill layers `0..AIR_LAYERS`; layers above
    /// are left as they are.
    pub fn write_air_velocity(&self, dst: &mut [f32], dst_layers: usize) {
        let n = AIR_RES * AIR_LAYERS * AIR_RES;
        let dst_layers = dst_layers.max(AIR_LAYERS);
        if dst.len() < AIR_RES * dst_layers * AIR_RES * 4 || self.dust_u.len() < n || self.dust_v.len() < n {
            return;
        }
        for iz in 0..AIR_RES {
            for layer in 0..AIR_LAYERS {
                for ix in 0..AIR_RES {
                    let src = (layer * AIR_RES + iz) * AIR_RES + ix;
                    let dst_i = (ix + layer * AIR_RES + iz * AIR_RES * dst_layers) * 4;
                    dst[dst_i] = finite0(self.dust_u[src]);
                    dst[dst_i + 1] = 0.0;
                    dst[dst_i + 2] = finite0(self.dust_v[src]);
                    dst[dst_i + 3] = 0.0;
                }
            }
        }
    }

    /// Loudest airborne columns, then the settled film, up to `limit` marks.
    /// Heights are metres above the bed. This does not move mass.
    pub fn dust_marks(&self, limit: usize) -> Vec<DustMark> {
        if limit == 0 {
            return Vec::new();
        }
        let cell = self.air_cell();
        let mut air: Vec<(f32, usize, usize, usize)> = Vec::new();
        for layer in 0..AIR_LAYERS {
            for j in 0..AIR_RES {
                for i in 0..AIR_RES {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    let h = self.dust[idx];
                    if h > 2.0e-5 {
                        air.push((h, i, j, layer));
                    }
                }
            }
        }
        air.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let air_n = (limit * 2 / 3).max(1).min(limit);
        let mut marks = Vec::with_capacity(limit);
        for (_, i, j, layer) in air.into_iter().take(air_n) {
            let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
            let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
            let ground = self.sample(x, z);
            let idx = (layer * AIR_RES + j) * AIR_RES + i;
            marks.push(DustMark {
                x,
                y: ground.bed_m + ground.water_m + 0.16 + layer as f32 * 0.26,
                z,
                u: self.dust_u[idx],
                v: self.dust_v[idx],
                settled: false,
            });
        }
        let mut film: Vec<(f32, usize)> = Vec::new();
        for (idx, h) in self.settled.iter().enumerate() {
            if *h > 2.0e-5 {
                film.push((*h, idx));
            }
        }
        film.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, idx) in film.into_iter().take(limit.saturating_sub(marks.len())) {
            let i = idx % WINDOW_RES;
            let j = idx / WINDOW_RES;
            let (x, z) = self.coarse_xz(i, j);
            let ground = self.sample(x, z);
            marks.push(DustMark {
                x,
                y: ground.bed_m + ground.water_m + ground.slurry_m + 0.025,
                z,
                u: 0.0,
                v: 0.0,
                settled: true,
            });
        }
        marks
    }

    /// Write water and slurry back onto a coarser or matching ground grid.
    /// `deposit(x, z, water, slurry)` replaces the columns at that point.
    pub fn write_columns(&self, mut deposit: impl FnMut(f32, f32, f32, f32)) {
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let idx = j * WINDOW_RES + i;
                if self.water[idx] <= 1.0e-6 && self.slurry[idx] <= 1.0e-6 {
                    continue;
                }
                let (x, z) = self.coarse_xz(i, j);
                deposit(x, z, self.water[idx], self.slurry[idx]);
            }
        }
    }

    /// Force, buoyancy, and the scour account, read from the coarse sheet.
    pub fn hydro_load(&self) -> HydroLoad {
        let mask = if self.solids.is_empty() {
            vec![0u8; WINDOW_RES * WINDOW_RES]
        } else {
            self.solid_mask(false)
        };
        let mut force_x = 0.0f32;
        let mut force_z = 0.0f32;
        let mut wheel_blocks = 0u32;
        let mut belly_blocks = 0u32;
        let cell = COARSE_CELL_M;
        let res = WINDOW_RES;
        for m in &mask {
            if *m == 1 {
                wheel_blocks += 1;
            } else if *m == 2 {
                belly_blocks += 1;
            }
        }
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                if mask[idx] != 0 {
                    continue;
                }
                let h = self.water[idx];
                if h < 1.0e-4 || !h.is_finite() {
                    continue;
                }
                for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                        continue;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    if mask[nidx] == 0 {
                        continue;
                    }
                    let nx = di as f32;
                    let nz = dj as f32;
                    let hydro = 1000.0 * GRAVITY * h * h * 0.5 * cell;
                    force_x += nx * hydro;
                    force_z += nz * hydro;
                    let vn = self.uw[idx] * nx + self.vw[idx] * nz;
                    if vn > 0.0 {
                        let dyn_f = 0.5 * 1000.0 * vn * vn * h * cell;
                        force_x += nx * dyn_f;
                        force_z += nz * dyn_f;
                    }
                }
            }
        }
        let mut wheel_b = 0.0f32;
        let mut hull_b = 0.0f32;
        let mut displaced = 0.0f32;
        for solid in &self.solids {
            let Some((bed, water)) = self.ring_surface(solid) else {
                continue;
            };
            if solid.radius > 0.0 {
                let vol = cylinder_segment_volume(water, solid.radius, solid.width);
                displaced += vol;
                wheel_b += 1000.0 * GRAVITY * vol;
            } else {
                let eta = bed + water;
                let height = (solid.top_y - solid.bottom_y).max(0.0);
                let sub = (eta - solid.bottom_y).clamp(0.0, height);
                let vol = (solid.half_length * 2.0) * (solid.half_width * 2.0) * sub * 0.36;
                displaced += vol;
                hull_b += 1000.0 * GRAVITY * vol;
            }
        }
        HydroLoad {
            force_x,
            force_z,
            buoyancy_n: wheel_b + hull_b,
            hull_buoyancy_n: hull_b,
            wheel_buoyancy_n: wheel_b,
            displaced_m3: displaced,
            eroded_m3: self.eroded_m3,
            hole_m: self.scour_hole_m,
            wheel_blocks,
            belly_blocks,
        }
    }

    fn ring_surface(&self, solid: &HydroSolid) -> Option<(f32, f32)> {
        let reach = solid.half_length.max(solid.half_width) + 0.55;
        let i0 = ((solid.x - reach - self.origin[0]) / COARSE_CELL_M).floor() as i32;
        let i1 = ((solid.x + reach - self.origin[0]) / COARSE_CELL_M).floor() as i32;
        let j0 = ((solid.z - reach - self.origin[1]) / COARSE_CELL_M).floor() as i32;
        let j1 = ((solid.z + reach - self.origin[1]) / COARSE_CELL_M).floor() as i32;
        let mut bed = 0.0f32;
        let mut water = 0.0f32;
        let mut n = 0.0f32;
        for j in j0..=j1 {
            for i in i0..=i1 {
                if i < 0 || j < 0 || i >= WINDOW_RES as i32 || j >= WINDOW_RES as i32 {
                    continue;
                }
                let (x, z) = self.coarse_xz(i as usize, j as usize);
                if solid_contains(solid, x, z) {
                    continue;
                }
                if (x - solid.x).hypot(z - solid.z) > reach {
                    continue;
                }
                let idx = j as usize * WINDOW_RES + i as usize;
                if self.water[idx] < 0.01 {
                    continue;
                }
                bed += self.bed[idx];
                water += self.water[idx];
                n += 1.0;
            }
        }
        if n < 1.0 {
            None
        } else {
            Some((bed / n, water / n))
        }
    }

    fn solid_mask(&self, fine: bool) -> Vec<u8> {
        let (res, cell, origin, water, bed) = if fine {
            (
                FINE_RES,
                FINE_CELL_M,
                self.fine_origin,
                self.fine_water.as_slice(),
                self.fine_bed.as_slice(),
            )
        } else {
            (
                WINDOW_RES,
                COARSE_CELL_M,
                self.origin,
                self.water.as_slice(),
                self.bed.as_slice(),
            )
        };
        let mut mask = vec![0u8; res * res];
        // Only cells inside a solid's bounding circle can overlap it. Walls
        // first, then contacts, so a contact still wins a shared cell. The
        // per-cell test is the same `solid_overlaps` as a full sweep.
        for contact in [false, true] {
            let value = if contact { 1u8 } else { 2u8 };
            for solid in self.solids.iter().filter(|s| s.contact == contact) {
                if solid.half_length <= 1.0e-4 || solid.half_width <= 1.0e-4 {
                    continue;
                }
                let reach = solid.half_length.max(0.0).hypot(solid.half_width.max(0.0)) + cell;
                if !reach.is_finite() || !solid.x.is_finite() || !solid.z.is_finite() {
                    continue;
                }
                let lo_i = ((solid.x - reach - origin[0]) / cell - 0.5).floor();
                let hi_i = ((solid.x + reach - origin[0]) / cell - 0.5).ceil();
                let lo_j = ((solid.z - reach - origin[1]) / cell - 0.5).floor();
                let hi_j = ((solid.z + reach - origin[1]) / cell - 0.5).ceil();
                let top = (res - 1) as f32;
                if hi_i < 0.0 || hi_j < 0.0 || lo_i > top || lo_j > top {
                    continue;
                }
                let i0 = lo_i.max(0.0) as usize;
                let i1 = hi_i.min(top) as usize;
                let j0 = lo_j.max(0.0) as usize;
                let j1 = hi_j.min(top) as usize;
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        let idx = j * res + i;
                        if mask[idx] == value {
                            continue;
                        }
                        let x = origin[0] + (i as f32 + 0.5) * cell;
                        let z = origin[1] + (j as f32 + 0.5) * cell;
                        // The free surface beside a cell that the body has already
                        // emptied. The wall stays up for as long as that water stands
                        // above the solid's underside.
                        let mut eta = bed[idx] + water[idx];
                        for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
                            let ni = i as i32 + di;
                            let nj = j as i32 + dj;
                            if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                                continue;
                            }
                            let nidx = nj as usize * res + ni as usize;
                            eta = eta.max(bed[nidx] + water[nidx]);
                        }
                        if solid_overlaps(solid, x, z, eta, bed[idx]) {
                            mask[idx] = value;
                        }
                    }
                }
            }
        }
        mask
    }

    /// Excess shear beside a tire contact entrains bed into the slurry sheet.
    fn scour_contacts(&mut self, dt: f32) {
        if dt <= 0.0 || self.solids.is_empty() {
            return;
        }
        let mask = self.solid_mask(false);
        let res = WINDOW_RES;
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let mut cut = vec![0.0f32; res * res];
        for j in 1..res - 1 {
            for i in 1..res - 1 {
                let idx = j * res + i;
                if mask[idx] != 0 || self.water[idx] < 0.02 {
                    continue;
                }
                let mut touch = false;
                for (di, dj) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let nidx = (j as i32 + dj) as usize * res + (i as i32 + di) as usize;
                    if mask[nidx] == 1 {
                        touch = true;
                        break;
                    }
                }
                if !touch {
                    continue;
                }
                let speed = self.uw[idx].hypot(self.vw[idx]);
                if !speed.is_finite() {
                    continue;
                }
                let tau = 1000.0 * 0.03 * speed * speed;
                let tau_c = 1.5f32;
                if tau <= tau_c {
                    continue;
                }
                let rate = (0.0035 * (tau / tau_c - 1.0)).min(0.012);
                let room = (0.08 - self.bed_drop[idx]).max(0.0);
                cut[idx] = (rate * dt).min(room);
            }
        }
        for idx in 0..res * res {
            let dh = cut[idx];
            if dh <= 0.0 {
                continue;
            }
            self.bed[idx] -= dh;
            self.slurry[idx] += dh;
            self.us[idx] = self.uw[idx];
            self.vs[idx] = self.vw[idx];
            self.grain[idx] = Grain::Silt as u8;
            self.moisture[idx] = self.moisture[idx].max(0.9);
            self.bed_drop[idx] += dh;
            self.eroded_m3 += dh * area;
            self.scour_hole_m = self.scour_hole_m.max(self.bed_drop[idx]);
        }
    }

    /// Pull the water velocity in the paddle strip toward the paddle stroke.
    /// The pull fades over the outer quarter of the strip on each side so the
    /// strip edge is not a velocity step.
    fn drive_waves(&mut self, dt: f32) {
        let Some(w) = self.wave else {
            return;
        };
        if dt <= 0.0 || w.period_s <= 0.0 {
            return;
        }
        self.wave_t += dt;
        let len = w.dir_x.hypot(w.dir_z);
        if len < 1.0e-6 || !len.is_finite() {
            return;
        }
        let (dx, dz) = (w.dir_x / len, w.dir_z / len);
        let ramp = if w.ramp_s > 0.0 {
            (self.wave_t / w.ramp_s).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let phase = std::f32::consts::TAU * self.wave_t / w.period_s;
        let stroke = w.amplitude_m_s * ramp * phase.sin();
        // Relaxation toward the stroke, a fraction per step that does not
        // depend on dt for small steps.
        let pull = 1.0 - (-dt / 0.08).exp();
        let res = WINDOW_RES;
        let cell = COARSE_CELL_M;
        let span_x = (w.x1 - w.x0).max(cell);
        let span_z = (w.z1 - w.z0).max(cell);
        let i0 = (((w.x0 - self.origin[0]) / cell).floor() as i32).max(0);
        let i1 = (((w.x1 - self.origin[0]) / cell).ceil() as i32).min(res as i32);
        let j0 = (((w.z0 - self.origin[1]) / cell).floor() as i32).max(0);
        let j1 = (((w.z1 - self.origin[1]) / cell).ceil() as i32).min(res as i32);
        for j in j0..j1 {
            for i in i0..i1 {
                let (x, z) = self.coarse_xz(i as usize, j as usize);
                if x < w.x0 || x > w.x1 || z < w.z0 || z > w.z1 {
                    continue;
                }
                let idx = j as usize * res + i as usize;
                if self.water[idx] < 0.02 || !self.water[idx].is_finite() {
                    continue;
                }
                // Taper on the strip's own axes: 0 at the edge, 1 inside the middle half.
                let tx = ((x - w.x0).min(w.x1 - x) / (0.25 * span_x)).clamp(0.0, 1.0);
                let tz = ((z - w.z0).min(w.z1 - z) / (0.25 * span_z)).clamp(0.0, 1.0);
                let k = pull * tx.min(tz);
                let u = self.uw[idx];
                let v = self.vw[idx];
                let along = u * dx + v * dz;
                let d = (stroke - along) * k;
                let nu = u + d * dx;
                let nv = v + d * dz;
                if nu.is_finite() && nv.is_finite() {
                    self.uw[idx] = nu;
                    self.vw[idx] = nv;
                }
            }
        }
    }

    /// Manning bed friction on the coarse water sheet in open-water mode,
    /// implicit so a thin swash stops instead of reversing:
    /// `du/dt = -g n² |u| u / h^(4/3)`.
    fn bed_friction(&mut self, dt: f32) {
        let Some(o) = self.open_water else {
            return;
        };
        if dt <= 0.0 {
            return;
        }
        self.eddy_shear(o.eddy_m2_s, dt);
        if o.manning_n <= 0.0 {
            return;
        }
        let gn2 = GRAVITY * o.manning_n * o.manning_n;
        for idx in 0..WINDOW_RES * WINDOW_RES {
            let h = self.water[idx];
            if !(h > 1.0e-4) {
                continue;
            }
            let (u, v) = (self.uw[idx], self.vw[idx]);
            let speed = u.hypot(v);
            if !(speed > 0.0) {
                continue;
            }
            // Thin films are capped at 2 mm so the drag stays finite.
            let k = gn2 * speed / h.max(0.002).powf(4.0 / 3.0);
            let keep = 1.0 / (1.0 + dt * k);
            self.uw[idx] = u * keep;
            self.vw[idx] = v * keep;
        }
    }

    /// Horizontal eddy viscosity on the coarse water velocity, one explicit
    /// Laplacian from a snapshot. A dry neighbour that stands more than 3 cm
    /// above the free surface counts as still water (no slip), so a boulder
    /// or a bank drags the stream beside it. A dry neighbour just above the
    /// surface is a beach the front is running onto, and is skipped. Moves no
    /// volume.
    fn eddy_shear(&mut self, nu_t: f32, dt: f32) {
        if nu_t <= 0.0 {
            return;
        }
        const WET: f32 = 0.003;
        const STANDS_OUT: f32 = 0.03;
        let res = WINDOW_RES;
        let n = res * res;
        // Stable for the explicit 5-point Laplacian while k < 0.25.
        let k = (nu_t * dt / (COARSE_CELL_M * COARSE_CELL_M)).min(0.2);
        if self.shift_scratch.len() < 2 * n {
            self.shift_scratch.resize(2 * n, 0.0);
        }
        let (du, dv) = self.shift_scratch[..2 * n].split_at_mut(n);
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                du[idx] = 0.0;
                dv[idx] = 0.0;
                if !(self.water[idx] > WET) {
                    continue;
                }
                let (u0, v0) = (self.uw[idx], self.vw[idx]);
                let mut lu = 0.0f32;
                let mut lv = 0.0f32;
                for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                        // Window edge: no shear across it.
                        continue;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    let (un, vn) = if self.water[nidx] > WET {
                        (self.uw[nidx], self.vw[nidx])
                    } else if self.bed[nidx] > self.bed[idx] + self.water[idx] + STANDS_OUT {
                        // Ground standing out of the water: no slip.
                        (0.0, 0.0)
                    } else {
                        // A wet front running onto the beach: free slip.
                        continue;
                    };
                    lu += un - u0;
                    lv += vn - v0;
                }
                du[idx] = k * lu;
                dv[idx] = k * lv;
            }
        }
        for idx in 0..n {
            if du[idx] != 0.0 || dv[idx] != 0.0 {
                let nu = self.uw[idx] + du[idx];
                let nv = self.vw[idx] + dv[idx];
                if nu.is_finite() && nv.is_finite() {
                    self.uw[idx] = nu;
                    self.vw[idx] = nv;
                }
            }
        }
    }

    /// See [`Self::set_surface_diffusion`]. Fluxes are computed from one
    /// snapshot of the surface, then applied, so the sweep order does not matter.
    fn diffuse_surface(&mut self, dt: f32) {
        if self.surface_diffusion <= 0.0 || dt <= 0.0 {
            return;
        }
        let res = WINDOW_RES;
        let k = (self.surface_diffusion * dt / (COARSE_CELL_M * COARSE_CELL_M)).min(0.12);
        const WET: f32 = 0.003;
        let n = res * res;
        if self.shift_scratch.len() < n {
            self.shift_scratch.resize(n, 0.0);
        }
        let delta = &mut self.shift_scratch[..n];
        delta.fill(0.0);
        for j in 0..res {
            for i in 0..res {
                let a = j * res + i;
                let ha = self.water[a];
                if !(ha > WET) {
                    continue;
                }
                let ea = self.bed[a] + ha;
                // East and north faces; each face is visited once.
                for b in [
                    if i + 1 < res { Some(a + 1) } else { None },
                    if j + 1 < res { Some(a + res) } else { None },
                ]
                .into_iter()
                .flatten()
                {
                    let hb = self.water[b];
                    if !(hb > WET) {
                        continue;
                    }
                    let eb = self.bed[b] + hb;
                    let mut f = k * (ea - eb);
                    // Donor limit: a face never takes more than a quarter of the column.
                    if f > 0.0 {
                        f = f.min(0.25 * ha);
                    } else {
                        f = f.max(-0.25 * hb);
                    }
                    delta[a] -= f;
                    delta[b] += f;
                }
            }
        }
        for idx in 0..n {
            let d = delta[idx];
            if d != 0.0 && d.is_finite() {
                self.water[idx] = (self.water[idx] + d).max(0.0);
            }
        }
    }

    /// Outlet to bank, bank to inlet. See [`RiverReach`].
    fn feed_reach(&mut self) {
        let Some(r) = self.reach else {
            return;
        };
        let res = WINDOW_RES;
        let rows = r.rows.clamp(1, res / 4);
        let cell = COARSE_CELL_M;
        let area = cell * cell;
        let cross_origin = if r.along_z {
            self.origin[0]
        } else {
            self.origin[1]
        };
        let c0 = (((r.cross0 - cross_origin) / cell).floor() as i32).max(0) as usize;
        let c1 = (((r.cross1 - cross_origin) / cell).ceil() as i32).clamp(0, res as i32) as usize;
        if c1 <= c0 {
            return;
        }
        let at = |along: usize, cross: usize| -> usize {
            if r.along_z {
                along * res + cross
            } else {
                cross * res + along
            }
        };
        // Along-stream rows, upstream first.
        let row = |k: usize| -> usize {
            if r.positive {
                k
            } else {
                res - 1 - k
            }
        };
        let (dx, dz) = match (r.along_z, r.positive) {
            (false, true) => (1.0f32, 0.0f32),
            (false, false) => (-1.0, 0.0),
            (true, true) => (0.0, 1.0),
            (true, false) => (0.0, -1.0),
        };
        // Outlet: hold the stage, copy the interior velocity, bank the excess.
        let first_out = res - rows;
        for cross in c0..c1 {
            let src = at(row(first_out - 1), cross);
            let (su, sv) = (self.uw[src], self.vw[src]);
            let along = (su * dx + sv * dz).max(0.0);
            for k in first_out..res {
                let idx = at(row(k), cross);
                let target = (r.outlet_eta - self.bed[idx]).max(0.0);
                let h = self.water[idx];
                if !h.is_finite() || target <= 0.0 {
                    continue;
                }
                if h > target {
                    self.reach_bank_m3 += (h - target) * area;
                    self.water[idx] = target;
                } else if self.reach_bank_m3 > 0.0 {
                    let add = (target - h).min(self.reach_bank_m3 / area);
                    self.water[idx] = h + add;
                    self.reach_bank_m3 -= add * area;
                }
                if su.is_finite() && sv.is_finite() {
                    let cross_v = if r.along_z { su } else { sv };
                    if r.along_z {
                        self.uw[idx] = cross_v;
                        self.vw[idx] = along * dz;
                    } else {
                        self.uw[idx] = along * dx;
                        self.vw[idx] = cross_v;
                    }
                }
            }
        }
        // Inlet: fill from the bank toward the stage and push the water in.
        let inflow_c = if self.turbidity_enabled && r.inlet_turbidity.is_finite() {
            r.inlet_turbidity.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut inflow_m3 = 0.0f64;
        let mut deep = 0.0f32;
        for cross in c0..c1 {
            let idx = at(row(0), cross);
            deep = deep.max((r.inlet_eta - self.bed[idx]).max(0.0));
        }
        if deep <= 0.0 {
            return;
        }
        for k in 0..rows {
            for cross in c0..c1 {
                let idx = at(row(k), cross);
                let target = (r.inlet_eta - self.bed[idx]).max(0.0);
                let h = self.water[idx];
                if !h.is_finite() || target <= 0.0 {
                    continue;
                }
                if h > target {
                    self.reach_bank_m3 += (h - target) * area;
                    self.water[idx] = target;
                } else if self.reach_bank_m3 > 0.0 {
                    let add = (target - h).min(self.reach_bank_m3 / area);
                    self.water[idx] = h + add;
                    self.reach_bank_m3 -= add * area;
                }
                let speed = r.inlet_speed_m_s * (self.water[idx] / deep).clamp(0.0, 1.0).sqrt();
                self.uw[idx] = speed * dx;
                self.vw[idx] = speed * dz;
                if inflow_c > 0.0 {
                    let w = self.water[idx];
                    let want = inflow_c * w;
                    let have = self.susp[idx];
                    if w >= 1.0e-5 && have.is_finite() && have < want {
                        self.susp[idx] = want;
                        inflow_m3 += f64::from((want - have) * area);
                    }
                }
            }
        }
        self.entrained_m3 += inflow_m3 as f32;
    }

    /// Hold `fraction` of the water column as suspended solids on every wet
    /// coarse cell inside the box, as if the water had arrived carrying it.
    /// Solids added are booked as `entrained_m3`. Returns cubic metres added.
    pub fn suspend_uniform(&mut self, fraction: f32, x0: f32, x1: f32, z0: f32, z1: f32) -> f32 {
        let c = if fraction.is_finite() { fraction.clamp(0.0, 1.0) } else { 0.0 };
        if c <= 0.0 {
            return 0.0;
        }
        let area = COARSE_CELL_M * COARSE_CELL_M;
        let mut added = 0.0f64;
        for j in 0..WINDOW_RES {
            let z = self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M;
            if z < z0 || z > z1 {
                continue;
            }
            for i in 0..WINDOW_RES {
                let x = self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M;
                if x < x0 || x > x1 {
                    continue;
                }
                let idx = j * WINDOW_RES + i;
                let w = self.water[idx];
                if !w.is_finite() || w < 1.0e-5 {
                    continue;
                }
                let want = c * w;
                let have = self.susp[idx];
                if have.is_finite() && have < want {
                    self.susp[idx] = want;
                    added += f64::from((want - have) * area);
                }
            }
        }
        self.entrained_m3 += added as f32;
        added as f32
    }

    /// Saturate the ground under the film and drain it once the film has gone.
    fn soak_shore(&mut self, dt: f32) {
        let Some(soak) = self.soak else {
            return;
        };
        if dt <= 0.0 {
            return;
        }
        let pad = self.moisture_ambient;
        let base = soak.drain_s.max(1.0e-3);
        let keep_sand = (-dt / base).exp();
        let keep_silt = (-dt / (3.0 * base)).exp();
        let keep_clay = (-dt / (8.0 * base)).exp();
        for idx in 0..WINDOW_RES * WINDOW_RES {
            let film = self.water[idx] + self.slurry[idx];
            let m = self.moisture[idx];
            if film >= soak.wet_film_m {
                self.moisture[idx] = 1.0;
                continue;
            }
            if m <= pad || !m.is_finite() {
                continue;
            }
            let keep = match Grain::from_u8(self.grain[idx]) {
                Grain::Sand => keep_sand,
                Grain::Silt | Grain::Other | Grain::Water => keep_silt,
                Grain::Clay => keep_clay,
            };
            self.moisture[idx] = pad + (m - pad) * keep;
        }
    }

    /// Recycle a reach. The downstream lip copies the flowing interior so the
    /// bank does not reflect, and the volume that left is put back on the
    /// upstream lip. No lip, no change.
    fn cycle_stream(&mut self) {
        let Some(cyc) = self.cycle else {
            return;
        };
        let along_z = cyc.dir_z.abs() >= cyc.dir_x.abs();
        let cell = COARSE_CELL_M;
        let res = WINDOW_RES;
        let lip = |p: f32, origin: f32| -> i32 { ((p - origin) / cell).floor() as i32 };
        if along_z {
            let j_up = lip(cyc.z0, self.origin[1]);
            let j_dn = lip(cyc.z1, self.origin[1]);
            // Reference is a few cells inside the upstream lip, away from the
            // car and away from the bank. The last four rows copy it.
            let (j_ref, j_out0, j_out1, j_inlet) = if cyc.dir_z >= 0.0 {
                (j_up + 6, j_dn - 3, j_dn, j_up)
            } else {
                (j_dn - 6, j_up, j_up + 3, j_dn)
            };
            if j_up < 1
                || j_dn >= res as i32 - 1
                || j_ref <= j_out1 && j_ref >= j_out0
                || j_out0 <= j_up
            {
                return;
            }
            let i0 = lip(cyc.x0, self.origin[0]);
            let i1 = lip(cyc.x1, self.origin[0]);
            for i in i0..i1 {
                if i < 0 || i >= res as i32 {
                    continue;
                }
                self.recycle_column(
                    i as usize,
                    j_ref as usize,
                    j_out0 as usize,
                    j_out1 as usize,
                    j_inlet as usize,
                    true,
                );
            }
        } else {
            let i_up = lip(cyc.x0, self.origin[0]);
            let i_dn = lip(cyc.x1, self.origin[0]);
            let (i_ref, i_out0, i_out1, i_inlet) = if cyc.dir_x >= 0.0 {
                (i_up + 6, i_dn - 3, i_dn, i_up)
            } else {
                (i_dn - 6, i_up, i_up + 3, i_dn)
            };
            if i_up < 1 || i_dn >= res as i32 - 1 || i_out0 <= i_up {
                return;
            }
            let j0 = lip(cyc.z0, self.origin[1]);
            let j1 = lip(cyc.z1, self.origin[1]);
            for j in j0..j1 {
                if j < 0 || j >= res as i32 {
                    continue;
                }
                self.recycle_column(
                    j as usize,
                    i_ref as usize,
                    i_out0 as usize,
                    i_out1 as usize,
                    i_inlet as usize,
                    false,
                );
            }
        }
    }

    fn recycle_column(
        &mut self,
        cross: usize,
        interior: usize,
        out0: usize,
        out1: usize,
        inlet: usize,
        along_z: bool,
    ) {
        let res = WINDOW_RES;
        let at = |a: usize, b: usize| -> usize {
            if along_z {
                a * res + b
            } else {
                b * res + a
            }
        };
        let src = at(interior, cross);
        if self.water[src] < 0.02 || !self.water[src].is_finite() {
            return;
        }
        let mut surplus = 0.0f32;
        for out in out0..=out1 {
            let idx = at(out, cross);
            if self.water[idx].is_finite() {
                surplus += self.water[idx] - self.water[src];
            }
            self.water[idx] = self.water[src];
            self.uw[idx] = self.uw[src];
            self.vw[idx] = self.vw[src];
        }
        let inn = at(inlet, cross);
        let next = (self.water[inn] + surplus).max(0.0);
        if next.is_finite() {
            self.water[inn] = next;
        }
        if self.vw[src].abs() + self.uw[src].abs() > 0.05 {
            self.uw[inn] = self.uw[src];
            self.vw[inn] = self.vw[src];
        }
    }

    fn step_pair(&mut self, dt: f32, fine: bool) {
        if fine && !self.fine_on {
            return;
        }
        let prep = ProfTick::start();
        let cap = if fine { self.fine_sub_cap } else { 8.0 };
        let pace = if fine { self.fine_pace } else { 0 };
        let (res, cell) = if fine {
            (FINE_RES, FINE_CELL_M)
        } else {
            (WINDOW_RES, COARSE_CELL_M)
        };
        let mask = if fine && !self.fine_on {
            None
        } else if self.solids.is_empty() && self.cycle.is_none() {
            None
        } else if self.solids.is_empty() {
            let n = if fine {
                FINE_RES * FINE_RES
            } else {
                WINDOW_RES * WINDOW_RES
            };
            Some(vec![0u8; n])
        } else {
            Some(self.solid_mask(fine))
        };
        let n = res * res;
        let lift: &[(u32, f32)] = if fine {
            &self.wet.lift_f
        } else {
            &self.wet.lift_c
        };
        PAIR_SCRATCH.with(|slot| {
            let mut scratch = slot.borrow_mut();
            scratch.ensure();
            let water_ms;
            let slurry_ms;
            let prep_ms;
            {
                let (water, slurry, uw, vw, us, vs, bed_s, moisture, grain) = if fine {
                    (
                        &mut self.fine_water,
                        &mut self.fine_slurry,
                        &mut self.fine_uw,
                        &mut self.fine_vw,
                        &mut self.fine_us,
                        &mut self.fine_vs,
                        &self.fine_bed,
                        &self.fine_moisture,
                        &self.fine_grain,
                    )
                } else {
                    (
                        &mut self.water,
                        &mut self.slurry,
                        &mut self.uw,
                        &mut self.vw,
                        &mut self.us,
                        &mut self.vs,
                        &self.bed,
                        &self.moisture,
                        &self.grain,
                    )
                };
                scratch.bed[..n].copy_from_slice(&bed_s[..n]);
                // Bodies in the sheet raise the floor this step reads, by
                // the column they fill. The water itself is not touched.
                for &(i, o) in lift {
                    scratch.bed[i as usize] += o;
                }
                // The water viscosity row is shared by every sheet on this
                // thread. An open reach writes its mobility and puts the
                // water value back after the step.
                if let Some(o) = self.open_water {
                    scratch.nu_w[..n].fill(-o.mobility.max(1.0e-4));
                }
                // An open reach keeps its current through the window ends.
                let open = if fine {
                    None
                } else {
                    self.reach.map(|r| {
                        let o = if r.along_z {
                            self.origin[0]
                        } else {
                            self.origin[1]
                        };
                        let c0 = ((r.cross0 - o) / cell).floor().max(0.0) as usize;
                        let c1 = ((r.cross1 - o) / cell).ceil().max(0.0) as usize;
                        (r.along_z, c0, c1.min(res))
                    })
                };
                prep_ms = prep.ms();
                let PairScratch {
                    nu_w,
                    nu_s,
                    gate,
                    bed,
                } = &mut *scratch;
                let bed = &bed[..n];
                let bed_slurry: &[f32] = if fine { &self.fine_bed_slurry[..n] } else { bed };
                let nu_w = &nu_w[..n];
                let nu_s = &mut nu_s[..n];
                let gate = &gate[..n];
                // Water and slurry are separate sheets: their depth and velocity
                // arrays do not alias, and both only read the bed, gate, and
                // mask. Each one steps exactly as it did in series, on its own
                // worker with its own scratch. The per-sheet preparation (the
                // slurry viscosity, the soft window edge, the body displacement)
                // runs inside each worker too.
                let solid = mask.as_deref();
                let [fine_w, fine_s, coarse_w, coarse_s] = &mut self.sheet_scratch;
                let ([sc_w, sc_w2], [sc_s, sc_s2]) =
                    if fine { (fine_w, fine_s) } else { (coarse_w, coarse_s) };
                let run_water = || {
                    let tw = ProfTick::start();
                    // Soft edge of the window: slow the velocity, keep the mass.
                    if !fine {
                        soften_border(uw, vw, res, open);
                    }
                    if let Some(mask) = solid {
                        displace_occupied(water, uw, vw, mask, res);
                    }
                    step_sheet_eps_in(
                        water,
                        uw,
                        vw,
                        bed,
                        nu_w,
                        gate,
                        res,
                        cell,
                        dt,
                        SheetDrive::Gravity { vorticity: 0.08 },
                        1.0e-6,
                        solid,
                        false,
                        cap,
                        pace,
                        sc_w,
                        sc_w2,
                    );
                    tw.ms()
                };
                let run_slurry = || {
                    let ts = ProfTick::start();
                    for idx in 0..n {
                        nu_s[idx] = dynamic_viscosity(Grain::from_u8(grain[idx]), moisture[idx]);
                    }
                    if !fine {
                        soften_border(us, vs, res, open);
                    }
                    if let Some(mask) = solid {
                        displace_occupied(slurry, us, vs, mask, res);
                    }
                    let nu_s = &*nu_s;
                    step_sheet_eps_in(
                        slurry,
                        us,
                        vs,
                        bed_slurry,
                        nu_s,
                        gate,
                        res,
                        cell,
                        dt,
                        SheetDrive::Gravity { vorticity: 0.05 },
                        1.0e-6,
                        solid,
                        false,
                        cap,
                        pace,
                        sc_s,
                        sc_s2,
                    );
                    ts.ms()
                };
                #[cfg(not(target_arch = "wasm32"))]
                {
                    (water_ms, slurry_ms) =
                        rayon::join(|| serial_inner(run_water), || serial_inner(run_slurry));
                }
                #[cfg(target_arch = "wasm32")]
                {
                    // The closures borrow the sheets mutably, so they are FnMut;
                    // native code moves them into rayon::join, here they need a
                    // mutable binding to be called in sequence.
                    let (mut run_water, mut run_slurry) = (run_water, run_slurry);
                    water_ms = run_water();
                    slurry_ms = run_slurry();
                }
            }
            if self.open_water.is_some() {
                let n = WINDOW_RES * WINDOW_RES;
                scratch.nu_w[..n].fill(dynamic_viscosity(Grain::Water, 1.0));
            }
            self.sheet_times.prep_ms += prep_ms;
            self.sheet_times.water_ms += water_ms;
            self.sheet_times.slurry_ms += slurry_ms;
        });
    }

    fn apply_pushes(&mut self, pushes: &[TirePush], fine: bool) {
        for p in pushes {
            if fine && self.cell_is_refined(p.x, p.z) {
                self.push_on(p, true);
            }
        }
    }

    fn apply_pushes_outside(&mut self, pushes: &[TirePush]) {
        for p in pushes {
            if self.fine_held_at(p.x, p.z) {
                continue;
            }
            self.push_on(p, false);
        }
    }

    fn push_on(&mut self, p: &TirePush, fine: bool) {
        let (res, cell, origin, water, slurry, uw, vw, us, vs) = if fine {
            (
                FINE_RES,
                FINE_CELL_M,
                self.fine_origin,
                &mut self.fine_water,
                &mut self.fine_slurry,
                &mut self.fine_uw,
                &mut self.fine_vw,
                &mut self.fine_us,
                &mut self.fine_vs,
            )
        } else {
            (
                WINDOW_RES,
                COARSE_CELL_M,
                self.origin,
                &mut self.water,
                &mut self.slurry,
                &mut self.uw,
                &mut self.vw,
                &mut self.us,
                &mut self.vs,
            )
        };
        let fx = p.forward_x;
        let fz = p.forward_z;
        let len = (fx * fx + fz * fz).sqrt().max(1.0e-4);
        let fx = fx / len;
        let fz = fz / len;
        let rx = fz;
        let rz = -fx;
        let speed = p.speed.max(0.0);
        let reach = (p.radius * 1.4).hypot(p.width * 0.65 + cell) + cell;
        let (i0, i1) = index_span(origin[0], cell, res, p.x, reach);
        let (j0, j1) = index_span(origin[1], cell, res, p.z, reach);
        for j in j0..j1 {
            for i in i0..i1 {
                let x = origin[0] + (i as f32 + 0.5) * cell;
                let z = origin[1] + (j as f32 + 0.5) * cell;
                let dx = x - p.x;
                let dz = z - p.z;
                let along = dx * fx + dz * fz;
                let lat = dx * rx + dz * rz;
                if lat.abs() > p.width * 0.65 + cell {
                    continue;
                }
                if along < -p.radius * 1.4 || along > p.radius * 1.1 {
                    continue;
                }
                let idx = j * res + i;
                let flank = (lat.abs() / (p.width * 0.65 + cell)).clamp(0.0, 1.0);
                let shove = speed.max(0.4) * (1.0 - flank * flank);
                if water[idx] > 0.004 {
                    uw[idx] += fx * shove * 0.85;
                    vw[idx] += fz * shove * 0.85;
                    uw[idx] += rx * lat.signum() * shove * 0.18 * flank;
                    vw[idx] += rz * lat.signum() * shove * 0.18 * flank;
                }
                if slurry[idx] > 0.004 {
                    // Stretch behind the tire, shove ahead, part the shoulders.
                    let fore = if along > 0.0 { 1.0 } else { 0.72 };
                    us[idx] += fx * shove * fore;
                    vs[idx] += fz * shove * fore;
                    us[idx] += rx * lat.signum() * shove * 0.28 * flank;
                    vs[idx] += rz * lat.signum() * shove * 0.28 * flank;
                }
            }
        }
    }

    fn peel_spray(&mut self, pushes: &[TirePush], fine: bool, skip_held: bool) -> Vec<SprayEvent> {
        let mut out = Vec::new();
        let hold = if skip_held {
            self.fine_hold.clone()
        } else {
            Vec::new()
        };
        let coarse_origin = self.origin;
        let (res, cell, origin, water, uw, vw, bed, slurry) = if fine {
            (
                FINE_RES,
                FINE_CELL_M,
                self.fine_origin,
                &mut self.fine_water,
                &self.fine_uw,
                &self.fine_vw,
                &self.fine_bed,
                &self.fine_slurry,
            )
        } else {
            (
                WINDOW_RES,
                COARSE_CELL_M,
                self.origin,
                &mut self.water,
                &self.uw,
                &self.vw,
                &self.bed,
                &self.slurry,
            )
        };
        if pushes.is_empty() {
            return out;
        }
        let area = cell * cell;
        for p in pushes {
            if p.speed < 0.8 {
                continue;
            }
            let reach = p.radius * 1.3 + cell;
            let (i0, i1) = index_span(origin[0], cell, res, p.x, reach);
            let (j0, j1) = index_span(origin[1], cell, res, p.z, reach);
            for j in j0..j1 {
                for i in i0..i1 {
                    let x = origin[0] + (i as f32 + 0.5) * cell;
                    let z = origin[1] + (j as f32 + 0.5) * cell;
                    if skip_held && coarse_held(&hold, coarse_origin, x, z) {
                        continue;
                    }
                    let dx = x - p.x;
                    let dz = z - p.z;
                    if dx * dx + dz * dz > (p.radius * 1.3) * (p.radius * 1.3) {
                        continue;
                    }
                    let idx = j * res + i;
                    if water[idx] < 0.01 {
                        continue;
                    }
                    let take = (water[idx] * SPRAY_PEEL_SHARE).min(SPRAY_PEEL_MAX_M);
                    if take * area < 1.0e-6 {
                        continue;
                    }
                    water[idx] -= take;
                    // Hybrid H2, 2D -> 3D. A rolling tire throws water two
                    // ways: the bow shoves it out sideways in two wings,
                    // and the tread behind the axle flings a rooster tail
                    // back and up. The drop also keeps the column velocity.
                    // A fountain straight up (old vy 1.4 + 0.25 v, vx 0.3 u)
                    // drew as a tall vertical stripe.
                    let surface = bed[idx] + water[idx];
                    let (fx, fz) = (p.forward_x, p.forward_z);
                    let along = dx * fx + dz * fz;
                    let lat = dx * -fz + dz * fx;
                    let (lx, lz) = if lat >= 0.0 { (-fz, fx) } else { (fz, -fx) };
                    let j1 = 0.75 + 0.5 * spray_hash(idx as u32 ^ 0x51ed);
                    let j2 = 0.75 + 0.5 * spray_hash(idx as u32 ^ 0x9e37);
                    let v = p.speed;
                    let (out_l, out_f, up) = if along >= 0.0 {
                        // Bow wings: out and up at about half the rim speed.
                        (0.50 * v * j1, 0.15 * v, 0.55 * v * j2)
                    } else {
                        // Rooster tail off the tread.
                        (0.10 * v * j1, -0.35 * v, 0.60 * v * j2)
                    };
                    let liquid = water[idx] + take + slurry[idx];
                    let murk = if liquid > 1.0e-6 {
                        (slurry[idx] / liquid).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    out.push(SprayEvent {
                        x,
                        y: surface + 0.03,
                        z,
                        vx: uw[idx] + lx * out_l + fx * out_f,
                        vy: up,
                        vz: vw[idx] + lz * out_l + fz * out_f,
                        volume: take * area,
                        kind: 0,
                        murk,
                    });
                }
            }
        }
        out
    }

    fn fine_rect(&self) -> Option<[f32; 4]> {
        if !self.fine_on {
            return None;
        }
        let extent = FINE_RES as f32 * FINE_CELL_M;
        Some([
            self.fine_origin[0],
            self.fine_origin[0] + extent,
            self.fine_origin[1],
            self.fine_origin[1] + extent,
        ])
    }

    fn fine_held_at(&self, x: f32, z: f32) -> bool {
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return false;
        };
        self.fine_hold.get(j * WINDOW_RES + i).copied() == Some(1)
    }

    /// A coarse cell is refined only when the fine patch covers it completely
    /// and it sits under a tire, so the hand-off does not drop the uncovered part.
    fn cell_is_refined(&self, x: f32, z: f32) -> bool {
        if !self.fine_on || !self.fine_held_at(x, z) {
            return false;
        }
        let Some((i, j)) = self.coarse_ij(x, z) else {
            return false;
        };
        let x0 = self.origin[0] + i as f32 * COARSE_CELL_M;
        let z0 = self.origin[1] + j as f32 * COARSE_CELL_M;
        let x1 = x0 + COARSE_CELL_M;
        let z1 = z0 + COARSE_CELL_M;
        let Some(rect) = self.fine_rect() else {
            return false;
        };
        x0 >= rect[0] - 1.0e-4
            && z0 >= rect[2] - 1.0e-4
            && x1 <= rect[1] + 1.0e-4
            && z1 <= rect[3] + 1.0e-4
    }

    fn refine(&mut self, pushes: &[TirePush], dt: f32) {
        self.restrict_fine();
        if pushes.is_empty() {
            self.near_fade(dt);
            return;
        }
        let mut cx = 0.0f32;
        let mut cz = 0.0f32;
        for p in pushes {
            cx += p.x;
            cz += p.z;
        }
        let n = pushes.len() as f32;
        cx /= n;
        cz /= n;
        let extent = FINE_RES as f32 * FINE_CELL_M;
        self.fine_origin = snap_origin([cx, cz], extent, COARSE_CELL_M);
        self.fine_water.fill(0.0);
        self.fine_slurry.fill(0.0);
        self.fine_uw.fill(0.0);
        self.fine_vw.fill(0.0);
        self.fine_us.fill(0.0);
        self.fine_vs.fill(0.0);
        self.fine_hold.fill(0);
        let full = ((COARSE_CELL_M / FINE_CELL_M).round() as u32).pow(2);
        let r2 = FINE_TIRE_R_M * FINE_TIRE_R_M;
        let (col, row) = self.fine_coarse_axes();
        COLUMN_ACC.with(|slot| {
            let mut acc = slot.borrow_mut();
            acc.begin();
            for j in 0..FINE_RES {
                let cj = row[j];
                if cj == u16::MAX {
                    continue;
                }
                for i in 0..FINE_RES {
                    let ci = col[i];
                    if ci == u16::MAX {
                        continue;
                    }
                    let cidx = cj as usize * WINDOW_RES + ci as usize;
                    acc.mark(cidx);
                    acc.hits.push((j * FINE_RES + i, cidx));
                }
            }
            let ntouch = acc.touch.len();
            for t in 0..ntouch {
                let cidx = acc.touch[t];
                if acc.cnt[cidx] != full {
                    continue;
                }
                let ci = cidx % WINDOW_RES;
                let cj = cidx / WINDOW_RES;
                let (x, z) = self.coarse_xz(ci, cj);
                if near_tire_zone(pushes, x, z, r2) {
                    self.fine_hold[cidx] = 1;
                }
            }
            let nhit = acc.hits.len();
            for h in 0..nhit {
                let (fidx, cidx) = acc.hits[h];
                self.fine_moisture[fidx] = self.moisture[cidx];
                self.fine_grain[fidx] = self.grain[cidx];
                if self.fine_hold[cidx] != 1 {
                    // A fine cell the patch does not hold carries no fluid. Its
                    // ground is the free surface of its coarse column (water and
                    // slurry each at their own), not the bed left from an older
                    // patch position. Dry ground there was a slope the held edge
                    // drained down every step: in deep water a hole followed the
                    // patch. Fluid a tire throws over the rim still lands there
                    // and goes back to that coarse column.
                    self.fine_bed[fidx] = self.bed[cidx] + self.water[cidx];
                    self.fine_bed_slurry[fidx] = self.bed[cidx] + self.slurry[cidx];
                    continue;
                }
                self.fine_bed[fidx] = self.bed[cidx];
                self.fine_bed_slurry[fidx] = self.bed[cidx];
                self.fine_water[fidx] = self.water[cidx];
                self.fine_slurry[fidx] = self.slurry[cidx];
                self.fine_uw[fidx] = self.uw[cidx];
                self.fine_vw[fidx] = self.vw[cidx];
                self.fine_us[fidx] = self.us[cidx];
                self.fine_vs[fidx] = self.vs[cidx];
                self.fine_bed[fidx] = self.bed[cidx];
                self.fine_moisture[fidx] = self.moisture[cidx];
                self.fine_grain[fidx] = self.grain[cidx];
            }
            for t in 0..ntouch {
                let cidx = acc.touch[t];
                if self.fine_hold[cidx] == 1 {
                    self.water[cidx] = 0.0;
                    self.slurry[cidx] = 0.0;
                    self.uw[cidx] = 0.0;
                    self.vw[cidx] = 0.0;
                    self.us[cidx] = 0.0;
                    self.vs[cidx] = 0.0;
                }
            }
            acc.end_clear();
        });
        self.near_apply(dt);
        self.fine_on = true;
    }

    fn restrict_fine(&mut self) {
        if !self.fine_on {
            return;
        }
        let tick = ProfTick::start();
        self.near_capture();
        let fine_area = FINE_CELL_M * FINE_CELL_M;
        let coarse_area = COARSE_CELL_M * COARSE_CELL_M;
        COLUMN_ACC.with(|slot| {
            let mut acc = slot.borrow_mut();
            let (col, row) = self.fine_coarse_axes();
            acc.begin();
            for j in 0..FINE_RES {
                let cj = row[j];
                for i in 0..FINE_RES {
                    let ci = col[i];
                    let fidx = j * FINE_RES + i;
                    if cj == u16::MAX || ci == u16::MAX {
                        let x = self.fine_origin[0] + (i as f32 + 0.5) * FINE_CELL_M;
                        let z = self.fine_origin[1] + (j as f32 + 0.5) * FINE_CELL_M;
                        // Outside the coarse window: park the fine column's volume.
                        let key = world_key(x, z, COARSE_CELL_M);
                        let slot = self.parked.entry(key).or_default();
                        slot.water += self.fine_water[fidx] * fine_area / coarse_area;
                        slot.slurry += self.fine_slurry[fidx] * fine_area / coarse_area;
                        continue;
                    }
                    let cidx = cj as usize * WINDOW_RES + ci as usize;
                    acc.mark(cidx);
                    let (fw, fs) = (self.fine_water[fidx], self.fine_slurry[fidx]);
                    // A dry fine cell adds +0.0 to every sum, which leaves it
                    // unchanged. Only its count matters for the full-column test.
                    if fw == 0.0 && fs == 0.0 {
                        continue;
                    }
                    acc.w[cidx] += self.fine_water[fidx] * fine_area;
                    acc.s[cidx] += self.fine_slurry[fidx] * fine_area;
                    acc.uw[cidx] += self.fine_uw[fidx] * self.fine_water[fidx] * fine_area;
                    acc.vw[cidx] += self.fine_vw[fidx] * self.fine_water[fidx] * fine_area;
                    acc.us[cidx] += self.fine_us[fidx] * self.fine_slurry[fidx] * fine_area;
                    acc.vs[cidx] += self.fine_vs[fidx] * self.fine_slurry[fidx] * fine_area;
                }
            }
            let full = ((COARSE_CELL_M / FINE_CELL_M).round() as u32).pow(2);
            let ntouch = acc.touch.len();
            for t in 0..ntouch {
                let idx = acc.touch[t];
                let held = self.fine_hold.get(idx).copied() == Some(1);
                let w = acc.w[idx] / coarse_area;
                let s = acc.s[idx] / coarse_area;
                // A coarse cell that stayed off the 2 cm sheet keeps its depth.
                // Only volume that flowed onto a fine cell of that column is added.
                if !held && acc.cnt[idx] >= full {
                    if w <= 0.0 && s <= 0.0 {
                        continue;
                    }
                    let old_w = self.water[idx];
                    let old_s = self.slurry[idx];
                    let new_w = old_w + w;
                    let new_s = old_s + s;
                    if new_w > 1.0e-6 {
                        self.uw[idx] = (old_w * self.uw[idx] * coarse_area + acc.uw[idx])
                            / (new_w * coarse_area);
                        self.vw[idx] = (old_w * self.vw[idx] * coarse_area + acc.vw[idx])
                            / (new_w * coarse_area);
                    }
                    if new_s > 1.0e-6 {
                        self.us[idx] = (old_s * self.us[idx] * coarse_area + acc.us[idx])
                            / (new_s * coarse_area);
                        self.vs[idx] = (old_s * self.vs[idx] * coarse_area + acc.vs[idx])
                            / (new_s * coarse_area);
                    }
                    self.water[idx] = new_w;
                    self.slurry[idx] = new_s;
                    continue;
                }
                self.water[idx] = w;
                self.slurry[idx] = s;
                if w > 1.0e-6 {
                    self.uw[idx] = acc.uw[idx] / (w * coarse_area);
                    self.vw[idx] = acc.vw[idx] / (w * coarse_area);
                }
                if s > 1.0e-6 {
                    self.us[idx] = acc.us[idx] / (s * coarse_area);
                    self.vs[idx] = acc.vs[idx] / (s * coarse_area);
                }
            }
            acc.end_clear();
        });
        self.fine_water.fill(0.0);
        self.fine_slurry.fill(0.0);
        self.fine_on = false;
        self.sheet_times.restrict_ms += tick.ms();
    }

    fn covers_fine(&self, x: f32, z: f32) -> bool {
        if !self.fine_on {
            return false;
        }
        let extent = FINE_RES as f32 * FINE_CELL_M;
        x >= self.fine_origin[0]
            && x < self.fine_origin[0] + extent
            && z >= self.fine_origin[1]
            && z < self.fine_origin[1] + extent
    }

    fn step_dust(&mut self, dt: f32, _slow: bool) {
        let tick = ProfTick::start();
        let n = AIR_RES * AIR_RES;
        let mut live = false;
        for chunk in self.dust.chunks_exact(n) {
            if chunk.iter().any(|&d| d > 1.0e-6) {
                live = true;
                break;
            }
        }
        if !live {
            self.sheet_times.dust_ms += tick.ms();
            return;
        }
        let cell = self.air_cell();
        let wind = self.wind;
        DUST_PLANE.with(|plane| {
            let mut plane = plane.borrow_mut();
            plane.fit(n);
            for layer in 0..AIR_LAYERS {
                let start = layer * n;
                let h = &mut self.dust[start..start + n];
                if !h.iter().any(|&d| d > 1.0e-6) {
                    continue;
                }
                let u = &mut self.dust_u[start..start + n];
                let v = &mut self.dust_v[start..start + n];
                soften_border(u, v, AIR_RES, None);
                step_down2(
                    h,
                    u,
                    v,
                    &plane.bed,
                    &plane.nu,
                    &plane.gate,
                    AIR_RES,
                    cell,
                    dt,
                    SheetDrive::Wind {
                        x: wind[0],
                        z: wind[1],
                        vorticity: 0.35,
                    },
                    1.0e-6,
                    false,
                );
            }
        });
        self.sheet_times.dust_ms += tick.ms();
    }

    /// Passive fog on the dust wind step. Empty slabs are not scanned.
    /// The velocity scratch is not `air_velocity`: that channel stays the dust sheet.
    fn step_fog(&mut self, dt: f32) {
        if !self.fog_live || dt <= 0.0 {
            return;
        }
        let tick = ProfTick::start();
        let cell = self.air_cell();
        let wind = self.wind;
        let n = AIR_RES * AIR_RES;
        DUST_PLANE.with(|plane| {
            let mut plane = plane.borrow_mut();
            plane.fit(n);
            for layer in 0..AIR_LAYERS {
                let start = layer * n;
                let h = &mut self.fog[start..start + n];
                if h.iter().all(|v| *v <= 1.0e-8) {
                    continue;
                }
                let u = &mut self.fog_u[start..start + n];
                let v = &mut self.fog_v[start..start + n];
                soften_border(u, v, AIR_RES, None);
                step_down2(
                    h,
                    u,
                    v,
                    &plane.bed,
                    &plane.nu,
                    &plane.gate,
                    AIR_RES,
                    cell,
                    dt,
                    SheetDrive::Wind {
                        x: wind[0],
                        z: wind[1],
                        vorticity: 0.35,
                    },
                    1.0e-8,
                    false,
                );
            }
        });
        self.sheet_times.dust_ms += tick.ms();
    }

    /// Bottom fog returns to the water column. Upper slabs fall one step.
    fn settle_fog(&mut self) {
        if !self.fog_live {
            return;
        }
        let air_cell = self.air_cell();
        let air_area = air_cell * air_cell;
        let coarse_area = COARSE_CELL_M * COARSE_CELL_M;
        let fall = 0.45f32;
        let mut left = 0.0f32;
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * air_cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * air_cell;
                for layer in (0..AIR_LAYERS).rev() {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    let drop = self.fog[idx] * fall;
                    if drop <= 0.0 {
                        continue;
                    }
                    self.fog[idx] -= drop;
                    if layer == 0 {
                        if let Some((ci, cj)) = self.coarse_ij(x, z) {
                            let cidx = cj * WINDOW_RES + ci;
                            self.water[cidx] += drop * air_area / coarse_area;
                        } else {
                            let key = world_key(x, z, COARSE_CELL_M);
                            let slot = self.parked.entry(key).or_default();
                            slot.water += drop * air_area / coarse_area;
                        }
                    } else {
                        let below = ((layer - 1) * AIR_RES + j) * AIR_RES + i;
                        self.fog[below] += drop;
                    }
                }
                for layer in 0..AIR_LAYERS {
                    left += self.fog[(layer * AIR_RES + j) * AIR_RES + i];
                }
            }
        }
        self.fog_live = left > 1.0e-8 || !self.parked_fog.is_empty();
    }

    fn settle_dust(&mut self) {
        let air_cell = self.air_cell();
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * air_cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * air_cell;
                let ground = self.sample_coarse(x, z);
                let wet = !dust_may_lift(ground.moisture, ground.water_m);
                let fall = if wet { 0.62 } else { 0.05 };
                for layer in 0..AIR_LAYERS {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    let drop = self.dust[idx] * fall;
                    if drop <= 0.0 {
                        continue;
                    }
                    self.dust[idx] -= drop;
                    if layer == 0 {
                        if let Some((ci, cj)) = self.coarse_ij(x, z) {
                            let cidx = cj * WINDOW_RES + ci;
                            // Air cell and coarse cell differ. Convert thickness
                            // so the cubic metres match.
                            let air_area = air_cell * air_cell;
                            let coarse_area = COARSE_CELL_M * COARSE_CELL_M;
                            self.settled[cidx] += drop * air_area / coarse_area;
                        }
                    } else {
                        let below = ((layer - 1) * AIR_RES + j) * AIR_RES + i;
                        self.dust[below] += drop;
                    }
                }
                if wet {
                    // Do not loft what just landed.
                    let _ = ground;
                }
            }
        }
    }

    /// The lift already ran on the column the tire rolled into. The solids stay
    /// in that rut. Clear water the open sheet shoves sideways does not take
    /// them, so the water that remains is muddier. Where that film no longer
    /// covers the load, the grains settle. Booking the load down to the water
    /// line would publish turbidity 1. What still fits rides the along-track film.
    fn step_suspended(&mut self, dt: f32) {
        self.last_susp_capped = 0;
        if dt <= 0.0 || !dt.is_finite() {
            return;
        }
        self.advect_suspended(dt);
    }

    fn entrain_wake(&mut self, pushes: &[TirePush], dt: f32) {
        let res = WINDOW_RES as i32;
        let cell = COARSE_CELL_M;
        let area = cell * cell;
        let origin = self.origin;
        for p in pushes {
            if p.speed < 0.3 {
                continue;
            }
            let fl = (p.forward_x * p.forward_x + p.forward_z * p.forward_z)
                .sqrt()
                .max(1.0e-4);
            let fx = p.forward_x / fl;
            let fz = p.forward_z / fl;
            let rx = fz;
            let rz = -fx;
            let reach = p.radius * 3.4 + cell;
            let i0 = (((p.x - reach) - origin[0]) / cell).floor() as i32;
            let i1 = (((p.x + reach) - origin[0]) / cell).floor() as i32;
            let j0 = (((p.z - reach) - origin[1]) / cell).floor() as i32;
            let j1 = (((p.z + reach) - origin[1]) / cell).floor() as i32;
            let aft = p.radius * 3.2;
            let half = p.width * 0.55 + 0.08;
            for j in j0.max(0)..=j1.min(res - 1) {
                for i in i0.max(0)..=i1.min(res - 1) {
                    let x = origin[0] + (i as f32 + 0.5) * cell;
                    let z = origin[1] + (j as f32 + 0.5) * cell;
                    let dx = x - p.x;
                    let dz = z - p.z;
                    let along = dx * fx + dz * fz;
                    // Positive along is ahead of the tire. The plume is the aft sector.
                    if along > -0.05 || along < -aft {
                        continue;
                    }
                    let lat = (dx * rx + dz * rz).abs();
                    if lat > half {
                        continue;
                    }
                    let idx = j as usize * WINDOW_RES + i as usize;
                    let water = self.water[idx];
                    if !water.is_finite() || water < 0.004 {
                        continue;
                    }
                    let along_t = ((-along) / aft).clamp(0.0, 1.0);
                    let lat_t = (lat / half).clamp(0.0, 1.0);
                    let core = (1.0 - along_t) * (1.0 - lat_t);
                    let room = (water - self.susp[idx]).max(0.0);
                    // Per second, so a 30 Hz step and a 60 Hz step lift the same volume.
                    let add = (4.0 * dt * core * water).min(room);
                    if add <= 1.0e-8 {
                        continue;
                    }
                    self.susp[idx] += add;
                    self.entrained_m3 += add * area;
                }
            }
        }
    }

    fn advect_suspended(&mut self, dt: f32) {
        let res = WINDOW_RES;
        let cell = COARSE_CELL_M;
        let area = cell * cell;
        let mut i0 = res;
        let mut i1 = 0usize;
        let mut j0 = res;
        let mut j1 = 0usize;
        // One f64 sum per step: thousands of tiny f32 adds drift the ledger.
        let mut landed_m3 = 0.0f64;
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                let mut s = self.susp[idx];
                if !s.is_finite() || s <= 1.0e-12 {
                    self.susp[idx] = 0.0;
                    continue;
                }
                let w = self.water[idx];
                if !w.is_finite() || w < 1.0e-5 {
                    self.settled_suspended_m3 += s * area;
                    self.susp[idx] = 0.0;
                    continue;
                }
                if s > w {
                    // The film no longer covers the load. Booking down to the
                    // water line would publish turbidity 1 from a deeper load.
                    self.settled_suspended_m3 += s * area;
                    self.susp[idx] = 0.0;
                    continue;
                }
                // Grains fall out of the column at their fall speed: the flux
                // to the bed is w_s * C, C = s / w, so a shallow film clears
                // first and a deep pool stays cloudy longer.
                let keep = (-dt * SUSP_FALL_M_S / w).exp();
                let landed = s * (1.0 - keep);
                if landed > 0.0 {
                    landed_m3 += f64::from(landed) * f64::from(area);
                    s -= landed;
                    self.susp[idx] = s;
                }
                if s > 1.0e-8 {
                    i0 = i0.min(i);
                    i1 = i1.max(i);
                    j0 = j0.min(j);
                    j1 = j1.max(j);
                }
            }
        }
        self.settled_suspended_m3 += landed_m3 as f32;
        if i0 > i1 {
            return;
        }
        i0 = i0.saturating_sub(1);
        j0 = j0.saturating_sub(1);
        i1 = (i1 + 1).min(res - 1);
        j1 = (j1 + 1).min(res - 1);

        // Copy first so a cell outside the box keeps the solids it already holds.
        self.susp_next.copy_from_slice(&self.susp);
        self.susp_taken.fill(0.0);
        // The open sheet shoves water onto the shoulders. Solids that followed
        // that shove left the rut near 0.20. On a puddle they stay in the track
        // column and ride the along-track film. A river or a swell is a real
        // current in both axes, so there the plume rides u as well as v and
        // spreads by eddy mixing. A neighbour stops at the donor's fraction,
        // so a thin film cannot publish 1 from a thicker cell.
        let open = self.surface_driven();
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                let depth = self.susp[idx];
                let water = self.water[idx];
                if depth <= 1.0e-12
                    || !depth.is_finite()
                    || !water.is_finite()
                    || water < 1.0e-5
                {
                    continue;
                }
                // Clay in the same cell is a second sheet. Blending its
                // velocity would drag the plume off the water that holds it.
                let (u, v) = film_uv(water, 0.0, self.uw[idx], self.vw[idx], 0.0, 0.0);
                let u = if open && u.is_finite() { u } else { 0.0 };
                let v = if v.is_finite() { v } else { 0.0 };
                let c = depth / water;
                let mut fu = u.abs() * dt / cell;
                let mut fv = v.abs() * dt / cell;
                let sum = fu + fv;
                if sum > 0.45 {
                    fu *= 0.45 / sum;
                    fv *= 0.45 / sum;
                }
                let up_i = if u > 0.0 { (i + 1 < res).then(|| idx + 1) } else { (i > 0).then(|| idx - 1) };
                let up_j = if v > 0.0 { (j + 1 < res).then(|| idx + res) } else { (j > 0).then(|| idx - res) };
                for (frac, recv) in [(fu, up_i), (fv, up_j)] {
                    let give = depth * frac;
                    if give <= 1.0e-12 {
                        continue;
                    }
                    let Some(n) = recv else {
                        self.settled_suspended_m3 += give * area;
                        self.susp_next[idx] -= give;
                        continue;
                    };
                    let wr = self.water[n];
                    if !wr.is_finite() || wr < 1.0e-5 {
                        continue;
                    }
                    let have = self.susp[n] + self.susp_taken[n];
                    let room = (c * wr - have).max(0.0);
                    let accept = give.min(room).min((wr - have).max(0.0));
                    if accept <= 1.0e-12 {
                        continue;
                    }
                    self.susp_next[idx] -= accept;
                    self.susp_taken[n] += accept;
                }
                if !open {
                    continue;
                }
                // Eddy mixing, K = 0.01 + 0.2 |U| h m²/s (shear-flow scale).
                // Down the concentration gradient only, at most 0.1 per side.
                let speed = (u * u + v * v).sqrt();
                let k = 0.01 + 0.2 * speed * water;
                let fd = (k * dt / (cell * cell)).min(0.1);
                let nbrs = [
                    (i + 1 < res).then(|| idx + 1),
                    (i > 0).then(|| idx - 1),
                    (j + 1 < res).then(|| idx + res),
                    (j > 0).then(|| idx - res),
                ];
                for n in nbrs.into_iter().flatten() {
                    let wr = self.water[n];
                    if !wr.is_finite() || wr < 1.0e-5 {
                        continue;
                    }
                    let cn = self.susp[n] / wr;
                    if !(c > cn) {
                        continue;
                    }
                    let give = fd * (c - cn) * water.min(wr);
                    let have = self.susp[n] + self.susp_taken[n];
                    let accept = give.min((wr - have).max(0.0));
                    if accept <= 1.0e-12 {
                        continue;
                    }
                    self.susp_next[idx] -= accept;
                    self.susp_taken[n] += accept;
                }
            }
        }
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                self.susp_next[idx] += self.susp_taken[idx];
            }
        }
        let mut capped = 0u32;
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                let mut s = self.susp_next[idx];
                if !s.is_finite() || s <= 0.0 {
                    self.susp_next[idx] = 0.0;
                    continue;
                }
                let w = self.water[idx];
                if !w.is_finite() || w < 1.0e-5 {
                    self.settled_suspended_m3 += s * area;
                    s = 0.0;
                } else if s > w {
                    let excess = s - w;
                    self.settled_suspended_m3 += excess * area;
                    if excess > 1.0e-5 {
                        capped += 1;
                    }
                    s = w;
                }
                self.susp_next[idx] = s;
            }
        }
        self.last_susp_capped += capped;
        // Cells outside the box were copied and not rewritten. Swap the whole grid.
        std::mem::swap(&mut self.susp, &mut self.susp_next);
    }

    fn shift_coarse(&mut self, new_origin: [f32; 2]) {
        if (new_origin[0] - self.origin[0]).abs() < 1.0e-4
            && (new_origin[1] - self.origin[1]).abs() < 1.0e-4
        {
            return;
        }
        let tick = ProfTick::start();
        if let Some((di, dj)) =
            uniform_cell_shift(self.origin, new_origin, COARSE_CELL_M, WINDOW_RES)
        {
            self.shift_coarse_scroll(new_origin, di, dj);
        } else {
            self.sheet_times.window_hash += 1;
            self.shift_coarse_hash(new_origin);
        }
        self.sheet_times.window_ms += tick.ms();
    }

    /// Park only the strip that leaves the 40 m window and copy the overlap by cell index.
    /// Bed and grain stay on their array index, matching the hash shift.
    fn shift_coarse_scroll(&mut self, new_origin: [f32; 2], di: i32, dj: i32) {
        let res = WINDOW_RES;
        let n = res * res;
        let cell = COARSE_CELL_M;
        let origin = self.origin;
        each_leaving(res, di, dj, |i, j| {
            let idx = j * res + i;
            let x = origin[0] + (i as f32 + 0.5) * cell;
            let z = origin[1] + (j as f32 + 0.5) * cell;
            let key = world_key(x, z, cell);
            let slot = Parked {
                water: self.water[idx],
                slurry: self.slurry[idx],
                uw: self.uw[idx],
                vw: self.vw[idx],
                us: self.us[idx],
                vs: self.vs[idx],
                dust: self.settled[idx],
                moisture: self.moisture[idx],
                foam: self.foam[idx],
                turbid: self.turbid[idx],
                susp: self.susp[idx],
            };
            if park_slot(&slot, self.moisture_ambient) {
                self.parked.insert(key, slot);
            } else {
                self.parked.remove(&key);
            }
        });
        let track_susp = self.turbidity_enabled;
        let planes = if track_susp { 11 } else { 10 };
        let need = n * planes;
        if self.shift_scratch.len() < need {
            self.shift_scratch.resize(need, 0.0);
        }
        let mut scratch = std::mem::take(&mut self.shift_scratch);
        scratch[..n].copy_from_slice(&self.water);
        scratch[n..n * 2].copy_from_slice(&self.slurry);
        scratch[n * 2..n * 3].copy_from_slice(&self.uw);
        scratch[n * 3..n * 4].copy_from_slice(&self.vw);
        scratch[n * 4..n * 5].copy_from_slice(&self.us);
        scratch[n * 5..n * 6].copy_from_slice(&self.vs);
        scratch[n * 6..n * 7].copy_from_slice(&self.settled);
        scratch[n * 7..n * 8].copy_from_slice(&self.moisture);
        scratch[n * 8..n * 9].copy_from_slice(&self.foam);
        scratch[n * 9..n * 10].copy_from_slice(&self.turbid);
        if track_susp {
            scratch[n * 10..n * 11].copy_from_slice(&self.susp);
        }
        let res_i = res as i32;
        for j in 0..res {
            for i in 0..res {
                let dst = j * res + i;
                let si = i as i32 + di;
                let sj = j as i32 + dj;
                let keep = if (0..res_i).contains(&si) && (0..res_i).contains(&sj) {
                    let src = sj as usize * res + si as usize;
                    scratch[src] > 1.0e-8
                        || scratch[n + src] > 1.0e-8
                        || scratch[n * 6 + src] > 1.0e-8
                        || scratch[n * 7 + src] > 1.0e-4
                        || scratch[n * 8 + src] > 1.0e-5
                        || scratch[n * 9 + src] > 1.0e-8
                        || (track_susp && scratch[n * 10 + src] > 1.0e-8)
                } else {
                    false
                };
                if keep {
                    let src = (j as i32 + dj) as usize * res + (i as i32 + di) as usize;
                    self.water[dst] = scratch[src];
                    self.slurry[dst] = scratch[n + src];
                    self.uw[dst] = scratch[n * 2 + src];
                    self.vw[dst] = scratch[n * 3 + src];
                    self.us[dst] = scratch[n * 4 + src];
                    self.vs[dst] = scratch[n * 5 + src];
                    self.settled[dst] = scratch[n * 6 + src];
                    self.moisture[dst] = scratch[n * 7 + src];
                    self.foam[dst] = scratch[n * 8 + src];
                    self.turbid[dst] = scratch[n * 9 + src];
                    self.susp[dst] = if track_susp {
                        scratch[n * 10 + src]
                    } else {
                        0.0
                    };
                } else {
                    self.water[dst] = 0.0;
                    self.slurry[dst] = 0.0;
                    self.uw[dst] = 0.0;
                    self.vw[dst] = 0.0;
                    self.us[dst] = 0.0;
                    self.vs[dst] = 0.0;
                    self.settled[dst] = 0.0;
                    self.moisture[dst] = self.moisture_ambient;
                    self.foam[dst] = 0.0;
                    self.turbid[dst] = 0.0;
                    self.susp[dst] = 0.0;
                }
            }
        }
        self.shift_scratch = scratch;
        self.origin = new_origin;
        self.reset_tracer_indices();
        each_leaving(res, -di, -dj, |i, j| {
            let x = self.origin[0] + (i as f32 + 0.5) * cell;
            let z = self.origin[1] + (j as f32 + 0.5) * cell;
            let key = world_key(x, z, cell);
            if let Some(slot) = self.parked.remove(&key) {
                let idx = j * res + i;
                self.water[idx] = slot.water;
                self.slurry[idx] = slot.slurry;
                self.uw[idx] = slot.uw;
                self.vw[idx] = slot.vw;
                self.us[idx] = slot.us;
                self.vs[idx] = slot.vs;
                self.settled[idx] = slot.dust;
                self.moisture[idx] = slot.moisture;
                self.foam[idx] = slot.foam;
                self.turbid[idx] = slot.turbid;
                self.susp[idx] = slot.susp;
            }
        });
    }

    fn shift_coarse_hash(&mut self, new_origin: [f32; 2]) {
        let n = WINDOW_RES * WINDOW_RES;
        let mut water = vec![0.0f32; n];
        let mut slurry = vec![0.0f32; n];
        let mut uw = vec![0.0f32; n];
        let mut vw = vec![0.0f32; n];
        let mut us = vec![0.0f32; n];
        let mut vs = vec![0.0f32; n];
        let mut bed = vec![0.0f32; n];
        let mut moisture = vec![self.moisture_ambient; n];
        let mut grain = vec![Grain::Other as u8; n];
        let mut settled = vec![0.0f32; n];
        let mut foam = vec![0.0f32; n];
        let mut turbid = vec![0.0f32; n];
        let mut susp = vec![0.0f32; n];
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let x = self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M;
                let z = self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M;
                let idx = j * WINDOW_RES + i;
                let key = world_key(x, z, COARSE_CELL_M);
                let slot = Parked {
                    water: self.water[idx],
                    slurry: self.slurry[idx],
                    uw: self.uw[idx],
                    vw: self.vw[idx],
                    us: self.us[idx],
                    vs: self.vs[idx],
                    dust: self.settled[idx],
                    moisture: self.moisture[idx],
                    foam: self.foam[idx],
                    turbid: self.turbid[idx],
                    susp: self.susp[idx],
                };
                if park_slot(&slot, self.moisture_ambient) {
                    self.parked.insert(key, slot);
                } else {
                    self.parked.remove(&key);
                }
            }
        }
        self.origin = new_origin;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let x = self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M;
                let z = self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M;
                let key = world_key(x, z, COARSE_CELL_M);
                let idx = j * WINDOW_RES + i;
                if let Some(slot) = self.parked.remove(&key) {
                    water[idx] = slot.water;
                    slurry[idx] = slot.slurry;
                    uw[idx] = slot.uw;
                    vw[idx] = slot.vw;
                    us[idx] = slot.us;
                    vs[idx] = slot.vs;
                    settled[idx] = slot.dust;
                    moisture[idx] = slot.moisture;
                    foam[idx] = slot.foam;
                    turbid[idx] = slot.turbid;
                    susp[idx] = slot.susp;
                }
                bed[idx] = self.bed[idx];
                grain[idx] = self.grain[idx];
            }
        }
        self.water = water;
        self.slurry = slurry;
        self.uw = uw;
        self.vw = vw;
        self.us = us;
        self.vs = vs;
        self.settled = settled;
        self.susp = susp;
        self.bed = bed;
        self.moisture = moisture;
        self.grain = grain;
        self.foam = foam;
        self.turbid = turbid;
        self.reset_tracer_indices();
    }

    fn reset_tracer_indices(&mut self) {
        self.wet_idx.clear();
        self.shore_idx.clear();
        self.foam_live.clear();
        self.foam_live_mark.fill(0);
    }

    fn shift_air(&mut self, new_origin: [f32; 2], cell: f32) {
        if (new_origin[0] - self.air_origin[0]).abs() < 1.0e-4
            && (new_origin[1] - self.air_origin[1]).abs() < 1.0e-4
        {
            return;
        }
        let tick = ProfTick::start();
        if let Some((di, dj)) = uniform_cell_shift(self.air_origin, new_origin, cell, AIR_RES) {
            self.shift_air_scroll(new_origin, cell, di, dj);
        } else {
            self.sheet_times.window_hash += 1;
            self.shift_air_hash(new_origin, cell);
        }
        self.sheet_times.window_ms += tick.ms();
    }

    /// A shift collapses every column onto layer 0. The overlap does that without hashing.
    fn shift_air_scroll(&mut self, new_origin: [f32; 2], cell: f32, di: i32, dj: i32) {
        if self.fog_live {
            self.shift_fog_scroll(new_origin, cell, di, dj);
        }
        let res = AIR_RES;
        let n = res * res;
        let origin = self.air_origin;
        let area = cell * cell;
        if self.shift_scratch.len() < n {
            self.shift_scratch.resize(n, 0.0);
        }
        let mut scratch = std::mem::take(&mut self.shift_scratch);
        for j in 0..res {
            for i in 0..res {
                let mut vol = 0.0f32;
                for layer in 0..AIR_LAYERS {
                    let idx = (layer * res + j) * res + i;
                    vol += self.dust[idx];
                }
                scratch[j * res + i] = vol;
            }
        }
        each_leaving(res, di, dj, |i, j| {
            let vol = scratch[j * res + i];
            if vol > 1.0e-8 {
                let x = origin[0] + (i as f32 + 0.5) * cell;
                let z = origin[1] + (j as f32 + 0.5) * cell;
                let key = world_key(x, z, cell);
                *self.parked_dust.entry(key).or_insert(0.0) += vol * area;
            }
        });
        self.dust.fill(0.0);
        self.dust_u.fill(0.0);
        self.dust_v.fill(0.0);
        let res_i = res as i32;
        for j in 0..res {
            for i in 0..res {
                let si = i as i32 + di;
                let sj = j as i32 + dj;
                if !(0..res_i).contains(&si) || !(0..res_i).contains(&sj) {
                    continue;
                }
                let vol = scratch[sj as usize * res + si as usize];
                if vol > 1.0e-8 {
                    self.dust[j * res + i] = (vol * area) / area;
                }
            }
        }
        self.shift_scratch = scratch;
        self.air_origin = new_origin;
        each_leaving(res, -di, -dj, |i, j| {
            let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
            let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
            let key = world_key(x, z, cell);
            if let Some(vol) = self.parked_dust.remove(&key) {
                self.dust[j * res + i] = vol / area;
            }
        });
    }

    fn shift_fog_scroll(&mut self, new_origin: [f32; 2], cell: f32, di: i32, dj: i32) {
        let res = AIR_RES;
        let n = res * res;
        let origin = self.air_origin;
        let area = cell * cell;
        let mut column = vec![0.0f32; n];
        for j in 0..res {
            for i in 0..res {
                let mut vol = 0.0f32;
                for layer in 0..AIR_LAYERS {
                    vol += self.fog[(layer * res + j) * res + i];
                }
                column[j * res + i] = vol;
            }
        }
        each_leaving(res, di, dj, |i, j| {
            let vol = column[j * res + i];
            if vol > 1.0e-8 {
                let x = origin[0] + (i as f32 + 0.5) * cell;
                let z = origin[1] + (j as f32 + 0.5) * cell;
                *self.parked_fog.entry(world_key(x, z, cell)).or_insert(0.0) += vol * area;
            }
        });
        self.fog.fill(0.0);
        self.fog_u.fill(0.0);
        self.fog_v.fill(0.0);
        let res_i = res as i32;
        for j in 0..res {
            for i in 0..res {
                let si = i as i32 + di;
                let sj = j as i32 + dj;
                if !(0..res_i).contains(&si) || !(0..res_i).contains(&sj) {
                    continue;
                }
                let vol = column[sj as usize * res + si as usize];
                if vol > 1.0e-8 {
                    self.fog[j * res + i] = vol;
                }
            }
        }
        each_leaving(res, -di, -dj, |i, j| {
            let x = new_origin[0] + (i as f32 + 0.5) * cell;
            let z = new_origin[1] + (j as f32 + 0.5) * cell;
            if let Some(vol) = self.parked_fog.remove(&world_key(x, z, cell)) {
                self.fog[j * res + i] += vol / area;
            }
        });
        self.fog_live = self.fog.iter().any(|v| *v > 1.0e-8) || !self.parked_fog.is_empty();
    }

    fn shift_fog_hash(&mut self, new_origin: [f32; 2], cell: f32) {
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
                let mut vol = 0.0f32;
                for layer in 0..AIR_LAYERS {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    vol += self.fog[idx];
                    self.fog[idx] = 0.0;
                    self.fog_u[idx] = 0.0;
                    self.fog_v[idx] = 0.0;
                }
                if vol > 1.0e-8 {
                    *self.parked_fog.entry(world_key(x, z, cell)).or_insert(0.0) += vol * cell * cell;
                }
            }
        }
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = new_origin[0] + (i as f32 + 0.5) * cell;
                let z = new_origin[1] + (j as f32 + 0.5) * cell;
                if let Some(vol) = self.parked_fog.remove(&world_key(x, z, cell)) {
                    self.fog[j * AIR_RES + i] = vol / (cell * cell);
                }
            }
        }
        self.fog_live = self.fog.iter().any(|v| *v > 1.0e-8) || !self.parked_fog.is_empty();
    }

    fn shift_air_hash(&mut self, new_origin: [f32; 2], cell: f32) {
        if self.fog_live {
            self.shift_fog_hash(new_origin, cell);
        }
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
                let mut vol = 0.0f32;
                for layer in 0..AIR_LAYERS {
                    let idx = (layer * AIR_RES + j) * AIR_RES + i;
                    vol += self.dust[idx];
                    self.dust[idx] = 0.0;
                    self.dust_u[idx] = 0.0;
                    self.dust_v[idx] = 0.0;
                }
                let key = world_key(x, z, cell);
                if vol > 1.0e-8 {
                    *self.parked_dust.entry(key).or_insert(0.0) += vol * cell * cell;
                }
            }
        }
        self.air_origin = new_origin;
        for j in 0..AIR_RES {
            for i in 0..AIR_RES {
                let x = self.air_origin[0] + (i as f32 + 0.5) * cell;
                let z = self.air_origin[1] + (j as f32 + 0.5) * cell;
                let key = world_key(x, z, cell);
                if let Some(vol) = self.parked_dust.remove(&key) {
                    let idx = j * AIR_RES + i;
                    self.dust[idx] = vol / (cell * cell);
                }
            }
        }
    }

    fn sample_coarse(&self, x: f32, z: f32) -> GroundSample {
        let Some((i, j)) = self.coarse_ij(x, z) else {
            let key = world_key(x, z, COARSE_CELL_M);
            if let Some(p) = self.parked.get(&key) {
                return GroundSample {
                    water_m: p.water,
                    slurry_m: p.slurry,
                    u: p.us,
                    v: p.vs,
                    settled_dust_m: p.dust,
                    moisture: p.moisture,
                    bed_m: 0.0,
                };
            }
            return GroundSample::default();
        };
        let idx = j * WINDOW_RES + i;
        GroundSample {
            water_m: self.water[idx],
            slurry_m: self.slurry[idx],
            u: self.us[idx],
            v: self.vs[idx],
            settled_dust_m: self.settled[idx],
            moisture: self.moisture[idx],
            bed_m: self.bed[idx],
        }
    }

    fn sample_fine(&self, x: f32, z: f32) -> Option<GroundSample> {
        if !self.covers_fine(x, z) {
            return None;
        }
        let i = ((x - self.fine_origin[0]) / FINE_CELL_M).floor() as usize;
        let j = ((z - self.fine_origin[1]) / FINE_CELL_M).floor() as usize;
        if i >= FINE_RES || j >= FINE_RES {
            return None;
        }
        let idx = j * FINE_RES + i;
        if let Some((ci, cj)) = self.coarse_ij(x, z) {
            let held = self.fine_hold.get(cj * WINDOW_RES + ci).copied() == Some(1);
            if !held && self.fine_water[idx] <= 1.0e-8 && self.fine_slurry[idx] <= 1.0e-8 {
                return None;
            }
        }
        Some(GroundSample {
            water_m: self.fine_water[idx],
            slurry_m: self.fine_slurry[idx],
            u: self.fine_us[idx],
            v: self.fine_vs[idx],
            settled_dust_m: 0.0,
            moisture: self.fine_moisture[idx],
            bed_m: self.fine_bed[idx],
        })
    }

    fn coarse_xz(&self, i: usize, j: usize) -> (f32, f32) {
        (
            self.origin[0] + (i as f32 + 0.5) * COARSE_CELL_M,
            self.origin[1] + (j as f32 + 0.5) * COARSE_CELL_M,
        )
    }

    fn coarse_ij(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        Some((
            coarse_axis(self.origin[0], x)?,
            coarse_axis(self.origin[1], z)?,
        ))
    }

    pub(crate) fn coarse_i_of(&self, x: f32) -> Option<usize> {
        coarse_axis(self.origin[0], x)
    }

    pub(crate) fn coarse_j_of(&self, z: f32) -> Option<usize> {
        coarse_axis(self.origin[1], z)
    }

    /// Coarse column at a cell the fine sheet already resolved. Valid when the
    /// fine patch is packed (`fine_on` is false), which is when the soil mirror runs.
    pub(crate) fn coarse_depth(&self, i: usize, j: usize) -> (f32, f32) {
        let idx = j * WINDOW_RES + i;
        (self.water[idx], self.slurry[idx])
    }

    pub(crate) fn fine_active(&self) -> bool {
        self.fine_on
    }

    /// Column and row of the coarse window for each fine cell. Same floor as
    /// [`Self::coarse_ij`], computed once per axis instead of once per cell.
    fn fine_coarse_axes(&self) -> ([u16; FINE_RES], [u16; FINE_RES]) {
        let mut col = [u16::MAX; FINE_RES];
        let mut row = [u16::MAX; FINE_RES];
        for i in 0..FINE_RES {
            let x = self.fine_origin[0] + (i as f32 + 0.5) * FINE_CELL_M;
            if let Some(ci) = coarse_axis(self.origin[0], x) {
                col[i] = ci as u16;
            }
        }
        for j in 0..FINE_RES {
            let z = self.fine_origin[1] + (j as f32 + 0.5) * FINE_CELL_M;
            if let Some(cj) = coarse_axis(self.origin[1], z) {
                row[j] = cj as u16;
            }
        }
        (col, row)
    }

    fn air_cell(&self) -> f32 {
        (WINDOW_RES as f32 * COARSE_CELL_M) / AIR_RES as f32
    }

    fn air_ij(&self, x: f32, z: f32) -> Option<(usize, usize)> {
        let cell = self.air_cell();
        let i = ((x - self.air_origin[0]) / cell).floor();
        let j = ((z - self.air_origin[1]) / cell).floor();
        if i < 0.0 || j < 0.0 {
            return None;
        }
        let i = i as usize;
        let j = j as usize;
        if i >= AIR_RES || j >= AIR_RES {
            None
        } else {
            Some((i, j))
        }
    }
}

fn coarse_axis(origin: f32, p: f32) -> Option<usize> {
    let i = ((p - origin) / COARSE_CELL_M).floor();
    if i < 0.0 {
        return None;
    }
    let i = i as usize;
    if i >= WINDOW_RES {
        None
    } else {
        Some(i)
    }
}

fn rim_scale(h: &[f32], i: usize, j: usize, rad: i32, feather: f32) -> f32 {
    let n = WINDOW_RES as i32;
    let mut nearest = feather;
    for dj in -rad..=rad {
        for di in -rad..=rad {
            let ii = i as i32 + di;
            let jj = j as i32 + dj;
            let dry = ii < 0
                || jj < 0
                || ii >= n
                || jj >= n
                || h[(jj as usize) * WINDOW_RES + ii as usize] <= 1.0e-6;
            if dry {
                let d = (di as f32).hypot(dj as f32) * COARSE_CELL_M;
                nearest = nearest.min(d);
            }
        }
    }
    let t = (nearest / feather).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `open` is a reach band `(along_z, cross0, cross1)` in cells. Inside that
/// band only the banks' window edges soften, not the inlet and the outlet.
fn soften_border(u: &mut [f32], v: &mut [f32], res: usize, open: Option<(bool, usize, usize)>) {
    let edge = (res / 12).max(2);
    // Only the band within `edge` of a window side can soften: `d` is never
    // below min(di, dj). Interior rows visit just their two end strips.
    let ends = edge.min(res);
    for j in 0..res {
        let row_band = j.min(res - 1 - j) < edge;
        let (left, right) = if row_band { (res, res) } else { (ends, res - ends) };
        for i in (0..left).chain(right.max(left)..res) {
            let di = i.min(res - 1 - i);
            let dj = j.min(res - 1 - j);
            let d = match open {
                Some((false, c0, c1)) if (c0..c1).contains(&j) => dj,
                Some((true, c0, c1)) if (c0..c1).contains(&i) => di,
                _ => di.min(dj),
            };
            if d >= edge {
                continue;
            }
            let t = d as f32 / edge as f32;
            let w = t * t * (3.0 - 2.0 * t);
            let idx = j * res + i;
            u[idx] *= w;
            v[idx] *= w;
        }
    }
}

/// Tire disks, plus the rut between the axles on each side. The axle line
/// between left and right is not a zone.
fn near_tire_zone(pushes: &[TirePush], x: f32, z: f32, r2: f32) -> bool {
    for p in pushes {
        let dx = x - p.x;
        let dz = z - p.z;
        if dx * dx + dz * dz <= r2 {
            return true;
        }
    }
    for i in 0..pushes.len() {
        for j in (i + 1)..pushes.len() {
            let a = &pushes[i];
            let b = &pushes[j];
            let dx = (a.x - b.x).abs();
            // Same flank only. An axle segment would bridge the two ruts and
            // pull the dry middle into the fine sheet.
            if dx >= 0.45 {
                continue;
            }
            if dist2_segment(x, z, a.x, a.z, b.x, b.z) <= r2 {
                return true;
            }
        }
    }
    false
}

fn dist2_segment(px: f32, pz: f32, ax: f32, az: f32, bx: f32, bz: f32) -> f32 {
    let abx = bx - ax;
    let abz = bz - az;
    let ab2 = abx * abx + abz * abz;
    let t = if ab2 < 1.0e-8 {
        0.0
    } else {
        (((px - ax) * abx + (pz - az) * abz) / ab2).clamp(0.0, 1.0)
    };
    let dx = px - (ax + abx * t);
    let dz = pz - (az + abz * t);
    dx * dx + dz * dz
}

/// 0..1 hash for spray jitter. Deterministic, so goldens stay put.
fn spray_hash(n: u32) -> f32 {
    let mut x = n.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    (x & 0xFFFF) as f32 / 65535.0
}

fn coarse_held(hold: &[u8], origin: [f32; 2], x: f32, z: f32) -> bool {
    let i = ((x - origin[0]) / COARSE_CELL_M).floor();
    let j = ((z - origin[1]) / COARSE_CELL_M).floor();
    if i < 0.0 || j < 0.0 {
        return false;
    }
    let (i, j) = (i as usize, j as usize);
    i < WINDOW_RES && j < WINDOW_RES && hold.get(j * WINDOW_RES + i).copied() == Some(1)
}

/// Foam fraction from the Froude number of the film. Zero below 4 mm, at
/// rest, or when the film is not breaking.
fn breaking_foam(speed: f32, thickness: f32) -> f32 {
    if !speed.is_finite()
        || !thickness.is_finite()
        || thickness < 0.004
        || speed <= 0.05
    {
        return 0.0;
    }
    let froude = speed / (GRAVITY * thickness).sqrt();
    smoothstep(0.85, 1.45, froude)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn sample_foam(src: &[f32], res: usize, x: f32, z: f32) -> f32 {
    if !x.is_finite() || !z.is_finite() || x < 0.0 || z < 0.0 {
        return 0.0;
    }
    let i0 = x.floor() as usize;
    let j0 = z.floor() as usize;
    if i0 >= res || j0 >= res {
        return 0.0;
    }
    let i1 = (i0 + 1).min(res - 1);
    let j1 = (j0 + 1).min(res - 1);
    let tx = (x - i0 as f32).clamp(0.0, 1.0);
    let tz = (z - j0 as f32).clamp(0.0, 1.0);
    let a = src[j0 * res + i0];
    let b = src[j0 * res + i1];
    let c = src[j1 * res + i0];
    let d = src[j1 * res + i1];
    let v = a + (b - a) * tx + (c - a) * tz + (a - b - c + d) * tx * tz;
    if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }
}

/// Donor-cell fluxes of a thickness. `scratch` is `removed` then `added`,
/// each `res * res` long. The column sum does not change.
fn flux_thickness(
    h: &[f32],
    u: &[f32],
    v: &[f32],
    res: usize,
    cell: f32,
    dt: f32,
    i0: usize,
    i1: usize,
    j0: usize,
    j1: usize,
    scratch: &mut [f32],
) {
    let n = res * res;
    for j in j0..=j1 {
        let row = j * res;
        for i in i0..i1.min(res - 1) {
            let a = row + i;
            let b = a + 1;
            let vel = 0.5 * (u[a] + u[b]);
            flux_edge(h, scratch, n, a, b, vel, cell, dt);
        }
    }
    for j in j0..j1.min(res - 1) {
        for i in i0..=i1 {
            let a = j * res + i;
            let b = a + res;
            let vel = 0.5 * (v[a] + v[b]);
            flux_edge(h, scratch, n, a, b, vel, cell, dt);
        }
    }
}

/// Inclusive index box of cells that hold silt, grown by one cell so a flux
/// can land on the neighbour. `None` when the column is empty.
/// True when a wet cell touches a dry neighbour on the water column.
fn shore_wet_cell(water: &[f32], res: usize, i: usize, j: usize) -> bool {
    if i == 0 || j == 0 || i + 1 >= res || j + 1 >= res {
        return false;
    }
    for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
        let nidx = (j as i32 + dj) as usize * res + (i as i32 + di) as usize;
        let nh = water[nidx];
        if nh.is_finite() && nh < 0.004 {
            return true;
        }
    }
    false
}

fn turbid_span(h: &[f32], res: usize) -> Option<(usize, usize, usize, usize)> {
    let mut i0 = res;
    let mut i1 = 0usize;
    let mut j0 = res;
    let mut j1 = 0usize;
    for j in 0..res {
        let row = &h[j * res..j * res + res];
        // One branch-free pass per row; only a live row pays for its ends.
        if row.iter().fold(0.0f32, |m, &x| m.max(x)) <= 1.0e-8 {
            continue;
        }
        let first = row.iter().position(|&x| x > 1.0e-8).unwrap_or(0);
        let last = row.iter().rposition(|&x| x > 1.0e-8).unwrap_or(0);
        i0 = i0.min(first);
        i1 = i1.max(last);
        j0 = j0.min(j);
        j1 = j1.max(j);
    }
    if i0 > i1 {
        return None;
    }
    Some((
        i0.saturating_sub(1),
        (i1 + 1).min(res - 1),
        j0.saturating_sub(1),
        (j1 + 1).min(res - 1),
    ))
}

fn flux_edge(
    h: &[f32],
    scratch: &mut [f32],
    n: usize,
    a: usize,
    b: usize,
    vel: f32,
    cell: f32,
    dt: f32,
) {
    if !vel.is_finite() || vel.abs() < 1.0e-5 {
        return;
    }
    let (donor, recv) = if vel > 0.0 { (a, b) } else { (b, a) };
    let column = if h[donor].is_finite() { h[donor] } else { 0.0 };
    let room = column - scratch[donor];
    if room <= 1.0e-8 {
        return;
    }
    let frac = (vel.abs() * dt / cell).min(0.45);
    let dh = (column * frac).min(room);
    if dh <= 1.0e-8 || !dh.is_finite() {
        return;
    }
    scratch[donor] += dh;
    scratch[n + recv] += dh;
}

/// Shore foam production (0..1) and silt lift rate (m/s) for one wet cell.
/// Quiet water on the bank is `(0, 0)`. The border of the window is skipped:
/// the strain stencil needs both neighbours.
fn shore_cell(water: &[f32], uw: &[f32], vw: &[f32], res: usize, idx: usize) -> (f32, f32) {
    let i = idx % res;
    let j = idx / res;
    if i == 0 || j == 0 || i + 1 >= res || j + 1 >= res {
        return (0.0, 0.0);
    }
    let h = water[idx];
    if !h.is_finite() || h < 0.004 {
        return (0.0, 0.0);
    }
    let mut nx = 0.0f32;
    let mut nz = 0.0f32;
    let mut edge = false;
    for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
        let nidx = (j as i32 + dj) as usize * res + (i as i32 + di) as usize;
        let nh = water[nidx];
        if nh.is_finite() && nh < 0.004 {
            edge = true;
            nx += di as f32;
            nz += dj as f32;
        }
    }
    if !edge {
        return (0.0, 0.0);
    }
    let u = uw[idx];
    let v = vw[idx];
    if !u.is_finite() || !v.is_finite() {
        return (0.0, 0.0);
    }
    let film = u.hypot(v);
    let nlen = nx.hypot(nz).max(1.0e-4);
    let into = ((u * nx + v * nz) / nlen).max(0.0);
    let strain = (uw[idx + 1] - uw[idx - 1]).abs() + (vw[idx + res] - vw[idx - res]).abs();
    let shear_v = film.max(strain);
    if shear_v < 0.2 && into < 0.2 {
        return (0.0, 0.0);
    }
    let prod = breaking_foam(shear_v.max(into), h).max(smoothstep(0.25, 1.1, into));
    let foam = if prod > 0.02 { prod } else { 0.0 };
    let mut rate = 0.0f32;
    if h > 0.012 && h < 0.055 && (shear_v > 0.45 || into > 0.4) {
        let tau = 1000.0 * 0.025 * shear_v * shear_v;
        if tau > 3.0 || into > 0.45 {
            rate = (0.006 * (into + shear_v * 0.35)).min(0.01);
        }
    }
    (foam, rate)
}

/// Half-open index range covering cell centres within `reach` of `center`, plus one cell.
fn index_span(origin: f32, cell: f32, res: usize, center: f32, reach: f32) -> (usize, usize) {
    if !(cell > 0.0) || res == 0 {
        return (0, 0);
    }
    let lo = ((center - reach - origin) / cell - 1.0).floor().max(0.0);
    let hi = ((center + reach - origin) / cell + 2.0).ceil().max(0.0);
    let lo = (lo as usize).min(res);
    let hi = (hi as usize).min(res).max(lo);
    (lo, hi)
}

fn snap_origin(center: [f32; 2], extent: f32, cell: f32) -> [f32; 2] {
    let x = center[0] - extent * 0.5;
    let z = center[1] - extent * 0.5;
    [(x / cell).floor() * cell, (z / cell).floor() * cell]
}

fn world_key(x: f32, z: f32, cell: f32) -> (i32, i32) {
    ((x / cell).floor() as i32, (z / cell).floor() as i32)
}

/// Index delta of a window move when every cell key shifts by the same whole cell.
fn uniform_cell_shift(old: [f32; 2], new: [f32; 2], cell: f32, res: usize) -> Option<(i32, i32)> {
    if res == 0 || !(cell.is_finite() && cell > 0.0) {
        return None;
    }
    let key = |origin: [f32; 2], i: usize, j: usize| {
        world_key(
            origin[0] + (i as f32 + 0.5) * cell,
            origin[1] + (j as f32 + 0.5) * cell,
            cell,
        )
    };
    let old00 = key(old, 0, 0);
    let new00 = key(new, 0, 0);
    let di = new00.0 - old00.0;
    let dj = new00.1 - old00.1;
    let last = res - 1;
    let old11 = key(old, last, last);
    let new11 = key(new, last, last);
    if old11.0 - old00.0 != last as i32 || old11.1 - old00.1 != last as i32 {
        return None;
    }
    if new11.0 - new00.0 != last as i32 || new11.1 - new00.1 != last as i32 {
        return None;
    }
    if new11.0 - old11.0 != di || new11.1 - old11.1 != dj {
        return None;
    }
    let mid = res / 2;
    let oldm = key(old, mid, mid);
    let newm = key(new, mid, mid);
    if newm.0 - oldm.0 != di || newm.1 - oldm.1 != dj {
        return None;
    }
    Some((di, dj))
}

/// Cells that leave the window when the origin moves by `(di, dj)` cells.
fn each_leaving(res: usize, di: i32, dj: i32, mut f: impl FnMut(usize, usize)) {
    if res == 0 {
        return;
    }
    let n = res as i32;
    if di > 0 {
        let w = (di.min(n) as usize).min(res);
        for j in 0..res {
            for i in 0..w {
                f(i, j);
            }
        }
    } else if di < 0 {
        let w = ((-di).min(n) as usize).min(res);
        for j in 0..res {
            for i in (res - w)..res {
                f(i, j);
            }
        }
    }
    if dj > 0 {
        let h = (dj.min(n) as usize).min(res);
        for j in 0..h {
            for i in 0..res {
                let covered = if di > 0 {
                    (i as i32) < di
                } else if di < 0 {
                    (i as i32) >= n + di
                } else {
                    false
                };
                if !covered {
                    f(i, j);
                }
            }
        }
    } else if dj < 0 {
        let h = ((-dj).min(n) as usize).min(res);
        for j in (res - h)..res {
            for i in 0..res {
                let covered = if di > 0 {
                    (i as i32) < di
                } else if di < 0 {
                    (i as i32) >= n + di
                } else {
                    false
                };
                if !covered {
                    f(i, j);
                }
            }
        }
    }
}

/// Vorticity confinement, metres per second. The kick is this times N × ω × dt.
/// High enough that the body shear rolls up behind the rear face instead of
/// diffusing below [`WAKE_VORTICITY_MIN`].
pub const AIR_CONFINE: f32 = 18.0;
/// |ω| the wake behind the body must clear, 1/s. Scored 0.5–5 m aft of the
/// rear face, inside the track, not on the free-stream shear at the lee edge.
pub const WAKE_VORTICITY_MIN: f32 = 6.0;
/// Milliseconds allowed for one air step in an unoptimized `cargo test`:
/// inject, advect, confine, and the density flux. The proof drive must stay
/// under 4 ms, which is the budget for the step that feeds the GPU upload.
pub const AIR_STEP_BUDGET_MS: f32 = 4.0;
/// Gravity-sheet speed cap, metres per second. Published `vx` and `vz` stay inside it.
pub const FILM_SPEED_CAP_MPS: f32 = 1.6;
/// Milliseconds for one [`FluidCore::step`] wall. The plume test times that
/// call with the clay pond inside it, and asserts the mean and the p95.
/// Do not raise this, and do not treat a miss as known.
pub const SOLVER_STEP_BUDGET_MS: f32 = 7.0;
/// Density flux runs on this many air steps, with `dt` scaled so the mean rate holds.
pub const AIR_FLUX_STRIDE: u32 = 4;
/// Velocity substeps. The solve runs on this many dust steps, with `dt` scaled.
pub const AIR_SOLVE_STRIDE: u32 = 2;
/// Fine dust cells per coarse air cell in X and Z. Layers stay at the dust spacing.
/// The solve runs on this coarser active window and is written back onto the dust grid.
pub const AIR_COARSE: usize = 2;
/// Fine dust layers per coarse air layer.
pub const AIR_COARSE_Y: usize = 2;
/// Share of the wake residual that moves dust. Free-stream wind is already
/// applied by [`advect_dust_density`].
pub const AIR_DENSITY_COUPLING: f32 = 0.04;

const AIR_FEATHER_M: f32 = 0.22;
const AIR_WIND_RATE: f32 = 1.35;
const AIR_WIN_LAT: f32 = 1.9;
const AIR_WIN_Z0: f32 = -6.5;
const AIR_WIN_Z1: f32 = 2.0;

/// Niva body half-extents in the chassis frame, metres. +X right, +Y up, +Z forward.
pub const NIVA_BODY_HALF: [f32; 3] = [0.86, 0.62, 1.78];
/// Body centre relative to the chassis origin. The origin sits at the height
/// used by [`crate::dust::CHASSIS_Y`].
pub const NIVA_BODY_OFFSET: [f32; 3] = [0.0, 0.38, 0.02];
/// Hub centres relative to that chassis origin. Same plan as `dust::HUBS`.
pub const NIVA_WHEEL_HUB: [[f32; 3]; 4] = [
    [-0.74, -0.20, 1.08],
    [0.74, -0.20, 1.08],
    [-0.74, -0.20, -1.18],
    [0.74, -0.20, -1.18],
];
pub const NIVA_WHEEL_RADIUS: f32 = 0.38;
pub const NIVA_WHEEL_HALF_WIDTH: f32 = 0.14;

/// Wind and chassis for one air step. Velocity is stored as world metres per second.
#[derive(Clone, Copy, Debug)]
pub struct AirDrive {
    pub wind_x: f32,
    pub wind_z: f32,
    pub speed: f32,
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub chassis: [f32; 3],
    /// World velocity of the grid itself. The dust window is fixed to the chassis.
    pub scroll: [f32; 3],
    /// Wake sample in the chassis frame: |lateral| ≤ lat, y in range, forward in z.
    pub wake_lat: f32,
    pub wake_y0: f32,
    pub wake_y1: f32,
    pub wake_z0: f32,
    pub wake_z1: f32,
}

/// A box or a wheel that forces the air. `spin` is radians per second about `right`.
#[derive(Clone, Copy, Debug)]
pub struct AirSolid {
    pub center: [f32; 3],
    pub half: [f32; 3],
    pub right: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],
    /// World velocity of the solid's origin, before spin, metres per second.
    pub velocity: [f32; 3],
    pub spin: f32,
    pub cylinder: bool,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct AirReport {
    /// Peak smoothed |ω| in the wake, 1/s.
    pub peak_vorticity: f32,
    pub wake_cells: u32,
    pub nan_count: u32,
}

/// Body and four wheels for the straight Niva. Hubs match `dust::HUBS`.
pub fn niva_air(
    chassis: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    speed: f32,
    slip: [f32; 4],
    wind_x: f32,
    wind_z: f32,
) -> (AirDrive, [AirSolid; 5]) {
    let forward = norm3(forward, [0.0, 0.0, 1.0]);
    let right = norm3(right, [1.0, 0.0, 0.0]);
    let up = norm3(cross3(forward, right), [0.0, 1.0, 0.0]);
    let speed = if speed.is_finite() {
        speed.max(0.0)
    } else {
        0.0
    };
    let body_vel = scale3(forward, speed);
    let center = add3(
        chassis,
        add3(
            scale3(right, NIVA_BODY_OFFSET[0]),
            add3(
                scale3(up, NIVA_BODY_OFFSET[1]),
                scale3(forward, NIVA_BODY_OFFSET[2]),
            ),
        ),
    );
    let body = AirSolid {
        center,
        half: NIVA_BODY_HALF,
        right,
        forward,
        up,
        velocity: body_vel,
        spin: 0.0,
        cylinder: false,
        radius: 0.0,
    };
    let mut solids = [body; 5];
    for i in 0..4 {
        let h = NIVA_WHEEL_HUB[i];
        let hub = add3(
            chassis,
            add3(
                scale3(right, h[0]),
                add3(scale3(up, h[1]), scale3(forward, h[2])),
            ),
        );
        let omega = (speed + slip[i].abs()) / NIVA_WHEEL_RADIUS;
        solids[i + 1] = AirSolid {
            center: hub,
            half: [NIVA_WHEEL_HALF_WIDTH, NIVA_WHEEL_RADIUS, NIVA_WHEEL_RADIUS],
            right,
            forward,
            up,
            velocity: body_vel,
            spin: omega,
            cylinder: true,
            radius: NIVA_WHEEL_RADIUS,
        };
    }
    let rear = NIVA_BODY_OFFSET[2] - NIVA_BODY_HALF[2];
    let drive = AirDrive {
        wind_x,
        wind_z,
        speed,
        forward,
        right,
        up,
        chassis,
        scroll: [0.0, 0.0, speed * forward[2]],
        wake_lat: 1.35,
        wake_y0: 0.28,
        wake_y1: 1.45,
        wake_z0: rear - 4.6,
        wake_z1: rear - 0.50,
    };
    (drive, solids)
}

thread_local! {
    static AIR_OMEGA: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    static COL_LIVE: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    static AIR_WAKE: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// World AABB of every solid, expanded by the occupancy feather.
fn solids_aabb(solids: &[AirSolid]) -> [f32; 6] {
    if solids.is_empty() {
        return [0.0, 0.0, 0.0, -1.0, -1.0, -1.0];
    }
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for solid in solids {
        let (hx, hy, hz) = if solid.cylinder {
            (
                solid.half[0] + AIR_FEATHER_M,
                solid.radius + AIR_FEATHER_M,
                solid.radius + AIR_FEATHER_M,
            )
        } else {
            (
                solid.half[0] + AIR_FEATHER_M,
                solid.half[1] + AIR_FEATHER_M,
                solid.half[2] + AIR_FEATHER_M,
            )
        };
        let ex = solid.right[0].abs() * hx + solid.up[0].abs() * hy + solid.forward[0].abs() * hz;
        let ey = solid.right[1].abs() * hx + solid.up[1].abs() * hy + solid.forward[1].abs() * hz;
        let ez = solid.right[2].abs() * hx + solid.up[2].abs() * hy + solid.forward[2].abs() * hz;
        min[0] = min[0].min(solid.center[0] - ex);
        min[1] = min[1].min(solid.center[1] - ey);
        min[2] = min[2].min(solid.center[2] - ez);
        max[0] = max[0].max(solid.center[0] + ex);
        max[1] = max[1].max(solid.center[1] + ey);
        max[2] = max[2].max(solid.center[2] + ez);
    }
    [min[0], min[1], min[2], max[0], max[1], max[2]]
}

#[inline(always)]
fn point_in_aabb(p: [f32; 3], aabb: [f32; 6]) -> bool {
    p[0] >= aabb[0]
        && p[0] <= aabb[3]
        && p[1] >= aabb[1]
        && p[1] <= aabb[4]
        && p[2] >= aabb[2]
        && p[2] <= aabb[5]
}

#[inline(always)]
fn row_stream(streams: &[[f32; 3]], ix: usize, x: f32, drive: &AirDrive) -> [f32; 3] {
    if ix < streams.len() {
        streams[ix]
    } else {
        stream_at(x, drive)
    }
}

/// Inject the solids, advect the velocity, confine vorticity.
///
/// `vel` and `scratch` are world metres per second, xyz, indexed like the dust
/// volume: `x + y * nx + z * nx * ny`. `warm` is false on the first step so the
/// grid starts on the free stream instead of a dead calm.
pub fn step_vehicle_air(
    vel: &mut [f32],
    scratch: &mut [f32],
    origin: [f32; 3],
    cell: [f32; 3],
    res: [usize; 3],
    drive: &AirDrive,
    solids: &[AirSolid],
    dt: f32,
    warm: bool,
    fill_outside: bool,
) -> AirReport {
    let n = res[0].saturating_mul(res[1]).saturating_mul(res[2]);
    if dt <= 0.0 || !dt.is_finite() || n == 0 || vel.len() < n * 3 || scratch.len() < n * 3 {
        return AirReport {
            peak_vorticity: 0.0,
            wake_cells: 0,
            nan_count: 0,
        };
    }
    let (ix0, ix1, iz0, iz1) = air_window(origin, cell, res, drive);
    let aabb = solids_aabb(solids);
    let mut stream_row = [[0.0f32; 3]; 160];
    let stream_n = if res[0] <= stream_row.len() {
        res[0]
    } else {
        0
    };
    for ix in 0..stream_n {
        let x = origin[0] + (ix as f32 + 0.5) * cell[0];
        stream_row[ix] = stream_at(x, drive);
    }
    let streams: &[[f32; 3]] = if stream_n == 0 {
        &[]
    } else {
        &stream_row[..stream_n]
    };
    if !warm {
        for iz in 0..res[2] {
            for iy in 0..res[1] {
                for ix in 0..res[0] {
                    let x = origin[0] + (ix as f32 + 0.5) * cell[0];
                    store3(vel, at3(res, ix, iy, iz), row_stream(streams, ix, x, drive));
                }
            }
        }
    }
    if fill_outside {
        for iz in 0..res[2] {
            for iy in 0..res[1] {
                for ix in 0..res[0] {
                    if ix >= ix0 && ix <= ix1 && iz >= iz0 && iz <= iz1 {
                        continue;
                    }
                    let x = origin[0] + (ix as f32 + 0.5) * cell[0];
                    let i = at3(res, ix, iy, iz);
                    let fs = row_stream(streams, ix, x, drive);
                    store3(vel, i, fs);
                    store3(scratch, i, fs);
                }
            }
        }
    }
    let wind_a = 1.0 - (-dt * AIR_WIND_RATE).exp();
    let keep_w = 1.0 - wind_a;
    for iz in iz0..=iz1 {
        for iy in 0..res[1] {
            for ix in ix0..=ix1 {
                let p = cell_xyz(origin, cell, ix, iy, iz);
                let fs = row_stream(streams, ix, p[0], drive);
                let (occ, target) = if point_in_aabb(p, aabb) {
                    solid_target(p, solids, fs)
                } else {
                    (0.0, fs)
                };
                let i = at3(res, ix, iy, iz);
                let cur = load3(vel, i);
                let keep = (1.0 - occ) * keep_w;
                let wind_w = (1.0 - occ) * wind_a;
                let mixed = [
                    target[0] * occ + fs[0] * wind_w + cur[0] * keep,
                    target[1] * occ + fs[1] * wind_w + cur[1] * keep,
                    target[2] * occ + fs[2] * wind_w + cur[2] * keep,
                ];
                store3(scratch, i, mixed);
            }
        }
    }
    // Semi-Lagrangian in the chassis window. Upstream of the window is the free stream.
    let max_rel = 3.0 * cell[0].max(cell[2]) / dt;
    for iz in iz0..=iz1 {
        for iy in 0..res[1] {
            for ix in ix0..=ix1 {
                let i = at3(res, ix, iy, iz);
                let cur = load3(scratch, i);
                let mut rx = cur[0] - drive.scroll[0];
                let mut rz = cur[2] - drive.scroll[2];
                let sp = rx.hypot(rz);
                if sp > max_rel && sp.is_finite() {
                    let s = max_rel / sp;
                    rx *= s;
                    rz *= s;
                }
                let p = cell_xyz(origin, cell, ix, iy, iz);
                let bx = p[0] - rx * dt;
                let bz = p[2] - rz * dt;
                let sampled = sample_layer(scratch, origin, cell, res, drive, iy, bx, bz);
                store3(vel, i, sampled);
            }
        }
    }
    // Vorticity of the advected field, then confinement. One ω sample per cell.
    let mut nan_count = 0u32;
    AIR_OMEGA.with(|slot| {
        let mut omega = slot.borrow_mut();
        if omega.len() < n {
            omega.resize(n, 0.0);
        }
        for iz in iz0..=iz1 {
            for iy in 0..res[1] {
                for ix in ix0..=ix1 {
                    omega[at3(res, ix, iy, iz)] = vort_xz(vel, res, ix, iy, iz, cell);
                }
            }
        }
        for iz in iz0..=iz1 {
            for iy in 0..res[1] {
                for ix in ix0..=ix1 {
                    let i = at3(res, ix, iy, iz);
                    let mut v = load3(vel, i);
                    let edge = ix == ix0 || ix == ix1 || iz == iz0 || iz == iz1;
                    if !edge {
                        let w = omega[i];
                        if w.is_finite() && w.abs() > 1.0e-4 {
                            let wxp = omega[at3(res, ix + 1, iy, iz)].abs();
                            let wxm = omega[at3(res, ix - 1, iy, iz)].abs();
                            let wzp = omega[at3(res, ix, iy, iz + 1)].abs();
                            let wzm = omega[at3(res, ix, iy, iz - 1)].abs();
                            let mut nx = (wxp - wxm) / (2.0 * cell[0]);
                            let mut nz = (wzp - wzm) / (2.0 * cell[2]);
                            let nlen = nx.hypot(nz);
                            if nlen > 1.0e-4 && nlen.is_finite() {
                                nx /= nlen;
                                nz /= nlen;
                                let mut du = AIR_CONFINE * dt * (-nz * w);
                                let mut dw = AIR_CONFINE * dt * (nx * w);
                                let kick = du.hypot(dw);
                                if kick > 2.5 {
                                    let s = 2.5 / kick;
                                    du *= s;
                                    dw *= s;
                                }
                                if du.is_finite() && dw.is_finite() {
                                    v[0] += du;
                                    v[2] += dw;
                                }
                            }
                        }
                    }
                    let x = origin[0] + (ix as f32 + 0.5) * cell[0];
                    let fs = row_stream(streams, ix, x, drive);
                    if !v[0].is_finite() || !v[1].is_finite() || !v[2].is_finite() {
                        nan_count += 1;
                        v = fs;
                    }
                    let sp = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if sp > 32.0 && sp.is_finite() {
                        let s = 32.0 / sp;
                        v[0] *= s;
                        v[1] *= s;
                        v[2] *= s;
                    }
                    store3(vel, i, v);
                }
            }
        }
    });
    let (peak, cells) = wake_vorticity(vel, origin, cell, res, drive, ix0, ix1, iz0, iz1);
    AirReport {
        peak_vorticity: peak,
        wake_cells: cells,
        nan_count,
    }
}

/// Solve the Niva wake on a coarse active window and copy it onto the fine dust grid.
///
/// `warm` keeps the coarse velocity from the previous step. `full_stamp` writes
/// every fine cell; a steady heading only rewrites the window.
pub fn step_dust_air(
    fine: &mut [f32],
    coarse: &mut [f32],
    coarse_scratch: &mut [f32],
    origin: [f32; 3],
    fine_cell: [f32; 3],
    fine_res: [usize; 3],
    drive: &AirDrive,
    solids: &[AirSolid],
    dt: f32,
    warm: bool,
    full_stamp: bool,
) -> AirReport {
    let factor = AIR_COARSE.max(1);
    let empty = AirReport {
        peak_vorticity: 0.0,
        wake_cells: 0,
        nan_count: 0,
    };
    let fy = AIR_COARSE_Y.max(1);
    if fine_res[0] < factor
        || fine_res[2] < factor
        || fine_res[1] < fy
        || fine_res[0] % factor != 0
        || fine_res[2] % factor != 0
        || fine_res[1] % fy != 0
    {
        return empty;
    }
    let cres = [fine_res[0] / factor, fine_res[1] / fy, fine_res[2] / factor];
    let n = cres[0].saturating_mul(cres[1]).saturating_mul(cres[2]);
    if n == 0
        || coarse.len() < n * 3
        || coarse_scratch.len() < n * 3
        || fine.len()
            < fine_res[0]
                .saturating_mul(fine_res[1])
                .saturating_mul(fine_res[2])
                .saturating_mul(3)
    {
        return empty;
    }
    let ccell = [
        fine_cell[0] * factor as f32,
        fine_cell[1] * fy as f32,
        fine_cell[2] * factor as f32,
    ];
    let report = step_vehicle_air(
        coarse,
        coarse_scratch,
        origin,
        ccell,
        cres,
        drive,
        solids,
        dt,
        warm,
        full_stamp || !warm,
    );
    let (cx0, cx1, cz0, cz1) = if full_stamp {
        (0usize, cres[0] - 1, 0usize, cres[2] - 1)
    } else {
        air_window(origin, ccell, cres, drive)
    };
    for cz in cz0..=cz1 {
        for iy in 0..cres[1] {
            for cx in cx0..=cx1 {
                let src = load3(coarse, at3(cres, cx, iy, cz));
                let ix0 = cx * factor;
                let iz0 = cz * factor;
                let iy0 = iy * fy;
                for dz in 0..factor {
                    let iz = iz0 + dz;
                    for dy in 0..fy {
                        let iy_f = iy0 + dy;
                        for dx in 0..factor {
                            store3(fine, at3(fine_res, ix0 + dx, iy_f, iz), src);
                        }
                    }
                }
            }
        }
    }
    report
}

/// Move dust with the wake residual. The sum of `vol` is put back after the flux.
///
/// Returns the absolute gap between the flux sum and the sum that was restored,
/// in cubic metres. A clean step is a few ulps.
pub fn advect_density_by_air(
    vol: &mut [f32],
    scratch: &mut [f32],
    vel: &[f32],
    origin: [f32; 3],
    cell: [f32; 3],
    res: [usize; 3],
    drive: &AirDrive,
    solids: &[AirSolid],
    dt: f32,
    columns: &[u32],
) -> f32 {
    let n = res[0].saturating_mul(res[1]).saturating_mul(res[2]);
    if dt <= 0.0
        || n == 0
        || columns.is_empty()
        || vol.len() < n
        || scratch.len() < n
        || vel.len() < n * 3
        || res[0] == 0
    {
        return 0.0;
    }
    let mut x_lo = res[0];
    let mut x_hi = 0usize;
    let mut z_lo = res[2];
    let mut z_hi = 0usize;
    for &col in columns {
        let ix = (col & 0xffff) as usize;
        let iz = (col >> 16) as usize;
        if ix >= res[0] || iz >= res[2] {
            continue;
        }
        x_lo = x_lo.min(ix);
        x_hi = x_hi.max(ix);
        z_lo = z_lo.min(iz);
        z_hi = z_hi.max(iz);
    }
    if x_lo > x_hi || z_lo > z_hi {
        return 0.0;
    }
    let xa = x_lo.saturating_sub(1);
    let za = z_lo.saturating_sub(1);
    let xb = (x_hi + 1).min(res[0] - 1);
    let zb = (z_hi + 1).min(res[2] - 1);
    let mut before = 0.0f64;
    for iz in za..=zb {
        for iy in 0..res[1] {
            let a = at3(res, xa, iy, iz);
            let b = at3(res, xb, iy, iz) + 1;
            scratch[a..b].copy_from_slice(&vol[a..b]);
            for v in &vol[a..b] {
                before += *v as f64;
            }
        }
    }
    let dx = cell[0].max(1.0e-4);
    let dz = cell[2].max(1.0e-4);
    let aabb = solids_aabb(solids);
    let mut stream_row = [[0.0f32; 3]; 160];
    let stream_n = if res[0] <= stream_row.len() {
        res[0]
    } else {
        0
    };
    for ix in xa..=xb {
        if ix < stream_n {
            let x = origin[0] + (ix as f32 + 0.5) * cell[0];
            stream_row[ix] = stream_at(x, drive);
        }
    }
    let streams: &[[f32; 3]] = if stream_n == 0 {
        &[]
    } else {
        &stream_row[..stream_n]
    };
    let nx = res[0];
    let factor = AIR_COARSE.max(1);
    let cnx = res[0] / factor;
    let cny = res[1];
    let cnz = res[2] / factor;
    let couple = AIR_DENSITY_COUPLING;
    AIR_WAKE.with(|wake_slot| {
        let mut wake = wake_slot.borrow_mut();
        let cn = cnx.saturating_mul(cny).saturating_mul(cnz);
        if cn == 0 {
            return;
        }
        if wake.len() < cn * 2 {
            wake.resize(cn * 2, 0.0);
        }
        let cx0 = xa / factor;
        let cx1 = (xb / factor).min(cnx - 1);
        let cz0 = za / factor;
        let cz1 = (zb / factor).min(cnz - 1);
        let stride_y = cnx;
        let stride_z = cnx * cny;
        for cz in cz0..=cz1 {
            for iy in 0..cny {
                for cx in cx0..=cx1 {
                    let ix = (cx * factor).min(res[0] - 1);
                    let iz = (cz * factor).min(res[2] - 1);
                    let pc = [
                        origin[0] + (cx as f32 + 0.5) * cell[0] * factor as f32,
                        origin[1] + (iy as f32 + 0.5) * cell[1],
                        origin[2] + (cz as f32 + 0.5) * cell[2] * factor as f32,
                    ];
                    let (rx, rz) = if point_in_aabb(pc, aabb) && solid_occupancy(pc, solids) > 0.42
                    {
                        (0.0, 0.0)
                    } else {
                        let v = load3(vel, at3(res, ix, iy, iz));
                        let fs = row_stream(streams, ix, pc[0], drive);
                        (v[0] - fs[0], v[2] - fs[2])
                    };
                    let i = (cx + iy * stride_y + cz * stride_z) * 2;
                    wake[i] = rx;
                    wake[i + 1] = rz;
                }
            }
        }
        let wake = &wake[..cn * 2];
        COL_LIVE.with(|slot| {
            let mut live = slot.borrow_mut();
            let plane = nx.saturating_mul(res[2]);
            if live.len() < plane {
                live.resize(plane, 0);
            }
            live[..plane].fill(0);
            for &col in columns {
                let ix = (col & 0xffff) as usize;
                let iz = (col >> 16) as usize;
                if ix < nx && iz < res[2] {
                    live[ix + iz * nx] = 1;
                }
            }
            for &col in columns {
                let ix = (col & 0xffff) as usize;
                let iz = (col >> 16) as usize;
                if ix >= nx || iz >= res[2] {
                    continue;
                }
                flux_column(
                    scratch, vol, wake, res, stride_y, stride_z, factor, couple, dt, dx, dz, ix,
                    iz, xb, zb, true, true,
                );
                if ix > 0 && live[(ix - 1) + iz * nx] == 0 {
                    flux_column(
                        scratch,
                        vol,
                        wake,
                        res,
                        stride_y,
                        stride_z,
                        factor,
                        couple,
                        dt,
                        dx,
                        dz,
                        ix - 1,
                        iz,
                        xb,
                        zb,
                        true,
                        false,
                    );
                }
                if iz > 0 && live[ix + (iz - 1) * nx] == 0 {
                    flux_column(
                        scratch,
                        vol,
                        wake,
                        res,
                        stride_y,
                        stride_z,
                        factor,
                        couple,
                        dt,
                        dx,
                        dz,
                        ix,
                        iz - 1,
                        xb,
                        zb,
                        false,
                        true,
                    );
                }
            }
        });
    });
    let mut after = 0.0f64;
    for iz in za..=zb {
        for iy in 0..res[1] {
            let a = at3(res, xa, iy, iz);
            let b = at3(res, xb, iy, iz) + 1;
            for v in &scratch[a..b] {
                after += *v as f64;
            }
        }
    }
    let gap = (before - after).abs() as f32;
    if after > 1.0e-12 {
        let scale = (before / after) as f32;
        if scale.is_finite() && (scale - 1.0).abs() > 0.0 {
            for iz in za..=zb {
                for iy in 0..res[1] {
                    let a = at3(res, xa, iy, iz);
                    let b = at3(res, xb, iy, iz) + 1;
                    for v in &mut scratch[a..b] {
                        *v *= scale;
                    }
                }
            }
        }
    }
    for iz in za..=zb {
        for iy in 0..res[1] {
            let a = at3(res, xa, iy, iz);
            let b = at3(res, xb, iy, iz) + 1;
            vol[a..b].copy_from_slice(&scratch[a..b]);
        }
    }
    gap
}

#[inline(always)]
fn flux_column(
    scratch: &mut [f32],
    vol: &[f32],
    wake: &[f32],
    res: [usize; 3],
    stride_y: usize,
    stride_z: usize,
    factor: usize,
    couple: f32,
    dt: f32,
    dx: f32,
    dz: f32,
    ix: usize,
    iz: usize,
    xb: usize,
    zb: usize,
    do_x: bool,
    do_z: bool,
) {
    let cix = ix / factor;
    let ciz = iz / factor;
    let ny = res[1];
    for iy in 0..ny {
        let i0 = ix + iy * res[0] + iz * res[0] * ny;
        let base = (cix + iy * stride_y + ciz * stride_z) * 2;
        if do_x && ix < xb {
            let nb = ((ix + 1) / factor + iy * stride_y + ciz * stride_z) * 2;
            let face = 0.5 * (wake[base] + wake[nb]) * couple;
            transfer(scratch, vol, i0, i0 + 1, face, dt, dx);
        }
        if do_z && iz < zb {
            let nb = (cix + iy * stride_y + (iz + 1) / factor * stride_z) * 2;
            let face = 0.5 * (wake[base + 1] + wake[nb + 1]) * couple;
            transfer(scratch, vol, i0, i0 + res[0] * ny, face, dt, dz);
        }
    }
}

#[inline(always)]
fn transfer(next: &mut [f32], src: &[f32], a: usize, b: usize, face: f32, dt: f32, cell: f32) {
    if !face.is_finite() || face == 0.0 || src[a] <= 0.0 && src[b] <= 0.0 {
        return;
    }
    let frac = (face.abs() * dt / cell).min(0.18);
    if frac <= 0.0 {
        return;
    }
    if face > 0.0 {
        let take = src[a] * frac;
        next[a] -= take;
        next[b] += take;
    } else {
        let take = src[b] * frac;
        next[b] -= take;
        next[a] += take;
    }
}

fn solid_occupancy(p: [f32; 3], solids: &[AirSolid]) -> f32 {
    let mut best = 0.0f32;
    for solid in solids {
        best = best.max(occupancy(p, solid));
    }
    best
}

fn air_window(
    origin: [f32; 3],
    cell: [f32; 3],
    res: [usize; 3],
    drive: &AirDrive,
) -> (usize, usize, usize, usize) {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;
    for &lat in &[-AIR_WIN_LAT, AIR_WIN_LAT] {
        for &along in &[AIR_WIN_Z0, AIR_WIN_Z1] {
            let p = chassis_point(drive, lat, 0.0, along);
            min_x = min_x.min(p[0]);
            max_x = max_x.max(p[0]);
            min_z = min_z.min(p[2]);
            max_z = max_z.max(p[2]);
        }
    }
    let ix0 = index_floor(min_x, origin[0], cell[0], res[0]);
    let ix1 = index_floor(max_x, origin[0], cell[0], res[0]);
    let iz0 = index_floor(min_z, origin[2], cell[2], res[2]);
    let iz1 = index_floor(max_z, origin[2], cell[2], res[2]);
    (ix0.min(ix1), ix0.max(ix1), iz0.min(iz1), iz0.max(iz1))
}

fn index_floor(world: f32, origin: f32, cell: f32, n: usize) -> usize {
    if n == 0 || cell <= 0.0 {
        return 0;
    }
    let i = ((world - origin) / cell).floor() as i32;
    i.clamp(0, n as i32 - 1) as usize
}

fn wake_vorticity(
    vel: &[f32],
    origin: [f32; 3],
    cell: [f32; 3],
    res: [usize; 3],
    drive: &AirDrive,
    ix0: usize,
    ix1: usize,
    iz0: usize,
    iz1: usize,
) -> (f32, u32) {
    let mut peak = 0.0f32;
    let mut cells = 0u32;
    if ix1 <= ix0 + 1 || iz1 <= iz0 + 1 {
        return (0.0, 0);
    }
    for iz in (iz0 + 1)..iz1 {
        for iy in 0..res[1] {
            for ix in (ix0 + 1)..ix1 {
                let p = cell_xyz(origin, cell, ix, iy, iz);
                let rel = sub3(p, drive.chassis);
                let lat = dot3(rel, drive.right);
                let along = dot3(rel, drive.forward);
                let y = p[1];
                if lat.abs() > drive.wake_lat
                    || y < drive.wake_y0
                    || y > drive.wake_y1
                    || along < drive.wake_z0
                    || along > drive.wake_z1
                {
                    continue;
                }
                let w = vort_xz(vel, res, ix, iy, iz, cell);
                if !w.is_finite() {
                    continue;
                }
                let avg = 0.5 * w.abs()
                    + 0.125
                        * (vort_xz(vel, res, ix + 1, iy, iz, cell).abs()
                            + vort_xz(vel, res, ix - 1, iy, iz, cell).abs()
                            + vort_xz(vel, res, ix, iy, iz + 1, cell).abs()
                            + vort_xz(vel, res, ix, iy, iz - 1, cell).abs());
                if avg.is_finite() {
                    peak = peak.max(avg);
                    if avg > WAKE_VORTICITY_MIN {
                        cells += 1;
                    }
                }
            }
        }
    }
    (peak, cells)
}

#[inline(always)]
fn vort_xz(vel: &[f32], res: [usize; 3], ix: usize, iy: usize, iz: usize, cell: [f32; 3]) -> f32 {
    let nx = res[0];
    let nz = res[2];
    if nx < 3 || nz < 3 || ix == 0 || iz == 0 || ix + 1 >= nx || iz + 1 >= nz {
        return 0.0;
    }
    let vp = load3(vel, at3(res, ix + 1, iy, iz));
    let vm = load3(vel, at3(res, ix - 1, iy, iz));
    let wp = load3(vel, at3(res, ix, iy, iz + 1));
    let wm = load3(vel, at3(res, ix, iy, iz - 1));
    let dvdx = (vp[2] - vm[2]) / (2.0 * cell[0]);
    let dudz = (wp[0] - wm[0]) / (2.0 * cell[2]);
    dvdx - dudz
}

#[inline(always)]
fn sample_layer(
    vel: &[f32],
    origin: [f32; 3],
    cell: [f32; 3],
    res: [usize; 3],
    drive: &AirDrive,
    iy: usize,
    x: f32,
    z: f32,
) -> [f32; 3] {
    let fx = (x - origin[0]) / cell[0] - 0.5;
    let fz = (z - origin[2]) / cell[2] - 0.5;
    if !fx.is_finite() || !fz.is_finite() {
        return stream_at(x, drive);
    }
    let x0 = fx.floor().clamp(0.0, (res[0] - 1) as f32) as usize;
    let z0 = fz.floor().clamp(0.0, (res[2] - 1) as f32) as usize;
    let x1 = (x0 + 1).min(res[0] - 1);
    let z1 = (z0 + 1).min(res[2] - 1);
    let tx = (fx - x0 as f32).clamp(0.0, 1.0);
    let tz = (fz - z0 as f32).clamp(0.0, 1.0);
    let s00 = load3(vel, at3(res, x0, iy, z0));
    let s10 = load3(vel, at3(res, x1, iy, z0));
    let s01 = load3(vel, at3(res, x0, iy, z1));
    let s11 = load3(vel, at3(res, x1, iy, z1));
    let a = lerp3(s00, s10, tx);
    let b = lerp3(s01, s11, tx);
    lerp3(a, b, tz)
}

/// Free-stream world air velocity at `world_x` for this drive: the wind plus
/// the slipstream the dust stream hold drags along in the lee.
#[inline(always)]
pub fn stream_at(world_x: f32, drive: &AirDrive) -> [f32; 3] {
    let forward = drive.forward;
    let right = drive.right;
    let rel_x = drive.wind_x - forward[0] * drive.speed;
    let rel_z = drive.wind_z - forward[2] * drive.speed;
    let rel = [rel_x, 0.0, rel_z];
    let gu = dot3(rel, right);
    let gv = dot3(rel, forward) * dust_stream_hold(world_x);
    let planar = add3(scale3(right, gu), scale3(forward, gv));
    add3(planar, scale3(forward, drive.speed))
}

#[inline(always)]
fn solid_target(p: [f32; 3], solids: &[AirSolid], freestream: [f32; 3]) -> (f32, [f32; 3]) {
    let mut box_occ = 0.0f32;
    let mut box_vel = freestream;
    let mut cyl_occ = 0.0f32;
    let mut cyl_vel = freestream;
    for solid in solids {
        let occ = occupancy(p, solid);
        if occ <= 0.0 {
            continue;
        }
        let vel = solid_velocity(p, solid);
        if solid.cylinder {
            if occ > cyl_occ {
                cyl_occ = occ;
                cyl_vel = vel;
            }
        } else if occ > box_occ {
            box_occ = occ;
            box_vel = vel;
        }
    }
    if cyl_occ > 0.30 && cyl_occ + 0.05 >= box_occ {
        (cyl_occ, cyl_vel)
    } else if box_occ > 0.0 {
        (box_occ, box_vel)
    } else {
        (0.0, freestream)
    }
}

#[inline(always)]
fn solid_velocity(p: [f32; 3], solid: &AirSolid) -> [f32; 3] {
    let mut v = solid.velocity;
    if solid.cylinder && solid.spin != 0.0 {
        let r = sub3(p, solid.center);
        let tang = cross3(solid.right, r);
        v = add3(v, scale3(tang, solid.spin));
    }
    v
}

#[inline(always)]
fn occupancy(p: [f32; 3], solid: &AirSolid) -> f32 {
    let d = sub3(p, solid.center);
    let lx = dot3(d, solid.right);
    let ly = dot3(d, solid.up);
    let lz = dot3(d, solid.forward);
    let sd = if solid.cylinder {
        let axial = lx.abs() - solid.half[0];
        let radial = (ly * ly + lz * lz).sqrt() - solid.radius.max(0.05);
        sd_box(axial, radial, f32::MIN)
    } else {
        sd_box(
            lx.abs() - solid.half[0],
            ly.abs() - solid.half[1],
            lz.abs() - solid.half[2],
        )
    };
    let feather = AIR_FEATHER_M;
    let t = ((feather - sd) / (2.0 * feather)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
fn sd_box(dx: f32, dy: f32, dz: f32) -> f32 {
    if dz == f32::MIN {
        let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
        let inside = dx.max(dy).min(0.0);
        return outside + inside;
    }
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2) + dz.max(0.0).powi(2)).sqrt();
    let inside = dx.max(dy).max(dz).min(0.0);
    outside + inside
}

fn chassis_point(drive: &AirDrive, lat: f32, y: f32, along: f32) -> [f32; 3] {
    add3(
        drive.chassis,
        add3(
            scale3(drive.right, lat),
            add3(
                scale3(drive.up, y - drive.chassis[1]),
                scale3(drive.forward, along),
            ),
        ),
    )
}

#[inline(always)]
fn cell_xyz(origin: [f32; 3], cell: [f32; 3], ix: usize, iy: usize, iz: usize) -> [f32; 3] {
    [
        origin[0] + (ix as f32 + 0.5) * cell[0],
        origin[1] + (iy as f32 + 0.5) * cell[1],
        origin[2] + (iz as f32 + 0.5) * cell[2],
    ]
}

#[inline(always)]
fn at3(res: [usize; 3], ix: usize, iy: usize, iz: usize) -> usize {
    ix + iy * res[0] + iz * res[0] * res[1]
}

#[inline(always)]
fn load3(buf: &[f32], i: usize) -> [f32; 3] {
    let b = i * 3;
    [buf[b], buf[b + 1], buf[b + 2]]
}

#[inline(always)]
fn store3(buf: &mut [f32], i: usize, v: [f32; 3]) {
    let b = i * 3;
    buf[b] = v[0];
    buf[b + 1] = v[1];
    buf[b + 2] = v[2];
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

#[inline(always)]
fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline(always)]
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline(always)]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline(always)]
fn scale3(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn norm3(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len > 1.0e-6 && len.is_finite() {
        scale3(v, 1.0 / len)
    } else {
        fallback
    }
}

/// Bow height in metres. `k * speed * sqrt(depth / g)`, not a speed×depth flux.
pub fn bow_height_m(speed: f32, depth: f32) -> f32 {
    let v = speed.max(0.0);
    let d = depth.max(0.0);
    if v < 0.05 || d < 0.02 {
        return 0.0;
    }
    (0.42 * v * (d / GRAVITY).sqrt()).min(0.24).min(d * 0.85)
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        static REF_MARK: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    }

    fn sheet_work_boxes_ref(h: &[f32], res: usize, eps: f32, allow_split: bool) -> Vec<(usize, usize, usize, usize)> {
        let mut i0 = res;
        let mut i1 = 0usize;
        let mut j0 = res;
        let mut j1 = 0usize;
        let mut wet = 0usize;
        for j in 0..res {
            let row = &h[j * res..j * res + res];
            // Count without a branch per cell; only a wet row pays for its ends.
            let c = row.iter().map(|&x| (x > eps) as usize).sum::<usize>();
            if c == 0 {
                continue;
            }
            wet += c;
            let first = row.iter().position(|&x| x > eps).unwrap_or(0);
            let last = row.iter().rposition(|&x| x > eps).unwrap_or(0);
            i0 = i0.min(first);
            i1 = i1.max(last);
            j0 = j0.min(j);
            j1 = j1.max(j);
        }
        if wet == 0 {
            return Vec::new();
        }
        let halo3 = |a0: usize, a1: usize| (a0.saturating_sub(3), (a1 + 3).min(res - 1));
        let (x0, x1) = halo3(i0, i1);
        let (y0, y1) = halo3(j0, j1);
        let solid = wet == (i1 - i0 + 1) * (j1 - j0 + 1);
        if !allow_split || solid {
            return vec![(x0, x1, y0, y1)];
        }
        let n = res * res;
        REF_MARK.with(|mark| {
            let mut mark = mark.borrow_mut();
            if mark.len() < n {
                mark.resize(n, 0);
            }
            // Every wet cell lies in the wet bounding box, so the fill and the
            // reset only visit that box. Order of discovery is unchanged.
            for j in j0..=j1 {
                mark[j * res + i0..=j * res + i1].fill(0);
            }
            let mut queue = Vec::new();
            let mut boxes = Vec::new();
            for j in j0..=j1 {
                for i in i0..=i1 {
                    let s = j * res + i;
                    if h[s] <= eps || mark[s] != 0 {
                        continue;
                    }
                    let (mut ci0, mut ci1, mut cj0, mut cj1) = (i, i, j, j);
                    mark[s] = 1;
                    queue.clear();
                    queue.push(s);
                    while let Some(cur) = queue.pop() {
                        let cj = cur / res;
                        let ci = cur - cj * res;
                        ci0 = ci0.min(ci);
                        ci1 = ci1.max(ci);
                        cj0 = cj0.min(cj);
                        cj1 = cj1.max(cj);
                        let neigh = [
                            if ci > 0 { Some(cur - 1) } else { None },
                            if ci + 1 < res { Some(cur + 1) } else { None },
                            if cj > 0 { Some(cur - res) } else { None },
                            if cj + 1 < res { Some(cur + res) } else { None },
                        ];
                        for nidx in neigh.into_iter().flatten() {
                            if mark[nidx] == 0 && h[nidx] > eps {
                                mark[nidx] = 1;
                                queue.push(nidx);
                            }
                        }
                    }
                    // Two blur passes of radius 2, plus the gradient, sit inside this halo.
                    const SPLIT_HALO: usize = 6;
                    boxes.push((
                        ci0.saturating_sub(SPLIT_HALO),
                        (ci1 + SPLIT_HALO).min(res - 1),
                        cj0.saturating_sub(SPLIT_HALO),
                        (cj1 + SPLIT_HALO).min(res - 1),
                    ));
                }
            }
            merge_boxes(&mut boxes);
            // One body keeps the halo of 3, including a puddle with holes. Only
            // separated disks pay for their own boxes.
            if boxes.len() == 1 || boxes.len() > 8 {
                vec![(x0, x1, y0, y1)]
            } else {
                boxes
            }
        })
    }


    /// The run-length wet bodies give the same work boxes, in the same order,
    /// as the flood fill they replace.
    #[test]
    fn run_bodies_match_the_flood_fill() {
        let mut seed = 0x9e37_79b9_u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for case in 0..400 {
            let res = [16usize, 40, 100, FINE_RES][case % 4];
            let mut h = vec![0.0f32; res * res];
            let blobs = 1 + (rnd() * 12.0) as usize;
            for _ in 0..blobs {
                let cx = rnd() * res as f32;
                let cz = rnd() * res as f32;
                let r = 1.0 + rnd() * res as f32 * 0.2;
                for j in 0..res {
                    for i in 0..res {
                        let dx = i as f32 - cx;
                        let dz = j as f32 - cz;
                        if dx * dx + dz * dz < r * r && rnd() > 0.15 {
                            h[j * res + i] = 0.01 + rnd() * 0.1;
                        }
                    }
                }
            }
            if case % 7 == 0 {
                for x in h.iter_mut() {
                    if rnd() < 0.3 {
                        *x = 0.02;
                    }
                }
            }
            let want = sheet_work_boxes_ref(&h, res, 1.0e-4, true);
            let got = sheet_work_boxes(&h, res, 1.0e-4, true);
            assert_eq!(want, got, "case {case}");
        }
    }

    /// The branch-free open transport writes the same depths and the same
    /// removed/added sums as the serial edge loop, bit for bit.
    #[test]
    fn transport_open_rows_match_the_serial_edges() {
        let mut seed = 0x2545_f491_u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        for case in 0..96 {
            let res = [12usize, 37, 64, FINE_RES][case % 4];
            let n = res * res;
            let mut h = vec![0.0f32; n];
            let mut vel = vec![0.0f32; n];
            let mut gate = vec![1u8; n];
            let mut bed = vec![0.0f32; n];
            for idx in 0..n {
                let r = rnd();
                h[idx] = if r < 0.25 {
                    0.0
                } else if r < 0.3 {
                    5.0e-7
                } else if r < 0.33 {
                    5.0e-9
                } else {
                    rnd() * 0.3
                };
                let q = rnd();
                vel[idx] = if q < 0.15 {
                    0.0
                } else if q < 0.2 {
                    (rnd() - 0.5) * 3.0e-5
                } else {
                    (rnd() - 0.5) * 6.0
                };
                gate[idx] = (rnd() > 0.2) as u8;
                bed[idx] = (rnd() - 0.5) * 0.05;
            }
            let i0 = (rnd() * 3.0) as usize;
            let j0 = (rnd() * 3.0) as usize;
            let i1 = res - 1 - (rnd() * 3.0) as usize;
            let j1 = res - 1 - (rnd() * 3.0) as usize;
            let cell = 0.02 + rnd() * 0.2;
            let dt = 0.002 + rnd() * 0.03;
            for along_x in [true, false] {
                for with_bed in [false, true] {
                    let b = if with_bed { Some(&bed[..]) } else { None };
                    let mut h_ref = h.clone();
                    let mut h_new = h.clone();
                    let mut sc_ref = Scratch::new();
                    sc_ref.fit(n);
                    let mut sc_new = Scratch::new();
                    sc_new.fit(n);
                    transport_open_ref(
                        &mut h_ref, &vel, &gate, res, cell, dt, i0, i1, j0, j1, along_x, b, 0.45,
                        &mut sc_ref,
                    );
                    match b {
                        Some(bed) => transport_open_rows::<true>(
                            &mut h_new, &vel, &gate, res, cell, dt, i0, i1, j0, j1, along_x, bed,
                            0.45, &mut sc_new,
                        ),
                        None => transport_open_rows::<false>(
                            &mut h_new, &vel, &gate, res, cell, dt, i0, i1, j0, j1, along_x, &[],
                            0.45, &mut sc_new,
                        ),
                    }
                    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                    assert_eq!(bits(&h_ref), bits(&h_new), "depth case {case} x {along_x} bed {with_bed}");
                    assert_eq!(bits(&sc_ref.removed), bits(&sc_new.removed), "removed case {case}");
                    assert_eq!(bits(&sc_ref.added), bits(&sc_new.added), "added case {case}");
                    // The plain build of the same loops (no AVX2) agrees too.
                    let mut h_plain = h.clone();
                    let mut sc_plain = Scratch::new();
                    sc_plain.fit(n);
                    match b {
                        Some(bed) => transport_open_rows_body::<true>(
                            &mut h_plain, &vel, &gate, res, cell, dt, i0, i1, j0, j1, along_x,
                            bed, 0.45, &mut sc_plain,
                        ),
                        None => transport_open_rows_body::<false>(
                            &mut h_plain, &vel, &gate, res, cell, dt, i0, i1, j0, j1, along_x,
                            &[], 0.45, &mut sc_plain,
                        ),
                    }
                    assert_eq!(bits(&h_ref), bits(&h_plain), "plain depth case {case}");
                    assert_eq!(bits(&sc_ref.removed), bits(&sc_plain.removed), "plain removed {case}");
                    assert_eq!(bits(&sc_ref.added), bits(&sc_plain.added), "plain added {case}");
                }
            }
        }
    }

    #[test]
    fn slope_accelerates_downhill() {
        let res = 16usize;
        let cell = 0.10f32;
        let n = res * res;
        let mut h = vec![0.05f32; n];
        let mut u = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let mut bed = vec![0.0f32; n];
        for j in 0..res {
            for i in 0..res {
                bed[j * res + i] = -0.04 * j as f32;
            }
        }
        let nu = vec![40.0f32; n];
        let gate = vec![1u8; n];
        step_sheet(
            &mut h,
            &mut u,
            &mut v,
            &bed,
            &nu,
            &gate,
            res,
            cell,
            0.05,
            SheetDrive::Gravity { vorticity: 0.0 },
        );
        let idx = 8 * res + 8;
        assert!(v[idx] > 0.002, "downhill v {}", v[idx]);
    }

    #[test]
    fn viscosity_orders_clay_silt_sand_and_thins() {
        let clay = dynamic_viscosity(Grain::Clay, 0.55);
        let silt = dynamic_viscosity(Grain::Silt, 0.55);
        let sand = dynamic_viscosity(Grain::Sand, 0.55);
        assert!(clay > silt * 1.8, "clay {clay} silt {silt}");
        assert!(silt > sand * 1.5, "silt {silt} sand {sand}");
        let stiff = dynamic_viscosity(Grain::Clay, 0.30);
        let mid = dynamic_viscosity(Grain::Clay, 0.55);
        let thin = dynamic_viscosity(Grain::Clay, 0.85);
        assert!(stiff > mid && mid > thin, "{stiff} {mid} {thin}");
    }

    #[test]
    fn rut_heal_times_match_the_mix() {
        let clay = rut_heal(Grain::Clay, 0.55);
        let silt = rut_heal(Grain::Silt, 0.55);
        let thin = rut_heal(Grain::Clay, 0.90);
        eprintln!(
            "heal clay {:.2}s silt {:.2}s thin {:.2}s  relief8 clay {:.4}/{:.4} silt {:.4}",
            clay.level_s, silt.level_s, thin.level_s, clay.relief_8s, clay.relief0, silt.relief_8s
        );
        assert!(
            (10.0..=15.0).contains(&clay.level_s),
            "clay heal {:.2}",
            clay.level_s
        );
        assert!(
            (10.0..=15.0).contains(&silt.level_s),
            "silt heal {:.2}",
            silt.level_s
        );
        assert!(
            (1.0..=3.0).contains(&thin.level_s),
            "thin heal {:.2}",
            thin.level_s
        );
        assert!(clay.still_deep_at_8s, "clay relief fell early");
        assert!(silt.still_deep_at_8s, "silt relief fell early");
    }

    #[test]
    fn sheet_conserves_and_runs_downhill() {
        let res = 24usize;
        let cell = 0.2f32;
        let n = res * res;
        let mut h = vec![0.0f32; n];
        let mut u = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        let mut bed = vec![0.0f32; n];
        let nu = vec![dynamic_viscosity(Grain::Clay, 0.55); n];
        let gate = vec![1u8; n];
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                bed[idx] = -0.16 * j as f32;
                if (4..9).contains(&j) && (8..16).contains(&i) {
                    h[idx] = 0.10;
                }
            }
        }
        let before = column_sum(&h, cell);
        let centroid = |h: &[f32]| -> f32 {
            let mut m = 0.0f32;
            let mut mj = 0.0f32;
            for j in 0..res {
                for i in 0..res {
                    let s = h[j * res + i];
                    m += s;
                    mj += s * j as f32;
                }
            }
            if m <= 1.0e-6 {
                0.0
            } else {
                mj / m
            }
        };
        let c0 = centroid(&h);
        for _ in 0..60 {
            step_sheet(
                &mut h,
                &mut u,
                &mut v,
                &bed,
                &nu,
                &gate,
                res,
                cell,
                0.05,
                SheetDrive::Gravity { vorticity: 0.02 },
            );
        }
        let after = column_sum(&h, cell);
        let c1 = centroid(&h);
        assert!(
            (after - before).abs() < 1.0e-3,
            "volume {before} -> {after}"
        );
        assert!(
            c1 > c0 + 1.5,
            "slurry did not run downhill, centroid {c0:.2} -> {c1:.2}"
        );
    }

    #[test]
    fn dust_does_not_lift_when_wet_and_window_keeps_volume() {
        assert!(!dust_may_lift(0.55, 0.0));
        assert!(!dust_may_lift(0.2, 0.008));
        assert!(dust_may_lift(0.4, 0.004));
        let mut fluid = FluidCore::new([0.0, 0.0]);
        let poured = fluid.pour_water(-1.0, 1.0, -1.0, 1.0, 0.12);
        let before = fluid.ledger().total_m3;
        assert!(
            (before - poured).abs() < 1.0e-3,
            "pour {poured} ledger {before}"
        );
        fluid.recenter([55.0, 0.0]);
        fluid.recenter([0.0, 0.0]);
        let after = fluid.ledger();
        assert!(
            (after.total_m3 - before).abs() < 1.0e-3,
            "recenter {} -> {}",
            before,
            after.total_m3
        );
        assert!(after.water_m3 > poured * 0.5, "water did not return");
        let dry = fluid.inject_dust(
            DustInject {
                x: 2.0,
                z: 2.0,
                slip: 0.4,
                moisture: 0.1,
                water_m: 0.0,
                offer_m3_s: 0.01,
            },
            0.1,
        );
        let wet = fluid.inject_dust(
            DustInject {
                x: 0.0,
                z: 0.0,
                slip: 0.8,
                moisture: 0.7,
                water_m: 0.02,
                offer_m3_s: 0.01,
            },
            0.1,
        );
        assert!(dry > 0.0, "dry ground took no dust");
        assert!(wet == 0.0, "wet ground lofted dust");
    }

    #[test]
    fn moisture_stays_on_the_ground_when_the_window_moves() {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        fluid.raise_moisture_disk(0.2, -0.4, 0.45, 0.78);
        assert!(
            fluid.sample(0.2, -0.4).moisture > 0.7,
            "disk {}",
            fluid.sample(0.2, -0.4).moisture
        );
        fluid.recenter([30.0, 0.0]);
        assert!(
            fluid.sample(0.2, -0.4).moisture > 0.7,
            "parked moisture dropped {}",
            fluid.sample(0.2, -0.4).moisture
        );
        fluid.recenter([0.0, 0.0]);
        assert!(
            fluid.sample(0.2, -0.4).moisture > 0.7,
            "restored moisture {}",
            fluid.sample(0.2, -0.4).moisture
        );
        assert!(
            fluid.sample(6.0, 6.0).moisture < 0.05,
            "dry ground gained moisture {}",
            fluid.sample(6.0, 6.0).moisture
        );
    }

    /// Hybrid H2: water a tire throws comes back. While the tire runs the
    /// drops are booked as spray; once they have landed the air is empty,
    /// `spray_m3` is back to zero and the whole volume is in the sheet.
    /// The throw is wings and a tail, not a fountain: drops leave sideways
    /// or backwards and never stand much higher than the rim speed allows.
    #[test]
    fn tire_spray_lands_back_in_the_sheet() {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        fluid.pour_water(-6.0, 6.0, -6.0, 6.0, 0.10);
        let dt = 1.0 / 30.0;
        let mut peak = 0.0f32;
        let mut most_in_air = 0.0f32;
        let mut sideways = 0usize;
        let mut seen = 0usize;
        for k in 0..12 {
            fluid.push_tire(TirePush {
                x: 0.0,
                z: -1.0 + k as f32 * 8.0 * dt,
                forward_x: 0.0,
                forward_z: 1.0,
                radius: 0.38,
                width: 0.18,
                speed: 8.0,
                slip: 0.0,
            });
            fluid.step(dt);
            most_in_air = most_in_air.max(fluid.spray_m3());
            for d in fluid.spray_drops() {
                peak = peak.max(d.pos[1]);
                let h = d.vel[0].hypot(d.vel[2]);
                seen += 1;
                if h > 0.5 * d.vel[1].abs() {
                    sideways += 1;
                }
            }
        }
        assert!(most_in_air > 1.0e-4, "no spray at 8 m/s: {most_in_air}");
        assert!(peak < 2.5, "spray stood {peak} m high");
        assert!(sideways * 2 > seen, "fountain: {sideways} of {seen} sideways");
        // The tire is gone. From here only the drops come down; the sheet
        // plus the air must hold the same volume all the way.
        let up = fluid.ledger();
        assert!(up.spray_m3 > 1.0e-4, "nothing in the air: {}", up.spray_m3);
        for _ in 0..60 {
            fluid.step(dt);
        }
        let down = fluid.ledger();
        assert!(fluid.spray_drops().is_empty(), "{} drops still up", fluid.spray_drops().len());
        assert!(down.spray_m3 < 1.0e-9, "spray left {}", down.spray_m3);
        // Most of it is back as standing water (some soaks in meanwhile).
        let back = down.water_m3 - up.water_m3;
        assert!(back > 0.9 * up.spray_m3, "landing put back {back} of {} m3", up.spray_m3);
        assert!(
            (down.total_m3 - up.total_m3).abs() < 1.0e-4 * up.total_m3,
            "volume {} -> {}",
            up.total_m3,
            down.total_m3
        );
    }

    /// A still tire in 40 cm of water. The 2 cm patch must not drain the cells
    /// it holds into its own rim, or a hole follows the patch.
    #[test]
    fn fine_patch_leaves_no_hole_in_deep_water() {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        fluid.pour_water(-6.0, 6.0, -6.0, 6.0, 0.40);
        let before = fluid.ledger().total_m3;
        let dt = 1.0 / 30.0;
        for _ in 0..30 {
            fluid.push_tire(TirePush {
                x: 0.0,
                z: 0.0,
                forward_x: 0.0,
                forward_z: 1.0,
                radius: 0.01,
                width: 0.01,
                speed: 0.0,
                slip: 0.0,
            });
            fluid.step(dt);
        }
        let after = fluid.ledger().total_m3;
        assert!(
            (after - before).abs() < 1.0e-3 * before,
            "volume {before} -> {after}"
        );
        // Inside the patch and around its rim the surface stays at the pond
        // level. Before the wet columns were held, cells under the patch fell
        // to about 17 cm.
        let mut lowest = f32::MAX;
        let mut at = (0.0, 0.0);
        let mut z = -2.0f32;
        while z <= 2.0 {
            let mut x = -2.0f32;
            while x <= 2.0 {
                if x.hypot(z) >= 0.5 {
                    let s = fluid.sample(x, z);
                    if s.water_m < lowest {
                        lowest = s.water_m;
                        at = (x, z);
                    }
                }
                x += 0.1;
            }
            z += 0.1;
        }
        assert!(lowest > 0.37, "hole {lowest:.4} m at {at:?}");
    }

    #[test]
    fn bow_height_is_metres() {
        let deep = bow_height_m(1.8, 0.40);
        let slow = bow_height_m(0.4, 0.40);
        let shallow = bow_height_m(1.8, 0.14);
        assert!(deep > 0.05, "deep {deep}");
        assert!(deep > slow + 0.04, "fast {deep} slow {slow}");
        assert!(deep > shallow + 0.03, "deep {deep} shallow {shallow}");
        assert_eq!(bow_height_m(0.0, 0.4), 0.0);
    }

    /// Slip under a submerged tire lifts soil into the water. The silt book
    /// closes, a parked window does not drop it, and a dry cell stays empty.
    #[test]
    fn tire_slip_stirs_a_closed_silt_book() {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        let poured = fluid.pour_water(-3.0, 3.0, -3.0, 3.0, 0.07);
        let before = fluid.ledger().total_m3;
        let dt = 1.0 / 30.0;
        let push = TirePush {
            x: 0.15,
            z: 0.2,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.38,
            width: 0.28,
            speed: 8.0,
            slip: 1.2,
        };
        for _ in 0..10 {
            fluid.push_tire(push);
            fluid.step(dt);
            assert!(
                fluid.silt_book_gap() < 1.0e-3,
                "silt gap {} stirred {}",
                fluid.silt_book_gap(),
                fluid.stirred_m3()
            );
            assert_eq!(fluid.non_finite_count(), 0);
        }
        let (foam, turbid) = max_near(&fluid, push.x, push.z, 0.7);
        assert!(turbid > 1.0e-4, "silt at the tire {turbid}");
        assert!(foam > 0.05, "foam on the waterline {foam}");
        let (far_foam, far_silt) = fluid.foam_turbid_at(8.0, 8.0);
        assert_eq!(far_foam, 0.0, "far foam {far_foam}");
        assert_eq!(far_silt, 0.0, "far silt {far_silt}");
        let mut dry_bad = 0u32;
        for j in 0..WINDOW_RES {
            for i in 0..WINDOW_RES {
                let idx = j * WINDOW_RES + i;
                let film = fluid.water[idx] + fluid.slurry[idx];
                if film < 1.0e-5 && (fluid.foam[idx] != 0.0 || fluid.turbid[idx] != 0.0) {
                    dry_bad += 1;
                }
            }
        }
        assert_eq!(dry_bad, 0, "dry cells holding tracer {dry_bad}");
        let parked_stirred = fluid.stirred_m3();
        fluid.recenter([0.0, 40.0]);
        assert!(
            fluid.silt_book_gap() < 1.0e-3,
            "parked silt gap {}",
            fluid.silt_book_gap()
        );
        fluid.recenter([0.0, 0.0]);
        assert!(
            (fluid.stirred_m3() - parked_stirred).abs() < 1.0e-6,
            "stirred changed across the window"
        );
        assert!(fluid.silt_book_gap() < 1.0e-3, "restored gap {}", fluid.silt_book_gap());
        // Solids the wheel lifted into the turbidity column (#99) come from the
        // bed and are booked on top of the standing water and slurry.
        let books = fluid.ledger();
        let after = books.total_m3 - books.entrained_m3;
        assert!(
            (after - before).abs() < 1.0e-3,
            "standing book moved {before} -> {after} pour {poured} lifted {}",
            books.entrained_m3
        );
        // A tire that is not moving does not paint a foam disc on the wet
        // interior. A crest at the pour edge is the shoreline, not the tread.
        let mut still = FluidCore::new([0.0, 2.0]);
        still.pour_water(-2.0, 2.0, 0.0, 4.0, 0.06);
        still.push_tire(TirePush {
            speed: 0.0,
            slip: 0.0,
            ..push
        });
        still.step(dt);
        let (hub_foam, hub_silt) = still.foam_turbid_at(push.x, push.z);
        assert_eq!(hub_silt, 0.0, "parked tire stirred {hub_silt}");
        let mut disc = 0u32;
        let steps = (0.7 / COARSE_CELL_M).ceil() as i32;
        for dj in -steps..=steps {
            for di in -steps..=steps {
                let px = push.x + di as f32 * COARSE_CELL_M;
                let pz = push.z + dj as f32 * COARSE_CELL_M;
                if (px - push.x).hypot(pz - push.z) > 0.7 {
                    continue;
                }
                let Some((i, j)) = still.coarse_ij(px, pz) else {
                    continue;
                };
                if !interior_wet(&still, i, j) {
                    continue;
                }
                let idx = j * WINDOW_RES + i;
                if still.foam[idx] > 0.0 || still.turbid[idx] > 0.0 {
                    disc += 1;
                }
            }
        }
        assert_eq!(disc, 0, "parked tire painted {disc} interior cells, hub foam {hub_foam}");
    }

    fn interior_wet(fluid: &FluidCore, i: usize, j: usize) -> bool {
        let res = WINDOW_RES;
        let idx = j * res + i;
        let h = fluid.water[idx] + fluid.slurry[idx];
        if h < 0.02 {
            return false;
        }
        for (di, dj) in [(-1i32, 0), (1, 0), (0, -1), (0, 1)] {
            let ni = i as i32 + di;
            let nj = j as i32 + dj;
            if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                return false;
            }
            let nidx = nj as usize * res + ni as usize;
            let nh = fluid.water[nidx] + fluid.slurry[nidx];
            if !nh.is_finite() || nh < 0.02 {
                return false;
            }
        }
        true
    }

    fn max_near(fluid: &FluidCore, x: f32, z: f32, radius: f32) -> (f32, f32) {
        let mut foam = 0.0f32;
        let mut turbid = 0.0f32;
        let steps = (radius / COARSE_CELL_M).ceil() as i32 + 1;
        for dj in -steps..=steps {
            for di in -steps..=steps {
                let px = x + di as f32 * COARSE_CELL_M;
                let pz = z + dj as f32 * COARSE_CELL_M;
                if (px - x).hypot(pz - z) > radius {
                    continue;
                }
                let (f, t) = fluid.foam_turbid_at(px, pz);
                foam = foam.max(f);
                turbid = turbid.max(t);
            }
        }
        (foam, turbid)
    }
}
