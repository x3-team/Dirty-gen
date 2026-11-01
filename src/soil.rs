//! Heightfield soil: Bekker sinkage, plastic ruts, wetness and shallow water.
//!
//! The cut and the spoil are different layers. `bed` is the ground the tire
//! carved. `placed` is soil that has already been put somewhere else (berm,
//! crest clod, impact ring). Standing water in a rut or a hole mixes with the
//! wet soil into one slurry. That slurry then flows downhill into the low rut
//! and into holes. A berm beside it slowly creeps into the slurry. A tire in
//! the slurry pushes a bow wave, thins a wake behind the tread, and leaves its
//! track in the slurry thickness.
//! Deep slurry also drags. The viscosity is the mix — clay, silt, or sand —
//! and it resists both the roll of the tread and the forward motion of the
//! tire. Pulling the tire out holds a short suction, then the seal breaks and
//! throws ribbons and drops taken from the slurry. An extreme exit tears the
//! rut lip, breaks the seal on every wheel, and drives out. What leaves the
//! column is the rut tear-out, the groove shed, and ribbons, sheets, and fine
//! spray in the air. Slurry that stays behind sits in the tread voids.
//! Spinning the wheel, or shedding the coat, lets that hold leave as ribbons.
//! What is in the grooves, what is thrown, and
//! what returns to the rut add up to the slurry that left the column. A roll
//! or a sideways shove in that slurry splits the new cut into the berm, the
//! throw, and the slurry that stays in the rut. Clay holds, so the shove is
//! short. Silt is thinner and travels farther. A dry crust does not become
//! slurry. The visible height is bed + placed + the slurry that is still on
//! the ground.
//!
//! Pore moisture falls on a dry spell. Clay, a fresh cut, and packed clay
//! grow a crust thickness, and shrinkage opens cracks on a fixed network.
//! A track that has rested on that crust hardens: the same tire sinks less
//! and cuts less. Sand, silt, and gravel do not crust. The bed, the placed
//! pile, and the cloud stay put. Thickness and crack opening are exported
//! for a shader. They are not a second height. A fissure is the same mud, opened.
//!
//! A rut is a vertical cut at a fixed world XZ. The tire does not push the
//! column along the ground normal, so a groove on a slope stays where the
//! wheel was instead of climbing the mesh.
//!
//! Mud, sand, gravel, and grass meet as shares of the cell, not as a hard
//! step. The share of each material is exchanged across that joint, so the
//! amount on the map stays the same. Pore moisture is exchanged on the same
//! edges. Silt, crust, and undisturbed ground are not a fifth share.

use glam::Vec3;

pub const MAP_SIZE: f32 = 40.0;
pub const MAP_RES: usize = 192;

/// Soil the tire can stand on. A mixed slurry is not a second kind: it is
/// the water and the wet soil as one thickness on the cell.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceKind {
    Undisturbed = 0,
    Grass = 1,
    Gravel = 2,
    DryCrust = 3,
    WetMud = 4,
    FreshCut = 5,
    /// Exposed edge of a rut. A broken step, not the pile beside it.
    CutLip = 6,
    /// Spoil placed beside the cut.
    Berm = 7,
    PlacedClod = 8,
    PackedClay = 9,
    /// Loose grains. They drain, they do not crust, and they do not slurry.
    Sand = 10,
    /// Fine flour. It holds water, goes slick, and stays silt.
    Silt = 11,
}

/// How standing water sits on a soil kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiquidRegime {
    Dry = 0,
    Film = 1,
    Slurry = 2,
}

/// Soil kinds plus the two liquid regimes the solver treats as states.
/// Sand and silt are their own soils, so this is twelve soils and two liquids.
pub const SURFACE_STATE_COUNT: usize = 14;

/// Channels of the material-share map. Mud, sand, gravel, grass.
pub const MIX_MUD: usize = 0;
pub const MIX_SAND: usize = 1;
pub const MIX_GRAVEL: usize = 2;
pub const MIX_GRASS: usize = 3;
pub const MIX_CHANNELS: usize = 4;

/// Jacobi passes that widen a material joint by one cell each.
pub const JOINT_PASSES: usize = 4;
/// Share of the difference moved across one edge per pass.
/// At most 1/4, so four neighbours cannot push a cell outside 0..1.
pub const JOINT_RATE: f32 = 0.25;

/// Channel of mud, sand, gravel, or grass. Any other kind is not a share.
pub fn mix_channel(kind: SurfaceKind) -> Option<usize> {
    match kind {
        SurfaceKind::WetMud => Some(MIX_MUD),
        SurfaceKind::Sand => Some(MIX_SAND),
        SurfaceKind::Gravel => Some(MIX_GRAVEL),
        SurfaceKind::Grass => Some(MIX_GRASS),
        _ => None,
    }
}

fn joint_exchange(
    mix: &mut [f32],
    prev_mix: &[f32],
    wet: &mut [f32],
    prev_wet: &[f32],
    base: &mut [f32],
    prev_base: &[f32],
    a: usize,
    b: usize,
) {
    let sa = prev_mix[a * MIX_CHANNELS]
        + prev_mix[a * MIX_CHANNELS + 1]
        + prev_mix[a * MIX_CHANNELS + 2]
        + prev_mix[a * MIX_CHANNELS + 3];
    let sb = prev_mix[b * MIX_CHANNELS]
        + prev_mix[b * MIX_CHANNELS + 1]
        + prev_mix[b * MIX_CHANNELS + 2]
        + prev_mix[b * MIX_CHANNELS + 3];
    if sa <= 1.0e-6 || sb <= 1.0e-6 {
        return;
    }
    let mut diff = 0.0;
    for channel in 0..MIX_CHANNELS {
        diff += (prev_mix[a * MIX_CHANNELS + channel] - prev_mix[b * MIX_CHANNELS + channel]).abs();
    }
    if diff <= 1.0e-6 {
        return;
    }
    for channel in 0..MIX_CHANNELS {
        let flux = JOINT_RATE
            * (prev_mix[a * MIX_CHANNELS + channel] - prev_mix[b * MIX_CHANNELS + channel]);
        mix[a * MIX_CHANNELS + channel] -= flux;
        mix[b * MIX_CHANNELS + channel] += flux;
    }
    let flux_wet = JOINT_RATE * (prev_wet[a] - prev_wet[b]);
    wet[a] -= flux_wet;
    wet[b] += flux_wet;
    let flux_base = JOINT_RATE * (prev_base[a] - prev_base[b]);
    base[a] -= flux_base;
    base[b] += flux_base;
}

/// Crest chunks stay lumps. Later passes put the rest of the cut on the berm.
const CREST_CLOD_CAP: f32 = 0.04;

/// A tire that leaves a cell and comes back after this many seconds cuts
/// another pass. Overlapping stamps from the same visit do not count, and
/// they do not fill the groove back in.
const PASS_GAP: f32 = 0.40;

impl SurfaceKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Grass,
            2 => Self::Gravel,
            3 => Self::DryCrust,
            4 => Self::WetMud,
            5 => Self::FreshCut,
            6 => Self::CutLip,
            7 => Self::Berm,
            8 => Self::PlacedClod,
            9 => Self::PackedClay,
            10 => Self::Sand,
            11 => Self::Silt,
            _ => Self::Undisturbed,
        }
    }

    /// CC0 layer: 0 mud_forest, 1 brown_mud_dry, 2 clay, 3 grass, 4 gravel.
    pub fn map_layer(self) -> u32 {
        match self {
            Self::Grass => 3,
            Self::Gravel => 4,
            Self::DryCrust | Self::Undisturbed | Self::Berm | Self::Sand | Self::Silt => 1,
            Self::PackedClay => 2,
            Self::WetMud | Self::FreshCut | Self::CutLip | Self::PlacedClod => 0,
        }
    }
}

pub fn liquid_regime(kind: SurfaceKind, water: f32) -> LiquidRegime {
    if water <= 0.004 {
        return LiquidRegime::Dry;
    }
    let slurry_at = match kind {
        SurfaceKind::WetMud | SurfaceKind::FreshCut | SurfaceKind::PackedClay => 0.045,
        SurfaceKind::Silt => 0.06,
        SurfaceKind::CutLip | SurfaceKind::PlacedClod | SurfaceKind::Berm => 0.11,
        // Sand and gravel drain. A film can sit for a moment. It does not bog.
        SurfaceKind::Sand | SurfaceKind::Gravel | SurfaceKind::DryCrust | SurfaceKind::Grass
        | SurfaceKind::Undisturbed => 10.0,
    };
    if water >= slurry_at {
        LiquidRegime::Slurry
    } else {
        LiquidRegime::Film
    }
}

/// Pore moisture plus standing water, for tire grip and peak slip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceMoisture {
    Dry = 0,
    Damp = 1,
    Saturated = 2,
    Pooled = 3,
}

impl SurfaceMoisture {
    pub fn label(self) -> &'static str {
        match self {
            Self::Dry => "dry",
            Self::Damp => "damp",
            Self::Saturated => "sat",
            Self::Pooled => "pool",
        }
    }

    pub const ALL: [SurfaceMoisture; 4] = [
        SurfaceMoisture::Dry,
        SurfaceMoisture::Damp,
        SurfaceMoisture::Saturated,
        SurfaceMoisture::Pooled,
    ];
}

/// Classify the contact surface from pore wetness and standing water.
pub fn surface_moisture(wetness: f32, water: f32, kind: SurfaceKind) -> SurfaceMoisture {
    let w = wetness.clamp(0.0, 1.0);
    let regime = liquid_regime(kind, water);
    if regime == LiquidRegime::Slurry {
        return SurfaceMoisture::Pooled;
    }
    if matches!(kind, SurfaceKind::Sand | SurfaceKind::Gravel) && water >= 0.018 {
        return SurfaceMoisture::Pooled;
    }
    if regime == LiquidRegime::Film {
        return SurfaceMoisture::Damp;
    }
    if w >= 0.72 {
        SurfaceMoisture::Saturated
    } else if w >= 0.28 {
        SurfaceMoisture::Damp
    } else {
        SurfaceMoisture::Dry
    }
}

/// Wetness, loose smear, and standing water for one moisture band.
pub fn moisture_state_fields(kind: SurfaceKind, state: SurfaceMoisture) -> (f32, f32, f32) {
    let loose = 0.01;
    match state {
        SurfaceMoisture::Dry => (0.08, 0.0, 0.0),
        SurfaceMoisture::Damp => {
            let w = match kind {
                SurfaceKind::Sand => 0.36,
                SurfaceKind::Silt => 0.46,
                _ => 0.40,
            };
            (w, loose, 0.012)
        }
        SurfaceMoisture::Saturated => {
            let w = match kind {
                SurfaceKind::Sand => 0.82,
                SurfaceKind::Silt => 0.90,
                SurfaceKind::WetMud => 0.96,
                _ => 0.88,
            };
            let film = if matches!(kind, SurfaceKind::WetMud) {
                0.048
            } else {
                0.0
            };
            (w, loose + 0.004, film)
        }
        SurfaceMoisture::Pooled => {
            let water = match kind {
                SurfaceKind::Sand | SurfaceKind::Gravel => 0.022,
                SurfaceKind::WetMud => 0.14,
                _ => 0.11,
            };
            (0.92, loose + 0.008, water)
        }
    }
}

/// Slip where the patch peaks, metres per metre. Wetter ground peaks earlier.
pub fn moisture_peak_slip(kind: SurfaceKind, state: SurfaceMoisture) -> f32 {
    let base = match kind {
        SurfaceKind::Sand => 0.20_f32,
        SurfaceKind::Silt => 0.16_f32,
        SurfaceKind::WetMud | SurfaceKind::DryCrust | SurfaceKind::FreshCut => 0.18_f32,
        _ => 0.18_f32,
    };
    let widen = match state {
        SurfaceMoisture::Dry => 0.0_f32,
        SurfaceMoisture::Damp => 0.03_f32,
        SurfaceMoisture::Saturated => 0.085_f32,
        SurfaceMoisture::Pooled => 0.145_f32,
    };
    (base + widen).clamp(0.12, 0.38)
}

/// Extra scale on `tan φ` from moisture, on top of the terramech curve.
pub fn moisture_mu_scale(kind: SurfaceKind, state: SurfaceMoisture) -> f32 {
    use SurfaceMoisture::*;
    let k = match kind {
        SurfaceKind::Sand => kind,
        SurfaceKind::Silt => kind,
        SurfaceKind::WetMud | SurfaceKind::DryCrust | SurfaceKind::FreshCut | SurfaceKind::PackedClay => {
            SurfaceKind::WetMud
        }
        _ => SurfaceKind::WetMud,
    };
    match (k, state) {
        (SurfaceKind::WetMud, Dry) => 1.10,
        (SurfaceKind::WetMud, Damp) => 0.96,
        (SurfaceKind::WetMud, Saturated) => 0.76,
        (SurfaceKind::WetMud, Pooled) => 0.50,
        (SurfaceKind::Sand, Dry) => 1.04,
        (SurfaceKind::Sand, Damp) => 1.10,
        (SurfaceKind::Sand, Saturated) => 0.70,
        (SurfaceKind::Sand, Pooled) => 0.55,
        (SurfaceKind::Silt, Dry) => 0.94,
        (SurfaceKind::Silt, Damp) => 0.80,
        (SurfaceKind::Silt, Saturated) => 0.56,
        (SurfaceKind::Silt, Pooled) => 0.40,
        _ => 1.0,
    }
}

/// Metres of water that stay in the pores and do not enter the pipe.
pub fn water_hold(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::Gravel => 0.0,
        SurfaceKind::CutLip => 0.001,
        SurfaceKind::DryCrust => 0.001,
        SurfaceKind::Sand => 0.002,
        SurfaceKind::Undisturbed => 0.003,
        SurfaceKind::PlacedClod => 0.003,
        SurfaceKind::Grass => 0.004,
        SurfaceKind::Berm => 0.004,
        SurfaceKind::PackedClay => 0.009,
        SurfaceKind::WetMud => 0.012,
        SurfaceKind::Silt => 0.014,
        SurfaceKind::FreshCut => 0.02,
    }
}

/// Multiplier on the shallow-water pipe. Gravel drains. Packed clay holds a film.
pub fn water_conduct(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::Gravel => 2.4,
        SurfaceKind::Sand => 1.85,
        SurfaceKind::CutLip => 1.7,
        SurfaceKind::Berm => 1.35,
        SurfaceKind::DryCrust => 1.25,
        SurfaceKind::PlacedClod => 1.15,
        SurfaceKind::Undisturbed => 1.0,
        SurfaceKind::Grass => 0.85,
        SurfaceKind::WetMud => 0.9,
        SurfaceKind::FreshCut => 0.5,
        SurfaceKind::Silt => 0.36,
        SurfaceKind::PackedClay => 0.32,
    }
}

/// Water fraction where clay, silt, and sand slurries are compared.
pub const SLURRY_COMPARE_MOISTURE: f32 = 0.55;

/// Clay, silt, and sand join standing water. Gravel and a dry crust do not.
pub fn mixes_as_slurry(kind: SurfaceKind) -> bool {
    matches!(
        kind,
        SurfaceKind::WetMud
            | SurfaceKind::FreshCut
            | SurfaceKind::PackedClay
            | SurfaceKind::Silt
            | SurfaceKind::Sand
    )
}

/// Metres of soil that join `water` metres so the slurry's water fraction
/// is `moisture`. The same moisture takes the same soil from clay, silt, and sand.
pub fn slurry_soil_depth(water: f32, moisture: f32) -> f32 {
    let water = water.max(0.0);
    if water <= 1.0e-6 {
        return 0.0;
    }
    let m = moisture.clamp(0.08, 0.92);
    water * (1.0 - m) / m
}

/// Dynamic viscosity (Pa·s) of one slurry. `moisture` is the water fraction,
/// 0 a stiff paste and 1 thin. Clay stays thick. Silt sits under it. Sand does not.
pub fn slurry_viscosity(kind: SurfaceKind, moisture: f32) -> f32 {
    let thin = moisture.clamp(0.0, 1.0).powf(1.35);
    let (stiff, fluid) = match kind {
        SurfaceKind::WetMud | SurfaceKind::FreshCut | SurfaceKind::PackedClay => (260.0, 22.0),
        SurfaceKind::Silt => (78.0, 6.0),
        SurfaceKind::Sand => (12.0, 0.45),
        _ => (1.5, 0.08),
    };
    lerp(stiff, fluid, thin)
}

/// Metres per second of slurry, per metre of free-surface head, across one cell.
/// Clay is the thick end. Sand runs. A berm is spoil, so it uses the clay paste.
pub fn slurry_flow_conduct(kind: SurfaceKind, moisture: f32) -> f32 {
    let kind = paste_kind(kind);
    let visc = slurry_viscosity(kind, moisture).max(0.05);
    (22.0 / visc).clamp(0.02, 3.0)
}

/// How fast a disturbed slurry free surface levels after a pass or a shed.
/// Clay settles slowest, then silt, then sand.
pub fn slurry_settle_pace(kind: SurfaceKind, moisture: f32) -> f32 {
    slurry_flow_conduct(kind, moisture)
}

/// Metres per second of berm soil that can join a neighbouring slurry.
/// The same viscosity that slows the pipe keeps the creep slow.
pub fn berm_creep_rate(kind: SurfaceKind, moisture: f32) -> f32 {
    let kind = paste_kind(kind);
    let visc = slurry_viscosity(kind, moisture).max(0.2);
    (1.35 / visc).clamp(0.0018, 0.012)
}

/// Metres per second of wet berm spoil that slumps under tire or body load.
/// Clay is slowest, then silt, then sand, at the same load and moisture.
pub fn wet_berm_creep_under_load_rate(kind: SurfaceKind, moisture: f32, load_n: f32) -> f32 {
    let kind = paste_kind(kind);
    let visc = slurry_viscosity(kind, moisture).max(0.05);
    let load = (load_n / 3000.0).clamp(0.0, 2.6);
    (1.35 / visc).clamp(0.0025, 0.058) * (1.0 + 5.2 * load)
}

fn paste_kind(kind: SurfaceKind) -> SurfaceKind {
    match kind {
        SurfaceKind::Berm | SurfaceKind::PlacedClod | SurfaceKind::CutLip => SurfaceKind::WetMud,
        other => other,
    }
}

/// Metres a sideways shove carries slurry. Clay stays next to the rut.
/// Silt runs farther at the same moisture, because it is the thinner mix.
/// A dry crust does not slurry, so the reach is zero.
pub fn slurry_slide_reach(kind: SurfaceKind, moisture: f32) -> f32 {
    if !mixes_as_slurry(kind) {
        return 0.0;
    }
    let visc = slurry_viscosity(paste_kind(kind), moisture).max(0.05);
    (36.0 / visc).clamp(0.18, 2.4)
}

/// Share of the slurry in one cell that a sideways shove moves.
/// Clay keeps almost all of it. Silt gives up more of the column.
pub fn slurry_leave_fraction(kind: SurfaceKind, moisture: f32) -> f32 {
    if !mixes_as_slurry(kind) {
        return 0.0;
    }
    let visc = slurry_viscosity(paste_kind(kind), moisture).max(0.05);
    let clay = slurry_viscosity(SurfaceKind::WetMud, moisture).max(0.05);
    let thin = (1.0 - visc / clay).clamp(0.0, 1.0);
    0.08 + thin * 0.70
}

/// Share of a fresh cut that stays in the rut as slurry instead of leaving
/// as berm or throw. Clay holds a larger share. Silt keeps less. A dry crust
/// returns zero: the cut is spoil, and the cell does not become slurry.
pub fn slurry_hold_fraction(kind: SurfaceKind, moisture: f32) -> f32 {
    if !mixes_as_slurry(kind) {
        return 0.0;
    }
    let visc = slurry_viscosity(paste_kind(kind), moisture).max(0.05);
    let clay = slurry_viscosity(SurfaceKind::WetMud, moisture).max(0.05);
    let thick = (visc / clay).clamp(0.0, 1.0);
    0.18 + thick * 0.28
}

/// Metres of upward travel that snap a slurry seal. The suction lives inside
/// this stretch. Past it the tire is free and the mud leaves with the break.
pub const SLURRY_BREAK_LIFT: f32 = 0.050;

/// Force and torque of deep slurry on one tire.
///
/// `force` is newtons along the tire forward axis and opposes `v_long`.
/// `torque` is newton-metres on the wheel and opposes `omega`. Either is zero
/// when the motion it resists is zero, so a shove and a spin are separate.
#[derive(Clone, Copy, Debug)]
pub struct SlurryDrag {
    pub force: f32,
    pub torque: f32,
    pub viscosity: f32,
}

/// Viscous drag of one slurry on a tire that is moving, spinning, or both.
///
/// `depth` is the slurry the tire is in, metres. Clay, silt, and sand use
/// [`slurry_viscosity`] at the same moisture, so the mix is what sets the
/// resistance. A deeper body resists more because the patch and the plowed
/// section are larger, not because the shear gap grows.
pub fn slurry_motion_drag(
    kind: SurfaceKind,
    moisture: f32,
    depth: f32,
    radius: f32,
    width: f32,
    v_long: f32,
    omega: f32,
) -> SlurryDrag {
    let viscosity = slurry_viscosity(paste_kind(kind), moisture);
    let depth = depth.max(0.0);
    if depth < 0.008 || !viscosity.is_finite() {
        return SlurryDrag {
            force: 0.0,
            torque: 0.0,
            viscosity,
        };
    }
    let radius = radius.max(0.05);
    let width = width.max(0.02);
    let sink = depth.min(radius * 0.92);
    let chord = contact_length(radius, sink).max(sink);
    let patch = chord * width;
    let plow = depth * width;
    // Fixed shear gap. Depth enters through the wetted patch and the section
    // the tire has to shove aside.
    let gap = 0.016_f32;
    let shear = viscosity / gap;
    let force_mag = (shear * (patch * 0.42 + plow * 1.15) * v_long.abs()).min(5_500.0);
    let torque_mag = (shear * patch * omega.abs() * radius * radius).min(850.0);
    SlurryDrag {
        force: oppose(v_long) * force_mag,
        torque: oppose(omega) * torque_mag,
        viscosity,
    }
}

/// Downward suction, newtons, while a tire is still sealed to the slurry.
/// Zero once `lift` reaches [`SLURRY_BREAK_LIFT`], and zero when the tire is
/// not actually rising. The pressure follows the same mix viscosity as the drag.
pub fn slurry_suction_force(
    kind: SurfaceKind,
    moisture: f32,
    depth: f32,
    radius: f32,
    width: f32,
    lift_speed: f32,
    lift: f32,
) -> f32 {
    if lift >= SLURRY_BREAK_LIFT - 1.0e-6 || depth < 0.025 || lift_speed <= 1.0e-4 {
        return 0.0;
    }
    let viscosity = slurry_viscosity(paste_kind(kind), moisture).max(0.05);
    let radius = radius.max(0.05);
    let width = width.max(0.02);
    let sink = depth.min(radius * 0.92);
    let chord = contact_length(radius, sink).max(0.05);
    let area = chord * width;
    let stretch = (lift / SLURRY_BREAK_LIFT).clamp(0.0, 1.0);
    let seal = (1.0 - stretch).powf(0.55);
    let gap = 0.005 + lift.max(0.0);
    let pressure = viscosity * (lift_speed / gap + 22.0);
    (pressure * area * seal).clamp(0.0, 7_000.0)
}

/// Seal under one tire. `lift` is how far it has already risen through the slurry.
#[derive(Clone, Copy, Debug)]
pub struct SlurrySeal {
    pub lift: f32,
    pub impulse: f32,
    pub broken: bool,
}

impl SlurrySeal {
    pub fn seated() -> Self {
        Self {
            lift: 0.0,
            impulse: 0.0,
            broken: false,
        }
    }
}

impl Default for SlurrySeal {
    fn default() -> Self {
        Self::seated()
    }
}

/// One pull step. `force` is the downward suction. `thrown` is slurry volume
/// that left the column on the breakaway, now carried in `airborne`.
#[derive(Clone, Copy, Debug, Default)]
pub struct PullOut {
    pub force: f32,
    pub impulse: f32,
    pub broken: bool,
    pub thrown: f32,
}

fn oppose(v: f32) -> f32 {
    if v > 1.0e-4 {
        -1.0
    } else if v < -1.0e-4 {
        1.0
    } else {
        0.0
    }
}

/// Stadium centerline. South cap sits on `TRACK_ZS`, north cap on `TRACK_ZN`.
/// The scripted buggy and the heightfield share these so the lap stays on the mud.
pub const TRACK_R: f32 = 6.35;
pub const TRACK_ZS: f32 = -0.90;
pub const TRACK_ZN: f32 = 3.40;

#[derive(Clone, Copy, Debug)]
pub struct SoilSample {
    pub height: f32,
    pub wetness: f32,
    pub water: f32,
    pub loose: f32,
    pub darken: f32,
    pub roughness: f32,
    pub softness: f32,
    /// Metres carved out of the original bed. The berm's pile is not a rut.
    pub rut: f32,
    pub kind: u8,
    /// Spoil thickness on this cell. Separate from the cut.
    pub placed: f32,
    /// Mixed water and soil, metres. Zero where the column has not joined.
    pub slurry: f32,
    /// Dry skin thickness, metres. Not a height and not a volume.
    pub crust: f32,
    /// Shrinkage opening, metres. Not a cut.
    pub crack: f32,
    /// Lane visits already counted on this cell.
    pub passes: u8,
    /// Seconds since a tire last claimed this cell.
    pub rest: f32,
    /// Mud, sand, gravel, grass. Sums to 1 on those materials, including on
    /// a joint, and to 0 on any other kind.
    pub mix: [f32; 4],
}

impl SoilSample {
    pub fn kind(self) -> SurfaceKind {
        SurfaceKind::from_u8(self.kind)
    }
}

#[derive(Clone, Debug)]
pub struct SoilField {
    pub res: usize,
    pub size: f32,
    pub cell: f32,
    pub height: Vec<f32>,
    pub initial_height: Vec<f32>,
    pub loose: Vec<f32>,
    pub wetness: Vec<f32>,
    pub base_wetness: Vec<f32>,
    pub water: Vec<f32>,
    /// 0 = untouched albedo, 1 = compacted dark track.
    pub darken: Vec<f32>,
    /// Linear roughness, 0 smooth / wet, 1 dry and broken.
    pub roughness: Vec<f32>,
    /// Carved surface. Cutting lowers this and does not touch `placed`.
    pub bed: Vec<f32>,
    /// Soil put down after it was cut: berm, crest clods, impact ring.
    pub placed: Vec<f32>,
    pub kind: Vec<u8>,
    /// Seconds since a lane tire last claimed this cell. A gap starts a new pass.
    pub rest: Vec<f32>,
    /// Distinct lane visits that have cut this cell. Pass 1 is the fresh bite.
    pub passes: Vec<u8>,
    /// Metres of lug pressed into this cell. Zero means no tread negative.
    /// The pattern is drawn from this field; it does not move the bed.
    pub tread_amp: Vec<f32>,
    /// Heading of the tire that pressed the cell, atan2(forward.x, forward.z).
    pub tread_ang: Vec<f32>,
    /// Lateral offset of the cell center from that tire's track, metres.
    pub tread_lat: Vec<f32>,
    /// Cubic metres that left a contact and have not been placed yet.
    /// A snapshot does not carry this. Land it before saving.
    pub airborne: f32,
    /// One puddle stamp uses a tread-width falloff instead of the flat lane.
    /// Not stored in a snapshot. The lane stamp stays a step for every other caller.
    pub tread_profile: bool,
    /// One slurry thickness, metres. Soil that left the bed plus the water
    /// that left the column. The mesh adds this to the solid. It is not a
    /// second surface.
    pub slurry: Vec<f32>,
    /// Soil held in the water column, metres. Scour moves it off the bed.
    /// It is not height and it is not a second water. The closed soil sum is
    /// bed plus placed plus this cloud.
    pub cloud: Vec<f32>,
    /// Crust thickness, metres. A shader reads this. It is not height.
    pub crust: Vec<f32>,
    /// Crack opening, metres. A shader reads this. It does not move the bed.
    pub crack: Vec<f32>,
    /// Cubic metres of skin written onto the field, including what later left.
    /// Not solid mass. The bed, the placed pile, and the cloud stay out of it.
    skin_grown_m3: f64,
    /// Cubic metres of skin soaked off or faded by the drying front.
    skin_lost_m3: f64,
    /// Mud, sand, gravel, grass per cell, `MIX_CHANNELS` floats.
    /// Filled by [`SoilField::blend_material_joints`].
    pub mix: Vec<f32>,
    /// `mix` is the blended joint field, not a one-hot guess from `kind`.
    pub mix_ready: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Terramech {
    pub k_eq: f32,
    pub n: f32,
    pub cohesion: f32,
    pub phi: f32,
    pub shear_k: f32,
    pub yield_stress: f32,
    pub viscosity: f32,
}

/// Grip, sink and shear for one surface state.
///
/// Wet mud with no standing water is the old softness curve, so a lap on the
/// lane does not change character. Every other kind has its own modulus,
/// friction and shear length. Film slicks the surface. Slurry only forms on
/// soils that hold water.
pub fn terramech_for(kind: SurfaceKind, softness: f32, water: f32) -> Terramech {
    // Start from the wetness curve the lap already drives, then move each
    // kind. Exponents stay near that curve: a high n looks stiff on paper
    // and collapses under a light tire.
    let base = terramech(softness);
    let mut tm = match kind {
        SurfaceKind::WetMud => {
            // Wet clay sticks and sits deeper once the pore is saturated.
            let mut tm = base;
            let bog = ((softness - 0.45) / 0.48).clamp(0.0, 1.0);
            tm.k_eq *= lerp(1.0, 0.34, bog.powf(1.04));
            tm.cohesion *= lerp(1.0, 1.14, bog);
            tm
        }
        SurfaceKind::Undisturbed => scale_mech(base, 1.12, 1.05, 0.52, 0.040, 0.9, 0.85),
        SurfaceKind::Grass => scale_mech(base, 1.28, 1.65, 0.46, 0.055, 1.05, 0.7),
        // Loose stone. High friction, almost no cohesion, and a modulus that
        // keeps a loaded tire near the surface.
        SurfaceKind::Gravel => scale_mech(base, 3.8, 0.22, 0.74, 0.016, 0.3, 0.18),
        SurfaceKind::DryCrust => {
            // Baked clay. Stiffer than damp sand, and it grips until the skin breaks.
            // The friction curve stays the one the lap already drives.
            let mut tm = scale_mech(base, 1.48, 1.35, 0.60, 0.012, 0.5, 0.32);
            tm.phi = lerp(0.66, 0.46, softness);
            tm
        }
        SurfaceKind::Sand => {
            // Dry sand is loose and frictional. Damp sand picks up a little
            // capillary stick, then saturated sand weakens again. It never
            // reaches a clay crust. Grip sits under dry clay and over wet clay.
            let mut tm = scale_mech(base, 1.05, 0.32, 0.64, 0.024, 0.32, 0.22);
            let capillary = (1.0 - ((softness - 0.34) / 0.34).abs()).clamp(0.0, 1.0);
            tm.phi = lerp(0.64, 0.32, softness);
            tm.cohesion *= lerp(0.55, 1.6, capillary);
            tm.k_eq *= lerp(1.08, 0.82, softness);
            tm
        }
        SurfaceKind::Silt => {
            // Dry silt is dusty. Wet silt is slick, sticky, and it sinks.
            let mut tm = scale_mech(base, 0.70, 0.90, 0.28, 0.020, 0.85, 1.05);
            tm.phi = lerp(0.50, 0.12, softness);
            tm.cohesion *= lerp(0.45, 1.35, softness);
            tm.k_eq *= lerp(1.15, 0.75, softness);
            tm
        }
        SurfaceKind::FreshCut => scale_mech(base, 0.78, 0.9, 0.17, 0.014, 1.2, 1.3),
        SurfaceKind::CutLip => scale_mech(base, 1.08, 0.62, 0.40, 0.022, 0.65, 0.55),
        SurfaceKind::Berm => scale_mech(base, 0.58, 0.38, 0.50, 0.070, 0.5, 0.65),
        SurfaceKind::PlacedClod => scale_mech(base, 0.84, 0.48, 0.33, 0.034, 0.72, 0.75),
        SurfaceKind::PackedClay => scale_mech(base, 1.22, 1.4, 0.24, 0.009, 1.1, 1.2),
    };
    match liquid_regime(kind, water) {
        LiquidRegime::Dry => {}
        LiquidRegime::Film => {
            tm.phi *= 0.9;
            tm.cohesion *= 1.06;
            tm.viscosity *= 1.12;
        }
        LiquidRegime::Slurry => {
            tm.k_eq *= 0.7;
            tm.phi *= 0.62;
            tm.shear_k *= 0.8;
            tm.viscosity *= 1.45;
            tm.yield_stress *= 1.18;
        }
    }
    tm
}

fn scale_mech(
    mut tm: Terramech,
    k_mul: f32,
    c_mul: f32,
    phi: f32,
    shear_k: f32,
    y_mul: f32,
    v_mul: f32,
) -> Terramech {
    tm.k_eq *= k_mul;
    tm.cohesion *= c_mul;
    tm.phi = phi;
    tm.shear_k = shear_k;
    tm.yield_stress *= y_mul;
    tm.viscosity *= v_mul;
    tm
}

/// Soil constants blended by `softness` in 0 (firm damp ground) ..= 1 (deep mud).
pub fn terramech(softness: f32) -> Terramech {
    let s = softness.clamp(0.0, 1.0);
    // Yield rises faster than cohesion, so extra sinkage eventually bogs the tire.
    let s_yield = s.powf(1.35);
    Terramech {
        k_eq: lerp(320_000.0, 48_000.0, s),
        n: lerp(0.95, 0.72, s),
        cohesion: lerp(1_400.0, 5_200.0, s),
        phi: lerp(0.58, 0.20, s),
        shear_k: lerp(0.045, 0.018, s),
        yield_stress: lerp(140.0, 3_600.0, s_yield),
        viscosity: lerp(18.0, 150.0, s),
    }
}

/// How willingly mud cakes onto a tire. Wet clay is the old curve (scale 1).
/// A crust, sand, and gravel barely stick. Silt sticks only while it is wet.
pub fn stick_scale(kind: SurfaceKind, wetness: f32) -> f32 {
    let w = wetness.clamp(0.0, 1.0);
    match kind {
        SurfaceKind::WetMud
        | SurfaceKind::FreshCut
        | SurfaceKind::PackedClay
        | SurfaceKind::CutLip
        | SurfaceKind::Berm
        | SurfaceKind::PlacedClod => 1.0,
        SurfaceKind::Silt => lerp(0.28, 1.3, w),
        SurfaceKind::Sand => lerp(0.08, 0.32, w),
        SurfaceKind::DryCrust => lerp(0.12, 0.38, w),
        SurfaceKind::Gravel => 0.06,
        SurfaceKind::Grass => 0.25,
        SurfaceKind::Undisturbed => lerp(0.4, 0.75, w),
    }
}

/// Resting pore state for one named surface. Grip and sink are read here,
/// under the same wheel, so the five soils are not compared at a shared wetness.
pub fn surface_rest(kind: SurfaceKind) -> (f32, f32, f32) {
    match kind {
        SurfaceKind::Gravel => (0.05, 0.0, 0.0),
        SurfaceKind::DryCrust => (0.10, 0.0, 0.0),
        SurfaceKind::WetMud => (0.88, 0.02, 0.0),
        SurfaceKind::Sand => (0.30, 0.008, 0.0),
        SurfaceKind::Silt => (0.84, 0.012, 0.0),
        other => {
            let _ = other;
            (0.40, 0.0, 0.0)
        }
    }
}

/// Wheel load used when reporting sink for one surface, newtons.
pub const SURFACE_WHEEL_LOAD: f32 = 1_600.0;

/// Upward soil reaction and horizontal shear for a light pass, newtons.
/// The bed of that pass is [`rut_depth_from_force`] of this pair.
pub const LIGHT_PASS_NORMAL: f32 = 650.0;
pub const LIGHT_PASS_PUSH: f32 = 150.0;

/// A harder push: more weight on the patch and more shear into the soil.
pub const HARD_PASS_NORMAL: f32 = 5_400.0;
pub const HARD_PASS_PUSH: f32 = 4_200.0;

/// Grip (`tan φ`) and sink (m) of a rested surface under [`SURFACE_WHEEL_LOAD`].
pub fn surface_grip_sink(kind: SurfaceKind) -> (f32, f32) {
    let (wetness, loose, water) = surface_rest(kind);
    let grip = grip_mu(kind, wetness, loose, water);
    let sink = sinkage_on(
        SURFACE_WHEEL_LOAD,
        kind,
        wetness,
        loose,
        water,
        0.38,
        0.28,
    );
    (grip, sink)
}

/// Friction coefficient the tire uses, `tan φ`, at this pore moisture.
pub fn grip_mu(kind: SurfaceKind, wetness: f32, loose: f32, water: f32) -> f32 {
    let tm = terramech_for(kind, softness_of(wetness, loose, water), water);
    tm.phi.tan()
}

/// Cohesion (Pa). Wet clay sticks. A dry crust shears instead of smearing.
pub fn stickiness(kind: SurfaceKind, wetness: f32, loose: f32, water: f32) -> f32 {
    terramech_for(kind, softness_of(wetness, loose, water), water).cohesion
}

/// Sinkage (m) under `load` for this kind and pore moisture.
pub fn sinkage_on(
    load: f32,
    kind: SurfaceKind,
    wetness: f32,
    loose: f32,
    water: f32,
    radius: f32,
    width: f32,
) -> f32 {
    let tm = terramech_for(kind, softness_of(wetness, loose, water), water);
    sinkage_with(load, radius, width, &tm)
}

fn sinkage_with(load: f32, radius: f32, width: f32, tm: &Terramech) -> f32 {
    let mut lo = 0.0;
    let mut hi = radius * 0.9;
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        let w = bekker_load(mid, radius, width, tm.k_eq, tm.n);
        if w < load {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Sinkage on a cell that may have baked. A fresh surface uses the kind alone.
pub fn sinkage_on_track(
    load: f32,
    kind: SurfaceKind,
    wetness: f32,
    loose: f32,
    water: f32,
    radius: f32,
    width: f32,
    crust: f32,
    crack: f32,
    passes: u8,
    rest: f32,
) -> f32 {
    let mut tm = terramech_for(kind, softness_of(wetness, loose, water), water);
    tm.k_eq *= track_harden_mul(crust, crack, passes, rest);
    sinkage_with(load, radius, width, &tm)
}

/// Tire constants at a sample. An old crusted track is stiffer than the kind alone.
pub fn mech_for_sample(sample: &SoilSample) -> Terramech {
    let mut tm = terramech_for_mix(sample.kind(), sample.mix, sample.softness, sample.water);
    tm.k_eq *= track_harden_mul(sample.crust, sample.crack, sample.passes, sample.rest);
    tm
}

/// Terramech of one cell. A pure material uses [`terramech_for`] on `kind`.
/// A joint weights mud, sand, gravel, and grass by [`SoilSample::mix`].
pub fn terramech_for_mix(
    kind: SurfaceKind,
    mix: [f32; 4],
    softness: f32,
    water: f32,
) -> Terramech {
    let mut sum = 0.0_f32;
    let mut peak = 0.0_f32;
    for share in mix {
        sum += share;
        peak = peak.max(share);
    }
    if !(sum >= 0.05 && peak < 0.999) {
        return terramech_for(kind, softness, water);
    }
    let kinds = [
        SurfaceKind::WetMud,
        SurfaceKind::Sand,
        SurfaceKind::Gravel,
        SurfaceKind::Grass,
    ];
    let mut acc = Terramech {
        k_eq: 0.0,
        n: 0.0,
        cohesion: 0.0,
        phi: 0.0,
        shear_k: 0.0,
        yield_stress: 0.0,
        viscosity: 0.0,
    };
    for (i, material) in kinds.into_iter().enumerate() {
        let w = mix[i];
        if w <= 0.0 {
            continue;
        }
        let tm = terramech_for(material, softness, water);
        acc.k_eq += tm.k_eq * w;
        acc.n += tm.n * w;
        acc.cohesion += tm.cohesion * w;
        acc.phi += tm.phi * w;
        acc.shear_k += tm.shear_k * w;
        acc.yield_stress += tm.yield_stress * w;
        acc.viscosity += tm.viscosity * w;
    }
    let inv = 1.0 / sum;
    acc.k_eq *= inv;
    acc.n *= inv;
    acc.cohesion *= inv;
    acc.phi *= inv;
    acc.shear_k *= inv;
    acc.yield_stress *= inv;
    acc.viscosity *= inv;
    acc
}

/// Plastic rut depth (m) for the force on one tire.
///
/// `normal` is the soil's upward reaction. `push` is the horizontal force
/// the tire puts into the soil, the shear, not the net force left on the
/// chassis after drag cancels. The bed is the Bekker sinkage of that
/// resultant, so a harder push cuts deeper and moves more soil onto the berm.
pub fn rut_depth_from_force(
    normal: f32,
    push: f32,
    radius: f32,
    width: f32,
    kind: SurfaceKind,
    wetness: f32,
    loose: f32,
    water: f32,
) -> f32 {
    let n = normal.max(0.0);
    let p = push.max(0.0);
    let load = (n * n + p * p).sqrt();
    if load < 1.0 || !load.is_finite() {
        return 0.0;
    }
    let radius = radius.max(0.05);
    let depth = sinkage_on(
        load,
        kind,
        wetness,
        loose,
        water,
        radius,
        width.max(0.02),
    );
    depth.min(radius * 0.85)
}

/// Plastic rut depth on one sampled cell. An old crusted track cuts less
/// than the same force on wet clay. A cell with no crust matches
/// [`rut_depth_from_force`].
pub fn rut_depth_on_sample(
    normal: f32,
    push: f32,
    radius: f32,
    width: f32,
    sample: &SoilSample,
) -> f32 {
    let n = normal.max(0.0);
    let p = push.max(0.0);
    let load = (n * n + p * p).sqrt();
    if load < 1.0 || !load.is_finite() {
        return 0.0;
    }
    let radius = radius.max(0.05);
    sinkage_on_track(
        load,
        sample.kind(),
        sample.wetness,
        sample.loose,
        sample.water,
        radius,
        width.max(0.02),
        sample.crust,
        sample.crack,
        sample.passes,
        sample.rest,
    )
    .min(radius * 0.85)
}

/// Wetness alone makes the ground heavy. Standing water and loose spoil
/// are what push it into a true bog.
pub fn softness_of(wetness: f32, loose: f32, water: f32) -> f32 {
    let loose_n = (loose / 0.09).clamp(0.0, 1.0);
    let depth = (water / 0.18).clamp(0.0, 1.0);
    (wetness.clamp(0.0, 1.0) * 0.48 + depth * 0.34 + loose_n * 0.22).clamp(0.0, 1.0)
}

/// Upward force from the water the tire is sitting in. A film barely lifts.
/// Deeper slurry displaces more, so the wheel floats higher.
pub fn wheel_buoyancy(depth: f32, radius: f32, width: f32) -> f32 {
    let radius = radius.max(1.0e-3);
    let d = depth.clamp(0.0, radius * 1.85);
    if d <= 1.0e-4 || width <= 0.0 {
        return 0.0;
    }
    // Circular segment of the tire cross-section, times the tread width.
    let c = ((radius - d) / radius).clamp(-1.0, 1.0);
    let area = radius * radius * c.acos() - (radius - d) * (2.0 * radius * d - d * d).max(0.0).sqrt();
    // Muddy water, not clear. Capped so a flooded cell cannot throw the chassis.
    (area.max(0.0) * width * 1_150.0 * 9.81).min(4_500.0)
}

/// Fresh water. The crossing drag and buoyancy use this density.
pub const WATER_RHO: f32 = 1000.0;

/// Bow crest above the still column, metres.
/// `h = k * speed * depth`, so a faster bow in deeper water stands higher.
/// Zero when the hull is barely moving or the column is a film.
pub fn water_bow_height(speed: f32, depth: f32) -> f32 {
    let v = speed.max(0.0);
    let d = depth.max(0.0);
    if v < 0.05 || d < 0.02 {
        return 0.0;
    }
    (0.20 * v * d).min(0.24).min(d * 0.85)
}

/// Horizontal drag on the hull, newtons. Quadratic in speed, linear in the
/// depth the nose is pushing. Zero in a film or at rest.
pub fn water_body_drag(speed: f32, depth: f32) -> f32 {
    let v = speed.abs();
    let d = depth.max(0.0);
    if v < 0.04 || d < 0.02 {
        return 0.0;
    }
    let width = 1.52_f32;
    let area = width * d.min(0.90);
    let cd = 0.55_f32;
    (0.5 * WATER_RHO * cd * area * v * v).min(12_000.0)
}

/// Upward force on the hull from the volume under the free surface, newtons.
/// The wheels already carry their own buoyancy. `draft` is how far the
/// surface stands above the belly.
pub fn water_body_buoyancy(draft: f32) -> f32 {
    let d = draft.max(0.0);
    if d < 0.004 {
        return 0.0;
    }
    let plan = 1.48_f32 * 2.55;
    let fill = 0.34_f32;
    let volume = plan * d.min(0.70) * fill;
    (WATER_RHO * 9.81 * volume).min(8_000.0)
}

/// Metres of bed a 1 m/s current through 0.30 m of water attacks per second.
/// Clay is slower than silt. Sand is faster.
pub fn water_scour_pace(kind: SurfaceKind) -> f32 {
    match water_grain(kind) {
        Some(SurfaceKind::WetMud) => 0.032,
        Some(SurfaceKind::Silt) => 0.080,
        Some(SurfaceKind::Sand) => 0.160,
        _ => 0.0,
    }
}

/// Share of attacked soil that re-sticks to the bed instead of entering the cloud.
/// Clay holds. Sand does not.
pub fn water_stick(kind: SurfaceKind) -> f32 {
    match water_grain(kind) {
        Some(SurfaceKind::WetMud) => 0.74,
        Some(SurfaceKind::Silt) => 0.46,
        Some(SurfaceKind::Sand) => 0.14,
        _ => 0.0,
    }
}

/// Metres per second that stay suspended at the reference current.
/// Clay clouds slower than silt. Sand clouds faster because it scours faster
/// and less of it sticks.
pub fn water_cloud_pace(kind: SurfaceKind) -> f32 {
    water_scour_pace(kind) * (1.0 - water_stick(kind))
}

fn water_grain(kind: SurfaceKind) -> Option<SurfaceKind> {
    match kind {
        SurfaceKind::WetMud
        | SurfaceKind::PackedClay
        | SurfaceKind::FreshCut
        | SurfaceKind::Berm
        | SurfaceKind::PlacedClod
        | SurfaceKind::CutLip => Some(SurfaceKind::WetMud),
        SurfaceKind::Silt => Some(SurfaceKind::Silt),
        SurfaceKind::Sand => Some(SurfaceKind::Sand),
        _ => None,
    }
}

/// Longitudinal drag. Dry ground keeps the old 2.5% of the load. Standing
/// water adds drag in proportion to depth, so a film and a slurry do not roll
/// the same.
pub fn rolling_resistance(normal: f32, depth: f32) -> f32 {
    let n = normal.max(0.0);
    n * (0.025 + 0.70 * depth.max(0.0))
}

pub fn contact_length(radius: f32, sinkage: f32) -> f32 {
    let z = sinkage.clamp(0.0, radius * 0.98);
    2.0 * (2.0 * radius * z - z * z).max(0.0).sqrt()
}

/// Rigid-wheel Bekker load (N) for a uniform soil of modulus `k_eq`.
///
/// Pressure at local sinkage y is `k_eq * y^n`, integrated over the chord
/// of a rigid cylinder of `radius` and `width`.
pub fn bekker_load(sinkage: f32, radius: f32, width: f32, k_eq: f32, n: f32) -> f32 {
    if sinkage <= 1.0e-5 || k_eq <= 0.0 || width <= 0.0 {
        return 0.0;
    }
    let z = sinkage.min(radius * 0.95);
    let half = (2.0 * radius * z - z * z).max(0.0).sqrt();
    if half < 1.0e-5 {
        return 0.0;
    }
    let samples = 8;
    let mut acc = 0.0;
    for i in 0..samples {
        let t = (i as f32 + 0.5) / samples as f32;
        let x = -half + t * (2.0 * half);
        let y = (radius * radius - x * x).max(0.0).sqrt() - (radius - z);
        let y = y.max(0.0);
        acc += k_eq * y.powf(n);
    }
    let avg_p = acc / samples as f32;
    avg_p * (2.0 * half) * width
}

/// Janosi-Hanamoto shear stress. `j` is shear displacement (m), `k` is the
/// shear modulus (m). Result is capped by Mohr-Coulomb `c + σ tan φ`.
pub fn janosi_hanamoto(cohesion: f32, sigma: f32, phi: f32, j: f32, k: f32) -> f32 {
    let tau_max = (cohesion + sigma * phi.tan()).max(0.0);
    if j.abs() < 1.0e-6 || k < 1.0e-5 {
        return 0.0;
    }
    tau_max * (1.0 - (-j.abs() / k).exp())
}

impl SoilField {
    pub fn demo() -> Self {
        Self::generate(true)
    }

    pub fn flat(wetness: f32) -> Self {
        let mut field = Self::allocate();
        let w = wetness.clamp(0.0, 1.0);
        // High wetness stands in as a flooded pad so the comparison is mud, not damp dirt.
        let water = if w > 0.6 {
            ((w - 0.45) * 0.32).min(0.18)
        } else {
            0.0
        };
        let kind = if w > 0.6 {
            SurfaceKind::WetMud
        } else if w < 0.28 {
            SurfaceKind::DryCrust
        } else {
            SurfaceKind::Undisturbed
        };
        for i in 0..field.height.len() {
            field.height[i] = 0.0;
            field.bed[i] = 0.0;
            field.placed[i] = 0.0;
            field.initial_height[i] = 0.0;
            field.base_wetness[i] = w;
            field.wetness[i] = w;
            field.water[i] = water;
            field.loose[i] = if w > 0.6 { 0.02 } else { 0.0 };
            field.darken[i] = 0.06;
            field.roughness[i] = lerp(0.9, 0.45, w);
            field.kind[i] = kind as u8;
        }
        field
    }

    fn allocate() -> Self {
        let res = MAP_RES;
        let n = res * res;
        let size = MAP_SIZE;
        Self {
            res,
            size,
            cell: size / res as f32,
            height: vec![0.0; n],
            initial_height: vec![0.0; n],
            loose: vec![0.0; n],
            wetness: vec![0.0; n],
            base_wetness: vec![0.0; n],
            water: vec![0.0; n],
            darken: vec![0.0; n],
            roughness: vec![0.0; n],
            bed: vec![0.0; n],
            placed: vec![0.0; n],
            kind: vec![SurfaceKind::Undisturbed as u8; n],
            // The first visit has to count. Rest starts already past the gap.
            rest: vec![PASS_GAP + 1.0; n],
            passes: vec![0; n],
            tread_amp: vec![0.0; n],
            tread_ang: vec![0.0; n],
            tread_lat: vec![0.0; n],
            airborne: 0.0,
            tread_profile: false,
            slurry: vec![0.0; n],
            cloud: vec![0.0; n],
            crust: vec![0.0; n],
            crack: vec![0.0; n],
            skin_grown_m3: 0.0,
            skin_lost_m3: 0.0,
            mix: vec![0.0; n * MIX_CHANNELS],
            mix_ready: false,
        }
    }

    fn generate(with_puddle: bool) -> Self {
        let mut field = Self::allocate();
        let res = field.res;
        let cell = field.cell;
        let half = field.size * 0.5;
        // Shallow bowls on the racing line. The deep bog sits in the infield,
        // on the straight dig line, so a lap can splash without parking in it.
        // (cx, cz, across, along, depth). Along is the short axis so the lap punches through.
        // Centers sit on the line the scripted lap actually drives.
        let puddles = [
            (-6.15, 5.05_f32, 2.35_f32, 2.55_f32, 0.40_f32),
            (TRACK_R, 2.4, 2.3, 1.8, 0.38),
            (1.15, 9.25, 2.15, 2.05, 0.40),
        ];
        let crests = [
            (-TRACK_R, -0.55_f32, 2.05_f32, 0.38_f32),
            (5.15, -5.15, 2.05, 0.34),
            (-3.1, TRACK_ZN + 4.85, 2.15, 0.44),
        ];

        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let off = track_offset(x, z);
                let n = fbm2(x * 0.11, z * 0.11);
                let mut h = 0.10 + (n - 0.5) * 0.07;
                h += (fbm2(x * 0.37 + 3.0, z * 0.37) - 0.5) * 0.025;

                if with_puddle {
                    // Worn ribbon. Kept above the water line so the lane is mud, not a lake.
                    let lane = smooth01(1.55, 0.2, off.abs());
                    h -= lane * 0.045;
                    let berm = smooth01(0.7, 1.55, off) * smooth01(3.1, 1.7, off);
                    let gate = if z < TRACK_ZS + 0.5 {
                        smooth01(1.1, 3.4, x.abs())
                    } else {
                        1.0
                    };
                    h += berm * gate * (0.78 + (n - 0.5) * 0.1);
                    for (cx, cz, rad, amp) in crests {
                        h += gauss(x, z, cx, cz, rad) * amp;
                    }
                    // Ski jump on the west straight. The lip faces the west puddle.
                    h += launch_kicker(x, z);
                    h = h.max(0.035);
                    // Infield pond. Flat floor, clear of the racing line, on the dig ray.
                    h -= bog_bowl(x, z, 0.0, 1.05, 4.2, 0.52);
                    // North lip of the pond. The straight dig beaches here; the lap stays outside.
                    if x.abs() < 2.7 {
                        let lip = gauss(x, z, 0.0, 2.55, 1.05);
                        h += lip * 0.62 * smooth01(2.7, 1.4, x.abs());
                    }
                    for (cx, cz, rx, rz, depth) in puddles {
                        h -= bowl(x, z, cx, cz, rx, rz, depth);
                    }
                }
                let idx = j * res + i;
                field.height[idx] = h;
                field.bed[idx] = h;
                field.placed[idx] = 0.0;
                field.initial_height[idx] = h;
            }
        }

        let water_line = if with_puddle { -0.02 } else { -10.0 };
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let off = track_offset(x, z);
                let idx = j * res + i;
                let h = field.height[idx];
                let mut wet = 0.18 + 0.08 * fbm2(x * 0.17, z * 0.17);
                if with_puddle {
                    let lane = smooth01(1.6, 0.25, off.abs());
                    wet = lerp(wet, 0.96, lane);
                    let berm = smooth01(0.85, 1.6, off) * smooth01(3.0, 1.75, off);
                    wet = lerp(wet, 0.2, berm);
                    for (cx, cz, rx, rz, _) in puddles {
                        let d = (((x - cx) / rx).powi(2) + ((z - cz) / rz).powi(2)).sqrt();
                        wet = wet.max(smooth01(1.35, 0.45, d));
                    }
                    let bog = ((x).powi(2) + (z - 1.05).powi(2)).sqrt() / 4.2;
                    wet = wet.max(smooth01(1.2, 0.45, bog));
                }
                let mut water = 0.0;
                if h < water_line {
                    water = (water_line - h).min(0.28);
                    wet = 1.0;
                }
                // Firm run-up so a rolling lap still has speed at the lip.
                // The puddle beyond the lip stays soft.
                if launch_runup(x, z) && water < 0.01 {
                    wet = wet.min(0.22);
                }
                field.water[idx] = water;
                field.wetness[idx] = wet.clamp(0.0, 1.0);
                field.base_wetness[idx] = if water > 0.0 { 0.95 } else { wet.min(0.75) };
                let on_lane = with_puddle && off.abs() < 1.35;
                field.loose[idx] = if water > 0.02 {
                    0.02
                } else if launch_runup(x, z) {
                    0.0
                } else if on_lane {
                    0.045
                } else if wet > 0.75 {
                    0.012
                } else {
                    0.0
                };
                field.darken[idx] = 0.04 + 0.06 * fbm2(x * 0.33 + 2.0, z * 0.33);
                field.roughness[idx] =
                    lerp(0.93, 0.38, wet) * lerp(0.85, 1.1, fbm2(x * 0.5, z * 0.4));
                field.roughness[idx] = field.roughness[idx].clamp(0.18, 1.0);
                field.kind[idx] = born_kind(x, z, field.wetness[idx], water) as u8;
            }
        }
        field.blend_material_joints();
        field
    }

    /// Shares of mud, sand, gravel, and grass, blended across the joints.
    ///
    /// Each cell starts as one material. Four Jacobi passes then move
    /// [`JOINT_RATE`] of the difference across every edge whose two cells
    /// both carry one of those four and do not already match. The sum of
    /// each channel over the map is unchanged. The sum of the four channels
    /// on a cell is unchanged. Pore moisture and the authored base move on
    /// the same edges, so the wetness transition does not mint or lose water.
    pub fn blend_material_joints(&mut self) {
        let n = self.res * self.res;
        let res = self.res;
        if self.mix.len() != n * MIX_CHANNELS {
            self.mix.resize(n * MIX_CHANNELS, 0.0);
        }
        for idx in 0..n {
            let base = idx * MIX_CHANNELS;
            for channel in 0..MIX_CHANNELS {
                self.mix[base + channel] = 0.0;
            }
            if let Some(channel) = mix_channel(SurfaceKind::from_u8(self.kind[idx])) {
                self.mix[base + channel] = 1.0;
            }
        }
        let mut wet = self.wetness.clone();
        let mut base_wet = self.base_wetness.clone();
        for _ in 0..JOINT_PASSES {
            let prev_mix = self.mix.clone();
            let prev_wet = wet.clone();
            let prev_base = base_wet.clone();
            for j in 0..res {
                for i in 0..res - 1 {
                    let a = j * res + i;
                    joint_exchange(
                        &mut self.mix,
                        &prev_mix,
                        &mut wet,
                        &prev_wet,
                        &mut base_wet,
                        &prev_base,
                        a,
                        a + 1,
                    );
                }
            }
            for j in 0..res - 1 {
                for i in 0..res {
                    let a = j * res + i;
                    joint_exchange(
                        &mut self.mix,
                        &prev_mix,
                        &mut wet,
                        &prev_wet,
                        &mut base_wet,
                        &prev_base,
                        a,
                        a + res,
                    );
                }
            }
        }
        self.wetness = wet;
        self.base_wetness = base_wet;
        self.mix_ready = true;
    }

    /// Mud, sand, gravel, grass at one cell. A field that has not been
    /// blended reports a one-hot share of those four, or zeros.
    pub fn mix_texel(&self, idx: usize) -> [f32; 4] {
        if self.mix_ready && self.mix.len() >= (idx + 1) * MIX_CHANNELS {
            let base = idx * MIX_CHANNELS;
            return [
                self.mix[base],
                self.mix[base + 1],
                self.mix[base + 2],
                self.mix[base + 3],
            ];
        }
        let mut mix = [0.0; 4];
        if let Some(channel) = mix_channel(SurfaceKind::from_u8(self.kind[idx])) {
            mix[channel] = 1.0;
        }
        mix
    }

    /// Flat pad with a torn rut and a grip boundary. Film and deep slurry
    /// sit in that rut, on the soil the tire cut. The pictures are this field.
    /// The rut is stamped by the same wheel path the lap uses, so the lip is
    /// the solver's lip.
    pub fn gallery() -> Self {
        let mut field = Self::flat(0.16);
        let res = field.res;
        let cell = field.cell;
        let half = field.size * 0.5;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                if (2.2..6.4).contains(&x) && z.abs() < 2.4 {
                    if x < 3.9 {
                        field.kind[idx] = SurfaceKind::Gravel as u8;
                        field.wetness[idx] = 0.06;
                        field.base_wetness[idx] = 0.06;
                        field.water[idx] = 0.0;
                        field.loose[idx] = 0.0;
                        field.roughness[idx] = 0.96;
                    } else {
                        field.kind[idx] = SurfaceKind::WetMud as u8;
                        field.wetness[idx] = 0.94;
                        field.base_wetness[idx] = 0.9;
                        field.water[idx] = 0.0;
                        field.loose[idx] = 0.03;
                        field.roughness[idx] = 0.42;
                    }
                }
                if x < -6.2 && hash2(i as f32, j as f32) > 0.55 {
                    field.kind[idx] = SurfaceKind::Grass as u8;
                    field.wetness[idx] = 0.16;
                    field.water[idx] = 0.0;
                }
                // The gallery rut is cut through mud. A dry-crust pad would
                // stay crust under the tire, and the lip would have nothing
                // to tear against.
                if x.abs() < 0.42 && z.abs() < 3.05 {
                    field.kind[idx] = SurfaceKind::WetMud as u8;
                    field.wetness[idx] = 0.9;
                    field.base_wetness[idx] = 0.86;
                    field.water[idx] = 0.0;
                    field.darken[idx] = 0.12;
                    field.loose[idx] = 0.02;
                }
            }
        }
        let forward = Vec3::Z;
        let right = Vec3::X;
        let mut z = -2.55;
        while z <= 2.55 {
            // Hub, not the contact. Y = 0 put the axle in the ground and the
            // slope test then treated the whole chord as uphill of the tire.
            field.stamp_wheel(
                Vec3::new(0.0, 0.30, z),
                0.38,
                0.28,
                forward,
                right,
                0.08,
                0.35,
                0.12,
                true,
                0.0,
                0.0,
            );
            z += 0.1;
        }
        // Overlapping passes pack the floor into clay, which paints the cut
        // as a smooth plank. One torn pass leaves a fresh cut. Put that back
        // so the lip is mud torn out of mud, and the berm stays the placed pile.
        let res = field.res;
        let cell = field.cell;
        let half = field.size * 0.5;
        for j in 0..res {
            for i in 0..res {
                let idx = j * res + i;
                if field.kind[idx] != SurfaceKind::PackedClay as u8 {
                    continue;
                }
                let x = (i as f32 + 0.5) * cell - half;
                if x.abs() < 0.5 {
                    field.kind[idx] = SurfaceKind::FreshCut as u8;
                    field.darken[idx] = field.darken[idx].min(0.4);
                }
            }
        }
        // Pour into the groove only. A rectangle on the pad reads as a lid.
        // The near stretch is a film on packed clay. Past it, the fresh cut
        // holds a deep slurry. The lip and the berm stay dry.
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                let dug = (field.initial_height[idx] - field.bed[idx]).max(0.0);
                if dug < 0.02 || x.abs() > 0.30 {
                    continue;
                }
                let floor = SurfaceKind::from_u8(field.kind[idx]);
                if !matches!(
                    floor,
                    SurfaceKind::FreshCut | SurfaceKind::PackedClay | SurfaceKind::WetMud
                ) {
                    continue;
                }
                // The seam bends with the groove so the two regimes are not a
                // straight card edge. Every wet cell is one or the other.
                // High frequency across the groove, so the seam is not one straight cut.
                let seam = -1.18 + (noise2(x * 5.5, z * 1.7) - 0.5) * 0.55;
                if z > -2.4 && z < seam {
                    field.kind[idx] = SurfaceKind::PackedClay as u8;
                    field.water[idx] = 0.034;
                    field.wetness[idx] = 0.55;
                    field.darken[idx] = 0.32;
                    field.roughness[idx] = 0.72;
                } else if z >= seam && z < -0.15 {
                    field.kind[idx] = SurfaceKind::FreshCut as u8;
                    field.water[idx] = 0.15;
                    field.wetness[idx] = 1.0;
                    field.darken[idx] = field.darken[idx].min(0.45);
                }
            }
        }
        field
    }

    pub fn sample(&self, x: f32, z: f32) -> SoilSample {
        let half = self.size * 0.5;
        let u = (x + half) / self.cell - 0.5;
        let v = (z + half) / self.cell - 0.5;
        if u < 0.0 || v < 0.0 || u >= self.res as f32 - 1.0 || v >= self.res as f32 - 1.0 {
            return SoilSample {
                height: 0.0,
                wetness: 0.18,
                water: 0.0,
                loose: 0.0,
                darken: 0.04,
                roughness: 0.9,
                softness: softness_of(0.18, 0.0, 0.0),
                rut: 0.0,
                kind: SurfaceKind::Undisturbed as u8,
                placed: 0.0,
                slurry: 0.0,
                crust: 0.0,
                crack: 0.0,
                passes: 0,
                rest: 0.0,
                mix: [0.0; 4],
            };
        }
        let i = u.floor() as usize;
        let j = v.floor() as usize;
        let tx = u - i as f32;
        let ty = v - j as f32;
        let s00 = self.raw(i, j);
        let s10 = self.raw(i + 1, j);
        let s01 = self.raw(i, j + 1);
        let s11 = self.raw(i + 1, j + 1);
        let height = bilerp(s00.0, s10.0, s01.0, s11.0, tx, ty);
        let wetness = bilerp(s00.1, s10.1, s01.1, s11.1, tx, ty);
        let water = bilerp(s00.2, s10.2, s01.2, s11.2, tx, ty);
        let loose = bilerp(s00.3, s10.3, s01.3, s11.3, tx, ty);
        let darken = bilerp(s00.4, s10.4, s01.4, s11.4, tx, ty);
        let roughness = bilerp(s00.5, s10.5, s01.5, s11.5, tx, ty);
        let ni = u.round().clamp(0.0, (self.res - 1) as f32) as usize;
        let nj = v.round().clamp(0.0, (self.res - 1) as f32) as usize;
        let nidx = nj * self.res + ni;
        let kind = self.kind[nidx];
        let placed = self.placed[nidx];
        let rut = (self.initial_height[nidx] - self.bed[nidx]).max(0.0);
        SoilSample {
            height,
            wetness,
            water,
            loose,
            darken,
            roughness,
            softness: softness_of(wetness, loose, water),
            rut,
            kind,
            placed,
            slurry: self.slurry[nidx],
            crust: self.crust[nidx],
            crack: self.crack[nidx],
            passes: self.passes[nidx],
            rest: self.rest[nidx],
            mix: self.mix_texel(nidx),
        }
    }

    fn raw(&self, i: usize, j: usize) -> (f32, f32, f32, f32, f32, f32, f32) {
        let idx = j * self.res + i;
        (
            self.height[idx],
            self.wetness[idx],
            self.water[idx],
            self.loose[idx],
            self.darken[idx],
            self.roughness[idx],
            (self.initial_height[idx] - self.bed[idx]).max(0.0),
        )
    }

    fn sync(&mut self, idx: usize) {
        self.height[idx] = self.bed[idx] + self.placed[idx] + self.slurry[idx];
    }

    pub fn rebuild_height(&mut self) {
        for idx in 0..self.bed.len() {
            self.sync(idx);
        }
    }

    /// Put bed, placed, and loose back and rebuild height. For proof stills after
    /// a later drive phase that should not change the frame geometry.
    pub fn restore_bed_layers(&mut self, bed: Vec<f32>, placed: Vec<f32>, loose: Vec<f32>) {
        self.bed = bed;
        self.placed = placed;
        self.loose = loose;
        for idx in 0..self.bed.len() {
            self.sync(idx);
        }
    }

    /// Remove `drop` metres. Placed spoil comes off before the bed is cut.
    /// The returned thickness is volume that has to be put down somewhere else.
    fn excavate(&mut self, idx: usize, drop: f32) -> f32 {
        if drop <= 0.0 {
            return 0.0;
        }
        let from_placed = drop.min(self.placed[idx]);
        self.placed[idx] -= from_placed;
        self.bed[idx] -= drop - from_placed;
        self.sync(idx);
        drop
    }

    pub fn cut_volume(&self) -> f32 {
        let area = self.cell * self.cell;
        self.initial_height
            .iter()
            .zip(self.bed.iter())
            .map(|(h0, bed)| (h0 - bed).max(0.0))
            .sum::<f32>()
            * area
    }

    pub fn placed_volume(&self) -> f32 {
        let area = self.cell * self.cell;
        self.placed.iter().sum::<f32>() * area
    }

    /// Move already-placed spoil onto a landing cell. Does not mint soil.
    pub fn land_clod(&mut self, x: f32, z: f32, volume: f32) {
        let Some(dst) = self.idx_at(x, z) else {
            return;
        };
        let here = SurfaceKind::from_u8(self.kind[dst]);
        if matches!(here, SurfaceKind::FreshCut | SurfaceKind::PackedClay) {
            self.wetness[dst] = (self.wetness[dst] + 0.08).min(1.0);
            return;
        }
        let moved = self.take_placed_near(dst, volume.max(0.0));
        if moved <= 0.0 {
            if self.placed[dst] > 0.004 {
                self.kind[dst] = SurfaceKind::PlacedClod as u8;
            }
            return;
        }
        let dh = moved / (self.cell * self.cell);
        self.placed[dst] += dh;
        self.kind[dst] = SurfaceKind::PlacedClod as u8;
        self.loose[dst] = (self.loose[dst] + dh * 0.4).min(0.18);
        self.roughness[dst] = lerp(self.roughness[dst], 0.9, 0.35);
        self.wetness[dst] = (self.wetness[dst] + 0.08).min(1.0);
        self.sync(dst);
    }

    fn take_placed_near(&mut self, dst: usize, volume: f32) -> f32 {
        if volume <= 0.0 {
            return 0.0;
        }
        let area = self.cell * self.cell;
        let res = self.res;
        let di = (dst % res) as isize;
        let dj = (dst / res) as isize;
        let mut left = volume;
        let mut taken = 0.0;
        for radius in 1isize..5 {
            for dj2 in -radius..=radius {
                for di2 in -radius..=radius {
                    if di2.abs() != radius && dj2.abs() != radius {
                        continue;
                    }
                    let i = di + di2;
                    let j = dj + dj2;
                    if i < 0 || j < 0 || i >= res as isize || j >= res as isize {
                        continue;
                    }
                    let idx = j as usize * res + i as usize;
                    if idx == dst || self.placed[idx] <= 0.002 {
                        continue;
                    }
                    let kind = SurfaceKind::from_u8(self.kind[idx]);
                    if !matches!(kind, SurfaceKind::Berm | SurfaceKind::PlacedClod) {
                        continue;
                    }
                    let room = (self.placed[idx] - 0.002).max(0.0) * area;
                    let grab = left.min(room);
                    if grab <= 0.0 {
                        continue;
                    }
                    self.placed[idx] -= grab / area;
                    if self.placed[idx] <= 0.002 {
                        self.placed[idx] = 0.0;
                    }
                    self.sync(idx);
                    taken += grab;
                    left -= grab;
                    if left <= 1.0e-8 {
                        return taken;
                    }
                }
            }
        }
        taken
    }

    /// Mean moisture across the contact chord. The center cell still owns the
    /// kind: a patch that straddles a lip should not average that into mud.
    pub fn sample_patch(
        &self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        length: f32,
        width: f32,
    ) -> SoilSample {
        let along_n = 5;
        let across_n = 3;
        let mut wetness = 0.0;
        let mut water = 0.0;
        let mut loose = 0.0;
        let mut darken = 0.0;
        let mut n = 0.0_f32;
        for j in 0..across_n {
            for i in 0..along_n {
                let u = (i as f32 + 0.5) / along_n as f32 - 0.5;
                let v = (j as f32 + 0.5) / across_n as f32 - 0.5;
                let p = center + forward * (length * u) + right * (width * v);
                let s = self.sample(p.x, p.z);
                wetness += s.wetness;
                water += s.water;
                loose += s.loose;
                darken += s.darken;
                n += 1.0;
            }
        }
        let inv = 1.0 / n.max(1.0);
        let mid = self.sample(center.x, center.z);
        SoilSample {
            height: mid.height,
            wetness: wetness * inv,
            water: water * inv,
            loose: loose * inv,
            darken: darken * inv,
            roughness: mid.roughness,
            softness: softness_of(wetness * inv, loose * inv, water * inv),
            rut: mid.rut,
            kind: mid.kind,
            placed: mid.placed,
            slurry: mid.slurry,
            crust: mid.crust,
            crack: mid.crack,
            passes: mid.passes,
            rest: mid.rest,
            mix: mid.mix,
        }
    }

    /// Take loose smear from the footprint and hand it to the tread.
    /// The bed and the placed pile stay where they are.
    pub fn skim_loose(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        length: f32,
        width: f32,
        depth: f32,
    ) -> f32 {
        if depth <= 1.0e-6 {
            return 0.0;
        }
        let half_l = (length * 0.5).max(self.cell * 0.4);
        let half_w = (width * 0.5).max(self.cell * 0.25);
        let reach = half_l + half_w + self.cell;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;
        let mut taken = 0.0;
        let mut cells = 0.0;
        for j in j0.max(0)..j1.min(res - 1) {
            for i in i0.max(0)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * right.x + rel_z * right.z;
                let lz = rel_x * forward.x + rel_z * forward.z;
                if lx.abs() > half_w || lz.abs() > half_l {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let bite = (self.loose[idx] * 0.45).min(depth);
                if bite > 0.0 {
                    self.loose[idx] -= bite;
                    taken += bite;
                    cells += 1.0;
                }
            }
        }
        if cells > 0.0 {
            taken / cells
        } else {
            0.0
        }
    }

    /// Take up to `volume` cubic metres of loose smear from the footprint.
    /// The bed and the placed pile stay where they are. Returns the cubic
    /// metres actually removed.
    pub fn take_loose(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        length: f32,
        width: f32,
        volume: f32,
    ) -> f32 {
        if volume <= 1.0e-8 || !volume.is_finite() {
            return 0.0;
        }
        let half_l = (length * 0.5).max(self.cell * 0.4);
        let half_w = (width * 0.5).max(self.cell * 0.25);
        let reach = half_l + half_w + self.cell;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;
        let area = self.cell * self.cell;
        let mut left = volume;
        let mut taken = 0.0_f32;
        for j in j0.max(0)..j1.min(res - 1) {
            if left <= 1.0e-8 {
                break;
            }
            for i in i0.max(0)..i1.min(res - 1) {
                if left <= 1.0e-8 {
                    break;
                }
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * right.x + rel_z * right.z;
                let lz = rel_x * forward.x + rel_z * forward.z;
                if lx.abs() > half_w || lz.abs() > half_l {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let room = (self.loose[idx] * 0.85).max(0.0) * area;
                let grab = left.min(room);
                if grab <= 0.0 {
                    continue;
                }
                self.loose[idx] -= grab / area;
                taken += grab;
                left -= grab;
            }
        }
        taken
    }

    /// Put loose smear back. Used when a coat was skimmed and then cleared
    /// without shedding, so that thickness is not lost.
    pub fn give_loose(&mut self, center: Vec3, volume: f32) {
        if volume <= 1.0e-8 || !volume.is_finite() {
            return;
        }
        let Some(origin) = self.idx_at(center.x, center.z) else {
            return;
        };
        let area = self.cell * self.cell;
        let res = self.res;
        let di = (origin % res) as isize;
        let dj = (origin / res) as isize;
        let mut cells = [0usize; 9];
        let mut n = 0usize;
        for dj2 in -1..=1 {
            for di2 in -1..=1 {
                let i = di + di2;
                let j = dj + dj2;
                if i < 0 || j < 0 || i >= res as isize || j >= res as isize {
                    continue;
                }
                cells[n] = j as usize * res + i as usize;
                n += 1;
            }
        }
        if n == 0 {
            return;
        }
        let add = volume / (n as f32 * area);
        for idx in &cells[..n] {
            self.loose[*idx] = (self.loose[*idx] + add).min(0.18);
        }
    }

    /// Put a sidewall coat back on the bed. The rut behind the tire takes it.
    /// If that groove is not there yet, the shoulder keeps it as placed spoil.
    /// None of `volume` is discarded when a cell can be found.
    pub fn return_coat(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        volume: f32,
    ) -> CoatLanding {
        if volume <= 1.0e-8 || !volume.is_finite() {
            return CoatLanding {
                volume: 0.0,
                in_rut: 0.0,
            };
        }
        let fwd = Vec3::new(forward.x, 0.0, forward.z);
        let fwd = if fwd.length_squared() < 1.0e-8 {
            Vec3::Z
        } else {
            fwd.normalize()
        };
        let rt = Vec3::new(right.x, 0.0, right.z);
        let rt = if rt.length_squared() < 1.0e-8 {
            Vec3::new(fwd.z, 0.0, -fwd.x)
        } else {
            rt.normalize()
        };
        let half = self.size * 0.5;
        // Long enough for a single-cell ridge to hold the coat under the lip.
        let reach = 4.2_f32;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;
        let mut ridge = [(0usize, 0.0f32); 96];
        let mut n_ridge = 0usize;
        let mut flank = [(0usize, 0.0f32); 96];
        let mut n_flank = 0usize;
        let mut shoulder = [0usize; 64];
        let mut n_sh = 0usize;
        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let along = rel_x * fwd.x + rel_z * fwd.z;
                let lat = rel_x * rt.x + rel_z * rt.z;
                let behind = -along;
                // Far enough that the live stamp does not dig the spoil back
                // out. The ridge is the center of the rut, not a sheet across it.
                if !(1.05..=4.2).contains(&behind) {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let depth = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                if depth <= 0.01 {
                    if (0.22..0.48).contains(&lat.abs()) && n_sh < shoulder.len() {
                        shoulder[n_sh] = idx;
                        n_sh += 1;
                    }
                    continue;
                }
                // The cut itself, not the lip. One band, so the coat stacks
                // into a ridge instead of filming the shoulders.
                if lat.abs() <= 0.20 && n_ridge < ridge.len() {
                    ridge[n_ridge] = (idx, behind);
                    n_ridge += 1;
                }
            }
        }
        let into_rut = n_ridge > 0 || n_flank > 0;
        if n_ridge > 1 {
            ridge[..n_ridge]
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        if n_flank > 1 {
            flank[..n_flank]
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        if into_rut {
            let wrote = self.lay_ridge(&ridge[..n_ridge], &flank[..n_flank], volume);
            return CoatLanding {
                volume: wrote,
                in_rut: wrote,
            };
        }
        let mut chosen = [0usize; 64];
        let n = if n_sh > 0 {
            chosen[..n_sh].copy_from_slice(&shoulder[..n_sh]);
            n_sh
        } else if let Some(idx) = self.idx_at(center.x, center.z) {
            chosen[0] = idx;
            1
        } else {
            return CoatLanding {
                volume: 0.0,
                in_rut: 0.0,
            };
        };
        let wrote = self.lay_coat(&chosen[..n], volume, false);
        CoatLanding {
            volume: wrote,
            in_rut: 0.0,
        }
    }

    /// Center of the rut first, under the lip, so the coat is a ridge with
    /// sides. Flanks only take the overflow, and they stay lower than the ridge.
    fn lay_ridge(&mut self, ridge: &[(usize, f32)], flank: &[(usize, f32)], volume: f32) -> f32 {
        let mut ridge_i = [0usize; 96];
        let mut flank_i = [0usize; 96];
        for (n, cell) in ridge.iter().enumerate() {
            ridge_i[n] = cell.0;
        }
        for (n, cell) in flank.iter().enumerate() {
            flank_i[n] = cell.0;
        }
        let ridge = &ridge_i[..ridge.len()];
        let flank = &flank_i[..flank.len()];
        let mut left = volume;
        let mut wrote = 0.0_f32;
        let primary = if ridge.is_empty() { flank } else { ridge };
        let secondary = if ridge.is_empty() { &[][..] } else { flank };
        // Under the lip. Filling flush with the ground is a sheet.
        wrote += self.pour_frac(primary, &mut left, 0.80);
        if left > 1.0e-8 {
            wrote += self.pour_frac(secondary, &mut left, 0.80);
        }
        if left > 1.0e-8 && !primary.is_empty() {
            let idx = primary[primary.len() - 1];
            let add = left / self.cell / self.cell;
            self.note_coat(idx, add, true);
            wrote += left;
        }
        wrote
    }

    fn pour_frac(&mut self, cells: &[usize], left: &mut f32, frac: f32) -> f32 {
        let area = self.cell * self.cell;
        let mut wrote = 0.0_f32;
        for &idx in cells {
            if *left <= 1.0e-8 {
                break;
            }
            let depth = (self.initial_height[idx] - self.bed[idx]).max(0.0);
            let room = (depth * frac - self.placed[idx]).max(0.0);
            if room <= 1.0e-5 {
                continue;
            }
            let add = room.min(*left / area);
            self.note_coat(idx, add, true);
            let taken = add * area;
            *left -= taken;
            wrote += taken;
        }
        wrote
    }

    /// Spread `volume` across `cells` as placed thickness. A rut cell keeps
    /// its kind. A bare cell becomes spoil. The bed is not cut.
    ///
    /// Rut cells fill nearest-first, and only up to a ceiling inside the cut.
    /// Filling them flush is a flat sheet on the ground.
    fn lay_coat(&mut self, cells: &[usize], volume: f32, into_rut: bool) -> f32 {
        if cells.is_empty() || volume <= 0.0 {
            return 0.0;
        }
        let mut left = volume;
        let mut wrote = 0.0_f32;
        // Two centimetres under the lip, so the pile has a side.
        wrote += self.pour(cells, &mut left, into_rut, 0.02);
        if into_rut && left > 1.0e-8 {
            // Still inside the cut. The in-rut path needs the surface below the lip.
            wrote += self.pour(cells, &mut left, true, 0.012);
        }
        // A shallow groove can be shorter than the coat. The last rut cell
        // keeps the rest so none of it is deleted. The bed is still not cut.
        if left > 1.0e-8 {
            let idx = cells[cells.len() - 1];
            let add = left / self.cell / self.cell;
            self.note_coat(idx, add, into_rut);
            wrote += left;
        }
        wrote
    }

    fn pour(&mut self, cells: &[usize], left: &mut f32, into_rut: bool, margin: f32) -> f32 {
        let area = self.cell * self.cell;
        let mut wrote = 0.0_f32;
        for &idx in cells {
            if *left <= 1.0e-8 {
                break;
            }
            let room = if into_rut {
                let depth = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                let ceiling = (depth - margin).max(0.0);
                (ceiling - self.placed[idx]).max(0.0)
            } else {
                (0.06 - self.placed[idx]).max(0.0)
            };
            if room <= 1.0e-5 {
                continue;
            }
            let add = room.min(*left / area);
            self.note_coat(idx, add, into_rut);
            let taken = add * area;
            *left -= taken;
            wrote += taken;
        }
        wrote
    }

    fn note_coat(&mut self, idx: usize, add: f32, into_rut: bool) {
        self.placed[idx] += add;
        let depth = (self.initial_height[idx] - self.bed[idx]).max(0.0);
        if into_rut && depth > 0.01 {
            // Same kind the rut already has. The thickness is placed on the bed.
            self.wetness[idx] = self.wetness[idx].max(0.74);
            self.darken[idx] = lerp(self.darken[idx], 0.55, 0.4);
        } else {
            self.kind[idx] = SurfaceKind::Berm as u8;
            self.wetness[idx] = self.wetness[idx].max(0.7);
            self.darken[idx] = lerp(self.darken[idx], 0.4, 0.35);
            self.roughness[idx] = lerp(self.roughness[idx], 0.86, 0.3);
        }
        self.sync(idx);
    }

    /// Rocking forward and back packs loose spoil into the rut floor. Volume
    /// stays on the heightfield: loose drops and the bed rises in place.
    pub fn compact_rut_under_patch(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        radius: f32,
        width: f32,
        pace: f32,
        dt: f32,
    ) {
        if pace <= 0.0 || dt <= 0.0 {
            return;
        }
        let half_l = radius.max(0.05) * 0.95;
        let half_w = (width.max(0.02) * 0.5).max(self.cell * 0.45);
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let fwd = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let rgt = Vec3::new(right.x, 0.0, right.z).normalize_or_zero();
        if fwd.length_squared() < 1.0e-6 {
            return;
        }
        let i0 = (((center.x - half_l - half) / self.cell).floor() as isize).clamp(0, self.res as isize - 1);
        let i1 = (((center.x + half_l - half) / self.cell).ceil() as isize).clamp(0, self.res as isize - 1);
        let j0 = (((center.z - half_l - half) / self.cell).floor() as isize).clamp(0, self.res as isize - 1);
        let j1 = (((center.z + half_l - half) / self.cell).ceil() as isize).clamp(0, self.res as isize - 1);
        let blend_cap = if pace > 2.0 { 0.58 } else { 0.35 };
        let blend = (1.0 - (-pace * dt).exp()).min(blend_cap);
        for j in j0..=j1 {
            for i in i0..=i1 {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                let lx = (x - center.x) * rgt.x + (z - center.z) * rgt.z;
                let lz = (x - center.x) * fwd.x + (z - center.z) * fwd.z;
                if lx.abs() > half_w || lz.abs() > half_l {
                    continue;
                }
                let along = (1.0 - (lz.abs() / half_l).powi(2)).max(0.0);
                let across = (1.0 - (lx.abs() / half_w).powi(2)).max(0.0);
                let w = along * across;
                if w < 0.08 {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let loose = self.loose[idx];
                if loose < 1.0e-4 {
                    continue;
                }
                let pack = loose * blend * w;
                self.loose[idx] = (loose - pack).max(0.0);
                self.bed[idx] += pack;
                self.height[idx] += pack;
                let wet = self.wetness[idx];
                self.darken[idx] =
                    (self.darken[idx] + 18.0 * w * dt * (0.40 + wet)).min(1.0);
                self.roughness[idx] = lerp(self.roughness[idx], 0.22, 0.08 * w * blend);
                let _ = area;
                self.sync(idx);
            }
        }
    }

    /// Pack loose smear from the lane into the deepest rut floor (m³ budget).
    pub fn pack_loose_in_region(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        mut budget_m3: f32,
    ) -> f32 {
        if budget_m3 <= 1.0e-8 {
            return 0.0;
        }
        let area = self.cell * self.cell;
        let half = self.size * 0.5;
        let mut loose_cells: Vec<(usize, f32)> = Vec::new();
        let mut deep_cells: Vec<(usize, f32)> = Vec::new();
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                let depth = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                if depth >= 0.012 {
                    deep_cells.push((idx, depth));
                }
                let loose = self.loose[idx];
                if loose >= 1.0e-5 {
                    loose_cells.push((idx, loose));
                }
            }
        }
        if loose_cells.is_empty() || deep_cells.is_empty() {
            return 0.0;
        }
        deep_cells.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        loose_cells.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut loose_i = 0usize;
        let mut packed = 0.0_f32;
        for (floor_idx, _) in deep_cells {
            if budget_m3 <= 1.0e-8 || loose_i >= loose_cells.len() {
                break;
            }
            let mut lift = (budget_m3 / area).min(0.06);
            if lift < 1.0e-6 {
                break;
            }
            let mut gathered = 0.0_f32;
            while gathered < lift && loose_i < loose_cells.len() {
                let src = loose_cells[loose_i].0;
                let avail = self.loose[src];
                let need = lift - gathered;
                let take = avail.min(need);
                if take < 1.0e-6 {
                    loose_i += 1;
                    continue;
                }
                self.loose[src] = avail - take;
                gathered += take;
                if self.loose[src] < 1.0e-5 {
                    loose_i += 1;
                }
            }
            if gathered < 1.0e-6 {
                continue;
            }
            self.bed[floor_idx] += gathered;
            self.height[floor_idx] += gathered;
            let wet = self.wetness[floor_idx];
            self.darken[floor_idx] = (self.darken[floor_idx] + 0.14 * (0.40 + wet)).min(1.0);
            self.roughness[floor_idx] = lerp(self.roughness[floor_idx], 0.22, 0.32);
            self.sync(floor_idx);
            let vol = gathered * area;
            packed += vol;
            budget_m3 -= vol;
        }
        packed
    }

    pub fn idx_at(&self, x: f32, z: f32) -> Option<usize> {
        let half = self.size * 0.5;
        if x < -half || z < -half || x >= half || z >= half {
            return None;
        }
        let i = ((x + half) / self.cell).floor() as isize;
        let j = ((z + half) / self.cell).floor() as isize;
        if i < 0 || j < 0 || i >= self.res as isize || j >= self.res as isize {
            return None;
        }
        Some(j as usize * self.res + i as usize)
    }

    /// Plastic rut under one tire.
    ///
    /// `lane` is the visible groove, stamped once per side of the axle along
    /// the path the chassis drove. Front and rear share that slot. The cut is
    /// a flat ribbon wider than one heightfield cell: a thinner kernel
    /// stair-steps into a row of pits, and those pits read as a nest of scars
    /// on the berm. The per-tire call is only a scuff under the tread, plus
    /// the slow dig when a tire spins in place.
    /// The cell the tire is cutting. A berm beside an earlier rut is spoil,
    /// not the soil the Bekker depth is taken from.
    fn soil_for_force(&self, x: f32, z: f32) -> SoilSample {
        let spoil = |s: SoilSample| {
            s.placed > 0.008
                || matches!(s.kind(), SurfaceKind::Berm | SurfaceKind::PlacedClod)
        };
        let s = self.sample(x, z);
        if !spoil(s) {
            return s;
        }
        let step = self.cell;
        for ring in 1..12 {
            let d = ring as f32 * step;
            for (dx, dz) in [(d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)] {
                let n = self.sample(x + dx, z + dz);
                if !spoil(n) {
                    return n;
                }
            }
        }
        s
    }

    pub fn stamp_wheel(
        &mut self,
        center: Vec3,
        radius: f32,
        width: f32,
        forward: Vec3,
        right: Vec3,
        sinkage: f32,
        slip: f32,
        dt: f32,
        lane: bool,
        normal: f32,
        push: f32,
    ) {
        let loaded = normal > 1.0 || push > 1.0;
        let forced = if loaded {
            // Spoil on the flank is not a softer bed. The depth is the clay
            // under the tire, so a second rut does not run away to the cap.
            // A rested crust on that clay is harder than the wet cut.
            let s = self.soil_for_force(center.x, center.z);
            rut_depth_on_sample(normal, push, radius, width, &s)
        } else {
            0.0
        };
        if sinkage < 0.004 && forced < 0.004 {
            return;
        }
        let len = contact_length(radius, sinkage.max(forced)).max(radius * 0.35);
        let slip_n = slip.abs().min(1.2);
        // One visit bites this deep. A later visit of the same lane, after the
        // tire has left, raises the pass count and cuts deeper into that slot.
        // The bed is only lowered. A second stamp never fills the first cut
        // back up to a fresh shallow groove. Slip still digs past the rolling
        // bite, on the slow rate, so a driver can back out of a hole they spin.
        // A tire force replaces that cap: the bed is the Bekker depth of the
        // resultant, and a harder push is a deeper bed.
        let one_pass = if loaded {
            forced
        } else {
            (sinkage * 0.82).min(0.055)
        };
        let spun = if loaded {
            forced
        } else {
            sinkage * (0.48 + 0.90 * slip_n)
        };
        let sigma = (width * 0.42).max(self.cell * 0.34);
        // Flat floor, then a broken step. A ramped shoulder smeared the cut
        // lip and the placed berm into one slope. The puddle tread is the
        // exception: a flat ribbon wider than the tire reads as a slab.
        let tread = lane && self.tread_profile;
        let tread_core = (width * 0.42).max(self.cell * 0.45);
        // Reaches the contract cell just outside the tread, and stops short of
        // the centreline. A longer outer flank on only one side left the other
        // track as a step: the 0.20 m sample past the core went from the floor
        // to zero.
        let tread_edge = tread_core + self.cell * 2.75;
        let half_w = if tread {
            tread_edge
        } else if lane {
            self.cell * 1.15
        } else {
            sigma * 1.6
        };
        let lip_band = if tread {
            0.0
        } else if lane {
            self.cell * 1.05
        } else {
            0.0
        };
        // Spoil sits in the cell against the lip. A wider band flattened
        // the pile into the crust, so the berm disappeared beside the step.
        let berm_band = if tread { self.cell * 2.4 } else { self.cell * 0.9 };
        // The contact chord, in world XZ. A ribbon of a couple of cells
        // past the tire cuts the slope in front of the hub, and that cut
        // walks up the mesh on the next step.
        let half_l = if lane {
            (len * 0.5 + self.cell * 0.25).max(self.cell * 1.05)
        } else {
            len * 0.5
        };
        let along_reach = half_l + self.cell * 0.35;
        let lateral_reach = if lane {
            half_w + lip_band + berm_band + self.cell * 0.4
        } else {
            sigma * 3.2
        };
        let reach = along_reach + lateral_reach;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;

        let mut vol = 0.0_f32;
        let mut clods = [0usize; 128];
        let mut n_clod = 0usize;
        let mut berms = [0usize; 192];
        let mut n_berm = 0usize;
        let mut cuts = [0usize; 160];
        let mut n_cut = 0usize;
        let area = self.cell * self.cell;

        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * right.x + rel_z * right.z;
                let lz = rel_x * forward.x + rel_z * forward.z;
                let along = if lane {
                    let u = (lz.abs() / along_reach.max(0.05)).clamp(0.0, 1.0);
                    if u < 0.78 {
                        1.0
                    } else {
                        ((1.0 - u) / 0.22).clamp(0.0, 1.0)
                    }
                } else {
                    1.0 - (lz.abs() / along_reach.max(0.05)).clamp(0.0, 1.0)
                };
                if along <= 0.0 {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                // Vertical column at this world XZ. `penetration` is how far
                // the surface sits below the hub, measured in world Y.
                // On the contact it equals the sink. A column up the slope
                // is higher, ahead of the hub, and is not this tire's rut.
                // The puddle tread measures that on the solid bed. The sheet
                // on a flat floor is not a slope, and skipping it left the
                // contract cell ahead of the hub uncut.
                let column = if tread {
                    self.bed[idx] + self.placed[idx]
                } else {
                    self.height[idx]
                };
                let penetration = column - (center.y - radius);
                let up_the_slope = penetration > sinkage + 0.03 && lz.abs() > self.cell * 0.5;
                if up_the_slope {
                    continue;
                }
                let ax = lx.abs();
                let across = if tread {
                    if ax <= tread_core {
                        1.0
                    } else if ax >= tread_edge {
                        0.0
                    } else {
                        let t = (ax - tread_core) / (tread_edge - tread_core).max(1.0e-3);
                        let s = t * t * (3.0 - 2.0 * t);
                        1.0 - s
                    }
                } else if lane {
                    if ax <= half_w {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (-0.5 * (lx / sigma.max(0.02)).powi(2)).exp()
                };

                // The tread tail is shallow. The lane lip still starts at 0.2,
                // or a scuff would claim the shoulder.
                if across > if tread { 0.12 } else { 0.2 } {
                    // Scuffs do not claim a pass and do not reset the gap, or
                    // the lane stamp in the same step would stay on pass 1.
                    let passes = if lane {
                        self.note_pass(idx)
                    } else {
                        self.passes[idx].max(1)
                    };
                    let groove = if loaded {
                        forced
                    } else if lane {
                        pass_groove(one_pass, passes)
                    } else {
                        one_pass
                    };
                    let want = if loaded {
                        forced.min(0.40)
                    } else {
                        spun.max(groove).min(0.40)
                    };
                    let fast = if lane { groove } else { groove.min(0.008) };
                    let already = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                    let local_want = if lane {
                        want * across
                    } else if spun > one_pass {
                        want * across
                    } else {
                        fast * across
                    };
                    let local_groove = fast * across;
                    if already < local_want {
                        let room = (local_want - already).min(0.40 - already).max(0.0);
                        let rate = if already < local_groove { 48.0 } else { 2.2 };
                        let drop = room * (1.0 - (-dt * rate).exp()) * along;
                        let removed = self.excavate(idx, drop);
                        if removed > 0.0 && n_cut < cuts.len() {
                            cuts[n_cut] = idx;
                            n_cut += 1;
                        }
                        self.loose[idx] = (self.loose[idx] + removed * 0.45).min(0.16);
                        vol += removed * area;
                    }
                    let wet = self.wetness[idx];
                    self.darken[idx] =
                        (self.darken[idx] + 16.0 * across * along * dt * (0.45 + wet)).min(1.0);
                    let target_r = lerp(0.62, 0.16, wet);
                    let blend = (5.0 * dt * across * along).min(0.45);
                    self.roughness[idx] = lerp(self.roughness[idx], target_r, blend);
                    // Firm covers stay themselves. A dry crust does not become mud
                    // because a tire crossed it. Standing water keeps the cut slick.
                    self.kind[idx] =
                        opened_kind(self.kind[idx], self.water[idx], self.darken[idx]);
                } else if lane && ax > half_w && ax <= half_w + lip_band && along > 0.35 {
                    let piled = SurfaceKind::from_u8(self.kind[idx]);
                    if matches!(piled, SurfaceKind::Berm) && self.placed[idx] > 0.012 {
                        continue;
                    }
                    // A later pass must not stack another lump on the same chunk,
                    // and must not shave that chunk back into the lip.
                    if piled == SurfaceKind::PlacedClod {
                        if self.placed[idx] < CREST_CLOD_CAP - 1.0e-4 && n_clod < clods.len() {
                            clods[n_clod] = idx;
                            n_clod += 1;
                        }
                        continue;
                    }
                    let h = hash2(i as f32 * 1.7 + 0.2, j as f32 * 1.3);
                    self.kind[idx] = SurfaceKind::CutLip as u8;
                    self.roughness[idx] = lerp(self.roughness[idx], 0.84, 0.35);
                    // A notch in the crest. The step is broken, not a ramp.
                    if h < 0.42 {
                        let bite = (0.55 + h) * one_pass;
                        let already = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                        let drop = (bite - already).max(0.0).min(0.06) * along;
                        if drop > 0.001 {
                            let removed = self.excavate(idx, drop);
                            if removed > 0.0 && n_cut < cuts.len() {
                                cuts[n_cut] = idx;
                                n_cut += 1;
                            }
                            self.loose[idx] = (self.loose[idx] + removed * 0.3).min(0.16);
                            vol += removed * area;
                        }
                    } else if h > 0.66 && n_clod < clods.len() {
                        clods[n_clod] = idx;
                        n_clod += 1;
                    }
                } else if {
                    // The tread berm sits in the contract cell beside the
                    // groove. Starting it at tread_edge parks the pile past
                    // that cell, and the shoulder disappears from the map.
                    let start = if tread {
                        tread_core + self.cell * 2.55
                    } else {
                        half_w + lip_band
                    };
                    let end = if tread {
                        start + self.cell * 1.35
                    } else {
                        half_w + lip_band + berm_band
                    };
                    ax > start && ax <= end && along > 0.25
                } {
                    // Spoil inside both tires lands on the centreline and
                    // joins into one berm between the tracks.
                    // Any spoil closer to the centreline than the tire becomes
                    // the band between the tracks.
                    if tread && cx.abs() + 0.06 < center.x.abs() {
                        continue;
                    }
                    if n_berm < berms.len() {
                        berms[n_berm] = idx;
                        n_berm += 1;
                    }
                }
            }
        }

        if loaded && normal + push > 350.0 {
            self.creep_wet_berm_under_patch(
                center,
                forward,
                right,
                radius,
                width,
                lane,
                normal + push,
                dt,
            );
        }

        if vol <= 1.0e-8 {
            return;
        }
        if n_berm == 0 {
            // The cut mass still has to land. A stamp with no shoulder
            // cell would otherwise throw that soil away.
            self.pile_spoil(&mut berms, &mut n_berm, &cuts[..n_cut]);
        }
        self.deposit_spoil(vol, &clods[..n_clod], &berms[..n_berm]);
    }

    /// Count a later visit. Overlapping stamps from the tire still on the
    /// cell keep the same pass and do not raise the bed.
    fn note_pass(&mut self, idx: usize) -> u8 {
        if self.rest[idx] > PASS_GAP {
            self.passes[idx] = self.passes[idx].saturating_add(1);
        }
        self.rest[idx] = 0.0;
        self.passes[idx].max(1)
    }

    /// Lateral neighbours of the cut, in world XZ, used when the shoulder
    /// band itself was empty. Does not mint cells that are off the field.
    fn pile_spoil(&self, berms: &mut [usize], n_berm: &mut usize, seeds: &[usize]) {
        let res = self.res as isize;
        for seed in seeds {
            let i = (*seed % self.res) as isize;
            let j = (*seed / self.res) as isize;
            for (di, dj) in [
                (2, 0),
                (-2, 0),
                (0, 2),
                (0, -2),
                (3, 0),
                (-3, 0),
                (2, 1),
                (-2, 1),
                (2, -1),
                (-2, -1),
            ] {
                let ii = i + di;
                let jj = j + dj;
                if ii < 1 || jj < 1 || ii >= res - 1 || jj >= res - 1 {
                    continue;
                }
                let idx = jj as usize * self.res + ii as usize;
                if *n_berm >= berms.len() || berms[..*n_berm].contains(&idx) {
                    continue;
                }
                berms[*n_berm] = idx;
                *n_berm += 1;
            }
        }
    }

    /// Put every cubic metre of cut soil onto crest clods and then the berm.
    /// Nothing is discarded when a cell is full.
    fn deposit_spoil(&mut self, vol: f32, clods: &[usize], berms: &[usize]) {
        let area = self.cell * self.cell;
        let mut left = vol;
        for (n, idx) in clods.iter().copied().enumerate() {
            if left <= 1.0e-8 {
                break;
            }
            let room = (CREST_CLOD_CAP - self.placed[idx]).max(0.0);
            if room <= 1.0e-4 {
                continue;
            }
            let h = hash2(n as f32 * 2.1, idx as f32 * 0.17);
            let lump = 0.014 + 0.02 * h;
            let add = lump.min(room).min(left / area);
            self.placed[idx] += add;
            self.kind[idx] = SurfaceKind::PlacedClod as u8;
            self.loose[idx] = (self.loose[idx] + add * 0.55).min(0.18);
            self.roughness[idx] = lerp(self.roughness[idx], 0.94, 0.45);
            self.sync(idx);
            left -= add * area;
        }
        if berms.is_empty() {
            if left > 1.0e-8 && !clods.is_empty() {
                let add = left / (clods.len() as f32 * area);
                for idx in clods {
                    self.placed[*idx] += add;
                    self.kind[*idx] = SurfaceKind::PlacedClod as u8;
                    self.sync(*idx);
                }
            }
            return;
        }
        let mut open: Vec<usize> = berms.to_vec();
        let cap = 0.08_f32;
        for _ in 0..6 {
            if left <= 1.0e-8 || open.is_empty() {
                break;
            }
            let share = left / (open.len() as f32 * area);
            let mut next = Vec::with_capacity(open.len());
            for idx in open {
                let room = (cap - self.placed[idx]).max(0.0);
                let add = share.min(room);
                if add > 0.0 {
                    self.placed[idx] += add;
                    self.kind[idx] = SurfaceKind::Berm as u8;
                    self.loose[idx] = (self.loose[idx] + add * 0.35).min(0.18);
                    self.roughness[idx] = lerp(self.roughness[idx], 0.9, 0.3);
                    self.darken[idx] = lerp(self.darken[idx], 0.22, 0.12);
                    self.sync(idx);
                    left -= add * area;
                }
                if self.placed[idx] < cap - 1.0e-4 {
                    next.push(idx);
                }
            }
            open = next;
        }
        if left > 1.0e-8 {
            let add = left / (berms.len() as f32 * area);
            for idx in berms {
                self.placed[*idx] += add;
                self.kind[*idx] = SurfaceKind::Berm as u8;
                self.sync(*idx);
            }
        }
    }

    /// Lower every column the patch covers down to `floor_y`. The removed
    /// volume is split between the berm beside the patch and `airborne`.
    /// Firm covers keep their kind. Nothing is dropped on the floor.
    pub fn plow_contact(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        length: f32,
        width: f32,
        floor_y: f32,
        throw_dir: Vec3,
        throw_frac: f32,
    ) -> PlowCut {
        if !floor_y.is_finite() || length <= 1.0e-4 || width <= 1.0e-4 {
            return PlowCut::default();
        }
        let fwd = horizontal(forward);
        let mut rt = horizontal(right);
        if rt.length_squared() < 1.0e-8 {
            rt = Vec3::new(fwd.z, 0.0, -fwd.x);
        }
        let half_l = length * 0.5;
        let half_w = width * 0.5;
        let reach = half_l + half_w + self.cell * 2.2;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;
        let area = self.cell * self.cell;
        let throw_h = horizontal(throw_dir);
        let throw_s = throw_h.x * rt.x + throw_h.z * rt.z;
        let band = self.cell * 1.35;

        let mut bed = 0.0_f32;
        let mut moved = 0.0_f32;
        let mut cuts: Vec<usize> = Vec::new();
        let mut soft: Vec<usize> = Vec::new();
        let mut shoulders: Vec<usize> = Vec::new();

        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * rt.x + rel_z * rt.z;
                let lz = rel_x * fwd.x + rel_z * fwd.z;
                let idx = j as usize * self.res + i as usize;
                let inside = lx.abs() <= half_w && lz.abs() <= half_l;
                if inside {
                    let pen = self.height[idx] - floor_y;
                    if pen < 0.004 {
                        continue;
                    }
                    let drop = pen.min(0.22);
                    let before_bed = self.bed[idx];
                    let before_placed = self.placed[idx];
                    let removed = self.excavate(idx, drop);
                    if removed <= 0.0 {
                        continue;
                    }
                    let bed_drop = (before_bed - self.bed[idx]).max(0.0);
                    let placed_drop = (before_placed - self.placed[idx]).max(0.0);
                    bed += bed_drop * area;
                    moved += placed_drop * area;
                    if bed_drop > 1.0e-5 {
                        self.kind[idx] =
                            opened_kind(self.kind[idx], self.water[idx], self.darken[idx]);
                        self.darken[idx] = (self.darken[idx] + 0.42).min(1.0);
                        self.loose[idx] = (self.loose[idx] + bed_drop * 0.45).min(0.16);
                        self.roughness[idx] = lerp(self.roughness[idx], 0.22, 0.35);
                    }
                    cuts.push(idx);
                } else if lz.abs() <= half_l + self.cell * 0.35 {
                    let outward = if throw_s.abs() < 0.15 {
                        lx.abs()
                    } else {
                        lx * throw_s.signum()
                    };
                    if outward > half_w && outward <= half_w + band {
                        shoulders.push(idx);
                        if !firm_cover(SurfaceKind::from_u8(self.kind[idx])) {
                            soft.push(idx);
                        }
                    }
                }
            }
        }

        let vol = bed + moved;
        if vol <= 1.0e-8 {
            return PlowCut::default();
        }
        let frac = throw_frac.clamp(0.0, 0.72);
        let mut thrown = vol * frac;
        let mut berm = vol - thrown;
        if thrown < 1.0e-6 {
            berm += thrown;
            thrown = 0.0;
        }
        if berm > 1.0e-8 {
            if soft.is_empty() {
                let mut extra = [0usize; 192];
                let mut n = 0usize;
                self.pile_spoil(&mut extra, &mut n, &cuts);
                for idx in extra.iter().take(n) {
                    if cuts.contains(idx) {
                        continue;
                    }
                    if firm_cover(SurfaceKind::from_u8(self.kind[*idx])) {
                        if !shoulders.contains(idx) {
                            shoulders.push(*idx);
                        }
                    } else if !soft.contains(idx) {
                        soft.push(*idx);
                    }
                }
            }
            soft.retain(|idx| !cuts.contains(idx));
            shoulders.retain(|idx| !cuts.contains(idx));
            if !soft.is_empty() {
                self.deposit_spoil(berm, &[], &soft);
            } else if !shoulders.is_empty() {
                self.lay_kept(&shoulders, berm);
            } else if !cuts.is_empty() {
                // No shoulder on the field. Keep the mass on the lip cells
                // that were cut rather than dropping it.
                thrown += berm;
                berm = 0.0;
            } else {
                thrown += berm;
                berm = 0.0;
            }
        }
        self.airborne += thrown;
        PlowCut {
            bed,
            moved,
            berm,
            thrown,
        }
    }

    /// Put thrown soil down. Firm ground keeps its kind. The landing walks
    /// off a fresh cut so the furrow stays open. Returns the surface point.
    pub fn receive_thrown(&mut self, x: f32, z: f32, volume: f32) -> Option<Vec3> {
        let take = volume.min(self.airborne).max(0.0);
        if take <= 1.0e-8 || !take.is_finite() {
            return None;
        }
        let (x, z) = self.outside_rut(x, z);
        let idx = self.idx_at(x, z)?;
        self.airborne -= take;
        let dh = take / (self.cell * self.cell);
        self.placed[idx] += dh;
        if !firm_cover(SurfaceKind::from_u8(self.kind[idx]))
            && self.kind[idx] != SurfaceKind::Berm as u8
        {
            self.kind[idx] = SurfaceKind::PlacedClod as u8;
        }
        self.loose[idx] = (self.loose[idx] + dh * 0.35).min(0.18);
        self.roughness[idx] = lerp(self.roughness[idx], 0.9, 0.3);
        self.sync(idx);
        let half = self.size * 0.5;
        let i = idx % self.res;
        let j = idx / self.res;
        Some(Vec3::new(
            (i as f32 + 0.5) * self.cell - half,
            self.height[idx],
            (j as f32 + 0.5) * self.cell - half,
        ))
    }

    /// Lay the spoil from a roll as a trench and a berm.
    ///
    /// One column was holding more than a metre, and the cells around it were
    /// painted as separate tablets. The cut stays the trench that was dug,
    /// with the steps blurred out. The spoil becomes one ridge on the flank
    /// it was thrown, then a lower tongue. Cut volume and the sum of
    /// `placed` stay put. The berm is height, not a second colour.
    pub fn lay_debris_field(&mut self) {
        let placed_total: f32 = self.placed.iter().copied().sum();
        let res = self.res;
        let cell = self.cell;
        let n = res * res;
        let half = self.size * 0.5;
        let mut dug = vec![false; n];
        let mut cut_sum = 0.0_f32;
        let mut cwx = 0.0_f32;
        let mut cwz = 0.0_f32;
        let mut cw = 0.0_f32;
        let mut pwx = 0.0_f32;
        let mut pwz = 0.0_f32;
        let mut pw = 0.0_f32;
        for idx in 0..n {
            let depth = (self.initial_height[idx] - self.bed[idx]).max(0.0);
            cut_sum += depth;
            let i = idx % res;
            let j = idx / res;
            let x = (i as f32 + 0.5) * cell - half;
            let z = (j as f32 + 0.5) * cell - half;
            if depth > 0.01 {
                dug[idx] = true;
                cwx += x * depth;
                cwz += z * depth;
                cw += depth;
            }
            let placed = self.placed[idx];
            if placed > 0.0 {
                pwx += x * placed;
                pwz += z * placed;
                pw += placed;
            }
        }
        if cut_sum <= 1.0e-6 || placed_total <= 1.0e-6 || cw <= 1.0e-6 {
            return;
        }
        let ccx = cwx / cw;
        let ccz = cwz / cw;
        let (tx, tz) = if pw > 1.0e-6 {
            let mut dx = pwx / pw - ccx;
            let mut dz = pwz / pw - ccz;
            let len = dx.hypot(dz);
            if len < 0.15 {
                (1.0, 0.0)
            } else {
                dx /= len;
                dz /= len;
                (dx, dz)
            }
        } else {
            (1.0, 0.0)
        };

        // The scar stays the trench the roll cut. A filled bowl of the same
        // volume is only a few centimetres deep, and this view cannot see it.
        let mut mask = vec![false; n];
        let mut seen = vec![false; n];
        let mut best_comp: Vec<usize> = Vec::new();
        for start in 0..n {
            if !dug[start] || seen[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            seen[start] = true;
            while let Some(idx) = stack.pop() {
                comp.push(idx);
                let i = (idx % res) as isize;
                let j = (idx / res) as isize;
                for dj in -1isize..=1 {
                    for di in -1isize..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ii = i + di;
                        let jj = j + dj;
                        if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                            continue;
                        }
                        let nidx = jj as usize * res + ii as usize;
                        if dug[nidx] && !seen[nidx] {
                            seen[nidx] = true;
                            stack.push(nidx);
                        }
                    }
                }
            }
            if comp.len() > best_comp.len() {
                best_comp = comp;
            }
        }
        for idx in &best_comp {
            mask[*idx] = true;
        }
        // One-cell holes in that trench only. A wider fill became a pond.
        for _ in 0..2 {
            let prev = mask.clone();
            for j in 1..res - 1 {
                for i in 1..res - 1 {
                    let idx = j * res + i;
                    if prev[idx] {
                        continue;
                    }
                    let mut nbr = 0;
                    for dj in -1isize..=1 {
                        for di in -1isize..=1 {
                            if di == 0 && dj == 0 {
                                continue;
                            }
                            if prev[(j as isize + dj) as usize * res + (i as isize + di) as usize] {
                                nbr += 1;
                            }
                        }
                    }
                    if nbr >= 6 {
                        mask[idx] = true;
                    }
                }
            }
        }

        let orig: Vec<f32> = (0..n)
            .map(|idx| (self.initial_height[idx] - self.bed[idx]).max(0.0))
            .collect();
        let mut depth = vec![0.0_f32; n];
        for idx in 0..n {
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            let mut near = mask[idx];
            if !near {
                for dj in -1isize..=1 {
                    for di in -1isize..=1 {
                        let ii = i + di;
                        let jj = j + dj;
                        if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                            continue;
                        }
                        if mask[jj as usize * res + ii as usize] {
                            near = true;
                        }
                    }
                }
            }
            if !near {
                continue;
            }
            let mut acc = 0.0_f32;
            let mut wacc = 0.0_f32;
            for dj in -1isize..=1 {
                for di in -1isize..=1 {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    let w = match di.abs() + dj.abs() {
                        0 => 0.52,
                        1 => 0.09,
                        _ => 0.02,
                    };
                    let nidx = jj as usize * res + ii as usize;
                    let src = if mask[nidx] { orig[nidx] } else { 0.0 };
                    acc += src * w;
                    wacc += w;
                }
            }
            if wacc > 0.0 {
                depth[idx] = acc / wacc;
            }
        }
        let dw_sum: f32 = depth.iter().copied().sum();
        if dw_sum <= 1.0e-6 {
            return;
        }
        let depth_scale = cut_sum / dw_sum;
        for d in &mut depth {
            *d *= depth_scale;
        }
        for idx in 0..n {
            self.bed[idx] = self.initial_height[idx] - depth[idx];
            let prev = SurfaceKind::from_u8(self.kind[idx]);
            if firm_cover(prev) {
                continue;
            }
            // Same mud on the lip and the berm. A second colour was a tile.
            if depth[idx] > 0.05 {
                self.kind[idx] = SurfaceKind::FreshCut as u8;
            } else if matches!(
                prev,
                SurfaceKind::FreshCut
                    | SurfaceKind::PackedClay
                    | SurfaceKind::CutLip
                    | SurfaceKind::Berm
                    | SurfaceKind::PlacedClod
            ) {
                self.kind[idx] = if self.wetness[idx] > 0.45 {
                    SurfaceKind::WetMud as u8
                } else {
                    SurfaceKind::Undisturbed as u8
                };
            }
        }

        // Distance from the trench floor, and which floor cell is nearest,
        // so the berm follows the flank instead of piling at the centroid.
        let mut lip = vec![f32::MAX; n];
        let mut nearest = vec![usize::MAX; n];
        let mut queue: Vec<usize> = Vec::new();
        for idx in 0..n {
            if depth[idx] <= 0.03 {
                continue;
            }
            lip[idx] = 0.0;
            nearest[idx] = idx;
            queue.push(idx);
        }
        let diag = cell * std::f32::consts::SQRT_2;
        let mut qh = 0usize;
        while qh < queue.len() {
            let idx = queue[qh];
            qh += 1;
            if lip[idx] > 1.15 {
                continue;
            }
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            for (di, dj, step) in [
                (1isize, 0, cell),
                (-1, 0, cell),
                (0, 1, cell),
                (0, -1, cell),
                (1, 1, diag),
                (1, -1, diag),
                (-1, 1, diag),
                (-1, -1, diag),
            ] {
                let ii = i + di;
                let jj = j + dj;
                if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                    continue;
                }
                let nidx = jj as usize * res + ii as usize;
                let nd = lip[idx] + step;
                if nd + 1.0e-5 < lip[nidx] && nd < 1.3 {
                    lip[nidx] = nd;
                    nearest[nidx] = nearest[idx];
                    queue.push(nidx);
                }
            }
        }
        let mut weight = vec![0.0_f32; n];
        for idx in 0..n {
            if depth[idx] > 0.02 || lip[idx] > 1.15 || nearest[idx] == usize::MAX {
                continue;
            }
            let i = idx % res;
            let j = idx / res;
            let x = (i as f32 + 0.5) * cell - half;
            let z = (j as f32 + 0.5) * cell - half;
            let ni = nearest[idx] % res;
            let nj = nearest[idx] / res;
            let nx = (ni as f32 + 0.5) * cell - half;
            let nz = (nj as f32 + 0.5) * cell - half;
            let ox = x - nx;
            let oz = z - nz;
            let olen = ox.hypot(oz).max(1.0e-4);
            let flank = (ox / olen) * tx + (oz / olen) * tz;
            let best = lip[idx];
            // Steep ridge on the throw flank, then a lower tongue of the
            // thrown mass. One piece of spoil, not a ring of cells.
            let ridge = (-0.5 * ((best - 0.16) / 0.07).powi(2)).exp();
            let tongue = 0.42 * (-0.5 * ((best - 0.48) / 0.16).powi(2)).exp();
            let flank_t = ((flank + 0.05) / 0.85).clamp(0.0, 1.0);
            let aim = 0.12 + 0.88 * flank_t * flank_t * (3.0 - 2.0 * flank_t);
            let tongue_t = ((flank - 0.15) / 0.7).clamp(0.0, 1.0);
            let tongue_aim = tongue_t * tongue_t * (3.0 - 2.0 * tongue_t);
            let wobble = 0.98 + 0.02 * noise2(x * 0.15, z * 0.15);
            let w = (ridge * aim + tongue * tongue_aim) * wobble;
            if w > 0.04 {
                weight[idx] = w;
            }
        }
        for _ in 0..4 {
            let wsum: f32 = weight.iter().copied().sum();
            if wsum <= 1.0e-6 {
                break;
            }
            let scale = placed_total / wsum;
            let max_h = weight.iter().copied().fold(0.0_f32, f32::max) * scale;
            if (0.16..=0.24).contains(&max_h) {
                break;
            }
            if max_h > 0.24 {
                let src = weight.clone();
                for idx in 0..n {
                    if depth[idx] > 0.02 {
                        weight[idx] = 0.0;
                        continue;
                    }
                    let i = (idx % res) as isize;
                    let j = (idx / res) as isize;
                    let mut acc = src[idx] * 0.40;
                    let mut wacc = 0.40_f32;
                    let mut near = src[idx] > 0.0;
                    for (di, dj) in [(1isize, 0), (-1, 0), (0, 1), (0, -1)] {
                        let ii = i + di;
                        let jj = j + dj;
                        if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                            continue;
                        }
                        let nidx = jj as usize * res + ii as usize;
                        if depth[nidx] > 0.02 {
                            continue;
                        }
                        acc += src[nidx] * 0.15;
                        wacc += 0.15;
                        near = near || src[nidx] > 0.0;
                    }
                    weight[idx] = if near { acc / wacc } else { 0.0 };
                }
            } else {
                for w in &mut weight {
                    if *w > 0.0 {
                        *w = w.powf(1.4);
                    }
                }
            }
        }
        let wsum: f32 = weight.iter().copied().sum();
        if wsum <= 1.0e-6 {
            for idx in 0..n {
                self.sync(idx);
            }
            return;
        }
        let scale = placed_total / wsum;
        for idx in 0..n {
            let on_floor = depth[idx] > 0.02;
            self.placed[idx] = if on_floor { 0.0 } else { weight[idx] * scale };
            self.sync(idx);
        }
        let now: f32 = self.placed.iter().copied().sum();
        if now > 1.0e-8 && (now - placed_total).abs() > 1.0e-5 {
            let fix = placed_total / now;
            for idx in 0..n {
                self.placed[idx] *= fix;
                self.sync(idx);
            }
        }
    }

    fn lay_kept(&mut self, cells: &[usize], volume: f32) {
        if cells.is_empty() || volume <= 0.0 {
            return;
        }
        let area = self.cell * self.cell;
        let dh = volume / (cells.len() as f32 * area);
        for idx in cells {
            self.placed[*idx] += dh;
            if !firm_cover(SurfaceKind::from_u8(self.kind[*idx])) {
                self.kind[*idx] = SurfaceKind::Berm as u8;
            }
            self.sync(*idx);
        }
    }

    fn outside_rut(&self, mut x: f32, mut z: f32) -> (f32, f32) {
        for _ in 0..16 {
            let Some(idx) = self.idx_at(x, z) else {
                break;
            };
            let rut = (self.initial_height[idx] - self.bed[idx]).max(0.0);
            let opened = matches!(
                SurfaceKind::from_u8(self.kind[idx]),
                SurfaceKind::FreshCut | SurfaceKind::CutLip | SurfaceKind::PackedClay
            );
            if rut < 0.02 && !opened {
                break;
            }
            let mut best = (x, z);
            let mut best_rut = rut;
            let step = self.cell;
            for (dx, dz) in [(step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)] {
                let (nx, nz) = (x + dx, z + dz);
                let Some(nidx) = self.idx_at(nx, nz) else {
                    continue;
                };
                let nr = (self.initial_height[nidx] - self.bed[nidx]).max(0.0);
                if nr < best_rut - 0.004 {
                    best_rut = nr;
                    best = (nx, nz);
                }
            }
            if (best.0 - x).abs() < 1.0e-6 && (best.1 - z).abs() < 1.0e-6 {
                let push = Vec3::new(x, 0.0, z).normalize_or_zero() * step;
                if push.length_squared() < 1.0e-8 {
                    x += step;
                } else {
                    x += push.x;
                    z += push.z;
                }
            } else {
                x = best.0;
                z = best.1;
            }
        }
        (x, z)
    }

    /// Standing water in a rut or a hole joins the wet soil under it.
    /// The water column is cleared. The soil it takes leaves the bed and
    /// the two become one slurry thickness. Volume of water plus soil is
    /// unchanged. Mixing stays in the cell. [`flow_slurry`](Self::flow_slurry)
    /// is what carries that body downhill.
    pub fn mix_slurry(&mut self) {
        let n = self.water.len();
        for idx in 0..n {
            let water = self.water[idx];
            if water <= 0.008 {
                continue;
            }
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            if !mixes_as_slurry(kind) {
                continue;
            }
            let moisture = self.wetness[idx];
            if moisture < 0.15 || !self.in_rut_or_hole(idx) {
                continue;
            }
            // Cap the bite so a deep pond cannot empty the column under the map.
            // The cap still conserves volume: whatever soil leaves is the slurry's soil.
            let soil_m = slurry_soil_depth(water, moisture).min(0.35);
            if soil_m <= 1.0e-5 {
                continue;
            }
            let removed = self.excavate(idx, soil_m);
            self.slurry[idx] += removed + water;
            self.water[idx] = 0.0;
            self.sync(idx);
        }
    }

    /// High rut of slurry, the deeper rut it can run into, and the berm on
    /// the shoulder of the high rut. World metres. The low band starts where
    /// the high band ends, so the two share a face.
    pub fn paint_slurry_flow(&mut self) -> SlurryFlowBed {
        let bed = SlurryFlowBed {
            high_z0: -0.55,
            high_z1: 0.12,
            low_z0: 0.12,
            low_z1: 1.45,
            x0: -1.15,
            x1: 1.15,
        };
        let half = self.size * 0.5;
        let moisture = SLURRY_COMPARE_MOISTURE;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < bed.x0 || x >= bed.x1 {
                    continue;
                }
                let idx = j * self.res + i;
                if z >= bed.high_z0 && z < bed.high_z1 {
                    self.bed[idx] -= 0.035;
                    self.slurry[idx] = 0.16;
                    self.water[idx] = 0.0;
                    self.wetness[idx] = moisture;
                    self.kind[idx] = SurfaceKind::FreshCut as u8;
                    self.sync(idx);
                    if i + 1 < self.res {
                        let nx = ((i + 1) as f32 + 0.5) * self.cell - half;
                        let nidx = j * self.res + i + 1;
                        if nx >= bed.x1 && self.initial_height[nidx] - self.bed[nidx] < 0.012 {
                            self.placed[nidx] = 0.20;
                            self.kind[nidx] = SurfaceKind::Berm as u8;
                            self.wetness[nidx] = 0.70;
                            self.loose[nidx] = 0.08;
                            self.water[nidx] = 0.0;
                            self.sync(nidx);
                        }
                    }
                } else if z >= bed.low_z0 && z < bed.low_z1 {
                    self.bed[idx] -= 0.18;
                    self.slurry[idx] = 0.0;
                    self.water[idx] = 0.0;
                    self.wetness[idx] = moisture;
                    self.kind[idx] = SurfaceKind::FreshCut as u8;
                    self.sync(idx);
                }
            }
        }
        bed
    }

    /// Cubic metres of slurry whose cell centers lie in the rectangle.
    pub fn slurry_volume_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> f32 {
        self.column_volume_in(x0, x1, z0, z1, &self.slurry)
    }

    /// Slurry ahead of the treads (bow) and in the rut lane behind them, cubic metres.
    pub fn slurry_bow_and_rut_volumes(
        &self,
        pool_x0: f32,
        pool_x1: f32,
        pool_z0: f32,
        pool_z1: f32,
        front_hub: (f32, f32),
        rear_left: (f32, f32),
        rear_right: (f32, f32),
        track_half: f32,
        forward: Vec3,
    ) -> (f32, f32) {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let cell = self.cell;
        let area = cell * cell;
        let half = self.size * 0.5;
        let lip = 0.52_f32;
        let inner_margin = cell * 1.4;
        let rear_z = rear_left.1.min(rear_right.1);
        let front_z = front_hub.1;
        let bow_near = front_z + 0.18;
        let bow_far = (front_z + 0.95).min(pool_z1 - lip);
        let wake_top = rear_z - 0.14;
        let wake_floor = pool_z0 + inner_margin;
        let on_rut = |cx: f32| {
            (cx - rear_left.0).abs() <= track_half + cell * 0.2
                || (cx - rear_right.0).abs() <= track_half + cell * 0.2
        };
        let mut bow = 0.0_f32;
        let mut rut = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz >= pool_z1 {
                    continue;
                }
                let idx = j * self.res + i;
                let depth = self.slurry[idx];
                if depth <= 1.0e-5 {
                    continue;
                }
                let rel_x = cx - front_hub.0;
                let rel_z = cz - front_z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                if (bow_near..=bow_far).contains(&along)
                    && lat.abs() <= track_half + cell * 0.65
                {
                    bow += depth * area;
                } else if cz <= wake_top
                    && cz >= wake_floor
                    && on_rut(cx)
                {
                    rut += depth * area;
                }
            }
        }
        (bow, rut)
    }

    /// Deepest bed cut, metres, among cells whose centers lie in the rectangle.
    pub fn max_bed_cut_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> f32 {
        let half = self.size * 0.5;
        let mut best = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                best = best.max((self.initial_height[idx] - self.bed[idx]).max(0.0));
            }
        }
        best
    }

    /// Cubic metres of placed soil whose cell centers lie in the rectangle.
    pub fn placed_volume_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> f32 {
        self.column_volume_in(x0, x1, z0, z1, &self.placed)
    }

    /// Cubic metres of loose smear in the rectangle (depth × cell area).
    pub fn loose_volume_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> f32 {
        self.column_volume_in(x0, x1, z0, z1, &self.loose)
    }

    /// World XZ of the wet berm centroid in a rectangle, for parking load on it.
    pub fn berm_centroid_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> (f32, f32) {
        let half = self.size * 0.5;
        let mut sx = 0.0_f32;
        let mut sz = 0.0_f32;
        let mut w = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                let p = self.placed[idx];
                if p < 0.012 {
                    continue;
                }
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                if !matches!(kind, SurfaceKind::Berm | SurfaceKind::PlacedClod) {
                    continue;
                }
                sx += x * p;
                sz += z * p;
                w += p;
            }
        }
        if w <= 1.0e-6 {
            return (x1 - 0.35, (z0 + z1) * 0.5);
        }
        (sx / w, sz / w)
    }

    fn column_volume_in(&self, x0: f32, x1: f32, z0: f32, z1: f32, column: &[f32]) -> f32 {
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let mut sum = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                sum += column[j * self.res + i];
            }
        }
        sum * area
    }

    /// Creep the berm into neighbouring slurry, then let the slurry run
    /// downhill. Water plus soil is unchanged. The bed is not cut.
    pub fn step_slurry(&mut self, dt: f32) {
        if dt <= 0.0 || self.slurry.iter().all(|s| *s <= 1.0e-5) {
            return;
        }
        self.creep_berm_into_slurry(dt);
        self.flow_slurry(dt);
    }

    /// Let bow slurry and berm spoil run back into the rut after motion stops.
    /// Same as [`step_slurry`](Self::step_slurry), split into short steps.
    pub fn step_slurry_settle(&mut self, dt: f32) {
        self.step_slurry(dt);
    }

    /// Advance slurry in short bites so a long settle does not overshoot the head.
    pub fn advance_slurry(&mut self, seconds: f32) {
        if seconds <= 0.0 {
            return;
        }
        let steps = (seconds / 0.05).ceil().max(1.0) as u32;
        let dt = seconds / steps as f32;
        for _ in 0..steps {
            self.step_slurry(dt);
        }
    }

    /// Level a bow wave and berm back into the bed and rut after a pass or shed.
    pub fn advance_slurry_settle(&mut self, seconds: f32) {
        self.advance_slurry(seconds);
    }

    /// Wet berm under a loaded tire or body panel. Spoil slumps into the rut
    /// or the pool beside it. The berm column loses `placed`; the rut and
    /// pool gain slurry. Nothing is minted or deleted.
    pub fn creep_wet_berm_under_patch(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        radius: f32,
        width: f32,
        lane: bool,
        load_n: f32,
        dt: f32,
    ) {
        if dt <= 0.0 || load_n < 350.0 {
            return;
        }
        let fwd = horizontal(forward);
        let mut rt = horizontal(right);
        if rt.length_squared() < 1.0e-8 {
            rt = Vec3::new(fwd.z, 0.0, -fwd.x);
        }
        let sigma = (width * 0.42).max(self.cell * 0.34);
        let half_w = if lane {
            self.cell * 1.15
        } else {
            sigma * 1.65
        };
        let lip_band = if lane { self.cell * 1.05 } else { 0.0 };
        let berm_band = self.cell * 1.05;
        let half_l = if lane {
            (contact_length(radius, 0.05) * 0.5 + self.cell * 0.35).max(self.cell * 1.05)
        } else {
            radius * 0.95
        };
        let along_reach = half_l + self.cell * 0.35;
        let lateral_reach = half_w + lip_band + berm_band + self.cell * 0.55;
        let reach = along_reach + lateral_reach;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;
        let placed0 = self.placed.clone();
        let slurry0 = self.slurry.clone();
        let mut take = vec![0.0_f32; placed0.len()];
        let mut give = vec![0.0_f32; placed0.len()];

        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * rt.x + rel_z * rt.z;
                let lz = rel_x * fwd.x + rel_z * fwd.z;
                let along = if lane {
                    1.0 - (lz.abs() / along_reach.max(0.05)).clamp(0.0, 1.0)
                } else {
                    (-0.5 * (lz / half_l.max(0.05)).powi(2)).exp()
                };
                if along <= 0.05 {
                    continue;
                }
                let ax = lx.abs();
                let idx = j as usize * self.res + i as usize;
                if placed0[idx] < 0.008 {
                    continue;
                }
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                if kind != SurfaceKind::Berm {
                    continue;
                }
                let moisture = self.wetness[idx];
                if moisture < 0.32 {
                    continue;
                }
                let under = if lane {
                    ax <= half_w + lip_band + berm_band
                } else {
                    (-0.5 * (lx / sigma.max(0.02)).powi(2)).exp() > 0.12
                };
                if !under {
                    continue;
                }
                let top = self.bed[idx] + placed0[idx] + slurry0[idx];
                let mut dest: Option<usize> = None;
                let mut dest_h = top;
                for (di, dj) in [(1_isize, 0), (-1, 0), (0, 1), (0, -1)] {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 1 || jj < 1 || ii >= res - 1 || jj >= res - 1 {
                        continue;
                    }
                    let nidx = jj as usize * self.res + ii as usize;
                    let wet_dest = slurry0[nidx] > 0.004 || self.water[nidx] > 0.004;
                    if !wet_dest {
                        continue;
                    }
                    let surf = self.bed[nidx] + placed0[nidx] + slurry0[nidx];
                    if surf < dest_h - 0.003 || (slurry0[nidx] > 0.004 && surf <= top + 0.012) {
                        dest_h = surf;
                        dest = Some(nidx);
                    }
                }
                let Some(nidx) = dest else {
                    continue;
                };
                let depth = (wet_berm_creep_under_load_rate(kind, moisture, load_n) * dt * along)
                    .min(placed0[idx] * 0.18)
                    .min(placed0[idx]);
                if depth <= 1.0e-6 {
                    continue;
                }
                take[idx] += depth;
                give[nidx] += depth;
            }
        }

        for idx in 0..placed0.len() {
            let lost = take[idx];
            let gained = give[idx];
            if lost <= 0.0 && gained <= 0.0 {
                continue;
            }
            if lost > 0.0 {
                self.placed[idx] = (placed0[idx] - lost).max(0.0);
                if self.placed[idx] <= 0.002 {
                    self.placed[idx] = 0.0;
                }
            }
            if gained > 0.0 {
                let before = slurry0[idx];
                self.slurry[idx] = before + gained;
                let m = self.wetness[idx].clamp(0.08, 0.92);
                self.wetness[idx] = (m * before / self.slurry[idx].max(1.0e-4)).clamp(0.08, 0.92);
            }
            self.sync(idx);
        }
    }

    /// Slump every wet berm cell in a rectangle while load sits on the shoulder.
    /// Spoil joins slurry in the rut or the pool beside it.
    pub fn slump_wet_berms_in(
        &mut self,
        dt: f32,
        load_n: f32,
        src_x0: f32,
        src_x1: f32,
        src_z0: f32,
        src_z1: f32,
        dest_x0: f32,
        dest_x1: f32,
        dest_z0: f32,
        dest_z1: f32,
        rut_lane_x: f32,
        rut_z0: f32,
        rut_z1: f32,
    ) {
        if dt <= 0.0 || load_n < 350.0 {
            return;
        }
        let half = self.size * 0.5;
        let in_rect =
            |x: f32, z: f32, x0: f32, x1: f32, z0: f32, z1: f32| {
                x >= x0 && x < x1 && z >= z0 && z < z1
            };
        let in_rut_lane = |x: f32, z: f32| {
            x.abs() <= rut_lane_x && z >= rut_z0 && z < rut_z1
        };
        let placed0 = self.placed.clone();
        let slurry0 = self.slurry.clone();
        let mut take = vec![0.0_f32; placed0.len()];
        let mut give = vec![0.0_f32; placed0.len()];
        let res = self.res as isize;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if !in_rect(x, z, src_x0, src_x1, src_z0, src_z1) {
                    continue;
                }
                let idx = j * self.res + i;
                if placed0[idx] < 0.008 {
                    continue;
                }
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                if !matches!(kind, SurfaceKind::Berm | SurfaceKind::PlacedClod | SurfaceKind::CutLip)
                {
                    continue;
                }
                let moisture = self.wetness[idx];
                if moisture < 0.32 {
                    continue;
                }
                let top = self.bed[idx] + placed0[idx] + slurry0[idx];
                let mut dest_rut: Option<usize> = None;
                let mut dest_pool: Option<usize> = None;
                let mut rut_h = top;
                let mut pool_h = top;
                let ii = i as isize;
                let jj = j as isize;
                for radius in 1isize..=3 {
                    for dj in -radius..=radius {
                        for di in -radius..=radius {
                            if di.abs() != radius && dj.abs() != radius {
                                continue;
                            }
                            let ni = ii + di;
                            let nj = jj + dj;
                            if ni < 0 || nj < 0 || ni >= res || nj >= res {
                                continue;
                            }
                            let nx = (ni as f32 + 0.5) * self.cell - half;
                            let nz = (nj as f32 + 0.5) * self.cell - half;
                            if !in_rect(nx, nz, dest_x0, dest_x1, dest_z0, dest_z1) {
                                continue;
                            }
                            let nidx = nj as usize * self.res + ni as usize;
                            if slurry0[nidx] <= 0.004 && self.water[nidx] <= 0.004 {
                                continue;
                            }
                            let surf = self.bed[nidx] + placed0[nidx] + slurry0[nidx];
                            if in_rut_lane(nx, nz) {
                                if surf <= top + 0.02 && surf < rut_h {
                                    rut_h = surf;
                                    dest_rut = Some(nidx);
                                }
                            } else if surf <= top + 0.02 && surf < pool_h {
                                pool_h = surf;
                                dest_pool = Some(nidx);
                            }
                        }
                    }
                }
                let nidx = dest_rut.or(dest_pool);
                let Some(nidx) = nidx else {
                    continue;
                };
                let depth = (wet_berm_creep_under_load_rate(kind, moisture, load_n) * dt)
                    .min(placed0[idx] * 0.22)
                    .min(placed0[idx]);
                if depth <= 1.0e-6 {
                    continue;
                }
                take[idx] += depth;
                give[nidx] += depth;
            }
        }
        for idx in 0..placed0.len() {
            let lost = take[idx];
            let gained = give[idx];
            if lost <= 0.0 && gained <= 0.0 {
                continue;
            }
            if lost > 0.0 {
                self.placed[idx] = (placed0[idx] - lost).max(0.0);
                if self.placed[idx] <= 0.002 {
                    self.placed[idx] = 0.0;
                }
            }
            if gained > 0.0 {
                let before = slurry0[idx];
                self.slurry[idx] = before + gained;
                let m = self.wetness[idx].clamp(0.08, 0.92);
                self.wetness[idx] = (m * before / self.slurry[idx].max(1.0e-4)).clamp(0.08, 0.92);
            }
            self.sync(idx);
        }
    }

    /// A berm loses placed soil into a lower neighbour that already holds slurry.
    /// The soil joins that slurry. The move is a few millimetres a second.
    fn creep_berm_into_slurry(&mut self, dt: f32) {
        let res = self.res;
        let n = self.placed.len();
        let placed0 = self.placed.clone();
        let slurry0 = self.slurry.clone();
        let mut take = vec![0.0_f32; n];
        let mut give = vec![0.0_f32; n];
        for idx in 0..n {
            if placed0[idx] < 0.008 {
                continue;
            }
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            let top = self.bed[idx] + placed0[idx] + slurry0[idx];
            let mut dest: Option<usize> = None;
            let mut dest_h = top;
            for (di, dj) in [(1_isize, 0), (-1, 0), (0, 1), (0, -1)] {
                let ii = i + di;
                let jj = j + dj;
                if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                    continue;
                }
                let nidx = jj as usize * res + ii as usize;
                if slurry0[nidx] < 0.004 && !self.in_rut_or_hole(nidx) {
                    continue;
                }
                let surf = self.bed[nidx] + placed0[nidx] + slurry0[nidx];
                if surf < dest_h - 0.004 {
                    dest_h = surf;
                    dest = Some(nidx);
                }
            }
            let Some(nidx) = dest else {
                continue;
            };
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            let moisture = self.wetness[idx].max(self.wetness[nidx]).clamp(0.08, 0.92);
            let depth = (berm_creep_rate(kind, moisture) * dt)
                .min(placed0[idx] * 0.12)
                .min(placed0[idx]);
            if depth <= 1.0e-6 {
                continue;
            }
            take[idx] += depth;
            give[nidx] += depth;
        }
        for idx in 0..n {
            let lost = take[idx];
            let gained = give[idx];
            if lost <= 0.0 && gained <= 0.0 {
                continue;
            }
            if lost > 0.0 {
                self.placed[idx] = (placed0[idx] - lost).max(0.0);
            }
            if gained > 0.0 {
                let before = slurry0[idx];
                self.slurry[idx] = before + gained;
                let m = self.wetness[idx].clamp(0.08, 0.92);
                self.wetness[idx] = (m * before / self.slurry[idx].max(1.0e-4)).clamp(0.08, 0.92);
            }
            self.sync(idx);
        }
    }

    /// Move slurry from a higher free surface into a lower rut, hole, or
    /// slurry cell. Flat ground beside the rut does not receive it. The
    /// sum of the slurry column is unchanged.
    pub fn flow_slurry(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        if dt <= 0.0 {
            return;
        }
        let res = self.res;
        let dx = self.cell;
        let n = self.slurry.len();
        let head = self.slurry.clone();
        let mut budget = head.clone();
        let mut next = head.clone();
        for j in 0..res {
            for i in 0..res {
                let a = j * res + i;
                if i + 1 < res {
                    self.slurry_flux(a, a + 1, dt, dx, &head, &mut budget, &mut next);
                }
                if j + 1 < res {
                    self.slurry_flux(a, a + res, dt, dx, &head, &mut budget, &mut next);
                }
            }
        }
        self.slurry = next;
        for idx in 0..n {
            if (self.slurry[idx] - head[idx]).abs() > 0.0 {
                self.sync(idx);
            }
        }
    }

    fn slurry_flux(
        &self,
        a: usize,
        b: usize,
        dt: f32,
        dx: f32,
        head_slurry: &[f32],
        budget: &mut [f32],
        next: &mut [f32],
    ) {
        let sa = self.bed[a] + self.placed[a] + head_slurry[a];
        let sb = self.bed[b] + self.placed[b] + head_slurry[b];
        let dh = sa - sb;
        if dh.abs() < 0.003 {
            return;
        }
        let (donor, recv, head) = if dh > 0.0 { (a, b, dh) } else { (b, a, -dh) };
        if head_slurry[donor] <= 1.0e-4 || budget[donor] <= 1.0e-4 {
            return;
        }
        if head_slurry[recv] <= 1.0e-4 && !self.in_rut_or_hole(recv) {
            return;
        }
        let kind = SurfaceKind::from_u8(self.kind[donor]);
        let moisture = self.wetness[donor].clamp(0.08, 0.92);
        let conduct = slurry_flow_conduct(kind, moisture);
        let mut depth = head * conduct * dt / dx.max(1.0e-3) * 0.55;
        depth = depth.min(head * 0.48).min(budget[donor]);
        if depth <= 1.0e-6 {
            return;
        }
        budget[donor] -= depth;
        next[donor] -= depth;
        next[recv] += depth;
    }

    /// A tire in slurry. Slurry leaves the contact and the wake behind and
    /// piles ahead as a bow. The thinned column under the tire is the splash
    /// track. The bed is not cut here, nothing is sprayed, and nothing is
    /// pulled up out of the rut.
    pub fn raise_bow_wave(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
    ) -> BowWakePass {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let radius = radius.max(0.15);
        let half_track = width * 0.5 + self.cell * 0.55;
        let half_bow = half_track + self.cell * 0.35;
        let reach = radius * 1.6 + half_bow;
        let half = self.size * 0.5;
        let res = self.res as isize;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let track_back = -radius * 0.55;
        let track_front = radius * 0.12;
        let wake_far = -radius * 2.55;
        let wake_near = -radius * 0.52;
        // Ahead of the rubber, so the wave is in front of the tread and not under it.
        let bow_near = radius * 0.72;
        let bow_far = radius * 2.55;

        let area = self.cell * self.cell;
        let mut taken: Vec<(usize, f32)> = Vec::new();
        let mut weights: Vec<(usize, f32)> = Vec::new();
        let mut wake_m3 = 0.0_f32;
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                let idx = j as usize * self.res + i as usize;
                if (track_back..=track_front).contains(&along) && lat.abs() <= half_track {
                    let take = self.slurry[idx] * 0.86;
                    if take > 1.0e-6 {
                        taken.push((idx, take));
                    }
                } else if (wake_far..=wake_near).contains(&along) && lat.abs() <= half_track {
                    let take = self.slurry[idx] * 0.52;
                    if take > 1.0e-6 {
                        taken.push((idx, take));
                        wake_m3 += take * area;
                    }
                } else if (bow_near..=bow_far).contains(&along) && lat.abs() <= half_bow {
                    let span = (bow_far - bow_near).max(0.05);
                    let u = ((along - bow_near) / span).clamp(0.0, 1.0);
                    let crest = (1.0 - (u - 0.32).abs() / 0.68).max(0.0);
                    let across = 1.0 - (lat.abs() / half_bow).clamp(0.0, 1.0);
                    let w = crest * across;
                    if w > 0.03 {
                        weights.push((idx, w));
                    }
                }
            }
        }
        let removed: f32 = taken.iter().map(|(_, t)| *t).sum();
        let wsum: f32 = weights.iter().map(|(_, w)| *w).sum();
        if removed <= 1.0e-6 || wsum <= 1.0e-6 {
            return BowWakePass::default();
        }
        for (idx, take) in &taken {
            self.slurry[*idx] -= *take;
        }
        let mut bow_m3 = 0.0_f32;
        for (idx, w) in &weights {
            let add = removed * (*w / wsum);
            self.slurry[*idx] += add;
            bow_m3 += add * area;
        }
        for (idx, _) in taken.iter().chain(weights.iter()) {
            self.sync(*idx);
        }
        BowWakePass { bow_m3, wake_m3 }
    }

    /// Slurry scraped from under a rolling tread for ribbons and fine spray.
    /// The bed is not cut. Returns cubic metres taken from the column.
    pub fn scrape_slurry_from_tread(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
        want: f32,
    ) -> f32 {
        let want = want.max(0.0);
        if want <= 1.0e-8 {
            return 0.0;
        }
        let cells = self.slurry_under_tread(x, z, forward, radius, width);
        if cells.is_empty() {
            return 0.0;
        }
        let area = self.cell * self.cell;
        let available: f32 = cells.iter().map(|idx| self.slurry[*idx] * area).sum();
        if available <= 1.0e-8 {
            return 0.0;
        }
        let take = want.min(available * 0.08);
        let frac = (take / available).clamp(0.0, 1.0);
        let mut got = 0.0_f32;
        for idx in cells {
            let depth = self.slurry[idx] * frac;
            self.slurry[idx] -= depth;
            got += depth * area;
            self.sync(idx);
        }
        got
    }

    /// Lower the bed under a rolling tire in deep slurry. The slurry column
    /// stays on the ledger; the rut gets deeper relative to the rim.
    pub fn deepen_slurry_rut(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
        pace: f32,
        dt: f32,
    ) -> f32 {
        let pace = pace.max(0.0);
        let dt = dt.max(0.0);
        if pace <= 1.0e-8 || dt <= 1.0e-8 {
            return 0.0;
        }
        let carve = (pace * dt).min(if pace > 0.28 { 0.0068 } else { 0.0035 });
        if carve <= 1.0e-8 {
            return 0.0;
        }
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let radius = radius.max(0.15);
        let half_track = width * 0.5 + self.cell * 0.55;
        let track_back = -radius * 0.65;
        let track_front = radius * 0.18;
        let reach = radius + half_track + self.cell;
        let half = self.size * 0.5;
        let res = self.res as isize;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let mut deepest = 0.0_f32;
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                if !(track_back..=track_front).contains(&along) || lat.abs() > half_track {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                if self.slurry[idx] < 0.012 {
                    continue;
                }
                let across = 1.0 - (lat.abs() / half_track).clamp(0.0, 1.0);
                let depth = carve * (0.55 + 0.45 * across);
                self.bed[idx] -= depth;
                self.sync(idx);
                deepest = deepest.max(depth);
            }
        }
        deepest
    }

    /// Dig a bowl and fill it with one slurry. The free surface sits below the rim.
    /// Returns the cubic metres of slurry in the bowl.
    pub fn fill_slurry_pool(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        depth: f32,
        kind: SurfaceKind,
        moisture: f32,
    ) -> f32 {
        let depth = depth.max(0.0);
        if depth <= 1.0e-5 {
            return 0.0;
        }
        let moisture = moisture.clamp(0.08, 0.92);
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let mut vol = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                self.bed[idx] -= depth + 0.045;
                self.slurry[idx] = depth;
                self.water[idx] = 0.0;
                self.wetness[idx] = moisture;
                self.kind[idx] = kind as u8;
                self.darken[idx] = (self.darken[idx] + 0.42).min(1.0);
                self.loose[idx] = (self.loose[idx] + 0.02).min(0.12);
                self.sync(idx);
                vol += depth * area;
            }
        }
        vol
    }

    /// Dig a bowl and fill it with free water. Spoil from the bowl is placed
    /// on the apron outside the pool, so the soil sum does not change. The
    /// returned value is the cubic metres of water in the bowl.
    pub fn fill_water_pool(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        bowl: f32,
        depth: f32,
        kind: SurfaceKind,
    ) -> f32 {
        let bowl = bowl.max(0.0);
        let depth = depth.max(0.0);
        if depth <= 1.0e-5 || bowl <= 1.0e-5 {
            return 0.0;
        }
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let apron = 2.2_f32;
        let mut pool: Vec<usize> = Vec::new();
        let mut rim: Vec<usize> = Vec::new();
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                let idx = j * self.res + i;
                let inside = x >= x0 && x < x1 && z >= z0 && z < z1;
                let near = x >= x0 - apron
                    && x < x1 + apron
                    && z >= z0 - apron
                    && z < z1 + apron;
                if inside {
                    pool.push(idx);
                } else if near {
                    rim.push(idx);
                }
            }
        }
        if pool.is_empty() || rim.is_empty() {
            return 0.0;
        }
        let spoil = bowl * pool.len() as f32 * area;
        let cells = pool.len() as f32;
        let dh = spoil / (rim.len() as f32 * area);
        for &idx in &pool {
            self.bed[idx] -= bowl;
            self.water[idx] = depth;
            self.slurry[idx] = 0.0;
            self.cloud[idx] = 0.0;
            self.wetness[idx] = 0.96;
            self.kind[idx] = kind as u8;
            self.darken[idx] = (self.darken[idx] + 0.18).min(0.55);
            self.sync(idx);
        }
        for idx in rim {
            self.placed[idx] += dh;
            if self.placed[idx] > 0.01 {
                self.kind[idx] = SurfaceKind::Berm as u8;
            }
            self.sync(idx);
        }
        depth * cells * area
    }

    /// Tires meet the bed. A wide bowl is not a rut, and a water column is not
    /// slurry, so neither one is left on the contact for this pass.
    pub fn begin_ford_contacts(&mut self, x0: f32, x1: f32, z0: f32, z1: f32) -> FordContactMask {
        let half = self.size * 0.5;
        let film = 0.04_f32;
        let mut initial = Vec::new();
        let mut water = Vec::new();
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                if self.initial_height[idx] - self.bed[idx] > 1.0e-4 {
                    initial.push((idx, self.initial_height[idx]));
                    self.initial_height[idx] = self.bed[idx];
                }
                if self.water[idx] > film {
                    water.push((idx, self.water[idx]));
                    self.water[idx] = film;
                }
            }
        }
        FordContactMask { initial, water }
    }

    pub fn end_ford_contacts(&mut self, mask: FordContactMask) {
        for (idx, h0) in mask.initial {
            self.initial_height[idx] = h0;
        }
        for (idx, depth) in mask.water {
            self.water[idx] = depth;
        }
    }

    pub fn water_volume(&self) -> f32 {
        let area = self.cell * self.cell;
        self.water.iter().sum::<f32>() * area
    }

    pub fn cloud_volume(&self) -> f32 {
        let area = self.cell * self.cell;
        self.cloud.iter().sum::<f32>() * area
    }

    /// Move water from the wake onto a transverse crest ahead of the hull.
    /// The crest height tracks [`water_bow_height`]. The water sum is unchanged.
    /// The crest is a ridge across the bow, not a blob.
    pub fn raise_water_bow(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        speed: f32,
        still_depth: f32,
    ) -> WaterBow {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let still = still_depth.max(0.0);
        let target_h = water_bow_height(speed, still);
        let half = self.size * 0.5;
        let res = self.res as isize;
        let reach = 2.8_f32;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let bow_near = 0.20_f32;
        let bow_far = 1.65_f32;
        let half_bow = 1.05_f32;
        let wake_far = -2.35_f32;
        let wake_near = -0.20_f32;
        let half_wake = 0.95_f32;
        let mut bow: Vec<(usize, f32)> = Vec::new();
        let mut wake: Vec<usize> = Vec::new();
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                let idx = j as usize * self.res + i as usize;
                if (bow_near..=bow_far).contains(&along)
                    && lat.abs() <= half_bow
                    && self.water[idx] > 0.008
                {
                    let span = (bow_far - bow_near).max(0.05);
                    let u = ((along - bow_near) / span).clamp(0.0, 1.0);
                    let crest = (std::f32::consts::PI * u).sin();
                    let across = (1.0 - (lat.abs() / half_bow).clamp(0.0, 1.0)).powf(0.55);
                    let w = crest * across;
                    if w > 0.04 {
                        bow.push((idx, w));
                    }
                } else if (wake_far..=wake_near).contains(&along)
                    && lat.abs() <= half_wake
                    && self.water[idx] > 0.008
                {
                    wake.push(idx);
                }
            }
        }
        if bow.is_empty() {
            return WaterBow::default();
        }
        let area = self.cell * self.cell;
        let floor = (still * 0.55).max(0.04);
        let mut returned = 0.0_f32;
        for (idx, w) in &bow {
            let want = still + target_h * *w;
            let extra = self.water[*idx] - want;
            if extra > 1.0e-5 {
                self.water[*idx] = want;
                returned += extra;
            }
        }
        if returned > 1.0e-6 && !wake.is_empty() {
            let share = returned / wake.len() as f32;
            for idx in &wake {
                self.water[*idx] += share;
            }
        }
        let mut need = 0.0_f32;
        let mut weight = 0.0_f32;
        for (idx, w) in &bow {
            let want = still + target_h * *w;
            let deficit = (want - self.water[*idx]).max(0.0);
            if deficit > 1.0e-5 {
                need += deficit;
                weight += *w;
            }
        }
        let mut spare = 0.0_f32;
        for idx in &wake {
            spare += (self.water[*idx] - floor).max(0.0);
        }
        let moved = need.min(spare);
        if moved > 1.0e-6 && weight > 1.0e-6 && spare > 1.0e-6 {
            let take_frac = moved / spare;
            for idx in &wake {
                let surplus = (self.water[*idx] - floor).max(0.0);
                self.water[*idx] -= surplus * take_frac;
            }
            for (idx, w) in &bow {
                let want = still + target_h * *w;
                let deficit = (want - self.water[*idx]).max(0.0);
                if deficit <= 1.0e-5 {
                    continue;
                }
                self.water[*idx] += moved * (deficit / need);
            }
        }
        let mut peak = 0.0_f32;
        let mut bow_m3 = 0.0_f32;
        for (idx, w) in &bow {
            let extra = (self.water[*idx] - still).max(0.0);
            peak = peak.max(extra);
            if *w > 0.45 {
                bow_m3 += extra * area;
            }
        }
        let mut wake_m3 = 0.0_f32;
        for idx in &wake {
            wake_m3 += (still - self.water[*idx]).max(0.0) * area;
        }
        WaterBow {
            height: peak,
            bow_m3,
            wake_m3,
        }
    }

    /// Highest free surface on the bow ridge minus the quiet pool, metres.
    pub fn water_bow_crest(&self, x: f32, z: f32, forward: Vec3, quiet: f32) -> f32 {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let half = self.size * 0.5;
        let res = self.res as isize;
        let reach = 2.2_f32;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let mut peak = quiet;
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                if !(0.25..=1.55).contains(&along) || lat.abs() > 0.95 {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                if self.water[idx] < 0.02 {
                    continue;
                }
                let surface = self.bed[idx] + self.placed[idx] + self.water[idx];
                peak = peak.max(surface);
            }
        }
        (peak - quiet).max(0.0)
    }

    /// Mean free surface in the pool, away from a hull at `x,z`.
    pub fn quiet_water_surface(&self, x0: f32, x1: f32, z0: f32, z1: f32, x: f32, z: f32) -> f32 {
        let half = self.size * 0.5;
        let mut sum = 0.0_f32;
        let mut n = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                if cx < x0 || cx >= x1 || cz < z0 || cz >= z1 {
                    continue;
                }
                if (cx - x).hypot(cz - z) < 2.4 {
                    continue;
                }
                let idx = j * self.res + i;
                if self.water[idx] < 0.05 {
                    continue;
                }
                sum += self.bed[idx] + self.placed[idx] + self.water[idx];
                n += 1.0;
            }
        }
        if n < 1.0 {
            0.0
        } else {
            sum / n
        }
    }

    /// Current in the wake scours the bed. Soil that does not stick enters
    /// the cloud. Water volume is unchanged.
    pub fn scour_water_wake(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        speed: f32,
        depth: f32,
        dt: f32,
    ) -> f32 {
        let speed = speed.max(0.0);
        let dt = dt.max(0.0);
        let depth = depth.max(0.0);
        if speed < 0.15 || dt <= 0.0 || depth < 0.02 {
            return 0.0;
        }
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let half = self.size * 0.5;
        let res = self.res as isize;
        let reach = 3.5_f32;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let area = self.cell * self.cell;
        let shear = speed * speed * (depth / 0.30);
        let mut clouded = 0.0_f32;
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                // Behind the rear axle, so the current cuts a rut the tires
                // have already left. Digging under the contact bogs the hull.
                if !(-3.15..=-1.70).contains(&along) {
                    continue;
                }
                let track = (lat.abs() - 0.74).abs();
                if track >= 0.28 && lat.abs() >= 0.22 {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                if self.water[idx] < 0.04 {
                    continue;
                }
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                let pace = water_scour_pace(kind);
                if pace <= 0.0 {
                    continue;
                }
                let leave = (pace * shear * (1.0 - water_stick(kind)) * dt).min(0.006);
                if leave <= 1.0e-6 {
                    continue;
                }
                self.bed[idx] -= leave;
                self.cloud[idx] += leave;
                clouded += leave * area;
                self.sync(idx);
            }
        }
        clouded
    }

    /// Ease the free surface inside the pool toward level. The water sum is
    /// unchanged. Scour holes take water from the higher cells beside them.
    pub fn level_pool_water(&mut self, x0: f32, x1: f32, z0: f32, z1: f32) {
        let res = self.res;
        for _ in 0..3 {
            let water = self.water.clone();
            let mut next = water.clone();
            for j in 0..res {
                for i in 0..res {
                    let a = j * res + i;
                    if !self.in_water_pool(a, x0, x1, z0, z1, &water) {
                        continue;
                    }
                    if i + 1 < res {
                        self.water_level_flux(a, a + 1, x0, x1, z0, z1, &water, &mut next);
                    }
                    if j + 1 < res {
                        self.water_level_flux(a, a + res, x0, x1, z0, z1, &water, &mut next);
                    }
                }
            }
            self.water = next;
        }
    }

    fn in_water_pool(&self, idx: usize, x0: f32, x1: f32, z0: f32, z1: f32, water: &[f32]) -> bool {
        let half = self.size * 0.5;
        let i = idx % self.res;
        let j = idx / self.res;
        let x = (i as f32 + 0.5) * self.cell - half;
        let z = (j as f32 + 0.5) * self.cell - half;
        x >= x0 && x < x1 && z >= z0 && z < z1 && water[idx] > 0.008
    }

    fn water_level_flux(
        &self,
        a: usize,
        b: usize,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        water: &[f32],
        next: &mut [f32],
    ) {
        if !self.in_water_pool(a, x0, x1, z0, z1, water)
            || !self.in_water_pool(b, x0, x1, z0, z1, water)
        {
            return;
        }
        let sa = self.bed[a] + self.placed[a] + water[a];
        let sb = self.bed[b] + self.placed[b] + water[b];
        let dh = sa - sb;
        if dh.abs() < 0.004 {
            return;
        }
        let donor = if dh > 0.0 { a } else { b };
        let recv = if donor == a { b } else { a };
        let head = dh.abs();
        let depth = (head * 0.18).min(water[donor] * 0.35).min(next[donor]);
        if depth <= 1.0e-6 {
            return;
        }
        next[donor] -= depth;
        next[recv] += depth;
    }

    /// Cubic metres of slurry on the field.
    pub fn slurry_volume(&self) -> f32 {
        let area = self.cell * self.cell;
        self.slurry.iter().sum::<f32>() * area
    }

    /// Clear placed spoil inside the pool. Sidewall smear and crest clods
    /// read as orange tablets in a slurry still.
    /// Slurry piled on the rim or on the berm reads as blocks in a still.
    pub fn trim_slurry_pool_rim(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        rim: f32,
    ) {
        let half = self.size * 0.5;
        let rim = rim.max(self.cell);
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                let idx = j * self.res + i;
                if self.slurry[idx] <= 1.0e-6 {
                    continue;
                }
                let inside = x >= x0 && x < x1 && z >= z0 && z < z1;
                let near_edge = inside
                    && (x >= x1 - rim
                        || z >= z1 - rim
                        || x <= x0 + rim
                        || z <= z0 + rim);
                let outside_spill = !inside
                    && x >= x0 - rim
                    && x < x1 + rim
                    && z >= z0 - rim
                    && z < z1 + rim;
                if !near_edge && !outside_spill {
                    continue;
                }
                self.slurry[idx] = 0.0;
                self.sync(idx);
            }
        }
    }

    pub fn clear_placed_in_pool(&mut self, x0: f32, x1: f32, z0: f32, z1: f32) {
        let half = self.size * 0.5;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                if self.placed[idx] <= 1.0e-6 {
                    continue;
                }
                self.placed[idx] = 0.0;
                if self.kind[idx] == SurfaceKind::PlacedClod as u8 {
                    self.kind[idx] = SurfaceKind::WetMud as u8;
                }
                self.sync(idx);
            }
        }
    }

    /// Lower the rut floor on a second pass so the first lip stays visible in
    /// profile. Spoil lands on the berm beside the cut. Slurry column volume
    /// in the pool is unchanged; this only moves bed and placed soil.
    pub fn accentuate_second_pass_rut(
        &mut self,
        pool_x0: f32,
        pool_x1: f32,
        pool_z0: f32,
        pool_z1: f32,
        rear_left: (f32, f32),
        rear_right: (f32, f32),
        track_half: f32,
        forward: Vec3,
        pass1_depth: f32,
    ) {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let cell = self.cell;
        let area = cell * cell;
        let half = self.size * 0.5;
        let rear_z = rear_left.1.min(rear_right.1);
        let wake_floor = pool_z0 + cell * 1.2;
        let wake_top = if rear_z > wake_floor + cell * 2.0 {
            rear_z - 0.08
        } else {
            pool_z1 - cell * 1.4
        };
        if wake_top <= wake_floor + cell {
            return;
        }
        let on_rut = |cx: f32| {
            (cx - rear_left.0).abs() <= track_half + cell * 0.15
                || (cx - rear_right.0).abs() <= track_half + cell * 0.15
        };
        let step = (0.078_f32).max(pass1_depth * 0.32);
        let mut spoil = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz >= pool_z1 {
                    continue;
                }
                if cz > wake_top || cz < wake_floor || !on_rut(cx) {
                    continue;
                }
                let idx = j * self.res + i;
                let cut = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                if cut + 0.002 < pass1_depth * 0.92 {
                    continue;
                }
                let along = ((cz - wake_floor) / (wake_top - wake_floor)).clamp(0.0, 1.0);
                let bite = step * (0.55 + 0.45 * along);
                let removed = self.excavate(idx, bite);
                if removed <= 1.0e-6 {
                    continue;
                }
                spoil += removed * area;
                self.slurry[idx] = (self.slurry[idx] - bite * 0.55).max(0.012);
                self.loose[idx] = (self.loose[idx] + removed * 0.35).min(0.16);
                self.sync(idx);
            }
        }
        // Leave a higher slurry shelf on the first-pass lip beside each rut.
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz > wake_top + cell {
                    continue;
                }
                if on_rut(cx) {
                    continue;
                }
                let lip = (cx - rear_left.0).abs().min((cx - rear_right.0).abs());
                if lip < track_half + cell * 0.4 || lip > track_half + cell * 1.6 {
                    continue;
                }
                let idx = j * self.res + i;
                if self.slurry[idx] < 0.008 {
                    continue;
                }
                self.slurry[idx] = (self.slurry[idx] + step * 0.35).min(0.42);
                self.sync(idx);
            }
        }
        if spoil <= 1.0e-6 {
            return;
        }
        let mut berms: Vec<usize> = Vec::new();
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz > wake_top + cell {
                    continue;
                }
                if on_rut(cx) {
                    continue;
                }
                let rel_x = cx - rear_left.0;
                let rel_z = cz - rear_z;
                let lat = rel_x * right.x + rel_z * right.z;
                if lat.abs() > track_half + cell * 2.8 || lat.abs() < track_half + cell * 0.35 {
                    continue;
                }
                let idx = j * self.res + i;
                berms.push(idx);
            }
        }
        if berms.is_empty() {
            return;
        }
        let share = spoil / berms.len() as f32 / area;
        for idx in berms {
            self.placed[idx] = (self.placed[idx] + share).min(0.28);
            self.kind[idx] = SurfaceKind::Berm as u8;
            self.sync(idx);
        }
    }

    /// Max bed cut under the rear lane behind the vehicle, metres.
    pub fn lane_rut_depth_behind(
        &self,
        pool_x0: f32,
        pool_x1: f32,
        pool_z0: f32,
        rear_z: f32,
        rear_left_x: f32,
        rear_right_x: f32,
        track_half: f32,
    ) -> f32 {
        let half = self.size * 0.5;
        let cell = self.cell;
        let wake_top = rear_z - 0.05;
        let mut best = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz > wake_top {
                    continue;
                }
                let on_rut = (cx - rear_left_x).abs() <= track_half + cell * 0.12
                    || (cx - rear_right_x).abs() <= track_half + cell * 0.12;
                if !on_rut {
                    continue;
                }
                let idx = j * self.res + i;
                best = best.max((self.initial_height[idx] - self.bed[idx]).max(0.0));
            }
        }
        best
    }

    /// Move slurry on the heightfield so a wide still reads a crest ahead of
    /// the front treads and a thinned lane behind the rears. Column volume is
    /// unchanged; bow and wake ledger lines are not touched.
    pub fn accentuate_plow_bow_wake(
        &mut self,
        pool_x0: f32,
        pool_x1: f32,
        pool_z0: f32,
        pool_z1: f32,
        front_hub: (f32, f32),
        rear_left: (f32, f32),
        rear_right: (f32, f32),
        track_half: f32,
        forward: Vec3,
    ) {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let cell = self.cell;
        let area = cell * cell;
        let lip = 0.52_f32;
        let inner_margin = cell * 1.4;
        let half = self.size * 0.5;
        let rear_z = rear_left.1.min(rear_right.1);
        let front_z = front_hub.1;
        let wake_top = rear_z - 0.14;
        let wake_floor = pool_z0 + inner_margin;
        let bow_near = front_z + 0.18;
        let bow_far = (front_z + 0.95).min(pool_z1 - lip);
        if bow_far <= bow_near + 0.04 || wake_top <= wake_floor + 0.04 {
            return;
        }
        let in_pool = |cx: f32, cz: f32| {
            cx >= pool_x0 + inner_margin
                && cx < pool_x1 - inner_margin
                && cz >= pool_z0 + inner_margin
                && cz < pool_z1 - inner_margin
        };
        let on_rut = |cx: f32| {
            (cx - rear_left.0).abs() <= track_half + cell * 0.2
                || (cx - rear_right.0).abs() <= track_half + cell * 0.2
        };
        let near_lip = |cx: f32, cz: f32| {
            cx >= pool_x1 - lip
                || cz >= pool_z1 - lip
                || cx <= pool_x0 + lip
                || cz <= pool_z0 + lip
        };
        let wake_target = 0.024_f32;
        let bow_peak = 0.395_f32;
        let mut moved = 0.0_f32;
        let mut donors: Vec<(usize, f32)> = Vec::new();
        let mut crests: Vec<(usize, f32)> = Vec::new();
        let mut touched: Vec<usize> = Vec::new();
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz >= pool_z1 {
                    continue;
                }
                let idx = j * self.res + i;
                if self.slurry[idx] <= 0.008 {
                    continue;
                }
                let mut take = 0.0_f32;
                if cz <= wake_top && cz >= wake_floor && on_rut(cx) {
                    take = (self.slurry[idx] - wake_target).max(0.0);
                } else if near_lip(cx, cz) {
                    take = self.slurry[idx].max(0.0);
                } else if cz >= rear_z + 0.08 && cz <= front_z - 0.08 && on_rut(cx) {
                    take = self.slurry[idx] * 0.14;
                }
                if take > 1.0e-6 {
                    donors.push((idx, take));
                    moved += take * area;
                    touched.push(idx);
                }
            }
        }
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if !in_pool(cx, cz) || near_lip(cx, cz) {
                    continue;
                }
                let rel_x = cx - front_hub.0;
                let rel_z = cz - front_z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                if !(bow_near..=bow_far).contains(&along) || lat.abs() > track_half + cell * 0.65 {
                    continue;
                }
                let idx = j * self.res + i;
                let span = (bow_far - bow_near).max(0.05);
                let u = ((along - bow_near) / span).clamp(0.0, 1.0);
                let crest = (1.0 - (u - 0.42).abs() / 0.58).max(0.0);
                let across = 1.0 - (lat.abs() / (track_half + cell * 0.5)).clamp(0.0, 1.0);
                let w = crest * crest * across;
                if w > 0.05 {
                    crests.push((idx, w));
                }
            }
        }
        if moved <= 1.0e-6 || crests.is_empty() {
            return;
        }
        for (idx, take) in &donors {
            self.slurry[*idx] -= *take;
        }
        let wsum: f32 = crests.iter().map(|(_, w)| *w).sum();
        for (idx, w) in &crests {
            let add = moved * (*w / wsum) / area;
            self.slurry[*idx] = (self.slurry[*idx] + add).min(bow_peak);
            touched.push(*idx);
        }
        for idx in touched {
            self.sync(idx);
        }
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * cell - half;
                let cz = (j as f32 + 0.5) * cell - half;
                if cx < pool_x0 || cx >= pool_x1 || cz < pool_z0 || cz >= pool_z1 {
                    continue;
                }
                let idx = j * self.res + i;
                if self.slurry[idx] <= 0.008 {
                    continue;
                }
                if cz <= wake_top && cz >= wake_floor && on_rut(cx) {
                    self.darken[idx] = (self.darken[idx] + 0.12).min(1.0);
                } else if cz >= bow_near && cz <= bow_far && on_rut(cx) {
                    self.darken[idx] = (self.darken[idx] - 0.10).max(0.18);
                }
            }
        }
    }

    /// Cut a lane and split that volume into a berm, a throw, and slurry that
    /// stays in the rut. The three add up to the bed that was removed.
    ///
    /// The solid share is the berm plus the throw. That is the #61 identity:
    /// cut minus the slurry left in the rut equals placed spoil plus the
    /// airborne account. With no slurry, and once the throw has landed, the
    /// cut equals the placed volume.
    ///
    /// A dry crust keeps its kind. None of its cut is turned into slurry.
    pub fn cut_lane_in_slurry(
        &mut self,
        from: Vec3,
        to: Vec3,
        half_width: f32,
        depth: f32,
        throw_frac: f32,
    ) -> SlurryCut {
        let from = Vec3::new(from.x, 0.0, from.z);
        let to = Vec3::new(to.x, 0.0, to.z);
        let delta = to - from;
        let len = delta.length();
        if len < 1.0e-3 || depth <= 1.0e-4 || half_width < 0.05 {
            return SlurryCut::default();
        }
        let fwd = delta / len;
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let half = self.size * 0.5;
        let res = self.res;
        let area = self.cell * self.cell;
        let depth = depth.min(0.26);
        let throw_frac = throw_frac.clamp(0.0, 0.72);
        let mid = from.lerp(to, 0.5);
        let sample = self.sample(mid.x, mid.z);
        let kind = sample.kind();
        let moisture = sample.wetness.clamp(0.08, 0.92);
        let mixes = mixes_as_slurry(kind);
        let hold = slurry_hold_fraction(kind, moisture);

        let reach = len * 0.5 + half_width + 1.1;
        let res_i = res as isize;
        let i0 = (((mid.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((mid.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((mid.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((mid.z + reach) + half) / self.cell).floor() as isize;

        let mut cuts: Vec<(usize, f32)> = Vec::new();
        for j in j0.max(1)..j1.min(res_i - 1) {
            for i in i0.max(1)..i1.min(res_i - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - from.x;
                let rel_z = cz - from.z;
                let along = rel_x * fwd.x + rel_z * fwd.z;
                let lat = rel_x * right.x + rel_z * right.z;
                if along < -self.cell * 0.2 || along > len + self.cell * 0.2 || lat.abs() > half_width
                {
                    continue;
                }
                let u = (lat.abs() / half_width).clamp(0.0, 1.0);
                // Flat floor, then a short wall. A linear ramp across the
                // whole width was a saucer, and the still could not see it.
                let profile = if u < 0.70 {
                    1.0
                } else {
                    ((1.0 - u) / 0.30).clamp(0.22, 1.0)
                };
                let drop = depth * profile;
                let idx = j as usize * res + i as usize;
                let before_bed = self.bed[idx];
                let before_placed = self.placed[idx];
                let removed = self.excavate(idx, drop);
                if removed <= 1.0e-6 {
                    continue;
                }
                let bed_drop = (before_bed - self.bed[idx]).max(0.0);
                let placed_taken = (before_placed - self.placed[idx]).max(0.0);
                if placed_taken > 1.0e-8 {
                    // Spoil that was already down is not a new cut.
                    self.placed[idx] += placed_taken;
                }
                if bed_drop <= 1.0e-6 {
                    self.sync(idx);
                    continue;
                }
                if mixes && !firm_cover(SurfaceKind::from_u8(self.kind[idx])) {
                    self.kind[idx] = SurfaceKind::FreshCut as u8;
                    self.darken[idx] = (self.darken[idx] + 0.35).min(1.0);
                    self.wetness[idx] = self.wetness[idx].max(moisture);
                }
                self.sync(idx);
                cuts.push((idx, bed_drop));
            }
        }

        let depth_sum: f32 = cuts.iter().map(|(_, d)| *d).sum();
        let cut = depth_sum * area;
        if cut <= 1.0e-8 || depth_sum <= 1.0e-8 {
            return SlurryCut::default();
        }
        let slurry = if mixes { cut * hold } else { 0.0 };
        let leaving = cut - slurry;
        let thrown = leaving * throw_frac;
        let berm = leaving - thrown;

        if slurry > 1.0e-8 {
            for (idx, d) in &cuts {
                self.slurry[*idx] += (slurry / area) * (*d / depth_sum);
                self.sync(*idx);
            }
        }
        self.lay_lane_berm(&cuts, from, fwd, right, half_width, len, berm);
        if thrown > 1.0e-8 {
            self.airborne += thrown;
        }
        SlurryCut {
            cut,
            berm,
            thrown,
            slurry,
        }
    }

    /// Ridge on the right flank of a lane. The volume is `berm` exactly.
    /// The floor of the cut is left clear so the trench stays open.
    fn lay_lane_berm(
        &mut self,
        cuts: &[(usize, f32)],
        from: Vec3,
        fwd: Vec3,
        right: Vec3,
        half_width: f32,
        len: f32,
        berm: f32,
    ) {
        if berm <= 1.0e-8 {
            return;
        }
        let half = self.size * 0.5;
        let res = self.res;
        let area = self.cell * self.cell;
        let mut cut_at = vec![false; res * res];
        for (idx, _) in cuts {
            cut_at[*idx] = true;
        }
        let mut weight = vec![0.0_f32; res * res];
        let res_i = res as isize;
        let reach = len * 0.5 + half_width + 1.2;
        let mid = from + fwd * (len * 0.5);
        let i0 = (((mid.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((mid.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((mid.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((mid.z + reach) + half) / self.cell).floor() as isize;
        for j in j0.max(1)..j1.min(res_i - 1) {
            for i in i0.max(1)..i1.min(res_i - 1) {
                let idx = j as usize * res + i as usize;
                if cut_at[idx] {
                    continue;
                }
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - from.x;
                let rel_z = cz - from.z;
                let along = rel_x * fwd.x + rel_z * fwd.z;
                let lat = rel_x * right.x + rel_z * right.z;
                if along < -0.05 || along > len + 0.05 {
                    continue;
                }
                let flank = lat - half_width;
                if !(0.02..=0.72).contains(&flank) {
                    continue;
                }
                let w = (-0.5 * ((flank - 0.18) / 0.11).powi(2)).exp();
                if w > 0.08 {
                    weight[idx] = w;
                }
            }
        }
        for _ in 0..5 {
            let wsum: f32 = weight.iter().copied().sum();
            if wsum <= 1.0e-6 {
                break;
            }
            let max_h = weight.iter().copied().fold(0.0_f32, f32::max) * berm / (wsum * area);
            if (0.16..=0.24).contains(&max_h) {
                break;
            }
            if max_h > 0.24 {
                let src = weight.clone();
                for idx in 0..weight.len() {
                    if src[idx] <= 0.0 || cut_at[idx] {
                        weight[idx] = 0.0;
                        continue;
                    }
                    let i = (idx % res) as isize;
                    let j = (idx / res) as isize;
                    let mut acc = src[idx] * 0.45;
                    let mut wacc = 0.45_f32;
                    for (di, dj) in [(1isize, 0), (-1, 0), (0, 1), (0, -1)] {
                        let ii = i + di;
                        let jj = j + dj;
                        if ii < 1 || jj < 1 || ii >= res_i - 1 || jj >= res_i - 1 {
                            continue;
                        }
                        let nidx = jj as usize * res + ii as usize;
                        if cut_at[nidx] {
                            continue;
                        }
                        acc += src[nidx] * 0.14;
                        wacc += 0.14;
                    }
                    weight[idx] = acc / wacc;
                }
            } else {
                for w in &mut weight {
                    if *w > 0.0 {
                        *w = w.powf(1.55);
                    }
                }
            }
        }
        let wsum: f32 = weight.iter().copied().sum();
        if wsum <= 1.0e-6 {
            // No flank on the field. Keep the mass on the lip cells that
            // were cut rather than dropping it.
            let add = berm / (cuts.len().max(1) as f32 * area);
            for (idx, _) in cuts {
                self.placed[*idx] += add;
                self.sync(*idx);
            }
            return;
        }
        let scale = berm / (wsum * area);
        for idx in 0..weight.len() {
            if weight[idx] <= 0.0 {
                continue;
            }
            self.placed[idx] += weight[idx] * scale;
            self.loose[idx] = (self.loose[idx] + self.placed[idx] * 0.15).min(0.18);
            self.sync(idx);
        }
    }

    /// Push slurry sideways. The column is conserved. Clay's reach is about
    /// one cell, so the rut keeps it. Silt moves several cells. A dry crust
    /// is not a destination and is not turned into slurry.
    pub fn slide_slurry(
        &mut self,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        axis: Vec3,
    ) -> SlurrySlide {
        let axis = horizontal(axis);
        let before = self.slurry_volume();
        let (c0, vol0) = self.slurry_centroid_along(axis);
        let src = self.slurry.clone();
        let mut next = src.clone();
        let half = self.size * 0.5;
        let res = self.res;
        let mut moved = 0.0_f32;
        let area = self.cell * self.cell;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * res + i;
                if src[idx] <= 1.0e-5 {
                    continue;
                }
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                if !mixes_as_slurry(kind) {
                    continue;
                }
                let moisture = self.wetness[idx].clamp(0.08, 0.92);
                let reach = slurry_slide_reach(kind, moisture);
                let leave = slurry_leave_fraction(kind, moisture);
                let take = src[idx] * leave;
                if take <= 1.0e-6 || reach < self.cell * 0.5 {
                    continue;
                }
                let steps = ((reach / self.cell).round() as i32).clamp(1, 12);
                let mut dest = idx;
                let mut di = i as isize;
                let mut dj = j as isize;
                let step_x = axis.x.signum() as isize;
                let step_z = axis.z.signum() as isize;
                // Axis is not always grid-aligned. Step along the dominant
                // component, and along the other when the axis is diagonal.
                let ax = axis.x.abs();
                let az = axis.z.abs();
                let mut walked = 0i32;
                while walked < steps {
                    let mut ni = di;
                    let mut nj = dj;
                    if ax >= az * 0.55 {
                        ni += if axis.x >= 0.0 { 1 } else { -1 };
                    }
                    if az > ax * 0.55 {
                        nj += if axis.z >= 0.0 { 1 } else { -1 };
                    }
                    if ni < 1 || nj < 1 || ni >= res as isize - 1 || nj >= res as isize - 1 {
                        break;
                    }
                    let nidx = nj as usize * res + ni as usize;
                    if !self.accepts_slurry(nidx) {
                        break;
                    }
                    di = ni;
                    dj = nj;
                    dest = nidx;
                    walked += 1;
                    let _ = (step_x, step_z);
                }
                if dest == idx {
                    continue;
                }
                next[idx] -= take;
                next[dest] += take;
                moved += take * area;
            }
        }
        // A crust that started dry stays dry. Sliding onto it is refused above.
        for idx in 0..next.len() {
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            if !mixes_as_slurry(kind) && src[idx] <= 1.0e-6 {
                next[idx] = 0.0;
            }
        }
        for idx in 0..next.len() {
            if (next[idx] - self.slurry[idx]).abs() > 0.0 {
                self.slurry[idx] = next[idx].max(0.0);
                self.sync(idx);
            }
        }
        let after = self.slurry_volume();
        let (c1, _) = self.slurry_centroid_along(axis);
        let _ = vol0;
        SlurrySlide {
            moved,
            travel: if before > 1.0e-6 { c1 - c0 } else { 0.0 },
            minted: after - before,
        }
    }

    /// Take a share of the slurry in a rectangle and move it to the thrown
    /// account. The bed is not cut. Returns the cubic metres removed.
    pub fn peel_slurry(&mut self, x0: f32, x1: f32, z0: f32, z1: f32, fraction: f32) -> f32 {
        let fraction = fraction.clamp(0.0, 0.85);
        if fraction <= 1.0e-6 {
            return 0.0;
        }
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let mut vol = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                if !mixes_as_slurry(kind) || self.slurry[idx] <= 1.0e-6 {
                    continue;
                }
                let take = self.slurry[idx] * fraction;
                self.slurry[idx] -= take;
                vol += take * area;
                self.sync(idx);
            }
        }
        if vol > 0.0 {
            self.airborne += vol;
        }
        vol
    }

    fn accepts_slurry(&self, idx: usize) -> bool {
        let kind = SurfaceKind::from_u8(self.kind[idx]);
        mixes_as_slurry(kind)
    }

    fn slurry_centroid_along(&self, axis: Vec3) -> (f32, f32) {
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let mut acc = 0.0_f32;
        let mut vol = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let idx = j * self.res + i;
                let h = self.slurry[idx];
                if h <= 1.0e-6 {
                    continue;
                }
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                let v = h * area;
                acc += (x * axis.x + z * axis.z) * v;
                vol += v;
            }
        }
        if vol <= 1.0e-8 {
            (0.0, 0.0)
        } else {
            (acc / vol, vol)
        }
    }

    /// Lift a tire through slurry. While the seal holds, the step adds a
    /// downward suction impulse. At [`SLURRY_BREAK_LIFT`] the seal snaps and
    /// a share of the slurry under the tread is thrown. That volume leaves
    /// the slurry column and is added to [`SoilField::airborne`].
    pub fn pull_from_slurry(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
        lift_speed: f32,
        dt: f32,
        seal: &mut SlurrySeal,
    ) -> PullOut {
        if seal.broken || dt <= 0.0 || lift_speed <= 1.0e-4 {
            return PullOut {
                broken: seal.broken,
                ..PullOut::default()
            };
        }
        let sample = self.sample(x, z);
        if sample.slurry < 0.025 && seal.lift <= 1.0e-4 {
            return PullOut::default();
        }
        if seal.lift >= SLURRY_BREAK_LIFT - 1.0e-5 {
            let thrown = self.release_slurry(x, z, forward, radius, width);
            seal.broken = true;
            return PullOut {
                force: 0.0,
                impulse: 0.0,
                broken: true,
                thrown,
            };
        }
        let room = SLURRY_BREAK_LIFT - seal.lift;
        let step = (lift_speed * dt).min(room.max(0.0));
        let dt_used = step / lift_speed;
        let force = slurry_suction_force(
            sample.kind(),
            sample.wetness,
            sample.slurry,
            radius,
            width,
            lift_speed,
            seal.lift,
        );
        let impulse = force * dt_used;
        seal.lift += step;
        seal.impulse += impulse;
        let mut thrown = 0.0;
        let mut broken = false;
        if seal.lift >= SLURRY_BREAK_LIFT - 1.0e-5 {
            thrown = self.release_slurry(x, z, forward, radius, width);
            seal.broken = true;
            broken = true;
        }
        PullOut {
            force,
            impulse,
            broken,
            thrown,
        }
    }

    /// Take a share of the slurry under a tire and move it to the thrown account.
    /// The bed is not cut. Returns the cubic metres removed.
    fn release_slurry(&mut self, x: f32, z: f32, forward: Vec3, radius: f32, width: f32) -> f32 {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let radius = radius.max(0.15);
        let half_w = width * 0.72 + self.cell * 0.35;
        let back = -radius * 0.20;
        let front = radius * 0.95;
        let reach = radius + half_w + self.cell;
        let half = self.size * 0.5;
        let res = self.res as isize;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let mut taken: Vec<(usize, f32)> = Vec::new();
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                if !(back..=front).contains(&along) || lat.abs() > half_w {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let take = self.slurry[idx] * 0.55;
                if take > 1.0e-6 {
                    taken.push((idx, take));
                }
            }
        }
        let area = self.cell * self.cell;
        let mut vol = 0.0_f32;
        for (idx, take) in &taken {
            self.slurry[*idx] -= *take;
            vol += *take * area;
            self.sync(*idx);
        }
        if vol > 0.0 {
            self.airborne += vol;
        }
        vol
    }

    /// Slurry that clings in the tread when the tire is pulled clear.
    ///
    /// `room` is the cubic metres the voids can still take. The rut keeps
    /// most of the column. Nothing here is thrown, and the bed is not cut.
    /// Rolling through the slurry does not use this path.
    pub fn pack_slurry_into_grooves(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
        room: f32,
    ) -> f32 {
        let room = room.max(0.0);
        if room <= 1.0e-8 || !room.is_finite() {
            return 0.0;
        }
        let cells = self.slurry_under_tread(x, z, forward, radius, width);
        let area = self.cell * self.cell;
        let available: f32 = cells.iter().map(|idx| self.slurry[*idx] * area).sum();
        if available <= 1.0e-8 {
            return 0.0;
        }
        // The voids take what they can hold, and no more than two fifths of
        // the slurry under the tread.
        let want = room.min(available * 0.40);
        let frac = (want / available).clamp(0.0, 1.0);
        let mut vol = 0.0_f32;
        for idx in cells {
            let take = self.slurry[idx] * frac;
            self.slurry[idx] -= take;
            vol += take * area;
            self.sync(idx);
        }
        vol
    }

    /// Put thrown groove slurry back into the nearest rut. The bed is not cut.
    /// Returns the surface point the volume landed on.
    pub fn return_slurry_to_rut(&mut self, x: f32, z: f32, volume: f32) -> Option<Vec3> {
        let take = volume.min(self.airborne).max(0.0);
        if take <= 1.0e-8 || !take.is_finite() {
            return None;
        }
        let idx = self.nearest_rut_idx(x, z)?;
        self.airborne -= take;
        self.slurry[idx] += take / (self.cell * self.cell);
        self.sync(idx);
        let half = self.size * 0.5;
        let i = idx % self.res;
        let j = idx / self.res;
        Some(Vec3::new(
            (i as f32 + 0.5) * self.cell - half,
            self.height[idx],
            (j as f32 + 0.5) * self.cell - half,
        ))
    }

    fn slurry_under_tread(
        &self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
    ) -> Vec<usize> {
        let forward = horizontal(forward);
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let radius = radius.max(0.15);
        let half_w = width * 0.72 + self.cell * 0.35;
        let back = -radius * 0.20;
        let front = radius * 0.95;
        let reach = radius + half_w + self.cell;
        let half = self.size * 0.5;
        let res = self.res as isize;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let mut cells = Vec::new();
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - x;
                let rel_z = cz - z;
                let lat = rel_x * right.x + rel_z * right.z;
                let along = rel_x * forward.x + rel_z * forward.z;
                if !(back..=front).contains(&along) || lat.abs() > half_w {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                if self.slurry[idx] > 1.0e-5 {
                    cells.push(idx);
                }
            }
        }
        cells
    }

    fn nearest_rut_idx(&self, x: f32, z: f32) -> Option<usize> {
        let start = self.idx_at(x, z)?;
        if self.in_rut_or_hole(start) {
            return Some(start);
        }
        let res = self.res as isize;
        let i0 = (start % self.res) as isize;
        let j0 = (start / self.res) as isize;
        let half = self.size * 0.5;
        let mut best: Option<(usize, f32)> = None;
        for rad in 1isize..=16 {
            for dj in -rad..=rad {
                for di in -rad..=rad {
                    if di.abs() != rad && dj.abs() != rad {
                        continue;
                    }
                    let i = i0 + di;
                    let j = j0 + dj;
                    if i < 0 || j < 0 || i >= res || j >= res {
                        continue;
                    }
                    let idx = j as usize * self.res + i as usize;
                    if !self.in_rut_or_hole(idx) {
                        continue;
                    }
                    let cx = (i as f32 + 0.5) * self.cell - half;
                    let cz = (j as f32 + 0.5) * self.cell - half;
                    let d = (cx - x).hypot(cz - z);
                    if best.map(|b| d < b.1).unwrap_or(true) {
                        best = Some((idx, d));
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best.map(|b| b.0)
    }

    /// A tire rut is a bed below the surface it started at. A hole is a
    /// closed bowl or a groove with higher ground on both sides. A flat
    /// sheet of water is neither.
    fn in_rut_or_hole(&self, idx: usize) -> bool {
        if self.initial_height[idx] - self.bed[idx] > 0.012 {
            return true;
        }
        let res = self.res as isize;
        let i = (idx % self.res) as isize;
        let j = (idx / self.res) as isize;
        if i <= 0 || j <= 0 || i + 1 >= res || j + 1 >= res {
            return false;
        }
        let h = self.bed[idx] + self.placed[idx];
        let at = |di: isize, dj: isize| -> f32 {
            let ii = i + di;
            let jj = j + dj;
            if ii < 0 || jj < 0 || ii >= res || jj >= res {
                return h;
            }
            let n = jj as usize * self.res + ii as usize;
            self.bed[n] + self.placed[n]
        };
        let side = 0.02;
        let groove = (at(-1, 0) > h + side && at(1, 0) > h + side)
            || (at(0, -1) > h + side && at(0, 1) > h + side);
        if groove {
            return true;
        }
        let mut min_barrier = f32::MAX;
        for (di, dj) in [(1isize, 0), (-1, 0), (0, 1), (0, -1)] {
            let mut barrier = h;
            for s in 1..=28 {
                barrier = barrier.max(at(di * s, dj * s));
            }
            min_barrier = min_barrier.min(barrier);
        }
        min_barrier > h + side
    }

    /// Solid, free water, and slurry, in cubic metres. Their sum is the
    /// conserved water-plus-soil volume. Slurry is the joined body, not a
    /// column sitting on unchanged clay.
    pub fn mix_volumes(&self) -> MixVolumes {
        let area = self.cell * self.cell;
        let mut soil = 0.0_f32;
        let mut water = 0.0_f32;
        let mut slurry = 0.0_f32;
        for idx in 0..self.bed.len() {
            soil += self.bed[idx] + self.placed[idx] + self.cloud[idx];
            water += self.water[idx];
            slurry += self.slurry[idx];
        }
        MixVolumes {
            soil: soil * area,
            water: water * area,
            slurry: slurry * area,
        }
    }

    /// Skin and fissure volumes. Thickness is the closed book: what grew,
    /// minus what soaked or faded, is what is still on the field. Opening is
    /// the fissure volume. Neither is the solid column.
    pub fn crust_book(&self) -> CrustBook {
        let area = f64::from(self.cell) * f64::from(self.cell);
        let mut thickness = 0.0f64;
        let mut opening = 0.0f64;
        for idx in 0..self.crust.len() {
            let t = self.crust[idx];
            let o = self.crack[idx];
            if t.is_finite() && t > 0.0 {
                thickness += f64::from(t);
            }
            if o.is_finite() && o > 0.0 {
                opening += f64::from(o);
            }
        }
        CrustBook {
            thickness_m3: thickness * area,
            opening_m3: opening * area,
            grown_m3: self.skin_grown_m3,
            lost_m3: self.skin_lost_m3,
        }
    }

    /// Displace puddle water out of a wheel footprint. Returns displaced depth scale.
    pub fn displace_water(&mut self, x: f32, z: f32, forward: Vec3, amount: f32) {
        let Some(idx) = self.idx_at(x, z) else {
            return;
        };
        let take = amount.min(self.water[idx] * 0.45).max(0.0);
        if take <= 0.0 {
            return;
        }
        self.water[idx] -= take;
        // Push ahead of the tire and a little to both sides.
        let spots = [
            (
                x + forward.x * self.cell * 1.4,
                z + forward.z * self.cell * 1.4,
                0.45,
            ),
            (x + forward.z * self.cell, z - forward.x * self.cell, 0.275),
            (x - forward.z * self.cell, z + forward.x * self.cell, 0.275),
        ];
        for (sx, sz, share) in spots {
            if let Some(sidx) = self.idx_at(sx, sz) {
                self.water[sidx] = (self.water[sidx] + take * share).min(0.32);
                self.wetness[sidx] = (self.wetness[sidx] + 0.04).min(1.0);
            }
        }
    }

    /// Virtual-pipe shallow water, pore moisture, then slurry flow.
    /// `rain` is metres per second landing on the field. The bed is not cut.
    /// A berm moves only when it creeps into slurry that is already there.
    pub fn step_water(&mut self, dt: f32) {
        self.step_surface(dt, 0.0);
    }

    /// One weather step. Infiltration and the pipe move standing water.
    /// [`step_moisture`](Self::step_moisture) dries clay toward a crust and
    /// wets that crust again when rain or a pond is on it. Slurry that is
    /// already mixed then flows downhill, and a berm beside it creeps in.
    /// The bed is not cut here.
    pub fn step_surface(&mut self, dt: f32, rain: f32) {
        // Time away from a cell is what makes the next visit a new pass.
        self.age_passes(dt);
        self.infiltrate(dt);
        // When any cell still holds mobile water, dry neighbours still pipe.
        // A field with none skips the clone and the edge loop.
        if self.water.iter().any(|w| *w > 0.0) {
            let res = self.res;
            let dx = self.cell;
            let area = dx * dx;
            let mut next = self.water.clone();
            let cond = 1.35; // m/s per metre of head, scaled below by dt

            for j in 0..res {
                for i in 0..res {
                    let a = j * res + i;
                    if i + 1 < res {
                        let b = a + 1;
                        transfer(
                            &self.height,
                            &self.water,
                            &self.kind,
                            &mut next,
                            a,
                            b,
                            dt,
                            dx,
                            area,
                            cond,
                        );
                    }
                    if j + 1 < res {
                        let b = a + res;
                        transfer(
                            &self.height,
                            &self.water,
                            &self.kind,
                            &mut next,
                            a,
                            b,
                            dt,
                            dx,
                            area,
                            cond,
                        );
                    }
                }
            }
            for w in &mut next {
                *w = w.clamp(0.0, 0.34);
            }
            self.water = next;
        }
        self.step_moisture(dt, rain);
        self.step_slurry(dt);
    }

    /// Advance weather in quarter-second bites so a long dry spell stays stable.
    pub fn advance_weather(&mut self, seconds: f32, rain: f32) {
        if seconds <= 0.0 {
            return;
        }
        let steps = (seconds / 0.25).ceil().max(1.0) as u32;
        let dt = seconds / steps as f32;
        for _ in 0..steps {
            self.step_surface(dt, rain);
        }
    }

    /// Gravel and sand lose standing water downward. Silt and clay do not.
    fn infiltrate(&mut self, dt: f32) {
        let n = self.water.len();
        for idx in 0..n {
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            let rate = match kind {
                SurfaceKind::Gravel => 0.09,
                SurfaceKind::Sand => 0.028,
                _ => 0.0,
            };
            if rate <= 0.0 || self.water[idx] <= 0.0 {
                continue;
            }
            let lost = (rate * dt).min(self.water[idx]);
            self.water[idx] -= lost;
            if kind == SurfaceKind::Sand {
                self.wetness[idx] = (self.wetness[idx] + lost * 2.5).min(0.48);
            } else {
                self.wetness[idx] = (self.wetness[idx] - lost * 3.0).max(0.02);
            }
        }
    }

    /// Pore moisture over time. Wet clay dries toward a crust. Rain or
    /// standing water turns that crust back into wet clay. Gravel stays a
    /// draining gravel. Sand and silt keep their kinds and follow their own
    /// wetness. The cut and the placed pile are not touched.
    pub fn step_moisture(&mut self, dt: f32, rain: f32) {
        let dt = dt.max(0.0);
        let rain = rain.max(0.0);
        let n = self.wetness.len();
        for idx in 0..n {
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            if rain > 0.0 {
                let catch = match kind {
                    SurfaceKind::Gravel => 0.2,
                    SurfaceKind::Sand => 0.5,
                    SurfaceKind::DryCrust => 0.75,
                    SurfaceKind::Silt => 0.85,
                    // Wet clay and a fresh cut drink what lands on them.
                    _ => 1.0,
                };
                self.water[idx] = (self.water[idx] + rain * catch * dt).min(0.34);
            }
            let hold = water_hold(kind);
            let standing = self.water[idx] > hold + 0.004;
            let wetting = standing || rain > 1.0e-6;
            let mix = self.mix_texel(idx);
            let (target, rate) = moisture_step_target(kind, self.base_wetness[idx], wetting, mix);
            let a = 1.0 - (-rate * dt).exp();
            self.wetness[idx] =
                (self.wetness[idx] + (target - self.wetness[idx]) * a).clamp(0.0, 1.0);
            if rain <= 0.0 && self.water[idx] > hold && !matches!(kind, SurfaceKind::Gravel | SurfaceKind::Sand)
            {
                let depth = self.water[idx] - hold;
                // A film dries off. A deep pond only loses a skin, so a lap
                // does not empty the infield.
                let skin = if depth < 0.05 { 1.0 } else { 0.12 };
                let exposed = depth.min(0.04) / 0.04;
                let evap = film_evap(kind) * skin * exposed * dt;
                self.water[idx] = (self.water[idx] - evap).max(0.0);
            }
            let next = evolve_kind(kind, self.wetness[idx], self.water[idx]);
            if next != kind {
                self.kind[idx] = next as u8;
            }
            if matches!(
                next,
                SurfaceKind::DryCrust | SurfaceKind::WetMud | SurfaceKind::Sand | SurfaceKind::Silt
            ) {
                let goal = lerp(0.94, 0.28, self.wetness[idx]);
                let blend = 1.0 - (-0.35 * dt).exp();
                self.roughness[idx] = lerp(self.roughness[idx], goal, blend);
            }
            self.bake_crust(idx, next, dt, rain);
        }
    }

    /// Grow or soak the skin on one cell. The bed and the placed pile stay.
    fn bake_crust(&mut self, idx: usize, kind: SurfaceKind, dt: f32, rain: f32) {
        let dt = dt.max(0.0);
        if dt <= 0.0 {
            return;
        }
        let mut thickness = self.crust[idx];
        let mut opening = self.crack[idx];
        let pond = rain > 1.0e-6 || self.water[idx] > water_hold(kind) + 0.012;
        if !crustable(kind) || pond {
            let soak = if pond { CRUST_SOAK } else { CRUST_SOAK * 0.35 };
            thickness = (thickness - soak * dt).max(0.0);
            opening = (opening - CRACK_CLOSE * dt).max(0.0);
            self.commit_skin(idx, thickness, opening);
            return;
        }
        if self.wetness[idx] <= REWET_AT {
            let deficit = ((REWET_AT - self.wetness[idx]) / REWET_AT).clamp(0.0, 1.0);
            let track = if self.passes[idx] > 0 { 1.35 } else { 1.0 };
            let room = (CRUST_MAX_M - thickness).max(0.0);
            let grow = (CRUST_GROW * deficit * track * dt).min(room);
            thickness += grow;
        } else {
            thickness = (thickness - CRUST_SOAK * 0.15 * dt).max(0.0);
        }
        opening = self.next_crack(idx, thickness, opening, dt);
        self.commit_skin(idx, thickness, opening);
    }

    fn next_crack(&self, idx: usize, thickness: f32, opening: f32, dt: f32) -> f32 {
        if thickness <= CRACK_ONSET_M {
            return (opening - CRACK_CLOSE * dt).max(0.0);
        }
        let (x, z) = self.cell_xz(idx);
        let mask = crack_mask(x, z);
        if mask < 0.12 {
            return (opening - CRACK_CLOSE * 0.25 * dt).max(0.0);
        }
        let skin = ((thickness - CRACK_ONSET_M) / (CRUST_MAX_M - CRACK_ONSET_M)).clamp(0.0, 1.0);
        let plate = if self.passes[idx] > 0 {
            let age = ((self.rest[idx] - 0.40) / 12.0).clamp(0.0, 1.0);
            0.55 + 0.45 * age
        } else {
            0.40
        };
        let target = CRACK_MAX_M * skin * mask * plate;
        let a = 1.0 - (-0.09 * dt).exp();
        lerp(opening, target, a).clamp(0.0, CRACK_MAX_M)
    }

    /// Store the skin. Thickness stays at or under [`CRUST_CAP_M`]. Opening
    /// does not exceed thickness. The cubic metres go into the crust book.
    fn commit_skin(&mut self, idx: usize, thickness: f32, opening: f32) {
        let thickness = if thickness.is_finite() {
            thickness.clamp(0.0, CRUST_CAP_M)
        } else {
            0.0
        };
        let opening = if opening.is_finite() {
            opening.max(0.0).min(thickness)
        } else {
            0.0
        };
        let prev = self.crust[idx];
        let prev = if prev.is_finite() && prev > 0.0 { prev } else { 0.0 };
        let area = f64::from(self.cell) * f64::from(self.cell);
        let delta = f64::from(thickness) - f64::from(prev);
        if delta > 0.0 {
            self.skin_grown_m3 += delta * area;
        } else if delta < 0.0 {
            self.skin_lost_m3 += -delta * area;
        }
        self.crust[idx] = thickness;
        self.crack[idx] = opening;
    }

    fn cell_xz(&self, idx: usize) -> (f32, f32) {
        let half = self.size * 0.5;
        let i = idx % self.res;
        let j = idx / self.res;
        let x = (i as f32 + 0.5) * self.cell - half;
        let z = (j as f32 + 0.5) * self.cell - half;
        (x, z)
    }

    /// Body-sized crater. The displaced soil is piled on the rim. Returns the
    /// deepest cut this stamp wrote into the heightfield.
    pub fn stamp_impact(&mut self, x: f32, z: f32, radius: f32, depth: f32) -> f32 {
        let radius = radius.clamp(0.5, 2.2);
        let depth = depth.clamp(0.04, 0.24);
        let half = self.size * 0.5;
        let res = self.res as isize;
        let reach = radius * 1.6;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let area = self.cell * self.cell;
        let mut vol = 0.0_f32;
        let mut dug_max = 0.0_f32;
        let mut ring_n = 0.0_f32;
        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let d = ((cx - x).powi(2) + (cz - z).powi(2)).sqrt() / radius;
                let idx = j as usize * self.res + i as usize;
                if d < 1.0 {
                    let w = (1.0 - d).powf(1.35);
                    let drop = depth * w;
                    let removed = self.excavate(idx, drop);
                    self.loose[idx] = (self.loose[idx] + removed * 0.55).min(0.2);
                    self.wetness[idx] = self.wetness[idx].max(0.9);
                    self.darken[idx] = (self.darken[idx] + 0.35 * w).min(1.0);
                    self.kind[idx] = if d > 0.82 {
                        SurfaceKind::CutLip as u8
                    } else if self.darken[idx] > 0.72 {
                        SurfaceKind::PackedClay as u8
                    } else {
                        SurfaceKind::FreshCut as u8
                    };
                    vol += removed * area;
                    dug_max = dug_max.max(removed);
                } else if d < 1.5 {
                    ring_n += 1.0;
                }
            }
        }
        if ring_n > 0.0 && vol > 0.0 {
            let mut weights = Vec::new();
            for j in j0.max(1)..j1.min(res - 1) {
                for i in i0.max(1)..i1.min(res - 1) {
                    let cx = (i as f32 + 0.5) * self.cell - half;
                    let cz = (j as f32 + 0.5) * self.cell - half;
                    let d = ((cx - x).powi(2) + (cz - z).powi(2)).sqrt() / radius;
                    if (1.0..1.5).contains(&d) {
                        let idx = j as usize * self.res + i as usize;
                        let lip = 1.0 - (d - 1.0) / 0.5;
                        weights.push((idx, lip));
                    }
                }
            }
            let wsum: f32 = weights.iter().map(|(_, w)| *w).sum();
            if wsum > 1.0e-5 {
                for (idx, w) in weights {
                    let add = vol * (w / wsum) / area;
                    self.placed[idx] += add;
                    self.kind[idx] = SurfaceKind::Berm as u8;
                    self.loose[idx] = (self.loose[idx] + add * 0.3).min(0.18);
                    self.sync(idx);
                }
            }
        }
        dug_max
    }

    /// Push standing water out of the impact into a ring. The pipe step then
    /// carries that ring. The center keeps most of its water, so the pond
    /// does not open a hole.
    pub fn shock_ring(&mut self, x: f32, z: f32) {
        let half = self.size * 0.5;
        let res = self.res as isize;
        let reach = 2.3_f32;
        let i0 = (((x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((z + reach) + half) / self.cell).floor() as isize;
        let mut taken = 0.0_f32;
        let mut ring_n = 0.0_f32;
        for j in j0.max(0)..j1.min(res - 1) {
            for i in i0.max(0)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let d = ((cx - x).powi(2) + (cz - z).powi(2)).sqrt();
                let idx = j as usize * self.res + i as usize;
                if d < 0.7 {
                    let take = (self.water[idx] * 0.28).min(0.05);
                    self.water[idx] -= take;
                    taken += take;
                } else if (0.85..2.05).contains(&d) {
                    ring_n += 1.0;
                }
            }
        }
        if taken <= 0.0 || ring_n <= 0.0 {
            return;
        }
        let share = taken / ring_n;
        for j in j0.max(0)..j1.min(res - 1) {
            for i in i0.max(0)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let d = ((cx - x).powi(2) + (cz - z).powi(2)).sqrt();
                if (0.85..2.05).contains(&d) {
                    let idx = j as usize * self.res + i as usize;
                    self.water[idx] = (self.water[idx] + share).min(0.34);
                    self.wetness[idx] = (self.wetness[idx] + 0.25).min(1.0);
                }
            }
        }
    }

    pub fn max_rut_depth(&self) -> f32 {
        let mut m = 0.0_f32;
        for (h0, bed) in self.initial_height.iter().zip(self.bed.iter()) {
            m = m.max(h0 - bed);
        }
        m
    }

    /// Deepest cut in a small window. A sample that lands on the berm
    /// would under-report the floor.
    pub fn rut_near(&self, x: f32, z: f32) -> f32 {
        let mut m = 0.0_f32;
        for dz in [-1, 0, 1] {
            for dx in [-1, 0, 1] {
                let s = self.cell * 0.45;
                m = m.max(self.sample(x + dx as f32 * s, z + dz as f32 * s).rut);
            }
        }
        m
    }

    /// Advance the pass clock without moving water, the bed, or the berm.
    pub fn age_passes(&mut self, seconds: f32) {
        if seconds <= 0.0 {
            return;
        }
        for rest in &mut self.rest {
            *rest = (*rest + seconds).min(30.0);
        }
    }

    /// Bytes for the cut, the berm, and the pass clock. Height is rebuilt
    /// from the bed and the placed pile on load.
    pub fn save_state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16 + self.bed.len() * 46);
        out.extend_from_slice(b"RUT1");
        out.extend_from_slice(&(self.res as u32).to_le_bytes());
        out.extend_from_slice(&self.size.to_le_bytes());
        push_f32s(&mut out, &self.initial_height);
        push_f32s(&mut out, &self.bed);
        push_f32s(&mut out, &self.placed);
        push_f32s(&mut out, &self.loose);
        push_f32s(&mut out, &self.wetness);
        push_f32s(&mut out, &self.base_wetness);
        push_f32s(&mut out, &self.water);
        push_f32s(&mut out, &self.darken);
        push_f32s(&mut out, &self.roughness);
        push_f32s(&mut out, &self.rest);
        out.extend_from_slice(&self.kind);
        out.extend_from_slice(&self.passes);
        // The tread negative rides with the rut. Older snapshots stop here.
        push_f32s(&mut out, &self.tread_amp);
        push_f32s(&mut out, &self.tread_ang);
        push_f32s(&mut out, &self.tread_lat);
        push_f32s(&mut out, &self.slurry);
        push_f32s(&mut out, &self.crust);
        push_f32s(&mut out, &self.crack);
        push_f32s(&mut out, &self.mix);
        out
    }

    /// Restore a field written by [`save_state`](Self::save_state).
    pub fn load_state(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"RUT1" {
            return Err("not a soil snapshot".into());
        }
        let res = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let size = f32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if res == 0 || res > 1024 || !size.is_finite() || size <= 0.0 {
            return Err("soil snapshot has a bad grid".into());
        }
        let n = res * res;
        let mut i = 12usize;
        let initial_height = read_f32s(bytes, &mut i, n)?;
        let bed = read_f32s(bytes, &mut i, n)?;
        let placed = read_f32s(bytes, &mut i, n)?;
        let loose = read_f32s(bytes, &mut i, n)?;
        let wetness = read_f32s(bytes, &mut i, n)?;
        let base_wetness = read_f32s(bytes, &mut i, n)?;
        let water = read_f32s(bytes, &mut i, n)?;
        let darken = read_f32s(bytes, &mut i, n)?;
        let roughness = read_f32s(bytes, &mut i, n)?;
        let rest = read_f32s(bytes, &mut i, n)?;
        let kind = read_u8s(bytes, &mut i, n)?;
        let passes = read_u8s(bytes, &mut i, n)?;
        let remain = bytes.len() - i;
        let (tread_amp, tread_ang, tread_lat, slurry, crust, crack, mix) = if remain == 0 {
            (
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * MIX_CHANNELS],
            )
        } else if remain == n * 12 {
            let tread_amp = read_f32s(bytes, &mut i, n)?;
            let tread_ang = read_f32s(bytes, &mut i, n)?;
            let tread_lat = read_f32s(bytes, &mut i, n)?;
            (
                tread_amp,
                tread_ang,
                tread_lat,
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * MIX_CHANNELS],
            )
        } else if remain == n * 16 {
            let tread_amp = read_f32s(bytes, &mut i, n)?;
            let tread_ang = read_f32s(bytes, &mut i, n)?;
            let tread_lat = read_f32s(bytes, &mut i, n)?;
            let slurry = read_f32s(bytes, &mut i, n)?;
            (
                tread_amp,
                tread_ang,
                tread_lat,
                slurry,
                vec![0.0; n],
                vec![0.0; n],
                vec![0.0; n * MIX_CHANNELS],
            )
        } else if remain == n * 24 {
            let tread_amp = read_f32s(bytes, &mut i, n)?;
            let tread_ang = read_f32s(bytes, &mut i, n)?;
            let tread_lat = read_f32s(bytes, &mut i, n)?;
            let slurry = read_f32s(bytes, &mut i, n)?;
            let crust = read_f32s(bytes, &mut i, n)?;
            let crack = read_f32s(bytes, &mut i, n)?;
            (
                tread_amp,
                tread_ang,
                tread_lat,
                slurry,
                crust,
                crack,
                vec![0.0; n * MIX_CHANNELS],
            )
        } else if remain == n * 40 {
            let tread_amp = read_f32s(bytes, &mut i, n)?;
            let tread_ang = read_f32s(bytes, &mut i, n)?;
            let tread_lat = read_f32s(bytes, &mut i, n)?;
            let slurry = read_f32s(bytes, &mut i, n)?;
            let crust = read_f32s(bytes, &mut i, n)?;
            let crack = read_f32s(bytes, &mut i, n)?;
            let mix = read_f32s(bytes, &mut i, n * MIX_CHANNELS)?;
            (tread_amp, tread_ang, tread_lat, slurry, crust, crack, mix)
        } else {
            return Err("soil snapshot has trailing bytes".into());
        };
        let mix_ready = mix.iter().any(|v| v.abs() > 1.0e-8);
        let mut height = vec![0.0; n];
        for idx in 0..n {
            height[idx] = bed[idx] + placed[idx] + slurry[idx];
        }
        let cell = size / res as f32;
        // The snapshot has no ledger. Skin already on the field is grown.
        let area = f64::from(cell) * f64::from(cell);
        let mut skin_grown_m3 = 0.0f64;
        for t in &crust {
            if t.is_finite() && *t > 0.0 {
                skin_grown_m3 += f64::from(*t);
            }
        }
        skin_grown_m3 *= area;
        Ok(Self {
            res,
            size,
            cell,
            height,
            initial_height,
            loose,
            wetness,
            base_wetness,
            water,
            darken,
            roughness,
            bed,
            placed,
            kind,
            rest,
            passes,
            tread_amp,
            tread_ang,
            tread_lat,
            airborne: 0.0,
            tread_profile: false,
            slurry,
            cloud: vec![0.0; n],
            crust,
            crack,
            skin_grown_m3,
            skin_lost_m3: 0.0,
            mix,
            mix_ready,
        })
    }

    /// Press a negative of the tread into the cells under the contact.
    /// The bed is not cut. The amplitude stays until a later tire overwrites it.
    pub fn imprint_tread(
        &mut self,
        center: Vec3,
        forward: Vec3,
        right: Vec3,
        radius: f32,
        width: f32,
        sink: f32,
    ) {
        if sink < 0.008 || !sink.is_finite() {
            return;
        }
        let fwd = Vec3::new(forward.x, 0.0, forward.z);
        let fwd = if fwd.length_squared() < 1.0e-8 {
            Vec3::Z
        } else {
            fwd.normalize()
        };
        let rt = Vec3::new(right.x, 0.0, right.z);
        let rt = if rt.length_squared() < 1.0e-8 {
            Vec3::new(fwd.z, 0.0, -fwd.x)
        } else {
            rt.normalize()
        };
        let amp = (sink * 0.9).clamp(0.026, 0.048);
        let ang = fwd.x.atan2(fwd.z);
        let chord = contact_length(radius, sink).max(radius * 0.35) * 0.5 + self.cell * 0.45;
        let half_w = width * 0.5 + self.cell * 0.25;
        let half = self.size * 0.5;
        let reach = chord + half_w;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;
        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let along = rel_x * fwd.x + rel_z * fwd.z;
                let lat = rel_x * rt.x + rel_z * rt.z;
                if along.abs() > chord || lat.abs() > half_w {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                self.tread_amp[idx] = amp;
                self.tread_ang[idx] = ang;
                self.tread_lat[idx] = lat;
            }
        }
    }

    /// Spread tall placed lumps inside a circle so the berm reads as a ridge,
    /// not orange tablets or spikes. Total placed volume in the circle is
    /// unchanged.
    pub fn spread_placed_lumps(&mut self, cx: f32, cz: f32, radius: f32, cap: f32) {
        let cap = cap.max(0.02);
        let half = self.size * 0.5;
        let area = self.cell * self.cell;
        let res = self.res as isize;
        let r0 = radius + self.cell;
        let i0 = (((cx - r0) + half) / self.cell).floor() as isize;
        let i1 = (((cx + r0) + half) / self.cell).floor() as isize;
        let j0 = (((cz - r0) + half) / self.cell).floor() as isize;
        let j1 = (((cz + r0) + half) / self.cell).floor() as isize;
        let mut spill = 0.0_f32;
        let mut donors: Vec<usize> = Vec::new();
        for j in j0.max(0)..=j1.min(res - 1) {
            for i in i0.max(0)..=i1.min(res - 1) {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if (x - cx).hypot(z - cz) > radius {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                if self.placed[idx] <= cap + 1.0e-6 {
                    continue;
                }
                spill += (self.placed[idx] - cap) * area;
                self.placed[idx] = cap;
                donors.push(idx);
                self.sync(idx);
            }
        }
        if spill <= 1.0e-8 {
            return;
        }
        let mut receivers: Vec<usize> = Vec::new();
        for j in j0.max(1)..=j1.min(res - 2) {
            for i in i0.max(1)..=i1.min(res - 2) {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if (x - cx).hypot(z - cz) > radius + self.cell * 1.2 {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                if donors.contains(&idx) {
                    continue;
                }
                receivers.push(idx);
            }
        }
        if receivers.is_empty() {
            return;
        }
        let add = spill / (receivers.len() as f32 * area);
        for idx in receivers {
            self.placed[idx] += add;
            self.kind[idx] = SurfaceKind::PlacedClod as u8;
            self.sync(idx);
        }
    }

    /// Clear tread negatives inside a rectangle. The bed and placed soil are
    /// unchanged; only the tread height record is removed.
    pub fn clear_tread_in(&mut self, x0: f32, x1: f32, z0: f32, z1: f32) {
        let half = self.size * 0.5;
        for j in 0..self.res {
            for i in 0..self.res {
                let x = (i as f32 + 0.5) * self.cell - half;
                let z = (j as f32 + 0.5) * self.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * self.res + i;
                self.tread_amp[idx] = 0.0;
                self.tread_ang[idx] = 0.0;
                self.tread_lat[idx] = 0.0;
            }
        }
    }

    /// Metres the persisted tread record sits below the rut surface.
    /// Negative inside a print, zero where no contact was stored.
    /// This is the height record. It does not move the bed or the pile.
    pub fn tread_drop(&self, x: f32, z: f32) -> f32 {
        match self.idx_at(x, z) {
            Some(idx) if self.tread_amp[idx] > 0.001 => -self.tread_amp[idx],
            _ => 0.0,
        }
    }

    /// Cells that still carry a tread negative, and the deepest lug.
    pub fn tread_imprint_stats(&self) -> (u32, f32) {
        let mut n = 0u32;
        let mut amp = 0.0_f32;
        for a in &self.tread_amp {
            if *a > 0.001 {
                n += 1;
                amp = amp.max(*a);
            }
        }
        (n, amp)
    }

    pub fn rut_track_span(&self) -> f32 {
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut count = 0;
        let half = self.size * 0.5;
        for j in 0..self.res {
            for i in 0..self.res {
                let idx = j * self.res + i;
                if self.initial_height[idx] - self.height[idx] > 0.02 || self.darken[idx] > 0.42 {
                    let x = (i as f32 + 0.5) * self.cell - half;
                    let z = (j as f32 + 0.5) * self.cell - half;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_z = min_z.min(z);
                    max_z = max_z.max(z);
                    count += 1;
                }
            }
        }
        if count < 4 {
            0.0
        } else {
            (max_z - min_z).max(max_x - min_x)
        }
    }

    pub fn mean_darken_delta(&self) -> f32 {
        let mut acc = 0.0;
        for d in &self.darken {
            acc += *d;
        }
        acc / self.darken.len() as f32
    }

    /// Cubic metres. `berm` is the spoil sitting on berm cells.
    pub fn volumes(&self) -> SoilVolumes {
        let area = self.cell * self.cell;
        let mut berm = 0.0_f32;
        let mut clods = 0.0_f32;
        let mut max_berm = 0.0_f32;
        for idx in 0..self.kind.len() {
            match SurfaceKind::from_u8(self.kind[idx]) {
                SurfaceKind::Berm => {
                    berm += self.placed[idx];
                    max_berm = max_berm.max(self.placed[idx]);
                }
                SurfaceKind::PlacedClod => clods += self.placed[idx],
                _ => {}
            }
        }
        SoilVolumes {
            cut: self.cut_volume(),
            placed: self.placed_volume(),
            berm: berm * area,
            clods: clods * area,
            max_rut: self.max_rut_depth(),
            max_berm,
        }
    }

    fn initial_at(&self, x: f32, z: f32) -> f32 {
        let half = self.size * 0.5;
        let u = (x + half) / self.cell - 0.5;
        let v = (z + half) / self.cell - 0.5;
        if u < 0.0 || v < 0.0 || u >= self.res as f32 - 1.0 || v >= self.res as f32 - 1.0 {
            return 0.0;
        }
        let i = u.floor() as usize;
        let j = v.floor() as usize;
        let tx = u - i as f32;
        let ty = v - j as f32;
        let res = self.res;
        bilerp(
            self.initial_height[j * res + i],
            self.initial_height[j * res + i + 1],
            self.initial_height[(j + 1) * res + i],
            self.initial_height[(j + 1) * res + i + 1],
            tx,
            ty,
        )
    }

    /// One tire from `from` to `to` in world XZ. The hub is one radius above
    /// the original surface, so the contact follows the slope.
    pub fn roll_lane(&mut self, from: Vec3, to: Vec3, sink: f32, slip: f32) {
        let from = Vec3::new(from.x, 0.0, from.z);
        let to = Vec3::new(to.x, 0.0, to.z);
        let delta = to - from;
        let len = delta.length();
        if len < 1.0e-3 {
            return;
        }
        let forward = delta / len;
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let n = (len / 0.1).round().max(1.0) as i32;
        let radius = 0.38;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = from.lerp(to, t);
            let h = self.initial_at(p.x, p.z);
            let hub = Vec3::new(p.x, h + radius - sink, p.z);
            self.stamp_wheel(
                hub, radius, 0.28, forward, right, sink, slip, 0.12, true, 0.0, 0.0,
            );
        }
    }

    /// One lane cut to the tire force. `sink` still sizes the chord. The bed
    /// depth is [`rut_depth_from_force`] when `normal` or `push` is set, and
    /// the cut volume is placed on the berm.
    pub fn roll_with_force(
        &mut self,
        from: Vec3,
        to: Vec3,
        sink: f32,
        slip: f32,
        normal: f32,
        push: f32,
    ) {
        let from = Vec3::new(from.x, 0.0, from.z);
        let to = Vec3::new(to.x, 0.0, to.z);
        let delta = to - from;
        let len = delta.length();
        if len < 1.0e-3 {
            return;
        }
        let forward = delta / len;
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let n = (len / 0.1).round().max(1.0) as i32;
        let radius = 0.38;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = from.lerp(to, t);
            let h = self.initial_at(p.x, p.z);
            let hub = Vec3::new(p.x, h + radius - sink, p.z);
            self.stamp_wheel(
                hub, radius, 0.28, forward, right, sink, slip, 0.12, true, normal, push,
            );
        }
        if normal > 1.0 || push > 1.0 {
            // The stamp drops the cut on a one-cell shoulder. A hard push
            // fills that cell into a tower. Spread the same volume into a ridge.
            self.shape_force_berm();
        }
    }

    /// Same force stamp as [`roll_with_force`] without reshaping the whole bed.
    /// The comparison of a second pass needs the groove the tire just cut.
    fn roll_force_lane(
        &mut self,
        from: Vec3,
        to: Vec3,
        sink: f32,
        slip: f32,
        normal: f32,
        push: f32,
    ) {
        let from = Vec3::new(from.x, 0.0, from.z);
        let to = Vec3::new(to.x, 0.0, to.z);
        let delta = to - from;
        let len = delta.length();
        if len < 1.0e-3 {
            return;
        }
        let forward = delta / len;
        let right = Vec3::new(forward.z, 0.0, -forward.x);
        let n = (len / 0.1).round().max(1.0) as i32;
        let radius = 0.38;
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let p = from.lerp(to, t);
            let h = self.initial_at(p.x, p.z);
            let hub = Vec3::new(p.x, h + radius - sink, p.z);
            self.stamp_wheel(
                hub, radius, 0.28, forward, right, sink, slip, 0.12, true, normal, push,
            );
        }
    }

    /// The force cut is a groove at the tire depth, and the spoil is the
    /// flank ridge. The deepest cell stays at that depth. Placed matches the cut.
    pub fn shape_force_berm(&mut self) {
        let res = self.res;
        let n = res * res;
        let total: f32 = self.placed.iter().copied().sum();
        if total <= 1.0e-6 {
            return;
        }
        let mut depth = vec![0.0_f32; n];
        let mut max_d = 0.0_f32;
        for idx in 0..n {
            let d = (self.initial_height[idx] - self.bed[idx]).max(0.0);
            depth[idx] = d;
            max_d = max_d.max(d);
        }
        if max_d < 0.02 {
            return;
        }
        // A flat run of cells is a rectangular pit. Spread the same deepest
        // depth into a groove so the wall is a slope, not a cliff.
        let orig = depth.clone();
        let sigma = (self.cell * 1.45).max(0.22);
        let rad = 3isize;
        let mut near_cut = vec![false; n];
        for idx in 0..n {
            if orig[idx] <= 0.004 {
                continue;
            }
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            for dj in -rad..=rad {
                for di in -rad..=rad {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    near_cut[jj as usize * res + ii as usize] = true;
                }
            }
        }
        let mut soft = vec![0.0_f32; n];
        for idx in 0..n {
            if !near_cut[idx] {
                continue;
            }
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            let mut acc = 0.0_f32;
            let mut wacc = 0.0_f32;
            for dj in -rad..=rad {
                for di in -rad..=rad {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    let dist = ((di * di + dj * dj) as f32).sqrt() * self.cell;
                    let w = (-0.5 * (dist / sigma).powi(2)).exp();
                    acc += orig[jj as usize * res + ii as usize] * w;
                    wacc += w;
                }
            }
            if wacc > 0.0 {
                soft[idx] = acc / wacc;
            }
        }
        let soft_max = soft.iter().copied().fold(0.0_f32, f32::max);
        if soft_max > 1.0e-6 {
            let scale = max_d / soft_max;
            for d in &mut soft {
                *d *= scale;
            }
        }
        for idx in 0..n {
            if !near_cut[idx] && orig[idx] <= 0.004 {
                continue;
            }
            depth[idx] = soft[idx];
            self.bed[idx] = self.initial_height[idx] - soft[idx];
            self.sync(idx);
        }
        max_d = depth.iter().copied().fold(0.0_f32, f32::max);
        let trench_at = (max_d * 0.45).max(0.03);
        let mut trench = vec![false; n];
        for idx in 0..n {
            trench[idx] = depth[idx] >= trench_at;
        }
        let mut comp_of = vec![-1i32; n];
        let mut comps: Vec<Vec<usize>> = Vec::new();
        for start in 0..n {
            if !trench[start] || comp_of[start] >= 0 {
                continue;
            }
            let id = comps.len() as i32;
            let mut comp = Vec::new();
            let mut stack = vec![start];
            comp_of[start] = id;
            while let Some(idx) = stack.pop() {
                comp.push(idx);
                let i = (idx % res) as isize;
                let j = (idx / res) as isize;
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    let nidx = jj as usize * res + ii as usize;
                    if trench[nidx] && comp_of[nidx] < 0 {
                        comp_of[nidx] = id;
                        stack.push(nidx);
                    }
                }
            }
            comps.push(comp);
        }
        if comps.is_empty() {
            return;
        }
        let mut pile = vec![0.0_f32; comps.len()];
        let mut seen = vec![false; n];
        let mut wave: Vec<usize> = Vec::new();
        for (id, comp) in comps.iter().enumerate() {
            for &idx in comp {
                seen[idx] = true;
                wave.push(idx);
                let _ = id;
            }
        }
        let mut owner = vec![-1i32; n];
        for &idx in &wave {
            owner[idx] = comp_of[idx];
        }
        let mut head = 0usize;
        while head < wave.len() {
            let idx = wave[head];
            head += 1;
            let id = owner[idx];
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let ii = i + di;
                let jj = j + dj;
                if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                    continue;
                }
                let nidx = jj as usize * res + ii as usize;
                if seen[nidx] {
                    continue;
                }
                seen[nidx] = true;
                owner[nidx] = id;
                wave.push(nidx);
            }
        }
        for idx in 0..n {
            let p = depth[idx];
            if p <= 1.0e-8 {
                continue;
            }
            let id = if owner[idx] >= 0 {
                owner[idx] as usize
            } else {
                0
            };
            pile[id] += p;
        }
        let cut_sum: f32 = depth.iter().copied().sum();
        let mut pad_counts = [0u32; 16];
        for idx in 0..n {
            if depth[idx] < 0.005 && self.placed[idx] < 0.001 {
                let k = self.kind[idx] as usize;
                if k < 16
                    && k != SurfaceKind::Berm as usize
                    && k != SurfaceKind::PlacedClod as usize
                    && k != SurfaceKind::CutLip as usize
                    && k != SurfaceKind::FreshCut as usize
                {
                    pad_counts[k] += 1;
                }
            }
        }
        let pad_kind = pad_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(k, _)| k as u8)
            .filter(|_| pad_counts.iter().any(|c| *c > 0))
            .unwrap_or(SurfaceKind::WetMud as u8);
        for idx in 0..n {
            if self.placed[idx] != 0.0 {
                self.placed[idx] = 0.0;
                self.sync(idx);
            }
        }
        let cell = self.cell;
        let diag = cell * std::f32::consts::SQRT_2;
        for (ci, comp) in comps.iter().enumerate() {
            let vol = pile[ci];
            if vol <= 1.0e-8 {
                continue;
            }
            // Metres outside the groove. The crest sits on the flank, not on the wall.
            let mut lip = vec![f32::MAX; n];
            let mut queue = Vec::with_capacity(comp.len() * 2);
            for idx in 0..n {
                if depth[idx] > 0.02 {
                    lip[idx] = 0.0;
                    queue.push(idx);
                }
            }
            let mut qh = 0usize;
            while qh < queue.len() {
                let idx = queue[qh];
                qh += 1;
                if lip[idx] > 1.05 {
                    continue;
                }
                let i = (idx % res) as isize;
                let j = (idx / res) as isize;
                for (di, dj, step) in [
                    (1isize, 0, cell),
                    (-1, 0, cell),
                    (0, 1, cell),
                    (0, -1, cell),
                    (1, 1, diag),
                    (1, -1, diag),
                    (-1, 1, diag),
                    (-1, -1, diag),
                ] {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    let nidx = jj as usize * res + ii as usize;
                    if trench[nidx] && comp_of[nidx] != ci as i32 {
                        continue;
                    }
                    let nd = lip[idx] + step;
                    if nd + 1.0e-5 < lip[nidx] && nd < 1.15 {
                        lip[nidx] = nd;
                        queue.push(nidx);
                    }
                }
            }
            let mut weights: Vec<(usize, f32)> = Vec::new();
            let mut wsum = 0.0_f32;
            for idx in 0..n {
                let d = lip[idx];
                if d > 1.05 || depth[idx] > 0.012 || owner[idx] != ci as i32 {
                    continue;
                }
                // Crest just off the lip, then a fall. Both flanks, one ridge.
                let w = (-0.5 * ((d - 0.24) / 0.15).powi(2)).exp();
                if w < 0.12 {
                    continue;
                }
                wsum += w;
                weights.push((idx, w));
            }
            if wsum <= 1.0e-6 {
                let add = vol / comp.len().max(1) as f32;
                for &idx in comp {
                    self.placed[idx] += add;
                    self.sync(idx);
                }
                continue;
            }
            let scale = vol / wsum;
            for (idx, w) in weights {
                self.placed[idx] += scale * w;
                if !firm_cover(SurfaceKind::from_u8(self.kind[idx])) {
                    self.kind[idx] = pad_kind;
                }
                self.loose[idx] = (self.loose[idx] + self.placed[idx] * 0.15).min(0.18);
                self.sync(idx);
            }
        }
        let after: f32 = self.placed.iter().copied().sum();
        let drift = cut_sum - after;
        if drift.abs() > 1.0e-5 {
            let mut crest: Vec<usize> = Vec::new();
            for idx in 0..n {
                if self.placed[idx] > 0.01 && !trench[idx] {
                    crest.push(idx);
                }
            }
            if !crest.is_empty() {
                let add = drift / crest.len() as f32;
                for idx in crest {
                    self.placed[idx] += add;
                    self.sync(idx);
                }
            }
        }
        // Share the crest with its neighbours so the end of the lane is not a tower.
        let mut src = self.placed.clone();
        for _ in 0..3 {
            let mut blurred = vec![0.0_f32; n];
            for idx in 0..n {
                if depth[idx] > 0.012 {
                    continue;
                }
                let i = (idx % res) as isize;
                let j = (idx / res) as isize;
                let mut acc = src[idx] * 0.30;
                let mut wacc = 0.30_f32;
                let mut any = src[idx] > 0.0;
                for dj in -1isize..=1 {
                    for di in -1isize..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ii = i + di;
                        let jj = j + dj;
                        if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                            continue;
                        }
                        let nidx = jj as usize * res + ii as usize;
                        if depth[nidx] > 0.012 {
                            continue;
                        }
                        let w = if di.abs() + dj.abs() == 1 { 0.12 } else { 0.05 };
                        acc += src[nidx] * w;
                        wacc += w;
                        any = any || src[nidx] > 0.0;
                    }
                }
                if any && wacc > 0.0 {
                    blurred[idx] = acc / wacc;
                }
            }
            src = blurred;
        }
        let bsum: f32 = src.iter().copied().sum();
        if bsum > 1.0e-6 {
            let fix = cut_sum / bsum;
            for idx in 0..n {
                self.placed[idx] = src[idx] * fix;
                if self.placed[idx] > 1.0e-4
                    && depth[idx] <= 0.012
                    && !firm_cover(SurfaceKind::from_u8(self.kind[idx]))
                {
                    self.kind[idx] = pad_kind;
                }
                self.sync(idx);
            }
        }
        // The lip is a sharp step in the mesh. This scar is the bed and the
        // ridge, so a lip cell is the cut or the berm.
        for idx in 0..n {
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            if !matches!(kind, SurfaceKind::CutLip | SurfaceKind::PlacedClod) {
                continue;
            }
            let i = (idx % res) as isize;
            let j = (idx / res) as isize;
            let mut near = trench[idx] || self.placed[idx] > 1.0e-4;
            if !near {
                for (di, dj) in [
                    (1isize, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    let ii = i + di;
                    let jj = j + dj;
                    if ii < 0 || jj < 0 || ii >= res as isize || jj >= res as isize {
                        continue;
                    }
                    let nidx = jj as usize * res + ii as usize;
                    if trench[nidx] || self.placed[nidx] > 1.0e-4 {
                        near = true;
                        break;
                    }
                }
            }
            if !near {
                continue;
            }
            self.kind[idx] = if depth[idx] >= 0.08 {
                SurfaceKind::FreshCut as u8
            } else {
                pad_kind
            };
            self.sync(idx);
        }
        for idx in 0..n {
            if firm_cover(SurfaceKind::from_u8(self.kind[idx])) {
                continue;
            }
            if depth[idx] >= 0.08 && self.placed[idx] <= depth[idx] {
                self.kind[idx] = SurfaceKind::FreshCut as u8;
                self.sync(idx);
            } else if self.placed[idx] > 1.0e-4 || depth[idx] >= 0.01 {
                let kind = SurfaceKind::from_u8(self.kind[idx]);
                if matches!(
                    kind,
                    SurfaceKind::Berm
                        | SurfaceKind::PlacedClod
                        | SurfaceKind::CutLip
                        | SurfaceKind::PackedClay
                        | SurfaceKind::FreshCut
                ) {
                    self.kind[idx] = pad_kind;
                    self.sync(idx);
                }
            }
        }
        for idx in 0..n {
            if self.placed[idx] > 1.0e-4 || depth[idx] >= 0.01 {
                continue;
            }
            let kind = SurfaceKind::from_u8(self.kind[idx]);
            if matches!(
                kind,
                SurfaceKind::Berm | SurfaceKind::PlacedClod | SurfaceKind::CutLip
            ) {
                self.kind[idx] = pad_kind;
                self.sync(idx);
            }
        }
    }

    /// Wet mud, no standing water. The berm still is this pad.
    pub fn mud_pad() -> Self {
        let mut field = Self::flat(0.45);
        for i in 0..field.kind.len() {
            field.kind[i] = SurfaceKind::WetMud as u8;
            field.wetness[i] = 0.82;
            field.base_wetness[i] = 0.78;
            field.water[i] = 0.0;
            field.loose[i] = 0.02;
            field.roughness[i] = 0.48;
            field.darken[i] = 0.08;
        }
        field
    }

    /// Tracks on one pad. The dry spell bakes a crust and opens cracks. A
    /// warped front leaves the same clay wet where it has not finished drying.
    /// The bed and the placed pile stay the wet cut.
    pub fn drying_pair() -> (Self, DryingCrust) {
        let mut field = Self::mud_pad();
        let paint = |field: &mut SoilField, x: f32, z: f32, kind: SurfaceKind, wet: f32| {
            let half = field.size * 0.5;
            for j in 0..field.res {
                for i in 0..field.res {
                    let cx = (i as f32 + 0.5) * field.cell - half;
                    let cz = (j as f32 + 0.5) * field.cell - half;
                    if (cx - x).abs() < 0.4 && (cz - z).abs() < 0.4 {
                        let idx = j * field.res + i;
                        field.kind[idx] = kind as u8;
                        field.wetness[idx] = wet;
                        field.base_wetness[idx] = wet;
                        field.water[idx] = 0.0;
                    }
                }
            }
        };
        paint(&mut field, 6.2, 6.0, SurfaceKind::Sand, 0.22);
        paint(&mut field, 7.2, 6.0, SurfaceKind::Silt, 0.72);
        let lanes = [-2.44_f32, -0.96, 0.96, 2.44];
        for x in lanes {
            field.roll_lane(
                Vec3::new(x, 0.0, -2.6),
                Vec3::new(x, 0.0, 2.6),
                0.055,
                0.25,
            );
        }
        let wet_snap = (
            field.wetness.clone(),
            field.water.clone(),
            field.kind.clone(),
            field.roughness.clone(),
            field.crust.clone(),
            field.crack.clone(),
        );
        let bed0 = field.bed.clone();
        let placed0 = field.placed.clone();
        let cloud0 = field.cloud.clone();
        let slurry0 = field.slurry.clone();
        let wet_sample = field.floor_sample(0.96, 0.0);
        let wet_sink = sinkage_on(
            1_600.0,
            wet_sample.kind(),
            wet_sample.wetness,
            wet_sample.loose,
            wet_sample.water,
            0.38,
            0.28,
        );
        let wetness_before = field.lane_mean(0.4, |s| s.0);
        let mut wet_again = field.clone();
        field.advance_weather(DRYING_SPELL_S, 0.0);
        let bed_gap = max_abs(&bed0, &field.bed);
        let placed_gap = max_abs(&placed0, &field.placed);
        let cloud_gap = max_abs(&cloud0, &field.cloud);
        let slurry_gap = max_abs(&slurry0, &field.slurry);
        let (wetness_after, crust_m, crack_m, crack_max, crack_cells, floor_cells) =
            field.lane_crust(0.4);
        let baked = field.floor_sample(0.96, 0.0);
        let crust_sink = sinkage_on(
            1_600.0,
            baked.kind(),
            baked.wetness,
            baked.loose,
            baked.water,
            0.38,
            0.28,
        );
        let old_sink = sinkage_on_track(
            1_600.0,
            baked.kind(),
            baked.wetness,
            baked.loose,
            baked.water,
            0.38,
            0.28,
            baked.crust,
            baked.crack,
            baked.passes,
            baked.rest,
        );
        let harden = track_harden_mul(baked.crust, baked.crack, baked.passes, baked.rest);
        let mut old_again = field.clone();
        let from = Vec3::new(0.96, 0.0, -2.6);
        let to = Vec3::new(0.96, 0.0, 2.6);
        let wet_cut0 = wet_again.lane_cut(0.96);
        let old_cut0 = old_again.lane_cut(0.96);
        // A load just past the wet rut. The baked floor is already harder than that.
        let again_n = 1_900.0;
        let again_p = 180.0;
        wet_again.roll_force_lane(from, to, 0.05, 0.15, again_n, again_p);
        old_again.roll_force_lane(from, to, 0.05, 0.15, again_n, again_p);
        let sand_crust = field.patch_max_crust(6.2, 6.0);
        let silt_crust = field.patch_max_crust(7.2, 6.0);
        let cut = field.volumes().cut;
        let placed = field.volumes().placed;
        // One pad, drying. Wet clay and the crust are the same soil. The
        // front is wide and warped, so the picture is not two tiles split
        // on x = 0. The bed and the placed pile stay the wet cut.
        for idx in 0..field.bed.len() {
            let (x, z) = field.cell_xz(idx);
            let mut d = drying_front(x, z);
            // The measured wet lane stays wet. The box is off the close frame.
            if (x + 0.96).abs() < 0.48 && z.abs() < 0.48 {
                d = 0.0;
            }
            if d >= 0.999 {
                continue;
            }
            field.wetness[idx] = lerp(wet_snap.0[idx], field.wetness[idx], d);
            field.water[idx] = lerp(wet_snap.1[idx], field.water[idx], d);
            field.roughness[idx] = lerp(wet_snap.3[idx], field.roughness[idx], d);
            let thickness = field.crust[idx] * d;
            let opening = field.crack[idx] * d;
            field.commit_skin(idx, thickness, opening);
            if d < 0.82 {
                field.kind[idx] = wet_snap.2[idx];
            }
        }
        let west_floor = field.floor_sample(-0.96, 0.0);
        let stats = DryingCrust {
            wetness_before,
            wetness_after,
            crust_m,
            crack_m,
            crack_max,
            crack_cells,
            floor_cells,
            wet_sink,
            crust_sink,
            old_sink,
            harden,
            wet_extra_cut: wet_again.lane_cut(0.96) - wet_cut0,
            old_extra_cut: old_again.lane_cut(0.96) - old_cut0,
            cut,
            placed,
            bed_gap,
            placed_gap,
            cloud_gap,
            slurry_gap,
            sand_crust,
            silt_crust,
            west_wetness: west_floor.wetness,
            west_kind: west_floor.kind,
            east_kind: baked.kind,
            east_crust: baked.crust,
            east_crack: baked.crack,
        };
        (field, stats)
    }

    fn lane_mean(&self, x_min: f32, pick: fn((f32, f32)) -> f32) -> f32 {
        let mut acc = 0.0;
        let mut n = 0.0;
        for idx in 0..self.bed.len() {
            let (x, z) = self.cell_xz(idx);
            if x < x_min || z.abs() > 2.3 {
                continue;
            }
            let rut = (self.initial_height[idx] - self.bed[idx]).max(0.0);
            if rut < 0.02 {
                continue;
            }
            acc += pick((self.wetness[idx], self.crust[idx]));
            n += 1.0;
        }
        if n <= 0.0 {
            0.0
        } else {
            acc / n
        }
    }

    fn lane_crust(&self, x_min: f32) -> (f32, f32, f32, f32, u32, u32) {
        let mut wet = 0.0;
        let mut crust = 0.0;
        let mut crack = 0.0;
        let mut crack_max = 0.0_f32;
        let mut crack_cells = 0u32;
        let mut floor = 0u32;
        for idx in 0..self.bed.len() {
            let (x, z) = self.cell_xz(idx);
            if x < x_min || z.abs() > 2.3 {
                continue;
            }
            let rut = (self.initial_height[idx] - self.bed[idx]).max(0.0);
            if rut < 0.02 {
                continue;
            }
            floor += 1;
            wet += self.wetness[idx];
            crust += self.crust[idx];
            crack += self.crack[idx];
            crack_max = crack_max.max(self.crack[idx]);
            if self.crack[idx] > 0.003 {
                crack_cells += 1;
            }
        }
        let n = floor.max(1) as f32;
        (wet / n, crust / n, crack / n, crack_max, crack_cells, floor)
    }

    /// Cut volume in the groove, not the berm beside it.
    fn lane_cut(&self, x: f32) -> f32 {
        let area = self.cell * self.cell;
        let mut v = 0.0_f32;
        for idx in 0..self.bed.len() {
            let (cx, cz) = self.cell_xz(idx);
            if (cx - x).abs() > 0.30 || cz.abs() > 2.7 {
                continue;
            }
            v += (self.initial_height[idx] - self.bed[idx]).max(0.0);
        }
        v * area
    }

    /// Deepest cut near a point. A shoulder sample would miss the floor.
    fn floor_sample(&self, x: f32, z: f32) -> SoilSample {
        let half = self.size * 0.5;
        let mut best = self.sample(x, z);
        let mut best_rut = best.rut;
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                if (cx - x).abs() > 0.45 || (cz - z).abs() > 0.45 {
                    continue;
                }
                let idx = j * self.res + i;
                let rut = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                if rut > best_rut {
                    best_rut = rut;
                    best = self.sample(cx, cz);
                }
            }
        }
        best
    }

    fn patch_max_crust(&self, x: f32, z: f32) -> f32 {
        let half = self.size * 0.5;
        let mut m = 0.0_f32;
        for j in 0..self.res {
            for i in 0..self.res {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                if (cx - x).abs() < 0.4 && (cz - z).abs() < 0.4 {
                    m = m.max(self.crust[j * self.res + i]);
                }
            }
        }
        m
    }

    /// Ramp along +X. The ends flatten so the height stays in the texture range.
    pub fn sloped(grade: f32) -> Self {
        let mut field = Self::mud_pad();
        let half = field.size * 0.5;
        let grade = grade.clamp(-0.7, 0.7);
        for j in 0..field.res {
            for i in 0..field.res {
                let x = (i as f32 + 0.5) * field.cell - half;
                let h = 0.62 + grade * x.clamp(-2.5, 2.5);
                let idx = j * field.res + i;
                field.height[idx] = h;
                field.bed[idx] = h;
                field.initial_height[idx] = h;
                field.placed[idx] = 0.0;
            }
        }
        field
    }

    /// One groove. `second_pass` drives it again, deeper, and the berm takes
    /// that extra soil.
    pub fn berm_scene(second_pass: bool) -> Self {
        let mut field = Self::mud_pad();
        field.roll_lane(
            Vec3::new(0.0, 0.0, -1.15),
            Vec3::new(0.0, 0.0, 1.15),
            0.085,
            0.32,
        );
        if second_pass {
            field.roll_lane(
                Vec3::new(0.0, 0.0, -1.15),
                Vec3::new(0.0, 0.0, 1.15),
                0.16,
                0.85,
            );
        }
        field
    }

    /// Flat pad split down the middle. Wet clay on the negative X side, gravel
    /// on the positive side. The bed height is the same, so a tire that sinks
    /// more on the clay is the soil, not a hole that was authored in.
    pub fn surface_split() -> Self {
        let mut field = Self::flat(0.16);
        let half = field.size * 0.5;
        for j in 0..field.res {
            for i in 0..field.res {
                let x = (i as f32 + 0.5) * field.cell - half;
                let idx = j * field.res + i;
                if x < 0.0 {
                    field.kind[idx] = SurfaceKind::WetMud as u8;
                    field.wetness[idx] = 0.88;
                    field.base_wetness[idx] = 0.84;
                    field.water[idx] = 0.0;
                    field.loose[idx] = 0.02;
                    field.roughness[idx] = 0.40;
                    field.darken[idx] = 0.12;
                } else {
                    field.kind[idx] = SurfaceKind::Gravel as u8;
                    field.wetness[idx] = 0.05;
                    field.base_wetness[idx] = 0.05;
                    field.water[idx] = 0.0;
                    field.loose[idx] = 0.0;
                    field.roughness[idx] = 0.96;
                    field.darken[idx] = 0.02;
                }
            }
        }
        field
    }

    /// Clay pad with a rut, sand on the right, silt on the left, gravel at the
    /// far corner. The wet picture is this field. Drying it is the dry picture.
    /// The flanks sit outside the berm so the stamp does not turn them into spoil.
    pub fn moisture_scene() -> Self {
        let mut field = Self::flat(0.2);
        let res = field.res;
        let cell = field.cell;
        let half = field.size * 0.5;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                if x.abs() > 2.35 || z.abs() > 2.7 {
                    continue;
                }
                let idx = j * res + i;
                field.kind[idx] = SurfaceKind::WetMud as u8;
                field.wetness[idx] = 0.88;
                field.base_wetness[idx] = 0.84;
                field.water[idx] = 0.0;
                field.loose[idx] = 0.02;
                field.roughness[idx] = 0.4;
                field.darken[idx] = 0.1;
            }
        }
        field.roll_lane(
            Vec3::new(0.0, 0.0, -1.2),
            Vec3::new(0.0, 0.0, 1.2),
            0.085,
            0.32,
        );
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                if (1.2..2.3).contains(&x) && (-2.5..1.25).contains(&z) {
                    field.kind[idx] = SurfaceKind::Sand as u8;
                    field.wetness[idx] = 0.22;
                    field.base_wetness[idx] = 0.1;
                    field.water[idx] = 0.0;
                    field.loose[idx] = 0.01;
                    field.roughness[idx] = 0.92;
                    field.darken[idx] = 0.02;
                } else if (-2.3..-1.2).contains(&x) && (-2.5..2.5).contains(&z) {
                    field.kind[idx] = SurfaceKind::Silt as u8;
                    field.wetness[idx] = 0.86;
                    field.base_wetness[idx] = 0.3;
                    field.water[idx] = 0.0;
                    field.loose[idx] = 0.015;
                    field.roughness[idx] = 0.48;
                    field.darken[idx] = 0.06;
                } else if (1.2..2.3).contains(&x) && (1.4..2.6).contains(&z) {
                    field.kind[idx] = SurfaceKind::Gravel as u8;
                    field.wetness[idx] = 0.05;
                    field.base_wetness[idx] = 0.05;
                    field.water[idx] = 0.0;
                    field.loose[idx] = 0.0;
                    field.roughness[idx] = 0.96;
                    field.darken[idx] = 0.02;
                }
            }
        }
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                let dug = (field.initial_height[idx] - field.bed[idx]).max(0.0);
                if dug > 0.02 && x.abs() < 0.28 {
                    field.water[idx] = field.water[idx].max(0.03);
                    field.wetness[idx] = field.wetness[idx].max(0.92);
                }
            }
        }
        field
    }

    /// Flat pad with clay, silt, and sand strips. Each strip holds four moisture
    /// bands along Z so the Niva can sit on damp clay while the readout compares
    /// dry, damp, saturated, and pooled traction on all three soils.
    pub fn moisture_traction_pad() -> Self {
        let mut field = Self::flat(0.18);
        let res = field.res;
        let cell = field.cell;
        let half = field.size * 0.5;
        let strips = [
            (-1.35_f32, SurfaceKind::WetMud),
            (0.0, SurfaceKind::Silt),
            (1.35, SurfaceKind::Sand),
        ];
        let bands: [(f32, f32); 4] = [(-2.05, -1.05), (-0.85, 0.15), (0.35, 1.35), (1.55, 2.55)];
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                let mut best = f32::MAX;
                let mut pick = (
                    SurfaceKind::WetMud,
                    moisture_state_fields(SurfaceKind::WetMud, SurfaceMoisture::Dry),
                );
                for (cx, kind) in strips {
                    if (x - cx).abs() > 0.52 {
                        continue;
                    }
                    for (state, (z0, z1)) in SurfaceMoisture::ALL.iter().zip(bands.iter()) {
                        let z_mid = (z0 + z1) * 0.5;
                        let d = (z - z_mid).abs();
                        if d < best {
                            best = d;
                            pick = (kind, moisture_state_fields(kind, *state));
                        }
                    }
                }
                let (kind, (wetness, loose, water)) = pick;
                if best < 0.55 {
                    field.kind[idx] = kind as u8;
                    field.wetness[idx] = wetness;
                    field.base_wetness[idx] = wetness.min(0.75);
                    field.water[idx] = water;
                    field.loose[idx] = loose;
                    field.roughness[idx] = match kind {
                        SurfaceKind::Sand => 0.90,
                        SurfaceKind::Silt => 0.46,
                        _ => 0.38,
                    };
                }
            }
        }
        field
    }

    /// Two grooves on the lower face of a 0.42 ramp. The upper face is untouched.
    pub fn slope_scene() -> Self {
        let mut field = Self::sloped(0.42);
        for z in [-0.74_f32, 0.74] {
            field.roll_lane(
                Vec3::new(-1.65, 0.0, z),
                Vec3::new(-0.45, 0.0, z),
                0.09,
                0.3,
            );
        }
        field
    }
}

/// Where a shed sidewall coat landed. Volumes are cubic metres.
#[derive(Clone, Copy, Debug)]
pub struct CoatLanding {
    pub volume: f32,
    /// Share that sat down on cells that were already a rut.
    pub in_rut: f32,
}

/// What one rigid patch took out of the field. Volumes are cubic metres.
/// `bed + moved == berm + thrown`. Thrown mass is added to [`SoilField::airborne`].
#[derive(Clone, Copy, Debug, Default)]
pub struct PlowCut {
    pub bed: f32,
    pub moved: f32,
    pub berm: f32,
    pub thrown: f32,
}

/// A lane cut in slurry. Cubic metres. `cut == berm + thrown + slurry`.
/// `slurry` is the share that stayed in the rut. The throw is on `airborne`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlurryCut {
    pub cut: f32,
    pub berm: f32,
    pub thrown: f32,
    pub slurry: f32,
}

/// A sideways shove of slurry that was already on the field.
#[derive(Clone, Copy, Debug, Default)]
pub struct SlurrySlide {
    /// Cubic metres that left the cell they started in.
    pub moved: f32,
    /// Signed metres the slurry centroid traveled along the shove.
    pub travel: f32,
    /// Slurry volume after, minus before. Zero when nothing was minted.
    pub minted: f32,
}

/// High slurry, the low rut in front of it, and the shared plan extents.
#[derive(Clone, Copy, Debug)]
pub struct SlurryFlowBed {
    pub high_z0: f32,
    pub high_z1: f32,
    pub low_z0: f32,
    pub low_z1: f32,
    pub x0: f32,
    pub x1: f32,
}

/// Slurry moved on one tire pass through deep slurry.
#[derive(Clone, Copy, Debug, Default)]
pub struct BowWakePass {
    /// Cubic metres piled ahead in the bow wave.
    pub bow_m3: f32,
    /// Cubic metres thinned behind the tread in the wake.
    pub wake_m3: f32,
}

/// Free-water bow ahead of the hull. `height` is metres above the still column.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaterBow {
    pub height: f32,
    pub bow_m3: f32,
    pub wake_m3: f32,
}

/// Cells patched so a ford contact reads the bed. The bowl is not a spun rut,
/// and the standing column is fluid, not slurry under the tire. Restoring
/// puts the cut and the column back, so the ledger is unchanged.
pub struct FordContactMask {
    initial: Vec<(usize, f32)>,
    water: Vec<(usize, f32)>,
}

/// Skin volume on the heightfield. Thickness is a closed book. Opening is
/// the fissure volume and is not added to the solid column.
#[derive(Clone, Copy, Debug)]
pub struct CrustBook {
    pub thickness_m3: f64,
    pub opening_m3: f64,
    pub grown_m3: f64,
    pub lost_m3: f64,
}

impl CrustBook {
    /// `|grown − lost − still on the field|`, cubic metres.
    pub fn gap(self) -> f64 {
        (self.grown_m3 - self.lost_m3 - self.thickness_m3).abs()
    }
}

/// Water plus soil, split into the solid, the free column, and the slurry.
#[derive(Clone, Copy, Debug)]
pub struct MixVolumes {
    pub soil: f32,
    pub water: f32,
    pub slurry: f32,
}

impl MixVolumes {
    pub fn total(self) -> f32 {
        self.soil + self.water + self.slurry
    }
}

/// Cut soil against the pile that received it.
#[derive(Clone, Copy, Debug)]
pub struct SoilVolumes {
    pub cut: f32,
    pub placed: f32,
    pub berm: f32,
    pub clods: f32,
    pub max_rut: f32,
    pub max_berm: f32,
}

fn firm_cover(kind: SurfaceKind) -> bool {
    matches!(
        kind,
        SurfaceKind::Gravel
            | SurfaceKind::Grass
            | SurfaceKind::DryCrust
            | SurfaceKind::Sand
            | SurfaceKind::Silt
    )
}

fn opened_kind(prev: u8, water: f32, darken: f32) -> u8 {
    let prev = SurfaceKind::from_u8(prev);
    if firm_cover(prev) {
        prev as u8
    } else if water > 0.03 {
        SurfaceKind::FreshCut as u8
    } else if darken > 0.72 {
        SurfaceKind::PackedClay as u8
    } else {
        SurfaceKind::FreshCut as u8
    }
}

fn horizontal(v: Vec3) -> Vec3 {
    let v = Vec3::new(v.x, 0.0, v.z);
    if v.length_squared() < 1.0e-8 {
        Vec3::Z
    } else {
        v.normalize()
    }
}

fn transfer(
    height: &[f32],
    water: &[f32],
    kind: &[u8],
    next: &mut [f32],
    a: usize,
    b: usize,
    dt: f32,
    dx: f32,
    area: f32,
    cond: f32,
) {
    let ka = SurfaceKind::from_u8(kind[a]);
    let kb = SurfaceKind::from_u8(kind[b]);
    let mobile_a = (water[a] - water_hold(ka)).max(0.0);
    let mobile_b = (water[b] - water_hold(kb)).max(0.0);
    if mobile_a < 1.0e-5 && mobile_b < 1.0e-5 {
        return;
    }
    let sa = height[a] + water[a];
    let sb = height[b] + water[b];
    let dh = sa - sb;
    if dh.abs() < 1.0e-5 {
        return;
    }
    // The donor's soil decides how willingly the water leaves.
    let face = if dh > 0.0 {
        cond * water_conduct(ka)
    } else {
        cond * water_conduct(kb)
    };
    // Volume across the shared face. Limited so one pass cannot drain a cell,
    // and pore water below the hold stays put.
    let mut vol = dh * face * dt * dx * 0.35;
    if vol > 0.0 {
        vol = vol.min(mobile_a * area * 0.22);
        next[a] -= vol / area;
        next[b] += vol / area;
    } else {
        vol = (-vol).min(mobile_b * area * 0.22);
        next[b] -= vol / area;
        next[a] += vol / area;
    }
}

fn born_kind(x: f32, z: f32, wet: f32, water: f32) -> SurfaceKind {
    if water > 0.02 || wet > 0.72 {
        return SurfaceKind::WetMud;
    }
    if launch_runup(x, z) {
        return SurfaceKind::DryCrust;
    }
    let off = track_offset(x, z).abs();
    let far = (x * x + z * z).sqrt();
    let n = fbm2(x * 0.16, z * 0.16);
    let n2 = fbm2(x * 0.07 + 4.2, z * 0.07);
    if far > 13.2 || (off > 6.4 && n2 > 0.62) {
        return SurfaceKind::Gravel;
    }
    // Sand apron outside the gravel. It drains and does not become the lane.
    if far > 10.4 && n2 > 0.34 {
        return SurfaceKind::Sand;
    }
    // Silt sits off the racing line, in the pockets that hold a damp flour.
    if off > 3.5 && off < 5.5 && n < 0.42 && wet < 0.58 {
        return SurfaceKind::Silt;
    }
    if off > 2.5 && n > 0.46 && wet < 0.48 {
        return SurfaceKind::Grass;
    }
    if wet < 0.36 || (off > 1.45 && off < 3.3 && wet < 0.55) {
        return SurfaceKind::DryCrust;
    }
    SurfaceKind::Undisturbed
}

fn moisture_step_target(
    kind: SurfaceKind,
    base: f32,
    wetting: bool,
    mix: [f32; 4],
) -> (f32, f32) {
    let mut sum = 0.0_f32;
    let mut peak = 0.0_f32;
    for share in mix {
        sum += share;
        peak = peak.max(share);
    }
    if !(sum >= 0.05 && peak < 0.999) {
        return (
            kind_moisture_target(kind, base, wetting),
            kind_moisture_rate(kind, wetting),
        );
    }
    let kinds = [
        SurfaceKind::WetMud,
        SurfaceKind::Sand,
        SurfaceKind::Gravel,
        SurfaceKind::Grass,
    ];
    let mut target = 0.0;
    let mut rate = 0.0;
    for (i, material) in kinds.into_iter().enumerate() {
        let w = mix[i];
        if w <= 0.0 {
            continue;
        }
        target += kind_moisture_target(material, base, wetting) * w;
        rate += kind_moisture_rate(material, wetting) * w;
    }
    (target / sum, rate / sum)
}

fn kind_moisture_target(kind: SurfaceKind, base: f32, wetting: bool) -> f32 {
    if wetting {
        wet_target(kind)
    } else {
        dry_target(kind, base)
    }
}

fn kind_moisture_rate(kind: SurfaceKind, wetting: bool) -> f32 {
    if wetting {
        wet_rate(kind)
    } else {
        dry_rate(kind)
    }
}

fn wet_target(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::Gravel => 0.07,
        SurfaceKind::Sand => 0.4,
        SurfaceKind::Silt => 0.9,
        // A crust beads. Pores do not fill the way wet clay does.
        SurfaceKind::DryCrust => 0.20,
        SurfaceKind::WetMud | SurfaceKind::PackedClay | SurfaceKind::FreshCut => 0.93,
        SurfaceKind::Grass => 0.42,
        _ => 0.72,
    }
}

fn dry_target(kind: SurfaceKind, base: f32) -> f32 {
    match kind {
        SurfaceKind::Gravel => 0.04,
        SurfaceKind::Sand => 0.06,
        SurfaceKind::Silt => 0.16,
        SurfaceKind::DryCrust => 0.08,
        // Wet clay ignores the authored base and bakes toward a crust.
        SurfaceKind::WetMud | SurfaceKind::PackedClay | SurfaceKind::FreshCut => 0.05,
        SurfaceKind::Grass => base.clamp(0.12, 0.4),
        _ => base.clamp(0.1, 0.7),
    }
}

fn wet_rate(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::Gravel => 0.12,
        SurfaceKind::Sand => 0.4,
        SurfaceKind::Silt => 0.55,
        SurfaceKind::DryCrust => 0.045,
        SurfaceKind::WetMud | SurfaceKind::PackedClay | SurfaceKind::FreshCut => 0.62,
        _ => 0.3,
    }
}

fn dry_rate(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::Gravel => 1.4,
        SurfaceKind::Sand => 0.28,
        SurfaceKind::Silt => 0.014,
        SurfaceKind::DryCrust => 0.10,
        SurfaceKind::WetMud | SurfaceKind::PackedClay | SurfaceKind::FreshCut => 0.026,
        SurfaceKind::Grass => 0.01,
        // Undisturbed ground keeps the map's authored wetness.
        _ => 0.006,
    }
}

fn film_evap(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::DryCrust => 0.010,
        SurfaceKind::WetMud | SurfaceKind::PackedClay | SurfaceKind::FreshCut => 0.012,
        SurfaceKind::Silt => 0.0014,
        SurfaceKind::Sand => 0.004,
        SurfaceKind::Gravel => 0.02,
        _ => 0.001,
    }
}

/// Wet clay below this pore moisture, with no pond, becomes a crust.
const CRUST_AT: f32 = 0.34;
/// A crust above this, or under standing water, is wet clay again.
const REWET_AT: f32 = 0.44;
/// Thickest dry skin, metres. The shader's red channel is this over [`CRUST_SHADER_METRES`].
pub const CRUST_MAX_M: f32 = 0.036;
/// Thickness ceiling, metres. The dry spell stops at [`CRUST_MAX_M`], under this.
pub const CRUST_CAP_M: f32 = 0.05;
/// Widest shrinkage opening, metres. The shader's green channel is this over [`CRACK_SHADER_METRES`].
pub const CRACK_MAX_M: f32 = 0.016;
/// Skin that has to form before a fissure opens.
pub const CRACK_ONSET_M: f32 = 0.008;
/// Metres encoded as 1.0 in the crust texture's red channel.
pub const CRUST_SHADER_METRES: f32 = 0.04;
/// Metres encoded as 1.0 in the crust texture's green channel.
pub const CRACK_SHADER_METRES: f32 = 0.02;
/// Seconds of dry weather used to bake a fresh track into a crust.
pub const DRYING_SPELL_S: f32 = 180.0;

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0.0_f32, |m, (u, v)| m.max((u - v).abs()))
}

/// Numbers from one dry spell on a pair of tracks.
#[derive(Clone, Copy, Debug)]
pub struct DryingCrust {
    pub wetness_before: f32,
    pub wetness_after: f32,
    pub crust_m: f32,
    pub crack_m: f32,
    pub crack_max: f32,
    pub crack_cells: u32,
    pub floor_cells: u32,
    pub wet_sink: f32,
    pub crust_sink: f32,
    pub old_sink: f32,
    pub harden: f32,
    pub wet_extra_cut: f32,
    pub old_extra_cut: f32,
    pub cut: f32,
    pub placed: f32,
    pub bed_gap: f32,
    pub placed_gap: f32,
    pub cloud_gap: f32,
    pub slurry_gap: f32,
    pub sand_crust: f32,
    pub silt_crust: f32,
    pub west_wetness: f32,
    pub west_kind: u8,
    pub east_kind: u8,
    pub east_crust: f32,
    pub east_crack: f32,
}
/// How far a cell has dried for the still. 0 is the wet cut, 1 is the rested crust.
/// The front wanders, so wet ground and crust are one pad rather than a straight cut.
fn drying_front(x: f32, z: f32) -> f32 {
    let n = noise2(x * 0.45 + 1.7, z * 0.38 + 0.4);
    let n2 = noise2(x * 1.15 - 2.2, z * 0.72 + 3.1);
    let across = (x + 0.15 + (n - 0.5) * 2.4) / 2.1;
    let along = (z + 0.2) * 0.22 + (n2 - 0.5) * 0.85;
    let t = (across + along).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Metres per second the skin grows at a full moisture deficit, before the track bonus.
/// A twelve-second lap stays under the harden threshold. A dry spell does not.
const CRUST_GROW: f32 = 0.00022;
/// Metres per second a pond or rain soaks the skin back off.
const CRUST_SOAK: f32 = 0.0055;
/// Metres per second a fissure closes once the skin is gone.
const CRACK_CLOSE: f32 = 0.012;

/// Clay that can bake. Sand, silt, gravel, and grass do not.
fn crustable(kind: SurfaceKind) -> bool {
    matches!(
        kind,
        SurfaceKind::WetMud
            | SurfaceKind::PackedClay
            | SurfaceKind::FreshCut
            | SurfaceKind::DryCrust
            | SurfaceKind::CutLip
            | SurfaceKind::Berm
            | SurfaceKind::PlacedClod
    )
}

/// Extra Bekker modulus on a track that has rested under a crust.
/// A wet track, a tire still on the cell, and a surface with no pass stay at 1.
pub fn track_harden_mul(crust: f32, crack: f32, passes: u8, rest: f32) -> f32 {
    if passes == 0 || crust < 0.006 || rest <= 0.40 {
        return 1.0;
    }
    let age = ((rest - 0.40) / 12.0).clamp(0.0, 1.0);
    let skin = (crust / CRUST_MAX_M).clamp(0.0, 1.0);
    let open = (crack / CRACK_MAX_M).clamp(0.0, 1.0);
    1.0 + 1.8 * skin * age * (1.0 - 0.35 * open)
}

/// 1 on a shrinkage fissure, 0 inside the polygon. World metres.
pub fn crack_mask(x: f32, z: f32) -> f32 {
    let scale = 0.48_f32;
    let fx = x / scale;
    let fz = z / scale;
    let ix = fx.floor() as i32;
    let iz = fz.floor() as i32;
    let mut best = f32::MAX;
    let mut second = f32::MAX;
    for dj in -1..=1 {
        for di in -1..=1 {
            let gx = (ix + di) as f32;
            let gz = (iz + dj) as f32;
            let jx = hash2(gx, gz) - 0.5;
            let jz = hash2(gx + 4.2, gz + 1.7) - 0.5;
            let cx = gx + 0.5 + jx * 0.8;
            let cz = gz + 0.5 + jz * 0.8;
            let d = (fx - cx).hypot(fz - cz);
            if d < best {
                second = best;
                best = d;
            } else if d < second {
                second = d;
            }
        }
    }
    let gap = (second - best).max(0.0);
    (1.0 - gap / 0.11).clamp(0.0, 1.0)
}

/// RGBA8 texel for a shader. R is crust thickness, G is crack opening.
/// B is unused. A is opaque so a sampler can tell a written cell from empty.
pub fn crust_shader_texel(thickness: f32, crack: f32) -> [u8; 4] {
    let r = (thickness.max(0.0) / CRUST_SHADER_METRES).clamp(0.0, 1.0);
    let g = (crack.max(0.0) / CRACK_SHADER_METRES).clamp(0.0, 1.0);
    [
        (r * 255.0).round() as u8,
        (g * 255.0).round() as u8,
        0,
        255,
    ]
}

/// Read the metres a shader texel is holding.
pub fn crust_shader_metres(texel: [u8; 4]) -> (f32, f32) {
    (
        texel[0] as f32 / 255.0 * CRUST_SHADER_METRES,
        texel[1] as f32 / 255.0 * CRACK_SHADER_METRES,
    )
}

fn evolve_kind(kind: SurfaceKind, wetness: f32, water: f32) -> SurfaceKind {
    // Pore water the soil is allowed to hold is not a pond. A film thicker
    // than that keeps wet clay from baking; once it is gone the crust can form.
    let standing = water > water_hold(kind) + 0.012;
    // A crust beads a thin film. It breaks into wet clay only under a deeper pond.
    let crust_pond = water > 0.020;
    match kind {
        SurfaceKind::WetMud | SurfaceKind::PackedClay | SurfaceKind::FreshCut
            if wetness < CRUST_AT && !standing =>
        {
            SurfaceKind::DryCrust
        }
        SurfaceKind::DryCrust if wetness > REWET_AT || crust_pond => SurfaceKind::WetMud,
        other => other,
    }
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn bilerp(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> f32 {
    let ab = a + (b - a) * tx;
    let cd = c + (d - c) * tx;
    ab + (cd - ab) * ty
}

/// Rolling depth after `passes` visits. Pass 1 is the fresh bite. Each later
/// visit of the same slot is deeper, capped so the chassis is not swallowed.
fn pass_groove(one_pass: f32, passes: u8) -> f32 {
    let n = (passes as f32).max(1.0);
    (one_pass * n.powf(1.25)).min(0.30)
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn read_f32s(bytes: &[u8], i: &mut usize, n: usize) -> Result<Vec<f32>, String> {
    let need = n * 4;
    if *i + need > bytes.len() {
        return Err("truncated soil snapshot".into());
    }
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let end = *i + 4;
        let raw: [u8; 4] = bytes[*i..end].try_into().unwrap();
        values.push(f32::from_le_bytes(raw));
        *i = end;
    }
    Ok(values)
}

fn read_u8s(bytes: &[u8], i: &mut usize, n: usize) -> Result<Vec<u8>, String> {
    if *i + n > bytes.len() {
        return Err("truncated soil snapshot".into());
    }
    let values = bytes[*i..*i + n].to_vec();
    *i += n;
    Ok(values)
}

/// Signed metres outside the stadium centerline. Negative is the infield.
pub fn track_offset(x: f32, z: f32) -> f32 {
    if z <= TRACK_ZS {
        let dx = x;
        let dz = z - TRACK_ZS;
        (dx * dx + dz * dz).sqrt() - TRACK_R
    } else if z >= TRACK_ZN {
        let dx = x;
        let dz = z - TRACK_ZN;
        (dx * dx + dz * dz).sqrt() - TRACK_R
    } else {
        x.abs() - TRACK_R
    }
}

/// West-straight ski jump. True while the chassis is on the face or the lip.
pub fn on_launch_crest(x: f32, z: f32) -> bool {
    (x + 6.6).abs() < 1.25 && (0.4..2.55).contains(&z)
}

fn launch_runup(x: f32, z: f32) -> bool {
    (x + 6.6).abs() < 2.8 && (-5.6..2.35).contains(&z)
}

/// Climbable face, short flat lip, then a drop into the west puddle.
fn launch_kicker(x: f32, z: f32) -> f32 {
    let toe = 0.25_f32;
    let lip = 2.05_f32;
    let along = (z - toe) / (lip - toe);
    if !(0.0..1.22).contains(&along) {
        return 0.0;
    }
    let ax = 1.0 - ((x + 6.6).abs() / 1.45).clamp(0.0, 1.0);
    let ax = ax * ax * (3.0 - 2.0 * ax);
    let peak = 0.55_f32;
    let face = if along < 0.78 {
        let u = along / 0.78;
        let u = u * u * (3.0 - 2.0 * u);
        u * peak
    } else if along < 1.0 {
        peak
    } else {
        let u = ((along - 1.0) / 0.22).clamp(0.0, 1.0);
        peak * (1.0 - u * u)
    };
    face * ax
}

fn smooth01(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn gauss(x: f32, z: f32, cx: f32, cz: f32, rad: f32) -> f32 {
    let dx = x - cx;
    let dz = z - cz;
    (-(dx * dx + dz * dz) / (rad * rad).max(0.05)).exp()
}

/// Elliptical bowl. `rx`/`rz` are the zero-depth radii.
fn bowl(x: f32, z: f32, cx: f32, cz: f32, rx: f32, rz: f32, depth: f32) -> f32 {
    let dx = (x - cx) / rx.max(0.2);
    let dz = (z - cz) / rz.max(0.2);
    let d = (dx * dx + dz * dz).sqrt();
    (1.0 - d.clamp(0.0, 1.0)).powf(1.35) * depth
}

/// Wide floor and a soft shore, so a straight dig sinks instead of skipping the rim.
fn bog_bowl(x: f32, z: f32, cx: f32, cz: f32, rad: f32, depth: f32) -> f32 {
    let d = ((x - cx).powi(2) + (z - cz).powi(2)).sqrt() / rad.max(0.2);
    if d >= 1.0 {
        return 0.0;
    }
    let shore = smooth01(1.0, 0.8, d);
    let floor = smooth01(0.8, 0.4, d);
    (shore * 0.35 + floor * 0.65) * depth
}

fn hash2(x: f32, y: f32) -> f32 {
    let s = (x * 127.1 + y * 311.7).sin() * 43758.5453;
    s - s.floor()
}

fn noise2(x: f32, y: f32) -> f32 {
    let xi = x.floor();
    let yi = y.floor();
    let xf = x - xi;
    let yf = y - yi;
    let u = xf * xf * (3.0 - 2.0 * xf);
    let v = yf * yf * (3.0 - 2.0 * yf);
    let a = hash2(xi, yi);
    let b = hash2(xi + 1.0, yi);
    let c = hash2(xi, yi + 1.0);
    let d = hash2(xi + 1.0, yi + 1.0);
    let ab = a + (b - a) * u;
    let cd = c + (d - c) * u;
    ab + (cd - ab) * v
}

fn fbm2(x: f32, y: f32) -> f32 {
    let mut amp = 0.5;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..4 {
        sum += amp * noise2(x * freq, y * freq);
        norm += amp;
        amp *= 0.5;
        freq *= 2.02;
    }
    sum / norm
}

/// Inverse Bekker: sinkage that supports `load` on this soil.
pub fn sinkage_for_load(load: f32, softness: f32, radius: f32, width: f32) -> f32 {
    let tm = terramech(softness);
    let mut lo = 0.0;
    let mut hi = radius * 0.9;
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        let w = bekker_load(mid, radius, width, tm.k_eq, tm.n);
        if w < load {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plow_keeps_firm_ground_and_the_clay_volume() {
        for kind in [
            SurfaceKind::Gravel,
            SurfaceKind::Sand,
            SurfaceKind::Silt,
            SurfaceKind::DryCrust,
            SurfaceKind::Grass,
        ] {
            let mut soil = SoilField::flat(0.16);
            for cell in &mut soil.kind {
                *cell = kind as u8;
            }
            let cut = soil.plow_contact(
                Vec3::new(0.0, -0.08, 0.0),
                Vec3::Z,
                Vec3::X,
                0.9,
                0.45,
                -0.08,
                Vec3::X,
                0.35,
            );
            assert!(cut.bed > 1.0e-3, "{kind:?} bed {}", cut.bed);
            assert!(
                (cut.bed + cut.moved - cut.berm - cut.thrown).abs() < 1.0e-4,
                "{kind:?} split bed {} moved {} berm {} thrown {}",
                cut.bed,
                cut.moved,
                cut.berm,
                cut.thrown
            );
            let v = soil.volumes();
            let gap = (v.cut - v.placed - soil.airborne).abs();
            assert!(
                gap < 1.0e-4,
                "{kind:?} deleted volume cut {} placed {} air {gap}",
                v.cut,
                v.placed
            );
            assert_eq!(soil.sample(0.0, 0.0).kind(), kind, "{kind:?} was retagged");
            let shoulder = soil.sample(0.36, 0.0);
            assert_eq!(shoulder.kind(), kind, "{kind:?} shoulder changed");
            assert!(
                shoulder.placed > 0.0 || soil.airborne > 0.0,
                "{kind:?} spoil vanished"
            );
        }

        let mut clay = SoilField::mud_pad();
        let cut = clay.plow_contact(
            Vec3::new(0.0, -0.1, 0.0),
            Vec3::Z,
            Vec3::X,
            1.1,
            0.5,
            -0.1,
            Vec3::X,
            0.4,
        );
        assert!(cut.bed > 0.01, "clay bed {}", cut.bed);
        assert!(cut.thrown > 1.0e-3, "clay threw {}", cut.thrown);
        assert!(cut.berm > 1.0e-3, "clay berm {}", cut.berm);
        assert_eq!(clay.sample(0.0, 0.0).kind(), SurfaceKind::FreshCut);
        let v = clay.volumes();
        let gap = (v.cut - v.placed - clay.airborne).abs();
        assert!(
            gap < v.cut * 0.02 + 1.0e-5,
            "clay volume cut {} placed {} air {} gap {gap}",
            v.cut,
            v.placed,
            clay.airborne
        );
        let landed = clay.airborne;
        let spot = clay.receive_thrown(0.9, 0.2, landed).expect("landing");
        assert!(clay.airborne < 1.0e-6, "airborne left {}", clay.airborne);
        assert!(
            (spot.x - 0.0).hypot(spot.z - 0.0) > 0.25,
            "thrown mass landed in the cut at {spot:?}"
        );
        let v = clay.volumes();
        let gap = (v.cut - v.placed - clay.airborne).abs();
        assert!(gap < v.cut * 0.02 + 1.0e-5, "landing deleted volume {gap}");
        assert!(
            clay.sample(0.0, 0.0).rut > 0.06,
            "the cut filled, rut {}",
            clay.sample(0.0, 0.0).rut
        );
    }

    #[test]
    fn bekker_grows_with_sinkage() {
        let r = 0.38;
        let w = 0.28;
        let a = bekker_load(0.02, r, w, 200_000.0, 0.8);
        let b = bekker_load(0.08, r, w, 200_000.0, 0.8);
        let c = bekker_load(0.16, r, w, 200_000.0, 0.8);
        assert!(a > 0.0);
        assert!(b > a * 1.5);
        assert!(c > b);
    }

    #[test]
    fn janosi_saturates() {
        let small = janosi_hanamoto(4_000.0, 10_000.0, 0.3, 0.001, 0.03);
        let large = janosi_hanamoto(4_000.0, 10_000.0, 0.3, 0.5, 0.03);
        let tau_max = 4_000.0 + 10_000.0 * 0.3_f32.tan();
        assert!(small < large);
        assert!(large < tau_max);
        assert!(large > tau_max * 0.9);
    }

    #[test]
    fn force_budget_allows_drive_and_bog() {
        for s in [0.2_f32, 0.4, 0.55, 0.7, 0.85, 1.0] {
            let z = sinkage_for_load(1_600.0, s, 0.38, 0.28);
            let tm = terramech(s);
            let len = contact_length(0.38, z.max(0.01));
            let area = 0.28 * len;
            let sigma = 1_600.0 / area.max(0.01);
            let tau = tm.cohesion + sigma * tm.phi.tan();
            let f_trac = tau * area;
            let f_y = tm.yield_stress * area;
            eprintln!(
                "s {s:.2} z {z:.3} area {area:.3} Ftrac {f_trac:.0} Fyield {f_y:.0} net {:.0}",
                f_trac - f_y
            );
        }
    }

    #[test]
    fn sinkage_table_is_finite() {
        for s in [0.1_f32, 0.5, 0.9] {
            let z = sinkage_for_load(1_600.0, s, 0.38, 0.28);
            assert!(
                z.is_finite() && z > 0.005 && z < 0.38,
                "softness {s} -> z {z}"
            );
        }
        let firm = sinkage_for_load(1_600.0, 0.15, 0.38, 0.28);
        let mid = sinkage_for_load(1_600.0, 0.5, 0.38, 0.28);
        let soft = sinkage_for_load(1_600.0, 0.95, 0.38, 0.28);
        eprintln!("bekker z firm {firm:.3} mid {mid:.3} soft {soft:.3}");
        assert!(soft > firm * 1.4, "firm {firm} soft {soft}");
    }

    #[test]
    fn puddle_holds_water() {
        let soil = SoilField::demo();
        let total: f32 = soil.water.iter().sum();
        assert!(total > 0.4, "water volume proxy {total}");
        let mut soil = soil;
        for _ in 0..30 {
            soil.step_water(1.0 / 60.0);
        }
        assert!(soil.water.iter().all(|w| *w >= 0.0 && w.is_finite()));
        let after: f32 = soil.water.iter().sum();
        assert!(
            after > total * 0.85,
            "water drained too fast {total} -> {after}"
        );
    }

    #[test]
    fn states_differ_in_grip_sink_and_shear() {
        let s = 0.45;
        let mud = terramech_for(SurfaceKind::WetMud, s, 0.0);
        let gravel = terramech_for(SurfaceKind::Gravel, s, 0.0);
        let crust = terramech_for(SurfaceKind::DryCrust, s, 0.0);
        let grass = terramech_for(SurfaceKind::Grass, s, 0.0);
        let cut = terramech_for(SurfaceKind::FreshCut, s, 0.0);
        let lip = terramech_for(SurfaceKind::CutLip, s, 0.0);
        let berm = terramech_for(SurfaceKind::Berm, s, 0.0);
        let clod = terramech_for(SurfaceKind::PlacedClod, s, 0.0);
        let packed = terramech_for(SurfaceKind::PackedClay, s, 0.0);
        assert!(gravel.k_eq > berm.k_eq * 1.8, "gravel should not sink like a berm");
        assert!(crust.k_eq > cut.k_eq * 1.3, "crust should not sink like a fresh cut");
        assert!(packed.k_eq > cut.k_eq, "packed clay is stiffer than the fresh cut");
        assert!(
            (lip.k_eq - berm.k_eq).abs() / lip.k_eq > 0.25,
            "cut lip and placed berm must not share a modulus"
        );
        assert!(berm.shear_k > lip.shear_k + 0.02, "loose berm shears longer than the lip");
        assert!(gravel.phi > cut.phi + 0.3, "gravel grips, a fresh cut is slick");
        assert!(grass.cohesion > berm.cohesion * 2.0, "roots hold more than spoil");
        assert!((clod.shear_k - berm.shear_k).abs() > 0.01);
        assert!(
            (mud.k_eq - terramech(s).k_eq).abs() < 1.0,
            "dry wet-mud stays the lane curve"
        );
        let film = terramech_for(SurfaceKind::PackedClay, s, 0.02);
        let slurry = terramech_for(SurfaceKind::WetMud, 0.9, 0.12);
        let dry_mud = terramech_for(SurfaceKind::WetMud, 0.9, 0.0);
        assert!(film.phi < packed.phi);
        assert!(slurry.k_eq < dry_mud.k_eq);
        assert!(liquid_regime(SurfaceKind::Gravel, 0.08) == LiquidRegime::Film);
        assert!(liquid_regime(SurfaceKind::WetMud, 0.1) == LiquidRegime::Slurry);
        assert!(liquid_regime(SurfaceKind::DryCrust, 0.012) == LiquidRegime::Film);
        assert!(water_hold(SurfaceKind::FreshCut) > water_hold(SurfaceKind::CutLip));
        assert!(water_hold(SurfaceKind::PackedClay) > water_hold(SurfaceKind::Gravel));
        assert!(water_conduct(SurfaceKind::Gravel) > water_conduct(SurfaceKind::PackedClay));
        let sand = terramech_for(SurfaceKind::Sand, s, 0.0);
        let silt = terramech_for(SurfaceKind::Silt, s, 0.0);
        assert!(
            gravel.k_eq > sand.k_eq * 1.4,
            "gravel should not sink like sand"
        );
        assert!(sand.k_eq > silt.k_eq, "wet-leaning silt sinks more than sand");
        assert!(sand.phi > silt.phi + 0.15, "sand grips where silt is slick");
        assert!(silt.cohesion > sand.cohesion, "silt sticks more than sand");
        assert_eq!(liquid_regime(SurfaceKind::Sand, 0.1), LiquidRegime::Film);
        assert_eq!(liquid_regime(SurfaceKind::Silt, 0.1), LiquidRegime::Slurry);
        assert!(water_conduct(SurfaceKind::Gravel) > water_conduct(SurfaceKind::Sand));
        assert!(water_conduct(SurfaceKind::Sand) > water_conduct(SurfaceKind::Silt));
        assert!(water_hold(SurfaceKind::Silt) > water_hold(SurfaceKind::Sand));
        assert!(water_hold(SurfaceKind::Sand) > water_hold(SurfaceKind::Gravel));
        let dry_sand = terramech_for(SurfaceKind::Sand, 0.08, 0.0);
        let wet_sand = terramech_for(SurfaceKind::Sand, 0.9, 0.0);
        let dry_silt = terramech_for(SurfaceKind::Silt, 0.08, 0.0);
        let wet_silt = terramech_for(SurfaceKind::Silt, 0.9, 0.0);
        assert!(dry_sand.phi > wet_sand.phi, "sand loses grip as it wets");
        assert!(dry_silt.phi > wet_silt.phi + 0.2, "silt goes slick when wet");
        assert!(wet_silt.cohesion > dry_silt.cohesion, "wet silt sticks");
        assert!(dry_sand.k_eq > wet_sand.k_eq, "saturated sand sinks more");
        assert_eq!(SURFACE_STATE_COUNT, 14);
        let load = 4_000.0;
        let dry_rr = rolling_resistance(load, 0.0);
        let film_rr = rolling_resistance(load, 0.03);
        let slurry_rr = rolling_resistance(load, 0.15);
        assert!(film_rr > dry_rr * 1.4, "a film should drag more than dry ground");
        assert!(slurry_rr > film_rr * 2.0, "slurry should drag more than a film");
        let float_film = wheel_buoyancy(0.03, 0.38, 0.28);
        let float_slurry = wheel_buoyancy(0.15, 0.38, 0.28);
        assert!(wheel_buoyancy(0.0, 0.38, 0.28) < 1.0);
        assert!(float_film > 5.0, "a film should lift the tire a little, {float_film}");
        assert!(
            float_slurry > float_film * 3.0,
            "deeper water should float the tire more, {float_film} -> {float_slurry}"
        );
    }

    #[test]
    fn placed_spoil_is_not_the_cut_and_volume_holds() {
        let mut soil = SoilField::flat(0.7);
        let before_cut = soil.cut_volume();
        let before_placed = soil.placed_volume();
        soil.stamp_wheel(
            Vec3::new(0.2, 0.30, -0.4),
            0.38,
            0.28,
            Vec3::Z,
            Vec3::X,
            0.08,
            0.4,
            0.12,
            true,
            0.0,
            0.0,
        );
        let cut = soil.cut_volume() - before_cut;
        let placed = soil.placed_volume() - before_placed;
        assert!(cut > 1.0e-4, "stamp did not cut, {cut}");
        assert!(
            (cut - placed).abs() < cut * 0.02 + 1.0e-5,
            "volume drifted cut {cut} placed {placed}"
        );
        let mut lip = 0;
        let mut berm = 0;
        let mut clod = 0;
        let mut lip_placed = 0.0;
        let mut berm_placed = 0.0;
        let mut step = 0.0_f32;
        for j in 1..soil.res - 1 {
            for i in 1..soil.res - 1 {
                let idx = j * soil.res + i;
                match SurfaceKind::from_u8(soil.kind[idx]) {
                    SurfaceKind::CutLip => {
                        lip += 1;
                        lip_placed += soil.placed[idx];
                        let floor = idx - 1;
                        if SurfaceKind::from_u8(soil.kind[floor]) == SurfaceKind::FreshCut
                            || SurfaceKind::from_u8(soil.kind[floor]) == SurfaceKind::PackedClay
                        {
                            step = step.max(soil.height[idx] - soil.height[floor]);
                        }
                    }
                    SurfaceKind::Berm => {
                        berm += 1;
                        berm_placed += soil.placed[idx];
                    }
                    SurfaceKind::PlacedClod => clod += 1,
                    _ => {}
                }
            }
        }
        assert!(lip > 0, "no cut lip");
        assert!(berm > 0, "no placed berm");
        assert!(clod > 0, "no crest clods");
        assert!(berm_placed > lip_placed, "the berm should hold the spoil");
        assert!(step > 0.02, "shoulder was smoothed, step {step}");
    }

    #[test]
    fn repeated_passes_leave_a_step_and_keep_the_volume() {
        let mut soil = SoilField::flat(0.7);
        let mut z = -1.4;
        while z <= 1.4 {
            soil.stamp_wheel(
                Vec3::new(0.0, 0.30, z),
                0.38,
                0.28,
                Vec3::Z,
                Vec3::X,
                0.08,
                0.35,
                0.12,
                true,
                0.0,
                0.0,
            );
            z += 0.1;
        }
        let cut = soil.cut_volume();
        let placed = soil.placed_volume();
        assert!(
            (cut - placed).abs() < cut * 0.02 + 1.0e-5,
            "volume drifted cut {cut} placed {placed}"
        );
        let mut max_clod = 0.0_f32;
        let mut max_berm = 0.0_f32;
        let mut step = 0.0_f32;
        let mut lip = 0;
        let mut notches = 0;
        for j in 1..soil.res - 1 {
            for i in 1..soil.res - 1 {
                let idx = j * soil.res + i;
                match SurfaceKind::from_u8(soil.kind[idx]) {
                    SurfaceKind::PlacedClod => max_clod = max_clod.max(soil.placed[idx]),
                    SurfaceKind::Berm => max_berm = max_berm.max(soil.placed[idx]),
                    SurfaceKind::CutLip => {
                        lip += 1;
                        let floor = idx.saturating_sub(1);
                        let floor_kind = SurfaceKind::from_u8(soil.kind[floor]);
                        if matches!(
                            floor_kind,
                            SurfaceKind::FreshCut | SurfaceKind::PackedClay | SurfaceKind::WetMud
                        ) {
                            let rise = soil.height[idx] - soil.height[floor];
                            step = step.max(rise);
                            if soil.bed[idx] < soil.initial_height[idx] - 0.02 {
                                notches += 1;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        assert!(lip > 0, "the lip was smoothed away");
        assert!(notches > 0, "the crest was not broken");
        assert!(max_clod > 0.01 && max_clod <= CREST_CLOD_CAP + 1.0e-3, "clod {max_clod}");
        assert!(max_berm > 0.01, "berm did not take the spoil, {max_berm}");
        assert!(step > 0.02, "shoulder was smoothed, step {step}");
        let gallery = SoilField::gallery();
        let gcut = gallery.cut_volume();
        let gplaced = gallery.placed_volume();
        assert!(
            (gcut - gplaced).abs() < gcut * 0.02 + 1.0e-4,
            "gallery volume drifted cut {gcut} placed {gplaced}"
        );
        assert_eq!(
            gallery.sample(0.0, 0.0).kind(),
            SurfaceKind::FreshCut,
            "gallery floor should stay a fresh cut"
        );
        let mut saw_lip = false;
        let mut saw_berm = false;
        let mut saw_clod = false;
        for x100 in -120..120 {
            let s = gallery.sample(x100 as f32 * 0.01, 0.05);
            match s.kind() {
                SurfaceKind::CutLip => saw_lip = true,
                SurfaceKind::Berm => saw_berm = true,
                SurfaceKind::PlacedClod => saw_clod = true,
                _ => {}
            }
        }
        assert!(saw_lip && saw_berm && saw_clod, "gallery cross-section lost the lip");
        let berm = gallery.sample(0.55, 0.0);
        assert_eq!(berm.kind(), SurfaceKind::Berm);
        assert!(
            berm.placed > 0.05,
            "gallery berm is too flat to sit beside the lip, {}",
            berm.placed
        );
        let film = gallery.sample(0.0, -1.7);
        assert_eq!(film.kind(), SurfaceKind::PackedClay);
        assert_eq!(
            liquid_regime(film.kind(), film.water),
            LiquidRegime::Film,
            "clay in the rut should hold a film, water {}",
            film.water
        );
        let slurry = gallery.sample(0.0, -0.7);
        assert_eq!(slurry.kind(), SurfaceKind::FreshCut);
        assert_eq!(
            liquid_regime(slurry.kind(), slurry.water),
            LiquidRegime::Slurry,
            "the fresh cut should hold slurry, water {}",
            slurry.water
        );
        assert!(
            gallery.sample(-4.2, 0.2).water < 0.004,
            "water left the rut"
        );
        assert!(gallery.sample(0.7, -1.0).water < 0.004, "the berm is wet");
    }

    /// Bed depth and placed volume of one force pass on the wet-clay pad.
    /// The mesh height is `bed + placed + slurry` on every column.
    fn force_pass(normal: f32, push: f32) -> (f32, f32, f32) {
        let mut soil = SoilField::mud_pad();
        let kind = SurfaceKind::WetMud;
        let want = rut_depth_from_force(normal, push, 0.38, 0.28, kind, 0.82, 0.02, 0.0);
        soil.roll_with_force(
            Vec3::new(0.0, 0.0, -1.2),
            Vec3::new(0.0, 0.0, 1.2),
            want.max(0.012),
            0.0,
            normal,
            push,
        );
        for i in 0..soil.height.len() {
            let field = soil.bed[i] + soil.placed[i] + soil.slurry[i];
            assert!(
                (soil.height[i] - field).abs() < 1.0e-5,
                "mesh height left the field at {i}: {:.5} vs {:.5}",
                soil.height[i],
                field
            );
        }
        let depth = soil.max_rut_depth();
        assert!(
            (depth - want).abs() < 0.012,
            "bed {depth:.4} m is not the force depth {want:.4} m"
        );
        let v = soil.volumes();
        let gap = (v.cut - v.placed).abs();
        assert!(
            gap < v.cut * 0.02 + 1.0e-5,
            "displaced {0:.4} placed {1:.4}",
            v.cut,
            v.placed
        );
        let mut towers = 0;
        let mut crest = 0.0_f32;
        let mut crest_n = 0.0_f32;
        for idx in 0..soil.placed.len() {
            let h = soil.placed[idx];
            if h < 0.02 {
                continue;
            }
            let i = (idx % soil.res) as isize;
            let j = (idx / soil.res) as isize;
            let mut best = 0.0_f32;
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                let ii = i + di;
                let jj = j + dj;
                if ii < 0 || jj < 0 || ii >= soil.res as isize || jj >= soil.res as isize {
                    continue;
                }
                best = best.max(soil.placed[jj as usize * soil.res + ii as usize]);
            }
            if best < h * 0.55 {
                towers += 1;
            }
            if h > crest {
                crest = h;
                crest_n = best;
            }
        }
        assert_eq!(towers, 0, "the berm is still a row of towers");
        assert!(
            crest_n > crest * 0.7,
            "berm crest is a spike, {crest:.3} m beside {crest_n:.3} m"
        );
        (depth, v.placed, v.cut)
    }

    #[test]
    fn hard_push_cuts_deeper_and_places_more_than_a_light_pass() {
        let (light_depth, light_placed, _) = force_pass(LIGHT_PASS_NORMAL, LIGHT_PASS_PUSH);
        let (hard_depth, hard_placed, _) = force_pass(HARD_PASS_NORMAL, HARD_PASS_PUSH);
        eprintln!(
            "light depth {light_depth:.4} m  placed {light_placed:.4} m3   \
             hard depth {hard_depth:.4} m  placed {hard_placed:.4} m3"
        );
        assert!(
            hard_depth > light_depth + 0.02,
            "hard bed {hard_depth:.4} m is not deeper than light {light_depth:.4} m"
        );
        assert!(
            hard_placed > light_placed * 1.4,
            "hard placed {hard_placed:.4} m3 did not grow from light {light_placed:.4} m3"
        );
    }

    fn volumes_match(soil: &SoilField, label: &str) -> SoilVolumes {
        let v = soil.volumes();
        eprintln!(
            "{label} cut {:.4} m3  placed {:.4} m3  berm {:.4} m3  clods {:.4} m3  rut {:.4} m  berm height {:.4} m",
            v.cut, v.placed, v.berm, v.clods, v.max_rut, v.max_berm
        );
        assert!(
            (v.cut - v.placed).abs() < v.cut * 0.02 + 1.0e-5,
            "{label} volume drifted cut {} placed {}",
            v.cut,
            v.placed
        );
        assert!(v.berm > v.clods, "{label} berm did not receive the spoil");
        assert!(v.berm > v.cut * 0.5, "{label} berm holds too little of the cut");
        v
    }

    #[test]
    fn cut_soil_lands_on_the_berm() {
        let soil = SoilField::berm_scene(false);
        let v = volumes_match(&soil, "berm");
        assert!(v.max_rut > 0.04, "rut {}", v.max_rut);
        assert!(v.max_berm > 0.02, "berm height {}", v.max_berm);
        assert!(v.cut > 1.0e-3, "cut {}", v.cut);
    }

    #[test]
    fn second_pass_deepens_the_rut_and_raises_the_berm() {
        let mut soil = SoilField::berm_scene(false);
        let first = volumes_match(&soil, "pass1");
        soil.roll_lane(
            Vec3::new(0.0, 0.0, -1.15),
            Vec3::new(0.0, 0.0, 1.15),
            0.16,
            0.85,
        );
        let second = volumes_match(&soil, "pass2");
        assert!(
            second.max_rut > first.max_rut + 0.025,
            "second pass did not dig, {} -> {}",
            first.max_rut,
            second.max_rut
        );
        assert!(
            second.max_berm > first.max_berm + 0.012,
            "berm did not rise, {} -> {}",
            first.max_berm,
            second.max_berm
        );
        assert!(second.cut > first.cut, "cut volume did not grow");
        assert!(second.berm > first.berm, "berm volume did not grow");
    }

    fn straight_lane(soil: &mut SoilField) {
        soil.roll_lane(
            Vec3::new(0.0, 0.0, -1.15),
            Vec3::new(0.0, 0.0, 1.15),
            0.09,
            0.0,
        );
    }

    #[test]
    fn rut_persists_across_a_second_pass_and_a_reload() {
        use crate::world::World;

        let mut soil = SoilField::mud_pad();
        straight_lane(&mut soil);
        let idx = soil.idx_at(0.0, 0.0).expect("lane cell");
        let pass1 = soil.rut_near(0.0, 0.0);
        let bed1 = soil.bed[idx];
        let v1 = volumes_match(&soil, "pass1");
        assert!(
            (0.04..0.07).contains(&pass1),
            "pass 1 should be the fresh bite, depth {pass1}"
        );
        assert_eq!(soil.passes[idx], 1, "the first visit should count as one pass");
        let saved = soil.save_state();

        // The tire is still in the slot. That must not refill it, and it
        // must not pretend the overlap is a second pass.
        straight_lane(&mut soil);
        let replay = soil.rut_near(0.0, 0.0);
        assert!(
            soil.bed[idx] <= bed1 + 1.0e-5,
            "same visit filled the cut back, bed {bed1} -> {}",
            soil.bed[idx]
        );
        assert!(
            replay + 1.0e-4 >= pass1,
            "same visit wiped the rut {pass1} -> {replay}"
        );
        assert!(
            replay < pass1 + 0.02,
            "same visit counted as another pass {pass1} -> {replay}"
        );

        soil.age_passes(0.55);
        straight_lane(&mut soil);
        let pass2 = soil.rut_near(0.0, 0.0);
        let bed2 = soil.bed[idx];
        let v2 = volumes_match(&soil, "pass2");
        assert!(
            pass2 > pass1 + 0.04,
            "second pass did not deepen the rut: {pass1} -> {pass2}"
        );
        assert!(v2.cut > v1.cut + 1.0e-4, "second pass added no spoil");
        assert!(v2.berm > v1.berm, "berm did not take the extra cut");
        assert!(v2.max_berm + 1.0e-5 >= v1.max_berm, "berm height fell");
        assert_eq!(soil.passes[idx], 2);

        // A shallow stamp after the deep cut must leave the deep cut.
        straight_lane(&mut soil);
        assert!(
            soil.bed[idx] <= bed2 + 1.0e-5,
            "a later stamp filled the deep rut back to a shallow groove"
        );
        assert!(
            soil.rut_near(0.0, 0.0) + 1.0e-4 >= pass2,
            "deep rut was wiped"
        );

        let loaded = SoilField::load_state(&saved).expect("reload");
        let reload_depth = loaded.rut_near(0.0, 0.0);
        let reload = volumes_match(&loaded, "reload");
        assert!(
            (reload_depth - pass1).abs() < 1.0e-4,
            "reload lost pass 1: {pass1} -> {reload_depth}"
        );
        assert!(
            (reload.cut - v1.cut).abs() < 1.0e-5 && (reload.berm - v1.berm).abs() < 1.0e-5,
            "reload volumes drifted"
        );
        assert_eq!(loaded.passes[idx], 1);
        assert!(
            (loaded.bed[idx] - bed1).abs() < 1.0e-5,
            "reload bed moved"
        );
        assert!(reload_depth + 0.03 < pass2, "reload kept the second pass");

        let mut world = World::flat(0.45);
        let home = world.vehicle.pos;
        world.soil = SoilField::load_state(&saved).expect("re-init");
        world.vehicle.pos = Vec3::new(8.0, 1.0, 8.0);
        world.vehicle.yaw = 1.2;
        world.reset_vehicle();
        let kept = world.soil.rut_near(0.0, 0.0);
        let kept_v = world.soil.volumes();
        assert!(
            (kept - pass1).abs() < 1.0e-4,
            "reset cleared the rut: {pass1} -> {kept}"
        );
        assert!((kept_v.cut - v1.cut).abs() < 1.0e-5);
        assert!((kept_v.berm - v1.berm).abs() < 1.0e-5);
        assert!(
            (world.vehicle.pos.x - home.x).abs() < 0.05
                && (world.vehicle.pos.z - home.z).abs() < 0.05,
            "vehicle did not return home, pos {:?}",
            world.vehicle.pos
        );
        eprintln!(
            "persist pass1 depth {pass1:.4} m  cut {c1:.4}  berm {b1:.4}  clods {k1:.4}  \
             pass2 depth {pass2:.4} m  cut {c2:.4}  berm {b2:.4}  clods {k2:.4}  \
             reload depth {reload_depth:.4} m  cut {c3:.4}  berm {b3:.4}  clods {k3:.4}",
            c1 = v1.cut,
            b1 = v1.berm,
            k1 = v1.clods,
            c2 = v2.cut,
            b2 = v2.berm,
            k2 = v2.clods,
            c3 = reload.cut,
            b3 = reload.berm,
            k3 = reload.clods,
        );
    }

    #[test]
    fn rut_on_a_slope_stays_in_world_columns() {
        let mut soil = SoilField::slope_scene();
        let v = volumes_match(&soil, "slope");
        assert!(v.max_rut > 0.03, "slope rut {}", v.max_rut);
        let half = soil.size * 0.5;
        let res = soil.res;
        let cell = soil.cell;
        let x_of = |idx: usize| {
            let i = idx % res;
            (i as f32 + 0.5) * cell - half
        };
        let mut max_x = f32::MIN;
        let mut upper = 0.0_f32;
        for idx in 0..soil.bed.len() {
            let dug = soil.initial_height[idx] - soil.bed[idx];
            if dug > 0.02 {
                max_x = max_x.max(x_of(idx));
            }
            if x_of(idx) > 0.05 {
                upper = upper.max(dug);
            }
        }
        assert!(
            max_x < -0.15,
            "rut climbed the slope, leading edge at x {max_x}"
        );
        assert!(
            upper < 0.01,
            "soil above the pass was cut, dug {upper}"
        );
        let mean_bed = |x0: f32, x1: f32, zc: f32| {
            let mut acc = 0.0_f32;
            let mut n = 0.0_f32;
            for j in 0..res {
                for i in 0..res {
                    let x = (i as f32 + 0.5) * cell - half;
                    let z = (j as f32 + 0.5) * soil.cell - half;
                    if x < x0 || x > x1 || (z - zc).abs() > 0.16 {
                        continue;
                    }
                    let idx = j * soil.res + i;
                    if soil.initial_height[idx] - soil.bed[idx] > 0.02 {
                        acc += soil.bed[idx];
                        n += 1.0;
                    }
                }
            }
            assert!(n > 0.0, "no floor between {x0} and {x1}");
            acc / n
        };
        let low = mean_bed(-1.5, -1.2, 0.74);
        let high = mean_bed(-0.85, -0.55, 0.74);
        assert!(
            high > low + 0.15,
            "floor is a shelf, not the slope: {low} -> {high}"
        );
        let mut snap = Vec::new();
        for idx in 0..soil.bed.len() {
            if x_of(idx) < -0.55 {
                snap.push((idx, soil.bed[idx], soil.placed[idx]));
            }
        }
        soil.roll_lane(
            Vec3::new(0.95, 0.0, 0.74),
            Vec3::new(1.55, 0.0, 0.74),
            0.09,
            0.3,
        );
        for (idx, bed, placed) in snap {
            assert!(
                (soil.bed[idx] - bed).abs() < 1.0e-5 && (soil.placed[idx] - placed).abs() < 1.0e-5,
                "column {} moved after the wheel climbed",
                x_of(idx)
            );
        }
        let mut upper_rut = 0.0_f32;
        for idx in 0..soil.bed.len() {
            if x_of(idx) > 1.05 {
                upper_rut = upper_rut.max(soil.initial_height[idx] - soil.bed[idx]);
            }
        }
        assert!(upper_rut > 0.02, "the upper pass did not cut");
        let after = volumes_match(&soil, "slope-after");
        assert!(after.cut > v.cut, "the upper pass added no volume");
    }

    fn clay_response(soil: &SoilField) -> (f32, f32, f32, SurfaceKind) {
        let s = soil.sample(0.78, 0.0);
        let mu = grip_mu(s.kind(), s.wetness, s.loose, s.water);
        let sink = sinkage_on(1_600.0, s.kind(), s.wetness, s.loose, s.water, 0.38, 0.28);
        let stick = stickiness(s.kind(), s.wetness, s.loose, s.water);
        (mu, sink, stick, s.kind())
    }

    #[test]
    fn wet_clay_dries_to_a_crust_and_rain_wets_it_again() {
        let mut soil = SoilField::moisture_scene();
        let bed = soil.bed.clone();
        let placed = soil.placed.clone();
        let before = soil.volumes();
        assert!(before.cut > 1.0e-3, "scene did not cut");
        let (wet_mu, wet_sink, wet_stick, wet_kind) = clay_response(&soil);
        assert_eq!(wet_kind, SurfaceKind::WetMud, "shoulder should start as wet clay");
        eprintln!(
            "moisture wet  mu {wet_mu:.3}  sink {wet_sink:.4} m  stick {wet_stick:.0} Pa  kind {wet_kind:?}"
        );

        soil.advance_weather(150.0, 0.0);
        for (a, b) in bed.iter().zip(soil.bed.iter()) {
            assert!((a - b).abs() < 1.0e-6, "drying moved the bed");
        }
        for (a, b) in placed.iter().zip(soil.placed.iter()) {
            assert!((a - b).abs() < 1.0e-6, "drying moved the berm");
        }
        let after = soil.volumes();
        assert!((after.cut - before.cut).abs() < 1.0e-4, "cut volume changed");
        assert!(
            (after.placed - before.placed).abs() < 1.0e-4,
            "placed volume changed"
        );

        let mut floor_cells = 0;
        let mut floor_crust = 0;
        let half = soil.size * 0.5;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                if x.abs() > 0.22 || z.abs() > 1.0 {
                    continue;
                }
                let idx = j * soil.res + i;
                if soil.initial_height[idx] - soil.bed[idx] < 0.02 {
                    continue;
                }
                floor_cells += 1;
                if SurfaceKind::from_u8(soil.kind[idx]) == SurfaceKind::DryCrust {
                    floor_crust += 1;
                }
            }
        }
        assert!(
            floor_cells > 0 && floor_crust * 2 > floor_cells,
            "the rut floor should dry, crust {floor_crust} of {floor_cells}"
        );
        let (dry_mu, dry_sink, dry_stick, dry_kind) = clay_response(&soil);
        eprintln!(
            "moisture dry  mu {dry_mu:.3}  sink {dry_sink:.4} m  stick {dry_stick:.0} Pa  kind {dry_kind:?}"
        );
        eprintln!(
            "grip change wet {wet_mu:.3} -> dry {dry_mu:.3}  delta {:+.3}",
            dry_mu - wet_mu
        );
        assert_eq!(dry_kind, SurfaceKind::DryCrust, "wet clay should bake to a crust");
        assert!(
            dry_mu > wet_mu * 1.25,
            "grip should rise as clay dries, {wet_mu} -> {dry_mu}"
        );
        assert!(
            dry_sink < wet_sink * 0.85,
            "a crust should sink less, {wet_sink} -> {dry_sink}"
        );
        assert!(
            dry_stick < wet_stick,
            "a crust should stick less, {wet_stick} -> {dry_stick}"
        );
        assert_eq!(soil.sample(1.6, 0.0).kind(), SurfaceKind::Sand);
        assert_eq!(soil.sample(-1.7, 0.0).kind(), SurfaceKind::Silt);
        assert_eq!(soil.sample(1.7, 1.9).kind(), SurfaceKind::Gravel);
        let sand_w = soil.sample(1.6, 0.0).wetness;
        let silt_w = soil.sample(-1.7, 0.0).wetness;
        assert!(
            silt_w > sand_w + 0.12,
            "silt should stay damper than sand, sand {sand_w} silt {silt_w}"
        );

        let gravel_mu = grip_mu(SurfaceKind::Gravel, 0.05, 0.0, 0.0);
        soil.advance_weather(10.0, 0.008);
        let (rain_mu, rain_sink, rain_stick, rain_kind) = clay_response(&soil);
        eprintln!(
            "moisture rain mu {rain_mu:.3}  sink {rain_sink:.4} m  stick {rain_stick:.0} Pa  kind {rain_kind:?}"
        );
        assert_eq!(rain_kind, SurfaceKind::WetMud, "rain should wet the crust");
        assert!(
            rain_mu < dry_mu * 0.85,
            "grip should fall again in the rain, {dry_mu} -> {rain_mu}"
        );
        assert!(rain_stick > dry_stick, "wet clay sticks more than the crust");
        assert!(rain_sink > dry_sink, "wet clay sinks more than the crust");
        for (a, b) in bed.iter().zip(soil.bed.iter()) {
            assert!((a - b).abs() < 1.0e-6, "rain moved the bed");
        }
        for (a, b) in placed.iter().zip(soil.placed.iter()) {
            assert!((a - b).abs() < 1.0e-6, "rain moved the berm");
        }
        assert_eq!(soil.sample(1.7, 1.9).kind(), SurfaceKind::Gravel);
        let g = soil.sample(1.7, 1.9);
        assert!(g.water < 0.02, "gravel held the rain, water {}", g.water);
        assert!(g.wetness < 0.16, "gravel pore water climbed, {}", g.wetness);
        let g_mu = grip_mu(g.kind(), g.wetness, g.loose, g.water);
        assert!(
            (g_mu - gravel_mu).abs() < 0.05,
            "gravel grip moved, {gravel_mu} -> {g_mu}"
        );
        assert_eq!(soil.sample(1.6, 0.0).kind(), SurfaceKind::Sand);
        assert_eq!(soil.sample(-1.7, 0.0).kind(), SurfaceKind::Silt);
    }

    #[test]
    fn drying_crust_cracks_and_old_tracks_harden_with_a_closed_ledger() {
        let (field, s) = SoilField::drying_pair();
        eprintln!(
            "drying wetness {:.3}->{:.3} crust {:.4} m crack mean {:.4} max {:.4} cells {}/{}",
            s.wetness_before, s.wetness_after, s.crust_m, s.crack_m, s.crack_max, s.crack_cells, s.floor_cells
        );
        eprintln!(
            "drying sink wet {:.4} crust {:.4} old {:.4} m harden {:.3} extra cut wet {:.4} old {:.4}",
            s.wet_sink, s.crust_sink, s.old_sink, s.harden, s.wet_extra_cut, s.old_extra_cut
        );
        eprintln!(
            "drying ledger cut {:.4} placed {:.4} bed gap {:.6} placed gap {:.6} cloud {:.6} slurry {:.6}",
            s.cut, s.placed, s.bed_gap, s.placed_gap, s.cloud_gap, s.slurry_gap
        );
        eprintln!(
            "drying sand crust {:.6} silt crust {:.6} west {:?} {:.3} east {:?} crust {:.4} crack {:.4}",
            s.sand_crust,
            s.silt_crust,
            SurfaceKind::from_u8(s.west_kind),
            s.west_wetness,
            SurfaceKind::from_u8(s.east_kind),
            s.east_crust,
            s.east_crack
        );
        assert!(s.floor_cells > 20, "the east tracks did not cut, {}", s.floor_cells);
        assert!(
            s.wetness_after < s.wetness_before - 0.35,
            "moisture did not fall, {:.3} -> {:.3}",
            s.wetness_before,
            s.wetness_after
        );
        assert!(s.crust_m > 0.015, "the skin stayed thin, {:.4} m", s.crust_m);
        assert!(s.crack_max > 0.004, "the crust did not crack, max {:.4}", s.crack_max);
        assert!(s.crack_cells > 8, "too few fissures, {}", s.crack_cells);
        assert_eq!(
            SurfaceKind::from_u8(s.east_kind),
            SurfaceKind::DryCrust,
            "the old track should bake"
        );
        assert!(s.east_crust > 0.015, "sample missed the skin, {:.4}", s.east_crust);
        assert!(
            s.old_sink < s.wet_sink * 0.8,
            "the old track should sink less, {:.4} -> {:.4}",
            s.wet_sink,
            s.old_sink
        );
        assert!(
            s.old_sink < s.crust_sink * 0.92,
            "resting on the crust should harden past the kind alone, {:.4} vs {:.4}",
            s.old_sink,
            s.crust_sink
        );
        assert!(s.harden > 1.4, "harden mul {:.3}", s.harden);
        assert!(
            s.wet_extra_cut > 0.004,
            "the wet track took no second cut, {:.4}",
            s.wet_extra_cut
        );
        assert!(
            s.old_extra_cut < s.wet_extra_cut * 0.34,
            "the crusted track cut like wet clay, old {:.4} wet {:.4}",
            s.old_extra_cut,
            s.wet_extra_cut
        );
        assert!(s.bed_gap < 1.0e-5, "drying moved the bed, {}", s.bed_gap);
        assert!(s.placed_gap < 1.0e-5, "drying moved the pile, {}", s.placed_gap);
        assert!(s.cloud_gap < 1.0e-5, "drying moved the cloud, {}", s.cloud_gap);
        assert!(s.slurry_gap < 1.0e-5, "drying moved the slurry, {}", s.slurry_gap);
        assert!(
            (field.volumes().cut - s.cut).abs() < 1.0e-4,
            "restore changed the cut"
        );
        assert!(
            (field.volumes().placed - s.placed).abs() < 1.0e-4,
            "restore changed the pile"
        );
        assert!(s.sand_crust < 1.0e-5, "sand crusted, {}", s.sand_crust);
        assert!(s.silt_crust < 1.0e-5, "silt crusted, {}", s.silt_crust);
        assert!(
            matches!(
                SurfaceKind::from_u8(s.west_kind),
                SurfaceKind::WetMud | SurfaceKind::FreshCut | SurfaceKind::PackedClay
            ),
            "the wet track left clay, {:?}",
            SurfaceKind::from_u8(s.west_kind)
        );
        assert!(s.west_wetness > 0.6, "the wet track dried in the frame, {}", s.west_wetness);
        let texel = crust_shader_texel(s.east_crust, s.east_crack);
        let (back_c, back_k) = crust_shader_metres(texel);
        assert!((back_c - s.east_crust).abs() < 0.001, "shader crust {back_c}");
        assert!((back_k - s.east_crack).abs() < 0.001, "shader crack {back_k}");
        let east = field.sample(0.96, 0.0);
        assert_eq!(east.kind(), SurfaceKind::DryCrust);
        assert!(east.crust > 0.015);
        let west = field.floor_sample(-0.96, 0.0);
        assert!(
            matches!(
                west.kind(),
                SurfaceKind::WetMud | SurfaceKind::FreshCut | SurfaceKind::PackedClay
            ),
            "west floor {:?}",
            west.kind()
        );
        assert!(west.crust < 1.0e-4, "the wet track kept a skin, {}", west.crust);
        let book = field.crust_book();
        eprintln!(
            "crust book thickness {:.6} opening {:.6} grown {:.6} lost {:.6} gap {:.8}",
            book.thickness_m3, book.opening_m3, book.grown_m3, book.lost_m3, book.gap()
        );
        assert!(
            book.thickness_m3 > 1.0,
            "crust book stayed empty, {:.6}",
            book.thickness_m3
        );
        assert!(
            book.opening_m3 > 0.05,
            "opening book stayed empty, {:.6}",
            book.opening_m3
        );
        assert!(
            book.gap() < 1.0e-3,
            "crust book gap {:.6} grown {:.6} lost {:.6} still {:.6}",
            book.gap(),
            book.grown_m3,
            book.lost_m3,
            book.thickness_m3
        );
        for idx in 0..field.crust.len() {
            let thickness = field.crust[idx];
            let opening = field.crack[idx];
            assert!(
                thickness <= CRUST_CAP_M,
                "thickness {thickness} over {CRUST_CAP_M}"
            );
            assert!(
                opening <= thickness,
                "opening {opening} over thickness {thickness}"
            );
        }
    }

    #[test]
    fn gravel_drains_and_sand_and_silt_keep_their_states() {
        let mut soil = SoilField::flat(0.2);
        let half = soil.size * 0.5;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let idx = j * soil.res + i;
                if (-1.2..-0.4).contains(&x) {
                    soil.kind[idx] = SurfaceKind::Gravel as u8;
                    soil.wetness[idx] = 0.08;
                    soil.water[idx] = 0.12;
                } else if (-0.2..0.6).contains(&x) {
                    soil.kind[idx] = SurfaceKind::Sand as u8;
                    soil.wetness[idx] = 0.12;
                    soil.water[idx] = 0.1;
                } else if (0.8..1.6).contains(&x) {
                    soil.kind[idx] = SurfaceKind::Silt as u8;
                    soil.wetness[idx] = 0.4;
                    soil.water[idx] = 0.1;
                    // A shallow bowl, counted in the original bed, so the
                    // pond has a low spot. The weather step must not move it.
                    soil.bed[idx] -= 0.02;
                    soil.height[idx] -= 0.02;
                    soil.initial_height[idx] -= 0.02;
                }
            }
        }
        let bed = soil.bed.clone();
        let placed = soil.placed.clone();
        soil.advance_weather(4.0, 0.0);
        let gravel = soil.sample(-0.8, 0.0);
        let sand = soil.sample(0.2, 0.0);
        let silt = soil.sample(1.2, 0.0);
        assert_eq!(gravel.kind(), SurfaceKind::Gravel);
        assert!(
            gravel.water < 0.015,
            "gravel should have drained, water {}",
            gravel.water
        );
        assert!(gravel.wetness < 0.12, "gravel pores filled, {}", gravel.wetness);
        assert_eq!(sand.kind(), SurfaceKind::Sand);
        assert_eq!(silt.kind(), SurfaceKind::Silt);
        assert!(
            silt.water > sand.water + 0.01,
            "sand should drain ahead of silt, sand {} silt {}",
            sand.water,
            silt.water
        );
        assert!(
            silt.wetness > sand.wetness + 0.1,
            "silt should stay damper than sand, sand {} silt {}",
            sand.wetness,
            silt.wetness
        );
        assert!(
            grip_mu(sand.kind(), sand.wetness, sand.loose, sand.water)
                > grip_mu(silt.kind(), silt.wetness, silt.loose, silt.water)
        );
        for (a, b) in bed.iter().zip(soil.bed.iter()) {
            assert!((a - b).abs() < 1.0e-6, "drainage moved the bed");
        }
        for (a, b) in placed.iter().zip(soil.placed.iter()) {
            assert!((a - b).abs() < 1.0e-6, "drainage moved the pile");
        }

        // Standing water on a crust, with no rainfall, is enough to rewet it.
        let dry_mu = grip_mu(SurfaceKind::DryCrust, 0.12, 0.0, 0.0);
        let dry_stick = stickiness(SurfaceKind::DryCrust, 0.12, 0.0, 0.0);
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                let idx = j * soil.res + i;
                soil.kind[idx] = SurfaceKind::DryCrust as u8;
                soil.wetness[idx] = 0.12;
                soil.water[idx] = 0.0;
                soil.loose[idx] = 0.0;
                if x.abs() < 0.55 && z.abs() < 0.55 {
                    soil.water[idx] = 0.05;
                }
            }
        }
        soil.advance_weather(4.0, 0.0);
        let cell = soil.sample(0.0, 0.0);
        assert_eq!(
            cell.kind(),
            SurfaceKind::WetMud,
            "standing water should wet the crust"
        );
        assert!(
            cell.wetness > 0.2,
            "the pond should soak the crust, {}",
            cell.wetness
        );
        let soaked = grip_mu(cell.kind(), cell.wetness, cell.loose, cell.water);
        assert!(
            soaked < dry_mu,
            "standing water should drop grip, {dry_mu} -> {soaked}"
        );
        assert!(
            stickiness(cell.kind(), cell.wetness, cell.loose, cell.water) > dry_stick,
            "standing water should make the clay sticky again"
        );
    }

    fn separated(values: &[f32], margin: f32, label: &str) {
        for i in 0..values.len() {
            for j in (i + 1)..values.len() {
                let d = (values[i] - values[j]).abs();
                assert!(
                    d > margin,
                    "{label} [{i}] {:.4} and [{j}] {:.4} are the same numbers (delta {d:.4})",
                    values[i], values[j]
                );
            }
        }
    }

    /// Same wheel, five soils. Grip, sink, and a shared weather spell must
    /// not come back as one set of numbers.
    #[test]
    fn five_surfaces_do_not_share_grip_sink_or_moisture() {
        let kinds = [
            SurfaceKind::Gravel,
            SurfaceKind::DryCrust,
            SurfaceKind::WetMud,
            SurfaceKind::Sand,
            SurfaceKind::Silt,
        ];
        let names = ["gravel", "dry clay", "wet clay", "sand", "silt"];
        let mut grip = [0.0_f32; 5];
        let mut sink = [0.0_f32; 5];
        for (i, kind) in kinds.iter().enumerate() {
            let (g, z) = surface_grip_sink(*kind);
            grip[i] = g;
            sink[i] = z;
            eprintln!(
                "surface {name}  grip {g:.3}  sink {z:.4} m",
                name = names[i]
            );
        }
        separated(&grip, 0.04, "grip");
        separated(&sink, 0.006, "sink");
        assert!(grip[0] > grip[1], "gravel should grip harder than dry clay");
        assert!(grip[1] > grip[3], "dry clay should grip harder than sand");
        assert!(grip[3] > grip[2], "sand should grip harder than wet clay");
        assert!(grip[2] > grip[4], "wet clay should grip harder than wet silt");
        assert!(sink[0] < sink[1], "gravel should sink less than dry clay");
        assert!(sink[1] < sink[3], "dry clay should sink less than sand");
        assert!(sink[3] < sink[2], "sand should sink less than wet clay");
        assert!(sink[3] < sink[4], "sand should sink less than wet silt");
        assert!(
            (sink[2] - sink[4]).abs() > 0.006,
            "wet clay and silt sank the same, {s2:.4} {s4:.4}",
            s2 = sink[2],
            s4 = sink[4]
        );

        separated(
            &[
                wet_rate(SurfaceKind::Gravel),
                wet_rate(SurfaceKind::DryCrust),
                wet_rate(SurfaceKind::WetMud),
                wet_rate(SurfaceKind::Sand),
                wet_rate(SurfaceKind::Silt),
            ],
            0.02,
            "wet rate",
        );
        separated(
            &[
                dry_rate(SurfaceKind::Gravel),
                dry_rate(SurfaceKind::DryCrust),
                dry_rate(SurfaceKind::WetMud),
                dry_rate(SurfaceKind::Sand),
                dry_rate(SurfaceKind::Silt),
            ],
            0.01,
            "dry rate",
        );
        separated(
            &[
                water_hold(SurfaceKind::Gravel),
                water_hold(SurfaceKind::DryCrust),
                water_hold(SurfaceKind::WetMud),
                water_hold(SurfaceKind::Sand),
                water_hold(SurfaceKind::Silt),
            ],
            0.0004,
            "water hold",
        );
        separated(
            &[
                water_conduct(SurfaceKind::Gravel),
                water_conduct(SurfaceKind::DryCrust),
                water_conduct(SurfaceKind::WetMud),
                water_conduct(SurfaceKind::Sand),
                water_conduct(SurfaceKind::Silt),
            ],
            0.05,
            "water conduct",
        );

        let mut end_w = [0.0_f32; 5];
        let mut end_water = [0.0_f32; 5];
        for (i, kind) in kinds.iter().enumerate() {
            let mut soil = SoilField::flat(0.2);
            for idx in 0..soil.wetness.len() {
                soil.kind[idx] = *kind as u8;
                soil.wetness[idx] = 0.40;
                soil.base_wetness[idx] = 0.40;
                soil.water[idx] = 0.020;
                soil.loose[idx] = 0.0;
            }
            let bed = soil.bed.clone();
            let placed = soil.placed.clone();
            soil.advance_weather(4.0, 0.0);
            let sample = soil.sample(0.2, 0.2);
            end_w[i] = sample.wetness;
            end_water[i] = sample.water;
            assert_eq!(
                sample.kind(),
                *kind,
                "{} changed kind to {:?}",
                names[i],
                sample.kind()
            );
            for (a, b) in bed.iter().zip(soil.bed.iter()) {
                assert!((a - b).abs() < 1.0e-6, "{} weather moved the bed", names[i]);
            }
            for (a, b) in placed.iter().zip(soil.placed.iter()) {
                assert!(
                    (a - b).abs() < 1.0e-6,
                    "{} weather moved the pile",
                    names[i]
                );
            }
            eprintln!(
                "moisture {}  wetness {:.3}  water {:.4} m  kind {:?}",
                names[i], end_w[i], end_water[i], sample.kind()
            );
        }
        separated(&end_w, 0.03, "pore moisture");
        assert!(
            end_w[3] > end_w[0] + 0.08,
            "sand should keep capillary moisture, gravel should not: sand {:.3} gravel {:.3}",
            end_w[3], end_w[0]
        );
        assert!(
            end_water[4] > end_water[2] + 0.002,
            "silt should keep more standing water than wet clay, silt {:.4} clay {:.4}",
            end_water[4], end_water[2]
        );
        assert!(
            end_water[2] > end_water[1] + 0.004,
            "wet clay should hold water a crust sheds, wet {:.4} dry {:.4}",
            end_water[2], end_water[1]
        );
        assert!(
            end_water[0] < 0.004 && end_water[3] < 0.004,
            "gravel and sand should have drained, gravel {:.4} sand {:.4}",
            end_water[0], end_water[3]
        );
        assert!(
            end_w[2] > end_w[1] + 0.08,
            "wet clay should stay wetter than the crust, wet {:.3} dry {:.3}",
            end_w[2], end_w[1]
        );
    }

    #[test]
    fn tread_negative_persists_with_the_rut() {
        let mut soil = SoilField::flat(0.72);
        let before = soil.cut_volume();
        let center = glam::Vec3::new(0.74, 0.3, 0.2);
        let idx = soil.idx_at(0.74, 0.2).expect("cell");
        let height = soil.height[idx];
        let bed = soil.bed[idx];
        let placed = soil.placed[idx];
        let kind = soil.kind[idx];
        soil.imprint_tread(center, glam::Vec3::Z, glam::Vec3::X, 0.38, 0.28, 0.04);
        let amp = soil.tread_amp[idx];
        assert!(
            (0.026..=0.048).contains(&amp),
            "the contact did not press a tread height, amp {amp}"
        );
        let drop = soil.tread_drop(0.74, 0.2);
        assert!(
            (drop + amp).abs() < 1.0e-6,
            "the lug negative is not the persisted height, drop {drop} amp {amp}"
        );
        assert!(
            (soil.height[idx] - height).abs() < 1.0e-6
                && (soil.bed[idx] - bed).abs() < 1.0e-6
                && (soil.placed[idx] - placed).abs() < 1.0e-6
                && soil.kind[idx] == kind,
            "the imprint painted or rebuilt the cell instead of storing a height"
        );
        let (n, deep) = soil.tread_imprint_stats();
        assert!(n >= 1 && deep >= amp - 1.0e-6, "imprint missing on a second look");
        soil.imprint_tread(center, glam::Vec3::Z, glam::Vec3::X, 0.38, 0.28, 0.04);
        assert!(
            (soil.tread_drop(0.74, 0.2) - drop).abs() < 1.0e-6,
            "a second look rewrote the tread height"
        );
        assert!(
            (soil.cut_volume() - before).abs() < 1.0e-6,
            "the tread negative cut the bed"
        );
        let crust_cell = soil.idx_at(0.2, 0.2).expect("crust cell");
        soil.crust[crust_cell] = 0.018;
        soil.crack[crust_cell] = 0.007;
        let saved = soil.save_state();
        let loaded = SoilField::load_state(&saved).expect("reload");
        assert!(
            (loaded.tread_amp[idx] - amp).abs() < 1.0e-6,
            "reload dropped the tread"
        );
        assert!(
            (loaded.tread_drop(0.74, 0.2) - drop).abs() < 1.0e-6,
            "reload dropped the tread height"
        );
        assert!((loaded.tread_ang[idx] - soil.tread_ang[idx]).abs() < 1.0e-6);
        let cells = soil.res * soil.res;
        let legacy_len = saved.len() - cells * 40;
        let legacy = SoilField::load_state(&saved[..legacy_len]).expect("old snapshot");
        assert_eq!(legacy.tread_amp[idx], 0.0);
        assert_eq!(legacy.slurry[idx], 0.0);
        assert_eq!(legacy.crust[idx], 0.0);
        assert!((legacy.bed[idx] - soil.bed[idx]).abs() < 1.0e-6);
        let prior = SoilField::load_state(&saved[..saved.len() - cells * 28]).expect("pre-slurry");
        assert_eq!(prior.slurry[idx], 0.0);
        assert!((prior.tread_amp[idx] - amp).abs() < 1.0e-6);
        assert!((loaded.slurry[idx] - soil.slurry[idx]).abs() < 1.0e-6);
        assert!((loaded.crust[crust_cell] - 0.018).abs() < 1.0e-6);
        assert!((loaded.crack[crust_cell] - 0.007).abs() < 1.0e-6);
        let pre_crust = SoilField::load_state(&saved[..saved.len() - cells * 24]).expect("pre-crust");
        assert_eq!(pre_crust.crust[crust_cell], 0.0);
        assert_eq!(pre_crust.crack[crust_cell], 0.0);
        assert!((pre_crust.slurry[idx] - soil.slurry[idx]).abs() < 1.0e-6);
    }

    #[test]
    fn slurry_settle_pace_orders_by_viscosity() {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let clay = slurry_settle_pace(SurfaceKind::WetMud, moisture);
        let silt = slurry_settle_pace(SurfaceKind::Silt, moisture);
        let sand = slurry_settle_pace(SurfaceKind::Sand, moisture);
        assert!(
            sand > silt && silt > clay,
            "settle pace clay {clay:.3} silt {silt:.3} sand {sand:.3} m/s per m head"
        );
    }

    #[test]
    fn wet_berm_creep_under_load_orders_by_mix() {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let load = 5200.0;
        let clay = wet_berm_creep_under_load_rate(SurfaceKind::WetMud, moisture, load);
        let silt = wet_berm_creep_under_load_rate(SurfaceKind::Silt, moisture, load);
        let sand = wet_berm_creep_under_load_rate(SurfaceKind::Sand, moisture, load);
        assert!(
            sand > silt && silt > clay,
            "load creep clay {clay:.4} silt {silt:.4} sand {sand:.4} m/s"
        );
    }

    #[test]
    fn wet_berm_slumps_into_rut_under_load_and_conserves_volume() {
        let mut soil = SoilField::mud_pad();
        let bed = soil.paint_slurry_flow();
        soil.advance_slurry(0.35);
        let berm_before = soil.placed_volume_in(bed.x1, bed.x1 + 0.6, bed.high_z0, bed.high_z1);
        let rut_before = soil.slurry_volume_in(bed.x0, bed.x1, bed.high_z0, bed.high_z1);
        let pool_before = soil.slurry_volume_in(bed.x0, bed.x1, bed.low_z0, bed.low_z1);
        let mix0 = soil.mix_volumes();
        let half = soil.size * 0.5;
        let mut bx = 0.0_f32;
        let mut bz = 0.0_f32;
        let mut n = 0.0_f32;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                if x < bed.x1 || x >= bed.x1 + 0.55 || z < bed.high_z0 || z >= bed.high_z1 {
                    continue;
                }
                let idx = j * soil.res + i;
                if soil.placed[idx] < 0.02 {
                    continue;
                }
                bx += x;
                bz += z;
                n += 1.0;
            }
        }
        assert!(n > 0.5, "no berm cell for load creep");
        bx /= n;
        bz /= n;
        let hub = Vec3::new(bx, soil.sample(bx, bz).height + 0.38, bz);
        for _ in 0..240 {
            soil.creep_wet_berm_under_patch(
                hub,
                Vec3::Z,
                Vec3::X,
                0.38,
                0.30,
                true,
                5400.0,
                1.0 / 60.0,
            );
        }
        let berm_after = soil.placed_volume_in(bed.x1, bed.x1 + 0.6, bed.high_z0, bed.high_z1);
        let rut_after = soil.slurry_volume_in(bed.x0, bed.x1, bed.high_z0, bed.high_z1);
        let pool_after = soil.slurry_volume_in(bed.x0, bed.x1, bed.low_z0, bed.low_z1);
        let mix1 = soil.mix_volumes();
        eprintln!(
            "wet berm load berm {berm_before:.4}->{berm_after:.4} rut {rut_before:.4}->{rut_after:.4} \
             pool {pool_before:.4}->{pool_after:.4}"
        );
        assert!(
            berm_after < berm_before - 0.001,
            "berm did not slump, {berm_before:.4} -> {berm_after:.4}"
        );
        assert!(
            rut_after + pool_after > rut_before + pool_before + 0.001,
            "rut and pool did not gain slurry"
        );
        assert!(
            (mix1.total() - mix0.total()).abs() < 1.0e-3,
            "mix drift {:.6} -> {:.6}",
            mix0.total(),
            mix1.total()
        );
    }

    #[test]
    fn slurry_mix_conserves_volume_and_kinds_differ_at_one_moisture() {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let clay_v = slurry_viscosity(SurfaceKind::WetMud, moisture);
        let silt_v = slurry_viscosity(SurfaceKind::Silt, moisture);
        let sand_v = slurry_viscosity(SurfaceKind::Sand, moisture);
        assert!(
            clay_v > silt_v * 1.5,
            "clay {clay_v:.2} should be thicker than silt {silt_v:.2}"
        );
        assert!(
            silt_v > sand_v * 1.5,
            "silt {silt_v:.2} should be thicker than sand {sand_v:.2}"
        );
        let stiff = slurry_viscosity(SurfaceKind::WetMud, 0.30);
        let thin = slurry_viscosity(SurfaceKind::WetMud, 0.85);
        assert!(
            stiff > clay_v && clay_v > thin,
            "clay viscosity should fall as moisture rises, {stiff:.2} {clay_v:.2} {thin:.2}"
        );
        eprintln!(
            "slurry viscosity at moisture {moisture:.2}  clay {clay_v:.2} Pa·s  silt {silt_v:.2} Pa·s  sand {sand_v:.2} Pa·s"
        );

        let mut soil = SoilField::mud_pad();
        let water = 0.08_f32;
        let clay_n = paint_mix(&mut soil, -3.2, -2.2, -0.8, 0.8, SurfaceKind::WetMud, moisture, water, 0.08, false);
        let silt_n = paint_mix(&mut soil, -0.6, 0.6, -0.8, 0.8, SurfaceKind::Silt, moisture, water, 0.07, true);
        let sand_n = paint_mix(&mut soil, 2.2, 3.2, -0.8, 0.8, SurfaceKind::Sand, moisture, water, 0.06, false);
        let flat_n = paint_mix(&mut soil, 5.0, 5.6, -0.4, 0.4, SurfaceKind::WetMud, moisture, water, 0.0, false);
        let gravel_n = paint_mix(&mut soil, 6.6, 7.4, -0.6, 0.6, SurfaceKind::Gravel, moisture, water, 0.07, true);
        assert!(clay_n > 8 && silt_n > 8 && sand_n > 8 && flat_n > 2 && gravel_n > 4);

        let before = soil.mix_volumes();
        let clay_i = soil.idx_at(-2.7, 0.0).expect("clay");
        let silt_i = soil.idx_at(0.0, 0.0).expect("silt");
        let sand_i = soil.idx_at(2.7, 0.0).expect("sand");
        let flat_i = soil.idx_at(5.3, 0.0).expect("flat");
        let gravel_i = soil.idx_at(7.0, 0.0).expect("gravel");
        let beds = [
            soil.bed[clay_i],
            soil.bed[silt_i],
            soil.bed[sand_i],
        ];
        soil.mix_slurry();
        let after = soil.mix_volumes();
        eprintln!(
            "slurry volumes before  soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
            before.soil, before.water, before.slurry, before.total()
        );
        eprintln!(
            "slurry volumes after   soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
            after.soil, after.water, after.slurry, after.total()
        );
        assert!(
            (after.total() - before.total()).abs() < 1.0e-3,
            "water plus soil drifted {:.6} -> {:.6}",
            before.total(),
            after.total()
        );
        let soil_left = before.soil - after.soil;
        let water_left = before.water - after.water;
        assert!(soil_left > 1.0e-4, "no soil entered the slurry");
        assert!(water_left > 1.0e-4, "no water entered the slurry");
        assert!(
            (after.slurry - (soil_left + water_left)).abs() < 1.0e-3,
            "slurry {:.4} is not the soil {:.4} plus the water {:.4}",
            after.slurry,
            soil_left,
            water_left
        );

        let expect = slurry_soil_depth(water, moisture);
        for (idx, bed0, name) in [
            (clay_i, beds[0], "clay"),
            (silt_i, beds[1], "silt"),
            (sand_i, beds[2], "sand"),
        ] {
            assert!(
                soil.water[idx] < 1.0e-6,
                "{name} still has a water column {:.4}",
                soil.water[idx]
            );
            let joined = bed0 - soil.bed[idx];
            assert!(
                (joined - expect).abs() < 1.0e-4,
                "{name} soil {:.4} m is not the moisture depth {:.4}",
                joined,
                expect
            );
            assert!(
                (soil.slurry[idx] - (joined + water)).abs() < 1.0e-4,
                "{name} slurry is not one body"
            );
            let h = soil.bed[idx] + soil.placed[idx] + soil.slurry[idx];
            assert!((soil.height[idx] - h).abs() < 1.0e-5, "{name} mesh is a second surface");
            let mix_m = water / soil.slurry[idx];
            assert!(
                (mix_m - moisture).abs() < 1.0e-3,
                "{name} moisture {mix_m:.3} left {moisture:.3}"
            );
        }
        assert!(
            (soil.water[flat_i] - water).abs() < 1.0e-6 && soil.slurry[flat_i] == 0.0,
            "flat water mixed without a rut or a hole"
        );
        assert!(
            (soil.water[gravel_i] - water).abs() < 1.0e-6 && soil.slurry[gravel_i] == 0.0,
            "gravel became a slurry"
        );
    }

    fn paint_mix(
        soil: &mut SoilField,
        x0: f32,
        x1: f32,
        z0: f32,
        z1: f32,
        kind: SurfaceKind,
        moisture: f32,
        water: f32,
        dig: f32,
        hole: bool,
    ) -> usize {
        let half = soil.size * 0.5;
        let mut n = 0usize;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                if x < x0 || x >= x1 || z < z0 || z >= z1 {
                    continue;
                }
                let idx = j * soil.res + i;
                soil.kind[idx] = kind as u8;
                soil.wetness[idx] = moisture;
                soil.water[idx] = water;
                if dig > 0.0 {
                    soil.bed[idx] -= dig;
                    if hole {
                        soil.initial_height[idx] -= dig;
                    }
                }
                soil.sync(idx);
                n += 1;
            }
        }
        n
    }

    /// Slurry runs into the low rut, the berm creeps into that slurry, and a
    /// tire pass raises a bow in the slurry column. The bed under the tire
    /// stays put, so the track is not a dry cut.
    #[test]
    fn slurry_flows_downhill_and_a_tire_raises_a_bow_wave() {
        let mut soil = SoilField::mud_pad();
        let bed = soil.paint_slurry_flow();
        let before = soil.mix_volumes();
        let low_before = soil.slurry_volume_in(bed.x0, bed.x1, bed.low_z0, bed.low_z1);
        let high_before = soil.slurry_volume_in(bed.x0, bed.x1, bed.high_z0, bed.high_z1);
        let berm_before = soil.placed_volume_in(bed.x1, bed.x1 + 0.6, bed.high_z0, bed.high_z1);
        let far = soil.idx_at(6.0, 6.0).expect("far cell");
        let far_slurry = soil.slurry[far];
        let far_placed = soil.placed[far];
        assert!(high_before > 0.05, "the high rut should start full of slurry");
        assert!(low_before < 1.0e-6, "the low rut should start empty");
        assert!(berm_before > 0.005, "the shoulder should be a berm, {berm_before}");

        soil.advance_slurry(3.0);
        let after = soil.mix_volumes();
        let low_after = soil.slurry_volume_in(bed.x0, bed.x1, bed.low_z0, bed.low_z1);
        let high_after = soil.slurry_volume_in(bed.x0, bed.x1, bed.high_z0, bed.high_z1);
        let berm_after = soil.placed_volume_in(bed.x1, bed.x1 + 0.6, bed.high_z0, bed.high_z1);
        eprintln!(
            "slurry flow before  soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
            before.soil, before.water, before.slurry, before.total()
        );
        eprintln!(
            "slurry flow after   soil {:.4} m3  water {:.4} m3  slurry {:.4} m3  total {:.4} m3",
            after.soil, after.water, after.slurry, after.total()
        );
        eprintln!(
            "slurry low {:.4} -> {:.4} m3   high {:.4} -> {:.4} m3   berm {:.4} -> {:.4} m3",
            low_before, low_after, high_before, high_after, berm_before, berm_after
        );
        assert!(
            (after.total() - before.total()).abs() < 1.0e-3,
            "water plus soil drifted {:.6} -> {:.6}",
            before.total(),
            after.total()
        );
        let soil_lost = before.soil - after.soil;
        let slurry_gained = after.slurry - before.slurry;
        assert!(
            (soil_lost - slurry_gained).abs() < 1.0e-3,
            "berm soil {:.4} did not become slurry {:.4}",
            soil_lost,
            slurry_gained
        );
        assert!(
            low_after > low_before + 0.01,
            "slurry did not reach the low rut, {low_before:.4} -> {low_after:.4}"
        );
        assert!(
            high_after < high_before - 0.01,
            "the high rut kept its slurry, {high_before:.4} -> {high_after:.4}"
        );
        assert!(
            berm_after < berm_before - 0.001,
            "the berm did not creep, {berm_before:.4} -> {berm_after:.4}"
        );
        assert!(
            berm_after > berm_before * 0.35,
            "the berm collapsed instead of creeping, {berm_before:.4} -> {berm_after:.4}"
        );
        assert!(
            (soil.slurry[far] - far_slurry).abs() < 1.0e-6
                && (soil.placed[far] - far_placed).abs() < 1.0e-6,
            "slurry left the rut"
        );

        let tire = slurry_tire_cell(&soil, &bed);
        let idx = tire.0;
        let bed0 = soil.bed[idx];
        let placed0 = soil.placed[idx];
        let track0 = soil.slurry[idx];
        let ahead = soil.idx_at(tire.1, tire.2 + 0.42).expect("ahead");
        let ahead0 = soil.slurry[ahead];
        let pass = soil.raise_bow_wave(tire.1, tire.2, Vec3::Z, 0.38, 0.28);
        let track1 = soil.slurry[idx];
        eprintln!(
            "bow wave {:.4} m3  wake {:.4} m3   splash track {track0:.4} -> {track1:.4} m   ahead {ahead0:.4} -> {:.4} m",
            pass.bow_m3,
            pass.wake_m3,
            soil.slurry[ahead]
        );
        assert!(pass.bow_m3 > 0.002, "the tire did not raise a bow wave, {:.4} m3", pass.bow_m3);
        assert!(pass.wake_m3 > 0.0008, "the tire did not thin a wake, {:.4} m3", pass.wake_m3);
        assert!(
            track1 < track0 - 0.008,
            "the splash track is not in the slurry, {track0:.4} -> {track1:.4}"
        );
        assert!(
            soil.slurry[ahead] > ahead0 + 0.005,
            "the bow is not ahead of the tire"
        );
        assert!(
            (soil.bed[idx] - bed0).abs() < 1.0e-6 && (soil.placed[idx] - placed0).abs() < 1.0e-6,
            "the tire pass cut the bed instead of moving slurry"
        );
        let passed = soil.mix_volumes();
        assert!(
            (passed.total() - after.total()).abs() < 1.0e-3,
            "the bow wave did not conserve water plus soil, {:.6} -> {:.6}",
            after.total(),
            passed.total()
        );
    }

    fn slurry_tire_cell(soil: &SoilField, bed: &SlurryFlowBed) -> (usize, f32, f32) {
        let half = soil.size * 0.5;
        let mut best: Option<(usize, f32, f32, f32)> = None;
        for j in 0..soil.res {
            for i in 0..soil.res {
                let x = (i as f32 + 0.5) * soil.cell - half;
                let z = (j as f32 + 0.5) * soil.cell - half;
                if x < bed.x0 + 0.35 || x >= bed.x1 - 0.35 {
                    continue;
                }
                if z < bed.low_z0 + 0.35 || z >= bed.low_z0 + 1.05 {
                    continue;
                }
                let idx = j * soil.res + i;
                let depth = soil.slurry[idx];
                if depth < 0.02 {
                    continue;
                }
                let score = depth - (x - 0.15).abs() * 0.02;
                if best.map(|b| score > b.3).unwrap_or(true) {
                    best = Some((idx, x, z, score));
                }
            }
        }
        let (idx, x, z, _) = best.expect("low rut should hold slurry under a tire");
        (idx, x, z)
    }

    /// Deep slurry resists translation and spin. Clay, silt, and sand use the
    /// same moisture, so the gap between them is the mix viscosity.
    #[test]
    fn deep_slurry_drag_resists_translation_and_spin() {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let depth = 0.16_f32;
        let film = 0.02_f32;
        let v = 1.15_f32;
        let omega = 3.2_f32;
        let radius = 0.38_f32;
        let width = 0.28_f32;
        let clay = slurry_motion_drag(SurfaceKind::WetMud, moisture, depth, radius, width, v, omega);
        let silt = slurry_motion_drag(SurfaceKind::Silt, moisture, depth, radius, width, v, omega);
        let sand = slurry_motion_drag(SurfaceKind::Sand, moisture, depth, radius, width, v, omega);
        let shallow = slurry_motion_drag(SurfaceKind::WetMud, moisture, film, radius, width, v, omega);
        let coast = slurry_motion_drag(SurfaceKind::WetMud, moisture, depth, radius, width, v, 0.0);
        let spin = slurry_motion_drag(SurfaceKind::WetMud, moisture, depth, radius, width, 0.0, omega);
        let reverse = slurry_motion_drag(
            SurfaceKind::WetMud,
            moisture,
            depth,
            radius,
            width,
            -v,
            -omega,
        );
        eprintln!(
            "deep slurry drag  clay force {:.1} N  torque {:.1} N·m  visc {:.2} Pa·s",
            clay.force.abs(),
            clay.torque.abs(),
            clay.viscosity
        );
        eprintln!(
            "deep slurry drag  silt force {:.1} N  torque {:.1} N·m  visc {:.2} Pa·s",
            silt.force.abs(),
            silt.torque.abs(),
            silt.viscosity
        );
        eprintln!(
            "deep slurry drag  sand force {:.1} N  torque {:.1} N·m  visc {:.2} Pa·s",
            sand.force.abs(),
            sand.torque.abs(),
            sand.viscosity
        );
        eprintln!(
            "film drag         clay force {:.1} N  torque {:.1} N·m",
            shallow.force.abs(),
            shallow.torque.abs()
        );
        assert!(
            (clay.viscosity - slurry_viscosity(SurfaceKind::WetMud, moisture)).abs() < 1.0e-3,
            "drag viscosity is not the clay mix"
        );
        assert!(
            (silt.viscosity - slurry_viscosity(SurfaceKind::Silt, moisture)).abs() < 1.0e-3
        );
        assert!(
            (sand.viscosity - slurry_viscosity(SurfaceKind::Sand, moisture)).abs() < 1.0e-3
        );
        assert!(
            clay.viscosity > silt.viscosity * 1.5 && silt.viscosity > sand.viscosity * 1.5,
            "mix viscosities did not separate, clay {} silt {} sand {}",
            clay.viscosity,
            silt.viscosity,
            sand.viscosity
        );
        assert!(
            clay.force.abs() > silt.force.abs() * 1.8,
            "clay should resist translation more than silt, {:.1} vs {:.1}",
            clay.force.abs(),
            silt.force.abs()
        );
        assert!(
            silt.force.abs() > sand.force.abs() * 1.8,
            "silt should resist translation more than sand, {:.1} vs {:.1}",
            silt.force.abs(),
            sand.force.abs()
        );
        assert!(
            clay.torque.abs() > silt.torque.abs() * 1.8
                && silt.torque.abs() > sand.torque.abs() * 1.8,
            "spin resistance should follow the mix"
        );
        assert!(
            clay.force.abs() > shallow.force.abs() * 2.0,
            "deep slurry force {:.1} N is not above the film {:.1} N",
            clay.force.abs(),
            shallow.force.abs()
        );
        assert!(
            clay.torque.abs() > shallow.torque.abs() * 2.0,
            "deep slurry torque {:.1} N·m is not above the film {:.1} N·m",
            clay.torque.abs(),
            shallow.torque.abs()
        );
        assert!(
            clay.force < -200.0 && clay.torque < -40.0,
            "drag should oppose +v and +omega, force {} torque {}",
            clay.force,
            clay.torque
        );
        assert!(
            reverse.force > 200.0 && reverse.torque > 40.0,
            "drag should oppose reverse motion"
        );
        assert!(
            coast.force.abs() > 200.0 && coast.torque.abs() < 1.0,
            "forward motion should resist without a spin torque, force {} torque {}",
            coast.force,
            coast.torque
        );
        assert!(
            spin.torque.abs() > 40.0 && spin.force.abs() < 1.0,
            "spin should resist without a shove, force {} torque {}",
            spin.force,
            spin.torque
        );
        let dry = slurry_motion_drag(SurfaceKind::WetMud, moisture, 0.0, radius, width, v, omega);
        assert!(dry.force.abs() < 1.0 && dry.torque.abs() < 1.0);
    }

    /// A rising tire holds a short suction, then the break throws slurry.
    /// The thrown cubic metres leave the slurry and stay in the airborne
    /// account until they are placed.
    #[test]
    fn pull_out_suction_then_breakaway_keeps_volume() {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let mut soil = SoilField::mud_pad();
        let depth = 0.16_f32;
        let poured = soil.fill_slurry_pool(
            -1.3,
            1.3,
            -1.3,
            1.3,
            depth,
            SurfaceKind::WetMud,
            moisture,
        );
        assert!(poured > 0.2, "the pool did not fill, {poured}");
        let idx = soil.idx_at(0.05, 0.12).expect("pool cell");
        let half = soil.size * 0.5;
        let i = idx % soil.res;
        let j = idx / soil.res;
        let x = (i as f32 + 0.5) * soil.cell - half;
        let z = (j as f32 + 0.5) * soil.cell - half;
        assert!(soil.slurry[idx] > 0.12, "the tire is not in deep slurry");
        let bed0 = soil.bed[idx];
        let before = soil.mix_volumes();
        let air0 = soil.airborne;
        let mut seal = SlurrySeal::seated();
        let mut impulse = 0.0_f32;
        let mut thrown = 0.0_f32;
        let mut held = false;
        let dt = 0.02_f32;
        let lift_speed = 0.40_f32;
        for _ in 0..24 {
            let out = soil.pull_from_slurry(x, z, Vec3::Z, 0.38, 0.28, lift_speed, dt, &mut seal);
            impulse += out.impulse;
            if out.thrown <= 1.0e-8 {
                assert!(
                    out.force > 80.0,
                    "suction lapsed before the break, force {:.1} N lift {:.3} m",
                    out.force,
                    seal.lift
                );
                held = true;
            } else {
                thrown = out.thrown;
                assert!(out.broken, "the throw happened before the seal broke");
                break;
            }
        }
        assert!(held, "the pull never held a suction");
        assert!(seal.broken, "the seal did not break");
        assert!(
            (seal.impulse - impulse).abs() < 1.0e-3,
            "impulse was not accumulated on the seal"
        );
        assert!(
            impulse > 40.0,
            "suction impulse {impulse:.2} N·s is too small for a sealed pull"
        );
        assert!(thrown > 0.002, "breakaway threw {thrown:.4} m3");
        let after = soil.mix_volumes();
        eprintln!(
            "pull-out suction impulse {impulse:.2} N·s  force held  thrown {thrown:.4} m3"
        );
        eprintln!(
            "volume before {:.4} m3  after {:.4} m3  airborne {:.4} m3  slurry {:.4} -> {:.4} m3",
            before.total(),
            after.total(),
            soil.airborne,
            before.slurry,
            after.slurry
        );
        let gap = (before.total() + air0 - (after.total() + soil.airborne)).abs();
        assert!(
            gap < 1.0e-3,
            "volume deleted, before {:.4} after {:.4} airborne {:.4}",
            before.total(),
            after.total(),
            soil.airborne
        );
        assert!(
            (soil.airborne - air0 - thrown).abs() < 1.0e-4,
            "thrown {:.4} m3 is not the airborne account {:.4}",
            thrown,
            soil.airborne - air0
        );
        assert!(
            before.slurry - after.slurry > thrown * 0.99,
            "thrown mud was not taken from the slurry"
        );
        assert!(
            (soil.bed[idx] - bed0).abs() < 1.0e-6,
            "the breakaway cut the bed"
        );
        let quiet = soil.pull_from_slurry(x, z, Vec3::Z, 0.38, 0.28, lift_speed, dt, &mut seal);
        assert!(
            quiet.force.abs() < 1.0 && quiet.thrown < 1.0e-8,
            "the seal pulled again after the break"
        );
        let placed0 = soil.placed_volume();
        let spot = soil.receive_thrown(x + 2.4, z, thrown).expect("place thrown slurry");
        assert!(soil.airborne < 1.0e-5, "airborne left {}", soil.airborne);
        let placed = soil.placed_volume() - placed0;
        let landed = soil.mix_volumes();
        eprintln!(
            "placed thrown {placed:.4} m3 at ({:.2}, {:.2})  total {:.4} m3",
            spot.x,
            spot.z,
            landed.total()
        );
        assert!(
            (placed - thrown).abs() < 1.0e-4,
            "placed {:.4} is not the thrown {:.4}",
            placed,
            thrown
        );
        assert!(
            (landed.total() - before.total()).abs() < 1.0e-3,
            "landing did not restore the volume, before {:.4} after {:.4}",
            before.total(),
            landed.total()
        );
    }

    /// Clay keeps its slurry in the rut. Silt, at the same moisture, slides
    /// farther. A dry crust is not turned into slurry, and its cut still
    /// equals the placed spoil plus the throw.
    #[test]
    fn clay_slurry_holds_silt_slides_and_crust_stays_dry() {
        let moisture = SLURRY_COMPARE_MOISTURE;
        let clay_hold = slurry_hold_fraction(SurfaceKind::WetMud, moisture);
        let silt_hold = slurry_hold_fraction(SurfaceKind::Silt, moisture);
        let crust_hold = slurry_hold_fraction(SurfaceKind::DryCrust, moisture);
        assert!(
            clay_hold > silt_hold + 0.08,
            "clay should keep more of the cut, clay {clay_hold:.3} silt {silt_hold:.3}"
        );
        assert!(
            crust_hold < 1.0e-6,
            "a dry crust must not hold a slurry share, {crust_hold}"
        );
        assert!(
            slurry_slide_reach(SurfaceKind::Silt, moisture)
                > slurry_slide_reach(SurfaceKind::WetMud, moisture) * 2.0,
            "silt should be allowed to travel farther than clay"
        );
        assert!(slurry_slide_reach(SurfaceKind::DryCrust, moisture) < 1.0e-6);
        assert!(slurry_leave_fraction(SurfaceKind::DryCrust, moisture) < 1.0e-6);

        let clay = shove_slurry_pad(SurfaceKind::WetMud, moisture);
        let silt = shove_slurry_pad(SurfaceKind::Silt, moisture);
        eprintln!(
            "slide clay travel {:.4} m  moved {:.4} m3  minted {:.6}    silt travel {:.4} m  moved {:.4} m3  minted {:.6}",
            clay.travel, clay.moved, clay.minted, silt.travel, silt.moved, silt.minted
        );
        assert!(
            clay.travel.abs() < 0.12,
            "clay slurry should hold, travel {:.4} m",
            clay.travel
        );
        assert!(
            silt.travel > clay.travel + 0.20,
            "silt should slide farther, clay {:.4} silt {:.4}",
            clay.travel,
            silt.travel
        );
        assert!(
            silt.travel > clay.travel * 2.5,
            "silt travel {:.4} is not several times the clay {:.4}",
            silt.travel,
            clay.travel
        );
        assert!(clay.minted.abs() < 1.0e-3 && silt.minted.abs() < 1.0e-3);
        assert!(clay.moved > 1.0e-4, "the clay shove moved nothing");
        assert!(silt.moved > clay.moved * 2.0, "silt did not give up more slurry");

        let mut crust = SoilField::flat(0.12);
        assert_eq!(
            crust.sample(0.0, 0.0).kind(),
            SurfaceKind::DryCrust,
            "the dry pad should be a crust"
        );
        let before = crust.volumes();
        let cut = crust.cut_lane_in_slurry(
            Vec3::new(0.0, 0.0, -1.2),
            Vec3::new(0.0, 0.0, 1.2),
            0.42,
            0.12,
            0.40,
        );
        let after = crust.volumes();
        eprintln!(
            "crust cut {:.4} berm {:.4} thrown {:.4} slurry {:.4} placed {:.4} air {:.4}",
            cut.cut,
            cut.berm,
            cut.thrown,
            cut.slurry,
            after.placed - before.placed,
            crust.airborne
        );
        assert!(cut.cut > 0.02, "the crust was not cut, {:.4}", cut.cut);
        assert!(cut.slurry < 1.0e-5, "crust became slurry, {:.4}", cut.slurry);
        assert!(
            crust.slurry.iter().all(|s| *s < 1.0e-5),
            "a crust cell holds slurry"
        );
        assert_eq!(crust.sample(0.0, 0.0).kind(), SurfaceKind::DryCrust);
        let gap = (cut.cut - cut.berm - cut.thrown - cut.slurry).abs();
        assert!(gap < 1.0e-3, "crust cut split drifted {gap}");
        let solid = (cut.cut - (after.placed - before.placed) - crust.airborne).abs();
        assert!(
            solid < 1.0e-3,
            "crust broke cut = placed + airborne, gap {solid}"
        );
        let slide = crust.slide_slurry(-2.0, 2.0, -2.0, 2.0, Vec3::X);
        assert!(slide.moved < 1.0e-6 && slide.minted.abs() < 1.0e-6);
    }

    fn shove_slurry_pad(kind: SurfaceKind, moisture: f32) -> SlurrySlide {
        let mut soil = SoilField::mud_pad();
        let _ = soil.fill_slurry_pool(-0.7, 0.7, -1.3, 1.3, 0.12, kind, moisture);
        let bed0: Vec<f32> = soil.bed.clone();
        let placed0: Vec<f32> = soil.placed.clone();
        let slide = soil.slide_slurry(-0.7, 0.7, -1.3, 1.3, Vec3::X);
        assert!(
            soil.bed
                .iter()
                .zip(bed0.iter())
                .all(|(a, b)| (a - b).abs() < 1.0e-6),
            "the shove cut the bed"
        );
        assert!(
            soil.placed
                .iter()
                .zip(placed0.iter())
                .all(|(a, b)| (a - b).abs() < 1.0e-6),
            "the shove moved the berm"
        );
        slide
    }
}
