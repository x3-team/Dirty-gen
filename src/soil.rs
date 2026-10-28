//! Heightfield soil: Bekker sinkage, plastic ruts, wetness and shallow water.
//!
//! The cut and the spoil are different layers. `bed` is the ground the tire
//! carved. `placed` is soil that has already been put somewhere else (berm,
//! crest clod, impact ring). Standing water in a rut or a hole mixes with the
//! wet soil into one slurry. That slurry then flows downhill into the low rut
//! and into holes. A berm beside it slowly creeps into the slurry. A tire in
//! the slurry pushes a bow wave and leaves its track in the slurry thickness.
//! The visible height is bed + placed + that slurry, so the picture is not a
//! solid floor under a separate water column.
//!
//! A rut is a vertical cut at a fixed world XZ. The tire does not push the
//! column along the ground normal, so a groove on a slope stays where the
//! wheel was instead of climbing the mesh.

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

/// Metres per second of berm soil that can join a neighbouring slurry.
/// The same viscosity that slows the pipe keeps the creep slow.
pub fn berm_creep_rate(kind: SurfaceKind, moisture: f32) -> f32 {
    let kind = paste_kind(kind);
    let visc = slurry_viscosity(kind, moisture).max(0.2);
    (1.35 / visc).clamp(0.0018, 0.012)
}

fn paste_kind(kind: SurfaceKind) -> SurfaceKind {
    match kind {
        SurfaceKind::Berm | SurfaceKind::PlacedClod | SurfaceKind::CutLip => SurfaceKind::WetMud,
        other => other,
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
    /// One slurry thickness, metres. Soil that left the bed plus the water
    /// that left the column. The mesh adds this to the solid. It is not a
    /// second surface.
    pub slurry: Vec<f32>,
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
        SurfaceKind::WetMud => base,
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
            slurry: vec![0.0; n],
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
        field
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
            let s = self.soil_for_force(center.x, center.z);
            rut_depth_from_force(
                normal,
                push,
                radius,
                width,
                s.kind(),
                s.wetness,
                s.loose,
                s.water,
            )
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
        // lip and the placed berm into one slope.
        let half_w = if lane { self.cell * 1.15 } else { sigma * 1.6 };
        let lip_band = if lane { self.cell * 1.05 } else { 0.0 };
        // Spoil sits in the cell against the lip. A wider band flattened
        // the pile into the crust, so the berm disappeared beside the step.
        let berm_band = self.cell * 0.9;
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
                let penetration = self.height[idx] - (center.y - radius);
                let up_the_slope = penetration > sinkage + 0.03 && lz.abs() > self.cell * 0.5;
                if up_the_slope {
                    continue;
                }
                let ax = lx.abs();
                let across = if lane {
                    if ax <= half_w {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    (-0.5 * (lx / sigma.max(0.02)).powi(2)).exp()
                };

                if across > 0.2 {
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
                } else if ax > half_w + lip_band && ax <= half_w + lip_band + berm_band && along > 0.25
                {
                    if n_berm < berms.len() {
                        berms[n_berm] = idx;
                        n_berm += 1;
                    }
                }
            }
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

    /// Cubic metres of placed soil whose cell centers lie in the rectangle.
    pub fn placed_volume_in(&self, x0: f32, x1: f32, z0: f32, z1: f32) -> f32 {
        self.column_volume_in(x0, x1, z0, z1, &self.placed)
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
                if slurry0[nidx] < 0.004 {
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

    /// A tire in slurry. Slurry leaves the contact and piles ahead as a bow.
    /// The thinned column under the tire is the splash track. The bed is not
    /// cut, nothing is sprayed, and nothing is pulled up out of the rut.
    /// Returns the bow height in metres: the most slurry added on one cell.
    pub fn raise_bow_wave(
        &mut self,
        x: f32,
        z: f32,
        forward: Vec3,
        radius: f32,
        width: f32,
    ) -> f32 {
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
        // Ahead of the rubber, so the wave is in front of the tread and not under it.
        let bow_near = radius * 0.95;
        let bow_far = radius * 2.2;

        let mut taken: Vec<(usize, f32)> = Vec::new();
        let mut weights: Vec<(usize, f32)> = Vec::new();
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
                    let take = self.slurry[idx] * 0.64;
                    if take > 1.0e-6 {
                        taken.push((idx, take));
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
            return 0.0;
        }
        for (idx, take) in &taken {
            self.slurry[*idx] -= *take;
        }
        let mut bow = 0.0_f32;
        for (idx, w) in &weights {
            let add = removed * (*w / wsum);
            self.slurry[*idx] += add;
            bow = bow.max(add);
        }
        for (idx, _) in taken.iter().chain(weights.iter()) {
            self.sync(*idx);
        }
        bow
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
            soil += self.bed[idx] + self.placed[idx];
            water += self.water[idx];
            slurry += self.slurry[idx];
        }
        MixVolumes {
            soil: soil * area,
            water: water * area,
            slurry: slurry * area,
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
            let target = if wetting {
                wet_target(kind)
            } else {
                dry_target(kind, self.base_wetness[idx])
            };
            let rate = if wetting {
                wet_rate(kind)
            } else {
                dry_rate(kind)
            };
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
        }
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
        let (tread_amp, tread_ang, tread_lat, slurry) = if remain == 0 {
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n])
        } else if remain == n * 12 {
            let tread_amp = read_f32s(bytes, &mut i, n)?;
            let tread_ang = read_f32s(bytes, &mut i, n)?;
            let tread_lat = read_f32s(bytes, &mut i, n)?;
            (tread_amp, tread_ang, tread_lat, vec![0.0; n])
        } else if remain == n * 16 {
            let tread_amp = read_f32s(bytes, &mut i, n)?;
            let tread_ang = read_f32s(bytes, &mut i, n)?;
            let tread_lat = read_f32s(bytes, &mut i, n)?;
            let slurry = read_f32s(bytes, &mut i, n)?;
            (tread_amp, tread_ang, tread_lat, slurry)
        } else {
            return Err("soil snapshot has trailing bytes".into());
        };
        let mut height = vec![0.0; n];
        for idx in 0..n {
            height[idx] = bed[idx] + placed[idx] + slurry[idx];
        }
        Ok(Self {
            res,
            size,
            cell: size / res as f32,
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
            slurry,
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
        let legacy_len = saved.len() - cells * 16;
        let legacy = SoilField::load_state(&saved[..legacy_len]).expect("old snapshot");
        assert_eq!(legacy.tread_amp[idx], 0.0);
        assert_eq!(legacy.slurry[idx], 0.0);
        assert!((legacy.bed[idx] - soil.bed[idx]).abs() < 1.0e-6);
        let prior = SoilField::load_state(&saved[..saved.len() - cells * 4]).expect("pre-slurry");
        assert_eq!(prior.slurry[idx], 0.0);
        assert!((prior.tread_amp[idx] - amp).abs() < 1.0e-6);
        assert!((loaded.slurry[idx] - soil.slurry[idx]).abs() < 1.0e-6);
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
        let bow = soil.raise_bow_wave(tire.1, tire.2, Vec3::Z, 0.38, 0.28);
        let track1 = soil.slurry[idx];
        eprintln!(
            "bow wave height {bow:.4} m   splash track {track0:.4} -> {track1:.4} m   ahead {ahead0:.4} -> {:.4} m",
            soil.slurry[ahead]
        );
        assert!(bow > 0.01, "the tire did not raise a bow wave, {bow:.4} m");
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
}
