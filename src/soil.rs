//! Heightfield soil: Bekker sinkage, plastic ruts, wetness and shallow water.
//!
//! The cut and the spoil are different layers. `bed` is the ground the tire
//! carved. `placed` is soil that has already been put somewhere else (berm,
//! crest clod, impact ring). The visible height is their sum, and a cell's
//! kind says which one the tire is on.

use glam::Vec3;

pub const MAP_SIZE: f32 = 40.0;
pub const MAP_RES: usize = 192;

/// Soil the tire can stand on. Liquid film and slurry sit on top of these
/// and are not stored as a second height.
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
}

/// How standing water sits on a soil kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiquidRegime {
    Dry = 0,
    Film = 1,
    Slurry = 2,
}

/// Soil kinds plus the two liquid regimes the solver treats as states.
pub const SURFACE_STATE_COUNT: usize = 12;

/// Crest chunks stay lumps. Later passes put the rest of the cut on the berm.
const CREST_CLOD_CAP: f32 = 0.04;

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
            _ => Self::Undisturbed,
        }
    }

    /// CC0 layer: 0 mud_forest, 1 brown_mud_dry, 2 clay, 3 grass, 4 gravel.
    pub fn map_layer(self) -> u32 {
        match self {
            Self::Grass => 3,
            Self::Gravel => 4,
            Self::DryCrust | Self::Undisturbed | Self::Berm => 1,
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
        SurfaceKind::CutLip | SurfaceKind::PlacedClod | SurfaceKind::Berm => 0.11,
        _ => 10.0,
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
        SurfaceKind::DryCrust => 0.002,
        SurfaceKind::Undisturbed => 0.003,
        SurfaceKind::PlacedClod => 0.003,
        SurfaceKind::Grass => 0.004,
        SurfaceKind::Berm => 0.004,
        SurfaceKind::PackedClay => 0.009,
        SurfaceKind::WetMud => 0.012,
        SurfaceKind::FreshCut => 0.02,
    }
}

/// Multiplier on the shallow-water pipe. Gravel drains. Packed clay holds a film.
pub fn water_conduct(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::Gravel => 2.4,
        SurfaceKind::CutLip => 1.7,
        SurfaceKind::Berm => 1.35,
        SurfaceKind::DryCrust => 1.25,
        SurfaceKind::PlacedClod => 1.15,
        SurfaceKind::Undisturbed => 1.0,
        SurfaceKind::Grass => 0.85,
        SurfaceKind::WetMud => 0.9,
        SurfaceKind::FreshCut => 0.5,
        SurfaceKind::PackedClay => 0.32,
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
        SurfaceKind::Gravel => scale_mech(base, 1.55, 0.22, 0.74, 0.016, 0.3, 0.18),
        SurfaceKind::DryCrust => scale_mech(base, 1.32, 1.35, 0.60, 0.012, 0.5, 0.32),
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

/// Wetness alone makes the ground heavy. Standing water and loose spoil
/// are what push it into a true bog.
pub fn softness_of(wetness: f32, loose: f32, water: f32) -> f32 {
    let loose_n = (loose / 0.09).clamp(0.0, 1.0);
    let depth = (water / 0.18).clamp(0.0, 1.0);
    (wetness.clamp(0.0, 1.0) * 0.48 + depth * 0.34 + loose_n * 0.22).clamp(0.0, 1.0)
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

    /// Flat pad with a torn rut, a grip boundary, and film beside deep slurry.
    /// The pictures are this field. The rut is stamped by the same wheel path
    /// the lap uses, so the lip is the solver's lip.
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
                if (-5.5..-3.25).contains(&x) {
                    if z < -0.15 {
                        field.kind[idx] = SurfaceKind::PackedClay as u8;
                        field.wetness[idx] = 0.55;
                        field.base_wetness[idx] = 0.5;
                        field.water[idx] = 0.036;
                        field.loose[idx] = 0.0;
                        field.roughness[idx] = 0.28;
                        field.darken[idx] = 0.8;
                    } else if z > 0.4 {
                        field.kind[idx] = SurfaceKind::WetMud as u8;
                        field.wetness[idx] = 1.0;
                        field.base_wetness[idx] = 0.95;
                        field.water[idx] = 0.16;
                        field.loose[idx] = 0.05;
                        field.roughness[idx] = 0.22;
                    } else {
                        field.kind[idx] = SurfaceKind::DryCrust as u8;
                        field.wetness[idx] = 0.22;
                        field.base_wetness[idx] = 0.2;
                        field.water[idx] = 0.022;
                        field.roughness[idx] = 0.9;
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
            field.stamp_wheel(
                Vec3::new(0.0, 0.0, z),
                0.38,
                0.28,
                forward,
                right,
                0.08,
                0.35,
                0.12,
                true,
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
        self.height[idx] = self.bed[idx] + self.placed[idx];
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
    ) {
        if sinkage < 0.004 {
            return;
        }
        let len = contact_length(radius, sinkage).max(radius * 0.35);
        let slip_n = slip.abs().min(1.2);
        // The groove is the Bekker bite. A fast straight used to leave only a few
        // millimetres: the plastic rate was tuned for a spun puddle, and the tire
        // was gone before that cut arrived, so the lane behind the vehicle stayed
        // smooth. The first several centimetres now stay in the heightfield after
        // one pass. Slip still digs deeper than that, on the slow rate, so a driver
        // can back out of a hole they are spinning.
        let groove = (sinkage * 0.82).min(0.055);
        let spun = sinkage * (0.48 + 0.90 * slip_n);
        let want = spun.max(groove).min(0.40);
        // A rolling per-tire stamp stops at a scuff the shader does not darken.
        // The visible pair is the `lane` call. Spinning still reaches `want`.
        let fast = if lane { groove } else { groove.min(0.008) };
        let sigma = (width * 0.42).max(self.cell * 0.34);
        // Flat floor, then a broken step. A ramped shoulder smeared the cut
        // lip and the placed berm into one slope.
        let half_w = if lane { self.cell * 1.15 } else { sigma * 1.6 };
        let lip_band = if lane { self.cell * 1.05 } else { 0.0 };
        // Spoil sits in the cell against the lip. A wider band flattened
        // the pile into the crust, so the berm disappeared beside the step.
        let berm_band = self.cell * 0.9;
        let half_l = if lane {
            (len * 0.5).max(self.cell * 2.2)
        } else {
            len * 0.5
        };
        let along_reach = if lane {
            half_l + self.cell * 0.8
        } else {
            half_l + self.cell * 0.35
        };
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
                    let already = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                    let local_want = if lane {
                        want * across
                    } else if spun > groove {
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
                        self.loose[idx] = (self.loose[idx] + removed * 0.45).min(0.16);
                        vol += removed * area;
                    }
                    let wet = self.wetness[idx];
                    self.darken[idx] =
                        (self.darken[idx] + 16.0 * across * along * dt * (0.45 + wet)).min(1.0);
                    let target_r = lerp(0.62, 0.16, wet);
                    let blend = (5.0 * dt * across * along).min(0.45);
                    self.roughness[idx] = lerp(self.roughness[idx], target_r, blend);
                    let prev = SurfaceKind::from_u8(self.kind[idx]);
                    // Firm covers stay themselves. A dry crust does not become mud
                    // because a tire crossed it. Standing water keeps the cut slick.
                    self.kind[idx] = if matches!(
                        prev,
                        SurfaceKind::Gravel | SurfaceKind::Grass | SurfaceKind::DryCrust
                    ) {
                        prev as u8
                    } else if self.water[idx] > 0.03 {
                        SurfaceKind::FreshCut as u8
                    } else if self.darken[idx] > 0.72 {
                        SurfaceKind::PackedClay as u8
                    } else {
                        SurfaceKind::FreshCut as u8
                    };
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
                        let bite = (0.55 + h) * groove.min(0.055);
                        let already = (self.initial_height[idx] - self.bed[idx]).max(0.0);
                        let drop = (bite - already).max(0.0).min(0.06) * along;
                        if drop > 0.001 {
                            let removed = self.excavate(idx, drop);
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
        self.deposit_spoil(vol, &clods[..n_clod], &berms[..n_berm]);
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

    /// Virtual-pipe shallow water. Dry cells are skipped.
    pub fn step_water(&mut self, dt: f32) {
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

        let n = res * res;
        for idx in 0..n {
            if self.water[idx] > 0.004 {
                self.wetness[idx] = (self.wetness[idx] + dt * 0.55).min(1.0);
            } else {
                let base = self.base_wetness[idx];
                self.wetness[idx] += (base - self.wetness[idx]) * (dt * 0.12).min(0.25);
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
    if far > 10.5 || (off > 4.8 && n2 > 0.58) {
        return SurfaceKind::Gravel;
    }
    if off > 2.5 && n > 0.46 && wet < 0.48 {
        return SurfaceKind::Grass;
    }
    if wet < 0.36 || (off > 1.45 && off < 3.3 && wet < 0.55) {
        return SurfaceKind::DryCrust;
    }
    SurfaceKind::Undisturbed
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn bilerp(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> f32 {
    let ab = a + (b - a) * tx;
    let cd = c + (d - c) * tx;
    ab + (cd - ab) * ty
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
        assert_eq!(SURFACE_STATE_COUNT, 12);
    }

    #[test]
    fn placed_spoil_is_not_the_cut_and_volume_holds() {
        let mut soil = SoilField::flat(0.7);
        let before_cut = soil.cut_volume();
        let before_placed = soil.placed_volume();
        soil.stamp_wheel(
            Vec3::new(0.2, 0.0, -0.4),
            0.38,
            0.28,
            Vec3::Z,
            Vec3::X,
            0.08,
            0.4,
            0.12,
            true,
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
                Vec3::new(0.0, 0.0, z),
                0.38,
                0.28,
                Vec3::Z,
                Vec3::X,
                0.08,
                0.35,
                0.12,
                true,
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
    }
}
