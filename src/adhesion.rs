//! Mud adhesion on the tire, the arches, the belly, the sills, and the body.
//!
//! Sticky clay at mid moisture is what sticks. A slurry runs off. Dry sand
//! barely holds. The layer is real volume taken from the loose film: what
//! left the ground is either still stuck or has been shed. Spin throws that
//! shed as ribbons and clods. Standing water washes it off. A dry contact
//! cracks it off. The stuck mass rides on the chassis, a clump on the tread
//! unbalances the wheel, a packed arch can brake that wheel, and a packed
//! void lowers μ.
//!
//! [`Adhesion::shader_layers`] is the export a body shader can read. Thickness
//! is metres. Clay, silt, sand, and water are the fraction of that zone.
//! This module does not move the slurry surface.

use glam::Vec3;

use crate::soil::{grip_mu, liquid_regime, LiquidRegime, SoilField, SurfaceKind};

/// Zones, in shader order.
pub const ZONE_TIRE: usize = 0;
pub const ZONE_ARCH: usize = 1;
pub const ZONE_BELLY: usize = 2;
pub const ZONE_SILL: usize = 3;
pub const ZONE_BODY: usize = 4;
pub const ZONE_COUNT: usize = 5;

pub const ZONE_NAMES: [&str; ZONE_COUNT] = ["tire", "arch", "belly", "sill", "body"];

/// One tire, crown plus both sidewalls, square metres.
pub const TIRE_AREA: f32 = 0.72;
/// Inner face of one arch.
pub const ARCH_AREA: f32 = 0.36;
pub const BELLY_AREA: f32 = 1.40;
/// One sill.
pub const SILL_AREA: f32 = 0.46;
pub const BODY_AREA: f32 = 2.60;

const TIRE_MAX: f32 = 0.016;
const ARCH_MAX: f32 = 0.030;
const BELLY_MAX: f32 = 0.014;
const SILL_MAX: f32 = 0.012;
const BODY_MAX: f32 = 0.008;

/// Arch thickness that starts to drag the spinning wheel, metres.
pub const ARCH_BRAKE_AT: f32 = 0.010;

const PARTS: usize = 12;
const TIRE0: usize = 0;
const ARCH0: usize = 4;
const PART_BELLY: usize = 8;
const PART_SILL_L: usize = 9;
const PART_SILL_R: usize = 10;
const PART_BODY: usize = 11;

/// Live driving. Small enough that a lap does not pack the voids.
pub const LIVE_OFFER_M3_S: f32 = 0.00012;
/// A scripted clay pass. Fills the tread and the arches.
pub const PASS_OFFER_M3_S: f32 = 0.0060;

/// Reference spin for the unbalance number in a report, radians per second.
pub const IMBALANCE_OMEGA: f32 = 18.0;

/// Soil in one layer, cubic metres. The four components are the whole volume.
#[derive(Clone, Copy, Debug, Default)]
pub struct Composition {
    pub clay: f32,
    pub silt: f32,
    pub sand: f32,
    pub water: f32,
}

impl Composition {
    pub fn volume(self) -> f32 {
        (self.clay + self.silt + self.sand + self.water).max(0.0)
    }

    pub fn mass_kg(self) -> f32 {
        self.clay * 1_760.0 + self.silt * 1_550.0 + self.sand * 1_600.0 + self.water * 1_000.0
    }

    fn add(&mut self, other: Self) {
        self.clay += other.clay;
        self.silt += other.silt;
        self.sand += other.sand;
        self.water += other.water;
    }

    /// Take `frac` of what is here. `frac` is clamped to 0..1.
    fn take(&mut self, frac: f32) -> Self {
        let f = frac.clamp(0.0, 1.0);
        if f <= 0.0 || self.volume() <= 0.0 {
            return Self::default();
        }
        let out = Self {
            clay: self.clay * f,
            silt: self.silt * f,
            sand: self.sand * f,
            water: self.water * f,
        };
        self.clay -= out.clay;
        self.silt -= out.silt;
        self.sand -= out.sand;
        self.water -= out.water;
        out
    }

    fn scale(self, gain: f32) -> Self {
        let g = gain.max(0.0);
        Self {
            clay: self.clay * g,
            silt: self.silt * g,
            sand: self.sand * g,
            water: self.water * g,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Part {
    mix: Composition,
}

/// Thickness and composition of one zone, for a shader.
///
/// `clay`, `silt`, `sand`, and `water` are fractions of the zone. They sum
/// to 1 when the zone holds soil, and they are 0 when it is clean.
#[derive(Clone, Copy, Debug)]
pub struct ShaderLayer {
    pub thickness_m: f32,
    pub clay: f32,
    pub silt: f32,
    pub sand: f32,
    pub water: f32,
    pub mass_kg: f32,
}

impl ShaderLayer {
    fn from_mix(mix: Composition, area: f32) -> Self {
        let volume = mix.volume();
        let inv = if volume > 1.0e-8 { 1.0 / volume } else { 0.0 };
        Self {
            thickness_m: if area > 1.0e-4 { volume / area } else { 0.0 },
            clay: mix.clay * inv,
            silt: mix.silt * inv,
            sand: mix.sand * inv,
            water: mix.water * inv,
            mass_kg: mix.mass_kg(),
        }
    }
}

/// What one step did to the coat. The still reads this; the lap only needs
/// the groove and the panels.
#[derive(Clone, Copy, Debug)]
pub struct AdherePaint {
    /// Void fill, 0..1, one entry per tire.
    pub groove: [f32; 4],
    /// Sidewall coat, 0..1.
    pub sidewall: [f32; 4],
    /// Lug film, kept under the lap's smear gate.
    pub smear: [f32; 4],
    /// Body-panel cover, 0..1, in the vehicle panel order.
    pub panel: [f32; 12],
}

/// Numbers for a pass: zones, mass, shed, μ, arch brake, unbalance.
#[derive(Clone, Copy, Debug)]
pub struct AdhesionReport {
    pub zones: [ShaderLayer; ZONE_COUNT],
    pub stuck_kg: f32,
    pub stuck_m3: f32,
    pub removed_m3: f32,
    pub shed_m3: f32,
    pub shed_ribbon_m3: f32,
    pub shed_clod_m3: f32,
    /// μ of clean mid-moisture clay.
    pub mu_clean: f32,
    /// μ once the tread void is packed. `mu_clean` times [`tread_mu_scale`].
    pub mu_packed: f32,
    /// μ at the tread fill this coat actually reached.
    pub mu_now: f32,
    pub tire_pack: f32,
    /// Peak arch-brake torque on the four wheels, newton-metres.
    pub arch_brake_nm: f32,
    /// Peak rotating unbalance at [`IMBALANCE_OMEGA`], newtons.
    pub imbalance_n: f32,
}

/// One tire's contact, already solved. `slurry` is read so a slurry can run
/// off. This step does not write the slurry surface.
#[derive(Clone, Copy, Debug)]
pub struct AdhereContact {
    pub index: usize,
    pub active: bool,
    pub slip: f32,
    pub speed: f32,
    pub length: f32,
    pub width: f32,
    pub center: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub kind: SurfaceKind,
    pub wetness: f32,
    pub water: f32,
    pub slurry: f32,
}

impl Default for AdhereContact {
    fn default() -> Self {
        Self {
            index: 0,
            active: false,
            slip: 0.0,
            speed: 0.0,
            length: 0.16,
            width: 0.28,
            center: Vec3::ZERO,
            forward: Vec3::Z,
            right: Vec3::X,
            kind: SurfaceKind::WetMud,
            wetness: 0.0,
            water: 0.0,
            slurry: 0.0,
        }
    }
}

/// Stuck soil. `removed` is everything taken off the ground. `shed` is
/// everything that later left the vehicle. The two plus what is still stuck
/// are the same soil.
#[derive(Clone, Debug)]
pub struct Adhesion {
    parts: [Part; PARTS],
    removed: f32,
    shed: f32,
    shed_ribbon: f32,
    shed_clod: f32,
}

impl Default for Adhesion {
    fn default() -> Self {
        Self {
            parts: [Part::default(); PARTS],
            removed: 0.0,
            shed: 0.0,
            shed_ribbon: 0.0,
            shed_clod: 0.0,
        }
    }
}

impl Adhesion {
    pub fn stuck_volume(&self) -> f32 {
        self.parts.iter().map(|p| p.mix.volume()).sum()
    }

    pub fn mass_kg(&self) -> f32 {
        self.parts.iter().map(|p| p.mix.mass_kg()).sum()
    }

    pub fn removed_m3(&self) -> f32 {
        self.removed
    }

    pub fn shed_m3(&self) -> f32 {
        self.shed
    }

    pub fn shed_ribbon_m3(&self) -> f32 {
        self.shed_ribbon
    }

    pub fn shed_clod_m3(&self) -> f32 {
        self.shed_clod
    }

    /// `removed - stuck - shed`. Zero when the account is closed.
    pub fn ledger_gap(&self) -> f32 {
        self.removed - self.stuck_volume() - self.shed
    }

    /// Spawn settle is not a pass. Hand the stuck volume back and zero the account.
    pub fn take_all(&mut self) -> f32 {
        let stuck = self.stuck_volume();
        *self = Self::default();
        stuck
    }

    pub fn tire_pack(&self, wheel: usize) -> f32 {
        let cap = TIRE_AREA * TIRE_MAX;
        if cap <= 1.0e-6 {
            return 0.0;
        }
        (self.parts[TIRE0 + wheel].mix.volume() / cap).clamp(0.0, 1.0)
    }

    pub fn mean_tire_pack(&self) -> f32 {
        (0..4).map(|i| self.tire_pack(i)).sum::<f32>() / 4.0
    }

    fn thickness_of(&self, part: usize, area: f32) -> f32 {
        self.parts[part].mix.volume() / area.max(1.0e-4)
    }

    fn zone_mix(&self, zone: usize) -> (Composition, f32) {
        let (range, area) = match zone {
            ZONE_TIRE => (TIRE0..ARCH0, TIRE_AREA * 4.0),
            ZONE_ARCH => (ARCH0..PART_BELLY, ARCH_AREA * 4.0),
            ZONE_BELLY => (PART_BELLY..PART_SILL_L, BELLY_AREA),
            ZONE_SILL => (PART_SILL_L..PART_BODY, SILL_AREA * 2.0),
            _ => (PART_BODY..PARTS, BODY_AREA),
        };
        let mut mix = Composition::default();
        for i in range {
            mix.add(self.parts[i].mix);
        }
        (mix, area)
    }

    /// Five zones. This is the shader export.
    pub fn shader_layers(&self) -> [ShaderLayer; ZONE_COUNT] {
        let mut out = [ShaderLayer::from_mix(Composition::default(), 1.0); ZONE_COUNT];
        for zone in 0..ZONE_COUNT {
            let (mix, area) = self.zone_mix(zone);
            out[zone] = ShaderLayer::from_mix(mix, area);
        }
        out
    }

    /// Torque opposing the wheel once that arch is packed, newton-metres.
    pub fn arch_brake_nm(&self, wheel: usize) -> f32 {
        let thick = self.thickness_of(ARCH0 + wheel, ARCH_AREA);
        let excess = (thick - ARCH_BRAKE_AT).max(0.0);
        (excess * 32_000.0).min(480.0)
    }

    pub fn peak_arch_brake_nm(&self) -> f32 {
        (0..4).map(|i| self.arch_brake_nm(i)).fold(0.0, f32::max)
    }

    /// Rotating unbalance from the clump on one tire, newtons.
    /// Eccentricity grows with how full the void is.
    pub fn imbalance_n(&self, wheel: usize, omega: f32) -> f32 {
        let pack = self.tire_pack(wheel);
        let mass = self.parts[TIRE0 + wheel].mix.mass_kg();
        let e = 0.032 * pack;
        (mass * e * omega * omega).min(1_200.0)
    }

    pub fn peak_imbalance_n(&self, omega: f32) -> f32 {
        (0..4).map(|i| self.imbalance_n(i, omega)).fold(0.0, f32::max)
    }

    pub fn tire_mass_kg(&self, wheel: usize) -> f32 {
        self.parts[TIRE0 + wheel].mix.mass_kg()
    }

    /// Cover for the existing body draw. 0 is clean, 1 is a full layer.
    pub fn paint(&self) -> AdherePaint {
        let mut panel = [0.0_f32; 12];
        let mut groove = [0.0_f32; 4];
        let mut sidewall = [0.0_f32; 4];
        let mut smear = [0.0_f32; 4];
        for i in 0..4 {
            let pack = self.tire_pack(i);
            groove[i] = pack;
            sidewall[i] = (pack * 0.92).min(1.0);
            smear[i] = (pack * 0.40).min(0.55);
            let arch = (self.thickness_of(ARCH0 + i, ARCH_AREA) / ARCH_MAX).clamp(0.0, 1.0);
            panel[6 + i] = arch;
        }
        let belly = (self.thickness_of(PART_BELLY, BELLY_AREA) / BELLY_MAX).clamp(0.0, 1.0);
        let sill_l = (self.thickness_of(PART_SILL_L, SILL_AREA) / SILL_MAX).clamp(0.0, 1.0);
        let sill_r = (self.thickness_of(PART_SILL_R, SILL_AREA) / SILL_MAX).clamp(0.0, 1.0);
        let body = (self.thickness_of(PART_BODY, BODY_AREA) / BODY_MAX).clamp(0.0, 1.0);
        panel[5] = belly;
        panel[2] = sill_l.max(panel[6]).max(panel[8]);
        panel[3] = sill_r.max(panel[7]).max(panel[9]);
        panel[1] = body;
        panel[4] = body * 0.85;
        panel[0] = body * 0.35;
        AdherePaint {
            groove,
            sidewall,
            smear,
            panel,
        }
    }

    pub fn report(&self, mu_clean: f32) -> AdhesionReport {
        let pack = self.mean_tire_pack();
        AdhesionReport {
            zones: self.shader_layers(),
            stuck_kg: self.mass_kg(),
            stuck_m3: self.stuck_volume(),
            removed_m3: self.removed,
            shed_m3: self.shed,
            shed_ribbon_m3: self.shed_ribbon,
            shed_clod_m3: self.shed_clod,
            mu_clean,
            mu_packed: tread_mu(mu_clean, 1.0),
            mu_now: tread_mu(mu_clean, pack),
            tire_pack: pack,
            arch_brake_nm: self.peak_arch_brake_nm(),
            imbalance_n: self.peak_imbalance_n(IMBALANCE_OMEGA),
        }
    }

    fn room(&self, part: usize) -> f32 {
        let cap = part_cap(part);
        (cap - self.parts[part].mix.volume()).max(0.0)
    }

    fn release(&mut self, part: usize, frac: f32, ribbons: bool) -> f32 {
        let gone = self.parts[part].mix.take(frac);
        let volume = gone.volume();
        if volume <= 1.0e-10 {
            return 0.0;
        }
        self.shed += volume;
        if ribbons {
            self.shed_ribbon += volume;
        } else {
            self.shed_clod += volume;
        }
        volume
    }
}

fn part_cap(part: usize) -> f32 {
    match part {
        0..=3 => TIRE_AREA * TIRE_MAX,
        4..=7 => ARCH_AREA * ARCH_MAX,
        PART_BELLY => BELLY_AREA * BELLY_MAX,
        PART_SILL_L | PART_SILL_R => SILL_AREA * SILL_MAX,
        _ => BODY_AREA * BODY_MAX,
    }
}

/// How much of an offered film sticks. One is a grab, zero runs off.
///
/// The peak is clay near pore wetness 0.45. A slurry is the floor. Dry sand
/// sits on that floor. Silt sticks, and less than the clay.
pub fn stick_hold(kind: SurfaceKind, wetness: f32, water: f32, slurry: f32) -> f32 {
    if slurry > 0.02 || liquid_regime(kind, water) == LiquidRegime::Slurry {
        return 0.015;
    }
    let w = wetness.clamp(0.0, 1.0);
    let mid = (-0.5 * ((w - 0.45) / 0.16).powi(2)).exp();
    let affinity = match kind {
        SurfaceKind::WetMud
        | SurfaceKind::FreshCut
        | SurfaceKind::PackedClay
        | SurfaceKind::CutLip
        | SurfaceKind::Berm
        | SurfaceKind::PlacedClod => 1.0,
        SurfaceKind::Silt => 0.62,
        SurfaceKind::Sand => 0.11,
        SurfaceKind::DryCrust => 0.07,
        SurfaceKind::Gravel => 0.035,
        SurfaceKind::Grass => 0.12,
        SurfaceKind::Undisturbed => 0.30,
    };
    let dry_sand = if kind == SurfaceKind::Sand && w < 0.22 {
        0.22
    } else {
        1.0
    };
    (affinity * mid * dry_sand).clamp(0.0, 1.0)
}

/// μ of a tread. `pack` is 0 on a clean void and 1 when the void is full.
/// A full pack shears soil against soil, so μ falls to 56% of the clean face.
pub fn tread_mu_scale(pack: f32) -> f32 {
    let t = pack.clamp(0.0, 1.0);
    if t < 0.02 {
        return 1.0;
    }
    let u = ((t - 0.02) / 0.98).clamp(0.0, 1.0);
    1.0 - 0.44 * u
}

pub fn tread_mu(clean_mu: f32, pack: f32) -> f32 {
    clean_mu.max(0.0) * tread_mu_scale(pack)
}

/// Clean μ of mid-moisture clay, the reference for a packed-tread comparison.
pub fn mid_clay_mu() -> f32 {
    grip_mu(SurfaceKind::WetMud, 0.46, 0.02, 0.0)
}

fn compose(kind: SurfaceKind, wetness: f32, water: f32, volume: f32) -> Composition {
    if volume <= 0.0 {
        return Composition::default();
    }
    let pore = (0.12 + 0.28 * wetness.clamp(0.0, 1.0) + 0.22 * (water / 0.08).clamp(0.0, 1.0))
        .clamp(0.08, 0.55);
    let solid = volume * (1.0 - pore);
    let (c, s, a) = solid_split(kind);
    Composition {
        clay: solid * c,
        silt: solid * s,
        sand: solid * a,
        water: volume - solid,
    }
}

fn solid_split(kind: SurfaceKind) -> (f32, f32, f32) {
    match kind {
        SurfaceKind::Sand | SurfaceKind::Gravel => (0.04, 0.08, 0.88),
        SurfaceKind::Silt => (0.16, 0.76, 0.08),
        SurfaceKind::Grass | SurfaceKind::Undisturbed => (0.46, 0.34, 0.20),
        _ => (0.84, 0.12, 0.04),
    }
}

fn ribbon_share(mix: Composition) -> f32 {
    let v = mix.volume();
    if v <= 1.0e-8 {
        return 0.7;
    }
    let clay = mix.clay / v;
    let sand = mix.sand / v;
    let water = mix.water / v;
    if sand > 0.55 && water < 0.18 {
        return 0.18;
    }
    if clay > 0.45 || water > 0.20 {
        return 0.74;
    }
    0.42
}

fn weights(wheel: usize) -> (f32, f32, f32, f32, f32) {
    // tire, arch, belly, sill, body
    if wheel < 2 {
        (0.44, 0.26, 0.08, 0.12, 0.10)
    } else {
        (0.40, 0.30, 0.14, 0.10, 0.06)
    }
}

/// Lift, stick, shed. `offer_m3_s` is the cubic metres one wheel can lift
/// in a second when [`stick_hold`] is 1. Shed soil goes back to the loose
/// film, so the heightfield's cut and slurry are not the adhesion account.
pub fn step(
    coat: &mut Adhesion,
    soil: &mut SoilField,
    contacts: &[AdhereContact; 4],
    offer_m3_s: f32,
    dt: f32,
) -> AdherePaint {
    let dt = dt.max(0.0);
    if dt <= 0.0 {
        return coat.paint();
    }
    let mut put_back = 0.0_f32;
    let mut put_at = Vec3::ZERO;
    let mut put_n = 0.0_f32;

    for c in contacts.iter() {
        if !c.active || c.index > 3 {
            continue;
        }
        let wheel = c.index;
        let hold = stick_hold(c.kind, c.wetness, c.water, c.slurry);
        let want = (offer_m3_s * hold * dt).clamp(0.0, 0.0025);
        if want > 1.0e-8 {
            let length = c.length.max(0.08);
            let width = c.width.max(0.08);
            let taken = soil.take_loose(c.center, c.forward, c.right, length, width, want);
            if taken > 1.0e-10 {
                let mix = compose(c.kind, c.wetness, c.water, taken);
                coat.removed += taken;
                put_back += deposit(coat, wheel, mix);
                put_at += c.center;
                put_n += 1.0;
            }
        }

        let slurry = c.slurry > 0.02 || liquid_regime(c.kind, c.water) == LiquidRegime::Slurry;
        if slurry {
            let frac = 1.0 - (-3.4 * dt).exp();
            put_back += release_pair(coat, wheel, frac, true);
            put_back += coat.release(PART_BELLY, frac * 0.8, true);
            let sill = if wheel % 2 == 0 {
                PART_SILL_L
            } else {
                PART_SILL_R
            };
            put_back += coat.release(sill, frac * 0.8, true);
            put_back += coat.release(PART_BODY, frac * 0.45, true);
        }

        let spin = ((c.slip.abs() - 0.32) / 0.40).clamp(0.0, 1.0);
        if spin > 1.0e-3 {
            let frac = 1.0 - (-2.8 * spin * dt).exp();
            put_back += release_split(coat, TIRE0 + wheel, frac);
            put_back += release_split(coat, ARCH0 + wheel, frac * 0.55);
        }

        if c.water > 0.028 && !slurry {
            let depth = (c.water / 0.08).clamp(0.35, 2.0);
            let frac = 1.0 - (-1.8 * depth * dt).exp();
            let ribbons = ribbon_share(coat.parts[TIRE0 + wheel].mix) >= 0.5;
            put_back += release_pair(coat, wheel, frac, ribbons);
            put_back += coat.release(PART_BELLY, frac * 0.75, ribbons);
            let sill = if wheel % 2 == 0 {
                PART_SILL_L
            } else {
                PART_SILL_R
            };
            put_back += coat.release(sill, frac * 0.75, ribbons);
            put_back += coat.release(PART_BODY, frac * 0.4, ribbons);
        }

        put_at += c.center;
        put_n += 1.0;
    }

    let mean_wet = mean_active(contacts, |c| c.wetness);
    let mean_water = mean_active(contacts, |c| c.water);
    let any = contacts.iter().any(|c| c.active);
    if any && mean_wet < 0.18 && mean_water < 0.006 {
        let frac = 1.0 - (-1.25 * dt).exp();
        for part in 0..PARTS {
            put_back += coat.release(part, frac, false);
        }
    }

    if put_back > 1.0e-10 {
        let at = if put_n > 0.0 {
            put_at / put_n
        } else {
            Vec3::ZERO
        };
        soil.give_loose(at, put_back);
    }
    coat.paint()
}

fn mean_active(contacts: &[AdhereContact; 4], f: impl Fn(&AdhereContact) -> f32) -> f32 {
    let mut acc = 0.0;
    let mut n = 0.0;
    for c in contacts {
        if c.active {
            acc += f(c);
            n += 1.0;
        }
    }
    if n > 0.0 {
        acc / n
    } else {
        1.0
    }
}

/// Stick `mix` onto the wheel's zones. The return is volume that did not fit,
/// already counted as clods.
fn deposit(coat: &mut Adhesion, wheel: usize, mix: Composition) -> f32 {
    let (tire, arch, belly, sill, body) = weights(wheel);
    let sill_part = if wheel % 2 == 0 {
        PART_SILL_L
    } else {
        PART_SILL_R
    };
    let plan = [
        (TIRE0 + wheel, tire),
        (ARCH0 + wheel, arch),
        (PART_BELLY, belly),
        (sill_part, sill),
        (PART_BODY, body),
    ];
    let mut overflow = 0.0_f32;
    for (part, share) in plan {
        let piece = mix.scale(share);
        let volume = piece.volume();
        if volume <= 1.0e-12 {
            continue;
        }
        let room = coat.room(part);
        if volume <= room + 1.0e-12 {
            coat.parts[part].mix.add(piece);
            continue;
        }
        let keep = if volume > 1.0e-12 { room / volume } else { 0.0 };
        let kept = piece.scale(keep);
        let left = piece.volume() - kept.volume();
        coat.parts[part].mix.add(kept);
        if left > 1.0e-10 {
            coat.shed += left;
            coat.shed_clod += left;
            overflow += left;
        }
    }
    overflow
}

fn release_pair(coat: &mut Adhesion, wheel: usize, frac: f32, ribbons: bool) -> f32 {
    coat.release(TIRE0 + wheel, frac, ribbons) + coat.release(ARCH0 + wheel, frac, ribbons)
}

/// `frac` of the part leaves. Ribbons and clods split by the mix.
fn release_split(coat: &mut Adhesion, part: usize, frac: f32) -> f32 {
    let frac = frac.clamp(0.0, 1.0);
    if frac <= 0.0 {
        return 0.0;
    }
    let share = ribbon_share(coat.parts[part].mix);
    let ribbon_frac = frac * share;
    let mut volume = coat.release(part, ribbon_frac, true);
    let still = (1.0 - ribbon_frac).max(1.0e-6);
    let clod_frac = (frac * (1.0 - share) / still).clamp(0.0, 1.0);
    volume += coat.release(part, clod_frac, false);
    volume
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::{Contact, Vehicle, MASS};

    fn pad(kind: SurfaceKind, wetness: f32, water: f32, loose: f32) -> SoilField {
        let mut soil = SoilField::flat(0.4);
        for i in 0..soil.kind.len() {
            soil.kind[i] = kind as u8;
            soil.wetness[i] = wetness;
            soil.water[i] = water;
            soil.loose[i] = loose;
        }
        soil
    }

    fn rolling(kind: SurfaceKind, wetness: f32, water: f32, slurry: f32) -> [AdhereContact; 4] {
        let mut contacts = [AdhereContact::default(); 4];
        let xs = [-0.74_f32, 0.74, -0.74, 0.74];
        for i in 0..4 {
            contacts[i] = AdhereContact {
                index: i,
                active: true,
                slip: 0.08,
                speed: 1.6,
                length: 0.22,
                width: 0.28,
                center: Vec3::new(xs[i], 0.3, 0.0),
                forward: Vec3::Z,
                right: Vec3::X,
                kind,
                wetness,
                water,
                slurry,
            };
        }
        contacts
    }

    fn drive(soil: &mut SoilField, kind: SurfaceKind, wetness: f32, water: f32, slurry: f32, seconds: f32) -> Adhesion {
        let mut coat = Adhesion::default();
        let mut contacts = rolling(kind, wetness, water, slurry);
        let dt = 1.0 / 60.0;
        let steps = (seconds / dt).round() as u32;
        for s in 0..steps {
            let z = -1.5 + (s as f32) * dt * 1.6;
            for c in &mut contacts {
                c.center.z = z;
            }
            step(&mut coat, soil, &contacts, PASS_OFFER_M3_S, dt);
        }
        coat
    }

    #[test]
    fn sticky_clay_at_mid_moisture_beats_slurry_and_dry_sand() {
        let clay = stick_hold(SurfaceKind::WetMud, 0.45, 0.0, 0.0);
        let sat = stick_hold(SurfaceKind::WetMud, 0.90, 0.0, 0.0);
        let slurry = stick_hold(SurfaceKind::WetMud, 0.92, 0.12, 0.08);
        let dry_sand = stick_hold(SurfaceKind::Sand, 0.06, 0.0, 0.0);
        let damp_sand = stick_hold(SurfaceKind::Sand, 0.45, 0.0, 0.0);
        let silt = stick_hold(SurfaceKind::Silt, 0.45, 0.0, 0.0);
        assert!(clay > 0.85, "mid clay {clay}");
        assert!(clay > silt * 1.4, "clay {clay} silt {silt}");
        assert!(clay > sat * 8.0, "saturated clay still grabbed, {sat}");
        assert!(slurry < 0.03, "slurry stuck {slurry}");
        assert!(dry_sand < 0.04, "dry sand held {dry_sand}");
        assert!(clay > dry_sand * 20.0, "clay {clay} sand {dry_sand}");
        assert!(damp_sand < clay * 0.2, "damp sand {damp_sand}");
        assert!(damp_sand > dry_sand, "damp sand should beat dry sand");
    }

    #[test]
    fn clay_pass_sticks_more_than_sand_and_balances() {
        let mut clay_soil = pad(SurfaceKind::WetMud, 0.46, 0.0, 0.08);
        let mut sand_soil = pad(SurfaceKind::Sand, 0.06, 0.0, 0.08);
        let clay = drive(&mut clay_soil, SurfaceKind::WetMud, 0.46, 0.0, 0.0, 3.2);
        let sand = drive(&mut sand_soil, SurfaceKind::Sand, 0.06, 0.0, 0.0, 3.2);
        assert!(
            clay.ledger_gap().abs() < 1.0e-4,
            "clay ledger {}",
            clay.ledger_gap()
        );
        assert!(
            sand.ledger_gap().abs() < 1.0e-4,
            "sand ledger {}",
            sand.ledger_gap()
        );
        assert!(
            clay.stuck_volume() > sand.stuck_volume() * 8.0,
            "clay {:.4} sand {:.4}",
            clay.stuck_volume(),
            sand.stuck_volume()
        );
        assert!(clay.mass_kg() > 8.0, "mass {}", clay.mass_kg());
        let layers = clay.shader_layers();
        for (i, layer) in layers.iter().enumerate() {
            assert!(
                layer.thickness_m > 0.001,
                "{} thickness {}",
                ZONE_NAMES[i],
                layer.thickness_m
            );
            let sum = layer.clay + layer.silt + layer.sand + layer.water;
            assert!((sum - 1.0).abs() < 1.0e-3, "{} composition {sum}", ZONE_NAMES[i]);
        }
        assert!(layers[ZONE_TIRE].clay > 0.5, "tire clay {}", layers[ZONE_TIRE].clay);
    }

    #[test]
    fn slurry_runs_off_a_coat() {
        let mut soil = pad(SurfaceKind::WetMud, 0.46, 0.0, 0.08);
        let mut coat = drive(&mut soil, SurfaceKind::WetMud, 0.46, 0.0, 0.0, 1.6);
        let before = coat.stuck_volume();
        assert!(before > 0.004, "no coat to lose, {before}");
        let mut contacts = rolling(SurfaceKind::WetMud, 0.92, 0.14, 0.10);
        let dt = 1.0 / 30.0;
        for _ in 0..24 {
            step(&mut coat, &mut soil, &contacts, PASS_OFFER_M3_S, dt);
            for c in &mut contacts {
                c.center.z += 0.05;
            }
        }
        assert!(
            coat.stuck_volume() < before * 0.35,
            "slurry left {:.4} of {:.4}",
            coat.stuck_volume(),
            before
        );
        assert!(coat.shed_ribbon_m3() > coat.shed_clod_m3(), "runoff was clods");
        assert!(coat.ledger_gap().abs() < 1.0e-4, "ledger {}", coat.ledger_gap());
    }

    #[test]
    fn spin_sheds_ribbons_and_clods() {
        let mut soil = pad(SurfaceKind::WetMud, 0.46, 0.0, 0.08);
        let mut coat = drive(&mut soil, SurfaceKind::WetMud, 0.46, 0.0, 0.0, 1.4);
        let mut contacts = rolling(SurfaceKind::WetMud, 0.46, 0.0, 0.0);
        for c in &mut contacts {
            c.slip = 0.95;
        }
        let dt = 1.0 / 60.0;
        for _ in 0..50 {
            step(&mut coat, &mut soil, &contacts, 0.0, dt);
        }
        assert!(coat.shed_ribbon_m3() > 1.0e-4, "ribbons {}", coat.shed_ribbon_m3());
        assert!(coat.shed_clod_m3() > 1.0e-5, "clods {}", coat.shed_clod_m3());
        assert!(
            coat.shed_ribbon_m3() > coat.shed_clod_m3(),
            "wet clay spin should throw ribbons"
        );
        assert!(coat.ledger_gap().abs() < 1.0e-4, "ledger {}", coat.ledger_gap());

        let mut dry = Adhesion::default();
        dry.parts[0].mix = Composition {
            clay: 0.0002,
            silt: 0.0002,
            sand: 0.0040,
            water: 0.0002,
        };
        dry.removed = dry.stuck_volume();
        let mut sand_contacts = rolling(SurfaceKind::Sand, 0.08, 0.0, 0.0);
        for c in &mut sand_contacts {
            c.slip = 0.95;
            c.active = c.index == 0;
        }
        let mut sand_soil = pad(SurfaceKind::Sand, 0.08, 0.0, 0.0);
        for _ in 0..40 {
            step(&mut dry, &mut sand_soil, &sand_contacts, 0.0, dt);
        }
        assert!(
            dry.shed_clod_m3() > dry.shed_ribbon_m3(),
            "dry sand spin ribbons {} clods {}",
            dry.shed_ribbon_m3(),
            dry.shed_clod_m3()
        );
        assert!(dry.ledger_gap().abs() < 1.0e-4, "sand ledger {}", dry.ledger_gap());
    }

    #[test]
    fn water_washes_the_coat_off() {
        let mut soil = pad(SurfaceKind::WetMud, 0.46, 0.0, 0.08);
        let mut coat = drive(&mut soil, SurfaceKind::WetMud, 0.46, 0.0, 0.0, 1.2);
        let before = coat.stuck_volume();
        let contacts = rolling(SurfaceKind::Undisturbed, 0.40, 0.05, 0.0);
        let dt = 1.0 / 30.0;
        for _ in 0..30 {
            step(&mut coat, &mut soil, &contacts, 0.0, dt);
        }
        assert!(
            coat.stuck_volume() < before * 0.45,
            "wash left {:.4} of {:.4}",
            coat.stuck_volume(),
            before
        );
        assert!(coat.shed_m3() > before * 0.4, "shed {}", coat.shed_m3());
        assert!(coat.ledger_gap().abs() < 1.0e-4, "ledger {}", coat.ledger_gap());
    }

    #[test]
    fn a_dry_contact_cracks_the_coat_off() {
        let mut soil = pad(SurfaceKind::DryCrust, 0.08, 0.0, 0.0);
        let mut coat = Adhesion::default();
        let mix = Composition {
            clay: 0.006,
            silt: 0.001,
            sand: 0.0004,
            water: 0.0012,
        };
        for i in 0..4 {
            coat.parts[i].mix = mix;
            coat.parts[ARCH0 + i].mix = mix.scale(0.4);
        }
        coat.removed = coat.stuck_volume();
        let before = coat.stuck_volume();
        let contacts = rolling(SurfaceKind::DryCrust, 0.08, 0.0, 0.0);
        let dt = 1.0 / 30.0;
        for _ in 0..50 {
            step(&mut coat, &mut soil, &contacts, 0.0, dt);
        }
        assert!(
            coat.stuck_volume() < before * 0.25,
            "dry coat stayed {:.4} of {:.4}",
            coat.stuck_volume(),
            before
        );
        assert!(coat.shed_clod_m3() > coat.shed_ribbon_m3(), "crack was ribbons");
        assert!(coat.ledger_gap().abs() < 1.0e-4, "ledger {}", coat.ledger_gap());
    }

    #[test]
    fn packed_tread_drops_mu() {
        let clean = mid_clay_mu();
        let packed = tread_mu(clean, 1.0);
        let dust = tread_mu(clean, 0.01);
        assert!(clean > 0.2, "clean μ {clean}");
        assert!(packed < clean * 0.62, "packed {packed} clean {clean}");
        assert!((dust - clean).abs() < 1.0e-4, "dust changed μ");
        assert!(tread_mu_scale(1.0) < 0.60);
    }

    #[test]
    fn packed_arch_brakes_the_wheel_and_mass_is_weight() {
        let mut coat = Adhesion::default();
        coat.parts[ARCH0].mix = compose(SurfaceKind::WetMud, 0.46, 0.0, ARCH_AREA * 0.022);
        assert!(coat.arch_brake_nm(0) > 250.0, "brake {}", coat.arch_brake_nm(0));
        assert!(coat.arch_brake_nm(1) < 1.0, "empty arch braked");
        coat.parts[ARCH0].mix = compose(SurfaceKind::WetMud, 0.46, 0.0, ARCH_AREA * 0.006);
        assert!(
            coat.arch_brake_nm(0) < 1.0,
            "thin arch braked {}",
            coat.arch_brake_nm(0)
        );

        let mut vehicle = Vehicle::new();
        vehicle.wheels[0].omega = 16.0;
        vehicle.adhesion.parts[ARCH0].mix =
            compose(SurfaceKind::WetMud, 0.46, 0.0, ARCH_AREA * 0.022);
        let soil = SoilField::flat(0.12);
        let contacts = [Contact::default(); 4];
        for _ in 0..50 {
            vehicle.integrate(&contacts, &soil, 1.0 / 60.0);
        }
        assert!(
            vehicle.wheels[0].omega.abs() < 0.4,
            "packed arch left omega {}",
            vehicle.wheels[0].omega
        );

        let mut heavy = Vehicle::new();
        heavy.adhesion.parts[PART_BELLY].mix =
            compose(SurfaceKind::WetMud, 0.46, 0.0, 0.05);
        let extra = heavy.adhesion.mass_kg();
        assert!(extra > 60.0, "mass {extra}");
        assert!((heavy.total_mass() - (MASS + extra)).abs() < 1.0e-2);
        let mut light = Vehicle::new();
        let mut supported = [Contact::default(); 4];
        for i in 0..4 {
            supported[i].active = true;
            supported[i].index = i;
            supported[i].normal = MASS * 9.81 / 4.0;
            supported[i].forward = Vec3::Z;
            supported[i].right = Vec3::X;
            supported[i].contact_point = light.pos;
        }
        // One undamped step. The springs still hold the clean weight, so only
        // the stuck mass falls, and the belly stays clear of the bed.
        light.pos.y = 2.0;
        heavy.pos.y = 2.0;
        light.integrate(&supported, &soil, 0.18);
        heavy.integrate(&supported, &soil, 0.18);
        assert!(
            light.vel.y.abs() < 0.05,
            "sprung weight drifted {}",
            light.vel.y
        );
        assert!(
            heavy.vel.y < -0.15,
            "extra mud did not add weight {}",
            heavy.vel.y
        );
    }

    #[test]
    fn a_packed_tire_unbalances_the_wheel() {
        let mut coat = Adhesion::default();
        assert!(coat.imbalance_n(0, 20.0) < 1.0e-3);
        coat.parts[0].mix = compose(SurfaceKind::WetMud, 0.46, 0.0, TIRE_AREA * TIRE_MAX);
        let force = coat.imbalance_n(0, IMBALANCE_OMEGA);
        assert!(force > 40.0, "unbalance {force}");
        assert!(coat.tire_pack(0) > 0.95);
    }
}
