//! The tire meets the soil on a patch, not a point.
//!
//! The patch is the chord of the sunk wheel. Lug faces carry the shear.
//! The voids between them take mud when the wheel rolls, and a spinning
//! tread pulls that mud back out. Mud left on the lug face is a slicker
//! interface than the bare soil, so the same ground grips less once the
//! tread is coated. A coat also climbs the sidewall. That thickness is real
//! volume, skimmed from the loose smear, and when it sheds it is placed
//! back on the rut. It is not deleted. The cut itself is a different budget.
//! Slurry from a pull-out can sit in the same voids. It stays until the
//! wheel spins or the coat sheds, and then it leaves as ribbons.

use crate::soil::{
    contact_length, moisture_mu_scale, moisture_peak_slip, moisture_state_fields,
    sinkage_on, softness_of, stick_scale, surface_moisture, terramech_for, wheel_buoyancy,
    SurfaceKind, SurfaceMoisture,
};

/// Share of the patch that is lug. The rest is void the tread can pack.
pub const LUG_FRAC: f32 = 0.62;
/// How deep a void is, metres. Packed mud occupies this, it does not dig the bed.
pub const GROOVE_DEPTH: f32 = 0.016;
/// Slip where the patch is at peak shear. Past this the tread is spinning.
pub const PEAK_SLIP: f32 = 0.18;
/// Both sidewalls of one tire, square metres.
pub const SIDEWALL_AREA: f32 = 0.34;
/// Thickness of a full sidewall coat, metres.
pub const SIDEWALL_THICK: f32 = 0.025;
/// Cubic metres on one tire when `TreadCoat::sidewall` is 1.
pub const SIDEWALL_VOLUME: f32 = SIDEWALL_AREA * SIDEWALL_THICK;

/// Cubic metres of slurry the tread voids hold when they are full.
///
/// The open share of the crown, around the whole tire. Slurry mounds up to
/// the lug, so the hold is deeper than the dry pack in [`GROOVE_DEPTH`].
pub fn groove_slurry_capacity(radius: f32, width: f32) -> f32 {
    let radius = radius.max(0.05);
    let width = width.max(0.02);
    let depth = GROOVE_DEPTH * 1.75;
    std::f32::consts::TAU * radius * width * (1.0 - LUG_FRAC) * depth
}

/// Let slurry out of the voids. `spin` is 0 on a roll and 1 when the tread
/// is spinning. `coat_shed` is the sidewall fraction that left this step.
/// Either one opens the voids. A parked or rolling tire keeps the pack.
/// The return is cubic metres that left, for the caller to throw as ribbons.
pub fn release_groove_slurry(held: &mut f32, spin: f32, coat_shed: f32, dt: f32) -> f32 {
    let now = (*held).max(0.0);
    if now <= 1.0e-8 || dt <= 0.0 || !now.is_finite() {
        *held = now;
        return 0.0;
    }
    // Below this the tire is still rolling. Only a real spin opens the voids.
    let spin = if spin < 0.12 { 0.0 } else { spin.clamp(0.0, 1.0) };
    let coat = coat_shed.max(0.0);
    if spin < 1.0e-3 && coat <= 1.0e-6 {
        *held = now;
        return 0.0;
    }
    // Full spin empties the voids in about a second. A coat that is already
    // leaving pulls the slurry with it on this step.
    let from_spin = now * (1.0 - (-3.2 * spin * dt).exp());
    let from_coat = now * (coat * 12.0).clamp(0.0, 1.0);
    let out = from_spin.max(from_coat).min(now);
    *held = now - out;
    out
}

/// Mud the tread is holding. `groove` is packed in the voids. `smear` is the
/// film on the lug faces, and that film is what changes grip. `sidewall` is
/// the coat on the carcass, 0..1 of [`SIDEWALL_VOLUME`].
#[derive(Clone, Copy, Debug)]
pub struct TreadCoat {
    pub groove: f32,
    pub smear: f32,
    pub sidewall: f32,
}

impl TreadCoat {
    pub fn clean() -> Self {
        Self {
            groove: 0.0,
            smear: 0.0,
            sidewall: 0.0,
        }
    }

    pub fn sidewall_volume(self) -> f32 {
        self.sidewall.clamp(0.0, 1.0) * SIDEWALL_VOLUME
    }
}

/// What [`step_tread`] moved this step. Fractions of the tread, not cubic metres.
#[derive(Clone, Copy, Debug)]
pub struct TreadStep {
    /// Void fraction thrown off the tire. The existing clods carry this.
    pub thrown: f32,
    /// Sidewall fraction that left the carcass. Multiply by [`SIDEWALL_VOLUME`]
    /// and place that soil back on the bed.
    pub shed: f32,
    /// Sidewall fraction that stuck. Multiply by [`SIDEWALL_VOLUME`] and take
    /// that soil from the loose smear.
    pub picked: f32,
}

/// One tire's footprint and the force it can pass this step.
#[derive(Clone, Copy, Debug)]
pub struct ContactPatch {
    pub length: f32,
    pub width: f32,
    /// Lug area. Voids do not carry shear.
    pub lug_area: f32,
    /// Mean Janosi mobilization along the chord, 0 at the entry and higher at the exit.
    pub shear: f32,
    /// Friction coefficient the lug face is actually using.
    pub mu: f32,
    /// Multiplier on the soil shear. One on a clean tread at or below the peak.
    pub scale: f32,
    /// Longitudinal capacity, newtons.
    pub grip: f32,
    /// Rate loose soil is pressed into the voids, 1/s.
    pub squeeze: f32,
    /// Rate packed soil is dragged back out, 1/s.
    pub pull: f32,
    /// Fraction of the void pack leaving the tire per second. The existing
    /// clods throw this. It is not a new particle.
    pub fling: f32,
}

/// Shear along the chord. Station 0 is the entry, where displacement is still
/// zero. A rolling tire has only built shear by the exit. A spinning tire is
/// past the peak on almost the whole patch.
pub fn integrated_shear(slip: f32, length: f32, shear_k: f32) -> f32 {
    let i = slip.abs();
    let length = length.max(1.0e-4);
    let k = shear_k.max(1.0e-4);
    if i < 1.0e-5 {
        return 0.0;
    }
    const N: usize = 8;
    let mut acc = 0.0;
    for s in 0..N {
        let x = length * (s as f32 + 0.5) / N as f32;
        let j = i * x;
        acc += 1.0 - (-j / k).exp();
    }
    acc / N as f32
}

/// Residual after the peak. A clean spin still holds a share of the peak.
/// Mud on the lug face drops the slide further.
pub fn spin_residual(slip: f32, smear: f32) -> f32 {
    spin_residual_at(slip, smear, PEAK_SLIP)
}

pub fn spin_residual_at(slip: f32, smear: f32, peak_slip: f32) -> f32 {
    let peak = peak_slip.max(0.08);
    let x = slip.abs();
    if x <= peak {
        return 1.0;
    }
    let t = ((x - peak) / 0.70).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    let floor = lerp(0.64, 0.30, smear.clamp(0.0, 1.0));
    lerp(1.0, floor, s)
}

/// Rises to 1 at the peak slip, then falls with the residual. This is the
/// shape of the patch force, separate from the soil's own friction.
pub fn traction_factor(slip: f32, smear: f32) -> f32 {
    traction_factor_at(slip, smear, PEAK_SLIP)
}

pub fn traction_factor_at(slip: f32, smear: f32, peak_slip: f32) -> f32 {
    let peak = peak_slip.max(0.08);
    let x = slip.abs();
    let rise = 1.0 - (-(x / peak).min(8.0) * 2.4).exp();
    let peak_h = 1.0 - (-2.4_f32).exp();
    (rise / peak_h) * spin_residual_at(x, smear, peak)
}

/// Bare soil uses `phi`. A smeared lug shears mud against mud.
pub fn face_mu(phi: f32, smear: f32) -> f32 {
    let clean = phi.tan().max(0.02);
    let coated = (clean * 0.58).max(0.05);
    lerp(clean, coated, smear.clamp(0.0, 1.0))
}

/// Reference load and slip for moisture traction readouts.
pub const MOISTURE_TRACTION_NORMAL: f32 = 3_000.0;
pub const MOISTURE_TRACTION_SLIP: f32 = 0.15;
pub const MOISTURE_TRACTION_SINK: f32 = 0.05;

#[derive(Clone, Copy, Debug)]
pub struct MoistureTractionRow {
    pub mu: f32,
    pub peak_slip: f32,
    pub grip: f32,
    pub wetness: f32,
    pub water: f32,
}

/// Grip, friction, and peak slip for one soil kind and moisture band.
pub fn moisture_traction_row(
    kind: SurfaceKind,
    state: SurfaceMoisture,
) -> MoistureTractionRow {
    let (wetness, loose, water) = moisture_state_fields(kind, state);
    let patch = contact_patch(
        0.38,
        0.28,
        MOISTURE_TRACTION_SINK,
        MOISTURE_TRACTION_SLIP,
        MOISTURE_TRACTION_NORMAL,
        kind,
        wetness,
        loose,
        water,
        TreadCoat::clean(),
    );
    MoistureTractionRow {
        mu: patch.mu,
        peak_slip: moisture_peak_slip(kind, state),
        grip: patch.grip,
        wetness,
        water,
    }
}

/// Reference normal per tire for sink and contact readouts (Niva static pose).
pub const MOISTURE_SINKAGE_NORMAL: f32 = MOISTURE_TRACTION_NORMAL;

#[derive(Clone, Copy, Debug)]
pub struct MoistureSinkageRow {
    pub sink_m: f32,
    pub contact_m2: f32,
    pub wetness: f32,
    pub water_m: f32,
    pub buoy_n: f32,
    pub suction_n: f32,
}

/// Equilibrium Bekker sink under `normal`, minus buoyancy from standing water.
pub fn sinkage_at_normal(
    normal: f32,
    kind: SurfaceKind,
    wetness: f32,
    loose: f32,
    water: f32,
    radius: f32,
    width: f32,
) -> (f32, f32, f32) {
    let radius = radius.max(0.05);
    let width = width.max(0.02);
    let buoy = wheel_buoyancy(water, radius, width);
    let soil_load = (normal - buoy).max(0.0);
    let sink = if soil_load < 1.0 {
        0.0
    } else {
        sinkage_on(soil_load, kind, wetness, loose, water, radius, width)
    };
    let length = contact_length(radius, sink.max(1.0e-4));
    let area = (length * width).max(0.0);
    (sink, area, buoy)
}

/// Breakaway multiplier after rocking compacts the rut. Clay keeps most of its
/// hold; sand gives way sooner once the loose spoil is packed firm.
pub fn stuck_compact_breakaway(kind: SurfaceKind, compact: f32) -> f32 {
    let compact = compact.clamp(0.0, 1.0);
    let release = match kind {
        SurfaceKind::WetMud => 0.14,
        SurfaceKind::Silt => 0.26,
        SurfaceKind::Sand => 0.42,
        _ => 0.20,
    };
    1.0 - release * compact
}

/// Relative rate rocking compacts loose spoil back into the rut floor.
pub fn stuck_rock_compact_pace(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::WetMud => 1.05,
        SurfaceKind::Silt => 1.35,
        SurfaceKind::Sand => 1.85,
        _ => 1.2,
    }
}

/// Reference hold at saturation: clay hardest, sand lets go soonest.
pub fn stuck_saturation_hold(kind: SurfaceKind) -> f32 {
    match kind {
        SurfaceKind::WetMud => 1.0,
        SurfaceKind::Silt => 0.78,
        SurfaceKind::Sand => 0.55,
        _ => 0.70,
    }
}

/// Sink depth, contact patch, moisture, and seal suction for one soil band.
pub fn moisture_sinkage_row(kind: SurfaceKind, state: SurfaceMoisture) -> MoistureSinkageRow {
    let (wetness, loose, water) = moisture_state_fields(kind, state);
    let (sink_m, contact_m2, buoy_n) = sinkage_at_normal(
        MOISTURE_SINKAGE_NORMAL,
        kind,
        wetness,
        loose,
        water,
        0.38,
        0.28,
    );
    let moisture = surface_moisture(wetness, water, kind);
    let suction_n = if matches!(moisture, SurfaceMoisture::Pooled | SurfaceMoisture::Saturated) {
        crate::soil::slurry_suction_force(
            kind,
            wetness.clamp(0.08, 0.92),
            sink_m.max(0.02),
            0.38,
            0.28,
            0.04,
            0.0,
        )
    } else {
        0.0
    };
    MoistureSinkageRow {
        sink_m,
        contact_m2,
        wetness,
        water_m: water,
        buoy_n,
        suction_n,
    }
}

pub fn contact_patch(
    radius: f32,
    width: f32,
    sinkage: f32,
    slip: f32,
    normal: f32,
    kind: SurfaceKind,
    wetness: f32,
    loose: f32,
    water: f32,
    coat: TreadCoat,
) -> ContactPatch {
    let width = width.max(0.01);
    let sinkage = sinkage.max(0.0);
    let length = contact_length(radius.max(0.05), sinkage).max(radius.max(0.05) * 0.2);
    let full = length * width;
    let lug_area = full * LUG_FRAC;
    if normal < 1.0 || sinkage < 1.0e-4 {
        return ContactPatch {
            length,
            width,
            lug_area,
            shear: 0.0,
            mu: 0.0,
            scale: 1.0,
            grip: 0.0,
            squeeze: 0.0,
            pull: 0.0,
            fling: 0.0,
        };
    }
    let soft = softness_of(wetness, loose, water);
    let tm = terramech_for(kind, soft, water);
    let moisture = surface_moisture(wetness, water, kind);
    let peak_slip = moisture_peak_slip(kind, moisture);
    let m_scale = moisture_mu_scale(kind, moisture);
    let sigma = (normal / full).clamp(0.0, 80_000.0);
    let shear = integrated_shear(slip, length, tm.shear_k);
    let clean_mu = (tm.phi.tan().max(0.02)) * m_scale;
    let mu = face_mu((clean_mu.max(0.02)).atan(), coat.smear);
    let mu_ratio = (mu / clean_mu.max(1.0e-4)).clamp(0.2, 1.25);
    // One on a clean tread at or below the peak. The Janosi term already
    // builds with slip, so this only cuts the force once the tire is spinning
    // or the lugs are coated.
    let scale = (spin_residual_at(slip, coat.smear, peak_slip) * mu_ratio).clamp(0.20, 1.15);
    let tau = tm.cohesion + sigma * clean_mu;
    let grip = tau * lug_area * traction_factor_at(slip, coat.smear, peak_slip) * mu_ratio;

    let stick = stick_scale(kind, wetness);
    let loose_n = (loose / 0.06).clamp(0.0, 1.0);
    let wet = wetness.clamp(0.0, 1.0);
    let supply = stick
        * (0.30 + 0.70 * loose_n)
        * (0.35 + 0.65 * wet + 0.25 * (water / 0.06).min(1.0));
    let pressure = (normal / 1_800.0).clamp(0.25, 3.0);
    let slip_n = slip.abs();
    // Rolling presses the scoop into the voids and leaves it there.
    // Spinning still scoops, then the tread drags that mud straight back out.
    let rolling = (1.0 - (slip_n / 0.34).min(1.0)).powi(2);
    // Pull starts once the tread is past a roll. A slip of ~0.2 still packs.
    let spinning = ((slip_n - 0.28) / 0.40).clamp(0.0, 1.0);
    let squeeze = 2.8 * supply * pressure * (0.50 + 1.35 * rolling + 0.85 * spinning);
    let pull = 12.0 * spinning * (0.35 + 0.65 * supply.max(0.2));
    let thrown_share = lerp(0.20, 0.86, spinning);
    let fling = pull * coat.groove.clamp(0.0, 1.0) * thrown_share;

    ContactPatch {
        length,
        width,
        lug_area,
        shear,
        mu,
        scale,
        grip,
        squeeze,
        pull,
        fling,
    }
}

/// Move mud between the voids, the lug faces, and the sidewall.
///
/// Rolling squeezes loose soil into the voids and wipes the lug clean, and
/// a share of that clay climbs the carcass. Spinning drags the pack out of
/// the voids and throws the sidewall coat off. The shed fraction is volume
/// the caller places back on the rut.
pub fn step_tread(coat: &mut TreadCoat, slip: f32, squeeze: f32, pull: f32, dt: f32) -> TreadStep {
    let dt = dt.max(0.0);
    let slip_n = slip.abs();
    // A gentle roll still wipes the lug. The wipe dies once the tread is
    // past the shear peak and the void is being dragged back out.
    let rolling = (1.0 - (slip_n / 0.45).min(1.0)).max(0.0);
    let spinning = ((slip_n - 0.35) / 0.55).clamp(0.0, 1.0);
    let into = squeeze * (1.0 - coat.groove).clamp(0.0, 1.0) * dt;
    let out = (pull * coat.groove.clamp(0.0, 1.0) * dt).min(coat.groove.max(0.0));
    coat.groove = (coat.groove + into - out).clamp(0.0, 1.0);
    // Rolling keeps the pack in the void. Only a real spin paints the lug.
    let to_face = out * lerp(0.05, 0.62, spinning);
    let thrown = (out - to_face).max(0.0);
    let wipe = (2.8 * rolling * coat.smear * dt).min(coat.smear.max(0.0));
    coat.smear = (coat.smear + to_face - wipe).clamp(0.0, 1.0);
    coat.groove = (coat.groove + wipe * 0.85).min(1.0);

    let before = coat.sidewall.max(0.0);
    let room = (1.0 - before).max(0.0);
    // The packed void wets the carcass while the tire is rolling. A spin
    // sheds that coat. The two do not run together, so a roll can hold clay
    // in the grooves and on the sidewall at the same time.
    let picked = (squeeze
        * 0.48
        * (0.30 + 0.70 * coat.groove.clamp(0.0, 1.0))
        * rolling
        * room
        * dt)
        .min(room);
    let shedding = (7.5 * spinning * (before + picked) * dt).min(before + picked);
    coat.sidewall = (before + picked - shedding).clamp(0.0, 1.0);
    TreadStep {
        thrown,
        shed: shedding,
        picked,
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::soil::SoilField;

    fn wet_patch(slip: f32, coat: TreadCoat) -> ContactPatch {
        contact_patch(
            0.38,
            0.28,
            0.06,
            slip,
            3_200.0,
            SurfaceKind::WetMud,
            0.78,
            0.04,
            0.0,
            coat,
        )
    }

    #[test]
    fn patch_grows_with_sinkage() {
        let shallow = contact_patch(
            0.38,
            0.28,
            0.02,
            0.12,
            3_200.0,
            SurfaceKind::WetMud,
            0.7,
            0.03,
            0.0,
            TreadCoat::clean(),
        );
        let deep = contact_patch(
            0.38,
            0.28,
            0.10,
            0.12,
            3_200.0,
            SurfaceKind::WetMud,
            0.7,
            0.03,
            0.0,
            TreadCoat::clean(),
        );
        assert!(shallow.length > 0.05, "no patch, length {}", shallow.length);
        assert!(deep.length > shallow.length * 1.6, "{} vs {}", deep.length, shallow.length);
        assert!((shallow.width - 0.28).abs() < 1.0e-4);
        assert!(shallow.lug_area < shallow.length * shallow.width);
        assert!(deep.shear > 0.0);
    }

    #[test]
    fn mud_on_the_lug_lowers_grip_at_the_same_slip() {
        let clean = wet_patch(
            0.40,
            TreadCoat {
                groove: 0.4,
                smear: 0.0,
                sidewall: 0.0,
            },
        );
        let caked = wet_patch(
            0.40,
            TreadCoat {
                groove: 0.4,
                smear: 0.85,
                sidewall: 0.0,
            },
        );
        assert!(caked.mu < clean.mu * 0.8, "mu {} -> {}", clean.mu, caked.mu);
        assert!(
            caked.grip < clean.grip * 0.75,
            "coat should cut grip {} -> {}",
            clean.grip,
            caked.grip
        );
        assert!(caked.scale < clean.scale);
    }

    #[test]
    fn rolling_packs_grooves_and_spinning_throws() {
        let mut roll = TreadCoat::clean();
        let mut spin = TreadCoat::clean();
        let mut roll_thrown = 0.0;
        let mut spin_thrown = 0.0;
        let dt = 1.0 / 60.0;
        for _ in 0..(60 * 2) {
            let r = wet_patch(0.08, roll);
            let s = wet_patch(1.05, spin);
            roll_thrown += step_tread(&mut roll, 0.08, r.squeeze, r.pull, dt).thrown;
            spin_thrown += step_tread(&mut spin, 1.05, s.squeeze, s.pull, dt).thrown;
        }
        let roll_grip = wet_patch(0.08, roll).grip;
        let spin_grip = wet_patch(1.05, spin).grip;
        eprintln!(
            "roll groove {:.2} smear {:.2} grip {:.0} N  thrown {:.3}",
            roll.groove, roll.smear, roll_grip, roll_thrown
        );
        eprintln!(
            "spin groove {:.2} smear {:.2} grip {:.0} N  thrown {:.3}",
            spin.groove, spin.smear, spin_grip, spin_thrown
        );
        assert!(roll.groove > 0.55, "rolling should pack the voids, {}", roll.groove);
        assert!(
            spin.groove < roll.groove * 0.45,
            "spinning should empty the voids, roll {} spin {}",
            roll.groove,
            spin.groove
        );
        assert!(
            spin.smear > roll.smear + 0.15,
            "spinning should coat the lugs, roll {} spin {}",
            roll.smear,
            spin.smear
        );
        assert!(
            spin_thrown > roll_thrown * 3.0,
            "spinning should throw more, roll {roll_thrown} spin {spin_thrown}"
        );
        assert!(
            spin_grip < roll_grip * 0.72,
            "spinning should grip less, roll {roll_grip} spin {spin_grip}"
        );
    }

    #[test]
    fn patch_keeps_the_wet_to_dry_grip() {
        let coat = TreadCoat::clean();
        let wet = contact_patch(
            0.38,
            0.28,
            0.05,
            0.15,
            3_000.0,
            SurfaceKind::WetMud,
            0.82,
            0.02,
            0.0,
            coat,
        );
        let dry = contact_patch(
            0.38,
            0.28,
            0.05,
            0.15,
            3_000.0,
            SurfaceKind::DryCrust,
            0.12,
            0.0,
            0.0,
            coat,
        );
        eprintln!("patch wet mu {:.3} grip {:.0}   dry mu {:.3} grip {:.0}", wet.mu, wet.grip, dry.mu, dry.grip);
        assert!(
            dry.mu > wet.mu * 1.25,
            "a crust should grip more than wet clay, {} -> {}",
            wet.mu,
            dry.mu
        );
        assert!(dry.grip > wet.grip, "dry {} wet {}", dry.grip, wet.grip);
    }

    #[test]
    fn packing_the_tread_does_not_move_the_cut() {
        let mut soil = SoilField::flat(0.7);
        let before_cut = soil.cut_volume();
        let before_placed = soil.placed_volume();
        soil.stamp_wheel(
            glam::Vec3::new(0.2, 0.30, -0.4),
            0.38,
            0.28,
            glam::Vec3::Z,
            glam::Vec3::X,
            0.08,
            0.1,
            0.12,
            true,
            0.0,
            0.0,
        );
        let cut = soil.cut_volume() - before_cut;
        let placed = soil.placed_volume() - before_placed;
        assert!(cut > 1.0e-4, "stamp did not cut");
        assert!(
            (cut - placed).abs() < cut * 0.02 + 1.0e-5,
            "volume drifted cut {cut} placed {placed}"
        );
        let loose_before: f32 = soil.loose.iter().sum();
        let took = soil.skim_loose(
            glam::Vec3::new(0.2, 0.30, -0.4),
            glam::Vec3::Z,
            glam::Vec3::X,
            0.45,
            0.28,
            0.01,
        );
        assert!(took > 0.0, "the patch skimmed nothing");
        assert!(
            (soil.cut_volume() - before_cut - cut).abs() < 1.0e-5,
            "skimming moved the cut"
        );
        assert!(
            (soil.placed_volume() - before_placed - placed).abs() < 1.0e-5,
            "skimming moved the berm"
        );
        let loose_after: f32 = soil.loose.iter().sum();
        assert!(
            loose_after < loose_before,
            "loose smear should have entered the tread, {loose_before} -> {loose_after}"
        );
    }

    #[test]
    fn roll_fills_grooves_and_sidewall_shed_returns_to_the_rut() {
        let mut roll = TreadCoat::clean();
        let mut shed = 0.0_f32;
        let mut picked = 0.0_f32;
        let dt = 1.0 / 60.0;
        for _ in 0..(60 * 2) {
            let patch = wet_patch(0.08, roll);
            let step = step_tread(&mut roll, 0.08, patch.squeeze, patch.pull, dt);
            picked += step.picked;
            shed += step.shed;
        }
        let roll_grip = wet_patch(0.08, roll).grip;
        assert!(
            roll.groove > 0.95,
            "rolling should pack the voids, groove {}",
            roll.groove
        );
        assert!(
            roll.sidewall > 0.90,
            "rolling should build a sidewall coat, {}",
            roll.sidewall
        );
        assert!(
            shed < 0.08,
            "a roll should keep the coat, shed fraction {shed}"
        );
        let held = roll.sidewall;
        // The spin starts from that coated tire, which is the picture:
        // grooves full and a carcass coat, then the tread empties and the
        // coat falls off.
        let mut spin = roll;
        for _ in 0..(60 * 2) {
            let patch = wet_patch(1.05, spin);
            let step = step_tread(&mut spin, 1.05, patch.squeeze, patch.pull, dt);
            picked += step.picked;
            shed += step.shed;
        }
        let spin_grip = wet_patch(1.05, spin).grip;
        eprintln!(
            "roll groove {:.3} sidewall {:.3} grip {:.0} N",
            roll.groove, held, roll_grip
        );
        eprintln!(
            "spin groove {:.3} sidewall {:.3} smear {:.3} grip {:.0} N  shed {:.3}",
            spin.groove, spin.sidewall, spin.smear, spin_grip, shed
        );
        assert!(
            spin.groove < 0.40 && spin.groove < roll.groove * 0.45,
            "spinning should empty the voids, roll {} spin {}",
            roll.groove,
            spin.groove
        );
        assert!(
            spin_grip < roll_grip * 0.45,
            "spinning should grip less, roll {roll_grip} spin {spin_grip}"
        );
        assert!(
            spin.sidewall < held * 0.25,
            "spinning should shed the sidewall, {held} -> {}",
            spin.sidewall
        );
        assert!(
            (picked - shed - spin.sidewall).abs() < 1.0e-3,
            "coat fraction drifted picked {picked} shed {shed} held {}",
            spin.sidewall
        );
        let shed_vol = shed * SIDEWALL_VOLUME;
        assert!(shed_vol > 1.0e-4, "shed volume {shed_vol} m3");

        let mut soil = SoilField::mud_pad();
        soil.roll_lane(
            glam::Vec3::new(0.0, 0.0, -1.2),
            glam::Vec3::new(0.0, 0.0, 1.2),
            0.09,
            0.0,
        );
        for loose in &mut soil.loose {
            *loose = loose.max(0.08);
        }
        let cut = soil.cut_volume();
        let placed = soil.placed_volume();
        let loose_before = loose_volume(&soil);
        let got = soil.take_loose(
            glam::Vec3::new(0.0, 0.3, 0.0),
            glam::Vec3::Z,
            glam::Vec3::X,
            1.6,
            0.8,
            picked * SIDEWALL_VOLUME,
        );
        let land = soil.return_coat(
            glam::Vec3::new(0.0, 0.3, 0.6),
            glam::Vec3::Z,
            glam::Vec3::X,
            shed_vol,
        );
        let loose_after = loose_volume(&soil);
        eprintln!(
            "sidewall shed {shed_vol:.5} m3  took {got:.5}  landed {:.5} in rut {:.5}  \
             cut {:.5} placed {:.5} -> {:.5}",
            land.volume,
            land.in_rut,
            cut,
            placed,
            soil.placed_volume()
        );
        assert!(
            (got - picked * SIDEWALL_VOLUME).abs() < 1.0e-5,
            "the coat was not skimmed from loose, want {} got {got}",
            picked * SIDEWALL_VOLUME
        );
        assert!(
            (loose_before - loose_after - got).abs() < 1.0e-4,
            "loose volume drifted"
        );
        assert!(
            (soil.cut_volume() - cut).abs() < 1.0e-5,
            "returning the coat cut the bed, {cut} -> {}",
            soil.cut_volume()
        );
        assert!(
            (land.volume - shed_vol).abs() < 1.0e-5,
            "shed volume was deleted, want {shed_vol} landed {}",
            land.volume
        );
        assert!(
            land.in_rut > shed_vol * 0.99 && (land.in_rut - land.volume).abs() < 1.0e-5,
            "shed volume missed the rut, in rut {} of {shed_vol}",
            land.in_rut
        );
        assert!(
            (soil.placed_volume() - placed - land.volume).abs() < 1.0e-4,
            "placed spoil did not take the coat, delta {} want {}",
            soil.placed_volume() - placed,
            land.volume
        );
    }

    fn loose_volume(soil: &SoilField) -> f32 {
        let area = soil.cell * soil.cell;
        soil.loose.iter().sum::<f32>() * area
    }
}
