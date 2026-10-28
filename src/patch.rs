//! The tire meets the soil on a patch, not a point.
//!
//! The patch is the chord of the sunk wheel. Lug faces carry the shear.
//! The voids between them take mud when the wheel rolls, and a spinning
//! tread pulls that mud back out. Mud left on the lug face is a slicker
//! interface than the bare soil, so the same ground grips less once the
//! tread is coated. The cut/placed budget is not this mass: packed mud is
//! skimmed from the loose smear.

use crate::soil::{
    contact_length, softness_of, stick_scale, terramech_for, SurfaceKind,
};

/// Share of the patch that is lug. The rest is void the tread can pack.
pub const LUG_FRAC: f32 = 0.62;
/// How deep a void is, metres. Packed mud occupies this, it does not dig the bed.
pub const GROOVE_DEPTH: f32 = 0.016;
/// Slip where the patch is at peak shear. Past this the tread is spinning.
pub const PEAK_SLIP: f32 = 0.18;

/// Mud the tread is holding. `groove` is packed in the voids. `smear` is the
/// film on the lug faces, and that film is what changes grip.
#[derive(Clone, Copy, Debug)]
pub struct TreadCoat {
    pub groove: f32,
    pub smear: f32,
}

impl TreadCoat {
    pub fn clean() -> Self {
        Self {
            groove: 0.0,
            smear: 0.0,
        }
    }
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
    let x = slip.abs();
    if x <= PEAK_SLIP {
        return 1.0;
    }
    let t = ((x - PEAK_SLIP) / 0.70).clamp(0.0, 1.0);
    let s = t * t * (3.0 - 2.0 * t);
    let floor = lerp(0.64, 0.30, smear.clamp(0.0, 1.0));
    lerp(1.0, floor, s)
}

/// Rises to 1 at the peak slip, then falls with the residual. This is the
/// shape of the patch force, separate from the soil's own friction.
pub fn traction_factor(slip: f32, smear: f32) -> f32 {
    let x = slip.abs();
    let rise = 1.0 - (-(x / PEAK_SLIP).min(8.0) * 2.4).exp();
    let peak = 1.0 - (-2.4_f32).exp();
    (rise / peak) * spin_residual(x, smear)
}

/// Bare soil uses `phi`. A smeared lug shears mud against mud.
pub fn face_mu(phi: f32, smear: f32) -> f32 {
    let clean = phi.tan().max(0.02);
    let coated = (clean * 0.58).max(0.05);
    lerp(clean, coated, smear.clamp(0.0, 1.0))
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
    let sigma = (normal / full).clamp(0.0, 80_000.0);
    let shear = integrated_shear(slip, length, tm.shear_k);
    let clean_mu = tm.phi.tan().max(0.02);
    let mu = face_mu(tm.phi, coat.smear);
    let mu_ratio = (mu / clean_mu).clamp(0.2, 1.25);
    // One on a clean tread at or below the peak. The Janosi term already
    // builds with slip, so this only cuts the force once the tire is spinning
    // or the lugs are coated.
    let scale = (spin_residual(slip, coat.smear) * mu_ratio).clamp(0.20, 1.15);
    let tau = tm.cohesion + sigma * clean_mu;
    let grip = tau * lug_area * traction_factor(slip, coat.smear) * mu_ratio;

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

/// Move mud between the voids and the lug faces.
///
/// Returns the void fraction thrown off the tire this step. Rolling squeezes
/// loose soil in and wipes the face clean. Spinning drags the pack out.
pub fn step_tread(coat: &mut TreadCoat, slip: f32, squeeze: f32, pull: f32, dt: f32) -> f32 {
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
    thrown
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
        let clean = wet_patch(0.40, TreadCoat { groove: 0.4, smear: 0.0 });
        let caked = wet_patch(
            0.40,
            TreadCoat {
                groove: 0.4,
                smear: 0.85,
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
            roll_thrown += step_tread(&mut roll, 0.08, r.squeeze, r.pull, dt);
            spin_thrown += step_tread(&mut spin, 1.05, s.squeeze, s.pull, dt);
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
}
