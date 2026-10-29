//! Dust lofted by a rolling tire on dry ground.
//!
//! One tire throws fines at
//!
//! ```text
//! m_dot = k * dryness * max(speed - u0, 0) ^ 1.5
//! ```
//!
//! kilograms per second. `dryness` is 1 on a bone-dry pad and 0 once the
//! moisture reaches [`WET_MOISTURE`] or standing water reaches [`WET_WATER_M`].
//! The near-wake density is that mass mixed into the air the tire sweeps,
//!
//! ```text
//! rho = m_dot / (speed * wake_section)
//! ```
//!
//! so it rises with dryness and, above the loft threshold, with speed.
//! Parcels are born in the wake, just aft of the contact, and drag relaxes
//! their velocity onto the wind. Mass that falls back to the pad leaves the
//! airborne cloud. Wet ground adds none.

use glam::Vec3;

/// Moisture, 0 dry .. 1 saturated, where loft stops.
pub const WET_MOISTURE: f32 = 0.55;
/// Standing water that stops loft even on a dry reading, metres.
pub const WET_WATER_M: f32 = 0.008;
/// Ground speed below which the tread does not throw fines, metres per second.
pub const LOFT_SPEED_M_S: f32 = 0.5;
/// `k` in the emission law, kg/s per (m/s)^1.5.
pub const LOFT_K: f32 = 5.5e-4;
/// Tire width that sets the wake cross-section, metres.
pub const WAKE_WIDTH_M: f32 = 0.28;
/// Height of the mixed near wake, metres.
pub const WAKE_HEIGHT_M: f32 = 0.55;
/// Wake cross-section, square metres.
pub const WAKE_SECTION_M2: f32 = WAKE_WIDTH_M * WAKE_HEIGHT_M;

/// Solver tire radius. Matches the Niva contact used by the rest of the engine.
pub const WHEEL_RADIUS_M: f32 = 0.38;
/// Hub offsets in chassis space, FL FR RL RR. +Z is forward, +X is right.
pub const HUBS: [[f32; 3]; 4] = [
    [-0.74, -0.20, 1.08],
    [0.74, -0.20, 1.08],
    [-0.74, -0.20, -1.18],
    [0.74, -0.20, -1.18],
];
/// Chassis origin height that puts the contacts on y = 0.
pub const CHASSIS_Y: f32 = 0.20 + WHEEL_RADIUS_M;

const PUFFS_PER_TIRE: usize = 3;
const PUFF_R0_M: f32 = 0.30;
const PUFF_SPREAD: f32 = 0.95;
const PUFF_R_MAX_M: f32 = 1.55;
const LIFT_M_S: f32 = 0.85;
const SETTLE_M_S: f32 = 0.38;
const DRAG_TAU_S: f32 = 1.15;
const GROUND_Y: f32 = 0.02;

/// Proof drive. Dry pad, a crosswind toward -X and a little aft.
pub const PROOF_SPEED_M_S: f32 = 8.0;
pub const PROOF_MOISTURE: f32 = 0.04;
pub const PROOF_WATER_M: f32 = 0.0;
pub const PROOF_SECONDS: f32 = 4.0;
pub const PROOF_DT_S: f32 = 1.0 / 60.0;

pub fn proof_wind() -> Vec3 {
    // Crosswind toward +X and a little aft. From the left-side proof camera
    // the cloud shears into the scene and stays behind the tires.
    Vec3::new(1.5, 0.0, -0.9)
}

/// 1 on dry ground, 0 at the wet cutoff and anywhere water is standing.
pub fn dryness(moisture: f32, water_m: f32) -> f32 {
    if water_m >= WET_WATER_M || moisture >= WET_MOISTURE {
        return 0.0;
    }
    let m = moisture.max(0.0);
    ((WET_MOISTURE - m) / WET_MOISTURE).clamp(0.0, 1.0)
}

/// Kilograms per second lofted by one tire.
pub fn loft_rate_kg_s(moisture: f32, water_m: f32, speed: f32) -> f32 {
    let dry = dryness(moisture, water_m);
    let excess = (speed - LOFT_SPEED_M_S).max(0.0);
    if dry == 0.0 || excess == 0.0 {
        0.0
    } else {
        LOFT_K * dry * excess * excess.sqrt()
    }
}

/// Near-wake mass density, kilograms per cubic metre.
///
/// The tire sweeps `speed * wake_section` of air each second and mixes
/// `loft_rate` into it. Stopped and wet tires are zero.
pub fn wake_density_kg_m3(moisture: f32, water_m: f32, speed: f32) -> f32 {
    let speed = speed.max(0.0);
    if speed <= LOFT_SPEED_M_S {
        return 0.0;
    }
    loft_rate_kg_s(moisture, water_m, speed) / (speed * WAKE_SECTION_M2)
}

#[derive(Clone, Copy, Debug)]
pub struct Tire {
    pub index: u8,
    pub contact: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub speed: f32,
}

/// Straight drive on y = 0, nose along +Z, chassis x = 0.
pub fn tires_at(chassis_z: f32, speed: f32) -> [Tire; 4] {
    let forward = Vec3::new(0.0, 0.0, 1.0);
    let right = Vec3::new(1.0, 0.0, 0.0);
    std::array::from_fn(|i| {
        let h = HUBS[i];
        let contact = Vec3::new(h[0], 0.0, chassis_z + h[2]);
        Tire {
            index: i as u8,
            contact,
            forward,
            right,
            speed,
        }
    })
}

pub fn chassis_at(chassis_z: f32) -> Vec3 {
    Vec3::new(0.0, CHASSIS_Y, chassis_z)
}

#[derive(Clone, Copy, Debug)]
pub struct Parcel {
    pub pos: Vec3,
    pub vel: Vec3,
    pub mass_kg: f32,
    pub radius_m: f32,
    pub age_s: f32,
    pub wheel: u8,
}

#[derive(Clone, Debug)]
pub struct DustTrail {
    pub parcels: Vec<Parcel>,
    pub emitted_kg: f32,
    pub settled_kg: f32,
    pub wind: Vec3,
    rng: u32,
}

impl DustTrail {
    pub fn new(wind: Vec3) -> Self {
        Self {
            parcels: Vec::new(),
            emitted_kg: 0.0,
            settled_kg: 0.0,
            wind,
            rng: 0xD057_u32,
        }
    }

    pub fn airborne_kg(&self) -> f32 {
        self.parcels.iter().map(|p| p.mass_kg).sum()
    }

    pub fn centroid(&self) -> Option<Vec3> {
        let mut mass = 0.0;
        let mut acc = Vec3::ZERO;
        for p in &self.parcels {
            acc += p.pos * p.mass_kg;
            mass += p.mass_kg;
        }
        if mass <= 1.0e-8 {
            None
        } else {
            Some(acc / mass)
        }
    }

    /// Mass-weighted centroid offset from the rear-axle midpoint.
    /// `behind` is positive when the cloud sits aft of the rear contacts.
    /// `lateral` is the offset along +X.
    pub fn rear_offset(&self, chassis_z: f32) -> Option<(f32, f32)> {
        let c = self.centroid()?;
        let rear_z = chassis_z + HUBS[2][2];
        Some((rear_z - c.z, c.x))
    }

    /// Mean density of parcel mass whose centre sits in the 6 m wake box
    /// behind the rear axle, kilograms per cubic metre.
    pub fn near_wake_bulk_kg_m3(&self, chassis_z: f32) -> f32 {
        let rear = chassis_z + HUBS[2][2];
        let z0 = rear - 6.0;
        let z1 = rear - 0.25;
        let vol = 6.0 * 3.2 * 2.0;
        let mut mass = 0.0;
        for p in &self.parcels {
            if (-1.6..=1.6).contains(&p.pos.x)
                && (0.0..=2.0).contains(&p.pos.y)
                && (z0..=z1).contains(&p.pos.z)
            {
                mass += p.mass_kg;
            }
        }
        mass / vol
    }

    pub fn step(&mut self, tires: &[Tire], moisture: f32, water_m: f32, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        for tire in tires {
            self.spawn(tire, moisture, water_m, dt);
        }
        self.advect(dt);
    }

    /// One step where each tire can sit on its own pad.
    pub fn step_pads(&mut self, tires: &[Tire], moisture: &[f32], water_m: &[f32], dt: f32) {
        if dt <= 0.0 {
            return;
        }
        for (i, tire) in tires.iter().enumerate() {
            let m = moisture.get(i).copied().unwrap_or(0.0);
            let w = water_m.get(i).copied().unwrap_or(0.0);
            self.spawn(tire, m, w, dt);
        }
        self.advect(dt);
    }

    fn spawn(&mut self, tire: &Tire, moisture: f32, water_m: f32, dt: f32) {
        let rate = loft_rate_kg_s(moisture, water_m, tire.speed);
        if rate <= 0.0 {
            return;
        }
        let each = rate * dt / PUFFS_PER_TIRE as f32;
        for _ in 0..PUFFS_PER_TIRE {
            let along = 0.28 + 0.45 * self.rnd();
            let side = (self.rnd() - 0.5) * WAKE_WIDTH_M * 1.4;
            // Low grit plus a billow that stands up behind the tread.
            let up = if self.rnd() < 0.25 {
                0.08 + 0.20 * self.rnd()
            } else {
                0.32 + 0.88 * self.rnd()
            };
            let pos = tire.contact - tire.forward * along + tire.right * side + Vec3::Y * up;
            let lift = LIFT_M_S * (0.55 + 0.9 * self.rnd());
            let aft = 0.25 + 0.35 * self.rnd();
            let vel = -tire.forward * aft + tire.right * (self.rnd() - 0.5) * 0.55 + Vec3::Y * lift;
            self.parcels.push(Parcel {
                pos,
                vel,
                mass_kg: each,
                radius_m: PUFF_R0_M,
                age_s: 0.0,
                wheel: tire.index,
            });
            self.emitted_kg += each;
        }
    }

    fn advect(&mut self, dt: f32) {
        let target = Vec3::new(self.wind.x, -SETTLE_M_S, self.wind.z);
        let blend = 1.0 - (-dt / DRAG_TAU_S).exp();
        let mut settled = 0.0;
        for p in &mut self.parcels {
            p.vel += (target - p.vel) * blend;
            p.pos += p.vel * dt;
            p.age_s += dt;
            let grown = PUFF_R0_M + PUFF_SPREAD * p.age_s.sqrt();
            p.radius_m = grown.min(PUFF_R_MAX_M);
            if p.pos.y <= GROUND_Y && p.vel.y <= 0.0 && p.age_s > 0.35 {
                settled += p.mass_kg;
                p.mass_kg = 0.0;
            }
        }
        if settled > 0.0 {
            self.settled_kg += settled;
            self.parcels.retain(|p| p.mass_kg > 0.0);
        }
    }

    fn rnd(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }
}

/// Drive four tires in a straight line and return the cloud plus the final chassis z.
pub fn drive(
    speed: f32,
    moisture: f32,
    water_m: f32,
    wind: Vec3,
    seconds: f32,
    dt: f32,
) -> (DustTrail, f32) {
    let mut dust = DustTrail::new(wind);
    let steps = ((seconds / dt).round() as i32).max(0);
    let mut z = 0.0;
    for _ in 0..steps {
        let tires = tires_at(z, speed);
        dust.step(&tires, moisture, water_m, dt);
        z += speed * dt;
    }
    (dust, z)
}

/// Straight drive where each tire can sit on its own pad.
pub fn drive_mixed(
    speed: f32,
    moisture: [f32; 4],
    water_m: [f32; 4],
    wind: Vec3,
    seconds: f32,
    dt: f32,
) -> (DustTrail, f32) {
    let mut dust = DustTrail::new(wind);
    let steps = ((seconds / dt).round() as i32).max(0);
    let mut z = 0.0;
    for _ in 0..steps {
        let tires = tires_at(z, speed);
        dust.step_pads(&tires, &moisture, &water_m, dt);
        z += speed * dt;
    }
    (dust, z)
}

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub moisture: f32,
    pub water_m: f32,
    pub speed: f32,
    pub dryness: f32,
    pub rate_kg_s: f32,
    pub density_kg_m3: f32,
}

/// Reference emission table. Wet rows are zeros.
pub fn reference_samples() -> [Sample; 8] {
    let rows = [
        (0.00, 0.0, 3.0),
        (0.00, 0.0, 8.0),
        (0.00, 0.0, 12.0),
        (0.04, 0.0, 8.0),
        (0.30, 0.0, 8.0),
        (0.48, 0.0, 8.0),
        (0.55, 0.0, 8.0),
        (0.00, 0.02, 8.0),
    ];
    rows.map(|(moisture, water_m, speed)| Sample {
        moisture,
        water_m,
        speed,
        dryness: dryness(moisture, water_m),
        rate_kg_s: loft_rate_kg_s(moisture, water_m, speed),
        density_kg_m3: wake_density_kg_m3(moisture, water_m, speed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} != {b} (±{eps})");
    }

    #[test]
    fn wet_ground_and_standing_water_and_a_stop_loft_nothing() {
        assert_eq!(dryness(0.55, 0.0), 0.0);
        assert_eq!(dryness(0.9, 0.0), 0.0);
        assert_eq!(dryness(0.0, 0.008), 0.0);
        assert_eq!(dryness(0.0, 0.05), 0.0);
        assert_eq!(loft_rate_kg_s(0.8, 0.0, 12.0), 0.0);
        assert_eq!(loft_rate_kg_s(0.0, 0.02, 12.0), 0.0);
        assert_eq!(loft_rate_kg_s(0.0, 0.0, 0.5), 0.0);
        assert_eq!(loft_rate_kg_s(0.0, 0.0, 0.0), 0.0);
        assert_eq!(wake_density_kg_m3(0.7, 0.0, 10.0), 0.0);
        assert_eq!(wake_density_kg_m3(0.0, 0.01, 10.0), 0.0);
        assert_eq!(wake_density_kg_m3(0.0, 0.0, 0.2), 0.0);

        let (wet, _) = drive(10.0, 0.86, 0.0, Vec3::ZERO, 2.0, PROOF_DT_S);
        assert_eq!(wet.emitted_kg, 0.0);
        assert!(wet.parcels.is_empty());
        let (pond, _) = drive(10.0, 0.02, 0.03, Vec3::ZERO, 2.0, PROOF_DT_S);
        assert_eq!(pond.emitted_kg, 0.0);
        let (still, _) = drive(0.0, 0.0, 0.0, Vec3::ZERO, 2.0, PROOF_DT_S);
        assert_eq!(still.emitted_kg, 0.0);
    }

    #[test]
    fn rate_and_wake_density_follow_dryness_and_speed() {
        let dry8 = loft_rate_kg_s(0.0, 0.0, 8.0);
        let excess = 7.5_f32;
        approx(dry8, LOFT_K * excess * excess.sqrt(), 1.0e-6);
        approx(
            wake_density_kg_m3(0.0, 0.0, 8.0),
            dry8 / (8.0 * WAKE_SECTION_M2),
            1.0e-6,
        );

        let d30 = dryness(0.30, 0.0);
        approx(d30, (0.55 - 0.30) / 0.55, 1.0e-6);
        approx(loft_rate_kg_s(0.30, 0.0, 8.0), dry8 * d30, 1.0e-6);
        approx(
            wake_density_kg_m3(0.30, 0.0, 8.0),
            wake_density_kg_m3(0.0, 0.0, 8.0) * d30,
            1.0e-6,
        );

        let speeds = [1.0_f32, 3.0, 8.0, 14.0];
        let moist = [0.0_f32, 0.2, 0.4];
        for &m in &moist {
            for w in speeds.windows(2) {
                let a = wake_density_kg_m3(m, 0.0, w[0]);
                let b = wake_density_kg_m3(m, 0.0, w[1]);
                assert!(b > a, "speed {w:?} moisture {m}: {a} -> {b}");
            }
        }
        for &u in &speeds {
            for mm in moist.windows(2) {
                let a = wake_density_kg_m3(mm[0], 0.0, u);
                let b = wake_density_kg_m3(mm[1], 0.0, u);
                assert!(a > b && b > 0.0, "moisture {mm:?} at {u}: {a} -> {b}");
            }
        }
    }

    #[test]
    fn simulated_bulk_density_rises_with_dryness_and_speed() {
        let secs = 2.5;
        let bulk = |m, u| {
            let (d, z) = drive(u, m, 0.0, Vec3::ZERO, secs, PROOF_DT_S);
            d.near_wake_bulk_kg_m3(z)
        };
        let dry = bulk(0.02, 8.0);
        let damp = bulk(0.32, 8.0);
        let slow = bulk(0.02, 3.0);
        let wet = bulk(0.80, 8.0);
        let (pond, z) = drive(8.0, 0.02, 0.04, Vec3::ZERO, secs, PROOF_DT_S);
        assert!(dry > damp, "dry {dry} damp {damp}");
        assert!(dry > slow, "dry {dry} slow {slow}");
        assert!(damp > 0.0 && slow > 0.0);
        assert_eq!(wet, 0.0);
        assert_eq!(pond.near_wake_bulk_kg_m3(z), 0.0);
        assert_eq!(pond.emitted_kg, 0.0);
    }

    #[test]
    fn cloud_sits_behind_the_tires_and_wind_carries_it() {
        let (calm, z) = drive(
            PROOF_SPEED_M_S,
            PROOF_MOISTURE,
            0.0,
            Vec3::ZERO,
            PROOF_SECONDS,
            PROOF_DT_S,
        );
        let (blown, zb) = drive(
            PROOF_SPEED_M_S,
            PROOF_MOISTURE,
            0.0,
            proof_wind(),
            PROOF_SECONDS,
            PROOF_DT_S,
        );
        let (behind, lateral) = calm.rear_offset(z).unwrap();
        let (behind_w, lateral_w) = blown.rear_offset(zb).unwrap();
        assert!(behind > 3.0, "calm cloud behind {behind} m");
        assert!(behind_w > 3.0, "wind cloud behind {behind_w} m");
        assert!(lateral.abs() < 0.25, "calm lateral {lateral}");
        assert!(
            lateral_w > lateral + 1.0,
            "wind should carry the cloud toward +X: calm {lateral} wind {lateral_w}"
        );

        let mut ahead = 0.0_f32;
        let front_z = zb + HUBS[0][2];
        for p in &blown.parcels {
            if p.pos.z > front_z - 0.05 {
                ahead += p.mass_kg;
            }
        }
        assert!(
            ahead / blown.airborne_kg() < 0.02,
            "mass ahead of the front tires {}",
            ahead
        );

        let max_y = blown
            .parcels
            .iter()
            .map(|p| p.pos.y)
            .fold(0.0_f32, f32::max);
        assert!(max_y > 0.6 && max_y < 3.2, "cloud height {max_y}");
    }

    #[test]
    fn mass_ledger_closes_and_each_dry_tire_contributes() {
        let (dust, _) = drive(8.0, 0.05, 0.0, proof_wind(), 3.0, PROOF_DT_S);
        let sum = dust.airborne_kg() + dust.settled_kg;
        approx(sum, dust.emitted_kg, 1.0e-4);
        assert!(dust.emitted_kg > 0.05);
        let mut per = [0.0_f32; 4];
        for p in &dust.parcels {
            per[p.wheel as usize] += p.mass_kg;
        }
        for (i, m) in per.iter().enumerate() {
            assert!(*m > 0.0, "tire {i} lofted nothing");
        }
    }

    #[test]
    fn a_wet_patch_under_one_tire_does_not_raise_dust_there() {
        let mut moisture = [0.04; 4];
        let mut water = [0.0; 4];
        moisture[2] = 0.9;
        water[1] = 0.05;
        let (dust, _) = drive_mixed(8.0, moisture, water, Vec3::ZERO, 1.5, PROOF_DT_S);
        let mut per = [0.0_f32; 4];
        for p in &dust.parcels {
            per[p.wheel as usize] += p.mass_kg;
        }
        assert_eq!(per[1], 0.0, "standing water");
        assert_eq!(per[2], 0.0, "wet moisture");
        assert!(per[0] > 0.0 && per[3] > 0.0);
    }

    #[test]
    fn proof_frame_has_a_dry_cloud_and_a_zero_wet_control() {
        let (dust, z) = drive(
            PROOF_SPEED_M_S,
            PROOF_MOISTURE,
            PROOF_WATER_M,
            proof_wind(),
            PROOF_SECONDS,
            PROOF_DT_S,
        );
        let (behind, lateral) = dust.rear_offset(z).unwrap();
        assert!(dust.airborne_kg() > 0.05, "{}", dust.airborne_kg());
        assert!(behind > 3.0, "{behind}");
        assert!(lateral > 0.8, "{lateral}");
        assert!(dust.near_wake_bulk_kg_m3(z) > 0.0);
        let (wet, _) = drive(
            PROOF_SPEED_M_S,
            0.92,
            0.0,
            proof_wind(),
            PROOF_SECONDS,
            PROOF_DT_S,
        );
        assert_eq!(wet.emitted_kg, 0.0);
        let rate = loft_rate_kg_s(PROOF_MOISTURE, 0.0, PROOF_SPEED_M_S);
        let expected = rate * 4.0 * PROOF_SECONDS;
        approx(dust.emitted_kg, expected, expected * 0.002);
    }
}
