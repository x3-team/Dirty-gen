//! River ford with a sand beach, as initial conditions for the shared sheet.
//!
//! This module only lays out the bed, the still water, and the drivers. It
//! does not shape a wave, a wake, a foam line, or a wet band. The current
//! comes from the bed slope and the open-channel lips, the flow around the
//! rocks and the run-up on the sand come from the height–velocity step, and
//! the wet sand is the pore water the film leaves behind. The renderer reads
//! all of it from the render contract.
//!
//! Layout, world metres, X east, Z north, centred on the ford:
//!
//! * The river runs east (+X) in a channel from `z = 1` to `z = 9`, thalweg
//!   0.5 m deep at `z = 5`, free surface falling `RIVER_SLOPE` per metre.
//! * The south bank is a sand beach, about 1:11, from 0.35 m of water at
//!   `z = 1` to the still waterline near `z = -3` and on up to a dry terrace.
//! * The north bank is a gravel bank, about 1:3, up to its terrace.
//! * Boulders sit in the channel, some under the surface and some standing
//!   out of it, with a clear lane at `x ∈ [-1.6, 1.6]` for the crossing.
//! * A paddle strip near the north bank sends a short swell south across the
//!   current, which runs up the beach.

use crate::fluid_core::{FluidCore, Grain, OpenWater, RiverReach, ShoreSoak, WaveMaker};

/// Free-surface fall of the river along +X, metres per metre.
pub const RIVER_SLOPE: f32 = 0.003;
/// Southern edge of the channel floor, where the beach starts to climb.
pub const CHANNEL_Z0: f32 = 1.0;
/// Northern edge of the channel floor, where the gravel bank starts.
pub const CHANNEL_Z1: f32 = 9.0;
/// Thalweg depth under the still surface, metres.
pub const THALWEG_M: f32 = 0.5;
/// Water depth at the foot of the beach, metres.
pub const BEACH_FOOT_M: f32 = 0.35;
/// Beach rise per metre southward.
pub const BEACH_SLOPE: f32 = 1.0 / 11.0;
/// Gravel bank rise per metre northward.
pub const BANK_SLOPE: f32 = 1.0 / 3.0;
/// Height of both terraces above the still surface at `x = 0`.
pub const TERRACE_M: f32 = 0.45;
/// Initial current, metres per second, on the channel floor.
pub const RIVER_SPEED: f32 = 0.75;
/// Half width of the rock-free crossing lane around `x = 0`.
pub const LANE_HALF_M: f32 = 1.6;
/// Free-surface diffusion on the reach, m²/s. See [`FluidCore::set_surface_diffusion`].
pub const SURFACE_DIFFUSION_M2_S: f32 = 0.024;
/// Manning roughness of the gravel-and-sand bed, s/m^(1/3).
pub const MANNING_N: f32 = 0.035;

fn manning_n() -> f32 {
    std::env::var("SILT_FORD_MANNING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MANNING_N)
}
/// Depth-averaged eddy viscosity of the reach, m²/s.
pub const EDDY_M2_S: f32 = 0.06;

fn eddy() -> f32 {
    std::env::var("SILT_FORD_EDDY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(EDDY_M2_S)
}

/// Pore moisture of the dry terrace sand.
pub const DRY_SAND_MOISTURE: f32 = 0.12;

/// A boulder as a bump in the bed. `top` is the crown, metres relative to the
/// still free surface at that `x`: negative is under water, positive stands out.
#[derive(Clone, Copy, Debug)]
pub struct Rock {
    pub x: f32,
    pub z: f32,
    pub rx: f32,
    pub rz: f32,
    pub yaw: f32,
    pub top: f32,
}

/// Boulders of the ford. Kept out of the crossing lane.
pub const ROCKS: [Rock; 11] = [
    Rock {
        x: -6.5,
        z: 4.8,
        rx: 1.1,
        rz: 0.8,
        yaw: 0.4,
        top: 0.42,
    },
    Rock {
        x: -11.0,
        z: 6.6,
        rx: 0.8,
        rz: 0.65,
        yaw: -0.3,
        top: 0.3,
    },
    Rock {
        x: -13.5,
        z: 3.4,
        rx: 0.9,
        rz: 0.6,
        yaw: 0.9,
        top: -0.08,
    },
    Rock {
        x: -3.8,
        z: 7.2,
        rx: 0.5,
        rz: 0.4,
        yaw: 0.0,
        top: -0.12,
    },
    Rock {
        x: -3.4,
        z: 3.2,
        rx: 0.7,
        rz: 0.55,
        yaw: 1.2,
        top: 0.2,
    },
    Rock {
        x: 4.0,
        z: 5.0,
        rx: 1.2,
        rz: 0.9,
        yaw: -0.6,
        top: 0.5,
    },
    Rock {
        x: 6.4,
        z: 2.6,
        rx: 0.5,
        rz: 0.42,
        yaw: 0.2,
        top: -0.06,
    },
    Rock {
        x: 8.6,
        z: 7.0,
        rx: 0.9,
        rz: 0.7,
        yaw: 0.7,
        top: 0.32,
    },
    Rock {
        x: 11.6,
        z: 4.2,
        rx: 1.0,
        rz: 0.7,
        yaw: -0.2,
        top: -0.1,
    },
    Rock {
        x: 14.8,
        z: 5.8,
        rx: 0.9,
        rz: 0.75,
        yaw: 0.5,
        top: 0.36,
    },
    Rock {
        x: -8.6,
        z: 0.3,
        rx: 0.6,
        rz: 0.45,
        yaw: 0.3,
        top: 0.18,
    },
];

/// Still free surface at `x`, metres.
pub fn still_surface(x: f32) -> f32 {
    -RIVER_SLOPE * x
}

/// Southern still waterline on the beach, `z` where the bed meets the surface.
pub fn still_waterline_z() -> f32 {
    CHANNEL_Z0 - BEACH_FOOT_M / BEACH_SLOPE
}

fn smooth01(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Bed relative to the still surface at that `x`, before the rocks.
fn bank_profile(z: f32) -> f32 {
    if z < CHANNEL_Z0 {
        // Beach: straight sand slope from the channel foot up to the terrace.
        let rise = -BEACH_FOOT_M + (CHANNEL_Z0 - z) * BEACH_SLOPE;
        rise.min(TERRACE_M)
    } else if z > CHANNEL_Z1 {
        let rise = -BEACH_FOOT_M + (z - CHANNEL_Z1) * BANK_SLOPE;
        rise.min(TERRACE_M)
    } else {
        // Channel floor: a shallow V from the beach foot down to the thalweg
        // and back up toward the gravel bank.
        let mid = 0.5 * (CHANNEL_Z0 + CHANNEL_Z1);
        let half = 0.5 * (CHANNEL_Z1 - CHANNEL_Z0);
        let t = 1.0 - ((z - mid) / half).abs();
        -(BEACH_FOOT_M + (THALWEG_M - BEACH_FOOT_M) * smooth01(0.0, 1.0, t))
    }
}

/// Height a rock adds to `base`, metres. A dome with a slightly lumpy crown.
fn rock_lift(rock: &Rock, x: f32, z: f32, base: f32, surface: f32) -> f32 {
    let (s, c) = rock.yaw.sin_cos();
    let dx = x - rock.x;
    let dz = z - rock.z;
    let a = (c * dx - s * dz) / rock.rx;
    let b = (s * dx + c * dz) / rock.rz;
    let r2 = a * a + b * b;
    if r2 >= 1.0 {
        return 0.0;
    }
    let crown = surface + rock.top;
    let rise = (crown - base).max(0.0);
    let ang = b.atan2(a);
    let lump = 1.0 + 0.1 * (3.0 * ang + rock.x).sin() * r2;
    // Boulder: a rounded top and steep sides, not a dome.
    rise * (1.0 - r2 * r2).max(0.0).powf(0.35) * lump
}

/// Bed and still water column at `(x, z)`, for [`FluidCore::map_water_bed`].
pub fn ford_bed(x: f32, z: f32) -> (f32, f32) {
    let surface = still_surface(x);
    let base = surface + bank_profile(z);
    let mut bed = base;
    for rock in &ROCKS {
        bed = bed.max(base + rock_lift(rock, x, z, base, surface));
    }
    (bed, (surface - bed).max(0.0))
}

/// True inside a boulder's footprint.
pub fn on_rock(x: f32, z: f32) -> bool {
    ROCKS.iter().any(|r| {
        let (s, c) = r.yaw.sin_cos();
        let dx = x - r.x;
        let dz = z - r.z;
        let a = (c * dx - s * dz) / r.rx;
        let b = (s * dx + c * dz) / r.rz;
        a * a + b * b < 1.0
    })
}

/// Grain of the solids: sand on the beach, silt on the channel floor,
/// clay-bound gravel on the north bank. Rocks keep the silt label; the sheet
/// does not erode them because nothing scours without a tire contact.
pub fn ford_grain(_x: f32, z: f32) -> Grain {
    if z < CHANNEL_Z0 + 0.6 {
        Grain::Sand
    } else if z > CHANNEL_Z1 {
        Grain::Clay
    } else {
        Grain::Silt
    }
}

/// Pore moisture before any wave: saturated under water, damp just above the
/// still line, dry sand on the terrace.
fn initial_moisture(x: f32, z: f32) -> f32 {
    let (bed, water) = ford_bed(x, z);
    if water > 0.0 {
        return 1.0;
    }
    let above = bed - still_surface(x);
    // Capillary fringe: 6 cm above the still line is still damp.
    DRY_SAND_MOISTURE + (0.85 - DRY_SAND_MOISTURE) * (1.0 - smooth01(0.0, 0.06, above))
}

/// Paddle settings for the swell that runs up the beach.
#[derive(Clone, Copy, Debug)]
pub struct Swell {
    pub amplitude_m_s: f32,
    pub period_s: f32,
}

impl Default for Swell {
    fn default() -> Self {
        Self {
            amplitude_m_s: 0.6,
            period_s: 4.5,
        }
    }
}

/// Lay the ford into `fluid`, whose window must be centred near the origin.
/// Returns the water volume on the sheet, cubic metres.
pub fn seed_ford(fluid: &mut FluidCore, swell: Option<Swell>) -> f32 {
    fluid.read_substrate(|x, z| {
        let (bed, _) = ford_bed(x, z);
        (bed, initial_moisture(x, z), ford_grain(x, z))
    });
    fluid.map_grain(ford_grain);
    let vol = fluid.map_water_bed(ford_bed);
    fluid.map_water_velocity(|x, z, h| {
        if !(CHANNEL_Z0..CHANNEL_Z1).contains(&z) || on_rock(x, z) {
            return (0.0, 0.0);
        }
        // Slower on the shallow margins, full speed on the thalweg.
        let k = (h / THALWEG_M).clamp(0.0, 1.0).sqrt();
        (RIVER_SPEED * k, 0.0)
    });
    let (ox, _, extent) = fluid.window_bounds();
    // Inlet and outlet on the channel rows only. The beach and the bank stay
    // closed at the window edge, so the swash is not fed or drained there.
    fluid.set_river_reach(Some(RiverReach {
        along_z: false,
        positive: true,
        cross0: CHANNEL_Z0 - 0.4,
        cross1: CHANNEL_Z1 + 0.4,
        rows: 3,
        inlet_eta: still_surface(ox),
        outlet_eta: still_surface(ox + extent),
        inlet_speed_m_s: RIVER_SPEED,
    }));
    fluid.set_wave_maker(swell.map(|s| WaveMaker {
        x0: ox + 1.0,
        x1: ox + extent - 1.0,
        z0: 6.6,
        z1: 8.6,
        dir_x: 0.0,
        dir_z: -1.0,
        amplitude_m_s: s.amplitude_m_s,
        period_s: s.period_s,
        ramp_s: 2.0 * s.period_s,
    }));
    fluid.set_surface_diffusion(SURFACE_DIFFUSION_M2_S);
    fluid.set_open_water(Some(OpenWater {
        mobility: 1.0,
        manning_n: manning_n(),
        eddy_m2_s: eddy(),
    }));
    fluid.set_shore_soak(Some(ShoreSoak {
        wet_film_m: 0.003,
        drain_s: 25.0,
    }));
    vol
}
