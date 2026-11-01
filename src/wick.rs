//! Capillary pore-water transport between ground cells.
//!
//! Pore moisture used to change only inside each cell: every kind relaxed to
//! its own wet or dry target. Two kinds side by side met as a hard step. Here
//! water also moves sideways, cell to cell, down the total head: the height of
//! the cell minus its matric suction. Suction follows Brooks-Corey for each
//! soil class, and so does the unsaturated conductivity, taken from the cell
//! the water leaves. A clay beside sand pulls water out of the sand at the
//! same saturation, because clay holds its water at a much higher suction.
//! Sand next to wet mud is fed by it and stays damp a few cells out, until
//! drying in each cell wins over the supply. That damp band is the wet edge
//! at a soil joint. Nothing here adds or removes water: what one cell loses
//! the other gains, in cubic metres of pore water.

use crate::soil::SurfaceKind;

/// Soil classes with their own water retention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Gravel = 0,
    Sand = 1,
    Loam = 2,
    Silt = 3,
    Clay = 4,
}

pub const CLASSES: usize = 5;

/// Brooks-Corey retention and conductivity of one class.
#[derive(Clone, Copy, Debug)]
pub struct Hydraulics {
    /// Air-entry suction, metres of water.
    pub air_entry_m: f32,
    /// Pore-size index.
    pub lambda: f32,
    /// Saturated conductivity, m/s.
    pub ks: f32,
    /// Porosity, m³/m³.
    pub porosity: f32,
}

/// Textbook values (Rawls, Brakensiek and Saxton 1982, rounded).
pub const HYDRAULICS: [Hydraulics; CLASSES] = [
    Hydraulics { air_entry_m: 0.02, lambda: 0.70, ks: 1.0e-3, porosity: 0.30 },
    Hydraulics { air_entry_m: 0.07, lambda: 0.59, ks: 5.8e-5, porosity: 0.42 },
    Hydraulics { air_entry_m: 0.11, lambda: 0.25, ks: 3.7e-6, porosity: 0.46 },
    Hydraulics { air_entry_m: 0.21, lambda: 0.23, ks: 1.9e-6, porosity: 0.50 },
    Hydraulics { air_entry_m: 0.37, lambda: 0.13, ks: 1.7e-7, porosity: 0.48 },
];

/// The moisture model runs on a compressed clock: sand drains in seconds,
/// clay bakes in minutes. Capillary flow is run on the same clock so the two
/// stay in proportion.
pub const TIME_SCALE: f32 = 3000.0;
/// Depth of the pore layer that trades water sideways, metres. Pore water
/// moved out of a ponded cell comes out of its pond over this depth.
pub const LAYER_M: f32 = 0.10;
/// Lowest effective saturation for suction. Drier pores are held as this.
const SE_MIN: f32 = 0.03;
/// Most of a cell's pore water one edge may move in one step.
const EDGE_SHARE_MAX: f32 = 0.12;
/// Table size for suction and conductivity over saturation 0..1.
const LUT: usize = 64;

pub fn class_of(kind: SurfaceKind) -> Class {
    match kind {
        SurfaceKind::Gravel => Class::Gravel,
        SurfaceKind::Sand => Class::Sand,
        SurfaceKind::Silt => Class::Silt,
        SurfaceKind::Undisturbed | SurfaceKind::Grass => Class::Loam,
        SurfaceKind::DryCrust
        | SurfaceKind::WetMud
        | SurfaceKind::FreshCut
        | SurfaceKind::CutLip
        | SurfaceKind::Berm
        | SurfaceKind::PlacedClod
        | SurfaceKind::PackedClay => Class::Clay,
    }
}

/// Matric suction, metres of water, at pore saturation `s`.
pub fn suction(h: &Hydraulics, s: f32) -> f32 {
    h.air_entry_m * s.clamp(SE_MIN, 1.0).powf(-1.0 / h.lambda)
}

/// Unsaturated conductivity, m/s, at pore saturation `s`.
pub fn conductivity(h: &Hydraulics, s: f32) -> f32 {
    h.ks * s.clamp(0.0, 1.0).powf(3.0 + 2.0 / h.lambda)
}

struct Tables {
    suction: [[f32; LUT + 1]; CLASSES],
    conductivity: [[f32; LUT + 1]; CLASSES],
}

fn tables() -> &'static Tables {
    static T: std::sync::OnceLock<Tables> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = Tables {
            suction: [[0.0; LUT + 1]; CLASSES],
            conductivity: [[0.0; LUT + 1]; CLASSES],
        };
        for c in 0..CLASSES {
            for k in 0..=LUT {
                let s = k as f32 / LUT as f32;
                t.suction[c][k] = suction(&HYDRAULICS[c], s);
                t.conductivity[c][k] = conductivity(&HYDRAULICS[c], s);
            }
        }
        t
    })
}

fn lut(row: &[f32; LUT + 1], s: f32) -> f32 {
    let x = s.clamp(0.0, 1.0) * LUT as f32;
    let k = (x as usize).min(LUT - 1);
    let f = x - k as f32;
    row[k] + (row[k + 1] - row[k]) * f
}

/// Row bands. One call moves water across the edges of one band, over the
/// time since that band last ran, so the cost per call is a quarter field.
pub const BANDS: usize = 4;

/// Move pore water across the edges of one band of rows.
///
/// `wetness` is pore saturation 0..1 per cell. `cell` gives each cell's soil
/// class and the film it holds in its surface before it ponds, metres.
/// `height` is the ground surface, metres. `water` is the standing column,
/// metres. A ponded cell is saturated at zero suction with its pond depth as
/// pressure, and what it gives comes out of the pond. `dt` is this call's
/// step; the band runs for `dt * BANDS`. `band` is the caller's cursor.
/// Total water, `LAYER_M * porosity * wetness + water` summed over the
/// cells, is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn wick(
    wetness: &mut [f32],
    water: &mut [f32],
    height: &[f32],
    cell: impl Fn(usize) -> (Class, f32),
    res: usize,
    cell_m: f32,
    dt: f32,
    band: &mut usize,
) {
    let n = res * res;
    debug_assert_eq!(wetness.len(), n);
    if dt <= 0.0 || n == 0 {
        return;
    }
    let b = *band % BANDS;
    *band = (b + 1) % BANDS;
    let per = res.div_ceil(BANDS);
    let ra = b * per;
    let rb = (ra + per).min(res);
    if ra >= rb {
        return;
    }
    let t = tables();
    let gain = TIME_SCALE * dt * BANDS as f32 / cell_m;
    // State of the band and the row past it, read once at the start: class,
    // hold, total head, conductivity. Fluxes are Jacobi on that state and are
    // written into both ends at once, so the sum is exact.
    let rc = (rb + 1).min(res);
    thread_local! {
        static STATE: std::cell::RefCell<Vec<(Class, f32, f32, f32)>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }
    STATE.with(|cache| {
    let mut cache = cache.borrow_mut();
    cache.clear();
    for idx in ra * res..rc * res {
        let (class, hold) = cell(idx);
        let c = class as usize;
        let pond = water[idx] - hold;
        cache.push(if pond > 0.0 {
            (class, hold, height[idx] + pond, HYDRAULICS[c].ks)
        } else {
            let s = wetness[idx];
            (
                class,
                hold,
                height[idx] - lut(&t.suction[c], s),
                lut(&t.conductivity[c], s),
            )
        });
    }
    let base = ra * res;
    let mut edge = |a: usize, b: usize, wetness: &mut [f32], water: &mut [f32]| {
        let sa = cache[a - base];
        let sb = cache[b - base];
        // Gravel pores are too wide to hold a meniscus: it is a capillary
        // break and trades no water sideways.
        if sa.0 == Class::Gravel || sb.0 == Class::Gravel {
            return;
        }
        let dh = sa.2 - sb.2;
        if dh == 0.0 {
            return;
        }
        let ((from, sf), (to, st)) = if dh > 0.0 { ((a, sa), (b, sb)) } else { ((b, sb), (a, sa)) };
        let pf = HYDRAULICS[sf.0 as usize].porosity;
        let pt = HYDRAULICS[st.0 as usize].porosity;
        let pond = (water[from] - sf.1).max(0.0);
        let wet_to = water[to] > st.1;
        // Pond to pond is surface flow; the pipe step moves it.
        if pond > 0.0 && wet_to {
            return;
        }
        // Face conductivity. Between two pore cells it is the geometric mean:
        // a wet cell beside a dry one passes what the dry side can carry, so
        // a fine soil over a coarse one stays a capillary barrier and a sharp
        // step in clay softens over days, not seconds. A pond is a saturated
        // source and feeds the front at its own saturated conductivity.
        let k = if pond > 0.0 { sf.3 } else { (sf.3 * st.3).sqrt() };
        // Pore water moved, m³ per m³ of a cell-deep layer: q * dt / dx.
        let mut moved = k * dh.abs() / cell_m * gain;
        // A pond gives from the pond; its pores stay full under it.
        let stock = if pond > 0.0 { pond / LAYER_M } else { pf * wetness[from] };
        moved = moved.min(EDGE_SHARE_MAX * stock);
        if !wet_to {
            moved = moved.min(EDGE_SHARE_MAX * pt * (1.0 - wetness[to]));
        }
        if moved <= 0.0 {
            return;
        }
        if pond > 0.0 {
            water[from] -= moved * LAYER_M;
        } else {
            wetness[from] -= moved / pf;
        }
        if wet_to {
            // Seepage into a pond rises into the pond.
            water[to] += moved * LAYER_M;
        } else {
            wetness[to] += moved / pt;
        }
    };
    for j in ra..rb {
        for i in 0..res {
            let a = j * res + i;
            if i + 1 < res {
                edge(a, a + 1, wetness, water);
            }
            if j + 1 < res {
                edge(a, a + res, wetness, water);
            }
        }
    }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn from_u8(c: u8) -> Class {
        [Class::Gravel, Class::Sand, Class::Loam, Class::Silt, Class::Clay][c as usize]
    }

    fn water(w: &[f32], pond: &[f32], class: &[u8]) -> f64 {
        w.iter()
            .zip(class)
            .zip(pond)
            .map(|((s, c), p)| f64::from(*s * HYDRAULICS[*c as usize].porosity * LAYER_M + *p))
            .sum()
    }

    #[test]
    fn clay_pulls_water_out_of_sand_at_the_same_saturation() {
        let s = 0.5;
        assert!(suction(&HYDRAULICS[Class::Clay as usize], s) > 5.0 * suction(&HYDRAULICS[Class::Sand as usize], s));
        let res = 8;
        let n = res * res;
        let class: Vec<u8> = (0..n).map(|i| if i % res < 4 { Class::Clay as u8 } else { Class::Sand as u8 }).collect();
        let mut w = vec![s; n];
        let height = vec![0.0; n];
        let mut pond = vec![0.0; n];
        let hold = vec![0.002; n];
        let before = water(&w, &pond, &class);
        let mut band = 0;
        for _ in 0..200 {
            wick(&mut w, &mut pond, &height, |i| (from_u8(class[i]), hold[i]), res, 0.208, 1.0 / 60.0, &mut band);
        }
        let after = water(&w, &pond, &class);
        assert!((before - after).abs() < 1.0e-5, "water {before} -> {after}");
        let clay_edge = w[3];
        let sand_edge = w[4];
        assert!(clay_edge > s + 0.005, "clay did not drink, {clay_edge}");
        assert!(sand_edge < s - 0.005, "sand did not give, {sand_edge}");
    }

    #[test]
    fn a_pond_feeds_dry_sand_and_keeps_the_sum() {
        let res = 12;
        let n = res * res;
        let class = vec![Class::Sand as u8; n];
        let mut w = vec![0.06; n];
        let mut pond = vec![0.0; n];
        let hold = vec![0.002; n];
        for j in 0..res {
            pond[j * res] = 0.05;
        }
        let height = vec![0.0; n];
        let before = water(&w, &pond, &class);
        let mut band = 0;
        for _ in 0..240 {
            wick(&mut w, &mut pond, &height, |i| (from_u8(class[i]), hold[i]), res, 0.208, 1.0 / 60.0, &mut band);
        }
        let after = water(&w, &pond, &class);
        assert!((before - after).abs() < 1.0e-5, "water {before} -> {after}");
        assert!(pond[0] < 0.05, "the pond gave nothing");
        assert!(w[1] > 0.3 && w[1] > w[3], "no wetting front, {:?}", &w[..5]);
    }

    #[test]
    fn a_dry_field_stays_put() {
        let res = 16;
        let n = res * res;
        let class = vec![Class::Loam as u8; n];
        let mut w = vec![0.3; n];
        let mut pond = vec![0.0; n];
        let mut band = 0;
        for _ in 0..BANDS {
            wick(&mut w, &mut pond, &vec![0.0; n], |i| (from_u8(class[i]), 0.003), res, 0.2, 0.25, &mut band);
        }
        assert!(w.iter().all(|s| (*s - 0.3).abs() < 1.0e-6));
    }
}
