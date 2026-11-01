//! Prints in soft ground at 2.5 cm: a boot sole in wet sand, a ball's
//! groove in mud, the dent a dropped crate leaves.
//!
//! The coarse ground (the 20 cm soil grid or the sheet's bed) does not
//! change. This layer holds an offset over it on world-anchored 64 × 64
//! tiles, allocated on first touch and kept, so a trail stays for the whole
//! scene. Untouched ground is offset 0.
//!
//! A body presses the ground with a plate law, `p = k · s^n`, where `s` is
//! how far the cell has been pushed below where it started. That memory is
//! what makes a print carry the load that made it: a body standing in its
//! own print meets exactly its weight, and a heavier one sinks further.
//! Pressing is plastic. Of the volume pushed down, `bulge` comes up in a
//! rim around the print and the rest is compaction, booked on a ledger.
//! Dry sand then slides to its angle of repose; wet sand and mud keep the
//! wall.

use std::collections::HashMap;

use glam::Vec2;

use crate::fluid_core::WetShape;
use crate::soil::SurfaceKind;

pub const IMPRINT_CELL_M: f32 = 0.025;
const TILE_N: usize = 64;
/// Window side in cells, the same layout as the fine snow track.
pub const IMPRINT_RES: usize = 640;

/// How one ground material takes a print.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImprintMat {
    /// Plate modulus, Pa/m^n.
    pub k: f32,
    pub n: f32,
    /// Share of the pressed volume that comes up in the rim. The rest packs.
    pub bulge: f32,
    /// Steepest slope loose rim material keeps, rise over run.
    pub repose_tan: f32,
    /// Coulomb friction against a hard body.
    pub mu: f32,
}

/// Plate law for a ground kind. Calibrated so an 80 kg walker (about
/// 35 kPa under one sole) leaves about 1.5 cm in damp sand, 3–4 cm in dry
/// sand, 5–6 cm in soft mud and a few millimetres on firm ground.
pub fn imprint_mat(kind: SurfaceKind, wetness: f32, film_m: f32) -> ImprintMat {
    let w = wetness.clamp(0.0, 1.0);
    let flood = (film_m / 0.05).clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t.clamp(0.0, 1.0);
    let mut m = match kind {
        SurfaceKind::Sand => {
            // Dry sand flows, damp sand is at its stiffest from capillary
            // stick, saturated sand softens again.
            let damp = (1.0 - ((w - 0.40) / 0.30).abs()).clamp(0.0, 1.0);
            let wet = ((w - 0.70) / 0.30).clamp(0.0, 1.0);
            ImprintMat {
                k: lerp(lerp(1.0e6, 2.3e6, damp), 1.4e6, wet),
                n: 1.0,
                bulge: lerp(lerp(0.85, 0.60, damp), 0.70, wet),
                repose_tan: lerp(lerp(0.67, 2.2, damp), 1.1, wet),
                mu: lerp(0.62, 0.55, w),
            }
        }
        SurfaceKind::WetMud | SurfaceKind::Silt | SurfaceKind::FreshCut | SurfaceKind::Berm => {
            ImprintMat {
                k: lerp(3.0e6, 6.0e5, w),
                n: 1.0,
                bulge: lerp(0.55, 0.95, w),
                repose_tan: lerp(2.5, 1.3, w),
                mu: lerp(0.60, 0.30, w),
            }
        }
        SurfaceKind::Gravel => ImprintMat {
            k: 4.4e6,
            n: 1.0,
            bulge: 0.40,
            repose_tan: 0.80,
            mu: 0.70,
        },
        SurfaceKind::Grass => ImprintMat {
            k: lerp(6.0e6, 3.0e6, w),
            n: 1.0,
            bulge: 0.25,
            repose_tan: 3.0,
            mu: lerp(0.55, 0.40, w),
        },
        SurfaceKind::PackedClay | SurfaceKind::DryCrust => ImprintMat {
            k: lerp(1.2e7, 2.0e6, w),
            n: 1.0,
            bulge: 0.35,
            repose_tan: 3.0,
            mu: lerp(0.65, 0.35, w),
        },
        _ => ImprintMat {
            k: lerp(7.0e6, 1.2e6, w),
            n: 1.0,
            bulge: lerp(0.30, 0.80, w),
            repose_tan: lerp(3.0, 1.5, w),
            mu: lerp(0.60, 0.40, w),
        },
    };
    // Standing water: the sole meets a slick, softer top.
    m.mu *= lerp(1.0, 0.8, flood);
    m.k *= lerp(1.0, 0.85, flood);
    m
}

/// One 2.5 cm cell.
#[derive(Clone, Copy, Debug, Default)]
pub struct ImprintCell {
    /// Height over the coarse ground, metres. Negative in a print, positive on a rim.
    pub dz: f32,
    /// How far this cell has been pressed below where it was, metres.
    pub sink: f32,
}

/// What one press did.
#[derive(Clone, Copy, Debug, Default)]
pub struct PressOut {
    pub cells: u32,
    /// Deepest new push on this press, metres.
    pub deepest_m: f32,
    /// Volume pushed down, m³.
    pub pressed_m3: f32,
    /// Volume raised in the rim, m³.
    pub rim_m3: f32,
}

#[derive(Default)]
pub struct Imprint {
    tiles: HashMap<(i32, i32), Vec<ImprintCell>>,
    /// Volume that went into compaction, m³. With the offsets, the ledger:
    /// Σ dz·A + compacted = 0.
    pub compacted_m3: f64,
    pub pressed_m3: f64,
}

#[inline]
pub fn cell_of(x: f32) -> i32 {
    (x / IMPRINT_CELL_M).floor() as i32
}

#[inline]
pub fn cell_center(g: i32) -> f32 {
    (g as f32 + 0.5) * IMPRINT_CELL_M
}

impl Imprint {
    pub fn new() -> Self {
        Self::default()
    }

    fn key(gi: i32, gj: i32) -> ((i32, i32), usize) {
        let n = TILE_N as i32;
        (
            (gi.div_euclid(n), gj.div_euclid(n)),
            gj.rem_euclid(n) as usize * TILE_N + gi.rem_euclid(n) as usize,
        )
    }

    pub fn get(&self, gi: i32, gj: i32) -> ImprintCell {
        let (k, i) = Self::key(gi, gj);
        self.tiles.get(&k).map(|t| t[i]).unwrap_or_default()
    }

    pub fn cell_mut(&mut self, gi: i32, gj: i32) -> &mut ImprintCell {
        let (k, i) = Self::key(gi, gj);
        &mut self
            .tiles
            .entry(k)
            .or_insert_with(|| vec![ImprintCell::default(); TILE_N * TILE_N])[i]
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Offset at the cell holding the point, metres.
    pub fn offset_at(&self, x: f32, z: f32) -> f32 {
        self.get(cell_of(x), cell_of(z)).dz
    }

    /// Bilinear offset, metres.
    pub fn offset_bilinear(&self, x: f32, z: f32) -> f32 {
        let u = x / IMPRINT_CELL_M - 0.5;
        let v = z / IMPRINT_CELL_M - 0.5;
        let i = u.floor() as i32;
        let j = v.floor() as i32;
        let tx = u - i as f32;
        let ty = v - j as f32;
        let a = self.get(i, j).dz;
        let b = self.get(i + 1, j).dz;
        let c = self.get(i, j + 1).dz;
        let d = self.get(i + 1, j + 1).dz;
        let ab = a + (b - a) * tx;
        let cd = c + (d - c) * tx;
        ab + (cd - ab) * ty
    }

    /// Sum of offsets times cell area, m³. Zero when nothing moved, and
    /// minus the compaction once something has.
    pub fn net_m3(&self) -> f64 {
        let a = (IMPRINT_CELL_M * IMPRINT_CELL_M) as f64;
        self.tiles
            .values()
            .flat_map(|t| t.iter())
            .map(|c| c.dz as f64 * a)
            .sum()
    }

    /// Plate pressure if the cell were pushed to `y`, pascals. Zero when `y`
    /// is above the ground there. `ground` is the coarse height plus offset.
    pub fn pressure(&self, mat: ImprintMat, gi: i32, gj: i32, ground: f32, y: f32) -> f32 {
        if y > ground + 1.0e-4 {
            return 0.0;
        }
        let s = self.get(gi, gj).sink + (ground - y).max(0.0);
        mat.k * s.max(0.0).powf(mat.n)
    }

    /// Push every cell under the shape down onto its underside. `base` is
    /// the coarse ground, `mat` the plate law there. Volume pushed down comes
    /// up in a ring just outside the contact, by `mat.bulge`.
    pub fn press(
        &mut self,
        center: Vec2,
        shape: &WetShape,
        base: &dyn Fn(f32, f32) -> f32,
        mat: &dyn Fn(f32, f32) -> ImprintMat,
    ) -> PressOut {
        let mut out = PressOut::default();
        let reach = shape.reach() + IMPRINT_CELL_M;
        let area = IMPRINT_CELL_M * IMPRINT_CELL_M;
        let i0 = cell_of(center.x - reach);
        let i1 = cell_of(center.x + reach);
        let j0 = cell_of(center.y - reach);
        let j1 = cell_of(center.y + reach);
        let mut bulge_m3 = 0.0f32;
        let mut contact: Vec<(i32, i32)> = Vec::new();
        let mut covered: Vec<(i32, i32)> = Vec::new();
        for gj in j0..=j1 {
            for gi in i0..=i1 {
                let x = cell_center(gi);
                let z = cell_center(gj);
                let Some((lo, _)) = shape.span(x - center.x, z - center.y) else {
                    continue;
                };
                covered.push((gi, gj));
                let g = base(x, z) + self.get(gi, gj).dz;
                if lo >= g {
                    continue;
                }
                let d = g - lo;
                let m = mat(x, z);
                let c = self.cell_mut(gi, gj);
                c.dz -= d;
                c.sink += d;
                out.cells += 1;
                out.deepest_m = out.deepest_m.max(d);
                out.pressed_m3 += d * area;
                bulge_m3 += d * area * m.bulge.clamp(0.0, 1.0);
                contact.push((gi, gj));
            }
        }
        if out.cells == 0 {
            return out;
        }
        // Rim: a band two to three cells wide just outside the covered cells,
        // heaviest on the lip. Cells under the body take none.
        let band = 3i32;
        let mut ring: Vec<(i32, i32, f32)> = Vec::new();
        let mut wsum = 0.0f32;
        let is_cov = |gi: i32, gj: i32, cov: &[(i32, i32)]| cov.binary_search(&(gi, gj)).is_ok();
        let mut cov = covered.clone();
        cov.sort_unstable();
        let mut con = contact.clone();
        con.sort_unstable();
        for gj in (j0 - band)..=(j1 + band) {
            for gi in (i0 - band)..=(i1 + band) {
                if is_cov(gi, gj, &cov) {
                    continue;
                }
                // Distance in cells to the nearest pressed cell.
                let mut best = i32::MAX;
                for dj in -band..=band {
                    for di in -band..=band {
                        if is_cov(gi + di, gj + dj, &con) {
                            best = best.min(di.abs().max(dj.abs()));
                        }
                    }
                }
                if best == i32::MAX || best > band {
                    continue;
                }
                let w = (band + 1 - best) as f32;
                ring.push((gi, gj, w));
                wsum += w;
            }
        }
        if wsum > 0.0 && bulge_m3 > 0.0 {
            for &(gi, gj, w) in &ring {
                self.cell_mut(gi, gj).dz += bulge_m3 * w / wsum / area;
            }
            out.rim_m3 = bulge_m3;
        }
        self.pressed_m3 += out.pressed_m3 as f64;
        self.compacted_m3 += (out.pressed_m3 - out.rim_m3) as f64;
        // Loose rim slides to repose.
        let pad = band + 2;
        let m = mat(center.x, center.y);
        self.relax(i0 - pad, i1 + pad, j0 - pad, j1 + pad, m.repose_tan, 3, &cov);
        out
    }

    /// Slide offset down slopes steeper than `repose_tan`. Volume moves, it
    /// is not made. Cells still under the body (`hold`) do not move.
    pub fn relax(
        &mut self,
        i0: i32,
        i1: i32,
        j0: i32,
        j1: i32,
        repose_tan: f32,
        iters: usize,
        hold: &[(i32, i32)],
    ) {
        let max_step = repose_tan * IMPRINT_CELL_M;
        for _ in 0..iters {
            let mut moved = false;
            for gj in j0..=j1 {
                for gi in i0..=i1 {
                    if hold.binary_search(&(gi, gj)).is_ok() {
                        continue;
                    }
                    let h = self.get(gi, gj).dz;
                    for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                        let (ni, nj) = (gi + di, gj + dj);
                        if hold.binary_search(&(ni, nj)).is_ok() {
                            continue;
                        }
                        let hn = self.get(ni, nj).dz;
                        let drop = h - hn - max_step;
                        if drop > 1.0e-5 {
                            let q = drop * 0.25;
                            self.cell_mut(gi, gj).dz -= q;
                            self.cell_mut(ni, nj).dz += q;
                            moved = true;
                        }
                    }
                }
            }
            if !moved {
                break;
            }
        }
    }

    /// Window of `IMPRINT_RES`² offsets centred on `center`, the layout of the
    /// fine snow track window: R surface offset, G 0 (no snow), B pressed
    /// depth `sink`, A 0.
    pub fn window(&self, center: Vec2) -> (Vec2, Vec<[f32; 4]>) {
        let half = IMPRINT_RES as i32 / 2;
        let gi0 = cell_of(center.x) - half;
        let gj0 = cell_of(center.y) - half;
        let mut texels = vec![[0.0f32; 4]; IMPRINT_RES * IMPRINT_RES];
        for j in 0..IMPRINT_RES {
            for i in 0..IMPRINT_RES {
                let c = self.get(gi0 + i as i32, gj0 + j as i32);
                texels[j * IMPRINT_RES + i] = [c.dz, 0.0, c.sink, 0.0];
            }
        }
        (
            Vec2::new(gi0 as f32 * IMPRINT_CELL_M, gj0 as f32 * IMPRINT_CELL_M),
            texels,
        )
    }
}
