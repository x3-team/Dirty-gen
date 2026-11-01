//! Shrinkage fracture of a drying clay skin.
//!
//! The skin on each cell wants to shrink by a free strain that grows as the
//! pore moisture falls below the rewet point. The wet ground under it does
//! not shrink, so the skin is held to it by a shear bond. Neighbouring skin
//! cells are tied by springs across each edge: a normal spring along the
//! edge axis and a weaker shear spring across it. Equilibrium of that lattice
//! gives the tension on every edge. An edge whose tension passes the tensile
//! strength breaks for good and stops carrying load. The load it shed moves
//! to the edges next to it, so a fissure runs on from its tip and a network
//! of plates forms on its own. No pattern is authored. The only randomness is
//! the flaw spread of edge strength, a property of the material.
//!
//! The opening of a broken edge is the gap between the two plates: the
//! relative displacement plus the free shrinkage across that edge. A cell's
//! opening is the widest broken edge it touches. Nothing here moves the bed,
//! the placed pile, the slurry or the water.

/// Linear shrinkage strain per unit of moisture lost below the rewet point.
/// Clay shrinks a few percent between plastic and dry.
pub const SHRINK_PER_MOISTURE: f32 = 0.11;
/// Tensile failure strain of a drying clay skin, before the flaw spread.
pub const FAIL_STRAIN: f32 = 0.0085;
/// Spread of edge strength around [`FAIL_STRAIN`], as a fraction (0.3 is ±30 %).
pub const FLAW_SPREAD: f32 = 0.30;
/// Stress-transfer length of the thickest skin, metres. The bond to the wet
/// ground gives the skin a length over which a fissure relieves it. A thinner
/// skin is a softer plate on the same bond, so that length scales with the
/// square root of thickness.
pub const SHADOW_AT_MAX_M: f32 = 0.42;
/// Most strain the bond to the wet ground can carry per cell at the thickest
/// skin, before it slips. A plate shorter than about `2 * FAIL_STRAIN / SLIP`
/// cells cannot be pulled to failure, so drying past that widens the
/// fissures it has instead of cutting new ones. A thinner skin is held by the
/// same bond with less plate, so the cap in strain grows as the skin thins
/// and its plates come out smaller.
pub const SLIP_AT_MAX: f32 = 0.0030;
/// Shear spring stiffness as a fraction of the normal spring.
pub const SHEAR_RATIO: f32 = 0.45;
/// Gauss-Seidel sweeps per call. The displacements are kept between calls.
const SWEEPS: usize = 3;
/// Row bands of the plate box. One call relaxes one band.
pub const BANDS: usize = 6;
/// Most edges a single call may break. Drying is slow next to a sweep, so the
/// worst edges break first and the field rebalances before the next ones go.
const BREAKS_PER_CALL: usize = 64;

/// Edge bits on a cell: the edge to `+X` and the edge to `+Z`.
const EDGE_X: u8 = 1;
const EDGE_Z: u8 = 2;

/// Lattice state that lives between steps. Empty until a skin first forms.
#[derive(Clone, Debug, Default)]
pub struct SkinLattice {
    /// In-plane displacement over the cell size, X and Z.
    ux: Vec<f32>,
    uz: Vec<f32>,
    /// Broken-edge bits, [`EDGE_X`] and [`EDGE_Z`].
    broken: Vec<u8>,
    /// Edges broken so far. A fresh skin starts at 0.
    pub fissures: u32,
    /// Band of rows the next call relaxes.
    band: usize,
}

/// One cell of input to the lattice.
#[derive(Clone, Copy, Debug)]
pub struct SkinCell {
    /// Skin thickness, metres. At or under the onset the cell is not a plate.
    pub thickness: f32,
    /// Free shrinkage strain the skin wants.
    pub strain: f32,
}

/// Free shrinkage strain at a pore moisture. Zero at and above `rewet_at`.
pub fn free_strain(wetness: f32, rewet_at: f32) -> f32 {
    (rewet_at - wetness).max(0.0) * SHRINK_PER_MOISTURE
}

/// Stress-transfer length at a thickness, metres.
pub fn shadow_length(thickness: f32, max_thickness: f32) -> f32 {
    SHADOW_AT_MAX_M * (thickness.max(0.0) / max_thickness.max(1.0e-6)).sqrt()
}

fn flaw(idx: usize, axis: u8) -> f32 {
    // Integer hash of the edge, 0..1. The material's flaw spread, not a shape.
    let mut h = (idx as u32).wrapping_mul(0x9E37_79B1) ^ u32::from(axis).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297A_2D39);
    h ^= h >> 15;
    (h & 0x00FF_FFFF) as f32 / 16_777_215.0
}

fn edge_strength(idx: usize, axis: u8) -> f32 {
    FAIL_STRAIN * (1.0 + FLAW_SPREAD * (2.0 * flaw(idx, axis) - 1.0))
}

/// Displacement of one cell on a bond that slips past `cap`.
/// `diag` already holds the bond stiffness `bond`.
fn settle(rhs: f32, diag: f32, bond: f32, cap: f32) -> f32 {
    let elastic = rhs / diag.max(1.0e-9);
    if (bond * elastic).abs() <= cap {
        return elastic;
    }
    let springs = diag - bond;
    if springs <= 1.0e-9 {
        return 0.0;
    }
    (rhs - cap * elastic.signum()) / springs
}

impl SkinLattice {
    fn ensure(&mut self, n: usize) {
        if self.ux.len() != n {
            self.ux = vec![0.0; n];
            self.uz = vec![0.0; n];
            self.broken = vec![0; n];
            self.fissures = 0;
        }
    }

    /// Whether the edge from `idx` along `axis` is broken.
    pub fn is_broken(&self, idx: usize, axis_x: bool) -> bool {
        let bit = if axis_x { EDGE_X } else { EDGE_Z };
        self.broken.get(idx).is_some_and(|b| b & bit != 0)
    }

    /// Advance the lattice one call and write each cell's opening, metres.
    ///
    /// `cells` and `opening` are `res * res`. `cell_m` is the cell size.
    /// `onset` is the thickness under which a skin is still plastic.
    /// `max_thickness` sets [`shadow_length`]. A cell under the onset is not a
    /// plate: its edges heal and its opening is left to the caller.
    ///
    /// One call relaxes and checks one band of [`BANDS`] across the rows that
    /// hold a plate, then moves to the next band. Drying is slow next to a
    /// frame, so a band a call keeps the cost flat. Returns false when no cell
    /// is a plate, so the caller can skip the field.
    pub fn step(
        &mut self,
        res: usize,
        cell_m: f32,
        onset: f32,
        max_thickness: f32,
        cells: &[SkinCell],
        opening: &mut [f32],
    ) -> bool {
        let n = res * res;
        debug_assert_eq!(cells.len(), n);
        debug_assert_eq!(opening.len(), n);
        let plate = |idx: usize| cells[idx].thickness > onset;
        // Rows and columns that hold a plate.
        let (mut i0, mut i1, mut j0, mut j1) = (res, 0usize, res, 0usize);
        for (idx, c) in cells.iter().enumerate() {
            if c.thickness > onset {
                let i = idx % res;
                let j = idx / res;
                i0 = i0.min(i);
                i1 = i1.max(i);
                j0 = j0.min(j);
                j1 = j1.max(j);
            }
        }
        if i0 > i1 {
            if self.fissures != 0 || self.ux.iter().any(|u| *u != 0.0) {
                self.ux.iter_mut().for_each(|u| *u = 0.0);
                self.uz.iter_mut().for_each(|u| *u = 0.0);
                self.broken.iter_mut().for_each(|b| *b = 0);
                self.fissures = 0;
            }
            return false;
        }
        self.ensure(n);
        let rows = j1 - j0 + 1;
        let band = self.band % BANDS;
        self.band = (band + 1) % BANDS;
        let per = rows.div_ceil(BANDS);
        let ra = j0 + band * per;
        let rb = (ra + per).min(j1 + 1);
        if ra >= rb {
            return true;
        }

        // A soft cell lets go: its edges heal and it stops. An edge into a
        // soft cell is free, not broken: no load and no gap. One row of halo
        // on each side keeps the band's own edges honest.
        let ha = ra.saturating_sub(1);
        let hb = (rb + 1).min(res);
        for j in ha..hb {
            for i in 0..res {
                let idx = j * res + i;
                if !plate(idx) {
                    self.ux[idx] = 0.0;
                    self.uz[idx] = 0.0;
                }
                let b = self.broken[idx];
                if b == 0 {
                    continue;
                }
                let mut keep = b;
                if b & EDGE_X != 0 && !(i + 1 < res && plate(idx) && plate(idx + 1)) {
                    keep &= !EDGE_X;
                }
                if b & EDGE_Z != 0 && !(j + 1 < res && plate(idx) && plate(idx + res)) {
                    keep &= !EDGE_Z;
                }
                if keep != b {
                    self.fissures = self
                        .fissures
                        .saturating_sub((b & !keep).count_ones());
                    self.broken[idx] = keep;
                }
            }
        }

        // Edge spring: the mean plate stiffness, thickness over the thickest
        // skin; 0 when broken or when a side is soft. The bond and its slip
        // cap work out the same on every plate: a stiffer plate on the same
        // bond is what makes the shadow grow with the square root of thickness.
        let inv_max = 1.0 / max_thickness.max(1.0e-6);
        let kx = |s: &Self, idx: usize| -> f32 {
            if idx % res + 1 < res && plate(idx) && plate(idx + 1) && s.broken[idx] & EDGE_X == 0 {
                0.5 * (cells[idx].thickness + cells[idx + 1].thickness) * inv_max
            } else {
                0.0
            }
        };
        let kz = |s: &Self, idx: usize| -> f32 {
            if idx + res < n && plate(idx) && plate(idx + res) && s.broken[idx] & EDGE_Z == 0 {
                0.5 * (cells[idx].thickness + cells[idx + res].thickness) * inv_max
            } else {
                0.0
            }
        };
        let e = |a: usize, b: usize| 0.5 * (cells[a].strain + cells[b].strain);
        let bond = (cell_m / SHADOW_AT_MAX_M) * (cell_m / SHADOW_AT_MAX_M);
        let cap = SLIP_AT_MAX;

        for _ in 0..SWEEPS {
            for j in ra..rb {
                for i in i0..=i1 {
                    let idx = j * res + i;
                    if !plate(idx) {
                        continue;
                    }
                    let mut diag_x = bond;
                    let mut rhs_x = 0.0;
                    let mut diag_z = bond;
                    let mut rhs_z = 0.0;
                    // +X and -X: normal for ux, shear for uz.
                    let k = kx(self, idx);
                    if k > 0.0 {
                        let r = idx + 1;
                        diag_x += k;
                        rhs_x += k * (self.ux[r] + e(idx, r));
                        diag_z += k * SHEAR_RATIO;
                        rhs_z += k * SHEAR_RATIO * self.uz[r];
                    }
                    if i > 0 {
                        let l = idx - 1;
                        let k = kx(self, l);
                        if k > 0.0 {
                            diag_x += k;
                            rhs_x += k * (self.ux[l] - e(idx, l));
                            diag_z += k * SHEAR_RATIO;
                            rhs_z += k * SHEAR_RATIO * self.uz[l];
                        }
                    }
                    // +Z and -Z: normal for uz, shear for ux.
                    let k = kz(self, idx);
                    if k > 0.0 {
                        let u = idx + res;
                        diag_z += k;
                        rhs_z += k * (self.uz[u] + e(idx, u));
                        diag_x += k * SHEAR_RATIO;
                        rhs_x += k * SHEAR_RATIO * self.ux[u];
                    }
                    if j > 0 {
                        let d = idx - res;
                        let k = kz(self, d);
                        if k > 0.0 {
                            diag_z += k;
                            rhs_z += k * (self.uz[d] - e(idx, d));
                            diag_x += k * SHEAR_RATIO;
                            rhs_x += k * SHEAR_RATIO * self.ux[d];
                        }
                    }
                    self.ux[idx] = settle(rhs_x, diag_x, bond, cap);
                    self.uz[idx] = settle(rhs_z, diag_z, bond, cap);
                }
            }
        }

        // Edge tension as a strain: stretch over the cell plus the free
        // shrinkage. The worst edges past their strength break.
        let mut over: Vec<(f32, usize, u8)> = Vec::new();
        for j in ra..rb {
            for i in i0..=i1 {
                let idx = j * res + i;
                if kx(self, idx) > 0.0 {
                    let t = self.ux[idx + 1] - self.ux[idx] + e(idx, idx + 1);
                    let ratio = t / edge_strength(idx, EDGE_X);
                    if ratio > 1.0 {
                        over.push((ratio, idx, EDGE_X));
                    }
                }
                if kz(self, idx) > 0.0 {
                    let t = self.uz[idx + res] - self.uz[idx] + e(idx, idx + res);
                    let ratio = t / edge_strength(idx, EDGE_Z);
                    if ratio > 1.0 {
                        over.push((ratio, idx, EDGE_Z));
                    }
                }
            }
        }
        let most = BREAKS_PER_CALL.div_ceil(BANDS).max(4);
        if over.len() > most {
            over.select_nth_unstable_by(most - 1, |a, b| b.0.total_cmp(&a.0));
            over.truncate(most);
        }
        for (_, idx, bit) in over {
            self.broken[idx] |= bit;
            self.fissures += 1;
        }

        // Opening of a broken edge is the gap between its two plates,
        // metres. Every row of the box is rewritten, so a band that was not
        // relaxed this call still reports its last gaps.
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                if plate(idx) {
                    opening[idx] = 0.0;
                }
            }
        }
        for j in j0..=j1 {
            for i in i0..=i1 {
                let idx = j * res + i;
                let b = self.broken[idx];
                if b == 0 {
                    continue;
                }
                if b & EDGE_X != 0 && i + 1 < res {
                    let r = idx + 1;
                    let gap = ((self.ux[r] - self.ux[idx] + e(idx, r)) * cell_m).max(0.0);
                    opening[idx] = opening[idx].max(gap);
                    opening[r] = opening[r].max(gap);
                }
                if b & EDGE_Z != 0 && idx + res < n {
                    let u = idx + res;
                    let gap = ((self.uz[u] - self.uz[idx] + e(idx, u)) * cell_m).max(0.0);
                    opening[idx] = opening[idx].max(gap);
                    opening[u] = opening[u].max(gap);
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ONSET: f32 = 0.008;
    const MAX_T: f32 = 0.036;

    fn uniform(res: usize, thickness: f32, strain: f32) -> Vec<SkinCell> {
        vec![SkinCell { thickness, strain }; res * res]
    }

    fn run(lat: &mut SkinLattice, res: usize, cells: &[SkinCell], calls: usize) -> Vec<f32> {
        let mut open = vec![0.0; res * res];
        for _ in 0..calls {
            lat.step(res, 0.208, ONSET, MAX_T, cells, &mut open);
        }
        open
    }

    #[test]
    fn a_wet_skin_does_not_crack() {
        let res = 32;
        let mut lat = SkinLattice::default();
        let open = run(&mut lat, res, &uniform(res, MAX_T, FAIL_STRAIN * 0.5), 40);
        assert_eq!(lat.fissures, 0);
        assert!(open.iter().all(|o| *o == 0.0));
    }

    #[test]
    fn a_soft_field_skips_the_lattice() {
        let res = 16;
        let mut lat = SkinLattice::default();
        let mut open = vec![0.0; res * res];
        let busy = lat.step(res, 0.2, ONSET, MAX_T, &uniform(res, ONSET * 0.5, 0.05), &mut open);
        assert!(!busy);
        assert!(lat.ux.is_empty(), "allocated for nothing");
    }

    /// The network is connected lines, not scattered dots, and its cell
    /// spacing grows with the skin.
    #[test]
    fn a_drying_skin_breaks_into_plates_that_grow_with_thickness() {
        let res = 64;
        let mut stats = Vec::new();
        for thickness in [0.020, MAX_T] {
            let mut lat = SkinLattice::default();
            let mut open = vec![0.0; res * res];
            // Strain rises slowly, as a skin drying over a long spell.
            for step in 0..400 {
                let s = 0.045 * (step as f32 / 300.0).min(1.0);
                lat.step(res, 0.208, ONSET, MAX_T, &uniform(res, thickness, s), &mut open);
            }
            let cracked = open.iter().filter(|o| **o > 1.0e-4).count();
            // Connected: most broken edges touch another broken edge.
            let mut linked = 0;
            let mut total = 0;
            for idx in 0..res * res {
                for axis_x in [true, false] {
                    if !lat.is_broken(idx, axis_x) {
                        continue;
                    }
                    total += 1;
                    let i = idx % res;
                    let j = idx / res;
                    let mut near = 0;
                    for dj in -1i32..=1 {
                        for di in -1i32..=1 {
                            let (ni, nj) = (i as i32 + di, j as i32 + dj);
                            if ni < 0 || nj < 0 || ni >= res as i32 || nj >= res as i32 {
                                continue;
                            }
                            let nidx = nj as usize * res + ni as usize;
                            for ax in [true, false] {
                                if (nidx, ax) != (idx, axis_x) && lat.is_broken(nidx, ax) {
                                    near += 1;
                                }
                            }
                        }
                    }
                    if near >= 1 {
                        linked += 1;
                    }
                }
            }
            let max_open = open.iter().cloned().fold(0.0f32, f32::max);
            let plates = plate_sizes(&lat, res);
            let mean_plate = plates.iter().sum::<usize>() as f32 / plates.len().max(1) as f32;
            eprintln!("plates {} mean cells {mean_plate:.1}", plates.len());
            if std::env::var("SKIN_ASCII").is_ok() {
                for j in 0..24 {
                    let mut row = String::new();
                    for i in 0..48 {
                        let idx = j * res + i;
                        row.push(if lat.is_broken(idx, true) { '|' } else { ' ' });
                        row.push(if lat.is_broken(idx, false) { '_' } else { '.' });
                    }
                    eprintln!("{row}");
                }
            }
            eprintln!(
                "skin {thickness} m: fissures {} cracked cells {cracked}/{} linked {linked}/{total} max opening {max_open:.4} m",
                lat.fissures,
                res * res
            );
            stats.push((lat.fissures, linked, total, cracked, max_open, mean_plate));
        }
        for (thickness, (fissures, linked, total, _cracked, max_open, mean_plate)) in
            [0.020, MAX_T].into_iter().zip(stats.iter().copied())
        {
            assert!(fissures > 20, "no network at {thickness}");
            assert!(linked as f32 > 0.85 * total as f32, "dots, not lines: {linked}/{total}");
            assert!(max_open > 0.002 && max_open < 0.06, "opening {max_open}");
            assert!(mean_plate > 2.0, "crumbs, not plates, at {thickness}: {mean_plate}");
        }
        assert!(stats[1].5 > 6.0, "thick plates {}", stats[1].5);
        assert!(
            stats[1].5 > stats[0].5 * 1.5 && stats[1].0 < stats[0].0,
            "a thick skin should break into fewer, larger plates: thin {} thick {}",
            stats[0].0,
            stats[1].0
        );
    }

    /// Cells per plate: groups joined by intact edges.
    fn plate_sizes(lat: &SkinLattice, res: usize) -> Vec<usize> {
        let n = res * res;
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        for start in 0..n {
            if seen[start] {
                continue;
            }
            let mut stack = vec![start];
            seen[start] = true;
            let mut count = 0;
            while let Some(idx) = stack.pop() {
                count += 1;
                let i = idx % res;
                let j = idx / res;
                let mut go = |nidx: usize, open: bool| {
                    if open && !seen[nidx] {
                        seen[nidx] = true;
                        stack.push(nidx);
                    }
                };
                if i + 1 < res {
                    go(idx + 1, !lat.is_broken(idx, true));
                }
                if i > 0 {
                    go(idx - 1, !lat.is_broken(idx - 1, true));
                }
                if j + 1 < res {
                    go(idx + res, !lat.is_broken(idx, false));
                }
                if j > 0 {
                    go(idx - res, !lat.is_broken(idx - res, false));
                }
            }
            out.push(count);
        }
        out
    }

    #[test]
    fn rewet_heals_the_network() {
        let res = 32;
        let mut lat = SkinLattice::default();
        run(&mut lat, res, &uniform(res, MAX_T, 0.04), 60);
        assert!(lat.fissures > 0);
        let mut open = vec![0.01; res * res];
        lat.step(res, 0.208, ONSET, MAX_T, &uniform(res, 0.0, 0.0), &mut open);
        assert_eq!(lat.fissures, 0);
        assert!(lat.broken.iter().all(|b| *b == 0));
    }
}
