//! Near field: the detail of the 2 cm sheet that outlives one step.
//!
//! The 2 cm patch under the tires is laid over the 20 cm sheet each step and
//! summed back into it. Before this module, `refine` laid every held column
//! down as one flat depth and one velocity, so a wave, a bow, or the hole a
//! tire left was flattened to 20 cm at the start of every step.
//!
//! What is kept here is the residual: fine depth minus the column mean for
//! water and slurry, and fine water velocity minus the column's
//! thickness-weighted velocity. Slurry keeps no velocity detail: it is a
//! Bingham mud that comes to rest inside a step, and a kept slurry velocity
//! thinned the standing slurry below its golden. The
//! residual of a column sums to zero, so it carries no volume and the
//! ledger does not see it. The 20 cm sheet stays the owner of every cubic
//! metre. `restrict_fine` captures the residual, `refine` lays it back on
//! top of the column's depth.
//!
//! Tiles are keyed by the world 20 cm cell, not by the patch, so the detail
//! stays where it was made when the patch moves with the Niva. A tile whose
//! column is no longer held fades with [`NEAR_RESIDUAL_LIFE_S`] and is
//! dropped when nothing is left in it.
//!
//! Velocity detail is kept only on columns whose four neighbours are held.
//! On a rim column the wall at the coarse surface drives a small drift that
//! a kept velocity would add up, step after step, into a hole in deep water
//! (`fine_patch_leaves_no_hole_in_deep_water`).
//!
//! A residual never makes a cell negative. If the column lost depth while it
//! was not held, the whole residual is scaled down until the deepest trough
//! just reaches zero. Scaling keeps the zero sum.

use super::*;

/// Fine cells along one side of a coarse cell.
pub(crate) const NEAR_SUB: usize = 10;
const NEAR_SUB2: usize = NEAR_SUB * NEAR_SUB;
/// e-folding time of the residual of a column the patch no longer holds, s.
pub const NEAR_RESIDUAL_LIFE_S: f32 = 1.5;
/// A faded tile is dropped once no depth residual is above this, metres.
const NEAR_DROP_M: f32 = 1.0e-5;
/// Film under this, metres, carries no velocity residual.
const NEAR_FILM_M: f32 = 1.0e-6;

/// Residual of one coarse column, row-major over its 10 × 10 fine cells.
#[derive(Clone)]
pub(crate) struct NearTile {
    w: [f32; NEAR_SUB2],
    s: [f32; NEAR_SUB2],
    uw: [f32; NEAR_SUB2],
    vw: [f32; NEAR_SUB2],
    /// Captured from a held column on the last restrict. Not faded this step.
    fresh: bool,
    /// Seconds since the column was last held. The fade is applied when the
    /// tile is laid down again, so an idle tile costs one add a step.
    idle_s: f32,
    /// Largest depth residual at capture, metres.
    peak: f32,
}

impl NearTile {
    fn zero() -> Self {
        Self {
            w: [0.0; NEAR_SUB2],
            s: [0.0; NEAR_SUB2],
            uw: [0.0; NEAR_SUB2],
            vw: [0.0; NEAR_SUB2],
            fresh: false,
            idle_s: 0.0,
            peak: 0.0,
        }
    }

    /// Share of the captured residual still there.
    fn left(&self) -> f32 {
        if self.idle_s <= 0.0 {
            1.0
        } else {
            (-self.idle_s / NEAR_RESIDUAL_LIFE_S).exp()
        }
    }
}

/// World-anchored residual tiles of the 2 cm water and slurry sheets.
pub(crate) struct NearField {
    tiles: HashMap<(i32, i32), Box<NearTile>>,
    /// Off: every step lays the patch down flat, as before this module.
    /// `SILT_NEAR=0` turns it off for a run; `FluidCore::set_near_field` for a test.
    on: bool,
}

impl Default for NearField {
    fn default() -> Self {
        Self {
            tiles: HashMap::new(),
            on: near_env_on(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn near_env_on() -> bool {
    std::env::var("SILT_NEAR").map(|v| v.trim() != "0").unwrap_or(true)
}

#[cfg(target_arch = "wasm32")]
fn near_env_on() -> bool {
    true
}

impl NearField {
    /// Tiles alive now. For tests and the perf print.
    pub(crate) fn len(&self) -> usize {
        self.tiles.len()
    }
}

/// Removes the mean so a residual sums to exactly zero in f32 as far as it can.
fn recentre(r: &mut [f32; NEAR_SUB2]) {
    let mean = r.iter().map(|&v| f64::from(v)).sum::<f64>() / NEAR_SUB2 as f64;
    let mean = mean as f32;
    for v in r.iter_mut() {
        *v -= mean;
    }
}

/// Largest |v|. Four lanes so the compiler can keep it in one vector register.
fn abs_peak(r: &[f32; NEAR_SUB2]) -> f32 {
    let mut hi = [0.0f32; 4];
    for c in r.chunks_exact(4) {
        for l in 0..4 {
            let a = c[l].abs();
            hi[l] = if a > hi[l] { a } else { hi[l] };
        }
    }
    hi[0].max(hi[1]).max(hi[2]).max(hi[3])
}

/// Scale that keeps `base + a * r` at or above zero everywhere, 0..=1.
fn floor_scale(base: f32, r: &[f32; NEAR_SUB2]) -> f32 {
    let low = r.iter().copied().fold(0.0f32, f32::min);
    if low >= 0.0 || base + low >= 0.0 {
        return 1.0;
    }
    if base <= 0.0 {
        return 0.0;
    }
    (base / -low).clamp(0.0, 1.0)
}

impl FluidCore {
    /// Held columns that lie wholly on the patch: (coarse index, world key, first fine i, first fine j).
    fn near_columns(&self) -> Vec<(usize, (i32, i32), usize, usize)> {
        let (col, row) = self.fine_coarse_axes();
        let runs = |axis: &[u16; FINE_RES]| {
            let mut out: Vec<(u16, usize)> = Vec::with_capacity(FINE_RES / NEAR_SUB + 2);
            let mut i = 0;
            while i < FINE_RES {
                let c = axis[i];
                let mut n = 1;
                while i + n < FINE_RES && axis[i + n] == c {
                    n += 1;
                }
                if c != u16::MAX && n == NEAR_SUB {
                    out.push((c, i));
                }
                i += n;
            }
            out
        };
        let cols = runs(&col);
        let rows = runs(&row);
        let mut out = Vec::new();
        for &(cj, j0) in &rows {
            for &(ci, i0) in &cols {
                let cidx = cj as usize * WINDOW_RES + ci as usize;
                if self.fine_hold.get(cidx).copied() != Some(1) {
                    continue;
                }
                let (x, z) = self.coarse_xz(ci as usize, cj as usize);
                out.push((cidx, world_key(x, z, COARSE_CELL_M), i0, j0));
            }
        }
        out
    }

    /// True when the four neighbours of this coarse column are held too.
    fn near_inner(&self, cidx: usize) -> bool {
        let i = cidx % WINDOW_RES;
        let j = cidx / WINDOW_RES;
        if i == 0 || j == 0 || i + 1 >= WINDOW_RES || j + 1 >= WINDOW_RES {
            return false;
        }
        [cidx - 1, cidx + 1, cidx - WINDOW_RES, cidx + WINDOW_RES]
            .iter()
            .all(|&n| self.fine_hold.get(n).copied() == Some(1))
    }

    /// Keep the sub-cell detail of every held column before the patch is summed.
    pub(super) fn near_capture(&mut self) {
        if !self.fine_on || !self.near.on {
            return;
        }
        for (cidx, key, i0, j0) in self.near_columns() {
            let mut sw = 0.0f32;
            let mut ss = 0.0f32;
            for b in 0..NEAR_SUB {
                let f = (j0 + b) * FINE_RES + i0;
                sw += self.fine_water[f..f + NEAR_SUB].iter().sum::<f32>();
                ss += self.fine_slurry[f..f + NEAR_SUB].iter().sum::<f32>();
            }
            // A dry column has no detail. Most held columns on a dry track are.
            if sw <= 0.0 && ss <= 0.0 {
                self.near.tiles.remove(&key);
                continue;
            }
            let n = NEAR_SUB2 as f32;
            let (mw, ms) = (sw / n, ss / n);
            let mut rw = [0.0f32; NEAR_SUB2];
            let mut rs = [0.0f32; NEAR_SUB2];
            for b in 0..NEAR_SUB {
                let f = (j0 + b) * FINE_RES + i0;
                let k = b * NEAR_SUB;
                for a in 0..NEAR_SUB {
                    rw[k + a] = self.fine_water[f + a] - mw;
                    rs[k + a] = self.fine_slurry[f + a] - ms;
                }
            }
            let peak = abs_peak(&rw).max(abs_peak(&rs));
            let inner = self.near_inner(cidx);
            // A flat rim column keeps nothing: laying it down adds nothing.
            if peak <= NEAR_DROP_M && !inner {
                self.near.tiles.remove(&key);
                continue;
            }
            recentre(&mut rw);
            recentre(&mut rs);
            let tile = self
                .near
                .tiles
                .entry(key)
                .or_insert_with(|| Box::new(NearTile::zero()));
            tile.w = rw;
            tile.s = rs;
            if inner && sw > 1.0e-10 {
                let (mut mu, mut mv) = (0.0f32, 0.0f32);
                for b in 0..NEAR_SUB {
                    let f = (j0 + b) * FINE_RES + i0;
                    for a in 0..NEAR_SUB {
                        let w = self.fine_water[f + a];
                        mu += w * self.fine_uw[f + a];
                        mv += w * self.fine_vw[f + a];
                    }
                }
                let (uw, vw) = (mu / sw, mv / sw);
                for b in 0..NEAR_SUB {
                    let f = (j0 + b) * FINE_RES + i0;
                    let k = b * NEAR_SUB;
                    for a in 0..NEAR_SUB {
                        let wet = self.fine_water[f + a] > NEAR_FILM_M;
                        tile.uw[k + a] = if wet { self.fine_uw[f + a] - uw } else { 0.0 };
                        tile.vw[k + a] = if wet { self.fine_vw[f + a] - vw } else { 0.0 };
                    }
                }
            } else {
                tile.uw = [0.0; NEAR_SUB2];
                tile.vw = [0.0; NEAR_SUB2];
            }
            tile.peak = peak;
            tile.idle_s = 0.0;
            tile.fresh = true;
        }
    }

    /// Lay the kept detail back on the flat columns `refine` just copied, then
    /// fade the tiles whose column the patch did not hold on the last step.
    pub(super) fn near_apply(&mut self, dt: f32) {
        if !self.near.tiles.is_empty() {
            for (cidx, key, i0, j0) in self.near_columns() {
                let Some(tile) = self.near.tiles.get(&key) else {
                    continue;
                };
                let f0 = j0 * FINE_RES + i0;
                let (bw, bs) = (self.fine_water[f0], self.fine_slurry[f0]);
                // Velocity detail only away from the patch rim. On a rim column
                // the wall at the coarse surface feeds a drift that a kept
                // velocity would add up step after step into a hole.
                let inner = self.near_inner(cidx);
                let left = tile.left();
                let aw = left * floor_scale(bw / left.max(1.0e-6), &tile.w);
                let as_ = left * floor_scale(bs / left.max(1.0e-6), &tile.s);
                for b in 0..NEAR_SUB {
                    for a in 0..NEAR_SUB {
                        let f = (j0 + b) * FINE_RES + i0 + a;
                        let k = b * NEAR_SUB + a;
                        let w = (bw + aw * tile.w[k]).max(0.0);
                        let s = (bs + as_ * tile.s[k]).max(0.0);
                        self.fine_water[f] = w;
                        self.fine_slurry[f] = s;
                        if inner && w > NEAR_FILM_M {
                            self.fine_uw[f] += left * tile.uw[k];
                            self.fine_vw[f] += left * tile.vw[k];
                        }
                    }
                }
            }
        }
        self.near_fade(dt);
    }

    /// Fade the tiles whose column the patch did not hold on the last restrict.
    pub(super) fn near_fade(&mut self, dt: f32) {
        if self.near.tiles.is_empty() {
            return;
        }
        let dt = if dt > 0.0 && dt.is_finite() { dt } else { 0.0 };
        self.near.tiles.retain(|_, t| {
            if t.fresh {
                t.fresh = false;
                return true;
            }
            t.idle_s += dt;
            t.peak * t.left() > NEAR_DROP_M
        });
    }

    /// Residual tiles alive now.
    pub fn near_tiles(&self) -> usize {
        self.near.len()
    }

    /// Keep (true, the default) or drop the 2 cm detail between steps.
    pub fn set_near_field(&mut self, on: bool) {
        self.near.on = on;
        if !on {
            self.near.tiles.clear();
        }
    }

    /// 2 cm water depth of the patch, row-major `FINE_RES` × `FINE_RES`, and its
    /// world origin. Empty between steps; the caller reads it after `refine`.
    pub fn fine_water_debug(&mut self, pushes: &[TirePush]) -> (Vec<f32>, [f32; 2]) {
        self.refine(pushes, 0.0);
        let out = (self.fine_water.clone(), self.fine_origin);
        self.restrict_fine();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_scale_keeps_cells_non_negative() {
        let mut r = [0.0f32; NEAR_SUB2];
        r[0] = -0.04;
        r[1] = 0.04;
        assert_eq!(floor_scale(0.05, &r), 1.0);
        let a = floor_scale(0.01, &r);
        assert!((0.01 + a * r[0]).abs() < 1.0e-7);
        assert_eq!(floor_scale(0.0, &r), 0.0);
    }

    #[test]
    fn recentre_sums_to_zero() {
        let mut r = [0.0f32; NEAR_SUB2];
        for (n, v) in r.iter_mut().enumerate() {
            *v = (n as f32 * 0.37).sin() * 0.01 + 0.003;
        }
        recentre(&mut r);
        let sum: f64 = r.iter().map(|&v| f64::from(v)).sum();
        assert!(sum.abs() < 1.0e-6, "{sum}");
    }

    fn tire(x: f32) -> TirePush {
        TirePush {
            x,
            z: 0.0,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.38,
            width: 0.2,
            speed: 3.0,
            slip: 0.3,
        }
    }

    fn fine_spread(core: &FluidCore) -> f32 {
        let mut spread = 0.0f32;
        for (_, _, i0, j0) in core.near_columns() {
            let mut lo = f32::MAX;
            let mut hi = f32::MIN;
            for b in 0..NEAR_SUB {
                for a in 0..NEAR_SUB {
                    let w = core.fine_water[(j0 + b) * FINE_RES + i0 + a];
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
            }
            spread = spread.max(hi - lo);
        }
        spread
    }

    /// The hole and bow a tire made are still on the 2 cm sheet when the
    /// next step lays the patch down, and laying it down moves no volume.
    #[test]
    fn detail_survives_refine_and_keeps_volume() {
        let dt = 1.0 / 30.0;
        let mut core = FluidCore::new([0.0, 0.0]);
        core.pour_water(-2.0, 2.0, -2.0, 2.0, 0.08);
        for _ in 0..6 {
            core.push_tire(tire(0.7));
            core.step(dt);
        }
        assert!(core.near_tiles() > 0, "no residual kept");
        let before = core.ledger().water_m3;
        core.refine(&[tire(0.7)], dt);
        let spread = fine_spread(&core);
        assert!(spread > 1.0e-3, "patch laid down flat, spread {spread}");
        let laid = core.ledger().water_m3;
        core.restrict_fine();
        let after = core.ledger().water_m3;
        assert!((laid - before).abs() < 1.0e-6, "refine moved {} m3", laid - before);
        assert!((after - before).abs() < 1.0e-6, "restrict moved {} m3", after - before);
        for w in &core.fine_water {
            assert!(*w >= 0.0);
        }
    }

    /// With no tire on the column, the kept detail fades and the tile goes.
    #[test]
    fn unheld_tiles_fade_out() {
        let dt = 1.0 / 30.0;
        let mut core = FluidCore::new([0.0, 0.0]);
        core.pour_water(-2.0, 2.0, -2.0, 2.0, 0.08);
        for _ in 0..6 {
            core.push_tire(tire(0.7));
            core.step(dt);
        }
        assert!(core.near_tiles() > 0);
        for _ in 0..(12.0 * NEAR_RESIDUAL_LIFE_S / dt) as usize {
            core.step(dt);
        }
        assert_eq!(core.near_tiles(), 0);
    }
}

#[cfg(test)]
mod bench {
    use super::*;

    /// Cost of one capture over a wet patch under four tires. Not a cap.
    /// `cargo test --release --lib near_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn near_bench() {
        let dt = 1.0 / 30.0;
        let mut core = FluidCore::new([0.0, 0.0]);
        core.pour_water(-3.0, 3.0, -3.0, 3.0, 0.08);
        let tire = |x: f32, z: f32| TirePush {
            x,
            z,
            forward_x: 0.0,
            forward_z: 1.0,
            radius: 0.38,
            width: 0.2,
            speed: 3.0,
            slip: 0.3,
        };
        let tires = [
            tire(-0.7, -1.2),
            tire(0.7, -1.2),
            tire(-0.7, 1.2),
            tire(0.7, 1.2),
        ];
        for _ in 0..20 {
            for p in tires {
                core.push_tire(p);
            }
            core.step(dt);
        }
        core.refine(&tires, dt);
        let n = 200;
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            core.near_capture();
        }
        let us = t0.elapsed().as_secs_f64() * 1.0e6 / n as f64;
        eprintln!(
            "NEARBENCH capture {us:.1} us over {} held columns",
            core.near_columns().len()
        );
    }
}
