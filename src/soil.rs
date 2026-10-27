//! Heightfield soil: Bekker sinkage, plastic ruts, wetness and shallow water.

use glam::Vec3;

pub const MAP_SIZE: f32 = 40.0;
pub const MAP_RES: usize = 192;

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
    /// Metres carved out of the original surface. Feeds slip-sinkage drag.
    pub rut: f32,
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
        for i in 0..field.height.len() {
            field.height[i] = 0.0;
            field.initial_height[i] = 0.0;
            field.base_wetness[i] = w;
            field.wetness[i] = w;
            field.water[i] = water;
            field.loose[i] = if w > 0.6 { 0.02 } else { 0.0 };
            field.darken[i] = 0.06;
            field.roughness[i] = lerp(0.9, 0.45, w);
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
        let rut = bilerp(s00.6, s10.6, s01.6, s11.6, tx, ty).max(0.0);
        SoilSample {
            height,
            wetness,
            water,
            loose,
            darken,
            roughness,
            softness: softness_of(wetness, loose, water),
            rut,
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
            self.initial_height[idx] - self.height[idx],
        )
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

    /// Plastic rut, berms, albedo and roughness stamp for one wheel.
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
        let reach = len * 0.5 + width + self.cell * 2.0;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;

        let mut vol = 0.0_f32;
        let mut left_cells = [0usize; 64];
        let mut right_cells = [0usize; 64];
        let mut n_left = 0;
        let mut n_right = 0;
        let area = self.cell * self.cell;

        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * right.x + rel_z * right.z;
                let lz = rel_x * forward.x + rel_z * forward.z;
                let half_w = width * 0.5;
                let half_l = len * 0.5;
                // Cells the tire overlaps, not only those whose centre sits inside
                // the tread. Otherwise a 28 cm tire writes one dashed cell.
                let tread = half_w + self.cell * 0.5;
                let along_reach = half_l + self.cell * 0.5;
                let idx = j as usize * self.res + i as usize;

                let on_tread = lx.abs() <= tread && lz.abs() <= along_reach;
                let on_berm = lx.abs() > tread
                    && lx.abs() < tread + self.cell * 0.85
                    && lz.abs() <= along_reach;

                if on_tread {
                    let across = 1.0 - (lx.abs() / tread).clamp(0.0, 1.0);
                    let along = 1.0 - (lz.abs() / along_reach.max(0.05)).clamp(0.0, 1.0);
                    let wgt = (across * along).clamp(0.0, 1.0);
                    let already = (self.initial_height[idx] - self.height[idx]).max(0.0);
                    if already < want && wgt > 0.04 {
                        let room = (want - already).min(0.40 - already).max(0.0);
                        // Groove depth in one pass. Deeper than `groove` stays slow.
                        let rate = if already < groove { 48.0 } else { 2.2 };
                        let drop = room * (1.0 - (-dt * rate).exp()) * (0.62 + 0.38 * wgt);
                        self.height[idx] -= drop;
                        self.loose[idx] = (self.loose[idx] + drop * 0.45).min(0.16);
                        vol += drop * area;
                    }
                    // Tracks stay dark and change roughness even when the rut is already at depth.
                    let wet = self.wetness[idx];
                    self.darken[idx] = (self.darken[idx] + 16.0 * wgt * dt * (0.45 + wet)).min(1.0);
                    let target_r = lerp(0.62, 0.16, wet);
                    let blend = (5.0 * dt * wgt).min(0.45);
                    self.roughness[idx] = lerp(self.roughness[idx], target_r, blend);
                } else if on_berm {
                    if lx > 0.0 {
                        if n_right < right_cells.len() {
                            right_cells[n_right] = idx;
                            n_right += 1;
                        }
                    } else if n_left < left_cells.len() {
                        left_cells[n_left] = idx;
                        n_left += 1;
                    }
                }
            }
        }

        // Most of the cut soil sits on the shoulder, so the berm is the light lip
        // beside each groove. A wide scatter made a film instead of a ridge.
        let berm_vol = vol * 0.9;
        let n_berm = (n_left + n_right).max(1) as f32;
        let dh = berm_vol / (n_berm * area);
        for k in 0..n_left {
            let idx = left_cells[k];
            self.height[idx] += dh;
            self.roughness[idx] = lerp(self.roughness[idx], 0.96, 0.25);
            self.darken[idx] = lerp(self.darken[idx], 0.28, 0.15);
            self.loose[idx] = (self.loose[idx] + dh * 0.4).min(0.18);
        }
        for k in 0..n_right {
            let idx = right_cells[k];
            self.height[idx] += dh;
            self.roughness[idx] = lerp(self.roughness[idx], 0.96, 0.25);
            self.darken[idx] = lerp(self.darken[idx], 0.28, 0.15);
            self.loose[idx] = (self.loose[idx] + dh * 0.4).min(0.18);
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
                    self.height[idx] -= drop;
                    self.loose[idx] = (self.loose[idx] + drop * 0.55).min(0.2);
                    self.wetness[idx] = self.wetness[idx].max(0.9);
                    self.darken[idx] = (self.darken[idx] + 0.35 * w).min(1.0);
                    vol += drop * area;
                    dug_max = dug_max.max(drop);
                } else if d < 1.5 {
                    ring_n += 1.0;
                }
            }
        }
        if ring_n > 0.0 && vol > 0.0 {
            let dh = vol * 0.7 / (ring_n * area);
            for j in j0.max(1)..j1.min(res - 1) {
                for i in i0.max(1)..i1.min(res - 1) {
                    let cx = (i as f32 + 0.5) * self.cell - half;
                    let cz = (j as f32 + 0.5) * self.cell - half;
                    let d = ((cx - x).powi(2) + (cz - z).powi(2)).sqrt() / radius;
                    if (1.0..1.5).contains(&d) {
                        let idx = j as usize * self.res + i as usize;
                        let lip = 1.0 - (d - 1.0) / 0.5;
                        self.height[idx] += dh * lip;
                    }
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
        for (h0, h) in self.initial_height.iter().zip(self.height.iter()) {
            m = m.max(h0 - h);
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
    next: &mut [f32],
    a: usize,
    b: usize,
    dt: f32,
    dx: f32,
    area: f32,
    cond: f32,
) {
    if water[a] < 1.0e-5 && water[b] < 1.0e-5 {
        return;
    }
    let sa = height[a] + water[a];
    let sb = height[b] + water[b];
    let dh = sa - sb;
    if dh.abs() < 1.0e-5 {
        return;
    }
    // Volume across the shared face. Limited so one pass cannot drain a cell.
    let mut vol = dh * cond * dt * dx * 0.35;
    if vol > 0.0 {
        vol = vol.min(water[a] * area * 0.22);
        next[a] -= vol / area;
        next[b] += vol / area;
    } else {
        vol = (-vol).min(water[b] * area * 0.22);
        next[b] -= vol / area;
        next[a] += vol / area;
    }
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
}
