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
    /// Seconds since a tire last claimed this cell. A gap starts a new pass.
    pub rest: Vec<f32>,
    /// Distinct passes that have cut this cell. The groove deepens with this.
    pub passes: Vec<u8>,
}

/// A tire that leaves and comes back after this many seconds cuts a new pass.
const PASS_GAP: f32 = 0.40;

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

/// Terramechanics for one contact. A wet rut is remolded and lubricated:
/// the modulus drops so a slow tire sinks, and friction drops so the tread
/// slips, while the yield that holds the chassis stays.
pub fn terramech_at(softness: f32, rut: f32, water: f32) -> Terramech {
    let mut tm = terramech(softness);
    let wet = (water / 0.08).clamp(0.0, 1.0);
    let cut = (rut / 0.05).clamp(0.0, 1.0);
    let lube = wet * cut;
    if lube > 0.0 {
        tm.k_eq *= lerp(1.0, 0.58, lube);
        tm.phi *= lerp(1.0, 0.36, lube);
        tm.cohesion *= lerp(1.0, 0.32, lube);
        tm.shear_k *= lerp(1.0, 2.2, lube);
        tm.yield_stress *= lerp(1.0, 1.15, lube);
    }
    tm
}

/// Wetness alone makes the ground heavy. Standing water and loose spoil
/// are what push it into a true bog.
pub fn softness_of(wetness: f32, loose: f32, water: f32) -> f32 {
    let loose_n = (loose / 0.09).clamp(0.0, 1.0);
    let depth = (water / 0.18).clamp(0.0, 1.0);
    (wetness.clamp(0.0, 1.0) * 0.48 + depth * 0.34 + loose_n * 0.22).clamp(0.0, 1.0)
}

/// Softness at a contact. Dry ground is unchanged. Water sitting in a cut
/// remolds the floor, so the next tire sinks further than it would on the
/// same wetness with no rut.
pub fn softness_with_rut(wetness: f32, loose: f32, water: f32, rut: f32) -> f32 {
    let base = softness_of(wetness, loose, water);
    let wet = (water / 0.10).clamp(0.0, 1.0);
    let cut = (rut / 0.06).clamp(0.0, 1.0);
    let remold = cut * wet;
    (base + 0.35 * remold * (1.0 - base)).clamp(0.0, 1.0)
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
            rest: vec![PASS_GAP + 1.0; n],
            passes: vec![0; n],
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
            softness: softness_with_rut(wetness, loose, water, rut),
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

    /// Plastic rut under one tire.
    ///
    /// `lane` is the visible groove, stamped once per side of the axle along
    /// the path the chassis drove. Front and rear share that slot. The cut is
    /// a flat ribbon wider than one heightfield cell: a thinner kernel
    /// stair-steps into a row of pits, and those pits read as a nest of scars
    /// on the berm. The per-tire call is only a scuff under the tread, plus
    /// the slow dig when a tire spins in place.
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
        lane: bool,
    ) {
        if sinkage < 0.004 {
            return;
        }
        let len = contact_length(radius, sinkage).max(radius * 0.35);
        let slip_n = slip.abs().min(1.2);
        // The groove is the Bekker bite of one pass. A second pass of the same
        // lane raises the pass count and cuts deeper into that slot. Darkening
        // the albedo does not stand in for the missing depth. Slip still digs
        // past the rolling groove, on the slow rate, so a driver can back out
        // of a hole they are spinning.
        let one_pass = (sinkage * 0.82).min(0.055);
        let spun = sinkage * (0.48 + 0.90 * slip_n);
        let sigma = (width * 0.42).max(self.cell * 0.34);
        // Flat floor, about two cells wide, then a short shoulder. A gaussian
        // narrower than the grid left a dotted scar beside every tire.
        let half_w = if lane { self.cell * 1.15 } else { sigma * 1.6 };
        let shoulder = if lane { self.cell * 0.7 } else { 0.0 };
        let half_l = if lane {
            (len * 0.5).max(self.cell * 2.2)
        } else {
            len * 0.5
        };
        let along_reach = if lane {
            half_l + self.cell * 0.8
        } else {
            half_l + self.cell * 0.35
        };
        // Berm starts just outside the cut and is wide enough to hold the
        // whole displaced volume without a second groove.
        let berm_inner = if lane {
            half_w + shoulder * 0.82
        } else {
            sigma * 1.85
        };
        let berm_outer = berm_inner + self.cell * 3.0;
        let lateral_reach = berm_outer + self.cell * 0.35;
        let reach = along_reach + lateral_reach;
        let half = self.size * 0.5;
        let i0 = (((center.x - reach) + half) / self.cell).floor() as isize;
        let i1 = (((center.x + reach) + half) / self.cell).floor() as isize;
        let j0 = (((center.z - reach) + half) / self.cell).floor() as isize;
        let j1 = (((center.z + reach) + half) / self.cell).floor() as isize;
        let res = self.res as isize;

        let mut vol = 0.0_f32;
        let mut berms: Vec<(usize, f32)> = Vec::new();
        let area = self.cell * self.cell;
        // Only the lane ribbon counts as a pass. A tread scuff must not
        // promote the same visit into a second, deeper cut.
        let claim_pass = lane;

        for j in j0.max(1)..j1.min(res - 1) {
            for i in i0.max(1)..i1.min(res - 1) {
                let cx = (i as f32 + 0.5) * self.cell - half;
                let cz = (j as f32 + 0.5) * self.cell - half;
                let rel_x = cx - center.x;
                let rel_z = cz - center.z;
                let lx = rel_x * right.x + rel_z * right.z;
                let lz = rel_x * forward.x + rel_z * forward.z;
                let along = if lane {
                    let u = (lz.abs() / along_reach.max(0.05)).clamp(0.0, 1.0);
                    if u < 0.78 {
                        1.0
                    } else {
                        ((1.0 - u) / 0.22).clamp(0.0, 1.0)
                    }
                } else {
                    1.0 - (lz.abs() / along_reach.max(0.05)).clamp(0.0, 1.0)
                };
                if along <= 0.0 {
                    continue;
                }
                let idx = j as usize * self.res + i as usize;
                let ax = lx.abs();
                let across = if lane {
                    if ax <= half_w {
                        1.0
                    } else if shoulder > 0.0 && ax <= half_w + shoulder {
                        1.0 - (ax - half_w) / shoulder
                    } else {
                        0.0
                    }
                } else {
                    (-0.5 * (lx / sigma.max(0.02)).powi(2)).exp()
                };

                if across > 0.2 {
                    let passes = self.note_pass(idx, claim_pass);
                    let groove = pass_groove(one_pass, passes);
                    let want = spun.max(groove).min(0.40);
                    // A rolling per-tire stamp stops at a scuff. The visible
                    // pair is the lane call. Spinning still reaches `want`.
                    let fast = if lane { groove } else { groove.min(0.008) };
                    let already = (self.initial_height[idx] - self.height[idx]).max(0.0);
                    let local_want = if lane {
                        want * across
                    } else if spun > one_pass {
                        want * across
                    } else {
                        fast * across
                    };
                    let local_fast = fast * across;
                    if already < local_want {
                        let room = (local_want - already).min(0.40 - already).max(0.0);
                        let rate = if already < local_fast { 36.0 } else { 2.2 };
                        let drop = room * (1.0 - (-dt * rate).exp()) * along;
                        self.height[idx] -= drop;
                        self.loose[idx] = (self.loose[idx] + drop * 0.45).min(0.16);
                        vol += drop * area;
                    }
                    let wet = self.wetness[idx];
                    self.darken[idx] =
                        (self.darken[idx] + 16.0 * across * along * dt * (0.45 + wet)).min(1.0);
                    let target_r = lerp(0.62, 0.16, wet);
                    let blend = (5.0 * dt * across * along).min(0.45);
                    self.roughness[idx] = lerp(self.roughness[idx], target_r, blend);
                } else if ax > berm_inner && ax < berm_outer {
                    let span = (berm_outer - berm_inner).max(self.cell * 0.25);
                    let edge = 1.0 - ((ax - berm_inner) / span).clamp(0.0, 1.0);
                    let w = along * (0.35 + 0.65 * edge);
                    if w > 0.0 {
                        berms.push((idx, w));
                    }
                }
            }
        }

        // Every cubic metre cut out of the rut is piled on the shoulders.
        // Nothing is deleted and nothing is painted back into the floor.
        self.deposit_berm(&berms, vol, area);
    }

    fn note_pass(&mut self, idx: usize, claim: bool) -> u8 {
        if claim {
            if self.rest[idx] > PASS_GAP {
                self.passes[idx] = self.passes[idx].saturating_add(1);
            }
            self.rest[idx] = 0.0;
        }
        self.passes[idx].max(1)
    }

    fn deposit_berm(&mut self, berms: &[(usize, f32)], vol: f32, area: f32) {
        if vol <= 1.0e-8 || berms.is_empty() {
            return;
        }
        let sum: f32 = berms.iter().map(|(_, w)| *w).sum();
        if sum <= 1.0e-8 {
            return;
        }
        for (idx, w) in berms {
            let add = vol * (*w / sum) / area;
            self.height[*idx] += add;
            self.roughness[*idx] = lerp(self.roughness[*idx], 0.96, 0.25);
            self.darken[*idx] = lerp(self.darken[*idx], 0.28, 0.15);
            self.loose[*idx] = (self.loose[*idx] + add * 0.4).min(0.18);
        }
    }

    /// One rolling pass of a lane-width tire. `dir` is the travel direction.
    /// Returns the rut depth halfway along the pass.
    pub fn roll_pass(
        &mut self,
        origin: Vec3,
        dir: Vec3,
        distance: f32,
        sinkage: f32,
        slip: f32,
    ) -> f32 {
        let dir = dir.normalize_or_zero();
        if dir.length_squared() < 0.5 {
            return 0.0;
        }
        let right = Vec3::new(dir.z, 0.0, -dir.x);
        let step = (self.cell * 0.55).max(0.08);
        let mut travelled = 0.0_f32;
        while travelled < distance {
            let p = origin + dir * travelled;
            self.stamp_wheel(p, 0.38, 0.28, dir, right, sinkage, slip, 1.0 / 60.0, true);
            travelled += step;
        }
        let mid = origin + dir * (distance * 0.5);
        self.rut_near(mid.x, mid.z)
    }

    /// Deepest rut in a small window. A sample that lands on a shoulder
    /// would under-report the floor.
    pub fn rut_near(&self, x: f32, z: f32) -> f32 {
        let mut m = 0.0_f32;
        for dz in [-1, 0, 1] {
            for dx in [-1, 0, 1] {
                let s = self.cell * 0.45;
                m = m.max(self.sample(x + dx as f32 * s, z + dz as f32 * s).rut);
            }
        }
        m
    }

    /// Puddle, a bank that holds it, and higher ground past the bank.
    /// Water cannot leave until a rut cuts the bank.
    pub fn sculpt_banked_puddle(&mut self) {
        let res = self.res;
        let half = self.size * 0.5;
        let cell = self.cell;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                let (h, water) = if x < -1.15 {
                    (0.0, 0.14)
                } else if x < 0.95 {
                    (0.20, 0.0)
                } else {
                    (0.16, 0.0)
                };
                self.height[idx] = h;
                self.initial_height[idx] = h;
                self.water[idx] = water;
                let wet = if water > 0.0 { 1.0 } else { 0.22 };
                self.wetness[idx] = wet;
                self.base_wetness[idx] = wet;
                self.loose[idx] = 0.0;
                self.darken[idx] = 0.05;
                self.roughness[idx] = if water > 0.0 { 0.35 } else { 0.9 };
                self.rest[idx] = PASS_GAP + 1.0;
                self.passes[idx] = 0;
            }
        }
    }

    /// Step the shallow-water pipes for `seconds`, in stable slices.
    pub fn age_water(&mut self, seconds: f32) {
        if seconds <= 0.0 {
            return;
        }
        let steps = (seconds * 60.0).round().max(1.0) as u32;
        let dt = seconds / steps as f32;
        for _ in 0..steps {
            self.step_water(dt);
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

    /// Virtual-pipe shallow water.
    ///
    /// A tire sloshes depth out of its footprint. These sweeps then walk that
    /// water toward the lower free surface, in place, so a film on a sloping
    /// rut drains along the cut and stacks at the low bed instead of sitting
    /// where the tire left it. Each edge pays the receiver from the donor, and
    /// nothing is clamped off, so the volume stays.
    pub fn step_water(&mut self, dt: f32) {
        // The field clock lives here so a tire that has rolled away can be
        // told from one that is still on the same cell.
        for rest in &mut self.rest {
            *rest = (*rest + dt).min(30.0);
        }
        let res = self.res;
        // Half a head would flatten one pair in a single edge. Stay under
        // that so a berm is not jumped. The sweep is in place and flips
        // direction, so water cascades downhill. A Jacobi copy of the same
        // gain cancels in the middle of a uniform slope and the film sits.
        let sweeps = ((dt * 360.0).round() as u32).clamp(2, 8);
        let gain = 0.32_f32;
        for s in 0..sweeps {
            self.sweep_pipes(s % 2 == 1, gain);
        }

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

    fn sweep_pipes(&mut self, flip: bool, gain: f32) {
        let res = self.res;
        for n in 0..res {
            let j = if flip { res - 1 - n } else { n };
            for m in 0..res {
                let i = if flip { res - 1 - m } else { m };
                let a = j * res + i;
                if i + 1 < res {
                    relax(&self.height, &mut self.water, a, a + 1, gain);
                }
                if j + 1 < res {
                    relax(&self.height, &mut self.water, a, a + res, gain);
                }
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

/// Move water from the higher free surface to the lower one.
///
/// `gain` is a fraction of the head and must stay under 0.5. The donor cannot
/// give water it does not have, and the same amount is added next door, so
/// this edge does not create or delete volume.
fn relax(height: &[f32], water: &mut [f32], a: usize, b: usize, gain: f32) {
    let wa = water[a];
    let wb = water[b];
    if wa < 1.0e-6 && wb < 1.0e-6 {
        return;
    }
    let dh = (height[a] + wa) - (height[b] + wb);
    if dh.abs() < 1.0e-4 {
        return;
    }
    let flow = dh.abs() * gain;
    if dh > 0.0 {
        let flow = flow.min(wa);
        water[a] = wa - flow;
        water[b] = wb + flow;
    } else {
        let flow = flow.min(wb);
        water[b] = wb - flow;
        water[a] = wa + flow;
    }
}

/// Rolling depth after `passes` visits. Pass 1 is the fresh bite; each
/// later pass of the same slot is deeper, with a cap so the chassis is not
/// swallowed.
fn pass_groove(one_pass: f32, passes: u8) -> f32 {
    let n = (passes as f32).max(1.0);
    // Pass 1 is the fresh bite. Pass 2 is a little more than twice as deep,
    // so a second visit cuts a bank instead of polishing the same scuff.
    (one_pass * n.powf(1.25)).min(0.30)
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

    fn height_sum(soil: &SoilField) -> f32 {
        soil.height.iter().sum()
    }

    #[test]
    fn second_pass_deepens_the_same_rut() {
        let mut soil = SoilField::flat(0.25);
        let origin = Vec3::new(-2.2, 0.0, 0.0);
        let dir = Vec3::X;
        let before = height_sum(&soil);
        let first = soil.roll_pass(origin, dir, 6.4, 0.09, 0.0);
        let after_first = height_sum(&soil);
        // Saturated albedo. A second pass has nothing left to repaint.
        for d in &mut soil.darken {
            *d = 1.0;
        }
        soil.age_water(0.55);
        let second = soil.roll_pass(origin, dir, 6.4, 0.09, 0.0);
        let after_second = height_sum(&soil);
        let floor = soil.sample(1.0, 0.0);
        let berm = soil.sample(1.0, 0.45).height.max(soil.sample(1.0, -0.45).height);
        eprintln!(
            "pass depths {first:.4} -> {second:.4}  berm {berm:.4} floor {:.4}  volume {before:.4} {after_first:.4} {after_second:.4}",
            floor.height
        );
        assert!(
            first > 0.035,
            "first pass did not cut a rut, depth {first:.4}"
        );
        assert!(
            second > first + 0.015,
            "second pass did not deepen the rut: {first:.4} -> {second:.4}"
        );
        assert!(
            (after_first - before).abs() < 5.0e-3,
            "first pass deleted soil: height sum {before:.5} -> {after_first:.5}"
        );
        assert!(
            (after_second - before).abs() < 8.0e-3,
            "second pass deleted soil: height sum {before:.5} -> {after_second:.5}"
        );
        assert!(
            berm > floor.height + 0.02,
            "displaced soil did not raise the berm: berm {berm:.4} floor {:.4}",
            floor.height
        );
        assert!(
            soil.darken.iter().all(|d| (*d - 1.0).abs() < 1.0e-5 || *d <= 1.0),
            "darken left the 0..1 range"
        );
    }

    #[test]
    fn puddle_flows_into_the_rut_and_stays() {
        let mut soil = SoilField::flat(0.2);
        soil.sculpt_banked_puddle();
        let stored: f32 = soil.water.iter().sum();
        soil.age_water(1.5);
        let dry_before = soil.sample(2.2, 0.0).water;
        assert!(
            dry_before < 0.004,
            "water crossed the bank before any rut, depth {dry_before:.4}"
        );
        let origin = Vec3::new(-2.4, 0.0, 0.0);
        let dir = Vec3::X;
        let first = soil.roll_pass(origin, dir, 6.6, 0.09, 0.0);
        soil.age_water(0.55);
        let second = soil.roll_pass(origin, dir, 6.6, 0.09, 0.0);
        let bank = soil.sample(0.0, 0.0);
        eprintln!(
            "channel passes {first:.4} -> {second:.4}  bank height {:.4} rut {:.4}",
            bank.height, bank.rut
        );
        assert!(
            second > first + 0.012,
            "the rut that cuts the bank did not deepen: {first:.4} -> {second:.4}"
        );
        assert!(
            bank.height < 0.13,
            "rut did not cut the bank below the puddle surface, height {:.4}",
            bank.height
        );
        soil.age_water(4.0);
        let in_rut = soil.sample(2.2, 0.0).water;
        let shoulder = soil.sample(2.2, 1.15).water;
        let held_total: f32 = soil.water.iter().sum();
        eprintln!(
            "rut water {in_rut:.4}  shoulder {shoulder:.4}  volume {stored:.2} -> {held_total:.2}"
        );
        assert!(
            in_rut > 0.025,
            "puddle water did not flow into the rut, depth {in_rut:.4}"
        );
        assert!(
            shoulder < in_rut * 0.5,
            "water left the rut onto the bank: rut {in_rut:.4} shoulder {shoulder:.4}"
        );
        soil.age_water(2.5);
        let still = soil.sample(2.2, 0.0).water;
        let after: f32 = soil.water.iter().sum();
        eprintln!("rut water after waiting {still:.4}  volume {after:.2}");
        assert!(
            still > 0.02,
            "water did not stay in the rut, depth {still:.4}"
        );
        assert!(
            after > stored * 0.92,
            "water was deleted: {stored:.2} -> {after:.2}"
        );
    }

    fn water_sum(soil: &SoilField) -> f32 {
        soil.water.iter().sum()
    }

    /// Grade down toward +x, a tire groove, and a bowl that is the bottom.
    fn lay_sloping_rut(soil: &mut SoilField) {
        let res = soil.res;
        let half = soil.size * 0.5;
        let cell = soil.cell;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let t = ((x + 4.0) / 6.4).clamp(0.0, 1.0);
                let mut h = 0.34 - t * 0.22;
                // Past the bowl the ground rises, so the bowl is the low point.
                if x > 4.15 {
                    h = h.max(0.20);
                }
                let dx = x - 3.15;
                let r = (dx * dx + z * z).sqrt() / 1.15;
                if r < 1.0 {
                    h -= (1.0 - r).powf(1.45) * 0.20;
                }
                let idx = j * res + i;
                soil.height[idx] = h;
                soil.initial_height[idx] = h;
                soil.water[idx] = 0.0;
                soil.wetness[idx] = 0.22;
                soil.base_wetness[idx] = 0.22;
                soil.loose[idx] = 0.0;
                soil.darken[idx] = 0.05;
                soil.roughness[idx] = 0.9;
                soil.rest[idx] = PASS_GAP + 1.0;
                soil.passes[idx] = 0;
            }
        }
        soil.roll_pass(Vec3::new(-3.7, 0.0, 0.0), Vec3::X, 6.3, 0.09, 0.0);
        // The tire crosses the lip. Put the bowl back under the groove so the
        // pool has a bed the rut drains into.
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let dx = x - 3.15;
                let r = (dx * dx + z * z).sqrt() / 1.15;
                if r < 1.0 {
                    let t = ((x + 4.0) / 6.4).clamp(0.0, 1.0);
                    let bed = 0.34 - t * 0.22 - (1.0 - r).powf(1.45) * 0.20;
                    let idx = j * res + i;
                    if soil.height[idx] > bed {
                        soil.height[idx] = bed;
                        soil.initial_height[idx] = bed;
                    }
                }
            }
        }
    }

    #[test]
    fn slosh_drains_downhill_inside_the_rut() {
        let mut soil = SoilField::flat(0.22);
        lay_sloping_rut(&mut soil);
        let res = soil.res;
        let half = soil.size * 0.5;
        let cell = soil.cell;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                let rut = soil.initial_height[idx] - soil.height[idx];
                // Standing water in the upper and middle groove, not in the bowl.
                if rut > 0.025 && (-3.4..1.7).contains(&x) && z.abs() < 0.35 {
                    soil.water[idx] = 0.05;
                    soil.wetness[idx] = 1.0;
                }
            }
        }
        let stored = water_sum(&soil);
        assert!(stored > 1.0, "rut was not filled, sum {stored:.3}");

        // A tire drives downhill through the film and shoves it around.
        let mut x = -3.2_f32;
        while x < 1.5 {
            soil.displace_water(x, 0.0, Vec3::X, 0.025);
            x += cell * 0.7;
        }
        let after_slosh = water_sum(&soil);
        let high_idx = soil.idx_at(-2.6, 0.0).expect("high cell");
        let high_after_slosh = soil.water[high_idx];
        eprintln!(
            "slosh high {high_after_slosh:.4}  volume {stored:.3} -> {after_slosh:.3}"
        );
        assert!(
            high_after_slosh > 0.012,
            "the tire emptied the high end before the water could drain, depth {high_after_slosh:.4}"
        );

        soil.age_water(5.0);
        let mut pool_idx = high_idx;
        let mut pool_bed = f32::MAX;
        let mut high_vol = 0.0_f32;
        let mut low_vol = 0.0_f32;
        let mut side_vol = 0.0_f32;
        for j in 0..res {
            for i in 0..res {
                let x = (i as f32 + 0.5) * cell - half;
                let z = (j as f32 + 0.5) * cell - half;
                let idx = j * res + i;
                let w = soil.water[idx];
                if x < 0.0 && z.abs() < 0.55 {
                    high_vol += w;
                }
                if x > 2.3 && z.abs() < 1.4 {
                    low_vol += w;
                    if soil.height[idx] < pool_bed {
                        pool_bed = soil.height[idx];
                        pool_idx = idx;
                    }
                }
                if z.abs() > 0.55 && x < 2.0 {
                    side_vol += w;
                }
            }
        }
        let high = soil.water[high_idx];
        let pool = soil.water[pool_idx];
        let after = water_sum(&soil);
        let high_bed = soil.height[high_idx];
        let pool_surface = soil.height[pool_idx] + pool;
        eprintln!(
            "drained high {high:.4} (bed {high_bed:.3}) pool {pool:.4} surface {pool_surface:.3}  \
             high_vol {high_vol:.3} low_vol {low_vol:.3} side {side_vol:.3}  volume {after_slosh:.3} -> {after:.3}"
        );
        assert!(
            high < 0.004,
            "high end of the rut did not dry, depth {high:.4}"
        );
        assert!(
            pool > 0.04,
            "low end did not hold the water, depth {pool:.4}"
        );
        assert!(
            high_bed > pool_surface + 0.02,
            "pool climbed the rut: bed {high_bed:.3} surface {pool_surface:.3}"
        );
        assert!(
            high_vol < after * 0.08,
            "water stayed as a sheet up the rut, high sum {high_vol:.3} of {after:.3}"
        );
        assert!(
            low_vol > after * 0.75,
            "water did not pool at the bottom, low sum {low_vol:.3} of {after:.3}"
        );
        assert!(
            side_vol < pool,
            "water left the rut onto the bank: side {side_vol:.3} pool cell {pool:.3}"
        );
        assert!(soil.water.iter().all(|w| *w >= 0.0 && w.is_finite()));
        let drift = (after - after_slosh).abs();
        assert!(
            drift < after_slosh * 0.005 + 1.0e-3,
            "water step did not conserve volume: {after_slosh:.4} -> {after:.4}"
        );
    }

    fn sinkage_on(tm: &Terramech, load: f32) -> f32 {
        let radius = 0.38;
        let width = 0.28;
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

    /// Slip ratio at which Janosi shear matches `drive` (N). 1 means the
    /// tread cannot hook up and the tire spins.
    fn slip_for_drive(tm: &Terramech, sinkage: f32, load: f32, drive: f32) -> f32 {
        let radius = 0.38;
        let width = 0.28;
        let length = contact_length(radius, sinkage.max(0.005)).max(0.05);
        let area = width * length;
        let sigma = load / area.max(0.01);
        let tau_max = tm.cohesion + sigma * tm.phi.tan();
        if tau_max * area <= drive {
            return 1.0;
        }
        let mut lo = 0.0;
        let mut hi = 1.5;
        for _ in 0..22 {
            let mid = 0.5 * (lo + hi);
            let tau = janosi_hanamoto(tm.cohesion, sigma, tm.phi, mid * length, tm.shear_k);
            if tau * area < drive {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }

    #[test]
    fn wet_rut_sinks_and_slips_more_than_dry_ground() {
        let load = 1_600.0_f32;
        // Slow throttle, about 0.42 of the buggy's 500 Nm.
        let drive = 210.0 / 0.38;
        let dry_tm = terramech_at(softness_of(0.08, 0.0, 0.0), 0.0, 0.0);
        let wet_soft = softness_with_rut(1.0, 0.08, 0.07, 0.07);
        let wet_tm = terramech_at(wet_soft, 0.07, 0.07);
        let z_dry = sinkage_on(&dry_tm, load);
        let z_wet = sinkage_on(&wet_tm, load);
        let slip_dry = slip_for_drive(&dry_tm, z_dry, load, drive);
        let slip_wet = slip_for_drive(&wet_tm, z_wet, load, drive);
        eprintln!(
            "dry z {z_dry:.4} slip {slip_dry:.3}   wet rut z {z_wet:.4} slip {slip_wet:.3}  soft {wet_soft:.3}"
        );
        assert!(
            z_wet > z_dry * 1.4,
            "wet rut sinkage {z_wet:.4} is not deeper than dry {z_dry:.4}"
        );
        assert!(
            slip_wet > slip_dry + 0.2,
            "wet rut slip {slip_wet:.3} is not above dry {slip_dry:.3}"
        );
    }
}
