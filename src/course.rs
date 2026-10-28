//! Motocross course for the picture. The oval in `soil::demo` stays so the
//! solver tests keep their lap. Nothing here changes Bekker, stamping, or water.

use std::sync::OnceLock;

use glam::Vec3;

use crate::soil::SoilField;
use crate::world::Input;
#[cfg(test)]
use crate::world::World;

pub const SIZE: f32 = 64.0;
pub const RES: usize = 192;

const OUTER_HALF: f32 = 3.35;
const INNER_HALF: f32 = 2.15;
const INNER_C: (f32, f32) = (0.2, 0.6);

const WHOOP_U0: f32 = 0.655;
const WHOOP_U1: f32 = 0.805;
const SLOPE_U0: f32 = 0.332;
const SLOPE_U1: f32 = 0.390;
const HERO_U: f32 = 0.358;
const HOLE_U: f32 = 0.255;

#[derive(Clone, Copy)]
struct Pt {
    x: f32,
    z: f32,
    s: f32,
    yaw: f32,
    curv: f32,
    rx: f32,
    rz: f32,
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub x: f32,
    pub z: f32,
    pub s: f32,
    pub u: f32,
    pub lat: f32,
    pub yaw: f32,
    pub curv: f32,
}

struct Geom {
    outer: Vec<Pt>,
    inner: Vec<Pt>,
    outer_len: f32,
    inner_len: f32,
    grid: usize,
    cell: f32,
    outer_grid: Vec<u32>,
    inner_grid: Vec<u32>,
}

fn geom() -> &'static Geom {
    static G: OnceLock<Geom> = OnceLock::new();
    G.get_or_init(build_geom)
}

fn build_geom() -> Geom {
    // South start straight, east sweeper, north table, west whoops.
    // Counter-clockwise. Not a circle: the straights are what read from above.
    // Long south straight, east climb, north table, west whoops.
    // Wider than it is tall so the lap is a course, not a coin.
    let outer_pts = [
        (-18.0, -12.6),
        (-6.0, -15.2),
        (8.0, -14.8),
        (18.5, -12.0),
        (23.4, -4.2),
        (21.6, 5.2),
        (14.2, 12.4),
        (2.0, 15.0),
        (-10.4, 14.2),
        (-19.2, 9.2),
        (-23.6, 1.0),
        (-21.0, -7.4),
    ];
    // Clockwise kidney. Positive lateral stays the island.
    let (cx, cz) = INNER_C;
    let inner_pts = [
        (cx + 0.4, cz + 6.0),
        (cx + 4.8, cz + 3.4),
        (cx + 5.8, cz - 0.8),
        (cx + 3.0, cz - 4.6),
        (cx - 0.6, cz - 5.8),
        (cx - 5.0, cz - 2.6),
        (cx - 5.6, cz + 1.6),
        (cx - 2.6, cz + 4.8),
    ];
    let (outer, outer_len) = discretize(&outer_pts, 16);
    let (inner, inner_len) = discretize(&inner_pts, 12);
    let grid = 96;
    let cell = SIZE / grid as f32;
    let half = SIZE * 0.5;
    let mut outer_grid = vec![0u32; grid * grid];
    let mut inner_grid = vec![0u32; grid * grid];
    for j in 0..grid {
        for i in 0..grid {
            let x = (i as f32 + 0.5) * cell - half;
            let z = (j as f32 + 0.5) * cell - half;
            outer_grid[j * grid + i] = nearest(&outer, x, z);
            inner_grid[j * grid + i] = nearest(&inner, x, z);
        }
    }
    Geom {
        outer,
        inner,
        outer_len,
        inner_len,
        grid,
        cell,
        outer_grid,
        inner_grid,
    }
}

fn discretize(pts: &[(f32, f32)], steps: usize) -> (Vec<Pt>, f32) {
    let n = pts.len();
    let mut raw: Vec<(f32, f32, f32, f32)> = Vec::new();
    for i in 0..n {
        let p0 = pts[(i + n - 1) % n];
        let p1 = pts[i];
        let p2 = pts[(i + 1) % n];
        let p3 = pts[(i + 2) % n];
        for s in 0..steps {
            let t = s as f32 / steps as f32;
            let (x, z) = catmull(p0, p1, p2, p3, t);
            let (dx, dz) = catmull_d(p0, p1, p2, p3, t);
            raw.push((x, z, dx, dz));
        }
    }
    let mut arc = vec![0.0f32; raw.len()];
    let mut total = 0.0f32;
    for i in 1..raw.len() {
        let dx = raw[i].0 - raw[i - 1].0;
        let dz = raw[i].1 - raw[i - 1].1;
        total += (dx * dx + dz * dz).sqrt();
        arc[i] = total;
    }
    let close = {
        let dx = raw[0].0 - raw[raw.len() - 1].0;
        let dz = raw[0].1 - raw[raw.len() - 1].1;
        (dx * dx + dz * dz).sqrt()
    };
    total += close;
    let mut out = Vec::with_capacity(raw.len());
    for i in 0..raw.len() {
        let (x, z, dx, dz) = raw[i];
        let _len = (dx * dx + dz * dz).sqrt().max(1e-4);
        let yaw = dx.atan2(dz);
        let prev = if i == 0 { raw.len() - 1 } else { i - 1 };
        let next = (i + 1) % raw.len();
        let y0 = raw[prev].2.atan2(raw[prev].3);
        let y1 = raw[next].2.atan2(raw[next].3);
        let ds = if next > prev {
            (arc[next] - arc[prev]).max(0.05)
        } else {
            (total - arc[prev] + arc[next]).max(0.05)
        };
        let curv = wrap_pi(y1 - y0) / ds;
        out.push(Pt {
            x,
            z,
            s: arc[i],
            yaw,
            curv,
            rx: yaw.cos(),
            rz: -yaw.sin(),
        });
    }
    (out, total)
}

fn catmull(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32), t: f32) -> (f32, f32) {
    let t2 = t * t;
    let t3 = t2 * t;
    let c = |a: f32, b: f32, c: f32, d: f32| {
        0.5 * ((2.0 * b)
            + (-a + c) * t
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3)
    };
    (c(p0.0, p1.0, p2.0, p3.0), c(p0.1, p1.1, p2.1, p3.1))
}

fn catmull_d(p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32), t: f32) -> (f32, f32) {
    let t2 = t * t;
    let c = |a: f32, b: f32, c: f32, d: f32| {
        0.5 * ((-a + c) + 2.0 * (2.0 * a - 5.0 * b + 4.0 * c - d) * t
            + 3.0 * (-a + 3.0 * b - 3.0 * c + d) * t2)
    };
    (c(p0.0, p1.0, p2.0, p3.0), c(p0.1, p1.1, p2.1, p3.1))
}

fn nearest(pts: &[Pt], x: f32, z: f32) -> u32 {
    let mut best = 0u32;
    let mut best_d = f32::MAX;
    for (i, p) in pts.iter().enumerate() {
        let d = (p.x - x).powi(2) + (p.z - z).powi(2);
        if d < best_d {
            best_d = d;
            best = i as u32;
        }
    }
    best
}

fn refine(pts: &[Pt], guess: usize, x: f32, z: f32) -> usize {
    let n = pts.len();
    let mut best = guess % n;
    let mut best_d = f32::MAX;
    for k in 0..24 {
        let i = (guess + n - 12 + k) % n;
        let p = &pts[i];
        let d = (p.x - x).powi(2) + (p.z - z).powi(2);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn hit_of(pts: &[Pt], len: f32, index: usize, x: f32, z: f32) -> Hit {
    let p = &pts[index % pts.len()];
    let dx = x - p.x;
    let dz = z - p.z;
    Hit {
        x: p.x,
        z: p.z,
        s: p.s,
        u: if len > 1e-3 { p.s / len } else { 0.0 },
        lat: dx * p.rx + dz * p.rz,
        yaw: p.yaw,
        curv: p.curv,
    }
}

fn grid_index(g: &Geom, x: f32, z: f32) -> usize {
    let half = SIZE * 0.5;
    let i = ((x + half) / g.cell).floor() as isize;
    let j = ((z + half) / g.cell).floor() as isize;
    let i = i.clamp(0, g.grid as isize - 1) as usize;
    let j = j.clamp(0, g.grid as isize - 1) as usize;
    j * g.grid + i
}

pub fn outer_hit(x: f32, z: f32) -> Hit {
    let g = geom();
    let guess = g.outer_grid[grid_index(g, x, z)] as usize;
    let i = refine(&g.outer, guess, x, z);
    hit_of(&g.outer, g.outer_len, i, x, z)
}

pub fn inner_hit(x: f32, z: f32) -> Hit {
    let g = geom();
    let guess = g.inner_grid[grid_index(g, x, z)] as usize;
    let i = refine(&g.inner, guess, x, z);
    hit_of(&g.inner, g.inner_len, i, x, z)
}

pub fn at_u(u: f32) -> Hit {
    let g = geom();
    let s = u.rem_euclid(1.0) * g.outer_len;
    let mut best = 0usize;
    let mut best_d = f32::MAX;
    for (i, p) in g.outer.iter().enumerate() {
        let d = (p.s - s).abs();
        let d = d.min(g.outer_len - d);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    let p = &g.outer[best];
    Hit {
        x: p.x,
        z: p.z,
        s: p.s,
        u: p.s / g.outer_len,
        lat: 0.0,
        yaw: p.yaw,
        curv: p.curv,
    }
}

pub fn hero() -> (f32, f32, f32) {
    let h = at_u(HERO_U);
    (h.x, h.z, h.yaw)
}

pub fn hole_center() -> (f32, f32) {
    let h = at_u(HOLE_U);
    (h.x, h.z)
}

/// True when a prop would sit off both ribbons.
pub fn clear_of_track(x: f32, z: f32, margin: f32) -> bool {
    outer_hit(x, z).lat.abs() > margin && inner_hit(x, z).lat.abs() > margin
}

/// Island inside the inner loop, or the forest outside the outer berm.
/// The outer lap runs counter-clockwise, so positive lateral is outside.
/// The inner lap runs clockwise, so positive lateral is the island.
pub fn forest_spot(x: f32, z: f32) -> bool {
    let o = outer_hit(x, z);
    let i = inner_hit(x, z);
    let island = i.lat > 3.2;
    let outside = o.lat > 5.3;
    (island || outside) && o.lat.abs() > 4.2 && i.lat.abs() > 2.6
}

pub fn field() -> SoilField {
    static CACHE: OnceLock<SoilField> = OnceLock::new();
    CACHE.get_or_init(build_field).clone()
}

fn build_field() -> SoilField {
    let g = geom();
    let res = RES;
    let n = res * res;
    let cell = SIZE / res as f32;
    let half = SIZE * 0.5;
    let hole = hole_center();
    let pool = at_u(0.335);
    let mut field = SoilField {
        res,
        size: SIZE,
        cell,
        height: vec![0.0; n],
        initial_height: vec![0.0; n],
        loose: vec![0.0; n],
        wetness: vec![0.0; n],
        base_wetness: vec![0.0; n],
        water: vec![0.0; n],
        darken: vec![0.0; n],
        roughness: vec![0.0; n],
    };
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5) * cell - half;
            let z = (j as f32 + 0.5) * cell - half;
            let o = {
                let guess = g.outer_grid[grid_index(g, x, z)] as usize;
                let idx = refine(&g.outer, guess, x, z);
                hit_of(&g.outer, g.outer_len, idx, x, z)
            };
            let inn = {
                let guess = g.inner_grid[grid_index(g, x, z)] as usize;
                let idx = refine(&g.inner, guess, x, z);
                hit_of(&g.inner, g.inner_len, idx, x, z)
            };
            let mut h = land(x, z);
            let outer_w = smooth(OUTER_HALF + 2.5, OUTER_HALF - 0.25, o.lat.abs());
            let inner_w = smooth(INNER_HALF + 2.1, INNER_HALF - 0.2, inn.lat.abs());
            let track = outer_surface(o, pool);
            // Outside the ribbon, ease the hillside down from the racing line
            // so the climb is a slope, not a curb around a floating oval.
            if o.lat > 0.6 {
                let fade = smooth(OUTER_HALF + 8.0, OUTER_HALF + 0.4, o.lat);
                h = lerp(h, track.max(h), fade);
            }
            if outer_w >= inner_w && outer_w > 0.0 {
                h = lerp(h, track, outer_w);
                h += outer_berm(o);
            } else if inner_w > 0.0 {
                h = lerp(h, inner_surface(inn), inner_w);
                h += inner_berm(inn);
            }
            // The bowl is cut in world space so the bank rings the hole.
            // A ribbon-only dip read as a flat quad.
            h -= bowl_drop(x, z, hole);
            h = h.clamp(-0.95, 2.25);
            let idx = j * res + i;
            field.height[idx] = h;
            field.initial_height[idx] = h;
        }
    }
    for j in 0..res {
        for i in 0..res {
            let x = (i as f32 + 0.5) * cell - half;
            let z = (j as f32 + 0.5) * cell - half;
            let idx = j * res + i;
            let h = field.height[idx];
            let kind = kind_at(x, z, hole, pool);
            let (wet, loose, dark, water) = match kind {
                Kind::Grass => (0.16, 0.0, 0.05, 0.0),
                Kind::Gravel => (0.07, 0.0, 0.08, 0.0),
                Kind::Dry => (0.22, 0.045, 0.16, 0.0),
                Kind::Mud => (0.94, 0.04, 0.40, 0.0),
                Kind::Clay => (0.86, 0.012, 0.58, 0.0),
                Kind::Rock => (0.06, 0.0, 0.04, 0.0),
            };
            let mut water = water;
            let mut wet = wet;
            let drop = bowl_drop(x, z, hole);
            if drop > 0.02 {
                // Shallow at the bank, deeper in the middle. The rim is the shoreline.
                water = (drop * 0.22).clamp(0.012, 0.20);
                wet = 1.0;
            }
            let pool_d = (x - pool.x).hypot(z - pool.z);
            if pool_d < 2.15 && h < 0.08 {
                water = water.max((0.02 - h).clamp(0.03, 0.12));
                wet = 1.0;
            }
            field.water[idx] = water;
            field.wetness[idx] = wet;
            field.base_wetness[idx] = if water > 0.0 { 0.95 } else { wet.min(0.8) };
            field.loose[idx] = loose;
            field.darken[idx] = dark;
            field.roughness[idx] = lerp(0.92, 0.36, wet).clamp(0.18, 1.0);
        }
    }
    field
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Grass,
    Gravel,
    Dry,
    Mud,
    Clay,
    Rock,
}

fn kind_at(x: f32, z: f32, hole: (f32, f32), pool: Hit) -> Kind {
    let o = outer_hit(x, z);
    let i = inner_hit(x, z);
    let on_outer = o.lat.abs() < OUTER_HALF + 0.35;
    let on_inner = i.lat.abs() < INNER_HALF + 0.25;
    let berm_o = o.lat.abs() > OUTER_HALF - 0.15 && o.lat.abs() < OUTER_HALF + 2.3;
    let berm_i = i.lat.abs() > INNER_HALF - 0.1 && i.lat.abs() < INNER_HALF + 1.8;
    let n = hash2(x * 0.17, z * 0.17);
    let far = x.abs().max(z.abs());
    if far > 29.5 && n > 0.45 {
        return Kind::Rock;
    }
    if on_outer && rut_depth(o) > 0.12 && groove_mask(o.lat) > 0.45 {
        return Kind::Clay;
    }
    if on_inner && groove_mask(i.lat) > 0.55 && (i.u - 0.5).abs() < 0.12 {
        return Kind::Clay;
    }
    let hole_d = (x - hole.0).hypot(z - hole.1);
    let pool_d = (x - pool.x).hypot(z - pool.z);
    if hole_d < 3.4 || pool_d < 2.3 {
        return Kind::Mud;
    }
    if on_outer && o.u < 0.11 || on_outer && o.u > 0.93 {
        return if n > 0.72 { Kind::Dry } else { Kind::Gravel };
    }
    if on_outer && o.u > 0.50 && o.u < 0.61 {
        return if n > 0.55 { Kind::Gravel } else { Kind::Dry };
    }
    // The lane is dry dirt. Mud stays in the hole and the slope puddle,
    // clay stays in the ruts. A full mud ribbon reads as a painted ring.
    if on_outer && o.u > WHOOP_U0 && o.u < WHOOP_U1 {
        return if groove_mask(o.lat) > 0.5 {
            Kind::Clay
        } else if n > 0.62 {
            Kind::Mud
        } else {
            Kind::Dry
        };
    }
    if on_outer || on_inner {
        return if n > 0.78 { Kind::Mud } else { Kind::Dry };
    }
    if berm_o || berm_i {
        return Kind::Dry;
    }
    if n > 0.82 && far > 18.0 {
        return Kind::Rock;
    }
    Kind::Grass
}

/// mud, dry, clay, grass, rock. Matches the five ground packs.
pub fn blend(x: f32, z: f32) -> [f32; 5] {
    let hole = hole_center();
    let pool = at_u(0.335);
    let kind = kind_at(x, z, hole, pool);
    let n = hash2(x * 0.23 + 2.0, z * 0.23);
    let (mud, dry, clay, grass, rock) = match kind {
        Kind::Grass => (0.02, 0.08 + n * 0.06, 0.0, 0.90 - n * 0.06, 0.0),
        Kind::Gravel => (0.04, 0.12, 0.0, 0.02, 0.82),
        Kind::Dry => (0.08, 0.72, 0.14, 0.06, 0.0),
        Kind::Mud => (0.78, 0.08, 0.14, 0.0, 0.0),
        Kind::Clay => (0.22, 0.06, 0.72, 0.0, 0.0),
        Kind::Rock => (0.0, 0.08, 0.0, 0.06, 0.86),
    };
    let sum = (mud + dry + clay + grass + rock).max(1e-4);
    [mud / sum, dry / sum, clay / sum, grass / sum, rock / sum]
}

/// Round pit. Zero on the rim, deepest at the hole center.
fn bowl_drop(x: f32, z: f32, hole: (f32, f32)) -> f32 {
    let dx = (x - hole.0) / 3.85;
    let dz = (z - hole.1) / 3.45;
    let d = (dx * dx + dz * dz).sqrt();
    if d >= 1.0 {
        return 0.0;
    }
    let t = 1.0 - d;
    t * t * (3.0 - 2.0 * t) * 0.86
}

fn land(x: f32, z: f32) -> f32 {
    let n = fbm(x * 0.09, z * 0.09);
    let n2 = fbm(x * 0.31 + 4.0, z * 0.28);
    // The lap is low in the south and high in the north. The hill matches
    // that so the ribbon is cut into the slope instead of riding a curb.
    let hill = smooth(-14.0, 12.0, z * 0.9 + x * 0.12);
    0.20 + (n - 0.5) * 0.12 + (n2 - 0.5) * 0.04 + hill * 1.9
}

fn outer_surface(o: Hit, pool: Hit) -> f32 {
    let mut h = elev(o.u);
    h += whoops(o);
    h += table_bump(o.u);
    let pd = (o.x - pool.x).hypot(o.z - pool.z);
    if pd < 2.4 {
        let t = (1.0 - pd / 2.4).clamp(0.0, 1.0);
        h -= t * t * 0.22;
    }
    let depth = rut_depth(o);
    if depth > 0.0 {
        let side = (1.0 - (o.lat.abs() / (OUTER_HALF - 0.15)).clamp(0.0, 1.0)).powf(0.8);
        h += rut_delta(o.lat) * depth * side;
    }
    h
}

fn elev(u: f32) -> f32 {
    const K: &[(f32, f32)] = &[
        (0.00, 0.40),
        (0.10, 0.34),
        (0.20, 0.28),
        (0.27, 0.16),
        (0.332, 0.22),
        (0.385, 1.95),
        (0.43, 1.70),
        (0.50, 1.62),
        (0.62, 1.52),
        (0.66, 1.12),
        (0.80, 0.52),
        (0.90, 0.42),
        (1.00, 0.40),
    ];
    let u = u.rem_euclid(1.0);
    for i in 0..K.len() - 1 {
        if u >= K[i].0 && u <= K[i + 1].0 {
            let t = (u - K[i].0) / (K[i + 1].0 - K[i].0);
            let s = t * t * (3.0 - 2.0 * t);
            return K[i].1 + (K[i + 1].1 - K[i].1) * s;
        }
    }
    K[0].1
}

fn whoops(o: Hit) -> f32 {
    if o.u < WHOOP_U0 || o.u > WHOOP_U1 {
        return 0.0;
    }
    let edge = smooth(WHOOP_U0, WHOOP_U0 + 0.018, o.u) * smooth(WHOOP_U1, WHOOP_U1 - 0.018, o.u);
    let side = 1.0 - (o.lat.abs() / OUTER_HALF).clamp(0.0, 1.0);
    let phase = (o.u - WHOOP_U0) * geom().outer_len / 3.65;
    let hump = 0.5 - 0.5 * (phase * std::f32::consts::TAU).cos();
    hump * 0.95 * edge * side.powf(0.7)
}

fn table_bump(u: f32) -> f32 {
    let u0 = 0.498;
    let u1 = 0.612;
    let t = (u - u0) / (u1 - u0);
    if !(0.0..1.0).contains(&t) {
        return 0.0;
    }
    let shape = if t < 0.24 {
        smooth(0.0, 0.24, t)
    } else if t > 0.74 {
        smooth(1.0, 0.74, t)
    } else {
        1.0
    };
    shape * 0.62
}

fn rut_depth(o: Hit) -> f32 {
    if o.u > SLOPE_U0 && o.u < SLOPE_U1 {
        return 0.26;
    }
    if o.u > 0.50 && o.u < 0.61 {
        return 0.0;
    }
    if o.u > WHOOP_U0 && o.u < WHOOP_U1 {
        return 0.05;
    }
    if o.u > 0.22 && o.u < 0.30 {
        return 0.04;
    }
    0.09
}

fn groove_mask(lat: f32) -> f32 {
    let g = |c: f32| {
        let d = (lat - c).abs();
        smooth(0.34, 0.05, d)
    };
    g(-0.74).max(g(0.74))
}

fn rut_delta(lat: f32) -> f32 {
    let groove = |c: f32| {
        let d = (lat - c).abs();
        let t = (1.0 - d / 0.36).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let ridge = |c: f32, w: f32| {
        let d = (lat - c).abs();
        let t = (1.0 - d / w).clamp(0.0, 1.0);
        t * t
    };
    let cut = groove(-0.74) + groove(0.74);
    let pile = ridge(-1.16, 0.34) + ridge(1.16, 0.34) + ridge(0.0, 0.24) * 0.4;
    pile * 0.5 - cut
}

fn outer_berm(o: Hit) -> f32 {
    let band = smooth(OUTER_HALF - 0.2, OUTER_HALF + 0.45, o.lat.abs())
        * smooth(OUTER_HALF + 2.4, OUTER_HALF + 0.7, o.lat.abs());
    // Only the outside of a real corner. A berm on every metre reads as a moat.
    let turn = (o.curv.abs() - 0.04).max(0.0).min(0.12);
    let side = if o.lat > 0.3 { 1.0 } else { 0.1 };
    band * side * (0.04 + turn * 7.5)
}

fn inner_surface(i: Hit) -> f32 {
    // Sit on the hillside. A fixed height dug a crater into the middle of the lap.
    let mut h = land(i.x, i.z) + hash2(i.x * 0.2, i.z * 0.2) * 0.03;
    let u = i.u;
    // Small table on the north side of the inner loop.
    let t = (u - 0.02) / 0.16;
    if (0.0..1.0).contains(&t) {
        let shape = if t < 0.28 {
            smooth(0.0, 0.28, t)
        } else if t > 0.72 {
            smooth(1.0, 0.72, t)
        } else {
            1.0
        };
        let side = 1.0 - (i.lat.abs() / INNER_HALF).clamp(0.0, 1.0);
        h += shape * 0.42 * side;
    }
    if (u - 0.50).abs() < 0.12 {
        let tt = 1.0 - (u - 0.50).abs() / 0.12;
        let side = 1.0 - (i.lat.abs() / INNER_HALF).clamp(0.0, 1.0);
        h -= tt * tt * 0.18 * side;
    }
    if (u - 0.50).abs() > 0.14 {
        let side = (1.0 - (i.lat.abs() / INNER_HALF).clamp(0.0, 1.0)).powf(0.85);
        h += rut_delta(i.lat) * 0.02 * side;
    }
    h
}

fn inner_berm(i: Hit) -> f32 {
    let band = smooth(INNER_HALF - 0.15, INNER_HALF + 0.35, i.lat.abs())
        * smooth(INNER_HALF + 1.6, INNER_HALF + 0.5, i.lat.abs());
    let turn = (i.curv.abs() - 0.06).max(0.0).min(0.16);
    let side = if i.lat >= 0.0 { 1.0 } else { 0.15 };
    band * side * (0.02 + turn * 2.2)
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn wrap_pi(mut a: f32) -> f32 {
    let pi = std::f32::consts::PI;
    while a > pi {
        a -= pi * 2.0;
    }
    while a < -pi {
        a += pi * 2.0;
    }
    a
}

fn hash2(x: f32, y: f32) -> f32 {
    let s = (x * 127.1 + y * 311.7).sin() * 43758.5453;
    s - s.floor()
}

fn noise2(x: f32, y: f32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash2(x0, y0);
    let b = hash2(x0 + 1.0, y0);
    let c = hash2(x0, y0 + 1.0);
    let d = hash2(x0 + 1.0, y0 + 1.0);
    a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy
}

fn fbm(x: f32, y: f32) -> f32 {
    let mut a = 0.5;
    let mut f = 1.0;
    let mut s = 0.0;
    let mut n = 0.0;
    for _ in 0..4 {
        s += a * noise2(x * f, y * f);
        n += a;
        a *= 0.5;
        f *= 2.05;
    }
    s / n
}

/// Pure pursuit on the outer loop, for the window. Stills do not use it.
pub fn drive(pos: Vec3, yaw: f32, speed: f32) -> Input {
    let hit = outer_hit(pos.x, pos.z);
    let ahead = at_u(hit.u + 6.0 / geom().outer_len.max(1.0));
    let desired = (ahead.x - pos.x).atan2(ahead.z - pos.z);
    let mut err = wrap_pi(desired - yaw);
    err -= (hit.lat * 0.18).clamp(-0.45, 0.45);
    let steer = (err / 0.38).clamp(-1.0, 1.0);
    let throttle = if speed > 5.5 { 0.2 } else { 0.65 };
    Input {
        throttle,
        brake: 0.0,
        steer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn course_has_slope_ruts_whoops_hole_and_table() {
        let soil = field();
        let min = soil.height.iter().copied().fold(f32::MAX, f32::min);
        let max = soil.height.iter().copied().fold(0.0_f32, f32::max);
        let water: f32 = soil.water.iter().sum();
        assert!(max - min > 1.35, "relief {min:.2}..{max:.2}");
        assert!(water > 6.0, "standing water {water:.2}");

        let (hx, hz, yaw) = hero();
        let right = (yaw.cos(), -yaw.sin());
        let c = soil.sample(hx, hz).height;
        let l = soil
            .sample(hx - right.0 * 0.74, hz - right.1 * 0.74)
            .height;
        let r = soil
            .sample(hx + right.0 * 0.74, hz + right.1 * 0.74)
            .height;
        assert!(
            c > l + 0.06 && c > r + 0.06,
            "slope ruts center {c:.3} wheels {l:.3} {r:.3}"
        );

        let mut prev = soil.sample(at_u(WHOOP_U0).x, at_u(WHOOP_U0).z).height;
        let mut rises = 0;
        let mut was_up = false;
        for k in 1..28 {
            let u = WHOOP_U0 + (WHOOP_U1 - WHOOP_U0) * (k as f32 / 28.0);
            let p = at_u(u);
            let h = soil.sample(p.x, p.z).height;
            let up = h > prev + 0.02;
            if up && !was_up {
                rises += 1;
            }
            was_up = up;
            prev = h;
        }
        assert!(rises >= 3, "whoops rises {rises}");

        let flat_a = soil.sample(at_u(0.54).x, at_u(0.54).z).height;
        let flat_b = soil.sample(at_u(0.57).x, at_u(0.57).z).height;
        let toe = soil.sample(at_u(0.49).x, at_u(0.49).z).height;
        assert!(
            (flat_a - flat_b).abs() < 0.12 && flat_a > toe + 0.25,
            "table {flat_a:.2} {flat_b:.2} toe {toe:.2}"
        );

        let (bx, bz) = hole_center();
        let bed = soil.sample(bx, bz);
        assert!(bed.water > 0.04, "hole water {}", bed.water);
        assert!(bed.height < toe, "hole should sit below the table");

        let o = outer_hit(0.2, 0.6);
        let i = inner_hit(0.2, 0.6);
        assert!(o.lat < -8.0, "infield inside the outer loop, lat {}", o.lat);
        assert!(i.lat > 3.0, "inner island signed inside, lat {}", i.lat);
    }

    #[test]
    fn parked_niva_stays_in_the_rut() {
        let (x, z, _) = hero();
        let world = World::course();
        let dx = world.vehicle.pos.x - x;
        let dz = world.vehicle.pos.z - z;
        let d = (dx * dx + dz * dz).sqrt();
        assert!(d < 1.6, "Niva slid {d:.2} m to {:?}", world.vehicle.pos);
        assert!(world.bike.pos.x > 100.0, "bike should be off the course");
    }
}
