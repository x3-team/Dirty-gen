//! Pebbles and dried clods scattered on the mud.
//!
//! This only reads the heightfield. Placement is denser on the outside berm and
//! on the bank of a puddle, and thin where water is standing. The mesh is built
//! with y = 0 on the surface: the draw path adds the live height so a piece
//! stays half sunk when a rut moves the bed.

use glam::{EulerRot, Quat, Vec3};

use crate::soil::{track_offset, SoilField, TRACK_ZN};

/// Grey stone. The mesh shader, not the ground lighting.
pub const PANEL_PEBBLE: f32 = 47.0;
/// Dried mud crust. Distinct from the wet splash clod on panel 46.
pub const PANEL_HARD_CLOD: f32 = 48.0;

const STEP: f32 = 0.48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Band {
    Open = 0,
    Berm = 1,
    Shore = 2,
    Water = 3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Pebble,
    Clod,
}

#[derive(Clone, Copy, Debug)]
pub struct Piece {
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub radius: f32,
    pub height: f32,
    /// Fraction of `height` below the surface. The rest stands proud.
    pub sink: f32,
    pub kind: Kind,
    pub seed: u32,
}

impl Piece {
    pub fn bottom(&self) -> f32 {
        -self.height * self.sink
    }

    pub fn top(&self) -> f32 {
        self.height * (1.0 - self.sink)
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DressingVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    /// Anchor xz. The mesh shader tilts the normal from the slope sampled here.
    pub uv: [f32; 2],
    pub panel: f32,
}

pub struct DressingMesh {
    pub vertices: Vec<DressingVertex>,
    pub indices: Vec<u32>,
}

pub fn scatter(soil: &SoilField) -> Vec<Piece> {
    let mut out = Vec::new();
    for_each_cell(soil, |id, x, z| {
        let here = band(soil, x, z);
        let gate = hash01(id.wrapping_mul(3) + 3);
        if gate < weight(here) {
            out.push(piece_at(x, z, id, here));
        }
    });
    out
}

pub fn dressing_mesh(soil: &SoilField) -> DressingMesh {
    let mut mesh = DressingMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for piece in scatter(soil) {
        add_piece(&mut mesh, &piece);
    }
    mesh
}

/// Outside bank of the west puddle, for a close still.
pub fn west_puddle_shore(soil: &SoilField) -> (f32, f32) {
    let cx = -6.15;
    let cz = 5.05;
    let dx = cx;
    let dz = cz - TRACK_ZN;
    let len = (dx * dx + dz * dz).sqrt().max(0.01);
    let dir_x = dx / len;
    let dir_z = dz / len;
    let mut run = Vec::new();
    for i in 0..64 {
        let t = i as f32 * 0.07;
        let x = cx + dir_x * t;
        let z = cz + dir_z * t;
        if band(soil, x, z) == Band::Shore {
            run.push((x, z));
        } else if !run.is_empty() {
            break;
        }
    }
    if run.is_empty() {
        return (cx + dir_x * 2.4, cz + dir_z * 2.4);
    }
    // Middle of the bank, so a still can show water on one side and mud on the other.
    run[run.len() / 2]
}

/// Crest of the outside berm on the west straight.
pub fn west_berm(soil: &SoilField) -> (f32, f32) {
    let mut best = (-8.0_f32, 0.85_f32);
    let mut best_h = f32::MIN;
    for i in 0..30 {
        let x = -7.05 - i as f32 * 0.07;
        let z = 0.85;
        if berm_weight(track_offset(x, z)) < 0.65 {
            continue;
        }
        let sample = soil.sample(x, z);
        if sample.water < 0.01 && sample.height > best_h {
            best_h = sample.height;
            best = (x, z);
        }
    }
    best
}

fn for_each_cell(soil: &SoilField, mut visit: impl FnMut(u32, f32, f32)) {
    let span = soil.size * 0.5 - 1.2;
    let limit = soil.size * 0.5 - 0.45;
    let mut z = -span;
    let mut row = 0u32;
    while z < span {
        let mut x = -span;
        let mut col = 0u32;
        while x < span {
            let id = row.wrapping_mul(1024) + col + 1;
            let jx = (hash01(id.wrapping_mul(3) + 1) - 0.5) * STEP * 0.8;
            let jz = (hash01(id.wrapping_mul(3) + 2) - 0.5) * STEP * 0.8;
            let px = (x + jx).clamp(-limit, limit);
            let pz = (z + jz).clamp(-limit, limit);
            visit(id, px, pz);
            x += STEP;
            col += 1;
        }
        z += STEP;
        row += 1;
    }
}

fn weight(band: Band) -> f32 {
    match band {
        // A few pieces on open mud so the field is not one brown sheet.
        Band::Open => 0.07,
        Band::Berm => 0.72,
        Band::Shore => 0.76,
        // Standing water keeps a rare pebble. The bank carries the rest.
        Band::Water => 0.025,
    }
}

fn band(soil: &SoilField, x: f32, z: f32) -> Band {
    let water = soil.sample(x, z).water;
    if water > 0.055 {
        return Band::Water;
    }
    if shore_weight(soil, x, z) > 0.48 {
        return Band::Shore;
    }
    if berm_weight(track_offset(x, z)) > 0.48 {
        return Band::Berm;
    }
    Band::Open
}

/// Outside shoulder of the stadium. The infield side of the line is not a berm.
fn berm_weight(off: f32) -> f32 {
    if off <= 0.0 {
        return 0.0;
    }
    let rise = smooth(0.55, 1.25, off);
    let fall = 1.0 - smooth(2.05, 3.2, off);
    rise * fall
}

fn shore_weight(soil: &SoilField, x: f32, z: f32) -> f32 {
    let here = soil.sample(x, z).water;
    if here > 0.08 {
        return 0.0;
    }
    let mut peak = 0.0_f32;
    let mut nearest = 10.0_f32;
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let (s, c) = a.sin_cos();
        for r in [0.42_f32, 0.95] {
            let w = soil.sample(x + c * r, z + s * r).water;
            if w > 0.018 {
                peak = peak.max(w);
                nearest = nearest.min(r);
            }
        }
    }
    if peak < 0.015 {
        return 0.0;
    }
    let film = smooth(0.0, 0.01, here) * (1.0 - smooth(0.03, 0.07, here));
    let bank = if here < 0.028 {
        smooth(1.05, 0.32, nearest) * smooth(0.015, 0.05, peak)
    } else {
        0.0
    };
    film.max(bank).clamp(0.0, 1.0)
}

fn piece_at(x: f32, z: f32, id: u32, here: Band) -> Piece {
    let kind_roll = hash01(id.wrapping_mul(5) + 11);
    let clod_cut = match here {
        // Dried crust piles on the shoulder. Water mostly leaves pebbles.
        Band::Berm => 0.66,
        Band::Shore => 0.50,
        Band::Water => 0.18,
        Band::Open => 0.42,
    };
    let kind = if kind_roll < clod_cut {
        Kind::Clod
    } else {
        Kind::Pebble
    };
    let u = hash01(id.wrapping_mul(5) + 12);
    let (radius, raw_h) = match kind {
        Kind::Pebble => {
            let r = if u < 0.72 {
                0.05 + u * 0.08
            } else {
                0.10 + u * 0.07
            };
            let h = r * (0.82 + hash01(id.wrapping_mul(5) + 13) * 0.4);
            (r, h)
        }
        Kind::Clod => {
            let r = if u < 0.65 {
                0.09 + u * 0.10
            } else {
                0.15 + u * 0.12
            };
            let h = r * (0.40 + hash01(id.wrapping_mul(5) + 13) * 0.26);
            (r, h)
        }
    };
    // Past halfway, so the mud line cuts the body and not just the bottom pole.
    let sink = 0.52 + hash01(id.wrapping_mul(5) + 14) * 0.14;
    // Proud and buried both have to read. Scaling height keeps the sink fraction.
    let mut height = raw_h;
    if height * (1.0 - sink) < 0.03 {
        height = 0.03 / (1.0 - sink);
    }
    if height * sink < 0.022 {
        height = 0.022 / sink;
    }
    Piece {
        x,
        z,
        yaw: hash01(id.wrapping_mul(5) + 15) * std::f32::consts::TAU,
        // A small tip. A hard roll stands the piece on a corner and it reads as a shard.
        pitch: (hash01(id.wrapping_mul(5) + 16) - 0.5) * 0.16,
        roll: (hash01(id.wrapping_mul(5) + 17) - 0.5) * 0.16,
        radius,
        height,
        sink,
        kind,
        seed: id,
    }
}

fn smooth(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash01(n: u32) -> f32 {
    let mut x = n.wrapping_mul(0x9E37_79B9) ^ 0xA5A5_5A5A;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x & 0x00FF_FFFF) as f32 / 16_777_215.0
}

fn add_piece(mesh: &mut DressingMesh, piece: &Piece) {
    match piece.kind {
        Kind::Pebble => add_body(mesh, piece, Vec3::ZERO, 1.0, 1.0, 8, 1),
        Kind::Clod => {
            add_body(mesh, piece, Vec3::ZERO, 1.0, 1.0, 8, 1);
            let ang = hash01(piece.seed.wrapping_mul(9) + 4) * std::f32::consts::TAU;
            let dist = piece.radius * (0.22 + hash01(piece.seed.wrapping_mul(9) + 5) * 0.16);
            add_body(
                mesh,
                piece,
                Vec3::new(ang.cos() * dist, 0.0, ang.sin() * dist),
                0.72,
                0.66,
                7,
                2,
            );
        }
    }
}

fn add_body(
    mesh: &mut DressingMesh,
    piece: &Piece,
    offset: Vec3,
    radius_scale: f32,
    height_scale: f32,
    sides: usize,
    salt: u32,
) {
    let q = Quat::from_euler(EulerRot::YXZ, piece.yaw, piece.pitch, piece.roll);
    let place = |p: Vec3| {
        let r = q * (p + offset);
        Vec3::new(piece.x + r.x, r.y, piece.z + r.z)
    };
    let h = piece.height * height_scale;
    let y0 = -h * piece.sink;
    let y1 = h * (1.0 - piece.sink);
    let radius = piece.radius * radius_scale;
    let center = place(Vec3::new(0.0, (y0 + y1) * 0.5, 0.0));
    let panel = match piece.kind {
        Kind::Pebble => PANEL_PEBBLE,
        Kind::Clod => PANEL_HARD_CLOD,
    };
    let uv = [piece.x, piece.z];
    let seed = piece.seed ^ salt.wrapping_mul(0x9E37);

    let (ring_y, ring_r) = match piece.kind {
        // Bulge just under the mud line, crown above it.
        Kind::Pebble => ([0.30_f32, 0.58], [1.0_f32, 0.55]),
        Kind::Clod => ([0.26, 0.50], [1.15, 0.72]),
    };

    let mut lower = Vec::with_capacity(sides);
    let mut upper = Vec::with_capacity(sides);
    for (ri, ring) in [&mut lower, &mut upper].into_iter().enumerate() {
        let y = y0 + (y1 - y0) * ring_y[ri];
        let rad = radius * ring_r[ri];
        for i in 0..sides {
            // Rings share an angle. A half-step twist made long spikes.
            let a = i as f32 / sides as f32 * std::f32::consts::TAU;
            let j = 0.90 + hash01(seed.wrapping_add((ri * 16 + i) as u32)) * 0.2;
            let squash = 0.88 + hash01(seed.wrapping_add(90 + i as u32)) * 0.22;
            let lift = (hash01(seed.wrapping_add(140 + (ri * 8 + i) as u32)) - 0.5) * h * 0.06;
            ring.push(place(Vec3::new(
                a.cos() * rad * j,
                y + lift,
                a.sin() * rad * j * squash,
            )));
        }
    }
    let bot = place(Vec3::new(
        (hash01(seed.wrapping_add(200)) - 0.5) * radius * 0.16,
        y0,
        (hash01(seed.wrapping_add(201)) - 0.5) * radius * 0.16,
    ));
    let top = place(Vec3::new(
        (hash01(seed.wrapping_add(202)) - 0.5) * radius * 0.22,
        y1,
        (hash01(seed.wrapping_add(203)) - 0.5) * radius * 0.22,
    ));
    for i in 0..sides {
        let i1 = (i + 1) % sides;
        add_outward(mesh, bot, lower[i], lower[i1], center, panel, uv);
        add_outward(mesh, lower[i], upper[i], lower[i1], center, panel, uv);
        add_outward(mesh, lower[i1], upper[i], upper[i1], center, panel, uv);
        add_outward(mesh, upper[i], top, upper[i1], center, panel, uv);
    }
}

fn add_outward(
    mesh: &mut DressingMesh,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    center: Vec3,
    panel: f32,
    uv: [f32; 2],
) {
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1.0e-10 {
        return;
    }
    let mid = (a + b + c) * (1.0 / 3.0);
    let (b, c) = if n.dot(mid - center) < 0.0 {
        (c, b)
    } else {
        (b, c)
    };
    let n = (b - a).cross(c - a).normalize_or_zero();
    if n.length_squared() < 0.5 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    for p in [a, b, c] {
        mesh.vertices.push(DressingVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv,
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn berm_and_shore_are_dressed_more_than_open_mud_or_water() {
        let soil = SoilField::demo();
        let mut cells = [0u32; 4];
        let mut got = [0u32; 4];
        for_each_cell(&soil, |id, x, z| {
            let b = band(&soil, x, z) as usize;
            cells[b] += 1;
            let gate = hash01(id.wrapping_mul(3) + 3);
            if gate < weight(band(&soil, x, z)) {
                got[b] += 1;
            }
        });
        let dens = |i: usize| got[i] as f32 / cells[i].max(1) as f32;
        let open = dens(Band::Open as usize);
        let berm = dens(Band::Berm as usize);
        let shore = dens(Band::Shore as usize);
        let water = dens(Band::Water as usize);
        eprintln!(
            "cells open {} berm {} shore {} water {}  dens {:.3} {:.3} {:.3} {:.3}",
            cells[0], cells[1], cells[2], cells[3], open, berm, shore, water
        );
        assert!(cells[Band::Berm as usize] > 80, "berm cells");
        assert!(cells[Band::Shore as usize] > 40, "shore cells");
        assert!(cells[Band::Water as usize] > 30, "water cells");
        assert!(berm > open * 4.0, "berm {berm} open {open}");
        assert!(shore > open * 4.0, "shore {shore} open {open}");
        assert!(shore > water * 5.0, "shore {shore} water {water}");
        assert!(berm > 0.4, "berm {berm}");
        assert!(shore > 0.4, "shore {shore}");
        assert!(water < 0.12, "water {water}");
        assert!(open < 0.16, "open {open}");

        let items = scatter(&soil);
        assert_eq!(items.len(), got.iter().sum::<u32>() as usize);
        for piece in &items {
            assert!(piece.x.abs() < soil.size * 0.5);
            assert!(piece.z.abs() < soil.size * 0.5);
            assert!((0.50..0.68).contains(&piece.sink), "sink {}", piece.sink);
            assert!(piece.bottom() < -0.018, "bottom {}", piece.bottom());
            assert!(piece.top() > 0.025, "top {}", piece.top());
        }
    }

    #[test]
    fn berms_carry_clods_and_water_carries_pebbles() {
        let soil = SoilField::demo();
        let items = scatter(&soil);
        let mut berm_n = 0u32;
        let mut berm_clod = 0u32;
        let mut water_n = 0u32;
        let mut water_pebble = 0u32;
        for piece in &items {
            match band(&soil, piece.x, piece.z) {
                Band::Berm => {
                    berm_n += 1;
                    if piece.kind == Kind::Clod {
                        berm_clod += 1;
                    }
                }
                Band::Water => {
                    water_n += 1;
                    if piece.kind == Kind::Pebble {
                        water_pebble += 1;
                    }
                }
                _ => {}
            }
        }
        assert!(berm_n > 40, "berm pieces {berm_n}");
        assert!(berm_clod * 2 > berm_n, "clods {berm_clod} of {berm_n}");
        if water_n >= 8 {
            assert!(
                water_pebble * 2 > water_n,
                "pebbles {water_pebble} of {water_n}"
            );
        }
    }

    #[test]
    fn anchors_sit_on_the_bands_they_name() {
        let soil = SoilField::demo();
        let (sx, sz) = west_puddle_shore(&soil);
        assert_eq!(band(&soil, sx, sz), Band::Shore, "shore at {sx},{sz}");
        let deep = soil.sample(-6.15, 5.05).water;
        let edge = soil.sample(sx, sz).water;
        assert!(deep > edge + 0.04, "deep {deep} edge {edge}");

        let (bx, bz) = west_berm(&soil);
        assert_eq!(band(&soil, bx, bz), Band::Berm, "berm at {bx},{bz}");
        let h = soil.sample(bx, bz).height;
        assert!(h > 0.35, "berm height {h}");
        assert!(soil.sample(bx, bz).water < 0.01);
    }

    #[test]
    fn every_piece_crosses_the_surface() {
        let soil = SoilField::demo();
        let mesh = dressing_mesh(&soil);
        assert!(mesh.indices.len() % 3 == 0);
        assert!(mesh.vertices.len() > 1_000);
        let mut span = std::collections::HashMap::<(i32, i32), (f32, f32, f32)>::new();
        for v in &mesh.vertices {
            assert!(v.panel == PANEL_PEBBLE || v.panel == PANEL_HARD_CLOD);
            assert!(v.nrm[0].is_finite() && v.nrm[1].is_finite() && v.nrm[2].is_finite());
            let key = (
                (v.uv[0] * 1000.0).round() as i32,
                (v.uv[1] * 1000.0).round() as i32,
            );
            let e = span.entry(key).or_insert((v.pos[1], v.pos[1], v.panel));
            e.0 = e.0.min(v.pos[1]);
            e.1 = e.1.max(v.pos[1]);
        }
        assert!(span.len() > 400, "pieces {}", span.len());
        for ((x, z), (lo, hi, _)) in span {
            assert!(lo < -0.008, "not buried at {x},{z}: {lo}");
            assert!(hi > 0.01, "not proud at {x},{z}: {hi}");
            let sunk = -lo / (hi - lo).max(1.0e-4);
            assert!((0.35..0.82).contains(&sunk), "sunk {sunk} at {x},{z}");
        }
    }
}
