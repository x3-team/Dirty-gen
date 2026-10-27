//! Creased panels and a torn fender, drawn on top of the buggy after a hit.
//!
//! The vehicle mesh stays as it is. These sheets are the damage: sharp folds,
//! a silver tear, mud on the lip, and one panel that hangs off the front wheel.

use glam::{Mat4, Vec3};
use silt::body_damage::BodyDamage;
use silt::vehicle::Vehicle;

use crate::mesh::{CpuMesh, MeshVertex};

const SKIN: f32 = 48.0;
const MUD: f32 = 5.0;
const CLOD: f32 = 49.0;
const BARE: f32 = 47.0;

/// Inboard mount of the torn front-left fender, just ahead of the cockpit.
pub const LOOSE_HINGE: Vec3 = Vec3::new(-0.18, 0.32, 1.00);

pub fn loose_transform(vehicle: &Vehicle, damage: &BodyDamage) -> Mat4 {
    let body = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
    // Positive X drops the forward edge, so the torn fender hangs in front of the wheel.
    body * Mat4::from_translation(LOOSE_HINGE) * Mat4::from_rotation_x(damage.loose)
}

pub fn dent_mesh(damage: &BodyDamage) -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    if !damage.hit {
        return mesh;
    }
    let depth = 0.09 + 0.06 * damage.dent;
    // Hood, seen from above. Orange facets, mud in the pit, metal on the sharp fold.
    add_crumple(
        &mut mesh,
        Vec3::new(0.0, 0.24, 1.30),
        Vec3::X * 0.28,
        Vec3::Z * 0.22,
        Vec3::Y,
        depth,
        SKIN,
        5,
        4,
        false,
    );
    // Left of the nose, above the lamp, so the three-quarter view shows a crushed corner.
    add_crumple(
        &mut mesh,
        Vec3::new(-0.12, 0.16, 1.62),
        Vec3::X * 0.16,
        Vec3::Y * 0.08,
        Vec3::Z,
        depth * 0.85,
        SKIN,
        4,
        3,
        false,
    );
    // Left flank with the bite the fender tore out of.
    add_crumple(
        &mut mesh,
        Vec3::new(-0.50, 0.24, 0.85),
        Vec3::Z * 0.28,
        Vec3::Y * 0.14,
        Vec3::new(-1.0, 0.0, 0.0),
        depth * 0.85,
        SKIN,
        5,
        4,
        true,
    );
    mesh
}

pub fn loose_mesh(damage: &BodyDamage) -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    if !damage.hit {
        return mesh;
    }
    let nu = 6;
    let nv = 7;
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    let count = (nu + 1) * (nv + 1);
    let mut p = vec![Vec3::ZERO; count];
    let mut nrm = vec![Vec3::ZERO; count];
    for j in 0..=nv {
        for i in 0..=nu {
            let u = i as f32 / nu as f32;
            let v = j as f32 / nv as f32;
            p[id(i, j)] = loose_point(u, v);
        }
    }
    for j in 0..nv {
        for i in 0..nu {
            let a = id(i, j);
            let b = id(i + 1, j);
            let d = id(i, j + 1);
            let c = id(i + 1, j + 1);
            let n1 = (p[b] - p[a]).cross(p[d] - p[a]);
            let n2 = (p[c] - p[b]).cross(p[d] - p[b]);
            nrm[a] += n1;
            nrm[b] += n1 + n2;
            nrm[d] += n1 + n2;
            nrm[c] += n2;
        }
    }
    for n in &mut nrm {
        *n = n.normalize_or_zero();
        if n.length_squared() < 1.0e-8 {
            *n = Vec3::Y;
        }
    }
    let thick = 0.02;
    for j in 0..nv {
        for i in 0..nu {
            let corner = |ix, jy| {
                let k = id(ix, jy);
                (p[k], p[k] - nrm[k] * thick)
            };
            let (a, ab) = corner(i, j);
            let (b, bb) = corner(i + 1, j);
            let (c, cb) = corner(i + 1, j + 1);
            let (d, db) = corner(i, j + 1);
            // Orange on both sides. A fold would otherwise show the dark back as a hole.
            // Only the forward lip is bare metal, where the fender tore.
            let front = if j + 1 == nv { BARE } else { SKIN };
            let back = front;
            add_tri(&mut mesh, a, b, c, front);
            add_tri(&mut mesh, a, c, d, front);
            add_tri(&mut mesh, ab, cb, bb, back);
            add_tri(&mut mesh, ab, db, cb, back);
        }
    }
    let edge = |mesh: &mut CpuMesh, i0, j0, i1, j1, panel| {
        let a = id(i0, j0);
        let b = id(i1, j1);
        add_tri(mesh, p[a], p[b], p[b] - nrm[b] * thick, panel);
        add_tri(
            mesh,
            p[a],
            p[b] - nrm[b] * thick,
            p[a] - nrm[a] * thick,
            panel,
        );
    };
    for i in 0..nu {
        edge(&mut mesh, i, 0, i + 1, 0, MUD);
        edge(&mut mesh, i, nv, i + 1, nv, BARE);
    }
    for j in 0..nv {
        let outer = if j * 2 >= nv { BARE } else { MUD };
        edge(&mut mesh, 0, j, 0, j + 1, MUD);
        edge(&mut mesh, nu, j, nu, j + 1, outer);
    }
    for i in [0usize, 2, 3, 4, 6] {
        let k = id(i, nv);
        let r = 0.055 + (i % 3) as f32 * 0.016;
        add_clod(&mut mesh, p[k] + nrm[k] * (0.02 + r * 0.4), r);
    }
    add_clod(&mut mesh, p[id(3, 4)] + nrm[id(3, 4)] * 0.03, 0.05);
    mesh
}

fn loose_point(u: f32, v: f32) -> Vec3 {
    // u runs outward, v runs from the hinge toward the torn forward edge.
    let mut x = -0.02 - 0.86 * u;
    let mut z = 0.92 * v;
    let mut y = 0.02 - 0.10 * v * v;
    let crease = 1.0 - ((v - 0.42) / 0.08).abs();
    if crease > 0.0 {
        y -= crease * 0.06;
    }
    let ridge = 1.0 - ((u - 0.55) / 0.12).abs();
    if ridge > 0.0 && v > 0.12 {
        y += ridge * 0.045;
    }
    if v > 0.7 {
        let e = ((v - 0.7) / 0.3).clamp(0.0, 1.0);
        let jag = (u * 13.0).sin() * 0.7 + (u * 5.5 + 0.3).cos() * 0.3;
        z += e * jag * 0.11;
        y -= e * (0.04 + 0.06 * jag.abs());
        x -= e * 0.09 * jag.abs();
    }
    Vec3::new(x, y, z)
}

fn add_crumple(
    mesh: &mut CpuMesh,
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    outward: Vec3,
    depth: f32,
    skin: f32,
    cells_u: usize,
    cells_v: usize,
    tear: bool,
) {
    let mut grid = vec![Vec3::ZERO; (cells_u + 1) * (cells_v + 1)];
    let mut bowl = vec![0.0_f32; grid.len()];
    let at = |i: usize, j: usize| j * (cells_u + 1) + i;
    for j in 0..=cells_v {
        for i in 0..=cells_u {
            let u = i as f32 / cells_u as f32;
            let v = j as f32 / cells_v as f32;
            let (out, valley) = relief(u, v);
            bowl[at(i, j)] = valley;
            grid[at(i, j)] = center
                + tangent * (u * 2.0 - 1.0)
                + bitangent * (v * 2.0 - 1.0)
                + outward * depth * out;
        }
    }
    for j in 0..cells_v {
        for i in 0..cells_u {
            let a = grid[at(i, j)];
            let b = grid[at(i + 1, j)];
            let c = grid[at(i + 1, j + 1)];
            let d = grid[at(i, j + 1)];
            let valley =
                (bowl[at(i, j)] + bowl[at(i + 1, j)] + bowl[at(i + 1, j + 1)] + bowl[at(i, j + 1)])
                    * 0.25;
            let torn = tear && (2..4).contains(&i) && (1..3).contains(&j);
            if torn {
                add_tear_lip(mesh, a, b, c, d, -outward);
            } else {
                let panel = if valley > 0.78 {
                    CLOD
                } else if valley > 0.62 {
                    BARE
                } else {
                    skin
                };
                add_tri(mesh, a, b, c, panel);
                add_tri(mesh, a, c, d, panel);
            }
        }
    }
    if !tear {
        let mut best = 0.0_f32;
        let mut where_ = center;
        for (idx, v) in bowl.iter().enumerate() {
            if *v > best {
                best = *v;
                where_ = grid[idx];
            }
        }
        add_clod(mesh, where_ + outward * 0.02, 0.05);
        add_clod(
            mesh,
            where_ + tangent * 0.35 - outward * depth * 0.15,
            0.038,
        );
    } else {
        add_clod(mesh, center - outward * 0.02 + bitangent * 0.15, 0.048);
        add_clod(mesh, center + tangent * 0.25 - bitangent * 0.1, 0.04);
    }
}

/// Outward displacement, and how deep the valley is (0 edge, 1 pit).
fn relief(u: f32, v: f32) -> (f32, f32) {
    let fu = (u * std::f32::consts::TAU * 1.5).sin();
    let fv = (v * std::f32::consts::TAU * 2.0).sin();
    let diag = ((u * 1.25 + v * 0.8) * std::f32::consts::TAU).sin();
    let ridge = fu.abs() * 0.5 + fv.abs() * 0.28 + diag.abs() * 0.4;
    let du = u * 2.0 - 1.0;
    let dv = (v * 2.0 - 1.0) * 0.85;
    let valley = (1.0 - (du * du + dv * dv).sqrt()).max(0.0).powf(1.05);
    let out = ridge * 0.48 - valley;
    (out, valley)
}

fn add_tear_lip(mesh: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3, d: Vec3, inward: Vec3) {
    let mid = (a + b + c + d) * 0.25 + inward * 0.025;
    let pull = 0.46;
    let a2 = a.lerp(mid, pull);
    let b2 = b.lerp(mid, pull);
    let c2 = c.lerp(mid, pull);
    let d2 = d.lerp(mid, pull);
    add_tri(mesh, a, b, b2, BARE);
    add_tri(mesh, a, b2, a2, BARE);
    add_tri(mesh, b, c, c2, BARE);
    add_tri(mesh, b, c2, b2, BARE);
    add_tri(mesh, c, d, d2, BARE);
    add_tri(mesh, c, d2, c2, BARE);
    add_tri(mesh, d, a, a2, BARE);
    add_tri(mesh, d, a2, d2, CLOD);
}

fn add_clod(mesh: &mut CpuMesh, c: Vec3, r: f32) {
    let s = [
        c + Vec3::new(0.0, r * 0.85, 0.0),
        c + Vec3::new(r, -0.25 * r, 0.1 * r),
        c + Vec3::new(-0.65 * r, -0.2 * r, 0.55 * r),
        c + Vec3::new(-0.15 * r, -0.35 * r, -0.75 * r),
        c + Vec3::new(0.45 * r, 0.15 * r, 0.6 * r),
    ];
    add_tri(mesh, s[0], s[1], s[4], CLOD);
    add_tri(mesh, s[0], s[4], s[2], CLOD);
    add_tri(mesh, s[0], s[2], s[3], CLOD);
    add_tri(mesh, s[0], s[3], s[1], CLOD);
    add_tri(mesh, s[1], s[3], s[2], CLOD);
    add_tri(mesh, s[1], s[2], s[4], CLOD);
}

fn add_tri(mesh: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3, panel: f32) {
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1.0e-10 {
        return;
    }
    let n = n.normalize();
    let base = mesh.vertices.len() as u32;
    for p in [a, b, c] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv: [p.x * 1.6 + p.z * 0.2, p.y * 1.35 + 0.55],
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit() -> BodyDamage {
        let mut d = BodyDamage::new();
        d.note_contact(6.0);
        d
    }

    #[test]
    fn clean_body_has_no_dent_mesh() {
        let d = BodyDamage::new();
        assert!(dent_mesh(&d).indices.is_empty());
        assert!(loose_mesh(&d).indices.is_empty());
    }

    #[test]
    fn hit_creases_panels_and_tears_the_fender() {
        let d = hit();
        let dents = dent_mesh(&d);
        assert!(dents.indices.len() > 24);
        assert!(
            dents.vertices.iter().any(|v| (v.panel - BARE).abs() < 0.01),
            "crease should show bare metal"
        );
        let hood: Vec<f32> = dents
            .vertices
            .iter()
            .filter(|v| v.pos[2] > 1.15 && v.pos[2] < 1.55)
            .map(|v| v.pos[1])
            .collect();
        let min_y = hood.iter().copied().fold(f32::MAX, f32::min);
        let max_y = hood.iter().copied().fold(f32::MIN, f32::max);
        assert!(
            max_y - min_y > 0.04,
            "hood dent is flat, span {:.3}",
            max_y - min_y
        );

        let loose = loose_mesh(&d);
        assert!(loose.vertices.len() > 40);
        assert!(loose.vertices.iter().any(|v| (v.panel - BARE).abs() < 0.01));
        assert!(loose.vertices.iter().any(|v| (v.panel - CLOD).abs() < 0.01));
        let mut xs: Vec<f32> = loose.vertices.iter().map(|v| v.pos[0]).collect();
        xs.sort_by(|a, b| a.total_cmp(b));
        let n = (xs.len() / 6).max(4);
        let outer = &xs[..n];
        let mean = outer.iter().sum::<f32>() / n as f32;
        let var = outer.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
        assert!(var > 4.0e-4, "torn edge is a straight cut, var {var}");
        assert!(xs[0] < -0.55, "fender does not reach outward, {}", xs[0]);
    }
}
