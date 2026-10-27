//! Sheet of thrown mud. A curved ribbon with thickness, standing in the
//! particle's frame: local X across the sheet, Y up, Z the thin axis.

use glam::Vec3;

use crate::mesh::{CpuMesh, MeshVertex};

/// Spray sheet. Not a body panel, so adhesion does not own it.
/// Panel 47 is the puddle shore. Sheets sit on 48 so the two do not share a draw.
pub const SHEET_PANEL: f32 = 48.0;

pub fn add_spray_sheet(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let w = p.size * 1.15;
    let h = p.size * 1.4;
    let thick = (p.size * 0.09).clamp(0.022, 0.04);
    let slump = if p.grounded { 0.32 } else { 1.0 };
    let cols = 5usize;
    let rows = 4usize;
    let mut pts = Vec::with_capacity((rows + 1) * (cols + 1));
    for r in 0..=rows {
        let v = r as f32 / rows as f32;
        for c in 0..=cols {
            let u = c as f32 / cols as f32;
            let x = (u - 0.5) * 2.0 * w;
            let y = v * h * slump;
            let bow = (v * std::f32::consts::PI).sin() * w * 0.22 * slump;
            let edge = (1.0 - (u - 0.5).abs() * 1.5).max(0.12);
            pts.push(Vec3::new(x, y, bow * edge));
        }
    }
    let idx = |r: usize, c: usize| r * (cols + 1) + c;
    let xform = |q: Vec3| p.pos + p.rot * q;
    for side in [-1.0_f32, 1.0] {
        for r in 0..rows {
            for c in 0..cols {
                let mut quad = [
                    pts[idx(r, c)] + Vec3::Z * thick * side,
                    pts[idx(r, c + 1)] + Vec3::Z * thick * side,
                    pts[idx(r + 1, c + 1)] + Vec3::Z * thick * side,
                    pts[idx(r + 1, c)] + Vec3::Z * thick * side,
                ];
                if side < 0.0 {
                    quad.swap(1, 3);
                }
                push_tri(mesh, xform(quad[0]), xform(quad[1]), xform(quad[2]));
                push_tri(mesh, xform(quad[0]), xform(quad[2]), xform(quad[3]));
            }
        }
    }
    // Rim so the sheet has an edge when the camera is almost on it.
    let mut edge_pair = |a: Vec3, b: Vec3, c: Vec3, d: Vec3| {
        push_tri(mesh, a, b, c);
        push_tri(mesh, a, c, d);
    };
    for r in 0..rows {
        let y0 = pts[idx(r, 0)];
        let y1 = pts[idx(r + 1, 0)];
        let z0 = pts[idx(r, cols)];
        let z1 = pts[idx(r + 1, cols)];
        edge_pair(
            xform(y0 + Vec3::Z * thick),
            xform(y0 - Vec3::Z * thick),
            xform(y1 - Vec3::Z * thick),
            xform(y1 + Vec3::Z * thick),
        );
        edge_pair(
            xform(z0 - Vec3::Z * thick),
            xform(z0 + Vec3::Z * thick),
            xform(z1 + Vec3::Z * thick),
            xform(z1 - Vec3::Z * thick),
        );
    }
}

fn push_tri(mesh: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3) {
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1.0e-12 {
        return;
    }
    let n = n.normalize();
    let base = mesh.vertices.len() as u32;
    for p in [a, b, c] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv: [p.x * 0.4, p.y * 0.4],
            panel: SHEET_PANEL,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}
