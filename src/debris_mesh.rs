//! Grass blades, forked twigs and wet clods for bits stuck on the buggy.
//! Vertices are already in world space. The vehicle meshes are not touched.

use glam::{Mat4, Vec3};
use silt::debris::{Bit, Debris, KIND_CLOD, KIND_GRASS, KIND_TWIG};
use silt::vehicle::Vehicle;

use crate::mesh::{CpuMesh, MeshVertex};

// Shore is 47 and spray sheets are 48. Stuck bits start at 50.
const PANEL_GRASS: f32 = 50.0;
const PANEL_TWIG: f32 = 51.0;
/// Stuck wet mud. Impact clods stay on panel 46.
const PANEL_STUCK: f32 = 52.0;
const MAX_VERTS: usize = 7800;

pub fn debris_mesh(vehicle: &Vehicle, debris: &Debris) -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for bit in debris.bits() {
        if !bit.alive {
            continue;
        }
        if mesh.vertices.len() + 180 > MAX_VERTS {
            break;
        }
        let (mut pos, axis, side) = bit.frame(vehicle);
        if axis.length_squared() < 1.0e-6 {
            continue;
        }
        // Sit the piece on the outside of the tread or panel. A tangent axis
        // would leave the clod buried in the tire.
        let out = surface_out(vehicle, bit, pos);
        let lift = match bit.kind {
            KIND_CLOD => bit.length * 0.45,
            KIND_TWIG => 0.045,
            _ => 0.012,
        };
        pos += out * lift;
        match bit.kind {
            KIND_GRASS => add_grass(&mut mesh, bit, pos, axis, side),
            KIND_TWIG => add_twig(&mut mesh, bit, pos, axis, side),
            KIND_CLOD => add_clod(&mut mesh, bit, pos, axis, side),
            _ => {}
        }
    }
    mesh
}

fn surface_out(vehicle: &Vehicle, bit: &Bit, pos: Vec3) -> Vec3 {
    if bit.mount < 4 {
        let body = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
        let center = body.transform_point3(vehicle.wheel_center_local(bit.mount as usize));
        let out = pos - center;
        if out.length_squared() > 1.0e-6 {
            return out.normalize();
        }
    }
    let (_, axis, _) = bit.frame(vehicle);
    if axis.length_squared() > 1.0e-6 {
        axis
    } else {
        Vec3::Y
    }
}

fn add_grass(mesh: &mut CpuMesh, bit: &Bit, pos: Vec3, axis: Vec3, side: Vec3) {
    let width = (bit.length * 0.08).clamp(0.018, 0.042);
    let thick = width * 0.45;
    // Two blades in a narrow fan, the way a few stalks leave one clod.
    add_blade(mesh, pos, axis, side, bit.length, width, thick, bit.twist);
    let q = glam::Quat::from_axis_angle(axis, 0.55 + bit.twist * 0.15);
    let side2 = q * side;
    add_blade(
        mesh,
        pos + side * width * 0.35,
        (axis + side * 0.28).normalize_or_zero(),
        side2,
        bit.length * 0.82,
        width * 0.85,
        thick,
        bit.twist + 1.3,
    );
}

fn add_blade(
    mesh: &mut CpuMesh,
    pos: Vec3,
    axis: Vec3,
    side: Vec3,
    length: f32,
    width: f32,
    thick: f32,
    twist: f32,
) {
    let axis = axis.normalize_or_zero();
    let side = reorth(side, axis);
    let up = axis.cross(side).normalize_or_zero();
    let bend = up * (length * 0.05) + side * (twist.sin() * length * 0.03);
    let p0 = pos;
    let p1 = pos + axis * (length * 0.55) + bend * 0.55;
    let p2 = pos + axis * length + bend;
    segment(
        mesh,
        p0,
        p1,
        side,
        up,
        width,
        thick,
        width * 0.62,
        thick * 0.7,
        0.0,
        0.55,
        PANEL_GRASS,
    );
    segment(
        mesh,
        p1,
        p2,
        side,
        up,
        width * 0.62,
        thick * 0.7,
        width * 0.08,
        thick * 0.08,
        0.55,
        1.0,
        PANEL_GRASS,
    );
}

fn add_twig(mesh: &mut CpuMesh, bit: &Bit, pos: Vec3, axis: Vec3, side: Vec3) {
    let axis = axis.normalize_or_zero();
    let side = reorth(side, axis);
    let up = axis.cross(side).normalize_or_zero();
    let radius = (bit.length * 0.045).clamp(0.02, 0.038);
    let bend = up * (bit.length * 0.12 * (bit.twist.sin()));
    let mid = pos + axis * (bit.length * 0.48) + bend * 0.4;
    let tip = pos + axis * bit.length + bend;
    stick(mesh, pos, mid, radius, radius * 0.8, 0.0, 0.48, PANEL_TWIG);
    stick(
        mesh,
        mid,
        tip,
        radius * 0.8,
        radius * 0.35,
        0.48,
        1.0,
        PANEL_TWIG,
    );
    // A fork, like the twigs caught across the belly and the sidewall.
    let branch_dir = (axis + side * 0.85 + up * 0.25 * bit.twist.cos()).normalize_or_zero();
    let fork = mid + branch_dir * (bit.length * 0.42);
    stick(
        mesh,
        mid,
        fork,
        radius * 0.62,
        radius * 0.22,
        0.35,
        0.9,
        PANEL_TWIG,
    );
}

fn add_clod(mesh: &mut CpuMesh, bit: &Bit, pos: Vec3, axis: Vec3, side: Vec3) {
    let axis = axis.normalize_or_zero();
    let side = reorth(side, axis);
    let bin = axis.cross(side).normalize_or_zero();
    let radius = bit.length.max(0.06);
    let center = pos;
    let squash = Vec3::new(1.15, 0.58, 1.05);
    let mut pts = Vec::with_capacity(14);
    let mut nrm = Vec::with_capacity(14);
    let place = |dir: Vec3| {
        let j = 0.90 + hash(bit.seed, (dir.x * 9.0 + dir.z * 4.0) as u32) * 0.16;
        let shaped = side * dir.x * radius * squash.x * j
            + axis * dir.y * radius * squash.y * j
            + bin * dir.z * radius * squash.z * j;
        let n = side * dir.x / squash.x + axis * dir.y / squash.y + bin * dir.z / squash.z;
        (center + shaped, n.normalize_or_zero())
    };
    let (p, n) = place(Vec3::Y);
    pts.push(p);
    nrm.push(n);
    let stacks: usize = 3;
    let slices: usize = 6;
    for s in 1..stacks {
        let phi = s as f32 / stacks as f32 * std::f32::consts::PI;
        for i in 0..slices {
            let th = i as f32 / slices as f32 * std::f32::consts::TAU;
            let dir = Vec3::new(th.cos() * phi.sin(), phi.cos(), th.sin() * phi.sin());
            let (p, n) = place(dir);
            pts.push(p);
            nrm.push(n);
        }
    }
    let (p, n) = place(-Vec3::Y);
    pts.push(p);
    nrm.push(n);
    let ring = |s: usize, i: usize| 1 + (s - 1) * slices + (i % slices);
    for i in 0..slices {
        tri(
            mesh,
            pts[0],
            nrm[0],
            pts[ring(1, i)],
            nrm[ring(1, i)],
            pts[ring(1, i + 1)],
            nrm[ring(1, i + 1)],
            PANEL_STUCK,
            center,
            0.4,
        );
    }
    for s in 1..stacks.saturating_sub(1) {
        for i in 0..slices {
            let a = ring(s, i);
            let b = ring(s, i + 1);
            let c = ring(s + 1, i);
            let d = ring(s + 1, i + 1);
            tri(
                mesh,
                pts[a],
                nrm[a],
                pts[c],
                nrm[c],
                pts[b],
                nrm[b],
                PANEL_STUCK,
                center,
                0.5,
            );
            tri(
                mesh,
                pts[b],
                nrm[b],
                pts[c],
                nrm[c],
                pts[d],
                nrm[d],
                PANEL_STUCK,
                center,
                0.5,
            );
        }
    }
    let bot = pts.len() - 1;
    let last = stacks - 1;
    for i in 0..slices {
        tri(
            mesh,
            pts[bot],
            nrm[bot],
            pts[ring(last, i + 1)],
            nrm[ring(last, i + 1)],
            pts[ring(last, i)],
            nrm[ring(last, i)],
            PANEL_STUCK,
            center,
            0.3,
        );
    }
}

fn segment(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    side: Vec3,
    up: Vec3,
    w0: f32,
    t0: f32,
    w1: f32,
    t1: f32,
    v0: f32,
    v1: f32,
    panel: f32,
) {
    let s0 = side * w0;
    let u0 = up * t0;
    let s1 = side * w1;
    let u1 = up * t1;
    let a0 = a + s0;
    let a1 = a - s0 * 0.45 + u0;
    let a2 = a - s0 * 0.45 - u0;
    let b0 = b + s1;
    let b1 = b - s1 * 0.45 + u1;
    let b2 = b - s1 * 0.45 - u1;
    let center = (a + b) * 0.5;
    quad(mesh, a0, b0, b1, a1, v0, v1, panel, center);
    quad(mesh, a1, b1, b2, a2, v0, v1, panel, center);
    quad(mesh, a2, b2, b0, a0, v0, v1, panel, center);
}

fn stick(mesh: &mut CpuMesh, a: Vec3, b: Vec3, r0: f32, r1: f32, v0: f32, v1: f32, panel: f32) {
    let axis = b - a;
    if axis.length_squared() < 1.0e-8 {
        return;
    }
    let dir = axis.normalize();
    let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
    let side = dir.cross(helper).normalize();
    let up = dir.cross(side).normalize();
    let center = (a + b) * 0.5;
    let n = 4;
    for i in 0..n {
        let t0 = i as f32 / n as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
        let p = |t: f32, r: f32, o: Vec3| o + (side * t.cos() + up * t.sin()) * r;
        quad(
            mesh,
            p(t0, r0, a),
            p(t0, r1, b),
            p(t1, r1, b),
            p(t1, r0, a),
            v0,
            v1,
            panel,
            center,
        );
    }
}

fn quad(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    d: Vec3,
    v0: f32,
    v1: f32,
    panel: f32,
    center: Vec3,
) {
    tri_pos(
        mesh,
        a,
        b,
        c,
        panel,
        center,
        [0.0, v0],
        [1.0, v1],
        [0.0, v1],
    );
    tri_pos(
        mesh,
        a,
        c,
        d,
        panel,
        center,
        [0.0, v0],
        [0.0, v1],
        [1.0, v0],
    );
}

fn tri(
    mesh: &mut CpuMesh,
    a: Vec3,
    na: Vec3,
    b: Vec3,
    nb: Vec3,
    c: Vec3,
    nc: Vec3,
    panel: f32,
    center: Vec3,
    v: f32,
) {
    let face = (b - a).cross(c - a);
    if face.length_squared() < 1.0e-10 {
        return;
    }
    let (b, nb, c, nc) = if face.dot(a - center) < 0.0 {
        (c, nc, b, nb)
    } else {
        (b, nb, c, nc)
    };
    let base = mesh.vertices.len() as u32;
    for (p, n) in [(a, na), (b, nb), (c, nc)] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.normalize_or_zero().to_array(),
            uv: [0.5, v],
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn tri_pos(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    panel: f32,
    center: Vec3,
    ua: [f32; 2],
    ub: [f32; 2],
    uc: [f32; 2],
) {
    let face = (b - a).cross(c - a);
    if face.length_squared() < 1.0e-10 {
        return;
    }
    let (b, c, ub, uc, n) = if face.dot(a - center) < 0.0 {
        (c, b, uc, ub, -face.normalize_or_zero())
    } else {
        (b, c, ub, uc, face.normalize_or_zero())
    };
    let base = mesh.vertices.len() as u32;
    for (p, uv) in [(a, ua), (b, ub), (c, uc)] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv,
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn reorth(side: Vec3, axis: Vec3) -> Vec3 {
    let s = side - axis * side.dot(axis);
    if s.length_squared() < 1.0e-6 {
        let helper = if axis.y.abs() < 0.85 {
            Vec3::Y
        } else {
            Vec3::X
        };
        axis.cross(helper).normalize_or_zero()
    } else {
        s.normalize_or_zero()
    }
}

fn hash(seed: u32, i: u32) -> f32 {
    let mut x = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9));
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    (x & 0xFFFF) as f32 / 65535.0
}
