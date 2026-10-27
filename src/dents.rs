//! Creased panels and a torn fender, drawn on top of the buggy after a hit.
//!
//! The vehicle mesh stays as it is. These sheets are the damage. The surface
//! stays smooth so a fold is a normal-map crease in the shader, not a black
//! facet. Paint chips and the bare lip live on the same panels. The torn
//! front fender hangs off the hinge with a jagged edge.

use glam::{Mat4, Vec3};
use silt::body_damage::BodyDamage;
use silt::vehicle::Vehicle;

use crate::mesh::{CpuMesh, MeshVertex};

const SKIN: f32 = 48.0;
const CLOD: f32 = 49.0;
const BARE: f32 = 47.0;

/// GPU buffers in `render` are sized for these. A finer sheet has to stay inside.
#[cfg(test)]
const MAX_VERTS: usize = 4096;
#[cfg(test)]
const MAX_INDEX: usize = 12288;

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
    // Hood, seen from above. A smooth bowl; the shader creases the paint.
    add_crumple(
        &mut mesh,
        Vec3::new(0.0, 0.24, 1.30),
        Vec3::X * 0.28,
        Vec3::Z * 0.22,
        Vec3::Y,
        depth,
        false,
    );
    // Left of the nose, above the lamp.
    add_crumple(
        &mut mesh,
        Vec3::new(-0.12, 0.16, 1.62),
        Vec3::X * 0.16,
        Vec3::Y * 0.08,
        Vec3::Z,
        depth * 0.85,
        false,
    );
    // Left flank. The outer corner peels down where the fender tore away.
    add_crumple(
        &mut mesh,
        Vec3::new(-0.50, 0.24, 0.85),
        Vec3::Z * 0.28,
        Vec3::Y * 0.14,
        Vec3::new(-1.0, 0.0, 0.0),
        depth * 0.85,
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
    let nu = 18;
    let nv = 22;
    let (p, mut n) = build_grid(nu, nv, loose_point);
    // Face the sheet upward. The curl's underside is the only second side.
    let flip = n[n.len() / 2].dot(Vec3::Y) < 0.0;
    if flip {
        for v in &mut n {
            *v = -*v;
        }
    }
    emit_grid(&mut mesh, nu, nv, &p, &n, loose_cell, 0.012, flip, 0.68);
    emit_rim(&mut mesh, nu, nv, &p, &n, 0.012, loose_rim);
    emit_teeth(&mut mesh, nu, nv, &p, &n);
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    for i in (2..nu).step_by(4) {
        let k = id(i, nv);
        let r = 0.026 + (i % 3) as f32 * 0.008;
        add_clod(&mut mesh, p[k] + n[k] * 0.012, n[k], r);
    }
    mesh
}

fn loose_point(u: f32, v: f32) -> Vec3 {
    // u runs outward, v runs from the hinge toward the torn forward edge.
    let jag = tear_jag(u);
    let tear = smoothstep(0.74, 1.0, v);
    let curl = smoothstep(0.86, 1.0, v);
    let mut x = -0.03 - 0.82 * u;
    let mut z = 0.84 * v;
    // Gentle crown and droop. Sharp folds are the shader normal map.
    let crown = (u * std::f32::consts::PI).sin() * 0.03 * (1.0 - 0.35 * v);
    let bend = (v * std::f32::consts::TAU).sin() * 0.01 * (u * (1.0 - u)).clamp(0.0, 1.0).sqrt();
    let mut y = 0.035 + crown + bend - 0.10 * v * v;
    x += tear * jag * 0.09;
    z += tear * jag * 0.11;
    y -= tear * (0.02 + 0.03 * jag.abs());
    // The lip rolls under so the torn edge reads as a hanging sheet, not a cut.
    y -= curl * 0.09;
    z -= curl * 0.03;
    Vec3::new(x, y, z)
}

fn tear_jag(u: f32) -> f32 {
    let w = (u * std::f32::consts::TAU).sin() * 0.62
        + (u * std::f32::consts::TAU * 2.0 + 1.4).sin() * 0.38;
    w * w.abs()
}

fn loose_cell(_i: usize, _j: usize, _nu: usize, _nv: usize) -> f32 {
    // Paint covers the face. Bare metal is the rim, the teeth, and shader chips.
    SKIN
}

fn loose_rim(i0: usize, j0: usize, i1: usize, j1: usize, nu: usize, nv: usize) -> f32 {
    let v = (j0.max(j1) as f32) / nv as f32;
    let hinge = j0 == 0 && j1 == 0;
    let inboard = i0 == 0 && i1 == 0 && v < 0.45;
    if hinge || inboard {
        return SKIN;
    }
    // Mud on the outboard lip, bare metal on the tear itself.
    if (i0 == nu || i1 == nu) && v > 0.4 && v < 0.82 {
        return CLOD;
    }
    BARE
}

fn add_crumple(
    mesh: &mut CpuMesh,
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    outward: Vec3,
    depth: f32,
    tear: bool,
) {
    let nu = 12;
    let nv = 8;
    let (p, mut n) = build_grid(nu, nv, |u, v| {
        dent_point(center, tangent, bitangent, outward, depth, u, v, tear)
    });
    // One shell, facing out. The back would double the buffer and only shows the chassis.
    let flip = n[n.len() / 2].dot(outward) < 0.0;
    if flip {
        for v in &mut n {
            *v = -*v;
        }
    }
    let panel = move |i, j, cu, cv| dent_cell(i, j, cu, cv, tear);
    emit_grid(mesh, nu, nv, &p, &n, panel, 0.0, flip, 2.0);
    emit_rim(mesh, nu, nv, &p, &n, 0.01, move |i0, j0, i1, j1, cu, cv| {
        if tear {
            let u = (i0.max(i1) as f32) / cu as f32;
            let v = (j0.min(j1) as f32) / cv as f32;
            if u > 0.7 && v < 0.4 {
                return BARE;
            }
        }
        SKIN
    });
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    let mut best = -1.0_f32;
    let mut where_ = center;
    let mut up = outward;
    for j in 0..=nv {
        for i in 0..=nu {
            let u = i as f32 / nu as f32;
            let v = j as f32 / nv as f32;
            let (_, valley) = relief(u, v);
            if valley > best {
                best = valley;
                let k = id(i, j);
                where_ = p[k];
                up = n[k];
            }
        }
    }
    add_clod(mesh, where_ + up * 0.012, up, 0.04);
    if tear {
        let k = id(nu, 1);
        add_clod(mesh, p[k] + n[k] * 0.01, n[k], 0.032);
        add_clod(
            mesh,
            p[id(nu - 2, 2)] + n[id(nu - 2, 2)] * 0.01,
            n[id(nu - 2, 2)],
            0.026,
        );
        emit_corner_tabs(mesh, nu, nv, &p, &n);
    } else {
        add_chip(
            mesh,
            where_ + tangent * 0.12 + up * 0.008,
            up,
            tangent,
            0.028,
        );
        add_chip(
            mesh,
            where_ - tangent * 0.08 + bitangent * 0.06 + up * 0.008,
            up,
            bitangent,
            0.022,
        );
    }
}

fn dent_point(
    center: Vec3,
    tangent: Vec3,
    bitangent: Vec3,
    outward: Vec3,
    depth: f32,
    u: f32,
    v: f32,
    tear: bool,
) -> Vec3 {
    let (out, _) = relief(u, v);
    let mut p = center
        + tangent * (u * 2.0 - 1.0)
        + bitangent * (v * 2.0 - 1.0)
        + outward * (depth * out + 0.012);
    if tear {
        // Forward-lower corner peels off the flank and hangs. No hole.
        let peel = smoothstep(0.58, 1.0, u) * smoothstep(0.42, 0.0, v);
        let jag = (v * 11.0 + u * 4.0).sin();
        p += outward * peel * 0.04;
        p -= Vec3::Y * peel * (0.045 + 0.02 * jag.abs());
        p += tangent.normalize_or_zero() * peel * jag * 0.028;
    }
    p
}

fn dent_cell(i: usize, j: usize, nu: usize, nv: usize, tear: bool) -> f32 {
    let u = (i as f32 + 0.5) / nu as f32;
    let v = (j as f32 + 0.5) / nv as f32;
    if tear && u > 0.74 && v < 0.32 {
        return BARE;
    }
    // A few scraped cells. The shader chips the rest of the paint.
    if !tear && matches!((i, j), (3, 2) | (8, 5) | (5, 3)) {
        return BARE;
    }
    SKIN
}

/// Outward displacement, and how deep the valley is (0 edge, 1 pit).
fn relief(u: f32, v: f32) -> (f32, f32) {
    let du = u * 2.0 - 1.0;
    let dv = (v * 2.0 - 1.0) * 0.85;
    let valley = (1.0 - (du * du + dv * dv).sqrt()).max(0.0).powf(1.15);
    let ridge = (1.0 - ((u * 0.75 + v - 0.95) / 0.22).abs()).max(0.0);
    let out = ridge * 0.16 - valley;
    (out, valley)
}

fn build_grid(nu: usize, nv: usize, point: impl Fn(f32, f32) -> Vec3) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut p = vec![Vec3::ZERO; (nu + 1) * (nv + 1)];
    for j in 0..=nv {
        for i in 0..=nu {
            let u = i as f32 / nu as f32;
            let v = j as f32 / nv as f32;
            p[j * (nu + 1) + i] = point(u, v);
        }
    }
    let n = smooth_normals(nu, nv, &p);
    (p, n)
}

fn smooth_normals(nu: usize, nv: usize, p: &[Vec3]) -> Vec<Vec3> {
    let mut n = vec![Vec3::ZERO; p.len()];
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    for j in 0..nv {
        for i in 0..nu {
            let a = id(i, j);
            let b = id(i + 1, j);
            let c = id(i + 1, j + 1);
            let d = id(i, j + 1);
            let n1 = (p[b] - p[a]).cross(p[d] - p[a]);
            let n2 = (p[c] - p[b]).cross(p[d] - p[b]);
            n[a] += n1;
            n[b] += n1 + n2;
            n[d] += n1 + n2;
            n[c] += n2;
        }
    }
    for v in &mut n {
        *v = v.normalize_or_zero();
        if v.length_squared() < 1.0e-8 {
            *v = Vec3::Y;
        }
    }
    // A jag can flip one vertex through the sheet. Pull it back into the
    // hemisphere so that facet cannot shade as a black spike.
    let avg = n.iter().copied().sum::<Vec3>().normalize_or_zero();
    if avg.length_squared() > 0.25 {
        for v in &mut n {
            if v.dot(avg) < 0.3 {
                *v = (*v + avg).normalize();
            }
        }
    }
    n
}

fn emit_grid(
    mesh: &mut CpuMesh,
    nu: usize,
    nv: usize,
    p: &[Vec3],
    n: &[Vec3],
    mut panel_of: impl FnMut(usize, usize, usize, usize) -> f32,
    thick: f32,
    flip: bool,
    back_from: f32,
) {
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    for j in 0..nv {
        for i in 0..nu {
            let panel = panel_of(i, j, nu, nv);
            let v_cell = (j as f32 + 0.5) / nv as f32;
            let corner = |ix: usize, jy: usize| {
                let k = id(ix, jy);
                let uv = [ix as f32 / nu as f32, jy as f32 / nv as f32];
                (p[k], n[k], uv)
            };
            let (a, na, ua) = corner(i, j);
            let (b, nb, ub) = corner(i + 1, j);
            let (c, nc, uc) = corner(i + 1, j + 1);
            let (d, nd, ud) = corner(i, j + 1);
            if flip {
                add_tri(mesh, a, c, b, na, nc, nb, ua, uc, ub, panel);
                add_tri(mesh, a, d, c, na, nd, nc, ua, ud, uc, panel);
            } else {
                add_tri(mesh, a, b, c, na, nb, nc, ua, ub, uc, panel);
                add_tri(mesh, a, c, d, na, nc, nd, ua, uc, ud, panel);
            }
            if thick > 0.0 && v_cell >= back_from {
                let ab = a - na * thick;
                let bb = b - nb * thick;
                let cb = c - nc * thick;
                let db = d - nd * thick;
                if flip {
                    add_tri(mesh, ab, bb, cb, -na, -nb, -nc, ua, ub, uc, panel);
                    add_tri(mesh, ab, cb, db, -na, -nc, -nd, ua, uc, ud, panel);
                } else {
                    add_tri(mesh, ab, cb, bb, -na, -nc, -nb, ua, uc, ub, panel);
                    add_tri(mesh, ab, db, cb, -na, -nd, -nc, ua, ud, uc, panel);
                }
            }
        }
    }
}

fn emit_rim(
    mesh: &mut CpuMesh,
    nu: usize,
    nv: usize,
    p: &[Vec3],
    n: &[Vec3],
    thick: f32,
    mut edge_panel: impl FnMut(usize, usize, usize, usize, usize, usize) -> f32,
) {
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    let center = p[id(nu / 2, nv / 2)];
    let mut edge = |mesh: &mut CpuMesh, i0: usize, j0: usize, i1: usize, j1: usize| {
        let a = id(i0, j0);
        let b = id(i1, j1);
        let panel = edge_panel(i0, j0, i1, j1, nu, nv);
        let pa = p[a];
        let pb = p[b];
        let qa = pa - n[a] * thick;
        let qb = pb - n[b] * thick;
        let ua = [i0 as f32 / nu as f32, j0 as f32 / nv as f32];
        let ub = [i1 as f32 / nu as f32, j1 as f32 / nv as f32];
        let mid = (pa + pb) * 0.5;
        let mut face = (pb - pa).cross(qb - pa);
        if face.length_squared() < 1.0e-12 {
            return;
        }
        face = face.normalize();
        if face.dot(mid - center) < 0.0 {
            add_tri(mesh, pa, qb, pb, -face, -face, -face, ua, ub, ub, panel);
            add_tri(mesh, pa, qa, qb, -face, -face, -face, ua, ua, ub, panel);
        } else {
            add_tri(mesh, pa, pb, qb, face, face, face, ua, ub, ub, panel);
            add_tri(mesh, pa, qb, qa, face, face, face, ua, ub, ua, panel);
        }
    };
    for i in 0..nu {
        edge(mesh, i, 0, i + 1, 0);
        edge(mesh, i, nv, i + 1, nv);
    }
    for j in 0..nv {
        edge(mesh, 0, j, 0, j + 1);
        edge(mesh, nu, j, nu, j + 1);
    }
}

fn emit_teeth(mesh: &mut CpuMesh, nu: usize, nv: usize, p: &[Vec3], n: &[Vec3]) {
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    for i in (0..nu).step_by(3) {
        let a = p[id(i, nv)];
        let b = p[id((i + 2).min(nu), nv)];
        let sheet = n[id(i, nv)];
        let prev = p[id(i, nv.saturating_sub(1))];
        let forward = (a - prev).normalize_or_zero();
        let len = 0.055 + ((i * 3) % 4) as f32 * 0.022;
        let side = ((i % 5) as f32 - 2.0) * 0.008;
        let tip = (a + b) * 0.5 + forward * len * 0.35 + Vec3::new(side, -len * 0.85, 0.008);
        let uv = [i as f32 / nu as f32, 1.02];
        add_tab(mesh, a, b, tip, sheet, uv);
    }
}

fn emit_corner_tabs(mesh: &mut CpuMesh, nu: usize, nv: usize, p: &[Vec3], n: &[Vec3]) {
    let id = |i: usize, j: usize| j * (nu + 1) + i;
    for j in [0usize, 1] {
        let a = p[id(nu, j)];
        let b = p[id(nu, (j + 1).min(nv))];
        let sheet = n[id(nu, j)];
        let len = 0.035 + j as f32 * 0.02;
        let tip = (a + b) * 0.5 + Vec3::new(-0.02, -len, 0.015);
        add_tab(mesh, a, b, tip, sheet, [1.0, j as f32 / nv as f32]);
    }
}

fn add_tab(mesh: &mut CpuMesh, a: Vec3, b: Vec3, tip: Vec3, sheet: Vec3, uv: [f32; 2]) {
    let face = (b - a).cross(tip - a);
    if face.length_squared() < 1.0e-12 {
        return;
    }
    // Keep the lighting normal with the winding. A small pull toward the sheet
    // stops a dangling tab from turning its back on the sun.
    let face_n = face.normalize();
    let lit = (face_n + sheet.normalize_or_zero() * 0.35).normalize_or_zero();
    let n = if lit.length_squared() < 1.0e-8 {
        face_n
    } else {
        lit
    };
    add_tri(mesh, a, b, tip, n, n, n, uv, uv, uv, BARE);
    add_tri(mesh, a, tip, b, -n, -n, -n, uv, uv, uv, BARE);
}

fn add_chip(mesh: &mut CpuMesh, c: Vec3, n: Vec3, t: Vec3, scale: f32) {
    let n = n.normalize_or_zero();
    if n.length_squared() < 1.0e-8 {
        return;
    }
    let t = {
        let t = t - n * t.dot(n);
        if t.length_squared() < 1.0e-8 {
            Vec3::X
        } else {
            t.normalize()
        }
    };
    let b = n.cross(t);
    let lift = n * 0.005;
    let a = c + lift - t * scale * 0.55 - b * scale * 0.22;
    let d = c + lift + t * scale * 0.85 - b * scale * 0.12;
    let e = c + lift + t * scale * 0.15 + b * scale * 0.48;
    let f = c + lift - t * scale * 0.15 + b * scale * 0.05 + n * 0.001;
    let uv = [0.5, 0.5];
    add_tri(mesh, a, d, f, n, n, n, uv, uv, uv, BARE);
    add_tri(mesh, a, f, e, n, n, n, uv, uv, uv, BARE);
}

fn add_clod(mesh: &mut CpuMesh, c: Vec3, up: Vec3, r: f32) {
    let up = {
        let up = up.normalize_or_zero();
        if up.length_squared() < 1.0e-8 {
            Vec3::Y
        } else {
            up
        }
    };
    let helper = if up.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
    let t = up.cross(helper).normalize();
    let b = up.cross(t).normalize();
    let h = r * 0.42;
    let s = [
        c + up * h,
        c + t * r * 1.05 - up * h * 0.15,
        c - t * r * 0.7 + b * r * 0.75 - up * h * 0.1,
        c - b * r * 0.9 - up * h * 0.08,
        c + t * r * 0.35 + b * r * 0.5 + up * h * 0.12,
    ];
    add_tri(
        mesh,
        s[0],
        s[1],
        s[4],
        up,
        up,
        up,
        [0.2, 0.8],
        [0.8, 0.8],
        [0.5, 0.2],
        CLOD,
    );
    add_tri(
        mesh,
        s[0],
        s[4],
        s[2],
        up,
        up,
        up,
        [0.2, 0.8],
        [0.5, 0.2],
        [0.1, 0.3],
        CLOD,
    );
    add_tri(
        mesh,
        s[0],
        s[2],
        s[3],
        up,
        up,
        up,
        [0.2, 0.8],
        [0.1, 0.3],
        [0.7, 0.2],
        CLOD,
    );
    add_tri(
        mesh,
        s[0],
        s[3],
        s[1],
        up,
        up,
        up,
        [0.2, 0.8],
        [0.7, 0.2],
        [0.8, 0.8],
        CLOD,
    );
    add_tri(
        mesh,
        s[1],
        s[3],
        s[2],
        -up,
        -up,
        -up,
        [0.8, 0.8],
        [0.7, 0.2],
        [0.1, 0.3],
        CLOD,
    );
    add_tri(
        mesh,
        s[1],
        s[2],
        s[4],
        -up,
        -up,
        -up,
        [0.8, 0.8],
        [0.1, 0.3],
        [0.5, 0.2],
        CLOD,
    );
}

fn add_tri(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    na: Vec3,
    nb: Vec3,
    nc: Vec3,
    ua: [f32; 2],
    ub: [f32; 2],
    uc: [f32; 2],
    panel: f32,
) {
    if (b - a).cross(c - a).length_squared() < 1.0e-12 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    for (p, nrm, uv) in [(a, na, ua), (b, nb, ub), (c, nc, uc)] {
        let nrm = {
            let n = nrm.normalize_or_zero();
            if n.length_squared() < 1.0e-8 {
                Vec3::Y
            } else {
                n
            }
        };
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: nrm.to_array(),
            uv,
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit() -> BodyDamage {
        let mut d = BodyDamage::new();
        d.note_contact(6.0);
        d
    }

    fn fits(mesh: &CpuMesh) {
        assert!(
            mesh.vertices.len() <= MAX_VERTS,
            "verts {} exceed {MAX_VERTS}",
            mesh.vertices.len()
        );
        assert!(
            mesh.indices.len() <= MAX_INDEX,
            "indices {} exceed {MAX_INDEX}",
            mesh.indices.len()
        );
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
        fits(&dents);
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
        fits(&loose);
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

        let lip_y = loose
            .vertices
            .iter()
            .filter(|v| (v.panel - BARE).abs() < 0.01)
            .map(|v| v.pos[1])
            .fold(f32::MAX, f32::min);
        assert!(lip_y < -0.05, "torn edge does not hang, {lip_y}");
    }

    #[test]
    fn sheets_keep_a_panel_per_face_and_honest_normals() {
        let d = hit();
        for mesh in [dent_mesh(&d), loose_mesh(&d)] {
            assert_eq!(mesh.indices.len() % 3, 0);
            let mut fought = 0;
            let mut faces = 0;
            for tri in mesh.indices.chunks(3) {
                let v = |k: u32| &mesh.vertices[k as usize];
                let p0 = v(tri[0]).panel;
                assert!((p0 - v(tri[1]).panel).abs() < 1.0e-4);
                assert!((p0 - v(tri[2]).panel).abs() < 1.0e-4);
                let a = Vec3::from_array(v(tri[0]).pos);
                let b = Vec3::from_array(v(tri[1]).pos);
                let c = Vec3::from_array(v(tri[2]).pos);
                let face = (b - a).cross(c - a);
                if face.length_squared() < 1.0e-12 {
                    continue;
                }
                faces += 1;
                let na = Vec3::from_array(v(tri[0]).nrm);
                if na.dot(face.normalize()) < 0.15 {
                    fought += 1;
                }
            }
            assert!(faces > 0);
            let frac = fought as f32 / faces as f32;
            assert!(frac < 0.08, "normals fight the visible face ({frac})");
        }
    }

    #[test]
    fn loose_sheet_uvs_cover_the_panel() {
        let loose = loose_mesh(&hit());
        let skin: Vec<_> = loose
            .vertices
            .iter()
            .filter(|v| (v.panel - SKIN).abs() < 0.01)
            .collect();
        assert!(skin.len() > 20);
        let umin = skin.iter().map(|v| v.uv[0]).fold(f32::MAX, f32::min);
        let umax = skin.iter().map(|v| v.uv[0]).fold(f32::MIN, f32::max);
        let vmin = skin.iter().map(|v| v.uv[1]).fold(f32::MAX, f32::min);
        let vmax = skin.iter().map(|v| v.uv[1]).fold(f32::MIN, f32::max);
        assert!(umax - umin > 0.7, "u span {}", umax - umin);
        assert!(vmax - vmin > 0.7, "v span {}", vmax - vmin);
    }
}
