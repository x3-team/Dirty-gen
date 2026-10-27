//! Thin grass blades, round woody twigs, and wet mud clods for bits stuck
//! on the buggy. Vertices are already in world space. The vehicle meshes
//! are not touched.
//!
//! A square stick or a faceted ball reads as a gray block under the chassis.
//! Blades stay a millimetre thick, twigs stay a few millimetres across, and
//! a clod is several smooth lobes. When its hold runs out, the small lobes
//! pull off so the cake looks like it is shedding.

use glam::{Mat4, Quat, Vec3};
use silt::debris::{Bit, Debris, KIND_CLOD, KIND_GRASS, KIND_TWIG, MOUNT_FREE};
use silt::vehicle::Vehicle;

use crate::mesh::{CpuMesh, MeshVertex};

const PANEL_GRASS: f32 = 47.0;
const PANEL_TWIG: f32 = 48.0;
/// Stuck wet mud. Impact clods stay on panel 46.
const PANEL_STUCK: f32 = 49.0;
/// Renderer debris buffers are 8192 vertices and 24000 indices.
const MAX_VERTS: usize = 7900;
const MAX_INDEX: usize = 22000;

pub fn debris_mesh(vehicle: &Vehicle, debris: &Debris) -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for bit in debris.bits() {
        if !bit.alive {
            continue;
        }
        let (mut pos, axis, side) = bit.frame(vehicle);
        if axis.length_squared() < 1.0e-6 {
            continue;
        }
        // Sit the piece on the outside of the tread or panel. A tangent axis
        // would leave the clod buried in the tire.
        let out = surface_out(vehicle, bit, pos);
        let lift = match bit.kind {
            KIND_CLOD => bit.length * 0.42,
            KIND_TWIG => 0.006,
            _ => 0.004,
        };
        pos += out * lift;
        if bit.mount == MOUNT_FREE && bit.grounded && bit.kind == KIND_CLOD {
            pos += Vec3::Y * bit.length * 0.2;
        }
        let mut piece = CpuMesh {
            vertices: Vec::new(),
            indices: Vec::new(),
        };
        match bit.kind {
            KIND_GRASS => add_grass(&mut piece, bit, pos, axis, side),
            KIND_TWIG => add_twig(&mut piece, bit, pos, axis, side),
            KIND_CLOD => add_clod(&mut piece, bit, pos, axis, side),
            _ => {}
        }
        if !append(&mut mesh, piece) {
            break;
        }
    }
    mesh
}

fn append(mesh: &mut CpuMesh, piece: CpuMesh) -> bool {
    if piece.vertices.is_empty() {
        return true;
    }
    if mesh.vertices.len() + piece.vertices.len() > MAX_VERTS
        || mesh.indices.len() + piece.indices.len() > MAX_INDEX
    {
        return false;
    }
    let base = mesh.vertices.len() as u32;
    mesh.vertices.extend(piece.vertices);
    mesh.indices
        .extend(piece.indices.into_iter().map(|i| i + base));
    true
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
    let axis = axis.normalize_or_zero();
    if axis.length_squared() < 1.0e-8 {
        return;
    }
    let side = reorth(side, axis);
    let width = (bit.length * 0.034).clamp(0.0040, 0.0110);
    let thick = (width * 0.11).clamp(0.00045, 0.0009);
    // A few stalks leave one root, the way grass sticks out of a clod.
    let fan = [
        (0.25_f32, 1.0_f32, 1.0_f32, 0.12_f32),
        (1.35, 0.74, 0.62, 0.48),
        (-1.15, 0.58, 0.55, -0.42),
    ];
    for (k, (ang, len_k, wid_k, lean)) in fan.iter().enumerate() {
        let q = Quat::from_axis_angle(axis, bit.twist * 0.35 + ang);
        let side_k = q * side;
        let axis_k = (axis + side_k * lean).normalize_or_zero();
        let dry = if hash(bit.seed, k as u32 + 3) > 0.74 {
            0.86
        } else {
            hash(bit.seed, 20 + k as u32) * 0.22
        };
        let root = pos + side_k * width * (k as f32 - 1.0) * 0.55;
        add_blade(
            mesh,
            root,
            axis_k,
            side_k,
            bit.length * len_k,
            width * wid_k,
            thick,
            bit.twist + k as f32 * 1.7,
            dry,
        );
    }
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
    dry: f32,
) {
    if length < 1.0e-3 || width < 1.0e-4 {
        return;
    }
    let axis = axis.normalize_or_zero();
    let side = reorth(side, axis);
    let up0 = axis.cross(side).normalize_or_zero();
    let samples = 4;
    let mut center = [Vec3::ZERO; 4];
    let mut side_s = [Vec3::X; 4];
    let mut up_s = [Vec3::Z; 4];
    let mut half_w = [0.0_f32; 4];
    let mut half_t = [0.0_f32; 4];
    let mut v = [0.0_f32; 4];
    for i in 0..samples {
        let t = i as f32 / (samples - 1) as f32;
        let bow = t * t;
        // A blade arcs. A straight extrusion reads as a wire or a rod.
        center[i] = pos
            + axis * (length * t)
            + Vec3::Y * (-length * 0.42 * bow)
            + side * (twist.sin() * length * 0.30 * t)
            + up0 * (twist.cos() * length * 0.38 * bow);
        v[i] = t;
        // Widest at the base, a point at the tip. Real blades are not bars.
        let belly = (1.0 - t).powf(0.55);
        half_w[i] = (width * 0.5 * belly).max(0.00025);
        half_t[i] = thick * 0.5 * (1.0 - t * 0.45);
    }
    for i in 0..samples {
        let dir = if i + 1 < samples {
            center[i + 1] - center[i]
        } else {
            center[i] - center[i - 1]
        };
        let tangent_i = if dir.length_squared() < 1.0e-10 {
            axis
        } else {
            dir.normalize()
        };
        let spin = Quat::from_axis_angle(tangent_i, twist * 0.2 + v[i] * 2.1);
        let side_i = reorth(spin * side, tangent_i);
        let up_i = tangent_i.cross(side_i).normalize_or_zero();
        side_s[i] = side_i;
        up_s[i] = up_i;
    }
    for i in 0..samples - 1 {
        let a = station(center[i], side_s[i], up_s[i], half_w[i], half_t[i]);
        let b = station(
            center[i + 1],
            side_s[i + 1],
            up_s[i + 1],
            half_w[i + 1],
            half_t[i + 1],
        );
        let inside = (center[i] + center[i + 1]) * 0.5;
        let va = v[i];
        let vb = v[i + 1];
        // Front, back, and the two thin edges. The face is the blade;
        // the edge is about a millimetre so it does not vanish edge-on.
        quad(
            mesh,
            [a[0], a[1], b[1], b[0]],
            [[dry, va], [dry, va], [dry, vb], [dry, vb]],
            PANEL_GRASS,
            inside,
        );
        quad(
            mesh,
            [a[3], a[2], b[2], b[3]],
            [[dry, va], [dry, va], [dry, vb], [dry, vb]],
            PANEL_GRASS,
            inside,
        );
        quad(
            mesh,
            [a[0], b[0], b[3], a[3]],
            [[dry, va], [dry, vb], [dry, vb], [dry, va]],
            PANEL_GRASS,
            inside,
        );
        quad(
            mesh,
            [a[1], a[2], b[2], b[1]],
            [[dry, va], [dry, va], [dry, vb], [dry, vb]],
            PANEL_GRASS,
            inside,
        );
    }
}

/// Front-left, front-right, back-right, back-left.
fn station(center: Vec3, side: Vec3, up: Vec3, half_w: f32, half_t: f32) -> [Vec3; 4] {
    let s = side * half_w;
    let u = up * half_t;
    [
        center + s + u,
        center - s + u,
        center - s - u,
        center + s - u,
    ]
}

fn add_twig(mesh: &mut CpuMesh, bit: &Bit, pos: Vec3, axis: Vec3, side: Vec3) {
    let axis = axis.normalize_or_zero();
    let side = reorth(side, axis);
    let up = axis.cross(side).normalize_or_zero();
    // A few millimetres across. The old stick was a 4 cm square bar.
    let radius = (bit.length * 0.009).clamp(0.0016, 0.0042);
    let curl = (hash(bit.seed, 5) - 0.35) * 1.4 + bit.twist.sin() * 0.6;
    let bend = (up * curl + side * (hash(bit.seed, 6) - 0.5) * 0.8) * (bit.length * 0.22);
    let p1 = pos + axis * (bit.length * 0.36) + bend * 0.28;
    let p2 = pos + axis * (bit.length * 0.70) + bend * 0.72;
    let tip = pos + axis * bit.length + bend;
    tube(
        mesh,
        &[pos, p1, p2, tip],
        &[
            radius * 1.05,
            radius * 1.7,
            radius * 0.70,
            radius * 0.26,
        ],
        0.0,
        1.0,
        PANEL_TWIG,
        7,
        bit.seed,
    );
    let branch_dir = (axis * 0.45 + side * 0.85 + up * 0.22 * bit.twist.cos()).normalize_or_zero();
    let f1 = p1 + branch_dir * (bit.length * 0.20);
    let f2 = p1 + branch_dir * (bit.length * 0.40) + up * (bit.length * 0.03);
    tube(
        mesh,
        &[p1, f1, f2],
        &[radius * 0.52, radius * 0.36, radius * 0.14],
        0.42,
        0.96,
        PANEL_TWIG,
        6,
        bit.seed.wrapping_add(9),
    );
}

fn add_clod(mesh: &mut CpuMesh, bit: &Bit, pos: Vec3, axis: Vec3, side: Vec3) {
    let axis = axis.normalize_or_zero();
    let side = reorth(side, axis);
    let bin = axis.cross(side).normalize_or_zero();
    let radius = bit.length.max(0.018);
    let shift = shed_shift(bit, side);
    // Overlapping lumps, not a pancake. The small ones crack off as hold fails.
    lobe(
        mesh,
        pos,
        axis,
        side,
        bin,
        radius,
        Vec3::new(1.08, 0.86, 0.96),
        bit.seed,
        5,
        10,
        0.0,
    );
    lobe(
        mesh,
        pos + side * radius * 1.05 + axis * radius * 0.18 + shift,
        axis,
        side,
        bin,
        radius * 0.62,
        Vec3::new(0.92, 1.05, 0.88),
        bit.seed.wrapping_add(4),
        4,
        8,
        0.45,
    );
    lobe(
        mesh,
        pos - side * radius * 0.15 + bin * radius * 0.95 + axis * radius * 0.1 + shift * 1.45,
        axis,
        side,
        bin,
        radius * 0.40,
        Vec3::new(1.12, 0.78, 1.0),
        bit.seed.wrapping_add(8),
        3,
        7,
        0.82,
    );
}

/// How far a side lobe has cracked off. Fresh mud stays one cake.
fn shed_shift(bit: &Bit, side: Vec3) -> Vec3 {
    let gap = if bit.mount == MOUNT_FREE {
        if bit.grounded {
            0.01
        } else {
            0.055
        }
    } else if bit.hold < 0.30 {
        (0.30 - bit.hold) * 0.18
    } else {
        0.0
    };
    if gap <= 0.0 {
        Vec3::ZERO
    } else {
        Vec3::Y * -gap + side * gap * 0.7
    }
}

fn lobe(
    mesh: &mut CpuMesh,
    center: Vec3,
    axis: Vec3,
    side: Vec3,
    bin: Vec3,
    radius: f32,
    squash: Vec3,
    seed: u32,
    stacks: usize,
    slices: usize,
    uv_bias: f32,
) {
    if radius < 1.0e-4 || stacks < 2 || slices < 3 {
        return;
    }
    let mut pts = Vec::with_capacity(2 + (stacks - 1) * slices);
    let mut nrm = Vec::with_capacity(pts.capacity());
    let mut crown = Vec::with_capacity(pts.capacity());
    let place = |dir: Vec3| {
        let scale = dir_scale(dir, seed);
        let shaped = side * dir.x * radius * squash.x * scale
            + axis * dir.y * radius * squash.y * scale
            + bin * dir.z * radius * squash.z * scale;
        let n = side * dir.x / squash.x.max(0.05)
            + axis * dir.y / squash.y.max(0.05)
            + bin * dir.z / squash.z.max(0.05);
        let crown = ((scale - 0.66) / 0.56).clamp(0.0, 1.0);
        (center + shaped, n.normalize_or_zero(), crown)
    };
    let (p, n, c) = place(Vec3::Y);
    pts.push(p);
    nrm.push(n);
    crown.push(c);
    for s in 1..stacks {
        let phi = s as f32 / stacks as f32 * std::f32::consts::PI;
        for i in 0..slices {
            let th = i as f32 / slices as f32 * std::f32::consts::TAU;
            let dir = Vec3::new(th.cos() * phi.sin(), phi.cos(), th.sin() * phi.sin());
            let (p, n, c) = place(dir);
            pts.push(p);
            nrm.push(n);
            crown.push(c);
        }
    }
    let (p, n, c) = place(-Vec3::Y);
    pts.push(p);
    nrm.push(n);
    crown.push(c);

    let base = mesh.vertices.len() as u32;
    for i in 0..pts.len() {
        mesh.vertices.push(MeshVertex {
            pos: pts[i].to_array(),
            nrm: nrm[i].to_array(),
            uv: [uv_bias, crown[i]],
            panel: PANEL_STUCK,
        });
    }
    let ring = |s: usize, i: usize| base + 1 + ((s - 1) * slices + (i % slices)) as u32;
    let id = |local: usize| base + local as u32;
    for i in 0..slices {
        tri_idx(mesh, id(0), ring(1, i), ring(1, i + 1), center);
    }
    for s in 1..stacks.saturating_sub(1) {
        for i in 0..slices {
            let a = ring(s, i);
            let b = ring(s, i + 1);
            let c = ring(s + 1, i);
            let d = ring(s + 1, i + 1);
            tri_idx(mesh, a, c, b, center);
            tri_idx(mesh, b, c, d, center);
        }
    }
    let bot = id(pts.len() - 1);
    let last = stacks - 1;
    for i in 0..slices {
        tri_idx(mesh, bot, ring(last, i + 1), ring(last, i), center);
    }
}

fn dir_scale(dir: Vec3, seed: u32) -> f32 {
    let spin = hash(seed, 2) * std::f32::consts::TAU;
    let c = spin.cos();
    let s = spin.sin();
    let x = dir.x * c - dir.z * s;
    let z = dir.x * s + dir.z * c;
    let wave = (x * 2.15).sin() * (z * 1.65).cos() * 0.20 + (dir.y * 2.4).sin() * 0.12;
    (0.84 + wave).clamp(0.66, 1.22)
}

fn tube(
    mesh: &mut CpuMesh,
    pts: &[Vec3],
    radii: &[f32],
    v0: f32,
    v1: f32,
    panel: f32,
    sides: usize,
    seed: u32,
) {
    if pts.len() < 2 || pts.len() != radii.len() || sides < 3 {
        return;
    }
    let mut side = reorth(Vec3::Y, (pts[1] - pts[0]).normalize_or_zero());
    if side.length_squared() < 1.0e-8 {
        side = Vec3::X;
    }
    let stations = pts.len();
    let mut frames: Vec<(Vec3, Vec3)> = Vec::with_capacity(stations);
    for i in 0..stations {
        let dir = if i + 1 < stations {
            pts[i + 1] - pts[i]
        } else {
            pts[i] - pts[i - 1]
        };
        let dir = if dir.length_squared() < 1.0e-10 {
            Vec3::Y
        } else {
            dir.normalize()
        };
        side = reorth(side, dir);
        let up = dir.cross(side).normalize_or_zero();
        frames.push((side, up));
    }
    let base = mesh.vertices.len() as u32;
    for s in 0..stations {
        let v = v0 + (v1 - v0) * (s as f32 / (stations - 1) as f32);
        let (side, up) = frames[s];
        for i in 0..sides {
            let ang = i as f32 / sides as f32 * std::f32::consts::TAU;
            let radial = side * ang.cos() + up * ang.sin();
            let ridge = if i == 0 { 1.28 } else { 1.0 };
            let wobble = (0.78 + hash(seed, (s * 16 + i) as u32) * 0.36) * ridge;
            let p = pts[s] + radial * radii[s] * wobble;
            mesh.vertices.push(MeshVertex {
                pos: p.to_array(),
                nrm: radial.normalize_or_zero().to_array(),
                uv: [i as f32 / sides as f32, v],
                panel,
            });
        }
    }
    let at = |s: usize, i: usize| base + (s * sides + (i % sides)) as u32;
    for s in 0..stations - 1 {
        let center = (pts[s] + pts[s + 1]) * 0.5;
        for i in 0..sides {
            tri_idx(mesh, at(s, i), at(s + 1, i), at(s, i + 1), center);
            tri_idx(mesh, at(s, i + 1), at(s + 1, i), at(s + 1, i + 1), center);
        }
    }
}

fn quad(mesh: &mut CpuMesh, corners: [Vec3; 4], uvs: [[f32; 2]; 4], panel: f32, inside: Vec3) {
    let [a, b, c, d] = corners;
    let face = (b - a).cross(c - a);
    if face.length_squared() < 1.0e-12 {
        return;
    }
    let outward = if face.dot(a - inside) >= 0.0 {
        face.normalize()
    } else {
        -face.normalize()
    };
    let base = mesh.vertices.len() as u32;
    for (p, uv) in [(a, uvs[0]), (b, uvs[1]), (c, uvs[2]), (d, uvs[3])] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: outward.to_array(),
            uv,
            panel,
        });
    }
    if face.dot(a - inside) >= 0.0 {
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    } else {
        mesh.indices
            .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
    }
}

fn tri_idx(mesh: &mut CpuMesh, i0: u32, i1: u32, i2: u32, center: Vec3) {
    let a = Vec3::from_array(mesh.vertices[i0 as usize].pos);
    let b = Vec3::from_array(mesh.vertices[i1 as usize].pos);
    let c = Vec3::from_array(mesh.vertices[i2 as usize].pos);
    let face = (b - a).cross(c - a);
    if face.length_squared() < 1.0e-12 {
        return;
    }
    if face.dot(a - center) >= 0.0 {
        mesh.indices.extend_from_slice(&[i0, i1, i2]);
    } else {
        mesh.indices.extend_from_slice(&[i0, i2, i1]);
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use silt::debris::{MOUNT_BODY, MOUNT_FREE};
    use silt::soil::{SoilField, SoilSample};
    use silt::vehicle::{Contact, Vehicle};

    fn bit(kind: u8, length: f32, hold: f32, mount: u8) -> Bit {
        Bit {
            kind,
            mount,
            local: Vec3::ZERO,
            axis: Vec3::Y,
            side: Vec3::X,
            length,
            twist: 0.4,
            hold,
            alive: true,
            seed: 11,
            pos: Vec3::ZERO,
            vel: Vec3::ZERO,
            grounded: false,
        }
    }

    fn mesh_of(bit: Bit) -> CpuMesh {
        let mut debris = Debris::new();
        debris.place(bit);
        debris_mesh(&Vehicle::new(), &debris)
    }

    fn tri_altitude(mesh: &CpuMesh, panel: f32) -> (f32, f32) {
        let mut max_a = 0.0_f32;
        let mut longest = 0.0_f32;
        for tri in mesh.indices.chunks_exact(3) {
            let va = mesh.vertices[tri[0] as usize];
            if (va.panel - panel).abs() > 0.1 {
                continue;
            }
            let a = Vec3::from_array(va.pos);
            let b = Vec3::from_array(mesh.vertices[tri[1] as usize].pos);
            let c = Vec3::from_array(mesh.vertices[tri[2] as usize].pos);
            let ab = (b - a).length();
            let bc = (c - b).length();
            let ca = (a - c).length();
            let longest_edge = ab.max(bc).max(ca);
            if longest_edge < 1.0e-6 {
                continue;
            }
            longest = longest.max(longest_edge);
            let area = (b - a).cross(c - a).length() * 0.5;
            max_a = max_a.max(2.0 * area / longest_edge);
        }
        (max_a, longest)
    }

    #[test]
    fn grass_blades_are_thin() {
        let mesh = mesh_of(bit(KIND_GRASS, 0.32, 0.8, MOUNT_BODY));
        let (altitude, longest) = tri_altitude(&mesh, PANEL_GRASS);
        assert!(
            altitude < 0.014,
            "blade cross-section {altitude:.4} m is a bar, not a blade"
        );
        assert!(longest > 0.05, "blade span {longest:.3}");
        let ys: Vec<f32> = mesh
            .vertices
            .iter()
            .filter(|v| (v.panel - PANEL_GRASS).abs() < 0.1)
            .map(|v| v.pos[1])
            .collect();
        let span = ys.iter().cloned().fold(f32::MIN, f32::max)
            - ys.iter().cloned().fold(f32::MAX, f32::min);
        assert!(span > 0.16, "blade length {span:.3}");
        let mut y0 = 1.0_f32;
        let mut y1 = 0.0_f32;
        for v in &mesh.vertices {
            if (v.panel - PANEL_GRASS).abs() < 0.1 {
                y0 = y0.min(v.uv[1]);
                y1 = y1.max(v.uv[1]);
            }
        }
        assert!(y1 - y0 > 0.7, "root-to-tip uv {y0:.2}..{y1:.2}");
    }

    #[test]
    fn twigs_are_narrow_round_sticks() {
        let mesh = mesh_of(bit(KIND_TWIG, 0.42, 0.8, MOUNT_BODY));
        let (altitude, _) = tri_altitude(&mesh, PANEL_TWIG);
        assert!(
            altitude < 0.012,
            "twig face {altitude:.4} m is a block"
        );
        let ring: Vec<Vec3> = mesh
            .vertices
            .iter()
            .filter(|v| (v.panel - PANEL_TWIG).abs() < 0.1)
            .take(7)
            .map(|v| Vec3::from_array(v.pos))
            .collect();
        assert!(ring.len() == 7, "root ring {}", ring.len());
        let center = ring.iter().copied().sum::<Vec3>() / ring.len() as f32;
        let normal = (ring[1] - ring[0])
            .cross(ring[2] - ring[0])
            .normalize_or_zero();
        let side = reorth(Vec3::Y, normal);
        let up = normal.cross(side).normalize_or_zero();
        let mut around = [false; 12];
        let mut max_r = 0.0_f32;
        for p in &ring {
            let d = *p - center;
            let x = d.dot(side);
            let y = d.dot(up);
            max_r = max_r.max((x * x + y * y).sqrt());
            let ang = y.atan2(x);
            let bin = ((ang + std::f32::consts::PI) / std::f32::consts::TAU * 12.0) as usize;
            around[bin.min(11)] = true;
        }
        assert!(max_r < 0.009, "twig root radius {max_r:.4} m");
        let bins = around.iter().filter(|b| **b).count();
        assert!(
            bins >= 6,
            "round section uses {bins} sectors, a square uses 4"
        );
        let far = mesh
            .vertices
            .iter()
            .filter(|v| (v.panel - PANEL_TWIG).abs() < 0.1)
            .map(|v| (Vec3::from_array(v.pos) - center).length())
            .fold(0.0_f32, f32::max);
        assert!(far > 0.2, "twig length {far:.3}");
    }

    #[test]
    fn clods_are_lumpy_and_shed() {
        let fresh = mesh_of(bit(KIND_CLOD, 0.05, 0.9, MOUNT_BODY));
        let (altitude, _) = tri_altitude(&fresh, PANEL_STUCK);
        assert!(
            altitude > 0.012,
            "clod is a sheet, altitude {altitude:.4}"
        );
        let mut diag = 0usize;
        let mut n = 0usize;
        let mut centroid = Vec3::ZERO;
        for v in &fresh.vertices {
            centroid += Vec3::from_array(v.pos);
            n += 1;
            let normal = Vec3::from_array(v.nrm).abs();
            let axis = normal.x.max(normal.y).max(normal.z);
            if axis < 0.92 {
                diag += 1;
            }
        }
        assert!(n > 40, "clod verts {n}");
        centroid /= n as f32;
        assert!(
            diag * 2 > n,
            "clod normals are axis-aligned like a box ({diag}/{n})"
        );
        let mut shedding = bit(KIND_CLOD, 0.05, 0.02, MOUNT_BODY);
        shedding.seed = 11;
        let shed = mesh_of(shedding);
        let reach = |mesh: &CpuMesh| {
            mesh.vertices
                .iter()
                .map(|v| (Vec3::from_array(v.pos) - centroid).length())
                .fold(0.0_f32, f32::max)
        };
        let grown = reach(&shed) - reach(&fresh);
        assert!(
            grown > 0.012,
            "a weak clod should crack a lobe off, grew {grown:.4} m"
        );
        let mut falling = bit(KIND_CLOD, 0.05, 1.0, MOUNT_FREE);
        falling.seed = 11;
        falling.pos = Vehicle::new().pos;
        falling.axis = Vec3::Y;
        falling.side = Vec3::X;
        let air = mesh_of(falling);
        let air_grown = reach(&air) - reach(&fresh);
        assert!(
            air_grown > 0.02,
            "a falling clod should come apart, grew {air_grown:.4} m"
        );
    }

    #[test]
    fn a_full_coat_fits_the_debris_buffers() {
        let mut debris = Debris::new();
        let mut vehicle = Vehicle::new();
        vehicle.wheels[0].mud = 0.5;
        let mut contacts = [Contact::default(); 4];
        for i in 0..4 {
            contacts[i].active = true;
            contacts[i].index = i;
            contacts[i].sample = SoilSample {
                height: 0.0,
                wetness: 0.9,
                water: 0.04,
                loose: 0.04,
                darken: 0.2,
                roughness: 0.4,
                softness: 0.8,
                rut: 0.02,
            };
        }
        let soil = SoilField::flat(0.9);
        for _ in 0..(4.0 * 60.0) as i32 {
            debris.step(&vehicle, &contacts, &soil, 1.0 / 60.0);
        }
        let mesh = debris_mesh(&vehicle, &debris);
        assert!(mesh.vertices.len() <= MAX_VERTS, "verts {}", mesh.vertices.len());
        assert!(mesh.indices.len() <= MAX_INDEX, "indices {}", mesh.indices.len());
        assert!(mesh.vertices.iter().any(|v| (v.panel - PANEL_GRASS).abs() < 0.1));
        assert!(mesh.vertices.iter().any(|v| (v.panel - PANEL_TWIG).abs() < 0.1));
        assert!(mesh.vertices.iter().any(|v| (v.panel - PANEL_STUCK).abs() < 0.1));
        assert!(mesh.vertices.iter().all(|v| v.pos.iter().all(|c| c.is_finite())));
        assert!(mesh.vertices.iter().all(|v| v.nrm.iter().all(|c| c.is_finite())));
    }
}
