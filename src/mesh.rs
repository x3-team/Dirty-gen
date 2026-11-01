//! Procedural buggy, wheels and a small patch of countryside.
//!
//! Geometry is flat-shaded and carries a panel id. The shader turns that id
//! into paint, rubber or mud. Wheel and shock meshes stay in the spaces the
//! vehicle solver already uses, so the picture matches the contact model.

use glam::{Quat, Vec3};

use silt::bike::Bike;
use silt::vehicle::{hardpoints, Vehicle, WHEEL_RADIUS, WHEEL_WIDTH};

pub const PANEL_LIGHT: f32 = 16.0;
pub const PANEL_TAIL: f32 = 17.0;
pub const PANEL_SKIRT: f32 = 40.0;
pub const PANEL_HILL: f32 = 41.0;
#[allow(dead_code)]
pub const PANEL_ROCK: f32 = 42.0;
#[allow(dead_code)]
pub const PANEL_REED: f32 = 43.0;
pub const PANEL_TREE: f32 = 44.0;
pub const PANEL_TRUNK: f32 = 45.0;
/// Wet mud clod. Not body paint, and not a camera-facing card.
pub const PANEL_CLOD: f32 = 46.0;
/// Dry dust lofted off the track. A pale film, not wet mud.
pub const PANEL_DUST: f32 = 47.0;
/// Water fleck. Not a clod and not body paint.
pub const PANEL_SPRAY: f32 = 48.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    pub nrm: [f32; 3],
    pub uv: [f32; 2],
    pub panel: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainVertex {
    pub xz: [f32; 2],
    pub ij: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ParticleVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

pub struct CpuMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl CpuMesh {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }
}

fn uv_of(p: Vec3) -> [f32; 2] {
    [p.x * 1.6 + p.z * 0.2, p.y * 1.35 + 0.55]
}

fn add_tri(mesh: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3, panel: f32) {
    let n = (b - a).cross(c - a).normalize_or_zero();
    if n.length_squared() < 1.0e-8 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    for p in [a, b, c] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv: uv_of(p),
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn add_quad(mesh: &mut CpuMesh, p: [Vec3; 4], panel: f32) {
    add_tri(mesh, p[0], p[1], p[2], panel);
    add_tri(mesh, p[0], p[2], p[3], panel);
}

fn add_quad_uv(mesh: &mut CpuMesh, p: [Vec3; 4], uv: [[f32; 2]; 4], panel: f32) {
    let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
    if n.length_squared() < 1.0e-8 {
        return;
    }
    let push = |mesh: &mut CpuMesh, pos: Vec3, uv: [f32; 2]| {
        mesh.vertices.push(MeshVertex {
            pos: pos.to_array(),
            nrm: n.to_array(),
            uv,
            panel,
        });
    };
    let base = mesh.vertices.len() as u32;
    push(mesh, p[0], uv[0]);
    push(mesh, p[1], uv[1]);
    push(mesh, p[2], uv[2]);
    push(mesh, p[3], uv[3]);
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn add_box(mesh: &mut CpuMesh, c: Vec3, h: Vec3, panel: f32) {
    let (x, y, z) = (Vec3::X * h.x, Vec3::Y * h.y, Vec3::Z * h.z);
    // +Y
    add_quad(
        mesh,
        [c - x + y - z, c - x + y + z, c + x + y + z, c + x + y - z],
        panel,
    );
    // -Y
    add_quad(
        mesh,
        [c - x - y - z, c + x - y - z, c + x - y + z, c - x - y + z],
        panel,
    );
    // +X
    add_quad(
        mesh,
        [c + x - y - z, c + x + y - z, c + x + y + z, c + x - y + z],
        panel,
    );
    // -X
    add_quad(
        mesh,
        [c - x - y - z, c - x - y + z, c - x + y + z, c - x + y - z],
        panel,
    );
    // +Z
    add_quad(
        mesh,
        [c - x - y + z, c + x - y + z, c + x + y + z, c - x + y + z],
        panel,
    );
    // -Z
    add_quad(
        mesh,
        [c + x - y - z, c - x - y - z, c - x + y - z, c + x + y - z],
        panel,
    );
}

fn add_cone(mesh: &mut CpuMesh, base: Vec3, height: f32, radius: f32, panel: f32, segs: usize) {
    let apex = base + Vec3::Y * height;
    for i in 0..segs {
        let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let p0 = base + Vec3::new(t0.cos(), 0.0, t0.sin()) * radius;
        let p1 = base + Vec3::new(t1.cos(), 0.0, t1.sin()) * radius;
        add_tri(mesh, p0, apex, p1, panel);
        add_tri(mesh, base, p1, p0, panel);
    }
}

fn add_tube(mesh: &mut CpuMesh, a: Vec3, b: Vec3, radius: f32, panel: f32, segs: usize) {
    let axis = b - a;
    let len = axis.length();
    if len < 1.0e-4 {
        return;
    }
    let dir = axis / len;
    let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
    let u = dir.cross(helper).normalize();
    let v = dir.cross(u).normalize();
    for i in 0..segs {
        let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let n0 = u * t0.cos() + v * t0.sin();
        let n1 = u * t1.cos() + v * t1.sin();
        add_quad(
            mesh,
            [
                a + n0 * radius,
                a + n1 * radius,
                b + n1 * radius,
                b + n0 * radius,
            ],
            panel,
        );
    }
}

/// Grooved tire. Local axle is +X, radius matches the solver.
pub fn wheel_mesh() -> CpuMesh {
    let mut mesh = CpuMesh::new();
    let segs = 18;
    // (x, radius) profile. Alternating radii cut tread grooves.
    let profile = [
        (-0.140, 0.30),
        (-0.115, 0.355),
        (-0.085, 0.385),
        (-0.055, 0.348),
        (-0.025, 0.388),
        (0.005, 0.350),
        (0.035, 0.380),
        (0.065, 0.348),
        (0.095, 0.385),
        (0.120, 0.350),
        (0.140, 0.30),
    ];
    let panel = 20.0;
    for w in profile.windows(2) {
        let (x0, r0) = w[0];
        let (x1, r1) = w[1];
        for i in 0..segs {
            let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
            let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
            let scale = WHEEL_RADIUS / 0.390;
            let p = |x: f32, r: f32, t: f32| Vec3::new(x, t.sin() * r * scale, t.cos() * r * scale);
            // +X along the axle, then around the tire, so the normal points outward.
            let quad = [p(x0, r0, t0), p(x1, r1, t0), p(x1, r1, t1), p(x0, r0, t1)];
            let n = (quad[1] - quad[0])
                .cross(quad[2] - quad[0])
                .normalize_or_zero();
            let base = mesh.vertices.len() as u32;
            let uvs = [
                [t0 / std::f32::consts::TAU, 0.0],
                [t0 / std::f32::consts::TAU, 1.0],
                [t1 / std::f32::consts::TAU, 1.0],
                [t1 / std::f32::consts::TAU, 0.0],
            ];
            for (k, p) in quad.iter().enumerate() {
                mesh.vertices.push(MeshVertex {
                    pos: p.to_array(),
                    nrm: n.to_array(),
                    uv: uvs[k],
                    panel,
                });
            }
            mesh.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    add_disc(&mut mesh, -0.140, 0.30, -1.0, segs, panel);
    add_disc(&mut mesh, 0.140, 0.30, 1.0, segs, panel);
    add_disc(&mut mesh, -0.148, 0.12, -1.0, 10, panel);
    add_disc(&mut mesh, 0.148, 0.12, 1.0, 10, panel);
    let _ = WHEEL_RADIUS;
    let _ = WHEEL_WIDTH;
    mesh
}

fn add_disc(mesh: &mut CpuMesh, x: f32, radius: f32, sign: f32, segs: usize, panel: f32) {
    let center = Vec3::new(x, 0.0, 0.0);
    let n = Vec3::new(sign, 0.0, 0.0);
    for i in 0..segs {
        let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let p = |t: f32| center + Vec3::new(0.0, t.sin() * radius, t.cos() * radius);
        let (a, b) = if sign > 0.0 {
            (p(t1), p(t0))
        } else {
            (p(t0), p(t1))
        };
        let base = mesh.vertices.len() as u32;
        for (k, vert) in [center, a, b].iter().enumerate() {
            mesh.vertices.push(MeshVertex {
                pos: vert.to_array(),
                nrm: n.to_array(),
                // uv.y > 1 marks the sidewall so tread mud stays in the grooves.
                uv: [k as f32 / segs as f32, 2.0],
                panel,
            });
        }
        mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
}

/// Coil from `a` to `b`. Length is the distance, so a rising wheel shortens it.
fn add_helix(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    coil_radius: f32,
    tube_radius: f32,
    coils: usize,
    steps: usize,
    panel: f32,
) {
    let axis = b - a;
    let len = axis.length();
    if len < 1.0e-3 || coils == 0 || steps == 0 {
        return;
    }
    let dir = axis / len;
    let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
    let u = dir.cross(helper).normalize();
    let v = dir.cross(u).normalize();
    let n = coils * steps;
    let mut prev = a + u * coil_radius;
    for i in 1..=n {
        let t = i as f32 / n as f32;
        let ang = t * coils as f32 * std::f32::consts::TAU;
        let center = a + axis * t;
        let p = center + (u * ang.cos() + v * ang.sin()) * coil_radius;
        add_tube(mesh, prev, p, tube_radius, panel, 4);
        prev = p;
    }
}

/// Short inboard plate. The tire sits outside it.
fn add_fender_lip(mesh: &mut CpuMesh, sign: f32, z: f32, panel: f32) {
    add_box(
        mesh,
        Vec3::new(sign * 0.40, 0.12, z),
        Vec3::new(0.045, 0.018, 0.16),
        panel,
    );
}

/// Open-wheel buggy in vehicle local space. Y up, Z forward, origin at the CG.
/// The chassis stays inside the tires so the wheels read as separate from above.
pub fn buggy_body() -> CpuMesh {
    let mut m = CpuMesh::new();
    let cage = 0.0;
    let nose = 1.0;
    let left = 2.0;
    let right = 3.0;
    let rear = 4.0;
    let belly = 5.0;
    let engine = 10.0;
    let seat = 11.0;

    // Orange frame rails. Narrow: tires live at x = ±0.74.
    add_box(
        &mut m,
        Vec3::new(-0.26, 0.02, -0.02),
        Vec3::new(0.065, 0.07, 1.28),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.26, 0.02, -0.02),
        Vec3::new(0.065, 0.07, 1.28),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.02, 1.18),
        Vec3::new(0.28, 0.055, 0.06),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.04, -1.12),
        Vec3::new(0.28, 0.055, 0.06),
        rear,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.04, 0.10),
        Vec3::new(0.26, 0.04, 0.055),
        nose,
    );

    // Belly. Impact mud sticks here.
    add_box(
        &mut m,
        Vec3::new(0.0, -0.12, 0.02),
        Vec3::new(0.30, 0.016, 1.00),
        belly,
    );

    // Cockpit sides, inboard of the tires.
    add_box(
        &mut m,
        Vec3::new(-0.34, 0.14, -0.02),
        Vec3::new(0.03, 0.11, 0.42),
        left,
    );
    add_box(
        &mut m,
        Vec3::new(0.34, 0.14, -0.02),
        Vec3::new(0.03, 0.11, 0.42),
        right,
    );

    // Short nose and lamps. No hood box over the front tires.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.06, 1.40),
        Vec3::new(0.20, 0.05, 0.14),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.00, 1.54),
        Vec3::new(0.24, 0.025, 0.035),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(-0.12, 0.08, 1.53),
        Vec3::new(0.045, 0.03, 0.018),
        PANEL_LIGHT,
    );
    add_box(
        &mut m,
        Vec3::new(0.12, 0.08, 1.53),
        Vec3::new(0.045, 0.03, 0.018),
        PANEL_LIGHT,
    );

    // Rear bulkhead.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.16, -1.26),
        Vec3::new(0.26, 0.12, 0.045),
        rear,
    );
    add_box(
        &mut m,
        Vec3::new(-0.16, 0.16, -1.31),
        Vec3::new(0.045, 0.028, 0.014),
        PANEL_TAIL,
    );
    add_box(
        &mut m,
        Vec3::new(0.16, 0.16, -1.31),
        Vec3::new(0.045, 0.028, 0.014),
        PANEL_TAIL,
    );

    // Two seats.
    add_box(
        &mut m,
        Vec3::new(-0.15, 0.16, -0.08),
        Vec3::new(0.11, 0.045, 0.16),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(0.15, 0.16, -0.08),
        Vec3::new(0.11, 0.045, 0.16),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(-0.15, 0.32, -0.22),
        Vec3::new(0.105, 0.15, 0.035),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(0.15, 0.32, -0.22),
        Vec3::new(0.105, 0.15, 0.035),
        seat,
    );

    // Engine between the rear rails.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.18, -0.72),
        Vec3::new(0.16, 0.12, 0.18),
        engine,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.32, -0.68),
        Vec3::new(0.07, 0.04, 0.07),
        engine,
    );

    // Fender lips stay on the frame. The tire is outside them.
    add_fender_lip(&mut m, -1.0, 1.08, 6.0);
    add_fender_lip(&mut m, 1.0, 1.08, 7.0);
    add_fender_lip(&mut m, -1.0, -1.18, 8.0);
    add_fender_lip(&mut m, 1.0, -1.18, 9.0);

    // Roll cage. Radius is large enough to read at the isometric frame.
    let r = 0.048;
    let rl = Vec3::new(-0.32, 0.06, -0.46);
    let rr = Vec3::new(0.32, 0.06, -0.46);
    let rtl = Vec3::new(-0.32, 0.88, -0.40);
    let rtr = Vec3::new(0.32, 0.88, -0.40);
    add_tube(&mut m, rl, rtl, r, cage, 5);
    add_tube(&mut m, rr, rtr, r, cage, 5);
    add_tube(&mut m, rtl, rtr, r, cage, 5);
    let fl = Vec3::new(-0.28, 0.08, 0.62);
    let fr = Vec3::new(0.28, 0.08, 0.62);
    let ftl = Vec3::new(-0.30, 0.74, 0.22);
    let ftr = Vec3::new(0.30, 0.74, 0.22);
    add_tube(&mut m, fl, ftl, 0.044, cage, 5);
    add_tube(&mut m, fr, ftr, 0.044, cage, 5);
    add_tube(&mut m, ftl, ftr, 0.044, cage, 5);
    add_tube(&mut m, ftl, rtl, 0.04, cage, 5);
    add_tube(&mut m, ftr, rtr, 0.04, cage, 5);
    add_tube(&mut m, Vec3::new(-0.28, 0.10, -1.02), rl, 0.038, cage, 4);
    add_tube(&mut m, Vec3::new(0.28, 0.10, -1.02), rr, 0.038, cage, 4);
    add_box(
        &mut m,
        Vec3::new(0.0, 0.86, -0.08),
        Vec3::new(0.24, 0.012, 0.26),
        cage,
    );

    m
}

/// One A-arm and one coil per wheel, in the body frame. Ends track `wheel_center_local`,
/// so the spring shortens when the tire compresses and hangs long at droop.
pub fn suspension_mesh(vehicle: &Vehicle) -> CpuMesh {
    let mut m = CpuMesh::new();
    let hp = hardpoints();
    let arm = 1.0;
    let spring = 10.0;
    for i in 0..4 {
        let wc = vehicle.wheel_center_local(i);
        let sign = hp[i].x.signum();
        let z = hp[i].z;
        let lower_f = Vec3::new(sign * 0.28, -0.02, z + 0.18);
        let lower_r = Vec3::new(sign * 0.28, -0.02, z - 0.18);
        let knuckle = Vec3::new(wc.x - sign * 0.04, wc.y, wc.z);
        add_tube(&mut m, lower_f, knuckle, 0.055, arm, 5);
        add_tube(&mut m, lower_r, knuckle, 0.055, arm, 5);
        let upright_top = knuckle + Vec3::Y * 0.18;
        add_tube(&mut m, knuckle, upright_top, 0.04, spring, 4);
        let tower = Vec3::new(sign * 0.30, 0.50, z);
        let perch = knuckle + Vec3::Y * 0.05;
        // Three fat turns. A thin helix disappears at this camera distance.
        add_helix(&mut m, tower, perch, 0.11, 0.036, 3, 5, spring);
        add_tube(&mut m, tower, perch, 0.022, spring, 4);
    }
    m
}

/// Dirt bike in the same local space as the solver: Y up, Z forward, origin at the CG.
/// Forks and the swingarm live in `bike_suspension` so they follow the wheels.
/// The tubes are fat on purpose: a 3 cm spine disappears at the isometric distance.
pub fn bike_body() -> CpuMesh {
    let mut m = CpuMesh::new();
    let frame = 10.0;
    let tank = 1.0;
    let seat = 11.0;
    let plastic = 4.0;
    let jersey = 2.0;
    // Spine and cradle.
    add_tube(
        &mut m,
        Vec3::new(0.0, 0.02, -0.50),
        Vec3::new(0.0, 0.10, 0.46),
        0.055,
        frame,
        6,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, -0.02, -0.05),
        Vec3::new(0.10, 0.06, 0.18),
        frame,
    );
    // Engine, low, between the wheels.
    add_box(
        &mut m,
        Vec3::new(0.0, -0.04, -0.06),
        Vec3::new(0.16, 0.12, 0.20),
        10.0,
    );
    // Tank. The orange block is what reads from the isometric camera.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.26, 0.16),
        Vec3::new(0.16, 0.10, 0.26),
        tank,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.34, 0.20),
        Vec3::new(0.11, 0.04, 0.16),
        tank,
    );
    // Seat and tail.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.24, -0.16),
        Vec3::new(0.11, 0.05, 0.24),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.20, -0.42),
        Vec3::new(0.07, 0.04, 0.12),
        plastic,
    );
    // Steering head and bar. The fork itself is the moving mesh.
    add_tube(
        &mut m,
        Vec3::new(0.0, 0.18, 0.36),
        Vec3::new(0.0, 0.52, 0.48),
        0.04,
        frame,
        5,
    );
    add_tube(
        &mut m,
        Vec3::new(-0.28, 0.56, 0.30),
        Vec3::new(0.28, 0.56, 0.30),
        0.028,
        frame,
        5,
    );
    // Front number plate and headlight.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.34, 0.58),
        Vec3::new(0.10, 0.08, 0.015),
        0.0,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.22, 0.62),
        Vec3::new(0.05, 0.035, 0.02),
        PANEL_LIGHT,
    );
    // Crouched rider. A bare frame reads as a second buggy at this distance.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.42, -0.02),
        Vec3::new(0.13, 0.12, 0.10),
        jersey,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.58, 0.06),
        Vec3::new(0.09, 0.09, 0.09),
        0.0,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.64, 0.10),
        Vec3::new(0.08, 0.07, 0.08),
        seat,
    );
    add_tube(
        &mut m,
        Vec3::new(-0.10, 0.46, 0.02),
        Vec3::new(-0.22, 0.56, 0.22),
        0.035,
        jersey,
        5,
    );
    add_tube(
        &mut m,
        Vec3::new(0.10, 0.46, 0.02),
        Vec3::new(0.22, 0.56, 0.22),
        0.035,
        jersey,
        5,
    );
    // Belly, where impact mud sticks.
    add_box(
        &mut m,
        Vec3::new(0.0, -0.16, -0.02),
        Vec3::new(0.10, 0.025, 0.30),
        5.0,
    );
    m
}

/// Fork, swingarm and shock. Ends track the bike's wheel centers, so the
/// spring shortens under load and hangs at droop in the air.
pub fn bike_suspension(bike: &Bike) -> CpuMesh {
    let mut m = CpuMesh::new();
    let front = bike.wheel_center_local(0);
    let rear = bike.wheel_center_local(1);
    let steel = 10.0;
    let arm = 1.0;
    let head = Vec3::new(0.0, 0.46, 0.40);
    add_tube(&mut m, head, front, 0.028, steel, 5);
    add_tube(
        &mut m,
        head + Vec3::new(0.045, 0.0, 0.0),
        front + Vec3::new(0.045, 0.02, 0.0),
        0.02,
        steel,
        4,
    );
    add_tube(
        &mut m,
        head + Vec3::new(-0.045, 0.0, 0.0),
        front + Vec3::new(-0.045, 0.02, 0.0),
        0.02,
        steel,
        4,
    );
    let pivot = Vec3::new(0.0, 0.02, -0.12);
    add_tube(&mut m, pivot, rear + Vec3::new(0.0, 0.02, 0.0), 0.03, arm, 5);
    add_tube(
        &mut m,
        pivot + Vec3::new(0.06, 0.0, 0.0),
        rear + Vec3::new(0.05, 0.0, 0.04),
        0.02,
        arm,
        4,
    );
    let tower = Vec3::new(0.0, 0.30, -0.22);
    let perch = rear + Vec3::new(0.0, 0.08, 0.12);
    add_helix(&mut m, tower, perch, 0.055, 0.02, 4, 4, steel);
    add_tube(&mut m, tower, perch, 0.012, steel, 4);
    // Fenders ride with the wheels. A plate above each tyre is what separates
    // the silhouette from the buggy's open wheels.
    add_box(
        &mut m,
        front + Vec3::new(0.0, 0.28, 0.02),
        Vec3::new(0.09, 0.025, 0.20),
        arm,
    );
    add_box(
        &mut m,
        rear + Vec3::new(0.0, 0.30, -0.02),
        Vec3::new(0.10, 0.025, 0.22),
        arm,
    );
    m
}

pub fn terrain_mesh(res: usize, size: f32) -> (Vec<TerrainVertex>, Vec<u32>) {
    // The solver stores one height per ~20 cm. A rut is often a single cell,
    // which is one kink and reads as a tint. Draw a finer mesh and sample the
    // same heightfield so the displaced walls have real triangles.
    // Lug pitch is ~20 cm. At sub 4 a void is one quad and reads as a card.
    let sub = 8usize;
    let n = (res - 1) * sub + 1;
    let mut verts = Vec::with_capacity(n * n);
    let half = size * 0.5;
    let cell = size / res as f32;
    for j in 0..n {
        for i in 0..n {
            let fi = i as f32 / sub as f32;
            let fj = j as f32 / sub as f32;
            let x = (fi + 0.5) * cell - half;
            let z = (fj + 0.5) * cell - half;
            verts.push(TerrainVertex {
                xz: [x, z],
                ij: [fi, fj],
            });
        }
    }
    let mut indices = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = ((j + 1) * n + i) as u32;
            let c = ((j + 1) * n + i + 1) as u32;
            let d = (j * n + i + 1) as u32;
            // +Z then +X winds upward.
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    (verts, indices)
}

fn hash(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747796405).wrapping_add(2891336453);
    x = (x >> 16) ^ x;
    (x & 0x00ff_ffff) as f32 / 16_777_215.0
}

/// Low ring of ground. The inner edge is buried in the skirt so the
/// skyline grows out of the field instead of sitting on floating boxes.
fn add_horizon(mesh: &mut CpuMesh) {
    let segs = 72usize;
    let rings = 7usize;
    let r0 = 26.0;
    let r1 = 58.0;
    let mut crest = vec![0.0_f32; segs];
    for i in 0..segs {
        let a = hash(i as u32 * 17 + 3);
        let b = hash(i as u32 * 17 + 11);
        crest[i] = 2.6 + a * 2.0 + (b - 0.5) * 0.35;
    }
    let src = crest.clone();
    for i in 0..segs {
        let prev = src[(i + segs - 1) % segs];
        let next = src[(i + 1) % segs];
        crest[i] = src[i] * 0.5 + prev * 0.25 + next * 0.25;
    }
    for j in 0..rings {
        let t0 = j as f32 / rings as f32;
        let t1 = (j + 1) as f32 / rings as f32;
        let rad0 = r0 + (r1 - r0) * t0;
        let rad1 = r0 + (r1 - r0) * t1;
        let hump = |t: f32| (t * std::f32::consts::PI).sin();
        for i in 0..segs {
            let i1 = (i + 1) % segs;
            let a0 = i as f32 / segs as f32 * std::f32::consts::TAU;
            let a1 = i1 as f32 / segs as f32 * std::f32::consts::TAU;
            let y_at = |t: f32, c: f32| -0.18 + hump(t) * c;
            let p = |rad: f32, ang: f32, y: f32| Vec3::new(ang.cos() * rad, y, ang.sin() * rad);
            // Inner edge, then the next segment, then the outer edge: +Y normal.
            add_quad(
                mesh,
                [
                    p(rad0, a0, y_at(t0, crest[i])),
                    p(rad0, a1, y_at(t0, crest[i1])),
                    p(rad1, a1, y_at(t1, crest[i1])),
                    p(rad1, a0, y_at(t1, crest[i])),
                ],
                PANEL_HILL,
            );
        }
    }
}

/// Horizon ring so the mud has somewhere to sit.
pub fn environment_mesh() -> CpuMesh {
    let mut m = CpuMesh::new();
    let y = 0.0;
    // Skirt outside the 40 m heightfield. Winding is +Y so it is a floor,
    // not a back face, and the field does not end as a wall.
    let far = 90.0;
    let edge = 19.6;
    let quad = |m: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3, d: Vec3| {
        add_quad(m, [a, b, c, d], PANEL_SKIRT);
    };
    // +Y: a -> b -> c is counter-clockwise when looking down.
    quad(
        &mut m,
        Vec3::new(-far, y, edge),
        Vec3::new(-far, y, far),
        Vec3::new(far, y, far),
        Vec3::new(far, y, edge),
    );
    quad(
        &mut m,
        Vec3::new(-far, y, -far),
        Vec3::new(-far, y, -edge),
        Vec3::new(far, y, -edge),
        Vec3::new(far, y, -far),
    );
    quad(
        &mut m,
        Vec3::new(edge, y, -edge),
        Vec3::new(edge, y, edge),
        Vec3::new(far, y, edge),
        Vec3::new(far, y, -edge),
    );
    quad(
        &mut m,
        Vec3::new(-far, y, -edge),
        Vec3::new(-far, y, edge),
        Vec3::new(-edge, y, edge),
        Vec3::new(-edge, y, -edge),
    );

    m
}

/// Faceted distance hills. Off for a proof that must not end in a wall.
pub fn horizon_mesh() -> CpuMesh {
    let mut m = CpuMesh::new();
    add_horizon(&mut m);
    m
}

fn chunk_hash(seed: u32, n: u32) -> f32 {
    let mut x = seed ^ n.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x & 0xFFFF) as f32 / 65535.0
}

fn push_smooth(mesh: &mut CpuMesh, p: Vec3, n: Vec3, panel: f32) {
    mesh.vertices.push(MeshVertex {
        pos: p.to_array(),
        nrm: n.normalize_or_zero().to_array(),
        uv: uv_of(p),
        panel,
    });
}

fn add_tri_smooth(
    mesh: &mut CpuMesh,
    a: Vec3,
    na: Vec3,
    b: Vec3,
    nb: Vec3,
    c: Vec3,
    nc: Vec3,
    panel: f32,
    center: Vec3,
) {
    let face = (b - a).cross(c - a);
    if face.length_squared() < 1.0e-10 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    if face.dot(a - center) < 0.0 {
        push_smooth(mesh, a, na, panel);
        push_smooth(mesh, c, nc, panel);
        push_smooth(mesh, b, nb, panel);
    } else {
        push_smooth(mesh, a, na, panel);
        push_smooth(mesh, b, nb, panel);
        push_smooth(mesh, c, nc, panel);
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2]);
}

/// Soft mud blob. Normals stay smooth so the silhouette is a clod, not a gem.
/// `slump` flattens a piece that has already landed.
fn add_lump(
    mesh: &mut CpuMesh,
    center: Vec3,
    rot: Quat,
    radius: f32,
    seed: u32,
    stacks: usize,
    slices: usize,
    slump: bool,
) {
    let axis = Vec3::new(
        0.88 + chunk_hash(seed, 1) * 0.22,
        0.80 + chunk_hash(seed, 2) * 0.24,
        0.86 + chunk_hash(seed, 3) * 0.22,
    );
    // Landed mud spreads and sinks. A tall axis reads as a stone sitting on the soil.
    let squash = if slump {
        Vec3::new(1.48, 0.36, 1.48)
    } else {
        Vec3::ONE
    };
    let center = if slump {
        center - Vec3::Y * radius * 0.16
    } else {
        center
    };
    let mut pts = Vec::with_capacity(2 + stacks * slices);
    let mut nrm = Vec::with_capacity(pts.capacity());
    let place = |dir: Vec3, rad: f32| {
        let shaped = Vec3::new(
            dir.x * rad * axis.x * squash.x,
            dir.y * rad * axis.y * squash.y,
            dir.z * rad * axis.z * squash.z,
        );
        let n = Vec3::new(
            dir.x / squash.x.max(0.05),
            dir.y / squash.y.max(0.05),
            dir.z / squash.z.max(0.05),
        );
        (center + rot * shaped, (rot * n).normalize_or_zero())
    };
    let (p, n) = place(Vec3::Y, radius * (0.92 + chunk_hash(seed, 4) * 0.12));
    pts.push(p);
    nrm.push(n);
    for s in 1..stacks {
        let phi = s as f32 / stacks as f32 * std::f32::consts::PI;
        for i in 0..slices {
            let th = i as f32 / slices as f32 * std::f32::consts::TAU;
            let dir = Vec3::new(th.cos() * phi.sin(), phi.cos(), th.sin() * phi.sin());
            // Mild swell. A wide jitter makes spikes, and spikes read as broken rock.
            let j = 0.90 + chunk_hash(seed, (s * 16 + i) as u32) * 0.18;
            let (p, n) = place(dir, radius * j);
            pts.push(p);
            nrm.push(n);
        }
    }
    let (p, n) = place(-Vec3::Y, radius * (0.88 + chunk_hash(seed, 5) * 0.14));
    pts.push(p);
    nrm.push(n);
    let ring = |s: usize, i: usize| 1 + (s - 1) * slices + (i % slices);
    for i in 0..slices {
        let a = ring(1, i);
        let b = ring(1, i + 1);
        add_tri_smooth(
            mesh,
            pts[0],
            nrm[0],
            pts[a],
            nrm[a],
            pts[b],
            nrm[b],
            PANEL_CLOD,
            center,
        );
    }
    for s in 1..stacks.saturating_sub(1) {
        for i in 0..slices {
            let a = ring(s, i);
            let b = ring(s, i + 1);
            let c = ring(s + 1, i);
            let d = ring(s + 1, i + 1);
            add_tri_smooth(
                mesh,
                pts[a],
                nrm[a],
                pts[c],
                nrm[c],
                pts[b],
                nrm[b],
                PANEL_CLOD,
                center,
            );
            add_tri_smooth(
                mesh,
                pts[b],
                nrm[b],
                pts[c],
                nrm[c],
                pts[d],
                nrm[d],
                PANEL_CLOD,
                center,
            );
        }
    }
    let bot = pts.len() - 1;
    let last = stacks - 1;
    for i in 0..slices {
        let a = ring(last, i + 1);
        let b = ring(last, i);
        add_tri_smooth(
            mesh,
            pts[bot],
            nrm[bot],
            pts[a],
            nrm[a],
            pts[b],
            nrm[b],
            PANEL_CLOD,
            center,
        );
    }
}

fn add_piece(mesh: &mut CpuMesh, p: &silt::particles::Particle, stacks: usize, slices: usize) {
    add_lump(
        mesh,
        p.pos,
        p.rot,
        p.size,
        p.seed,
        stacks,
        slices,
        p.grounded,
    );
}

/// Airborne ledger soil. A flat ribbon of the carried volume. A sphere of
/// `cbrt(volume)` was a blob several times the field.
fn add_soil_ribbon(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let mut dir = Vec3::new(p.vel.x, p.vel.y * 0.22, p.vel.z);
    if dir.length_squared() < 1.0e-6 {
        dir = Vec3::Z;
    }
    dir = dir.normalize();
    let mut side = dir.cross(Vec3::Y);
    if side.length_squared() < 1.0e-6 {
        side = dir.cross(Vec3::X);
    }
    side = side.normalize();
    let up = side.cross(dir).normalize();
    let length = (0.22 + p.vel.length() * 0.05).clamp(0.24, 0.80);
    let width = (p.volume / length).sqrt().clamp(0.018, 0.09);
    let thick = (p.volume / (length * width)).clamp(0.005, 0.022);
    let segs = 4usize;
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let taper = (1.0 - (t * 2.0 - 1.0).abs()).clamp(0.15, 1.0);
        let c = p.pos + dir * (length * (t - 0.5));
        let w = side * width * 0.5 * taper;
        let h = up * thick * 0.5 * taper;
        ring.push([c - w - h, c + w - h, c + w + h, c - w + h]);
    }
    for s in 0..segs {
        let a = ring[s];
        let b = ring[s + 1];
        add_quad(mesh, [a[3], b[3], b[2], a[2]], PANEL_CLOD);
        add_quad(mesh, [a[0], a[1], b[1], b[0]], PANEL_CLOD);
        add_quad(mesh, [a[1], a[2], b[2], b[1]], PANEL_CLOD);
        add_quad(mesh, [b[0], a[0], a[3], b[3]], PANEL_CLOD);
    }
    let a = ring[0];
    let b = ring[segs];
    add_quad(mesh, [a[0], a[3], a[2], a[1]], PANEL_CLOD);
    add_quad(mesh, [b[0], b[1], b[2], b[3]], PANEL_CLOD);
}

/// A fast sideways spin throws a flat ribbon. The tread peel stays curled.
fn spin_ribbon(vel: Vec3) -> bool {
    vel.length() > 6.0
}

/// Straight band in the sheet frame. No bow, so the silhouette is not a crescent.
fn add_flat_strip(mesh: &mut CpuMesh, p: &silt::particles::Particle, length: f32, width: f32, thick: f32) {
    let (dir, side, up) = slurry_sheet_axes(p.vel);
    let segs = 8usize;
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        // Tips narrow a little. The run of the strip stays full width.
        let edge = (t * (1.0 - t) * 4.0).clamp(0.0, 1.0);
        let taper = edge.powf(0.18).clamp(0.7, 1.0);
        let c = p.pos + dir * (length * (t - 0.5));
        let w = side * width * 0.5 * taper;
        let h = up * thick * 0.5;
        ring.push([c - w - h, c + w - h, c + w + h, c - w + h]);
    }
    for s in 0..segs {
        let a = ring[s];
        let b = ring[s + 1];
        add_quad(mesh, [a[3], b[3], b[2], a[2]], PANEL_CLOD);
        add_quad(mesh, [a[0], a[1], b[1], b[0]], PANEL_CLOD);
        add_quad(mesh, [a[1], a[2], b[2], b[1]], PANEL_CLOD);
        add_quad(mesh, [b[0], a[0], a[3], b[3]], PANEL_CLOD);
    }
}

/// Curled slurry in the air. Pinched at both ends so it reads as a ribbon or a
/// roll sheet, not a flat tablet, a ground stamp, or a box.
fn add_slurry_curl(mesh: &mut CpuMesh, p: &silt::particles::Particle, roll_sheet: bool) {
    // Long axis stays in the XZ plane. A roll sheet lies down. A ribbon stands
    // up so a camera behind the car sees the face and not the thin edge.
    // A non-ledger strip is the same curl, kept short so it leaves the tread
    // instead of hanging off the body as a board.
    let shed = !roll_sheet && !p.ledger;
    let mut dir = Vec3::new(p.vel.x, 0.0, p.vel.z);
    if dir.length_squared() < 1.0e-6 {
        dir = Vec3::new(p.vel.x.max(0.2), 0.0, -1.0);
    }
    dir = dir.normalize();
    let mut side = dir.cross(Vec3::Y);
    if side.length_squared() < 1.0e-6 {
        side = dir.cross(Vec3::X);
    }
    side = side.normalize();
    let up = side.cross(dir).normalize();
    // A shed curl lies on the track so the chase camera sees the strip.
    // A standing card from that camera is a flat shard.
    let (dir, side, up) = if shed {
        let side = Vec3::new(-dir.z, 0.0, dir.x).normalize_or_zero();
        let side = if side.length_squared() < 1.0e-6 {
            Vec3::X
        } else {
            side
        };
        (dir, side, Vec3::Y)
    } else if !roll_sheet {
        let side = Vec3::Y;
        let up = dir.cross(side).normalize();
        (dir, side, up)
    } else {
        (dir, side, up)
    };
    let speed = p.vel.length();
    let length = if roll_sheet {
        (0.44 + speed * 0.022).clamp(0.48, 0.68)
    } else if shed {
        (0.48 + speed * 0.02).clamp(0.52, 0.68)
    } else {
        (0.44 + speed * 0.035).clamp(0.48, 0.72)
    };
    let width = (p.volume / length).sqrt();
    let width = if roll_sheet {
        width.clamp(0.09, 0.17)
    } else if shed {
        width.clamp(0.07, 0.11)
    } else {
        width.clamp(0.012, 0.048)
    };
    let thick = (p.volume / (length * width * 0.55)).clamp(0.006, 0.022);
    let length = if shed { length } else { length.min(width * 7.2) };
    let segs = if shed { 12usize } else { 8usize };
    let panel = if roll_sheet { PANEL_CLOD } else { 49.0 };
    let side_sweep = if roll_sheet {
        0.10
    } else if shed {
        0.62
    } else {
        0.55
    };
    let twist_scale = if roll_sheet {
        0.42
    } else if shed {
        0.70
    } else {
        0.55
    };
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let taper = if shed {
            (0.82 + 0.18 * (t * (1.0 - t) * 4.0).sqrt()).clamp(0.78, 1.0)
        } else {
            (t * (1.0 - t) * 4.0).powf(0.55).clamp(0.08, 1.0)
        };
        if shed {
            // A short strip down the rut. It twists and lifts along its
            // length. The centerline does not bend into a crescent.
            let roll = (t - 0.5) * 1.15;
            let lift = (t * std::f32::consts::PI).sin();
            let half_w = 0.058 * taper;
            let skin = 0.012;
            let side_t = (side * roll.cos() + up * roll.sin()).normalize();
            let up_t = dir.cross(side_t).normalize();
            let c = p.pos
                + dir * (length * (t - 0.5))
                + up * lift * 0.055
                + side_t * lift * 0.012;
            let w = side_t * half_w;
            let h = up_t * skin;
            ring.push([c - w - h, c + w - h, c + w + h, c - w + h]);
        } else {
            let arc = (t * std::f32::consts::PI).sin();
            let twist = (t - 0.5) * twist_scale;
            let side_t = side * twist.cos() + up * twist.sin();
            let up_t = -side * twist.sin() + up * twist.cos();
            let c = p.pos
                + dir * (length * (t - 0.5))
                + up * (arc - 0.55) * length * 0.16
                + side * (t - 0.5) * width * side_sweep;
            let w = side_t * width * 0.5 * taper;
            let h = up_t * thick * 0.5 * (0.45 + 0.55 * taper);
            ring.push([c - w - h, c + w - h, c + w + h, c - w + h]);
        }
    }
    for s in 0..segs {
        let a = ring[s];
        let b = ring[s + 1];
        add_quad(mesh, [a[3], b[3], b[2], a[2]], panel);
        add_quad(mesh, [a[0], a[1], b[1], b[0]], panel);
        add_quad(mesh, [a[1], a[2], b[2], b[1]], panel);
        add_quad(mesh, [b[0], a[0], a[3], b[3]], panel);
    }
    let a = ring[0];
    let b = ring[segs];
    add_quad(mesh, [a[0], a[3], a[2], a[1]], panel);
    add_quad(mesh, [b[0], b[1], b[2], b[3]], panel);
}

/// Jagged offset for a torn slurry edge. Corners stay sharp. About −1..1.
fn torn_edge(z: f32, salt: f32) -> f32 {
    let u = z * 2.15 + salt;
    let i = u.floor();
    let f = u - i;
    let cell = |n: f32| -> f32 {
        let h = (n * 12.9898 + salt * 78.233).sin() * 43758.5453;
        let r = h - h.floor();
        if r > 0.74 {
            -1.0
        } else if r > 0.58 {
            0.95
        } else if r < 0.16 {
            -0.62
        } else {
            (r - 0.40) * 0.55
        }
    };
    cell(i) + (cell(i + 1.0) - cell(i)) * f
}

fn slurry_hash(n: f32, salt: f32) -> f32 {
    let h = (n * 12.9898 + salt * 78.233).sin() * 43758.5453;
    h - h.floor()
}

fn ground_at(ground: Option<&silt::soil::SoilField>, x: f32, z: f32, fallback: f32) -> f32 {
    let h = ground.map(|soil| soil.sample(x, z).height).unwrap_or(fallback);
    if h < 0.05 { fallback } else { h }
}

/// A torn patch lying on the soil. Each station picks its own width, so the
/// piece is not a rectangle or a band of one width.
fn add_torn_patch(
    mesh: &mut CpuMesh,
    origin: Vec3,
    dir: Vec3,
    side: Vec3,
    piece_len: f32,
    half_w: f32,
    thick: f32,
    ground: Option<&silt::soil::SoilField>,
    salt: f32,
) {
    let segs = 6usize;
    let across = 4usize;
    let panel = 49.0;
    let mut rows: Vec<Vec<Vec3>> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let center = origin + dir * (t * piece_len);
        let h = ground_at(ground, center.x, center.z, origin.y);
        let along = slurry_hash(center.z * 0.37 + t * 0.55, salt);
        let tear = torn_edge(center.z * 0.45, salt);
        let left = half_w * (0.28 + along * 1.25 + tear * 0.5);
        let right = half_w
            * (0.28 + slurry_hash(center.z * 0.37 + t * 0.55, salt + 2.7) * 1.25 + torn_edge(center.z * 0.45, salt + 4.0) * 0.5);
        let left = left.clamp(0.02, 0.70);
        let right = right.clamp(0.02, 0.70);
        let along_thick = (t * std::f32::consts::PI).sin();
        let station = 0.25 + slurry_hash(center.z, salt + 5.0) * 0.9;
        let mut row = Vec::with_capacity(across);
        for a in 0..across {
            let u = a as f32 / (across - 1) as f32;
            let x = -left + (left + right) * u;
            let from_mid = (u - 0.5).abs() * 2.0;
            let lift = thick * along_thick * station * (1.0 - from_mid * from_mid);
            row.push(Vec3::new(center.x, 0.0, center.z) + side * x + Vec3::Y * (h + 0.004 + lift));
        }
        rows.push(row);
    }
    for s in 0..segs {
        for a in 0..across - 1 {
            let a00 = rows[s][a];
            let a10 = rows[s][a + 1];
            let a01 = rows[s + 1][a];
            let a11 = rows[s + 1][a + 1];
            add_quad(mesh, [a00, a10, a11, a01], panel);
            add_quad(mesh, [a00, a01, a11, a10], panel);
        }
    }
}

/// One tire line of slurry. Separate torn sheets, thin smears, and clods,
/// with gaps. Not one band of constant width.
fn add_rut_ribbon(
    mesh: &mut CpuMesh,
    p: &silt::particles::Particle,
    ground: Option<&silt::soil::SoilField>,
) {
    let mut dir = Vec3::new(p.vel.x, 0.0, p.vel.z);
    if dir.length_squared() < 1.0e-6 {
        dir = Vec3::new(0.0, 0.0, -1.0);
    }
    dir = dir.normalize();
    let mut side = Vec3::new(-dir.z, 0.0, dir.x);
    if side.length_squared() < 1.0e-6 {
        side = Vec3::X;
    }
    side = side.normalize();
    let length = p.size.clamp(0.7, 12.0);
    let lead = 0.16_f32;
    let end = lead + (length - lead).max(0.5);
    let mut dist = lead;
    let mut i = 0i32;
    while dist < end && i < 24 {
        let origin = p.pos + dir * dist;
        let hz = origin.z;
        let gap_roll = slurry_hash(hz, 9.1);
        // A break. The next piece does not continue the last edge.
        if gap_roll < 0.22 {
            dist += 0.55 + slurry_hash(hz, 1.2) * 1.05;
            i += 1;
            continue;
        }
        let yaw = (slurry_hash(hz, 11.0) - 0.5) * 0.55;
        let (c, s) = (yaw.cos(), yaw.sin());
        let dir_p = dir * c + side * s;
        let side_p = side * c - dir * s;
        let off = (slurry_hash(hz, 8.0) - 0.5) * 0.46;
        let roll = slurry_hash(hz, 6.6);
        let (piece_len, half_w, thick, advance) = if roll > 0.72 {
            // A thin smear dragged along the rut. Wide, almost no height.
            let len = 0.70 + slurry_hash(hz, 3.1) * 0.90;
            (
                len,
                0.22 + slurry_hash(hz, 4.4) * 0.28,
                0.0012 + slurry_hash(hz, 5.5) * 0.003,
                len + 0.12 + slurry_hash(hz, 1.4) * 0.35,
            )
        } else if roll < 0.34 {
            // A clod. Thicker than the sheet, still lying down in the rut.
            let len = 0.16 + slurry_hash(hz, 3.1) * 0.18;
            (
                len,
                0.07 + slurry_hash(hz, 4.4) * 0.08,
                0.09 + slurry_hash(hz, 5.5) * 0.07,
                len + 0.08 + slurry_hash(hz, 1.4) * 0.20,
            )
        } else {
            let len = 0.16 + slurry_hash(hz, 3.1) * 0.48;
            (
                len,
                0.045 + slurry_hash(hz, 4.4) * 0.16,
                0.008 + slurry_hash(hz, 5.5) * 0.020,
                len + 0.08 + slurry_hash(hz, 1.4) * 0.40,
            )
        };
        add_torn_patch(
            mesh,
            origin + side * off,
            dir_p,
            side_p,
            piece_len,
            half_w,
            thick,
            ground,
            1.7 + i as f32 * 3.1,
        );
        if slurry_hash(hz, 7.7) > 0.64 {
            let side_off = if slurry_hash(hz, 8.8) > 0.5 { 1.0 } else { -1.0 }
                * (0.12 + slurry_hash(hz, 0.4) * 0.24);
            add_torn_patch(
                mesh,
                origin + side * (off + side_off) + dir * (piece_len * 0.3),
                dir_p,
                side_p,
                0.10 + slurry_hash(hz, 1.9) * 0.12,
                0.05 + slurry_hash(hz, 2.8) * 0.05,
                0.07 + slurry_hash(hz, 3.7) * 0.06,
                ground,
                6.0 + i as f32,
            );
        }
        dist += advance;
        i += 1;
    }
}

/// Liquid slurry torn off a tire. A curled sheet, pinched at both ends, so it
/// is a ribbon rather than a box, a tablet, or a cube.
fn add_slurry_ribbon(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    if !p.ledger {
        add_rut_ribbon(mesh, p, None);
        return;
    }
    add_slurry_curl(mesh, p, false);
}

/// Length, width, and thickness for a roll sheet. The width stands up, so a
/// camera looking along the lane sees the face and not the thin edge.
fn slurry_sheet_axes(vel: Vec3) -> (Vec3, Vec3, Vec3) {
    let mut dir = Vec3::new(vel.x, vel.y * 0.22, vel.z);
    if dir.length_squared() < 1.0e-6 {
        dir = Vec3::X;
    }
    dir = dir.normalize();
    let mut side = Vec3::Y - dir * dir.y;
    if side.length_squared() < 1.0e-6 {
        side = dir.cross(Vec3::Z);
    }
    if side.length_squared() < 1.0e-6 {
        side = Vec3::Z;
    }
    side = side.normalize();
    let mut up = side.cross(dir);
    if up.length_squared() < 1.0e-6 {
        up = Vec3::Z;
    }
    (dir, side, up.normalize())
}

/// A roll's slurry. Wider than a tread ribbon, still curled — not a flat stamp.
fn add_slurry_sheet(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    add_slurry_curl(mesh, p, true);
}

/// A small liquid drop torn off with the ribbons. A short pinched fleck.
/// The ledger volume stays on the particle; this mark does not swell into a
/// teardrop blob, a sphere, or a tablet.
fn add_mud_drop(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    // A short fleck. The long axis follows the horizontal throw, and it stays
    // under 7.5 cm so it is not a bar, a tablet, or a teardrop blob.
    // Arch water falls, so its long axis is the fall and not a flat coin.
    let falling = (p.kind == 9 && p.vel.y < -0.4) || (p.kind == 0 && p.vel.length() > 0.4);
    let mut dir = if falling {
        p.vel
    } else {
        Vec3::new(p.vel.x, 0.0, p.vel.z)
    };
    if dir.length_squared() < 1.0e-4 {
        dir = Vec3::new(p.vel.x, 0.0, -1.0);
    }
    dir = dir.normalize();
    let mut side = if falling {
        Vec3::new(-dir.z, 0.0, dir.x)
    } else {
        dir.cross(Vec3::Y)
    };
    if side.length_squared() < 1.0e-6 {
        side = Vec3::Z;
    }
    side = side.normalize();
    let width_axis = if falling { side } else { Vec3::Y };
    let thick_axis = dir.cross(width_axis).normalize_or_zero();
    let thick_axis = if thick_axis.length_squared() < 1.0e-6 {
        side
    } else {
        thick_axis
    };
    let scale = if p.kind == 9 {
        (p.size / 0.05).clamp(0.7, 1.2)
    } else if p.kind == 0 {
        (p.size / 0.05).clamp(0.8, 1.7)
    } else {
        1.0
    };
    let panel = if p.kind == 0 || p.kind == 9 {
        PANEL_SPRAY
    } else {
        PANEL_CLOD
    };
    let length = if p.kind == 9 {
        0.14
    } else if p.kind == 0 {
        0.11
    } else {
        0.058
    } * scale;
    let width = if p.kind == 9 { 0.016 } else { 0.020 } * scale;
    let thick = 0.010_f32 * scale;
    let segs = 5usize;
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let taper = (0.28 + 0.72 * (t * (1.0 - t) * 4.0).sqrt()).clamp(0.28, 1.0);
        let c = p.pos + dir * (length * (t - 0.5));
        let w = width_axis * width * 0.5 * taper;
        let h = thick_axis * thick * 0.5 * taper;
        ring.push([c - w - h, c + w - h, c + w + h, c - w + h]);
    }
    for s in 0..segs {
        let a = ring[s];
        let b = ring[s + 1];
        add_quad(mesh, [a[3], b[3], b[2], a[2]], panel);
        add_quad(mesh, [a[0], a[1], b[1], b[0]], panel);
        add_quad(mesh, [a[1], a[2], b[2], b[1]], panel);
        add_quad(mesh, [b[0], a[0], a[3], b[3]], panel);
    }
    let a = ring[0];
    let b = ring[segs];
    add_quad(mesh, [a[0], a[3], a[2], a[1]], panel);
    add_quad(mesh, [b[0], b[1], b[2], b[3]], panel);
}

fn puff_quad(mesh: &mut CpuMesh, center: Vec3, axis_u: Vec3, axis_v: Vec3, ru: f32, rv: f32, panel: f32) {
    let u = axis_u * ru;
    let v = axis_v * rv;
    let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    add_quad_uv(
        mesh,
        [center - u - v, center + u - v, center + u + v, center - u + v],
        uv,
        panel,
    );
}

/// Soft dust from the air grid. Crossed puffs with a radial fade, stretched
/// a little along the wind. Ground film is one flat puff. Not a tube, a
/// sphere, or a hard card.
pub fn dust_cloud(items: &[silt::particles::Particle]) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    const MAX_VERTS: usize = 12000;
    for p in items {
        if !p.alive || p.life <= 0.0 || p.kind != 8 {
            continue;
        }
        if mesh.vertices.len() + 48 > MAX_VERTS {
            break;
        }
        if p.pos.y < 0.10 {
            continue;
        }
        // One card. A second crossing card read as a second disc.
        // Smaller marks sit further down the trail and use a fainter panel.
        let panel = if p.size < 0.40 { 47.30 } else { PANEL_DUST };
        let mut wind = Vec3::new(p.vel.x, 0.0, p.vel.z);
        if wind.length_squared() < 1.0e-4 {
            wind = Vec3::new(0.86, 0.0, -0.51);
        }
        wind = wind.normalize();
        let up = Vec3::new(-0.10, 0.62, -0.22).normalize();
        let r = p.size.clamp(0.16, 0.58);
        puff_quad(&mut mesh, p.pos, wind, up, r * 1.35, r * 0.72, panel);
    }
    mesh
}

#[cfg(test)]
mod slurry_shapes {
    use super::*;
    use silt::particles::Particle;

    fn piece(kind: u8, volume: f32, vel: Vec3) -> Particle {
        Particle {
            pos: Vec3::new(0.0, 0.5, 0.0),
            vel,
            life: 2.0,
            max_life: 2.0,
            size: 0.05,
            kind,
            alive: true,
            volume,
            ledger: true,
            ..Particle::default()
        }
    }

    /// Length, width, thickness in the same frame the ribbon mesh uses.
    fn ribbon_frame(mesh: &CpuMesh, vel: Vec3) -> [f32; 3] {
        let mut dir = Vec3::new(vel.x, vel.y * 0.35, vel.z);
        if dir.length_squared() < 1.0e-6 {
            dir = Vec3::Z;
        }
        dir = dir.normalize();
        let mut side = dir.cross(Vec3::Y);
        if side.length_squared() < 1.0e-6 {
            side = dir.cross(Vec3::X);
        }
        side = side.normalize();
        let up = side.cross(dir).normalize();
        let (dir, side, up) = (dir, side, up);
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(-f32::MAX);
        for v in &mesh.vertices {
            let p = Vec3::from_array(v.pos);
            let q = Vec3::new(p.dot(dir), p.dot(side), p.dot(up));
            min = min.min(q);
            max = max.max(q);
        }
        let span = max - min;
        [span.x, span.y, span.z]
    }

    /// Length, width, thickness in the frame `add_slurry_sheet` builds.
    fn sheet_frame(mesh: &CpuMesh, vel: Vec3) -> [f32; 3] {
        let (dir, side, up) = slurry_sheet_axes(vel);
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(-f32::MAX);
        for v in &mesh.vertices {
            let p = Vec3::from_array(v.pos);
            let q = Vec3::new(p.dot(dir), p.dot(side), p.dot(up));
            min = min.min(q);
            max = max.max(q);
        }
        let span = max - min;
        [span.x, span.y, span.z]
    }

    fn extents(mesh: &CpuMesh) -> [f32; 3] {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(-f32::MAX);
        for v in &mesh.vertices {
            let p = Vec3::from_array(v.pos);
            min = min.min(p);
            max = max.max(p);
        }
        let mut e = (max - min).to_array();
        e.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        e
    }

    #[test]
    fn breakaway_mud_is_a_ribbon_and_a_drop() {
        let mut ribbon = CpuMesh::new();
        add_slurry_ribbon(
            &mut ribbon,
            &piece(5, 0.004, Vec3::new(0.0, 0.15, 5.0)),
        );
        let mut drop = CpuMesh::new();
        add_mud_drop(&mut drop, &piece(6, 0.0008, Vec3::new(0.05, 3.4, 0.0)));
        assert!(ribbon.vertices.len() > 24, "ribbon mesh missing");
        assert!(drop.vertices.len() > 24, "drop mesh missing");
        let r = extents(&ribbon);
        let d = extents(&drop);
        assert!(
            r[0] > r[1] * 2.4 && r[0] > r[2] * 3.0,
            "ribbon is not a long sheet {r:?}"
        );
        assert!(
            r[1] / r[0] < 0.42,
            "ribbon reads as a tablet {r:?}"
        );
        assert!(d[0] < 0.075, "drop is a teardrop blob {d:?}");
        assert!(
            d[0] > d[1] * 2.2,
            "drop is round like a sphere {d:?}"
        );
        assert!(
            d[1] / d[0] < 0.45,
            "drop reads as a tablet {d:?}"
        );
        assert!(d[1] > d[2] * 1.8, "drop is a bar {d:?}");
        assert!(d[0] / d[1] < 6.0, "drop is a needle {d:?}");
    }

    #[test]
    fn groove_shed_is_ribbons_not_blocks() {
        use silt::particles::{Particles, KIND_SLURRY_DROP, KIND_SLURRY_RIBBON};
        let mut particles = Particles::new();
        let volume = 0.0024_f32;
        let carried = particles.throw_groove_ribbons(Vec3::new(0.2, 0.4, 0.1), Vec3::Z, Vec3::X, volume);
        assert!((carried - volume).abs() < 1.0e-5, "carried {carried} of {volume}");
        let pieces: Vec<_> = particles
            .items
            .iter()
            .filter(|p| p.alive && p.volume > 1.0e-8)
            .collect();
        assert!(pieces.len() >= 3, "ribbon count {}", pieces.len());
        for p in &pieces {
            assert_eq!(p.kind, KIND_SLURRY_RIBBON, "groove shed used kind {}", p.kind);
            assert_ne!(p.kind, KIND_SLURRY_DROP);
            let mut mesh = CpuMesh::new();
            add_slurry_ribbon(&mut mesh, p);
            let mut e = ribbon_frame(&mesh, p.vel);
            e.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            assert!(
                e[0] > e[1] * 2.4 && e[0] > e[2] * 3.0,
                "groove shed is not a ribbon {e:?}"
            );
            assert!(e[1] / e[0] < 0.42, "groove shed reads as a tablet {e:?}");
            assert!(e[0] / e[2] > 4.0, "groove shed is a sphere or a cube {e:?}");
        }
    }

    #[test]
    fn roll_sheet_is_a_sheet_and_not_a_chunk() {
        use silt::particles::{Particles, KIND_SLURRY_DROP, KIND_SLURRY_RIBBON, KIND_SLURRY_SHEET};
        let mut particles = Particles::new();
        let sheets = particles.throw_roll_sheets(Vec3::new(0.0, 0.5, 0.0), Vec3::Z, Vec3::X, 0.016);
        let ribbons = particles.throw_spin_ribbons(Vec3::new(0.2, 0.4, 0.1), Vec3::Z, Vec3::X, 0.0045);
        assert!((sheets - 0.016).abs() < 1.0e-5);
        assert!((ribbons - 0.0045).abs() < 1.0e-5);
        for p in &particles.items {
            if p.volume <= 1.0e-8 {
                continue;
            }
            assert_ne!(p.kind, KIND_SLURRY_DROP);
            assert_ne!(p.kind, 1);
            let mut mesh = CpuMesh::new();
            let mut e = if p.kind == KIND_SLURRY_SHEET {
                add_slurry_sheet(&mut mesh, p);
                ribbon_frame(&mesh, p.vel)
            } else {
                assert_eq!(p.kind, KIND_SLURRY_RIBBON);
                add_slurry_ribbon(&mut mesh, p);
                ribbon_frame(&mesh, p.vel)
            };
            e.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            assert!(
                e[0] > e[1] * 2.4 && e[0] > e[2] * 3.0,
                "shed is not a long thin sheet {e:?} kind {}",
                p.kind
            );
            assert!(e[1] / e[0] < 0.42, "shed reads as a tablet {e:?}");
            assert!(e[0] / e[2] > 4.0, "shed is a sphere or a chunk {e:?}");
            if p.kind == KIND_SLURRY_SHEET {
                assert!(
                    e[1] > 0.085 && e[1] > e[2] * 1.2 && e[0] > e[1] * 1.8,
                    "roll sheet is a hair or a tablet {e:?}"
                );
            }
        }
    }

    /// Flying slurry is drawn as splash sprites from its volume, not as
    /// ribbon or slab meshes.
    #[test]
    fn breakaway_is_splash_not_slabs() {
        use silt::particles::Particles;
        let mut particles = Particles::new();
        let threw = particles.throw_slurry(Vec3::new(0.4, 0.2, 0.6), Vec3::Z, Vec3::X, 0.012);
        assert!(threw.ribbons > 0.0 && threw.drops > 0.0);
        let mesh = mud_chunks(&particles.items);
        assert!(mesh.vertices.is_empty(), "slurry meshed as solids: {}", mesh.vertices.len());
        let eye = Vec3::new(4.0, 2.0, 4.0);
        let sprites = splash_sprites(&[], &particles.items, eye, Vec3::ZERO, 6000);
        assert!(sprites.len() >= 6 * 8, "breakaway not drawn: {} verts", sprites.len());
        assert!(sprites.iter().all(|v| v.uv[0] >= SPLASH_SLURRY_UV), "slurry drawn as water");
        // No sprite is a card: every quad stays within a few cm of a piece.
        let piece_top = particles.items.iter().map(|p| p.pos.y).fold(f32::MIN, f32::max);
        let top = sprites.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
        assert!(top < piece_top + 0.3, "sprite stands {top} over pieces at {piece_top}");
    }
}

/// Soft wet mud in world space. Large impacts are a pair of lumps; droplets are
/// smaller lumps. Ledger soil is not in this mesh once it has landed: that
/// volume is the heightfield. Nothing here is a camera-facing quad or a faceted stone.
pub fn mud_chunks(items: &[silt::particles::Particle]) -> CpuMesh {
    mud_chunks_with_ground(items, None, true)
}

/// Same chunks, with picture ribbons laid on the heightfield instead of a flat y.
pub fn mud_chunks_with_ground(
    items: &[silt::particles::Particle],
    ground: Option<&silt::soil::SoilField>,
    include_drops: bool,
) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    const MAX_VERTS: usize = 14000;
    // Liquid slurry (kinds 5, 6, 7) is not a mesh any more. A ledger piece
    // in the air is drawn by `splash_sprites` from its volume; once it lands
    // its volume is in the slurry map and the ground shader draws it. The
    // picture-only rut ribbons (no volume) drew as dark slabs at the wheels.
    let _ = ground;
    for kind in [9u8, 3, 1, 4, 0, 2] {
        for p in items {
            if !p.alive || p.life <= 0.0 || p.kind != kind {
                continue;
            }
            if p.kind == silt::particles::KIND_SLURRY_DROP || p.kind == 0 || p.kind == 9 {
                if include_drops && mesh.vertices.len() + 200 < MAX_VERTS {
                    add_mud_drop(&mut mesh, p);
                }
                continue;
            }
            if p.kind == silt::particles::KIND_SLURRY_SHEET {
                if mesh.vertices.len() + 400 < MAX_VERTS {
                    add_slurry_sheet(&mut mesh, p);
                }
                continue;
            }
            if p.kind == silt::particles::KIND_SLURRY_RIBBON {
                // Picture ribbons were drawn above. A tiny ledger piece is
                // the volume in flight, not a sheet the camera should see.
                if p.ledger && p.volume < 0.0015 {
                    continue;
                }
                if !p.ledger {
                    continue;
                }
                if mesh.vertices.len() + 400 < MAX_VERTS {
                    add_slurry_ribbon(&mut mesh, p);
                }
                continue;
            }
            if p.ledger {
                if !p.deposited && p.volume > 1.0e-8 && mesh.vertices.len() + 400 < MAX_VERTS {
                    add_soil_ribbon(&mut mesh, p);
                }
                continue;
            }
            if mesh.vertices.len() + 700 > MAX_VERTS {
                return mesh;
            }
            if p.kind == 3 {
                add_piece(&mut mesh, p, 8, 16);
                // Two side lobes so the clump is a soft mass, not one pebble.
                for (slot, scale, dist) in [(8u32, 0.74, 0.26), (13u32, 0.52, 0.20)] {
                    let dir = Vec3::new(
                        chunk_hash(p.seed, slot) * 2.0 - 1.0,
                        chunk_hash(p.seed, slot + 1) * 0.28,
                        chunk_hash(p.seed, slot + 2) * 2.0 - 1.0,
                    )
                    .normalize_or_zero();
                    let lobe = silt::particles::Particle {
                        pos: p.pos + p.rot * dir * p.size * dist,
                        size: p.size * scale,
                        seed: p.seed ^ 0xA5A5_5A5Au32.wrapping_mul(slot | 1),
                        ..*p
                    };
                    add_piece(&mut mesh, &lobe, 6, 12);
                }
            } else if p.kind == 1 {
                let mut q = *p;
                q.size = p.size.max(0.08);
                add_piece(&mut mesh, &q, 4, 6);
            } else {
                let mut q = *p;
                q.size = if p.kind == 2 {
                    p.size.max(0.045)
                } else {
                    p.size.max(0.07)
                };
                add_piece(&mut mesh, &q, 4, 7);
            }
        }
    }
    mesh
}

/// Smooth UV sphere.
fn add_sphere(mesh: &mut CpuMesh, c: Vec3, r: f32, rot: glam::Quat, panel: f32, stacks: usize, slices: usize) {
    let p = |i: usize, j: usize| -> Vec3 {
        let th = i as f32 / stacks as f32 * std::f32::consts::PI;
        let ph = j as f32 / slices as f32 * std::f32::consts::TAU;
        rot * Vec3::new(th.sin() * ph.cos(), th.cos(), th.sin() * ph.sin())
    };
    for i in 0..stacks {
        for j in 0..slices {
            let (a, b, cc, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
            add_tri_smooth(mesh, c + a * r, a, c + b * r, b, c + cc * r, cc, panel, c);
            add_tri_smooth(mesh, c + a * r, a, c + cc * r, cc, c + d * r, d, panel, c);
        }
    }
}

/// Smooth capsule-ish limb: a tube with a sphere at each end.
fn add_limb(mesh: &mut CpuMesh, a: Vec3, b: Vec3, ra: f32, rb: f32, panel: f32) {
    let axis = b - a;
    let len = axis.length();
    if len < 1.0e-4 {
        return;
    }
    let dir = axis / len;
    let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
    let u = dir.cross(helper).normalize();
    let v = dir.cross(u).normalize();
    let segs = 12;
    let mid = (a + b) * 0.5;
    for i in 0..segs {
        let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let n0 = u * t0.cos() + v * t0.sin();
        let n1 = u * t1.cos() + v * t1.sin();
        add_tri_smooth(mesh, a + n0 * ra, n0, a + n1 * ra, n1, b + n1 * rb, n1, panel, mid);
        add_tri_smooth(mesh, a + n0 * ra, n0, b + n1 * rb, n1, b + n0 * rb, n0, panel, mid);
    }
    add_sphere(mesh, a, ra, glam::Quat::IDENTITY, panel, 6, 10);
    add_sphere(mesh, b, rb, glam::Quat::IDENTITY, panel, 6, 10);
}

/// Box with a rotation.
fn add_obox(mesh: &mut CpuMesh, c: Vec3, h: Vec3, rot: glam::Quat, panel: f32) {
    let x = rot * Vec3::X * h.x;
    let y = rot * Vec3::Y * h.y;
    let z = rot * Vec3::Z * h.z;
    let faces = [
        [c - x + y - z, c - x + y + z, c + x + y + z, c + x + y - z],
        [c - x - y - z, c + x - y - z, c + x - y + z, c - x - y + z],
        [c + x - y - z, c + x + y - z, c + x + y + z, c + x - y + z],
        [c - x - y - z, c - x - y + z, c - x + y + z, c - x + y - z],
        [c - x - y + z, c + x - y + z, c + x + y + z, c - x + y + z],
        [c - x - y - z, c - x + y - z, c + x + y - z, c + x - y - z],
    ];
    for f in faces {
        add_quad(mesh, f, panel);
    }
}

/// Two-bone leg or arm: the joint bends toward `bend`.
fn two_bone(root: Vec3, tip: Vec3, l1: f32, l2: f32, bend: Vec3) -> Vec3 {
    let d = tip - root;
    let len = d.length().clamp(1.0e-3, l1 + l2 - 1.0e-3);
    let dir = d.normalize_or_zero();
    let a = ((l1 * l1 + len * len - l2 * l2) / (2.0 * l1 * len)).clamp(-1.0, 1.0).acos();
    let side = (bend - dir * bend.dot(dir)).normalize_or_zero();
    root + (dir * a.cos() + side * a.sin()) * l1
}

/// A boulder filling a box: a rounded superellipsoid with a lumpy crown.
/// The solver's body is still the box; this only rounds its picture.
fn add_rock(mesh: &mut CpuMesh, c: Vec3, h: Vec3, rot: glam::Quat) {
    let stacks = 12;
    let slices = 20;
    let e = 0.55;
    let sp = |v: f32| v.signum() * v.abs().powf(e);
    let p = |i: usize, j: usize| -> Vec3 {
        let th = i as f32 / stacks as f32 * std::f32::consts::PI;
        let ph = j as f32 / slices as f32 * std::f32::consts::TAU;
        let d = Vec3::new(sp(th.sin()) * sp(ph.cos()), sp(th.cos()), sp(th.sin()) * sp(ph.sin()));
        let lump = 1.0 + 0.07 * (3.1 * ph + 1.7 * th).sin() * (2.3 * th).sin() + 0.04 * (7.0 * ph).cos() * th.sin();
        rot * (d * h * lump)
    };
    for i in 0..stacks {
        for j in 0..slices {
            let (a, b, cc, d) = (p(i, j), p(i + 1, j), p(i + 1, j + 1), p(i, j + 1));
            add_tri(mesh, c + a, c + cc, c + b, 0.0);
            add_tri(mesh, c + a, c + d, c + cc, 0.0);
        }
    }
}

/// Panels: 17 glossy red (buoys and balls), 0 dark stone (rocks), 47 light
/// timber (crates), 10 grey suit (the walker, like the reference's mannequin).
pub fn bodies_mesh(bodies: &silt::bodies::Bodies) -> CpuMesh {
    use silt::bodies::{BodyShape, HIP_HEIGHT};
    let mut m = CpuMesh::new();
    for b in &bodies.items {
        match b.shape {
            BodyShape::Ball { r, .. } => add_sphere(&mut m, b.pos, r, b.rot, 17.0, 14, 24),
            BodyShape::Crate { half } => {
                if b.mass / b.volume() > 1500.0 {
                    add_rock(&mut m, b.pos, half, b.rot);
                } else {
                    add_obox(&mut m, b.pos, half, b.rot, 47.0);
                }
            }
        }
    }
    for w in &bodies.walkers {
        let suit = 10.0;
        let fwd2 = w.forward();
        let fwd = Vec3::new(fwd2.x, 0.0, fwd2.y);
        let right = Vec3::new(fwd.z, 0.0, -fwd.x);
        let pelvis = w.pos;
        let scale = HIP_HEIGHT / 0.92;
        let chest = pelvis + Vec3::Y * 0.50 * scale + fwd * 0.03;
        let neck = chest + Vec3::Y * 0.14;
        // Legs: hip to the sole, knee forward.
        for (k, foot) in w.feet.iter().enumerate() {
            let side = if k == 0 { -1.0 } else { 1.0 };
            let hip = pelvis + right * (0.10 * side);
            let ankle = foot.pos + Vec3::Y * 0.08 - fwd * 0.04;
            let knee = two_bone(hip, ankle, 0.46 * scale, 0.45 * scale, fwd);
            add_limb(&mut m, hip, knee, 0.075, 0.058, suit);
            add_limb(&mut m, knee, ankle, 0.055, 0.042, suit);
            let fyaw = glam::Quat::from_rotation_y(foot.yaw);
            add_obox(&mut m, foot.pos + Vec3::Y * 0.04 + fyaw * Vec3::Z * 0.03, Vec3::new(0.05, 0.04, 0.13), fyaw, suit);
        }
        // Torso and head.
        add_limb(&mut m, pelvis + Vec3::Y * 0.02, chest, 0.15, 0.17, suit);
        add_limb(&mut m, chest, neck, 0.08, 0.06, suit);
        add_sphere(&mut m, neck + Vec3::Y * 0.13, 0.11, glam::Quat::IDENTITY, suit, 10, 16);
        // Arms swing against the legs.
        let swing = (w.feet[0].pos - w.feet[1].pos).dot(fwd).clamp(-0.6, 0.6);
        for k in 0..2 {
            let side = if k == 0 { -1.0 } else { 1.0 };
            let shoulder = chest + right * (0.20 * side) + Vec3::Y * 0.02;
            let reach = -swing * side * 0.45;
            let hand = shoulder + Vec3::Y * -0.58 + fwd * reach + right * (0.06 * side);
            let elbow = two_bone(shoulder, hand, 0.30, 0.30, -fwd + Vec3::Y * 0.1);
            add_limb(&mut m, shoulder, elbow, 0.055, 0.045, suit);
            add_limb(&mut m, elbow, hand, 0.042, 0.035, suit);
        }
    }
    m
}

/// Material tag carried in `uv.x` of a splash sprite: water keeps uv in
/// 0..1, slurry is shifted by 2.
pub const SPLASH_SLURRY_UV: f32 = 2.0;
/// Camera shutter for the motion streak, seconds.
const SPLASH_SHUTTER_S: f32 = 1.0 / 240.0;

/// Hybrid H2 picture of the volume in the air. Each parcel (a sheet drop
/// from `FluidCore::spray_drops` or a ledger slurry piece) is drawn as a
/// small cluster of droplets that spreads as it ages, every droplet a
/// camera-facing sprite stretched along its velocity by the shutter. Only
/// parcels that carry volume are drawn; the cluster size follows that
/// volume. No cards, no boxes, nothing taller than the throw.
pub fn splash_sprites(
    drops: &[silt::fluid_core::LiveSpray],
    items: &[silt::particles::Particle],
    eye: Vec3,
    cam_vel: Vec3,
    max_sprites: usize,
) -> Vec<ParticleVertex> {
    let mut out: Vec<ParticleVertex> = Vec::with_capacity(max_sprites.min(16384) * 6);
    // More parcels than the preset can draw: draw every k-th and grow each
    // sprite by sqrt(k), so the covered area (what the eye reads) stays.
    let slurry_n = items
        .iter()
        .filter(|p| splash_slurry_piece(p))
        .count();
    let want = drops.len() + slurry_n;
    let stride = want.div_ceil(max_sprites.max(1)).max(1);
    let grow = (stride as f32).sqrt();
    let mut room = max_sprites;
    let mut emit = |out: &mut Vec<ParticleVertex>,
                    room: &mut usize,
                    pos: Vec3,
                    vel: Vec3,
                    volume: f32,
                    age: f32,
                    seed: u32,
                    murk: f32| {
        if *room == 0 || !pos.is_finite() || !vel.is_finite() || !(volume > 0.0) {
            return;
        }
        let slurry = murk > 0.5;
        // A parcel of V m3 breaks into drops; draw a handful that stand for
        // it. The equivalent sphere radius of the parcel sets the size.
        let r_eq = (volume * 3.0 / (4.0 * std::f32::consts::PI)).cbrt();
        let n = ((volume / 1.5e-5).round() as usize).clamp(1, 8);
        let spread = r_eq * 1.5 + age * if slurry { 0.08 } else { 0.20 };
        let drop_r = if slurry {
            (r_eq * 0.45).clamp(0.006, 0.02)
        } else {
            (r_eq * 0.9).clamp(0.008, 0.03)
        } * grow;
        let speed = vel.length();
        let along = if speed > 1.0e-3 { vel / speed } else { Vec3::Y };
        for k in 0..n {
            if *room == 0 {
                return;
            }
            let h = |s: u32| chunk_hash(seed, k as u32 * 7 + s);
            let off = if n == 1 {
                Vec3::ZERO
            } else {
                Vec3::new(h(1) - 0.5, (h(2) - 0.5) * 0.6, h(3) - 0.5) * 2.0 * spread
            };
            let c = pos + off + along * (h(4) - 0.5) * spread;
            let r = drop_r * (0.6 + 0.8 * h(5));
            let to_eye = (eye - c).normalize_or_zero();
            if to_eye == Vec3::ZERO {
                continue;
            }
            // Streak axis: the velocity across the view, seen from the
            // moving camera (a chase camera rides with the car).
            let rel = vel - cam_vel;
            let mut u = rel - to_eye * rel.dot(to_eye);
            let ul = u.length();
            let streak = ul * SPLASH_SHUTTER_S;
            if ul > 1.0e-4 {
                u /= ul;
            } else {
                u = Vec3::Y.cross(to_eye).normalize_or_zero();
                if u == Vec3::ZERO {
                    u = Vec3::X;
                }
            }
            let v = to_eye.cross(u).normalize_or_zero();
            let half_u = r + streak * 0.5;
            let half_v = r;
            let g = h(6);
            let clay = [0.20 + 0.06 * g, 0.13 + 0.04 * g, 0.075 + 0.02 * g];
            let clear = [0.92, 0.95, 0.97];
            let m = murk.clamp(0.0, 1.0);
            let col = [
                clear[0] + (clay[0] - clear[0]) * m,
                clear[1] + (clay[1] - clear[1]) * m,
                clear[2] + (clay[2] - clear[2]) * m,
            ];
            let alpha = if slurry {
                0.95
            } else {
                // A long thin streak is fainter than a round drop.
                let thin = (r / half_u).clamp(0.3, 1.0);
                0.45 + 0.4 * thin
            };
            let tag = if slurry { SPLASH_SLURRY_UV } else { 0.0 };
            let corners = [
                (c - u * half_u - v * half_v, [0.0, 0.0]),
                (c + u * half_u - v * half_v, [1.0, 0.0]),
                (c + u * half_u + v * half_v, [1.0, 1.0]),
                (c - u * half_u + v * half_v, [0.0, 1.0]),
            ];
            for i in [0usize, 1, 2, 0, 2, 3] {
                let (p, uv) = corners[i];
                out.push(ParticleVertex {
                    pos: p.to_array(),
                    color: [col[0], col[1], col[2], alpha],
                    uv: [uv[0] + tag, uv[1]],
                });
            }
            *room -= 1;
        }
    };
    // Slurry first: fewer, heavier, and the eye looks for them on mud.
    let mut i = 0usize;
    for p in items {
        if !splash_slurry_piece(p) {
            continue;
        }
        i += 1;
        if i % stride != 0 {
            continue;
        }
        let age = (p.max_life - p.life).max(0.0);
        emit(&mut out, &mut room, p.pos, p.vel, p.volume, age, p.seed, 1.0);
    }
    for (i, d) in drops.iter().enumerate() {
        if i % stride != 0 {
            continue;
        }
        emit(
            &mut out,
            &mut room,
            Vec3::from_array(d.pos),
            Vec3::from_array(d.vel),
            d.volume,
            d.age,
            0x5EED_0000 ^ i as u32,
            d.murk,
        );
    }
    out
}

/// A ledger slurry piece still in the air: kinds 5, 6, 7 with volume.
fn splash_slurry_piece(p: &silt::particles::Particle) -> bool {
    p.alive
        && p.life > 0.0
        && p.ledger
        && !p.grounded
        && !p.deposited
        && p.volume > 0.0
        && matches!(
            p.kind,
            silt::particles::KIND_SLURRY_RIBBON
                | silt::particles::KIND_SLURRY_DROP
                | silt::particles::KIND_SLURRY_SHEET
        )
}
