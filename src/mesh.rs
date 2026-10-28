//! Procedural buggy, wheels and a small patch of countryside.
//!
//! Geometry is flat-shaded and carries a panel id. The shader turns that id
//! into paint, rubber or mud. Wheel and shock meshes stay in the spaces the
//! vehicle solver already uses, so the picture matches the contact model.

use std::collections::{HashMap, HashSet};

use glam::{Quat, Vec3};

use silt::bike::Bike;
use silt::soil::track_offset;
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
    fn new() -> Self {
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
    add_tube(
        &mut m,
        pivot,
        rear + Vec3::new(0.0, 0.02, 0.0),
        0.03,
        arm,
        5,
    );
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

/// Subdivisions of one solver cell on the uniform draw grid this replaced.
/// Ground away from the oval stays on that grid.
const BASE_SUB: usize = 3;

/// Bowls in `SoilField::generate`. The draw grid spends triangles here; the
/// solver grid is untouched.
const PUDDLES: [(f32, f32, f32, f32); 3] = [
    (-6.15, 5.05, 2.35, 2.55),
    (6.35, 2.40, 2.30, 1.80),
    (1.15, 9.25, 2.15, 2.05),
];
const BOG_CENTER: (f32, f32) = (0.0, 1.05);
const BOG_RAD: f32 = 4.2;

pub fn terrain_mesh(res: usize, size: f32) -> (Vec<TerrainVertex>, Vec<u32>) {
    // The solver still stores one height per ~20 cm. This mesh only chooses
    // where those samples are drawn. One red-green pass is ×4 triangles, two
    // is ×16. The second pass is the tyre-wall band and the puddle rims.
    // The apron stays on the base grid so the water pass does not pay for empty field.
    let (mut verts, mut tris) = uniform_terrain(res, size, BASE_SUB);
    refine_marked(&mut verts, &mut tris, |x, z| detail_level(x, z) >= 1);
    refine_marked(&mut verts, &mut tris, |x, z| detail_level(x, z) >= 2);
    let mut indices = Vec::with_capacity(tris.len() * 3);
    for t in tris {
        indices.extend_from_slice(&t);
    }
    (verts, indices)
}

/// 0 keeps the base grid, 1 splits once (×4 triangles), 2 splits again (×16).
fn detail_level(x: f32, z: f32) -> u8 {
    let off = track_offset(x, z).abs();
    let mut level = 0u8;
    // Oval, berm, and standing water. One split takes the edge from ~7 cm to ~3.5 cm.
    if off < 3.9 || water_basin(x, z) {
        level = 1;
    }
    // Tyre walls sit near ±0.7 m. A second split is spent there and on the
    // puddle rims, not across the whole apron: the water pass runs this mesh.
    if (0.40..1.28).contains(&off) || puddle_shore(x, z) {
        level = 2;
    }
    level
}

fn water_basin(x: f32, z: f32) -> bool {
    for (cx, cz, rx, rz) in PUDDLES {
        if ellipse_d(x, z, cx, cz, rx + 0.7, rz + 0.7) <= 1.0 {
            return true;
        }
    }
    bog_r(x, z) < BOG_RAD + 0.75
}

fn puddle_shore(x: f32, z: f32) -> bool {
    for (cx, cz, rx, rz) in PUDDLES {
        let inner = ellipse_d(x, z, cx, cz, rx * 0.62, rz * 0.62);
        let outer = ellipse_d(x, z, cx, cz, rx + 0.40, rz + 0.40);
        if inner >= 1.0 && outer <= 1.0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
fn bog_shore(x: f32, z: f32) -> bool {
    // Pond rim. One split (the basin) already quarters the old facets.
    let r = bog_r(x, z);
    (3.45..4.70).contains(&r)
}

fn ellipse_d(x: f32, z: f32, cx: f32, cz: f32, rx: f32, rz: f32) -> f32 {
    let dx = (x - cx) / rx.max(0.2);
    let dz = (z - cz) / rz.max(0.2);
    (dx * dx + dz * dz).sqrt()
}

fn bog_r(x: f32, z: f32) -> f32 {
    let dx = x - BOG_CENTER.0;
    let dz = z - BOG_CENTER.1;
    (dx * dx + dz * dz).sqrt()
}

fn uniform_terrain(res: usize, size: f32, sub: usize) -> (Vec<TerrainVertex>, Vec<[u32; 3]>) {
    if res < 2 || sub == 0 {
        return (Vec::new(), Vec::new());
    }
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
    let mut tris = Vec::with_capacity((n - 1) * (n - 1) * 2);
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let a = (j * n + i) as u32;
            let b = ((j + 1) * n + i) as u32;
            let c = ((j + 1) * n + i + 1) as u32;
            let d = (j * n + i + 1) as u32;
            // +Z then +X winds upward.
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    (verts, tris)
}

fn edge_key(a: u32, b: u32) -> u64 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    ((lo as u64) << 32) | hi as u64
}

struct EdgeAdj {
    tris: [u32; 2],
    n: u8,
}

fn tri_centroid(verts: &[TerrainVertex], t: [u32; 3]) -> (f32, f32) {
    let a = verts[t[0] as usize].xz;
    let b = verts[t[1] as usize].xz;
    let c = verts[t[2] as usize].xz;
    ((a[0] + b[0] + c[0]) / 3.0, (a[1] + b[1] + c[1]) / 3.0)
}

fn split_count(t: [u32; 3], split: &HashSet<u64>) -> u8 {
    let mut n = 0u8;
    if split.contains(&edge_key(t[0], t[1])) {
        n += 1;
    }
    if split.contains(&edge_key(t[1], t[2])) {
        n += 1;
    }
    if split.contains(&edge_key(t[2], t[0])) {
        n += 1;
    }
    n
}

/// Red-green split. Marked triangles become four. A neighbour that shares
/// one split edge becomes two, so the mesh stays watertight. A neighbour
/// with two split edges is promoted so it does not keep a T-junction.
fn refine_marked(
    verts: &mut Vec<TerrainVertex>,
    tris: &mut Vec<[u32; 3]>,
    mark: impl Fn(f32, f32) -> bool,
) {
    if tris.is_empty() {
        return;
    }
    let mut adj: HashMap<u64, EdgeAdj> = HashMap::with_capacity(tris.len() * 2);
    for (i, t) in tris.iter().enumerate() {
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let e = adj
                .entry(edge_key(a, b))
                .or_insert(EdgeAdj { tris: [0, 0], n: 0 });
            assert!(e.n < 2, "draw mesh edge already has two triangles");
            e.tris[e.n as usize] = i as u32;
            e.n += 1;
        }
    }

    let mut split: HashSet<u64> = HashSet::with_capacity(tris.len());
    let mut red = vec![false; tris.len()];
    let mut queue: Vec<u32> = Vec::new();
    for (i, t) in tris.iter().enumerate() {
        let (x, z) = tri_centroid(verts, *t);
        if mark(x, z) {
            queue.push(i as u32);
        }
    }

    while let Some(i) = queue.pop() {
        if red[i as usize] {
            continue;
        }
        red[i as usize] = true;
        let t = tris[i as usize];
        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
            let k = edge_key(a, b);
            if !split.insert(k) {
                continue;
            }
            let Some(edge) = adj.get(&k) else {
                continue;
            };
            for n in 0..edge.n as usize {
                let ni = edge.tris[n];
                if ni != i && !red[ni as usize] && split_count(tris[ni as usize], &split) >= 2 {
                    queue.push(ni);
                }
            }
        }
    }

    let mut mids: HashMap<u64, u32> = HashMap::with_capacity(split.len());
    let mut midpoint = |verts: &mut Vec<TerrainVertex>, a: u32, b: u32| -> u32 {
        let k = edge_key(a, b);
        if let Some(id) = mids.get(&k) {
            return *id;
        }
        let va = verts[a as usize];
        let vb = verts[b as usize];
        let id = verts.len() as u32;
        verts.push(TerrainVertex {
            xz: [(va.xz[0] + vb.xz[0]) * 0.5, (va.xz[1] + vb.xz[1]) * 0.5],
            ij: [(va.ij[0] + vb.ij[0]) * 0.5, (va.ij[1] + vb.ij[1]) * 0.5],
        });
        mids.insert(k, id);
        id
    };

    let mut out = Vec::with_capacity(tris.len() * 2);
    for t in tris.iter() {
        let [a, b, c] = *t;
        let ab = split.contains(&edge_key(a, b));
        let bc = split.contains(&edge_key(b, c));
        let ca = split.contains(&edge_key(c, a));
        let n = ab as u8 + bc as u8 + ca as u8;
        assert!(
            n != 2,
            "red-green closure left a triangle with two split edges"
        );
        if n == 3 {
            let mab = midpoint(verts, a, b);
            let mbc = midpoint(verts, b, c);
            let mca = midpoint(verts, c, a);
            out.push([a, mab, mca]);
            out.push([mab, b, mbc]);
            out.push([mca, mbc, c]);
            out.push([mab, mbc, mca]);
        } else if ab {
            let m = midpoint(verts, a, b);
            out.push([a, m, c]);
            out.push([m, b, c]);
        } else if bc {
            let m = midpoint(verts, b, c);
            out.push([a, b, m]);
            out.push([a, m, c]);
        } else if ca {
            let m = midpoint(verts, c, a);
            out.push([a, b, m]);
            out.push([m, b, c]);
        } else {
            out.push([a, b, c]);
        }
    }
    *tris = out;
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
    let y = -0.02;
    // Skirt outside the 40 m heightfield so the world does not end in a void.
    let far = 78.0;
    let edge = 20.0;
    add_quad(
        &mut m,
        [
            Vec3::new(-far, y, edge),
            Vec3::new(far, y, edge),
            Vec3::new(far, y, far),
            Vec3::new(-far, y, far),
        ],
        PANEL_SKIRT,
    );
    add_quad(
        &mut m,
        [
            Vec3::new(-far, y, -far),
            Vec3::new(far, y, -far),
            Vec3::new(far, y, -edge),
            Vec3::new(-far, y, -edge),
        ],
        PANEL_SKIRT,
    );
    add_quad(
        &mut m,
        [
            Vec3::new(edge, y, -edge),
            Vec3::new(far, y, -edge),
            Vec3::new(far, y, edge),
            Vec3::new(edge, y, edge),
        ],
        PANEL_SKIRT,
    );
    add_quad(
        &mut m,
        [
            Vec3::new(-far, y, -edge),
            Vec3::new(-edge, y, -edge),
            Vec3::new(-edge, y, edge),
            Vec3::new(-far, y, edge),
        ],
        PANEL_SKIRT,
    );

    // Low rolling silhouette. The inner ring is tucked into the skirt so it
    // sits on the ground instead of floating as boxes.
    add_horizon(&mut m);

    let trees = [
        Vec3::new(-16.0, 0.0, 14.0),
        Vec3::new(18.0, 0.0, -16.0),
        Vec3::new(14.0, 0.0, 16.0),
        Vec3::new(-18.0, 0.0, -11.0),
        Vec3::new(20.0, 0.0, 6.0),
    ];
    for (k, p) in trees.iter().enumerate() {
        let trunk_h = 1.35 + hash(20 + k as u32) * 0.7;
        add_tube(&mut m, *p, *p + Vec3::Y * trunk_h, 0.09, PANEL_TRUNK, 6);
        // Stacked cones so the crown has volume instead of a flat card.
        let layers = [
            (trunk_h * 0.42, 1.25, 1.45),
            (trunk_h * 0.85, 1.0, 1.2),
            (trunk_h * 1.25, 0.72, 0.95),
        ];
        for (i, (y, radius, height)) in layers.iter().enumerate() {
            let j = hash(40 + k as u32 * 3 + i as u32);
            let jx = (j - 0.5) * 0.2;
            let jz = (hash(90 + k as u32 * 3 + i as u32) - 0.5) * 0.2;
            add_cone(
                &mut m,
                *p + Vec3::new(jx, *y, jz),
                *height,
                *radius * (0.9 + j * 0.2),
                PANEL_TREE,
                6,
            );
        }
    }
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
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
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
    panel: f32,
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
            mesh, pts[0], nrm[0], pts[a], nrm[a], pts[b], nrm[b], panel, center,
        );
    }
    for s in 1..stacks.saturating_sub(1) {
        for i in 0..slices {
            let a = ring(s, i);
            let b = ring(s, i + 1);
            let c = ring(s + 1, i);
            let d = ring(s + 1, i + 1);
            add_tri_smooth(
                mesh, pts[a], nrm[a], pts[c], nrm[c], pts[b], nrm[b], panel, center,
            );
            add_tri_smooth(
                mesh, pts[b], nrm[b], pts[c], nrm[c], pts[d], nrm[d], panel, center,
            );
        }
    }
    let bot = pts.len() - 1;
    let last = stacks - 1;
    for i in 0..slices {
        let a = ring(last, i + 1);
        let b = ring(last, i);
        add_tri_smooth(
            mesh, pts[bot], nrm[bot], pts[a], nrm[a], pts[b], nrm[b], panel, center,
        );
    }
}

fn add_piece(mesh: &mut CpuMesh, p: &silt::particles::Particle, stacks: usize, slices: usize) {
    // Tire spray is a bright sheet of water. Clods stay wet mud.
    let panel = if p.kind == 0 || p.kind == 4 { 47.0 } else { PANEL_CLOD };
    add_lump(
        mesh, p.pos, p.rot, p.size, p.seed, stacks, slices, p.grounded, panel,
    );
}

/// Soft wet mud in world space. Large impacts are a pair of lumps; droplets are
/// smaller lumps. Nothing here is a camera-facing quad or a faceted stone.
pub fn mud_chunks(items: &[silt::particles::Particle]) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    const MAX_VERTS: usize = 14000;
    // Draw the big clumps first so a full splash cannot crowd them out.
    for kind in [3u8, 1, 4, 0, 2] {
        for p in items {
            if !p.alive || p.life <= 0.0 || p.kind != kind {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pack(tris: &[[u32; 3]]) -> Vec<u32> {
        let mut indices = Vec::with_capacity(tris.len() * 3);
        for t in tris {
            indices.extend_from_slice(t);
        }
        indices
    }

    fn centroid(verts: &[TerrainVertex], tri: &[u32]) -> (f32, f32) {
        tri_centroid(verts, [tri[0], tri[1], tri[2]])
    }

    fn count_where(
        verts: &[TerrainVertex],
        indices: &[u32],
        pred: impl Fn(f32, f32) -> bool,
    ) -> usize {
        indices
            .chunks(3)
            .filter(|tri| {
                let (x, z) = centroid(verts, tri);
                pred(x, z)
            })
            .count()
    }

    fn mean_edge(verts: &[TerrainVertex], indices: &[u32], pred: impl Fn(f32, f32) -> bool) -> f32 {
        let mut sum = 0.0f64;
        let mut n = 0u64;
        for tri in indices.chunks(3) {
            let (x, z) = centroid(verts, tri);
            if !pred(x, z) {
                continue;
            }
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let a = verts[tri[i] as usize].xz;
                let b = verts[tri[j] as usize].xz;
                let dx = (a[0] - b[0]) as f64;
                let dz = (a[1] - b[1]) as f64;
                sum += (dx * dx + dz * dz).sqrt();
                n += 1;
            }
        }
        (sum / n.max(1) as f64) as f32
    }

    fn assert_oriented(indices: &[u32]) {
        let mut dir: HashMap<u64, u8> = HashMap::with_capacity(indices.len());
        for tri in indices.chunks(3) {
            assert!(tri[0] != tri[1] && tri[1] != tri[2] && tri[2] != tri[0]);
            for (u, v) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
                let k = ((u as u64) << 32) | v as u64;
                let n = dir.entry(k).or_insert(0);
                *n += 1;
                assert_eq!(*n, 1, "edge {u}->{v} is used twice");
            }
        }
    }

    fn assert_upward(verts: &[TerrainVertex], indices: &[u32]) {
        let mut seen = 0u32;
        for tri in indices.chunks(3).step_by(64) {
            let a = verts[tri[0] as usize].xz;
            let b = verts[tri[1] as usize].xz;
            let c = verts[tri[2] as usize].xz;
            // Y of (b-a) × (c-a). Positive faces up; the xz cross has the opposite sign.
            let abx = b[0] - a[0];
            let abz = b[1] - a[1];
            let acx = c[0] - a[0];
            let acz = c[1] - a[1];
            let y = abz * acx - abx * acz;
            assert!(y > 1.0e-8, "triangle winds downward");
            seen += 1;
        }
        assert!(seen > 4);
    }

    fn assert_ij(verts: &[TerrainVertex], res: usize, size: f32) {
        let cell = size / res as f32;
        let half = size * 0.5;
        for v in verts.iter().step_by(32) {
            let fi = (v.xz[0] + half) / cell - 0.5;
            let fj = (v.xz[1] + half) / cell - 0.5;
            assert!((fi - v.ij[0]).abs() < 1.0e-3 && (fj - v.ij[1]).abs() < 1.0e-3);
            assert!(v.ij[0] >= -1.0e-3 && v.ij[1] >= -1.0e-3);
            assert!(v.ij[0] <= res as f32 - 1.0 + 1.0e-3);
            assert!(v.ij[1] <= res as f32 - 1.0 + 1.0e-3);
        }
    }

    /// A vertex in the interior of an edge that the mesh still draws whole.
    fn assert_no_t_junctions(verts: &[TerrainVertex], indices: &[u32]) {
        let inv = 20.0;
        let mut bins: HashMap<(i32, i32), Vec<u32>> = HashMap::new();
        for (i, v) in verts.iter().enumerate() {
            let key = (
                (v.xz[0] * inv).floor() as i32,
                (v.xz[1] * inv).floor() as i32,
            );
            bins.entry(key).or_default().push(i as u32);
        }
        for tri in indices.chunks(3) {
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let a = tri[i];
                let b = tri[j];
                let pa = verts[a as usize].xz;
                let pb = verts[b as usize].xz;
                let abx = pb[0] - pa[0];
                let abz = pb[1] - pa[1];
                let ab2 = abx * abx + abz * abz;
                if ab2 < 1.0e-12 {
                    continue;
                }
                let i0 = (pa[0].min(pb[0]) * inv).floor() as i32 - 1;
                let i1 = (pa[0].max(pb[0]) * inv).floor() as i32 + 1;
                let j0 = (pa[1].min(pb[1]) * inv).floor() as i32 - 1;
                let j1 = (pa[1].max(pb[1]) * inv).floor() as i32 + 1;
                let ab_len = ab2.sqrt();
                for ix in i0..=i1 {
                    for jz in j0..=j1 {
                        let Some(bucket) = bins.get(&(ix, jz)) else {
                            continue;
                        };
                        for &v in bucket {
                            if v == a || v == b {
                                continue;
                            }
                            let p = verts[v as usize].xz;
                            let vx = p[0] - pa[0];
                            let vz = p[1] - pa[1];
                            let cross = (vx * abz - vz * abx).abs();
                            if cross > 2.0e-4 * ab_len {
                                continue;
                            }
                            let dot = vx * abx + vz * abz;
                            if dot > 1.0e-4 * ab2 && dot < ab2 - 1.0e-4 * ab2 {
                                panic!("T-junction at vertex {v} on edge {a}-{b}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn partial_refine_stays_manifold() {
        let (mut verts, mut tris) = uniform_terrain(16, 8.0, 2);
        let before = tris.len();
        refine_marked(&mut verts, &mut tris, |x, _z| x < -0.4);
        refine_marked(&mut verts, &mut tris, |x, z| x < -1.2 && z > 0.0);
        assert!(tris.len() > before);
        let indices = pack(&tris);
        assert_oriented(&indices);
        assert_upward(&verts, &indices);
        assert_no_t_junctions(&verts, &indices);
        let fine = mean_edge(&verts, &indices, |x, _z| x < -1.5);
        let coarse = mean_edge(&verts, &indices, |x, _z| x > 2.0);
        assert!(
            fine < coarse * 0.4,
            "fine edge {fine:.4} should be well under coarse {coarse:.4}"
        );
    }

    #[test]
    fn oval_has_at_least_four_times_the_triangles() {
        const RES: usize = 192;
        const SIZE: f32 = 40.0;
        let (base_v, base_t) = uniform_terrain(RES, SIZE, BASE_SUB);
        let base_i = pack(&base_t);
        let (dense_v, dense_i) = terrain_mesh(RES, SIZE);
        let on_oval = |x: f32, z: f32| track_offset(x, z).abs() <= 3.2;
        let on_rut = |x: f32, z: f32| {
            let off = track_offset(x, z).abs();
            (0.40..1.28).contains(&off)
        };
        let on_far = |x: f32, z: f32| track_offset(x, z).abs() > 10.0 && !water_basin(x, z);
        let on_shore = |x: f32, z: f32| puddle_shore(x, z);
        let on_bog = |x: f32, z: f32| bog_shore(x, z);

        let base_tris = base_i.len() / 3;
        let dense_tris = dense_i.len() / 3;
        let base_oval = count_where(&base_v, &base_i, on_oval);
        let dense_oval = count_where(&dense_v, &dense_i, on_oval);
        let base_rut = count_where(&base_v, &base_i, on_rut);
        let dense_rut = count_where(&dense_v, &dense_i, on_rut);
        let base_far = count_where(&base_v, &base_i, on_far);
        let dense_far = count_where(&dense_v, &dense_i, on_far);
        let base_shore = count_where(&base_v, &base_i, on_shore);
        let dense_shore = count_where(&dense_v, &dense_i, on_shore);
        let base_bog = count_where(&base_v, &base_i, on_bog);
        let dense_bog = count_where(&dense_v, &dense_i, on_bog);
        let oval_ratio = dense_oval as f64 / base_oval.max(1) as f64;
        let rut_ratio = dense_rut as f64 / base_rut.max(1) as f64;
        let far_ratio = dense_far as f64 / base_far.max(1) as f64;
        let shore_ratio = dense_shore as f64 / base_shore.max(1) as f64;
        let bog_ratio = dense_bog as f64 / base_bog.max(1) as f64;
        let rut_edge = mean_edge(&dense_v, &dense_i, on_rut);
        let far_edge = mean_edge(&dense_v, &dense_i, on_far);
        let shore_edge = mean_edge(&dense_v, &dense_i, on_shore);
        let bog_edge = mean_edge(&dense_v, &dense_i, on_bog);
        eprintln!(
            "tris {base_tris} -> {dense_tris} ({:.2}x)  verts {} -> {}",
            dense_tris as f64 / base_tris as f64,
            base_v.len(),
            dense_v.len()
        );
        eprintln!(
            "oval {base_oval} -> {dense_oval} ({oval_ratio:.2}x)  rut {base_rut} -> {dense_rut} ({rut_ratio:.2}x)"
        );
        eprintln!(
            "far {base_far} -> {dense_far} ({far_ratio:.2}x)  shore {base_shore} -> {dense_shore} ({shore_ratio:.2}x)  bog {base_bog} -> {dense_bog} ({bog_ratio:.2}x)"
        );
        eprintln!(
            "edge m  rut {rut_edge:.4}  shore {shore_edge:.4}  bog {bog_edge:.4}  far {far_edge:.4}"
        );

        assert!(
            dense_oval >= base_oval * 4,
            "oval triangles {base_oval} -> {dense_oval} ({oval_ratio:.2}x), want >= 4x"
        );
        assert!(
            rut_ratio + 0.05 >= oval_ratio,
            "rut band should be at least as dense as the oval ({rut_ratio:.2}x vs {oval_ratio:.2}x)"
        );
        assert!(
            shore_ratio >= 4.0 && bog_ratio >= 3.5,
            "puddle shore {base_shore} -> {dense_shore} ({shore_ratio:.2}x), bog {base_bog} -> {dense_bog} ({bog_ratio:.2}x)"
        );
        assert!(
            rut_edge < far_edge * 0.45 && shore_edge < far_edge * 0.45 && bog_edge < far_edge * 0.7,
            "track and shores should be finer than the apron"
        );
        assert!(
            far_ratio < 1.35,
            "apron was refined too ({far_ratio:.2}x); keep the empty field on the base grid"
        );
        // The water pass evaluates every vertex. Stay under ~2.5× the old vertex load.
        assert!(
            dense_tris < 1_800_000,
            "triangle count {dense_tris} is past the real-time budget"
        );
        assert_oriented(&dense_i);
        assert_upward(&dense_v, &dense_i);
        assert_ij(&dense_v, RES, SIZE);
        assert_no_t_junctions(&dense_v, &dense_i);
    }

    /// Cells. Must match `DRAW_SIGMA` / `draw_axis` in shader.wgsl.
    const DRAW_SIGMA: f32 = 0.64;

    fn draw_gauss(d: f32) -> f32 {
        let q = d / DRAW_SIGMA;
        (-0.5 * q * q).exp()
    }

    /// Normalized 5-tap gaussian and d(weight)/d(index). `f` is the fractional index.
    fn draw_axis(f: f32) -> ([f32; 5], [f32; 5]) {
        let mut g = [0.0; 5];
        let mut dg = [0.0; 5];
        let inv_s2 = 1.0 / (DRAW_SIGMA * DRAW_SIGMA);
        let mut sum = 0.0;
        let mut dsum = 0.0;
        for i in 0..5 {
            let d = f - (i as f32 - 2.0);
            let gi = draw_gauss(d);
            let dgi = gi * (-d * inv_s2);
            g[i] = gi;
            dg[i] = dgi;
            sum += gi;
            dsum += dgi;
        }
        let mut w = [0.0; 5];
        let mut dw = [0.0; 5];
        let inv = 1.0 / sum;
        let inv2 = inv * inv;
        for i in 0..5 {
            w[i] = g[i] * inv;
            dw[i] = (dg[i] * sum - g[i] * dsum) * inv2;
        }
        (w, dw)
    }

    fn sample_height(soil: &silt::soil::SoilField, i: isize, j: isize) -> f32 {
        let ii = i.clamp(0, soil.res as isize - 1) as usize;
        let jj = j.clamp(0, soil.res as isize - 1) as usize;
        soil.height[jj * soil.res + ii]
    }

    /// Draw height. Same gather the terrain vertex shader uses.
    fn draw_height(soil: &silt::soil::SoilField, x: f32, z: f32) -> f32 {
        let half = soil.size * 0.5;
        let u = (x + half) / soil.cell - 0.5;
        let v = (z + half) / soil.cell - 0.5;
        let (wx, _) = draw_axis(u - u.floor());
        let (wz, _) = draw_axis(v - v.floor());
        let i0 = u.floor() as isize;
        let j0 = v.floor() as isize;
        let mut h = 0.0;
        for j in 0..5 {
            let mut row = 0.0;
            for i in 0..5 {
                row += wx[i] * sample_height(soil, i0 + i as isize - 2, j0 + j as isize - 2);
            }
            h += wz[j] * row;
        }
        h
    }

    fn draw_slope_x(soil: &silt::soil::SoilField, x: f32, z: f32) -> f32 {
        let half = soil.size * 0.5;
        let u = (x + half) / soil.cell - 0.5;
        let v = (z + half) / soil.cell - 0.5;
        let (_wx, dwx) = draw_axis(u - u.floor());
        let (wz, _) = draw_axis(v - v.floor());
        let i0 = u.floor() as isize;
        let j0 = v.floor() as isize;
        let mut dhdu = 0.0;
        for j in 0..5 {
            let mut drow = 0.0;
            for i in 0..5 {
                drow += dwx[i] * sample_height(soil, i0 + i as isize - 2, j0 + j as isize - 2);
            }
            dhdu += wz[j] * drow;
        }
        dhdu / soil.cell
    }

    fn max_turn_deg(xs: &[f32], hs: &[f32]) -> f32 {
        let mut max_k = 0.0f32;
        for i in 1..xs.len().saturating_sub(1) {
            let v1x = xs[i] - xs[i - 1];
            let v1y = hs[i] - hs[i - 1];
            let v2x = xs[i + 1] - xs[i];
            let v2y = hs[i + 1] - hs[i];
            let n1 = (v1x * v1x + v1y * v1y).sqrt();
            let n2 = (v2x * v2x + v2y * v2y).sqrt();
            if n1 < 1.0e-4 || n2 < 1.0e-4 {
                continue;
            }
            let dot = ((v1x * v2x + v1y * v2y) / (n1 * n2)).clamp(-1.0, 1.0);
            max_k = max_k.max(dot.acos().to_degrees());
        }
        max_k
    }

    fn bilinear_height(soil: &silt::soil::SoilField, x: f32, z: f32) -> f32 {
        let half = soil.size * 0.5;
        let u = (x + half) / soil.cell - 0.5;
        let v = (z + half) / soil.cell - 0.5;
        let i = u.floor() as isize;
        let j = v.floor() as isize;
        let fx = u - i as f32;
        let fy = v - j as f32;
        let h = |ii: isize, jj: isize| {
            let ii = ii.clamp(0, soil.res as isize - 1) as usize;
            let jj = jj.clamp(0, soil.res as isize - 1) as usize;
            soil.height[jj * soil.res + ii]
        };
        let h00 = h(i, j);
        let h10 = h(i + 1, j);
        let h01 = h(i, j + 1);
        let h11 = h(i + 1, j + 1);
        let a = h00 + (h10 - h00) * fx;
        let b = h01 + (h11 - h01) * fx;
        a + (b - a) * fy
    }

    fn section_xs(verts: &[TerrainVertex], indices: &[u32], z: f32, x0: f32, x1: f32) -> Vec<f32> {
        let mut xs = Vec::new();
        for tri in indices.chunks(3) {
            let p = [
                verts[tri[0] as usize].xz,
                verts[tri[1] as usize].xz,
                verts[tri[2] as usize].xz,
            ];
            for (i, j) in [(0, 1), (1, 2), (2, 0)] {
                let (a, b) = (p[i], p[j]);
                let az = a[1];
                let bz = b[1];
                if (az - z) * (bz - z) > 0.0 {
                    continue;
                }
                let dz = bz - az;
                if dz.abs() < 1.0e-8 {
                    continue;
                }
                let t = (z - az) / dz;
                if !(0.0..=1.0).contains(&t) {
                    continue;
                }
                let x = a[0] + (b[0] - a[0]) * t;
                if (x0..=x1).contains(&x) {
                    xs.push(x);
                }
            }
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs.dedup_by(|a, b| (*a - *b).abs() < 5.0e-4);
        xs
    }

    /// Sagitta inside one bilinear patch, and the sharpest corner where two
    /// patches meet. A straight ramp has no sagitta; the corner is the facet.
    fn max_cell_sagitta(xs: &[f32], hs: &[f32], cell: f32) -> (f32, f32) {
        let mut max_s = 0.0f32;
        let mut max_kink = 0.0f32;
        let half = 20.0f32;
        let patch = |x: f32| ((x + half) / cell - 0.5).floor() as i32;
        let mut i = 0;
        while i + 1 < xs.len() {
            let id = patch(xs[i]);
            let mut j = i;
            while j < xs.len() && patch(xs[j]) == id {
                j += 1;
            }
            if j - i >= 3 {
                let x0 = xs[i];
                let x1 = xs[j - 1];
                let h0 = hs[i];
                let h1 = hs[j - 1];
                for k in i..j {
                    let t = (xs[k] - x0) / (x1 - x0).max(1.0e-4);
                    let chord = h0 + (h1 - h0) * t;
                    max_s = max_s.max((hs[k] - chord).abs());
                }
            }
            if j + 1 < xs.len() && j >= 2 {
                let v1x = xs[j - 1] - xs[i];
                let v1y = hs[j - 1] - hs[i];
                let v2x = xs[j + 1] - xs[j];
                let v2y = hs[j + 1] - hs[j];
                let n1 = (v1x * v1x + v1y * v1y).sqrt();
                let n2 = (v2x * v2x + v2y * v2y).sqrt();
                if n1 > 1.0e-4 && n2 > 1.0e-4 {
                    let dot = ((v1x * v2x + v1y * v2y) / (n1 * n2)).clamp(-1.0, 1.0);
                    max_kink = max_kink.max(dot.acos().to_degrees());
                }
            }
            i = j.max(i + 1);
        }
        (max_s, max_kink)
    }

    #[test]
    fn rut_wall_is_sampled_from_the_solver_grid() {
        let (base_v, base_t) = uniform_terrain(192, 40.0, BASE_SUB);
        let base_i = pack(&base_t);
        let (dense_v, dense_i) = terrain_mesh(192, 40.0);
        let mut soil = silt::soil::SoilField::demo();
        for k in 0..10 {
            let z = 0.15 + k as f32 * 0.28;
            soil.stamp_wheel(
                Vec3::new(6.35, 0.3, z),
                0.38,
                0.28,
                Vec3::Z,
                Vec3::X,
                0.09,
                0.2,
                0.85,
                true,
            );
        }
        let z = 1.45_f32;
        let span = (5.55, 7.35);
        let base_xs = section_xs(&base_v, &base_i, z, span.0, span.1);
        let dense_xs = section_xs(&dense_v, &dense_i, z, span.0, span.1);
        let base_h: Vec<f32> = base_xs
            .iter()
            .map(|x| bilinear_height(&soil, *x, z))
            .collect();
        let dense_bi: Vec<f32> = dense_xs
            .iter()
            .map(|x| bilinear_height(&soil, *x, z))
            .collect();
        let dense_h: Vec<f32> = dense_xs.iter().map(|x| draw_height(&soil, *x, z)).collect();
        let (base_s, base_k) = max_cell_sagitta(&base_xs, &base_h, soil.cell);
        let bi_turn = max_turn_deg(&dense_xs, &dense_bi);
        let draw_turn = max_turn_deg(&dense_xs, &dense_h);
        let mut max_dev = 0.0f32;
        let mut bi_min = f32::MAX;
        let mut draw_at_bi_min = 0.0f32;
        for (x, (bi, draw)) in dense_xs.iter().zip(dense_bi.iter().zip(dense_h.iter())) {
            max_dev = max_dev.max((draw - bi).abs());
            if *bi < bi_min {
                bi_min = *bi;
                draw_at_bi_min = *draw;
            }
            let _ = x;
        }
        let slope_x = 6.70_f32;
        let numerical = (draw_height(&soil, slope_x + 0.002, z)
            - draw_height(&soil, slope_x - 0.002, z))
            / 0.004;
        let analytic = draw_slope_x(&soil, slope_x, z);
        let mut line = String::new();
        for (x, h) in dense_xs.iter().zip(dense_h.iter()) {
            line.push_str(&format!(" {:.3}:{:.4}", x, h));
        }
        eprintln!(
            "rut samples base {} dense {}",
            base_xs.len(),
            dense_xs.len()
        );
        eprintln!(
            "cell sagitta m bilinear {base_s:.5}  patch kink {base_k:.2}  turn deg bilinear {bi_turn:.2} draw {draw_turn:.2}  max |draw-bi| {max_dev:.4}  floor bi {bi_min:.4} draw {draw_at_bi_min:.4}"
        );
        eprintln!("slope dh/dx num {numerical:.4} ana {analytic:.4}");
        eprintln!("draw profile{line}");
        assert!(
            dense_xs.len() > base_xs.len() * 3,
            "rut section should carry the finer grid"
        );
        assert!(
            (analytic - numerical).abs() < 0.02,
            "draw slope {analytic:.4} disagrees with finite difference {numerical:.4}"
        );
        // The solver lip is still a corner. The draw height has to leave that ramp.
        assert!(
            bi_turn > 18.0,
            "expected the bilinear tyre lip to stay a sharp corner, got {bi_turn:.1}°"
        );
        assert!(
            draw_turn < 12.0 && draw_turn < bi_turn * 0.5,
            "draw lip still corners at {draw_turn:.1}° (bilinear {bi_turn:.1}°)"
        );
        assert!(
            max_dev > 0.008,
            "draw height stayed on the bilinear ramp (max deviation {max_dev:.4} m)"
        );
        assert!(
            (draw_at_bi_min - bi_min).abs() < 0.015,
            "rut floor moved from {bi_min:.4} to {draw_at_bi_min:.4}"
        );
        let _ = base_s;
    }

    #[test]
    fn draw_kernel_stays_normalized() {
        for t in [0.0_f32, 0.17, 0.5, 0.83] {
            let (w, dw) = draw_axis(t);
            let sum: f32 = w.iter().sum();
            let dsum: f32 = dw.iter().sum();
            assert!((sum - 1.0).abs() < 1.0e-4, "weights sum to {sum} at {t}");
            assert!(
                dsum.abs() < 1.0e-4,
                "weight derivative sums to {dsum} at {t}"
            );
        }
    }
}
