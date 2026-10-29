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
    // sits on the ground instead of floating as boxes. Trees, rocks and grass
    // are the Gobkit meshes placed around the oval.
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

/// Liquid slurry torn off a tire. A curled sheet, pinched at both ends, so it
/// is a ribbon rather than a box, a tablet, or a cube.
fn add_slurry_ribbon(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let mut dir = Vec3::new(p.vel.x, p.vel.y * 0.35, p.vel.z);
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
    let speed = p.vel.length();
    let length = (0.34 + speed * 0.04).clamp(0.42, 0.78);
    let width = (p.volume / length).sqrt().clamp(0.022, 0.10);
    let thick = (p.volume / (length * width * 0.55)).clamp(0.004, 0.014);
    let segs = 8usize;
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        // Pinched ends. A constant width would be a tablet.
        let taper = (t * (1.0 - t) * 4.0).powf(0.55).clamp(0.08, 1.0);
        let arc = (t * std::f32::consts::PI).sin();
        let twist = (t - 0.5) * 0.7;
        let side_t = side * twist.cos() + up * twist.sin();
        let up_t = -side * twist.sin() + up * twist.cos();
        let c = p.pos
            + dir * (length * (t - 0.5))
            + up * (arc - 0.55) * length * 0.16
            + side * (t - 0.5) * width * 0.35;
        let w = side_t * width * 0.5 * taper;
        let h = up_t * thick * 0.5 * (0.45 + 0.55 * taper);
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

/// A roll's slurry. Wider than the spin ribbon, still pinched and thin, so it
/// reads as a sheet and not a tablet, a sphere, or a teardrop.
fn add_slurry_sheet(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let (dir, side, up) = slurry_sheet_axes(p.vel);
    let length = 1.15_f32;
    let width = 0.16_f32;
    let thick = 0.005_f32;
    let segs = 8usize;
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let taper = (t * (1.0 - t) * 4.0).powf(0.35).clamp(0.22, 1.0);
        let arc = (t * std::f32::consts::PI).sin();
        // Bow in the face, so the silhouette is a curled sheet and not a lens.
        let c = p.pos
            + dir * (length * (t - 0.5))
            + side * (arc - 0.62) * 0.10
            + up * (arc - 0.55) * 0.02;
        let w = side * width * 0.5 * taper;
        let h = up * thick * 0.5 * (0.5 + 0.5 * taper);
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

/// A small liquid drop torn off with the ribbons. A short pinched fleck.
/// The ledger volume stays on the particle; this mark does not swell into a
/// teardrop blob, a sphere, or a tablet.
fn add_mud_drop(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    // Long axis in the horizontal plane, width vertical, thickness to the side.
    // A side camera then sees a small fleck instead of an edge-on hair.
    let mut dir = Vec3::new(p.vel.x * 0.2, 0.0, p.vel.z);
    if dir.length_squared() < 1.0e-4 {
        dir = Vec3::new(p.vel.x, 0.0, -1.0);
    }
    dir = dir.normalize();
    let up = Vec3::Y;
    let side = dir.cross(up).normalize();
    let wobble = (p.seed % 5) as f32;
    let length = 0.048 + wobble * 0.003;
    let width = 0.016;
    let thick = 0.0026;
    let segs = 4usize;
    let mut ring: Vec<[Vec3; 4]> = Vec::with_capacity(segs + 1);
    for s in 0..=segs {
        let t = s as f32 / segs as f32;
        let taper = (t * (1.0 - t) * 4.0).powf(0.7).clamp(0.16, 1.0);
        let arc = (t * std::f32::consts::PI).sin();
        let c = p.pos + dir * (length * (t - 0.5)) + up * (arc - 0.5) * length * 0.06;
        let w = side * width * 0.5 * taper;
        let h = up * thick * 0.5 * (0.55 + 0.45 * taper);
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
                sheet_frame(&mesh, p.vel)
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
                    e[1] > 0.14 && e[1] > e[2] * 3.0,
                    "roll sheet is a hair or a tablet {e:?}"
                );
            }
        }
    }

    #[test]
    fn mud_chunks_includes_the_breakaway() {
        use silt::particles::Particles;
        let mut particles = Particles::new();
        let threw = particles.throw_slurry(Vec3::new(0.4, 0.2, 0.6), Vec3::Z, Vec3::X, 0.012);
        assert!(threw.ribbons > 0.0 && threw.drops > 0.0);
        let mesh = mud_chunks(&particles.items);
        assert!(
            mesh.vertices.len() > 80,
            "breakaway pieces were not meshed, verts {}",
            mesh.vertices.len()
        );
    }
}

/// Soft wet mud in world space. Large impacts are a pair of lumps; droplets are
/// smaller lumps. Ledger soil is not in this mesh once it has landed: that
/// volume is the heightfield. Nothing here is a camera-facing quad or a faceted stone.
pub fn mud_chunks(items: &[silt::particles::Particle]) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    const MAX_VERTS: usize = 14000;
    // Draw the big clumps first so a full splash cannot crowd them out.
    // 7 is a roll sheet, 5 and 6 are the slurry breakaway: ribbons, then drops.
    for kind in [7u8, 5, 6, 3, 1, 4, 0, 2] {
        for p in items {
            if !p.alive || p.life <= 0.0 || p.kind != kind {
                continue;
            }
            if p.ledger {
                if !p.deposited && p.volume > 1.0e-8 && mesh.vertices.len() + 400 < MAX_VERTS {
                    if p.kind == silt::particles::KIND_SLURRY_DROP {
                        add_mud_drop(&mut mesh, p);
                    } else if p.kind == silt::particles::KIND_SLURRY_SHEET {
                        add_slurry_sheet(&mut mesh, p);
                    } else if p.kind == silt::particles::KIND_SLURRY_RIBBON {
                        add_slurry_ribbon(&mut mesh, p);
                    } else {
                        add_soil_ribbon(&mut mesh, p);
                    }
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
