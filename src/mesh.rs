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

pub fn terrain_mesh(res: usize, size: f32) -> (Vec<TerrainVertex>, Vec<u32>) {
    // The solver stores one height per ~20 cm. A rut is often a single cell,
    // which is one kink and reads as a tint. Draw a finer mesh and sample the
    // same heightfield so the displaced walls have real triangles.
    let sub = 3usize;
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

fn push_clod(mesh: &mut CpuMesh, p: Vec3, n: Vec3, thickness: f32, phase: f32) {
    let n = n.normalize_or_zero();
    mesh.vertices.push(MeshVertex {
        pos: p.to_array(),
        nrm: n.to_array(),
        // uv.x is optical thickness in metres. The shader turns a thick core
        // dark and a thin edge into caramel. uv.y breaks up the wet highlight.
        uv: [thickness.max(0.004), phase],
        panel: PANEL_CLOD,
    });
}

fn add_tri_clod(
    mesh: &mut CpuMesh,
    a: Vec3,
    na: Vec3,
    b: Vec3,
    nb: Vec3,
    c: Vec3,
    nc: Vec3,
    center: Vec3,
    thickness: f32,
    phase: f32,
) {
    let face = (b - a).cross(c - a);
    if face.length_squared() < 1.0e-10 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    if face.dot(a - center) < 0.0 {
        push_clod(mesh, a, na, thickness, phase);
        push_clod(mesh, c, nc, thickness, phase);
        push_clod(mesh, b, nb, thickness, phase);
    } else {
        push_clod(mesh, a, na, thickness, phase);
        push_clod(mesh, b, nb, thickness, phase);
        push_clod(mesh, c, nc, thickness, phase);
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

/// One wet mass. Lobes come from a smooth radial field, so the silhouette is a
/// clod. `stretch` flings it along the flight without moving the simulation.
struct Lump {
    center: Vec3,
    rot: Quat,
    radius: f32,
    seed: u32,
    squash: Vec3,
    stretch_axis: Vec3,
    stretch: f32,
    thickness: f32,
    phase: f32,
}

fn hash_dir(seed: u32, n: u32) -> Vec3 {
    let z = chunk_hash(seed, n) * 2.0 - 1.0;
    let a = chunk_hash(seed, n + 1) * std::f32::consts::TAU;
    let r = (1.0 - z * z).max(0.0).sqrt();
    Vec3::new(r * a.cos(), z, r * a.sin())
}

impl Lump {
    fn from_particle(p: &silt::particles::Particle, radius: f32, thickness: f32) -> Self {
        let mut squash = Vec3::new(
            0.82 + chunk_hash(p.seed, 1) * 0.36,
            0.78 + chunk_hash(p.seed, 2) * 0.40,
            0.84 + chunk_hash(p.seed, 3) * 0.32,
        );
        let mut center = p.pos;
        if p.grounded {
            // Spread into the soil. A tall axis reads as a stone on the mud.
            squash.x *= 1.42;
            squash.y *= 0.46;
            squash.z *= 1.42;
            center -= Vec3::Y * radius * 0.14;
        }
        let (stretch_axis, stretch) = if p.grounded {
            (Vec3::Y, 1.0)
        } else {
            let speed = p.vel.length();
            if speed < 0.35 {
                (Vec3::Y, 1.0)
            } else {
                ((p.vel / speed), (1.0 + speed * 0.06).clamp(1.04, 1.32))
            }
        };
        Self {
            center,
            rot: p.rot,
            radius,
            seed: p.seed,
            squash,
            stretch_axis,
            stretch,
            thickness,
            phase: chunk_hash(p.seed, 6),
        }
    }

    fn radius_at(&self, dir: Vec3) -> f32 {
        let dir = dir.normalize_or_zero();
        let mut r = 0.72;
        for k in 0..5u32 {
            let axis = hash_dir(self.seed, 11 + k * 4);
            let amp = 0.22 + chunk_hash(self.seed, 60 + k) * 0.62;
            let d = dir.dot(axis).max(0.0);
            // Cubed so the bulge stays on one side instead of rounding into a ball.
            r += amp * d * d * d;
        }
        let pinch = hash_dir(self.seed, 4);
        let cut = dir.dot(pinch);
        r *= 1.0 - 0.22 * cut * cut;
        self.radius * r.max(0.45)
    }

    fn place(&self, dir: Vec3) -> Vec3 {
        let dir = dir.normalize_or_zero();
        let rad = self.radius_at(dir);
        let shaped = Vec3::new(
            dir.x * rad * self.squash.x,
            dir.y * rad * self.squash.y,
            dir.z * rad * self.squash.z,
        );
        let spun = self.rot * shaped;
        let along = spun.dot(self.stretch_axis);
        self.center + spun + self.stretch_axis * along * (self.stretch - 1.0)
    }

    fn sample(&self, dir: Vec3) -> (Vec3, Vec3) {
        let dir = dir.normalize_or_zero();
        let p = self.place(dir);
        let helper = if dir.y.abs() > 0.85 { Vec3::X } else { Vec3::Y };
        let t = dir.cross(helper).normalize_or_zero();
        let b = dir.cross(t).normalize_or_zero();
        let e = 0.14;
        let px = self.place(dir + t * e);
        let py = self.place(dir + b * e);
        let mut n = (px - p).cross(py - p);
        if n.dot(p - self.center) < 0.0 {
            n = -n;
        }
        if n.length_squared() < 1.0e-8 {
            n = p - self.center;
        }
        (p, n.normalize_or_zero())
    }
}

fn emit_lump(mesh: &mut CpuMesh, lump: &Lump, stacks: usize, slices: usize) {
    if stacks < 3 || slices < 3 {
        return;
    }
    let mut pts = Vec::with_capacity(2 + (stacks - 1) * slices);
    let mut nrm = Vec::with_capacity(pts.capacity());
    let (p, n) = lump.sample(Vec3::Y);
    pts.push(p);
    nrm.push(n);
    for s in 1..stacks {
        let phi = s as f32 / stacks as f32 * std::f32::consts::PI;
        for i in 0..slices {
            let th = i as f32 / slices as f32 * std::f32::consts::TAU;
            let dir = Vec3::new(th.cos() * phi.sin(), phi.cos(), th.sin() * phi.sin());
            let (p, n) = lump.sample(dir);
            pts.push(p);
            nrm.push(n);
        }
    }
    let (p, n) = lump.sample(-Vec3::Y);
    pts.push(p);
    nrm.push(n);
    let ring = |s: usize, i: usize| 1 + (s - 1) * slices + (i % slices);
    let tri = |mesh: &mut CpuMesh, a: usize, b: usize, c: usize| {
        add_tri_clod(
            mesh,
            pts[a],
            nrm[a],
            pts[b],
            nrm[b],
            pts[c],
            nrm[c],
            lump.center,
            lump.thickness,
            lump.phase,
        );
    };
    for i in 0..slices {
        tri(mesh, 0, ring(1, i), ring(1, i + 1));
    }
    for s in 1..stacks.saturating_sub(1) {
        for i in 0..slices {
            let a = ring(s, i);
            let b = ring(s, i + 1);
            let c = ring(s + 1, i);
            let d = ring(s + 1, i + 1);
            tri(mesh, a, c, b);
            tri(mesh, b, c, d);
        }
    }
    let bot = pts.len() - 1;
    let last = stacks - 1;
    for i in 0..slices {
        tri(mesh, bot, ring(last, i + 1), ring(last, i));
    }
}

fn emit_piece(mesh: &mut CpuMesh, p: &silt::particles::Particle, stacks: usize, slices: usize) {
    let lump = Lump::from_particle(p, p.size, p.size * if p.grounded { 0.2 } else { 0.5 });
    emit_lump(mesh, &lump, stacks, slices);
}

/// Solid part of the splash. A big impact is a body with lobes, so the mass
/// has thickness. Droplets that have landed stay as small pats. Droplets still
/// flying are drawn as soft spray.
pub fn mud_chunks(items: &[silt::particles::Particle]) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    const MAX_VERTS: usize = 14000;
    const MAX_INDICES: usize = 46000;
    for kind in [3u8, 1, 4, 0, 2] {
        for p in items {
            if !p.alive || p.life <= 0.0 || p.kind != kind {
                continue;
            }
            // Airborne droplets are the soft spray, not another pebble.
            if matches!(p.kind, 0 | 2 | 4) && !p.grounded {
                continue;
            }
            if mesh.vertices.len() + 500 > MAX_VERTS || mesh.indices.len() + 1800 > MAX_INDICES {
                return mesh;
            }
            if p.kind == 3 {
                emit_piece(&mut mesh, p, 8, 14);
                for (slot, scale, dist) in
                    [(8u32, 0.64, 0.62), (13u32, 0.48, 0.5), (21u32, 0.4, 0.42)]
                {
                    let dir = hash_dir(p.seed, slot);
                    let lobe = silt::particles::Particle {
                        pos: p.pos + p.rot * dir * p.size * dist,
                        size: p.size * scale,
                        seed: p.seed ^ 0xA5A5_5A5Au32.wrapping_mul(slot | 1),
                        ..*p
                    };
                    emit_piece(&mut mesh, &lobe, 5, 10);
                }
            } else if p.kind == 1 {
                let mut q = *p;
                q.size = p.size.max(0.09);
                emit_piece(&mut mesh, &q, 5, 9);
                if !p.grounded {
                    let dir = hash_dir(p.seed, 17);
                    let lobe = silt::particles::Particle {
                        pos: q.pos + q.rot * dir * q.size * 0.32,
                        size: q.size * 0.62,
                        seed: q.seed ^ 0x3C3C_3C3C,
                        ..q
                    };
                    emit_piece(&mut mesh, &lobe, 4, 8);
                }
            } else {
                let mut q = *p;
                q.size = if p.kind == 2 {
                    p.size.max(0.04)
                } else {
                    p.size.max(0.06)
                };
                emit_piece(&mut mesh, &q, 4, 8);
            }
        }
    }
    mesh
}

fn cam_axes(eye: Vec3, center: Vec3) -> (Vec3, Vec3, Vec3) {
    let to_cam = (eye - center).normalize_or_zero();
    let mut right = Vec3::Y.cross(to_cam);
    if right.length_squared() < 1.0e-6 {
        right = Vec3::X.cross(to_cam);
    }
    right = right.normalize_or_zero();
    let up = to_cam.cross(right).normalize_or_zero();
    (right, up, to_cam)
}

fn push_sprite(
    out: &mut Vec<ParticleVertex>,
    center: Vec3,
    axis_u: Vec3,
    axis_v: Vec3,
    radius: f32,
    alpha: f32,
    sheet: bool,
    seed: f32,
) {
    if out.len() + 6 > 6144 || radius < 0.004 || alpha <= 0.02 {
        return;
    }
    let color = [
        radius,
        if sheet { 1.0 } else { 0.0 },
        seed.clamp(0.0, 1.0),
        alpha.clamp(0.0, 0.92),
    ];
    let corner = |u: f32, v: f32, su: f32, sv: f32| ParticleVertex {
        pos: (center + axis_u * u + axis_v * v).to_array(),
        color,
        uv: [su, sv],
    };
    let hx = radius;
    let hy = radius;
    // Two triangles, camera-facing. The shader feathers the ellipse.
    let verts = [
        corner(-hx, -hy, 0.0, 0.0),
        corner(hx, -hy, 1.0, 0.0),
        corner(hx, hy, 1.0, 1.0),
        corner(-hx, -hy, 0.0, 0.0),
        corner(hx, hy, 1.0, 1.0),
        corner(-hx, hy, 0.0, 1.0),
    ];
    out.extend_from_slice(&verts);
}

fn push_bead(
    out: &mut Vec<ParticleVertex>,
    center: Vec3,
    eye: Vec3,
    vel: Vec3,
    radius: f32,
    alpha: f32,
    seed: f32,
) {
    let (right, up, to_cam) = cam_axes(eye, center);
    let planar = vel - to_cam * vel.dot(to_cam);
    let speed = planar.length();
    let (axis_u, axis_v, rx, ry) = if speed > 0.25 {
        let along = planar / speed;
        let side = to_cam.cross(along).normalize_or_zero();
        let stretch = (speed / 2.6).clamp(1.0, 2.3);
        (
            along,
            side,
            radius * stretch,
            radius * (0.82 / stretch.sqrt()),
        )
    } else {
        (right, up, radius, radius)
    };
    // Rebuild through push_sprite's uniform radius by scaling the axes.
    let axis_u = axis_u * (rx / radius.max(0.001));
    let axis_v = axis_v * (ry / radius.max(0.001));
    push_sprite(
        out,
        center + to_cam * 0.01,
        axis_u,
        axis_v,
        radius,
        alpha,
        false,
        seed,
    );
}

fn push_sheet_sprite(
    out: &mut Vec<ParticleVertex>,
    center: Vec3,
    eye: Vec3,
    vel: Vec3,
    radius: f32,
    alpha: f32,
    seed: f32,
) {
    let (_right, _up, to_cam) = cam_axes(eye, center);
    let planar = vel - to_cam * vel.dot(to_cam);
    let along = if planar.length() > 0.2 {
        planar.normalize()
    } else {
        cam_axes(eye, center).0
    };
    let side = to_cam.cross(along).normalize_or_zero();
    push_sprite(
        out,
        center + to_cam * 0.02,
        along * 1.7,
        side * 0.72,
        radius,
        alpha,
        true,
        seed,
    );
}

/// Soft part of the splash. These are camera-facing beads and thin sheets.
/// The solid clumps stay in `mud_chunks`. Nothing here changes the flight.
pub fn mud_spray(items: &[silt::particles::Particle], eye: Vec3) -> Vec<ParticleVertex> {
    let mut out = Vec::new();
    for p in items {
        if !p.alive || p.life <= 0.0 || p.grounded {
            continue;
        }
        if out.len() + 6 > 6000 {
            break;
        }
        let life = (p.life / p.max_life.max(0.05)).clamp(0.15, 1.0);
        let seed = chunk_hash(p.seed, 3);
        match p.kind {
            0 => push_bead(
                &mut out,
                p.pos,
                eye,
                p.vel,
                p.size.max(0.05) * 1.35,
                0.5 * life,
                seed,
            ),
            2 => push_bead(
                &mut out,
                p.pos,
                eye,
                p.vel,
                p.size.max(0.03) * 1.2,
                0.62 * life,
                seed,
            ),
            4 => push_bead(
                &mut out,
                p.pos,
                eye,
                p.vel,
                p.size.max(0.05) * 0.85,
                0.9 * life,
                seed,
            ),
            3 => {
                for i in 0..14u32 {
                    let mut dir = hash_dir(p.seed, 80 + i * 3);
                    dir.y = dir.y.abs() * 0.55 + 0.2;
                    dir = dir.normalize_or_zero();
                    if p.vel.length() > 0.3 {
                        dir = (dir + p.vel.normalize_or_zero() * 0.85).normalize_or_zero();
                    }
                    // Outside the solid mass, so the beads read as spray and not bubbles on the skin.
                    let dist = p.size * (1.25 + chunk_hash(p.seed, 90 + i) * 1.6);
                    let radius = p.size * (0.045 + chunk_hash(p.seed, 100 + i) * 0.07);
                    let mist = i % 3 == 0;
                    push_bead(
                        &mut out,
                        p.pos + dir * dist,
                        eye,
                        p.vel,
                        radius.max(0.018),
                        if mist { 0.28 } else { 0.72 } * life,
                        chunk_hash(p.seed, 110 + i),
                    );
                }
                let side = {
                    let s = p.vel.cross(Vec3::Y);
                    if s.length_squared() < 1.0e-4 {
                        Vec3::X
                    } else {
                        s.normalize()
                    }
                };
                push_sheet_sprite(
                    &mut out,
                    p.pos + side * p.size * 0.9 + p.vel.normalize_or_zero() * p.size * 0.35,
                    eye,
                    p.vel,
                    p.size * 0.7,
                    0.26 * life,
                    seed,
                );
            }
            1 => {
                for i in 0..4u32 {
                    let dir = hash_dir(p.seed, 40 + i * 2);
                    push_bead(
                        &mut out,
                        p.pos + dir * p.size * 0.8,
                        eye,
                        p.vel,
                        p.size * 0.28,
                        0.46 * life,
                        chunk_hash(p.seed, 50 + i),
                    );
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod chunk_tests {
    use super::*;
    use silt::particles::{Particle, Particles};

    fn piece(kind: u8, grounded: bool) -> Particle {
        Particle {
            pos: Vec3::new(0.0, 1.2, 0.0),
            vel: Vec3::new(0.6, 3.2, 0.15),
            life: 1.4,
            max_life: 2.5,
            size: 0.42,
            kind,
            alive: true,
            seed: 0x51_17,
            spin: Vec3::new(0.2, 0.4, 0.1),
            rot: Quat::from_rotation_y(0.4),
            grounded,
            deposited: false,
        }
    }

    fn bounds(mesh: &CpuMesh) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in &mesh.vertices {
            let p = Vec3::from_array(v.pos);
            min = min.min(p);
            max = max.max(p);
        }
        (min, max)
    }

    #[test]
    fn flying_chunk_has_volume_and_thickness() {
        let mesh = mud_chunks(&[piece(3, false)]);
        assert!(
            mesh.vertices.len() > 120,
            "lobes and a sheet should thicken the clod, verts {}",
            mesh.vertices.len()
        );
        assert_eq!(mesh.indices.len() % 3, 0);
        let (min, max) = bounds(&mesh);
        let ext = max - min;
        assert!(
            ext.x > 0.35 && ext.y > 0.28 && ext.z > 0.28,
            "clod is a volume, got {ext}"
        );
        assert!(
            ext.y > 0.2,
            "flying clod should keep a wet thickness, got {}",
            ext.y
        );
        for v in &mesh.vertices {
            assert!((v.panel - PANEL_CLOD).abs() < 0.01);
            assert!(v.uv[0] > 0.01 && v.uv[0] < 0.8, "thickness {}", v.uv[0]);
            let n = Vec3::from_array(v.nrm);
            assert!((n.length() - 1.0).abs() < 0.08, "normal len {}", n.length());
        }
    }

    #[test]
    fn landed_chunk_slumps_but_stays_thick() {
        let mesh = mud_chunks(&[piece(3, true)]);
        let (min, max) = bounds(&mesh);
        let ext = max - min;
        assert!(
            ext.y < ext.x && ext.y < ext.z,
            "landed mud spreads, got {ext}"
        );
        assert!(
            ext.y > 0.08,
            "landed mud keeps a wet thickness, got {}",
            ext.y
        );
    }

    #[test]
    fn airborne_droplets_are_soft_sprites() {
        let p = piece(4, false);
        let mesh = mud_chunks(&[p]);
        assert!(
            mesh.vertices.is_empty(),
            "droplets in the air are spray, not solid pebbles"
        );
        let spray = mud_spray(&[p], Vec3::new(3.0, 2.0, 4.0));
        assert!(spray.len() >= 6);
        assert_eq!(spray.len() % 6, 0);
        let mut alpha = 0.0_f32;
        for v in &spray {
            assert!((0.0..=1.0).contains(&v.uv[0]));
            assert!((0.0..=1.0).contains(&v.uv[1]));
            alpha = alpha.max(v.color[3]);
        }
        assert!(alpha > 0.4 && alpha < 1.0, "soft alpha {alpha}");
    }

    #[test]
    fn impact_burst_builds_chunks_and_spray() {
        let mut sys = Particles::new();
        let n = sys.impact_burst(Vec3::new(0.0, 0.3, 0.0));
        assert!(n >= 20);
        let mesh = mud_chunks(&sys.items);
        let spray = mud_spray(&sys.items, Vec3::new(2.0, 1.5, 2.0));
        assert!(mesh.vertices.len() > 400, "chunks {}", mesh.vertices.len());
        assert!(spray.len() > 60, "spray {}", spray.len());
        assert!(spray.len() <= 6144);
        assert!(mesh.vertices.len() <= 14000);
        assert!(mesh.indices.len() <= 46000);
    }

    /// Offscreen stills of the wet splash. Needs lavapipe (`-- --ignored`).
    #[test]
    #[ignore]
    fn capture_mud_pbr_stills() {
        use crate::render::{
            gpu_formats, open_adapter, prepare_software_vulkan, write_png, Renderer,
        };
        use silt::world::{CameraPose, World};
        use std::path::PathBuf;

        let software = std::env::var_os("SILT_SOFTWARE").is_some();
        if software {
            prepare_software_vulkan();
        }
        let (_instance, adapter, device, queue) = open_adapter(software, None);
        let formats = gpu_formats(&adapter);
        let width = 1280u32;
        let height = 720u32;
        let mut renderer = Renderer::new(
            device,
            queue,
            adapter.get_info().name,
            width,
            height,
            formats,
        );
        let mut world = World::flat(0.92);
        world.vehicle.pos = Vec3::new(30.0, 8.0, 30.0);
        world.bike.pos = Vec3::new(-30.0, 8.0, 30.0);
        let h = world.soil.sample(0.0, 0.0).height;
        world.particles.impact_burst(Vec3::new(0.0, h + 0.12, 0.0));
        let fwd = Vec3::new(0.2, 0.0, 1.0).normalize();
        for _ in 0..6 {
            world
                .particles
                .clod(Vec3::new(0.05, h + 0.18, 0.0), fwd, Vec3::X, 2.2);
        }
        let media = PathBuf::from("/cursor/stores/bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e/media");
        let shoot = |world: &World, renderer: &mut Renderer, name: &str, back: f32, lift: f32| {
            let mut min = Vec3::splat(f32::MAX);
            let mut max = Vec3::splat(f32::MIN);
            let mut focus = Vec3::ZERO;
            let mut n = 0.0_f32;
            for p in &world.particles.items {
                if !(p.alive && matches!(p.kind, 1 | 3 | 4)) {
                    continue;
                }
                let rad = p.size.max(0.2);
                min = min.min(p.pos - Vec3::splat(rad));
                max = max.max(p.pos + Vec3::splat(rad));
                if p.kind == 3 {
                    focus += p.pos;
                    n += 1.0;
                }
            }
            if n < 1.0 {
                focus = (min + max) * 0.5;
            } else {
                focus /= n;
            }
            let ext = (max - min).max_element().max(1.3);
            let eye = focus + Vec3::new(ext * back, ext * lift, ext * back * 1.08);
            let pose = CameraPose {
                eye,
                target: focus + Vec3::new(0.0, -ext * 0.08, 0.0),
                ortho_width: None,
            };
            eprintln!("capture {name} focus {focus:?} ext {ext:.2} eye {eye:?}");
            let rgba = renderer.draw(world, pose, None, true).expect("readback");
            let dest = media.join(name);
            write_png(&dest, width, height, &rgba).expect("png");
            eprintln!("wrote {}", dest.display());
        };
        for _ in 0..10 {
            world.particles.update(&mut world.soil, 1.0 / 60.0);
        }
        shoot(&world, &mut renderer, "mud-pbr-splash.png", 1.65, 0.34);
        for _ in 0..20 {
            world.particles.update(&mut world.soil, 1.0 / 60.0);
        }
        shoot(&world, &mut renderer, "mud-pbr-chunks.png", 1.02, 0.3);
    }
}
