//! Procedural buggy, wheels and a small patch of countryside.
//!
//! Geometry is flat-shaded and carries a panel id. The shader turns that id
//! into paint, rubber or mud. Wheel and shock meshes stay in the spaces the
//! vehicle solver already uses, so the picture matches the contact model.

use glam::Vec3;

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
/// Legacy id. Flying soil is clay, mud, gravel, or dust.
#[allow(dead_code)]
pub const PANEL_CLOD: f32 = 46.0;
pub const PANEL_CLAY: f32 = 50.0;
pub const PANEL_MUD: f32 = 51.0;
pub const PANEL_GRAVEL: f32 = 52.0;
pub const PANEL_DUST: f32 = 53.0;

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

/// Torn soil. Extents come from the piece volume, so a ribbon is not a ball
/// and a clod is not a sprite.

fn soil_panel(p: &silt::particles::Particle) -> f32 {
    if p.shape == silt::particles::SHAPE_DUST {
        PANEL_DUST
    } else if p.face == silt::particles::FACE_MUD {
        PANEL_MUD
    } else if p.face == silt::particles::FACE_GRAVEL {
        PANEL_GRAVEL
    } else {
        PANEL_CLAY
    }
}

fn soil_uv(p: &silt::particles::Particle) -> [f32; 2] {
    [p.wet.clamp(0.0, 1.0), p.dark.clamp(0.0, 1.0)]
}

fn add_tri_tint(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    panel: f32,
    uv: [f32; 2],
    outward_from: Vec3,
) {
    let mut b = b;
    let mut c = c;
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1.0e-12 {
        return;
    }
    let mid = (a + b + c) / 3.0;
    if n.dot(mid - outward_from) < 0.0 {
        std::mem::swap(&mut b, &mut c);
    }
    let n = (b - a).cross(c - a).normalize_or_zero();
    let base = mesh.vertices.len() as u32;
    for p in [a, b, c] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv,
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn add_tri_double(mesh: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3, panel: f32, uv: [f32; 2]) {
    let n = (b - a).cross(c - a);
    if n.length_squared() < 1.0e-12 {
        return;
    }
    let n = n.normalize_or_zero();
    let base = mesh.vertices.len() as u32;
    for (p, normal) in [(a, n), (b, n), (c, n), (a, -n), (c, -n), (b, -n)] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: normal.to_array(),
            uv,
            panel,
        });
    }
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base + 3, base + 4, base + 5]);
}

fn world_of(p: &silt::particles::Particle, local: Vec3) -> Vec3 {
    let mut q = local;
    if p.grounded {
        q.y *= 0.32;
    } else {
        // The mesh trails along the throw. Centering it put half the ribbon
        // back through the tire.
        let shift = match p.shape {
            silt::particles::SHAPE_RIBBON | silt::particles::SHAPE_DROP => 0.48,
            silt::particles::SHAPE_CLOD => 0.3,
            _ => 0.22,
        };
        q.x += p.span.x * shift;
    }
    p.pos + p.rot * q
}

fn add_torn_clod(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let radii = p.span.max(Vec3::splat(0.004));
    let mut v = [Vec3::ZERO; 6];
    let dirs = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    for (i, d) in dirs.iter().enumerate() {
        let j = 0.68 + chunk_hash(p.seed, i as u32) * 0.58;
        let tear = if i == 0 {
            1.48
        } else if i == 5 {
            0.58
        } else if i == 3 {
            0.72
        } else {
            1.0
        };
        v[i] = Vec3::new(d.x * radii.x, d.y * radii.y, d.z * radii.z) * j * tear;
        v[i].x += v[i].y * (0.28 + chunk_hash(p.seed, 8) * 0.2);
        v[i].z += v[i].x * (0.12 + chunk_hash(p.seed, 9) * 0.22);
    }
    // A bite out of one face, so the silhouette is a torn slab.
    v[4] *= 0.48 + chunk_hash(p.seed, 11) * 0.28;
    let faces = [
        (0usize, 4, 2),
        (4, 1, 2),
        (1, 5, 2),
        (5, 0, 2),
        (4, 0, 3),
        (1, 4, 3),
        (5, 1, 3),
        (0, 5, 3),
    ];
    let panel = soil_panel(p);
    let uv = soil_uv(p);
    for (k, (ia, ib, ic)) in faces.iter().enumerate() {
        let a = v[*ia];
        let b = v[*ib];
        let c = v[*ic];
        let mid = (a + b + c) / 3.0;
        let n = (b - a).cross(c - a).normalize_or_zero();
        let kick = (chunk_hash(p.seed, 20 + k as u32) - 0.42) * radii.y * 0.95;
        let m = mid + n * kick;
        let aw = world_of(p, a);
        let bw = world_of(p, b);
        let cw = world_of(p, c);
        let mw = world_of(p, m);
        add_tri_tint(mesh, aw, bw, mw, panel, uv, p.pos);
        add_tri_tint(mesh, bw, cw, mw, panel, uv, p.pos);
        add_tri_tint(mesh, cw, aw, mw, panel, uv, p.pos);
    }
}

fn add_ribbon(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let length = p.span.x.max(0.02);
    let width = p.span.y.max(0.008);
    let thick = p.span.z.max(0.003);
    let segs = 6usize;
    let panel = soil_panel(p);
    let uv = soil_uv(p);
    let mut ring = Vec::with_capacity(segs + 1);
    for i in 0..=segs {
        let t = i as f32 / segs as f32;
        let edge = if i == 0 || i == segs { 0.18 } else { 1.0 };
        let belly = (t * (1.0 - t) * 4.0).powf(0.6).max(0.32);
        let tear = 0.62 + chunk_hash(p.seed, i as u32) * 0.55;
        let w = width * 0.5 * belly * tear * edge;
        let th = thick * 0.5 * (0.55 + 0.45 * belly) * edge;
        let x = (t - 0.5) * length;
        let curl = ((t - 0.32) * 2.1).sin() * thick * 0.85;
        let jag = (chunk_hash(p.seed, 24 + i as u32) - 0.5) * width * 0.16;
        let twist = (t - 0.5) * 0.45 + (chunk_hash(p.seed, 6) - 0.5) * 0.2;
        let (s, c) = twist.sin_cos();
        let rot = |y: f32, z: f32| Vec3::new(x, y * c - z * s + curl, y * s + z * c);
        let top = w * (0.85 + chunk_hash(p.seed, 40 + i as u32) * 0.4);
        let bot = w * (0.6 + chunk_hash(p.seed, 50 + i as u32) * 0.55);
        ring.push([
            rot(top + jag, th * 0.15),
            rot(jag * 0.25, th),
            rot(-bot + jag * 0.35, -th * 0.1),
            rot(jag * 0.1, -th * 0.85),
        ]);
    }
    let pairs = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
    for i in 0..segs {
        for (u, v) in pairs {
            let a = world_of(p, ring[i][u]);
            let b = world_of(p, ring[i + 1][u]);
            let c = world_of(p, ring[i + 1][v]);
            let d = world_of(p, ring[i][v]);
            add_tri_double(mesh, a, b, c, panel, uv);
            add_tri_double(mesh, a, c, d, panel, uv);
        }
    }
}

fn add_drop(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let rx = p.span.x.max(0.008);
    let ry = p.span.y.max(0.004);
    let rz = p.span.z.max(0.003);
    let panel = soil_panel(p);
    let uv = soil_uv(p);
    // A pinched triangular tail, not a lathed ball.
    let stations = [0.0_f32, 0.18, 0.48, 0.78, 1.0];
    let scales = [0.05_f32, 0.28, 1.0, 0.62, 0.16];
    let mut rings: Vec<[Vec3; 3]> = Vec::new();
    let yaw = chunk_hash(p.seed, 3) * 0.8;
    for (i, (t, scale)) in stations.iter().zip(scales).enumerate() {
        let jag = 0.75 + chunk_hash(p.seed, i as u32) * 0.45;
        let s = scale * jag;
        let x = (t - 0.5) * 2.0 * rx;
        let kick = (chunk_hash(p.seed, 14 + i as u32) - 0.35) * ry * 0.45 * s;
        let mut ring = [Vec3::ZERO; 3];
        for k in 0..3 {
            let a = yaw + k as f32 / 3.0 * std::f32::consts::TAU;
            let rad_y = if k == 0 { ry * 1.25 } else { ry * 0.72 };
            let rad_z = if k == 1 { rz * 1.15 } else { rz * 0.7 };
            ring[k] = Vec3::new(x, kick + a.sin() * rad_y * s, a.cos() * rad_z * s);
        }
        rings.push(ring);
    }
    for i in 0..rings.len() - 1 {
        for k in 0..3 {
            let k2 = (k + 1) % 3;
            let a = world_of(p, rings[i][k]);
            let b = world_of(p, rings[i][k2]);
            let c = world_of(p, rings[i + 1][k2]);
            let d = world_of(p, rings[i + 1][k]);
            add_tri_double(mesh, a, b, c, panel, uv);
            add_tri_double(mesh, a, c, d, panel, uv);
        }
    }
    let mut cap = |ring: [Vec3; 3]| {
        add_tri_double(
            mesh,
            world_of(p, ring[0]),
            world_of(p, ring[1]),
            world_of(p, ring[2]),
            panel,
            uv,
        );
    };
    cap(rings[0]);
    cap(*rings.last().unwrap());
}

fn add_dust(mesh: &mut CpuMesh, p: &silt::particles::Particle) {
    let rx = p.span.x.max(0.008) * 0.5;
    let hy = p.span.y.max(0.001) * 0.5;
    let rz = p.span.z.max(0.006) * 0.5;
    let panel = soil_panel(p);
    let uv = soil_uv(p);
    let n = 5usize;
    let mut top = Vec::with_capacity(n);
    let mut bot = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f32 / n as f32 * std::f32::consts::TAU + 0.35;
        let j = 0.5 + chunk_hash(p.seed, i as u32) * 0.7;
        let x = a.cos() * rx * j;
        let z = a.sin() * rz * j * (0.75 + chunk_hash(p.seed, 10 + i as u32) * 0.4);
        top.push(Vec3::new(x, hy, z));
        bot.push(Vec3::new(x * 0.92, -hy, z * 0.92));
    }
    for i in 1..n - 1 {
        add_tri_double(
            mesh,
            world_of(p, top[0]),
            world_of(p, top[i]),
            world_of(p, top[i + 1]),
            panel,
            uv,
        );
        add_tri_double(
            mesh,
            world_of(p, bot[0]),
            world_of(p, bot[i + 1]),
            world_of(p, bot[i]),
            panel,
            uv,
        );
    }
    for i in 0..n {
        let j = (i + 1) % n;
        add_tri_double(
            mesh,
            world_of(p, top[i]),
            world_of(p, bot[i]),
            world_of(p, top[j]),
            panel,
            uv,
        );
        add_tri_double(
            mesh,
            world_of(p, top[j]),
            world_of(p, bot[i]),
            world_of(p, bot[j]),
            panel,
            uv,
        );
    }
}

/// Irregular chunks, ribbons, drops, and dust. Nothing here is a sphere or a
/// camera-facing card.
pub fn mud_chunks(items: &[silt::particles::Particle]) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    const MAX_VERTS: usize = 15000;
    let mut order: Vec<usize> = (0..items.len())
        .filter(|&i| items[i].alive && items[i].life > 0.0)
        .collect();
    order.sort_by(|&a, &b| {
        let pa = &items[a];
        let pb = &items[b];
        (pa.grounded as u8).cmp(&(pb.grounded as u8)).then(
            pb.volume
                .partial_cmp(&pa.volume)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    for i in order {
        let p = &items[i];
        if mesh.vertices.len() + 180 > MAX_VERTS {
            break;
        }
        match p.shape {
            silt::particles::SHAPE_RIBBON => add_ribbon(&mut mesh, p),
            silt::particles::SHAPE_DROP => add_drop(&mut mesh, p),
            silt::particles::SHAPE_DUST => add_dust(&mut mesh, p),
            _ => add_torn_clod(&mut mesh, p),
        }
    }
    mesh
}
