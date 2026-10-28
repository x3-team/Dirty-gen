//! Procedural buggy, wheels and a small patch of countryside.
//!
//! Geometry is flat-shaded and carries a panel id. The shader turns that id
//! into paint, rubber or mud. Wheel and shock meshes stay in the spaces the
//! vehicle solver already uses, so the picture matches the contact model.

use glam::{Mat4, Quat, Vec3};

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
/// Continuous sheet thrown off a coated tread. Drawn, not simulated.
pub const PANEL_SHEET: f32 = 47.0;
/// Clay packed into a tread cut. Matte, and not a coat painted on the rubber.
pub const PANEL_PACK: f32 = 48.0;

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

    pub fn append(&mut self, other: &CpuMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|i| i + base));
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

fn value_noise(x: f32, y: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let h = |ix: i32, iy: i32| chunk_hash(ix as u32 ^ 0xA5A5_5A5A, iy as u32 ^ 0x3C3C_C3C3);
    let a = h(x0, y0);
    let b = h(x0 + 1, y0);
    let c = h(x0, y0 + 1);
    let d = h(x0 + 1, y0 + 1);
    a * (1.0 - sx) * (1.0 - sy) + b * sx * (1.0 - sy) + c * (1.0 - sx) * sy + d * sx * sy
}

/// Unscaled tread radius at an axle position. Matches `wheel_mesh`.
fn profile_radius(x: f32) -> f32 {
    const P: [(f32, f32); 11] = [
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
    if x <= P[0].0 {
        return P[0].1;
    }
    for w in P.windows(2) {
        if x <= w[1].0 + 1.0e-5 {
            let t = ((x - w[0].0) / (w[1].0 - w[0].0)).clamp(0.0, 1.0);
            return w[0].1 + (w[1].1 - w[0].1) * t;
        }
    }
    P[10].1
}

fn transform_mesh(mesh: &mut CpuMesh, xform: Mat4) {
    for v in &mut mesh.vertices {
        let p = xform.transform_point3(Vec3::from_array(v.pos));
        let n = xform
            .transform_vector3(Vec3::from_array(v.nrm))
            .normalize_or_zero();
        v.pos = p.to_array();
        v.nrm = n.to_array();
    }
}

/// Grid surface. `hint` picks which side is front-facing so backface culling
/// keeps the outside of a tread cut and the top of a sheet.
fn add_grid<F>(mesh: &mut CpuMesh, nu: usize, nv: usize, panel: f32, hint: Vec3, mut at: F)
where
    F: FnMut(usize, usize) -> Vec3,
{
    let stride = nv + 1;
    let npts = (nu + 1) * stride;
    let mut pts = Vec::with_capacity(npts);
    for iu in 0..=nu {
        for iv in 0..=nv {
            pts.push(at(iu, iv));
        }
    }
    let mut nrm = vec![Vec3::ZERO; npts];
    for iu in 0..=nu {
        for iv in 0..=nv {
            let iu0 = iu.saturating_sub(1);
            let iu1 = (iu + 1).min(nu);
            let iv0 = iv.saturating_sub(1);
            let iv1 = (iv + 1).min(nv);
            let du = pts[iu1 * stride + iv] - pts[iu0 * stride + iv];
            let dv = pts[iu * stride + iv1] - pts[iu * stride + iv0];
            let mut n = du.cross(dv);
            if n.length_squared() < 1.0e-10 {
                n = hint;
            }
            nrm[iu * stride + iv] = n;
        }
    }
    let mid = npts / 2;
    let mid_p = pts[mid];
    // A zero hint means "outward from the axle", which is the same side all
    // the way around a tread cut. A fixed world up would flip the bottom.
    let hint = if hint.length_squared() < 1.0e-8 {
        Vec3::new(0.0, mid_p.y, mid_p.z)
    } else {
        hint
    };
    let mut flip = false;
    if nrm[mid].dot(hint) < 0.0 {
        flip = true;
        for n in &mut nrm {
            *n = -*n;
        }
    }
    let base = mesh.vertices.len() as u32;
    for i in 0..npts {
        push_smooth(mesh, pts[i], nrm[i], panel);
    }
    for iu in 0..nu {
        for iv in 0..nv {
            let a = base + (iu * stride + iv) as u32;
            let b = base + ((iu + 1) * stride + iv) as u32;
            let c = base + ((iu + 1) * stride + iv + 1) as u32;
            let d = base + (iu * stride + iv + 1) as u32;
            if flip {
                mesh.indices.extend_from_slice(&[a, c, b, a, d, c]);
            } else {
                mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
}

/// Same grid, extruded into a thin shell so the rim has thickness.
fn add_shell<F>(mesh: &mut CpuMesh, nu: usize, nv: usize, panel: f32, thick: f32, mut at: F)
where
    F: FnMut(usize, usize) -> Vec3,
{
    let stride = nv + 1;
    let npts = (nu + 1) * stride;
    let mut pts = Vec::with_capacity(npts);
    for iu in 0..=nu {
        for iv in 0..=nv {
            pts.push(at(iu, iv));
        }
    }
    let mut nrm = vec![Vec3::Y; npts];
    for iu in 0..=nu {
        for iv in 0..=nv {
            let iu0 = iu.saturating_sub(1);
            let iu1 = (iu + 1).min(nu);
            let iv0 = iv.saturating_sub(1);
            let iv1 = (iv + 1).min(nv);
            let du = pts[iu1 * stride + iv] - pts[iu0 * stride + iv];
            let dv = pts[iu * stride + iv1] - pts[iu * stride + iv0];
            let n = du.cross(dv);
            if n.length_squared() > 1.0e-10 {
                nrm[iu * stride + iv] = n.normalize();
            }
        }
    }
    let mid = npts / 2;
    let geo_up = nrm[mid].y >= 0.0;
    if !geo_up {
        for n in &mut nrm {
            *n = -*n;
        }
    }
    let half = thick * 0.5;
    let top_p: Vec<Vec3> = pts
        .iter()
        .zip(nrm.iter())
        .map(|(p, n)| *p + *n * half)
        .collect();
    let bot_p: Vec<Vec3> = pts
        .iter()
        .zip(nrm.iter())
        .map(|(p, n)| *p - *n * half)
        .collect();
    let base = mesh.vertices.len() as u32;
    for i in 0..npts {
        push_smooth(mesh, top_p[i], nrm[i], panel);
    }
    for i in 0..npts {
        push_smooth(mesh, bot_p[i], -nrm[i], panel);
    }
    let top = |iu: usize, iv: usize| base + (iu * stride + iv) as u32;
    let bot = |iu: usize, iv: usize| base + (npts + iu * stride + iv) as u32;
    // After the flip, stored normals point up and match triangle a,b,c.
    for iu in 0..nu {
        for iv in 0..nv {
            let a = top(iu, iv);
            let b = top(iu + 1, iv);
            let c = top(iu + 1, iv + 1);
            let d = top(iu, iv + 1);
            let e = bot(iu, iv);
            let f = bot(iu + 1, iv);
            let g = bot(iu + 1, iv + 1);
            let h = bot(iu, iv + 1);
            if geo_up {
                mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
                mesh.indices.extend_from_slice(&[e, g, f, e, h, g]);
            } else {
                mesh.indices.extend_from_slice(&[a, c, b, a, d, c]);
                mesh.indices.extend_from_slice(&[e, f, g, e, g, h]);
            }
        }
    }
    let center = pts[mid];
    let emit_rim = |mesh: &mut CpuMesh, a: u32, b: u32, c: u32, d: u32, outward: Vec3| {
        let pa = Vec3::from_array(mesh.vertices[a as usize].pos);
        let pb = Vec3::from_array(mesh.vertices[b as usize].pos);
        let pc = Vec3::from_array(mesh.vertices[c as usize].pos);
        let n = (pb - pa).cross(pc - pa);
        if n.dot(outward) >= 0.0 {
            mesh.indices.extend_from_slice(&[a, b, c, a, c, d]);
        } else {
            mesh.indices.extend_from_slice(&[a, c, b, a, d, c]);
        }
    };
    for iu in 0..nu {
        let outward = (pts[iu * stride] - center).normalize_or_zero();
        emit_rim(
            mesh,
            top(iu, 0),
            top(iu + 1, 0),
            bot(iu + 1, 0),
            bot(iu, 0),
            outward,
        );
        let outward = (pts[iu * stride + nv] - center).normalize_or_zero();
        emit_rim(
            mesh,
            top(iu, nv),
            bot(iu, nv),
            bot(iu + 1, nv),
            top(iu + 1, nv),
            outward,
        );
    }
    for iv in 0..nv {
        let outward = (pts[iv] - center).normalize_or_zero();
        emit_rim(
            mesh,
            top(0, iv),
            bot(0, iv),
            bot(0, iv + 1),
            top(0, iv + 1),
            outward,
        );
        let outward = (pts[nu * stride + iv] - center).normalize_or_zero();
        emit_rim(
            mesh,
            top(nu, iv),
            top(nu, iv + 1),
            bot(nu, iv + 1),
            bot(nu, iv),
            outward,
        );
    }
}

/// Clay sitting in the three deep cuts. The ribs stay rubber. World space,
/// and only a picture of the volume the tire already holds.
pub fn tread_clay(vehicle: &Vehicle) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    let body = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
    let scale = WHEEL_RADIUS / 0.390;
    // Crown pairs. The low ring between them is the cut.
    const CUTS: [(f32, f32, f32, f32); 3] = [
        (-0.085, 0.385, -0.025, 0.388),
        (-0.025, 0.388, 0.035, 0.380),
        (0.035, 0.380, 0.095, 0.385),
    ];
    for i in 0..4 {
        let wheel = vehicle.wheels[i];
        let coat = wheel.clay_coat();
        if wheel.clay_thickness() < 1.0e-4 || coat < 0.02 {
            continue;
        }
        let xform = body
            * Mat4::from_translation(vehicle.wheel_center_local(i))
            * Mat4::from_rotation_y(wheel.steer)
            * Mat4::from_rotation_x(wheel.spin);
        let mut local = CpuMesh::new();
        for (cut, (x0, r0, x1, r1)) in CUTS.iter().copied().enumerate() {
            add_groove(&mut local, x0, r0, x1, r1, scale, coat, i as u32 * 4 + cut as u32);
        }
        transform_mesh(&mut local, xform);
        mesh.append(&local);
    }
    mesh
}

fn add_groove(
    mesh: &mut CpuMesh,
    x0: f32,
    r0: f32,
    x1: f32,
    r1: f32,
    scale: f32,
    coat: f32,
    seed: u32,
) {
    let nu = 40;
    let nv = 6;
    // A narrow rubber rib stays proud. The rest of the cut is the packed layer.
    let inset = 0.05;
    let fill = (0.62 + coat * 0.40).clamp(0.2, 1.0);
    add_grid(mesh, nu, nv, PANEL_PACK, Vec3::ZERO, |iu, iv| {
        let ang = iu as f32 / nu as f32 * std::f32::consts::TAU;
        let s = inset + (1.0 - 2.0 * inset) * (iv as f32 / nv as f32);
        let x = x0 + (x1 - x0) * s;
        let lip = r0 + (r1 - r0) * s;
        let rubber = profile_radius(x);
        let edge = ((s - inset) / (1.0 - 2.0 * inset)).clamp(0.0, 1.0);
        let arch = (edge * std::f32::consts::PI).sin();
        let lump = 0.62 + 0.38 * value_noise(ang * 2.2 + seed as f32 * 0.17, x * 36.0);
        let grain = 0.86 + 0.14 * value_noise(ang * 9.0, x * 70.0 + 2.0);
        let depth = (lip - rubber).max(0.0);
        let mut r = rubber + 0.005 + depth * fill * arch * lump * grain;
        r = r.min(lip - 0.003).max(rubber + 0.004);
        let rr = r * scale;
        Vec3::new(x, ang.sin() * rr, ang.cos() * rr)
    });
}

/// One wet sheet leaving each coated, moving tire. Not a particle, so it
/// does not land or change the heightfield.
pub fn flung_sheet(vehicle: &Vehicle) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    if vehicle.speed() < 0.6 {
        return mesh;
    }
    let fwd = vehicle.forward();
    if fwd.length_squared() < 1.0e-6 {
        return mesh;
    }
    let ori = vehicle.orientation();
    for i in 0..4 {
        let wheel = vehicle.wheels[i];
        let coat = wheel.clay_coat();
        if wheel.clay_thickness() < 1.0e-4 || coat < 0.05 {
            continue;
        }
        let hub = vehicle.pos + ori * vehicle.wheel_center_local(i);
        let sign = hardpoints()[i].x.signum();
        let axle = (ori * Quat::from_rotation_y(wheel.steer) * Vec3::X).normalize_or_zero();
        if axle.length_squared() < 1.0e-6 {
            continue;
        }
        add_fling(&mut mesh, hub, fwd, axle * sign, wheel.spin, coat, i as u32);
    }
    mesh
}

fn add_fling(mesh: &mut CpuMesh, hub: Vec3, fwd: Vec3, out: Vec3, spin: f32, coat: f32, seed: u32) {
    // Leave the crown and run back along the rut. That is the open air beside
    // the tire. A forward throw runs into the camera and reads as a loose ribbon.
    let peel = (-fwd * 0.42 + Vec3::Y * 0.86 + out * 0.08).normalize();
    // On the rubber, so the root is the tread and not a gap above it.
    let origin = hub + peel * (WHEEL_RADIUS * 1.01);
    let throw = (-fwd * 0.86 - out * 0.16).normalize_or_zero();
    let side = throw.cross(Vec3::Y).normalize_or_zero();
    if throw.length_squared() < 1.0e-6 || side.length_squared() < 1.0e-6 {
        return;
    }
    let nu = 22;
    let nv = 14;
    let length = 0.72 * (0.82 + 0.18 * coat);
    let phase = spin * 0.35;
    // A thin shell, so the lip has thickness. The tip wave stays shallow;
    // deep cuts in that edge were reading as separate feathers.
    add_shell(mesh, nu, nv, PANEL_SHEET, 0.012, |iu, iv| {
        let u = iu as f32 / nu as f32;
        let v = iv as f32 / nv as f32;
        let reach = 0.90 + 0.10 * value_noise(v * 2.1 + seed as f32 * 0.13, 1.7);
        let uu = u * reach;
        let vc = v - 0.5;
        let hug = (1.0 - uu * 2.4).max(0.0);
        let side_tear = 1.0 - vc.abs().powf(1.5) * uu * 0.22;
        let arc = (uu * std::f32::consts::PI * 0.78).sin();
        let curl = (v * std::f32::consts::PI).sin();
        let span = (0.18 + uu * 0.24) * (0.84 + 0.16 * coat) * side_tear.max(0.45);
        // Edges drop into a trough, so the sheet is not a flat card.
        let gutter = (curl - 1.0) * (0.028 + uu * 0.07);
        let ripple = (uu * 6.5 + v * 2.4 + phase).sin() * 0.007 * uu;
        let wander = (value_noise(uu * 2.0 + seed as f32, v * 1.6) - 0.5) * 0.018 * uu;
        // One edge climbs as the sheet leaves, so the face is a curl and not a card.
        let twist = vc * uu * 0.07;
        origin
            + throw * (uu * length * (1.0 - hug * 0.35))
            + side * (vc * span + wander)
            + Vec3::Y * (arc * 0.13 * (1.0 - hug * 0.7) + gutter + ripple + twist)
    });
}
