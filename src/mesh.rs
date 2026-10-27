//! Procedural buggy, wheels and a small patch of countryside.
//!
//! Geometry is flat-shaded and carries a panel id. The shader turns that id
//! into paint, rubber or mud. Wheel and shock meshes stay in the spaces the
//! vehicle solver already uses, so the picture matches the contact model.

use glam::{Quat, Vec3};

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
