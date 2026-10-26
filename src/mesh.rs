//! Procedural buggy, wheels and a small patch of countryside.
//!
//! Geometry is flat-shaded and carries a panel id. The shader turns that id
//! into paint, rubber or mud. Wheel and shock meshes stay in the spaces the
//! vehicle solver already uses, so the picture matches the contact model.

use glam::Vec3;

use silt::vehicle::{hardpoints, Vehicle, WHEEL_RADIUS, WHEEL_WIDTH};

pub const PANEL_LIGHT: f32 = 16.0;
pub const PANEL_TAIL: f32 = 17.0;
pub const PANEL_SKIRT: f32 = 40.0;
pub const PANEL_HILL: f32 = 41.0;
pub const PANEL_ROCK: f32 = 42.0;
pub const PANEL_REED: f32 = 43.0;
pub const PANEL_TREE: f32 = 44.0;
pub const PANEL_TRUNK: f32 = 45.0;

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
        mesh.indices
            .extend_from_slice(&[base, base + 1, base + 2]);
    }
}

fn add_fender(mesh: &mut CpuMesh, x: f32, z: f32, panel: f32) {
    // Nominal wheel center sits near y = -0.18 once the spring has settled.
    let cy = -0.16;
    let radius = 0.50;
    let segs = 6;
    let a_start = 0.18 * std::f32::consts::PI;
    let a_span = 0.64 * std::f32::consts::PI;
    let half_w = 0.20;
    for i in 0..segs {
        let a0 = a_start + a_span * (i as f32 / segs as f32);
        let a1 = a_start + a_span * ((i + 1) as f32 / segs as f32);
        let p = |a: f32, xo: f32| {
            Vec3::new(
                x + xo,
                cy + a.sin() * radius,
                z + a.cos() * radius,
            )
        };
        add_quad(
            mesh,
            [p(a0, -half_w), p(a0, half_w), p(a1, half_w), p(a1, -half_w)],
            panel,
        );
        // Lip on the outer edge so the fender reads from the side.
        let lip = if x > 0.0 { 0.04 } else { -0.04 };
        add_quad(
            mesh,
            [
                p(a0, half_w * x.signum()),
                p(a0, half_w * x.signum() + lip),
                p(a1, half_w * x.signum() + lip),
                p(a1, half_w * x.signum()),
            ],
            panel,
        );
    }
}

/// Sand-rail buggy in vehicle local space. Y up, Z forward, origin at the CG.
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

    // Frame rails.
    add_box(
        &mut m,
        Vec3::new(-0.30, -0.02, 0.05),
        Vec3::new(0.055, 0.07, 1.38),
        cage,
    );
    add_box(
        &mut m,
        Vec3::new(0.30, -0.02, 0.05),
        Vec3::new(0.055, 0.07, 1.38),
        cage,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, -0.02, 1.28),
        Vec3::new(0.32, 0.05, 0.06),
        cage,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, -0.02, -1.22),
        Vec3::new(0.32, 0.05, 0.06),
        cage,
    );

    // Belly and floor. These pick up the heavy mud.
    add_box(
        &mut m,
        Vec3::new(0.0, -0.14, 0.05),
        Vec3::new(0.36, 0.022, 1.18),
        belly,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, -0.05, -0.05),
        Vec3::new(0.28, 0.02, 0.72),
        belly,
    );

    // Orange side pods.
    add_box(
        &mut m,
        Vec3::new(-0.50, 0.04, 0.20),
        Vec3::new(0.09, 0.15, 0.92),
        left,
    );
    add_box(
        &mut m,
        Vec3::new(0.50, 0.04, 0.20),
        Vec3::new(0.09, 0.15, 0.92),
        right,
    );
    add_box(
        &mut m,
        Vec3::new(-0.46, 0.16, 0.55),
        Vec3::new(0.07, 0.05, 0.42),
        left,
    );
    add_box(
        &mut m,
        Vec3::new(0.46, 0.16, 0.55),
        Vec3::new(0.07, 0.05, 0.42),
        right,
    );

    // Nose, hood, bumper, plate, lamps.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.02, 1.48),
        Vec3::new(0.34, 0.12, 0.22),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.16, 1.28),
        Vec3::new(0.28, 0.045, 0.28),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, -0.08, 1.62),
        Vec3::new(0.42, 0.035, 0.05),
        nose,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.02, 1.70),
        Vec3::new(0.10, 0.045, 0.012),
        cage,
    );
    add_box(
        &mut m,
        Vec3::new(-0.20, 0.05, 1.69),
        Vec3::new(0.07, 0.045, 0.025),
        PANEL_LIGHT,
    );
    add_box(
        &mut m,
        Vec3::new(0.20, 0.05, 1.69),
        Vec3::new(0.07, 0.045, 0.025),
        PANEL_LIGHT,
    );

    // Rear bulkhead and lamps.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.10, -1.32),
        Vec3::new(0.42, 0.18, 0.07),
        rear,
    );
    add_box(
        &mut m,
        Vec3::new(-0.30, 0.14, -1.39),
        Vec3::new(0.07, 0.04, 0.02),
        PANEL_TAIL,
    );
    add_box(
        &mut m,
        Vec3::new(0.30, 0.14, -1.39),
        Vec3::new(0.07, 0.04, 0.02),
        PANEL_TAIL,
    );

    // Seats.
    add_box(
        &mut m,
        Vec3::new(-0.20, 0.12, -0.18),
        Vec3::new(0.14, 0.06, 0.20),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(0.20, 0.12, -0.18),
        Vec3::new(0.14, 0.06, 0.20),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(-0.20, 0.28, -0.34),
        Vec3::new(0.13, 0.16, 0.045),
        seat,
    );
    add_box(
        &mut m,
        Vec3::new(0.20, 0.28, -0.34),
        Vec3::new(0.13, 0.16, 0.045),
        seat,
    );

    // Rear engine and a small intake.
    add_box(
        &mut m,
        Vec3::new(0.0, 0.08, -0.82),
        Vec3::new(0.26, 0.16, 0.28),
        engine,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 0.28, -0.78),
        Vec3::new(0.10, 0.06, 0.10),
        engine,
    );
    add_tube(
        &mut m,
        Vec3::new(0.22, 0.06, -0.95),
        Vec3::new(0.48, 0.02, -1.25),
        0.035,
        engine,
        6,
    );
    add_tube(
        &mut m,
        Vec3::new(-0.22, 0.06, -0.95),
        Vec3::new(-0.48, 0.02, -1.25),
        0.035,
        engine,
        6,
    );

    // Fenders sit above the four wheels.
    let hp = hardpoints();
    add_fender(&mut m, hp[0].x, hp[0].z, 6.0);
    add_fender(&mut m, hp[1].x, hp[1].z, 7.0);
    add_fender(&mut m, hp[2].x, hp[2].z, 8.0);
    add_fender(&mut m, hp[3].x, hp[3].z, 9.0);

    // Cage: rear hoop, front hoop, roof, side bars.
    let rl = Vec3::new(-0.46, -0.02, -0.42);
    let rr = Vec3::new(0.46, -0.02, -0.42);
    let rtl = Vec3::new(-0.46, 1.08, -0.42);
    let rtr = Vec3::new(0.46, 1.08, -0.42);
    add_tube(&mut m, rl, rtl, 0.028, cage, 6);
    add_tube(&mut m, rr, rtr, 0.028, cage, 6);
    add_tube(&mut m, rtl, rtr, 0.028, cage, 6);
    let fl = Vec3::new(-0.40, 0.02, 0.92);
    let fr = Vec3::new(0.40, 0.02, 0.92);
    let ftl = Vec3::new(-0.40, 0.78, 0.92);
    let ftr = Vec3::new(0.40, 0.78, 0.92);
    add_tube(&mut m, fl, ftl, 0.026, cage, 5);
    add_tube(&mut m, fr, ftr, 0.026, cage, 5);
    add_tube(&mut m, ftl, ftr, 0.026, cage, 5);
    add_tube(&mut m, ftl, rtl, 0.024, cage, 5);
    add_tube(&mut m, ftr, rtr, 0.024, cage, 5);
    add_tube(
        &mut m,
        Vec3::new(-0.46, 0.42, -0.42),
        Vec3::new(-0.46, 0.42, 0.70),
        0.022,
        cage,
        5,
    );
    add_tube(
        &mut m,
        Vec3::new(0.46, 0.42, -0.42),
        Vec3::new(0.46, 0.42, 0.70),
        0.022,
        cage,
        5,
    );
    add_box(
        &mut m,
        Vec3::new(0.0, 1.10, -0.05),
        Vec3::new(0.40, 0.015, 0.32),
        nose,
    );

    m
}

/// Shocks and solid axles in vehicle local space. Rebuilt when the springs move.
pub fn suspension_mesh(vehicle: &Vehicle) -> CpuMesh {
    let mut m = CpuMesh::new();
    let hp = hardpoints();
    for i in 0..4 {
        let wc = vehicle.wheel_center_local(i);
        add_tube(&mut m, hp[i], wc, 0.02, 0.0, 6);
        let spring_top = hp[i] + (wc - hp[i]) * 0.15;
        let spring_bot = hp[i] + (wc - hp[i]) * 0.72;
        add_tube(&mut m, spring_top, spring_bot, 0.045, 0.0, 7);
    }
    let fl = vehicle.wheel_center_local(0);
    let fr = vehicle.wheel_center_local(1);
    let rl = vehicle.wheel_center_local(2);
    let rr = vehicle.wheel_center_local(3);
    add_tube(&mut m, fl, fr, 0.035, 0.0, 6);
    add_tube(&mut m, rl, rr, 0.035, 0.0, 6);
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

/// Two short crossed cards. Tall thin boxes read as neon sticks.
fn add_tuft(mesh: &mut CpuMesh, base: Vec3, height: f32, panel: f32) {
    let root = base + Vec3::new(0.0, -0.05, 0.0);
    let w = height * 1.25;
    for k in 0..2 {
        let a = k as f32 * std::f32::consts::FRAC_PI_2 + 0.35;
        let d = Vec3::new(a.cos(), 0.0, a.sin());
        let tip = Vec3::Y * height + d * (height * 0.12);
        add_quad(
            mesh,
            [
                root - d * w,
                root + d * w,
                root + d * (w * 0.22) + tip,
                root - d * (w * 0.22) + tip,
            ],
            panel,
        );
    }
}

/// Horizon, rocks and reeds so the mud has somewhere to sit.
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

    // Rocks off the drive line (the scripted run stays near x = 0).
    let rocks = [
        Vec3::new(6.4, 0.15, -12.5),
        Vec3::new(-7.2, 0.12, -6.0),
        Vec3::new(8.5, 0.18, 3.2),
        Vec3::new(-9.0, 0.1, 7.5),
        Vec3::new(5.2, 0.08, 8.8),
        Vec3::new(-5.6, 0.14, 12.4),
        Vec3::new(11.0, 0.2, -2.0),
    ];
    for (k, p) in rocks.iter().enumerate() {
        let s = 0.28 + hash(k as u32) * 0.35;
        add_box(
            &mut m,
            *p,
            Vec3::new(s, s * 0.55, s * (0.7 + hash(k as u32 + 3) * 0.5)),
            PANEL_ROCK,
        );
    }

    // Short tufts. Tall boxes read as neon sticks.
    let puddle = Vec3::new(0.0, 0.0, 0.4);
    for k in 0..10 {
        let ang = k as f32 / 10.0 * std::f32::consts::TAU + 0.4;
        let rad = 9.2 + hash(k) * 1.6;
        let x = puddle.x + ang.cos() * rad;
        if x.abs() < 3.0 {
            continue;
        }
        let z = puddle.z + ang.sin() * rad;
        let h = 0.11 + hash(k * 3) * 0.08;
        add_tuft(&mut m, Vec3::new(x, 0.0, z), h, PANEL_REED);
    }

    let trees = [
        Vec3::new(-16.0, 0.0, 14.0),
        Vec3::new(18.0, 0.0, -16.0),
        Vec3::new(14.0, 0.0, 16.0),
        Vec3::new(-18.0, 0.0, -11.0),
        Vec3::new(20.0, 0.0, 6.0),
    ];
    for (k, p) in trees.iter().enumerate() {
        let trunk_h = 1.35 + hash(20 + k as u32) * 0.7;
        add_tube(
            &mut m,
            *p,
            *p + Vec3::Y * trunk_h,
            0.09,
            PANEL_TRUNK,
            6,
        );
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
