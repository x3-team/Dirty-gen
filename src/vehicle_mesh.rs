//! Procedural open-wheel buggy and motocross bike.
//!
//! Local space matches the solver: Y up, Z forward, origin at the center of
//! gravity. Wheel radius and width are the contact sizes, so the tire sits on
//! the same circle the terramechanics model uses. Panel ids 0–11 are the
//! adhesion slots; mud in the shader still keys off those ids.
//!
//! Geometry is built here, not loaded. Nothing in this file is a scanned or
//! downloaded model.

use glam::Vec3;

use silt::bike::{Bike, BIKE_RADIUS, BIKE_WIDTH};
use silt::vehicle::{self, Vehicle, WHEEL_RADIUS, WHEEL_WIDTH};

use crate::mesh::{CpuMesh, MeshVertex};

pub const P_CAGE: f32 = 0.0;
pub const P_NOSE: f32 = 1.0;
pub const P_LEFT: f32 = 2.0;
pub const P_RIGHT: f32 = 3.0;
pub const P_REAR: f32 = 4.0;
pub const P_BELLY: f32 = 5.0;
pub const P_FENDER_FL: f32 = 6.0;
pub const P_FENDER_FR: f32 = 7.0;
pub const P_FENDER_RL: f32 = 8.0;
pub const P_FENDER_RR: f32 = 9.0;
pub const P_ENGINE: f32 = 10.0;
pub const P_SEAT: f32 = 11.0;
pub const P_LAMP: f32 = 16.0;
pub const P_TAIL: f32 = 17.0;
/// Wet aluminum: rims, forks, uprights, exhaust.
pub const P_ALLOY: f32 = 18.0;
pub const P_RUBBER: f32 = 20.0;
pub const P_JERSEY: f32 = 21.0;
pub const P_HELMET: f32 = 22.0;
pub const P_VISOR: f32 = 23.0;
pub const P_BOOT: f32 = 24.0;
pub const P_PLATE: f32 = 25.0;
/// Orange lower chassis, A-arms, and coil springs. The silver cage stays on panel 0.
pub const P_ORANGE: f32 = 19.0;
/// Damper rod inside the coil. Reads as bare metal next to the orange spring.
pub const P_CHROME: f32 = 26.0;

/// Dynamic suspension buffers in `render`. The per-frame upload must fit.
pub const BUGGY_SUSP_VERTS: usize = 16384;
pub const BUGGY_SUSP_INDICES: usize = 32768;
pub const BIKE_SUSP_VERTS: usize = 8192;
pub const BIKE_SUSP_INDICES: usize = 16384;

const BUGGY_PROFILE: &[(f32, f32)] = &[
    (-1.00, 0.62),
    (-0.86, 0.84),
    (-0.70, 0.95),
    (-0.54, 0.99),
    (-0.40, 0.90),
    (-0.26, 1.00),
    (-0.12, 0.91),
    (0.00, 1.00),
    (0.12, 0.91),
    (0.26, 1.00),
    (0.40, 0.90),
    (0.54, 0.99),
    (0.70, 0.95),
    (0.86, 0.84),
    (1.00, 0.62),
];

const BIKE_PROFILE: &[(f32, f32)] = &[
    (-1.00, 0.55),
    (-0.70, 0.88),
    (-0.46, 1.00),
    (-0.20, 1.00),
    (0.00, 1.00),
    (0.20, 1.00),
    (0.46, 1.00),
    (0.70, 0.88),
    (1.00, 0.55),
];

struct B {
    mesh: CpuMesh,
}

impl B {
    fn new() -> Self {
        Self {
            mesh: CpuMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
            },
        }
    }

    fn emit(
        &mut self,
        a: Vec3,
        b: Vec3,
        c: Vec3,
        n: Vec3,
        panel: f32,
        uva: [f32; 2],
        uvb: [f32; 2],
        uvc: [f32; 2],
    ) {
        let n = if n.length_squared() < 1.0e-12 {
            Vec3::Y
        } else {
            n.normalize()
        };
        let base = self.mesh.vertices.len() as u32;
        for (p, uv) in [(a, uva), (b, uvb), (c, uvc)] {
            self.mesh.vertices.push(MeshVertex {
                pos: p.to_array(),
                nrm: n.to_array(),
                uv,
                panel,
            });
        }
        self.mesh
            .indices
            .extend_from_slice(&[base, base + 1, base + 2]);
    }

    fn tri(&mut self, a: Vec3, b: Vec3, c: Vec3, panel: f32) {
        let n = (b - a).cross(c - a);
        if n.length_squared() < 1.0e-12 {
            return;
        }
        self.emit(a, b, c, n, panel, uv_body(a), uv_body(b), uv_body(c));
    }

    fn quad(&mut self, p: [Vec3; 4], panel: f32) {
        self.tri(p[0], p[1], p[2], panel);
        self.tri(p[0], p[2], p[3], panel);
    }

    fn tri_uv(
        &mut self,
        a: Vec3,
        b: Vec3,
        c: Vec3,
        panel: f32,
        uva: [f32; 2],
        uvb: [f32; 2],
        uvc: [f32; 2],
    ) {
        let n = (b - a).cross(c - a);
        if n.length_squared() < 1.0e-12 {
            return;
        }
        self.emit(a, b, c, n, panel, uva, uvb, uvc);
    }

    fn quad_uv(&mut self, p: [Vec3; 4], panel: f32, uv: [[f32; 2]; 4]) {
        self.tri_uv(p[0], p[1], p[2], panel, uv[0], uv[1], uv[2]);
        self.tri_uv(p[0], p[2], p[3], panel, uv[0], uv[2], uv[3]);
    }

    fn box_at(&mut self, c: Vec3, h: Vec3, panel: f32) {
        if h.x <= 1.0e-4 || h.y <= 1.0e-4 || h.z <= 1.0e-4 {
            return;
        }
        let (x, y, z) = (Vec3::X * h.x, Vec3::Y * h.y, Vec3::Z * h.z);
        self.quad(
            [c - x + y - z, c - x + y + z, c + x + y + z, c + x + y - z],
            panel,
        );
        self.quad(
            [c - x - y - z, c + x - y - z, c + x - y + z, c - x - y + z],
            panel,
        );
        self.quad(
            [c + x - y - z, c + x + y - z, c + x + y + z, c + x - y + z],
            panel,
        );
        self.quad(
            [c - x - y - z, c - x - y + z, c - x + y + z, c - x + y - z],
            panel,
        );
        self.quad(
            [c - x - y + z, c + x - y + z, c + x + y + z, c - x + y + z],
            panel,
        );
        self.quad(
            [c + x - y - z, c - x - y - z, c - x + y - z, c + x + y - z],
            panel,
        );
    }

    fn tube(&mut self, a: Vec3, b: Vec3, radius: f32, panel: f32, segs: usize, caps: bool) {
        let axis = b - a;
        let len = axis.length();
        if len < 1.0e-4 || radius < 1.0e-4 || segs < 3 {
            return;
        }
        let dir = axis / len;
        let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
        let u = dir.cross(helper).normalize();
        let v = dir.cross(u);
        let ring = |t: f32| u * t.cos() + v * t.sin();
        for i in 0..segs {
            let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
            let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
            let n0 = ring(t0);
            let n1 = ring(t1);
            self.quad(
                [
                    a + n0 * radius,
                    a + n1 * radius,
                    b + n1 * radius,
                    b + n0 * radius,
                ],
                panel,
            );
        }
        if caps {
            for i in 0..segs {
                let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
                let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
                self.tri(b, b + ring(t0) * radius, b + ring(t1) * radius, panel);
                self.tri(a, a + ring(t1) * radius, a + ring(t0) * radius, panel);
            }
        }
    }

    fn helix(
        &mut self,
        a: Vec3,
        b: Vec3,
        coil_r: f32,
        tube_r: f32,
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
        let v = dir.cross(u);
        let n = coils * steps;
        let mut prev = a + u * coil_r;
        for i in 1..=n {
            let t = i as f32 / n as f32;
            let ang = t * coils as f32 * std::f32::consts::TAU;
            let center = a + axis * t;
            let p = center + (u * ang.cos() + v * ang.sin()) * coil_r;
            self.tube(prev, p, tube_r, panel, 5, false);
            prev = p;
        }
    }

    fn torus(&mut self, c: Vec3, axis: Vec3, ring_r: f32, tube_r: f32, panel: f32) {
        let dir = axis.normalize_or_zero();
        if dir.length_squared() < 0.5 {
            return;
        }
        let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
        let u = dir.cross(helper).normalize();
        let v = dir.cross(u);
        let n = 12;
        for i in 0..n {
            let a0 = i as f32 / n as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / n as f32 * std::f32::consts::TAU;
            let p = |a: f32| c + (u * a.cos() + v * a.sin()) * ring_r;
            self.tube(p(a0), p(a1), tube_r, panel, 5, false);
        }
    }

    fn sphere(&mut self, c: Vec3, r: f32, panel: f32, stacks: usize, slices: usize) {
        if r <= 1.0e-4 || stacks < 2 || slices < 3 {
            return;
        }
        for i in 0..stacks {
            let v0 = i as f32 / stacks as f32 * std::f32::consts::PI;
            let v1 = (i + 1) as f32 / stacks as f32 * std::f32::consts::PI;
            for j in 0..slices {
                let u0 = j as f32 / slices as f32 * std::f32::consts::TAU;
                let u1 = (j + 1) as f32 / slices as f32 * std::f32::consts::TAU;
                let p = |v: f32, u: f32| {
                    c + Vec3::new(u.sin() * v.sin(), v.cos(), u.cos() * v.sin()) * r
                };
                let p00 = p(v0, u0);
                let p01 = p(v0, u1);
                let p11 = p(v1, u1);
                let p10 = p(v1, u0);
                // Pole triangles collapse; `tri` drops a zero area.
                self.quad([p00, p10, p11, p01], panel);
            }
        }
    }

    fn finish(self, verts: usize, indices: usize, label: &str) -> CpuMesh {
        assert!(
            self.mesh.vertices.len() <= verts && self.mesh.indices.len() <= indices,
            "{label} is {} verts / {} indices, budget {verts} / {indices}",
            self.mesh.vertices.len(),
            self.mesh.indices.len()
        );
        self.mesh
    }
}

fn uv_body(p: Vec3) -> [f32; 2] {
    [p.x * 1.6 + p.z * 0.2, p.y * 1.35 + 0.55]
}

fn fender_panel(index: usize) -> f32 {
    match index {
        0 => P_FENDER_FL,
        1 => P_FENDER_FR,
        2 => P_FENDER_RL,
        _ => P_FENDER_RR,
    }
}

/// Grooved buggy tire plus a dished alloy wheel. Axle is +X.
/// Outer radius is [`WHEEL_RADIUS`], width is [`WHEEL_WIDTH`].
pub fn buggy_wheel() -> CpuMesh {
    let mut b = B::new();
    add_wheel(
        &mut b,
        WHEEL_WIDTH * 0.5,
        WHEEL_RADIUS,
        32,
        8.0,
        0.042,
        0.0,
        BUGGY_PROFILE,
    );
    b.finish(48_000, 120_000, "buggy wheel")
}

/// Knobby motocross tire. Radius [`BIKE_RADIUS`], width [`BIKE_WIDTH`].
/// The hub sticks out past the rubber so the fork and swingarm can meet it
/// without crossing the carcass.
pub fn bike_wheel() -> CpuMesh {
    let mut b = B::new();
    add_wheel(
        &mut b,
        BIKE_WIDTH * 0.5,
        BIKE_RADIUS,
        30,
        10.0,
        0.058,
        0.055,
        BIKE_PROFILE,
    );
    b.finish(48_000, 120_000, "bike wheel")
}

fn add_wheel(
    b: &mut B,
    half_w: f32,
    radius: f32,
    segs: usize,
    knobs: f32,
    knob_depth: f32,
    hub_extra: f32,
    profile: &[(f32, f32)],
) {
    let mut rings: Vec<Vec<Vec3>> = Vec::with_capacity(profile.len());
    for &(xf, rf) in profile {
        let x = xf * half_w;
        let mut ring = Vec::with_capacity(segs);
        for s in 0..segs {
            let t = s as f32 / segs as f32 * std::f32::consts::TAU;
            let mut r = rf * radius;
            if knobs > 0.0 && xf.abs() < 0.58 {
                let crown = (1.0 - xf.abs() / 0.58).clamp(0.0, 1.0);
                let stagger = if xf > 0.02 { 0.5 } else { 0.0 };
                let f = ((t / std::f32::consts::TAU) * knobs + stagger).fract();
                if f > 0.48 {
                    r -= knob_depth * crown;
                }
            }
            ring.push(Vec3::new(x, t.sin() * r, t.cos() * r));
        }
        rings.push(ring);
    }
    for i in 0..profile.len() - 1 {
        let side = profile[i].0.abs() > 0.72 && profile[i + 1].0.abs() > 0.72;
        for s in 0..segs {
            let s1 = (s + 1) % segs;
            let t0 = s as f32 / segs as f32;
            let t1 = s1 as f32 / segs as f32;
            let t1 = if s1 == 0 { 1.0 } else { t1 };
            let y = if side { 2.0 } else { 0.35 };
            let p00 = rings[i][s];
            let p10 = rings[i + 1][s];
            let p11 = rings[i + 1][s1];
            let p01 = rings[i][s1];
            // Outward for a surface of revolution. See the winding note in
            // `tire_normals_point_out`.
            b.quad_uv(
                [p00, p10, p11, p01],
                P_RUBBER,
                [
                    [t0, y],
                    [t0, if side { 2.0 } else { 0.65 }],
                    [t1, if side { 2.0 } else { 0.65 }],
                    [t1, y],
                ],
            );
        }
    }

    let bead = radius * profile[0].1;
    let face_x = half_w * 0.55;
    let hub_r = radius * 0.20;
    let disc_r = bead * 0.98;
    for sign in [-1.0_f32, 1.0] {
        annulus(
            b,
            sign * face_x,
            hub_r * 0.85,
            disc_r,
            sign,
            segs.min(24),
            P_ALLOY,
        );
        b.tube(
            Vec3::new(sign * face_x, 0.0, 0.0),
            Vec3::new(sign * (face_x + half_w * 0.06), 0.0, 0.0),
            disc_r,
            P_ALLOY,
            16,
            false,
        );
    }
    let hub_half = face_x.max(half_w * 0.3) + hub_extra;
    b.tube(
        Vec3::new(-hub_half, 0.0, 0.0),
        Vec3::new(hub_half, 0.0, 0.0),
        hub_r,
        P_ALLOY,
        10,
        true,
    );
    let spokes = if hub_extra < 0.001 { 8 } else { 5 };
    for sign in [-1.0_f32, 1.0] {
        for k in 0..spokes {
            let ang = k as f32 / spokes as f32 * std::f32::consts::TAU + sign * 0.12;
            spoke(
                b,
                sign * face_x,
                hub_r * 1.15,
                disc_r * 0.92,
                ang,
                0.028,
                P_ALLOY,
            );
        }
    }
    // Beadlock ring on the buggy wheel. The bike hub already sticks out past the rubber.
    if hub_extra < 0.001 {
        for sign in [-1.0_f32, 1.0] {
            b.torus(
                Vec3::new(sign * face_x, 0.0, 0.0),
                Vec3::X,
                disc_r * 0.97,
                0.014,
                P_ALLOY,
            );
        }
    }
}

fn annulus(b: &mut B, x: f32, r0: f32, r1: f32, sign: f32, segs: usize, panel: f32) {
    let n = Vec3::new(sign, 0.0, 0.0);
    for i in 0..segs {
        let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let p = |r: f32, t: f32| Vec3::new(x, t.sin() * r, t.cos() * r);
        let quad = if sign > 0.0 {
            [p(r0, t0), p(r1, t0), p(r1, t1), p(r0, t1)]
        } else {
            [p(r0, t0), p(r0, t1), p(r1, t1), p(r1, t0)]
        };
        // Force the geometric winding, then keep it. `quad` stores that normal.
        let _ = n;
        b.quad(quad, panel);
    }
}

fn spoke(b: &mut B, x: f32, r0: f32, r1: f32, ang: f32, half_w: f32, panel: f32) {
    let dir = Vec3::new(0.0, ang.sin(), ang.cos());
    let side = Vec3::new(0.0, ang.cos(), -ang.sin());
    let o = Vec3::new(x, 0.0, 0.0);
    let a0 = dir * r0 + side * half_w;
    let a1 = dir * r0 - side * half_w;
    let b0 = dir * r1 + side * half_w * 0.62;
    let b1 = dir * r1 - side * half_w * 0.62;
    let sign = x.signum();
    if sign >= 0.0 {
        b.quad([o + a0, o + b0, o + b1, o + a1], panel);
    } else {
        b.quad([o + a0, o + a1, o + b1, o + b0], panel);
    }
}

/// Open-wheel buggy. The tub stays inside x = ±0.42 so the tires at ±0.74
/// and the A-arms between them read as separate pieces.
pub fn buggy_body() -> CpuMesh {
    let mut b = B::new();
    // Lower chassis is the orange tube frame from the stills. The silver
    // roll cage sits on top of it and stays inside the open wheels.
    let rail_x = 0.30;
    let rail_y = 0.02;
    b.tube(
        Vec3::new(-rail_x, rail_y, -1.36),
        Vec3::new(-rail_x, rail_y, 1.42),
        0.052,
        P_ORANGE,
        7,
        true,
    );
    b.tube(
        Vec3::new(rail_x, rail_y, -1.36),
        Vec3::new(rail_x, rail_y, 1.42),
        0.052,
        P_ORANGE,
        7,
        true,
    );
    b.tube(
        Vec3::new(-0.26, 0.22, -1.10),
        Vec3::new(-0.26, 0.22, 1.18),
        0.040,
        P_ORANGE,
        6,
        true,
    );
    b.tube(
        Vec3::new(0.26, 0.22, -1.10),
        Vec3::new(0.26, 0.22, 1.18),
        0.040,
        P_ORANGE,
        6,
        true,
    );
    for z in [-1.22_f32, -0.55, 0.15, 0.85, 1.28] {
        b.tube(
            Vec3::new(-rail_x, rail_y, z),
            Vec3::new(rail_x, rail_y, z),
            0.040,
            P_ORANGE,
            6,
            true,
        );
    }

    b.box_at(
        Vec3::new(0.0, -0.145, 0.02),
        Vec3::new(0.30, 0.018, 1.02),
        P_BELLY,
    );
    b.box_at(
        Vec3::new(0.0, 0.09, -0.05),
        Vec3::new(0.24, 0.016, 0.42),
        P_BELLY,
    );

    // Stepped nose, ahead of the front axle and narrower than the track.
    b.box_at(
        Vec3::new(0.0, 0.08, 1.28),
        Vec3::new(0.24, 0.07, 0.12),
        P_NOSE,
    );
    b.box_at(
        Vec3::new(0.0, 0.07, 1.48),
        Vec3::new(0.16, 0.055, 0.10),
        P_NOSE,
    );
    b.box_at(
        Vec3::new(0.0, 0.055, 1.64),
        Vec3::new(0.09, 0.04, 0.08),
        P_NOSE,
    );
    b.box_at(
        Vec3::new(0.0, 0.09, 1.66),
        Vec3::new(0.045, 0.012, 0.05),
        P_PLATE,
    );
    b.tube(
        Vec3::new(-0.28, 0.04, 1.48),
        Vec3::new(0.28, 0.04, 1.48),
        0.036,
        P_ORANGE,
        6,
        true,
    );
    b.tube(
        Vec3::new(-0.24, 0.16, 1.36),
        Vec3::new(-0.10, 0.10, 1.62),
        0.030,
        P_ORANGE,
        5,
        true,
    );
    b.tube(
        Vec3::new(0.24, 0.16, 1.36),
        Vec3::new(0.10, 0.10, 1.62),
        0.030,
        P_ORANGE,
        5,
        true,
    );
    b.tube(
        Vec3::new(-0.13, 0.10, 1.58),
        Vec3::new(-0.13, 0.10, 1.70),
        0.055,
        P_LAMP,
        8,
        true,
    );
    b.tube(
        Vec3::new(0.13, 0.10, 1.58),
        Vec3::new(0.13, 0.10, 1.70),
        0.055,
        P_LAMP,
        8,
        true,
    );

    b.box_at(
        Vec3::new(0.0, 0.16, -1.28),
        Vec3::new(0.26, 0.12, 0.05),
        P_REAR,
    );
    b.tube(
        Vec3::new(-0.16, 0.18, -1.30),
        Vec3::new(-0.16, 0.18, -1.38),
        0.045,
        P_TAIL,
        7,
        true,
    );
    b.tube(
        Vec3::new(0.16, 0.18, -1.30),
        Vec3::new(0.16, 0.18, -1.38),
        0.045,
        P_TAIL,
        7,
        true,
    );
    b.box_at(
        Vec3::new(0.0, 1.02, -0.92),
        Vec3::new(0.52, 0.016, 0.11),
        P_REAR,
    );
    b.tube(
        Vec3::new(-0.28, 1.00, -0.48),
        Vec3::new(-0.46, 1.02, -0.92),
        0.028,
        P_CAGE,
        5,
        true,
    );
    b.tube(
        Vec3::new(0.28, 1.00, -0.48),
        Vec3::new(0.46, 1.02, -0.92),
        0.028,
        P_CAGE,
        5,
        true,
    );

    b.box_at(
        Vec3::new(-0.36, 0.16, -0.05),
        Vec3::new(0.045, 0.10, 0.55),
        P_LEFT,
    );
    b.box_at(
        Vec3::new(0.36, 0.16, -0.05),
        Vec3::new(0.045, 0.10, 0.55),
        P_RIGHT,
    );

    for x in [-0.14_f32, 0.14] {
        b.box_at(
            Vec3::new(x, 0.20, -0.06),
            Vec3::new(0.12, 0.055, 0.18),
            P_SEAT,
        );
        b.box_at(
            Vec3::new(x, 0.42, -0.22),
            Vec3::new(0.115, 0.22, 0.045),
            P_SEAT,
        );
        b.box_at(
            Vec3::new(x, 0.62, -0.24),
            Vec3::new(0.08, 0.05, 0.04),
            P_SEAT,
        );
    }
    b.torus(Vec3::new(-0.08, 0.40, 0.48), Vec3::Z, 0.13, 0.02, P_SEAT);
    b.tube(
        Vec3::new(-0.08, 0.28, 0.36),
        Vec3::new(-0.08, 0.40, 0.48),
        0.018,
        P_CAGE,
        5,
        true,
    );
    b.box_at(
        Vec3::new(0.0, 0.30, 0.58),
        Vec3::new(0.20, 0.045, 0.03),
        P_CAGE,
    );

    b.box_at(
        Vec3::new(0.0, 0.20, -0.78),
        Vec3::new(0.16, 0.13, 0.20),
        P_ENGINE,
    );
    b.box_at(
        Vec3::new(0.0, 0.36, -0.76),
        Vec3::new(0.12, 0.045, 0.14),
        P_ENGINE,
    );
    for i in 0..4 {
        let z = -0.64 - i as f32 * 0.07;
        b.box_at(
            Vec3::new(0.0, 0.22, z),
            Vec3::new(0.20, 0.09, 0.012),
            P_ALLOY,
        );
    }
    b.tube(
        Vec3::new(0.16, 0.12, -0.62),
        Vec3::new(0.28, 0.16, -1.15),
        0.03,
        P_ALLOY,
        6,
        true,
    );
    b.tube(
        Vec3::new(-0.16, 0.12, -0.62),
        Vec3::new(-0.28, 0.16, -1.15),
        0.03,
        P_ALLOY,
        6,
        true,
    );

    // Silver space-frame. Roof is a level rectangle, tall enough to clear the seats.
    let r = 0.048;
    let rl = Vec3::new(-0.32, 0.06, -0.55);
    let rr = Vec3::new(0.32, 0.06, -0.55);
    let rtl = Vec3::new(-0.30, 1.02, -0.48);
    let rtr = Vec3::new(0.30, 1.02, -0.48);
    b.tube(rl, rtl, r, P_CAGE, 7, true);
    b.tube(rr, rtr, r, P_CAGE, 7, true);
    b.tube(rtl, rtr, r, P_CAGE, 7, true);
    b.tube(rl, rr, 0.040, P_CAGE, 6, true);
    let fl = Vec3::new(-0.28, 0.08, 0.72);
    let fr = Vec3::new(0.28, 0.08, 0.72);
    let ftl = Vec3::new(-0.28, 0.96, 0.22);
    let ftr = Vec3::new(0.28, 0.96, 0.22);
    b.tube(fl, ftl, 0.046, P_CAGE, 6, true);
    b.tube(fr, ftr, 0.046, P_CAGE, 6, true);
    b.tube(ftl, ftr, 0.044, P_CAGE, 6, true);
    b.tube(ftl, rtl, 0.042, P_CAGE, 6, true);
    b.tube(ftr, rtr, 0.042, P_CAGE, 6, true);
    b.tube(
        Vec3::new(-0.29, 0.48, 0.48),
        Vec3::new(-0.31, 0.48, -0.50),
        0.034,
        P_CAGE,
        6,
        true,
    );
    b.tube(
        Vec3::new(0.29, 0.48, 0.48),
        Vec3::new(0.31, 0.48, -0.50),
        0.034,
        P_CAGE,
        6,
        true,
    );
    b.tube(
        Vec3::new(0.0, 0.98, 0.22),
        Vec3::new(0.0, 0.98, -0.48),
        0.032,
        P_CAGE,
        5,
        true,
    );
    b.tube(Vec3::new(-0.28, 0.08, -1.05), rl, 0.038, P_ORANGE, 6, true);
    b.tube(Vec3::new(0.28, 0.08, -1.05), rr, 0.038, P_ORANGE, 6, true);

    for hp in vehicle::hardpoints() {
        let sign = hp.x.signum();
        b.tube(
            Vec3::new(sign * 0.30, 0.02, hp.z),
            Vec3::new(sign * 0.30, 0.58, hp.z),
            0.046,
            P_ORANGE,
            6,
            true,
        );
        let panel = fender_panel(if hp.z > 0.0 {
            if sign < 0.0 {
                0
            } else {
                1
            }
        } else if sign < 0.0 {
            2
        } else {
            3
        });
        // Short lip inboard of the tire. The tread stays open, matching the stills.
        b.box_at(
            Vec3::new(sign * 0.40, 0.58, hp.z),
            Vec3::new(0.035, 0.016, 0.18),
            panel,
        );
        let flap_z = if hp.z > 0.0 { hp.z - 0.50 } else { hp.z + 0.50 };
        b.box_at(
            Vec3::new(sign * 0.38, 0.16, flap_z),
            Vec3::new(0.02, 0.12, 0.07),
            panel,
        );
    }
    b.tube(
        Vec3::new(-0.30, 0.56, 1.08),
        Vec3::new(0.30, 0.56, 1.08),
        0.036,
        P_CAGE,
        6,
        true,
    );
    b.tube(
        Vec3::new(-0.30, 0.56, -1.18),
        Vec3::new(0.30, 0.56, -1.18),
        0.036,
        P_CAGE,
        6,
        true,
    );

    b.tube(
        Vec3::new(-0.44, 0.04, -0.70),
        Vec3::new(-0.44, 0.04, 0.62),
        0.034,
        P_ORANGE,
        6,
        true,
    );
    b.tube(
        Vec3::new(0.44, 0.04, -0.70),
        Vec3::new(0.44, 0.04, 0.62),
        0.034,
        P_ORANGE,
        6,
        true,
    );

    b.finish(80_000, 200_000, "buggy body")
}

/// Double wishbones, a coilover, and a stub axle per wheel. Ends follow
/// `wheel_center_local`, so the spring shortens on bump and hangs at droop.
pub fn buggy_suspension(vehicle: &Vehicle) -> CpuMesh {
    let mut b = B::new();
    let hps = vehicle::hardpoints();
    for i in 0..4 {
        let wc = vehicle.wheel_center_local(i);
        let sign = hps[i].x.signum();
        let z = hps[i].z;
        // Knuckle stays inboard of the inner sidewall (|x| = 0.60) so the
        // coil, whose radius is 8 cm, does not enter the rubber.
        let knuckle = Vec3::new(wc.x - sign * 0.26, wc.y, wc.z);
        let upright_top = knuckle + Vec3::Y * 0.26;
        // Wide A-arms. The triangle reads from the front and from below.
        let lower_f = Vec3::new(sign * 0.12, -0.06, z + 0.24);
        let lower_r = Vec3::new(sign * 0.12, -0.06, z - 0.24);
        let upper_f = Vec3::new(sign * 0.16, 0.42, z + 0.14);
        let upper_r = Vec3::new(sign * 0.16, 0.42, z - 0.14);
        b.tube(lower_f, knuckle, 0.040, P_ORANGE, 6, true);
        b.tube(lower_r, knuckle, 0.040, P_ORANGE, 6, true);
        b.tube(lower_f, lower_r, 0.032, P_ORANGE, 5, true);
        b.tube(upper_f, upright_top, 0.030, P_ORANGE, 6, true);
        b.tube(upper_r, upright_top, 0.030, P_ORANGE, 6, true);
        b.tube(upper_f, upper_r, 0.024, P_ORANGE, 5, true);
        b.tube(knuckle, upright_top, 0.038, P_ALLOY, 6, true);
        let hub_face = Vec3::new(wc.x - sign * (WHEEL_WIDTH * 0.5 - 0.01), wc.y, wc.z);
        b.tube(knuckle, hub_face, 0.034, P_ALLOY, 6, true);
        b.tube(
            Vec3::new(sign * 0.06, -0.04, z),
            knuckle,
            0.026,
            P_ALLOY,
            5,
            true,
        );
        if i < 2 {
            let rack = Vec3::new(sign * 0.10, 0.06, z);
            b.tube(rack, knuckle + Vec3::Y * 0.06, 0.018, P_CHROME, 5, true);
        }
        // Long-travel coilover, inboard of the sidewall. Orange spring, chrome rod.
        let tower = Vec3::new(sign * 0.34, 0.64, z);
        let perch = knuckle + Vec3::Y * 0.04;
        let axis = perch - tower;
        let body_end = tower + axis * 0.42;
        b.tube(tower, perch, 0.012, P_CHROME, 5, true);
        b.tube(perch, body_end, 0.026, P_ALLOY, 6, true);
        b.helix(tower, perch, 0.062, 0.020, 6, 4, P_ORANGE);
    }
    b.finish(BUGGY_SUSP_VERTS, BUGGY_SUSP_INDICES, "buggy suspension")
}

fn bike_head() -> Vec3 {
    Vec3::new(0.0, 0.62, 0.46)
}

fn bike_pivot() -> Vec3 {
    Vec3::new(0.0, 0.06, -0.08)
}

/// Motocross bike: cradle frame, tank, seat, and a crouched rider.
/// Forks and the swingarm live in [`bike_suspension`].
pub fn bike_body() -> CpuMesh {
    let mut b = B::new();
    let head = bike_head();
    let pivot = bike_pivot();
    let head_low = Vec3::new(0.0, 0.40, 0.38);
    b.tube(head_low, head, 0.048, P_CAGE, 6, true);
    b.tube(head, Vec3::new(0.0, 0.48, -0.02), 0.05, P_CAGE, 6, true);
    b.tube(
        Vec3::new(0.0, 0.48, -0.02),
        Vec3::new(0.0, 0.36, -0.46),
        0.042,
        P_CAGE,
        6,
        true,
    );
    b.tube(head_low, Vec3::new(0.0, 0.02, 0.16), 0.046, P_CAGE, 6, true);
    b.tube(Vec3::new(0.0, 0.02, 0.16), pivot, 0.046, P_ENGINE, 6, true);
    b.tube(pivot, Vec3::new(0.0, 0.34, -0.42), 0.036, P_CAGE, 6, true);

    b.box_at(
        Vec3::new(0.0, 0.08, -0.02),
        Vec3::new(0.14, 0.12, 0.18),
        P_ENGINE,
    );
    b.box_at(
        Vec3::new(0.0, 0.02, 0.10),
        Vec3::new(0.10, 0.08, 0.06),
        P_ENGINE,
    );
    for i in 0..3 {
        b.box_at(
            Vec3::new(0.0, 0.10, 0.20 + i as f32 * 0.035),
            Vec3::new(0.13, 0.08, 0.01),
            P_ALLOY,
        );
    }
    b.box_at(
        Vec3::new(0.0, -0.06, -0.02),
        Vec3::new(0.10, 0.02, 0.22),
        P_BELLY,
    );

    b.box_at(
        Vec3::new(0.0, 0.44, 0.14),
        Vec3::new(0.15, 0.10, 0.22),
        P_NOSE,
    );
    b.box_at(
        Vec3::new(0.0, 0.52, 0.20),
        Vec3::new(0.10, 0.035, 0.12),
        P_NOSE,
    );
    b.box_at(
        Vec3::new(0.0, 0.40, -0.16),
        Vec3::new(0.10, 0.045, 0.20),
        P_SEAT,
    );
    b.box_at(
        Vec3::new(0.0, 0.34, -0.40),
        Vec3::new(0.07, 0.04, 0.10),
        P_REAR,
    );

    b.tube(
        head + Vec3::new(0.0, 0.02, -0.02),
        Vec3::new(0.0, 0.70, 0.34),
        0.03,
        P_CAGE,
        5,
        true,
    );
    b.tube(
        Vec3::new(-0.34, 0.70, 0.32),
        Vec3::new(0.34, 0.70, 0.32),
        0.032,
        P_ALLOY,
        6,
        true,
    );
    b.tube(
        Vec3::new(-0.34, 0.70, 0.32),
        Vec3::new(-0.40, 0.66, 0.30),
        0.028,
        P_BOOT,
        5,
        true,
    );
    b.tube(
        Vec3::new(0.34, 0.70, 0.32),
        Vec3::new(0.40, 0.66, 0.30),
        0.028,
        P_BOOT,
        5,
        true,
    );

    // Number plate sits above the front tire, which tops out near y = 0.50.
    b.box_at(
        Vec3::new(0.0, 0.66, 0.78),
        Vec3::new(0.12, 0.10, 0.015),
        P_PLATE,
    );
    b.box_at(
        Vec3::new(0.0, 0.66, 0.80),
        Vec3::new(0.035, 0.055, 0.012),
        P_BOOT,
    );
    b.tube(
        Vec3::new(0.0, 0.52, 0.70),
        Vec3::new(0.0, 0.52, 0.82),
        0.04,
        P_LAMP,
        7,
        true,
    );

    b.tube(
        Vec3::new(0.14, 0.04, -0.08),
        Vec3::new(0.16, 0.10, -0.48),
        0.032,
        P_ALLOY,
        6,
        true,
    );
    b.box_at(
        Vec3::new(0.16, 0.12, -0.52),
        Vec3::new(0.04, 0.04, 0.07),
        P_ENGINE,
    );

    for s in [-1.0_f32, 1.0] {
        b.box_at(
            Vec3::new(s * 0.16, 0.02, -0.02),
            Vec3::new(0.06, 0.014, 0.025),
            P_CAGE,
        );
    }

    add_rider(&mut b);
    b.finish(80_000, 200_000, "bike body")
}

fn add_rider(b: &mut B) {
    let hip = Vec3::new(0.0, 0.48, -0.10);
    let shoulder = Vec3::new(0.0, 0.66, 0.08);
    let helmet = Vec3::new(0.0, 0.80, 0.14);
    b.tube(hip, shoulder, 0.09, P_JERSEY, 7, true);
    b.sphere(helmet, 0.12, P_HELMET, 6, 8);
    b.box_at(
        helmet + Vec3::new(0.0, -0.01, 0.09),
        Vec3::new(0.07, 0.035, 0.025),
        P_VISOR,
    );
    b.tube(
        shoulder + Vec3::new(-0.06, 0.0, 0.0),
        Vec3::new(-0.30, 0.68, 0.30),
        0.045,
        P_JERSEY,
        6,
        true,
    );
    b.tube(
        shoulder + Vec3::new(0.06, 0.0, 0.0),
        Vec3::new(0.30, 0.68, 0.30),
        0.045,
        P_JERSEY,
        6,
        true,
    );
    b.tube(
        hip + Vec3::new(-0.04, 0.0, 0.0),
        Vec3::new(-0.14, 0.26, 0.06),
        0.05,
        P_BOOT,
        6,
        true,
    );
    b.tube(
        hip + Vec3::new(0.04, 0.0, 0.0),
        Vec3::new(0.14, 0.26, 0.06),
        0.05,
        P_BOOT,
        6,
        true,
    );
    b.tube(
        Vec3::new(-0.14, 0.26, 0.06),
        Vec3::new(-0.18, 0.04, -0.02),
        0.042,
        P_BOOT,
        5,
        true,
    );
    b.tube(
        Vec3::new(0.14, 0.26, 0.06),
        Vec3::new(0.18, 0.04, -0.02),
        0.042,
        P_BOOT,
        5,
        true,
    );
    b.sphere(Vec3::new(-0.32, 0.67, 0.31), 0.04, P_BOOT, 4, 6);
    b.sphere(Vec3::new(0.32, 0.67, 0.31), 0.04, P_BOOT, 4, 6);
}

/// Fork, swingarm, shock, and the fenders. They track the wheel centers.
pub fn bike_suspension(bike: &Bike) -> CpuMesh {
    let mut b = B::new();
    let front = bike.wheel_center_local(0);
    let rear = bike.wheel_center_local(1);
    let head = bike_head();
    let pivot = bike_pivot();
    // Legs sit just outside the rubber and drop onto the protruding hub.
    let axle_x = BIKE_WIDTH * 0.5 + 0.04;
    for s in [-1.0_f32, 1.0] {
        let x = s * axle_x;
        let crown = head + Vec3::new(x * 0.55, 0.0, 0.0);
        let axle = Vec3::new(x, front.y, front.z);
        let span = axle - crown;
        let len = span.length().max(0.05);
        let dir = span / len;
        let upper = crown + dir * (len * 0.52).min(len - 0.04);
        b.tube(crown, upper, 0.032, P_ALLOY, 6, true);
        b.tube(upper - dir * 0.05, axle, 0.042, P_ALLOY, 6, true);
    }
    b.tube(
        head + Vec3::new(-0.08, -0.02, 0.0),
        head + Vec3::new(0.08, -0.02, 0.0),
        0.028,
        P_ALLOY,
        5,
        true,
    );

    for s in [-1.0_f32, 1.0] {
        let x = s * axle_x;
        b.tube(
            pivot + Vec3::new(x * 0.45, 0.0, 0.0),
            Vec3::new(x, rear.y, rear.z),
            0.03,
            P_CAGE,
            6,
            true,
        );
    }
    b.tube(
        pivot + Vec3::new(-axle_x * 0.5, 0.0, 0.0),
        pivot + Vec3::new(axle_x * 0.5, 0.0, 0.0),
        0.032,
        P_ENGINE,
        5,
        true,
    );

    let tower = Vec3::new(0.0, 0.40, -0.20);
    let perch = rear + Vec3::new(0.0, 0.06, 0.16);
    let rod_end = tower.lerp(perch, 0.5);
    b.tube(tower, rod_end, 0.014, P_ALLOY, 5, true);
    b.tube(rod_end, perch, 0.032, P_ENGINE, 6, true);
    b.helix(tower, perch, 0.055, 0.018, 5, 4, P_ENGINE);

    add_guard(&mut b, front, BIKE_RADIUS, 0.10, -0.35, 1.15, P_NOSE);
    add_guard(&mut b, rear, BIKE_RADIUS, 0.11, -1.15, 0.55, P_REAR);
    b.finish(BIKE_SUSP_VERTS, BIKE_SUSP_INDICES, "bike suspension")
}

fn add_guard(b: &mut B, axle: Vec3, radius: f32, half_w: f32, start: f32, end: f32, panel: f32) {
    let segs = 5;
    let thick = 0.03;
    let mid = radius + 0.05;
    for i in 0..segs {
        let t0 = start + (end - start) * (i as f32 / segs as f32);
        let t1 = start + (end - start) * ((i + 1) as f32 / segs as f32);
        let dir = |t: f32| Vec3::new(0.0, t.cos(), t.sin());
        let p = |t: f32, r: f32, x: f32| axle + dir(t) * r + Vec3::new(x, 0.0, 0.0);
        let outer = mid + thick;
        b.quad(
            [
                p(t0, outer, -half_w),
                p(t0, outer, half_w),
                p(t1, outer, half_w),
                p(t1, outer, -half_w),
            ],
            panel,
        );
        b.quad(
            [
                p(t0, mid, half_w),
                p(t0, mid, -half_w),
                p(t1, mid, -half_w),
                p(t1, mid, half_w),
            ],
            panel,
        );
        b.quad(
            [
                p(t0, mid, -half_w),
                p(t0, outer, -half_w),
                p(t1, outer, -half_w),
                p(t1, mid, -half_w),
            ],
            panel,
        );
        b.quad(
            [
                p(t0, outer, half_w),
                p(t0, mid, half_w),
                p(t1, mid, half_w),
                p(t1, outer, half_w),
            ],
            panel,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn yz_radius(v: &MeshVertex) -> f32 {
        (v.pos[1] * v.pos[1] + v.pos[2] * v.pos[2]).sqrt()
    }

    fn panels(mesh: &CpuMesh) -> Vec<i32> {
        let mut ids: Vec<i32> = mesh
            .vertices
            .iter()
            .map(|v| (v.panel + 0.5) as i32)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn assert_indexed(mesh: &CpuMesh) {
        assert!(!mesh.vertices.is_empty(), "empty mesh");
        assert!(!mesh.indices.is_empty(), "no triangles");
        assert_eq!(mesh.indices.len() % 3, 0);
        for i in &mesh.indices {
            assert!(
                (*i as usize) < mesh.vertices.len(),
                "index {i} out of {}",
                mesh.vertices.len()
            );
        }
    }

    #[test]
    fn wheels_match_solver_sizes() {
        let buggy = buggy_wheel();
        assert_indexed(&buggy);
        let rubber: Vec<_> = buggy
            .vertices
            .iter()
            .filter(|v| (v.panel - P_RUBBER).abs() < 0.1)
            .collect();
        let max_r = rubber.iter().map(|v| yz_radius(v)).fold(0.0_f32, f32::max);
        let min_r = rubber.iter().map(|v| yz_radius(v)).fold(10.0_f32, f32::min);
        assert!((max_r - WHEEL_RADIUS).abs() < 0.004, "buggy radius {max_r}");
        assert!(min_r < max_r - 0.04, "buggy tread has no groove");
        let max_x = buggy
            .vertices
            .iter()
            .map(|v| v.pos[0].abs())
            .fold(0.0_f32, f32::max);
        assert!(max_x <= WHEEL_WIDTH * 0.5 + 0.01, "buggy width {max_x}");

        let bike = bike_wheel();
        assert_indexed(&bike);
        let rubber: Vec<_> = bike
            .vertices
            .iter()
            .filter(|v| (v.panel - P_RUBBER).abs() < 0.1)
            .collect();
        let max_r = rubber.iter().map(|v| yz_radius(v)).fold(0.0_f32, f32::max);
        assert!((max_r - BIKE_RADIUS).abs() < 0.004, "bike radius {max_r}");
        let max_x = bike
            .vertices
            .iter()
            .filter(|v| (v.panel - P_RUBBER).abs() < 0.1)
            .map(|v| v.pos[0].abs())
            .fold(0.0_f32, f32::max);
        assert!(
            (max_x - BIKE_WIDTH * 0.5).abs() < 0.004,
            "bike rubber width {max_x}"
        );
        assert!(panels(&bike).contains(&18), "bike wheel needs an alloy hub");
    }

    #[test]
    fn tire_normals_point_outward() {
        for mesh in [buggy_wheel(), bike_wheel()] {
            let mut outward = 0.0;
            let mut n = 0.0;
            for v in &mesh.vertices {
                if (v.panel - P_RUBBER).abs() > 0.1 {
                    continue;
                }
                let radial = Vec3::new(0.0, v.pos[1], v.pos[2]);
                if radial.length() < 0.05 {
                    continue;
                }
                outward += Vec3::from(v.nrm).dot(radial.normalize());
                n += 1.0;
            }
            let avg = outward / n;
            assert!(avg > 0.35, "rubber normals face in ({avg})");
        }
    }

    #[test]
    fn bodies_expose_mud_panels_and_suspension_fits() {
        let body = buggy_body();
        assert_indexed(&body);
        let ids = panels(&body);
        for id in 0..12 {
            assert!(ids.contains(&id), "buggy missing mud panel {id}");
        }
        assert!(body.vertices.iter().any(|v| v.pos[0].abs() < 0.42));
        assert!(ids.contains(&19), "orange lower chassis");
        let crown: Vec<_> = buggy_wheel()
            .vertices
            .into_iter()
            .filter(|v| (v.panel - P_RUBBER).abs() < 0.1 && v.pos[0].abs() < 0.04)
            .collect();
        let crown_max = crown.iter().map(yz_radius).fold(0.0_f32, f32::max);
        let crown_min = crown.iter().map(yz_radius).fold(10.0_f32, f32::min);
        assert!(
            crown_max - crown_min > 0.03,
            "tread blocks {crown_min}..{crown_max}"
        );

        let bike = bike_body();
        assert_indexed(&bike);
        let ids = panels(&bike);
        for id in [P_NOSE, P_BELLY, P_SEAT, P_HELMET, P_JERSEY] {
            assert!(
                ids.iter().any(|p| (*p as f32 - id).abs() < 0.6),
                "bike missing {id}"
            );
        }

        let mut vehicle = Vehicle::new();
        for compression in [0.0, 0.08, vehicle::SUSP_TRAVEL] {
            for w in &mut vehicle.wheels {
                w.compression = compression;
            }
            let susp = buggy_suspension(&vehicle);
            assert_indexed(&susp);
            assert!(susp.vertices.len() <= BUGGY_SUSP_VERTS);
            assert!(susp.indices.len() <= BUGGY_SUSP_INDICES);
            assert!(
                panels(&susp).contains(&19),
                "A-arms and springs stay orange"
            );
        }

        let mut bike = Bike::new();
        for compression in [0.0, 0.1, 0.32] {
            for w in &mut bike.wheels {
                w.compression = compression;
            }
            let susp = bike_suspension(&bike);
            assert_indexed(&susp);
            assert!(
                susp.vertices.len() <= BIKE_SUSP_VERTS,
                "{}",
                susp.vertices.len()
            );
            assert!(susp.indices.len() <= BIKE_SUSP_INDICES);
        }
    }
}
