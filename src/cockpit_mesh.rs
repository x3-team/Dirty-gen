//! Tube-cage interior for the cockpit cameras.
//!
//! Wheel, dash, seats, cage, wet metal, and mud on the cowl. Drawn only
//! while a cockpit view is active. The front and the sides stay open, so
//! the track is still in frame. The buggy and bike meshes are not edited here.

use glam::Vec3;

use silt::cockpit::WHEEL_PIVOT;

use crate::mesh::{CpuMesh, MeshVertex, PANEL_CLOD};

/// Wet painted tube, dash, and floor. Matched in `cockpit_shade`.
const PANEL_WET_METAL: f32 = 47.0;
/// Seat cloth.
const PANEL_FABRIC: f32 = 48.0;
/// Cowl sheet. `uv.y` is local z so the nose can carry more mud.
const PANEL_COWL: f32 = 49.0;
/// Steering rim. Spokes stay on the wet-metal id.
const PANEL_RIM: f32 = 50.0;

// Rises toward the nose. A slope that falls with the sight line is edge-on
// from the seat, so the sheet would vanish behind the dash.
const COWL_Z0: f32 = 0.48;
const COWL_Z1: f32 = 1.42;
const COWL_Y0: f32 = 0.33;
const COWL_Y1: f32 = 0.40;
const COWL_X0: f32 = 0.36;
const COWL_X1: f32 = 0.20;

pub fn shell() -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    add_cage(&mut mesh);
    add_roof(&mut mesh);
    add_dash(&mut mesh);
    add_cowl(&mut mesh);
    add_floor(&mut mesh);
    add_seats(&mut mesh);
    add_column(&mut mesh);
    mesh
}

pub fn wheel() -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    // Ring in XY. +Z is the column, aimed at the driver by `wheel_local`.
    add_torus(&mut mesh, Vec3::ZERO, 0.158, 0.020, PANEL_RIM);
    let hub = PANEL_WET_METAL;
    add_tube(
        &mut mesh,
        Vec3::new(0.0, 0.0, -0.02),
        Vec3::new(0.0, 0.0, 0.028),
        0.034,
        hub,
        8,
    );
    add_disc(&mut mesh, Vec3::new(0.0, 0.0, 0.030), 0.028, 1.0, hub, 10);
    // Three spokes, one of them down, the way the stills read.
    for ang in [-std::f32::consts::FRAC_PI_2, 0.55, 2.60] {
        let dir = Vec3::new(ang.cos(), ang.sin(), 0.0);
        add_tube(&mut mesh, dir * 0.036, dir * 0.142, 0.011, hub, 5);
    }
    // A few clods stuck to the rim. They spin with the wheel.
    for ang in [0.4_f32, 2.1, 3.7, 5.3] {
        let dir = Vec3::new(ang.cos(), ang.sin(), 0.0);
        add_box(
            &mut mesh,
            dir * 0.162 + Vec3::Z * 0.014,
            Vec3::new(0.012, 0.007, 0.006),
            PANEL_CLOD,
        );
    }
    mesh
}

fn add_cage(mesh: &mut CpuMesh) {
    let metal = PANEL_WET_METAL;
    let a_l = Vec3::new(-0.42, 0.34, 0.78);
    let a_r = Vec3::new(0.42, 0.34, 0.78);
    let h_l = Vec3::new(-0.40, 0.80, 0.42);
    let h_r = Vec3::new(0.40, 0.80, 0.42);
    let m_l = Vec3::new(-0.42, 0.96, -0.06);
    let m_r = Vec3::new(0.42, 0.96, -0.04);
    let r_l = Vec3::new(-0.42, 0.96, -0.56);
    let r_r = Vec3::new(0.42, 0.96, -0.56);
    let f_l = Vec3::new(-0.40, 0.16, -0.58);
    let f_r = Vec3::new(0.40, 0.16, -0.58);
    let b_l = Vec3::new(-0.44, 0.20, -0.18);
    let b_r = Vec3::new(0.44, 0.20, -0.18);
    let bt_l = Vec3::new(-0.43, 0.98, -0.16);
    let bt_r = Vec3::new(0.43, 0.98, -0.14);
    let s_lf = Vec3::new(-0.43, 0.20, 0.62);
    let s_lr = Vec3::new(-0.42, 0.16, -0.50);
    let s_rf = Vec3::new(0.43, 0.20, 0.62);
    let s_rr = Vec3::new(0.42, 0.16, -0.50);
    // Hip rail, below the sight line. A tube across the dash blocks the cowl.
    let p_l = Vec3::new(-0.43, 0.28, 0.55);
    let p_r = Vec3::new(0.43, 0.28, 0.55);
    let q_l = Vec3::new(-0.44, 0.26, -0.35);
    let q_r = Vec3::new(0.44, 0.26, -0.35);
    let harness_l = Vec3::new(-0.30, 0.58, -0.46);
    let harness_r = Vec3::new(0.30, 0.58, -0.46);

    let main = 0.032;
    let bar = 0.024;
    let tubes = [
        (a_l, h_l, main),
        (a_r, h_r, main),
        (h_l, h_r, main),
        (h_l, m_l, main),
        (h_r, m_r, main),
        (m_l, r_l, main),
        (m_r, r_r, main),
        (r_l, r_r, main),
        (f_l, r_l, main),
        (f_r, r_r, main),
        (b_l, bt_l, bar),
        (b_r, bt_r, bar),
        (bt_l, bt_r, bar),
        (bt_l, m_l, bar),
        (bt_r, m_r, bar),
        (s_lf, s_lr, bar),
        (s_rf, s_rr, bar),
        (p_l, q_l, bar),
        (p_r, q_r, bar),
        (harness_l, harness_r, bar),
        (f_l, harness_l, bar),
        (f_r, harness_r, bar),
    ];
    for (a, b, radius) in tubes {
        add_tube(mesh, a, b, radius, metal, 8);
    }
    let joints = [
        a_l, a_r, h_l, h_r, m_l, m_r, r_l, r_r, f_l, f_r, b_l, b_r, bt_l, bt_r, s_lf, s_lr, s_rf,
        s_rr, p_l, p_r, q_l, q_r, harness_l, harness_r,
    ];
    for joint in joints {
        add_octa(mesh, joint, 0.040, metal);
    }
    // Mirror under the header. Small, so the view forward stays open.
    add_box(
        mesh,
        Vec3::new(0.0, 0.74, 0.40),
        Vec3::new(0.09, 0.028, 0.012),
        metal,
    );
}

fn add_roof(mesh: &mut CpuMesh) {
    // Visor over the seats and the header. The nose stays open to the sky.
    add_box(
        mesh,
        Vec3::new(0.0, 0.92, 0.08),
        Vec3::new(0.36, 0.010, 0.34),
        PANEL_WET_METAL,
    );
}

fn add_dash(mesh: &mut CpuMesh) {
    let metal = PANEL_WET_METAL;
    // Low face. The cowl continues forward from the top edge.
    add_box(
        mesh,
        Vec3::new(0.0, 0.24, 0.40),
        Vec3::new(0.38, 0.07, 0.07),
        metal,
    );
    // Closes the gap under the dash so the orange nose stays out of the tub.
    add_box(
        mesh,
        Vec3::new(0.0, 0.16, 0.34),
        Vec3::new(0.36, 0.06, 0.02),
        metal,
    );
    // Gauges on the driver side of the face. The wheel covers the middle.
    add_gauge(mesh, Vec3::new(-0.24, 0.27, 0.325));
    add_gauge(mesh, Vec3::new(0.06, 0.28, 0.325));
    add_gauge(mesh, Vec3::new(0.20, 0.26, 0.325));
    // Cheeks hide the orange side pods beside the dash.
    for x in [-0.36_f32, 0.36] {
        add_box(
            mesh,
            Vec3::new(x, 0.20, 0.18),
            Vec3::new(0.055, 0.11, 0.20),
            metal,
        );
    }
    let smears = [
        (-0.16, 0.318, 0.40),
        (0.08, 0.318, 0.44),
        (0.22, 0.316, 0.38),
    ];
    for (x, y, z) in smears {
        add_box(
            mesh,
            Vec3::new(x, y, z),
            Vec3::new(0.045, 0.008, 0.028),
            PANEL_CLOD,
        );
    }
}

fn add_gauge(mesh: &mut CpuMesh, face: Vec3) {
    add_box(
        mesh,
        face + Vec3::new(0.0, 0.0, 0.012),
        Vec3::new(0.046, 0.046, 0.012),
        PANEL_WET_METAL,
    );
    add_torus(mesh, face, 0.034, 0.006, PANEL_WET_METAL);
    add_disc(mesh, face + Vec3::Z * -0.002, 0.026, -1.0, PANEL_FABRIC, 10);
}

fn add_cowl(mesh: &mut CpuMesh) {
    let xl0 = -COWL_X0;
    let xr0 = COWL_X0;
    let xl1 = -COWL_X1;
    let xr1 = COWL_X1;
    let top = [
        Vec3::new(xl0, COWL_Y0, COWL_Z0),
        Vec3::new(xl1, COWL_Y1, COWL_Z1),
        Vec3::new(xr1, COWL_Y1, COWL_Z1),
        Vec3::new(xr0, COWL_Y0, COWL_Z0),
    ];
    let uv = [
        [xl0, COWL_Z0],
        [xl1, COWL_Z1],
        [xr1, COWL_Z1],
        [xr0, COWL_Z0],
    ];
    add_quad_uv(mesh, top, uv, PANEL_COWL);
    let thick = 0.04;
    let bottom = [
        top[0] - Vec3::Y * thick,
        top[3] - Vec3::Y * thick,
        top[2] - Vec3::Y * thick,
        top[1] - Vec3::Y * thick,
    ];
    let metal = PANEL_WET_METAL;
    add_quad(mesh, bottom, metal);
    add_quad(
        mesh,
        [
            top[0],
            top[0] - Vec3::Y * thick,
            top[1] - Vec3::Y * thick,
            top[1],
        ],
        metal,
    );
    add_quad(
        mesh,
        [
            top[3],
            top[2],
            top[2] - Vec3::Y * thick,
            top[3] - Vec3::Y * thick,
        ],
        metal,
    );
    add_quad(
        mesh,
        [
            top[1],
            top[1] - Vec3::Y * thick,
            top[2] - Vec3::Y * thick,
            top[2],
        ],
        metal,
    );
    add_quad(
        mesh,
        [
            top[3],
            top[3] - Vec3::Y * thick,
            top[0] - Vec3::Y * thick,
            top[0],
        ],
        metal,
    );

    // Heavier toward the nose. The sheet shader already wets the paint.
    let clumps = [
        (0.0, 0.78, 0.04),
        (-0.10, 0.90, 0.048),
        (0.12, 0.98, 0.042),
        (-0.04, 1.08, 0.055),
        (0.08, 1.20, 0.06),
        (-0.08, 1.30, 0.05),
        (0.0, 1.40, 0.065),
        (0.10, 0.84, 0.032),
    ];
    for (x, z, s) in clumps {
        let y = cowl_y(z) + 0.012 + s * 0.35;
        add_box(
            mesh,
            Vec3::new(x, y, z),
            Vec3::new(s, s * 0.45, s * 0.75),
            PANEL_CLOD,
        );
    }
    // Broad wet smears on the sheet the seat cameras look down on.
    // Up-facing quads only: a box side would fail the cowl normal check.
    for (x, z, sx, sz) in [
        (0.0, 0.72, 0.16, 0.06),
        (-0.08, 0.96, 0.12, 0.07),
        (0.06, 1.16, 0.14, 0.08),
    ] {
        add_cowl_smear(mesh, x, z, sx, sz);
    }
}

fn add_cowl_smear(mesh: &mut CpuMesh, x: f32, z: f32, sx: f32, sz: f32) {
    let y = cowl_y(z) + 0.006;
    let p = [
        Vec3::new(x - sx, y, z - sz),
        Vec3::new(x - sx, y, z + sz),
        Vec3::new(x + sx, y, z + sz),
        Vec3::new(x + sx, y, z - sz),
    ];
    let uv = [
        [p[0].x, p[0].z],
        [p[1].x, p[1].z],
        [p[2].x, p[2].z],
        [p[3].x, p[3].z],
    ];
    add_quad_uv(mesh, p, uv, PANEL_COWL);
}

fn cowl_y(z: f32) -> f32 {
    let t = ((z - COWL_Z0) / (COWL_Z1 - COWL_Z0)).clamp(0.0, 1.0);
    COWL_Y0 + (COWL_Y1 - COWL_Y0) * t
}

fn add_floor(mesh: &mut CpuMesh) {
    let metal = PANEL_WET_METAL;
    add_box(
        mesh,
        Vec3::new(0.0, 0.11, 0.08),
        Vec3::new(0.36, 0.012, 0.55),
        metal,
    );
    add_box(
        mesh,
        Vec3::new(0.0, 0.18, 0.00),
        Vec3::new(0.040, 0.055, 0.20),
        metal,
    );
    add_tube(
        mesh,
        Vec3::new(0.0, 0.24, 0.08),
        Vec3::new(0.02, 0.36, 0.02),
        0.010,
        metal,
        5,
    );
    add_octa(mesh, Vec3::new(0.02, 0.36, 0.02), 0.018, metal);
    // Sill walls stop at the hip. Above that the side is open to the track.
    // The outer skin covers the orange side pods from inside the tub.
    for x in [-0.37_f32, 0.37] {
        add_box(
            mesh,
            Vec3::new(x, 0.18, 0.02),
            Vec3::new(0.028, 0.09, 0.46),
            metal,
        );
    }
    for (x, z) in [(-0.16, 0.22), (0.14, -0.05), (0.02, 0.34)] {
        add_box(
            mesh,
            Vec3::new(x, 0.132, z),
            Vec3::new(0.06, 0.012, 0.045),
            PANEL_CLOD,
        );
    }
    // Pedals under the dash, driver side.
    add_box(
        mesh,
        Vec3::new(-0.20, 0.16, 0.40),
        Vec3::new(0.035, 0.028, 0.045),
        metal,
    );
    add_box(
        mesh,
        Vec3::new(-0.08, 0.15, 0.38),
        Vec3::new(0.030, 0.022, 0.04),
        metal,
    );
}

fn add_seats(mesh: &mut CpuMesh) {
    // Both buckets sit ahead of the driver eye, one each side of the tunnel,
    // so the back, bolster, and cushion land in the frame instead of beside the lens.
    add_bucket(mesh, -0.22, -0.34);
    add_bucket(mesh, 0.22, -0.08);
}

fn add_bucket(mesh: &mut CpuMesh, x: f32, z_bias: f32) {
    let fabric = PANEL_FABRIC;
    let hx = 0.10;
    let z = |v: f32| v + z_bias;
    add_box(
        mesh,
        Vec3::new(x, 0.22, z(0.02)),
        Vec3::new(hx, 0.05, 0.15),
        fabric,
    );
    add_box(
        mesh,
        Vec3::new(x, 0.26, z(0.15)),
        Vec3::new(hx * 0.88, 0.028, 0.035),
        fabric,
    );
    let outer = if x < 0.0 { -1.0 } else { 1.0 };
    add_box(
        mesh,
        Vec3::new(x + outer * 0.10, 0.34, z(0.04)),
        Vec3::new(0.032, 0.10, 0.13),
        fabric,
    );
    add_box(
        mesh,
        Vec3::new(x - outer * 0.095, 0.32, z(0.05)),
        Vec3::new(0.028, 0.09, 0.12),
        fabric,
    );
    let y0 = 0.26;
    let z0 = z(0.06);
    let y1 = 0.56;
    let z1 = z(-0.06);
    let face = [
        Vec3::new(x - hx, y0, z0),
        Vec3::new(x + hx, y0, z0),
        Vec3::new(x + hx * 0.92, y1, z1),
        Vec3::new(x - hx * 0.92, y1, z1),
    ];
    add_extruded(mesh, face, Vec3::new(0.0, -0.02, -0.045), fabric);
    // Ribs on the rear face, the side the driver looks at.
    for yt in [0.40_f32, 0.62] {
        let yy = y0 + (y1 - y0) * yt;
        let zz = z0 + (z1 - z0) * yt;
        add_box(
            mesh,
            Vec3::new(x, yy - 0.02, zz - 0.058),
            Vec3::new(hx * 0.68, 0.016, 0.010),
            fabric,
        );
    }
    add_box(
        mesh,
        Vec3::new(x + outer * 0.11, 0.56, z(-0.01)),
        Vec3::new(0.030, 0.08, 0.038),
        fabric,
    );
    add_box(
        mesh,
        Vec3::new(x, 0.70, z(-0.06)),
        Vec3::new(0.048, 0.042, 0.030),
        fabric,
    );
    let buckle = Vec3::new(x - outer * 0.02, 0.30, z(0.06));
    add_tube(
        mesh,
        Vec3::new(x + outer * 0.06, 0.60, z(-0.08)),
        buckle,
        0.008,
        PANEL_WET_METAL,
        4,
    );
    add_tube(
        mesh,
        Vec3::new(x - outer * 0.05, 0.56, z(-0.06)),
        buckle,
        0.008,
        PANEL_WET_METAL,
        4,
    );
    add_octa(mesh, buckle, 0.016, PANEL_WET_METAL);
    // One dried smear on the cushion.
    add_box(
        mesh,
        Vec3::new(x + outer * 0.04, 0.26, z(0.02)),
        Vec3::new(0.035, 0.01, 0.028),
        PANEL_CLOD,
    );
}

fn add_column(mesh: &mut CpuMesh) {
    // From the hub into the dash. The hub itself is the steered wheel mesh.
    let hub = WHEEL_PIVOT;
    add_tube(
        mesh,
        hub + Vec3::new(0.01, -0.01, 0.03),
        Vec3::new(-0.06, 0.30, 0.40),
        0.016,
        PANEL_WET_METAL,
        6,
    );
}

fn add_tri(mesh: &mut CpuMesh, a: Vec3, b: Vec3, c: Vec3, panel: f32) {
    add_tri_uv(mesh, a, b, c, uv_of(a), uv_of(b), uv_of(c), panel);
}

fn add_tri_uv(
    mesh: &mut CpuMesh,
    a: Vec3,
    b: Vec3,
    c: Vec3,
    ua: [f32; 2],
    ub: [f32; 2],
    uc: [f32; 2],
    panel: f32,
) {
    let n = (b - a).cross(c - a).normalize_or_zero();
    if n.length_squared() < 1.0e-8 {
        return;
    }
    let base = mesh.vertices.len() as u32;
    for (p, uv) in [(a, ua), (b, ub), (c, uc)] {
        mesh.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.to_array(),
            uv,
            panel,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn uv_of(p: Vec3) -> [f32; 2] {
    [p.x * 1.6 + p.z * 0.2, p.y * 1.35 + 0.55]
}

fn add_quad(mesh: &mut CpuMesh, p: [Vec3; 4], panel: f32) {
    add_tri(mesh, p[0], p[1], p[2], panel);
    add_tri(mesh, p[0], p[2], p[3], panel);
}

fn add_quad_uv(mesh: &mut CpuMesh, p: [Vec3; 4], uv: [[f32; 2]; 4], panel: f32) {
    add_tri_uv(mesh, p[0], p[1], p[2], uv[0], uv[1], uv[2], panel);
    add_tri_uv(mesh, p[0], p[2], p[3], uv[0], uv[2], uv[3], panel);
}

fn add_extruded(mesh: &mut CpuMesh, face: [Vec3; 4], extrude: Vec3, panel: f32) {
    let back = [
        face[0] + extrude,
        face[1] + extrude,
        face[2] + extrude,
        face[3] + extrude,
    ];
    add_quad(mesh, face, panel);
    add_quad(mesh, [back[0], back[3], back[2], back[1]], panel);
    for i in 0..4 {
        let j = (i + 1) % 4;
        add_quad(mesh, [face[i], back[i], back[j], face[j]], panel);
    }
}

fn add_box(mesh: &mut CpuMesh, c: Vec3, h: Vec3, panel: f32) {
    let (x, y, z) = (Vec3::X * h.x, Vec3::Y * h.y, Vec3::Z * h.z);
    add_quad(
        mesh,
        [c - x + y - z, c - x + y + z, c + x + y + z, c + x + y - z],
        panel,
    );
    add_quad(
        mesh,
        [c - x - y - z, c + x - y - z, c + x - y + z, c - x - y + z],
        panel,
    );
    add_quad(
        mesh,
        [c + x - y - z, c + x + y - z, c + x + y + z, c + x - y + z],
        panel,
    );
    add_quad(
        mesh,
        [c - x - y - z, c - x - y + z, c - x + y + z, c - x + y - z],
        panel,
    );
    add_quad(
        mesh,
        [c - x - y + z, c + x - y + z, c + x + y + z, c - x + y + z],
        panel,
    );
    add_quad(
        mesh,
        [c + x - y - z, c - x - y - z, c - x + y - z, c + x + y - z],
        panel,
    );
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

fn add_octa(mesh: &mut CpuMesh, c: Vec3, r: f32, panel: f32) {
    let px = c + Vec3::X * r;
    let nx = c - Vec3::X * r;
    let py = c + Vec3::Y * r;
    let ny = c - Vec3::Y * r;
    let pz = c + Vec3::Z * r;
    let nz = c - Vec3::Z * r;
    let faces = [
        [px, py, pz],
        [px, pz, ny],
        [px, ny, nz],
        [px, nz, py],
        [nx, pz, py],
        [nx, ny, pz],
        [nx, nz, ny],
        [nx, py, nz],
    ];
    for face in faces {
        add_tri(mesh, face[0], face[1], face[2], panel);
    }
}

fn add_torus(mesh: &mut CpuMesh, center: Vec3, radius: f32, tube: f32, panel: f32) {
    let major = 18;
    let minor = 8;
    for i in 0..major {
        let a0 = i as f32 / major as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / major as f32 * std::f32::consts::TAU;
        for j in 0..minor {
            let b0 = j as f32 / minor as f32 * std::f32::consts::TAU;
            let b1 = (j + 1) as f32 / minor as f32 * std::f32::consts::TAU;
            let p = |a: f32, b: f32| {
                let r = radius + tube * b.cos();
                center + Vec3::new(r * a.cos(), r * a.sin(), tube * b.sin())
            };
            add_quad(mesh, [p(a0, b0), p(a1, b0), p(a1, b1), p(a0, b1)], panel);
        }
    }
}

fn add_disc(mesh: &mut CpuMesh, c: Vec3, radius: f32, normal_z: f32, panel: f32, segs: usize) {
    for i in 0..segs {
        let t0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let t1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let p = |t: f32| c + Vec3::new(radius * t.cos(), radius * t.sin(), 0.0);
        if normal_z >= 0.0 {
            add_tri(mesh, c, p(t0), p(t1), panel);
        } else {
            add_tri(mesh, c, p(t1), p(t0), panel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use silt::cockpit::{cockpit_pose, wheel_local, COCKPIT_VIEWS};
    use silt::vehicle::Vehicle;

    fn has_panel(mesh: &CpuMesh, panel: f32) -> bool {
        mesh.vertices.iter().any(|v| (v.panel - panel).abs() < 0.1)
    }

    fn bounds(mesh: &CpuMesh) -> (Vec3, Vec3) {
        let mut lo = Vec3::splat(f32::MAX);
        let mut hi = Vec3::splat(f32::MIN);
        for v in &mesh.vertices {
            let p = Vec3::from(v.pos);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        (lo, hi)
    }

    fn assert_unit_normals(mesh: &CpuMesh) {
        assert_eq!(mesh.indices.len() % 3, 0);
        for v in &mesh.vertices {
            let n = Vec3::from(v.nrm);
            assert!((n.length() - 1.0).abs() < 1.0e-3, "normal {n:?}");
            assert!(v.pos.iter().all(|c| c.is_finite()));
        }
    }

    #[test]
    fn cockpit_furniture_is_closed_triangles() {
        let shell = shell();
        let wheel = wheel();
        assert!(shell.indices.len() > 200);
        assert!(wheel.indices.len() > 36);
        assert_unit_normals(&shell);
        assert_unit_normals(&wheel);
        assert!(has_panel(&shell, PANEL_CLOD));
        assert!(has_panel(&shell, PANEL_WET_METAL));
        assert!(has_panel(&shell, PANEL_FABRIC));
        assert!(has_panel(&shell, PANEL_COWL));
        assert!(has_panel(&wheel, PANEL_RIM));
        assert!(has_panel(&wheel, PANEL_WET_METAL));
    }

    #[test]
    fn cage_seats_and_cowl_span_the_tub() {
        let shell = shell();
        let (lo, hi) = bounds(&shell);
        assert!(hi.y > 0.9, "roof tubes, hi={hi:?}");
        assert!(lo.x < -0.38 && hi.x > 0.38, "cage width {lo:?} {hi:?}");
        assert!(hi.z > 1.3, "cowl reaches the nose");
        assert!(lo.z < -0.45, "rear hoop");
        assert!(
            shell
                .vertices
                .iter()
                .filter(|v| (v.panel - PANEL_COWL).abs() < 0.1)
                .all(|v| v.nrm[1] > 0.35),
            "cowl sheet faces up"
        );
        assert!(shell.vertices.iter().any(|v| {
            (v.panel - PANEL_FABRIC).abs() < 0.1 && v.pos[1] > 0.6 && v.pos[1] < 0.85
        }));
    }

    fn dist_to_seg(p: Vec3, a: Vec3, b: Vec3) -> f32 {
        let ab = b - a;
        let len2 = ab.length_squared();
        if len2 < 1.0e-12 {
            return (p - a).length();
        }
        let t = ((p - a).dot(ab) / len2).clamp(0.0, 1.0);
        (p - (a + ab * t)).length()
    }

    fn dist_to_tri(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
        let ab = b - a;
        let ac = c - a;
        let n = ab.cross(ac);
        let nlen = n.length();
        if nlen < 1.0e-6 {
            return dist_to_seg(p, a, b).min(dist_to_seg(p, a, c));
        }
        let nn = n / nlen;
        let plane = (p - a).dot(nn);
        let proj = p - nn * plane;
        let bp = proj - a;
        let d00 = ab.dot(ab);
        let d01 = ab.dot(ac);
        let d11 = ac.dot(ac);
        let d20 = bp.dot(ab);
        let d21 = bp.dot(ac);
        let denom = d00 * d11 - d01 * d01;
        if denom.abs() < 1.0e-10 {
            return plane.abs();
        }
        let v = (d11 * d20 - d01 * d21) / denom;
        let w = (d00 * d21 - d01 * d20) / denom;
        let u = 1.0 - v - w;
        if u >= -1.0e-3 && v >= -1.0e-3 && w >= -1.0e-3 {
            return plane.abs();
        }
        dist_to_seg(p, a, b)
            .min(dist_to_seg(p, b, c))
            .min(dist_to_seg(p, c, a))
    }

    fn nearest(mesh: &CpuMesh, p: Vec3) -> (f32, Vec3, f32) {
        let mut best = f32::MAX;
        let mut mid = Vec3::ZERO;
        let mut panel = 0.0;
        for tri in mesh.indices.chunks(3) {
            let a = Vec3::from(mesh.vertices[tri[0] as usize].pos);
            let b = Vec3::from(mesh.vertices[tri[1] as usize].pos);
            let c = Vec3::from(mesh.vertices[tri[2] as usize].pos);
            let d = dist_to_tri(p, a, b, c);
            if d < best {
                best = d;
                mid = (a + b + c) / 3.0;
                panel = mesh.vertices[tri[0] as usize].panel;
            }
        }
        (best, mid, panel)
    }

    fn clearance(mesh: &CpuMesh, p: Vec3) -> f32 {
        nearest(mesh, p).0
    }

    #[test]
    fn cockpit_eyes_stay_out_of_the_shell() {
        let shell = shell();
        let wheel = wheel();
        let vehicle = Vehicle::new();
        let inv = vehicle.orientation().inverse();
        for index in 0..COCKPIT_VIEWS {
            let pose = cockpit_pose(&vehicle, index);
            let eye = inv * (pose.eye - vehicle.pos);
            let shell_d = clearance(&shell, eye);
            let eye_w = wheel_local(0.0).inverse().transform_point3(eye);
            let wheel_d = clearance(&wheel, eye_w);
            assert!(
                shell_d > 0.03,
                "view {index} eye {eye:?} is {shell_d:.3} from the shell"
            );
            assert!(
                wheel_d > 0.03,
                "view {index} eye is {wheel_d:.3} from the wheel"
            );
        }
    }
}
