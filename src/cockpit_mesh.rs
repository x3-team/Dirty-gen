//! Wheel, wet dash, and muddy hood for the cockpit cameras.
//!
//! Drawn only while a cockpit view is active. The buggy and bike meshes
//! owned by the look pass are not edited here.

use glam::Vec3;

use silt::vehicle::{PANEL_BELLY, PANEL_CAGE, PANEL_NOSE, PANEL_SEAT};

use crate::mesh::{CpuMesh, MeshVertex, PANEL_CLOD};

pub fn shell() -> CpuMesh {
    let mut mesh = CpuMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    add_dash(&mut mesh);
    add_hood(&mut mesh);
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
    add_torus(&mut mesh, 0.145, 0.018, PANEL_SEAT as f32);
    let hub = PANEL_CAGE as f32;
    add_tube(
        &mut mesh,
        Vec3::new(0.0, 0.0, -0.02),
        Vec3::new(0.0, 0.0, 0.025),
        0.032,
        hub,
        8,
    );
    // Three spokes, one of them down, the way the stills read.
    for ang in [-std::f32::consts::FRAC_PI_2, 0.524, 2.618] {
        let dir = Vec3::new(ang.cos(), ang.sin(), 0.0);
        add_tube(&mut mesh, dir * 0.034, dir * 0.128, 0.012, hub, 5);
    }
    mesh
}

fn add_dash(mesh: &mut CpuMesh) {
    let dark = PANEL_BELLY as f32;
    add_box(
        mesh,
        Vec3::new(0.0, 0.30, 0.58),
        Vec3::new(0.38, 0.11, 0.10),
        dark,
    );
    // Water beads on the dash. The clod panel is the wet dark coat.
    let beads = [
        (-0.22, 0.52),
        (-0.08, 0.64),
        (0.06, 0.50),
        (0.18, 0.60),
        (0.28, 0.54),
        (-0.16, 0.58),
        (0.02, 0.56),
        (0.12, 0.66),
        (-0.28, 0.62),
        (0.22, 0.51),
    ];
    for (x, z) in beads {
        add_box(
            mesh,
            Vec3::new(x, 0.422, z),
            Vec3::new(0.012, 0.008, 0.010),
            PANEL_CLOD,
        );
    }
    // Beads on the face toward the driver, so the dash reads wet from the seat.
    let face = [
        (-0.22, 0.34),
        (-0.08, 0.28),
        (0.06, 0.36),
        (0.18, 0.30),
        (0.28, 0.34),
    ];
    for (x, y) in face {
        add_box(
            mesh,
            Vec3::new(x, y, 0.47),
            Vec3::new(0.014, 0.010, 0.008),
            PANEL_CLOD,
        );
    }
}

fn add_hood(mesh: &mut CpuMesh) {
    let rear_y = 0.50;
    let front_y = 0.28;
    let z0 = 0.70;
    let z1 = 1.55;
    let xl = -0.40;
    let xr = 0.40;
    let paint = PANEL_NOSE as f32;
    let top = [
        Vec3::new(xl, rear_y, z0),
        Vec3::new(xl, front_y, z1),
        Vec3::new(xr, front_y, z1),
        Vec3::new(xr, rear_y, z0),
    ];
    // Mud sheet sits a centimetre off the paint so the two don't flicker.
    let lift = Vec3::new(0.0, 0.016, 0.0);
    let mud: [Vec3; 4] = std::array::from_fn(|i| top[i] + lift);
    // The face the driver looks along is the mud sheet. Paint stays on the edges.
    add_quad(mesh, mud, PANEL_CLOD);
    let thick = 0.04;
    let bottom = [
        top[0] - Vec3::Y * thick,
        top[3] - Vec3::Y * thick,
        top[2] - Vec3::Y * thick,
        top[1] - Vec3::Y * thick,
    ];
    add_quad(mesh, bottom, paint);
    add_quad(
        mesh,
        [
            top[0],
            top[0] - Vec3::Y * thick,
            top[1] - Vec3::Y * thick,
            top[1],
        ],
        paint,
    );
    add_quad(
        mesh,
        [
            top[3],
            top[2],
            top[2] - Vec3::Y * thick,
            top[3] - Vec3::Y * thick,
        ],
        paint,
    );
    add_quad(
        mesh,
        [
            top[1],
            top[1] - Vec3::Y * thick,
            top[2] - Vec3::Y * thick,
            top[2],
        ],
        paint,
    );
    add_quad(
        mesh,
        [
            top[3],
            top[3] - Vec3::Y * thick,
            top[0] - Vec3::Y * thick,
            top[0],
        ],
        paint,
    );

    // Clods along the hood, heavier toward the nose.
    let clumps = [
        (-0.16, 0.92, 0.05),
        (0.10, 1.05, 0.06),
        (0.22, 1.28, 0.07),
        (-0.22, 1.22, 0.055),
        (0.0, 1.40, 0.08),
        (-0.08, 1.48, 0.06),
        (0.16, 0.84, 0.045),
        (-0.28, 1.08, 0.05),
    ];
    for (x, z, s) in clumps {
        let y = hood_y(z) + 0.02 + s;
        add_box(
            mesh,
            Vec3::new(x, y, z),
            Vec3::new(s, s * 0.55, s * 0.8),
            PANEL_CLOD,
        );
    }
}

fn hood_y(z: f32) -> f32 {
    let t = ((z - 0.70) / (1.55 - 0.70)).clamp(0.0, 1.0);
    0.50 + (0.28 - 0.50) * t
}

fn add_seats(mesh: &mut CpuMesh) {
    let seat = PANEL_SEAT as f32;
    for x in [-0.16_f32, 0.16] {
        add_box(
            mesh,
            Vec3::new(x, 0.18, -0.02),
            Vec3::new(0.13, 0.055, 0.17),
            seat,
        );
        add_box(
            mesh,
            Vec3::new(x, 0.40, -0.20),
            Vec3::new(0.13, 0.18, 0.045),
            seat,
        );
        let side = x.signum();
        add_box(
            mesh,
            Vec3::new(x + side * 0.11, 0.28, -0.04),
            Vec3::new(0.028, 0.08, 0.15),
            seat,
        );
    }
    add_box(
        mesh,
        Vec3::new(0.0, 0.24, 0.02),
        Vec3::new(0.045, 0.05, 0.12),
        PANEL_BELLY as f32,
    );
}

fn add_column(mesh: &mut CpuMesh) {
    add_tube(
        mesh,
        Vec3::new(-0.12, 0.47, 0.28),
        Vec3::new(-0.06, 0.40, 0.50),
        0.016,
        PANEL_CAGE as f32,
        6,
    );
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
            uv: [p.x * 1.6 + p.z * 0.2, p.y * 1.35 + 0.55],
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

fn add_torus(mesh: &mut CpuMesh, radius: f32, tube: f32, panel: f32) {
    let major = 16;
    let minor = 6;
    for i in 0..major {
        let a0 = i as f32 / major as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / major as f32 * std::f32::consts::TAU;
        for j in 0..minor {
            let b0 = j as f32 / minor as f32 * std::f32::consts::TAU;
            let b1 = (j + 1) as f32 / minor as f32 * std::f32::consts::TAU;
            let p = |a: f32, b: f32| {
                let r = radius + tube * b.cos();
                Vec3::new(r * a.cos(), r * a.sin(), tube * b.sin())
            };
            add_quad(mesh, [p(a0, b0), p(a1, b0), p(a1, b1), p(a0, b1)], panel);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cockpit_furniture_is_closed_triangles() {
        let shell = shell();
        let wheel = wheel();
        assert!(shell.indices.len() > 36);
        assert!(wheel.indices.len() > 36);
        assert_eq!(shell.indices.len() % 3, 0);
        assert_eq!(wheel.indices.len() % 3, 0);
        assert!(shell
            .vertices
            .iter()
            .any(|v| (v.panel - PANEL_CLOD).abs() < 0.1));
        assert!(shell
            .vertices
            .iter()
            .any(|v| (v.panel - PANEL_BELLY as f32).abs() < 0.1));
    }
}
