//! Camera-facing puffs for the tan dust. Wide and low, so a side view is a
//! trail and a rear view is a fan. The mud clods stay on their own mesh.

use glam::Vec3;
use silt::dust::Dust;

use crate::mesh::ParticleVertex;

pub fn dust_billboards(dust: &Dust, eye: Vec3, target: Vec3) -> Vec<ParticleVertex> {
    let mut fwd = target - eye;
    if fwd.length_squared() < 1.0e-8 {
        fwd = Vec3::Z;
    }
    fwd = fwd.normalize();
    let mut right = fwd.cross(Vec3::Y);
    if right.length_squared() < 1.0e-8 {
        right = Vec3::X;
    }
    right = right.normalize();
    let up = right.cross(fwd).normalize();

    let mut order: Vec<usize> = dust
        .motes
        .iter()
        .enumerate()
        .filter(|(_, m)| m.alive && m.life > 0.0)
        .map(|(i, _)| i)
        .collect();
    order.sort_by(|&a, &b| {
        let da = (dust.motes[a].pos - eye).length_squared();
        let db = (dust.motes[b].pos - eye).length_squared();
        db.total_cmp(&da)
    });
    const MAX_QUADS: usize = 900;
    if order.len() > MAX_QUADS {
        let drop = order.len() - MAX_QUADS;
        order.drain(0..drop);
    }

    let mut verts = Vec::with_capacity(order.len() * 6);
    for i in order {
        let m = &dust.motes[i];
        let life_u = (m.life / m.max_life).clamp(0.0, 1.0);
        let age = 1.0 - life_u;
        let fade = life_u * life_u.sqrt();
        let alpha = (m.alpha * fade).clamp(0.0, 0.85);
        if alpha < 0.02 {
            continue;
        }
        let half_w = m.size * m.wide * 0.48;
        let half_h = m.size * (0.62 + 0.20 * age) * 0.5;
        // Sit the puff on the tread so the side view is a cloud, not a stain.
        let center = Vec3::new(m.pos.x, m.pos.y + half_h * 0.70, m.pos.z);
        push_quad(
            &mut verts,
            center,
            right * half_w,
            up * half_h,
            m.rgb,
            alpha,
        );
    }
    verts
}

fn push_quad(
    out: &mut Vec<ParticleVertex>,
    center: Vec3,
    axis_r: Vec3,
    axis_u: Vec3,
    rgb: [f32; 3],
    alpha: f32,
) {
    let tl = center - axis_r + axis_u;
    let tr = center + axis_r + axis_u;
    let br = center + axis_r - axis_u;
    let bl = center - axis_r - axis_u;
    let color = [rgb[0], rgb[1], rgb[2], alpha];
    let v = |p: Vec3, uv: [f32; 2]| ParticleVertex {
        pos: p.to_array(),
        color,
        uv,
    };
    // uv.y grows toward the ground so the shader can warm the base.
    for (p, uv) in [
        (tl, [0.0, 0.0]),
        (tr, [1.0, 0.0]),
        (br, [1.0, 1.0]),
        (tl, [0.0, 0.0]),
        (br, [1.0, 1.0]),
        (bl, [0.0, 1.0]),
    ] {
        out.push(v(p, uv));
    }
}
