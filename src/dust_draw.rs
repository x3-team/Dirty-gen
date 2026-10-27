//! Camera-facing balls of tan dust. Each mote is a soft sphere of silt, and
//! the larger ones carry a second lobe above the first so the puff has height.
//! A wide card stretched along the horizon read as a ribbon on the ground.

use glam::Vec3;
use silt::dust::{Dust, DUST_GRIT};

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
    // Two quads per billow, under the 6144-vertex dust buffer.
    const MAX_MOTES: usize = 470;
    if order.len() > MAX_MOTES {
        let drop = order.len() - MAX_MOTES;
        order.drain(0..drop);
    }

    let mut verts = Vec::with_capacity(order.len() * 12);
    for i in order {
        let m = &dust.motes[i];
        let life_u = (m.life / m.max_life).clamp(0.0, 1.0);
        // Hold the body, then thin it. A steep early fade turned the cloud to gauze.
        let fade = life_u.powf(0.72);
        let density = (m.alpha * fade).clamp(0.0, 1.15);
        if density < 0.04 {
            continue;
        }
        let radius = m.size * 0.5;
        let wide = m.wide.clamp(0.85, 1.15);
        let half_w = radius * wide;
        let half_h = radius * (2.0 - wide);
        // The sim position is the center of the ball. Buried centers would
        // lose their dense core to the ground depth test.
        let center = Vec3::new(m.pos.x, m.pos.y.max(m.floor + radius * 0.35), m.pos.z);
        let axis_r = right * half_w;
        let axis_u = up * half_h;
        if m.kind != DUST_GRIT && m.size > 0.16 && verts.len() + 12 <= 5760 {
            let mut trail = m.vel;
            trail.y = 0.0;
            if trail.length_squared() < 1.0e-6 {
                trail = fwd;
            }
            trail = trail.normalize();
            let flank = Vec3::Y.cross(trail);
            let flip = if i % 2 == 0 { -1.0 } else { 1.0 };
            // A lobe tucked into the same puff. A ball sitting on top of
            // another ball read as a cluster of bubbles, not one billow.
            let crown = center + Vec3::Y * (m.size * 0.16) + flank * (m.size * 0.1 * flip)
                - trail * (m.size * 0.05);
            push_quad(
                &mut verts,
                crown,
                axis_r * 0.78,
                axis_u * 0.72,
                pale(m.rgb),
                density * 0.55,
            );
        }
        if verts.len() + 6 > 5760 {
            break;
        }
        push_quad(&mut verts, center, axis_r, axis_u, m.rgb, density);
    }
    verts
}

fn pale(rgb: [f32; 3]) -> [f32; 3] {
    [
        (rgb[0] + 0.05).min(0.86),
        (rgb[1] + 0.04).min(0.68),
        (rgb[2] + 0.025).min(0.46),
    ]
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
    // uv.y grows toward the bottom of the card so the shader can shade the belly.
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

#[cfg(test)]
mod tests {
    use super::*;
    use silt::dust::Dust;

    fn throw_plume() -> Dust {
        let mut dust = Dust::new();
        let dt = 1.0 / 120.0;
        for _ in 0..24 {
            dust.note_wheel(2, Vec3::new(0.0, 0.1, 0.0), Vec3::Z, Vec3::X, 3.3, 0.9, 0.0);
            dust.update(dt);
        }
        for _ in 0..28 {
            dust.note_wheel(
                2,
                Vec3::new(0.2, 0.1, 0.4),
                Vec3::Z,
                Vec3::X,
                3.3,
                0.16,
                0.0,
            );
            dust.update(dt);
        }
        dust
    }

    #[test]
    fn billboards_are_round_puffs_with_height() {
        let dust = throw_plume();
        let eye = Vec3::new(-7.0, 1.6, 0.3);
        let verts = dust_billboards(&dust, eye, Vec3::new(0.2, 0.45, 0.0));
        assert!(
            verts.len() > 48 && verts.len() % 6 == 0,
            "expected a cloud of quads, got {} verts",
            verts.len()
        );
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for quad in verts.chunks(6) {
            let p0 = Vec3::from_array(quad[0].pos);
            let p1 = Vec3::from_array(quad[1].pos);
            let p5 = Vec3::from_array(quad[5].pos);
            let w = (p1 - p0).length();
            let h = (p0 - p5).length();
            let aspect = w / h.max(1.0e-4);
            assert!(
                (0.72..1.40).contains(&aspect),
                "quad is a ribbon, aspect {aspect:.2} ({w:.2} x {h:.2})"
            );
            for v in quad {
                min_y = min_y.min(v.pos[1]);
                max_y = max_y.max(v.pos[1]);
            }
        }
        assert!(
            max_y - min_y > 0.6,
            "drawn plume is flat, height {:.2}",
            max_y - min_y
        );
    }
}
