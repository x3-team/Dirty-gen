//! Small top-right map of the stadium oval. Dots track the two vehicles.

use glam::Vec3;

use crate::soil::{TRACK_R, TRACK_ZN, TRACK_ZS};
use crate::world::demo_camera;

use super::positions::{VehiclePos, Which};

pub struct MapVert {
    pub x: f32,
    pub y: f32,
    pub rgba: [f32; 4],
}

pub struct MapMesh {
    pub verts: Vec<MapVert>,
    pub indices: Vec<u32>,
}

const PANEL: [f32; 4] = [0.078, 0.055, 0.035, 0.92];
const EDGE: [f32; 4] = [0.72, 0.58, 0.40, 0.96];
const INFIELD: [f32; 4] = [0.20, 0.13, 0.08, 1.0];
const LANE: [f32; 4] = [0.55, 0.40, 0.24, 1.0];
const LINE: [f32; 4] = [0.90, 0.80, 0.62, 1.0];
const BUGGY: [f32; 4] = [0.96, 0.93, 0.86, 1.0];
const BIKE: [f32; 4] = [0.95, 0.54, 0.16, 1.0];
const RING: [f32; 4] = [0.05, 0.03, 0.02, 1.0];

/// Screen-space widget. Pixel origin is the top-left of the frame.
pub fn widget_mesh(width: f32, height: f32, vehicles: &[VehiclePos; 2]) -> MapMesh {
    let mut mesh = MapMesh {
        verts: Vec::new(),
        indices: Vec::new(),
    };
    let Some(widget) = place_widget(width, height) else {
        return mesh;
    };
    let world_line = stadium(72);
    let axes = iso_axes();
    let projected: Vec<(f32, f32)> = world_line
        .iter()
        .map(|&(x, z)| project(x, z, axes))
        .collect();
    let frame = fit_frame(&projected);
    let to_px = |u: f32, v: f32| -> (f32, f32) {
        (
            widget.inner_x + u * widget.inner,
            widget.inner_y + (1.0 - v) * widget.inner,
        )
    };
    let line: Vec<(f32, f32)> = projected
        .iter()
        .map(|&p| {
            let (u, v) = frame.uv(p);
            to_px(u, v)
        })
        .collect();

    round_rect(
        &mut mesh,
        widget.x - 1.5,
        widget.y - 1.5,
        widget.side + 3.0,
        widget.side + 3.0,
        widget.radius + 1.5,
        EDGE,
    );
    round_rect(
        &mut mesh,
        widget.x,
        widget.y,
        widget.side,
        widget.side,
        widget.radius,
        PANEL,
    );

    let ppm = widget.inner / frame.span;
    // Two cream edges with mud between them, the treatment that stayed readable
    // at thumbnail size in the corner-map stills.
    let stroke = (0.20 * ppm).max(1.35);
    let half_gap = (0.95 * ppm).max(3.4);
    let (outer_mid, inner_mid) = offset_band(&line, half_gap);
    let (_, infield) = offset_band(&inner_mid, stroke * 0.5);
    fan(&mut mesh, &infield, INFIELD);
    band(&mut mesh, &outer_mid, &inner_mid, LANE);
    let (outer_a, outer_b) = offset_band(&outer_mid, stroke * 0.5);
    let (inner_a, inner_b) = offset_band(&inner_mid, stroke * 0.5);
    band(&mut mesh, &outer_a, &outer_b, LINE);
    band(&mut mesh, &inner_a, &inner_b, LINE);

    let mut dots = Vec::new();
    for vehicle in vehicles {
        if !vehicle.x.is_finite() || !vehicle.z.is_finite() || !vehicle.yaw.is_finite() {
            continue;
        }
        let (u, v) = frame.uv(project(vehicle.x, vehicle.z, axes));
        let (u, v) = (u.clamp(0.06, 0.94), v.clamp(0.06, 0.94));
        dots.push(Dot {
            at: to_px(u, v),
            fill: if vehicle.which == Which::Buggy {
                BUGGY
            } else {
                BIKE
            },
        });
    }
    separate_dots(&mut dots, 8.0);
    let radius = (0.46 * ppm).clamp(3.4, 5.0);
    for dot in &dots {
        disc(&mut mesh, dot.at, radius + 1.6, RING);
    }
    for dot in &dots {
        disc(&mut mesh, dot.at, radius, dot.fill);
    }
    mesh
}

struct Dot {
    at: (f32, f32),
    fill: [f32; 4],
}

struct Widget {
    x: f32,
    y: f32,
    side: f32,
    radius: f32,
    inner_x: f32,
    inner_y: f32,
    inner: f32,
}

fn place_widget(width: f32, height: f32) -> Option<Widget> {
    if width < 64.0 || height < 64.0 {
        return None;
    }
    let margin = 14.0_f32.min(width * 0.03).min(height * 0.03);
    let side = 168.0_f32.min(width - margin * 2.0).min(height * 0.46);
    if side < 72.0 {
        return None;
    }
    let pad = (side * 0.10).clamp(8.0, 16.0);
    Some(Widget {
        x: width - margin - side,
        y: margin,
        side,
        radius: (side * 0.075).clamp(6.0, 10.0),
        inner_x: width - margin - side + pad,
        inner_y: margin + pad,
        inner: side - pad * 2.0,
    })
}

struct Frame {
    origin_x: f32,
    origin_y: f32,
    span: f32,
}

impl Frame {
    fn uv(&self, p: (f32, f32)) -> (f32, f32) {
        (
            (p.0 - self.origin_x) / self.span,
            (p.1 - self.origin_y) / self.span,
        )
    }
}

fn fit_frame(points: &[(f32, f32)]) -> Frame {
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for &(x, y) in points {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let cx = (min_x + max_x) * 0.5;
    let cy = (min_y + max_y) * 0.5;
    let half = (max_x - min_x).max(max_y - min_y) * 0.5 + 2.5;
    Frame {
        origin_x: cx - half,
        origin_y: cy - half,
        span: half * 2.0,
    }
}

/// Ground axes of the fixed isometric camera, so the oval matches that view.
/// Screen-right is the camera's right; screen-up is the ground part of camera up.
fn iso_axes() -> ((f32, f32), (f32, f32)) {
    let pose = demo_camera(0.0, Vec3::ZERO, 0.0);
    let forward = (pose.target - pose.eye).normalize_or_zero();
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let up = right.cross(forward);
    let up = unit(up.x, up.z);
    ((right.x, right.z), up)
}

fn project(x: f32, z: f32, axes: ((f32, f32), (f32, f32))) -> (f32, f32) {
    let (right, up) = axes;
    let cx = 0.0;
    let cz = (TRACK_ZS + TRACK_ZN) * 0.5;
    let dx = x - cx;
    let dz = z - cz;
    (dx * right.0 + dz * right.1, dx * up.0 + dz * up.1)
}

/// Stadium centerline, same segments the scripted lap drives.
fn stadium(n: usize) -> Vec<(f32, f32)> {
    let n = n.max(12);
    let straight = TRACK_ZN - TRACK_ZS;
    let q = std::f32::consts::FRAC_PI_2 * TRACK_R;
    let half = std::f32::consts::PI * TRACK_R;
    let total = 2.0 * straight + 2.0 * half;
    (0..n)
        .map(|i| {
            let s = total * (i as f32) / n as f32;
            point_on_stadium(s, straight, q, half, total)
        })
        .collect()
}

fn point_on_stadium(s: f32, straight: f32, q: f32, half: f32, total: f32) -> (f32, f32) {
    let s = s.rem_euclid(total);
    if s < q {
        let a = -std::f32::consts::FRAC_PI_2 - (s / q) * std::f32::consts::FRAC_PI_2;
        (TRACK_R * a.cos(), TRACK_ZS + TRACK_R * a.sin())
    } else if s < q + straight {
        let t = (s - q) / straight;
        (-TRACK_R, TRACK_ZS + t * straight)
    } else if s < q + straight + half {
        let u = (s - q - straight) / half;
        let a = std::f32::consts::PI - u * std::f32::consts::PI;
        (TRACK_R * a.cos(), TRACK_ZN + TRACK_R * a.sin())
    } else if s < q + 2.0 * straight + half {
        let t = (s - q - straight - half) / straight;
        (TRACK_R, TRACK_ZN - t * straight)
    } else {
        let u = (s - q - 2.0 * straight - half) / q;
        let a = -u * std::f32::consts::FRAC_PI_2;
        (TRACK_R * a.cos(), TRACK_ZS + TRACK_R * a.sin())
    }
}

fn offset_band(line: &[(f32, f32)], dist: f32) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
    let c = centroid(line);
    let n = line.len();
    let mut outer = Vec::with_capacity(n);
    let mut inner = Vec::with_capacity(n);
    for i in 0..n {
        let prev = line[(i + n - 1) % n];
        let next = line[(i + 1) % n];
        let (tx, ty) = (next.0 - prev.0, next.1 - prev.1);
        let len = (tx * tx + ty * ty).sqrt().max(1.0e-4);
        let mut nx = -ty / len;
        let mut ny = tx / len;
        let (vx, vy) = (c.0 - line[i].0, c.1 - line[i].1);
        if nx * vx + ny * vy < 0.0 {
            nx = -nx;
            ny = -ny;
        }
        inner.push((line[i].0 + nx * dist, line[i].1 + ny * dist));
        outer.push((line[i].0 - nx * dist, line[i].1 - ny * dist));
    }
    (outer, inner)
}

fn centroid(pts: &[(f32, f32)]) -> (f32, f32) {
    if pts.is_empty() {
        return (0.0, 0.0);
    }
    let (sx, sy) = pts
        .iter()
        .fold((0.0, 0.0), |(sx, sy), p| (sx + p.0, sy + p.1));
    (sx / pts.len() as f32, sy / pts.len() as f32)
}

fn separate_dots(dots: &mut [Dot], min_px: f32) {
    if dots.len() < 2 {
        return;
    }
    let dx = dots[1].at.0 - dots[0].at.0;
    let dy = dots[1].at.1 - dots[0].at.1;
    let d = (dx * dx + dy * dy).sqrt();
    if d >= min_px {
        return;
    }
    let (ux, uy) = if d < 1.0e-3 {
        (1.0, 0.0)
    } else {
        (dx / d, dy / d)
    };
    let push = (min_px - d) * 0.5;
    dots[0].at.0 -= ux * push;
    dots[0].at.1 -= uy * push;
    dots[1].at.0 += ux * push;
    dots[1].at.1 += uy * push;
}

fn unit(x: f32, y: f32) -> (f32, f32) {
    let len = (x * x + y * y).sqrt();
    if len < 1.0e-4 {
        (0.0, -1.0)
    } else {
        (x / len, y / len)
    }
}

fn push_tri(mesh: &mut MapMesh, a: MapVert, b: MapVert, c: MapVert) {
    let base = mesh.verts.len() as u32;
    mesh.verts.push(a);
    mesh.verts.push(b);
    mesh.verts.push(c);
    mesh.indices.extend([base, base + 1, base + 2]);
}

fn vert(p: (f32, f32), rgba: [f32; 4]) -> MapVert {
    MapVert {
        x: p.0,
        y: p.1,
        rgba,
    }
}

fn fan(mesh: &mut MapMesh, ring: &[(f32, f32)], color: [f32; 4]) {
    if ring.len() < 3 {
        return;
    }
    let c = centroid(ring);
    for i in 0..ring.len() {
        let j = (i + 1) % ring.len();
        push_tri(
            mesh,
            vert(c, color),
            vert(ring[i], color),
            vert(ring[j], color),
        );
    }
}

fn band(mesh: &mut MapMesh, outer: &[(f32, f32)], inner: &[(f32, f32)], color: [f32; 4]) {
    let n = outer.len().min(inner.len());
    for i in 0..n {
        let j = (i + 1) % n;
        push_tri(
            mesh,
            vert(outer[i], color),
            vert(inner[i], color),
            vert(inner[j], color),
        );
        push_tri(
            mesh,
            vert(outer[i], color),
            vert(inner[j], color),
            vert(outer[j], color),
        );
    }
}

fn disc(mesh: &mut MapMesh, c: (f32, f32), r: f32, color: [f32; 4]) {
    let segs = 14;
    for i in 0..segs {
        let a0 = (i as f32) / segs as f32 * std::f32::consts::TAU;
        let a1 = ((i + 1) as f32) / segs as f32 * std::f32::consts::TAU;
        push_tri(
            mesh,
            vert(c, color),
            vert((c.0 + a0.cos() * r, c.1 + a0.sin() * r), color),
            vert((c.0 + a1.cos() * r, c.1 + a1.sin() * r), color),
        );
    }
}

fn round_rect(mesh: &mut MapMesh, x: f32, y: f32, w: f32, h: f32, r: f32, color: [f32; 4]) {
    let r = r.min(w * 0.5).min(h * 0.5).max(0.0);
    let segs = 4;
    let corners = [
        (x + w - r, y + r, -std::f32::consts::FRAC_PI_2, 0.0),
        (x + w - r, y + h - r, 0.0, std::f32::consts::FRAC_PI_2),
        (
            x + r,
            y + h - r,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
        ),
        (
            x + r,
            y + r,
            std::f32::consts::PI,
            std::f32::consts::PI + std::f32::consts::FRAC_PI_2,
        ),
    ];
    let mut ring = Vec::with_capacity(corners.len() * (segs + 1));
    for (cx, cy, a0, a1) in corners {
        for i in 0..=segs {
            let t = i as f32 / segs as f32;
            let a = a0 + (a1 - a0) * t;
            ring.push((cx + a.cos() * r, cy + a.sin() * r));
        }
    }
    fan(mesh, &ring, color);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::{read_positions, Which};

    fn sample_world() -> crate::world::World {
        let mut world = crate::world::World::flat(0.2);
        world.vehicle.pos.x = -TRACK_R;
        world.vehicle.pos.z = (TRACK_ZS + TRACK_ZN) * 0.5;
        world.vehicle.yaw = 0.0;
        world.bike.pos.x = TRACK_R;
        world.bike.pos.z = (TRACK_ZS + TRACK_ZN) * 0.5;
        world.bike.yaw = std::f32::consts::PI;
        world
    }

    fn color_centroid(mesh: &MapMesh, color: [f32; 4]) -> Option<(f32, f32)> {
        let mut n = 0.0;
        let mut sx = 0.0;
        let mut sy = 0.0;
        for v in &mesh.verts {
            if v.rgba == color {
                sx += v.x;
                sy += v.y;
                n += 1.0;
            }
        }
        (n > 0.0).then_some((sx / n, sy / n))
    }

    #[test]
    fn corner_map_keeps_both_dots_on_the_oval() {
        let world = sample_world();
        let vehicles = read_positions(&world);
        assert_eq!(vehicles[0].which, Which::Buggy);
        assert_eq!(vehicles[1].which, Which::Bike);
        let mesh = widget_mesh(1280.0, 720.0, &vehicles);
        assert!(mesh.indices.len() >= 3);
        assert!(mesh.verts.len() <= 4096, "verts {}", mesh.verts.len());
        assert!(mesh.indices.len() <= 8192, "indices {}", mesh.indices.len());
        assert_eq!(mesh.indices.len() % 3, 0);
        for index in &mesh.indices {
            assert!((*index as usize) < mesh.verts.len());
        }
        for v in &mesh.verts {
            assert!(v.x > 1280.0 * 0.7, "map stays in the right corner");
            assert!(v.y < 720.0 * 0.45, "map stays in the top corner");
        }
        let buggy = color_centroid(&mesh, BUGGY).expect("buggy dot");
        let bike = color_centroid(&mesh, BIKE).expect("bike dot");
        let dx = buggy.0 - bike.0;
        let dy = buggy.1 - bike.1;
        assert!(
            (dx * dx + dy * dy).sqrt() > 6.0,
            "dots stay apart, got {buggy:?} {bike:?}"
        );
        // West straight is on the right in the isometric view.
        assert!(
            buggy.0 > bike.0 + 24.0,
            "west buggy should sit right of the east bike"
        );
        assert!(color_centroid(&mesh, LANE).is_some());
    }
}
