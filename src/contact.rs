//! Vehicle-to-vehicle contacts.
//!
//! The chassis stay in `vehicle` and `bike`. This solver only exchanges
//! momentum between whatever rigid bodies the world hands it: the buggy, the
//! bike, and a second buggy when one is in the list. Soil stamps and splash
//! are untouched.
//!
//! Angular velocity is stored like the chassis integrators: local
//! `(pitch rate, yaw rate, roll rate)`, mapped to the world by the YXZ
//! orientation.

use glam::{Quat, Vec3};

const ITERATIONS: usize = 16;
const SLOP: f32 = 0.015;
const SPECULATIVE: f32 = 0.03;
const RESTITUTION: f32 = 0.2;
const BOUNCE_SPEED: f32 = 1.05;
const FRICTION: f32 = 0.62;
const MAX_SHAPES: usize = 6;

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    pub local: Vec3,
    pub yaw: f32,
    pub kind: ShapeKind,
}

#[derive(Clone, Copy, Debug)]
pub enum ShapeKind {
    Sphere { radius: f32 },
    Box { half: Vec3 },
}

impl Shape {
    pub fn sphere(local: Vec3, radius: f32) -> Self {
        Self {
            local,
            yaw: 0.0,
            kind: ShapeKind::Sphere { radius },
        }
    }

    pub fn cube(local: Vec3, half: Vec3, yaw: f32) -> Self {
        Self {
            local,
            yaw,
            kind: ShapeKind::Box { half },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Body {
    pub pos: Vec3,
    pub vel: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub pitch_rate: f32,
    pub yaw_rate: f32,
    pub roll_rate: f32,
    pub mass: f32,
    /// Inverse inertia along local (pitch, yaw, roll).
    pub inv_inertia: Vec3,
    pub shapes: [Shape; MAX_SHAPES],
    pub shape_count: u8,
    /// How long the upright assist should stay out of the way after a hit.
    pub loosen: f32,
}

impl Body {
    pub fn kinematic(
        pos: Vec3,
        vel: Vec3,
        yaw: f32,
        pitch: f32,
        roll: f32,
        pitch_rate: f32,
        yaw_rate: f32,
        roll_rate: f32,
        mass: f32,
        inv_inertia: Vec3,
        loosen: f32,
    ) -> Self {
        Self {
            pos,
            vel,
            yaw,
            pitch,
            roll,
            pitch_rate,
            yaw_rate,
            roll_rate,
            mass,
            inv_inertia,
            shapes: [Shape::sphere(Vec3::ZERO, 0.0); MAX_SHAPES],
            shape_count: 0,
            loosen,
        }
    }

    pub fn push_shape(&mut self, shape: Shape) {
        if (self.shape_count as usize) < MAX_SHAPES {
            self.shapes[self.shape_count as usize] = shape;
            self.shape_count += 1;
        }
    }

    pub fn orientation(&self) -> Quat {
        Quat::from_euler(glam::EulerRot::YXZ, self.yaw, self.pitch, self.roll)
    }

    fn ang_world(&self) -> Vec3 {
        self.orientation() * Vec3::new(self.pitch_rate, self.yaw_rate, self.roll_rate)
    }

    fn vel_at(&self, point: Vec3) -> Vec3 {
        self.vel + self.ang_world().cross(point - self.pos)
    }
}

#[derive(Clone, Copy)]
struct Contact {
    normal: Vec3,
    point: Vec3,
    pen: f32,
}

/// Deepest overlap among the bodies, in metres. Zero when they are apart.
pub fn overlap(bodies: &[Body]) -> f32 {
    let mut max_pen = 0.0_f32;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            if let Some(c) = deepest(&bodies[i], &bodies[j]) {
                max_pen = max_pen.max(c.pen);
            }
        }
    }
    max_pen.max(0.0)
}

/// Sequential impulses plus positional separation. Safe to call with one body.
pub fn solve(bodies: &mut [Body]) {
    if bodies.len() < 2 {
        return;
    }
    // Velocity uses the contact from the overlapping pose. Regenerating it
    // after each shove flips the normal and cancels the torque that tips them.
    let mut found = [(
        0usize,
        0usize,
        Contact {
            normal: Vec3::Y,
            point: Vec3::ZERO,
            pen: 0.0,
        },
    ); 8];
    let mut n_found = 0;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let Some(contact) = deepest(&bodies[i], &bodies[j]) else {
                continue;
            };
            if contact.pen <= 0.0 && approaching(&bodies[i], &bodies[j], contact) >= 0.0 {
                continue;
            }
            if n_found < found.len() {
                found[n_found] = (i, j, contact);
                n_found += 1;
            }
        }
    }
    for k in 0..n_found {
        let (i, j, contact) = found[k];
        let (left, right) = bodies.split_at_mut(j);
        impulse(&mut left[i], &mut right[0], contact);
    }
    for _ in 0..ITERATIONS {
        let mut hit = false;
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                let Some(contact) = deepest(&bodies[i], &bodies[j]) else {
                    continue;
                };
                if contact.pen <= SLOP {
                    continue;
                }
                hit = true;
                let (left, right) = bodies.split_at_mut(j);
                separate(&mut left[i], &mut right[0], contact);
            }
        }
        if !hit {
            break;
        }
    }
}

fn approaching(a: &Body, b: &Body, contact: Contact) -> f32 {
    let rel = b.vel_at(contact.point) - a.vel_at(contact.point);
    rel.dot(contact.normal)
}

fn separate(a: &mut Body, b: &mut Body, contact: Contact) {
    let depth = contact.pen - SLOP;
    if depth <= 0.0 || !contact.normal.is_finite() {
        return;
    }
    let inv_a = 1.0 / a.mass;
    let inv_b = 1.0 / b.mass;
    let sum = inv_a + inv_b;
    if sum < 1.0e-8 {
        return;
    }
    let corr = depth.min(1.0) * 0.9;
    let n = contact.normal;
    a.pos -= n * corr * (inv_a / sum);
    b.pos += n * corr * (inv_b / sum);
}

fn impulse(a: &mut Body, b: &mut Body, contact: Contact) {
    let n = contact.normal;
    if !n.is_finite() || a.mass < 1.0 || b.mass < 1.0 {
        return;
    }
    let rel = b.vel_at(contact.point) - a.vel_at(contact.point);
    let vn = rel.dot(n);
    let approach = (-vn).max(0.0);
    let denom = (response(a, contact.point, n) + response(b, contact.point, n)).max(1.0e-6);
    let mut jn = 0.0;
    if vn < 0.0 {
        let e = if approach > BOUNCE_SPEED {
            RESTITUTION
        } else {
            0.0
        };
        jn = ((1.0 + e) * approach / denom).clamp(0.0, 8_000.0);
        apply_impulse(a, contact.point, -n * jn);
        apply_impulse(b, contact.point, n * jn);
    }

    // Incoming slide only. The normal impulse already coupled the offset into
    // spin; friction against the post-impact point velocity would unwind it.
    let vt = rel - n * rel.dot(n);
    let slip = vt.length();
    if slip > 0.04 {
        let t = vt / slip;
        let denom_t = (response(a, contact.point, t) + response(b, contact.point, t)).max(1.0e-6);
        let stop = slip / denom_t;
        // Muddy panels slip. Coulomb friction rides on the normal hit, so a
        // head-on still shoves while a deep overlap does not glue them together.
        let jt = stop.min(FRICTION * jn);
        if jt > 0.0 {
            apply_impulse(a, contact.point, t * jt);
            apply_impulse(b, contact.point, -t * jt);
        }
    }

    if approach > 0.9 || jn > 140.0 {
        let hold = (0.42 + approach * 0.08).clamp(0.42, 1.1);
        a.loosen = a.loosen.max(hold);
        b.loosen = b.loosen.max(hold);
    }
}

fn apply_impulse(body: &mut Body, point: Vec3, impulse: Vec3) {
    if !impulse.is_finite() {
        return;
    }
    body.vel += impulse / body.mass;
    let r = point - body.pos;
    let torque_local = body.orientation().inverse() * r.cross(impulse);
    body.pitch_rate += torque_local.x * body.inv_inertia.x;
    body.yaw_rate += torque_local.y * body.inv_inertia.y;
    body.roll_rate += torque_local.z * body.inv_inertia.z;
    body.pitch_rate = body.pitch_rate.clamp(-12.0, 12.0);
    body.yaw_rate = body.yaw_rate.clamp(-12.0, 12.0);
    body.roll_rate = body.roll_rate.clamp(-12.0, 12.0);
    if body.vel.length_squared() > 40.0 * 40.0 {
        body.vel = body.vel.normalize() * 40.0;
    }
}

/// Scalar point-velocity along `dir` gained from a unit impulse along `dir`.
fn response(body: &Body, point: Vec3, dir: Vec3) -> f32 {
    let r = point - body.pos;
    let local = body.orientation().inverse() * r.cross(dir);
    let dw_local = Vec3::new(
        local.x * body.inv_inertia.x,
        local.y * body.inv_inertia.y,
        local.z * body.inv_inertia.z,
    );
    let dw = body.orientation() * dw_local;
    let dv = dir / body.mass + dw.cross(r);
    dv.dot(dir).max(0.0)
}

fn deepest(a: &Body, b: &Body) -> Option<Contact> {
    let mut best: Option<Contact> = None;
    for i in 0..a.shape_count as usize {
        for j in 0..b.shape_count as usize {
            let Some(c) = shape_contact(a, &a.shapes[i], b, &b.shapes[j]) else {
                continue;
            };
            if best.as_ref().is_none_or(|prev| c.pen > prev.pen) {
                best = Some(c);
            }
        }
    }
    best
}

fn shape_contact(a: &Body, sa: &Shape, b: &Body, sb: &Shape) -> Option<Contact> {
    match (sa.kind, sb.kind) {
        (ShapeKind::Sphere { radius: ra }, ShapeKind::Sphere { radius: rb }) => {
            sphere_sphere(a, sa, ra, b, sb, rb)
        }
        (ShapeKind::Sphere { radius }, ShapeKind::Box { .. }) => {
            sphere_box(b, sb, a, sa, radius).map(flip)
        }
        (ShapeKind::Box { .. }, ShapeKind::Sphere { radius }) => sphere_box(a, sa, b, sb, radius),
        (ShapeKind::Box { half: ha }, ShapeKind::Box { half: hb }) => box_box(a, sa, ha, b, sb, hb),
    }
}

fn flip(c: Contact) -> Contact {
    Contact {
        normal: -c.normal,
        point: c.point,
        pen: c.pen,
    }
}

fn sphere_sphere(a: &Body, sa: &Shape, ra: f32, b: &Body, sb: &Shape, rb: f32) -> Option<Contact> {
    let ca = shape_center(a, sa);
    let cb = shape_center(b, sb);
    let delta = cb - ca;
    let dist = delta.length();
    let pen = ra + rb - dist;
    if pen < -SPECULATIVE {
        return None;
    }
    let normal = if dist > 1.0e-5 { delta / dist } else { Vec3::Z };
    Some(Contact {
        normal,
        point: ca + normal * ra,
        pen,
    })
}

fn sphere_box(
    box_body: &Body,
    box_shape: &Shape,
    sphere_body: &Body,
    sphere_shape: &Shape,
    radius: f32,
) -> Option<Contact> {
    let ShapeKind::Box { half } = box_shape.kind else {
        return None;
    };
    let obb = obb(box_body, box_shape, half);
    let center = shape_center(sphere_body, sphere_shape);
    let local = obb.inverse_rotate(center - obb.center);
    let clamped = Vec3::new(
        local.x.clamp(-half.x, half.x),
        local.y.clamp(-half.y, half.y),
        local.z.clamp(-half.z, half.z),
    );
    let delta = local - clamped;
    let dist2 = delta.length_squared();
    if dist2 > 1.0e-8 {
        let dist = dist2.sqrt();
        let pen = radius - dist;
        if pen < -SPECULATIVE {
            return None;
        }
        let normal = obb.rotate(delta / dist);
        return Some(Contact {
            normal,
            point: obb.center + obb.rotate(clamped),
            pen,
        });
    }
    let (axis, sign, face) = nearest_face(local, half);
    let normal = obb.axes[axis] * sign;
    Some(Contact {
        normal,
        point: obb.center + obb.rotate(clamped),
        pen: face + radius,
    })
}

fn box_box(a: &Body, sa: &Shape, ha: Vec3, b: &Body, sb: &Shape, hb: Vec3) -> Option<Contact> {
    let oa = obb(a, sa, ha);
    let ob = obb(b, sb, hb);
    let t = ob.center - oa.center;
    let mut best_pen = f32::MAX;
    let mut best_n = Vec3::Y;
    let mut axes = [Vec3::ZERO; 15];
    let mut n_axes = 0;
    for i in 0..3 {
        axes[n_axes] = oa.axes[i];
        n_axes += 1;
        axes[n_axes] = ob.axes[i];
        n_axes += 1;
    }
    for i in 0..3 {
        for j in 0..3 {
            let cross = oa.axes[i].cross(ob.axes[j]);
            if cross.length_squared() > 1.0e-8 {
                axes[n_axes] = cross;
                n_axes += 1;
            }
        }
    }
    for axis in axes.iter().take(n_axes) {
        let mut axis = *axis;
        let len2 = axis.length_squared();
        if len2 < 1.0e-8 {
            continue;
        }
        axis /= len2.sqrt();
        let signed = t.dot(axis);
        if signed < 0.0 {
            axis = -axis;
        }
        let dist = signed.abs();
        let pen = project(&oa, axis) + project(&ob, axis) - dist;
        if pen < 0.0 {
            return None;
        }
        if pen < best_pen {
            best_pen = pen;
            best_n = axis;
        }
    }
    if best_pen > 1.0e6 || best_pen < -SPECULATIVE {
        return None;
    }
    if best_n.dot(t) < 0.0 {
        best_n = -best_n;
    }
    Some(Contact {
        normal: best_n,
        point: (oa.center + ob.center) * 0.5,
        pen: best_pen,
    })
}

struct Obb {
    center: Vec3,
    axes: [Vec3; 3],
    half: Vec3,
}

impl Obb {
    fn rotate(&self, local: Vec3) -> Vec3 {
        self.axes[0] * local.x + self.axes[1] * local.y + self.axes[2] * local.z
    }

    fn inverse_rotate(&self, world: Vec3) -> Vec3 {
        Vec3::new(
            world.dot(self.axes[0]),
            world.dot(self.axes[1]),
            world.dot(self.axes[2]),
        )
    }
}

fn obb(body: &Body, shape: &Shape, half: Vec3) -> Obb {
    let rot = body.orientation() * Quat::from_rotation_y(shape.yaw);
    Obb {
        center: body.pos + body.orientation() * shape.local,
        half,
        axes: [rot * Vec3::X, rot * Vec3::Y, rot * Vec3::Z],
    }
}

fn project(obb: &Obb, axis: Vec3) -> f32 {
    obb.half.x * obb.axes[0].dot(axis).abs()
        + obb.half.y * obb.axes[1].dot(axis).abs()
        + obb.half.z * obb.axes[2].dot(axis).abs()
}

fn shape_center(body: &Body, shape: &Shape) -> Vec3 {
    body.pos + body.orientation() * shape.local
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_body(pos: Vec3, vel: Vec3, mass: f32, half: Vec3) -> Body {
        let mut body = Body::kinematic(
            pos,
            vel,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            mass,
            Vec3::new(1.0 / 480.0, 1.0 / 720.0, 1.0 / 260.0),
            0.0,
        );
        body.push_shape(Shape::cube(Vec3::ZERO, half, 0.0));
        body
    }

    fn momentum(bodies: &[Body]) -> Vec3 {
        bodies
            .iter()
            .fold(Vec3::ZERO, |acc, b| acc + b.vel * b.mass)
    }

    fn coast(bodies: &mut [Body], frames: u32) {
        let dt = 1.0 / 60.0;
        for _ in 0..frames {
            for body in bodies.iter_mut() {
                body.pos += body.vel * dt;
            }
            solve(bodies);
        }
    }

    #[test]
    fn separated_boxes_do_not_overlap() {
        let a = box_body(Vec3::ZERO, Vec3::ZERO, 640.0, Vec3::splat(0.5));
        let b = box_body(
            Vec3::new(0.0, 0.0, 3.0),
            Vec3::ZERO,
            640.0,
            Vec3::splat(0.5),
        );
        assert!(overlap(&[a, b]) < 1.0e-4);
    }

    #[test]
    fn resting_overlap_is_pushed_apart() {
        let mut a = box_body(Vec3::ZERO, Vec3::ZERO, 640.0, Vec3::new(0.5, 0.4, 0.8));
        let mut b = box_body(
            Vec3::new(0.0, 0.0, 1.2),
            Vec3::ZERO,
            155.0,
            Vec3::new(0.3, 0.4, 0.6),
        );
        let before = overlap(&[a, b]);
        let gap0 = (b.pos - a.pos).length();
        assert!(before > 0.2, "expected a planted overlap, got {before}");
        let mut bodies = [a, b];
        solve(&mut bodies);
        a = bodies[0];
        b = bodies[1];
        let after = overlap(&[a, b]);
        assert!(
            after < 0.03,
            "overlap left sitting at {after:.3} (was {before:.3})"
        );
        let gap = (b.pos - a.pos).length();
        assert!(gap > gap0 + 0.1, "bodies did not separate");
    }

    #[test]
    fn head_on_hit_exchanges_momentum() {
        let a = box_body(
            Vec3::new(0.0, 0.0, -1.3),
            Vec3::new(0.0, 0.0, 6.0),
            640.0,
            Vec3::new(0.4, 0.4, 0.6),
        );
        let b = box_body(
            Vec3::new(0.0, 0.0, 1.3),
            Vec3::new(0.0, 0.0, -2.0),
            640.0,
            Vec3::new(0.4, 0.4, 0.6),
        );
        let mut bodies = [a, b];
        let before = momentum(&bodies);
        coast(&mut bodies, 40);
        let after = momentum(&bodies);
        let drift = (after - before).length();
        assert!(drift < 8.0, "momentum drifted by {drift:.2}");
        assert!(
            bodies[0].vel.z < 3.0,
            "chaser kept its speed, vz {}",
            bodies[0].vel.z
        );
        assert!(
            bodies[1].vel.z > 1.0,
            "target was not pushed, vz {}",
            bodies[1].vel.z
        );
        assert!(overlap(&bodies) < 0.03, "pen {}", overlap(&bodies));
    }

    #[test]
    fn offset_hit_spins_and_a_high_hit_tips() {
        let mut chase = box_body(
            Vec3::new(0.0, 0.0, -1.6),
            Vec3::new(0.0, 0.0, 7.0),
            640.0,
            Vec3::new(0.45, 0.45, 0.7),
        );
        let mut target = box_body(
            Vec3::new(0.55, 0.0, 0.4),
            Vec3::ZERO,
            640.0,
            Vec3::new(0.45, 0.45, 0.7),
        );
        let mut pair = [chase, target];
        coast(&mut pair, 30);
        chase = pair[0];
        target = pair[1];
        let spun = chase.yaw_rate.abs().max(target.yaw_rate.abs());
        assert!(spun > 0.25, "offset hit did not spin, yaw rate {spun:.3}");

        let mut base = box_body(
            Vec3::new(0.0, 0.8, 0.0),
            Vec3::ZERO,
            640.0,
            Vec3::new(0.5, 0.55, 1.2),
        );
        let mut dropped = box_body(
            Vec3::new(0.45, 1.55, 0.1),
            Vec3::new(0.0, -4.0, 0.2),
            155.0,
            Vec3::new(0.28, 0.35, 0.55),
        );
        dropped.inv_inertia = Vec3::new(1.0 / 18.0, 1.0 / 22.0, 1.0 / 6.5);
        let mut stack = [base, dropped];
        let mut tipped = 0.0_f32;
        let dt = 1.0 / 60.0;
        for _ in 0..25 {
            for body in stack.iter_mut() {
                body.pos += body.vel * dt;
            }
            solve(&mut stack);
            tipped = tipped
                .max(stack[0].roll_rate.abs())
                .max(stack[0].pitch_rate.abs());
        }
        base = stack[0];
        assert!(
            tipped > 0.2,
            "high hit did not tip the chassis, peak {tipped:.3} roll {} pitch {}",
            base.roll_rate,
            base.pitch_rate
        );
        assert!(
            overlap(&stack) < 0.04,
            "stacked overlap {}",
            overlap(&stack)
        );
    }

    #[test]
    fn two_buggies_and_a_bike_all_take_part() {
        use crate::bike::Bike;
        use crate::vehicle::Vehicle;

        let mut chase = Vehicle::new();
        let mut lead = Vehicle::new();
        let mut bike = Bike::new();
        chase.pos = Vec3::new(0.0, 0.8, -2.2);
        chase.yaw = 0.0;
        chase.vel = Vec3::new(0.0, 0.0, 8.0);
        lead.pos = Vec3::new(0.15, 0.8, 1.1);
        lead.yaw = 0.2;
        lead.vel = Vec3::ZERO;
        bike.pos = Vec3::new(0.05, 0.8, 2.55);
        bike.yaw = 0.0;
        bike.vel = Vec3::ZERO;
        let mut bodies = [
            chase.contact_body(),
            lead.contact_body(),
            bike.contact_body(),
        ];
        let before = momentum(&bodies);
        coast(&mut bodies, 50);
        let after = momentum(&bodies);
        assert!(
            (after - before).length() < 40.0,
            "three-body momentum drifted {}",
            (after - before).length()
        );
        assert!(
            bodies[1].vel.length() > 0.8,
            "second buggy was not hit, vel {:?}",
            bodies[1].vel
        );
        assert!(
            bodies[2].vel.length() > 0.4,
            "bike was not hit, vel {:?}",
            bodies[2].vel
        );
        assert!(bodies[0].vel.z < 7.0, "chaser did not slow");
        assert!(
            overlap(&bodies) < 0.04,
            "left overlapping {}",
            overlap(&bodies)
        );
    }
}

fn nearest_face(local: Vec3, half: Vec3) -> (usize, f32, f32) {
    let mut best_i = 0;
    let mut best_sign = 1.0;
    let mut best = f32::MAX;
    let coords = [local.x, local.y, local.z];
    let extents = [half.x, half.y, half.z];
    for i in 0..3 {
        let pos = extents[i] - coords[i];
        let neg = extents[i] + coords[i];
        if pos < best {
            best = pos;
            best_i = i;
            best_sign = 1.0;
        }
        if neg < best {
            best = neg;
            best_i = i;
            best_sign = -1.0;
        }
    }
    (best_i, best_sign, best)
}
