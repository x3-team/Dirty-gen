//! Seat cameras for the buggy. C walks isometric, chase, then these views.
//!
//! The poses sit inside the tube cage: a wheel, a wet dash, muddy cowl,
//! and the two seats, from the driver seat, the passenger seat, and the
//! outside angles. The cage stays open, so the track is still in frame.
//! Chase and the fixed isometric frame stay in the cycle.

use glam::{Mat4, Quat, Vec3};

use crate::vehicle::Vehicle;
use crate::world::CameraPose;

/// How many cockpit seats and angles C walks before it returns to isometric.
pub const COCKPIT_VIEWS: u8 = 6;

/// Steering-wheel hub in the buggy frame. The wheel mesh is built around the origin.
pub const WHEEL_PIVOT: Vec3 = Vec3::new(-0.12, 0.47, 0.28);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriveCam {
    Iso,
    Chase,
    Cockpit(u8),
}

impl DriveCam {
    /// Next camera. Isometric and chase are never dropped from the cycle.
    pub fn cycle(self) -> Self {
        match self {
            DriveCam::Iso => DriveCam::Chase,
            DriveCam::Chase => DriveCam::Cockpit(0),
            DriveCam::Cockpit(i) => {
                let next = i.wrapping_add(1);
                if next < COCKPIT_VIEWS {
                    DriveCam::Cockpit(next)
                } else {
                    DriveCam::Iso
                }
            }
        }
    }

    pub fn is_cockpit(self) -> bool {
        matches!(self, DriveCam::Cockpit(_))
    }
}

struct View {
    eye: Vec3,
    target: Vec3,
    /// Vertical field of view, degrees. `perspective_rh` takes the vertical angle.
    fov_y_deg: f32,
    near: f32,
}

fn views() -> [View; COCKPIT_VIEWS as usize] {
    [
        // Driver. Above the seat backs, wheel inboard, muddy cowl ahead, the other bucket low in frame.
        View {
            eye: Vec3::new(0.00, 0.72, -0.50),
            target: Vec3::new(-0.06, 0.34, 1.12),
            fov_y_deg: 62.0,
            near: 0.05,
        },
        // Passenger, up above the bucket, looking across the wheel and the wet dash.
        View {
            eye: Vec3::new(0.30, 0.78, 0.22),
            target: Vec3::new(-0.16, 0.42, 0.90),
            fov_y_deg: 54.0,
            near: 0.05,
        },
        // Low, over the lap: the wheel, the wet dash, and the muddy cowl.
        View {
            eye: Vec3::new(0.10, 0.64, -0.34),
            target: Vec3::new(-0.02, 0.30, 1.05),
            fov_y_deg: 60.0,
            near: 0.05,
        },
        // On the muddy cowl, looking back under the roof into both seats.
        View {
            eye: Vec3::new(0.16, 0.86, 1.02),
            target: Vec3::new(-0.02, 0.46, -0.22),
            fov_y_deg: 58.0,
            near: 0.06,
        },
        // High on the left, into the open cage: seats, wheel, dash.
        View {
            eye: Vec3::new(-1.28, 1.08, 0.18),
            target: Vec3::new(0.02, 0.42, 0.02),
            fov_y_deg: 50.0,
            near: 0.08,
        },
        // Front quarter. The muddy cowl leads, then the dash and the wheel.
        View {
            eye: Vec3::new(0.55, 1.18, 2.05),
            target: Vec3::new(-0.04, 0.38, 0.20),
            fov_y_deg: 44.0,
            near: 0.08,
        },
    ]
}

/// World pose for one cockpit still. `index` wraps, so a stale key cannot miss.
pub fn cockpit_pose(vehicle: &Vehicle, index: u8) -> CameraPose {
    let view = &views()[index as usize % COCKPIT_VIEWS as usize];
    let body = Mat4::from_translation(vehicle.pos) * Mat4::from_quat(vehicle.orientation());
    CameraPose {
        eye: body.transform_point3(view.eye),
        target: body.transform_point3(view.target),
        ortho_width: None,
        fov_deg: Some(view.fov_y_deg),
        near: Some(view.near),
        cockpit: true,
    }
}

/// Column aims at the driver. Positive steer spins the wheel about that column.
pub fn wheel_local(steer: f32) -> Mat4 {
    let column = Vec3::new(0.10, 0.18, -0.72).normalize();
    let align = Quat::from_rotation_arc(Vec3::Z, column);
    let spin = Quat::from_rotation_z(-steer * 2.4);
    Mat4::from_translation(WHEEL_PIVOT) * Mat4::from_quat(align * spin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vehicle::Vehicle;

    #[test]
    fn cycle_keeps_chase_and_iso() {
        assert_eq!(DriveCam::Iso.cycle(), DriveCam::Chase);
        assert_eq!(DriveCam::Chase.cycle(), DriveCam::Cockpit(0));
        assert_eq!(DriveCam::Cockpit(COCKPIT_VIEWS - 1).cycle(), DriveCam::Iso);

        let mut cam = DriveCam::Iso;
        let mut iso = 0;
        let mut chase = 0;
        let mut cockpit = 0;
        let steps = 2 + COCKPIT_VIEWS as i32;
        for _ in 0..steps {
            match cam {
                DriveCam::Iso => iso += 1,
                DriveCam::Chase => chase += 1,
                DriveCam::Cockpit(_) => cockpit += 1,
            }
            cam = cam.cycle();
        }
        assert_eq!(iso, 1);
        assert_eq!(chase, 1);
        assert_eq!(cockpit, COCKPIT_VIEWS as i32);
        assert_eq!(cam, DriveCam::Iso);
    }

    fn local_eye_forward(index: u8) -> (Vec3, Vec3) {
        let vehicle = Vehicle::new();
        let pose = cockpit_pose(&vehicle, index);
        assert!(pose.cockpit);
        assert!(pose.ortho_width.is_none());
        assert!(pose.fov_deg.unwrap() > 20.0);
        assert!(pose.near.unwrap() < 0.2);
        let inv = vehicle.orientation().inverse();
        let eye = inv * (pose.eye - vehicle.pos);
        let forward = inv * (pose.target - pose.eye);
        (eye, forward)
    }

    #[test]
    fn driver_looks_down_the_hood() {
        let (eye, forward) = local_eye_forward(0);
        assert!(eye.z < -0.2, "driver sits behind the wheel");
        assert!(eye.y > 0.5);
        assert!(forward.z > forward.x.abs());
        assert!(forward.z > 0.0);
    }

    #[test]
    fn passenger_looks_across_the_wheel() {
        let (eye, forward) = local_eye_forward(1);
        assert!(eye.x > 0.2, "passenger side");
        assert!(forward.x < 0.0, "looking across to the wheel");
        assert!(forward.z > 0.0);
    }

    #[test]
    fn hood_looks_back_into_the_seats() {
        let (eye, forward) = local_eye_forward(3);
        assert!(eye.z > 0.8, "camera is out on the hood");
        assert!(forward.z < 0.0, "looking back");
    }

    #[test]
    fn wheel_spins_with_steer() {
        let straight = wheel_local(0.0);
        let turned = wheel_local(0.4);
        let spoke = Vec3::new(0.12, 0.0, 0.0);
        let a = straight.transform_point3(spoke);
        let b = turned.transform_point3(spoke);
        assert!((a - b).length() > 0.02);
    }
}
