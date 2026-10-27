//! One recorded lap, played back from the still cameras.
//!
//! The window stores the pedals until the buggy closes the oval once. R
//! replays that tape from a fresh start. The view walks a side elevation, a
//! top view, a bumper view, and a rear three-quarter. The track is unread
//! except to notice when the lap has closed.

use glam::Vec3;

use crate::soil::{TRACK_R, TRACK_ZN, TRACK_ZS};
use crate::vehicle::Vehicle;
use crate::world::{CameraPose, Input};

/// Cameras that match the muddy-buggy stills. Playback walks the first four
/// in order: side, top, bumper, rear three-quarter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayView {
    Side,
    Top,
    Bumper,
    RearThreeQuarter,
}

#[derive(Clone, Copy)]
struct RecordedStep {
    buggy: Input,
    bike: Input,
}

/// Tape of one lap, plus the playback cursor.
pub struct LapReplay {
    steps: Vec<RecordedStep>,
    cursor: usize,
    playing: bool,
    sealed: bool,
    progress: f32,
    last_s: Option<f32>,
}

impl Default for LapReplay {
    fn default() -> Self {
        Self::new()
    }
}

impl LapReplay {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            cursor: 0,
            playing: false,
            sealed: false,
            progress: 0.0,
            last_s: None,
        }
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn lap_ready(&self) -> bool {
        self.sealed && !self.steps.is_empty()
    }

    pub fn recorded_steps(&self) -> usize {
        self.steps.len()
    }

    /// Store the inputs just applied to the buggy and the bike.
    ///
    /// `pos` is the buggy position after that step. Recording stops once the
    /// along-track progress closes the oval, so the tape stays one lap.
    pub fn note(&mut self, buggy: Input, bike: Input, pos: Vec3) {
        if self.playing || self.sealed {
            return;
        }
        self.steps.push(RecordedStep { buggy, bike });
        let progress = self.advance(pos);
        let long_enough = self.steps.len() > 90;
        if long_enough && progress >= track_perimeter() * 0.97 {
            self.sealed = true;
        } else if self.steps.len() >= 45 * 60 {
            // A lap that never quite closes still has a tape R can play.
            self.sealed = true;
        }
    }

    /// Begin playback at the first sample. False when no lap is stored yet.
    pub fn start(&mut self) -> bool {
        if !self.lap_ready() {
            return false;
        }
        self.playing = true;
        self.cursor = 0;
        true
    }

    /// Next recorded pedals. `None` ends playback.
    pub fn take_inputs(&mut self) -> Option<(Input, Input)> {
        if !self.playing || self.cursor >= self.steps.len() {
            self.playing = false;
            return None;
        }
        let step = self.steps[self.cursor];
        self.cursor += 1;
        Some((step.buggy, step.bike))
    }

    pub fn camera(&self, vehicle: &Vehicle) -> CameraPose {
        let n = self.steps.len().max(1) as f32;
        let u = (self.cursor as f32 / n).clamp(0.0, 0.999);
        replay_camera(view_for_fraction(u), vehicle)
    }

    fn advance(&mut self, pos: Vec3) -> f32 {
        let s = closest_s(pos);
        let total = track_perimeter();
        if let Some(prev) = self.last_s {
            let mut ds = s - prev;
            if ds > total * 0.5 {
                ds -= total;
            }
            if ds < -total * 0.5 {
                ds += total;
            }
            self.progress += ds;
        }
        self.last_s = Some(s);
        self.progress
    }
}

/// Which still-camera this fraction of the tape is on.
pub fn view_for_fraction(u: f32) -> ReplayView {
    let u = u.clamp(0.0, 0.999);
    match (u * 4.0) as u32 {
        0 => ReplayView::Side,
        1 => ReplayView::Top,
        2 => ReplayView::Bumper,
        _ => ReplayView::RearThreeQuarter,
    }
}

/// World-up for a view. A straight-down top camera cannot use `+Y` as up:
/// that axis is the view direction, and the look-at basis collapses.
pub fn view_up(eye: Vec3, target: Vec3) -> Vec3 {
    let toward = target - eye;
    if toward.normalize_or_zero().dot(Vec3::Y).abs() > 0.92 {
        Vec3::Z
    } else {
        Vec3::Y
    }
}

/// Pose for one still camera. Each one follows the buggy's yaw and frames the
/// body the way the reference stills do: broadside, straight down, low on the
/// nose, and behind one rear corner.
pub fn replay_camera(view: ReplayView, vehicle: &Vehicle) -> CameraPose {
    let pos = vehicle.pos;
    let (forward, right) = yaw_basis(vehicle.yaw);
    match view {
        ReplayView::Side => CameraPose {
            eye: pos + right * 5.6 + Vec3::Y * 1.15,
            target: pos + Vec3::Y * 0.4,
            ortho_width: None,
            fov_deg: None,
            near: None,
            cockpit: false,
        },
        ReplayView::Top => CameraPose {
            eye: pos + Vec3::Y * 18.0,
            target: pos + Vec3::Y * 0.2,
            ortho_width: Some(11.0),
            fov_deg: None,
            near: None,
            cockpit: false,
        },
        ReplayView::Bumper => CameraPose {
            // Low and just ahead of the nose, looking back at the front wheels.
            eye: pos + forward * 3.8 + right * 0.4 + Vec3::Y * -0.08,
            target: pos + Vec3::Y * 0.28,
            ortho_width: None,
            fov_deg: None,
            near: None,
            cockpit: false,
        },
        ReplayView::RearThreeQuarter => CameraPose {
            eye: pos - forward * 4.4 + right * 2.5 + Vec3::Y * 1.65,
            target: pos + forward * 0.8 + Vec3::Y * 0.45,
            ortho_width: None,
            fov_deg: None,
            near: None,
            cockpit: false,
        },
    }
}

fn yaw_basis(yaw: f32) -> (Vec3, Vec3) {
    let forward = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    let forward = if forward.length_squared() < 1.0e-6 {
        Vec3::Z
    } else {
        forward.normalize()
    };
    let right = Vec3::new(forward.z, 0.0, -forward.x);
    (forward, right)
}

fn track_perimeter() -> f32 {
    let straight = TRACK_ZN - TRACK_ZS;
    let half = std::f32::consts::PI * TRACK_R;
    2.0 * straight + 2.0 * half
}

/// Clockwise from the south point. Same stadium the scripted lap already drives.
fn point_on_track(s: f32) -> Vec3 {
    let straight = TRACK_ZN - TRACK_ZS;
    let q = std::f32::consts::FRAC_PI_2 * TRACK_R;
    let half = std::f32::consts::PI * TRACK_R;
    let total = track_perimeter();
    let s = s.rem_euclid(total);
    if s < q {
        let a = -std::f32::consts::FRAC_PI_2 - (s / q) * std::f32::consts::FRAC_PI_2;
        Vec3::new(TRACK_R * a.cos(), 0.0, TRACK_ZS + TRACK_R * a.sin())
    } else if s < q + straight {
        let t = (s - q) / straight;
        Vec3::new(-TRACK_R, 0.0, TRACK_ZS + t * straight)
    } else if s < q + straight + half {
        let u = (s - q - straight) / half;
        let a = std::f32::consts::PI - u * std::f32::consts::PI;
        Vec3::new(TRACK_R * a.cos(), 0.0, TRACK_ZN + TRACK_R * a.sin())
    } else if s < q + 2.0 * straight + half {
        let t = (s - q - straight - half) / straight;
        Vec3::new(TRACK_R, 0.0, TRACK_ZN - t * straight)
    } else {
        let u = (s - q - 2.0 * straight - half) / q;
        let a = -u * std::f32::consts::FRAC_PI_2;
        Vec3::new(TRACK_R * a.cos(), 0.0, TRACK_ZS + TRACK_R * a.sin())
    }
}

fn closest_s(pos: Vec3) -> f32 {
    let total = track_perimeter();
    let n = 96;
    let mut best_s = 0.0;
    let mut best_d = f32::MAX;
    for i in 0..n {
        let s = total * (i as f32) / n as f32;
        let p = point_on_track(s);
        let d = (p.x - pos.x).powi(2) + (p.z - pos.z).powi(2);
        if d < best_d {
            best_d = d;
            best_s = s;
        }
    }
    let mut s = best_s;
    let mut step = total / n as f32;
    for _ in 0..5 {
        step *= 0.5;
        let dl = {
            let p = point_on_track(s - step);
            (p.x - pos.x).powi(2) + (p.z - pos.z).powi(2)
        };
        let dr = {
            let p = point_on_track(s + step);
            (p.x - pos.x).powi(2) + (p.z - pos.z).powi(2)
        };
        if dl < dr && dl < best_d {
            s -= step;
            best_d = dl;
        } else if dr < best_d {
            s += step;
            best_d = dr;
        }
    }
    s.rem_euclid(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::{demo_camera, demo_drive, demo_drive_bike, World};

    #[test]
    fn centerline_circuit_seals_one_lap() {
        let mut replay = LapReplay::new();
        assert!(!replay.start());
        let total = track_perimeter();
        let n = 360;
        for i in 0..=n {
            let p = point_on_track(total * (i as f32) / n as f32);
            replay.note(
                Input {
                    throttle: 1.0,
                    brake: 0.0,
                    steer: 0.0,
                },
                Input::default(),
                p,
            );
            if replay.lap_ready() {
                break;
            }
        }
        assert!(replay.lap_ready(), "progress never closed the oval");
        let count = replay.recorded_steps();
        assert!(count > 90 && count < n, "tape length {count}");
        replay.note(Input::default(), Input::default(), point_on_track(0.2));
        assert_eq!(replay.recorded_steps(), count);
        assert!(replay.start());
        assert!(replay.is_playing());
        let mut got = 0;
        while replay.take_inputs().is_some() {
            got += 1;
        }
        assert_eq!(got, count);
        assert!(replay.take_inputs().is_none());
        assert!(!replay.is_playing());
    }

    #[test]
    fn playback_walks_the_still_cameras() {
        assert_eq!(view_for_fraction(0.0), ReplayView::Side);
        assert_eq!(view_for_fraction(0.24), ReplayView::Side);
        assert_eq!(view_for_fraction(0.25), ReplayView::Top);
        assert_eq!(view_for_fraction(0.49), ReplayView::Top);
        assert_eq!(view_for_fraction(0.50), ReplayView::Bumper);
        assert_eq!(view_for_fraction(0.74), ReplayView::Bumper);
        assert_eq!(view_for_fraction(0.75), ReplayView::RearThreeQuarter);
        assert_eq!(view_for_fraction(0.99), ReplayView::RearThreeQuarter);

        let mut vehicle = Vehicle::new();
        vehicle.yaw = 0.4;
        let (forward, right) = yaw_basis(vehicle.yaw);
        let pos = vehicle.pos;

        let side = replay_camera(ReplayView::Side, &vehicle);
        let side_dir = (side.eye - pos).normalize();
        assert!(side_dir.dot(right) > 0.85, "side camera is not broadside");
        assert!(
            side_dir.dot(forward).abs() < 0.2,
            "side camera is not square"
        );
        assert!(side.ortho_width.is_none());
        assert_eq!(view_up(side.eye, side.target), Vec3::Y);

        let top = replay_camera(ReplayView::Top, &vehicle);
        let down = (top.target - top.eye).normalize();
        assert!(down.y < -0.95, "top camera is not overhead");
        let flat = Vec3::new(top.eye.x - pos.x, 0.0, top.eye.z - pos.z).length();
        assert!(flat < 0.05, "top camera is not over the buggy");
        assert_eq!(top.ortho_width, Some(11.0));
        assert_eq!(view_up(top.eye, top.target), Vec3::Z);

        let bumper = replay_camera(ReplayView::Bumper, &vehicle);
        let ahead = (bumper.eye - pos).dot(forward);
        assert!(ahead > 3.0 && ahead < 5.0, "bumper eye {ahead}");
        assert!(
            bumper.eye.y < pos.y + 0.15,
            "bumper camera should sit near the tires"
        );
        let look = (bumper.target - bumper.eye).normalize();
        assert!(
            look.dot(forward) < -0.85,
            "bumper camera should face the nose"
        );
        assert!(bumper.ortho_width.is_none());

        let rear = replay_camera(ReplayView::RearThreeQuarter, &vehicle);
        let back = (rear.eye - pos).dot(forward);
        let lateral = (rear.eye - pos).dot(right).abs();
        assert!(back < -3.0, "rear camera is not behind");
        assert!(lateral > 1.5, "rear camera is not off to the side");
        let rear_look = (rear.target - rear.eye).normalize();
        assert!(rear_look.dot(forward) > 0.5);

        let iso = demo_camera(0.0, Vec3::ZERO, 0.0);
        assert_eq!(view_up(iso.eye, iso.target), Vec3::Y);

        for view in [
            ReplayView::Side,
            ReplayView::Top,
            ReplayView::Bumper,
            ReplayView::RearThreeQuarter,
        ] {
            let pose = replay_camera(view, &vehicle);
            let up = view_up(pose.eye, pose.target);
            let basis = glam::Mat4::look_at_rh(pose.eye, pose.target, up);
            assert!(
                basis.to_cols_array().iter().all(|v| v.is_finite()),
                "{view:?} look-at is degenerate"
            );
        }
    }

    #[test]
    fn scripted_lap_replays_from_the_same_pedals() {
        let mut world = World::demo();
        let mut replay = LapReplay::new();
        let mut guard = 0;
        while !replay.lap_ready() && guard < 45 * 60 {
            let bike = demo_drive_bike(&world);
            let buggy = demo_drive(&world);
            world.step_bike(1.0 / 60.0, &bike);
            world.step(1.0 / 60.0, &buggy);
            replay.note(buggy, bike, world.vehicle.pos);
            guard += 1;
        }
        assert!(
            replay.lap_ready(),
            "scripted drive did not close a lap in {:.1}s at {:?}",
            guard as f32 / 60.0,
            world.vehicle.pos
        );
        let end = world.vehicle.pos;
        let steps = replay.recorded_steps();
        assert!(replay.start());

        let mut again = World::demo();
        let mut played = 0;
        while let Some((buggy, bike)) = replay.take_inputs() {
            again.step_bike(1.0 / 60.0, &bike);
            again.step(1.0 / 60.0, &buggy);
            played += 1;
        }
        assert_eq!(played, steps);
        let miss = (again.vehicle.pos - end).length();
        assert!(miss < 0.05, "replay drifted {miss} m from the recorded lap");
        assert!(!replay.is_playing());
    }
}
