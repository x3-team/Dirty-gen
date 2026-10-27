//! Lap clock for the mud oval.
//!
//! The start line is the south gate: the radial cut across the centerline
//! where the buggy is parked. The clock starts when the watched vehicle
//! leaves that gate or crosses it, then counts one lap each time the same
//! vehicle comes around and crosses it again. V switches the watched
//! vehicle and the clock starts over.

use glam::Vec3;

use crate::soil::{track_offset, TRACK_R, TRACK_ZN, TRACK_ZS};

/// Readout before the watched vehicle has taken the south gate.
pub const STANDBY_READOUT: &str = "Старт · 0 кругов · 0:00.0";

/// Share of the centerline required between two gate crossings.
const LAP_FRACTION: f32 = 0.78;
/// A single step longer than this is a glitch, not travel along the oval.
const GLITCH_STEP: f32 = 8.0;
/// Ignore a centerline jump larger than a real step at this rate.
const MAX_ALONG_STEP: f32 = 2.5;
/// Inside this |x| the vehicle is still standing on the line.
const STRIP_X: f32 = 0.9;
/// A crossing counts once the vehicle is this far onto the new side.
const COMMIT_X: f32 = 0.45;
/// Side of the line. Closer than this is still "on" it.
const SIDE_X: f32 = 0.12;
/// Half-length of the gate, across the road, in metres.
const GATE_HALF_Z: f32 = 3.2;
/// How far off the ribbon a sample can be and still be the gate.
const MAX_OFFSET: f32 = 4.0;
/// Metres away from the parked spot before a vehicle sitting on the line
/// is treated as having taken the start.
const LEAVE_PARK: f32 = 2.0;

/// Timing for one vehicle on the oval.
#[derive(Clone, Debug)]
pub struct LapClock {
    vehicle: Option<u8>,
    prev: Option<Vec3>,
    prev_s: Option<f32>,
    /// Last side clear of the line. West is -1, east is +1.
    gate_side: Option<i8>,
    /// Side abandoned at the gate, until the cross is committed or given up.
    cross_from: Option<i8>,
    /// First sample while the vehicle was parked on the line.
    strip_anchor: Option<Vec3>,
    started: bool,
    /// Completed circuits. The lap underway is this plus one, after the start.
    laps: u32,
    lap: f32,
    last_lap: Option<f32>,
    /// Signed metres of centerline since the last start or lap. Clockwise is positive.
    along: f32,
    text: String,
}

impl Default for LapClock {
    fn default() -> Self {
        Self::new()
    }
}

impl LapClock {
    pub fn new() -> Self {
        Self {
            vehicle: None,
            prev: None,
            prev_s: None,
            gate_side: None,
            cross_from: None,
            strip_anchor: None,
            started: false,
            laps: 0,
            lap: 0.0,
            last_lap: None,
            along: 0.0,
            text: STANDBY_READOUT.to_string(),
        }
    }

    pub fn readout(&self) -> &str {
        &self.text
    }

    pub fn started(&self) -> bool {
        self.started
    }

    /// Completed laps. Zero until the vehicle has crossed the gate again
    /// after a full circuit.
    pub fn laps(&self) -> u32 {
        self.laps
    }

    /// Seconds on the lap underway. Zero before the start and just after a lap.
    pub fn lap_time(&self) -> f32 {
        self.lap
    }

    pub fn last_lap(&self) -> Option<f32> {
        self.last_lap
    }

    /// Advance the clock by one simulation step.
    ///
    /// `vehicle` is 0 for the buggy and 1 for the bike. A change starts a
    /// fresh clock so a teleport between the two machines is not a lap.
    pub fn advance(&mut self, dt: f32, vehicle: u8, pos: Vec3) {
        if !pos.x.is_finite() || !pos.z.is_finite() || !dt.is_finite() {
            return;
        }
        let dt = dt.max(0.0);
        if self.vehicle != Some(vehicle) {
            self.arm(vehicle, pos);
            return;
        }
        let Some(prev) = self.prev else {
            self.arm(vehicle, pos);
            return;
        };
        let s = closest_s(pos);
        if horiz(prev, pos) > GLITCH_STEP {
            self.prev = Some(pos);
            self.prev_s = Some(s);
            self.gate_side = side_of(pos.x).or(self.gate_side);
            self.cross_from = None;
            if self.started {
                self.lap += dt;
                self.refresh();
            }
            return;
        }

        if self.started {
            self.lap += dt;
            if let Some(ps) = self.prev_s {
                let ds = unwrap_ds(s - ps, perimeter());
                if ds.abs() <= MAX_ALONG_STEP {
                    self.along += ds;
                }
            }
        } else if self.strip_anchor.is_none() && in_strip(pos) {
            self.strip_anchor = Some(pos);
        }

        self.note_crossing(prev, pos);
        let committed = self.take_commit(pos);
        let left_parking = !self.started && self.left_parking(pos);

        if !self.started {
            if committed || left_parking {
                self.begin(pos);
            }
        } else if committed
            && self.direction_matches()
            && self.along.abs() >= perimeter() * LAP_FRACTION
        {
            self.finish();
        }

        self.prev = Some(pos);
        self.prev_s = Some(s);
        self.refresh();
    }

    fn arm(&mut self, vehicle: u8, pos: Vec3) {
        *self = Self::new();
        self.vehicle = Some(vehicle);
        self.prev = Some(pos);
        self.prev_s = Some(closest_s(pos));
        self.gate_side = side_of(pos.x);
    }

    fn begin(&mut self, pos: Vec3) {
        self.started = true;
        self.laps = 0;
        self.lap = 0.0;
        self.last_lap = None;
        self.along = 0.0;
        self.cross_from = None;
        self.strip_anchor = None;
        self.gate_side = side_of(pos.x).or(self.gate_side);
    }

    fn finish(&mut self) {
        self.laps += 1;
        self.last_lap = Some(self.lap);
        self.lap = 0.0;
        self.along = 0.0;
        self.cross_from = None;
    }

    fn left_parking(&self, pos: Vec3) -> bool {
        let Some(anchor) = self.strip_anchor else {
            return false;
        };
        horiz(anchor, pos) >= LEAVE_PARK && track_offset(pos.x, pos.z).abs() <= MAX_OFFSET
    }

    /// Clockwise progress finishes on the west side of the gate. The other
    /// way finishes going east. A stale cross from the north cap fails this.
    fn direction_matches(&self) -> bool {
        match self.gate_side {
            Some(side) => self.along * f32::from(side) < 0.0,
            None => false,
        }
    }

    fn note_crossing(&mut self, prev: Vec3, pos: Vec3) {
        let Some(side) = side_of(pos.x) else {
            return;
        };
        let on_line = on_gate(prev) && on_gate(pos);
        if !on_line {
            self.cross_from = None;
        } else if let Some(old) = self.gate_side {
            if side != old && self.cross_from.is_none() {
                self.cross_from = Some(old);
            }
        }
        self.gate_side = Some(side);
    }

    fn take_commit(&mut self, pos: Vec3) -> bool {
        if !on_gate(pos) {
            self.cross_from = None;
            return false;
        }
        let Some(from) = self.cross_from else {
            return false;
        };
        let Some(now) = self.gate_side else {
            return false;
        };
        if pos.x.abs() < COMMIT_X {
            return false;
        }
        if now == from {
            self.cross_from = None;
            return false;
        }
        self.cross_from = None;
        true
    }

    fn refresh(&mut self) {
        self.text = if !self.started {
            STANDBY_READOUT.to_string()
        } else if let Some(last) = self.last_lap {
            format!(
                "{} · последний {} · {}",
                laps_label(self.laps),
                format_time(last),
                format_time(self.lap)
            )
        } else {
            format!("{} · {}", laps_label(self.laps), format_time(self.lap))
        };
    }
}

fn in_strip(pos: Vec3) -> bool {
    on_gate(pos) && pos.x.abs() < STRIP_X
}

fn on_gate(pos: Vec3) -> bool {
    (pos.z - south_z()).abs() <= GATE_HALF_Z && track_offset(pos.x, pos.z).abs() <= MAX_OFFSET
}

fn south_z() -> f32 {
    TRACK_ZS - TRACK_R
}

fn side_of(x: f32) -> Option<i8> {
    if x > SIDE_X {
        Some(1)
    } else if x < -SIDE_X {
        Some(-1)
    } else {
        None
    }
}

fn horiz(a: Vec3, b: Vec3) -> f32 {
    let dx = a.x - b.x;
    let dz = a.z - b.z;
    (dx * dx + dz * dz).sqrt()
}

fn unwrap_ds(mut ds: f32, total: f32) -> f32 {
    if ds > total * 0.5 {
        ds -= total;
    } else if ds < -total * 0.5 {
        ds += total;
    }
    ds
}

fn perimeter() -> f32 {
    let straight = TRACK_ZN - TRACK_ZS;
    let half = std::f32::consts::PI * TRACK_R;
    2.0 * straight + 2.0 * half
}

/// Clockwise from the south point: west along the south cap, north up the
/// west straight. Same stadium as the mud loop, from the public track radii.
fn point_on_track(s: f32) -> Vec3 {
    let straight = TRACK_ZN - TRACK_ZS;
    let q = std::f32::consts::FRAC_PI_2 * TRACK_R;
    let half = std::f32::consts::PI * TRACK_R;
    let total = perimeter();
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
    let total = perimeter();
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

fn format_time(t: f32) -> String {
    let t = if t.is_finite() { t.max(0.0) } else { 0.0 };
    let tenths_total = (t * 10.0).floor() as u32;
    let tenths = tenths_total % 10;
    let secs_total = tenths_total / 10;
    let secs = secs_total % 60;
    let mins = secs_total / 60;
    format!("{mins}:{secs:02}.{tenths}")
}

fn laps_label(n: u32) -> String {
    let n10 = n % 10;
    let n100 = n % 100;
    let word = if (11..15).contains(&n100) {
        "кругов"
    } else if n10 == 1 {
        "круг"
    } else if (2..5).contains(&n10) {
        "круга"
    } else {
        "кругов"
    };
    format!("{n} {word}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{demo_drive, demo_drive_bike, Input, World};

    const DT: f32 = 1.0 / 60.0;

    fn feed(clock: &mut LapClock, vehicle: u8, pos: Vec3) {
        clock.advance(DT, vehicle, pos);
    }

    #[test]
    fn standby_until_the_gate_is_left() {
        let mut clock = LapClock::new();
        assert_eq!(clock.readout(), STANDBY_READOUT);
        let parked = Vec3::new(0.0, 0.6, -8.2);
        for i in 0..180 {
            let x = if i % 2 == 0 { 0.04 } else { -0.04 };
            feed(&mut clock, 0, Vec3::new(x, 0.6, parked.z));
        }
        assert!(!clock.started());
        assert_eq!(clock.laps(), 0);
        assert_eq!(clock.readout(), STANDBY_READOUT);
    }

    #[test]
    fn leaving_the_gate_starts_without_counting_a_lap() {
        let mut clock = LapClock::new();
        feed(&mut clock, 0, Vec3::new(0.0, 0.6, -8.2));
        for i in 1..50 {
            let x = -i as f32 * 0.08;
            feed(&mut clock, 0, Vec3::new(x, 0.6, -8.0));
        }
        assert!(clock.started(), "{}", clock.readout());
        assert_eq!(clock.laps(), 0);
        assert!(clock.last_lap().is_none());
        assert!(clock.readout().starts_with("0 кругов"));
    }

    #[test]
    fn wiggling_on_the_line_does_not_count() {
        let mut clock = LapClock::new();
        feed(&mut clock, 0, Vec3::new(0.0, 0.6, -8.2));
        for i in 1..40 {
            feed(&mut clock, 0, Vec3::new(-i as f32 * 0.1, 0.6, -7.8));
        }
        assert!(clock.started());
        for i in 0..8 {
            let x = if i % 2 == 0 { 1.2 } else { -1.2 };
            feed(&mut clock, 0, Vec3::new(x, 0.6, -7.25));
        }
        assert_eq!(clock.laps(), 0, "{}", clock.readout());
    }

    #[test]
    fn one_clockwise_circuit_counts_a_lap() {
        let mut clock = LapClock::new();
        let start = Vec3::new(0.0, 0.6, -8.2);
        feed(&mut clock, 0, start);
        let mut pos = start;
        for i in 1..40 {
            pos.x = -i as f32 * 0.1;
            pos.z = -7.9;
            feed(&mut clock, 0, pos);
        }
        assert!(clock.started());
        let mut s = closest_s(pos);
        let mut guard = 0;
        while clock.laps() < 1 && guard < 2000 {
            s += 0.2;
            let p = point_on_track(s);
            feed(&mut clock, 0, Vec3::new(p.x, 0.6, p.z));
            guard += 1;
        }
        assert_eq!(clock.laps(), 1, "steps {guard} {}", clock.readout());
        let last = clock.last_lap().expect("lap time");
        assert!(last > 1.0 && last < 12.0, "lap {last}");
        assert!(clock.readout().contains("1 круг"));
        assert!(clock.readout().contains("последний"));
        let again = guard;
        let mut guard = 0;
        while clock.laps() < 2 && guard < 2000 {
            s += 0.2;
            let p = point_on_track(s);
            feed(&mut clock, 0, Vec3::new(p.x, 0.6, p.z));
            guard += 1;
        }
        assert_eq!(clock.laps(), 2, "second circuit {}", clock.readout());
        assert!(
            clock.readout().starts_with("2 круга"),
            "{}",
            clock.readout()
        );
        assert!(guard + 5 > again / 2, "second lap ended too soon");
    }

    #[test]
    fn switching_vehicle_starts_a_fresh_clock() {
        let mut clock = LapClock::new();
        feed(&mut clock, 0, Vec3::new(0.0, 0.6, -8.2));
        for i in 1..40 {
            feed(&mut clock, 0, Vec3::new(-i as f32 * 0.1, 0.6, -7.6));
        }
        assert!(clock.started());
        feed(&mut clock, 1, Vec3::new(3.05, 0.7, -7.25));
        assert!(!clock.started());
        assert_eq!(clock.laps(), 0);
        assert_eq!(clock.readout(), STANDBY_READOUT);
    }

    #[test]
    fn bike_crossing_the_gate_starts_the_clock() {
        let mut clock = LapClock::new();
        let mut x = 3.05;
        feed(&mut clock, 1, Vec3::new(x, 0.7, -7.25));
        assert!(!clock.started());
        while x > -1.2 {
            x -= 0.08;
            feed(&mut clock, 1, Vec3::new(x, 0.7, -7.25));
        }
        assert!(clock.started(), "{}", clock.readout());
        assert_eq!(clock.laps(), 0);
    }

    #[test]
    fn time_format_and_plural() {
        assert_eq!(format_time(0.0), "0:00.0");
        assert_eq!(format_time(1.25), "0:01.2");
        assert_eq!(format_time(61.04), "1:01.0");
        assert_eq!(laps_label(0), "0 кругов");
        assert_eq!(laps_label(1), "1 круг");
        assert_eq!(laps_label(2), "2 круга");
        assert_eq!(laps_label(5), "5 кругов");
        assert_eq!(laps_label(11), "11 кругов");
        assert_eq!(laps_label(21), "21 круг");
        assert_eq!(laps_label(22), "22 круга");
    }

    #[test]
    fn settled_buggy_does_not_trip_the_gate() {
        let mut world = World::demo();
        let mut clock = LapClock::new();
        for _ in 0..60 * 3 {
            world.step(DT, &Input::default());
            clock.advance(DT, 0, world.vehicle.pos);
        }
        assert!(
            !clock.started(),
            "{} pos {:?}",
            clock.readout(),
            world.vehicle.pos
        );
    }

    #[test]
    fn scripted_buggy_completes_a_timed_lap() {
        let mut world = World::demo();
        let mut clock = LapClock::new();
        let limit = 60 * 45;
        let mut steps = 0;
        while steps < limit && clock.laps() < 1 {
            let bike = demo_drive_bike(&world);
            let buggy = demo_drive(&world);
            world.step_bike(DT, &bike);
            world.step(DT, &buggy);
            clock.advance(DT, 0, world.vehicle.pos);
            steps += 1;
        }
        eprintln!(
            "scripted lap steps {steps} t {:.2} readout {} pos {:?} started {}",
            world.time,
            clock.readout(),
            world.vehicle.pos,
            clock.started()
        );
        assert!(clock.started(), "south gate did not start the clock");
        assert!(
            clock.laps() >= 1,
            "no completed lap after {steps} steps: {}",
            clock.readout()
        );
        let last = clock.last_lap().expect("lap time");
        assert!(last > 4.0 && last < 40.0, "lap time {last}");
    }
}
