//! Dents and one loose panel after a hit.
//!
//! This is a picture. It does not answer the contact, move the chassis, or
//! touch fuel and fire. The landing contact calls [`BodyDamage::note_contact`].

#[derive(Clone, Debug)]
pub struct BodyDamage {
    /// 0 is clean. 1 is a hard hit. Latches on the first real impact.
    pub dent: f32,
    /// Hinge angle of the torn front fender, in radians.
    pub loose: f32,
    pub loose_rate: f32,
    /// Angle the torn panel hangs at once the swing dies.
    pub hang: f32,
    pub hit: bool,
}

impl Default for BodyDamage {
    fn default() -> Self {
        Self::new()
    }
}

impl BodyDamage {
    pub fn new() -> Self {
        Self {
            dent: 0.0,
            loose: 0.0,
            loose_rate: 0.0,
            hang: 0.0,
            hit: false,
        }
    }

    /// Existing body contact. `impact_speed` is how fast the shell met the ground.
    /// A brush does not crease the panels.
    pub fn note_contact(&mut self, impact_speed: f32) {
        if self.hit || !(impact_speed >= 1.4) {
            return;
        }
        self.hit = true;
        self.dent = (impact_speed / 7.0).clamp(0.55, 1.0);
        self.hang = 0.58 + 0.16 * self.dent;
        self.loose = 0.05;
        self.loose_rate = 2.4 + self.dent * 1.3;
    }

    pub fn step(&mut self, dt: f32) {
        if !self.hit || dt <= 0.0 {
            return;
        }
        let k = 24.0;
        let damp = 4.4;
        self.loose_rate += (k * (self.hang - self.loose) - damp * self.loose_rate) * dt;
        self.loose = (self.loose + self.loose_rate * dt).clamp(-0.15, 1.45);
        if !self.loose.is_finite() || !self.loose_rate.is_finite() {
            self.loose = self.hang;
            self.loose_rate = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hard_hit_dents_and_the_panel_hangs() {
        let mut d = BodyDamage::new();
        d.note_contact(0.4);
        assert!(!d.hit);
        d.note_contact(5.5);
        assert!(d.hit);
        assert!(d.dent > 0.5);
        let kicked = d.loose;
        for _ in 0..50 {
            d.step(1.0 / 60.0);
        }
        assert!(
            d.loose > kicked + 0.25,
            "torn panel did not swing, {:.3}",
            d.loose
        );
        for _ in 0..360 {
            d.step(1.0 / 60.0);
        }
        assert!(
            (d.loose - d.hang).abs() < 0.2,
            "panel did not settle, {:.3} hang {:.3}",
            d.loose,
            d.hang
        );
        let settled = d.loose;
        d.note_contact(12.0);
        assert!((d.loose - settled).abs() < 1.0e-4);
    }
}
