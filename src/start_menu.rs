//! Start-screen hook.
//!
//! `ui/start-menu.js` writes `#silt-menu-state`:
//! `data-phase` is `menu` or `race`, `data-vehicle` is `buggy` or `bike`,
//! and `data-action` is `start` or `restart`. The action is cleared here so
//! it fires once. This does not draw the scene.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ride {
    Buggy,
    Bike,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuAction {
    None,
    Start,
    Restart,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hook {
    pub ride: Ride,
    pub action: MenuAction,
    /// While true, drive keys do not take a vehicle. The lap keeps running.
    pub holding: bool,
}

pub fn interpret(phase: &str, vehicle: &str, action: &str, screen_present: bool) -> Hook {
    let ride = if vehicle.trim().eq_ignore_ascii_case("bike") {
        Ride::Bike
    } else {
        Ride::Buggy
    };
    let action = match action.trim().to_ascii_lowercase().as_str() {
        "start" => MenuAction::Start,
        "restart" => MenuAction::Restart,
        _ => MenuAction::None,
    };
    let holding = match action {
        MenuAction::Start => false,
        MenuAction::Restart => screen_present,
        MenuAction::None => screen_present && !phase.trim().eq_ignore_ascii_case("race"),
    };
    Hook {
        ride,
        action,
        holding,
    }
}

pub fn poll() -> Option<Hook> {
    #[cfg(target_arch = "wasm32")]
    {
        return poll_dom();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        None
    }
}

#[cfg(target_arch = "wasm32")]
fn poll_dom() -> Option<Hook> {
    let window = web_sys::window()?;
    let document = window.document()?;
    let node = document.get_element_by_id("silt-menu-state")?;
    let phase = node.get_attribute("data-phase").unwrap_or_default();
    let vehicle = node.get_attribute("data-vehicle").unwrap_or_default();
    let action = node.get_attribute("data-action").unwrap_or_default();
    if !action.is_empty() {
        let _ = node.set_attribute("data-action", "");
    }
    let screen_present = document.get_element_by_id("silt-start").is_some();
    Some(interpret(&phase, &vehicle, &action, screen_present))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_hands_over_the_bike() {
        let hook = interpret("menu", "bike", "start", true);
        assert_eq!(hook.ride, Ride::Bike);
        assert_eq!(hook.action, MenuAction::Start);
        assert!(!hook.holding);
    }

    #[test]
    fn restart_returns_to_the_menu_on_the_buggy() {
        let hook = interpret("race", "buggy", "restart", true);
        assert_eq!(hook.ride, Ride::Buggy);
        assert_eq!(hook.action, MenuAction::Restart);
        assert!(hook.holding);
    }

    #[test]
    fn the_open_screen_holds_the_lap() {
        let hook = interpret("menu", " buggy ", "", true);
        assert_eq!(hook.ride, Ride::Buggy);
        assert_eq!(hook.action, MenuAction::None);
        assert!(hook.holding);
    }

    #[test]
    fn a_race_phase_lets_the_player_drive() {
        let hook = interpret("race", "bike", "", true);
        assert!(!hook.holding);
        assert_eq!(hook.ride, Ride::Bike);
    }

    #[test]
    fn without_the_screen_the_lap_stays_playable() {
        let hook = interpret("menu", "bike", "", false);
        assert!(!hook.holding);
        assert_eq!(hook.action, MenuAction::None);
    }

    #[test]
    fn an_unknown_vehicle_is_the_buggy() {
        let hook = interpret("menu", "truck", "later", true);
        assert_eq!(hook.ride, Ride::Buggy);
        assert_eq!(hook.action, MenuAction::None);
        assert!(hook.holding);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_poll_leaves_the_window_alone() {
        assert!(poll().is_none());
    }
}
