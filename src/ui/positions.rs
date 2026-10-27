//! Reads where the two machines are. The corner map is the only caller.

use crate::world::World;

/// Which machine a map dot stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Which {
    Buggy,
    Bike,
}

/// Plan-view pose. `x` and `z` are world metres, `yaw` is the chassis heading.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehiclePos {
    pub which: Which,
    pub x: f32,
    pub z: f32,
    pub yaw: f32,
}

/// Both vehicles, buggy first. Positions are the chassis, not a wheel.
pub fn read_positions(world: &World) -> [VehiclePos; 2] {
    [
        VehiclePos {
            which: Which::Buggy,
            x: world.vehicle.pos.x,
            z: world.vehicle.pos.z,
            yaw: world.vehicle.yaw,
        },
        VehiclePos {
            which: Which::Bike,
            x: world.bike.pos.x,
            z: world.bike.pos.z,
            yaw: world.bike.yaw,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_buggy_then_bike() {
        let mut world = World::flat(0.2);
        world.vehicle.pos.x = -6.35;
        world.vehicle.pos.z = 1.1;
        world.vehicle.yaw = 0.4;
        world.bike.pos.x = 6.1;
        world.bike.pos.z = -2.0;
        world.bike.yaw = -1.2;
        let [buggy, bike] = read_positions(&world);
        assert_eq!(buggy.which, Which::Buggy);
        assert_eq!(bike.which, Which::Bike);
        assert!((buggy.x + 6.35).abs() < 1.0e-5);
        assert!((buggy.z - 1.1).abs() < 1.0e-5);
        assert!((buggy.yaw - 0.4).abs() < 1.0e-5);
        assert!((bike.x - 6.1).abs() < 1.0e-5);
        assert!((bike.z + 2.0).abs() < 1.0e-5);
        assert!((bike.yaw + 1.2).abs() < 1.0e-5);
    }
}
