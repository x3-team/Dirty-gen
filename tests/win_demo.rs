use silt::soil::SurfaceKind;
use silt::World;

#[test]
fn win_playground_has_sand_mud_and_pooled_water() {
    let world = World::win_demo();
    let sand = world.soil.sample(2.2, 0.0);
    let mud = world.soil.sample(-2.2, 0.0);
    let puddle = world.soil.sample(-2.35, -0.95);
    let seam = world.soil.sample(-1.05, 0.15);
    assert_eq!(sand.kind(), SurfaceKind::Sand);
    assert_eq!(mud.kind(), SurfaceKind::WetMud);
    assert!(
        puddle.water > 0.03,
        "puddle water {:.4}",
        puddle.water
    );
    assert!(
        seam.water > 0.03,
        "seam puddle water {:.4}",
        seam.water
    );
}

/// Hero wheel sits on the seam pan; contact patch must keep solver water (no shader hole).
#[test]
fn win_demo_seam_water_under_wheel_patch() {
    let (x, z) = (-1.05_f32, 0.15_f32);
    let world = World::win_demo();
    let hub = world.soil.sample(x, z);
    assert!(
        hub.water > 0.03,
        "hub water {:.4}",
        hub.water
    );
    for (dx, dz) in [(0.0_f32, 0.0), (0.18, 0.0), (-0.15, 0.12), (0.0, -0.2)] {
        let s = world.soil.sample(x + dx, z + dz);
        assert!(
            s.water > 0.015,
            "water under patch offset ({dx:.2},{dz:.2}) got {:.4}",
            s.water
        );
    }
}
