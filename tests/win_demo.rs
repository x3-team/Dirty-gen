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
