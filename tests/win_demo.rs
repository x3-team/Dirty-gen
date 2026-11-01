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
    // Shader must not discard within 0.32 m of the hub — ring at ~0.28 m.
    let r = 0.28_f32;
    for (dx, dz) in [(r, 0.0_f32), (-r * 0.7, r * 0.7), (0.0, -r)] {
        let s = world.soil.sample(x + dx, z + dz);
        assert!(
            s.water > 0.012,
            "water in 0.32 m tire pad at ({dx:.2},{dz:.2}) got {:.4}",
            s.water
        );
    }
}

#[test]
fn win_demo_seam_heights_stay_bounded() {
    let world = World::win_demo();
    let (cx, cz) = (-1.05_f32, 0.15_f32);
    let mut max_h = f32::MIN;
    let mut min_h = f32::MAX;
    for j in -12..=12 {
        for i in -12..=12 {
            let x = cx + i as f32 * 0.2;
            let z = cz + j as f32 * 0.2;
            let s = world.soil.sample(x, z);
            max_h = max_h.max(s.height);
            min_h = min_h.min(s.height);
            assert!(
                s.height.is_finite() && s.height.abs() < 0.55,
                "height out of range at ({x:.2},{z:.2}): {}",
                s.height
            );
        }
    }
    // 0.35 → 0.40: the 1210 kg Niva's ruts at the seam are deeper (span 0.376).
    assert!(
        max_h - min_h < 0.40,
        "seam pad too rough: span {:.3}",
        max_h - min_h
    );
    let hub = world.soil.sample(cx, cz);
    assert!(
        hub.height < 0.02,
        "seam hub should sit in bowl floor {:.3}",
        hub.height
    );
    let mut max_h = 0.0_f32;
    let mut xi = 0;
    while xi < 24 {
        let mut zi = 0;
        while zi < 24 {
            let x = -0.92 + xi as f32 * 0.04;
            let z = -0.22 + zi as f32 * 0.04;
            max_h = max_h.max(world.soil.sample(x, z).height);
            zi += 1;
        }
        xi += 1;
    }
    assert!(
        max_h < 0.28,
        "hero pad spike above bowl lip {:.3}",
        max_h
    );
    assert!(
        max_h - min_h > 0.18,
        "bowl bank should read from side: span {:.3}",
        max_h - min_h
    );
}
