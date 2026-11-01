//! A person wading the beach of the ford, buoys riding the swell, a rock in
//! the swash and a ball rolled across the sand. The scene for the bodies
//! (`crate::bodies`) against the UE5 beach shots.
//!
//! The ford is [`crate::ford::seed_ford`] with its swell. Nothing here draws
//! a wake, a print or a ring. The walker's legs and the buoys are volumes in
//! the sheet, the soles and the ball press the sand through the 2.5 cm
//! imprint layer, and the renderer reads the sheet as for the Niva.

use glam::{Vec2, Vec3};

use crate::bodies::{bed_base, Bodies, Body, PrintGround, Walker};
use crate::fluid_core::FluidCore;
use crate::ford::{seed_ford, still_waterline_z, sync_soil, Swell};
use crate::imprint::Imprint;
use crate::world::World;

pub struct WadeScene {
    pub world: World,
    pub fluid: FluidCore,
    pub bodies: Bodies,
    pub imprint: Imprint,
    pub time_s: f32,
}

/// Beach depth grows 1 m in 11 southward from the still waterline.
pub fn beach_z(depth_m: f32) -> f32 {
    still_waterline_z() + depth_m * 11.0
}

impl WadeScene {
    pub fn new(swell: Option<Swell>, settle_s: f32) -> Self {
        let mut fluid = FluidCore::new([0.0, 0.0]);
        seed_ford(&mut fluid, swell);
        let dt = 1.0 / 60.0;
        for _ in 0..(settle_s.max(0.0) / dt) as u32 {
            fluid.step(dt);
        }
        let mut world = World::flat(0.0);
        world.soil = crate::soil::SoilField::flat(0.0);
        world.particles.clear();
        world.vehicle.clear_loose_mud();
        // The Niva is parked out of the frame; this scene is about the bodies.
        world.vehicle.pos = Vec3::new(-15.0, 0.6, -14.0);
        world.vehicle.vel = Vec3::ZERO;
        world.bike.pos = Vec3::new(80.0, 0.0, 80.0);
        sync_soil(&fluid, &mut world.soil, true);

        let mut imprint = Imprint::new();
        let mut bodies = Bodies::new();
        {
            let ground = PrintGround {
                imprint: &mut imprint,
                base: bed_base(&fluid),
            };
            // Off the dry terrace, down into the swash, then along the beach
            // at shin to knee depth.
            let mut w = Walker::new(Vec2::new(-5.5, beach_z(-0.15)), 0.8, 1.35, &ground);
            w.path = vec![
                Vec2::new(-2.5, beach_z(0.12)),
                Vec2::new(1.5, beach_z(0.26)),
                Vec2::new(6.0, beach_z(0.30)),
            ];
            bodies.walkers.push(w);
        }
        // Buoys in the shallows: 40 cm, 6 kg, half a buoy's volume under.
        for (x, d) in [(-0.6f32, 0.28f32), (2.4, 0.22), (4.6, 0.31)] {
            let z = beach_z(d);
            let y = fluid.sample(x, z).bed_m + d + 0.05;
            bodies.items.push(Body::ball(Vec3::new(x, y, z), 0.2, 6.0, true));
        }
        // A rock in the swash.
        let rz = beach_z(0.06);
        let rb = fluid.sample(1.2, rz).bed_m;
        bodies.items.push(Body::crate_box(
            Vec3::new(1.2, rb + 0.3, rz),
            Vec3::new(0.45, 0.28, 0.6),
            2600.0 * 8.0 * 0.45 * 0.28 * 0.6,
            0.5,
        ));
        // A ball rolled across the damp sand above the waterline.
        let bz = beach_z(-0.05);
        let bb = fluid.sample(-6.0, bz).bed_m;
        let mut ball = Body::ball(Vec3::new(-6.0, bb + 0.16, bz), 0.16, 4.0, false);
        ball.vel = Vec3::new(2.4, 0.0, 0.3);
        ball.omega = Vec3::new(0.3 / 0.16, 0.0, -2.4 / 0.16);
        bodies.items.push(ball);

        Self {
            world,
            fluid,
            bodies,
            imprint,
            time_s: 0.0,
        }
    }

    pub fn step(&mut self, dt: f32) {
        {
            let mut ground = PrintGround {
                imprint: &mut self.imprint,
                base: bed_base(&self.fluid),
            };
            self.bodies.step(dt, &mut ground, Some(&self.fluid));
        }
        self.bodies.hand_off(&mut self.fluid);
        self.fluid.step(dt);
        sync_soil(&self.fluid, &mut self.world.soil, false);
        self.world.time += dt;
        self.time_s += dt;
    }
}
