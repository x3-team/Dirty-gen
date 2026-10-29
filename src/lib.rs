//! Silt — real-time soil and off-road vehicle solver.
//!
//! The soil is a heightfield with plastic sinkage, loose mud, wetness and
//! shallow water. Wheel forces come from Bekker pressure-sinkage, Janosi-Hanamoto
//! shear and a Bingham-like drag. Rendering lives in the `silt` binary.

pub mod bike;
pub mod dust;
pub mod fluid_core;
pub mod particles;
pub mod patch;
pub mod render_contract;
pub mod soil;
pub mod vehicle;
pub mod world;

pub use world::{
    demo_camera, demo_drive, demo_drive_bike, dig_input, BellyShelfReport, CameraPose, Input,
    ThroughDrive, ThroughReport, World,
};
