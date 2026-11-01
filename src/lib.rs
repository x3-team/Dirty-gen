//! Silt — real-time soil and off-road vehicle solver.
//!
//! The soil is a heightfield with plastic sinkage, loose mud, wetness and
//! shallow water. Wheel forces come from Bekker pressure-sinkage, Janosi-Hanamoto
//! shear and a Bingham-like drag. Rendering lives in the `silt` binary.

pub mod adhesion;
pub mod bike;
pub mod dust;
pub mod fluid_core;
pub mod ford;
pub mod lod;
pub mod particles;
pub mod patch;
pub mod quality;
pub mod render_contract;
pub mod smoke;
pub mod snow;
pub mod soil;
pub mod vehicle;
pub mod world;

pub use world::{
    demo_camera, demo_drive, demo_drive_bike, dig_input, BellyShelfReport, CameraPose, Input,
    DriveParts, ThroughDrive, ThroughReport, World,
};
