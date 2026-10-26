//! Silt — real-time soil and off-road vehicle solver.
//!
//! The soil is a heightfield with plastic sinkage, loose mud, wetness and
//! shallow water. Wheel forces come from Bekker pressure-sinkage, Janosi-Hanamoto
//! shear and a Bingham-like drag. Rendering lives in the `silt` binary.

pub mod particles;
pub mod soil;
pub mod vehicle;
pub mod world;

pub use world::{demo_camera, demo_drive, dig_input, CameraPose, Input, World};
