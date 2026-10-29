//! Dust on the shared air grid.
//!
//! Slurry's `fluid_core` owns the moving window, advection, pressure
//! projection, and vorticity confinement. This crate is the dust caller:
//! tires inject by slip and dryness, dust settles onto the ground, and the
//! renderer receives a 3D density texture.
//!
//! `src/fluid_core.rs` is not on this branch. Do not add a second copy.

mod dust;
mod gpu;

pub use dust::*;
pub use gpu::{upload_density_texture, GpuDensity};
