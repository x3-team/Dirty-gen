//! Dust on the shared air grid.
//!
//! Slurry's `fluid_core` owns the moving window, pressure projection, and
//! vorticity confinement. This crate injects dust by slip and dryness, advects
//! the density, settles it, and hands the renderer a 3D density texture.
//!
//! `src/fluid_core.rs` is not on this branch. The density step below is the
//! stand-in for that module's advection. It does not project pressure and it
//! does not confine vorticity. Do not add a second `fluid_core`.

#[cfg(has_fluid_core)]
include!(concat!(env!("OUT_DIR"), "/fluid_core_link.rs"));
mod dust;
mod gpu;

pub use dust::*;
pub use gpu::{upload_density_texture, GpuDensity};
