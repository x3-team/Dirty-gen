//! Corner map. The lap timer and the start menu live elsewhere.

mod map;
mod positions;

pub use map::{widget_mesh, MapMesh, MapVert};
pub use positions::{read_positions, VehiclePos, Which};
