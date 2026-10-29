# Dirty-gen

Dust caller for the Lada Niva. The air grid is `fluid_core` (slurry ledger): 128 by 8 layers, wind via `set_wind`, settle every 6 frames from the bottom layer.

A tire offers `k * slip * dryness` cubic metres per second through `DustInject`. `dust_may_lift` is false at moisture 0.55 or at 8 mm of water, and `inject_dust` then returns 0. Density on the 128×8×128 window is carried by `set_wind` plus the wake, then settles. When `src/fluid_core.rs` is present that step calls its advection. Settled dust stays down. Emitted volume equals airborne plus settled. The renderer receives a 3D density texture.

```
cargo test
cargo run --release -- /path/to/dust-trail-proof.png
```

The proof is 1280×720, the Niva only, no text plate. The exterior mesh is Greg McKechnie’s Lada Niva (CC BY 4.0); see `assets/vehicle/NIVA-LICENSE.txt`.
