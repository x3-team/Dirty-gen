# Dirty-gen

Dust trail solver for the Lada Niva.

On dry ground each tire lofts fines. The mass rate is `k * dryness * max(speed - 0.5, 0) ^ 1.5` kilograms per second. The near-wake density is that rate mixed into the air the tire sweeps, so it rises with dryness and with speed. Wind drags the parcels. Moisture at or above 0.55, or standing water at or above 8 mm, lofts nothing.

```
cargo test
cargo run --release -- /path/to/dust-trail-proof.png
```

The proof is 1280×720, the Niva only, no text plate. The exterior mesh is Greg McKechnie’s Lada Niva (CC BY 4.0); see `assets/vehicle/NIVA-LICENSE.txt`.
