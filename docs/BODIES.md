# Free bodies: walker, balls, crates, rocks

`src/bodies.rs`, `src/imprint.rs`, `src/fluid_core/wet.rs`, scene `src/wade.rs`.

The reference shows a person wading the swash, buoys riding it, a rock in
the shallows, a ball rolling a groove into mud and boot prints in snow. None
of those effects are drawn. Each body hands the sheet the volume it fills and
the drag it took, and presses the ground through that ground's own law.

## Water

A body is a volume that shares the water column. On each step the part of a
column it fills below the free surface (`WetShape::span`, supersampled 4×4 per
20 cm cell) is added to the floor the sheet step reads, and only there. The
water arrays are never touched, so volume is exact. The step moves water off
the raised floor: the ring when a ball drops in, the bow ahead of a leg, the
trough behind it.

The free surface a body sits in is the ring around it from the last step
(`WetLoad::surface_y`), not the column under it, which is what the lift is
changing (reading that back rang at two steps).

The body does its own Archimedes buoyancy, quadratic drag against the ring's
current, heave damping and added mass. The horizontal drag impulse goes back
into the water around it with the opposite sign (`WetBody::impulse_*`), so
momentum is exchanged, not made.

Bodies do not pull the 2 cm patch over themselves. The patch's rim is open to
dry fine cells (non-held cells keep water 0 and a stale bed), so in 40 cm of
water each step drains the held cells into the rim, and `restrict_fine` adds
it to the coarse ring: a 20 cm hole follows the patch. Tires in deep ford
water would show the same. A body that is under a tire's patch anyway is
filled on the fine grid too.

## Foam and drops

Where a body cuts the free surface (`wet_splash`, after the ring is read),
it foams the waterline once its bow breaks. The sheet derives each body's
speed from the positions it is handed, takes the ring's current off it and
forms the body Froude number `U / sqrt(g D)`, with `D` the waterline width.
The bow sheet of a surface-piercing cylinder breaks from about 0.5 and is
fully broken and ventilated by about 1.4 (Chaplin & Teigen 2003); below the
slowest water wave, 0.23 m/s, nothing breaks. A body dropping through the
surface throws a crown on the same law with its entry speed. The foam goes
into the #104 tracer field (`add_foam`), whose pass carries and decays it, so
a wading leg leaves a trail and a buoy riding the swell in place leaves none.

Past 0.6 m/s the broken sheet sheds drops, `Q = 0.02 · D² · (v − 0.6)` m³/s,
taken from the ring cells as water and booked on `spray_m3`. A 13 cm leg
swinging at 2 m/s throws about half a litre a second; four seconds of wading
throw 1.5 l, a 44 cm ball dropped from 60 cm throws 0.8 l. The drops are the
spray buffer's `kind 0`, so the contract and the renderer already carry them.

## Ground

`Ground` is what a body stands on: `surface`, `pressure(x, z, y)`, `mu`,
`press`. Contact is a bearing limit, not a spring: the ground carries at most
the plate law's pressure at the body's underside, so a body past that sinks
and the press digs the print deeper. A body resting in its own print meets
exactly the load that made it.

- `PrintGround` (sand, mud, gravel, grass): `p = k·s` with sinkage memory `s`
  on a 2.5 cm offset layer over the coarse ground (`Imprint`, sparse world
  tiles). `bulge` of the pushed volume rises in a rim, the rest is
  compaction on a ledger (`Σ dz·A + compacted = 0`). Dry sand slides to 34°,
  damp sand and mud keep the wall. Calibration (80 kg walker): damp sand
  1.6 cm, dry sand about 3.5 cm, soft mud 5–6 cm.
- `SnowTrack` (from #123): the bearing pressure is the track's own compaction
  law turned around, the pressure that packs the column from the blanket
  density to the density it would have at `y`. A boot packs the column and
  shears 12 % out to a low rim. Mass closes to 1e-10.

## Bodies

- `Body::ball` (solid or shell): full rotation, rolling friction at the
  contact, rolling drag from the groove depth.
- `Body::crate_box`: level box, yaw only. Rocks are dense crates drawn as
  boulders.
- `Walker`: pace controller plus a step cycle. Each planted sole carries the
  weight (×1.2 at heel strike) at the height where the ground's bearing equals
  it (bisection on `pressure`). Both legs are columns in the sheet; water drag
  on them slows the walk (1.4 m/s asked, about 0.9 m/s at 40 cm).

## Proof

`cargo test --release --test bodies`: buoy draft within 2 % of Archimedes and
agreeing with the sheet's displaced volume; a dropped ball's ring at 1.2 m
(+1.2 cm / −1.1 cm); a rock that sinks and sits in the sand; a bow over the
trough at a wading leg; prints in damp sand (1.6 cm, rim 5 mm, one per step),
a groove and berm from a ball in mud, 10 cm boot prints in fresh snow; water,
imprint and snow ledgers closed; no hole left after a body leaves; foam at
wading legs and in a trail behind, a crown of rising drops from an impact,
no foam and no drops from a buoy at rest; water plus drops closed.

`capture --bench out --bench-scenes bench/scenes.txt` renders `wade-swash`,
`wade-buoys`, `wade-rock`. `SILT_BODIES_DUMP=dir cargo test --release --test
bodies dump_maps -- --ignored` writes the top-down surface, prints and boot
maps.

## Not yet

- The render contract and the sheet are 20 cm. A 13 cm leg fills part of one
  cell, so its wake is one or two cells wide, and the 2.5 cm prints are not in
  any binding yet. Both need a fine window: the prints fit the `track_fine`
  layout of #123 (R surface over ground), the water needs a persistent fine
  sheet (the GPU sheet, M4) or a closed-rim fine patch.
- Foam is on the 20 cm grid, so a leg's ring is two or three cells. Main's
  shader does not draw contract foam yet; #135 does (binding 9 foam).
- Boxes do not tip; the walker is kinematic above the soles, not a ragdoll.
