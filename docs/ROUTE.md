# Niva route

One run of the Niva through every surface the slice has, in the order a driver
meets them, a fixed camera per segment, each still next to the matching UE5
fluid-mod reference, and the CPU time of every solver step along the way.

```sh
cargo build --release
./target/release/silt capture --software --width 1280 --height 720 --route out/route
python3 tools/bench_sbs.py out/route --scenes bench/route.txt --refs <ue5-bench/refs>
open out/route/index.html        # SBS + score fields; out/route/route.json = CPU per leg
```

About 2.5 minutes on lavapipe. `--route-scenes <file>` reads another shot list.
`SILT_ROUTE_SNOW=<m>` changes the snow blanket (default 0.30 m).

## Legs

`silt::route` (`src/route.rs`) holds the legs, `src/route_bench.rs` drives and
draws them. Every leg is the solver the tests and the UE5 bench already run;
nothing here paints a surface.

| # | Leg | Solver | Segments (shots) |
|---|---|---|---|
| 1 | `drive` | `ThroughDrive`, 8 m/s, dt 1/30 | puddle, clay slurry ruts, dry exit with dust |
| 2 | `crust` | `World::wet_clay` + `drying_crust_frame`, grass/sand/gravel strips + `blend_material_joints` | cracked crust lanes, soil seams |
| 3 | `shore` | `route::ShoreLeg`: `FordDrive` with swell + `Bodies` (walker, two buoys), dt 1/60 | sand shallows/swash, walker in the water, ford, silt plume |
| 4 | `snow` | `route::SnowLeg`: 30 cm blanket on the 2.5 cm `SnowTrack`, real Niva, straight, left turn, straight, stop | ruts and berms, the turn |

The legs are separate scenes, not one world. The soil map and the sheet
window are 40 m and there is no streaming between scenes yet, so the route
cannot be one continuous map. What carries over is the car: panels and
tires keep the mud (`route::carry_coat`). The order differs from "dust, mud,
puddle" because the through-drive runs puddle, then slurry, then the dry exit.

## Scores (2026-11-01, main 0f936de + this change)

Honest 1-5 against the reference in `bench/route.txt`. No SnowRunner shots are
in the reference set; every reference is a UE5 fluid-mod frame. Average 2.2.

| Shot | Ref | Score | What is there | What is missing |
|---|---|---:|---|---|
| puddle | ue5-104 | 1 | bow ring, pit, wet berm | the water barely reads at the car; spray drawn as tall vertical cards over the roof; dark rectangular slabs at the wheels |
| mud | ue5-105 | 2 | two ruts, berms, darker wet track | no wet gloss or viscous folds on the slurry; same card pillars and slabs |
| dust | ue5-93 | 2 | trail behind the car, drifts with the wind | a thin tube, not a cloud; no volume, no self-shadow, no billowing |
| crust | video frame | 2 | polygon cracks on the pad, rolled lanes | seams are hard straight rectangles; cracks only read close; no reference for a crust (mud video used) |
| sand | ue5-98 | 2 | swash line, wet sand, car sinking into the water | water flat grey-blue, no sun glints, no foam lines on the swash; white jagged fringes around rocks |
| walker | ue5-100 | 2 | rings around legs and buoys, buoys ride | mannequin body; water without colour or depth; rock fringes |
| ford | ue5-104 | 3 | car mid-river, bow splash, wake, clear shallows over the bed | flat sky in the water, little surface detail, no caustics |
| plume | ue5-103 | 2 | brown plume behind the wheels | plume faint, no foam streaks at rocks, no flow texture |
| snow-ruts | ue5-96 | 3 | deep ruts with tread lines | flat-shaded snow (no AO, sparkle or blue shadow); ruts wash out when the sun is behind the camera; berms do not read |
| snow-turn | ue5-97 | 3 | turn arc with several tread lines | one pass only, no prints; berms smooth; same shading |

## CPU along the route

`out/route/route.json`. Intel Xeon @ 2.80 GHz, 4 cores (this container), release,
lavapipe drawing between steps only at stills. Step = one solver step of the leg.

| Leg | dt | Steps | Mean ms | p95 ms | Max ms | ms per 1/60 s | One-off |
|---|---|---:|---:|---:|---:|---:|---:|
| drive | 1/30 | 149 | 6.54 | 9.07 | 23.05 | 3.27 | 819 ms set-up (pour + drain) |
| crust | - | - | - | - | - | - | 1440 ms drying spell |
| shore | 1/60 | 1081 | 8.36 | 11.42 | 19.91 | 8.36 | 1438 ms river settle |
| snow | 1/60 (2 x 1/120) | 901 | 3.52 | 6.33 | 7.10 | 3.52 | - |

Caps are unchanged. This container is slower than the CI runner: untouched
main fails `tests/e2e_niva.rs` here with an e2e step of 14.30 ms against the
11.82 cap, so read the absolute numbers as about 20% high and the order of the
legs as the finding. The drive's p95 (9.07 ms) is over the 7 ms `DRIVE_P95`
cap on this machine for the same reason; CI's release log is the measurement
that counts.
The shore leg (ford sheet with swell, the car as walls, a walker and buoys as
wet bodies) is the most expensive surface: 8.4 ms per 60 Hz frame, over the
7 ms solver frame budget, and it has no cap of its own.

## Weakest spots, ranked

The next work plan. Owner is the system the fix belongs in.

1. **Spray and slurry drawn as tall vertical cards, dark slabs at the wheels**
   (puddle, mud). The worst thing on the route: reads as broken geometry.
   Render: particle cards for spray/slurry sheets.
   *Fixed (spray/slurry PR):* the "cards" were the water mesh: tire wake rings
   (`solver_shock_rings`, unit sines per tire) were added to the vertex
   height and stood the sheet up 2-3 m at the wheels; now they shade the
   normal only. The slabs were picture-only rut ribbons with no volume; they
   are gone. Spray and flying slurry follow hybrid H2: drops leave the sheet
   as bow wings and a rooster tail, land back into the column (volume plus
   a share of momentum; `spray_m3` is now what is in the air), tread mud
   leaves the rim by rolling kinematics, and both draw as volume-sized
   droplet sprites (`mesh::splash_sprites`, `fs_splash`).
2. **Puddle water does not read at the car** (puddle). Needs a check of sheet
   depth vs water shading at dusk. Physics (puddle pour/drain) or render
   (water column at `daylight = false`); unverified which.
3. **Shore leg CPU 8.4 ms mean / 11.4 ms p95 per frame** on 4 cores. Physics:
   ford sheet with swell + hydro solids + wet bodies.
4. **Dust is a thin tube, not a cloud** (dust). Physics: emission and density
   behind the car; render: volumetric dust once the density is there.
5. **Water shading** (sand, walker, ford, plume): flat grey-blue, no sun
   glints, no foam lines on the swash, weak brown plume. Render (water
   shader) plus foam/turbidity fields for the lines and the plume.
6. **White jagged fringes around rocks** (sand, walker, ford). Render:
   waterline/foam at rock bumps.
7. **Soil seams are hard rectangles** (crust). Physics: one joint blend pass at
   20 cm leaves a 1-cell transition; render: no height blend between materials.
8. **Mud has no wet gloss or viscous folds** (mud). Render: wet slurry
   shading; physics: slurry surface resolution off the 2 cm tire zone.
9. **Snow shading** (snow): no AO or micro-shadow, so ruts vanish with the sun
   behind the camera; no sparkle or blue shadow. Render. Berms smooth and one
   track pair only: physics.
10. **Walker is a mannequin, Niva mud coat barely shows** after the mud leg
    although the panels carry it. Art/render.
11. **Route is four scenes, not one world.** Engine: 40 m map and window, no
    streaming. Needed for one continuous drive.
12. **Environment** (flat sky, hard map edge in every shore and snow shot).
    Out of focus by earlier decision; it still costs points in every SBS.
