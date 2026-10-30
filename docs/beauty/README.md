# Beauty pass: IBL, wet ground, rut water, Niva mud, contract v1

Only the picture changes here. `soil.rs`, the force formulas and the solver fields are untouched; every value the shaders read (wetness, water depth, kind, panel mud mass and wetness) is the solver's own.

## Light

- `src/env.rs` builds an equirect radiance map with a box-filtered mip chain and nine SH terms for diffuse sky light.
- An HDRI named `assets/hdri/sky_*.hdr`, or any file given as `SILT_HDRI=/path/file.hdr`, is turned so its sun sits on the scene sun and scaled to the analytic sky's level, so the exposure the soil was tuned at holds. Without one, the analytic sky is baked into the map.
- The two in-repo HDRIs are not picked up on their own: `lebombo_2k.hdr` is an interior (walls and a window fill the sky), `rogland_clear_night_2k.hdr` is a night sky. Both still work through `SILT_HDRI`. An outdoor daytime sky dropped in as `sky_<name>.hdr` becomes the default.
- `shade_pbr` in `shader.wgsl`: GGX sun with the existing shadow map, split-sum IBL (Karis env BRDF), horizon and roughness specular occlusion, metallic from the Niva MR maps. ACES filmic tonemap replaces `1 - exp(-x)`. Thin aerial haze that warms toward the sun.

## Ground textures (beauty pass)

- Wet mud PBR sets live under `assets/beauty/wet_mud-opt/` (ambientCG Ground037 and Ground048, CC0).
- Sand PBR sets live under `assets/beauty/sand-opt/` (Poly Haven aerial beach and ambientCG Ground033, CC0).
- At runtime the ground array has seven layers: the five CC0 grounds, then Ground048 for wet mud (`SurfaceKind::WetMud`) and Ground033 for sand (`SurfaceKind::Sand`). Ground037 is a mossy lawn and aerial_beach_01 a gray wet beach, so neither is used. Sets without AO get a white AO map.
- The terrain fetches one layer per pixel (`layer_uv`) instead of sampling all five and picking one.

## Ground

- Wetness lowers roughness per surface kind (wet mud, fresh cut, packed clay, silt, sand...) and darkens by porosity.
- Above ~0.72 wetness a water film collects in the low texels of the height map first, and the shading normal relaxes there.
- Dry crust and dried clay/silt get a cell-crack pattern (dark matte cracks, curled plates), driven by the solver's wetness and kind; dry sand gets sparse quartz glints.
- Saturated soil relaxes its normal map toward the cut (water fills the grain), so wet mud is smoother than the dry map.
- Water the solver keeps in a cut is drawn on the terrain mesh (that is how #64 draws it): a calm rippled sheet, mirror-smooth, over opaque brown silt.
- Normal-map relief fades with distance and a variance term widens the lobe, so far wet ground reads as sheen instead of sparkle (the ground textures have no mips).

## Rut water

Ponds outside a cut use the water mesh and the solver's column (`mat.g`), with #64's discard rules. Beer–Lambert absorption through the column over the real soil pass, silt scattering lit by sun and sky, Fresnel sky reflection, a short heightfield march so a grazing reflection shows the rut bank instead of sky, capillary ripples, a GGX sun glint, pale foam on a pushed crest, feathered shore.

## Niva body mud

`niva_body_mud` maps each solver panel (fenders by hub, sides, nose, rear, belly) onto the shell in model space, so the pattern rides with the car. Heavy low and in the arches, flecks higher, smears stretched along the body, hood and roof mostly flecks. Thin edges dry first: wet mud is dark and glossy, dry mud pale and matte.

## Render contract v1 (#88)

The picture reads the published maps of `docs/RENDER_CONTRACT.md` and nothing else from the sheet. `src/beauty_contract.wgsl` binds the contract as group 4 of the terrain, water and dust-volume pipelines and is appended to `shader.wgsl` when the device can bind it (five bind groups, storage buffers). Otherwise `src/beauty_nocontract.wgsl` supplies the same reader functions returning the picture's own cache, so WebGL2 keeps the old look. Every reader fades to the fallback over the last two cells of the 40 m window.

| Map | What the picture does with it |
|---|---|
| `moisture` | wetness for roughness, darkening, water in the grain, wet normals |
| `film` | water + slurry thickness: the water sheet is ground + film at its true height; Beer–Lambert column, slurry far more opaque than water |
| `flow` | ripples and lace foam drift downstream; streaks comb along fast water |
| `foam_crest` | foam fraction as a lace advected by flow; a crest of a few mm adds a thin lace |
| `wet_edge` | a damp halo and a pale tide line on the dry side of the boundary |
| `dust_settled` | matte grey cover from 0.04 mm, a dark smear where the ground is wet |
| `crust` | crack network (domain-warped Voronoi F2−F1 plus a finer secondary net), width from the opening. v1 writes 0, so only the soil cache drives cracks today |
| `dust_fog` | fullscreen raymarch (28 steps, trilinear) up to scene depth: sun with the shadow map and a forward HG lobe, sky fill, headlamp in-scatter. Picture dust cards are off while the volume is valid |
| `spray` | materials 0/2/4/6/9 become drops smeared along their last frame of travel; slurry 5/7 becomes torn ribbons along the ballistic path, width from mass, edges and holes torn in the shader. Clods 1/3 and anything slower than 0.5 m/s stay in the chunk mesh |

**Round 2 (FluidNinja target).** The beauty drive runs in day light on a sand field (`Renderer::sand_field`: kind 0 and the skirt use the CC0 sand, picture only) under a baked cumulus sky. Water: clear Beer–Lambert (5.5/7/10 per metre; slurry stays 90–140), screen-space reflections from a copy of the opaque frame (water draws in its own pass with read-only depth), a soaked dark rim sampled from the film around each texel, caustics on the bed (warped drifting cell nets, killed by depth and slurry), a tire collar, bow lift and V wake from the tire poses, and bubble-wall foam. Dust: billow noise inside the published cells and a three-step self-shadow toward the sun. With the contract, slurry lying in the rut is the film; the cut-out rut strips are not drawn. `--beauty-crust <dir>` renders `drying-crust`: one mud drying from the rim in, pale baked plates with domain-warped cracks.

The Niva headlamps (frame uniform `lamps`) light the ground, the water glints and the dust; they are on in the dusk scene only.

Contract notes for the solver side: `flow` and `foam_crest` carry NaN in some dry cells of the fluid sheet; the readers treat non-finite texels as 0. In the through-drive the puddle's `foam_crest` stays at 0 and the crest at about 9 mm, and the spray buffer holds only slurry (5), no water droplets off the arches, so the puddle has no white water to draw.

### Through-drive stills and video

```bash
cargo run --release -- capture --software --width 1280 --height 720 --beauty-drive /tmp/drive --beauty-video
```

Stills `puddle-wave` (z −17.5), `slurry-spray` (z −6.5), `dust-exit` (z 9.0) and `through-drive-beauty.mp4` at 30 fps on the chase camera. `SILT_BEAUTY_STILLS=puddle-wave` renders just the named stills. The checked-in frames are `drive/*.jpg`.

## Proof stills

```bash
cargo run --release -- capture --software --width 1280 --height 720 --beauty docs/beauty
```

Writes PNG stills (`niva-rut-water`, `rut-water-close`, `niva-body-mud`, `niva-low-sun`, `wet-gloss`, `sand-clay-wet`, `sand-clay-dry`); the checked-in proof frames here are the matching `.jpg` files.

The rut scene is the hard pass from `--force-proof`: both tracks cut by `roll_with_force` at the depth `rut_depth_from_force(HARD_PASS_NORMAL, HARD_PASS_PUSH, ...)` gives (19.2 cm). Water is poured into the grooves to 7 cm under the old surface and settled by two seconds of `step_water`. The body mud is loaded by the solver in a bog (roll, then spin) and carried over. `before-after-*.jpg` pair the #64 head (left) with this branch (right).
