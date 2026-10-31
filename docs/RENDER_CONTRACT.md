# Render contract

Version **2**. The solver publishes this and the debug view reads only this. Picture shading is separate: materials, light, HDRI, fog, headlights, and wet sheen are not maps in this contract.

Change the contract only by bumping `RENDER_CONTRACT_VERSION` in `src/render_contract.rs` and in `src/render_contract.wgsl`, and by adding a row to the changelog below. A new channel, a new unit, or a new resolution is a version bump. Filling a channel that was already defined is not. Writing 0 for a non-finite sample is not. Recording that foam and spray were empty, and filling those same fields from the sheet, is not.

The demo this contract serves is the Lada Niva. The window follows that vehicle.

## Window

Coarse ground window, after the 2 cm tire patch has been restricted back. The fine patch is not its own map.

| | |
|---|---|
| Cells | 200 × 200 |
| Cell | 0.20 m |
| Extent | 40 m × 40 m |
| Origin | minimum corner, snapped to the cell |
| Axes | X east, Y up, Z north, metres |
| Texel (i, j) | centre `origin + ((i + 0.5) * cell, (j + 0.5) * cell)`, i along +X, j along +Z |

With the fluid sheet attached, the origin is that sheet's window. Otherwise it is the same snap around the Niva `(x, z)`.

Valid after the solver step that produced the frame, and until the next step. The renderer uploads the maps at the start of the draw. It does not step the solver.

## Bind group

Group 0 of the debug shader. Layout: `contract_bind_group_layout`. Header: `ContractHeader`, 96 bytes. Spray slot: `SprayGpu`, 48 bytes.

`debug_mode` in the header is a viewer selector (0 ground, 1 rut, 2 moisture, 3 water, 4 slurry, 5 flow speed, 6 settled dust, 7 crust, 8 crack, 9 wet edge, 10 foam, 11 crest, 12 spray count, 13 airborne dust, 14 air speed). It is not a solver map. Changing which mode is on screen does not bump the version.

| Binding | Name | Format | Channels | Units |
|---|---|---|---|---|
| 0 | header | uniform | see `ContractHeader` | — |
| 1 | `ground_rut` | `rgba32float` | R ground, G rut, B bed, A placed | m |
| 2 | `moisture` | `r32float` | R pore moisture | 0..1 |
| 3 | `film` | `rg32float` | R water, G slurry | m |
| 4 | `flow` | `rg32float` | R velocity +X, G velocity +Z | m/s |
| 5 | `dust_settled` | `r32float` | R settled dust depth | m |
| 6 | `dust_fog` | `r32float` 3D | R airborne volume fraction | m³/m³ |
| 7 | `crust` | `rg32float` | R crust thickness, G crack opening | m |
| 8 | `wet_edge` | `r32float` | R edge | 0 or 1 |
| 9 | `foam_crest` | `rg32float` | R foam, G crest | 0..1, m |
| 10 | `spray` | storage, read-only | `SprayGpu` × 4096 | see below |
| 11 | `air_velocity` | `rgba32float` 3D | R +X, G +Y, B +Z, A 0 | m/s |

Ground maps are 200 × 200. `dust_fog` is not that grid. Spray is a buffer, not a grid.

### Ground and rut — binding 1

Solid ground is bed + placed. Rut is how far the bed sits below the height the cell started at, and it does not include water or slurry. Bed and placed are the two layers that sum to ground.

Writer: the soil heightfield at the texel centre. Off that 40 m soil square, bed comes from the fluid sheet when it is attached, and rut and placed are 0.

Valid: `VALID_GROUND` is set on every publish.

### Moisture — binding 2

Pore moisture, 0 dry to 1 saturated. Not the film thickness.

Writer: the fluid sheet's moisture when the sheet is attached, otherwise the soil wetness.

Valid: `VALID_MOISTURE` is set on every publish.

### Water and slurry — binding 3

Thickness of the standing water column and the slurry column, metres. They are separate. Ground height does not include them. The visible top of a wet cell is ground + slurry, with water on that.

Writer: the fluid coarse cell when the sheet is attached (fine patch already restricted). Otherwise the soil `water` and `slurry` arrays. A bow drawn only onto the picture soil is not copied back into the sheet.

Valid: `VALID_FILM` is set on every publish.

### Flow — binding 4

Horizontal velocity of the film, metres per second, world +X and +Z. Thickness-weighted: `(water * water_velocity + slurry * slurry_velocity) / (water + slurry)`. A cell with less than 0.01 mm of film is 0.

Writer: the fluid sheet. `header.flow_source` is 1 when that sheet was attached and 0 when it was not. Without the sheet the texels are 0 and `VALID_FLOW` is clear. Those zeros are not a measured still surface.

A dry cell, film under 0.01 mm, is 0 even if the stored velocity is not. A sample that is not finite is written as 0. The same rule covers every map and the spray buffer: no NaN, no Inf.

Valid: `VALID_FLOW` only when `flow_source` is 1, after `FluidCore::step`.

### Settled dust — binding 5

Dust that has landed, as a depth in metres on the 40 m window.

Writer: the dust ledger when it is attached. A settled mark is cubic metres on a 0.50 m cell; depth is that volume divided by `0.50 * 0.50`, added to every window texel whose centre lies in the mark. Without the ledger, the fluid sheet's settled depth is copied per coarse cell. With neither, the map is 0 and `VALID_DUST_SETTLED` is clear.

Valid: `VALID_DUST_SETTLED` when `header.dust_source` is 1 (ledger) or 2 (fluid air).

### Dust and fog — binding 6

Airborne dust and fog as a volume fraction, cubic metres of dust per cubic metre of air. Not clamped.

| | Ledger (`dust_source` 1) | Fluid air (`dust_source` 2) |
|---|---|---|
| Size | 128 × 8 × 128 | 128 × 8 × 128 |
| Cell | 0.125 m × 0.25 m × 0.125 m | `(40/128)` m × 0.25 m × `(40/128)` m |
| Coverage | 16 m × 2 m × 16 m, vehicle frame | the 40 m window |
| Index | `x + layer * 128 + z * 128 * 8` | same index after conversion |

Texel centre is `air_origin + (i + 0.5) * air_cell`. `air_origin` is the corner of texel (0, 0, 0), Y up. The ledger origin moves with the Niva. The fluid-air fallback stores a depth in metres in each slab and publishes `depth / 0.25`.

The ledger wins when both are present. It is the closed dust account.

Writer: `Dust::publish`, or `FluidCore::air_depth` when the ledger is absent.

Valid: `VALID_DUST_FOG` under the same rule as settled dust. `dust_source` 0 means the grid was not written.

### Air velocity — binding 11

World air velocity on the same 128 × 8 × 128 grid as `dust_fog`. Same origin and cell: `header.air_origin_*`, `header.air_cell_*`. Texel centre is `air_origin + (i + 0.5) * air_cell`.

| | |
|---|---|
| Format | `rgba32float` 3D |
| Index | `(x + layer * 128 + z * 128 * 8) * 4` |
| R | velocity +X, east, m/s |
| G | velocity +Y, up, m/s |
| B | velocity +Z, north, m/s |
| A | 0 |
| Resolution | 128 × 8 × 128 |
| Cell | the `dust_fog` cell for this `dust_source` |

The step is 2.5D. Advection and vorticity confinement run in the horizontal plane of each layer. The vertical channel is stored. Wheel spin writes it. The horizontal step does not advect between layers.

Writer, when `dust_source` is 1: `step_vehicle_air` on the dust ledger, packed by `Dust::write_air_velocity`. The Niva body and the four wheels inject velocity. Vorticity confinement rolls the wake. Wind relaxes the field toward the free stream outside the solids. The solve runs on a 0.25 m × 0.50 m × 0.25 m active window around the Niva and is copied onto this 128 × 8 × 128 grid. This grid is valid after `Dust::step` / `Dust::step_pads`. Before the first air step the samples are 0.

Writer, when `dust_source` is 2 and the ledger is absent: `FluidCore::write_air_velocity`. R and B are that sheet's horizontal dust velocity, G is 0. That path does not run the body wake.

Valid: `VALID_AIR` when `dust_source` is 1 or 2, the same publishes as `dust_fog`. `dust_source` 0 leaves the binding 0 and the bit clear. A non-finite sample is written as 0.

### Crust and cracks — binding 7

Crust thickness and crack opening, both metres. Opening is the gap in the crust, not a second height and not a colour.

Writer: the soil crust state. This sheet has no thickness or opening arrays, so both channels are 0 and `VALID_CRUST` is still set. A dry-crust soil kind is not a stand-in for either channel. When those arrays exist, the same binding carries them in metres. That fill does not bump the version.

Valid: every publish, including the zeros.

### Wet / dry edge — binding 8

1 when this cell's wet flag differs from any of the four neighbours inside the window. 0 otherwise. A neighbour outside the window is ignored.

Wet means pore moisture ≥ 0.45, or water + slurry ≥ 4 mm. The flag uses the moisture and film just written into this contract.

Writer: the publisher, after moisture and film. It does not feed back into the solver.

Valid: `VALID_WET_EDGE` on every publish.

### Foam and crest — binding 9

Foam is a fraction from 0 to 1. Crest is a height in metres.

Foam is 0 when the film is thinner than 4 mm. Otherwise it is the maximum of three readings of the same sheet, each from 0 to 1:

- Froude: a smoothstep of `speed / sqrt(9.81 * thickness)` from 0.85 to 1.45. Speed 0 stays 0.
- Wave crest: a smoothstep of how far the film stands above the mean of its four in-window neighbours, from 4 mm to 18 mm.
- Tires: the wet footprint of a tire that shoved the sheet on this step, full near the contact and 0 at 1.45 tire radii.

Crest is the placed berm, metres, plus that same film excess. A flat pond adds nothing. A bow or a berm does. Foam does not use the berm. A non-finite foam or crest sample is written as 0.

Writer: the publisher, from flow, film, placed height, and the tire shoves of the step that just ran. It does not feed back into the solver. This field did not exist on the sheet before version 1.

Valid: `VALID_FOAM` on every publish. Without a fluid sheet, foam is 0. Crest from placed soil is still real.

### Spray — binding 10

Live pieces the particle step is carrying. Dust picture marks (material 8) are not in this buffer. The cubic metres of dust stay on the dust grid.

| Field | Units |
|---|---|
| `pos_mass.xyz` | world position, m |
| `pos_mass.w` | mass, kg |
| `vel_age.xyz` | world velocity, m/s |
| `vel_age.w` | age, s, `max_life - life` |
| `material` | id below |
| `alive` | 1 |

Mass is `volume_m3 * density`. A piece with no ledger volume has mass 0.

| Id | Piece | Density |
|---|---|---|
| 0 | water droplet | 1000 kg/m³ |
| 1 | flung clod | 1600 |
| 2 | drip | 1600 |
| 3 | impact chunk | 1600 |
| 4 | impact droplet | 1000 |
| 5 | slurry ribbon | 1450 |
| 6 | slurry drop | 1450 |
| 7 | slurry sheet | 1450 |
| 9 | water fleck | 1000 |

`header.spray_count` is how many of the 4096 slots are live, in particle-list order. Slots past the count are zero. Pieces beyond 4096 are not copied this frame.

Writer: the particle list, after the particle step, then the water drops the sheet has peeled and is still carrying in the air. Picture marks are not copied into that list. The peeled drops are not picture cards. A water drop is material 0, mass `volume * 1000` kg, age in seconds since it left the sheet. They are still in this buffer while the Niva leaves the ford.

Valid: `VALID_SPRAY` on every publish. A count of 0 is a valid empty frame. A non-finite position, velocity, mass, or age is written as 0.

## Header flags

`header.version` is 2. `header.valid` is a bit mask:

| Bit | Value | Map |
|---|---|---|
| 0 | `VALID_GROUND` | ground and rut |
| 1 | `VALID_MOISTURE` | moisture |
| 2 | `VALID_FILM` | water and slurry |
| 3 | `VALID_FLOW` | flow, only with the fluid sheet |
| 4 | `VALID_DUST_SETTLED` | settled dust |
| 5 | `VALID_DUST_FOG` | airborne dust |
| 6 | `VALID_CRUST` | crust and cracks |
| 7 | `VALID_WET_EDGE` | wet / dry edge |
| 8 | `VALID_FOAM` | foam and crest |
| 9 | `VALID_SPRAY` | spray |
| 10 | `VALID_AIR` | air velocity |

`header.frame` is the fluid sheet's frame counter, or 0 when the sheet is not attached. `header.time_s` is the world clock.

## Debug view

`src/render_contract.wgsl` is a top-down debug view of this bind group. It samples these maps and the spray buffer. It does not read the picture textures.

`Renderer.debug_contract` draws that view. It stays off for the Niva picture, so a contract publish does not replace a proof frame.

## Picture cache

The current Niva picture still uploads a separate cache for the mesh. That cache is not this contract. Do not add channels to it. New readers use the bindings above.

Each frame the picture cache is:

| Cache | Format | What it holds |
|---|---|---|
| height | `rg32float`, or `rgba8unorm` if the device cannot sample that | R `soil.height` (bed + placed + slurry), metres. G stamp, `height - initial`, metres. The rgba8 path packs height `(h + 1) / 3.5` and stamp `(stamp + 0.75) / 1.5` into 16-bit pairs. |
| material | `rgba8unorm` | R wetness 0..1, G `(water + slurry) / scale` with scale 0.90 or 0.34, B darken 0..1, A soil kind / 255 |
| imprint | `rgba8unorm` | R tread amplitude / 0.06 m, G heading / τ, B lateral offset mapped from ±0.40 m, A 1 when a print is present |
| chunk mesh | vertex and index buffers | triangles built from particles, not the spray slots |
| dust mesh | vertex and index buffers | triangles for dust picture cards, not `dust_fog` |
| model uniform | mud on the body | adhesion on panels, not the ground |

The picture cache covers the fixed 192 × 192 soil square, 40 m on a side, centred at the world origin. The contract window is the moving 200 × 200 sheet.

## Changelog

### 1

Initial freeze. Ground and rut, moisture, water and slurry thickness, flow, settled dust, airborne dust and fog, crust and crack opening, wet/dry edge, foam and crest, and the spray buffer. Foam and crest are new. Crust and crack channels are defined and written as 0 until the sheet stores those metres.

Dry cells and any other non-finite sample are written as 0. Layout unchanged. Foam on a published frame is the Froude term, the wave-crest lip, and the wet tire footprint, and it stays 0 where the film is under 4 mm. Spray includes the water drops the sheet peeled, still in the air as the Niva leaves the ford.

### 2

`air_velocity`, binding 11, `rgba32float` 3D, 128 × 8 × 128. R, G, B are world velocity in m/s (+X, +Y, +Z). A is 0. Same origin and cell as `dust_fog`. `VALID_AIR` is bit 10, set when `dust_source` is 1 or 2.

With the dust ledger, `step_vehicle_air` writes the field: the Niva body and wheels inject velocity, vorticity confinement leaves a wake, and wind advects the field. Without the ledger, the samples are the fluid sheet's horizontal dust velocity and a zero vertical channel. The header stays 96 bytes. Debug mode 14 shows air speed.
