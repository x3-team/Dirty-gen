# Ground geometry: LOD tree, height between samples, fine track

This is how the drawn ground gets its vertices and its height. It does not
shape relief. Every height comes from the solver: the 20 cm soil heightfield
today, and the fine track field once the solver publishes it.

## What was there

- One uniform grid, 8 vertices per 20 cm soil cell over the 40 m square:
  1529 × 1529 vertices, 4.67 M triangles, drawn in the shadow, ground, and
  water passes. 2.6 cm between vertices everywhere, near or far.
- Height between solver samples was bilinear. A rut one cell wide was a V of
  two flat facets, and a berm a tent ridge.
- The solver's 2 cm patch exists only inside `fluid_core` (`fine_bed` is the
  coarse bed copied down), and the contract publishes only the 20 cm window.
  So there is no finer rut geometry to draw yet, whatever the mesh density.

## What this adds

### LOD tree (`src/terrain_lod.rs`, `src/terrain_draw.rs`)

CDLOD (Strugar 2009). One shared 32 × 32 quad patch, instanced per node of a
quadtree over a 64 m root. Level 0 has a 1/64 m (1.56 cm) vertex step; each
level doubles it. A level-k step is drawn out to 256 steps from the camera:

| Level | Step | Out to |
|---|---|---|
| 0 | 1.56 cm | 4 m |
| 1 | 3.1 cm | 8 m |
| 2 | 6.3 cm | 16 m |
| 3 | 12.5 cm | 32 m |
| 4 | 25 cm | beyond |

In the outer 30 % of each ring the odd vertices morph onto the next coarser
grid, so there is no step and no crack where levels meet. All sizes are
powers of two in metres, so a vertex shared by two nodes is the same float on
both sides. `shared_edges_close` checks that on every seam, including seams
between levels. Nodes outside the camera are culled; the shadow list is
culled to the light's box.

The shadow pass picks its own list (not culled to the camera) with the same
ring as the view. A coarser shadow ring (96 steps) was tried: wherever the
fine track has relief, the coarse shadow ground sits above or below the
drawn one and the ruts fill with blocky self-shadow. It stays a knob.

On the proof frames: 4.67 M → 0.37–0.40 M ground triangles in view (ground
and water pass each), and 4.67 M → 1.26–1.32 M in the shadow pass. Ground
across all three passes: 14.0 M → about 2.1 M on `high`.

### Fine track filtering

The fine field is 2.5 cm; a LOD vertex 10 cm out reads it 4 texels apart and
would land on lugs and gaps at random (far ruts read as speckle). On upload
the field gets 5 box-filtered levels (2.5 to 40 cm). A LOD vertex reads the
level that matches its step, a ground fragment the level that matches its
pixel footprint (`tg_footprint`). The vertex step under the camera
goes from 2.6 cm to 1.56 cm.

### Height between solver samples (`src/terrain_geom.wgsl`)

Monotone bicubic (Fritsch–Butland tangents) through the 20 cm samples. It
passes through every solver value, never overshoots them, and is C1 across
cells. A one-cell rut is a rounded groove and a berm a rounded crest, with
the same floor depth and crest height the solver stored. Used for the
vertex height and for the shading normal, so the two agree.

### Fine track slot

Group 1, binding 10: `track_fine_tex`, rgba32float. Binding 11: the geometry
uniform with the window (origin, cell, res) and a valid flag.
`TerrainGeometry::set_track_fine(queue, window, texels)` uploads a publish;
`clear_track_fine()` turns it off. While no field is published, the ground
is exactly the 20 cm reconstruction.

Reading the field (agreed with the solver thread, numbers pending the
contract):

- R: surface offset in metres over contract ground (bed + placed, binding 1
  R) at the fine texel centre, lugs and snow included. The whole window is
  valid; untouched cells carry the virgin surface.
- `set_track_fine` takes a `ground(x, z)` callback (contract ground,
  bilinear) and uploads ground + R, so the drawn surface does not depend on
  what the 20 cm height map holds (the snow capture keeps the snow top there).
- G snow depth, B compaction, A lug depth: not geometry; left for shading.
- Inside the window the ground is that surface, feathered over 25 cm into
  the cubic 20 cm ground at the window edge.
- The renderer allocates 640 × 640 (`TRACK_FINE_RES`). Change that constant
  if the contract settles on another size.

Level 0 is 1.56 cm, finer than the proposed 2.5 cm cell, so the field gets
at least one vertex per texel out to 4 m.

## Knobs

| Env | Default | |
|---|---|---|
| `SILT_TERRAIN_LOD` | 1 | 0 draws the old uniform grid (built only then) |
| `SILT_TERRAIN_CUBIC` | 1 | 0 is bilinear between samples |
| `SILT_TERRAIN_DEBUG` | 0 | 1 tints by level and draws the triangle grid |
| `SILT_TERRAIN_LOD_RANGE` | 256 | ring radius in vertex steps (≥ 64) |
| `SILT_TERRAIN_SHADOW_RANGE` | = view | shadow pass ring radius (≥ 32) |

Quality presets (`silt::quality::active().rut_tess`, docs/GPU_BUDGET.md)
pick the ring: low (1) 128, medium (2) 192, high (4) 256 steps, the same
for view and shadow. The env knobs override the preset. `silt budget`
counts the ground's instanced draws.

## Proof

`silt capture --software --terrain-proof <dir>` draws the force-rut scene
(two hard-pass ruts in wet clay beside the Niva) from three cameras, each as
before (uniform grid, bilinear), after (LOD, cubic), and lod (debug).

- `rut-along-before-after.jpg`, `rut-across-before-after.jpg`,
  `rut-mid-before-after.jpg`: left before, right after.
- `rut-along-lod.jpg`, `rut-mid-lod.jpg`: levels (red 0, yellow 1, green 2,
  cyan 3, blue 4) with the triangle grid.

The lumps along the berm in the debug view are the solver's per-cell berm,
not the mesh. Smoother berms and finer rut walls need the fine track field.
