# Fields the beauty scenes need that contract v2 does not carry

The beauty pass reads only the maps in `docs/RENDER_CONTRACT.md`. Where a scene
needs something the contract does not publish, the shader does not draw it.
This file lists those fields, the layout the shaders are written against, and
what is drawn from them. The readers are in `src/beauty_contract.wgsl`
(`bc_snow`, `bc_soil_mix`, stubs in `beauty_nocontract.wgsl`); the terrain hook is in
`fs_terrain` (`src/shader.wgsl`, "Requested contract fields"). Until the
header sets the bits, both readers return 0 and the hook does nothing.

All maps share the ground grid: 200 x 200, the header's `origin_x/z` and `cell_m`.

## Scene 1, mud: soil seams

Contract v2 has no soil type. The kind map in the picture cache is a single id
per 0.20 m cell, which steps at every seam, and it is not part of the contract.

Requested: **binding 13 `soil_mix`**, `rgba32float`, fractions of
R mud, G grass, B sand, A gravel in the cell (0..1, sum <= 1).
Header bit 12, `VALID_SOIL_MIX`.

Drawn from it: the ground layers blended by the fractions (a seam is a few
cells of mix, not a step), and the existing moisture (binding 2) darkens and
glosses the mud side of a seam. The wet edge at a seam needs nothing new.

## Scene 2, snow: rut, berms, belly strip, tracks

Contract v2 has no snow. The snow ruts still today comes from the snow scene's
own height and compaction arrays written into the picture cache, not from the
contract.

Requested: **binding 12 `snow`**, `rgba32float`:
R snow depth over the ground (m), G density (kg/m3),
B compaction 0..1 (fresh 0, packed by a tread 1), A belly scrape depth (m).
Header bit 11, `VALID_SNOW`. The rut and berm shape stays in binding 1
(ground = bed + placed, as for mud) so the mesh and the water need no change.

Drawn from it: fresh powder white and rough, packed tread grey with a satin
sheen, the belly strip as polished packed snow and ice, tracks from compaction.

## Scene 4, crust: thickness and cracks

Binding 7 exists and `VALID_CRUST` is set, but both channels are 0 (the sheet
has no thickness or opening arrays). The crust still therefore shows only the
drying from moisture; plates and cracks are not drawn.

Needed: binding 7 filled as documented, R crust thickness (m),
G crack opening (m). No layout change.

## Scenes 3 and 5

Sand (wet/dry, shallows) and dust (settled b5, airborne b6, air velocity b11)
are covered by contract v2.
