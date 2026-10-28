# Silt assets

The motocross picture uses the Lada Niva and the photoreal forest. The dune buggy and the Gobkit kit stay in the tree so the older oval frame still builds. They are not drawn on the course.

## Vehicles

- vehicle/dune-buggy.glb — Grab3D https://grab3d.com/models/vehicles/dune-buggy/ (full GLB: /dl/dune-buggy/dune-buggy.glb). CC0 1.0. See vehicle/LICENSE.txt. Hidden on the course.
- vehicle/niva-greg/lada_niva.glb — Lada Niva by Greg McKechnie, https://sketchfab.com/3d-models/lada-niva-26ebf5872bce4ff4ad2baee41882d6b2. CC BY 4.0. Meshopt and quantization were flattened for the renderer; textures stayed WebP. Changes do not remove the attribution requirement.

## Ground (CC0)

- ground/{mud_forest,brown_mud_dry,rocky_gravel,grass_ground}/ — Poly Haven (https://polyhaven.com/a/<id>, via api.polyhaven.com/files), 2K JPG: diff, nor_gl, rough, ao, disp, arm. CC0. https://polyhaven.com/license
- ground/clay_001/ — ambientCG Clay001 2K-JPG (https://ambientcg.com/view?id=Clay001). CC0.

The course blends those five packs by soil state: dry dirt on the racing line, mud in the hole and the slope puddle, clay in the rut floors, gravel on the start and the table, grass off the ribbon, rock in the far corners.

## Forest and rocks (CC0, Poly Haven models, 1K glTF)

https://polyhaven.com/license — CC0. No Quixel / Megascans (that license does not allow shipping the source assets in this public repo). No Kenney.

- env/photoreal/island_tree.glb — island_tree_02. Scanning: Rob Tuytel. Cleanup: Rico Cilliers. https://polyhaven.com/a/island_tree_02. Shipped with the kit. The course does not scatter it: the leaf cards break into specks at lap scale. The canopy is searsia, fern, and sorrel.
- env/photoreal/searsia.glb — searsia_lucida. Modeling: James Ray Cock. Photography: Jenelle van Heerden. https://polyhaven.com/a/searsia_lucida
- env/photoreal/fern.glb — fern_02. Rob Tuytel, Rico Cilliers. https://polyhaven.com/a/fern_02
- env/photoreal/shrub.glb — shrub_sorrel_01. Rico Cilliers. https://polyhaven.com/a/shrub_sorrel_01
- env/photoreal/boulder.glb — boulder_01. Rico Cilliers. https://polyhaven.com/a/boulder_01
- env/photoreal/rock_moss.glb — rock_moss_set_01. Kless Gyzen. https://polyhaven.com/a/rock_moss_set_01
- env/photoreal/grass.glb — grass_medium_01. Rob Tuytel, Rico Cilliers. https://polyhaven.com/a/grass_medium_01
- env/photoreal/dead_trunk.glb — dead_tree_trunk_02. Jenelle van Heerden, Rico Cilliers. https://polyhaven.com/a/dead_tree_trunk_02

Simplified with gltf-transform where the download was too heavy to ship. Leaf cards that were black-backed JPEGs get their alpha from that background at load. Fern and sorrel stay opaque because their JPEGs have no usable mask.

## Gobkit (CC0, not the hero forest)

- env/*.glb — Gobkit Nature Kit, from https://gobkit.com/api/free (gobkit.com/freebies/environment/). CC0. The showcase scene is not included. Low-poly and stylized, so the course frame does not instance these trees. Grass001.glb is still the unused prop atlas placeholder.
