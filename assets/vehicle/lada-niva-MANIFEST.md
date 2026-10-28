# Lada Niva — Greg McKechnie

## Status
**DOWNLOADED**

The demo loads `lada_niva-hq.glb` (meshopt, original WebP maps). The 2K slim package is not shipped.

## Source
| Field | Value |
|--------|--------|
| Title | Lada Niva |
| Author | Greg McKechnie (`@mckechniegreg6`) |
| Sketchfab URL | https://sketchfab.com/3d-models/lada-niva-26ebf5872bce4ff4ad2baee41882d6b2 |
| UID | `26ebf5872bce4ff4ad2baee41882d6b2` |
| ArtStation (renders) | https://www.artstation.com/artwork/48nvZ4 |
| License | **CC BY 4.0** (Creative Commons Attribution 4.0) — https://creativecommons.org/licenses/by/4.0/ |
| Commercial use | Allowed with attribution |
| Approx. polycount | ~75 k tris / ~51 k verts (upload); renderVertexCount ~225 k |
| Downloadable (API) | `true` |

## Attribution (credits string)
```
Lada Niva by Greg McKechnie (https://sketchfab.com/3d-models/lada-niva-26ebf5872bce4ff4ad2baee41882d6b2) — CC BY 4.0
```

## Files in this folder
| Path | Size | Notes |
|------|------|--------|
| `lada_niva-hq.glb` | 22 905 392 bytes (~21.8 MiB) | Shipped mesh. Meshopt, WebP 4096 on the body, tires, chassis, and interior from the original maps. The GLB is not recompressed. |

## HQ package
- Extensions: `EXT_meshopt_compression`, `EXT_texture_webp`, `KHR_mesh_quantization`
- Body, tire, chassis, and interior WebP stay 4096 in the GLB.
- The renderer box-filters those maps into a 1024 array. The texels are averaged, not recolored, and the GLB is not recompressed.
- Glass, accessories, and the needle are smaller and are repeated onto that array without a filter.
- The demo does not repaint the body.
