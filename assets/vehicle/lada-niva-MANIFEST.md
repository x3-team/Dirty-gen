# Lada Niva — Greg McKechnie

## Status
**DOWNLOADED**

Original GLB on disk; slim package ready for CloudAgent transfer (single file ≤25 MB).

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
| `lada_niva.glb` | 190 464 544 bytes (~181.6 MiB) | Original Sketchfab GLB — **do not delete** |
| `lada_niva-slim.glb` | 9 626 456 bytes (~9.18 MiB) | Transfer package: WebP ≤2048, meshopt, hierarchy preserved |

## Slim compression
```
npx --yes @gltf-transform/cli optimize \
  lada_niva.glb lada_niva-slim.glb \
  --texture-compress webp \
  --texture-size 2048 \
  --compress meshopt \
  --meshopt-level high \
  --join false \
  --flatten false \
  --simplify false \
  --palette false
```
- Tool: `@gltf-transform/cli` 4.5.1  
- Result: **190.46 MB → 9.63 MB** (~19.8×)  
- Extensions required in slim: `EXT_meshopt_compression`, `EXT_texture_webp`, `KHR_mesh_quantization`  
- Split archive: not needed (single file under 25 MB)

## CloudAgent transfer notes
- Limits: ≤25 MB/file, ≤46 MB/call.
- Ready for **one** reply with a single attachment.
- Suggested attachments:
  - `file:///workspace/silt-assets/vehicle/niva-greg/lada_niva-slim.glb`
  - (optional) `file:///workspace/silt-assets/vehicle/niva-greg/MANIFEST.md`

## Target project
Project Dirty gen: `bc-2f1d8ac8-9cfa-48e7-b1ed-3090c0c4926e`  
Ready to hand off: **YES** (slim prepared; CloudAgent not launched from this step).
