# Beauty readers for the contract fields in Cursor's open PRs

The beauty pass reads only the maps in `docs/RENDER_CONTRACT.md`. Bindings and
valid bits are assigned by the solver PRs below; the shaders use those and
define none of their own. Each reader checks the header version
(`RENDER_CONTRACT_VERSION`) and its valid bit and returns 0 until the solver
sets it. Readers: `src/beauty_contract.wgsl` (stubs in `beauty_nocontract.wgsl`).

| PR | Version | Field | Binding | Bit | Reader | State in beauty |
|---|---|---|---|---|---|---|
| #99 | 3 | turbidity | 4 A (flow becomes rgba32float) | 1 << 11 | `bc_turbidity` | live: reads b4.A, murk in the water only where slurry or turbidity > 0 |
| #105 | 4 | snow height, density, compaction | from #105 | from #105 | `bc_snow` | dark: #105 still names binding 12 / bit 11, which #106 and #99 hold |
| #106 | 5 | water fog, 3D on the air grid | 12 | 1 << 13 | `bc_fog_water` | dark until binding 12 is in main's layout |
| #107 | 6 | soil shares R mud, G sand, B gravel, A grass | 13 | 1 << 12 | `bc_soil_mix` | dark until binding 13 is in main's layout; terrain blend written for that order |
| #103 | 2 | crust thickness, crack opening | 7 | 1 << 6 | `bc_crust` | reads b7 already; plates and cracks show once the channels are non-zero |
| #104 | 2 | foam | 9 | 1 << 8 | `bc_foam_crest` / `bc_foam_soft` | reads b9 already |

What the hooks draw once live (terrain, "Requested contract fields" in
`src/shader.wgsl`): soil shares blend the ground layers per cell, with the mud
side of a joint wet and glossy from binding 2; snow covers the ground by height,
greys and glosses with compaction, the underbody strip as packed snow.

Open question for #105: which binding and bit the snow maps take once #106 (binding
12, bit 13) and #99 (bit 11) are in.
