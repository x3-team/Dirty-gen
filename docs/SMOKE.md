# Smoke and fire

A world-anchored 3D grid on wgpu compute (`src/smoke.rs`, `src/smoke.wgsl`)
and a fullscreen ray march over it (`src/smoke_pass.rs`,
`src/smoke_volume.wgsl`, pass label `smoke-volume`). Off unless a scene
enables it. Needs compute, so there is no smoke on WebGL2 (`low` in the
browser).

Why not the dust/fog air window (`dust.rs`, PR #127): that window rides with
the Niva, is 4 m tall and has no pressure solve. A smoke column stands still
in the world, is taller, and needs pressure and vorticity to roll. The two
grids do not share data. When the dust/fog march lands on main, the smoke
texture joins that loop (`air_sigma` + an emission term) and this pass goes
away; the bind group already carries what that needs (density, emission,
origin, cell).

## Sources (all MIT, licences checked 2026-11-01)

| Base | What is taken |
| --- | --- |
| GPU Gems 3 ch. 30 (Crane, Llamas, Tariq), as ported in Scrawk/GPU-GEMS-3D-Fluid-Simulation | Kernel structure: advect, vorticity confinement, divergence, Jacobi pressure, project; collocated grid |
| doyubkim/fluid-engine-dev (Jet) `GridSmokeSolver` | Buoyancy from temperature and soot, decay of both |
| PavelDoGreat/WebGL-Fluid-Simulation | Reference for confinement strength and dissipation |
| Bevy `volumetric_fog.wgsl` (MIT/Apache, already approved) | March: Henyey-Greenstein phase, Beer-Lambert, energy-conserving step |
| Planck's law + CIE 1931 fit of Wyman, Sloan, Shirley (JCGT 2013) | Flame colour (`smoke::blackbody`), computed, no code taken |

No Shadertoy code, no BoatAttack.

## Solver

Per step (one compute pass, `smoke-sim`): semi-Lagrangian velocity,
MacCormack scalars with the min/max clamp, curl, forces (fuel burn into heat
and soot, sources with jitter, Boussinesq buoyancy `g/T_amb · ΔT − w · soot`,
vorticity confinement, relax to wind, speed cap), divergence, 24 Jacobi
sweeps (warm start), project, publish. Sides and top are open (wind, no soot,
p = 0); the ground is a wall.

The publish kernel writes one `rgba16float` 3D texture: soot, ΔT / 1000 K,
burn rate, and the sun transmittance marched per cell (so the ray march needs
no shadow loop of its own).

Known limits:

* 24 compact Jacobi sweeps remove about half of the divergence (test
  `smoke_column_rises_in_still_air`); the rest is long-wavelength.
* At 8 cm cells the column is softer than the UE5 reference; sub-cell detail
  is a rising noise erosion in the march, not resolved flow.

## Tests

`cargo test --release --test smoke -- --nocapture` (lavapipe is picked when
`lvp_icd.json` exists; skips without an adapter): the column rises over 2.5 m
in 3 s and stays over the vent in still air, wind leans it, a fire burns
more than 600 K over ambient and makes soot, no NaN.

## Bench

`bench/scenes.txt`: `smoke-column` (preset `smoke`, two cold vents, 8 s,
against ue5-89) and `fire` (preset `fire`, open flame, 6 s, no UE5 ref).

## GPU budget (estimate, not measured: no GPU on the cloud box)

Grid for the bench smoke scene: 60 × 76 × 60 = 274k cells, 8 cm.

| Pass | medium (GTX 1660) | How |
| --- | ---: | --- |
| `smoke-sim` | ~1.0 ms per step, ~0.5 ms per frame at a 30 Hz step | ~32 dispatches, ~400 B DRAM traffic per cell |
| `smoke-volume` | ~1.5 ms full res today, ~0.5 ms at half res | 64 steps, one filtered 3D fetch + 2 noise octaves per step |

The 12 ms frame (docs/GPU_BUDGET.md, PR #126) is already 11.7 ms full, so
smoke has to come out of the `dust-volume` row: a scene with smoke draws it
in the same half-resolution march. Until that merge, `smoke-volume` runs at
full resolution and is for stills and the bench. Knobs: grid cell,
`jacobi_iters`, and the `QUALITY_SMOKE_STEPS` / `SMOKE_SHADOW_STEPS`
overrides (WGSL defaults 64 / 24, both within the 64-step lint).
