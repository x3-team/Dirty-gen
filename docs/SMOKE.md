# Smoke and fire

A world-anchored 3D grid on wgpu compute (`src/smoke.rs`, `src/smoke.wgsl`)
and a fullscreen ray march over it (`src/smoke_pass.rs`,
`fs_fog` in `src/beauty_contract.wgsl`, group 5 of the dust march, pass label `dust-volume`). Off unless a scene
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
| stegu/psrdnoise `psrdnoise3.wgsl` (Gustavson, McEwan, MIT) | `sdnoise3`, simplex noise with analytic gradient, copied with its licence header into `beauty_contract.wgsl` |
| Bridson et al. 2007, curl noise | Divergence-free detail: curl of three offset `sdnoise3` |
| Kim et al. 2008, wavelet turbulence | Detail amplitude from the unresolved eddy speed the solver publishes, Kolmogorov 2^(-5/6) per octave |
| Wrenninge 2015, multiple-scattering octaves | Two extra sun octaves at half extinction so thick smoke is bright |

No Shadertoy code, no BoatAttack.

## Solver

Per step (one compute pass, `smoke-sim`): semi-Lagrangian velocity,
MacCormack scalars with the min/max clamp, curl, forces (fuel burn into heat
and soot, sources with jitter, Boussinesq buoyancy `g/T_amb · ΔT − w · soot`,
vorticity confinement, relax to wind, speed cap), divergence, 24 Jacobi
sweeps (warm start), project, publish. Sides and top are open (wind, no soot,
p = 0); the ground is a wall.

The publish kernel writes two `rgba16float` 3D textures: soot, ΔT / 1000 K,
burn rate, and the sun transmittance marched per cell (so the ray march needs
no shadow loop of its own); and velocity plus the unresolved eddy speed
|curl v|·h that scales the sub-grid detail.

Known limits:

* 24 compact Jacobi sweeps remove about half of the divergence (test
  `smoke_column_rises_in_still_air`); the rest is long-wavelength.
* Sub-cell detail is not resolved flow. Kim 2008 advects a finer density grid
  with the synthesized eddies; here the eddies displace the density lookup in
  the march (`QUALITY_SMOKE_DETAIL` octaves, default 2), so the grid stays at
  8 cm. Over time the detail rides with the local velocity.

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
| `smoke-volume` | ~2.5 ms full res today, ~0.8 ms at half res | 64 steps; per step in smoke: 3 filtered 3D fetches + 6 simplex noises (2 curl octaves) |

The 12 ms frame (docs/GPU_BUDGET.md, PR #126) is already 11.7 ms full, so
smoke has to come out of the `dust-volume` row: a scene with smoke draws it
in the same half-resolution march. Until that merge, `smoke-volume` runs at
full resolution and is for stills and the bench. Knobs: grid cell,
`jacobi_iters`, and the `QUALITY_SMOKE_STEPS` / `QUALITY_SMOKE_DETAIL` /
`SMOKE_SHADOW_STEPS` overrides (WGSL defaults 64 / 2 / 24, within the 64-step
lint). Detail 1 octave on `medium` and 0 on `low` would cut the march cost
roughly in half and to a third.
