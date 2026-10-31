# Performance budgets

Release only (`cargo test --release`). Debug builds keep the air check under 4 ms and do not assert the frame or the e2e step. Do not raise these caps.

| Budget | Cap | Assert |
| --- | --- | --- |
| Solver frame | ≤ 7 ms | `tests/frame_bench.rs`, `World::demo` at 1/60, release |
| ThroughDrive solver sum | ≤ 7 ms | `tests/e2e_niva.rs` `DRIVE_SOLVER`, release. Subsystems of `ThroughDrive::step` at dt = 1/30, except the picture |
| E2e step mean | ≤ 33.004 ms | `tests/e2e_niva.rs`, `dust.step_pads` then `fluid.step` at dt = 1/30, release |
| Air step | ≤ 1.5 ms | same e2e air mean, release. Every build still requires `< 4` ms (`AIR_STEP_BUDGET_MS`) |

33.004 ms is the release CI mean on AMD EPYC 7763, 4 cores (run 41202858950, commit 8a826c5). The log prints that mean and p95; it does not print a sample median. It replaces 37.477 ms, which was a Xeon measurement of 31.231 ms plus 20 percent and left about 12 percent under the CI mean. The cap is not the 7 ms frame budget, and it is not raised when the solver sum is still above 7 ms.

A pull request that changes the solver prints, in the release CI log, the step count, the mean, p95, the hardware line, and the milliseconds of each subsystem (`DRIVE_*` from `ThroughDrive::step` at dt = 1/30).

## CI, commit 8a826c5

AMD EPYC 7763 64-Core Processor, 4 cores. Release e2e step mean 33.004 ms, p95 53.265 ms, air 0.309 ms. `ThroughDrive::step` solver sum 27.748 ms (water 8.855, slurry 7.699, soil 3.087, air 0.407, dust 4.767, restrict_fine 0.591, window 0.529, prep 0.343, sheet rest 1.470). Drive mean 28.168 ms, drive p95 38.654 ms. Solver frame 4.026 ms. Same job as the e2e line above.

## ThroughDrive::step after #97

Intel(R) Xeon(R) Processor, 4 cores. dt = 1/30. 24 warmup steps, then 96. Milliseconds are means. Before is commit 8a826c5 on this machine. After is the active 2 cm zone, the 2× dust sheet, and row-parallel transport and gravity, on `main` at 5eee5fa. `restrict_fine` and the 40 m window are their own rows. The solver sum adds every row except the picture (`other`).

| Subsystem | Before | After |
| --- | ---: | ---: |
| Water | 6.387 | 6.125 |
| Slurry | 5.546 | 4.554 |
| Soil / rut | 2.211 | 2.198 |
| Air | 0.311 | 0.301 |
| Dust | 3.600 | 2.039 |
| restrict_fine | 0.403 | 0.394 |
| Window 40 m | 0.369 | 0.345 |
| Sheet prep | 0.220 | 0.215 |
| Sheet rest | 1.118 | 0.579 |
| Solver sum | 20.164 | 16.750 |
| Step mean | 20.461 | 17.022 |
| Step p95 | 27.423 | 23.514 |

The solver sum is not under 7 ms. The 7 ms cap is unchanged and is not applied to this sum. What is left is the 2 cm sheet: water 6.125 ms and slurry 4.554 ms. Those are the CFL substeps and a forward Gauss–Seidel confinement, which stays serial so the standing slurry depth stays inside its golden. A Jacobi sweep on that confinement moved the slurry depth to 0.0503 m, outside 0.0536 ± 0.003. Dust on the 2× grid is 2.039 ms. Soil is 2.198 ms.

The 2 cm sheet is copied only for coarse cells within 0.56 m of a tire, or of the segment between the two tires on one side. The rest of the standing water stays on the 20 cm sheet. The dry middle between the ruts is not refined.

Release e2e on the same Xeon: step mean 17.921 ms (budget 33.004), p95 30.305 ms, air 0.236 ms (budget 1.5). Solver frame 2.422 ms (budget 7).

## ThroughDrive::step on this change

Base is `main` at 5eee5fa (merge of #97), not the #97 branch. Intel(R) Xeon(R) Processor, 4 cores. dt = 1/30. 24 warmup steps, then 96. The before column is that base on this machine. The after column is the `DRIVE_*` print from release `e2e_niva` on the same machine. Caps are unchanged: solver frame ≤ 7 ms, e2e step ≤ 33.004 ms, air ≤ 1.5 ms.

| Subsystem | Before | After |
| --- | ---: | ---: |
| Water | 6.040 | 0.927 |
| Slurry | 4.415 | 0.539 |
| Soil / rut | 2.208 | 0.951 |
| Air | 0.309 | 0.295 |
| Dust | 2.048 | 1.899 |
| restrict_fine | 0.395 | 0.169 |
| Window 40 m | 0.347 | 0.354 |
| Sheet prep | 0.219 | 0.209 |
| Sheet rest | 0.566 | 0.316 |
| Solver sum | 16.547 | 5.659 |
| Step mean | 16.828 | 5.916 |

The open 2 cm sheet takes one CFL pass and lets flat, slow tiles sleep. It skips the small divergence projection; a wall, the coarse sheet, and gravity keep that projection. Confinement on that open sheet is the same snapshot Jacobi the row workers run, on the calling thread, with the vorticity stored once per cell. Open wind does not build the pressure field the wind force never reads. The halved dust volume steps one layer per worker; each layer starts from a zero scratch, and the ground-sheet scratch is restored when the step returns. The body in the e2e and the creek stays on the forward Gauss–Seidel sweep. A Jacobi sweep there moved standing slurry to 0.0503 m, outside 0.0536 ± 0.003, and it is not restored.

Release e2e on this Xeon: slurry depth 0.0555 m, puddle depth 0.0893 m, ledger gap 0.000492 m³. Step mean 16.509 ms (budget 33.004), p95 31.104 ms, air 0.235 ms (budget 1.5). Drive p95 8.104 ms. Solver frame 2.505 ms (budget 7). A second timing of the same 96 steps read solver sum 5.695 ms.

## Fog on the shared air grid

Base is `main` at 396af7e. Intel(R) Xeon(R) Processor, 4 cores. Caps are unchanged: solver frame ≤ 7 ms, ThroughDrive solver sum ≤ 7 ms, e2e step mean ≤ 33.004 ms, air ≤ 1.5 ms. Fog is not injected on the e2e or the frame bench. An empty fog grid is one compare, outside the air timer.

Release `e2e_niva` on this machine: step mean 14.041 ms, p95 28.049 ms, air 0.247 ms. `ThroughDrive::step` at dt = 1/30, 96 steps after warmup: water 0.823, slurry 0.541, soil 0.974, air 0.296, dust 1.034, restrict_fine 0.206, window 0.360, prep 0.209, sheet rest 0.301, solver sum 4.745 ms. Drive mean 5.010 ms, drive p95 6.117 ms. Solver frame 2.438 ms.

Release `fog_air`, Niva, dt = 1/60, 78 timed steps with water lofted into fog: step mean 0.976 ms, step max 1.688 ms. An earlier pass of the same binary read 1.015 ms mean and 2.156 ms max. That mean is the fog step, and it stays under the 7 ms solver-frame cap. The fluid-sheet fog case closed the water ledger to 8.55e-6 m³.
