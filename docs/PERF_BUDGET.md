# Performance budgets

Release only (`cargo test --release`). Debug builds keep the air check under 4 ms and do not assert the frame or the e2e step. Do not raise these caps.

| Budget | Cap | Assert |
| --- | --- | --- |
| Solver frame | ≤ 7 ms | `tests/frame_bench.rs`, `World::demo` at 1/60, release |
| E2e step mean | ≤ 33.004 ms | `tests/e2e_niva.rs`, `dust.step_pads` then `fluid.step` at dt = 1/30, release |
| Air step | ≤ 1.5 ms | same e2e air mean, release. Every build still requires `< 4` ms (`AIR_STEP_BUDGET_MS`) |

33.004 ms is the release CI mean on AMD EPYC 7763, 4 cores (run 41202858950, commit 8a826c5). The log prints that mean and p95; it does not print a sample median. It replaces 37.477 ms, which was a Xeon measurement of 31.231 ms plus 20 percent and left about 12 percent under the CI mean. The cap is not the 7 ms frame budget, and it is not raised when the solver sum is still above 7 ms.

A pull request that changes the solver prints, in the release CI log, the step count, the mean, p95, the hardware line, and the milliseconds of each subsystem (`DRIVE_*` from `ThroughDrive::step` at dt = 1/30).

## CI, commit 8a826c5

AMD EPYC 7763 64-Core Processor, 4 cores. Release e2e step mean 33.004 ms, p95 53.265 ms, air 0.309 ms. `ThroughDrive::step` solver sum 27.748 ms (water 8.855, slurry 7.699, soil 3.087, air 0.407, dust 4.767, restrict_fine 0.591, window 0.529, prep 0.343, sheet rest 1.470). Drive mean 28.168 ms, drive p95 38.654 ms. Solver frame 4.026 ms. Same job as the e2e line above.

## ThroughDrive::step on this change

Intel(R) Xeon(R) Processor, 4 cores. dt = 1/30. 24 warmup steps, then 96. Milliseconds are means. Before is commit 8a826c5 on this machine. After is the active 2 cm zone, the 2× dust sheet, and row-parallel transport and gravity. `restrict_fine` and the 40 m window are their own rows. The solver sum adds every row except the picture (`other`).

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
