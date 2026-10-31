# Performance budgets

Release only (`cargo test --release`). Debug builds keep the air check under 4 ms and do not assert the frame or the e2e step. Do not raise these caps.

| Budget | Cap | Assert |
| --- | --- | --- |
| Solver frame | ≤ 7 ms | `tests/frame_bench.rs`, `World::demo` at 1/60, release |
| E2e step mean | ≤ 37.477 ms | `tests/e2e_niva.rs`, `dust.step_pads` then `fluid.step` at dt = 1/30, release |
| Air step | ≤ 1.5 ms | same e2e air mean, release. Every build still requires `< 4` ms (`AIR_STEP_BUDGET_MS`) |

37.477 ms is a measured 31.231 ms plus 20 percent. It is not the 7 ms frame cap.

A pull request that changes the solver prints, in the release CI log, the step count, the mean, p95, the hardware line, and the milliseconds of each subsystem (`DRIVE_*` from `ThroughDrive::step` at dt = 1/30).

## ThroughDrive::step on this change

Intel(R) Xeon(R) Processor, 4 cores. dt = 1/30. 24 warmup steps, then 96. Milliseconds are means. `restrict_fine` and the 40 m window are their own rows. The solver sum adds every row except the picture (`other`).

| Subsystem | Before | After |
| --- | ---: | ---: |
| Water | 8.805 | 6.387 |
| Slurry | 7.573 | 5.546 |
| Soil / rut | 2.169 | 2.211 |
| Air | 0.296 | 0.311 |
| Dust | 5.783 | 3.600 |
| restrict_fine | 0.389 | 0.403 |
| Window 40 m | 5.303 | 0.369 |
| Sheet prep | 0.212 | 0.220 |
| Sheet rest | 1.162 | 1.118 |
| Solver sum | 31.693 | 20.164 |
| Step mean | 32.014 | 20.461 |
| Step p95 | 40.864 | 27.423 |

The solver sum is not under 7 ms. The 7 ms cap is unchanged and is not applied to this sum. What is left is the 2 cm sheet under the tires: water 6.387 ms, slurry 5.546 ms, and the dust planes 3.600 ms. That is the CFL substeps, Gauss-Seidel confinement, and the open-sheet flux. `restrict_fine` is 0.403 ms. The 40 m window was a full-grid hash on every cell shift; an integer scroll brought it to 0.369 ms (0 steps fell back to the hash).

Release e2e on the same machine: step mean 24.229 ms (budget 37.477), air 0.254 ms (budget 1.5). Solver frame 2.467 ms (budget 7).
