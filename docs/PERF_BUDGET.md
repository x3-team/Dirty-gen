# Performance budgets

Release only (`cargo test --release`). Debug builds keep the air check under 4 ms and do not assert the frame or the e2e step. Do not raise these caps.

| Budget | Cap | Assert |
| --- | --- | --- |
| Solver frame | ≤ 7 ms | `tests/frame_bench.rs`, `World::demo` at 1/60, release |
| ThroughDrive solver sum | ≤ 7 ms | `tests/e2e_niva.rs` `DRIVE_SOLVER`, release. Subsystems of `ThroughDrive::step` at dt = 1/30, except the picture |
| E2e step mean | ≤ min(1.15 × same-CPU calibration, 33.004 ms) | `tests/e2e_niva.rs`, `dust.step_pads` then `fluid.step` at dt = 1/30, release |
| ThroughDrive dust | ≤ 1.0 ms | `DRIVE_DUST` in that release log |
| ThroughDrive step p95 | ≤ 7 ms | `DRIVE_P95` in that release log |
| Air step | ≤ 1.5 ms | same e2e air mean, release. Every build still requires `< 4` ms (`AIR_STEP_BUDGET_MS`) |

The e2e budget is 15% over a calibration from the same CPU, then capped at 33.004 ms. EPYC 7763 uses 18.6 ms (budget 21.390). EPYC 9V45 uses 10.278 ms from run 41218969837, commit 0cf4957 (budget 11.820). EPYC 9V74 uses 15.152 ms from run 41216401571, commit a172a1b (budget 17.425). Any other model uses the 7763 calibration, so a 9V45 number is not applied there. The log prints `E2E_CALIBRATION`, `E2E_BUDGET`, the mean, and p95. It does not print a sample median. The cap is not the 7 ms frame budget.

The drive p95 tail was the 40 m window parking every pad cell. `pull_substrate` writes the flat wetness onto the whole sheet, and each forward step inserted that strip into `parked`. The map doubled through 16384 and 32768 during the timed run and rehashed on that step. Cells that only hold the pad moisture are not inserted; a wet track still is, and a cell that enters without a record is written back to that pad. Gravity on this drive stays at one CFL substep. Open wind stays on Courant 0.42 and keeps the divergence projection. The airborne dust sheet is the full 128² plane; a halved sheet moves the published ledger. On AVX2 that donor cell runs eight lanes at a time; each lane is one independent row or column. The table under “ThroughDrive::step on this change” is the earlier Xeon measurement, including its halved dust sheet.

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
