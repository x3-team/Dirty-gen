# Spike: water and slurry sheet on wgpu compute

Status: draft, not for merge. Off by default: `SILT_SHEET_GPU=1` routes the
water and slurry sheets of `FluidCore` through `src/gpu_sheet.rs`.

## What was ported

`fluid_core::substep` for the gravity drive, both paths:

* walled sheet (wheels and body in the water): free-surface lift on solid
  cells, no-slip ghost, gravity with the wall mobility boost, confinement,
  divergence filter, redirect around walls, no-penetration, banked transport;
* open sheet (2 cm under the tires and the 20 cm window without solids).

19 WGSL kernels, one thread per cell, whole grid (no work boxes, no sleeping
tiles). Dust and fog (wind drive) are not ported.

Transport is a gather with the same per-face order as the CPU scatter. The CPU
`room` clamp never binds (two faces of at most 0.45 each), so a face moves the
same dh on both sides and volume is conserved exactly as on the CPU.

The one scheme change is vorticity confinement: snapshot Jacobi everywhere.
The CPU walled sheet uses a forward Gauss–Seidel sweep, which has no
order-equivalent on a GPU. A red–black sweep was tried and did not change the
result (0.0501 m slurry).

## Results (cloud box, Xeon 2.8 GHz 4 cores, GPU = lavapipe)

`tests/gpu_sheet.rs`, 90 steps at 1/30 from the same start, CPU vs GPU:

| Scene | Volume gap GPU / CPU | Depth diff mean / max |
| --- | --- | --- |
| walled water 200² | 4.45e-6 / 4.44e-6 | 0.053 mm / 0.69 mm |
| walled slurry 200² | 4.96e-6 / 4.92e-6 | 0.028 mm / 0.39 mm |
| open 2 cm 160² | 2.15e-5 / 1.89e-5 | 0.18 mm / 77 mm (one cell, both clamp at 1.6 m/s) |

`tests/e2e_niva.rs` goldens with `SILT_SHEET_GPU=1`:

| Golden | Band | CPU | GPU | CPU with Jacobi confinement |
| --- | --- | --- | --- | --- |
| puddle depth | 0.0893 ± 0.003 | 0.0893 | 0.0893 | 0.0893 |
| tire hole | 0.0020 ± 0.002 | 0.0021 | 0.0016 | 0.0016 |
| bow head | 0.0403 ± 0.003 | 0.0402 | 0.0403 | 0.0403 |
| slurry depth | 0.0536 ± 0.003 | 0.0555 | **0.0500 (out)** | 0.0500 |
| slurry rut | 0.0128 ± 0.003 | 0.0128 | 0.0132 | 0.0132 |
| ledger gap | < 1e-3 m³ | 0.00050 | 0.00053 | — |
| water / slurry / soil books | ± 1e-3 | ok | ok | — |

The GPU run and a CPU run with Jacobi confinement print the same numbers to
four digits, including settled dust 0.015548. The port is faithful; the only
golden that moves is the standing slurry, and it moves because of the sweep
order, as `docs/PERF_BUDGET.md` already recorded (0.0503).

Timing on lavapipe (a CPU rasterizer on the same 4 cores, so not a GPU
number) per step: walled water CPU 3.9 ms / GPU drop-in 8.7 ms (upload 0.32,
readback 0.37), open 2 cm CPU 1.7 / GPU 4.0. Neither the cloud box nor the
GitHub runners have a GPU. Real numbers need a machine with one:

    cargo test --release --test gpu_sheet -- --ignored --nocapture --test-threads=1

## Recommendation

No-go as a drop-in replacement now; go later as a GPU-resident solver at
higher resolution.

* The sheet costs water 0.93 + slurry 0.54 ms on the CI Xeon (`ThroughDrive`).
  A drop-in pays an upload and a blocking readback four times a frame (two
  sheets × two grids). One submit-and-map round trip is already 0.2–1 ms on a
  discrete GPU, so drop-in cannot beat the CPU at 200² and 160².
* A win needs the whole `FluidCore` on the GPU (tire pushes, refine/restrict,
  hydro load, spray, ledger) with small async readbacks a frame late. That is
  a rewrite measured in weeks, and it pays off when the grid grows 4–16×
  (for example 5 cm cells over the 40 m window), which is where the look of
  the UE5 reference comes from.
* The browser build runs on WebGL2, which has no compute shaders. A GPU solver
  needs WebGPU or a kept CPU path. CI runners have no GPU, so CI perf caps
  cannot gate a GPU path.
* The slurry golden encodes the Gauss–Seidel order. Moving to the GPU means
  re-baselining that one number (0.0500) or retuning slurry confinement.
