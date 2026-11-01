# GPU budget and quality presets

Goal: the game runs at 60 FPS at 1920×1080 on a mid-range card. The CPU solver has its own budget in `docs/PERF_BUDGET.md` (≤ 7 ms). This file covers the GPU. The code values live in `src/quality.rs`; this file explains them.

## Reference hardware (assumption)

| Preset | Reference GPU | Target |
| --- | --- | --- |
| `low` | GTX 1050 Ti / Iris Xe / browser WebGL2 | 1080p 60 FPS on the 1050 Ti |
| `medium` | GTX 1660 / RTX 3050 / RX 6600 | 1080p 60 FPS (the main target) |
| `high` | RTX 3070 / RX 6700 XT | 1080p 60 FPS, 1440p acceptable |

The GPU frame budget is **12 ms on the preset's own reference card**. That is 16.7 ms (one 60 Hz refresh) minus driver work, compositing and present jitter. The CPU solver steps while the GPU draws the previous frame, so the CPU 7 ms and the GPU 12 ms do not add up.

Default preset: the window uses `medium`, or `low` on a software adapter. The browser uses `low` on WebGL2 and `medium` on WebGPU. Captures and proof stills stay on `high`, so the committed images do not change.

How to choose a preset: `SILT_QUALITY=low|medium|high`, `silt --quality low`, or `?quality=low` in the browser. **F4** cycles presets in the window. **F3** toggles the perf overlay.

## Per-pass budget, ms on the reference card

The labels are the render pass labels. The overlay and `silt bench` print the same labels.

| Pass | low | medium | high | What is in it |
| --- | ---: | ---: | ---: | --- |
| `shadow` | 1.2 | 1.2 | 1.0 | Sun shadow map: terrain, vehicle, props |
| `shadow-near` | — | 0.4 | 0.5 | Close shadow around the vehicle (beauty #60) |
| `planar` | — | — | 0.9 | Vehicle mirror over standing water (beauty #60) |
| `main` | 5.6 | 4.5 | 3.8 | Opaque: sky, terrain with rut geometry, vehicle, props |
| `water-ssr` | — | 0.8 | 0.9 | SSR on water (beauty #60) |
| `water` | 1.4 | 1.2 | 1.0 | Water / slurry surface, turbidity and foam (`water_detail` 0 / 1 / 2: map tint only / one flow-advected silt and foam layer / second octave, billow self-shadow, bubbles) |
| `after-water` | 1.0 | 0.8 | 0.7 | Chunks, spray, dust sprites, glass |
| `dust-volume` | 2.0 | 2.0 | 2.2 | Volumetric dust/fog ray march (at reduced resolution) |
| `dust-volume-up` | 0.3 | 0.4 | 0.4 | Upsampling the volume to full screen |
| `blit` | 0.3 | 0.3 | 0.2 | Output to the window, render-scale upsampling; tone-maps the `Rgba16Float` scene target on the way (HDR) |
| `overlay` | 0.1 | 0.1 | 0.1 | F3 overlay |
| **Sum** | 11.9 | 11.7 | 11.7 | ≤ 12 is asserted by `quality::tests::budget_rows_sum_under_frame` |

The scene target is `Rgba16Float` (linear light) where the adapter can render, blend and filter it, `Rgba8Unorm` otherwise or with `SILT_HDR=0`. The window does not pay for a separate tone-mapping pass: the blit applies the curve. Captures, `SILT_POST=1` and the contract debug view run a `tonemap` pass into an 8-bit copy instead (one full-screen read + write, ~0.1 ms at 1080p, estimate). The half-float targets and the 8-bit copy add about 50 MB of video memory at 1080p (color, scene copy, three reflect copies, 8-bit copy).

On `main`, only the `shadow`, `main`, `contract-debug`, `blit` and `overlay` passes exist today. The other rows are budgets for the passes still in progress: beauty #60, volumetric air, rut geometry #120.

## What the presets change

| Knob | low | medium | high | Used by |
| --- | ---: | ---: | ---: | --- |
| Render scale | 0.75 | 1.0 | 1.0 | `Renderer` (internal targets, blit upsamples) |
| Sun shadow, texels | 1024 | 2048 | 2048 | `Renderer` |
| Near shadow, texels | 0 | 1024 | 2048 | beauty #60 (`QUALITY_SHADOW_NEAR_RES`) |
| Fog/dust steps | 12 | 24 | 40 | volume pass (`QUALITY_FOG_STEPS`) |
| Fog sun-shadow steps | 1 | 2 | 3 | `QUALITY_FOG_SHADOW_STEPS` |
| Air layers drawn | 8 | 16 | 16 | `QUALITY_AIR_LAYERS` |
| SSR steps | 0 (off) | 16 | 32 | `QUALITY_SSR_STEPS` |
| Planar reflection | off | off | on | `QUALITY_PLANAR` |
| Rut subdivisions per 20 cm cell | 1 | 2 | 4 | rut geometry / CDLOD #120 (`QUALITY_RUT_TESS`) |
| Particles per frame | 2000 | 6000 | 12000 | particles |
| Max material texture side | 1024 | 2048 | 4096 | Niva maps (`niva.rs`) |

Shaders receive the knobs as WGSL `override` constants. The default in the WGSL is the `high` value. The pipeline passes `compilation_options.constants = quality.constants_for(src)`. That function passes only the names the module declares, because wgpu rejects a constant the shader does not have. Example:

```wgsl
override QUALITY_FOG_STEPS: i32 = 40;
for (var k: i32 = 0; k < QUALITY_FOG_STEPS; k = k + 1) { ... }
```

## Static budget in CI (`gpu-budget` job)

There is no GPU in CI. The job checks what can be counted without one:

1. `tests/gpu_budget.rs`
   * Every loop with a literal bound and every `*_STEPS` in `src/*.wgsl` must be ≤ 64 iterations per pixel. Raymarch step counts go through `override QUALITY_*`, not bigger literals.
   * `shader.wgsl` must validate in naga and translate to GLSL ES 3.00 (WebGL2) for every entry point. Each entry point may use ≤ 16 textures and ≤ 12 uniform blocks, and no storage buffers.
2. `silt budget --software`: renders frames of the demo lap at 1920×1080 on lavapipe for each preset. It counts draw calls, triangles (all passes), and the texture and buffer memory the device holds (`wgpu` `counters`).

| Preset | Draw calls | Triangles/frame | Textures, MiB | Buffers, MiB |
| --- | ---: | ---: | ---: | ---: |
| low | ≤ 400 | ≤ 2.0 M | ≤ 512 | ≤ 256 |
| medium | ≤ 500 | ≤ 3.0 M | ≤ 1024 | ≤ 384 |
| high | ≤ 700 | ≤ 5.0 M | ≤ 2560 | ≤ 512 |

Memory: `medium` must fit on a 6 GB card next to the OS and the browser. `low` must fit on a 2 GB card.

### First measurement (main 0c04f3e + this PR, lavapipe)

| Preset | Draws | Triangles | Textures, MiB | Buffers, MiB |
| --- | ---: | ---: | ---: | ---: |
| low | 29 | 14.04 M | 197 | 122 |
| medium | 29 | 14.04 M | 605 | 122 |
| high | 29 | 14.04 M | 2141 | 122 |

* **Triangles: over budget on every preset.** The 4.67 M-triangle terrain is drawn three times (shadow, main, water). PR #120 (CDLOD, 0.37–0.40 M in view) fixes this. Until it lands, CI runs with `GPU_BUDGET_ALLOW=triangles`: the row is printed but does not fail the job. Remove the allowance after #120.
* **Textures:** before this PR, every preset held 2141 MiB. The Niva has 3 arrays × 8 layers × 4096² with mips, and small maps were stretched to 4096. The preset texture-size cap brings this down to 605 MiB on `medium` and 197 MiB on `low`.
* **WebGL2:** `textureNumLevels` in `fs_mesh` has no GLSL ES 3.00 form, so naga cannot build that program for WebGL2. It is replaced by `floor(log2(side))`, which gives the same value for a full mip chain.

## Measuring on a real GPU

There is no GPU in the cloud or in CI. Real numbers come from your machine:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bench.ps1
```

```bash
scripts/bench.sh
```

The script builds release. Then, for each preset (`low`, `medium`, `high`) and scene (`lap` = the window's demo loop, `niva` = the Niva through-drive with the fluid sheet and dust), it opens a 1920×1080 window with vsync off. It runs 120 warm-up frames and 1200 measured frames with a fixed dt, so every machine draws the same frames. It writes `bench-results/<preset>-<scene>.json` and `bench-results/summary.txt`.

One run by hand:

```bash
cargo run --release -- bench --quality medium --scene niva --out bench-results/medium-niva.json
```

Lines in the report:

* `BENCH_FPS`: mean FPS and the 1% low.
* `BENCH_CPU`: wall time per frame, the solver (`sim`), and draw encoding.
* `BENCH_GPU`: GPU time per frame from timestamp queries, PASS/OVER against 12 ms (p95).
* `BENCH_PASS <label>`: each pass against its row in the table above.
* `BENCH_DRAWS`: draw calls and triangles.

If the adapter has no timestamp queries (some drivers, WebGL2), the GPU lines say `collected=0`. FPS and CPU are still measured.
