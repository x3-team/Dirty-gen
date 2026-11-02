# Third-party code

Parts of the beauty shaders (`src/shader.wgsl`) are ported from the open
projects below. The WGSL was rewritten for our bindings: Bevy's `.wesl`
imports and `@if` variants were expanded by hand, and only the pieces
listed here were taken.

## Bevy — bevy_pbr

- Source: https://github.com/bevyengine/bevy, `crates/bevy_pbr/src/render/`
  (`shadow_sampling.wesl`, `pbr_lighting.wesl`, `utils.wesl`) and
  `crates/bevy_pbr/src/light_probe/environment_map.wesl`, commit
  `ae6152152d6fb8308ad6038314b48ef295e80fbc`.
- License: MIT OR Apache-2.0. Copyright (c) Bevy Contributors.
- Used for: PCSS sun shadows (blocker search on the D3D sample points,
  Jimenez '14 spiral with interleaved gradient noise), `D_GGX`,
  `V_SmithGGXCorrelated`, `F_Schlick`, `Fd_Burley`, `F_AB`, specular
  multiscatter compensation, and the environment-map (IBL) light with its
  multiscatter split and `radiance_sample_direction`.

## bevy_water

- Source: https://github.com/Neopallium/bevy_water,
  `assets/shaders/water_functions.wgsl`, `assets/shaders/noise/*.wgsl`,
  commit `da5fc110edf2d031831e72794f3793e44efcd8f9`.
- License: MIT OR Apache-2.0. Copyright (c) Robert G. Jakabosky. The fbm
  follows Inigo Quilez and Munrocket (MIT).
- Used for: the wave pattern, value noise and fbm behind the puddle ripple
  normals and the caustics.

## WebGL Water

- Source: https://github.com/evanw/webgl-water, `renderer.js`, commit
  `73eda8be832b649367b25ea5690c1f0181bb56ad`.
- License: MIT. Copyright 2011 Evan Wallace.
- Used for: caustics as the area ratio of light refracted through the
  surface onto the bed, and refraction of the view ray onto the bed.

## wgpu examples

- Source: https://github.com/gfx-rs/wgpu, `examples/features/src/shadow`,
  `examples/features/src/water`, commit
  `2375b1cd83da57b4f70acd8db590524cfeb49890`.
- License: MIT OR Apache-2.0. Copyright (c) The gfx-rs developers.
- Consulted for the shadow and water pipeline layout; no code copied yet.
