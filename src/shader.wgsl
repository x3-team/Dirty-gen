// Silt forward shading. Soil height is a real displacement; mud on the buggy
// is a mask driven by the adhesion state, not a flat tint.

struct Frame {
    view_proj: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    amb: vec4<f32>,
    params: vec4<f32>, // width, height, cell, map res
    misc: vec4<f32>,   // time, shadow size, _, _
    // Four tires. Even slot: center.xyz, radius.
    // Odd slot: forward.x, forward.z, sink, active.
    tires: array<vec4<f32>, 8>,
    // Turn crest. snow_arc2.y is the bin count; 0 leaves the heightfield alone.
    // snow_arc: center.x, center.z, sweep, r_lo
    // snow_arc2: dr, bins, theta in, theta out
    snow_arc: vec4<f32>,
    snow_arc2: vec4<f32>,
    snow_profile: array<vec4<f32>, 8>,
    // Irradiance / pi of the env map as nine SH terms (see env.rs).
    sh: array<vec4<f32>, 9>,
    env: vec4<f32>,    // env mips, IBL gain, sky is an HDRI, exposure
    // Niva headlamps: left xyz + intensity, right xyz + sand-field flag,
    // beam dir + outer cone cos.
    lamps: array<vec4<f32>, 3>,
    // Air, picture only. 0: wind xyz m/s, ground fog 1/m. 1: Niva position,
    // fog scale height. 2: Niva velocity, yaw.
    air: array<vec4<f32>, 3>,
    // Near shadow cascade round the Niva, right half of the shadow atlas.
    light_vp_near: mat4x4<f32>,
    // Water surface y at each tire (soil + solved film); < -100 unknown.
    tire_water: vec4<f32>,
    // x: 1 = depth debug.
    debug: vec4<f32>,
    // Last frame's view-projection (SSR history reprojection).
    prev_view_proj: mat4x4<f32>,
    // x: SSR history valid. y, z: half-res SSR target size.
    ssr_info: vec4<f32>,
    reflect_view_proj: mat4x4<f32>,
    reflect_flags: vec4<f32>,
    // x: water detail (0 low, 1 medium, 2 high). y: 1 with a fluid sheet.
    // w: 1 when the scene target is linear HDR (see `present`).
    water_fx: vec4<f32>,
}

struct Model {
    world: mat4x4<f32>,
    mud0: vec4<f32>,
    mud1: vec4<f32>,
    mud2: vec4<f32>,
    mud3: vec4<f32>,
    mud4: vec4<f32>,
    mud5: vec4<f32>,
    mud6: vec4<f32>,
    mud7: vec4<f32>,
    mud8: vec4<f32>,
    mud9: vec4<f32>,
    mud10: vec4<f32>,
    mud11: vec4<f32>,
    // x: 0 body, 1 procedural tire, 2 Niva carcass.
    // y: sidewall coat on a Niva tire, wheel mud on the others.
    // z: groove fill. w: lug smear.
    extra: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var env_tex: texture_2d<f32>;
@group(0) @binding(2) var env_samp: sampler;

// Post group: scene depth and a copy of the opaque frame (water and dust
// volume pipelines only; meshes bind the model uniform at group 2).
@group(2) @binding(1) var post_depth: texture_2d<f32>;
@group(2) @binding(2) var post_color: texture_2d<f32>;
// Tileable curl (rgb) and billow (a) noise for the dust and fog detail.
@group(2) @binding(3) var air_noise: texture_3d<f32>;
@group(2) @binding(4) var air_samp: sampler;
// Half-res water SSR. In the water pass: this frame's trace. In the SSR
// pass itself: last frame's, for the temporal blend.
// A: rgb premultiplied by the hit weight, a = weight.
// B: x = view depth, y/z = normal xz, w = 1 where the sheet was traced.
@group(2) @binding(5) var ssr_a_tex: texture_2d<f32>;
@group(2) @binding(6) var ssr_b_tex: texture_2d<f32>;
// The Niva mirrored about the water level (planar reflection, render.rs),
// alpha 0 where nothing was drawn.
@group(2) @binding(7) var planar_tex: texture_2d<f32>;

@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var mat_tex: texture_2d<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var shadow_tex: texture_2d<f32>;

@group(1) @binding(4) var ground_diff: texture_2d_array<f32>;
@group(1) @binding(5) var ground_nor: texture_2d_array<f32>;
@group(1) @binding(6) var ground_pack: texture_2d_array<f32>;
@group(1) @binding(7) var ground_samp: sampler;
@group(1) @binding(8) var imprint_tex: texture_2d<f32>;
// R: crust thickness / 0.04 m. G: crack opening / 0.02 m. Not a color.
@group(1) @binding(9) var crust_tex: texture_2d<f32>;
@group(1) @binding(12) var scene_color: texture_2d<f32>;
@group(1) @binding(13) var scene_depth: texture_2d<f32>;
@group(1) @binding(14) var scene_samp: sampler;
@group(1) @binding(15) var pool_reflect_tex: texture_2d<f32>;
// Contract turbidity (R, sqrt), foam (G), flow vx (B), vz (A) on the soil grid.
@group(1) @binding(16) var water_fx_tex: texture_2d<f32>;

@group(2) @binding(0) var<uniform> model: Model;

@group(3) @binding(0) var buggy_base: texture_2d<f32>;
@group(3) @binding(1) var buggy_nor: texture_2d<f32>;
@group(3) @binding(2) var buggy_mr: texture_2d<f32>;
@group(3) @binding(3) var tire_base: texture_2d<f32>;
@group(3) @binding(4) var tire_nor: texture_2d<f32>;
@group(3) @binding(5) var prop_tex: texture_2d<f32>;
@group(3) @binding(6) var paint_samp: sampler;
@group(3) @binding(7) var niva_base: texture_2d_array<f32>;
@group(3) @binding(8) var niva_nor: texture_2d_array<f32>;
@group(3) @binding(9) var niva_mr: texture_2d_array<f32>;

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Water surface noise, ported from bevy_water (Neopallium/bevy_water,
// assets/shaders/water_functions.wgsl and noise/*.wgsl; MIT/Apache-2.0,
// the fbm after Inigo Quilez, MIT; see THIRD_PARTY.md). Scaled from ocean
// swell down to a puddle: the crests run about a metre apart and the fbm
// breaks them into centimetre ripples.
fn bw_random2d(v: vec2<f32>) -> f32 {
    return fract(sin(dot(v, vec2<f32>(12.9898, 78.233))) * 43758.5453123);
}

fn bw_vnoise2d(v: vec2<f32>) -> f32 {
    let i = floor(v);
    let f = fract(v);
    let a = bw_random2d(i);
    let b = bw_random2d(i + vec2<f32>(1.0, 0.0));
    let c = bw_random2d(i + vec2<f32>(0.0, 1.0));
    let d = bw_random2d(i + vec2<f32>(1.0, 1.0));
    let u = smoothstep(vec2<f32>(0.0), vec2<f32>(1.0), f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

fn bw_fbm(v2: vec2<f32>) -> f32 {
    let m2 = mat2x2<f32>(vec2<f32>(0.8, 0.6), vec2<f32>(-0.6, 0.8));
    var p = v2;
    var f = 0.5000 * bw_vnoise2d(p); p = m2 * p * 2.02;
    f = f + 0.2500 * bw_vnoise2d(p); p = m2 * p * 2.03;
    f = f + 0.1250 * bw_vnoise2d(p); p = m2 * p * 2.01;
    f = f + 0.0625 * bw_vnoise2d(p);
    return f / 0.9375;
}

fn bw_wave(p: vec2<f32>, t: f32) -> f32 {
    let time = t * 0.5 + 23.0;
    let wave_y = cos(p.y / 5.0 + time / 0.5);
    let wave_x = smoothstep(1.0, 0.0, abs(sin(p.x / 2.0 + wave_y + time)));
    return wave_x + bw_fbm(p) / 2.0 - 1.0;
}

// Two layers scrolling against each other along the wind, as bevy_water's
// sample_directional_wave at quality 2.
fn bw_wave_height(xz: vec2<f32>, t: f32) -> f32 {
    let w = frame.air[0].xz;
    let dir = select(vec2<f32>(0.8, 0.6), normalize(w), dot(w, w) > 1e-4);
    let p = xz * PUDDLE_WAVE_SCALE;
    let r = vec2<f32>(-(p.x * dir.x + p.y * dir.y), p.y * dir.x - p.x * dir.y);
    let time = t / 2.0;
    return (bw_wave((r - time) * 0.3, t) * 0.3 + bw_wave((r + time) * 0.4, t) * 0.3) * PUDDLE_WAVE_AMP;
}

const PUDDLE_WAVE_SCALE: f32 = 40.0;
const PUDDLE_WAVE_AMP: f32 = 0.003;

fn bw_wave_normal(xz: vec2<f32>, t: f32) -> vec3<f32> {
    let delta = 0.012;
    let h = bw_wave_height(xz, t);
    let hx = bw_wave_height(xz + vec2<f32>(delta, 0.0), t);
    let hz = bw_wave_height(xz + vec2<f32>(0.0, delta), t);
    return normalize(vec3<f32>(h - hx, delta, h - hz));
}

// Caustics after WebGL Water (evanw/webgl-water, renderer.js; MIT): light
// refracted through the rippled surface lands on the bed, and a patch of
// bed gets brighter by the ratio of the area it would have under a flat
// surface to the area the bent rays squeeze it into. WebGL Water takes the
// areas from screen derivatives of a projected mesh; here the map is
// differenced in world space so it works in any control flow.
fn caustic_land(xz: vec2<f32>, depth: f32, t: f32) -> vec2<f32> {
    let sun = normalize(frame.sun_dir.xyz);
    let ray = refract(-sun, bw_wave_normal(xz, t), 1.0 / 1.333);
    return xz + ray.xz * (depth / max(-ray.y, 0.2));
}

fn webgl_water_caustic(xz: vec2<f32>, depth: f32, t: f32) -> f32 {
    let e = 0.01;
    let c = caustic_land(xz, depth, t);
    let ax = caustic_land(xz + vec2<f32>(e, 0.0), depth, t) - c;
    let az = caustic_land(xz + vec2<f32>(0.0, e), depth, t) - c;
    let new_area = abs(ax.x * az.y - ax.y * az.x);
    return clamp(e * e / max(new_area, 1e-8), 0.0, 6.0);
}

fn mud_of(panel: i32) -> vec4<f32> {
    if (panel == 0) { return model.mud0; }
    else if (panel == 1) { return model.mud1; }
    else if (panel == 2) { return model.mud2; }
    else if (panel == 3) { return model.mud3; }
    else if (panel == 4) { return model.mud4; }
    else if (panel == 5) { return model.mud5; }
    else if (panel == 6) { return model.mud6; }
    else if (panel == 7) { return model.mud7; }
    else if (panel == 8) { return model.mud8; }
    else if (panel == 9) { return model.mud9; }
    else if (panel == 10) { return model.mud10; }
    else if (panel == 11) { return model.mud11; }
    return vec4<f32>(0.0);
}

// Filmic curve (Narkowicz ACES fit). The old 1 - exp(-x) flattened every
// highlight into the same pale tan, so wet gloss never read as gloss.
// Input is already exposed (see `expose`). `fs_tonemap` in render.rs is the
// same expression.
fn aces_fit(x_in: vec3<f32>) -> vec3<f32> {
    let x = max(x_in, vec3<f32>(0.0));
    let m = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return pow(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn expose(c: vec3<f32>) -> vec3<f32> {
    return max(c, vec3<f32>(0.0)) * frame.env.w * 0.78;
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    return aces_fit(expose(c));
}

// Scene colour target (docs/LOOK_GAP.md, thread 1). With an HDR target
// (`water_fx.w` = 1, `Rgba16Float`) the scene shaders write exposed linear
// light and the tone curve runs once, in the blit or the tonemap pass.
// Without one (an adapter that cannot render and blend half floats, or
// SILT_HDR=0) they write the display value into the 8-bit target.
fn scene_hdr() -> bool {
    return frame.water_fx.w > 0.5;
}

fn display_curve(c: vec3<f32>) -> vec3<f32> {
    return tonemap(c);
}

fn present(c: vec3<f32>) -> vec4<f32> {
    if (scene_hdr()) {
        return vec4<f32>(expose(c), 1.0);
    }
    return vec4<f32>(tonemap(c), 1.0);
}

// A colour authored as the final display value (particles, debug tints,
// fog composited in display space) into what the scene target holds:
// the inverse of `aces_fit`, so the tonemap pass gives the same display
// value back.
fn from_display(d: vec3<f32>) -> vec3<f32> {
    if (!scene_hdr()) {
        return d;
    }
    return aces_inverse(d);
}

// Inverse of `aces_fit` (display value -> exposed linear light).
fn aces_inverse(d: vec3<f32>) -> vec3<f32> {
    let m = min(pow(clamp(d, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2)), vec3<f32>(0.999));
    let qa = vec3<f32>(2.51) - 2.43 * m;
    let qb = vec3<f32>(0.03) - 0.59 * m;
    let qc = -0.14 * m;
    return (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, vec3<f32>(0.0)))) / (2.0 * qa);
}

// The old exp curve a few water paths still grade with, as a display value.
fn legacy_curve(c: vec3<f32>, k: f32) -> vec3<f32> {
    let mapped = vec3<f32>(1.0) - exp(-max(c, vec3<f32>(0.0)) * k);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

// A sample of the scene colour copy (SSR, refraction, planar pass) as the
// display value the water shading was tuned on.
fn scene_to_display(c: vec3<f32>) -> vec3<f32> {
    if (scene_hdr()) {
        return aces_fit(c);
    }
    return c;
}

// Equirect lookup. Matches texel_dir in env.rs.
fn env_uv(d: vec3<f32>) -> vec2<f32> {
    let dir = normalize(d);
    var phi = atan2(dir.z, dir.x);
    if (phi < 0.0) {
        phi = phi + 6.2831853;
    }
    let theta = acos(clamp(dir.y, -1.0, 1.0));
    return vec2<f32>(phi / 6.2831853, theta / 3.14159265);
}

// Prefiltered radiance. Roughness picks a box-filtered level of the map.
fn env_radiance(d: vec3<f32>, rough: f32) -> vec3<f32> {
    let lod = clamp(rough * 7.5, 0.0, max(frame.env.x - 2.0, 0.0));
    return textureSampleLevel(env_tex, env_samp, env_uv(d), lod).rgb;
}

// Diffuse sky light from the SH terms, already divided by pi.
fn sh_irradiance(n: vec3<f32>) -> vec3<f32> {
    let s = frame.sh;
    var e = s[0].rgb * 0.282095;
    e += s[1].rgb * (0.488603 * n.y);
    e += s[2].rgb * (0.488603 * n.z);
    e += s[3].rgb * (0.488603 * n.x);
    e += s[4].rgb * (1.092548 * n.x * n.y);
    e += s[5].rgb * (1.092548 * n.y * n.z);
    e += s[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0));
    e += s[7].rgb * (1.092548 * n.x * n.z);
    e += s[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// Split-sum environment BRDF, Karis' analytic fit.
fn env_brdf(f0: vec3<f32>, rough: f32, ndv: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = rough * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + vec3<f32>(ab.y);
}

fn ggx_d(ndh: f32, a: f32) -> f32 {
    let a2 = a * a;
    let d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * d * d);
}

fn sky_color(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let h = clamp(dir.y, -0.05, 1.0);
    var col = mix(
        frame.sky_horizon.rgb,
        frame.sky_zenith.rgb,
        select(smoothstep(-0.02, 0.55, h), smoothstep(-0.12, 0.78, h), win_demo_flat_terrain()),
    );
    let sun = normalize(frame.sun_dir.xyz);
    let d = max(dot(dir, sun), 0.0);
    col += frame.sun_color.rgb * pow(d, 900.0) * 4.0;
    col += frame.sun_color.rgb * pow(d, 18.0) * 0.55;
    let p = dir.xz / (abs(dir.y) * 0.45 + 0.25);
    let n = noise(p * 0.18);
    let cloud = smoothstep(0.58, 0.82, n) * smoothstep(0.08, 0.4, dir.y);
    col = mix(col, vec3<f32>(0.93, 0.94, 0.96), cloud * 0.7);
    return col;
}

fn decode_depth(c: vec4<f32>) -> f32 {
    return dot(c, vec4<f32>(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));
}

fn encode_depth(d: f32) -> vec4<f32> {
    var enc = vec4<f32>(1.0, 255.0, 65025.0, 16581375.0) * clamp(d, 0.0, 0.9999);
    enc = fract(enc);
    enc = enc - enc.yzww * vec4<f32>(1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0, 0.0);
    return enc;
}

// Percentage-closer soft shadows, ported from Bevy's bevy_pbr
// shadow_sampling.wesl (MIT/Apache-2.0, see THIRD_PARTY.md): a blocker search
// on the D3D sample points, then the Jimenez '14 spiral rotated per pixel by
// interleaved gradient noise. Our map is packed depth, not a comparison
// texture, so the hardware 2x2 compare is done by hand.
const PCSS_D3D: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(0.125, -0.375), vec2<f32>(-0.125, 0.375), vec2<f32>(0.625, 0.125), vec2<f32>(-0.375, -0.625),
    vec2<f32>(-0.625, 0.625), vec2<f32>(-0.875, -0.125), vec2<f32>(0.375, 0.875), vec2<f32>(0.875, -0.875),
);
const PCSS_SPIRAL: array<vec2<f32>, 8> = array<vec2<f32>, 8>(
    vec2<f32>(-0.7071, 0.7071), vec2<f32>(0.0, -0.875), vec2<f32>(0.5303, 0.5303), vec2<f32>(-0.625, 0.0),
    vec2<f32>(0.3536, -0.3536), vec2<f32>(0.0, 0.375), vec2<f32>(-0.1768, -0.1768), vec2<f32>(0.125, 0.0),
);

fn interleaved_gradient_noise(px: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * px.x + 0.00583715 * px.y));
}

// One cascade is a SHADOW x SHADOW tile of the atlas starting at column x0.
fn shadow_depth_at(c: vec2<i32>, lo: vec2<i32>, hi: vec2<i32>) -> f32 {
    return decode_depth(textureLoad(shadow_tex, clamp(c, lo, hi), 0));
}

// Lit fraction from a bilinear 2x2 compare, as a comparison sampler would.
fn shadow_cmp_2x2(texel: vec2<f32>, z: f32, lo: vec2<i32>, hi: vec2<i32>) -> f32 {
    let t = texel - 0.5;
    let b = vec2<i32>(floor(t));
    let f = t - floor(t);
    let l00 = select(1.0, 0.0, z > shadow_depth_at(b, lo, hi));
    let l10 = select(1.0, 0.0, z > shadow_depth_at(b + vec2<i32>(1, 0), lo, hi));
    let l01 = select(1.0, 0.0, z > shadow_depth_at(b + vec2<i32>(0, 1), lo, hi));
    let l11 = select(1.0, 0.0, z > shadow_depth_at(b + vec2<i32>(1, 1), lo, hi));
    return mix(mix(l00, l10, f.x), mix(l01, l11, f.x), f.y);
}

// PCSS in one cascade. Returns -1 when the point is off that cascade.
fn shadow_cascade(world: vec3<f32>, n: vec3<f32>, m: mat4x4<f32>, tile: i32, margin: f32) -> f32 {
    let lp = m * vec4<f32>(world, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (any(uv < vec2<f32>(margin)) || any(uv > vec2<f32>(1.0 - margin)) || ndc.z < 0.0 || ndc.z > 1.0) {
        return -1.0;
    }
    let atlas = vec2<i32>(textureDimensions(shadow_tex));
    let size = atlas.y;
    let dims = vec2<f32>(f32(size));
    let lo = vec2<i32>(tile * size, 0);
    let hi = lo + vec2<i32>(size - 1);
    let texel = uv * dims + vec2<f32>(f32(lo.x), 0.0);
    // Flat ground can shadow a few centimetres of rut. A wall keeps a larger bias.
    let slope = clamp(1.0 - n.y, 0.0, 1.0);
    // Texel size and depth range in metres, read back from the ortho matrix.
    let texel_m = 2.0 / (length(vec3<f32>(m[0].x, m[1].x, m[2].x)) * dims.x);
    let depth_m = 1.0 / max(length(vec3<f32>(m[0].z, m[1].z, m[2].z)), 1e-6);
    // Bias in metres, from the texel footprint: a low sun spreads one texel
    // over a long run of ground and would print the map's grid.
    let low_sun = 1.0 - smoothstep(0.3, 0.7, normalize(frame.sun_dir.xyz).y);
    let bias_m = texel_m * (1.5 + slope * 6.0 + low_sun * 4.0) + 0.004;
    let z = ndc.z - bias_m / depth_m;
    // The sun disc and the bright sky round it blur a shadow by ~2 cm per
    // metre of gap. Search as wide as the tallest gap the car can make.
    let light_size = 0.02;
    let search = clamp(1.6 * light_size / max(texel_m, 1e-5), 2.0, 16.0);
    var blk = vec2<f32>(0.0);
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let s = shadow_depth_at(vec2<i32>(floor(texel + PCSS_D3D[k] * search)), lo, hi);
        blk += select(vec2<f32>(0.0), vec2<f32>(s, 1.0), z > s);
    }
    if (blk.y == 0.0) {
        return 1.0;
    }
    let gap = (z - blk.x / blk.y) * depth_m;
    let blur = max(gap * light_size / max(texel_m, 1e-5), 0.5) * 2.0;
    // Per-pixel rotation from where this point lands on screen.
    let cp = frame.view_proj * vec4<f32>(world, 1.0);
    let px = (cp.xy / cp.w * 0.5 + 0.5) * max(frame.params.xy, vec2<f32>(1.0));
    let ang = 6.2831853 * interleaved_gradient_noise(px);
    let rot = mat2x2<f32>(cos(ang), -sin(ang), sin(ang), cos(ang));
    // Two turns of the spiral, a quarter turn apart: sixteen taps, so a still
    // frame without temporal accumulation stays quiet.
    let rot2 = mat2x2<f32>(0.7071, -0.7071, 0.7071, 0.7071) * rot;
    var lit = 0.0;
    for (var k: i32 = 0; k < 8; k = k + 1) {
        lit += shadow_cmp_2x2(texel + rot * PCSS_SPIRAL[k] * blur, z, lo, hi);
        lit += shadow_cmp_2x2(texel + rot2 * PCSS_SPIRAL[k] * blur * 0.8, z, lo, hi);
    }
    return lit / 16.0;
}

fn shadow_factor(world: vec3<f32>, n: vec3<f32>) -> f32 {
    // Near cascade first, faded into the far one over its outer rim.
    var lit = shadow_cascade(world, n, frame.light_vp, 0, 0.0);
    let near = shadow_cascade(world, n, frame.light_vp_near, 1, 0.02);
    if (near >= 0.0) {
        let lp = frame.light_vp_near * vec4<f32>(world, 1.0);
        let e = max(abs(lp.x), abs(lp.y)) / lp.w;
        let w = 1.0 - smoothstep(0.8, 0.96, e);
        lit = select(near, mix(lit, near, w), lit >= 0.0);
    }
    if (lit < 0.0) {
        return 1.0;
    }
    // 0.92 so the airborne buggy and the clods throw a shadow you can see.
    return 1.0 - (1.0 - lit) * 0.92;
}

// Does a ray from `o` along `d` hit the Niva's body box within `t_max`?
// The water under the car mirrors its dark underside and sills, not the
// sky, wherever the screen-space march misses them.
fn niva_box_hit(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> bool {
    let yaw = frame.air[2].w;
    let fwd = vec2<f32>(sin(yaw), cos(yaw));
    let rel = o - frame.air[1].xyz;
    // Local frame: x across the car, y up, z along it.
    let lo = vec3<f32>(rel.x * fwd.y - rel.z * fwd.x, rel.y, dot(rel.xz, fwd));
    let ld = vec3<f32>(d.x * fwd.y - d.z * fwd.x, d.y, dot(d.xz, fwd));
    let half_ext = vec3<f32>(0.8, 0.45, 1.8);
    let inv = 1.0 / select(ld, vec3<f32>(1e-6), abs(ld) < vec3<f32>(1e-6));
    let t0 = (-half_ext - lo) * inv;
    let t1 = (half_ext - lo) * inv;
    let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return tn <= tf && tf > 0.0 && tn < t_max;
}

// Distance along a ray to the nearest tire, as a cylinder round the axle
// 22 cm wide; -1 on a miss. The mirror image of the rubber in the water.
fn tire_ray_hit(o: vec3<f32>, d: vec3<f32>, t_max: f32) -> f32 {
    var best = -1.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let f = frame.tires[i * 2 + 1];
        let fwd = normalize(vec2<f32>(f.x, f.y) + vec2<f32>(1e-5, 0.0));
        let axis = vec3<f32>(-fwd.y, 0.0, fwd.x);
        let rel = o - c.xyz;
        // Project out the axle: a circle problem in the wheel plane.
        let rp = rel - axis * dot(rel, axis);
        let dp = d - axis * dot(d, axis);
        let a = dot(dp, dp);
        let b = dot(rp, dp);
        let cc = dot(rp, rp) - c.w * c.w;
        let disc = b * b - a * cc;
        if (a < 1e-6 || disc < 0.0) {
            continue;
        }
        let t = (-b - sqrt(disc)) / a;
        if (t > 0.0 && t < t_max && abs(dot(rel + d * t, axis)) < 0.11 && (best < 0.0 || t < best)) {
            best = t;
        }
        // The flat sidewalls: from the side most mirror rays meet a disc,
        // not the tread.
        let da = dot(d, axis);
        if (abs(da) > 1e-4) {
            for (var sgn: i32 = -1; sgn <= 1; sgn = sgn + 2) {
                let tc = (f32(sgn) * 0.11 - dot(rel, axis)) / da;
                let pc = rel + d * tc;
                let radial = pc - axis * dot(pc, axis);
                if (tc > 0.0 && tc < t_max && dot(radial, radial) < c.w * c.w && (best < 0.0 || tc < best)) {
                    best = tc;
                }
            }
        }
    }
    return best;
}

// Sky occlusion on the ground under the vehicle: the body hides most of the
// sky from the soil below it and each tire hides it round its contact patch.
// This is what seats the car; the sun shadow alone reads as a decal.
fn niva_ground_ao(world: vec3<f32>) -> f32 {
    let yaw = frame.air[2].w;
    let fwd = vec2<f32>(sin(yaw), cos(yaw));
    // The footprint is pushed along the sun by the sill height (~0.35 m),
    // so the dark sits where the body's shadow falls, not round it.
    let sun_s = normalize(frame.sun_dir.xyz);
    let shift = -sun_s.xz / max(sun_s.y, 0.3) * 0.35;
    let d = world.xz - frame.air[1].xz - shift;
    let l = vec2<f32>(d.x * fwd.y - d.y * fwd.x, dot(d, fwd));
    // Signed distance to the 1.6 x 3.6 m footprint.
    let q = abs(l) - vec2<f32>(0.8, 1.8);
    let sd = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
    let height = max(world.y - (frame.air[1].y - 0.9), 0.0);
    // Dense inside, gone 30 cm past the edge; densest under the sills.
    let sill = 1.0 - smoothstep(0.0, 0.35, abs(abs(l.x) - 0.8));
    // Soft edge: under clear shallows a tight edge printed a box on the bed.
    var ao = 1.0 - mix(0.72, 0.9, sill) * (1.0 - smoothstep(-0.25, 0.3, sd)) * (1.0 - smoothstep(0.3, 1.2, height));
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let contact = c.xyz - vec3<f32>(0.0, c.w, 0.0);
        let dd = world - contact;
        let r2 = dot(dd.xz, dd.xz) + dd.y * dd.y * 4.0;
        ao *= 1.0 - 0.55 * exp(-r2 / 0.05);
    }
    return ao;
}

// Aerial perspective: a thin haze that warms toward the sun, then the old
// far fog so the horizon still melts into the sky.
fn atmosphere(world: vec3<f32>, col: vec3<f32>) -> vec3<f32> {
    let to = world - frame.cam_pos.xyz;
    let dist = length(to);
    let dir = to / max(dist, 1e-4);
    let sun = normalize(frame.sun_dir.xyz);
    let haze = 1.0 - exp(-dist * 0.0032);
    let toward = pow(max(dot(dir, sun), 0.0), 6.0);
    let air = mix(frame.sky_horizon.rgb * 0.55, frame.sun_color.rgb * 0.5, toward * 0.6);
    var out = mix(col, air, haze * 0.45);
    let fog = smoothstep(58.0, 150.0, dist);
    let fog_col = frame.sky_horizon.rgb * 0.28 + frame.sky_zenith.rgb * 0.12;
    out = mix(out, fog_col, fog * 0.65);
    return out;
}

const LAMP_COLOR: vec3<f32> = vec3<f32>(1.0, 0.86, 0.64);
const LAMP_POWER: f32 = 9.0;

// Beam of one headlamp at a world point: cone weight over distance squared,
// and the unit direction the light travels.
fn lamp_beam(i: i32, world: vec3<f32>) -> vec4<f32> {
    let to = world - frame.lamps[i].xyz;
    let d2 = max(dot(to, to), 0.09);
    let l = to * inverseSqrt(d2);
    let outer = frame.lamps[2].w;
    let cone = smoothstep(outer, mix(outer, 1.0, 0.6), dot(l, frame.lamps[2].xyz));
    return vec4<f32>(l, cone / d2);
}

// Diffuse and GGX from both headlamps. Zero when the lamps are off.
fn lamp_light(world: vec3<f32>, n: vec3<f32>, view: vec3<f32>, diff_col: vec3<f32>, f0: vec3<f32>, rough: f32) -> vec3<f32> {
    let on = frame.lamps[0].w;
    if (on <= 0.0) {
        return vec3<f32>(0.0);
    }
    var col = vec3<f32>(0.0);
    let a = rough * rough;
    for (var i: i32 = 0; i < 2; i = i + 1) {
        let b = lamp_beam(i, world);
        let ldir = -b.xyz;
        let ndl = clamp(dot(n, ldir), 0.0, 1.0);
        if (ndl <= 0.0 || b.w <= 0.0) {
            continue;
        }
        let h = normalize(ldir + view);
        let ndv = max(dot(n, view), 1e-3);
        let fs = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - clamp(dot(view, h), 0.0, 1.0), 5.0);
        let spec = ggx_d(clamp(dot(n, h), 0.0, 1.0), a) * fs / max(4.0 * ndv, 1e-3);
        // Only smooth, wet surfaces glint; dry gravel under a near lamp sparkled white.
        let glossy = 1.0 - smoothstep(0.3, 0.7, rough);
        col += (diff_col * ndl + min(spec, vec3<f32>(8.0)) * ndl * glossy) * b.w;
    }
    return col * LAMP_COLOR * LAMP_POWER * on;
}

// GGX sun plus image-based sky. `ao` darkens only the sky terms; the sun has
// its own shadow map. Sun color is in the old units (Lambert * pi * 1.25).
// Lighting below is Bevy's bevy_pbr (pbr_lighting.wesl, environment_map.wesl;
// MIT/Apache-2.0, see THIRD_PARTY.md), unrolled for one sun and our env map.
// Our light units leave out the 1/pi of a Lambert lobe, so the BRDFs are
// scaled by pi to keep the exposure the scenes were tuned at.
fn D_GGX(roughness: f32, NdotH: f32) -> f32 {
    let oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    let a = NdotH * roughness;
    let k = roughness / (oneMinusNdotHSquared + a * a);
    return k * k * (1.0 / 3.14159265);
}

fn V_SmithGGXCorrelated(roughness: f32, NdotV: f32, NdotL: f32) -> f32 {
    let a2 = roughness * roughness;
    let lambdaV = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    let lambdaL = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    return 0.5 / (lambdaV + lambdaL);
}

fn F_Schlick_vec(f0: vec3<f32>, f90: f32, VdotH: f32) -> vec3<f32> {
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

fn F_Schlick(f0: f32, f90: f32, VdotH: f32) -> f32 {
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

fn Fd_Burley(roughness: f32, NdotV: f32, NdotL: f32, LdotH: f32) -> f32 {
    let f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    return F_Schlick(1.0, f90, NdotL) * F_Schlick(1.0, f90, NdotV) * (1.0 / 3.14159265);
}

fn F_AB(perceptual_roughness: f32, NdotV: f32) -> vec2<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = perceptual_roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    return max(vec2<f32>(-1.04, 1.04) * a004 + r.zw, vec2<f32>(0.00005));
}

// Frostbite listing 22: rough lobes lean toward the normal.
fn radiance_sample_direction(N: vec3<f32>, R: vec3<f32>, roughness: f32) -> vec3<f32> {
    let smoothness = clamp(1.0 - roughness, 0.0, 1.0);
    return mix(N, R, smoothness * (sqrt(smoothness) + roughness));
}

// Set by the terrain under water: light reaching the bed came through the
// rippled surface, so the shadow lookup is bent by the ripples.
var<private> g_shadow_wobble: vec3<f32> = vec3<f32>(0.0);
// Under water: x = extra penumbra radius (m), y = light the turbid column
// scatters into the shadow (0..1).
var<private> g_shadow_soft: vec2<f32> = vec2<f32>(0.0);

// Shadow with a wider penumbra: four taps round the point, then lifted by
// the light scattered into it.
fn shadow_soft(p: vec3<f32>, n: vec3<f32>) -> f32 {
    var s = shadow_factor(p, n);
    let r = g_shadow_soft.x;
    if (r > 0.0) {
        // Two rings, turned per pixel so the taps blur instead of stacking
        // copies of the silhouette.
        let rot = hash12(floor(p.xz * 400.0)) * 6.2832;
        for (var k: i32 = 0; k < 8; k = k + 1) {
            let a = rot + f32(k) * 0.7854;
            let rr = r * select(1.0, 0.5, (k & 1) == 1);
            s += shadow_factor(p + vec3<f32>(cos(a), 0.0, sin(a)) * rr, n);
        }
        s /= 9.0;
    }
    return mix(g_shadow_soft.y, 1.0, s);
}

fn shade_pbr(
    world: vec3<f32>,
    n_in: vec3<f32>,
    albedo: vec3<f32>,
    rough_in: f32,
    metal: f32,
    ao: f32,
    emit: vec3<f32>,
) -> vec3<f32> {
    let n = normalize(n_in);
    let sun = normalize(frame.sun_dir.xyz);
    let view = normalize(frame.cam_pos.xyz - world);
    let perceptual = clamp(rough_in, 0.089, 1.0);
    let rough = perceptual * perceptual;
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let ndv = max(dot(n, view), 1e-4);
    let halfv = normalize(sun + view);
    let ndh = clamp(dot(n, halfv), 0.0, 1.0);
    let ldh = clamp(dot(sun, halfv), 0.0, 1.0);
    let shadow = shadow_soft(world + n * 0.012 + g_shadow_wobble, n);
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    let diff_col = albedo * (1.0 - metal);
    let f_ab = F_AB(perceptual, ndv);

    // Sun: Burley diffuse plus GGX with Filament's multiscatter compensation.
    let f90 = clamp(dot(f0, vec3<f32>(50.0 * 0.33)), 0.0, 1.0);
    let F = F_Schlick_vec(f0, f90, ldh);
    var spec = D_GGX(rough, ndh) * V_SmithGGXCorrelated(rough, ndv, ndl) * F;
    spec *= 1.0 + f0 * (1.0 / (f_ab.x + f_ab.y) - 1.0);
    let sun_in = frame.sun_color.rgb * ndl * shadow * 1.25;
    var col = (diff_col * Fd_Burley(perceptual, ndv, ndl, ldh) + spec) * sun_in * 3.14159265;

    // Sky: Bevy's environment_map_light with the multiscatter split.
    let gain = frame.env.y;
    let refl = reflect(-view, n);
    let ems = 1.0 - (f_ab.x + f_ab.y);
    // A reflection that dips under the surface sees soil, not sky.
    let horizon = clamp(1.0 + dot(refl, n), 0.0, 1.0);
    // Rough soil shadows its own grains, so a grazing sky sheen stays faint;
    // without this the sand wears a milky veil of sky.
    let spec_occ = clamp(pow(ndv + ao, exp2(-16.0 * perceptual - 1.0)) - 1.0 + ao, 0.0, 1.0)
        * horizon * horizon * f90 * mix(1.0, 0.3, smoothstep(0.35, 0.9, perceptual));
    let fss_ess = (f0 * f_ab.x + f_ab.y) * spec_occ;
    let f_avg = f0 + (1.0 - f0) / 21.0;
    let fms_ems = fss_ess * f_avg / (1.0 - ems * f_avg) * ems;
    let kd = diff_col * (1.0 - (fss_ess + fms_ems));
    let radiance = env_radiance(radiance_sample_direction(n, refl, perceptual), perceptual);
    // Ground in shadow still sees the sky, but less of the sun-lit half of it.
    let sky_occ = mix(0.72, 1.0, shadow);
    col += ((fms_ems + kd) * sh_irradiance(n) * ao + fss_ess * radiance) * gain * sky_occ;
    col += lamp_light(world, n, view, diff_col, f0, perceptual);
    col += emit;
    return atmosphere(world, col);
}

fn shade(world: vec3<f32>, n_in: vec3<f32>, albedo: vec3<f32>, rough: f32, emit: vec3<f32>) -> vec3<f32> {
    return shade_pbr(world, n_in, albedo, rough, 0.0, 1.0, emit);
}

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    return vec4<f32>(x, y, 1.0, 1.0);
}

@fragment
fn fs_sky(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / max(frame.params.xy, vec2<f32>(1.0));
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let w = frame.inv_view_proj * ndc;
    let dir = w.xyz / w.w - frame.cam_pos.xyz;
    if (frame.env.z > 0.5 || sand_field()) {
        // HDRI sky, or the baked cloudy sky the water reflects. The disc is
        // the scene sun, drawn where the light comes from.
        let d = normalize(dir);
        let sun = normalize(frame.sun_dir.xyz);
        let disc = pow(max(dot(d, sun), 0.0), 900.0) * 4.0;
        var sky = env_radiance(d, 0.0) + frame.sun_color.rgb * disc;
        if (sand_field()) {
            // Far dunes on the horizon, a few hundred metres out, softened by
            // the haze. Backdrop only: nothing in the scene touches them.
            let phi = atan2(d.z, d.x);
            let a = phi * 5.0;
            let prof = 0.004 + 0.04 * bw_fbm(vec2<f32>(a, 1.3)) + 0.012 * bw_fbm(vec2<f32>(a * 3.1, 7.7));
            let ea = 0.004;
            let slope = (0.04 * bw_fbm(vec2<f32>(a + ea * 5.0, 1.3)) - 0.04 * bw_fbm(vec2<f32>(a, 1.3))) / ea;
            let horizon_y = d.y;
            if (horizon_y < prof) {
                // Lit by the sun on the faces that turn to it.
                let side = vec3<f32>(-sin(phi), 0.0, cos(phi));
                let fn_ = normalize(vec3<f32>(0.0, 1.0, 0.0) * 0.6 - d * 0.6 + side * slope * 2.0);
                let sand = vec3<f32>(0.6, 0.44, 0.28);
                let lit = sand * (frame.sun_color.rgb * clamp(dot(fn_, sun), 0.0, 1.0) * 1.25
                    + sh_irradiance(fn_) * frame.env.y);
                // Haze: the dunes sit far out, so most of their colour is air.
                let air = env_radiance(vec3<f32>(d.x, 0.02, d.z), 0.3);
                let depth_k = smoothstep(prof, prof - 0.02, horizon_y);
                sky = mix(air, lit, mix(0.5, 0.72, depth_k));
            }
        }
        // The backdrop sky a stop down and a little bluer, so the HDRI's pale
        // haze does not blow out to white above the dunes.
        let sl = dot(sky, vec3<f32>(0.2126, 0.7152, 0.0722));
        sky = max(mix(vec3<f32>(sl), sky, 1.35) * vec3<f32>(0.88, 0.95, 1.08), vec3<f32>(0.0)) * 0.62;
        return present(sky);
    }
    return present(sky_color(dir));
}

struct TerrIn {
    @location(0) xz: vec2<f32>,
    @location(1) ij: vec2<f32>,
}

struct TerrOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) ij: vec2<f32>,
}

fn load_height_stamp(ij: vec2<i32>) -> vec2<f32> {
    let dims = vec2<i32>(textureDimensions(height_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    let t = textureLoad(height_tex, c, 0);
    // WebGL path packs 16-bit height and stamp into an RGBA8 texel.
    if (frame.misc.z > 0.5) {
        let h = (floor(t.r * 255.0 + 0.5) * 256.0 + floor(t.g * 255.0 + 0.5)) / 65535.0;
        let s = (floor(t.b * 255.0 + 0.5) * 256.0 + floor(t.a * 255.0 + 0.5)) / 65535.0;
        return vec2<f32>(h * 3.5 - 1.0, s * 1.5 - 0.75);
    }
    return t.rg;
}

fn height_at(ij: vec2<i32>) -> f32 {
    return load_height_stamp(ij).x;
}

fn imprint_at(ij: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(imprint_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(imprint_tex, c, 0);
}

fn crust_at(ij: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(crust_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(crust_tex, c, 0);
}

// Thickness and opening, bilinear, in the export's 0..1 range.
fn crust_bilinear(xz: vec2<f32>) -> vec2<f32> {
    let ij = world_to_ij(xz);
    let i = floor(ij);
    let f = fract(ij);
    let i0 = vec2<i32>(i);
    let c00 = crust_at(i0).rg;
    let c10 = crust_at(i0 + vec2<i32>(1, 0)).rg;
    let c01 = crust_at(i0 + vec2<i32>(0, 1)).rg;
    let c11 = crust_at(i0 + vec2<i32>(1, 1)).rg;
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

// Same network the solver opens. Gap is in cell units of the 0.48 m lattice.
fn crack_gap(xz: vec2<f32>) -> f32 {
    let scale = 0.48;
    let fx = xz.x / scale;
    let fz = xz.y / scale;
    let ix = floor(fx);
    let iz = floor(fz);
    var best = 1e9;
    var second = 1e9;
    for (var dj: i32 = -1; dj <= 1; dj++) {
        for (var di: i32 = -1; di <= 1; di++) {
            let gx = ix + f32(di);
            let gz = iz + f32(dj);
            let jx = hash12(vec2<f32>(gx, gz)) - 0.5;
            let jz = hash12(vec2<f32>(gx + 4.2, gz + 1.7)) - 0.5;
            let cx = gx + 0.5 + jx * 0.8;
            let cz = gz + 0.5 + jz * 0.8;
            let d = length(vec2<f32>(fx - cx, fz - cz));
            if (d < best) {
                second = best;
                best = d;
            } else if (d < second) {
                second = d;
            }
        }
    }
    return max(second - best, 0.0);
}

fn cell_center(ij: vec2<i32>) -> vec2<f32> {
    let cell = max(frame.params.z, 1e-4);
    let half = cell * frame.params.w * 0.5;
    return vec2<f32>((f32(ij.x) + 0.5) * cell - half, (f32(ij.y) + 0.5) * cell - half);
}

// The pressed track is the heightfield. A lug grid on a flat floor read as a
// checkerboard card in the rut, so the print is not stamped on again here.
fn tread_delta(xz: vec2<f32>) -> f32 {
    return (xz.x - xz.x) * 0.0;
}

// x = patch weight, y = carcass underside, z = hide the square stamp.
// The bowl follows the circle across the contact, then fades at the crown
// so it does not cut a shelf through the sidewall.
fn tire_contact(xz: vec2<f32>, a: vec4<f32>, b: vec4<f32>) -> vec3<f32> {
    if (b.w < 0.5) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let center = a.xyz;
    let radius = max(a.w, 0.05);
    let fwd = vec2<f32>(b.x, b.y);
    let fwd_n = fwd / max(length(fwd), 1e-4);
    let rel = xz - center.xz;
    let along = dot(rel, fwd_n);
    let lat = length(rel - fwd_n * along);
    let sink = max(b.z, 0.004);
    let chord = sqrt(max(2.0 * radius * sink - sink * sink, 0.0));
    let reach = max(chord, 0.08);
    let half_w = 0.102;
    let ell = length(vec2<f32>(along / reach, lat / half_w));
    var weight = 0.0;
    var underside = center.y - radius;
    if (ell < 1.25) {
        let radial2 = radius * radius - along * along;
        if (radial2 > 0.002) {
            underside = center.y - sqrt(radial2) - 0.002;
            let rise = underside - (center.y - radius);
            let buried = 1.0 - smoothstep(sink * 0.72, sink * 1.15, rise);
            // Full depth across the crown, then a short wall up to the lip
            // so the mud meets the rubber instead of leaving a moat.
            weight = (1.0 - smoothstep(0.78, 1.04, ell)) * buried;
        }
    }
    // Beside the rubber only. The open rut behind the tread stays, so the
    // pad does not sit in the groove as a flat card.
    let ahead = 1.0 - smoothstep(0.22, 0.72, along);
    let not_behind = 1.0 - smoothstep(-0.02, 0.20, -along);
    let across = 1.0 - smoothstep(0.12, 0.28, lat);
    let closure = ahead * not_behind * across * (1.0 - weight);
    return vec3<f32>(clamp(weight, 0.0, 1.0), underside, clamp(closure, 0.0, 1.0));
}

// x = carcass underside of the strongest tire, y = weight, z = stamp closure.
fn tire_seat(xz: vec2<f32>, base_h: f32) -> vec3<f32> {
    var under = base_h;
    var w = 0.0;
    var close = 0.0;
    let c0 = tire_contact(xz, frame.tires[0], frame.tires[1]);
    if (c0.x > w) { under = c0.y; }
    w = max(w, c0.x);
    close = max(close, c0.z);
    let c1 = tire_contact(xz, frame.tires[2], frame.tires[3]);
    if (c1.x > w) { under = c1.y; }
    w = max(w, c1.x);
    close = max(close, c1.z);
    let c2 = tire_contact(xz, frame.tires[4], frame.tires[5]);
    if (c2.x > w) { under = c2.y; }
    w = max(w, c2.x);
    close = max(close, c2.z);
    let c3 = tire_contact(xz, frame.tires[6], frame.tires[7]);
    if (c3.x > w) { under = c3.y; }
    w = max(w, c3.x);
    close = max(close, c3.z);
    return vec3<f32>(under, w, close);
}

// Metres the stamp added. Negative is soil the tire removed.
fn stamp_at(ij: vec2<i32>) -> f32 {
    return load_height_stamp(ij).y;
}

fn mat_at(ij: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(mat_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(mat_tex, c, 0);
}

fn kind_of(ij: vec2<i32>) -> u32 {
    return u32(floor(mat_at(ij).a * 255.0 + 0.5));
}

// 0 mud_forest, 1 brown_mud_dry, 2 clay, 3 grass, 4 gravel, 5 beauty wet
// mud, 6 beauty sand. Silt (11) starts from the dry map and gets its own tint.
// The beauty drive lays the undisturbed field as sand (lamps[1].w).
fn sand_field() -> bool {
    return frame.lamps[1].w > 0.5;
}

fn layer_of(kind: u32) -> i32 {
    if (kind == 0u && sand_field()) { return 6; }
    if (kind == 1u) { return 3; }
    if (kind == 2u) { return 4; }
    if (kind == 9u) { return 2; }
    if (kind == 4u) { return 5; }
    // Bank sand takes the beauty sand layer on the beauty field only; on
    // main's scenes it stays main's layer (the bench bank drew as a tan plate).
    if (kind == 10u && sand_field()) { return 6; }
    if (kind == 0u || kind == 3u || kind == 7u || kind == 11u) { return 1; }
    return 0;
}

// Metres per repeat of each ground layer.
fn layer_uv(layer: i32, xz: vec2<f32>) -> vec2<f32> {
    var tile = 3.2;
    if (layer == 1) { tile = 2.7; }
    else if (layer == 2) { tile = 3.5; }
    else if (layer == 3) { tile = 2.3; }
    else if (layer == 4) { tile = 3.3; }
    else if (layer == 5) { tile = 2.6; }
    else if (layer == 6) { tile = 2.2; }
    return xz / tile;
}

fn disp_scale(kind: u32) -> f32 {
    if (kind == 6u) { return 0.16; }
    if (kind == 2u || kind == 3u || kind == 11u) { return 0.006; }
    if (kind == 10u) { return 0.02; }
    if (kind == 8u) { return 0.05; }
    // Packed clay and the fresh cut keep their map relief, so a film
    // and a slurry sit in the groove instead of on a flat card.
    if (kind == 9u) { return 0.08; }
    if (kind == 5u) { return 0.055; }
    if (kind == 1u) { return 0.03; }
    return 0.018;
}

fn pick5(layer: i32, a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>, e: vec4<f32>) -> vec4<f32> {
    if (layer == 1) { return b; }
    if (layer == 2) { return c; }
    if (layer == 3) { return d; }
    if (layer == 4) { return e; }
    return a;
}

// World x = (i + 0.5) * cell - half. Inverse of the heightfield layout.
fn world_to_ij(xz: vec2<f32>) -> vec2<f32> {
    let cell = max(frame.params.z, 1e-4);
    let half = cell * frame.params.w * 0.5;
    return (xz + vec2<f32>(half)) / cell - vec2<f32>(0.5);
}

fn kind_nearest(xz: vec2<f32>) -> u32 {
    return kind_of(vec2<i32>(round(world_to_ij(xz))));
}

fn lip_near(xz: vec2<f32>) -> bool {
    let i0 = vec2<i32>(floor(world_to_ij(xz)));
    let k00 = kind_of(i0);
    let k10 = kind_of(i0 + vec2<i32>(1, 0));
    let k01 = kind_of(i0 + vec2<i32>(0, 1));
    let k11 = kind_of(i0 + vec2<i32>(1, 1));
    // Placed spoil stays bilinear. A sharp face on it was a flat card.
    return k00 == 6u || k10 == 6u || k01 == 6u || k11 == 6u;
}

fn height_raw(xz: vec2<f32>) -> f32 {
    let ij = world_to_ij(xz);
    let i = floor(ij);
    let f = fract(ij);
    let i0 = vec2<i32>(i);
    let h00 = height_at(i0);
    let h10 = height_at(i0 + vec2<i32>(1, 0));
    let h01 = height_at(i0 + vec2<i32>(0, 1));
    let h11 = height_at(i0 + vec2<i32>(1, 1));
    // Cubic between samples and the fine track live in terrain_geom.wgsl.
    return tg_refine(xz, mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y));
}

fn snow_profile_at(i: i32) -> f32 {
    let v = frame.snow_profile[i / 4];
    let c = i % 4;
    if (c == 0) { return v.x; }
    if (c == 1) { return v.y; }
    if (c == 2) { return v.z; }
    return v.w;
}

// The turn lip is a circle. Sampling it from the cell grid draws a staircase.
// Along the arc, follow the radial profile instead.
fn snow_arc_height(xz: vec2<f32>) -> f32 {
    let raw = height_raw(xz);
    if (frame.snow_arc2.y < 1.5) {
        return raw;
    }
    let rel = xz - frame.snow_arc.xy;
    let radius = length(rel);
    let r_lo = frame.snow_arc.w;
    let dr = max(frame.snow_arc2.x, 1e-4);
    let n = i32(frame.snow_arc2.y);
    let r_hi = r_lo + dr * f32(n - 1);
    let theta = atan2(rel.y, rel.x);
    let th_in = frame.snow_arc2.z;
    let th_out = frame.snow_arc2.w;
    if (theta < th_in || theta > th_out || radius < r_lo || radius > r_hi) {
        return raw;
    }
    let x = clamp((radius - r_lo) / dr, 0.0, f32(n - 1));
    let i = i32(floor(x));
    let f = fract(x);
    let i1 = min(i + 1, n - 1);
    let prof = mix(snow_profile_at(i), snow_profile_at(i1), f);
    let edge = 0.22;
    var fade = 1.0;
    if (theta < th_in + edge) {
        fade = (theta - th_in) / edge;
    } else if (theta > th_out - edge) {
        fade = (th_out - theta) / edge;
    }
    fade = clamp(fade, 0.0, 1.0);
    fade = fade * fade * (3.0 - 2.0 * fade);
    return mix(raw, prof, fade);
}

fn height_bilinear(xz: vec2<f32>) -> f32 {
    if (frame.sun_dir.w > 0.5 && frame.snow_arc2.y > 1.5) {
        return snow_arc_height(xz);
    }
    return height_raw(xz);
}

fn stamp_bilinear(xz: vec2<f32>) -> f32 {
    let ij = world_to_ij(xz);
    let i = floor(ij);
    let f = fract(ij);
    let i0 = vec2<i32>(i);
    let s00 = stamp_at(i0);
    let s10 = stamp_at(i0 + vec2<i32>(1, 0));
    let s01 = stamp_at(i0 + vec2<i32>(0, 1));
    let s11 = stamp_at(i0 + vec2<i32>(1, 1));
    return mix(mix(s00, s10, f.x), mix(s01, s11, f.x), f.y);
}

// The sill between the hubs. Per-tire closure never reached it, so the
// cut under the body stayed a hole. The box stays inside the tires: a
// pad behind the tread was an orange card lying in the rut.
fn body_cover(xz: vec2<f32>) -> f32 {
    let live = frame.tires[1].w + frame.tires[3].w + frame.tires[5].w + frame.tires[7].w;
    if (live < 0.5) {
        return 0.0;
    }
    var minx = 1e9;
    var maxx = -1e9;
    var minz = 1e9;
    var maxz = -1e9;
    if (frame.tires[1].w > 0.5) {
        minx = min(minx, frame.tires[0].x);
        maxx = max(maxx, frame.tires[0].x);
        minz = min(minz, frame.tires[0].z);
        maxz = max(maxz, frame.tires[0].z);
    }
    if (frame.tires[3].w > 0.5) {
        minx = min(minx, frame.tires[2].x);
        maxx = max(maxx, frame.tires[2].x);
        minz = min(minz, frame.tires[2].z);
        maxz = max(maxz, frame.tires[2].z);
    }
    if (frame.tires[5].w > 0.5) {
        minx = min(minx, frame.tires[4].x);
        maxx = max(maxx, frame.tires[4].x);
        minz = min(minz, frame.tires[4].z);
        maxz = max(maxz, frame.tires[4].z);
    }
    if (frame.tires[7].w > 0.5) {
        minx = min(minx, frame.tires[6].x);
        maxx = max(maxx, frame.tires[6].x);
        minz = min(minz, frame.tires[6].z);
        maxz = max(maxz, frame.tires[6].z);
    }
    let dx = max((minx + 0.28) - xz.x, xz.x - (maxx - 0.28));
    let dz = max((minz + 0.35) - xz.y, xz.y - (maxz - 0.35));
    return 1.0 - smoothstep(0.0, 0.16, max(dx, dz));
}

// Put the pad back under the carcass. The stamp stays in the field.
// x = drawn height, y = bowl weight, z = how fully this column is covered.
fn apply_seat(xz: vec2<f32>, base_h: f32) -> vec3<f32> {
    // Win-demo stills: tire seat on the heightfield reads as spikes and holes.
    if (skip_tire_seat()) {
        return vec3<f32>(base_h, 0.0, 0.0);
    }
    let seated = tire_seat(xz, base_h);
    let cover = max(seated.z, body_cover(xz));
    if (cover < 0.001 && seated.y < 0.001) {
        return vec3<f32>(base_h, 0.0, 0.0);
    }
    let pre = height_bilinear(xz) - stamp_bilinear(xz);
    var h = mix(base_h, pre, cover);
    h = mix(h, seated.x, seated.y);
    return vec3<f32>(h, seated.y, cover);
}

// Same samples as height_bilinear, but the step sits in a short wall.
// A linear blend across a 20 cm cell shades a 10 cm rut as a tint.
fn height_sharp(xz: vec2<f32>) -> f32 {
    let ij = world_to_ij(xz);
    let base = floor(ij);
    let f = fract(ij);
    let i0 = vec2<i32>(base);
    let h00 = height_at(i0);
    let h10 = height_at(i0 + vec2<i32>(1, 0));
    let h01 = height_at(i0 + vec2<i32>(0, 1));
    let h11 = height_at(i0 + vec2<i32>(1, 1));
    // The wall sits in the last fifth of the cell. A wide blend was the ramp.
    let sx = smoothstep(0.78, 0.98, f.x);
    let sy = smoothstep(0.78, 0.98, f.y);
    return mix(mix(h00, h10, sx), mix(h01, h11, sx), sy);
}

fn mat_bilinear(xz: vec2<f32>) -> vec4<f32> {
    let ij = world_to_ij(xz);
    let i = floor(ij);
    let f = fract(ij);
    let i0 = vec2<i32>(i);
    let m00 = mat_at(i0);
    let m10 = mat_at(i0 + vec2<i32>(1, 0));
    let m01 = mat_at(i0 + vec2<i32>(0, 1));
    let m11 = mat_at(i0 + vec2<i32>(1, 1));
    return mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
}

// Continuous slope of the real displaced height. Not the triangle normal,
// which is constant across a 20 cm cell and flattens a 14 cm groove.
fn sample_h(xz: vec2<f32>, torn: bool) -> f32 {
    if (win_demo_flat_terrain()) {
        return height_bilinear(xz);
    }
    if (torn) {
        return height_sharp(xz);
    }
    return height_bilinear(xz);
}

// The pressed track and the shed spoil are one surface. height_sharp turns
// every 20 cm cell into a flat tile, which reads as loose boxes.
fn track_near(xz: vec2<f32>) -> bool {
    let ij = vec2<i32>(round(world_to_ij(xz)));
    // Include the lip beside the print. A sharp step one cell out was a hard line.
    for (var dz: i32 = -2; dz <= 2; dz++) {
        for (var dx: i32 = -2; dx <= 2; dx++) {
            if (imprint_at(ij + vec2<i32>(dx, dz)).a > 0.5) {
                return true;
            }
        }
    }
    return false;
}

fn terrain_normal(xz: vec2<f32>) -> vec3<f32> {
    let torn = !win_demo_flat_terrain() && lip_near(xz) && !track_near(xz);
    // Short step. A wider sample flattens a tyre-width cut into the lane slope.
    let e = max(frame.params.z * select(0.28, 0.16, torn), 0.04);
    let h_l = sample_h(xz - vec2<f32>(e, 0.0), torn);
    let h_r = sample_h(xz + vec2<f32>(e, 0.0), torn);
    let h_d = sample_h(xz - vec2<f32>(0.0, e), torn);
    let h_u = sample_h(xz + vec2<f32>(0.0, e), torn);
    var n = normalize(vec3<f32>(h_l - h_r, e * 2.0, h_d - h_u));
    if (!torn) {
        // Away from the lip, a centimetre wall stays a shallow tilt.
        let se = max(frame.params.z * 0.7, 0.1);
        let s_l = stamp_bilinear(xz - vec2<f32>(se, 0.0));
        let s_r = stamp_bilinear(xz + vec2<f32>(se, 0.0));
        let s_d = stamp_bilinear(xz - vec2<f32>(0.0, se));
        let s_u = stamp_bilinear(xz + vec2<f32>(0.0, se));
        let raw = vec3<f32>(s_l - s_r, 0.0, s_d - s_u);
        let mag = length(raw);
        if (mag > 0.004) {
            let gain = min(3.5, 0.07 / mag);
            n = normalize(n + raw * gain);
        }
    }
    let grit = noise(xz * 2.4);
    let grit_x = noise(xz * 2.4 + vec2<f32>(0.45, 0.0));
    let grit_z = noise(xz * 2.4 + vec2<f32>(0.0, 0.45));
    let grit_gain = select(0.035, 0.09, torn);
    n = normalize(n + vec3<f32>(grit - grit_x, 0.0, grit - grit_z) * grit_gain);
    // The live bowl is a vertex displacement. Shade that slope too, or the
    // patch stays a flat card under the carcass.
    let live = frame.tires[1].w + frame.tires[3].w + frame.tires[5].w + frame.tires[7].w;
    if (live > 0.5 && !skip_tire_seat()) {
        let se_t = 0.03;
        let hx1 = apply_seat(xz + vec2<f32>(se_t, 0.0), height_bilinear(xz + vec2<f32>(se_t, 0.0))).x;
        let hx0 = apply_seat(xz - vec2<f32>(se_t, 0.0), height_bilinear(xz - vec2<f32>(se_t, 0.0))).x;
        let hz1 = apply_seat(xz + vec2<f32>(0.0, se_t), height_bilinear(xz + vec2<f32>(0.0, se_t))).x;
        let hz0 = apply_seat(xz - vec2<f32>(0.0, se_t), height_bilinear(xz - vec2<f32>(0.0, se_t))).x;
        let bx1 = height_bilinear(xz + vec2<f32>(se_t, 0.0));
        let bx0 = height_bilinear(xz - vec2<f32>(se_t, 0.0));
        let bz1 = height_bilinear(xz + vec2<f32>(0.0, se_t));
        let bz0 = height_bilinear(xz - vec2<f32>(0.0, se_t));
        let extra = vec3<f32>((hx0 - hx1) - (bx0 - bx1), 0.0, (hz0 - hz1) - (bz0 - bz1));
        if (dot(extra, extra) > 1e-8) {
            n = normalize(n + extra * (0.65 / se_t));
        }
    }
    let lug = tread_delta(xz);
    if (lug < -0.001) {
        let te = 0.025;
        let dx = tread_delta(xz + vec2<f32>(te, 0.0)) - tread_delta(xz - vec2<f32>(te, 0.0));
        let dz = tread_delta(xz + vec2<f32>(0.0, te)) - tread_delta(xz - vec2<f32>(0.0, te));
        n = normalize(n + vec3<f32>(-dx, 0.0, -dz) * (0.85 / te));
    }
    return n;
}

fn ripple_height(xz: vec2<f32>, t: f32) -> f32 {
    let x = xz.x;
    let z = xz.y;
    // Long waves. A tight lattice of sines reads as a checker from above.
    return 0.012 * sin(x * 0.72 + z * 0.18 + t * 0.55)
        + 0.009 * sin(z * 0.95 - x * 0.22 - t * 0.42)
        + 0.005 * sin(x * 1.5 + z * 1.25 + t * 0.8);
}

// Analytic water normal. Screen derivatives of the mesh are a hard grid.
fn ripple_normal(xz: vec2<f32>, t: f32) -> vec3<f32> {
    let x = xz.x;
    let z = xz.y;
    let dx = 0.012 * cos(x * 0.72 + z * 0.18 + t * 0.55) * 0.72
        - 0.009 * cos(z * 0.95 - x * 0.22 - t * 0.42) * 0.22
        + 0.005 * cos(x * 1.5 + z * 1.25 + t * 0.8) * 1.5;
    let dz = 0.012 * cos(x * 0.72 + z * 0.18 + t * 0.55) * 0.18
        + 0.009 * cos(z * 0.95 - x * 0.22 - t * 0.42) * 0.95
        + 0.005 * cos(x * 1.5 + z * 1.25 + t * 0.8) * 1.25;
    // A small tilt. Large normals turned the pond into a chrome dome.
    return normalize(vec3<f32>(-dx * 1.6, 1.0, -dz * 1.6));
}

fn terrain_height(xz: vec2<f32>) -> f32 {
    if (win_demo_flat_terrain()) {
        return height_bilinear(xz);
    }
    if (track_near(xz)) {
        return height_bilinear(xz);
    }
    // The lip is a step. Everywhere else stays bilinear so a berm is not a stair.
    if (lip_near(xz)) {
        return height_sharp(xz);
    }
    return height_bilinear(xz);
}

fn terrain_vertex(in: TerrIn) -> TerrOut {
    if (win_demo_flat_terrain()) {
        let h = win_demo_clamp_surface_y(in.xz, height_bilinear(in.xz));
        var o: TerrOut;
        o.clip = frame.view_proj * vec4<f32>(in.xz.x, h, in.xz.y, 1.0);
        o.world = vec3<f32>(in.xz.x, h, in.xz.y);
        o.ij = in.ij;
        return o;
    }
    let kind = kind_nearest(in.xz);
    // Every layer is sampled. The lip is mud_forest, and only that lip is loud.
    let d0 = textureSampleLevel(ground_pack, ground_samp, in.xz / 3.2, 0, 0.0).b;
    let d1 = textureSampleLevel(ground_pack, ground_samp, in.xz / 2.7, 1, 0.0).b;
    let d2 = textureSampleLevel(ground_pack, ground_samp, in.xz / 3.5, 2, 0.0).b;
    let d3 = textureSampleLevel(ground_pack, ground_samp, in.xz / 2.3, 3, 0.0).b;
    let d4 = textureSampleLevel(ground_pack, ground_samp, in.xz / 3.3, 4, 0.0).b;
    var disp = d0;
    let layer = layer_of(kind);
    if (layer == 1) { disp = d1; }
    else if (layer == 2) { disp = d2; }
    else if (layer == 3) { disp = d3; }
    else if (layer == 4) { disp = d4; }
    else if (layer >= 5) {
        disp = textureSampleLevel(ground_pack, ground_samp, layer_uv(layer, in.xz), layer, 0.0).b;
    }
    if (kind == 6u) { disp = d0; }
    // Clay has no height map. Packed tread in a cut keeps the mud it was pressed from.
    if (kind == 9u && -stamp_bilinear(in.xz) > 0.012) {
        disp = d0;
    }
    let base = terrain_height(in.xz);
    let seated = apply_seat(in.xz, base);
    // Under the carcass the cylinder is the surface. The stored lugs show once the tire has left.
    // Closure is the lip beside the rubber, not another tread.
    let on_tire = seated.y > 0.25 || seated.z > 0.40;
    let pattern = select(tread_delta(in.xz), 0.0, on_tire);
    let quiet = clamp(max(seated.y, seated.z), 0.0, 1.0);
    // The print and the spoil are the height. The gravel map was breaking them into tiles.
    let calm = select(0.0, 1.0, track_near(in.xz));
    var h = seated.x + pattern + (disp - 0.5) * disp_scale(kind) * (1.0 - quiet) * (1.0 - 0.35 * calm);
    // A column the tire moved is the solver field: bed plus placed berm.
    // Map relief and the lug ribbon are not stacked on that height. The
    // carcass may only press the field down onto the rubber.
    if (abs(stamp_bilinear(in.xz)) > 0.004) {
        h = base;
        if (seated.y > 0.2 && seated.x < base) {
            h = mix(base, seated.x, seated.y);
        }
    }
    // The heightfield ends. Ease the map relief down so the rim is not a lip.
    let rim = smoothstep(15.5, 19.2, max(abs(in.xz.x), abs(in.xz.y)));
    h = mix(h, base, rim);
    // On the beauty sand the published ground (binding 1: bed + placed) is
    // the base, so the tire-force cut and its berm under the puddle are in
    // the mesh and the water stands on the same ground it was solved on.
    if (sand_field()) {
        h = bc_ground_soft(in.xz, base) + (h - base);
    }
    h = win_demo_clamp_surface_y(in.xz, h);
    // World Y at this column's XZ. Adding the rut along the normal would
    // walk the groove up a slope as the wheel moves.
    let world = vec3<f32>(in.xz.x, h, in.xz.y);
    var o: TerrOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    return o;
}

@fragment
fn fs_terrain(in: TerrOut) -> @location(0) vec4<f32> {
    tg_set_pixel_footprint(in.world.xz);
    let xz = in.world.xz;
    var n = terrain_normal(xz);
    if (win_demo_flat_terrain() && win_demo_in_pool(xz) && n.y > 0.62) {
        discard;
    }
    let demo_seam_early = win_demo_flat_terrain() && distance(xz, beauty_seam_center()) < 3.35;
    if (demo_seam_early && n.y < 0.58) {
        discard;
    }
    // On the beauty sand the mesh is the contract ground (bc_ground_soft);
    // the solver heightfield's normal stepped on the 0.20 m rut cells and lit
    // the rut wall as an orange plate. Take the normal from the same filtered
    // contract height, over a 0.3 m baseline.
    if (sand_field() && bc_inside(xz) > 0.0) {
        let ge = 0.15;
        let hx1 = bc_ground_soft(xz + vec2<f32>(ge, 0.0), terrain_height(xz + vec2<f32>(ge, 0.0)));
        let hx0 = bc_ground_soft(xz - vec2<f32>(ge, 0.0), terrain_height(xz - vec2<f32>(ge, 0.0)));
        let hz1 = bc_ground_soft(xz + vec2<f32>(0.0, ge), terrain_height(xz + vec2<f32>(0.0, ge)));
        let hz0 = bc_ground_soft(xz - vec2<f32>(0.0, ge), terrain_height(xz - vec2<f32>(0.0, ge)));
        let nc = normalize(vec3<f32>(hx0 - hx1, 2.0 * ge, hz0 - hz1));
        n = normalize(mix(n, nc, bc_inside(xz)));
    }
    var kind = kind_nearest(xz);
    let xz_fw = length(fwidth(xz));

    let mat_dims = vec2<f32>(textureDimensions(mat_tex));
    let mat = textureSampleLevel(mat_tex, mat_samp, (in.ij + vec2<f32>(0.5)) / mat_dims, 0.0);
    // Pore moisture and the film are the contract's inside its window.
    // Moisture wicks sideways through the grain: a jittered ring of taps
    // spreads the 0.20 m cell edge into a soft damp front.
    var wet_ring = 0.0;
    let wj = noise(xz * 2.3) * 0.5 + 0.75;
    for (var k: i32 = 0; k < 6; k = k + 1) {
        let a = f32(k) * 1.0472 + wj * 2.0;
        wet_ring += bc_moisture(xz + vec2<f32>(cos(a), sin(a)) * 0.38 * wj, mat.r);
    }
    // Ground under standing water is saturated whatever the pore map says:
    // the rut cells under the puddle drew as bleached, cracked dry crust.
    let film_w = bc_film(xz);
    // The tire-force cut under the puddle is wet sand that dropped (contract
    // binding 1); the kind map re-labels those cells as packed mud, which
    // printed the two tracks as dark tiles with straight cell edges.
    // Same for the dry tracks behind the wheels: damp sand, not mud holes.
    if (sand_field() && kind != 0u && kind != 10u) {
        kind = 0u;
    }
    var wet = max(mix(wet_ring / 6.0, bc_moisture(xz, mat.r), 0.35), smoothstep(0.002, 0.012, film_w.r + film_w.g));
    let film_c = bc_film(xz);
    var water = mix(mat.g * 0.34, film_c.r + film_c.g, bc_inside(xz));
    let demo_seam_terrain = win_demo_flat_terrain() && distance(xz, beauty_seam_center()) < 3.35;
    let horiz_terrain = n.y > 0.58;
    // Main's puddle demo (render flag 2) keeps its own column and wetness.
    if (puddle_beauty_on()) {
        wet = mat.r;
        water = select(mat.g * 0.34, water_column(xz), !demo_seam_terrain || horiz_terrain);
    }
    let darken = mat.b;
    let pool_beauty = puddle_beauty_on() && !demo_seam_terrain;
    let pool_fill = select(0.0, smoothstep(0.045, 0.11, mat.g), pool_beauty);
    // Dark wet band at the liquid edge (peak mid mat.g), not a bright interior dome.
    let pool_shore = select(
        0.0,
        smoothstep(0.038, 0.052, mat.g) * (1.0 - smoothstep(0.068, 0.088, mat.g)),
        pool_beauty,
    );
    let pool_interior = select(0.0, smoothstep(0.020, 0.075, mat.g), pool_beauty);
    let pool_col = water_column(xz);
    let seam_wet_rim = select(
        0.0,
        smoothstep(0.055, 0.018, pool_col) * (1.0 - smoothstep(0.004, 0.014, pool_col)),
        demo_seam_terrain,
    );
    // Exported skin, metres. Zero where the pad has not baked.
    let crust_rg = crust_bilinear(xz);
    // No dry skin under standing water: the crust pad printed pale cracked
    // plates through the puddle round the Niva.
    let thickness = select(crust_rg.x * 0.04 * (1.0 - smoothstep(0.003, 0.015, film_w.r + film_w.g)), 0.0, demo_seam_terrain);

    // Snow is its own layer. sun_dir.w selects it. Compaction is mat.b.
    // The ground albedo stays off so the blanket, the rut, and the berm are the field.
    if (frame.sun_dir.w > 0.5) {
        // Albedo held under ~0.75 so a sunlit face does not clip to a flat
        // white sheet; the relief reads from the shading and the sky fill
        // (blue from the HDRI) in the shadows.
        // The frame's exposure is set for the sand; at snow's true 0.8 every
        // sunlit face clipped to one white sheet. Held to ~0.5 here.
        let fresh = vec3<f32>(0.4, 0.425, 0.46);
        let packed = vec3<f32>(0.27, 0.3, 0.35);
        var albedo = mix(fresh, packed, darken);
        let fw_s = length(fwidth(xz));
        let grain = mix(0.5, noise(xz * 6.5), 1.0 - smoothstep(0.05, 0.2, fw_s));
        albedo *= mix(0.94, 1.04, grain);
        let rough_out = mix(0.92, 0.8, darken);
        // Cavity AO from the heightfield: the floor of a rut and the foot of
        // a berm see less sky than the open blanket.
        var ring = 0.0;
        for (var k: i32 = 0; k < 8; k = k + 1) {
            let a = f32(k) * 0.7854;
            ring += terrain_height(xz + vec2<f32>(cos(a), sin(a)) * 0.35);
        }
        let cav = clamp((ring / 8.0 - in.world.y) / 0.07, 0.0, 1.0);
        let ao_s = 1.0 - 0.6 * cav;
        // Stronger relief in the shading normal: the berms and the rut walls
        // are a few cm and vanished under the flat sky light.
        var n_s = normalize(vec3<f32>(n.x * 1.8, n.y, n.z * 1.8));
        // Wind-packed surface grain: a material texture in the normal only,
        // faded where a pixel covers it.
        let gq_s = xz * 3.0;
        let gs0 = noise(gq_s) + 0.5 * noise(gq_s * 2.7);
        let gsx = noise(gq_s + vec2<f32>(0.05, 0.0)) + 0.5 * noise((gq_s + vec2<f32>(0.05, 0.0)) * 2.7) - gs0;
        let gsz = noise(gq_s + vec2<f32>(0.0, 0.05)) + 0.5 * noise((gq_s + vec2<f32>(0.0, 0.05)) * 2.7) - gs0;
        n_s = normalize(n_s - vec3<f32>(gsx, 0.0, gsz) * 3.0 * (1.0 - smoothstep(0.03, 0.15, fw_s)));
        // A cold tint in the shade: what the sun does not reach is lit by
        // the blue sky only.
        let sun_d = normalize(frame.sun_dir.xyz);
        let shade_k = 1.0 - clamp(dot(n_s, sun_d) * 2.0, 0.0, 1.0);
        albedo *= mix(vec3<f32>(1.0), vec3<f32>(0.86, 0.93, 1.06), shade_k * 0.6 + cav * 0.3);
        let lit = shade_pbr(in.world, n_s, albedo, rough_out, 0.0, ao_s, vec3<f32>(0.0));
        return present(lit);
    }

    let uv0 = xz / 3.2;
    // Packed clay draws as the mud round it (main).
    var layer = layer_of(kind);
    if (kind == 9u) {
        layer = 0;
    }
    // One fetch from the layer this cell is. Sampling every layer and
    // picking one cost five times the bandwidth for the same texel.
    let luv = layer_uv(layer, xz);
    let diff = textureSample(ground_diff, ground_samp, luv, layer).rgb;
    let ntex = textureSample(ground_nor, ground_samp, luv, layer).rgb;
    let pack = textureSample(ground_pack, ground_samp, luv, layer);
    let ao = clamp(pack.g, 0.0, 1.0);
    let rough_map = clamp(pack.r * (1.0 - pack.a * 0.2), 0.05, 1.0);
    let n_geo = n;
    var tng = vec3<f32>(1.0, 0.0, 0.0);
    tng = normalize(tng - n * dot(n, tng));
    let btg = cross(tng, n);
    var ts = ntex * 2.0 - vec3<f32>(1.0);
    // Dry crust and gravel stay flatter. The torn lip keeps the mud normal.
    var n_gain = 1.0;
    if (track_near(xz)) { n_gain = 0.7; }
    else if (kind == 2u || kind == 3u || kind == 10u || kind == 11u) { n_gain = 0.28; }
    else if (kind == 0u && sand_field()) { n_gain = 0.5; }
    else if (kind == 6u) { n_gain = 1.7; }
    else if (kind == 8u) { n_gain = 1.35; }
    else if (kind == 5u) { n_gain = 1.35; }
    else if (kind == 9u) { n_gain = 1.7; }
    // The ground maps carry no mips. Far away their normals alias into
    // sparkle under a wet lobe, so the relief fades with distance.
    let far = smoothstep(6.0, 22.0, length(in.world - frame.cam_pos.xyz));
    // The beauty mud map is a churned mulch; at full strength every crumb
    // catches the sun.
    if (layer == 5) { n_gain *= 0.55; }
    ts = vec3<f32>(ts.xy * n_gain * mix(1.0, 0.3, far), ts.z);
    n = normalize(tng * ts.x + btg * ts.y + n * ts.z);
    var albedo = diff * mix(1.0, ao, 0.75);
    let muddy = kind == 4u || kind == 5u || kind == 9u || kind == 6u;
    let wet_amt = select(0.12, 0.32, muddy);
    // On the beauty sand the contract's film and wet band own the wetness;
    // the soil's coarse wetness cells smeared into streaks at this angle.
    albedo = mix(albedo, albedo * vec3<f32>(0.66, 0.58, 0.48), smoothstep(0.18, 0.92, wet) * wet_amt * select(1.0, 0.0, sand_field()) * (1.0 - pool_interior * 0.92));
    // Dark bed under the rim — water pass draws the film; terrain stays matte and wet.
    albedo = mix(albedo, albedo * vec3<f32>(0.022, 0.014, 0.010), pool_shore * 2.05);
    albedo = mix(albedo, albedo * vec3<f32>(0.30, 0.26, 0.22), pool_interior * 0.62);
    albedo = mix(albedo, albedo * vec3<f32>(0.28, 0.22, 0.16), seam_wet_rim * 0.85);
    albedo = mix(albedo, albedo * vec3<f32>(0.42, 0.36, 0.30), pool_fill * 0.48);
    if (demo_seam_terrain) {
        let h = height_bilinear(xz);
        let level = win_demo_pool_level();
        let fill = level - h;
        let shore = smoothstep(0.015, 0.11, fill) * (1.0 - smoothstep(0.20, 0.30, fill));
        albedo = mix(albedo, albedo * vec3<f32>(0.34, 0.28, 0.22), shore * 0.75);
    }
    if (kind == 9u) {
        albedo *= mix(1.0, 0.8, smoothstep(0.35, 0.9, darken));
    } else if (kind == 7u) {
        // Placed spoil reads drier and lighter than the torn lip beside it.
        albedo = mix(albedo, albedo * vec3<f32>(1.14, 1.05, 0.86), 0.62);
    } else if (kind == 8u) {
        albedo = mix(albedo, albedo * vec3<f32>(0.72, 0.58, 0.42), 0.28);
    } else if (kind == 5u) {
        albedo = mix(albedo, albedo * vec3<f32>(0.62, 0.5, 0.38), 0.22);
    } else if (kind == 3u) {
        // Baked crust: pale and matte, not the dark lane it dried out of.
        albedo = mix(albedo, albedo * vec3<f32>(1.22, 1.1, 0.9), 0.7);
    } else if (kind == 10u || (kind == 0u && sand_field())) {
        // Sand stays a warm grain. Moisture darkens it a little. It does not turn into clay.
        // The beauty sand map is already a pale grain; only warm it.
        albedo = albedo * vec3<f32>(0.5, 0.44, 0.34);
        albedo = mix(albedo, albedo * vec3<f32>(0.82, 0.74, 0.58), smoothstep(0.15, 0.55, wet) * 0.55 * select(1.0, 0.0, sand_field()));
    } else if (kind == 11u) {
        // Silt is a gray flour. Wet silt goes dark. Dry silt stays lighter than clay.
        albedo = albedo * vec3<f32>(0.98, 0.94, 0.82);
        albedo = mix(albedo, albedo * vec3<f32>(0.52, 0.48, 0.42), smoothstep(0.2, 0.85, wet));
    }
    // Where the skin is still growing, blend the two maps already on the
    // ground. A hard kind edge was a tile. Full wet and full crust stay
    // on the branches above.
    let skin = smoothstep(0.006, 0.028, thickness);
    let clay = kind == 3u || kind == 4u || kind == 5u || kind == 9u || kind == 6u || kind == 7u || kind == 8u;
    if (clay && skin > 0.04 && skin < 0.96) {
        let wet_tex = textureSample(ground_diff, ground_samp, uv0, 0).rgb;
        let dry_tex = textureSample(ground_diff, ground_samp, layer_uv(1, xz), 1).rgb;
        let wet_c = wet_tex * mix(vec3<f32>(1.0), vec3<f32>(0.66, 0.58, 0.48), 0.55);
        var dry_c = dry_tex;
        dry_c = mix(dry_c, dry_c * vec3<f32>(1.22, 1.1, 0.9), 0.7);
        albedo = mix(wet_c, dry_c, skin);
    }
    // The exit print. A fresh field sits under this ramp, so the lane goes dark
    // without a new surface kind.
    albedo *= mix(1.0, 0.36, smoothstep(0.20, 0.52, darken));

    let shore = smoothstep(0.0, 0.02, water) * (1.0 - smoothstep(0.035, 0.11, water)) * (1.0 - pool_interior);
    let drowned = smoothstep(0.012, 0.05, water) * (1.0 - pool_interior);
    // Film and slurry live on this mesh. A glossy sheet over the cut is a flat card.
    let dug = max(-stamp_bilinear(xz), 0.0);
    // On the beauty sand the cut is read from the contract (binding 1 G/A,
    // filtered) below; this 0/1 test on the CPU stamp printed its cells as
    // black stair-stepped holes round the tires.
    let in_cut = dug > 0.012 && water > 0.008 && !sand_field();
    var depth_t = 0.0;
    if (in_cut) {
        // A 3 cm film stays a tint on the clay. Only the deep end murks the mesh.
        depth_t = smoothstep(0.04, 0.12, water);
        let bed_muddy = kind == 4u || kind == 5u || kind == 6u || kind == 8u || kind == 9u;
        let film_tint = select(vec3<f32>(0.90, 0.86, 0.78), vec3<f32>(0.74, 0.56, 0.40), bed_muddy);
        let deep_tint = select(vec3<f32>(0.42, 0.36, 0.28), vec3<f32>(0.16, 0.07, 0.03), bed_muddy);
        let tint = mix(film_tint, deep_tint, depth_t);
        let murk = mix(select(0.05, 0.1, bed_muddy), select(0.55, 0.88, bed_muddy), depth_t);
        albedo = mix(albedo, albedo * tint, murk);
    }
    var rough_out = mix(mix(rough_map, 0.16, shore * (1.0 - drowned)), max(rough_map, 0.72), drowned);
    if (in_cut) {
        // Keep the clay and mud grain. The shore roughness was the card.
        rough_out = mix(rough_map, max(rough_map, 0.58), depth_t);
    }
    if (kind == 3u) { rough_out = max(rough_out, 0.8); }
    if (kind == 10u) { rough_out = max(rough_out, 0.74); }
    if (kind == 11u) { rough_out = mix(rough_out, 0.2, smoothstep(0.35, 0.85, wet)); }
    let lug_h = tread_delta(xz);
    if (lug_h < -0.002) {
        // Pressed lugs: the grain packs dark in the groove of each lug.
        let k = clamp(-lug_h / 0.035, 0.0, 1.0);
        albedo *= mix(1.0, 0.42, k);
    }
    // Fissure colours come from the crust, not from the darkened lug groove.
    let skin_albedo = albedo;
    // Opening in the crust. The stored fissure is about a centimetre, a few
    // pixels here, so the draw is wider. Centre is the gap (clay, lighter
    // than the walls). Each side is a wall, the sunless one darker. Both
    // outer lips are the plate edge that pulled back.
    if (thickness > 0.008) {
        let gapv = crack_gap(xz);
        let skin_m = clamp((thickness - 0.008) / 0.028, 0.0, 1.0);
        let half = mix(0.022, 0.034, skin_m);
        let dist = gapv * 0.24;
        if (skin_m > 0.2 && dist < half * 1.12) {
            let u = clamp(dist / half, 0.0, 1.12);
            let gx = crack_gap(xz + vec2<f32>(0.004, 0.0)) - crack_gap(xz - vec2<f32>(0.004, 0.0));
            let gz = crack_gap(xz + vec2<f32>(0.0, 0.004)) - crack_gap(xz - vec2<f32>(0.0, 0.004));
            let side = normalize(vec2<f32>(gx, gz) + vec2<f32>(1e-4, 0.0));
            let sun_xz = frame.sun_dir.xz;
            // side points out of the gap. This wall faces the sun when -side does.
            let faces_sun = dot(side, sun_xz) < 0.0;
            let floor_w = 1.0 - smoothstep(0.26, 0.42, u);
            let wall_w = smoothstep(0.34, 0.48, u) * (1.0 - smoothstep(0.64, 0.78, u));
            let lip_w = smoothstep(0.72, 0.84, u) * (1.0 - smoothstep(1.00, 1.12, u));
            let plate = skin_albedo;
            let floor_c = plate * vec3<f32>(0.58, 0.36, 0.20);
            let wall_c = plate * select(vec3<f32>(0.30, 0.20, 0.13), vec3<f32>(0.12, 0.07, 0.045), !faces_sun);
            let lip_c = plate * vec3<f32>(1.72, 1.40, 1.08);
            albedo = mix(albedo, lip_c, lip_w);
            albedo = mix(albedo, wall_c, wall_w);
            albedo = mix(albedo, floor_c, floor_w);
            let floor_n = vec3<f32>(0.0, 1.0, 0.0);
            let wall_n = normalize(vec3<f32>(-side.x * 0.45, 0.78, -side.y * 0.45));
            let lip_n = vec3<f32>(0.0, 1.0, 0.0);
            let keep = clamp(1.0 - max(floor_w, max(wall_w, lip_w)), 0.0, 1.0);
            n = normalize(floor_n * floor_w + wall_n * wall_w + lip_n * lip_w + n * keep);
            rough_out = mix(rough_out, 0.97, clamp(floor_w + wall_w, 0.0, 1.0));
        }
    }
    // Moisture is shown, not simulated: wetness from the solver closes the
    // pores (rougher -> smoother) and floods the low texels of the height map
    // first, so a damp lane glints in its hollows before it shines all over.
    let dry = 1.0 - smoothstep(0.25, 0.7, wet);
    // Crack opening, metres, from the contract (written as 0 in version 1).
    // Until the sheet stores it, a dry crust kind or dried clay draws a
    // hairline net of its own; the published opening widens it when it lands.
    let crust_c = bc_crust(xz);
    let opening = crust_c.y;
    let kind_crust = kind == 3u || ((kind == 9u || kind == 11u || kind == 4u) && dry > 0.02);
    // Cracks open as the mud dries: none in the wet, hairlines in the
    // damp ring, full gaps on the baked plates. No threshold, no line.
    let crack_age = smoothstep(0.08, 0.9, dry);
    if (kind_crust || opening > 0.0005) {
        // Baked crust: plates curl up, cracks go dark and matte.
        let scale = select(2.2, 3.4, kind == 3u);
        let edge = crack_edge(xz * scale);
        // Crack half-width in cell units: a hairline, or the published gap.
        let width = max(select(0.0, select(select(0.022, 0.034, kind == 4u), 0.05, kind == 3u), kind_crust) * select(mix(0.35, 1.0, crack_age), 1.0, kind == 3u), opening * scale * 0.5);
        let crack = (1.0 - smoothstep(width * 0.4, width * 1.6 + 0.01, edge))
            * select(max(crack_age * 0.8, smoothstep(0.0005, 0.004, opening)), 1.0, kind == 3u);
        albedo = mix(albedo, albedo * 0.38, crack);
        let plate = smoothstep(0.05, 0.4, edge);
        albedo = mix(albedo, albedo * 1.08, plate * select(dry * 0.5, 1.0, kind == 3u) * 0.5);
        if (kind == 4u) {
            // Dried mud bleaches to a pale grey-brown; the plates curl and
            // catch light at their rims.
            let pale = dot(albedo, vec3<f32>(0.3, 0.5, 0.2));
            let baked = mix(albedo, vec3<f32>(pale), 0.35) * 2.1 + vec3<f32>(0.03, 0.025, 0.02);
            albedo = mix(albedo, baked, smoothstep(0.3, 0.98, dry));
            let rim = smoothstep(0.03, 0.09, edge) * (1.0 - smoothstep(0.09, 0.2, edge));
            albedo *= 1.0 + rim * 0.25 * dry;
        }
        rough_out = mix(rough_out, 0.95, crack);
        let ex = crack_edge(xz * scale + vec2<f32>(0.02, 0.0)) - edge;
        let ez = crack_edge(xz * scale + vec2<f32>(0.0, 0.02)) - edge;
        n = normalize(n - vec3<f32>(ex, 0.0, ez) * 4.0 * crack);
    }
    var wet_rough = 0.32;
    var porosity = 0.5;
    if (kind == 4u) { wet_rough = 0.2; porosity = 0.8; }
    else if (kind == 5u) { wet_rough = 0.10; porosity = 0.85; }
    else if (kind == 9u) { wet_rough = 0.16; porosity = 0.55; }
    else if (kind == 6u) { wet_rough = 0.24; porosity = 0.7; }
    else if (kind == 8u) { wet_rough = 0.16; porosity = 0.7; }
    else if (kind == 7u) { wet_rough = 0.36; porosity = 0.55; }
    else if (kind == 11u) { wet_rough = 0.14; porosity = 0.7; }
    else if (kind == 10u) { wet_rough = 0.40; porosity = 0.9; }
    else if (kind == 1u) { wet_rough = 0.55; porosity = 0.3; }
    else if (kind == 2u) { wet_rough = 0.30; porosity = 0.25; }
    else if (kind == 3u) { wet_rough = 0.85; porosity = 0.1; }
    // On the beauty sand the published moisture sits near 0.75 over metres
    // round the pool; read as damp it flattened and glossed all that sand
    // into a grey halo. Sand only looks wet near saturation.
    let damp = select(smoothstep(0.3, 0.95, wet), smoothstep(0.9, 0.99, wet), sand_field());
    // A film collects in the low half of the grain first.
    let hollow = 1.0 - smoothstep(0.3, 0.62, pack.b);
    var film = clamp(smoothstep(0.86, 1.0, wet) * hollow + shore * 0.6, 0.0, 1.0)
        * select(1.0, 0.0, kind == 3u || kind == 2u);
    // Water the solver holds in a cut is drawn on this mesh. Its top is a
    // calm sheet: smooth, flat, rippled, and it reflects the sky and the bank.
    let rut_water = select(0.0, smoothstep(0.012, 0.04, water), in_cut);
    film = max(film, rut_water);
    if (layer == 5) {
        // Ground048 reads mauve under a blue sky. Pull it to soil brown.
        albedo *= vec3<f32>(0.98, 0.8, 0.58);
    }
    // On the beauty sand the shore band owns the wet look: the coarse
    // moisture cells printed rust-brown blotches over the shallows.
    albedo = mix(albedo, albedo * vec3<f32>(0.52, 0.47, 0.42), damp * porosity * 0.6 * select(1.0, 0.0, sand_field() && (kind == 0u || kind == 10u)));
    // Water fills the grain of saturated soil, so wet mud is smoother than
    // the dry map it was painted from.
    n = normalize(mix(n, n_geo, damp * porosity * 0.55));
    rough_out = mix(rough_out, wet_rough, damp);
    rough_out = mix(rough_out, 0.1, film * 0.85);
    // Water fills the micro relief, so the shading normal relaxes to the cut.
    n = normalize(mix(n, n_geo, film * 0.7));
    if (rut_water > 0.0) {
        let rip = ripple_normal(xz, frame.misc.x);
        let sheet = normalize(vec3<f32>(rip.x * 0.8, 1.0, rip.z * 0.8));
        n = normalize(mix(n, sheet, rut_water));
        rough_out = mix(rough_out, 0.03, rut_water);
        // Silt held in the column: the bed fades into opaque brown water.
        let murk_col = vec3<f32>(0.10, 0.062, 0.03) * mix(1.0, 0.6, smoothstep(0.04, 0.14, water));
        albedo = mix(albedo, murk_col, rut_water * smoothstep(0.01, 0.06, water));
    }
    if (sand_field() && (kind == 0u || kind == 10u)) {
        let sv = sand_variety(xz, albedo, n, rough_out, film_c.r + film_c.g, xz_fw);
        albedo = sv.albedo;
        n = sv.n;
        rough_out = sv.rough;
    }
    // ---- Contract water on the bed: soaked dark rim, saturated bed, caustics.
    var caustic = 0.0;
    let inside_t = bc_inside(xz);
    if (inside_t > 0.0) {
        let film_t = film_c.r + film_c.g;
        // Wet band from the published pore moisture: the solver says how
        // wet the sand at the shore is, the shader only darkens and glosses
        // it. `wet` is already spread over the 0.20 m cells, so the band
        // fades into the dry without a step.
        // Plus the band the shore wicks up: distance to the published film,
        // read on rings out to a metre. Full strength to 0.5 m, dry by 1 m.
        var shore_d = 9.0;
        for (var k: i32 = 0; k < 8; k = k + 1) {
            // Jittered per pixel: fixed rings stepped the band in 11 cm terraces.
            let sj = hash12(floor(xz * 300.0));
            let a = f32(k) * 0.7854 + 0.39 + sj * 0.78;
            let o = vec2<f32>(cos(a), sin(a));
            for (var r: i32 = 1; r <= 10; r = r + 1) {
                let dist = (f32(r) - sj) * 0.11;
                let f = bc_film(xz + o * dist);
                if (f.r + f.g > 0.004) {
                    shore_d = min(shore_d, dist);
                }
            }
        }
        // 20-40 cm of dark wet sand round the whole shore, from the film.
        // 30-50 cm of damp sand that fades out into the dry, the outer edge
        // wandering with the grain.
        // A dark core to ~45 cm, then a soft fade into the dry by ~1 m.
        let band_w = mix(0.8, 1.05, noise(xz * 2.3 + vec2<f32>(1.0, 4.0)));
        // Width from the published pore moisture; a narrow core at the
        // waterline where the map is coarser than the band.
        let moist = bc_moisture(xz, 0.0);
        // 10-20 cm of dark wet sand from the contract's wet edge, then the
        // published pore moisture fades it out into the dry.
        let band = max((1.0 - smoothstep(0.06, 0.14, shore_d)) * (0.85 + 0.15 * band_w),
            smoothstep(0.85, 0.98, moist) * 0.5 * (1.0 - smoothstep(0.12, 0.25, shore_d)));
        // A glossy rim right at the water, a hand's width.
        let rim = 1.0 - smoothstep(0.01, 0.04, shore_d);
        // The published moisture only darkens where it is near saturation;
        // past the rim the sand dries back to its own colour.
        let margin = max(max(smoothstep(0.0005, 0.004, film_t), smoothstep(0.9, 0.99, wet) * 0.4), band);
        let soak = clamp(margin, 0.0, 1.0) * inside_t;
        // Under the film the water pass draws the depth; do not stack the two.
        let under_pre = smoothstep(0.002, 0.012, film_t);
        // Wet sand: darker and more saturated, the water fills the grain.
        let wet_lum = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
        // Wet sand at ~0.5 of the dry albedo, a little more saturated; under
        // the film a shade lighter, so the shallows show a sandy bed.
        let film_in = smoothstep(0.0, 0.0008, film_t);
        // Same grain, darker: wet sand at ~0.55 of the dry albedo, barely
        // more saturated, no brown cast (the old multiplier read as murk).
        let wet_col = max(mix(vec3<f32>(wet_lum), albedo, mix(1.05, 1.06, film_in)), vec3<f32>(0.0))
            * mix(vec3<f32>(0.57, 0.55, 0.52), vec3<f32>(0.87, 0.85, 0.8), film_in);
        albedo = mix(albedo, wet_col, soak);
        // A thin pale line of dried silt right where the film runs out, on
        // the land side only, broken along its length.
        let silt_line = (1.0 - smoothstep(0.0, 0.035, shore_d)) * (1.0 - smoothstep(0.003, 0.0045, film_t))
            * smoothstep(0.3, 0.55, noise(xz * 13.0) * 0.7 + noise(xz * 41.0) * 0.3) * inside_t;
        albedo = mix(albedo, vec3<f32>(0.56, 0.5, 0.41), silt_line * 0.35);
        // The gloss belongs to the band on dry land; under even a thin film
        // the water layer carries the reflection and the bed stays matte.
        let on_land = 1.0 - smoothstep(0.003, 0.0045, film_t);
        rough_out = mix(rough_out, mix(0.32, 0.12, rim), smoothstep(0.35, 0.9, soak) * on_land);
        // Water fills the ripples of the damp band: a smooth sheen, not a
        // field of bright crests.
        n = normalize(mix(n, n_geo, soak * 0.8 * (1.0 - under_pre)));
        // The margin owns the gloss on the sand: the old pore-film sheen read
        // as water a metre past the shore.
        if (sand_field()) {
            // Backlit, any sheen on the rippled sand lit every crest white.
            // Dry and damp sand stay matte; only the hand's-width rim at the
            // water is glossy.
            rough_out = max(rough_out, mix(mix(0.88, 0.75, band), 0.5, rim * on_land));
        }
        let under = smoothstep(0.002, 0.012, film_t);
        if (under > 0.0) {
            // The bed under clear water, lit by the light the rippled surface
            // focuses. Slurry kills the caustics.
            let slurry_f = film_c.g / max(film_t, 1e-4);
            let ratio = webgl_water_caustic(xz, film_t, frame.misc.x);
            // In the last centimetre the swell's caustic lines lay parallel
            // stripes on the rim; they build in with depth and break up in patches.
            let c_patch = smoothstep(0.3, 0.7, noise(xz * 1.6 + vec2<f32>(4.0, -3.0)));
            caustic = max(ratio - 0.8, 0.0) * 2.0 * (1.0 - slurry_f) * under * smoothstep(0.005, 0.02, film_t)
                * (1.0 - smoothstep(0.08, 0.2, film_t) * 0.6) * mix(0.6, 1.0, c_patch)
                // Where a pixel covers a whole caustic cell the pattern
                // averages into a flat milky wash over the bed: fade it.
                * (1.0 - smoothstep(0.008, 0.03, xz_fw));
            // Saturated: already x0.55 from the wet band above; a second
            // darkening printed a black rim along the shallows.
            // The water layer owns the specular; a backlit bed under a thin
            // film lit up every ripple crest white.
            rough_out = mix(rough_out, 0.95, under);
        }
    } else if (sand_field()) {
        // Past the contract window the sand keeps no pore-film sheen.
        rough_out = max(rough_out, 0.55);
    }
    // ---- Ruts and berms from the contract: a pressed, smoother floor and
    // crumbled spoil on the berm. The lug print itself is the soil's own
    // tread imprint (tread_delta), so it lines up with the tire.
    let rp = bc_rut_soft(xz);
    if ((rp.x > 0.004 || rp.y > 0.004) && (!track_near(xz) || sand_field())) {
        let rut_k = smoothstep(0.004, 0.025, rp.x);
        // Relief from the published cut and berm: their slope lights the
        // sunward wall and shades the other, so the groove reads on the bed.
        let re = select(0.12, 0.25, sand_field());
        let rx = bc_rut_soft(xz + vec2<f32>(re, 0.0));
        let rz = bc_rut_soft(xz + vec2<f32>(0.0, re));
        let hx = (rx.y - rx.x) - (rp.y - rp.x);
        let hz = (rz.y - rz.x) - (rp.y - rp.x);
        n = normalize(n + vec3<f32>(-hx / re, 0.0, -hz / re) * select(2.5, 0.9, sand_field()));
        // Packed and a little damper than the loose top: the floor goes darker,
        // so the lane reads from a distance.
        // On sand the pressed floor is wet sand: darker and more saturated,
        // with a sky sheen, never a black hole.
        if (sand_field()) {
            let r_lum = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
            let r_wet = max(mix(vec3<f32>(r_lum), albedo, 1.1), vec3<f32>(0.0)) * vec3<f32>(0.66, 0.6, 0.53);
            albedo = mix(albedo, r_wet, rut_k);
            rough_out = mix(rough_out, 0.22, rut_k);
        } else {
            albedo *= mix(1.0, 0.55, rut_k);
            // A pressed floor is smoother and holds water.
            rough_out = mix(rough_out, 0.35, rut_k);
        }
        n = normalize(mix(n, n_geo, rut_k * 0.3));
        let berm_k = smoothstep(0.004, 0.03, rp.y);
        if (berm_k > 0.0) {
            // Torn crumbs: a lumpy normal and a drier, lighter top.
            let bq = xz * 9.0;
            let b0 = noise(bq);
            let bx = noise(bq + vec2<f32>(0.3, 0.0));
            let bz = noise(bq + vec2<f32>(0.0, 0.3));
            n = normalize(n + vec3<f32>(b0 - bx, 0.0, b0 - bz) * 2.2 * berm_k);
            albedo = mix(albedo, albedo * vec3<f32>(1.12, 1.05, 0.92), berm_k * 0.5 * dry);
            rough_out = mix(rough_out, max(rough_out, 0.8), berm_k * dry);
        }
    }
    if (track_near(xz) && rut_water < 0.5) {
        // A mirror on the lug print was a white sheet over it. Keep the print
        // a damp matte unless water stands on it.
        rough_out = max(rough_out, mix(0.92, 0.5, damp));
    }
    if (kind == 10u && dry > 0.5) {
        // Dry sand: sparse quartz glints, one grain per texel cell.
        let cell = floor(xz * 140.0);
        let g = hash12(cell);
        if (g > 0.986) {
            rough_out = 0.06;
            let t = vec2<f32>(hash12(cell + vec2<f32>(3.1, 7.7)), hash12(cell + vec2<f32>(9.2, 1.3))) - 0.5;
            n = normalize(n + vec3<f32>(t.x, 0.0, t.y) * 0.8);
        }
    }
    // Wet / dry edge from the contract: a damp halo darkens the dry side and
    // a pale silt tide line marks where the water stopped. Broken by noise so
    // the 0.20 m cells do not print as a staircase.
    // No contract wet-edge tint here: on the sand field it was off, and on
    // main's scenes a 17th texture (binding 8) in fs_terrain segfaulted
    // lavapipe's fragment JIT on the LOD terrain (gpu-budget CI, #152).
    // The airborne plume (b6) shades the ground under it: optical depth
    // toward the sun through the published volume, as a soft darkening.
    if (bc_inside(xz) > 0.0) {
        let sun_g = normalize(frame.sun_dir.xyz);
        var od_g = 0.0;
        for (var j: i32 = 0; j < 6; j = j + 1) {
            od_g += bc_fog_at(in.world + sun_g * (0.2 + f32(j) * 0.5)) * 0.5;
        }
        albedo *= mix(1.0, 0.72, 1.0 - exp(-od_g * 60000.0 * 0.45));
    }
    // Settled dust, metres. A tenth of a millimetre greys the soil, a couple
    // of millimetres hide it. Dust is matte and fills the grain.
    let dust_m = bc_dust_settled(xz);
    if (dust_m > 0.0) {
        let speck = noise(xz * 11.0) * 0.6 + noise(xz * 37.0) * 0.4;
        let cover = smoothstep(0.00004, 0.0018, dust_m * (0.6 + 0.8 * speck));
        // On wet ground the dust turns to a dark smear, not a pale powder.
        let dust_col = mix(vec3<f32>(0.40, 0.34, 0.27), vec3<f32>(0.16, 0.12, 0.08), smoothstep(0.35, 0.7, wet));
        // Dust off this sand is the sand's own colour, a shade lighter; a
        // neutral grey read as lilac sky-sheen next to the warm grain.
        let own_dust = mix(albedo * vec3<f32>(1.1, 1.08, 1.04), dust_col, select(0.2, 0.0, sand_field()));
        albedo = mix(albedo, own_dust, cover * 0.35);
        rough_out = mix(rough_out, 0.95, cover * (1.0 - smoothstep(0.35, 0.7, wet)));
        n = normalize(mix(n, n_geo, cover * 0.25));
    }
    // Specular AA: a normal that turns faster than a pixel widens the lobe,
    // so a wet grain far away reads as sheen, not white sparkle.
    let dndx = dpdx(n);
    let dndy = dpdy(n);
    let variance = 0.25 * (dot(dndx, dndx) + dot(dndy, dndy));
    rough_out = sqrt(clamp(rough_out * rough_out + min(2.0 * variance, 0.2), 0.0, 1.0));
    var emit = vec3<f32>(0.0);
    let film_b = film_c.r + film_c.g;
    if (bc_inside(xz) > 0.5 && film_b > 0.003) {
        // Sunlight refracted through the ripples lands a little off where a
        // flat surface would put it; exaggerated so the shadow edge on the
        // bed wobbles visibly.
        let sun_d = normalize(frame.sun_dir.xyz);
        let wn = bw_wave_normal(xz, frame.misc.x);
        let r_w = refract(-sun_d, wn, 1.0 / 1.333);
        let r_f = refract(-sun_d, vec3<f32>(0.0, 1.0, 0.0), 1.0 / 1.333);
        let off = (r_w.xz / min(r_w.y, -0.2) - r_f.xz / min(r_f.y, -0.2)) * min(film_b, 0.1) * 30.0;
        g_shadow_wobble = vec3<f32>(off.x, 0.0, off.y);
        let deep_s = smoothstep(0.005, 0.06, film_b);
        g_shadow_soft = vec2<f32>(0.07 * deep_s, 0.15 * deep_s);
    }
    if (caustic > 0.01) {
        let sun_d = normalize(frame.sun_dir.xyz);
        let s_b = shadow_soft(in.world + n_geo * 0.02 + g_shadow_wobble, n_geo);
        // Caustics crowd along the shadow's edge, where the bent light
        // folds over the penumbra.
        let edge = 4.0 * s_b * (1.0 - s_b);
        let sun_k = clamp(sun_d.y, 0.0, 1.0) * s_b;
        emit = albedo * frame.sun_color.rgb * caustic * (sun_k + edge * 0.8) * 2.4;
    }
    if (frame.debug.x > 8.5) {
        // 9: published moisture (r), film > 0 (g), the spread wet term (b).
        return vec4<f32>(from_display(vec3<f32>(bc_moisture(xz, 0.0), select(0.0, 1.0, film_c.r + film_c.g > 0.0005), wet)), 1.0);
    }
    // Mud on the sand until soil_mix exists: where the contract publishes
    // slurry, or the pore moisture is saturated in a tire cut, the sand is
    // a dark glossy slurry. Read from binding 3 G, binding 2 and binding 1 G.
    if (sand_field() && bc_inside(xz) > 0.0) {
        let fc_m = bc_film(xz);
        let cut_m = bc_rut_soft(xz).x;
        let mud_k = max(smoothstep(0.0003, 0.003, fc_m.g),
            smoothstep(0.95, 0.995, bc_moisture(xz, 0.0)) * smoothstep(0.004, 0.02, cut_m)) * bc_inside(xz);
        if (mud_k > 0.0) {
            let lum_m = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
            let slurry_col = vec3<f32>(0.2, 0.15, 0.1) * (0.75 + 0.5 * lum_m);
            albedo = mix(albedo, slurry_col, mud_k);
            rough_out = mix(rough_out, 0.18, mud_k);
            n = normalize(mix(n, n_geo, mud_k * 0.7));
        }
    }
    // ---- Requested contract fields (docs/beauty/CONTRACT_REQUESTS.md). Dark
    // until the solver publishes them: both readers return 0 today.
    // Soil seams: blend the ground layers by the published fractions, and
    // the seam stays wet where the moisture map says so.
    let soil_mix = bc_soil_mix(xz);
    let mix_sum = dot(soil_mix, vec4<f32>(1.0));
    if (mix_sum > 0.01) {
        // #107 order: R mud, G sand, B gravel, A grass.
        let fr = soil_mix / max(mix_sum, 1.0);
        let c_mud = textureSampleLevel(ground_diff, ground_samp, layer_uv(5, xz), 5, 0.0).rgb;
        let c_sand = textureSampleLevel(ground_diff, ground_samp, layer_uv(6, xz), 6, 0.0).rgb;
        let c_grav = textureSampleLevel(ground_diff, ground_samp, layer_uv(4, xz), 4, 0.0).rgb;
        let c_grass = textureSampleLevel(ground_diff, ground_samp, layer_uv(3, xz), 3, 0.0).rgb;
        let seam_col = c_mud * fr.x + c_sand * fr.y + c_grav * fr.z + c_grass * fr.w;
        let seam_wet = smoothstep(0.6, 0.95, bc_moisture(xz, 0.0));
        albedo = mix(albedo, seam_col * mix(1.0, 0.55, seam_wet), min(mix_sum, 1.0));
        rough_out = mix(rough_out, 0.35, seam_wet * fr.x);
    }
    // Snow: depth covers the ground, compaction turns fresh powder into
    // grey packed snow with a satin sheen, the belly scrape shows ice.
    let snow_c = bc_snow(xz);
    if (snow_c.r > 0.0) {
        let cover = smoothstep(0.0, 0.03, snow_c.r);
        let fresh = vec3<f32>(0.86, 0.89, 0.93);
        let packed = vec3<f32>(0.62, 0.66, 0.71);
        let scrape = smoothstep(0.0, 0.02, snow_c.a);
        albedo = mix(albedo, mix(mix(fresh, packed, snow_c.b), vec3<f32>(0.55, 0.6, 0.66), scrape), cover);
        rough_out = mix(rough_out, mix(mix(0.9, 0.55, snow_c.b), 0.25, scrape), cover);
        n = normalize(mix(n, n_geo, cover * 0.6));
    }
    let lit = shade_pbr(in.world, n, albedo, rough_out, 0.0, mix(1.0, ao, 0.6) * niva_ground_ao(in.world), emit);
    return present(lit);
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash12(p), hash12(p + vec2<f32>(17.3, 5.1)));
}

// Border distance of a jittered cell net (F2 - F1).
fn cell_border(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let g = vec2<f32>(f32(x), f32(y));
            let r = g + hash22(i + g) - f;
            let d = dot(r, r);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    return sqrt(d2) - sqrt(d1);
}

// Drying cracks. A straight-edged cell net reads as tiles, so the domain is
// warped by low noise and a second, finer net adds the short branches that
// split a plate. The value is distance to the nearest crack.
fn crack_edge(p: vec2<f32>) -> f32 {
    let warp = vec2<f32>(noise(p * 0.9 + vec2<f32>(3.7, 1.1)), noise(p * 0.9 + vec2<f32>(8.3, 4.6))) - 0.5;
    let q = p + warp * 0.55;
    let main = cell_border(q);
    let fine_warp = vec2<f32>(noise(q * 2.4), noise(q * 2.4 + vec2<f32>(5.2, 9.4))) - 0.5;
    let fine = cell_border(q * 2.3 + fine_warp * 0.6 + vec2<f32>(0.37, 0.71));
    // Branches only where the noise lets a plate split, and thinner.
    let split = smoothstep(0.45, 0.7, noise(q * 1.3 + vec2<f32>(11.0, 2.0)));
    return min(main, mix(1.0, fine * 1.6, split));
}

struct WaterOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) ij: vec2<f32>,
    @location(2) depth: f32,
}

fn water_depth(xz: vec2<f32>) -> f32 {
    let raw = mat_bilinear(xz).g;
    if (win_demo_flat_terrain()) {
        return raw * select(0.34, 0.28, puddle_beauty_on());
    }
    // Dilate slightly so a dry speck inside the bowl does not punch a black hole.
    let e = 0.28;
    let d0 = raw;
    let d1 = mat_bilinear(xz + vec2<f32>(e, 0.0)).g;
    let d2 = mat_bilinear(xz - vec2<f32>(e, 0.0)).g;
    let d3 = mat_bilinear(xz + vec2<f32>(0.0, e)).g;
    let d4 = mat_bilinear(xz - vec2<f32>(0.0, e)).g;
    return max(d0, max(d1, max(d2, max(d3, d4)))) * 0.34;
}

// Solved column for a crossing. misc.w selects the wider pack so a bow
// above the old 0.34 m film range is the free surface, not a flat sheet.
fn water_column(xz: vec2<f32>) -> f32 {
    let raw = mat_bilinear(xz).g;
    if (puddle_beauty_on()) {
        // Mat.g is 0..1 over 0.28 m (see upload_soil liquid_scale).
        return raw * 0.28;
    }
    let scale = select(0.34, 0.90, water_column_on());
    return raw * scale;
}

// A dry patch with water on both sides is a hole in the sheet, not a shore.
// The outer bank has water on only one side, so this does not grow the pond.
fn covered_depth(xz: vec2<f32>) -> f32 {
    let here = water_depth(xz);
    if (win_demo_flat_terrain()) {
        return here;
    }
    let e = 1.35;
    let px = water_depth(xz + vec2<f32>(e, 0.0));
    let nx = water_depth(xz - vec2<f32>(e, 0.0));
    let pz = water_depth(xz + vec2<f32>(0.0, e));
    let nz = water_depth(xz - vec2<f32>(0.0, e));
    let across = max(min(px, nx), min(pz, nz));
    return max(here, across);
}

// Metres above the bed. The solved column can be most of a metre; drawing
// that is a mound. The pool stays a film. Only the tallest part of the bow
// stands up, so the crest is a wave in the surface.
fn shallow_water(column: f32) -> f32 {
    // A film on the ground. The bow adds a low crest, not a mound.
    // The solved column is left in the simulation; only a fraction is drawn.
    let film = min(column, 0.11) * 0.10;
    let extra = max(column - 0.11, 0.0);
    let t = clamp(extra / 0.58, 0.0, 1.0);
    let wave = t * 0.045;
    return 0.003 + film + wave;
}

fn render_flags() -> u32 {
    return u32(frame.misc.w + 0.5);
}

fn water_column_on() -> bool {
    return (render_flags() & 1u) != 0u;
}

fn puddle_beauty_on() -> bool {
    return (render_flags() & 2u) != 0u;
}

fn win_demo_flat_terrain() -> bool {
    return (render_flags() & 4u) != 0u;
}

// Open water (river, sea shore): the column is the free surface.
fn open_water_on() -> bool {
    return (render_flags() & 32u) != 0u;
}

fn skip_tire_seat() -> bool {
    return puddle_beauty_on() || win_demo_flat_terrain();
}

// Win-demo stills share one terrain grid with the water sheet. One hot
// vertex in vs_water was stretching whole triangles into sky spikes.
fn win_demo_clamp_surface_y(xz: vec2<f32>, y: f32) -> f32 {
    if (!win_demo_flat_terrain()) {
        return y;
    }
    let level = win_demo_pool_level();
    // Flat pool top only — do not drape the sheet on bowl side walls.
    return min(y, level + 0.15);
}

// Windows win-demo seam pool: one flat plane, not a carved bowl mesh.
fn beauty_seam_center() -> vec2<f32> {
    return vec2<f32>(-1.05, 0.15);
}

// CPU seam pan ellipse — one continuous column for the hero pool (not per-tire pads).
fn win_demo_pool_level() -> f32 {
    return 0.22;
}

fn win_demo_seam_column(xz: vec2<f32>) -> f32 {
    if (!win_demo_flat_terrain()) {
        return 0.0;
    }
    let c = beauty_seam_center();
    let dx = (xz.x - c.x) / 1.32;
    let dz = (xz.y - c.y) / 1.02;
    let r = sqrt(dx * dx + dz * dz);
    if (r > 1.40) {
        return 0.0;
    }
    let core = clamp(1.0 - r / 1.15, 0.0, 1.0);
    let edge = smoothstep(1.40, 0.88, r);
    return (0.05 + core * core * 0.22) * edge;
}

fn win_demo_in_pool(xz: vec2<f32>) -> bool {
    let c = beauty_seam_center();
    let dx = (xz.x - c.x) / 1.32;
    let dz = (xz.y - c.y) / 1.02;
    return sqrt(dx * dx + dz * dz) < 1.40;
}

fn beauty_seam_weight(xz: vec2<f32>) -> f32 {
    if (!puddle_beauty_on()) {
        return 0.0;
    }
    let c = beauty_seam_center();
    let dx = (xz.x - c.x) / 1.05;
    let dz = (xz.y - c.y) / 0.92;
    let d = sqrt(dx * dx + dz * dz);
    return smoothstep(1.05, 0.38, d);
}

fn beauty_seam_plane_y(xz: vec2<f32>) -> f32 {
    let c = beauty_seam_center();
    let bed = (
        height_bilinear(c)
        + height_bilinear(c + vec2<f32>(0.55, 0.0))
        + height_bilinear(c - vec2<f32>(0.55, 0.0))
        + height_bilinear(c + vec2<f32>(0.0, 0.55))
        + height_bilinear(c - vec2<f32>(0.0, 0.55))
    ) * 0.2;
    // Win-demo stills need one level sheet; tilt reads as a dry berm under the chassis.
    let tilt = select(
        (height_bilinear(xz) - height_bilinear(c)) * 0.08,
        0.0,
        win_demo_flat_terrain(),
    );
    return bed + 0.008 + tilt;
}

// Playground pools: bed + column is the flat free surface. Do not inflate the
// column into a mound — depth shows in the tint, not vertex height.
fn beauty_pool_film(column: f32) -> f32 {
    return 0.003 + min(column, 0.24) * 0.042;
}

fn water_surface_lift(column: f32) -> f32 {
    if (puddle_beauty_on()) {
        return beauty_pool_film(column);
    }
    return shallow_water(column);
}

fn sky_reflect(dir: vec3<f32>) -> vec3<f32> {
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    return mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, pow(t, 0.62));
}

// Schlick Fresnel (R0 ≈ 0.02 air→water). Ultra-Water-style sky/body split.
fn schlick_fresnel(cos_theta: f32) -> f32 {
    let R0 = 0.02;
    let c = clamp(cos_theta, 0.0, 1.0);
    return R0 + (1.0 - R0) * pow(1.0 - c, 5.0);
}

fn solver_column_grad(xz: vec2<f32>) -> vec2<f32> {
    let e = 0.22;
    let c0 = water_column(xz);
    let cx = water_column(xz + vec2<f32>(e, 0.0)) - c0;
    let cz = water_column(xz + vec2<f32>(0.0, e)) - c0;
    return vec2<f32>(cx, cz) / e;
}

// Standing rings from the solved column (shore + contact), not vertex dilation.
fn solver_shock_rings(xz: vec2<f32>, t: f32) -> f32 {
    let g = solver_column_grad(xz);
    let edge = clamp(length(g) * 4.5, 0.0, 1.0);
    var sum = 0.0;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let hub = frame.tires[i * 2u].xyz.xz;
        let d = distance(xz, hub);
        if (d > 2.8 || d < 0.05) {
            continue;
        }
        let phase = d * 14.5 - t * 11.0 + f32(i) * 0.7;
        let ring = sin(phase) * exp(-d * 1.65);
        let ring2 = sin(d * 7.2 - t * 5.5 + f32(i)) * exp(-d * 0.95) * 0.45;
        sum += (ring + ring2) * (0.35 + edge * 0.85);
    }
    return sum;
}

fn scene_uv_from_world(p: vec3<f32>) -> vec2<f32> {
    let clip = frame.view_proj * vec4<f32>(p, 1.0);
    if (clip.w <= 0.02) {
        return vec2<f32>(-1.0, -1.0);
    }
    var uv = clip.xy / clip.w;
    return vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5);
}

fn pool_reflect_ready() -> bool {
    return win_demo_flat_terrain() && (u32(frame.misc.w) & 8u) != 0u;
}

fn pool_reflect_geom_pass() -> bool {
    return (u32(frame.misc.w) & 16u) != 0u;
}

fn pool_reflect_mirror_world(world: vec4<f32>) -> vec4<f32> {
    if (!pool_reflect_geom_pass()) {
        return world;
    }
    let plane = win_demo_pool_level();
    var p = world;
    p.y = 2.0 * plane - p.y;
    return p;
}

fn pool_reflect_uv(x: f32, z: f32) -> vec2<f32> {
    let level = win_demo_pool_level();
    let clip = frame.reflect_view_proj * vec4<f32>(x, level, z, 1.0);
    if (clip.w <= 0.02) {
        return vec2<f32>(-1.0, -1.0);
    }
    let ndc = clip.xy / clip.w;
    return vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 + ndc.y * 0.5);
}

fn pool_reflect_tex_sample(uv: vec2<f32>) -> vec3<f32> {
    if (uv.x < 0.005 || uv.x > 0.995 || uv.y < 0.005 || uv.y > 0.995) {
        return vec3<f32>(0.0);
    }
    let dims = textureDimensions(pool_reflect_tex);
    let tc = vec2<i32>(
        i32(clamp(uv.x, 0.0, 1.0) * f32(dims.x - 1u)),
        i32(clamp(uv.y, 0.0, 1.0) * f32(dims.y - 1u)),
    );
    return scene_to_display(textureSampleLevel(pool_reflect_tex, scene_samp, uv, 0.0).rgb);
}

// Win-demo pool: sample the mirrored-geometry pass (Niva + sky), not SSR over terrain.
fn pool_planar_reflect_sample(world: vec3<f32>) -> vec3<f32> {
    if (!pool_reflect_ready()) {
        return vec3<f32>(0.0);
    }
    var center = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        center += frame.tires[i * 2u].xyz;
    }
    center = center * 0.25 + vec3<f32>(0.0, 0.68, 0.0);
    let uv_plane = pool_reflect_uv(world.x, world.z);
    let uv_body = pool_reflect_uv(center.x, center.z);
    let s_plane = pool_reflect_tex_sample(uv_plane);
    let s_body = pool_reflect_tex_sample(uv_body);
    let under = smoothstep(0.95, 0.18, distance(world.xz, center.xz));
    if (under > 0.45 && length(s_body) > 0.03) {
        return s_body;
    }
    if (length(s_plane) > 0.03) {
        return s_plane;
    }
    return s_body;
}

// wgpu water.wgsl-style screen reflection: march in reflection direction and
// sample the pre-water colour buffer (Niva + sky already drawn).
fn pool_scene_reflect(world: vec3<f32>, nrm: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let refl = reflect(-view, nrm);
    var accum = vec3<f32>(0.0);
    var wsum = 0.0;
    var p = world + nrm * 0.014;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        p = p + refl * 0.07;
        let uv = scene_uv_from_world(p);
        if (uv.x > 0.03 && uv.x < 0.97 && uv.y > 0.03 && uv.y < 0.97) {
            let s = scene_to_display(textureSampleLevel(scene_color, scene_samp, uv, 0.0).rgb);
            let w = 1.0 / (1.0 + f32(i) * 0.22);
            accum += s * w;
            wsum += w;
        }
    }
    if (wsum > 0.02) {
        return accum / wsum;
    }
    return sky_reflect(refl);
}

fn scene_ssr(world: vec3<f32>, nrm: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    if (!puddle_beauty_on() && !win_demo_flat_terrain()) {
        return vec3<f32>(0.0);
    }
    let t = frame.misc.x;
    let e = 0.05;
    let r0 = solver_shock_rings(world.xz, t);
    let rx = solver_shock_rings(world.xz + vec2<f32>(e, 0.0), t) - r0;
    let rz = solver_shock_rings(world.xz + vec2<f32>(0.0, e), t) - r0;
    let n_pert = normalize(nrm + vec3<f32>(-rx * 4.0, 0.0, -rz * 4.0));
    let refl = reflect(-view, n_pert);
    var p = world + nrm * 0.018;
    var hit_uv = vec2<f32>(0.0);
    var found = false;
    for (var step: u32 = 0u; step < 14u; step = step + 1u) {
        p = p + refl * 0.11;
        let uv = scene_uv_from_world(p);
        if (uv.x > 0.02 && uv.x < 0.98 && uv.y > 0.02 && uv.y < 0.98) {
            hit_uv = uv;
            found = true;
        }
    }
    if (!found) {
        return vec3<f32>(0.0);
    }
    return scene_to_display(textureSampleLevel(scene_color, scene_samp, hit_uv, 0.0).rgb);
}

fn reflect_vehicle_scene(world: vec3<f32>, nrm: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let demo_side = win_demo_flat_terrain();
    var center = vec3<f32>(0.0);
    var live = 0.0;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let slot = i * 2u;
        center += frame.tires[slot].xyz;
        live = max(live, frame.tires[slot + 1u].w);
    }
    center = center * 0.25 + vec3<f32>(0.0, 0.68, 0.0);
    if (live < 0.05) {
        return vec3<f32>(0.0);
    }
    let refl = reflect(-view, nrm);
    var accum = vec3<f32>(0.0);
    var wt = 0.0;
    let pts = array<vec3<f32>, 3>(
        center,
        center + vec3<f32>(-0.42, 0.12, 0.18),
        center + vec3<f32>(0.38, 0.08, -0.22),
    );
    for (var pi: u32 = 0u; pi < 3u; pi = pi + 1u) {
        let p = pts[pi];
        let to_c = p - world;
        let aim = max(dot(refl, normalize(to_c)), 0.0);
        let beauty = puddle_beauty_on();
        if (aim < select(select(0.04, 0.012, beauty), 0.008, demo_side)) {
            continue;
        }
        let veh_uv = scene_uv_from_world(p);
        if (veh_uv.x < 0.02 || veh_uv.x > 0.98 || veh_uv.y < 0.02 || veh_uv.y > 0.98) {
            continue;
        }
        let samp = scene_to_display(textureSampleLevel(scene_color, scene_samp, veh_uv, 0.0).rgb);
        let w = pow(aim, select(select(0.28, 0.18, beauty), 0.12, demo_side));
        accum += samp * w;
        wt += w;
    }
    if (wt < 0.04) {
        return vec3<f32>(0.0);
    }
    let beauty = puddle_beauty_on();
    return (accum / wt) * select(select(1.45, 1.85, beauty), 2.05, demo_side);
}

fn beauty_bow_wave(xz: vec2<f32>, t: f32) -> f32 {
    if (!puddle_beauty_on() && !win_demo_flat_terrain()) {
        return 0.0;
    }
    var sum = 0.0;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let slot = i * 2u;
        let c = frame.tires[slot].xyz;
        let tire_meta = frame.tires[slot + 1u];
        if (tire_meta.w < 0.12) {
            continue;
        }
        if (win_demo_flat_terrain()) {
            if (i >= 2u || tire_meta.w < 2.0) {
                continue;
            }
        }
        let fwd = normalize(vec2<f32>(tire_meta.x, tire_meta.y));
        let rel = xz - c.xz;
        let dist = length(rel);
        if (dist > 1.35 || dist < 0.04) {
            continue;
        }
        let dir = rel / max(dist, 1.0e-4);
        let ahead = dot(dir, fwd);
        if (ahead < 0.08) {
            continue;
        }
        let drive = select(tire_meta.z, tire_meta.w * 0.0038, win_demo_flat_terrain());
        let bow_scale = select(1.0, 2.35, win_demo_flat_terrain());
        let bow = exp(-dist * 1.9) * ahead * (0.038 + drive * 0.09) * bow_scale;
        sum += bow * sin(dist * 9.2 - t * 13.5 - ahead * 6.0 + f32(i));
    }
    return sum;
}

fn niva_planar_reflect(world: vec3<f32>, nrm: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let plane_y = select(world.y, win_demo_pool_level(), win_demo_flat_terrain());
    let refl = reflect(-view, nrm);
    var accum = vec3<f32>(0.0);
    var wt = 0.0;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let slot = i * 2u;
        let hub = frame.tires[slot].xyz;
        let tire_meta = frame.tires[slot + 1u];
        if (tire_meta.w < 0.2) {
            continue;
        }
        let body = hub + vec3<f32>(0.0, 0.78, 0.0);
        let mirror = vec3<f32>(body.x, 2.0 * plane_y - body.y, body.z);
        let to_m = normalize(mirror - world);
        let aim = max(dot(refl, to_m), 0.0);
        let aim_gate = select(0.22, 0.06, win_demo_flat_terrain());
        if (aim < aim_gate) {
            continue;
        }
        let uv = scene_uv_from_world(mirror);
        if (uv.x < 0.04 || uv.x > 0.96 || uv.y < 0.04 || uv.y > 0.96) {
            continue;
        }
        let samp = scene_to_display(textureSampleLevel(scene_color, scene_samp, uv, 0.0).rgb);
        let w = aim * (0.55 + tire_meta.w * 0.45);
        accum += samp * w;
        wt += w;
    }
    if (wt < 0.05) {
        return vec3<f32>(0.0);
    }
    return (accum / wt) * 1.08;
}

fn niva_water_reflect(world: vec3<f32>, refl_dir: vec3<f32>) -> vec3<f32> {
    var center = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        center += frame.tires[i * 2u].xyz;
    }
    center = center * 0.25 + vec3<f32>(0.0, 0.58, 0.0);
    let to_veh = normalize(center - world);
    let aim = dot(normalize(refl_dir), to_veh);
    if (aim < 0.22) {
        return vec3<f32>(0.0);
    }
    let d = distance(world.xz, center.xz);
    let gloss = exp(-d * 0.22) * smoothstep(0.22, 0.72, aim);
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(to_veh, sun), 0.0, 1.0);
    let tan = vec3<f32>(0.72, 0.60, 0.44);
    return tan * (0.28 + 1.05 * ndl) * gloss;
}

fn water_near_tire_hub(xz: vec2<f32>, radius: f32) -> bool {
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let c = frame.tires[i * 2u].xyz.xz;
        if (distance(xz, c) < radius) {
            return true;
        }
    }
    return false;
}

fn tire_wake_ripples(xz: vec2<f32>, t: f32) -> f32 {
    let beauty = puddle_beauty_on();
    let demo = win_demo_flat_terrain();
    var sum = 0.0;
    if (!demo) {
        sum = solver_shock_rings(xz, t);
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let slot = i * 2u;
        let c = frame.tires[slot].xyz;
        let tire_meta = frame.tires[slot + 1u];
        let spin_rads = tire_meta.w;
        var tire_on = select(spin_rads, select(1.0, 0.0, spin_rads > 0.35), spin_rads <= 2.0);
        if (demo && spin_rads > 2.0) {
            if (i >= 2u) {
                continue;
            }
            tire_on = 1.0;
        }
        let d = distance(xz, c.xz);
        if ((beauty || demo) && d < 3.6) {
            tire_on = max(tire_on, 1.0 - smoothstep(0.35, 3.2, d));
        }
        if ((beauty || demo) && tire_meta.z > 0.015 && d < 2.4) {
            if (!demo || spin_rads > 2.0) {
                tire_on = 1.0;
            }
        }
        if (tire_on < 0.05) {
            continue;
        }
        let col_hub = water_column(c.xz);
        let near = exp(-d * 3.2);
        let ring = near * sin(d * 11.5 - t * 10.5 + f32(i) * 0.9);
        let ring2 = exp(-d * 1.8) * sin(d * 6.2 - t * 6.0 + f32(i) * 1.3) * 0.55;
        let splash = exp(-d * 4.0) * sin(d * 18.0 - t * 15.0) * 0.022;
        let contact = tire_meta.z * 22.0 + col_hub * 6.0;
        let amp = select(1.0, 5.8 + contact * 1.8, beauty);
        let spin_fac = select(0.0, clamp((spin_rads - 2.0) * 0.07, 0.0, 1.0), demo && spin_rads > 2.0);
        let demo_amp = select(amp, 1.02 + contact * 0.05 + spin_fac * 0.28, demo);
        sum = sum + (ring * 0.078 + ring2 * 0.048 + splash) * tire_on * demo_amp;
    }
    return sum;
}

fn beauty_tire_splash(xz: vec2<f32>, t: f32) -> vec3<f32> {
    if (win_demo_flat_terrain()) {
        return vec3<f32>(0.0);
    }
    if (!puddle_beauty_on() && !win_demo_flat_terrain()) {
        return vec3<f32>(0.0);
    }
    var spray = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let slot = i * 2u;
        let c = frame.tires[slot].xyz;
        let tire_meta = frame.tires[slot + 1u];
        if (tire_meta.w < 0.08) {
            continue;
        }
        let rel = xz - c.xz;
        let d = length(rel);
        if (d > 0.92 || d < 0.06) {
            continue;
        }
        let ang = atan2(rel.y, rel.x);
        let fan = smoothstep(0.18, 0.55, tire_meta.z) * (0.5 + 0.5 * sin(ang * 5.0 - t * 14.0 + f32(i)));
        let lift = pow(smoothstep(0.72, 0.12, d), 1.6);
        let drop = sin(d * 28.0 - t * 19.0 + f32(i) * 2.1) * 0.5 + 0.5;
        spray += vec3<f32>(0.86, 0.94, 1.0) * fan * lift * drop * exp(-d * 2.2) * 0.95;
    }
    return spray;
}

fn puddle_normal(xz: vec2<f32>, t: f32, column: f32) -> vec3<f32> {
    let e = 0.048;
    let bow = beauty_bow_wave(xz, t);
    let demo_wake = win_demo_flat_terrain() && t > 5.0;
    let wake = select(tire_wake_ripples(xz, t), select(0.0, tire_wake_ripples(xz, t), demo_wake), win_demo_flat_terrain());
    let h0 = ripple_height(xz, t) + wake + bow;
    let hx = ripple_height(xz + vec2<f32>(e, 0.0), t)
        + select(tire_wake_ripples(xz + vec2<f32>(e, 0.0), t), select(0.0, tire_wake_ripples(xz + vec2<f32>(e, 0.0), t), demo_wake), win_demo_flat_terrain())
        + beauty_bow_wave(xz + vec2<f32>(e, 0.0), t);
    let hz = ripple_height(xz + vec2<f32>(0.0, e), t)
        + select(tire_wake_ripples(xz + vec2<f32>(0.0, e), t), select(0.0, tire_wake_ripples(xz + vec2<f32>(0.0, e), t), demo_wake), win_demo_flat_terrain())
        + beauty_bow_wave(xz + vec2<f32>(0.0, e), t);
    let g = select(free_swell_grad(xz, t), vec2<f32>(0.0, 0.0), puddle_beauty_on());
    let norm_scale = select(2.4, 1.35, win_demo_flat_terrain());
    let dx = (hx - h0) / e + g.x * 0.85;
    let dz = (hz - h0) / e + g.y * 0.85;
    let n = normalize(vec3<f32>(-dx * norm_scale, 1.0, -dz * norm_scale));
    let rip = ripple_normal(xz, t);
    let w = smoothstep(0.004, 0.06, column);
    return normalize(mix(rip, n, w * 0.65 + 0.2));
}

// Long swells on the free surface. A few centimetres, so the sheet stays
// on the ground. A tight lattice would read as a checker.
fn free_swell(xz: vec2<f32>, t: f32) -> f32 {
    return 0.11 * sin(xz.x * 0.72 + xz.y * 1.05 + t * 0.7)
        + 0.07 * sin(xz.y * 1.45 - xz.x * 0.28 - t * 0.45);
}

fn free_swell_grad(xz: vec2<f32>, t: f32) -> vec2<f32> {
    let a = cos(xz.x * 0.72 + xz.y * 1.05 + t * 0.7);
    let b = cos(xz.y * 1.45 - xz.x * 0.28 - t * 0.45);
    let dx = 0.11 * a * 0.72 - 0.07 * b * 0.28;
    let dz = 0.11 * a * 1.05 + 0.07 * b * 1.45;
    return vec2<f32>(dx, dz);
}

fn water_bed(xz: vec2<f32>) -> f32 {
    // Blur a single-cell rut out of the sheet so the pool stays one surface.
    let e = 0.55;
    return (
        height_bilinear(xz)
        + height_bilinear(xz + vec2<f32>(e, 0.0))
        + height_bilinear(xz - vec2<f32>(e, 0.0))
        + height_bilinear(xz + vec2<f32>(0.0, e))
        + height_bilinear(xz - vec2<f32>(0.0, e))
    ) * 0.2;
}

// Screen-space reflection off the water: march the reflected ray in world
// space, project each step, and stop where it passes behind the opaque
// frame's depth. Returns the copied (display) colour and a hit weight that
// fades at the screen edge and with distance. The sky and far ground stay
// the env map and the bank march.
fn water_ssr(world: vec3<f32>, dir: vec3<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(post_depth));
    // Per-ray jitter of the march start: the fixed geometric steps cut the
    // mirror image into horizontal bands; jittered, they average out.
    let jit = hash12(world.xz * 731.0 + dir.xz * 97.0);
    var t = 0.04 + jit * 0.08;
    var t_prev = 0.0;
    for (var i: i32 = 0; i < 28; i = i + 1) {
        let p = world + dir * t;
        let c = frame.view_proj * vec4<f32>(p, 1.0);
        if (c.w <= 0.01) {
            break;
        }
        let ndc = c.xyz / c.w;
        let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
            break;
        }
        let pix = vec2<i32>(uv * dims);
        let d = textureLoad(post_depth, pix, 0).x;
        if (d < 0.9999 && ndc.z > d) {
            // Behind the surface: compare distances so a thin object does
            // not swallow everything behind it.
            let sw = frame.inv_view_proj * vec4<f32>(ndc.xy, d, 1.0);
            let sp = sw.xyz / sw.w;
            let gap = length(p - frame.cam_pos.xyz) - length(sp - frame.cam_pos.xyz);
            // A hit below the surface it starts from is a submerged part
            // seen through the water in the copy; the mirror ends at the
            // waterline.
            if (sp.y < world.y - 0.01) {
                return vec4<f32>(0.0);
            }
            if (gap < 0.35 + t * 0.08) {
                // Binary search between the last step in front and this one
                // behind: the coarse geometric steps landed on the same few
                // rows and cut the mirror into horizontal blocks.
                var lo = t_prev;
                var hi = t;
                for (var b: i32 = 0; b < 5; b = b + 1) {
                    let tm = 0.5 * (lo + hi);
                    let cb = frame.view_proj * vec4<f32>(world + dir * tm, 1.0);
                    let nb = cb.xyz / cb.w;
                    let ub = clamp(vec2<f32>(nb.x * 0.5 + 0.5, 0.5 - nb.y * 0.5), vec2<f32>(0.0), vec2<f32>(0.9999));
                    let db = textureLoad(post_depth, vec2<i32>(ub * dims), 0).x;
                    if (db < 0.9999 && nb.z > db) {
                        hi = tm;
                    } else {
                        lo = tm;
                    }
                }
                let cm = frame.view_proj * vec4<f32>(world + dir * hi, 1.0);
                let uvm = vec2<f32>(cm.x / cm.w * 0.5 + 0.5, 0.5 - cm.y / cm.w * 0.5);
                let uvh = clamp(uvm, vec2<f32>(0.0), vec2<f32>(0.9999));
                let ph = vec2<i32>(uvh * dims);
                let dmax = vec2<i32>(dims) - vec2<i32>(1);
                var col = scene_to_display(textureLoad(post_color, ph, 0).rgb) * 0.4;
                col += scene_to_display(textureLoad(post_color, clamp(ph + vec2<i32>(2, 0), vec2<i32>(0), dmax), 0).rgb) * 0.15;
                col += scene_to_display(textureLoad(post_color, clamp(ph - vec2<i32>(2, 0), vec2<i32>(0), dmax), 0).rgb) * 0.15;
                col += scene_to_display(textureLoad(post_color, clamp(ph + vec2<i32>(0, 2), vec2<i32>(0), dmax), 0).rgb) * 0.15;
                col += scene_to_display(textureLoad(post_color, clamp(ph - vec2<i32>(0, 2), vec2<i32>(0), dmax), 0).rgb) * 0.15;
                let edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
                // Fade to the IBL sky toward the screen edge and with the
                // gap, so a ray leaving the frame blends out, not cuts off.
                // The waterline cut-off, soft and from the refined hit: a
                // binary test on the coarse texel cut the mirror into stairs.
                let hz = (frame.view_proj * vec4<f32>(world + dir * hi, 1.0)).z / cm.w;
                let hw = frame.inv_view_proj * vec4<f32>(uvh.x * 2.0 - 1.0, 1.0 - uvh.y * 2.0, max(textureLoad(post_depth, ph, 0).x, hz), 1.0);
                let sub_k = smoothstep(-0.005, 0.04, hw.y / hw.w - world.y);
                let w = sub_k * smoothstep(0.0, 0.15, edge) * (1.0 - smoothstep(6.0, 14.0, t))
                    * (1.0 - smoothstep(0.15 + t * 0.04, 0.35 + t * 0.08, gap));
                return vec4<f32>(col, w);
            }
        }
        t_prev = t;
        t = t * 1.16 + 0.05 * (0.75 + 0.5 * fract(jit * 7.0 + f32(i) * 0.618));
    }
    return vec4<f32>(0.0);
}

// Water the Niva's tires push around: a foam collar at the contact, a bow
// ahead, a V wake and a trail behind. Picture only, from the tire poses in
// the frame; x is foam 0..1, y is how far the bow lifts the sheet, metres.
// The open sand field is not one flat swatch: tone drifts over metres,
// damp patches hold in the low spots, hummocks and wind ripples catch the
// sun, and pebbles lie scattered in the grain. Shared by the heightfield and
// the skirt past it, so the field has no seam. Picture only.
struct SandOut {
    albedo: vec3<f32>,
    n: vec3<f32>,
    rough: f32,
}

// `fw` is length(fwidth(xz)), taken by the caller in uniform control flow.
fn sand_variety(xz: vec2<f32>, albedo_in: vec3<f32>, n_in: vec3<f32>, rough_in: f32, under_water: f32, fw: f32) -> SandOut {
    var albedo = albedo_in;
    var n = n_in;
    var rough_out = rough_in;
    let big = noise(xz * 0.11 + vec2<f32>(13.0, 7.0)) * 0.6 + noise(xz * 0.37) * 0.4;
    let warm = noise(xz * 0.23 + vec2<f32>(-4.0, 9.0));
    albedo *= mix(0.93, 1.05, big) * mix(vec3<f32>(1.02, 0.99, 0.95), vec3<f32>(0.97, 0.99, 1.02), warm);
    // Graded toward the FluidNinja beach: warmer, more saturated, a little
    // more contrast, so the field is not a pale wash.
    let lum = dot(albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
    albedo = max(mix(vec3<f32>(lum), albedo, 1.1) * vec3<f32>(1.04, 0.96, 0.86), vec3<f32>(0.0));
    albedo *= 1.12;
    // Wind ripples and grain: centimetre relief in the normal, faded out
    // where a pixel covers more than the ripple so it does not shimmer.
    let micro = 1.0 - smoothstep(0.012, 0.05, fw);
    if (micro > 0.0) {
        let wdir = normalize(vec2<f32>(0.8, 0.6));
        // Small, wandering ripples: the crest line bends and breaks, and the
        // set fades out in patches.
        // Two sets from slightly different winds, the wavelength drifting
        // between 5 and 8 cm, crests that break off and fork, and an
        // asymmetric profile (gentle stoss, steep lee).
        let warp = noise(xz * 2.2) * 20.0 + noise(xz * 0.7) * 14.0 + noise(xz * 9.0) * 4.0;
        let wl_k = mix(80.0, 125.0, noise(xz * 0.6 + vec2<f32>(9.0, 3.0)));
        let wph = dot(xz, wdir) * wl_k + warp;
        let wdir2 = normalize(vec2<f32>(0.45, 0.89));
        let wph2 = dot(xz, wdir2) * 150.0 + warp * 0.6 + noise(xz * 5.0) * 4.0;
        let crest = smoothstep(0.22, 0.55, noise(xz * vec2<f32>(5.5, 7.5) + vec2<f32>(2.0, 8.0)));
        let fw1 = fract(wph / 6.2832);
        let saw = select(1.0 - (fw1 - 0.7) / 0.3, fw1 / 0.7, fw1 < 0.7) * 2.0 - 1.0;
        let set1 = mix(cos(wph), saw, 0.5) * mix(0.35, 1.0, smoothstep(0.2, 0.7, noise(xz * 1.3 + vec2<f32>(4.0, 1.0))));
        let set2 = cos(wph2) * smoothstep(0.5, 0.85, noise(xz * 0.9 + vec2<f32>(-3.0, 6.0))) * 0.6;
        let wdir3 = normalize(vec2<f32>(0.95, -0.3));
        let set3 = cos(dot(xz, wdir3) * 60.0 + warp * 0.8) * smoothstep(0.55, 0.9, noise(xz * 0.5 + vec2<f32>(11.0, -2.0))) * 0.7;
        let ripple = (set1 + set2 + set3) * mix(0.15, 1.0, crest);
        let gq = xz * 140.0;
        let g0 = hash12(floor(gq));
        let gx = hash12(floor(gq) + vec2<f32>(1.0, 0.0)) - g0;
        let gz = hash12(floor(gq) + vec2<f32>(0.0, 1.0)) - g0;
        // Under water the ripples are slumped and silted over.
        let sub = mix(1.0, 0.75, smoothstep(0.002, 0.02, under_water));
        n = normalize(n + (vec3<f32>(wdir.x, 0.0, wdir.y) * ripple * 0.5 * sub + vec3<f32>(gx, 0.0, gz) * 0.12) * micro);
        // The lee side of each ripple holds a little shade-darker grain.
        albedo *= 1.0 - 0.06 * smoothstep(0.3, 1.0, ripple) * micro;
        albedo *= 1.0 + (g0 - 0.5) * 0.14 * micro;
    }
    // Grain: millimetre relief from two octaves of smooth value noise (the
    // bevy_water noise), each faded once a pixel covers its cells.
    for (var o: i32 = 0; o < 2; o = o + 1) {
        let f = select(330.0, 1100.0, o == 1);
        let fade = 1.0 - smoothstep(0.6 / f, 2.5 / f, fw);
        if (fade > 0.0) {
            let q = xz * f + vec2<f32>(f32(o) * 31.0, 7.0);
            let e = 0.35;
            let c0 = bw_vnoise2d(q);
            let cx = bw_vnoise2d(q + vec2<f32>(e, 0.0)) - c0;
            let cz = bw_vnoise2d(q + vec2<f32>(0.0, e)) - c0;
            n = normalize(n - vec3<f32>(cx, 0.0, cz) * select(1.6, 1.0, o == 1) * fade);
            albedo *= 1.0 + (c0 - 0.5) * select(0.45, 0.3, o == 1) * fade;
        }
    }
    // Grain that survives the distance: a few millimetres, unfiltered past
    // where the fine octaves fade.
    let mfade = 1.0 - smoothstep(0.008, 0.04, fw);
    if (mfade > 0.0) {
        let mq = xz * 160.0;
        let mg = bw_vnoise2d(mq) * 0.6 + bw_vnoise2d(mq * 2.1 + vec2<f32>(9.0, 2.0)) * 0.4;
        albedo *= 1.0 + (mg - 0.5) * 0.3 * mfade;
        let mx = bw_vnoise2d(mq + vec2<f32>(0.3, 0.0)) - bw_vnoise2d(mq);
        let mz = bw_vnoise2d(mq + vec2<f32>(0.0, 0.3)) - bw_vnoise2d(mq);
        n = normalize(n - vec3<f32>(mx, 0.0, mz) * 0.5 * mfade);
    }
    // Coarse grain and shell grit in clumps, a centimetre across.
    let cfade = 1.0 - smoothstep(0.004, 0.02, fw);
    if (cfade > 0.0) {
        let cq = xz * 70.0;
        let cg = bw_vnoise2d(cq) * 0.65 + bw_vnoise2d(cq * 2.7 + vec2<f32>(3.0, 1.0)) * 0.35;
        albedo *= 1.0 + (cg - 0.5) * 0.22 * cfade;
    }
    // Coarse ripples, 25-40 cm, that still read across the pool.
    let cr_fade = 1.0 - smoothstep(0.05, 0.25, fw);
    if (cr_fade > 0.0) {
        let cdir = normalize(vec2<f32>(0.75, 0.66));
        let cwarp = noise(xz * 0.9) * 6.0 + noise(xz * 2.7) * 2.0;
        let cph = dot(xz, cdir) * mix(16.0, 25.0, noise(xz * 0.3 + vec2<f32>(2.0, 5.0))) + cwarp;
        let cpatch = smoothstep(0.3, 0.7, noise(xz * 0.45 + vec2<f32>(-7.0, 3.0)));
        n = normalize(n + vec3<f32>(cdir.x, 0.0, cdir.y) * cos(cph) * 0.18 * mix(0.4, 1.0, cpatch) * cr_fade);
        albedo *= 1.0 - 0.02 * smoothstep(0.4, 1.0, sin(cph)) * cpatch * cr_fade;
    }
    // Speckle that holds at distance: dark and light grains in a fine net.
    let sp_fade = 1.0 - smoothstep(0.02, 0.15, fw);
    let sp = hash12(floor(xz * 45.0)) * 0.6 + hash12(floor(xz * 23.0) + vec2<f32>(7.0, 1.0)) * 0.4;
    albedo *= 1.0 + (sp - 0.5) * 0.36 * sp_fade;
    let gcell = floor(xz * 45.0);
    let gn = vec2<f32>(hash12(gcell + vec2<f32>(3.0, 9.0)), hash12(gcell + vec2<f32>(11.0, 2.0))) - vec2<f32>(0.5);
    n = normalize(n + vec3<f32>(gn.x, 0.0, gn.y) * 0.35 * sp_fade);
    // Mottling and pocks that hold at distance: the far field is not a flat card.
    let pq = xz * 9.0;
    let pm = noise(pq) * 0.5 + noise(pq * 2.3 + vec2<f32>(4.0, 7.0)) * 0.3 + noise(pq * 5.1) * 0.2;
    albedo *= 1.0 + (pm - 0.5) * 0.1;
    let px = noise((xz + vec2<f32>(0.02, 0.0)) * 9.0) - noise(pq);
    let pz = noise((xz + vec2<f32>(0.0, 0.02)) * 9.0) - noise(pq);
    n = normalize(n - vec3<f32>(px, 0.0, pz) * 0.7);
    // Hummocks: half-metre swells in the normal only.
    let hq = xz * 1.7;
    let h0 = noise(hq) + 0.5 * noise(hq * 2.3 + vec2<f32>(5.0, 1.0));
    let hx = noise(hq + vec2<f32>(0.05, 0.0)) + 0.5 * noise((hq + vec2<f32>(0.05, 0.0)) * 2.3 + vec2<f32>(5.0, 1.0));
    let hz = noise(hq + vec2<f32>(0.0, 0.05)) + 0.5 * noise((hq + vec2<f32>(0.0, 0.05)) * 2.3 + vec2<f32>(5.0, 1.0));
    n = normalize(n + vec3<f32>(h0 - hx, 0.0, h0 - hz) * 1.0);
    // Damp patches in the hollows: darker, a little smoother.
    let damp_p = smoothstep(0.58, 0.72, noise(xz * 0.19 + vec2<f32>(2.0, -6.0)) * 0.7 + (1.0 - h0 / 1.5) * 0.3);
    albedo = mix(albedo, albedo * vec3<f32>(0.8, 0.76, 0.7), damp_p * 0.2 * (1.0 - smoothstep(0.0, 0.02, under_water)));
    rough_out = mix(rough_out, rough_out * 0.75, damp_p);
    return SandOut(albedo, n, rough_out);
}

// Slope of the rings round each wheel that stands in water, as a normal
// offset. Capillary rings: about 9 cm apart, running out at ~0.35 m/s and
// fading over a metre.
fn wheel_ripples(xz: vec2<f32>, surf: f32, t: f32) -> vec3<f32> {
    var g = vec2<f32>(0.0);
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        if (c.y - c.w > surf + 0.04) {
            continue;
        }
        let d = xz - c.xz;
        let r = length(d);
        if (r < 0.12 || r > 2.5) {
            continue;
        }
        // No rings: the water the tire stirs is a patch of short chop that
        // dies out within half a metre, plus the slope of the low swell
        // pushed up against the rubber.
        let amp = exp(-r / 0.3) * smoothstep(0.1, 0.22, r);
        let q = xz * 17.0 + vec2<f32>(f32(i) * 5.0, 0.0);
        let chop = vec2<f32>(noise(q + vec2<f32>(t * 1.3, 0.0)) - noise(q * 1.9 - vec2<f32>(0.0, t * 1.7)),
            noise(q + vec2<f32>(7.0, t * 1.1)) - noise(q * 2.3 + vec2<f32>(t * 1.5, 3.0)));
        let swell = d / r * exp(-pow((r - 0.25) / 0.1, 2.0)) * 0.6;
        // A few capillary rings close round the rubber, 5 cm apart, gone by
        // ~45 cm, broken a little by the chop.
        let ring = d / r * cos((r - 0.2) * 125.0 - t * 4.0 + noise(q * 0.5) * 1.5)
            * exp(-max(r - 0.22, 0.0) / 0.12) * smoothstep(0.17, 0.24, r) * 1.4;
        g += (chop * 1.6 + swell + ring) * amp + ring * 0.6;
    }
    return vec3<f32>(g.x, 0.0, g.y) * 0.25;
}

// x = waterline at this point of a tire standing in the solved sheet
// (the film level round the carcass plus the collar the water pass draws),
// y = 1 when on a wet tire, z = carcass bottom.
fn tire_waterline(p: vec3<f32>) -> vec4<f32> {
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let wl = frame.tire_water[i];
        if (wl > -100.0 && length(p.xz - c.xz) < c.w * 1.15 && p.y < c.y + c.w && wl > c.y - c.w + 0.004) {
            // The water mesh is coarse and only rises a third of the bow at
            // the rubber; follow the sheet that is drawn, or the silt tint
            // shows above the water as mud plastered on the tire.
            return vec4<f32>(wl, 1.0, c.y - c.w, 0.0);
        }
    }
    return vec4<f32>(0.0);
}

fn tire_wake(xz: vec2<f32>, surf: f32) -> vec2<f32> {
    var foam = 0.0;
    var bulge = 0.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let f = frame.tires[i * 2 + 1];
        if (c.y - c.w > surf + 0.04) {
            continue;
        }
        let fwd = normalize(vec2<f32>(f.x, f.y) + vec2<f32>(1e-5, 0.0));
        let side = vec2<f32>(-fwd.y, fwd.x);
        let d = xz - c.xz;
        let a = dot(d, fwd);
        let b = abs(dot(d, side));
        let half_w = 0.11;
        let out_b = max(b - half_w, 0.0);
        let ring = length(vec2<f32>(max(abs(a) - 0.2, 0.0), out_b));
        let collar = exp(-ring * ring / 0.012);
        let bow = exp(-pow(max(a - 0.2, 0.0) / 0.25, 2.0)) * step(0.0, a) * exp(-out_b / 0.14);
        let back = max(-a - 0.1, 0.0);
        let arm = abs(out_b - back * 0.36);
        let wake = exp(-arm * arm / (0.004 + back * 0.02)) * exp(-back / 2.4) * step(a, 0.0);
        let trail = exp(-out_b * out_b / 0.01) * exp(-back / 1.6) * step(a, 0.0);
        foam = max(foam, max(collar, max(wake, trail * 0.7)));
        bulge = max(bulge, bow * 0.025 + collar * 0.012);
    }
    return vec2<f32>(foam, bulge);
}

// The beauty sand's water sheet (contract film); water_vertex calls it.
fn beauty_water_vertex(in: TerrIn) -> WaterOut {
    let local = height_bilinear(in.xz);
    let dug = max(-stamp_bilinear(in.xz), 0.0);
    let depth = covered_depth(in.xz);
    let column = water_column(in.xz);
    let free = frame.misc.w > 0.5 && column > 0.04;
    let rip = ripple_height(in.xz, frame.misc.x) * smoothstep(0.0, 0.03, depth);
    // Film on the bed, but never below the soil. A bump left uncovered
    // reads as a black hole. bed + full depth was a sheet 30 cm up that
    // cut the tire like a clip plane. The pond stays that blurred surface.
    // A cut holds its own sheet down in the groove.
    // A crossing sets misc.w and the sheet is the solved column, so the
    // bow is the crest of that column.
    let cover = local + 0.01;
    let pond = max(water_bed(in.xz) + 0.016 + rip, cover) + smoothstep(0.02, 0.16, depth) * 0.012;
    let fill = smoothstep(0.04, 0.12, depth) * min(dug * 0.5, 0.032);
    let in_rut = local + 0.01 + rip + fill;
    let y_sheet = select(pond, in_rut, dug > 0.018 && depth > 0.004);
    // The column is the solved water. Drawing it at full height is a mound.
    // The sheet sits on the ground. The bow is a crest in that sheet, plus a swell.
    let swell = select(0.0, free_swell(in.xz, frame.misc.x), free);
    let surface = water_bed(in.xz) + shallow_water(column) + rip * 0.45 + swell;
    var y = select(y_sheet, surface, free);
    var d_out = select(depth, column, free);
    // Inside the contract window the sheet is the published film: water plus
    // slurry over the ground, at its true thickness.
    let inside = bc_inside(in.xz);
    if (inside > 0.0) {
        let film = bc_film_soft(in.xz);
        let thick = film.r + film.g;
        let lift = 0.0;
        let ground_c = bc_ground_soft(in.xz, local);
        // Level water stays level: ground and film through one filter, and
        // never above the ground plus the unblurred film, so the blurred film
        // does not climb the rut banks as a rounded tube.
        let raw_f = bc_film(in.xz);
        let y_lvl = min(bc_surface_soft(in.xz), ground_c + raw_f.r + raw_f.g);
        let y_c = y_lvl + rip * smoothstep(0.01, 0.04, thick) * 0.6 + lift;
        y = mix(y, y_c, inside);
        d_out = mix(d_out, thick, inside);
        // Round a tire standing in the sheet, the drawn surface settles to the
        // solved level at the rubber (the ring mean the tire pass uses), so it
        // cuts the tire on one clean line: a bow bulge on the coarse mesh cut
        // the tread higher than the sidewall, a step.
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let c = frame.tires[i * 2];
            let wl = frame.tire_water[i];
            if (wl > -100.0 && wl > c.y - c.w + 0.004) {
                // Only where the film already stands: lifting the dry shore
                // to the tire's level drew a round sheet at the wheel.
                let k = (1.0 - smoothstep(0.3, 0.55, length(in.xz - c.xz))) * inside * smoothstep(0.004, 0.03, thick);
                y = mix(y, wl + rip * 0.2, k);
            }
        }
    }
    let world = vec3<f32>(in.xz.x, y, in.xz.y);
    var o: WaterOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    o.depth = d_out;
    return o;
}

struct SsrOut {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
}

// View depth (clip w) of a world point: the key the bilateral upscale and
// the history test compare.
fn ssr_view_depth(p: vec3<f32>) -> f32 {
    return (frame.view_proj * vec4<f32>(p, 1.0)).w;
}

// The large-scale surface normal for the half-res trace: the published
// surface slope, the long swell and the wind ripples. The capillary detail
// is below a half-res texel and is left to the roughness blur.
fn ssr_surface_normal(xz: vec2<f32>, column_c: f32, depth: f32) -> vec3<f32> {
    let n = ripple_normal(xz, frame.misc.x);
    let dc = 0.2;
    let s0 = bc_surface_soft(xz);
    let sx = bc_surface_soft(xz + vec2<f32>(dc, 0.0));
    let sz = bc_surface_soft(xz + vec2<f32>(0.0, dc));
    let c_n = normalize(vec3<f32>(-(sx - s0) / dc, 1.0, -(sz - s0) / dc));
    var nrm = normalize(c_n + vec3<f32>(n.x * 0.8, 0.0, n.z * 0.8));
    let t_now = frame.misc.x;
    let wv = frame.air[0].xz;
    let wd0 = select(vec2<f32>(0.8, 0.6), normalize(wv), dot(wv, wv) > 1e-4);
    let swing = (noise(xz * 0.8 + vec2<f32>(3.0, 1.0)) - 0.5) * 1.8 + (noise(xz * 2.6) - 0.5) * 0.6;
    let wd = vec2<f32>(wd0.x * cos(swing) - wd0.y * sin(swing), wd0.x * sin(swing) + wd0.y * cos(swing));
    let wp = vec2<f32>(-wd.y, wd.x);
    var wg = vec2<f32>(0.0);
    for (var o: i32 = 0; o < 2; o = o + 1) {
        let fo = f32(o);
        let f_along = 16.0 * pow(2.5, fo);
        let f_across = 9.0 * pow(2.6, fo);
        let drift = t_now * (0.9 + 0.7 * fo);
        let q = vec2<f32>(dot(xz, wd) * f_along - drift * f_along * 0.1, dot(xz, wp) * f_across) + vec2<f32>(fo * 17.0, 3.0);
        let h0 = bw_vnoise2d(q);
        let ha = bw_vnoise2d(q + vec2<f32>(0.25, 0.0)) - h0;
        let hc = bw_vnoise2d(q + vec2<f32>(0.0, 0.25)) - h0;
        let amp = select(0.035, 0.018, o == 1);
        wg += (wd * ha * f_along + wp * hc * f_across) * 4.0 * amp / f_along;
    }
    let gust = smoothstep(0.42, 0.72, noise(xz * 0.45 + wd * t_now * 0.4) * 0.7 + noise(xz * 1.7 - wd * t_now * 0.3) * 0.3);
    return normalize(nrm - vec3<f32>(wg.x, 0.0, wg.y) * mix(0.45, 1.6, gust) * smoothstep(0.003, 0.015, depth));
}

// Half-resolution SSR of the water sheet. Four jittered rays per texel,
// the jitter rotated every frame, blended with last frame's result
// reprojected through the previous view-projection. The water pass reads
// it back with a depth- and normal-aware upscale.
@fragment
fn fs_water_ssr(in: WaterOut) -> SsrOut {
    var o: SsrOut;
    let xz = in.world.xz;
    let inside = bc_inside(xz);
    if (inside < 0.5) {
        discard;
    }
    let film_c = bc_film(xz);
    let column_c = film_c.r + film_c.g;
    if (column_c < 0.003) {
        discard;
    }
    let nrm = ssr_surface_normal(xz, column_c, column_c);
    let view = normalize(frame.cam_pos.xyz - in.world);
    let refl_dir = reflect(-view, nrm);
    let vfl = normalize(vec2<f32>(view.x, view.z) + vec2<f32>(1e-5, 0.0));
    let seed = fract(hash12(floor(in.clip.xy) + vec2<f32>(17.0, 3.0)) + fract(frame.misc.x * 7.31) * 0.618);
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var k: i32 = 0; k < 4; k = k + 1) {
        let along = ((f32(k) + seed) / 4.0 - 0.5) * 0.22;
        let side = (fract(seed * 13.7 + f32(k) * 0.37) - 0.5) * 0.06;
        let jd = vec3<f32>(vfl.x * along - vfl.y * side, 0.0, vfl.y * along + vfl.x * side);
        let sk = water_ssr(in.world + nrm * 0.005, normalize(refl_dir + jd));
        sum += sk.rgb * sk.a;
        wsum += sk.a;
    }
    var a = vec4<f32>(sum, wsum) * 0.25;
    let vd = ssr_view_depth(in.world);
    // Temporal accumulation: the same world point in last frame's half-res
    // target, accepted when its stored view depth matches.
    if (frame.ssr_info.x > 0.5) {
        let pc = frame.prev_view_proj * vec4<f32>(in.world, 1.0);
        if (pc.w > 0.01) {
            let puv = vec2<f32>(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
            if (all(puv >= vec2<f32>(0.0)) && all(puv < vec2<f32>(1.0))) {
                let ip = vec2<i32>(puv * frame.ssr_info.yz);
                let hb = textureLoad(ssr_b_tex, ip, 0);
                if (hb.w > 0.5 && abs(hb.x - pc.w) < 0.03 * pc.w) {
                    let ha = textureLoad(ssr_a_tex, ip, 0);
                    a = mix(ha, a, 0.2);
                }
            }
        }
    }
    // A NaN or overflow in one texel would spread through the history and
    // the upscale; drop it.
    let ok = all(a == a) && all(abs(a) < vec4<f32>(1e3)) && vd == vd;
    o.a = select(vec4<f32>(0.0), clamp(a, vec4<f32>(0.0), vec4<f32>(64.0)), ok);
    o.b = select(vec4<f32>(0.0), vec4<f32>(vd, nrm.x, nrm.z, 1.0), ok);
    return o;
}

// Bilateral upscale of the half-res SSR at a full-res water pixel: a 4x4
// window of half-res texels, Gaussian in screen space with a radius that
// grows with the roughness, weighted by view depth and normal so the
// reflection does not bleed across the waterline or a tire.
fn ssr_upsample(clip_xy: vec2<f32>, vd: f32, nrm: vec3<f32>, rough: f32) -> vec4<f32> {
    let hc = clip_xy * 0.5 - vec2<f32>(0.5);
    let sigma = mix(1.4, 2.6, smoothstep(0.06, 0.2, rough));
    let r = i32(ceil(sigma * 1.5));
    let base = vec2<i32>(floor(hc));
    let dmax = vec2<i32>(textureDimensions(ssr_a_tex)) - vec2<i32>(1);
    var acc = vec4<f32>(0.0);
    var wt = 0.0;
    for (var j: i32 = -r; j <= r + 1; j = j + 1) {
        for (var i: i32 = -r; i <= r + 1; i = i + 1) {
            let ip = clamp(base + vec2<i32>(i, j), vec2<i32>(0), dmax);
            let b = textureLoad(ssr_b_tex, ip, 0);
            if (b.w < 0.5) {
                continue;
            }
            let dpx = vec2<f32>(ip) - hc;
            let ws = exp(-dot(dpx, dpx) / (2.0 * sigma * sigma));
            let wz = exp(-abs(b.x - vd) / (0.015 * vd + 0.01));
            let nb = vec3<f32>(b.y, sqrt(max(1.0 - b.y * b.y - b.z * b.z, 0.0)), b.z);
            let wn = pow(max(dot(nb, nrm), 0.0), 16.0);
            let w = ws * wz * wn;
            let av = textureLoad(ssr_a_tex, ip, 0);
            if (!(all(av == av) && w == w)) {
                continue;
            }
            acc += av * w;
            wt += w;
        }
    }
    if (wt < 1e-5) {
        return vec4<f32>(0.0);
    }
    let a = acc / wt;
    if (a.a < 1e-3) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(clamp(a.rgb / a.a, vec3<f32>(0.0), vec3<f32>(4.0)), clamp(a.a, 0.0, 1.0));
}

// The beauty sand's water (contract film); fs_water calls it.
fn beauty_water_fragment(in: WaterOut) -> vec4<f32> {
    let xz = in.world.xz;
    let kind = kind_nearest(xz);
    let bed_layer = layer_of(kind);
    // Only the bed's average tint goes into the water colour; a full-res
    // lookup smeared into streaks along the grazing view. The grain itself
    // comes through from the soil pass below.
    let bed = textureSampleLevel(ground_diff, ground_samp, layer_uv(bed_layer, xz), bed_layer, 7.0).rgb;
    let own = mat_bilinear(xz).g * 0.34;
    let dug = max(-stamp_bilinear(xz), 0.0);
    // A cut keeps film and slurry on the soil mesh. Dilated water on the lip
    // was drawing flat cards in the pockets beside the floor.
    let e = vec2<f32>(0.45, 0.0);
    let ez = vec2<f32>(0.0, 0.45);
    let near_cut = dug > 0.008
        || -stamp_bilinear(xz + e) > 0.02
        || -stamp_bilinear(xz - e) > 0.02
        || -stamp_bilinear(xz + ez) > 0.02
        || -stamp_bilinear(xz - ez) > 0.02;
    let free = frame.misc.w > 0.5 && in.depth > 0.04;
    let inside = bc_inside(xz);
    // The mask is the published depth itself (bilinear), not the blurred
    // copy: the blur spread the sheet over the rut banks.
    let film_c = bc_film(xz);
    var column_c = film_c.r + film_c.g;
    // Under and right round a tire the film map has a hole where the rubber
    // displaces the water; the sheet there stands at the solved level.
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let wl = frame.tire_water[i];
        // Only where the published film already stands: forcing the solved
        // level over a 0.5 m circle drew a dark disc on the shore.
        let k_t = (1.0 - smoothstep(0.3, 0.5, length(xz - c.xz))) * smoothstep(0.003, 0.006, column_c);
        if (inside > 0.5 && wl > -100.0 && wl > c.y - c.w + 0.004) {
            column_c = max(column_c, in.depth * k_t);
        }
    }
    // The waterline is where the published depth runs out; the cut-off
    // wanders a few millimetres with the grain so the edge is torn, not ruled.
    // The contract's wet flag: water + slurry >= 4 mm (RENDER_CONTRACT,
    // binding 8). The sheet ends exactly there; nothing is drawn on top.
    let edge_t = 0.003;
    if (inside > 0.5) {
        if (column_c < edge_t) {
            discard;
        }
        // Slurry left as a thin skin in a rut is the rut's wet floor, drawn
        // on the soil; a flat sheet over it reads as a plank.
        if (film_c.g > film_c.r && column_c < 0.03) {
            discard;
        }
    } else if (!free && (near_cut || own < 0.008 || in.depth < 0.008)) {
        discard;
    }
    let n = ripple_normal(xz, frame.misc.x);
    let flat = normalize(vec3<f32>(n.x * 0.8, 1.0, n.z * 0.8));
    // Slope of the low sheet, not of the full column. A full-column normal
    // is a wall.
    let de = 0.28;
    let y0 = shallow_water(water_column(xz));
    let yx = shallow_water(water_column(xz + vec2<f32>(de, 0.0)));
    let yz = shallow_water(water_column(xz + vec2<f32>(0.0, de)));
    let g = free_swell_grad(xz, frame.misc.x);
    // The crest is only a few centimetres. Stretch its normal so the face
    // of the wave reads, without standing the water up into a mound.
    let dx = (yx - y0) / de;
    let dz = (yz - y0) / de;
    let slope = normalize(vec3<f32>(-(dx * 2.6 + g.x * 2.4), 1.0, -(dz * 2.6 + g.y * 2.4)));
    var nrm = select(flat, slope, free);
    // Inside the contract window the column is the published film: water
    // and slurry thickness, metres. Outside it, the picture cache. The
    // surface is ground plus film, so a wave in the film tilts the normal.
    if (inside > 0.0) {
        // Ground and film through one filter: two different kernels bent
        // the level water in a narrow rut into a tube.
        let dc = 0.2;
        let s0 = bc_surface_soft(xz);
        let sx = bc_surface_soft(xz + vec2<f32>(dc, 0.0));
        let sz = bc_surface_soft(xz + vec2<f32>(0.0, dc));
        let c_n = normalize(vec3<f32>(-(sx - s0) / dc, 1.0, -(sz - s0) / dc));
        let c_n2 = normalize(c_n + vec3<f32>(n.x * 0.8, 0.0, n.z * 0.8));
        // select, not mix: the picture-cache slope can be NaN on the
        // through-drive field, and mix carries a NaN through a zero weight.
        nrm = select(normalize(mix(nrm, c_n2, inside)), c_n2, inside > 0.99);
    }
    let depth = mix(in.depth, column_c, inside);
    // Murk from the published slurry fraction and, with contract v3, the
    // suspended solids the wheels lift (binding 4 A).
    let slurry_frac = max(select(0.0, film_c.g / max(column_c, 1e-4), column_c > 1e-4) * inside, bc_turbidity(xz));
    // Published flow (binding 4). A still pool publishes ~0; until contract
    // v3 the drift direction is taken down the gradient of the published
    // water depth (deep to shallow), a few cm/s, only for the ripple drift.
    var flow = bc_flow(xz);
    let fg_e = 0.2;
    let dgx = (bc_film_soft(xz + vec2<f32>(fg_e, 0.0)).r - bc_film_soft(xz - vec2<f32>(fg_e, 0.0)).r) / (2.0 * fg_e);
    let dgz = (bc_film_soft(xz + vec2<f32>(0.0, fg_e)).r - bc_film_soft(xz - vec2<f32>(0.0, fg_e)).r) / (2.0 * fg_e);
    let dgrad = vec2<f32>(dgx, dgz);
    let drift_dir = select(vec2<f32>(0.0), -normalize(dgrad), dot(dgrad, dgrad) > 1e-8);
    let ripple_flow = select(drift_dir * 0.04, flow, dot(flow, flow) > 1e-6);
    let speed = length(flow);
    let t_now = frame.misc.x;
    // Ripples from bevy_water's wave pattern, the same field that throws
    // the caustics on the bed. Still water just breathes.
    let calm = smoothstep(0.015, 0.06, depth);
    let bw_n = bw_wave_normal(xz - flow * t_now * 0.5, t_now);
    // The bevy_water swell read as parallel bands from this low view:
    // keep it weak and only in patches.
    let bw_patch = smoothstep(0.35, 0.75, noise(xz * 0.7 + vec2<f32>(-5.0, 2.0)));
    nrm = normalize(nrm + vec3<f32>(bw_n.x, 0.0, bw_n.z) * mix(0.04, 0.14, calm) * bw_patch);
    // Cat's-paw wind ripples: two octaves of value noise stretched across
    // the wind, so the crests run side-on to it and break the sky up.
    let wv = frame.air[0].xz;
    let wd0 = select(vec2<f32>(0.8, 0.6), normalize(wv), dot(wv, wv) > 1e-4);
    // Gusts swing the ripple direction by up to ~50 degrees patch to patch.
    let swing = (noise(xz * 0.8 + vec2<f32>(3.0, 1.0)) - 0.5) * 1.8 + (noise(xz * 2.6) - 0.5) * 0.6;
    let wd = vec2<f32>(wd0.x * cos(swing) - wd0.y * sin(swing), wd0.x * sin(swing) + wd0.y * cos(swing));
    let wp = vec2<f32>(-wd.y, wd.x);
    var wg = vec2<f32>(0.0);
    let lens_d = length(in.world - frame.cam_pos.xyz);
    // Calmer far off: the stretched crests drew the sky mirror as long streaks.
    let near_k = mix(1.0, 0.45, smoothstep(4.0, 9.0, lens_d));
    for (var o: i32 = 0; o < 3; o = o + 1) {
        let fo = f32(o);
        let f_along = 16.0 * pow(2.5, fo);
        let f_across = 9.0 * pow(2.6, fo);
        let drift = t_now * (0.9 + 0.7 * fo);
        let q = vec2<f32>(dot(xz, wd) * f_along - drift * f_along * 0.1, dot(xz, wp) * f_across) + vec2<f32>(f32(o) * 17.0, 3.0);
        let h0 = bw_vnoise2d(q);
        let ha = bw_vnoise2d(q + vec2<f32>(0.25, 0.0)) - h0;
        let hc = bw_vnoise2d(q + vec2<f32>(0.0, 0.25)) - h0;
        // Back to world slope: d/d(along) and d/d(across).
        let amp = select(select(0.035, 0.018, o == 1), 0.009, o == 2) * near_k
            * (1.0 - smoothstep(0.1, 0.45, length(fwidth(xz)) * f_along));
        wg += (wd * ha * f_along + wp * hc * f_across) * 4.0 * amp / f_along;
    }
    // Cat's paws: wind touches the water in patches; between them it is glassy.
    let gust = smoothstep(0.42, 0.72, noise(xz * 0.45 + wd * t_now * 0.4) * 0.7 + noise(xz * 1.7 - wd * t_now * 0.3) * 0.3);
    nrm = normalize(nrm - vec3<f32>(wg.x, 0.0, wg.y) * mix(0.45, 1.6, gust) * smoothstep(0.003, 0.015, depth));
    // Fine capillary ripples carried by the flow: short crests across the
    // current, so the sheet near the lens is a rippled window, not a glaze.
    let fdir = select(vec2<f32>(0.8, 0.6), normalize(ripple_flow), dot(ripple_flow, ripple_flow) > 1e-8);
    let fper = vec2<f32>(-fdir.y, fdir.x);
    let fq = vec2<f32>(dot(xz, fdir) * 26.0 - t_now * 3.0, dot(xz, fper) * 11.0) + ripple_flow * t_now * 20.0;
    let f0 = bw_vnoise2d(fq);
    let fa = bw_vnoise2d(fq + vec2<f32>(0.25, 0.0)) - f0;
    let fc2 = bw_vnoise2d(fq + vec2<f32>(0.0, 0.25)) - f0;
    let fslope = (fdir * fa * 26.0 + fper * fc2 * 11.0) * 4.0 * 0.004;
    let w_fw = length(fwidth(xz));
    nrm = normalize(nrm - vec3<f32>(fslope.x, 0.0, fslope.y) * smoothstep(0.002, 0.01, depth) * (1.0 - smoothstep(0.004, 0.012, w_fw)));
    // Rings run out from each wheel standing in the water, and the bow
    // pushes a swell ahead of it.
    // Faded where a pixel spans the chop: at the grazing view it aliased into
    // bright speckles round the tires.
    nrm = normalize(nrm + wheel_ripples(xz, in.world.y, t_now) * calm_or_film(depth) * (1.0 - smoothstep(0.003, 0.01, w_fw)));
    // Rings where the published drops land (Ultra-Water's impact rings on
    // the contract spray buffer), band-limited to the pixel.
    let drop_g = bc_drop_rings(xz, in.world.y, 2.0 / max(w_fw, 1e-3));
    nrm = normalize(nrm - vec3<f32>(drop_g.x, 0.0, drop_g.y) * smoothstep(0.003, 0.006, depth));
    // Running water combs into streaks along the current.
    if (speed > 0.05) {
        let along = flow / speed;
        let u = dot(xz, along);
        let v = dot(xz, vec2<f32>(-along.y, along.x));
        let streak = noise(vec2<f32>(u * 1.6 - t_now * speed * 1.6, v * 14.0));
        let sv = noise(vec2<f32>(u * 1.6 - t_now * speed * 1.6, v * 14.0 + 0.4)) - streak;
        let k = smoothstep(0.05, 0.8, speed) * 0.5 * calm;
        nrm = normalize(nrm + vec3<f32>(-along.y, 0.0, along.x) * sv * k);
    }

    let view = normalize(frame.cam_pos.xyz - in.world);
    let ndv = max(dot(nrm, view), 0.02);
    let sun = normalize(frame.sun_dir.xyz);
    // Light enters through the rippled surface: the shadow it carries into
    // the silt wobbles with the ripples instead of a flat stamp.
    let sun_in = refract(-normalize(frame.sun_dir.xyz), nrm, 1.0 / 1.333);
    let sun_flat = refract(-normalize(frame.sun_dir.xyz), vec3<f32>(0.0, 1.0, 0.0), 1.0 / 1.333);
    let wobble = (sun_in.xz - sun_flat.xz) * 0.35;
    g_shadow_soft = vec2<f32>(0.07, 0.0);
    let shadow = shadow_soft(in.world + vec3<f32>(wobble.x, 0.01, wobble.y), nrm);
    g_shadow_soft = vec2<f32>(0.0);

    // Muddy water is a suspension: Beer-Lambert through the column, red
    // passes furthest. The bed below is the real soil pass, seen through alpha.
    // Beer-Lambert along the view ray through the water, to whatever opaque
    // surface lies behind it (bevy_water's depth_diff_view): the bed, or a
    // tire standing in the pool. A submerged tread fades with how much
    // water covers it, so the tire meets the water at a line instead of
    // floating over clear glass.
    let pc = frame.view_proj * vec4<f32>(in.world, 1.0);
    let puv = clamp(vec2<f32>(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5), vec2<f32>(0.0), vec2<f32>(0.9999));
    let pdims = vec2<f32>(textureDimensions(post_depth));
    let d_behind = textureLoad(post_depth, vec2<i32>(puv * pdims), 0).x;
    let bw = frame.inv_view_proj * vec4<f32>(pc.x / pc.w, pc.y / pc.w, d_behind, 1.0);
    let behind = bw.xyz / bw.w;
    let view_path = max(length(behind - frame.cam_pos.xyz) - length(in.world - frame.cam_pos.xyz), 0.0);
    let bed_path = depth / max(ndv, 0.1);
    // Murk by the water standing over what lies behind (bed or submerged
    // rubber), from the published depth, with the slant capped: the full
    // grazing path from a lens 20 cm up turned a 5 cm sheet into a brown
    // board. Rubber and bed share it, so a tire fades into the murk with
    // depth instead of ending on a hard edge.
    let cover_v = select(depth, clamp(in.world.y - behind.y, 0.0, depth + 0.03), d_behind < 0.9999);
    let path = cover_v / max(ndv, 0.4);
    // Where the surface cuts a tire: a meniscus and a collar of dirty foam.
    var near_tire = 0.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        near_tire = max(near_tire, 1.0 - smoothstep(0.35, 0.6, length(xz - frame.tires[i * 2].xz)));
    }
    let contact = (1.0 - smoothstep(0.004, 0.03, view_path)) * near_tire * step(d_behind, 0.9999);
    // Rubber seen through the surface: the stirred silt hides it after a
    // couple of centimetres, so the tire ends at a sharp line.
    let tw_b = tire_waterline(behind);
    // Only rubber counts: the bed under the wheel lies inside the same
    // cylinder test and would turn the water to glass there.
    let over_tire = tw_b.y * step(d_behind, 0.9999) * smoothstep(0.008, 0.02, behind.y - height_bilinear(behind.xz));
    // Slurry is far more opaque than muddy water.
    // Puddle water is clear enough to show its bed through 10 cm; only the
    // published slurry fraction turns it to opaque mud.
    // Clear water eats red first, so the deep middle turns teal and dark
    // while the shallow rim stays the colour of the sand under it.
    // The thin rim is a clear film over the sand; the silt load builds
    // with depth, so by 5-10 cm the column is murky.
    var collar_t = 0.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        collar_t = max(collar_t, 0.0 * length(xz - frame.tires[i * 2].xz));
    }
    // Round a tire the stirred silt hides the rubber within a few cm.
    // Clear at a few centimetres (the sand reads through 5 cm), murky by
    // 15-20 cm: the silt load grows with the published depth.
    // Clear water absorbs (red first) and scatters little: the shallows are
    // a window onto the sand. Only the published slurry clouds it.
    let sigma = mix(vec3<f32>(2.6, 2.2, 3.2) * mix(0.03, 1.3, smoothstep(0.03, 0.18, depth)),
        vec3<f32>(90.0, 110.0, 140.0), slurry_frac);
    let trans = exp(-sigma * path);
    // Light scattered by the silt in the column, lit by sun and sky.
    // Deep columns go darker: light that scatters back out has crossed more silt.
    let deep_t = smoothstep(0.02, 0.12, depth);
    // Silt the tires stir up round their contact patches.
    var stir = 0.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let dd = xz - c.xz;
        // Murk only where the contract publishes slurry: no circle per tire.
        stir = max(stir, 0.0 * dot(dd, dd));
    }
    var silt = vec3<f32>(0.17, 0.10, 0.042) * mix(1.0, 0.45, deep_t);
    // What clear water scatters back is blue-green and dim; a little silt
    // warms it. Slurry is the brown suspension.
    // Brown suspension: silt scatters a warm, fairly bright light back out,
    // which is what keeps a turbid pool from reading as varnish.
    // Silt is a bright scatterer: a turbid pool returns a good share of the
    // light as a yellow-brown glow, which is what reads as mud, not glass.
    let clear_sc = mix(vec3<f32>(0.21, 0.18, 0.12), vec3<f32>(0.12, 0.1, 0.06), smoothstep(0.02, 0.15, depth));
    // Shallow water shows the sand's own warmth; deep water the teal.
    silt = mix(clear_sc, silt, 0.2 * slurry_frac);
    silt = mix(silt, vec3<f32>(0.13, 0.075, 0.035), max(slurry_frac, max(stir * 0.35, collar_t * 0.35)));
    // The middle of the pool is deeper and murkier than the clear rim.
    // Deeper reads murkier and greyer, not black: a dark centre was taken
    // for the car's shadow.
    silt = mix(silt, vec3<f32>(0.22, 0.13, 0.05), deep_t * 0.7 * slurry_frac) * mix(1.0, 0.6, smoothstep(0.04, 0.2, depth));
    let ndl = clamp(dot(vec3<f32>(0.0, 1.0, 0.0), sun), 0.0, 1.0);
    // Silt in the shade still glows with sky light scattered sideways in the
    // column: the car's shadow on turbid water is softer than on dry sand.
    let light = frame.sun_color.rgb * ndl * mix(mix(0.55, 0.3, collar_t), 1.0, shadow) * 1.25 * 0.45 + sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * frame.env.y * 1.15;
    let body = silt * light;

    // Fresnel reflection of the sky, and a tight GGX sun glint on the ripples.
    // Schlick: the sky shows at a grazing angle, the bed when looking down.
    // Rippled water is a rough mirror: its sky reflectance is the split-sum
    // F_AB of bevy_pbr at that roughness, which falls well under the smooth
    // Schlick value at a grazing look.
    let w_rough = (mix(0.14, 0.08, smoothstep(0.01, 0.08, depth)) * mix(0.8, 1.2, gust) + near_tire * 0.15)
        * mix(1.0, 0.9, smoothstep(3.0, 7.0, length(in.world - frame.cam_pos.xyz)))
        // Specular AA: roughen by how much the normal turns across the pixel.
        + min(length(fwidth(nrm)) * 1.5, 0.25);
    let w_ab = F_AB(w_rough, ndv);
    // Capped: a turbid, rippled sheet never turns into a full mirror even
    // at the far grazing end.
    // Near the lens the sheet is looked into, not along: the bed shows and
    // the sky mirror fades. The muddy collar at a tire is matte.
    let cam_d = length(in.world - frame.cam_pos.xyz);
    // Far end of the pool, seen at a grazing angle, mirrors sky and car
    // like the sea's edge; the near shallows are looked into.
    let far_k = smoothstep(2.5, 6.5, cam_d);
    // Schlick through the split-sum at the ripple roughness: from the low
    // lens the near sheet is already at 75-85 degrees, so it mirrors the sky
    // and the car (FluidNinja's light grazing shallows). Only the last
    // millimetres of film at the edge thin it out.
    let fres = min((0.02 * w_ab.x + w_ab.y) * mix(1.0, 1.6, far_k), 0.95) * (1.0 - 0.6 * collar_t)
        * mix(0.25, 1.0, smoothstep(0.003, 0.012, depth))
        // The near shallows are looked into: the sky film greyed the sand,
        // and a few cm in a rut read as a lilac tube.
        * mix(0.6 * mix(0.45, 1.0, smoothstep(0.02, 0.08, depth)), 1.0, far_k);
    // Ultra-Water's Fresnel for what the screen trace finds (the Niva):
    // Schlick with a roughness-aware F90 (Lagarde), no damping for the near
    // shallows, so the body mirrors in the sheet in front of it.
    let f90 = max(1.0 - 0.7 * sqrt(w_rough * w_rough), 0.02);
    let f_uw = (0.02 + (f90 - 0.02) * pow(1.0 - ndv, 5.0)) * (1.0 - 0.6 * collar_t);
    let refl_dir = reflect(-view, nrm);
    // The sky the water mirrors is the sky drawn behind it, at full radiance.
    // Same grade as the backdrop sky, so the mirror is not brighter than
    // the sky it mirrors.
    // Far off, the ripple slopes tip most mirror rays above the hazy
    // horizon band, so the far sheet carries the blue of the open sky.
    var refl = env_radiance(normalize(refl_dir + vec3<f32>(0.0, mix(0.12, 0.22, far_k), 0.0)), w_rough);
    let rl = dot(refl, vec3<f32>(0.2126, 0.7152, 0.0722));
    // A turbid sheet mirrors the sky a little muted and warm; the cool
    // grade of the backdrop turned it into lavender tin.
    refl = max(mix(vec3<f32>(rl), refl, mix(1.05, 1.15, far_k)) * vec3<f32>(0.97, 1.0, 1.04), vec3<f32>(0.0)) * 0.92;
    // The sun sits in the HDRI too; its blurred lobe smeared a silver sheet
    // over the backlit pool. The sun's own glint is drawn below as a sharp
    // GGX term, so clamp the env lobe to the bright sky round it.
    refl = refl * min(1.0, 0.95 / max(dot(refl, vec3<f32>(0.2126, 0.7152, 0.0722)), 1e-4));
    // A rut is a trench: a grazing reflection meets the bank before the sky.
    // March the heightfield a short way along the ray and show the soil it hits.
    // Per-pixel jittered start and a soft hit: a fixed stride and a binary
    // test cut the bank's mirror into pixel stairs along the waterline.
    var p = in.world + refl_dir * (0.04 + 0.12 * hash12(floor(in.clip.xy) + vec2<f32>(5.0, 11.0)));
    let stride = refl_dir * 0.12;
    for (var i: i32 = 0; i < 14; i = i + 1) {
        p = p + stride * (1.0 + f32(i) * 0.25);
        let rise = bc_ground_soft(p.xz, height_bilinear(p.xz)) - p.y;
        if (rise > 0.0) {
            let bank_n = terrain_normal(p.xz);
            let bank_l = frame.sun_color.rgb * clamp(dot(bank_n, sun), 0.0, 1.0) * 1.25 * shadow_factor(p, bank_n)
                + sh_irradiance(bank_n) * frame.env.y;
            // Fade with distance so a far wall does not cut a hard edge.
            let near = (1.0 - smoothstep(0.6, 2.2, length(p - in.world))) * smoothstep(0.0, 0.04, rise);
            // The bank is the same sand, a shade darker where it is wet: a fixed
            // dark-brown soil printed a brown ring round the whole shore.
            refl = mix(refl, bed * vec3<f32>(0.62, 0.6, 0.56) * bank_l, near);
            break;
        }
    }
    // Only right under the body: farther out the screen-space march finds
    // the car itself, and a box would print a slab into the mirror.
    if (frame.lamps[1].w > 0.5 && niva_box_hit(in.world, refl_dir, 0.9)) {
        // The underside and the sills in the reflection: dark, lit only by
        // the sky bounced off the ground.
        // Blend, do not replace: the underside reflection is a dim layer
        // over the water, the shadow lives on the bed beneath it.
        refl = mix(refl, sh_irradiance(vec3<f32>(0.0, -1.0, 0.0)) * frame.env.y * 0.25, 0.15);
    }
    // The tire in the mirror: dark rubber from the waterline down, broken
    // by the ripples, since the ray leaves from the rippled surface.
    let th = tire_ray_hit(in.world + nrm * 0.003, refl_dir, 0.9);
    if (th > 0.0) {
        let rubber = sh_irradiance(vec3<f32>(0.0, -0.3, 1.0)) * frame.env.y * vec3<f32>(0.05, 0.045, 0.04);
        // The tire's mirror is the SSR's job; no analytic disc here.
        refl = refl;
    }
    let halfv = normalize(sun + view);
    let ndh = clamp(dot(nrm, halfv), 0.0, 1.0);
    let ndl_w = clamp(dot(nrm, sun), 0.0, 1.0);
    // Backlit: the sun sits behind the car, so a glint near the lens burns
    // the foreground white; keep it to the far water.
    let g_a = clamp(0.035 + length(fwidth(nrm)) * 3.0, 0.035, 0.45);
    let glint = ggx_d(ndh, g_a) * 0.008 * (0.035 / g_a) * ndl_w * shadow * calm * mix(0.1, 1.0, smoothstep(2.5, 7.0, cam_d));
    let sun_spec = frame.sun_color.rgb * min(glint, 40.0);

    // Foam is the published fraction (Froude above 0.85). It is broken into
    // a lace that rides the current, not a flat white decal. A crest the
    // picture swell pushes up adds a little churn where no sheet is attached.
    let fc = bc_foam_soft(xz);
    let lace_p = xz * 9.0 - flow * t_now * 9.0;
    let lace = noise(lace_p) * 0.6 + noise(lace_p * 2.7 + vec2<f32>(3.1, 1.7)) * 0.4;
    let churn = smoothstep(0.7, 0.45, nrm.y) * smoothstep(0.06, 0.14, depth) * (1.0 - inside);
    // The published fraction is the foam. A crest of a few millimetres still
    // breaks the surface into a thin lace where the bow pushes it up.
    let crest_lace = smoothstep(0.002, 0.012, fc.g) * 0.55;
    // Fast water breaks white over the bed, and the tires churn a collar,
    // a bow and a V wake.
    let race = smoothstep(0.4, 1.6, speed) * 0.6;
    // Round a tire the churn stays mostly in the silt; only a few clumps break white.
    let wake = 0.0;
    // Swash at the shore: a thin film that runs up and drains, laced white
    // where it breaks on the sand.
    // A thin line of foam where the film runs out on the sand, a few
    // centimetres wide, broken and drifting with the swash.
    // The line sits where the film runs out against dry sand and deepens
    // into the pool on the other side; a thin film in a rut has no shore.
    let fe = 0.22;
    let fpx = bc_film(xz + vec2<f32>(fe, 0.0));
    let fnx = bc_film(xz - vec2<f32>(fe, 0.0));
    let fpz = bc_film(xz + vec2<f32>(0.0, fe));
    let fnz = bc_film(xz - vec2<f32>(0.0, fe));
    let nmin = min(min(fpx.r + fpx.g, fnx.r + fnx.g), min(fpz.r + fpz.g, fnz.r + fnz.g));
    let nmax = max(max(fpx.r + fpx.g, fnx.r + fnx.g), max(fpz.r + fpz.g, fnz.r + fnz.g));
    let wash = 0.5 + 0.5 * sin(t_now * 1.7 + noise(xz * 1.3) * 6.0);
    let shore_band = (1.0 - smoothstep(0.0006 + 0.0015 * wash, 0.003 + 0.002 * wash, nmin))
        * smoothstep(0.006, 0.02, nmax) * inside;
    let gaps = smoothstep(0.32, 0.62, noise(xz * 7.0 + vec2<f32>(t_now * 0.3, -t_now * 0.2)));
    // No white ring round the pool: the shore is the dark wet band on the
    // sand, foam only where a crest or a tire breaks the water.
    // A narrow broken line of dirty foam right at the waterline, in clumps.
    let clumps = smoothstep(0.38, 0.6, noise(xz * 6.0 + vec2<f32>(t_now * 0.1, 0.0)) * 0.6 + noise(xz * 19.0) * 0.4);
    // The waterline itself: the thin edge of the film, wherever deeper
    // water lies inward of it.
    let edge_line = smoothstep(0.003, 0.004, depth) * (1.0 - smoothstep(0.0045, 0.0065, depth))
        * smoothstep(0.003, 0.008, nmax - depth) * (1.0 - smoothstep(0.0008, 0.002, nmin)) * inside;
    let torn = smoothstep(0.3, 0.55, noise(xz * 14.0 + vec2<f32>(t_now * 0.2, 0.0)) * 0.7 + noise(xz * 41.0) * 0.3);
    let shore = max(shore_band * (1.0 - smoothstep(0.008, 0.016, depth)) * clumps, edge_line * torn);
    // Contact foam at the waterline (Ultra-Water): where the published depth
    // runs out against the bank, but only where the contract says the water
    // is working there (published foam or flow), so a still pool has none.
    let uw_contact = (1.0 - smoothstep(0.003, 0.012, depth)) * smoothstep(0.003, 0.008, nmax - depth)
        * max(fc.r, smoothstep(0.05, 0.35, speed)) * torn;
    let foam_amt = clamp(max(max(max(max(fc.r, crest_lace), max(max(churn, race), wake)), max(shore * (1.0 - near_tire) * 0.25, 0.0)), uw_contact * 0.9), 0.0, 1.0);
    // Foam is a net of bubble walls, not a blotch: cell borders of a
    // drifting Voronoi, gated by the lace.
    let bubbles = 1.0 - smoothstep(0.0, 0.12, cell_border(lace_p * 2.2));
    let lace_f = smoothstep(1.0 - foam_amt * 0.9, 1.05 - foam_amt * 0.6, lace);
    // Thin foam: a fine net of bubble walls and a few drifting flecks over
    // the published fraction, never a solid sheet (FluidNinja's lace).
    let fine = 1.0 - smoothstep(0.0, 0.07, cell_border(lace_p * 4.1 + vec2<f32>(1.3, 0.0)));
    let foam = clamp((bubbles * 0.55 + fine * 0.35) * smoothstep(0.05, 0.6, foam_amt) * mix(0.5, 1.0, lace)
        + lace_f * 0.18, 0.0, 0.8)
        * smoothstep(0.002, 0.005, depth);
    // Foam on silty water is dirty beige, never white.
    // Silt-laden scum, a shade lighter than the water, not a white sheet: the
    // published foam round the parked Niva read as pale cracked plates.
    let foam_col = mix(vec3<f32>(0.26, 0.2, 0.13), vec3<f32>(0.34, 0.28, 0.2), 1.0 - slurry_frac) * light;

    // Blend over the soil: dst * T * (1 - F) + body * (1 - T) * (1 - F) + refl * F.
    var opacity = clamp(1.0 - (trans.r + trans.g + trans.b) / 3.0, 0.0, 1.0);
    // Over the rubber: opacity from the cover above it, not the long grazing
    // path, so the round bottom of the tire shows through the murk and the
    // wheel does not read as cut flat at the waterline.
    // The stirred collar round each tire is opaque mud, not a window.
    var alpha = 1.0 - (1.0 - opacity) * (1.0 - fres);
    // The body and the tires hide the sky from the water under them too.
    let w_ao = niva_ground_ao(in.world);
    // Under the body the sheet sees no sky: dark, and opaque enough to show it.
    alpha = max(alpha, (1.0 - w_ao) * 0.4 * smoothstep(0.01, 0.05, depth));
    // The silt surface round a tire still catches the sky: keep it lighter
    // than the rubber so the waterline reads. The bed below takes the full AO.
    var lit = (body * max(opacity, 1.0 - w_ao) * (1.0 - fres) + refl * fres) * w_ao / max(alpha, 1e-3) + sun_spec;
    // Headlamps glint on the ripples and light the silt in the column.
    lit += lamp_light(in.world, nrm, view, silt * opacity, vec3<f32>(0.02), 0.07) / max(alpha, 0.3);
    lit = mix(lit, foam_col, foam * 0.45);
    alpha = max(alpha, foam * 0.5);
    // Feather the shoreline so the edge is a wet margin, not a cut line.
    // The thinnest film is a clear sheet over the sand: it keeps its
    // reflection and glint and lets the grain through.
    // The waterline is the published film: no shape is painted on it here.
    // A sharp waterline, not a fade: the film ends in a meniscus a few
    // millimetres wide that catches the sky as a thin bright line.
    // Only on the edge line itself: over a broad shallow rim the thin-film
    // band covered whole metres and lit them up as white streaks.
    let meniscus = min(edge_line * 1.6, 1.0) * smoothstep(0.003, 0.0045, depth) * (1.0 - smoothstep(0.0055, 0.009, depth));
    // The film runs out clear over the sand: no bright line at the edge.
    // A thin light lip where the film runs out: the meniscus catches the sky
    // and a little of the sun.
    // No bright contour along the waterline: it read as a drawn isoline.
    // A pixel or two wide whatever the bank slope: the depth band alone drew
    // a grey stroke round the shore from this low view.
    let dfw = max(fwidth(depth), 1e-5);
    let glint_line = (1.0 - smoothstep(0.0, 1.5, abs(depth - 0.0042) / dfw)) * torn * inside * (1.0 - near_tire)
        * smoothstep(0.003, 0.008, nmax - depth);
    lit = mix(lit, refl * 1.2 + frame.sun_color.rgb * 0.1 * shadow, glint_line * 0.35);
    alpha = max(alpha, glint_line * 0.3);
    alpha = max(alpha * smoothstep(0.003, 0.0045, depth), foam * 0.85 * smoothstep(0.003, 0.0045, depth));
    lit = atmosphere(in.world, lit);
    var out_rgb = tonemap(lit);
    // The car and anything else standing in the frame mirror in the sheet.
    // Three rays spread by the ripples: the car's mirror image breaks up
    // instead of sitting under it like glass.
    // Five rays spread mostly toward and away from the lens (a per-pixel
    // offset plus the ripple slope): the ripples stretch the body's mirror
    // image into soft vertical streaks instead of a sharp copy of the sill.
    // Half-res trace, accumulated over frames, upscaled by depth and
    // normal and blurred by the ripple roughness. Where it found nothing the
    // weight is low and the env (IBL) reflection already in out_rgb stays.
    let ssr0 = ssr_upsample(in.clip.xy, ssr_view_depth(in.world), nrm, w_rough);
    // Planar reflection of the Niva (wgpu water example, Ultra-Water): the
    // mirrored body at this pixel, shifted by the ripple slope and smeared
    // toward and away from the lens by the roughness.
    let pl_dims = vec2<f32>(textureDimensions(planar_tex));
    var pl = vec4<f32>(0.0);
    for (var k: i32 = -2; k <= 2; k = k + 1) {
        let off = vec2<f32>(nrm.x, nrm.z) * 0.02 * vec2<f32>(1.0, 0.6)
            + vec2<f32>(0.0, f32(k) * (1.0 + w_rough * 40.0)) / pl_dims;
        let puv_p = clamp(in.clip.xy / pl_dims + off, vec2<f32>(0.0), vec2<f32>(1.0));
        pl += textureSampleLevel(planar_tex, air_samp, puv_p, 0.0) * (3.0 - abs(f32(k)));
    }
    pl /= 9.0;
    // The planar image wins where it has the body; the screen trace keeps
    // the rest (the bank, the far scene).
    let pl_rgb = scene_to_display(pl.rgb / max(pl.a, 1e-3));
    let ssr = vec4<f32>(mix(ssr0.rgb, pl_rgb, pl.a), max(ssr0.a * (1.0 - pl.a), pl.a));
    if (ssr.a > 0.0) {
        // Near the Niva the sheet mirrors the sills, body and tires with its
        // Fresnel weight, blurred by the stretched rays; elsewhere a trace.
        let near_car = 1.0 - smoothstep(0.4, 2.2, max(length(xz - frame.air[1].xz) - 1.2, 0.0));
        let k = ssr.a * (1.0 - foam) * smoothstep(0.003, 0.006, depth) * min(f_uw, 0.85);
        out_rgb = mix(out_rgb, ssr.rgb * 0.92 * mix(1.0, w_ao, 0.35), clamp(k / max(alpha, 0.05), 0.0, 1.0));
        alpha = max(alpha, k);
    }
    // The car standing in the water mirrors in it: the tire's inverted image
    // hangs from the waterline, the sills and underbody darken the sheet
    // beneath them. Muted and broken by the ripples, but present even near
    // the lens, where the sky mirror is faded out.
    // Three rays bent by the ripple normal (plus a little noise) soften the
    // silhouette and make its edges wave; the image fades with how far the
    // mirrored point stands above the water.
    let mirror_o = in.world + nrm * 0.003;
    var mir_w = 0.0;
    // Six rays spread mostly toward and away from the lens: the ripples
    // stretch a mirror image vertically. Each ray fades with the height of
    // the point it meets above the water, so the image dies within 10-15 cm.
    let vfl = normalize(vec2<f32>(view.x, view.z) + vec2<f32>(1e-5, 0.0));
    let px_h = hash12(floor(in.clip.xy));
    for (var k: i32 = 0; k < 6; k = k + 1) {
        let fk = f32(k);
        let along = (fk + px_h) / 6.0 - 0.5;
        let side = noise(xz * 29.0 + vec2<f32>(fk * 7.0, t_now * 0.7)) - 0.5;
        let jn = vec3<f32>(vfl.x * along * 0.5 - vfl.y * side * 0.12, 0.0, vfl.y * along * 0.5 + vfl.x * side * 0.12);
        let mn = normalize(nrm * vec3<f32>(1.6, 1.0, 1.6) + jn);
        let md = reflect(-view, mn);
        let tm = tire_ray_hit(mirror_o, md, 0.6);
        if (tm > 0.0) {
            let h = max(mirror_o.y + md.y * tm - in.world.y, 0.0);
            mir_w += 1.0 - smoothstep(0.0, 0.08, h);
        }
    }
    mir_w /= 6.0;
    if (mir_w > 0.0) {
        let rub = sh_irradiance(vec3<f32>(0.0, -0.3, 1.0)) * frame.env.y * vec3<f32>(0.06, 0.05, 0.045);
        // Mirror weight is the Fresnel of this pixel: from the low lens the
        // sheet by the tire is looked into, so the image is a faint band
        // hanging from the waterline, not a dark disc.
        let w = 0.0 * mir_w;
        out_rgb = mix(out_rgb, tonemap(rub), w);
        alpha = max(alpha, w);
    }
    // Refraction after WebGL Water: bend the view ray into the water and
    // read the bed where it lands instead of straight below. The blend the
    // pipeline would do is done here, against the shifted copy.
    let refr = refract(-view, nrm, 1.0 / 1.333);
    // The ripples bend the bed under them: shift the lookup by the surface
    // slope, scaled by how much water lies between.
    // The slant run is capped: from a lens 20 cm up a full refracted run
    // dragged the car's shadow on the bed a metre toward the lens as a slab.
    let bed_hit = in.world + refr * min(depth / max(-refr.y, 0.3), 0.08) - vec3<f32>(nrm.x, 0.0, nrm.z) * min(depth, 0.06) * mix(0.25, 1.2, max(smoothstep(0.03, 0.1, depth), slurry_frac));
    let rc = frame.view_proj * vec4<f32>(bed_hit, 1.0);
    let dims_s = vec2<f32>(textureDimensions(post_color));
    let ruv = vec2<f32>(rc.x / rc.w * 0.5 + 0.5, 0.5 - rc.y / rc.w * 0.5);
    let here_c = frame.view_proj * vec4<f32>(in.world, 1.0);
    let huv = vec2<f32>(here_c.x / here_c.w * 0.5 + 0.5, 0.5 - here_c.y / here_c.w * 0.5);
    let rpix = vec2<i32>(clamp(ruv, vec2<f32>(0.0), vec2<f32>(0.9999)) * dims_s);
    let hpix = vec2<i32>(clamp(huv, vec2<f32>(0.0), vec2<f32>(0.9999)) * dims_s);
    // A sample that lands on something standing above the water (a tire)
    // would pull it under the surface; keep the straight-down texel then.
    let front = textureLoad(post_depth, rpix, 0).x < here_c.z / here_c.w;
    let off_screen = any(ruv < vec2<f32>(0.0)) || any(ruv > vec2<f32>(1.0));
    let dst = scene_to_display(textureLoad(post_color, select(rpix, hpix, front || over_tire > 0.5 || off_screen), 0).rgb);
    // Soft intersection with the rubber (bevy_water depth_diff): the sheet
    // thins over the last centimetre of view path in front of the tire, so
    // the waterline is a filtered line, not the pixel steps of two meshes.
    // The waterline at the rubber is the depth-diff fade alone, everywhere.
    // The sheet fades out where it meets anything standing in it: the
    // rubber, and the bank where the terrain mesh cuts the surface (that
    // cut was a hard black plate under the tire). The depth threshold is
    // widened by fwidth so it stays a couple of pixels soft at any range.
    let soft_tire = smoothstep(0.0, 0.05, view_path);
    let d_fw = max(fwidth(depth), 1e-5);
    let a_out = clamp(alpha, 0.0, 1.0) * smoothstep(edge_t - d_fw, 0.0055 + 2.0 * d_fw, depth) * soft_tire;
    // Beer-Lambert per channel on the bed seen through the water: the red
    // goes first, so shallow stays the sand's warm colour and the deeper
    // middle turns teal.
    // The per-channel tint printed rust-orange isolines over the shallows:
    // under a centimetre the bed keeps its own colour.
    let tint = mix(vec3<f32>(1.0), clamp(trans / max((trans.r + trans.g + trans.b) / 3.0, 1e-3), vec3<f32>(0.0), vec3<f32>(1.12)),
        smoothstep(0.03, 0.15, depth) * 0.6);
    if (frame.debug.x > 7.5) {
        return vec4<f32>(scene_to_display(textureLoad(post_color, hpix, 0).rgb), 1.0);
    }
    if (frame.debug.x > 6.5) {
        // 7: SSR weight (r), sky occlusion by the body (g), alpha (b).
        return vec4<f32>(ssr.a, 1.0 - w_ao, a_out, 1.0);
    }
    if (frame.debug.x > 5.5) {
        // 6: foam (r), slurry fraction (g), alpha (b).
        return vec4<f32>(foam, slurry_frac, a_out, 1.0);
    }
    if (frame.debug.x > 4.5) {
        // 5: film thickness 0-2 cm (blue 0, green 1 cm, red 2 cm), a line per mm.
        let dd5 = select(in.depth, column_c, inside > 0.5);
        let r5 = clamp(dd5 / 0.02, 0.0, 1.0);
        let ramp5 = select(mix(vec3<f32>(0.0, 0.25, 1.0), vec3<f32>(0.0, 0.9, 0.1), r5 * 2.0),
            mix(vec3<f32>(0.0, 0.9, 0.1), vec3<f32>(1.0, 0.05, 0.0), r5 * 2.0 - 1.0), r5 > 0.5);
        let mm = abs(fract(dd5 * 1000.0 + 0.5) - 0.5) / max(fwidth(dd5 * 1000.0), 1e-4);
        return vec4<f32>(mix(ramp5, vec3<f32>(0.0), (1.0 - smoothstep(0.5, 1.5, mm)) * 0.5), 1.0);
    }
    if (frame.debug.x > 2.5) {
        // 3: the water's own term only. 4: the bed seen through it only.
        return vec4<f32>(select(out_rgb * a_out, dst * tint * (1.0 - a_out), frame.debug.x > 3.5), 1.0);
    }
    if (frame.debug.x > 0.5) {
        // Depth debug: 0 cm blue, 5 cm green, 10 cm red, a line every 1 cm,
        // laid half over the scene so the rubber under it still shows.
        let dd = select(in.depth, column_c, inside > 0.5);
        let r = clamp(dd / 0.10, 0.0, 1.0);
        let ramp = select(mix(vec3<f32>(0.0, 0.25, 1.0), vec3<f32>(0.0, 0.9, 0.1), r * 2.0),
            mix(vec3<f32>(0.0, 0.9, 0.1), vec3<f32>(1.0, 0.05, 0.0), r * 2.0 - 1.0), r > 0.5);
        let cm = abs(fract(dd * 100.0 + 0.5) - 0.5) / max(fwidth(dd * 100.0), 1e-4);
        let line = 1.0 - smoothstep(0.5, 1.5, cm);
        return vec4<f32>(mix(dst * 0.7 + ramp * 0.3, vec3<f32>(0.0), line * 0.6), 1.0);
    }
    // The Niva's mirror image also lies in the thin edge of the sheet,
    // where the water itself has faded out (3-5.5 mm of published film):
    // only the trace's hits, at Ultra-Water's Fresnel, no sky sheen.
    let thin_k = smoothstep(0.003, 0.004, depth) * (1.0 - clamp(a_out, 0.0, 1.0)) * soft_tire * (1.0 - foam);
    let sheen = clamp(ssr.a * min(f_uw, 0.85) * thin_k, 0.0, 0.85);
    // The bed under the car sees little sky either.
    let base = dst * tint * (1.0 - a_out) * mix(1.0, w_ao, 0.7) + out_rgb * a_out;
    return vec4<f32>(mix(base, ssr.rgb * 0.92 * mix(1.0, w_ao, 0.35), sheen), 1.0);
}

// A film thinner than a few millimetres breaks the mirror into a sheen.
fn calm_or_film(depth: f32) -> f32 {
    return smoothstep(0.003, 0.015, depth);
}


// Open water: a river or a sea shore. The solved column is the free surface,
// bed + depth, not a film on the bed. A dry vertex next to the water carries
// the neighbour's surface flat into the bank, so the shoreline is where that
// surface meets the ground and the depth test cuts it.
fn open_eta(xz: vec2<f32>) -> vec2<f32> {
    let column = water_column(xz);
    return vec2<f32>(height_bilinear(xz) + column, column);
}

fn open_water_vertex(in: TerrIn) -> WaterOut {
    let here = open_eta(in.xz);
    let bed = here.x - here.y;
    let wet = smoothstep(0.004, 0.03, here.y);
    let rip = ripple_height(in.xz, frame.misc.x) * smoothstep(0.03, 0.2, here.y) * 0.6;
    var y = bed - 0.04;
    if (here.y > 0.004) {
        y = here.x + rip * wet;
    } else {
        let e = frame.params.z;
        var best = -1.0e6;
        for (var k = 0; k < 8; k = k + 1) {
            let a = f32(k) * 0.7853982;
            let n = open_eta(in.xz + vec2<f32>(cos(a), sin(a)) * e);
            if (n.y > 0.004) {
                best = max(best, n.x);
            }
        }
        if (best > -1.0e5) {
            y = best;
        }
    }
    let world = vec3<f32>(in.xz.x, y, in.xz.y);
    var o: WaterOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    o.depth = here.y;
    return o;
}

// ---- Turbid water and foam -------------------------------------------------
// Everything here reads the contract (binding 4 A turbidity, binding 4 RG
// flow, binding 9 foam) through water_fx_tex. Where the silt is, where the
// foam is and where both drift is the solver's. The noise below is only the
// texture inside a solved cloud, carried by the solved flow (flow-map
// technique as in three.js Water2.js, MIT): it never adds silt or foam where
// the maps are 0.

fn water_fx_at(ij: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(water_fx_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(water_fx_tex, c, 0);
}

// x turbidity 0..1, y foam 0..1, zw flow m/s.
fn water_fx(xz: vec2<f32>) -> vec4<f32> {
    if (frame.water_fx.y < 0.5) {
        return vec4<f32>(0.0);
    }
    let ij = world_to_ij(xz);
    let i0 = vec2<i32>(floor(ij));
    let f = fract(ij);
    let a = water_fx_at(i0);
    let b = water_fx_at(i0 + vec2<i32>(1, 0));
    let c = water_fx_at(i0 + vec2<i32>(0, 1));
    let d = water_fx_at(i0 + vec2<i32>(1, 1));
    let m = mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
    let t = m.r * m.r;
    return vec4<f32>(t, m.g, (m.b - 0.5) * 4.0, (m.a - 0.5) * 4.0);
}

fn fbm2(p: vec2<f32>) -> f32 {
    return noise(p) * 0.62 + noise(p * 2.13 + vec2<f32>(5.2, 1.3)) * 0.38;
}

// Two half-period phases of the same noise, each dragged by the flow and
// faded at its reset, so the pattern drifts downstream without a seam.
fn flow_noise(xz: vec2<f32>, flow: vec2<f32>, scale: f32, period: f32, octaves: i32) -> f32 {
    let t = frame.misc.x / period;
    let p0 = fract(t);
    let p1 = fract(t + 0.5);
    let w = abs(p0 - 0.5) * 2.0;
    let drift = flow * period;
    let q0 = (xz - drift * p0) * scale;
    let q1 = (xz - drift * p1) * scale + vec2<f32>(17.3, 9.1);
    var n0 = noise(q0);
    var n1 = noise(q1);
    if (octaves > 1) {
        n0 = fbm2(q0);
        n1 = fbm2(q1);
    }
    return mix(n0, n1, w);
}

struct Murk {
    // Silt concentration the eye sees, 0 clear to 1 opaque.
    cloud: f32,
    foam: f32,
    // Extinction, 1/m.
    sigma: f32,
    // Light on the billows: thick cores lit, the sun-far flank in shade.
    shade: f32,
}

fn murk_at(xz: vec2<f32>) -> Murk {
    let fx = water_fx(xz);
    var m: Murk;
    let detail = i32(frame.water_fx.x + 0.5);
    let flow = fx.zw;
    var t = fx.x;
    if (detail >= 1 && t > 0.0005) {
        // Billows inside the plume: thick cores, thin edges, all inside
        // the solved column. Mean factor ~1, so the cloud keeps its mass.
        // Two scales: metre-sized billows and the curl inside them.
        let big = flow_noise(xz, flow, 0.75, 3.1, detail);
        let small = flow_noise(xz + vec2<f32>(3.7, 1.9), flow, 2.4, 1.9, 1);
        let n = clamp(big * 0.7 + small * 0.3, 0.0, 1.0);
        t = t * (0.2 + 1.6 * n * n);
        m.shade = 0.8 + 0.4 * n;
        if (detail >= 2) {
            // One tap toward the sun: a billow whose sun side is thinner
            // than its core shades its own far flank.
            let sun = normalize(frame.sun_dir.xyz);
            let lean = sun.xz / max(length(sun.xz), 1e-3) * 0.35;
            let big_s = flow_noise(xz + lean, flow, 0.75, 3.1, detail);
            m.shade = clamp(0.85 + 1.4 * (big - big_s) + 0.3 * n, 0.55, 1.35);
        }
    } else {
        m.shade = 1.0;
    }
    m.cloud = 1.0 - exp(-t * 16.0);
    m.sigma = mix(3.2, 70.0, m.cloud);
    var foam = fx.y;
    if (foam > 0.002) {
        if (detail >= 1) {
            // Foam is a lace of bubble rafts that closes up as the fraction
            // rises, streaked along the current.
            let sp = length(flow);
            let dir = select(vec2<f32>(1.0, 0.0), flow / max(sp, 1e-4), sp > 0.05);
            let across = vec2<f32>(-dir.y, dir.x);
            let stretch = 1.0 + min(sp, 2.0) * 0.9;
            let fp = vec2<f32>(dot(xz, dir) / stretch, dot(xz, across));
            let fflow = vec2<f32>(sp / stretch, 0.0);
            let cells = flow_noise(fp, fflow, 6.5, 1.7, detail);
            let cover = clamp(foam * 1.15, 0.0, 1.0);
            var lace = smoothstep(1.0 - cover - 0.07, 1.0 - cover + 0.07, cells);
            if (detail >= 2) {
                let bub = flow_noise(fp * 2.7, fflow * 2.7, 9.0, 1.3, 1);
                lace = lace * (0.6 + 0.4 * smoothstep(0.3, 0.75, bub));
            }
            foam = lace * smoothstep(0.0, 0.1, fx.y) * 0.9;
        } else {
            foam = smoothstep(0.02, 0.7, foam) * 0.7;
        }
    }
    m.foam = clamp(foam, 0.0, 1.0);
    return m;
}

// Lit colour of silt-laden water: the bed's own soil, washed lighter by
// fine particles and lit by sun and sky from above.
fn silt_color(bed_col: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_dir.xyz);
    let lit = clamp(sun.y, 0.0, 1.0);
    let soil = mix(bed_col, vec3<f32>(0.40, 0.25, 0.11), 0.65);
    return soil * (frame.amb.rgb * 0.8 + frame.sky_zenith.rgb * 0.18 + frame.sun_color.rgb * lit * 0.42);
}

fn foam_color(bed_col: vec3<f32>, cloud: f32, nrm: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(nrm, sun), 0.0, 1.0) * 0.7 + 0.3;
    // Foam on muddy water is cream, not white.
    let alb = mix(vec3<f32>(0.74, 0.73, 0.69), vec3<f32>(0.62, 0.53, 0.41), cloud * 0.7);
    let sh = shadow_factor(world, nrm);
    return alb * (frame.amb.rgb * 0.8 + frame.sky_zenith.rgb * 0.3 + frame.sun_color.rgb * ndl * sh * 0.95);
}

fn open_water_fragment(in: WaterOut, bed_col: vec3<f32>) -> vec4<f32> {
    let xz = in.world.xz;
    // Depth of water over the ground under this fragment.
    let d = in.world.y - height_bilinear(xz);
    if (d <= 0.0) {
        discard;
    }
    // Surface slope from the solved column (bow, wake, swell), plus ripples.
    let de = 0.2;
    let e0 = open_eta(xz);
    let ex = open_eta(xz + vec2<f32>(de, 0.0));
    let ez = open_eta(xz + vec2<f32>(0.0, de));
    var gx = 0.0;
    var gz = 0.0;
    if (ex.y > 0.004 && e0.y > 0.004) {
        gx = clamp((ex.x - e0.x) / de, -0.6, 0.6);
    }
    if (ez.y > 0.004 && e0.y > 0.004) {
        gz = clamp((ez.x - e0.x) / de, -0.6, 0.6);
    }
    let r = ripple_normal(xz, frame.misc.x);
    let nrm = normalize(vec3<f32>(-gx * 1.6 + r.x * 0.5, 1.0, -gz * 1.6 + r.z * 0.5));
    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let ndotv = clamp(dot(nrm, view), 0.0, 1.0);
    let fres = 0.02 + 0.98 * pow(1.0 - ndotv, 5.0);
    let refl = sky_color(reflect(-view, nrm));
    // River water: clear in the shallows, olive-brown with depth. The ground
    // under it is the terrain pass; alpha is how much of it the column hides.
    let mk = murk_at(xz);
    let path = d / max(ndotv, 0.25);
    let t = exp(-path * mk.sigma);
    let lit = clamp(dot(vec3<f32>(0.0, 1.0, 0.0), sun), 0.0, 1.0);
    let clear = vec3<f32>(0.075, 0.095, 0.07) * (0.35 + 1.4 * lit) + bed_col * 0.08;
    let scatter = mix(clear, silt_color(bed_col) * mk.shade, mk.cloud);
    let spec = pow(clamp(dot(nrm, normalize(sun + view)), 0.0, 1.0), 220.0)
        * shadow_factor(in.world, nrm);
    let shore = smoothstep(0.0, 0.025, d);
    let hide = 1.0 - t * (1.0 - fres);
    let alpha = clamp(hide * shore, 0.0, 1.0);
    var col = (scatter * (1.0 - t) * (1.0 - fres) + refl * fres) / max(hide, 1.0e-3);
    col += frame.sun_color.rgb * spec * 2.5 / max(alpha, 0.05);
    // Foam floats on the surface: it hides the column and kills the glint.
    let fc = foam_color(bed_col, mk.cloud, nrm, in.world);
    col = mix(col, fc, mk.foam);
    let a_out = mix(max(alpha, min(spec * 2.0, 1.0) * shore), shore, mk.foam);
    let gamma = from_display(legacy_curve(col, 1.08));
    return vec4<f32>(gamma, a_out);
}

fn water_vertex(in: TerrIn) -> WaterOut {
    if (open_water_on()) {
        return open_water_vertex(in);
    }
    if (sand_field()) {
        return beauty_water_vertex(in);
    }
    if (win_demo_flat_terrain()) {
        let t = frame.misc.x;
        let column = max(water_column(in.xz), win_demo_seam_column(in.xz));
        let bed = height_bilinear(in.xz);
        let level = win_demo_pool_level();
        let calm = smoothstep(5.0, 4.0, t);
        let rip = ripple_height(in.xz, t) * 0.004 * (1.0 - calm * 0.85);
        let bow = max(beauty_bow_wave(in.xz, t), 0.0);
        let driven = smoothstep(4.0, 5.5, t);
        var y = level + rip + bow * (0.022 + driven * 1.05);
        y = win_demo_clamp_surface_y(in.xz, y);
        var o: WaterOut;
        o.clip = frame.view_proj * vec4<f32>(in.xz.x, y, in.xz.y, 1.0);
        o.world = vec3<f32>(in.xz.x, y, in.xz.y);
        o.ij = in.ij;
        o.depth = max(column, max(level - bed, 0.0));
        return o;
    }
    let local = height_bilinear(in.xz);
    let dug = max(-stamp_bilinear(in.xz), 0.0);
    let depth = covered_depth(in.xz);
    let column = water_column(in.xz);
    let beauty = puddle_beauty_on();
    let free = water_column_on() && column > 0.004;
    let rip_scale = select(1.0, 1.0, beauty);
    let rip = ripple_height(in.xz, frame.misc.x) * smoothstep(0.0, 0.03, depth) * rip_scale;
    // Wake ripples stay in the normal on beauty pools; lifting the sheet here
    // reads as blue spheres under the tires.
    let wake = select(tire_wake_ripples(in.xz, frame.misc.x), 0.0, beauty);
    // Film on the bed, but never below the soil. A bump left uncovered
    // reads as a black hole. bed + full depth was a sheet 30 cm up that
    // cut the tire like a clip plane. The pond stays that blurred surface.
    // A cut holds its own sheet down in the groove.
    // A crossing sets misc.w and the sheet is the solved column, so the
    // bow is the crest of that column.
    let cover = local + 0.01;
    let pond = max(water_bed(in.xz) + 0.016 + rip, cover) + smoothstep(0.02, 0.16, depth) * 0.012;
    let fill = smoothstep(0.04, 0.12, depth) * min(dug * 0.5, 0.032);
    let in_rut = local + 0.01 + rip + fill;
    let y_sheet = select(pond, in_rut, dug > 0.018 && depth > 0.004);
    // The column is the solved water. Drawing it at full height is a mound.
    // The sheet sits on the ground. The bow is a crest in that sheet, plus a swell.
    let swell = select(0.0, free_swell(in.xz, frame.misc.x), free);
    let swell_amp = select(1.0, 0.0, beauty);
    let rut_floor = max(water_bed(in.xz), local + 0.006);
    let base = select(water_bed(in.xz), rut_floor, beauty && dug > 0.012);
    let seam_flat = beauty_seam_weight(in.xz);
    let in_beauty_pool = beauty && (column > 0.001 || seam_flat > 0.08);
    let lift = max(column, column * 0.14);
    let rip_vert = rip * select(0.45, 0.0, beauty);
    let flat_pool = mix(
        water_bed(in.xz) + 0.006 + rip_vert,
        beauty_seam_plane_y(in.xz) + rip_vert,
        seam_flat,
    );
    let surface = base + water_surface_lift(lift) + rip_vert + wake + swell * swell_amp;
    var y = select(select(y_sheet, surface, free), flat_pool, in_beauty_pool);
    let seam_span = select(1.42, 3.25, win_demo_flat_terrain());
    if (beauty && (distance(in.xz, beauty_seam_center()) < seam_span || column > 0.0008)) {
        y = flat_pool;
    }
    y = win_demo_clamp_surface_y(in.xz, y);
    let world = vec3<f32>(in.xz.x, y, in.xz.y);
    var o: WaterOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    o.depth = select(depth, max(column, column * 0.18), free);
    return o;
}

fn shade_seam_pool(
    in: WaterOut,
    xz: vec2<f32>,
    bed: vec3<f32>,
    column_here: f32,
    _tire_pad: bool,
    ultra: bool,
) -> vec4<f32> {
    let t = frame.misc.x;
    var nrm = normalize(puddle_normal(xz, t, column_here));
    if (win_demo_flat_terrain()) {
        if (frame.reflect_flags.y > 0.5) {
            let planar = pool_planar_reflect_sample(in.world);
            return vec4<f32>(from_display(pow(max(planar, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2))), 0.96);
        }
        let reflect_ready = pool_reflect_ready();
        let bow = beauty_bow_wave(xz, t);
        let driven = smoothstep(4.0, 5.5, t);
        let ripple_mix = clamp(0.12 + max(bow, 0.0) * 3.2 * (0.35 + driven * 0.65), 0.12, 0.58);
        nrm = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), nrm, ripple_mix));
        nrm.y = max(nrm.y, 0.55);
        let view = normalize(frame.cam_pos.xyz - in.world);
        let sun = normalize(frame.sun_dir.xyz);
        let ndotv = clamp(dot(nrm, view), 0.0, 1.0);
        let fres = schlick_fresnel(ndotv);
        let ndl = clamp(dot(nrm, sun), 0.0, 1.0);
        let water_tint = vec3<f32>(0.05, 0.34, 0.52);
        let body = water_tint * (0.12 + 0.88 * ndl);
        var reflect_col = pool_planar_reflect_sample(in.world);
        let refl_dir = reflect(-view, nrm);
        if (length(reflect_col) < 0.02 && !reflect_ready) {
            reflect_col = sky_reflect(refl_dir);
        }
        let fres_w = clamp(fres * 0.28 + 0.94, 0.0, 1.0);
        var col = reflect_col * (0.08 + 0.92 * fres_w) + body * (1.0 - fres_w) * 0.12;
        var bow_foam = max(beauty_bow_wave(xz, t), 0.0) * smoothstep(4.0, 5.5, t) * 2.4;
        if (reflect_ready) {
            bow_foam = 0.0;
        }
        col = mix(col, vec3<f32>(0.90, 0.94, 0.98), clamp(bow_foam, 0.0, 1.0) * 0.55);
        let halfv = normalize(sun + view);
        let sheen = pow(clamp(dot(nrm, halfv), 0.0, 1.0), 128.0);
        col += vec3<f32>(0.82, 0.88, 0.96) * sheen * 0.08;
        let c = beauty_seam_center();
        let dx = (xz.x - c.x) / 1.32;
        let dz = (xz.y - c.y) / 1.02;
        let r = sqrt(dx * dx + dz * dz);
        let edge_a = smoothstep(1.40, 0.68, r);
        let core_a = smoothstep(1.05, 0.42, r);
        var alpha = max(edge_a * 0.92, core_a * 0.97);
        if (reflect_ready && win_demo_in_pool(xz)) {
            alpha = max(alpha, 0.94);
        }
        let gamma = from_display(legacy_curve(col, 1.05));
        return vec4<f32>(gamma, alpha);
    }
    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let ndotv = clamp(dot(nrm, view), 0.0, 1.0);
    let fres = schlick_fresnel(ndotv);
    let ndl = clamp(dot(nrm, sun), 0.0, 1.0);
    let bed_h = height_bilinear(xz);
    let level = win_demo_pool_level();
    let thickness = max(column_here, max(in.depth, level - bed_h));
    let deep = smoothstep(0.04, 0.22, thickness);
    let shallow = 1.0 - smoothstep(0.012, 0.10, thickness);
    let water_tint = vec3<f32>(0.05, 0.34, 0.52);
    let water_deep = vec3<f32>(0.02, 0.16, 0.30);
    var body = mix(water_tint, water_deep, deep);
    body = mix(mix(bed * vec3<f32>(0.42, 0.34, 0.26), water_tint * 0.62, 0.55), body, 1.0 - shallow * 0.35);
    body *= 0.22 + 0.88 * ndl;
    let refl_dir = reflect(-view, nrm);
    var reflect_col = pool_scene_reflect(in.world, nrm, view);
    let ibl = sky_reflect(refl_dir);
    let sun_disk = pow(clamp(dot(refl_dir, sun), 0.0, 1.0), 96.0);
    reflect_col = mix(reflect_col, ibl * 1.52 + frame.sun_color.rgb * sun_disk * 0.72, 0.52);
    let veh_scene = reflect_vehicle_scene(in.world, nrm, view);
    let planar = niva_planar_reflect(in.world, nrm, view);
    if (length(veh_scene) > 0.03) {
        reflect_col = mix(reflect_col, veh_scene, select(0.62, 0.78, ultra));
    }
    if (length(planar) > 0.02) {
        reflect_col = mix(reflect_col, planar, select(0.48, 0.62, ultra));
    }
    let fres_w = clamp(
        fres * select(0.82, 0.95, ultra) + select(0.38, 0.52, ultra),
        0.0,
        1.0,
    );
    var col = mix(body, reflect_col, fres_w);
    let g = length(solver_column_grad(xz));
    let shore_foam = smoothstep(0.08, 0.22, thickness) * (1.0 - smoothstep(0.22, 0.32, thickness));
    let edge_foam = smoothstep(0.04, 0.14, g);
    let foam = clamp(shore_foam * 0.55 + edge_foam * 0.35, 0.0, 1.0);
    col = mix(col, vec3<f32>(0.90, 0.94, 0.98), foam * 0.62);
    let shore_wet = smoothstep(0.11, 0.018, column_here) * (1.0 - smoothstep(0.0, 0.012, column_here));
    col = mix(col, mix(bed * vec3<f32>(0.28, 0.22, 0.16), col, 0.55), shore_wet * 0.45);
    let halfv = normalize(sun + view);
    let sheen = pow(clamp(dot(nrm, halfv), 0.0, 1.0), select(64.0, 128.0, ultra));
    col += vec3<f32>(0.82, 0.88, 0.96) * sheen * select(0.22, 0.38, ultra);
    var edge_a = smoothstep(0.0, 0.016, column_here);
    col = mix(mix(bed * vec3<f32>(0.36, 0.30, 0.24), col, 0.65), col, edge_a);
    var alpha = mix(0.76, 0.91, deep) * edge_a;
    alpha = max(alpha, select(0.74, 0.82, ultra) * edge_a);
    let gamma = from_display(legacy_curve(col, 1.05));
    return vec4<f32>(gamma, alpha);
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    let beauty = puddle_beauty_on();
    let xz = in.world.xz;
    if (win_demo_flat_terrain() && !win_demo_in_pool(xz)) {
        discard;
    }
    let kind = kind_nearest(xz);
    let uv0 = xz / 3.2;
    let uv1 = xz / 2.7;
    let uv2 = xz / 3.5;
    let uv3 = xz / 2.3;
    let uv4 = xz / 3.3;
    let bed = pick5(
        layer_of(kind),
        textureSample(ground_diff, ground_samp, uv0, 0),
        textureSample(ground_diff, ground_samp, uv1, 1),
        textureSample(ground_diff, ground_samp, uv2, 2),
        textureSample(ground_diff, ground_samp, uv3, 3),
        textureSample(ground_diff, ground_samp, uv4, 4),
    ).rgb;
    if (open_water_on()) {
        return open_water_fragment(in, bed);
    }
    if (sand_field()) {
        let bw = beauty_water_fragment(in);
        return vec4<f32>(from_display(bw.rgb), bw.a);
    }
    let own = mat_bilinear(xz).g * 0.34;
    let dug = max(-stamp_bilinear(xz), 0.0);
    // A cut keeps film and slurry on the soil mesh. Dilated water on the lip
    // was drawing flat cards in the pockets beside the floor.
    let e = vec2<f32>(0.45, 0.0);
    let ez = vec2<f32>(0.0, 0.45);
    let near_cut = dug > 0.008
        || -stamp_bilinear(xz + e) > 0.02
        || -stamp_bilinear(xz - e) > 0.02
        || -stamp_bilinear(xz + ez) > 0.02
        || -stamp_bilinear(xz - ez) > 0.02;
    var column_here = water_column(xz);
    let demo_column = win_demo_seam_column(xz);
    if (win_demo_flat_terrain()) {
        column_here = max(column_here, demo_column);
    }
    let seam_still = win_demo_flat_terrain() && distance(xz, beauty_seam_center()) < 3.25;
    let free = water_column_on() && in.depth > 0.004;
    let tire_pad = water_near_tire_hub(xz, 0.32);
    let seam_flat = beauty_seam_weight(xz);
    let in_seam_pan = (beauty || win_demo_flat_terrain()) && distance(xz, beauty_seam_center()) < 3.25;
    let pool_here = column_here > 0.0035
        || (beauty && tire_pad)
        || (beauty && column_here > 0.0025)
        || (beauty && seam_flat > 0.12)
        || (in_seam_pan && column_here > 0.0008);
    let beauty_pool = beauty && pool_here;
    let demo_pool = win_demo_flat_terrain() && demo_column > 0.00045;
    let keep_water = free
        || pool_here
        || demo_pool
        || (win_demo_flat_terrain() && win_demo_in_pool(xz))
        || (beauty && tire_pad)
        || (beauty && in_seam_pan)
        || (seam_still && column_here > 0.0005);
    if (win_demo_flat_terrain()) {
        if (!keep_water) {
            discard;
        }
    } else if (!keep_water && (near_cut || own < 0.008 || in.depth < 0.008)) {
        discard;
    }
    let seam_pool = select(
        (beauty || win_demo_flat_terrain()) && (in_seam_pan || tire_pad || column_here > 0.0008),
        win_demo_flat_terrain() && (demo_column > 0.00035 || tire_pad || win_demo_in_pool(xz)),
        win_demo_flat_terrain(),
    );
    if (seam_pool) {
        let ultra = beauty && puddle_beauty_on();
        return shade_seam_pool(in, xz, bed, column_here, tire_pad, ultra);
    }
    let free_water = free || beauty_pool || (seam_still && column_here > 0.0004);
    let pool_norm = beauty || (win_demo_flat_terrain() && (seam_still || column_here > 0.002));
    let n = select(ripple_normal(xz, frame.misc.x), puddle_normal(xz, frame.misc.x, column_here), pool_norm);
    let flat = normalize(vec3<f32>(n.x * 0.8, 1.0, n.z * 0.8));
    // Slope of the low sheet, not of the full column. A full-column normal
    // is a wall.
    let de = 0.28;
    let y0 = water_surface_lift(water_column(xz));
    let yx = water_surface_lift(water_column(xz + vec2<f32>(de, 0.0)));
    let yz = water_surface_lift(water_column(xz + vec2<f32>(0.0, de)));
    let g = free_swell_grad(xz, frame.misc.x);
    // The crest is only a few centimetres. Stretch its normal so the face
    // of the wave reads, without standing the water up into a mound.
    let dx = (yx - y0) / de;
    let dz = (yz - y0) / de;
    let slope = normalize(vec3<f32>(-(dx * 2.6 + g.x * 2.4), 1.0, -(dz * 2.6 + g.y * 2.4)));
    let nrm = select(
        flat,
        select(slope, normalize(puddle_normal(xz, frame.misc.x, column_here)), pool_norm),
        free,
    );
    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let ndotv = clamp(dot(nrm, view), 0.0, 1.0);
    let fres = schlick_fresnel(ndotv);
    let depth_t = smoothstep(0.006, 0.075, in.depth);
    // Film shows the bed. Deep slurry is that same bed, darker and opaque.
    let muddy = kind == 4u || kind == 5u || kind == 6u || kind == 8u || kind == 9u;
    var film_col = mix(bed, bed * vec3<f32>(0.5, 0.36, 0.22), 0.7);
    var deep_col = mix(bed * vec3<f32>(0.32, 0.18, 0.1), vec3<f32>(0.022, 0.009, 0.003), 0.72);
    if (!muddy) {
        film_col = mix(bed, vec3<f32>(0.24, 0.2, 0.14), 0.2);
        deep_col = mix(bed, vec3<f32>(0.07, 0.055, 0.04), 0.38);
    }
    let absorption = mix(film_col, deep_col, depth_t);
    let quiet = mix(bed, vec3<f32>(0.1, 0.045, 0.016), select(0.3, 0.72, muddy));
    let reflect_amt = fres * (0.22 + (1.0 - depth_t) * 0.16);
    var col = mix(absorption, quiet, reflect_amt);
    let spec = pow(clamp(dot(nrm, normalize(sun + view)), 0.0, 1.0), 28.0);
    let spec_col = mix(bed, vec3<f32>(0.04, 0.016, 0.005), 0.65);
    col += spec_col * spec * shadow_factor(in.world, nrm);
    let shore = smoothstep(0.002, 0.07, in.depth);
    let reveal = smoothstep(0.016, 0.034, dug);
    let open = mix(0.24, 0.42, smoothstep(0.015, 0.10, in.depth));
    let cover = select(mix(0.34, 0.58, depth_t), mix(0.55, 0.97, depth_t), muddy);
    // A damp speck still opens the groove. A film and a slurry keep the
    // opacity of the bed under them.
    let skim = 1.0 - smoothstep(0.012, 0.028, in.depth);
    let solved = shore * cover * mix(1.0, mix(1.0, open, skim), reveal);
    // A bowl is a cut, and the groove path opens that cut. The ford sheet is
    // the free surface, so the crest stays a surface instead of a tint.
    let shore_fade = smoothstep(0.003, 0.07, in.depth);
    var alpha = select(solved, shore_fade, free_water);
    if (free_water) {
        // Turbid water. The bed shows through the troughs. Crests catch the
        // sun so the swell reads as a wave, not a flat stain.
        let soil = bed;
        let deep = smoothstep(0.02, 0.12, column_here);
        let murk = soil * vec3<f32>(0.26, 0.17, 0.08);
        var body = mix(murk, vec3<f32>(0.07, 0.045, 0.018), deep * 0.5);
        let ndl = clamp(dot(nrm, sun), 0.0, 1.0);
        col = body * (0.10 + 2.8 * ndl);
        let halfv = normalize(sun + view);
        let spec_pow = select(28.0, 128.0, beauty);
        let sheen = pow(clamp(dot(nrm, halfv), 0.0, 1.0), spec_pow);
        col += vec3<f32>(0.62, 0.46, 0.24) * sheen * select(1.5, 2.4, beauty);
        alpha = shore_fade * mix(0.46, 0.72, deep);
        if (beauty || win_demo_flat_terrain()) {
            let ultra = beauty && puddle_beauty_on();
            let refl_dir = reflect(-view, nrm);
            let ibl = sky_reflect(refl_dir);
            let sun_disk = pow(clamp(dot(refl_dir, sun), 0.0, 1.0), 112.0);
            let ssr = scene_ssr(in.world, nrm, view);
            let planar = niva_planar_reflect(in.world, nrm, view);
            let veh_scene = reflect_vehicle_scene(in.world, nrm, view);
            let veh = niva_water_reflect(in.world, refl_dir);
            let sky_base = ibl * 1.42 + frame.sun_color.rgb * sun_disk * 0.78;
            var reflect_col = mix(sky_base, veh_scene, select(0.55, 0.82, ultra));
            if (length(veh_scene) < 0.03) {
                reflect_col = mix(sky_base, veh * 1.1, 0.52);
            }
            if (length(ssr) > 0.02) {
                reflect_col = mix(reflect_col, ssr, select(0.42, 0.68, ultra));
            }
            if (length(planar) > 0.02) {
                reflect_col = mix(reflect_col, planar, select(0.38, 0.58, ultra));
            }
            if (length(veh_scene) > 0.04) {
                reflect_col = mix(reflect_col, veh_scene, 0.38);
            }
            let sky_uv = scene_uv_from_world(in.world + normalize(refl_dir) * 0.55);
            if (sky_uv.x > 0.02 && sky_uv.x < 0.98) {
                let sky_samp = scene_to_display(textureSampleLevel(scene_color, scene_samp, sky_uv, 0.0).rgb);
                reflect_col = mix(reflect_col, mix(sky_base, sky_samp, 0.78), select(0.32, 0.48, ultra));
            }
            let ibl_lit = reflect_col;
            let rim = smoothstep(0.22, 0.006, column_here) * (1.0 - smoothstep(0.0002, 0.005, column_here));
            col = mix(col, vec3<f32>(0.002, 0.0015, 0.001), rim * 1.25);
            let shore_wet = smoothstep(0.13, 0.012, column_here) * (1.0 - smoothstep(0.001, 0.010, column_here));
            col = mix(col, col * vec3<f32>(0.14, 0.10, 0.08), shore_wet * 0.82);
            let rings = tire_wake_ripples(xz, frame.misc.x);
            let bow_h = beauty_bow_wave(xz, frame.misc.x);
            col += vec3<f32>(0.14, 0.42, 0.58) * rings * select(3.2, 5.2, ultra);
            col += vec3<f32>(0.42, 0.52, 0.62) * max(bow_h, 0.0) * select(8.0, 14.0, ultra);
            let shallow = 1.0 - smoothstep(0.008, 0.095, column_here);
            let water_tint = vec3<f32>(0.06, 0.38, 0.55);
            let water_deep = vec3<f32>(0.03, 0.22, 0.38);
            body = mix(water_tint, water_deep, deep);
            let bed_shallow = mix(soil * vec3<f32>(0.34, 0.28, 0.22), water_tint * 0.55, 0.62);
            body = mix(body, bed_shallow, shallow * 0.72);
            let body_lit = body * (0.18 + 0.95 * ndl);
            let fres_mix = clamp(fres * select(0.75, 0.92, ultra) + select(0.48, 0.62, ultra), 0.0, 1.0);
            col = mix(body_lit, ibl_lit, fres_mix);
            if (length(veh_scene) < 0.04 && length(planar) < 0.04) {
                col = mix(col, ibl_lit * 0.55 + veh * 1.35, select(0.55, 0.72, ultra));
            }
            col += vec3<f32>(0.88, 0.92, 1.0) * sheen * select(0.45, 0.72, ultra);
            col += beauty_tire_splash(xz, frame.misc.x) * select(0.55, 1.0, ultra);
            alpha = mix(0.58, 0.90, deep) * shore_fade;
            alpha = mix(alpha, alpha * 0.55, shallow * 0.78);
            if (in_seam_pan) {
                alpha = max(alpha, select(0.68, 0.74, ultra) * shore_fade);
            }
            if (tire_pad) {
                alpha = max(alpha, select(0.72, 0.80, ultra) * shore_fade);
            }
        }
    }
    // Contract silt and foam on the film and the puddle sheet.
    let mk = murk_at(xz);
    if (mk.cloud > 0.0 || mk.foam > 0.0) {
        let lit_silt = silt_color(bed) * mk.shade * (0.55 + 0.45 * shadow_factor(in.world, nrm));
        col = mix(col, mix(lit_silt, col, fres * 0.6), mk.cloud * 0.85);
        alpha = mix(alpha, max(alpha, shore_fade * 0.92), mk.cloud);
        col = mix(col, foam_color(bed, mk.cloud, nrm, in.world), mk.foam);
        alpha = mix(alpha, shore_fade, mk.foam);
    }
    let gamma = from_display(legacy_curve(col, 1.08));
    return vec4<f32>(gamma, alpha);
}

struct MeshIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) panel: f32,
}

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) panel: f32,
    @location(4) recess: f32,
    // x: tread void on the Niva crown. y: sidewall band. Both from the mesh.
    @location(5) carcass: vec2<f32>,
    // Model-space position, so coat patterns ride with the body.
    @location(6) local: vec3<f32>,
}

// 1 inside a tread void, 0 on a lug face. uv.x winds around the tire,
// uv.y runs from bead to bead with the crown at 0.5.
fn tread_void(uv: vec2<f32>) -> f32 {
    // The CC0 carcass is three rings: beads at v = 0 and 1, shoulders near
    // 0.12 and 0.88, and one crown ring at v = 0.5. The crown is the groove.
    let from_crown = abs(uv.y - 0.5);
    let on_tread = 1.0 - smoothstep(0.36, 0.47, from_crown);
    let center = 1.0 - smoothstep(0.04, 0.20, from_crown);
    let lug = abs(fract(uv.x * 0.5) - 0.5);
    let gaps = smoothstep(0.34, 0.08, lug);
    let ribs = abs(fract(from_crown * 6.0) - 0.5);
    let channel = smoothstep(0.28, 0.06, ribs);
    return clamp(on_tread * max(center * gaps, channel * 0.85), 0.0, 1.0);
}

fn displace_tread(pos: vec3<f32>, uv: vec2<f32>, panel: f32) -> vec3<f32> {
    if (panel < 60.5 || panel > 61.5) {
        return pos;
    }
    let void = tread_void(uv);
    let fill = clamp(model.extra.z, 0.0, 1.0);
    // An empty void sits under the lug. Mud packed by a rolling tire rises
    // back toward the crown, so the groove reads full instead of as a hole.
    let depth = 0.026 * void * (1.0 - 0.94 * fill);
    let radial = vec3<f32>(0.0, pos.y, pos.z);
    let len = length(radial);
    if (len < 1e-4) {
        return pos;
    }
    return pos - radial / len * depth;
}

// Crown voids and the sidewall band of the Lada carcass. The mesh is a
// smooth barrel, so the grooves are this mask, driven by the contact patch.
fn niva_carcass(pos: vec3<f32>, nrm: vec3<f32>, panel: f32) -> vec2<f32> {
    if (model.extra.x < 1.5 || panel < 84.5 || panel > 85.5) {
        return vec2<f32>(0.0, 0.0);
    }
    let radial = length(pos.yz);
    let axle = abs(pos.x);
    let rubber = smoothstep(0.27, 0.33, radial);
    let facing_out = smoothstep(0.40, 0.72, length(nrm.yz));
    let facing_side = smoothstep(0.40, 0.75, abs(nrm.x));
    let side = smoothstep(0.060, 0.082, axle);
    let crown = rubber * (1.0 - smoothstep(0.046, 0.074, axle)) * facing_out;
    let u = atan2(pos.z, pos.y) / 6.2831853;
    let row = select(0.0, 0.5, pos.x > 0.0);
    let lug = abs(fract(u * 12.0 + row + 1.0) - 0.5);
    let gap = smoothstep(0.22, 0.06, lug);
    let shoulder = smoothstep(0.010, 0.028, axle);
    let channel = 1.0 - smoothstep(0.006, 0.022, axle);
    let void = crown * max(gap * shoulder, channel * 0.65);
    let crown_band = 1.0 - smoothstep(0.046, 0.074, axle);
    let wall = rubber * side * (1.0 - crown_band) * max(facing_side, facing_out);
    return vec2<f32>(clamp(void, 0.0, 1.0), clamp(wall, 0.0, 1.0));
}

fn displace_niva(pos: vec3<f32>, nrm: vec3<f32>, panel: f32) -> vec3<f32> {
    let void = niva_carcass(pos, nrm, panel).x;
    let fill = clamp(model.extra.z, 0.0, 1.0);
    // Empty void sits under the lug. Packed clay rises back to the crown.
    let depth = 0.020 * void * (1.0 - 0.90 * fill);
    let radial = vec3<f32>(0.0, pos.y, pos.z);
    let len = length(radial);
    if (len < 1e-4) {
        return pos;
    }
    return pos - radial / len * depth;
}

fn displace_mesh(pos: vec3<f32>, nrm: vec3<f32>, uv: vec2<f32>, panel: f32) -> vec3<f32> {
    return displace_niva(displace_tread(pos, uv, panel), nrm, panel);
}

fn tread_recess(pos: vec3<f32>, panel: f32) -> f32 {
    // Procedural tire only. Its grooves are cut into the mesh. The CC0 tire
    // is a smooth barrel; its voids come from the uv, not the radius.
    if (panel < 19.5 || panel > 20.5) {
        return 0.0;
    }
    let radial = length(pos.yz);
    let axle = abs(pos.x);
    let drop = 0.374 - radial;
    let tread = 1.0 - smoothstep(0.10, 0.145, axle);
    let band = smoothstep(0.004, 0.012, drop) * (1.0 - smoothstep(0.055, 0.09, drop));
    return band * tread;
}

@vertex
fn vs_mesh(in: MeshIn) -> MeshOut {
    let pos = displace_mesh(in.pos, in.nrm, in.uv, in.panel);
    var world = model.world * vec4<f32>(pos, 1.0);
    world = pool_reflect_mirror_world(world);
    var o: MeshOut;
    o.clip = frame.view_proj * world;
    o.world = world.xyz;
    var nrm = normalize((model.world * vec4<f32>(in.nrm, 0.0)).xyz);
    if (pool_reflect_geom_pass()) {
        nrm.y = -nrm.y;
    }
    o.nrm = nrm;
    o.uv = in.uv;
    o.panel = in.panel;
    o.recess = tread_recess(pos, in.panel);
    o.carcass = niva_carcass(in.pos, in.nrm, in.panel);
    o.local = pos;
    return o;
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) depth: f32,
}

// On the beauty sand the drawn ground is the contract's (binding 1), which
// this pass cannot read; the solver's own heightfield here held a thin ridge
// that threw a long black line from the car. The sand under a high sun has no
// relief worth a cast shadow, so it casts none.
fn shadow_terrain_vertex(in: TerrIn) -> ShadowOut {
    let base = height_bilinear(in.xz);
    let h = select(apply_seat(in.xz, base).x, base, skip_tire_seat()) - select(0.0, 50.0, sand_field());
    let world = vec3<f32>(in.xz.x, h, in.xz.y);
    let clip = frame.light_vp * vec4<f32>(world, 1.0);
    var o: ShadowOut;
    o.clip = clip;
    o.depth = clip.z / max(clip.w, 1e-5);
    return o;
}

@vertex
fn vs_shadow_mesh(in: MeshIn) -> ShadowOut {
    let pos = displace_mesh(in.pos, in.nrm, in.uv, in.panel);
    let world = model.world * vec4<f32>(pos, 1.0);
    let clip = frame.light_vp * world;
    // Slurry patches lying on the beauty sand are a few millimetres thick;
    // in the shadow map they printed blocky dark blotches beside them.
    if (sand_field() && in.panel > 48.5 && in.panel < 49.5) {
        var q: ShadowOut;
        q.clip = vec4<f32>(9.0, 9.0, 9.0, 1.0);
        q.depth = 1.0;
        return q;
    }
    var o: ShadowOut;
    o.clip = clip;
    o.depth = clip.z / max(clip.w, 1e-5);
    return o;
}

@fragment
fn fs_shadow(in: ShadowOut) -> @location(0) vec4<f32> {
    return encode_depth(in.depth);
}


fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn tbn_normal(n_in: vec3<f32>, nm: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);
    var t = vec3<f32>(1.0, 0.0, 0.0);
    t = normalize(t - n * dot(n, t));
    let b = cross(n, t);
    let ts = nm * 2.0 - vec3<f32>(1.0);
    return normalize(t * ts.x + b * ts.y + n * ts.z);
}

fn mud_normal(n: vec3<f32>, p: vec3<f32>, amount: f32) -> vec3<f32> {
    let e = 0.04;
    let h0 = noise(p.xz * 7.0 + vec2<f32>(p.y * 2.0, 0.0));
    let hx = noise((p.xz + vec2<f32>(e, 0.0)) * 7.0 + vec2<f32>(p.y * 2.0, 0.0));
    let hz = noise((p.xz + vec2<f32>(0.0, e)) * 7.0 + vec2<f32>(p.y * 2.0, 0.0));
    return normalize(n + vec3<f32>(h0 - hx, 0.0, h0 - hz) * amount * 1.6);
}

fn panel_paint(panel: f32, uv: vec2<f32>, world: vec3<f32>) -> vec4<f32> {
    // rgb albedo, a roughness. Panel ids match the vehicle solver.
    let grit = hash12(floor(world.xz * 8.0 + vec2<f32>(world.y, world.y * 0.3)));
    if (panel < 0.5) {
        return vec4<f32>(vec3<f32>(0.28, 0.30, 0.32) * (0.92 + grit * 0.1), 0.22);
    } else if (panel < 1.5) {
        return vec4<f32>(vec3<f32>(0.82, 0.26, 0.04) * (0.94 + grit * 0.08), 0.38);
    } else if (panel < 2.5) {
        return vec4<f32>(vec3<f32>(0.74, 0.20, 0.03), 0.42);
    } else if (panel < 3.5) {
        return vec4<f32>(vec3<f32>(0.72, 0.19, 0.028), 0.42);
    } else if (panel < 4.5) {
        return vec4<f32>(vec3<f32>(0.55, 0.14, 0.02), 0.46);
    } else if (panel < 5.5) {
        return vec4<f32>(vec3<f32>(0.08, 0.07, 0.06), 0.62);
    } else if (panel < 9.5) {
        return vec4<f32>(vec3<f32>(0.70, 0.18, 0.03), 0.40);
    } else if (panel < 10.5) {
        // Springs and the engine. Bright enough to read as steel from above.
        return vec4<f32>(vec3<f32>(0.55, 0.56, 0.58) * (0.9 + grit * 0.12), 0.24);
    } else if (panel < 11.5) {
        return vec4<f32>(vec3<f32>(0.045, 0.04, 0.038), 0.8);
    } else if (panel < 16.5) {
        return vec4<f32>(1.0, 0.9, 0.55, 0.15);
    } else if (panel < 17.5) {
        return vec4<f32>(0.7, 0.02, 0.015, 0.2);
    } else if (panel < 40.5) {
        let g = noise(world.xz * 0.22);
        let dirt = mix(vec3<f32>(0.20, 0.15, 0.09), vec3<f32>(0.30, 0.22, 0.13), g);
        return vec4<f32>(dirt, 0.94);
    } else if (panel < 41.5) {
        let g = noise(world.xz * 0.07);
        return vec4<f32>(vec3<f32>(0.092, 0.048, 0.024) * (0.78 + g * 0.4), 0.97);
    } else if (panel < 42.5) {
        return vec4<f32>(vec3<f32>(0.07, 0.046, 0.028), 0.94);
    } else if (panel < 43.5) {
        return vec4<f32>(vec3<f32>(0.032, 0.048, 0.020), 0.92);
    } else if (panel < 44.5) {
        let g = noise(world.xz * 0.9 + vec2<f32>(world.y, 0.2));
        let h = clamp((world.y - 1.0) / 3.2, 0.0, 1.0);
        let leaf = mix(vec3<f32>(0.035, 0.10, 0.03), vec3<f32>(0.09, 0.18, 0.045), h);
        return vec4<f32>(leaf * (0.78 + g * 0.35), 0.88);
    } else if (panel < 45.5) {
        let g = noise(world.xz * 3.0 + vec2<f32>(world.y * 2.0, 0.0));
        return vec4<f32>(vec3<f32>(0.10, 0.058, 0.032) * (0.8 + g * 0.35), 0.92);
    } else if (panel < 46.5) {
        // Wet clod. Dark saturated mud. The orange frame stays on panel 1.
        let g = noise(world.xz * 3.2 + vec2<f32>(world.y * 2.2, 0.4));
        let col = mix(vec3<f32>(0.018, 0.008, 0.004), vec3<f32>(0.032, 0.014, 0.006), g);
        return vec4<f32>(col, 0.06);
    } else if (panel < 47.5) {
        let g = noise(world.xz * 2.2 + vec2<f32>(world.y * 1.4, 0.3));
        let col = mix(vec3<f32>(0.46, 0.38, 0.26), vec3<f32>(0.68, 0.58, 0.40), g);
        return vec4<f32>(col, 0.72);
    } else if (panel < 48.5) {
        return vec4<f32>(vec3<f32>(0.72, 0.80, 0.84), 0.18);
    }
    return vec4<f32>(vec3<f32>(0.16, 0.10, 0.06), 0.9);
}

struct Coat {
    albedo: vec3<f32>,
    rough: f32,
    nrm: vec3<f32>,
    mask: f32,
}

// Weight of the solver's mud panel `p` at this point of the Niva body.
// Chassis frame: +x right, +z nose, hubs near x = +-0.64, z = +-1.0.
fn niva_panel_mud(local: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    let fl = 1.0 - smoothstep(0.35, 0.75, length(local.xz - vec2<f32>(-0.64, 1.0)));
    let fr = 1.0 - smoothstep(0.35, 0.75, length(local.xz - vec2<f32>(0.64, 1.0)));
    let rl = 1.0 - smoothstep(0.35, 0.75, length(local.xz - vec2<f32>(-0.64, -1.0)));
    let rr = 1.0 - smoothstep(0.35, 0.75, length(local.xz - vec2<f32>(0.64, -1.0)));
    let left = smoothstep(0.45, 0.7, -local.x);
    let right = smoothstep(0.45, 0.7, local.x);
    let nose = smoothstep(1.35, 1.75, local.z);
    let rear = smoothstep(-1.45, -1.85, local.z);
    let belly = smoothstep(0.1, -0.6, n.y);
    var acc = model.mud6 * fl + model.mud7 * fr + model.mud8 * rl + model.mud9 * rr;
    acc += model.mud2 * left + model.mud3 * right + model.mud1 * nose + model.mud4 * rear;
    acc += model.mud5 * belly;
    var w = fl + fr + rl + rr + left + right + nose + rear + belly;
    // Anywhere else on the shell gets the cage's light dusting.
    acc += model.mud0 * 0.35;
    w += 0.35;
    return acc / w;
}

// Thrown mud on the Niva shell. Heavy and wet low on the body and in the
// arches, flecks higher up, dry dusty edges where the film is thin. Driven by
// the solver's panel mass and wetness; the pattern is model space so it rides
// with the car.
fn niva_body_mud(local: vec3<f32>, world: vec3<f32>, n: vec3<f32>) -> Coat {
    let m = niva_panel_mud(local, n);
    let mass = clamp(m.x, 0.0, 1.3);
    let wet = clamp(m.w, 0.0, 1.0);
    // Ground under the body from the four tire contacts: sampling the
    // height map here pushed fs_mesh past WebGL2's 16 texture units.
    var ground = 0.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        ground += (c.y - c.w) * 0.25;
    }
    let above = world.y - ground;
    // Spray reaches the sills first and the roof last.
    let reach = mix(0.7, 1.2, smoothstep(0.05, 0.9, mass));
    let low = 1.0 - smoothstep(reach * 0.45, reach, above);
    // Hood and roof catch flecks, not the sheet that the arches throw sideways.
    let top = smoothstep(0.45, 0.85, n.y);
    let amount = clamp(mass * 1.4, 0.0, 1.0) * mix(0.08, 1.0, low) * (1.0 - top * 0.85);
    let q = local * vec3<f32>(5.0, 7.0, 3.2);
    let blot = noise(q.xz + vec2<f32>(q.y, 0.0)) * 0.55
        + noise(q.zy * 2.1 + vec2<f32>(q.x, 3.0)) * 0.3
        + noise(q.xy * 4.7 + vec2<f32>(q.z, 7.0)) * 0.15;
    // Rearward smears: stretched along the car, short in height.
    let smear = noise(vec2<f32>(local.z * 1.6 + local.x * 0.3, local.y * 11.0));
    let field = max(blot, smear * 1.05 * low);
    let thr = 1.0 - amount * 0.7;
    var mask = smoothstep(thr, thr + 0.1, field);
    // Flecks thrown above the main coat.
    let fp = local * 42.0;
    let cell = floor(fp);
    let fleck_h = hash12(cell.xz + vec2<f32>(cell.y * 1.7, 0.0));
    let round = 1.0 - smoothstep(0.18, 0.42, length(fract(fp) - vec3<f32>(0.5)));
    let fleck = step(1.0 - clamp(mass, 0.0, 1.0) * mix(0.03, 0.2, low), fleck_h) * smoothstep(0.08, 0.3, mass) * round;
    mask = max(mask, fleck * 0.9);
    // Thick in the middle of a blot, thin at its edge. Thin film dries first.
    let thick = smoothstep(thr, thr + 0.35, field);
    let dry = clamp((1.0 - wet) + (1.0 - thick) * 0.45, 0.0, 1.0);
    let wet_col = vec3<f32>(0.13, 0.075, 0.036);
    let dry_col = vec3<f32>(0.42, 0.34, 0.25);
    var c: Coat;
    c.albedo = mix(wet_col, dry_col, smoothstep(0.35, 0.85, dry)) * (0.85 + 0.3 * noise(q.xz * 3.0));
    c.rough = mix(0.14, 0.92, smoothstep(0.3, 0.8, dry));
    let e = 0.05;
    let b0 = noise(q.xz * 2.0);
    let bx = noise((q.xz + vec2<f32>(e, 0.0)) * 2.0);
    let bz = noise((q.xz + vec2<f32>(0.0, e)) * 2.0);
    c.nrm = normalize(n + vec3<f32>(b0 - bx, 0.0, b0 - bz) * 2.2 * thick);
    c.mask = clamp(mask, 0.0, 1.0);
    return c;
}

// Spray in flight from the contract's spray slots (mesh.rs builds the strips).
// Panel 50 is a slurry ribbon, 51 a water drop, 52 a mud drop. The fraction
// of the panel is the distance along the strip, tail 0 to head 0.98; uv.x is
// metres along the piece from its launch, uv.y runs across 0..1.
// The sky on the water film over flying mud, at full radiance like the puddle.
fn wet_skin(n: vec3<f32>, view: vec3<f32>) -> vec3<f32> {
    let ndv = clamp(dot(n, view), 0.0, 1.0);
    // Schlick: the sky shows at a grazing angle, the bed when looking down.
    let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    return env_radiance(reflect(-view, n), 0.04) * fres * 0.6;
}

fn spray_strip(in: MeshOut) -> vec4<f32> {
    let kind = floor(in.panel + 0.01);
    let t = clamp(fract(in.panel + 0.001) / 0.98, 0.0, 1.0);
    let across = abs(in.uv.y - 0.5) * 2.0;
    let u = in.uv.x;
    let view = normalize(frame.cam_pos.xyz - in.world);
    var n = normalize(in.nrm);
    if (dot(n, view) < 0.0) {
        n = -n;
    }
    let sun = normalize(frame.sun_dir.xyz);
    if (kind > 52.5) {
        // A drop as a lit sphere: the quad faces the camera, the normal is
        // rebuilt from the disc, and the outline wobbles so it is not a dot.
        let q = in.uv * 2.0 - 1.0;
        let ang = atan2(q.y, q.x);
        let seed = fract(in.panel) * 40.0;
        let lump = noise(vec2<f32>(ang * 1.3 + seed, seed)) * 0.3 + noise(vec2<f32>(ang * 3.1, seed + 3.0)) * 0.14;
        let r2 = dot(q, q) / ((1.0 - lump) * (1.0 - lump));
        if (r2 > 1.0) {
            discard;
        }
        let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), view) + vec3<f32>(1e-4, 0.0, 0.0));
        let up = cross(view, right);
        let nb = normalize(right * q.x + up * q.y + view * sqrt(max(1.0 - r2, 0.0)));
        let soft = 1.0 - smoothstep(0.8, 1.0, r2);
        if (kind < 53.5) {
            let albedo = mix(vec3<f32>(0.2, 0.13, 0.07), vec3<f32>(0.3, 0.2, 0.11), noise(vec2<f32>(seed, 1.0)));
            var lit = shade_pbr(in.world, nb, albedo, 0.1, 0.0, 1.0, vec3<f32>(0.0));
            lit += wet_skin(nb, view);
            let back = pow(clamp(dot(-view, sun), 0.0, 1.0), 3.0);
            lit += frame.sun_color.rgb * vec3<f32>(0.36, 0.2, 0.08) * back * (1.0 - nb.y * 0.5) * 0.4;
            return vec4<f32>(tonemap(atmosphere(in.world, lit)), soft);
        }
        // Clear bead: the sky and the sun mirrored on a ball, a dim body.
        let ndv = clamp(dot(nb, view), 0.0, 1.0);
        let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
        let refl = env_radiance(reflect(-view, nb), 0.02) * 0.9;
        let body = vec3<f32>(0.22, 0.18, 0.13) * (sh_irradiance(nb) * frame.env.y + frame.sun_color.rgb * 0.3);
        let halfv = normalize(sun + view);
        let glint = frame.sun_color.rgb * min(ggx_d(clamp(dot(nb, halfv), 0.0, 1.0), 0.02) * 0.02, 30.0);
        let lit = mix(body, refl, fres) + glint;
        return vec4<f32>(tonemap(atmosphere(in.world, lit)), soft * clamp(0.45 + 0.55 * fres + glint.g * 0.1, 0.0, 1.0));
    }
    if (kind < 50.5) {
        // Torn slurry. Each edge tears on its own, the tail frays into
        // tongues, and a few holes open where the sheet thinned.
        let side = select(3.7, 11.3, in.uv.y > 0.5);
        let tear = noise(vec2<f32>(u * 34.0, side)) * 0.62 + noise(vec2<f32>(u * 95.0, side + 5.0)) * 0.38;
        let fray = 1.0 - smoothstep(0.0, 0.35, t);
        let tongue = noise(vec2<f32>(u * 12.0, in.uv.y * 6.0));
        let limit = 1.0 - 0.5 * tear - fray * (0.55 - 0.45 * tongue);
        if (across > limit) {
            discard;
        }
        let hole = noise(vec2<f32>(u * 26.0, in.uv.y * 4.0 + 2.0));
        if (hole > 0.80 - 0.1 * fray) {
            discard;
        }
        // Rounded cross-section and a wobbling skin, so the sheet is a
        // ribbon of liquid and not a flat card.
        // A rope of liquid: the normal wraps round the strip like a cylinder
        // out to the torn edge, so the sun draws a highlight along its length.
        let sx = clamp((in.uv.y - 0.5) * 2.0 / max(limit, 0.25), -0.97, 0.97);
        let bump = noise(vec2<f32>(u * 48.0, in.uv.y * 3.0)) - 0.5;
        let tside = normalize(cross(n, view) + vec3<f32>(1e-4, 0.0, 0.0));
        n = normalize(n * sqrt(1.0 - sx * sx) + tside * (sx + bump * 0.35));
        let g = noise(vec2<f32>(u * 9.0, in.uv.y * 2.0));
        // Liquid clay: brown, not tar. The gloss is the sky on a wet skin.
        let albedo = mix(vec3<f32>(0.2, 0.13, 0.07), vec3<f32>(0.3, 0.2, 0.11), g);
        var lit = shade_pbr(in.world, n, albedo, 0.1, 0.0, 1.0, vec3<f32>(0.0));
        lit += wet_skin(n, view);
        // Both faces see the sky; a strip is not a closed solid.
        lit += albedo * sh_irradiance(-n) * frame.env.y * 0.5;
        // Thin edges let the low sun through as a warm rim.
        let thin = smoothstep(0.4, 1.0, across / max(limit, 0.05));
        let back = pow(clamp(dot(-view, sun), 0.0, 1.0), 2.0);
        lit += frame.sun_color.rgb * vec3<f32>(0.42, 0.24, 0.10) * back * (0.25 + thin) * 0.6;
        let a = 0.96 * (1.0 - 0.45 * thin);
        return vec4<f32>(tonemap(lit), a);
    }
    // A drop smeared along its last frame of travel: a bright head, a fading tail.
    let r = across;
    let head = smoothstep(0.0, 0.6, t);
    let edge = 1.0 - smoothstep(0.55, 1.0, r);
    if (edge * head < 0.02) {
        discard;
    }
    let tside = normalize(cross(n, view) + vec3<f32>(1e-4, 0.0, 0.0));
    n = normalize(n + tside * (in.uv.y - 0.5) * 2.4);
    if (kind < 51.5) {
        // Murky water. Mostly reflection and a sun glint, little body.
        let ndv = clamp(dot(n, view), 0.0, 1.0);
        let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
        let refl = env_radiance(reflect(-view, n), 0.05) * frame.env.y;
        let body = vec3<f32>(0.20, 0.15, 0.10) * (sh_irradiance(n) * frame.env.y + frame.sun_color.rgb * 0.25);
        let halfv = normalize(sun + view);
        let glint = frame.sun_color.rgb * min(ggx_d(clamp(dot(n, halfv), 0.0, 1.0), 0.02) * 0.02, 30.0);
        let back = pow(clamp(dot(-view, sun), 0.0, 1.0), 6.0);
        var lit = mix(body, refl, fres) + glint + frame.sun_color.rgb * vec3<f32>(0.5, 0.42, 0.3) * back * 0.4;
        lit = atmosphere(in.world, lit);
        return vec4<f32>(tonemap(lit), clamp(edge * head * (0.55 + 0.4 * fres), 0.0, 1.0));
    }
    let albedo = vec3<f32>(0.075, 0.042, 0.022);
    let lit = shade_pbr(in.world, n, albedo, 0.2, 0.0, 1.0, vec3<f32>(0.0));
    return vec4<f32>(tonemap(lit), clamp(edge * head * 0.95, 0.0, 1.0));
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    // Planar reflection pass: what lies under the water level is not
    // mirrored (the oblique clip plane of three's Reflector).
    if (frame.debug.y > 0.5 && in.world.y < frame.debug.z) {
        discard;
    }
    let mesh_fw = length(fwidth(in.world.xz));
    if (in.panel > 49.9 && in.panel < 54.99) {
        let ss = spray_strip(in);
        return vec4<f32>(from_display(ss.rgb), ss.a);
    }
    let panel = i32(in.panel + 0.5);
    var paint = panel_paint(in.panel, in.uv, in.world);
    var albedo = paint.rgb;
    var rough = paint.a;
    var emit = vec3<f32>(0.0);
    var metal = 0.0;

    var nrm = in.nrm;
    var alpha = 1.0;
    // The floor outside the heightfield. Same gravel as the field, so the rim is not a wall.
    if (in.panel > 39.5 && in.panel < 40.5) {
        let xz = in.world.xz;
        // The sand field carries on past the heightfield as the same sand.
        let sk_layer = select(1, 6, sand_field());
        let uv = layer_uv(sk_layer, xz);
        let diff = textureSample(ground_diff, ground_samp, uv, sk_layer).rgb;
        let ntex = textureSample(ground_nor, ground_samp, uv, sk_layer).rgb;
        let pack = textureSample(ground_pack, ground_samp, uv, sk_layer);
        let ts = ntex * 2.0 - vec3<f32>(1.0);
        let sk_gain = select(0.85, 0.5, sand_field());
        let n = normalize(vec3<f32>(ts.x * sk_gain, 1.0, ts.y * sk_gain));
        let ao = clamp(pack.g, 0.0, 1.0);
        var skirt_albedo = diff * mix(1.0, ao, 0.75);
        var skirt_rough = clamp(pack.r * (1.0 - pack.a * 0.2), 0.2, 1.0);
        var skirt_n = n;
        if (sand_field()) {
            skirt_albedo *= vec3<f32>(0.5, 0.44, 0.34);
            let sv = sand_variety(xz, skirt_albedo, skirt_n, skirt_rough, 0.0, mesh_fw);
            skirt_albedo = sv.albedo;
            skirt_n = sv.n;
            skirt_rough = sv.rough;
        }
        let lit_skirt = shade(in.world, skirt_n, skirt_albedo, skirt_rough, vec3<f32>(0.0));
        return vec4<f32>(present(lit_skirt).rgb, 1.0);
    }
    if (in.panel > 79.5 && in.panel < 88.5) {
        let layer = i32(in.panel + 0.5) - 80;
        let dims = vec2<f32>(textureDimensions(niva_base));
        let dx = dpdx(in.uv) * dims.x;
        let dy = dpdy(in.uv) * dims.y;
        // Full mip chain (render.rs mip_count): last level = floor(log2(side)).
        // textureNumLevels has no GLSL ES 3.00 form and breaks WebGL2.
        let max_lod = floor(log2(max(dims.x, dims.y)));
        let lod = clamp(0.5 * log2(max(dot(dx, dx), dot(dy, dy))), 0.0, max_lod);
        let base = textureSampleLevel(niva_base, paint_samp, in.uv, layer, lod);
        let mr = textureSampleLevel(niva_mr, paint_samp, in.uv, layer, lod);
        let nm = textureSampleLevel(niva_nor, paint_samp, in.uv, layer, lod).rgb;
        if (layer == 7 && base.a < 0.35) {
            discard;
        }
        // WebP base color is sRGB. Decode here so the texels stay as authored.
        albedo = srgb_to_linear(base.rgb);
        rough = clamp(mr.g, 0.045, 1.0);
        metal = clamp(mr.b, 0.0, 1.0);
        nrm = tbn_normal(in.nrm, nm);
        if (layer == 3) {
            alpha = clamp(max(base.a, 0.22), 0.22, 0.62);
            albedo = mix(albedo, vec3<f32>(0.62, 0.78, 0.84), 0.35);
            rough = 0.05;
        } else if (model.extra.x > 1.5) {
            // The Niva carcass reads the patch. extra.y is the sidewall coat,
            // extra.z the groove fill. Body mud does not tint this tire.
            let void = in.carcass.x;
            let wall = in.carcass.y;
            let fill = clamp(model.extra.z, 0.0, 1.0);
            let side = clamp(model.extra.y, 0.0, 1.0);
            let grit = noise(in.uv * vec2<f32>(37.0, 11.0) + vec2<f32>(in.world.y, 0.4));
            let empty = void * (1.0 - smoothstep(0.22, 0.62, fill));
            albedo = albedo * mix(1.0, 0.22, empty);
            let packed = void * smoothstep(0.22, 0.62, fill);
            // Wet clay. A strong emit turned the coat into a flat orange shell.
            let clay = vec3<f32>(0.11, 0.038, 0.012) * (0.75 + 0.55 * grit);
            albedo = mix(albedo, clay, packed);
            rough = mix(rough, 0.16, packed);
            emit = vec3<f32>(0.045, 0.016, 0.005) * packed;
            let coat = wall * smoothstep(0.18, 0.55, side);
            albedo = mix(albedo, clay, coat * 0.92);
            rough = mix(rough, 0.20, coat);
            emit = emit + vec3<f32>(0.035, 0.012, 0.004) * coat;
            // Rims: an old wheel, not a new chrome one. Dull, grimy steel
            // with road dirt worked into it.
            let steel = smoothstep(0.4, 0.8, metal);
            let grime = noise(in.world.xz * 23.0 + vec2<f32>(in.world.y * 31.0, 1.7)) * 0.6
                + noise(in.uv * vec2<f32>(61.0, 17.0)) * 0.4;
            albedo = mix(albedo, albedo * vec3<f32>(0.78, 0.72, 0.62) * mix(0.8, 1.05, grime), steel);
            rough = mix(rough, max(rough, mix(0.42, 0.55, grime)), steel);
            // Mud thrown up on the lower half of the wheel: dense near the
            // ground, specks thinning toward the hub.
            for (var ti: i32 = 0; ti < 4; ti = ti + 1) {
                let tc = frame.tires[ti * 2];
                if (length(in.world.xz - tc.xz) < tc.w * 1.3) {
                    let low = smoothstep(tc.y + 0.02, tc.y - tc.w * 0.85, in.world.y);
                    let speck = noise(in.world.xz * 41.0 + vec2<f32>(in.world.y * 57.0, 3.0)) * 0.65
                        + noise(in.world.xz * 97.0 + vec2<f32>(in.world.y * 113.0, 9.0)) * 0.35;
                    let splash = smoothstep(0.62 - 0.4 * low, 0.75 - 0.4 * low, speck) * low * steel;
                    albedo = mix(albedo, vec3<f32>(0.2, 0.14, 0.085) * (0.8 + 0.4 * speck), splash);
                    rough = mix(rough, 0.85, splash);
                    metal = mix(metal, 0.0, splash);
                }
            }
        } else if (layer != 7) {
            let coat = niva_body_mud(in.local, in.world, nrm);
            albedo = mix(albedo, coat.albedo, coat.mask);
            rough = mix(rough, coat.rough, coat.mask);
            metal = mix(metal, 0.0, coat.mask);
            nrm = normalize(mix(nrm, coat.nrm, coat.mask));
        }
    } else if (in.panel > 69.5 && in.panel < 70.5) {
        let c = textureSampleLevel(prop_tex, paint_samp, in.uv, 0.0);
        if (c.a < 0.2) {
            discard;
        }
        albedo = c.rgb * 0.8;
        rough = 0.84;
    } else if (in.panel > 64.5 && in.panel < 65.5) {
        albedo = vec3<f32>(0.52, 0.77, 0.85);
        rough = 0.06;
        alpha = 0.45;
    } else if (in.panel > 59.5 && in.panel < 60.5) {
        albedo = textureSampleLevel(buggy_base, paint_samp, in.uv, 0.0).rgb;
        let mr = textureSampleLevel(buggy_mr, paint_samp, in.uv, 0.0);
        let nm = textureSampleLevel(buggy_nor, paint_samp, in.uv, 0.0).rgb;
        rough = clamp(mr.g, 0.04, 1.0);
        nrm = tbn_normal(in.nrm, nm);
        let cover = max(model.mud5.x, max(model.mud1.x, max(model.mud4.x, max(model.mud2.x, model.mud3.x))));
        let down = smoothstep(0.45, -0.2, nrm.y);
        let mask = smoothstep(0.04, 0.4, cover) * (0.25 + 0.75 * down);
        albedo = mix(albedo, vec3<f32>(0.05, 0.024, 0.012), mask * 0.88);
        rough = mix(rough, mix(0.35, 0.08, model.mud5.w), mask);
    } else if (in.panel > 60.5 && in.panel < 61.5) {
        albedo = textureSampleLevel(tire_base, paint_samp, in.uv, 0.0).rgb;
        let nm = textureSampleLevel(tire_nor, paint_samp, in.uv, 0.0).rgb;
        nrm = tbn_normal(in.nrm, nm);
        rough = 0.86;
        let void = tread_void(in.uv);
        let fill = clamp(model.extra.z, 0.0, 1.0);
        let smear = clamp(model.extra.w, 0.0, 1.0);
        let from_crown = abs(in.uv.y - 0.5);
        let on_tread = 1.0 - smoothstep(0.36, 0.47, from_crown);
        let grit = noise(in.uv * vec2<f32>(1.7, 9.0) + vec2<f32>(in.world.y, 0.3));
        // Rolling leaves the mud in the grooves. The lug faces stay rubber
        // until a spin smears them. Beads (v near 0 and 1) stay clean.
        let groove_mask = void * on_tread * smoothstep(0.28, 0.62, fill) * (0.90 + 0.10 * grit);
        let face_mask = (1.0 - void) * on_tread * smoothstep(0.35, 0.85, smear) * 0.35;
        let mask = clamp(max(groove_mask, face_mask), 0.0, 1.0);
        let packed = vec3<f32>(0.74, 0.32, 0.08);
        let film = vec3<f32>(0.11, 0.045, 0.016);
        albedo = mix(albedo, mix(film, packed, smoothstep(0.15, 0.7, void * fill)), mask);
        rough = mix(rough, mix(0.42, 0.08, groove_mask), mask);
        // The rear face of the carcass points away from the low sun.
        emit = vec3<f32>(0.42, 0.16, 0.04) * groove_mask;
    } else if (in.panel > 61.5 && in.panel < 62.5) {
        albedo = vec3<f32>(0.69, 0.73, 0.78);
        rough = 0.25;
    } else if (in.panel > 62.5 && in.panel < 63.5) {
        albedo = vec3<f32>(0.93, 0.46, 0.27);
        rough = 0.30;
    } else if (in.panel > 63.5 && in.panel < 64.5) {
        albedo = vec3<f32>(0.045, 0.050, 0.068);
        rough = 0.55;
    } else if (model.extra.x > 0.5) {
        let sidewall = select(0.0, 1.0, in.uv.y > 1.5);
        let mud = clamp(model.extra.y, 0.0, 1.2);
        let fill = max(clamp(model.extra.z, 0.0, 1.0), smoothstep(0.05, 0.45, mud));
        let smear = clamp(model.extra.w, 0.0, 1.0);
        let void = clamp(in.recess, 0.0, 1.0);
        let grit = noise(in.world.xy * 9.0 + vec2<f32>(in.world.z * 4.0, in.uv.x));
        let groove_mask = void * fill * (0.84 + 0.16 * grit) * (1.0 - sidewall);
        let face_mask = (1.0 - void) * smear * 0.7 * (1.0 - sidewall);
        let mask = clamp(max(groove_mask, face_mask), 0.0, 1.0);
        albedo = mix(vec3<f32>(0.02, 0.018, 0.016), vec3<f32>(0.20, 0.085, 0.028), mask);
        rough = mix(0.92, mix(0.36, 0.08, groove_mask), mask);
        nrm = mud_normal(in.nrm, in.world, mask);
    } else if (panel >= 0 && panel < 12) {
        let m = mud_of(panel);
        let cover = clamp(m.x, 0.0, 1.3);
        let drip = clamp(m.y, 0.0, 1.0);
        let streak = clamp(m.z, 0.0, 1.0);
        let wet = clamp(m.w, 0.0, 1.0);
        let p = in.world;
        // World-space clumps. A sine of uv reads as a decal stripe.
        let lump = noise(p.xz * 8.5 + vec2<f32>(p.y * 3.5, f32(panel))) * 0.6
            + noise(vec2<f32>(p.x * 3.0 + p.z, p.y * 11.0)) * 0.4;
        let coat = smoothstep(0.52 - cover * 0.7, 0.74 - cover * 0.28, lump);
        let grit = 0.78 + 0.22 * noise(p.xz * 22.0 + vec2<f32>(p.y * 9.0, 2.0));
        var mask = coat * grit * smoothstep(0.02, 0.22, cover);
        // A few thin gravity drips. Wide columns read as stripes, so these stay narrow.
        let down = smoothstep(1.6, 0.2, in.uv.y);
        let col_u = floor(in.uv.x * 2.4 + f32(panel) * 1.3);
        let pick = hash12(vec2<f32>(col_u, f32(panel) + 3.0));
        let column = smoothstep(0.84, 0.92, pick);
        let fx = fract(in.uv.x * 2.4 + f32(panel) * 1.3);
        let bead = smoothstep(0.10, 0.0, abs(fx - 0.5));
        let drip_mask = column * bead * smoothstep(0.06, 0.4, drip) * (0.2 + 0.8 * down);
        let smear = noise(vec2<f32>(p.x * 5.0 + p.z * 3.0, p.y * 0.45 + streak));
        let streak_mask = smoothstep(0.78, 0.92, smear) * streak * smoothstep(0.1, 0.45, cover) * down;
        mask = clamp(max(mask, max(drip_mask, streak_mask * 0.75)), 0.0, 1.0);
        let mud_col = mix(vec3<f32>(0.09, 0.048, 0.024), vec3<f32>(0.04, 0.02, 0.012), wet);
        albedo = mix(albedo, mud_col, mask);
        rough = mix(rough, mix(0.42, 0.07, wet), mask);
        nrm = mud_normal(in.nrm, p, mask);
    }

    if (in.panel > 15.5 && in.panel < 16.5) {
        emit = vec3<f32>(1.3, 1.05, 0.55);
        albedo = vec3<f32>(1.0, 0.95, 0.7);
    } else if (in.panel > 16.5 && in.panel < 17.5) {
        emit = vec3<f32>(0.9, 0.02, 0.01);
    }

    // Anything standing in the published film is soaked to just above the
    // waterline: tire rubber, rim and sills go dark and glossy there.
    // Mesh passes do not bind the contract group; the picture cache carries
    // the same solved water for them.
    if (sand_field()) {
        // WebGL2 caps fs_mesh at 16 textures, so the soak reads the
        // per-tire waterline the frame already carries (solved from the
        // contract on the CPU) instead of sampling the height/water maps.
        var wl = 0.0;
        var n_wet = 0.0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let c = frame.tires[i * 2];
            let tw_i = frame.tire_water[i];
            if (tw_i > -100.0 && tw_i > c.y - c.w + 0.003) {
                wl += tw_i;
                n_wet += 1.0;
            }
        }
        var wd = 0.0;
        if (n_wet > 0.5 && length(in.world.xz - frame.air[1].xz) < 2.5) {
            wl = wl / n_wet;
            wd = 0.05;
        }
        let tw = tire_waterline(in.world);
        if (tw.y > 0.5) {
            wl = tw.x;
            wd = max(wd, 0.02);
        }
        // Wet to 5-8 cm above the waterline, the line wandering a little.
        // The tread rides up out of the water wet, so the wet line climbs
        // along the circumference instead of cutting straight across.
        var tread = 0.0;
        for (var ti: i32 = 0; ti < 4; ti = ti + 1) {
            let tc = frame.tires[ti * 2];
            let tf = frame.tires[ti * 2 + 1];
            let tfw = normalize(vec2<f32>(tf.x, tf.y) + vec2<f32>(1e-5, 0.0));
            let tax = vec3<f32>(-tfw.y, 0.0, tfw.x);
            let rel_t = in.world - tc.xyz;
            let rad = length(rel_t - tax * dot(rel_t, tax));
            if (length(in.world.xz - tc.xz) < tc.w * 1.15) {
                tread = smoothstep(tc.w - 0.07, tc.w - 0.025, rad);
            }
        }
        let wet_top = wl + 0.02 + 0.004 * tread + 0.008 * noise(in.world.xz * 17.0 + vec2<f32>(in.world.y * 9.0, 0.0));
        let soaked = (1.0 - smoothstep(wet_top - 0.008, wet_top + 0.004, in.world.y)) * smoothstep(0.003, 0.01, wd);
        albedo = mix(albedo, albedo * vec3<f32>(0.3, 0.27, 0.23), soaked);
        rough = mix(rough, 0.28, soaked);
        // Beads of water on the rubber just above the wet band: tiny glossy spots.
        let bead_z = in.world.y - wet_top;
        let beads = smoothstep(0.82, 0.9, noise(in.world.xz * 90.0 + vec2<f32>(in.world.y * 110.0, 3.0)))
            * (1.0 - smoothstep(0.0, 0.06, bead_z)) * step(0.0, bead_z) * smoothstep(0.003, 0.01, wd);
        rough = mix(rough, 0.06, beads * 0.0);
        // A rim of silt dried on above the waterline, in patches.
        let splash = (1.0 - smoothstep(wl + 0.08, wl + 0.2, in.world.y)) * (1.0 - soaked)
            * smoothstep(0.4, 0.65, noise(in.world.xz * 9.0 + vec2<f32>(in.world.y * 14.0, 0.0)))
            * smoothstep(0.003, 0.01, wd);
        // Not on the rubber: a tan smear there read as the tire's foot
        // standing on the water.
        let on_tire = tire_waterline(in.world).y;
        albedo = mix(albedo, vec3<f32>(0.2, 0.15, 0.1), splash * 0.85 * (1.0 - on_tire));
        rough = mix(rough, 0.8, splash);
    }
    if (sand_field() && length(in.world.xz - frame.air[1].xz) < 3.2) {
        // Fill on the Niva: sunlit sand bounces warm light up into the
        // flanks and sills, and the open sky fills what the sun misses.
        let nn = normalize(nrm);
        let sun_d = normalize(frame.sun_dir.xyz);
        let ground = vec3<f32>(0.5, 0.4, 0.28) * frame.sun_color.rgb * max(sun_d.y, 0.0) * 1.25;
        let bounce = ground * clamp(0.5 - 0.5 * nn.y, 0.0, 1.0) * 0.55;
        let sky_fill = sh_irradiance(nn) * frame.env.y * 0.9;
        emit += albedo * (1.0 - metal) * (bounce + sky_fill);
    }
    var lit = shade_pbr(in.world, nrm, albedo, rough, metal, 1.0, emit);
    // Clods skip the blue sky fill. That fill is what turned wet mud into a gray stone.
    if (in.panel > 45.5 && in.panel < 46.5) {
        var n = normalize(nrm);
        // Soft skin. A strong bump turns the highlight into gravel.
        let e = 0.06;
        let h0 = noise(in.world.xz * 3.4 + vec2<f32>(in.world.y * 2.4, 0.2));
        let hx = noise((in.world.xz + vec2<f32>(e, 0.0)) * 3.4 + vec2<f32>(in.world.y * 2.4, 0.2));
        let hz = noise((in.world.xz + vec2<f32>(0.0, e)) * 3.4 + vec2<f32>(in.world.y * 2.4, 0.2));
        n = normalize(n + vec3<f32>(h0 - hx, 0.04, h0 - hz) * 0.28);
        let sun = normalize(frame.sun_dir.xyz);
        let view = normalize(frame.cam_pos.xyz - in.world);
        let ndl = clamp(dot(n, sun), 0.0, 1.0);
        let ndotv = clamp(dot(n, view), 0.0, 1.0);
        let shadow = shadow_factor(in.world + n * 0.02, n);
        // Darker than the berm. A crown in the soil's own brown reads as a dry stone.
        let tone = clamp((albedo.r - 0.016) / 0.018, 0.0, 1.0);
        let body = mix(vec3<f32>(0.0090, 0.0031, 0.0011), vec3<f32>(0.0155, 0.0052, 0.0018), tone);
        let wrap = ndl * 0.28 + 0.72;
        lit = body * (0.85 + wrap * shadow * 0.22);
        // The flat top has almost one normal from this camera, so a Phong lobe
        // paints the whole crown the color of a dry stone. Gloss is short streaks.
        let streak = noise(in.world.xz * 6.5 + vec2<f32>(in.world.y * 4.0, 0.4));
        let wetspot = smoothstep(0.80, 0.94, streak);
        let facing = pow(ndotv, 16.0);
        let fres = pow(1.0 - ndotv, 3.0);
        let gloss = vec3<f32>(0.22, 0.08, 0.026);
        let shine = 0.45 + shadow * 0.55;
        lit += gloss * facing * 0.65 * shine;
        lit += gloss * wetspot * clamp(n.y, 0.0, 1.0) * shine;
        lit += gloss * fres * 0.08;
        let belly = smoothstep(0.2, -0.45, n.y);
        lit = mix(lit, lit * vec3<f32>(0.42, 0.28, 0.18), belly * 0.75);
    }
    if (in.panel > 46.5 && in.panel < 47.5) {
        let d = length(in.uv - vec2<f32>(0.5, 0.5));
        // A gaussian, so overlapping marks are one cloud. A hard radial
        // falloff read as a disc.
        let soft = exp(-d * d * 7.0);
        if (soft < 0.05) {
            discard;
        }
        let g = noise(in.world.xz * 1.6 + vec2<f32>(in.world.y * 0.8, 0.3));
        let col = mix(vec3<f32>(0.20, 0.15, 0.09), vec3<f32>(0.32, 0.24, 0.15), g);
        let sun = normalize(frame.sun_dir.xyz);
        let ndl = clamp(dot(normalize(in.nrm), sun), 0.0, 1.0);
        let tail = smoothstep(47.08, 47.34, in.panel);
        let strength = mix(0.46, 0.10, tail);
        let mapped_dust = present(col * (0.72 + 0.28 * ndl));
        return vec4<f32>(mapped_dust.rgb, soft * strength);
    }
    if (in.panel > 47.5 && in.panel < 48.5) {
        let mapped_spray = present(vec3<f32>(0.72, 0.82, 0.88));
        return vec4<f32>(mapped_spray.rgb, 0.92);
    }
    if (in.panel > 48.5 && in.panel < 49.5 && sand_field()) {
        // Thrown wet sand on the dry field: damp sand, not a clay slab. The
        // rim feathers into the ground it lies on, broken by the grain.
        let xz = in.world.xz;
        let uv6 = layer_uv(6, xz);
        let diff = textureSample(ground_diff, ground_samp, uv6, 6).rgb;
        let pack6 = textureSample(ground_pack, ground_samp, uv6, 6);
        var sand = diff * mix(1.0, clamp(pack6.g, 0.0, 1.0), 0.75) * vec3<f32>(0.5, 0.44, 0.34);
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let sv = sand_variety(xz, sand, up, 0.9, 0.0, mesh_fw);
        let edge = min(min(in.uv.x, 1.0 - in.uv.x), min(in.uv.y, 1.0 - in.uv.y)) * 2.0;
        let rag = noise(xz * 9.0) * 0.6 + noise(xz * 23.0) * 0.4;
        let core = smoothstep(0.0, 0.9, edge * (0.6 + 0.8 * rag));
        let wet_lum = dot(sv.albedo, vec3<f32>(0.2126, 0.7152, 0.0722));
        let damp_col = max(mix(vec3<f32>(wet_lum), sv.albedo, 1.08), vec3<f32>(0.0)) * vec3<f32>(0.66, 0.6, 0.52);
        let alb = mix(sv.albedo, damp_col, core * 0.85);
        let rough_p = mix(sv.rough, 0.6, core);
        let lit_p = shade(in.world, sv.n, alb, rough_p, vec3<f32>(0.0));
        return vec4<f32>(present(lit_p).rgb, 1.0);
    }
    if (in.panel > 48.5 && in.panel < 49.5) {
        let g = noise(in.world.xz * 7.5 + vec2<f32>(in.world.y * 2.0, 0.2));
        let blot = noise(in.world.xz * 1.6 + vec2<f32>(2.0, 0.4));
        let clay0 = mix(vec3<f32>(0.09, 0.045, 0.02), vec3<f32>(0.18, 0.09, 0.035), g);
        let clay = mix(clay0, clay0 * 0.5, blot * blot);
        let face = smoothstep(0.25, 0.9, abs(in.nrm.y));
        let clay_s = mix(clay * 0.42, clay, face);
        let lit_rib = shade(in.world, in.nrm, clay_s, 0.88, vec3<f32>(0.0));
        return vec4<f32>(present(lit_rib).rgb, 1.0);
    }
    // Rubber in the water: below the solved waterline it sinks into the
    // silt within a few centimetres; at the line a bright meniscus and a
    // broken collar of dirty foam ring the tire.
    let tw = tire_waterline(in.world);
    if (sand_field() && tw.y > 0.5 && frame.debug.x < 0.5) {
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let sun_d = normalize(frame.sun_dir.xyz);
        let sky = sh_irradiance(up) * frame.env.y;
        let sun_l = frame.sun_color.rgb * max(sun_d.y, 0.0) * 0.45;
        let dz = in.world.y - tw.x;
        let silt_l = vec3<f32>(0.3, 0.225, 0.12) * (sun_l * 0.7 + sky * 1.1);
        // The water pass tints what it covers; here the rubber only loses the
        // sky it no longer sees, so it stays darker than the silt round it.
        lit = lit * mix(1.0, 0.68, 1.0 - exp(min(dz, 0.0) * 12.0));
        let foam_n = noise(in.world.xz * 14.0 + vec2<f32>(in.world.y * 60.0, 0.0)) * 0.6
            + noise(in.world.xz * 37.0 + vec2<f32>(in.world.y * 50.0, 0.0)) * 0.4;
        // Foam only in rare clumps on the lee side of the tire.
        let wv_m = frame.air[0].xz;
        let lee_d = select(vec2<f32>(0.8, 0.6), normalize(wv_m), dot(wv_m, wv_m) > 1e-4);
        var lee = 0.0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let c = frame.tires[i * 2];
            if (length(in.world.xz - c.xz) < c.w * 1.15) {
                lee = smoothstep(0.0, 0.25, dot(normalize(in.world.xz - c.xz + vec2<f32>(1e-4, 0.0)), lee_d));
            }
        }
        let collar = smoothstep(-0.004, 0.0, dz) * (1.0 - smoothstep(0.008, 0.02, dz))
            * smoothstep(0.62, 0.78, foam_n) * lee;
        let foam_l = vec3<f32>(0.52, 0.44, 0.33) * (sun_l * 1.4 + sky * 1.2);
        lit = mix(lit, foam_l * 0.8, collar * 0.0);
        // A meniscus 1-2 cm tall where the water climbs the rubber.
        let men = smoothstep(-0.002, 0.001, dz) * (1.0 - smoothstep(0.005, 0.01, dz));
        // The contact reads as a thin dark line: the meniscus is wet silt,
        // a shade darker than the water beside it.
        // A small meniscus: the water climbs a few mm up the rubber and
        // catches the sky right above the dark contact.
        // A crisp light waterline across the rubber: the meniscus catches
        // the sky in a band a few millimetres tall.
        // The drawn sheet rides a ripple and a low swell above the solved
        // level, so the band runs 2 cm up and the part left above the
        // actual cut shows as the line.
        // A dark wet meniscus climbing 1-2 cm up the rubber, its top edge
        // wandering with the ripples.
        let wav = (noise(in.world.xz * 26.0 + vec2<f32>(in.world.y * 30.0, frame.misc.x * 0.8)) - 0.5) * 0.012;
        // A thin light roll a few mm tall where the water climbs the rubber,
        // catching the sky; the soaked band above it stays dark.
        // A small meniscus: the water climbs a few mm and wets the rubber
        // darker, no light line.
        // A thin light line where the water climbs the rubber, catching the
        // sky; the soaked band above it is the dark wet rubber.
        // A soft meniscus 1-2 cm tall: brightest at the line, fading up the
        // rubber, its top wandering with the ripple.
        let men_w = smoothstep(-0.004, 0.0, dz) * (1.0 - smoothstep(0.0, 0.015 + wav * 0.4, dz));
        lit = mix(lit, sky * vec3<f32>(0.42, 0.41, 0.4) + sun_l * 0.08, men_w * 0.45);
    }
    // The rim's centre bore is open in the mesh and showed as a black hole:
    // close it with a dished steel cap, lit by sky and sun.
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let c = frame.tires[i * 2];
        let f = frame.tires[i * 2 + 1];
        if (f.w < 0.5) {
            continue;
        }
        let fw2 = normalize(vec2<f32>(f.x, f.y) + vec2<f32>(1e-5, 0.0));
        let axis = vec3<f32>(-fw2.y, 0.0, fw2.x);
        let rel = in.world - c.xyz;
        let ax = dot(rel, axis);
        let rad = length(rel - axis * ax);
        if (rad < c.w * 0.24 && abs(ax) < 0.16) {
            // A domed cap: the normal tips out to ~45 degrees at the rim, so it
            // mirrors sky above, horizon across the middle, sand below.
            let cap_n = normalize(axis * sign(dot(frame.cam_pos.xyz - c.xyz, axis)) + (rel - axis * ax) / (c.w * 0.24) * 0.55);
            let sun_d = normalize(frame.sun_dir.xyz);
            let cap_v = normalize(frame.cam_pos.xyz - in.world);
            let cap_l = frame.sun_color.rgb * max(dot(cap_n, sun_d), 0.0) * 0.45 + sh_irradiance(cap_n) * frame.env.y;
            let ring = smoothstep(c.w * 0.17, c.w * 0.22, rad);
            // Brushed steel: mostly the environment it mirrors (the same HDRI
            // as the rim), a little diffuse, a tight sun lobe.
            let cap_r = reflect(-cap_v, cap_n);
            let cap_f0 = vec3<f32>(0.5, 0.5, 0.5);
            let cap_nv = clamp(dot(cap_n, cap_v), 0.0, 1.0);
            let cap_f = cap_f0 + (vec3<f32>(1.0) - cap_f0) * pow(1.0 - cap_nv, 5.0);
            // Satin steel: a blurred mirror of the IBL and a broad sun lobe.
            // Satin steel, roughness ~0.35: a blurred IBL and a broad sun lobe.
            // Roughness ~0.5: mostly the irradiance, a faint blurred sky.
            let cap_env = mix(env_radiance(cap_r, 0.8), sh_irradiance(cap_r) * 0.45, 0.85);
            let cap_spec = cap_env * frame.env.y * 1.3 * cap_f
                + frame.sun_color.rgb * pow(max(dot(cap_r, sun_d), 0.0), 8.0) * 0.06;
            // The body and the arch hide the upper sky from the lower half.
            let cap_occ = mix(0.55, 1.0, smoothstep(-0.4, 0.3, cap_r.y));
            var cap_lit = (vec3<f32>(0.1) * cap_l + cap_spec * cap_occ) * mix(0.7, 1.05, ring);
            // Dried mud film: patchy, thicker toward the bottom of the cap.
            let rel_w = rel - axis * ax;
            let dirt_n = noise(vec2<f32>(dot(rel_w, vec3<f32>(fw2.x, 0.0, fw2.y)), rel_w.y) * 60.0 + vec2<f32>(f32(i) * 5.0, 1.0));
            let low_c = smoothstep(c.y + 0.02, c.y - c.w * 0.85, in.world.y);
            let dirt = clamp(smoothstep(0.62 - 0.4 * low_c, 0.75 - 0.4 * low_c, dirt_n) * mix(0.4, 1.0, low_c) + low_c * 0.25, 0.0, 0.9);
            let mud_c = vec3<f32>(0.32, 0.24, 0.15) * cap_l * 0.9;
            lit = mix(cap_lit, mud_c, dirt);
        }
    }
    var mapped = present(lit);
    if (frame.debug.x > 0.5 && frame.debug.x < 2.5 && tw.y > 0.5) {
        // Depth debug on the rubber: magenta at the carcass bottom, cyan at
        // the waterline, a white tick every 2 cm between.
        let bottom = tw.z;
        let above_b = in.world.y - bottom;
        let tick = 1.0 - smoothstep(0.0015, 0.003, abs(fract(above_b / 0.02 + 0.5) - 0.5) * 0.02);
        var c = mapped.rgb;
        c = mix(c, vec3<f32>(1.0), tick * step(above_b, tw.x - bottom) * 0.8);
        c = mix(c, vec3<f32>(1.0, 0.0, 1.0), 1.0 - smoothstep(0.004, 0.008, abs(above_b)));
        c = mix(c, vec3<f32>(0.0, 1.0, 1.0), 1.0 - smoothstep(0.003, 0.006, abs(in.world.y - tw.x)));
        mapped = vec4<f32>(c, mapped.a);
    }
    return vec4<f32>(mapped.rgb, alpha);
}

struct PartIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
}

struct PartOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_particle(in: PartIn) -> PartOut {
    var o: PartOut;
    o.clip = frame.view_proj * vec4<f32>(in.pos, 1.0);
    o.color = in.color;
    o.uv = in.uv;
    return o;
}

@fragment
fn fs_particle(in: PartOut) -> @location(0) vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    // Broken clod edge. A round falloff stacked into an orange coin.
    let wobble = 0.10 * sin(in.uv.x * 11.0 + in.uv.y * 7.0)
        + 0.06 * sin(in.uv.x * 23.0 - in.uv.y * 17.0);
    if (d > 0.40 + wobble) {
        discard;
    }
    let fall = smoothstep(0.40, 0.12, d);
    // Color is already the dirty brown we want on screen. A gamma lift turned spray white.
    return vec4<f32>(from_display(in.color.rgb * mix(0.65, 1.0, fall)), in.color.a * fall);
}
