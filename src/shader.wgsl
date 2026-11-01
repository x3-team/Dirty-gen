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
    reflect_view_proj: mat4x4<f32>,
    reflect_flags: vec4<f32>,
    // x: water detail (0 low, 1 medium, 2 high). y: 1 with a fluid sheet.
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

fn present(c: vec3<f32>) -> vec4<f32> {
    let mapped = vec3<f32>(1.0) - exp(-max(c, vec3<f32>(0.0)) * 1.08);
    return vec4<f32>(pow(mapped, vec3<f32>(1.0 / 2.2)), 1.0);
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

fn shadow_factor(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let lp = frame.light_vp * vec4<f32>(world, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let dims = vec2<f32>(textureDimensions(shadow_tex));
    let base = vec2<i32>(uv * dims);
    // Flat ground can shadow a few centimetres of rut. A wall keeps a larger bias.
    let slope = clamp(1.0 - n.y, 0.0, 1.0);
    let bias = 0.00015 + slope * 0.0014;
    var blocked = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let c = clamp(base + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
            let s = decode_depth(textureLoad(shadow_tex, c, 0));
            if (ndc.z > s + bias) {
                blocked = blocked + 1.0;
            }
        }
    }
    // 0.92 so the airborne buggy and the clods throw a shadow you can see.
    return 1.0 - blocked / 9.0 * 0.92;
}

fn shade(world: vec3<f32>, n_in: vec3<f32>, albedo: vec3<f32>, rough: f32, emit: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let hemi = mix(frame.amb.rgb, frame.sky_zenith.rgb * 0.65, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let shadow = shadow_factor(world + n * 0.012, n);
    let view = normalize(frame.cam_pos.xyz - world);
    let halfv = normalize(sun + view);
    // Specular is reserved for smooth wet surfaces. Dry soil stays matte.
    let gloss = smoothstep(0.22, 0.82, 1.0 - rough);
    let spec_pow = mix(12.0, 160.0, gloss);
    let spec = pow(clamp(dot(n, halfv), 0.0, 1.0), spec_pow) * gloss * shadow;
    var col = albedo * (hemi * 0.62 + frame.sun_color.rgb * ndl * shadow * 1.25);
    col += frame.sun_color.rgb * spec * 0.7;
    col += emit;
    let dist = length(world - frame.cam_pos.xyz);
    let fog = smoothstep(58.0, 150.0, dist);
    let fog_col = frame.sky_horizon.rgb * 0.28 + frame.sky_zenith.rgb * 0.12;
    col = mix(col, fog_col, fog * 0.65);
    return col;
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

// 0 mud_forest, 1 brown_mud_dry, 2 clay, 3 grass, 4 gravel.
// Sand (10) and silt (11) start from the dry map and get their own tint.
fn layer_of(kind: u32) -> i32 {
    if (kind == 1u) { return 3; }
    if (kind == 2u) { return 4; }
    if (kind == 9u) { return 2; }
    if (kind == 0u || kind == 3u || kind == 7u || kind == 10u || kind == 11u) { return 1; }
    return 0;
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
    // The rut keeps the kind the solver wrote. A wet-mud pad on the cover
    // was a flat card in the groove behind the tire.
    let kind = kind_nearest(xz);

    let mat_dims = vec2<f32>(textureDimensions(mat_tex));
    let mat = textureSampleLevel(mat_tex, mat_samp, (in.ij + vec2<f32>(0.5)) / mat_dims, 0.0);
    let wet = mat.r;
    let demo_seam_terrain = win_demo_flat_terrain() && distance(xz, beauty_seam_center()) < 3.35;
    let horiz_terrain = n.y > 0.58;
    let water = select(
        mat.g * 0.34,
        water_column(xz),
        puddle_beauty_on() && (!demo_seam_terrain || horiz_terrain),
    );
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
    let thickness = select(crust_rg.x * 0.04, 0.0, demo_seam_terrain);

    // Snow is its own layer. sun_dir.w selects it. Compaction is mat.b.
    // The ground albedo stays off so the blanket, the rut, and the berm are the field.
    if (frame.sun_dir.w > 0.5) {
        let fresh = vec3<f32>(0.79, 0.83, 0.87);
        let packed = vec3<f32>(0.47, 0.53, 0.60);
        var albedo = mix(fresh, packed, darken);
        let grain = noise(xz * 6.5);
        albedo *= mix(0.93, 1.05, grain);
        let rough_out = mix(0.96, 0.86, darken);
        let lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
        return present(lit);
    }

    let uv0 = xz / 3.2;
    let uv1 = xz / 2.7;
    let uv2 = xz / 3.5;
    let uv3 = xz / 2.3;
    let uv4 = xz / 3.3;
    // Packed clay is the clay map, one salmon tone. In the rut that sheet
    // was a flat card. Draw it as the mud around it.
    var layer = layer_of(kind);
    if (kind == 9u) {
        layer = 0;
    }
    let diff = pick5(
        layer,
        textureSample(ground_diff, ground_samp, uv0, 0),
        textureSample(ground_diff, ground_samp, uv1, 1),
        textureSample(ground_diff, ground_samp, uv2, 2),
        textureSample(ground_diff, ground_samp, uv3, 3),
        textureSample(ground_diff, ground_samp, uv4, 4),
    ).rgb;
    let ntex = pick5(
        layer,
        textureSample(ground_nor, ground_samp, uv0, 0),
        textureSample(ground_nor, ground_samp, uv1, 1),
        textureSample(ground_nor, ground_samp, uv2, 2),
        textureSample(ground_nor, ground_samp, uv3, 3),
        textureSample(ground_nor, ground_samp, uv4, 4),
    ).rgb;
    let pack = pick5(
        layer,
        textureSample(ground_pack, ground_samp, uv0, 0),
        textureSample(ground_pack, ground_samp, uv1, 1),
        textureSample(ground_pack, ground_samp, uv2, 2),
        textureSample(ground_pack, ground_samp, uv3, 3),
        textureSample(ground_pack, ground_samp, uv4, 4),
    );
    let ao = clamp(pack.g, 0.0, 1.0);
    let rough_map = clamp(pack.r * (1.0 - pack.a * 0.2), 0.05, 1.0);
    var tng = vec3<f32>(1.0, 0.0, 0.0);
    tng = normalize(tng - n * dot(n, tng));
    let btg = cross(tng, n);
    var ts = ntex * 2.0 - vec3<f32>(1.0);
    // Dry crust and gravel stay flatter. The torn lip keeps the mud normal.
    var n_gain = 1.0;
    if (track_near(xz)) { n_gain = 0.7; }
    else if (kind == 2u || kind == 3u || kind == 10u || kind == 11u) { n_gain = 0.28; }
    else if (kind == 6u) { n_gain = 1.7; }
    else if (kind == 8u) { n_gain = 1.35; }
    else if (kind == 5u) { n_gain = 1.35; }
    else if (kind == 9u) { n_gain = 1.7; }
    ts = vec3<f32>(ts.xy * n_gain, ts.z);
    n = normalize(tng * ts.x + btg * ts.y + n * ts.z);
    var albedo = diff * mix(1.0, ao, 0.75);
    let muddy = kind == 4u || kind == 5u || kind == 9u || kind == 6u;
    let wet_amt = select(0.12, 0.32, muddy);
    albedo = mix(albedo, albedo * vec3<f32>(0.66, 0.58, 0.48), smoothstep(0.18, 0.92, wet) * wet_amt * (1.0 - pool_interior * 0.92));
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
    } else if (kind == 10u) {
        // Sand stays a warm grain. Moisture darkens it a little. It does not turn into clay.
        albedo = albedo * vec3<f32>(1.28, 1.08, 0.7);
        albedo = mix(albedo, albedo * vec3<f32>(0.82, 0.74, 0.58), smoothstep(0.15, 0.55, wet) * 0.55);
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
        let dry_tex = textureSample(ground_diff, ground_samp, uv1, 1).rgb;
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
    let in_cut = dug > 0.012 && water > 0.008;
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
    if (track_near(xz)) {
        // A flat highlight on the print was a white sheet over the lugs.
        rough_out = max(rough_out, 0.92);
    }
    // Fissure colours come from the crust, not from the darkened lug groove.
    let skin_albedo = albedo;
    if (lug_h < -0.003) {
        let k = clamp(-lug_h / 0.045, 0.0, 1.0);
        albedo *= mix(1.0, 0.52, k);
    }
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
    var lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
    // The sun sits low, so a tight Phong highlight misses this camera.
    // A wide lobe on the shore is the wet sheen. Inside the cut it stays small
    // so the bed texture is what you see.
    let view = normalize(frame.cam_pos.xyz - in.world);
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let halfv = normalize(normalize(frame.sun_dir.xyz) + view);
    let facing = pow(1.0 - ndotv, 2.0);
    let wide = pow(clamp(dot(n, halfv), 0.0, 1.0), 2.2);
    var wet_spec = (facing * 0.75 + wide * 1.15) * shore * mix(1.0, 0.16, drowned);
    if (in_cut) {
        wet_spec *= mix(0.28, 0.05, depth_t);
    }
    lit += vec3<f32>(0.22, 0.095, 0.032) * wet_spec;
    return present(lit);
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
    return textureSampleLevel(pool_reflect_tex, scene_samp, uv, 0.0).rgb;
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
            let s = textureSampleLevel(scene_color, scene_samp, uv, 0.0).rgb;
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
    return textureSampleLevel(scene_color, scene_samp, hit_uv, 0.0).rgb;
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
        let samp = textureSampleLevel(scene_color, scene_samp, veh_uv, 0.0).rgb;
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
        let samp = textureSampleLevel(scene_color, scene_samp, uv, 0.0).rgb;
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
    let mapped = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * 1.08);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, a_out);
}

fn water_vertex(in: TerrIn) -> WaterOut {
    if (open_water_on()) {
        return open_water_vertex(in);
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
            return vec4<f32>(pow(max(planar, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2)), 0.96);
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
        let mapped = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * 1.05);
        let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
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
    let mapped = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * 1.05);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
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
                let sky_samp = textureSampleLevel(scene_color, scene_samp, sky_uv, 0.0).rgb;
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
    let mapped = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * 1.08);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
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
    return o;
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) depth: f32,
}

fn shadow_terrain_vertex(in: TerrIn) -> ShadowOut {
    let base = height_bilinear(in.xz);
    let h = select(apply_seat(in.xz, base).x, base, skip_tire_seat());
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

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    let panel = i32(in.panel + 0.5);
    var paint = panel_paint(in.panel, in.uv, in.world);
    var albedo = paint.rgb;
    var rough = paint.a;
    var emit = vec3<f32>(0.0);

    var nrm = in.nrm;
    var alpha = 1.0;
    // The floor outside the heightfield. Same gravel as the field, so the rim is not a wall.
    if (in.panel > 39.5 && in.panel < 40.5) {
        let xz = in.world.xz;
        let uv = xz / 2.7;
        let diff = textureSample(ground_diff, ground_samp, uv, 1).rgb;
        let ntex = textureSample(ground_nor, ground_samp, uv, 1).rgb;
        let pack = textureSample(ground_pack, ground_samp, uv, 1);
        let ts = ntex * 2.0 - vec3<f32>(1.0);
        let n = normalize(vec3<f32>(ts.x * 0.85, 1.0, ts.y * 0.85));
        let ao = clamp(pack.g, 0.0, 1.0);
        let skirt_albedo = diff * mix(1.0, ao, 0.75);
        let skirt_rough = clamp(pack.r * (1.0 - pack.a * 0.2), 0.2, 1.0);
        let lit_skirt = shade(in.world, n, skirt_albedo, skirt_rough, vec3<f32>(0.0));
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
        } else if (layer != 7) {
            let mud = clamp(model.extra.y, 0.0, 1.2);
            let cover = max(mud, max(model.mud5.x, max(model.mud1.x, max(model.mud4.x, max(model.mud2.x, model.mud3.x)))));
            let roof = smoothstep(0.55, 0.92, nrm.y);
            let clay = vec3<f32>(0.32, 0.12, 0.04);
            let coat = cover * (1.0 - roof);
            albedo = mix(albedo, clay, coat * 0.94);
            rough = mix(rough, 0.18, coat);
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

    var lit = shade(in.world, nrm, albedo, rough, emit);
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
    let mapped = present(lit);
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
    return vec4<f32>(in.color.rgb * mix(0.65, 1.0, fall), in.color.a * fall);
}
