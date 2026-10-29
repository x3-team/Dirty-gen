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
    // Irradiance / pi of the env map as nine SH terms (see env.rs).
    sh: array<vec4<f32>, 9>,
    env: vec4<f32>,    // env mips, IBL gain, sky is an HDRI, exposure
    // Niva headlamps: left xyz + intensity, right xyz, beam dir + outer cone cos.
    lamps: array<vec4<f32>, 3>,
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

@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var mat_tex: texture_2d<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var shadow_tex: texture_2d<f32>;

@group(1) @binding(4) var ground_diff: texture_2d_array<f32>;
@group(1) @binding(5) var ground_nor: texture_2d_array<f32>;
@group(1) @binding(6) var ground_pack: texture_2d_array<f32>;
@group(1) @binding(7) var ground_samp: sampler;
@group(1) @binding(8) var imprint_tex: texture_2d<f32>;

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

// Filmic curve (Narkowicz ACES fit). The old 1 - exp(-x) flattened every
// highlight into the same pale tan, so wet gloss never read as gloss.
fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0)) * frame.env.w * 0.78;
    let m = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return pow(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn present(c: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(tonemap(c), 1.0);
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
    var col = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, smoothstep(-0.02, 0.55, h));
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
    let rough = clamp(rough_in, 0.04, 1.0);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let ndv = max(dot(n, view), 1e-3);
    let halfv = normalize(sun + view);
    let ndh = clamp(dot(n, halfv), 0.0, 1.0);
    let vdh = clamp(dot(view, halfv), 0.0, 1.0);
    let shadow = shadow_factor(world + n * 0.012, n);
    let f0 = mix(vec3<f32>(0.04), albedo, metal);
    let diff_col = albedo * (1.0 - metal);
    let a = rough * rough;
    let k = (rough + 1.0) * (rough + 1.0) / 8.0;
    let g = ndl / (ndl * (1.0 - k) + k) * ndv / (ndv * (1.0 - k) + k);
    let fs = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - vdh, 5.0);
    let spec = ggx_d(ndh, a) * g * fs / max(4.0 * ndl * ndv, 1e-3);
    let sun_in = frame.sun_color.rgb * ndl * shadow * 1.25;
    var col = diff_col * sun_in + spec * sun_in * 3.14159265;

    let gain = frame.env.y;
    let refl = reflect(-view, n);
    // A reflection that dips under the surface sees soil, not sky.
    let horizon = clamp(1.0 + dot(refl, n), 0.0, 1.0);
    // Rough soil shadows its own grains, so a grazing sky sheen stays faint.
    let spec_occ = ao * horizon * horizon * mix(1.0, 0.3, smoothstep(0.35, 0.9, rough));
    let env_spec = env_radiance(refl, rough) * env_brdf(f0, rough, ndv) * spec_occ;
    // Ground in shadow still sees the sky, but less of the sun-lit half of it.
    let sky_occ = mix(0.72, 1.0, shadow);
    col += (diff_col * sh_irradiance(n) * ao + env_spec) * gain * sky_occ;
    col += lamp_light(world, n, view, diff_col, f0, rough);
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
    if (frame.env.z > 0.5) {
        // HDRI sky. The disc is the scene sun, drawn where the light comes from.
        let d = normalize(dir);
        let sun = normalize(frame.sun_dir.xyz);
        let disc = pow(max(dot(d, sun), 0.0), 900.0) * 4.0;
        return present(env_radiance(d, 0.0) + frame.sun_color.rgb * disc);
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

fn cell_center(ij: vec2<i32>) -> vec2<f32> {
    let cell = max(frame.params.z, 1e-4);
    let half = cell * frame.params.w * 0.5;
    return vec2<f32>((f32(ij.x) + 0.5) * cell - half, (f32(ij.y) + 0.5) * cell - half);
}

// One pressed track. A shallow ribbon ties the lugs into the rut floor,
// and each lug is a deeper void in that ribbon. Stored amplitude is unchanged.
fn tread_delta(xz: vec2<f32>) -> f32 {
    let ij = vec2<i32>(round(world_to_ij(xz)));
    let rec = imprint_at(ij);
    if (rec.a < 0.5) {
        return 0.0;
    }
    let amp = rec.r * 0.06;
    let ang = rec.g * 6.2831853;
    let fwd = vec2<f32>(sin(ang), cos(ang));
    let right = vec2<f32>(fwd.y, -fwd.x);
    let rel = xz - cell_center(ij);
    let lat = dot(rel, right) + rec.b * 0.8 - 0.4;
    let along = dot(xz, fwd);
    let pitch = 0.19897;
    let u = fract(along / pitch);
    let row = select(0.0, 0.5, lat > 0.0);
    let lug = abs(fract(u + row) - 0.5);
    // Visual crown, not the wider physics lane. The edge fades into the rut.
    // One groove the width of the crown. The lugs bite deeper into that floor
    // so the print stays a track when the tire has moved on.
    // The void is the lug. A deep flat floor was a sheet lying on the print.
    let ribbon = 1.0 - smoothstep(0.09, 0.15, abs(lat));
    let block = 1.0 - smoothstep(0.12, 0.34, lug);
    let floor = ribbon * 0.12;
    return -amp * max(floor, block * ribbon);
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
    // Hide the stamp and the berms beside the rubber. The rut further
    // behind stays, so the pressed track is still there after the pass.
    let ahead = 1.0 - smoothstep(0.22, 0.72, along);
    let not_behind = 1.0 - smoothstep(0.10, 0.55, -along);
    let across = 1.0 - smoothstep(0.12, 0.55, lat);
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
fn layer_of(kind: u32) -> i32 {
    if (kind == 1u) { return 3; }
    if (kind == 2u) { return 4; }
    if (kind == 9u) { return 2; }
    if (kind == 4u) { return 5; }
    if (kind == 10u) { return 6; }
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

fn height_bilinear(xz: vec2<f32>) -> f32 {
    let ij = world_to_ij(xz);
    let i = floor(ij);
    let f = fract(ij);
    let i0 = vec2<i32>(i);
    let h00 = height_at(i0);
    let h10 = height_at(i0 + vec2<i32>(1, 0));
    let h01 = height_at(i0 + vec2<i32>(0, 1));
    let h11 = height_at(i0 + vec2<i32>(1, 1));
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
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

// The sill sits between the hubs. Per-tire closure never reached it, so the
// cut and the berms stayed as blocks and a hole under the body.
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
    let dx = max(minx - 1.35 - xz.x, xz.x - (maxx + 1.35));
    let dz = max(minz - 0.88 - xz.y, xz.y - (maxz + 0.88));
    return 1.0 - smoothstep(0.0, 0.22, max(dx, dz));
}

// Put the pad back under the carcass. The stamp stays in the field.
// x = drawn height, y = bowl weight, z = how fully this column is covered.
fn apply_seat(xz: vec2<f32>, base_h: f32) -> vec3<f32> {
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
    let torn = lip_near(xz) && !track_near(xz);
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
    if (live > 0.5) {
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
    if (track_near(xz)) {
        return height_bilinear(xz);
    }
    // The lip is a step. Everywhere else stays bilinear so a berm is not a stair.
    if (lip_near(xz)) {
        return height_sharp(xz);
    }
    return height_bilinear(xz);
}

@vertex
fn vs_terrain(in: TerrIn) -> TerrOut {
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
    let xz = in.world.xz;
    var n = terrain_normal(xz);
    var kind = kind_nearest(xz);

    let mat_dims = vec2<f32>(textureDimensions(mat_tex));
    let mat = textureSampleLevel(mat_tex, mat_samp, (in.ij + vec2<f32>(0.5)) / mat_dims, 0.0);
    // Pore moisture and the film are the contract's inside its window.
    let wet = bc_moisture(xz, mat.r);
    let film_c = bc_film(xz);
    let water = mix(mat.g * 0.34, film_c.r + film_c.g, bc_inside(xz));
    let darken = mat.b;

    let uv0 = xz / 3.2;
    let layer = layer_of(kind);
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
    albedo = mix(albedo, albedo * vec3<f32>(0.66, 0.58, 0.48), smoothstep(0.18, 0.92, wet) * wet_amt);
    if (kind == 9u) {
        albedo *= mix(1.0, 0.8, smoothstep(0.35, 0.9, darken));
        // The clay color is almost one tone. In the cut, the mud under it shows.
        // The printed rut stays the soil it was pressed into. The clay
        // multiply was a pale sheet over the lugs.
        if (-stamp_bilinear(xz) > 0.012 && !track_near(xz)) {
            let mud = textureSampleLevel(ground_diff, ground_samp, uv0, 0, 0.0).rgb;
            albedo = mud * vec3<f32>(1.08, 0.78, 0.52);
        }
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
        // The beauty sand map is already a pale grain; only warm it.
        albedo = albedo * vec3<f32>(0.5, 0.44, 0.34);
        albedo = mix(albedo, albedo * vec3<f32>(0.82, 0.74, 0.58), smoothstep(0.15, 0.55, wet) * 0.55);
    } else if (kind == 11u) {
        // Silt is a gray flour. Wet silt goes dark. Dry silt stays lighter than clay.
        albedo = albedo * vec3<f32>(0.98, 0.94, 0.82);
        albedo = mix(albedo, albedo * vec3<f32>(0.52, 0.48, 0.42), smoothstep(0.2, 0.85, wet));
    }
    // The exit print. A fresh field sits under this ramp, so the lane goes dark
    // without a new surface kind.
    albedo *= mix(1.0, 0.36, smoothstep(0.20, 0.52, darken));

    let shore = smoothstep(0.0, 0.02, water) * (1.0 - smoothstep(0.035, 0.11, water));
    let drowned = smoothstep(0.012, 0.05, water);
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
    if (lug_h < -0.003) {
        let k = clamp(-lug_h / 0.045, 0.0, 1.0);
        albedo *= mix(1.0, 0.52, k);
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
    let kind_crust = kind == 3u || ((kind == 9u || kind == 11u || kind == 4u) && dry > 0.6);
    if (kind_crust || opening > 0.0005) {
        // Baked crust: plates curl up, cracks go dark and matte.
        let scale = select(2.2, 3.4, kind == 3u);
        let edge = crack_edge(xz * scale);
        // Crack half-width in cell units: a hairline, or the published gap.
        let width = max(select(0.0, select(0.022, 0.05, kind == 3u), kind_crust), opening * scale * 0.5);
        let crack = (1.0 - smoothstep(width * 0.4, width * 1.6 + 0.01, edge))
            * select(max(dry * 0.75, smoothstep(0.0005, 0.004, opening)), 1.0, kind == 3u);
        albedo = mix(albedo, albedo * 0.38, crack);
        let plate = smoothstep(0.05, 0.4, edge);
        albedo = mix(albedo, albedo * 1.08, plate * select(dry * 0.5, 1.0, kind == 3u) * 0.5);
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
    let damp = smoothstep(0.3, 0.95, wet);
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
    albedo = mix(albedo, albedo * vec3<f32>(0.52, 0.47, 0.42), damp * porosity * 0.6);
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
    let edge_c = bc_wet_edge(xz);
    if (edge_c > 0.0) {
        let wob = noise(xz * 5.0) * 0.7 + noise(xz * 17.0) * 0.3;
        let band = smoothstep(0.2, 0.65, edge_c * (0.55 + 0.9 * wob));
        let dry_side = 1.0 - smoothstep(0.35, 0.55, wet);
        albedo = mix(albedo, albedo * vec3<f32>(0.72, 0.68, 0.64), band * dry_side);
        rough_out = mix(rough_out, rough_out * 0.8, band * dry_side);
        let tide = smoothstep(0.55, 0.85, edge_c * (0.5 + wob)) * (1.0 - band * 0.5) * dry_side;
        albedo = mix(albedo, albedo * 1.25 + vec3<f32>(0.02, 0.018, 0.014), tide * 0.5);
    }
    // Settled dust, metres. A tenth of a millimetre greys the soil, a couple
    // of millimetres hide it. Dust is matte and fills the grain.
    let dust_m = bc_dust_settled(xz);
    if (dust_m > 0.0) {
        let speck = noise(xz * 11.0) * 0.6 + noise(xz * 37.0) * 0.4;
        let cover = smoothstep(0.00004, 0.0018, dust_m * (0.6 + 0.8 * speck));
        // On wet ground the dust turns to a dark smear, not a pale powder.
        let dust_col = mix(vec3<f32>(0.40, 0.34, 0.27), vec3<f32>(0.16, 0.12, 0.08), smoothstep(0.35, 0.7, wet));
        albedo = mix(albedo, dust_col, cover * 0.85);
        rough_out = mix(rough_out, 0.95, cover * (1.0 - smoothstep(0.35, 0.7, wet)));
        n = normalize(mix(n, n_geo, cover * 0.6));
    }
    // Specular AA: a normal that turns faster than a pixel widens the lobe,
    // so a wet grain far away reads as sheen, not white sparkle.
    let dndx = dpdx(n);
    let dndy = dpdy(n);
    let variance = 0.25 * (dot(dndx, dndx) + dot(dndy, dndy));
    rough_out = sqrt(clamp(rough_out * rough_out + min(2.0 * variance, 0.2), 0.0, 1.0));
    let lit = shade_pbr(in.world, n, albedo, rough_out, 0.0, mix(1.0, ao, 0.6), vec3<f32>(0.0));
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
    // Dilate slightly so a dry speck inside the bowl does not punch a black hole.
    let e = 0.28;
    let d0 = mat_bilinear(xz).g;
    let d1 = mat_bilinear(xz + vec2<f32>(e, 0.0)).g;
    let d2 = mat_bilinear(xz - vec2<f32>(e, 0.0)).g;
    let d3 = mat_bilinear(xz + vec2<f32>(0.0, e)).g;
    let d4 = mat_bilinear(xz - vec2<f32>(0.0, e)).g;
    return max(d0, max(d1, max(d2, max(d3, d4)))) * 0.34;
}

// Solved column for a crossing. misc.w selects the wider pack so a bow
// above the old 0.34 m film range is the free surface, not a flat sheet.
fn water_column(xz: vec2<f32>) -> f32 {
    let scale = select(0.34, 0.90, frame.misc.w > 0.5);
    return mat_bilinear(xz).g * scale;
}

// A dry patch with water on both sides is a hole in the sheet, not a shore.
// The outer bank has water on only one side, so this does not grow the pond.
fn covered_depth(xz: vec2<f32>) -> f32 {
    let here = water_depth(xz);
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

@vertex
fn vs_water(in: TerrIn) -> WaterOut {
    let local = height_bilinear(in.xz);
    let dug = max(-stamp_bilinear(in.xz), 0.0);
    let depth = covered_depth(in.xz);
    let column = water_column(in.xz);
    let free = frame.misc.w > 0.5 && column > 0.004;
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
        let film = bc_film(in.xz);
        let thick = film.r + film.g;
        let y_c = local + thick + rip * smoothstep(0.01, 0.04, thick) * 0.6;
        y = mix(y, y_c, inside);
        d_out = mix(d_out, thick, inside);
    }
    let world = vec3<f32>(in.xz.x, y, in.xz.y);
    var o: WaterOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    o.depth = d_out;
    return o;
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    let xz = in.world.xz;
    let kind = kind_nearest(xz);
    let bed_layer = layer_of(kind);
    let bed = textureSample(ground_diff, ground_samp, layer_uv(bed_layer, xz), bed_layer).rgb;
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
    let free = frame.misc.w > 0.5 && in.depth > 0.004;
    let inside = bc_inside(xz);
    let film_c = bc_film(xz);
    let column_c = film_c.r + film_c.g;
    if (inside > 0.5) {
        if (column_c < 0.004) {
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
        let dc = 0.2;
        let s0 = height_bilinear(xz) + column_c;
        let f_x = bc_film(xz + vec2<f32>(dc, 0.0));
        let f_z = bc_film(xz + vec2<f32>(0.0, dc));
        let sx = height_bilinear(xz + vec2<f32>(dc, 0.0)) + f_x.r + f_x.g;
        let sz = height_bilinear(xz + vec2<f32>(0.0, dc)) + f_z.r + f_z.g;
        let c_n = normalize(vec3<f32>(-(sx - s0) / dc * 1.6, 1.0, -(sz - s0) / dc * 1.6));
        let c_n2 = normalize(c_n + vec3<f32>(n.x * 0.8, 0.0, n.z * 0.8));
        // select, not mix: the picture-cache slope can be NaN on the
        // through-drive field, and mix carries a NaN through a zero weight.
        nrm = select(normalize(mix(nrm, c_n2, inside)), c_n2, inside > 0.99);
    }
    let depth = mix(in.depth, column_c, inside);
    let slurry_frac = select(0.0, film_c.g / max(column_c, 1e-4), column_c > 1e-4) * inside;
    let flow = bc_flow(xz);
    let speed = length(flow);
    let t_now = frame.misc.x;
    // Capillary ripples on top of the long swell. Two octaves of value noise
    // carried downstream by the published flow; still water just breathes.
    let calm = smoothstep(0.015, 0.06, depth);
    let drift = vec2<f32>(t_now * 0.35, t_now * 0.21) - flow * t_now * 4.0;
    let re = 0.03;
    let q = xz * 7.0;
    let r0 = noise(q + drift) + 0.5 * noise(q * 2.3 - drift * 1.4);
    let rx = noise(q + vec2<f32>(re * 7.0, 0.0) + drift) + 0.5 * noise((q + vec2<f32>(re * 7.0, 0.0)) * 2.3 - drift * 1.4);
    let rz = noise(q + vec2<f32>(0.0, re * 7.0) + drift) + 0.5 * noise((q + vec2<f32>(0.0, re * 7.0)) * 2.3 - drift * 1.4);
    nrm = normalize(nrm + vec3<f32>(r0 - rx, 0.0, r0 - rz) * 0.22 * calm);
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
    let shadow = shadow_factor(in.world + nrm * 0.01, nrm);

    // Muddy water is a suspension: Beer-Lambert through the column, red
    // passes furthest. The bed below is the real soil pass, seen through alpha.
    let path = depth / max(ndv, 0.25);
    // Slurry is far more opaque than muddy water.
    let sigma = mix(vec3<f32>(26.0, 34.0, 48.0), vec3<f32>(90.0, 110.0, 140.0), slurry_frac);
    let trans = exp(-sigma * path);
    // Light scattered by the silt in the column, lit by sun and sky.
    // Deep columns go darker: light that scatters back out has crossed more silt.
    let deep_t = smoothstep(0.04, 0.25, depth);
    var silt = mix(vec3<f32>(0.17, 0.10, 0.042), bed * vec3<f32>(0.7, 0.52, 0.32), 0.35) * mix(1.0, 0.45, deep_t);
    silt = mix(silt, vec3<f32>(0.13, 0.075, 0.035), slurry_frac);
    let ndl = clamp(dot(vec3<f32>(0.0, 1.0, 0.0), sun), 0.0, 1.0);
    let light = frame.sun_color.rgb * ndl * shadow * 1.25 * 0.45 + sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * frame.env.y;
    let body = silt * light;

    // Fresnel reflection of the sky, and a tight GGX sun glint on the ripples.
    let fres = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let refl_dir = reflect(-view, nrm);
    var refl = env_radiance(refl_dir, 0.05) * frame.env.y * 1.15;
    // A rut is a trench: a grazing reflection meets the bank before the sky.
    // March the heightfield a short way along the ray and show the soil it hits.
    var p = in.world + refl_dir * 0.04;
    let stride = refl_dir * 0.12;
    for (var i: i32 = 0; i < 14; i = i + 1) {
        p = p + stride * (1.0 + f32(i) * 0.25);
        if (height_bilinear(p.xz) > p.y + 0.015) {
            let bank_n = terrain_normal(p.xz);
            let bank_l = frame.sun_color.rgb * clamp(dot(bank_n, sun), 0.0, 1.0) * 1.25 * shadow_factor(p, bank_n)
                + sh_irradiance(bank_n) * frame.env.y;
            // Fade with distance so a far wall does not cut a hard edge.
            let near = 1.0 - smoothstep(0.6, 2.2, length(p - in.world));
            refl = mix(refl, vec3<f32>(0.09, 0.055, 0.03) * bank_l, near);
            break;
        }
    }
    let halfv = normalize(sun + view);
    let ndh = clamp(dot(nrm, halfv), 0.0, 1.0);
    let ndl_w = clamp(dot(nrm, sun), 0.0, 1.0);
    let glint = ggx_d(ndh, 0.035) * 0.02 * ndl_w * shadow * calm;
    let sun_spec = frame.sun_color.rgb * min(glint, 40.0);

    // Foam is the published fraction (Froude above 0.85). It is broken into
    // a lace that rides the current, not a flat white decal. A crest the
    // picture swell pushes up adds a little churn where no sheet is attached.
    let fc = bc_foam_crest(xz);
    let lace_p = xz * 9.0 - flow * t_now * 9.0;
    let lace = noise(lace_p) * 0.6 + noise(lace_p * 2.7 + vec2<f32>(3.1, 1.7)) * 0.4;
    let churn = smoothstep(0.7, 0.45, nrm.y) * smoothstep(0.06, 0.14, depth) * (1.0 - inside);
    // The published fraction is the foam. A crest of a few millimetres still
    // breaks the surface into a thin lace where the bow pushes it up.
    let crest_lace = smoothstep(0.002, 0.012, fc.g) * 0.55;
    let foam_amt = max(max(fc.r, crest_lace), churn);
    let foam = smoothstep(1.0 - foam_amt * 0.9, 1.05 - foam_amt * 0.6, lace) * smoothstep(0.004, 0.012, depth);
    let foam_col = mix(vec3<f32>(0.55, 0.44, 0.30), vec3<f32>(0.72, 0.64, 0.52), 1.0 - slurry_frac) * light;

    // Blend over the soil: dst * T * (1 - F) + body * (1 - T) * (1 - F) + refl * F.
    let opacity = clamp(1.0 - (trans.r + trans.g + trans.b) / 3.0, 0.0, 1.0);
    var alpha = 1.0 - (1.0 - opacity) * (1.0 - fres);
    var lit = (body * opacity * (1.0 - fres) + refl * fres) / max(alpha, 1e-3) + sun_spec;
    // Headlamps glint on the ripples and light the silt in the column.
    lit += lamp_light(in.world, nrm, view, silt * opacity, vec3<f32>(0.02), 0.07) / max(alpha, 0.3);
    lit = mix(lit, foam_col, foam * 0.8);
    alpha = max(alpha, foam * 0.9);
    // Feather the shoreline so the edge is a wet margin, not a cut line.
    alpha = alpha * smoothstep(0.012, 0.03, depth);
    lit = atmosphere(in.world, lit);
    return vec4<f32>(tonemap(lit), clamp(alpha, 0.0, 1.0));
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
    let world = model.world * vec4<f32>(pos, 1.0);
    var o: MeshOut;
    o.clip = frame.view_proj * world;
    o.world = world.xyz;
    o.nrm = normalize((model.world * vec4<f32>(in.nrm, 0.0)).xyz);
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

@vertex
fn vs_shadow_terrain(in: TerrIn) -> ShadowOut {
    let h = apply_seat(in.xz, height_bilinear(in.xz)).x;
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
    let ground = terrain_height(world.xz);
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
        let roll = (in.uv.y - 0.5) * 1.6;
        let bump = noise(vec2<f32>(u * 48.0, in.uv.y * 3.0)) - 0.5;
        let tside = normalize(cross(n, view) + vec3<f32>(1e-4, 0.0, 0.0));
        n = normalize(n + tside * (roll + bump * 0.5));
        let g = noise(vec2<f32>(u * 9.0, in.uv.y * 2.0));
        // Liquid clay: brown, not tar. The gloss is the sky on a wet skin.
        let albedo = mix(vec3<f32>(0.16, 0.095, 0.050), vec3<f32>(0.24, 0.15, 0.08), g);
        var lit = shade_pbr(in.world, n, albedo, 0.14, 0.0, 1.0, vec3<f32>(0.0));
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
    if (in.panel > 49.9 && in.panel < 52.99) {
        return spray_strip(in);
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
        let max_lod = max(f32(textureNumLevels(niva_base)) - 1.0, 0.0);
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
