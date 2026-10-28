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
    extra: vec4<f32>, // is_wheel, wheel mud 0..1, _, _
}

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var mat_tex: texture_2d<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var shadow_tex: texture_2d<f32>;

@group(2) @binding(0) var<uniform> model: Model;

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

// Metres the stamp added. Negative is soil the tire removed.
fn stamp_at(ij: vec2<i32>) -> f32 {
    return load_height_stamp(ij).y;
}

fn mat_at(ij: vec2<i32>) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(mat_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(mat_tex, c, 0);
}

// World x = (i + 0.5) * cell - half. Inverse of the heightfield layout.
fn world_to_ij(xz: vec2<f32>) -> vec2<f32> {
    let cell = max(frame.params.z, 1e-4);
    let half = cell * frame.params.w * 0.5;
    return (xz + vec2<f32>(half)) / cell - vec2<f32>(0.5);
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

// Cells. Wide enough that a one-cell tyre lip becomes an S, narrow enough
// that a flat rut floor stays within a few millimetres of the solver.
const DRAW_SIGMA: f32 = 0.64;

fn draw_gauss(d: f32) -> f32 {
    let q = d / DRAW_SIGMA;
    return exp(-0.5 * q * q);
}

struct DrawAxis {
    w: array<f32, 5>,
    dw: array<f32, 5>,
};

// Normalized gaussian and its derivative in index space. Radius 2 cells.
// Kept in lockstep with `draw_axis` in mesh.rs tests.
fn draw_axis(f: f32) -> DrawAxis {
    var g: array<f32, 5>;
    var dg: array<f32, 5>;
    var sum = 0.0;
    var dsum = 0.0;
    let inv_s2 = 1.0 / (DRAW_SIGMA * DRAW_SIGMA);
    for (var i = 0; i < 5; i++) {
        let d = f - (f32(i) - 2.0);
        let gi = draw_gauss(d);
        let dgi = gi * (-d * inv_s2);
        g[i] = gi;
        dg[i] = dgi;
        sum += gi;
        dsum += dgi;
    }
    var ax: DrawAxis;
    let inv = 1.0 / sum;
    let inv2 = inv * inv;
    for (var i = 0; i < 5; i++) {
        ax.w[i] = g[i] * inv;
        ax.dw[i] = (dg[i] * sum - g[i] * dsum) * inv2;
    }
    return ax;
}

fn height_draw(xz: vec2<f32>) -> f32 {
    let ij = world_to_ij(xz);
    let o = vec2<i32>(floor(ij));
    let fx = draw_axis(fract(ij).x);
    let fz = draw_axis(fract(ij).y);
    var h = 0.0;
    for (var j = 0; j < 5; j++) {
        var row = 0.0;
        for (var i = 0; i < 5; i++) {
            row += fx.w[i] * height_at(o + vec2<i32>(i - 2, j - 2));
        }
        h += fz.w[j] * row;
    }
    return h;
}

// xy: dh/dx, dh/dz in metres. zw: stamp slope, same kernel, same units.
fn draw_grads(xz: vec2<f32>) -> vec4<f32> {
    let ij = world_to_ij(xz);
    let o = vec2<i32>(floor(ij));
    let fx = draw_axis(fract(ij).x);
    let fz = draw_axis(fract(ij).y);
    let cell = max(frame.params.z, 1e-4);
    var dhdx = 0.0;
    var dhdz = 0.0;
    var dsdx = 0.0;
    var dsdz = 0.0;
    for (var j = 0; j < 5; j++) {
        var row_h = 0.0;
        var row_s = 0.0;
        var drow_h = 0.0;
        var drow_s = 0.0;
        for (var i = 0; i < 5; i++) {
            let hs = load_height_stamp(o + vec2<i32>(i - 2, j - 2));
            row_h += fx.w[i] * hs.x;
            row_s += fx.w[i] * hs.y;
            drow_h += fx.dw[i] * hs.x;
            drow_s += fx.dw[i] * hs.y;
        }
        dhdx += fz.w[j] * drow_h;
        dsdx += fz.w[j] * drow_s;
        dhdz += fz.dw[j] * row_h;
        dsdz += fz.dw[j] * row_s;
    }
    return vec4<f32>(dhdx, dhdz, dsdx, dsdz) / cell;
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
    let sx = smoothstep(0.55, 0.96, f.x);
    let sy = smoothstep(0.55, 0.96, f.y);
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

// Slope of the curved draw height. A finite difference of the bilinear
// ramp still kinks on the solver-cell edge the tyre lip sits on.
// `grit_amp` is the dry grain. Wet mud passes a small value so the sheen
// follows this slope instead of breaking into gravel.
fn slope_draw(xz: vec2<f32>) -> vec3<f32> {
    let g = draw_grads(xz);
    return normalize(vec3<f32>(-g.x, 1.0, -g.y));
}

// Wider sample of the same curved height. The short gaussian still rings
// once per solver cell under the low sun, and that ring draws the sheen
// as a stair. Water stays on this wider slope. The tyre lip stays on
// slope_draw.
fn slope_draw_wide(xz: vec2<f32>, step: f32) -> vec3<f32> {
    let e = max(step, 0.05);
    let h_l = height_draw(xz - vec2<f32>(e, 0.0));
    let h_r = height_draw(xz + vec2<f32>(e, 0.0));
    let h_d = height_draw(xz - vec2<f32>(0.0, e));
    let h_u = height_draw(xz + vec2<f32>(0.0, e));
    return normalize(vec3<f32>(h_l - h_r, e * 2.0, h_d - h_u));
}

fn terrain_normal(xz: vec2<f32>, grit_amp: f32) -> vec3<f32> {
    // The slope is the curved height difference. Amplifying the stamp on
    // its own turned a centimetre of noise into a cliff, and the sheen
    // into rings.
    var n = slope_draw(xz);
    let grit = noise(xz * 2.4);
    let grit_x = noise(xz * 2.4 + vec2<f32>(0.45, 0.0));
    let grit_z = noise(xz * 2.4 + vec2<f32>(0.0, 0.45));
    n = normalize(n + vec3<f32>(grit - grit_x, 0.0, grit - grit_z) * grit_amp);
    return n;
}

// Height minus a wide neighbourhood. Positive on a berm, negative in a bowl.
// A short radius turned the two edges of one tyre cut into extra scars.
fn local_relief(xz: vec2<f32>) -> f32 {
    let h = height_bilinear(xz);
    let r = max(frame.params.z * 5.0, 0.9);
    let d = r * 0.7071;
    let avg = (
        height_bilinear(xz + vec2<f32>(r, 0.0))
        + height_bilinear(xz - vec2<f32>(r, 0.0))
        + height_bilinear(xz + vec2<f32>(0.0, r))
        + height_bilinear(xz - vec2<f32>(0.0, r))
        + height_bilinear(xz + vec2<f32>(d, d))
        + height_bilinear(xz + vec2<f32>(-d, d))
        + height_bilinear(xz + vec2<f32>(d, -d))
        + height_bilinear(xz + vec2<f32>(-d, -d))
    ) * 0.125;
    return h - avg;
}

// Sky without the sun disc. The disc is one point; folding it into the wet
// lobe turned a slope into a white plate.
fn sky_dome(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let h = clamp(dir.y, 0.0, 1.0);
    var col = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, smoothstep(0.0, 0.55, h));
    let p = dir.xz / (abs(dir.y) * 0.45 + 0.25);
    let n = noise(p * 0.18);
    let cloud = smoothstep(0.58, 0.82, n) * smoothstep(0.08, 0.45, dir.y);
    col = mix(col, vec3<f32>(0.93, 0.94, 0.96), cloud * 0.65);
    return col;
}

// Sheen on a wet slope. Flat mud looks back at the camera and contributes
// nothing, so standing water cannot fill in as a disc. The broad term is the
// same clay as the lane, darker. The sky is only a thin streak: a wide lobe
// of the zenith is what read as blue plastic.
fn mud_sheen(world: vec3<f32>, n_in: vec3<f32>, rough: f32, wet: f32, water: f32) -> vec3<f32> {
    let film = smoothstep(0.004, 0.02, water);
    let wet_g = smoothstep(0.48, 0.88, wet);
    let smooth_g = smoothstep(0.86, 0.32, rough);
    let gate = smooth_g * max(wet_g, film);
    if (gate <= 0.001) {
        return vec3<f32>(0.0);
    }
    let n = normalize(n_in);
    // A few degrees of real wall. Flat film stays matte, so the rut floor
    // is the soil colour from fs_terrain and not this coat.
    let along = smoothstep(0.012, 0.055, 1.0 - n.y);
    if (along <= 0.001) {
        return vec3<f32>(0.0);
    }
    let view = normalize(frame.cam_pos.xyz - world);
    let sun = normalize(frame.sun_dir.xyz);
    let halfv = normalize(sun + view);
    let ndoth = clamp(dot(n, halfv), 0.0, 1.0);
    let refl = reflect(-view, n);
    let sky_gate = smoothstep(0.04, 0.22, refl.y);
    let sky = sky_dome(refl);
    let shadow = shadow_factor(world + n * 0.02, n);
    // A slope still glints when the reflection is not aimed at the sky.
    let amp = gate * along * shadow * mix(0.55, 1.0, sky_gate);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    // Shadowed slurry stays brown. A sky-coloured ambient here was the blue.
    let wrap = 0.58 + 0.42 * ndl * shadow;
    // Lane clay is (0.105, 0.052, 0.026). This is that hue, darker.
    let clay = vec3<f32>(0.068, 0.034, 0.017);
    let standing = smoothstep(0.003, 0.014, water);
    let body = clay * wrap * gate * along * mix(0.28, 1.0, standing);
    // Wet churned edges. A broad warm gloss, plus a pale sky tint that stays
    // off the saturated zenith. Flat standing water is the sheet, not this lobe.
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let pale = mix(vec3<f32>(0.48, 0.42, 0.34), sky, 0.15);
    let graze = pow(1.0 - ndotv, 3.0);
    let edge_refl = pale * graze * amp * 0.34;
    let soft = pow(ndoth, 8.0);
    let wet_glint = vec3<f32>(0.58, 0.40, 0.22) * soft * amp * 1.15;
    let sun_tight = pow(ndoth, 46.0);
    let glint = vec3<f32>(0.72, 0.52, 0.30) * sun_tight * amp * 0.4;
    return body * 0.35 + wet_glint + edge_refl + glint;
}

// Along a tyre cut. The stamp rises toward the berm, so its gradient
// points across the rut. Open water has no cut and returns zero.
fn rut_along(xz: vec2<f32>) -> vec2<f32> {
    let e = 0.26;
    let gx = stamp_bilinear(xz + vec2<f32>(e, 0.0)) - stamp_bilinear(xz - vec2<f32>(e, 0.0));
    let gz = stamp_bilinear(xz + vec2<f32>(0.0, e)) - stamp_bilinear(xz - vec2<f32>(0.0, e));
    let g = vec2<f32>(gx, gz);
    let glen = length(g);
    if (glen < 0.006) {
        return vec2<f32>(0.0);
    }
    let across = g / glen;
    return vec2<f32>(-across.y, across.x);
}

struct Ripple {
    h: f32,
    // dh/dx, dh/dz of the unitless field. The axis is local.
    grad: vec2<f32>,
    // 0 in the gaps, 1 where a patch of ripple actually exists.
    brk: f32,
}

fn ridge(p: vec2<f32>) -> f32 {
    // Contours of warped noise. They close and die. They are not a phase.
    let n = noise(p);
    let r = 1.0 - abs(n * 2.0 - 1.0);
    return smoothstep(0.55, 0.92, r);
}

// Unitless crest in rut space. Irregular ridge fragments, biased along the
// cut, with gaps so nothing runs on as a streak.
fn broken_h(s: f32, c: f32, t: f32) -> vec2<f32> {
    let w0 = noise(vec2<f32>(s * 1.2, c * 1.5) + t * 0.04);
    let w1 = noise(vec2<f32>(c * 1.4 + 3.1, s * 0.85) - t * 0.03);
    let s2 = s + (w0 - 0.5) * 1.15;
    let c2 = c + (w1 - 0.5) * 0.85;
    // Slightly longer along the rut. The field is still a contour, not a comb.
    let p = vec2<f32>(s2 * 2.6, c2 * 3.4);
    let bend = vec2<f32>(
        noise(p * 0.45 + t * 0.02),
        noise(p * 0.45 + vec2<f32>(2.2, 6.0)),
    );
    let q = p + (bend - 0.5) * 1.8;
    let gap = smoothstep(0.38, 0.66, noise(vec2<f32>(s2 * 1.7 + 0.4, c2 * 2.1)));
    let h1 = ridge(q) * gap;
    let q2 = vec2<f32>(s2 * 5.4 + 1.8, c2 * 4.7) + (bend - 0.5) * 1.2;
    let gap2 = smoothstep(0.46, 0.74, noise(vec2<f32>(c2 * 3.3 + 2.0, s2 * 2.8)));
    let h2 = ridge(q2) * gap2;
    let h = clamp(h1 * 0.72 + h2 * 0.5, 0.0, 1.0);
    let brk = clamp(gap * h1 + gap2 * 0.55, 0.0, 1.0);
    return vec2<f32>(h, brk);
}

fn ripple_wave(xz: vec2<f32>, t: f32) -> Ripple {
    var along = rut_along(xz);
    if (length(along) < 0.5) {
        along = vec2<f32>(0.71, 0.70);
    }
    let across = vec2<f32>(-along.y, along.x);
    let s = dot(xz, along);
    let c = dot(xz, across);
    let h = broken_h(s, c, t);
    // Crests are a few centimetres wide. The step has to be shorter than that.
    let e = 0.012;
    let hs = broken_h(s + e, c, t).x;
    let hc = broken_h(s, c + e, t).x;
    // Slope on the flank, not the unitless derivative. About half a unit of
    // tilt where a crest actually rises.
    let scale = 0.007;
    let dhs = (hs - h.x) / e * scale;
    let dhc = (hc - h.x) / e * scale;
    var w: Ripple = Ripple(0.0, vec2<f32>(0.0), 0.0);
    w.h = h.x * 0.006;
    w.grad = vec2<f32>(dhs * along.x + dhc * across.x, dhs * along.y + dhc * across.y);
    w.brk = h.y;
    return w;
}

fn ripple_height(xz: vec2<f32>, t: f32) -> f32 {
    return ripple_wave(xz, t).h;
}

fn standing_depth(xz: vec2<f32>) -> f32 {
    return mat_bilinear(xz).g * 0.34;
}

// How far the sheet sits above the clay. A film in the shallows, the free
// surface once the column is deep. The sag cups a rut so the sheet follows
// the cut instead of spanning it like a taut cover.
fn sheet_lift(xz: vec2<f32>, depth: f32) -> f32 {
    let rise = smoothstep(0.012, 0.075, depth);
    let cut = max(-stamp_bilinear(xz), 0.0);
    let sag = smoothstep(0.012, 0.042, cut) * min(depth, 0.055) * 0.4;
    return mix(0.011, depth, rise) - sag;
}

// Drawn clay plus the standing column. Same reconstruction the terrain uses,
// so the film is above the mesh you see and not under a bilinear ledge.
fn sheet_y(xz: vec2<f32>) -> f32 {
    return height_draw(xz) + sheet_lift(xz, standing_depth(xz));
}

// Analytic water normal. Screen derivatives of the mesh are a hard grid.
// The base slope is the sheet in the rut. The wave is a small tilt on it.
fn ripple_normal(xz: vec2<f32>, t: f32) -> vec3<f32> {
    let e = 0.16;
    let y_l = sheet_y(xz - vec2<f32>(e, 0.0));
    let y_r = sheet_y(xz + vec2<f32>(e, 0.0));
    let y_d = sheet_y(xz - vec2<f32>(0.0, e));
    let y_u = sheet_y(xz + vec2<f32>(0.0, e));
    let sx = (y_l - y_r) / (e * 2.0);
    let sz = (y_d - y_u) / (e * 2.0);
    let wave = ripple_wave(xz, t);
    let amp = smoothstep(0.004, 0.02, standing_depth(xz));
    // A broad wobble breaks a flat mirror. The ridge tilt stays small so the
    // pool does not grow a comb of streaks along the rut.
    let q = xz * 0.85;
    let n0 = noise(q);
    let nx = noise(q + vec2<f32>(0.37, 0.08));
    let nz = noise(q + vec2<f32>(0.06, 0.37));
    let wobble = amp * 0.028;
    return normalize(vec3<f32>(
        sx + wave.grad.x * amp * 0.12 + (n0 - nx) * wobble,
        1.0,
        sz + wave.grad.y * amp * 0.12 + (n0 - nz) * wobble,
    ));
}

fn terrain_height(xz: vec2<f32>) -> f32 {
    // Curved reconstruction of the solver samples. Bilinear leaves the tyre
    // lip as a straight ramp with a corner on the cell boundary.
    return height_draw(xz);
}

@vertex
fn vs_terrain(in: TerrIn) -> TerrOut {
    // Draw height, not the solver bilinear. Integer ij would flatten every
    // subdivided vertex back onto one ramp per cell.
    let h = terrain_height(in.xz);
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
    // Same xz as the height so wetness sits on the displaced cell, not a
    // texel to the side of the rut.
    let mat = mat_bilinear(xz);
    let wet = mat.r;
    let water = mat.g * 0.34;
    let darken = mat.b;
    let stored_rough = mat.a;

    let film = smoothstep(0.006, 0.035, water);
    let wet_face = smoothstep(0.18, 0.8, wet);
    // Grain on dry dirt. Wet faces keep the heightfield slope intact.
    let grit_amp = mix(0.36, mix(0.04, 0.012, film), max(wet_face, film));
    var n = terrain_normal(xz, grit_amp);

    let grit = noise(xz * 1.4) * 0.6 + noise(xz * 4.8) * 0.4;
    // Dark saturated soil. The noon tan read as flat beige.
    let dry = vec3<f32>(0.105, 0.052, 0.026) * (0.86 + grit * 0.28);
    let soaked = vec3<f32>(0.04, 0.022, 0.012);
    var albedo = mix(dry, soaked, smoothstep(0.12, 0.88, wet));
    // Compacted soil is a little darker. This is not the groove.
    albedo *= mix(1.0, 0.86, smoothstep(0.25, 0.8, darken));

    // The floor is soil the tire removed. Comparing with a neighbour turned the
    // two edges of one cut into extra dark lines, which read as a nest of scars.
    let stamp = stamp_bilinear(xz);
    let cut = max(-stamp, 0.0);
    // Shallow shoulders of the kernel stay the lane colour. Only the tyre
    // centre, a few centimetres down, reads as the groove.
    let groove = smoothstep(0.030, 0.046, cut);
    // Dark mud in a dry trough. Under a shallow film the floor stays clay,
    // grit included, so the water sheet can show that grain.
    // Clay stays readable while the film is still thin. By 18 cm the sheet
    // covers the floor, so the ink under it does not matter.
    let floor_film = smoothstep(0.004, 0.018, water) * (1.0 - smoothstep(0.11, 0.17, water));
    let ink = groove * mix(0.88, 0.16, floor_film);
    albedo = mix(albedo, vec3<f32>(0.010, 0.0045, 0.002), ink);
    let mottled = dry * (0.52 + noise(xz * 3.4) * 0.62 + noise(xz * 9.4) * 0.42);
    albedo = mix(albedo, mottled, floor_film * 0.95);
    // The wet slope keeps a clean normal for the bank sheen. The shallow
    // floor puts a little clay grain back so it reads through the film.
    let cg = noise(xz * 5.6);
    let cgx = noise(xz * 5.6 + vec2<f32>(0.41, 0.07));
    let cgz = noise(xz * 5.6 + vec2<f32>(0.05, 0.41));
    n = normalize(n + vec3<f32>(cg - cgx, 0.0, cg - cgz) * floor_film * 0.6);
    let pile = max(stamp, 0.0);
    let lip = smoothstep(0.012, 0.026, pile) * (1.0 - groove);
    // Loose soil on the berm. Lighter than the lane, only where the stamp piled it.
    albedo = mix(albedo, vec3<f32>(0.15, 0.078, 0.034), lip * 0.92);

    // The bank of the same heightfield stays lighter. Deep water hides it;
    // the sheet in fs_water is what covers the bowl.
    let relief = local_relief(xz);
    let bank = smoothstep(0.04, 0.12, relief) * (1.0 - groove);
    let deep_w = smoothstep(0.030, 0.080, water);
    let bank_col = mix(vec3<f32>(0.16, 0.082, 0.036), vec3<f32>(0.062, 0.032, 0.014), wet_face);
    albedo = mix(albedo, bank_col, bank * mix(0.45, 0.28, wet_face) * (1.0 - deep_w));

    // Dry dirt stays rough. Wet mud takes the stored smoothness. A water film
    // is smoother, and the highlight is still the slope lobe, not a mirror fill.
    let dry_r = max(stored_rough, 0.92);
    let wet_r = clamp(stored_rough, 0.15, 0.72);
    let rough_out = mix(dry_r, mix(wet_r, 0.09, film), max(wet_face, film));

    var lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
    // Standing water is the sheet in fs_water. Recolouring the clay here
    // hid the floor the shallow film is meant to show.
    // Sheen follows the curved height, not the bilinear kink. Standing
    // water uses the wider sample so the bowl does not stair. A rut keeps
    // the short slope, or the tyre wall disappears. Grit stays off this
    // normal so the streak is the slope.
    let n_sheen = normalize(mix(
        slope_draw(xz),
        slope_draw_wide(xz, frame.params.z * 2.1),
        film,
    ));
    lit += mud_sheen(in.world, n_sheen, rough_out, wet, water);
    return present(lit);
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

@vertex
fn vs_water(in: TerrIn) -> WaterOut {
    let depth = covered_depth(in.xz);
    let amp = smoothstep(0.0, 0.025, depth);
    let rip = ripple_height(in.xz, frame.misc.x) * amp;
    // On the drawn clay, lifted by the column. A dry speck with water on
    // both sides takes the covered column so the sheet does not hole.
    let y = max(sheet_y(in.xz), height_draw(in.xz) + sheet_lift(in.xz, depth)) + rip;
    let world = vec3<f32>(in.xz.x, y, in.xz.y);
    var o: WaterOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    o.depth = depth;
    return o;
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    let xz = in.world.xz;
    // Local column, not the dilated sheet depth. The dilation fills holes;
    // it must not turn a 2 cm film into the deep pool.
    let local = standing_depth(xz);
    let jag = noise(xz * 1.6 + vec2<f32>(0.2, 1.1)) * 0.55 + noise(xz * 4.4 + 1.7) * 0.45;
    let walked = local + (jag - 0.48) * 0.04;
    // A dry speck inside a filled bowl keeps the column. A real 2 cm film does not.
    let interior = smoothstep(0.02, 0.0, local) * step(0.05, in.depth);
    let depth = mix(max(walked, 0.0), max(walked, in.depth), interior);
    if (depth < 0.004) {
        discard;
    }
    // Ragged shoulder only where a pool meets dry mud. A broad 2 cm film has
    // no dry neighbour, and a thin speck inside the pool has no dry side,
    // so neither one is eaten into holes.
    let edge = noise(xz * 0.55 + vec2<f32>(0.4, 2.2)) * 0.65
        + noise(xz * 1.7 + vec2<f32>(1.7, 0.3)) * 0.35;
    let shore_cut = 0.006 + edge * 0.03;
    let se = 0.46;
    let px = standing_depth(xz + vec2<f32>(se, 0.0));
    let nx = standing_depth(xz - vec2<f32>(se, 0.0));
    let pz = standing_depth(xz + vec2<f32>(0.0, se));
    let nz = standing_depth(xz - vec2<f32>(0.0, se));
    let near_deep = max(max(px, nx), max(pz, nz));
    let near_dry = min(min(px, nx), min(pz, nz));
    if (near_dry < 0.005 && near_deep > 0.06 && local < shore_cut) {
        discard;
    }
    let t = frame.misc.x;
    var n = ripple_normal(xz, t);
    if (n.y < 0.0) {
        n = -n;
    }
    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let shadow = shadow_factor(in.world + n * 0.02, n);
    // Warm wrap. shade() folds the blue zenith into the hemi, which turned
    // the pool into a gray sheet.
    let wrap = 0.9 + 0.08 * ndl * shadow;
    // Film stays a brown tint. The filled rut is an opaque tan pool.
    // The step is tight so the pool meets the mud on a line, not a soft halo.
    let cover = smoothstep(0.03, 0.075, depth);
    let shallow_c = vec3<f32>(0.09, 0.042, 0.02);
    let deep_c = vec3<f32>(0.172, 0.112, 0.058);
    let q = xz + vec2<f32>(t * 0.006, -t * 0.004);
    let murk = noise(q * 0.45) * 0.55 + noise(q * 1.15 + 1.3) * 0.45;
    let grit = noise(q * 4.2 + vec2<f32>(2.0, 0.4));
    var body = mix(shallow_c, deep_c, cover);
    body *= 0.9 + murk * 0.16;
    body = mix(body, body * vec3<f32>(0.82, 0.76, 0.64), smoothstep(0.84, 0.96, grit) * cover * 0.25);
    var col = body * wrap;
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let flat = smoothstep(0.78, 0.97, n.y);
    let pool = smoothstep(0.06, 0.12, depth);
    // Dull warm lift on the flat of the pool. Kept small so it is not a gloss.
    let fres = pow(1.0 - ndotv, 3.0);
    let broken = smoothstep(0.4, 0.75, noise(xz * 0.42 + vec2<f32>(1.2, 0.4)));
    col += vec3<f32>(0.12, 0.10, 0.07) * fres * flat * pool * broken * 0.05;
    let shore = smoothstep(0.002, 0.008, depth);
    // About 0.22 at 2 cm. Opaque once the rut is a real pool.
    var alpha = mix(0.2, 0.99, cover) * shore;
    alpha = clamp(alpha, 0.0, 0.99);
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
}

@vertex
fn vs_mesh(in: MeshIn) -> MeshOut {
    let world = model.world * vec4<f32>(in.pos, 1.0);
    var o: MeshOut;
    o.clip = frame.view_proj * world;
    o.world = world.xyz;
    o.nrm = normalize((model.world * vec4<f32>(in.nrm, 0.0)).xyz);
    o.uv = in.uv;
    o.panel = in.panel;
    return o;
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) depth: f32,
}

@vertex
fn vs_shadow_terrain(in: TerrIn) -> ShadowOut {
    let h = height_draw(in.xz);
    let world = vec3<f32>(in.xz.x, h, in.xz.y);
    let clip = frame.light_vp * vec4<f32>(world, 1.0);
    var o: ShadowOut;
    o.clip = clip;
    o.depth = clip.z / max(clip.w, 1e-5);
    return o;
}

@vertex
fn vs_shadow_mesh(in: MeshIn) -> ShadowOut {
    let world = model.world * vec4<f32>(in.pos, 1.0);
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
        let g = noise(world.xz * 0.35);
        return vec4<f32>(vec3<f32>(0.055, 0.030, 0.016) * (0.85 + g * 0.25), 0.96);
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
    if (model.extra.x > 0.5) {
        let ang = in.uv.x * 6.2831853;
        let groove = smoothstep(0.15, 0.8, 0.5 + 0.5 * sin(ang * 14.0));
        let sidewall = select(0.0, 1.0, in.uv.y > 1.5);
        let mud = clamp(model.extra.y, 0.0, 1.2);
        // Existing adhesion. ~0.4 of the cap already cakes the tread.
        let load = smoothstep(0.04, 0.28, mud);
        let wp = in.world;
        let n = noise(wp.xy * 9.0 + vec2<f32>(wp.z * 4.0, ang));
        let n2 = noise(wp.yz * 16.0 + vec2<f32>(ang, 2.0));
        let clump = smoothstep(0.48, 0.82, n * 0.65 + n2 * 0.35);
        var mask = load * (0.88 + 0.12 * clump);
        mask = mask * mix(0.94 + 0.06 * groove, 0.7 + 0.25 * clump, sidewall);
        mask = clamp(mask, 0.0, 1.0);
        albedo = mix(vec3<f32>(0.02, 0.018, 0.016), vec3<f32>(0.12, 0.058, 0.028), mask);
        rough = mix(0.92, mix(0.28, 0.07, groove * (1.0 - sidewall)), mask);
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
    if (in.panel > 47.5 && in.panel < 48.5) {
        // Continuous muddy sheet thrown off a tire. Brown, not a white ball.
        let fray = noise(in.world.xz * 2.4 + vec2<f32>(in.world.y * 1.4, 0.4));
        if (in.uv.y > 0.93 && fray < 0.42) {
            discard;
        }
        let n = normalize(nrm);
        let sun = normalize(frame.sun_dir.xyz);
        let ndl = clamp(dot(n, sun), 0.0, 1.0);
        let shadow = shadow_factor(in.world + n * 0.02, n);
        let grit = noise(in.world.xz * 6.0 + vec2<f32>(in.world.y * 2.2, 1.0));
        var sheet = vec3<f32>(0.145, 0.068, 0.028) * (0.88 + grit * 0.2);
        sheet *= 0.72 + ndl * max(shadow, 0.5) * 0.5;
        let lip = smoothstep(0.62, 0.96, in.uv.y);
        sheet = mix(sheet, vec3<f32>(0.2, 0.12, 0.055), lip * 0.28);
        return present(sheet);
    }
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
    return present(lit);
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
