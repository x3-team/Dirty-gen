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
    misc: vec4<f32>,   // time, shadow size, packed height, pixel hint
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
@group(1) @binding(4) var clay_tex: texture_2d<f32>;
@group(1) @binding(5) var clay_samp: sampler;

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
// Clay grain is a separate normal. Wet mud keeps this slope so the sheen
// does not break into gravel.
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

// World size of one pixel on the ground. Perspective stores a negative
// angular size and scales by distance here, so the lod does not jump on
// triangle edges the way dpdx does.
fn ground_pixel(world: vec3<f32>) -> f32 {
    let hint = frame.misc.w;
    if (hint < 0.0) {
        let dist = length(frame.cam_pos.xyz - world);
        let elev = abs(frame.cam_pos.y - world.y) / max(dist, 0.05);
        return dist * (-hint) / max(elev, 0.22);
    }
    return max(hint, 1.0e-4);
}

fn clay_at(world: vec3<f32>, uv: vec2<f32>, scale: f32) -> vec4<f32> {
    let texels = ground_pixel(world) * scale * 256.0;
    let lod = clamp(log2(max(texels, 1.0)), 0.0, 8.0);
    return textureSampleLevel(clay_tex, clay_samp, uv, lod);
}

// ~0.6 m tile for crumbs up close, ~2.4 m tile for patches on the oval.
// Normals stay on the fine tile.
fn clay_sample(world: vec3<f32>) -> vec4<f32> {
    let xz = world.xz;
    let fine = clay_at(world, xz * 1.72, 1.72);
    let wide = clay_at(world, xz * 0.42 + vec2<f32>(0.17, 0.41), 0.42);
    return vec4<f32>(
        fine.rg,
        fine.b * 0.70 + wide.b * 0.30,
        fine.a * 0.40 + wide.a * 0.60,
    );
}

fn terrain_normal(xz: vec2<f32>, clay: vec4<f32>, grit_amp: f32) -> vec3<f32> {
    // Geometric slope stays the curved height. The clay normal is a small
    // tangent tilt, not a second displacement, so a crumb cannot become a cliff.
    let geo = normalize(slope_draw(xz));
    let nx = (clay.r * 2.0 - 1.0) * grit_amp;
    let nz = (clay.g * 2.0 - 1.0) * grit_amp;
    let ny = sqrt(clamp(1.0 - nx * nx - nz * nz, 0.0, 1.0));
    var t = vec3<f32>(1.0, 0.0, 0.0) - geo * geo.x;
    if (dot(t, t) < 1.0e-4) {
        t = vec3<f32>(0.0, 0.0, 1.0) - geo * geo.z;
    }
    t = normalize(t);
    let b = normalize(cross(t, geo));
    return normalize(t * nx + b * nz + geo * ny);
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

// Thin sky sheen. Flat mud looks back at the camera and contributes nothing,
// so standing water cannot fill in as a disc. The highlight is the real
// heightfield slope: a rut wall or the side of a bowl.
fn mud_sheen(world: vec3<f32>, n_in: vec3<f32>, rough: f32, wet: f32, water: f32) -> vec3<f32> {
    let film = smoothstep(0.008, 0.04, water);
    let wet_g = smoothstep(0.48, 0.88, wet);
    let smooth_g = smoothstep(0.86, 0.32, rough);
    let gate = smooth_g * max(wet_g, film);
    if (gate <= 0.001) {
        return vec3<f32>(0.0);
    }
    let n = normalize(n_in);
    // A few degrees of real wall. Flat film stays matte.
    let along = smoothstep(0.012, 0.055, 1.0 - n.y);
    if (along <= 0.001) {
        return vec3<f32>(0.0);
    }
    let view = normalize(frame.cam_pos.xyz - world);
    let sun = normalize(frame.sun_dir.xyz);
    let halfv = normalize(sun + view);
    let ndoth = clamp(dot(n, halfv), 0.0, 1.0);
    let refl = reflect(-view, n);
    // Below the horizon the mirror would be a gray ground fill.
    let sky_gate = smoothstep(0.04, 0.22, refl.y);
    // Wet mud takes the sky in the reflection. Standing water keeps a tighter
    // sun-aligned streak so the bowl does not fill in.
    let sky_lobe = pow(clamp(refl.y, 0.0, 1.0), mix(3.2, 6.5, film));
    let sun_lobe = pow(ndoth, mix(4.5, 9.0, film));
    let sky = sky_dome(refl);
    let shadow = shadow_factor(world + n * 0.02, n);
    let amp = gate * along * sky_gate * shadow;
    let lobe = mix(sky_lobe * 0.8 + sun_lobe * 0.35, sun_lobe * 0.7 + sky_lobe * 0.25, film);
    // Capped so a cloud reflection stays a streak, not a white plate.
    return min(sky * lobe * amp * 5.2, vec3<f32>(0.40));
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
    // Curved reconstruction of the solver samples. Bilinear leaves the tyre
    // lip as a straight ramp with a corner on the cell boundary. Water stays
    // on height_bilinear so the sheet does not pay for this gather.
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
    let clay = clay_sample(in.world);

    // Dark saturated soil. The noon tan read as flat beige. Clay multiplies
    // this after the wet mixes so a trough stays the dark colour.
    let dry = vec3<f32>(0.105, 0.052, 0.026);
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
    // Dark mud in the trough.
    albedo = mix(albedo, vec3<f32>(0.010, 0.0045, 0.002), groove * 0.88);
    let pile = max(stamp, 0.0);
    let lip = smoothstep(0.012, 0.026, pile) * (1.0 - groove);
    // Loose soil on the berm. Lighter than the lane, only where the stamp piled it.
    albedo = mix(albedo, vec3<f32>(0.15, 0.078, 0.034), lip * 0.92);

    // Standing water darkens the bowl. The bank of the same heightfield stays lighter.
    let relief = local_relief(xz);
    let bowl = smoothstep(0.045, 0.14, -relief);
    let bank = smoothstep(0.04, 0.12, relief) * (1.0 - groove);
    let depth_t = smoothstep(0.004, 0.10, water);
    albedo = mix(
        albedo,
        vec3<f32>(0.012, 0.005, 0.0022),
        max(bowl, depth_t) * mix(0.4, 0.78, max(wet_face, film)),
    );
    let bank_col = mix(vec3<f32>(0.16, 0.082, 0.036), vec3<f32>(0.062, 0.032, 0.014), wet_face);
    albedo = mix(albedo, bank_col, bank * mix(0.45, 0.28, wet_face) * (1.0 - depth_t));

    // Dry dirt takes the clay albedo and normal. A wet trough or water film
    // fades both back to the smooth dark slope the sheen already uses.
    let wet_gate = max(groove, max(wet_face, film));
    let grain = 1.0 - wet_gate * 0.94;
    // The authored amount sits in a narrow band. Stretch it so pores and
    // crumbs separate, and keep the middle near the old flat colour.
    let amount = clamp((clay.b - 0.10) * 2.35, 0.0, 1.0);
    let warm = clamp((clay.a - 0.08) * 2.25, 0.0, 1.0);
    let lum = mix(0.32, 1.48, amount);
    let tint = mix(vec3<f32>(0.74, 1.12, 1.06), vec3<f32>(1.26, 0.86, 0.70), warm);
    albedo *= mix(vec3<f32>(1.0), tint * lum, grain);
    let grit_amp = mix(1.0, 0.04, wet_gate);
    let n = terrain_normal(xz, clay, grit_amp);

    // Dry dirt stays rough. Wet mud takes the stored smoothness. A water film
    // is smoother, and the highlight is still the slope lobe, not a mirror fill.
    let dry_r = max(stored_rough, 0.92);
    let wet_r = clamp(stored_rough, 0.15, 0.72);
    let rough_out = mix(dry_r, mix(wet_r, 0.09, film), max(wet_face, film));

    var lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
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
    let bed = water_bed(in.xz);
    let depth = covered_depth(in.xz);
    let rip = ripple_height(in.xz, frame.misc.x) * smoothstep(0.0, 0.03, depth);
    // Film on the bed, but never below the soil. A bump left uncovered
    // reads as a black hole. bed + full depth was a sheet 30 cm up that
    // cut the tire like a clip plane.
    let cover = height_bilinear(in.xz) + 0.01;
    let y = max(bed + 0.016 + rip, cover) + smoothstep(0.02, 0.16, depth) * 0.012;
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
    // The old sheet sat a centimetre above a blurred bed with an up-facing
    // normal. That is the flat gray disc. Standing water is the sheen on the
    // displaced mud in fs_terrain, so this pass draws nothing.
    if (in.depth < 1.0e6) {
        discard;
    }
    return vec4<f32>(0.0);
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
