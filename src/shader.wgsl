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

@group(1) @binding(4) var ground_diff: texture_2d<f32>;
@group(1) @binding(5) var ground_nor: texture_2d<f32>;
@group(1) @binding(6) var ground_pack: texture_2d<f32>;
@group(1) @binding(7) var ground_samp: sampler;

@group(2) @binding(0) var<uniform> model: Model;

@group(3) @binding(0) var buggy_base: texture_2d<f32>;
@group(3) @binding(1) var buggy_nor: texture_2d<f32>;
@group(3) @binding(2) var buggy_mr: texture_2d<f32>;
@group(3) @binding(3) var tire_base: texture_2d<f32>;
@group(3) @binding(4) var tire_nor: texture_2d<f32>;
@group(3) @binding(5) var prop_tex: texture_2d<f32>;
@group(3) @binding(6) var paint_samp: sampler;
@group(3) @binding(7) var flora_tex: texture_2d<f32>;
@group(3) @binding(8) var niva_tex: texture_2d<f32>;

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
    let h = dir.y;
    // The lap cameras look down, so the sky in frame is only a low band.
    // The blue has to arrive just above the land, not halfway to the zenith.
    let t = smoothstep(-0.20, 0.10, h);
    let horizon = frame.sky_horizon.rgb;
    let zenith = frame.sky_zenith.rgb;
    var col = mix(horizon, zenith, t);
    let sun = normalize(frame.sun_dir.xyz);
    let d = max(dot(dir, sun), 0.0);
    let warm = pow(d, 3.0) * smoothstep(0.18, -0.08, h);
    col = mix(col, vec3<f32>(0.98, 0.78, 0.52), warm * 0.42);
    col += frame.sun_color.rgb * pow(d, 1200.0) * 2.4;
    col += frame.sun_color.rgb * pow(d, 10.0) * 0.28 * smoothstep(-0.12, 0.15, h);
    let p = dir.xz / (abs(dir.y) * 0.35 + 0.16);
    let n = noise(p * 0.20);
    let cloud = smoothstep(0.50, 0.78, n) * smoothstep(-0.08, 0.14, h);
    col = mix(col, vec3<f32>(0.97, 0.97, 0.98), cloud * 0.62);
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
    let fog = smoothstep(96.0, 240.0, dist);
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

// Continuous slope of the real displaced height. Not the triangle normal,
// which is constant across a 20 cm cell and flattens a 14 cm groove.
fn terrain_normal(xz: vec2<f32>) -> vec3<f32> {
    // Short step. A wider sample flattens a tyre-width cut into the lane slope.
    let e = max(frame.params.z * 0.28, 0.05);
    let h_l = height_bilinear(xz - vec2<f32>(e, 0.0));
    let h_r = height_bilinear(xz + vec2<f32>(e, 0.0));
    let h_d = height_bilinear(xz - vec2<f32>(0.0, e));
    let h_u = height_bilinear(xz + vec2<f32>(0.0, e));
    var n = normalize(vec3<f32>(h_l - h_r, e * 2.0, h_d - h_u));
    // The berm's real slope stays as it is. Only the stamp's slope is tipped,
    // so a centimetre wall catches the low sun without serrating the berm.
    let se = max(frame.params.z * 0.7, 0.1);
    let s_l = stamp_bilinear(xz - vec2<f32>(se, 0.0));
    let s_r = stamp_bilinear(xz + vec2<f32>(se, 0.0));
    let s_d = stamp_bilinear(xz - vec2<f32>(0.0, se));
    let s_u = stamp_bilinear(xz + vec2<f32>(0.0, se));
    let raw = vec3<f32>(s_l - s_r, 0.0, s_d - s_u);
    let mag = length(raw);
    // A centimetre of wall stays a shallow tilt. Millimetre noise does not
    // become a cord, and a few centimetres does not become a cliff.
    // A few centimetres of tire wall. Sub-millimetre noise stays under the grit.
    if (mag > 0.004) {
        let gain = min(3.5, 0.07 / mag);
        n = normalize(n + raw * gain);
    }
    let grit = noise(xz * 2.4);
    let grit_x = noise(xz * 2.4 + vec2<f32>(0.45, 0.0));
    let grit_z = noise(xz * 2.4 + vec2<f32>(0.0, 0.45));
    // Keep the grit under the slope of a centimetre-deep stamp.
    n = normalize(n + vec3<f32>(grit - grit_x, 0.0, grit - grit_z) * 0.035);
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
    // Bilinear across the cell. The per-cell step that keeps a rut wall
    // turns a smooth berm into a corrugated ring from the isometric camera.
    return height_bilinear(xz);
}

@vertex
fn vs_terrain(in: TerrIn) -> TerrOut {
    // Sample the same heightfield. Integer ij would flatten every subdivided
    // vertex inside a cell, so a 9 cm rut stays one kink.
    // Displacement is the painted map, a few centimetres, not the solver height.
    let map_size = max(frame.params.z * frame.params.w, 1.0);
    let guv = in.xz / map_size + vec2<f32>(0.5);
    let disp = textureSampleLevel(ground_pack, ground_samp, guv, 0.0).b;
    let h = terrain_height(in.xz) + (disp - 0.5) * 0.05;
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
    let h = height_bilinear(xz);
    var n = terrain_normal(xz);

    let mat_dims = vec2<f32>(textureDimensions(mat_tex));
    let mat = textureSampleLevel(mat_tex, mat_samp, (in.ij + vec2<f32>(0.5)) / mat_dims, 0.0);
    let wet = mat.r;
    let water = mat.g * 0.34;
    let darken = mat.b;

    let map_size = max(frame.params.z * frame.params.w, 1.0);
    let guv = xz / map_size + vec2<f32>(0.5);
    let diff = textureSample(ground_diff, ground_samp, guv).rgb;
    let ntex = textureSample(ground_nor, ground_samp, guv).rgb;
    let pack = textureSample(ground_pack, ground_samp, guv);
    let ao = clamp(pack.g, 0.0, 1.0);
    // Roughness mixes the rough map with ARM.g. Metallic is ARM.b.
    let rough_map = clamp(pack.r * (1.0 - pack.a * 0.2), 0.05, 1.0);
    var tng = vec3<f32>(1.0, 0.0, 0.0);
    tng = normalize(tng - n * dot(n, tng));
    let btg = cross(tng, n);
    let ts = ntex * 2.0 - vec3<f32>(1.0);
    n = normalize(tng * ts.x + btg * ts.y + n * ts.z);
    var albedo = diff * mix(1.0, ao, 0.8);
    albedo = mix(albedo, albedo * vec3<f32>(0.42, 0.40, 0.36), smoothstep(0.12, 0.88, wet));
    // Compacted soil is a little darker. This is not the groove.
    albedo *= mix(1.0, 0.86, smoothstep(0.25, 0.8, darken));

    // The floor is soil the tire removed. Comparing with a neighbour turned the
    // two edges of one cut into extra dark lines, which read as a nest of scars.
    let stamp = stamp_bilinear(xz);
    let cut = max(-stamp, 0.0);
    // Shallow shoulders of the kernel stay the lane colour. Only the tyre
    // centre, a few centimetres down, reads as the groove.
    let groove = smoothstep(0.030, 0.046, cut);
    // Dark mud in the cut. A wide sheen here turned the floor into a tan stripe.
    albedo = mix(albedo, vec3<f32>(0.010, 0.0045, 0.002), groove * 0.88);
    let pile = max(stamp, 0.0);
    let lip = smoothstep(0.012, 0.026, pile) * (1.0 - groove);
    // Loose soil on the shoulder. Lighter than the lane, only where the stamp piled it.
    albedo = mix(albedo, vec3<f32>(0.15, 0.078, 0.034), lip * 0.92);

    // Wet rim on the soil. The sky reflection is the water sheet, not this grid.
    let rim = smoothstep(0.0, 0.035, water) * (1.0 - smoothstep(0.04, 0.12, water));
    albedo = mix(albedo, vec3<f32>(0.012, 0.008, 0.006), rim * 0.85);

    // The lane stays matte. Shine is the soil ringing a puddle. The same lobe
    // on the tire cut turned the groove into a tan stripe, so the floor stays dark.
    let shore = smoothstep(0.0, 0.02, water) * (1.0 - smoothstep(0.035, 0.11, water));
    // Standing water is drawn as its own sheet. A mirror bed under that film
    // is the gray metal disc.
    let drowned = smoothstep(0.012, 0.05, water);
    let rough_out = mix(mix(rough_map, 0.16, shore * (1.0 - drowned)), max(rough_map, 0.78), drowned);
    var lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
    // The sun sits low, so a tight Phong highlight misses this camera.
    // A wide lobe on the shore is the wet sheen.
    let view = normalize(frame.cam_pos.xyz - in.world);
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let halfv = normalize(normalize(frame.sun_dir.xyz) + view);
    let facing = pow(1.0 - ndotv, 2.0);
    let wide = pow(clamp(dot(n, halfv), 0.0, 1.0), 2.2);
    let wet_spec = (facing * 0.75 + wide * 1.15) * shore * mix(1.0, 0.16, drowned);
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
    // Dilate slightly so a dry speck inside the bowl does not punch a black hole.
    let e = 0.28;
    let d0 = mat_bilinear(xz).g;
    let d1 = mat_bilinear(xz + vec2<f32>(e, 0.0)).g;
    let d2 = mat_bilinear(xz - vec2<f32>(e, 0.0)).g;
    let d3 = mat_bilinear(xz + vec2<f32>(0.0, e)).g;
    let d4 = mat_bilinear(xz - vec2<f32>(0.0, e)).g;
    return max(d0, max(d1, max(d2, max(d3, d4)))) * 0.34;
}

// Fill a dry speck inside the bowl. A wide step turned the pond into a block.
fn covered_depth(xz: vec2<f32>) -> f32 {
    let here = water_depth(xz);
    let e = 0.42;
    let px = water_depth(xz + vec2<f32>(e, 0.0));
    let nx = water_depth(xz - vec2<f32>(e, 0.0));
    let pz = water_depth(xz + vec2<f32>(0.0, e));
    let nz = water_depth(xz - vec2<f32>(0.0, e));
    let across = max(min(px, nx), min(pz, nz));
    return max(here, across);
}

@vertex
fn vs_water(in: TerrIn) -> WaterOut {
    let soil = height_bilinear(in.xz);
    let depth = covered_depth(in.xz);
    let rip = ripple_height(in.xz, frame.misc.x) * smoothstep(0.0, 0.04, depth);
    // On the soil, in the hole. Averaging the bed lifted a flat card off the bowl.
    let y = soil + 0.02 + rip;
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
    if (in.depth < 0.004) {
        discard;
    }
    // Slope of the bed, so the pond shades as a bowl instead of a flat fill.
    let e = 0.55;
    let xz = in.world.xz;
    let slope = normalize(vec3<f32>(
        height_bilinear(xz - vec2<f32>(e, 0.0)) - height_bilinear(xz + vec2<f32>(e, 0.0)),
        e * 2.0,
        height_bilinear(xz - vec2<f32>(0.0, e)) - height_bilinear(xz + vec2<f32>(0.0, e)),
    ));
    let rip = ripple_normal(xz, frame.misc.x);
    let nrm = normalize(slope + vec3<f32>(rip.x, 0.0, rip.z) * 0.22);
    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let ndotv = clamp(dot(nrm, view), 0.0, 1.0);
    let fres = pow(1.0 - ndotv, 2.2);
    let depth_t = smoothstep(0.004, 0.055, in.depth);
    let shore_col = vec3<f32>(0.11, 0.048, 0.018);
    let deep_col = vec3<f32>(0.030, 0.013, 0.005);
    var col = mix(shore_col, deep_col, depth_t);
    let refl = sky_color(reflect(-view, nrm));
    col = mix(col, refl, fres * 0.34);
    let spec = pow(clamp(dot(nrm, normalize(sun + view)), 0.0, 1.0), 48.0);
    col += vec3<f32>(0.06, 0.045, 0.030) * spec * shadow_factor(in.world, nrm);
    let ndl = clamp(dot(nrm, sun), 0.0, 1.0);
    col += deep_col * ndl * 0.35;
    let alpha = smoothstep(0.003, 0.028, in.depth) * mix(0.62, 0.92, depth_t);
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
    let h = height_bilinear(in.xz);
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

struct ShadowCutOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) depth: f32,
    @location(1) uv: vec2<f32>,
    @location(2) panel: f32,
}

@vertex
fn vs_shadow_cut(in: MeshIn) -> ShadowCutOut {
    let world = model.world * vec4<f32>(in.pos, 1.0);
    let clip = frame.light_vp * world;
    var o: ShadowCutOut;
    o.clip = clip;
    o.depth = clip.z / max(clip.w, 1e-5);
    o.uv = in.uv;
    o.panel = in.panel;
    return o;
}

@fragment
fn fs_shadow_cut(in: ShadowCutOut) -> @location(0) vec4<f32> {
    if (in.panel > 70.5 && in.panel < 71.5) {
        let c = textureSample(flora_tex, paint_samp, in.uv);
        if (c.a < 0.12) {
            discard;
        }
    }
    return encode_depth(in.depth);
}

@fragment
fn fs_shadow(in: ShadowOut) -> @location(0) vec4<f32> {
    return encode_depth(in.depth);
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
        // Skirt. Grass and dry dirt, so the square does not end in a dark void.
        let g = noise(world.xz * 0.09);
        let col = mix(vec3<f32>(0.15, 0.13, 0.07), vec3<f32>(0.20, 0.17, 0.09), g);
        return vec4<f32>(col, 0.95);
    } else if (panel < 41.5) {
        let g = noise(world.xz * 0.05);
        let h = clamp(world.y / 3.4, 0.0, 1.0);
        let dirt = vec3<f32>(0.17, 0.13, 0.07);
        let grass = vec3<f32>(0.13, 0.16, 0.07);
        return vec4<f32>(mix(dirt, grass, h) * (0.86 + g * 0.28), 0.95);
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
    var alpha = 1.0;
    if (in.panel > 69.5 && in.panel < 70.5) {
        let c = textureSampleLevel(prop_tex, paint_samp, in.uv, 0.0);
        if (c.a < 0.2) {
            discard;
        }
        albedo = c.rgb * 0.8;
        rough = 0.84;
    } else if (in.panel > 70.5 && in.panel < 71.5) {
        let c = textureSample(flora_tex, paint_samp, in.uv);
        if (c.a < 0.12) {
            discard;
        }
        albedo = c.rgb;
        rough = 0.78;
    } else if (in.panel > 71.5 && in.panel < 72.5) {
        let c = textureSampleLevel(niva_tex, paint_samp, in.uv, 0.0).rgb;
        albedo = c.rgb;
        rough = 0.48;
        let ground = height_bilinear(in.world.xz);
        let splash = smoothstep(0.62, 0.16, in.world.y - ground);
        albedo = mix(albedo, vec3<f32>(0.07, 0.034, 0.016), splash * 0.62);
        rough = mix(rough, 0.22, splash * 0.5);
    } else if (in.panel > 72.5 && in.panel < 73.5) {
        let c = textureSampleLevel(niva_tex, paint_samp, in.uv, 0.0).rgb;
        albedo = mix(vec3<f32>(0.55, 0.74, 0.82), c, 0.35);
        rough = 0.06;
        alpha = 0.42;
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
        let mud = clamp(model.extra.y, 0.0, 1.2);
        let load = smoothstep(0.03, 0.38, mud);
        albedo = mix(albedo, vec3<f32>(0.10, 0.05, 0.022), load * 0.82);
        rough = mix(rough, 0.2, load * 0.65);
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
