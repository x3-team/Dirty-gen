// Volumetric exhaust. Each card is a soft gray-blue wisp with density
// inside it, not a white dash. The haze pass refracts the scene through
// the hot column: a small bend plus a faint color split, no flame tint.

struct SmokeFrame {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    params: vec4<f32>, // time, width, height, _
}

struct Vin {
    @location(0) pos: vec4<f32>,
    @location(1) vel: vec4<f32>,
    @location(2) geom: vec4<f32>, // size, alpha, corner.x, corner.y
    @location(3) color_seed: vec4<f32>,
}

struct Vout {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) seed: f32,
}

@group(0) @binding(0) var<uniform> smoke: SmokeFrame;

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

fn fbm(p: vec2<f32>) -> f32 {
    let a = noise(p);
    let b = noise(p * 2.13 + vec2<f32>(1.7, 9.2));
    let c = noise(p * 4.07 + vec2<f32>(8.3, 2.8));
    return a * 0.55 + b * 0.30 + c * 0.15;
}

@vertex
fn vs_smoke(in: Vin) -> Vout {
    let right = smoke.cam_right.xyz;
    let up = smoke.cam_up.xyz;
    // A volume card, a little taller than it is wide, leaning with the curl.
    // Velocity does not stretch it into a streak.
    let vel = in.vel.xyz;
    let lean = clamp(dot(vel, right) * 1.1, -0.4, 0.4);
    var axis = up + right * lean;
    axis = axis / max(length(axis), 0.0001);
    var sidev = right - axis * dot(right, axis);
    sidev = sidev / max(length(sidev), 0.0001);

    var width = max(in.geom.x, 0.02);
    // Almost round. A tall card reads as a dash once the wisp is thin.
    var len = width * 1.22;
    let c0 = smoke.view_proj * vec4<f32>(in.pos.xyz, 1.0);
    let c1 = smoke.view_proj * vec4<f32>(in.pos.xyz + sidev * width, 1.0);
    let px = length((c1.xy / max(c1.w, 0.0001) - c0.xy / max(c0.w, 0.0001)) * vec2<f32>(smoke.params.y, smoke.params.z) * 0.5);
    // Only a puff that has reached the lens is capped, so a close wisp
    // stays a volume and a far one is not inflated into a white tick.
    if (px > 150.0) {
        let s = 150.0 / px;
        width = width * s;
        len = len * s;
    }
    let corner = in.geom.zw;
    let world = in.pos.xyz + axis * corner.y * len + sidev * corner.x * width;
    var o: Vout;
    o.clip = smoke.view_proj * vec4<f32>(world, 1.0);
    o.uv = corner + vec2<f32>(0.5);
    o.color = vec4<f32>(in.color_seed.xyz, in.geom.y);
    o.seed = in.color_seed.w;
    return o;
}

@fragment
fn fs_smoke(in: Vout) -> @location(0) vec4<f32> {
    let t = smoke.params.x;
    let seed = in.seed;
    let flow = vec2<f32>(in.uv.x * 2.5 + seed * 5.2, in.uv.y * 2.7 + seed * 1.4 - t * 0.12);
    let w1 = fbm(flow);
    let w2 = fbm(flow * 1.8 + vec2<f32>(3.4, 7.1));
    var p = in.uv - vec2<f32>(0.5);
    // Warp the ball into an irregular wisp. The body stays continuous.
    p = p + vec2<f32>(w1 - 0.5, w2 - 0.5) * 0.30;
    let q = p * vec2<f32>(1.18, 1.02);
    let body = exp(-dot(q, q) * 5.2);
    let grain = smoothstep(0.20, 0.80, fbm(flow * 2.1 + vec2<f32>(w1, t * 0.08)));
    // Feather the card itself so the quad corners never read as a square.
    let mask = smoothstep(0.50, 0.16, length((in.uv - vec2<f32>(0.5)) * vec2<f32>(1.02, 0.90)));
    // The floor keeps a soft volume. Grain only models the inside of it.
    let density = body * mix(0.50, 1.0, grain) * (0.14 + in.color.a) * mask;
    if (density < 0.018) {
        discard;
    }
    let rgb = in.color.rgb * mix(vec3<f32>(0.93, 0.98, 1.06), vec3<f32>(1.0, 1.02, 1.04), grain);
    return vec4<f32>(min(rgb, vec3<f32>(0.86, 0.90, 0.96)), density);
}

struct Plume {
    origin_s: vec4<f32>, // xyz origin, w strength
    axis_len: vec4<f32>, // xyz axis, w length
    radii: vec4<f32>,    // x r0, y r1
}

struct HazeFrame {
    view_proj: mat4x4<f32>,
    cam_right: vec4<f32>,
    params: vec4<f32>, // time, width, height, _
    plumes: array<Plume, 3>,
}

@group(0) @binding(0) var<uniform> haze: HazeFrame;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_samp: sampler;

@vertex
fn vs_haze(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn project_px(p: vec3<f32>) -> vec3<f32> {
    let c = haze.view_proj * vec4<f32>(p, 1.0);
    let ndc = c.xy / max(c.w, 0.0001);
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    return vec3<f32>(uv * haze.params.yz, c.w);
}

fn one_plume(origin_s: vec4<f32>, axis_len: vec4<f32>, radii: vec4<f32>, px: vec2<f32>) -> vec2<f32> {
    let strength = origin_s.w;
    if (strength < 0.02) {
        return vec2<f32>(0.0);
    }
    let origin = origin_s.xyz;
    let axis = axis_len.xyz;
    let len = max(axis_len.w, 0.05);
    let a = project_px(origin);
    let b = project_px(origin + axis * len);
    if (a.z < 0.05 || b.z < 0.05) {
        return vec2<f32>(0.0);
    }
    let ab = b.xy - a.xy;
    let ab_len2 = max(dot(ab, ab), 0.001);
    let t = clamp(dot(px - a.xy, ab) / ab_len2, 0.0, 1.0);
    let q = a.xy + ab * t;
    let dist = length(px - q);
    let mouth = project_px(origin + haze.cam_right.xyz * max(radii.x, 0.02));
    let floor_px = 8.0 * (haze.params.z / 720.0);
    let r0 = max(length(mouth.xy - a.xy), floor_px);
    let r1 = max(r0 * (radii.y / max(radii.x, 0.02)), r0);
    let r = mix(r0, r1, t);
    let fall = smoothstep(r, r * 0.15, dist);
    if (fall < 0.001) {
        return vec2<f32>(0.0);
    }
    let dir = ab / sqrt(ab_len2);
    let perp = vec2<f32>(-dir.y, dir.x);
    let time = haze.params.x;
    // Broad heat cells, plus a finer ripple, so the column reads as a volume.
    let n1 = fbm(px * 0.014 + vec2<f32>(time * 0.45, -time * 0.70));
    let n2 = noise(px * 0.032 + vec2<f32>(-time * 1.1, time * 0.55) + vec2<f32>(n1, 3.0));
    let cell = (n1 - 0.5) * 0.70 + (n2 - 0.5) * 0.30;
    let amp = fall * strength * mix(0.62, 1.0, n1) * mix(1.0, 0.45, t);
    // A few pixels at the lip, softer up the column. Enough to bend an edge.
    let scale = 16.0 * (haze.params.z / 720.0);
    return perp * cell * amp * scale + dir * (n2 - 0.5) * amp * scale * 0.30;
}

@fragment
fn fs_haze(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = frag.xy;
    var off = one_plume(haze.plumes[0].origin_s, haze.plumes[0].axis_len, haze.plumes[0].radii, px);
    off = off + one_plume(haze.plumes[1].origin_s, haze.plumes[1].axis_len, haze.plumes[1].radii, px);
    off = off + one_plume(haze.plumes[2].origin_s, haze.plumes[2].axis_len, haze.plumes[2].radii, px);
    let mag = max(length(off), 0.0001);
    // Cap the bend so the refraction stays a shimmer.
    let cap = 7.5 * (haze.params.z / 720.0);
    off = off * min(cap / mag, 1.0);
    let base = px / haze.params.yz;
    let delta = off / haze.params.yz;
    let edge = vec2<f32>(0.001);
    let limit = vec2<f32>(0.999);
    let uv_r = clamp(base + delta * 1.35, edge, limit);
    let uv_g = clamp(base + delta, edge, limit);
    let uv_b = clamp(base + delta * 0.62, edge, limit);
    let sample_r = textureSample(scene_tex, scene_samp, uv_r);
    let sample_g = textureSample(scene_tex, scene_samp, uv_g);
    let sample_b = textureSample(scene_tex, scene_samp, uv_b);
    return vec4<f32>(sample_r.r, sample_g.g, sample_b.b, sample_g.a);
}
