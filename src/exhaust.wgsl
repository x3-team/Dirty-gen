// Exhaust wisps and the heat shimmer just aft of the tailpipe.
// Smoke is a cool gray streak. The haze pass only moves texels.

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

@vertex
fn vs_smoke(in: Vin) -> Vout {
    let right = smoke.cam_right.xyz;
    let up = smoke.cam_up.xyz;
    let vel = in.vel.xyz;
    let v_right = dot(vel, right);
    let v_up = dot(vel, up);
    let v_len = length(vec2<f32>(v_right, v_up));
    var axis = up;
    var sidev = right;
    if (v_len > 0.08) {
        axis = (right * v_right + up * v_up) / v_len;
        sidev = right * v_up - up * v_right;
        sidev = sidev / max(length(sidev), 0.0001);
    }
    var width = max(in.geom.x, 0.01);
    var len = width * (1.25 + clamp(length(vel), 0.0, 1.2) * 0.35);
    let c0 = smoke.view_proj * vec4<f32>(in.pos.xyz, 1.0);
    let c1 = smoke.view_proj * vec4<f32>(in.pos.xyz + sidev * width, 1.0);
    let px = length((c1.xy / max(c1.w, 0.0001) - c0.xy / max(c0.w, 0.0001)) * vec2<f32>(smoke.params.y, smoke.params.z) * 0.5);
    // Wide shots still need a readable wisp. Close-ups already clear this floor.
    if (px > 0.25 && px < 4.5) {
        let s = 4.5 / px;
        width = width * s;
        len = len * s;
    }
    // A puff that reaches the lens stays a wisp, not a white disc.
    if (px > 36.0) {
        let s = 36.0 / px;
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
    let q = (in.uv - vec2<f32>(0.5)) * vec2<f32>(1.2, 0.7);
    let d = length(q);
    var a = smoothstep(0.52, 0.04, d);
    let strand = noise(vec2<f32>(in.uv.x * 4.0 + in.seed * 9.0, in.uv.y * 2.4 + smoke.params.x * 0.2));
    let gap = noise(vec2<f32>(in.uv.y * 2.6 - smoke.params.x * 0.3, in.seed * 11.0));
    // Soft breaks, not dashed strokes. The wisp stays thin because alpha is low.
    a = a * mix(0.55, 1.0, smoothstep(0.15, 0.75, strand * 0.65 + gap * 0.35)) * in.color.a;
    if (a < 0.02) {
        discard;
    }
    return vec4<f32>(in.color.rgb, a);
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
    let floor_px = 12.0 * (haze.params.z / 720.0);
    let r0 = max(length(mouth.xy - a.xy), floor_px);
    let r1 = max(r0 * (radii.y / max(radii.x, 0.02)), r0);
    let r = mix(r0, r1, t);
    let fall = smoothstep(r, r * 0.12, dist);
    if (fall < 0.001) {
        return vec2<f32>(0.0);
    }
    let dir = ab / sqrt(ab_len2);
    let perp = vec2<f32>(-dir.y, dir.x);
    let time = haze.params.x;
    let n1 = noise(px * 0.065 + vec2<f32>(time * 1.5, -time * 2.6));
    let n2 = noise(px * 0.028 + vec2<f32>(-time * 0.55, time * 0.4));
    let amp = fall * strength * mix(1.0, 0.22, t) * 11.0 * (haze.params.z / 720.0);
    return perp * (n1 - 0.5) * amp + dir * (n2 - 0.5) * amp * 0.28;
}

@fragment
fn fs_haze(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = frag.xy;
    var off = one_plume(haze.plumes[0].origin_s, haze.plumes[0].axis_len, haze.plumes[0].radii, px);
    off = off + one_plume(haze.plumes[1].origin_s, haze.plumes[1].axis_len, haze.plumes[1].radii, px);
    off = off + one_plume(haze.plumes[2].origin_s, haze.plumes[2].axis_len, haze.plumes[2].radii, px);
    let mag = max(length(off), 0.0001);
    let cap = 6.5 * (haze.params.z / 720.0);
    off = off * min(cap / mag, 1.0);
    let uv = clamp((px + off) / haze.params.yz, vec2<f32>(0.001), vec2<f32>(0.999));
    return textureSample(scene_tex, scene_samp, uv);
}
