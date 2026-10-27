// Game fuel show. Dark glossy liquid, flame on that liquid, a short ground blast.
// The mesh is rebuilt from the live ribbon each frame. This is not a flipbook.

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
    params: vec4<f32>,
    misc: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;

struct FuelIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
}

struct FuelOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world: vec3<f32>,
}

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
fn vs_fuel(in: FuelIn) -> FuelOut {
    var o: FuelOut;
    o.clip = frame.view_proj * vec4<f32>(in.pos, 1.0);
    o.color = in.color;
    o.uv = in.uv;
    o.world = in.pos;
    return o;
}

fn fs_pool(in: FuelOut, alpha: f32) -> vec4<f32> {
    // uv.y is 0.5 down the middle of the sheet and 0 or 1 on the rim.
    let across = abs(in.uv.y - 0.5) * 2.0;
    // Hard lip. The mud's wet edge fades; this one cuts.
    let edge = smoothstep(1.0, 0.84, across);
    if (edge < 0.02) {
        discard;
    }
    let t = frame.misc.x;
    let n2 = noise(in.world.xz * 2.6 + vec2<f32>(t * 0.45, 1.3));
    let streak = smoothstep(0.62, 0.92, noise(in.world.xz * 1.55 + vec2<f32>(t * 0.32, 3.4)));
    let view = normalize(frame.cam_pos.xyz - in.world);
    let n = normalize(vec3<f32>(
        (noise(in.world.xz * 3.2 + vec2<f32>(0.15, t * 0.2)) - n2) * 1.6,
        1.0,
        (noise(in.world.xz * 3.2 + vec2<f32>(t * 0.2, 0.4)) - n2) * 1.6
    ));
    let facing = clamp(dot(n, view), 0.0, 1.0);
    let sun = normalize(frame.sun_dir.xyz);
    let spec = pow(clamp(dot(n, normalize(sun + view)), 0.0, 1.0), 36.0);
    var col = in.color.rgb;
    // Dark body, a few cool glints. The sheet stays darker than the mud around it.
    col += vec3<f32>(0.82, 0.88, 0.94) * streak * facing * 0.4;
    col += vec3<f32>(1.0, 0.97, 0.92) * spec * 1.15;
    let fres = pow(1.0 - facing, 2.4);
    col += vec3<f32>(0.55, 0.62, 0.72) * fres * 0.12;
    let lip = smoothstep(0.68, 0.94, across);
    col = mix(col, vec3<f32>(0.46, 0.045, 0.016), lip);
    col *= mix(1.0, 0.42, smoothstep(0.94, 1.0, across));
    return vec4<f32>(col, alpha * edge);
}

fn fs_stream(in: FuelOut, alpha: f32) -> vec4<f32> {
    let across = abs(in.uv.x - 0.5) * 2.0;
    let edge = smoothstep(1.0, 0.58, across);
    if (edge < 0.03) {
        discard;
    }
    let flow = fract(in.uv.y * 1.8 - frame.misc.x * 1.45);
    let thread = smoothstep(0.40, 0.88, noise(vec2<f32>(in.uv.x * 8.0, flow * 5.5)));
    let rim = smoothstep(0.28, 0.95, across);
    var col = mix(vec3<f32>(0.012, 0.008, 0.007), vec3<f32>(0.52, 0.04, 0.016), rim);
    col += vec3<f32>(0.8, 0.86, 0.92) * thread * (1.0 - across) * 0.32;
    return vec4<f32>(col, alpha * edge * (0.72 + 0.28 * thread));
}

fn fs_ripple(in: FuelOut, alpha: f32) -> vec4<f32> {
    let band = 1.0 - abs(in.uv.x - 0.5) * 2.0;
    let a = smoothstep(0.0, 0.4, band) * alpha;
    if (a < 0.025) {
        discard;
    }
    return vec4<f32>(vec3<f32>(0.82, 0.88, 0.94), a);
}

fn fs_bead(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    if (d > 0.48) {
        discard;
    }
    let spec = pow(clamp(1.0 - d * 1.7, 0.0, 1.0), 8.0);
    let col = vec3<f32>(0.02, 0.014, 0.012) + vec3<f32>(0.85, 0.9, 0.95) * spec;
    return vec4<f32>(col, alpha);
}

fn fs_flame(in: FuelOut, alpha: f32) -> vec4<f32> {
    let u = in.uv.x;
    let v = in.uv.y;
    let t = frame.misc.x;
    let flick = noise(vec2<f32>(u * 4.0 + t * 1.3, v * 2.8 - t * 5.2));
    let side = smoothstep(0.0, 0.16, u) * smoothstep(1.0, 0.82, u);
    let top = smoothstep(1.05, 0.22, v + (flick - 0.42) * 0.4);
    let body = side * top;
    if (body < 0.035) {
        discard;
    }
    let tip = smoothstep(0.12, 0.9, v);
    var col = mix(vec3<f32>(0.98, 0.22, 0.015), vec3<f32>(1.0, 0.8, 0.12), tip);
    col = mix(col, vec3<f32>(1.0, 0.97, 0.6), smoothstep(0.58, 1.0, v) * flick);
    return vec4<f32>(col, alpha * body);
}

fn fs_reflect(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    let a = smoothstep(0.52, 0.06, d) * alpha;
    if (a < 0.03) {
        discard;
    }
    let col = mix(in.color.rgb, vec3<f32>(1.0, 0.7, 0.18), smoothstep(0.42, 0.02, d));
    return vec4<f32>(col, a);
}

fn fs_smoke(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    let puff = smoothstep(0.50, 0.08, d);
    if (puff < 0.03) {
        discard;
    }
    let wob = noise(in.uv * 3.2 + vec2<f32>(frame.misc.x * 0.35, 0.4));
    return vec4<f32>(in.color.rgb * (0.82 + 0.18 * wob), alpha * puff);
}

fn fs_spark(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    if (d > 0.46) {
        discard;
    }
    let core = smoothstep(0.46, 0.0, d);
    let col = mix(vec3<f32>(1.0, 0.5, 0.08), vec3<f32>(1.0, 0.98, 0.86), core);
    return vec4<f32>(col, alpha * core);
}

fn fs_ball(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    let t = frame.misc.x;
    let n = noise(in.uv * 5.0 + vec2<f32>(t * 6.5, t * 2.2));
    let rad = 0.40 + n * 0.07;
    if (d > rad) {
        discard;
    }
    let hot = smoothstep(rad, 0.02, d);
    var col = mix(vec3<f32>(0.72, 0.1, 0.012), vec3<f32>(1.0, 0.88, 0.4), hot);
    col = mix(col, vec3<f32>(1.0, 0.97, 0.78), smoothstep(0.5, 1.0, hot));
    return vec4<f32>(col, alpha * smoothstep(rad, rad * 0.32, d));
}

fn fs_streak(in: FuelOut, alpha: f32) -> vec4<f32> {
    let side = smoothstep(0.0, 0.18, in.uv.x) * smoothstep(1.0, 0.82, in.uv.x);
    let along = smoothstep(0.0, 0.06, in.uv.y);
    let col = mix(vec3<f32>(1.0, 0.96, 0.72), vec3<f32>(1.0, 0.42, 0.06), in.uv.y);
    return vec4<f32>(col, alpha * side * along);
}

@fragment
fn fs_fuel(in: FuelOut) -> @location(0) vec4<f32> {
    let scaled = in.color.a * 0.5;
    let style = floor(scaled + 0.001);
    let alpha = fract(scaled) * 2.0;
    if (style < 0.5) {
        return fs_pool(in, alpha);
    }
    if (style < 1.5) {
        return fs_stream(in, alpha);
    }
    if (style < 2.5) {
        return fs_ripple(in, alpha);
    }
    if (style < 3.5) {
        return fs_bead(in, alpha);
    }
    if (style < 4.5) {
        return fs_flame(in, alpha);
    }
    if (style < 5.5) {
        return fs_reflect(in, alpha);
    }
    if (style < 6.5) {
        return fs_smoke(in, alpha);
    }
    if (style < 7.5) {
        return fs_spark(in, alpha);
    }
    if (style < 8.5) {
        return fs_ball(in, alpha);
    }
    return fs_streak(in, alpha);
}
