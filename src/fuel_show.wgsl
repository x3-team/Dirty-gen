// Filmed fuel. Dark glossy liquid with ripples, flame on that wet patch,
// gray smoke off the tips, and a short ground shock. No lava cracks.

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

fn sat(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn ripple_h(xz: vec2<f32>, t: f32) -> f32 {
    let a = sin(dot(xz, vec2<f32>(1.65, 0.72)) * 2.6 - t * 2.8);
    let b = sin(dot(xz, vec2<f32>(-0.55, 1.85)) * 3.1 - t * 2.1);
    let c = noise(xz * 1.45 + vec2<f32>(t * 0.18, -t * 0.12));
    return a * 0.45 + b * 0.35 + c * 0.5;
}

fn liquid_normal(world: vec3<f32>, t: f32) -> vec3<f32> {
    // Analytic slope only. Screen derivatives sparkle into a grid once the
    // sheet triangles shrink to a few pixels.
    let e = 0.16;
    let h0 = ripple_h(world.xz, t);
    let hx = (ripple_h(world.xz + vec2<f32>(e, 0.0), t) - h0) / e;
    let hz = (ripple_h(world.xz + vec2<f32>(0.0, e), t) - h0) / e;
    return normalize(vec3<f32>(-hx * 0.16, 1.0, -hz * 0.16));
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
    let t = frame.misc.x;
    // Derivatives before any discard so the quad stays uniform.
    let n = liquid_normal(in.world, t);
    let across = abs(in.uv.y - 0.5) * 2.0;
    let wob = (noise(in.world.xz * 1.6) - 0.5) * 0.14;
    let edge = smoothstep(1.04, 0.7, across + wob);
    if (edge < 0.02) {
        discard;
    }
    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let n_dot_v = sat(dot(n, view));
    let n_dot_l = sat(dot(n, sun));
    let halfv = normalize(sun + view);
    let n_dot_h = sat(dot(n, halfv));
    let f0 = 0.06;
    let fres = f0 + (1.0 - f0) * pow(1.0 - n_dot_v, 4.0);
    // Vertex red is heat. The body stays dark; heat is fire reflected in the wet patch.
    let heat = smoothstep(0.18, 0.7, in.color.r);
    let lip = smoothstep(0.66, 0.98, across);
    var col = vec3<f32>(0.05, 0.032, 0.022);
    col = mix(col, vec3<f32>(0.14, 0.04, 0.018), lip);
    let grain = 0.92 + 0.08 * noise(in.world.xz * 4.2);
    col *= grain;
    let glint = pow(n_dot_h, 48.0) * n_dot_l;
    let broken = smoothstep(0.72, 0.94, noise(in.world.xz * 1.3 + vec2<f32>(t * 0.2, 0.4)));
    col += vec3<f32>(0.55, 0.62, 0.7) * glint * broken;
    col += vec3<f32>(0.28, 0.34, 0.4) * fres * (0.25 + 0.75 * (1.0 - n.y));
    col += vec3<f32>(0.72, 0.22, 0.04) * heat * (0.08 + fres * 0.4);
    return vec4<f32>(col, alpha * edge);
}

fn fs_stream(in: FuelOut, alpha: f32) -> vec4<f32> {
    let across = abs(in.uv.x - 0.5) * 2.0;
    let edge = smoothstep(1.0, 0.42, across);
    if (edge < 0.03) {
        discard;
    }
    let t = frame.misc.x;
    let flow = fract(in.uv.y * 3.2 - t * 1.8);
    let thread = noise(vec2<f32>(in.uv.x * 6.0, flow * 7.0));
    let glint = smoothstep(0.72, 0.95, thread) * smoothstep(0.55, 0.15, across);
    var col = vec3<f32>(0.04, 0.025, 0.018);
    col += vec3<f32>(0.58, 0.64, 0.7) * glint;
    let rim = smoothstep(0.45, 0.95, across);
    col = mix(col, vec3<f32>(0.2, 0.045, 0.018), rim);
    return vec4<f32>(col, alpha * edge);
}

fn fs_ripple(in: FuelOut, alpha: f32) -> vec4<f32> {
    // Soft crest. The quad edge fades out so the ring is not a polygon outline.
    let band = 1.0 - abs(in.uv.x - 0.5) * 2.0;
    let crest = smoothstep(0.0, 0.72, band);
    let a = crest * crest * alpha;
    if (a < 0.02) {
        discard;
    }
    let col = vec3<f32>(0.58, 0.64, 0.7);
    return vec4<f32>(col, a);
}

fn fs_bead(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    if (d > 0.48) {
        discard;
    }
    let spec = pow(sat(1.0 - d * 1.7), 12.0);
    var col = vec3<f32>(0.05, 0.032, 0.022) + vec3<f32>(0.7, 0.76, 0.82) * spec;
    let rim = smoothstep(0.22, 0.48, d);
    col = mix(col, vec3<f32>(0.16, 0.04, 0.016), rim * 0.4);
    return vec4<f32>(col, alpha);
}

fn fs_flame(in: FuelOut, alpha: f32) -> vec4<f32> {
    let u = in.uv.x;
    let v = in.uv.y;
    let t = frame.misc.x;
    let n1 = noise(vec2<f32>(u * 3.2 + in.world.x * 0.7 + t * 1.6, v * 2.4 - t * 4.6));
    let n2 = noise(vec2<f32>(u * 7.5 - t * 2.2, v * 5.5 - t * 7.0 + in.world.z));
    let width = mix(0.42, 0.06, smoothstep(0.0, 1.0, v));
    let x = abs(u - 0.5);
    let side = smoothstep(width, width * 0.35, x + (n1 - 0.5) * 0.1);
    let tip = smoothstep(1.08, 0.18, v + (n1 - 0.45) * 0.42);
    let lick = mix(0.72, 1.0, smoothstep(0.25, 0.8, n2));
    let body = side * tip * lick;
    if (body < 0.04) {
        discard;
    }
    let base = vec3<f32>(0.96, 0.2, 0.02);
    let mid = vec3<f32>(1.0, 0.46, 0.05);
    let tipc = vec3<f32>(1.0, 0.88, 0.26);
    var col = mix(base, mid, smoothstep(0.0, 0.42, v));
    col = mix(col, tipc, smoothstep(0.38, 0.9, v));
    let core = smoothstep(0.2, 0.0, x) * smoothstep(0.62, 0.02, v);
    col = mix(col, vec3<f32>(1.0, 0.96, 0.72), core * 0.65);
    // Gray smoke only on the ragged tips.
    let smoke = smoothstep(0.7, 1.0, v) * smoothstep(0.25, 0.75, 1.0 - n2);
    col = mix(col, vec3<f32>(0.66, 0.68, 0.7), smoke);
    let a = alpha * body * mix(0.95, 0.4, smoothstep(0.15, 0.95, v)) * (1.0 - smoke * 0.35);
    return vec4<f32>(col, a);
}

fn fs_reflect(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    let a = smoothstep(0.52, 0.06, d) * alpha;
    if (a < 0.03) {
        discard;
    }
    return vec4<f32>(in.color.rgb, a);
}

fn fs_smoke(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    let puff = smoothstep(0.5, 0.0, d);
    let a = alpha * puff * puff;
    if (a < 0.03) {
        discard;
    }
    return vec4<f32>(in.color.rgb, a);
}

fn fs_spark(in: FuelOut, alpha: f32) -> vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    if (d > 0.46) {
        discard;
    }
    let core = smoothstep(0.46, 0.0, d);
    let col = mix(vec3<f32>(1.0, 0.55, 0.12), vec3<f32>(1.0, 0.98, 0.88), core);
    return vec4<f32>(col, alpha * core);
}

// Ground shock: uv.y runs from the trailing dust to the bright front.
fn fs_ball(in: FuelOut, alpha: f32) -> vec4<f32> {
    let n = noise(in.world.xz * 5.5 + vec2<f32>(frame.misc.x * 0.4, 2.0));
    let lead = smoothstep(0.55, 0.9, in.uv.y) * smoothstep(1.0, 0.84, in.uv.y);
    let body = smoothstep(0.0, 0.2, in.uv.y) * smoothstep(1.0, 0.4, in.uv.y);
    var col = vec3<f32>(0.4, 0.32, 0.24);
    col = mix(col, vec3<f32>(0.72, 0.64, 0.52), lead);
    col *= 0.82 + 0.18 * n;
    let a = alpha * (body * 0.8 + lead * 0.85) * (0.7 + 0.3 * n);
    if (a < 0.03) {
        discard;
    }
    return vec4<f32>(col, a);
}

fn fs_streak(in: FuelOut, alpha: f32) -> vec4<f32> {
    let side = smoothstep(0.0, 0.2, in.uv.x) * smoothstep(1.0, 0.8, in.uv.x);
    let along = smoothstep(0.0, 0.08, in.uv.y);
    let col = mix(vec3<f32>(1.0, 0.94, 0.7), vec3<f32>(1.0, 0.46, 0.08), in.uv.y);
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
