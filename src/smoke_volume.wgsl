// Smoke and fire volume: one fullscreen ray march over the `smoke.wgsl` grid.
//
// Single scattering of sun and sky through soot, with the sun transmittance
// the solver already marched per cell (texture .a) and the scene shadow map,
// so a body in the smoke throws a shadow into it. Flame light is black-body
// emission from the temperature. Lighting model after the approved Bevy
// volumetric_fog.wgsl (MIT/Apache): Henyey-Greenstein phase, Beer-Lambert,
// energy-conserving step integration (Hillaire 2015).
//
// When the dust/fog march lands on main, `smoke_sigma` and the emission term
// move into that loop and this pass goes away.

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
    tires: array<vec4<f32>, 8>,
    snow_arc: vec4<f32>,
    snow_arc2: vec4<f32>,
    snow_profile: array<vec4<f32>, 8>,
}

struct SmokeDraw {
    // grid min corner xyz, cell m
    origin: vec4<f32>,
    // grid size m xyz, extinction per metre at soot 1
    size: vec4<f32>,
    // soot albedo rgb, phase g
    albedo: vec4<f32>,
    // emission gain, ambient gain, kelvin per LUT step, LUT steps
    light: vec4<f32>,
    // Black body: linear rgb (max 1), relative power. Entry k is at k * light.z kelvin over ambient.
    lut: array<vec4<f32>, 24>,
    // Fire lights: position xyz, radius; colour rgb (linear, W-ish), _.
    fire_pos: array<vec4<f32>, 4>,
    fire_col: array<vec4<f32>, 4>,
    // number of fire lights, detail strength, detail scale 1/m, rise speed m/s
    detail: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var<uniform> sd: SmokeDraw;
@group(1) @binding(1) var vol: texture_3d<f32>;
@group(1) @binding(2) var vol_samp: sampler;
@group(1) @binding(3) var scene_depth: texture_depth_2d;
@group(1) @binding(4) var shadow_tex: texture_2d<f32>;

override QUALITY_SMOKE_STEPS: i32 = 64;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_smoke(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

fn decode_depth(c: vec4<f32>) -> f32 {
    return dot(c, vec4<f32>(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));
}

// One tap of the sun shadow map: 1 lit, 0 behind the Niva or a prop.
fn scene_sun(p: vec3<f32>) -> f32 {
    let lp = frame.light_vp * vec4<f32>(p, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let dims = vec2<f32>(textureDimensions(shadow_tex));
    let c = clamp(vec2<i32>(uv * dims), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let s = decode_depth(textureLoad(shadow_tex, c, 0));
    return select(1.0, 0.0, ndc.z > s + 0.0006);
}

fn hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5));
}

fn blackbody(dt_k: f32) -> vec4<f32> {
    let n = i32(sd.light.w);
    let x = clamp(dt_k / sd.light.z, 0.0, f32(n - 1));
    let k = min(i32(x), n - 2);
    return mix(sd.lut[k], sd.lut[k + 1], x - f32(k));
}

fn hash3(p: vec3<f32>) -> f32 {
    let q = fract(p * 0.3183099 + vec3<f32>(0.1, 0.2, 0.3)) * 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

// Value noise, 0..1.
fn vnoise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = p - i;
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(mix(hash3(i), hash3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
                mix(hash3(i + vec3<f32>(0.0, 1.0, 0.0)), hash3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x), u.y);
    let b = mix(mix(hash3(i + vec3<f32>(0.0, 0.0, 1.0)), hash3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
                mix(hash3(i + vec3<f32>(0.0, 1.0, 1.0)), hash3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x), u.y);
    return mix(a, b, u.z);
}

// Sub-grid detail. The grid resolves the column and its large rolls; below
// one cell the edge is eroded by two octaves of noise that rise with the
// plume (Schneider 2015 cloud erosion). Dense cores keep their density.
fn detail_sigma(p: vec3<f32>, sigma: f32) -> f32 {
    let k = sd.detail.y;
    if (k <= 0.0) {
        return sigma;
    }
    let q = (p - vec3<f32>(0.0, frame.misc.x * sd.detail.w, 0.0)) * sd.detail.z;
    let n = vnoise(q) * 0.65 + vnoise(q * 2.7 + vec3<f32>(3.1, 1.7, 5.3)) * 0.35;
    // Thin smoke erodes, thick smoke barely.
    let edge = exp(-sigma * 0.6);
    let er = clamp((n - 0.5 * edge * k) / max(1.0 - 0.5 * edge * k, 1e-3), 0.0, 1.0);
    return sigma * mix(1.0, er * 1.6, edge * k);
}

fn fire_light(p: vec3<f32>) -> vec3<f32> {
    var l = vec3<f32>(0.0);
    let n = i32(sd.detail.x);
    for (var k: i32 = 0; k < 4; k = k + 1) {
        if (k >= n) {
            break;
        }
        let d = p - sd.fire_pos[k].xyz;
        let r2 = dot(d, d) + sd.fire_pos[k].w * sd.fire_pos[k].w;
        l = l + sd.fire_col[k].rgb / r2;
    }
    return l * (1.0 / (4.0 * 3.14159265));
}

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let mapped = vec3<f32>(1.0) - exp(-max(c, vec3<f32>(0.0)) * 1.08);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_smoke(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let res = frame.params.xy;
    let ndc = vec2<f32>(in.pos.x / res.x * 2.0 - 1.0, 1.0 - in.pos.y / res.y * 2.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ro = frame.cam_pos.xyz;
    let rd = normalize(far.xyz / far.w - ro);

    // Stop at the opaque scene.
    let d = textureLoad(scene_depth, px, 0);
    let hit = frame.inv_view_proj * vec4<f32>(ndc, d, 1.0);
    let t_scene = select(1e6, length(hit.xyz / hit.w - ro), d < 1.0);

    let bmin = sd.origin.xyz;
    let bmax = sd.origin.xyz + sd.size.xyz;
    let inv = 1.0 / rd;
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;
    let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    let t_near = max(tn, 0.0);
    let t_far = min(tf, t_scene);
    if (t_far <= t_near) {
        discard;
    }

    let steps = QUALITY_SMOKE_STEPS;
    let dt = (t_far - t_near) / f32(steps);
    let jitter = hash(in.pos.xy + vec2<f32>(frame.misc.x * 13.1, 0.0));
    let sun = normalize(frame.sun_dir.xyz);
    let phase = hg(dot(rd, sun), sd.albedo.w);
    // Sky light from above and the ground bounce from below, flat.
    let sky = mix(frame.amb.rgb, frame.sky_zenith.rgb, 0.6) * sd.light.y;

    var trans = 1.0;
    var scat = vec3<f32>(0.0);
    var emit = vec3<f32>(0.0);
    for (var k: i32 = 0; k < 64; k = k + 1) {
        if (k >= steps || trans < 0.01) {
            break;
        }
        let t = t_near + (f32(k) + jitter) * dt;
        let p = ro + rd * t;
        let uvw = (p - bmin) / sd.size.xyz;
        let s = textureSampleLevel(vol, vol_samp, uvw, 0.0);
        let sigma = detail_sigma(p, max(s.r, 0.0) * sd.size.w);
        let flame = max(s.b, 0.0);
        if (sigma < 1e-4 && flame < 1e-3) {
            continue;
        }
        let step_t = exp(-sigma * dt);
        // Energy-conserving: integrate the in-scatter over the step.
        let w = select(dt, (1.0 - step_t) / max(sigma, 1e-5), sigma > 1e-5);
        // Sun: single scattering plus a softer multiple-scattering octave
        // (Wrenninge 2015: lower extinction, flatter phase).
        let lit = scene_sun(p);
        let ms = pow(max(s.a, 0.0), 0.25) * 0.5 * (1.0 / (4.0 * 3.14159265));
        let sun_l = frame.sun_color.rgb * lit * (s.a * phase + ms);
        // Height above the floor darkens the sky term a little at the base.
        scat = scat + trans * sigma * w * sd.albedo.rgb * (sun_l + sky + fire_light(p));
        // Flame: the burning gas glows by its temperature.
        let bb = blackbody(s.g * 1000.0);
        emit = emit + trans * bb.rgb * bb.w * flame * sd.light.x * dt;
        trans = trans * step_t;
    }
    let alpha = 1.0 - trans;
    if (alpha < 0.002 && max(emit.r, max(emit.g, emit.b)) < 0.002) {
        discard;
    }
    // Premultiplied: smoke colour over the tonemapped scene, flame on top.
    let smoke = tonemap(scat / max(alpha, 1e-3)) * alpha;
    let glow = tonemap(emit);
    return vec4<f32>(min(smoke + glow, vec3<f32>(1.0)), alpha);
}
