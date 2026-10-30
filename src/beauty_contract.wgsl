// Picture side of the physics → render contract (docs/RENDER_CONTRACT.md,
// version 1). Appended to shader.wgsl when the device can bind the contract.
// These functions only READ the published maps. Nothing here feeds the solver.
//
// The contract window is 200 × 200 cells of 0.20 m that moves with the Niva.
// Outside it every reader returns the caller's fallback, so the fixed picture
// soil still shades where the contract has no texel.

struct BcHeader {
    version: u32,
    window_res: u32,
    air_res: u32,
    air_layers: u32,
    cell_m: f32,
    extent_m: f32,
    origin_x: f32,
    origin_z: f32,
    air_origin_x: f32,
    air_origin_y: f32,
    air_origin_z: f32,
    air_cell_x: f32,
    air_cell_y: f32,
    air_cell_z: f32,
    spray_count: u32,
    spray_capacity: u32,
    frame: u32,
    valid: u32,
    time_s: f32,
    center_x: f32,
    center_z: f32,
    flow_source: u32,
    dust_source: u32,
    debug_mode: u32,
}

@group(4) @binding(0) var<uniform> bc_header: BcHeader;
@group(4) @binding(1) var bc_ground_rut: texture_2d<f32>;
@group(4) @binding(2) var bc_moisture_tex: texture_2d<f32>;
@group(4) @binding(3) var bc_film_tex: texture_2d<f32>;
@group(4) @binding(4) var bc_flow_tex: texture_2d<f32>;
@group(4) @binding(5) var bc_dust_settled_tex: texture_2d<f32>;
@group(4) @binding(6) var bc_dust_fog: texture_3d<f32>;
@group(4) @binding(7) var bc_crust_tex: texture_2d<f32>;
@group(4) @binding(8) var bc_wet_edge_tex: texture_2d<f32>;
@group(4) @binding(9) var bc_foam_crest_tex: texture_2d<f32>;

const BC_VALID_MOISTURE: u32 = 2u;
const BC_VALID_FILM: u32 = 4u;
const BC_VALID_FLOW: u32 = 8u;
const BC_VALID_DUST_SETTLED: u32 = 16u;
const BC_VALID_DUST_FOG: u32 = 32u;
const BC_VALID_CRUST: u32 = 64u;
const BC_VALID_WET_EDGE: u32 = 128u;
const BC_VALID_FOAM: u32 = 256u;

fn bc_has(bit: u32) -> bool {
    return bc_header.version == 1u && (bc_header.valid & bit) != 0u;
}

// Texel coordinate of a world point: texel (i, j) is centred at
// origin + (i + 0.5, j + 0.5) * cell.
fn bc_texel(xz: vec2<f32>) -> vec2<f32> {
    let cell = max(bc_header.cell_m, 1e-4);
    return (xz - vec2<f32>(bc_header.origin_x, bc_header.origin_z)) / cell - vec2<f32>(0.5);
}

// 1 well inside the window, 0 outside, a two-cell fade at the rim.
fn bc_inside(xz: vec2<f32>) -> f32 {
    if (bc_header.version != 1u) {
        return 0.0;
    }
    let t = bc_texel(xz);
    let n = f32(bc_header.window_res);
    let d = min(min(t.x, t.y), min(n - 1.0 - t.x, n - 1.0 - t.y));
    return smoothstep(0.0, 2.0, d);
}

fn bc_load(tex: texture_2d<f32>, ij: vec2<i32>) -> vec4<f32> {
    let n = i32(bc_header.window_res);
    let c = clamp(ij, vec2<i32>(0), vec2<i32>(n - 1));
    return textureLoad(tex, c, 0);
}

// A texel that is not a finite number reads as zero. A NaN fails every
// comparison, so it never passes the range test. The fluid sheet publishes
// NaN velocity and crest in some dry cells (contract v1), and one NaN tap
// would blank the whole blend.
fn bc_finite(v: vec4<f32>) -> vec4<f32> {
    let ok = abs(v) < vec4<f32>(1.0e4);
    return select(vec4<f32>(0.0), v, ok);
}

// The maps are 32-bit float and not filterable; blend the four texels here.
fn bc_bilinear(tex: texture_2d<f32>, xz: vec2<f32>) -> vec4<f32> {
    let t = bc_texel(xz);
    let i = floor(t);
    let f = t - i;
    let i0 = vec2<i32>(i);
    let a = bc_finite(bc_load(tex, i0));
    let b = bc_finite(bc_load(tex, i0 + vec2<i32>(1, 0)));
    let c = bc_finite(bc_load(tex, i0 + vec2<i32>(0, 1)));
    let d = bc_finite(bc_load(tex, i0 + vec2<i32>(1, 1)));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn bc_moisture(xz: vec2<f32>, fallback: f32) -> f32 {
    if (!bc_has(BC_VALID_MOISTURE)) {
        return fallback;
    }
    return mix(fallback, clamp(bc_bilinear(bc_moisture_tex, xz).r, 0.0, 1.0), bc_inside(xz));
}

// Water and slurry thickness, metres.
fn bc_film(xz: vec2<f32>) -> vec2<f32> {
    if (!bc_has(BC_VALID_FILM)) {
        return vec2<f32>(0.0);
    }
    return max(bc_bilinear(bc_film_tex, xz).rg, vec2<f32>(0.0)) * bc_inside(xz);
}

// Film velocity, m/s, world +X and +Z. Zero without the fluid sheet.
fn bc_flow(xz: vec2<f32>) -> vec2<f32> {
    if (!bc_has(BC_VALID_FLOW)) {
        return vec2<f32>(0.0);
    }
    return bc_bilinear(bc_flow_tex, xz).rg * bc_inside(xz);
}

// 0/1 per cell; bilinear makes it a band one cell wide either side.
fn bc_wet_edge(xz: vec2<f32>) -> f32 {
    if (!bc_has(BC_VALID_WET_EDGE)) {
        return 0.0;
    }
    return bc_bilinear(bc_wet_edge_tex, xz).r * bc_inside(xz);
}

// Foam fraction 0..1 and crest height, metres.
fn bc_foam_crest(xz: vec2<f32>) -> vec2<f32> {
    if (!bc_has(BC_VALID_FOAM)) {
        return vec2<f32>(0.0);
    }
    return bc_bilinear(bc_foam_crest_tex, xz).rg * bc_inside(xz);
}

// Crust thickness and crack opening, metres. Written as 0 in version 1.
fn bc_crust(xz: vec2<f32>) -> vec2<f32> {
    if (!bc_has(BC_VALID_CRUST)) {
        return vec2<f32>(0.0);
    }
    return max(bc_bilinear(bc_crust_tex, xz).rg, vec2<f32>(0.0)) * bc_inside(xz);
}

const BC_VALID_GROUND: u32 = 1u;

// Rut depth below the starting height and placed spoil, metres.
fn bc_rut(xz: vec2<f32>) -> vec2<f32> {
    if (!bc_has(BC_VALID_GROUND)) {
        return vec2<f32>(0.0);
    }
    let g = bc_bilinear(bc_ground_rut, xz);
    return max(vec2<f32>(g.g, g.a), vec2<f32>(0.0)) * bc_inside(xz);
}

// Settled dust depth, metres.
fn bc_dust_settled(xz: vec2<f32>) -> f32 {
    if (!bc_has(BC_VALID_DUST_SETTLED)) {
        return 0.0;
    }
    return max(bc_bilinear(bc_dust_settled_tex, xz).r, 0.0) * bc_inside(xz);
}

// ---------------------------------------------------------------------------
// Airborne dust and fog as a volume. A fullscreen pass marches each pixel's
// ray through the published dust_fog grid up to the scene depth and blends the
// in-scattered light over the frame. Sun with its shadow map, sky fill, and
// the Niva headlamps.

// Extinction per unit volume fraction, 1/m. A fraction of 1e-4 is a thin
// haze (0.36 per metre), 1e-3 a cloud you lose the far side of.
const FOG_SIGMA: f32 = 3600.0;
const FOG_STEPS: i32 = 28;
const DUST_ALBEDO: vec3<f32> = vec3<f32>(0.62, 0.52, 0.40);

fn fog_tap(i: vec3<i32>) -> f32 {
    let res = vec3<i32>(i32(bc_header.air_res), i32(bc_header.air_layers), i32(bc_header.air_res));
    if (any(i < vec3<i32>(0)) || any(i >= res)) {
        return 0.0;
    }
    let v = textureLoad(bc_dust_fog, i, 0).r;
    return select(0.0, max(v, 0.0), abs(v) < 1.0e4);
}

// Volume fraction at a world point, trilinear over the published texels.
fn bc_fog_at(p: vec3<f32>) -> f32 {
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let origin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    let t = (p - origin) / max(cell, vec3<f32>(1e-4)) - vec3<f32>(0.5);
    let i = floor(t);
    let f = t - i;
    let i0 = vec3<i32>(i);
    let c000 = fog_tap(i0);
    let c100 = fog_tap(i0 + vec3<i32>(1, 0, 0));
    let c010 = fog_tap(i0 + vec3<i32>(0, 1, 0));
    let c110 = fog_tap(i0 + vec3<i32>(1, 1, 0));
    let c001 = fog_tap(i0 + vec3<i32>(0, 0, 1));
    let c101 = fog_tap(i0 + vec3<i32>(1, 0, 1));
    let c011 = fog_tap(i0 + vec3<i32>(0, 1, 1));
    let c111 = fog_tap(i0 + vec3<i32>(1, 1, 1));
    let x00 = mix(c000, c100, f.x);
    let x10 = mix(c010, c110, f.x);
    let x01 = mix(c001, c101, f.x);
    let x11 = mix(c011, c111, f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Nearest texel, for the cheap march toward the sun.
fn bc_fog_near(p: vec3<f32>) -> f32 {
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let origin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    return fog_tap(vec3<i32>(floor((p - origin) / max(cell, vec3<f32>(1e-4)))));
}

// Billows inside the published cells. The grid is 12.5 cm; a plume is
// rounded lobes and wisps, so the density is modulated by drifting noise
// that keeps the cell's mean. Picture only.
fn fog_billow(p: vec3<f32>, t: f32) -> f32 {
    let drift = vec2<f32>(0.9, -0.55) * t;
    let q = p.xz * 1.6 - drift + vec2<f32>(p.y * 0.9, -p.y * 0.6);
    let lobes = noise(q) * 0.62 + noise(q * 2.7 + vec2<f32>(5.2, 1.3)) * 0.38;
    let wisp = noise(q * 6.1 - drift * 1.7);
    return clamp(0.25 + lobes * 1.25 + (wisp - 0.5) * 0.35, 0.0, 2.0);
}

// Henyey-Greenstein. Dust throws most light forward.
fn hg_phase(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    let d = max(1.0 + g2 - 2.0 * g * cos_t, 1e-4);
    return (1.0 - g2) / (12.566371 * d * sqrt(d));
}

@vertex
fn vs_fog(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_fog(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    if (!bc_has(BC_VALID_DUST_FOG)) {
        discard;
    }
    let d = textureLoad(scene_depth, vec2<i32>(frag.xy), 0);
    let uv = frag.xy / max(frame.params.xy, vec2<f32>(1.0));
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d, 1.0);
    let w = frame.inv_view_proj * ndc;
    let ro = frame.cam_pos.xyz;
    let seg = w.xyz / w.w - ro;
    let dist = min(length(seg), 80.0);
    let rd = normalize(seg);

    // Clip the ray to the published box.
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let bmin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    let bmax = bmin + cell * vec3<f32>(f32(bc_header.air_res), f32(bc_header.air_layers), f32(bc_header.air_res));
    let inv = vec3<f32>(1.0) / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));
    let ta = (bmin - ro) * inv;
    let tb = (bmax - ro) * inv;
    let t_near = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
    let t_far = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), min(max(ta.z, tb.z), dist));
    if (t_far <= t_near) {
        discard;
    }

    let sun = normalize(frame.sun_dir.xyz);
    let sun_phase = hg_phase(dot(rd, sun), 0.55) * 12.566371;
    let sky = sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * frame.env.y;
    let lamp_on = frame.lamps[0].w;
    let dt = (t_far - t_near) / f32(FOG_STEPS);
    let jitter = hash12(frag.xy + vec2<f32>(frame.misc.x * 37.0, 0.0));
    var trans = 1.0;
    var light = vec3<f32>(0.0);
    for (var k: i32 = 0; k < FOG_STEPS; k = k + 1) {
        let p = ro + rd * (t_near + (f32(k) + jitter) * dt);
        let base = bc_fog_at(p);
        if (base * FOG_SIGMA < 1e-3) {
            continue;
        }
        let sigma = base * FOG_SIGMA * fog_billow(p, frame.misc.x);
        // Self-shadow: three coarse steps toward the sun through the cloud,
        // so the sunlit side glows and the far side goes grey.
        var od = 0.0;
        for (var j: i32 = 1; j <= 3; j = j + 1) {
            od += bc_fog_near(p + sun * (f32(j) * 0.35));
        }
        let sun_t = exp(-od * FOG_SIGMA * 0.35);
        var l_in = frame.sun_color.rgb * 1.25 * sun_phase * sun_t * shadow_factor(p, vec3<f32>(0.0, 1.0, 0.0)) + sky;
        if (lamp_on > 0.0) {
            for (var i: i32 = 0; i < 2; i = i + 1) {
                let b = lamp_beam(i, p);
                // The light travels along b.xyz; the camera sees it scattered back along -rd.
                l_in += LAMP_COLOR * LAMP_POWER * lamp_on * b.w * hg_phase(dot(b.xyz, -rd), 0.6) * 12.566371;
            }
        }
        let step_t = exp(-sigma * dt);
        light += trans * DUST_ALBEDO * l_in * (1.0 - step_t);
        trans *= step_t;
        if (trans < 0.01) {
            break;
        }
    }
    let alpha = 1.0 - trans;
    if (alpha < 0.002) {
        discard;
    }
    return vec4<f32>(tonemap(light / alpha), alpha);
}
