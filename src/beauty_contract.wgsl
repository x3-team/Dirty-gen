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
// Air velocity, m/s, same origin and cell as dust_fog (contract v2).
@group(4) @binding(11) var bc_air_tex: texture_3d<f32>;

const BC_VALID_MOISTURE: u32 = 2u;
const BC_VALID_FILM: u32 = 4u;
const BC_VALID_FLOW: u32 = 8u;
const BC_VALID_DUST_SETTLED: u32 = 16u;
const BC_VALID_DUST_FOG: u32 = 32u;
const BC_VALID_CRUST: u32 = 64u;
const BC_VALID_WET_EDGE: u32 = 128u;
const BC_VALID_FOAM: u32 = 256u;

fn bc_has(bit: u32) -> bool {
    return bc_header.version >= 1u && (bc_header.valid & bit) != 0u;
}

// Texel coordinate of a world point: texel (i, j) is centred at
// origin + (i + 0.5, j + 0.5) * cell.
fn bc_texel(xz: vec2<f32>) -> vec2<f32> {
    let cell = max(bc_header.cell_m, 1e-4);
    return (xz - vec2<f32>(bc_header.origin_x, bc_header.origin_z)) / cell - vec2<f32>(0.5);
}

// 1 well inside the window, 0 outside, a two-cell fade at the rim.
fn bc_inside(xz: vec2<f32>) -> f32 {
    if (bc_header.version < 1u) {
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

// The maps are 32-bit float and not filterable; blend the four texels here.
fn bc_bilinear(tex: texture_2d<f32>, xz: vec2<f32>) -> vec4<f32> {
    let t = bc_texel(xz);
    let i = floor(t);
    let f = t - i;
    let i0 = vec2<i32>(i);
    let a = (bc_load(tex, i0));
    let b = (bc_load(tex, i0 + vec2<i32>(1, 0)));
    let c = (bc_load(tex, i0 + vec2<i32>(0, 1)));
    let d = (bc_load(tex, i0 + vec2<i32>(1, 1)));
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

// Extinction per unit volume fraction, 1/m. Spheres of radius r with an
// extinction efficiency of 2 give 1.5 / r; road dust is about 25 micrometres,
// so 6e4. A fraction of 1e-5 is a haze (0.6 per metre), 1e-4 a plume you
// cannot see the far side of.
const FOG_SIGMA: f32 = 60000.0;
// Plumes look denser than the cell mean: the dust rides in clumps.
// 0.45: the plume read as an opaque wall at 1.2.
const FOG_CLUMP: f32 = 0.45;
const FOG_STEPS: i32 = 40;
// Ground fog past this distance is integrated in closed form, so the steps
// stay where the swirls are.
const FOG_MARCH_M: f32 = 40.0;
const DUST_ALBEDO: vec3<f32> = vec3<f32>(0.58, 0.47, 0.34);
// Water droplets: white, a touch blue from the sky they scatter.
const FOG_ALBEDO: vec3<f32> = vec3<f32>(0.9, 0.92, 0.95);

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

// The plume as the picture draws it: the published cell, plus what the cells
// below and upwind hold, so dust lifts and trails downwind of where the
// ledger counts it. Picture only; the ledger's volume is not changed.
fn fog_plume(p: vec3<f32>) -> f32 {
    // Upwind along the published air (b11): what that air carried here
    // in the last ~0.4 s and ~1 s.
    let wind = air_velocity(p);
    let c = bc_fog_at(p);
    var s = bc_fog_at(p - wind * 0.4 - vec3<f32>(0.0, 0.3, 0.0));
    s += bc_fog_at(p - wind * 1.0 - vec3<f32>(0.0, 0.6, 0.0));
    s += bc_fog_at(p + vec3<f32>(0.55, -0.35, 0.3));
    s += bc_fog_at(p + vec3<f32>(-0.4, -0.35, -0.55));
    return max(c, c * 0.5 + s * 0.2);
}

// ---------------------------------------------------------------------------
// Air velocity. Contract v2 publishes `air_velocity` from the solver; v1 does
// not, so `bc_air_published` returns w = 0 and the picture builds the air from
// what v1 has plus the frame: the wind, the film's flow dragging the air just
// above it, and the Niva's wake. When v2 lands, only bc_air_published reads
// the new binding; everything below stays.
const BC_VALID_AIR_VELOCITY: u32 = 1024u;

fn bc_air_published(p: vec3<f32>) -> vec4<f32> {
    if (!bc_has(BC_VALID_AIR_VELOCITY)) {
        return vec4<f32>(0.0);
    }
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let origin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    let res = vec3<i32>(i32(bc_header.air_res), i32(bc_header.air_layers), i32(bc_header.air_res));
    let t = (p - origin) / max(cell, vec3<f32>(1e-4)) - vec3<f32>(0.5);
    let i0 = vec3<i32>(floor(t));
    let f = t - floor(t);
    var acc = vec3<f32>(0.0);
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let o = vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1);
        let w = mix(1.0 - f.x, f.x, f32(o.x)) * mix(1.0 - f.y, f.y, f32(o.y)) * mix(1.0 - f.z, f.z, f32(o.z));
        let c = clamp(i0 + o, vec3<i32>(0), res - vec3<i32>(1));
        let v = textureLoad(bc_air_tex, c, 0).xyz;
        acc += select(vec3<f32>(0.0), v, all(abs(v) < vec3<f32>(1e3))) * w;
    }
    return vec4<f32>(acc, 1.0);
}

// The Niva in its own frame: x right, z forward, metres from its centre.
fn niva_local(p: vec3<f32>) -> vec3<f32> {
    let yaw = frame.air[2].w;
    let fwd = vec2<f32>(sin(yaw), cos(yaw));
    let d = p - frame.air[1].xyz;
    return vec3<f32>(d.x * fwd.y - d.z * fwd.x, d.y, dot(d.xz, fwd));
}

fn air_velocity(p: vec3<f32>) -> vec3<f32> {
    let pub_v = bc_air_published(p);
    if (pub_v.w > 0.5) {
        return pub_v.xyz;
    }
    var v = frame.air[0].xyz;
    if (bc_has(BC_VALID_FLOW)) {
        let ground = frame.air[1].y - 0.5;
        v += vec3<f32>(bc_flow(p.xz), 0.0).xzy * 0.4 * exp(-max(p.y - ground, 0.0) / 0.4);
    }
    // Wake: the air behind the body is dragged after it and rolls up into a
    // pair of counter-rotating vortices off the sides.
    let car_v = frame.air[2].xyz;
    let speed = length(car_v.xz);
    if (speed > 0.2) {
        let l = niva_local(p);
        let behind = -l.z - 1.9;
        if (behind > -1.0) {
            let along = exp(-max(behind, 0.0) / 4.5) * smoothstep(-1.0, 0.3, behind);
            let spread = 1.0 + max(behind, 0.0) * 0.25;
            let side = exp(-(l.x * l.x) / (spread * spread * 1.1)) * exp(-max(l.y - 0.4, 0.0) / (1.2 * spread));
            v += car_v * 0.55 * along * side;
            // Vortex cores at x = +-0.9 m, 0.6 m up; spin about the travel axis.
            let yaw = frame.air[2].w;
            let fwd = vec3<f32>(sin(yaw), 0.0, cos(yaw));
            let right = vec3<f32>(fwd.z, 0.0, -fwd.x);
            for (var k: i32 = 0; k < 2; k = k + 1) {
                let sgn = select(-1.0, 1.0, k == 1);
                let r = vec2<f32>(l.x - sgn * 0.9 * spread, l.y - 0.6);
                let r2 = dot(r, r) + 0.15;
                let swirl = sgn * speed * 0.7 * along / r2 * exp(-r2 / (2.0 * spread));
                v += (right * -r.y + vec3<f32>(0.0, 1.0, 0.0) * r.x) * swirl;
            }
        }
    }
    return v;
}

// Small-scale detail for a plume or fog the solver resolves only coarsely:
// curl-warped billow noise, carried by the air velocity. Two copies of the
// noise run on staggered 1.6 s cycles and cross-fade, so the advected detail
// never stretches without bound (the flow-map trick). Returns about 1 on
// average, 0 in the gaps, up to 2.6 in a lobe.
fn air_detail(p: vec3<f32>, v: vec3<f32>, t: f32, scale: f32) -> f32 {
    let cycle = 1.6;
    let ph0 = fract(t / cycle);
    let ph1 = fract(t / cycle + 0.5);
    let w0 = 1.0 - abs(1.0 - 2.0 * ph0);
    // Curl is smooth: one tap, not advected, warps both copies.
    let curl = textureSampleLevel(air_noise, air_samp, p * 0.09, 0.0).xyz;
    let q = p + curl * 0.6;
    let q0 = (q - v * ph0 * cycle) * scale;
    let q1 = (q - v * ph1 * cycle + vec3<f32>(0.37, 0.11, 0.73)) * scale;
    let d0 = textureSampleLevel(air_noise, air_samp, q0, 0.0).w * 0.65
        + textureSampleLevel(air_noise, air_samp, q0 * 2.3 + curl * 0.2, 0.0).w * 0.35;
    let d1 = textureSampleLevel(air_noise, air_samp, q1, 0.0).w * 0.65
        + textureSampleLevel(air_noise, air_samp, q1 * 2.3 + curl * 0.2, 0.0).w * 0.35;
    // The billow channel sits around 0.6 with lobes near 1; stretch it so
    // lobes stand out and the gaps between them clear.
    return 2.6 * smoothstep(0.32, 0.95, mix(d1, d0, w0));
}

// Picture ground fog: thinning with height, parted by the Niva's body.
fn ground_fog(p: vec3<f32>) -> f32 {
    let k = frame.air[0].w;
    if (k <= 0.0) {
        return 0.0;
    }
    let ground = frame.air[1].y - 0.5;
    var f = k * exp(-max(p.y - ground, 0.0) / max(frame.air[1].w, 0.1));
    // The body pushes the fog aside; the cleared hole trails a few metres.
    let l = niva_local(p);
    let box = abs(l) - vec3<f32>(1.1, 0.9, 2.1);
    let inside = length(max(box, vec3<f32>(0.0))) + min(max(box.x, max(box.y, box.z)), 0.0);
    let hole = 1.0 - smoothstep(0.0, 0.8, inside);
    let behind = -l.z - 2.0;
    let trail = select(0.0, exp(-behind / 3.0) * exp(-(l.x * l.x) / 1.4) * 0.6, behind > 0.0);
    return f * (1.0 - max(hole, trail));
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
    let has_dust = bc_has(BC_VALID_DUST_FOG);
    let fog_k = frame.air[0].w;
    if (!has_dust && fog_k <= 0.0) {
        discard;
    }
    // Half resolution: each texel stands for a 2x2 block of the frame. The
    // nearest of the four depths keeps the march from running past an edge.
    let px = vec2<i32>(frag.xy) * 2;
    let dmax = vec2<i32>(textureDimensions(scene_depth)) - vec2<i32>(1);
    var d = 1.0;
    for (var k: i32 = 0; k < 4; k = k + 1) {
        d = min(d, textureLoad(scene_depth, min(px + vec2<i32>(k & 1, k >> 1u), dmax), 0));
    }
    let uv = (vec2<f32>(px) + vec2<f32>(1.0)) / max(frame.params.xy, vec2<f32>(1.0));
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d, 1.0);
    let w = frame.inv_view_proj * ndc;
    let ro = frame.cam_pos.xyz;
    let seg = w.xyz / w.w - ro;
    let dist = min(length(seg), 90.0);
    let rd = normalize(seg);
    let inv = vec3<f32>(1.0) / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));

    // The published dust box, lifted 1 m for the plume (see fog_plume).
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let bmin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    let bmax = bmin + cell * vec3<f32>(f32(bc_header.air_res), f32(bc_header.air_layers), f32(bc_header.air_res));
    let bmax_l = bmax + vec3<f32>(0.0, 1.0, 0.0);
    var t_near = 1e9;
    var t_far = 0.0;
    if (has_dust) {
        let ta = (bmin - ro) * inv;
        let tb = (bmax_l - ro) * inv;
        let n0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
        let f0 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), min(max(ta.z, tb.z), dist));
        if (f0 > n0) {
            t_near = n0;
            t_far = f0;
        }
    }
    // The ground-fog slab, up to six scale heights.
    if (fog_k > 0.0) {
        let top = frame.air[1].y - 0.5 + 6.0 * frame.air[1].w;
        var f0 = dist;
        var n0 = 0.0;
        if (ro.y > top) {
            n0 = select(1e9, (top - ro.y) * inv.y, rd.y < 0.0);
        } else if (rd.y > 0.0) {
            f0 = min(dist, (top - ro.y) * inv.y);
        }
        if (f0 > n0) {
            t_near = min(t_near, n0);
            t_far = max(t_far, f0);
        }
    }
    if (t_far <= t_near) {
        discard;
    }
    let t_end = t_far;
    t_far = min(t_far, t_near + FOG_MARCH_M);

    let sun = normalize(frame.sun_dir.xyz);
    let mu = dot(rd, sun);
    let dust_phase = hg_phase(mu, 0.55) * 12.566371;
    // Fog droplets scatter harder forward: the glow round a low sun.
    let fog_phase = mix(hg_phase(mu, 0.8), hg_phase(mu, 0.2), 0.45) * 12.566371;
    let sky = (sh_irradiance(vec3<f32>(0.0, 1.0, 0.0)) * 1.3 + frame.sun_color.rgb * max(sun.y, 0.0) * vec3<f32>(0.2, 0.15, 0.1)) * frame.env.y;
    let lamp_on = frame.lamps[0].w;
    let t_now = frame.misc.x;
    let range = t_far - t_near;
    // Interleaved gradient noise: neighbours get evenly spread offsets, so the
    // banding breaks into a fine, even grain instead of white-noise speckle.
    let ign_p = frag.xy + vec2<f32>(5.588238 * f32(u32(t_now * 30.0) % 64u));
    let jitter = fract(52.982918 * fract(dot(ign_p, vec2<f32>(0.06711056, 0.00583715))));
    let fog_h = max(frame.air[1].w, 0.1);
    let ground = frame.air[1].y - 0.5;
    var trans = 1.0;
    var light = vec3<f32>(0.0);
    // Steps bunch up near the camera (t grows with the square of the step
    // index), and shorten inside a dense plume so one step never jumps from
    // clear air to opaque.
    var t = t_near + range * pow(jitter / f32(FOG_STEPS), 2.0);
    var dt = 0.0;
    for (var k: i32 = 0; k < FOG_STEPS + 16; k = k + 1) {
        t += dt;
        if (t >= t_far) {
            break;
        }
        let u = sqrt(max(t - t_near, 0.0) / range);
        dt = max(range * (2.0 * u + 1.0 / f32(FOG_STEPS)) / f32(FOG_STEPS), 0.05);
        let p = ro + rd * t;
        var dust = 0.0;
        if (has_dust && all(p >= bmin) && all(p <= bmax_l)) {
            let wall = min(min(p.x - bmin.x, bmax.x - p.x), min(p.z - bmin.z, bmax.z - p.z));
            // Thins out upward over the top two metres and at the walls, so
            // the plume melts into the air instead of ending on a ruled edge.
            let fade = smoothstep(0.0, 5.0, wall) * (1.0 - smoothstep(bmax.y - 2.0, bmax_l.y, p.y));
            dust = fog_plume(p) * FOG_SIGMA * FOG_CLUMP * fade;
        }
        let fog = ground_fog(p);
        if (dust + fog < 1e-3) {
            continue;
        }
        // The solver gives the big shape; the curl noise carried by the air
        // adds the swirls. Fog keeps more of its body than a dust plume.
        let v = air_velocity(p);
        let detail = air_detail(p, v, t_now, 0.32);
        let s_dust = dust * detail;
        let s_fog = fog * mix(1.0, detail, 0.7);
        let sigma = s_dust + s_fog;
        dt = min(dt, max(0.9 / max(sigma, 1e-4), 0.08));
        let wd = s_dust / max(sigma, 1e-6);
        // Self-shadow toward the sun: three coarse dust steps, and the fog
        // column above the point in closed form.
        var od = 0.0;
        if (dust > 0.0) {
            for (var j: i32 = 1; j <= 3; j = j + 1) {
                od += bc_fog_at(p + sun * (f32(j) * 0.35)) * FOG_SIGMA * FOG_CLUMP * 0.35;
            }
        }
        od += fog * fog_h / max(sun.y, 0.08);
        let sun_t = max(exp(-od), 0.3 * exp(-od * 0.12));
        let phase = mix(fog_phase, dust_phase, wd);
        // Lifted off the ground so the shadow map's texels do not print as acne.
        let lit = shadow_factor(p + vec3<f32>(0.0, 0.2, 0.0), vec3<f32>(0.0, 1.0, 0.0));
        var l_in = frame.sun_color.rgb * 1.25 * phase * sun_t * lit + sky;
        if (lamp_on > 0.0) {
            for (var i: i32 = 0; i < 2; i = i + 1) {
                let b = lamp_beam(i, p);
                let c = dot(b.xyz, -rd);
                let lp = mix(hg_phase(c, 0.8), hg_phase(c, 0.6), wd) * 12.566371;
                l_in += LAMP_COLOR * LAMP_POWER * lamp_on * b.w * lp;
            }
        }
        let albedo = mix(FOG_ALBEDO, DUST_ALBEDO, wd);
        let step_t = exp(-sigma * dt);
        light += trans * albedo * l_in * (1.0 - step_t);
        trans *= step_t;
        if (trans < 0.01) {
            break;
        }
    }
    // The rest of the ray through the fog: height fog has a closed-form
    // optical depth, lit by the unshadowed sun and the sky.
    if (fog_k > 0.0 && t_end > t_far && trans > 0.01) {
        let p0 = ro + rd * t_far;
        let len = t_end - t_far;
        let dy = rd.y * len;
        let h0 = max(p0.y - ground, 0.0);
        var od_far = fog_k * exp(-h0 / fog_h) * len;
        if (abs(dy) > 1e-3) {
            od_far = fog_k * fog_h * len / dy * (exp(-h0 / fog_h) - exp(-max(h0 + dy, 0.0) / fog_h));
        }
        let step_t = exp(-max(od_far, 0.0));
        let l_far = frame.sun_color.rgb * 1.25 * fog_phase * 0.6 + sky;
        light += trans * FOG_ALBEDO * l_far * (1.0 - step_t);
        trans *= step_t;
    }
    let alpha = 1.0 - trans;
    if (alpha < 0.002) {
        discard;
    }
    return vec4<f32>(tonemap(light / alpha) * alpha, alpha);
}

// Upsample of the half-resolution march over the frame. Premultiplied in,
// straight out for the alpha blend.
@group(0) @binding(20) var fog_half: texture_2d<f32>;
@group(0) @binding(21) var fog_half_samp: sampler;

@fragment
fn fs_fog_up(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let full = vec2<f32>(textureDimensions(fog_half)) * 2.0;
    let c = textureSampleLevel(fog_half, fog_half_samp, frag.xy / full, 0.0);
    if (c.a < 0.002) {
        discard;
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

// Solid ground (bed + placed), metres, from binding 1; the caller's height
// outside the window. The water sheet stands on this, not on the drawn
// terrain, which lifts a pad under the carcass.
fn bc_ground(xz: vec2<f32>, fallback: f32) -> f32 {
    if (!bc_has(BC_VALID_GROUND)) {
        return fallback;
    }
    return mix(fallback, bc_bilinear(bc_ground_rut, xz).r, bc_inside(xz));
}

// The 0.20 m cells low-passed with a 3 x 3 binomial at one cell: the picture
// reads the same maps, but a cut or a ponded cell no longer prints as a
// rectangle with straight walls.
fn bc_film_soft(xz: vec2<f32>) -> vec2<f32> {
    let e = max(bc_header.cell_m, 0.05) * 1.6;
    var acc = vec2<f32>(0.0);
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let w = (2.0 - abs(f32(i))) * (2.0 - abs(f32(j)));
            acc += bc_film(xz + vec2<f32>(f32(i), f32(j)) * e) * w;
        }
    }
    return acc / 16.0;
}

fn bc_ground_soft(xz: vec2<f32>, fallback: f32) -> f32 {
    let e = max(bc_header.cell_m, 0.05);
    var acc = 0.0;
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let w = (2.0 - abs(f32(i))) * (2.0 - abs(f32(j)));
            acc += bc_ground(xz + vec2<f32>(f32(i), f32(j)) * e, fallback) * w;
        }
    }
    return acc / 16.0;
}

fn bc_rut_soft(xz: vec2<f32>) -> vec2<f32> {
    // 5 x 5 binomial (1 4 6 4 1) over 0.75-cell steps of bilinear taps: about
    // a 1.5-cell Gaussian, so the 0.20 m cells of G/A no longer step.
    let e = max(bc_header.cell_m, 0.05) * 0.75;
    var acc = vec2<f32>(0.0);
    for (var j: i32 = -2; j <= 2; j = j + 1) {
        for (var i: i32 = -2; i <= 2; i = i + 1) {
            let wi = select(select(1.0, 4.0, abs(i) == 1), 6.0, i == 0);
            let wj = select(select(1.0, 4.0, abs(j) == 1), 6.0, j == 0);
            acc += bc_rut(xz + vec2<f32>(f32(i), f32(j)) * e) * (wi * wj);
        }
    }
    return acc / 256.0;
}

// Published foam low-passed with the same 5 x 5 binomial as the rut: the
// 0.20 m foam cells printed as grey plates with stepped edges.
fn bc_foam_soft(xz: vec2<f32>) -> vec2<f32> {
    let e = max(bc_header.cell_m, 0.05) * 0.75;
    var acc = vec2<f32>(0.0);
    for (var j: i32 = -2; j <= 2; j = j + 1) {
        for (var i: i32 = -2; i <= 2; i = i + 1) {
            let wi = select(select(1.0, 4.0, abs(i) == 1), 6.0, i == 0);
            let wj = select(select(1.0, 4.0, abs(j) == 1), 6.0, j == 0);
            acc += bc_foam_crest(xz + vec2<f32>(f32(i), f32(j)) * e) * (wi * wj);
        }
    }
    return acc / 256.0;
}

// ---- Requested fields, not in contract v2 yet (docs/beauty/CONTRACT_REQUESTS.md).
// The bits are reserved here so the readers below compile and stay dark:
// until the header sets them, every reader returns 0 and nothing is drawn.
const BC_VALID_SNOW: u32 = 2048u;      // bit 11, requested binding 12
const BC_VALID_SOIL_MIX: u32 = 4096u;  // bit 12, requested binding 13

// Requested binding 12, rgba32float on the ground grid: R snow depth (m),
// G density (kg/m3), B compaction 0..1 (fresh 0, packed by a tire 1),
// A belly scrape depth (m) where the underbody cut the blanket.
fn bc_snow(xz: vec2<f32>) -> vec4<f32> {
    if (!bc_has(BC_VALID_SNOW)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(0.0);
}

// Requested binding 13, rgba32float on the ground grid: fractions of mud,
// grass, sand and gravel in the cell (sum <= 1). Fractions, not a kind id,
// so a seam between two soils blends over the cells instead of stepping.
fn bc_soil_mix(xz: vec2<f32>) -> vec4<f32> {
    if (!bc_has(BC_VALID_SOIL_MIX)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(0.0);
}

// Free-surface height, ground + water + slurry, through one 3 x 3 binomial.
fn bc_surface_soft(xz: vec2<f32>) -> f32 {
    let e = max(bc_header.cell_m, 0.05) * 1.3;
    var acc = 0.0;
    for (var j: i32 = -1; j <= 1; j = j + 1) {
        for (var i: i32 = -1; i <= 1; i = i + 1) {
            let w = (2.0 - abs(f32(i))) * (2.0 - abs(f32(j)));
            let q = xz + vec2<f32>(f32(i), f32(j)) * e;
            let f = bc_film(q);
            acc += (bc_ground(q, 0.0) + f.r + f.g) * w;
        }
    }
    return acc / 16.0;
}
