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
// Spray pieces (binding 10), SprayGpu x 4096, live count in the header.
struct BcSpray {
    pos_mass: vec4<f32>,
    vel_age: vec4<f32>,
    material: u32,
    alive: u32,
    pad0: u32,
    pad1: u32,
}
@group(4) @binding(10) var<storage, read> bc_spray: array<BcSpray>;
// Air velocity, m/s, same origin and cell as dust_fog (contract v2).
@group(4) @binding(11) var bc_air_tex: texture_3d<f32>;
// Solver mist, water volume fraction on the air grid (contract v5, #106).
@group(4) @binding(12) var bc_fog_mist_tex: texture_3d<f32>;

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
// Steps and shadow taps come from the quality preset (docs/GPU_BUDGET.md,
// #126): pipeline overrides, the high preset as the default.
override QUALITY_FOG_STEPS: i32 = 40;
override QUALITY_FOG_SHADOW_STEPS: i32 = 3;
// Air layers the march reads, from the ground up. The grid has 16 (4 m).
override QUALITY_AIR_LAYERS: i32 = 16;
// Water droplets of mist, radius ~10 micrometres: 1.5 / r per volume fraction.
const MIST_SIGMA: f32 = 150000.0;
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

// Solver mist (binding 12), same grid and index as dust.
fn mist_tap(i: vec3<i32>) -> f32 {
    let res = vec3<i32>(i32(bc_header.air_res), i32(bc_header.air_layers), i32(bc_header.air_res));
    if (any(i < vec3<i32>(0)) || any(i >= res)) {
        return 0.0;
    }
    let v = textureLoad(bc_fog_mist_tex, i, 0).r;
    return select(0.0, max(v, 0.0), abs(v) < 1.0e4);
}

// Mist volume fraction at a world point, trilinear.
fn bc_mist_at(p: vec3<f32>) -> f32 {
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let origin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    let t = (p - origin) / max(cell, vec3<f32>(1e-4)) - vec3<f32>(0.5);
    let i0 = vec3<i32>(floor(t));
    let f = t - floor(t);
    var acc = 0.0;
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let o = vec3<i32>(k & 1, (k >> 1u) & 1, (k >> 2u) & 1);
        let w = mix(1.0 - f.x, f.x, f32(o.x)) * mix(1.0 - f.y, f.y, f32(o.y)) * mix(1.0 - f.z, f.z, f32(o.z));
        acc += mist_tap(i0 + o) * w;
    }
    return acc;
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

// The plume as the solver publishes it. Since the air grid is 4 m tall and
// the wake lifts dust itself (16 layers), the picture no longer moves dust
// up or downwind of the cell that holds it.
fn fog_plume(p: vec3<f32>) -> f32 {
    return bc_fog_at(p);
}

// Extinction, 1/m, of the published dust (x) and mist (y) at `p`: the cell
// means, faded at the side walls of the window so the plume does not end on
// a ruled edge. No picture detail.
fn air_sigma(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    if (any(p < bmin) || any(p > bmax)) {
        return vec2<f32>(0.0);
    }
    let wall = min(min(p.x - bmin.x, bmax.x - p.x), min(p.z - bmin.z, bmax.z - p.z));
    // The 4 m lid: what the wake mixed up to it thins over the top 0.75 m.
    let fade = smoothstep(0.0, 3.0, wall) * (1.0 - smoothstep(bmax.y - 0.75, bmax.y, p.y));
    return vec2<f32>(bc_fog_at(p) * FOG_SIGMA * FOG_CLUMP, bc_mist_at(p) * MIST_SIGMA) * fade;
}

// `air_sigma` over two trilinear taps half a cell apart on a diagonal of the
// plane: the published cells stop printing as blocks where neighbouring
// columns ride the lee at different speeds. For the view samples only.
fn air_sigma_soft(p: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let h = 0.5 * vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y * 0.5, -bc_header.air_cell_z);
    return 0.5 * (air_sigma(p + h, bmin, bmax) + air_sigma(p - h, bmin, bmax));
}

// `air_sigma` blurred over a disc of radius r, for light paths only: two
// taps on a diagonal turned by the pixel's jitter. Light that reaches a point
// has crossed a cone, not a line, so the shadow should not print the solver's
// cells; the half-res tent then averages the turns of neighbouring pixels.
fn air_sigma_cone(p: vec3<f32>, r: f32, turn: f32, bmin: vec3<f32>, bmax: vec3<f32>) -> vec2<f32> {
    let a = turn * 6.2831853;
    let h = vec3<f32>(cos(a), 0.6 * sin(a * 1.7), sin(a)) * r;
    return 0.5 * (air_sigma(p + h, bmin, bmax) + air_sigma(p - h, bmin, bmax));
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

// ---------------------------------------------------------------------------
// Smoke and fire (`silt::smoke`, GPU grid, world-anchored): group 5 of the
// dust march, so soot, dust and mist share one march and one budget. With no
// smoke the slot holds zero constants and 1 x 1 x 1 zero textures.
// Look after the smoke thread's march (Bevy volumetric_fog.wgsl, MIT/Apache):
// HG single scattering with the sun transmittance the solver marched per
// cell (.a), two Wrenninge octaves, black-body flame emission.
struct SmokeDraw {
    // grid min corner xyz (world), cell m
    origin: vec4<f32>,
    // grid size m xyz, extinction per metre at soot 1 (0: no smoke)
    size: vec4<f32>,
    // soot albedo rgb, phase g
    albedo: vec4<f32>,
    // emission gain, ambient gain, kelvin per LUT step, LUT steps
    light: vec4<f32>,
    // Black body: linear rgb (max 1), relative power; entry k at k * light.z K over ambient.
    lut: array<vec4<f32>, 24>,
    // Fire lights: position xyz, radius; colour rgb.
    fire_pos: array<vec4<f32>, 4>,
    fire_col: array<vec4<f32>, 4>,
    // fire light count, detail gain, noise frequency per cell, reference eddy speed m/s
    detail: vec4<f32>,
}

@group(5) @binding(0) var<uniform> sd: SmokeDraw;
// r soot, g temperature over ambient / 1000 K, b burn rate 1/s, a sun transmittance.
@group(5) @binding(1) var smoke_vol: texture_3d<f32>;
@group(5) @binding(2) var smoke_samp: sampler;
// xyz velocity m/s, w unresolved eddy speed |curl v| * h.
@group(5) @binding(3) var smoke_flow: texture_3d<f32>;

// Curl-noise octaves of sub-grid smoke detail. 0 turns it off.
override QUALITY_SMOKE_DETAIL: i32 = 2;
// Multiple-scattering gain on the smoke's sun octaves.
const SMOKE_MS_GAIN: f32 = 5.0;

fn smoke_blackbody(dt_k: f32) -> vec4<f32> {
    let n = i32(sd.light.w);
    let x = clamp(dt_k / sd.light.z, 0.0, f32(n - 1));
    let k = min(i32(x), n - 2);
    return mix(sd.lut[k], sd.lut[k + 1], x - f32(k));
}

// Curl of three offset simplex noises (Bridson et al. 2007): divergence
// free, so the detail swirls instead of bunching.
fn smoke_curl(q: vec3<f32>) -> vec3<f32> {
    let a = sdnoise3(q);
    let b = sdnoise3(q + vec3<f32>(31.416, 47.853, 12.679));
    let c = sdnoise3(q + vec3<f32>(-19.123, 7.391, 53.749));
    return vec3<f32>(
        c.gradient.y - b.gradient.z,
        a.gradient.z - c.gradient.x,
        b.gradient.x - a.gradient.y,
    );
}

// Sub-grid detail after Kim et al. 2008: curl noise below one cell, scaled by
// the unresolved eddy speed and falling 2^(-5/6) per octave (Kolmogorov). It
// displaces the density lookup instead of advecting a finer grid.
fn smoke_detail_offset(p: vec3<f32>, flow: vec4<f32>) -> vec3<f32> {
    let gain = sd.detail.y;
    if (gain <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = sd.origin.w;
    let e = clamp(0.4 + flow.w / max(sd.detail.w, 1e-3), 0.0, 1.5);
    let drift = flow.xyz * frame.misc.x;
    var off = vec3<f32>(0.0);
    var scale = sd.detail.z / h;
    var amp = h;
    for (var k: i32 = 0; k < QUALITY_SMOKE_DETAIL; k = k + 1) {
        off = off + smoke_curl((p - drift) * scale + vec3<f32>(f32(k) * 17.0)) * amp;
        scale = scale * 2.0;
        amp = amp * 0.5612;
    }
    return off * gain * e;
}

fn smoke_fire_light(p: vec3<f32>) -> vec3<f32> {
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

// Smoke sample at p: (soot, temperature/1000 K, burn rate, sun transmittance),
// with the sub-grid detail displacing the lookup. Zero outside the grid.
fn smoke_at(p: vec3<f32>) -> vec4<f32> {
    let bmin = sd.origin.xyz;
    let uvw = (p - bmin) / sd.size.xyz;
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
        return vec4<f32>(0.0);
    }
    var s = textureSampleLevel(smoke_vol, smoke_samp, uvw, 0.0);
    if (s.r < 1e-4 && s.b < 1e-3) {
        return vec4<f32>(0.0);
    }
    // Detail displaces the soot (and its sun transmittance) only. The flame
    // (temperature, burn rate) keeps the undisplaced sample, and the detail
    // fades inside it: per-pixel noise on an emitter printed as confetti that
    // the half-res upsample cannot average.
    let flow = textureSampleLevel(smoke_flow, smoke_samp, uvw, 0.0);
    let off = smoke_detail_offset(p, flow) * (1.0 - saturate(s.b));
    if (dot(off, off) > 0.0) {
        let d = textureSampleLevel(smoke_vol, smoke_samp, (p + off - bmin) / sd.size.xyz, 0.0);
        s = vec4<f32>(d.r, s.g, s.b, d.a);
    }
    return max(s, vec4<f32>(0.0));
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
    let has_smoke = sd.size.w > 0.0 && sd.size.x > 0.0;
    let fog_k = frame.air[0].w;
    if (!has_dust && fog_k <= 0.0 && !has_smoke) {
        discard;
    }
    // Half resolution: each texel stands for a 2x2 block of the frame. The
    // nearest of the four depths keeps the march from running past an edge.
    let px = vec2<i32>(frag.xy) * 2;
    let dmax = vec2<i32>(textureDimensions(post_depth)) - vec2<i32>(1);
    var d = 1.0;
    for (var k: i32 = 0; k < 4; k = k + 1) {
        d = min(d, textureLoad(post_depth, min(px + vec2<i32>(k & 1, k >> 1u), dmax), 0).x);
    }
    let uv = (vec2<f32>(px) + vec2<f32>(1.0)) / max(frame.params.xy, vec2<f32>(1.0));
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, d, 1.0);
    let w = frame.inv_view_proj * ndc;
    let ro = frame.cam_pos.xyz;
    let seg = w.xyz / w.w - ro;
    let dist = min(length(seg), 90.0);
    let rd = normalize(seg);
    let inv = vec3<f32>(1.0) / select(rd, vec3<f32>(1e-6), abs(rd) < vec3<f32>(1e-6));

    // The published air box, up to the layers this preset reads.
    let cell = vec3<f32>(bc_header.air_cell_x, bc_header.air_cell_y, bc_header.air_cell_z);
    let bmin = vec3<f32>(bc_header.air_origin_x, bc_header.air_origin_y, bc_header.air_origin_z);
    let layers = f32(min(i32(bc_header.air_layers), max(QUALITY_AIR_LAYERS, 1)));
    let bmax = bmin + cell * vec3<f32>(f32(bc_header.air_res), layers, f32(bc_header.air_res));
    let bmax_l = bmax;
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
    // The smoke grid, world-anchored, may stand outside the air box.
    if (has_smoke) {
        let ta = (sd.origin.xyz - ro) * inv;
        let tb = (sd.origin.xyz + sd.size.xyz - ro) * inv;
        let n0 = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), max(min(ta.z, tb.z), 0.0));
        let f0 = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), min(max(ta.z, tb.z), dist));
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
    // Flame light, added over the cloud rather than through its alpha.
    var emit = vec3<f32>(0.0);
    // Steps bunch up near the camera (t grows with the square of the step
    // index), and shorten inside a dense plume so one step never jumps from
    // clear air to opaque.
    let steps = max(QUALITY_FOG_STEPS, 4);
    let fsteps = f32(steps);
    var t = t_near + range * pow(jitter / fsteps, 2.0);
    var dt = 0.0;
    // Sky light is from above and the sand below bounces the sun.
    let ground_bounce = DUST_ALBEDO * frame.sun_color.rgb * max(sun.y, 0.0) * 0.12 * frame.env.y;
    for (var k: i32 = 0; k < steps + 16; k = k + 1) {
        t += dt;
        if (t >= t_far) {
            break;
        }
        let u = sqrt(max(t - t_near, 0.0) / range);
        dt = max(range * (2.0 * u + 1.0 / fsteps) / fsteps, 0.05);
        // Each sample sits at the pixel's jitter inside its own step. With
        // the jitter only on the first step, near-level rays of a row took
        // their samples at the same depths and the plume printed as lines.
        let p = ro + rd * max(t + dt * (jitter - 0.5), t_near);
        var dm = vec2<f32>(0.0);
        if (has_dust) {
            dm = air_sigma_soft(p, bmin, bmax);
        }
        let dust = dm.x;
        let mist = dm.y;
        let fog = ground_fog(p) + mist;
        // Smoke from the GPU grid at the same point: soot extinction, and the
        // burning gas that glows.
        var sm = vec4<f32>(0.0);
        if (has_smoke) {
            sm = smoke_at(p);
        }
        let s_smoke = sm.r * sd.size.w;
        let flame = sm.b;
        let has_air = dust + fog >= 1e-3;
        if (!has_air && s_smoke < 1e-4 && flame < 1e-3) {
            continue;
        }
        // Dust and mist: in-scattered radiance times albedo, per unit sigma.
        var sig_air = 0.0;
        var l_air = vec3<f32>(0.0);
        if (has_air) {
            // The solver gives the big shape; the curl noise carried by the air
            // adds the swirls. Detail erodes the thin rim of a plume harder than
            // its core, so edges go wispy and the body stays a body.
            // The air solve runs on 0.25 x 0.5 x 0.25 m blocks copied onto the
            // fine grid, so its velocity steps at every block face; carried by
            // that, the noise tore into rectangles. Two taps one block apart on a
            // diagonal turn each step into a ramp.
            let o = vec3<f32>(bc_header.air_cell_x * 2.0, bc_header.air_cell_y * 2.0, -bc_header.air_cell_z * 2.0) * 0.5;
            let v = 0.5 * (air_velocity(p + o) + air_velocity(p - o));
            let detail = air_detail(p, v, t_now, 0.32);
            let core = 1.0 - exp(-dust * 0.6);
            let s_dust = dust * mix(detail, 0.6 + 0.4 * detail, core);
            let s_fog = fog * mix(1.0, detail, 0.7);
            let sigma = s_dust + s_fog;
            let wd = s_dust / max(sigma, 1e-6);
            // Self-shadow toward the sun through the published grid: taps 0.4,
            // 0.9, 1.9 m long (doubling), so a 3 m column darkens its far side.
            // The point's own detail scales the first tap.
            // Turns for the light cones: smooth in the world, so they neither
            // print as a pixel pattern nor line up along screen rows.
            let turn = textureSampleLevel(air_noise, air_samp, p * 0.12 + vec3<f32>(0.21, 0.47, 0.13), 0.0).xyz * 3.0;
            var od = 0.0;
            if (has_dust) {
                var s0 = 0.0;
                var len = 0.4;
                for (var j: i32 = 0; j < QUALITY_FOG_SHADOW_STEPS; j = j + 1) {
                    // Jittered along the tap so layer faces do not line up into
                    // stripes, and widened with distance like the cone of light.
                    let q = p + sun * (s0 + (0.25 + 0.5 * fract(turn.y + f32(j) * 0.618)) * len);
                    let a = air_sigma_cone(q, 0.15 + 0.35 * (s0 + 0.5 * len), turn.x + f32(j) * 0.37, bmin, bmax);
                    od += (a.x + a.y) * len * select(1.0, detail, j == 0);
                    s0 += len;
                    len *= 2.2;
                }
            }
            od += ground_fog(p) * fog_h / max(sun.y, 0.08);
            // Multiple scattering, after Wrenninge et al.: each octave sees a
            // thinner medium and a rounder phase, so the lit core glows instead
            // of going flat grey behind its first metre.
            var ms = 0.0;
            var a_o = 1.0;
            var b_o = 1.0;
            var c_o = 1.0;
            for (var o: i32 = 0; o < 3; o = o + 1) {
                let ph = mix(mix(hg_phase(mu, 0.8 * c_o), hg_phase(mu, 0.2 * c_o), 0.45), hg_phase(mu, 0.55 * c_o), wd);
                ms += a_o * ph * exp(-od * b_o);
                a_o *= 0.5;
                b_o *= 0.5;
                c_o *= 0.5;
            }
            ms *= 12.566371;
            // Powder: the sunward rim of a dense puff is darker than its inside,
            // where light has scattered in from all sides.
            let powder = 0.45 + 0.55 * (1.0 - exp(-sigma * 0.8));
            // Sky through the plume above the point: dark undersides, bright tops.
            var od_up = 0.0;
            if (has_dust) {
                let a1 = air_sigma_cone(p + vec3<f32>(0.0, 0.5, 0.0), 0.35, turn.z, bmin, bmax);
                let a2 = air_sigma_cone(p + vec3<f32>(0.0, 1.5, 0.0), 0.8, turn.z + 0.25, bmin, bmax);
                od_up = (a1.x + a1.y) * 0.8 + (a2.x + a2.y) * 1.2;
            }
            let sky_vis = exp(-od_up * 0.9);
            // Lifted off the ground so the shadow map's texels do not print as acne.
            // Two taps a step apart along the ray: the map's texels no longer
            // print as bricks through a thin plume.
            let lit = 0.5 * (shadow_factor(p + vec3<f32>(0.0, 0.2, 0.0), vec3<f32>(0.0, 1.0, 0.0))
                + shadow_factor(p - rd * (0.5 * dt) + vec3<f32>(0.0, 0.2, 0.0), vec3<f32>(0.0, 1.0, 0.0)));
            var l_in = frame.sun_color.rgb * 1.25 * ms * powder * lit + sky * mix(0.75, 1.0, 1.0 - wd) * sky_vis + ground_bounce * (1.0 - sky_vis * 0.5);
            if (lamp_on > 0.0) {
                for (var i: i32 = 0; i < 2; i = i + 1) {
                    let b = lamp_beam(i, p);
                    let c = dot(b.xyz, -rd);
                    let lp = mix(hg_phase(c, 0.8), hg_phase(c, 0.6), wd) * 12.566371;
                    l_in += LAMP_COLOR * LAMP_POWER * lamp_on * b.w * lp;
                }
            }
            sig_air = sigma;
            l_air = mix(FOG_ALBEDO, DUST_ALBEDO, wd) * l_in;
        }
        // Soot: the solver's sun transmittance (.a) through the grid replaces
        // shadow taps; the scene shadow map still cuts the Niva's shadow in.
        var l_smoke = vec3<f32>(0.0);
        if (s_smoke >= 1e-4) {
            let ta = sm.a;
            let ms_s = (0.5 * sqrt(ta) + 0.25 * sqrt(sqrt(ta))) * SMOKE_MS_GAIN * (1.0 / 12.566371);
            let lit_s = shadow_factor(p + vec3<f32>(0.0, 0.2, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            let sun_s = frame.sun_color.rgb * lit_s * (ta * hg_phase(mu, sd.albedo.w) + ms_s);
            l_smoke = sd.albedo.rgb * (sun_s + sky * sd.light.y + smoke_fire_light(p));
        }
        let sigma_all = sig_air + s_smoke;
        dt = min(dt, max(0.9 / max(sigma_all, 1e-4), 0.08));
        let step_t = exp(-sigma_all * dt);
        light += trans * (sig_air * l_air + s_smoke * l_smoke) / max(sigma_all, 1e-6) * (1.0 - step_t);
        if (flame >= 1e-3) {
            let bb = smoke_blackbody(sm.g * 1000.0);
            emit += trans * bb.rgb * bb.w * flame * sd.light.x * dt;
        }
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
    let glow = tonemap(emit);
    if (alpha < 0.002 && max(glow.r, max(glow.g, glow.b)) < 0.002) {
        discard;
    }
    // Premultiplied; the flame adds on top (the upsample lifts alpha to the
    // glow so the blend keeps it additive).
    return vec4<f32>(min(tonemap(light / max(alpha, 1e-4)) * alpha + glow, vec3<f32>(1.0)), alpha);
}

// Upsample of the half-resolution march over the frame. Premultiplied in,
// straight out for the alpha blend.
@group(0) @binding(20) var fog_half: texture_2d<f32>;
@group(0) @binding(21) var fog_half_samp: sampler;

fn fog_up_sample(frag: vec4<f32>) -> vec4<f32> {
    let full = vec2<f32>(textureDimensions(fog_half)) * 2.0;
    // Four bilinear taps half a half-res texel apart: a 4 x 4 tent over the
    // march, which melts the per-pixel jitter of the steps into a smooth
    // gradient. Premultiplied, so a thin rim does not bleed dark.
    let uv = frag.xy / full;
    let e = vec2<f32>(1.0) / full;
    var c = textureSampleLevel(fog_half, fog_half_samp, uv + vec2<f32>(-e.x, -e.y), 0.0);
    c += textureSampleLevel(fog_half, fog_half_samp, uv + vec2<f32>(e.x, -e.y), 0.0);
    c += textureSampleLevel(fog_half, fog_half_samp, uv + vec2<f32>(-e.x, e.y), 0.0);
    c += textureSampleLevel(fog_half, fog_half_samp, uv + vec2<f32>(e.x, e.y), 0.0);
    // A wider ring, turned 45 degrees: the march's interleaved jitter repeats
    // over a few half-res texels, and a dense plume printed it as hatching.
    let r = 3.0 * e;
    var w = textureSampleLevel(fog_half, fog_half_samp, uv + vec2<f32>(r.x, 0.0), 0.0);
    w += textureSampleLevel(fog_half, fog_half_samp, uv - vec2<f32>(r.x, 0.0), 0.0);
    w += textureSampleLevel(fog_half, fog_half_samp, uv + vec2<f32>(0.0, r.y), 0.0);
    w += textureSampleLevel(fog_half, fog_half_samp, uv - vec2<f32>(0.0, r.y), 0.0);
    c = c * 0.15 + w * 0.1;
    // Flame glow is premultiplied colour above alpha. Raising alpha to the
    // brightest channel keeps the straight colour within 1, so the alpha
    // blend adds it instead of clipping it.
    let a = max(c.a, max(c.r, max(c.g, c.b)));
    if (a < 0.002) {
        discard;
    }
    return vec4<f32>(c.rgb / a, a);
}

@fragment
fn fs_fog_up(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return fog_up_sample(frag);
}

// The same composite into the linear HDR target, for when the 8-bit
// picture is not made from it directly (SILT_POST reads the HDR target).
// Blending in linear light thins the edges of the cloud a little.
@fragment
fn fs_fog_up_hdr(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let c = fog_up_sample(frag);
    return vec4<f32>(aces_inverse(c.rgb), c.a);
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

// ---- Fields from Cursor's open contract PRs. Bits and bindings are theirs
// (docs/beauty/CONTRACT_REQUESTS.md); every reader checks the header
// version (RENDER_CONTRACT_VERSION) and its valid bit, and returns 0 until
// the solver sets it. Bindings 12 and 13 are not in main's layout yet, so
// their textures are declared when those PRs land; until then the readers
// stay dark.
const BC_VALID_TURBIDITY: u32 = 2048u; // 1 << 11, #99 (v3): binding 4 A
const BC_VALID_MIX: u32 = 4096u;       // 1 << 12, #107 (v6): binding 13
const BC_VALID_FOG: u32 = 8192u;       // 1 << 13, #106 (v5): binding 12, 3D

// Suspended solids in the water, 0..1 (contract v3, binding 4 A).
fn bc_turbidity(xz: vec2<f32>) -> f32 {
    if (bc_header.version < 3u || !bc_has(BC_VALID_TURBIDITY)) {
        return 0.0;
    }
    return clamp(bc_bilinear(bc_flow_tex, xz).a, 0.0, 1.0) * bc_inside(xz);
}

// Snow column (#105): height m, density kg/m3, compaction 0..1. That PR
// still names binding 12 / bit 11, which fog (#106) and turbidity (#99)
// hold; dark until #105 is rebased onto its own binding and bit.
fn bc_snow(xz: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(0.0);
}

// Material shares (#107, v6, binding 13): R mud, G sand, B gravel, A grass.
fn bc_soil_mix(xz: vec2<f32>) -> vec4<f32> {
    if (bc_header.version < 6u || !bc_has(BC_VALID_MIX)) {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(0.0);
}

// Solver water fog (#106, v5, binding 12), volume fraction on the air grid.
fn bc_fog_water(p: vec3<f32>) -> f32 {
    if (bc_header.version < 5u || !bc_has(BC_VALID_FOG)) {
        return 0.0;
    }
    return bc_mist_at(p);
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

const BC_VALID_SPRAY: u32 = 512u; // 1 << 9, binding 10

// Rings where the published water drops (spray materials 0, 4, 9) meet the
// sheet: a drop at or just under the surface level `surf` is an impact, its
// age the time since it crossed the level. Cauchy-Poisson rings after
// Ultra-Water's impactWaves: k = g t^2 / (4 r^2), splash radius L, band
// limited by the pixel (k_max). Returns the slope (dh/dx, dh/dz).
fn bc_drop_rings(xz: vec2<f32>, surf: f32, k_max: f32) -> vec2<f32> {
    if (!bc_has(BC_VALID_SPRAY)) {
        return vec2<f32>(0.0);
    }
    let n = min(bc_header.spray_count, 512u);
    var g = vec2<f32>(0.0);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let s = bc_spray[i];
        if (s.alive == 0u || !(s.material == 0u || s.material == 4u || s.material == 9u)) {
            continue;
        }
        let dy = s.pos_mass.y - surf;
        let vy = s.vel_age.y;
        if (dy > 0.03 || vy > 0.0) {
            continue;
        }
        let dv = xz - s.pos_mass.xz;
        let r = length(dv) + 1e-3;
        if (r > 0.6) {
            continue;
        }
        let age = 0.08 + max(-dy, 0.0) / max(-vy, 0.5);
        let l = 0.015;
        let k = 9.81 * age * age / (4.0 * r * r);
        let kl2 = k * k * l * l;
        let amp = 0.0015 * 2.72 * kl2 * exp(-kl2 - k * k / (k_max * k_max) - 0.12 * age) * l / (l + r);
        g += dv / r * (amp * k * sin(k * r));
    }
    return g;
}

// psrdnoise3.wgsl

//
// Authors: Stefan Gustavson (stefan.gustavson@gmail.com)
// and Ian McEwan (ijm567@gmail.com)
// Version 2022-02-28, published under the MIT license (see below)
//
// Copyright (c) 2021-2022 Stefan Gustavson and Ian McEwan.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//

// Only sdnoise3 and its helpers are kept; the struct syntax is updated to
// current WGSL (comma field separators).

struct NG3 {
	noise: f32,
	gradient: vec3<f32>,
};

fn mod289v4f(i: vec4<f32>) -> vec4<f32> {
	return i - floor(i / 289.0) * 289.0;
}

fn permute289v4f(i: vec4<f32>) -> vec4<f32>
{
	var im: vec4<f32> = mod289v4f(i);
	return mod289v4f((im*34.0 + 10.0)*im);
}

fn sdnoise3(x: vec3<f32>) -> NG3
{
	let M = mat3x3<f32>(0.0, 1.0, 1.0, 1.0, 0.0, 1.0,  1.0, 1.0, 0.0);
	let Mi = mat3x3<f32>(-0.5, 0.5, 0.5, 0.5,-0.5, 0.5, 0.5, 0.5,-0.5);

	var uvw: vec3<f32>;
	var i0: vec3<f32>;
	var i1: vec3<f32>;
	var i2: vec3<f32>;
	var i3: vec3<f32>;
	var f0: vec3<f32>;
	var gt_: vec3<f32>;
	var lt_: vec3<f32>;
	var gt: vec3<f32>;
	var lt: vec3<f32>;
	var o1: vec3<f32>;
	var o2: vec3<f32>;
	var v0: vec3<f32>;
	var v1: vec3<f32>;
	var v2: vec3<f32>;
	var v3: vec3<f32>;
	var x0: vec3<f32>;
	var x1: vec3<f32>;
	var x2: vec3<f32>;
	var x3: vec3<f32>;
	
	uvw = M * x;
	i0 = floor(uvw);
	f0 = uvw - i0;
	gt_ = step(f0.xyx, f0.yzz);
	lt_ = 1.0 - gt_;
	gt = vec3<f32>(lt_.z, gt_.xy);
	lt = vec3<f32>(lt_.xy, gt_.z);
	o1 = min( gt, lt );
	o2 = max( gt, lt );
	i1 = i0 + o1;
	i2 = i0 + o2;
	i3 = i0 + vec3<f32>(1.0,1.0,1.0);
	v0 = Mi * i0;
	v1 = Mi * i1;
	v2 = Mi * i2;
	v3 = Mi * i3;
	x0 = x - v0;
	x1 = x - v1;
	x2 = x - v2;
	x3 = x - v3;

	var hash: vec4<f32>;
	var theta: vec4<f32>;
	var sz: vec4<f32>;
	var psi: vec4<f32>;
	var St: vec4<f32>;
	var Ct: vec4<f32>;
	var sz_: vec4<f32>;

	hash = permute289v4f( permute289v4f( permute289v4f( 
		vec4<f32>(i0.z, i1.z, i2.z, i3.z ))
		+ vec4<f32>(i0.y, i1.y, i2.y, i3.y ))
		+ vec4<f32>(i0.x, i1.x, i2.x, i3.x ));
	theta = hash * 3.883222077;
	sz = hash * -0.006920415 + 0.996539792;
	psi = hash * 0.108705628;
	Ct = cos(theta);
	St = sin(theta);
	sz_ = sqrt( 1.0 - sz*sz );

	var gx: vec4<f32>;
	var gy: vec4<f32>;
	var gz: vec4<f32>;

	gx = Ct * sz_;
	gy = St * sz_;
	gz = sz;  
	
	var g0: vec3<f32>;
	var g1: vec3<f32>;
	var g2: vec3<f32>;
	var g3: vec3<f32>;
	var w: vec4<f32>;
	var w2: vec4<f32>;
	var w3: vec4<f32>;
	var gdotx: vec4<f32>;
	var n: f32;
	
	g0 = vec3<f32>(gx.x, gy.x, gz.x);
	g1 = vec3<f32>(gx.y, gy.y, gz.y);
	g2 = vec3<f32>(gx.z, gy.z, gz.z);
	g3 = vec3<f32>(gx.w, gy.w, gz.w);
	w = 0.5 - vec4<f32>(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3));
	w = max(w, vec4<f32>(0.0, 0.0, 0.0, 0.0));
	w2 = w * w;
	w3 = w2 * w;
	gdotx = vec4<f32>(dot(g0,x0), dot(g1,x1), dot(g2,x2), dot(g3,x3));
	n = 39.5 * dot(w3, gdotx);

	var dw: vec4<f32> = -6.0 * w2 * gdotx;
	var dn0: vec3<f32> = w3.x * g0 + dw.x * x0;
	var dn1: vec3<f32> = w3.y * g1 + dw.y * x1;
	var dn2: vec3<f32> = w3.z * g2 + dw.z * x2;
	var dn3: vec3<f32> = w3.w * g3 + dw.w * x3;
	var g: vec3<f32> = 39.5 * (dn0 + dn1 + dn2 + dn3);
	
	return NG3(n, g);
}
