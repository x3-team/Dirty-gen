// Debug view of the physics → render contract.
//
// This shader is a debug view. It reads only the contract bind group.
// It does not shade materials, light, HDRI, fog, headlights, or wet sheen.
//
// Change this contract only by bumping RENDER_CONTRACT_VERSION and
// recording the change in docs/RENDER_CONTRACT.md.

const RENDER_CONTRACT_VERSION: u32 = 2u;

struct ContractHeader {
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

struct SprayParticle {
    pos_mass: vec4<f32>,
    vel_age: vec4<f32>,
    material: u32,
    alive: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> header: ContractHeader;
@group(0) @binding(1) var ground_rut: texture_2d<f32>;
@group(0) @binding(2) var moisture_tex: texture_2d<f32>;
@group(0) @binding(3) var film_tex: texture_2d<f32>;
@group(0) @binding(4) var flow_tex: texture_2d<f32>;
@group(0) @binding(5) var dust_settled_tex: texture_2d<f32>;
@group(0) @binding(6) var dust_fog: texture_3d<f32>;
@group(0) @binding(7) var crust_tex: texture_2d<f32>;
@group(0) @binding(8) var wet_edge_tex: texture_2d<f32>;
@group(0) @binding(9) var foam_crest_tex: texture_2d<f32>;
@group(0) @binding(10) var<storage, read> spray: array<SprayParticle>;
@group(0) @binding(11) var air_velocity: texture_3d<f32>;

struct DebugOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) ij: vec2<f32>,
    @location(2) world_xz: vec2<f32>,
}

fn corner_of(tri: u32) -> vec2<f32> {
    if (tri == 1u) { return vec2<f32>(1.0, 0.0); }
    if (tri == 2u || tri == 4u) { return vec2<f32>(1.0, 1.0); }
    if (tri == 5u) { return vec2<f32>(0.0, 1.0); }
    return vec2<f32>(0.0, 0.0);
}

fn ramp(v: f32, lo: f32, hi: f32) -> vec3<f32> {
    let t = clamp((v - lo) / max(hi - lo, 1e-4), 0.0, 1.0);
    return vec3<f32>(t, t * 0.85, 0.25 + t * 0.5);
}

fn fog_at(world_x: f32, world_z: f32) -> f32 {
    let cell = vec3<f32>(header.air_cell_x, header.air_cell_y, header.air_cell_z);
    if (cell.x <= 0.0 || cell.y <= 0.0 || cell.z <= 0.0) {
        return 0.0;
    }
    let p = vec3<f32>(world_x, header.air_origin_y + cell.y * 1.5, world_z);
    let origin = vec3<f32>(header.air_origin_x, header.air_origin_y, header.air_origin_z);
    let i = vec3<i32>(floor((p - origin) / cell));
    let dims = vec3<i32>(textureDimensions(dust_fog, 0));
    if (any(i < vec3<i32>(0)) || any(i >= dims)) {
        return 0.0;
    }
    return textureLoad(dust_fog, i, 0).r;
}

fn air_speed_at(world_x: f32, world_z: f32) -> f32 {
    let cell = vec3<f32>(header.air_cell_x, header.air_cell_y, header.air_cell_z);
    if (cell.x <= 0.0 || cell.y <= 0.0 || cell.z <= 0.0) {
        return 0.0;
    }
    let p = vec3<f32>(world_x, header.air_origin_y + cell.y * 1.5, world_z);
    let origin = vec3<f32>(header.air_origin_x, header.air_origin_y, header.air_origin_z);
    let i = vec3<i32>(floor((p - origin) / cell));
    let dims = vec3<i32>(textureDimensions(air_velocity, 0));
    if (any(i < vec3<i32>(0)) || any(i >= dims)) {
        return 0.0;
    }
    return length(textureLoad(air_velocity, i, 0).rgb);
}

fn field_color(ij: vec2<i32>, world_xz: vec2<f32>) -> vec3<f32> {
    let mode = header.debug_mode;
    let ground = textureLoad(ground_rut, ij, 0);
    let film = textureLoad(film_tex, ij, 0);
    let flow = textureLoad(flow_tex, ij, 0);
    let crust = textureLoad(crust_tex, ij, 0);
    let foam = textureLoad(foam_crest_tex, ij, 0);
    if (mode == 1u) { return ramp(ground.g, 0.0, 0.25); }
    if (mode == 2u) { return ramp(textureLoad(moisture_tex, ij, 0).r, 0.0, 1.0); }
    if (mode == 3u) { return ramp(film.r, 0.0, 0.12); }
    if (mode == 4u) { return ramp(film.g, 0.0, 0.12); }
    if (mode == 5u) { return ramp(length(flow.rg), 0.0, 1.5); }
    if (mode == 6u) { return ramp(textureLoad(dust_settled_tex, ij, 0).r, 0.0, 0.01); }
    if (mode == 7u) { return ramp(crust.r, 0.0, 0.04); }
    if (mode == 8u) { return ramp(crust.g, 0.0, 0.02); }
    if (mode == 9u) { return ramp(textureLoad(wet_edge_tex, ij, 0).r, 0.0, 1.0); }
    if (mode == 10u) { return ramp(foam.r, 0.0, 1.0); }
    if (mode == 11u) { return ramp(foam.g, 0.0, 0.08); }
    if (mode == 13u) { return ramp(fog_at(world_xz.x, world_xz.y), 0.0, 0.02); }
    if (mode == 14u) { return ramp(air_speed_at(world_xz.x, world_xz.y), 0.0, 12.0); }
    return ramp(ground.r, -0.4, 0.8);
}

@vertex
fn vs_debug(@builtin(vertex_index) vid: u32) -> DebugOut {
    var out: DebugOut;
    let n = max(header.window_res, 1u);
    let cell_id = vid / 6u;
    if (cell_id >= n * n) {
        out.clip = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        out.color = vec4<f32>(0.0);
        out.ij = vec2<f32>(0.0);
        out.world_xz = vec2<f32>(0.0);
        return out;
    }
    let i = cell_id % n;
    let j = cell_id / n;
    let corner = corner_of(vid % 6u);
    let xz = vec2<f32>(header.origin_x, header.origin_z)
        + (vec2<f32>(f32(i), f32(j)) + corner) * max(header.cell_m, 1e-4);
    let extent = max(header.extent_m, 1e-4);
    let nx = (xz.x - header.origin_x) / extent * 2.0 - 1.0;
    let nz = (xz.y - header.origin_z) / extent * 2.0 - 1.0;
    let ij = vec2<i32>(i32(i), i32(j));
    let ground = textureLoad(ground_rut, ij, 0).r;
    out.clip = vec4<f32>(nx, nz, 0.5 - ground * 0.002, 1.0);
    out.color = vec4<f32>(field_color(ij, xz), 1.0);
    out.ij = vec2<f32>(f32(i), f32(j));
    out.world_xz = xz;
    return out;
}

@fragment
fn fs_debug(in: DebugOut) -> @location(0) vec4<f32> {
    if (header.version != RENDER_CONTRACT_VERSION) {
        return vec4<f32>(1.0, 0.0, 1.0, 1.0);
    }
    if (header.debug_mode == 12u) {
        var n = 0.0;
        let lim = min(header.spray_count, 256u);
        let i = i32(in.ij.x);
        let j = i32(in.ij.y);
        for (var s: u32 = 0u; s < lim; s = s + 1u) {
            let p = spray[s].pos_mass.xyz;
            let ix = i32(floor((p.x - header.origin_x) / max(header.cell_m, 1e-4)));
            let iz = i32(floor((p.z - header.origin_z) / max(header.cell_m, 1e-4)));
            if (ix == i && iz == j && spray[s].alive > 0u) {
                n = n + 1.0;
            }
        }
        let t = clamp(n / 4.0, 0.0, 1.0);
        return vec4<f32>(t, 0.35 * t, 0.1, 1.0);
    }
    // Keep the header time in the compiled view so a stale uniform is visible
    // as a one-lsb shift, not as lighting.
    let flick = step(0.5, fract(header.time_s * 0.0 + header.center_x * 0.0));
    return vec4<f32>(in.color.rgb, 1.0) + vec4<f32>(flick * 0.0);
}
