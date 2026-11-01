// Ground geometry: where the vertices go and how the solver height is read
// between its samples. Appended to shader.wgsl (see render.rs).
//
// Nothing here invents relief. The height at a vertex is the solver's
// heightfield, read either bilinearly or with a monotone cubic that passes
// through every solver sample and never overshoots them, plus the fine
// track field when the solver publishes one.

struct TerrainGeom {
    // x: finest vertex step (m), y: ring radius in steps, z: height the
    // LOD distance is measured at, w: morph start as a share of the ring.
    lod: vec4<f32>,
    // x: corner of the top node, y: lowest drawn x/z, z: highest drawn x/z,
    // w: 1 = LOD debug colours.
    ext: vec4<f32>,
    // x: 1 = monotone cubic between solver samples, 0 = bilinear.
    // y: 1 = the fine track texture holds a solver publish.
    // z: feather at the fine window border, metres.
    recon: vec4<f32>,
    // Fine track window: origin x, origin z (minimum corner), cell (m), res.
    fine: vec4<f32>,
}

// Fine track field from the solver, rgba32float, not filterable.
// R: surface offset (m) over the bilinear 20 cm ground at the texel centre.
// G: snow depth, B: compaction, A: lug depth. Only R moves the ground.
@group(1) @binding(10) var track_fine_tex: texture_2d<f32>;
@group(1) @binding(11) var<uniform> tgeom: TerrainGeom;

// Fritsch-Butland tangent: harmonic mean of the two slopes, 0 at a turn.
// A groove bottom and a berm crest stay at the solver value, no ringing.
fn tg_mono_tangent(a: f32, b: f32) -> f32 {
    if (a * b <= 0.0) {
        return 0.0;
    }
    return 2.0 * a * b / (a + b);
}

fn tg_mono(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let m1 = tg_mono_tangent(p1 - p0, p2 - p1);
    let m2 = tg_mono_tangent(p2 - p1, p3 - p2);
    let t2 = t * t;
    let t3 = t2 * t;
    return (2.0 * t3 - 3.0 * t2 + 1.0) * p1
        + (t3 - 2.0 * t2 + t) * m1
        + (-2.0 * t3 + 3.0 * t2) * p2
        + (t3 - t2) * m2;
}

fn tg_row(i: i32, j: i32, t: f32) -> f32 {
    return tg_mono(
        height_at(vec2<i32>(i - 1, j)),
        height_at(vec2<i32>(i, j)),
        height_at(vec2<i32>(i + 1, j)),
        height_at(vec2<i32>(i + 2, j)),
        t,
    );
}

// Monotone bicubic through the 20 cm solver samples. C1 across cells, so a
// one-cell rut is a rounded groove instead of a V of two flat facets.
fn tg_height_cubic(xz: vec2<f32>) -> f32 {
    let ij = world_to_ij(xz);
    let b = floor(ij);
    let f = ij - b;
    let i = i32(b.x);
    let j = i32(b.y);
    return tg_mono(
        tg_row(i, j - 1, f.x),
        tg_row(i, j, f.x),
        tg_row(i, j + 1, f.x),
        tg_row(i, j + 2, f.x),
        f.y,
    );
}

fn tg_fine_load(ij: vec2<i32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(track_fine_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(track_fine_tex, c, 0).r;
}

// x: offset (m), y: weight (1 inside the window, 0 past its feathered edge).
fn tg_fine(xz: vec2<f32>) -> vec2<f32> {
    if (tgeom.recon.y < 0.5) {
        return vec2<f32>(0.0);
    }
    let cell = max(tgeom.fine.z, 1e-4);
    let res = tgeom.fine.w;
    let local = (xz - tgeom.fine.xy) / cell;
    let edge = min(min(local.x, local.y), min(res - local.x, res - local.y)) * cell;
    let w = smoothstep(0.0, max(tgeom.recon.z, 1e-3), edge);
    if (w <= 0.0) {
        return vec2<f32>(0.0);
    }
    let g = local - vec2<f32>(0.5);
    let g0 = floor(g);
    let f = g - g0;
    let i0 = vec2<i32>(g0);
    let a = tg_fine_load(i0);
    let b = tg_fine_load(i0 + vec2<i32>(1, 0));
    let c = tg_fine_load(i0 + vec2<i32>(0, 1));
    let d = tg_fine_load(i0 + vec2<i32>(1, 1));
    return vec2<f32>(mix(mix(a, b, f.x), mix(c, d, f.x), f.y), w);
}

// Called by height_raw with its bilinear read. The fine field is defined
// over that bilinear ground, so inside its window the base is bilinear.
fn tg_refine(xz: vec2<f32>, bilinear: f32) -> f32 {
    var h = bilinear;
    if (tgeom.recon.x > 0.5) {
        h = tg_height_cubic(xz);
    }
    let fine = tg_fine(xz);
    if (fine.y > 0.0) {
        h = mix(h, bilinear + fine.x, fine.y);
    }
    return h;
}

// ---- LOD placement --------------------------------------------------------

struct LodIn {
    // Patch grid coordinate, 0..32.
    @location(0) g: vec2<f32>,
    // Instance: origin x, origin z, drawn size, level.
    @location(1) node: vec4<f32>,
}

// xz of the vertex after the morph, and the morph amount (0..1).
// CPU twin: terrain_lod::vertex_xz. Keep the two identical.
fn tg_lod_place(g: vec2<f32>, node: vec4<f32>) -> vec3<f32> {
    let level = node.w;
    let step = tgeom.lod.x * exp2(level);
    let h = node.z / 32.0;
    let root = vec2<f32>(tgeom.ext.x);
    // Global grid index on this level. Half steps of a quarter patch
    // collapse onto the level grid.
    let gl = floor((node.xy - root) / step + g * (h / step));
    let p = root + gl * step;
    let d = distance(vec3<f32>(p.x, tgeom.lod.z, p.y), frame.cam_pos.xyz);
    let end = step * tgeom.lod.y;
    let prev = select(end * 0.5, 0.0, level < 0.5);
    let start = prev + (end - prev) * tgeom.lod.w;
    let k = clamp((d - start) / max(end - start, 1e-4), 0.0, 1.0);
    let odd = fract(gl * 0.5) * 2.0;
    let gm = gl - odd * k;
    let xz = clamp(root + gm * step, vec2<f32>(tgeom.ext.y), vec2<f32>(tgeom.ext.z));
    return vec3<f32>(xz, k);
}

fn tg_terr_in(g: vec2<f32>, node: vec4<f32>) -> TerrIn {
    let xz = tg_lod_place(g, node).xy;
    return TerrIn(xz, world_to_ij(xz));
}

@vertex
fn vs_terrain(in: TerrIn) -> TerrOut {
    return terrain_vertex(in);
}

@vertex
fn vs_terrain_lod(in: LodIn) -> TerrOut {
    return terrain_vertex(tg_terr_in(in.g, in.node));
}

@vertex
fn vs_water(in: TerrIn) -> WaterOut {
    return water_vertex(in);
}

@vertex
fn vs_water_lod(in: LodIn) -> WaterOut {
    return water_vertex(tg_terr_in(in.g, in.node));
}

@vertex
fn vs_shadow_terrain(in: TerrIn) -> ShadowOut {
    return shadow_terrain_vertex(in);
}

@vertex
fn vs_shadow_terrain_lod(in: LodIn) -> ShadowOut {
    return shadow_terrain_vertex(tg_terr_in(in.g, in.node));
}

// ---- LOD debug view -------------------------------------------------------

struct LodDebugOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    // Vertex grid of this level, in steps.
    @location(1) grid: vec2<f32>,
    // Level plus the morph toward the next one.
    @location(2) level: f32,
}

@vertex
fn vs_terrain_lod_debug(in: LodIn) -> LodDebugOut {
    let p = tg_lod_place(in.g, in.node);
    let t = terrain_vertex(TerrIn(p.xy, world_to_ij(p.xy)));
    let step = tgeom.lod.x * exp2(in.node.w);
    var o: LodDebugOut;
    o.clip = t.clip;
    o.world = t.world;
    o.grid = (p.xy - vec2<f32>(tgeom.ext.x)) / step;
    o.level = in.node.w + p.z;
    return o;
}

fn tg_level_color(l: f32) -> vec3<f32> {
    let i = i32(floor(l));
    if (i == 0) { return vec3<f32>(0.95, 0.35, 0.30); }
    if (i == 1) { return vec3<f32>(0.95, 0.70, 0.25); }
    if (i == 2) { return vec3<f32>(0.55, 0.85, 0.30); }
    if (i == 3) { return vec3<f32>(0.25, 0.75, 0.85); }
    if (i == 4) { return vec3<f32>(0.35, 0.45, 0.95); }
    return vec3<f32>(0.70, 0.45, 0.90);
}

// Ground lit plainly, tinted by level, with the triangle grid drawn on it.
@fragment
fn fs_terrain_lod_debug(in: LodDebugOut) -> @location(0) vec4<f32> {
    let n = terrain_normal(in.world.xz);
    let lo = tg_level_color(in.level);
    let hi = tg_level_color(in.level + 1.0);
    let tint = mix(lo, hi, fract(in.level));
    let fw = max(fwidth(in.grid), vec2<f32>(1e-4));
    let dist = abs(fract(in.grid + vec2<f32>(0.5)) - vec2<f32>(0.5)) / fw;
    let line = 1.0 - clamp(min(dist.x, dist.y), 0.0, 1.0);
    // Where lines are denser than two pixels the grid reads as a flat tone.
    let dense = smoothstep(0.25, 0.5, max(fw.x, fw.y));
    let albedo = tint * (1.0 - 0.55 * line * (1.0 - dense)) * mix(1.0, 0.75, dense);
    let lit = shade(in.world, n, albedo * 0.6, 0.95, vec3<f32>(0.0));
    return present(lit);
}
