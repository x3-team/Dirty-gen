// Smoke and fire on a world-anchored 3D grid, wgpu compute.
//
// Structure after GPU Gems 3 ch. 30 (Crane, Llamas, Tariq) as ported in
// Scrawk/GPU-GEMS-3D-Fluid-Simulation (MIT): advect, buoyancy, vorticity
// confinement, divergence, Jacobi pressure, project. Buoyancy from
// temperature and soot as in doyubkim/fluid-engine-dev GridSmokeSolver (MIT).
// Scalars use MacCormack (Selle et al. 2008) with the min/max clamp.
//
// Collocated grid, cell centres. Index x + nx * (y + ny * z). +Y is up.
// Sides and top are open: outside air is the wind with no soot and no heat,
// pressure 0. The ground (y < 0) is a wall: no flow through, pressure mirrors.
//
// scal: x soot (volume fraction scale, 1 = opaque at `ext` per metre),
//       y temperature above ambient (K), z fuel (0..1), w unused.

struct Source {
    // centre xyz (world m), radius m
    pos: vec4<f32>,
    // fuel /s, soot /s, heat K/s, upward speed m/s
    rate: vec4<f32>,
}

struct Params {
    // nx, ny, nz, frame counter
    dims: vec4<u32>,
    // world origin of cell (0,0,0)'s corner xyz, cell size m
    origin: vec4<f32>,
    // dt s, buoyancy per K (m/s² / K), soot weight (m/s² per unit soot), vorticity eps
    step: vec4<f32>,
    // soot decay /s, heat cooling /s, velocity damping toward wind /s, fuel burn /s
    decay: vec4<f32>,
    // wind xyz m/s, ambient K
    wind: vec4<f32>,
    // sun direction xyz (towards the sun), soot extinction per metre at soot 1
    sun: vec4<f32>,
    // heat per unit fuel burnt K, soot per unit fuel burnt, max speed m/s, source jitter m/s
    combust: vec4<f32>,
    sources: array<Source, 4>,
    // number of sources, _, _, _
    count: vec4<u32>,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> scal_in: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> scal_out: array<vec4<f32>>;
// Curl (xyz, |w|) after `k_curl`; the forward MacCormack estimate before it.
@group(0) @binding(5) var<storage, read_write> work: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> p_in: array<f32>;
@group(0) @binding(7) var<storage, read_write> p_out: array<f32>;
@group(0) @binding(8) var<storage, read_write> div: array<f32>;
// r soot, g temperature / 1000 K, b flame (burning fuel), a sun transmittance.
@group(0) @binding(9) var out_tex: texture_storage_3d<rgba16float, write>;

override SMOKE_SHADOW_STEPS: i32 = 24;

fn dims() -> vec3<i32> {
    return vec3<i32>(P.dims.xyz);
}

fn idx(c: vec3<i32>) -> u32 {
    let d = dims();
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn inside(c: vec3<i32>) -> bool {
    let d = dims();
    return all(c >= vec3<i32>(0)) && all(c < d);
}

// Velocity at a cell, with the open boundary and the ground mirror.
fn vel_at(src: u32, c_in: vec3<i32>) -> vec3<f32> {
    var c = c_in;
    var mirror = false;
    if (c.y < 0) {
        c.y = -1 - c.y;
        mirror = true;
    }
    if (!inside(c)) {
        return P.wind.xyz;
    }
    var v: vec3<f32>;
    if (src == 0u) {
        v = vel_in[idx(c)].xyz;
    } else {
        v = vel_out[idx(c)].xyz;
    }
    if (mirror) {
        v.y = -v.y;
    }
    return v;
}

fn scal_at(src: u32, c_in: vec3<i32>) -> vec4<f32> {
    var c = c_in;
    c.y = max(c.y, 0);
    if (!inside(c)) {
        return vec4<f32>(0.0);
    }
    if (src == 0u) {
        return scal_in[idx(c)];
    }
    return work[idx(c)];
}

// Trilinear in cell-centre coordinates (cell i has its centre at i).
fn vel_lerp(src: u32, p: vec3<f32>) -> vec3<f32> {
    let f = floor(p);
    let t = p - f;
    let c = vec3<i32>(f);
    let a = mix(
        mix(vel_at(src, c), vel_at(src, c + vec3<i32>(1, 0, 0)), t.x),
        mix(vel_at(src, c + vec3<i32>(0, 1, 0)), vel_at(src, c + vec3<i32>(1, 1, 0)), t.x),
        t.y,
    );
    let b = mix(
        mix(vel_at(src, c + vec3<i32>(0, 0, 1)), vel_at(src, c + vec3<i32>(1, 0, 1)), t.x),
        mix(vel_at(src, c + vec3<i32>(0, 1, 1)), vel_at(src, c + vec3<i32>(1, 1, 1)), t.x),
        t.y,
    );
    return mix(a, b, t.z);
}

fn scal_lerp(src: u32, p: vec3<f32>) -> vec4<f32> {
    let f = floor(p);
    let t = p - f;
    let c = vec3<i32>(f);
    let a = mix(
        mix(scal_at(src, c), scal_at(src, c + vec3<i32>(1, 0, 0)), t.x),
        mix(scal_at(src, c + vec3<i32>(0, 1, 0)), scal_at(src, c + vec3<i32>(1, 1, 0)), t.x),
        t.y,
    );
    let b = mix(
        mix(scal_at(src, c + vec3<i32>(0, 0, 1)), scal_at(src, c + vec3<i32>(1, 0, 1)), t.x),
        mix(scal_at(src, c + vec3<i32>(0, 1, 1)), scal_at(src, c + vec3<i32>(1, 1, 1)), t.x),
        t.y,
    );
    return mix(a, b, t.z);
}

// Min and max of the 8 corners around p in scal_in: the MacCormack clamp.
fn scal_bounds(p: vec3<f32>) -> array<vec4<f32>, 2> {
    let c = vec3<i32>(floor(p));
    var lo = vec4<f32>(1e9);
    var hi = vec4<f32>(-1e9);
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let s = scal_at(0u, c + o);
        lo = min(lo, s);
        hi = max(hi, s);
    }
    return array<vec4<f32>, 2>(lo, hi);
}

fn backtrace(c: vec3<i32>, v: vec3<f32>, sign: f32) -> vec3<f32> {
    let d = vec3<f32>(dims());
    var p = vec3<f32>(c) - sign * P.step.x * v / P.origin.w;
    // Half a cell under the ground is still ground.
    p.y = max(p.y, 0.0);
    return clamp(p, vec3<f32>(-1.0), d);
}

fn cell_of(gid: vec3<u32>) -> vec3<i32> {
    return vec3<i32>(gid);
}

// 1. Semi-Lagrangian velocity, forward scalar estimate into `work`.
@compute @workgroup_size(4, 4, 4)
fn k_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let i = idx(c);
    let v = vel_in[i].xyz;
    let p = backtrace(c, v, 1.0);
    vel_out[i] = vec4<f32>(vel_lerp(0u, p), 0.0);
    work[i] = scal_lerp(0u, p);
}

// 2. MacCormack correction for the scalars, with decay.
@compute @workgroup_size(4, 4, 4)
fn k_maccormack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let i = idx(c);
    let v = vel_in[i].xyz;
    let hat = work[i];
    let p_fwd = backtrace(c, v, -1.0);
    // Sample the forward estimate back at x + dt v.
    let bar = scal_lerp(1u, p_fwd);
    var s = hat + 0.5 * (scal_in[i] - bar);
    let b = scal_bounds(backtrace(c, v, 1.0));
    s = clamp(s, b[0], b[1]);
    s = max(s, vec4<f32>(0.0));
    let dt = P.step.x;
    s.x = s.x * exp(-P.decay.x * dt);
    s.y = s.y * exp(-P.decay.y * dt);
    scal_out[i] = s;
}

fn hash(n: u32) -> f32 {
    var x = n;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x * 9u;
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return f32(x & 0xffffu) / 65535.0;
}

// 3. Curl of the advected velocity into `work` (xyz, |w|).
@compute @workgroup_size(4, 4, 4)
fn k_curl(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let h2 = 2.0 * P.origin.w;
    let l = vel_at(1u, c - vec3<i32>(1, 0, 0));
    let r = vel_at(1u, c + vec3<i32>(1, 0, 0));
    let d = vel_at(1u, c - vec3<i32>(0, 1, 0));
    let u = vel_at(1u, c + vec3<i32>(0, 1, 0));
    let b = vel_at(1u, c - vec3<i32>(0, 0, 1));
    let f = vel_at(1u, c + vec3<i32>(0, 0, 1));
    let w = vec3<f32>(
        (u.z - d.z) - (f.y - b.y),
        (f.x - b.x) - (r.z - l.z),
        (r.y - l.y) - (u.x - d.x),
    ) / h2;
    work[idx(c)] = vec4<f32>(w, length(w));
}

fn curl_mag(c_in: vec3<i32>) -> f32 {
    var c = clamp(c_in, vec3<i32>(0), dims() - vec3<i32>(1));
    return work[idx(c)].w;
}

// 4. Combustion, sources, cooling, buoyancy, vorticity confinement.
@compute @workgroup_size(4, 4, 4)
fn k_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let i = idx(c);
    let dt = P.step.x;
    let h = P.origin.w;
    var v = vel_out[i].xyz;
    var s = scal_out[i];

    // Combustion: fuel burns at a first-order rate into heat and soot.
    let burn = s.z * (1.0 - exp(-P.decay.w * dt));
    s.z = s.z - burn;
    s.y = s.y + burn * P.combust.x;
    s.x = s.x + burn * P.combust.y;

    // Sources.
    let world = P.origin.xyz + (vec3<f32>(c) + 0.5) * h;
    let n = min(P.count.x, 4u);
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        let src = P.sources[k];
        let dist = length(world - src.pos.xyz);
        let w = 1.0 - smoothstep(0.5 * src.pos.w, src.pos.w, dist);
        if (w > 0.0) {
            s.z = s.z + src.rate.x * dt * w;
            s.x = s.x + src.rate.y * dt * w;
            s.y = s.y + src.rate.z * dt * w;
            // Jitter breaks the symmetry so the column can roll.
            let seed = i * 747796405u + P.dims.w * 2891336453u + k;
            let j = vec3<f32>(hash(seed) - 0.5, hash(seed ^ 0x9e3779b9u) - 0.5, hash(seed + 0x85ebca6bu) - 0.5);
            let target_v = vec3<f32>(0.0, src.rate.w, 0.0) + j * P.combust.w;
            v = mix(v, target_v, clamp(w * 0.5, 0.0, 1.0));
        }
    }

    s.z = clamp(s.z, 0.0, 1.0);

    // Boussinesq buoyancy: hot air rises, soot weighs it down.
    v.y = v.y + dt * (P.step.y * s.y - P.step.z * s.x);

    // Vorticity confinement, f = eps h (N x w).
    let eta = vec3<f32>(
        curl_mag(c + vec3<i32>(1, 0, 0)) - curl_mag(c - vec3<i32>(1, 0, 0)),
        curl_mag(c + vec3<i32>(0, 1, 0)) - curl_mag(c - vec3<i32>(0, 1, 0)),
        curl_mag(c + vec3<i32>(0, 0, 1)) - curl_mag(c - vec3<i32>(0, 0, 1)),
    );
    let el = length(eta);
    if (el > 1e-6) {
        let nrm = eta / el;
        v = v + dt * P.step.w * h * cross(nrm, work[i].xyz);
    }

    // Relax toward the wind and cap the speed.
    v = P.wind.xyz + (v - P.wind.xyz) * exp(-P.decay.z * dt);
    let sp = length(v);
    if (sp > P.combust.z) {
        v = v * (P.combust.z / sp);
    }
    if (c.y == 0) {
        v.y = max(v.y, 0.0);
    }
    vel_out[i] = vec4<f32>(v, 0.0);
    scal_out[i] = s;
}

// 5. Divergence of vel_out.
@compute @workgroup_size(4, 4, 4)
fn k_divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let l = vel_at(1u, c - vec3<i32>(1, 0, 0)).x;
    let r = vel_at(1u, c + vec3<i32>(1, 0, 0)).x;
    let d = vel_at(1u, c - vec3<i32>(0, 1, 0)).y;
    let u = vel_at(1u, c + vec3<i32>(0, 1, 0)).y;
    let b = vel_at(1u, c - vec3<i32>(0, 0, 1)).z;
    let f = vel_at(1u, c + vec3<i32>(0, 0, 1)).z;
    div[idx(c)] = ((r - l) + (u - d) + (f - b)) / (2.0 * P.origin.w);
}

fn p_at(c_in: vec3<i32>) -> f32 {
    var c = c_in;
    // Ground: pressure mirrors (no flow through).
    c.y = max(c.y, 0);
    if (!inside(c)) {
        return 0.0;
    }
    return p_in[idx(c)];
}

// 6. One Jacobi sweep of lap(p) = div.
@compute @workgroup_size(4, 4, 4)
fn k_jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let h = P.origin.w;
    let sum = p_at(c - vec3<i32>(1, 0, 0)) + p_at(c + vec3<i32>(1, 0, 0))
        + p_at(c - vec3<i32>(0, 1, 0)) + p_at(c + vec3<i32>(0, 1, 0))
        + p_at(c - vec3<i32>(0, 0, 1)) + p_at(c + vec3<i32>(0, 0, 1));
    p_out[idx(c)] = (sum - h * h * div[idx(c)]) / 6.0;
}

// 7. Subtract the pressure gradient.
@compute @workgroup_size(4, 4, 4)
fn k_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let i = idx(c);
    let g = vec3<f32>(
        p_at(c + vec3<i32>(1, 0, 0)) - p_at(c - vec3<i32>(1, 0, 0)),
        p_at(c + vec3<i32>(0, 1, 0)) - p_at(c - vec3<i32>(0, 1, 0)),
        p_at(c + vec3<i32>(0, 0, 1)) - p_at(c - vec3<i32>(0, 0, 1)),
    ) / (2.0 * P.origin.w);
    var v = vel_out[i].xyz - g;
    if (c.y == 0) {
        v.y = max(v.y, 0.0);
    }
    vel_out[i] = vec4<f32>(v, 0.0);
}

// 8. Texture for the renderer, with the sun transmittance per cell.
@compute @workgroup_size(4, 4, 4)
fn k_publish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = cell_of(gid);
    if (!inside(c)) {
        return;
    }
    let s = scal_out[idx(c)];
    let d = vec3<f32>(dims());
    let sun = normalize(P.sun.xyz);
    // March towards the sun in cell units, one cell per step.
    var tau = 0.0;
    var p = vec3<f32>(c) + sun * 0.5;
    for (var k: i32 = 0; k < SMOKE_SHADOW_STEPS; k = k + 1) {
        if (any(p < vec3<f32>(-0.5)) || any(p > d - vec3<f32>(0.5))) {
            break;
        }
        tau = tau + scal_out[idx(vec3<i32>(round(p)))].x;
        p = p + sun;
    }
    let trans = exp(-tau * P.sun.w * P.origin.w);
    let burn = s.z * (1.0 - exp(-P.decay.w * P.step.x)) / max(P.step.x, 1e-4);
    textureStore(out_tex, c, vec4<f32>(s.x, s.y / 1000.0, burn, trans));
}
