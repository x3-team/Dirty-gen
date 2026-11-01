// GPU port of the height–velocity sheet step (`fluid_core::substep`).
//
// One thread per cell, whole grid, no work boxes. Each entry point is one
// pass of the CPU substep. wgpu puts a barrier between dispatches, so the
// passes read what the previous pass wrote.
//
// Two places differ from the CPU on purpose:
// * Vorticity confinement is a snapshot Jacobi sweep everywhere. The CPU
//   walled sheet runs a forward Gauss–Seidel sweep there; that order is what
//   holds the e2e standing slurry at 0.0536 m. Jacobi gives 0.0500 m on the
//   CPU and on this port alike (a red–black sweep gives 0.0501 m).
// * Transport is a gather: each cell sums the fluxes of its own two faces in
//   the same order as the CPU scatter. A donor gives at most 0.45 of its
//   column per face, so the CPU `room` clamp never binds and the result is
//   the same per face. Both sides of a face compute the same dh from the
//   same inputs, so the volume moved out of one cell is the volume moved in.

struct Params {
    res: u32,
    solid: u32,
    project: u32,
    pad: u32,
    cell: f32,
    dt: f32,
    vort: f32,
    eps: f32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<storage, read_write> u: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
// bed [0, n), viscosity [n, 2n)
@group(0) @binding(4) var<storage, read_write> stat: array<f32>;
// bit 0: gate, bit 1: solid
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;
// scratch planes, n floats each, see the K_* indices
@group(0) @binding(6) var<storage, read_write> scr: array<f32>;

const K_ETA: u32 = 0u;
const K_TG: u32 = 1u;
const K_TS: u32 = 2u;
const K_P: u32 = 3u;
const K_W: u32 = 4u;
const K_B: u32 = 5u;
const K_Q: u32 = 6u;
const K_RREM: u32 = 7u;
const K_H2: u32 = 8u;
// four redirect shares per cell from plane 9
const K_RSH: u32 = 9u;

const GRAVITY: f32 = 9.81;

fn gk(t: u32) -> f32 {
    switch t {
        case 0u, 4u: { return 0.06; }
        case 1u, 3u: { return 0.24; }
        default: { return 0.40; }
    }
}

fn sk(t: u32) -> f32 {
    switch t {
        case 0u, 4u: { return 0.02; }
        case 1u, 3u: { return 0.08; }
        default: { return 0.80; }
    }
}

fn ncell() -> u32 {
    return P.res * P.res;
}

fn sget(k: u32, idx: u32) -> f32 {
    return scr[k * ncell() + idx];
}

fn sset(k: u32, idx: u32, x: f32) {
    scr[k * ncell() + idx] = x;
}

fn bed(idx: u32) -> f32 {
    return stat[idx];
}

fn nu(idx: u32) -> f32 {
    return stat[ncell() + idx];
}

fn is_solid(idx: u32) -> bool {
    return P.solid != 0u && (flags[idx] & 2u) != 0u;
}

fn gate(idx: u32) -> bool {
    return (flags[idx] & 1u) != 0u;
}

// Neighbour order of the CPU loops: -x, +x, -z, +z.
fn dir4(d: u32) -> vec2<i32> {
    switch d {
        case 0u: { return vec2<i32>(-1, 0); }
        case 1u: { return vec2<i32>(1, 0); }
        case 2u: { return vec2<i32>(0, -1); }
        default: { return vec2<i32>(0, 1); }
    }
}

fn finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn cl(i: i32) -> u32 {
    return u32(clamp(i, 0, i32(P.res) - 1));
}

fn at(i: i32, j: i32) -> u32 {
    return cl(j) * P.res + cl(i);
}

fn inside(i: i32, j: i32) -> bool {
    return i >= 0 && j >= 0 && i < i32(P.res) && j < i32(P.res);
}

fn sheet_mobility_for(n: f32) -> f32 {
    var boost = 0.0;
    if n < 12.0 {
        let t = clamp((12.0 - max(n, 0.0)) / 12.0, 0.0, 1.0);
        boost = t * t * 1.6;
    }
    return 0.042 * (1.0 + boost);
}

fn sheet_damp(n: f32) -> f32 {
    let rel = clamp(max(n, 0.05) / 231.0, 0.002, 12.0);
    if rel >= 0.42 {
        return 1.70 * pow(rel, 0.30);
    }
    let at_silt = 1.70 * pow(0.42, 0.30);
    let t = clamp(rel / 0.42, 0.0, 1.0);
    return at_silt * pow(t, 1.15);
}

fn finite_speed(x: f32, lim: f32) -> f32 {
    if !finite(x) {
        return 0.0;
    }
    return clamp(x, -lim, lim);
}

// Central difference with the edge index clamped, as `grad` on the CPU.
fn grad_plane(k: u32, i: i32, j: i32) -> vec2<f32> {
    let two = 2.0 * P.cell;
    let dx = (sget(k, at(i + 1, j)) - sget(k, at(i - 1, j))) / two;
    let dz = (sget(k, at(i, j + 1)) - sget(k, at(i, j - 1))) / two;
    return vec2<f32>(dx, dz);
}

fn vorticity_at(i: i32, j: i32) -> f32 {
    let two = 2.0 * P.cell;
    let dvdx = (v[at(i + 1, j)] - v[at(i - 1, j)]) / two;
    let dudz = (u[at(i, j + 1)] - u[at(i, j - 1)]) / two;
    return dvdx - dudz;
}

fn cell_id(gid: vec3<u32>) -> bool {
    return gid.x < P.res && gid.y < P.res;
}

// ---------------------------------------------------------------- pressure

@compute @workgroup_size(8, 8)
fn k_eta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    sset(K_ETA, idx, bed(idx) + h[idx]);
}

// Solid cell takes the mean free surface of the wet open cells within two.
@compute @workgroup_size(8, 8)
fn k_lift(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    let idx = gid.y * P.res + gid.x;
    var out = sget(K_ETA, idx);
    if is_solid(idx) {
        var acc = 0.0;
        var n = 0.0;
        for (var dj = -2; dj <= 2; dj++) {
            for (var di = -2; di <= 2; di++) {
                if di == 0 && dj == 0 { continue; }
                let ni = i + di;
                let nj = j + dj;
                if !inside(ni, nj) { continue; }
                let nidx = u32(nj) * P.res + u32(ni);
                if is_solid(nidx) || h[nidx] <= P.eps { continue; }
                acc += sget(K_ETA, nidx);
                n += 1.0;
            }
        }
        if n > 0.0 {
            out = acc / n;
        }
    }
    sset(K_TG, idx, out);
}

@compute @workgroup_size(8, 8)
fn k_lift_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    if is_solid(idx) {
        sset(K_ETA, idx, sget(K_TG, idx));
    }
}

fn blur_x(src: u32, gid: vec3<u32>) {
    let i = i32(gid.x);
    let j = i32(gid.y);
    let idx = gid.y * P.res + gid.x;
    var g = 0.0;
    var s = 0.0;
    for (var t = 0u; t < 5u; t++) {
        let x = sget(src, at(i + i32(t) - 2, j));
        g += x * gk(t);
        s += x * sk(t);
    }
    sset(K_TG, idx, g);
    sset(K_TS, idx, s);
}

fn blur_z(gid: vec3<u32>) -> vec2<f32> {
    let i = i32(gid.x);
    let j = i32(gid.y);
    var g = 0.0;
    var s = 0.0;
    for (var t = 0u; t < 5u; t++) {
        let n = at(i, j + i32(t) - 2);
        g += sget(K_TG, n) * gk(t);
        s += sget(K_TS, n) * sk(t);
    }
    return vec2<f32>(g, s);
}

@compute @workgroup_size(8, 8)
fn k_blur_x_eta(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    blur_x(K_ETA, gid);
}

@compute @workgroup_size(8, 8)
fn k_blur_z_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    let gs = blur_z(gid);
    let filtered = 0.62 * gs.x + 0.38 * gs.y;
    sset(K_P, idx, 0.45 * filtered + 0.55 * sget(K_ETA, idx));
}

// ---------------------------------------------------------------- velocity

// No-slip ghost: a solid cell holds minus the mean of its wet open neighbours.
// Reads open cells, writes solid cells, so one pass is race free.
@compute @workgroup_size(8, 8)
fn k_ghost(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    if !is_solid(idx) { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    var au = 0.0;
    var av = 0.0;
    var n = 0.0;
    for (var d = 0u; d < 4u; d++) {
        let ni = i + dir4(d).x;
        let nj = j + dir4(d).y;
        if !inside(ni, nj) { continue; }
        let nidx = u32(nj) * P.res + u32(ni);
        if is_solid(nidx) || h[nidx] <= P.eps { continue; }
        au += u[nidx];
        av += v[nidx];
        n += 1.0;
    }
    if n > 0.0 {
        u[idx] = -au / n;
        v[idx] = -av / n;
    } else {
        u[idx] = 0.0;
        v[idx] = 0.0;
    }
}

fn solid_near1(i: i32, j: i32) -> bool {
    for (var dj = -1; dj <= 1; dj++) {
        for (var di = -1; di <= 1; di++) {
            if di == 0 && dj == 0 { continue; }
            let ni = i + di;
            let nj = j + dj;
            if !inside(ni, nj) { continue; }
            if is_solid(u32(nj) * P.res + u32(ni)) { return true; }
        }
    }
    return false;
}

@compute @workgroup_size(8, 8)
fn k_gravity(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    if h[idx] <= P.eps { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    let g = grad_plane(K_P, i, j);
    var mob = sheet_mobility_for(nu(idx));
    if P.solid != 0u && h[idx] > 0.08 && solid_near1(i, j) {
        mob = max(mob, 1.0);
    }
    u[idx] += -GRAVITY * g.x * mob * P.dt;
    v[idx] += -GRAVITY * g.y * mob * P.dt;
}

@compute @workgroup_size(8, 8)
fn k_vort(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    sset(K_W, idx, vorticity_at(i32(gid.x), i32(gid.y)));
}

// Snapshot Jacobi confinement from the vorticity plane.
fn confine(gid: vec3<u32>) {
    let idx = gid.y * P.res + gid.x;
    if h[idx] <= P.eps { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    let w = sget(K_W, idx);
    if !finite(w) || abs(w) < 1.0e-4 { return; }
    let two = 2.0 * P.cell;
    let wxp = abs(sget(K_W, at(i + 1, j)));
    let wxm = abs(sget(K_W, at(i - 1, j)));
    let wzp = abs(sget(K_W, at(i, j + 1)));
    let wzm = abs(sget(K_W, at(i, j - 1)));
    let nx = (wxp - wxm) / two;
    let nz = (wzp - wzm) / two;
    let du = P.vort * P.dt * (-nz * w);
    let dv = P.vort * P.dt * (nx * w);
    if finite(du) && finite(dv) && sqrt(du * du + dv * dv) <= 0.25 {
        u[idx] += du;
        v[idx] += dv;
    }
}

@compute @workgroup_size(8, 8)
fn k_confine(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    confine(gid);
}

@compute @workgroup_size(8, 8)
fn k_div(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    let i = i32(gid.x);
    let j = i32(gid.y);
    let two = 2.0 * P.cell;
    let ux = (u[at(i + 1, j)] - u[at(i - 1, j)]) / two;
    let vz = (v[at(i, j + 1)] - v[at(i, j - 1)]) / two;
    let e = sget(K_ETA, idx);
    let lap_h = (sget(K_ETA, at(i + 1, j)) + sget(K_ETA, at(i - 1, j))
        + sget(K_ETA, at(i, j + 1)) + sget(K_ETA, at(i, j - 1)) - 4.0 * e)
        / (P.cell * P.cell);
    let overload = 1.0 + clamp(nu(idx) / 231.0, 0.0, 6.0);
    sset(K_B, idx, (ux + vz) * overload - 0.015 * overload * lap_h);
}

@compute @workgroup_size(8, 8)
fn k_blur_x_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    blur_x(K_B, gid);
}

@compute @workgroup_size(8, 8)
fn k_blur_z_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    let gs = blur_z(gid);
    sset(K_Q, idx, 0.62 * gs.x + 0.38 * gs.y);
}

@compute @workgroup_size(8, 8)
fn k_vel(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    if h[idx] <= P.eps { return; }
    var uu = u[idx];
    var vv = v[idx];
    if P.project != 0u {
        let q = grad_plane(K_Q, i32(gid.x), i32(gid.y));
        uu += 0.012 * P.dt * q.x;
        vv += 0.012 * P.dt * q.y;
    }
    let decay = exp(-sheet_damp(nu(idx)) * P.dt);
    u[idx] = finite_speed(uu * decay, 1.6);
    v[idx] = finite_speed(vv * decay, 1.6);
}

// ---------------------------------------------------------------- walls

// Discharge aimed at a wall turns to the open flanks. Pass A: each wet open
// cell books its four shares in direction order, with the CPU room clamp.
@compute @workgroup_size(8, 8)
fn k_redir_a(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    let n = ncell();
    for (var d = 0u; d < 4u; d++) {
        scr[K_RSH * n + 4u * idx + d] = 0.0;
    }
    sset(K_RREM, idx, 0.0);
    let hi = h[idx];
    if is_solid(idx) || hi < 0.02 { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    var removed = 0.0;
    for (var d = 0u; d < 4u; d++) {
        let di = dir4(d).x;
        let dj = dir4(d).y;
        let ni = i + di;
        let nj = j + dj;
        if !inside(ni, nj) { continue; }
        if !is_solid(u32(nj) * P.res + u32(ni)) { continue; }
        var vn = u[idx] * f32(di) + v[idx] * f32(dj);
        let ui = i - di;
        let uj = j - dj;
        if inside(ui, uj) {
            let uidx = u32(uj) * P.res + u32(ui);
            if !is_solid(uidx) {
                let feed = u[uidx] * f32(di) + v[uidx] * f32(dj);
                if feed > vn { vn = feed; }
            }
        }
        if vn < 0.05 { continue; }
        var n_dest = 0.0;
        var flank = 0.0;
        for (var s = 0; s < 2; s++) {
            var pi = 0;
            var pj = 0;
            if di != 0 { pj = select(1, -1, s == 0); } else { pi = select(1, -1, s == 0); }
            let qi = i + pi;
            let qj = j + pj;
            if !inside(qi, qj) { continue; }
            let q = u32(qj) * P.res + u32(qi);
            if is_solid(q) { continue; }
            if h[q] <= 1.0e-6 && bed(q) > bed(idx) + hi + 1.0e-3 { continue; }
            flank += h[q];
            n_dest += 1.0;
        }
        if n_dest == 0.0 { continue; }
        let excess = hi - flank / n_dest;
        if excess < 2.0e-3 { continue; }
        let room = hi - removed;
        if room < 1.0e-6 { continue; }
        let frac = min(vn * P.dt / P.cell, 0.45);
        let dh = min(min(excess, hi * frac), room);
        if dh < 1.0e-8 { continue; }
        removed += dh;
        scr[K_RSH * n + 4u * idx + d] = dh / n_dest;
    }
    sset(K_RREM, idx, removed);
}

fn dest_ok(q: u32, src: u32) -> bool {
    if is_solid(q) { return false; }
    if h[q] <= 1.0e-6 && bed(q) > bed(src) + h[src] + 1.0e-3 { return false; }
    return true;
}

// Pass B: gather. A sender above or below this cell turned its x discharge
// (directions 0, 1) onto us; a sender left or right turned its z discharge.
@compute @workgroup_size(8, 8)
fn k_redir_b(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    let i = i32(gid.x);
    let j = i32(gid.y);
    let n = ncell();
    var added = 0.0;
    {
        // sender at (i, j-1) and (i, j+1): its directions 0 and 1
        for (var s = 0; s < 2; s++) {
            let sj = j + select(1, -1, s == 0);
            if !inside(i, sj) { continue; }
            let src = u32(sj) * P.res + u32(i);
            if !dest_ok(idx, src) { continue; }
            added += scr[K_RSH * n + 4u * src + 0u];
            added += scr[K_RSH * n + 4u * src + 1u];
        }
        // sender at (i-1, j) and (i+1, j): its directions 2 and 3
        for (var s = 0; s < 2; s++) {
            let si = i + select(1, -1, s == 0);
            if !inside(si, j) { continue; }
            let src = u32(j) * P.res + u32(si);
            if !dest_ok(idx, src) { continue; }
            added += scr[K_RSH * n + 4u * src + 2u];
            added += scr[K_RSH * n + 4u * src + 3u];
        }
    }
    let rem = sget(K_RREM, idx);
    var out = h[idx];
    if rem > 0.0 || added > 0.0 {
        out = max(h[idx] - rem + added, 0.0);
    }
    sset(K_H2, idx, out);
}

@compute @workgroup_size(8, 8)
fn k_copy_h2(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    h[idx] = sget(K_H2, idx);
}

// No penetration: drop the velocity that points into a solid neighbour.
@compute @workgroup_size(8, 8)
fn k_block(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    if is_solid(idx) || h[idx] <= P.eps { return; }
    let i = i32(gid.x);
    let j = i32(gid.y);
    var ux = u[idx];
    var vx = v[idx];
    var hit = false;
    for (var d = 0u; d < 4u; d++) {
        let ni = i + dir4(d).x;
        let nj = j + dir4(d).y;
        if !inside(ni, nj) { continue; }
        if !is_solid(u32(nj) * P.res + u32(ni)) { continue; }
        let nx = f32(dir4(d).x);
        let nz = f32(dir4(d).y);
        let vn = ux * nx + vx * nz;
        if vn > 0.0 {
            ux -= vn * nx;
            vx -= vn * nz;
            hit = true;
        }
    }
    if hit {
        u[idx] = ux;
        v[idx] = vx;
    }
}

// ---------------------------------------------------------------- transport

// which = 0 reads h, 1 reads the H2 plane.
fn hsrc(which: u32, idx: u32) -> f32 {
    if which == 0u {
        return h[idx];
    }
    return sget(K_H2, idx);
}

fn speed(axis: u32, idx: u32) -> f32 {
    if axis == 0u {
        return u[idx];
    }
    return v[idx];
}

// Signed column moved from a to b over one face, `move_edge_bufs`.
fn face(which: u32, axis: u32, a: u32, b: u32) -> f32 {
    if P.solid != 0u && (is_solid(a) || is_solid(b)) {
        return 0.0;
    }
    let vel = 0.5 * (speed(axis, a) + speed(axis, b));
    if abs(vel) < 1.0e-5 {
        return 0.0;
    }
    var donor = a;
    var recv = b;
    var sgn = 1.0;
    if vel <= 0.0 {
        donor = b;
        recv = a;
        sgn = -1.0;
    }
    let hd = hsrc(which, donor);
    let hr = hsrc(which, recv);
    if P.solid != 0u {
        if hr <= 1.0e-6 && bed(recv) > bed(donor) + hd + 1.0e-3 {
            return 0.0;
        }
    }
    if hd <= 1.0e-8 {
        return 0.0;
    }
    if !gate(recv) && hr <= 1.0e-6 {
        return 0.0;
    }
    let frac = min(abs(vel) * P.dt / P.cell, 0.45);
    let dh = hd * frac;
    if dh <= 1.0e-8 {
        return 0.0;
    }
    return sgn * dh;
}

fn transport_cell(which: u32, axis: u32, gid: vec3<u32>) -> f32 {
    let idx = gid.y * P.res + gid.x;
    var pos = gid.x;
    var stride = 1u;
    if axis == 1u {
        pos = gid.y;
        stride = P.res;
    }
    var removed = 0.0;
    var added = 0.0;
    if pos > 0u {
        let f = face(which, axis, idx - stride, idx);
        if f > 0.0 { added += f; } else if f < 0.0 { removed += -f; }
    }
    if pos + 1u < P.res {
        let f = face(which, axis, idx, idx + stride);
        if f > 0.0 { removed += f; } else if f < 0.0 { added += -f; }
    }
    let hc = hsrc(which, idx);
    if removed > 0.0 || added > 0.0 {
        return max(hc - removed + added, 0.0);
    }
    return hc;
}

@compute @workgroup_size(8, 8)
fn k_tx(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    sset(K_H2, idx, transport_cell(0u, 0u, gid));
}

@compute @workgroup_size(8, 8)
fn k_tz(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !cell_id(gid) { return; }
    let idx = gid.y * P.res + gid.x;
    let hn = transport_cell(1u, 1u, gid);
    if P.eps > 0.0 && hn < P.eps {
        h[idx] = 0.0;
        u[idx] = 0.0;
        v[idx] = 0.0;
    } else {
        h[idx] = hn;
    }
}
