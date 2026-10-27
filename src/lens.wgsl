// Water and mud on the camera glass. The pass samples a copy of the frame
// and bends it through each drop, so the scene stays readable behind the glass.
// Colors are already gamma-encoded. Keep BEAD_CAP / RUN_CAP in sync with src/lens.rs.

struct V {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

struct Bead {
    xy: vec2<f32>,
    radii: vec2<f32>,
    params: vec4<f32>,
}

struct Run {
    xy: vec2<f32>,
    len_w: vec2<f32>,
    params: vec4<f32>,
}

struct Lens {
    head: vec4<f32>,
    beads: array<Bead, 28>,
    runs: array<Run, 16>,
}

struct Hit {
    mask: f32,
    n: vec3<f32>,
    strength: f32,
}

@group(0) @binding(0) var<uniform> lens: Lens;
@group(0) @binding(1) var scene_tex: texture_2d<f32>;
@group(0) @binding(2) var scene_samp: sampler;

@vertex
fn vs_lens(@builtin(vertex_index) vid: u32) -> V {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    var o: V;
    o.clip = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}

fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn over(dst: vec4<f32>, src: vec3<f32>, src_a: f32) -> vec4<f32> {
    let a = clamp(src_a, 0.0, 1.0);
    let out_a = a + dst.a * (1.0 - a);
    let out_c = (src * a + dst.rgb * dst.a * (1.0 - a)) / max(out_a, 1e-4);
    return vec4<f32>(out_c, out_a);
}

fn scene_rgb(uv: vec2<f32>) -> vec3<f32> {
    let c = clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0));
    return textureSampleLevel(scene_tex, scene_samp, c, 0.0).rgb;
}

fn miss() -> Hit {
    var hit: Hit;
    hit.mask = 0.0;
    hit.n = vec3<f32>(0.0, 0.0, 1.0);
    hit.strength = 0.0;
    return hit;
}

fn shade_glass(uv: vec2<f32>, n_in: vec3<f32>, aspect: f32, clear: bool, seed: f32, gloss: f32, strength: f32, rim_scale: f32, mud_amt: f32) -> vec3<f32> {
    let n = normalize(n_in);
    let bend = vec2<f32>(n.x / max(aspect, 0.2), n.y) * strength;
    // A convex bead magnifies: the rim samples toward the middle of the drop.
    let col_r = scene_rgb(uv - bend * 1.06).r;
    let col_g = scene_rgb(uv - bend).g;
    let col_b = scene_rgb(uv - bend * 0.94).b;
    var col = vec3<f32>(col_r, col_g, col_b);
    let thick = saturate(n.z);
    if (!clear) {
        let tone = fract(seed * 7.13);
        let tint = mix(vec3<f32>(0.55, 0.32, 0.14), vec3<f32>(0.80, 0.52, 0.26), tone);
        // Caramel only where the drop is thick. Edges stay the scene.
        let density = thick * thick * mud_amt;
        col = col * mix(vec3<f32>(1.0), tint, density);
    }
    // A tight rim, so the middle of the drop stays the refracted scene.
    let rim = pow(saturate(1.0 - thick), 2.4);
    let refl = select(vec3<f32>(0.78, 0.62, 0.40), vec3<f32>(0.98, 0.99, 1.0), clear);
    col = mix(col, refl, rim * rim_scale);
    let catchlight = vec2<f32>(n.x + 0.32, n.y + 0.40);
    let glint = smoothstep(0.18, 0.02, length(catchlight));
    let light = normalize(vec3<f32>(-0.40, -0.62, 0.70));
    let halfv = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
    let sheen = pow(saturate(dot(n, halfv)), 24.0) * 0.07 * gloss;
    col = col + vec3<f32>(1.0, 0.99, 0.96) * (glint * gloss + sheen);
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn bead_hit(uv: vec2<f32>, b: Bead, aspect: f32) -> Hit {
    let rx = max(b.radii.x, 1e-4);
    let ry = max(b.radii.y, 1e-4);
    let q = vec2<f32>((uv.x - b.xy.x) * aspect, uv.y - b.xy.y);
    let p = vec2<f32>(q.x / rx, q.y / ry);
    let r2 = dot(p, p);
    if (r2 >= 1.0) {
        return miss();
    }
    let r = sqrt(r2);
    let nz = sqrt(max(1.0 - r2, 0.0));
    var hit: Hit;
    hit.n = vec3<f32>(p.x, p.y, nz);
    hit.mask = smoothstep(1.0, 0.86, r);
    let clear = b.params.w > 0.5;
    hit.strength = max(rx, ry) * select(0.42, 0.78, clear);
    return hit;
}

fn run_hit(uv: vec2<f32>, s: Run, aspect: f32) -> Hit {
    let slant = clamp(s.params.z, -1.1, 1.1);
    let dir = normalize(vec2<f32>(slant, 1.0));
    let rel = vec2<f32>((uv.x - s.xy.x) * aspect, uv.y - s.xy.y);
    let along = dot(rel, dir);
    let len = max(s.len_w.x, 0.001);
    let width = max(s.len_w.y, 0.0005);
    if (along < -width * 2.0 || along > len + width * 3.0) {
        return miss();
    }
    let wob = 0.94 + 0.06 * sin(along * 30.0 + s.params.y * 11.0);
    let taper = mix(0.78, 1.0, saturate(along / max(len, 1e-3)));
    let rad = width * wob * taper;
    let t = clamp(along, 0.0, len);
    let body = rel - dir * t;
    let body_d = length(body);
    let bulb = rel - dir * len;
    let bulb_d = length(bulb);
    let bulb_r = width * (1.65 + 0.35 * s.params.y);
    let body_u = body_d / max(rad, 1e-4);
    let bulb_u = bulb_d / max(bulb_r, 1e-4);
    if (bulb_u <= body_u) {
        if (bulb_u >= 1.0) {
            return miss();
        }
        let nz = sqrt(max(1.0 - bulb_u * bulb_u, 0.0));
        let dir_b = bulb / max(bulb_d, 1e-4);
        var hit: Hit;
        hit.n = vec3<f32>(dir_b * bulb_u, nz);
        hit.mask = smoothstep(1.0, 0.75, bulb_u);
        hit.strength = max(bulb_r, 0.012) * 0.9;
        return hit;
    }
    if (body_u >= 1.0) {
        return miss();
    }
    let nz = sqrt(max(1.0 - body_u * body_u, 0.0));
    let dir_b = body / max(body_d, 1e-4);
    var hit: Hit;
    hit.n = vec3<f32>(dir_b * body_u, nz);
    hit.mask = smoothstep(1.0, 0.72, body_u);
    hit.strength = max(rad, 0.008) * select(0.7, 1.05, s.params.w > 0.5);
    return hit;
}

fn mist(uv: vec2<f32>, film: f32, scroll: f32, aspect: f32) -> vec4<f32> {
    if (film < 0.05) {
        return vec4<f32>(0.0);
    }
    let gy = uv.y - scroll;
    let cell = vec2<f32>(floor(uv.x * 46.0), floor(gy * 26.0));
    let h = hash12(cell);
    let gate = mix(0.992, 0.962, clamp(film, 0.0, 1.0));
    if (h < gate) {
        return vec4<f32>(0.0);
    }
    let j = vec2<f32>(hash12(cell + vec2<f32>(19.2, 3.7)), hash12(cell + vec2<f32>(4.1, 27.7)));
    let center = (cell + vec2<f32>(0.18, 0.18) + j * 0.64) / vec2<f32>(46.0, 26.0);
    let q = vec2<f32>((uv.x - center.x) * aspect, gy - center.y);
    let rad = 0.0035 + h * 0.0045;
    let r = length(q);
    if (r > rad) {
        return vec4<f32>(0.0);
    }
    let nrm = r / rad;
    let p = q / rad;
    let nz = sqrt(max(1.0 - nrm * nrm, 0.0));
    let n = normalize(vec3<f32>(p.x, p.y, nz));
    let light = normalize(vec3<f32>(-0.40, -0.62, 0.68));
    let halfv = normalize(light + vec3<f32>(0.0, 0.0, 1.0));
    let spec = pow(saturate(dot(n, halfv)), 40.0);
    let rim = pow(1.0 - nz, 3.0);
    let col = mix(vec3<f32>(0.75, 0.84, 0.9), vec3<f32>(1.0, 1.0, 1.0), spec);
    let a = spec * 0.8 + rim * 0.28;
    return vec4<f32>(col, a);
}

@fragment
fn fs_lens(in: V) -> @location(0) vec4<f32> {
    let aspect = max(lens.head.z, 0.2);
    let film = lens.head.x;
    let scroll = lens.head.y;
    var acc = vec4<f32>(0.0);
    let specks = mist(in.uv, film, scroll, aspect);
    if (specks.a > 0.01) {
        acc = over(acc, specks.rgb, specks.a);
    }

    for (var i = 0u; i < 16u; i = i + 1u) {
        let s = lens.runs[i];
        if (s.params.x < 0.03 || s.len_w.x < 0.001) {
            continue;
        }
        let hit = run_hit(in.uv, s, aspect);
        if (hit.mask < 0.02) {
            continue;
        }
        let clear = s.params.w > 0.5;
        let col = shade_glass(in.uv, hit.n, aspect, clear, s.params.y, select(0.55, 0.8, clear), hit.strength, select(0.55, 0.75, clear), select(0.1, 0.0, clear));
        acc = over(acc, col, s.params.x * hit.mask);
    }

    for (var i = 0u; i < 28u; i = i + 1u) {
        let b = lens.beads[i];
        if (b.params.x < 0.03 || b.radii.x < 1e-4) {
            continue;
        }
        let hit = bead_hit(in.uv, b, aspect);
        if (hit.mask < 0.02) {
            continue;
        }
        let clear = b.params.w > 0.5;
        let col = shade_glass(in.uv, hit.n, aspect, clear, b.params.y, b.params.z, hit.strength, select(0.42, 0.8, clear), select(0.55, 0.0, clear));
        acc = over(acc, col, b.params.x * hit.mask);
    }

    if (acc.a < 0.02) {
        discard;
    }
    return vec4<f32>(acc.rgb, acc.a);
}
