// Camera glass after a puddle. Colors are already in the framebuffer's
// gamma space: caramel mud, clear beads with a white catchlight.
// Keep BEAD_CAP / RUN_CAP in sync with src/lens.rs.

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

@group(0) @binding(0) var<uniform> lens: Lens;

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

fn mud_rgb(seed: f32, shade: f32) -> vec3<f32> {
    let tan = vec3<f32>(0.50, 0.26, 0.10);
    let dark = vec3<f32>(0.16, 0.07, 0.028);
    let tone = fract(seed * 7.13);
    return mix(dark, tan, 0.22 + 0.7 * tone) * (0.78 + 0.28 * shade);
}

fn bead_mask(uv: vec2<f32>, b: Bead, aspect: f32) -> vec2<f32> {
    let q = vec2<f32>((uv.x - b.xy.x) * aspect, uv.y - b.xy.y);
    var p = vec2<f32>(q.x / max(b.radii.x, 1e-4), q.y / max(b.radii.y, 1e-4));
    let tail = smoothstep(0.15, -0.95, p.y);
    p.x = p.x / mix(1.0, 0.42, tail);
    let ang = atan2(p.y, p.x);
    let wob = 0.84 + 0.14 * sin(ang * 3.0 + b.params.y * 6.28318) + 0.05 * sin(ang * 7.0 + b.params.y);
    let r = length(p);
    let mask = smoothstep(wob, wob * 0.55, r);
    return vec2<f32>(mask, r / max(wob, 1e-3));
}

fn run_mask(uv: vec2<f32>, s: Run, aspect: f32) -> vec2<f32> {
    let slant = clamp(s.params.z, -1.4, 1.4);
    let dir = normalize(vec2<f32>(slant, 1.0));
    let rel = vec2<f32>((uv.x - s.xy.x) * aspect, uv.y - s.xy.y);
    let along = dot(rel, dir);
    let across = dot(rel, vec2<f32>(dir.y, -dir.x));
    let len = max(s.len_w.x, 0.001);
    let width = max(s.len_w.y, 0.0008);
    let t = clamp(along / len, 0.0, 1.0);
    let wob = 1.0 + 0.38 * sin(along * 28.0 + s.params.y * 17.0) + 0.16 * sin(along * 9.0 + 2.0);
    let bulge = exp(-pow((t - 0.78) * 7.0, 2.0)) * width * (0.45 + 0.35 * s.params.y);
    let half = (width * mix(0.45, 1.15, smoothstep(0.05, 0.85, t)) + bulge) * wob;
    var body = 0.0;
    if (along > -0.01 && along < len + width) {
        let cap = smoothstep(-0.01, 0.03, along) * smoothstep(len + width, len * 0.92, along);
        body = smoothstep(half, half * 0.22, abs(across)) * cap;
    }
    let bulb_rel = rel - dir * len;
    let bulb_r = width * (1.35 + 0.45 * s.params.y);
    let bulb = smoothstep(bulb_r, bulb_r * 0.2, length(bulb_rel));
    let mask = max(body, bulb);
    let shade = select(0.85, 0.62, bulb > body);
    return vec2<f32>(mask, shade);
}

fn mist(uv: vec2<f32>, film: f32, scroll: f32, aspect: f32) -> vec4<f32> {
    if (film < 0.05) {
        return vec4<f32>(0.0);
    }
    let gy = uv.y - scroll;
    let cell = vec2<f32>(floor(uv.x * 46.0), floor(gy * 26.0));
    let h = hash12(cell);
    let gate = mix(0.992, 0.955, clamp(film, 0.0, 1.0));
    if (h < gate) {
        return vec4<f32>(0.0);
    }
    let j = vec2<f32>(hash12(cell + vec2<f32>(19.2, 3.7)), hash12(cell + vec2<f32>(4.1, 27.7)));
    let center = (cell + vec2<f32>(0.18, 0.18) + j * 0.64) / vec2<f32>(46.0, 26.0);
    let q = vec2<f32>((uv.x - center.x) * aspect, gy - center.y);
    let rad = 0.0045 + h * 0.007;
    let r = length(q);
    if (r > rad) {
        return vec4<f32>(0.0);
    }
    let n = r / rad;
    let rim = smoothstep(0.45, 0.82, n) * smoothstep(1.0, 0.82, n);
    let spec = smoothstep(0.28, 0.0, length(q - vec2<f32>(-rad * 0.32, -rad * 0.34)) / rad);
    let col = mix(vec3<f32>(0.22, 0.16, 0.1), vec3<f32>(0.96, 0.97, 0.98), spec);
    let a = clamp(spec * 0.85 + rim * 0.4 + (1.0 - n) * 0.12, 0.0, 0.9);
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
        let alpha = s.params.x;
        if (alpha < 0.03) {
            continue;
        }
        let m = run_mask(in.uv, s, aspect);
        if (m.x < 0.02) {
            continue;
        }
        let clear = s.params.w > 0.5;
        var col: vec3<f32>;
        var a: f32;
        if (clear) {
            let rim = smoothstep(0.25, 0.75, m.x) * smoothstep(1.0, 0.8, m.x);
            col = vec3<f32>(0.8, 0.86, 0.9);
            a = alpha * rim * 0.55;
        } else {
            let body = mud_rgb(s.params.y, m.y);
            let rim = vec3<f32>(0.72, 0.48, 0.22);
            col = mix(body, rim, smoothstep(0.78, 0.98, m.x) * 0.35);
            let glint = smoothstep(0.72, 1.0, m.y) * smoothstep(0.15, 0.55, m.x);
            col = col + vec3<f32>(0.28, 0.16, 0.06) * glint * 0.45;
            a = alpha * smoothstep(0.0, 0.45, m.x);
        }
        acc = over(acc, col, a);
    }

    for (var i = 0u; i < 28u; i = i + 1u) {
        let b = lens.beads[i];
        let alpha = b.params.x;
        if (alpha < 0.03 || b.radii.x < 1e-4) {
            continue;
        }
        let m = bead_mask(in.uv, b, aspect);
        if (m.x < 0.02) {
            continue;
        }
        let clear = b.params.w > 0.5;
        let q = vec2<f32>((in.uv.x - b.xy.x) * aspect, in.uv.y - b.xy.y);
        let spec_at = q / max(b.radii, vec2<f32>(1e-4)) - vec2<f32>(-0.32, -0.38);
        let spec = smoothstep(0.34, 0.0, length(spec_at)) * b.params.z;
        var col: vec3<f32>;
        var a: f32;
        if (clear) {
            let rim = smoothstep(0.5, 0.86, m.y) * smoothstep(1.02, 0.78, m.y);
            col = mix(vec3<f32>(0.2, 0.14, 0.09), vec3<f32>(0.97, 0.98, 1.0), spec);
            a = alpha * (spec * 0.9 + rim * 0.5 + m.x * 0.14);
        } else {
            let body = mud_rgb(b.params.y, 1.0 - m.y * 0.15);
            let rim = vec3<f32>(0.74, 0.46, 0.2);
            col = mix(body, rim, smoothstep(0.7, 0.96, m.y) * 0.4);
            col = col + vec3<f32>(0.85, 0.7, 0.45) * spec * 0.7;
            let lump = hash12(floor(in.uv * vec2<f32>(140.0, 90.0)) + b.params.y);
            col = col * (0.88 + 0.18 * lump);
            a = alpha * smoothstep(0.0, 0.35, m.x);
        }
        acc = over(acc, col, a);
    }

    if (acc.a < 0.02) {
        discard;
    }
    return vec4<f32>(acc.rgb, acc.a);
}
