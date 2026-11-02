// Post-processing prototype. See src/post.rs and docs/LOOK_GAP.md.
// Own code. Techniques: ACES fitted curve (Hill, as in MJP BakingLab, MIT),
// 13-tap bloom downsample (Jimenez, SIGGRAPH 2014), FXAA-style edge blend
// (Lottes 2009 paper), linear-sampled 9-tap Gaussian.

override POST_BLOOM: u32 = 1u;
// 1: t0 is the linear HDR scene target. 0: it is the 8-bit display image.
override POST_HDR: u32 = 0u;

struct V {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@group(0) @binding(0) var t0: texture_2d<f32>;
@group(0) @binding(1) var t1: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;

// Look. Exposure lines mid grey up with the old curve; the rest is the look.
const EXPOSURE: f32 = 1.45;
const BLOOM_THRESHOLD: f32 = 1.7;
const BLOOM_KNEE: f32 = 0.8;
const BLOOM_STRENGTH: f32 = 0.22;
// Bloom targets are 8-bit; this scale keeps up to 4.0 in range.
const BLOOM_ENCODE: f32 = 0.25;
const SATURATION: f32 = 1.10;
const CONTRAST: f32 = 1.08;
const VIGNETTE: f32 = 0.22;
const SHARPEN: f32 = 0.30;

@vertex
fn vs_post(@builtin(vertex_index) vid: u32) -> V {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    var o: V;
    o.clip = vec4<f32>(x, y, 0.0, 1.0);
    o.uv = vec2<f32>(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
    return o;
}

// The display value is gamma(ACES fit(c)) (`aces_fit` in shader.wgsl). Undo it.
fn decode(s: vec3<f32>) -> vec3<f32> {
    let m = min(pow(clamp(s, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(2.2)), vec3<f32>(0.999));
    let qa = vec3<f32>(2.51) - 2.43 * m;
    let qb = vec3<f32>(0.03) - 0.59 * m;
    let qc = -0.14 * m;
    return (-qb + sqrt(max(qb * qb - 4.0 * qa * qc, vec3<f32>(0.0)))) / (2.0 * qa);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

// The display-encoded value, what the edge blend and grade were tuned on.
fn encode(c: vec3<f32>) -> vec3<f32> {
    let x = max(c, vec3<f32>(0.0));
    let m = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return pow(clamp(m, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(1.0 / 2.2));
}

fn raw(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(t0, samp, uv, 0.0).rgb;
}

fn src(uv: vec2<f32>) -> vec3<f32> {
    if (POST_HDR == 1u) {
        return encode(raw(uv));
    }
    return raw(uv);
}

// Linear light: exact from the HDR target, decoded from the 8-bit one.
fn src_linear(uv: vec2<f32>) -> vec3<f32> {
    if (POST_HDR == 1u) {
        return max(raw(uv), vec3<f32>(0.0));
    }
    return decode(raw(uv));
}

// ---------------------------------------------------------------- bloom

fn bright(c_lin: vec3<f32>) -> vec3<f32> {
    let c = c_lin * EXPOSURE;
    let l = max(c.r, max(c.g, c.b));
    // Soft knee: quadratic ramp from threshold - knee to threshold + knee.
    var soft = clamp(l - BLOOM_THRESHOLD + BLOOM_KNEE, 0.0, 2.0 * BLOOM_KNEE);
    soft = soft * soft / (4.0 * BLOOM_KNEE + 1e-4);
    let w = max(soft, l - BLOOM_THRESHOLD) / max(l, 1e-4);
    return c * w;
}

@fragment
fn fs_bloom_down(in: V) -> @location(0) vec4<f32> {
    // Output is 1/4 resolution; the 13 taps cover about 8x8 source texels.
    let px = 2.0 / vec2<f32>(textureDimensions(t0));
    let uv = in.uv;
    let a = bright(src_linear(uv + px * vec2<f32>(-2.0, -2.0)));
    let b = bright(src_linear(uv + px * vec2<f32>(0.0, -2.0)));
    let c = bright(src_linear(uv + px * vec2<f32>(2.0, -2.0)));
    let d = bright(src_linear(uv + px * vec2<f32>(-2.0, 0.0)));
    let e = bright(src_linear(uv));
    let f = bright(src_linear(uv + px * vec2<f32>(2.0, 0.0)));
    let g = bright(src_linear(uv + px * vec2<f32>(-2.0, 2.0)));
    let h = bright(src_linear(uv + px * vec2<f32>(0.0, 2.0)));
    let i = bright(src_linear(uv + px * vec2<f32>(2.0, 2.0)));
    let j = bright(src_linear(uv + px * vec2<f32>(-1.0, -1.0)));
    let k = bright(src_linear(uv + px * vec2<f32>(1.0, -1.0)));
    let l = bright(src_linear(uv + px * vec2<f32>(-1.0, 1.0)));
    let m = bright(src_linear(uv + px * vec2<f32>(1.0, 1.0)));
    var o = e * 0.125;
    o += (a + c + g + i) * 0.03125;
    o += (b + d + f + h) * 0.0625;
    o += (j + k + l + m) * 0.125;
    return vec4<f32>(sqrt(clamp(o * BLOOM_ENCODE, vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}

fn blur(uv: vec2<f32>, dir: vec2<f32>) -> vec4<f32> {
    let px = dir / vec2<f32>(textureDimensions(t0));
    // 9-tap Gaussian folded into 5 bilinear fetches. Values are sqrt-encoded,
    // so square, blur, re-encode.
    let w0 = 0.2270270270;
    let w1 = 0.3162162162;
    let w2 = 0.0702702703;
    let o1 = 1.3846153846;
    let o2 = 3.2307692308;
    var c = pow(textureSampleLevel(t0, samp, uv, 0.0).rgb, vec3<f32>(2.0)) * w0;
    c += pow(textureSampleLevel(t0, samp, uv + px * o1, 0.0).rgb, vec3<f32>(2.0)) * w1;
    c += pow(textureSampleLevel(t0, samp, uv - px * o1, 0.0).rgb, vec3<f32>(2.0)) * w1;
    c += pow(textureSampleLevel(t0, samp, uv + px * o2, 0.0).rgb, vec3<f32>(2.0)) * w2;
    c += pow(textureSampleLevel(t0, samp, uv - px * o2, 0.0).rgb, vec3<f32>(2.0)) * w2;
    return vec4<f32>(sqrt(c), 1.0);
}

@fragment
fn fs_bloom_blur_h(in: V) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(1.5, 0.0));
}

@fragment
fn fs_bloom_blur_v(in: V) -> @location(0) vec4<f32> {
    return blur(in.uv, vec2<f32>(0.0, 1.5));
}

// ---------------------------------------------------------------- final

fn aces_fitted(c_in: vec3<f32>) -> vec3<f32> {
    // sRGB -> ACES AP1-ish input (RRT_SAT folded in), rows.
    let v = vec3<f32>(
        dot(c_in, vec3<f32>(0.59719, 0.35458, 0.04823)),
        dot(c_in, vec3<f32>(0.07600, 0.90834, 0.01566)),
        dot(c_in, vec3<f32>(0.02840, 0.13383, 0.83777)),
    );
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    let r = a / b;
    let o = vec3<f32>(
        dot(r, vec3<f32>(1.60475, -0.53108, -0.07367)),
        dot(r, vec3<f32>(-0.10208, 1.10813, -0.00605)),
        dot(r, vec3<f32>(-0.00327, -0.07276, 1.07602)),
    );
    return clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
}

// FXAA-style: find the edge from the 2x2 diagonal luma, blend along it.
// Away from edges, a light unsharp mask instead (the 8-bit scene is soft).
fn edge_aa(uv: vec2<f32>) -> vec3<f32> {
    let px = 1.0 / vec2<f32>(textureDimensions(t0));
    let rgb_m = src(uv);
    let rgb_nw = src(uv + vec2<f32>(-1.0, -1.0) * px);
    let rgb_ne = src(uv + vec2<f32>(1.0, -1.0) * px);
    let rgb_sw = src(uv + vec2<f32>(-1.0, 1.0) * px);
    let rgb_se = src(uv + vec2<f32>(1.0, 1.0) * px);
    let l_m = luma(rgb_m);
    let l_nw = luma(rgb_nw);
    let l_ne = luma(rgb_ne);
    let l_sw = luma(rgb_sw);
    let l_se = luma(rgb_se);
    let l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    let l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    let range = l_max - l_min;
    if (range < max(0.04, l_max * 0.15)) {
        let avg = (rgb_nw + rgb_ne + rgb_sw + rgb_se) * 0.25;
        return clamp(rgb_m + (rgb_m - avg) * SHARPEN, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    var dir = vec2<f32>(
        -((l_nw + l_ne) - (l_sw + l_se)),
        (l_nw + l_sw) - (l_ne + l_se),
    );
    let reduce = max((l_nw + l_ne + l_sw + l_se) * 0.03125, 1.0 / 128.0);
    let scale = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * scale, vec2<f32>(-8.0), vec2<f32>(8.0)) * px;
    let rgb_a = 0.5 * (src(uv + dir * (1.0 / 3.0 - 0.5)) + src(uv + dir * (2.0 / 3.0 - 0.5)));
    let rgb_b = rgb_a * 0.5 + 0.25 * (src(uv - dir * 0.5) + src(uv + dir * 0.5));
    let l_b = luma(rgb_b);
    if (l_b < l_min || l_b > l_max) {
        return rgb_a;
    }
    return rgb_b;
}

@fragment
fn fs_post(in: V) -> @location(0) vec4<f32> {
    var c = decode(edge_aa(in.uv)) * EXPOSURE;
    if (POST_BLOOM == 1u) {
        let b = textureSampleLevel(t1, samp, in.uv, 0.0).rgb;
        c += b * b / BLOOM_ENCODE * BLOOM_STRENGTH;
    }
    var d = aces_fitted(c);
    // Grade in display-linear space.
    let l = luma(d);
    d = mix(vec3<f32>(l), d, SATURATION);
    // Slight S around mid grey (in gamma space so it acts on what you see).
    var g = pow(max(d, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    g = clamp((g - 0.5) * CONTRAST + 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
    // Split tone: cool shadows (sky fill), neutral highlights. A warm
    // highlight tint turned snow cream, so highlights stay untouched.
    let lg = luma(g);
    g *= mix(vec3<f32>(0.965, 0.99, 1.045), vec3<f32>(1.0), smoothstep(0.15, 0.6, lg));
    // Vignette.
    let q = in.uv - 0.5;
    let r2 = dot(q, q) * 2.0;
    g *= 1.0 - VIGNETTE * r2 * r2;
    return vec4<f32>(clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
