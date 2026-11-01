// Smoke and fire volume: one fullscreen ray march over the `smoke.wgsl` grid.
//
// Single scattering of sun and sky through soot, with the sun transmittance
// the solver already marched per cell (texture .a) and the scene shadow map,
// so a body in the smoke throws a shadow into it. Flame light is black-body
// emission from the temperature. Lighting model after the approved Bevy
// volumetric_fog.wgsl (MIT/Apache): Henyey-Greenstein phase, Beer-Lambert,
// energy-conserving step integration (Hillaire 2015).
//
// When the dust/fog march lands on main, `smoke_sigma` and the emission term
// move into that loop and this pass goes away.

struct Frame {
    view_proj: mat4x4<f32>,
    light_vp: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,
    sun_dir: vec4<f32>,
    sun_color: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    amb: vec4<f32>,
    params: vec4<f32>,
    misc: vec4<f32>,
    tires: array<vec4<f32>, 8>,
    snow_arc: vec4<f32>,
    snow_arc2: vec4<f32>,
    snow_profile: array<vec4<f32>, 8>,
}

struct SmokeDraw {
    // grid min corner xyz, cell m
    origin: vec4<f32>,
    // grid size m xyz, extinction per metre at soot 1
    size: vec4<f32>,
    // soot albedo rgb, phase g
    albedo: vec4<f32>,
    // emission gain, ambient gain, kelvin per LUT step, LUT steps
    light: vec4<f32>,
    // Black body: linear rgb (max 1), relative power. Entry k is at k * light.z kelvin over ambient.
    lut: array<vec4<f32>, 24>,
    // Fire lights: position xyz, radius; colour rgb (linear, W-ish), _.
    fire_pos: array<vec4<f32>, 4>,
    fire_col: array<vec4<f32>, 4>,
    // number of fire lights, detail gain, base noise frequency (per cell),
    // reference eddy speed m/s
    detail: vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var<uniform> sd: SmokeDraw;
@group(1) @binding(1) var vol: texture_3d<f32>;
@group(1) @binding(2) var vol_samp: sampler;
@group(1) @binding(3) var scene_depth: texture_depth_2d;
@group(1) @binding(4) var shadow_tex: texture_2d<f32>;

@group(1) @binding(5) var flow_vol: texture_3d<f32>;

override QUALITY_SMOKE_STEPS: i32 = 64;
// Multiple-scattering gain on the sun octaves.
const MS_GAIN: f32 = 5.0;
// Curl-noise octaves of sub-grid detail. 0 turns it off.
override QUALITY_SMOKE_DETAIL: i32 = 2;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
}

@vertex
fn vs_smoke(@builtin(vertex_index) i: u32) -> VsOut {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: VsOut;
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

fn decode_depth(c: vec4<f32>) -> f32 {
    return dot(c, vec4<f32>(1.0, 1.0 / 255.0, 1.0 / 65025.0, 1.0 / 16581375.0));
}

// One tap of the sun shadow map: 1 lit, 0 behind the Niva or a prop.
fn scene_sun(p: vec3<f32>) -> f32 {
    let lp = frame.light_vp * vec4<f32>(p, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let dims = vec2<f32>(textureDimensions(shadow_tex));
    let c = clamp(vec2<i32>(uv * dims), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
    let s = decode_depth(textureLoad(shadow_tex, c, 0));
    return select(1.0, 0.0, ndc.z > s + 0.0006);
}

fn hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cos_t, 1.5));
}

fn blackbody(dt_k: f32) -> vec4<f32> {
    let n = i32(sd.light.w);
    let x = clamp(dt_k / sd.light.z, 0.0, f32(n - 1));
    let k = min(i32(x), n - 2);
    return mix(sd.lut[k], sd.lut[k + 1], x - f32(k));
}

// Curl of a vector potential made of three offset simplex noises
// (Bridson et al. 2007, "Curl-noise for procedural fluid flow"). Divergence
// free, so the detail swirls instead of bunching.
fn curl_noise(q: vec3<f32>) -> vec3<f32> {
    let a = sdnoise3(q);
    let b = sdnoise3(q + vec3<f32>(31.416, 47.853, 12.679));
    let c = sdnoise3(q + vec3<f32>(-19.123, 7.391, 53.749));
    return vec3<f32>(
        c.gradient.y - b.gradient.z,
        a.gradient.z - c.gradient.x,
        b.gradient.x - a.gradient.y,
    );
}

// Sub-grid detail after Kim et al. 2008 (wavelet turbulence): the grid
// resolves the column and its big rolls; below one cell, curl noise adds the
// eddies the grid cannot hold, scaled by the unresolved eddy speed the solver
// publishes (|curl v| * h) and falling off per octave by the Kolmogorov
// spectrum, E(k) ~ k^(-5/3), so velocity ~ 2^(-5/6) per octave. Kim advects a
// finer density grid with that velocity; here it displaces the density lookup
// in the march instead, so the grid stays the same size.
fn detail_offset(p: vec3<f32>, flow: vec4<f32>) -> vec3<f32> {
    let gain = sd.detail.y;
    if (gain <= 0.0) {
        return vec3<f32>(0.0);
    }
    let h = sd.origin.w;
    // Eddy speed relative to the reference; quiet air gets no detail.
    // Quiet air still carries some sub-grid eddies (floor 0.4): the
    // estimate only sees the resolved shear.
    let e = clamp(0.4 + flow.w / max(sd.detail.w, 1e-3), 0.0, 1.5);
    // Noise rides with the flow so the detail climbs with the plume.
    let drift = flow.xyz * frame.misc.x;
    var off = vec3<f32>(0.0);
    var scale = sd.detail.z / h;
    var amp = h;
    for (var k: i32 = 0; k < QUALITY_SMOKE_DETAIL; k = k + 1) {
        off = off + curl_noise((p - drift) * scale + vec3<f32>(f32(k) * 17.0)) * amp;
        scale = scale * 2.0;
        amp = amp * 0.5612; // 2^(-5/6)
    }
    return off * gain * e;
}

fn fire_light(p: vec3<f32>) -> vec3<f32> {
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

fn hash(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

fn tonemap(c: vec3<f32>) -> vec3<f32> {
    let mapped = vec3<f32>(1.0) - exp(-max(c, vec3<f32>(0.0)) * 1.08);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_smoke(in: VsOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let res = frame.params.xy;
    let ndc = vec2<f32>(in.pos.x / res.x * 2.0 - 1.0, 1.0 - in.pos.y / res.y * 2.0);
    let far = frame.inv_view_proj * vec4<f32>(ndc, 1.0, 1.0);
    let ro = frame.cam_pos.xyz;
    let rd = normalize(far.xyz / far.w - ro);

    // Stop at the opaque scene.
    let d = textureLoad(scene_depth, px, 0);
    let hit = frame.inv_view_proj * vec4<f32>(ndc, d, 1.0);
    let t_scene = select(1e6, length(hit.xyz / hit.w - ro), d < 1.0);

    let bmin = sd.origin.xyz;
    let bmax = sd.origin.xyz + sd.size.xyz;
    let inv = 1.0 / rd;
    let t0 = (bmin - ro) * inv;
    let t1 = (bmax - ro) * inv;
    let tn = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let tf = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    let t_near = max(tn, 0.0);
    let t_far = min(tf, t_scene);
    if (t_far <= t_near) {
        discard;
    }

    let steps = QUALITY_SMOKE_STEPS;
    let dt = (t_far - t_near) / f32(steps);
    let jitter = hash(in.pos.xy + vec2<f32>(frame.misc.x * 13.1, 0.0));
    let sun = normalize(frame.sun_dir.xyz);
    let phase = hg(dot(rd, sun), sd.albedo.w);
    // Sky light from above and the ground bounce from below, flat.
    let sky = mix(frame.amb.rgb, frame.sky_zenith.rgb, 0.6) * sd.light.y;

    var trans = 1.0;
    var scat = vec3<f32>(0.0);
    var emit = vec3<f32>(0.0);
    for (var k: i32 = 0; k < 64; k = k + 1) {
        if (k >= steps || trans < 0.01) {
            break;
        }
        let t = t_near + (f32(k) + jitter) * dt;
        let p = ro + rd * t;
        var uvw = (p - bmin) / sd.size.xyz;
        var s = textureSampleLevel(vol, vol_samp, uvw, 0.0);
        if (s.r < 1e-4 && s.b < 1e-3) {
            continue;
        }
        let flow = textureSampleLevel(flow_vol, vol_samp, uvw, 0.0);
        let off = detail_offset(p, flow);
        if (dot(off, off) > 0.0) {
            uvw = (p + off - bmin) / sd.size.xyz;
            s = textureSampleLevel(vol, vol_samp, uvw, 0.0);
        }
        let sigma = max(s.r, 0.0) * sd.size.w;
        let flame = max(s.b, 0.0);
        if (sigma < 1e-4 && flame < 1e-3) {
            continue;
        }
        let step_t = exp(-sigma * dt);
        // Energy-conserving: integrate the in-scatter over the step.
        let w = select(dt, (1.0 - step_t) / max(sigma, 1e-5), sigma > 1e-5);
        let lit = scene_sun(p);
        // Thick smoke is bright: the light that single scattering loses comes
        // back after many bounces. Two extra octaves (Wrenninge 2015): each at
        // half the extinction (T^0.5, T^0.25), isotropic, weighted 0.5^k * gain.
        let ta = max(s.a, 0.0);
        let ms = (0.5 * sqrt(ta) + 0.25 * sqrt(sqrt(ta))) * MS_GAIN * (1.0 / (4.0 * 3.14159265));
        let sun_l = frame.sun_color.rgb * lit * (s.a * phase + ms);
        scat = scat + trans * sigma * w * sd.albedo.rgb * (sun_l + sky + fire_light(p));
        // Flame: the burning gas glows by its temperature.
        let bb = blackbody(s.g * 1000.0);
        emit = emit + trans * bb.rgb * bb.w * flame * sd.light.x * dt;
        trans = trans * step_t;
    }
    let alpha = 1.0 - trans;
    if (alpha < 0.002 && max(emit.r, max(emit.g, emit.b)) < 0.002) {
        discard;
    }
    // Premultiplied: smoke colour over the tonemapped scene, flame on top.
    let smoke = tonemap(scat / max(alpha, 1e-3)) * alpha;
    let glow = tonemap(emit);
    return vec4<f32>(min(smoke + glow, vec3<f32>(1.0)), alpha);
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
