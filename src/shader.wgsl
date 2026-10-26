// Silt forward shading. Soil height is a real displacement; mud on the buggy
// is a mask driven by the adhesion state, not a flat tint.

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
    params: vec4<f32>, // width, height, cell, map res
    misc: vec4<f32>,   // time, shadow size, _, _
}

struct Model {
    world: mat4x4<f32>,
    mud0: vec4<f32>,
    mud1: vec4<f32>,
    mud2: vec4<f32>,
    mud3: vec4<f32>,
    mud4: vec4<f32>,
    mud5: vec4<f32>,
    mud6: vec4<f32>,
    mud7: vec4<f32>,
    mud8: vec4<f32>,
    mud9: vec4<f32>,
    mud10: vec4<f32>,
    mud11: vec4<f32>,
    extra: vec4<f32>, // is_wheel, wheel mud 0..1, _, _
}

@group(0) @binding(0) var<uniform> frame: Frame;

@group(1) @binding(0) var height_tex: texture_2d<f32>;
@group(1) @binding(1) var mat_tex: texture_2d<f32>;
@group(1) @binding(2) var mat_samp: sampler;
@group(1) @binding(3) var shadow_tex: texture_depth_2d;

@group(2) @binding(0) var<uniform> model: Model;

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

fn noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash12(i);
    let b = hash12(i + vec2<f32>(1.0, 0.0));
    let c = hash12(i + vec2<f32>(0.0, 1.0));
    let d = hash12(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn mud_of(panel: i32) -> vec4<f32> {
    if (panel == 0) { return model.mud0; }
    else if (panel == 1) { return model.mud1; }
    else if (panel == 2) { return model.mud2; }
    else if (panel == 3) { return model.mud3; }
    else if (panel == 4) { return model.mud4; }
    else if (panel == 5) { return model.mud5; }
    else if (panel == 6) { return model.mud6; }
    else if (panel == 7) { return model.mud7; }
    else if (panel == 8) { return model.mud8; }
    else if (panel == 9) { return model.mud9; }
    else if (panel == 10) { return model.mud10; }
    else if (panel == 11) { return model.mud11; }
    return vec4<f32>(0.0);
}

fn present(c: vec3<f32>) -> vec4<f32> {
    let mapped = vec3<f32>(1.0) - exp(-max(c, vec3<f32>(0.0)) * 1.08);
    return vec4<f32>(pow(mapped, vec3<f32>(1.0 / 2.2)), 1.0);
}

fn sky_color(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let h = clamp(dir.y, -0.05, 1.0);
    var col = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, smoothstep(-0.02, 0.55, h));
    let sun = normalize(frame.sun_dir.xyz);
    let d = max(dot(dir, sun), 0.0);
    col += frame.sun_color.rgb * pow(d, 900.0) * 4.0;
    col += frame.sun_color.rgb * pow(d, 18.0) * 0.55;
    let p = dir.xz / (abs(dir.y) * 0.45 + 0.25);
    let n = noise(p * 0.18);
    let cloud = smoothstep(0.58, 0.82, n) * smoothstep(0.08, 0.4, dir.y);
    col = mix(col, vec3<f32>(0.93, 0.94, 0.96), cloud * 0.7);
    return col;
}

fn shadow_factor(world: vec3<f32>, n: vec3<f32>) -> f32 {
    let lp = frame.light_vp * vec4<f32>(world, 1.0);
    let ndc = lp.xyz / lp.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if (uv.x < 0.0 || uv.y < 0.0 || uv.x > 1.0 || uv.y > 1.0 || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }
    let dims = vec2<f32>(textureDimensions(shadow_tex));
    let base = vec2<i32>(uv * dims);
    let ndl = clamp(dot(n, normalize(frame.sun_dir.xyz)), 0.0, 1.0);
    let bias = 0.0018 + (1.0 - ndl) * 0.0045;
    var blocked = 0.0;
    for (var y: i32 = -1; y <= 1; y = y + 1) {
        for (var x: i32 = -1; x <= 1; x = x + 1) {
            let c = clamp(base + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(dims) - vec2<i32>(1));
            let s = textureLoad(shadow_tex, c, 0);
            if (ndc.z > s + bias) {
                blocked = blocked + 1.0;
            }
        }
    }
    return 1.0 - blocked / 9.0 * 0.72;
}

fn shade(world: vec3<f32>, n_in: vec3<f32>, albedo: vec3<f32>, rough: f32, emit: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let hemi = mix(frame.amb.rgb, frame.sky_zenith.rgb, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let shadow = shadow_factor(world + n * 0.04, n);
    let view = normalize(frame.cam_pos.xyz - world);
    let halfv = normalize(sun + view);
    let spec_pow = mix(10.0, 140.0, clamp(1.0 - rough, 0.0, 1.0));
    let spec = pow(clamp(dot(n, halfv), 0.0, 1.0), spec_pow) * (1.0 - rough) * shadow;
    var col = albedo * (hemi * 0.55 + frame.sun_color.rgb * ndl * shadow * 1.15);
    col += frame.sun_color.rgb * spec * 0.45;
    col += emit;
    let dist = length(world - frame.cam_pos.xyz);
    let fog = smoothstep(32.0, 78.0, dist);
    col = mix(col, frame.sky_horizon.rgb, fog);
    return col;
}

@vertex
fn vs_sky(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid & 1u) * 4 - 1);
    let y = f32(i32((vid >> 1u) & 1u) * 4 - 1);
    return vec4<f32>(x, y, 1.0, 1.0);
}

@fragment
fn fs_sky(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy / max(frame.params.xy, vec2<f32>(1.0));
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let w = frame.inv_view_proj * ndc;
    let dir = w.xyz / w.w - frame.cam_pos.xyz;
    return present(sky_color(dir));
}

struct TerrIn {
    @location(0) xz: vec2<f32>,
    @location(1) ij: vec2<f32>,
}

struct TerrOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) ij: vec2<f32>,
}

fn height_at(ij: vec2<i32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(height_tex));
    let c = clamp(ij, vec2<i32>(0), dims - vec2<i32>(1));
    return textureLoad(height_tex, c, 0).r;
}

@vertex
fn vs_terrain(in: TerrIn) -> TerrOut {
    let ij = vec2<i32>(in.ij);
    let h = height_at(ij);
    let world = vec3<f32>(in.xz.x, h, in.xz.y);
    var o: TerrOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    return o;
}

@vertex
fn vs_shadow_terrain(in: TerrIn) -> @builtin(position) vec4<f32> {
    let h = height_at(vec2<i32>(in.ij));
    let world = vec3<f32>(in.xz.x, h, in.xz.y);
    return frame.light_vp * vec4<f32>(world, 1.0);
}

@fragment
fn fs_terrain(in: TerrOut) -> @location(0) vec4<f32> {
    let ij = vec2<i32>(in.ij);
    let h = height_at(ij);
    // Mostly smooth, with enough slope left that a rut still catches the sun.
    var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    if (n.y < 0.0) {
        n = -n;
    }
    n = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), n, 0.48));
    let bump = noise(in.world.xz * 3.1);
    let bump_r = noise(in.world.xz * 3.1 + vec2<f32>(0.35, 0.0));
    let bump_u = noise(in.world.xz * 3.1 + vec2<f32>(0.0, 0.35));
    n = normalize(n + vec3<f32>((bump - bump_r), 0.0, (bump - bump_u)) * 0.55);

    let mat_dims = vec2<f32>(textureDimensions(mat_tex));
    let mat = textureSampleLevel(mat_tex, mat_samp, (in.ij + vec2<f32>(0.5)) / mat_dims, 0.0);
    let wet = mat.r;
    let water = mat.g * 0.34;
    let darken = mat.b;
    let rough = clamp(mat.a, 0.05, 1.0);

    let grit = noise(in.world.xz * 1.7) * 0.65 + noise(in.world.xz * 6.5) * 0.35;
    let dry = vec3<f32>(0.20, 0.11, 0.055) * (0.88 + grit * 0.22);
    let soaked = vec3<f32>(0.035, 0.018, 0.012);
    var albedo = mix(dry, soaked, smoothstep(0.15, 0.92, wet));
    albedo = mix(albedo, albedo * vec3<f32>(0.32, 0.26, 0.20), smoothstep(0.08, 0.62, darken));

    let avg = (
        height_at(ij + vec2<i32>(1, 0)) + height_at(ij + vec2<i32>(-1, 0))
        + height_at(ij + vec2<i32>(0, 1)) + height_at(ij + vec2<i32>(0, -1))
    ) * 0.25;
    let rut = clamp((avg - h) / 0.07, 0.0, 1.0);
    albedo *= mix(1.0, 0.55, rut);

    var rough_out = mix(rough, 0.08, smoothstep(0.01, 0.06, water));
    var col = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));

    if (water > 0.012) {
        let view = normalize(frame.cam_pos.xyz - in.world);
        let sun = normalize(frame.sun_dir.xyz);
        let refl = reflect(-view, n);
        let fres = pow(1.0 - clamp(dot(n, view), 0.0, 1.0), 3.0);
        let mirror = sky_color(refl);
        let spec = pow(clamp(dot(n, normalize(sun + view)), 0.0, 1.0), 180.0);
        let glaze = mix(vec3<f32>(0.03, 0.045, 0.04), mirror, 0.35 + fres * 0.6);
        col = mix(col, glaze, smoothstep(0.012, 0.05, water) * (0.45 + fres * 0.4));
        col += frame.sun_color.rgb * spec * shadow_factor(in.world, n) * 0.8;
        rough_out = 0.05;
    }
    return present(col);
}

struct WaterOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) ij: vec2<f32>,
    @location(2) depth: f32,
}

@vertex
fn vs_water(in: TerrIn) -> WaterOut {
    let ij = vec2<i32>(in.ij);
    let h = height_at(ij);
    let mat = textureLoad(mat_tex, clamp(ij, vec2<i32>(0), vec2<i32>(textureDimensions(mat_tex)) - vec2<i32>(1)), 0);
    let depth = mat.g * 0.34;
    let world = vec3<f32>(in.xz.x, h + depth * 0.92, in.xz.y);
    var o: WaterOut;
    o.clip = frame.view_proj * vec4<f32>(world, 1.0);
    o.world = world;
    o.ij = in.ij;
    o.depth = depth;
    return o;
}

@fragment
fn fs_water(in: WaterOut) -> @location(0) vec4<f32> {
    if (in.depth < 0.008) {
        discard;
    }
    var n = normalize(cross(dpdx(in.world), dpdy(in.world)));
    if (n.y < 0.0) {
        n = -n;
    }
    n = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), n, 0.22));
    let rip = sin(in.world.x * 5.5 + frame.misc.x * 2.2) * cos(in.world.z * 4.8 - frame.misc.x * 1.7);
    n = normalize(n + vec3<f32>(rip, 0.0, cos(in.world.x * 4.0 + in.world.z * 3.2)) * 0.03);

    let view = normalize(frame.cam_pos.xyz - in.world);
    let sun = normalize(frame.sun_dir.xyz);
    let fres = pow(1.0 - clamp(dot(normalize(n), view), 0.0, 1.0), 3.2);
    let mirror = sky_color(reflect(-view, n));
    let spec = pow(clamp(dot(n, normalize(sun + view)), 0.0, 1.0), 240.0);
    var col = mix(vec3<f32>(0.04, 0.06, 0.055), mirror, 0.25 + fres * 0.7);
    col += frame.sun_color.rgb * spec * shadow_factor(in.world, n);
    let shore = smoothstep(0.008, 0.03, in.depth);
    let alpha = shore * (0.42 + fres * 0.5);
    let mapped = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * 1.08);
    let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, alpha);
}

struct MeshIn {
    @location(0) pos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) panel: f32,
}

struct MeshOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) panel: f32,
}

@vertex
fn vs_mesh(in: MeshIn) -> MeshOut {
    let world = model.world * vec4<f32>(in.pos, 1.0);
    var o: MeshOut;
    o.clip = frame.view_proj * world;
    o.world = world.xyz;
    o.nrm = normalize((model.world * vec4<f32>(in.nrm, 0.0)).xyz);
    o.uv = in.uv;
    o.panel = in.panel;
    return o;
}

@vertex
fn vs_shadow_mesh(in: MeshIn) -> @builtin(position) vec4<f32> {
    let world = model.world * vec4<f32>(in.pos, 1.0);
    return frame.light_vp * world;
}

fn panel_paint(panel: f32, uv: vec2<f32>, world: vec3<f32>) -> vec4<f32> {
    // rgb albedo, a roughness. Panel ids match the vehicle solver.
    let grit = hash12(floor(world.xz * 8.0 + vec2<f32>(world.y, world.y * 0.3)));
    if (panel < 0.5) {
        return vec4<f32>(vec3<f32>(0.16, 0.17, 0.18) * (0.9 + grit * 0.15), 0.38);
    } else if (panel < 1.5) {
        return vec4<f32>(vec3<f32>(0.78, 0.24, 0.035) * (0.92 + grit * 0.12), 0.42);
    } else if (panel < 2.5) {
        return vec4<f32>(vec3<f32>(0.72, 0.20, 0.03), 0.45);
    } else if (panel < 3.5) {
        return vec4<f32>(vec3<f32>(0.70, 0.19, 0.028), 0.45);
    } else if (panel < 4.5) {
        return vec4<f32>(vec3<f32>(0.62, 0.16, 0.025), 0.48);
    } else if (panel < 5.5) {
        return vec4<f32>(vec3<f32>(0.10, 0.09, 0.08), 0.7);
    } else if (panel < 9.5) {
        return vec4<f32>(vec3<f32>(0.66, 0.18, 0.03), 0.5);
    } else if (panel < 10.5) {
        return vec4<f32>(vec3<f32>(0.18, 0.18, 0.19), 0.55);
    } else if (panel < 11.5) {
        return vec4<f32>(vec3<f32>(0.045, 0.04, 0.038), 0.8);
    } else if (panel < 16.5) {
        return vec4<f32>(1.0, 0.9, 0.55, 0.15);
    } else if (panel < 17.5) {
        return vec4<f32>(0.7, 0.02, 0.015, 0.2);
    } else if (panel < 40.5) {
        let g = noise(world.xz * 0.35);
        return vec4<f32>(vec3<f32>(0.16, 0.09, 0.045) * (0.85 + g * 0.3), 0.92);
    } else if (panel < 41.5) {
        let g = noise(world.xz * 0.2);
        return vec4<f32>(vec3<f32>(0.22, 0.24, 0.16) * (0.8 + g * 0.35), 0.9);
    } else if (panel < 42.5) {
        return vec4<f32>(vec3<f32>(0.28, 0.25, 0.20), 0.85);
    } else if (panel < 43.5) {
        return vec4<f32>(vec3<f32>(0.12, 0.24, 0.08), 0.7);
    } else if (panel < 44.5) {
        return vec4<f32>(vec3<f32>(0.16, 0.28, 0.10), 0.75);
    }
    return vec4<f32>(vec3<f32>(0.22, 0.14, 0.08), 0.8);
}

@fragment
fn fs_mesh(in: MeshOut) -> @location(0) vec4<f32> {
    let panel = i32(in.panel + 0.5);
    var paint = panel_paint(in.panel, in.uv, in.world);
    var albedo = paint.rgb;
    var rough = paint.a;
    var emit = vec3<f32>(0.0);

    if (model.extra.x > 0.5) {
        let ang = in.uv.x * 6.28318;
        let groove = smoothstep(0.25, 0.75, 0.5 + 0.5 * sin(ang * 22.0));
        let sidewall = select(0.0, 1.0, in.uv.y > 1.5);
        albedo = vec3<f32>(0.012, 0.012, 0.013);
        albedo = mix(albedo, vec3<f32>(0.03, 0.03, 0.032), groove * 0.5);
        let mud = model.extra.y;
        let packed = clamp(mud * (0.25 + 0.85 * groove) * (1.0 - sidewall * 0.65), 0.0, 1.0);
        let clump = smoothstep(0.55, 0.9, hash12(floor(vec2<f32>(ang * 8.0, in.uv.y * 6.0))));
        let mask = clamp(packed * (0.45 + 0.7 * clump), 0.0, 1.0);
        albedo = mix(albedo, vec3<f32>(0.09, 0.045, 0.025), mask);
        rough = mix(0.85, 0.25, mask);
    } else if (panel >= 0 && panel < 12) {
        let m = mud_of(panel);
        let cover = clamp(m.x, 0.0, 1.4);
        let drip = m.y;
        let streak = m.z;
        let wet = m.w;
        let bands = 0.5 + 0.5 * sin(in.uv.x * 17.0 + in.uv.y * 11.0 + streak * 5.0);
        let run = smoothstep(0.62, 0.92, bands) * drip;
        let down = smoothstep(0.1, 0.9, 0.55 - in.uv.y);
        let cell = floor(in.uv * vec2<f32>(7.0, 5.0) + vec2<f32>(streak * 2.0, cover * 3.0));
        let clump = smoothstep(0.42, 0.88, hash12(cell));
        var mask = cover * (0.35 + 0.65 * clump);
        mask = mask + run * cover * down * 0.45;
        mask = clamp(mask, 0.0, 1.0);
        let mud_col = mix(vec3<f32>(0.16, 0.09, 0.045), vec3<f32>(0.07, 0.035, 0.02), clamp(wet, 0.0, 1.0));
        albedo = mix(albedo, mud_col, smoothstep(0.04, 0.4, mask));
        let thin = smoothstep(0.78, 0.96, bands) * drip * smoothstep(0.08, 0.45, cover);
        albedo = mix(albedo, mud_col * 0.75, thin * 0.55);
        rough = mix(rough, mix(0.55, 0.12, wet), clamp(mask + thin, 0.0, 1.0));
    }

    if (in.panel > 15.5 && in.panel < 16.5) {
        emit = vec3<f32>(1.3, 1.05, 0.55);
        albedo = vec3<f32>(1.0, 0.95, 0.7);
    } else if (in.panel > 16.5 && in.panel < 17.5) {
        emit = vec3<f32>(0.9, 0.02, 0.01);
    }

    return present(shade(in.world, in.nrm, albedo, rough, emit));
}

struct PartIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
}

struct PartOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
}

@vertex
fn vs_particle(in: PartIn) -> PartOut {
    var o: PartOut;
    o.clip = frame.view_proj * vec4<f32>(in.pos, 1.0);
    o.color = in.color;
    o.uv = in.uv;
    return o;
}

@fragment
fn fs_particle(in: PartOut) -> @location(0) vec4<f32> {
    let d = length(in.uv - vec2<f32>(0.5));
    if (d > 0.5) {
        discard;
    }
    let fall = smoothstep(0.5, 0.12, d);
    let rgb = pow(in.color.rgb, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(rgb, in.color.a * fall);
}
