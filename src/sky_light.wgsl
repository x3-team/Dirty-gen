// Overcast wet-oval light. Matched to the Kling stills: a flat gray sky,
// fog sitting on the ground, and glossy mud that mirrors that sky.
// lighting.rs splices these functions over the sunny grade in shader.wgsl.

fn overcast_grade(c: vec3<f32>) -> vec3<f32> {
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Cool and flat, but the mud stays dark. A hard lift turned the lane white.
    var flat = mix(c, vec3<f32>(luma), 0.34);
    flat = flat * vec3<f32>(0.93, 0.97, 1.03);
    let lifted = flat * 0.94 + vec3<f32>(0.008);
    let mapped = vec3<f32>(1.0) - exp(-max(lifted, vec3<f32>(0.0)) * 1.05);
    return pow(mapped, vec3<f32>(1.0 / 2.2));
}

fn ground_fog(col: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let dist = length(world - frame.cam_pos.xyz);
    let above = max(world.y, 0.0);
    // Thick against the soil, thinner on a berm, gone above the hills.
    let height = exp(-above * 0.42);
    // The near oval stays dark. Fog builds across the far bank into the horizon.
    let dist_f = smoothstep(20.0, 40.0, dist);
    let fog = clamp(height * (0.04 + dist_f * 0.86), 0.0, 0.86);
    return mix(col, frame.sky_horizon.rgb, fog);
}

fn present(c: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(overcast_grade(c), 1.0);
}

fn sky_color(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let h = clamp(dir.y, 0.0, 1.0);
    // Flat overcast. No sun disc and no cumulus.
    var col = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, smoothstep(0.0, 0.38, h));
    let band = smoothstep(0.0, 0.06, dir.y) * (1.0 - smoothstep(0.06, 0.22, dir.y));
    col = col + frame.sky_horizon.rgb * band * 0.04;
    return col;
}

fn shade(world: vec3<f32>, n_in: vec3<f32>, albedo: vec3<f32>, rough: f32, emit: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    // Soft wrap. An overcast day has no hard terminator.
    let wrap = ndl * 0.34 + 0.66;
    let shadow = shadow_factor(world + n * 0.012, n);
    // A hint of contact, not a black cut.
    let occ = mix(0.64, 1.0, shadow);
    let hemi = mix(frame.amb.rgb, frame.sky_zenith.rgb, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let view = normalize(frame.cam_pos.xyz - world);
    let gloss = smoothstep(0.15, 0.70, 1.0 - rough);
    let refl = reflect(-view, n);
    let sky_refl = sky_color(refl);
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let fres = pow(1.0 - ndotv, 2.0);
    let halfv = normalize(sun + view);
    let broad = pow(clamp(dot(n, halfv), 0.0, 1.0), 3.5);
    var col = albedo * hemi * (0.58 + 0.42 * wrap) * occ;
    col = col + frame.sun_color.rgb * albedo * wrap * occ * 0.28;
    // Smooth wet surfaces mirror the gray sky. Dry soil stays matte.
    col = mix(col, sky_refl, gloss * (0.20 + fres * 0.62) * occ);
    col = col + sky_refl * broad * gloss * 0.22 * occ;
    col = col + emit;
    return col;
}
