// One overcast light for the wet oval and the glTF buggy.
// Soft sky, a thin mist on the far ground, and the sky mirrored in a rut.
// Paint keeps its hue. A mix toward luma turned the orange cage tan.
// lighting.rs splices these functions over the sunny grade in shader.wgsl.

fn overcast_grade(c: vec3<f32>) -> vec3<f32> {
    let exposed = max(c, vec3<f32>(0.0));
    // Per channel. Tonemapping the luma alone lifts green and the cage goes tan.
    var col = exposed / (vec3<f32>(1.0) + exposed);
    col = pow(col, vec3<f32>(1.0 / 2.2));
    let luma = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    col = mix(vec3<f32>(luma), col, 1.45);
    return clamp(col, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn ground_fog(col: vec3<f32>, world: vec3<f32>) -> vec3<f32> {
    let dist = length(world - frame.cam_pos.xyz);
    let above = max(world.y, 0.0);
    // Hugs the soil. A berm is already out of it, so the ring stays readable.
    let height = exp(-above * 2.2);
    // The near oval stays clear. Only the far reeds take a thin mist.
    let dist_f = smoothstep(26.0, 52.0, dist);
    let fog = clamp(height * dist_f * 0.50, 0.0, 0.34);
    return mix(col, frame.sky_horizon.rgb, fog);
}

fn present(c: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(overcast_grade(c), 1.0);
}

fn sky_color(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let h = clamp(dir.y, 0.0, 1.0);
    // Soft overcast. No sun disc. The horizon opens into the mist.
    let t = smoothstep(0.0, 0.62, h);
    let curve = t * t * (3.0 - 2.0 * t);
    var col = mix(frame.sky_horizon.rgb, frame.sky_zenith.rgb, curve);
    let band = smoothstep(0.0, 0.03, dir.y) * (1.0 - smoothstep(0.05, 0.34, dir.y));
    col = col + frame.sky_horizon.rgb * band * 0.06;
    return col;
}

// Below the horizon a mirror shows mud, not a second copy of the bright sky.
fn reflected_sky(dir_in: vec3<f32>) -> vec3<f32> {
    let dir = normalize(dir_in);
    let sky = sky_color(dir);
    let ground = smoothstep(0.04, -0.16, dir.y);
    return mix(sky, sky * vec3<f32>(0.16, 0.14, 0.12), ground);
}

// Across a tyre cut. A flat pad falls back to the long axis of the oval.
fn streak_axis(xz: vec2<f32>) -> vec2<f32> {
    let e = max(frame.params.z * 1.35, 0.22);
    let gx = stamp_bilinear(xz + vec2<f32>(e, 0.0)) - stamp_bilinear(xz - vec2<f32>(e, 0.0));
    let gz = stamp_bilinear(xz + vec2<f32>(0.0, e)) - stamp_bilinear(xz - vec2<f32>(0.0, e));
    let glen = length(vec2<f32>(gx, gz));
    let across = select(
        vec2<f32>(0.42, 0.91),
        vec2<f32>(gx, gz) / max(glen, 1e-4),
        glen > 0.012,
    );
    return normalize(across);
}

// A few broad ribbons across the water. A tight sine drew hairline contours.
fn water_ribbon(xz: vec2<f32>) -> f32 {
    let across = streak_axis(xz);
    let along = vec2<f32>(-across.y, across.x);
    let u = dot(xz, across);
    let v = dot(xz, along);
    let wander = noise(vec2<f32>(v * 0.18, 1.7));
    let band = abs(fract(u * 0.38 + wander * 0.45) - 0.5);
    let wide = 1.0 - smoothstep(0.05, 0.20, band);
    let dash = smoothstep(0.22, 0.62, noise(vec2<f32>(v * 0.42, u * 0.05)));
    return wide * (0.35 + 0.65 * dash);
}

fn silver() -> vec3<f32> {
    return frame.sky_horizon.rgb * vec3<f32>(1.06, 1.03, 1.0);
}

fn shade(world: vec3<f32>, n_in: vec3<f32>, albedo: vec3<f32>, rough: f32, emit: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);
    let sun = normalize(frame.sun_dir.xyz);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    // Soft wrap. An overcast day has no hard terminator.
    let wrap = ndl * 0.28 + 0.72;
    let shadow = shadow_factor(world + n * 0.012, n);
    let occ = mix(0.58, 1.0, shadow);
    // Zenith lights the sky and the upward faces. Soil albedo is dark, so it stays down.
    let hemi = mix(frame.amb.rgb, frame.sky_zenith.rgb * 0.85, clamp(n.y * 0.5 + 0.5, 0.0, 1.0));
    let view = normalize(frame.cam_pos.xyz - world);
    let gloss = smoothstep(0.15, 0.70, 1.0 - rough);
    let sky_refl = reflected_sky(reflect(-view, n));
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let fres = pow(1.0 - ndotv, 2.0);
    let halfv = normalize(sun + view);
    let broad = pow(clamp(dot(n, halfv), 0.0, 1.0), 4.0);
    var col = albedo * hemi * (0.85 + 0.28 * wrap) * occ;
    col = col + frame.sun_color.rgb * albedo * wrap * occ * 0.65;
    // Add the sky on the highlight. Replacing the albedo here turned orange paint gray.
    let spec = gloss * (0.04 + fres * 0.10) * occ;
    col = col + sky_refl * spec;
    col = col + sky_refl * broad * gloss * 0.08 * occ;
    col = col + emit;
    return col;
}
