// Chevron darkening inside an existing rut, and the puddle mirror.
// Neither function writes height. The soil solver is unchanged.

fn rut_chevron_scale(xz: vec2<f32>, world: vec3<f32>, groove: f32) -> f32 {
    if (groove < 0.02) {
        return 1.0;
    }
    let view = normalize(frame.cam_pos.xyz - world);
    let above = smoothstep(0.22, 0.68, view.y);
    let e = 0.22;
    let sx = stamp_bilinear(xz + vec2<f32>(e, 0.0)) - stamp_bilinear(xz - vec2<f32>(e, 0.0));
    let sz = stamp_bilinear(xz + vec2<f32>(0.0, e)) - stamp_bilinear(xz - vec2<f32>(0.0, e));
    var across = vec2<f32>(-sz, sx);
    let alen = length(across);
    if (alen < 1e-4) {
        across = vec2<f32>(1.0, 0.0);
    } else {
        across = across / alen;
    }
    let along = vec2<f32>(-across.y, across.x);
    let s = dot(xz, along);
    let a = dot(xz, across);
    let v = abs(fract(s * 4.6 + abs(a) * 3.4) - 0.5);
    let notch = smoothstep(0.06, 0.34, v);
    return mix(1.0, mix(0.62, 0.46, notch), groove * above);
}

fn puddle_color(world: vec3<f32>, nrm: vec3<f32>, depth_t: f32, absorption: vec3<f32>) -> vec3<f32> {
    let view = normalize(frame.cam_pos.xyz - world);
    let ndotv = clamp(dot(nrm, view), 0.0, 1.0);
    let fres = pow(1.0 - ndotv, 2.0);
    let above = smoothstep(0.28, 0.75, view.y);
    let refl = reflect(-view, nrm);
    var sky = sky_color(refl);
    // Warm the sky sample. An uncolored reflection is a gray metal disc.
    sky = sky * vec3<f32>(0.40, 0.22, 0.10) + vec3<f32>(0.055, 0.022, 0.008);
    let mirror = mix(0.58, 0.26, above) * (0.42 + 0.58 * fres);
    let amt = clamp(mirror * (0.5 + (1.0 - depth_t) * 0.3), 0.0, 0.7);
    return mix(absorption, sky, amt);
}
