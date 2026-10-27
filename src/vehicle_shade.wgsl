// Wet paint, tire mud, and spray-sheet light. Included immediately before fs_mesh.
// Panel 19 is the orange chassis and coil. Panel 26 is the damper rod.
// Panel 48 is a spray sheet. a < 0 means this helper does not own the panel.

struct VehicleHit {
    albedo: vec3<f32>,
    rough: f32,
    nrm: vec3<f32>,
    used: f32,
}

fn mud_above(world: vec3<f32>) -> f32 {
    let view = normalize(frame.cam_pos.xyz - world);
    return smoothstep(0.18, 0.72, view.y);
}

fn chocolate(wet: f32, above: f32, under: f32) -> vec3<f32> {
    let side = vec3<f32>(0.11, 0.045, 0.016);
    let top = vec3<f32>(0.072, 0.030, 0.011);
    let belly = vec3<f32>(0.030, 0.011, 0.004);
    var c = mix(side, top, above);
    c = mix(c, belly, under);
    return c * mix(1.0, 0.78, wet * 0.4);
}

fn vehicle_panel_paint(panel: f32, world: vec3<f32>) -> vec4<f32> {
    let grit = hash12(floor(world.xz * 5.0 + vec2<f32>(world.y * 2.0, panel)));
    let silver = vec3<f32>(0.78, 0.80, 0.82) * (0.90 + grit * 0.10);
    let orange = vec3<f32>(0.95, 0.34, 0.045) * (0.94 + grit * 0.08);
    if (panel < 0.5) {
        return vec4<f32>(silver, 0.12);
    } else if (panel < 4.5) {
        return vec4<f32>(orange, 0.16);
    } else if (panel < 5.5) {
        return vec4<f32>(vec3<f32>(0.10, 0.09, 0.08), 0.40);
    } else if (panel < 9.5) {
        return vec4<f32>(orange * 0.92, 0.18);
    } else if (panel < 10.5) {
        return vec4<f32>(vec3<f32>(0.22, 0.22, 0.24), 0.22);
    } else if (panel < 11.5) {
        return vec4<f32>(vec3<f32>(0.03, 0.028, 0.026), 0.72);
    } else if (panel > 17.5 && panel < 18.5) {
        return vec4<f32>(vec3<f32>(0.015, 0.015, 0.016), 0.34);
    } else if (panel > 18.5 && panel < 19.5) {
        return vec4<f32>(orange, 0.11);
    } else if (panel > 20.5 && panel < 21.5) {
        return vec4<f32>(vec3<f32>(0.08, 0.16, 0.48), 0.55);
    } else if (panel > 21.5 && panel < 22.5) {
        return vec4<f32>(vec3<f32>(0.90, 0.91, 0.92), 0.18);
    } else if (panel > 22.5 && panel < 23.5) {
        return vec4<f32>(vec3<f32>(0.02, 0.04, 0.06), 0.08);
    } else if (panel > 23.5 && panel < 24.5) {
        return vec4<f32>(vec3<f32>(0.02, 0.02, 0.02), 0.60);
    } else if (panel > 24.5 && panel < 25.5) {
        return vec4<f32>(vec3<f32>(0.85, 0.82, 0.70), 0.40);
    } else if (panel > 25.5 && panel < 26.5) {
        return vec4<f32>(silver, 0.06);
    }
    return vec4<f32>(0.0, 0.0, 0.0, -1.0);
}

fn vehicle_wheel(panel: f32, uv: vec2<f32>, world: vec3<f32>, n: vec3<f32>) -> VehicleHit {
    let mud = clamp(model.extra.y, 0.0, 1.2);
    let above = mud_above(world);
    if (panel > 17.5 && panel < 18.5) {
        let speck = smoothstep(0.55, 0.86, noise(world.xy * 12.0 + vec2<f32>(world.z, 1.0)));
        let mask = smoothstep(0.04, 0.35, mud) * speck;
        let albedo = mix(vec3<f32>(0.018, 0.018, 0.02), chocolate(0.4, above, 0.0), mask);
        let rough = mix(0.38, mix(0.16, 0.42, above), mask);
        return VehicleHit(albedo, rough, n, 1.0);
    }
    let ang = uv.x * 6.2831853;
    let sidewall = select(0.0, 1.0, uv.y > 1.5);
    let block = 0.5 + 0.5 * sin(ang * 8.0);
    let chev = 0.5 + 0.5 * sin(ang * 8.0 + abs(uv.y) * 6.0);
    let groove = smoothstep(0.25, 0.8, block * 0.65 + chev * 0.35);
    let nse = noise(world.xy * 9.0 + vec2<f32>(world.z * 4.0, ang));
    let n2 = noise(world.yz * 16.0 + vec2<f32>(ang, 2.0));
    let clump = smoothstep(0.46, 0.8, nse * 0.65 + n2 * 0.35);
    let load = smoothstep(0.03, 0.24, mud);
    var mask = load * mix(0.82 + 0.18 * clump, 0.62 + 0.30 * clump, sidewall);
    mask = mask * mix(1.0, mix(0.55, 1.0, groove), above * (1.0 - sidewall));
    mask = clamp(mask, 0.0, 1.0);
    let rubber = vec3<f32>(0.012, 0.011, 0.010);
    let albedo = mix(rubber, chocolate(0.55, above, 0.0), mask);
    var rough = mix(0.92, mix(0.14, 0.42, above), mask);
    rough = mix(rough, 0.55, sidewall * mask * above);
    let nrm = mud_normal(n, world, mask * mix(0.9, 0.45, above));
    return VehicleHit(albedo, rough, nrm, 1.0);
}

fn coat_body_mud(albedo: vec3<f32>, rough: f32, nrm: vec3<f32>, world: vec3<f32>, mask: f32, wet: f32) -> VehicleHit {
    let above = mud_above(world);
    let under = smoothstep(0.12, -0.45, nrm.y);
    let m = clamp(max(mask, under * 0.72), 0.0, 1.0);
    let alb = mix(albedo, chocolate(wet, above, under), m);
    var r = mix(rough, mix(0.08, 0.48, above), m);
    r = mix(r, 0.62, under * m);
    let n = mud_normal(nrm, world, m * mix(0.7, 1.15, under));
    return VehicleHit(alb, r, n, 1.0);
}

fn clod_lighting(world: vec3<f32>, n_in: vec3<f32>) -> vec3<f32> {
    var n = normalize(n_in);
    let e = 0.06;
    let h0 = noise(world.xz * 3.4 + vec2<f32>(world.y * 2.4, 0.2));
    let hx = noise((world.xz + vec2<f32>(e, 0.0)) * 3.4 + vec2<f32>(world.y * 2.4, 0.2));
    let hz = noise((world.xz + vec2<f32>(0.0, e)) * 3.4 + vec2<f32>(world.y * 2.4, 0.2));
    n = normalize(n + vec3<f32>(h0 - hx, 0.04, h0 - hz) * 0.35);
    let sun = normalize(frame.sun_dir.xyz);
    let view = normalize(frame.cam_pos.xyz - world);
    let above = smoothstep(0.18, 0.72, view.y);
    let under = smoothstep(0.15, -0.45, n.y);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let shadow = shadow_factor(world + n * 0.02, n);
    let tone = noise(world.xz * 2.2 + vec2<f32>(world.y, 0.3));
    var body = mix(vec3<f32>(0.09, 0.036, 0.013), vec3<f32>(0.055, 0.022, 0.008), tone);
    body = mix(body, body * 0.62, above * 0.45);
    body = mix(body, vec3<f32>(0.028, 0.010, 0.004), under);
    let wrap = ndl * 0.45 + 0.55;
    var lit = body * (0.7 + wrap * shadow * 0.55);
    let streak = noise(world.xz * 6.5 + vec2<f32>(world.y * 4.0, 0.4));
    let wetspot = smoothstep(0.72, 0.92, streak);
    let gloss = vec3<f32>(0.32, 0.12, 0.04);
    let side = 1.0 - above;
    lit += gloss * pow(ndotv, mix(8.0, 24.0, side)) * (0.15 + side * 0.45) * shadow;
    lit += gloss * wetspot * side * 0.35 * shadow;
    lit += gloss * pow(1.0 - ndotv, 3.0) * mix(0.18, 0.05, above);
    return lit;
}

fn sheet_lighting(world: vec3<f32>, n_in: vec3<f32>) -> vec3<f32> {
    let n = normalize(n_in);
    let view = normalize(frame.cam_pos.xyz - world);
    let sun = normalize(frame.sun_dir.xyz);
    let above = smoothstep(0.2, 0.75, view.y);
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let ndl = clamp(dot(n, sun), 0.0, 1.0);
    let shadow = shadow_factor(world + n * 0.02, n);
    let body = mix(vec3<f32>(0.12, 0.048, 0.016), vec3<f32>(0.07, 0.028, 0.010), above);
    var lit = body * (0.55 + ndl * shadow * 0.9);
    let fres = pow(1.0 - ndotv, 2.2);
    let gloss = vec3<f32>(0.40, 0.16, 0.05);
    lit += gloss * fres * mix(0.65, 0.22, above) * (0.4 + shadow * 0.6);
    lit += gloss * pow(clamp(dot(n, normalize(sun + view)), 0.0, 1.0), mix(24.0, 8.0, above)) * 0.35 * shadow;
    return lit;
}

fn vehicle_sheen(world: vec3<f32>, n: vec3<f32>, rough: f32) -> vec3<f32> {
    let gloss = smoothstep(0.28, 0.05, rough);
    let view = normalize(frame.cam_pos.xyz - world);
    let nn = normalize(n);
    let fres = pow(1.0 - clamp(dot(nn, view), 0.0, 1.0), 2.5);
    let sky = sky_color(reflect(-view, nn));
    return sky * fres * gloss * 0.18;
}
