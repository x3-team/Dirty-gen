//! Shared overcast for the wet oval and the glTF buggy.
//!
//! Soft sky, a thin mist on the far ground, and the sky mirrored in a rut.
//! The soil stays darker than the air. Paint keeps its hue: a mix toward luma
//! turned the orange cage tan. This module splices `sky_light.wgsl` over the
//! sunny grade. The buggy uses the same `shade`, sky, and ground fog.

use glam::Vec3;

pub struct SkyLight {
    pub sun: Vec3,
    pub sun_color: [f32; 4],
    pub sky_zenith: [f32; 4],
    pub sky_horizon: [f32; 4],
    pub ambient: [f32; 4],
}

pub fn overcast() -> SkyLight {
    SkyLight {
        // High and diffuse. Streaks are the sky in the water, not a hard sun.
        sun: Vec3::new(0.22, 0.95, 0.18).normalize(),
        // Warm enough that orange paint survives. The sky itself stays gray.
        sun_color: [0.62, 0.50, 0.36, 1.0],
        // Soft overcast: a light zenith, and a brighter horizon the mist takes.
        sky_zenith: [0.62, 0.66, 0.74, 1.0],
        sky_horizon: [0.88, 0.90, 0.93, 1.0],
        // Under the sky, so the soil can stay darker than the air.
        ambient: [0.20, 0.19, 0.17, 1.0],
    }
}

pub fn shader_source() -> String {
    apply(include_str!("shader.wgsl").to_string())
}

/// Splice the overcast grade onto a shader that may already include the mud-look hooks.
pub fn apply(mut src: String) -> String {
    src = strip_fn(&src, "present");
    src = strip_fn(&src, "sky_color");
    src = strip_fn(&src, "shade");
    src.push('\n');
    src.push_str(include_str!("sky_light.wgsl"));
    src = patch(
        &src,
        r#"    // The lane stays matte. Shine is the soil ringing a puddle. The same lobe
    // on the tire cut turned the groove into a tan stripe, so the floor stays dark.
    let shore = smoothstep(0.0, 0.02, water) * (1.0 - smoothstep(0.035, 0.11, water));
    // Standing water is drawn as its own sheet. A mirror bed under that film
    // is the gray metal disc.
    let drowned = smoothstep(0.012, 0.05, water);
    let rough_out = mix(mix(0.94, 0.2, shore * (1.0 - drowned)), 0.90, drowned);
    var lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
    // The sun sits low, so a tight Phong highlight misses this camera.
    // A wide lobe on the shore is the wet sheen.
    let view = normalize(frame.cam_pos.xyz - in.world);
    let ndotv = clamp(dot(n, view), 0.0, 1.0);
    let halfv = normalize(normalize(frame.sun_dir.xyz) + view);
    let facing = pow(1.0 - ndotv, 2.0);
    let wide = pow(clamp(dot(n, halfv), 0.0, 1.0), 2.2);
    let wet_spec = (facing * 0.75 + wide * 1.15) * shore * mix(1.0, 0.16, drowned);
    lit += vec3<f32>(0.22, 0.095, 0.032) * wet_spec;
"#,
        r#"    // The lane stays matte. Shine is water in the tyre cut and on a puddle lip.
    let shore = smoothstep(0.0, 0.02, water) * (1.0 - smoothstep(0.035, 0.11, water));
    // Standing water is its own sheet. A second mirror under that film is a gray disc.
    let drowned = smoothstep(0.012, 0.05, water);
    let rough_out = mix(0.96, 0.90, drowned);
    var lit = shade(in.world, n, albedo, rough_out, vec3<f32>(0.0));
    // The tyre cut mirrors the sky. A wall of the groove faces the bank, and a
    // raw mirror of that wall is mud, so the wet floor takes the sky.
    // Most of the tyre cut sits around wetness 0.2. A gate at 0.6 left that groove black.
    let in_rut = smoothstep(0.003, 0.008, cut) * smoothstep(0.10, 0.18, wet);
    let view = normalize(frame.cam_pos.xyz - in.world);
    let n_up = normalize(mix(n, vec3<f32>(0.0, 1.0, 0.0), 0.78));
    let sky_hit = reflected_sky(reflect(-view, n_up));
    let ribbon = water_ribbon(xz);
    lit = mix(lit, sky_hit, clamp(in_rut * (0.74 + 0.16 * ribbon), 0.0, 0.90));
    let shore_glint = shore * (1.0 - drowned) * 0.35;
    lit = mix(lit, sky_hit, shore_glint);
"#,
    );
    src = patch_all(
        &src,
        "return present(lit);",
        "return present(ground_fog(lit, in.world));",
        2,
    );
    // The puddle sheet. A tyre cut keeps a sky film; opening that film showed
    // the black groove underneath instead of the reflection.
    src = patch(
        &src,
        "    var col = puddle_color(in.world, nrm, depth_t, absorption);\n    \
         let spec = pow(clamp(dot(nrm, normalize(sun + view)), 0.0, 1.0), 28.0);\n    \
         col += vec3<f32>(0.04, 0.016, 0.005) * spec * shadow_factor(in.world, nrm);\n    \
         let shore = smoothstep(0.002, 0.07, in.depth);\n    \
         // Only a cut of a couple of centimetres thins the film, and only on that cut.\n    \
         // The rest of the pond stays a dark cover, shore and middle alike.\n    \
         let dug = max(-stamp_bilinear(in.world.xz), 0.0);\n    \
         let reveal = smoothstep(0.016, 0.034, dug);\n    \
         let open = mix(0.24, 0.42, smoothstep(0.015, 0.10, in.depth));\n    \
         let alpha = shore * mix(0.88, 0.98, depth_t) * mix(1.0, open, reveal);\n",
        "    let ndotv = clamp(dot(nrm, view), 0.0, 1.0);\n    \
         let fres = pow(1.0 - ndotv, 2.0);\n    \
         let skin = 1.0 - depth_t * 0.62;\n    \
         let body = reflected_sky(reflect(-view, nrm));\n    \
         let dug = max(-stamp_bilinear(in.world.xz), 0.0);\n    \
         let in_rut = smoothstep(0.008, 0.022, dug);\n    \
         let reflect_amt = mix(mix(0.48, 0.78, fres) * mix(0.45, 0.88, skin), 0.97, in_rut);\n    \
         var col = mix(absorption, body, reflect_amt);\n    \
         let ribbon = water_ribbon(in.world.xz);\n    \
         let wrap = clamp(dot(nrm, sun) * 0.15 + 0.85, 0.0, 1.0);\n    \
         col = col * wrap;\n    \
         col = mix(col, silver(), max(ribbon * skin * 0.62, in_rut * 0.90));\n    \
         let shore = smoothstep(0.002, 0.07, in.depth);\n    \
         let reveal = smoothstep(0.016, 0.034, dug);\n    \
         let open = mix(0.24, 0.42, smoothstep(0.015, 0.10, in.depth));\n    \
         let alpha = shore * mix(0.88, 0.98, depth_t) * mix(1.0, open, reveal * (1.0 - in_rut));\n",
    );
    src = patch(
        &src,
        "    let mapped = vec3<f32>(1.0) - exp(-max(col, vec3<f32>(0.0)) * 1.08);\n    \
         let gamma = pow(mapped, vec3<f32>(1.0 / 2.2));\n    \
         return vec4<f32>(gamma, alpha);\n",
        "    col = ground_fog(col, in.world);\n    \
         return vec4<f32>(overcast_grade(col), alpha);\n",
    );
    src
}

fn patch(src: &str, old: &str, new: &str) -> String {
    patch_all(src, old, new, 1)
}

fn patch_all(src: &str, old: &str, new: &str, expect: usize) -> String {
    let count = src.matches(old).count();
    assert_eq!(
        count, expect,
        "lighting hook: expected {expect} match(es), found {count}"
    );
    src.replace(old, new)
}

fn strip_fn(src: &str, name: &str) -> String {
    let marker = format!("fn {name}(");
    let start = src
        .find(&marker)
        .unwrap_or_else(|| panic!("lighting hook: shader is missing `{marker}`"));
    let bytes = src.as_bytes();
    let mut i = start;
    let mut depth = 0i32;
    let mut started = false;
    while i < bytes.len() {
        if bytes[i] == b'/' && i + 1 < bytes.len() && bytes[i + 1] == b'/' {
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        match bytes[i] {
            b'{' => {
                depth += 1;
                started = true;
            }
            b'}' => {
                depth -= 1;
                if started && depth == 0 {
                    let end = i + 1;
                    let mut out = String::with_capacity(src.len());
                    out.push_str(&src[..start]);
                    out.push_str(&src[end..]);
                    return out;
                }
            }
            _ => {}
        }
        i += 1;
    }
    panic!("lighting hook: unclosed `{marker}`");
}

#[cfg(test)]
mod tests {
    fn luma(c: &[f32; 4]) -> f32 {
        c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
    }

    #[test]
    fn splices_photographic_light_once() {
        let src = super::shader_source();
        assert_eq!(src.matches("fn sky_color(").count(), 1);
        assert_eq!(src.matches("fn shade(").count(), 1);
        assert_eq!(src.matches("fn present(").count(), 1);
        assert_eq!(src.matches("fn ground_fog(").count(), 1);
        assert_eq!(src.matches("fn water_ribbon(").count(), 1);
        assert_eq!(src.matches("fn streak_axis(").count(), 1);
        assert!(src.contains("ground_fog(lit, in.world)"));
        assert!(src.contains("exp(-above * 2.2)"));
        assert!(src.contains("smoothstep(26.0, 52.0, dist)"));
        assert!(src.contains("let in_rut = smoothstep(0.003, 0.008, cut)"));
        assert!(src.contains("let in_rut = smoothstep(0.008, 0.022, dug)"));
        assert!(src.contains("reflected_sky(reflect(-view, n_up))"));
        assert!(src.contains("in_rut * 0.90"));
        assert!(src.contains("mix(vec3<f32>(luma), col, 1.45)"));
        assert!(!src.contains("mix(c, vec3<f32>(luma), 0.22)"));
        assert!(!src.contains("pow(d, 900.0)"));
        assert!(!src.contains("0.22, 0.095, 0.032"));
        assert!(!src.contains("vec3<f32>(0.10, 0.042, 0.015)"));
        assert!(!src.contains("smoothstep(0.72, 1.0, wet)"));
        assert!(!src.contains("flat * 0.94 + vec3<f32>(0.008)"));
    }

    #[test]
    fn sky_is_a_soft_gradient_and_the_ground_stays_darker() {
        let sky = super::overcast();
        let zen = luma(&sky.sky_zenith);
        let hor = luma(&sky.sky_horizon);
        let amb = luma(&sky.ambient);
        assert!(
            hor > zen + 0.18,
            "horizon {hor:.3} should open above zenith {zen:.3}"
        );
        assert!(
            (0.45..0.78).contains(&zen),
            "zenith {zen:.3} should stay a soft light gray"
        );
        assert!(hor < 0.97, "horizon {hor:.3} should not blow out");
        assert!(sky.sun.y > 0.85, "sun should stay high and soft");
        assert!(
            amb + 0.15 < zen,
            "ambient {amb:.3} should sit under the sky {zen:.3}"
        );
    }
}
