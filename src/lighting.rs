//! Photographic overcast for the wet oval.
//!
//! The Kling stills are a soft gray sky, mist sitting on the ground, and
//! bright streaks where water fills a rut or a puddle. The soil stays dark.
//! This module owns those colors and splices `sky_light.wgsl` over the sunny
//! grade. The renderer only asks for the shader and the rig.

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
        sun_color: [0.26, 0.28, 0.31, 1.0],
        // Soft overcast: a light zenith, and a brighter horizon the mist takes.
        sky_zenith: [0.58, 0.62, 0.68, 1.0],
        sky_horizon: [0.88, 0.90, 0.93, 1.0],
        // Under the sky, so the soil can stay darker than the air.
        ambient: [0.12, 0.13, 0.15, 1.0],
    }
}

pub fn shader_source() -> String {
    let mut src = include_str!("shader.wgsl").to_string();
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
    // Water in the tyre cut mirrors the bright sky as one ribbon.
    // A per-cell lobe turned that ribbon into speckles.
    let in_rut = smoothstep(0.012, 0.030, cut) * smoothstep(0.35, 0.80, wet) * (1.0 - drowned);
    let axis = streak_axis(xz);
    let along = vec2<f32>(-axis.y, axis.x);
    // Bright along the cut, with grain so the highlight is not a flat white decal.
    let grain = noise(vec2<f32>(dot(xz, along) * 1.3, dot(xz, axis) * 0.35));
    let spark = smoothstep(0.80, 0.96, noise(xz * 4.5));
    let shine = frame.sky_horizon.rgb * (0.58 + 0.28 * grain + spark * 0.35);
    let shore_glint = shore * (1.0 - drowned);
    lit = mix(lit, shine, clamp(in_rut * 0.86 + shore_glint * 0.25, 0.0, 1.0));
"#,
    );
    src = patch_all(
        &src,
        "return present(lit);",
        "return present(ground_fog(lit, in.world));",
        2,
    );
    src = patch(
        &src,
        "    // Quiet reflection. sky_color carries the sun disc and white clouds.\n    \
         let quiet = vec3<f32>(0.10, 0.042, 0.015);\n    \
         let reflect_amt = fres * (0.22 + (1.0 - depth_t) * 0.16);\n    \
         var col = mix(absorption, quiet, reflect_amt);\n    \
         let spec = pow(clamp(dot(nrm, normalize(sun + view)), 0.0, 1.0), 28.0);\n    \
         col += vec3<f32>(0.04, 0.016, 0.005) * spec * shadow_factor(in.world, nrm);\n",
        r#"    // Pale mirror of the overcast, with a few broad streaks.
    // The deep middle stays darker so the pond is not one gray disc.
    let skin = 1.0 - depth_t * 0.62;
    let body = reflected_sky(reflect(-view, nrm));
    let reflect_amt = mix(0.58, 0.86, fres) * mix(0.46, 1.0, skin);
    var col = mix(absorption, body, reflect_amt);
    let ribbon = water_ribbon(in.world.xz);
    let wrap = clamp(dot(nrm, sun) * 0.15 + 0.85, 0.0, 1.0);
    col = col * wrap;
    col = mix(col, silver(), ribbon * skin * 0.82);
"#,
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
        assert!(src.contains("smoothstep(18.0, 40.0, dist)"));
        assert!(src.contains("let in_rut = smoothstep(0.012, 0.030, cut)"));
        assert!(src.contains("mix(c, vec3<f32>(luma), 0.22)"));
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
