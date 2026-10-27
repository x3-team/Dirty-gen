//! Overcast sky and ground fog for the wet oval.
//!
//! The Kling stills (flat gray sky, fog on the ground, glossy mud) drive the
//! rig. This module owns those colors and splices `sky_light.wgsl` over the
//! sunny grade. The renderer only asks for the shader and the rig.

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
        // High and soft. A low sun rakes long shadows through the fog.
        sun: Vec3::new(0.30, 0.90, 0.28).normalize(),
        sun_color: [0.16, 0.17, 0.19, 1.0],
        sky_zenith: [0.52, 0.56, 0.60, 1.0],
        sky_horizon: [0.84, 0.86, 0.87, 1.0],
        ambient: [0.20, 0.21, 0.23, 1.0],
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
        "    lit += vec3<f32>(0.22, 0.095, 0.032) * wet_spec;\n",
        "    let wet_refl = sky_color(reflect(-view, n));\n    \
         lit = mix(lit, wet_refl, clamp(wet_spec, 0.0, 0.20));\n",
    );
    src = patch(
        &src,
        "    let wet_spec = (facing * 0.75 + wide * 1.15) * shore * mix(1.0, 0.16, drowned);\n",
        "    let damp = smoothstep(0.72, 1.0, wet);\n    \
         let wet_spec = (facing * 0.40 + wide * 0.18) * max(shore * 0.40, damp * 0.30);\n",
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
        "    // The sheet mirrors the overcast sky. That is the glossy mud.\n    \
         let refl = reflect(-view, nrm);\n    \
         let sky_refl = sky_color(refl);\n    \
         let reflect_amt = mix(0.36, 0.74, fres);\n    \
         var col = mix(absorption, sky_refl, reflect_amt);\n    \
         let spec = pow(clamp(dot(nrm, normalize(sun + view)), 0.0, 1.0), 8.0);\n    \
         col += sky_refl * spec * 0.22;\n",
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
    #[test]
    fn splices_overcast_functions_once() {
        let src = super::shader_source();
        assert_eq!(src.matches("fn sky_color(").count(), 1);
        assert_eq!(src.matches("fn shade(").count(), 1);
        assert_eq!(src.matches("fn present(").count(), 1);
        assert_eq!(src.matches("fn ground_fog(").count(), 1);
        assert!(src.contains("ground_fog(lit, in.world)"));
        assert!(!src.contains("pow(d, 900.0)"));
        assert!(!src.contains("0.22, 0.095, 0.032"));
        assert!(!src.contains("vec3<f32>(0.10, 0.042, 0.015)"));
    }
}
