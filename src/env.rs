//! Image-based light for the picture only. The solver never reads it.
//!
//! One equirect radiance map with a box-filtered mip chain (roughness picks the
//! level) and nine SH terms for diffuse irradiance. The map is an HDRI from
//! `assets/hdri/` when one is there, turned so its sun sits on the scene sun.
//! Without one, the analytic sky the shader already draws is baked into it.

use glam::Vec3;

pub const ENV_W: u32 = 512;
pub const ENV_H: u32 = 256;

pub struct EnvMap {
    /// Linear RGBA f32, level 0 first. Each level halves both sides.
    pub levels: Vec<(u32, u32, Vec<[f32; 4]>)>,
    /// Irradiance / pi as nine RGB SH terms, cosine lobe folded in.
    pub sh: [[f32; 4]; 9],
    /// True when the map came from an HDRI file.
    pub from_file: bool,
}

pub struct SkyLight {
    pub sun: Vec3,
    pub sun_color: Vec3,
    pub zenith: Vec3,
    pub horizon: Vec3,
}

/// Direction of the centre of texel (x, y). Matches `env_dir_uv` in the shader.
pub fn texel_dir(x: f32, y: f32, w: u32, h: u32) -> Vec3 {
    let phi = (x + 0.5) / w as f32 * std::f32::consts::TAU;
    let theta = (y + 0.5) / h as f32 * std::f32::consts::PI;
    Vec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin())
}

fn hash12(x: f32, y: f32) -> f32 {
    let h = x * 127.1 + y * 311.7;
    (h.sin() * 43758.547).rem_euclid(1.0)
}

fn noise(x: f32, y: f32) -> f32 {
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (x - ix, y - iy);
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = hash12(ix, iy);
    let b = hash12(ix + 1.0, iy);
    let c = hash12(ix, iy + 1.0);
    let d = hash12(ix + 1.0, iy + 1.0);
    let ab = a + (b - a) * ux;
    let cd = c + (d - c) * ux;
    ab + (cd - ab) * uy
}

fn smooth(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The shader's `sky_color` without the sun disc (the sun is direct light),
/// and a lit brown ground below the horizon so a wet floor reflects soil.
fn sky_radiance(dir: Vec3, sky: &SkyLight) -> Vec3 {
    let h = dir.y.clamp(-0.05, 1.0);
    let mut col = sky.horizon.lerp(sky.zenith, smooth(-0.02, 0.55, h));
    let d = dir.dot(sky.sun).max(0.0);
    col += sky.sun_color * d.powf(18.0) * 0.55;
    let s = 1.0 / (dir.y.abs() * 0.45 + 0.25);
    let n = noise(dir.x * s * 0.18, dir.z * s * 0.18);
    let cloud = smooth(0.58, 0.82, n) * smooth(0.08, 0.4, dir.y);
    col = col.lerp(Vec3::new(0.93, 0.94, 0.96), cloud * 0.7);
    if dir.y < 0.0 {
        let ground = Vec3::new(0.16, 0.11, 0.07)
            * (sky.sun_color * sky.sun.y.max(0.0) * 1.25 + sky.zenith * 0.4);
        col = col.lerp(ground, smooth(0.0, -0.08, dir.y));
    }
    col
}

pub fn procedural(sky: &SkyLight) -> EnvMap {
    let mut px = Vec::with_capacity((ENV_W * ENV_H) as usize);
    for y in 0..ENV_H {
        for x in 0..ENV_W {
            let c = sky_radiance(texel_dir(x as f32, y as f32, ENV_W, ENV_H), sky);
            px.push([c.x, c.y, c.z, 1.0]);
        }
    }
    finish(px, false)
}

/// Decode an HDRI, resample it to the map size, turn it so its brightest
/// patch sits at the sun's azimuth, and scale it to the analytic sky's level
/// so the exposure the soil was tuned at still holds.
pub fn from_hdr(bytes: &[u8], sky: &SkyLight) -> Option<EnvMap> {
    let img = image::load_from_memory(bytes).ok()?.to_rgb32f();
    let (sw, sh) = img.dimensions();
    if sw < 8 || sh < 4 {
        return None;
    }
    let src = img.into_raw();
    let fetch = |x: i64, y: i64| -> Vec3 {
        let x = x.rem_euclid(sw as i64) as usize;
        let y = y.clamp(0, sh as i64 - 1) as usize;
        let i = (y * sw as usize + x) * 3;
        Vec3::new(src[i], src[i + 1], src[i + 2])
    };
    // Box resample. A 4K file to 512 wide is an average, not a point pick.
    let fx = sw as f32 / ENV_W as f32;
    let fy = sh as f32 / ENV_H as f32;
    let mut base = vec![Vec3::ZERO; (ENV_W * ENV_H) as usize];
    for y in 0..ENV_H {
        for x in 0..ENV_W {
            let x0 = (x as f32 * fx) as i64;
            let y0 = (y as f32 * fy) as i64;
            let nx = fx.ceil().max(1.0) as i64;
            let ny = fy.ceil().max(1.0) as i64;
            let mut acc = Vec3::ZERO;
            for j in 0..ny {
                for i in 0..nx {
                    acc += fetch(x0 + i, y0 + j);
                }
            }
            base[(y * ENV_W + x) as usize] = acc / (nx * ny) as f32;
        }
    }
    // Brightest column in the upper half is the HDRI sun.
    let mut best = (0u32, -1.0f32);
    for y in 0..ENV_H / 2 {
        for x in 0..ENV_W {
            let c = base[(y * ENV_W + x) as usize];
            let l = c.dot(Vec3::new(0.2126, 0.7152, 0.0722));
            if l > best.1 {
                best = (x, l);
            }
        }
    }
    let sun_phi = sky.sun.z.atan2(sky.sun.x).rem_euclid(std::f32::consts::TAU);
    let sun_x = sun_phi / std::f32::consts::TAU * ENV_W as f32 - 0.5;
    let shift = (sun_x.round() as i64 - best.0 as i64).rem_euclid(ENV_W as i64) as u32;
    let mut px = vec![[0.0f32; 4]; (ENV_W * ENV_H) as usize];
    for y in 0..ENV_H {
        for x in 0..ENV_W {
            let c = base[(y * ENV_W + x) as usize];
            let to = (y * ENV_W + (x + shift) % ENV_W) as usize;
            px[to] = [c.x, c.y, c.z, 1.0];
        }
    }
    // Match the mean sky (sun clipped) to the analytic one the scene was lit with.
    let mut file_sum = 0.0f64;
    let mut proc_sum = 0.0f64;
    let mut lums: Vec<f32> = Vec::new();
    for y in 0..ENV_H / 2 {
        for x in 0..ENV_W {
            let c = px[(y * ENV_W + x) as usize];
            lums.push(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]);
        }
    }
    let mut sorted = lums.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let clip = sorted[(sorted.len() as f32 * 0.98) as usize].max(1e-6);
    for (k, l) in lums.iter().enumerate() {
        let y = k as u32 / ENV_W;
        let x = k as u32 % ENV_W;
        let w = (std::f32::consts::PI * (y as f32 + 0.5) / ENV_H as f32).sin() as f64;
        file_sum += l.min(clip) as f64 * w;
        let p = sky_radiance(texel_dir(x as f32, y as f32, ENV_W, ENV_H), sky);
        proc_sum += p.dot(Vec3::new(0.2126, 0.7152, 0.0722)) as f64 * w;
    }
    let scale = if file_sum > 0.0 { (proc_sum / file_sum) as f32 } else { 1.0 };
    for c in px.iter_mut() {
        // The sun stays direct light. Clamp the HDRI disc so it does not add
        // a second, unshadowed sun to every glossy surface.
        let k = scale;
        c[0] = (c[0] * k).min(24.0);
        c[1] = (c[1] * k).min(24.0);
        c[2] = (c[2] * k).min(24.0);
    }
    Some(finish(px, true))
}

fn finish(base: Vec<[f32; 4]>, from_file: bool) -> EnvMap {
    let mut levels = vec![(ENV_W, ENV_H, base)];
    loop {
        let (w, h, prev) = levels.last().unwrap();
        if *w <= 1 && *h <= 1 {
            break;
        }
        let nw = (*w / 2).max(1);
        let nh = (*h / 2).max(1);
        let mut next = vec![[0.0f32; 4]; (nw * nh) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let mut acc = [0.0f32; 4];
                let mut n = 0.0;
                for j in 0..2 {
                    for i in 0..2 {
                        let sx = (x * 2 + i).min(w - 1);
                        let sy = (y * 2 + j).min(h - 1);
                        let c = prev[(sy * w + sx) as usize];
                        for k in 0..4 {
                            acc[k] += c[k];
                        }
                        n += 1.0;
                    }
                }
                next[(y * nw + x) as usize] = acc.map(|v| v / n);
            }
        }
        levels.push((nw, nh, next));
    }
    let sh = project_sh(&levels[3]);
    EnvMap {
        levels,
        sh,
        from_file,
    }
}

/// Ramamoorthi–Hanrahan irradiance SH, divided by pi so the shader gets the
/// mean radiance a Lambert surface sees.
fn project_sh(level: &(u32, u32, Vec<[f32; 4]>)) -> [[f32; 4]; 9] {
    let (w, h, px) = level;
    let mut c = [Vec3::ZERO; 9];
    let mut total = 0.0f32;
    for y in 0..*h {
        for x in 0..*w {
            let d = texel_dir(x as f32, y as f32, *w, *h);
            let theta = std::f32::consts::PI * (y as f32 + 0.5) / *h as f32;
            let dw = theta.sin();
            total += dw;
            let p = px[(y * w + x) as usize];
            let col = Vec3::new(p[0], p[1], p[2]) * dw;
            let basis = [
                0.282095,
                0.488603 * d.y,
                0.488603 * d.z,
                0.488603 * d.x,
                1.092548 * d.x * d.y,
                1.092548 * d.y * d.z,
                0.315392 * (3.0 * d.z * d.z - 1.0),
                1.092548 * d.x * d.z,
                0.546274 * (d.x * d.x - d.y * d.y),
            ];
            for k in 0..9 {
                c[k] += col * basis[k];
            }
        }
    }
    let norm = 4.0 * std::f32::consts::PI / total;
    // Cosine-lobe convolution (A_l), then / pi.
    let a = [
        std::f32::consts::PI,
        2.094395,
        2.094395,
        2.094395,
        0.785398,
        0.785398,
        0.785398,
        0.785398,
        0.785398,
    ];
    let mut out = [[0.0f32; 4]; 9];
    for k in 0..9 {
        let v = c[k] * norm * a[k] / std::f32::consts::PI;
        out[k] = [v.x, v.y, v.z, 0.0];
    }
    out
}

/// IEEE half from f32, round to nearest. Enough for radiance.
pub fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00;
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7bff;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x80_0000) >> (1 - e + 13);
        return sign | m as u16;
    }
    let m = (mant + 0x1000) >> 13;
    if m & 0x400 != 0 {
        return sign | (((e + 1) as u16) << 10);
    }
    sign | ((e as u16) << 10) | (m as u16 & 0x3ff)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn find_hdri() -> Option<Vec<u8>> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/hdri");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("hdr")))
        .collect();
    files.sort();
    // Only an outdoor sky lights this scene. `sky_*.hdr` is picked up by
    // itself; any other file (an interior such as lebombo, a night sky) only
    // by naming it in SILT_HDRI.
    let path = std::env::var_os("SILT_HDRI")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            files
                .iter()
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("sky_"))
                })
                .cloned()
        })?;
    eprintln!("hdri {}", path.display());
    std::fs::read(path).ok()
}

#[cfg(target_arch = "wasm32")]
pub fn find_hdri() -> Option<Vec<u8>> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trips_common_values() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(-2.0), 0xc000);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(1.0e9), 0x7bff);
    }

    #[test]
    fn sh_of_a_flat_sky_is_its_radiance() {
        let n = 32 * 16;
        let px = vec![[0.5f32, 0.5, 0.5, 1.0]; n];
        let sh = project_sh(&(32, 16, px));
        // Constant radiance L gives E/pi = L for every normal.
        let e = sh[0][0] * 0.282095;
        assert!((e - 0.5).abs() < 0.02, "{e}");
    }
}
