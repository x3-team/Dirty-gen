//! Image-based light for the picture only. The solver never reads it.
//!
//! One equirect radiance map with a box-filtered mip chain (roughness picks the
//! level) and nine SH terms for diffuse irradiance. The map is an HDRI from
//! `assets/hdri/` when one is there, turned so its sun sits on the scene sun.
//! Without one, the analytic sky the shader already draws is baked into it.

use glam::Vec3;

pub const ENV_W: u32 = 1024;
pub const ENV_H: u32 = 512;

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
    /// Daylight: a Preetham clear sky (turbidity 2.4) at the zenith level
    /// instead of the two-colour gradient.
    pub physical: bool,
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

/// Perez distribution, Preetham 1999: `(1 + A e^(B / cos t)) (1 + C e^(D g) + E cos^2 g)`.
fn perez(c: [f32; 5], cos_t: f32, gamma: f32) -> f32 {
    (1.0 + c[0] * (c[1] / cos_t.max(0.01)).exp())
        * (1.0 + c[2] * (c[3] * gamma).exp() + c[4] * gamma.cos().powi(2))
}

/// Clear-sky radiance in linear sRGB, 1.0 at the zenith luminance.
fn preetham(dir: Vec3, sun: Vec3) -> Vec3 {
    let t = 2.4f32;
    let ts = sun.y.clamp(0.02, 1.0).acos();
    let cy = [0.1787 * t - 1.4630, -0.3554 * t + 0.4275, -0.0227 * t + 5.3251, 0.1206 * t - 2.5771, -0.0670 * t + 0.3703];
    let cx = [-0.0193 * t - 0.2592, -0.0665 * t + 0.0008, -0.0004 * t + 0.2125, -0.0641 * t - 0.8989, -0.0033 * t + 0.0452];
    let cyy = [-0.0167 * t - 0.2608, -0.0950 * t + 0.0092, -0.0079 * t + 0.2102, -0.0441 * t - 1.6537, -0.0109 * t + 0.0529];
    let (t2, t3) = (ts * ts, ts * ts * ts);
    let xz = t * t * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * ts)
        + t * (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * ts + 0.00394)
        + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * ts + 0.25886);
    let yz = t * t * (0.00275 * t3 - 0.00610 * t2 + 0.00317 * ts)
        + t * (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * ts + 0.00516)
        + (0.15346 * t3 - 0.26756 * t2 + 0.06670 * ts + 0.26688);
    let cos_t = dir.y.max(0.015);
    let gamma = dir.dot(sun).clamp(-1.0, 1.0).acos();
    let lum = perez(cy, cos_t, gamma) / perez(cy, 1.0, ts);
    let x = xz * perez(cx, cos_t, gamma) / perez(cx, 1.0, ts);
    let y = yz * perez(cyy, cos_t, gamma) / perez(cyy, 1.0, ts);
    // xyY to XYZ to linear sRGB.
    let big_x = x / y * lum;
    let big_z = (1.0 - x - y) / y * lum;
    Vec3::new(
        3.2406 * big_x - 1.5372 * lum - 0.4986 * big_z,
        -0.9689 * big_x + 1.8758 * lum + 0.0415 * big_z,
        0.0557 * big_x - 0.2040 * lum + 1.0570 * big_z,
    )
    .max(Vec3::ZERO)
}

fn fbm(x: f32, z: f32, octaves: u32) -> f32 {
    let (mut n, mut amp, mut f, mut norm) = (0.0, 0.5, 1.0, 0.0);
    for o in 0..octaves {
        let r = o as f32 * 1.7;
        n += noise(x * f + 3.1 * r, z * f - 1.9 * r) * amp;
        norm += amp;
        amp *= 0.5;
        f *= 2.07;
    }
    n / norm
}

/// Fair-weather cumulus on a deck 1.6 km up, seen in perspective: big
/// heaps overhead, a dense band toward the horizon. Returns (colour, cover).
fn cumulus(dir: Vec3, sky: &SkyLight, behind: Vec3) -> (Vec3, f32) {
    if dir.y <= 0.004 {
        return (Vec3::ZERO, 0.0);
    }
    let dist = 1.6 / dir.y.max(0.03);
    let (px, pz) = (dir.x * dist, dir.z * dist);
    // Coverage: large heaps with a cauliflower edge.
    let base = fbm(px * 0.55 + 7.0, pz * 0.55 - 2.0, 3);
    let detail = fbm(px * 2.6, pz * 2.6, 4);
    let n = base * 0.78 + detail * 0.22;
    let cover = smooth(0.50, 0.66, n);
    // Thickness: how deep into the heap; the belly goes grey.
    let thick = smooth(0.52, 0.80, n);
    // Light comes from the sun side of each heap: offset sample toward the sun.
    let s = Vec3::new(sky.sun.x, 0.0, sky.sun.z).normalize_or_zero() * 0.35;
    let toward = fbm((px + s.x) * 0.55 + 7.0, (pz + s.z) * 0.55 - 2.0, 3) * 0.78 + detail * 0.22;
    let lit = smooth(-0.06, 0.05, n - toward);
    let sun_k = sky.sun_color * 0.62;
    let top = Vec3::splat(1.02) + sun_k * 0.25;
    let belly = Vec3::new(0.50, 0.55, 0.64);
    let mut col = belly.lerp(top, (1.0 - thick * 0.7).max(lit * 0.9).clamp(0.0, 1.0));
    // Silver lining where a thin edge faces the sun.
    let d = dir.dot(sky.sun).max(0.0);
    col += sky.sun_color * (1.0 - thick) * d.powf(8.0) * 0.6;
    // Far cloud melts into the haze.
    let haze = smooth(0.25, 0.02, dir.y);
    col = col.lerp(behind * 1.08, haze * 0.65);
    (col, cover * smooth(0.004, 0.06, dir.y))
}

/// The analytic sky without the sun disc (the sun is direct light),
/// and a lit brown ground below the horizon so a wet floor reflects soil.
fn sky_radiance(dir: Vec3, sky: &SkyLight) -> Vec3 {
    let d = dir.dot(sky.sun).max(0.0);
    let mut col = if sky.physical {
        let zl = sky.zenith.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        preetham(Vec3::new(dir.x, dir.y.max(0.0), dir.z).normalize(), sky.sun) * zl
    } else {
        let h = dir.y.clamp(-0.05, 1.0);
        sky.horizon.lerp(sky.zenith, smooth(-0.02, 0.55, h)) + sky.sun_color * d.powf(18.0) * 0.55
    };
    let (cloud_col, cover) = cumulus(dir, sky, col);
    col = col.lerp(cloud_col, cover);
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
    // A "puresky" file mirrors the sky below the horizon. Under it lies the
    // lit soil, the same ground the analytic sky puts there.
    for y in ENV_H / 2..ENV_H {
        for x in 0..ENV_W {
            let dir = texel_dir(x as f32, y as f32, ENV_W, ENV_H);
            let k = smooth(0.0, -0.08, dir.y);
            if k <= 0.0 {
                continue;
            }
            let g = sky_radiance(dir, sky) / scale.max(1e-6);
            let c = &mut px[(y * ENV_W + x) as usize];
            c[0] += (g.x - c[0]) * k;
            c[1] += (g.y - c[1]) * k;
            c[2] += (g.z - c[2]) * k;
        }
    }
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
    // Only an outdoor sky lights this scene. `sky.hdr` or `sky_*.hdr` is
    // picked up by itself; any other file (an interior such as lebombo, a night sky) only
    // by naming it in SILT_HDRI.
    let path = std::env::var_os("SILT_HDRI")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            files
                .iter()
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n == "sky.hdr" || n.starts_with("sky_"))
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
