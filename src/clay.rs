//! Tileable clay grain for the ground shader.
//!
//! One RGBA8 image: tangent-space normal in RG, albedo amount in B, warm/cool
//! mix in A. The soil solver never reads this. Dry dirt samples it at full
//! strength; the fragment shader fades it where the trough is wet.

pub const CLAY_RES: u32 = 256;

pub fn clay_image() -> Vec<u8> {
    let n = CLAY_RES as usize;
    let mut height = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            height[j * n + i] = clay_height(u, v);
        }
    }
    let mut lo = f32::MAX;
    let mut hi = f32::MIN;
    for h in &height {
        lo = lo.min(*h);
        hi = hi.max(*h);
    }
    let span = (hi - lo).max(1.0e-4);
    // The height is smooth, so a texel step is small. This scale makes the
    // fine octave a shallow crumb and leaves the broad patches nearly flat.
    let slope = 6.5;
    let mut rgba = vec![0u8; n * n * 4];
    for j in 0..n {
        for i in 0..n {
            let h = height[j * n + i];
            let hl = height[j * n + (i + n - 1) % n];
            let hr = height[j * n + (i + 1) % n];
            let hd = height[((j + n - 1) % n) * n + i];
            let hu = height[((j + 1) % n) * n + i];
            let dx = (hl - hr) * slope;
            let dz = (hd - hu) * slope;
            let inv = 1.0 / (dx * dx + 1.0 + dz * dz).sqrt();
            let nx = dx * inv;
            let nz = dz * inv;
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let h01 = (h - lo) / span;
            let patch = value_noise(u * 3.0, v * 3.0, 3) * 0.46
                + value_noise(u * 5.0, v * 5.0, 5) * 0.34
                + value_noise(u * 8.0, v * 8.0, 8) * 0.20;
            let fleck = value_noise(u * 18.0, v * 18.0, 18);
            let pore = value_noise(u * 47.0, v * 47.0, 47);
            let warm = value_noise(u * 4.0, v * 4.0, 4) * 0.55
                + value_noise(u * 9.0, v * 9.0, 9) * 0.45;
            // Wide range. 0.5 is neutral in the shader; pits go dark, crumbs light.
            let lum = (0.08 + patch * 0.46 + h01 * 0.20 + fleck * 0.16 + pore * 0.10)
                .clamp(0.04, 0.96);
            let tint = (0.08 + warm * 0.78 + h01 * 0.10).clamp(0.0, 1.0);
            let o = (j * n + i) * 4;
            rgba[o] = encode_axis(nx);
            rgba[o + 1] = encode_axis(nz);
            rgba[o + 2] = (lum * 255.0).round() as u8;
            rgba[o + 3] = (tint * 255.0).round() as u8;
        }
    }
    rgba
}

/// Full mip chain, finest first. Each level is tightly packed RGBA (no row pad).
pub fn clay_mips(base: &[u8]) -> Vec<(u32, Vec<u8>)> {
    let mut size = CLAY_RES;
    let mut prev = base.to_vec();
    let mut levels = vec![(size, prev.clone())];
    while size > 1 {
        let next = size / 2;
        let mut dst = vec![0u8; (next * next * 4) as usize];
        for y in 0..next {
            for x in 0..next {
                let mut acc = [0u32; 4];
                for dy in 0..2 {
                    for dx in 0..2 {
                        let sx = x * 2 + dx;
                        let sy = y * 2 + dy;
                        let i = ((sy * size + sx) * 4) as usize;
                        for c in 0..4 {
                            acc[c] += prev[i + c] as u32;
                        }
                    }
                }
                let o = ((y * next + x) * 4) as usize;
                for c in 0..4 {
                    dst[o + c] = (acc[c] / 4) as u8;
                }
            }
        }
        size = next;
        prev = dst.clone();
        levels.push((size, dst));
    }
    levels
}

fn encode_axis(v: f32) -> u8 {
    ((v.clamp(-1.0, 1.0) * 0.5 + 0.5) * 255.0).round() as u8
}

fn clay_height(u: f32, v: f32) -> f32 {
    // Irregular clay. A Worley lattice reads as a dot grid once the oval
    // packs a crumb into a pixel, so the bumps are value noise.
    let broad = value_noise(u * 4.0, v * 4.0, 4);
    let mid = fbm(u, v, 9, 4);
    let fine = fbm(u, v, 36, 3);
    let grit = value_noise(u * 96.0, v * 96.0, 96);
    broad * 0.22 + mid * 0.48 + fine * 0.22 + grit * 0.08
}

fn fbm(u: f32, v: f32, mut period: i32, octaves: i32) -> f32 {
    let mut acc = 0.0;
    let mut wsum = 0.0;
    let mut amp = 0.55;
    for _ in 0..octaves {
        acc += value_noise(u * period as f32, v * period as f32, period) * amp;
        wsum += amp;
        amp *= 0.5;
        period *= 2;
    }
    acc / wsum
}

fn fade(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn hash(ix: i32, iy: i32) -> f32 {
    let mut n = ix.wrapping_mul(374761393).wrapping_add(iy.wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    n ^= n >> 16;
    (n as u32 as f32) * (1.0 / 4294967296.0)
}

fn value_noise(x: f32, y: f32, period: i32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = fade(x - x0 as f32);
    let fy = fade(y - y0 as f32);
    let p = period.max(1);
    let at = |i: i32, j: i32| hash(i.rem_euclid(p), j.rem_euclid(p));
    let a = at(x0, y0);
    let b = at(x0 + 1, y0);
    let c = at(x0, y0 + 1);
    let d = at(x0 + 1, y0 + 1);
    a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clay_tiles_and_is_not_flat() {
        for k in 0..8 {
            let t = k as f32 / 8.0;
            let a = clay_height(0.0, t);
            let b = clay_height(1.0, t);
            assert!((a - b).abs() < 1.0e-4, "u seam {a} {b}");
            let c = clay_height(t, 0.0);
            let d = clay_height(t, 1.0);
            assert!((c - d).abs() < 1.0e-4, "v seam {c} {d}");
        }
        let img = clay_image();
        assert_eq!(img.len(), (CLAY_RES * CLAY_RES * 4) as usize);
        let mut max_nx = 0i32;
        let mut sum_nx = 0i32;
        let mut min_b = 255i32;
        let mut max_b = 0i32;
        let mut min_a = 255i32;
        let mut max_a = 0i32;
        let n = (CLAY_RES * CLAY_RES) as i32;
        for p in img.chunks(4) {
            max_nx = max_nx.max((p[0] as i32 - 128).abs());
            sum_nx += (p[0] as i32 - 128).abs();
            min_b = min_b.min(p[2] as i32);
            max_b = max_b.max(p[2] as i32);
            min_a = min_a.min(p[3] as i32);
            max_a = max_a.max(p[3] as i32);
        }
        let mean_nx = sum_nx / n;
        assert!(max_b - min_b > 80, "albedo span {}", max_b - min_b);
        assert!(max_a - min_a > 40, "tint span {}", max_a - min_a);
        assert!(max_nx > 24, "normal is flat, max {max_nx}");
        assert!(mean_nx > 4, "normal is too gentle, mean {mean_nx}");
        let mips = clay_mips(&img);
        assert_eq!(mips.len(), 9);
        assert_eq!(mips.last().unwrap().0, 1);
    }
}
