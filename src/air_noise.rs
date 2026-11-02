//! Tileable 3D noise for the picture's dust and fog. The solver never reads it.
//!
//! One RGBA8 snorm volume, `AIR_NOISE` texels on a side, wrapping on every
//! axis. RGB is a divergence-free curl field (the curl of a vector potential
//! made of three value-noise fbm fields), scaled so its longest vector is 1.
//! A is billow noise, -1 to 1. The fog pass warps its lookups by the curl and
//! advects them with the air velocity, so the detail swirls instead of
//! sliding.

pub const AIR_NOISE: u32 = 64;

fn hash3(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8DA6_B343)
        ^ (y as u32).wrapping_mul(0xD816_3841)
        ^ (z as u32).wrapping_mul(0xCB1A_B31F)
        ^ seed.wrapping_mul(0x9E37_79B9);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0xFFFF) as f32 / 65535.0
}

/// Value noise with period `per` cells on every axis.
fn vnoise(p: [f32; 3], per: i32, seed: u32) -> f32 {
    let i = p.map(|v| v.floor());
    let f = [p[0] - i[0], p[1] - i[1], p[2] - i[2]];
    let u = f.map(|t| t * t * (3.0 - 2.0 * t));
    let (ix, iy, iz) = (i[0] as i32, i[1] as i32, i[2] as i32);
    let at = |dx: i32, dy: i32, dz: i32| {
        hash3(
            (ix + dx).rem_euclid(per),
            (iy + dy).rem_euclid(per),
            (iz + dz).rem_euclid(per),
            seed,
        )
    };
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(at(0, 0, 0), at(1, 0, 0), u[0]);
    let x10 = lerp(at(0, 1, 0), at(1, 1, 0), u[0]);
    let x01 = lerp(at(0, 0, 1), at(1, 0, 1), u[0]);
    let x11 = lerp(at(0, 1, 1), at(1, 1, 1), u[0]);
    lerp(lerp(x00, x10, u[1]), lerp(x01, x11, u[1]), u[2])
}

/// Three octaves, tiling over the whole volume: the base period is 4 cells
/// of a 16-texel lattice, each octave doubles it.
fn fbm(x: f32, y: f32, z: f32, seed: u32) -> f32 {
    let s = 4.0 / AIR_NOISE as f32;
    let (mut n, mut amp, mut norm) = (0.0, 0.5, 0.0);
    for o in 0..3 {
        let per = 4 << o;
        let k = s * (1 << o) as f32;
        n += vnoise([x * k, y * k, z * k], per, seed + o) * amp;
        norm += amp;
        amp *= 0.5;
    }
    n / norm
}

/// The volume, x fastest, then y, then z. Each texel is RGBA8 snorm.
pub fn curl_volume() -> Vec<[i8; 4]> {
    let n = AIR_NOISE as usize;
    let mut psi = vec![[0.0f32; 3]; n * n * n];
    let mut billow = vec![0.0f32; n * n * n];
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (fx, fy, fz) = (x as f32, y as f32, z as f32);
                let i = x + y * n + z * n * n;
                psi[i] = [
                    fbm(fx, fy, fz, 11),
                    fbm(fx + 17.0, fy, fz, 23),
                    fbm(fx, fy + 31.0, fz, 37),
                ];
                // Billows: the folded fbm reads as rounded lobes.
                let b = fbm(fx, fy, fz, 53);
                billow[i] = 1.0 - (2.0 * b - 1.0).abs() * 2.0;
            }
        }
    }
    let at = |x: usize, y: usize, z: usize, dx: i32, dy: i32, dz: i32| -> [f32; 3] {
        let xi = (x as i32 + dx).rem_euclid(n as i32) as usize;
        let yi = (y as i32 + dy).rem_euclid(n as i32) as usize;
        let zi = (z as i32 + dz).rem_euclid(n as i32) as usize;
        psi[xi + yi * n + zi * n * n]
    };
    let mut curl = vec![[0.0f32; 3]; n * n * n];
    let mut longest = 1e-6f32;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let dx = |c: usize| (at(x, y, z, 1, 0, 0)[c] - at(x, y, z, -1, 0, 0)[c]) * 0.5;
                let dy = |c: usize| (at(x, y, z, 0, 1, 0)[c] - at(x, y, z, 0, -1, 0)[c]) * 0.5;
                let dz = |c: usize| (at(x, y, z, 0, 0, 1)[c] - at(x, y, z, 0, 0, -1)[c]) * 0.5;
                let c = [dy(2) - dz(1), dz(0) - dx(2), dx(1) - dy(0)];
                longest = longest.max((c[0] * c[0] + c[1] * c[1] + c[2] * c[2]).sqrt());
                curl[x + y * n + z * n * n] = c;
            }
        }
    }
    let q = |v: f32| (v.clamp(-1.0, 1.0) * 127.0).round() as i8;
    curl.iter()
        .zip(&billow)
        .map(|(c, b)| [q(c[0] / longest), q(c[1] / longest), q(c[2] / longest), q(*b)])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbm_tiles_across_the_volume() {
        let p = AIR_NOISE as f32;
        for (x, y, z) in [(0.3, 5.1, 9.7), (12.0, 40.5, 3.3)] {
            let a = fbm(x, y, z, 11);
            let b = fbm(x + p, y, z + p, 11);
            assert!((a - b).abs() < 1e-4, "{a} {b}");
        }
    }

    #[test]
    fn curl_has_no_divergence() {
        let v = curl_volume();
        let n = AIR_NOISE as usize;
        let get = |x: usize, y: usize, z: usize, c: usize| {
            v[(x % n) + (y % n) * n + (z % n) * n * n][c] as f32 / 127.0
        };
        // Central differences of a central-difference curl cancel; allow quantisation.
        let mut worst = 0.0f32;
        for (x, y, z) in [(5, 9, 13), (30, 2, 41), (60, 33, 7)] {
            let div = (get(x + 1, y, z, 0) - get(x + n - 1, y, z, 0))
                + (get(x, y + 1, z, 1) - get(x, y + n - 1, z, 1))
                + (get(x, y, z + 1, 2) - get(x, y, z + n - 1, 2));
            worst = worst.max(div.abs());
        }
        assert!(worst < 0.25, "{worst}");
    }
}
