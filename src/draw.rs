//! Software frame of the dust field. Flat sun, no materials, no text.

use std::f32::consts::PI;

use glam::Vec3;

use dust_trail::DensityTexture;

use crate::niva_mesh::{Tri, BODY, CHASSIS, GLASS, LAMP_F, LAMP_R, TRIM, WHEEL};

const EXTINCTION: f32 = 640.0;
const SPLAT_RADIUS_M: f32 = 0.38;
const MIN_RHO: f32 = 2.0e-4;

pub struct Camera {
    pub eye: Vec3,
    pub target: Vec3,
    pub fov_y: f32,
}

struct Basis {
    eye: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    f: f32,
    w: f32,
    h: f32,
}

impl Basis {
    fn new(cam: &Camera, w: u32, h: u32) -> Self {
        let forward = (cam.target - cam.eye).normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let f = (h as f32 * 0.5) / (cam.fov_y * 0.5).tan();
        Self {
            eye: cam.eye,
            forward,
            right,
            up,
            f,
            w: w as f32,
            h: h as f32,
        }
    }

    fn project(&self, p: Vec3) -> Option<(f32, f32, f32)> {
        let v = p - self.eye;
        let z = v.dot(self.forward);
        if z < 0.12 {
            return None;
        }
        let x = v.dot(self.right);
        let y = v.dot(self.up);
        let sx = self.w * 0.5 + x / z * self.f;
        let sy = self.h * 0.5 - y / z * self.f;
        Some((sx, sy, z))
    }
}

/// Raster the Niva and splat the 3D density texture. This is not a volume render.
pub fn render(
    chassis_tris: &[Tri],
    chassis: Vec3,
    density: &DensityTexture,
    cam: &Camera,
    width: u32,
    height: u32,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    let mut color = vec![0.0_f32; n * 3];
    let mut depth = vec![1.0e6_f32; n];
    let basis = Basis::new(cam, width, height);
    fill_sky(&mut color, h, w);
    draw_ground(&mut color, &mut depth, &basis, chassis, w);
    let light = Vec3::new(-0.32, 0.88, 0.34).normalize();
    for tri in chassis_tris {
        let p = [tri.p[0] + chassis, tri.p[1] + chassis, tri.p[2] + chassis];
        draw_tri(&mut color, &mut depth, &basis, p, tri.kind, light, w);
    }
    let mut tau = vec![0.0_f32; n];
    for (pos, mass) in density.splats(MIN_RHO) {
        splat(&mut tau, &depth, &basis, pos, mass, SPLAT_RADIUS_M, w);
    }
    composite(&mut color, &tau, w, h);
    encode(&color)
}

fn fill_sky(color: &mut [f32], h: usize, w: usize) {
    let zenith = [0.46, 0.56, 0.66];
    let horizon = [0.72, 0.70, 0.62];
    for y in 0..h {
        let t = (y as f32 / h as f32).clamp(0.0, 1.0);
        let k = t * t;
        let rgb = [
            zenith[0] * (1.0 - k) + horizon[0] * k,
            zenith[1] * (1.0 - k) + horizon[1] * k,
            zenith[2] * (1.0 - k) + horizon[2] * k,
        ];
        for x in 0..w {
            let i = (y * w + x) * 3;
            color[i] = rgb[0];
            color[i + 1] = rgb[1];
            color[i + 2] = rgb[2];
        }
    }
}

fn draw_ground(color: &mut [f32], depth: &mut [f32], cam: &Basis, chassis: Vec3, w: usize) {
    let x0 = -18.0_f32;
    let x1 = 14.0_f32;
    let z0 = chassis.z - 22.0;
    let z1 = chassis.z + 8.0;
    let step = 1.15_f32;
    let mut z = z0;
    while z < z1 {
        let mut x = x0;
        while x < x1 {
            let y = 0.0;
            let a = Vec3::new(x, y, z);
            let b = Vec3::new(x + step, y, z);
            let c = Vec3::new(x + step, y, z + step);
            let d = Vec3::new(x, y, z + step);
            let shade = 0.97 + 0.03 * hash(x as i32, z as i32);
            let rgb = [0.30 * shade, 0.20 * shade, 0.12 * shade];
            draw_tri_rgb(color, depth, cam, [a, b, c], rgb, w);
            draw_tri_rgb(color, depth, cam, [a, c, d], rgb, w);
            x += step;
        }
        z += step;
    }
}

fn orient(ax: f32, ay: f32, bx: f32, by: f32, cx: f32, cy: f32) -> f32 {
    (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
}

fn hash(x: i32, z: i32) -> f32 {
    let mut n = x
        .wrapping_mul(374761393)
        .wrapping_add(z.wrapping_mul(668265263));
    n = (n ^ (n >> 13)).wrapping_mul(1274126177);
    (n & 0xffff) as f32 / 65535.0
}

fn albedo(kind: u8) -> [f32; 3] {
    match kind {
        BODY => [0.56, 0.50, 0.40],
        GLASS => [0.15, 0.19, 0.23],
        TRIM => [0.11, 0.11, 0.12],
        LAMP_F => [0.86, 0.82, 0.66],
        LAMP_R => [0.52, 0.09, 0.06],
        WHEEL => [0.045, 0.045, 0.048],
        CHASSIS => [0.09, 0.09, 0.09],
        _ => [0.4, 0.4, 0.4],
    }
}

fn draw_tri(
    color: &mut [f32],
    depth: &mut [f32],
    cam: &Basis,
    p: [Vec3; 3],
    kind: u8,
    light: Vec3,
    w: usize,
) {
    let n = (p[1] - p[0]).cross(p[2] - p[0]);
    let n = if n.length_squared() > 1.0e-12 {
        n.normalize()
    } else {
        Vec3::Y
    };
    let nd = n.dot(light).abs();
    let wrap = if kind == GLASS { 0.55 } else { 0.34 };
    let gain = 1.0 - wrap;
    let shade = wrap + gain * nd;
    let base = albedo(kind);
    let rgb = [base[0] * shade, base[1] * shade, base[2] * shade];
    draw_tri_rgb(color, depth, cam, p, rgb, w);
}

fn draw_tri_rgb(
    color: &mut [f32],
    depth: &mut [f32],
    cam: &Basis,
    p: [Vec3; 3],
    rgb: [f32; 3],
    w: usize,
) {
    for tri in clip_near(cam, p, 0.18) {
        draw_tri_front(color, depth, cam, tri, rgb, w);
    }
}

fn clip_near(cam: &Basis, p: [Vec3; 3], near: f32) -> Vec<[Vec3; 3]> {
    let z = [
        (p[0] - cam.eye).dot(cam.forward),
        (p[1] - cam.eye).dot(cam.forward),
        (p[2] - cam.eye).dot(cam.forward),
    ];
    let inside = [z[0] >= near, z[1] >= near, z[2] >= near];
    let n_in = inside.iter().filter(|b| **b).count();
    if n_in == 3 {
        return vec![p];
    }
    if n_in == 0 {
        return Vec::new();
    }
    let mut poly = Vec::with_capacity(4);
    for i in 0..3 {
        let j = (i + 1) % 3;
        if inside[i] {
            poly.push(p[i]);
        }
        if inside[i] != inside[j] {
            let t = (near - z[i]) / (z[j] - z[i]);
            poly.push(p[i] + (p[j] - p[i]) * t);
        }
    }
    if poly.len() == 3 {
        vec![[poly[0], poly[1], poly[2]]]
    } else if poly.len() >= 4 {
        vec![[poly[0], poly[1], poly[2]], [poly[0], poly[2], poly[3]]]
    } else {
        Vec::new()
    }
}

fn draw_tri_front(
    color: &mut [f32],
    depth: &mut [f32],
    cam: &Basis,
    p: [Vec3; 3],
    rgb: [f32; 3],
    w: usize,
) {
    let Some((x0, y0, z0)) = cam.project(p[0]) else {
        return;
    };
    let Some((x1, y1, z1)) = cam.project(p[1]) else {
        return;
    };
    let Some((x2, y2, z2)) = cam.project(p[2]) else {
        return;
    };
    let mut sx = [x0, x1, x2];
    let mut sy = [y0, y1, y2];
    let mut sz = [z0, z1, z2];
    let mut area = orient(sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]);
    if area < 0.0 {
        sx.swap(1, 2);
        sy.swap(1, 2);
        sz.swap(1, 2);
        area = -area;
    }
    if area < 0.5 {
        return;
    }
    let min_x = sx[0].min(sx[1]).min(sx[2]).floor().max(0.0) as i32;
    let max_x = sx[0].max(sx[1]).max(sx[2]).ceil().min(cam.w - 1.0) as i32;
    let min_y = sy[0].min(sy[1]).min(sy[2]).floor().max(0.0) as i32;
    let max_y = sy[0].max(sy[1]).max(sy[2]).ceil().min(cam.h - 1.0) as i32;
    if min_x > max_x || min_y > max_y {
        return;
    }
    let iz = [1.0 / sz[0], 1.0 / sz[1], 1.0 / sz[2]];
    for y in min_y..=max_y {
        let py = y as f32 + 0.5;
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let w0 = orient(sx[1], sy[1], sx[2], sy[2], px, py);
            let w1 = orient(sx[2], sy[2], sx[0], sy[0], px, py);
            let w2 = orient(sx[0], sy[0], sx[1], sy[1], px, py);
            if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                continue;
            }
            let inv = (w0 * iz[0] + w1 * iz[1] + w2 * iz[2]) / area;
            let z = 1.0 / inv;
            let i = y as usize * w + x as usize;
            if z < depth[i] {
                depth[i] = z;
                let o = i * 3;
                color[o] = rgb[0];
                color[o + 1] = rgb[1];
                color[o + 2] = rgb[2];
            }
        }
    }
}

fn splat(
    tau: &mut [f32],
    depth: &[f32],
    cam: &Basis,
    pos: Vec3,
    mass_kg: f32,
    radius_m: f32,
    w: usize,
) {
    let Some((sx, sy, z)) = cam.project(pos) else {
        return;
    };
    let rad = radius_m / z * cam.f;
    if rad < 0.8 {
        return;
    }
    let sigma = rad * 0.50;
    let inv = 1.0 / (2.0 * sigma * sigma);
    let sigma_w = (radius_m * 0.50).max(0.05);
    let column = EXTINCTION * mass_kg / (2.0 * PI * sigma_w * sigma_w);
    let r = (sigma * 2.4).ceil() as i32;
    let x0 = (sx as i32 - r).max(0);
    let y0 = (sy as i32 - r).max(0);
    let x1 = (sx as i32 + r).min(cam.w as i32 - 1);
    let y1 = (sy as i32 + r).min(cam.h as i32 - 1);
    for y in y0..=y1 {
        let dy = y as f32 + 0.5 - sy;
        for x in x0..=x1 {
            let dx = x as f32 + 0.5 - sx;
            let g = (-(dx * dx + dy * dy) * inv).exp();
            if g < 0.04 {
                continue;
            }
            let i = y as usize * w + x as usize;
            if z < depth[i] - 0.05 {
                tau[i] += column * g;
            }
        }
    }
}

fn composite(color: &mut [f32], tau: &[f32], w: usize, h: usize) {
    let dust = [0.84, 0.66, 0.38];
    let core = [0.58, 0.38, 0.16];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let t = tau[i];
            if t < 0.015 {
                continue;
            }
            let a = 1.0 - (-t.min(6.0)).exp();
            let u = (t / 1.4).clamp(0.0, 1.0);
            let rgb = [
                dust[0] * (1.0 - u) + core[0] * u,
                dust[1] * (1.0 - u) + core[1] * u,
                dust[2] * (1.0 - u) + core[2] * u,
            ];
            let o = i * 3;
            color[o] = rgb[0] * a + color[o] * (1.0 - a);
            color[o + 1] = rgb[1] * a + color[o + 1] * (1.0 - a);
            color[o + 2] = rgb[2] * a + color[o + 2] * (1.0 - a);
        }
    }
}

fn encode(color: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(color.len());
    for &c in color {
        out.push(srgb(c));
    }
    out
}

fn srgb(l: f32) -> u8 {
    let x = l.clamp(0.0, 1.0);
    let s = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0 + 0.5) as u8
}

pub fn write_png(
    path: &std::path::Path,
    width: u32,
    height: u32,
    rgb: &[u8],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .and_then(|mut w| w.write_image_data(rgb))
        .map_err(|err| std::io::Error::new(std::io::ErrorKind::Other, err))
}
