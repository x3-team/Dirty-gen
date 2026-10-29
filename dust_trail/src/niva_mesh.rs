//! Exterior Lada Niva, fitted onto the solver hardpoints.
//!
//! The bin is the CC BY mesh with textures and the number plate removed.
//! See `assets/vehicle/NIVA-LICENSE.txt`.

use std::collections::HashMap;

use dust_trail::{HUBS, WHEEL_RADIUS_M};
use glam::Vec3;

const BIN: &[u8] = include_bytes!("../assets/vehicle/niva-exterior.bin");

pub const BODY: u8 = 0;
pub const GLASS: u8 = 1;
pub const TRIM: u8 = 2;
pub const LAMP_F: u8 = 3;
pub const LAMP_R: u8 = 4;
pub const WHEEL: u8 = 5;
pub const CHASSIS: u8 = 6;

#[derive(Clone, Copy)]
pub struct Tri {
    pub p: [Vec3; 3],
    pub kind: u8,
}

struct Part {
    kind: u8,
    hub: Vec3,
    verts: Vec<Vec3>,
    idx: Vec<u32>,
}

pub fn load_chassis() -> Vec<Tri> {
    let mut parts = read_parts();
    fit(&mut parts);
    let mut tris = Vec::new();
    for part in &parts {
        for t in part.idx.chunks_exact(3) {
            tris.push(Tri {
                p: [
                    part.verts[t[0] as usize],
                    part.verts[t[1] as usize],
                    part.verts[t[2] as usize],
                ],
                kind: part.kind,
            });
        }
    }
    tris
}

fn read_parts() -> Vec<Part> {
    let mut c = Cursor::new(BIN);
    let magic = c.bytes(4);
    assert_eq!(magic, b"NV01", "niva mesh magic");
    let n = c.u32() as usize;
    let mut parts = Vec::with_capacity(n);
    for _ in 0..n {
        let kind = c.u8();
        let hub = Vec3::new(c.f32(), c.f32(), c.f32());
        let nv = c.u32() as usize;
        let ni = c.u32() as usize;
        let mut verts = Vec::with_capacity(nv);
        for _ in 0..nv {
            verts.push(Vec3::new(c.f32(), c.f32(), c.f32()));
        }
        let mut idx = Vec::with_capacity(ni);
        for _ in 0..ni {
            idx.push(c.u32());
        }
        parts.push(Part {
            kind,
            hub,
            verts,
            idx,
        });
    }
    assert!(c.rest() == 0, "trailing bytes in niva mesh");
    parts
}

fn fit(parts: &mut [Part]) {
    let wheels: Vec<usize> = parts
        .iter()
        .enumerate()
        .filter(|(_, p)| p.kind == WHEEL)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(wheels.len(), 4, "niva wheels");
    let hubs: Vec<Vec3> = wheels.iter().map(|&i| parts[i].hub).collect();
    let mid_x = hubs.iter().map(|h| h.x).sum::<f32>() / hubs.len() as f32;
    let mid_z = hubs.iter().map(|h| h.z).sum::<f32>() / hubs.len() as f32;
    let mut slot = [None; 4];
    for (i, h) in hubs.iter().enumerate() {
        let front = h.z >= mid_z;
        let left = h.x < mid_x;
        let idx = match (left, front) {
            (true, true) => 0,
            (false, true) => 1,
            (true, false) => 2,
            (false, false) => 3,
        };
        slot[idx] = Some(i);
    }
    let src: [Vec3; 4] = std::array::from_fn(|i| hubs[slot[i].expect("wheel slot")]);
    let dst: [Vec3; 4] = std::array::from_fn(|i| Vec3::from(HUBS[i]));
    let src_mid = (src[0] + src[1] + src[2] + src[3]) * 0.25;
    let dst_mid = (dst[0] + dst[1] + dst[2] + dst[3]) * 0.25;
    let track = (src[1].x - src[0].x).abs().max(0.25);
    let base = (src[0].z - src[2].z).abs().max(0.25);
    let s = ((1.48 / track) + ((1.08 - (-1.18)) / base)) * 0.5;
    let flip_x = if src[1].x >= src[0].x { 1.0 } else { -1.0 };
    let flip_z = if src[0].z >= src[2].z { 1.0 } else { -1.0 };
    let map_p = |p: Vec3| {
        let d = p - src_mid;
        Vec3::new(d.x * s * flip_x, d.y * s, d.z * s * flip_z) + dst_mid
    };
    for part in parts.iter_mut() {
        for v in &mut part.verts {
            *v = map_p(*v);
        }
        part.hub = map_p(part.hub);
    }
    for (sidx, widx) in slot.into_iter().enumerate() {
        let part = &mut parts[wheels[widx.expect("slot")]];
        let hub = dst[sidx];
        let radius = wheel_radius(&part.verts, part.hub);
        let scale = WHEEL_RADIUS_M / radius.max(0.05);
        for v in &mut part.verts {
            *v = hub + (*v - part.hub) * scale;
        }
        part.hub = hub;
        round_tire(part);
    }
}

/// Split outer chords and push them back out to the tire circle.
fn round_tire(part: &mut Part) {
    for _ in 0..2 {
        let r_max = part
            .verts
            .iter()
            .map(|v| (v.y - part.hub.y).hypot(v.z - part.hub.z))
            .fold(0.01_f32, f32::max);
        subdivide_tire(part, r_max);
    }
}

fn subdivide_tire(part: &mut Part, r_max: f32) {
    let mut mids: HashMap<(u32, u32), u32> = HashMap::new();
    let old = std::mem::take(&mut part.idx);
    let mut next = Vec::with_capacity(old.len() * 4);
    for tri in old.chunks_exact(3) {
        let m01 = tire_mid(part, &mut mids, tri[0], tri[1], r_max);
        let m12 = tire_mid(part, &mut mids, tri[1], tri[2], r_max);
        let m20 = tire_mid(part, &mut mids, tri[2], tri[0], r_max);
        next.extend_from_slice(&[
            tri[0], m01, m20, m01, tri[1], m12, m20, m12, tri[2], m01, m12, m20,
        ]);
    }
    part.idx = next;
}

fn tire_mid(
    part: &mut Part,
    mids: &mut HashMap<(u32, u32), u32>,
    a: u32,
    b: u32,
    r_max: f32,
) -> u32 {
    let key = if a < b { (a, b) } else { (b, a) };
    if let Some(&i) = mids.get(&key) {
        return i;
    }
    let pa = part.verts[a as usize];
    let pb = part.verts[b as usize];
    let mut p = (pa + pb) * 0.5;
    let hub = part.hub;
    let ra = (pa.y - hub.y).hypot(pa.z - hub.z);
    let rb = (pb.y - hub.y).hypot(pb.z - hub.z);
    if ra > r_max * 0.78 && rb > r_max * 0.78 {
        let r = (ra + rb) * 0.5;
        let dy = p.y - hub.y;
        let dz = p.z - hub.z;
        let len = dy.hypot(dz).max(1.0e-4);
        p.y = hub.y + dy / len * r;
        p.z = hub.z + dz / len * r;
    }
    let i = part.verts.len() as u32;
    part.verts.push(p);
    mids.insert(key, i);
    i
}

fn wheel_radius(verts: &[Vec3], hub: Vec3) -> f32 {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in verts {
        let d = *v - hub;
        min = min.min(d);
        max = max.max(d);
    }
    let ext = max - min;
    let axle = if ext.x <= ext.y && ext.x <= ext.z {
        0
    } else if ext.y <= ext.z {
        1
    } else {
        2
    };
    let mut r = 0.01_f32;
    for v in verts {
        let d = *v - hub;
        let rad = match axle {
            0 => d.y.hypot(d.z),
            1 => d.x.hypot(d.z),
            _ => d.x.hypot(d.y),
        };
        r = r.max(rad);
    }
    r
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, i: 0 }
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        s
    }
    fn u8(&mut self) -> u8 {
        self.bytes(1)[0]
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
    fn rest(&self) -> usize {
        self.b.len() - self.i
    }
}
