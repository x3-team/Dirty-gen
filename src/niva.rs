//! Lada Niva mesh. The physics body is still the four-wheel solver; this module
//! only builds the picture. Wheel meshes stay separate so they can spin.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use image::imageops::FilterType;
use silt::vehicle::WHEEL_RADIUS;

use crate::cc0::RgbaTex;
use crate::mesh::{CpuMesh, MeshVertex};

const GLB: &[u8] = include_bytes!("../assets/vehicle/lada_niva-slim.glb");
const LAYERS: usize = 8;
const TEX: u32 = 512;

pub struct NivaScene {
    pub body: CpuMesh,
    pub glass: CpuMesh,
    pub wheels: [CpuMesh; 4],
    pub base: Vec<RgbaTex>,
    pub nor: Vec<RgbaTex>,
    pub mr: Vec<RgbaTex>,
}

pub fn load() -> NivaScene {
    let doc = gltf::Gltf::from_slice_without_validation(GLB).expect("niva glb");
    let bin = doc.blob.as_deref().expect("niva bin");
    let decoded = decompress(&doc, bin);
    let mut body = CpuMesh::new();
    let mut glass = CpuMesh::new();
    let mut raw_wheels: Vec<(Vec3, CpuMesh)> = Vec::new();
    let scene = doc
        .default_scene()
        .or_else(|| doc.scenes().next())
        .expect("niva scene");
    for node in scene.nodes() {
        walk(
            &node,
            Mat4::IDENTITY,
            bin,
            &decoded,
            &mut body,
            &mut glass,
            &mut raw_wheels,
        );
    }
    let mut cloud = body
        .vertices
        .iter()
        .chain(glass.vertices.iter())
        .map(|v| Vec3::from(v.pos).abs().max_element())
        .fold(0.0_f32, f32::max);
    for (_, mesh) in &raw_wheels {
        for v in &mesh.vertices {
            cloud = cloud.max(Vec3::from(v.pos).abs().max_element());
        }
    }
    // Sketchfab authored the Niva in centimetres.
    let unit = if cloud > 30.0 { 0.01 } else { 1.0 };
    scale_mesh(&mut body, unit);
    scale_mesh(&mut glass, unit);
    for (hub, mesh) in &mut raw_wheels {
        *hub *= unit;
        scale_mesh(mesh, unit);
    }
    let wheels = fit_to_hardpoints(&mut body, &mut glass, raw_wheels);
    let (base, nor, mr) = textures(&doc, bin);
    let (bmin, bmax) = mesh_bounds(&body);
    let ext = bmax - bmin;
    eprintln!(
        "niva body {} tris, glass {}, wheels {}/{}/{}/{} size {:.2}x{:.2}x{:.2} m",
        body.indices.len() / 3,
        glass.indices.len() / 3,
        wheels[0].indices.len() / 3,
        wheels[1].indices.len() / 3,
        wheels[2].indices.len() / 3,
        wheels[3].indices.len() / 3,
        ext.x,
        ext.y,
        ext.z
    );
    NivaScene {
        body,
        glass,
        wheels,
        base,
        nor,
        mr,
    }
}

fn walk(
    node: &gltf::Node,
    parent: Mat4,
    bin: &[u8],
    decoded: &[u8],
    body: &mut CpuMesh,
    glass: &mut CpuMesh,
    wheels: &mut Vec<(Vec3, CpuMesh)>,
) {
    let world = parent * local_matrix(node);
    let name = node.name().unwrap_or("");
    if let Some(mesh) = node.mesh() {
        let mut cpu = CpuMesh::new();
        append(&mut cpu, &mesh, bin, decoded, world);
        if name.starts_with("wheel_low") {
            let hub = world.transform_point3(Vec3::ZERO);
            wheels.push((hub, cpu));
        } else {
            let glass_only = mesh
                .primitives()
                .all(|p| p.material().index() == Some(3));
            if glass_only {
                glass.append(&mut cpu);
            } else {
                // A node can mix glass and paint. Split was already done per
                // primitive inside append when the destination is chosen here
                // only if every primitive is glass. Mixed nodes go to the body
                // and glass is pulled back out below.
                split_glass(&mut cpu, body, glass);
            }
        }
    }
    for child in node.children() {
        walk(&child, world, bin, decoded, body, glass, wheels);
    }
}

fn split_glass(src: &mut CpuMesh, body: &mut CpuMesh, glass: &mut CpuMesh) {
    // Vertices are already tagged. Glass is material 3 → panel 83.
    let mut remap_body = vec![u32::MAX; src.vertices.len()];
    let mut remap_glass = vec![u32::MAX; src.vertices.len()];
    for tri in src.indices.chunks_exact(3) {
        let glass_tri = src.vertices[tri[0] as usize].panel > 82.5
            && src.vertices[tri[0] as usize].panel < 83.5;
        let (dst, remap) = if glass_tri {
            (&mut *glass, &mut remap_glass)
        } else {
            (&mut *body, &mut remap_body)
        };
        let mut out = [0u32; 3];
        for (k, &ix) in tri.iter().enumerate() {
            if remap[ix as usize] == u32::MAX {
                remap[ix as usize] = dst.vertices.len() as u32;
                dst.vertices.push(src.vertices[ix as usize]);
            }
            out[k] = remap[ix as usize];
        }
        dst.indices.extend(out);
    }
}

fn fit_to_hardpoints(
    body: &mut CpuMesh,
    glass: &mut CpuMesh,
    raw: Vec<(Vec3, CpuMesh)>,
) -> [CpuMesh; 4] {
    assert!(raw.len() >= 4, "niva wheels {}", raw.len());
    let mut hubs: Vec<Vec3> = raw.iter().map(|(h, _)| *h).collect();
    let mid_x = hubs.iter().map(|h| h.x).sum::<f32>() / hubs.len() as f32;
    let mid_z = hubs.iter().map(|h| h.z).sum::<f32>() / hubs.len() as f32;
    // 0 FL, 1 FR, 2 RL, 3 RR. +Z is forward, +X is right.
    let mut slot: [Option<usize>; 4] = [None; 4];
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
    for (i, s) in slot.iter().enumerate() {
        assert!(s.is_some(), "niva missing wheel slot {i}, hubs {hubs:?}");
    }
    let src: [Vec3; 4] = std::array::from_fn(|i| hubs[slot[i].unwrap()]);
    let dst = [
        Vec3::new(-0.74, -0.20, 1.08),
        Vec3::new(0.74, -0.20, 1.08),
        Vec3::new(-0.74, -0.20, -1.18),
        Vec3::new(0.74, -0.20, -1.18),
    ];
    let src_mid = (src[0] + src[1] + src[2] + src[3]) * 0.25;
    let dst_mid = (dst[0] + dst[1] + dst[2] + dst[3]) * 0.25;
    let track = (src[1].x - src[0].x).abs().max(0.25);
    let base = (src[0].z - src[2].z).abs().max(0.25);
    let sx = 1.48 / track;
    let sz = (1.08 - (-1.18)) / base;
    // One scale keeps the body the shape Greg McKechnie modeled. The two
    // axis fits differ by a few centimetres, which would squash the cabin.
    let s = (sx + sz) * 0.5;
    let flip_x = if src[1].x >= src[0].x { 1.0 } else { -1.0 };
    let flip_z = if src[0].z >= src[2].z { 1.0 } else { -1.0 };
    eprintln!("niva hubs {src:?} scale {s:.3} flip ({flip_x},{flip_z})");
    let map_p = |p: Vec3| -> Vec3 {
        let d = p - src_mid;
        Vec3::new(d.x * s * flip_x, d.y * s, d.z * s * flip_z) + dst_mid
    };
    let map_n = |n: Vec3| -> Vec3 {
        Vec3::new(n.x * flip_x / s, n.y / s, n.z * flip_z / s).normalize_or_zero()
    };
    apply(body, map_p, map_n);
    apply(glass, map_p, map_n);
    let mut out: [Option<CpuMesh>; 4] = [None, None, None, None];
    for (i, (hub, mut mesh)) in raw.into_iter().enumerate() {
        let _ = &mut hubs;
        let idx = slot.iter().position(|s| *s == Some(i));
        let Some(idx) = idx else { continue };
        apply(&mut mesh, map_p, map_n);
        bake_wheel(&mut mesh, map_p(hub));
        out[idx] = Some(mesh);
    }
    out.map(|m| m.expect("wheel slot"))
}

fn apply(mesh: &mut CpuMesh, map_p: impl Fn(Vec3) -> Vec3, map_n: impl Fn(Vec3) -> Vec3) {
    for v in &mut mesh.vertices {
        v.pos = map_p(Vec3::from(v.pos)).to_array();
        v.nrm = map_n(Vec3::from(v.nrm)).to_array();
    }
}

fn bake_wheel(mesh: &mut CpuMesh, hub: Vec3) {
    for v in &mut mesh.vertices {
        v.pos = (Vec3::from(v.pos) - hub).to_array();
    }
    if mesh.vertices.is_empty() {
        return;
    }
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in &mesh.vertices {
        let p = Vec3::from(v.pos);
        min = min.min(p);
        max = max.max(p);
    }
    let ext = max - min;
    let (axis, rot) = if ext.y <= ext.x && ext.y <= ext.z {
        (1, Mat4::from_rotation_z(-std::f32::consts::FRAC_PI_2))
    } else if ext.z <= ext.x && ext.z <= ext.y {
        (2, Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2))
    } else {
        (0, Mat4::IDENTITY)
    };
    let _ = axis;
    let center = (min + max) * 0.5;
    for v in &mut mesh.vertices {
        let p = rot.transform_point3(Vec3::from(v.pos) - center);
        let n = rot.transform_vector3(Vec3::from(v.nrm)).normalize_or_zero();
        v.pos = p.to_array();
        v.nrm = n.to_array();
    }
    let mut radius = 0.01_f32;
    for v in &mesh.vertices {
        let p = Vec3::from(v.pos);
        radius = radius.max(p.y.abs()).max(p.z.abs());
    }
    let s = WHEEL_RADIUS / radius;
    for v in &mut mesh.vertices {
        v.pos = (Vec3::from(v.pos) * s).to_array();
    }
}

fn mesh_bounds(mesh: &CpuMesh) -> (Vec3, Vec3) {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in &mesh.vertices {
        let p = Vec3::from(v.pos);
        min = min.min(p);
        max = max.max(p);
    }
    (min, max)
}

fn scale_mesh(mesh: &mut CpuMesh, unit: f32) {
    if (unit - 1.0).abs() < 1.0e-6 {
        return;
    }
    for v in &mut mesh.vertices {
        v.pos = (Vec3::from(v.pos) * unit).to_array();
    }
}

fn local_matrix(node: &gltf::Node) -> Mat4 {
    match node.transform() {
        gltf::scene::Transform::Matrix { matrix } => Mat4::from_cols_array_2d(&matrix),
        gltf::scene::Transform::Decomposed {
            translation,
            rotation,
            scale,
        } => Mat4::from_scale_rotation_translation(
            Vec3::from(scale),
            Quat::from_xyzw(rotation[0], rotation[1], rotation[2], rotation[3]),
            Vec3::from(translation),
        ),
    }
}

fn append(dst: &mut CpuMesh, mesh: &gltf::Mesh, bin: &[u8], decoded: &[u8], world: Mat4) {
    let normal_mat = world.inverse().transpose();
    let flip = world.determinant() < 0.0;
    for prim in mesh.primitives() {
        let mat = prim.material().index().unwrap_or(0) as f32;
        let panel = 80.0 + mat;
        let positions = read_vec3(
            prim.get(&gltf::Semantic::Positions).expect("niva pos"),
            bin,
            decoded,
        );
        let normals = prim
            .get(&gltf::Semantic::Normals)
            .map(|acc| read_vec3(acc, bin, decoded))
            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
        let uvs = prim
            .get(&gltf::Semantic::TexCoords(0))
            .map(|acc| read_vec2(acc, bin, decoded))
            .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
        let indices = prim
            .indices()
            .map(|acc| read_indices(acc, bin, decoded))
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        let base = dst.vertices.len() as u32;
        for i in 0..positions.len() {
            let p = world.transform_point3(Vec3::from(positions[i]));
            let n = normal_mat
                .transform_vector3(Vec3::from(normals[i]))
                .normalize_or_zero();
            dst.vertices.push(MeshVertex {
                pos: p.to_array(),
                nrm: n.to_array(),
                uv: uvs[i],
                panel,
            });
        }
        for tri in indices.chunks_exact(3) {
            if flip {
                dst.indices
                    .extend([base + tri[0], base + tri[2], base + tri[1]]);
            } else {
                dst.indices
                    .extend([base + tri[0], base + tri[1], base + tri[2]]);
            }
        }
    }
}

/// Quantized glTF accessors (normalized i16 positions, octahedral i8 normals).
/// The stock reader only accepts float attributes.
fn read_vec3(accessor: gltf::Accessor, bin: &[u8], decoded: &[u8]) -> Vec<[f32; 3]> {
    read_floats(accessor, bin, decoded, 3)
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect()
}

fn read_vec2(accessor: gltf::Accessor, bin: &[u8], decoded: &[u8]) -> Vec<[f32; 2]> {
    read_floats(accessor, bin, decoded, 2)
        .chunks_exact(2)
        .map(|c| [c[0], c[1]])
        .collect()
}

fn read_indices(accessor: gltf::Accessor, bin: &[u8], decoded: &[u8]) -> Vec<u32> {
    let view = accessor.view().expect("index view");
    let buf = buffer_bytes(view.buffer().index(), bin, decoded);
    let stride = view.stride().unwrap_or(accessor.size());
    let start = view.offset() + accessor.offset();
    let mut out = Vec::with_capacity(accessor.count());
    for i in 0..accessor.count() {
        let at = start + i * stride;
        let v = match accessor.data_type() {
            gltf::accessor::DataType::U8 => buf[at] as u32,
            gltf::accessor::DataType::U16 => u16::from_le_bytes([buf[at], buf[at + 1]]) as u32,
            gltf::accessor::DataType::U32 => {
                u32::from_le_bytes([buf[at], buf[at + 1], buf[at + 2], buf[at + 3]])
            }
            other => panic!("index type {other:?}"),
        };
        out.push(v);
    }
    out
}

fn read_floats(accessor: gltf::Accessor, bin: &[u8], decoded: &[u8], comps: usize) -> Vec<f32> {
    let view = accessor.view().expect("attr view");
    let buf = buffer_bytes(view.buffer().index(), bin, decoded);
    let stride = view.stride().unwrap_or(accessor.size());
    let start = view.offset() + accessor.offset();
    let norm = accessor.normalized();
    let mut out = Vec::with_capacity(accessor.count() * comps);
    for i in 0..accessor.count() {
        let at = start + i * stride;
        for c in 0..comps {
            out.push(component(buf, at, c, accessor.data_type(), norm));
        }
    }
    out
}

fn component(buf: &[u8], at: usize, comp: usize, ty: gltf::accessor::DataType, norm: bool) -> f32 {
    let width = match ty {
        gltf::accessor::DataType::I8 | gltf::accessor::DataType::U8 => 1,
        gltf::accessor::DataType::I16 | gltf::accessor::DataType::U16 => 2,
        gltf::accessor::DataType::U32 | gltf::accessor::DataType::F32 => 4,
    };
    let p = at + comp * width;
    match ty {
        gltf::accessor::DataType::F32 => f32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]),
        gltf::accessor::DataType::I8 => {
            let v = buf[p] as i8 as f32;
            if norm { v / 127.0 } else { v }
        }
        gltf::accessor::DataType::U8 => {
            let v = buf[p] as f32;
            if norm { v / 255.0 } else { v }
        }
        gltf::accessor::DataType::I16 => {
            let v = i16::from_le_bytes([buf[p], buf[p + 1]]) as f32;
            if norm { (v / 32767.0).max(-1.0) } else { v }
        }
        gltf::accessor::DataType::U16 => {
            let v = u16::from_le_bytes([buf[p], buf[p + 1]]) as f32;
            if norm { v / 65535.0 } else { v }
        }
        gltf::accessor::DataType::U32 => u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]) as f32,
    }
}

fn buffer_bytes<'a>(index: usize, bin: &'a [u8], decoded: &'a [u8]) -> &'a [u8] {
    match index {
        0 => bin,
        1 => decoded,
        other => panic!("niva buffer {other}"),
    }
}

fn decompress(doc: &gltf::Gltf, bin: &[u8]) -> Vec<u8> {
    let len = doc.buffers().nth(1).map(|b| b.length()).unwrap_or(0);
    let mut out = vec![0u8; len];
    for view in doc.views() {
        let Some(ext) = view.extension_value("EXT_meshopt_compression") else {
            continue;
        };
        let buffer = ext.get("buffer").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let off = ext.get("byteOffset").and_then(|v| v.as_u64()).unwrap() as usize;
        let n = ext.get("byteLength").and_then(|v| v.as_u64()).unwrap() as usize;
        let stride = ext.get("byteStride").and_then(|v| v.as_u64()).unwrap() as usize;
        let count = ext.get("count").and_then(|v| v.as_u64()).unwrap() as usize;
        let mode = ext.get("mode").and_then(|v| v.as_str()).unwrap();
        let filter = ext.get("filter").and_then(|v| v.as_str());
        assert_eq!(buffer, 0, "meshopt source buffer");
        let src = &bin[off..off + n];
        let bytes = match (mode, stride, filter) {
            ("TRIANGLES", 2, _) => {
                let idx = meshopt::decode_index_buffer::<u16>(src, count).expect("meshopt index");
                bytes_of(&idx)
            }
            ("ATTRIBUTES", 4, Some("OCTAHEDRAL")) => {
                let mut verts =
                    meshopt::decode_vertex_buffer::<[i8; 4]>(src, count).expect("meshopt oct");
                meshopt::decode_filter_oct(&mut verts);
                bytes_of(&verts)
            }
            ("ATTRIBUTES", 4, _) => {
                let verts =
                    meshopt::decode_vertex_buffer::<[u16; 2]>(src, count).expect("meshopt uv");
                bytes_of(&verts)
            }
            ("ATTRIBUTES", 8, _) => {
                let verts =
                    meshopt::decode_vertex_buffer::<[i16; 4]>(src, count).expect("meshopt pos");
                bytes_of(&verts)
            }
            other => panic!("unexpected meshopt {other:?}"),
        };
        let dst = view.offset();
        assert_eq!(
            bytes.len(),
            view.length(),
            "meshopt size {} vs view {}",
            bytes.len(),
            view.length()
        );
        out[dst..dst + bytes.len()].copy_from_slice(&bytes);
    }
    out
}

fn bytes_of<T>(data: &[T]) -> Vec<u8> {
    let size = std::mem::size_of_val(data);
    let mut out = vec![0u8; size];
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, out.as_mut_ptr(), size);
    }
    out
}

fn textures(doc: &gltf::Gltf, bin: &[u8]) -> (Vec<RgbaTex>, Vec<RgbaTex>, Vec<RgbaTex>) {
    let mut images: HashMap<usize, Vec<u8>> = HashMap::new();
    for image in doc.images() {
        let gltf::image::Source::View { view, .. } = image.source() else {
            continue;
        };
        if view.buffer().index() != 0 {
            continue;
        }
        let start = view.offset();
        let end = start + view.length();
        images.insert(image.index(), bin[start..end].to_vec());
    }
    let mut tex_image = HashMap::new();
    for texture in doc.textures() {
        let Some(ext) = texture.extension_value("EXT_texture_webp") else {
            continue;
        };
        let source = ext.get("source").and_then(|v| v.as_u64()).unwrap() as usize;
        tex_image.insert(texture.index(), source);
    }
    let mut base = Vec::with_capacity(LAYERS);
    let mut nor = Vec::with_capacity(LAYERS);
    let mut mr = Vec::with_capacity(LAYERS);
    for _ in 0..LAYERS {
        base.push(flat(180, 180, 180, 255));
        nor.push(flat(128, 128, 255, 255));
        mr.push(flat(0, 255, 0, 255));
    }
    for mat in doc.materials() {
        let Some(idx) = mat.index() else { continue };
        if idx >= LAYERS {
            continue;
        }
        let pbr = mat.pbr_metallic_roughness();
        base[idx] = take(
            &images,
            &tex_image,
            pbr.base_color_texture().map(|t| t.texture().index()),
            true,
            flat(180, 180, 180, 255),
        );
        nor[idx] = take(
            &images,
            &tex_image,
            mat.normal_texture().map(|t| t.texture().index()),
            false,
            flat(128, 128, 255, 255),
        );
        mr[idx] = take(
            &images,
            &tex_image,
            pbr.metallic_roughness_texture().map(|t| t.texture().index()),
            false,
            flat(0, 255, 0, 255),
        );
    }
    (base, nor, mr)
}

fn take(
    images: &HashMap<usize, Vec<u8>>,
    tex_image: &HashMap<usize, usize>,
    texture: Option<usize>,
    srgb: bool,
    fallback: RgbaTex,
) -> RgbaTex {
    let Some(texture) = texture else {
        return fallback;
    };
    let Some(image) = tex_image.get(&texture).and_then(|i| images.get(i)) else {
        return fallback;
    };
    decode_tex(image, srgb).unwrap_or(fallback)
}

fn decode_tex(bytes: &[u8], srgb: bool) -> Option<RgbaTex> {
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let img = image::imageops::resize(&img, TEX, TEX, FilterType::Triangle);
    let mut rgba = img.into_raw();
    if srgb {
        for px in rgba.chunks_exact_mut(4) {
            for c in &mut px[..3] {
                *c = linear_u8(*c);
            }
        }
    }
    Some(RgbaTex {
        width: TEX,
        height: TEX,
        rgba,
    })
}

fn flat(r: u8, g: u8, b: u8, a: u8) -> RgbaTex {
    RgbaTex {
        width: TEX,
        height: TEX,
        rgba: vec![r, g, b, a].repeat((TEX * TEX) as usize),
    }
}

fn linear_u8(c: u8) -> u8 {
    let s = c as f32 / 255.0;
    let l = if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    };
    (l.clamp(0.0, 1.0) * 255.0).round() as u8
}

trait AppendMesh {
    fn append(&mut self, other: &mut CpuMesh);
}

impl AppendMesh for CpuMesh {
    fn append(&mut self, other: &mut CpuMesh) {
        let base = self.vertices.len() as u32;
        self.vertices.append(&mut other.vertices);
        self.indices
            .extend(other.indices.drain(..).map(|i| i + base));
    }
}
