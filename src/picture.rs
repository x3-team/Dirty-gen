//! The Lada Niva on the course. Poly Haven and Gobkit plants stay on disk
//! and are not instanced: the leaf cards turned into specks and the kit
//! only cost triangles.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use image::imageops::FilterType;

use crate::cc0::RgbaTex;
use crate::mesh::{CpuMesh, MeshVertex};

const ATLAS: u32 = 2048;
const CELL: u32 = 512;

const NIVA: &[u8] = include_bytes!("../assets/vehicle/niva-greg/lada_niva.glb");

pub const PANEL_NIVA: f32 = 72.0;
pub const PANEL_NIVA_GLASS: f32 = 73.0;

/// Centimetre Sketchfab model onto the physics hubs.
/// Wheel nodes sit near (±64.2, 32.8, ±100.5) cm. Scale 0.011244 then a
/// small shift lands those hubs on (±0.72, 0.22, 1.08 / -1.18).
pub fn niva_fit() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, -0.149, -0.05)) * Mat4::from_scale(Vec3::splat(0.011244))
}

pub struct Picture {
    pub flora: CpuMesh,
    pub niva: CpuMesh,
    pub niva_glass: CpuMesh,
    pub flora_tex: RgbaTex,
    pub niva_tex: RgbaTex,
}

pub fn load() -> Picture {
    let mut niva_atlas = Atlas::new();
    let (niva, niva_glass) = load_niva(&mut niva_atlas);
    eprintln!(
        "picture niva {} tris, glass {} tris, atlas slots {}",
        niva.indices.len() / 3,
        niva_glass.indices.len() / 3,
        niva_atlas.next
    );
    Picture {
        flora: CpuMesh::new(),
        niva,
        niva_glass,
        flora_tex: RgbaTex {
            width: 4,
            height: 4,
            rgba: vec![0; 4 * 4 * 4],
        },
        niva_tex: niva_atlas.finish(),
    }
}

struct Atlas {
    rgba: Vec<u8>,
    next: u32,
}

impl Atlas {
    fn new() -> Self {
        Self {
            rgba: vec![0; (ATLAS * ATLAS * 4) as usize],
            next: 0,
        }
    }

    fn insert(&mut self, img: image::RgbaImage) -> [f32; 4] {
        let cols = ATLAS / CELL;
        let slot = self.next;
        self.next += 1;
        if slot >= cols * cols {
            panic!("atlas full ({slot} slots)");
        }
        let (sw, sh) = img.dimensions();
        let scale = (CELL as f32 / sw.max(sh) as f32).min(1.0);
        let nw = ((sw as f32) * scale).round().max(1.0) as u32;
        let nh = ((sh as f32) * scale).round().max(1.0) as u32;
        let img = if nw != sw || nh != sh {
            image::imageops::resize(&img, nw, nh, FilterType::Triangle)
        } else {
            img
        };
        let ox = (slot % cols) * CELL + (CELL - nw) / 2;
        let oy = (slot / cols) * CELL + (CELL - nh) / 2;
        for y in 0..nh {
            for x in 0..nw {
                let p = img.get_pixel(x, y).0;
                let i = (((oy + y) * ATLAS + (ox + x)) * 4) as usize;
                self.rgba[i..i + 4].copy_from_slice(&p);
            }
        }
        let s = ATLAS as f32;
        [ox as f32 / s, oy as f32 / s, nw as f32 / s, nh as f32 / s]
    }

    fn finish(self) -> RgbaTex {
        RgbaTex {
            width: ATLAS,
            height: ATLAS,
            rgba: self.rgba,
        }
    }
}

fn load_niva(atlas: &mut Atlas) -> (CpuMesh, CpuMesh) {
    let mut body = CpuMesh::new();
    let mut glass = CpuMesh::new();
    append_glb(&mut body, NIVA, atlas, PANEL_NIVA, false, "niva");
    // Second pass would re-decode. Split by panel while appending instead.
    // append_glb already put glass on PANEL_NIVA_GLASS in `body` when the
    // material blends. Pull those vertices into the glass mesh.
    split_glass(&mut body, &mut glass);
    let fit = niva_fit();
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for v in body.vertices.iter().chain(glass.vertices.iter()) {
        let p = fit.transform_point3(Vec3::from(v.pos));
        min = min.min(p);
        max = max.max(p);
    }
    eprintln!(
        "niva fitted bbox ({:.2},{:.2},{:.2}) .. ({:.2},{:.2},{:.2})",
        min.x, min.y, min.z, max.x, max.y, max.z
    );
    (body, glass)
}

fn split_glass(body: &mut CpuMesh, glass: &mut CpuMesh) {
    let mut keep_v = Vec::new();
    let mut map = vec![u32::MAX; body.vertices.len()];
    for (i, v) in body.vertices.iter().enumerate() {
        if (v.panel - PANEL_NIVA_GLASS).abs() < 0.1 {
            continue;
        }
        map[i] = keep_v.len() as u32;
        keep_v.push(*v);
    }
    let mut keep_i = Vec::new();
    let mut glass_i = Vec::new();
    let mut glass_map: HashMap<u32, u32> = HashMap::new();
    for tri in body.indices.chunks_exact(3) {
        let panel = body.vertices[tri[0] as usize].panel;
        if (panel - PANEL_NIVA_GLASS).abs() < 0.1 {
            for &ix in tri {
                let slot = glass_map.entry(ix).or_insert_with(|| {
                    let id = glass.vertices.len() as u32;
                    glass.vertices.push(body.vertices[ix as usize]);
                    id
                });
                glass_i.push(*slot);
            }
        } else {
            keep_i.extend([map[tri[0] as usize], map[tri[1] as usize], map[tri[2] as usize]]);
        }
    }
    body.vertices = keep_v;
    body.indices = keep_i;
    glass.indices = glass_i;
}

fn append_glb(
    dst: &mut CpuMesh,
    glb: &[u8],
    atlas: &mut Atlas,
    default_panel: f32,
    _ground: bool,
    label: &str,
) {
    let doc = gltf::Gltf::from_slice_without_validation(glb)
        .unwrap_or_else(|err| panic!("{label} glb: {err}"));
    let blob = doc.blob.as_deref().unwrap_or_else(|| panic!("{label} bin"));
    let count = doc.nodes().len();
    let mut locals = vec![Mat4::IDENTITY; count];
    let mut parents = vec![None; count];
    for node in doc.nodes() {
        locals[node.index()] = node_matrix(&node);
        for child in node.children() {
            parents[child.index()] = Some(node.index());
        }
    }
    let mut cache: HashMap<usize, [f32; 4]> = HashMap::new();
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let xform = world_matrix(node.index(), &locals, &parents);
        for prim in mesh.primitives() {
            let mat = prim.material();
            let blend = matches!(mat.alpha_mode(), gltf::material::AlphaMode::Blend);
            let panel = if blend && default_panel == PANEL_NIVA {
                PANEL_NIVA_GLASS
            } else {
                default_panel
            };
            let punch = matches!(
                mat.alpha_mode(),
                gltf::material::AlphaMode::Blend | gltf::material::AlphaMode::Mask
            );
            let rect = albedo_rect(&doc, blob, &prim, atlas, &mut cache, punch);
            let (su, sv, ou, ov) = uv_xform(&prim);
            append_prim(dst, &prim, blob, xform, panel, rect, [su, sv, ou, ov]);
        }
    }
}

fn albedo_rect(
    doc: &gltf::Gltf,
    blob: &[u8],
    prim: &gltf::Primitive,
    atlas: &mut Atlas,
    cache: &mut HashMap<usize, [f32; 4]>,
    punch: bool,
) -> [f32; 4] {
    let Some(info) = prim.material().pbr_metallic_roughness().base_color_texture() else {
        return atlas.insert(solid([180, 170, 150, 255]));
    };
    let index = info.texture().source().index();
    if let Some(rect) = cache.get(&index) {
        return *rect;
    }
    let rect = atlas.insert(decode_image(doc, blob, index, punch));
    cache.insert(index, rect);
    rect
}

fn decode_image(doc: &gltf::Gltf, blob: &[u8], index: usize, punch: bool) -> image::RgbaImage {
    let image = doc.images().nth(index).unwrap_or_else(|| panic!("image {index}"));
    let gltf::image::Source::View { view, .. } = image.source() else {
        panic!("external gltf image");
    };
    let start = view.offset();
    let bytes = &blob[start..start + view.length()];
    let mut img = image::load_from_memory(bytes)
        .unwrap_or_else(|err| panic!("image {index}: {err}"))
        .to_rgba8();
    if punch {
        punch_black(&mut img);
    }
    for px in img.pixels_mut() {
        px[0] = linear_u8(px[0]);
        px[1] = linear_u8(px[1]);
        px[2] = linear_u8(px[2]);
    }
    img
}

fn punch_black(img: &mut image::RgbaImage) {
    let n = (img.width() * img.height()).max(1) as f32;
    let mut dark = 0u32;
    for p in img.pixels() {
        if p[0].max(p[1]).max(p[2]) < 18 {
            dark += 1;
        }
    }
    if dark as f32 / n < 0.22 {
        return;
    }
    for p in img.pixels_mut() {
        let m = p[0].max(p[1]).max(p[2]) as f32;
        let a = ((m - 6.0) / 28.0).clamp(0.0, 1.0);
        p[3] = (a.sqrt() * 255.0) as u8;
    }
}

fn solid(rgba: [u8; 4]) -> image::RgbaImage {
    image::RgbaImage::from_pixel(4, 4, image::Rgba(rgba))
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

fn uv_xform(prim: &gltf::Primitive) -> (f32, f32, f32, f32) {
    let Some(info) = prim.material().pbr_metallic_roughness().base_color_texture() else {
        return (1.0, 1.0, 0.0, 0.0);
    };
    let Some(tr) = info.texture_transform() else {
        return (1.0, 1.0, 0.0, 0.0);
    };
    let s = tr.scale();
    let o = tr.offset();
    (s[0], s[1], o[0], o[1])
}

fn append_prim(
    dst: &mut CpuMesh,
    prim: &gltf::Primitive,
    blob: &[u8],
    node: Mat4,
    panel: f32,
    rect: [f32; 4],
    xform_uv: [f32; 4],
) {
    let reader = prim.reader(|buffer| if buffer.index() == 0 { Some(blob) } else { None });
    let Some(pos_acc) = prim.get(&gltf::Semantic::Positions) else {
        return;
    };
    // Quantized Sketchfab meshes are normalized shorts. The stock reader only
    // yields float positions, so those vertices are decoded here.
    let positions = read_vec3(&pos_acc, blob);
    if positions.is_empty() {
        return;
    }
    let normals = prim
        .get(&gltf::Semantic::Normals)
        .map(|acc| read_vec3(&acc, blob))
        .unwrap_or_default();
    let uvs = prim
        .get(&gltf::Semantic::TexCoords(0))
        .map(|acc| read_vec2(&acc, blob))
        .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
    let indices: Vec<u32> = reader
        .read_indices()
        .map(|ix| ix.into_u32().collect())
        .unwrap_or_else(|| (0..positions.len() as u32).collect());
    let base = dst.vertices.len() as u32;
    let [su, sv, ou, ov] = xform_uv;
    for i in 0..positions.len() {
        let p = node.transform_point3(Vec3::from(positions[i]));
        let n0 = if i < normals.len() {
            Vec3::from(normals[i])
        } else {
            Vec3::Y
        };
        let n = node.transform_vector3(n0);
        let uv = if i < uvs.len() { uvs[i] } else { [0.0, 0.0] };
        let u = (uv[0] * su + ou).rem_euclid(1.0);
        let v = (uv[1] * sv + ov).rem_euclid(1.0);
        dst.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.normalize_or_zero().to_array(),
            uv: [rect[0] + u * rect[2], rect[1] + v * rect[3]],
            panel,
        });
    }
    let tris = indices.len() / 3;
    for t in 0..tris {
        dst.indices.extend([
            base + indices[t * 3],
            base + indices[t * 3 + 1],
            base + indices[t * 3 + 2],
        ]);
    }
}

fn read_vec3(accessor: &gltf::Accessor, blob: &[u8]) -> Vec<[f32; 3]> {
    let mut out = Vec::with_capacity(accessor.count());
    for i in 0..accessor.count() {
        out.push([
            component_at(accessor, blob, i, 0),
            component_at(accessor, blob, i, 1),
            component_at(accessor, blob, i, 2),
        ]);
    }
    out
}

fn read_vec2(accessor: &gltf::Accessor, blob: &[u8]) -> Vec<[f32; 2]> {
    let mut out = Vec::with_capacity(accessor.count());
    for i in 0..accessor.count() {
        out.push([
            component_at(accessor, blob, i, 0),
            component_at(accessor, blob, i, 1),
        ]);
    }
    out
}

fn component_at(accessor: &gltf::Accessor, blob: &[u8], index: usize, component: usize) -> f32 {
    let Some(view) = accessor.view() else {
        return 0.0;
    };
    let width = match accessor.data_type() {
        gltf::accessor::DataType::I8 | gltf::accessor::DataType::U8 => 1,
        gltf::accessor::DataType::I16 | gltf::accessor::DataType::U16 => 2,
        gltf::accessor::DataType::U32 | gltf::accessor::DataType::F32 => 4,
    };
    let dims = match accessor.dimensions() {
        gltf::accessor::Dimensions::Scalar => 1,
        gltf::accessor::Dimensions::Vec2 => 2,
        gltf::accessor::Dimensions::Vec3 => 3,
        gltf::accessor::Dimensions::Vec4 => 4,
        gltf::accessor::Dimensions::Mat2 => 4,
        gltf::accessor::Dimensions::Mat3 => 9,
        gltf::accessor::Dimensions::Mat4 => 16,
    };
    let stride = view.stride().unwrap_or(width * dims);
    let o = view.offset() + accessor.offset() + index * stride + component * width;
    if o + width > blob.len() {
        return 0.0;
    }
    let norm = accessor.normalized();
    match accessor.data_type() {
        gltf::accessor::DataType::F32 => f32::from_le_bytes(blob[o..o + 4].try_into().unwrap()),
        gltf::accessor::DataType::I16 => {
            let v = i16::from_le_bytes(blob[o..o + 2].try_into().unwrap());
            if norm {
                (v as f32 / 32767.0).max(-1.0)
            } else {
                v as f32
            }
        }
        gltf::accessor::DataType::U16 => {
            let v = u16::from_le_bytes(blob[o..o + 2].try_into().unwrap());
            if norm {
                v as f32 / 65535.0
            } else {
                v as f32
            }
        }
        gltf::accessor::DataType::I8 => {
            let v = blob[o] as i8;
            if norm {
                (v as f32 / 127.0).max(-1.0)
            } else {
                v as f32
            }
        }
        gltf::accessor::DataType::U8 => {
            let v = blob[o];
            if norm {
                v as f32 / 255.0
            } else {
                v as f32
            }
        }
        gltf::accessor::DataType::U32 => {
            u32::from_le_bytes(blob[o..o + 4].try_into().unwrap()) as f32
        }
    }
}

fn node_matrix(node: &gltf::Node) -> Mat4 {
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

fn world_matrix(mut index: usize, locals: &[Mat4], parents: &[Option<usize>]) -> Mat4 {
    let mut m = Mat4::IDENTITY;
    for _ in 0..32 {
        m = locals[index] * m;
        match parents[index] {
            Some(parent) => index = parent,
            None => break,
        }
    }
    m
}
