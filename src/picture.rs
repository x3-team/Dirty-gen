//! Photoreal course dressing and the Lada Niva.
//!
//! Trees, shrubs and rocks are Poly Haven CC0. The Niva is Greg McKechnie's
//! Sketchfab model (CC BY 4.0). Gobkit stays in the repo; it is too stylized
//! to carry the forest on its own, so it is not instanced on this course.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec2, Vec3};
use image::imageops::FilterType;
use silt::course;
use silt::soil::SoilField;

use crate::cc0::RgbaTex;
use crate::mesh::{CpuMesh, MeshVertex};

const ATLAS: u32 = 2048;
const CELL: u32 = 512;

const ISLAND: &[u8] = include_bytes!("../assets/env/photoreal/island_tree.glb");
const SEARSIA: &[u8] = include_bytes!("../assets/env/photoreal/searsia.glb");
const BOULDER: &[u8] = include_bytes!("../assets/env/photoreal/boulder.glb");
const TRUNK: &[u8] = include_bytes!("../assets/env/photoreal/dead_trunk.glb");
const MOSS: &[u8] = include_bytes!("../assets/env/photoreal/rock_moss.glb");
const GRASS: &[u8] = include_bytes!("../assets/env/photoreal/grass.glb");
const FERN: &[u8] = include_bytes!("../assets/env/photoreal/fern.glb");
const SHRUB: &[u8] = include_bytes!("../assets/env/photoreal/shrub.glb");
const NIVA: &[u8] = include_bytes!("../assets/vehicle/niva-greg/lada_niva.glb");

pub const PANEL_FLORA: f32 = 71.0;
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
    let mut flora_atlas = Atlas::new();
    let kits = [
        load_kit(ISLAND, &mut flora_atlas, "island_tree"),
        load_kit(SEARSIA, &mut flora_atlas, "searsia"),
        load_kit(BOULDER, &mut flora_atlas, "boulder"),
        load_kit(TRUNK, &mut flora_atlas, "dead_trunk"),
        load_kit(MOSS, &mut flora_atlas, "rock_moss"),
        load_kit(GRASS, &mut flora_atlas, "grass"),
        load_kit(FERN, &mut flora_atlas, "fern"),
        load_kit(SHRUB, &mut flora_atlas, "shrub"),
    ];
    let soil = course::field();
    let flora = scatter(&kits, &soil);
    let mut niva_atlas = Atlas::new();
    let (niva, niva_glass) = load_niva(&mut niva_atlas);
    eprintln!(
        "picture flora {} tris, niva {} tris, glass {} tris, atlas slots {}/{}",
        flora.indices.len() / 3,
        niva.indices.len() / 3,
        niva_glass.indices.len() / 3,
        flora_atlas.next,
        niva_atlas.next
    );
    Picture {
        flora,
        niva,
        niva_glass,
        flora_tex: flora_atlas.finish(),
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

fn load_kit(glb: &[u8], atlas: &mut Atlas, label: &str) -> CpuMesh {
    let mut mesh = CpuMesh::new();
    append_glb(&mut mesh, glb, atlas, PANEL_FLORA, false, label);
    let min_y = mesh
        .vertices
        .iter()
        .map(|v| v.pos[1])
        .fold(f32::MAX, f32::min);
    if min_y.is_finite() {
        for v in &mut mesh.vertices {
            v.pos[1] -= min_y;
        }
    }
    eprintln!(
        "flora {label} tris {}",
        mesh.indices.len() / 3
    );
    mesh
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

fn scatter(kits: &[CpuMesh; 8], soil: &SoilField) -> CpuMesh {
    let mut dst = CpuMesh::new();
    let tau = std::f32::consts::TAU;
    // Three belts past the berm. Crowns overlap, so the lap sits in woods.
    // The south slot in front of the wide camera stays low shrubs.
    for ring in 0..3u32 {
        let count = 22 + ring * 6;
        let radius = 23.2 + ring as f32 * 2.7;
        for i in 0..count {
            let n = i + ring * 50;
            let a = (i as f32 + (jitter(n) - 0.5) * 0.7) / count as f32 * tau;
            let r = radius + (jitter(n + 4) - 0.5) * 1.6;
            let Some((x, z)) = push_outside(a.cos() * r, a.sin() * r) else {
                continue;
            };
            if !shot_clear(x, z) {
                continue;
            }
            let yaw = jitter(n + 8) * tau;
            // Wide camera looks up the south side. Tall crowns there hide the lap.
            if z < -16.5 && x.abs() < 11.0 {
                stamp(
                    &mut dst,
                    &kits[7],
                    soil,
                    x,
                    z,
                    yaw,
                    18.0 + jitter(n + 2) * 8.0,
                );
                continue;
            }
            let scale = 3.15 + jitter(n + 11) * 1.45;
            stamp(&mut dst, &kits[1], soil, x, z, yaw, scale);
            if i % 2 == 0 {
                let jx = x + (jitter(n + 15) - 0.5) * 2.4;
                let jz = z + (jitter(n + 19) - 0.5) * 2.4;
                if course::forest_spot(jx, jz)
                    && shot_clear(jx, jz)
                    && jx.abs() < 31.0
                    && jz.abs() < 31.0
                {
                    stamp(
                        &mut dst,
                        &kits[1],
                        soil,
                        jx,
                        jz,
                        yaw + 1.1,
                        scale * (0.72 + jitter(n + 21) * 0.2),
                    );
                }
            }
        }
    }
    // Understory between the crowns. Shrub is a few centimetres; fern is short.
    for i in 0..64u32 {
        let a = (i as f32 + jitter(700 + i)) / 64.0 * tau;
        let r = 24.0 + jitter(800 + i) * 6.2;
        let x = a.cos() * r;
        let z = a.sin() * r;
        if !course::clear_of_track(x, z, 4.6) || !shot_clear(x, z) || x.abs() > 31.0 || z.abs() > 31.0
        {
            continue;
        }
        let shrub = i % 2 == 0;
        let kit = if shrub { 7 } else { 6 };
        let scale = if shrub {
            20.0 + jitter(900 + i) * 10.0
        } else {
            3.2 + jitter(900 + i) * 1.4
        };
        stamp(&mut dst, &kits[kit], soil, x, z, jitter(1000 + i) * tau, scale);
    }
    for (i, (x, z)) in [(28.0, 6.0), (-29.5, 10.0), (16.0, -29.0), (-6.0, 29.2), (29.0, -18.0)]
        .into_iter()
        .enumerate()
    {
        if let Some((x, z)) = push_outside(x, z) {
            if shot_clear(x, z) {
                stamp(
                    &mut dst,
                    &kits[2],
                    soil,
                    x,
                    z,
                    jitter(200 + i as u32) * tau,
                    1.2 + jitter(210 + i as u32) * 0.5,
                );
            }
        }
    }
    for (i, (x, z)) in [(-28.0, 20.0), (22.0, -26.0), (27.5, 24.0), (-22.0, -26.0)]
        .into_iter()
        .enumerate()
    {
        if let Some((x, z)) = push_outside(x, z) {
            if shot_clear(x, z) {
                stamp(&mut dst, &kits[3], soil, x, z, jitter(300 + i as u32) * tau, 1.05);
            }
        }
    }
    for i in 0..6u32 {
        let a = i as f32 / 6.0 * tau + 0.4;
        let x = a.cos() * 29.4;
        let z = a.sin() * 29.4;
        if let Some((x, z)) = push_outside(x, z) {
            if shot_clear(x, z) {
                stamp(&mut dst, &kits[4], soil, x, z, jitter(400 + i) * tau, 0.62);
            }
        }
    }
    for i in 0..22u32 {
        let a = i as f32 / 22.0 * tau + 0.15;
        let r = 26.5 + jitter(500 + i) * 3.8;
        let x = a.cos() * r;
        let z = a.sin() * r;
        if course::clear_of_track(x, z, 6.5) && shot_clear(x, z) && x.abs() < 31.0 && z.abs() < 31.0 {
            stamp(
                &mut dst,
                &kits[5],
                soil,
                x,
                z,
                jitter(520 + i) * tau,
                2.0 + jitter(540 + i) * 0.9,
            );
        }
    }
    // Opaque sorrel, so the belt reads as woods and not only leaf specks.
    for i in 0..48u32 {
        let a = (i as f32 + 0.37) / 48.0 * tau + jitter(1200 + i) * 0.08;
        let r = 24.5 + jitter(1300 + i) * 5.5;
        let x = a.cos() * r;
        let z = a.sin() * r;
        if course::clear_of_track(x, z, 4.8) && shot_clear(x, z) && x.abs() < 31.0 && z.abs() < 31.0
        {
            stamp(
                &mut dst,
                &kits[7],
                soil,
                x,
                z,
                jitter(1400 + i) * tau,
                26.0 + jitter(1500 + i) * 12.0,
            );
        }
    }
    // Low cover on the island. Nothing tall enough to hide the inner lap.
    stamp(&mut dst, &kits[7], soil, -0.8, 1.1, 0.4, 14.0);
    stamp(&mut dst, &kits[6], soil, 1.1, -0.6, 1.2, 2.4);
    dst
}

/// Keep the rut camera and the puddle camera from looking through a bush.
fn shot_clear(x: f32, z: f32) -> bool {
    let (hx, hz) = course::hole_center();
    // Wide enough that the pond camera is not inside a leaf card.
    if (x - hx).hypot(z - hz) < 13.0 {
        return false;
    }
    let (vx, vz, _) = course::hero();
    if (x - vx).hypot(z - vz) < 10.0 {
        return false;
    }
    true
}

fn push_outside(x: f32, z: f32) -> Option<(f32, f32)> {
    if course::forest_spot(x, z) && x.abs() < 31.2 && z.abs() < 31.2 {
        return Some((x, z));
    }
    let mut p = Vec2::new(x, z);
    if p.length() < 0.5 {
        return None;
    }
    for _ in 0..10 {
        p *= 1.06;
        if course::forest_spot(p.x, p.y) && p.x.abs() < 31.2 && p.y.abs() < 31.2 {
            return Some((p.x, p.y));
        }
    }
    None
}

fn stamp(dst: &mut CpuMesh, src: &CpuMesh, soil: &SoilField, x: f32, z: f32, yaw: f32, scale: f32) {
    let h = soil.sample(x, z).height;
    let q = Quat::from_rotation_y(yaw);
    let base = dst.vertices.len() as u32;
    for v in &src.vertices {
        let p = Vec3::from(v.pos) * scale;
        let p = Vec3::new(x, h, z) + q * p;
        let n = q * Vec3::from(v.nrm);
        dst.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.normalize_or_zero().to_array(),
            uv: v.uv,
            panel: v.panel,
        });
    }
    for idx in &src.indices {
        dst.indices.push(base + idx);
    }
}

fn jitter(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747796405).wrapping_add(2891336453);
    x = (x >> 16) ^ x;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    (x & 0xffff) as f32 / 65535.0
}
