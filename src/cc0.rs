//! CC0 picture assets: Grab3D dune buggy, Poly Haven / ambientCG ground maps,
//! Gobkit trees, rocks and grass. The soil solver is not involved.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec2, Vec3};
use image::imageops::FilterType;
use silt::soil::track_offset;
use silt::world::World;

use crate::mesh::{CpuMesh, MeshVertex};

const SRC_DIM: u32 = 512;
const BUGGY_DIM: u32 = 1024;

const BUGGY_GLB: &[u8] = include_bytes!("../assets/vehicle/dune-buggy.glb");

struct PropBytes {
    name: &'static str,
    bytes: &'static [u8],
    scale: f32,
}

const PROPS: &[PropBytes] = &[
    PropBytes { name: "TreeHigh001", bytes: include_bytes!("../assets/env/TreeHigh001.glb"), scale: 0.0065 },
    PropBytes { name: "TreeHigh002", bytes: include_bytes!("../assets/env/TreeHigh002.glb"), scale: 0.0065 },
    PropBytes { name: "TreeHigh003", bytes: include_bytes!("../assets/env/TreeHigh003.glb"), scale: 0.0062 },
    PropBytes { name: "TreeMed001", bytes: include_bytes!("../assets/env/TreeMed001.glb"), scale: 0.0080 },
    PropBytes { name: "TreeMed002", bytes: include_bytes!("../assets/env/TreeMed002.glb"), scale: 0.0076 },
    PropBytes { name: "TreeMed003", bytes: include_bytes!("../assets/env/TreeMed003.glb"), scale: 0.0072 },
    PropBytes { name: "TreeLow001", bytes: include_bytes!("../assets/env/TreeLow001.glb"), scale: 0.010 },
    PropBytes { name: "TreeLow002", bytes: include_bytes!("../assets/env/TreeLow002.glb"), scale: 0.010 },
    PropBytes { name: "TreeLow003", bytes: include_bytes!("../assets/env/TreeLow003.glb"), scale: 0.0095 },
    PropBytes { name: "TreeLow004", bytes: include_bytes!("../assets/env/TreeLow004.glb"), scale: 0.0090 },
    PropBytes { name: "Rock001", bytes: include_bytes!("../assets/env/Rock001.glb"), scale: 0.0045 },
    PropBytes { name: "Rock002", bytes: include_bytes!("../assets/env/Rock002.glb"), scale: 0.0042 },
    PropBytes { name: "Rock003", bytes: include_bytes!("../assets/env/Rock003.glb"), scale: 0.0040 },
    PropBytes { name: "Grass001", bytes: include_bytes!("../assets/env/Grass001.glb"), scale: 0.0022 },
    PropBytes { name: "Grass002", bytes: include_bytes!("../assets/env/Grass002.glb"), scale: 0.0024 },
    PropBytes { name: "Grass003", bytes: include_bytes!("../assets/env/Grass003.glb"), scale: 0.0020 },
];

const PROP_ATLAS: &[u8] = include_bytes!("../assets/env/Grass001.glb");

struct MapSpec {
    diff: &'static str,
    nor: &'static str,
    rough: Option<&'static str>,
    ao: Option<&'static str>,
    disp: Option<&'static str>,
    arm: Option<&'static str>,
}

const GROUND: &[MapSpec] = &[
    MapSpec {
        diff: "ground/mud_forest/mud_forest_diff_2k.jpg",
        nor: "ground/mud_forest/mud_forest_nor_gl_2k.jpg",
        rough: Some("ground/mud_forest/mud_forest_rough_2k.jpg"),
        ao: Some("ground/mud_forest/mud_forest_ao_2k.jpg"),
        disp: Some("ground/mud_forest/mud_forest_disp_2k.jpg"),
        arm: Some("ground/mud_forest/mud_forest_arm_2k.jpg"),
    },
    MapSpec {
        diff: "ground/brown_mud_dry/brown_mud_dry_diff_2k.jpg",
        nor: "ground/brown_mud_dry/brown_mud_dry_nor_gl_2k.jpg",
        rough: Some("ground/brown_mud_dry/brown_mud_dry_rough_2k.jpg"),
        ao: Some("ground/brown_mud_dry/brown_mud_dry_ao_2k.jpg"),
        disp: Some("ground/brown_mud_dry/brown_mud_dry_disp_2k.jpg"),
        arm: Some("ground/brown_mud_dry/brown_mud_dry_arm_2k.jpg"),
    },
    MapSpec {
        diff: "ground/clay_001/Clay001_2K-JPG_Color.jpg",
        nor: "ground/clay_001/Clay001_2K-JPG_NormalGL.jpg",
        rough: Some("ground/clay_001/Clay001_2K-JPG_Roughness.jpg"),
        ao: Some("ground/clay_001/Clay001_2K-JPG_AmbientOcclusion.jpg"),
        disp: None,
        arm: None,
    },
    MapSpec {
        diff: "ground/grass_ground/grass_ground_diff_2k.jpg",
        nor: "ground/grass_ground/grass_ground_nor_gl_2k.jpg",
        rough: Some("ground/grass_ground/grass_ground_rough_2k.jpg"),
        ao: Some("ground/grass_ground/grass_ground_ao_2k.jpg"),
        disp: Some("ground/grass_ground/grass_ground_disp_2k.jpg"),
        arm: Some("ground/grass_ground/grass_ground_arm_2k.jpg"),
    },
    MapSpec {
        diff: "ground/rocky_gravel/rocky_gravel_diff_2k.jpg",
        nor: "ground/rocky_gravel/rocky_gravel_nor_gl_2k.jpg",
        rough: Some("ground/rocky_gravel/rocky_gravel_rough_2k.jpg"),
        ao: Some("ground/rocky_gravel/rocky_gravel_ao_2k.jpg"),
        disp: Some("ground/rocky_gravel/rocky_gravel_disp_2k.jpg"),
        arm: Some("ground/rocky_gravel/rocky_gravel_arm_2k.jpg"),
    },
];

#[derive(Debug)]
pub struct RgbaTex {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub struct Cc0Scene {
    pub body: CpuMesh,
    pub glass: CpuMesh,
    pub wheels: [CpuMesh; 4],
    pub props: CpuMesh,
    /// Five CC0 grounds, unmixed: mud_forest, brown_mud_dry, clay, grass, gravel.
    pub ground_diff: [RgbaTex; 5],
    pub ground_nor: [RgbaTex; 5],
    pub ground_pack: [RgbaTex; 5],
    pub buggy_base: RgbaTex,
    pub buggy_nor: RgbaTex,
    pub buggy_mr: RgbaTex,
    pub tire_base: RgbaTex,
    pub tire_nor: RgbaTex,
    pub prop_tex: RgbaTex,
}

/// Body fit after the mesh X mirror. Hubs land on the physics hardpoints.
pub fn buggy_fit() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, -0.51, -0.143))
        * Mat4::from_scale(Vec3::new(0.974, 1.0, 1.108))
}

struct Tex {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

struct GroundMaps {
    diff: Tex,
    nor: Tex,
    rough: Tex,
    ao: Tex,
    disp: Option<Tex>,
    arm: Option<Tex>,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load() -> Cc0Scene {
    let maps = GROUND
        .iter()
        .map(|spec| {
            eprintln!("cc0 {}", spec.diff);
            GroundMaps {
                diff: tex_from_bytes(&read_file(spec.diff), SRC_DIM, true),
                nor: tex_from_bytes(&read_file(spec.nor), SRC_DIM, false),
                rough: tex_from_bytes(&read_file(spec.rough.expect("rough")), SRC_DIM, false),
                ao: tex_from_bytes(&read_file(spec.ao.expect("ao")), SRC_DIM, false),
                disp: spec.disp.map(|p| tex_from_bytes(&read_file(p), SRC_DIM, false)),
                arm: spec.arm.map(|p| tex_from_bytes(&read_file(p), SRC_DIM, false)),
            }
        })
        .collect();
    build(maps)
}

#[cfg(target_arch = "wasm32")]
pub async fn load() -> Cc0Scene {
    let mut maps = Vec::with_capacity(GROUND.len());
    for spec in GROUND {
        web_sys::console::log_1(&format!("cc0 {}", spec.diff).into());
        maps.push(GroundMaps {
            diff: tex_from_bytes(&fetch_file(spec.diff).await, SRC_DIM, true),
            nor: tex_from_bytes(&fetch_file(spec.nor).await, SRC_DIM, false),
            rough: tex_from_bytes(&fetch_file(spec.rough.expect("rough")).await, SRC_DIM, false),
            ao: tex_from_bytes(&fetch_file(spec.ao.expect("ao")).await, SRC_DIM, false),
            disp: match spec.disp {
                Some(p) => Some(tex_from_bytes(&fetch_file(p).await, SRC_DIM, false)),
                None => None,
            },
            arm: match spec.arm {
                Some(p) => Some(tex_from_bytes(&fetch_file(p).await, SRC_DIM, false)),
                None => None,
            },
        });
    }
    build(maps)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_file(rel: &str) -> Vec<u8> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("assets").join(rel);
    std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()))
}

#[cfg(target_arch = "wasm32")]
async fn fetch_file(rel: &str) -> Vec<u8> {
    use wasm_bindgen::JsCast;
    let url = format!("assets/{rel}");
    let window = web_sys::window().expect("window");
    let value = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(&url))
        .await
        .unwrap_or_else(|err| panic!("fetch {url}: {err:?}"));
    let resp: web_sys::Response = value.dyn_into().expect("response");
    if !resp.ok() {
        panic!("fetch {url} status {}", resp.status());
    }
    let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().expect("array buffer"))
        .await
        .expect("array buffer body");
    js_sys::Uint8Array::new(&buf).to_vec()
}

fn tex_from_bytes(bytes: &[u8], max_dim: u32, srgb: bool) -> Tex {
    let img = image::load_from_memory(bytes)
        .unwrap_or_else(|err| panic!("image: {err}"))
        .to_rgba8();
    let (w, h) = img.dimensions();
    let img = if w.max(h) > max_dim {
        let s = max_dim as f32 / w.max(h) as f32;
        let nw = ((w as f32) * s).round().max(1.0) as u32;
        let nh = ((h as f32) * s).round().max(1.0) as u32;
        image::imageops::resize(&img, nw, nh, FilterType::Triangle)
    } else {
        img
    };
    let (w, h) = img.dimensions();
    let mut rgba = img.into_raw();
    if srgb {
        for px in rgba.chunks_exact_mut(4) {
            px[0] = linear_u8(px[0]);
            px[1] = linear_u8(px[1]);
            px[2] = linear_u8(px[2]);
        }
    }
    Tex { w, h, rgba }
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

impl Tex {
    fn sample(&self, u: f32, v: f32) -> [f32; 4] {
        let w = self.w as f32;
        let h = self.h as f32;
        let x = u.rem_euclid(1.0) * w - 0.5;
        let y = v.rem_euclid(1.0) * h - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let tap = |xi: f32, yi: f32| -> [f32; 4] {
            let ix = xi.rem_euclid(w) as u32 % self.w;
            let iy = yi.rem_euclid(h) as u32 % self.h;
            let i = ((iy * self.w + ix) * 4) as usize;
            let p = &self.rgba[i..i + 4];
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
                p[3] as f32 / 255.0,
            ]
        };
        let lerp = |a: [f32; 4], b: [f32; 4], t: f32| -> [f32; 4] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
                a[3] + (b[3] - a[3]) * t,
            ]
        };
        let a = tap(x0, y0);
        let b = tap(x0 + 1.0, y0);
        let c = tap(x0, y0 + 1.0);
        let d = tap(x0 + 1.0, y0 + 1.0);
        lerp(lerp(a, b, fx), lerp(c, d, fx), fy)
    }
}

fn build(maps: Vec<GroundMaps>) -> Cc0Scene {
    let (ground_diff, ground_nor, ground_pack) = pack_layers(&maps);
    let (body, glass, wheels) = load_buggy();
    let props = place_props();
    let (buggy_base, buggy_nor, buggy_mr, tire_base, tire_nor) = buggy_textures();
    let prop_tex = prop_texture();
    eprintln!(
        "cc0 meshes body {} glass {} props {} wheels {}/{}/{}/{}",
        body.indices.len() / 3,
        glass.indices.len() / 3,
        props.indices.len() / 3,
        wheels[0].indices.len() / 3,
        wheels[1].indices.len() / 3,
        wheels[2].indices.len() / 3,
        wheels[3].indices.len() / 3
    );
    Cc0Scene {
        body,
        glass,
        wheels,
        props,
        ground_diff,
        ground_nor,
        ground_pack,
        buggy_base,
        buggy_nor,
        buggy_mr,
        tire_base,
        tire_nor,
        prop_tex,
    }
}

fn rgba_tex(width: u32, height: u32, rgba: Vec<u8>) -> RgbaTex {
    RgbaTex { width, height, rgba }
}

fn u8f(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn pack_layers(maps: &[GroundMaps]) -> ([RgbaTex; 5], [RgbaTex; 5], [RgbaTex; 5]) {
    assert_eq!(maps.len(), 5, "five CC0 grounds");
    let mut diff = Vec::with_capacity(5);
    let mut nor = Vec::with_capacity(5);
    let mut pack = Vec::with_capacity(5);
    for map in maps {
        assert_eq!(map.diff.w, map.nor.w);
        assert_eq!(map.diff.h, map.nor.h);
        diff.push(rgba_tex(map.diff.w, map.diff.h, map.diff.rgba.clone()));
        nor.push(rgba_tex(map.nor.w, map.nor.h, map.nor.rgba.clone()));
        let n = (map.rough.w * map.rough.h) as usize;
        let mut bytes = vec![0u8; n * 4];
        for i in 0..n {
            let r = map.rough.rgba[i * 4] as f32 / 255.0;
            let a = map.ao.rgba[i * 4] as f32 / 255.0;
            let (arm_r, arm_g, arm_b) = match &map.arm {
                Some(arm) => {
                    let s = i * 4;
                    (
                        arm.rgba[s] as f32 / 255.0,
                        arm.rgba[s + 1] as f32 / 255.0,
                        arm.rgba[s + 2] as f32 / 255.0,
                    )
                }
                None => (a, r, 0.0),
            };
            let rough = r * 0.65 + arm_g * 0.35;
            let ao = (a * arm_r).sqrt();
            let disp = match &map.disp {
                Some(d) => d.rgba[i * 4] as f32 / 255.0,
                None => 0.5,
            };
            let o = i * 4;
            bytes[o] = u8f(rough);
            bytes[o + 1] = u8f(ao);
            bytes[o + 2] = u8f(disp);
            bytes[o + 3] = u8f(arm_b);
        }
        pack.push(rgba_tex(map.rough.w, map.rough.h, bytes));
    }
    (
        diff.try_into().unwrap(),
        nor.try_into().unwrap(),
        pack.try_into().unwrap(),
    )
}

fn load_buggy() -> (CpuMesh, CpuMesh, [CpuMesh; 4]) {
    let doc = gltf::Gltf::from_slice(BUGGY_GLB).expect("dune buggy");
    let blob = doc.blob.as_deref().expect("dune buggy bin");
    let mut body = CpuMesh::new();
    let mut glass = CpuMesh::new();
    let mut wheels: HashMap<String, CpuMesh> = HashMap::new();
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        let name = node.name().unwrap_or("");
        if name.contains("_wheel_") {
            let mut cpu = CpuMesh::new();
            append_mesh(&mut cpu, &mesh, blob, Mat4::IDENTITY, false);
            wheels.insert(name.to_string(), cpu);
        } else if name.contains("window") {
            let mat = node_matrix(&node);
            append_mesh(&mut glass, &mesh, blob, mat, true);
        } else {
            let mat = node_matrix(&node);
            append_mesh(&mut body, &mesh, blob, mat, true);
        }
    }
    let order = [
        "dune-buggy_wheel_fr",
        "dune-buggy_wheel_fl",
        "dune-buggy_wheel_rr",
        "dune-buggy_wheel_rl",
    ];
    let wheels = order.map(|name| {
        wheels
            .remove(name)
            .unwrap_or_else(|| panic!("missing wheel {name}"))
    });
    (body, glass, wheels)
}

fn buggy_textures() -> (RgbaTex, RgbaTex, RgbaTex, RgbaTex, RgbaTex) {
    let doc = gltf::Gltf::from_slice(BUGGY_GLB).expect("dune buggy");
    let blob = doc.blob.as_deref().expect("dune buggy bin");
    let mut images: HashMap<String, Vec<u8>> = HashMap::new();
    for image in doc.images() {
        let name = image.name().unwrap_or("").to_string();
        let gltf::image::Source::View { view, .. } = image.source() else {
            continue;
        };
        let start = view.offset();
        let end = start + view.length();
        images.insert(name, blob[start..end].to_vec());
    }
    let take = |needle: &str| -> Vec<u8> {
        images
            .iter()
            .find(|(name, _)| name.contains(needle))
            .map(|(_, bytes)| bytes.clone())
            .unwrap_or_else(|| panic!("buggy image {needle}"))
    };
    let base = tex_from_bytes(&take("body_basecolor"), BUGGY_DIM, true);
    let nor = tex_from_bytes(&take("body_normal"), BUGGY_DIM, false);
    let mr = tex_from_bytes(&take("metallic"), BUGGY_DIM, false);
    let tire = tex_from_bytes(&take("tire_color"), 256, true);
    let tire_n = tex_from_bytes(&take("tire_normal"), 256, false);
    (
        owned_tex(base),
        owned_tex(nor),
        owned_tex(mr),
        owned_tex(tire),
        owned_tex(tire_n),
    )
}

fn prop_texture() -> RgbaTex {
    let doc = gltf::Gltf::from_slice(PROP_ATLAS).expect("prop atlas");
    let blob = doc.blob.as_deref().expect("prop bin");
    let image = doc.images().next().expect("prop image");
    let gltf::image::Source::View { view, .. } = image.source() else {
        panic!("prop image source");
    };
    let start = view.offset();
    let bytes = &blob[start..start + view.length()];
    owned_tex(tex_from_bytes(bytes, 512, true))
}

fn owned_tex(tex: Tex) -> RgbaTex {
    RgbaTex {
        width: tex.w,
        height: tex.h,
        rgba: tex.rgba,
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

fn panel_of(name: &str) -> f32 {
    let n = name.to_ascii_lowercase();
    if n.contains("window") {
        65.0
    } else if n.contains("tire") {
        61.0
    } else if n.contains("rim") {
        62.0
    } else if n.contains("hub") || n.contains("paint") {
        63.0
    } else if n.contains("trim") || n.contains("charcoal") {
        64.0
    } else {
        60.0
    }
}

fn append_mesh(dst: &mut CpuMesh, mesh: &gltf::Mesh, blob: &[u8], node: Mat4, mirror: bool) {
    for prim in mesh.primitives() {
        let panel = panel_of(prim.material().name().unwrap_or(""));
        let reader = prim.reader(|buffer| if buffer.index() == 0 { Some(blob) } else { None });
        let positions: Vec<[f32; 3]> = reader.read_positions().expect("pos").collect();
        let normals: Vec<[f32; 3]> = reader.read_normals().expect("nrm").collect();
        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|t| t.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0, 0.0]; positions.len()]);
        let indices: Vec<u32> = reader
            .read_indices()
            .map(|ix| ix.into_u32().collect())
            .unwrap_or_else(|| (0..positions.len() as u32).collect());
        let base = dst.vertices.len() as u32;
        for i in 0..positions.len() {
            let mut p = node.transform_point3(Vec3::from(positions[i]));
            let n0 = if i < normals.len() {
                Vec3::from(normals[i])
            } else {
                Vec3::Y
            };
            let mut n = node.transform_vector3(n0);
            if mirror {
                p.x = -p.x;
                n.x = -n.x;
            }
            let uv = if i < uvs.len() { uvs[i] } else { [0.0, 0.0] };
            dst.vertices.push(MeshVertex {
                pos: p.to_array(),
                nrm: n.normalize_or_zero().to_array(),
                uv,
                panel,
            });
        }
        let tris = indices.len() / 3;
        for t in 0..tris {
            let a = base + indices[t * 3];
            let b = base + indices[t * 3 + 1];
            let c = base + indices[t * 3 + 2];
            if mirror {
                dst.indices.extend([a, c, b]);
            } else {
                dst.indices.extend([a, b, c]);
            }
        }
    }
}

fn place_props() -> CpuMesh {
    let soil = World::demo().soil;
    let mut mesh = CpuMesh::new();
    let by_name = |name: &str| -> &PropBytes {
        PROPS
            .iter()
            .find(|p| p.name == name)
            .unwrap_or_else(|| panic!("prop {name}"))
    };
    let trees = [
        ("TreeHigh001", -12.2, 0.4),
        ("TreeHigh002", 12.4, 1.2),
        ("TreeHigh003", -1.2, 14.4),
        ("TreeMed001", -11.8, 7.2),
        ("TreeMed002", 12.0, -6.4),
        ("TreeMed003", 2.8, -12.6),
        ("TreeLow001", -13.2, -4.4),
        ("TreeLow002", 13.4, 6.6),
        ("TreeLow003", -8.8, -10.8),
        ("TreeLow004", 7.2, 13.4),
        ("TreeMed001", -5.2, 13.8),
        ("TreeLow002", 10.8, -10.4),
    ];
    for (i, (name, x, z)) in trees.iter().enumerate() {
        let (x, z) = push_out(*x, *z, 3.3);
        let h = soil.sample(x, z).height;
        let prop = by_name(name);
        append_prop(&mut mesh, prop.bytes, x, h, z, hash01(i as u32 * 9 + 2) * std::f32::consts::TAU, prop.scale);
    }
    let rocks = [
        ("Rock001", -9.4, 1.6),
        ("Rock002", 9.3, 2.8),
        ("Rock003", -2.4, -10.0),
        ("Rock001", 3.6, 12.2),
        ("Rock002", -11.2, -5.6),
        ("Rock003", 10.8, 8.4),
        ("Rock001", 1.4, -11.4),
        ("Rock002", -9.0, 9.6),
        ("Rock003", 8.6, -8.8),
    ];
    for (i, (name, x, z)) in rocks.iter().enumerate() {
        let (x, z) = push_out(*x, *z, 2.15);
        let h = soil.sample(x, z).height;
        let prop = by_name(name);
        let scale = prop.scale * (0.85 + 0.3 * hash01(40 + i as u32));
        append_prop(&mut mesh, prop.bytes, x, h, z, hash01(20 + i as u32) * std::f32::consts::TAU, scale);
    }
    let grasses = ["Grass001", "Grass002", "Grass003"];
    let mut kept = 0u32;
    let mut n = 0u32;
    while kept < 52 && n < 500 {
        let x = (hash01(n.wrapping_mul(3) + 11) * 2.0 - 1.0) * 15.5;
        let z = (hash01(n.wrapping_mul(5) + 19) * 2.0 - 1.0) * 16.0 + 1.1;
        n += 1;
        let off = track_offset(x, z);
        let bog = (x * x + (z - 1.05).powi(2)).sqrt();
        if off.abs() < 1.85 || off.abs() > 8.2 || bog < 3.5 {
            continue;
        }
        // Keep the surface-state stills clear of blades: the rut, the grip
        // stripe, and the film/slurry patch.
        let on_still = (x.abs() < 1.7 && z.abs() < 3.3)
            || ((2.4..6.4).contains(&x) && (-1.6..2.0).contains(&z))
            || ((-6.6..-2.6).contains(&x) && (-2.4..1.8).contains(&z));
        if on_still {
            continue;
        }
        let h = soil.sample(x, z).height;
        let name = grasses[(kept as usize) % grasses.len()];
        let prop = by_name(name);
        let scale = prop.scale * (0.75 + 0.5 * hash01(n + 70));
        append_prop(
            &mut mesh,
            prop.bytes,
            x,
            h,
            z,
            hash01(n + 90) * std::f32::consts::TAU,
            scale,
        );
        kept += 1;
    }
    mesh
}

fn push_out(x: f32, z: f32, min_off: f32) -> (f32, f32) {
    if track_offset(x, z).abs() >= min_off {
        return (x, z);
    }
    let mut p = Vec2::new(x, z - 1.25);
    if p.length() < 0.2 {
        p = Vec2::new(1.0, 0.0);
    }
    for _ in 0..14 {
        p *= 1.14;
        let x = p.x;
        let z = p.y + 1.25;
        if track_offset(x, z).abs() >= min_off {
            return (x, z);
        }
    }
    (x, z)
}

fn hash01(n: u32) -> f32 {
    let mut x = n.wrapping_mul(747796405).wrapping_add(2891336453);
    x = (x >> 16) ^ x;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    (x & 0xffff) as f32 / 65535.0
}

fn append_prop(dst: &mut CpuMesh, glb: &[u8], x: f32, h: f32, z: f32, yaw: f32, scale: f32) {
    let doc = gltf::Gltf::from_slice(glb).expect("prop glb");
    let blob = doc.blob.as_deref().expect("prop bin");
    let mut local = CpuMesh::new();
    for node in doc.nodes() {
        let Some(mesh) = node.mesh() else { continue };
        append_mesh(&mut local, &mesh, blob, node_matrix(&node), false);
    }
    let min_y = local
        .vertices
        .iter()
        .map(|v| v.pos[1])
        .fold(f32::MAX, f32::min);
    let q = Quat::from_rotation_y(yaw);
    let base = dst.vertices.len() as u32;
    for v in &local.vertices {
        let p = Vec3::new(v.pos[0], v.pos[1] - min_y, v.pos[2]) * scale;
        let p = Vec3::new(x, h, z) + q * p;
        let n = q * Vec3::from(v.nrm);
        dst.vertices.push(MeshVertex {
            pos: p.to_array(),
            nrm: n.normalize_or_zero().to_array(),
            uv: v.uv,
            panel: 70.0,
        });
    }
    for idx in &local.indices {
        dst.indices.push(base + idx);
    }
}
