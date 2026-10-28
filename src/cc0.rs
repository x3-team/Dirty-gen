//! CC0 picture assets: Grab3D dune buggy, Poly Haven / ambientCG ground maps,
//! Gobkit trees, rocks and grass. The soil solver is not involved.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};
use image::imageops::FilterType;
use silt::course;

use crate::mesh::{CpuMesh, MeshVertex};

pub const GROUND_RES: u32 = 2048;

/// Bake sources. The shipped maps stay 4K WebP; this only sizes the in-memory sample.
const SRC_DIM: u32 = 2048;
const BUGGY_DIM: u32 = 1024;

const BUGGY_GLB: &[u8] = include_bytes!("../assets/vehicle/dune-buggy.glb");

const PROP_ATLAS: &[u8] = include_bytes!("../assets/env/Grass001.glb");

struct MapSpec {
    diff: &'static str,
    nor: &'static str,
    disp: Option<&'static str>,
    /// Poly Haven ARM: R ambient occlusion, G roughness, B metal.
    /// When this is set, separate ao/rough files are not loaded.
    arm: Option<&'static str>,
    rough: Option<&'static str>,
    ao: Option<&'static str>,
}

const GROUND: &[MapSpec] = &[
    MapSpec {
        diff: "ground/mud_forest/mud_forest_diff_4k.webp",
        nor: "ground/mud_forest/mud_forest_nor_gl_4k.webp",
        disp: Some("ground/mud_forest/mud_forest_disp_4k.webp"),
        arm: Some("ground/mud_forest/mud_forest_arm_4k.webp"),
        rough: None,
        ao: None,
    },
    MapSpec {
        diff: "ground/brown_mud_dry/brown_mud_dry_diff_4k.webp",
        nor: "ground/brown_mud_dry/brown_mud_dry_nor_gl_4k.webp",
        disp: Some("ground/brown_mud_dry/brown_mud_dry_disp_4k.webp"),
        arm: Some("ground/brown_mud_dry/brown_mud_dry_arm_4k.webp"),
        rough: None,
        ao: None,
    },
    MapSpec {
        diff: "ground/clay_001/Clay001_diff_4k.webp",
        nor: "ground/clay_001/Clay001_nor_gl_4k.webp",
        disp: Some("ground/clay_001/Clay001_disp_4k.webp"),
        arm: None,
        rough: Some("ground/clay_001/Clay001_rough_4k.webp"),
        ao: Some("ground/clay_001/Clay001_ao_4k.webp"),
    },
    MapSpec {
        diff: "ground/grass_ground/grass_ground_diff_4k.webp",
        nor: "ground/grass_ground/grass_ground_nor_gl_4k.webp",
        disp: Some("ground/grass_ground/grass_ground_disp_4k.webp"),
        arm: Some("ground/grass_ground/grass_ground_arm_4k.webp"),
        rough: None,
        ao: None,
    },
    MapSpec {
        diff: "ground/rocky_gravel/rocky_gravel_diff_4k.webp",
        nor: "ground/rocky_gravel/rocky_gravel_nor_gl_4k.webp",
        disp: Some("ground/rocky_gravel/rocky_gravel_disp_4k.webp"),
        arm: Some("ground/rocky_gravel/rocky_gravel_arm_4k.webp"),
        rough: None,
        ao: None,
    },
];

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
    pub ground_diff: RgbaTex,
    pub ground_nor: RgbaTex,
    pub ground_pack: RgbaTex,
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
    disp: Option<Tex>,
    arm: Option<Tex>,
    rough: Option<Tex>,
    ao: Option<Tex>,
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load() -> Cc0Scene {
    let maps = GROUND.iter().map(load_spec).collect();
    build(maps)
}

#[cfg(target_arch = "wasm32")]
pub async fn load() -> Cc0Scene {
    let mut maps = Vec::with_capacity(GROUND.len());
    for spec in GROUND {
        web_sys::console::log_1(&format!("cc0 {}", spec.diff).into());
        maps.push(GroundMaps {
            diff: tex_from_bytes(&fetch_file(spec.diff).await, SRC_DIM, false),
            nor: tex_from_bytes(&fetch_file(spec.nor).await, SRC_DIM, false),
            disp: match spec.disp {
                Some(p) => Some(tex_from_bytes(&fetch_file(p).await, SRC_DIM, false)),
                None => None,
            },
            arm: match spec.arm {
                Some(p) => Some(tex_from_bytes(&fetch_file(p).await, SRC_DIM, false)),
                None => None,
            },
            rough: match spec.rough {
                Some(p) if spec.arm.is_none() => {
                    Some(tex_from_bytes(&fetch_file(p).await, SRC_DIM, false))
                }
                _ => None,
            },
            ao: match spec.ao {
                Some(p) if spec.arm.is_none() => {
                    Some(tex_from_bytes(&fetch_file(p).await, SRC_DIM, false))
                }
                _ => None,
            },
        });
    }
    build(maps)
}

#[cfg(not(target_arch = "wasm32"))]
fn load_spec(spec: &MapSpec) -> GroundMaps {
    eprintln!("cc0 {}", spec.diff);
    let file = |rel: &str| tex_from_bytes(&read_file(rel), SRC_DIM, false);
    // ARM carries ao, roughness, and metal. Clay has no ARM, so it keeps rough and ao.
    let separate = spec.arm.is_none();
    GroundMaps {
        diff: file(spec.diff),
        nor: file(spec.nor),
        disp: spec.disp.map(file),
        arm: spec.arm.map(file),
        rough: if separate { spec.rough.map(file) } else { None },
        ao: if separate { spec.ao.map(file) } else { None },
    }
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

fn srgb_to_linear(s: f32) -> f32 {
    let s = s.clamp(0.0, 1.0);
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(l: f32) -> u8 {
    let l = l.clamp(0.0, 1.0);
    let s = if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (s.clamp(0.0, 1.0) * 255.0).round() as u8
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
    let (ground_diff, ground_nor, ground_pack) = bake_ground(&maps);
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

fn bake_ground(maps: &[GroundMaps]) -> (RgbaTex, RgbaTex, RgbaTex) {
    let n = (GROUND_RES * GROUND_RES) as usize;
    let mut diff = vec![0u8; n * 4];
    let mut nor = vec![0u8; n * 4];
    let mut pack = vec![0u8; n * 4];
    let mut sum_w = [0.0f32; 5];
    let tiles = [3.2f32, 2.7, 3.5, 2.3, 3.3];
    for j in 0..GROUND_RES {
        for i in 0..GROUND_RES {
            let x = (i as f32 + 0.5) / GROUND_RES as f32 * course::SIZE - course::SIZE * 0.5;
            let z = (j as f32 + 0.5) / GROUND_RES as f32 * course::SIZE - course::SIZE * 0.5;
            let w = weights(x, z);
            for k in 0..5 {
                sum_w[k] += w[k];
            }
            let mut alb = [0.0f32; 3];
            let mut nrm = [0.0f32; 3];
            let mut rough = 0.0f32;
            let mut ao = 0.0f32;
            let mut disp = 0.0f32;
            let mut metal = 0.0f32;
            let mut wn = 0.0f32;
            for (k, map) in maps.iter().enumerate() {
                let wk = w[k];
                if wk < 0.02 {
                    continue;
                }
                wn += wk;
                let u = x / tiles[k];
                let v = z / tiles[k];
                let d = map.diff.sample(u, v);
                // Blend in linear. The bake is stored sRGB so 8-bit does not band the dirt.
                alb[0] += srgb_to_linear(d[0]) * wk;
                alb[1] += srgb_to_linear(d[1]) * wk;
                alb[2] += srgb_to_linear(d[2]) * wk;
                let ns = map.nor.sample(u, v);
                nrm[0] += (ns[0] * 2.0 - 1.0) * wk;
                nrm[1] += (ns[1] * 2.0 - 1.0) * wk;
                nrm[2] += (ns[2] * 2.0 - 1.0) * wk;
                let (ao_s, rough_s, metal_s) = if let Some(arm) = &map.arm {
                    let s = arm.sample(u, v);
                    (s[0], s[1], s[2])
                } else {
                    let rough = map
                        .rough
                        .as_ref()
                        .expect("rough map")
                        .sample(u, v)[0];
                    let ao = map.ao.as_ref().expect("ao map").sample(u, v)[0];
                    (ao, rough, 0.0)
                };
                rough += rough_s * wk;
                ao += ao_s * wk;
                let dv = match &map.disp {
                    Some(disp) => disp.sample(u, v)[0],
                    None => 0.5,
                };
                disp += dv * wk;
                metal += metal_s * wk;
            }
            if wn < 1e-4 {
                alb = [0.15, 0.12, 0.08];
                nrm = [0.0, 0.0, 1.0];
                rough = 0.9;
                ao = 1.0;
                disp = 0.5;
                metal = 0.0;
            } else {
                let inv = 1.0 / wn;
                alb[0] *= inv;
                alb[1] *= inv;
                alb[2] *= inv;
                nrm[0] *= inv;
                nrm[1] *= inv;
                nrm[2] *= inv;
                rough *= inv;
                ao *= inv;
                disp *= inv;
                metal *= inv;
            }
            let nl = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt().max(1e-5);
            let o = ((j * GROUND_RES + i) * 4) as usize;
            diff[o] = linear_to_srgb_u8(alb[0]);
            diff[o + 1] = linear_to_srgb_u8(alb[1]);
            diff[o + 2] = linear_to_srgb_u8(alb[2]);
            diff[o + 3] = 255;
            nor[o] = u8f(nrm[0] / nl * 0.5 + 0.5);
            nor[o + 1] = u8f(nrm[1] / nl * 0.5 + 0.5);
            nor[o + 2] = u8f(nrm[2] / nl * 0.5 + 0.5);
            nor[o + 3] = 255;
            pack[o] = u8f(rough);
            pack[o + 1] = u8f(ao);
            pack[o + 2] = u8f(disp);
            pack[o + 3] = u8f(metal);
        }
    }
    let pixels = (GROUND_RES * GROUND_RES) as f32;
    eprintln!(
        "cc0 weights mud {:.2} dry {:.2} clay {:.2} grass {:.2} rock {:.2}",
        sum_w[0] / pixels,
        sum_w[1] / pixels,
        sum_w[2] / pixels,
        sum_w[3] / pixels,
        sum_w[4] / pixels
    );
    (
        rgba_tex(GROUND_RES, GROUND_RES, diff),
        rgba_tex(GROUND_RES, GROUND_RES, nor),
        rgba_tex(GROUND_RES, GROUND_RES, pack),
    )
}

fn rgba_tex(width: u32, height: u32, rgba: Vec<u8>) -> RgbaTex {
    RgbaTex { width, height, rgba }
}

fn u8f(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn weights(x: f32, z: f32) -> [f32; 5] {
    course::blend(x, z)
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
    // Gobkit meshes stay on disk. The course forest is the photoreal kits.
    let mut mesh = CpuMesh::new();
    mesh.vertices.push(MeshVertex {
        pos: [0.0, -50.0, 0.0],
        nrm: [0.0, 1.0, 0.0],
        uv: [0.0, 0.0],
        panel: 70.0,
    });
    mesh.vertices.push(MeshVertex {
        pos: [0.02, -50.0, 0.0],
        nrm: [0.0, 1.0, 0.0],
        uv: [1.0, 0.0],
        panel: 70.0,
    });
    mesh.vertices.push(MeshVertex {
        pos: [0.0, -50.0, 0.02],
        nrm: [0.0, 1.0, 0.0],
        uv: [0.0, 1.0],
        panel: 70.0,
    });
    mesh.indices.extend([0, 1, 2]);
    mesh
}
