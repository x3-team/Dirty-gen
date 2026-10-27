//! glTF buggy. The cage, arms, coilovers and tires replace the procedural
//! body. Wheel centers still come from the solver; this module only poses
//! the drawn suspension onto those centers.
//!
//! Mount numbers match `tools/build_buggy_gltf.py`. `load_buggy` refuses a
//! file whose extras disagree.

use glam::Vec3;
use serde_json::Value;
use silt::vehicle::{hardpoints, Vehicle};

pub const LOWER_IN_X: f32 = 0.26;
pub const LOWER_IN_Y: f32 = -0.04;
pub const LOWER_HALF_Z: f32 = 0.18;
pub const LOWER_OUT_X: f32 = 0.55;
pub const LOWER_OUT_DY: f32 = -0.02;
pub const UPPER_IN_X: f32 = 0.24;
pub const UPPER_IN_Y: f32 = 0.34;
pub const UPPER_HALF_Z: f32 = 0.11;
pub const UPPER_OUT_X: f32 = 0.52;
pub const UPPER_OUT_DY: f32 = 0.20;
pub const TOWER_X: f32 = 0.30;
pub const TOWER_Y: f32 = 0.52;
pub const PERCH_DY: f32 = 0.12;

/// PBR vertex for the buggy only. The bike and the ground stay on `MeshVertex`.
/// `color.a` is roughness. `mud` is the static mask (1 = caked).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrVertex {
    pub pos: [f32; 3],
    pub emit: f32,
    pub nrm: [f32; 3],
    pub panel: f32,
    pub tangent: [f32; 4],
    pub color: [f32; 4],
    pub uv: [f32; 2],
    pub metal: f32,
    pub mud: f32,
}

pub struct PbrMesh {
    pub vertices: Vec<PbrVertex>,
    pub indices: Vec<u32>,
    pub length: f32,
    pub half_z: f32,
}

pub struct BuggyAsset {
    pub chassis: PbrMesh,
    pub wheel: PbrMesh,
    pub arm: PbrMesh,
    pub arm_upper: PbrMesh,
    pub coil: PbrMesh,
    pub upright: PbrMesh,
    pub normal_png: Vec<u8>,
}

pub fn load_buggy() -> BuggyAsset {
    let asset = parse_glb(include_bytes!("../assets/buggy.glb"));
    require_arm(&asset.arm, LOWER_HALF_Z);
    require_arm(&asset.arm_upper, UPPER_HALF_Z);
    if asset.coil.length < 0.5 || asset.upright.length < 0.5 {
        panic!("coil and upright are authored along a unit axis");
    }
    if !asset.normal_png.starts_with(b"\x89PNG") {
        panic!("buggy normal map is not a png");
    }
    asset
}

fn require_arm(mesh: &PbrMesh, half_z: f32) {
    if (mesh.half_z - half_z).abs() > 1.0e-4 {
        panic!("arm half_z {} != {half_z}", mesh.half_z);
    }
    if mesh.length < 0.05 {
        panic!("arm length {} is not a real A-arm", mesh.length);
    }
}

/// Arms and coilovers in body space, so the existing body matrix still draws them.
/// Each corner's panel id is that wheel's fender, which is where splash already lands.
pub fn pose_suspension(vehicle: &Vehicle, asset: &BuggyAsset) -> PbrMesh {
    let mut out = PbrMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        length: 0.0,
        half_z: 0.0,
    };
    let hp = hardpoints();
    for i in 0..4 {
        let sign = hp[i].x.signum();
        let z = hp[i].z;
        let wy = vehicle.wheel_center_local(i).y;
        let panel = 6.0 + i as f32;
        pose_arm(
            &mut out,
            &asset.arm,
            sign,
            z,
            wy,
            LOWER_IN_X,
            LOWER_IN_Y,
            LOWER_OUT_X,
            LOWER_OUT_DY,
            panel,
        );
        pose_arm(
            &mut out,
            &asset.arm_upper,
            sign,
            z,
            wy,
            UPPER_IN_X,
            UPPER_IN_Y,
            UPPER_OUT_X,
            UPPER_OUT_DY,
            panel,
        );
        let tower = Vec3::new(sign * TOWER_X, TOWER_Y, z);
        let perch = Vec3::new(sign * LOWER_OUT_X, wy + LOWER_OUT_DY + PERCH_DY, z);
        pose_span(&mut out, &asset.coil, tower, perch, panel);
        let lower = Vec3::new(sign * LOWER_OUT_X, wy + LOWER_OUT_DY, z);
        let upper = Vec3::new(sign * UPPER_OUT_X, wy + UPPER_OUT_DY, z);
        pose_span(&mut out, &asset.upright, lower, upper, panel);
    }
    out
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn coil_length(vehicle: &Vehicle, index: usize) -> f32 {
    let hp = hardpoints()[index];
    let sign = hp.x.signum();
    let wy = vehicle.wheel_center_local(index).y;
    let tower = Vec3::new(sign * TOWER_X, TOWER_Y, hp.z);
    let perch = Vec3::new(sign * LOWER_OUT_X, wy + LOWER_OUT_DY + PERCH_DY, hp.z);
    (perch - tower).length()
}

fn pose_arm(
    out: &mut PbrMesh,
    src: &PbrMesh,
    sign: f32,
    z: f32,
    wheel_y: f32,
    in_x: f32,
    in_y: f32,
    out_x: f32,
    out_dy: f32,
    panel: f32,
) {
    let mid = Vec3::new(sign * in_x, in_y, z);
    let knuckle = Vec3::new(sign * out_x, wheel_y + out_dy, z);
    let front = Vec3::new(sign * in_x, in_y, z + src.half_z);
    let rear = Vec3::new(sign * in_x, in_y, z - src.half_z);
    let delta = knuckle - mid;
    let len = delta.length().max(1.0e-4);
    let x_axis = delta / len;
    let z_hint = front - rear;
    let z_axis = (z_hint - x_axis * z_hint.dot(x_axis)).normalize_or_zero();
    let y_axis = z_axis.cross(x_axis);
    let scale = len / src.length.max(1.0e-4);
    transform_part(
        out,
        src,
        mid,
        x_axis,
        y_axis,
        z_axis,
        Vec3::new(scale, 1.0, 1.0),
        panel,
    );
}

fn pose_span(out: &mut PbrMesh, src: &PbrMesh, from: Vec3, to: Vec3, panel: f32) {
    let axis = to - from;
    let len = axis.length().max(1.0e-4);
    let dir = axis / len;
    let helper = if dir.y.abs() < 0.85 { Vec3::Y } else { Vec3::X };
    let x_axis = helper.cross(dir).normalize_or_zero();
    let z_axis = x_axis.cross(dir);
    let scale_y = len / src.length.max(1.0e-4);
    transform_part(
        out,
        src,
        from,
        x_axis,
        dir,
        z_axis,
        Vec3::new(1.0, scale_y, 1.0),
        panel,
    );
}

fn transform_part(
    out: &mut PbrMesh,
    src: &PbrMesh,
    origin: Vec3,
    x_axis: Vec3,
    y_axis: Vec3,
    z_axis: Vec3,
    scale: Vec3,
    panel: f32,
) {
    let base = out.vertices.len() as u32;
    for v in &src.vertices {
        let p = Vec3::from(v.pos) * scale;
        let pos = origin + x_axis * p.x + y_axis * p.y + z_axis * p.z;
        let n = Vec3::from(v.nrm);
        let ns = Vec3::new(
            n.x / scale.x.max(1.0e-4),
            n.y / scale.y.max(1.0e-4),
            n.z / scale.z.max(1.0e-4),
        );
        let nrm = (x_axis * ns.x + y_axis * ns.y + z_axis * ns.z).normalize_or_zero();
        let t = Vec3::new(v.tangent[0], v.tangent[1], v.tangent[2]) * scale;
        let tangent = (x_axis * t.x + y_axis * t.y + z_axis * t.z).normalize_or_zero();
        let mut nv = *v;
        nv.pos = pos.to_array();
        nv.nrm = nrm.to_array();
        nv.tangent = [tangent.x, tangent.y, tangent.z, v.tangent[3]];
        nv.panel = panel;
        out.vertices.push(nv);
    }
    for i in &src.indices {
        out.indices.push(base + *i);
    }
}

fn parse_glb(bytes: &[u8]) -> BuggyAsset {
    if bytes.len() < 20 || &bytes[0..4] != b"glTF" {
        panic!("assets/buggy.glb is not a glTF binary");
    }
    let mut offset = 12usize;
    let json_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 8;
    let json: Value = serde_json::from_slice(&bytes[offset..offset + json_len])
        .expect("buggy.glb json");
    offset += json_len;
    let bin_len = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    if &bytes[offset..offset + 4] != b"BIN\0" {
        panic!("buggy.glb is missing its BIN chunk");
    }
    offset += 4;
    let bin = &bytes[offset..offset + bin_len];
    check_mounts(&json);

    let mut found = std::collections::HashMap::new();
    for mesh in json["meshes"].as_array().expect("meshes") {
        let name = mesh["name"].as_str().expect("mesh name").to_string();
        found.insert(name, read_mesh(bin, &json, mesh));
    }
    let mut take = |name: &str| -> PbrMesh {
        found
            .remove(name)
            .unwrap_or_else(|| panic!("buggy.glb missing mesh {name}"))
    };
    let image = json["images"][0]["bufferView"].as_u64().expect("normal image") as usize;
    let normal_png = view_bytes(bin, &json, image).to_vec();
    BuggyAsset {
        chassis: take("chassis"),
        wheel: take("wheel"),
        arm: take("arm"),
        arm_upper: take("arm_upper"),
        coil: take("coil"),
        upright: take("upright"),
        normal_png,
    }
}

fn check_mounts(json: &Value) {
    let m = &json["asset"]["extras"]["silt"]["mounts"];
    let eq = |key: &str, want: f32| {
        let got = m[key]
            .as_f64()
            .unwrap_or_else(|| panic!("buggy.glb missing mount {key}")) as f32;
        if (got - want).abs() > 1.0e-4 {
            panic!("buggy.glb {key} is {got}, solver expects {want}");
        }
    };
    eq("lower_in_x", LOWER_IN_X);
    eq("lower_in_y", LOWER_IN_Y);
    eq("lower_half_z", LOWER_HALF_Z);
    eq("lower_out_x", LOWER_OUT_X);
    eq("lower_out_dy", LOWER_OUT_DY);
    eq("upper_in_x", UPPER_IN_X);
    eq("upper_in_y", UPPER_IN_Y);
    eq("upper_half_z", UPPER_HALF_Z);
    eq("upper_out_x", UPPER_OUT_X);
    eq("upper_out_dy", UPPER_OUT_DY);
    eq("tower_x", TOWER_X);
    eq("tower_y", TOWER_Y);
    eq("perch_dy", PERCH_DY);
    eq("wheel_x", 0.74);
    eq("front_z", 1.08);
    eq("rear_z", -1.18);
}

fn read_mesh(bin: &[u8], json: &Value, mesh: &Value) -> PbrMesh {
    let prim = &mesh["primitives"][0];
    let attrs = &prim["attributes"];
    let pos = f32s(bin, json, &attrs["POSITION"]);
    let nrm = f32s(bin, json, &attrs["NORMAL"]);
    let tan = f32s(bin, json, &attrs["TANGENT"]);
    let uv = f32s(bin, json, &attrs["TEXCOORD_0"]);
    let col = f32s(bin, json, &attrs["COLOR_0"]);
    let metal = f32s(bin, json, &attrs["_METAL"]);
    let mud = f32s(bin, json, &attrs["_MUD"]);
    let panel = f32s(bin, json, &attrs["_PANEL"]);
    let emit = f32s(bin, json, &attrs["_EMIT"]);
    let indices = u32s(bin, json, &prim["indices"]);
    let count = pos.len() / 3;
    if nrm.len() != count * 3
        || tan.len() != count * 4
        || uv.len() != count * 2
        || col.len() != count * 4
        || metal.len() != count
        || mud.len() != count
        || panel.len() != count
        || emit.len() != count
    {
        panic!("buggy mesh {} has mismatched attributes", mesh["name"]);
    }
    let mut vertices = Vec::with_capacity(count);
    for i in 0..count {
        vertices.push(PbrVertex {
            pos: [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]],
            emit: emit[i],
            nrm: [nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]],
            panel: panel[i],
            tangent: [tan[i * 4], tan[i * 4 + 1], tan[i * 4 + 2], tan[i * 4 + 3]],
            color: [col[i * 4], col[i * 4 + 1], col[i * 4 + 2], col[i * 4 + 3]],
            uv: [uv[i * 2], uv[i * 2 + 1]],
            metal: metal[i],
            mud: mud[i],
        });
    }
    let n = vertices.len() as u32;
    if indices.len() % 3 != 0 || indices.iter().any(|i| *i >= n) {
        panic!("buggy mesh {} has a bad index buffer", mesh["name"]);
    }
    let extras = &mesh["extras"];
    PbrMesh {
        vertices,
        indices,
        length: extras["length"].as_f64().unwrap_or(0.0) as f32,
        half_z: extras["half_z"].as_f64().unwrap_or(0.0) as f32,
    }
}

fn f32s(bin: &[u8], json: &Value, id: &Value) -> Vec<f32> {
    accessor_bytes(bin, json, id)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn u32s(bin: &[u8], json: &Value, id: &Value) -> Vec<u32> {
    accessor_bytes(bin, json, id)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn accessor_bytes<'a>(bin: &'a [u8], json: &Value, id: &Value) -> &'a [u8] {
    let acc = &json["accessors"][id.as_u64().expect("accessor id") as usize];
    let count = acc["count"].as_u64().unwrap() as usize;
    let comps = match acc["type"].as_str().unwrap() {
        "SCALAR" => 1,
        "VEC2" => 2,
        "VEC3" => 3,
        "VEC4" => 4,
        other => panic!("unexpected gltf type {other}"),
    };
    let comp_ty = acc["componentType"].as_u64().unwrap();
    if comp_ty != 5126 && comp_ty != 5125 {
        panic!("buggy.glb uses an unexpected component type {comp_ty}");
    }
    let view = acc["bufferView"].as_u64().unwrap() as usize;
    let extra = acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let raw = view_bytes(bin, json, view);
    let nbytes = count * comps * 4;
    &raw[extra..extra + nbytes]
}

fn view_bytes<'a>(bin: &'a [u8], json: &Value, index: usize) -> &'a [u8] {
    let view = &json["bufferViews"][index];
    let start = view["byteOffset"].as_u64().unwrap_or(0) as usize;
    let len = view["byteLength"].as_u64().unwrap() as usize;
    &bin[start..start + len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_layout_matches_the_pbr_shader() {
        assert_eq!(std::mem::size_of::<PbrVertex>(), 80);
        assert_eq!(std::mem::offset_of!(PbrVertex, pos), 0);
        assert_eq!(std::mem::offset_of!(PbrVertex, emit), 12);
        assert_eq!(std::mem::offset_of!(PbrVertex, nrm), 16);
        assert_eq!(std::mem::offset_of!(PbrVertex, panel), 28);
        assert_eq!(std::mem::offset_of!(PbrVertex, tangent), 32);
        assert_eq!(std::mem::offset_of!(PbrVertex, color), 48);
        assert_eq!(std::mem::offset_of!(PbrVertex, uv), 64);
        assert_eq!(std::mem::offset_of!(PbrVertex, metal), 72);
        assert_eq!(std::mem::offset_of!(PbrVertex, mud), 76);
    }

    #[test]
    fn gltf_buggy_has_cage_wheels_springs_and_a_mud_mask() {
        let asset = load_buggy();
        assert!(asset.chassis.vertices.len() > 1000, "cage is still a few rods");
        assert!(asset.wheel.vertices.len() > 400);
        assert!(asset.arm.indices.len() > 30);
        assert!(asset.coil.vertices.len() > 100, "coilover missing");
        assert!(asset.upright.vertices.len() > 10);
        let radius = asset
            .wheel
            .vertices
            .iter()
            .map(|v| (v.pos[1] * v.pos[1] + v.pos[2] * v.pos[2]).sqrt())
            .fold(0.0f32, f32::max);
        assert!(
            (radius - silt::vehicle::WHEEL_RADIUS).abs() < 0.02,
            "tire radius {radius}"
        );
        let mut low = 0.0;
        let mut low_n = 0.0;
        let mut high = 0.0;
        let mut high_n = 0.0;
        for v in &asset.chassis.vertices {
            assert!(v.pos.iter().all(|c| c.is_finite()));
            assert!(v.nrm.iter().all(|c| c.is_finite()));
            if v.pos[1] < 0.0 {
                low += v.mud;
                low_n += 1.0;
            }
            if v.pos[1] > 0.72 {
                high += v.mud;
                high_n += 1.0;
            }
        }
        assert!(low_n > 10.0 && high_n > 10.0);
        assert!(
            low / low_n > high / high_n + 0.25,
            "lower chassis mud {} vs upper {}",
            low / low_n,
            high / high_n
        );
        let orange = asset.chassis.vertices.iter().any(|v| {
            v.color[0] > 0.55 && v.color[1] < 0.45 && v.color[2] < 0.2 && v.mud < 0.25
        });
        assert!(orange, "clean orange cage tubes missing");
        let chrome = asset.coil.vertices.iter().any(|v| v.metal > 0.8 && v.mud < 0.25);
        assert!(chrome, "coil damper is not metal");
        for (i, hp) in hardpoints().iter().enumerate() {
            let target = Vec3::new(hp.x.signum() * TOWER_X, TOWER_Y, hp.z);
            let best = asset
                .chassis
                .vertices
                .iter()
                .map(|v| (Vec3::from(v.pos) - target).length())
                .fold(1.0e9, f32::min);
            assert!(best < 1.0e-3, "tower {i} is {best} off the cage");
        }
    }

    #[test]
    fn coil_shortens_when_the_wheel_compresses() {
        let asset = load_buggy();
        let rest = Vehicle::new();
        let mut bumped = rest.clone();
        bumped.wheels[0].compression = 0.28;
        let posed_rest = pose_suspension(&rest, &asset);
        let posed_bump = pose_suspension(&bumped, &asset);
        assert_eq!(posed_rest.vertices.len(), posed_bump.vertices.len());
        assert_eq!(posed_rest.indices.len() % 3, 0);
        let moved = posed_rest
            .vertices
            .iter()
            .zip(&posed_bump.vertices)
            .any(|(a, b)| (Vec3::from(a.pos) - Vec3::from(b.pos)).length() > 0.02);
        assert!(moved, "suspension ignored wheel travel");
        let slack = coil_length(&rest, 0);
        let tight = coil_length(&bumped, 0);
        assert!(tight < slack - 0.08, "coil {slack} -> {tight}");
        // Rear wheels were not touched, so that coil stays put.
        assert!((coil_length(&rest, 2) - coil_length(&bumped, 2)).abs() < 1.0e-4);
    }
}
