//! Camera-centred level of detail for the drawn ground (CDLOD).
//!
//! The ground used to be one uniform grid, eight vertices per 20 cm soil
//! cell over the whole 40 m square: 2.5 million vertices, drawn three times
//! a frame (shadow, ground, water), and still no finer than 2.6 cm under the
//! camera. This module picks square nodes from a quadtree instead. A node is
//! one shared 32 × 32 quad patch, scaled to its level. Near the camera the
//! vertex step is 1/64 m (1.56 cm), and it doubles with each distance ring.
//!
//! Nothing here shapes the ground. The vertex shader still reads the solver
//! height at every vertex; this only decides where the vertices are.
//!
//! Selection follows Strugar, "Continuous Distance-Dependent Level of Detail
//! for Rendering Heightmaps" (2009). Every size is a power of two in metres,
//! so a vertex shared by two nodes lands on the same float in both and the
//! seams stay closed. In the outer part of each ring the shader morphs the
//! odd vertices onto the next coarser grid, so a level change is not a step.

use glam::{Mat4, Vec3, Vec4};

/// Quads along one side of a node patch.
pub const PATCH_QUADS: u32 = 32;
/// Vertex step of the finest level, metres. 1/64 m, exact in binary.
pub const FINEST_STEP: f32 = 1.0 / 64.0;
/// Levels in the tree. Level 0 is the finest. The top node is
/// `PATCH_QUADS * FINEST_STEP * 2^(LEVELS - 1)` = 64 m on a side.
pub const LEVELS: u32 = 8;
/// Corner of the top node. The 40 m soil square sits inside it.
pub const ROOT_ORIGIN: f32 = -32.0;
/// Default ring radius in vertex steps. A level-k vertex step is drawn out
/// to `RANGE_STEPS` steps from the camera: 1.56 cm to 4 m, 3.1 cm to 8 m,
/// 6.3 cm to 16 m, 12.5 cm to 32 m. The old grid was 2.6 cm everywhere.
pub const RANGE_STEPS: f32 = 256.0;
/// Where the morph to the coarser grid starts, as a share of the ring.
pub const MORPH_START: f32 = 0.7;

/// Vertex step of a level, metres.
pub fn level_step(level: u32) -> f32 {
    FINEST_STEP * (1u32 << level) as f32
}

/// Side of a node at a level, metres.
pub fn level_size(level: u32) -> f32 {
    level_step(level) * PATCH_QUADS as f32
}

/// One drawn patch. Packed as the per-instance vertex attribute.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LodNode {
    /// Minimum corner, world X and Z.
    pub origin: [f32; 2],
    /// Side of the drawn square, metres. Half the level size when the node
    /// is the quarter of a parent whose child was out of range.
    pub size: f32,
    /// Level whose vertex step and morph ring this patch uses.
    pub level: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct LodParams {
    /// Ring radius in vertex steps; see [`RANGE_STEPS`].
    pub range_steps: f32,
    /// Height the distance is measured at. The shader uses the same value,
    /// so the morph and the selection agree exactly.
    pub y_ref: f32,
    /// Drawn ground square, world X and Z, min and max.
    pub extent_min: [f32; 2],
    pub extent_max: [f32; 2],
    /// Lowest and highest ground, for the frustum test.
    pub y_min: f32,
    pub y_max: f32,
}

impl LodParams {
    pub fn for_square(half: f32, y_min: f32, y_max: f32) -> Self {
        Self {
            range_steps: RANGE_STEPS,
            y_ref: 0.0,
            extent_min: [-half, -half],
            extent_max: [half, half],
            y_min,
            y_max,
        }
    }

    /// Ring radius of a level, metres.
    pub fn range(&self, level: u32) -> f32 {
        level_step(level) * self.range_steps
    }

    /// Morph start and end of a level, metres. The shader reads these.
    pub fn morph(&self, level: u32) -> (f32, f32) {
        let end = self.range(level);
        let prev = if level == 0 { 0.0 } else { self.range(level - 1) };
        (prev + (end - prev) * MORPH_START, end)
    }
}

/// Six planes of a view-projection, for the node cull. wgpu clip space:
/// x, y in -w..w, z in 0..w.
pub fn frustum_planes(view_proj: &Mat4) -> [Vec4; 6] {
    let r0 = view_proj.row(0);
    let r1 = view_proj.row(1);
    let r2 = view_proj.row(2);
    let r3 = view_proj.row(3);
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
}

fn box_outside(planes: &[Vec4; 6], lo: Vec3, hi: Vec3) -> bool {
    for p in planes {
        // The corner furthest along the plane normal.
        let v = Vec3::new(
            if p.x >= 0.0 { hi.x } else { lo.x },
            if p.y >= 0.0 { hi.y } else { lo.y },
            if p.z >= 0.0 { hi.z } else { lo.z },
        );
        if p.x * v.x + p.y * v.y + p.z * v.z + p.w < 0.0 {
            return true;
        }
    }
    false
}

struct Select<'a> {
    cam: Vec3,
    params: &'a LodParams,
    planes: Option<[Vec4; 6]>,
    out: Vec<LodNode>,
}

impl Select<'_> {
    /// Squared distance from the camera to a node's square at `y_ref`.
    fn dist2(&self, x0: f32, z0: f32, size: f32) -> f32 {
        let dx = (x0 - self.cam.x).max(0.0).max(self.cam.x - (x0 + size));
        let dz = (z0 - self.cam.z).max(0.0).max(self.cam.z - (z0 + size));
        let dy = self.cam.y - self.params.y_ref;
        dx * dx + dy * dy + dz * dz
    }

    fn outside_extent(&self, x0: f32, z0: f32, size: f32) -> bool {
        let p = self.params;
        x0 >= p.extent_max[0]
            || z0 >= p.extent_max[1]
            || x0 + size <= p.extent_min[0]
            || z0 + size <= p.extent_min[1]
    }

    fn culled(&self, x0: f32, z0: f32, size: f32) -> bool {
        match &self.planes {
            None => false,
            Some(planes) => box_outside(
                planes,
                Vec3::new(x0, self.params.y_min, z0),
                Vec3::new(x0 + size, self.params.y_max, z0 + size),
            ),
        }
    }

    fn push(&mut self, x0: f32, z0: f32, size: f32, level: u32) {
        if self.outside_extent(x0, z0, size) || self.culled(x0, z0, size) {
            return;
        }
        self.out.push(LodNode {
            origin: [x0, z0],
            size,
            level: level as f32,
        });
    }

    /// True when this node's area is handled (drawn, culled, or off the
    /// ground). False hands it back to the parent.
    fn node(&mut self, x0: f32, z0: f32, level: u32) -> bool {
        let size = level_size(level);
        if self.outside_extent(x0, z0, size) {
            return true;
        }
        let d2 = self.dist2(x0, z0, size);
        let top = level == LEVELS - 1;
        let r = self.params.range(level);
        if !top && d2 > r * r {
            return false;
        }
        if self.culled(x0, z0, size) {
            return true;
        }
        if level == 0 {
            self.push(x0, z0, size, 0);
            return true;
        }
        let rc = self.params.range(level - 1);
        if d2 > rc * rc {
            self.push(x0, z0, size, level);
            return true;
        }
        let half = size * 0.5;
        for (cx, cz) in [(x0, z0), (x0 + half, z0), (x0, z0 + half), (x0 + half, z0 + half)] {
            if !self.node(cx, cz, level - 1) {
                // That quarter is drawn by this level.
                self.push(cx, cz, half, level);
            }
        }
        true
    }
}

/// Pick the patches for one view. `view_proj` culls nodes outside the
/// camera; pass `None` for a pass that must see the whole ground, such as
/// the shadow map.
pub fn select(cam: Vec3, params: &LodParams, view_proj: Option<&Mat4>) -> Vec<LodNode> {
    let mut s = Select {
        cam,
        params,
        planes: view_proj.map(frustum_planes),
        out: Vec::with_capacity(512),
    };
    s.node(ROOT_ORIGIN, ROOT_ORIGIN, LEVELS - 1);
    s.out
}

/// The shared patch: grid coordinates 0..=PATCH_QUADS, and its triangles.
pub fn patch_mesh() -> (Vec<[f32; 2]>, Vec<u32>) {
    let n = PATCH_QUADS + 1;
    let mut verts = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            verts.push([i as f32, j as f32]);
        }
    }
    let mut idx = Vec::with_capacity((PATCH_QUADS * PATCH_QUADS * 6) as usize);
    for j in 0..PATCH_QUADS {
        for i in 0..PATCH_QUADS {
            let a = j * n + i;
            let b = (j + 1) * n + i;
            let c = (j + 1) * n + i + 1;
            let d = j * n + i + 1;
            // Same winding as the old grid: +Z then +X faces up.
            idx.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    (verts, idx)
}

/// CPU twin of the vertex shader placement (`tg_lod_xz` in
/// `terrain_geom.wgsl`). Used by the tests to prove the seams close.
pub fn vertex_xz(g: [f32; 2], node: &LodNode, params: &LodParams, cam: Vec3) -> [f32; 2] {
    let level = node.level as u32;
    let step = level_step(level);
    let h = node.size / PATCH_QUADS as f32;
    let mut out = [0.0f32; 2];
    let mut gg = [0.0f32; 2];
    for a in 0..2 {
        // Global grid index on this level. A quarter patch has half steps;
        // those collapse onto the level grid.
        let gl = (node.origin[a] - ROOT_ORIGIN) / step + g[a] * (h / step);
        gg[a] = gl.floor();
    }
    let px = ROOT_ORIGIN + gg[0] * step;
    let pz = ROOT_ORIGIN + gg[1] * step;
    let dx = px - cam.x;
    let dy = params.y_ref - cam.y;
    let dz = pz - cam.z;
    let d = (dx * dx + dy * dy + dz * dz).sqrt();
    let (m0, m1) = params.morph(level);
    let k = ((d - m0) / (m1 - m0)).clamp(0.0, 1.0);
    for a in 0..2 {
        let odd = (gg[a] * 0.5).fract() * 2.0;
        let gm = gg[a] - odd * k;
        let p = ROOT_ORIGIN + gm * step;
        out[a] = p.clamp(params.extent_min[a], params.extent_max[a]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> LodParams {
        LodParams::for_square(20.0, -1.0, 3.0)
    }

    fn clip(n: &LodNode, p: &LodParams) -> (f32, f32, f32, f32) {
        let x0 = n.origin[0].max(p.extent_min[0]);
        let z0 = n.origin[1].max(p.extent_min[1]);
        let x1 = (n.origin[0] + n.size).min(p.extent_max[0]);
        let z1 = (n.origin[1] + n.size).min(p.extent_max[1]);
        (x0, z0, x1, z1)
    }

    #[test]
    fn nodes_tile_the_ground_once() {
        let p = params();
        for cam in [
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(-6.5, 1.2, 5.0),
            Vec3::new(19.0, 0.3, -19.5),
            Vec3::new(40.0, 10.0, 40.0),
        ] {
            let nodes = select(cam, &p, None);
            let mut area = 0.0f64;
            for n in &nodes {
                let (x0, z0, x1, z1) = clip(n, &p);
                area += ((x1 - x0) * (z1 - z0)) as f64;
            }
            assert!((area - 1600.0).abs() < 1e-3, "area {area} at {cam:?}");
            for (i, a) in nodes.iter().enumerate() {
                for b in nodes.iter().skip(i + 1) {
                    let ox = (a.origin[0] + a.size).min(b.origin[0] + b.size)
                        - a.origin[0].max(b.origin[0]);
                    let oz = (a.origin[1] + a.size).min(b.origin[1] + b.size)
                        - a.origin[1].max(b.origin[1]);
                    assert!(ox <= 0.0 || oz <= 0.0, "overlap {a:?} {b:?}");
                }
            }
        }
    }

    #[test]
    fn finest_step_under_the_camera() {
        let p = params();
        let cam = Vec3::new(-6.5, 1.2, 5.0);
        let nodes = select(cam, &p, None);
        let under = nodes
            .iter()
            .find(|n| {
                cam.x >= n.origin[0]
                    && cam.x < n.origin[0] + n.size
                    && cam.z >= n.origin[1]
                    && cam.z < n.origin[1] + n.size
            })
            .expect("node under camera");
        assert_eq!(under.level, 0.0);
        // The old grid drew 2.5 M vertices. The tree draws far fewer.
        let verts = nodes.len() * 33 * 33;
        assert!(verts < 1_200_000, "{} nodes, {verts} vertices", nodes.len());
    }

    /// Two nodes that touch put every vertex of the shared edge on the same
    /// point, including after the morph. That is what keeps the seams shut.
    #[test]
    fn shared_edges_close() {
        let p = params();
        for cam in [Vec3::new(0.3, 1.4, -0.7), Vec3::new(-6.5, 1.2, 5.0)] {
            let nodes = select(cam, &p, None);
            let mut cross = 0usize;
            // Every edge vertex of every node, after placement.
            let mut pts: Vec<[f32; 2]> = Vec::new();
            for n in &nodes {
                for t in 0..=PATCH_QUADS {
                    let t = t as f32;
                    for g in [[t, 0.0], [t, 32.0], [0.0, t], [32.0, t]] {
                        pts.push(vertex_xz(g, n, &p, cam));
                    }
                }
            }
            // An edge of one node lies on the edge of its neighbour. A
            // vertex on a neighbour's edge must coincide with one of the
            // neighbour's placed edge points, or lie on a segment between
            // two of them (a coarser neighbour's straight edge).
            for a in &nodes {
                let level = a.level as u32;
                for b in &nodes {
                    if std::ptr::eq(a, b) || (b.level - a.level).abs() > 1.0 {
                        continue;
                    }
                    let touch_x = (a.origin[0] + a.size - b.origin[0]).abs() < 1e-6;
                    if !touch_x {
                        continue;
                    }
                    let z0 = a.origin[1].max(b.origin[1]);
                    let z1 = (a.origin[1] + a.size).min(b.origin[1] + b.size);
                    if z1 <= z0 {
                        continue;
                    }
                    // Walk a's right edge and b's left edge on the overlap.
                    let ha = a.size / 32.0;
                    let hb = b.size / 32.0;
                    let mut ea: Vec<[f32; 2]> = (0..=32)
                        .map(|t| [32.0, t as f32])
                        .filter(|g| {
                            let z = a.origin[1] + g[1] * ha;
                            z >= z0 - 1e-6 && z <= z1 + 1e-6
                        })
                        .map(|g| vertex_xz(g, a, &p, cam))
                        .collect();
                    let mut eb: Vec<[f32; 2]> = (0..=32)
                        .map(|t| [0.0, t as f32])
                        .filter(|g| {
                            let z = b.origin[1] + g[1] * hb;
                            z >= z0 - 1e-6 && z <= z1 + 1e-6
                        })
                        .map(|g| vertex_xz(g, b, &p, cam))
                        .collect();
                    ea.sort_by(|u, v| u[1].total_cmp(&v[1]));
                    eb.sort_by(|u, v| u[1].total_cmp(&v[1]));
                    ea.dedup();
                    eb.dedup();
                    if a.level != b.level {
                        cross += 1;
                    }
                    // Same set of distinct points on both sides.
                    assert_eq!(
                        ea, eb,
                        "seam a={a:?} b={b:?} level {level} cam {cam:?}"
                    );
                }
            }
            assert!(!pts.is_empty());
            assert!(cross > 4, "only {cross} seams between levels were checked");
        }
    }

    #[test]
    fn cull_drops_nodes_behind_the_camera() {
        let p = params();
        let eye = Vec3::new(0.0, 1.5, 0.0);
        let view = Mat4::look_at_rh(eye, eye + Vec3::new(0.0, -0.3, 1.0), Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.05, 200.0);
        let vp = proj * view;
        let all = select(eye, &p, None);
        let seen = select(eye, &p, Some(&vp));
        assert!(seen.len() < all.len());
        // Nothing in front of the camera is lost.
        for n in &all {
            let cz = n.origin[1] + n.size * 0.5;
            let cx = n.origin[0] + n.size * 0.5;
            if cz > 2.0 && cx.abs() < 0.2 * cz {
                assert!(seen.contains(n), "lost {n:?}");
            }
        }
    }
}
