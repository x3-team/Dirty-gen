//! Game-only fuel show for the buggy.
//!
//! A hard flip or a heavy landing can open a hole. Liquid runs out as a ribbon
//! on the heightfield, a spark runs into that ribbon, the burn follows the
//! liquid, and a short ground blast throws the mud chunks already in the world.
//! The ribbon is simulated. It is not a played clip, and it is not a real fire.

use glam::Vec3;

use crate::particles::Particles;
use crate::soil::SoilField;
use crate::vehicle::Vehicle;

const NODE_GAP: f32 = 0.42;
const SHEET: f32 = 0.08;
const MAX_NODES: usize = 16;
const POUR: f32 = 1.15;
const SPARK: f32 = 0.34;
const BURN_MIN: f32 = 1.9;
const BURN_MAX: f32 = 2.4;
const BLAST: f32 = 0.52;
const ASH: f32 = 1.1;
const SEEP: f32 = 1.45;

/// Same layout as the renderer's particle vertex, so the show can reuse that buffer.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FuelVert {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    pub uv: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Idle,
    Pour,
    Spark,
    Burn,
    Blast,
    Ash,
    Done,
}

#[derive(Clone, Copy)]
struct Node {
    pos: Vec3,
    liquid: f32,
    heat: f32,
    half_l: f32,
    half_r: f32,
}

#[derive(Clone, Copy)]
struct Ripple {
    pos: Vec3,
    radius: f32,
    life: f32,
    max_life: f32,
}

#[derive(Clone, Copy)]
struct Bead {
    pos: Vec3,
    vel: Vec3,
    life: f32,
    max_life: f32,
}

#[derive(Clone, Copy)]
struct Puff {
    pos: Vec3,
    vel: Vec3,
    life: f32,
    max_life: f32,
    size: f32,
    /// 0 is the gray plume over the flame. 1 is the black blast smoke.
    dark: f32,
}

#[derive(Clone, Copy)]
struct Streak {
    pos: Vec3,
    vel: Vec3,
    life: f32,
    max_life: f32,
}

pub struct FuelShow {
    nodes: Vec<Node>,
    ripples: Vec<Ripple>,
    beads: Vec<Bead>,
    puffs: Vec<Puff>,
    streaks: Vec<Streak>,
    phase: Phase,
    phase_t: f32,
    ruptured: bool,
    threw: bool,
    prev_impact: f32,
    prev_vy: f32,
    rng: u32,
    leak: Vec3,
    hit: Vec3,
    spark_from: Vec3,
    spark_to: Vec3,
    spark_pos: Vec3,
    spark_node: usize,
    flow: Vec3,
    side: Vec3,
    blast_at: Vec3,
    pool_r: [f32; 6],
    seep: f32,
    /// Seconds since the show was created. Ripple displacement uses the same
    /// clock the frame uniform feeds the shader.
    clock: f32,
    acc_ripple: f32,
    acc_bead: f32,
    acc_puff: f32,
}

impl FuelShow {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            ripples: Vec::new(),
            beads: Vec::new(),
            puffs: Vec::new(),
            streaks: Vec::new(),
            phase: Phase::Idle,
            phase_t: 0.0,
            ruptured: false,
            threw: false,
            prev_impact: -1.0,
            prev_vy: 0.0,
            rng: 0x00C0_FFEE,
            leak: Vec3::ZERO,
            hit: Vec3::ZERO,
            spark_from: Vec3::ZERO,
            spark_to: Vec3::ZERO,
            spark_pos: Vec3::ZERO,
            spark_node: 0,
            flow: Vec3::new(0.0, 0.0, -1.0),
            side: Vec3::X,
            blast_at: Vec3::ZERO,
            pool_r: [0.8; 6],
            seep: 0.0,
            clock: 0.0,
            acc_ripple: 0.0,
            acc_bead: 0.0,
            acc_puff: 0.0,
        }
    }

    pub fn is_live(&self) -> bool {
        !matches!(self.phase, Phase::Idle | Phase::Done)
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn threw_chunks(&self) -> bool {
        self.threw
    }

    pub fn label(&self) -> &'static str {
        match self.phase {
            Phase::Idle => "idle",
            Phase::Pour => "pour",
            Phase::Spark => "spark",
            Phase::Burn => "burn",
            Phase::Blast => "blast",
            Phase::Ash => "ash",
            Phase::Done => "done",
        }
    }

    pub fn tick(
        &mut self,
        dt: f32,
        vehicle: &Vehicle,
        soil: &SoilField,
        particles: &mut Particles,
        impact_time: f32,
    ) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        let dt = dt.min(1.0 / 20.0);
        self.clock += dt;
        if self.phase == Phase::Idle && self.armed(vehicle, impact_time) {
            self.begin(vehicle);
        }
        match self.phase {
            Phase::Pour => self.pour(dt, vehicle, soil, particles),
            Phase::Spark => self.spark_phase(dt, soil, particles),
            Phase::Burn => self.burn(dt, soil, particles),
            Phase::Blast => self.blast(dt, soil),
            Phase::Ash => self.ash(dt, soil),
            Phase::Idle | Phase::Done => {}
        }
        self.prev_vy = vehicle.vel.y;
        self.prev_impact = impact_time;
    }

    fn armed(&self, vehicle: &Vehicle, impact_time: f32) -> bool {
        if self.ruptured {
            return false;
        }
        let up = vehicle.orientation() * Vec3::Y;
        let flipped = vehicle.pitch.abs() > 1.15 || vehicle.roll.abs() > 0.95 || up.y < 0.2;
        let violent = vehicle.pitch_rate.abs() > 2.0
            || vehicle.roll_rate.abs() > 2.0
            || vehicle.speed() > 3.0
            || vehicle.vel.length() > 5.0;
        let hard_flip = flipped && violent;
        let fresh_hit = impact_time > self.prev_impact + 1.0e-4 && impact_time > 0.0;
        let slam =
            self.prev_vy < -3.2 && vehicle.vel.y > self.prev_vy + 1.4 && vehicle.speed() > 2.0;
        hard_flip || fresh_hit || slam
    }

    fn begin(&mut self, vehicle: &Vehicle) {
        self.ruptured = true;
        self.phase = Phase::Pour;
        self.phase_t = 0.0;
        self.threw = false;
        self.rng = 0x00C0_FFEE;
        self.nodes.clear();
        self.ripples.clear();
        self.beads.clear();
        self.puffs.clear();
        self.streaks.clear();
        self.seep = 0.0;
        self.acc_ripple = 0.0;
        self.acc_bead = 0.0;
        self.acc_puff = 0.0;
        let back = vehicle.orientation() * Vec3::new(0.0, 0.0, -1.0);
        self.side = flat_norm(vehicle.orientation() * Vec3::X);
        // Mostly off to the shoulder, so the sheet is not hidden in the rut.
        self.flow = flat_norm(
            self.side
                + Vec3::new(back.x, 0.0, back.z) * 0.45
                + Vec3::new(self.rnd() - 0.5, 0.0, self.rnd() - 0.5) * 0.35,
        );
        for i in 0..self.pool_r.len() {
            let lobe = 0.95 + self.rnd() * 0.65;
            self.pool_r[i] = lobe;
        }
    }

    fn rnd(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (self.rng >> 8) as f32 / 16_777_216.0
    }

    fn pour(&mut self, dt: f32, vehicle: &Vehicle, soil: &SoilField, particles: &mut Particles) {
        self.phase_t += dt;
        self.leak = leak_of(vehicle);
        let mouth = self.leak + self.side * 3.4;
        let under = on_soil(soil, mouth);
        let dir = flow_dir(soil, under, vehicle, self.flow);
        let landing = on_soil(soil, under + dir * 0.4);
        if self.nodes.is_empty() {
            self.push_node(landing);
        }
        let last = self.nodes.last().map(|n| n.pos).unwrap_or(landing);
        let gap = flat_len(landing - last);
        if gap > NODE_GAP {
            let next = on_soil(soil, last + flat_norm(landing - last) * NODE_GAP);
            self.push_node(next);
        } else {
            self.seep += dt * SEEP;
            if self.seep >= NODE_GAP {
                self.seep = 0.0;
                let tip = self.nodes.last().map(|n| n.pos).unwrap_or(landing);
                self.push_node(on_soil(soil, tip + dir * NODE_GAP));
            }
        }
        self.hit = landing;
        self.stick(soil);
        self.spawn_pour(dt);
        self.drift(dt, soil);
        self.hold_chunks(particles);
        let poured = self.phase_t > POUR && self.nodes.len() >= 3;
        let gave_up = self.phase_t > POUR + 0.6;
        if poured || gave_up {
            self.enter_spark();
        }
    }

    fn enter_spark(&mut self) {
        if self.nodes.is_empty() {
            self.push_node(self.hit);
        }
        let mut best_i = 0;
        let mut best_d = f32::MAX;
        for (i, node) in self.nodes.iter().enumerate() {
            let d = flat_len(node.pos - self.hit);
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        self.spark_node = best_i;
        self.spark_from = self.stream_top();
        self.spark_to = self.nodes[best_i].pos + Vec3::Y * 0.16;
        self.spark_pos = self.spark_from;
        self.phase = Phase::Spark;
        self.phase_t = 0.0;
    }

    fn spark_phase(&mut self, dt: f32, soil: &SoilField, particles: &mut Particles) {
        self.phase_t += dt;
        let u = (self.phase_t / SPARK).clamp(0.0, 1.0);
        let s = u * u * (3.0 - 2.0 * u);
        self.spark_pos = self.spark_from.lerp(self.spark_to, s);
        self.stick(soil);
        self.drift(dt, soil);
        self.hold_chunks(particles);
        if self.phase_t >= SPARK {
            if let Some(node) = self.nodes.get_mut(self.spark_node) {
                node.heat = 1.0;
            }
            self.phase = Phase::Burn;
            self.phase_t = 0.0;
        }
    }

    fn burn(&mut self, dt: f32, soil: &SoilField, particles: &mut Particles) {
        self.phase_t += dt;
        self.spread(dt, 8.5);
        self.consume(dt);
        self.stick(soil);
        self.puff_over_flame(dt);
        self.drift(dt, soil);
        self.hold_chunks(particles);
        let hot = self
            .nodes
            .iter()
            .filter(|n| n.heat > 0.62 && n.liquid > 0.18)
            .count();
        let cover = hot as f32 / self.nodes.len().max(1) as f32;
        if (cover > 0.7 && self.phase_t > BURN_MIN) || self.phase_t > BURN_MAX {
            self.enter_blast(particles);
        }
    }

    fn enter_blast(&mut self, particles: &mut Particles) {
        self.blast_at = self.hottest();
        self.phase = Phase::Blast;
        self.phase_t = 0.0;
        self.throw_chunks(particles);
        self.threw = true;
        self.spawn_blast();
    }

    fn blast(&mut self, dt: f32, soil: &SoilField) {
        self.phase_t += dt;
        self.spread(dt, 4.0);
        self.drift(dt, soil);
        if self.phase_t > BLAST {
            self.phase = Phase::Ash;
            self.phase_t = 0.0;
        }
    }

    fn ash(&mut self, dt: f32, soil: &SoilField) {
        self.phase_t += dt;
        for node in &mut self.nodes {
            node.heat = (node.heat - dt * 1.15).max(0.0);
            node.liquid = (node.liquid - dt * 0.8).max(0.0);
        }
        self.drift(dt, soil);
        let smoke = self.puffs.iter().any(|p| p.life > 0.08);
        if (self.phase_t > ASH && !smoke) || self.phase_t > ASH + 0.85 {
            self.finish();
        }
    }

    fn finish(&mut self) {
        self.phase = Phase::Done;
        self.nodes.clear();
        self.ripples.clear();
        self.beads.clear();
        self.puffs.clear();
        self.streaks.clear();
    }

    fn push_node(&mut self, p: Vec3) {
        if let Some(last) = self.nodes.last() {
            if flat_len(p - last.pos) < NODE_GAP * 0.72 {
                return;
            }
        }
        let half_l = 0.85 + self.rnd() * 0.7;
        let half_r = 0.78 + self.rnd() * 0.72;
        self.nodes.push(Node {
            pos: p,
            liquid: 1.0,
            heat: 0.0,
            half_l,
            half_r,
        });
        if self.phase == Phase::Pour && self.nodes.len() > MAX_NODES {
            self.nodes.remove(0);
        }
    }

    fn stick(&mut self, soil: &SoilField) {
        for node in &mut self.nodes {
            node.pos.y = soil.sample(node.pos.x, node.pos.z).height + SHEET;
        }
        self.hit.y = soil.sample(self.hit.x, self.hit.z).height + SHEET;
    }

    /// Heat walks only along nodes that still hold liquid.
    fn spread(&mut self, dt: f32, rate: f32) {
        let prev: Vec<f32> = self.nodes.iter().map(|n| n.heat).collect();
        let k = 1.0 - (-dt * rate).exp();
        for i in 0..self.nodes.len() {
            if self.nodes[i].liquid < 0.15 {
                self.nodes[i].heat *= (1.0 - dt * 2.0).max(0.0);
                continue;
            }
            let mut inbound = prev[i];
            if i > 0 && self.nodes[i - 1].liquid >= 0.15 {
                inbound = inbound.max(prev[i - 1]);
            }
            if i + 1 < prev.len() && self.nodes[i + 1].liquid >= 0.15 {
                inbound = inbound.max(prev[i + 1]);
            }
            if inbound > self.nodes[i].heat {
                self.nodes[i].heat += (inbound - self.nodes[i].heat) * k;
            }
        }
    }

    fn consume(&mut self, dt: f32) {
        for node in &mut self.nodes {
            if node.heat > 0.55 {
                // The wet patch stays under the flame. It only thins a little.
                node.liquid = (node.liquid - dt * 0.035).max(0.4);
            }
        }
    }

    fn spawn_pour(&mut self, dt: f32) {
        self.acc_ripple += dt;
        if self.acc_ripple > 0.12 {
            self.acc_ripple = 0.0;
            self.ripples.push(Ripple {
                pos: self.hit,
                radius: 0.16,
                life: 0.8,
                max_life: 0.8,
            });
            if self.ripples.len() > 8 {
                self.ripples.remove(0);
            }
        }
        self.acc_bead += dt;
        if self.acc_bead > 0.05 {
            self.acc_bead = 0.0;
            for _ in 0..2 {
                let ang = self.rnd() * std::f32::consts::TAU;
                let sp = 0.7 + self.rnd() * 1.5;
                let up = 1.0 + self.rnd() * 1.2;
                self.beads.push(Bead {
                    pos: self.hit + Vec3::Y * 0.08,
                    vel: Vec3::new(ang.cos() * sp, up, ang.sin() * sp),
                    life: 0.4,
                    max_life: 0.4,
                });
            }
            let top = self.stream_top();
            let fall = (self.hit - top).normalize_or_zero();
            self.beads.push(Bead {
                pos: top,
                vel: fall * 3.4,
                life: 0.32,
                max_life: 0.32,
            });
            if self.beads.len() > 24 {
                let extra = self.beads.len() - 24;
                self.beads.drain(0..extra);
            }
        }
    }

    fn puff_over_flame(&mut self, dt: f32) {
        self.acc_puff += dt;
        if self.acc_puff < 0.13 {
            return;
        }
        self.acc_puff = 0.0;
        let hot: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.heat > 0.45 && n.liquid > 0.2)
            .map(|(i, _)| i)
            .collect();
        if hot.is_empty() {
            return;
        }
        let pick = hot[(self.rnd() * hot.len() as f32) as usize % hot.len()];
        // Wisps above the tips. Cards that cross the soil turn into a depth grid.
        let lift = 0.95 + self.rnd() * 0.35;
        let pos = self.nodes[pick].pos + Vec3::Y * lift;
        let rise = 0.4 + self.rnd() * 0.2;
        let drift = self.rnd() * 0.2 - 0.1;
        let size = 0.22 + self.rnd() * 0.12;
        self.push_puff(pos, Vec3::new(drift, rise, drift * 0.2), 1.2, size, 0.0);
    }

    fn spawn_blast(&mut self) {
        // The visible blast is the ground ring. Extra cards here z-fight into a grid.
    }

    fn push_puff(&mut self, pos: Vec3, vel: Vec3, life: f32, size: f32, dark: f32) {
        if self.puffs.len() >= 16 {
            return;
        }
        self.puffs.push(Puff {
            pos,
            vel,
            life,
            max_life: life,
            size,
            dark,
        });
    }

    fn drift(&mut self, dt: f32, soil: &SoilField) {
        for ripple in &mut self.ripples {
            ripple.life -= dt;
            ripple.radius += dt * 1.65;
        }
        self.ripples.retain(|r| r.life > 0.0);
        for bead in &mut self.beads {
            bead.life -= dt;
            bead.vel.y -= 12.0 * dt;
            bead.pos += bead.vel * dt;
            let h = soil.sample(bead.pos.x, bead.pos.z).height;
            if bead.pos.y < h + 0.04 {
                bead.life = 0.0;
            }
        }
        self.beads.retain(|b| b.life > 0.0);
        for puff in &mut self.puffs {
            puff.life -= dt;
            puff.pos += puff.vel * dt;
            puff.size += dt * (0.4 + puff.dark * 0.8);
            puff.vel.y = (puff.vel.y - dt * 0.12).max(0.2);
        }
        self.puffs.retain(|p| p.life > 0.0);
        for streak in &mut self.streaks {
            streak.life -= dt;
            streak.vel.y -= 9.0 * dt;
            streak.pos += streak.vel * dt;
        }
        self.streaks.retain(|s| s.life > 0.0);
    }

    fn hold_chunks(&self, particles: &mut Particles) {
        let origin = if self.nodes.is_empty() {
            self.hit
        } else {
            self.nodes[0].pos
        };
        for piece in &mut particles.items {
            if !piece.alive || piece.life <= 0.0 {
                continue;
            }
            if !is_chunk(piece.kind) {
                continue;
            }
            if flat_len(piece.pos - origin) < 8.0 || flat_len(piece.pos - self.hit) < 8.0 {
                piece.life = piece.life.max(1.25);
            }
        }
    }

    /// Kick the mud clumps the landing already threw. The blast does not invent a new debris mesh.
    fn throw_chunks(&mut self, particles: &mut Particles) {
        let origin = self.blast_at;
        let mut n = 0u32;
        for piece in &mut particles.items {
            if !piece.alive || piece.life <= 0.0 {
                continue;
            }
            if !is_chunk(piece.kind) {
                continue;
            }
            let dist = flat_len(piece.pos - origin);
            if dist > 7.0 {
                continue;
            }
            let mut dir = Vec3::new(piece.pos.x - origin.x, 0.0, piece.pos.z - origin.z);
            if dir.length_squared() < 0.04 {
                let ang = piece.seed as f32 * 0.17;
                dir = Vec3::new(ang.cos(), 0.0, ang.sin());
            }
            let dir = dir.normalize_or_zero();
            let kick = (1.0 - dist / 7.0).clamp(0.25, 1.0);
            piece.grounded = false;
            piece.vel += dir * (3.2 + 4.6 * kick) + Vec3::Y * (3.0 + 3.2 * kick);
            piece.spin += Vec3::new(2.2, 1.4, 1.1) * kick;
            piece.life = piece.life.max(1.5);
            n += 1;
        }
        if n < 4 {
            particles.impact_burst(origin + Vec3::Y * 0.15);
        }
    }

    fn hottest(&self) -> Vec3 {
        let mut best = self.hit;
        let mut heat = -1.0;
        for node in &self.nodes {
            if node.heat > heat {
                heat = node.heat;
                best = node.pos;
            }
        }
        best
    }

    fn heat_at(&self, p: Vec3) -> f32 {
        let mut heat: f32 = 0.0;
        for node in &self.nodes {
            let d = flat_len(node.pos - p);
            if d < 1.15 {
                heat = heat.max(node.heat * (1.0 - d / 1.15));
            }
        }
        heat
    }

    fn stream_top(&self) -> Vec3 {
        if self.leak.y < self.hit.y + 0.34 {
            Vec3::new(self.leak.x, self.hit.y + 0.72, self.leak.z)
        } else {
            self.leak
        }
    }

    fn fade(&self) -> f32 {
        match self.phase {
            Phase::Ash => (1.0 - self.phase_t / ASH).clamp(0.0, 1.0),
            Phase::Idle | Phase::Done => 0.0,
            _ => 1.0,
        }
    }

    fn blast_visual(&self) -> (f32, f32) {
        match self.phase {
            Phase::Blast => {
                let u = (self.phase_t / BLAST).clamp(0.0, 1.0);
                let rad = 0.55 + u * 4.6;
                let alpha = if u < 0.32 {
                    1.0
                } else {
                    (1.0 - (u - 0.32) / 0.68).clamp(0.0, 1.0)
                };
                (rad, alpha)
            }
            Phase::Ash => {
                let u = (self.phase_t / 0.3).clamp(0.0, 1.0);
                (4.8, (1.0 - u) * 0.22)
            }
            _ => (0.0, 0.0),
        }
    }
}

pub fn vertices(show: &FuelShow, cam: Vec3, soil: &SoilField) -> Vec<FuelVert> {
    if matches!(show.phase, Phase::Idle | Phase::Done) {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(1024);
    let fade = show.fade();
    write_pool(&mut out, show, soil, fade);
    write_fan(&mut out, show, soil, fade);
    write_ripples(&mut out, show, soil);
    if show.phase == Phase::Pour || (show.phase == Phase::Spark && show.phase_t < 0.2) {
        write_stream(&mut out, show, cam, fade);
    }
    write_beads(&mut out, show, cam);
    write_flames(&mut out, show, cam, fade);
    if show.phase == Phase::Spark {
        let (right, up) = cam_axes(cam, show.spark_pos);
        push_card(
            &mut out,
            show.spark_pos,
            right * 0.1,
            up * 0.1,
            rgba(7, [1.0, 0.96, 0.75], 0.95),
        );
    }
    write_smoke(&mut out, show, cam);
    write_blast(&mut out, show, cam, soil);
    write_streaks(&mut out, show);
    if out.len() > 6000 {
        out.truncate(6000);
    }
    out
}

fn write_pool(out: &mut Vec<FuelVert>, show: &FuelShow, soil: &SoilField, fade: f32) {
    if show.nodes.len() < 2 {
        return;
    }
    let cols = 4;
    let rows = 3;
    for i in 0..show.nodes.len() - 1 {
        let a = &show.nodes[i];
        let b = &show.nodes[i + 1];
        let tangent = flat_norm(b.pos - a.pos);
        let right = Vec3::new(-tangent.z, 0.0, tangent.x);
        let ca = liquid_rgba(a.heat, 0.96 * fade);
        let cb = liquid_rgba(b.heat, 0.96 * fade);
        for row in 0..rows {
            let t0 = row as f32 / rows as f32;
            let t1 = (row + 1) as f32 / rows as f32;
            for col in 0..cols {
                let s0 = col as f32 / cols as f32;
                let s1 = (col + 1) as f32 / cols as f32;
                let p = |t: f32, s: f32| -> Vec3 { sheet_point(show, soil, a, b, right, t, s) };
                let c0 = lerp_rgba(ca, cb, t0);
                let c1 = lerp_rgba(ca, cb, t1);
                push_quad_c(
                    out,
                    [p(t0, s0), p(t0, s1), p(t1, s1), p(t1, s0)],
                    [c0, c0, c1, c1],
                    [[t0, s0], [t0, s1], [t1, s1], [t1, s0]],
                );
            }
        }
    }
}

fn write_fan(out: &mut Vec<FuelVert>, show: &FuelShow, soil: &SoilField, fade: f32) {
    let heat = show.heat_at(show.hit);
    let color = liquid_rgba(heat, 0.96 * fade);
    let grow = if show.phase == Phase::Pour {
        (0.4 + show.phase_t / 0.45).clamp(0.4, 1.0)
    } else {
        1.0
    };
    let mut center = on_sheet(soil, show.hit);
    center.y += liquid_lift(center, show.hit, show.clock) + 0.02;
    let segs = 16;
    let rings = 4;
    for ring in 0..rings {
        let u0 = ring as f32 / rings as f32;
        let u1 = (ring + 1) as f32 / rings as f32;
        for i in 0..segs {
            let a0 = i as f32 / segs as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
            let at = |ang: f32, u: f32| -> Vec3 {
                let r = pool_radius(show, ang) * grow * u;
                let mut p = on_sheet(
                    soil,
                    Vec3::new(center.x + ang.cos() * r, 0.0, center.z + ang.sin() * r),
                );
                p.y += liquid_lift(p, show.hit, show.clock) + (1.0 - u) * 0.02;
                p
            };
            let p2 = at(a1, u1);
            let p3 = at(a0, u1);
            // uv.y is 0.5 on the crown and 0 on the rim, so the shader cuts the lip.
            let y0 = 0.5 * (1.0 - u0);
            let y1 = 0.5 * (1.0 - u1);
            if ring == 0 {
                push_tri(
                    out,
                    [center, p2, p3],
                    color,
                    [
                        [0.5, 0.5],
                        [a1 / std::f32::consts::TAU, y1],
                        [a0 / std::f32::consts::TAU, y1],
                    ],
                );
            } else {
                let p0 = at(a0, u0);
                let p1 = at(a1, u0);
                push_quad(
                    out,
                    [p0, p1, p2, p3],
                    color,
                    [
                        [a0 / std::f32::consts::TAU, y0],
                        [a1 / std::f32::consts::TAU, y0],
                        [a1 / std::f32::consts::TAU, y1],
                        [a0 / std::f32::consts::TAU, y1],
                    ],
                );
            }
        }
    }
}

fn write_ripples(out: &mut Vec<FuelVert>, show: &FuelShow, soil: &SoilField) {
    for ripple in &show.ripples {
        let alpha = (ripple.life / ripple.max_life).clamp(0.0, 1.0) * 0.22;
        let color = rgba(2, [0.48, 0.54, 0.6], alpha);
        let segs = 48;
        let inner = (ripple.radius - 0.12).max(0.06);
        let outer = ripple.radius + 0.12;
        for i in 0..segs {
            let a0 = i as f32 / segs as f32 * std::f32::consts::TAU;
            let a1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
            let p = |ang: f32, r: f32| {
                on_sheet(
                    soil,
                    Vec3::new(
                        ripple.pos.x + ang.cos() * r,
                        0.0,
                        ripple.pos.z + ang.sin() * r,
                    ),
                )
            };
            push_quad(
                out,
                [p(a0, inner), p(a0, outer), p(a1, outer), p(a1, inner)],
                color,
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            );
        }
    }
}

fn write_stream(out: &mut Vec<FuelVert>, show: &FuelShow, cam: Vec3, fade: f32) {
    let top = show.stream_top();
    let bot = show.hit + Vec3::Y * 0.16;
    let right = sheet_right((bot - top).normalize_or_zero(), cam, top);
    let color = rgba(1, [0.02, 0.01, 0.008], 0.9 * fade);
    let threads = [(0.0, 0.1, 0.34), (-0.15, 0.035, 0.11), (0.14, 0.03, 0.09)];
    for (off, w0, w1) in threads {
        push_stream(out, cam, top + right * off, bot, w0, w1, color);
    }
}

fn write_beads(out: &mut Vec<FuelVert>, show: &FuelShow, cam: Vec3) {
    for bead in &show.beads {
        let (right, up) = cam_axes(cam, bead.pos);
        let a = (bead.life / bead.max_life).clamp(0.0, 1.0);
        push_card(
            out,
            bead.pos,
            right * 0.07,
            up * 0.09,
            rgba(3, [0.02, 0.014, 0.01], 0.9 * a),
        );
    }
}

fn write_flames(out: &mut Vec<FuelVert>, show: &FuelShow, cam: Vec3, fade: f32) {
    let len = show.nodes.len();
    for (i, node) in show.nodes.iter().enumerate() {
        if node.heat < 0.48 || node.liquid < 0.18 {
            continue;
        }
        let tangent = if i + 1 < len {
            flat_norm(show.nodes[i + 1].pos - node.pos)
        } else if i > 0 {
            flat_norm(node.pos - show.nodes[i - 1].pos)
        } else {
            Vec3::Z
        };
        // Stay on this node. The burn test rejects verts on the cold or dry neighbours.
        let height = 0.42 + node.heat * 0.48 * node.liquid;
        let alpha = (0.86 + 0.1 * node.heat) * fade;
        let color = rgba(4, [1.0, 0.36, 0.03], alpha);
        let base = node.pos + Vec3::Y * 0.14;
        let (cam_right, _) = cam_axes(cam, base);
        let mut face = Vec3::new(cam_right.x, 0.0, cam_right.z);
        if face.length_squared() < 1.0e-6 {
            face = Vec3::X;
        }
        face = face.normalize();
        let to = (cam - base).normalize_or_zero();
        let lean = Vec3::new(to.x, 0.0, to.z) * (height * 0.08);
        // Width stays on this node so the flame does not reach a cold or dry neighbour.
        push_card(out, base, face * 0.2, Vec3::Y * height + lean, color);
        push_card(
            out,
            base + tangent * 0.06,
            face * 0.14,
            Vec3::Y * height * 0.7 + lean,
            color,
        );
    }
}

fn write_smoke(out: &mut Vec<FuelVert>, show: &FuelShow, cam: Vec3) {
    for puff in &show.puffs {
        let (right, up) = cam_axes(cam, puff.pos);
        let life = (puff.life / puff.max_life).clamp(0.0, 1.0);
        let rgb = [0.55, 0.56, 0.57];
        let alpha = life * 0.4;
        let to_cam = (cam - puff.pos).normalize_or_zero();
        let pos = puff.pos + Vec3::Y * 0.2 + to_cam * 0.55;
        push_card(
            out,
            pos,
            right * puff.size,
            up * puff.size * 0.75,
            rgba(6, rgb, alpha),
        );
    }
}

fn write_blast(out: &mut Vec<FuelVert>, show: &FuelShow, cam: Vec3, soil: &SoilField) {
    let (rad, alpha) = show.blast_visual();
    if alpha < 0.03 || rad < 0.05 {
        return;
    }
    let origin = on_sheet(soil, show.blast_at);
    let segs = 40;
    let band = (0.22 + rad * 0.07).min(0.55);
    let dust = rgba(8, [0.55, 0.46, 0.36], alpha * 0.7);
    for i in 0..segs {
        let a0 = i as f32 / segs as f32 * std::f32::consts::TAU;
        let a1 = (i + 1) as f32 / segs as f32 * std::f32::consts::TAU;
        let wobble = |ang: f32| 0.86 + 0.14 * (ang * 3.0).sin().abs();
        let d0 = Vec3::new(a0.cos(), 0.0, a0.sin());
        let d1 = Vec3::new(a1.cos(), 0.0, a1.sin());
        let p = |dir: Vec3, r: f32, ang: f32| {
            let mut q = on_sheet(soil, origin + dir * r * wobble(ang));
            q.y += 0.05;
            q
        };
        let outer0 = rad;
        let inner0 = (rad - band).max(0.08);
        push_quad(
            out,
            [
                p(d0, inner0, a0),
                p(d0, outer0, a0),
                p(d1, outer0, a1),
                p(d1, inner0, a1),
            ],
            dust,
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        );
    }
    // A short hot pop at the center, gone once the ring has left the patch.
    if rad < 1.7 {
        let f = ((1.7 - rad) / 1.3).clamp(0.0, 1.0) * alpha;
        let (right, up) = cam_axes(cam, origin + Vec3::Y * 0.25);
        push_card(
            out,
            origin + Vec3::Y * 0.12,
            right * 0.42,
            up * 0.38,
            rgba(7, [1.0, 0.9, 0.62], f),
        );
    }
}

fn write_streaks(out: &mut Vec<FuelVert>, show: &FuelShow) {
    for streak in &show.streaks {
        let speed = streak.vel.length().max(0.2);
        let dir = streak.vel / speed;
        let mut side = dir.cross(Vec3::Y);
        if side.length_squared() < 1.0e-6 {
            side = Vec3::X;
        }
        side = side.normalize() * 0.06;
        let tail = dir * (0.16 + speed * 0.03);
        let life = (streak.life / streak.max_life).clamp(0.0, 1.0);
        push_card(
            out,
            streak.pos - tail,
            side,
            tail,
            rgba(9, [1.0, 0.9, 0.5], life * 0.95),
        );
    }
}

fn push_stream(
    out: &mut Vec<FuelVert>,
    cam: Vec3,
    top: Vec3,
    bot: Vec3,
    width0: f32,
    width1: f32,
    color: [f32; 4],
) {
    let steps = 5;
    for i in 0..steps {
        let t0 = i as f32 / steps as f32;
        let t1 = (i + 1) as f32 / steps as f32;
        let p0 = top.lerp(bot, t0);
        let p1 = top.lerp(bot, t1);
        let w0 = width0 + (width1 - width0) * t0;
        let w1 = width0 + (width1 - width0) * t1;
        let tangent = (p1 - p0).normalize_or_zero();
        let r0 = sheet_right(tangent, cam, p0) * w0;
        let r1 = sheet_right(tangent, cam, p1) * w1;
        push_quad(
            out,
            [p0 - r0, p0 + r0, p1 + r1, p1 - r1],
            color,
            [[0.0, t0], [1.0, t0], [1.0, t1], [0.0, t1]],
        );
    }
}

fn sheet_point(
    show: &FuelShow,
    soil: &SoilField,
    a: &Node,
    b: &Node,
    right: Vec3,
    t: f32,
    s: f32,
) -> Vec3 {
    let half_l = a.half_l + (b.half_l - a.half_l) * t;
    let half_r = a.half_r + (b.half_r - a.half_r) * t;
    let center = a.pos.lerp(b.pos, t);
    let bow = 1.0 + 0.1 * (t * std::f32::consts::PI).sin();
    let wob = 0.9 + 0.1 * ((center.x * 1.7 + center.z * 2.3).sin() * 0.5 + 0.5);
    let side = (-half_r + (half_l + half_r) * s) * bow * wob;
    let mut q = on_sheet(soil, center + right * side);
    let across = (s - 0.5).abs() * 2.0;
    q.y += liquid_lift(q, show.hit, show.clock);
    q.y += (1.0 - across) * 0.016;
    if across > 0.82 {
        q.y += 0.012;
    }
    q
}

fn liquid_lift(p: Vec3, origin: Vec3, t: f32) -> f32 {
    let dx = p.x - origin.x;
    let dz = p.z - origin.z;
    let d = (dx * dx + dz * dz).sqrt();
    let ring = (d * 7.4 - t * 3.5).sin() * (-d * 0.42).exp();
    let chop = (p.x * 2.4 + p.z * 1.3 + t * 1.7).sin();
    (0.006 + ring * 0.02 + chop * 0.005).max(0.0)
}

fn pool_radius(show: &FuelShow, ang: f32) -> f32 {
    let n = show.pool_r.len() as f32;
    let x = ang / std::f32::consts::TAU * n;
    let i0 = x.floor() as usize % show.pool_r.len();
    let i1 = (i0 + 1) % show.pool_r.len();
    let f = x.fract();
    let s = f * f * (3.0 - 2.0 * f);
    show.pool_r[i0] + (show.pool_r[i1] - show.pool_r[i0]) * s
}

fn lerp_rgba(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

fn leak_of(vehicle: &Vehicle) -> Vec3 {
    // A point on the rear body. The stream starts here. There is no tank mesh.
    vehicle.pos + vehicle.orientation() * Vec3::new(0.12, 0.2, -1.02)
}

fn on_soil(soil: &SoilField, p: Vec3) -> Vec3 {
    on_sheet(soil, p)
}

fn on_sheet(soil: &SoilField, p: Vec3) -> Vec3 {
    let h = soil.sample(p.x, p.z).height;
    // Extra lift so the sheet clears the ground in the depth buffer.
    Vec3::new(p.x, h + SHEET + 0.1, p.z)
}

fn flow_dir(soil: &SoilField, at: Vec3, vehicle: &Vehicle, flow: Vec3) -> Vec3 {
    let e = 0.45;
    let down = Vec3::new(
        soil.sample(at.x - e, at.z).height - soil.sample(at.x + e, at.z).height,
        0.0,
        soil.sample(at.x, at.z - e).height - soil.sample(at.x, at.z + e).height,
    );
    let slide = Vec3::new(vehicle.vel.x, 0.0, vehicle.vel.z);
    flat_norm(down * 3.0 + slide * 0.25 + flow)
}

fn is_chunk(kind: u8) -> bool {
    kind == 1 || kind == 3 || kind == 4
}

fn flat_len(v: Vec3) -> f32 {
    Vec3::new(v.x, 0.0, v.z).length()
}

fn flat_norm(v: Vec3) -> Vec3 {
    let flat = Vec3::new(v.x, 0.0, v.z);
    if flat.length_squared() < 1.0e-6 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        flat.normalize()
    }
}

fn sheet_right(tangent: Vec3, cam: Vec3, pos: Vec3) -> Vec3 {
    let mut right = tangent.cross(cam - pos);
    if right.length_squared() < 1.0e-6 {
        right = tangent.cross(Vec3::Y);
    }
    if right.length_squared() < 1.0e-6 {
        Vec3::X
    } else {
        right.normalize()
    }
}

fn cam_axes(cam: Vec3, pos: Vec3) -> (Vec3, Vec3) {
    let to = (cam - pos).normalize_or_zero();
    let right = Vec3::Y.cross(to);
    let right = if right.length_squared() < 1.0e-6 {
        Vec3::X
    } else {
        right.normalize()
    };
    let up = right.cross(to);
    let up = if up.length_squared() < 1.0e-6 {
        Vec3::Y
    } else {
        up.normalize()
    };
    (right, up)
}

fn liquid_rgba(heat: f32, alpha: f32) -> [f32; 4] {
    let k = heat.clamp(0.0, 1.0);
    rgba(
        0,
        [0.010 + 0.84 * k, 0.008 + 0.20 * k, 0.007 + 0.025 * k],
        alpha,
    )
}

fn rgba(style: u32, rgb: [f32; 3], alpha: f32) -> [f32; 4] {
    [
        rgb[0],
        rgb[1],
        rgb[2],
        style as f32 * 2.0 + alpha.clamp(0.0, 0.98),
    ]
}

fn push_card(out: &mut Vec<FuelVert>, bottom: Vec3, across: Vec3, up: Vec3, color: [f32; 4]) {
    push_quad(
        out,
        [
            bottom - across,
            bottom + across,
            bottom + across + up,
            bottom - across + up,
        ],
        color,
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    );
}

fn push_quad(out: &mut Vec<FuelVert>, p: [Vec3; 4], color: [f32; 4], uv: [[f32; 2]; 4]) {
    push_quad_c(out, p, [color, color, color, color], uv);
}

fn push_quad_c(out: &mut Vec<FuelVert>, p: [Vec3; 4], color: [[f32; 4]; 4], uv: [[f32; 2]; 4]) {
    const ORDER: [usize; 6] = [0, 1, 2, 0, 2, 3];
    for i in ORDER {
        out.push(FuelVert {
            pos: p[i].to_array(),
            color: color[i],
            uv: uv[i],
        });
    }
}

fn push_tri(out: &mut Vec<FuelVert>, p: [Vec3; 3], color: [f32; 4], uv: [[f32; 2]; 3]) {
    for i in 0..3 {
        out.push(FuelVert {
            pos: p[i].to_array(),
            color,
            uv: uv[i],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Input, World};

    fn style_of(v: &FuelVert) -> i32 {
        (v.color[3] * 0.5 + 0.001).floor() as i32
    }

    fn flipped() -> Vehicle {
        let mut vehicle = Vehicle::new();
        vehicle.pos = Vec3::new(0.0, 1.5, 0.0);
        vehicle.pitch = 2.2;
        vehicle.pitch_rate = -5.4;
        vehicle.roll = 0.35;
        vehicle.roll_rate = 2.3;
        vehicle.vel = Vec3::new(0.3, -0.4, 3.2);
        vehicle
    }

    #[test]
    fn ribbon_sits_on_the_soil_before_it_burns() {
        let soil = SoilField::flat(0.35);
        let mut show = FuelShow::new();
        let mut parts = Particles::new();
        let vehicle = flipped();
        for _ in 0..50 {
            show.tick(1.0 / 60.0, &vehicle, &soil, &mut parts, 0.0);
        }
        assert_eq!(show.label(), "pour");
        assert!(show.node_count() >= 3, "nodes {}", show.node_count());
        assert!(!show.ripples.is_empty(), "the hit should throw ripples");
        for node in &show.nodes {
            let h = soil.sample(node.pos.x, node.pos.z).height;
            assert!(
                (node.pos.y - (h + SHEET)).abs() < 0.02,
                "ribbon left the mud y {} h {}",
                node.pos.y,
                h
            );
            assert!(node.liquid > 0.5);
            assert!(node.heat < 0.05, "flame before the spark");
        }
        let cam = Vec3::new(-12.0, 10.0, -11.0);
        let verts = vertices(&show, cam, &soil);
        assert!(verts.iter().any(|v| style_of(v) == 0), "pool");
        assert!(
            verts.iter().any(|v| style_of(v) == 1),
            "stream from the hole"
        );
        assert!(verts.iter().any(|v| style_of(v) == 2), "ripples");
        assert!(verts.iter().all(|v| style_of(v) != 4), "no flame yet");
        assert!(
            verts
                .iter()
                .filter(|v| style_of(v) == 0)
                .any(|v| v.color[0] < 0.05),
            "unburnt fuel should be darker than the mud"
        );
    }

    #[test]
    fn ignition_stops_at_a_dry_gap() {
        let mut show = FuelShow::new();
        show.phase = Phase::Burn;
        for i in 0..8 {
            show.nodes.push(Node {
                pos: Vec3::new(i as f32 * 0.4, 0.05, 0.0),
                liquid: 1.0,
                heat: 0.0,
                half_l: 0.4,
                half_r: 0.4,
            });
        }
        show.nodes[0].heat = 1.0;
        show.nodes[4].liquid = 0.0;
        for _ in 0..36 {
            show.spread(1.0 / 60.0, 8.5);
        }
        assert!(show.nodes[1].heat > 0.5, "near {}", show.nodes[1].heat);
        assert!(show.nodes[2].heat > 0.25, "along {}", show.nodes[2].heat);
        assert!(
            show.nodes[4].heat < 0.08,
            "dry node caught {}",
            show.nodes[4].heat
        );
        assert!(
            show.nodes[5].heat < 0.08,
            "heat crossed the gap {}",
            show.nodes[5].heat
        );
    }

    #[test]
    fn fire_sits_only_on_hot_liquid() {
        let mut show = FuelShow::new();
        show.phase = Phase::Burn;
        show.hit = Vec3::new(2.5, 0.05, 2.5);
        show.pool_r = [0.7; 6];
        show.nodes.push(Node {
            pos: Vec3::new(0.0, 0.05, 0.0),
            liquid: 1.0,
            heat: 0.0,
            half_l: 0.5,
            half_r: 0.5,
        });
        show.nodes.push(Node {
            pos: Vec3::new(0.8, 0.05, 0.0),
            liquid: 1.0,
            heat: 1.0,
            half_l: 0.5,
            half_r: 0.5,
        });
        show.nodes.push(Node {
            pos: Vec3::new(1.6, 0.05, 0.0),
            liquid: 0.0,
            heat: 1.0,
            half_l: 0.5,
            half_r: 0.5,
        });
        let verts = vertices(&show, Vec3::new(-10.0, 12.0, -8.0), &SoilField::flat(0.0));
        let mut cold = false;
        let mut hot = false;
        let mut flame_on_hot = false;
        let mut flame_on_cold = false;
        let mut flame_on_dry = false;
        for vert in &verts {
            if style_of(vert) == 0 {
                if vert.color[0] < 0.06 {
                    cold = true;
                }
                if vert.color[0] > 0.45 {
                    hot = true;
                }
            }
            if style_of(vert) == 4 {
                let x = vert.pos[0];
                if (0.45..1.2).contains(&x) {
                    flame_on_hot = true;
                }
                if x < 0.28 {
                    flame_on_cold = true;
                }
                if x > 1.25 {
                    flame_on_dry = true;
                }
            }
        }
        assert!(cold, "cold liquid missing");
        assert!(hot, "burning liquid should reflect");
        assert!(flame_on_hot, "flame should sit on the wet patch");
        assert!(!flame_on_cold, "flame on unlit liquid");
        assert!(!flame_on_dry, "flame on a dry node");
    }

    #[test]
    fn show_runs_the_full_sequence_and_throws_chunks() {
        let soil = SoilField::flat(0.3);
        let mut show = FuelShow::new();
        let mut parts = Particles::new();
        let vehicle = flipped();
        parts.spawn(vehicle.pos, Vec3::ZERO, 4.0, 0.42, 3);
        parts.items[0].grounded = true;
        parts.items[0].vel = Vec3::ZERO;
        parts.items[0].pos = Vec3::new(vehicle.pos.x, 0.35, vehicle.pos.z);
        let mut seen = Vec::new();
        let mut burn_frames = 0u32;
        let mut early_hot = 0usize;
        let mut late_hot = 0usize;
        for _ in 0..420 {
            let before = show.label();
            show.tick(1.0 / 60.0, &vehicle, &soil, &mut parts, 0.0);
            let after = show.label();
            if after != before {
                seen.push(after);
            }
            if after == "burn" {
                burn_frames += 1;
                let hot = show.nodes.iter().filter(|n| n.heat > 0.55).count();
                if burn_frames == 4 {
                    early_hot = hot;
                }
                if burn_frames == 50 {
                    late_hot = hot;
                }
            }
        }
        assert_eq!(
            seen,
            ["pour", "spark", "burn", "blast", "ash", "done"],
            "sequence {seen:?}"
        );
        assert!(early_hot >= 1, "spark did not catch");
        assert!(
            late_hot >= early_hot,
            "burn front {early_hot} -> {late_hot}"
        );
        assert!(show.threw_chunks());
        assert!(
            !parts.items[0].grounded,
            "blast left the clump on the ground"
        );
        assert!(
            parts.items[0].vel.length() > 2.5,
            "clump vel {}",
            parts.items[0].vel.length()
        );
        assert_eq!(show.node_count(), 0);
        assert!(vertices(&show, Vec3::new(-8.0, 9.0, -8.0), &soil).is_empty());
    }

    #[test]
    fn gentle_chassis_does_not_rupture() {
        let soil = SoilField::flat(0.2);
        let mut show = FuelShow::new();
        let mut parts = Particles::new();
        let mut vehicle = Vehicle::new();
        vehicle.vel = Vec3::new(0.0, 0.0, 4.0);
        vehicle.pitch = 0.08;
        vehicle.pitch_rate = 0.15;
        for _ in 0..120 {
            show.tick(1.0 / 60.0, &vehicle, &soil, &mut parts, 0.0);
        }
        assert!(!show.is_live());
        assert_eq!(show.node_count(), 0);
    }

    #[test]
    fn heavy_hit_ruptures() {
        let soil = SoilField::flat(0.3);
        let mut show = FuelShow::new();
        let mut parts = Particles::new();
        let mut vehicle = Vehicle::new();
        vehicle.pos = Vec3::new(1.0, 0.8, 1.0);
        vehicle.vel = Vec3::new(0.0, -0.2, 0.0);
        show.tick(1.0 / 60.0, &vehicle, &soil, &mut parts, 0.0);
        assert!(!show.is_live());
        show.tick(1.0 / 60.0, &vehicle, &soil, &mut parts, 2.4);
        assert!(show.is_live(), "phase {}", show.label());
    }

    #[test]
    fn world_hook_starts_after_a_hard_flip() {
        let mut world = World::flat(0.25);
        world.vehicle.pitch = 2.25;
        world.vehicle.pitch_rate = -5.5;
        world.vehicle.roll_rate = 2.2;
        world.vehicle.vel = Vec3::new(0.2, 0.4, 4.0);
        world.vehicle.pos.y += 0.7;
        for _ in 0..36 {
            world.step(1.0 / 60.0, &Input::default());
        }
        assert!(world.fuel.is_live(), "phase {}", world.fuel.label());
        assert!(
            world.fuel.node_count() >= 2,
            "nodes {}",
            world.fuel.node_count()
        );
    }

    #[test]
    fn demo_lap_plays_the_show_after_chunks_settle() {
        let mut world = World::demo();
        let mut seen = Vec::new();
        let mut last = "idle";
        let mut noted = false;
        let mut phase_at_chunks = "idle";
        let mut threw_at_chunks = false;
        let mut blast_t = -1.0_f32;
        for _ in 0..(12.0 * 60.0) as u32 {
            let input = crate::demo_drive(&world);
            world.step(1.0 / 60.0, &input);
            let label = world.fuel.label();
            if label != last {
                seen.push(label);
                last = label;
                if label == "blast" {
                    blast_t = world.time;
                }
            }
            if !noted && world.time >= 6.4 {
                phase_at_chunks = world.fuel.label();
                threw_at_chunks = world.fuel.threw_chunks();
                noted = true;
            }
        }
        assert!(
            seen.windows(4)
                .any(|w| w == ["pour", "spark", "burn", "blast"]),
            "lap sequence {seen:?}"
        );
        assert!(
            !threw_at_chunks,
            "blast threw clumps before they could settle, phase {phase_at_chunks}"
        );
        assert!(
            blast_t > 6.4,
            "blast at {blast_t}, phase then {phase_at_chunks}"
        );
    }

    #[test]
    fn world_drive_stays_quiet() {
        let mut world = World::flat(0.2);
        world.run_seconds(2.0, |_| Input {
            throttle: 1.0,
            brake: 0.0,
            steer: 0.0,
        });
        assert!(!world.fuel.is_live(), "phase {}", world.fuel.label());
        assert_eq!(world.fuel.node_count(), 0);
    }
}
