//! Graphics quality presets and the GPU frame budget.
//!
//! Reference hardware (assumption, see `docs/GPU_BUDGET.md`):
//!
//! | Preset | Reference GPU | Target |
//! | --- | --- | --- |
//! | `low` | GTX 1050 Ti / Iris Xe / browser WebGL2 | 1080p, 60 FPS on the 1050 Ti |
//! | `medium` | GTX 1660 / RTX 3050 / RX 6600 | 1080p, 60 FPS |
//! | `high` | RTX 3070 / RX 6700 XT | 1080p, 60 FPS (1440p ok) |
//!
//! Every preset gets the same GPU frame budget on its own reference card:
//! [`GPU_FRAME_BUDGET_MS`]. 16.7 ms is one 60 Hz refresh; the rest is
//! driver, compositor and present jitter. The CPU solver has its own 7 ms
//! budget (`docs/PERF_BUDGET.md`) and runs while the GPU draws the previous
//! frame, so the two do not add.
//!
//! Shaders read the knobs as WGSL `override` constants named in
//! [`Quality::wgsl_constants`]. A shader declares only the ones it uses;
//! [`Quality::constants_for`] filters the table against the source, because
//! wgpu rejects a pipeline constant the module does not declare.
//!
//! Captures and proof stills stay on [`Preset::High`] so committed images do
//! not move when the window default changes.

use std::sync::atomic::{AtomicU8, Ordering};

/// GPU frame budget on the preset's reference card, milliseconds.
pub const GPU_FRAME_BUDGET_MS: f32 = 12.0;

/// WebGL2 guarantees 2048; every desktop GPU and WebGPU give 8192. The
/// browser build must stay under this for any single texture side.
pub const WEBGL2_MAX_TEXTURE_SIDE: u32 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    Low,
    Medium,
    High,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Low, Preset::Medium, Preset::High];

    pub fn name(self) -> &'static str {
        match self {
            Preset::Low => "low",
            Preset::Medium => "medium",
            Preset::High => "high",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "low" | "l" | "0" => Some(Preset::Low),
            "medium" | "med" | "m" | "1" => Some(Preset::Medium),
            "high" | "h" | "2" => Some(Preset::High),
            _ => None,
        }
    }

    pub fn next(self) -> Self {
        match self {
            Preset::Low => Preset::Medium,
            Preset::Medium => Preset::High,
            Preset::High => Preset::Low,
        }
    }

    pub fn reference_gpu(self) -> &'static str {
        match self {
            Preset::Low => "GTX 1050 Ti / Iris Xe / WebGL2",
            Preset::Medium => "GTX 1660 / RTX 3050 / RX 6600",
            Preset::High => "RTX 3070 / RX 6700 XT",
        }
    }

    pub fn quality(self) -> Quality {
        Quality::preset(self)
    }
}

/// Every knob a preset sets. Fields that the `main` renderer does not draw
/// yet (fog, SSR, planar, tessellation) are the contract for the passes that
/// land later (beauty PR #60, volumetric air, rut geometry).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quality {
    pub preset: Preset,
    /// Internal render resolution over the window size. The blit upsamples.
    pub render_scale: f32,
    /// Sun shadow map side, texels.
    pub shadow_res: u32,
    /// Second, tight shadow map around the vehicle. 0 = off.
    pub shadow_near_res: u32,
    /// Ray-march steps of the dust/fog volume per pixel.
    pub fog_steps: u32,
    /// Sun self-shadow steps per fog sample.
    pub fog_shadow_steps: u32,
    /// Air slabs the volume pass samples (contract air grid layers).
    pub air_layers: u32,
    /// Screen-space reflection steps on water. 0 = off (sky + IBL only).
    pub ssr_steps: u32,
    /// Planar mirror pass for the vehicle over standing water.
    pub planar_reflection: bool,
    /// Rut geometry subdivisions per 20 cm coarse cell edge.
    pub rut_tess: u32,
    /// Spray / chunk particles drawn per frame.
    pub max_particles: u32,
    /// Largest side of a material texture (vehicle maps are 4096 at source).
    pub max_texture_side: u32,
}

impl Quality {
    pub const fn preset(preset: Preset) -> Self {
        match preset {
            Preset::Low => Quality {
                preset,
                render_scale: 0.75,
                shadow_res: 1024,
                shadow_near_res: 0,
                fog_steps: 12,
                fog_shadow_steps: 1,
                air_layers: 8,
                ssr_steps: 0,
                planar_reflection: false,
                rut_tess: 1,
                max_particles: 2000,
                max_texture_side: 1024,
            },
            Preset::Medium => Quality {
                preset,
                render_scale: 1.0,
                shadow_res: 2048,
                shadow_near_res: 1024,
                fog_steps: 24,
                fog_shadow_steps: 2,
                air_layers: 16,
                ssr_steps: 16,
                planar_reflection: false,
                rut_tess: 2,
                max_particles: 6000,
                max_texture_side: 2048,
            },
            Preset::High => Quality {
                preset,
                render_scale: 1.0,
                shadow_res: 2048,
                shadow_near_res: 2048,
                fog_steps: 40,
                fog_shadow_steps: 3,
                air_layers: 16,
                ssr_steps: 32,
                planar_reflection: true,
                rut_tess: 4,
                max_particles: 12000,
                max_texture_side: 4096,
            },
        }
    }

    /// Internal target size for a window of `width × height`.
    pub fn scaled(&self, width: u32, height: u32) -> (u32, u32) {
        let s = self.render_scale.clamp(0.25, 2.0);
        (
            ((width as f32 * s).round() as u32).max(1),
            ((height as f32 * s).round() as u32).max(1),
        )
    }

    /// WGSL `override` names and values for this preset.
    pub fn wgsl_constants(&self) -> [(&'static str, f64); 8] {
        [
            ("QUALITY_FOG_STEPS", self.fog_steps as f64),
            ("QUALITY_FOG_SHADOW_STEPS", self.fog_shadow_steps as f64),
            ("QUALITY_AIR_LAYERS", self.air_layers as f64),
            ("QUALITY_SSR_STEPS", self.ssr_steps as f64),
            ("QUALITY_RUT_TESS", self.rut_tess as f64),
            ("QUALITY_SHADOW_RES", self.shadow_res as f64),
            ("QUALITY_SHADOW_NEAR_RES", self.shadow_near_res as f64),
            (
                "QUALITY_PLANAR",
                if self.planar_reflection { 1.0 } else { 0.0 },
            ),
        ]
    }

    /// The subset of [`Quality::wgsl_constants`] that `wgsl` declares with
    /// `override`. Pass it as `compilation_options.constants`.
    pub fn constants_for(&self, wgsl: &str) -> std::collections::HashMap<String, f64> {
        self.wgsl_constants()
            .into_iter()
            .filter(|(name, _)| declares_override(wgsl, name))
            .map(|(name, value)| (name.to_string(), value))
            .collect()
    }
}

fn declares_override(wgsl: &str, name: &str) -> bool {
    wgsl.lines().any(|line| {
        let line = line.trim_start();
        let Some(rest) = line.strip_prefix("override") else {
            return false;
        };
        let rest = rest.trim_start();
        rest.strip_prefix(name)
            .map(|tail| {
                tail.trim_start()
                    .starts_with(|c: char| c == ':' || c == '=' || c == ';')
            })
            .unwrap_or(false)
    })
}

/// Per-pass GPU budget on the preset's reference card, milliseconds. Labels
/// are the render pass labels the renderer stamps. A pass a preset does not
/// encode keeps its row with 0.0.
pub fn pass_budgets(preset: Preset) -> &'static [(&'static str, f32)] {
    match preset {
        Preset::Low => &LOW_PASSES,
        Preset::Medium => &MEDIUM_PASSES,
        Preset::High => &HIGH_PASSES,
    }
}

/// Budget of one labelled pass, if the table names it.
pub fn pass_budget(preset: Preset, label: &str) -> Option<f32> {
    pass_budgets(preset)
        .iter()
        .find(|(name, _)| *name == label)
        .map(|(_, ms)| *ms)
}

// Same split on every preset's own reference card. The opaque `main` pass
// carries terrain, rut geometry, vehicle and props. Volumetric air is
// `dust-volume` (+ `dust-volume-up`, the upsample). The sum stays under
// GPU_FRAME_BUDGET_MS with ~0.3 ms reserve; `budget_rows_sum_under_frame`
// checks that.
const LOW_PASSES: [(&str, f32); 11] = [
    ("shadow", 1.2),
    ("shadow-near", 0.0),
    ("planar", 0.0),
    ("main", 5.6),
    ("water-ssr", 0.0),
    ("water", 1.4),
    ("after-water", 1.0),
    ("dust-volume", 2.0),
    ("dust-volume-up", 0.3),
    ("blit", 0.3),
    ("overlay", 0.1),
];

const MEDIUM_PASSES: [(&str, f32); 11] = [
    ("shadow", 1.2),
    ("shadow-near", 0.4),
    ("planar", 0.0),
    ("main", 4.5),
    ("water-ssr", 0.8),
    ("water", 1.2),
    ("after-water", 0.8),
    ("dust-volume", 2.0),
    ("dust-volume-up", 0.4),
    ("blit", 0.3),
    ("overlay", 0.1),
];

const HIGH_PASSES: [(&str, f32); 11] = [
    ("shadow", 1.0),
    ("shadow-near", 0.5),
    ("planar", 0.9),
    ("main", 3.8),
    ("water-ssr", 0.9),
    ("water", 1.0),
    ("after-water", 0.7),
    ("dust-volume", 2.2),
    ("dust-volume-up", 0.4),
    ("blit", 0.2),
    ("overlay", 0.1),
];

/// Static limits the CI budget check (`silt budget`) asserts per preset at
/// 1920×1080. Measured on lavapipe, which counts the same draws and
/// allocations a hardware adapter would. Triangles are everything submitted
/// in one frame, all passes (the terrain is drawn by shadow, main and water).
/// Texture memory is what the device holds: `medium` must fit a 6 GB card
/// next to the OS and a browser, `low` a 2 GB one.
#[derive(Clone, Copy, Debug)]
pub struct StaticBudget {
    pub draw_calls: u32,
    pub triangles: u64,
    /// Texture memory the device holds, MiB.
    pub texture_mib: f64,
    /// Buffer memory the device holds, MiB.
    pub buffer_mib: f64,
    /// Largest per-pixel loop bound in any WGSL loop with a literal bound.
    pub max_loop_bound: u32,
}

pub const fn static_budget(preset: Preset) -> StaticBudget {
    match preset {
        Preset::Low => StaticBudget {
            draw_calls: 400,
            triangles: 2_000_000,
            texture_mib: 512.0,
            buffer_mib: 256.0,
            max_loop_bound: 64,
        },
        Preset::Medium => StaticBudget {
            draw_calls: 500,
            triangles: 3_000_000,
            texture_mib: 1024.0,
            buffer_mib: 384.0,
            max_loop_bound: 64,
        },
        Preset::High => StaticBudget {
            draw_calls: 700,
            triangles: 5_000_000,
            texture_mib: 2560.0,
            buffer_mib: 512.0,
            max_loop_bound: 64,
        },
    }
}

static ACTIVE: AtomicU8 = AtomicU8::new(2);

/// Preset the renderer builds with. Captures leave it on `High`.
pub fn active() -> Quality {
    match ACTIVE.load(Ordering::Relaxed) {
        0 => Quality::preset(Preset::Low),
        1 => Quality::preset(Preset::Medium),
        _ => Quality::preset(Preset::High),
    }
}

pub fn set_active(preset: Preset) {
    let v = match preset {
        Preset::Low => 0,
        Preset::Medium => 1,
        Preset::High => 2,
    };
    ACTIVE.store(v, Ordering::Relaxed);
}

/// Preset for an interactive window: `SILT_QUALITY` (or `--quality`, parsed
/// by the caller) wins, else `medium`, else `low` on WebGL2 or a software
/// adapter.
pub fn auto_preset(requested: Option<&str>, webgl2: bool, software: bool) -> Preset {
    if let Some(p) = requested.and_then(Preset::parse) {
        return p;
    }
    if webgl2 || software {
        Preset::Low
    } else {
        Preset::Medium
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_rows_sum_under_frame() {
        for p in Preset::ALL {
            let sum: f32 = pass_budgets(p).iter().map(|(_, ms)| ms).sum();
            assert!(
                sum <= GPU_FRAME_BUDGET_MS,
                "{} passes sum {sum} > {GPU_FRAME_BUDGET_MS}",
                p.name()
            );
        }
    }

    #[test]
    fn presets_are_monotone() {
        let [l, m, h] = Preset::ALL.map(Quality::preset);
        assert!(l.render_scale <= m.render_scale && m.render_scale <= h.render_scale);
        assert!(l.shadow_res <= m.shadow_res && m.shadow_res <= h.shadow_res);
        assert!(l.fog_steps < m.fog_steps && m.fog_steps < h.fog_steps);
        assert!(l.ssr_steps <= m.ssr_steps && m.ssr_steps <= h.ssr_steps);
        assert!(l.rut_tess <= m.rut_tess && m.rut_tess <= h.rut_tess);
        for q in [l, m, h] {
            // Sun and near maps share one atlas row (beauty branch layout).
            assert!(q.shadow_res + q.shadow_near_res <= WEBGL2_MAX_TEXTURE_SIDE);
            assert!(q.fog_steps + 16 <= static_budget(q.preset).max_loop_bound);
        }
    }

    #[test]
    fn override_filter_matches_declarations_only() {
        let src = "override QUALITY_FOG_STEPS: u32 = 40u;\nconst QUALITY_SSR_STEPS: u32 = 2u;\n// override QUALITY_RUT_TESS\noverride QUALITY_AIR_LAYERS_X: u32 = 1u;\n";
        let c = Quality::preset(Preset::Low).constants_for(src);
        assert_eq!(c.len(), 1);
        assert_eq!(c["QUALITY_FOG_STEPS"], 12.0);
    }

    #[test]
    fn parse_and_auto() {
        assert_eq!(Preset::parse("Medium"), Some(Preset::Medium));
        assert_eq!(Preset::parse("ultra"), None);
        assert_eq!(auto_preset(None, true, false), Preset::Low);
        assert_eq!(auto_preset(None, false, false), Preset::Medium);
        assert_eq!(auto_preset(Some("high"), true, false), Preset::High);
        assert_eq!(Quality::preset(Preset::Low).scaled(1920, 1080), (1440, 810));
    }
}
