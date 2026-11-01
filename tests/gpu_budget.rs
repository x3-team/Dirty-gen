//! Static GPU budget, no GPU needed (CI has none).
//!
//! * Every `src/*.wgsl` loop with a literal bound, and every `*_STEPS`
//!   constant, stays under the per-pixel cap. Raymarch steps belong in
//!   `override QUALITY_*` constants that the preset sets
//!   (`silt::quality::Quality::wgsl_constants`), not in bigger literals.
//! * `shader.wgsl` validates with naga and translates to GLSL ES 3.00 for
//!   every entry point, so the WebGL2 build still compiles.
//! * WebGL2 limits: at most 16 sampled textures and 12 uniform blocks per
//!   entry point.
//!
//! Draw calls, triangles and texture memory need a device; `silt budget
//! --software` checks those on lavapipe (see `.github/workflows/ci.yml`).

use std::path::Path;

use silt::quality::{static_budget, Preset, Quality};

fn wgsl_files() -> Vec<(String, String)> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("wgsl") {
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            out.push((name, std::fs::read_to_string(&path).unwrap()));
        }
    }
    out.sort();
    out
}

#[test]
fn wgsl_loop_bounds_under_cap() {
    let cap = Preset::ALL
        .iter()
        .map(|p| static_budget(*p).max_loop_bound)
        .max()
        .unwrap();
    let for_re = regex_lite::Regex::new(
        r"for\s*\(\s*var\s+\w+\s*(?::\s*\w+)?\s*=\s*(-?\d+)u?\s*;\s*\w+\s*(<=|<)\s*(-?\d+)u?",
    )
    .unwrap();
    let steps_re =
        regex_lite::Regex::new(r"(?:const|override)\s+(\w*STEPS\w*)\s*(?::\s*\w+)?\s*=\s*(\d+)")
            .unwrap();
    let mut seen = 0;
    let mut report = Vec::new();
    for (name, src) in wgsl_files() {
        for (lineno, line) in src.lines().enumerate() {
            if let Some(c) = for_re.captures(line) {
                let lo: i64 = c[1].parse().unwrap();
                let hi: i64 = c[3].parse().unwrap();
                let n = hi - lo + if &c[2] == "<=" { 1 } else { 0 };
                seen += 1;
                if n > cap as i64 {
                    report.push(format!("{name}:{} loop of {n} > {cap}", lineno + 1));
                }
            }
            if let Some(c) = steps_re.captures(line) {
                let n: u32 = c[2].parse().unwrap();
                println!("{name}:{} {} = {n}", lineno + 1, &c[1]);
                if n > cap {
                    report.push(format!("{name}:{} {} = {n} > {cap}", lineno + 1, &c[1]));
                }
            }
        }
    }
    println!("WGSL_LOOPS literal={seen} cap={cap}");
    assert!(
        report.is_empty(),
        "per-pixel loop budget:\n{}",
        report.join("\n")
    );
}

#[test]
fn shader_validates_and_runs_on_webgl2() {
    // The renderer compiles shader.wgsl with terrain_geom.wgsl and the
    // contract readers appended (render.rs). WebGL2 has no storage buffers or
    // fifth bind group, so it gets the stand-in readers: validate that module.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let src = format!(
        "{}\n{}\n{}",
        std::fs::read_to_string(dir.join("shader.wgsl")).unwrap(),
        std::fs::read_to_string(dir.join("terrain_geom.wgsl")).unwrap(),
        std::fs::read_to_string(dir.join("beauty_nocontract.wgsl")).unwrap()
    );
    let module = naga::front::wgsl::parse_str(&src).expect("shader.wgsl parses");
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("shader.wgsl validates without extra capabilities");

    // Per entry point: WebGL2 counts the units one program binds.
    let mut worst = (0usize, 0usize, String::new());
    for (i, ep) in module.entry_points.iter().enumerate() {
        let fi = info.get_entry_point(i);
        let mut textures = 0usize;
        let mut uniforms = 0usize;
        let mut names: Vec<String> = Vec::new();
        for (h, var) in module.global_variables.iter() {
            if fi[h].is_empty() {
                continue;
            }
            match var.space {
                naga::AddressSpace::Uniform => uniforms += 1,
                naga::AddressSpace::Handle => {
                    if let naga::TypeInner::Image { .. } = module.types[var.ty].inner {
                        textures += 1;
                        names.push(var.name.clone().unwrap_or_default());
                    }
                }
                naga::AddressSpace::Storage { .. } => {
                    panic!("{} uses a storage buffer; WebGL2 has none", ep.name)
                }
                _ => {}
            }
        }
        assert!(
            textures <= 16,
            "{}: {textures} textures > WebGL2 16 units: {}",
            ep.name,
            names.join(", ")
        );
        assert!(
            uniforms <= 12,
            "{}: {uniforms} uniform blocks > WebGL2 12",
            ep.name
        );
        if textures > worst.0 {
            worst = (textures, uniforms, ep.name.clone());
        }
    }
    println!(
        "SHADER_BINDINGS max_textures={} uniforms={} entry={}",
        worst.0, worst.1, worst.2
    );

    let options = naga::back::glsl::Options {
        version: naga::back::glsl::Version::Embedded {
            version: 300,
            is_webgl: true,
        },
        writer_flags: naga::back::glsl::WriterFlags::empty(),
        binding_map: Default::default(),
        zero_initialize_workgroup_memory: false,
    };
    let mut n = 0;
    for ep in module.entry_points.iter() {
        let pipeline_options = naga::back::glsl::PipelineOptions {
            shader_stage: ep.stage,
            entry_point: ep.name.clone(),
            multiview: None,
        };
        let mut out = String::new();
        let mut writer = naga::back::glsl::Writer::new(
            &mut out,
            &module,
            &info,
            &options,
            &pipeline_options,
            naga::proc::BoundsCheckPolicies::default(),
        )
        .unwrap_or_else(|e| panic!("GLSL ES writer for {}: {e}", ep.name));
        writer
            .write()
            .unwrap_or_else(|e| panic!("GLSL ES for {}: {e}", ep.name));
        n += 1;
    }
    println!("SHADER_GLSL_ES entry_points={n}");
    assert!(n > 0);
}

#[test]
fn quality_constants_apply_to_an_override_shader() {
    // The pattern the volume / SSR / rut passes use: the preset sets the
    // step count through an override; the WGSL default is the `high` value.
    let src = r#"
override QUALITY_FOG_STEPS: i32 = 40;
@fragment fn fs() -> @location(0) vec4<f32> {
    var a = 0.0;
    for (var k: i32 = 0; k < QUALITY_FOG_STEPS; k = k + 1) { a = a + 0.01; }
    return vec4<f32>(a);
}
"#;
    let module = naga::front::wgsl::parse_str(src).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    for p in Preset::ALL {
        let consts = Quality::preset(p).constants_for(src);
        assert_eq!(consts.len(), 1);
        let (m, _) = naga::back::pipeline_constants::process_overrides(&module, &info, &consts)
            .unwrap_or_else(|e| panic!("{}: {e}", p.name()));
        assert!(m.overrides.iter().count() <= 1);
    }
}
