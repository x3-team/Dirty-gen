fn main() {
    println!("cargo:rerun-if-changed=src/fluid_core.rs");
    println!("cargo::rustc-check-cfg=cfg(has_fluid_core)");
    if std::path::Path::new("src/fluid_core.rs").exists() {
        println!("cargo:rustc-cfg=has_fluid_core");
        let out = std::env::var("OUT_DIR").expect("OUT_DIR");
        let path = std::path::Path::new("src/fluid_core.rs")
            .canonicalize()
            .expect("fluid_core.rs");
        let link = format!(
            "#[path = \"{}\"]\nmod fluid_core;\n",
            path.display().to_string().replace('\\', "\\\\")
        );
        std::fs::write(format!("{out}/fluid_core_link.rs"), link).expect("link");
    }
}
