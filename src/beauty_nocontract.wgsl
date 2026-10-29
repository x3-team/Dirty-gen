// Stand-ins for beauty_contract.wgsl on a device that cannot bind the
// render contract (WebGL2 has no storage buffers). Every reader returns the
// caller's fallback or zero, so the picture keeps its own soil cache.

fn bc_inside(xz: vec2<f32>) -> f32 {
    return 0.0;
}

fn bc_moisture(xz: vec2<f32>, fallback: f32) -> f32 {
    return fallback;
}

fn bc_film(xz: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn bc_flow(xz: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn bc_wet_edge(xz: vec2<f32>) -> f32 {
    return 0.0;
}

fn bc_foam_crest(xz: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn bc_crust(xz: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.0);
}

fn bc_dust_settled(xz: vec2<f32>) -> f32 {
    return 0.0;
}
