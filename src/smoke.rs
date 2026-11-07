//! Smoke and fire: a world-anchored 3D grid on wgpu compute.
//!
//! The dust and fog air window (`dust.rs`) rides with the Niva and is 4 m
//! tall with no pressure solve. A smoke column stands still in the world, is
//! taller, and needs the pressure solve and vorticity to roll, so it gets its
//! own small grid around the fire here. Kernels are in `smoke.wgsl`:
//! GPU Gems 3 ch. 30 structure (advect, vorticity confinement, Jacobi
//! pressure, project) with temperature and soot buoyancy and a first-order
//! fuel burn for the flame.
//!
//! The grid publishes one `rgba16float` 3D texture (soot, temperature / 1000 K,
//! burn rate, sun transmittance). The renderer marches it; nothing else reads
//! it. It is not part of the render contract.
//!
//! Compute needs WebGPU or a native backend. On WebGL2 there is no smoke.

use glam::Vec3;

const SHADER: &str = include_str!("smoke.wgsl");

/// Kernels, in step order. `k_jacobi` runs `jacobi_iters` times.
const ENTRIES: [&str; 8] = [
    "k_advect",
    "k_maccormack",
    "k_curl",
    "k_forces",
    "k_divergence",
    "k_jacobi",
    "k_project",
    "k_publish",
];

/// Up to this many sources per grid.
pub const MAX_SOURCES: usize = 4;

/// Ambient air, K.
pub const AMBIENT_K: f32 = 293.0;
/// Gravity, m/s².
pub const GRAVITY: f32 = 9.81;

/// A fire or a smoke vent on the ground. World metres.
#[derive(Clone, Copy, Debug)]
pub struct FireSource {
    pub pos: Vec3,
    pub radius: f32,
    /// Fuel fed per second (fraction of a cell per second at the centre).
    /// Burnt fuel becomes heat and soot; 0 for a cold smoke vent.
    pub fuel_s: f32,
    /// Soot fed directly per second (smouldering, a smoke grenade).
    pub soot_s: f32,
    /// Heat fed directly per second, K/s.
    pub heat_k_s: f32,
    /// Upward speed the source pushes, m/s.
    pub up_m_s: f32,
}

impl FireSource {
    /// A campfire-sized open flame with a sooty plume.
    pub fn campfire(pos: Vec3) -> Self {
        Self {
            pos,
            radius: 0.5,
            fuel_s: 8.0,
            soot_s: 0.0,
            heat_k_s: 0.0,
            up_m_s: 0.5,
        }
    }

    /// Cold smoke from a vent, no flame (the UE5 smoke-column reference).
    pub fn smoke_vent(pos: Vec3) -> Self {
        Self {
            pos,
            radius: 0.45,
            fuel_s: 0.0,
            soot_s: 1.5,
            heat_k_s: 260.0,
            up_m_s: 1.4,
        }
    }
}

/// Solver constants. Defaults are tuned against the UE5 smoke column.
#[derive(Clone, Copy, Debug)]
pub struct SmokeConfig {
    /// Cells x, y, z.
    pub dims: [u32; 3],
    /// Cell edge, metres.
    pub cell_m: f32,
    /// World position of the grid's min corner.
    pub origin: Vec3,
    /// Jacobi sweeps per step. Even; rounded up.
    pub jacobi_iters: u32,
    /// Soot lost per second (dilution below the grid scale).
    pub soot_decay_s: f32,
    /// Heat lost per second (mixing and radiation).
    pub cool_s: f32,
    /// Velocity relaxed toward the wind per second.
    pub damping_s: f32,
    /// Fuel burnt per second (first order).
    pub burn_s: f32,
    /// Temperature rise per unit of fuel burnt, K.
    pub heat_per_fuel_k: f32,
    /// Soot per unit of fuel burnt.
    pub soot_per_fuel: f32,
    /// Weight of soot against buoyancy, m/s² per unit soot.
    pub soot_weight: f32,
    /// Vorticity confinement strength.
    pub vorticity: f32,
    /// Speed cap, m/s.
    pub max_speed_m_s: f32,
    /// Random speed at the source, m/s.
    pub jitter_m_s: f32,
    /// Extinction per metre at soot 1.
    pub extinction_m: f32,
}

impl SmokeConfig {
    /// A grid centred on `centre` at ground level, `w` m wide and `h` m tall.
    pub fn around(centre: Vec3, w: f32, h: f32, cell_m: f32) -> Self {
        let nx = ((w / cell_m).round() as u32).max(8).next_multiple_of(4);
        let ny = ((h / cell_m).round() as u32).max(8).next_multiple_of(4);
        let half = nx as f32 * cell_m * 0.5;
        Self {
            dims: [nx, ny, nx],
            cell_m,
            origin: Vec3::new(centre.x - half, centre.y, centre.z - half),
            jacobi_iters: 24,
            soot_decay_s: 0.03,
            cool_s: 0.9,
            damping_s: 0.03,
            burn_s: 14.0,
            heat_per_fuel_k: 520.0,
            soot_per_fuel: 0.3,
            soot_weight: 0.4,
            vorticity: 6.0,
            max_speed_m_s: 6.0,
            jitter_m_s: 0.9,
            extinction_m: 7.0,
        }
    }

    pub fn cells(&self) -> usize {
        (self.dims[0] * self.dims[1] * self.dims[2]) as usize
    }

    pub fn size(&self) -> Vec3 {
        Vec3::new(
            self.dims[0] as f32,
            self.dims[1] as f32,
            self.dims[2] as f32,
        ) * self.cell_m
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuSource {
    pos: [f32; 4],
    rate: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    dims: [u32; 4],
    origin: [f32; 4],
    step: [f32; 4],
    decay: [f32; 4],
    wind: [f32; 4],
    sun: [f32; 4],
    combust: [f32; 4],
    sources: [GpuSource; MAX_SOURCES],
    count: [u32; 4],
}

/// Readback summary for tests and the bench log.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmokeStats {
    /// Soot summed over the grid times the cell volume, m³ at soot 1.
    pub soot_m3: f32,
    /// Soot-weighted centre, world metres.
    pub soot_centre: Vec3,
    /// Highest cell with soot above 1e-3, metres above the grid floor.
    pub top_m: f32,
    pub max_temp_k: f32,
    pub max_speed_m_s: f32,
    /// RMS divergence after the projection, 1/s.
    pub div_rms: f32,
    pub nan: u32,
}

/// The grid on the GPU. Shares the renderer's device.
pub struct GpuSmoke {
    pub config: SmokeConfig,
    pub sources: Vec<FireSource>,
    pub wind: Vec3,
    pub sun: Vec3,
    frame: u32,
    params: wgpu::Buffer,
    vel: [wgpu::Buffer; 2],
    scal: [wgpu::Buffer; 2],
    work: wgpu::Buffer,
    pressure: [wgpu::Buffer; 2],
    div: wgpu::Buffer,
    texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    #[allow(dead_code)]
    flow: wgpu::Texture,
    /// Velocity and unresolved eddy speed, same grid as `view`.
    pub flow_view: wgpu::TextureView,
    /// `[velocity phase][pressure phase]`.
    bind: [[wgpu::BindGroup; 2]; 2],
    pipes: Vec<wgpu::ComputePipeline>,
    /// Which of `vel`/`scal` holds the current state.
    phase: usize,
}

/// True when the device can run the kernels (compute + storage textures).
pub fn supported(device: &wgpu::Device) -> bool {
    let l = device.limits();
    l.max_storage_buffers_per_shader_stage >= 8 && l.max_storage_textures_per_shader_stage >= 2
}

impl GpuSmoke {
    pub fn new(device: &wgpu::Device, config: SmokeConfig) -> Self {
        let mut config = config;
        config.jacobi_iters = config.jacobi_iters.next_multiple_of(2).max(2);
        let n = config.cells() as u64;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("smoke"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let vel = [
            storage("smoke-vel-a", n * 16),
            storage("smoke-vel-b", n * 16),
        ];
        let scal = [
            storage("smoke-scal-a", n * 16),
            storage("smoke-scal-b", n * 16),
        ];
        let work = storage("smoke-work", n * 16);
        let pressure = [storage("smoke-p-a", n * 4), storage("smoke-p-b", n * 4)];
        let div = storage("smoke-div", n * 4);
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("smoke-params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke-volume"),
            size: wgpu::Extent3d {
                width: config.dims[0],
                height: config.dims[1],
                depth_or_array_layers: config.dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let flow = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("smoke-flow"),
            size: wgpu::Extent3d {
                width: config.dims[0],
                height: config.dims[1],
                depth_or_array_layers: config.dims[2],
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let flow_view = flow.create_view(&Default::default());

        let buf = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("smoke"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                buf(1, true),
                buf(2, false),
                buf(3, true),
                buf(4, false),
                buf(5, false),
                buf(6, true),
                buf(7, false),
                buf(8, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });
        let make = |v: usize, p: usize| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("smoke"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: vel[v].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vel[1 - v].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: scal[v].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: scal[1 - v].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: work.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: pressure[p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: pressure[1 - p].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: div.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: wgpu::BindingResource::TextureView(&flow_view),
                    },
                ],
            })
        };
        let bind = [[make(0, 0), make(0, 1)], [make(1, 0), make(1, 1)]];
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("smoke"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipes = ENTRIES
            .iter()
            .map(|e| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(e),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(e),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        Self {
            config,
            sources: Vec::new(),
            wind: Vec3::ZERO,
            sun: Vec3::new(0.3, 0.9, 0.2).normalize(),
            frame: 0,
            params,
            vel,
            scal,
            work,
            pressure,
            div,
            texture,
            view,
            flow,
            flow_view,
            bind,
            pipes,
            phase: 0,
        }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Clear the air to still, cold and clean.
    pub fn reset(&mut self, queue: &wgpu::Queue) {
        let n = self.config.cells();
        let zero4 = vec![0u8; n * 16];
        let zero1 = vec![0u8; n * 4];
        let wind = [self.wind.x, self.wind.y, self.wind.z, 0.0f32];
        let vel: Vec<[f32; 4]> = vec![wind; n];
        for b in &self.vel {
            queue.write_buffer(b, 0, bytemuck::cast_slice(&vel));
        }
        for b in &self.scal {
            queue.write_buffer(b, 0, &zero4);
        }
        queue.write_buffer(&self.work, 0, &zero4);
        for b in &self.pressure {
            queue.write_buffer(b, 0, &zero1);
        }
        queue.write_buffer(&self.div, 0, &zero1);
        self.frame = 0;
        self.phase = 0;
    }

    fn params(&self, dt: f32) -> Params {
        let c = &self.config;
        let mut sources = [GpuSource {
            pos: [0.0; 4],
            rate: [0.0; 4],
        }; MAX_SOURCES];
        for (dst, s) in sources.iter_mut().zip(self.sources.iter()) {
            dst.pos = [s.pos.x, s.pos.y, s.pos.z, s.radius];
            dst.rate = [s.fuel_s, s.soot_s, s.heat_k_s, s.up_m_s];
        }
        let sun = self.sun.normalize_or_zero();
        Params {
            dims: [c.dims[0], c.dims[1], c.dims[2], self.frame],
            origin: [c.origin.x, c.origin.y, c.origin.z, c.cell_m],
            step: [dt, GRAVITY / AMBIENT_K, c.soot_weight, c.vorticity],
            decay: [c.soot_decay_s, c.cool_s, c.damping_s, c.burn_s],
            wind: [self.wind.x, self.wind.y, self.wind.z, AMBIENT_K],
            sun: [sun.x, sun.y, sun.z, c.extinction_m],
            combust: [
                c.heat_per_fuel_k,
                c.soot_per_fuel,
                c.max_speed_m_s,
                c.jitter_m_s,
            ],
            sources,
            count: [self.sources.len().min(MAX_SOURCES) as u32, 0, 0, 0],
        }
    }

    /// Record one step into `encoder`. Writes the params now.
    pub fn encode_step(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        dt: f32,
    ) {
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.params(dt)));
        let d = self.config.dims;
        let groups = (d[0].div_ceil(4), d[1].div_ceil(4), d[2].div_ceil(4));
        let v = self.phase;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("smoke-sim"),
            timestamp_writes: None,
        });
        let run = |pass: &mut wgpu::ComputePass<'_>, k: usize, p: usize| {
            pass.set_pipeline(&self.pipes[k]);
            pass.set_bind_group(0, &self.bind[v][p], &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        };
        run(&mut pass, 0, 0); // advect
        run(&mut pass, 1, 0); // maccormack
        run(&mut pass, 2, 0); // curl
        run(&mut pass, 3, 0); // forces
        run(&mut pass, 4, 0); // divergence
        for it in 0..self.config.jacobi_iters as usize {
            run(&mut pass, 5, it % 2);
        }
        // Even sweep count: the result is back in pressure[0].
        run(&mut pass, 6, 0); // project
        run(&mut pass, 7, 0); // publish
        drop(pass);
        self.phase = 1 - self.phase;
        self.frame = self.frame.wrapping_add(1);
    }

    /// One step on its own submission.
    pub fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("smoke"),
        });
        self.encode_step(queue, &mut encoder, dt);
        queue.submit(Some(encoder.finish()));
    }

    /// Read the current state back. Slow; for tests and logs.
    pub fn stats(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> SmokeStats {
        let n = self.config.cells();
        let cur = self.phase;
        let vel = read_f32(device, queue, &self.vel[cur], n * 4);
        let scal = read_f32(device, queue, &self.scal[cur], n * 4);
        let div = read_f32(device, queue, &self.div, n);
        let c = &self.config;
        let [nx, ny, _] = c.dims.map(|v| v as usize);
        let vol = c.cell_m.powi(3);
        let mut st = SmokeStats::default();
        let mut wsum = 0.0f64;
        let mut centre = [0.0f64; 3];
        let mut top = 0usize;
        let mut d2 = 0.0f64;
        for i in 0..n {
            let s = &scal[i * 4..i * 4 + 4];
            let v = &vel[i * 4..i * 4 + 4];
            if s.iter().chain(v.iter()).any(|x| !x.is_finite()) {
                st.nan += 1;
                continue;
            }
            let x = i % nx;
            let y = (i / nx) % ny;
            let z = i / (nx * ny);
            let soot = s[0] as f64;
            wsum += soot;
            centre[0] += soot * x as f64;
            centre[1] += soot * y as f64;
            centre[2] += soot * z as f64;
            if s[0] > 1e-3 {
                top = top.max(y + 1);
            }
            st.max_temp_k = st.max_temp_k.max(s[1]);
            st.max_speed_m_s = st
                .max_speed_m_s
                .max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
            d2 += (div[i] as f64).powi(2);
        }
        st.soot_m3 = (wsum * vol as f64) as f32;
        if wsum > 0.0 {
            st.soot_centre = c.origin
                + (Vec3::new(
                    (centre[0] / wsum) as f32,
                    (centre[1] / wsum) as f32,
                    (centre[2] / wsum) as f32,
                ) + Vec3::splat(0.5))
                    * c.cell_m;
        }
        st.top_m = top as f32 * c.cell_m;
        st.div_rms = (d2 / n as f64).sqrt() as f32;
        st
    }

    /// Divergence of the current velocity after projection, recomputed.
    /// `stats().div_rms` reports the field before the last projection.
    pub fn divergence_after(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> f32 {
        let n = self.config.cells();
        let vel = read_f32(device, queue, &self.vel[self.phase], n * 4);
        let [nx, ny, nz] = self.config.dims.map(|v| v as i64);
        let h = self.config.cell_m;
        let at = |x: i64, y: i64, z: i64| -> [f32; 3] {
            let (y, m) = if y < 0 { (-1 - y, true) } else { (y, false) };
            if x < 0 || x >= nx || y >= ny || z < 0 || z >= nz {
                return [self.wind.x, self.wind.y, self.wind.z];
            }
            let i = (x + nx * (y + ny * z)) as usize * 4;
            [vel[i], if m { -vel[i + 1] } else { vel[i + 1] }, vel[i + 2]]
        };
        let mut d2 = 0.0f64;
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let d = (at(x + 1, y, z)[0] - at(x - 1, y, z)[0] + at(x, y + 1, z)[1]
                        - at(x, y - 1, z)[1]
                        + at(x, y, z + 1)[2]
                        - at(x, y, z - 1)[2])
                        / (2.0 * h);
                    d2 += (d as f64).powi(2);
                }
            }
        }
        (d2 / (nx * ny * nz) as f64).sqrt() as f32
    }
}

fn read_f32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, n: usize) -> Vec<f32> {
    let size = (n * 4) as u64;
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("smoke-readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &dst, 0, size);
    queue.submit(Some(encoder.finish()));
    let slice = dst.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map").expect("map smoke buffer");
    let out = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    dst.unmap();
    out
}

/// Linear sRGB of a black body at `kelvin`, scaled so the largest channel is 1,
/// and its radiant exitance relative to 1500 K (Stefan–Boltzmann, T⁴).
///
/// Planck's law integrated against the CIE 1931 2° observer. The observer is
/// the multi-lobe Gaussian fit of Wyman, Sloan and Shirley (JCGT 2013).
pub fn blackbody(kelvin: f32) -> ([f32; 3], f32) {
    let t = kelvin.max(300.0) as f64;
    let (mut x, mut y, mut z) = (0.0f64, 0.0f64, 0.0f64);
    let g = |l: f64, mu: f64, s1: f64, s2: f64| {
        let s = if l < mu { s1 } else { s2 };
        (-0.5 * ((l - mu) / s).powi(2)).exp()
    };
    let mut l = 380.0f64;
    while l <= 780.0 {
        let xb = 1.056 * g(l, 599.8, 37.9, 31.0) + 0.362 * g(l, 442.0, 16.0, 26.7)
            - 0.065 * g(l, 501.1, 20.4, 26.2);
        let yb = 0.821 * g(l, 568.8, 46.9, 40.5) + 0.286 * g(l, 530.9, 16.3, 31.1);
        let zb = 1.217 * g(l, 437.0, 11.8, 36.0) + 0.681 * g(l, 459.0, 26.0, 13.8);
        let lm = l * 1e-9;
        // Planck, constants folded: 2hc² / λ⁵ / (exp(hc / λkT) − 1).
        let b = 1.0 / (lm.powi(5) * ((1.438_776_9e-2 / (lm * t)).exp() - 1.0));
        x += xb * b;
        y += yb * b;
        z += zb * b;
        l += 5.0;
    }
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let gg = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    let rgb = [r.max(0.0), gg.max(0.0), b.max(0.0)];
    let m = rgb.iter().cloned().fold(1e-30, f64::max);
    let power = (t / 1500.0).powi(4) as f32;
    (
        [
            (rgb[0] / m) as f32,
            (rgb[1] / m) as f32,
            (rgb[2] / m) as f32,
        ],
        power,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackbody_goes_red_to_white() {
        let (low, p_low) = blackbody(900.0);
        let (mid, _) = blackbody(1500.0);
        let (hot, p_hot) = blackbody(6500.0);
        // Dull red: almost no blue.
        assert!(low[0] == 1.0 && low[2] < 0.01, "{low:?}");
        // Orange: green between, blue low.
        assert!(mid[1] > 0.1 && mid[1] < 0.7 && mid[2] < 0.2, "{mid:?}");
        // D65-ish: all channels high.
        assert!(hot.iter().all(|c| *c > 0.85), "{hot:?}");
        assert!(p_hot > p_low * 1000.0);
    }

    #[test]
    fn params_layout_is_256_bytes() {
        assert_eq!(std::mem::size_of::<Params>(), 256);
    }
}
