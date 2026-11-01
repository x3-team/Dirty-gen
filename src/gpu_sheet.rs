//! Spike: the height–velocity sheet step on wgpu compute.
//!
//! Same passes as `fluid_core::substep`, one thread per cell, whole grid.
//! Off by default. `SILT_SHEET_GPU=1` routes `FluidCore`'s water and slurry
//! sheets through here; everything else in the core stays on the CPU, so each
//! call uploads the sheet, runs the substeps, and reads it back.
//!
//! `SILT_SHEET_GPU_SOFTWARE=1` asks for a fallback adapter (lavapipe).

use std::time::Instant;

use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("gpu_sheet.wgsl");
const PLANES: usize = 13;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    res: u32,
    solid: u32,
    project: u32,
    pad: u32,
    cell: f32,
    dt: f32,
    vort: f32,
    eps: f32,
}

struct Grid {
    res: usize,
    h: wgpu::Buffer,
    u: wgpu::Buffer,
    v: wgpu::Buffer,
    stat: wgpu::Buffer,
    flags: wgpu::Buffer,
    params: wgpu::Buffer,
    readback: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

/// One sheet step's inputs. Slices are `res * res`.
pub struct SheetJob<'a> {
    pub h: &'a mut [f32],
    pub u: &'a mut [f32],
    pub v: &'a mut [f32],
    pub bed: &'a [f32],
    pub nu: &'a [f32],
    pub gate: &'a [u8],
    pub solid: Option<&'a [u8]>,
    pub res: usize,
    pub cell: f32,
    pub dt: f32,
    pub vorticity: f32,
    pub eps: f32,
    pub nsub: u32,
    pub project: bool,
}

/// Milliseconds of the last call, wall clock on the calling thread.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuTimes {
    pub upload_ms: f64,
    pub compute_ms: f64,
    pub readback_ms: f64,
}

pub struct GpuSheet {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipes: Vec<(&'static str, wgpu::ComputePipeline)>,
    layout: wgpu::BindGroupLayout,
    grids: Vec<Grid>,
    pub adapter_name: String,
    pub last: GpuTimes,
}

const ENTRIES: [&str; 19] = [
    "k_eta",
    "k_lift",
    "k_lift_copy",
    "k_blur_x_eta",
    "k_blur_z_p",
    "k_ghost",
    "k_gravity",
    "k_vort",
    "k_confine",
    "k_div",
    "k_blur_x_b",
    "k_blur_z_q",
    "k_vel",
    "k_redir_a",
    "k_redir_b",
    "k_copy_h2",
    "k_block",
    "k_tx",
    "k_tz",
];

impl GpuSheet {
    /// Headless device. None when no adapter is present.
    pub fn new(software: bool) -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: if software {
                wgpu::Backends::VULKAN
            } else {
                wgpu::Backends::all()
            },
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: software,
            compatible_surface: None,
        }))?;
        let adapter_name = format!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("silt-sheet"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("gpu_sheet device lost: {reason:?} {msg}");
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gpu_sheet"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gpu_sheet"),
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
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                storage(5),
                storage(6),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gpu_sheet"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let mut pipes = Vec::new();
        for name in ENTRIES.iter() {
            let p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(name),
                compilation_options: Default::default(),
                cache: None,
            });
            pipes.push((*name, p));
        }
        Some(Self {
            device,
            queue,
            pipes,
            layout,
            grids: Vec::new(),
            adapter_name,
            last: GpuTimes::default(),
        })
    }

    fn pipe(&self, name: &str) -> &wgpu::ComputePipeline {
        &self
            .pipes
            .iter()
            .find(|(n, _)| *n == name)
            .expect("gpu_sheet entry")
            .1
    }

    fn grid_index(&mut self, res: usize) -> usize {
        if let Some(k) = self.grids.iter().position(|g| g.res == res) {
            return k;
        }
        let n = res * res;
        let dev = &self.device;
        let mk = |label: &str, floats: usize| {
            dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (floats * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let h = mk("h", n);
        let u = mk("u", n);
        let v = mk("v", n);
        let stat = mk("stat", 2 * n);
        let flags = mk("flags", n);
        let scr = mk("scr", PLANES * n);
        let params = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&Params::zeroed()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let readback = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (3 * n * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = dev.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gpu_sheet"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: stat.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: flags.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: scr.as_entire_binding(),
                },
            ],
        });
        // scr only lives inside the bind group.
        drop(scr);
        self.grids.push(Grid {
            res,
            h,
            u,
            v,
            stat,
            flags,
            params,
            readback,
            bind,
        });
        self.grids.len() - 1
    }

    /// Upload the sheet and its inputs.
    pub fn upload(&mut self, job: &SheetJob) {
        let t = Instant::now();
        let k = self.grid_index(job.res);
        let n = job.res * job.res;
        let g = &self.grids[k];
        let q = &self.queue;
        q.write_buffer(&g.h, 0, bytemuck::cast_slice(&job.h[..n]));
        q.write_buffer(&g.u, 0, bytemuck::cast_slice(&job.u[..n]));
        q.write_buffer(&g.v, 0, bytemuck::cast_slice(&job.v[..n]));
        q.write_buffer(&g.stat, 0, bytemuck::cast_slice(&job.bed[..n]));
        q.write_buffer(&g.stat, (n * 4) as u64, bytemuck::cast_slice(&job.nu[..n]));
        let flags: Vec<u32> = (0..n)
            .map(|i| {
                let gate = (job.gate[i] != 0) as u32;
                let solid = job.solid.map(|s| (s[i] != 0) as u32).unwrap_or(0);
                gate | (solid << 1)
            })
            .collect();
        q.write_buffer(&g.flags, 0, bytemuck::cast_slice(&flags));
        self.last.upload_ms = t.elapsed().as_secs_f64() * 1e3;
    }

    /// Run `job.nsub` substeps on what is resident. Waits for the GPU.
    pub fn run(&mut self, job: &SheetJob) {
        let t = Instant::now();
        let k = self.grid_index(job.res);
        let solid = job.solid.is_some();
        let params = Params {
            res: job.res as u32,
            solid: solid as u32,
            project: job.project as u32,
            pad: 0,
            cell: job.cell,
            dt: job.dt / job.nsub.max(1) as f32,
            vort: job.vorticity,
            eps: job.eps,
        };
        let g = &self.grids[k];
        self.queue.write_buffer(&g.params, 0, bytemuck::bytes_of(&params));
        let groups = (job.res as u32).div_ceil(8);
        let mut seq: Vec<&str> = Vec::with_capacity(20);
        seq.push("k_eta");
        if solid {
            seq.extend(["k_lift", "k_lift_copy"]);
        }
        seq.extend(["k_blur_x_eta", "k_blur_z_p"]);
        if solid {
            seq.push("k_ghost");
        }
        seq.push("k_gravity");
        if job.vorticity > 0.0 {
            seq.extend(["k_vort", "k_confine"]);
        }
        if job.project {
            seq.extend(["k_div", "k_blur_x_b", "k_blur_z_q"]);
        }
        seq.push("k_vel");
        if solid {
            seq.extend(["k_redir_a", "k_redir_b", "k_copy_h2", "k_block", "k_ghost"]);
        }
        seq.extend(["k_tx", "k_tz"]);
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sheet") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("sheet"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &g.bind, &[]);
            for _ in 0..job.nsub.max(1) {
                for name in &seq {
                    pass.set_pipeline(self.pipe(name));
                    pass.dispatch_workgroups(groups, groups, 1);
                }
            }
        }
        self.queue.submit(Some(enc.finish()));
        self.device.poll(wgpu::Maintain::Wait);
        self.last.compute_ms = t.elapsed().as_secs_f64() * 1e3;
    }

    /// Copy h, u, v back into the job's slices.
    pub fn download(&mut self, job: &mut SheetJob) {
        let t = Instant::now();
        let k = self.grid_index(job.res);
        let n = job.res * job.res;
        let g = &self.grids[k];
        let bytes = (n * 4) as u64;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(&g.h, 0, &g.readback, 0, bytes);
        enc.copy_buffer_to_buffer(&g.u, 0, &g.readback, bytes, bytes);
        enc.copy_buffer_to_buffer(&g.v, 0, &g.readback, 2 * bytes, bytes);
        self.queue.submit(Some(enc.finish()));
        let slice = g.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        {
            let data = slice.get_mapped_range();
            let all: &[f32] = bytemuck::cast_slice(&data);
            job.h[..n].copy_from_slice(&all[..n]);
            job.u[..n].copy_from_slice(&all[n..2 * n]);
            job.v[..n].copy_from_slice(&all[2 * n..3 * n]);
        }
        g.readback.unmap();
        self.last.readback_ms = t.elapsed().as_secs_f64() * 1e3;
    }

    /// Upload, step, read back.
    pub fn step(&mut self, job: &mut SheetJob) {
        self.upload(job);
        self.run(job);
        self.download(job);
    }
}

use bytemuck::Zeroable;
