//! Post-processing prototype (docs/LOOK_GAP.md, item 1).
//!
//! The scene shaders still tone-map in place and write an 8-bit
//! `exp`-curve image (`present()` in shader.wgsl). This pass undoes that
//! curve to get an approximate linear HDR value back (up to about 5.8),
//! then runs a small film-style stack on it:
//!
//! 1. bloom: soft-threshold downsample to 1/4 resolution, separable blur
//! 2. FXAA-style edge blend (luma edges only, no temporal history)
//! 3. exposure + bloom + ACES fitted tone curve
//! 4. grade: saturation, split toning, vignette, light sharpen
//!
//! The 8-bit round trip limits the highlight range. The real fix is an
//! `Rgba16Float` scene target with the tone curve only here (LOOK_GAP
//! thread 1). This module is the drop-in proof that the stack alone moves
//! the picture, at a cost of one full-screen and three quarter-screen passes.
//!
//! Opt-in with `SILT_POST=1` while it is a prototype. Off by default, so the
//! golden frames and committed stills do not change.
//!
//! Sources (techniques, own code):
//! * ACES fitted curve: Stephen Hill's fit as published in MJP's BakingLab
//!   (MIT), `ACESFitted` with the RRT/ODT input and output matrices.
//! * Bloom downsample: the 13-tap pattern from Jimenez, "Next Generation Post
//!   Processing in Call of Duty: Advanced Warfare" (SIGGRAPH 2014 talk).
//! * Edge blend: the FXAA idea (Lottes 2009): luma contrast picks the edge
//!   direction, blend across it. Written from the paper, not the NVIDIA source.

use crate::gpu_time::{pass_timestamps, GpuTime};

const POST_WGSL: &str = include_str!("post.wgsl");
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// `SILT_POST=1` turns the prototype on. Anything else, or unset, is off.
pub fn enabled() -> bool {
    std::env::var("SILT_POST").map(|v| v.trim() == "1").unwrap_or(false)
}

struct Target {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

fn target(device: &wgpu::Device, label: &str, w: u32, h: u32, copy_src: bool) -> Target {
    let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    if copy_src {
        usage |= wgpu::TextureUsages::COPY_SRC;
    }
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    Target { tex, view }
}

pub struct Post {
    bloom: bool,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    pipe_down: wgpu::RenderPipeline,
    pipe_blur_h: wgpu::RenderPipeline,
    pipe_blur_v: wgpu::RenderPipeline,
    pipe_final: wgpu::RenderPipeline,
    out: Target,
    bloom_a: Target,
    bloom_b: Target,
    bg_down: wgpu::BindGroup,
    bg_blur_h: wgpu::BindGroup,
    bg_blur_v: wgpu::BindGroup,
    bg_final: wgpu::BindGroup,
}

impl Post {
    pub fn new(
        device: &wgpu::Device,
        quality: &silt::quality::Quality,
        width: u32,
        height: u32,
        src: &wgpu::TextureView,
    ) -> Self {
        // Bloom is the only part with a real cost. The low preset keeps the
        // tone curve, grade and edge blend, which are one full-screen pass.
        let bloom = quality.preset != silt::quality::Preset::Low;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("post"),
            source: wgpu::ShaderSource::Wgsl(POST_WGSL.into()),
        });
        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                tex_entry(0),
                tex_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let constants: std::collections::HashMap<String, f64> =
            [("POST_BLOOM".to_string(), if bloom { 1.0 } else { 0.0 })].into();
        let pipe = |entry: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_post"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &constants,
                        ..Default::default()
                    },
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview: None,
                cache: None,
            })
        };
        let pipe_down = pipe("fs_bloom_down");
        let pipe_blur_h = pipe("fs_bloom_blur_h");
        let pipe_blur_v = pipe("fs_bloom_blur_v");
        let pipe_final = pipe("fs_post");
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let (out, bloom_a, bloom_b) = Self::targets(device, width, height);
        let bgs = Self::bind_groups(device, &layout, &sampler, src, &out, &bloom_a, &bloom_b);
        Self {
            bloom,
            layout,
            sampler,
            pipe_down,
            pipe_blur_h,
            pipe_blur_v,
            pipe_final,
            out,
            bloom_a,
            bloom_b,
            bg_down: bgs.0,
            bg_blur_h: bgs.1,
            bg_blur_v: bgs.2,
            bg_final: bgs.3,
        }
    }

    fn targets(device: &wgpu::Device, w: u32, h: u32) -> (Target, Target, Target) {
        let (bw, bh) = (w.div_ceil(4), h.div_ceil(4));
        (
            target(device, "post-out", w, h, true),
            target(device, "post-bloom-a", bw, bh, false),
            target(device, "post-bloom-b", bw, bh, false),
        )
    }

    fn bind_groups(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
        src: &wgpu::TextureView,
        _out: &Target,
        a: &Target,
        b: &Target,
    ) -> (wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup) {
        let bg = |t0: &wgpu::TextureView, t1: &wgpu::TextureView, label| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(t0),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(t1),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            })
        };
        // Binding 1 of the bloom passes is unused; any view that is not the
        // render target will do.
        (
            bg(src, &b.view, "post-down"),
            bg(&a.view, src, "post-blur-h"),
            bg(&b.view, src, "post-blur-v"),
            bg(src, &a.view, "post-final"),
        )
    }

    pub fn resize(&mut self, device: &wgpu::Device, w: u32, h: u32, src: &wgpu::TextureView) {
        let (out, a, b) = Self::targets(device, w, h);
        let bgs = Self::bind_groups(device, &self.layout, &self.sampler, src, &out, &a, &b);
        self.out = out;
        self.bloom_a = a;
        self.bloom_b = b;
        self.bg_down = bgs.0;
        self.bg_blur_h = bgs.1;
        self.bg_blur_v = bgs.2;
        self.bg_final = bgs.3;
    }

    pub fn output(&self) -> &wgpu::Texture {
        &self.out.tex
    }

    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.out.view
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, clock: Option<&GpuTime>) {
        let run = |encoder: &mut wgpu::CommandEncoder,
                   label: &'static str,
                   view: &wgpu::TextureView,
                   pipe: &wgpu::RenderPipeline,
                   bg: &wgpu::BindGroup,
                   time: Option<wgpu::RenderPassTimestampWrites>| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: time,
                occlusion_query_set: None,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.draw(0..3, 0..1);
        };
        if self.bloom {
            run(
                encoder,
                "post-bloom",
                &self.bloom_a.view,
                &self.pipe_down,
                &self.bg_down,
                pass_timestamps(clock, "post-bloom"),
            );
            run(encoder, "post-bloom-h", &self.bloom_b.view, &self.pipe_blur_h, &self.bg_blur_h, None);
            run(encoder, "post-bloom-v", &self.bloom_a.view, &self.pipe_blur_v, &self.bg_blur_v, None);
        }
        run(
            encoder,
            "post",
            &self.out.view,
            &self.pipe_final,
            &self.bg_final,
            pass_timestamps(clock, "post"),
        );
    }
}
