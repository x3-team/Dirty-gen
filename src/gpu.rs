//! Upload the dust scalar as a 3D `r32float` texture.
//!
//! The renderer binds this texture. Nothing here raymarches it.

use crate::dust::DensityTexture;

pub struct GpuDensity {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub texture: wgpu::Texture,
    pub format: wgpu::TextureFormat,
    pub adapter_name: String,
    nx: u32,
    ny: u32,
    nz: u32,
    row_bytes: u32,
}

impl GpuDensity {
    pub fn extent(&self) -> wgpu::Extent3d {
        self.texture.size()
    }

    /// Read the 3D texture back to kilograms per cubic metre, x-fastest.
    pub fn read_density(&self) -> Result<Vec<f32>, String> {
        let n = (self.nx * self.ny * self.nz) as usize;
        let row = self.row_bytes as u64;
        let size = row * self.ny as u64 * self.nz as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dust-density-readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("dust-density-readback"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.row_bytes),
                    rows_per_image: Some(self.ny),
                },
            },
            self.texture.size(),
        );
        self.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|err| err.to_string())?
            .map_err(|err| format!("{err:?}"))?;
        let view = slice.get_mapped_range();
        let mut out = vec![0.0f32; n];
        let row_floats = (self.row_bytes / 4) as usize;
        let nx = self.nx as usize;
        let ny = self.ny as usize;
        let nz = self.nz as usize;
        for iz in 0..nz {
            for iy in 0..ny {
                let row_off = ((iz * ny + iy) * row_floats) * 4;
                for ix in 0..nx {
                    let b = row_off + ix * 4;
                    let bytes: [u8; 4] = view[b..b + 4]
                        .try_into()
                        .map_err(|_| "short readback".to_string())?;
                    out[ix + iy * nx + iz * nx * ny] = f32::from_le_bytes(bytes);
                }
            }
        }
        drop(view);
        staging.unmap();
        Ok(out)
    }
}

/// Create a sampled 3D texture and copy `tex.samples` into it.
pub fn upload_density_texture(tex: &DensityTexture) -> Result<GpuDensity, String> {
    ensure_software_icd_if_headless();
    if tex.nx == 0 || tex.ny == 0 || tex.nz == 0 {
        return Err("empty density texture".into());
    }
    let n = (tex.nx * tex.ny * tex.nz) as usize;
    if tex.samples.len() != n {
        return Err(format!("sample count {} != {}", tex.samples.len(), n));
    }
    let row_bytes = align_256(tex.nx * 4);
    let mut desc = wgpu::InstanceDescriptor::default();
    desc.backends = wgpu::Backends::VULKAN;
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "no Vulkan adapter".to_string())?;
    let info = adapter.get_info();
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("dust-density"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
        },
        None,
    ))
    .map_err(|err| err.to_string())?;
    let format = wgpu::TextureFormat::R32Float;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("dust-density"),
        size: wgpu::Extent3d {
            width: tex.nx,
            height: tex.ny,
            depth_or_array_layers: tex.nz,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let _view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("dust-density"),
        format: Some(format),
        dimension: Some(wgpu::TextureViewDimension::D3),
        ..Default::default()
    });
    let bytes = pack_rows(&tex.samples, tex.nx, tex.ny, tex.nz, row_bytes);
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(row_bytes),
            rows_per_image: Some(tex.ny),
        },
        texture.size(),
    );
    device.poll(wgpu::Maintain::Wait);
    Ok(GpuDensity {
        device,
        queue,
        texture,
        format,
        adapter_name: format!("{} ({:?})", info.name, info.backend),
        nx: tex.nx,
        ny: tex.ny,
        nz: tex.nz,
        row_bytes,
    })
}

fn align_256(n: u32) -> u32 {
    (n + 255) & !255
}

fn pack_rows(samples: &[f32], nx: u32, ny: u32, nz: u32, row_bytes: u32) -> Vec<u8> {
    let row_floats = (row_bytes / 4) as usize;
    let mut bytes = vec![0u8; row_floats * ny as usize * nz as usize * 4];
    let nx = nx as usize;
    let ny = ny as usize;
    let nz = nz as usize;
    for iz in 0..nz {
        for iy in 0..ny {
            let dst = (iz * ny + iy) * row_floats * 4;
            for ix in 0..nx {
                let rho = samples[ix + iy * nx + iz * nx * ny];
                let b = rho.to_le_bytes();
                let o = dst + ix * 4;
                bytes[o..o + 4].copy_from_slice(&b);
            }
        }
    }
    bytes
}

fn ensure_software_icd_if_headless() {
    if std::env::var_os("VK_ICD_FILENAMES").is_some() {
        return;
    }
    let dri = std::path::Path::new("/dev/dri");
    let has_dri = dri
        .read_dir()
        .map(|mut d| d.next().is_some())
        .unwrap_or(false);
    if has_dri {
        return;
    }
    let icd = "/usr/share/vulkan/icd.d/lvp_icd.json";
    if std::path::Path::new(icd).exists() {
        std::env::set_var("VK_ICD_FILENAMES", icd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dust::{drive, proof_wind, DensityTexture, PROOF_DT_S};

    #[test]
    fn renderer_receives_a_3d_density_texture() {
        let (dust, _) = drive(8.0, 1.6, 0.04, 0.0, proof_wind(), 0.4, PROOF_DT_S);
        let mut tex = DensityTexture::empty();
        dust.publish(&mut tex);
        let gpu = upload_density_texture(&tex).expect("upload 3D density");
        assert_eq!(gpu.texture.dimension(), wgpu::TextureDimension::D3);
        assert_eq!(gpu.format, wgpu::TextureFormat::R32Float);
        let size = gpu.extent();
        assert_eq!(size.width, 128);
        assert_eq!(size.height, tex.ny);
        assert_eq!(size.depth_or_array_layers, 128);
        let back = gpu.read_density().expect("readback");
        assert_eq!(back.len(), tex.samples.len());
        let mut max_i = 0usize;
        let mut max_r = 0.0f32;
        for (i, rho) in tex.samples.iter().enumerate() {
            if *rho > max_r {
                max_r = *rho;
                max_i = i;
            }
        }
        assert!(max_r > 0.0, "proof texture is empty");
        let err = (back[max_i] - tex.samples[max_i]).abs();
        assert!(err <= 1.0e-6 * max_r.max(1.0), "readback {err}");
        assert_eq!(back[0].to_bits(), tex.samples[0].to_bits());
    }
}
