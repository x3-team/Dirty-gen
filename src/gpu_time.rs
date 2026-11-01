//! GPU timestamp queries for one frame.
//!
//! Query 0 is the start of the shadow pass. Query 1 is the end of the last
//! pass in the frame. Queries 2 and 3 are reserved for a pool reflect pass.
//! This tree does not encode that pass, so reflect milliseconds are never
//! printed.

use std::cell::Cell;

use wgpu::AdapterInfo;

pub const Q_FRAME_BEGIN: u32 = 0;
pub const Q_FRAME_END: u32 = 1;
/// Pool reflect pass. Unused: this tree does not encode one.
#[allow(dead_code)]
pub const Q_REFLECT_BEGIN: u32 = 2;
/// Pool reflect pass. Unused: this tree does not encode one.
#[allow(dead_code)]
pub const Q_REFLECT_END: u32 = 3;
const QUERY_COUNT: u32 = 4;
const QUERY_BYTES: u64 = 8;
const RESOLVE_BYTES: u64 = 256;

/// This tree does not encode `win-demo-pool-reflect`.
const REFLECT_PASS_ENCODED: bool = false;

pub const UNCOLLECTED_NOTE: &str = "GPU_TIME_NOTE milliseconds were not collected";

pub struct GpuSample {
    pub frame_ms: Option<f64>,
    /// Set only when a reflect pass wrote queries 2 and 3. Nothing in this
    /// tree does that.
    #[allow(dead_code)]
    pub reflect_ms: Option<f64>,
}

impl GpuSample {
    pub fn empty() -> Self {
        Self {
            frame_ms: None,
            reflect_ms: None,
        }
    }
}

pub struct GpuTime {
    query: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    period_ns: f32,
    frame_written: Cell<bool>,
    reflect_written: Cell<bool>,
}

impl GpuTime {
    pub fn try_new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let period_ns = queue.get_timestamp_period();
        if !period_ns.is_finite() || period_ns <= 0.0 {
            return None;
        }
        let query = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu-time"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERY_COUNT,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-time-resolve"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu-time-read"),
            size: RESOLVE_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Some(Self {
            query,
            resolve,
            readback,
            period_ns,
            frame_written: Cell::new(false),
            reflect_written: Cell::new(false),
        })
    }

    pub fn note_frame(&self) {
        self.frame_written.set(true);
    }

    /// The pool reflect pass wrote queries 2 and 3 in this frame.
    #[allow(dead_code)]
    pub fn mark_reflect(&self) {
        self.reflect_written.set(true);
    }

    #[allow(dead_code)]
    pub fn reflect_writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query,
            beginning_of_pass_write_index: Some(Q_REFLECT_BEGIN),
            end_of_pass_write_index: Some(Q_REFLECT_END),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        // Unwritten queries are not resolved. Reflect slots stay out until
        // [`GpuTime::mark_reflect`] says the pass wrote them.
        let count = if self.reflect_written.get() {
            QUERY_COUNT
        } else if self.frame_written.get() {
            2
        } else {
            return;
        };
        encoder.resolve_query_set(&self.query, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            &self.readback,
            0,
            count as u64 * QUERY_BYTES,
        );
    }

    pub fn read(&self, device: &wgpu::Device) -> GpuSample {
        if !self.frame_written.get() {
            return GpuSample::empty();
        }
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        let _ = device.poll(wgpu::Maintain::Wait);
        if rx.recv().ok().and_then(|res| res.ok()).is_none() {
            self.frame_written.set(false);
            self.reflect_written.set(false);
            return GpuSample::empty();
        }
        let frame_ms;
        let reflect_ms;
        {
            let data = slice.get_mapped_range();
            let ticks = |index: usize| -> u64 {
                let offset = index * QUERY_BYTES as usize;
                u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
            };
            frame_ms = ticks_to_ms(ticks(0), ticks(1), self.period_ns);
            reflect_ms = if self.reflect_written.get() {
                ticks_to_ms(ticks(2), ticks(3), self.period_ns)
            } else {
                None
            };
        }
        self.readback.unmap();
        self.frame_written.set(false);
        self.reflect_written.set(false);
        GpuSample {
            frame_ms,
            reflect_ms,
        }
    }
}

pub fn pass_timestamps<'a>(
    clock: Option<&'a GpuTime>,
    begin: Option<u32>,
    end: Option<u32>,
) -> Option<wgpu::RenderPassTimestampWrites<'a>> {
    let clock = clock?;
    if begin.is_none() && end.is_none() {
        return None;
    }
    Some(wgpu::RenderPassTimestampWrites {
        query_set: &clock.query,
        beginning_of_pass_write_index: begin,
        end_of_pass_write_index: end,
    })
}

fn ticks_to_ms(begin: u64, end: u64, period_ns: f32) -> Option<f64> {
    if end <= begin || !period_ns.is_finite() || period_ns <= 0.0 {
        return None;
    }
    let ms = (end - begin) as f64 * f64::from(period_ns) / 1_000_000.0;
    if !ms.is_finite() || ms <= 0.0 || ms > 60_000.0 {
        None
    } else {
        Some(ms)
    }
}

pub fn is_hardware_gpu(info: &AdapterInfo) -> bool {
    let kind_ok = matches!(
        info.device_type,
        wgpu::DeviceType::DiscreteGpu
            | wgpu::DeviceType::IntegratedGpu
            | wgpu::DeviceType::VirtualGpu
    );
    kind_ok && !software_adapter_name(&info.name)
}

fn software_adapter_name(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    [
        "llvmpipe",
        "lavapipe",
        "swiftshader",
        "softpipe",
        "microsoft basic render",
        "warp",
        "virgl",
        "virtualbox",
        "vmware",
        "software adapter",
        "software raster",
    ]
    .iter()
    .any(|marker| name.contains(marker))
}

pub fn frame_uncollected(reason: &str) -> String {
    format!("GPU_FRAME collected=0 reason={reason}")
}

pub fn reflect_uncollected(reason: &str) -> String {
    format!("GPU_REFLECT collected=0 reason={reason}")
}

/// Frame and reflect lines for a run that must not print a millisecond number.
pub fn early_report(reason: &str) -> [String; 3] {
    [
        frame_uncollected(reason),
        reflect_uncollected(reason),
        UNCOLLECTED_NOTE.to_string(),
    ]
}

/// Reflect line for this tree. The pass is not encoded, so a sample is ignored
/// and the line stays `collected=0`.
pub fn reflect_report(ms: Option<f64>) -> String {
    if !REFLECT_PASS_ENCODED {
        return reflect_uncollected("no-reflect-pass");
    }
    match ms {
        Some(ms) if ms.is_finite() && ms > 0.0 => format!("GPU_REFLECT ms={ms:.3}"),
        _ => reflect_uncollected("timestamp-unread"),
    }
}

pub fn frame_milliseconds(mean: f64, samples: usize) -> String {
    format!("GPU_FRAME ms={mean:.3} samples={samples}")
}

pub fn host_cpu_line() -> String {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    format!("{}, {cores} cores", cpu_model())
}

fn cpu_model() -> String {
    #[cfg(target_os = "linux")]
    {
        return linux_cpu();
    }
    #[cfg(target_os = "windows")]
    {
        return windows_cpu();
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        "unknown cpu".to_string()
    }
}

#[cfg(target_os = "linux")]
fn linux_cpu() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let rest = line.strip_prefix("model name")?;
                rest.split(':').nth(1).map(|name| name.trim().to_string())
            })
        })
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown cpu".to_string())
}

#[cfg(target_os = "windows")]
fn windows_cpu() -> String {
    let output = std::process::Command::new("reg")
        .args([
            "query",
            r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            "/v",
            "ProcessorNameString",
        ])
        .output();
    let Ok(output) = output else {
        return "unknown cpu".to_string();
    };
    let text = String::from_utf8_lossy(&output.stdout);
    for line in text.lines() {
        if let Some(rest) = line.split("REG_SZ").nth(1) {
            let name = rest.trim();
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    "unknown cpu".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_millisecond_number(text: &str) -> bool {
        text.contains("ms=")
    }

    fn adapter(name: &str, device_type: wgpu::DeviceType) -> AdapterInfo {
        AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Dx12,
        }
    }

    #[test]
    fn basic_render_driver_is_not_hardware() {
        let info = adapter(
            "Microsoft Basic Render Driver",
            wgpu::DeviceType::IntegratedGpu,
        );
        assert!(!is_hardware_gpu(&info));
        assert!(!is_hardware_gpu(&adapter("WARP", wgpu::DeviceType::Cpu)));
        assert!(!is_hardware_gpu(&adapter("llvmpipe", wgpu::DeviceType::Cpu)));
        assert!(!is_hardware_gpu(&adapter(
            "NVIDIA GeForce",
            wgpu::DeviceType::Cpu
        )));
        assert!(is_hardware_gpu(&adapter(
            "NVIDIA GeForce RTX 4090",
            wgpu::DeviceType::DiscreteGpu
        )));
    }

    #[test]
    fn software_and_missing_lines_have_no_millisecond_number() {
        for reason in [
            "no-adapter",
            "software-adapter",
            "no-timestamp-query",
            "timestamp-unread",
        ] {
            for line in early_report(reason) {
                assert!(!has_millisecond_number(&line), "{line}");
                if line.starts_with("GPU_") && !line.starts_with("GPU_TIME_NOTE") {
                    assert!(line.contains("collected=0"), "{line}");
                    let digits: String = line.chars().filter(|c| c.is_ascii_digit()).collect();
                    assert_eq!(digits, "0", "{line}");
                }
            }
        }
        let reflect = reflect_report(Some(12.5));
        assert!(reflect.contains("collected=0"), "{reflect}");
        assert!(!has_millisecond_number(&reflect), "{reflect}");
        assert!(!reflect.contains("12.5"), "{reflect}");
        assert!(!REFLECT_PASS_ENCODED);
    }
}
