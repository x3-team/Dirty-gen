//! GPU timestamp queries, one begin/end pair per render pass.
//!
//! [`GpuTime::pass`] hands a render pass its timestamp writes and records the
//! pass label. After submit the resolved ticks go to one of three readback
//! buffers, so the window never waits on the GPU: [`GpuTime::poll`] returns
//! the newest finished frame, a frame or two late. [`GpuTime::read`] waits,
//! for the benchmark and `capture --gpu-time`.
//!
//! The frame time is the first pass begin to the last pass end. It does not
//! include present or the compositor.
//!
//! This tree encodes no pool reflect pass, so reflect milliseconds are never
//! printed.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use wgpu::AdapterInfo;

/// Passes timed per frame. Extra passes run untimed.
pub const MAX_PASSES: usize = 24;
const QUERY_COUNT: u32 = (MAX_PASSES * 2) as u32;
const QUERY_BYTES: u64 = 8;
const RESOLVE_BYTES: u64 = QUERY_COUNT as u64 * QUERY_BYTES;
const RING: usize = 3;

/// This tree does not encode `win-demo-pool-reflect`.
const REFLECT_PASS_ENCODED: bool = false;

pub const UNCOLLECTED_NOTE: &str = "GPU_TIME_NOTE milliseconds were not collected";

#[derive(Clone, Debug, Default)]
pub struct GpuSample {
    pub frame_ms: Option<f64>,
    /// Never set in this tree; see [`reflect_report`].
    #[allow(dead_code)]
    pub reflect_ms: Option<f64>,
    /// Milliseconds per labelled pass, in encode order.
    pub passes: Vec<(&'static str, f64)>,
}

impl GpuSample {
    pub fn empty() -> Self {
        Self::default()
    }
}

const IDLE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const READY: u8 = 3;
const FAILED: u8 = 4;

struct Slot {
    buf: wgpu::Buffer,
    labels: RefCell<Vec<&'static str>>,
    state: Arc<AtomicU8>,
    /// Frame counter at copy time, so `poll` returns the newest.
    frame: Cell<u64>,
}

pub struct GpuTime {
    query: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    ring: Vec<Slot>,
    period_ns: f32,
    labels: RefCell<Vec<&'static str>>,
    frame: Cell<u64>,
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
        let ring = (0..RING)
            .map(|_| Slot {
                buf: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu-time-read"),
                    size: RESOLVE_BYTES,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                labels: RefCell::new(Vec::new()),
                state: Arc::new(AtomicU8::new(IDLE)),
                frame: Cell::new(0),
            })
            .collect();
        Some(Self {
            query,
            resolve,
            ring,
            period_ns,
            labels: RefCell::new(Vec::with_capacity(MAX_PASSES)),
            frame: Cell::new(0),
        })
    }

    /// Timestamp writes for the next pass. `None` once [`MAX_PASSES`] passes
    /// are timed in this frame.
    pub fn pass(&self, label: &'static str) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let mut labels = self.labels.borrow_mut();
        if labels.len() >= MAX_PASSES {
            return None;
        }
        let i = labels.len() as u32;
        labels.push(label);
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.query,
            beginning_of_pass_write_index: Some(2 * i),
            end_of_pass_write_index: Some(2 * i + 1),
        })
    }

    /// Resolve this frame's queries into a free readback slot. A frame with
    /// no free slot (GPU three frames behind) is dropped, not waited on.
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut labels = self.labels.borrow_mut();
        if labels.is_empty() {
            return;
        }
        let Some(slot) = self
            .ring
            .iter()
            .find(|s| s.state.load(Ordering::Acquire) == IDLE)
        else {
            labels.clear();
            return;
        };
        let count = labels.len() as u32 * 2;
        encoder.resolve_query_set(&self.query, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            &slot.buf,
            0,
            count as u64 * QUERY_BYTES,
        );
        *slot.labels.borrow_mut() = std::mem::take(&mut *labels);
        self.frame.set(self.frame.get() + 1);
        slot.frame.set(self.frame.get());
        slot.state.store(COPIED, Ordering::Release);
    }

    /// Call after `queue.submit` of the encoder passed to [`GpuTime::resolve`].
    pub fn after_submit(&self) {
        for slot in &self.ring {
            if slot.state.load(Ordering::Acquire) != COPIED {
                continue;
            }
            slot.state.store(MAPPING, Ordering::Release);
            let state = slot.state.clone();
            slot.buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                state.store(if res.is_ok() { READY } else { FAILED }, Ordering::Release);
            });
        }
    }

    /// Newest finished frame without waiting. `None` while none is ready.
    pub fn poll(&self, device: &wgpu::Device) -> Option<GpuSample> {
        let _ = device.poll(wgpu::Maintain::Poll);
        self.collect()
    }

    /// Wait for every frame in flight and return the newest.
    pub fn read(&self, device: &wgpu::Device) -> GpuSample {
        silt::gpu_wait::wait_idle(&device);
        self.collect().unwrap_or_default()
    }

    fn collect(&self) -> Option<GpuSample> {
        let mut best: Option<(u64, GpuSample)> = None;
        for slot in &self.ring {
            match slot.state.load(Ordering::Acquire) {
                READY => {}
                FAILED => {
                    slot.state.store(IDLE, Ordering::Release);
                    continue;
                }
                _ => continue,
            }
            let sample = {
                let data = slot.buf.slice(..).get_mapped_range();
                let ticks = |index: usize| -> u64 {
                    let offset = index * QUERY_BYTES as usize;
                    u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap())
                };
                let labels = slot.labels.borrow();
                let mut passes = Vec::with_capacity(labels.len());
                let mut first = u64::MAX;
                let mut last = 0u64;
                for (i, label) in labels.iter().enumerate() {
                    let (b, e) = (ticks(2 * i), ticks(2 * i + 1));
                    if let Some(ms) = ticks_to_ms(b, e, self.period_ns) {
                        passes.push((*label, ms));
                        first = first.min(b);
                        last = last.max(e);
                    }
                }
                GpuSample {
                    frame_ms: if last > first {
                        ticks_to_ms(first, last, self.period_ns)
                    } else {
                        None
                    },
                    reflect_ms: None,
                    passes,
                }
            };
            slot.buf.unmap();
            slot.state.store(IDLE, Ordering::Release);
            let frame = slot.frame.get();
            if best.as_ref().map(|(f, _)| frame > *f).unwrap_or(true) {
                best = Some((frame, sample));
            }
        }
        best.map(|(_, s)| s)
    }
}

/// Timestamp writes for one pass when the clock is armed.
pub fn pass_timestamps<'a>(
    clock: Option<&'a GpuTime>,
    label: &'static str,
) -> Option<wgpu::RenderPassTimestampWrites<'a>> {
    clock?.pass(label)
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
