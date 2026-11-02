//! Blocking GPU waits that survive a slow device.
//!
//! wgpu 23 `device.poll(Maintain::Wait)` waits at most 60 s for the fence and
//! ignores a timeout: it then frees the command buffers of the submission it
//! waited for and fires `map_async` callbacks as if the work had finished.
//! On lavapipe with a cold Mesa shader cache the first frame of the beauty
//! shaders takes longer than that. The command pool was reset under the
//! running queue thread (use-after-free, `capture --bench` segfault) and
//! readbacks copied a frame that was not drawn yet (black stills).
//!
//! `wait_idle` polls the fence without a deadline instead. `Maintain::Poll`
//! only retires submissions the fence reports as done.

#[cfg(not(target_arch = "wasm32"))]
pub fn wait_idle(device: &wgpu::Device) {
    let mut sleep_us = 50u64;
    loop {
        if device.poll(wgpu::Maintain::Poll).is_queue_empty() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_micros(sleep_us));
        sleep_us = (sleep_us * 2).min(20_000);
    }
}

/// The browser drives the queue itself; there is nothing to block on.
#[cfg(target_arch = "wasm32")]
pub fn wait_idle(device: &wgpu::Device) {
    let _ = device.poll(wgpu::Maintain::Poll);
}
