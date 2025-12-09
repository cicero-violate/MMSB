# Layer 0: Physical Memory

This directory houses the physical memory layer responsible for GPU/CPU allocation, device management, and synchronization primitives.

Contents:
- `allocator.rs` and `allocator_stats.rs` for allocation logic and metrics
- `device.rs`, `device_registry.rs`, and `host_device_sync.rs` for device enumeration and coordination
- Julia-facing APIs such as `PageAllocator.jl`, `UnifiedMemory.jl`, `DeviceSync.jl`, and `GPUKernels.jl`

Follow the project schedule DAG to ensure dependencies are respected when evolving this layer.
