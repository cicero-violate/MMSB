//! MMSB Device
//!
//! Manages device-specific buffer registries and synchronization.
//! Works with mmsb-allocator for physical memory management.
//!
//! According to canonical dependencies:
//! - Depends on: mmsb-primitives, mmsb-allocator
//! - Authority: NONE (device management only)

pub mod device;
pub mod device_registry;
pub mod host_device_sync;
pub mod simd_mask;
pub mod integrity_checker;

pub use device_registry::DeviceBufferRegistry;
pub use host_device_sync::HostDeviceSync;
