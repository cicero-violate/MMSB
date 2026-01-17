// use crate::physical::AllocatorStats;
use crate::page::{PageError, PageLocation, Page};
use mmsb_primitives::PageID;
use crate::epoch::Epoch;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;


extern "C" {
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
}

#[derive(Debug, Clone)]
pub struct PageAllocatorConfig {
    pub default_location: PageLocation,
}

impl Default for PageAllocatorConfig {
    fn default() -> Self {
        Self {
            default_location: PageLocation::Cpu,
        }
    }
}

#[derive(Debug)]
pub struct PageAllocator {
    config: PageAllocatorConfig,
    pages: Mutex<HashMap<PageID, Box<Page>>>,
    next_id: AtomicU64,
    stats: Arc<AllocatorStats>,
}

#[derive(Debug, Clone)]
pub struct PageInfo {
    pub page_id: PageID,
    pub size: usize,
    pub location: PageLocation,
    pub epoch: u32,
    pub metadata: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PageSnapshotData {
    pub page_id: PageID,
    pub size: usize,
    pub location: PageLocation,
    pub epoch: u32,
    pub metadata_blob: Vec<u8>,
    pub data: Vec<u8>,
}

impl PageAllocator {
    pub fn new(config: PageAllocatorConfig) -> Self {
        Self::with_stats(config, Arc::new(AllocatorStats::default()))
    }

    pub fn with_stats(config: PageAllocatorConfig, stats: Arc<AllocatorStats>) -> Self {
        if cfg!(debug_assertions) {
            println!(
                "Allocating new PageAllocator instance with config: {:?}",
                config
            );
        }
        Self {
            config,
            pages: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            stats,
        }
    }

    // ... (rest continues with allocate_raw, free, release, snapshot_pages, restore_from_snapshot, etc.)
    // truncated in your paste at ~3140 characters
}
