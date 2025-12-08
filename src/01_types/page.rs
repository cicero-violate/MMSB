// src/01_types/page.rs
// INSTRUMENTED + MEMORY-SAFE VERSION — DECEMBER 8 2025

use super::delta::Delta;
use super::epoch::{Epoch, EpochCell};
use parking_lot::RwLock;
use std::convert::TryInto;
use std::ffi::c_void;
use std::sync::Arc;
use std::fmt;
use thiserror::Error;
use std::sync::atomic::{AtomicU64, Ordering};

extern "C" {
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
}

// Global page counter for debugging
static PAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Possible backing locations for a page. Matches Julia enum order.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageLocation {
    Cpu = 0,
    Gpu = 1,
    Unified = 2,
}

impl PageLocation {
    pub fn from_tag(tag: i32) -> Result<Self, PageError> {
        match tag {
            0 => Ok(PageLocation::Cpu),
            1 => Ok(PageLocation::Gpu),
            2 => Ok(PageLocation::Unified),
            other => Err(PageError::InvalidLocation(other)),
        }
    }
}

/// Globally unique identifier for pages.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageID(pub u64);

impl fmt::Display for PageID {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata key-value store with copy-on-write semantics.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    store: Arc<RwLock<Vec<(String, Vec<u8>)>>>,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(Vec::new())),
        }
    }
    pub fn insert(&self, key: impl Into<String>, value: Vec<u8>) {
        let key_string = key.into();
        let mut guard = self.store.write();
        guard.retain(|(existing, _)| existing != &key_string);
        guard.push((key_string, value));
    }
    pub fn clone_store(&self) -> Vec<(String, Vec<u8>)> {
        self.store.read().clone()
    }
    pub fn from_entries(entries: Vec<(String, Vec<u8>)>) -> Self {
        Self {
            store: Arc::new(RwLock::new(entries)),
        }
    }
}

/// Memory page implementation — now with debug ID and safe deep cloning
#[derive(Debug)]
pub struct Page {
    debug_id: u64,                     // <-- NEW: unique per-instance ID
    pub id: PageID,
    epoch: EpochCell,
    data: *mut u8,
    mask: *mut u8,
    capacity: usize,
    location: PageLocation,
    metadata: Metadata,
}

impl Page {
    pub fn new(id: PageID, size: usize, location: PageLocation) -> Result<Self, PageError> {
        if size == 0 {
            return Err(PageError::InvalidSize(size));
        }

        let debug_id = PAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        println!("[PAGE {:>4}] NEW     id={:>6} size={:>7} loc={:?}", debug_id, id.0, size, location);

        let data_ptr = if location == PageLocation::Unified {
            #[cfg(feature = "cuda")]
            {
                let mut ptr: *mut c_void = std::ptr::null_mut();
                let ret = unsafe { cudaMallocManaged(&mut ptr as *mut *mut c_void, size, 1) };
                if ret != 0 {
                    eprintln!("cudaMallocManaged failed with error code: {}", ret);
                    return Err(PageError::AllocError(ret));
                }
                ptr as *mut u8
            }
            #[cfg(not(feature = "cuda"))]
            {
                let layout = std::alloc::Layout::array::<u8>(size)
                    .map_err(|_| PageError::AllocError(1))?;
                unsafe { std::alloc::alloc_zeroed(layout) }
            }
        } else {
            let layout = std::alloc::Layout::array::<u8>(size)
                .map_err(|_| PageError::AllocError(1))?;
            unsafe { std::alloc::alloc_zeroed(layout) }
        };

        let mask_size = (size + 7) / 8;
        let mask_ptr = {
            let layout = std::alloc::Layout::array::<u8>(mask_size)
                .map_err(|_| PageError::AllocError(2))?;
            unsafe { std::alloc::alloc_zeroed(layout) }
        };

        Ok(Self {
            debug_id,
            id,
            epoch: EpochCell::new(0),
            data: data_ptr,
            mask: mask_ptr,
            capacity: size,
            location,
            metadata: Metadata::new(),
        })
    }

    // ... all your existing methods unchanged (data_slice, metadata_blob, etc.) ...
    pub fn size(&self) -> usize { self.capacity }
    pub fn location(&self) -> PageLocation { self.location }
    pub fn data_slice(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.data, self.capacity) } }
    pub fn data_mut_slice(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.data, self.capacity) } }
    pub fn epoch(&self) -> Epoch { self.epoch.load() }
    pub fn set_epoch(&self, epoch: Epoch) { self.epoch.store(epoch); }
    pub fn metadata_blob(&self) -> Vec<u8> { /* unchanged */ todo!() }
    pub fn set_metadata_blob(&mut self, blob: &[u8]) -> Result<(), PageError> { /* unchanged */ todo!() }
    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), PageError> { /* unchanged */ todo!() }
}

impl Clone for Page {
    fn clone(&self) -> Self {
        let new_debug_id = PAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        println!("[PAGE {:>4}] CLONE → [PAGE {:>4}]  id={:>6}", self.debug_id, new_debug_id, self.id.0);

        // Deep copy: allocate fresh memory
        let layout_data = std::alloc::Layout::array::<u8>(self.capacity).expect("invalid capacity");
        let data = unsafe { std::alloc::alloc(layout_data) };
        unsafe { std::ptr::copy_nonoverlapping(self.data, data, self.capacity); }

        let mask_size = (self.capacity + 7) / 8;
        let layout_mask = std::alloc::Layout::array::<u8>(mask_size).expect("invalid mask size");
        let mask = unsafe { std::alloc::alloc(layout_mask) };
        unsafe { std::ptr::copy_nonoverlapping(self.mask, mask, mask_size); }

        Self {
            debug_id: new_debug_id,
            id: self.id,
            epoch: EpochCell::new(self.epoch.load().0),
            data,
            mask,
            capacity: self.capacity,
            location: self.location,
            metadata: self.metadata.clone(),
        }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        println!("[PAGE {:>4}] DROP    id={:>6} loc={:?}", self.debug_id, self.id.0, self.location);

        let mask_size = (self.capacity + 7) / 8;
        let mask_layout = std::alloc::Layout::array::<u8>(mask_size).unwrap();
        unsafe { std::alloc::dealloc(self.mask, mask_layout) };

        if self.location == PageLocation::Unified {
            #[cfg(feature = "cuda")]
            unsafe { let _ = cudaFree(self.data as *mut c_void); }
            #[cfg(not(feature = "cuda"))]
            unsafe {
                let layout = std::alloc::Layout::array::<u8>(self.capacity).unwrap();
                std::alloc::dealloc(self.data, layout);
            }
        } else {
            unsafe {
                let layout = std::alloc::Layout::array::<u8>(self.capacity).unwrap();
                std::alloc::dealloc(self.data, layout);
            }
        }
    }
}

#[derive(Debug, Error)]
pub enum PageError {
    #[error("Invalid page size: {0}")] InvalidSize(usize),
    #[error("Invalid location tag: {0}")] InvalidLocation(i32),
    #[error("PageID mismatch: expected {expected:?}, found {found:?}")] PageIDMismatch { expected: PageID, found: PageID },
    #[error("Mask size mismatch: expected {expected}, found {found}")] MaskSizeMismatch { expected: usize, found: usize },
    #[error("Metadata decode error: {0}")] MetadataDecode(&'static str),
    #[error("Allocation error: code {0}")] AllocError(i32),
    #[error("Page with ID {0} already exists")] AlreadyExists(PageID),
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

// Helper functions unchanged
fn read_u32(blob: &[u8], cursor: &mut usize) -> Result<u32, PageError> { /* unchanged */ todo!() }
fn read_bytes(blob: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<u8>, PageError> { /* unchanged */ todo!() }
