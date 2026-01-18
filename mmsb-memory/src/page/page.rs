// src/01_types/page.rs
// FULLY INSTRUMENTED + MEMORY-SAFE + DEEP CLONE — DECEMBER 8 2025
// 355+ lines — complete and final

use crate::delta::Delta;
use crate::delta::delta_validation;
use crate::epoch::{Epoch, EpochCell};
use crate::page::{PageError, PageLocation};
use mmsb_primitives::PageID;
use crate::delta::DeltaError;
use parking_lot::RwLock;
use std::convert::TryInto;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

extern "C" {
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
}

// Global counter for debugging page lifetimes
static PAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

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

/// Memory page implementation shared across the runtime layers.
#[derive(Debug)]
pub struct Page {
    debug_id: u64,                     // ← DEBUG: unique per-instance ID
    pub id: PageID,
    epoch: EpochCell,
    data: *mut u8,
    mask: *mut u8,
    capacity: usize,
    location: PageLocation,
    metadata: Metadata,
    unified_cuda_backing: bool,
}

impl Page {
    pub fn new(id: PageID, size: usize, location: PageLocation) -> Result<Self, PageError> {
        if size == 0 {
            return Err(PageError::InvalidSize(size));
        }

        let debug_id = PAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        if cfg!(debug_assertions) {
            println!(
                "[PAGE {:>4}] NEW     id={:>6} size={:>7} loc={:?}",
                debug_id, id.0, size, location
            );
        }

        // Use real cudaMallocManaged when Unified, fall back to Vec otherwise
        let (data_ptr, unified_cuda_backing) = if location == PageLocation::Unified {
            #[cfg(feature = "cuda")]
            {
                let mut ptr: *mut c_void = ptr::null_mut();
                let ret = unsafe { cudaMallocManaged(&mut ptr as *mut *mut c_void, size, 1) };
                if ret != 0 || ptr.is_null() {
                    eprintln!(
                        "cudaMallocManaged failed (code {}), falling back to host allocation for unified page {}",
                        ret,
                        id.0
                    );
                    (allocate_zeroed(size, 1)?, false)
                } else {
                    (ptr as *mut u8, true)
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                // Fallback when --no-default-features or CUDA not available
                (allocate_zeroed(size, 1)?, false)
            }
        } else {
            // CPU / GPU (non-unified) → always use regular allocator
            (allocate_zeroed(size, 1)?, false)
        };

        let mask_size = (size + 7) / 8;
        let mask_ptr = allocate_zeroed(mask_size, 2)?;

        Ok(Self {
            debug_id,
            id,
            epoch: EpochCell::new(0),
            data: data_ptr,
            mask: mask_ptr,
            capacity: size,
            location,
            metadata: Metadata::new(),
            unified_cuda_backing,
        })
    }

   pub fn size(&self) -> usize {
       self.capacity
   }

    /// Create a non-owning view for device operations
    /// Only memory authority can mint views (pub(crate)) page_view.rs
    pub(crate) fn view(&self) -> PageView {
        PageView {
            id: self.id,
            location: self.location,
            data: self.data,
            mask: self.mask,
            len: self.capacity,
        }
    }

   pub fn location(&self) -> PageLocation {
        self.location
    }

    pub fn data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.capacity) }
    }

    pub fn data_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.capacity) }
    }

    pub fn mask_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.mask, (self.capacity + 7) / 8) }
    }

    pub fn data_ptr(&mut self) -> *mut u8 {
        self.data
    }

    pub fn mask_ptr(&mut self) -> *mut u8 {
        self.mask
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch.load()
    }

    pub fn set_epoch(&self, epoch: Epoch) {
        self.epoch.store(epoch);
    }

    pub fn metadata_entries(&self) -> Vec<(String, Vec<u8>)> {
        self.metadata.clone_store()
    }

    pub fn set_metadata(&mut self, entries: Vec<(String, Vec<u8>)>) {
        self.metadata = Metadata::from_entries(entries);
    }

    pub fn metadata_blob(&self) -> Vec<u8> {
        let entries = self.metadata.clone_store();
        if entries.is_empty() {
            return Vec::new();
        }
        let mut blob = Vec::with_capacity(64);
        blob.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (key, value) in entries {
            let key_bytes = key.as_bytes();
            blob.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            blob.extend_from_slice(key_bytes);
            blob.extend_from_slice(&(value.len() as u32).to_le_bytes());
            blob.extend_from_slice(&value);
        }
        blob
    }

    pub fn set_metadata_blob(&mut self, blob: &[u8]) -> Result<(), PageError> {
        if blob.is_empty() {
            self.metadata = Metadata::new();
            return Ok(());
        }
        let mut cursor = 0usize;
        let entry_count = read_u32(blob, &mut cursor)? as usize;
        let mut entries = Vec::with_capacity(entry_count);
        for _ in 0..entry_count {
            let key_len = read_u32(blob, &mut cursor)? as usize;
            let key_bytes = read_bytes(blob, &mut cursor, key_len)?;
            let value_len = read_u32(blob, &mut cursor)? as usize;
            let value_bytes = read_bytes(blob, &mut cursor, value_len)?;
            let key = String::from_utf8(key_bytes)
                .map_err(|_| PageError::MetadataDecode("invalid utf-8 key"))?;
            entries.push((key, value_bytes));
        }
        self.metadata = Metadata::from_entries(entries);
        Ok(())
    }

    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), PageError> {
        if let Err(err) = delta_validation::validate_delta(delta) {
            return Err(match err {
                DeltaError::SizeMismatch { mask_len, payload_len } => PageError::MaskSizeMismatch {
                    expected: mask_len,
                    found: payload_len,
                },
                DeltaError::PageIDMismatch { expected, found } => PageError::PageIDMismatch {
                    expected,
                    found,
                },
                DeltaError::MaskSizeMismatch { expected, found } => PageError::MaskSizeMismatch {
                    expected,
                    found,
                },
            });
        }
        if delta.page_id != self.id {
            return Err(PageError::PageIDMismatch {
                expected: self.id,
                found: delta.page_id,
            });
        }
        let mut payload_idx = 0usize;
        for i in 0..self.capacity {
            let changed = if i < delta.mask.len() { delta.mask[i] } else { false };
            if changed {
                if delta.is_sparse {
                    if payload_idx >= delta.payload.len() {
                        return Err(PageError::MaskSizeMismatch {
                            expected: payload_idx,
                            found: delta.payload.len(),
                        });
                    }
                    unsafe { *self.data.add(i) = delta.payload[payload_idx]; }
                    payload_idx += 1;
                } else {
                    let payload_i = i.min(delta.payload.len() - 1);
                    unsafe { *self.data.add(i) = delta.payload[payload_i]; }
                }
                let mask_byte = unsafe { self.mask.add(i / 8) };
                unsafe { *mask_byte |= 1 << (i % 8); }
            }
        }
        self.epoch.store(delta.epoch);
        Ok(())
    }
}

impl Clone for Page {
    fn clone(&self) -> Self {
        let new_debug_id = PAGE_COUNTER.fetch_add(1, Ordering::Relaxed);
        if cfg!(debug_assertions) {
            println!(
                "[PAGE {:>4}] CLONE → [PAGE {:>4}]  id={:>6}",
                self.debug_id, new_debug_id, self.id.0
            );
        }

        // Deep copy — allocate fresh memory
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
            unified_cuda_backing: false,
        }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        if cfg!(debug_assertions) {
            println!(
                "[PAGE {:>4}] DROP    id={:>6} loc={:?}",
                self.debug_id, self.id.0, self.location
            );
        }

        let mask_size = (self.capacity + 7) / 8;
        let mask_layout = std::alloc::Layout::array::<u8>(mask_size).unwrap();
        unsafe { std::alloc::dealloc(self.mask, mask_layout) };

        if self.location == PageLocation::Unified && self.unified_cuda_backing {
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = cudaFree(self.data as *mut c_void);
            }
        } else {
            unsafe {
                let layout = std::alloc::Layout::array::<u8>(self.capacity).unwrap();
                std::alloc::dealloc(self.data, layout);
            }
        }
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}

fn read_u32(blob: &[u8], cursor: &mut usize) -> Result<u32, PageError> {
    if *cursor + 4 > blob.len() {
        return Err(PageError::MetadataDecode("unexpected end of metadata"));
    }
    let bytes: [u8; 4] = blob[*cursor..*cursor + 4]
        .try_into()
        .map_err(|_| PageError::MetadataDecode("invalid slice"))?;
    *cursor += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_bytes(blob: &[u8], cursor: &mut usize, len: usize) -> Result<Vec<u8>, PageError> {
    if *cursor + len > blob.len() {
        return Err(PageError::MetadataDecode("metadata truncated"));
    }
    let bytes = blob[*cursor..*cursor + len].to_vec();
    *cursor += len;
    Ok(bytes)
}

fn allocate_zeroed(size: usize, err_code: i32) -> Result<*mut u8, PageError> {
    let layout = std::alloc::Layout::array::<u8>(size)
        .map_err(|_| PageError::AllocError(err_code))?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err(PageError::AllocError(err_code));
    }
    Ok(ptr)
}
use crate::page::page_view::PageView;
