use super::delta::Delta;
use super::epoch::{Epoch, EpochCell};
use parking_lot::RwLock;
use std::convert::TryInto;
use std::ffi::c_void;
use std::sync::Arc;
use std::fmt;
use thiserror::Error;

extern "C" {
    fn cudaMallocManaged(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
    fn cudaFree(ptr: *mut c_void) -> i32;
}

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

/// Memory page implementation shared across the runtime layers.
#[derive(Debug)]
pub struct Page {
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

         // Use real cudaMallocManaged when Unified, fall back to Vec otherwise
         let data_ptr = if location == PageLocation::Unified {
             #[cfg(feature = "cuda")]
             {
                 let mut ptr = std::ptr::null_mut();
                 // cudaMemAttachGlobal = 1, cudaMemAttachHost = 2 — 1 is correct for multi-GPU access
                 let ret = unsafe { cudaMallocManaged(&mut ptr as *mut _ as *mut *mut c_void, size, 1) };
                 if ret != 0 {
                     return Err(PageError::AllocError(ret));
                 }
                 ptr as *mut u8
             }
             #[cfg(not(feature = "cuda"))]
             {
                 // Fallback when --no-default-features or CUDA not available
                 let layout = std::alloc::Layout::array::<u8>(size)
                     .map_err(|_| PageError::AllocError(1))?;
                 unsafe { std::alloc::alloc_zeroed(layout) }
             }
         } else {
             // CPU / GPU (non-unified) → always use regular allocator
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
             id,
             epoch: EpochCell::new(0),
             data: data_ptr,
             mask: mask_ptr,
             capacity: size,
             location,
             metadata: Metadata::new(),
         })
     }

    pub fn size(&self) -> usize {
        self.capacity
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
            return Vec::new(); // Just return owned empty vec
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
        if delta.page_id != self.id {
            return Err(PageError::PageIDMismatch {
                expected: self.id,
                found: delta.page_id,
            });
        }
        // Removed the size check to allow partial masks

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
                    let payload_i = i.min(delta.payload.len() - 1); // Use last if short
                    unsafe { *self.data.add(i) = delta.payload[payload_i]; }
                }
                // Mark dirty bit
                let mask_byte = unsafe { self.mask.add(i / 8) };
                unsafe { *mask_byte |= 1 << (i % 8); }
            }
        }
        self.epoch.store(delta.epoch);
        Ok(())
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        let mask_size = (self.capacity + 7) / 8;
        let mask_layout = std::alloc::Layout::array::<u8>(mask_size).unwrap();
        unsafe { std::alloc::dealloc(self.mask, mask_layout) };

        if self.location == PageLocation::Unified {
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = cudaFree(self.data as *mut c_void); // ignore error — we're dying anyway
            }
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

impl Clone for Page {
    fn clone(&self) -> Self {
        let mut new_page = Self::new(self.id, self.capacity, self.location).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(self.data, new_page.data, self.capacity);
            std::ptr::copy_nonoverlapping(self.mask, new_page.mask, (self.capacity + 7) / 8);
        }
        new_page.set_epoch(self.epoch());
        new_page.metadata = self.metadata.clone();
        new_page
    }
}

#[derive(Debug, Error)]
pub enum PageError {
    #[error("Invalid page size: {0}")]
    InvalidSize(usize),
    #[error("Invalid location tag: {0}")]
    InvalidLocation(i32),
    #[error("PageID mismatch: expected {expected:?}, found {found:?}")]
    PageIDMismatch { expected: PageID, found: PageID },
    #[error("Mask size mismatch: expected {expected}, found {found}")]
    MaskSizeMismatch { expected: usize, found: usize },
    #[error("Metadata decode error: {0}")]
    MetadataDecode(&'static str),
    #[error("Allocation error: code {0}")]
    AllocError(i32),
    #[error("Page with ID {0} already exists")]
    AlreadyExists(PageID),
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
