use crate::delta::Delta;
use crate::epoch::{Epoch, EpochCell};
use crate::page::{PageError, PageLocation};
use mmsb_primitives::PageID;
use parking_lot::RwLock;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

static PAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Default)]
pub struct Metadata {
    store: Vec<(String, Vec<u8>)>,
}

impl Metadata {
    pub fn new() -> Self {
        Self { store: Vec::new() }
    }

    pub fn insert(&mut self, key: impl Into<String>, value: Vec<u8>) {
        let key = key.into();
        self.store.retain(|(k, _)| k != &key);
        self.store.push((key, value));
    }

    pub fn clone_store(&self) -> Vec<(String, Vec<u8>)> {
        self.store.clone()
    }
}

#[derive(Debug)]
pub struct Page {
    debug_id: u64,
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

        let data = unsafe {
            let layout = Layout::array::<u8>(size).map_err(|_| PageError::AllocError(-1))?;
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                return Err(PageError::AllocError(-2));
            }
            ptr
        };

        let mask_size = (size + 7) / 8;
        let mask = unsafe {
            let layout = Layout::array::<u8>(mask_size).map_err(|_| PageError::AllocError(-3))?;
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                dealloc(data, Layout::array::<u8>(size).unwrap());
                return Err(PageError::AllocError(-4));
            }
            ptr
        };

        Ok(Self {
            debug_id,
            id,
            epoch: EpochCell::new(0),
            data,
            mask,
            capacity: size,
            location,
            metadata: Metadata::new(),
            unified_cuda_backing: false,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn location(&self) -> PageLocation {
        self.location
    }

    pub fn epoch(&self) -> Epoch {
        self.epoch.load()
    }

    pub fn data_ptr(&self) -> *const u8 {
        self.data
    }

    pub fn data_mut_ptr(&mut self) -> *mut u8 {
        self.data
    }

    pub fn mask_ptr(&self) -> *const u8 {
        self.mask
    }

    pub fn mask_mut_ptr(&mut self) -> *mut u8 {
        self.mask
    }

    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    pub fn apply_delta(&mut self, delta: &Delta) -> Result<(), PageError> {
        // Minimal implementation – replace with your real logic
        if delta.payload.len() > self.capacity {
            return Err(PageError::InvalidSize(delta.payload.len()));
        }

        unsafe {
            ptr::copy_nonoverlapping(
                delta.payload.as_ptr(),
                self.data_mut_ptr(),
                delta.payload.len(),
            );
        }

        self.epoch.store(delta.epoch);
        Ok(())
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        unsafe {
            let data_layout = Layout::array::<u8>(self.capacity).unwrap();
            dealloc(self.data, data_layout);

            let mask_size = (self.capacity + 7) / 8;
            let mask_layout = Layout::array::<u8>(mask_size).unwrap();
            dealloc(self.mask, mask_layout);
        }
    }
}
