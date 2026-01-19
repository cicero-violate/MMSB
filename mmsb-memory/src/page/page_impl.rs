use crate::delta::Delta;
use crate::epoch::{Epoch, EpochCell};
use mmsb_primitives::PageID;
use super::*;
use std::sync::atomic::{AtomicU64, Ordering};

static PAGE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct Page {
    debug_id: u64,
    id: PageID,
    epoch: EpochCell,
    data: *mut u8,
    capacity: usize,
    location: PageLocation,
}

impl Page {
    pub fn new(id: PageID, size: usize, location: PageLocation) -> Result<Self, PageError> {
        if size == 0 {
            return Err(PageError::InvalidSize(size));
        }

        let layout = std::alloc::Layout::array::<u8>(size).unwrap();
        let data = unsafe { std::alloc::alloc_zeroed(layout) };
        if data.is_null() {
            return Err(PageError::AllocationFailed);
        }

        Ok(Self {
            debug_id: PAGE_COUNTER.fetch_add(1, Ordering::Relaxed),
            id,
            epoch: EpochCell::new(0),
            data,
            capacity: size,
            location,
        })
    }
}

impl PageAccess for Page {
    fn id(&self) -> PageID { self.id }
    fn size(&self) -> usize { self.capacity }
    fn location(&self) -> PageLocation { self.location }

    fn epoch(&self) -> Epoch { self.epoch.load() }
    fn set_epoch(&self, epoch: Epoch) { self.epoch.store(epoch); }

    fn data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.capacity) }
    }

    fn data_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data, self.capacity) }
    }
}

impl DeltaAppliable for Page {
    fn apply_delta(&mut self, delta: &Delta) -> Result<(), PageError> {
        if delta.page_id != self.id {
            return Err(PageError::PageIDMismatch);
        }
        
        // Apply payload data
        let data_slice = self.data_mut_slice();
        
        // Verify payload fits within page
        if delta.payload.len() > data_slice.len() {
            return Err(PageError::InvalidSize(delta.payload.len()));
        }
        
        // Apply based on mask
        if delta.mask.len() != delta.payload.len() {
            return Err(PageError::MaskSizeMismatch);
        }
        
        for (i, &mask_bit) in delta.mask.iter().enumerate() {
            if i >= data_slice.len() {
                break;
            }
            if mask_bit {
                data_slice[i] = delta.payload[i];
            }
        }
        self.set_epoch(delta.epoch);
        Ok(())
    }
}

impl PageViewProvider for Page {
    fn view(&self) -> PageView {
        PageView {
            id: self.id,
            location: self.location,
            data: self.data,
            mask: std::ptr::null_mut(),
            len: self.capacity,
        }
    }
}

impl Drop for Page {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::array::<u8>(self.capacity).unwrap();
        unsafe { std::alloc::dealloc(self.data, layout) };
    }
}

unsafe impl Send for Page {}
unsafe impl Sync for Page {}
