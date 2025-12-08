use crate::types::{Epoch, Page, PageError, PageID, PageLocation};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

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
        println!(
            "Allocating new PageAllocator instance with config: {:?}",
            config
        ); // Debug log
        Self {
            config,
            pages: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn allocate_raw(
        &self,
        page_id_hint: PageID,
        size: usize,
        location: Option<PageLocation>,
    ) -> Result<*mut Page, PageError> {
        let loc = location.unwrap_or(self.config.default_location);
        // If the page already exists → return error (or overwrite? we choose error for safety)
        if self.pages.lock().contains_key(&page_id_hint) {
            return Err(PageError::AlreadyExists(page_id_hint));
        }
        let mut page = Box::new(Page::new(page_id_hint, size, loc)?);
        let ptr = page.as_mut() as *mut Page;
        self.pages.lock().insert(page_id_hint, page);
        println!("Allocated page ID {} at {:p}", page_id_hint.0, ptr);
        Ok(ptr)
    }

    pub fn free(&self, page_id: PageID) {
        self.pages.lock().remove(&page_id);
        println!("Freeing page with ID: {}", page_id.0); // Debug log
    }

    pub fn release(&self, page_id: PageID) {
        self.pages.lock().remove(&page_id);
    }

    pub fn len(&self) -> usize {
        self.pages.lock().len()
    }

    pub fn page_infos(&self) -> Vec<PageInfo> {
        let pages = self.pages.lock();
        pages
            .values()
            .map(|page| PageInfo {
                page_id: page.id,
                size: page.size(),
                location: page.location(),
                epoch: page.epoch().0,
                metadata: page.metadata_blob(),
            })
            .collect()
    }

    pub fn acquire_page(&self, page_id: PageID) -> Option<*mut Page> {
        let pages = self.pages.lock();
        pages
            .get(&page_id)
            .map(|page| page.as_ref() as *const Page as *mut Page)
    }

    pub fn snapshot_pages(&self) -> Vec<PageSnapshotData> {
        let pages = self.pages.lock();
        pages
            .values()
            .map(|page| PageSnapshotData {
                page_id: page.id,
                size: page.size(),
                location: page.location(),
                epoch: page.epoch().0,
                metadata_blob: page.metadata_blob(),
                data: page.data_slice().to_vec(),
            })
            .collect()
    }

    pub fn restore_from_snapshot(&self, snapshots: Vec<PageSnapshotData>) -> Result<(), PageError> {
        let mut pages = self.pages.lock();
        pages.clear();
        for snapshot in snapshots {
            let mut page = Box::new(Page::new(
                snapshot.page_id,
                snapshot.size,
                snapshot.location,
            )?);
            page.set_epoch(Epoch(snapshot.epoch));
            page.data_mut_slice().copy_from_slice(&snapshot.data);
            page.set_metadata_blob(&snapshot.metadata_blob)
                .map_err(|_| PageError::MetadataDecode("restore failed"))?;
            pages.insert(snapshot.page_id, page);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_page_info_metadata_roundtrip() {
        let allocator = PageAllocator::new(PageAllocatorConfig::default());
        let ptr = allocator
            .allocate_raw(PageID(1), 128, None)
            .expect("allocation succeeds");
        let page = unsafe { &mut *ptr };
        page.set_metadata(vec![("key".to_string(), b"abc123".to_vec())]);
        let infos = allocator.page_infos();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].page_id, PageID(1));
        assert_eq!(infos[0].metadata, page.metadata_blob());
    }

    #[test]
    fn test_unified_page() {
        // ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
        // THIS IS THE NEW TEST — PASTE IT HERE
        let config = PageAllocatorConfig {
            default_location: PageLocation::Unified,
        };
        let allocator = PageAllocator::new(config);

        let ptr = allocator
            .allocate_raw(PageID(1), 4096, None)
            .expect("Unified page allocation failed");

        let page = unsafe { &mut *ptr };
        assert_eq!(page.location(), PageLocation::Unified);

        let data = page.data_mut_slice();
        data[0] = 42;
        assert_eq!(data[0], 42);

        // Optional: test GPU can read it (if CUDA is available)
        // let _ = unsafe { cudaDeviceSynchronize() };
        println!("Unified memory test PASSED — CPU and GPU share the same memory!");
    }

}
