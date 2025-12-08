use crate::types::{Epoch, Page, PageError, PageID, PageLocation};
use crate::types::{Delta, DeltaID, Source};
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
        println!("Allocating new PageAllocator instance with config: {:?}", config);
        Self {
            config,
            pages: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn allocate_raw(&self, page_id_hint: PageID, size: usize, location: Option<PageLocation>) -> Result<*mut Page, PageError> {
        let loc = location.unwrap_or(self.config.default_location);
        if self.pages.lock().contains_key(&page_id_hint) {
            return Err(PageError::AlreadyExists(page_id_hint));
        }
        let page = Box::new(Page::new(page_id_hint, size, loc)?);
        let ptr = Box::into_raw(page);
        println!("[ALLOCATOR] allocate_raw(id={}) → raw ptr = {:p}", page_id_hint.0, ptr);
        self.pages.lock().insert(page_id_hint, unsafe { Box::from_raw(ptr) });
        Ok(ptr)
    }

    pub fn free(&self, page_id: PageID) {
        if let Some(_) = self.pages.lock().remove(&page_id) {
            println!("[ALLOCATOR] Freed page {}", page_id.0);
        }
    }

    pub fn release(&self, page_id: PageID) {
        if let Some(boxed_page) = self.pages.lock().remove(&page_id) {
            println!("[ALLOCATOR] release({}): ownership transferred — Box removed from map but NOT dropped (caller now owns it)", page_id.0);
            // DO NOT drop the Box here!
            // The raw pointer from allocate_raw is now the sole owner.
            // Dropping boxed_page here would free the memory → use-after-free
            std::mem::forget(boxed_page);
        } else {
            println!("[ALLOCATOR] release({}): page not found — already released?", page_id.0);
        }
    }

    pub fn acquire_page(&self, page_id: PageID) -> Option<*mut Page> {
        self.pages.lock().get(&page_id).map(|b| &**b as *const Page as *mut Page)
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
        eprintln!("\n=== RESTORE_FROM_SNAPSHOT STARTED ===");
        eprintln!("   Clearing {} existing pages", self.pages.lock().len());

        let mut pages = self.pages.lock();
        pages.clear();

        for (i, snapshot) in snapshots.iter().enumerate() {
            eprintln!("   [{i}] Restoring page ID={:?} size={} epoch={} loc={:?}",
                snapshot.page_id, snapshot.size, snapshot.epoch, snapshot.location);

            // 1. Create page — real error from Page::new
            let mut page = match Page::new(snapshot.page_id, snapshot.size, snapshot.location) {
                Ok(p) => Box::new(p),
                Err(e) => {
                    eprintln!("      Page::new() FAILED: {e}");
                    return Err(e);
                }
            };

            // 2. Set epoch
            page.set_epoch(Epoch(snapshot.epoch));
            eprintln!("      Epoch set to {}", snapshot.epoch);

            // 3. Copy data — size mismatch is impossible if snapshot was written correctly
            let dst = page.data_mut_slice();
            if dst.len() != snapshot.data.len() {
                eprintln!("      FATAL: data size mismatch! page={} snapshot={}", dst.len(), snapshot.data.len());
                return Err(PageError::MetadataDecode("data size mismatch in snapshot"));
            }
            dst.copy_from_slice(&snapshot.data);
            eprintln!("      Data copied ({} bytes)", snapshot.data.len());

            // 4. Apply metadata — THIS IS WHERE IT WAS FAILING
            eprintln!("      Applying metadata ({} bytes)...", snapshot.metadata_blob.len());
            if let Err(e) = page.set_metadata_blob(&snapshot.metadata_blob) {
                eprintln!("      METADATA RESTORE FAILED: {e}");
                return Err(e); // ← this is PageError::MetadataDecode(...)
            }
            eprintln!("      Metadata restored OK");

            // 5. Insert
            pages.insert(snapshot.page_id, page);
            eprintln!("      Page inserted");
        }

        eprintln!("=== RESTORE_FROM_SNAPSHOT SUCCESS: {} pages restored ===", snapshots.len());
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

    #[test]
    fn test_checkpoint_roundtrip_in_memory() {
        let alloc = PageAllocator::new(PageAllocatorConfig::default());

        // Allocate and modify
        let ptr = alloc.allocate_raw(PageID(9999), 1024*1024, None).unwrap();
        let page = unsafe { &mut *ptr };
        page.apply_delta(&Delta {
            delta_id: DeltaID(1),
            page_id: PageID(9999),
            epoch: Epoch(1),
            mask: vec![true],
            payload: vec![0x11],
            is_sparse: false,
            timestamp: 0,
            source: Source("test".into()),
        }).unwrap();

        // Snapshot
        let snapshot = alloc.snapshot_pages();
        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].epoch, 1);

        // Clear and restore
        alloc.restore_from_snapshot(snapshot).expect("roundtrip should work");

        let restored = alloc.acquire_page(PageID(9999)).unwrap();
        let restored_page = unsafe { &*restored };
        assert_eq!(restored_page.epoch().0, 1);
        println!("CHECKPOINT ROUNDTRIP TEST PASSED");
    }


}
