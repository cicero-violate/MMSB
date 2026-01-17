use crate::page::{Page, PageError, PageLocation};
use mmsb_primitives::PageID;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

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
        Self {
            config,
            pages: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        }
    }

    /// Clear all allocated pages and reset ID counter.
    pub fn clear(&self) {
        let mut pages = self.pages.lock();
        pages.clear();
        self.next_id.store(1, Ordering::SeqCst);
    }

    pub fn allocate_raw(
        &self,
        page_id_hint: PageID,
        size: usize,
        location: Option<PageLocation>,
    ) -> Result<*mut Page, PageError> {
        let loc = location.unwrap_or(self.config.default_location);

        let mut pages = self.pages.lock();
        if pages.contains_key(&page_id_hint) {
            return Err(PageError::AlreadyExists(page_id_hint));
        }

        let page = Box::new(Page::new(page_id_hint, size, loc)?);
        let ptr = Box::into_raw(page);

        #[cfg(debug_assertions)]
        println!(
            "[ALLOCATOR] allocate_raw(id={}) → raw ptr = {:p}",
            page_id_hint.0, ptr
        );

        pages.insert(page_id_hint, unsafe { Box::from_raw(ptr) });

        Ok(ptr)
    }

    pub fn free(&self, page_id: PageID) {
        if let Some(_) = self.pages.lock().remove(&page_id) {
            #[cfg(debug_assertions)]
            println!("[ALLOCATOR] Freed page {}", page_id.0);
        }
    }

    pub fn release(&self, page_id: PageID) {
        if let Some(boxed_page) = self.pages.lock().remove(&page_id) {
            #[cfg(debug_assertions)]
            println!(
                "[ALLOCATOR] release({}): ownership transferred",
                page_id.0
            );
            // Prevent double drop – caller now owns the Box
            std::mem::forget(boxed_page);
        }
    }

    pub fn acquire_page(&self, page_id: PageID) -> Option<*mut Page> {
        let pages = self.pages.lock();
        pages.get(&page_id).map(|boxed| Box::as_mut(boxed) as *mut Page)
    }

    pub fn page_infos(&self) -> Vec<PageInfo> {
        let pages = self.pages.lock();
        pages
            .iter()
            .map(|(id, page)| PageInfo {
                page_id: *id,
                size: page.capacity,
                location: page.location,
                epoch: page.epoch.load().0,
                metadata: page.metadata.clone_store().into_iter().flat_map(|(k, v)| {
                    let mut bytes = k.into_bytes();
                    bytes.extend_from_slice(&v);
                    bytes
                }).collect(),
            })
            .collect()
    }

    pub fn snapshot_pages(&self) -> Vec<PageSnapshotData> {
        let pages = self.pages.lock();
        pages
            .iter()
            .map(|(id, page)| {
                let mut data = vec![0u8; page.capacity];
                unsafe {
                    std::ptr::copy_nonoverlapping(page.data, data.as_mut_ptr(), page.capacity);
                }
                PageSnapshotData {
                    page_id: *id,
                    size: page.capacity,
                    location: page.location,
                    epoch: page.epoch.load().0,
                    metadata_blob: vec![], // TODO: implement proper blob serialization if needed
                    data,
                }
            })
            .collect()
    }

    pub fn restore_from_snapshot(&self, snapshots: Vec<PageSnapshotData>) -> Result<(), PageError> {
        let mut pages = self.pages.lock();
        pages.clear();

        for snapshot in snapshots {
            let page = Page::new(
                snapshot.page_id,
                snapshot.size,
                snapshot.location,
            )?;

            unsafe {
                std::ptr::copy_nonoverlapping(
                    snapshot.data.as_ptr(),
                    page.data,
                    snapshot.size,
                );
            }

            page.epoch.store(snapshot.epoch.into());

            // TODO: restore metadata if blob is implemented

            pages.insert(snapshot.page_id, Box::new(page));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::page::{Delta, DeltaID, Source};
    use crate::epoch::Epoch;

    #[test]
    fn test_allocate_and_acquire() {
        let alloc = PageAllocator::new(PageAllocatorConfig::default());
        let ptr = alloc.allocate_raw(PageID(42), 1024, None).unwrap();
        let page = unsafe { &mut *ptr };
        assert_eq!(page.id.0, 42);
        assert_eq!(page.capacity, 1024);
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let alloc = PageAllocator::new(PageAllocatorConfig::default());

        alloc.allocate_raw(PageID(777), 4096, None).unwrap();

        let snapshot = alloc.snapshot_pages();
        assert_eq!(snapshot.len(), 1);

        alloc.restore_from_snapshot(snapshot).unwrap();

        let restored = alloc.acquire_page(PageID(777));
        assert!(restored.is_some());
    }
}
