use mmsb_primitives::PageID;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

// Placeholder - in full system, Page comes from mmsb-memory
#[derive(Debug, Clone)]
pub struct Page {
    pub id: PageID,
}

#[derive(Debug, Default)]
pub struct DeviceRegistry {
    pages: RwLock<HashMap<PageID, Arc<Page>>>,
}

impl DeviceRegistry {
    pub fn register(&self, page: Arc<Page>) {
        self.pages.write().insert(page.id, page);
    }

    pub fn unregister(&self, page_id: PageID) {
        self.pages.write().remove(&page_id);
    }

    pub fn get(&self, page_id: PageID) -> Option<Arc<Page>> {
        self.pages.read().get(&page_id).cloned()
    }
}
