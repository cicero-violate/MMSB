use mmsb_primitives::PageID;
use crate::page::Page;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

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

    pub fn contains(&self, page_id: PageID) -> bool {
        self.pages.read().contains_key(&page_id)
    }
}
