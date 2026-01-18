use mmsb_primitives::PageID;
use mmsb_memory::page::PageView;
use parking_lot::RwLock;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct DeviceRegistry {
    pages: RwLock<HashMap<PageID, PageView>>,
}

impl DeviceRegistry {
    pub fn register(&self, view: PageView) {
        self.pages.write().insert(view.id, view);
    }

    pub fn unregister(&self, page_id: PageID) {
        self.pages.write().remove(&page_id);
    }

    pub fn get(&self, page_id: PageID) -> Option<PageView> {
        self.pages.read().get(&page_id).cloned()
    }
}
