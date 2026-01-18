use mmsb_primitives::PageID;
use mmsb_memory::page::PageView;
use parking_lot::RwLock;
use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct DeviceBufferRegistry {
    map: RwLock<HashMap<PageID, PageView>>,
}

impl DeviceBufferRegistry {
    pub fn insert(&self, view: PageView) {
        self.map.write().insert(view.id, view);
    }

    pub fn remove(&self, page_id: PageID) {
        self.map.write().remove(&page_id);
    }

    pub fn len(&self) -> usize {
        self.map.read().len()
    }

    pub fn contains(&self, page_id: PageID) -> bool {
        self.map.read().contains_key(&page_id)
    }

    pub fn get(&self, page_id: PageID) -> Option<PageView> {
        self.map.read().get(&page_id).cloned()
    }
}
