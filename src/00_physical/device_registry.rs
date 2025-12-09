use crate::types::{Page, PageID};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Default)]
pub struct DeviceBufferRegistry {
    map: RwLock<HashMap<PageID, Arc<Page>>>,
}

impl DeviceBufferRegistry {
    pub fn insert(&self, page: Arc<Page>) {
        self.map.write().insert(page.id, page);
    }

    pub fn remove(&self, page_id: PageID) {
        self.map.write().remove(&page_id);
    }

    pub fn len(&self) -> usize {
        self.map.read().len()
    }
}
