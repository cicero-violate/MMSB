use mmsb_primitives::PageID;

#[derive(Debug, Default)]
pub struct HostDeviceSync {
    pending: Vec<PageID>,
}

impl HostDeviceSync {
    pub fn enqueue(&mut self, page_id: PageID) {
        self.pending.push(page_id);
    }

    pub fn drain(&mut self) -> Vec<PageID> {
        std::mem::take(&mut self.pending)
    }
}
