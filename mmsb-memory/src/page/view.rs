use mmsb_primitives::PageID;
use super::PageLocation;

#[derive(Debug, Clone, Copy)]
pub struct PageView {
    pub id: PageID,
    pub location: PageLocation,
    pub data: *mut u8,
    pub mask: *mut u8,
    pub len: usize,
}

impl PageView {
    pub fn data_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
}

unsafe impl Send for PageView {}
unsafe impl Sync for PageView {}
