use mmsb_primitives::PageID;
use super::*;
use super::page_impl::Page;

pub trait PageAllocatorLike {
    fn allocate(&self, id: PageID, size: usize) -> Result<(), PageError>;
    fn free(&self, id: PageID);
    fn get(&self, id: PageID) -> Option<&Page>;
}
