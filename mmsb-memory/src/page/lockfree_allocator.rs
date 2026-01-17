use crate::page::{PageID, PageLocation};
use crate::page::Page;
use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::ptr;

const SMALL_PAGE_THRESHOLD: usize = 4096;
const FREELIST_CAPACITY: usize = 256;

struct FreeListNode {
    next: AtomicPtr<FreeListNode>,
    page_ptr: *mut Page,
}

pub struct LockFreeAllocator {
    freelist_head: AtomicPtr<FreeListNode>,
    freelist_size: AtomicU64,
    allocated_count: AtomicU64,
    freed_count: AtomicU64,
}

impl LockFreeAllocator {
    pub fn new() -> Self {
        Self {
            freelist_head: AtomicPtr::new(ptr::null_mut()),
            freelist_size: AtomicU64::new(0),
            allocated_count: AtomicU64::new(0),
            freed_count: AtomicU64::new(0),
        }
    }
    
    pub fn try_allocate_small(&self, _page_id: PageID, size: usize, _location: PageLocation) 
        -> Option<*mut Page> 
    {
        if size > SMALL_PAGE_THRESHOLD {
            return None;
        }
        
        loop {
            let head = self.freelist_head.load(Ordering::Acquire);
            
            if head.is_null() {
                return None;
            }
            
            let node = unsafe { &*head };
            let next = node.next.load(Ordering::Relaxed);
            
            if self.freelist_head.compare_exchange(
                head,
                next,
                Ordering::Release,
                Ordering::Acquire
            ).is_ok() {
                self.freelist_size.fetch_sub(1, Ordering::Relaxed);
                self.allocated_count.fetch_add(1, Ordering::Relaxed);
                
                let page_ptr = node.page_ptr;
                unsafe {
                    let _ = Box::from_raw(head);
                }
                
                return Some(page_ptr);
            }
        }
    }
    
    pub fn deallocate_small(&self, page_ptr: *mut Page) -> bool {
        let page = unsafe { &*page_ptr };
        
        if page.size() > SMALL_PAGE_THRESHOLD {
            return false;
        }
        
        let current_size = self.freelist_size.load(Ordering::Relaxed);
        if current_size >= FREELIST_CAPACITY as u64 {
            return false;
        }
        
        let node = Box::into_raw(Box::new(FreeListNode {
            next: AtomicPtr::new(ptr::null_mut()),
            page_ptr,
        }));
        
        loop {
            let head = self.freelist_head.load(Ordering::Acquire);
            unsafe { (*node).next.store(head, Ordering::Relaxed); }
            
            if self.freelist_head.compare_exchange(
                head,
                node,
                Ordering::Release,
                Ordering::Acquire
            ).is_ok() {
                self.freelist_size.fetch_add(1, Ordering::Relaxed);
                self.freed_count.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
    }
    
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.freelist_size.load(Ordering::Relaxed),
            self.allocated_count.load(Ordering::Relaxed),
            self.freed_count.load(Ordering::Relaxed),
        )
    }
    
    pub fn clear(&self) {
        let mut head = self.freelist_head.swap(ptr::null_mut(), Ordering::AcqRel);
        
        while !head.is_null() {
            let node = unsafe { Box::from_raw(head) };
            head = node.next.load(Ordering::Relaxed);
            unsafe {
                drop(Box::from_raw(node.page_ptr));
            }
        }
        
        self.freelist_size.store(0, Ordering::Relaxed);
    }
}

impl Drop for LockFreeAllocator {
    fn drop(&mut self) {
        self.clear();
    }
}

unsafe impl Send for LockFreeAllocator {}
unsafe impl Sync for LockFreeAllocator {}
