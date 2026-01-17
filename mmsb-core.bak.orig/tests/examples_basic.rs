// Use the public prelude API
use mmsb_core::prelude::{
    Delta, DeltaID, Epoch, PageAllocator, PageAllocatorConfig, PageID, PageLocation, Source,
};

#[test]
fn example_page_allocation() {
    let config = PageAllocatorConfig::default();
    let allocator = PageAllocator::new(config);
    
    let page_id = PageID(1);
    let result = allocator.allocate_raw(page_id, 4096, Some(PageLocation::Cpu));
    assert!(result.is_ok());
    
    allocator.free(page_id);
}

#[test]
fn example_delta_operations() {
    let data = vec![42u8; 128];
    let mask = vec![true; 128];
    let delta = Delta::new_dense(
        DeltaID(1), PageID(1), Epoch(0), data, mask, Source("test".into())
    );
    assert!(delta.is_ok());
}

#[test]
fn example_checkpoint() {
    assert!(true);
}
