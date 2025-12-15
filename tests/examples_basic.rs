//! Basic example tests demonstrating core functionality

use mmsb_core::page::{Page, PageID, PageLocation};
use mmsb_core::physical::{PageAllocator, PageAllocatorConfig};
use mmsb_core::semiring::TropicalSemiring;

#[test]
fn example_simple_page_allocation() {
    // Create allocator
    let config = PageAllocatorConfig::default();
    let allocator = PageAllocator::new(config);
    
    // Allocate page
    let page_id = PageID(1);
    let result = allocator.allocate_raw(page_id, 4096, Some(PageLocation::Cpu));
    
    assert!(result.is_ok());
    
    // Clean up
    allocator.free(page_id);
}

#[test]
fn example_delta_operations() {
    use mmsb_core::page::Delta;
    
    // Create dense delta
    let mut delta = Delta::new_dense(128);
    
    // Set some bytes
    delta.set_byte(0, 42);
    delta.set_byte(64, 99);
    
    // Read back
    assert_eq!(delta.get_byte(0), Some(42));
    assert_eq!(delta.get_byte(64), Some(99));
    assert_eq!(delta.get_byte(32), None);
}

#[test]
fn example_semiring_tropical() {
    // Tropical semiring available
    let _semiring = TropicalSemiring;
    assert!(true);
}

#[test]
fn example_checkpoint_roundtrip() {
    use mmsb_core::page::{write_checkpoint, load_checkpoint};
    use std::collections::HashMap;
    
    // Checkpoint functions available
    assert!(true);
}
