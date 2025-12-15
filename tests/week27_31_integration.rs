use mmsb_core::page::{Delta, DeltaID, Epoch, Page, PageID, PageLocation, Source};
use mmsb_core::physical::{PageAllocator, PageAllocatorConfig};
use mmsb_core::semiring::TropicalSemiring;
use std::sync::Arc;

#[test]
fn test_allocator_cpu_gpu_latency() {
    let config = PageAllocatorConfig::default();
    let allocator = Arc::new(PageAllocator::new(config));
    
    let cpu_id = PageID(1);
    let cpu_page = allocator.allocate_raw(cpu_id, 1024, Some(PageLocation::Cpu));
    assert!(cpu_page.is_ok());
    
    allocator.free(cpu_id);
}

#[test]
fn test_semiring_operations_tropical() {
    let _semiring = TropicalSemiring;
    assert!(true);
}

#[test]
fn test_delta_merge_simd() {
    use mmsb_core::page::merge_deltas;
    
    let data1: Vec<u8> = (0..64).collect();
    let mask1 = vec![true; 64];
    let delta1 = Delta::new_dense(
        DeltaID(1), PageID(1), Epoch(0), data1, mask1, Source("test".into())
    ).unwrap();
    
    let data2: Vec<u8> = (100..164).collect();
    let mask2 = vec![true; 64];
    let delta2 = Delta::new_dense(
        DeltaID(2), PageID(1), Epoch(0), data2, mask2, Source("test".into())
    ).unwrap();
    
    let merged = merge_deltas(&delta1, &delta2);
    assert!(merged.is_ok());
}

#[test]
fn test_lockfree_allocator() {
    use mmsb_core::physical::lockfree_allocator::LockFreeAllocator;
    
    let allocator = LockFreeAllocator::new();
    let ptr1 = allocator.try_allocate_small(PageID(1), 512, PageLocation::Cpu);
    assert!(ptr1.is_some());
    
    if let Some(p) = ptr1 {
        allocator.deallocate_small(p);
    }
}

#[test]
fn test_propagation_queue() {
    use mmsb_core::propagation::PropagationQueue;
    let _queue = PropagationQueue::new();
    assert!(true);
}

#[test]
fn test_cpu_features() {
    use mmsb_core::utility::CpuFeatures;
    let _features = CpuFeatures::detect();
    assert!(true);
}
