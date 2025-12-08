// tests/mmsb_tests.rs

use mmsb_core::runtime::allocator::{PageAllocator, PageAllocatorConfig};
use mmsb_core::runtime::checkpoint::{load_checkpoint, write_checkpoint};
use mmsb_core::runtime::tlog::TransactionLog;
use mmsb_core::types::{Delta, DeltaID, Epoch, Page, PageID, Source};
use std::sync::Arc;

fn read_page(page: &Page) -> Vec<u8> {
    page.data_slice().to_vec()
}

#[test]
fn test_page_info_metadata_roundtrip() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let ptr = allocator.allocate_raw(PageID(0), 128, None).unwrap();
    let page = unsafe { &mut *ptr };
    page.set_metadata(vec![("key".to_string(), b"abc123".to_vec())]);

    let infos = allocator.page_infos();
    assert_eq!(infos.len(), 1);
    assert_eq!(infos[0].metadata, page.metadata_blob());
}

#[test]
fn test_page_snapshot_and_restore() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let ptr = allocator.allocate_raw(PageID(0), 128, None).unwrap();
    let page = unsafe { &mut *ptr };
    page.data_mut_slice().fill(99);

    let snapshot = allocator.snapshot_pages();

    let new_allocator = PageAllocator::new(PageAllocatorConfig::default());
    new_allocator.restore_from_snapshot(snapshot).unwrap();

    let restored_ptr = new_allocator.acquire_page(page.id).unwrap();
    let restored_page = unsafe { &*restored_ptr };
    assert!(read_page(restored_page).iter().all(|&b| b == 99));
}

#[test]
fn test_thread_safe_allocator() {
    let allocator = Arc::new(PageAllocator::new(PageAllocatorConfig::default()));

    let handles: Vec<_> = (0..16)
        .map(|i| {
            let alloc = Arc::clone(&allocator);
            std::thread::spawn(move || {
                let ptr = alloc.allocate_raw(PageID(0), 256, None).unwrap();
                let page = unsafe { &mut *ptr };
                page.data_mut_slice()[0] = i as u8;
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(allocator.len(), 16);
}

#[test]
#[ignore = "GPU not yet implemented"]
fn test_gpu_delta_kernels() {}

#[test]
fn test_checkpoint_log_and_restore() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let log = TransactionLog::new("test_log.mmsb".to_string()).unwrap();

    let ptr = allocator.allocate_raw(PageID(0), 64, None).unwrap();
    let page = unsafe { &*ptr };
    let page_id = page.id; // <-- real ID assigned by allocator

    // Write exactly 64 bytes
    let msg = b"MMSB checkpoint test data 2025";
    let mut data = vec![0u8; 64];
    data[..msg.len()].copy_from_slice(msg);
    let page_mut = unsafe { &mut *ptr };
    page_mut.data_mut_slice().copy_from_slice(&data);

    let path = "checkpoint.mmsb";
    write_checkpoint(&allocator, &log, path.to_string()).unwrap();

    // Restore
    let new_allocator = PageAllocator::new(PageAllocatorConfig::default());
    let new_log = TransactionLog::new("restored_log.mmsb".to_string()).unwrap();
    load_checkpoint(&new_allocator, &new_log, path).unwrap();

    let restored_ptr = new_allocator
        .acquire_page(page_id)
        .expect("restored page must exist");
    let restored_page = unsafe { &*restored_ptr };

    assert_eq!(&read_page(restored_page)[..30], b"MMSB checkpoint test data 2025");

    // cleanup
    let _ = std::fs::remove_file(path);
    let _ = std::fs::remove_file("test_log.mmsb");
    let _ = std::fs::remove_file("restored_log.mmsb");
}

#[test]
fn test_invalid_page_deletion_is_safe() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    allocator.release(PageID(9999));
    assert_eq!(allocator.len(), 0);
}

#[test]
fn test_sparse_delta_application() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let ptr = allocator.allocate_raw(PageID(0), 8, None).unwrap();
    let page = unsafe { &*ptr };
    let real_id = page.id;

    let mask = vec![true, false, true, false, true, false, true, false];
    let payload = vec![11, 22, 33, 44];

    let delta = Delta::new_sparse(
        DeltaID(7),
        real_id,
        Epoch(10),
        mask,
        payload,
        Source("test".into()),
    )
    .unwrap();

    let page_mut = unsafe { &mut *ptr };
    page_mut.apply_delta(&delta).unwrap();

    assert_eq!(read_page(page_mut), vec![11, 0, 22, 0, 33, 0, 44, 0]);
}

#[test]
fn test_dense_delta_application() {
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let ptr = allocator.allocate_raw(PageID(0), 4, None).unwrap();
    let page = unsafe { &*ptr };
    let real_id = page.id;

    let delta = Delta::new_dense(
        DeltaID(8),
        real_id,
        Epoch(20),
        vec![99, 88, 77, 66],
        vec![true, true, true, true],
        Source("dense".into()),
    )
    .unwrap();

    let page_mut = unsafe { &mut *ptr };
    page_mut.apply_delta(&delta).unwrap();

    assert_eq!(read_page(page_mut), vec![99, 88, 77, 66]);
}

#[test]
#[ignore = "Julia API not available"]
fn test_api_public_interface() {}
