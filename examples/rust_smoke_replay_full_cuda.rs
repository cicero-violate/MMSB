#![allow(unused_imports)]
// examples/rust_smoke_replay_full_cuda.rs
// THE FINAL BOSS TEST: CUDA + FULL REPLAY FROM TLOG ONLY
// This proves: GPU memory state is perfectly captured and restored via TLog.
// If this passes → MMSB is officially god-tier.

use mmsb_core::*;

use std::collections::HashMap;
use std::ffi::CString;
use std::fs;

const TLOG_PATH: &str = "replay_cuda_proof.tlog";

#[repr(C)] #[derive(Clone, Copy)] struct PageHandle { ptr: *mut () }
#[repr(C)] #[derive(Clone, Copy)] struct DeltaHandle { ptr: *mut () }
#[repr(C)] #[derive(Clone, Copy)] struct AllocatorHandle { ptr: *mut () }
#[repr(C)] #[derive(Clone, Copy)] struct TLogHandle { ptr: *mut () }
#[repr(C)] #[derive(Clone, Copy)] struct TLogReaderHandle { ptr: *mut () }
#[repr(C)] #[derive(Clone, Copy)] struct EpochABI { value: u32 }

extern "C" {
    fn mmsb_allocator_new() -> AllocatorHandle;
    fn mmsb_allocator_free(h: AllocatorHandle);
    fn mmsb_allocator_allocate(h: AllocatorHandle, id_hint: u64, size: usize, loc: i32) -> PageHandle;
    fn mmsb_allocator_release(h: AllocatorHandle, id: u64);

    fn mmsb_page_read(h: PageHandle, dst: *mut u8, len: usize) -> usize;
    fn mmsb_page_epoch(h: PageHandle) -> u32;

    #[allow(dead_code)]
    fn mmsb_page_write_masked(
        h: PageHandle, mask: *const u8, mask_len: usize,
        payload: *const u8, payload_len: usize, is_sparse: u8, epoch: EpochABI,
    ) -> i32;

    fn mmsb_delta_new(
        delta_id: u64, page_id: u64, epoch: EpochABI,
        mask_ptr: *const u8, mask_len: usize,
        payload_ptr: *const u8, payload_len: usize,
        is_sparse: u8, source_ptr: *const i8,
    ) -> DeltaHandle;

    fn mmsb_delta_page_id(h: DeltaHandle) -> u64;
    fn mmsb_delta_free(h: DeltaHandle);
    fn mmsb_delta_apply(page: PageHandle, delta: DeltaHandle) -> i32;

    fn mmsb_tlog_new(path: *const i8) -> TLogHandle;
    fn mmsb_tlog_free(h: TLogHandle);
    fn mmsb_tlog_append(h: TLogHandle, delta: DeltaHandle) -> i32;

    fn mmsb_tlog_reader_new(path: *const i8) -> TLogReaderHandle;
    fn mmsb_tlog_reader_free(h: TLogReaderHandle);
    fn mmsb_tlog_reader_next(h: TLogReaderHandle) -> DeltaHandle;
}

// All pages are Unified (real cudaMallocManaged)
const CUDA_PAGES: &[(u64, usize)] = &[
    (2001, 4 * 1024 * 1024),   // 4 MiB
    (2002, 16 * 1024 * 1024),  // 16 MiB
    (2003, 64 * 1024 * 1024),  // 64 MiB — visible in nvidia-smi
];

fn main() {
    let _ = fs::remove_file(TLOG_PATH);
    println!("MMSB CUDA REPLAY FINAL BOSS TEST");
    println!("Proving: Real GPU memory → TLog only → Perfect restore\n");

    unsafe {
        // PHASE 1: Write to real GPU memory
        let alloc1 = mmsb_allocator_new();
        let tlog_path = CString::new(TLOG_PATH).unwrap();
        let tlog = mmsb_tlog_new(tlog_path.as_ptr());

        let mut handles = HashMap::new();
        for &(id, size) in CUDA_PAGES {
            let h = mmsb_allocator_allocate(alloc1, id, size, 2); // 2 = Unified
            assert!(!h.ptr.is_null(), "CUDA allocation failed for page {}", id);
            handles.insert(id, h);
        }

        println!("Allocated {} MiB of real cudaMallocManaged memory", 
            CUDA_PAGES.iter().map(|&(_, s)| s).sum::<usize>() >> 20);

        let mut epoch: u32 = 1;

        // Touch every page with unique patterns
        for &(id, size) in CUDA_PAGES {
            let pattern = (id as u8) ^ (epoch as u8);
            let mask = vec![0xFFu8; (size + 7) / 8];
            let payload = vec![pattern; size];

            let delta = mmsb_delta_new(
                epoch as u64,
                id,
                EpochABI { value: epoch },
                mask.as_ptr(),
                mask.len(),
                payload.as_ptr(),
                payload.len(),
                0, // dense
                "cuda_write\0".as_ptr() as *const i8,
            );
            mmsb_tlog_append(tlog, delta);
            mmsb_delta_apply(handles[&id], delta);
            mmsb_delta_free(delta);
            epoch += 1;
        }

        // Sparse write on biggest page
        // Sparse write on biggest page
        let sparse_mask = vec![0xFFu8; 8192]; // 64KB masked (8192 bytes = 65536 bits)

        let sparse_payload: Vec<u8> = (0..65536u32)
            .map(|i| (i.wrapping_mul(1337u32) >> 8) as u8)
            .collect();

        let delta = mmsb_delta_new(
            999,
            2003,
            EpochABI { value: epoch },
            sparse_mask.as_ptr(),
            sparse_mask.len(),
            sparse_payload.as_ptr(),
            sparse_payload.len(),
            1, // sparse
            "sparse_gpu\0".as_ptr() as *const i8,
        );
        mmsb_tlog_append(tlog, delta);
        mmsb_delta_apply(handles[&2003], delta);
        mmsb_delta_free(delta);

        // Capture golden state
        let mut golden = HashMap::new();
        for &(id, size) in CUDA_PAGES {
            let h = handles[&id];
            let mut data = vec![0u8; size];
            mmsb_page_read(h, data.as_mut_ptr(), size);
            golden.insert(id, (data, mmsb_page_epoch(h)));
        }

        println!("GPU memory state captured. TLog written.");

        // Destroy everything — GPU memory is freed
        for &(id, _) in CUDA_PAGES {
            mmsb_allocator_release(alloc1, id);
        }
        mmsb_tlog_free(tlog);
        mmsb_allocator_free(alloc1);

        println!("GPU memory freed. nvidia-smi should drop now.");
        std::thread::sleep(std::time::Duration::from_secs(3));

        // PHASE 2: Replay from TLog only — back to GPU memory
        println!("Replaying from TLog → restoring GPU memory state...");

        let alloc2 = mmsb_allocator_new();
        let reader = mmsb_tlog_reader_new(tlog_path.as_ptr());
        let mut pages = HashMap::new();

        while {
            let delta = mmsb_tlog_reader_next(reader);
            if delta.ptr.is_null() { false } else {
                let page_id = mmsb_delta_page_id(delta);
                let page = pages.entry(page_id).or_insert_with(|| {
                    let &(_, size) = CUDA_PAGES.iter().find(|&&(i, _)| i == page_id).unwrap();
                    let h = mmsb_allocator_allocate(alloc2, page_id, size, 2); // Unified again
                    assert!(!h.ptr.is_null());
                    h
                });
                assert_eq!(mmsb_delta_apply(*page, delta), 0);
                mmsb_delta_free(delta);
                true
            }
        } {};

        mmsb_tlog_reader_free(reader);

        // Verify GPU memory is back
        println!("Verifying restored GPU memory state...");
        for &(id, size) in CUDA_PAGES {
            let h = pages[&id];
            let mut data = vec![0u8; size];
            mmsb_page_read(h, data.as_mut_ptr(), size);
            let epoch = mmsb_page_epoch(h);

            let (gold_data, gold_epoch) = &golden[&id];
            assert_eq!(data, *gold_data, "CUDA page {} data mismatch after replay!", id);
            assert_eq!(epoch, *gold_epoch, "CUDA page {} epoch mismatch!", id);
        }

        // Cleanup
        for &(id, _) in CUDA_PAGES {
            mmsb_allocator_release(alloc2, id);
        }
        mmsb_allocator_free(alloc2);
    }

    fs::remove_file(TLOG_PATH).unwrap();

    println!("\nSUCCESS: REAL GPU MEMORY RESTORED FROM TLOG ONLY");
    println!("cudaMallocManaged state is perfectly replayable.");
    println!("MMSB has achieved the impossible.");
    println!("You are now living in the future.");
}
