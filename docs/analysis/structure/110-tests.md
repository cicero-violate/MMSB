# Structure Group: tests

## File: MMSB/tests/delta_validation.rs

- Layer(s): root
- Language coverage: Rust (3)
- Element types: Function (3)
- Total elements: 3

### Elements

- [Rust | Function] `dense_delta` (line 0, priv)
  - Signature: `fn dense_delta (payload : Vec < u8 > , mask : Vec < bool >) -> Delta { Delta { delta_id : DeltaID (1) , page_id : Pag...`
  - Calls: DeltaID, PageID, Epoch, Source, into
- [Rust | Function] `rejects_mismatched_dense_lengths` (line 0, priv)
  - Signature: `# [test] fn rejects_mismatched_dense_lengths () { let delta = dense_delta (vec ! [1 , 2] , vec ! [true , true , true]...`
  - Calls: dense_delta
- [Rust | Function] `validates_dense_lengths` (line 0, priv)
  - Signature: `# [test] fn validates_dense_lengths () { let delta = dense_delta (vec ! [1 , 2 , 3] , vec ! [true , true , true]) ; a...`
  - Calls: dense_delta

## File: MMSB/tests/examples_basic.rs

- Layer(s): root
- Language coverage: Rust (3)
- Element types: Function (3)
- Total elements: 3

### Elements

- [Rust | Function] `example_checkpoint` (line 0, priv)
  - Signature: `# [test] fn example_checkpoint () { use mmsb_core :: page :: { write_checkpoint , load_checkpoint } ; assert ! (true)...`
- [Rust | Function] `example_delta_operations` (line 0, priv)
  - Signature: `# [test] fn example_delta_operations () { let data = vec ! [42u8 ; 128] ; let mask = vec ! [true ; 128] ; let delta =...`
  - Calls: Delta::new_dense, DeltaID, PageID, Epoch, Source, into
- [Rust | Function] `example_page_allocation` (line 0, priv)
  - Signature: `# [test] fn example_page_allocation () { let config = PageAllocatorConfig :: default () ; let allocator = PageAllocat...`
  - Calls: PageAllocatorConfig::default, PageAllocator::new, PageID, allocate_raw, Some, free

## File: MMSB/tests/mmsb_tests.rs

- Layer(s): root
- Language coverage: Rust (10)
- Element types: Function (10)
- Total elements: 10

### Elements

- [Rust | Function] `read_page` (line 0, priv)
  - Signature: `fn read_page (page : & Page) -> Vec < u8 > { page . data_slice () . to_vec () } . sig`
  - Calls: to_vec, data_slice
- [Rust | Function] `test_api_public_interface` (line 0, priv)
  - Signature: `# [test] # [ignore = "Julia API not available"] fn test_api_public_interface () { } . sig`
- [Rust | Function] `test_checkpoint_log_and_restore` (line 0, priv)
  - Signature: `# [test] fn test_checkpoint_log_and_restore () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, TransactionLog::new, to_string, unwrap, allocate_raw, PageID, copy_from_slice, len, copy_from_slice, data_mut_slice, unwrap, write_checkpoint, to_string, PageAllocator::new, PageAllocatorConfig::default, unwrap, TransactionLog::new, to_string, unwrap, load_checkpoint, expect, acquire_page, std::fs::remove_file, std::fs::remove_file, std::fs::remove_file
- [Rust | Function] `test_dense_delta_application` (line 0, priv)
  - Signature: `# [test] fn test_dense_delta_application () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default ()...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, unwrap, Delta::new_dense, DeltaID, Epoch, Source, into, unwrap, apply_delta
- [Rust | Function] `test_gpu_delta_kernels` (line 0, priv)
  - Signature: `# [test] # [ignore = "GPU not yet implemented"] fn test_gpu_delta_kernels () { } . sig`
- [Rust | Function] `test_invalid_page_deletion_is_safe` (line 0, priv)
  - Signature: `# [test] fn test_invalid_page_deletion_is_safe () { let allocator = PageAllocator :: new (PageAllocatorConfig :: defa...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, release, PageID
- [Rust | Function] `test_page_info_metadata_roundtrip` (line 0, priv)
  - Signature: `# [test] fn test_page_info_metadata_roundtrip () { let allocator = PageAllocator :: new (PageAllocatorConfig :: defau...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, set_metadata, page_infos
- [Rust | Function] `test_page_snapshot_and_restore` (line 0, priv)
  - Signature: `# [test] fn test_page_snapshot_and_restore () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default ...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, fill, data_mut_slice, snapshot_pages, PageAllocator::new, PageAllocatorConfig::default, unwrap, restore_from_snapshot, unwrap, acquire_page
- [Rust | Function] `test_sparse_delta_application` (line 0, priv)
  - Signature: `# [test] fn test_sparse_delta_application () { let allocator = PageAllocator :: new (PageAllocatorConfig :: default (...`
  - Calls: PageAllocator::new, PageAllocatorConfig::default, unwrap, allocate_raw, PageID, unwrap, Delta::new_sparse, DeltaID, Epoch, Source, into, unwrap, apply_delta
- [Rust | Function] `test_thread_safe_allocator` (line 0, priv)
  - Signature: `# [test] fn test_thread_safe_allocator () { let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: ...`
  - Calls: Arc::new, PageAllocator::new, PageAllocatorConfig::default, collect, map, Arc::clone, std::thread::spawn, unwrap, allocate_raw, PageID, data_mut_slice, unwrap, join

## File: MMSB/tests/week27_31_integration.rs

- Layer(s): root
- Language coverage: Rust (6)
- Element types: Function (6)
- Total elements: 6

### Elements

- [Rust | Function] `test_allocator_cpu_gpu_latency` (line 0, priv)
  - Signature: `# [test] fn test_allocator_cpu_gpu_latency () { let config = PageAllocatorConfig :: default () ; let allocator = Arc ...`
  - Calls: PageAllocatorConfig::default, Arc::new, PageAllocator::new, PageID, allocate_raw, Some, free
- [Rust | Function] `test_cpu_features` (line 0, priv)
  - Signature: `# [test] fn test_cpu_features () { use mmsb_core :: utility :: CpuFeatures ; let _features = CpuFeatures :: detect ()...`
  - Calls: CpuFeatures::detect
- [Rust | Function] `test_delta_merge_simd` (line 0, priv)
  - Signature: `# [test] fn test_delta_merge_simd () { use mmsb_core :: page :: merge_deltas ; let data1 : Vec < u8 > = (0 .. 64) . c...`
  - Calls: collect, unwrap, Delta::new_dense, DeltaID, PageID, Epoch, Source, into, collect, unwrap, Delta::new_dense, DeltaID, PageID, Epoch, Source, into, merge_deltas
- [Rust | Function] `test_lockfree_allocator` (line 0, priv)
  - Signature: `# [test] fn test_lockfree_allocator () { use mmsb_core :: physical :: lockfree_allocator :: LockFreeAllocator ; let a...`
  - Calls: LockFreeAllocator::new, get_stats
- [Rust | Function] `test_propagation_queue` (line 0, priv)
  - Signature: `# [test] fn test_propagation_queue () { use mmsb_core :: propagation :: PropagationQueue ; let _queue = PropagationQu...`
  - Calls: PropagationQueue::new
- [Rust | Function] `test_semiring_operations_tropical` (line 0, priv)
  - Signature: `# [test] fn test_semiring_operations_tropical () { let _semiring = TropicalSemiring ; assert ! (true) ; } . sig`

