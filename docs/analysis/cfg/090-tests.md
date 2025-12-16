# CFG Group: tests

## Function: `dense_delta`

- File: MMSB/tests/delta_validation.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    dense_delta_0["ENTRY"]
    dense_delta_1["Delta { delta_id : DeltaID (1) , page_id : PageID (1) , epoch : Epoch (0) , m..."]
    dense_delta_2["EXIT"]
    dense_delta_0 --> dense_delta_1
    dense_delta_1 --> dense_delta_2
```

## Function: `example_checkpoint`

- File: MMSB/tests/examples_basic.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    example_checkpoint_0["ENTRY"]
    example_checkpoint_1["use"]
    example_checkpoint_2["macro assert"]
    example_checkpoint_3["EXIT"]
    example_checkpoint_0 --> example_checkpoint_1
    example_checkpoint_1 --> example_checkpoint_2
    example_checkpoint_2 --> example_checkpoint_3
```

## Function: `example_delta_operations`

- File: MMSB/tests/examples_basic.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    example_delta_operations_0["ENTRY"]
    example_delta_operations_1["let data = vec ! [42u8 ; 128]"]
    example_delta_operations_2["let mask = vec ! [true ; 128]"]
    example_delta_operations_3["let delta = Delta :: new_dense (DeltaID (1) , PageID (1) , Epoch (0) , data , mask , Sour..."]
    example_delta_operations_4["macro assert"]
    example_delta_operations_5["EXIT"]
    example_delta_operations_0 --> example_delta_operations_1
    example_delta_operations_1 --> example_delta_operations_2
    example_delta_operations_2 --> example_delta_operations_3
    example_delta_operations_3 --> example_delta_operations_4
    example_delta_operations_4 --> example_delta_operations_5
```

## Function: `example_page_allocation`

- File: MMSB/tests/examples_basic.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    example_page_allocation_0["ENTRY"]
    example_page_allocation_1["let config = PageAllocatorConfig :: default ()"]
    example_page_allocation_2["let allocator = PageAllocator :: new (config)"]
    example_page_allocation_3["let page_id = PageID (1)"]
    example_page_allocation_4["let result = allocator . allocate_raw (page_id , 4096 , Some (PageLocation :: Cpu))"]
    example_page_allocation_5["macro assert"]
    example_page_allocation_6["allocator . free (page_id)"]
    example_page_allocation_7["EXIT"]
    example_page_allocation_0 --> example_page_allocation_1
    example_page_allocation_1 --> example_page_allocation_2
    example_page_allocation_2 --> example_page_allocation_3
    example_page_allocation_3 --> example_page_allocation_4
    example_page_allocation_4 --> example_page_allocation_5
    example_page_allocation_5 --> example_page_allocation_6
    example_page_allocation_6 --> example_page_allocation_7
```

## Function: `read_page`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    read_page_0["ENTRY"]
    read_page_1["page . data_slice () . to_vec ()"]
    read_page_2["EXIT"]
    read_page_0 --> read_page_1
    read_page_1 --> read_page_2
```

## Function: `rejects_mismatched_dense_lengths`

- File: MMSB/tests/delta_validation.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    rejects_mismatched_dense_lengths_0["ENTRY"]
    rejects_mismatched_dense_lengths_1["let delta = dense_delta (vec ! [1 , 2] , vec ! [true , true , true])"]
    rejects_mismatched_dense_lengths_2["macro assert"]
    rejects_mismatched_dense_lengths_3["EXIT"]
    rejects_mismatched_dense_lengths_0 --> rejects_mismatched_dense_lengths_1
    rejects_mismatched_dense_lengths_1 --> rejects_mismatched_dense_lengths_2
    rejects_mismatched_dense_lengths_2 --> rejects_mismatched_dense_lengths_3
```

## Function: `test_allocator_cpu_gpu_latency`

- File: MMSB/tests/week27_31_integration.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_allocator_cpu_gpu_latency_0["ENTRY"]
    test_allocator_cpu_gpu_latency_1["let config = PageAllocatorConfig :: default ()"]
    test_allocator_cpu_gpu_latency_2["let allocator = Arc :: new (PageAllocator :: new (config))"]
    test_allocator_cpu_gpu_latency_3["let cpu_id = PageID (1)"]
    test_allocator_cpu_gpu_latency_4["let cpu_page = allocator . allocate_raw (cpu_id , 1024 , Some (PageLocation :: Cpu))"]
    test_allocator_cpu_gpu_latency_5["macro assert"]
    test_allocator_cpu_gpu_latency_6["allocator . free (cpu_id)"]
    test_allocator_cpu_gpu_latency_7["EXIT"]
    test_allocator_cpu_gpu_latency_0 --> test_allocator_cpu_gpu_latency_1
    test_allocator_cpu_gpu_latency_1 --> test_allocator_cpu_gpu_latency_2
    test_allocator_cpu_gpu_latency_2 --> test_allocator_cpu_gpu_latency_3
    test_allocator_cpu_gpu_latency_3 --> test_allocator_cpu_gpu_latency_4
    test_allocator_cpu_gpu_latency_4 --> test_allocator_cpu_gpu_latency_5
    test_allocator_cpu_gpu_latency_5 --> test_allocator_cpu_gpu_latency_6
    test_allocator_cpu_gpu_latency_6 --> test_allocator_cpu_gpu_latency_7
```

## Function: `test_api_public_interface`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 2
- Edges: 1

```mermaid
flowchart TD
    test_api_public_interface_0["ENTRY"]
    test_api_public_interface_1["EXIT"]
    test_api_public_interface_0 --> test_api_public_interface_1
```

## Function: `test_checkpoint_log_and_restore`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 23
- Edges: 22

```mermaid
flowchart TD
    test_checkpoint_log_and_restore_0["ENTRY"]
    test_checkpoint_log_and_restore_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_checkpoint_log_and_restore_2["let log = TransactionLog :: new ('test_log.mmsb' . to_string ()) . unwrap ()"]
    test_checkpoint_log_and_restore_3["let ptr = allocator . allocate_raw (PageID (0) , 64 , None) . unwrap ()"]
    test_checkpoint_log_and_restore_4["let page = unsafe { & * ptr }"]
    test_checkpoint_log_and_restore_5["let page_id = page . id"]
    test_checkpoint_log_and_restore_6["let msg = b'MMSB checkpoint test data 2025'"]
    test_checkpoint_log_and_restore_7["let mut data = vec ! [0u8 ; 64]"]
    test_checkpoint_log_and_restore_8["data [.. msg . len ()] . copy_from_slice (msg)"]
    test_checkpoint_log_and_restore_9["let page_mut = unsafe { & mut * ptr }"]
    test_checkpoint_log_and_restore_10["page_mut . data_mut_slice () . copy_from_slice (& data)"]
    test_checkpoint_log_and_restore_11["let path = 'checkpoint.mmsb'"]
    test_checkpoint_log_and_restore_12["write_checkpoint (& allocator , & log , path . to_string ()) . unwrap ()"]
    test_checkpoint_log_and_restore_13["let new_allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_checkpoint_log_and_restore_14["let new_log = TransactionLog :: new ('restored_log.mmsb' . to_string ()) . unwrap ()"]
    test_checkpoint_log_and_restore_15["load_checkpoint (& new_allocator , & new_log , path) . unwrap ()"]
    test_checkpoint_log_and_restore_16["let restored_ptr = new_allocator . acquire_page (page_id) . expect ('restored page must exist')"]
    test_checkpoint_log_and_restore_17["let restored_page = unsafe { & * restored_ptr }"]
    test_checkpoint_log_and_restore_18["macro assert_eq"]
    test_checkpoint_log_and_restore_19["let _ = std :: fs :: remove_file (path)"]
    test_checkpoint_log_and_restore_20["let _ = std :: fs :: remove_file ('test_log.mmsb')"]
    test_checkpoint_log_and_restore_21["let _ = std :: fs :: remove_file ('restored_log.mmsb')"]
    test_checkpoint_log_and_restore_22["EXIT"]
    test_checkpoint_log_and_restore_0 --> test_checkpoint_log_and_restore_1
    test_checkpoint_log_and_restore_1 --> test_checkpoint_log_and_restore_2
    test_checkpoint_log_and_restore_2 --> test_checkpoint_log_and_restore_3
    test_checkpoint_log_and_restore_3 --> test_checkpoint_log_and_restore_4
    test_checkpoint_log_and_restore_4 --> test_checkpoint_log_and_restore_5
    test_checkpoint_log_and_restore_5 --> test_checkpoint_log_and_restore_6
    test_checkpoint_log_and_restore_6 --> test_checkpoint_log_and_restore_7
    test_checkpoint_log_and_restore_7 --> test_checkpoint_log_and_restore_8
    test_checkpoint_log_and_restore_8 --> test_checkpoint_log_and_restore_9
    test_checkpoint_log_and_restore_9 --> test_checkpoint_log_and_restore_10
    test_checkpoint_log_and_restore_10 --> test_checkpoint_log_and_restore_11
    test_checkpoint_log_and_restore_11 --> test_checkpoint_log_and_restore_12
    test_checkpoint_log_and_restore_12 --> test_checkpoint_log_and_restore_13
    test_checkpoint_log_and_restore_13 --> test_checkpoint_log_and_restore_14
    test_checkpoint_log_and_restore_14 --> test_checkpoint_log_and_restore_15
    test_checkpoint_log_and_restore_15 --> test_checkpoint_log_and_restore_16
    test_checkpoint_log_and_restore_16 --> test_checkpoint_log_and_restore_17
    test_checkpoint_log_and_restore_17 --> test_checkpoint_log_and_restore_18
    test_checkpoint_log_and_restore_18 --> test_checkpoint_log_and_restore_19
    test_checkpoint_log_and_restore_19 --> test_checkpoint_log_and_restore_20
    test_checkpoint_log_and_restore_20 --> test_checkpoint_log_and_restore_21
    test_checkpoint_log_and_restore_21 --> test_checkpoint_log_and_restore_22
```

## Function: `test_cpu_features`

- File: MMSB/tests/week27_31_integration.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_cpu_features_0["ENTRY"]
    test_cpu_features_1["use"]
    test_cpu_features_2["let _features = CpuFeatures :: detect ()"]
    test_cpu_features_3["macro assert"]
    test_cpu_features_4["EXIT"]
    test_cpu_features_0 --> test_cpu_features_1
    test_cpu_features_1 --> test_cpu_features_2
    test_cpu_features_2 --> test_cpu_features_3
    test_cpu_features_3 --> test_cpu_features_4
```

## Function: `test_delta_merge_simd`

- File: MMSB/tests/week27_31_integration.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    test_delta_merge_simd_0["ENTRY"]
    test_delta_merge_simd_1["use"]
    test_delta_merge_simd_2["let data1 : Vec < u8 > = (0 .. 64) . collect ()"]
    test_delta_merge_simd_3["let mask1 = vec ! [true ; 64]"]
    test_delta_merge_simd_4["let delta1 = Delta :: new_dense (DeltaID (1) , PageID (1) , Epoch (0) , data1 , mask1 , So..."]
    test_delta_merge_simd_5["let data2 : Vec < u8 > = (100 .. 164) . collect ()"]
    test_delta_merge_simd_6["let mask2 = vec ! [true ; 64]"]
    test_delta_merge_simd_7["let delta2 = Delta :: new_dense (DeltaID (2) , PageID (1) , Epoch (0) , data2 , mask2 , So..."]
    test_delta_merge_simd_8["let merged = merge_deltas (& delta1 , & delta2)"]
    test_delta_merge_simd_9["macro assert"]
    test_delta_merge_simd_10["EXIT"]
    test_delta_merge_simd_0 --> test_delta_merge_simd_1
    test_delta_merge_simd_1 --> test_delta_merge_simd_2
    test_delta_merge_simd_2 --> test_delta_merge_simd_3
    test_delta_merge_simd_3 --> test_delta_merge_simd_4
    test_delta_merge_simd_4 --> test_delta_merge_simd_5
    test_delta_merge_simd_5 --> test_delta_merge_simd_6
    test_delta_merge_simd_6 --> test_delta_merge_simd_7
    test_delta_merge_simd_7 --> test_delta_merge_simd_8
    test_delta_merge_simd_8 --> test_delta_merge_simd_9
    test_delta_merge_simd_9 --> test_delta_merge_simd_10
```

## Function: `test_dense_delta_application`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_dense_delta_application_0["ENTRY"]
    test_dense_delta_application_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_dense_delta_application_2["let ptr = allocator . allocate_raw (PageID (0) , 4 , None) . unwrap ()"]
    test_dense_delta_application_3["let page = unsafe { & * ptr }"]
    test_dense_delta_application_4["let real_id = page . id"]
    test_dense_delta_application_5["let delta = Delta :: new_dense (DeltaID (8) , real_id , Epoch (20) , vec ! [99 , 88 , 77 ..."]
    test_dense_delta_application_6["let page_mut = unsafe { & mut * ptr }"]
    test_dense_delta_application_7["page_mut . apply_delta (& delta) . unwrap ()"]
    test_dense_delta_application_8["macro assert_eq"]
    test_dense_delta_application_9["EXIT"]
    test_dense_delta_application_0 --> test_dense_delta_application_1
    test_dense_delta_application_1 --> test_dense_delta_application_2
    test_dense_delta_application_2 --> test_dense_delta_application_3
    test_dense_delta_application_3 --> test_dense_delta_application_4
    test_dense_delta_application_4 --> test_dense_delta_application_5
    test_dense_delta_application_5 --> test_dense_delta_application_6
    test_dense_delta_application_6 --> test_dense_delta_application_7
    test_dense_delta_application_7 --> test_dense_delta_application_8
    test_dense_delta_application_8 --> test_dense_delta_application_9
```

## Function: `test_gpu_delta_kernels`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 2
- Edges: 1

```mermaid
flowchart TD
    test_gpu_delta_kernels_0["ENTRY"]
    test_gpu_delta_kernels_1["EXIT"]
    test_gpu_delta_kernels_0 --> test_gpu_delta_kernels_1
```

## Function: `test_invalid_page_deletion_is_safe`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_invalid_page_deletion_is_safe_0["ENTRY"]
    test_invalid_page_deletion_is_safe_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_invalid_page_deletion_is_safe_2["allocator . release (PageID (9999))"]
    test_invalid_page_deletion_is_safe_3["macro assert_eq"]
    test_invalid_page_deletion_is_safe_4["EXIT"]
    test_invalid_page_deletion_is_safe_0 --> test_invalid_page_deletion_is_safe_1
    test_invalid_page_deletion_is_safe_1 --> test_invalid_page_deletion_is_safe_2
    test_invalid_page_deletion_is_safe_2 --> test_invalid_page_deletion_is_safe_3
    test_invalid_page_deletion_is_safe_3 --> test_invalid_page_deletion_is_safe_4
```

## Function: `test_lockfree_allocator`

- File: MMSB/tests/week27_31_integration.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_lockfree_allocator_0["ENTRY"]
    test_lockfree_allocator_1["use"]
    test_lockfree_allocator_2["let allocator = LockFreeAllocator :: new ()"]
    test_lockfree_allocator_3["let (freelist_size , allocated , freed) = allocator . get_stats ()"]
    test_lockfree_allocator_4["macro assert_eq"]
    test_lockfree_allocator_5["macro assert_eq"]
    test_lockfree_allocator_6["macro assert_eq"]
    test_lockfree_allocator_7["EXIT"]
    test_lockfree_allocator_0 --> test_lockfree_allocator_1
    test_lockfree_allocator_1 --> test_lockfree_allocator_2
    test_lockfree_allocator_2 --> test_lockfree_allocator_3
    test_lockfree_allocator_3 --> test_lockfree_allocator_4
    test_lockfree_allocator_4 --> test_lockfree_allocator_5
    test_lockfree_allocator_5 --> test_lockfree_allocator_6
    test_lockfree_allocator_6 --> test_lockfree_allocator_7
```

## Function: `test_page_info_metadata_roundtrip`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_page_info_metadata_roundtrip_0["ENTRY"]
    test_page_info_metadata_roundtrip_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_page_info_metadata_roundtrip_2["let ptr = allocator . allocate_raw (PageID (0) , 128 , None) . unwrap ()"]
    test_page_info_metadata_roundtrip_3["let page = unsafe { & mut * ptr }"]
    test_page_info_metadata_roundtrip_4["page . set_metadata (vec ! [('key' . to_string () , b'abc123' . to_vec ())])"]
    test_page_info_metadata_roundtrip_5["let infos = allocator . page_infos ()"]
    test_page_info_metadata_roundtrip_6["macro assert_eq"]
    test_page_info_metadata_roundtrip_7["macro assert_eq"]
    test_page_info_metadata_roundtrip_8["EXIT"]
    test_page_info_metadata_roundtrip_0 --> test_page_info_metadata_roundtrip_1
    test_page_info_metadata_roundtrip_1 --> test_page_info_metadata_roundtrip_2
    test_page_info_metadata_roundtrip_2 --> test_page_info_metadata_roundtrip_3
    test_page_info_metadata_roundtrip_3 --> test_page_info_metadata_roundtrip_4
    test_page_info_metadata_roundtrip_4 --> test_page_info_metadata_roundtrip_5
    test_page_info_metadata_roundtrip_5 --> test_page_info_metadata_roundtrip_6
    test_page_info_metadata_roundtrip_6 --> test_page_info_metadata_roundtrip_7
    test_page_info_metadata_roundtrip_7 --> test_page_info_metadata_roundtrip_8
```

## Function: `test_page_snapshot_and_restore`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    test_page_snapshot_and_restore_0["ENTRY"]
    test_page_snapshot_and_restore_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_page_snapshot_and_restore_2["let ptr = allocator . allocate_raw (PageID (0) , 128 , None) . unwrap ()"]
    test_page_snapshot_and_restore_3["let page = unsafe { & mut * ptr }"]
    test_page_snapshot_and_restore_4["page . data_mut_slice () . fill (99)"]
    test_page_snapshot_and_restore_5["let snapshot = allocator . snapshot_pages ()"]
    test_page_snapshot_and_restore_6["let new_allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_page_snapshot_and_restore_7["new_allocator . restore_from_snapshot (snapshot) . unwrap ()"]
    test_page_snapshot_and_restore_8["let restored_ptr = new_allocator . acquire_page (page . id) . unwrap ()"]
    test_page_snapshot_and_restore_9["let restored_page = unsafe { & * restored_ptr }"]
    test_page_snapshot_and_restore_10["macro assert"]
    test_page_snapshot_and_restore_11["EXIT"]
    test_page_snapshot_and_restore_0 --> test_page_snapshot_and_restore_1
    test_page_snapshot_and_restore_1 --> test_page_snapshot_and_restore_2
    test_page_snapshot_and_restore_2 --> test_page_snapshot_and_restore_3
    test_page_snapshot_and_restore_3 --> test_page_snapshot_and_restore_4
    test_page_snapshot_and_restore_4 --> test_page_snapshot_and_restore_5
    test_page_snapshot_and_restore_5 --> test_page_snapshot_and_restore_6
    test_page_snapshot_and_restore_6 --> test_page_snapshot_and_restore_7
    test_page_snapshot_and_restore_7 --> test_page_snapshot_and_restore_8
    test_page_snapshot_and_restore_8 --> test_page_snapshot_and_restore_9
    test_page_snapshot_and_restore_9 --> test_page_snapshot_and_restore_10
    test_page_snapshot_and_restore_10 --> test_page_snapshot_and_restore_11
```

## Function: `test_propagation_queue`

- File: MMSB/tests/week27_31_integration.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    test_propagation_queue_0["ENTRY"]
    test_propagation_queue_1["use"]
    test_propagation_queue_2["let _queue = PropagationQueue :: new ()"]
    test_propagation_queue_3["macro assert"]
    test_propagation_queue_4["EXIT"]
    test_propagation_queue_0 --> test_propagation_queue_1
    test_propagation_queue_1 --> test_propagation_queue_2
    test_propagation_queue_2 --> test_propagation_queue_3
    test_propagation_queue_3 --> test_propagation_queue_4
```

## Function: `test_semiring_operations_tropical`

- File: MMSB/tests/week27_31_integration.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    test_semiring_operations_tropical_0["ENTRY"]
    test_semiring_operations_tropical_1["let _semiring = TropicalSemiring"]
    test_semiring_operations_tropical_2["macro assert"]
    test_semiring_operations_tropical_3["EXIT"]
    test_semiring_operations_tropical_0 --> test_semiring_operations_tropical_1
    test_semiring_operations_tropical_1 --> test_semiring_operations_tropical_2
    test_semiring_operations_tropical_2 --> test_semiring_operations_tropical_3
```

## Function: `test_sparse_delta_application`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    test_sparse_delta_application_0["ENTRY"]
    test_sparse_delta_application_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_sparse_delta_application_2["let ptr = allocator . allocate_raw (PageID (0) , 8 , None) . unwrap ()"]
    test_sparse_delta_application_3["let page = unsafe { & * ptr }"]
    test_sparse_delta_application_4["let real_id = page . id"]
    test_sparse_delta_application_5["let mask = vec ! [true , false , true , false , true , false , true , false]"]
    test_sparse_delta_application_6["let payload = vec ! [11 , 22 , 33 , 44]"]
    test_sparse_delta_application_7["let delta = Delta :: new_sparse (DeltaID (7) , real_id , Epoch (10) , mask , payload , So..."]
    test_sparse_delta_application_8["let page_mut = unsafe { & mut * ptr }"]
    test_sparse_delta_application_9["page_mut . apply_delta (& delta) . unwrap ()"]
    test_sparse_delta_application_10["macro assert_eq"]
    test_sparse_delta_application_11["EXIT"]
    test_sparse_delta_application_0 --> test_sparse_delta_application_1
    test_sparse_delta_application_1 --> test_sparse_delta_application_2
    test_sparse_delta_application_2 --> test_sparse_delta_application_3
    test_sparse_delta_application_3 --> test_sparse_delta_application_4
    test_sparse_delta_application_4 --> test_sparse_delta_application_5
    test_sparse_delta_application_5 --> test_sparse_delta_application_6
    test_sparse_delta_application_6 --> test_sparse_delta_application_7
    test_sparse_delta_application_7 --> test_sparse_delta_application_8
    test_sparse_delta_application_8 --> test_sparse_delta_application_9
    test_sparse_delta_application_9 --> test_sparse_delta_application_10
    test_sparse_delta_application_10 --> test_sparse_delta_application_11
```

## Function: `test_thread_safe_allocator`

- File: MMSB/tests/mmsb_tests.rs
- Branches: 0
- Loops: 1
- Nodes: 8
- Edges: 8

```mermaid
flowchart TD
    test_thread_safe_allocator_0["ENTRY"]
    test_thread_safe_allocator_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    test_thread_safe_allocator_2["let handles : Vec < _ > = (0 .. 16) . map (| i | { let alloc = Arc :: clone (& allocator) ; std :: thre..."]
    test_thread_safe_allocator_3["for h in handles"]
    test_thread_safe_allocator_4["h . join () . unwrap ()"]
    test_thread_safe_allocator_5["after for"]
    test_thread_safe_allocator_6["macro assert_eq"]
    test_thread_safe_allocator_7["EXIT"]
    test_thread_safe_allocator_0 --> test_thread_safe_allocator_1
    test_thread_safe_allocator_1 --> test_thread_safe_allocator_2
    test_thread_safe_allocator_2 --> test_thread_safe_allocator_3
    test_thread_safe_allocator_3 --> test_thread_safe_allocator_4
    test_thread_safe_allocator_4 --> test_thread_safe_allocator_3
    test_thread_safe_allocator_3 --> test_thread_safe_allocator_5
    test_thread_safe_allocator_5 --> test_thread_safe_allocator_6
    test_thread_safe_allocator_6 --> test_thread_safe_allocator_7
```

## Function: `validates_dense_lengths`

- File: MMSB/tests/delta_validation.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    validates_dense_lengths_0["ENTRY"]
    validates_dense_lengths_1["let delta = dense_delta (vec ! [1 , 2 , 3] , vec ! [true , true , true])"]
    validates_dense_lengths_2["macro assert"]
    validates_dense_lengths_3["EXIT"]
    validates_dense_lengths_0 --> validates_dense_lengths_1
    validates_dense_lengths_1 --> validates_dense_lengths_2
    validates_dense_lengths_2 --> validates_dense_lengths_3
```

