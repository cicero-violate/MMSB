# CFG Group: src/00_physical

## Function: `test_checkpoint_roundtrip_in_memory`

- File: MMSB/src/00_physical/allocator.rs
- Branches: 0
- Loops: 0
- Nodes: 14
- Edges: 13

```mermaid
flowchart TD
    test_checkpoint_roundtrip_in_memory_0["ENTRY"]
    test_checkpoint_roundtrip_in_memory_1["let alloc = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_checkpoint_roundtrip_in_memory_2["let ptr = alloc . allocate_raw (PageID (9999) , 1024 * 1024 , None) . unwrap ()"]
    test_checkpoint_roundtrip_in_memory_3["let page = unsafe { & mut * ptr }"]
    test_checkpoint_roundtrip_in_memory_4["page . apply_delta (& Delta { delta_id : DeltaID (1) , page_id : PageID (9999..."]
    test_checkpoint_roundtrip_in_memory_5["let snapshot = alloc . snapshot_pages ()"]
    test_checkpoint_roundtrip_in_memory_6["macro assert_eq"]
    test_checkpoint_roundtrip_in_memory_7["macro assert_eq"]
    test_checkpoint_roundtrip_in_memory_8["alloc . restore_from_snapshot (snapshot) . expect ('roundtrip should work')"]
    test_checkpoint_roundtrip_in_memory_9["let restored = alloc . acquire_page (PageID (9999)) . unwrap ()"]
    test_checkpoint_roundtrip_in_memory_10["let restored_page = unsafe { & * restored }"]
    test_checkpoint_roundtrip_in_memory_11["macro assert_eq"]
    test_checkpoint_roundtrip_in_memory_12["macro println"]
    test_checkpoint_roundtrip_in_memory_13["EXIT"]
    test_checkpoint_roundtrip_in_memory_0 --> test_checkpoint_roundtrip_in_memory_1
    test_checkpoint_roundtrip_in_memory_1 --> test_checkpoint_roundtrip_in_memory_2
    test_checkpoint_roundtrip_in_memory_2 --> test_checkpoint_roundtrip_in_memory_3
    test_checkpoint_roundtrip_in_memory_3 --> test_checkpoint_roundtrip_in_memory_4
    test_checkpoint_roundtrip_in_memory_4 --> test_checkpoint_roundtrip_in_memory_5
    test_checkpoint_roundtrip_in_memory_5 --> test_checkpoint_roundtrip_in_memory_6
    test_checkpoint_roundtrip_in_memory_6 --> test_checkpoint_roundtrip_in_memory_7
    test_checkpoint_roundtrip_in_memory_7 --> test_checkpoint_roundtrip_in_memory_8
    test_checkpoint_roundtrip_in_memory_8 --> test_checkpoint_roundtrip_in_memory_9
    test_checkpoint_roundtrip_in_memory_9 --> test_checkpoint_roundtrip_in_memory_10
    test_checkpoint_roundtrip_in_memory_10 --> test_checkpoint_roundtrip_in_memory_11
    test_checkpoint_roundtrip_in_memory_11 --> test_checkpoint_roundtrip_in_memory_12
    test_checkpoint_roundtrip_in_memory_12 --> test_checkpoint_roundtrip_in_memory_13
```

## Function: `test_page_info_metadata_roundtrip`

- File: MMSB/src/00_physical/allocator.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_page_info_metadata_roundtrip_0["ENTRY"]
    test_page_info_metadata_roundtrip_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    test_page_info_metadata_roundtrip_2["let ptr = allocator . allocate_raw (PageID (1) , 128 , None) . expect ('allocation succ..."]
    test_page_info_metadata_roundtrip_3["let page = unsafe { & mut * ptr }"]
    test_page_info_metadata_roundtrip_4["page . set_metadata (vec ! [('key' . to_string () , b'abc123' . to_vec ())])"]
    test_page_info_metadata_roundtrip_5["let infos = allocator . page_infos ()"]
    test_page_info_metadata_roundtrip_6["macro assert_eq"]
    test_page_info_metadata_roundtrip_7["macro assert_eq"]
    test_page_info_metadata_roundtrip_8["macro assert_eq"]
    test_page_info_metadata_roundtrip_9["EXIT"]
    test_page_info_metadata_roundtrip_0 --> test_page_info_metadata_roundtrip_1
    test_page_info_metadata_roundtrip_1 --> test_page_info_metadata_roundtrip_2
    test_page_info_metadata_roundtrip_2 --> test_page_info_metadata_roundtrip_3
    test_page_info_metadata_roundtrip_3 --> test_page_info_metadata_roundtrip_4
    test_page_info_metadata_roundtrip_4 --> test_page_info_metadata_roundtrip_5
    test_page_info_metadata_roundtrip_5 --> test_page_info_metadata_roundtrip_6
    test_page_info_metadata_roundtrip_6 --> test_page_info_metadata_roundtrip_7
    test_page_info_metadata_roundtrip_7 --> test_page_info_metadata_roundtrip_8
    test_page_info_metadata_roundtrip_8 --> test_page_info_metadata_roundtrip_9
```

## Function: `test_unified_page`

- File: MMSB/src/00_physical/allocator.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    test_unified_page_0["ENTRY"]
    test_unified_page_1["let config = PageAllocatorConfig { default_location : PageLocation :: Unified , }"]
    test_unified_page_2["let allocator = PageAllocator :: new (config)"]
    test_unified_page_3["let ptr = allocator . allocate_raw (PageID (1) , 4096 , None) . expect ('Unified page a..."]
    test_unified_page_4["let page = unsafe { & mut * ptr }"]
    test_unified_page_5["macro assert_eq"]
    test_unified_page_6["let data = page . data_mut_slice ()"]
    test_unified_page_7["data [0] = 42"]
    test_unified_page_8["macro assert_eq"]
    test_unified_page_9["macro println"]
    test_unified_page_10["EXIT"]
    test_unified_page_0 --> test_unified_page_1
    test_unified_page_1 --> test_unified_page_2
    test_unified_page_2 --> test_unified_page_3
    test_unified_page_3 --> test_unified_page_4
    test_unified_page_4 --> test_unified_page_5
    test_unified_page_5 --> test_unified_page_6
    test_unified_page_6 --> test_unified_page_7
    test_unified_page_7 --> test_unified_page_8
    test_unified_page_8 --> test_unified_page_9
    test_unified_page_9 --> test_unified_page_10
```

