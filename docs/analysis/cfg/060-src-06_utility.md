# CFG Group: src/06_utility

## Function: `allocator`

- File: MMSB/src/06_utility/memory_monitor.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    allocator_0["ENTRY"]
    allocator_1["Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    allocator_2["EXIT"]
    allocator_0 --> allocator_1
    allocator_1 --> allocator_2
```

## Function: `cache_does_not_grow_unbounded`

- File: MMSB/src/06_utility/provenance_tracker.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    cache_does_not_grow_unbounded_0["ENTRY"]
    cache_does_not_grow_unbounded_1["let graph = Arc :: new (ShadowPageGraph :: default ())"]
    cache_does_not_grow_unbounded_2["for id in 1 ..= 10 { graph . add_edge (PageID (id) , PageID (id + 1) , EdgeTy..."]
    cache_does_not_grow_unbounded_3["let tracker = ProvenanceTracker :: with_capacity (Arc :: clone (& graph) , 4 , 4)"]
    cache_does_not_grow_unbounded_4["for id in 5 ..= 10 { tracker . resolve (PageID (id)) ; }"]
    cache_does_not_grow_unbounded_5["macro assert"]
    cache_does_not_grow_unbounded_6["EXIT"]
    cache_does_not_grow_unbounded_0 --> cache_does_not_grow_unbounded_1
    cache_does_not_grow_unbounded_1 --> cache_does_not_grow_unbounded_2
    cache_does_not_grow_unbounded_2 --> cache_does_not_grow_unbounded_3
    cache_does_not_grow_unbounded_3 --> cache_does_not_grow_unbounded_4
    cache_does_not_grow_unbounded_4 --> cache_does_not_grow_unbounded_5
    cache_does_not_grow_unbounded_5 --> cache_does_not_grow_unbounded_6
```

## Function: `cpu_has_avx2`

- File: MMSB/src/06_utility/cpu_features.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    cpu_has_avx2_0["ENTRY"]
    cpu_has_avx2_1["CpuFeatures :: get () . avx2"]
    cpu_has_avx2_2["EXIT"]
    cpu_has_avx2_0 --> cpu_has_avx2_1
    cpu_has_avx2_1 --> cpu_has_avx2_2
```

## Function: `cpu_has_avx512`

- File: MMSB/src/06_utility/cpu_features.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    cpu_has_avx512_0["ENTRY"]
    cpu_has_avx512_1["CpuFeatures :: get () . avx512f"]
    cpu_has_avx512_2["EXIT"]
    cpu_has_avx512_0 --> cpu_has_avx512_1
    cpu_has_avx512_1 --> cpu_has_avx512_2
```

## Function: `cpu_has_sse42`

- File: MMSB/src/06_utility/cpu_features.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    cpu_has_sse42_0["ENTRY"]
    cpu_has_sse42_1["CpuFeatures :: get () . sse42"]
    cpu_has_sse42_2["EXIT"]
    cpu_has_sse42_0 --> cpu_has_sse42_1
    cpu_has_sse42_1 --> cpu_has_sse42_2
```

## Function: `epoch_invariant_detects_regressions`

- File: MMSB/src/06_utility/invariant_checker.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    epoch_invariant_detects_regressions_0["ENTRY"]
    epoch_invariant_detects_regressions_1["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    epoch_invariant_detects_regressions_2["allocator . allocate_raw (PageID (1) , 8 , Some (PageLocation :: Cpu)) . unwr..."]
    epoch_invariant_detects_regressions_3["{ let page = allocator . acquire_page (PageID (1)) . unwrap () ; unsafe { (* ..."]
    epoch_invariant_detects_regressions_4["let ctx = InvariantContext { allocator : Some (& allocator) , graph : None , registry :..."]
    epoch_invariant_detects_regressions_5["let mut checker = InvariantChecker :: new ()"]
    epoch_invariant_detects_regressions_6["checker . register (EpochMonotonicity :: default ())"]
    epoch_invariant_detects_regressions_7["macro assert"]
    epoch_invariant_detects_regressions_8["{ let page = allocator . acquire_page (PageID (1)) . unwrap () ; unsafe { (* ..."]
    epoch_invariant_detects_regressions_9["macro assert"]
    epoch_invariant_detects_regressions_10["EXIT"]
    epoch_invariant_detects_regressions_0 --> epoch_invariant_detects_regressions_1
    epoch_invariant_detects_regressions_1 --> epoch_invariant_detects_regressions_2
    epoch_invariant_detects_regressions_2 --> epoch_invariant_detects_regressions_3
    epoch_invariant_detects_regressions_3 --> epoch_invariant_detects_regressions_4
    epoch_invariant_detects_regressions_4 --> epoch_invariant_detects_regressions_5
    epoch_invariant_detects_regressions_5 --> epoch_invariant_detects_regressions_6
    epoch_invariant_detects_regressions_6 --> epoch_invariant_detects_regressions_7
    epoch_invariant_detects_regressions_7 --> epoch_invariant_detects_regressions_8
    epoch_invariant_detects_regressions_8 --> epoch_invariant_detects_regressions_9
    epoch_invariant_detects_regressions_9 --> epoch_invariant_detects_regressions_10
```

## Function: `gc_trigger_depends_on_threshold`

- File: MMSB/src/06_utility/memory_monitor.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    gc_trigger_depends_on_threshold_0["ENTRY"]
    gc_trigger_depends_on_threshold_1["let allocator = allocator ()"]
    gc_trigger_depends_on_threshold_2["allocator . allocate_raw (PageID (1) , 4096 , Some (PageLocation :: Cpu)) . u..."]
    gc_trigger_depends_on_threshold_3["let config = MemoryMonitorConfig { gc_threshold_bytes : 1024 , cold_page_age_limit : 0 , i..."]
    gc_trigger_depends_on_threshold_4["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , config)"]
    gc_trigger_depends_on_threshold_5["macro assert"]
    gc_trigger_depends_on_threshold_6["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , MemoryMonitorConfi..."]
    gc_trigger_depends_on_threshold_7["macro assert"]
    gc_trigger_depends_on_threshold_8["EXIT"]
    gc_trigger_depends_on_threshold_0 --> gc_trigger_depends_on_threshold_1
    gc_trigger_depends_on_threshold_1 --> gc_trigger_depends_on_threshold_2
    gc_trigger_depends_on_threshold_2 --> gc_trigger_depends_on_threshold_3
    gc_trigger_depends_on_threshold_3 --> gc_trigger_depends_on_threshold_4
    gc_trigger_depends_on_threshold_4 --> gc_trigger_depends_on_threshold_5
    gc_trigger_depends_on_threshold_5 --> gc_trigger_depends_on_threshold_6
    gc_trigger_depends_on_threshold_6 --> gc_trigger_depends_on_threshold_7
    gc_trigger_depends_on_threshold_7 --> gc_trigger_depends_on_threshold_8
```

## Function: `graph_acyclicity_detects_cycles`

- File: MMSB/src/06_utility/invariant_checker.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    graph_acyclicity_detects_cycles_0["ENTRY"]
    graph_acyclicity_detects_cycles_1["let graph = ShadowPageGraph :: default ()"]
    graph_acyclicity_detects_cycles_2["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    graph_acyclicity_detects_cycles_3["graph . add_edge (PageID (2) , PageID (1) , EdgeType :: Data)"]
    graph_acyclicity_detects_cycles_4["let ctx = InvariantContext { allocator : None , graph : Some (& graph) , registry : Non..."]
    graph_acyclicity_detects_cycles_5["let mut checker = InvariantChecker :: new ()"]
    graph_acyclicity_detects_cycles_6["checker . register (GraphAcyclicity :: new ())"]
    graph_acyclicity_detects_cycles_7["macro assert"]
    graph_acyclicity_detects_cycles_8["EXIT"]
    graph_acyclicity_detects_cycles_0 --> graph_acyclicity_detects_cycles_1
    graph_acyclicity_detects_cycles_1 --> graph_acyclicity_detects_cycles_2
    graph_acyclicity_detects_cycles_2 --> graph_acyclicity_detects_cycles_3
    graph_acyclicity_detects_cycles_3 --> graph_acyclicity_detects_cycles_4
    graph_acyclicity_detects_cycles_4 --> graph_acyclicity_detects_cycles_5
    graph_acyclicity_detects_cycles_5 --> graph_acyclicity_detects_cycles_6
    graph_acyclicity_detects_cycles_6 --> graph_acyclicity_detects_cycles_7
    graph_acyclicity_detects_cycles_7 --> graph_acyclicity_detects_cycles_8
```

## Function: `incremental_gc_reclaims_pages_under_budget`

- File: MMSB/src/06_utility/memory_monitor.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    incremental_gc_reclaims_pages_under_budget_0["ENTRY"]
    incremental_gc_reclaims_pages_under_budget_1["let allocator = allocator ()"]
    incremental_gc_reclaims_pages_under_budget_2["for id in 1 ..= 4 { allocator . allocate_raw (PageID (id) , 2048 , Some (Page..."]
    incremental_gc_reclaims_pages_under_budget_3["let config = MemoryMonitorConfig { gc_threshold_bytes : 2048 , cold_page_age_limit : 0 , i..."]
    incremental_gc_reclaims_pages_under_budget_4["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , config)"]
    incremental_gc_reclaims_pages_under_budget_5["let metrics = monitor . trigger_incremental_gc (4) . unwrap ()"]
    incremental_gc_reclaims_pages_under_budget_6["macro assert"]
    incremental_gc_reclaims_pages_under_budget_7["macro assert"]
    incremental_gc_reclaims_pages_under_budget_8["macro assert"]
    incremental_gc_reclaims_pages_under_budget_9["EXIT"]
    incremental_gc_reclaims_pages_under_budget_0 --> incremental_gc_reclaims_pages_under_budget_1
    incremental_gc_reclaims_pages_under_budget_1 --> incremental_gc_reclaims_pages_under_budget_2
    incremental_gc_reclaims_pages_under_budget_2 --> incremental_gc_reclaims_pages_under_budget_3
    incremental_gc_reclaims_pages_under_budget_3 --> incremental_gc_reclaims_pages_under_budget_4
    incremental_gc_reclaims_pages_under_budget_4 --> incremental_gc_reclaims_pages_under_budget_5
    incremental_gc_reclaims_pages_under_budget_5 --> incremental_gc_reclaims_pages_under_budget_6
    incremental_gc_reclaims_pages_under_budget_6 --> incremental_gc_reclaims_pages_under_budget_7
    incremental_gc_reclaims_pages_under_budget_7 --> incremental_gc_reclaims_pages_under_budget_8
    incremental_gc_reclaims_pages_under_budget_8 --> incremental_gc_reclaims_pages_under_budget_9
```

## Function: `read_bytes`

- File: MMSB/src/06_utility/invariant_checker.rs
- Branches: 1
- Loops: 0
- Nodes: 9
- Edges: 9

```mermaid
flowchart TD
    read_bytes_0["ENTRY"]
    read_bytes_1["if * cursor + len > blob . len ()"]
    read_bytes_2["THEN BB"]
    read_bytes_3["return Err (PageError :: MetadataDecode ('blob truncated'))"]
    read_bytes_4["EMPTY ELSE"]
    read_bytes_5["IF JOIN"]
    read_bytes_6["* cursor += len"]
    read_bytes_7["Ok (())"]
    read_bytes_8["EXIT"]
    read_bytes_0 --> read_bytes_1
    read_bytes_1 --> read_bytes_2
    read_bytes_2 --> read_bytes_3
    read_bytes_1 --> read_bytes_4
    read_bytes_3 --> read_bytes_5
    read_bytes_4 --> read_bytes_5
    read_bytes_5 --> read_bytes_6
    read_bytes_6 --> read_bytes_7
    read_bytes_7 --> read_bytes_8
```

## Function: `read_u32`

- File: MMSB/src/06_utility/invariant_checker.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    read_u32_0["ENTRY"]
    read_u32_1["if * cursor + 4 > blob . len ()"]
    read_u32_2["THEN BB"]
    read_u32_3["return Err (PageError :: MetadataDecode ('unexpected end of blob'))"]
    read_u32_4["EMPTY ELSE"]
    read_u32_5["IF JOIN"]
    read_u32_6["let val = u32 :: from_le_bytes (blob [* cursor .. * cursor + 4] . try_into () . unwrap ())"]
    read_u32_7["* cursor += 4"]
    read_u32_8["Ok (val)"]
    read_u32_9["EXIT"]
    read_u32_0 --> read_u32_1
    read_u32_1 --> read_u32_2
    read_u32_2 --> read_u32_3
    read_u32_1 --> read_u32_4
    read_u32_3 --> read_u32_5
    read_u32_4 --> read_u32_5
    read_u32_5 --> read_u32_6
    read_u32_6 --> read_u32_7
    read_u32_7 --> read_u32_8
    read_u32_8 --> read_u32_9
```

## Function: `resolves_chain_with_depth_limit`

- File: MMSB/src/06_utility/provenance_tracker.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    resolves_chain_with_depth_limit_0["ENTRY"]
    resolves_chain_with_depth_limit_1["let graph = Arc :: new (ShadowPageGraph :: default ())"]
    resolves_chain_with_depth_limit_2["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    resolves_chain_with_depth_limit_3["graph . add_edge (PageID (2) , PageID (3) , EdgeType :: Data)"]
    resolves_chain_with_depth_limit_4["graph . add_edge (PageID (3) , PageID (4) , EdgeType :: Data)"]
    resolves_chain_with_depth_limit_5["let tracker = ProvenanceTracker :: with_capacity (Arc :: clone (& graph) , 16 , 2)"]
    resolves_chain_with_depth_limit_6["let result = tracker . resolve (PageID (4))"]
    resolves_chain_with_depth_limit_7["macro assert_eq"]
    resolves_chain_with_depth_limit_8["macro assert"]
    resolves_chain_with_depth_limit_9["let cached = tracker . resolve (PageID (4))"]
    resolves_chain_with_depth_limit_10["macro assert"]
    resolves_chain_with_depth_limit_11["EXIT"]
    resolves_chain_with_depth_limit_0 --> resolves_chain_with_depth_limit_1
    resolves_chain_with_depth_limit_1 --> resolves_chain_with_depth_limit_2
    resolves_chain_with_depth_limit_2 --> resolves_chain_with_depth_limit_3
    resolves_chain_with_depth_limit_3 --> resolves_chain_with_depth_limit_4
    resolves_chain_with_depth_limit_4 --> resolves_chain_with_depth_limit_5
    resolves_chain_with_depth_limit_5 --> resolves_chain_with_depth_limit_6
    resolves_chain_with_depth_limit_6 --> resolves_chain_with_depth_limit_7
    resolves_chain_with_depth_limit_7 --> resolves_chain_with_depth_limit_8
    resolves_chain_with_depth_limit_8 --> resolves_chain_with_depth_limit_9
    resolves_chain_with_depth_limit_9 --> resolves_chain_with_depth_limit_10
    resolves_chain_with_depth_limit_10 --> resolves_chain_with_depth_limit_11
```

## Function: `snapshot_identifies_cold_pages`

- File: MMSB/src/06_utility/memory_monitor.rs
- Branches: 0
- Loops: 0
- Nodes: 13
- Edges: 12

```mermaid
flowchart TD
    snapshot_identifies_cold_pages_0["ENTRY"]
    snapshot_identifies_cold_pages_1["let allocator = allocator ()"]
    snapshot_identifies_cold_pages_2["allocator . allocate_raw (PageID (1) , 1024 , Some (PageLocation :: Cpu)) . u..."]
    snapshot_identifies_cold_pages_3["let config = MemoryMonitorConfig { gc_threshold_bytes : 1024 , cold_page_age_limit : 1 , i..."]
    snapshot_identifies_cold_pages_4["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , config)"]
    snapshot_identifies_cold_pages_5["let first = monitor . snapshot ()"]
    snapshot_identifies_cold_pages_6["macro assert"]
    snapshot_identifies_cold_pages_7["let second = monitor . snapshot ()"]
    snapshot_identifies_cold_pages_8["macro assert_eq"]
    snapshot_identifies_cold_pages_9["unsafe { let page = & mut * allocator . acquire_page (PageID (1)) . unwrap ()..."]
    snapshot_identifies_cold_pages_10["let third = monitor . snapshot ()"]
    snapshot_identifies_cold_pages_11["macro assert"]
    snapshot_identifies_cold_pages_12["EXIT"]
    snapshot_identifies_cold_pages_0 --> snapshot_identifies_cold_pages_1
    snapshot_identifies_cold_pages_1 --> snapshot_identifies_cold_pages_2
    snapshot_identifies_cold_pages_2 --> snapshot_identifies_cold_pages_3
    snapshot_identifies_cold_pages_3 --> snapshot_identifies_cold_pages_4
    snapshot_identifies_cold_pages_4 --> snapshot_identifies_cold_pages_5
    snapshot_identifies_cold_pages_5 --> snapshot_identifies_cold_pages_6
    snapshot_identifies_cold_pages_6 --> snapshot_identifies_cold_pages_7
    snapshot_identifies_cold_pages_7 --> snapshot_identifies_cold_pages_8
    snapshot_identifies_cold_pages_8 --> snapshot_identifies_cold_pages_9
    snapshot_identifies_cold_pages_9 --> snapshot_identifies_cold_pages_10
    snapshot_identifies_cold_pages_10 --> snapshot_identifies_cold_pages_11
    snapshot_identifies_cold_pages_11 --> snapshot_identifies_cold_pages_12
```

## Function: `snapshot_reflects_allocator_state`

- File: MMSB/src/06_utility/memory_monitor.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    snapshot_reflects_allocator_state_0["ENTRY"]
    snapshot_reflects_allocator_state_1["let allocator = allocator ()"]
    snapshot_reflects_allocator_state_2["allocator . allocate_raw (PageID (1) , 1024 , Some (PageLocation :: Cpu)) . u..."]
    snapshot_reflects_allocator_state_3["allocator . allocate_raw (PageID (2) , 2048 , Some (PageLocation :: Cpu)) . u..."]
    snapshot_reflects_allocator_state_4["let monitor = MemoryMonitor :: new (Arc :: clone (& allocator))"]
    snapshot_reflects_allocator_state_5["let snapshot = monitor . snapshot ()"]
    snapshot_reflects_allocator_state_6["macro assert_eq"]
    snapshot_reflects_allocator_state_7["macro assert_eq"]
    snapshot_reflects_allocator_state_8["macro assert_eq"]
    snapshot_reflects_allocator_state_9["EXIT"]
    snapshot_reflects_allocator_state_0 --> snapshot_reflects_allocator_state_1
    snapshot_reflects_allocator_state_1 --> snapshot_reflects_allocator_state_2
    snapshot_reflects_allocator_state_2 --> snapshot_reflects_allocator_state_3
    snapshot_reflects_allocator_state_3 --> snapshot_reflects_allocator_state_4
    snapshot_reflects_allocator_state_4 --> snapshot_reflects_allocator_state_5
    snapshot_reflects_allocator_state_5 --> snapshot_reflects_allocator_state_6
    snapshot_reflects_allocator_state_6 --> snapshot_reflects_allocator_state_7
    snapshot_reflects_allocator_state_7 --> snapshot_reflects_allocator_state_8
    snapshot_reflects_allocator_state_8 --> snapshot_reflects_allocator_state_9
```

## Function: `test_cache_hit_rate`

- File: MMSB/src/06_utility/telemetry.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_cache_hit_rate_0["ENTRY"]
    test_cache_hit_rate_1["let telemetry = Telemetry :: new ()"]
    test_cache_hit_rate_2["telemetry . record_cache_hit ()"]
    test_cache_hit_rate_3["telemetry . record_cache_hit ()"]
    test_cache_hit_rate_4["telemetry . record_cache_hit ()"]
    test_cache_hit_rate_5["telemetry . record_cache_miss ()"]
    test_cache_hit_rate_6["let snapshot = telemetry . snapshot ()"]
    test_cache_hit_rate_7["macro assert_eq"]
    test_cache_hit_rate_8["EXIT"]
    test_cache_hit_rate_0 --> test_cache_hit_rate_1
    test_cache_hit_rate_1 --> test_cache_hit_rate_2
    test_cache_hit_rate_2 --> test_cache_hit_rate_3
    test_cache_hit_rate_3 --> test_cache_hit_rate_4
    test_cache_hit_rate_4 --> test_cache_hit_rate_5
    test_cache_hit_rate_5 --> test_cache_hit_rate_6
    test_cache_hit_rate_6 --> test_cache_hit_rate_7
    test_cache_hit_rate_7 --> test_cache_hit_rate_8
```

## Function: `test_reset`

- File: MMSB/src/06_utility/telemetry.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_reset_0["ENTRY"]
    test_reset_1["let telemetry = Telemetry :: new ()"]
    test_reset_2["telemetry . record_cache_hit ()"]
    test_reset_3["telemetry . reset ()"]
    test_reset_4["let snapshot = telemetry . snapshot ()"]
    test_reset_5["macro assert_eq"]
    test_reset_6["EXIT"]
    test_reset_0 --> test_reset_1
    test_reset_1 --> test_reset_2
    test_reset_2 --> test_reset_3
    test_reset_3 --> test_reset_4
    test_reset_4 --> test_reset_5
    test_reset_5 --> test_reset_6
```

## Function: `test_telemetry_basic`

- File: MMSB/src/06_utility/telemetry.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    test_telemetry_basic_0["ENTRY"]
    test_telemetry_basic_1["let telemetry = Telemetry :: new ()"]
    test_telemetry_basic_2["telemetry . record_cache_hit ()"]
    test_telemetry_basic_3["telemetry . record_cache_miss ()"]
    test_telemetry_basic_4["telemetry . record_allocation (4096)"]
    test_telemetry_basic_5["let snapshot = telemetry . snapshot ()"]
    test_telemetry_basic_6["macro assert_eq"]
    test_telemetry_basic_7["macro assert_eq"]
    test_telemetry_basic_8["macro assert_eq"]
    test_telemetry_basic_9["macro assert_eq"]
    test_telemetry_basic_10["EXIT"]
    test_telemetry_basic_0 --> test_telemetry_basic_1
    test_telemetry_basic_1 --> test_telemetry_basic_2
    test_telemetry_basic_2 --> test_telemetry_basic_3
    test_telemetry_basic_3 --> test_telemetry_basic_4
    test_telemetry_basic_4 --> test_telemetry_basic_5
    test_telemetry_basic_5 --> test_telemetry_basic_6
    test_telemetry_basic_6 --> test_telemetry_basic_7
    test_telemetry_basic_7 --> test_telemetry_basic_8
    test_telemetry_basic_8 --> test_telemetry_basic_9
    test_telemetry_basic_9 --> test_telemetry_basic_10
```

## Function: `validate_metadata_blob`

- File: MMSB/src/06_utility/invariant_checker.rs
- Branches: 1
- Loops: 0
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    validate_metadata_blob_0["ENTRY"]
    validate_metadata_blob_1["if blob . is_empty ()"]
    validate_metadata_blob_2["THEN BB"]
    validate_metadata_blob_3["return Ok (())"]
    validate_metadata_blob_4["EMPTY ELSE"]
    validate_metadata_blob_5["IF JOIN"]
    validate_metadata_blob_6["let mut cursor = 0usize"]
    validate_metadata_blob_7["let entry_count = read_u32 (blob , & mut cursor) ?"]
    validate_metadata_blob_8["for _ in 0 .. entry_count { let key_len = read_u32 (blob , & mut cursor) ? as..."]
    validate_metadata_blob_9["Ok (())"]
    validate_metadata_blob_10["EXIT"]
    validate_metadata_blob_0 --> validate_metadata_blob_1
    validate_metadata_blob_1 --> validate_metadata_blob_2
    validate_metadata_blob_2 --> validate_metadata_blob_3
    validate_metadata_blob_1 --> validate_metadata_blob_4
    validate_metadata_blob_3 --> validate_metadata_blob_5
    validate_metadata_blob_4 --> validate_metadata_blob_5
    validate_metadata_blob_5 --> validate_metadata_blob_6
    validate_metadata_blob_6 --> validate_metadata_blob_7
    validate_metadata_blob_7 --> validate_metadata_blob_8
    validate_metadata_blob_8 --> validate_metadata_blob_9
    validate_metadata_blob_9 --> validate_metadata_blob_10
```

