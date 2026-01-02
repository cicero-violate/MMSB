# CFG Group: tests

## Function: `apply_random_deltas`

- File: MMSB/tests/stress_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    apply_random_deltas_0["ENTRY"]
    apply_random_deltas_1["for _ in 0 .. count { let page_id = PageID ((rng . next_u64 () % PAGE_COUNT) ..."]
    apply_random_deltas_2["EXIT"]
    apply_random_deltas_0 --> apply_random_deltas_1
    apply_random_deltas_1 --> apply_random_deltas_2
```

## Function: `assert_throughput`

- File: MMSB/tests/stress_throughput.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    assert_throughput_0["ENTRY"]
    assert_throughput_1["let secs = duration . as_secs_f64 ()"]
    assert_throughput_2["macro assert"]
    assert_throughput_3["let throughput = processed as f64 / secs"]
    assert_throughput_4["let phase_goal = if (min_target - MULTI_WORKER_TARGET) . abs () < f64 :: EPSILON { MULTI_GOAL ..."]
    assert_throughput_5["macro println"]
    assert_throughput_6["macro assert"]
    assert_throughput_7["EXIT"]
    assert_throughput_0 --> assert_throughput_1
    assert_throughput_1 --> assert_throughput_2
    assert_throughput_2 --> assert_throughput_3
    assert_throughput_3 --> assert_throughput_4
    assert_throughput_4 --> assert_throughput_5
    assert_throughput_5 --> assert_throughput_6
    assert_throughput_6 --> assert_throughput_7
```

## Function: `build_noop_delta`

- File: MMSB/tests/stress_throughput.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    build_noop_delta_0["ENTRY"]
    build_noop_delta_1["Delta { delta_id : DeltaID (delta_id) , page_id , epoch : Epoch ((delta_id % ..."]
    build_noop_delta_2["EXIT"]
    build_noop_delta_0 --> build_noop_delta_1
    build_noop_delta_1 --> build_noop_delta_2
```

## Function: `delta`

- File: MMSB/tests/benchmark_06_tick_latency.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    delta_0["ENTRY"]
    delta_1["Delta { delta_id : DeltaID (id) , page_id : PageID (page) , epoch : Epoch (id..."]
    delta_2["EXIT"]
    delta_0 --> delta_1
    delta_1 --> delta_2
```

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

## Function: `divergence`

- File: MMSB/tests/benchmark_09_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    divergence_0["ENTRY"]
    divergence_1["a . iter () . zip (b) . map (| (x , y) | (x - y) . powi (2)) . sum :: < f64 >..."]
    divergence_2["EXIT"]
    divergence_0 --> divergence_1
    divergence_1 --> divergence_2
```

## Function: `example_checkpoint`

- File: MMSB/tests/examples_basic.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    example_checkpoint_0["ENTRY"]
    example_checkpoint_1["macro assert"]
    example_checkpoint_2["EXIT"]
    example_checkpoint_0 --> example_checkpoint_1
    example_checkpoint_1 --> example_checkpoint_2
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

## Function: `fragmentation_probe_remains_stable`

- File: MMSB/tests/stress_memory.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    fragmentation_probe_remains_stable_0["ENTRY"]
    fragmentation_probe_remains_stable_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    fragmentation_probe_remains_stable_2["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , MemoryMonitorConfi..."]
    fragmentation_probe_remains_stable_3["let mut live_pages = Vec :: new ()"]
    fragmentation_probe_remains_stable_4["let mut next_page = 1u64"]
    fragmentation_probe_remains_stable_5["for iteration in 0 .. 512 { let burst = 32 + (iteration % 32) as usize ; for ..."]
    fragmentation_probe_remains_stable_6["let snapshot = monitor . snapshot ()"]
    fragmentation_probe_remains_stable_7["macro assert"]
    fragmentation_probe_remains_stable_8["let estimate = snapshot . avg_bytes_per_page * snapshot . total_pages"]
    fragmentation_probe_remains_stable_9["macro assert"]
    fragmentation_probe_remains_stable_10["macro println"]
    fragmentation_probe_remains_stable_11["EXIT"]
    fragmentation_probe_remains_stable_0 --> fragmentation_probe_remains_stable_1
    fragmentation_probe_remains_stable_1 --> fragmentation_probe_remains_stable_2
    fragmentation_probe_remains_stable_2 --> fragmentation_probe_remains_stable_3
    fragmentation_probe_remains_stable_3 --> fragmentation_probe_remains_stable_4
    fragmentation_probe_remains_stable_4 --> fragmentation_probe_remains_stable_5
    fragmentation_probe_remains_stable_5 --> fragmentation_probe_remains_stable_6
    fragmentation_probe_remains_stable_6 --> fragmentation_probe_remains_stable_7
    fragmentation_probe_remains_stable_7 --> fragmentation_probe_remains_stable_8
    fragmentation_probe_remains_stable_8 --> fragmentation_probe_remains_stable_9
    fragmentation_probe_remains_stable_9 --> fragmentation_probe_remains_stable_10
    fragmentation_probe_remains_stable_10 --> fragmentation_probe_remains_stable_11
```

## Function: `generate_delta_batch`

- File: MMSB/tests/stress_throughput.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    generate_delta_batch_0["ENTRY"]
    generate_delta_batch_1["(0 .. count) . map (| offset | { let delta_id = start_delta + offset as u64 ;..."]
    generate_delta_batch_2["EXIT"]
    generate_delta_batch_0 --> generate_delta_batch_1
    generate_delta_batch_1 --> generate_delta_batch_2
```

## Function: `graph_validator_detects_no_cycles`

- File: MMSB/tests/benchmark_03_graph.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    graph_validator_detects_no_cycles_0["ENTRY"]
    graph_validator_detects_no_cycles_1["let graph = ShadowPageGraph :: default ()"]
    graph_validator_detects_no_cycles_2["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    graph_validator_detects_no_cycles_3["graph . add_edge (PageID (2) , PageID (3) , EdgeType :: Data)"]
    graph_validator_detects_no_cycles_4["let validator = GraphValidator :: new (& graph)"]
    graph_validator_detects_no_cycles_5["let report = validator . detect_cycles ()"]
    graph_validator_detects_no_cycles_6["macro assert"]
    graph_validator_detects_no_cycles_7["macro assert"]
    graph_validator_detects_no_cycles_8["EXIT"]
    graph_validator_detects_no_cycles_0 --> graph_validator_detects_no_cycles_1
    graph_validator_detects_no_cycles_1 --> graph_validator_detects_no_cycles_2
    graph_validator_detects_no_cycles_2 --> graph_validator_detects_no_cycles_3
    graph_validator_detects_no_cycles_3 --> graph_validator_detects_no_cycles_4
    graph_validator_detects_no_cycles_4 --> graph_validator_detects_no_cycles_5
    graph_validator_detects_no_cycles_5 --> graph_validator_detects_no_cycles_6
    graph_validator_detects_no_cycles_6 --> graph_validator_detects_no_cycles_7
    graph_validator_detects_no_cycles_7 --> graph_validator_detects_no_cycles_8
```

## Function: `incremental_gc_latency_stays_within_budget`

- File: MMSB/tests/stress_memory.rs
- Branches: 1
- Loops: 0
- Nodes: 16
- Edges: 16

```mermaid
flowchart TD
    incremental_gc_latency_stays_within_budget_0["ENTRY"]
    incremental_gc_latency_stays_within_budget_1["if cfg ! (debug_assertions)"]
    incremental_gc_latency_stays_within_budget_2["THEN BB"]
    incremental_gc_latency_stays_within_budget_3["macro eprintln"]
    incremental_gc_latency_stays_within_budget_4["return"]
    incremental_gc_latency_stays_within_budget_5["EMPTY ELSE"]
    incremental_gc_latency_stays_within_budget_6["IF JOIN"]
    incremental_gc_latency_stays_within_budget_7["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    incremental_gc_latency_stays_within_budget_8["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , MemoryMonitorConfi..."]
    incremental_gc_latency_stays_within_budget_9["for idx in 0 .. (monitor . config () . incremental_batch_pages * 4) { allocat..."]
    incremental_gc_latency_stays_within_budget_10["for _ in 0 .. 3 { monitor . snapshot () ; }"]
    incremental_gc_latency_stays_within_budget_11["let metrics = monitor . trigger_incremental_gc (monitor . config () . incremental_batch_pag..."]
    incremental_gc_latency_stays_within_budget_12["let gc_ms = metrics . duration . as_secs_f64 () * 1_000.0"]
    incremental_gc_latency_stays_within_budget_13["macro assert"]
    incremental_gc_latency_stays_within_budget_14["macro println"]
    incremental_gc_latency_stays_within_budget_15["EXIT"]
    incremental_gc_latency_stays_within_budget_0 --> incremental_gc_latency_stays_within_budget_1
    incremental_gc_latency_stays_within_budget_1 --> incremental_gc_latency_stays_within_budget_2
    incremental_gc_latency_stays_within_budget_2 --> incremental_gc_latency_stays_within_budget_3
    incremental_gc_latency_stays_within_budget_3 --> incremental_gc_latency_stays_within_budget_4
    incremental_gc_latency_stays_within_budget_1 --> incremental_gc_latency_stays_within_budget_5
    incremental_gc_latency_stays_within_budget_4 --> incremental_gc_latency_stays_within_budget_6
    incremental_gc_latency_stays_within_budget_5 --> incremental_gc_latency_stays_within_budget_6
    incremental_gc_latency_stays_within_budget_6 --> incremental_gc_latency_stays_within_budget_7
    incremental_gc_latency_stays_within_budget_7 --> incremental_gc_latency_stays_within_budget_8
    incremental_gc_latency_stays_within_budget_8 --> incremental_gc_latency_stays_within_budget_9
    incremental_gc_latency_stays_within_budget_9 --> incremental_gc_latency_stays_within_budget_10
    incremental_gc_latency_stays_within_budget_10 --> incremental_gc_latency_stays_within_budget_11
    incremental_gc_latency_stays_within_budget_11 --> incremental_gc_latency_stays_within_budget_12
    incremental_gc_latency_stays_within_budget_12 --> incremental_gc_latency_stays_within_budget_13
    incremental_gc_latency_stays_within_budget_13 --> incremental_gc_latency_stays_within_budget_14
    incremental_gc_latency_stays_within_budget_14 --> incremental_gc_latency_stays_within_budget_15
```

## Function: `integrity_checker_accepts_valid_delta`

- File: MMSB/tests/benchmark_02_integrity.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    integrity_checker_accepts_valid_delta_0["ENTRY"]
    integrity_checker_accepts_valid_delta_1["let registry = Arc :: new (DeviceBufferRegistry :: default ())"]
    integrity_checker_accepts_valid_delta_2["let page = Arc :: new (Page :: new (PageID (1) , 8 , PageLocation :: Cpu) . unwrap ())"]
    integrity_checker_accepts_valid_delta_3["registry . insert (page)"]
    integrity_checker_accepts_valid_delta_4["let delta = Delta { delta_id : DeltaID (1) , page_id : PageID (1) , epoch : Epoch (1) , m..."]
    integrity_checker_accepts_valid_delta_5["let mut checker = DeltaIntegrityChecker :: new (Arc :: clone (& registry))"]
    integrity_checker_accepts_valid_delta_6["let report = checker . validate (& [delta])"]
    integrity_checker_accepts_valid_delta_7["macro assert"]
    integrity_checker_accepts_valid_delta_8["EXIT"]
    integrity_checker_accepts_valid_delta_0 --> integrity_checker_accepts_valid_delta_1
    integrity_checker_accepts_valid_delta_1 --> integrity_checker_accepts_valid_delta_2
    integrity_checker_accepts_valid_delta_2 --> integrity_checker_accepts_valid_delta_3
    integrity_checker_accepts_valid_delta_3 --> integrity_checker_accepts_valid_delta_4
    integrity_checker_accepts_valid_delta_4 --> integrity_checker_accepts_valid_delta_5
    integrity_checker_accepts_valid_delta_5 --> integrity_checker_accepts_valid_delta_6
    integrity_checker_accepts_valid_delta_6 --> integrity_checker_accepts_valid_delta_7
    integrity_checker_accepts_valid_delta_7 --> integrity_checker_accepts_valid_delta_8
```

## Function: `invariant_checker_reports_success`

- File: MMSB/tests/benchmark_08_invariants.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    invariant_checker_reports_success_0["ENTRY"]
    invariant_checker_reports_success_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    invariant_checker_reports_success_2["allocator . allocate_raw (PageID (42) , 256 , Some (PageLocation :: Cpu)) . u..."]
    invariant_checker_reports_success_3["let graph = ShadowPageGraph :: default ()"]
    invariant_checker_reports_success_4["let ctx = InvariantContext { allocator : Some (& allocator) , graph : Some (& graph) , ..."]
    invariant_checker_reports_success_5["let checker = InvariantChecker :: with_builtins ()"]
    invariant_checker_reports_success_6["for result in checker . run (& ctx) { assert ! (result . passed) ; }"]
    invariant_checker_reports_success_7["EXIT"]
    invariant_checker_reports_success_0 --> invariant_checker_reports_success_1
    invariant_checker_reports_success_1 --> invariant_checker_reports_success_2
    invariant_checker_reports_success_2 --> invariant_checker_reports_success_3
    invariant_checker_reports_success_3 --> invariant_checker_reports_success_4
    invariant_checker_reports_success_4 --> invariant_checker_reports_success_5
    invariant_checker_reports_success_5 --> invariant_checker_reports_success_6
    invariant_checker_reports_success_6 --> invariant_checker_reports_success_7
```

## Function: `make_delta`

- File: MMSB/tests/benchmark_05_throughput.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    make_delta_0["ENTRY"]
    make_delta_1["Delta { delta_id : DeltaID (id) , page_id : PageID (page) , epoch : Epoch (id..."]
    make_delta_2["EXIT"]
    make_delta_0 --> make_delta_1
    make_delta_1 --> make_delta_2
```

## Function: `memory_monitor_enforces_limits`

- File: MMSB/tests/benchmark_07_memory.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    memory_monitor_enforces_limits_0["ENTRY"]
    memory_monitor_enforces_limits_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    memory_monitor_enforces_limits_2["for id in 1 ..= 128 { allocator . allocate_raw (PageID (id) , 1024 , Some (Pa..."]
    memory_monitor_enforces_limits_3["let monitor = MemoryMonitor :: with_config (Arc :: clone (& allocator) , MemoryMonitorConfi..."]
    memory_monitor_enforces_limits_4["let snapshot = monitor . snapshot ()"]
    memory_monitor_enforces_limits_5["macro assert_eq"]
    memory_monitor_enforces_limits_6["macro assert"]
    memory_monitor_enforces_limits_7["let metrics = monitor . trigger_incremental_gc (32) . unwrap ()"]
    memory_monitor_enforces_limits_8["macro assert"]
    memory_monitor_enforces_limits_9["EXIT"]
    memory_monitor_enforces_limits_0 --> memory_monitor_enforces_limits_1
    memory_monitor_enforces_limits_1 --> memory_monitor_enforces_limits_2
    memory_monitor_enforces_limits_2 --> memory_monitor_enforces_limits_3
    memory_monitor_enforces_limits_3 --> memory_monitor_enforces_limits_4
    memory_monitor_enforces_limits_4 --> memory_monitor_enforces_limits_5
    memory_monitor_enforces_limits_5 --> memory_monitor_enforces_limits_6
    memory_monitor_enforces_limits_6 --> memory_monitor_enforces_limits_7
    memory_monitor_enforces_limits_7 --> memory_monitor_enforces_limits_8
    memory_monitor_enforces_limits_8 --> memory_monitor_enforces_limits_9
```

## Function: `multi_thread_10m_deltas_per_sec`

- File: MMSB/tests/stress_throughput.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    multi_thread_10m_deltas_per_sec_0["ENTRY"]
    multi_thread_10m_deltas_per_sec_1["if cfg ! (debug_assertions)"]
    multi_thread_10m_deltas_per_sec_2["THEN BB"]
    multi_thread_10m_deltas_per_sec_3["macro eprintln"]
    multi_thread_10m_deltas_per_sec_4["return"]
    multi_thread_10m_deltas_per_sec_5["EMPTY ELSE"]
    multi_thread_10m_deltas_per_sec_6["IF JOIN"]
    multi_thread_10m_deltas_per_sec_7["let allocator = prepare_allocator ()"]
    multi_thread_10m_deltas_per_sec_8["run_throughput_benchmark (& allocator , 8 , MULTI_WORKER_TARGET)"]
    multi_thread_10m_deltas_per_sec_9["EXIT"]
    multi_thread_10m_deltas_per_sec_0 --> multi_thread_10m_deltas_per_sec_1
    multi_thread_10m_deltas_per_sec_1 --> multi_thread_10m_deltas_per_sec_2
    multi_thread_10m_deltas_per_sec_2 --> multi_thread_10m_deltas_per_sec_3
    multi_thread_10m_deltas_per_sec_3 --> multi_thread_10m_deltas_per_sec_4
    multi_thread_10m_deltas_per_sec_1 --> multi_thread_10m_deltas_per_sec_5
    multi_thread_10m_deltas_per_sec_4 --> multi_thread_10m_deltas_per_sec_6
    multi_thread_10m_deltas_per_sec_5 --> multi_thread_10m_deltas_per_sec_6
    multi_thread_10m_deltas_per_sec_6 --> multi_thread_10m_deltas_per_sec_7
    multi_thread_10m_deltas_per_sec_7 --> multi_thread_10m_deltas_per_sec_8
    multi_thread_10m_deltas_per_sec_8 --> multi_thread_10m_deltas_per_sec_9
```

## Function: `mutate_graph`

- File: MMSB/tests/stress_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    mutate_graph_0["ENTRY"]
    mutate_graph_1["for _ in 0 .. 4 { let mut from = PageID ((rng . next_u64 () % PAGE_COUNT) + 1..."]
    mutate_graph_2["EXIT"]
    mutate_graph_0 --> mutate_graph_1
    mutate_graph_1 --> mutate_graph_2
```

## Function: `one_million_pages_under_1gb_full_run`

- File: MMSB/tests/stress_memory.rs
- Branches: 1
- Loops: 0
- Nodes: 16
- Edges: 16

```mermaid
flowchart TD
    one_million_pages_under_1gb_full_run_0["ENTRY"]
    one_million_pages_under_1gb_full_run_1["if cfg ! (debug_assertions)"]
    one_million_pages_under_1gb_full_run_2["THEN BB"]
    one_million_pages_under_1gb_full_run_3["macro eprintln"]
    one_million_pages_under_1gb_full_run_4["return"]
    one_million_pages_under_1gb_full_run_5["EMPTY ELSE"]
    one_million_pages_under_1gb_full_run_6["IF JOIN"]
    one_million_pages_under_1gb_full_run_7["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    one_million_pages_under_1gb_full_run_8["for idx in 0 .. TARGET_PAGES { allocator . allocate_raw (PageID (idx as u64 +..."]
    one_million_pages_under_1gb_full_run_9["let monitor = MemoryMonitor :: new (Arc :: clone (& allocator))"]
    one_million_pages_under_1gb_full_run_10["let snapshot = monitor . snapshot ()"]
    one_million_pages_under_1gb_full_run_11["macro assert_eq"]
    one_million_pages_under_1gb_full_run_12["macro assert"]
    one_million_pages_under_1gb_full_run_13["macro assert"]
    one_million_pages_under_1gb_full_run_14["macro println"]
    one_million_pages_under_1gb_full_run_15["EXIT"]
    one_million_pages_under_1gb_full_run_0 --> one_million_pages_under_1gb_full_run_1
    one_million_pages_under_1gb_full_run_1 --> one_million_pages_under_1gb_full_run_2
    one_million_pages_under_1gb_full_run_2 --> one_million_pages_under_1gb_full_run_3
    one_million_pages_under_1gb_full_run_3 --> one_million_pages_under_1gb_full_run_4
    one_million_pages_under_1gb_full_run_1 --> one_million_pages_under_1gb_full_run_5
    one_million_pages_under_1gb_full_run_4 --> one_million_pages_under_1gb_full_run_6
    one_million_pages_under_1gb_full_run_5 --> one_million_pages_under_1gb_full_run_6
    one_million_pages_under_1gb_full_run_6 --> one_million_pages_under_1gb_full_run_7
    one_million_pages_under_1gb_full_run_7 --> one_million_pages_under_1gb_full_run_8
    one_million_pages_under_1gb_full_run_8 --> one_million_pages_under_1gb_full_run_9
    one_million_pages_under_1gb_full_run_9 --> one_million_pages_under_1gb_full_run_10
    one_million_pages_under_1gb_full_run_10 --> one_million_pages_under_1gb_full_run_11
    one_million_pages_under_1gb_full_run_11 --> one_million_pages_under_1gb_full_run_12
    one_million_pages_under_1gb_full_run_12 --> one_million_pages_under_1gb_full_run_13
    one_million_pages_under_1gb_full_run_13 --> one_million_pages_under_1gb_full_run_14
    one_million_pages_under_1gb_full_run_14 --> one_million_pages_under_1gb_full_run_15
```

## Function: `prepare_allocator`

- File: MMSB/tests/stress_throughput.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    prepare_allocator_0["ENTRY"]
    prepare_allocator_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    prepare_allocator_2["for id in 0 .. PAGE_COUNT { allocator . allocate_raw (PageID (id + 1) , PAGE_..."]
    prepare_allocator_3["allocator"]
    prepare_allocator_4["EXIT"]
    prepare_allocator_0 --> prepare_allocator_1
    prepare_allocator_1 --> prepare_allocator_2
    prepare_allocator_2 --> prepare_allocator_3
    prepare_allocator_3 --> prepare_allocator_4
```

## Function: `provenance_tracker_resolves_with_cache`

- File: MMSB/tests/benchmark_10_provenance.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    provenance_tracker_resolves_with_cache_0["ENTRY"]
    provenance_tracker_resolves_with_cache_1["let graph = Arc :: new (ShadowPageGraph :: default ())"]
    provenance_tracker_resolves_with_cache_2["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    provenance_tracker_resolves_with_cache_3["graph . add_edge (PageID (2) , PageID (3) , EdgeType :: Data)"]
    provenance_tracker_resolves_with_cache_4["graph . add_edge (PageID (3) , PageID (4) , EdgeType :: Data)"]
    provenance_tracker_resolves_with_cache_5["let tracker = ProvenanceTracker :: with_capacity (Arc :: clone (& graph) , 8 , 8)"]
    provenance_tracker_resolves_with_cache_6["let result = tracker . resolve (PageID (4))"]
    provenance_tracker_resolves_with_cache_7["macro assert"]
    provenance_tracker_resolves_with_cache_8["macro assert"]
    provenance_tracker_resolves_with_cache_9["let cached = tracker . resolve (PageID (4))"]
    provenance_tracker_resolves_with_cache_10["macro assert"]
    provenance_tracker_resolves_with_cache_11["EXIT"]
    provenance_tracker_resolves_with_cache_0 --> provenance_tracker_resolves_with_cache_1
    provenance_tracker_resolves_with_cache_1 --> provenance_tracker_resolves_with_cache_2
    provenance_tracker_resolves_with_cache_2 --> provenance_tracker_resolves_with_cache_3
    provenance_tracker_resolves_with_cache_3 --> provenance_tracker_resolves_with_cache_4
    provenance_tracker_resolves_with_cache_4 --> provenance_tracker_resolves_with_cache_5
    provenance_tracker_resolves_with_cache_5 --> provenance_tracker_resolves_with_cache_6
    provenance_tracker_resolves_with_cache_6 --> provenance_tracker_resolves_with_cache_7
    provenance_tracker_resolves_with_cache_7 --> provenance_tracker_resolves_with_cache_8
    provenance_tracker_resolves_with_cache_8 --> provenance_tracker_resolves_with_cache_9
    provenance_tracker_resolves_with_cache_9 --> provenance_tracker_resolves_with_cache_10
    provenance_tracker_resolves_with_cache_10 --> provenance_tracker_resolves_with_cache_11
```

## Function: `purity_validator_covers_semiring_operations`

- File: MMSB/tests/benchmark_04_purity.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    purity_validator_covers_semiring_operations_0["ENTRY"]
    purity_validator_covers_semiring_operations_1["let validator = PurityValidator :: default ()"]
    purity_validator_covers_semiring_operations_2["let boolean_samples = vec ! [vec ! [true , false] , vec ! [true , true]]"]
    purity_validator_covers_semiring_operations_3["macro assert"]
    purity_validator_covers_semiring_operations_4["let tropical_samples = vec ! [vec ! [0.0 , 1.0 , 2.0] , vec ! [3.0 , 4.0]]"]
    purity_validator_covers_semiring_operations_5["macro assert"]
    purity_validator_covers_semiring_operations_6["EXIT"]
    purity_validator_covers_semiring_operations_0 --> purity_validator_covers_semiring_operations_1
    purity_validator_covers_semiring_operations_1 --> purity_validator_covers_semiring_operations_2
    purity_validator_covers_semiring_operations_2 --> purity_validator_covers_semiring_operations_3
    purity_validator_covers_semiring_operations_3 --> purity_validator_covers_semiring_operations_4
    purity_validator_covers_semiring_operations_4 --> purity_validator_covers_semiring_operations_5
    purity_validator_covers_semiring_operations_5 --> purity_validator_covers_semiring_operations_6
```

## Function: `random_edge_type`

- File: MMSB/tests/stress_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    random_edge_type_0["ENTRY"]
    random_edge_type_1["match rng . next_in_range (4) { 0 => EdgeType :: Data , 1 => EdgeType :: Cont..."]
    random_edge_type_2["EXIT"]
    random_edge_type_0 --> random_edge_type_1
    random_edge_type_1 --> random_edge_type_2
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

## Function: `replay_validator_divergence_under_threshold`

- File: MMSB/tests/benchmark_01_replay.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    replay_validator_divergence_under_threshold_0["ENTRY"]
    replay_validator_divergence_under_threshold_1["let path = temp_log_path ()"]
    replay_validator_divergence_under_threshold_2["let log = TransactionLog :: new (& path) . unwrap ()"]
    replay_validator_divergence_under_threshold_3["let allocator = PageAllocator :: new (PageAllocatorConfig :: default ())"]
    replay_validator_divergence_under_threshold_4["allocator . allocate_raw (PageID (1) , 1024 , Some (PageLocation :: Cpu)) . u..."]
    replay_validator_divergence_under_threshold_5["let mut validator = ReplayValidator :: new (1e-9)"]
    replay_validator_divergence_under_threshold_6["let checkpoint = validator . record_checkpoint (& allocator , & log) . unwrap ()"]
    replay_validator_divergence_under_threshold_7["let report = validator . validate_allocator (checkpoint , & allocator) . unwrap ()"]
    replay_validator_divergence_under_threshold_8["macro assert"]
    replay_validator_divergence_under_threshold_9["std :: fs :: remove_file (path) . ok ()"]
    replay_validator_divergence_under_threshold_10["EXIT"]
    replay_validator_divergence_under_threshold_0 --> replay_validator_divergence_under_threshold_1
    replay_validator_divergence_under_threshold_1 --> replay_validator_divergence_under_threshold_2
    replay_validator_divergence_under_threshold_2 --> replay_validator_divergence_under_threshold_3
    replay_validator_divergence_under_threshold_3 --> replay_validator_divergence_under_threshold_4
    replay_validator_divergence_under_threshold_4 --> replay_validator_divergence_under_threshold_5
    replay_validator_divergence_under_threshold_5 --> replay_validator_divergence_under_threshold_6
    replay_validator_divergence_under_threshold_6 --> replay_validator_divergence_under_threshold_7
    replay_validator_divergence_under_threshold_7 --> replay_validator_divergence_under_threshold_8
    replay_validator_divergence_under_threshold_8 --> replay_validator_divergence_under_threshold_9
    replay_validator_divergence_under_threshold_9 --> replay_validator_divergence_under_threshold_10
```

## Function: `run_throughput_benchmark`

- File: MMSB/tests/stress_throughput.rs
- Branches: 1
- Loops: 0
- Nodes: 14
- Edges: 14

```mermaid
flowchart TD
    run_throughput_benchmark_0["ENTRY"]
    run_throughput_benchmark_1["let engine = ThroughputEngine :: new (Arc :: clone (allocator) , workers , CHUNK_SIZE)"]
    run_throughput_benchmark_2["let mut total_processed = 0usize"]
    run_throughput_benchmark_3["let mut total_duration = Duration :: default ()"]
    run_throughput_benchmark_4["let mut next_delta = 0u64"]
    run_throughput_benchmark_5["let mut sample_checkpoint : Option < (usize , Duration) > = None"]
    run_throughput_benchmark_6["while total_processed < TOTAL_DELTAS { let remaining = TOTAL_DELTAS - total_p..."]
    run_throughput_benchmark_7["if let Some ((processed , duration)) = sample_checkpoint"]
    run_throughput_benchmark_8["THEN BB"]
    run_throughput_benchmark_9["assert_throughput (processed , duration , min_target)"]
    run_throughput_benchmark_10["EMPTY ELSE"]
    run_throughput_benchmark_11["IF JOIN"]
    run_throughput_benchmark_12["assert_throughput (total_processed , total_duration , min_target)"]
    run_throughput_benchmark_13["EXIT"]
    run_throughput_benchmark_0 --> run_throughput_benchmark_1
    run_throughput_benchmark_1 --> run_throughput_benchmark_2
    run_throughput_benchmark_2 --> run_throughput_benchmark_3
    run_throughput_benchmark_3 --> run_throughput_benchmark_4
    run_throughput_benchmark_4 --> run_throughput_benchmark_5
    run_throughput_benchmark_5 --> run_throughput_benchmark_6
    run_throughput_benchmark_6 --> run_throughput_benchmark_7
    run_throughput_benchmark_7 --> run_throughput_benchmark_8
    run_throughput_benchmark_8 --> run_throughput_benchmark_9
    run_throughput_benchmark_7 --> run_throughput_benchmark_10
    run_throughput_benchmark_9 --> run_throughput_benchmark_11
    run_throughput_benchmark_10 --> run_throughput_benchmark_11
    run_throughput_benchmark_11 --> run_throughput_benchmark_12
    run_throughput_benchmark_12 --> run_throughput_benchmark_13
```

## Function: `sampled_allocation_projects_under_budget`

- File: MMSB/tests/stress_memory.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    sampled_allocation_projects_under_budget_0["ENTRY"]
    sampled_allocation_projects_under_budget_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    sampled_allocation_projects_under_budget_2["let monitor = MemoryMonitor :: new (Arc :: clone (& allocator))"]
    sampled_allocation_projects_under_budget_3["for idx in 0 .. SAMPLE_PAGES { allocator . allocate_raw (PageID (idx as u64 +..."]
    sampled_allocation_projects_under_budget_4["let snapshot = monitor . snapshot ()"]
    sampled_allocation_projects_under_budget_5["macro assert"]
    sampled_allocation_projects_under_budget_6["let projected_total = snapshot . avg_bytes_per_page . saturating_mul (TARGET_PAGES)"]
    sampled_allocation_projects_under_budget_7["let one_gib = 1usize << 30"]
    sampled_allocation_projects_under_budget_8["macro assert"]
    sampled_allocation_projects_under_budget_9["macro println"]
    sampled_allocation_projects_under_budget_10["EXIT"]
    sampled_allocation_projects_under_budget_0 --> sampled_allocation_projects_under_budget_1
    sampled_allocation_projects_under_budget_1 --> sampled_allocation_projects_under_budget_2
    sampled_allocation_projects_under_budget_2 --> sampled_allocation_projects_under_budget_3
    sampled_allocation_projects_under_budget_3 --> sampled_allocation_projects_under_budget_4
    sampled_allocation_projects_under_budget_4 --> sampled_allocation_projects_under_budget_5
    sampled_allocation_projects_under_budget_5 --> sampled_allocation_projects_under_budget_6
    sampled_allocation_projects_under_budget_6 --> sampled_allocation_projects_under_budget_7
    sampled_allocation_projects_under_budget_7 --> sampled_allocation_projects_under_budget_8
    sampled_allocation_projects_under_budget_8 --> sampled_allocation_projects_under_budget_9
    sampled_allocation_projects_under_budget_9 --> sampled_allocation_projects_under_budget_10
```

## Function: `simulate`

- File: MMSB/tests/benchmark_09_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    simulate_0["ENTRY"]
    simulate_1["state . iter () . enumerate () . map (| (idx , value) | { let bias = (idx as ..."]
    simulate_2["EXIT"]
    simulate_0 --> simulate_1
    simulate_1 --> simulate_2
```

## Function: `single_thread_1m_deltas_per_sec`

- File: MMSB/tests/stress_throughput.rs
- Branches: 1
- Loops: 0
- Nodes: 10
- Edges: 10

```mermaid
flowchart TD
    single_thread_1m_deltas_per_sec_0["ENTRY"]
    single_thread_1m_deltas_per_sec_1["if cfg ! (debug_assertions)"]
    single_thread_1m_deltas_per_sec_2["THEN BB"]
    single_thread_1m_deltas_per_sec_3["macro eprintln"]
    single_thread_1m_deltas_per_sec_4["return"]
    single_thread_1m_deltas_per_sec_5["EMPTY ELSE"]
    single_thread_1m_deltas_per_sec_6["IF JOIN"]
    single_thread_1m_deltas_per_sec_7["let allocator = prepare_allocator ()"]
    single_thread_1m_deltas_per_sec_8["run_throughput_benchmark (& allocator , 1 , 1_000_000.0)"]
    single_thread_1m_deltas_per_sec_9["EXIT"]
    single_thread_1m_deltas_per_sec_0 --> single_thread_1m_deltas_per_sec_1
    single_thread_1m_deltas_per_sec_1 --> single_thread_1m_deltas_per_sec_2
    single_thread_1m_deltas_per_sec_2 --> single_thread_1m_deltas_per_sec_3
    single_thread_1m_deltas_per_sec_3 --> single_thread_1m_deltas_per_sec_4
    single_thread_1m_deltas_per_sec_1 --> single_thread_1m_deltas_per_sec_5
    single_thread_1m_deltas_per_sec_4 --> single_thread_1m_deltas_per_sec_6
    single_thread_1m_deltas_per_sec_5 --> single_thread_1m_deltas_per_sec_6
    single_thread_1m_deltas_per_sec_6 --> single_thread_1m_deltas_per_sec_7
    single_thread_1m_deltas_per_sec_7 --> single_thread_1m_deltas_per_sec_8
    single_thread_1m_deltas_per_sec_8 --> single_thread_1m_deltas_per_sec_9
```

## Function: `stability_resists_small_noise`

- File: MMSB/tests/benchmark_09_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    stability_resists_small_noise_0["ENTRY"]
    stability_resists_small_noise_1["let mut rng = NoiseRng :: new (42)"]
    stability_resists_small_noise_2["let mut baseline = vec ! [0.0 ; 64]"]
    stability_resists_small_noise_3["let mut perturbed = baseline . clone ()"]
    stability_resists_small_noise_4["let mut max_divergence : f64 = 0.0"]
    stability_resists_small_noise_5["for _ in 0 .. 128 { baseline = simulate (& baseline , 0.0) ; let noise = rng ..."]
    stability_resists_small_noise_6["macro assert"]
    stability_resists_small_noise_7["EXIT"]
    stability_resists_small_noise_0 --> stability_resists_small_noise_1
    stability_resists_small_noise_1 --> stability_resists_small_noise_2
    stability_resists_small_noise_2 --> stability_resists_small_noise_3
    stability_resists_small_noise_3 --> stability_resists_small_noise_4
    stability_resists_small_noise_4 --> stability_resists_small_noise_5
    stability_resists_small_noise_5 --> stability_resists_small_noise_6
    stability_resists_small_noise_6 --> stability_resists_small_noise_7
```

## Function: `temp_log_path`

- File: MMSB/tests/benchmark_01_replay.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    temp_log_path_0["ENTRY"]
    temp_log_path_1["let mut path = std :: env :: temp_dir ()"]
    temp_log_path_2["path . push (format ! ('mmsb_replay_{}.log' , SystemTime :: now () . duration..."]
    temp_log_path_3["path"]
    temp_log_path_4["EXIT"]
    temp_log_path_0 --> temp_log_path_1
    temp_log_path_1 --> temp_log_path_2
    temp_log_path_2 --> temp_log_path_3
    temp_log_path_3 --> temp_log_path_4
```

## Function: `ten_thousand_cycles_no_violations`

- File: MMSB/tests/stress_stability.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    ten_thousand_cycles_no_violations_0["ENTRY"]
    ten_thousand_cycles_no_violations_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    ten_thousand_cycles_no_violations_2["for id in 0 .. PAGE_COUNT { allocator . allocate_raw (PageID (id + 1) , PAGE_..."]
    ten_thousand_cycles_no_violations_3["let graph = ShadowPageGraph :: default ()"]
    ten_thousand_cycles_no_violations_4["let registry = DeviceBufferRegistry :: default ()"]
    ten_thousand_cycles_no_violations_5["let checker = InvariantChecker :: with_builtins ()"]
    ten_thousand_cycles_no_violations_6["let mut rng = Lcg :: new (0x5eed_cafe_d00d_f001)"]
    ten_thousand_cycles_no_violations_7["let mut delta_id = 0u64"]
    ten_thousand_cycles_no_violations_8["let mut probe = SignalProbe :: new (DELTA_WIDTH , 0.5)"]
    ten_thousand_cycles_no_violations_9["for cycle in 0 .. CYCLES { apply_random_deltas (& allocator , & mut rng , DEL..."]
    ten_thousand_cycles_no_violations_10["macro println"]
    ten_thousand_cycles_no_violations_11["EXIT"]
    ten_thousand_cycles_no_violations_0 --> ten_thousand_cycles_no_violations_1
    ten_thousand_cycles_no_violations_1 --> ten_thousand_cycles_no_violations_2
    ten_thousand_cycles_no_violations_2 --> ten_thousand_cycles_no_violations_3
    ten_thousand_cycles_no_violations_3 --> ten_thousand_cycles_no_violations_4
    ten_thousand_cycles_no_violations_4 --> ten_thousand_cycles_no_violations_5
    ten_thousand_cycles_no_violations_5 --> ten_thousand_cycles_no_violations_6
    ten_thousand_cycles_no_violations_6 --> ten_thousand_cycles_no_violations_7
    ten_thousand_cycles_no_violations_7 --> ten_thousand_cycles_no_violations_8
    ten_thousand_cycles_no_violations_8 --> ten_thousand_cycles_no_violations_9
    ten_thousand_cycles_no_violations_9 --> ten_thousand_cycles_no_violations_10
    ten_thousand_cycles_no_violations_10 --> ten_thousand_cycles_no_violations_11
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
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    test_thread_safe_allocator_0["ENTRY"]
    test_thread_safe_allocator_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    test_thread_safe_allocator_2["let handles : Vec < _ > = (0 .. 16) . map (| i | { let alloc = Arc :: clone (& allocator) ; std :: thre..."]
    test_thread_safe_allocator_3["for h in handles { h . join () . unwrap () ; }"]
    test_thread_safe_allocator_4["macro assert_eq"]
    test_thread_safe_allocator_5["EXIT"]
    test_thread_safe_allocator_0 --> test_thread_safe_allocator_1
    test_thread_safe_allocator_1 --> test_thread_safe_allocator_2
    test_thread_safe_allocator_2 --> test_thread_safe_allocator_3
    test_thread_safe_allocator_3 --> test_thread_safe_allocator_4
    test_thread_safe_allocator_4 --> test_thread_safe_allocator_5
```

## Function: `throughput_engine_exceeds_minimum_rate`

- File: MMSB/tests/benchmark_05_throughput.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    throughput_engine_exceeds_minimum_rate_0["ENTRY"]
    throughput_engine_exceeds_minimum_rate_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    throughput_engine_exceeds_minimum_rate_2["for id in 1 ..= 8 { allocator . allocate_raw (PageID (id) , 16 , Some (PageLo..."]
    throughput_engine_exceeds_minimum_rate_3["let engine = ThroughputEngine :: new (Arc :: clone (& allocator) , 2 , 32)"]
    throughput_engine_exceeds_minimum_rate_4["let deltas : Vec < _ > = (0 .. 512) . map (| idx | make_delta (idx , (idx % 8) + 1)) . collect ()"]
    throughput_engine_exceeds_minimum_rate_5["let metrics = engine . process_parallel (deltas) . unwrap ()"]
    throughput_engine_exceeds_minimum_rate_6["macro assert_eq"]
    throughput_engine_exceeds_minimum_rate_7["macro assert"]
    throughput_engine_exceeds_minimum_rate_8["EXIT"]
    throughput_engine_exceeds_minimum_rate_0 --> throughput_engine_exceeds_minimum_rate_1
    throughput_engine_exceeds_minimum_rate_1 --> throughput_engine_exceeds_minimum_rate_2
    throughput_engine_exceeds_minimum_rate_2 --> throughput_engine_exceeds_minimum_rate_3
    throughput_engine_exceeds_minimum_rate_3 --> throughput_engine_exceeds_minimum_rate_4
    throughput_engine_exceeds_minimum_rate_4 --> throughput_engine_exceeds_minimum_rate_5
    throughput_engine_exceeds_minimum_rate_5 --> throughput_engine_exceeds_minimum_rate_6
    throughput_engine_exceeds_minimum_rate_6 --> throughput_engine_exceeds_minimum_rate_7
    throughput_engine_exceeds_minimum_rate_7 --> throughput_engine_exceeds_minimum_rate_8
```

## Function: `tick_latency_stays_within_budget`

- File: MMSB/tests/benchmark_06_tick_latency.rs
- Branches: 0
- Loops: 0
- Nodes: 13
- Edges: 12

```mermaid
flowchart TD
    tick_latency_stays_within_budget_0["ENTRY"]
    tick_latency_stays_within_budget_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    tick_latency_stays_within_budget_2["for id in 1 ..= 4 { allocator . allocate_raw (PageID (id) , 8 , Some (PageLoc..."]
    tick_latency_stays_within_budget_3["let throughput = ThroughputEngine :: new (Arc :: clone (& allocator) , 2 , 16)"]
    tick_latency_stays_within_budget_4["let graph = Arc :: new (ShadowPageGraph :: default ())"]
    tick_latency_stays_within_budget_5["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    tick_latency_stays_within_budget_6["graph . add_edge (PageID (2) , PageID (3) , EdgeType :: Data)"]
    tick_latency_stays_within_budget_7["let memory = Arc :: new (MemoryMonitor :: with_config (Arc :: clone (& allocator) , Memory..."]
    tick_latency_stays_within_budget_8["let orchestrator = TickOrchestrator :: new (throughput , graph , memory)"]
    tick_latency_stays_within_budget_9["let deltas : Vec < _ > = (0 .. 64) . map (| idx | delta (idx , (idx % 4) + 1)) . collect ()"]
    tick_latency_stays_within_budget_10["let metrics = orchestrator . execute_tick (deltas) . unwrap ()"]
    tick_latency_stays_within_budget_11["macro assert"]
    tick_latency_stays_within_budget_12["EXIT"]
    tick_latency_stays_within_budget_0 --> tick_latency_stays_within_budget_1
    tick_latency_stays_within_budget_1 --> tick_latency_stays_within_budget_2
    tick_latency_stays_within_budget_2 --> tick_latency_stays_within_budget_3
    tick_latency_stays_within_budget_3 --> tick_latency_stays_within_budget_4
    tick_latency_stays_within_budget_4 --> tick_latency_stays_within_budget_5
    tick_latency_stays_within_budget_5 --> tick_latency_stays_within_budget_6
    tick_latency_stays_within_budget_6 --> tick_latency_stays_within_budget_7
    tick_latency_stays_within_budget_7 --> tick_latency_stays_within_budget_8
    tick_latency_stays_within_budget_8 --> tick_latency_stays_within_budget_9
    tick_latency_stays_within_budget_9 --> tick_latency_stays_within_budget_10
    tick_latency_stays_within_budget_10 --> tick_latency_stays_within_budget_11
    tick_latency_stays_within_budget_11 --> tick_latency_stays_within_budget_12
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

