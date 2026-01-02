# CFG Group: src/04_propagation

## Function: `applies_batches_in_parallel`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    applies_batches_in_parallel_0["ENTRY"]
    applies_batches_in_parallel_1["let allocator = Arc :: new (PageAllocator :: new (Default :: default ()))"]
    applies_batches_in_parallel_2["allocator . allocate_raw (PageID (1) , 4 , Some (PageLocation :: Cpu)) . unwr..."]
    applies_batches_in_parallel_3["allocator . allocate_raw (PageID (2) , 4 , Some (PageLocation :: Cpu)) . unwr..."]
    applies_batches_in_parallel_4["let engine = ThroughputEngine :: new (Arc :: clone (& allocator) , 2 , 4)"]
    applies_batches_in_parallel_5["let deltas = vec ! [make_delta (1 , 1 , b'\x01\x02\x03\x04') , make_delta (2 , 2 , b'\xAA\..."]
    applies_batches_in_parallel_6["let metrics = engine . process_parallel (deltas) . unwrap ()"]
    applies_batches_in_parallel_7["macro assert_eq"]
    applies_batches_in_parallel_8["unsafe { let page1 = & mut * allocator . acquire_page (PageID (1)) . unwrap (..."]
    applies_batches_in_parallel_9["EXIT"]
    applies_batches_in_parallel_0 --> applies_batches_in_parallel_1
    applies_batches_in_parallel_1 --> applies_batches_in_parallel_2
    applies_batches_in_parallel_2 --> applies_batches_in_parallel_3
    applies_batches_in_parallel_3 --> applies_batches_in_parallel_4
    applies_batches_in_parallel_4 --> applies_batches_in_parallel_5
    applies_batches_in_parallel_5 --> applies_batches_in_parallel_6
    applies_batches_in_parallel_6 --> applies_batches_in_parallel_7
    applies_batches_in_parallel_7 --> applies_batches_in_parallel_8
    applies_batches_in_parallel_8 --> applies_batches_in_parallel_9
```

## Function: `chunk_partitions`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 2
- Loops: 0
- Nodes: 17
- Edges: 18

```mermaid
flowchart TD
    chunk_partitions_0["ENTRY"]
    chunk_partitions_1["if partitions . is_empty ()"]
    chunk_partitions_2["THEN BB"]
    chunk_partitions_3["return Vec :: new ()"]
    chunk_partitions_4["EMPTY ELSE"]
    chunk_partitions_5["IF JOIN"]
    chunk_partitions_6["let chunk_size = ((partitions . len () + workers - 1) / workers) . max (1)"]
    chunk_partitions_7["let mut chunks = Vec :: new ()"]
    chunk_partitions_8["let mut current = Vec :: with_capacity (chunk_size)"]
    chunk_partitions_9["for entry in partitions { current . push (entry) ; if current . len () == chu..."]
    chunk_partitions_10["if ! current . is_empty ()"]
    chunk_partitions_11["THEN BB"]
    chunk_partitions_12["chunks . push (current)"]
    chunk_partitions_13["EMPTY ELSE"]
    chunk_partitions_14["IF JOIN"]
    chunk_partitions_15["chunks"]
    chunk_partitions_16["EXIT"]
    chunk_partitions_0 --> chunk_partitions_1
    chunk_partitions_1 --> chunk_partitions_2
    chunk_partitions_2 --> chunk_partitions_3
    chunk_partitions_1 --> chunk_partitions_4
    chunk_partitions_3 --> chunk_partitions_5
    chunk_partitions_4 --> chunk_partitions_5
    chunk_partitions_5 --> chunk_partitions_6
    chunk_partitions_6 --> chunk_partitions_7
    chunk_partitions_7 --> chunk_partitions_8
    chunk_partitions_8 --> chunk_partitions_9
    chunk_partitions_9 --> chunk_partitions_10
    chunk_partitions_10 --> chunk_partitions_11
    chunk_partitions_11 --> chunk_partitions_12
    chunk_partitions_10 --> chunk_partitions_13
    chunk_partitions_12 --> chunk_partitions_14
    chunk_partitions_13 --> chunk_partitions_14
    chunk_partitions_14 --> chunk_partitions_15
    chunk_partitions_15 --> chunk_partitions_16
```

## Function: `command`

- File: MMSB/src/04_propagation/propagation_queue.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    command_0["ENTRY"]
    command_1["let page = Arc :: new (Page :: new (PageID (id) , 8 , PageLocation :: Cpu) . unwrap ())"]
    command_2["PropagationCommand { page_id : page . id , page , dependencies : Vec :: new (..."]
    command_3["EXIT"]
    command_0 --> command_1
    command_1 --> command_2
    command_2 --> command_3
```

## Function: `delta_error_to_page`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    delta_error_to_page_0["ENTRY"]
    delta_error_to_page_1["match err { DeltaError :: SizeMismatch { mask_len , payload_len } => { PageEr..."]
    delta_error_to_page_2["EXIT"]
    delta_error_to_page_0 --> delta_error_to_page_1
    delta_error_to_page_1 --> delta_error_to_page_2
```

## Function: `drain_batch_respects_bounds`

- File: MMSB/src/04_propagation/propagation_queue.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    drain_batch_respects_bounds_0["ENTRY"]
    drain_batch_respects_bounds_1["let queue = PropagationQueue :: with_capacity (8)"]
    drain_batch_respects_bounds_2["for i in 0 .. 6 { queue . push (command (i)) ; }"]
    drain_batch_respects_bounds_3["let drained = queue . drain_batch (4)"]
    drain_batch_respects_bounds_4["macro assert_eq"]
    drain_batch_respects_bounds_5["macro assert_eq"]
    drain_batch_respects_bounds_6["EXIT"]
    drain_batch_respects_bounds_0 --> drain_batch_respects_bounds_1
    drain_batch_respects_bounds_1 --> drain_batch_respects_bounds_2
    drain_batch_respects_bounds_2 --> drain_batch_respects_bounds_3
    drain_batch_respects_bounds_3 --> drain_batch_respects_bounds_4
    drain_batch_respects_bounds_4 --> drain_batch_respects_bounds_5
    drain_batch_respects_bounds_5 --> drain_batch_respects_bounds_6
```

## Function: `enqueue_sparse`

- File: MMSB/src/04_propagation/sparse_message_passing.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    enqueue_sparse_0["ENTRY"]
    enqueue_sparse_1["queue . push (command)"]
    enqueue_sparse_2["EXIT"]
    enqueue_sparse_0 --> enqueue_sparse_1
    enqueue_sparse_1 --> enqueue_sparse_2
```

## Function: `gc_invoked_when_threshold_low`

- File: MMSB/src/04_propagation/tick_orchestrator.rs
- Branches: 0
- Loops: 0
- Nodes: 6
- Edges: 5

```mermaid
flowchart TD
    gc_invoked_when_threshold_low_0["ENTRY"]
    gc_invoked_when_threshold_low_1["let (orchestrator , _) = orchestrator (1)"]
    gc_invoked_when_threshold_low_2["let deltas = vec ! [sample_delta (1 , 1 , 0xAA) ; 8]"]
    gc_invoked_when_threshold_low_3["let metrics = orchestrator . execute_tick (deltas) . unwrap ()"]
    gc_invoked_when_threshold_low_4["macro assert"]
    gc_invoked_when_threshold_low_5["EXIT"]
    gc_invoked_when_threshold_low_0 --> gc_invoked_when_threshold_low_1
    gc_invoked_when_threshold_low_1 --> gc_invoked_when_threshold_low_2
    gc_invoked_when_threshold_low_2 --> gc_invoked_when_threshold_low_3
    gc_invoked_when_threshold_low_3 --> gc_invoked_when_threshold_low_4
    gc_invoked_when_threshold_low_4 --> gc_invoked_when_threshold_low_5
```

## Function: `make_delta`

- File: MMSB/src/04_propagation/throughput_engine.rs
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

## Function: `merges_multiple_deltas_per_page`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    merges_multiple_deltas_per_page_0["ENTRY"]
    merges_multiple_deltas_per_page_1["let allocator = Arc :: new (PageAllocator :: new (Default :: default ()))"]
    merges_multiple_deltas_per_page_2["allocator . allocate_raw (PageID (1) , 4 , Some (PageLocation :: Cpu)) . unwr..."]
    merges_multiple_deltas_per_page_3["let engine = ThroughputEngine :: new (Arc :: clone (& allocator) , 1 , 4)"]
    merges_multiple_deltas_per_page_4["let deltas = vec ! [make_delta (1 , 1 , b'\x01\x02\x03\x04') , make_delta (2 , 1 , b'\x10\..."]
    merges_multiple_deltas_per_page_5["let metrics = engine . process_parallel (deltas) . unwrap ()"]
    merges_multiple_deltas_per_page_6["macro assert_eq"]
    merges_multiple_deltas_per_page_7["unsafe { let page1 = & mut * allocator . acquire_page (PageID (1)) . unwrap (..."]
    merges_multiple_deltas_per_page_8["EXIT"]
    merges_multiple_deltas_per_page_0 --> merges_multiple_deltas_per_page_1
    merges_multiple_deltas_per_page_1 --> merges_multiple_deltas_per_page_2
    merges_multiple_deltas_per_page_2 --> merges_multiple_deltas_per_page_3
    merges_multiple_deltas_per_page_3 --> merges_multiple_deltas_per_page_4
    merges_multiple_deltas_per_page_4 --> merges_multiple_deltas_per_page_5
    merges_multiple_deltas_per_page_5 --> merges_multiple_deltas_per_page_6
    merges_multiple_deltas_per_page_6 --> merges_multiple_deltas_per_page_7
    merges_multiple_deltas_per_page_7 --> merges_multiple_deltas_per_page_8
```

## Function: `orchestrator`

- File: MMSB/src/04_propagation/tick_orchestrator.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    orchestrator_0["ENTRY"]
    orchestrator_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    orchestrator_2["for id in 1 ..= 4 { allocator . allocate_raw (PageID (id) , 32 , Some (PageLo..."]
    orchestrator_3["let throughput = ThroughputEngine :: new (Arc :: clone (& allocator) , 2 , 64)"]
    orchestrator_4["let graph = Arc :: new (ShadowPageGraph :: default ())"]
    orchestrator_5["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    orchestrator_6["let memory : Arc < dyn MemoryPressureHandler > = Arc :: new (TestMemoryHandler :: new (32 , threshold != usize :: MAX))"]
    orchestrator_7["(TickOrchestrator :: new (throughput , graph , memory) , allocator ,)"]
    orchestrator_8["EXIT"]
    orchestrator_0 --> orchestrator_1
    orchestrator_1 --> orchestrator_2
    orchestrator_2 --> orchestrator_3
    orchestrator_3 --> orchestrator_4
    orchestrator_4 --> orchestrator_5
    orchestrator_5 --> orchestrator_6
    orchestrator_6 --> orchestrator_7
    orchestrator_7 --> orchestrator_8
```

## Function: `partition_by_page`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    partition_by_page_0["ENTRY"]
    partition_by_page_1["let mut map : HashMap < PageID , Vec < usize > > = HashMap :: new ()"]
    partition_by_page_2["for idx in 0 .. batch . len () { if let Some (page_id) = batch . page_id_at (..."]
    partition_by_page_3["map . into_iter () . collect ()"]
    partition_by_page_4["EXIT"]
    partition_by_page_0 --> partition_by_page_1
    partition_by_page_1 --> partition_by_page_2
    partition_by_page_2 --> partition_by_page_3
    partition_by_page_3 --> partition_by_page_4
```

## Function: `passthrough`

- File: MMSB/src/04_propagation/propagation_fastpath.rs
- Branches: 0
- Loops: 0
- Nodes: 2
- Edges: 1

```mermaid
flowchart TD
    passthrough_0["ENTRY"]
    passthrough_1["EXIT"]
    passthrough_0 --> passthrough_1
```

## Function: `process_chunk`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    process_chunk_0["ENTRY"]
    process_chunk_1["let mut processed = 0usize"]
    process_chunk_2["for (page_id , indexes) in chunk { let ptr = allocator . acquire_page (page_i..."]
    process_chunk_3["Ok (processed)"]
    process_chunk_4["EXIT"]
    process_chunk_0 --> process_chunk_1
    process_chunk_1 --> process_chunk_2
    process_chunk_2 --> process_chunk_3
    process_chunk_3 --> process_chunk_4
```

## Function: `queue_roundtrip`

- File: MMSB/src/04_propagation/propagation_queue.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    queue_roundtrip_0["ENTRY"]
    queue_roundtrip_1["let queue = PropagationQueue :: with_capacity (8)"]
    queue_roundtrip_2["for i in 0 .. 8 { queue . push (command (i)) ; }"]
    queue_roundtrip_3["macro assert_eq"]
    queue_roundtrip_4["for i in 0 .. 8 { let popped = queue . pop () . unwrap () ; assert_eq ! (popp..."]
    queue_roundtrip_5["macro assert"]
    queue_roundtrip_6["EXIT"]
    queue_roundtrip_0 --> queue_roundtrip_1
    queue_roundtrip_1 --> queue_roundtrip_2
    queue_roundtrip_2 --> queue_roundtrip_3
    queue_roundtrip_3 --> queue_roundtrip_4
    queue_roundtrip_4 --> queue_roundtrip_5
    queue_roundtrip_5 --> queue_roundtrip_6
```

## Function: `reports_nonzero_throughput_for_large_batches`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 0
- Nodes: 11
- Edges: 10

```mermaid
flowchart TD
    reports_nonzero_throughput_for_large_batches_0["ENTRY"]
    reports_nonzero_throughput_for_large_batches_1["let allocator = Arc :: new (PageAllocator :: new (Default :: default ()))"]
    reports_nonzero_throughput_for_large_batches_2["for id in 1 ..= 8 { allocator . allocate_raw (PageID (id) , 8 , Some (PageLoc..."]
    reports_nonzero_throughput_for_large_batches_3["let engine = ThroughputEngine :: new (Arc :: clone (& allocator) , 4 , 256)"]
    reports_nonzero_throughput_for_large_batches_4["let mut deltas = Vec :: new ()"]
    reports_nonzero_throughput_for_large_batches_5["for i in 0 .. 2000u64 { let page = 1 + (i % 8) ; deltas . push (make_delta (i..."]
    reports_nonzero_throughput_for_large_batches_6["let metrics = engine . process_parallel (deltas) . unwrap ()"]
    reports_nonzero_throughput_for_large_batches_7["macro assert_eq"]
    reports_nonzero_throughput_for_large_batches_8["macro assert"]
    reports_nonzero_throughput_for_large_batches_9["macro assert"]
    reports_nonzero_throughput_for_large_batches_10["EXIT"]
    reports_nonzero_throughput_for_large_batches_0 --> reports_nonzero_throughput_for_large_batches_1
    reports_nonzero_throughput_for_large_batches_1 --> reports_nonzero_throughput_for_large_batches_2
    reports_nonzero_throughput_for_large_batches_2 --> reports_nonzero_throughput_for_large_batches_3
    reports_nonzero_throughput_for_large_batches_3 --> reports_nonzero_throughput_for_large_batches_4
    reports_nonzero_throughput_for_large_batches_4 --> reports_nonzero_throughput_for_large_batches_5
    reports_nonzero_throughput_for_large_batches_5 --> reports_nonzero_throughput_for_large_batches_6
    reports_nonzero_throughput_for_large_batches_6 --> reports_nonzero_throughput_for_large_batches_7
    reports_nonzero_throughput_for_large_batches_7 --> reports_nonzero_throughput_for_large_batches_8
    reports_nonzero_throughput_for_large_batches_8 --> reports_nonzero_throughput_for_large_batches_9
    reports_nonzero_throughput_for_large_batches_9 --> reports_nonzero_throughput_for_large_batches_10
```

## Function: `sample_delta`

- File: MMSB/src/04_propagation/tick_orchestrator.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    sample_delta_0["ENTRY"]
    sample_delta_1["Delta { delta_id : DeltaID (id) , page_id : PageID (page) , epoch : Epoch (id..."]
    sample_delta_2["EXIT"]
    sample_delta_0 --> sample_delta_1
    sample_delta_1 --> sample_delta_2
```

## Function: `test_basic_push_pop`

- File: MMSB/src/04_propagation/ring_buffer.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_basic_push_pop_0["ENTRY"]
    test_basic_push_pop_1["let buffer = LockFreeRingBuffer :: new (4)"]
    test_basic_push_pop_2["macro assert"]
    test_basic_push_pop_3["buffer . try_push (1) . unwrap ()"]
    test_basic_push_pop_4["buffer . try_push (2) . unwrap ()"]
    test_basic_push_pop_5["macro assert_eq"]
    test_basic_push_pop_6["macro assert_eq"]
    test_basic_push_pop_7["macro assert"]
    test_basic_push_pop_8["EXIT"]
    test_basic_push_pop_0 --> test_basic_push_pop_1
    test_basic_push_pop_1 --> test_basic_push_pop_2
    test_basic_push_pop_2 --> test_basic_push_pop_3
    test_basic_push_pop_3 --> test_basic_push_pop_4
    test_basic_push_pop_4 --> test_basic_push_pop_5
    test_basic_push_pop_5 --> test_basic_push_pop_6
    test_basic_push_pop_6 --> test_basic_push_pop_7
    test_basic_push_pop_7 --> test_basic_push_pop_8
```

## Function: `test_concurrent_producers_consumers`

- File: MMSB/src/04_propagation/ring_buffer.rs
- Branches: 0
- Loops: 0
- Nodes: 12
- Edges: 11

```mermaid
flowchart TD
    test_concurrent_producers_consumers_0["ENTRY"]
    test_concurrent_producers_consumers_1["let buffer = Arc :: new (LockFreeRingBuffer :: new (128))"]
    test_concurrent_producers_consumers_2["let total = 10_000"]
    test_concurrent_producers_consumers_3["let produced = Arc :: new (AtomicUsize :: new (0))"]
    test_concurrent_producers_consumers_4["let consumed = Arc :: new (AtomicUsize :: new (0))"]
    test_concurrent_producers_consumers_5["let mut handles = Vec :: new ()"]
    test_concurrent_producers_consumers_6["for _ in 0 .. 4 { let buf = Arc :: clone (& buffer) ; let counter = Arc :: cl..."]
    test_concurrent_producers_consumers_7["for _ in 0 .. 2 { let buf = Arc :: clone (& buffer) ; let counter = Arc :: cl..."]
    test_concurrent_producers_consumers_8["for handle in handles { handle . join () . unwrap () ; }"]
    test_concurrent_producers_consumers_9["macro assert_eq"]
    test_concurrent_producers_consumers_10["macro assert"]
    test_concurrent_producers_consumers_11["EXIT"]
    test_concurrent_producers_consumers_0 --> test_concurrent_producers_consumers_1
    test_concurrent_producers_consumers_1 --> test_concurrent_producers_consumers_2
    test_concurrent_producers_consumers_2 --> test_concurrent_producers_consumers_3
    test_concurrent_producers_consumers_3 --> test_concurrent_producers_consumers_4
    test_concurrent_producers_consumers_4 --> test_concurrent_producers_consumers_5
    test_concurrent_producers_consumers_5 --> test_concurrent_producers_consumers_6
    test_concurrent_producers_consumers_6 --> test_concurrent_producers_consumers_7
    test_concurrent_producers_consumers_7 --> test_concurrent_producers_consumers_8
    test_concurrent_producers_consumers_8 --> test_concurrent_producers_consumers_9
    test_concurrent_producers_consumers_9 --> test_concurrent_producers_consumers_10
    test_concurrent_producers_consumers_10 --> test_concurrent_producers_consumers_11
```

## Function: `test_wraparound_behavior`

- File: MMSB/src/04_propagation/ring_buffer.rs
- Branches: 0
- Loops: 0
- Nodes: 9
- Edges: 8

```mermaid
flowchart TD
    test_wraparound_behavior_0["ENTRY"]
    test_wraparound_behavior_1["let buffer = LockFreeRingBuffer :: new (2)"]
    test_wraparound_behavior_2["buffer . try_push (1) . unwrap ()"]
    test_wraparound_behavior_3["buffer . try_push (2) . unwrap ()"]
    test_wraparound_behavior_4["macro assert"]
    test_wraparound_behavior_5["macro assert_eq"]
    test_wraparound_behavior_6["macro assert"]
    test_wraparound_behavior_7["macro assert_eq"]
    test_wraparound_behavior_8["EXIT"]
    test_wraparound_behavior_0 --> test_wraparound_behavior_1
    test_wraparound_behavior_1 --> test_wraparound_behavior_2
    test_wraparound_behavior_2 --> test_wraparound_behavior_3
    test_wraparound_behavior_3 --> test_wraparound_behavior_4
    test_wraparound_behavior_4 --> test_wraparound_behavior_5
    test_wraparound_behavior_5 --> test_wraparound_behavior_6
    test_wraparound_behavior_6 --> test_wraparound_behavior_7
    test_wraparound_behavior_7 --> test_wraparound_behavior_8
```

## Function: `tick_metrics_capture_all_phases`

- File: MMSB/src/04_propagation/tick_orchestrator.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    tick_metrics_capture_all_phases_0["ENTRY"]
    tick_metrics_capture_all_phases_1["let (orchestrator , _) = orchestrator (usize :: MAX)"]
    tick_metrics_capture_all_phases_2["let deltas = vec ! [sample_delta (1 , 1 , 0xAA) ; 64]"]
    tick_metrics_capture_all_phases_3["let metrics = orchestrator . execute_tick (deltas) . unwrap ()"]
    tick_metrics_capture_all_phases_4["macro assert"]
    tick_metrics_capture_all_phases_5["macro assert"]
    tick_metrics_capture_all_phases_6["macro assert_eq"]
    tick_metrics_capture_all_phases_7["EXIT"]
    tick_metrics_capture_all_phases_0 --> tick_metrics_capture_all_phases_1
    tick_metrics_capture_all_phases_1 --> tick_metrics_capture_all_phases_2
    tick_metrics_capture_all_phases_2 --> tick_metrics_capture_all_phases_3
    tick_metrics_capture_all_phases_3 --> tick_metrics_capture_all_phases_4
    tick_metrics_capture_all_phases_4 --> tick_metrics_capture_all_phases_5
    tick_metrics_capture_all_phases_5 --> tick_metrics_capture_all_phases_6
    tick_metrics_capture_all_phases_6 --> tick_metrics_capture_all_phases_7
```

