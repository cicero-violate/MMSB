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
- Branches: 3
- Loops: 1
- Nodes: 19
- Edges: 22

```mermaid
flowchart TD
    chunk_partitions_0["ENTRY"]
    chunk_partitions_1["if partitions . is_empty ()"]
    chunk_partitions_2["return Vec :: new ()"]
    chunk_partitions_3["if join"]
    chunk_partitions_4["let chunk_size = ((partitions . len () + workers - 1) / workers) . max (1)"]
    chunk_partitions_5["let mut chunks = Vec :: new ()"]
    chunk_partitions_6["let mut current = Vec :: with_capacity (chunk_size)"]
    chunk_partitions_7["for entry in partitions"]
    chunk_partitions_8["current . push (entry)"]
    chunk_partitions_9["if current . len () == chunk_size"]
    chunk_partitions_10["chunks . push (current)"]
    chunk_partitions_11["current = Vec :: with_capacity (chunk_size)"]
    chunk_partitions_12["if join"]
    chunk_partitions_13["after for"]
    chunk_partitions_14["if ! current . is_empty ()"]
    chunk_partitions_15["chunks . push (current)"]
    chunk_partitions_16["if join"]
    chunk_partitions_17["chunks"]
    chunk_partitions_18["EXIT"]
    chunk_partitions_0 --> chunk_partitions_1
    chunk_partitions_1 --> chunk_partitions_2
    chunk_partitions_2 --> chunk_partitions_3
    chunk_partitions_1 --> chunk_partitions_3
    chunk_partitions_3 --> chunk_partitions_4
    chunk_partitions_4 --> chunk_partitions_5
    chunk_partitions_5 --> chunk_partitions_6
    chunk_partitions_6 --> chunk_partitions_7
    chunk_partitions_7 --> chunk_partitions_8
    chunk_partitions_8 --> chunk_partitions_9
    chunk_partitions_9 --> chunk_partitions_10
    chunk_partitions_10 --> chunk_partitions_11
    chunk_partitions_11 --> chunk_partitions_12
    chunk_partitions_9 --> chunk_partitions_12
    chunk_partitions_12 --> chunk_partitions_7
    chunk_partitions_7 --> chunk_partitions_13
    chunk_partitions_13 --> chunk_partitions_14
    chunk_partitions_14 --> chunk_partitions_15
    chunk_partitions_15 --> chunk_partitions_16
    chunk_partitions_14 --> chunk_partitions_16
    chunk_partitions_16 --> chunk_partitions_17
    chunk_partitions_17 --> chunk_partitions_18
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
- Branches: 3
- Loops: 0
- Nodes: 10
- Edges: 11

```mermaid
flowchart TD
    delta_error_to_page_0["ENTRY"]
    delta_error_to_page_1["match err"]
    delta_error_to_page_2["arm DeltaError :: SizeMismatch { mask_len , payload_len }"]
    delta_error_to_page_3["PageError :: MaskSizeMismatch { expected : mask_len , found : payload_len , }"]
    delta_error_to_page_4["arm DeltaError :: PageIDMismatch { expected , found }"]
    delta_error_to_page_5["PageError :: PageIDMismatch { expected , found , }"]
    delta_error_to_page_6["arm DeltaError :: MaskSizeMismatch { expected , found }"]
    delta_error_to_page_7["PageError :: MaskSizeMismatch { expected , found , }"]
    delta_error_to_page_8["match join"]
    delta_error_to_page_9["EXIT"]
    delta_error_to_page_0 --> delta_error_to_page_1
    delta_error_to_page_1 --> delta_error_to_page_2
    delta_error_to_page_2 --> delta_error_to_page_3
    delta_error_to_page_1 --> delta_error_to_page_4
    delta_error_to_page_4 --> delta_error_to_page_5
    delta_error_to_page_1 --> delta_error_to_page_6
    delta_error_to_page_6 --> delta_error_to_page_7
    delta_error_to_page_3 --> delta_error_to_page_8
    delta_error_to_page_5 --> delta_error_to_page_8
    delta_error_to_page_7 --> delta_error_to_page_8
    delta_error_to_page_8 --> delta_error_to_page_9
```

## Function: `drain_batch_respects_bounds`

- File: MMSB/src/04_propagation/propagation_queue.rs
- Branches: 0
- Loops: 1
- Nodes: 9
- Edges: 9

```mermaid
flowchart TD
    drain_batch_respects_bounds_0["ENTRY"]
    drain_batch_respects_bounds_1["let queue = PropagationQueue :: with_capacity (8)"]
    drain_batch_respects_bounds_2["for i in 0 .. 6"]
    drain_batch_respects_bounds_3["queue . push (command (i))"]
    drain_batch_respects_bounds_4["after for"]
    drain_batch_respects_bounds_5["let drained = queue . drain_batch (4)"]
    drain_batch_respects_bounds_6["macro assert_eq"]
    drain_batch_respects_bounds_7["macro assert_eq"]
    drain_batch_respects_bounds_8["EXIT"]
    drain_batch_respects_bounds_0 --> drain_batch_respects_bounds_1
    drain_batch_respects_bounds_1 --> drain_batch_respects_bounds_2
    drain_batch_respects_bounds_2 --> drain_batch_respects_bounds_3
    drain_batch_respects_bounds_3 --> drain_batch_respects_bounds_2
    drain_batch_respects_bounds_2 --> drain_batch_respects_bounds_4
    drain_batch_respects_bounds_4 --> drain_batch_respects_bounds_5
    drain_batch_respects_bounds_5 --> drain_batch_respects_bounds_6
    drain_batch_respects_bounds_6 --> drain_batch_respects_bounds_7
    drain_batch_respects_bounds_7 --> drain_batch_respects_bounds_8
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
- Loops: 1
- Nodes: 11
- Edges: 11

```mermaid
flowchart TD
    orchestrator_0["ENTRY"]
    orchestrator_1["let allocator = Arc :: new (PageAllocator :: new (PageAllocatorConfig :: default ()))"]
    orchestrator_2["for id in 1 ..= 4"]
    orchestrator_3["allocator . allocate_raw (PageID (id) , 32 , Some (PageLocation :: Cpu)) . un..."]
    orchestrator_4["after for"]
    orchestrator_5["let throughput = ThroughputEngine :: new (Arc :: clone (& allocator) , 2 , 64)"]
    orchestrator_6["let graph = Arc :: new (ShadowPageGraph :: default ())"]
    orchestrator_7["graph . add_edge (PageID (1) , PageID (2) , EdgeType :: Data)"]
    orchestrator_8["let memory : Arc < dyn MemoryPressureHandler > = Arc :: new (TestMemoryHandler :: new (32 , threshold != usize :: MAX))"]
    orchestrator_9["(TickOrchestrator :: new (throughput , graph , memory) , allocator ,)"]
    orchestrator_10["EXIT"]
    orchestrator_0 --> orchestrator_1
    orchestrator_1 --> orchestrator_2
    orchestrator_2 --> orchestrator_3
    orchestrator_3 --> orchestrator_2
    orchestrator_2 --> orchestrator_4
    orchestrator_4 --> orchestrator_5
    orchestrator_5 --> orchestrator_6
    orchestrator_6 --> orchestrator_7
    orchestrator_7 --> orchestrator_8
    orchestrator_8 --> orchestrator_9
    orchestrator_9 --> orchestrator_10
```

## Function: `partition_by_page`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 1
- Loops: 1
- Nodes: 9
- Edges: 10

```mermaid
flowchart TD
    partition_by_page_0["ENTRY"]
    partition_by_page_1["let mut map : HashMap < PageID , Vec < usize > > = HashMap :: new ()"]
    partition_by_page_2["for idx in 0 .. batch . len ()"]
    partition_by_page_3["if let Some (page_id) = batch . page_id_at (idx)"]
    partition_by_page_4["map . entry (page_id) . or_default () . push (idx)"]
    partition_by_page_5["if join"]
    partition_by_page_6["after for"]
    partition_by_page_7["map . into_iter () . collect ()"]
    partition_by_page_8["EXIT"]
    partition_by_page_0 --> partition_by_page_1
    partition_by_page_1 --> partition_by_page_2
    partition_by_page_2 --> partition_by_page_3
    partition_by_page_3 --> partition_by_page_4
    partition_by_page_4 --> partition_by_page_5
    partition_by_page_3 --> partition_by_page_5
    partition_by_page_5 --> partition_by_page_2
    partition_by_page_2 --> partition_by_page_6
    partition_by_page_6 --> partition_by_page_7
    partition_by_page_7 --> partition_by_page_8
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
- Branches: 3
- Loops: 2
- Nodes: 20
- Edges: 24

```mermaid
flowchart TD
    process_chunk_0["ENTRY"]
    process_chunk_1["let mut processed = 0usize"]
    process_chunk_2["for (page_id , indexes) in chunk"]
    process_chunk_3["let ptr = allocator . acquire_page (page_id) . ok_or (PageError :: PageNotFound (page_i..."]
    process_chunk_4["if indexes . is_empty ()"]
    process_chunk_5["continue"]
    process_chunk_6["if join"]
    process_chunk_7["let mut merged : Option < Delta > = None"]
    process_chunk_8["for idx in indexes"]
    process_chunk_9["if let Some (delta) = batch . delta_at (idx)"]
    process_chunk_10["processed += 1"]
    process_chunk_11["merged = Some (match merged { Some (ref current) => merge_deltas (current , &..."]
    process_chunk_12["if join"]
    process_chunk_13["after for"]
    process_chunk_14["if let Some (final_delta) = merged"]
    process_chunk_15["unsafe { (* ptr) . apply_delta (& final_delta) ? ; }"]
    process_chunk_16["if join"]
    process_chunk_17["after for"]
    process_chunk_18["Ok (processed)"]
    process_chunk_19["EXIT"]
    process_chunk_0 --> process_chunk_1
    process_chunk_1 --> process_chunk_2
    process_chunk_2 --> process_chunk_3
    process_chunk_3 --> process_chunk_4
    process_chunk_4 --> process_chunk_5
    process_chunk_5 --> process_chunk_6
    process_chunk_4 --> process_chunk_6
    process_chunk_6 --> process_chunk_7
    process_chunk_7 --> process_chunk_8
    process_chunk_8 --> process_chunk_9
    process_chunk_9 --> process_chunk_10
    process_chunk_10 --> process_chunk_11
    process_chunk_11 --> process_chunk_12
    process_chunk_9 --> process_chunk_12
    process_chunk_12 --> process_chunk_8
    process_chunk_8 --> process_chunk_13
    process_chunk_13 --> process_chunk_14
    process_chunk_14 --> process_chunk_15
    process_chunk_15 --> process_chunk_16
    process_chunk_14 --> process_chunk_16
    process_chunk_16 --> process_chunk_2
    process_chunk_2 --> process_chunk_17
    process_chunk_17 --> process_chunk_18
    process_chunk_18 --> process_chunk_19
```

## Function: `queue_roundtrip`

- File: MMSB/src/04_propagation/propagation_queue.rs
- Branches: 0
- Loops: 2
- Nodes: 12
- Edges: 13

```mermaid
flowchart TD
    queue_roundtrip_0["ENTRY"]
    queue_roundtrip_1["let queue = PropagationQueue :: with_capacity (8)"]
    queue_roundtrip_2["for i in 0 .. 8"]
    queue_roundtrip_3["queue . push (command (i))"]
    queue_roundtrip_4["after for"]
    queue_roundtrip_5["macro assert_eq"]
    queue_roundtrip_6["for i in 0 .. 8"]
    queue_roundtrip_7["let popped = queue . pop () . unwrap ()"]
    queue_roundtrip_8["macro assert_eq"]
    queue_roundtrip_9["after for"]
    queue_roundtrip_10["macro assert"]
    queue_roundtrip_11["EXIT"]
    queue_roundtrip_0 --> queue_roundtrip_1
    queue_roundtrip_1 --> queue_roundtrip_2
    queue_roundtrip_2 --> queue_roundtrip_3
    queue_roundtrip_3 --> queue_roundtrip_2
    queue_roundtrip_2 --> queue_roundtrip_4
    queue_roundtrip_4 --> queue_roundtrip_5
    queue_roundtrip_5 --> queue_roundtrip_6
    queue_roundtrip_6 --> queue_roundtrip_7
    queue_roundtrip_7 --> queue_roundtrip_8
    queue_roundtrip_8 --> queue_roundtrip_6
    queue_roundtrip_6 --> queue_roundtrip_9
    queue_roundtrip_9 --> queue_roundtrip_10
    queue_roundtrip_10 --> queue_roundtrip_11
```

## Function: `reports_nonzero_throughput_for_large_batches`

- File: MMSB/src/04_propagation/throughput_engine.rs
- Branches: 0
- Loops: 2
- Nodes: 16
- Edges: 17

```mermaid
flowchart TD
    reports_nonzero_throughput_for_large_batches_0["ENTRY"]
    reports_nonzero_throughput_for_large_batches_1["let allocator = Arc :: new (PageAllocator :: new (Default :: default ()))"]
    reports_nonzero_throughput_for_large_batches_2["for id in 1 ..= 8"]
    reports_nonzero_throughput_for_large_batches_3["allocator . allocate_raw (PageID (id) , 8 , Some (PageLocation :: Cpu)) . unw..."]
    reports_nonzero_throughput_for_large_batches_4["after for"]
    reports_nonzero_throughput_for_large_batches_5["let engine = ThroughputEngine :: new (Arc :: clone (& allocator) , 4 , 256)"]
    reports_nonzero_throughput_for_large_batches_6["let mut deltas = Vec :: new ()"]
    reports_nonzero_throughput_for_large_batches_7["for i in 0 .. 2000u64"]
    reports_nonzero_throughput_for_large_batches_8["let page = 1 + (i % 8)"]
    reports_nonzero_throughput_for_large_batches_9["deltas . push (make_delta (i + 1 , page , & [i as u8 ; 8]))"]
    reports_nonzero_throughput_for_large_batches_10["after for"]
    reports_nonzero_throughput_for_large_batches_11["let metrics = engine . process_parallel (deltas) . unwrap ()"]
    reports_nonzero_throughput_for_large_batches_12["macro assert_eq"]
    reports_nonzero_throughput_for_large_batches_13["macro assert"]
    reports_nonzero_throughput_for_large_batches_14["macro assert"]
    reports_nonzero_throughput_for_large_batches_15["EXIT"]
    reports_nonzero_throughput_for_large_batches_0 --> reports_nonzero_throughput_for_large_batches_1
    reports_nonzero_throughput_for_large_batches_1 --> reports_nonzero_throughput_for_large_batches_2
    reports_nonzero_throughput_for_large_batches_2 --> reports_nonzero_throughput_for_large_batches_3
    reports_nonzero_throughput_for_large_batches_3 --> reports_nonzero_throughput_for_large_batches_2
    reports_nonzero_throughput_for_large_batches_2 --> reports_nonzero_throughput_for_large_batches_4
    reports_nonzero_throughput_for_large_batches_4 --> reports_nonzero_throughput_for_large_batches_5
    reports_nonzero_throughput_for_large_batches_5 --> reports_nonzero_throughput_for_large_batches_6
    reports_nonzero_throughput_for_large_batches_6 --> reports_nonzero_throughput_for_large_batches_7
    reports_nonzero_throughput_for_large_batches_7 --> reports_nonzero_throughput_for_large_batches_8
    reports_nonzero_throughput_for_large_batches_8 --> reports_nonzero_throughput_for_large_batches_9
    reports_nonzero_throughput_for_large_batches_9 --> reports_nonzero_throughput_for_large_batches_7
    reports_nonzero_throughput_for_large_batches_7 --> reports_nonzero_throughput_for_large_batches_10
    reports_nonzero_throughput_for_large_batches_10 --> reports_nonzero_throughput_for_large_batches_11
    reports_nonzero_throughput_for_large_batches_11 --> reports_nonzero_throughput_for_large_batches_12
    reports_nonzero_throughput_for_large_batches_12 --> reports_nonzero_throughput_for_large_batches_13
    reports_nonzero_throughput_for_large_batches_13 --> reports_nonzero_throughput_for_large_batches_14
    reports_nonzero_throughput_for_large_batches_14 --> reports_nonzero_throughput_for_large_batches_15
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
- Loops: 3
- Nodes: 22
- Edges: 24

```mermaid
flowchart TD
    test_concurrent_producers_consumers_0["ENTRY"]
    test_concurrent_producers_consumers_1["let buffer = Arc :: new (LockFreeRingBuffer :: new (128))"]
    test_concurrent_producers_consumers_2["let total = 10_000"]
    test_concurrent_producers_consumers_3["let produced = Arc :: new (AtomicUsize :: new (0))"]
    test_concurrent_producers_consumers_4["let consumed = Arc :: new (AtomicUsize :: new (0))"]
    test_concurrent_producers_consumers_5["let mut handles = Vec :: new ()"]
    test_concurrent_producers_consumers_6["for _ in 0 .. 4"]
    test_concurrent_producers_consumers_7["let buf = Arc :: clone (& buffer)"]
    test_concurrent_producers_consumers_8["let counter = Arc :: clone (& produced)"]
    test_concurrent_producers_consumers_9["handles . push (thread :: spawn (move | | { loop { let next = counter . fetch..."]
    test_concurrent_producers_consumers_10["after for"]
    test_concurrent_producers_consumers_11["for _ in 0 .. 2"]
    test_concurrent_producers_consumers_12["let buf = Arc :: clone (& buffer)"]
    test_concurrent_producers_consumers_13["let counter = Arc :: clone (& consumed)"]
    test_concurrent_producers_consumers_14["handles . push (thread :: spawn (move | | { while counter . load (Ordering ::..."]
    test_concurrent_producers_consumers_15["after for"]
    test_concurrent_producers_consumers_16["for handle in handles"]
    test_concurrent_producers_consumers_17["handle . join () . unwrap ()"]
    test_concurrent_producers_consumers_18["after for"]
    test_concurrent_producers_consumers_19["macro assert_eq"]
    test_concurrent_producers_consumers_20["macro assert"]
    test_concurrent_producers_consumers_21["EXIT"]
    test_concurrent_producers_consumers_0 --> test_concurrent_producers_consumers_1
    test_concurrent_producers_consumers_1 --> test_concurrent_producers_consumers_2
    test_concurrent_producers_consumers_2 --> test_concurrent_producers_consumers_3
    test_concurrent_producers_consumers_3 --> test_concurrent_producers_consumers_4
    test_concurrent_producers_consumers_4 --> test_concurrent_producers_consumers_5
    test_concurrent_producers_consumers_5 --> test_concurrent_producers_consumers_6
    test_concurrent_producers_consumers_6 --> test_concurrent_producers_consumers_7
    test_concurrent_producers_consumers_7 --> test_concurrent_producers_consumers_8
    test_concurrent_producers_consumers_8 --> test_concurrent_producers_consumers_9
    test_concurrent_producers_consumers_9 --> test_concurrent_producers_consumers_6
    test_concurrent_producers_consumers_6 --> test_concurrent_producers_consumers_10
    test_concurrent_producers_consumers_10 --> test_concurrent_producers_consumers_11
    test_concurrent_producers_consumers_11 --> test_concurrent_producers_consumers_12
    test_concurrent_producers_consumers_12 --> test_concurrent_producers_consumers_13
    test_concurrent_producers_consumers_13 --> test_concurrent_producers_consumers_14
    test_concurrent_producers_consumers_14 --> test_concurrent_producers_consumers_11
    test_concurrent_producers_consumers_11 --> test_concurrent_producers_consumers_15
    test_concurrent_producers_consumers_15 --> test_concurrent_producers_consumers_16
    test_concurrent_producers_consumers_16 --> test_concurrent_producers_consumers_17
    test_concurrent_producers_consumers_17 --> test_concurrent_producers_consumers_16
    test_concurrent_producers_consumers_16 --> test_concurrent_producers_consumers_18
    test_concurrent_producers_consumers_18 --> test_concurrent_producers_consumers_19
    test_concurrent_producers_consumers_19 --> test_concurrent_producers_consumers_20
    test_concurrent_producers_consumers_20 --> test_concurrent_producers_consumers_21
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

