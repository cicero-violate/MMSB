# Functions A-F

## Layer: 01_page

### Rust Functions

#### `allocate_zeroed`

- **File:** MMSB/src/01_page/page.rs:0
- **Visibility:** Private
- **Calls:**
  - `map_err`
  - `std::alloc::Layout::array`
  - `PageError::AllocError`
  - `std::alloc::alloc_zeroed`
  - `is_null`
  - `Err`
  - `PageError::AllocError`
  - `Ok`

#### `apply_log`

- **File:** MMSB/src/01_page/tlog_replay.rs:0
- **Visibility:** Public
- **Calls:**
  - `find`
  - `iter_mut`
  - `apply_delta`

#### `bitpack_mask`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `enumerate`
  - `iter`

#### `bitunpack_mask`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `enumerate`
  - `iter_mut`
  - `len`

#### `checkpoint_validation_detects_divergence`

- **File:** MMSB/src/01_page/replay_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `temp_log_path`
  - `unwrap`
  - `TransactionLog::new`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `copy_from_slice`
  - `data_mut_slice`
  - `set_epoch`
  - `Epoch`
  - `copy_from_slice`
  - `data_mut_slice`
  - `ReplayValidator::new`
  - `unwrap`
  - `record_checkpoint`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `data_mut_slice`
  - `unwrap`
  - `validate_allocator`
  - `ok`
  - `fs::remove_file`

#### `compact`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Public
- **Calls:**
  - `len`
  - `to_vec`
  - `Vec::with_capacity`
  - `len`
  - `iter`
  - `next`
  - `push`
  - `clone`
  - `last_mut`
  - `merge`
  - `push`
  - `clone`

#### `compare_snapshots`

- **File:** MMSB/src/01_page/replay_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `HashMap::new`
  - `insert`
  - `Vec::new`
  - `remove`
  - `l2_distance`
  - `max`
  - `push`
  - `keys`
  - `push`
  - `sqrt`

#### `compress_delta_mask`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Public
- **Calls:**
  - `len`
  - `collect`
  - `map`
  - `iter`
  - `encode_rle`
  - `bitpack_mask`
  - `len`
  - `max`
  - `len`

#### `decode_rle`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`

#### `delta`

- **File:** MMSB/src/01_page/integrity_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `collect`
  - `map`
  - `iter`
  - `to_vec`
  - `Source`
  - `into`

#### `detects_orphan_and_epoch_errors`

- **File:** MMSB/src/01_page/integrity_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `DeviceBufferRegistry::default`
  - `insert`
  - `page`
  - `DeltaIntegrityChecker::new`
  - `Arc::clone`
  - `validate`

#### `encode_rle`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `Vec::new`
  - `is_empty`
  - `push`
  - `push`

## Layer: 02_semiring

### Rust Functions

#### `accumulate`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `add`
  - `mul`

#### `detects_impure_function`

- **File:** MMSB/src/02_semiring/purity_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PurityValidator::default`
  - `AtomicU32::new`
  - `validate_fn`
  - `fetch_add`

#### `fold_add`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `fold`
  - `into_iter`
  - `zero`
  - `add`

#### `fold_mul`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `fold`
  - `into_iter`
  - `one`
  - `mul`

## Layer: 03_dag

### Rust Functions

#### `detects_cycle`

- **File:** MMSB/src/03_dag/graph_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `ShadowPageGraph::default`
  - `add_edge`
  - `PageID`
  - `PageID`
  - `add_edge`
  - `PageID`
  - `PageID`
  - `add_edge`
  - `PageID`
  - `PageID`
  - `GraphValidator::new`
  - `detect_cycles`

#### `dfs`

- **File:** MMSB/src/03_dag/cycle_detection.rs:0
- **Visibility:** Private
- **Calls:**
  - `get`
  - `insert`
  - `get`
  - `dfs`
  - `insert`

## Layer: 04_propagation

### Rust Functions

#### `applies_batches_in_parallel`

- **File:** MMSB/src/04_propagation/throughput_engine.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `PageAllocator::new`
  - `Default::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `ThroughputEngine::new`
  - `Arc::clone`
  - `unwrap`
  - `process_parallel`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `unwrap`
  - `acquire_page`
  - `PageID`

#### `chunk_partitions`

- **File:** MMSB/src/04_propagation/throughput_engine.rs:0
- **Visibility:** Private
- **Calls:**
  - `is_empty`
  - `Vec::new`
  - `max`
  - `len`
  - `Vec::new`
  - `Vec::with_capacity`
  - `push`
  - `len`
  - `push`
  - `Vec::with_capacity`
  - `is_empty`
  - `push`

#### `command`

- **File:** MMSB/src/04_propagation/propagation_queue.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `unwrap`
  - `Page::new`
  - `PageID`
  - `Vec::new`

#### `delta_error_to_page`

- **File:** MMSB/src/04_propagation/throughput_engine.rs:0
- **Visibility:** Private

#### `drain_batch_respects_bounds`

- **File:** MMSB/src/04_propagation/propagation_queue.rs:0
- **Visibility:** Private
- **Calls:**
  - `PropagationQueue::with_capacity`
  - `push`
  - `command`
  - `drain_batch`

#### `enqueue_sparse`

- **File:** MMSB/src/04_propagation/sparse_message_passing.rs:0
- **Visibility:** Public
- **Calls:**
  - `push`

## Layer: 06_utility

### Rust Functions

#### `allocator`

- **File:** MMSB/src/06_utility/memory_monitor.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`

#### `cache_does_not_grow_unbounded`

- **File:** MMSB/src/06_utility/provenance_tracker.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `ShadowPageGraph::default`
  - `add_edge`
  - `PageID`
  - `PageID`
  - `ProvenanceTracker::with_capacity`
  - `Arc::clone`
  - `resolve`
  - `PageID`

#### `cpu_has_avx2`

- **File:** MMSB/src/06_utility/cpu_features.rs:0
- **Visibility:** Public
- **Calls:**
  - `CpuFeatures::get`

#### `cpu_has_avx512`

- **File:** MMSB/src/06_utility/cpu_features.rs:0
- **Visibility:** Public
- **Calls:**
  - `CpuFeatures::get`

#### `cpu_has_sse42`

- **File:** MMSB/src/06_utility/cpu_features.rs:0
- **Visibility:** Public
- **Calls:**
  - `CpuFeatures::get`

#### `epoch_invariant_detects_regressions`

- **File:** MMSB/src/06_utility/invariant_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `set_epoch`
  - `Epoch`
  - `Some`
  - `InvariantChecker::new`
  - `register`
  - `EpochMonotonicity::default`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `set_epoch`
  - `Epoch`

## Layer: root

### Rust Functions

#### `apply_random_deltas`

- **File:** MMSB/tests/stress_stability.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageID`
  - `next_u64`
  - `max`
  - `next_in_range`
  - `next_in_range`
  - `collect`
  - `map`
  - `iter`
  - `next_u32`
  - `DeltaID`
  - `Epoch`
  - `Source`
  - `String::new`
  - `wrapping_add`
  - `expect`
  - `acquire_page`
  - `expect`
  - `apply_delta`

#### `assert_throughput`

- **File:** MMSB/tests/stress_throughput.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_secs_f64`
  - `abs`

#### `build_deltas`

- **File:** MMSB/src/bin/phase6_bench.rs:0
- **Visibility:** Private
- **Calls:**
  - `collect`
  - `map`
  - `PageID`
  - `DeltaID`
  - `Epoch`
  - `Source`
  - `into`

#### `build_graph`

- **File:** MMSB/src/bin/phase6_bench.rs:0
- **Visibility:** Private
- **Calls:**
  - `ShadowPageGraph::default`
  - `add_edge`
  - `PageID`
  - `PageID`

#### `build_noop_delta`

- **File:** MMSB/tests/stress_throughput.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `Epoch`
  - `Vec::new`
  - `Vec::new`
  - `Source`
  - `String::new`

#### `convert_location`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageLocation::from_tag`

#### `delta`

- **File:** MMSB/tests/benchmark_06_tick_latency.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`

#### `dense_delta`

- **File:** MMSB/tests/delta_validation.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`

#### `diagnostics_enabled`

- **File:** MMSB/src/logging.rs:0
- **Visibility:** Private
- **Calls:**
  - `OnceLock::new`
  - `get_or_init`
  - `std::env::var`

#### `divergence`

- **File:** MMSB/tests/benchmark_09_stability.rs:0
- **Visibility:** Private
- **Calls:**
  - `sqrt`
  - `sum`
  - `map`
  - `zip`
  - `iter`
  - `powi`

#### `example_checkpoint`

- **File:** MMSB/tests/examples_basic.rs:0
- **Visibility:** Private

#### `example_delta_operations`

- **File:** MMSB/tests/examples_basic.rs:0
- **Visibility:** Private
- **Calls:**
  - `Delta::new_dense`
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`

#### `example_page_allocation`

- **File:** MMSB/tests/examples_basic.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocatorConfig::default`
  - `PageAllocator::new`
  - `PageID`
  - `allocate_raw`
  - `Some`
  - `free`

#### `fragmentation_probe_remains_stable`

- **File:** MMSB/tests/stress_memory.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `MemoryMonitor::with_config`
  - `Arc::clone`
  - `Vec::new`
  - `expect`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `push`
  - `PageID`
  - `len`
  - `pop`
  - `free`
  - `snapshot`
  - `snapshot`

