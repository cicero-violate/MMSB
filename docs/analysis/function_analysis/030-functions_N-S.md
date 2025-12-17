# Functions N-S

## Layer: 01_page

### Rust Functions

#### `now_ns`

- **File:** MMSB/src/01_page/delta.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_nanos`
  - `unwrap_or_default`
  - `duration_since`
  - `SystemTime::now`

#### `page`

- **File:** MMSB/src/01_page/integrity_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `unwrap`
  - `Page::new`
  - `PageID`

#### `rand_suffix`

- **File:** MMSB/src/01_page/replay_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_nanos`
  - `unwrap`
  - `duration_since`
  - `SystemTime::now`

#### `read_bytes`

- **File:** MMSB/src/01_page/page.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `Err`
  - `PageError::MetadataDecode`
  - `to_vec`
  - `Ok`

#### `read_frame`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Private
- **Calls:**
  - `read_exact`
  - `kind`
  - `Ok`
  - `Err`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `collect`
  - `map`
  - `iter`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Source`
  - `to_string`
  - `String::from_utf8_lossy`
  - `is_err`
  - `read_exact`
  - `Ok`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Some`
  - `to_string`
  - `String::from_utf8_lossy`
  - `Ok`
  - `Some`
  - `DeltaID`
  - `u64::from_le_bytes`
  - `PageID`
  - `u64::from_le_bytes`
  - `Epoch`
  - `u32::from_le_bytes`
  - `u64::from_le_bytes`

#### `read_log`

- **File:** MMSB/src/01_page/tlog_serialization.rs:0
- **Visibility:** Public
- **Calls:**
  - `File::open`
  - `BufReader::new`
  - `read_exact`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `Vec::new`
  - `is_err`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `collect`
  - `map`
  - `iter`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `read_exact`
  - `read_exact`
  - `u64::from_le_bytes`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Source`
  - `to_string`
  - `String::from_utf8_lossy`
  - `is_err`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Some`
  - `to_string`
  - `String::from_utf8_lossy`
  - `push`
  - `DeltaID`
  - `u64::from_le_bytes`
  - `PageID`
  - `u64::from_le_bytes`
  - `Epoch`
  - `u32::from_le_bytes`
  - `Ok`

#### `read_u32`

- **File:** MMSB/src/01_page/page.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `Err`
  - `PageError::MetadataDecode`
  - `map_err`
  - `try_into`
  - `PageError::MetadataDecode`
  - `Ok`
  - `u32::from_le_bytes`

#### `schema_valid`

- **File:** MMSB/src/01_page/integrity_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `count`
  - `filter`
  - `iter`
  - `len`
  - `len`
  - `len`

#### `serialize_frame`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Private
- **Calls:**
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `len`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `write_all`
  - `write_all`
  - `to_le_bytes`
  - `as_bytes`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `unwrap_or`
  - `map`
  - `as_ref`
  - `len`
  - `as_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `as_bytes`
  - `Ok`

#### `summary`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Public
- **Calls:**
  - `File::open`
  - `as_ref`
  - `kind`
  - `Err`
  - `Err`
  - `len`
  - `metadata`
  - `Ok`
  - `LogSummary::default`
  - `BufReader::new`
  - `validate_header`
  - `LogSummary::default`
  - `read_frame`
  - `unwrap_or`
  - `map`
  - `as_ref`
  - `len`
  - `as_bytes`
  - `len`
  - `len`
  - `max`
  - `Ok`

## Layer: 03_dag

### Rust Functions

#### `per_page_validation`

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
  - `GraphValidator::new`
  - `validate_page`
  - `PageID`

#### `reachable`

- **File:** MMSB/src/03_dag/graph_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `HashSet::new`
  - `pop`
  - `insert`
  - `get`
  - `push`
  - `collect`
  - `into_iter`

#### `strong_connect`

- **File:** MMSB/src/03_dag/graph_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `insert`
  - `insert`
  - `push`
  - `insert`
  - `get`
  - `contains_key`
  - `strong_connect`
  - `unwrap`
  - `get`
  - `unwrap`
  - `get_mut`
  - `min`
  - `contains`
  - `unwrap`
  - `get`
  - `unwrap`
  - `get_mut`
  - `min`
  - `unwrap`
  - `get`
  - `unwrap`
  - `get`
  - `Vec::new`
  - `unwrap`
  - `pop`
  - `remove`
  - `push`
  - `len`
  - `is_self_loop`

## Layer: 04_propagation

### Rust Functions

#### `orchestrator`

- **File:** MMSB/src/04_propagation/tick_orchestrator.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `ThroughputEngine::new`
  - `Arc::clone`
  - `Arc::new`
  - `ShadowPageGraph::default`
  - `add_edge`
  - `PageID`
  - `PageID`
  - `Arc::new`
  - `MemoryMonitor::with_config`
  - `Arc::clone`
  - `MemoryMonitorConfig::default`
  - `TickOrchestrator::new`

#### `partition_by_page`

- **File:** MMSB/src/04_propagation/throughput_engine.rs:0
- **Visibility:** Private
- **Calls:**
  - `HashMap::new`
  - `len`
  - `page_id_at`
  - `push`
  - `or_default`
  - `entry`
  - `collect`
  - `into_iter`

#### `passthrough`

- **File:** MMSB/src/04_propagation/propagation_fastpath.rs:0
- **Visibility:** Public

#### `process_chunk`

- **File:** MMSB/src/04_propagation/throughput_engine.rs:0
- **Visibility:** Private
- **Calls:**
  - `ok_or`
  - `acquire_page`
  - `PageError::PageNotFound`
  - `is_empty`
  - `delta_at`
  - `Some`
  - `map_err`
  - `merge_deltas`
  - `apply_delta`
  - `Ok`

#### `queue_roundtrip`

- **File:** MMSB/src/04_propagation/propagation_queue.rs:0
- **Visibility:** Private
- **Calls:**
  - `PropagationQueue::with_capacity`
  - `push`
  - `command`
  - `unwrap`
  - `pop`

#### `reports_nonzero_throughput_for_large_batches`

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
  - `ThroughputEngine::new`
  - `Arc::clone`
  - `Vec::new`
  - `push`
  - `make_delta`
  - `unwrap`
  - `process_parallel`

#### `sample_delta`

- **File:** MMSB/src/04_propagation/tick_orchestrator.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`

## Layer: 06_utility

### Rust Functions

#### `read_bytes`

- **File:** MMSB/src/06_utility/invariant_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `Err`
  - `PageError::MetadataDecode`
  - `Ok`

#### `read_u32`

- **File:** MMSB/src/06_utility/invariant_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `len`
  - `Err`
  - `PageError::MetadataDecode`
  - `u32::from_le_bytes`
  - `unwrap`
  - `try_into`
  - `Ok`

#### `resolves_chain_with_depth_limit`

- **File:** MMSB/src/06_utility/provenance_tracker.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
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
  - `ProvenanceTracker::with_capacity`
  - `Arc::clone`
  - `resolve`
  - `PageID`
  - `resolve`
  - `PageID`

#### `snapshot_identifies_cold_pages`

- **File:** MMSB/src/06_utility/memory_monitor.rs:0
- **Visibility:** Private
- **Calls:**
  - `allocator`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `MemoryMonitor::with_config`
  - `Arc::clone`
  - `snapshot`
  - `snapshot`
  - `unwrap`
  - `acquire_page`
  - `PageID`
  - `set_epoch`
  - `Epoch`
  - `snapshot`

#### `snapshot_reflects_allocator_state`

- **File:** MMSB/src/06_utility/memory_monitor.rs:0
- **Visibility:** Private
- **Calls:**
  - `allocator`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `Some`
  - `MemoryMonitor::new`
  - `Arc::clone`
  - `snapshot`

## Layer: root

### Rust Functions

#### `provenance_tracker_resolves_with_cache`

- **File:** MMSB/tests/benchmark_10_provenance.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
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
  - `ProvenanceTracker::with_capacity`
  - `Arc::clone`
  - `resolve`
  - `PageID`
  - `resolve`
  - `PageID`

#### `purity_validator_covers_semiring_operations`

- **File:** MMSB/tests/benchmark_04_purity.rs:0
- **Visibility:** Private
- **Calls:**
  - `PurityValidator::default`

#### `read_page`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `to_vec`
  - `data_slice`

#### `rejects_mismatched_dense_lengths`

- **File:** MMSB/tests/delta_validation.rs:0
- **Visibility:** Private
- **Calls:**
  - `dense_delta`

#### `replay_validator_divergence_under_threshold`

- **File:** MMSB/tests/benchmark_01_replay.rs:0
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
  - `ReplayValidator::new`
  - `unwrap`
  - `record_checkpoint`
  - `unwrap`
  - `validate_allocator`
  - `ok`
  - `std::fs::remove_file`

#### `set_last_error`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `with`
  - `borrow_mut`

#### `simulate`

- **File:** MMSB/tests/benchmark_09_stability.rs:0
- **Visibility:** Private
- **Calls:**
  - `collect`
  - `map`
  - `enumerate`
  - `iter`
  - `sin`
  - `tanh`

#### `slice_from_ptr`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Generics:** 'a, T
- **Calls:**
  - `is_null`
  - `slice::from_raw_parts`

#### `stability_resists_small_noise`

- **File:** MMSB/tests/benchmark_09_stability.rs:0
- **Visibility:** Private
- **Calls:**
  - `NoiseRng::new`
  - `clone`
  - `simulate`
  - `gaussian`
  - `simulate`
  - `divergence`
  - `max`

