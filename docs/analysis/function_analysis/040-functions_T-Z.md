# Functions T-Z

## Layer: 01_page

### Rust Functions

#### `temp_log_path`

- **File:** MMSB/src/01_page/replay_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `std::env::temp_dir`
  - `push`

#### `test_apply_to_pages`

- **File:** MMSB/src/01_page/columnar_delta.rs:0
- **Visibility:** Private
- **Calls:**
  - `ColumnarDeltaBatch::from_rows`
  - `HashMap::new`
  - `unwrap`
  - `Page::new`
  - `PageID`
  - `unwrap`
  - `Page::new`
  - `PageID`
  - `insert`
  - `PageID`
  - `insert`
  - `PageID`
  - `unwrap`
  - `apply_to_pages`
  - `unwrap`
  - `remove`
  - `PageID`
  - `unwrap`
  - `remove`
  - `PageID`

#### `test_checkpoint_roundtrip_in_memory`

- **File:** MMSB/src/01_page/allocator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `unwrap`
  - `apply_delta`
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`
  - `snapshot_pages`
  - `expect`
  - `restore_from_snapshot`
  - `unwrap`
  - `acquire_page`
  - `PageID`

#### `test_epoch_filter`

- **File:** MMSB/src/01_page/columnar_delta.rs:0
- **Visibility:** Private
- **Calls:**
  - `ColumnarDeltaBatch::from_rows`
  - `filter_epoch_eq`
  - `Epoch`

#### `test_page_info_metadata_roundtrip`

- **File:** MMSB/src/01_page/allocator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `expect`
  - `allocate_raw`
  - `PageID`
  - `set_metadata`
  - `page_infos`

#### `test_roundtrip`

- **File:** MMSB/src/01_page/columnar_delta.rs:0
- **Visibility:** Private
- **Calls:**
  - `ColumnarDeltaBatch::from_rows`
  - `clone`
  - `to_vec`

#### `test_unified_page`

- **File:** MMSB/src/01_page/allocator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `expect`
  - `allocate_raw`
  - `PageID`
  - `data_mut_slice`

#### `validate_delta`

- **File:** MMSB/src/01_page/delta_validation.rs:0
- **Visibility:** Public
- **Calls:**
  - `count`
  - `filter`
  - `iter`
  - `len`
  - `Err`
  - `len`
  - `len`
  - `len`
  - `Err`
  - `len`
  - `len`
  - `Ok`

#### `validate_header`

- **File:** MMSB/src/01_page/tlog.rs:0
- **Visibility:** Private
- **Calls:**
  - `seek`
  - `SeekFrom::Start`
  - `read_exact`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `Err`
  - `std::io::Error::new`
  - `Ok`

#### `write_checkpoint`

- **File:** MMSB/src/01_page/checkpoint.rs:0
- **Visibility:** Public
- **Calls:**
  - `snapshot_pages`
  - `current_offset`
  - `BufWriter::new`
  - `File::create`
  - `write_all`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `write_all`
  - `to_le_bytes`
  - `len`
  - `write_all`
  - `flush`
  - `Ok`

## Layer: 02_semiring

### Rust Functions

#### `validates_semiring_operations`

- **File:** MMSB/src/02_semiring/purity_validator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PurityValidator::default`

## Layer: 03_dag

### Rust Functions

#### `topological_sort`

- **File:** MMSB/src/03_dag/shadow_graph_traversal.rs:0
- **Visibility:** Public
- **Calls:**
  - `Vec::new`
  - `HashMap::new`
  - `HashMap::new`
  - `iter`
  - `read`
  - `insert`
  - `clone`
  - `or_insert`
  - `entry`
  - `iter`
  - `or_insert`
  - `entry`
  - `collect`
  - `filter_map`
  - `iter`
  - `Some`
  - `pop_front`
  - `push`
  - `get`
  - `get_mut`
  - `push_back`

## Layer: 04_propagation

### Rust Functions

#### `test_basic_push_pop`

- **File:** MMSB/src/04_propagation/ring_buffer.rs:0
- **Visibility:** Private
- **Calls:**
  - `LockFreeRingBuffer::new`
  - `unwrap`
  - `try_push`
  - `unwrap`
  - `try_push`

#### `test_concurrent_producers_consumers`

- **File:** MMSB/src/04_propagation/ring_buffer.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `LockFreeRingBuffer::new`
  - `Arc::new`
  - `AtomicUsize::new`
  - `Arc::new`
  - `AtomicUsize::new`
  - `Vec::new`
  - `Arc::clone`
  - `Arc::clone`
  - `push`
  - `thread::spawn`
  - `fetch_add`
  - `is_err`
  - `try_push`
  - `thread::yield_now`
  - `Arc::clone`
  - `Arc::clone`
  - `push`
  - `thread::spawn`
  - `load`
  - `is_some`
  - `try_pop`
  - `fetch_add`
  - `thread::sleep`
  - `Duration::from_micros`
  - `unwrap`
  - `join`

#### `test_wraparound_behavior`

- **File:** MMSB/src/04_propagation/ring_buffer.rs:0
- **Visibility:** Private
- **Calls:**
  - `LockFreeRingBuffer::new`
  - `unwrap`
  - `try_push`
  - `unwrap`
  - `try_push`

#### `tick_metrics_capture_all_phases`

- **File:** MMSB/src/04_propagation/tick_orchestrator.rs:0
- **Visibility:** Private
- **Calls:**
  - `orchestrator`
  - `unwrap`
  - `execute_tick`

## Layer: 05_adaptive

### Rust Functions

#### `test_locality_cost_empty`

- **File:** MMSB/src/05_adaptive/memory_layout.rs:0
- **Visibility:** Private
- **Calls:**
  - `MemoryLayout::new`
  - `HashMap::new`

#### `test_locality_optimizer`

- **File:** MMSB/src/05_adaptive/locality_optimizer.rs:0
- **Visibility:** Private
- **Calls:**
  - `LocalityOptimizer::new`
  - `add_edge`
  - `add_edge`
  - `compute_ordering`
  - `assign_addresses`

#### `test_memory_layout_creation`

- **File:** MMSB/src/05_adaptive/memory_layout.rs:0
- **Visibility:** Private
- **Calls:**
  - `MemoryLayout::new`

#### `test_optimize_layout`

- **File:** MMSB/src/05_adaptive/memory_layout.rs:0
- **Visibility:** Private
- **Calls:**
  - `MemoryLayout::new`
  - `insert`
  - `insert`
  - `insert`
  - `HashMap::new`
  - `insert`
  - `insert`
  - `optimize_layout`

#### `test_page_clustering`

- **File:** MMSB/src/05_adaptive/page_clustering.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageClusterer::new`
  - `HashMap::new`
  - `insert`
  - `insert`
  - `insert`
  - `cluster_pages`

## Layer: 06_utility

### Rust Functions

#### `test_cache_hit_rate`

- **File:** MMSB/src/06_utility/telemetry.rs:0
- **Visibility:** Private
- **Calls:**
  - `Telemetry::new`
  - `record_cache_hit`
  - `record_cache_hit`
  - `record_cache_hit`
  - `record_cache_miss`
  - `snapshot`

#### `test_reset`

- **File:** MMSB/src/06_utility/telemetry.rs:0
- **Visibility:** Private
- **Calls:**
  - `Telemetry::new`
  - `record_cache_hit`
  - `reset`
  - `snapshot`

#### `test_telemetry_basic`

- **File:** MMSB/src/06_utility/telemetry.rs:0
- **Visibility:** Private
- **Calls:**
  - `Telemetry::new`
  - `record_cache_hit`
  - `record_cache_miss`
  - `record_allocation`
  - `snapshot`

#### `validate_metadata_blob`

- **File:** MMSB/src/06_utility/invariant_checker.rs:0
- **Visibility:** Private
- **Calls:**
  - `is_empty`
  - `Ok`
  - `read_u32`
  - `read_u32`
  - `read_bytes`
  - `read_u32`
  - `read_bytes`
  - `Ok`

## Layer: root

### Rust Functions

#### `temp_log_path`

- **File:** MMSB/tests/benchmark_01_replay.rs:0
- **Visibility:** Private
- **Calls:**
  - `std::env::temp_dir`
  - `push`

#### `test_allocator_cpu_gpu_latency`

- **File:** MMSB/tests/week27_31_integration.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocatorConfig::default`
  - `Arc::new`
  - `PageAllocator::new`
  - `PageID`
  - `allocate_raw`
  - `Some`
  - `free`

#### `test_api_public_interface`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private

#### `test_checkpoint_log_and_restore`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `TransactionLog::new`
  - `to_string`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `copy_from_slice`
  - `len`
  - `copy_from_slice`
  - `data_mut_slice`
  - `unwrap`
  - `write_checkpoint`
  - `to_string`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `TransactionLog::new`
  - `to_string`
  - `unwrap`
  - `load_checkpoint`
  - `expect`
  - `acquire_page`
  - `std::fs::remove_file`
  - `std::fs::remove_file`
  - `std::fs::remove_file`

#### `test_cpu_features`

- **File:** MMSB/tests/week27_31_integration.rs:0
- **Visibility:** Private
- **Calls:**
  - `CpuFeatures::detect`

#### `test_delta_merge_simd`

- **File:** MMSB/tests/week27_31_integration.rs:0
- **Visibility:** Private
- **Calls:**
  - `collect`
  - `unwrap`
  - `Delta::new_dense`
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`
  - `collect`
  - `unwrap`
  - `Delta::new_dense`
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`
  - `merge_deltas`

#### `test_dense_delta_application`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `unwrap`
  - `Delta::new_dense`
  - `DeltaID`
  - `Epoch`
  - `Source`
  - `into`
  - `unwrap`
  - `apply_delta`

#### `test_gpu_delta_kernels`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private

#### `test_invalid_page_deletion_is_safe`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `release`
  - `PageID`

#### `test_lockfree_allocator`

- **File:** MMSB/tests/week27_31_integration.rs:0
- **Visibility:** Private
- **Calls:**
  - `LockFreeAllocator::new`
  - `get_stats`

#### `test_page_info_metadata_roundtrip`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `set_metadata`
  - `page_infos`

#### `test_page_snapshot_and_restore`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `fill`
  - `data_mut_slice`
  - `snapshot_pages`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `restore_from_snapshot`
  - `unwrap`
  - `acquire_page`

#### `test_propagation_queue`

- **File:** MMSB/tests/week27_31_integration.rs:0
- **Visibility:** Private
- **Calls:**
  - `PropagationQueue::new`

#### `test_semiring_operations_tropical`

- **File:** MMSB/tests/week27_31_integration.rs:0
- **Visibility:** Private

#### `test_sparse_delta_application`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `unwrap`
  - `Delta::new_sparse`
  - `DeltaID`
  - `Epoch`
  - `Source`
  - `into`
  - `unwrap`
  - `apply_delta`

#### `test_thread_safe_allocator`

- **File:** MMSB/tests/mmsb_tests.rs:0
- **Visibility:** Private
- **Calls:**
  - `Arc::new`
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `collect`
  - `map`
  - `Arc::clone`
  - `std::thread::spawn`
  - `unwrap`
  - `allocate_raw`
  - `PageID`
  - `data_mut_slice`
  - `unwrap`
  - `join`

#### `throughput_engine_exceeds_minimum_rate`

- **File:** MMSB/tests/benchmark_05_throughput.rs:0
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
  - `collect`
  - `map`
  - `make_delta`
  - `unwrap`
  - `process_parallel`

#### `tick_latency_stays_within_budget`

- **File:** MMSB/tests/benchmark_06_tick_latency.rs:0
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
  - `add_edge`
  - `PageID`
  - `PageID`
  - `Arc::new`
  - `MemoryMonitor::with_config`
  - `Arc::clone`
  - `MemoryMonitorConfig::default`
  - `TickOrchestrator::new`
  - `collect`
  - `map`
  - `delta`
  - `unwrap`
  - `execute_tick`

#### `validates_dense_lengths`

- **File:** MMSB/tests/delta_validation.rs:0
- **Visibility:** Private
- **Calls:**
  - `dense_delta`

#### `vec_from_ptr`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `is_null`
  - `Vec::new`
  - `to_vec`
  - `std::slice::from_raw_parts`

#### `write_report`

- **File:** MMSB/src/bin/phase6_bench.rs:0
- **Visibility:** Private
- **Calls:**
  - `as_secs_f64`
  - `duration_since`
  - `SystemTime::now`
  - `File::create`
  - `Ok`

