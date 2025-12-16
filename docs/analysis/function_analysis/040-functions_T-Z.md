# Functions T-Z

## Layer: 00_physical

### Rust Functions

#### `test_checkpoint_roundtrip_in_memory`

- **File:** MMSB/src/00_physical/allocator.rs:0
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

#### `test_page_info_metadata_roundtrip`

- **File:** MMSB/src/00_physical/allocator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `PageAllocatorConfig::default`
  - `expect`
  - `allocate_raw`
  - `PageID`
  - `set_metadata`
  - `page_infos`

#### `test_unified_page`

- **File:** MMSB/src/00_physical/allocator.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageAllocator::new`
  - `expect`
  - `allocate_raw`
  - `PageID`
  - `data_mut_slice`

### Julia Functions

#### `wait_gpu_queue`

- **File:** MMSB/src/00_physical/DeviceSync.jl:72
- **Signature:** `wait_gpu_queue(buf::GPUCommandBuffer)`
- **Calls:**
  - `ccall`
  - `time_ns`

## Layer: 01_page

### Rust Functions

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

### Julia Functions

#### `_with_rust_errors`

- **File:** MMSB/src/01_page/TLog.jl:28
- **Signature:** `_with_rust_errors(f::Function, context::String)`
- **Calls:**
  - `f`
  - `rethrow`
  - `rethrow_translated`

#### `verify_state_consistency`

- **File:** MMSB/src/01_page/ReplayEngine.jl:115
- **Signature:** `verify_state_consistency(state::MMSBState)::Bool`
- **Calls:**
  - `get_page`
  - `lock`
  - `read_page`
  - `replay_to_epoch`
  - `typemax`

## Layer: 02_semiring

### Julia Functions

#### `tropical_accumulate`

- **File:** MMSB/src/02_semiring/Semiring.jl:65
- **Signature:** `tropical_accumulate(left::Real, right::Real)`
- **Calls:**
  - `FFIWrapper.rust_semiring_tropical_accumulate`
  - `Float64`

#### `tropical_fold_add`

- **File:** MMSB/src/02_semiring/Semiring.jl:49
- **Signature:** `tropical_fold_add(values::AbstractVector{<:Real})`
- **Calls:**
  - `FFIWrapper.rust_semiring_tropical_fold_add`
  - `_FLOAT_BUF`

#### `tropical_fold_mul`

- **File:** MMSB/src/02_semiring/Semiring.jl:57
- **Signature:** `tropical_fold_mul(values::AbstractVector{<:Real})`
- **Calls:**
  - `FFIWrapper.rust_semiring_tropical_fold_mul`
  - `_FLOAT_BUF`

#### `tropical_semiring`

- **File:** MMSB/src/02_semiring/Semiring.jl:16
- **Signature:** `tropical_semiring()`
- **Calls:**
  - `SemiringOps`
  - `min`

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

### Julia Functions

#### `topological_order`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:280
- **Signature:** `topological_order(graph::ShadowPageGraph)::Vector{PageID}`
- **Calls:**
  - `GraphCycleError`
  - `get`
  - `get_children`
  - `isempty`
  - `keys`
  - `length`
  - `lock`
  - `popfirst!`
  - `processed`
  - `push!`
  - `throw`

#### `topological_sort`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:170
- **Signature:** `topological_sort(graph::ShadowPageGraph)::Vector{PageID}`
- **Calls:**
  - `GraphCycleError`
  - `_all_vertices`
  - `get`
  - `isempty`
  - `length`
  - `lock`
  - `popfirst!`
  - `push!`
  - `throw`

#### `unsubscribe!`

- **File:** MMSB/src/03_dag/EventSystem.jl:118
- **Signature:** `unsubscribe!(sub::EventSubscription)`
- **Calls:**
  - `filter!`
  - `haskey`
  - `lock`

## Layer: 04_propagation

### Julia Functions

#### `topological_order_subset`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:368
- **Signature:** `topological_order_subset(state::MMSBState, subset::Vector{PageID})::Vector{PageID}`
- **Calls:**
  - `Set`
  - `get`
  - `get_children`
  - `isempty`
  - `popfirst!`
  - `push!`

#### `with_transaction`

- **File:** MMSB/src/04_propagation/TransactionIsolation.jl:14
- **Signature:** `with_transaction(f,s)`
- **Calls:**
  - `begin_transaction`
  - `commit_transaction`
  - `f`
  - `rethrow`
  - `rollback_transaction`

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

### Julia Functions

#### `UtilityState`

- **File:** MMSB/src/06_utility/utility_engine.jl:27
- **Signature:** `UtilityState(max_history::Int`

#### `track_delta_latency!`

- **File:** MMSB/src/06_utility/Monitoring.jl:40
- **Signature:** `track_delta_latency!(state::MMSBState, duration_ns::UInt64)`
- **Calls:**
  - `UInt64`
  - `get`

#### `track_propagation_latency!`

- **File:** MMSB/src/06_utility/Monitoring.jl:45
- **Signature:** `track_propagation_latency!(state::MMSBState, duration_ns::UInt64)`
- **Calls:**
  - `UInt64`
  - `get`

#### `update_utility!`

- **File:** MMSB/src/06_utility/utility_engine.jl:58
- **Signature:** `update_utility!(state::UtilityState, costs::CostComponents)`
- **Calls:**
  - `compute_utility`
  - `length`
  - `popfirst!`
  - `push!`

#### `utility_trend`

- **File:** MMSB/src/06_utility/utility_engine.jl:75
- **Signature:** `utility_trend(state::UtilityState)`
- **Calls:**
  - `abs`
  - `length`
  - `mean`

## Layer: 07_intention

### Julia Functions

#### `utility_gradient`

- **File:** MMSB/src/07_intention/goal_emergence.jl:17
- **Signature:** `utility_gradient(utility_history::Vector{Float64})`
- **Calls:**
  - `length`
  - `max`
  - `sum`

#### `validate_plan`

- **File:** MMSB/src/07_intention/UpsertPlan.jl:18
- **Signature:** `validate_plan(plan::UpsertPlan)`
- **Calls:**
  - `error`
  - `isempty`
  - `length`

## Layer: 08_reasoning

### Julia Functions

#### `unify_constraints`

- **File:** MMSB/src/08_reasoning/logic_engine.jl:104
- **Signature:** `unify_constraints(c1::Constraint, c2::Constraint)`
- **Calls:**
  - `Constraint`
  - `c1.predicate`
  - `c2.predicate`
  - `merge`

## Layer: 09_planning

### Julia Functions

#### `temporal_difference`

- **File:** MMSB/src/09_planning/rl_planning.jl:155
- **Signature:** `temporal_difference(trajectory::Vector{Tuple{State, Action, Float64}}, α::Float64`

#### `value_iteration`

- **File:** MMSB/src/09_planning/rl_planning.jl:17
- **Signature:** `value_iteration(states::Vector{State}, actions::Vector{Action}, γ::Float64`

## Layer: 10_agent_interface

### Julia Functions

#### `unsubscribe`

- **File:** MMSB/src/10_agent_interface/event_subscription.jl:36
- **Signature:** `unsubscribe(id::UInt64)`
- **Calls:**
  - `haskey`

## Layer: 11_agents

### Julia Functions

#### `train_step!`

- **File:** MMSB/src/11_agents/rl_agent.jl:37
- **Signature:** `train_step!(agent::RLAgent, state::MMSBState, action::AgentAction)`
- **Calls:**
  - `compute_reward`
  - `observe`
  - `push_memory!`

## Layer: 12_applications

### Julia Functions

#### `temporal_reasoning`

- **File:** MMSB/src/12_applications/memory_driven_reasoning.jl:23
- **Signature:** `temporal_reasoning(ctx::ReasoningContext)::Vector{Any}`

## Layer: root

### Rust Functions

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

### Julia Functions

#### `_to_mutable`

- **File:** MMSB/benchmark/benchmarks.jl:472
- **Signature:** `_to_mutable(value)`
- **Calls:**
  - `Dict`
  - `_to_mutable`
  - `pairs`
  - `string`

#### `_trial_to_dict`

- **File:** MMSB/benchmark/benchmarks.jl:412
- **Signature:** `_trial_to_dict(trial)`
- **Calls:**
  - `Dict`
  - `maximum`
  - `mean`
  - `median`
  - `minimum`

#### `translate_error`

- **File:** MMSB/src/ffi/RustErrors.jl:47
- **Signature:** `translate_error(err::RustFFIError; message::Union{Nothing,String}`

#### `update_page`

- **File:** MMSB/src/API.jl:88
- **Signature:** `update_page(state::MMSBState, page_id::PageID, bytes::AbstractVector{UInt8}; source::Symbol`

