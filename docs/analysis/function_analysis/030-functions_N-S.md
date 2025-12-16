# Functions N-S

## Layer: 00_physical

### Julia Functions

#### `page_compare_kernel!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:143
- **Signature:** `page_compare_kernel!(result::CuDeviceArray{Bool,1}, page1::CuDeviceArray{UInt8,1}, page2::CuDeviceArray{UInt8,1})`
- **Calls:**
  - `blockDim`
  - `blockIdx`
  - `length`
  - `threadIdx`

#### `page_copy_kernel!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:83
- **Signature:** `page_copy_kernel!(dest::CuDeviceArray{UInt8,1}, src::CuDeviceArray{UInt8,1}, n::Int)`
- **Calls:**
  - `blockDim`
  - `blockIdx`
  - `threadIdx`

#### `page_zero_kernel!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:113
- **Signature:** `page_zero_kernel!(data::CuDeviceArray{UInt8,1})`
- **Calls:**
  - `blockDim`
  - `blockIdx`
  - `length`
  - `threadIdx`

#### `prefetch_pages_to_gpu!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:351
- **Signature:** `prefetch_pages_to_gpu!(state::MMSBState, page_ids::Vector{PageID})`
- **Calls:**
  - `batch_sync_to_gpu!`
  - `filter`
  - `get_page`

#### `prefetch_unified_to_cpu!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:168
- **Signature:** `prefetch_unified_to_cpu!(page::Page)`
- **Calls:**
  - `CUDA.functional`
  - `ccall`
  - `pointer`
  - `sizeof`

#### `prefetch_unified_to_gpu!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:146
- **Signature:** `prefetch_unified_to_gpu!(page::Page, device::CuDevice`
- **Calls:**
  - `CUDA.device`

#### `resize_page!`

- **File:** MMSB/src/00_physical/PageAllocator.jl:86
- **Signature:** `resize_page!(state::MMSBState, page_id::PageID, new_size::Int64)::Page`
- **Calls:**
  - `PageNotFoundError`
  - `UInt64`
  - `get`
  - `lock`
  - `throw`

#### `set_preferred_location!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:225
- **Signature:** `set_preferred_location!(page::Page, device::Union{CuDevice, Symbol})`
- **Calls:**
  - `CUDA.Mem.advise`

#### `sparse_delta_apply_kernel!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:187
- **Signature:** `sparse_delta_apply_kernel!(base::CuDeviceArray{UInt8,1}, indices::CuDeviceArray{Int32,1}, values::CuDeviceArray{UInt8,1}, n_changes::Int)`
- **Calls:**
  - `blockDim`
  - `blockIdx`
  - `threadIdx`

#### `sync_bidirectional!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:164
- **Signature:** `sync_bidirectional!(page1::Page, page2::Page)`
- **Calls:**
  - `CPU`
  - `sync_page_to_gpu!`

#### `sync_page_to_cpu!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:130
- **Signature:** `sync_page_to_cpu!(page::Page)`
- **Calls:**
  - `CUDA.synchronize`
  - `CUDA.unsafe_free!`

#### `sync_page_to_gpu!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:99
- **Signature:** `sync_page_to_gpu!(page::Page)`
- **Calls:**
  - `CuArray`

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

### Julia Functions

#### `Page`

- **File:** MMSB/src/01_page/Page.jl:27
- **Signature:** `Page(handle::FFIWrapper.RustPageHandle, id::PageID, location::PageLocation, size::Int; metadata::Dict{Symbol,Any}`

#### `_parse_metadata_array`

- **File:** MMSB/src/01_page/Delta.jl:279
- **Signature:** `_parse_metadata_array(parser::_MetadataParser)`
- **Calls:**
  - `_consume`
  - `_parse_metadata_value`
  - `_peek`
  - `_skip_ws`
  - `push!`

#### `_parse_metadata_number`

- **File:** MMSB/src/01_page/Delta.jl:329
- **Signature:** `_parse_metadata_number(parser::_MetadataParser)`
- **Calls:**
  - `error`
  - `lastindex`
  - `lowercase`
  - `occursin`
  - `parse`
  - `tryparse`

#### `_parse_metadata_object`

- **File:** MMSB/src/01_page/Delta.jl:254
- **Signature:** `_parse_metadata_object(parser::_MetadataParser)`
- **Calls:**
  - `_consume`
  - `_parse_metadata_string`
  - `_parse_metadata_value`
  - `_peek`
  - `_skip_ws`

#### `_parse_metadata_string`

- **File:** MMSB/src/01_page/Delta.jl:300
- **Signature:** `_parse_metadata_string(parser::_MetadataParser)`
- **Calls:**
  - `IOBuffer`
  - `String`
  - `_consume`
  - `error`
  - `lastindex`
  - `print`
  - `take!`

#### `_parse_metadata_value`

- **File:** MMSB/src/01_page/Delta.jl:231
- **Signature:** `_parse_metadata_value(parser::_MetadataParser)`
- **Calls:**
  - `_parse_metadata_array`
  - `_parse_metadata_number`
  - `_parse_metadata_object`
  - `_parse_metadata_string`
  - `_peek`
  - `_skip_ws`
  - `error`
  - `startswith`

#### `_peek`

- **File:** MMSB/src/01_page/Delta.jl:360
- **Signature:** `_peek(parser::_MetadataParser)`
- **Calls:**
  - `error`
  - `lastindex`

#### `_refresh_pages!`

- **File:** MMSB/src/01_page/TLog.jl:137
- **Signature:** `_refresh_pages!(state::MMSBState)`
- **Calls:**
  - `FFIWrapper.rust_allocator_acquire_page`
  - `FFIWrapper.rust_allocator_page_infos`
  - `Int`
  - `Page`
  - `PageID`
  - `PageLocation`
  - `activate!`
  - `codeunits`
  - `empty!`
  - `initialize!`
  - `lock`
  - `metadata_from_blob`
  - `unsafe_string`

#### `_skip_ws`

- **File:** MMSB/src/01_page/Delta.jl:348
- **Signature:** `_skip_ws(parser::_MetadataParser)`
- **Calls:**
  - `in`
  - `lastindex`

#### `new_delta_handle`

- **File:** MMSB/src/01_page/Delta.jl:55
- **Signature:** `new_delta_handle(id::DeltaID, page_id::PageID, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol`

#### `page_size_bytes`

- **File:** MMSB/src/01_page/Page.jl:44
- **Signature:** `page_size_bytes(page::Page)`

#### `query_log`

- **File:** MMSB/src/01_page/TLog.jl:66
- **Signature:** `query_log(state::MMSBState; page_id::Union{PageID,Nothing}`

#### `read_page`

- **File:** MMSB/src/01_page/Page.jl:64
- **Signature:** `read_page(page::Page)::Vector{UInt8}`
- **Calls:**
  - `FFIWrapper.rust_page_read!`

#### `replay_from_checkpoint`

- **File:** MMSB/src/01_page/ReplayEngine.jl:84
- **Signature:** `replay_from_checkpoint(path::AbstractString, target_epoch::Union{UInt32, Nothing}`

#### `replay_log`

- **File:** MMSB/src/01_page/TLog.jl:114
- **Signature:** `replay_log(state::MMSBState, target_epoch::UInt32)`
- **Calls:**
  - `engine.replay_to_epoch`
  - `getfield`
  - `parentmodule`

#### `replay_page_history`

- **File:** MMSB/src/01_page/ReplayEngine.jl:92
- **Signature:** `replay_page_history(state::MMSBState, page_id::PageID)::Vector{Vector{UInt8}}`
- **Calls:**
  - `FFIWrapper.rust_page_epoch`
  - `FFIWrapper.rust_page_write_masked!`
  - `Page`
  - `TLog.query_log`
  - `UInt8`
  - `_apply_delta!`
  - `activate!`
  - `fill`
  - `get_page`
  - `initialize!`
  - `push!`
  - `read_page`

#### `replay_to_epoch`

- **File:** MMSB/src/01_page/ReplayEngine.jl:58
- **Signature:** `replay_to_epoch(state::MMSBState, target_epoch::UInt32)::MMSBState`
- **Calls:**
  - `_all_deltas`
  - `_apply_delta!`
  - `_blank_state_like`
  - `get_page`

#### `replay_to_timestamp`

- **File:** MMSB/src/01_page/ReplayEngine.jl:71
- **Signature:** `replay_to_timestamp(state::MMSBState, target_time::UInt64)::MMSBState`
- **Calls:**
  - `_all_deltas`
  - `_apply_delta!`
  - `_blank_state_like`
  - `get_page`

#### `replay_with_predicate`

- **File:** MMSB/src/01_page/ReplayEngine.jl:129
- **Signature:** `replay_with_predicate(state::MMSBState, predicate::Function)`
- **Calls:**
  - `_all_deltas`
  - `predicate`
  - `push!`

#### `serialize_delta`

- **File:** MMSB/src/01_page/Delta.jl:82
- **Signature:** `serialize_delta(delta::Delta)::Vector{UInt8}`
- **Calls:**
  - `IOBuffer`
  - `Serialization.serialize`
  - `map`
  - `take!`

#### `set_intent_metadata!`

- **File:** MMSB/src/01_page/Delta.jl:110
- **Signature:** `set_intent_metadata!(delta::Delta, metadata::Union{Nothing,AbstractString,Dict{Symbol,Any}})`
- **Calls:**
  - `ArgumentError`
  - `FFIWrapper.rust_delta_set_intent_metadata!`
  - `String`
  - `_encode_metadata_dict`
  - `throw`
  - `typeof`

## Layer: 01_types

### Julia Functions

#### `register_page!`

- **File:** MMSB/src/01_types/MMSBState.jl:135
- **Signature:** `register_page!(state::MMSBState, page::Page)`
- **Calls:**
  - `lock`

## Layer: 02_semiring

### Julia Functions

#### `propagate_change!`

- **File:** MMSB/src/02_semiring/DeltaRouter.jl:89
- **Signature:** `propagate_change!(state::MMSBState, changed_page_id::PageID)`
- **Calls:**
  - `engine.propagate_change!`
  - `getfield`
  - `parentmodule`

#### `propagate_change!`

- **File:** MMSB/src/02_semiring/DeltaRouter.jl:94
- **Signature:** `propagate_change!(state::MMSBState, changed_pages::AbstractVector{PageID})`
- **Calls:**
  - `engine.propagate_change!`
  - `getfield`
  - `parentmodule`

#### `route_delta!`

- **File:** MMSB/src/02_semiring/DeltaRouter.jl:29
- **Signature:** `route_delta!(state::MMSBState, delta::Delta; propagate::Bool`

## Layer: 03_dag

### Julia Functions

#### `ShadowPageGraph`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:36
- **Signature:** `ShadowPageGraph()`
- **Calls:**
  - `ReentrantLock`
  - `new`

#### `_serialize_event`

- **File:** MMSB/src/03_dag/EventSystem.jl:190
- **Signature:** `_serialize_event(event_type::EventType, data)::Vector{UInt8}`
- **Calls:**
  - `IOBuffer`
  - `collect`
  - `serialize`
  - `take!`

#### `node`

- **File:** MMSB/src/03_dag/GraphDSL.jl:9
- **Signature:** `node(id)`

#### `remove_dependency!`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:86
- **Signature:** `remove_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID)`
- **Calls:**
  - `filter!`
  - `haskey`
  - `lock`

#### `remove_edge!`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:69
- **Signature:** `remove_edge!(graph::ShadowPageGraph, parent::PageID, child::PageID)`
- **Calls:**
  - `filter!`
  - `haskey`
  - `lock`

#### `reverse_postorder`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:339
- **Signature:** `reverse_postorder(graph::ShadowPageGraph, start::PageID)::Vector{PageID}`
- **Calls:**
  - `dfs_postorder`
  - `get_children`
  - `lock`
  - `push!`

#### `subscribe!`

- **File:** MMSB/src/03_dag/EventSystem.jl:102
- **Signature:** `subscribe!(event_type::EventType, handler::EventHandler; filter::Union{Function, Nothing}`

## Layer: 04_propagation

### Rust Functions

#### `passthrough`

- **File:** MMSB/src/04_propagation/propagation_fastpath.rs:0
- **Visibility:** Public

### Julia Functions

#### `propagate_change!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:198
- **Signature:** `propagate_change!(state::MMSBState, changed_page_id::PageID, mode::PropagationMode`

#### `propagate_change!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:203
- **Signature:** `propagate_change!(state::MMSBState, changed_pages::AbstractVector{PageID}, mode::PropagationMode`

#### `queue_recomputation!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:185
- **Signature:** `queue_recomputation!(state::MMSBState, page_id::PageID)`
- **Calls:**
  - `_buffer`
  - `push!`

#### `recompute_page!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:310
- **Signature:** `recompute_page!(state::MMSBState, page_id::PageID)`
- **Calls:**
  - `InvalidDeltaError`
  - `UInt64`
  - `eachindex`
  - `get`
  - `get_page`
  - `length`
  - `read_page`
  - `recompute_fn`
  - `throw`

#### `register_passthrough_recompute!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:171
- **Signature:** `register_passthrough_recompute!(state::MMSBState, target_page_id::PageID, source_page_id::PageID; transform`

#### `register_recompute_fn!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:158
- **Signature:** `register_recompute_fn!(state::MMSBState, page_id::PageID, fn::Function)`
- **Calls:**
  - `PageNotFoundError`
  - `UInt64`
  - `get_page`
  - `throw`

#### `replay_cuda_graph`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:90
- **Signature:** `replay_cuda_graph(state::MMSBState, stream::Ptr{Cvoid})`
- **Calls:**
  - `ccall`
  - `haskey`

#### `rollback_transaction`

- **File:** MMSB/src/04_propagation/TransactionIsolation.jl:13
- **Signature:** `rollback_transaction(s)`
- **Calls:**
  - `delete!`
  - `haskey`

#### `schedule!`

- **File:** MMSB/src/04_propagation/PropagationScheduler.jl:8
- **Signature:** `schedule!(engine, commands)`
- **Calls:**
  - `engine.drain`
  - `engine.enqueue`

#### `schedule_gpu_sync!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:345
- **Signature:** `schedule_gpu_sync!(state::MMSBState, page_id::PageID)`
- **Calls:**
  - `emit_event!`
  - `get_page`

#### `schedule_propagation!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:280
- **Signature:** `schedule_propagation!(state::MMSBState, changed_pages::Vector{PageID})`
- **Calls:**
  - `_collect_descendants`
  - `collect`
  - `queue_recomputation!`
  - `topological_order_subset`
  - `union!`

## Layer: 05_adaptive

### Julia Functions

#### `optimize_layout!`

- **File:** MMSB/src/05_adaptive/AdaptiveLayout.jl:39
- **Signature:** `optimize_layout!(state::LayoutState, access_pattern::Dict{Tuple{PageId, PageId}, Int})`
- **Calls:**
  - `UInt64`
  - `collect`
  - `compute_locality_score`
  - `enumerate`
  - `get`
  - `hotness`
  - `isempty`
  - `keys`
  - `sort!`
  - `time`

#### `reduce_entropy!`

- **File:** MMSB/src/05_adaptive/EntropyReduction.jl:42
- **Signature:** `reduce_entropy!(layout::Dict{UInt64, UInt64}, access_pattern::Dict{UInt64, Int}, page_size::Int)`
- **Calls:**
  - `UInt64`
  - `collect`
  - `compute_entropy`
  - `enumerate`
  - `sort`

#### `rewrite_dag!`

- **File:** MMSB/src/05_adaptive/GraphRewriting.jl:33
- **Signature:** `rewrite_dag!(dag, frequency_map::Dict{Tuple{Int, Int}, Int})`
- **Calls:**
  - `EdgeRewrite`
  - `can_reorder`
  - `collect`
  - `compute_edge_cost`
  - `keys`
  - `length`
  - `push!`
  - `reorderable`
  - `reordered`

## Layer: 06_utility

### Julia Functions

#### `PageDistribution`

- **File:** MMSB/src/06_utility/entropy_measure.jl:23
- **Signature:** `PageDistribution(counts::Dict{UInt64, Int})`
- **Calls:**
  - `PageDistribution`
  - `sum`
  - `values`

#### `normalize_costs`

- **File:** MMSB/src/06_utility/CostAggregation.jl:37
- **Signature:** `normalize_costs(costs::Vector{WeightedCost})`
- **Calls:**
  - `WeightedCost`
  - `else`
  - `haskey`
  - `isempty`
  - `maximum`
  - `minimum`
  - `push!`

#### `print_stats`

- **File:** MMSB/src/06_utility/Monitoring.jl:108
- **Signature:** `print_stats(state::MMSBState)`
- **Calls:**
  - `get_stats`
  - `println`
  - `round`

#### `record_access`

- **File:** MMSB/src/06_utility/MemoryPressure.jl:11
- **Signature:** `record_access(s,p)`
- **Calls:**
  - `LRUTracker`
  - `get!`

#### `reset_stats!`

- **File:** MMSB/src/06_utility/Monitoring.jl:119
- **Signature:** `reset_stats!(state::MMSBState)`
- **Calls:**
  - `UInt64`
  - `haskey`

#### `retry_with_backoff`

- **File:** MMSB/src/06_utility/ErrorRecovery.jl:15
- **Signature:** `retry_with_backoff(f,p`
- **Calls:**
  - `RetryPolicy`

#### `state_entropy`

- **File:** MMSB/src/06_utility/entropy_measure.jl:51
- **Signature:** `state_entropy(access_pattern::Dict{Tuple{UInt64, UInt64}, Int})`
- **Calls:**
  - `isempty`
  - `log2`
  - `sum`
  - `values`

## Layer: 07_intention

### Julia Functions

#### `select_best_intention`

- **File:** MMSB/src/07_intention/intention_engine.jl:66
- **Signature:** `select_best_intention(intentions::Vector{Intention}, utility::Float64)`
- **Calls:**
  - `argmax`
  - `evaluate_intention`
  - `isempty`

## Layer: 08_reasoning

### Julia Functions

#### `ReasoningState`

- **File:** MMSB/src/08_reasoning/ReasoningTypes.jl:92
- **Signature:** `ReasoningState()`
- **Calls:**
  - `ReasoningState`

#### `perform_inference`

- **File:** MMSB/src/08_reasoning/reasoning_engine.jl:102
- **Signature:** `perform_inference(dag, state::ReasoningState, target_node::UInt64)`
- **Calls:**
  - `Inference`
  - `LogicEngine.deduce`
  - `append!`
  - `filter`
  - `get`
  - `haskey`
  - `isempty`
  - `push!`

#### `propagate_constraints`

- **File:** MMSB/src/08_reasoning/constraint_propagation.jl:17
- **Signature:** `propagate_constraints(dag, state::ReasoningState, node_id::UInt64)`
- **Calls:**
  - `Constraint`
  - `Dict`
  - `filter`
  - `get`
  - `get!`
  - `merge`
  - `push!`

#### `reason_over_dag`

- **File:** MMSB/src/08_reasoning/reasoning_engine.jl:51
- **Signature:** `reason_over_dag(dag, state::ReasoningState)`
- **Calls:**
  - `ConstraintPropagation.forward_propagate`
  - `InferenceResult`
  - `PatternFormation.match_pattern`
  - `RuleEvaluation.evaluate_rules`
  - `StructuralInference.derive_constraints`
  - `append!`
  - `get`
  - `get!`
  - `isempty`
  - `keys`
  - `push!`

## Layer: 09_planning

### Julia Functions

#### `PlanningState`

- **File:** MMSB/src/09_planning/PlanningTypes.jl:88
- **Signature:** `PlanningState(initial_state::State)`
- **Calls:**
  - `DecisionGraph`
  - `PlanningState`

#### `norm`

- **File:** MMSB/src/09_planning/optimization_planning.jl:58
- **Signature:** `norm(x::Vector{Float64})`
- **Calls:**
  - `sqrt`
  - `sum`

#### `optimize_plan`

- **File:** MMSB/src/09_planning/optimization_planning.jl:17
- **Signature:** `optimize_plan(plan::Plan, objective::Function)`
- **Calls:**
  - `compute_gradient`
  - `extract_parameters`
  - `norm`
  - `reconstruct_plan`

#### `order_subgoals`

- **File:** MMSB/src/09_planning/goal_decomposition.jl:76
- **Signature:** `order_subgoals(subgoals::Vector{Goal}, state::State)`
- **Calls:**
  - `score_subgoal`
  - `sort!`

#### `parallel_rollout`

- **File:** MMSB/src/09_planning/rollout_simulation.jl:43
- **Signature:** `parallel_rollout(plans::Vector{Plan}, start_state::State, n_rollouts::Int`

#### `policy_iteration`

- **File:** MMSB/src/09_planning/rl_planning.jl:66
- **Signature:** `policy_iteration(states::Vector{State}, actions::Vector{Action}, γ::Float64`

#### `prepare_for_enzyme`

- **File:** MMSB/src/09_planning/optimization_planning.jl:158
- **Signature:** `prepare_for_enzyme(plan::Plan)`
- **Calls:**
  - `length`

#### `prune_graph`

- **File:** MMSB/src/09_planning/decision_graphs.jl:115
- **Signature:** `prune_graph(graph::DecisionGraph, threshold::Float64)`
- **Calls:**
  - `DecisionGraph`
  - `haskey`

#### `q_learning`

- **File:** MMSB/src/09_planning/rl_planning.jl:131
- **Signature:** `q_learning(episodes::Vector{Vector{Tuple{State, Action, Float64}}}, α::Float64`

#### `reconstruct_plan`

- **File:** MMSB/src/09_planning/optimization_planning.jl:60
- **Signature:** `reconstruct_plan(plan::Plan, params::Vector{Float64})`
- **Calls:**
  - `Action`
  - `Plan`
  - `enumerate`
  - `push!`
  - `sum`

#### `reconstruct_plan`

- **File:** MMSB/src/09_planning/search_algorithms.jl:209
- **Signature:** `reconstruct_plan(node::SearchNode, goal_id::UInt64)`
- **Calls:**
  - `Plan`
  - `UInt64`
  - `pushfirst!`

#### `replan`

- **File:** MMSB/src/09_planning/planning_engine.jl:96
- **Signature:** `replan(planning_state::PlanningState, plan_id::UInt64, feedback::Dict{Symbol, Any})`
- **Calls:**
  - `RolloutSimulation.simulate_plan`
  - `SearchAlgorithms.astar_search`
  - `StrategyGeneration.generate_strategies`
  - `get`
  - `strategy.plan_generator`

#### `score_subgoal`

- **File:** MMSB/src/09_planning/goal_decomposition.jl:83
- **Signature:** `score_subgoal(goal::Goal, state::State)`
- **Calls:**
  - `estimate_achievability`

#### `select_node`

- **File:** MMSB/src/09_planning/search_algorithms.jl:100
- **Signature:** `select_node(node::MCTSNode)`
- **Calls:**
  - `best_uct_child`
  - `isempty`

#### `select_strategy`

- **File:** MMSB/src/09_planning/strategy_generation.jl:75
- **Signature:** `select_strategy(strategies::Vector{Strategy}, goal::Goal, state::State)`
- **Calls:**
  - `argmax`
  - `isempty`
  - `push!`

#### `simulate`

- **File:** MMSB/src/09_planning/search_algorithms.jl:143
- **Signature:** `simulate(state::State, goal::Goal, actions::Vector{Action}, max_depth::Int)`
- **Calls:**
  - `apply_action`
  - `can_apply`
  - `filter`
  - `goal.predicate`
  - `isempty`
  - `rand`

#### `simulate_plan`

- **File:** MMSB/src/09_planning/rollout_simulation.jl:18
- **Signature:** `simulate_plan(plan::Plan, start_state::State)`
- **Calls:**
  - `RolloutResult`
  - `SearchAlgorithms.apply_action`
  - `SearchAlgorithms.can_apply`
  - `push!`

## Layer: 10_agent_interface

### Julia Functions

#### `observe`

- **File:** MMSB/src/10_agent_interface/AgentProtocol.jl:25
- **Signature:** `observe end`
- **Calls:**
  - `act!`
  - `plan`

#### `restore_checkpoint`

- **File:** MMSB/src/10_agent_interface/checkpoint_api.jl:17
- **Signature:** `restore_checkpoint(state::MMSBState, path::String)`
- **Calls:**
  - `load_checkpoint!`

#### `subscribe_to_events`

- **File:** MMSB/src/10_agent_interface/event_subscription.jl:29
- **Signature:** `subscribe_to_events(types::Vector{EventType}, callback::Function)::UInt64`
- **Calls:**
  - `Set`
  - `Subscription`

## Layer: 11_agents

### Julia Functions

#### `RLAgent`

- **File:** MMSB/src/11_agents/rl_agent.jl:20
- **Signature:** `RLAgent(initial_state::T; lr`

#### `SymbolicAgent`

- **File:** MMSB/src/11_agents/symbolic_agent.jl:25
- **Signature:** `SymbolicAgent()`
- **Calls:**
  - `AgentState`
  - `SymbolicAgent`

#### `neural_step!`

- **File:** MMSB/src/11_agents/hybrid_agent.jl:36
- **Signature:** `neural_step!(agent::HybridAgent, state::MMSBState, action::AgentAction)`
- **Calls:**
  - `train_step!`

#### `observe`

- **File:** MMSB/src/11_agents/hybrid_agent.jl:23
- **Signature:** `observe(agent::HybridAgent, state::MMSBState)`
- **Calls:**
  - `observe`
  - `return`

#### `observe`

- **File:** MMSB/src/11_agents/planning_agent.jl:23
- **Signature:** `observe(agent::PlanningAgent, state::MMSBState)`
- **Calls:**
  - `compute_intention`
  - `return`

#### `observe`

- **File:** MMSB/src/11_agents/rl_agent.jl:24
- **Signature:** `observe(agent::RLAgent, state::MMSBState)`
- **Calls:**
  - `length`
  - `return`

#### `observe`

- **File:** MMSB/src/11_agents/symbolic_agent.jl:27
- **Signature:** `observe(agent::SymbolicAgent, state::MMSBState)`
- **Calls:**
  - `return`
  - `structural_inference`

#### `push_memory!`

- **File:** MMSB/src/11_agents/AgentTypes.jl:29
- **Signature:** `push_memory!(mem::AgentMemory, obs, action, reward::Float64)`
- **Calls:**
  - `length`
  - `popfirst!`
  - `push!`

#### `symbolic_step!`

- **File:** MMSB/src/11_agents/hybrid_agent.jl:30
- **Signature:** `symbolic_step!(agent::HybridAgent, state::MMSBState)::Vector{AgentAction}`
- **Calls:**
  - `apply_rule`
  - `first`
  - `isempty`

## Layer: 12_applications

### Julia Functions

#### `query_llm`

- **File:** MMSB/src/12_applications/llm_tools.jl:19
- **Signature:** `query_llm(ctx::MMSBContext, prompt::String)::String`

#### `reason_over_memory`

- **File:** MMSB/src/12_applications/memory_driven_reasoning.jl:18
- **Signature:** `reason_over_memory(ctx::ReasoningContext)::Dict{Symbol, Any}`
- **Calls:**
  - `Dict`
  - `structural_inference`

#### `rebalance!`

- **File:** MMSB/src/12_applications/financial_modeling.jl:33
- **Signature:** `rebalance!(portfolio::Portfolio, target_weights::Dict{String, Float64})`

#### `register_agent!`

- **File:** MMSB/src/12_applications/multi_agent_system.jl:19
- **Signature:** `register_agent!(coord::AgentCoordinator, agent::AbstractAgent)`
- **Calls:**
  - `push!`

#### `simulate_step!`

- **File:** MMSB/src/12_applications/world_simulation.jl:35
- **Signature:** `simulate_step!(world::World)`

#### `store_llm_response`

- **File:** MMSB/src/12_applications/llm_tools.jl:25
- **Signature:** `store_llm_response(ctx::MMSBContext, response::String)`

## Layer: root

### Rust Functions

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

#### `set_last_error`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `with`
  - `borrow_mut`

#### `slice_from_ptr`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Generics:** 'a, T
- **Calls:**
  - `is_null`
  - `slice::from_raw_parts`

### Julia Functions

#### `_page`

- **File:** MMSB/benchmark/benchmarks.jl:53
- **Signature:** `_page(state, bytes::Int; location::Symbol`

#### `_populate_pages!`

- **File:** MMSB/benchmark/benchmarks.jl:57
- **Signature:** `_populate_pages!(state, count::Int, bytes::Int)`
- **Calls:**
  - `API.create_page`

#### `_replay_sequence!`

- **File:** MMSB/benchmark/benchmarks.jl:68
- **Signature:** `_replay_sequence!(state, page, epochs::Int, bytes::Int)`
- **Calls:**
  - `API.update_page`
  - `rand`

#### `_resolve_location`

- **File:** MMSB/src/API.jl:52
- **Signature:** `_resolve_location(location::Symbol)`
- **Calls:**
  - `CUDA.functional`
  - `GPUMemoryError`
  - `UnsupportedLocationError`
  - `string`
  - `throw`

#### `_seed_pages!`

- **File:** MMSB/benchmark/benchmarks.jl:61
- **Signature:** `_seed_pages!(state, count::Int, bytes::Int)`
- **Calls:**
  - `API.update_page`
  - `_page`
  - `rand`

#### `_select_suite`

- **File:** MMSB/benchmark/benchmarks.jl:423
- **Signature:** `_select_suite(categories::Vector{String})`
- **Calls:**
  - `BenchmarkGroup`
  - `haskey`

#### `_start_state`

- **File:** MMSB/benchmark/benchmarks.jl:43
- **Signature:** `_start_state(; enable_gpu::Bool`

#### `_stop_state!`

- **File:** MMSB/benchmark/benchmarks.jl:47
- **Signature:** `_stop_state!(state)`
- **Calls:**
  - `API.mmsb_stop`

#### `_stress_updates!`

- **File:** MMSB/benchmark/benchmarks.jl:74
- **Signature:** `_stress_updates!(state, pages, updates::Int, bytes::Int)`
- **Calls:**
  - `API.update_page`
  - `length`
  - `rand`

#### `query_page`

- **File:** MMSB/src/API.jl:113
- **Signature:** `query_page(state::MMSBState, page_id::PageID)::Vector{UInt8}`
- **Calls:**
  - `PageNotFoundError`
  - `UInt64`
  - `get_page`
  - `read_page`
  - `throw`

#### `register_error_hook`

- **File:** MMSB/src/ffi/FFIWrapper.jl:9
- **Signature:** `register_error_hook(f::Function)`

#### `run_benchmarks`

- **File:** MMSB/benchmark/benchmarks.jl:482
- **Signature:** `run_benchmarks(; save_results::Bool`

#### `rust_allocator_acquire_page`

- **File:** MMSB/src/ffi/FFIWrapper.jl:426
- **Signature:** `rust_allocator_acquire_page(handle::RustAllocatorHandle, page_id::UInt64)::RustPageHandle`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_allocator_allocate`

- **File:** MMSB/src/ffi/FFIWrapper.jl:185
- **Signature:** `rust_allocator_allocate( handle::RustAllocatorHandle, page_id::UInt64, size::Int, location::Int32`

#### `rust_allocator_free!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:178
- **Signature:** `rust_allocator_free!(handle::RustAllocatorHandle)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_allocator_get_page`

- **File:** MMSB/src/ffi/FFIWrapper.jl:214
- **Signature:** `rust_allocator_get_page(handle::RustAllocatorHandle, page_id::UInt64)::RustPageHandle`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_allocator_new`

- **File:** MMSB/src/ffi/FFIWrapper.jl:171
- **Signature:** `rust_allocator_new()::RustAllocatorHandle`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_allocator_page_infos`

- **File:** MMSB/src/ffi/FFIWrapper.jl:411
- **Signature:** `rust_allocator_page_infos(handle::RustAllocatorHandle)::Vector{RustPageInfo}`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`
  - `pointer`

#### `rust_allocator_release!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:202
- **Signature:** `rust_allocator_release!(handle::RustAllocatorHandle, page_id::UInt64)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_artifacts_available`

- **File:** MMSB/src/ffi/FFIWrapper.jl:68
- **Signature:** `rust_artifacts_available()`
- **Calls:**
  - `isfile`

#### `rust_checkpoint_load!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:402
- **Signature:** `rust_checkpoint_load!(allocator::RustAllocatorHandle, log::RustTLogHandle, path::AbstractString)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_checkpoint_write!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:393
- **Signature:** `rust_checkpoint_write!(allocator::RustAllocatorHandle, log::RustTLogHandle, path::AbstractString)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_apply!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:160
- **Signature:** `rust_delta_apply!(page::RustPageHandle, delta::RustDeltaHandle)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_epoch`

- **File:** MMSB/src/ffi/FFIWrapper.jl:297
- **Signature:** `rust_delta_epoch(handle::RustDeltaHandle)::UInt32`
- **Calls:**
  - `UInt32`
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_free!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:153
- **Signature:** `rust_delta_free!(handle::RustDeltaHandle)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_id`

- **File:** MMSB/src/ffi/FFIWrapper.jl:283
- **Signature:** `rust_delta_id(handle::RustDeltaHandle)::UInt64`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_intent_metadata`

- **File:** MMSB/src/ffi/FFIWrapper.jl:379
- **Signature:** `rust_delta_intent_metadata(handle::RustDeltaHandle)::Union{Nothing,String}`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`
  - `pointer`

#### `rust_delta_is_sparse`

- **File:** MMSB/src/ffi/FFIWrapper.jl:304
- **Signature:** `rust_delta_is_sparse(handle::RustDeltaHandle)::Bool`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_mask`

- **File:** MMSB/src/ffi/FFIWrapper.jl:334
- **Signature:** `rust_delta_mask(handle::RustDeltaHandle)::Vector{UInt8}`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`
  - `pointer`

#### `rust_delta_new`

- **File:** MMSB/src/ffi/FFIWrapper.jl:140
- **Signature:** `rust_delta_new(delta_id::UInt64, page_id::UInt64, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol; is_sparse::Bool`

#### `rust_delta_page_id`

- **File:** MMSB/src/ffi/FFIWrapper.jl:290
- **Signature:** `rust_delta_page_id(handle::RustDeltaHandle)::UInt64`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_delta_payload`

- **File:** MMSB/src/ffi/FFIWrapper.jl:348
- **Signature:** `rust_delta_payload(handle::RustDeltaHandle)::Vector{UInt8}`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`
  - `pointer`

#### `rust_delta_set_intent_metadata!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:362
- **Signature:** `rust_delta_set_intent_metadata!(handle::RustDeltaHandle, metadata::Union{Nothing,AbstractString})`
- **Calls:**
  - `String`
  - `_check_rust_error`
  - `ccall`
  - `codeunits`
  - `ensure_rust_artifacts`
  - `length`
  - `pointer`

#### `rust_delta_source`

- **File:** MMSB/src/ffi/FFIWrapper.jl:318
- **Signature:** `rust_delta_source(handle::RustDeltaHandle)::String`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`
  - `pointer`

#### `rust_delta_timestamp`

- **File:** MMSB/src/ffi/FFIWrapper.jl:311
- **Signature:** `rust_delta_timestamp(handle::RustDeltaHandle)::UInt64`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_get_last_error`

- **File:** MMSB/src/ffi/FFIWrapper.jl:434
- **Signature:** `rust_get_last_error()::Int32`
- **Calls:**
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_page_epoch`

- **File:** MMSB/src/ffi/FFIWrapper.jl:97
- **Signature:** `rust_page_epoch(handle::RustPageHandle)::UInt32`
- **Calls:**
  - `UInt32`
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_page_metadata_blob`

- **File:** MMSB/src/ffi/FFIWrapper.jl:104
- **Signature:** `rust_page_metadata_blob(handle::RustPageHandle)::Vector{UInt8}`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`
  - `pointer`

#### `rust_page_metadata_import!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:118
- **Signature:** `rust_page_metadata_import!(handle::RustPageHandle, blob::Vector{UInt8})`
- **Calls:**
  - `ccall`
  - `ensure_rust_artifacts`
  - `length`
  - `pointer`

#### `rust_page_read!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:76
- **Signature:** `rust_page_read!(handle::RustPageHandle, buffer::Vector{UInt8})`
- **Calls:**
  - `UInt`
  - `ccall`
  - `ensure_rust_artifacts`
  - `isempty`
  - `length`
  - `pointer`
  - `string`

#### `rust_page_write_masked!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:128
- **Signature:** `rust_page_write_masked!(handle::RustPageHandle, mask::Vector{UInt8}, payload::Vector{UInt8}; is_sparse::Bool`

#### `rust_semiring_boolean_accumulate`

- **File:** MMSB/src/ffi/FFIWrapper.jl:491
- **Signature:** `rust_semiring_boolean_accumulate(left::Bool, right::Bool)`
- **Calls:**
  - `UInt8`
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_semiring_boolean_fold_add`

- **File:** MMSB/src/ffi/FFIWrapper.jl:469
- **Signature:** `rust_semiring_boolean_fold_add(values::Vector{UInt8})::Bool`
- **Calls:**
  - `ccall`
  - `ensure_rust_artifacts`
  - `isempty`
  - `length`
  - `pointer`

#### `rust_semiring_boolean_fold_mul`

- **File:** MMSB/src/ffi/FFIWrapper.jl:480
- **Signature:** `rust_semiring_boolean_fold_mul(values::Vector{UInt8})::Bool`
- **Calls:**
  - `ccall`
  - `ensure_rust_artifacts`
  - `isempty`
  - `length`
  - `pointer`

#### `rust_semiring_tropical_accumulate`

- **File:** MMSB/src/ffi/FFIWrapper.jl:461
- **Signature:** `rust_semiring_tropical_accumulate(left::Float64, right::Float64)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_semiring_tropical_fold_add`

- **File:** MMSB/src/ffi/FFIWrapper.jl:439
- **Signature:** `rust_semiring_tropical_fold_add(values::Vector{Float64})::Float64`
- **Calls:**
  - `ccall`
  - `ensure_rust_artifacts`
  - `isempty`
  - `length`
  - `pointer`

#### `rust_semiring_tropical_fold_mul`

- **File:** MMSB/src/ffi/FFIWrapper.jl:450
- **Signature:** `rust_semiring_tropical_fold_mul(values::Vector{Float64})::Float64`
- **Calls:**
  - `ccall`
  - `ensure_rust_artifacts`
  - `isempty`
  - `length`
  - `pointer`

#### `rust_tlog_append!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:240
- **Signature:** `rust_tlog_append!(handle::RustTLogHandle, delta::RustDeltaHandle)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_tlog_free!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:233
- **Signature:** `rust_tlog_free!(handle::RustTLogHandle)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_tlog_new`

- **File:** MMSB/src/ffi/FFIWrapper.jl:226
- **Signature:** `rust_tlog_new(path::AbstractString)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_tlog_reader_free!`

- **File:** MMSB/src/ffi/FFIWrapper.jl:254
- **Signature:** `rust_tlog_reader_free!(handle::RustTLogReaderHandle)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_tlog_reader_new`

- **File:** MMSB/src/ffi/FFIWrapper.jl:247
- **Signature:** `rust_tlog_reader_new(path::AbstractString)`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_tlog_reader_next`

- **File:** MMSB/src/ffi/FFIWrapper.jl:261
- **Signature:** `rust_tlog_reader_next(handle::RustTLogReaderHandle)::RustDeltaHandle`
- **Calls:**
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

#### `rust_tlog_summary`

- **File:** MMSB/src/ffi/FFIWrapper.jl:268
- **Signature:** `rust_tlog_summary(path::AbstractString)`
- **Calls:**
  - `Int`
  - `RustTLogSummary`
  - `UInt32`
  - `_check_rust_error`
  - `ccall`
  - `ensure_rust_artifacts`

