# Functions A-F

## Layer: 00_physical

### Julia Functions

#### `CPUPropagationQueue`

- **File:** MMSB/src/00_physical/DeviceFallback.jl:9
- **Signature:** `CPUPropagationQueue()`
- **Calls:**
  - `CPUPropagationQueue`

#### `adaptive_prefetch_distance`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:194
- **Signature:** `adaptive_prefetch_distance(latency_history::Vector{Float64})::Int`
- **Calls:**
  - `isempty`
  - `mean`

#### `allocate_from_pool`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:54
- **Signature:** `allocate_from_pool(pool::GPUMemoryPool, size::UInt64)::Ptr{Cvoid}`
- **Calls:**
  - `ccall`

#### `allocate_page_arrays`

- **File:** MMSB/src/00_physical/PageAllocator.jl:99
- **Signature:** `allocate_page_arrays(size::Int64, location::PageLocation)`
- **Calls:**
  - `error`

#### `async_sync_page_to_gpu!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:222
- **Signature:** `async_sync_page_to_gpu!(page::Page, stream::CuStream)`
- **Calls:**
  - `CuArray`

#### `batch_sync_to_cpu!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:280
- **Signature:** `batch_sync_to_cpu!(pages::Vector{Page})`
- **Calls:**
  - `CUDA.synchronize`
  - `filter`
  - `isempty`
  - `sync_page_to_cpu!`

#### `batch_sync_to_gpu!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:253
- **Signature:** `batch_sync_to_gpu!(pages::Vector{Page})`
- **Calls:**
  - `filter`
  - `isempty`
  - `sync_page_to_gpu!`

#### `clone_page`

- **File:** MMSB/src/00_physical/PageAllocator.jl:106
- **Signature:** `clone_page(state::MMSBState, page_id::PageID)::Page`
- **Calls:**
  - `error`

#### `compute_optimal_kernel_config`

- **File:** MMSB/src/00_physical/GPUKernels.jl:230
- **Signature:** `compute_optimal_kernel_config(data_size::Int)::Tuple{Int, Int}`

#### `convert_to_unified!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:249
- **Signature:** `convert_to_unified!(page::Page)`
- **Calls:**
  - `sync_page_to_cpu!`

#### `create_gpu_command_buffer`

- **File:** MMSB/src/00_physical/DeviceSync.jl:38
- **Signature:** `create_gpu_command_buffer(capacity::UInt32`

#### `create_page!`

- **File:** MMSB/src/00_physical/PageAllocator.jl:22
- **Signature:** `create_page!(state::MMSBState, size::Int64, location::PageLocation; metadata::Dict{Symbol,Any}`

#### `create_unified_page!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:128
- **Signature:** `create_unified_page!(state::MMSBState, size::Int64)::Page`
- **Calls:**
  - `Page`
  - `allocate_page_id!`
  - `ccall`

#### `deallocate_to_pool`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:66
- **Signature:** `deallocate_to_pool(pool::GPUMemoryPool, ptr::Ptr{Cvoid}, size::UInt64)`
- **Calls:**
  - `ccall`

#### `delete_page!`

- **File:** MMSB/src/00_physical/PageAllocator.jl:49
- **Signature:** `delete_page!(state::MMSBState, page_id::PageID)`
- **Calls:**
  - `PageNotFoundError`
  - `UInt64`
  - `delete!`
  - `get`
  - `get_children`
  - `get_parents`
  - `lock`
  - `remove_dependency!`
  - `throw`

#### `delta_merge_kernel!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:33
- **Signature:** `delta_merge_kernel!(base::CuDeviceArray{UInt8,1}, mask::CuDeviceArray{Bool,1}, delta::CuDeviceArray{UInt8,1})`
- **Calls:**
  - `blockDim`
  - `blockIdx`
  - `length`
  - `threadIdx`

#### `disable_read_mostly_hint!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:292
- **Signature:** `disable_read_mostly_hint!(page::Page)`
- **Calls:**
  - `CUDA.Mem.advise`

#### `enable_read_mostly_hint!`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:278
- **Signature:** `enable_read_mostly_hint!(page::Page)`
- **Calls:**
  - `CUDA.Mem.advise`

#### `enqueue_propagation_command`

- **File:** MMSB/src/00_physical/DeviceSync.jl:60
- **Signature:** `enqueue_propagation_command(buf::GPUCommandBuffer, page::Page, deps::Vector{Page})`

#### `ensure_page_on_device!`

- **File:** MMSB/src/00_physical/DeviceSync.jl:194
- **Signature:** `ensure_page_on_device!(page::Page, target::PageLocation)::Bool`
- **Calls:**
  - `UnsupportedLocationError`
  - `string`
  - `sync_page_to_cpu!`
  - `sync_page_to_gpu!`
  - `throw`

#### `fallback_to_cpu`

- **File:** MMSB/src/00_physical/DeviceFallback.jl:10
- **Signature:** `fallback_to_cpu(f)`
- **Calls:**
  - `f`

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

#### `encode_rle`

- **File:** MMSB/src/01_page/tlog_compression.rs:0
- **Visibility:** Private
- **Calls:**
  - `Vec::new`
  - `is_empty`
  - `push`
  - `push`

### Julia Functions

#### `Delta`

- **File:** MMSB/src/01_page/Delta.jl:24
- **Signature:** `Delta(id::DeltaID, page_id::PageID, epoch::UInt32, mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol`

#### `Delta`

- **File:** MMSB/src/01_page/Delta.jl:36
- **Signature:** `Delta(handle::FFIWrapper.RustDeltaHandle)`
- **Calls:**
  - `FFIWrapper.rust_delta_epoch`
  - `FFIWrapper.rust_delta_free!`
  - `FFIWrapper.rust_delta_id`
  - `FFIWrapper.rust_delta_intent_metadata`
  - `FFIWrapper.rust_delta_is_sparse`
  - `FFIWrapper.rust_delta_mask`
  - `FFIWrapper.rust_delta_page_id`
  - `FFIWrapper.rust_delta_payload`
  - `FFIWrapper.rust_delta_source`
  - `FFIWrapper.rust_delta_timestamp`
  - `PageID`
  - `Symbol`
  - `error`
  - `finalizer`
  - `new`

#### `_all_deltas`

- **File:** MMSB/src/01_page/ReplayEngine.jl:54
- **Signature:** `_all_deltas(state::MMSBState)`
- **Calls:**
  - `TLog.query_log`

#### `_apply_delta!`

- **File:** MMSB/src/01_page/ReplayEngine.jl:48
- **Signature:** `_apply_delta!(page::Page, delta::Delta)`
- **Calls:**
  - `FFIWrapper.rust_delta_apply!`

#### `_apply_metadata!`

- **File:** MMSB/src/01_page/Page.jl:73
- **Signature:** `_apply_metadata!(page::Page, metadata::Dict{Symbol,Any})`
- **Calls:**
  - `FFIWrapper.rust_page_metadata_import!`
  - `_encode_metadata_dict`
  - `isempty`

#### `_blank_state_like`

- **File:** MMSB/src/01_page/ReplayEngine.jl:13
- **Signature:** `_blank_state_like(state::MMSBState)::MMSBState`
- **Calls:**
  - `FFIWrapper.rust_allocator_allocate`
  - `Int32`
  - `MMSBConfig`
  - `MMSBState`
  - `Page`
  - `UInt64`
  - `activate!`
  - `initialize!`
  - `lock`
  - `register_page!`

#### `_coerce_metadata_value`

- **File:** MMSB/src/01_page/Page.jl:96
- **Signature:** `_coerce_metadata_value(value::Any)`
- **Calls:**
  - `ArgumentError`
  - `String`
  - `codeunits`
  - `throw`

#### `_consume`

- **File:** MMSB/src/01_page/Delta.jl:354
- **Signature:** `_consume(parser::_MetadataParser, expected::Char)`
- **Calls:**
  - `_peek`
  - `error`

#### `_decode_metadata`

- **File:** MMSB/src/01_page/Delta.jl:224
- **Signature:** `_decode_metadata(json::String)`
- **Calls:**
  - `Dict`
  - `Symbol`
  - `_MetadataParser`
  - `_parse_metadata_value`
  - `error`

#### `_decode_metadata_blob`

- **File:** MMSB/src/01_page/Page.jl:102
- **Signature:** `_decode_metadata_blob(blob::Vector{UInt8})`
- **Calls:**
  - `IOBuffer`
  - `String`
  - `Symbol`
  - `isempty`
  - `read`

#### `_encode_metadata_dict`

- **File:** MMSB/src/01_page/Delta.jl:160
- **Signature:** `_encode_metadata_dict(metadata::AbstractDict)`
- **Calls:**
  - `String`
  - `Symbol`
  - `_encode_metadata_value`
  - `_escape_metadata_string`
  - `join`
  - `push!`
  - `string`

#### `_encode_metadata_dict`

- **File:** MMSB/src/01_page/Page.jl:81
- **Signature:** `_encode_metadata_dict(metadata::Dict{Symbol,Any})`
- **Calls:**
  - `IOBuffer`
  - `String`
  - `UInt32`
  - `_coerce_metadata_value`
  - `codeunits`
  - `length`
  - `take!`
  - `write`

#### `_encode_metadata_value`

- **File:** MMSB/src/01_page/Delta.jl:136
- **Signature:** `_encode_metadata_value(value)`
- **Calls:**
  - `Dict`
  - `String`
  - `Symbol`
  - `_encode_metadata_dict`
  - `_encode_metadata_value`
  - `_escape_metadata_string`
  - `join`
  - `string`

#### `_escape_metadata_string`

- **File:** MMSB/src/01_page/Delta.jl:169
- **Signature:** `_escape_metadata_string(str::AbstractString)`
- **Calls:**
  - `IOBuffer`
  - `String`
  - `print`
  - `take!`

#### `activate!`

- **File:** MMSB/src/01_page/Page.jl:52
- **Signature:** `activate!(page::Page)`

#### `append_to_log!`

- **File:** MMSB/src/01_page/TLog.jl:39
- **Signature:** `append_to_log!(state::MMSBState, delta::Delta)`
- **Calls:**
  - `FFIWrapper.rust_tlog_append!`
  - `_with_rust_errors`

#### `apply_delta!`

- **File:** MMSB/src/01_page/Delta.jl:61
- **Signature:** `apply_delta!(page_handle::FFIWrapper.RustPageHandle, delta::Delta)`
- **Calls:**
  - `FFIWrapper.rust_delta_apply!`

#### `checkpoint_log!`

- **File:** MMSB/src/01_page/TLog.jl:120
- **Signature:** `checkpoint_log!(state::MMSBState, path::AbstractString)`
- **Calls:**
  - `FFIWrapper.rust_checkpoint_write!`
  - `_with_rust_errors`

#### `compress_delta_mask`

- **File:** MMSB/src/01_page/TLog.jl:19
- **Signature:** `compress_delta_mask(mask::Vector{Bool}, mode::CompressionMode)`
- **Calls:**
  - `UInt8.`
  - `return`

#### `compute_diff`

- **File:** MMSB/src/01_page/ReplayEngine.jl:146
- **Signature:** `compute_diff(::MMSBState, ::MMSBState)`
- **Calls:**
  - `error`

#### `compute_log_statistics`

- **File:** MMSB/src/01_page/TLog.jl:105
- **Signature:** `compute_log_statistics(state::MMSBState)`
- **Calls:**
  - `Dict`
  - `log_summary`

#### `deactivate!`

- **File:** MMSB/src/01_page/Page.jl:58
- **Signature:** `deactivate!(page::Page)`

#### `dense_data`

- **File:** MMSB/src/01_page/Delta.jl:65
- **Signature:** `dense_data(delta::Delta)::Vector{UInt8}`
- **Calls:**
  - `copy`
  - `eachindex`
  - `length`

#### `deserialize_delta`

- **File:** MMSB/src/01_page/Delta.jl:100
- **Signature:** `deserialize_delta(bytes::Vector{UInt8})::Delta`
- **Calls:**
  - `Delta`
  - `IOBuffer`
  - `Serialization.deserialize`
  - `UInt8.`

## Layer: 01_types

### Julia Functions

#### `Base.showerror`

- **File:** MMSB/src/01_types/Errors.jl:46
- **Signature:** `Base.showerror(io::IO, err::InvalidDeltaError)`

#### `allocate_delta_id!`

- **File:** MMSB/src/01_types/MMSBState.jl:111
- **Signature:** `allocate_delta_id!(state::MMSBState)::DeltaID`
- **Calls:**
  - `DeltaID`
  - `lock`

#### `allocate_page_id!`

- **File:** MMSB/src/01_types/MMSBState.jl:91
- **Signature:** `allocate_page_id!(state::MMSBState)::PageID`
- **Calls:**
  - `_reserve_page_id_unlocked!`
  - `lock`

## Layer: 02_semiring

### Rust Functions

#### `accumulate`

- **File:** MMSB/src/02_semiring/semiring_ops.rs:0
- **Visibility:** Public
- **Generics:** S
- **Calls:**
  - `add`
  - `mul`

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

### Julia Functions

#### `_FLOAT_BUF`

- **File:** MMSB/src/02_semiring/Semiring.jl:34
- **Signature:** `_FLOAT_BUF(values::AbstractVector{<:Real})`

#### `_bool_buf`

- **File:** MMSB/src/02_semiring/Semiring.jl:36
- **Signature:** `_bool_buf(values::AbstractVector{Bool})`
- **Calls:**
  - `UInt8`
  - `eachindex`
  - `length`

#### `batch_route_deltas!`

- **File:** MMSB/src/02_semiring/DeltaRouter.jl:71
- **Signature:** `batch_route_deltas!(state::MMSBState, deltas::Vector{Delta})`
- **Calls:**
  - `collect`
  - `get!`
  - `get_page`
  - `isempty`
  - `propagate_change!`
  - `push!`
  - `route_delta!`
  - `sort`

#### `boolean_accumulate`

- **File:** MMSB/src/02_semiring/Semiring.jl:88
- **Signature:** `boolean_accumulate(left::Bool, right::Bool)`
- **Calls:**
  - `FFIWrapper.rust_semiring_boolean_accumulate`

#### `boolean_fold_add`

- **File:** MMSB/src/02_semiring/Semiring.jl:72
- **Signature:** `boolean_fold_add(values::AbstractVector{Bool})`
- **Calls:**
  - `FFIWrapper.rust_semiring_boolean_fold_add`
  - `_bool_buf`

#### `boolean_fold_mul`

- **File:** MMSB/src/02_semiring/Semiring.jl:80
- **Signature:** `boolean_fold_mul(values::AbstractVector{Bool})`
- **Calls:**
  - `FFIWrapper.rust_semiring_boolean_fold_mul`
  - `_bool_buf`

#### `boolean_semiring`

- **File:** MMSB/src/02_semiring/Semiring.jl:25
- **Signature:** `boolean_semiring()`
- **Calls:**
  - `SemiringOps`

#### `build_semiring`

- **File:** MMSB/src/02_semiring/SemiringConfig.jl:9
- **Signature:** `build_semiring(config::SemiringConfigOptions)`
- **Calls:**
  - `error`
  - `return`

#### `create_delta`

- **File:** MMSB/src/02_semiring/DeltaRouter.jl:47
- **Signature:** `create_delta(state::MMSBState, page_id::PageID, mask::AbstractVector{Bool}, data::AbstractVector{UInt8}; source::Symbol`

## Layer: 03_dag

### Rust Functions

#### `dfs`

- **File:** MMSB/src/03_dag/cycle_detection.rs:0
- **Visibility:** Private
- **Calls:**
  - `get`
  - `insert`
  - `get`
  - `dfs`
  - `insert`

### Julia Functions

#### `EventSubscription`

- **File:** MMSB/src/03_dag/EventSystem.jl:57
- **Signature:** `EventSubscription(id::UInt64, event_type::EventType, handler::EventHandler, filter::Union{Function, Nothing}`

#### `_all_vertices`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:154
- **Signature:** `_all_vertices(graph::ShadowPageGraph)`
- **Calls:**
  - `collect`
  - `keys`
  - `push!`

#### `_dfs_has_cycle`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:122
- **Signature:** `_dfs_has_cycle(graph::ShadowPageGraph, node::PageID, visited::Dict{PageID, Symbol})`
- **Calls:**
  - `_dfs_has_cycle`
  - `get`

#### `_ensure_vertex!`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:48
- **Signature:** `_ensure_vertex!(graph::ShadowPageGraph, node::PageID)`
- **Calls:**
  - `haskey`

#### `add_dependency!`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:63
- **Signature:** `add_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID, edge_type::EdgeType)`
- **Calls:**
  - `GraphCycleError`
  - `UInt64`
  - `_ensure_vertex!`
  - `has_cycle`
  - `lock`
  - `push!`
  - `remove_dependency!`
  - `throw`

#### `add_edge!`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:33
- **Signature:** `add_edge!(graph::ShadowPageGraph, parent::PageID, child::PageID, edge_type::EdgeType)`
- **Calls:**
  - `has_cycle_after_add`
  - `haskey`
  - `lock`
  - `push!`

#### `clear_subscriptions!`

- **File:** MMSB/src/03_dag/EventSystem.jl:138
- **Signature:** `clear_subscriptions!()`
- **Calls:**
  - `empty!`
  - `lock`

#### `compute_closure`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:373
- **Signature:** `compute_closure(graph::ShadowPageGraph, roots::Vector{PageID})::Set{PageID}`
- **Calls:**
  - `find_descendants`
  - `union!`

#### `create_debug_subscriber`

- **File:** MMSB/src/03_dag/EventSystem.jl:161
- **Signature:** `create_debug_subscriber(event_types::Vector{EventType}; verbose::Bool`

#### `create_logging_subscriber`

- **File:** MMSB/src/03_dag/EventSystem.jl:182
- **Signature:** `create_logging_subscriber(state::MMSBState, log_page_id::PageID)::Vector{EventSubscription}`
- **Calls:**
  - `instances`
  - `log_event_to_page!`
  - `subscribe!`

#### `detect_cycles`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:201
- **Signature:** `detect_cycles(graph::ShadowPageGraph)::Union{Vector{PageID}, Nothing}`
- **Calls:**
  - `dfs_cycle_detect`
  - `keys`
  - `lock`

#### `dfs_cycle_detect`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:236
- **Signature:** `dfs_cycle_detect(graph::ShadowPageGraph, node::PageID, color::Dict{PageID, Symbol}, parent::Dict{PageID, PageID})`
- **Calls:**
  - `dfs_cycle_detect`
  - `get_children`
  - `push!`
  - `reverse`

#### `emit_event!`

- **File:** MMSB/src/03_dag/EventSystem.jl:78
- **Signature:** `emit_event!(state::MMSBState, event_type::EventType, data...)`
- **Calls:**
  - `copy`
  - `get`
  - `lock`

#### `find_ancestors`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:167
- **Signature:** `find_ancestors(graph::ShadowPageGraph, node::PageID)::Set{PageID}`
- **Calls:**
  - `get_parents`
  - `isempty`
  - `lock`
  - `popfirst!`
  - `push!`

#### `find_descendants`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:137
- **Signature:** `find_descendants(graph::ShadowPageGraph, root::PageID)::Set{PageID}`
- **Calls:**
  - `get_children`
  - `isempty`
  - `lock`
  - `popfirst!`
  - `push!`

## Layer: 04_propagation

### Rust Functions

#### `enqueue_sparse`

- **File:** MMSB/src/04_propagation/sparse_message_passing.rs:0
- **Visibility:** Public
- **Calls:**
  - `push`

### Julia Functions

#### `_aggregate_children`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:213
- **Signature:** `_aggregate_children(state::MMSBState, parents::AbstractVector{PageID})`
- **Calls:**
  - `get!`
  - `get_children`
  - `push!`

#### `_apply_edges!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:234
- **Signature:** `_apply_edges!(state::MMSBState, page_id::PageID, edges::Set{EdgeType}, mode::PropagationMode)`
- **Calls:**
  - `_handle_data_dependency!`
  - `delete!`
  - `invalidate_compilation!`
  - `mark_page_stale!`
  - `schedule_gpu_sync!`

#### `_buffer`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:146
- **Signature:** `_buffer(state::MMSBState)::PropagationQueue`
- **Calls:**
  - `PropagationQueue`
  - `get!`

#### `_collect_descendants`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:260
- **Signature:** `_collect_descendants(state::MMSBState, page_id::PageID)::Set{PageID}`
- **Calls:**
  - `get_children`
  - `isempty`
  - `popfirst!`
  - `push!`

#### `_execute_command_buffer!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:226
- **Signature:** `_execute_command_buffer!(state::MMSBState, commands::Dict{PageID, Set{EdgeType}}, mode::PropagationMode)`
- **Calls:**
  - `_apply_edges!`

#### `batch_route_deltas!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:111
- **Signature:** `batch_route_deltas!(state::MMSBState, deltas::Vector{DeltaType})`
- **Calls:**
  - `haskey`
  - `isempty`
  - `push!`
  - `route_delta!`

#### `begin_transaction`

- **File:** MMSB/src/04_propagation/TransactionIsolation.jl:11
- **Signature:** `begin_transaction(s)`
- **Calls:**
  - `Transaction`
  - `rand`

#### `commit_transaction`

- **File:** MMSB/src/04_propagation/TransactionIsolation.jl:12
- **Signature:** `commit_transaction(s)`
- **Calls:**
  - `delete!`
  - `haskey`

#### `disable_graph_capture`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:69
- **Signature:** `disable_graph_capture(state::MMSBState)`
- **Calls:**
  - `ccall`
  - `haskey`

#### `enable_graph_capture`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:57
- **Signature:** `enable_graph_capture(state::MMSBState)`
- **Calls:**
  - `CUDAGraphState`
  - `get!`

#### `execute_propagation!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:296
- **Signature:** `execute_propagation!(state::MMSBState)`
- **Calls:**
  - `_buffer`
  - `delete!`
  - `isempty`
  - `popfirst!`
  - `recompute_page!`

## Layer: 05_adaptive

### Julia Functions

#### `analyze_locality`

- **File:** MMSB/src/05_adaptive/LocalityAnalysis.jl:38
- **Signature:** `analyze_locality(trace::AccessTrace, window_size::Int`

#### `can_reorder`

- **File:** MMSB/src/05_adaptive/GraphRewriting.jl:71
- **Signature:** `can_reorder(e1::Tuple{Int, Int}, e2::Tuple{Int, Int})`

#### `compute_edge_cost`

- **File:** MMSB/src/05_adaptive/GraphRewriting.jl:84
- **Signature:** `compute_edge_cost(edge::Tuple{Int, Int}, frequency_map::Dict{Tuple{Int, Int}, Int})`
- **Calls:**
  - `Float64`
  - `get`

#### `compute_entropy`

- **File:** MMSB/src/05_adaptive/EntropyReduction.jl:19
- **Signature:** `compute_entropy(access_frequencies::Dict{UInt64, Int})`
- **Calls:**
  - `isempty`
  - `log2`
  - `sum`
  - `values`

#### `compute_locality_score`

- **File:** MMSB/src/05_adaptive/AdaptiveLayout.jl:72
- **Signature:** `compute_locality_score(state::LayoutState, access_pattern::Dict{Tuple{PageId, PageId}, Int})`
- **Calls:**
  - `Int`
  - `abs`
  - `get`
  - `isnothing`

#### `compute_reuse_distance`

- **File:** MMSB/src/05_adaptive/LocalityAnalysis.jl:77
- **Signature:** `compute_reuse_distance(trace::AccessTrace, page_id::UInt64, current_idx::Int)`
- **Calls:**
  - `in`
  - `typemax`

#### `entropy_gradient`

- **File:** MMSB/src/05_adaptive/EntropyReduction.jl:63
- **Signature:** `entropy_gradient(access_pattern::Dict{UInt64, Int})`
- **Calls:**
  - `log2`
  - `sum`
  - `values`

## Layer: 06_utility

### Rust Functions

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

### Julia Functions

#### `_dfs_depth`

- **File:** MMSB/src/06_utility/Monitoring.jl:59
- **Signature:** `_dfs_depth(graph, node::PageID, depth::Int, visited::Set{PageID})`
- **Calls:**
  - `_dfs_depth`
  - `delete!`
  - `get`
  - `isempty`
  - `maximum`
  - `push!`

#### `aggregate_costs`

- **File:** MMSB/src/06_utility/CostAggregation.jl:27
- **Signature:** `aggregate_costs(costs::Vector{WeightedCost})`
- **Calls:**
  - `isempty`
  - `sum`

#### `compute_cache_cost`

- **File:** MMSB/src/06_utility/cost_functions.jl:30
- **Signature:** `compute_cache_cost(cache_misses::Int, cache_hits::Int)`

#### `compute_entropy`

- **File:** MMSB/src/06_utility/entropy_measure.jl:32
- **Signature:** `compute_entropy(dist::PageDistribution)`
- **Calls:**
  - `log2`
  - `values`

#### `compute_graph_depth`

- **File:** MMSB/src/06_utility/Monitoring.jl:50
- **Signature:** `compute_graph_depth(graph)`
- **Calls:**
  - `_dfs_depth`
  - `max`

#### `compute_latency_cost`

- **File:** MMSB/src/06_utility/cost_functions.jl:59
- **Signature:** `compute_latency_cost(total_latency_us::Int, num_ops::Int)`

#### `compute_memory_cost`

- **File:** MMSB/src/06_utility/cost_functions.jl:45
- **Signature:** `compute_memory_cost(bytes_allocated::Int, num_allocations::Int)`

#### `compute_utility`

- **File:** MMSB/src/06_utility/utility_engine.jl:44
- **Signature:** `compute_utility(costs::CostComponents, weights::Dict{Symbol, Float64})`

#### `entropy_reduction`

- **File:** MMSB/src/06_utility/entropy_measure.jl:71
- **Signature:** `entropy_reduction(old_entropy::Float64, new_entropy::Float64)`
- **Calls:**
  - `return`

#### `evict_lru_pages`

- **File:** MMSB/src/06_utility/MemoryPressure.jl:13
- **Signature:** `evict_lru_pages(s,n)`
- **Calls:**
  - `LRUTracker`
  - `collect`
  - `delete!`
  - `get!`
  - `length`
  - `min`
  - `push!`
  - `sort`

#### `exponential_backoff`

- **File:** MMSB/src/06_utility/ErrorRecovery.jl:12
- **Signature:** `exponential_backoff(a,p)`
- **Calls:**
  - `min`
  - `round`

#### `from_telemetry`

- **File:** MMSB/src/06_utility/cost_functions.jl:73
- **Signature:** `from_telemetry(snapshot)`
- **Calls:**
  - `CostComponents`
  - `Float64`
  - `Int`
  - `compute_cache_cost`
  - `compute_latency_cost`
  - `compute_memory_cost`

## Layer: 07_intention

### Julia Functions

#### `apply_preferences`

- **File:** MMSB/src/07_intention/structural_preferences.jl:40
- **Signature:** `apply_preferences(prefs::Vector{Preference}, state)`
- **Calls:**
  - `evaluate_preference`
  - `sum`

#### `compute_gradient`

- **File:** MMSB/src/07_intention/attractor_states.jl:25
- **Signature:** `compute_gradient(field::AttractorField, state::Vector{Float64})`
- **Calls:**
  - `length`
  - `sqrt`
  - `sum`
  - `zeros`
  - `zip`

#### `detect_goals`

- **File:** MMSB/src/07_intention/goal_emergence.jl:38
- **Signature:** `detect_goals(utility_state, threshold::Float64)`
- **Calls:**
  - `Dict`
  - `Goal`
  - `UInt64`
  - `abs`
  - `push!`
  - `utility_gradient`

#### `evaluate_intention`

- **File:** MMSB/src/07_intention/intention_engine.jl:50
- **Signature:** `evaluate_intention(intention::Intention, current_utility::Float64)`
- **Calls:**
  - `Float64`
  - `length`

#### `evaluate_preference`

- **File:** MMSB/src/07_intention/structural_preferences.jl:26
- **Signature:** `evaluate_preference(pref::Preference, state)`
- **Calls:**
  - `pref.constraint`

#### `evolve_state`

- **File:** MMSB/src/07_intention/attractor_states.jl:44
- **Signature:** `evolve_state(field::AttractorField, state::Vector{Float64}, dt::Float64)`
- **Calls:**
  - `compute_gradient`

#### `execute_upsert_plan!`

- **File:** MMSB/src/07_intention/intent_lowering.jl:29
- **Signature:** `execute_upsert_plan!(state::MMSBState, plan::UpsertPlan; source::Symbol`

#### `find_nearest_attractor`

- **File:** MMSB/src/07_intention/attractor_states.jl:54
- **Signature:** `find_nearest_attractor(field::AttractorField, state::Vector{Float64})`
- **Calls:**
  - `argmin`
  - `sqrt`
  - `sum`

#### `form_intention`

- **File:** MMSB/src/07_intention/intention_engine.jl:18
- **Signature:** `form_intention(utility_state, layout_state, id::UInt64)`
- **Calls:**
  - `UtilityEngine.utility_trend`

## Layer: 08_reasoning

### Julia Functions

#### `abduce`

- **File:** MMSB/src/08_reasoning/logic_engine.jl:43
- **Signature:** `abduce(observation::Constraint, rules::Vector{Rule})`
- **Calls:**
  - `Constraint`
  - `Dict`
  - `push!`

#### `analyze_edge_type`

- **File:** MMSB/src/08_reasoning/dependency_inference.jl:44
- **Signature:** `analyze_edge_type(dag, source::UInt64, target::UInt64)`
- **Calls:**
  - `haskey`

#### `analyze_flow`

- **File:** MMSB/src/08_reasoning/dependency_inference.jl:112
- **Signature:** `analyze_flow(dag, state::ReasoningState)`
- **Calls:**
  - `compute_critical_path`
  - `get`
  - `isempty`
  - `push!`
  - `sinks`
  - `sources`

#### `apply_rule`

- **File:** MMSB/src/08_reasoning/rule_evaluation.jl:47
- **Signature:** `apply_rule(rule::Rule, dag, node_id::UInt64, constraints::Vector{Constraint})`
- **Calls:**
  - `rule.action`
  - `rule.condition`

#### `backward_propagate`

- **File:** MMSB/src/08_reasoning/constraint_propagation.jl:94
- **Signature:** `backward_propagate(dag, state::ReasoningState, node_id::UInt64)`
- **Calls:**
  - `Constraint`
  - `Dict`
  - `get`
  - `get!`
  - `push!`

#### `check_consistency`

- **File:** MMSB/src/08_reasoning/structural_inference.jl:76
- **Signature:** `check_consistency(constraints::Vector{Constraint})`
- **Calls:**
  - `in`
  - `length`

#### `compute_critical_path`

- **File:** MMSB/src/08_reasoning/dependency_inference.jl:139
- **Signature:** `compute_critical_path(dag)`

#### `compute_dependency_strength`

- **File:** MMSB/src/08_reasoning/dependency_inference.jl:70
- **Signature:** `compute_dependency_strength(dag, source::UInt64, target::UInt64)`
- **Calls:**
  - `count_paths`
  - `get`
  - `min`

#### `count_paths`

- **File:** MMSB/src/08_reasoning/dependency_inference.jl:88
- **Signature:** `count_paths(dag, source::UInt64, target::UInt64, max_depth::Int)`
- **Calls:**
  - `count_paths`
  - `get`

#### `create_default_rules`

- **File:** MMSB/src/08_reasoning/rule_evaluation.jl:59
- **Signature:** `create_default_rules()`
- **Calls:**
  - `Constraint`
  - `Dict`
  - `Rule`
  - `UInt64`
  - `all`
  - `any`
  - `findfirst`
  - `get`
  - `length`
  - `push!`

#### `deduce`

- **File:** MMSB/src/08_reasoning/logic_engine.jl:17
- **Signature:** `deduce(premises::Vector{Constraint}, rules::Vector{Rule})`
- **Calls:**
  - `Constraint`
  - `Dict`
  - `all`
  - `push!`
  - `sort`

#### `derive_constraints`

- **File:** MMSB/src/08_reasoning/structural_inference.jl:58
- **Signature:** `derive_constraints(dag, state::ReasoningState)`
- **Calls:**
  - `infer_from_structure`
  - `isempty`
  - `keys`

#### `evaluate_rules`

- **File:** MMSB/src/08_reasoning/rule_evaluation.jl:17
- **Signature:** `evaluate_rules(dag, state::ReasoningState, node_id::UInt64)`
- **Calls:**
  - `Inference`
  - `get`
  - `isempty`
  - `push!`
  - `rule.action`
  - `rule.condition`

#### `extract_edges`

- **File:** MMSB/src/08_reasoning/pattern_formation.jl:127
- **Signature:** `extract_edges(dag, nodes::Vector{UInt64})`
- **Calls:**
  - `Set`
  - `get`
  - `push!`

#### `extract_subgraph_signature`

- **File:** MMSB/src/08_reasoning/pattern_formation.jl:97
- **Signature:** `extract_subgraph_signature(dag, nodes::Vector{UInt64})`
- **Calls:**
  - `Set`
  - `UInt8`
  - `get`
  - `length`
  - `min`
  - `push!`
  - `sort`

#### `extract_subgraphs`

- **File:** MMSB/src/08_reasoning/pattern_formation.jl:58
- **Signature:** `extract_subgraphs(dag, size::Int)`
- **Calls:**
  - `grow_subgraph`
  - `keys`
  - `length`
  - `push!`

#### `find_patterns`

- **File:** MMSB/src/08_reasoning/pattern_formation.jl:17
- **Signature:** `find_patterns(dag, min_frequency::Int`

#### `forward_propagate`

- **File:** MMSB/src/08_reasoning/constraint_propagation.jl:61
- **Signature:** `forward_propagate(dag, state::ReasoningState, start_nodes::Vector{UInt64})`
- **Calls:**
  - `InferenceResult`
  - `copy`
  - `isempty`
  - `pop!`
  - `propagate_constraints`
  - `push!`
  - `union!`

## Layer: 09_planning

### Julia Functions

#### `actions_from_params`

- **File:** MMSB/src/09_planning/optimization_planning.jl:138
- **Signature:** `actions_from_params(params::Matrix{Float64})`
- **Calls:**
  - `Action`
  - `Plan`
  - `UInt64`
  - `abs`
  - `abs.`
  - `push!`
  - `size`
  - `sum`

#### `adapt_strategy`

- **File:** MMSB/src/09_planning/strategy_generation.jl:108
- **Signature:** `adapt_strategy(strategy::Strategy, feedback::Dict{Symbol, Any})`
- **Calls:**
  - `Strategy`
  - `get`
  - `strategy.evaluation_fn`

#### `apply_action`

- **File:** MMSB/src/09_planning/search_algorithms.jl:191
- **Signature:** `apply_action(action::Action, state::State)`
- **Calls:**
  - `State`
  - `copy`
  - `effect`
  - `isa`

#### `apply_action_simple`

- **File:** MMSB/src/09_planning/decision_graphs.jl:52
- **Signature:** `apply_action_simple(action::Action, state::State)`
- **Calls:**
  - `State`
  - `UInt64`
  - `copy`
  - `rand`

#### `astar_search`

- **File:** MMSB/src/09_planning/search_algorithms.jl:17
- **Signature:** `astar_search(start_state::State, goal::Goal, actions::Vector{Action}, max_nodes::Int`

#### `backpropagate`

- **File:** MMSB/src/09_planning/search_algorithms.jl:165
- **Signature:** `backpropagate(node::MCTSNode, reward::Float64)`

#### `best_uct_child`

- **File:** MMSB/src/09_planning/search_algorithms.jl:107
- **Signature:** `best_uct_child(node::MCTSNode, c::Float64`

#### `build_decision_graph`

- **File:** MMSB/src/09_planning/decision_graphs.jl:17
- **Signature:** `build_decision_graph(state::State, goal::Goal, actions::Vector{Action}, depth::Int)`
- **Calls:**
  - `DecisionGraph`
  - `expand_graph!`

#### `can_apply`

- **File:** MMSB/src/09_planning/search_algorithms.jl:187
- **Signature:** `can_apply(action::Action, state::State)`
- **Calls:**
  - `all`
  - `pred`

#### `compute_gradient`

- **File:** MMSB/src/09_planning/optimization_planning.jl:44
- **Signature:** `compute_gradient(f::Function, x::Vector{Float64})`
- **Calls:**
  - `copy`
  - `differences`
  - `f`
  - `length`
  - `zeros`

#### `compute_heuristic`

- **File:** MMSB/src/09_planning/search_algorithms.jl:179
- **Signature:** `compute_heuristic(state::State, goal::Goal)`
- **Calls:**
  - `abs`
  - `haskey`

#### `compute_sequence_gradient`

- **File:** MMSB/src/09_planning/optimization_planning.jl:120
- **Signature:** `compute_sequence_gradient(params::Matrix{Float64}, state::State, goal::Goal)`
- **Calls:**
  - `copy`
  - `evaluate_action_sequence`
  - `size`
  - `zeros`

#### `create_plan`

- **File:** MMSB/src/09_planning/planning_engine.jl:24
- **Signature:** `create_plan(goal::Goal, state::State, actions::Vector{Action})`
- **Calls:**
  - `OptimizationPlanning.optimize_plan`
  - `StrategyGeneration.generate_strategies`
  - `StrategyGeneration.select_strategy`
  - `strategy.plan_generator`
  - `sum`

#### `create_subgoal_hierarchy`

- **File:** MMSB/src/09_planning/goal_decomposition.jl:59
- **Signature:** `create_subgoal_hierarchy(goals::Vector{Goal})`
- **Calls:**
  - `isempty`

#### `decompose_goal`

- **File:** MMSB/src/09_planning/goal_decomposition.jl:17
- **Signature:** `decompose_goal(goal::Goal, state::State)`
- **Calls:**
  - `Goal`
  - `UInt64`
  - `comp`
  - `enumerate`
  - `haskey`
  - `push!`

#### `estimate_achievability`

- **File:** MMSB/src/09_planning/goal_decomposition.jl:89
- **Signature:** `estimate_achievability(goal::Goal, state::State)`
- **Calls:**
  - `goal.predicate`

#### `evaluate_action_sequence`

- **File:** MMSB/src/09_planning/optimization_planning.jl:107
- **Signature:** `evaluate_action_sequence(params::Matrix{Float64}, state::State, goal::Goal)`
- **Calls:**
  - `State`
  - `abs`
  - `goal.predicate`
  - `size`

#### `evaluate_outcome`

- **File:** MMSB/src/09_planning/rollout_simulation.jl:63
- **Signature:** `evaluate_outcome(result::RolloutResult)`
- **Calls:**
  - `length`

#### `evaluate_policy`

- **File:** MMSB/src/09_planning/rl_planning.jl:116
- **Signature:** `evaluate_policy(π::Dict{UInt64, Action}, states::Vector{State}, actions::Vector{Action}, γ::Float64, V::Dict{UInt64, Float64})`
- **Calls:**
  - `expected_next_value`
  - `immediate_reward`

#### `execute_planning`

- **File:** MMSB/src/09_planning/planning_engine.jl:49
- **Signature:** `execute_planning(planning_state::PlanningState, goal_id::UInt64)`
- **Calls:**
  - `Goal`
  - `GoalDecomposition.decompose_goal`
  - `RolloutSimulation.simulate_plan`
  - `SearchAlgorithms.mcts_search`
  - `create_plan`
  - `length`

#### `expand_graph!`

- **File:** MMSB/src/09_planning/decision_graphs.jl:31
- **Signature:** `expand_graph!(graph::DecisionGraph, state::State, goal::Goal, actions::Vector{Action}, depth::Int)`
- **Calls:**
  - `all`
  - `apply_action_simple`
  - `expand_graph!`
  - `goal.predicate`
  - `haskey`
  - `pred`

#### `expand_node`

- **File:** MMSB/src/09_planning/search_algorithms.jl:129
- **Signature:** `expand_node(node::MCTSNode, actions::Vector{Action})`
- **Calls:**
  - `MCTSNode`
  - `apply_action`
  - `can_apply`
  - `push!`

#### `expected_next_value`

- **File:** MMSB/src/09_planning/rl_planning.jl:54
- **Signature:** `expected_next_value(s::State, a::Action, V::Dict{UInt64, Float64}, states::Vector{State})`
- **Calls:**
  - `get`

#### `extract_parameters`

- **File:** MMSB/src/09_planning/optimization_planning.jl:35
- **Signature:** `extract_parameters(plan::Plan)`
- **Calls:**
  - `parameters`
  - `push!`

#### `extract_plan_from_mcts`

- **File:** MMSB/src/09_planning/search_algorithms.jl:224
- **Signature:** `extract_plan_from_mcts(root::MCTSNode, goal_id::UInt64)`
- **Calls:**
  - `Plan`
  - `UInt64`
  - `best_uct_child`
  - `isempty`
  - `push!`

#### `find_optimal_path`

- **File:** MMSB/src/09_planning/decision_graphs.jl:63
- **Signature:** `find_optimal_path(graph::DecisionGraph, start_id::UInt64, goal::Goal)`
- **Calls:**
  - `goal.predicate`
  - `haskey`
  - `keys`
  - `length`
  - `pushfirst!`

## Layer: 10_agent_interface

### Julia Functions

#### `Core.Compiler.InferenceParams`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:52
- **Signature:** `Core.Compiler.InferenceParams(interp::MMSBInterpreter)`

#### `Core.Compiler.OptimizationParams`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:56
- **Signature:** `Core.Compiler.OptimizationParams(interp::MMSBInterpreter)`

#### `Core.Compiler.abstract_call_method`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:117
- **Signature:** `Core.Compiler.abstract_call_method( interp::MMSBInterpreter, method::Method, sig, sparams::Core.SimpleVector, hardlimit::Bool, si::Core.Compiler.StmtInfo )`
- **Calls:**
  - `invoke`
  - `log_method_call!`
  - `update_call_graph!`

#### `Core.Compiler.code_cache`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:68
- **Signature:** `Core.Compiler.code_cache(interp::MMSBInterpreter)`

#### `Core.Compiler.get_inference_cache`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:64
- **Signature:** `Core.Compiler.get_inference_cache(interp::MMSBInterpreter)`

#### `Core.Compiler.get_world_counter`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:60
- **Signature:** `Core.Compiler.get_world_counter(interp::MMSBInterpreter)`

#### `Core.Compiler.optimize`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:159
- **Signature:** `Core.Compiler.optimize( interp::MMSBInterpreter, opt::Core.Compiler.OptimizationState, params::OptimizationParams, result::InferenceResult )`
- **Calls:**
  - `copy`
  - `create_optimization_delta!`
  - `invoke`

#### `Core.Compiler.typeinf`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:84
- **Signature:** `Core.Compiler.typeinf(interp::MMSBInterpreter, frame::InferenceState)`
- **Calls:**
  - `create_inference_pages!`
  - `invoke`
  - `log_inference_result!`
  - `log_inference_start!`

#### `configure_instrumentation!`

- **File:** MMSB/src/10_agent_interface/InstrumentationManager.jl:100
- **Signature:** `configure_instrumentation!(state::MMSBState, config::InstrumentationConfig)`
- **Calls:**
  - `disable_instrumentation!`
  - `enable_instrumentation!`

#### `create_checkpoint`

- **File:** MMSB/src/10_agent_interface/checkpoint_api.jl:11
- **Signature:** `create_checkpoint(state::MMSBState, name::String)::String`
- **Calls:**
  - `checkpoint_log!`
  - `time_ns`

#### `create_inference_pages!`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:235
- **Signature:** `create_inference_pages!(state::MMSBState, frame::InferenceState)`
- **Calls:**
  - `copyto!`
  - `create_page!`
  - `length`
  - `serialize_codeinfo`

#### `disable_base_hooks!`

- **File:** MMSB/src/10_agent_interface/BaseHook.jl:74
- **Signature:** `disable_base_hooks!()`
- **Calls:**
  - `empty!`

#### `disable_compiler_hooks!`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:273
- **Signature:** `disable_compiler_hooks!()`

#### `disable_core_hooks!`

- **File:** MMSB/src/10_agent_interface/CoreHooks.jl:49
- **Signature:** `disable_core_hooks!()`

#### `disable_instrumentation!`

- **File:** MMSB/src/10_agent_interface/InstrumentationManager.jl:77
- **Signature:** `disable_instrumentation!(state::MMSBState)`
- **Calls:**
  - `BaseHooks.disable_base_hooks!`
  - `CompilerHooks.disable_compiler_hooks!`
  - `CoreHooks.disable_core_hooks!`

#### `emit_event`

- **File:** MMSB/src/10_agent_interface/event_subscription.jl:40
- **Signature:** `emit_event(event_type::EventType, data::Any)`
- **Calls:**
  - `sub.callback`
  - `values`

#### `enable_base_hooks!`

- **File:** MMSB/src/10_agent_interface/BaseHook.jl:37
- **Signature:** `enable_base_hooks!(state::MMSBState)`
- **Calls:**
  - `MMSB.hook_invoke`
  - `invoke`

#### `enable_compiler_hooks!`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:259
- **Signature:** `enable_compiler_hooks!(state::MMSBState)`

#### `enable_core_hooks!`

- **File:** MMSB/src/10_agent_interface/CoreHooks.jl:32
- **Signature:** `enable_core_hooks!(state::MMSBState)`
- **Calls:**
  - `enabled`

#### `enable_instrumentation!`

- **File:** MMSB/src/10_agent_interface/InstrumentationManager.jl:51
- **Signature:** `enable_instrumentation!(state::MMSBState, config::InstrumentationConfig)`
- **Calls:**
  - `BaseHooks.enable_base_hooks!`
  - `CompilerHooks.enable_compiler_hooks!`
  - `CoreHooks.enable_core_hooks!`

## Layer: 11_agents

### Julia Functions

#### `apply_rule`

- **File:** MMSB/src/11_agents/symbolic_agent.jl:37
- **Signature:** `apply_rule(agent::SymbolicAgent, rule::Rule, state::MMSBState)::Vector{AgentAction}`

#### `autodiff_loss`

- **File:** MMSB/src/11_agents/enzyme_integration.jl:18
- **Signature:** `autodiff_loss(f::Function, x::Vector{Float64})::Tuple{Float64, Vector{Float64}}`
- **Calls:**
  - `f`
  - `length`
  - `return`
  - `zeros`

#### `compute_reward`

- **File:** MMSB/src/11_agents/rl_agent.jl:32
- **Signature:** `compute_reward(agent::RLAgent, state::MMSBState, action::AgentAction)::Float64`

#### `create_policy_network`

- **File:** MMSB/src/11_agents/lux_models.jl:17
- **Signature:** `create_policy_network(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int})`

#### `create_value_network`

- **File:** MMSB/src/11_agents/lux_models.jl:11
- **Signature:** `create_value_network(input_dim::Int, hidden_dims::Vector{Int})`

#### `execute_plan_step`

- **File:** MMSB/src/11_agents/planning_agent.jl:36
- **Signature:** `execute_plan_step(agent::PlanningAgent)::Union{AgentAction, Nothing}`
- **Calls:**
  - `isempty`
  - `popfirst!`

## Layer: 12_applications

### Julia Functions

#### `add_entity!`

- **File:** MMSB/src/12_applications/world_simulation.jl:28
- **Signature:** `add_entity!(world::World, entity_type::Symbol, props::Dict{Symbol, Any})::Entity`
- **Calls:**
  - `Entity`
  - `allocate_page_id!`

#### `compute_value`

- **File:** MMSB/src/12_applications/financial_modeling.jl:25
- **Signature:** `compute_value(portfolio::Portfolio, prices::Dict{String, Float64})::Float64`
- **Calls:**
  - `get`

#### `coordinate_step!`

- **File:** MMSB/src/12_applications/multi_agent_system.jl:23
- **Signature:** `coordinate_step!(coord::AgentCoordinator)`
- **Calls:**
  - `observe`

## Layer: root

### Rust Functions

#### `convert_location`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `PageLocation::from_tag`

#### `dense_delta`

- **File:** MMSB/tests/delta_validation.rs:0
- **Visibility:** Private
- **Calls:**
  - `DeltaID`
  - `PageID`
  - `Epoch`
  - `Source`
  - `into`

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

### Julia Functions

#### `Base.showerror`

- **File:** MMSB/src/ffi/RustErrors.jl:31
- **Signature:** `Base.showerror(io::IO, err::RustFFIError)`
- **Calls:**
  - `get`
  - `print`

#### `_build_batch_deltas`

- **File:** MMSB/benchmark/benchmarks.jl:131
- **Signature:** `_build_batch_deltas(state, page_id::PageTypes.PageID, batch_size::Int)`
- **Calls:**
  - `DeltaRouter.create_delta`
  - `fill`
  - `push!`
  - `rand`

#### `_check_rust_error`

- **File:** MMSB/src/ffi/FFIWrapper.jl:14
- **Signature:** `_check_rust_error(context::AbstractString)`

#### `_checkpoint`

- **File:** MMSB/benchmark/benchmarks.jl:89
- **Signature:** `_checkpoint(state)`
- **Calls:**
  - `TLog.checkpoint_log!`
  - `tempname`

#### `_collect_instrumentation_report`

- **File:** MMSB/benchmark/benchmarks.jl:435
- **Signature:** `_collect_instrumentation_report()`
- **Calls:**
  - `API.update_page`
  - `GraphTypes.add_dependency!`
  - `Monitoring.get_stats`
  - `Monitoring.reset_stats!`
  - `PropagationEngine.register_passthrough_recompute!`
  - `Semiring.boolean_fold_add`
  - `Semiring.tropical_fold_add`
  - `_measure_ns`
  - `_page`
  - `_start_state`
  - `_stop_state!`
  - `rand`

#### `_default_message`

- **File:** MMSB/src/ffi/RustErrors.jl:42
- **Signature:** `_default_message(err::RustFFIError)`
- **Calls:**
  - `error`
  - `get`

#### `_format_bytes`

- **File:** MMSB/benchmark/helpers.jl:18
- **Signature:** `_format_bytes(bytes::Int)`
- **Calls:**
  - `round`

#### `_format_time`

- **File:** MMSB/benchmark/helpers.jl:6
- **Signature:** `_format_time(ns::Float64)`
- **Calls:**
  - `round`

#### `_full_system_benchmark!`

- **File:** MMSB/benchmark/benchmarks.jl:141
- **Signature:** `_full_system_benchmark!()`
- **Calls:**
  - `API.update_page`
  - `GraphTypes.add_dependency!`
  - `MMSB.MMSBStateTypes.MMSBState`
  - `PropagationEngine.register_passthrough_recompute!`
  - `ReplayEngine.replay_to_epoch`
  - `TLog.load_checkpoint!`
  - `UInt32`
  - `_checkpoint`
  - `_populate_pages!`
  - `_start_state`
  - `_stop_state!`
  - `isfile`
  - `length`
  - `rand`
  - `rm`

#### `analyze_results`

- **File:** MMSB/benchmark/helpers.jl:30
- **Signature:** `analyze_results(results)`
- **Calls:**
  - `_format_bytes`
  - `_format_time`
  - `mean`
  - `median`
  - `println`
  - `std`

#### `check_performance_targets`

- **File:** MMSB/benchmark/helpers.jl:51
- **Signature:** `check_performance_targets(results)`
- **Calls:**
  - `Dict`
  - `_format_time`
  - `haskey`
  - `median`
  - `println`
  - `round`

#### `check_rust_error`

- **File:** MMSB/src/ffi/RustErrors.jl:36
- **Signature:** `check_rust_error(context::AbstractString)`
- **Calls:**
  - `RustFFIError`
  - `String`
  - `rust_get_last_error`
  - `throw`

#### `compare_with_baseline`

- **File:** MMSB/benchmark/benchmarks.jl:518
- **Signature:** `compare_with_baseline(current_results)`
- **Calls:**
  - `JSON3.read`
  - `haskey`
  - `isfile`
  - `joinpath`
  - `median`
  - `println`
  - `read`
  - `round`
  - `string`

#### `create_page`

- **File:** MMSB/src/API.jl:68
- **Signature:** `create_page(state::MMSBState; size::Integer, location::Symbol`

#### `ensure_rust_artifacts`

- **File:** MMSB/src/ffi/FFIWrapper.jl:70
- **Signature:** `ensure_rust_artifacts()`
- **Calls:**
  - `error`
  - `rust_artifacts_available`

