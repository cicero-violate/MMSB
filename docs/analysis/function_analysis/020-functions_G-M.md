# Functions G-M

## Layer: 00_physical

### Julia Functions

#### `GPUMemoryPool`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:39
- **Signature:** `GPUMemoryPool()::GPUMemoryPool`
- **Calls:**
  - `GPUMemoryPool`
  - `ccall`

#### `get_pool_stats`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:76
- **Signature:** `get_pool_stats(pool::GPUMemoryPool)::Dict{Symbol, UInt64}`
- **Calls:**
  - `ccall`

#### `get_sync_statistics`

- **File:** MMSB/src/00_physical/DeviceSync.jl:309
- **Signature:** `get_sync_statistics(state::MMSBState)::Dict{Symbol, Any}`

#### `has_gpu_support`

- **File:** MMSB/src/00_physical/DeviceFallback.jl:5
- **Signature:** `has_gpu_support()`
- **Calls:**
  - `CUDA.functional`

#### `is_unified_memory_available`

- **File:** MMSB/src/00_physical/UnifiedMemory.jl:99
- **Signature:** `is_unified_memory_available()::Bool`
- **Calls:**
  - `CUDA.capability`
  - `CUDA.device`
  - `CUDA.functional`

#### `launch_delta_merge!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:62
- **Signature:** `launch_delta_merge!(base::CuArray{UInt8}, mask::CuArray{Bool}, delta::CuArray{UInt8})`
- **Calls:**
  - `compute_optimal_kernel_config`
  - `delta_merge_kernel!`
  - `length`

#### `launch_page_compare!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:160
- **Signature:** `launch_page_compare!(result::CuArray{Bool}, page1::CuArray{UInt8}, page2::CuArray{UInt8})`
- **Calls:**
  - `compute_optimal_kernel_config`
  - `length`
  - `page_compare_kernel!`

#### `launch_page_copy!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:100
- **Signature:** `launch_page_copy!(dest::CuArray{UInt8}, src::CuArray{UInt8}, n::Int)`
- **Calls:**
  - `compute_optimal_kernel_config`
  - `page_copy_kernel!`

#### `launch_page_zero!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:126
- **Signature:** `launch_page_zero!(data::CuArray{UInt8})`
- **Calls:**
  - `compute_optimal_kernel_config`
  - `length`
  - `page_zero_kernel!`

#### `launch_sparse_delta_apply!`

- **File:** MMSB/src/00_physical/GPUKernels.jl:206
- **Signature:** `launch_sparse_delta_apply!(base::CuArray{UInt8}, indices::CuArray{Int32}, values::CuArray{UInt8})`
- **Calls:**
  - `compute_optimal_kernel_config`
  - `length`
  - `sparse_delta_apply_kernel!`

#### `migrate_page!`

- **File:** MMSB/src/00_physical/PageAllocator.jl:72
- **Signature:** `migrate_page!(state::MMSBState, page_id::PageID, target_location::PageLocation)::Page`
- **Calls:**
  - `PageNotFoundError`
  - `UInt64`
  - `get`
  - `lock`
  - `throw`

## Layer: 01_page

### Rust Functions

#### `generate_mask`

- **File:** MMSB/src/01_page/simd_mask.rs:0
- **Visibility:** Public
- **Calls:**
  - `collect`
  - `map`
  - `zip`
  - `iter`
  - `iter`

#### `load_checkpoint`

- **File:** MMSB/src/01_page/checkpoint.rs:0
- **Visibility:** Public
- **Calls:**
  - `BufReader::new`
  - `File::open`
  - `read_exact`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `Err`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `Vec::with_capacity`
  - `read_exact`
  - `PageID`
  - `u64::from_le_bytes`
  - `read_exact`
  - `u64::from_le_bytes`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `i32::from_le_bytes`
  - `map_err`
  - `PageLocation::from_tag`
  - `std::io::Error::new`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `read_exact`
  - `u32::from_le_bytes`
  - `read_exact`
  - `push`
  - `restore_from_snapshot`
  - `Ok`
  - `Err`
  - `std::io::Error::new`

#### `merge_deltas`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Public
- **Calls:**
  - `merge`

#### `merge_dense_avx2`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Private
- **Calls:**
  - `min`
  - `len`
  - `len`
  - `_mm256_loadu_si256`
  - `add`
  - `as_ptr`
  - `_mm256_loadu_si256`
  - `add`
  - `as_ptr`
  - `_mm256_loadu_si256`
  - `as_ptr`
  - `_mm256_loadu_si256`
  - `as_ptr`
  - `_mm256_blendv_epi8`
  - `_mm256_storeu_si256`
  - `add`
  - `as_mut_ptr`
  - `_mm256_or_si256`
  - `_mm256_storeu_si256`
  - `as_mut_ptr`

#### `merge_dense_avx512`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Private
- **Calls:**
  - `min`
  - `len`
  - `len`
  - `_mm512_loadu_si512`
  - `add`
  - `as_ptr`
  - `_mm512_loadu_si512`
  - `add`
  - `as_ptr`
  - `_mm512_loadu_si512`
  - `as_ptr`
  - `_mm512_test_epi8_mask`
  - `_mm512_mask_blend_epi8`
  - `_mm512_storeu_si512`
  - `add`
  - `as_mut_ptr`

#### `merge_dense_simd`

- **File:** MMSB/src/01_page/delta_merge.rs:0
- **Visibility:** Public
- **Calls:**
  - `merge_dense_avx512`
  - `merge_dense_avx2`
  - `min`
  - `len`
  - `len`

### Julia Functions

#### `_iterate_log`

- **File:** MMSB/src/01_page/TLog.jl:56
- **Signature:** `_iterate_log(f::Function, path::AbstractString)`
- **Calls:**
  - `FFIWrapper.rust_tlog_reader_free!`
  - `FFIWrapper.rust_tlog_reader_new`
  - `f`

#### `get_deltas_for_page`

- **File:** MMSB/src/01_page/TLog.jl:95
- **Signature:** `get_deltas_for_page(state::MMSBState, pid::PageID)`
- **Calls:**
  - `query_log`

#### `get_deltas_in_range`

- **File:** MMSB/src/01_page/TLog.jl:97
- **Signature:** `get_deltas_in_range(state::MMSBState, start_idx::Int, end_idx::Int)`
- **Calls:**
  - `clamp`
  - `isempty`
  - `length`
  - `query_log`

#### `incremental_replay!`

- **File:** MMSB/src/01_page/ReplayEngine.jl:137
- **Signature:** `incremental_replay!(state::MMSBState, deltas::Vector{Delta})`
- **Calls:**
  - `_apply_delta!`
  - `get_page`

#### `initialize!`

- **File:** MMSB/src/01_page/Page.jl:46
- **Signature:** `initialize!(page::Page)`

#### `intent_metadata`

- **File:** MMSB/src/01_page/Delta.jl:130
- **Signature:** `intent_metadata(delta::Delta; parse::Bool`

#### `is_cpu_page`

- **File:** MMSB/src/01_page/Page.jl:43
- **Signature:** `is_cpu_page(page::Page)`

#### `is_gpu_page`

- **File:** MMSB/src/01_page/Page.jl:42
- **Signature:** `is_gpu_page(page::Page)`
- **Calls:**
  - `in`

#### `load_checkpoint!`

- **File:** MMSB/src/01_page/TLog.jl:128
- **Signature:** `load_checkpoint!(state::MMSBState, path::AbstractString)`
- **Calls:**
  - `FFIWrapper.rust_checkpoint_load!`
  - `_with_rust_errors`

#### `log_summary`

- **File:** MMSB/src/01_page/TLog.jl:48
- **Signature:** `log_summary(state::MMSBState)`
- **Calls:**
  - `FFIWrapper.rust_tlog_summary`
  - `_with_rust_errors`

#### `merge_deltas_simd!`

- **File:** MMSB/src/01_page/Delta.jl:194
- **Signature:** `merge_deltas_simd!( data_a::Vector{UInt8}, mask_a::Vector{Bool}, data_b::Vector{UInt8}, mask_b::Vector{Bool}, out_data::Vector{UInt8}, out_mask::Vector{Bool} )`
- **Calls:**
  - `ccall`
  - `length`
  - `min`

#### `metadata_from_blob`

- **File:** MMSB/src/01_page/Page.jl:117
- **Signature:** `metadata_from_blob(blob::Vector{UInt8})`
- **Calls:**
  - `_decode_metadata_blob`

## Layer: 01_types

### Julia Functions

#### `MMSBConfig`

- **File:** MMSB/src/01_types/MMSBState.jl:34
- **Signature:** `MMSBConfig(; enable_logging`
- **Calls:**
  - `joinpath`
  - `pwd`

#### `MMSBState`

- **File:** MMSB/src/01_types/MMSBState.jl:61
- **Signature:** `MMSBState(config::MMSBConfig)`
- **Calls:**
  - `DeltaID`
  - `FFIWrapper.rust_allocator_free!`
  - `FFIWrapper.rust_allocator_new`
  - `FFIWrapper.rust_tlog_free!`
  - `FFIWrapper.rust_tlog_new`
  - `PageID`
  - `ReentrantLock`
  - `ShadowPageGraph`
  - `error`
  - `finalizer`
  - `new`

#### `MMSBState`

- **File:** MMSB/src/01_types/MMSBState.jl:83
- **Signature:** `MMSBState()`
- **Calls:**
  - `MMSBConfig`
  - `MMSBState`

#### `get_page`

- **File:** MMSB/src/01_types/MMSBState.jl:124
- **Signature:** `get_page(state::MMSBState, id::PageID)::Union{Page, Nothing}`
- **Calls:**
  - `get`
  - `lock`

## Layer: 02_semiring

### Julia Functions

#### `length`

- **File:** MMSB/src/02_semiring/DeltaRouter.jl:58
- **Signature:** `length(mask_bytes)`
- **Calls:**
  - `InvalidDeltaError`
  - `length`
  - `throw`

## Layer: 03_dag

### Rust Functions

#### `has_cycle`

- **File:** MMSB/src/03_dag/cycle_detection.rs:0
- **Visibility:** Public
- **Calls:**
  - `clone`
  - `read`
  - `HashMap::new`
  - `get`
  - `insert`
  - `get`
  - `dfs`
  - `insert`
  - `keys`
  - `dfs`

### Julia Functions

#### `get_children`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:106
- **Signature:** `get_children(graph::ShadowPageGraph, parent::PageID)::Vector{Tuple{PageID, EdgeType}}`
- **Calls:**
  - `get`
  - `lock`

#### `get_children`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:102
- **Signature:** `get_children(graph::ShadowPageGraph, parent::PageID)::Vector{Tuple{PageID, EdgeType}}`
- **Calls:**
  - `copy`
  - `get`
  - `lock`

#### `get_parents`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:120
- **Signature:** `get_parents(graph::ShadowPageGraph, child::PageID)::Vector{Tuple{PageID, EdgeType}}`
- **Calls:**
  - `get`
  - `lock`

#### `get_parents`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:113
- **Signature:** `get_parents(graph::ShadowPageGraph, child::PageID)::Vector{Tuple{PageID, EdgeType}}`
- **Calls:**
  - `copy`
  - `get`
  - `lock`

#### `get_subscription_count`

- **File:** MMSB/src/03_dag/EventSystem.jl:150
- **Signature:** `get_subscription_count(event_type::EventType)::Int`
- **Calls:**
  - `get`
  - `length`
  - `lock`

#### `has_cycle`

- **File:** MMSB/src/03_dag/ShadowPageGraph.jl:141
- **Signature:** `has_cycle(graph::ShadowPageGraph)::Bool`
- **Calls:**
  - `_dfs_has_cycle`
  - `keys`

#### `has_edge`

- **File:** MMSB/src/03_dag/DependencyGraph.jl:88
- **Signature:** `has_edge(graph::ShadowPageGraph, parent::PageID, child::PageID)::Bool`
- **Calls:**
  - `any`
  - `haskey`
  - `lock`

#### `log_event!`

- **File:** MMSB/src/03_dag/EventSystem.jl:129
- **Signature:** `log_event!(state::MMSBState, event_type::EventType, data)`

#### `log_event_to_page!`

- **File:** MMSB/src/03_dag/EventSystem.jl:202
- **Signature:** `log_event_to_page!(state::MMSBState, page_id::PageID, event_type::EventType, data)`
- **Calls:**
  - `_serialize_event`
  - `falses`
  - `get_page`
  - `length`
  - `min`
  - `read_page`

## Layer: 04_propagation

### Julia Functions

#### `_handle_data_dependency!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:247
- **Signature:** `_handle_data_dependency!(state::MMSBState, page_id::PageID, mode::PropagationMode)`
- **Calls:**
  - `emit_event!`
  - `queue_recomputation!`
  - `recompute_page!`

#### `invalidate_compilation!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:355
- **Signature:** `invalidate_compilation!(state::MMSBState, page_id::PageID)`
- **Calls:**
  - `emit_event!`
  - `get_page`

#### `mark_page_stale!`

- **File:** MMSB/src/04_propagation/PropagationEngine.jl:334
- **Signature:** `mark_page_stale!(state::MMSBState, page_id::PageID)`
- **Calls:**
  - `emit_event!`
  - `get_page`

## Layer: 05_adaptive

### Julia Functions

#### `LayoutState`

- **File:** MMSB/src/05_adaptive/AdaptiveLayout.jl:29
- **Signature:** `LayoutState(page_size::Int)`
- **Calls:**
  - `LayoutState`

## Layer: 06_utility

### Julia Functions

#### `LRUTracker`

- **File:** MMSB/src/06_utility/MemoryPressure.jl:9
- **Signature:** `LRUTracker()`
- **Calls:**
  - `LRUTracker`

#### `get_stats`

- **File:** MMSB/src/06_utility/Monitoring.jl:74
- **Signature:** `get_stats(state::MMSBState)::MMSBStats`
- **Calls:**
  - `FFIWrapper.rust_tlog_summary`
  - `Int64`
  - `MMSBStats`
  - `UInt64`
  - `compute_graph_depth`
  - `get`
  - `length`
  - `lock`
  - `sum`
  - `values`

#### `is_fatal_error`

- **File:** MMSB/src/06_utility/ErrorRecovery.jl:14
- **Signature:** `is_fatal_error(e)`
- **Calls:**
  - `ccall`

#### `is_retryable_error`

- **File:** MMSB/src/06_utility/ErrorRecovery.jl:13
- **Signature:** `is_retryable_error(e)`
- **Calls:**
  - `ccall`

## Layer: 07_intention

### Julia Functions

#### `IntentionState`

- **File:** MMSB/src/07_intention/IntentionTypes.jl:51
- **Signature:** `IntentionState()`
- **Calls:**
  - `IntentionState`
  - `time`

#### `lower_intent_to_deltaspec`

- **File:** MMSB/src/07_intention/intent_lowering.jl:21
- **Signature:** `lower_intent_to_deltaspec(plan::UpsertPlan)`
- **Calls:**
  - `mask_to_bytes`
  - `validate_plan`

#### `mask_to_bytes`

- **File:** MMSB/src/07_intention/intent_lowering.jl:17
- **Signature:** `mask_to_bytes(mask::Vector{Bool})`

## Layer: 08_reasoning

### Julia Functions

#### `grow_subgraph`

- **File:** MMSB/src/08_reasoning/pattern_formation.jl:71
- **Signature:** `grow_subgraph(dag, start::UInt64, size::Int)`
- **Calls:**
  - `get`
  - `isempty`
  - `length`
  - `popfirst!`
  - `push!`

#### `induce`

- **File:** MMSB/src/08_reasoning/logic_engine.jl:68
- **Signature:** `induce(examples::Vector{Tuple{Vector{Constraint}, Constraint}})`
- **Calls:**
  - `Rule`
  - `UInt64`
  - `all`
  - `get!`
  - `length`
  - `push!`
  - `unique`

#### `infer_dependencies`

- **File:** MMSB/src/08_reasoning/dependency_inference.jl:17
- **Signature:** `infer_dependencies(dag, node_id::UInt64)`
- **Calls:**
  - `Dependency`
  - `analyze_edge_type`
  - `compute_dependency_strength`
  - `get`
  - `push!`

#### `infer_from_structure`

- **File:** MMSB/src/08_reasoning/structural_inference.jl:17
- **Signature:** `infer_from_structure(dag, node_id::UInt64, state::ReasoningState)`
- **Calls:**
  - `Constraint`
  - `Dict`
  - `all`
  - `get`
  - `length`
  - `push!`

#### `initialize_reasoning`

- **File:** MMSB/src/08_reasoning/reasoning_engine.jl:23
- **Signature:** `initialize_reasoning(dag)`
- **Calls:**
  - `DependencyInference.infer_dependencies`
  - `PatternFormation.find_patterns`
  - `ReasoningState`
  - `RuleEvaluation.create_default_rules`
  - `keys`

#### `match_pattern`

- **File:** MMSB/src/08_reasoning/pattern_formation.jl:148
- **Signature:** `match_pattern(dag, pattern::Pattern, start_node::UInt64)`
- **Calls:**
  - `PatternMatch`
  - `extract_subgraph_signature`
  - `grow_subgraph`
  - `length`

## Layer: 09_planning

### Julia Functions

#### `MCTSNode`

- **File:** MMSB/src/09_planning/search_algorithms.jl:98
- **Signature:** `MCTSNode(state::State)`
- **Calls:**
  - `MCTSNode`

#### `generate_strategies`

- **File:** MMSB/src/09_planning/strategy_generation.jl:19
- **Signature:** `generate_strategies(goal::Goal, state::State)`
- **Calls:**
  - `SearchAlgorithms.astar_search`
  - `SearchAlgorithms.mcts_search`
  - `Strategy`
  - `UInt64`
  - `hierarchical_planning`
  - `push!`

#### `gradient_descent_planning`

- **File:** MMSB/src/09_planning/optimization_planning.jl:88
- **Signature:** `gradient_descent_planning(initial_state::State, goal::Goal, α::Float64`

#### `hierarchical_planning`

- **File:** MMSB/src/09_planning/strategy_generation.jl:49
- **Signature:** `hierarchical_planning(goal::Goal, state::State, actions::Vector{Action})`
- **Calls:**
  - `GoalDecomposition.decompose_goal`
  - `Plan`
  - `SearchAlgorithms.apply_action`
  - `SearchAlgorithms.astar_search`
  - `UInt64`
  - `append!`

#### `immediate_reward`

- **File:** MMSB/src/09_planning/rl_planning.jl:50
- **Signature:** `immediate_reward(s::State, a::Action)`

#### `is_terminal`

- **File:** MMSB/src/09_planning/search_algorithms.jl:205
- **Signature:** `is_terminal(node::MCTSNode, goal::Goal)`
- **Calls:**
  - `goal.predicate`

#### `mcts_search`

- **File:** MMSB/src/09_planning/search_algorithms.jl:66
- **Signature:** `mcts_search(start_state::State, goal::Goal, actions::Vector{Action}, iterations::Int`

## Layer: 10_agent_interface

### Julia Functions

#### `InstrumentationConfig`

- **File:** MMSB/src/10_agent_interface/InstrumentationManager.jl:33
- **Signature:** `InstrumentationConfig()`

#### `MMSBInterpreter`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:42
- **Signature:** `MMSBInterpreter(state::MMSBState; world::UInt`
- **Calls:**
  - `Base.get_world_counter`

#### `hook_codeinfo_creation`

- **File:** MMSB/src/10_agent_interface/CoreHooks.jl:72
- **Signature:** `hook_codeinfo_creation(mi::Core.MethodInstance, ci::Core.CodeInfo)`
- **Calls:**
  - `create_ir_page!`
  - `time_ns`

#### `hook_getfield`

- **File:** MMSB/src/10_agent_interface/BaseHook.jl:160
- **Signature:** `hook_getfield(obj, field::Symbol)`
- **Calls:**
  - `objectid`
  - `time_ns`
  - `typeof`

#### `hook_invoke`

- **File:** MMSB/src/10_agent_interface/BaseHook.jl:107
- **Signature:** `hook_invoke(f, types::Type, args, kwargs)`
- **Calls:**
  - `time_ns`

#### `hook_methodinstance`

- **File:** MMSB/src/10_agent_interface/CoreHooks.jl:103
- **Signature:** `hook_methodinstance(mi::Core.MethodInstance)`
- **Calls:**
  - `time_ns`

#### `hook_setfield!`

- **File:** MMSB/src/10_agent_interface/BaseHook.jl:135
- **Signature:** `hook_setfield!(obj, field::Symbol, value)`
- **Calls:**
  - `objectid`
  - `time_ns`
  - `typeof`

#### `list_checkpoints`

- **File:** MMSB/src/10_agent_interface/checkpoint_api.jl:21
- **Signature:** `list_checkpoints(state::MMSBState)::Vector{String}`

#### `log_inference_result!`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:212
- **Signature:** `log_inference_result!(state::MMSBState, frame::InferenceState, result)`
- **Calls:**
  - `time_ns`

#### `log_inference_start!`

- **File:** MMSB/src/10_agent_interface/CompilerHooks.jl:194
- **Signature:** `log_inference_start!(state::MMSBState, frame::InferenceState)`
- **Calls:**
  - `time_ns`

## Layer: 11_agents

### Julia Functions

#### `generate_plan`

- **File:** MMSB/src/11_agents/planning_agent.jl:30
- **Signature:** `generate_plan(agent::PlanningAgent, state::MMSBState, goal::Any)::Vector{AgentAction}`
- **Calls:**
  - `search_plan`

#### `gradient_descent_step!`

- **File:** MMSB/src/11_agents/enzyme_integration.jl:11
- **Signature:** `gradient_descent_step!(params::Vector{Float64}, loss_fn::Function, lr::Float64)`
- **Calls:**
  - `Enzyme.gradient`

#### `infer_rules!`

- **File:** MMSB/src/11_agents/symbolic_agent.jl:32
- **Signature:** `infer_rules!(agent::SymbolicAgent, observations)`

## Layer: root

### Rust Functions

#### `log_error_code`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `kind`

#### `main`

- **File:** MMSB/build.rs:0
- **Visibility:** Private

#### `mask_from_bytes`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Private
- **Calls:**
  - `is_null`
  - `Vec::new`
  - `std::slice::from_raw_parts`
  - `saturating_mul`
  - `collect`
  - `map`
  - `iter`
  - `Vec::with_capacity`
  - `saturating_mul`
  - `push`
  - `len`
  - `truncate`

#### `mmsb_allocator_allocate`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `PageHandle::null`
  - `set_last_error`
  - `PageHandle::null`
  - `PageLocation::from_tag`
  - `Some`
  - `allocate_raw`
  - `PageID`
  - `set_last_error`
  - `PageHandle::null`

#### `mmsb_allocator_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_allocator_get_page`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `PageHandle::null`
  - `acquire_page`
  - `PageID`
  - `set_last_error`
  - `PageHandle::null`

#### `mmsb_allocator_list_pages`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `page_infos`
  - `min`
  - `len`
  - `with`
  - `borrow_mut`
  - `clear`
  - `extend`
  - `map`
  - `take`
  - `iter`
  - `clone`
  - `enumerate`
  - `take`
  - `iter`
  - `is_empty`
  - `std::ptr::null`
  - `as_ptr`
  - `len`
  - `add`

#### `mmsb_allocator_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `PageAllocatorConfig::default`
  - `PageAllocator::new`
  - `Box::new`
  - `Box::into_raw`

#### `mmsb_allocator_page_count`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_allocator_release`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `release`
  - `PageID`

#### `mmsb_checkpoint_load`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `to_str`
  - `CStr::from_ptr`
  - `is_null`
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `to_str`
  - `CStr::from_ptr`
  - `to_owned`
  - `set_last_error`
  - `checkpoint::load_checkpoint`
  - `set_last_error`

#### `mmsb_checkpoint_write`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `to_str`
  - `CStr::from_ptr`
  - `is_null`
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `to_str`
  - `CStr::from_ptr`
  - `to_owned`
  - `set_last_error`
  - `checkpoint::write_checkpoint`
  - `set_last_error`

#### `mmsb_delta_apply`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `apply_delta`
  - `set_last_error`

#### `mmsb_delta_copy_intent_metadata`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `as_bytes`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_delta_copy_mask`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `min`
  - `len`
  - `take`
  - `enumerate`
  - `iter`
  - `add`

#### `mmsb_delta_copy_payload`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_delta_copy_source`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `as_bytes`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_delta_epoch`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_delta_id`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_intent_metadata_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `unwrap_or`
  - `map`
  - `as_ref`
  - `len`

#### `mmsb_delta_is_sparse`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_mask_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_delta_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `mask_from_bytes`
  - `vec_from_ptr`
  - `is_null`
  - `to_string`
  - `to_string`
  - `to_string_lossy`
  - `CStr::from_ptr`
  - `DeltaID`
  - `PageID`
  - `into`
  - `Source`
  - `Box::new`
  - `Box::into_raw`

#### `mmsb_delta_page_id`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_delta_payload_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_delta_set_intent_metadata`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `is_null`
  - `std::slice::from_raw_parts`
  - `std::str::from_utf8`
  - `Some`
  - `to_string`
  - `set_last_error`

#### `mmsb_delta_source_len`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`

#### `mmsb_delta_timestamp`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`

#### `mmsb_error_is_fatal`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_fatal`

#### `mmsb_error_is_retryable`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_retryable`

#### `mmsb_get_last_error`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `with`
  - `borrow_mut`

#### `mmsb_page_epoch`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `epoch`

#### `mmsb_page_metadata_export`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `metadata_blob`
  - `min`
  - `len`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_page_metadata_import`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `std::slice::from_raw_parts`
  - `set_metadata_blob`
  - `set_last_error`

#### `mmsb_page_metadata_size`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `len`
  - `metadata_blob`

#### `mmsb_page_read`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `size`
  - `is_null`
  - `set_last_error`
  - `size`
  - `min`
  - `data_slice`
  - `std::ptr::copy_nonoverlapping`
  - `as_ptr`

#### `mmsb_page_write_masked`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `mask_from_bytes`
  - `vec_from_ptr`
  - `DeltaID`
  - `into`
  - `Source`
  - `into`
  - `apply_delta`
  - `set_last_error`

#### `mmsb_semiring_boolean_accumulate`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `accumulate`

#### `mmsb_semiring_boolean_fold_add`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `zero`
  - `slice_from_ptr`
  - `map`
  - `iter`
  - `fold_add`

#### `mmsb_semiring_boolean_fold_mul`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `one`
  - `slice_from_ptr`
  - `map`
  - `iter`
  - `fold_mul`

#### `mmsb_semiring_tropical_accumulate`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `accumulate`

#### `mmsb_semiring_tropical_fold_add`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `zero`
  - `slice_from_ptr`
  - `fold_add`
  - `copied`
  - `iter`

#### `mmsb_semiring_tropical_fold_mul`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `one`
  - `slice_from_ptr`
  - `fold_mul`
  - `copied`
  - `iter`

#### `mmsb_tlog_append`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `append`
  - `clone`
  - `set_last_error`
  - `log_error_code`

#### `mmsb_tlog_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_tlog_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `TLogHandle::null`
  - `CStr::from_ptr`
  - `to_str`
  - `set_last_error`
  - `TLogHandle::null`
  - `to_owned`
  - `TransactionLog::new`
  - `Box::into_raw`
  - `Box::new`
  - `set_last_error`
  - `log_error_code`
  - `TLogHandle::null`

#### `mmsb_tlog_reader_free`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `drop`
  - `Box::from_raw`

#### `mmsb_tlog_reader_new`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `TLogReaderHandle::null`
  - `to_str`
  - `CStr::from_ptr`
  - `to_owned`
  - `set_last_error`
  - `TLogReaderHandle::null`
  - `TransactionLogReader::open`
  - `Box::into_raw`
  - `Box::new`
  - `set_last_error`
  - `log_error_code`
  - `TLogReaderHandle::null`

#### `mmsb_tlog_reader_next`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `set_last_error`
  - `DeltaHandle::null`
  - `next`
  - `Box::new`
  - `Box::into_raw`
  - `DeltaHandle::null`
  - `set_last_error`
  - `log_error_code`
  - `DeltaHandle::null`

#### `mmsb_tlog_summary`

- **File:** MMSB/src/ffi.rs:0
- **Visibility:** Public
- **Calls:**
  - `is_null`
  - `is_null`
  - `set_last_error`
  - `to_str`
  - `CStr::from_ptr`
  - `set_last_error`
  - `crate::page::tlog::summary`
  - `set_last_error`
  - `log_error_code`

### Julia Functions

#### `_graph_bfs`

- **File:** MMSB/benchmark/benchmarks.jl:117
- **Signature:** `_graph_bfs(graph::GraphTypes.ShadowPageGraph, roots)::Int`
- **Calls:**
  - `GraphTypes.get_children`
  - `collect`
  - `isempty`
  - `length`
  - `popfirst!`
  - `push!`

#### `_graph_fixture`

- **File:** MMSB/benchmark/benchmarks.jl:101
- **Signature:** `_graph_fixture(node_count::Int, fanout::Int)`
- **Calls:**
  - `GraphTypes.ShadowPageGraph`
  - `GraphTypes.add_dependency!`
  - `PageTypes.PageID`
  - `in`
  - `min`

#### `_link_chain!`

- **File:** MMSB/benchmark/benchmarks.jl:81
- **Signature:** `_link_chain!(state, pages)`
- **Calls:**
  - `GraphTypes.add_dependency!`
  - `PropagationEngine.register_passthrough_recompute!`
  - `length`

#### `_measure_ns`

- **File:** MMSB/benchmark/benchmarks.jl:95
- **Signature:** `_measure_ns(f::Function)`
- **Calls:**
  - `f`
  - `time_ns`

#### `isnull`

- **File:** MMSB/src/ffi/FFIWrapper.jl:499
- **Signature:** `isnull(handle)`

#### `length`

- **File:** MMSB/src/API.jl:92
- **Signature:** `length(bytes)`
- **Calls:**
  - `InvalidDeltaError`
  - `UInt64`
  - `throw`

#### `mmsb_start`

- **File:** MMSB/src/API.jl:30
- **Signature:** `mmsb_start(; enable_gpu::Bool`

#### `mmsb_stop`

- **File:** MMSB/src/API.jl:44
- **Signature:** `mmsb_stop(state::MMSBState; checkpoint_path::Union{Nothing,String}`

