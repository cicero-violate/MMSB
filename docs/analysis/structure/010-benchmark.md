# Structure Group: benchmark

## File: MMSB/benchmark/benchmarks.jl

- Layer(s): root
- Language coverage: Julia (21)
- Element types: Function (20), Module (1)
- Total elements: 21

### Elements

- [Julia | Module] `MMSBBenchmarks` (line 1, pub)
- [Julia | Function] `_start_state` (line 43, pub)
  - Signature: `_start_state(; enable_gpu::Bool`
- [Julia | Function] `_stop_state!` (line 47, pub)
  - Signature: `_stop_state!(state)`
  - Calls: API.mmsb_stop
- [Julia | Function] `_page` (line 53, pub)
  - Signature: `_page(state, bytes::Int; location::Symbol`
- [Julia | Function] `_populate_pages!` (line 57, pub)
  - Signature: `_populate_pages!(state, count::Int, bytes::Int)`
  - Calls: API.create_page
- [Julia | Function] `_seed_pages!` (line 61, pub)
  - Signature: `_seed_pages!(state, count::Int, bytes::Int)`
  - Calls: API.update_page, _page, rand
- [Julia | Function] `_replay_sequence!` (line 68, pub)
  - Signature: `_replay_sequence!(state, page, epochs::Int, bytes::Int)`
  - Calls: API.update_page, rand
- [Julia | Function] `_stress_updates!` (line 74, pub)
  - Signature: `_stress_updates!(state, pages, updates::Int, bytes::Int)`
  - Calls: API.update_page, length, rand
- [Julia | Function] `_link_chain!` (line 81, pub)
  - Signature: `_link_chain!(state, pages)`
  - Calls: GraphTypes.add_dependency!, PropagationEngine.register_passthrough_recompute!, length
- [Julia | Function] `_checkpoint` (line 89, pub)
  - Signature: `_checkpoint(state)`
  - Calls: TLog.checkpoint_log!, tempname
- [Julia | Function] `_measure_ns` (line 95, pub)
  - Signature: `_measure_ns(f::Function)`
  - Calls: f, time_ns
- [Julia | Function] `_graph_fixture` (line 101, pub)
  - Signature: `_graph_fixture(node_count::Int, fanout::Int)`
  - Calls: GraphTypes.ShadowPageGraph, GraphTypes.add_dependency!, PageTypes.PageID, in, min
- [Julia | Function] `_graph_bfs` (line 117, pub)
  - Signature: `_graph_bfs(graph::GraphTypes.ShadowPageGraph, roots)::Int`
  - Calls: GraphTypes.get_children, collect, isempty, length, popfirst!, push!
- [Julia | Function] `_build_batch_deltas` (line 131, pub)
  - Signature: `_build_batch_deltas(state, page_id::PageTypes.PageID, batch_size::Int)`
  - Calls: DeltaRouter.create_delta, fill, push!, rand
- [Julia | Function] `_full_system_benchmark!` (line 141, pub)
  - Signature: `_full_system_benchmark!()`
  - Calls: API.update_page, GraphTypes.add_dependency!, MMSB.MMSBStateTypes.MMSBState, PropagationEngine.register_passthrough_recompute!, ReplayEngine.replay_to_epoch, TLog.load_checkpoint!, UInt32, _checkpoint, _populate_pages!, _start_state, _stop_state!, isfile, length, rand, rm
- [Julia | Function] `_trial_to_dict` (line 412, pub)
  - Signature: `_trial_to_dict(trial)`
  - Calls: Dict, maximum, mean, median, minimum
- [Julia | Function] `_select_suite` (line 423, pub)
  - Signature: `_select_suite(categories::Vector{String})`
  - Calls: BenchmarkGroup, haskey
- [Julia | Function] `_collect_instrumentation_report` (line 435, pub)
  - Signature: `_collect_instrumentation_report()`
  - Calls: API.update_page, GraphTypes.add_dependency!, Monitoring.get_stats, Monitoring.reset_stats!, PropagationEngine.register_passthrough_recompute!, Semiring.boolean_fold_add, Semiring.tropical_fold_add, _measure_ns, _page, _start_state, _stop_state!, rand
- [Julia | Function] `_to_mutable` (line 472, pub)
  - Signature: `_to_mutable(value)`
  - Calls: Dict, _to_mutable, pairs, string
- [Julia | Function] `run_benchmarks` (line 482, pub)
  - Signature: `run_benchmarks(; save_results::Bool`
- [Julia | Function] `compare_with_baseline` (line 518, pub)
  - Signature: `compare_with_baseline(current_results)`
  - Calls: JSON3.read, haskey, isfile, joinpath, median, println, read, round, string

## File: MMSB/benchmark/helpers.jl

- Layer(s): root
- Language coverage: Julia (5)
- Element types: Function (4), Module (1)
- Total elements: 5

### Elements

- [Julia | Module] `MMSBBenchmarkHelpers` (line 1, pub)
- [Julia | Function] `_format_time` (line 6, pub)
  - Signature: `_format_time(ns::Float64)`
  - Calls: round
- [Julia | Function] `_format_bytes` (line 18, pub)
  - Signature: `_format_bytes(bytes::Int)`
  - Calls: round
- [Julia | Function] `analyze_results` (line 30, pub)
  - Signature: `analyze_results(results)`
  - Calls: _format_bytes, _format_time, mean, median, println, std
- [Julia | Function] `check_performance_targets` (line 51, pub)
  - Signature: `check_performance_targets(results)`
  - Calls: Dict, _format_time, haskey, median, println, round

