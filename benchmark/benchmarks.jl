module MMSBBenchmarks

using BenchmarkTools
using CUDA
using Random
using Statistics
using Dates: now
using JSON3
using Base: time_ns

include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB

const GraphTypes = MMSB.GraphTypes
const PropagationEngine = MMSB.PropagationEngine
const TLog = MMSB.TLog
const ReplayEngine = MMSB.ReplayEngine
const API = MMSB.API
const Semiring = MMSB.Semiring
const Monitoring = MMSB.Monitoring
const DeltaRouter = MMSB.DeltaRouter
const PageTypes = MMSB.PageTypes
const Delta = MMSB.Delta

const SUITE = BenchmarkGroup()
const BENCH_CONFIG = (
    small_page=1024,
    large_page=256 * 1024,
    replay_epochs=5,
    persistence_pages=16,
    stress_pages=20,
    stress_updates=40,
    stress_edges=6,
)
const BENCH_PARAMS = BenchmarkTools.Parameters(
    seconds=0.05,
    samples=1,
    evals=1,
    gctrial=false,
    gcsample=false,
)

function _start_state(; enable_gpu::Bool=false)
    cfg = MMSB.MMSBStateTypes.MMSBConfig(enable_gpu=enable_gpu, enable_logging=false)
    return API.mmsb_start(config=cfg)
end

function _stop_state!(state)
    state === nothing && return nothing
    API.mmsb_stop(state)
    return nothing
end

function _page(state, bytes::Int; location::Symbol=:cpu)
    return API.create_page(state; size=bytes, location=location)
end

function _populate_pages!(state, count::Int, bytes::Int)
    return [API.create_page(state; size=bytes) for idx in 1:count]
end

function _seed_pages!(state, count::Int, bytes::Int)
    for page_idx in 1:count
        page = _page(state, bytes)
        API.update_page(state, page.id, rand(UInt8, bytes))
    end
end

function _replay_sequence!(state, page, epochs::Int, bytes::Int)
    for _ in 1:epochs
        API.update_page(state, page.id, rand(UInt8, bytes))
    end
end

function _stress_updates!(state, pages, updates::Int, bytes::Int)
    for _ in 1:updates
        page = pages[rand(1:length(pages))]
        API.update_page(state, page.id, rand(UInt8, bytes))
    end
end

function _link_chain!(state, pages)
    length(pages) < 2 && return
    for idx in 1:(length(pages) - 1)
        GraphTypes.add_dependency!(state.graph, pages[idx].id, pages[idx+1].id, GraphTypes.DATA_DEPENDENCY)
        PropagationEngine.register_passthrough_recompute!(state, pages[idx+1].id, pages[idx].id)
    end
end

function _checkpoint(state)
    path = tempname()
    TLog.checkpoint_log!(state, path)
    return path
end

function _measure_ns(f::Function)
    start = time_ns()
    f()
    return time_ns() - start
end

function _graph_fixture(node_count::Int, fanout::Int)
    graph = GraphTypes.ShadowPageGraph()
    nodes = [PageTypes.PageID(idx) for idx in 1:node_count]
    for idx in 1:node_count
        if idx == node_count
            continue
        end
        limit = min(node_count, idx + fanout)
        for child_idx in (idx+1):limit
            GraphTypes.add_dependency!(graph, nodes[idx], nodes[child_idx], GraphTypes.DATA_DEPENDENCY)
        end
    end
    roots = nodes[1:min(node_count, fanout)]
    return graph, roots
end

function _graph_bfs(graph::GraphTypes.ShadowPageGraph, roots)::Int
    visited = Set{PageTypes.PageID}()
    queue = collect(roots)
    while !isempty(queue)
        node = popfirst!(queue)
        node ∈ visited && continue
        push!(visited, node)
        for (child, _) in GraphTypes.get_children(graph, node)
            push!(queue, child)
        end
    end
    return length(visited)
end

function _build_batch_deltas(state, page_id::PageTypes.PageID, batch_size::Int)
    deltas = Delta[]
    for _ in 1:batch_size
        payload = rand(UInt8, BENCH_CONFIG.small_page)
        mask = fill(true, BENCH_CONFIG.small_page)
        push!(deltas, DeltaRouter.create_delta(state, page_id, mask, payload; source=:bench))
    end
    return deltas
end

function _full_system_benchmark!()
    state = _start_state(; enable_gpu=false)
    tmp = nothing
    restored = nothing
    try
        pages = _populate_pages!(state, 24, BENCH_CONFIG.small_page)
        for idx in 2:length(pages)
            GraphTypes.add_dependency!(state.graph, pages[idx-1].id, pages[idx].id, GraphTypes.DATA_DEPENDENCY)
            PropagationEngine.register_passthrough_recompute!(state, pages[idx].id, pages[idx-1].id)
        end
        for _ in 1:BENCH_CONFIG.replay_epochs
            API.update_page(state, pages[1].id, rand(UInt8, BENCH_CONFIG.small_page))
        end
        tmp = _checkpoint(state)
        restored = MMSB.MMSBStateTypes.MMSBState()
        TLog.load_checkpoint!(restored, tmp)
        ReplayEngine.replay_to_epoch(restored, UInt32(BENCH_CONFIG.replay_epochs))
    finally
        tmp !== nothing && isfile(tmp) && rm(tmp, force=true)
        restored !== nothing && _stop_state!(restored)
        _stop_state!(state)
    end
end

SUITE["allocation"] = BenchmarkGroup()

SUITE["allocation"]["cpu_1kb"] = @benchmarkable begin
    state = _start_state(; enable_gpu=false)
    _page(state, BENCH_CONFIG.small_page)
    _stop_state!(state)
end

SUITE["allocation"]["cpu_1mb"] = @benchmarkable begin
    state = _start_state(; enable_gpu=false)
    _page(state, BENCH_CONFIG.large_page)
    _stop_state!(state)
end

if CUDA.functional()
    SUITE["allocation"]["gpu_1kb"] = @benchmarkable begin
        state = _start_state(; enable_gpu=true)
        _page(state, BENCH_CONFIG.small_page; location=:gpu)
        _stop_state!(state)
    end

    SUITE["allocation"]["gpu_1mb"] = @benchmarkable begin
        state = _start_state(; enable_gpu=true)
        _page(state, BENCH_CONFIG.large_page; location=:gpu)
        _stop_state!(state)
    end
end

SUITE["delta"] = BenchmarkGroup()

SUITE["delta"]["cpu_sparse"] = @benchmarkable begin
    API.update_page(state, page_id, data)
end setup=(
    state = _start_state(; enable_gpu=false);
    page = _page(state, BENCH_CONFIG.small_page);
    page_id = page.id;
    data = zeros(UInt8, BENCH_CONFIG.small_page);
    data[1:clamp(div(BENCH_CONFIG.small_page, 100), 1, BENCH_CONFIG.small_page)] .= 0xff;
) teardown=(
    _stop_state!(state);
)

SUITE["delta"]["cpu_dense"] = @benchmarkable begin
    API.update_page(state, page_id, data)
end setup=(
    state = _start_state(; enable_gpu=false);
    page = _page(state, BENCH_CONFIG.small_page);
    page_id = page.id;
    data = rand(UInt8, BENCH_CONFIG.small_page);
) teardown=(
    _stop_state!(state);
)

if CUDA.functional()
    SUITE["delta"]["gpu_sparse"] = @benchmarkable begin
        API.update_page(state, page_id, data)
    end setup=(
        state = _start_state(; enable_gpu=true);
        page = _page(state, BENCH_CONFIG.large_page; location=:gpu);
        page_id = page.id;
        data = zeros(UInt8, BENCH_CONFIG.large_page);
        data[1:max(1, div(BENCH_CONFIG.large_page, 1024))] .= 0xff;
    ) teardown=(
        _stop_state!(state);
    )

    SUITE["delta"]["gpu_dense"] = @benchmarkable begin
        API.update_page(state, page_id, data)
    end setup=(
        state = _start_state(; enable_gpu=true);
        page = _page(state, BENCH_CONFIG.large_page; location=:gpu);
        page_id = page.id;
        data = rand(UInt8, BENCH_CONFIG.large_page);
    ) teardown=(
        _stop_state!(state);
    )
end

SUITE["semiring"] = BenchmarkGroup()

SUITE["semiring"]["tropical_fold_add"] = @benchmarkable begin
    Semiring.tropical_fold_add(values)
end setup=(
    values = rand(Float64, BENCH_CONFIG.small_page);
)

SUITE["semiring"]["tropical_fold_mul"] = @benchmarkable begin
    Semiring.tropical_fold_mul(values)
end setup=(
    values = rand(Float64, BENCH_CONFIG.small_page);
)

SUITE["semiring"]["tropical_accumulate"] = @benchmarkable begin
    Semiring.tropical_accumulate(left, right)
end setup=(
    left = rand();
    right = rand();
)

SUITE["semiring"]["boolean_fold_add"] = @benchmarkable begin
    Semiring.boolean_fold_add(values)
end setup=(
    values = rand(Bool, BENCH_CONFIG.small_page);
)

SUITE["semiring"]["boolean_fold_mul"] = @benchmarkable begin
    Semiring.boolean_fold_mul(values)
end setup=(
    values = rand(Bool, BENCH_CONFIG.small_page);
)

SUITE["semiring"]["boolean_accumulate"] = @benchmarkable begin
    Semiring.boolean_accumulate(left, right)
end setup=(
    left = rand(Bool);
    right = rand(Bool);
)

SUITE["propagation"] = BenchmarkGroup()

SUITE["propagation"]["single_hop"] = @benchmarkable begin
    API.update_page(state, parent_id, data)
end setup=(
    state = _start_state(; enable_gpu=false);
    parent = _page(state, BENCH_CONFIG.small_page);
    child = _page(state, BENCH_CONFIG.small_page);
    GraphTypes.add_dependency!(state.graph, parent.id, child.id, GraphTypes.DATA_DEPENDENCY);
    PropagationEngine.register_passthrough_recompute!(state, child.id, parent.id);
    parent_id = parent.id;
    data = rand(UInt8, BENCH_CONFIG.small_page);
) teardown=(
    _stop_state!(state);
)

SUITE["propagation"]["chain_10"] = @benchmarkable begin
    API.update_page(state, first_id, data)
end setup=(
    state = _start_state(; enable_gpu=false);
    pages = _populate_pages!(state, 10, BENCH_CONFIG.small_page);
    _link_chain!(state, pages);
    first_id = pages[1].id;
    data = rand(UInt8, BENCH_CONFIG.small_page);
) teardown=(
    _stop_state!(state);
)

SUITE["propagation"]["batch_32x4"] = @benchmarkable begin
    DeltaRouter.batch_route_deltas!(state, deltas)
end setup=(
    state = _start_state(; enable_gpu=false);
    pages = _populate_pages!(state, 32, BENCH_CONFIG.small_page);
    _link_chain!(state, pages);
    deltas = _build_batch_deltas(state, pages[1].id, 4);
) teardown=(
    _stop_state!(state);
)

SUITE["propagation"]["batch_128x2"] = @benchmarkable begin
    DeltaRouter.batch_route_deltas!(state, deltas)
end setup=(
    state = _start_state(; enable_gpu=false);
    pages = _populate_pages!(state, 128, BENCH_CONFIG.small_page);
    _link_chain!(state, pages);
    deltas = _build_batch_deltas(state, pages[1].id, 2);
) teardown=(
    _stop_state!(state);
)

SUITE["graph"] = BenchmarkGroup()

SUITE["graph"]["topological_sort_1024"] = @benchmarkable begin
    GraphTypes.topological_sort(graph)
end setup=(
    fixture = _graph_fixture(1024, 4);
    graph = fixture[1];
)

SUITE["graph"]["bfs_1024"] = @benchmarkable begin
    _graph_bfs(graph, roots)
end setup=(
    fixture = _graph_fixture(1024, 4);
    graph = fixture[1];
    roots = fixture[2];
)

SUITE["persistence"] = BenchmarkGroup()

SUITE["persistence"]["checkpoint_100pages"] = @benchmarkable begin
    TLog.checkpoint_log!(state, path)
end setup=(
    state = _start_state(; enable_gpu=false);
    _seed_pages!(state, BENCH_CONFIG.persistence_pages, BENCH_CONFIG.small_page);
    path = tempname();
) teardown=(
    isfile(path) && rm(path, force=true);
    _stop_state!(state);
)

SUITE["persistence"]["restore_100pages"] = @benchmarkable begin
    restored = MMSB.MMSBStateTypes.MMSBState();
    TLog.load_checkpoint!(restored, path)
end setup=(
    state = _start_state(; enable_gpu=false);
    _seed_pages!(state, BENCH_CONFIG.persistence_pages, BENCH_CONFIG.small_page);
    path = tempname();
    TLog.checkpoint_log!(state, path);
    _stop_state!(state);
) teardown=(
    isfile(path) && rm(path, force=true);
)

SUITE["replay"] = BenchmarkGroup()

SUITE["replay"]["100_epochs"] = @benchmarkable begin
    ReplayEngine.replay_to_epoch(state, UInt32(BENCH_CONFIG.replay_epochs))
end setup=(
    state = _start_state(; enable_gpu=false);
    page = _page(state, BENCH_CONFIG.small_page);
    _replay_sequence!(state, page, BENCH_CONFIG.replay_epochs, BENCH_CONFIG.small_page);
) teardown=(
    _stop_state!(state);
)

SUITE["stress"] = BenchmarkGroup()

SUITE["stress"]["typical_workload"] = @benchmarkable begin
    local_state = _start_state(; enable_gpu=false)
    pages = [_page(local_state, BENCH_CONFIG.small_page) for idx in 1:BENCH_CONFIG.stress_pages]
    split_idx = max(2, length(pages) ÷ 2)
    for rel in 1:BENCH_CONFIG.stress_edges
        parent = pages[rand(1:split_idx)]
        child = pages[rand(split_idx+1:length(pages))]
        GraphTypes.add_dependency!(local_state.graph, parent.id, child.id, GraphTypes.DATA_DEPENDENCY)
    end
    _stress_updates!(local_state, pages, BENCH_CONFIG.stress_updates, BENCH_CONFIG.small_page)
    tmp = tempname()
    TLog.checkpoint_log!(local_state, tmp)
    _stop_state!(local_state)
    isfile(tmp) && rm(tmp, force=true)
end

SUITE["system"] = BenchmarkGroup()

SUITE["system"]["full_pipeline"] = @benchmarkable begin
    _full_system_benchmark!()
end

function _trial_to_dict(trial)
    return Dict(
        "median_ns" => median(trial).time,
        "mean_ns" => mean(trial).time,
        "min_ns" => minimum(trial).time,
        "max_ns" => maximum(trial).time,
        "allocs" => trial.allocs,
        "memory_bytes" => trial.memory
    )
end

function _select_suite(categories::Vector{String})
    group = BenchmarkGroup()
    for cat in categories
        if haskey(SUITE, cat)
            group[cat] = SUITE[cat]
        else
            @warn "Requested benchmark category not found" category=cat
        end
    end
    return group
end

function _collect_instrumentation_report()
    state = _start_state(; enable_gpu=false)
    report = Dict{String,Any}()
    try
        Monitoring.reset_stats!(state)
        alloc_ns = _measure_ns(() -> begin
            _page(state, BENCH_CONFIG.small_page)
        end)
        semiring_tropical_ns = _measure_ns(() -> begin
            Semiring.tropical_fold_add(rand(Float64, BENCH_CONFIG.small_page))
        end)
        semiring_boolean_ns = _measure_ns(() -> begin
            Semiring.boolean_fold_add(rand(Bool, BENCH_CONFIG.small_page))
        end)
        parent = _page(state, BENCH_CONFIG.small_page)
        child = _page(state, BENCH_CONFIG.small_page)
        GraphTypes.add_dependency!(state.graph, parent.id, child.id, GraphTypes.DATA_DEPENDENCY)
        PropagationEngine.register_passthrough_recompute!(state, child.id, parent.id)
        payload = rand(UInt8, BENCH_CONFIG.small_page)
        prop_ns = _measure_ns(() -> begin
            API.update_page(state, parent.id, payload)
        end)
        stats = Monitoring.get_stats(state)
        report["allocation_ns"] = alloc_ns
        report["propagation_ns"] = prop_ns
        report["delta_avg_ns"] = stats.delta_apply_avg_ns
        report["propagation_avg_ns"] = stats.propagation_avg_ns
        report["total_pages"] = stats.total_pages
        report["total_deltas"] = stats.total_deltas
        report["semiring_tropical_add_ns"] = semiring_tropical_ns
        report["semiring_boolean_add_ns"] = semiring_boolean_ns
    finally
        _stop_state!(state)
    end
    return report
end

function _to_mutable(value)
    if value isa JSON3.Object
        return Dict(string(k) => _to_mutable(v) for (k, v) in pairs(value))
    elseif value isa JSON3.Array
        return [_to_mutable(v) for v in value]
    else
        return value
    end
end

function run_benchmarks(; save_results::Bool=true, params::BenchmarkTools.Parameters=BENCH_PARAMS,
                        categories::Union{Nothing,Vector{String}}=nothing)
    println("Running MMSB Benchmark Suite…")
    println("=" ^ 60)
    target_suite = categories === nothing ? SUITE : _select_suite(categories)
    isempty(keys(target_suite)) && error("No benchmarks selected to run")
    results = run(target_suite, params; verbose=true)
    if save_results
        results_path = joinpath(@__DIR__, "results", "baseline.json")
        mkpath(dirname(results_path))
        serialized = if isfile(results_path)
            _to_mutable(JSON3.read(read(results_path, String)))
        else
            Dict{String,Any}()
        end
        serialized["timestamp"] = string(now())
        serialized["julia_version"] = string(VERSION)
        serialized["cuda_available"] = CUDA.functional()
        serialized["benchmarks"] = get(serialized, "benchmarks", Dict{String,Any}())
        for (category, group) in results
            cat_key = string(category)
            cat_dict = Dict{String,Any}()
            for (name, trial) in group
                cat_dict[string(name)] = _trial_to_dict(trial)
            end
            serialized["benchmarks"][cat_key] = cat_dict
        end
        serialized["instrumentation"] = _collect_instrumentation_report()
        open(results_path, "w") do io
            JSON3.pretty(io, serialized)
        end
        println("\nResults saved to: $results_path")
    end
    return results
end

function compare_with_baseline(current_results)
    baseline_path = joinpath(@__DIR__, "results", "baseline.json")
    if !isfile(baseline_path)
        @warn "No baseline found. Run benchmarks first to establish baseline."
        return nothing
    end
    baseline = JSON3.read(read(baseline_path, String))
    println("\n" * "=" ^ 60)
    println("Performance Comparison vs Baseline")
    println("=" ^ 60)
    for (category, group) in current_results
        println("\n[$category]")
        for (name, trial) in group
            cat_key = string(category)
            name_key = string(name)
            if haskey(baseline["benchmarks"], cat_key) &&
               haskey(baseline["benchmarks"][cat_key], name_key)
                current_median = median(trial).time
                baseline_median = baseline["benchmarks"][cat_key][name_key]["median_ns"]
                ratio = current_median / baseline_median
                status = if ratio < 0.9
                    "✅ $(round((1 - ratio) * 100, digits=1))% faster"
                elseif ratio > 1.1
                    "⚠️ $(round((ratio - 1) * 100, digits=1))% slower"
                else
                    "➡️ similar"
                end
                println("  $name_key: $(round(current_median / 1e6, digits=2)) ms $status")
            end
        end
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    env_cats = strip(get(ENV, "MMSB_BENCH_CATEGORIES", ""))
    selected = isempty(env_cats) ? nothing : [strip(c) for c in split(env_cats, ',') if !isempty(strip(c))]
    results = run_benchmarks(categories=selected)
    include(joinpath(@__DIR__, "helpers.jl"))
    MMSBBenchmarkHelpers.analyze_results(results)
    MMSBBenchmarkHelpers.check_performance_targets(results)
end

end # module MMSBBenchmarks
