# src/utils/Monitoring.jl
"""
Monitoring utilities for MMSB.

Provides cumulative stats so operators and benchmarks can inspect
page registry size, delta throughput, graph shape, and timing data.
"""
module Monitoring

using ..PageTypes: Page, PageID, PageLocation, CPU_LOCATION, GPU_LOCATION, UNIFIED_LOCATION
using ..MMSBStateTypes: MMSBState
using ..FFIWrapper
using ..GraphTypes: get_children

export MMSBStats, get_stats, print_stats, reset_stats!, track_delta_latency!, track_propagation_latency!

mutable struct MMSBStats
    total_pages::Int
    cpu_pages::Int
    gpu_pages::Int
    unified_pages::Int
    total_page_bytes::Int64
    total_deltas::Int
    total_delta_bytes::Int64
    avg_delta_bytes::Float64
    graph_nodes::Int
    graph_edges::Int
    graph_max_depth::Int
    delta_apply_count::Int
    delta_apply_total_ns::UInt64
    delta_apply_avg_ns::Float64
    propagation_count::Int
    propagation_total_ns::UInt64
    propagation_avg_ns::Float64
end

const DELTA_METRICS = Dict{MMSBState, Tuple{Int, UInt64}}()
const PROP_METRICS = Dict{MMSBState, Tuple{Int, UInt64}}()

function track_delta_latency!(state::MMSBState, duration_ns::UInt64)
    count, total = get(DELTA_METRICS, state, (0, UInt64(0)))
    DELTA_METRICS[state] = (count + 1, total + duration_ns)
end

function track_propagation_latency!(state::MMSBState, duration_ns::UInt64)
    count, total = get(PROP_METRICS, state, (0, UInt64(0)))
    PROP_METRICS[state] = (count + 1, total + duration_ns)
end

function compute_graph_depth(graph)
    max_depth = 0
    for (root, _) in graph.deps
        depth = _dfs_depth(graph, root, 0, Set{PageID}())
        max_depth = max(max_depth, depth)
    end
    return max_depth
end

function _dfs_depth(graph, node::PageID, depth::Int, visited::Set{PageID})
    if node ∈ visited
        return depth
    end
    push!(visited, node)
    children = get(graph.deps, node, Tuple{PageID,Any}[])
    if isempty(children)
        delete!(visited, node)
        return depth
    end
    max_depth = maximum(_dfs_depth(graph, child[1], depth + 1, visited) for child in children)
    delete!(visited, node)
    return max_depth
end

function get_stats(state::MMSBState)::MMSBStats
    lock(state.lock) do
        cpu = 0; gpu = 0; unified = 0; bytes = Int64(0)
        for (_, page) in state.pages
            bytes += page.size
            if page.location == CPU_LOCATION
                cpu += 1
            elseif page.location == GPU_LOCATION
                gpu += 1
            else
                unified += 1
            end
        end
        total_pages = cpu + gpu + unified
        logging_enabled = state.config.enable_logging
        summary = logging_enabled ? FFIWrapper.rust_tlog_summary(state.config.tlog_path) : nothing
        total_deltas = summary === nothing ? 0 : summary.total_deltas
        total_delta_bytes = summary === nothing ? 0 : summary.total_bytes
        avg_delta = total_deltas > 0 ? total_delta_bytes / total_deltas : 0.0
        nodes = length(state.graph.deps)
        edges = sum((length(children) for children in values(state.graph.deps)); init=0)
        depth = compute_graph_depth(state.graph)
        delta_count, delta_total = get(DELTA_METRICS, state, (0, UInt64(0)))
        delta_avg = delta_count > 0 ? delta_total ÷ delta_count : 0.0
        prop_count, prop_total = get(PROP_METRICS, state, (0, UInt64(0)))
        prop_avg = prop_count > 0 ? prop_total ÷ prop_count : 0.0
        return MMSBStats(total_pages, cpu, gpu, unified, bytes,
                         total_deltas, total_delta_bytes, avg_delta,
                         nodes, edges, depth,
                         delta_count, delta_total, delta_avg,
                         prop_count, prop_total, prop_avg)
    end
end

function print_stats(state::MMSBState)
    stats = get_stats(state)
    println("=== MMSB Stats ===")
    println("Pages: ", stats.total_pages, " (CPU: ", stats.cpu_pages, ", GPU: ", stats.gpu_pages, ", Unified: ", stats.unified_pages, ")")
    println("Memory: ", stats.total_page_bytes, " bytes")
    println("Deltas: ", stats.total_deltas, " (avg bytes: ", round(stats.avg_delta_bytes, digits=2), ")")
    println("Graph nodes: ", stats.graph_nodes, " edges: ", stats.graph_edges, " max depth: ", stats.graph_max_depth)
    println("Delta apply avg ns: ", stats.delta_apply_avg_ns)
    println("Propagation avg ns: ", stats.propagation_avg_ns)
end

function reset_stats!(state::MMSBState)
    if haskey(DELTA_METRICS, state)
        DELTA_METRICS[state] = (0, UInt64(0))
    end
    if haskey(PROP_METRICS, state)
        PROP_METRICS[state] = (0, UInt64(0))
    end
end

end # module Monitoring
