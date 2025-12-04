# src/05_graph/PropagationEngine.jl
"""
PropagationEngine - Propagates changes through dependency graph

When a page changes, propagates invalidation and recomputation
through dependent pages based on edge types.
"""
module PropagationEngine

using Base: time_ns
using ..PageTypes: PageID
using ..MMSBStateTypes: MMSBState, get_page
using ..GraphTypes: EdgeType, DATA_DEPENDENCY, CONTROL_DEPENDENCY,
    GPU_SYNC_DEPENDENCY, COMPILER_DEPENDENCY, get_children
using ..EventSystem: emit_event!, PAGE_INVALIDATED, PAGE_STALE,
    GPU_SYNC_NEEDED, IR_INVALIDATED
using ..DeltaRouter: create_delta, route_delta!
using ..ErrorTypes: PageNotFoundError, InvalidDeltaError
using ..Monitoring: track_propagation_latency!

export propagate_change!, schedule_propagation!, execute_propagation!
export PropagationMode, IMMEDIATE, DEFERRED, BATCH
export register_recompute_fn!, register_passthrough_recompute!, queue_recomputation!

@enum PropagationMode begin
    IMMEDIATE  # Propagate immediately on change
    DEFERRED   # Queue for later batch propagation
    BATCH      # Accumulate and propagate in batches
end

const PROPAGATION_QUEUES = IdDict{MMSBState, Vector{PageID}}()

"""
Fetch (or create) the recomputation queue for a state.
"""
function _queue(state::MMSBState)::Vector{PageID}
    return get!(PROPAGATION_QUEUES, state) do
        PageID[]
    end
end

"""
    register_recompute_fn!(state, page_id, fn)

Attach a recompute function to a page. The function must take `(state, page)` and
return a `Vector{UInt8}` representing the new page contents.
"""
function register_recompute_fn!(state::MMSBState, page_id::PageID, fn::Function)
    page = get_page(state, page_id)
    page === nothing && throw(PageNotFoundError(UInt64(page_id), "register_recompute_fn!"))
    page.metadata[:recompute_fn] = fn
    return page
end

"""
    register_passthrough_recompute!(state, target_page_id, source_page_id; transform=identity)

Registers a recomputation closure that copies `source_page_id` into `target_page_id`
and optionally applies a `transform` function to the raw bytes.
"""
function register_passthrough_recompute!(state::MMSBState, target_page_id::PageID,
                                         source_page_id::PageID; transform=identity)
    register_recompute_fn!(state, target_page_id, function (st, _)
        source = get_page(st, source_page_id)
        source === nothing && throw(PageNotFoundError(UInt64(source_page_id), "register_passthrough_recompute!"))
        return Vector{UInt8}(transform(Vector{UInt8}(source.data)))
    end)
end

"""
    queue_recomputation!(state, page_id)

Ensure page_id is scheduled for recomputation exactly once.
"""
function queue_recomputation!(state::MMSBState, page_id::PageID)
    queue = _queue(state)
    page_id ∈ queue || push!(queue, page_id)
end

"""
    propagate_change!(state, changed_page_id, mode)

Propagate change notification through dependency graph.
"""
function propagate_change!(state::MMSBState, changed_page_id::PageID, 
                          mode::PropagationMode=IMMEDIATE)
    start_ns = time_ns()
    children = get_children(state.graph, changed_page_id)
    isempty(children) && return
    for (child_id, edge_type) in children
        handle_propagation!(state, child_id, edge_type, mode)
    end
    track_propagation_latency!(state, time_ns() - start_ns)
end

"""
    handle_propagation!(state, page_id, edge_type, mode)

Handle propagation for specific edge type.
"""
function handle_propagation!(state::MMSBState, page_id::PageID, 
                             edge_type::EdgeType, mode::PropagationMode)
    if edge_type == DATA_DEPENDENCY
        emit_event!(state, PAGE_INVALIDATED, page_id)
        mode == IMMEDIATE ? recompute_page!(state, page_id) : queue_recomputation!(state, page_id)
    elseif edge_type == CONTROL_DEPENDENCY
        mark_page_stale!(state, page_id)
    elseif edge_type == GPU_SYNC_DEPENDENCY
        schedule_gpu_sync!(state, page_id)
    elseif edge_type == COMPILER_DEPENDENCY
        invalidate_compilation!(state, page_id)
    end
end

"""
Collect all descendants of a set of pages.
"""
function _collect_descendants(state::MMSBState, page_id::PageID)::Set{PageID}
    descendants = Set{PageID}()
    queue = [page_id]
    while !isempty(queue)
        current = popfirst!(queue)
        for (child, _) in get_children(state.graph, current)
            if child ∉ descendants
                push!(descendants, child)
                push!(queue, child)
            end
        end
    end
    return descendants
end

"""
    schedule_propagation!(state, changed_pages)

Schedule batch propagation for multiple changed pages.
"""
function schedule_propagation!(state::MMSBState, changed_pages::Vector{PageID})
    affected = Set{PageID}()
    for page_id in changed_pages
        union!(affected, _collect_descendants(state, page_id))
    end
    ordered = topological_order_subset(state, collect(affected))
    for pid in ordered
        queue_recomputation!(state, pid)
    end
end

"""
    execute_propagation!(state)

Execute all queued propagation tasks.
"""
function execute_propagation!(state::MMSBState)
    queue = _queue(state)
    while !isempty(queue)
        page_id = popfirst!(queue)
        recompute_page!(state, page_id)
    end
end

"""
    recompute_page!(state, page_id)

Recompute page contents based on dependencies.
"""
function recompute_page!(state::MMSBState, page_id::PageID)
    page = get_page(state, page_id)
    page === nothing && return
    recompute_fn = get(page.metadata, :recompute_fn, nothing)
    recompute_fn === nothing && return
    baseline = Vector{UInt8}(page.data)
    new_data = recompute_fn(state, page)
    length(new_data) == page.size || throw(InvalidDeltaError("Recompute returned incorrect length", UInt64(page_id)))
    mask = Vector{Bool}(baseline .!= new_data)
    any(mask) || begin
        page.metadata[:stale] = false
        return
    end
    delta = create_delta(state, page_id, mask, new_data, :propagation)
    route_delta!(state, delta)
    page.metadata[:stale] = false
end

"""
Mark page as stale (needs recomputation on next access).
"""
function mark_page_stale!(state::MMSBState, page_id::PageID)
    page = get_page(state, page_id)
    if page !== nothing
        page.metadata[:stale] = true
        emit_event!(state, PAGE_STALE, page_id)
    end
end

"""
Schedule GPU synchronization for page.
"""
function schedule_gpu_sync!(state::MMSBState, page_id::PageID)
    page = get_page(state, page_id)
    page === nothing && return
    page.metadata[:gpu_sync_pending] = true
    emit_event!(state, GPU_SYNC_NEEDED, page_id)
end

"""
Invalidate compiled IR/artifacts for page.
"""
function invalidate_compilation!(state::MMSBState, page_id::PageID)
    page = get_page(state, page_id)
    if page !== nothing
        page.metadata[:ir_valid] = false
        emit_event!(state, IR_INVALIDATED, page_id)
    end
end

"""
    topological_order_subset(state, subset) -> Vector{PageID}

Compute topological order restricted to subset of nodes.
"""
function topological_order_subset(state::MMSBState, subset::Vector{PageID})::Vector{PageID}
    subset_set = Set(subset)
    indegree = Dict{PageID, Int}(pid => 0 for pid in subset)
    for parent in subset
        for (child, _) in get_children(state.graph, parent)
            if child ∈ subset_set
                indegree[child] = get(indegree, child, 0) + 1
            end
        end
    end
    queue = PageID[]
    for (pid, deg) in indegree
        deg == 0 && push!(queue, pid)
    end
    ordered = PageID[]
    while !isempty(queue)
        current = popfirst!(queue)
        push!(ordered, current)
        for (child, _) in get_children(state.graph, current)
            child ∈ subset_set || continue
            indegree[child] -= 1
            indegree[child] == 0 && push!(queue, child)
        end
    end
    return ordered
end

end # module PropagationEngine
