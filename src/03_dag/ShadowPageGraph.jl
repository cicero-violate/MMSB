# src/types/ShadowPageGraph.jl
"""
ShadowPageGraph - Dependency graph for page relationships

Tracks which pages depend on other pages.
Used for propagation when pages change.
"""
module GraphTypes

using ..PageTypes: PageID
using ..ErrorTypes: GraphCycleError

export ShadowPageGraph, EdgeType, add_dependency!, remove_dependency!,
       get_children, get_parents, has_cycle, topological_sort

"""
Edge types for different dependency relationships.
"""
@enum EdgeType begin
    DATA_DEPENDENCY      # Page B needs data from Page A
    CONTROL_DEPENDENCY   # Page B's computation depends on Page A
    GPU_SYNC_DEPENDENCY  # GPU buffer depends on CPU page
    COMPILER_DEPENDENCY  # IR page depends on source page
end

"""
    ShadowPageGraph

Directed acyclic graph of page dependencies.
"""
mutable struct ShadowPageGraph
    deps::Dict{PageID, Vector{Tuple{PageID, EdgeType}}}
    reverse_deps::Dict{PageID, Vector{Tuple{PageID, EdgeType}}}
    lock::ReentrantLock
    
    function ShadowPageGraph()
        return new(
            Dict{PageID, Vector{Tuple{PageID, EdgeType}}}(),
            Dict{PageID, Vector{Tuple{PageID, EdgeType}}}(),
            ReentrantLock(),
        )
    end
end

"""
Ensure a node exists in both adjacency dictionaries.
"""
function _ensure_vertex!(graph::ShadowPageGraph, node::PageID)
    if !haskey(graph.deps, node)
        graph.deps[node] = Vector{Tuple{PageID, EdgeType}}()
    end
    if !haskey(graph.reverse_deps, node)
        graph.reverse_deps[node] = Vector{Tuple{PageID, EdgeType}}()
    end
end

"""
    add_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID, 
                    edge_type::EdgeType)

Add directed edge from parent to child.
"""
function add_dependency!(graph::ShadowPageGraph, parent::PageID, 
                         child::PageID, edge_type::EdgeType)
    parent == child && throw(GraphCycleError(UInt64(parent), UInt64(child)))
    lock(graph.lock) do
        _ensure_vertex!(graph, parent)
        _ensure_vertex!(graph, child)
        edge = (child, edge_type)
        if edge ∉ graph.deps[parent]
            push!(graph.deps[parent], edge)
            push!(graph.reverse_deps[child], (parent, edge_type))
            if has_cycle(graph)
                remove_dependency!(graph, parent, child)
                throw(GraphCycleError(UInt64(parent), UInt64(child)))
            end
        end
    end
end

"""
    remove_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID)

Remove edge between parent and child.
"""
function remove_dependency!(graph::ShadowPageGraph, parent::PageID, child::PageID)
    lock(graph.lock) do
        if haskey(graph.deps, parent)
            filter!(edge -> edge[1] != child, graph.deps[parent])
        end
        if haskey(graph.reverse_deps, child)
            filter!(edge -> edge[1] != parent, graph.reverse_deps[child])
        end
    end
end

"""
    get_children(graph::ShadowPageGraph, parent::PageID) -> Vector{Tuple{PageID, EdgeType}}

Get all pages that depend on parent.
"""
function get_children(graph::ShadowPageGraph, parent::PageID)::Vector{Tuple{PageID, EdgeType}}
    lock(graph.lock) do
        return copy(get(graph.deps, parent, Tuple{PageID, EdgeType}[]))
    end
end

"""
    get_parents(graph::ShadowPageGraph, child::PageID) -> Vector{Tuple{PageID, EdgeType}}

Get all pages that child depends on.
"""
function get_parents(graph::ShadowPageGraph, child::PageID)::Vector{Tuple{PageID, EdgeType}}
    lock(graph.lock) do
        return copy(get(graph.reverse_deps, child, Tuple{PageID, EdgeType}[]))
    end
end

"""
Depth-first traversal used for cycle detection.
"""
function _dfs_has_cycle(graph::ShadowPageGraph, node::PageID, visited::Dict{PageID, Symbol})
    state = get(visited, node, :unvisited)
    state == :visiting && return true
    state == :visited && return false
    visited[node] = :visiting
    for (child, _) in get(graph.deps, node, Tuple{PageID, EdgeType}[])
        if _dfs_has_cycle(graph, child, visited)
            return true
        end
    end
    visited[node] = :visited
    return false
end

"""
    has_cycle(graph::ShadowPageGraph) -> Bool

Check if graph contains cycles (should always be false).
"""
function has_cycle(graph::ShadowPageGraph)::Bool
    visited = Dict{PageID, Symbol}()
    for node in keys(graph.deps)
        if _dfs_has_cycle(graph, node, visited)
            return true
        end
    end
    return false
end

"""
Gather all vertices present in the graph.
"""
function _all_vertices(graph::ShadowPageGraph)
    nodes = Set{PageID}()
    for key in keys(graph.deps)
        push!(nodes, key)
    end
    for key in keys(graph.reverse_deps)
        push!(nodes, key)
    end
    return collect(nodes)
end

"""
    topological_sort(graph::ShadowPageGraph) -> Vector{PageID}

Return topologically sorted page IDs.
"""
function topological_sort(graph::ShadowPageGraph)::Vector{PageID}
    lock(graph.lock) do
        nodes = _all_vertices(graph)
        indegree = Dict{PageID, Int}(node => 0 for node in nodes)
        for (node, parents) in graph.reverse_deps
            indegree[node] = get(indegree, node, 0) + length(parents)
        end
        queue = PageID[]
        for (node, deg) in indegree
            if deg == 0
                push!(queue, node)
            end
        end
        order = PageID[]
        while !isempty(queue)
            current = popfirst!(queue)
            push!(order, current)
            for (child, _) in get(graph.deps, current, Tuple{PageID, EdgeType}[])
                indegree[child] = get(indegree, child, 0) - 1
                if indegree[child] == 0
                    push!(queue, child)
                end
            end
        end
        length(order) == length(nodes) || throw(GraphCycleError(0, 0))
        return order
    end
end

end # module GraphTypes
