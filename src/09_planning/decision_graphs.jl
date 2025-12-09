"""
    DecisionGraphs

Build and analyze decision graphs for planning.
"""
module DecisionGraphs

export build_decision_graph, find_optimal_path, prune_graph

using ..PlanningTypes

"""
    build_decision_graph(state, goal, actions, depth) -> DecisionGraph

Construct decision graph up to given depth.
"""
function build_decision_graph(state::State, goal::Goal, actions::Vector{Action}, depth::Int)
    graph = DecisionGraph(
        Dict{UInt64, State}(),
        Dict{Tuple{UInt64, UInt64}, Action}(),
        Dict{UInt64, Float64}()
    )
    
    graph.nodes[state.id] = state
    graph.values[state.id] = state.utility
    
    expand_graph!(graph, state, goal, actions, depth)
    graph
end

function expand_graph!(graph::DecisionGraph, state::State, goal::Goal, actions::Vector{Action}, depth::Int)
    if depth <= 0 || goal.predicate(state)
        return
    end
    
    for action in actions
        if !all(pred -> pred(state), action.preconditions)
            continue
        end
        
        next_state = apply_action_simple(action, state)
        
        if !haskey(graph.nodes, next_state.id)
            graph.nodes[next_state.id] = next_state
            graph.values[next_state.id] = next_state.utility
            graph.edges[(state.id, next_state.id)] = action
            expand_graph!(graph, next_state, goal, actions, depth - 1)
        end
    end
end

function apply_action_simple(action::Action, state::State)
    new_features = copy(state.features)
    new_utility = state.utility - action.cost * 0.1
    State(state.id + UInt64(rand(1:1000)), new_features, new_utility)
end

"""
    find_optimal_path(graph, start_id, goal) -> Vector{Action}

Find highest value path through decision graph.
"""
function find_optimal_path(graph::DecisionGraph, start_id::UInt64, goal::Goal)
    # Bellman-Ford for optimal path
    distances = Dict{UInt64, Float64}()
    predecessors = Dict{UInt64, UInt64}()
    
    for node_id in keys(graph.nodes)
        distances[node_id] = -Inf
    end
    distances[start_id] = 0.0
    
    # Relaxation
    for _ in 1:(length(graph.nodes) - 1)
        for ((src, dst), action) in graph.edges
            new_dist = distances[src] + graph.values[dst] - action.cost
            if new_dist > distances[dst]
                distances[dst] = new_dist
                predecessors[dst] = src
            end
        end
    end
    
    # Find goal node
    goal_node = nothing
    for (node_id, state) in graph.nodes
        if goal.predicate(state)
            goal_node = node_id
            break
        end
    end
    
    if goal_node === nothing
        return Action[]
    end
    
    # Reconstruct path
    path = Action[]
    current = goal_node
    while haskey(predecessors, current)
        prev = predecessors[current]
        action = graph.edges[(prev, current)]
        pushfirst!(path, action)
        current = prev
    end
    
    path
end

"""
    prune_graph(graph, threshold) -> DecisionGraph

Remove low-value branches from graph.
"""
function prune_graph(graph::DecisionGraph, threshold::Float64)
    pruned = DecisionGraph(
        Dict{UInt64, State}(),
        Dict{Tuple{UInt64, UInt64}, Action}(),
        Dict{UInt64, Float64}()
    )
    
    for (node_id, state) in graph.nodes
        if graph.values[node_id] >= threshold
            pruned.nodes[node_id] = state
            pruned.values[node_id] = graph.values[node_id]
        end
    end
    
    for (edge, action) in graph.edges
        src, dst = edge
        if haskey(pruned.nodes, src) && haskey(pruned.nodes, dst)
            pruned.edges[edge] = action
        end
    end
    
    pruned
end

end # module
