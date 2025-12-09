"""
    SearchAlgorithms

Search algorithms for planning (A*, MCTS, etc).
"""
module SearchAlgorithms

export astar_search, mcts_search, beam_search, compute_heuristic

using ..PlanningTypes

"""
    astar_search(start_state, goal, actions, max_nodes) -> Union{Plan, Nothing}

A* search for optimal plan.
"""
function astar_search(start_state::State, goal::Goal, actions::Vector{Action}, max_nodes::Int=1000)
    start_node = SearchNode(start_state, nothing, nothing, 0.0, 0.0, 0.0)
    start_node.h_cost = compute_heuristic(start_state, goal)
    start_node.f_cost = start_node.g_cost + start_node.h_cost
    
    open_list = [start_node]
    closed_set = Set{UInt64}()
    nodes_expanded = 0
    
    while !isempty(open_list) && nodes_expanded < max_nodes
        # Get node with lowest f_cost
        current_idx = argmin([n.f_cost for n in open_list])
        current = open_list[current_idx]
        deleteat!(open_list, current_idx)
        
        # Check goal
        if goal.predicate(current.state)
            return reconstruct_plan(current, goal.id)
        end
        
        push!(closed_set, current.state.id)
        nodes_expanded += 1
        
        # Expand
        for action in actions
            if can_apply(action, current.state)
                new_state = apply_action(action, current.state)
                
                if new_state.id in closed_set
                    continue
                end
                
                g = current.g_cost + action.cost
                h = compute_heuristic(new_state, goal)
                
                child = SearchNode(new_state, current, action, g, h, g + h)
                push!(open_list, child)
            end
        end
    end
    
    nothing  # No plan found
end

"""
    mcts_search(start_state, goal, actions, iterations) -> Plan

Monte Carlo Tree Search.
"""
function mcts_search(start_state::State, goal::Goal, actions::Vector{Action}, iterations::Int=1000)
    root = MCTSNode(start_state)
    
    for _ in 1:iterations
        # Selection
        node = select_node(root)
        
        # Expansion
        if !is_terminal(node, goal) && node.visits > 0
            node = expand_node(node, actions)
        end
        
        # Simulation
        reward = simulate(node.state, goal, actions, 20)
        
        # Backpropagation
        backpropagate(node, reward)
    end
    
    # Extract best plan
    extract_plan_from_mcts(root, goal.id)
end

mutable struct MCTSNode
    state::State
    parent::Union{MCTSNode, Nothing}
    children::Vector{MCTSNode}
    action::Union{Action, Nothing}
    visits::Int
    total_reward::Float64
end

MCTSNode(state::State) = MCTSNode(state, nothing, MCTSNode[], nothing, 0, 0.0)

function select_node(node::MCTSNode)
    while !isempty(node.children)
        node = best_uct_child(node)
    end
    node
end

function best_uct_child(node::MCTSNode, c::Float64=1.414)
    best_child = node.children[1]
    best_score = -Inf
    
    for child in node.children
        if child.visits == 0
            return child
        end
        
        exploit = child.total_reward / child.visits
        explore = c * sqrt(log(node.visits) / child.visits)
        score = exploit + explore
        
        if score > best_score
            best_score = score
            best_child = child
        end
    end
    
    best_child
end

function expand_node(node::MCTSNode, actions::Vector{Action})
    for action in actions
        if can_apply(action, node.state)
            new_state = apply_action(action, node.state)
            child = MCTSNode(new_state)
            child.parent = node
            child.action = action
            push!(node.children, child)
            return child
        end
    end
    node
end

function simulate(state::State, goal::Goal, actions::Vector{Action}, max_depth::Int)
    current = state
    reward = 0.0
    
    for _ in 1:max_depth
        if goal.predicate(current)
            return reward + 10.0
        end
        
        applicable = filter(a -> can_apply(a, current), actions)
        if isempty(applicable)
            break
        end
        
        action = rand(applicable)
        current = apply_action(action, current)
        reward += current.utility - action.cost
    end
    
    reward
end

function backpropagate(node::MCTSNode, reward::Float64)
    current = node
    while current !== nothing
        current.visits += 1
        current.total_reward += reward
        current = current.parent
    end
end

"""
    compute_heuristic(state, goal) -> Float64

Heuristic estimate of cost to goal.
"""
function compute_heuristic(state::State, goal::Goal)
    # Simple heuristic: utility difference
    if haskey(state.features, :target_utility)
        return abs(state.utility - state.features[:target_utility])
    end
    1.0
end

function can_apply(action::Action, state::State)
    all(pred -> pred(state), action.preconditions)
end

function apply_action(action::Action, state::State)
    new_features = copy(state.features)
    new_utility = state.utility
    
    for effect in action.effects
        result = effect(new_features, new_utility)
        if isa(result, Tuple)
            new_features, new_utility = result
        end
    end
    
    State(state.id + 1, new_features, new_utility)
end

function is_terminal(node::MCTSNode, goal::Goal)
    goal.predicate(node.state)
end

function reconstruct_plan(node::SearchNode, goal_id::UInt64)
    actions = Action[]
    current = node
    cost = node.g_cost
    
    while current.parent !== nothing
        if current.action !== nothing
            pushfirst!(actions, current.action)
        end
        current = current.parent
    end
    
    Plan(UInt64(1), goal_id, actions, cost, node.state.utility, 0.9)
end

function extract_plan_from_mcts(root::MCTSNode, goal_id::UInt64)
    actions = Action[]
    current = root
    
    while !isempty(current.children)
        current = best_uct_child(current, 0.0)  # Pure exploitation
        if current.action !== nothing
            push!(actions, current.action)
        end
    end
    
    Plan(UInt64(1), goal_id, actions, 0.0, current.state.utility, 0.8)
end

end # module
