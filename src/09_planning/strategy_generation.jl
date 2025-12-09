"""
    StrategyGeneration

Generate planning strategies for different scenarios.
"""
module StrategyGeneration

export generate_strategies, select_strategy, adapt_strategy

using ..PlanningTypes
using ..SearchAlgorithms
using ..GoalDecomposition

"""
    generate_strategies(goal, state) -> Vector{Strategy}

Generate candidate strategies for achieving goal.
"""
function generate_strategies(goal::Goal, state::State)
    strategies = Strategy[]
    
    # Strategy 1: Direct A* search
    push!(strategies, Strategy(
        UInt64(1),
        "Direct A*",
        (g, s, acts) -> SearchAlgorithms.astar_search(s, g, acts, 1000),
        (plan) -> plan !== nothing ? plan.expected_utility / plan.expected_cost : 0.0
    ))
    
    # Strategy 2: MCTS exploration
    push!(strategies, Strategy(
        UInt64(2),
        "MCTS",
        (g, s, acts) -> SearchAlgorithms.mcts_search(s, g, acts, 500),
        (plan) -> plan.confidence
    ))
    
    # Strategy 3: Goal decomposition + sequential planning
    push!(strategies, Strategy(
        UInt64(3),
        "Hierarchical",
        (g, s, acts) -> hierarchical_planning(g, s, acts),
        (plan) -> plan.actions |> length |> x -> 1.0 / (1.0 + x)
    ))
    
    strategies
end

function hierarchical_planning(goal::Goal, state::State, actions::Vector{Action})
    subgoals = GoalDecomposition.decompose_goal(goal, state)
    all_actions = Action[]
    total_cost = 0.0
    
    current_state = state
    for subgoal in subgoals
        subplan = SearchAlgorithms.astar_search(current_state, subgoal, actions, 500)
        if subplan !== nothing
            append!(all_actions, subplan.actions)
            total_cost += subplan.expected_cost
            # Simulate state after subplan
            for action in subplan.actions
                current_state = SearchAlgorithms.apply_action(action, current_state)
            end
        end
    end
    
    Plan(UInt64(1), goal.id, all_actions, total_cost, current_state.utility, 0.7)
end

"""
    select_strategy(strategies, goal, state) -> Strategy

Select best strategy for current situation.
"""
function select_strategy(strategies::Vector{Strategy}, goal::Goal, state::State)
    # Score strategies based on goal characteristics
    scores = Float64[]
    
    for strategy in strategies
        score = 0.0
        
        # Prefer A* for simple goals
        if strategy.name == "Direct A*" && goal.priority < 0.5
            score += 1.0
        end
        
        # Prefer MCTS for complex exploration
        if strategy.name == "MCTS" && goal.priority > 0.8
            score += 1.5
        end
        
        # Prefer hierarchical for compound goals
        if strategy.name == "Hierarchical" && !isempty(goal.subgoals)
            score += 2.0
        end
        
        push!(scores, score)
    end
    
    strategies[argmax(scores)]
end

"""
    adapt_strategy(strategy, feedback) -> Strategy

Adapt strategy based on execution feedback.
"""
function adapt_strategy(strategy::Strategy, feedback::Dict{Symbol, Any})
    # If strategy underperformed, modify evaluation
    if get(feedback, :success, true) == false
        new_eval = (plan) -> strategy.evaluation_fn(plan) * 0.8
        return Strategy(
            strategy.id,
            strategy.name * " (adapted)",
            strategy.plan_generator,
            new_eval
        )
    end
    
    strategy
end

end # module
