"""
    PlanningTypes

Type definitions for Layer 9 planning engine.
"""
module PlanningTypes

export State, Action, Goal, Plan, SearchNode
export Strategy, RolloutResult, DecisionGraph
export PlanningState, PlanMetrics

struct State
    id::UInt64
    features::Dict{Symbol, Any}
    utility::Float64
end

struct Action
    id::UInt64
    name::String
    preconditions::Vector{Function}
    effects::Vector{Function}
    cost::Float64
end

struct Goal
    id::UInt64
    description::String
    predicate::Function
    priority::Float64
    subgoals::Vector{UInt64}
end

struct Plan
    id::UInt64
    goal_id::UInt64
    actions::Vector{Action}
    expected_cost::Float64
    expected_utility::Float64
    confidence::Float64
end

mutable struct SearchNode
    state::State
    parent::Union{SearchNode, Nothing}
    action::Union{Action, Nothing}
    g_cost::Float64  # Cost from start
    h_cost::Float64  # Heuristic to goal
    f_cost::Float64  # g + h
end

struct Strategy
    id::UInt64
    name::String
    plan_generator::Function
    evaluation_fn::Function
end

struct RolloutResult
    final_state::State
    total_reward::Float64
    trajectory::Vector{Tuple{State, Action}}
    success::Bool
end

struct DecisionGraph
    nodes::Dict{UInt64, State}
    edges::Dict{Tuple{UInt64, UInt64}, Action}
    values::Dict{UInt64, Float64}
end

mutable struct PlanningState
    current_state::State
    goals::Dict{UInt64, Goal}
    available_actions::Vector{Action}
    plans::Dict{UInt64, Plan}
    decision_graph::DecisionGraph
    strategies::Vector{Strategy}
end

struct PlanMetrics
    planning_time::Float64
    nodes_expanded::Int
    plan_length::Int
    plan_quality::Float64
end

function PlanningState(initial_state::State)
    PlanningState(
        initial_state,
        Dict{UInt64, Goal}(),
        Action[],
        Dict{UInt64, Plan}(),
        DecisionGraph(
            Dict{UInt64, State}(),
            Dict{Tuple{UInt64, UInt64}, Action}(),
            Dict{UInt64, Float64}()
        ),
        Strategy[]
    )
end

end # module
