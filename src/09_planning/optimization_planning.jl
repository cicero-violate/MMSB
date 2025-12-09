"""
    OptimizationPlanning

Gradient-based planning optimization with Enzyme.jl support.
"""
module OptimizationPlanning

export optimize_plan, gradient_descent_planning, prepare_for_enzyme

using ..PlanningTypes

"""
    optimize_plan(plan, objective) -> Plan

Optimize plan parameters to maximize objective.
"""
function optimize_plan(plan::Plan, objective::Function)
    # Extract parameters
    params = extract_parameters(plan)
    
    # Gradient descent
    α = 0.01
    for iter in 1:100
        grad = compute_gradient(objective, params)
        params -= α * grad
        
        if norm(grad) < 1e-6
            break
        end
    end
    
    reconstruct_plan(plan, params)
end

function extract_parameters(plan::Plan)
    # Extract numeric parameters (action costs, timings, etc)
    params = Float64[]
    for action in plan.actions
        push!(params, action.cost)
    end
    params
end

function compute_gradient(f::Function, x::Vector{Float64})
    # Finite differences (Enzyme.jl would be used for AD)
    h = 1e-7
    grad = zeros(length(x))
    
    for i in 1:length(x)
        x_plus = copy(x)
        x_plus[i] += h
        grad[i] = (f(x_plus) - f(x)) / h
    end
    
    grad
end

norm(x::Vector{Float64}) = sqrt(sum(x .^ 2))

function reconstruct_plan(plan::Plan, params::Vector{Float64})
    new_actions = Action[]
    for (i, action) in enumerate(plan.actions)
        new_action = Action(
            action.id,
            action.name,
            action.preconditions,
            action.effects,
            params[i]
        )
        push!(new_actions, new_action)
    end
    
    Plan(
        plan.id,
        plan.goal_id,
        new_actions,
        sum(params),
        plan.expected_utility,
        plan.confidence
    )
end

"""
    gradient_descent_planning(initial_state, goal, α, iterations) -> Plan

Plan via gradient descent in action space.
"""
function gradient_descent_planning(initial_state::State, goal::Goal, α::Float64=0.01, iterations::Int=100)
    # Initialize action sequence
    action_params = randn(10, 3)  # 10 actions, 3 params each
    
    for iter in 1:iterations
        # Compute objective
        obj = evaluate_action_sequence(action_params, initial_state, goal)
        
        # Gradient (simplified - would use Enzyme.jl)
        grad = compute_sequence_gradient(action_params, initial_state, goal)
        
        # Update
        action_params -= α * grad
    end
    
    # Convert to plan
    actions_from_params(action_params)
end

function evaluate_action_sequence(params::Matrix{Float64}, state::State, goal::Goal)
    # Simulate execution and measure utility
    utility = state.utility
    for i in 1:size(params, 1)
        # Apply action effect
        utility += params[i, 1] - abs(params[i, 2])
    end
    
    # Penalty if goal not reached
    goal_penalty = goal.predicate(State(state.id, state.features, utility)) ? 0.0 : -10.0
    utility + goal_penalty
end

function compute_sequence_gradient(params::Matrix{Float64}, state::State, goal::Goal)
    # Finite differences for gradient
    grad = zeros(size(params))
    h = 1e-6
    
    for i in 1:size(params, 1)
        for j in 1:size(params, 2)
            params_plus = copy(params)
            params_plus[i, j] += h
            
            grad[i, j] = (evaluate_action_sequence(params_plus, state, goal) - 
                         evaluate_action_sequence(params, state, goal)) / h
        end
    end
    
    grad
end

function actions_from_params(params::Matrix{Float64})
    actions = Action[]
    for i in 1:size(params, 1)
        push!(actions, Action(
            UInt64(i),
            "opt_action_$i",
            Function[],
            Function[],
            abs(params[i, 2])
        ))
    end
    
    Plan(UInt64(1), UInt64(1), actions, sum(abs.(params[:, 2])), 0.0, 0.7)
end

"""
    prepare_for_enzyme(plan) -> Dict

Prepare plan data structures for Enzyme.jl AD.
"""
function prepare_for_enzyme(plan::Plan)
    # Enzyme.jl requires mutable structures and proper type annotations
    # This function prepares data in enzyme-compatible format
    
    Dict{Symbol, Any}(
        :action_costs => [a.cost for a in plan.actions],
        :expected_utility => plan.expected_utility,
        :plan_length => length(plan.actions)
    )
end

end # module
