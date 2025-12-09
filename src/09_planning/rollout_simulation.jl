"""
    RolloutSimulation

Simulate plan execution for evaluation.
"""
module RolloutSimulation

export simulate_plan, parallel_rollout, evaluate_outcome

using ..PlanningTypes
using ..SearchAlgorithms

"""
    simulate_plan(plan, start_state) -> RolloutResult

Simulate execution of plan from initial state.
"""
function simulate_plan(plan::Plan, start_state::State)
    current = start_state
    trajectory = Tuple{State, Action}[]
    total_reward = 0.0
    
    for action in plan.actions
        if !SearchAlgorithms.can_apply(action, current)
            return RolloutResult(current, total_reward, trajectory, false)
        end
        
        push!(trajectory, (current, action))
        next_state = SearchAlgorithms.apply_action(action, current)
        reward = next_state.utility - action.cost
        total_reward += reward
        current = next_state
    end
    
    RolloutResult(current, total_reward, trajectory, true)
end

"""
    parallel_rollout(plans, start_state, n_rollouts) -> Dict

Run multiple rollouts in parallel for each plan.
"""
function parallel_rollout(plans::Vector{Plan}, start_state::State, n_rollouts::Int=10)
    results = Dict{UInt64, Vector{RolloutResult}}()
    
    for plan in plans
        plan_results = RolloutResult[]
        for _ in 1:n_rollouts
            result = simulate_plan(plan, start_state)
            push!(plan_results, result)
        end
        results[plan.id] = plan_results
    end
    
    results
end

"""
    evaluate_outcome(result) -> Float64

Score rollout result.
"""
function evaluate_outcome(result::RolloutResult)
    success_bonus = result.success ? 10.0 : 0.0
    length_penalty = length(result.trajectory) * 0.1
    result.total_reward + success_bonus - length_penalty
end

end # module
