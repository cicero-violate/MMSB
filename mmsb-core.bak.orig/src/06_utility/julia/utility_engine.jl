"""
    UtilityEngine

Core utility computation engine for Layer 6.

Aggregates cost components into a scalar utility value that guides
Layer 7 intention formation and system optimization decisions.
"""
module UtilityEngine

export UtilityState, compute_utility, update_utility!

using ..CostFunctions

"""
    UtilityState

Current utility state with cost history and weights.
"""
mutable struct UtilityState
    current_utility::Float64
    cost_weights::Dict{Symbol, Float64}
    history::Vector{Float64}
    max_history::Int
end

function UtilityState(max_history::Int = 100)
    UtilityState(
        0.0,
        Dict(:cache => 1.0, :memory => 0.5, :latency => 2.0),
        Float64[],
        max_history
    )
end

"""
    compute_utility(costs::CostComponents, weights::Dict) -> Float64

Compute utility as negative weighted sum of costs:
U = -Σᵢ wᵢ × cᵢ

Higher utility = better performance (lower costs).
"""
function compute_utility(costs::CostComponents, weights::Dict{Symbol, Float64})
    utility = 0.0
    utility -= weights[:cache] * costs.cache_miss_rate
    utility -= weights[:memory] * costs.memory_overhead_mb
    utility -= weights[:latency] * costs.avg_latency_us
    
    return utility
end

"""
    update_utility!(state::UtilityState, costs::CostComponents)

Update utility state with new cost measurements.
"""
function update_utility!(state::UtilityState, costs::CostComponents)
    new_utility = compute_utility(costs, state.cost_weights)
    state.current_utility = new_utility
    
    push!(state.history, new_utility)
    if length(state.history) > state.max_history
        popfirst!(state.history)
    end
    
    return new_utility
end

"""
    utility_trend(state::UtilityState) -> Symbol

Analyze utility trend: :improving, :degrading, or :stable
"""
function utility_trend(state::UtilityState)
    length(state.history) < 10 && return :unknown
    
    recent = @view state.history[end-9:end]
    first_half = mean(@view recent[1:5])
    second_half = mean(@view recent[6:10])
    
    diff = second_half - first_half
    threshold = 0.05 * abs(first_half)
    
    if diff > threshold
        return :improving
    elseif diff < -threshold
        return :degrading
    else
        return :stable
    end
end

end # module UtilityEngine
