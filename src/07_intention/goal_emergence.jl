"""
    GoalEmergence

Goal emergence from utility landscape analysis.
"""
module GoalEmergence

export detect_goals, Goal, utility_gradient

using ..IntentionTypes

"""
    utility_gradient(utility_history::Vector{Float64}) -> Float64

Compute ∇U from recent history.
"""
function utility_gradient(utility_history::Vector{Float64})
    length(utility_history) < 2 && return 0.0
    
    recent = @view utility_history[max(1, end-9):end]
    n = length(recent)
    
    # Linear regression slope
    x_mean = (n + 1) / 2.0
    y_mean = sum(recent) / n
    
    numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in 1:n)
    denominator = sum((i - x_mean)^2 for i in 1:n)
    
    denominator == 0.0 ? 0.0 : numerator / denominator
end

"""
    detect_goals(utility_state, threshold::Float64) -> Vector{Goal}

Detect emergent goals from utility analysis.
"""
function detect_goals(utility_state, threshold::Float64)
    goals = Goal[]
    
    # Goal 1: Maintain high utility
    if utility_state.current_utility > threshold
        push!(goals, Goal(
            UInt64(1),
            "Maintain current high utility",
            threshold,
            UInt64[],
            Dict(:type => :maintenance)
        ))
    end
    
    # Goal 2: Improve if degrading
    gradient = utility_gradient(utility_state.history)
    if gradient < -0.5
        push!(goals, Goal(
            UInt64(2),
            "Reverse utility degradation",
            utility_state.current_utility + abs(gradient) * 10,
            UInt64[],
            Dict(:type => :improvement)
        ))
    end
    
    return goals
end

end # module GoalEmergence
