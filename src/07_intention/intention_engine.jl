"""
    IntentionEngine

Core intention formation mechanism for Layer 7.
"""
module IntentionEngine

export form_intention, evaluate_intention, select_best_intention

using ..IntentionTypes
using ..UtilityEngine

"""
    form_intention(utility_state, layout_state) -> Intention

Form new intention based on utility gradient.
"""
function form_intention(utility_state, layout_state, id::UInt64)
    # Analyze utility trend to determine priority
    trend = UtilityEngine.utility_trend(utility_state)
    
    priority = if trend == :degrading
        CRITICAL
    elseif trend == :stable && utility_state.current_utility < -100
        HIGH
    else
        MEDIUM
    end
    
    # Estimate utility gain from reorganization
    expected_gain = abs(utility_state.current_utility) * 0.1
    
    target_pages = collect(keys(layout_state.placement))
    
    Intention(
        id,
        "Optimize memory layout",
        priority,
        expected_gain,
        target_pages,
        time()
    )
end

"""
    evaluate_intention(intention::Intention, current_utility::Float64) -> Float64

Evaluate intention quality score.
"""
function evaluate_intention(intention::Intention, current_utility::Float64)
    # Score = priority × expected_gain - cost_of_change
    priority_weight = Float64(intention.priority)
    base_score = priority_weight * intention.expected_utility_gain
    
    # Cost of applying intention
    change_cost = length(intention.target_pages) * 0.1
    
    return base_score - change_cost
end

"""
    select_best_intention(intentions::Vector{Intention}, utility::Float64) -> Union{Intention, Nothing}

Select highest-scoring intention from candidates.
"""
function select_best_intention(intentions::Vector{Intention}, utility::Float64)
    isempty(intentions) && return nothing
    
    scores = [evaluate_intention(i, utility) for i in intentions]
    best_idx = argmax(scores)
    
    return intentions[best_idx]
end

end # module IntentionEngine
