"""
    IntentionTypes

Core types for Layer 7 intention system.
"""
module IntentionTypes

export Intention, Goal, IntentionPriority, IntentionState

@enum IntentionPriority LOW=1 MEDIUM=2 HIGH=3 CRITICAL=4

"""
    Intention

Discrete intention to modify system state in specific direction.
"""
struct Intention
    id::UInt64
    description::String
    priority::IntentionPriority
    expected_utility_gain::Float64
    target_pages::Vector{UInt64}
    timestamp::Float64
end

"""
    Goal

Emergent goal representing desired system state or property.
"""
struct Goal
    id::UInt64
    description::String
    utility_threshold::Float64
    attractors::Vector{UInt64}  # Attractor state IDs
    structural_constraints::Dict{Symbol, Any}
end

"""
    IntentionState

Current state of intention system.
"""
mutable struct IntentionState
    active_intentions::Vector{Intention}
    emergent_goals::Vector{Goal}
    attractor_states::Dict{UInt64, Vector{Float64}}
    last_update::Float64
end

function IntentionState()
    IntentionState(
        Intention[],
        Goal[],
        Dict{UInt64, Vector{Float64}}(),
        time()
    )
end

end # module IntentionTypes
