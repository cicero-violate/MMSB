"""
    ReasoningTypes

Type definitions for Layer 8 reasoning engine.
"""
module ReasoningTypes

export Constraint, ConstraintType, Dependency, DependencyType
export Pattern, PatternMatch, Rule, RuleType, Inference
export InferenceResult, ReasoningState

@enum ConstraintType begin
    TYPE_CONSTRAINT
    VALUE_CONSTRAINT
    ORDERING_CONSTRAINT
    RESOURCE_CONSTRAINT
end

struct Constraint
    type::ConstraintType
    predicate::Function
    satisfied::Bool
    metadata::Dict{Symbol, Any}
end

@enum DependencyType begin
    DATA_DEPENDENCY
    CONTROL_DEPENDENCY
    RESOURCE_DEPENDENCY
    TEMPORAL_DEPENDENCY
end

struct Dependency
    type::DependencyType
    source::UInt64
    target::UInt64
    strength::Float64
    metadata::Dict{Symbol, Any}
end

struct Pattern
    id::UInt64
    nodes::Vector{UInt64}
    edges::Vector{Tuple{UInt64, UInt64}}
    signature::Vector{UInt8}
    frequency::Int
end

struct PatternMatch
    pattern_id::UInt64
    matched_nodes::Vector{UInt64}
    confidence::Float64
end

@enum RuleType begin
    INFERENCE_RULE
    TRANSFORMATION_RULE
    OPTIMIZATION_RULE
    SAFETY_RULE
end

struct Rule
    id::UInt64
    type::RuleType
    condition::Function
    action::Function
    priority::Int
end

struct Inference
    node_id::UInt64
    inferred_constraints::Vector{Constraint}
    confidence::Float64
    derivation::Vector{UInt64}
end

struct InferenceResult
    inferences::Vector{Inference}
    new_constraints::Dict{UInt64, Vector{Constraint}}
    propagated::Set{UInt64}
    patterns_found::Vector{PatternMatch}
end

mutable struct ReasoningState
    constraints::Dict{UInt64, Vector{Constraint}}
    dependencies::Dict{Tuple{UInt64, UInt64}, Dependency}
    patterns::Dict{UInt64, Pattern}
    rules::Vector{Rule}
    inference_cache::Dict{UInt64, Vector{Inference}}
end

function ReasoningState()
    ReasoningState(
        Dict{UInt64, Vector{Constraint}}(),
        Dict{Tuple{UInt64, UInt64}, Dependency}(),
        Dict{UInt64, Pattern}(),
        Rule[],
        Dict{UInt64, Vector{Inference}}()
    )
end

end # module
