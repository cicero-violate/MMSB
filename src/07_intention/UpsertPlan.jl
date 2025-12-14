"""
    UpsertPlan

Structured intent representation for conditional upserts.
"""
module UpsertPlan

export DeltaSpec, UpsertPlan, validate_plan

struct DeltaSpec
    target_page::UInt64
    payload::Vector{UInt8}
    mask::Vector{Bool}
end

struct UpsertPlan
    query::String
    predicate::Function
    deltaspec::DeltaSpec
    metadata::Dict{Symbol, Any}
end

function validate_plan(plan::UpsertPlan)
    isempty(plan.query) && error("UpsertPlan query must be non-empty")
    length(plan.deltaspec.payload) == length(plan.deltaspec.mask) ||
        error("Payload/mask length mismatch in UpsertPlan")
    plan
end

end # module UpsertPlan
