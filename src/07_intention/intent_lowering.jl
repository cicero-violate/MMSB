"""
    IntentLowering

Helpers to lower high-level intents into delta specifications for FFI.
"""
module IntentLowering

using ..UpsertPlan

export lower_intent_to_deltaspec, mask_to_bytes

function mask_to_bytes(mask::Vector{Bool})
    UInt8[m ? 1 : 0 for m in mask]
end

function lower_intent_to_deltaspec(plan::UpsertPlan.UpsertPlan)
    UpsertPlan.validate_plan(plan)
    (; page = plan.deltaspec.target_page,
     payload = plan.deltaspec.payload,
     mask_bytes = mask_to_bytes(plan.deltaspec.mask),
     metadata = plan.metadata)
end

end # module IntentLowering
