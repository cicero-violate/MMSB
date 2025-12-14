"""
    IntentLowering

Helpers to lower high-level intents into delta specifications for FFI.
"""
module IntentLowering

using ..UpsertPlan: DeltaSpec, UpsertPlan, validate_plan
using ..MMSBStateTypes: MMSBState, get_page
using ..PageTypes: PageID, read_page
using ..DeltaRouter
using ..DeltaTypes: set_intent_metadata!
using ..ErrorTypes: PageNotFoundError

export lower_intent_to_deltaspec, mask_to_bytes, execute_upsert_plan!

function mask_to_bytes(mask::Vector{Bool})
    UInt8[m ? 1 : 0 for m in mask]
end

function lower_intent_to_deltaspec(plan::UpsertPlan)
    validate_plan(plan)
    (; page = plan.deltaspec.target_page,
     payload = plan.deltaspec.payload,
     mask_bytes = mask_to_bytes(plan.deltaspec.mask),
     metadata = plan.metadata)
end

function execute_upsert_plan!(state::MMSBState, plan::UpsertPlan;
                              source::Symbol=:intent)
    validate_plan(plan)
    page = get_page(state, PageID(plan.deltaspec.target_page))
    page === nothing && throw(PageNotFoundError(plan.deltaspec.target_page, "execute_upsert_plan!"))
    snapshot = read_page(page)
    predicate_passed = plan.predicate(snapshot)
    if !predicate_passed
        return (applied=false, reason=:predicate_failed, query_snapshot=snapshot, delta=nothing)
    end
    metadata = Dict{Symbol,Any}(:query => plan.query)
    for (k, v) in plan.metadata
        metadata[k] = v
    end
    delta = DeltaRouter.create_delta(state, page.id, plan.deltaspec.mask, plan.deltaspec.payload;
                                     source=source)
    metadata[:delta_id] = Int(delta.id)
    set_intent_metadata!(delta, metadata)
    DeltaRouter.route_delta!(state, delta)
    return (applied=true, reason=:ok, query_snapshot=snapshot, delta=delta)
end

end # module IntentLowering
