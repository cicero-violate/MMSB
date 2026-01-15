# src/02_runtime/DeltaRouter.jl
"""
DeltaRouter - Routes deltas to pages and triggers propagation

Central dispatch mechanism for applying state changes.
Coordinates between CPU/GPU and triggers dependency updates.
"""
module DeltaRouter

using Base: time_ns
using ..PageTypes: Page, PageID
using ..DeltaTypes: Delta, set_intent_metadata!
using ..MMSBStateTypes: MMSBState, allocate_delta_id!, get_page
using ..ErrorTypes: InvalidDeltaError
using ..GraphTypes: get_children
using ..EventSystem: emit_event!, PAGE_CHANGED, PAGE_INVALIDATED, DELTA_APPLIED
using ..ErrorTypes: PageNotFoundError
using ..Monitoring: track_delta_latency!
using ..TLog: append_to_log!
using ..FFIWrapper

export route_delta!, create_delta, batch_route_deltas!

"""
    route_delta!(state::MMSBState, delta::Delta)

Main entry point for delta application.
"""
function route_delta!(state::MMSBState, delta::Delta; propagate::Bool=true)
    page = get_page(state, delta.page_id)
    page === nothing && throw(PageNotFoundError(UInt64(delta.page_id), "route_delta!"))
    start_ns = time_ns()
    GC.@preserve page delta begin
        FFIWrapper.rust_delta_apply!(page.handle, delta.handle)
    end
    # Track epoch in metadata
    page.metadata[:epoch_dirty] = delta.epoch
    append_to_log!(state, delta)
    track_delta_latency!(state, time_ns() - start_ns)
    emit_event!(state, DELTA_APPLIED, delta.page_id, delta.id, delta.epoch)
    propagate && propagate_change!(state, delta.page_id)
    emit_event!(state, PAGE_CHANGED, delta.page_id, delta.epoch)
    return page
end

"""
Construct a new delta with allocated ID.
"""
function create_delta(state::MMSBState, page_id::PageID,
                      mask::AbstractVector{Bool}, data::AbstractVector{UInt8};
                      source::Symbol=:router,
                      intent_metadata::Union{Nothing,AbstractString,Dict{Symbol,Any}}=nothing)::Delta
    page = get_page(state, page_id)
    page === nothing && throw(PageNotFoundError(UInt64(page_id), "create_delta"))
    mask_bytes = Vector{UInt8}(undef, length(mask))
    @inbounds for i in eachindex(mask)
        mask_bytes[i] = mask[i] ? UInt8(1) : UInt8(0)
    end
    data_vec = Vector{UInt8}(data)
    length(mask_bytes) == length(data_vec) || throw(InvalidDeltaError("Mask/data length mismatch", page_id))
    delta_id = allocate_delta_id!(state)
    epoch = GC.@preserve page begin
        UInt32(FFIWrapper.rust_page_epoch(page.handle) + 1)
    end
    delta = Delta(delta_id, page_id, epoch, mask_bytes, data_vec, source)
    intent_metadata === nothing || set_intent_metadata!(delta, intent_metadata)
    return delta
end

"""
Apply multiple deltas efficiently by grouping per page.
"""
function batch_route_deltas!(state::MMSBState, deltas::Vector{Delta})
    grouped = Dict{PageID, Vector{Delta}}()
    for delta in deltas
        push!(get!(grouped, delta.page_id, Delta[]), delta)
    end
    changed_pages = Set{PageID}()
    for (page_id, delta_group) in grouped
        page = get_page(state, page_id)
        page === nothing && continue
        sorted = sort(delta_group; by = d -> d.epoch)
        for delta in sorted
            route_delta!(state, delta; propagate=false)
            push!(changed_pages, delta.page_id)
        end
    end
    isempty(changed_pages) || propagate_change!(state, collect(changed_pages))
end

function propagate_change!(state::MMSBState, changed_page_id::PageID)
    engine = getfield(parentmodule(@__MODULE__), :PropagationEngine)
    engine.propagate_change!(state, changed_page_id)
end

function propagate_change!(state::MMSBState, changed_pages::AbstractVector{PageID})
    engine = getfield(parentmodule(@__MODULE__), :PropagationEngine)
    engine.propagate_change!(state, changed_pages)
end

end # module DeltaRouter
