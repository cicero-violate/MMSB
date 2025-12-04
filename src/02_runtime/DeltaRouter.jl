# src/02_runtime/DeltaRouter.jl
"""
DeltaRouter - Routes deltas to pages and triggers propagation

Central dispatch mechanism for applying state changes.
Coordinates between CPU/GPU and triggers dependency updates.
"""
module DeltaRouter

using CUDA
using Base: time_ns
using ..PageTypes: Page, PageID, is_gpu_page
using ..DeltaTypes: Delta, delta_byte_count, dense_data, is_sparse_delta
using ..MMSBStateTypes: MMSBState, allocate_delta_id!, get_page
using ..GraphTypes: get_children
using ..EventSystem: emit_event!, PAGE_CHANGED, PAGE_INVALIDATED, DELTA_APPLIED
using ..GPUKernels: launch_delta_merge!
using ..ErrorTypes: PageNotFoundError, InvalidDeltaError
using ..Monitoring: track_delta_latency!

export route_delta!, create_delta, apply_delta_to_page!, batch_route_deltas!

"""
    route_delta!(state::MMSBState, delta::Delta)

Main entry point for delta application.
"""
function route_delta!(state::MMSBState, delta::Delta)
    page = get_page(state, delta.page_id)
    page === nothing && throw(PageNotFoundError(UInt64(delta.page_id), "route_delta!"))
    validate_delta(page, delta) || throw(InvalidDeltaError("Validation failed for page $(page.id)", UInt64(page.id)))
    lock(state.lock) do
        push!(state.tlog, delta)
    end
    start_ns = time_ns()
    if is_gpu_page(page)
        route_delta_gpu!(page, delta)
    else
        route_delta_cpu!(page, delta)
    end
    track_delta_latency!(state, time_ns() - start_ns)
    emit_event!(state, DELTA_APPLIED, delta.page_id, delta.id, delta.epoch)
    propagate_change!(state, delta.page_id)
    emit_event!(state, PAGE_CHANGED, delta.page_id, delta.epoch)
    return page
end

"""
Construct a new delta with allocated ID.
"""
function create_delta(state::MMSBState, page_id::PageID, 
                      mask::AbstractVector{Bool}, data::AbstractVector{UInt8}, 
                     source::Symbol)::Delta
    mask_vec = collect(mask)
    data_vec = Vector{UInt8}(data)
    length(mask_vec) == length(data_vec) || throw(InvalidDeltaError("Mask/data length mismatch", UInt64(page_id)))
    page = get_page(state, page_id)
    page === nothing && throw(PageNotFoundError(UInt64(page_id), "create_delta"))
    delta_id = allocate_delta_id!(state)
    epoch = page.epoch + UInt32(1)
    return Delta(delta_id, page_id, epoch, mask_vec, data_vec, source)
end

"""
Apply delta to CPU page in-place.
"""
function route_delta_cpu!(page::Page, delta::Delta)
    sparse = is_sparse_delta(delta)
    data_idx = 1
    @inbounds for i in eachindex(delta.mask)
        if delta.mask[i]
            value = sparse ? delta.data[data_idx] : delta.data[i]
            page.data[i] = value
            page.mask[i] = true
            sparse && (data_idx += 1)
        end
    end
    page.epoch = delta.epoch
    page.metadata[:last_modified] = time_ns()
end

"""
Apply delta to GPU page via CUDA kernel.
"""
function route_delta_gpu!(page::Page, delta::Delta)
    gpu_mask = CuArray(delta.mask)
    dense = dense_data(delta)
    gpu_data = CuArray(dense)
    launch_delta_merge!(page.data, gpu_mask, gpu_data)
    CUDA.synchronize()
    page.mask .= gpu_mask
    page.epoch = delta.epoch
end

"""
Low-level delta application (used by replay engine).
"""
function apply_delta_to_page!(page::Page, delta::Delta)
    is_gpu_page(page) ? route_delta_gpu!(page, delta) : route_delta_cpu!(page, delta)
end

"""
Apply multiple deltas efficiently by grouping per page.
"""
function batch_route_deltas!(state::MMSBState, deltas::Vector{Delta})
    grouped = Dict{PageID, Vector{Delta}}()
    for delta in deltas
        push!(get!(grouped, delta.page_id, Delta[]), delta)
    end
    for (page_id, delta_group) in grouped
        page = get_page(state, page_id)
        page === nothing && continue
        sorted = sort(delta_group; by = d -> d.epoch)
        for delta in sorted
            route_delta!(state, delta)
        end
    end
end

"""
Check if delta is valid for page.
"""
function validate_delta(page::Page, delta::Delta)::Bool
    data_len = length(delta.data)
    changed = delta_byte_count(delta)
    return page.id == delta.page_id &&
           length(delta.mask) == page.size &&
           (data_len == page.size || data_len == changed) &&
           delta.epoch == page.epoch + 1
end

function propagate_change!(state::MMSBState, changed_page_id::PageID)
    engine = getfield(parentmodule(@__MODULE__), :PropagationEngine)
    engine.propagate_change!(state, changed_page_id)
end

end # module DeltaRouter
