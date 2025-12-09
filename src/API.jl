# src/API.jl
"""
Public API helpers for MMSB.

`mmsb_start`/`mmsb_stop` manage MMSB states without exposing
internal constructors. `create_page`, `update_page`, and `query_page`
provide ergonomic helpers for common operations, and `@mmsb`
keeps track of the active state during a scoped block.
"""
module API

using CUDA
using ..MMSBStateTypes: MMSBState, MMSBConfig, get_page
using ..PageAllocator: create_cpu_page!, create_gpu_page!
using ..DeltaRouter: create_delta, route_delta!
using ..TLog: checkpoint_log!
using ..PageTypes: Page, PageID, PageLocation, CPU_LOCATION, GPU_LOCATION, read_page
using ..ErrorTypes: PageNotFoundError, InvalidDeltaError, GPUMemoryError, UnsupportedLocationError

export mmsb_start, mmsb_stop, create_page, update_page, query_page, @mmsb

const ACTIVE_STATE = Base.RefValue{Union{Nothing,MMSBState}}(nothing)

"""
    mmsb_start(; enable_gpu=true, enable_instrumentation=false, config=nothing)

Create a new `MMSBState` using the provided configuration flags and
mark it as the active state for `@mmsb` blocks.
"""
function mmsb_start(; enable_gpu::Bool=true, enable_instrumentation::Bool=false,
                    config::Union{Nothing,MMSBConfig}=nothing)
    cfg = config === nothing ? MMSBConfig(enable_gpu=enable_gpu,
                                          enable_instrumentation=enable_instrumentation) : config
    state = MMSBState(cfg)
    ACTIVE_STATE[] = state
    return state
end

"""
    mmsb_stop(state; checkpoint_path=nothing)

Optionally checkpoint the state, then clear it as the active state.
"""
function mmsb_stop(state::MMSBState; checkpoint_path::Union{Nothing,String}=nothing)
    if checkpoint_path !== nothing
        checkpoint_log!(state, checkpoint_path)
    end
    ACTIVE_STATE[] === state && (ACTIVE_STATE[] = nothing)
    return nothing
end

function _resolve_location(location::Symbol)
    if location === :cpu
        return CPU_LOCATION
    elseif location === :gpu
        CUDA.functional() || throw(GPUMemoryError("CUDA not available for GPU page creation"))
        return GPU_LOCATION
    else
        throw(UnsupportedLocationError(string(location), "create_page"))
    end
end

"""
    create_page(state; size, location=:cpu, metadata=Dict())

Allocate a page on the requested location and return it.
"""
function create_page(state::MMSBState; size::Integer, location::Symbol=:cpu,
                     metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())::Page
    size > 0 || throw(ArgumentError("size must be positive"))
    loc = _resolve_location(location)
    page = if loc == CPU_LOCATION
        create_cpu_page!(state, size)
    else
        create_gpu_page!(state, size)
    end
    if !isempty(metadata)
        page.metadata = Dict{Symbol,Any}(metadata)
    end
    return page
end

"""
    update_page(state, page_id, bytes; source=:api)

Apply a full-page delta using the provided byte vector.
"""
function update_page(state::MMSBState, page_id::PageID, bytes::AbstractVector{UInt8};
                     source::Symbol=:api)
    page = get_page(state, page_id)
    page === nothing && throw(PageNotFoundError(UInt64(page_id), "update_page"))
    length(bytes) == page.size || throw(InvalidDeltaError("update payload size mismatch", UInt64(page_id)))
    data_vec = Vector{UInt8}(bytes)
    current = read_page(page)
    mask = Vector{Bool}(undef, length(current))
    local changed = false
    @inbounds for i in eachindex(current)
        diff = current[i] != data_vec[i]
        mask[i] = diff
        changed |= diff
    end
    changed || return page
    delta = create_delta(state, page_id, mask, data_vec, source)
    route_delta!(state, delta)
    return page
end

"""
    query_page(state, page_id) -> Vector{UInt8}

Return a copy of the page bytes for inspection.
"""
function query_page(state::MMSBState, page_id::PageID)::Vector{UInt8}
    page = get_page(state, page_id)
    page === nothing && throw(PageNotFoundError(UInt64(page_id), "query_page"))
    return read_page(page)
end

macro mmsb(state_expr, body)
    active_ref = :(Main.MMSB.API.ACTIVE_STATE)
    quote
        local __state = $(esc(state_expr))
        $active_ref[] = __state
        try
            $(esc(body))
        finally
            $active_ref[] = nothing
        end
    end
end

end # module API
