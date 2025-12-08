module TLog

using Base: C_NULL
using ..FFIWrapper
using ..MMSBStateTypes: MMSBState
using ..PageTypes: Page, PageID, PageLocation, metadata_from_blob, initialize!, activate!
using ..PageAllocator: allocator_handle
using ..DeltaTypes: Delta
using ..RustErrors: RustFFIError, rethrow_translated

export append_to_log!, query_log, get_deltas_for_page, get_deltas_in_range,
       compute_log_statistics, log_summary,
       checkpoint_log!, load_checkpoint!, replay_log

function _with_rust_errors(f::Function, context::String)
    try
        return f()
    catch err
        if err isa RustFFIError
            rethrow_translated(err; message=context)
        end
        rethrow()
    end
end

function append_to_log!(state::MMSBState, delta::Delta)
    _with_rust_errors("Transaction log append failed") do
        GC.@preserve state delta begin
            FFIWrapper.rust_tlog_append!(state.tlog_handle, delta.handle)
        end
    end
    return nothing
end

function log_summary(state::MMSBState)
    summary = _with_rust_errors("Transaction log summary failed: $(state.config.tlog_path)") do
        FFIWrapper.rust_tlog_summary(state.config.tlog_path)
    end
    summary === nothing && return (total_deltas = 0, total_bytes = 0, last_epoch = UInt32(0))
    summary
end

function _iterate_log(f::Function, path::AbstractString)
    reader = FFIWrapper.rust_tlog_reader_new(path)
    reader.ptr == C_NULL && return f(nothing)
    try
        return f(reader)
    finally
        FFIWrapper.rust_tlog_reader_free!(reader)
    end
end

function query_log(state::MMSBState; page_id::Union{PageID,Nothing}=nothing,
                   start_time::Union{UInt64,Nothing}=nothing,
                   end_time::Union{UInt64,Nothing}=nothing,
                   source::Union{Symbol,Nothing}=nothing)
    path = state.config.tlog_path
    _iterate_log(path) do reader
        reader === nothing && return Delta[]
        results = Delta[]
        while true
            handle = FFIWrapper.rust_tlog_reader_next(reader)
            if FFIWrapper.isnull(handle)
                break
            end
            pid = PageID(FFIWrapper.rust_delta_page_id(handle))
            ts = FFIWrapper.rust_delta_timestamp(handle)
            src = Symbol(FFIWrapper.rust_delta_source(handle))
            if (page_id !== nothing && pid != page_id) ||
               (start_time !== nothing && ts < start_time) ||
               (end_time !== nothing && ts > end_time) ||
               (source !== nothing && src != source)
                FFIWrapper.rust_delta_free!(handle)
                continue
            end
            push!(results, Delta(handle))
        end
        return results
    end
end

get_deltas_for_page(state::MMSBState, pid::PageID) = query_log(state; page_id=pid)

function get_deltas_in_range(state::MMSBState, start_idx::Int, end_idx::Int)
    deltas = query_log(state)
    isempty(deltas) && return deltas
    first_idx = clamp(start_idx, 1, length(deltas))
    last_idx = clamp(end_idx, first_idx, length(deltas))
    deltas[first_idx:last_idx]
end

function compute_log_statistics(state::MMSBState)
    summary = log_summary(state)
    return Dict(
        :total_deltas => summary.total_deltas,
        :total_delta_bytes => summary.total_bytes,
        :last_epoch => summary.last_epoch,
    )
end

function replay_log(state::MMSBState, target_epoch::UInt32)
    # placeholder helper that delegates to ReplayEngine
    engine = getfield(parentmodule(@__MODULE__), :ReplayEngine)
    engine.replay_to_epoch(state, target_epoch)
end

function checkpoint_log!(state::MMSBState, path::AbstractString)
    _with_rust_errors("Checkpoint write failed: $(path)") do
        GC.@preserve state begin
            FFIWrapper.rust_checkpoint_write!(allocator_handle(), state.tlog_handle, path)
        end
    end
end

function load_checkpoint!(state::MMSBState, path::AbstractString)
    _with_rust_errors("Checkpoint load failed: $(path)") do
        GC.@preserve state begin
            FFIWrapper.rust_checkpoint_load!(allocator_handle(), state.tlog_handle, path)
        end
    end
    _refresh_pages!(state)
end

function _refresh_pages!(state::MMSBState)
    infos = FFIWrapper.rust_allocator_page_infos(allocator_handle())
    lock(state.lock) do
        empty!(state.pages)
        for info in infos
            handle = FFIWrapper.rust_allocator_acquire_page(allocator_handle(), info.page_id)
            handle.ptr == C_NULL && continue
            page = Page(handle, PageID(info.page_id), PageLocation(info.location), Int(info.size))
            if info.metadata_len == 0 || info.metadata_ptr == C_NULL
                blob = UInt8[]
            else
                metadata = unsafe_string(Ptr{UInt8}(info.metadata_ptr), info.metadata_len)
                blob = Vector{UInt8}(codeunits(metadata))
            end
            page.metadata = metadata_from_blob(blob)
            # T6: Initialize restored pages from checkpoint
            initialize!(page)
            activate!(page)  # Pages from checkpoint are immediately usable
            state.pages[page.id] = page
        end
    end
end

end # module
