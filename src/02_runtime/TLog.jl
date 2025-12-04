# src/02_runtime/TLog.jl
"""
TLog - Transaction Log Management

Append-only log of all deltas for deterministic replay,
crash recovery, and temporal debugging.
"""
module TLog

using Serialization
using ..DeltaTypes: Delta, delta_byte_count, dense_data, serialize_delta, deserialize_delta
using ..PageTypes: Page, PageID, is_gpu_page, CPU_LOCATION, serialize_page, deserialize_page
using ..MMSBStateTypes: MMSBState, get_page, allocate_delta_id!
using ..ErrorTypes: SerializationError

export append_to_log!, replay_log, checkpoint_log!, load_checkpoint!,
       compress_log!, query_log, get_deltas_for_page, get_deltas_in_range,
       compute_log_statistics, trigger_checkpoint!

const CHECKPOINT_MAGIC = "MMSBCHK1"
const CHECKPOINT_VERSION = UInt32(1)

"""
Append delta to the state's transaction log.
"""
function append_to_log!(state::MMSBState, delta::Delta)
    needs_checkpoint = false
    lock(state.lock) do
        push!(state.tlog, delta)
        needs_checkpoint = length(state.tlog) > state.config.max_tlog_size
    end
    needs_checkpoint && trigger_checkpoint!(state)
end

"""
Replay deltas between epochs to reconstruct state snapshot.
"""
function replay_log(state::MMSBState, start_epoch::UInt32, 
                    end_epoch::UInt32)::Dict{PageID, Page}
    snapshot = Dict{PageID, Page}()
    deltas = Delta[]
    lock(state.lock) do
        for (id, page) in state.pages
            if start_epoch <= page.epoch <= end_epoch
                snapshot[id] = deepcopy(page)
            end
        end
        for delta in state.tlog
            if start_epoch <= delta.epoch <= end_epoch
                push!(deltas, delta)
            end
        end
    end
    for delta in sort(deltas; by = d -> d.epoch)
        page = get(snapshot, delta.page_id, nothing)
        page === nothing && continue
        if is_gpu_page(page)
            page.data = Vector{UInt8}(page.data)
            page.mask = Vector{Bool}(page.mask)
            page.location = CPU_LOCATION
        end
        @inbounds for i in eachindex(delta.mask)
            if delta.mask[i]
                page.data[i] = delta.data[i]
                page.mask[i] = true
            end
        end
        page.epoch = delta.epoch
    end
    return snapshot
end

function _snapshot_pages(state::MMSBState)::Dict{PageID, Vector{UInt8}}
    lock(state.lock) do
        return Dict{PageID, Vector{UInt8}}(
            page_id => serialize_page(page) for (page_id, page) in state.pages
        )
    end
end

function _snapshot_deltas(state::MMSBState)::Vector{Vector{UInt8}}
    lock(state.lock) do
        return [serialize_delta(delta) for delta in state.tlog]
    end
end

"""
Persist the transaction log to disk.
"""
function checkpoint_log!(state::MMSBState, path::String)
    pages = _snapshot_pages(state)
    deltas = _snapshot_deltas(state)
    open(path, "w") do io
        write(io, CHECKPOINT_MAGIC)
        write(io, CHECKPOINT_VERSION)
        write(io, time_ns())
        write(io, Int64(length(pages)))
        for (page_id, bytes) in pages
            write(io, page_id)
            write(io, Int64(length(bytes)))
            write(io, bytes)
        end
        write(io, Int64(length(deltas)))
        for bytes in deltas
            write(io, Int64(length(bytes)))
            write(io, bytes)
        end
    end
end

"""
Restore state from a checkpoint file.
"""
function load_checkpoint!(state::MMSBState, path::String)
    pages = Dict{PageID, Page}()
    deltas = Delta[]
    open(path, "r") do io
        magic = String(read(io, length(CHECKPOINT_MAGIC)))
        magic == CHECKPOINT_MAGIC || throw(SerializationError("Invalid checkpoint file"))
        version = read(io, UInt32)
        version <= CHECKPOINT_VERSION || throw(SerializationError("Unsupported checkpoint version $version"))
        _timestamp = read(io, UInt64)
        page_count = read(io, Int64)
        for _ in 1:page_count
            page_id = read(io, PageID)
            len = read(io, Int64)
            bytes = read(io, len)
            page = deserialize_page(bytes)
            pages[page_id] = page
        end
        delta_count = read(io, Int64)
        for _ in 1:delta_count
            len = read(io, Int64)
            bytes = read(io, len)
            push!(deltas, deserialize_delta(bytes))
        end
    end
    lock(state.lock) do
        state.pages = pages
        state.tlog = deltas
    end
end

"""
Merge deltas per page to keep log small.
"""
function _merge_page_deltas(state::MMSBState, page_id::PageID, deltas::Vector{Delta})::Vector{Delta}
    isempty(deltas) && return Delta[]
    sorted = sort(copy(deltas); by = d -> d.epoch)
    page = get_page(state, page_id)
    page_size = page === nothing ? length(sorted[end].mask) : page.size
    merged_mask = falses(page_size)
    merged_data = zeros(UInt8, page_size)
    for delta in sorted
        dense = dense_data(delta)
        @inbounds for i in 1:length(delta.mask)
            if delta.mask[i]
                merged_mask[i] = true
                merged_data[i] = dense[i]
            end
        end
    end
    new_id = allocate_delta_id!(state)
    merged = Delta(new_id, page_id, sorted[end].epoch, merged_mask, merged_data, :compressed)
    return [merged]
end

function compress_log!(state::MMSBState)
    deltas_copy = let copy_vec = nothing
        lock(state.lock) do
            copy_vec = copy(state.tlog)
        end
        copy_vec
    end
    grouped = Dict{PageID, Vector{Delta}}()
    for delta in deltas_copy
        push!(get!(grouped, delta.page_id, Delta[]), delta)
    end
    merged = Delta[]
    for (page_id, deltas) in grouped
        append!(merged, _merge_page_deltas(state, page_id, deltas))
    end
    merged_sorted = sort!(merged; by = d -> d.epoch)
    lock(state.lock) do
        state.tlog = merged_sorted
    end
end

"""
Query log with optional filters.
"""
function query_log(state::MMSBState; 
                   page_id::Union{PageID, Nothing}=nothing,
                   start_time::Union{UInt64, Nothing}=nothing,
                   end_time::Union{UInt64, Nothing}=nothing,
                   source::Union{Symbol, Nothing}=nothing)::Vector{Delta}
    lock(state.lock) do
        return [delta for delta in state.tlog if
            (page_id === nothing || delta.page_id == page_id) &&
            (start_time === nothing || delta.timestamp >= start_time) &&
            (end_time === nothing || delta.timestamp <= end_time) &&
            (source === nothing || delta.source == source)]
    end
end

"""
Get all deltas for a page.
"""
function get_deltas_for_page(state::MMSBState, page_id::PageID)::Vector{Delta}
    lock(state.lock) do
        return [delta for delta in state.tlog if delta.page_id == page_id]
    end
end

"""
Slice log by index range.
"""
function get_deltas_in_range(state::MMSBState, start_idx::Int, 
                             end_idx::Int)::Vector{Delta}
    lock(state.lock) do
        start_idx = max(start_idx, 1)
        end_idx = min(end_idx, length(state.tlog))
        return state.tlog[start_idx:end_idx]
    end
end

"""
Compute statistics about the log contents.
"""
function compute_log_statistics(state::MMSBState)::Dict{Symbol, Any}
    lock(state.lock) do
        bytes = sum(delta_byte_count, state.tlog)
        sources = Dict{Symbol, Int}()
        for delta in state.tlog
            sources[delta.source] = get(sources, delta.source, 0) + 1
        end
        return Dict(
            :total_deltas => length(state.tlog),
            :bytes_changed => bytes,
            :sources => sources,
        )
    end
end

"""
Trigger compression/checkpoint when log grows too large.
"""
function trigger_checkpoint!(state::MMSBState)
    compress_log!(state)
end

end # module TLog
