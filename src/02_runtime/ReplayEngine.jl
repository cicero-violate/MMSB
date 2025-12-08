module ReplayEngine

using ..PageTypes: Page, PageID, read_page, initialize!, activate!, PageLocation
using ..DeltaTypes: Delta
using ..MMSBStateTypes: MMSBState, MMSBConfig, register_page!, get_page
using ..FFIWrapper
using ..TLog
using ..PageAllocator: _allocator_handle

export replay_to_epoch, replay_to_timestamp, replay_from_checkpoint
export replay_page_history, verify_state_consistency
export replay_with_predicate, incremental_replay!, compute_diff

function _blank_state_like(state::MMSBState)::MMSBState
    config = MMSBConfig(
        enable_logging=state.config.enable_logging,
        enable_gpu=state.config.enable_gpu,
        enable_instrumentation=state.config.enable_instrumentation,
        page_size_default=state.config.page_size_default,
        max_tlog_size=state.config.max_tlog_size,
        checkpoint_interval=state.config.checkpoint_interval,
        tlog_path=state.config.tlog_path,
    )
    clone = MMSBState(config)

    lock(state.lock) do
        for (_, page) in state.pages
            # Use the global allocator to create a shadow page with the correct ID
            handle = FFIWrapper.rust_allocator_allocate(
                _allocator_handle(),
                UInt64(page.id),
                page.size,
                Int32(page.location)
            )
            shadow = Page(handle, page.id, page.location, page.size)

            # Initialize and copy current data
            initialize!(shadow)
            activate!(shadow)

            GC.@preserve page shadow begin
                data = read_page(page)
                mask = fill(UInt8(1), page.size)
                FFIWrapper.rust_page_write_masked!(
                    shadow.handle, mask, data;
                    is_sparse=false,
                    epoch=FFIWrapper.rust_page_epoch(page.handle)
                )
            end

            shadow.metadata = Dict{Symbol,Any}(page.metadata)
            register_page!(clone, shadow)
        end
    end

    clone
end

function _apply_delta!(page::Page, delta::Delta)
    GC.@preserve page delta begin
        FFIWrapper.rust_delta_apply!(page.handle, delta.handle)
    end
end

function _all_deltas(state::MMSBState)
    TLog.query_log(state)
end

function replay_to_epoch(state::MMSBState, target_epoch::UInt32)::MMSBState
    replay_state = _blank_state_like(state)
    GC.@preserve replay_state begin
        for delta in _all_deltas(state)
            delta.epoch <= target_epoch || continue
            page = get_page(replay_state, delta.page_id)
            page === nothing && continue
            _apply_delta!(page, delta)
        end
    end
    replay_state
end

function replay_to_timestamp(state::MMSBState, target_time::UInt64)::MMSBState
    replay_state = _blank_state_like(state)
    GC.@preserve replay_state begin
        for delta in _all_deltas(state)
            delta.timestamp <= target_time || continue
            page = get_page(replay_state, delta.page_id)
            page === nothing && continue
            _apply_delta!(page, delta)
        end
    end
    replay_state
end

function replay_from_checkpoint(path::AbstractString,
                                target_epoch::Union{UInt32, Nothing}=nothing)::MMSBState
    config = MMSBConfig(tlog_path=path)
    state = MMSBState(config)
    TLog.load_checkpoint!(state, path)
    target_epoch === nothing ? state : replay_to_epoch(state, target_epoch)
end

function replay_page_history(state::MMSBState, page_id::PageID)::Vector{Vector{UInt8}}
    history = Vector{Vector{UInt8}}()
    working = get_page(state, page_id)
    working === nothing && return history
    shadow = Page(working.id, working.size, working.location)
    # T6: Initialize shadow page for replay
    initialize!(shadow)
    activate!(shadow)
    GC.@preserve working shadow begin
        data = read_page(working)
        mask = fill(UInt8(1), working.size)
        FFIWrapper.rust_page_write_masked!(shadow.handle, mask, data;
                                           is_sparse=false,
                                           epoch=FFIWrapper.rust_page_epoch(working.handle))
        deltas = TLog.query_log(state; page_id=page_id)
        for delta in deltas
            _apply_delta!(shadow, delta)
            push!(history, read_page(shadow))
        end
    end
    history
end

function verify_state_consistency(state::MMSBState)::Bool
    replayed = replay_to_epoch(state, typemax(UInt32))
    lock(state.lock) do
        for (id, page) in state.pages
            replay_page = get_page(replayed, id)
            replay_page === nothing && return false
            if read_page(page) != read_page(replay_page)
                return false
            end
        end
    end
    true
end

function replay_with_predicate(state::MMSBState, predicate::Function)
    captures = Vector{Tuple{UInt32,Delta}}()
    for delta in _all_deltas(state)
        predicate(delta.epoch, delta) && push!(captures, (delta.epoch, delta))
    end
    captures
end

function incremental_replay!(state::MMSBState, deltas::Vector{Delta})
    for delta in deltas
        page = get_page(state, delta.page_id)
        page === nothing && continue
        _apply_delta!(page, delta)
    end
    state
end

compute_diff(::MMSBState, ::MMSBState) = error("compute_diff pending Rust replay parity")

end # module
