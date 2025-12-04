# src/02_runtime/ReplayEngine.jl
"""
ReplayEngine - Deterministic state reconstruction

Enables time-travel debugging and state verification by replaying
transaction logs to reconstruct historical states.
"""
module ReplayEngine

using Serialization
using ..PageTypes: Page, PageID
using ..DeltaTypes: Delta, dense_data
using ..MMSBStateTypes: MMSBState, MMSBConfig, register_page!, get_page
using ..DeltaRouter: apply_delta_to_page!, create_delta

export replay_to_epoch, replay_to_timestamp, replay_from_checkpoint
export replay_page_history, verify_state_consistency
export replay_with_predicate, incremental_replay!, compute_diff

"""
Deep copy a page for replay purposes.
"""
function _clone_page(page::Page)::Page
    clone = Page(page.id, page.size, page.location)
    clone.epoch = page.epoch
    clone.mask .= page.mask
    clone.data .= page.data
    clone.metadata = Dict{Symbol,Any}(page.metadata)
    return clone
end

"""
Create an empty MMSBState that mirrors another state's configuration and page registry (without data).
"""
function _blank_state_like(state::MMSBState)::MMSBState
    config = MMSBConfig(
        enable_logging=state.config.enable_logging,
        enable_gpu=state.config.enable_gpu,
        enable_instrumentation=state.config.enable_instrumentation,
        page_size_default=state.config.page_size_default,
        max_tlog_size=state.config.max_tlog_size,
        checkpoint_interval=state.config.checkpoint_interval,
    )
    new_state = MMSBState(config)
    lock(state.lock) do
        for (id, page) in state.pages
            shadow = Page(page.id, page.size, page.location)
            shadow.metadata = Dict{Symbol,Any}(page.metadata)
            register_page!(new_state, shadow)
        end
    end
    return new_state
end

"""
Create a deep copy of a state (used for predicate captures).
"""
function _snapshot_state(state::MMSBState)::MMSBState
    clone = MMSBState(deepcopy(state.config))
    lock(state.lock) do
        for (id, page) in state.pages
            register_page!(clone, _clone_page(page))
        end
    end
    return clone
end

"""
Iterate through deltas in chronological order with optional filter.
"""
function _chronological_deltas(state::MMSBState;
                               predicate::Function=(d -> true))::Vector{Delta}
    lock(state.lock) do
        return [delta for delta in sort(state.tlog; by = d -> d.epoch) if predicate(delta)]
    end
end

"""
    replay_to_epoch(state, target_epoch)

Reconstruct MMSB state at specific epoch.
"""
function replay_to_epoch(state::MMSBState, target_epoch::UInt32)::MMSBState
    replay_state = _blank_state_like(state)
    deltas = _chronological_deltas(state; predicate = d -> d.epoch <= target_epoch)
    for delta in deltas
        page = get_page(replay_state, delta.page_id)
        page === nothing && continue
        apply_delta_to_page!(page, delta)
    end
    return replay_state
end

"""
    replay_to_timestamp(state, target_time)

Reconstruct state at specific nanosecond timestamp.
"""
function replay_to_timestamp(state::MMSBState, target_time::UInt64)::MMSBState
    replay_state = _blank_state_like(state)
    deltas = _chronological_deltas(state; predicate = d -> d.timestamp <= target_time)
    for delta in deltas
        page = get_page(replay_state, delta.page_id)
        page === nothing && continue
        apply_delta_to_page!(page, delta)
    end
    return replay_state
end

"""
    replay_from_checkpoint(path, target_epoch)

Load checkpoint and optionally replay to specific epoch.
"""
function replay_from_checkpoint(checkpoint_path::String, 
                               target_epoch::Union{UInt32, Nothing}=nothing)::MMSBState
    io = open(checkpoint_path, "r")
    deltas, pages = deserialize(io)
    close(io)
    state = MMSBState(MMSBConfig())
    state.pages = deepcopy(pages)
    state.tlog = deepcopy(deltas)
    if target_epoch === nothing
        for (_, page) in state.pages
            page.epoch = UInt32(0)
        end
        incremental_replay!(state, state.tlog)
        return state
    else
        return replay_to_epoch(state, target_epoch)
    end
end

"""
    replay_page_history(state, page_id) -> Vector{Page}

Replay all versions of a single page.
"""
function replay_page_history(state::MMSBState, page_id::PageID)::Vector{Page}
    history = Page[]
    page_template = get_page(state, page_id)
    page_template === nothing && return history
    working_page = Page(page_template.id, page_template.size, page_template.location)
    working_page.metadata = Dict{Symbol,Any}(page_template.metadata)
    deltas = _chronological_deltas(state; predicate = d -> d.page_id == page_id)
    for delta in deltas
        apply_delta_to_page!(working_page, delta)
        push!(history, _clone_page(working_page))
    end
    return history
end

"""
    verify_state_consistency(state) -> Bool

Verify that current state matches tlog replay.
"""
function verify_state_consistency(state::MMSBState)::Bool
    reconstructed = replay_to_epoch(state, typemax(UInt32))
    lock(state.lock) do
        for (id, page) in state.pages
            replay_page = get_page(reconstructed, id)
            replay_page === nothing && return false
            if page.size != replay_page.size || page.epoch != replay_page.epoch
                return false
            end
            if Vector{UInt8}(page.data) != Vector{UInt8}(replay_page.data)
                return false
            end
        end
    end
    return true
end

"""
    replay_with_predicate(state, predicate) -> Vector{Tuple{UInt32,MMSBState}}

Replay log and capture states matching predicate.
"""
function replay_with_predicate(state::MMSBState, predicate::Function)
    captures = Vector{Tuple{UInt32,MMSBState}}()
    replay_state = _blank_state_like(state)
    for delta in _chronological_deltas(state)
        page = get_page(replay_state, delta.page_id)
        page === nothing && continue
        apply_delta_to_page!(page, delta)
        predicate(delta.epoch, delta) && push!(captures, (delta.epoch, _snapshot_state(replay_state)))
    end
    return captures
end

"""
    incremental_replay!(base_state, deltas)

Apply deltas to existing state (mutating).
"""
function incremental_replay!(base_state::MMSBState, deltas::Vector{Delta})
    sorted = sort(deltas; by = d -> d.epoch)
    for delta in sorted
        page = get_page(base_state, delta.page_id)
        page === nothing && continue
        apply_delta_to_page!(page, delta)
    end
    return base_state
end

"""
    compute_diff(state1, state2) -> Vector{Delta}

Compute deltas needed to transform state1 into state2.
"""
function compute_diff(state1::MMSBState, state2::MMSBState)::Vector{Delta}
    diffs = Delta[]
    ids = union(keys(state1.pages), keys(state2.pages))
    for id in ids
        page2 = get_page(state2, id)
        page2 === nothing && continue
        baseline = get_page(state1, id)
        baseline_bytes = baseline === nothing ? zeros(UInt8, page2.size) : Vector{UInt8}(baseline.data)
        target_bytes = Vector{UInt8}(page2.data)
        mask = Vector{Bool}(baseline_bytes .!= target_bytes)
        any(mask) || continue
        delta = create_delta(state2, id, mask, target_bytes, :diff)
        push!(diffs, delta)
    end
    return diffs
end

end # module ReplayEngine
