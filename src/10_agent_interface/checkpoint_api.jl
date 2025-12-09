"""
Checkpoint API for external agents.
"""
module CheckpointAPI

export create_checkpoint, restore_checkpoint, list_checkpoints

using ..MMSBStateTypes: MMSBState
using ..TLog: checkpoint_log!, load_checkpoint!

function create_checkpoint(state::MMSBState, name::String)::String
    path = "checkpoint_$(time_ns())_$(name).ckpt"
    checkpoint_log!(state, path)
    return path
end

function restore_checkpoint(state::MMSBState, path::String)
    load_checkpoint!(state, path)
end

function list_checkpoints(state::MMSBState)::Vector{String}
    # Placeholder - needs Rust implementation
    return String[]
end

end # module
