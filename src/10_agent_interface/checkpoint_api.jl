"""
Checkpoint API for external agents.
"""
module CheckpointAPI

export create_checkpoint, restore_checkpoint, list_checkpoints

using ..MMSBStateTypes: MMSBState
using ..FFIWrapper

function create_checkpoint(state::MMSBState, name::String)::String
    checkpoint_id = "ckpt_$(time_ns())_$(name)"
    result = ccall((:rust_checkpoint_create, LIBMMSB),
                  Int32, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t),
                  state.tlog_handle.ptr, pointer(checkpoint_id), sizeof(checkpoint_id))
    result == 0 || error("Checkpoint creation failed: $result")
    return checkpoint_id
end

function restore_checkpoint(state::MMSBState, checkpoint_id::String)
    result = ccall((:rust_checkpoint_restore, LIBMMSB),
                  Int32, (Ptr{Cvoid}, Ptr{UInt8}, Csize_t),
                  state.tlog_handle.ptr, pointer(checkpoint_id), sizeof(checkpoint_id))
    result == 0 || error("Checkpoint restore failed: $result")
end

function list_checkpoints(state::MMSBState)::Vector{String}
    # Placeholder - needs Rust implementation
    return String[]
end

end # module
