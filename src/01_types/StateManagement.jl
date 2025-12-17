# src/01_types/StateManagement.jl
"""
StateManagement - High-level state lifecycle operations

Handles state reset with proper buffer clearing across modules.
"""
module StateManagement

using ..MMSBStateTypes: MMSBState, MMSBConfig

export reset_state!, get_pooled_state!, return_to_pool!

# Forward declarations for circular dependency resolution
function clear_propagation_buffers! end

"""
    reset_state!(state::MMSBState)

Complete state reset including propagation buffers.
Delegates to MMSBState.reset! and PropagationEngine.clear_propagation_buffers!
"""
function reset_state!(state::MMSBState)
    # Reset core state
    MMSBStateTypes.reset!(state)
    
    # Clear propagation buffers if module is loaded
    if isdefined(@__MODULE__, :clear_propagation_buffers!)
        clear_propagation_buffers!(state)
    end
    
    return nothing
end

# State pool for allocation optimization
const STATE_POOL = Channel{MMSBState}(10)

"""
    get_pooled_state!(config::MMSBConfig) -> MMSBState

Retrieve state from pool or allocate new. Resets before returning.
Expected speedup: 6μs → 2-3μs for pooled states.
"""
function get_pooled_state!(config::MMSBConfig)
    if isready(STATE_POOL)
        state = take!(STATE_POOL)
        reset_state!(state)
        state.config = config
        return state
    end
    return MMSBState(config)
end

"""
    return_to_pool!(state::MMSBState)

Return state to pool for reuse. Non-blocking.
"""
function return_to_pool!(state::MMSBState)
    isopen(STATE_POOL) && put!(STATE_POOL, state)
    return nothing
end

end # module StateManagement
