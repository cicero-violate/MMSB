"""
Type system for external agents.
"""
module AgentTypes

export RLAgent, SymbolicAgent, HybridAgent, PlanningAgent
export AgentState, AgentMemory

using ..AgentProtocol: AbstractAgent
using ..MMSBStateTypes: MMSBState

mutable struct AgentState{T}
    internal_state::T
    memory::Dict{Symbol, Any}
    step_count::Int
end

AgentState(state::T) where T = AgentState{T}(state, Dict{Symbol, Any}(), 0)

struct AgentMemory
    observations::Vector{Any}
    actions::Vector{Any}
    rewards::Vector{Float64}
    max_size::Int
end

AgentMemory(max_size::Int=1000) = AgentMemory(Any[], Any[], Float64[], max_size)

function push_memory!(mem::AgentMemory, obs, action, reward::Float64)
    push!(mem.observations, obs)
    push!(mem.actions, action)
    push!(mem.rewards, reward)
    
    if length(mem.observations) > mem.max_size
        popfirst!(mem.observations)
        popfirst!(mem.actions)
        popfirst!(mem.rewards)
    end
end

end # module
