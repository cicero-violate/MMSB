"""
Multi-agent coordination and orchestration.
"""
module MultiAgentSystem

export AgentCoordinator, register_agent!, coordinate_step!

using ..MMSBStateTypes: MMSBState
using ..AgentProtocol: AbstractAgent, observe, AgentAction

mutable struct AgentCoordinator
    state::MMSBState
    agents::Vector{AbstractAgent}
    coordination_strategy::Symbol  # :sequential, :parallel, :priority
end

AgentCoordinator(state::MMSBState, strategy=:sequential) = AgentCoordinator(state, AbstractAgent[], strategy)

function register_agent!(coord::AgentCoordinator, agent::AbstractAgent)
    push!(coord.agents, agent)
end

function coordinate_step!(coord::AgentCoordinator)
    if coord.coordination_strategy == :sequential
        for agent in coord.agents
            obs = observe(agent, coord.state)
            # Execute agent step
        end
    end
end

end # module
