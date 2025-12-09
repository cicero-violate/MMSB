"""
Reinforcement Learning agents using MMSB as world model.
"""
module RLAgents

export RLAgent, train_step!, compute_reward

using ..AgentProtocol: AbstractAgent, observe, act!, AgentAction
using ..AgentTypes: AgentState, AgentMemory, push_memory!
using ..MMSBStateTypes: MMSBState

struct RLAgent{T} <: AbstractAgent
    agent_state::AgentState{T}
    memory::AgentMemory
    learning_rate::Float64
    discount_factor::Float64
end

function RLAgent(initial_state::T; lr=0.001, γ=0.99) where T
    RLAgent{T}(AgentState(initial_state), AgentMemory(), lr, γ)
end

function observe(agent::RLAgent, state::MMSBState)
    return (
        n_pages = length(state.pages),
        graph_size = length(state.graph.nodes),
        step = agent.agent_state.step_count
    )
end

function compute_reward(agent::RLAgent, state::MMSBState, action::AgentAction)::Float64
    # Placeholder reward function
    return -action.priority
end

function train_step!(agent::RLAgent, state::MMSBState, action::AgentAction)
    obs = observe(agent, state)
    reward = compute_reward(agent, state, action)
    push_memory!(agent.memory, obs, action, reward)
    agent.agent_state.step_count += 1
end

end # module
