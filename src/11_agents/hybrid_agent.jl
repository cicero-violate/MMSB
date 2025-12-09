"""
Hybrid agents combining symbolic reasoning and neural learning.
"""
module HybridAgents

export HybridAgent, symbolic_step!, neural_step!

using ..AgentProtocol: AbstractAgent, observe, AgentAction
using ..AgentTypes: AgentState
using ..SymbolicAgents: SymbolicAgent, apply_rule
using ..RLAgents: RLAgent, train_step!
using ..MMSBStateTypes: MMSBState

struct HybridAgent <: AbstractAgent
    symbolic::SymbolicAgent
    rl::RLAgent{Any}
    mix_ratio::Float64  # 0.0 = pure symbolic, 1.0 = pure RL
end

HybridAgent(mix_ratio=0.5) = HybridAgent(SymbolicAgent(), RLAgent(nothing, lr=0.001), mix_ratio)

function observe(agent::HybridAgent, state::MMSBState)
    return (
        symbolic_obs = observe(agent.symbolic, state),
        rl_obs = observe(agent.rl, state)
    )
end

function symbolic_step!(agent::HybridAgent, state::MMSBState)::Vector{AgentAction}
    rules = agent.symbolic.agent_state.internal_state
    isempty(rules) && return AgentAction[]
    return apply_rule(agent.symbolic, first(rules), state)
end

function neural_step!(agent::HybridAgent, state::MMSBState, action::AgentAction)
    train_step!(agent.rl, state, action)
end

end # module
