"""
Planning agents using Layer 9 planning engine.
"""
module PlanningAgents

export PlanningAgent, generate_plan, execute_plan_step

using ..AgentProtocol: AbstractAgent, AgentAction
import ..AgentProtocol: observe
using ..AgentTypes: AgentState
using ..MMSBStateTypes: MMSBState
using ..PlanningEngine: search_plan, rollout_simulation
using ..IntentionEngine: compute_intention

struct PlanningAgent <: AbstractAgent
    agent_state::AgentState{Vector{AgentAction}}
    horizon::Int
    search_depth::Int
end

PlanningAgent(horizon=10, depth=5) = PlanningAgent(AgentState(AgentAction[]), horizon, depth)

function observe(agent::PlanningAgent, state::MMSBState)
    return (
        intention = compute_intention(state),
        current_plan = agent.agent_state.internal_state
    )
end

function generate_plan(agent::PlanningAgent, state::MMSBState, goal::Any)::Vector{AgentAction}
    plan = search_plan(state, goal, agent.search_depth)
    agent.agent_state.internal_state = plan
    return plan
end

function execute_plan_step(agent::PlanningAgent)::Union{AgentAction, Nothing}
    isempty(agent.agent_state.internal_state) && return nothing
    action = popfirst!(agent.agent_state.internal_state)
    agent.agent_state.step_count += 1
    return action
end

end # module
