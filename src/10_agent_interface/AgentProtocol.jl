"""
Standard protocol for external agents interacting with MMSB.
"""
module AgentProtocol

export AbstractAgent, observe, act!, plan, AgentAction

using ..MMSBStateTypes: MMSBState
using ..PageTypes: Page, PageID
using ..DeltaTypes: Delta

abstract type AbstractAgent end

struct AgentAction
    delta::Delta
    target_page::PageID
    priority::Float64
end

"""
    observe(agent::AbstractAgent, state::MMSBState) -> Any

Agent reads current MMSB state. Returns agent-specific observation.
"""
function observe end

"""
    act!(agent::AbstractAgent, state::MMSBState, action::AgentAction)

Agent applies action to MMSB state via delta.
"""
function act! end

"""
    plan(agent::AbstractAgent, observation::Any) -> Vector{AgentAction}

Agent generates action sequence from observation.
"""
function plan end

end # module
