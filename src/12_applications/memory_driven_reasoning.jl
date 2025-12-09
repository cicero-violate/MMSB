"""
Memory-driven reasoning applications.
"""
module MemoryDrivenReasoning

export ReasoningContext, reason_over_memory, temporal_reasoning

using ..MMSBStateTypes: MMSBState
using ..ReasoningEngine: structural_inference
using ..TLog: replay_tlog

struct ReasoningContext
    state::MMSBState
    query::String
    time_range::Tuple{UInt64, UInt64}
end

function reason_over_memory(ctx::ReasoningContext)::Dict{Symbol, Any}
    inference = structural_inference(ctx.state.graph)
    return Dict(:inference => inference, :query => ctx.query)
end

function temporal_reasoning(ctx::ReasoningContext)::Vector{Any}
    # Replay TLog in time range and perform reasoning
    events = Any[]
    return events
end

end # module
