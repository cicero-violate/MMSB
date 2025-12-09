"""
Event subscription system for agents to monitor MMSB changes.
"""
module EventSubscription

export subscribe_to_events, unsubscribe, EventType, @event

using ..MMSBStateTypes: MMSBState
using ..GraphTypes: ShadowPageGraph

@enum EventType begin
    PAGE_CREATED
    PAGE_MODIFIED
    DELTA_APPLIED
    GRAPH_UPDATED
    CHECKPOINT_CREATED
end

mutable struct EventSubscription
    id::UInt64
    event_types::Set{EventType}
    callback::Function
    active::Bool
end

const SUBSCRIPTIONS = Dict{UInt64, EventSubscription}()
const NEXT_SUB_ID = Ref{UInt64}(1)

function subscribe_to_events(types::Vector{EventType}, callback::Function)::UInt64
    id = NEXT_SUB_ID[]
    NEXT_SUB_ID[] += 1
    SUBSCRIPTIONS[id] = EventSubscription(id, Set(types), callback, true)
    return id
end

function unsubscribe(id::UInt64)
    haskey(SUBSCRIPTIONS, id) && (SUBSCRIPTIONS[id].active = false)
end

function emit_event(event_type::EventType, data::Any)
    for sub in values(SUBSCRIPTIONS)
        sub.active && event_type ∈ sub.event_types && sub.callback(event_type, data)
    end
end

macro event(event_type, data)
    quote
        emit_event($(esc(event_type)), $(esc(data)))
    end
end

end # module
