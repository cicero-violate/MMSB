# src/05_graph/EventSystem.jl
"""
EventSystem - Event emission and subscription

Provides pub-sub mechanism for MMSB events.
Used for debugging, monitoring, and coordination.
"""
module EventSystem

using Serialization
using Logging
using ..PageTypes: PageID
using ..DeltaTypes: Delta
using ..MMSBStateTypes: MMSBState, get_page

export emit_event!, subscribe!, unsubscribe!, EventType
export EventHandler, EventSubscription, clear_subscriptions!,
       get_subscription_count, create_debug_subscriber,
       create_logging_subscriber, log_event_to_page!

"""
Event types in MMSB system.
"""
@enum EventType begin
    PAGE_CREATED
    PAGE_DELETED
    PAGE_CHANGED
    PAGE_INVALIDATED
    PAGE_STALE
    DELTA_APPLIED
    GPU_SYNC_NEEDED
    GPU_SYNC_COMPLETE
    IR_INVALIDATED
    INFERENCE_START
    INFERENCE_COMPLETE
    OPTIMIZATION_START
    OPTIMIZATION_COMPLETE
end

"""
    EventHandler

Callback function type for event handlers.
"""
const EventHandler = Function  # (state, event_type, data...) -> nothing

"""
    EventSubscription

Represents an active event subscription.
"""
struct EventSubscription
    id::UInt64
    event_type::EventType
    handler::EventHandler
    filter::Union{Function, Nothing}
    function EventSubscription(id::UInt64, event_type::EventType, 
                               handler::EventHandler, 
                               filter::Union{Function, Nothing}=nothing)
        return new(id, event_type, handler, filter)
    end
end

# Global subscription registry
const SUBSCRIPTIONS = Dict{EventType, Vector{EventSubscription}}()
const SUBSCRIPTION_LOCK = ReentrantLock()
const NEXT_SUBSCRIPTION_ID = Ref{UInt64}(1)

"""
    emit_event!(state::MMSBState, event_type::EventType, data...)

Emit event to all subscribers.
"""
function emit_event!(state::MMSBState, event_type::EventType, data...)
    subscribers = EventSubscription[]
    lock(SUBSCRIPTION_LOCK) do
        subscribers = copy(get(SUBSCRIPTIONS, event_type, EventSubscription[]))
    end
    for sub in subscribers
        if sub.filter === nothing || sub.filter(data...)
            try
                sub.handler(state, event_type, data...)
            catch err
                @error "Event handler error" event_type exception=(err, catch_backtrace())
            end
        end
    end
    if state.config.enable_logging
        log_event!(state, event_type, data)
    end
end

"""
    subscribe!(event_type::EventType, handler::EventHandler; filter=nothing)

Subscribe to events of given type.
"""
function subscribe!(event_type::EventType, handler::EventHandler;
                    filter::Union{Function, Nothing}=nothing)::EventSubscription
    lock(SUBSCRIPTION_LOCK) do
        id = NEXT_SUBSCRIPTION_ID[]
        NEXT_SUBSCRIPTION_ID[] += 1
        sub = EventSubscription(id, event_type, handler, filter)
        push!(get!(SUBSCRIPTIONS, event_type, EventSubscription[]), sub)
        return sub
    end
end

"""
    unsubscribe!(sub::EventSubscription)

Remove event subscription.
"""
function unsubscribe!(sub::EventSubscription)
    lock(SUBSCRIPTION_LOCK) do
        if haskey(SUBSCRIPTIONS, sub.event_type)
            filter!(s -> s.id != sub.id, SUBSCRIPTIONS[sub.event_type])
        end
    end
end

"""
Log event details when logging is enabled.
"""
function log_event!(state::MMSBState, event_type::EventType, data)
    @info "MMSB event" event_type data
end

"""
    clear_subscriptions!()

Remove all event subscriptions (for cleanup/testing).
"""
function clear_subscriptions!()
    lock(SUBSCRIPTION_LOCK) do
        empty!(SUBSCRIPTIONS)
        NEXT_SUBSCRIPTION_ID[] = 1
    end
end

"""
    get_subscription_count(event_type::EventType) -> Int

Get number of active subscriptions for event type.
"""
function get_subscription_count(event_type::EventType)::Int
    lock(SUBSCRIPTION_LOCK) do
        return length(get(SUBSCRIPTIONS, event_type, EventSubscription[]))
    end
end

"""
    create_debug_subscriber(event_types::Vector{EventType}; verbose=true)

Create debug subscribers that print events.
"""
function create_debug_subscriber(event_types::Vector{EventType};
                                 verbose::Bool=true)::Vector{EventSubscription}
    subscriptions = EventSubscription[]
    for event_type in event_types
        handler = (state, etype, data...) -> begin
            if verbose
                println("[EVENT] ", etype, " ", data)
            else
                println("[EVENT] ", etype)
            end
        end
        push!(subscriptions, subscribe!(event_type, handler))
    end
    return subscriptions
end

"""
    create_logging_subscriber(state::MMSBState, log_page_id::PageID)

Create subscriber that logs all events to a page.
"""
function create_logging_subscriber(state::MMSBState, log_page_id::PageID)::Vector{EventSubscription}
    handler = (st, etype, data...) -> log_event_to_page!(st, log_page_id, etype, data)
    return [subscribe!(etype, handler) for etype in instances(EventType)]
end

"""
Serialize an event to bytes for persistent storage.
"""
function _serialize_event(event_type::EventType, data)::Vector{UInt8}
    io = IOBuffer()
    serialize(io, (event_type, collect(data)))
    return take!(io)
end

"""
    log_event_to_page!(state::MMSBState, page_id::PageID, 
                       event_type::EventType, data)

Append event to log page.
"""
function log_event_to_page!(state::MMSBState, page_id::PageID, 
                            event_type::EventType, data)
    page = get_page(state, page_id)
    page === nothing && return
    bytes = _serialize_event(event_type, data)
    copylen = min(length(bytes), length(page.data))
    @inbounds for i in 1:copylen
        page.data[i] = bytes[i]
    end
    page.metadata[:event_log_entries] = get(page.metadata, :event_log_entries, 0) + 1
end

end # module EventSystem
