#!/usr/bin/env julia
using MMSB
using MMSB.GraphTypes: add_dependency!, DATA_DEPENDENCY
using MMSB.PropagationEngine: register_passthrough_recompute!, register_recompute_fn!,
    propagate_change!, IMMEDIATE

"""
    fit_payload(message::AbstractString, size::Int) -> Vector{UInt8}

Normalize the message to exactly `size` bytes by truncating or padding with zeros.
"""
function fit_payload(message::AbstractString, size::Int)::Vector{UInt8}
    bytes = Vector{UInt8}(undef, size)
    fill!(bytes, 0x00)
    src = collect(codeunits(message))
    copyto!(bytes, 1, src, 1, min(length(src), size))
    return bytes
end

"""
    register_pipeline(state) -> NamedTuple

Builds a three-page pipeline (source → derived → monitor) and wires recompute
functions so propagation can replay deterministic transforms.
"""
function register_pipeline(state)::NamedTuple
    source_meta = Dict{Symbol,Any}(:name => :source)
    derived_meta = Dict{Symbol,Any}(:name => :derived)
    monitor_meta = Dict{Symbol,Any}(:name => :monitor)

    source = create_page(state; size=32, location=:cpu, metadata=source_meta)
    derived = create_page(state; size=32, location=:cpu, metadata=derived_meta)
    monitor = create_page(state; size=32, location=:cpu, metadata=monitor_meta)

    add_dependency!(state.graph, source.id, derived.id, DATA_DEPENDENCY)
    add_dependency!(state.graph, derived.id, monitor.id, DATA_DEPENDENCY)

    register_passthrough_recompute!(state, derived.id, source.id)

    register_recompute_fn!(state, monitor.id, function (st, _)
        derived_payload = query_page(st, derived.id)
        summary = "derived=" * render(derived_payload)
        return fit_payload(summary, 32)
    end)

    return (source=source, derived=derived, monitor=monitor)
end

"""
    render(bytes::Vector{UInt8}) -> String

Trim trailing zeros and convert the payload to a UTF-8 string for logging.
"""
function render(bytes::Vector{UInt8})::String
    stop_idx = findfirst(==(0x00), bytes)
    last_idx = stop_idx === nothing ? length(bytes) : stop_idx - 1
    last_idx <= 0 && return ""
    return String(bytes[1:last_idx])
end
"""
    run_story(state, pipeline)

Applies multiple deltas to the source page and lets propagation recompute
the derived + monitor pages deterministically.
"""
function run_story(state, pipeline)
    updates = [
        "alpha-page" => :epoch1,
        "beta-page" => :epoch2,
        "gamma-page" => :epoch3,
    ]
    for (message, tag) in updates
        payload = fit_payload(string(message, "-", tag), pipeline.source.size)
        update_page(state, pipeline.source.id, payload; source=:tutorial)
        propagate_change!(state, pipeline.source.id, IMMEDIATE)
        derived_bytes = query_page(state, pipeline.derived.id)
        monitor_bytes = query_page(state, pipeline.monitor.id)
        println("[", tag, "] derived=", render(derived_bytes), " monitor=", render(monitor_bytes))
    end
end

"""
    run_tutorial()

High-level orchestration used by `examples/tutorial.jl`.
"""
function run_tutorial()
    state = mmsb_start(enable_gpu=false, enable_instrumentation=true)
    try
        pipeline = register_pipeline(state)
        @mmsb state begin
            run_story(state, pipeline)
        end
        println("Final stats => ", get_stats(state))
    finally
        mmsb_stop(state)
    end
end

run_tutorial()
