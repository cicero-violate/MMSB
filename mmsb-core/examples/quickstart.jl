#!/usr/bin/env julia
using MMSB
using MMSB.TLog: checkpoint_log!

"""
    build_payload(message::AbstractString, size::Int) -> Vector{UInt8}

Produce a size-aligned byte vector by zero-padding `message`
so it can be applied as a full-page delta without resizing.
"""
function build_payload(message::AbstractString, size::Int)::Vector{UInt8}
    bytes = Vector{UInt8}(undef, size)
    fill!(bytes, 0x00)
    src = collect(codeunits(message))
    copyto!(bytes, 1, src, 1, min(length(src), size))
    return bytes
end

"""
    printable(bytes::Vector{UInt8}) -> String

Trim trailing null bytes and turn the payload into a UTF-8 string.
"""
function printable(bytes::Vector{UInt8})::String
    stop_idx = findfirst(==(0x00), bytes)
    last_idx = stop_idx === nothing ? length(bytes) : stop_idx - 1
    last_idx <= 0 && return ""
    return String(bytes[1:last_idx])
end

"""
    run_quickstart()

Creates a CPU page, applies a delta, queries the bytes,
prints monitoring stats, and checkpoints the transaction log.
"""
function run_quickstart()
    state = mmsb_start()
    try
        metadata = Dict{Symbol,Any}(:name => :quickstart)
        page = create_page(state; size=16, location=:cpu, metadata=metadata)
        payload = build_payload("mmsb-delta-proof", page.size)
        update_page(state, page.id, payload; source=:quickstart)

        bytes = query_page(state, page.id)
        println("Page $(page.id) bytes: ", printable(bytes))

        println("Stats: ", get_stats(state))
        checkpoint_path = joinpath(mktempdir(), "quickstart.chk")
        checkpoint_log!(state, checkpoint_path)
        println("Checkpoint stored at $checkpoint_path")
    finally
        mmsb_stop(state)
    end
end

run_quickstart()
