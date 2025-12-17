# Detailed profiling to isolate performance bottlenecks
using Printf
using Statistics
using Dates
include(joinpath(@__DIR__, "..", "..", "src", "MMSB.jl"))
using .MMSB
const API = MMSB.API
const PropagationEngine = MMSB.PropagationEngine
const DeltaRouter = MMSB.DeltaRouter
const PageTypes = MMSB.PageTypes
const GraphTypes = MMSB.GraphTypes
macro time_ns(expr)
    quote
        start = time_ns()
        $(esc(expr))
        time_ns() - start
    end
end
function profile_allocation_breakdown()
    println("\n" * "="^80)
    println("INVESTIGATION 1: Allocation Overhead (12.2 μs vs 1 μs)")
    println("="^80)
    times = Dict{String, Vector{Float64}}()
    for _ in 1:50
        t_total = @time_ns begin
            state = API.mmsb_start(enable_gpu=false, enable_instrumentation=false,
                                   config=MMSB.MMSBStateTypes.MMSBConfig(enable_logging=false))
            page = API.create_page(state; size=1024)
            API.mmsb_stop(state)
        end
        push!(get!(times, "total", Float64[]), t_total / 1e3)
    end
    println(@sprintf("Median: %.2f μs", median(times["total"])))
    return times
end
function profile_propagation_breakdown()
    println("\n" * "="^80)
    println("INVESTIGATION 2: Propagation Overhead (176.5 μs vs 10 μs)")
    println("="^80)
    state = API.mmsb_start(enable_gpu=false, enable_instrumentation=false,
                           config=MMSB.MMSBStateTypes.MMSBConfig(enable_logging=false))
    parent = API.create_page(state; size=1024)
    child = API.create_page(state; size=1024)
    GraphTypes.add_dependency!(state.graph, parent.id, child.id, GraphTypes.DATA_DEPENDENCY)
    PropagationEngine.register_passthrough_recompute!(state, child.id, parent.id)
    times = Dict{String, Vector{Float64}}()
    for _ in 1:50
        data = rand(UInt8, 1024)
        t = @time_ns API.update_page(state, parent.id, data)
        push!(get!(times, "total", Float64[]), t / 1e3)
    end
    API.mmsb_stop(state)
    println(@sprintf("Median: %.2f μs", median(times["total"])))
    return times
end
function main()
    println("MMSB Performance Investigation")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    alloc = profile_allocation_breakdown()
    prop = profile_propagation_breakdown()
    println("\n" * "="^80)
    println("SUMMARY")
    println("="^80)
    println(@sprintf("Allocation: %.2f μs (target: 1 μs)", median(alloc["total"])))
    println(@sprintf("Propagation: %.2f μs (target: 10 μs)", median(prop["total"])))
end
main()
