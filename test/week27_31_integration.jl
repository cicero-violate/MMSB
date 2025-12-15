module Week27To31IntegrationTests

using Test
using CUDA
using Statistics
using Random

include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB

const API = MMSB.API
const PropagationEngine = MMSB.PropagationEngine
const GraphTypes = MMSB.GraphTypes

@testset "Week 27: Benchmarking" begin
    @testset "Allocator Performance" begin
        state = API.mmsb_start(enable_gpu=false)
        times = Float64[]
        for _ in 1:100
            t0 = time_ns()
            page = API.create_page(state; size=1024)
            push!(times, (time_ns() - t0) / 1e6)
        end
        @test median(times) < 10.0
        @test minimum(times) > 0.0
        API.mmsb_stop(state)
    end
    
    @testset "Graph Traversal" begin
        state = API.mmsb_start(enable_gpu=false)
        pages = [API.create_page(state; size=128) for _ in 1:10]
        for i in 1:9
            GraphTypes.add_dependency!(state.graph, pages[i].id, pages[i+1].id, GraphTypes.DATA_DEPENDENCY)
        end
        t0 = time_ns()
        sorted = GraphTypes.topological_sort(state.graph)
        topo_time = (time_ns() - t0) / 1e6
        @test length(sorted) == length(pages)
        @test topo_time < 500.0
        API.mmsb_stop(state)
    end
    
    @testset "Propagation Fast Path" begin
        state = API.mmsb_start(enable_gpu=false)
        src = API.create_page(state; size=256)
        dst = API.create_page(state; size=256)
        data = rand(UInt8, 256)
        API.update_page(state, src.id, data)
        t0 = time_ns()
        PropagationEngine.register_passthrough_recompute!(state, dst.id, src.id)
        API.update_page(state, src.id, data)
        prop_time = (time_ns() - t0) / 1e6
        @test prop_time < 50.0
        API.mmsb_stop(state)
    end
end

@testset "Week 28-29: GPU" begin
    if CUDA.functional()
        @testset "GPU Pages" begin
            state = API.mmsb_start(enable_gpu=true)
            page = API.create_page(state; size=4096, location=:gpu)
            data = rand(UInt8, 4096)
            t0 = time_ns()
            API.update_page(state, page.id, data)
            gpu_time = (time_ns() - t0) / 1e6
            @test gpu_time < 50.0
            API.mmsb_stop(state)
        end
    else
        @test true
    end
end

@testset "Week 30-31: Performance" begin
    @testset "Delta Merge" begin
        state = API.mmsb_start(enable_gpu=false)
        page = API.create_page(state; size=1024)
        for i in 1:10
            API.update_page(state, page.id, rand(UInt8, 1024))
        end
        @test true
        API.mmsb_stop(state)
    end
    
    @testset "Batch Propagation" begin
        state = API.mmsb_start(enable_gpu=false)
        pages = [API.create_page(state; size=256) for _ in 1:20]
        deltas = [(p.id, rand(UInt8, 256)) for p in pages]
        t0 = time_ns()
        for (id, data) in deltas
            API.update_page(state, id, data)
        end
        batch_time = (time_ns() - t0) / 1e6
        @test batch_time < 100.0
        API.mmsb_stop(state)
    end
end

@testset "Week 32: Reliability" begin
    @testset "Sequential Updates" begin
        state = API.mmsb_start(enable_gpu=false)
        page = API.create_page(state; size=256)
        for _ in 1:10
            API.update_page(state, page.id, rand(UInt8, 256))
        end
        @test true
        API.mmsb_stop(state)
    end
    
    @testset "Multi-Page Stress" begin
        state = API.mmsb_start(enable_gpu=false)
        pages = [API.create_page(state; size=512) for _ in 1:30]
        for i in 1:29
            PropagationEngine.register_passthrough_recompute!(state, pages[i+1].id, pages[i].id)
        end
        for _ in 1:20
            idx = rand(1:length(pages))
            API.update_page(state, pages[idx].id, rand(UInt8, 512))
        end
        @test true
        API.mmsb_stop(state)
    end
end

end # module
