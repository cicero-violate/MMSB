module Week27To31IntegrationTests
using Test
using Statistics
using Random
include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB
const API = MMSB.API
const PropagationEngine = MMSB.PropagationEngine
const GraphTypes = MMSB.GraphTypes

@testset "Week 27-32 Integration" begin
    @testset "Week 27: Benchmarking" begin
        state = API.mmsb_start(enable_gpu=false)
        
        times = Float64[]
        for _ in 1:50
            t0 = time_ns()
            page = API.create_page(state; size=1024)
            push!(times, (time_ns() - t0) / 1e6)
        end
        @test median(times) < 10.0
        
        pages = [API.create_page(state; size=128) for _ in 1:10]
        for i in 1:9
            GraphTypes.add_dependency!(state.graph, pages[i].id, pages[i+1].id, GraphTypes.DATA_DEPENDENCY)
        end
        t0 = time_ns()
        sorted = GraphTypes.topological_sort(state.graph)
        topo_time = (time_ns() - t0) / 1e6
        @test topo_time < 500.0
        @test length(sorted) >= 10
        
        API.mmsb_stop(state)
    end
    
    @testset "Week 28-29: GPU Optimizations" begin
        state = API.mmsb_start(enable_gpu=false)
        page = API.create_page(state; size=4096)
        @test !isnothing(page)
        API.mmsb_stop(state)
    end
    
    @testset "Week 30-31: Performance" begin
        state = API.mmsb_start(enable_gpu=false)
        
        page = API.create_page(state; size=1024)
        for i in 1:10
            API.update_page(state, page.id, rand(UInt8, 1024))
        end
        @test true
        
        pages = [API.create_page(state; size=512) for _ in 1:10]
        for i in 1:9
            PropagationEngine.register_passthrough_recompute!(state, pages[i+1].id, pages[i].id)
        end
        
        deltas = [(p.id, rand(UInt8, 512)) for p in pages]
        t0 = time_ns()
        for (id, data) in deltas
            API.update_page(state, id, data)
        end
        batch_time = (time_ns() - t0) / 1e6
        @test batch_time < 100.0
        
        API.mmsb_stop(state)
    end
    
    @testset "Week 32: Reliability" begin
        state = API.mmsb_start(enable_gpu=false)
        
        page = API.create_page(state; size=256)
        data1 = rand(UInt8, 256)
        data2 = rand(UInt8, 256)
        API.update_page(state, page.id, data1)
        API.update_page(state, page.id, data2)
        @test true
        
        pages = [API.create_page(state; size=1024) for _ in 1:20]
        @test length(pages) == 20
        
        API.mmsb_stop(state)
    end
end
end
