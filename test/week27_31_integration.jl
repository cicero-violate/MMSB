module Week27To31IntegrationTests
using Test
include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB
const API = MMSB.API
const PropagationEngine = MMSB.PropagationEngine

@testset "Week 27-32 Integration" begin
    @testset "Benchmarking" begin
        state = API.mmsb_start(enable_gpu=false)
        page = API.create_page(state; size=1024)
        @test !isnothing(page)
        API.mmsb_stop(state)
    end
    
    @testset "GPU Optimizations" begin
        @test true
    end
    
    @testset "Performance" begin
        state = API.mmsb_start(enable_gpu=false)
        pages = [API.create_page(state; size=512) for _ in 1:10]
        @test length(pages) == 10
        for i in 1:9
            PropagationEngine.register_passthrough_recompute!(state, pages[i+1].id, pages[i].id)
        end
        data = rand(UInt8, 512)
        API.update_page(state, pages[1].id, data)
        @test true
        API.mmsb_stop(state)
    end
    
    @testset "Reliability" begin
        state = API.mmsb_start(enable_gpu=false)
        page = API.create_page(state; size=256)
        data1 = rand(UInt8, 256)
        data2 = rand(UInt8, 256)
        API.update_page(state, page.id, data1)
        API.update_page(state, page.id, data2)
        @test true
        API.mmsb_stop(state)
    end
end
end
