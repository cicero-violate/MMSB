using Test
using MMSB
import MMSB: MMSBState, MMSBConfig, create_page, allocate_page_id!
import MMSB: MMSBContext, World, add_entity!, simulate_step!
import MMSB: AgentCoordinator, register_agent!, Portfolio, compute_value
import MMSB: ReasoningContext, reason_over_memory

@testset "Layer 12: Applications" begin
    config = MMSBConfig(tlog_path=tempname() * ".tlog")
    state = MMSBState(config)
    
    @testset "LLM Tools" begin
        ctx = MMSBContext(state, 2048)
        @test ctx.max_tokens == 2048
    end
    
    @testset "World Simulation" begin
        world = World(state, 0.01)
        @test world.dt == 0.01
        
        entity = add_entity!(world, :player, Dict(:health => 100))
        @test entity.entity_type == :player
        
        simulate_step!(world)
        @test world.time ≈ 0.01
    end
    
    @testset "Multi-Agent System" begin
        coord = AgentCoordinator(state, :sequential)
        @test coord.coordination_strategy == :sequential
    end
    
    @testset "Financial Modeling" begin
        portfolio = Portfolio(state, 10000.0)
        @test portfolio.cash == 10000.0
        
        prices = Dict("AAPL" => 150.0)
        value = compute_value(portfolio, prices)
        @test value == 10000.0
    end
    
    @testset "Memory-Driven Reasoning" begin
        ctx = ReasoningContext(state, "test query", (0, 1000))
        result = reason_over_memory(ctx)
        @test haskey(result, :query)
    end
end
