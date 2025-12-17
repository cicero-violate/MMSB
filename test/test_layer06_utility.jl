"""
Layer 6: Utility Engine - Test Suite
"""

using Test
const MMSB = Main.MMSB

@testset "Layer 6: Utility Engine" begin
    @testset "Cost Functions" begin
        # Cache cost
        cache_cost = MMSB.CostFunctions.compute_cache_cost(25, 75)
        @test cache_cost > 0.0
        @test cache_cost == 1000.0 * 0.25
        
        # Memory cost
        mem_cost = MMSB.CostFunctions.compute_memory_cost(4 * 1024 * 1024, 100)
        @test mem_cost > 0.0
        
        # Latency cost
        lat_cost = MMSB.CostFunctions.compute_latency_cost(10000, 10)
        @test lat_cost > 0.0
    end

    @testset "Entropy Measure" begin
        # Uniform distribution has maximum entropy
        pattern = Dict((UInt64(1), UInt64(2)) => 10,
                       (UInt64(2), UInt64(3)) => 10,
                       (UInt64(3), UInt64(4)) => 10)
        entropy = MMSB.EntropyMeasure.state_entropy(pattern)
        @test entropy > 0.0
        
        # Single access has zero entropy
        pattern_single = Dict((UInt64(1), UInt64(2)) => 100)
        entropy_single = MMSB.EntropyMeasure.state_entropy(pattern_single)
        @test entropy_single == 0.0
        
        # Entropy reduction
        reduction = MMSB.EntropyMeasure.entropy_reduction(2.0, 1.0)
        @test reduction == 0.5
    end

    @testset "Cost Aggregation" begin
        costs = [
            MMSB.CostAggregation.WeightedCost(:cache, 100.0, 1.0),
            MMSB.CostAggregation.WeightedCost(:memory, 50.0, 0.5),
            MMSB.CostAggregation.WeightedCost(:latency, 200.0, 2.0)
        ]
        
        total = MMSB.CostAggregation.aggregate_costs(costs)
        @test total == 100.0 * 1.0 + 50.0 * 0.5 + 200.0 * 2.0
        
        # Test normalization
        normalized = MMSB.CostAggregation.normalize_costs(costs)
        @test all(0.0 <= c.value <= 1.0 for c in normalized)
    end

    @testset "Utility Engine" begin
        state = MMSB.UtilityEngine.UtilityState(50)
        @test state.current_utility == 0.0
        @test isempty(state.history)
        
        # Update with costs
        costs = MMSB.CostFunctions.CostComponents(0.2, 4.0, 100.0, 10.0)
        utility = MMSB.UtilityEngine.update_utility!(state, costs)
        
        @test utility < 0.0  # Negative because costs are positive
        @test length(state.history) == 1
        @test state.current_utility == utility
    end
end
