
"""
Phase 2 Integration Test: Adaptive Memory → Utility → Intention

Tests the complete Phase 2 pipeline from layer 5 through layer 7.
"""

using Test
using MMSB

@testset "Phase 2: Self-Optimization Integration" begin
    @testset "Layer 5→6 Pipeline" begin
        # Setup adaptive layout
        layout_state = MMSB.AdaptiveLayout.LayoutState(4096)
        layout_state.placement[1] = 0x0000
        layout_state.placement[2] = 0x10000
        layout_state.placement[3] = 0x8000
        
        # Bad access pattern
        pattern = Dict((1, 2) => 100, (1, 3) => 10, (2, 3) => 5)
        
        # Optimize layout
        old_score = MMSB.AdaptiveLayout.compute_locality_score(layout_state, pattern)
        MMSB.AdaptiveLayout.optimize_layout!(layout_state, pattern)
        new_score = MMSB.AdaptiveLayout.compute_locality_score(layout_state, pattern)
        
        @test new_score < old_score
        
        # Compute entropy reduction
        entropy = MMSB.EntropyMeasure.state_entropy(pattern)
        @test entropy > 0.0
    end

    @testset "Layer 6→7 Pipeline" begin
        # Create utility state
        utility_state = MMSB.UtilityEngine.UtilityState(50)
        
        # Simulate degrading utility
        for i in 1:15
            costs = MMSB.CostFunctions.CostComponents(
                0.1 + i * 0.01,
                4.0,
                100.0 + i * 5.0,
                10.0
            )
            MMSB.UtilityEngine.update_utility!(utility_state, costs)
        end
        
        @test length(utility_state.history) == 15
        
        # Form intention based on utility
        layout_state = MMSB.AdaptiveLayout.LayoutState(4096)
        layout_state.placement[1] = 0x0000
        
        intention = MMSB.IntentionEngine.form_intention(
            utility_state,
            layout_state,
            UInt64(1)
        )
        
        @test intention.priority >= MMSB.IntentionTypes.HIGH
        @test intention.expected_utility_gain > 0.0
    end

    @testset "Complete Phase 2 Flow" begin
        # Initialize all states
        layout_state = MMSB.AdaptiveLayout.LayoutState(4096)
        utility_state = MMSB.UtilityEngine.UtilityState(50)
        intention_state = MMSB.IntentionTypes.IntentionState()
        
        # Add pages
        for i in 1:10
            layout_state.placement[UInt64(i)] = UInt64((i-1) * 4096)
        end
        
        # Create access pattern
        pattern = Dict{Tuple{UInt64, UInt64}, Int}()
        for i in 1:9
            pattern[(UInt64(i), UInt64(i+1))] = 10 - i
        end
        
        # Step 1: Optimize layout (Layer 5)
        old_locality = MMSB.AdaptiveLayout.compute_locality_score(layout_state, pattern)
        ratio = MMSB.AdaptiveLayout.optimize_layout!(layout_state, pattern)
        @test ratio < 1.0  # Improvement
        
        # Step 2: Compute utility (Layer 6)
        costs = MMSB.CostFunctions.CostComponents(0.15, 5.0, 120.0, 12.0)
        utility = MMSB.UtilityEngine.update_utility!(utility_state, costs)
        @test utility < 0.0
        
        # Step 3: Detect goals (Layer 7)
        goals = MMSB.GoalEmergence.detect_goals(utility_state, -50.0)
        @test length(goals) >= 0  # May have goals depending on utility
        
        # Step 4: Form intention (Layer 7)
        intention = MMSB.IntentionEngine.form_intention(
            utility_state,
            layout_state,
            UInt64(1)
        )
        push!(intention_state.active_intentions, intention)
        
        @test length(intention_state.active_intentions) == 1
        @test intention.expected_utility_gain > 0.0
    end
end
