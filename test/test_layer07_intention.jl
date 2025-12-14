"""
Layer 7: Intention Engine - Test Suite
"""

using Test
using MMSB

@testset "Layer 7: Intention Engine" begin
    @testset "Intention Types" begin
        intention = MMSB.IntentionTypes.Intention(
            UInt64(1),
            "Test intention",
            MMSB.IntentionTypes.HIGH,
            50.0,
            UInt64[1, 2, 3],
            time()
        )
        
        @test intention.id == 1
        @test intention.priority == MMSB.IntentionTypes.HIGH
        @test intention.expected_utility_gain == 50.0
    end

    @testset "Goal Emergence" begin
        utility_state = MMSB.UtilityEngine.UtilityState(50)
        utility_state.current_utility = 100.0
        
        goals = MMSB.GoalEmergence.detect_goals(utility_state, 50.0)
        @test length(goals) >= 1
        @test goals[1].description == "Maintain current high utility"
    end

    @testset "Attractor Dynamics" begin
        field = MMSB.AttractorStates.AttractorField(
            [[1.0, 1.0], [5.0, 5.0]],
            [1.0, 1.0]
        )
        
        state = [0.0, 0.0]
        gradient = MMSB.AttractorStates.compute_gradient(field, state)
        
        @test length(gradient) == 2
        @test all(gradient .!= 0.0)
        
        # Evolve state
        new_state = MMSB.AttractorStates.evolve_state(field, state, 0.1)
        @test new_state != state
    end

    @testset "Structural Preferences" begin
        layout_state = MMSB.AdaptiveLayout.LayoutState(4096)
        layout_state.placement[1] = 0x0000
        layout_state.locality_score = 10.0

        prefs = MMSB.StructuralPreferences.DEFAULT_PREFERENCES
        score = MMSB.StructuralPreferences.apply_preferences(prefs, layout_state)

        @test score < 0.0  # Negative because locality_score > 0
    end

    @testset "Upsert Plan validation" begin
        deltaspec = MMSB.UpsertPlan.DeltaSpec(1, UInt8[1, 2], [true, true])
        plan = MMSB.UpsertPlan.UpsertPlan("select *", x -> x > 0, deltaspec, Dict(:version => 1))
        @test MMSB.UpsertPlan.validate_plan(plan) === plan

        bad_spec = MMSB.UpsertPlan.DeltaSpec(1, UInt8[1], [true, true])
        bad_plan = MMSB.UpsertPlan.UpsertPlan("select *", x -> x > 0, bad_spec, Dict())
        @test_throws ErrorException MMSB.UpsertPlan.validate_plan(bad_plan)
    end

    @testset "Intent lowering" begin
        deltaspec = MMSB.UpsertPlan.DeltaSpec(42, UInt8[10, 11], [true, false])
        plan = MMSB.UpsertPlan.UpsertPlan("p", x -> x == 1, deltaspec, Dict(:intent => :test))
        lowered = MMSB.IntentLowering.lower_intent_to_deltaspec(plan)
        @test lowered.page == 42
        @test lowered.payload == UInt8[10, 11]
        @test lowered.mask_bytes == UInt8[1, 0]
        @test lowered.metadata[:intent] == :test
    end
end
