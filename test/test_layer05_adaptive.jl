"""
Layer 5: Adaptive Memory - Test Suite

Tests page reordering, clustering, and locality optimization.
"""

using Test
const MMSB = Main.MMSB

@testset "Layer 5: Adaptive Memory" begin
    @testset "Memory Layout Basics" begin
        state = MMSB.AdaptiveLayout.LayoutState(4096)
        @test state.page_size == 4096
        @test isempty(state.placement)
        @test state.locality_score == 0.0
    end

    @testset "Locality Score Computation" begin
        state = MMSB.AdaptiveLayout.LayoutState(4096)
        state.placement[1] = 0x0000
        state.placement[2] = 0x1000  # 1 page away
        state.placement[3] = 0x3000  # 3 pages away from p1
        
        # Access pattern: p1-p2 accessed 10 times, p2-p3 accessed 5 times
        pattern = Dict((UInt64(1), UInt64(2)) => 10,
                       (UInt64(2), UInt64(3)) => 5)
        
        score = MMSB.AdaptiveLayout.compute_locality_score(state, pattern)
        # Expected: 10*1 + 5*2 = 20
        @test score == 20.0
    end

    @testset "Page Reordering Optimization" begin
        state = MMSB.AdaptiveLayout.LayoutState(4096)
        # Initial bad layout: hot pages far apart
        state.placement[1] = 0x0000
        state.placement[2] = 0x10000  # 16 pages away
        state.placement[3] = 0x8000   # 8 pages away
        
        # Hottest pair: p1-p2 (100 accesses)
        pattern = Dict((UInt64(1), UInt64(2)) => 100,
                       (UInt64(1), UInt64(3)) => 10,
                       (UInt64(2), UInt64(3)) => 5)
        
        old_score = MMSB.AdaptiveLayout.compute_locality_score(state, pattern)
        ratio = MMSB.AdaptiveLayout.optimize_layout!(state, pattern)
        new_score = MMSB.AdaptiveLayout.compute_locality_score(state, pattern)
        
        @test new_score < old_score  # Optimization improves layout
        @test ratio < 1.0  # Improvement ratio < 1
        @test state.locality_score == new_score
    end

    @testset "Empty Pattern Handling" begin
        state = MMSB.AdaptiveLayout.LayoutState(4096)
        state.placement[1] = 0x0000
        
        pattern = Dict{Tuple{UInt64, UInt64}, Int}()
        ratio = MMSB.AdaptiveLayout.optimize_layout!(state, pattern)
        @test ratio ≈ 1.0  # No change for empty pattern
    end
end
