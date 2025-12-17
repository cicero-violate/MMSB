# Integration test for state pool with realistic workflow

using Test
using Random
import ..MMSB

"""
Simulate realistic MMSB workflow: create pages, build DAG, propagate changes.
"""
function realistic_workflow(state::MMSB.MMSBStateTypes.MMSBState, seed::Int)
    Random.seed!(seed)
    
    # Phase 1: Build computation graph
    input_pages = [MMSB.API.create_page(state; size=4096, location=:cpu) for _ in 1:3]
    intermediate_pages = [MMSB.API.create_page(state; size=4096, location=:cpu) for _ in 1:5]
    output_pages = [MMSB.API.create_page(state; size=4096, location=:cpu) for _ in 1:2]
    
    # Phase 2: Set up dependencies (tree structure)
    for inter in intermediate_pages[1:3]
        MMSB.PropagationEngine.register_passthrough_recompute!(state, inter.id, input_pages[1].id)
    end
    for inter in intermediate_pages[4:5]
        MMSB.PropagationEngine.register_passthrough_recompute!(state, inter.id, input_pages[2].id)
    end
    for output in output_pages
        MMSB.PropagationEngine.register_passthrough_recompute!(state, output.id, intermediate_pages[1].id)
        MMSB.PropagationEngine.register_passthrough_recompute!(state, output.id, intermediate_pages[4].id)
    end
    
    # Phase 3: Execute computation
    for input in input_pages
        data = rand(UInt8, 4096)
        MMSB.API.update_page(state, input.id, data)
    end
    
    # Phase 4: Trigger propagation
    for inter in intermediate_pages
        MMSB.PropagationEngine.recompute_page!(state, inter.id)
    end
    for output in output_pages
        MMSB.PropagationEngine.recompute_page!(state, output.id)
    end
    
    return (
        inputs = [p.id for p in input_pages],
        intermediates = [p.id for p in intermediate_pages],
        outputs = [p.id for p in output_pages],
        total_pages = length(state.pages),
        next_page_id = state.next_page_id[],
    )
end

@testset "State Pool Integration Tests" begin
    @testset "Full workflow with pool reuse" begin
        config = MMSB.MMSBStateTypes.MMSBConfig()
        results = []
        
        # Run workflow 3 times through pool
        for i in 1:3
            state = MMSB.StateManagement.get_pooled_state!(config)
            result = realistic_workflow(state, 42)
            push!(results, result)
            MMSB.StateManagement.return_to_pool!(state)
        end
        
        # All runs should produce identical structure
        @test all(r.total_pages == results[1].total_pages for r in results)
        @test all(r.next_page_id == results[1].next_page_id for r in results)
    end
    
    @testset "Allocation speedup verification" begin
        config = MMSB.MMSBStateTypes.MMSBConfig()
        
        # Measure fresh allocation
        fresh_times = Float64[]
        for _ in 1:10
            t0 = time_ns()
            state = MMSB.MMSBStateTypes.MMSBState(config)
            t1 = time_ns()
            push!(fresh_times, (t1 - t0) / 1e3)  # μs
        end
        
        # Warm up pool
        for _ in 1:5
            state = MMSB.StateManagement.get_pooled_state!(config)
            MMSB.StateManagement.return_to_pool!(state)
        end
        
        # Measure pooled allocation
        pooled_times = Float64[]
        for _ in 1:10
            t0 = time_ns()
            state = MMSB.StateManagement.get_pooled_state!(config)
            t1 = time_ns()
            push!(pooled_times, (t1 - t0) / 1e3)  # μs
            MMSB.StateManagement.return_to_pool!(state)
        end
        
        using Statistics
        fresh_median = median(fresh_times)
        pooled_median = median(pooled_times)
        speedup = fresh_median / pooled_median
        
        println("Fresh allocation: $(round(fresh_median, digits=2))μs")
        println("Pooled allocation: $(round(pooled_median, digits=2))μs")
        println("Speedup: $(round(speedup, digits=2))x")
        
        # Should see speedup (target: 6μs → 2-3μs = 2-3x)
        @test pooled_median < fresh_median
        @test pooled_median < 5.0  # Target: < 5μs
    end
    
    @testset "Pool with checkpoint/replay cycle" begin
        config = MMSB.MMSBStateTypes.MMSBConfig(tlog_path=tempname())
        
        # Execute workflow and checkpoint
        state1 = MMSB.StateManagement.get_pooled_state!(config)
        result1 = realistic_workflow(state1, 777)
        
        ckpt_path = joinpath(mktempdir(), "integration.ckpt")
        MMSB.TLog.checkpoint_log!(state1, ckpt_path)
        MMSB.StateManagement.return_to_pool!(state1)
        
        # Load and replay on pooled state
        state2 = MMSB.StateManagement.get_pooled_state!(config)
        MMSB.TLog.load_checkpoint!(state2, ckpt_path)
        
        # Verify loaded state
        @test length(state2.pages) == result1.total_pages
        
        # Execute replay
        target_epoch = UInt32(5)
        state_replayed = MMSB.ReplayEngine.replay_to_epoch(state2, target_epoch)
        
        # Verify replay produced valid state
        @test !isempty(state_replayed.pages)
        for (pid, page) in state_replayed.pages
            @test page.size > 0
            data = MMSB.PageTypes.read_page(page)
            @test length(data) == page.size
        end
        
        MMSB.StateManagement.return_to_pool!(state2)
    end
    
    @testset "Mixed fresh and pooled states" begin
        config = MMSB.MMSBStateTypes.MMSBConfig()
        
        # Create mix of fresh and pooled states
        states = []
        for i in 1:6
            if i % 2 == 0
                push!(states, MMSB.StateManagement.get_pooled_state!(config))
            else
                push!(states, MMSB.MMSBStateTypes.MMSBState(config))
            end
        end
        
        # Execute same workflow on all
        results = [realistic_workflow(s, 555) for s in states]
        
        # All should produce identical results
        reference = results[1]
        for r in results[2:end]
            @test r.total_pages == reference.total_pages
            @test r.next_page_id == reference.next_page_id
        end
        
        # Return pooled states
        for (i, state) in enumerate(states)
            if i % 2 == 0
                MMSB.StateManagement.return_to_pool!(state)
            end
        end
    end
    
    @testset "Pool under memory pressure" begin
        config = MMSB.MMSBStateTypes.MMSBConfig()
        
        # Create large working set
        for iteration in 1:20
            state = MMSB.StateManagement.get_pooled_state!(config)
            
            # Allocate significant memory
            pages = []
            for _ in 1:30
                p = MMSB.API.create_page(state; size=8192, location=:cpu)
                push!(pages, p.id)
                MMSB.API.update_page(state, p.id, rand(UInt8, 8192))
            end
            
            # Trigger GC periodically
            if iteration % 5 == 0
                GC.gc()
            end
            
            # Verify state integrity
            @test length(state.pages) == length(pages)
            
            MMSB.StateManagement.return_to_pool!(state)
        end
        
        # Final GC and verification
        GC.gc()
        @test true  # If we got here, pool survived memory pressure
    end
end
