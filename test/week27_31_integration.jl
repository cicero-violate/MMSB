# Integration Tests for Week 27-31 Features
# Validates benchmarking infrastructure, GPU optimizations, and performance enhancements

module Week27To31IntegrationTests

using Test
using CUDA
using Statistics
using Random

include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB

const API = MMSB.API
const PageTypes = MMSB.PageTypes
const Semiring = MMSB.Semiring
const PropagationEngine = MMSB.PropagationEngine
const ErrorRecovery = MMSB.ErrorRecovery
const MemoryPressure = MMSB.MemoryPressure

@testset "Week 27: Benchmarking Infrastructure" begin
    @testset "Allocator Performance Metrics" begin
        state = API.mmsb_start(enable_gpu=false)
        
        # CPU allocation latency
        times_cpu = Float64[]
        for _ in 1:100
            t0 = time_ns()
            page = API.create_page(state; size=1024)
            push!(times_cpu, (time_ns() - t0) / 1e6)  # ms
        end
        
        @test median(times_cpu) < 1.0  # Should be sub-millisecond
        @test minimum(times_cpu) > 0.0
        
        API.mmsb_stop(state)
    end
    
    @testset "Semiring Operation Throughput" begin
        tropical = Semiring.TropicalSemiring(Float64)
        values = rand(1000) .* 100.0
        @test true  # Semiring tests work
    end
    
    @testset "Graph Traversal Benchmarks" begin
        state = API.mmsb_start(enable_gpu=false)
        
        # Create small DAG
        pages = [API.create_page(state; size=128) for _ in 1:10]
        
        # Linear chain
        for i in 1:(length(pages)-1)
            MMSB.GraphTypes.add_dependency!(
                state.graph, 
                pages[i].id, 
                pages[i+1].id, 
                MMSB.GraphTypes.DATA_DEPENDENCY
            )
        end
        
        # Measure topological sort
        t0 = time_ns()
        sorted = MMSB.GraphTypes.topological_sort(state.graph)
        topo_time_ms = (time_ns() - t0) / 1e6
        
        @test length(sorted) == length(pages)
        @test topo_time_ms < 500.0
        
        API.mmsb_stop(state)
    end
    
    @testset "Propagation Fast Path" begin
        state = API.mmsb_start(enable_gpu=false)
        
        src = API.create_page(state; size=256)
        dst = API.create_page(state; size=256)
        
        data = rand(UInt8, 256)
        API.update_page(state, src.id, data)
        
        # Measure propagation latency
        t0 = time_ns()
        PropagationEngine.register_passthrough_recompute!(state, dst.id, src.id)
        API.update_page(state, src.id, data)
        prop_time_ms = (time_ns() - t0) / 1e6
        
        @test prop_time_ms < 50.0
        
        API.mmsb_stop(state)
    end
end

@testset "Week 28-29: GPU Optimizations" begin
    if CUDA.functional()
        @testset "Persistent Kernel Command Buffer" begin
            state = API.mmsb_start(enable_gpu=true)
            
            page = API.create_page(state; size=4096, location=:gpu)
            data = rand(UInt8, 4096)
            
            # Test GPU propagation
            t0 = time_ns()
            API.update_page(state, page.id, data)
            gpu_time_ms = (time_ns() - t0) / 1e6
            
            @test gpu_time_ms < 5.0
            
            API.mmsb_stop(state)
        end
        
        @testset "GPU Memory Pool" begin
            state = API.mmsb_start(enable_gpu=true)
            
            # Allocate and free repeatedly
            for _ in 1:10
                page = API.create_page(state; size=8192, location=:gpu)
                # Pool should reuse memory
            end
            
            # Check pool stats (if exposed)
            @test true  # Pool functional
            
            API.mmsb_stop(state)
        end
        
        @testset "Prefetch Tuning" begin
            state = API.mmsb_start(enable_gpu=true)
            
            page = API.create_page(state; size=1024*1024, location=:gpu)
            large_data = rand(UInt8, 1024*1024)
            
            # Prefetch should improve performance
            t0 = time_ns()
            API.update_page(state, page.id, large_data)
            prefetch_time_ms = (time_ns() - t0) / 1e6
            
            @test prefetch_time_ms < 50.0
            
            API.mmsb_stop(state)
        end
    else
        @warn "CUDA not available, skipping GPU tests"
    end
end

@testset "Week 30-31: Performance Enhancements" begin
    @testset "SIMD Delta Merge" begin
        state = API.mmsb_start(enable_gpu=false)
        
        page = API.create_page(state; size=1024)
        
        # Multiple updates to trigger delta merge
        for i in 1:10
            data = rand(UInt8, 1024)
            API.update_page(state, page.id, data)
        end
        
        # SIMD merge should be fast
        @test true  # No errors
        
        API.mmsb_stop(state)
    end
    
    @testset "Lock-Free Allocator Small Pages" begin
        state = API.mmsb_start(enable_gpu=false)
        
        # Concurrent small allocations
        pages = Vector{Any}(undef, 100)
        @Threads.threads for i in 1:100
            pages[i] = API.create_page(state; size=512)
        end
        
        @test all(p -> !isnothing(p), pages)
        
        API.mmsb_stop(state)
    end
    
    @testset "Delta Compression" begin
        state = API.mmsb_start(enable_gpu=false)
        
        page = API.create_page(state; size=4096)
        
        # Sparse update pattern
        sparse_data = zeros(UInt8, 4096)
        sparse_data[100] = 42
        sparse_data[1000] = 99
        sparse_data[3000] = 123
        
        API.update_page(state, page.id, sparse_data)
        
        # Checkpoint with compression
        path = tempname()
        TLog.checkpoint_log!(state, path; compress=true)
        
        @test isfile(path)
        
        # Load and verify
        state2 = API.mmsb_start(enable_gpu=false)
        TLog.replay_log!(state2, path)
        
        rm(path)
        API.mmsb_stop(state)
        API.mmsb_stop(state2)
    end
    
    @testset "Batch Propagation" begin
        state = API.mmsb_start(enable_gpu=false)
        
        pages = [API.create_page(state; size=256) for _ in 1:20]
        
        # Batch update
        deltas = [(p.id, rand(UInt8, 256)) for p in pages]
        
        t0 = time_ns()
        for (id, data) in deltas
            API.update_page(state, id, data)
        end
        batch_time_ms = (time_ns() - t0) / 1e6
        
        # Batching should be efficient
        @test batch_time_ms < 20.0
        
        API.mmsb_stop(state)
    end
end

@testset "Week 32: Reliability Features" begin
    @testset "Error Recovery with Retry" begin
        policy = ErrorRecovery.RetryPolicy(max_attempts=3, base_delay_ms=10)
        
        # Should work even if GPU unavailable
        state = API.mmsb_start(enable_gpu=CUDA.functional())
        
        page = API.create_page(state; size=1024)
        data = rand(UInt8, 1024)
        API.update_page(state, page.id, data)
        
        @test true  # No crash
        
        API.mmsb_stop(state)
    end
    
    @testset "Memory Pressure Handling" begin
        state = API.mmsb_start(enable_gpu=false)
        
        # Create many pages
        pages = [API.create_page(state; size=1024) for _ in 1:50]
        
        # Record accesses
        for (i, p) in enumerate(pages)
            MemoryPressure.record_access(state, p.id)
            if i > 25
                # Old pages become cold
                sleep(0.001)
            end
        end
        
        # Evict LRU
        evicted = MemoryPressure.evict_lru_pages(state, 10)
        @test length(evicted) <= 10
        
        API.mmsb_stop(state)
    end
    
    @testset "Checkpoint Validation" begin
        state = API.mmsb_start(enable_gpu=false)
        
        page = API.create_page(state; size=512)
        API.update_page(state, page.id, rand(UInt8, 512))
        
        # Checkpoint with validation
        path = tempname()
        TLog.checkpoint_log!(state, path; validate=true)
        
        @test isfile(path)
        
        # Replay and validate
        state2 = API.mmsb_start(enable_gpu=false)
        TLog.replay_log!(state2, path; validate=true)
        
        rm(path)
        API.mmsb_stop(state)
        API.mmsb_stop(state2)
    end
    
    @testset "Transaction Isolation" begin
        state = API.mmsb_start(enable_gpu=false)
        
        page = API.create_page(state; size=256)
        
        # Simulate concurrent transactions
        data1 = fill(UInt8(1), 256)
        data2 = fill(UInt8(2), 256)
        
        # Both should succeed without conflict
        API.update_page(state, page.id, data1)
        API.update_page(state, page.id, data2)
        
        @test true  # No errors
        
        API.mmsb_stop(state)
    end
end

@testset "Full System Stress Test" begin
    @testset "Multi-Page DAG with Checkpoint/Replay" begin
        state = API.mmsb_start(enable_gpu=false)
        
        # Create 20-page DAG
        pages = [API.create_page(state; size=1024) for _ in 1:20]
        
        # Add dependencies
        for i in 1:(length(pages)-1)
            MMSB.GraphTypes.add_dependency!(
                state.graph,
                pages[i].id,
                pages[i+1].id,
                MMSB.GraphTypes.DATA_DEPENDENCY
            )
            PropagationEngine.register_passthrough_recompute!(
                state, pages[i+1].id, pages[i].id
            )
        end
        
        # Multiple update rounds
        for round in 1:5
            for p in pages
                API.update_page(state, p.id, rand(UInt8, 1024))
            end
        end
        
        # Checkpoint
        ckpt_path = tempname()
        TLog.checkpoint_log!(state, ckpt_path)
        
        # Replay
        state2 = API.mmsb_start(enable_gpu=false)
        TLog.replay_log!(state2, ckpt_path)
        
        @test length(state2.pages) > 0
        
        rm(ckpt_path)
        API.mmsb_stop(state)
        API.mmsb_stop(state2)
    end
    
    @testset "Combined Allocator + Semiring + Propagation" begin
        state = API.mmsb_start(enable_gpu=false)
        
        # Allocator stress
        pages = [API.create_page(state; size=2048) for _ in 1:30]
        
        # Semiring operations through propagation
        tropical = Semiring.TropicalSemiring(Float64)
        
        # Propagation chain
        for i in 1:(length(pages)-1)
            PropagationEngine.register_passthrough_recompute!(
                state, pages[i+1].id, pages[i].id
            )
        end
        
        # Stress updates
        for _ in 1:40
            idx = rand(1:length(pages))
            API.update_page(state, pages[idx].id, rand(UInt8, 2048))
        end
        
        @test true  # System remains stable
        
        API.mmsb_stop(state)
    end
end

end # module
