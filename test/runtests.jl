using Test
using Base.Threads
import CUDA

include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB
using .MMSB.API
const PageID = MMSB.PageTypes.PageID
using .MMSB.PageTypes: read_page

const CUDA_AVAILABLE = try
    CUDA.functional()
catch
    false
end

@testset "Replay and Diff" begin
    state = MMSB.MMSBStateTypes.MMSBState(MMSB.MMSBStateTypes.MMSBConfig(tlog_path=tempname()))
    page = MMSB.PageAllocator.create_cpu_page!(state, 16)
    mask = falses(16); mask[1:6] .= true
    data = UInt8.(1:16)
    delta = MMSB.DeltaRouter.create_delta(state, page.id, collect(mask), data; source=:test)
    MMSB.DeltaRouter.route_delta!(state, delta)

    replayed = MMSB.ReplayEngine.replay_to_epoch(state, delta.epoch)
    GC.@preserve replayed begin
        replay_page = get(replayed.pages, page.id, nothing)
        @test replay_page !== nothing
        @test read_page(replay_page) == read_page(page)
    end

    baseline = MMSB.MMSBStateTypes.MMSBState(MMSB.MMSBStateTypes.MMSBConfig(tlog_path=tempname()))
    baseline_page = MMSB.PageAllocator.create_cpu_page!(baseline, 16)
    @test read_page(baseline_page) != read_page(page)

    # TODO(P8.4): re-enable once compute_diff is implemented
    # diffs = MMSB.ReplayEngine.compute_diff(baseline, state)
    # @test length(diffs) == 1
    # @test all(MMSB.DeltaTypes.dense_data(diffs[1])[1:6] .== UInt8.(1:6))

    mktemp() do path, io
        close(io)
        MMSB.TLog.checkpoint_log!(state, path)
        restored = MMSB.MMSBStateTypes.MMSBState()
        MMSB.TLog.load_checkpoint!(restored, path)
        GC.@preserve restored begin
            restored_page = get(restored.pages, page.id, nothing)
            @test restored_page !== nothing
            @test read_page(restored_page) == read_page(page)
            summary = MMSB.TLog.log_summary(state)
            @test summary.total_deltas == 1
        end
    end
end

@testset "Checkpoint error mapping" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    mktemp() do path, io
        close(io)
        open(path, "w") do file
            write(file, "corrupt checkpoint data")
        end
        @test_throws MMSB.ErrorTypes.SerializationError MMSB.TLog.load_checkpoint!(state, path)
    end
end

@testset "Thread-safe allocator" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    task_count = max(Threads.nthreads(), 1) * 4
    Threads.@threads for _ in 1:task_count
        page = MMSB.PageAllocator.create_cpu_page!(state, 8)
        MMSB.PageAllocator.resize_page!(state, page.id, 16)
        MMSB.PageAllocator.resize_page!(state, page.id, 8)
    end
    ids = sort!(collect(keys(state.pages)))
    @test length(ids) == task_count
    @test ids == collect(1:task_count)
    for pid in ids
        MMSB.PageAllocator.delete_page!(state, pid)
    end
    @test isempty(state.pages)
end

@testset "GPU delta kernels" begin
    if CUDA_AVAILABLE
        state = MMSB.MMSBStateTypes.MMSBState()
        page = MMSB.PageAllocator.create_gpu_page!(state, 8)
        mask = falses(8); mask[1:5] .= true
        data = UInt8.(21:28)
        delta = MMSB.DeltaRouter.create_delta(state, page.id, collect(mask), data; source=:gpu_test)
        MMSB.DeltaRouter.route_delta!(state, delta)
        CUDA.synchronize()
        host = read_page(page)
        @test host[1:5] == data[1:5]
    else
        @info "CUDA not functional; skipping GPU kernel tests"
        @test true
    end
end

@testset "Propagation recompute flow" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    parent = MMSB.PageAllocator.create_cpu_page!(state, 8)
    child = MMSB.PageAllocator.create_cpu_page!(state, 8)
    MMSB.GraphTypes.add_dependency!(state.graph, parent.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)
    MMSB.PropagationEngine.register_passthrough_recompute!(state, child.id, parent.id)

    mask = falses(8); mask[1:3] .= true
    data = UInt8.(11:18)
    delta = MMSB.DeltaRouter.create_delta(state, parent.id, collect(mask), data; source=:test)
    MMSB.DeltaRouter.route_delta!(state, delta)

    @test read_page(child)[1:3] == data[1:3]
    @test !get(child.metadata, :stale, false)
end

@testset "Propagation optimization behaviors" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    parent_a = MMSB.PageAllocator.create_cpu_page!(state, 4)
    parent_b = MMSB.PageAllocator.create_cpu_page!(state, 4)
    child = MMSB.PageAllocator.create_cpu_page!(state, 4)
    MMSB.GraphTypes.add_dependency!(state.graph, parent_a.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)
    MMSB.GraphTypes.add_dependency!(state.graph, parent_b.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)

    hits = Ref(0)
    MMSB.PropagationEngine.register_recompute_fn!(state, child.id,
        function (st, _)
            hits[] += 1
            page_a = MMSB.MMSBStateTypes.get_page(st, parent_a.id)
            page_b = MMSB.MMSBStateTypes.get_page(st, parent_b.id)
            return Vector{UInt8}(read_page(page_a) .+ read_page(page_b))
        end)

    mask = trues(4)
    payload_a = fill(UInt8(0x01), 4)
    payload_b = fill(UInt8(0x02), 4)
    delta_a = MMSB.DeltaRouter.create_delta(state, parent_a.id, mask, payload_a; source=:batch)
    delta_b = MMSB.DeltaRouter.create_delta(state, parent_b.id, mask, payload_b; source=:batch)
    MMSB.DeltaRouter.batch_route_deltas!(state, [delta_a, delta_b])

    @test hits[] == 1
    @test read_page(child) == fill(UInt8(0x03), 4)

    MMSB.PropagationEngine.queue_recomputation!(state, child.id)
    MMSB.PropagationEngine.queue_recomputation!(state, child.id)
    MMSB.PropagationEngine.execute_propagation!(state)
    @test hits[] == 2
end

@testset "Error handling" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    @test_throws MMSB.ErrorTypes.PageNotFoundError MMSB.PageAllocator.delete_page!(state, PageID(99))

    page = MMSB.PageAllocator.create_cpu_page!(state, 4)
    bad_mask = falses(2)
    bad_data = UInt8.(1:4)
    @test_throws MMSB.ErrorTypes.InvalidDeltaError MMSB.DeltaRouter.create_delta(state, page.id, bad_mask, bad_data; source=:test)

    mktemp() do path, io
        close(io)
        open(path, "w") do f
            write(f, "corrupt")
        end
        @test_throws MMSB.ErrorTypes.SerializationError MMSB.TLog.load_checkpoint!(state, path)
    end

    graph = state.graph
    parent = MMSB.PageAllocator.create_cpu_page!(state, 2)
    child = MMSB.PageAllocator.create_cpu_page!(state, 2)
    MMSB.GraphTypes.add_dependency!(graph, parent.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)
    @test_throws MMSB.ErrorTypes.GraphCycleError MMSB.GraphTypes.add_dependency!(graph, child.id, parent.id, MMSB.GraphTypes.DATA_DEPENDENCY)
end

@testset "Public API" begin
    state = API.mmsb_start(enable_gpu=false)
    page = API.create_page(state; size=4)
    API.update_page(state, page.id, UInt8[0x01, 0x02, 0x03, 0x04])
    @test API.query_page(state, page.id) == UInt8[0x01, 0x02, 0x03, 0x04]
    API.@mmsb state begin
        @test API.ACTIVE_STATE[] === state
        nothing
    end
    @test isnothing(API.ACTIVE_STATE[])
    mktemp() do path, io
        close(io)
        API.mmsb_stop(state; checkpoint_path=path)
        @test isfile(path)
    end
end

@testset "Monitoring" begin
    config = MMSB.MMSBStateTypes.MMSBConfig(enable_gpu=false, tlog_path=tempname())
    state = API.mmsb_start(config=config)
    page = API.create_page(state; size=2)
    API.update_page(state, page.id, UInt8[0x0a, 0x0b])
    stats = MMSB.get_stats(state)
    @test stats.total_pages == 1
    @test stats.cpu_pages == 1
    @test stats.total_deltas == 1
    @test stats.delta_apply_count >= 1
    MMSB.reset_stats!(state)
    stats2 = MMSB.get_stats(state)
    @test stats2.delta_apply_count == 0
end

@testset "Benchmark Smoke Test" begin
    config = MMSB.MMSBStateTypes.MMSBConfig(enable_gpu=false, tlog_path=tempname())
    state = API.mmsb_start(config=config)
    base_page = API.create_page(state; size=1024)
    payload = rand(UInt8, 1024)
    alloc_time = @elapsed API.create_page(state; size=2048)
    @test alloc_time < 1.0
    delta_time = @elapsed API.update_page(state, base_page.id, payload)
    @test delta_time < 1.0
    stats = MMSB.get_stats(state)
    @test stats.total_pages >= 2
    API.mmsb_stop(state)
end

include(joinpath(@__DIR__, "gc_stress_test.jl"))
include(joinpath(@__DIR__, "fuzz_replay.jl"))
include(joinpath(@__DIR__, "propagation_fuzz.jl"))
include(joinpath(@__DIR__, "checkpoint_fuzz.jl"))

include(joinpath(@__DIR__, "test_layer05_adaptive.jl"))
include(joinpath(@__DIR__, "test_layer06_utility.jl"))

# include(joinpath(@__DIR__, "test_week24_25_integration.jl"))

# Phase 4 tests
# include(joinpath(@__DIR__, "test_layer10_agent_interface.jl"))
# include(joinpath(@__DIR__, "test_layer11_agents.jl"))
# include(joinpath(@__DIR__, "test_layer12_applications.jl"))
# include(joinpath(@__DIR__, "test_phase4_integration.jl"))
