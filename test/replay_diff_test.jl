using Test
import MMSB

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
        @test MMSB.PageTypes.read_page(replay_page) == MMSB.PageTypes.read_page(page)
    end

    baseline = MMSB.MMSBStateTypes.MMSBState(MMSB.MMSBStateTypes.MMSBConfig(tlog_path=tempname()))
    baseline_page = MMSB.PageAllocator.create_cpu_page!(baseline, 16)
    @test MMSB.PageTypes.read_page(baseline_page) != MMSB.PageTypes.read_page(page)

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
            @test MMSB.PageTypes.read_page(restored_page) == MMSB.PageTypes.read_page(page)
            summary = MMSB.TLog.log_summary(state)
            @test summary.total_deltas == 1
        end
    end
end
