using Test
using .MMSB
using .MMSB.PageTypes: read_page

function _fuzz_state()
    config = MMSB.MMSBStateTypes.MMSBConfig(tlog_path=tempname())
    MMSB.MMSBStateTypes.MMSBState(config)
end

@testset "Fuzz: replay under random workloads" begin
    state = _fuzz_state()
    page = MMSB.PageAllocator.create_cpu_page!(state, 128)

    deltas = MMSB.DeltaTypes.Delta[]
    for i in 1:200
        mask = falses(128)
        mask[rand(1:128, rand(1:32))] .= true
        payload = rand(UInt8, 128)
        delta = MMSB.DeltaRouter.create_delta(state, page.id, collect(mask), payload; source=:fuzz)
        push!(deltas, delta)
        MMSB.DeltaRouter.route_delta!(state, delta)
        if i % 25 == 0
            GC.gc()
        end
    end

    # Randomized replay target - should not crash
    target_epoch = UInt32(maximum(d.epoch for d in deltas))
    GC.gc()
    replayed = MMSB.ReplayEngine.replay_to_epoch(state, target_epoch)
    GC.gc()
    replay_page = get(replayed.pages, page.id, nothing)
    @test replay_page !== nothing
    @test length(read_page(replay_page)) == 128
end
