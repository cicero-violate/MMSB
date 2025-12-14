using Test
using .MMSB
using .MMSB.PageTypes: read_page

function _checkpoint_fuzz_state(path_prefix::String=tempname())
    config = MMSB.MMSBStateTypes.MMSBConfig(tlog_path=path_prefix)
    MMSB.MMSBStateTypes.MMSBState(config)
end

@testset "Fuzz: checkpoint/load/replay cycles" begin
    state = _checkpoint_fuzz_state()

    # Create a few pages
    pages = PageID[]
    for _ in 1:3
        p = MMSB.PageAllocator.create_cpu_page!(state, 128)
        push!(pages, p.id)
    end

    deltas = MMSB.DeltaTypes.Delta[]

    # Generate random deltas
    for i in 1:150
        pid = rand(pages)
        mask = falses(128)
        mask[rand(1:128, rand(1:32))] .= true
        payload = rand(UInt8,128)
        d = MMSB.DeltaRouter.create_delta(state, pid, collect(mask), payload; source=:ckpt_fuzz)
        push!(deltas, d)
        MMSB.DeltaRouter.route_delta!(state,d)
        if i % 30 == 0
            GC.gc()
        end
    end

    # Choose a random checkpoint epoch
    target_epoch = UInt32(rand(1:length(deltas)))
    path = joinpath(mktempdir(), "ckpt.bin")

    # Write checkpoint
    GC.gc()
    MMSB.TLog.checkpoint_log!(state, path)
    GC.gc()

    # Load checkpoint into a fresh state and replay
    new_state = _checkpoint_fuzz_state()
    GC.gc()
    MMSB.TLog.load_checkpoint!(new_state, path)
    GC.gc()
    rep = MMSB.ReplayEngine.replay_to_epoch(new_state, target_epoch)

    # Structural sanity: all pages must exist and be readable
    for pid in pages
        pg = get(rep.pages, pid, nothing)
        @test pg !== nothing
        @test length(read_page(pg)) == 128
    end
end
