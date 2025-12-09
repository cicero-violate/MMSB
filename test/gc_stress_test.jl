using Test
using Base.Threads
using .MMSB
using .MMSB.PageTypes: read_page

@testset "GC Stress Tests" begin
    @testset "Replay under GC pressure" begin
        state = MMSB.MMSBStateTypes.MMSBState()
        page = MMSB.PageAllocator.create_cpu_page!(state, 1024)

        # Create many deltas to stress GC while exercising FFI and replay
        for i in 1:1000
            mask = falses(1024)
            mask[rand(1:1024, 100)] .= true
            data = rand(UInt8, 1024)
            delta = MMSB.DeltaRouter.create_delta(state, page.id, collect(mask), data, :stress)
            MMSB.DeltaRouter.route_delta!(state, delta)

            # Force GC periodically to try to surface lifetime bugs
            if i % 100 == 0
                GC.gc()
            end
        end

        # Replay should not crash even under GC pressure
        GC.gc()  # Force collection before replay
        replayed = MMSB.ReplayEngine.replay_to_epoch(state, UInt32(1000))
        GC.gc()  # Force collection after replay
        replay_page = get(replayed.pages, page.id, nothing)
        @test replay_page !== nothing
        @test length(read_page(replay_page)) == 1024
    end

    @testset "Concurrent page creation and GC" begin
        state = MMSB.MMSBStateTypes.MMSBState()

        Threads.@threads for _ in 1:100
            page = MMSB.PageAllocator.create_cpu_page!(state, 64)
            GC.gc()  # Stress GC during allocation and registration
            data = read_page(page)
            @test length(data) == 64
        end
    end
end
