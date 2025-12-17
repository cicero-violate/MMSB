using Test
import MMSB

@testset "Error handling" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    @test_throws MMSB.ErrorTypes.PageNotFoundError MMSB.PageAllocator.delete_page!(state, MMSB.PageTypes.PageID(99))

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
