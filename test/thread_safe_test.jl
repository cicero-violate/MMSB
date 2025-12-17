using Test
using Base.Threads
import MMSB

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
