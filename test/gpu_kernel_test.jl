using Test
import CUDA
import MMSB

const CUDA_AVAILABLE = try
    CUDA.functional()
catch
    false
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
        host = MMSB.PageTypes.read_page(page)
        @test host[1:5] == data[1:5]
    else
        @info "CUDA not functional; skipping GPU kernel tests"
        @test true
    end
end
