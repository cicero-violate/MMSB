using Test
import MMSB

@testset "Benchmark Smoke Test" begin
    config = MMSB.MMSBStateTypes.MMSBConfig(enable_gpu=false, tlog_path=tempname())
    state = MMSB.API.mmsb_start(config=config)
    base_page = MMSB.API.create_page(state; size=1024)
    payload = rand(UInt8, 1024)
    alloc_time = @elapsed MMSB.API.create_page(state; size=2048)
    @test alloc_time < 1.0
    delta_time = @elapsed MMSB.API.update_page(state, base_page.id, payload)
    @test delta_time < 1.0
    stats = MMSB.get_stats(state)
    @test stats.total_pages >= 2
    MMSB.API.mmsb_stop(state)
end
