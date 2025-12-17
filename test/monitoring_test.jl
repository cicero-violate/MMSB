using Test
import MMSB

@testset "Monitoring" begin
    config = MMSB.MMSBStateTypes.MMSBConfig(enable_gpu=false, tlog_path=tempname())
    state = MMSB.API.mmsb_start(config=config)
    page = MMSB.API.create_page(state; size=2)
    MMSB.API.update_page(state, page.id, UInt8[0x0a, 0x0b])
    stats = MMSB.get_stats(state)
    @test stats.total_pages == 1
    @test stats.cpu_pages == 1
    @test stats.total_deltas == 1
    @test stats.delta_apply_count >= 1
    MMSB.reset_stats!(state)
    stats2 = MMSB.get_stats(state)
    @test stats2.delta_apply_count == 0
end
