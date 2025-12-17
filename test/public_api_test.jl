using Test
import MMSB

@testset "Public API" begin
    state = MMSB.API.mmsb_start(enable_gpu=false)
    page = MMSB.API.create_page(state; size=4)
    MMSB.API.update_page(state, page.id, UInt8[0x01, 0x02, 0x03, 0x04])
    @test MMSB.API.query_page(state, page.id) == UInt8[0x01, 0x02, 0x03, 0x04]
    MMSB.API.@mmsb state begin
        @test MMSB.API.ACTIVE_STATE[] === state
        nothing
    end
    @test isnothing(MMSB.API.ACTIVE_STATE[])
    mktemp() do path, io
        close(io)
        MMSB.API.mmsb_stop(state; checkpoint_path=path)
        @test isfile(path)
    end
end
