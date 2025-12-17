using Test
import MMSB

@testset "Checkpoint error mapping" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    mktemp() do path, io
        close(io)
        open(path, "w") do file
            write(file, "corrupt checkpoint data")
        end
        @test_throws MMSB.ErrorTypes.SerializationError MMSB.TLog.load_checkpoint!(state, path)
    end
end
