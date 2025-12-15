# Basic example tests demonstrating core functionality in Julia

module ExamplesBasicTests

using Test

include(joinpath(@__DIR__, "..", "src", "MMSB.jl"))
using .MMSB

const API = MMSB.API
const Semiring = MMSB.Semiring
const TLog = MMSB.TLog

@testset "Example: Simple Page Creation" begin
    state = API.mmsb_start(enable_gpu=false)
    
    # Create a page
    page = API.create_page(state; size=4096)
    
    @test page.id.id > 0
    @test !isnothing(page)
    
    API.mmsb_stop(state)
end

@testset "Example: Page Update and Read" begin
    state = API.mmsb_start(enable_gpu=false)
    
    page = API.create_page(state; size=256)
    
    # Write data
    data = rand(UInt8, 256)
    API.update_page(state, page.id, data)
    
    # Read back
    retrieved = API.read_page(state, page.id)
    @test retrieved == data
    
    API.mmsb_stop(state)
end

@testset "Example: Tropical Semiring" begin
    tropical = Semiring.TropicalSemiring(Float64)
    
    a = 10.0
    b = 5.0
    
    # Tropical addition (min)
    sum_val = Semiring.add(tropical, a, b)
    @test sum_val == 5.0
    
    # Tropical multiplication (sum)
    prod_val = Semiring.multiply(tropical, a, b)
    @test prod_val == 15.0
end

@testset "Example: Checkpoint and Replay" begin
    # Create initial state
    state = API.mmsb_start(enable_gpu=false)
    
    page = API.create_page(state; size=512)
    data = rand(UInt8, 512)
    API.update_page(state, page.id, data)
    
    # Checkpoint
    path = tempname()
    TLog.checkpoint_log!(state, path)
    
    @test isfile(path)
    
    # Create new state and replay
    state2 = API.mmsb_start(enable_gpu=false)
    TLog.replay_log!(state2, path)
    
    @test length(state2.pages) > 0
    
    # Cleanup
    rm(path)
    API.mmsb_stop(state)
    API.mmsb_stop(state2)
end

@testset "Example: Page Dependencies" begin
    state = API.mmsb_start(enable_gpu=false)
    
    # Create two pages
    page1 = API.create_page(state; size=256)
    page2 = API.create_page(state; size=256)
    
    # Add dependency
    MMSB.GraphTypes.add_dependency!(
        state.graph,
        page1.id,
        page2.id,
        MMSB.GraphTypes.DATA_DEPENDENCY
    )
    
    # Register propagation
    MMSB.PropagationEngine.register_passthrough_recompute!(
        state, page2.id, page1.id
    )
    
    # Update source
    data = rand(UInt8, 256)
    API.update_page(state, page1.id, data)
    
    @test true  # Propagation succeeds
    
    API.mmsb_stop(state)
end

@testset "Example: Multiple Updates" begin
    state = API.mmsb_start(enable_gpu=false)
    
    page = API.create_page(state; size=1024)
    
    # Multiple sequential updates
    for i in 1:10
        data = fill(UInt8(i), 1024)
        API.update_page(state, page.id, data)
    end
    
    # Latest update should be applied
    result = API.read_page(state, page.id)
    @test all(x -> x == 10, result)
    
    API.mmsb_stop(state)
end

end # module
