using Test
using .MMSB
using .MMSB.PageTypes: read_page

function _propagation_fuzz_state()
    config = MMSB.MMSBStateTypes.MMSBConfig(tlog_path=tempname())
    MMSB.MMSBStateTypes.MMSBState(config)
end

@testset "Fuzz: propagation under random workloads" begin
    state = _propagation_fuzz_state()

    # Create a small parent/child graph
    parent = MMSB.PageAllocator.create_cpu_page!(state, 128)
    child1 = MMSB.PageAllocator.create_cpu_page!(state, 128)
    child2 = MMSB.PageAllocator.create_cpu_page!(state, 128)

    MMSB.GraphTypes.add_dependency!(state.graph, parent.id, child1.id, MMSB.GraphTypes.DATA_DEPENDENCY)
    MMSB.GraphTypes.add_dependency!(state.graph, parent.id, child2.id, MMSB.GraphTypes.DATA_DEPENDENCY)

    # Register trivial passthrough recompute fns for fuzz propagation
    MMSB.PropagationEngine.register_passthrough_recompute!(state, child1.id, parent.id)
    MMSB.PropagationEngine.register_passthrough_recompute!(state, child2.id, parent.id)

    # Apply random deltas to the parent and trigger propagation
    for i in 1:200
        mask = falses(128)
        mask[rand(1:128, rand(1:32))] .= true
        payload = rand(UInt8, 128)
        delta = MMSB.DeltaRouter.create_delta(state, parent.id, collect(mask), payload; source=:prop_fuzz)
        MMSB.DeltaRouter.route_delta!(state, delta)
        
        # Propagate changes
        MMSB.PropagationEngine.propagate_change!(state, parent.id)

        # Stress GC periodically
        if i % 25 == 0
            GC.gc()
        end
    end

    # Validate children remain readable and structurally valid
    @test length(read_page(child1)) == 128
    @test length(read_page(child2)) == 128
end
