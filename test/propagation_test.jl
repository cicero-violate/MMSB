using Test
import MMSB

@testset "Propagation recompute flow" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    parent = MMSB.PageAllocator.create_cpu_page!(state, 8)
    child = MMSB.PageAllocator.create_cpu_page!(state, 8)
    MMSB.GraphTypes.add_dependency!(state.graph, parent.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)
    MMSB.PropagationEngine.register_passthrough_recompute!(state, child.id, parent.id)

    mask = falses(8); mask[1:3] .= true
    data = UInt8.(11:18)
    delta = MMSB.DeltaRouter.create_delta(state, parent.id, collect(mask), data; source=:test)
    MMSB.DeltaRouter.route_delta!(state, delta)

    @test MMSB.PageTypes.read_page(child)[1:3] == data[1:3]
    @test !get(child.metadata, :stale, false)
end

@testset "Propagation optimization behaviors" begin
    state = MMSB.MMSBStateTypes.MMSBState()
    parent_a = MMSB.PageAllocator.create_cpu_page!(state, 4)
    parent_b = MMSB.PageAllocator.create_cpu_page!(state, 4)
    child = MMSB.PageAllocator.create_cpu_page!(state, 4)
    MMSB.GraphTypes.add_dependency!(state.graph, parent_a.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)
    MMSB.GraphTypes.add_dependency!(state.graph, parent_b.id, child.id, MMSB.GraphTypes.DATA_DEPENDENCY)

    hits = Ref(0)
    MMSB.PropagationEngine.register_recompute_fn!(state, child.id,
        function (st, _)
            hits[] += 1
            page_a = MMSB.MMSBStateTypes.get_page(st, parent_a.id)
            page_b = MMSB.MMSBStateTypes.get_page(st, parent_b.id)
            return Vector{UInt8}(MMSB.PageTypes.read_page(page_a) .+ MMSB.PageTypes.read_page(page_b))
        end)
    # Declare dependencies for signature system
    child.metadata[:recompute_deps] = [parent_a.id, parent_b.id]

    mask = trues(4)
    payload_a = fill(UInt8(0x01), 4)
    payload_b = fill(UInt8(0x02), 4)
    delta_a = MMSB.DeltaRouter.create_delta(state, parent_a.id, mask, payload_a; source=:batch)
    delta_b = MMSB.DeltaRouter.create_delta(state, parent_b.id, mask, payload_b; source=:batch)
    MMSB.DeltaRouter.batch_route_deltas!(state, [delta_a, delta_b])

    @test hits[] == 1
    @test MMSB.PageTypes.read_page(child) == fill(UInt8(0x03), 4)

    MMSB.PropagationEngine.queue_recomputation!(state, child.id)
    MMSB.PropagationEngine.queue_recomputation!(state, child.id)
    MMSB.PropagationEngine.execute_propagation!(state)
    # T1.2: Cache skips second recompute since epochs unchanged
    @test hits[] == 1
end
