# Test T1.3: Deterministic Replay Oracle

using Test
using MMSB.PageTypes: PageID, read_page
using MMSB.MMSBStateTypes: MMSBState, get_page
using MMSB.PropagationEngine: register_passthrough_recompute!
using MMSB.API: create_page, update_page

"""
    canonical_snapshot(state) -> Dict

Create deterministic snapshot of state for replay comparison.
"""
function canonical_snapshot(state::MMSBState)
    # Sort page IDs for deterministic iteration
    page_ids = sort(collect(keys(state.pages)))
    
    return Dict(
        "pages" => Dict(id => read_page(get_page(state, id)) for id in page_ids),
        "epochs" => Dict(id => get(get_page(state, id).metadata, :epoch_dirty, UInt32(0)) for id in page_ids),
        "graph" => serialize_graph(state.graph),
        "next_page_id" => state.next_page_id[],
        "next_delta_id" => state.next_delta_id[],  # Atomic read
    )
end

"""
    serialize_graph(graph) -> Dict

Deterministic graph serialization (sorted keys).
"""
function serialize_graph(graph)
    edges = Dict{PageID, Vector{Tuple{PageID, Any}}}()
    # Graph uses 'deps' not 'edges'
    for parent_id in sort(collect(keys(graph.deps)))
        children = graph.deps[parent_id]
        edges[parent_id] = sort(children; by = x -> x[1])
    end
    return Dict("edges" => edges)
end

@testset "T1.3: Deterministic Replay Oracle" begin
    @testset "Fresh state determinism" begin
        # Execute same operations on two fresh states
        ops = [
            (state) -> create_page(state; size=1024, location=:cpu),
            (state) -> create_page(state; size=2048, location=:cpu),
            (state) -> update_page(state, PageID(1), rand(UInt8, 1024)),
        ]
        
        state1 = MMSBState()
        state2 = MMSBState()
        
        for op in ops
            op(state1)
            op(state2)
        end
        
        snap1 = canonical_snapshot(state1)
        snap2 = canonical_snapshot(state2)
        
        @test snap1["next_page_id"] == snap2["next_page_id"]
        @test snap1["next_delta_id"] == snap2["next_delta_id"]
    end
    
    @testset "Epoch tracking determinism" begin
        state1 = MMSBState()
        state2 = MMSBState()
        
        # Same operations
        for _ in 1:2
            for st in [state1, state2]
                p = create_page(st; size=1024, location=:cpu)
                update_page(st, p.id, rand(UInt8, 1024))
            end
        end
        
        snap1 = canonical_snapshot(state1)
        snap2 = canonical_snapshot(state2)
        
        @test snap1["epochs"] == snap2["epochs"]
    end
    
    @testset "Propagation determinism" begin
        state1 = MMSBState()
        state2 = MMSBState()
        
        for st in [state1, state2]
            parent = create_page(st; size=1024, location=:cpu)
            child = create_page(st; size=1024, location=:cpu)
            register_passthrough_recompute!(st, child.id, parent.id)
            
            data = [0x01, 0x02, 0x03, zeros(UInt8, 1021)...]
            update_page(st, parent.id, collect(data))
            MMSB.PropagationEngine.recompute_page!(st, child.id)
        end
        
        snap1 = canonical_snapshot(state1)
        snap2 = canonical_snapshot(state2)
        
        @test snap1["pages"] == snap2["pages"]
        @test snap1["epochs"] == snap2["epochs"]
    end
    
    @testset "Signature determinism across runs" begin
        runs = []
        
        for _ in 1:3
            state = MMSBState()
            parent = create_page(state; size=1024, location=:cpu)
            child = create_page(state; size=1024, location=:cpu)
            
            register_passthrough_recompute!(state, child.id, parent.id)
            update_page(state, parent.id, ones(UInt8, 1024))
            MMSB.PropagationEngine.recompute_page!(state, child.id)
            
            push!(runs, canonical_snapshot(state))
        end
        
        # All runs should be identical
        @test runs[1] == runs[2] == runs[3]
    end
    
    @testset "No dict iteration ordering issues" begin
        state = MMSBState()
        pages = [create_page(state; size=1024, location=:cpu) for _ in 1:10]
        
        # Multiple snapshots should be identical
        snap1 = canonical_snapshot(state)
        snap2 = canonical_snapshot(state)
        snap3 = canonical_snapshot(state)
        
        @test snap1 == snap2 == snap3
    end
    
    @testset "Atomic delta ID determinism" begin
        state1 = MMSBState()
        state2 = MMSBState()
        
        # Same sequence of delta allocations
        ids1 = [MMSB.MMSBStateTypes.allocate_delta_id!(state1) for _ in 1:100]
        ids2 = [MMSB.MMSBStateTypes.allocate_delta_id!(state2) for _ in 1:100]
        
        @test ids1 == ids2
        @test all(ids1[i] < ids1[i+1] for i in 1:99)  # Monotonic
    end
end
