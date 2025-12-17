# Test T1.1: Recompute Dependency Signature System

using Test
using MMSB
using MMSB.API: create_page, update_page
using MMSB.PageTypes: PageID
using MMSB.PropagationEngine: RecomputeSignature, compute_signature, register_passthrough_recompute!, register_recompute_fn!
using MMSB.MMSBStateTypes: get_page

@testset "T1.1: Signature System" begin
    @testset "Signature creation" begin
        state = MMSBState()
        parent = create_page(state; size=1024)
        child = create_page(state; size=1024)
        
        # Register passthrough dependency
        register_passthrough_recompute!(state, child.id, parent.id)
        
        # Verify dependency stored
        @test haskey(child.metadata, :recompute_deps)
        @test child.metadata[:recompute_deps] == [parent.id]
        
        # Compute signature
        sig = compute_signature(state, child)
        @test sig isa RecomputeSignature
        @test sig.parent_ids == [parent.id]
        @test length(sig.parent_epochs) == 1
    end
    
    @testset "Signature tracks epochs" begin
        state = MMSBState()
        parent = create_page(state; size=1024)
        child = create_page(state; size=1024)
        
        register_passthrough_recompute!(state, child.id, parent.id)
        
        # Initial signature
        sig1 = compute_signature(state, child)
        initial_epoch = sig1.parent_epochs[1]
        
        # Update parent
        update_page(state, parent.id, rand(UInt8, 1024))
        
        # Signature should reflect new epoch
        sig2 = compute_signature(state, child)
        @test sig2.parent_epochs[1] > initial_epoch
    end
    
    @testset "Multiple dependencies" begin
        state = MMSBState()
        p1 = create_page(state; size=1024)
        p2 = create_page(state; size=1024)
        child = create_page(state; size=1024)
        
        # Register custom recompute with multiple deps
        register_recompute_fn!(state, child.id, function(st, pg)
            d1 = get_page(st, p1.id)
            d2 = get_page(st, p2.id)
            return zeros(UInt8, pg.size)
        end)
        child.metadata[:recompute_deps] = [p1.id, p2.id]
        
        sig = compute_signature(state, child)
        @test length(sig.parent_ids) == 2
        @test length(sig.parent_epochs) == 2
        @test sig.parent_ids == [p1.id, p2.id]
    end
    
    @testset "Missing parent returns zero epoch" begin
        state = MMSBState()
        child = create_page(state; size=1024)
        fake_parent_id = PageID(99999)
        
        child.metadata[:recompute_deps] = [fake_parent_id]
        sig = compute_signature(state, child)
        
        @test sig.parent_ids == [fake_parent_id]
        @test sig.parent_epochs == [UInt32(0)]
    end
    
    @testset "Deterministic signature ordering" begin
        state = MMSBState()
        p1 = create_page(state; size=1024)
        p2 = create_page(state; size=1024)
        child = create_page(state; size=1024)
        
        child.metadata[:recompute_deps] = [p1.id, p2.id]
        
        # Compute multiple times
        sig1 = compute_signature(state, child)
        sig2 = compute_signature(state, child)
        sig3 = compute_signature(state, child)
        
        @test sig1.parent_ids == sig2.parent_ids == sig3.parent_ids
        @test sig1.parent_epochs == sig2.parent_epochs == sig3.parent_epochs
    end
end
