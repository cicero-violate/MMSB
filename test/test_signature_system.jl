# Test T1.1: Recompute Dependency Signature System

using Test
using MMSB.PageTypes: PageID
using MMSB.PropagationEngine: RecomputeSignature, compute_signature, register_passthrough_recompute!, register_recompute_fn!, recompute_page!
using MMSB.MMSBStateTypes: MMSBState, get_page
using MMSB.API: create_page, update_page

@testset "T1.1: Signature System" begin
    @testset "Signature creation" begin
        state = MMSBState()
        parent = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
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
        parent = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
        register_passthrough_recompute!(state, child.id, parent.id)
        
        # Initial signature
        sig1 = compute_signature(state, child)
        initial_epoch = sig1.parent_epochs[1]
        
        # Update parent
        update_page(state, parent.id, rand(UInt8, 1024))
        
        # Signature should reflect new epoch
        sig2 = compute_signature(state, child)
        new_epoch = sig2.parent_epochs[1]
        @test new_epoch > initial_epoch
    end
    
    @testset "Multiple dependencies" begin
        state = MMSBState()
        p1 = create_page(state; size=1024, location=:cpu)
        p2 = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
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
        child = create_page(state; size=1024, location=:cpu)
        fake_parent_id = PageID(99999)
        
        child.metadata[:recompute_deps] = [fake_parent_id]
        sig = compute_signature(state, child)
        
        @test sig.parent_ids == [fake_parent_id]
        @test sig.parent_epochs == [UInt32(0)]
    end
    
    @testset "Deterministic signature ordering" begin
        state = MMSBState()
        p1 = create_page(state; size=1024, location=:cpu)
        p2 = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
        child.metadata[:recompute_deps] = [p1.id, p2.id]
        
        # Compute multiple times
        sig1 = compute_signature(state, child)
        sig2 = compute_signature(state, child)
        sig3 = compute_signature(state, child)
        
        @test sig1.parent_ids == sig2.parent_ids == sig3.parent_ids
        @test sig1.parent_epochs == sig2.parent_epochs == sig3.parent_epochs
    end
end

@testset "T1.2: Epoch Validation & Caching" begin
    @testset "Skip recompute when epochs unchanged" begin
        state = MMSBState()
        parent = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
        # Set parent content
        data = rand(UInt8, 1024)
        update_page(state, parent.id, data)
        
        register_passthrough_recompute!(state, child.id, parent.id)
        
        # First recompute - should execute
        recompute_page!(state, child.id)
        @test haskey(child.metadata, :last_signature)
        sig1 = child.metadata[:last_signature]
        
        # Second recompute without parent change - should skip
        recompute_page!(state, child.id)
        sig2 = child.metadata[:last_signature]
        @test sig1.parent_epochs == sig2.parent_epochs
    end
    
    @testset "Recompute when parent epoch changes" begin
        state = MMSBState()
        parent = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
        register_passthrough_recompute!(state, child.id, parent.id)
        
        # Initial recompute
        recompute_page!(state, child.id)
        sig_before = child.metadata[:last_signature]
        epoch_before = sig_before.parent_epochs[1]
        
        # Update parent
        update_page(state, parent.id, rand(UInt8, 1024))
        
        # Recompute should execute and update signature
        recompute_page!(state, child.id)
        sig_after = child.metadata[:last_signature]
        epoch_after = sig_after.parent_epochs[1]
        
        @test epoch_after > epoch_before
    end
    
    @testset "Fail fast on dependency set change" begin
        state = MMSBState()
        p1 = create_page(state; size=1024, location=:cpu)
        p2 = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
        # Initial dependency
        register_passthrough_recompute!(state, child.id, p1.id)
        recompute_page!(state, child.id)
        
        # Change dependency set (violates contract)
        child.metadata[:recompute_deps] = [p2.id]
        
        # Should error
        @test_throws ErrorException recompute_page!(state, child.id)
    end
    
    @testset "Signature stored on no-op recompute" begin
        state = MMSBState()
        parent = create_page(state; size=1024, location=:cpu)
        child = create_page(state; size=1024, location=:cpu)
        
        register_passthrough_recompute!(state, child.id, parent.id)
        
        # Recompute with no content change
        recompute_page!(state, child.id)
        @test haskey(child.metadata, :last_signature)
    end
end
