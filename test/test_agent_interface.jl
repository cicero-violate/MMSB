using Test
import MMSB: MMSBState, MMSBConfig, create_page
import MMSB: create_checkpoint, restore_checkpoint, list_checkpoints
import MMSB: subscribe_to_events, unsubscribe, emit_event
import MMSB: PAGE_CREATED, PAGE_MODIFIED, DELTA_APPLIED
import MMSB: AbstractAgent, observe, plan, AgentAction

@testset "Layer 10: Agent Interface" begin
    @testset "Checkpoint API" begin
        config = MMSBConfig(tlog_path=tempname() * ".tlog")
        state = MMSBState(config)
        
        # Test checkpoint creation
        ckpt_path = create_checkpoint(state, "test")
        @test !isempty(ckpt_path)
        @test endswith(ckpt_path, ".ckpt")
        
        # Test checkpoint exists
        @test isfile(ckpt_path)
        
        # Cleanup
        rm(ckpt_path, force=true)
    end
    
    @testset "Event Subscription" begin
        events_received = Ref(0)
        
        callback = (event_type, data) -> begin
            events_received[] += 1
        end
        
        sub_id = subscribe_to_events([PAGE_CREATED, PAGE_MODIFIED], callback)
        @test sub_id > 0
        
        emit_event(PAGE_CREATED, nothing)
        @test events_received[] == 1
        
        emit_event(DELTA_APPLIED, nothing)
        @test events_received[] == 1  # Should not trigger
        
        unsubscribe(sub_id)
        emit_event(PAGE_CREATED, nothing)
        @test events_received[] == 1  # Should not trigger after unsubscribe
    end
    
    @testset "Agent Protocol" begin
        struct TestAgent <: AbstractAgent end
        
        observe(agent::TestAgent, state::MMSBState) = state.pages
        plan(agent::TestAgent, obs::Any) = AgentAction[]
        
        config = MMSBConfig(tlog_path=tempname() * ".tlog")
        state = MMSBState(config)
        agent = TestAgent()
        
        obs = observe(agent, state)
        @test isa(obs, Dict)
        
        actions = plan(agent, obs)
        @test isa(actions, Vector{AgentAction})
    end
end
