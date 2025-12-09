using Test
import MMSB: MMSBState, MMSBConfig, create_page, emit_event
import MMSB: RLAgent, SymbolicAgent, PlanningAgent, HybridAgent
import MMSB: subscribe_to_events, unsubscribe, PAGE_CREATED, create_checkpoint
import MMSB: observe, World, add_entity!, AgentCoordinator, register_agent!
import MMSB: MMSBContext, ReasoningContext, reason_over_memory

@testset "Phase 4 Integration: Agents + Applications" begin
    config = MMSBConfig(tlog_path=tempname() * ".tlog")
    state = MMSBState(config)
    
    @testset "Agent Interface → External Agents" begin
        # Create checkpoint
        ckpt = create_checkpoint(state, "phase4_test")
        @test !isempty(ckpt)
        
        # Create RL agent
        agent = RLAgent(nothing, lr=0.001)
        obs = observe(agent, state)
        @test isa(obs, NamedTuple)
        
        # Subscribe to events
        event_count = Ref(0)
        sub_id = subscribe_to_events([PAGE_CREATED], (t, d) -> event_count[] += 1)
        
        # Create page and verify event
        page = create_page(state, size=64)
        emit_event(PAGE_CREATED, page.id)
        @test event_count[] == 1
        
        unsubscribe(sub_id)
    end
    
    @testset "External Agents → Applications" begin
        # Planning agent generates plan
        planner = PlanningAgent(10, 3)
        
        # World simulation
        world = World(state, 0.1)
        entity = add_entity!(world, :agent, Dict{Symbol,Any}(:goal => "test"))
        @test entity.entity_type == :agent
        
        # Multi-agent coordination
        coordinator = AgentCoordinator(state, :sequential)
        register_agent!(coordinator, planner)
        @test length(coordinator.agents) == 1
    end
    
    @testset "Full Pipeline: Reasoning → Planning → Agent → Application" begin
        # Symbolic agent uses reasoning
        symbolic = SymbolicAgent()
        
        # Hybrid agent combines symbolic + RL
        hybrid = HybridAgent(0.5)
        
        # LLM context uses MMSB state
        llm_ctx = MMSBContext(state, 2048)
        @test llm_ctx.max_tokens == 2048
        
        # Memory-driven reasoning
        reasoning_ctx = ReasoningContext(state, "integration test", (0, 1000))
        result = reason_over_memory(reasoning_ctx)
        @test haskey(result, :query)
    end
end
