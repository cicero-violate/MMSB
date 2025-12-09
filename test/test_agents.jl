using Test
using MMSB: MMSBState, MMSBConfig, observe, AgentAction
using MMSB.AgentTypes
using MMSB.RLAgents
using MMSB.SymbolicAgents
using MMSB.PlanningAgents
using MMSB.HybridAgents

@testset "Layer 11: External Agents" begin
    config = MMSBConfig(tlog_path=tempname() * ".tlog")
    state = MMSBState(config)
    
    @testset "Agent Types" begin
        agent_state = AgentState(42)
        @test agent_state.internal_state == 42
        @test agent_state.step_count == 0
        
        memory = AgentMemory(10)
        push_memory!(memory, :obs1, :act1, 1.0)
        @test length(memory.observations) == 1
    end
    
    @testset "RL Agent" begin
        agent = RLAgent(nothing, lr=0.01, γ=0.95)
        @test agent.learning_rate == 0.01
        
        obs = observe(agent, state)
        @test haskey(obs, :n_pages)
    end
    
    @testset "Symbolic Agent" begin
        agent = SymbolicAgent()
        obs = observe(agent, state)
        @test haskey(obs, :graph_structure)
    end
    
    @testset "Planning Agent" begin
        agent = PlanningAgent(10, 5)
        @test agent.horizon == 10
        
        step = execute_plan_step(agent)
        @test isnothing(step)  # Empty plan
    end
    
    @testset "Hybrid Agent" begin
        agent = HybridAgent(0.5)
        @test agent.mix_ratio == 0.5
        
        obs = observe(agent, state)
        @test haskey(obs, :symbolic_obs)
        @test haskey(obs, :rl_obs)
    end
end
