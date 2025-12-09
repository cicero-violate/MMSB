using Test
using MMSB

@testset "Phase 3: Cognition Integration" begin
    @testset "Layer 8: Reasoning Engine" begin
        # Create mock DAG
        dag = (
            nodes = Dict(
                UInt64(1) => Dict(:type => :input),
                UInt64(2) => Dict(:type => :compute),
                UInt64(3) => Dict(:type => :output)
            ),
            predecessors = Dict(
                UInt64(2) => [UInt64(1)],
                UInt64(3) => [UInt64(2)]
            ),
            successors = Dict(
                UInt64(1) => [UInt64(2)],
                UInt64(2) => [UInt64(3)]
            ),
            executed = Set{UInt64}()
        )
        
        # Initialize reasoning
        state = MMSB.ReasoningEngine.initialize_reasoning(dag)
        @test !isempty(state.rules)
        @test !isempty(state.dependencies)
        
        # Perform reasoning
        result = MMSB.ReasoningEngine.reason_over_dag(dag, state)
        @test result isa MMSB.InferenceResult
        @test !isempty(result.propagated)
    end
    
    @testset "Layer 9: Planning Engine" begin
        # Create initial state
        initial_state = MMSB.State(
            UInt64(1),
            Dict(:memory_usage => 50.0),
            0.0
        )
        
        # Create goal
        goal = MMSB.Goal(
            UInt64(1),
            "Optimize memory",
            (s) -> s.features[:memory_usage] < 30.0,
            1.0,
            UInt64[]
        )
        
        # Create actions
        actions = [
            MMSB.Action(
                UInt64(1),
                "defrag",
                Function[(s) -> true],
                Function[(f, u) -> (merge(f, Dict(:memory_usage => f[:memory_usage] * 0.8)), u + 5.0)],
                10.0
            )
        ]
        
        # Create plan
        plan = MMSB.PlanningEngine.create_plan(goal, initial_state, actions)
        @test plan !== nothing
        @test !isempty(plan.actions)
    end
    
    @testset "Reasoning → Planning Pipeline" begin
        # Create DAG with planning goal
        dag = (
            nodes = Dict(UInt64(1) => Dict(:needs_optimization => true)),
            predecessors = Dict{UInt64, Vector{UInt64}}(),
            successors = Dict{UInt64, Vector{UInt64}}(),
            executed = Set{UInt64}()
        )
        
        # Reason about DAG
        reason_state = MMSB.ReasoningEngine.initialize_reasoning(dag)
        reason_result = MMSB.ReasoningEngine.reason_over_dag(dag, reason_state)
        
        # Create planning state based on reasoning
        initial_state = MMSB.State(UInt64(1), Dict(:optimized => false), 0.0)
        planning_state = MMSB.PlanningState(initial_state)
        
        goal = MMSB.Goal(
            UInt64(1),
            "Execute optimizations",
            (s) -> get(s.features, :optimized, false) == true,
            0.9,
            UInt64[]
        )
        planning_state.goals[goal.id] = goal
        
        actions = [
            MMSB.Action(
                UInt64(1),
                "optimize",
                Function[(s) -> true],
                Function[(f, u) -> (merge(f, Dict(:optimized => true)), u + 10.0)],
                5.0
            )
        ]
        planning_state.available_actions = actions
        
        # Execute planning
        plan = MMSB.PlanningEngine.execute_planning(planning_state, goal.id)
        @test plan !== nothing
        
        # Simulate execution
        result = MMSB.RolloutSimulation.simulate_plan(plan, initial_state)
        @test result.success
    end
end
