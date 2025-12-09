"""
    PlanningEngine

Main planning engine coordinating all Layer 9 components.
"""
module PlanningEngine

export create_plan, execute_planning, replan

using ..PlanningTypes
using ..SearchAlgorithms
using ..GoalDecomposition
using ..StrategyGeneration
using ..RolloutSimulation
using ..DecisionGraphs
using ..RLPlanning
using ..OptimizationPlanning

"""
    create_plan(goal, state, actions) -> Plan

Create plan to achieve goal from state.
"""
function create_plan(goal::Goal, state::State, actions::Vector{Action})
    # Generate candidate strategies
    strategies = StrategyGeneration.generate_strategies(goal, state)
    
    # Select best strategy
    strategy = StrategyGeneration.select_strategy(strategies, goal, state)
    
    # Generate plan
    plan = strategy.plan_generator(goal, state, actions)
    
    # Optimize if possible
    if plan !== nothing
        optimized = OptimizationPlanning.optimize_plan(plan, 
            (p) -> p[1] - sum(p[2:end]))
        return optimized
    end
    
    plan
end

"""
    execute_planning(planning_state, goal_id) -> Plan

Full planning cycle for specific goal.
"""
function execute_planning(planning_state::PlanningState, goal_id::UInt64)
    goal = planning_state.goals[goal_id]
    
    # Decompose if complex
    if length(goal.subgoals) == 0
        subgoals = GoalDecomposition.decompose_goal(goal, planning_state.current_state)
        goal = Goal(
            goal.id,
            goal.description,
            goal.predicate,
            goal.priority,
            [sg.id for sg in subgoals]
        )
        
        # Add subgoals
        for sg in subgoals
            planning_state.goals[sg.id] = sg
        end
    end
    
    # Create plan
    plan = create_plan(goal, planning_state.current_state, planning_state.available_actions)
    
    if plan !== nothing
        # Simulate before execution
        result = RolloutSimulation.simulate_plan(plan, planning_state.current_state)
        
        if result.success
            planning_state.plans[plan.id] = plan
            return plan
        end
    end
    
    # Fallback: MCTS
    SearchAlgorithms.mcts_search(
        planning_state.current_state,
        goal,
        planning_state.available_actions,
        1000
    )
end

"""
    replan(planning_state, plan_id, feedback) -> Plan

Adapt existing plan based on execution feedback.
"""
function replan(planning_state::PlanningState, plan_id::UInt64, feedback::Dict{Symbol, Any})
    old_plan = planning_state.plans[plan_id]
    goal = planning_state.goals[old_plan.goal_id]
    
    # Get current state from feedback
    current = get(feedback, :current_state, planning_state.current_state)
    
    # If plan failed, try different strategy
    if get(feedback, :success, true) == false
        strategies = StrategyGeneration.generate_strategies(goal, current)
        
        # Try strategies in order until one works
        for strategy in strategies
            plan = strategy.plan_generator(goal, current, planning_state.available_actions)
            if plan !== nothing
                result = RolloutSimulation.simulate_plan(plan, current)
                if result.success
                    planning_state.plans[plan.id] = plan
                    return plan
                end
            end
        end
    end
    
    # Repair plan by re-searching from current state
    remaining_actions = old_plan.actions[get(feedback, :step, 1):end]
    
    new_plan = SearchAlgorithms.astar_search(
        current,
        goal,
        planning_state.available_actions,
        500
    )
    
    new_plan
end

end # module
