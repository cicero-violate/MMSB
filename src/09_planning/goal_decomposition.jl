"""
    GoalDecomposition

Decompose complex goals into subgoals.
"""
module GoalDecomposition

export decompose_goal, create_subgoal_hierarchy, order_subgoals

using ..PlanningTypes

"""
    decompose_goal(goal, state) -> Vector{Goal}

Break goal into achievable subgoals.
"""
function decompose_goal(goal::Goal, state::State)
    subgoals = Goal[]
    
    # Analyze goal predicate to identify components
    if haskey(goal.predicate, :components)
        components = goal.predicate[:components]
        for (i, comp) in enumerate(components)
            push!(subgoals, Goal(
                UInt64(goal.id * 100 + i),
                "Subgoal: $comp",
                (s) -> comp(s),
                goal.priority * (1.0 - i * 0.1),
                UInt64[]
            ))
        end
    else
        # Default: decompose by feature requirements
        push!(subgoals, Goal(
            goal.id * 100 + 1,
            "Prepare state",
            (s) -> true,
            goal.priority * 0.5,
            UInt64[]
        ))
        
        push!(subgoals, Goal(
            goal.id * 100 + 2,
            "Execute action",
            goal.predicate,
            goal.priority,
            UInt64[]
        ))
    end
    
    subgoals
end

"""
    create_subgoal_hierarchy(goals) -> Dict

Build dependency hierarchy among goals.
"""
function create_subgoal_hierarchy(goals::Vector{Goal})
    hierarchy = Dict{UInt64, Vector{UInt64}}()
    
    for goal in goals
        if !isempty(goal.subgoals)
            hierarchy[goal.id] = goal.subgoals
        end
    end
    
    hierarchy
end

"""
    order_subgoals(subgoals, state) -> Vector{Goal}

Order subgoals by priority and dependencies.
"""
function order_subgoals(subgoals::Vector{Goal}, state::State)
    # Sort by priority and achievability
    scored = [(g, score_subgoal(g, state)) for g in subgoals]
    sort!(scored, by=x->x[2], rev=true)
    [g for (g, _) in scored]
end

function score_subgoal(goal::Goal, state::State)
    priority_score = goal.priority * 100.0
    achievability = estimate_achievability(goal, state)
    priority_score * achievability
end

function estimate_achievability(goal::Goal, state::State)
    # Simple: check if already close to goal
    if goal.predicate(state)
        return 1.0
    end
    0.5  # Neutral
end

end # module
