"""
    ConstraintPropagation

Forward constraint propagation through DAG edges.
"""
module ConstraintPropagation

export propagate_constraints, forward_propagate, backward_propagate

using ..ReasoningTypes

"""
    propagate_constraints(dag, state, node_id) -> Set{UInt64}

Propagate constraints from node to successors.
"""
function propagate_constraints(dag, state::ReasoningState, node_id::UInt64)
    propagated = Set{UInt64}()
    
    # Get node constraints
    node_constraints = get(state.constraints, node_id, Constraint[])
    satisfied = filter(c -> c.satisfied, node_constraints)
    
    # Get successors
    succs = get(dag.successors, node_id, UInt64[])
    
    for succ in succs
        # Check edge dependency
        dep = get(state.dependencies, (node_id, succ), nothing)
        
        for c in satisfied
            # Propagate based on dependency type
            if dep !== nothing
                strength = dep.strength
                if dep.type == DATA_DEPENDENCY
                    # Strong propagation
                    propagated_c = Constraint(
                        c.type,
                        c.predicate,
                        strength > 0.8,
                        merge(c.metadata, Dict(:propagated_from => node_id))
                    )
                    
                    # Add to successor
                    succ_constraints = get!(state.constraints, succ, Constraint[])
                    push!(succ_constraints, propagated_c)
                    push!(propagated, succ)
                end
            end
        end
    end
    
    propagated
end

"""
    forward_propagate(dag, state, start_nodes) -> InferenceResult

Propagate constraints forward through DAG.
"""
function forward_propagate(dag, state::ReasoningState, start_nodes::Vector{UInt64})
    all_propagated = Set{UInt64}()
    worklist = copy(start_nodes)
    visited = Set{UInt64}()
    
    while !isempty(worklist)
        node = pop!(worklist)
        push!(visited, node)
        
        propagated = propagate_constraints(dag, state, node)
        union!(all_propagated, propagated)
        
        # Add unvisited successors to worklist
        for p in propagated
            if !(p in visited)
                push!(worklist, p)
            end
        end
    end
    
    InferenceResult(
        Inference[],
        state.constraints,
        all_propagated,
        PatternMatch[]
    )
end

"""
    backward_propagate(dag, state, node_id) -> Set{UInt64}

Propagate constraints backward to predecessors.
"""
function backward_propagate(dag, state::ReasoningState, node_id::UInt64)
    propagated = Set{UInt64}()
    
    node_constraints = get(state.constraints, node_id, Constraint[])
    preds = get(dag.predecessors, node_id, UInt64[])
    
    for pred in preds
        dep = get(state.dependencies, (pred, node_id), nothing)
        
        if dep !== nothing && dep.type == CONTROL_DEPENDENCY
            # Propagate control constraints backward
            for c in node_constraints
                if c.type == ORDERING_CONSTRAINT
                    pred_constraints = get!(state.constraints, pred, Constraint[])
                    push!(pred_constraints, Constraint(
                        c.type,
                        c.predicate,
                        false,
                        Dict(:back_propagated_from => node_id)
                    ))
                    push!(propagated, pred)
                end
            end
        end
    end
    
    propagated
end

end # module
