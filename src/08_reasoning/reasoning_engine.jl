"""
    ReasoningEngine

Main reasoning engine coordinating all Layer 8 components.
"""
module ReasoningEngine

export reason_over_dag, initialize_reasoning, perform_inference

using ..ReasoningTypes
using ..StructuralInference
using ..ConstraintPropagation
using ..DependencyInference
using ..PatternFormation
using ..RuleEvaluation
using ..LogicEngine

"""
    initialize_reasoning(dag) -> ReasoningState

Initialize reasoning state for DAG.
"""
function initialize_reasoning(dag)
    state = ReasoningState()
    
    # Create default rules
    state.rules = RuleEvaluation.create_default_rules()
    
    # Infer initial dependencies
    for node_id in keys(dag.nodes)
        deps = DependencyInference.infer_dependencies(dag, node_id)
        for dep in deps
            state.dependencies[(dep.source, dep.target)] = dep
        end
    end
    
    # Discover patterns
    patterns = PatternFormation.find_patterns(dag, 2)
    for p in patterns
        state.patterns[p.id] = p
    end
    
    state
end

"""
    reason_over_dag(dag, state) -> InferenceResult

Perform complete reasoning pass over DAG.
"""
function reason_over_dag(dag, state::ReasoningState)
    # Phase 1: Structural inference
    structural_constraints = StructuralInference.derive_constraints(dag, state)
    for (node_id, constraints) in structural_constraints
        existing = get!(state.constraints, node_id, Constraint[])
        append!(existing, constraints)
    end
    
    # Phase 2: Constraint propagation
    sources = UInt64[]
    for (node_id, _) in dag.nodes
        if isempty(get(dag.predecessors, node_id, UInt64[]))
            push!(sources, node_id)
        end
    end
    prop_result = ConstraintPropagation.forward_propagate(dag, state, sources)
    
    # Phase 3: Rule evaluation
    all_inferences = Inference[]
    for node_id in keys(dag.nodes)
        inferences = RuleEvaluation.evaluate_rules(dag, state, node_id)
        append!(all_inferences, inferences)
        
        # Cache inferences
        state.inference_cache[node_id] = inferences
    end
    
    # Phase 4: Pattern matching
    pattern_matches = PatternMatch[]
    for (_, pattern) in state.patterns
        for start_node in keys(dag.nodes)
            match = PatternFormation.match_pattern(dag, pattern, start_node)
            if match !== nothing
                push!(pattern_matches, match)
            end
        end
    end
    
    InferenceResult(
        all_inferences,
        state.constraints,
        prop_result.propagated,
        pattern_matches
    )
end

"""
    perform_inference(dag, state, target_node) -> Vector{Inference}

Perform focused inference for specific node.
"""
function perform_inference(dag, state::ReasoningState, target_node::UInt64)
    # Check cache
    if haskey(state.inference_cache, target_node)
        return state.inference_cache[target_node]
    end
    
    # Perform local reasoning
    inferences = Inference[]
    
    # Gather premises from predecessors
    preds = get(dag.predecessors, target_node, UInt64[])
    premises = Constraint[]
    for pred in preds
        pred_constraints = get(state.constraints, pred, Constraint[])
        append!(premises, filter(c -> c.satisfied, pred_constraints))
    end
    
    # Apply deduction
    if !isempty(premises)
        conclusions = LogicEngine.deduce(premises, state.rules)
        if !isempty(conclusions)
            push!(inferences, Inference(
                target_node,
                conclusions,
                0.85,
                preds
            ))
        end
    end
    
    # Cache and return
    state.inference_cache[target_node] = inferences
    inferences
end

end # module
