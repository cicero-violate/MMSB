"""
    RuleEvaluation

Evaluate logical rules on DAG nodes.
"""
module RuleEvaluation

export evaluate_rules, apply_rule, create_default_rules

using ..ReasoningTypes

"""
    evaluate_rules(dag, state, node_id) -> Vector{Inference}

Evaluate all applicable rules for a node.
"""
function evaluate_rules(dag, state::ReasoningState, node_id::UInt64)
    inferences = Inference[]
    
    node_constraints = get(state.constraints, node_id, Constraint[])
    
    for rule in state.rules
        # Check if rule applies
        if rule.condition(dag, node_id, node_constraints)
            # Apply rule action
            new_constraints = rule.action(dag, node_id, node_constraints)
            
            if !isempty(new_constraints)
                push!(inferences, Inference(
                    node_id,
                    new_constraints,
                    0.9,  # High confidence for rule-based inference
                    [rule.id]
                ))
            end
        end
    end
    
    inferences
end

"""
    apply_rule(rule, dag, node_id, constraints) -> Vector{Constraint}

Apply single rule to node.
"""
function apply_rule(rule::Rule, dag, node_id::UInt64, constraints::Vector{Constraint})
    if rule.condition(dag, node_id, constraints)
        return rule.action(dag, node_id, constraints)
    end
    Constraint[]
end

"""
    create_default_rules() -> Vector{Rule}

Create default reasoning rules.
"""
function create_default_rules()
    rules = Rule[]
    
    # Rule 1: Type propagation
    push!(rules, Rule(
        UInt64(1),
        INFERENCE_RULE,
        (dag, nid, cs) -> any(c -> c.type == TYPE_CONSTRAINT && c.satisfied, cs),
        (dag, nid, cs) -> begin
            type_c = findfirst(c -> c.type == TYPE_CONSTRAINT && c.satisfied, cs)
            if type_c !== nothing
                [Constraint(
                    TYPE_CONSTRAINT,
                    cs[type_c].predicate,
                    true,
                    Dict(:inferred => true)
                )]
            else
                Constraint[]
            end
        end,
        10
    ))
    
    # Rule 2: Ordering enforcement
    push!(rules, Rule(
        UInt64(2),
        SAFETY_RULE,
        (dag, nid, cs) -> length(get(dag.predecessors, nid, UInt64[])) > 0,
        (dag, nid, cs) -> begin
            preds = get(dag.predecessors, nid, UInt64[])
            [Constraint(
                ORDERING_CONSTRAINT,
                (x) -> all(p in get(dag, :executed, Set{UInt64}()), preds),
                false,
                Dict(:required_predecessors => preds)
            )]
        end,
        20
    ))
    
    # Rule 3: Resource availability
    push!(rules, Rule(
        UInt64(3),
        OPTIMIZATION_RULE,
        (dag, nid, cs) -> true,
        (dag, nid, cs) -> [Constraint(
            RESOURCE_CONSTRAINT,
            (x) -> true,  # Placeholder
            true,
            Dict(:resource_type => :memory)
        )],
        5
    ))
    
    rules
end

end # module
