# Structure Group: src/08_reasoning

## File: MMSB/src/08_reasoning/ReasoningTypes.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (10)
- Element types: Function (1), Module (1), Struct (8)
- Total elements: 10

### Elements

- [Julia | Module] `ReasoningTypes` (line 6, pub)
- [Julia | Struct] `Constraint` (line 19, pub)
  - Signature: `struct Constraint`
- [Julia | Struct] `Dependency` (line 33, pub)
  - Signature: `struct Dependency`
- [Julia | Struct] `Pattern` (line 41, pub)
  - Signature: `struct Pattern`
- [Julia | Struct] `PatternMatch` (line 49, pub)
  - Signature: `struct PatternMatch`
- [Julia | Struct] `Rule` (line 62, pub)
  - Signature: `struct Rule`
- [Julia | Struct] `Inference` (line 70, pub)
  - Signature: `struct Inference`
- [Julia | Struct] `InferenceResult` (line 77, pub)
  - Signature: `struct InferenceResult`
- [Julia | Struct] `ReasoningState` (line 84, pub)
  - Signature: `mutable struct ReasoningState`
- [Julia | Function] `ReasoningState` (line 92, pub)
  - Signature: `ReasoningState()`
  - Calls: ReasoningState

## File: MMSB/src/08_reasoning/constraint_propagation.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `ConstraintPropagation` (line 6, pub)
- [Julia | Function] `propagate_constraints` (line 17, pub)
  - Signature: `propagate_constraints(dag, state::ReasoningState, node_id::UInt64)`
  - Calls: Constraint, Dict, filter, get, get!, merge, push!
- [Julia | Function] `forward_propagate` (line 61, pub)
  - Signature: `forward_propagate(dag, state::ReasoningState, start_nodes::Vector{UInt64})`
  - Calls: InferenceResult, copy, isempty, pop!, propagate_constraints, push!, union!
- [Julia | Function] `backward_propagate` (line 94, pub)
  - Signature: `backward_propagate(dag, state::ReasoningState, node_id::UInt64)`
  - Calls: Constraint, Dict, get, get!, push!

## File: MMSB/src/08_reasoning/dependency_inference.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (7)
- Element types: Function (6), Module (1)
- Total elements: 7

### Elements

- [Julia | Module] `DependencyInference` (line 6, pub)
- [Julia | Function] `infer_dependencies` (line 17, pub)
  - Signature: `infer_dependencies(dag, node_id::UInt64)`
  - Calls: Dependency, analyze_edge_type, compute_dependency_strength, get, push!
- [Julia | Function] `analyze_edge_type` (line 44, pub)
  - Signature: `analyze_edge_type(dag, source::UInt64, target::UInt64)`
  - Calls: haskey
- [Julia | Function] `compute_dependency_strength` (line 70, pub)
  - Signature: `compute_dependency_strength(dag, source::UInt64, target::UInt64)`
  - Calls: count_paths, get, min
- [Julia | Function] `count_paths` (line 88, pub)
  - Signature: `count_paths(dag, source::UInt64, target::UInt64, max_depth::Int)`
  - Calls: count_paths, get
- [Julia | Function] `analyze_flow` (line 112, pub)
  - Signature: `analyze_flow(dag, state::ReasoningState)`
  - Calls: compute_critical_path, get, isempty, push!, sinks, sources
- [Julia | Function] `compute_critical_path` (line 139, pub)
  - Signature: `compute_critical_path(dag)`

## File: MMSB/src/08_reasoning/logic_engine.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (5)
- Element types: Function (4), Module (1)
- Total elements: 5

### Elements

- [Julia | Module] `LogicEngine` (line 6, pub)
- [Julia | Function] `deduce` (line 17, pub)
  - Signature: `deduce(premises::Vector{Constraint}, rules::Vector{Rule})`
  - Calls: Constraint, Dict, all, push!, sort
- [Julia | Function] `abduce` (line 43, pub)
  - Signature: `abduce(observation::Constraint, rules::Vector{Rule})`
  - Calls: Constraint, Dict, push!
- [Julia | Function] `induce` (line 68, pub)
  - Signature: `induce(examples::Vector{Tuple{Vector{Constraint}, Constraint}})`
  - Calls: Rule, UInt64, all, get!, length, push!, unique
- [Julia | Function] `unify_constraints` (line 104, pub)
  - Signature: `unify_constraints(c1::Constraint, c2::Constraint)`
  - Calls: Constraint, c1.predicate, c2.predicate, merge

## File: MMSB/src/08_reasoning/pattern_formation.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (7)
- Element types: Function (6), Module (1)
- Total elements: 7

### Elements

- [Julia | Module] `PatternFormation` (line 6, pub)
- [Julia | Function] `find_patterns` (line 17, pub)
  - Signature: `find_patterns(dag, min_frequency::Int`
- [Julia | Function] `extract_subgraphs` (line 58, pub)
  - Signature: `extract_subgraphs(dag, size::Int)`
  - Calls: grow_subgraph, keys, length, push!
- [Julia | Function] `grow_subgraph` (line 71, pub)
  - Signature: `grow_subgraph(dag, start::UInt64, size::Int)`
  - Calls: get, isempty, length, popfirst!, push!
- [Julia | Function] `extract_subgraph_signature` (line 97, pub)
  - Signature: `extract_subgraph_signature(dag, nodes::Vector{UInt64})`
  - Calls: Set, UInt8, get, length, min, push!, sort
- [Julia | Function] `extract_edges` (line 127, pub)
  - Signature: `extract_edges(dag, nodes::Vector{UInt64})`
  - Calls: Set, get, push!
- [Julia | Function] `match_pattern` (line 148, pub)
  - Signature: `match_pattern(dag, pattern::Pattern, start_node::UInt64)`
  - Calls: PatternMatch, extract_subgraph_signature, grow_subgraph, length

## File: MMSB/src/08_reasoning/reasoning_engine.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `ReasoningEngine` (line 6, pub)
- [Julia | Function] `initialize_reasoning` (line 23, pub)
  - Signature: `initialize_reasoning(dag)`
  - Calls: DependencyInference.infer_dependencies, PatternFormation.find_patterns, ReasoningState, RuleEvaluation.create_default_rules, keys
- [Julia | Function] `reason_over_dag` (line 51, pub)
  - Signature: `reason_over_dag(dag, state::ReasoningState)`
  - Calls: ConstraintPropagation.forward_propagate, InferenceResult, PatternFormation.match_pattern, RuleEvaluation.evaluate_rules, StructuralInference.derive_constraints, append!, get, get!, isempty, keys, push!
- [Julia | Function] `perform_inference` (line 102, pub)
  - Signature: `perform_inference(dag, state::ReasoningState, target_node::UInt64)`
  - Calls: Inference, LogicEngine.deduce, append!, filter, get, haskey, isempty, push!

## File: MMSB/src/08_reasoning/rule_evaluation.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `RuleEvaluation` (line 6, pub)
- [Julia | Function] `evaluate_rules` (line 17, pub)
  - Signature: `evaluate_rules(dag, state::ReasoningState, node_id::UInt64)`
  - Calls: Inference, get, isempty, push!, rule.action, rule.condition
- [Julia | Function] `apply_rule` (line 47, pub)
  - Signature: `apply_rule(rule::Rule, dag, node_id::UInt64, constraints::Vector{Constraint})`
  - Calls: rule.action, rule.condition
- [Julia | Function] `create_default_rules` (line 59, pub)
  - Signature: `create_default_rules()`
  - Calls: Constraint, Dict, Rule, UInt64, all, any, findfirst, get, length, push!

## File: MMSB/src/08_reasoning/structural_inference.jl

- Layer(s): 08_reasoning
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `StructuralInference` (line 6, pub)
- [Julia | Function] `infer_from_structure` (line 17, pub)
  - Signature: `infer_from_structure(dag, node_id::UInt64, state::ReasoningState)`
  - Calls: Constraint, Dict, all, get, length, push!
- [Julia | Function] `derive_constraints` (line 58, pub)
  - Signature: `derive_constraints(dag, state::ReasoningState)`
  - Calls: infer_from_structure, isempty, keys
- [Julia | Function] `check_consistency` (line 76, pub)
  - Signature: `check_consistency(constraints::Vector{Constraint})`
  - Calls: in, length

