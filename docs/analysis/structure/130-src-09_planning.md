# Structure Group: src/09_planning

## File: MMSB/src/09_planning/PlanningTypes.jl

- Layer(s): 09_planning
- Language coverage: Julia (12)
- Element types: Function (1), Module (1), Struct (10)
- Total elements: 12

### Elements

- [Julia | Module] `PlanningTypes` (line 6, pub)
- [Julia | Struct] `State` (line 12, pub)
  - Signature: `struct State`
- [Julia | Struct] `Action` (line 18, pub)
  - Signature: `struct Action`
- [Julia | Struct] `Goal` (line 26, pub)
  - Signature: `struct Goal`
- [Julia | Struct] `Plan` (line 34, pub)
  - Signature: `struct Plan`
- [Julia | Struct] `SearchNode` (line 43, pub)
  - Signature: `mutable struct SearchNode`
- [Julia | Struct] `Strategy` (line 52, pub)
  - Signature: `struct Strategy`
- [Julia | Struct] `RolloutResult` (line 59, pub)
  - Signature: `struct RolloutResult`
- [Julia | Struct] `DecisionGraph` (line 66, pub)
  - Signature: `struct DecisionGraph`
- [Julia | Struct] `PlanningState` (line 72, pub)
  - Signature: `mutable struct PlanningState`
- [Julia | Struct] `PlanMetrics` (line 81, pub)
  - Signature: `struct PlanMetrics`
- [Julia | Function] `PlanningState` (line 88, pub)
  - Signature: `PlanningState(initial_state::State)`
  - Calls: DecisionGraph, PlanningState

## File: MMSB/src/09_planning/decision_graphs.jl

- Layer(s): 09_planning
- Language coverage: Julia (6)
- Element types: Function (5), Module (1)
- Total elements: 6

### Elements

- [Julia | Module] `DecisionGraphs` (line 6, pub)
- [Julia | Function] `build_decision_graph` (line 17, pub)
  - Signature: `build_decision_graph(state::State, goal::Goal, actions::Vector{Action}, depth::Int)`
  - Calls: DecisionGraph, expand_graph!
- [Julia | Function] `expand_graph!` (line 31, pub)
  - Signature: `expand_graph!(graph::DecisionGraph, state::State, goal::Goal, actions::Vector{Action}, depth::Int)`
  - Calls: all, apply_action_simple, expand_graph!, goal.predicate, haskey, pred
- [Julia | Function] `apply_action_simple` (line 52, pub)
  - Signature: `apply_action_simple(action::Action, state::State)`
  - Calls: State, UInt64, copy, rand
- [Julia | Function] `find_optimal_path` (line 63, pub)
  - Signature: `find_optimal_path(graph::DecisionGraph, start_id::UInt64, goal::Goal)`
  - Calls: goal.predicate, haskey, keys, length, pushfirst!
- [Julia | Function] `prune_graph` (line 115, pub)
  - Signature: `prune_graph(graph::DecisionGraph, threshold::Float64)`
  - Calls: DecisionGraph, haskey

## File: MMSB/src/09_planning/goal_decomposition.jl

- Layer(s): 09_planning
- Language coverage: Julia (6)
- Element types: Function (5), Module (1)
- Total elements: 6

### Elements

- [Julia | Module] `GoalDecomposition` (line 6, pub)
- [Julia | Function] `decompose_goal` (line 17, pub)
  - Signature: `decompose_goal(goal::Goal, state::State)`
  - Calls: Goal, UInt64, comp, enumerate, haskey, push!
- [Julia | Function] `create_subgoal_hierarchy` (line 59, pub)
  - Signature: `create_subgoal_hierarchy(goals::Vector{Goal})`
  - Calls: isempty
- [Julia | Function] `order_subgoals` (line 76, pub)
  - Signature: `order_subgoals(subgoals::Vector{Goal}, state::State)`
  - Calls: score_subgoal, sort!
- [Julia | Function] `score_subgoal` (line 83, pub)
  - Signature: `score_subgoal(goal::Goal, state::State)`
  - Calls: estimate_achievability
- [Julia | Function] `estimate_achievability` (line 89, pub)
  - Signature: `estimate_achievability(goal::Goal, state::State)`
  - Calls: goal.predicate

## File: MMSB/src/09_planning/optimization_planning.jl

- Layer(s): 09_planning
- Language coverage: Julia (11)
- Element types: Function (10), Module (1)
- Total elements: 11

### Elements

- [Julia | Module] `OptimizationPlanning` (line 6, pub)
- [Julia | Function] `optimize_plan` (line 17, pub)
  - Signature: `optimize_plan(plan::Plan, objective::Function)`
  - Calls: compute_gradient, extract_parameters, norm, reconstruct_plan
- [Julia | Function] `extract_parameters` (line 35, pub)
  - Signature: `extract_parameters(plan::Plan)`
  - Calls: parameters, push!
- [Julia | Function] `compute_gradient` (line 44, pub)
  - Signature: `compute_gradient(f::Function, x::Vector{Float64})`
  - Calls: copy, differences, f, length, zeros
- [Julia | Function] `norm` (line 58, pub)
  - Signature: `norm(x::Vector{Float64})`
  - Calls: sqrt, sum
- [Julia | Function] `reconstruct_plan` (line 60, pub)
  - Signature: `reconstruct_plan(plan::Plan, params::Vector{Float64})`
  - Calls: Action, Plan, enumerate, push!, sum
- [Julia | Function] `gradient_descent_planning` (line 88, pub)
  - Signature: `gradient_descent_planning(initial_state::State, goal::Goal, α::Float64`
- [Julia | Function] `evaluate_action_sequence` (line 107, pub)
  - Signature: `evaluate_action_sequence(params::Matrix{Float64}, state::State, goal::Goal)`
  - Calls: State, abs, goal.predicate, size
- [Julia | Function] `compute_sequence_gradient` (line 120, pub)
  - Signature: `compute_sequence_gradient(params::Matrix{Float64}, state::State, goal::Goal)`
  - Calls: copy, evaluate_action_sequence, size, zeros
- [Julia | Function] `actions_from_params` (line 138, pub)
  - Signature: `actions_from_params(params::Matrix{Float64})`
  - Calls: Action, Plan, UInt64, abs, abs., push!, size, sum
- [Julia | Function] `prepare_for_enzyme` (line 158, pub)
  - Signature: `prepare_for_enzyme(plan::Plan)`
  - Calls: length

## File: MMSB/src/09_planning/planning_engine.jl

- Layer(s): 09_planning
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `PlanningEngine` (line 6, pub)
- [Julia | Function] `create_plan` (line 24, pub)
  - Signature: `create_plan(goal::Goal, state::State, actions::Vector{Action})`
  - Calls: OptimizationPlanning.optimize_plan, StrategyGeneration.generate_strategies, StrategyGeneration.select_strategy, strategy.plan_generator, sum
- [Julia | Function] `execute_planning` (line 49, pub)
  - Signature: `execute_planning(planning_state::PlanningState, goal_id::UInt64)`
  - Calls: Goal, GoalDecomposition.decompose_goal, RolloutSimulation.simulate_plan, SearchAlgorithms.mcts_search, create_plan, length
- [Julia | Function] `replan` (line 96, pub)
  - Signature: `replan(planning_state::PlanningState, plan_id::UInt64, feedback::Dict{Symbol, Any})`
  - Calls: RolloutSimulation.simulate_plan, SearchAlgorithms.astar_search, StrategyGeneration.generate_strategies, get, strategy.plan_generator

## File: MMSB/src/09_planning/rl_planning.jl

- Layer(s): 09_planning
- Language coverage: Julia (8)
- Element types: Function (7), Module (1)
- Total elements: 8

### Elements

- [Julia | Module] `RLPlanning` (line 6, pub)
- [Julia | Function] `value_iteration` (line 17, pub)
  - Signature: `value_iteration(states::Vector{State}, actions::Vector{Action}, γ::Float64`
- [Julia | Function] `immediate_reward` (line 50, pub)
  - Signature: `immediate_reward(s::State, a::Action)`
- [Julia | Function] `expected_next_value` (line 54, pub)
  - Signature: `expected_next_value(s::State, a::Action, V::Dict{UInt64, Float64}, states::Vector{State})`
  - Calls: get
- [Julia | Function] `policy_iteration` (line 66, pub)
  - Signature: `policy_iteration(states::Vector{State}, actions::Vector{Action}, γ::Float64`
- [Julia | Function] `evaluate_policy` (line 116, pub)
  - Signature: `evaluate_policy(π::Dict{UInt64, Action}, states::Vector{State}, actions::Vector{Action}, γ::Float64, V::Dict{UInt64, ...`
  - Calls: expected_next_value, immediate_reward
- [Julia | Function] `q_learning` (line 131, pub)
  - Signature: `q_learning(episodes::Vector{Vector{Tuple{State, Action, Float64}}}, α::Float64`
- [Julia | Function] `temporal_difference` (line 155, pub)
  - Signature: `temporal_difference(trajectory::Vector{Tuple{State, Action, Float64}}, α::Float64`

## File: MMSB/src/09_planning/rollout_simulation.jl

- Layer(s): 09_planning
- Language coverage: Julia (4)
- Element types: Function (3), Module (1)
- Total elements: 4

### Elements

- [Julia | Module] `RolloutSimulation` (line 6, pub)
- [Julia | Function] `simulate_plan` (line 18, pub)
  - Signature: `simulate_plan(plan::Plan, start_state::State)`
  - Calls: RolloutResult, SearchAlgorithms.apply_action, SearchAlgorithms.can_apply, push!
- [Julia | Function] `parallel_rollout` (line 43, pub)
  - Signature: `parallel_rollout(plans::Vector{Plan}, start_state::State, n_rollouts::Int`
- [Julia | Function] `evaluate_outcome` (line 63, pub)
  - Signature: `evaluate_outcome(result::RolloutResult)`
  - Calls: length

## File: MMSB/src/09_planning/search_algorithms.jl

- Layer(s): 09_planning
- Language coverage: Julia (16)
- Element types: Function (14), Module (1), Struct (1)
- Total elements: 16

### Elements

- [Julia | Module] `SearchAlgorithms` (line 6, pub)
- [Julia | Function] `astar_search` (line 17, pub)
  - Signature: `astar_search(start_state::State, goal::Goal, actions::Vector{Action}, max_nodes::Int`
- [Julia | Function] `mcts_search` (line 66, pub)
  - Signature: `mcts_search(start_state::State, goal::Goal, actions::Vector{Action}, iterations::Int`
- [Julia | Struct] `MCTSNode` (line 89, pub)
  - Signature: `mutable struct MCTSNode`
- [Julia | Function] `MCTSNode` (line 98, pub)
  - Signature: `MCTSNode(state::State)`
  - Calls: MCTSNode
- [Julia | Function] `select_node` (line 100, pub)
  - Signature: `select_node(node::MCTSNode)`
  - Calls: best_uct_child, isempty
- [Julia | Function] `best_uct_child` (line 107, pub)
  - Signature: `best_uct_child(node::MCTSNode, c::Float64`
- [Julia | Function] `expand_node` (line 129, pub)
  - Signature: `expand_node(node::MCTSNode, actions::Vector{Action})`
  - Calls: MCTSNode, apply_action, can_apply, push!
- [Julia | Function] `simulate` (line 143, pub)
  - Signature: `simulate(state::State, goal::Goal, actions::Vector{Action}, max_depth::Int)`
  - Calls: apply_action, can_apply, filter, goal.predicate, isempty, rand
- [Julia | Function] `backpropagate` (line 165, pub)
  - Signature: `backpropagate(node::MCTSNode, reward::Float64)`
- [Julia | Function] `compute_heuristic` (line 179, pub)
  - Signature: `compute_heuristic(state::State, goal::Goal)`
  - Calls: abs, haskey
- [Julia | Function] `can_apply` (line 187, pub)
  - Signature: `can_apply(action::Action, state::State)`
  - Calls: all, pred
- [Julia | Function] `apply_action` (line 191, pub)
  - Signature: `apply_action(action::Action, state::State)`
  - Calls: State, copy, effect, isa
- [Julia | Function] `is_terminal` (line 205, pub)
  - Signature: `is_terminal(node::MCTSNode, goal::Goal)`
  - Calls: goal.predicate
- [Julia | Function] `reconstruct_plan` (line 209, pub)
  - Signature: `reconstruct_plan(node::SearchNode, goal_id::UInt64)`
  - Calls: Plan, UInt64, pushfirst!
- [Julia | Function] `extract_plan_from_mcts` (line 224, pub)
  - Signature: `extract_plan_from_mcts(root::MCTSNode, goal_id::UInt64)`
  - Calls: Plan, UInt64, best_uct_child, isempty, push!

## File: MMSB/src/09_planning/strategy_generation.jl

- Layer(s): 09_planning
- Language coverage: Julia (5)
- Element types: Function (4), Module (1)
- Total elements: 5

### Elements

- [Julia | Module] `StrategyGeneration` (line 6, pub)
- [Julia | Function] `generate_strategies` (line 19, pub)
  - Signature: `generate_strategies(goal::Goal, state::State)`
  - Calls: SearchAlgorithms.astar_search, SearchAlgorithms.mcts_search, Strategy, UInt64, hierarchical_planning, push!
- [Julia | Function] `hierarchical_planning` (line 49, pub)
  - Signature: `hierarchical_planning(goal::Goal, state::State, actions::Vector{Action})`
  - Calls: GoalDecomposition.decompose_goal, Plan, SearchAlgorithms.apply_action, SearchAlgorithms.astar_search, UInt64, append!
- [Julia | Function] `select_strategy` (line 75, pub)
  - Signature: `select_strategy(strategies::Vector{Strategy}, goal::Goal, state::State)`
  - Calls: argmax, isempty, push!
- [Julia | Function] `adapt_strategy` (line 108, pub)
  - Signature: `adapt_strategy(strategy::Strategy, feedback::Dict{Symbol, Any})`
  - Calls: Strategy, get, strategy.evaluation_fn

