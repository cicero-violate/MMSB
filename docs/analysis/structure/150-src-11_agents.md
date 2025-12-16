# Structure Group: src/11_agents

## File: MMSB/src/11_agents/AgentTypes.jl

- Layer(s): 11_agents
- Language coverage: Julia (4)
- Element types: Function (1), Module (1), Struct (2)
- Total elements: 4

### Elements

- [Julia | Module] `AgentTypes` (line 4, pub)
- [Julia | Struct] `AgentState{T}` (line 12, pub)
  - Signature: `mutable struct AgentState{T}`
- [Julia | Struct] `AgentMemory` (line 20, pub)
  - Signature: `struct AgentMemory`
- [Julia | Function] `push_memory!` (line 29, pub)
  - Signature: `push_memory!(mem::AgentMemory, obs, action, reward::Float64)`
  - Calls: length, popfirst!, push!

## File: MMSB/src/11_agents/enzyme_integration.jl

- Layer(s): 11_agents
- Language coverage: Julia (3)
- Element types: Function (2), Module (1)
- Total elements: 3

### Elements

- [Julia | Module] `EnzymeIntegration` (line 4, pub)
- [Julia | Function] `gradient_descent_step!` (line 11, pub)
  - Signature: `gradient_descent_step!(params::Vector{Float64}, loss_fn::Function, lr::Float64)`
  - Calls: Enzyme.gradient
- [Julia | Function] `autodiff_loss` (line 18, pub)
  - Signature: `autodiff_loss(f::Function, x::Vector{Float64})::Tuple{Float64, Vector{Float64}}`
  - Calls: f, length, return, zeros

## File: MMSB/src/11_agents/hybrid_agent.jl

- Layer(s): 11_agents
- Language coverage: Julia (5)
- Element types: Function (3), Module (1), Struct (1)
- Total elements: 5

### Elements

- [Julia | Module] `HybridAgents` (line 4, pub)
- [Julia | Struct] `HybridAgent` (line 15, pub)
  - Signature: `struct HybridAgent <: AbstractAgent`
- [Julia | Function] `observe` (line 23, pub)
  - Signature: `observe(agent::HybridAgent, state::MMSBState)`
  - Calls: observe, return
- [Julia | Function] `symbolic_step!` (line 30, pub)
  - Signature: `symbolic_step!(agent::HybridAgent, state::MMSBState)::Vector{AgentAction}`
  - Calls: apply_rule, first, isempty
- [Julia | Function] `neural_step!` (line 36, pub)
  - Signature: `neural_step!(agent::HybridAgent, state::MMSBState, action::AgentAction)`
  - Calls: train_step!

## File: MMSB/src/11_agents/lux_models.jl

- Layer(s): 11_agents
- Language coverage: Julia (3)
- Element types: Function (2), Module (1)
- Total elements: 3

### Elements

- [Julia | Module] `LuxModels` (line 4, pub)
- [Julia | Function] `create_value_network` (line 11, pub)
  - Signature: `create_value_network(input_dim::Int, hidden_dims::Vector{Int})`
- [Julia | Function] `create_policy_network` (line 17, pub)
  - Signature: `create_policy_network(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int})`

## File: MMSB/src/11_agents/planning_agent.jl

- Layer(s): 11_agents
- Language coverage: Julia (5)
- Element types: Function (3), Module (1), Struct (1)
- Total elements: 5

### Elements

- [Julia | Module] `PlanningAgents` (line 4, pub)
- [Julia | Struct] `PlanningAgent` (line 15, pub)
  - Signature: `struct PlanningAgent <: AbstractAgent`
- [Julia | Function] `observe` (line 23, pub)
  - Signature: `observe(agent::PlanningAgent, state::MMSBState)`
  - Calls: compute_intention, return
- [Julia | Function] `generate_plan` (line 30, pub)
  - Signature: `generate_plan(agent::PlanningAgent, state::MMSBState, goal::Any)::Vector{AgentAction}`
  - Calls: search_plan
- [Julia | Function] `execute_plan_step` (line 36, pub)
  - Signature: `execute_plan_step(agent::PlanningAgent)::Union{AgentAction, Nothing}`
  - Calls: isempty, popfirst!

## File: MMSB/src/11_agents/rl_agent.jl

- Layer(s): 11_agents
- Language coverage: Julia (6)
- Element types: Function (4), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Julia | Module] `RLAgents` (line 4, pub)
- [Julia | Struct] `RLAgent{T}` (line 13, pub)
  - Signature: `struct RLAgent{T} <: AbstractAgent`
- [Julia | Function] `RLAgent` (line 20, pub)
  - Signature: `RLAgent(initial_state::T; lr`
- [Julia | Function] `observe` (line 24, pub)
  - Signature: `observe(agent::RLAgent, state::MMSBState)`
  - Calls: length, return
- [Julia | Function] `compute_reward` (line 32, pub)
  - Signature: `compute_reward(agent::RLAgent, state::MMSBState, action::AgentAction)::Float64`
- [Julia | Function] `train_step!` (line 37, pub)
  - Signature: `train_step!(agent::RLAgent, state::MMSBState, action::AgentAction)`
  - Calls: compute_reward, observe, push_memory!

## File: MMSB/src/11_agents/symbolic_agent.jl

- Layer(s): 11_agents
- Language coverage: Julia (7)
- Element types: Function (4), Module (1), Struct (2)
- Total elements: 7

### Elements

- [Julia | Module] `SymbolicAgents` (line 4, pub)
- [Julia | Struct] `Rule` (line 14, pub)
  - Signature: `struct Rule`
- [Julia | Struct] `SymbolicAgent` (line 20, pub)
  - Signature: `struct SymbolicAgent <: AbstractAgent`
- [Julia | Function] `SymbolicAgent` (line 25, pub)
  - Signature: `SymbolicAgent()`
  - Calls: AgentState, SymbolicAgent
- [Julia | Function] `observe` (line 27, pub)
  - Signature: `observe(agent::SymbolicAgent, state::MMSBState)`
  - Calls: return, structural_inference
- [Julia | Function] `infer_rules!` (line 32, pub)
  - Signature: `infer_rules!(agent::SymbolicAgent, observations)`
- [Julia | Function] `apply_rule` (line 37, pub)
  - Signature: `apply_rule(agent::SymbolicAgent, rule::Rule, state::MMSBState)::Vector{AgentAction}`

