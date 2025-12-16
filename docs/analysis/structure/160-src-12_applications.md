# Structure Group: src/12_applications

## File: MMSB/src/12_applications/financial_modeling.jl

- Layer(s): 12_applications
- Language coverage: Julia (5)
- Element types: Function (2), Module (1), Struct (2)
- Total elements: 5

### Elements

- [Julia | Module] `FinancialModeling` (line 4, pub)
- [Julia | Struct] `Asset` (line 11, pub)
  - Signature: `struct Asset`
- [Julia | Struct] `Portfolio` (line 17, pub)
  - Signature: `mutable struct Portfolio`
- [Julia | Function] `compute_value` (line 25, pub)
  - Signature: `compute_value(portfolio::Portfolio, prices::Dict{String, Float64})::Float64`
  - Calls: get
- [Julia | Function] `rebalance!` (line 33, pub)
  - Signature: `rebalance!(portfolio::Portfolio, target_weights::Dict{String, Float64})`

## File: MMSB/src/12_applications/llm_tools.jl

- Layer(s): 12_applications
- Language coverage: Julia (4)
- Element types: Function (2), Module (1), Struct (1)
- Total elements: 4

### Elements

- [Julia | Module] `LLMTools` (line 4, pub)
- [Julia | Struct] `MMSBContext` (line 11, pub)
  - Signature: `struct MMSBContext`
- [Julia | Function] `query_llm` (line 19, pub)
  - Signature: `query_llm(ctx::MMSBContext, prompt::String)::String`
- [Julia | Function] `store_llm_response` (line 25, pub)
  - Signature: `store_llm_response(ctx::MMSBContext, response::String)`

## File: MMSB/src/12_applications/memory_driven_reasoning.jl

- Layer(s): 12_applications
- Language coverage: Julia (4)
- Element types: Function (2), Module (1), Struct (1)
- Total elements: 4

### Elements

- [Julia | Module] `MemoryDrivenReasoning` (line 4, pub)
- [Julia | Struct] `ReasoningContext` (line 12, pub)
  - Signature: `struct ReasoningContext`
- [Julia | Function] `reason_over_memory` (line 18, pub)
  - Signature: `reason_over_memory(ctx::ReasoningContext)::Dict{Symbol, Any}`
  - Calls: Dict, structural_inference
- [Julia | Function] `temporal_reasoning` (line 23, pub)
  - Signature: `temporal_reasoning(ctx::ReasoningContext)::Vector{Any}`

## File: MMSB/src/12_applications/multi_agent_system.jl

- Layer(s): 12_applications
- Language coverage: Julia (4)
- Element types: Function (2), Module (1), Struct (1)
- Total elements: 4

### Elements

- [Julia | Module] `MultiAgentSystem` (line 4, pub)
- [Julia | Struct] `AgentCoordinator` (line 11, pub)
  - Signature: `mutable struct AgentCoordinator`
- [Julia | Function] `register_agent!` (line 19, pub)
  - Signature: `register_agent!(coord::AgentCoordinator, agent::AbstractAgent)`
  - Calls: push!
- [Julia | Function] `coordinate_step!` (line 23, pub)
  - Signature: `coordinate_step!(coord::AgentCoordinator)`
  - Calls: observe

## File: MMSB/src/12_applications/world_simulation.jl

- Layer(s): 12_applications
- Language coverage: Julia (5)
- Element types: Function (2), Module (1), Struct (2)
- Total elements: 5

### Elements

- [Julia | Module] `WorldSimulation` (line 4, pub)
- [Julia | Struct] `Entity` (line 13, pub)
  - Signature: `struct Entity`
- [Julia | Struct] `World` (line 19, pub)
  - Signature: `mutable struct World`
- [Julia | Function] `add_entity!` (line 28, pub)
  - Signature: `add_entity!(world::World, entity_type::Symbol, props::Dict{Symbol, Any})::Entity`
  - Calls: Entity, allocate_page_id!
- [Julia | Function] `simulate_step!` (line 35, pub)
  - Signature: `simulate_step!(world::World)`

