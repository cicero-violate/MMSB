# Layer Dependency Report

Generated: 2025-12-17 00:41:41

## Rust Layer Graph

### Layer Order
1. `root` (cycle)
2. `02_semiring`
3. `03_device`
4. `05_adaptive`
5. `06_utility`
6. `00_physical` (cycle)
7. `01_page` (cycle)
8. `01_types` (cycle)
9. `03_dag` (cycle)
10. `04_propagation` (cycle)

### Cycles Detected
- `00_physical`
- `01_page`
- `01_types`
- `03_dag`
- `04_propagation`
- `root`

### Layer Violations
- `00_physical` depends on `01_page` (6 references)
  - MMSB/src/00_physical/allocator.rs :: use crate :: page :: { Delta , DeltaID , Source } ;
  - MMSB/src/00_physical/allocator.rs :: use crate :: page :: { Epoch , Page , PageError , PageID , PageLocation } ;
  - MMSB/src/00_physical/device.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/00_physical/device_registry.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/00_physical/host_device_sync.rs :: use crate :: page :: PageID ;
  - MMSB/src/00_physical/lockfree_allocator.rs :: use crate :: page :: { Page , PageID , PageLocation } ;

### Dependency Edges
- `00_physical` → `01_page` (1 references)
  - MMSB/src/01_page/checkpoint.rs :: use crate :: physical :: allocator :: { PageAllocator , PageSnapshotData } ;
- `00_physical` → `root` (1 references)
  - MMSB/src/ffi.rs :: use crate :: physical :: allocator :: { PageAllocator , PageAllocatorConfig } ;
- `01_page` → `00_physical` (6 references, VIOLATION)
  - MMSB/src/00_physical/allocator.rs :: use crate :: page :: { Delta , DeltaID , Source } ;
  - MMSB/src/00_physical/allocator.rs :: use crate :: page :: { Epoch , Page , PageError , PageID , PageLocation } ;
  - MMSB/src/00_physical/device.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/00_physical/device_registry.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/00_physical/host_device_sync.rs :: use crate :: page :: PageID ;
  - MMSB/src/00_physical/lockfree_allocator.rs :: use crate :: page :: { Page , PageID , PageLocation } ;
- `01_page` → `01_types` (1 references)
  - MMSB/src/01_types/mod.rs :: pub use crate :: page :: * ;
- `01_page` → `03_dag` (3 references)
  - MMSB/src/03_dag/cycle_detection.rs :: use crate :: page :: PageID ;
  - MMSB/src/03_dag/shadow_graph.rs :: use crate :: page :: PageID ;
  - MMSB/src/03_dag/shadow_graph_traversal.rs :: use crate :: page :: PageID ;
- `01_page` → `04_propagation` (3 references)
  - MMSB/src/04_propagation/propagation_command_buffer.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/04_propagation/propagation_engine.rs :: use crate :: page :: { Page , PageID } ;
  - MMSB/src/04_propagation/propagation_fastpath.rs :: use crate :: page :: Page ;
- `01_page` → `root` (3 references)
  - MMSB/src/ffi.rs :: use crate :: page :: checkpoint ;
  - MMSB/src/ffi.rs :: use crate :: page :: tlog :: { TransactionLog , TransactionLogReader } ;
  - MMSB/src/ffi.rs :: use crate :: page :: { Delta , DeltaID , Epoch , Page , PageError , PageID , PageLocation , Source } ;
- `02_semiring` → `root` (1 references)
  - MMSB/src/ffi.rs :: use crate :: semiring :: { accumulate , fold_add , fold_mul , BooleanSemiring , Semiring , TropicalSemiring , } ;

### Unresolved References
- None.

## Julia Layer Graph

### Layer Order
1. `root`
2. `00_physical`
3. `01_page`
4. `01_types`
5. `02_semiring`
6. `03_dag`
7. `04_propagation`
8. `05_adaptive`
9. `06_utility`
10. `07_intention`
11. `08_reasoning`
12. `09_planning`
13. `10_agent_interface`
14. `11_agents`
15. `12_applications`

### Layer Violations
- None detected.

### Dependency Edges
- `00_physical` → `root` (4 references)
  - MMSB/src/MMSB.jl :: include("00_physical/DeviceSync.jl"
  - MMSB/src/MMSB.jl :: include("00_physical/GPUKernels.jl"
  - MMSB/src/MMSB.jl :: include("00_physical/PageAllocator.jl"
  - MMSB/src/MMSB.jl :: include("00_physical/UnifiedMemory.jl"
- `01_page` → `root` (5 references)
  - MMSB/src/MMSB.jl :: include("01_page/Delta.jl"
  - MMSB/src/MMSB.jl :: include("01_page/Page.jl"
  - MMSB/src/MMSB.jl :: include("01_page/ReplayEngine.jl"
  - MMSB/src/MMSB.jl :: include("01_page/TLog.jl"
  - MMSB/src/MMSB.jl :: using .DeltaTypes
- `01_types` → `root` (3 references)
  - MMSB/src/MMSB.jl :: include("01_types/Errors.jl"
  - MMSB/src/MMSB.jl :: include("01_types/MMSBState.jl"
  - MMSB/src/MMSB.jl :: using .MMSBStateTypes
- `02_semiring` → `root` (3 references)
  - MMSB/src/MMSB.jl :: include("02_semiring/DeltaRouter.jl"
  - MMSB/src/MMSB.jl :: include("02_semiring/Semiring.jl"
  - MMSB/src/MMSB.jl :: include("02_semiring/SemiringConfig.jl"
- `03_dag` → `root` (4 references)
  - MMSB/src/MMSB.jl :: include("03_dag/DependencyGraph.jl"
  - MMSB/src/MMSB.jl :: include("03_dag/EventSystem.jl"
  - MMSB/src/MMSB.jl :: include("03_dag/GraphDSL.jl"
  - MMSB/src/MMSB.jl :: include("03_dag/ShadowPageGraph.jl"
- `04_propagation` → `root` (2 references)
  - MMSB/src/MMSB.jl :: include("04_propagation/PropagationEngine.jl"
  - MMSB/src/MMSB.jl :: include("04_propagation/PropagationScheduler.jl"
- `05_adaptive` → `root` (4 references)
  - MMSB/src/MMSB.jl :: include("05_adaptive/AdaptiveLayout.jl"
  - MMSB/src/MMSB.jl :: include("05_adaptive/EntropyReduction.jl"
  - MMSB/src/MMSB.jl :: include("05_adaptive/GraphRewriting.jl"
  - MMSB/src/MMSB.jl :: include("05_adaptive/LocalityAnalysis.jl"
- `06_utility` → `root` (8 references)
  - MMSB/src/MMSB.jl :: include("06_utility/CostAggregation.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/ErrorRecovery.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/MemoryPressure.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/Monitoring.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/cost_functions.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/entropy_measure.jl"
  - MMSB/src/MMSB.jl :: include("06_utility/utility_engine.jl"
  - MMSB/src/MMSB.jl :: using .Monitoring
- `07_intention` → `root` (7 references)
  - MMSB/src/MMSB.jl :: include("07_intention/IntentionTypes.jl"
  - MMSB/src/MMSB.jl :: include("07_intention/UpsertPlan.jl"
  - MMSB/src/MMSB.jl :: include("07_intention/attractor_states.jl"
  - MMSB/src/MMSB.jl :: include("07_intention/goal_emergence.jl"
  - MMSB/src/MMSB.jl :: include("07_intention/intent_lowering.jl"
  - MMSB/src/MMSB.jl :: include("07_intention/intention_engine.jl"
  - MMSB/src/MMSB.jl :: include("07_intention/structural_preferences.jl"
- `08_reasoning` → `root` (10 references)
  - MMSB/src/MMSB.jl :: include("08_reasoning/ReasoningTypes.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/constraint_propagation.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/dependency_inference.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/logic_engine.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/pattern_formation.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/reasoning_engine.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/rule_evaluation.jl"
  - MMSB/src/MMSB.jl :: include("08_reasoning/structural_inference.jl"
  - MMSB/src/MMSB.jl :: using .ReasoningEngine
  - MMSB/src/MMSB.jl :: using .ReasoningTypes
- `09_planning` → `root` (12 references)
  - MMSB/src/MMSB.jl :: include("09_planning/PlanningTypes.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/decision_graphs.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/goal_decomposition.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/optimization_planning.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/planning_engine.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/rl_planning.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/rollout_simulation.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/search_algorithms.jl"
  - MMSB/src/MMSB.jl :: include("09_planning/strategy_generation.jl"
  - MMSB/src/MMSB.jl :: using .PlanningEngine
  - MMSB/src/MMSB.jl :: using .PlanningTypes
  - MMSB/src/MMSB.jl :: using .RolloutSimulation
- `10_agent_interface` → `root` (10 references)
  - MMSB/src/MMSB.jl :: include("10_agent_interface/AgentProtocol.jl"
  - MMSB/src/MMSB.jl :: include("10_agent_interface/BaseHook.jl"
  - MMSB/src/MMSB.jl :: include("10_agent_interface/CompilerHooks.jl"
  - MMSB/src/MMSB.jl :: include("10_agent_interface/CoreHooks.jl"
  - MMSB/src/MMSB.jl :: include("10_agent_interface/InstrumentationManager.jl"
  - MMSB/src/MMSB.jl :: include("10_agent_interface/checkpoint_api.jl"
  - MMSB/src/MMSB.jl :: include("10_agent_interface/event_subscription.jl"
  - MMSB/src/MMSB.jl :: using .AgentProtocol
  - MMSB/src/MMSB.jl :: using .CheckpointAPI
  - MMSB/src/MMSB.jl :: using .EventSubscription
- `11_agents` → `root` (12 references)
  - MMSB/src/MMSB.jl :: include("11_agents/AgentTypes.jl"
  - MMSB/src/MMSB.jl :: include("11_agents/enzyme_integration.jl"
  - MMSB/src/MMSB.jl :: include("11_agents/hybrid_agent.jl"
  - MMSB/src/MMSB.jl :: include("11_agents/lux_models.jl"
  - MMSB/src/MMSB.jl :: include("11_agents/planning_agent.jl"
  - MMSB/src/MMSB.jl :: include("11_agents/rl_agent.jl"
  - MMSB/src/MMSB.jl :: include("11_agents/symbolic_agent.jl"
  - MMSB/src/MMSB.jl :: using .AgentTypes
  - MMSB/src/MMSB.jl :: using .HybridAgents
  - MMSB/src/MMSB.jl :: using .PlanningAgents
  - MMSB/src/MMSB.jl :: using .RLAgents
  - MMSB/src/MMSB.jl :: using .SymbolicAgents
- `12_applications` → `root` (10 references)
  - MMSB/src/MMSB.jl :: include("12_applications/financial_modeling.jl"
  - MMSB/src/MMSB.jl :: include("12_applications/llm_tools.jl"
  - MMSB/src/MMSB.jl :: include("12_applications/memory_driven_reasoning.jl"
  - MMSB/src/MMSB.jl :: include("12_applications/multi_agent_system.jl"
  - MMSB/src/MMSB.jl :: include("12_applications/world_simulation.jl"
  - MMSB/src/MMSB.jl :: using .FinancialModeling
  - MMSB/src/MMSB.jl :: using .LLMTools
  - MMSB/src/MMSB.jl :: using .MemoryDrivenReasoning
  - MMSB/src/MMSB.jl :: using .MultiAgentSystem
  - MMSB/src/MMSB.jl :: using .WorldSimulation

### Unresolved References
- MMSB/benchmark/benchmarks.jl → `using .MMSB`
- MMSB/src/MMSB.jl → `using .API`

