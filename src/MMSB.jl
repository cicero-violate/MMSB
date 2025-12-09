# src/MMSB.jl
"""
MMSB - Memory-Mapped State Bus
"""
module MMSB

using CUDA

# ============================================================================
# ACTIVE LAYERS (Layers 0-5 + API)
# ============================================================================

# FFI bridge
include("ffi/FFIWrapper.jl")

# Core type system
include("01_types/Errors.jl")
include("01_page/Page.jl")
include("01_page/Delta.jl")
include("03_dag/ShadowPageGraph.jl")
include("01_types/MMSBState.jl")

# Rust error mapping
include("ffi/RustErrors.jl")

# Event system
include("03_dag/EventSystem.jl")

# Utilities
include("06_utility/Monitoring.jl")

# Physical memory layer
include("00_physical/PageAllocator.jl")
include("00_physical/UnifiedMemory.jl")
include("00_physical/DeviceSync.jl")
include("00_physical/GPUKernels.jl")

# Runtime layer
include("01_page/TLog.jl")
include("02_semiring/Semiring.jl")
include("02_semiring/SemiringConfig.jl")
include("02_semiring/DeltaRouter.jl")
include("01_page/ReplayEngine.jl")

# Graph and dependency tracking
include("03_dag/DependencyGraph.jl")
include("03_dag/GraphDSL.jl")
include("04_propagation/PropagationEngine.jl")
include("04_propagation/PropagationScheduler.jl")

# Layer 5: Adaptive Memory
include("05_adaptive/AdaptiveLayout.jl")
include("05_adaptive/GraphRewriting.jl")
include("05_adaptive/EntropyReduction.jl")
include("05_adaptive/LocalityAnalysis.jl")

# Public API (load after all infrastructure)
include("API.jl")

# ============================================================================
# INACTIVE LAYERS (To be enabled incrementally)
# ============================================================================

# # Layer 6: Utility Engine
include("06_utility/cost_functions.jl")
include("06_utility/utility_engine.jl")
include("06_utility/entropy_measure.jl")
include("06_utility/CostAggregation.jl")

# # Layer 7: Intention Engine
include("07_intention/IntentionTypes.jl")
include("07_intention/intention_engine.jl")
include("07_intention/goal_emergence.jl")
include("07_intention/structural_preferences.jl")
include("07_intention/attractor_states.jl")

# # Layer 8: Reasoning Engine
include("08_reasoning/ReasoningTypes.jl")
include("08_reasoning/structural_inference.jl")
include("08_reasoning/constraint_propagation.jl")
include("08_reasoning/dependency_inference.jl")
include("08_reasoning/pattern_formation.jl")
include("08_reasoning/rule_evaluation.jl")
include("08_reasoning/logic_engine.jl")
include("08_reasoning/reasoning_engine.jl")

# # Layer 9: Planning Engine
include("09_planning/PlanningTypes.jl")
include("09_planning/search_algorithms.jl")
include("09_planning/goal_decomposition.jl")
include("09_planning/strategy_generation.jl")
include("09_planning/rollout_simulation.jl")
include("09_planning/decision_graphs.jl")
include("09_planning/rl_planning.jl")
include("09_planning/optimization_planning.jl")
include("09_planning/planning_engine.jl")

# # Layer 10: Agent Interface
# include("10_agent_interface/BaseHook.jl")
# include("10_agent_interface/CoreHooks.jl")
# include("10_agent_interface/CompilerHooks.jl")
# include("10_agent_interface/InstrumentationManager.jl")
# include("10_agent_interface/checkpoint_api.jl")
# include("10_agent_interface/event_subscription.jl")
# include("10_agent_interface/AgentProtocol.jl")

# # Layer 11: External Agents
# include("11_agents/AgentTypes.jl")
# include("11_agents/rl_agent.jl")
# include("11_agents/symbolic_agent.jl")
# include("11_agents/enzyme_integration.jl")
# include("11_agents/lux_models.jl")
# include("11_agents/planning_agent.jl")
# include("11_agents/hybrid_agent.jl")

# # Layer 12: Applications
# include("12_applications/llm_tools.jl")
# include("12_applications/world_simulation.jl")
# include("12_applications/multi_agent_system.jl")
# include("12_applications/financial_modeling.jl")
# include("12_applications/memory_driven_reasoning.jl")

# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

using .API: mmsb_start, mmsb_stop, create_page, update_page, query_page, @mmsb
using .Monitoring: get_stats, print_stats, reset_stats!, track_delta_latency!, track_propagation_latency!

export MMSBState, Page, Delta, ShadowPageGraph
export create_page, delete_page, apply_delta
export add_dependency, remove_dependency
export enable_instrumentation, disable_instrumentation
export replay_from_log, checkpoint_state
export mmsb_start, mmsb_stop, update_page, query_page, @mmsb
export get_stats, print_stats, reset_stats!

# # Layer 8 exports
# using .ReasoningTypes
# using .ReasoningEngine
# export Constraint, Dependency, Pattern, Rule, Inference
# export ReasoningState, InferenceResult

# # Layer 9 exports
# using .PlanningTypes
# using .PlanningEngine
# using .RolloutSimulation
# export State, Action, Goal, Plan, SearchNode
# export Strategy, RolloutResult, DecisionGraph, PlanningState

# # Layer 10 exports
# using .CheckpointAPI
# using .EventSubscription
# using .AgentProtocol
# export create_checkpoint, restore_checkpoint, list_checkpoints
# export subscribe_to_events, unsubscribe, EventType, @event
# export AbstractAgent, observe, act!, plan, AgentAction

# # Layer 11 exports
# using .AgentTypes
# using .RLAgents
# using .SymbolicAgents
# using .PlanningAgents
# using .HybridAgents
# export RLAgent, SymbolicAgent, PlanningAgent, HybridAgent
# export AgentState, AgentMemory

# # Layer 12 exports
# using .LLMTools
# using .WorldSimulation
# using .MultiAgentSystem
# using .FinancialModeling
# using .MemoryDrivenReasoning
# export MMSBContext, World, AgentCoordinator, Portfolio, ReasoningContext

end # module MMSB
