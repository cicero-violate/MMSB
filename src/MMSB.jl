# src/MMSB.jl - main file
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
include("01_types/StateManagement.jl")

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

# Link circular dependency for state reset
StateManagement.clear_propagation_buffers! = PropagationEngine.clear_propagation_buffers!

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
include("06_utility/ErrorRecovery.jl")
include("06_utility/MemoryPressure.jl")

# Layers 7-12 (Intention → Applications) migrated to the MMSB-top package.
# See ../MMSB-top for the cognitive stack.

# ============================================================================
# PUBLIC API EXPORTS
# ============================================================================

using .API: mmsb_start, mmsb_stop, create_page, update_page, query_page, @mmsb
using .MMSBStateTypes: MMSBState, MMSBConfig
using .Monitoring: get_stats, print_stats, reset_stats!, track_delta_latency!, track_propagation_latency!
using .DeltaTypes: Delta

export MMSBState, Page, Delta, ShadowPageGraph
export MMSBConfig
export create_page, delete_page, apply_delta
export add_dependency, remove_dependency
export replay_from_log, checkpoint_state
export mmsb_start, mmsb_stop, update_page, query_page, @mmsb
export get_stats, print_stats, reset_stats!

end # module MMSB
