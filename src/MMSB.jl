# src/MMSB.jl
"""
MMSB - Memory-Mapped State Bus

A structured, versioned, delta-driven shared-memory substrate providing 
deterministic state coherence across CPU, GPU, and program components.

Entry point module that exports all public APIs.
"""
module MMSB

using CUDA

# Rust FFI bridge must be available before type modules choose their backend.
include("ffi/FFIWrapper.jl")

# Core type system
include("01_types/Errors.jl")
include("01_page/Page.jl")
include("01_page/Delta.jl")
include("03_dag/ShadowPageGraph.jl")
include("01_types/MMSBState.jl")

# Rust error mapping shim (depends on FFI + ErrorTypes)
include("ffi/RustErrors.jl")

# Event system is shared by runtime/graph layers
include("03_dag/EventSystem.jl")

# Device kernels are required by the runtime delta router

# Utilities
include("utils/Monitoring.jl")

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

# Instrumentation layer (Julia compiler hooks)
include("04_instrumentation/BaseHook.jl")
include("04_instrumentation/CoreHooks.jl")
include("04_instrumentation/CompilerHooks.jl")
include("04_instrumentation/InstrumentationManager.jl")

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

include("API.jl")

# Public API
using .API: mmsb_start, mmsb_stop, create_page, update_page, query_page, @mmsb
using .Monitoring: get_stats, print_stats, reset_stats!, track_delta_latency!, track_propagation_latency!

# Public API
export MMSBState, Page, Delta, ShadowPageGraph
export create_page, delete_page, apply_delta
export add_dependency, remove_dependency
export enable_instrumentation, disable_instrumentation
export replay_from_log, checkpoint_state
export mmsb_start, mmsb_stop, update_page, query_page, @mmsb
export get_stats, print_stats, reset_stats!

end # module MMSB
