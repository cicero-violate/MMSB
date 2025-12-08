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
include("01_types/Page.jl")
include("01_types/MMSBState.jl")
include("02_runtime/PageAllocator.jl")
include("01_types/Delta.jl")
include("01_types/ShadowPageGraph.jl")

# Rust error mapping shim (depends on FFI + ErrorTypes)
include("ffi/RustErrors.jl")

# Event system is shared by runtime/graph layers
include("05_graph/EventSystem.jl")

# Device kernels are required by the runtime delta router

# Utilities
include("utils/Monitoring.jl")

# Runtime layer
include("02_runtime/PageAllocator.jl")
include("02_runtime/TLog.jl")
include("02_runtime/DeltaRouter.jl")
include("02_runtime/ReplayEngine.jl")

# Remaining device management helpers

# Instrumentation layer (Julia compiler hooks)
include("04_instrumentation/BaseHook.jl")
include("04_instrumentation/CoreHooks.jl")
include("04_instrumentation/CompilerHooks.jl")
include("04_instrumentation/InstrumentationManager.jl")

# Graph and dependency tracking
include("05_graph/DependencyGraph.jl")
include("05_graph/PropagationEngine.jl")

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
