using Test

import MMSB

include(joinpath(@__DIR__, "replay_diff_test.jl"))
include(joinpath(@__DIR__, "checkpoint_error_test.jl"))
include(joinpath(@__DIR__, "thread_safe_test.jl"))
include(joinpath(@__DIR__, "gpu_kernel_test.jl"))
include(joinpath(@__DIR__, "propagation_test.jl"))
include(joinpath(@__DIR__, "error_handling_test.jl"))
include(joinpath(@__DIR__, "public_api_test.jl"))
include(joinpath(@__DIR__, "monitoring_test.jl"))
include(joinpath(@__DIR__, "benchmark_test.jl"))

include(joinpath(@__DIR__, "gc_stress_test.jl"))
include(joinpath(@__DIR__, "fuzz_replay.jl"))
include(joinpath(@__DIR__, "propagation_fuzz.jl"))
include(joinpath(@__DIR__, "checkpoint_fuzz.jl"))

include(joinpath(@__DIR__, "test_layer05_adaptive.jl"))
include(joinpath(@__DIR__, "test_layer06_utility.jl"))

include(joinpath(@__DIR__, "test_signature_system.jl"))
include(joinpath(@__DIR__, "determinism_test.jl"))
include(joinpath(@__DIR__, "state_pool_test.jl"))
include(joinpath(@__DIR__, "state_pool_integration_test.jl"))
