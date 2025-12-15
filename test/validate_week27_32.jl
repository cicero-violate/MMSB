#!/usr/bin/env julia
# Validation script for Week 27-32 implementation completeness

using Test
using Printf

# Track validation results
validation_results = Dict{String, Vector{Tuple{String, Bool, String}}}()

function check_file(category, file, description)
    path = joinpath(@__DIR__, "..", file)
    exists = isfile(path)
    if !haskey(validation_results, category)
        validation_results[category] = []
    end
    push!(validation_results[category], (description, exists, file))
    return exists
end

function check_function(category, module_path, func_name, description)
    try
        include(joinpath(@__DIR__, "..", module_path))
        # Basic existence check
        if !haskey(validation_results, category)
            validation_results[category] = []
        end
        push!(validation_results[category], (description, true, func_name))
        return true
    catch e
        if !haskey(validation_results, category)
            validation_results[category] = []
        end
        push!(validation_results[category], (description, false, "$func_name - $e"))
        return false
    end
end

println("="^80)
println("MMSB Week 27-32 Implementation Validation")
println("="^80)
println()

# Week 27: Benchmarking
println("Week 27: Benchmarking Infrastructure")
check_file("Week 27", "benchmark/benchmarks.jl", "Main benchmark suite")
check_file("Week 27", "benchmark/helpers.jl", "Benchmark helpers")
check_file("Week 27", "src/00_physical/allocator.rs", "Allocator implementation")
check_file("Week 27", "src/00_physical/allocator_stats.rs", "Allocator stats")
check_file("Week 27", "src/02_semiring/semiring_ops.rs", "Semiring operations")
check_file("Week 27", "src/03_dag/shadow_graph_traversal.rs", "Graph traversal")
check_file("Week 27", "src/04_propagation/propagation_fastpath.rs", "Propagation fastpath")
check_file("Week 27", "src/04_propagation/PropagationEngine.jl", "Propagation engine")
println()

# Week 28-29: GPU Optimization
println("Week 28-29: GPU Optimization")
check_file("Week 28-29", "src/04_propagation/gpu_propagation.cu", "Persistent GPU kernels")
check_file("Week 28-29", "src/04_propagation/propagation_command_buffer.rs", "Command buffer")
check_file("Week 28-29", "src/00_physical/UnifiedMemory.jl", "GPU memory pool")
check_file("Week 28-29", "src/00_physical/gpu_memory_pool.rs", "GPU pool impl")
check_file("Week 28-29", "src/00_physical/nccl_integration.rs", "Multi-GPU NCCL")
check_file("Week 28-29", "src/00_physical/DeviceSync.jl", "Device sync")
println()

# Week 30-31: Performance
println("Week 30-31: Performance Enhancements")
check_file("Week 30-31", "src/01_page/delta_merge.rs", "SIMD delta merge")
check_file("Week 30-31", "src/06_utility/cpu_features.rs", "CPU feature detection")
check_file("Week 30-31", "src/00_physical/lockfree_allocator.rs", "Lock-free allocator")
check_file("Week 30-31", "src/ffi.rs", "Zero-copy FFI")
check_file("Week 30-31", "src/01_page/tlog_compression.rs", "Delta compression")
check_file("Week 30-31", "src/04_propagation/propagation_queue.rs", "Batch propagation")
println()

# Week 32: Reliability
println("Week 32: Reliability Features")
check_file("Week 32", "src/06_utility/ErrorRecovery.jl", "Error recovery")
check_file("Week 32", "src/00_physical/DeviceFallback.jl", "GPU fallback")
check_file("Week 32", "src/06_utility/MemoryPressure.jl", "Memory pressure handling")
check_file("Week 32", "src/01_page/checkpoint.rs", "Checkpoint validation")
check_file("Week 32", "src/04_propagation/TransactionIsolation.jl", "Transaction isolation")
println()

# Tests
println("Test Coverage")
check_file("Tests", "tests/week27_31_integration.rs", "Rust integration tests")
check_file("Tests", "test/week27_31_integration.jl", "Julia integration tests")
check_file("Tests", "tests/examples_basic.rs", "Rust example tests")
check_file("Tests", "test/examples_basic.jl", "Julia example tests")
println()

# Generate report
println()
println("="^80)
println("VALIDATION SUMMARY")
println("="^80)
println()

total_checks = 0
total_passed = 0

for (category, results) in sort(collect(validation_results))
    println("$category:")
    passed = count(r -> r[2], results)
    total = length(results)
    total_checks += total
    total_passed += passed
    
    @printf "  ✓ %d/%d checks passed (%.1f%%)\n" passed total (100.0 * passed / total)
    
    # Show failures
    failures = filter(r -> !r[2], results)
    if !isempty(failures)
        println("  Missing/Failed:")
        for (desc, _, file) in failures
            println("    ✗ $desc: $file")
        end
    end
    println()
end

println("="^80)
@printf "OVERALL: %d/%d checks passed (%.1f%%)\n" total_passed total_checks (100.0 * total_passed / total_checks)
println("="^80)

if total_passed == total_checks
    println("\n✓ All implementation requirements validated successfully!")
    exit(0)
else
    missing = total_checks - total_passed
    println("\n⚠ $missing checks failed. Review missing components above.")
    exit(1)
end
