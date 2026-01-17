"""
    CostFunctions

Cost function components for utility computation in Layer 6.

Defines individual cost metrics (cache misses, memory overhead, latency)
that are aggregated into total utility scores.
"""
module CostFunctions

export CostComponents, compute_cache_cost, compute_memory_cost, compute_latency_cost

"""
    CostComponents

Individual cost metrics extracted from telemetry.
"""
struct CostComponents
    cache_miss_rate::Float64
    memory_overhead_mb::Float64
    avg_latency_us::Float64
    allocation_rate::Float64
end

"""
    compute_cache_cost(cache_misses, cache_hits) -> Float64

Compute cache miss penalty: C_cache = k₁ × (misses / (hits + misses))
"""
function compute_cache_cost(cache_misses::Int, cache_hits::Int)
    total = cache_hits + cache_misses
    total == 0 && return 0.0
    
    miss_rate = cache_misses / total
    k_cache = 1000.0  # Weight factor
    
    return k_cache * miss_rate
end

"""
    compute_memory_cost(bytes_allocated, num_allocations) -> Float64

Compute memory overhead cost: C_mem = k₂ × (MB allocated / allocation)
"""
function compute_memory_cost(bytes_allocated::Int, num_allocations::Int)
    num_allocations == 0 && return 0.0
    
    mb_per_alloc = (bytes_allocated / (1024 * 1024)) / num_allocations
    k_mem = 100.0
    
    return k_mem * mb_per_alloc
end

"""
    compute_latency_cost(total_latency_us, num_ops) -> Float64

Compute propagation latency cost: C_latency = k₃ × (avg latency in ms)
"""
function compute_latency_cost(total_latency_us::Int, num_ops::Int)
    num_ops == 0 && return 0.0
    
    avg_latency_ms = (total_latency_us / num_ops) / 1000.0
    k_latency = 10.0
    
    return k_latency * avg_latency_ms
end

"""
    from_telemetry(snapshot) -> CostComponents

Extract cost components from telemetry snapshot.
"""
function from_telemetry(snapshot)
    cache_cost = compute_cache_cost(
        Int(snapshot.cache_misses),
        Int(snapshot.cache_hits)
    )
    
    memory_cost = compute_memory_cost(
        Int(snapshot.bytes_allocated),
        Int(snapshot.allocations)
    )
    
    latency_cost = compute_latency_cost(
        Int(snapshot.propagation_latency_us),
        Int(snapshot.propagations)
    )
    
    allocation_rate = snapshot.allocations == 0 ? 0.0 :
        Float64(snapshot.allocations) / (Float64(snapshot.elapsed_ms) / 1000.0)
    
    CostComponents(cache_cost, memory_cost, latency_cost, allocation_rate)
end

end # module CostFunctions
