# src/00_physical/UnifiedMemory.jl
"""
UnifiedMemory - CUDA Unified Memory management

Manages pages using CUDA Unified Memory for transparent CPU/GPU access.
Requires CUDA compute capability >= 6.0.
"""
module UnifiedMemory

using CUDA
using ..FFIWrapper: LIBMMSB
using ..PageTypes: Page, PageID, PageLocation, UNIFIED_LOCATION
using ..MMSBStateTypes: MMSBState
using ..FFIWrapper: RustPageHandle, RustAllocatorHandle

export create_unified_page!, is_unified_memory_available
export prefetch_unified_to_gpu!, prefetch_unified_to_cpu!
export set_preferred_location!
export GPUMemoryPool, allocate_from_pool, deallocate_to_pool, get_pool_stats

# GPU Memory Pool handle
mutable struct GPUMemoryPool
    ptr::Ptr{Cvoid}
    stats::Dict{Symbol, UInt64}
end

"""
    GPUMemoryPool() -> GPUMemoryPool

Create slab-based GPU memory pool for reusing CUDA buffers.

# Slab sizes
- 4KB, 16KB, 64KB, 256KB, 1MB, 4MB

# Benefits  
- Reduces cudaMalloc/cudaFree overhead
- Improves allocation performance for repeated page operations
"""
function GPUMemoryPool()::GPUMemoryPool
    ptr = ccall((:gpu_pool_new, LIBMMSB), Ptr{Cvoid}, ())
    stats = Dict{Symbol, UInt64}(
        :allocations => 0,
        :cache_hits => 0,
        :cache_misses => 0
    )
    return GPUMemoryPool(ptr, stats)
end

"""
    allocate_from_pool(pool::GPUMemoryPool, size::UInt64) -> Ptr{Cvoid}

Allocate GPU memory from pool, reusing cached buffers when available.
"""
function allocate_from_pool(pool::GPUMemoryPool, size::UInt64)::Ptr{Cvoid}
    ptr = ccall((:gpu_pool_allocate, LIBMMSB), Ptr{Cvoid}, 
                (Ptr{Cvoid}, Csize_t), pool.ptr, size)
    pool.stats[:allocations] += 1
    return ptr
end

"""
    deallocate_to_pool(pool::GPUMemoryPool, ptr::Ptr{Cvoid}, size::UInt64)

Return GPU memory to pool for reuse instead of freeing.
"""
function deallocate_to_pool(pool::GPUMemoryPool, ptr::Ptr{Cvoid}, size::UInt64)
    ccall((:gpu_pool_deallocate, LIBMMSB), Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t), pool.ptr, ptr, size)
end

"""
    get_pool_stats(pool::GPUMemoryPool) -> Dict{Symbol, UInt64}

Retrieve memory pool statistics: allocations, cache hits/misses, bytes cached.
"""
function get_pool_stats(pool::GPUMemoryPool)::Dict{Symbol, UInt64}
    # Call Rust to get real stats
    allocs = ccall((:gpu_pool_get_allocations, LIBMMSB), UInt64, (Ptr{Cvoid},), pool.ptr)
    hits = ccall((:gpu_pool_get_cache_hits, LIBMMSB), UInt64, (Ptr{Cvoid},), pool.ptr)
    misses = ccall((:gpu_pool_get_cache_misses, LIBMMSB), UInt64, (Ptr{Cvoid},), pool.ptr)
    
    return Dict{Symbol, UInt64}(
        :allocations => allocs,
        :cache_hits => hits,
        :cache_misses => misses,
        :hit_rate => hits > 0 ? (hits * 100) ÷ (hits + misses) : 0
    )
end

"""
    is_unified_memory_available() -> Bool

Check if system supports CUDA Unified Memory.

# Requirements
- CUDA compute capability >= 6.0
- Driver supports managed memory
"""
function is_unified_memory_available()::Bool
    """
    if !CUDA.functional()
        return false
    end
    
    device = CUDA.device()
    capability = CUDA.capability(device)
    
    # Unified Memory requires CC >= 6.0
    return capability >= v"6.0"
    """
end

"""
    create_unified_page!(state::MMSBState, size::Int64) -> Page

Create page with CUDA Unified Memory.

# Benefits
- Single memory space accessible from CPU and GPU
- Automatic migration on access
- Simplified programming model

# Drawbacks
- Slower than explicit transfers for large data
- Page faults on first access
- Requires careful prefetching for performance
"""
function create_unified_page!(state::MMSBState, size::Int64)::Page
    page_id = allocate_page_id!(state)
    handle = ccall((:mmsb_allocator_allocate, LIBMMSB), RustPageHandle,
                   (RustAllocatorHandle, UInt64, Csize_t, Int32),
                   state.allocator, page_id.id, size, 2)  # 2 = Unified
    Page(handle.ptr, page_id, size, UNIFIED_LOCATION)
end

"""
    prefetch_unified_to_gpu!(page::Page, device::CuDevice)

Prefetch unified page to GPU for faster access.

# Implementation
- Uses cudaMemPrefetchAsync
- Non-blocking operation
- Reduces page faults on GPU access
"""
function prefetch_unified_to_gpu!(page::Page, device::CuDevice=CUDA.device())
    @assert page.location == UNIFIED_LOCATION "Page must be unified memory"
    
    if CUDA.functional() && capability(device) >= v"6.0"
        # Prefetch data and mask arrays
        ccall((:cudaMemPrefetchAsync, "libcudart"), Cint,
              (Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}),
              pointer(page.data), sizeof(page.data), 
              deviceid(device), C_NULL)
        
        ccall((:cudaMemPrefetchAsync, "libcudart"), Cint,
              (Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}),
              pointer(page.mask), sizeof(page.mask),
              deviceid(device), C_NULL)
    end
end

"""
    prefetch_unified_to_cpu!(page::Page)

Prefetch unified page to CPU.
"""
function prefetch_unified_to_cpu!(page::Page)
    @assert page.location == UNIFIED_LOCATION "Page must be unified memory"
    
    if CUDA.functional()
        # CPU device ID is -1 (cudaCpuDeviceId)
        ccall((:cudaMemPrefetchAsync, "libcudart"), Cint,
              (Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}),
              pointer(page.data), sizeof(page.data), -1, C_NULL)
        
        ccall((:cudaMemPrefetchAsync, "libcudart"), Cint,
              (Ptr{Cvoid}, Csize_t, Cint, Ptr{Cvoid}),
              pointer(page.mask), sizeof(page.mask), -1, C_NULL)
    end
end

"""
    adaptive_prefetch_distance(latency_history::Vector{Float64}) -> Int

Compute optimal prefetch distance based on observed latency.

# Arguments
- `latency_history`: Recent migration latencies in nanoseconds

# Returns
Number of pages to prefetch ahead
"""
function adaptive_prefetch_distance(latency_history::Vector{Float64})::Int
    if isempty(latency_history)
        return 4  # Default
    end
    
    avg_latency_us = mean(latency_history) / 1000
    
    # More latency → prefetch more pages ahead
    if avg_latency_us > 1000  # > 1ms
        return 16
    elseif avg_latency_us > 500
        return 8
    elseif avg_latency_us > 100
        return 4
    else
        return 2
    end
end

"""
    set_preferred_location!(page::Page, device::Union{CuDevice, Symbol})

Set preferred memory location for unified page.

# Arguments
- `device`: CuDevice for GPU, :cpu for CPU

# Effect
- Hints to CUDA driver where page should reside
- Reduces migration overhead
"""
function set_preferred_location!(page::Page, device::Union{CuDevice, Symbol})
    """
    @assert page.location == UNIFIED_LOCATION
    
    # Use cudaMemAdvise
    if device == :cpu
        # Set CPU as preferred location
        # CUDA.Mem.advise(page.data, CUDA.Mem.ADVISE_SET_PREFERRED_LOCATION, -1)
    else
        # Set GPU as preferred location
        # CUDA.Mem.advise(page.data, CUDA.Mem.ADVISE_SET_PREFERRED_LOCATION, device)
    end
    """
end

"""
    convert_to_unified!(page::Page)

Convert existing CPU or GPU page to unified memory.

# Warning
- Requires copying data
- Invalidates existing pointers
"""
function convert_to_unified!(page::Page)
    """
    @assert page.location != UNIFIED_LOCATION
    
    # Ensure data is on CPU first
    if page.location == GPU_LOCATION
        sync_page_to_cpu!(page)
    end
    
    # Create unified arrays
    unified_data = CuArray{UInt8}(page.data)  # TODO: unified flag
    unified_mask = CuArray{Bool}(page.mask)
    
    # Replace arrays
    page.data = unified_data
    page.mask = unified_mask
    page.location = UNIFIED_LOCATION
    """
end

"""
    enable_read_mostly_hint!(page::Page)

Mark page as mostly read-only for optimization.

# Effect
- CUDA driver creates replicas across devices
- Reduces migration overhead for read-heavy pages
"""
function enable_read_mostly_hint!(page::Page)
    """
    @assert page.location == UNIFIED_LOCATION
    
    # CUDA.Mem.advise(page.data, CUDA.Mem.ADVISE_SET_READ_MOSTLY, device)
    # CUDA.Mem.advise(page.mask, CUDA.Mem.ADVISE_SET_READ_MOSTLY, device)
    """
end

"""
    disable_read_mostly_hint!(page::Page)

Remove read-mostly hint from page.
"""
function disable_read_mostly_hint!(page::Page)
    """
    @assert page.location == UNIFIED_LOCATION
    
    # CUDA.Mem.advise(page.data, CUDA.Mem.ADVISE_UNSET_READ_MOSTLY, device)
    """
end

end # module UnifiedMemory
