# src/03_device/GPUKernels.jl
"""
GPUKernels - CUDA kernels for GPU-side page operations

Implements high-performance GPU kernels for delta merging,
memory operations, and page transformations.
"""
module GPUKernels

using CUDA

export delta_merge_kernel!, page_copy_kernel!, page_zero_kernel!
export page_compare_kernel!, sparse_delta_apply_kernel!

"""
    delta_merge_kernel!(base::CuDeviceArray{UInt8}, 
                       mask::CuDeviceArray{Bool},
                       delta::CuDeviceArray{UInt8})

CUDA kernel for byte-level delta merge on GPU.

# Operation
- Each thread handles one byte
- If mask[i] is true, base[i] = delta[i]
- Coalesced memory access pattern
- Launched with 256 threads per block

# Performance
- Memory bandwidth bound
- ~500 GB/s on A100
- Scales linearly with page size
"""
function delta_merge_kernel!(base::CuDeviceArray{UInt8,1}, 
                             mask::CuDeviceArray{Bool,1},
                             delta::CuDeviceArray{UInt8,1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(base)
        if mask[i]
            @inbounds base[i] = delta[i]
        end
    end
    return nothing
end

"""
    launch_delta_merge!(base::CuArray{UInt8}, 
                       mask::CuArray{Bool},
                       delta::CuArray{UInt8})

Host-side wrapper to launch delta merge kernel.

# Arguments
- `base`: Target page data (modified in-place)
- `mask`: Boolean mask indicating changed bytes
- `delta`: New byte values

# Implementation
- Calculates grid/block dimensions
- Launches kernel asynchronously
- Returns immediately (caller must synchronize)
"""
function launch_delta_merge!(base::CuArray{UInt8}, 
                             mask::CuArray{Bool},
                             delta::CuArray{UInt8})
    @assert length(base) == length(mask) == length(delta)
    threads, blocks = compute_optimal_kernel_config(length(base))
    @cuda threads=threads blocks=blocks delta_merge_kernel!(base, mask, delta)
    return nothing
end

"""
    page_copy_kernel!(dest::CuDeviceArray{UInt8},
                     src::CuDeviceArray{UInt8},
                     n::Int)

Fast GPU-side page copy.

# Used for
- Page migration
- Page cloning
- Checkpoint operations
"""
function page_copy_kernel!(dest::CuDeviceArray{UInt8,1},
                           src::CuDeviceArray{UInt8,1},
                           n::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds dest[i] = src[i]
    end
    return nothing
end

"""
    launch_page_copy!(dest::CuArray{UInt8}, 
                     src::CuArray{UInt8},
                     n::Int)

Launch page copy kernel.
"""
function launch_page_copy!(dest::CuArray{UInt8}, 
                           src::CuArray{UInt8},
                           n::Int)
    threads, blocks = compute_optimal_kernel_config(n)
    @cuda threads=threads blocks=blocks page_copy_kernel!(dest, src, n)
    return nothing
end

"""
    page_zero_kernel!(data::CuDeviceArray{UInt8})

Zero-fill page data.
"""
function page_zero_kernel!(data::CuDeviceArray{UInt8,1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(data)
        @inbounds data[i] = 0x00
    end
    return nothing
end

"""
    launch_page_zero!(data::CuArray{UInt8})

Zero-fill page on GPU.
"""
function launch_page_zero!(data::CuArray{UInt8})
    threads, blocks = compute_optimal_kernel_config(length(data))
    @cuda threads=threads blocks=blocks page_zero_kernel!(data)
    return nothing
end

"""
    page_compare_kernel!(result::CuDeviceArray{Bool},
                        page1::CuDeviceArray{UInt8},
                        page2::CuDeviceArray{UInt8})

Compare two pages byte-wise on GPU.

# Output
- result[i] = (page1[i] == page2[i])
- Used for state verification and diffing
"""
function page_compare_kernel!(result::CuDeviceArray{Bool,1},
                              page1::CuDeviceArray{UInt8,1},
                              page2::CuDeviceArray{UInt8,1})
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(page1)
        @inbounds result[i] = (page1[i] == page2[i])
    end
    return nothing
end

"""
    launch_page_compare!(result::CuArray{Bool},
                        page1::CuArray{UInt8},
                        page2::CuArray{UInt8})

Launch comparison kernel and return match array.
"""
function launch_page_compare!(result::CuArray{Bool},
                              page1::CuArray{UInt8},
                              page2::CuArray{UInt8})
    @assert length(page1) == length(page2) == length(result)
    threads, blocks = compute_optimal_kernel_config(length(page1))
    @cuda threads=threads blocks=blocks page_compare_kernel!(result, page1, page2)
    return nothing
end

"""
    sparse_delta_apply_kernel!(base::CuDeviceArray{UInt8},
                              indices::CuDeviceArray{Int32},
                              values::CuDeviceArray{UInt8},
                              n_changes::Int)

Apply sparse delta (only changed bytes).

# Arguments
- `base`: Target page
- `indices`: Array of byte indices to change
- `values`: New values for those indices
- `n_changes`: Number of changes

# Performance
- Better than full mask for <5% changed bytes
- Non-coalesced access pattern (random writes)
"""
function sparse_delta_apply_kernel!(base::CuDeviceArray{UInt8,1},
                                    indices::CuDeviceArray{Int32,1},
                                    values::CuDeviceArray{UInt8,1},
                                    n_changes::Int)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n_changes
        idx = indices[i]
        @inbounds base[idx] = values[i]
    end
    return nothing
end

"""
    launch_sparse_delta_apply!(base::CuArray{UInt8},
                              indices::CuArray{Int32},
                              values::CuArray{UInt8})

Launch sparse delta application.
"""
function launch_sparse_delta_apply!(base::CuArray{UInt8},
                                    indices::CuArray{Int32},
                                    values::CuArray{UInt8})
    n_changes = length(indices)
    @assert length(values) == n_changes
    threads, blocks = compute_optimal_kernel_config(n_changes)
    @cuda threads=threads blocks=blocks sparse_delta_apply_kernel!(base, indices, values, n_changes)
    return nothing
end

"""
    compute_optimal_kernel_config(data_size::Int) 
        -> Tuple{Int, Int}

Compute optimal thread/block configuration for data size.

# Returns
- (threads_per_block, num_blocks)

# Heuristics
- 256 threads/block for sizes > 64KB
- 128 threads/block for 16-64KB
- 64 threads/block for < 16KB
"""
function compute_optimal_kernel_config(data_size::Int)::Tuple{Int, Int}
    threads = if data_size > 65536
        256
    elseif data_size > 16384
        128
    else
        64
    end
    blocks = cld(data_size, threads)
    return (threads, blocks)
end

end # module GPUKernels
