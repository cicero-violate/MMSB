# src/types/Page.jl
"""
Page - Core memory page abstraction

A versioned, byte-addressable memory page that can live on CPU or GPU.
Supports delta-based updates with epoch tracking.
"""
module PageTypes

using CUDA
using Serialization
using ..ErrorTypes: GPUMemoryError, UnsupportedLocationError, SerializationError

export Page, PageLocation, PageID, serialize_page, deserialize_page,
       is_gpu_page, is_cpu_page, page_size_bytes

"""
Unique identifier for pages across the system.
"""
const PageID = UInt64

"""
Enum for page location (CPU, GPU, or unified memory).
"""
@enum PageLocation begin
    CPU_LOCATION
    GPU_LOCATION
    UNIFIED_LOCATION
end

"""
Allocate mask/data buffers for a page based on the requested location.
Falls back to CPU allocations if CUDA is unavailable.
"""
function _allocate_buffers(size::Int, location::PageLocation)
    @assert size > 0 "Page size must be positive"
    if location == CPU_LOCATION
        return (falses(size), zeros(UInt8, size))
    elseif location in (GPU_LOCATION, UNIFIED_LOCATION)
        CUDA.functional() || throw(GPUMemoryError("CUDA is not available; cannot allocate GPU page"))
        mask = CUDA.fill(false, size)
        data = CUDA.zeros(UInt8, size)
        return (mask, data)
    else
        throw(UnsupportedLocationError(string(location), "_allocate_buffers"))
    end
end

"""
    Page

Core memory page structure with versioning and device tracking.
"""
mutable struct Page
    id::PageID
    epoch::UInt32
    mask::Union{Vector{Bool}, BitVector, CuArray{Bool}}
    data::Union{Vector{UInt8}, CuArray{UInt8}}
    location::PageLocation
    size::Int64
    metadata::Dict{Symbol,Any}
    
    function Page(id::PageID, size::Int64, location::PageLocation)
        mask, data = _allocate_buffers(size, location)
        return new(id, UInt32(0), mask, data, location, size, Dict{Symbol,Any}())
    end
end

"""
Create a deep copy of the metadata dictionary to avoid accidental sharing.
"""
function _clone_metadata(meta::Dict{Symbol,Any})
    return isempty(meta) ? Dict{Symbol,Any}() : Dict{Symbol,Any}(meta)
end

"""
    is_gpu_page(page::Page) -> Bool

Check if page resides on GPU (explicit GPU or unified memory).
"""
is_gpu_page(page::Page)::Bool = page.location in (GPU_LOCATION, UNIFIED_LOCATION)

"""
    is_cpu_page(page::Page) -> Bool

Check if page resides on CPU memory only.
"""
is_cpu_page(page::Page)::Bool = page.location == CPU_LOCATION

"""
    page_size_bytes(page::Page) -> Int64

Get the size of the page in bytes.
"""
page_size_bytes(page::Page)::Int64 = page.size

"""
Pack a boolean vector into a byte array (little-endian bit packing).
"""
function _pack_bool_vector(flags::Vector{Bool})::Vector{UInt8}
    n = length(flags)
    bytes = fill(UInt8(0), cld(n, 8))
    for (idx, flag) in enumerate(flags)
        flag || continue
        byte_idx = Int((idx - 1) ÷ 8) + 1
        bit_idx = UInt8((idx - 1) % 8)
        bytes[byte_idx] |= UInt8(1) << bit_idx
    end
    return bytes
end

"""
Unpack a byte array produced by `_pack_bool_vector` back into booleans.
"""
function _unpack_bool_vector(bytes::Vector{UInt8}, len::Int)::Vector{Bool}
    result = falses(len)
    for idx in 1:len
        byte_idx = Int((idx - 1) ÷ 8) + 1
        bit_idx = UInt8((idx - 1) % 8)
        result[idx] = (bytes[byte_idx] & (UInt8(1) << bit_idx)) != 0
    end
    return result
end

"""
    serialize_page(page::Page) -> Vector{UInt8}

Serialize page metadata, mask, and data into a compressed byte vector.
"""
function _rle_compress(bytes::Vector{UInt8})::Vector{UInt8}
    output = UInt8[]
    i = 1
    len = length(bytes)
    while i <= len
        value = bytes[i]
        run = 1
        while i + run <= len && bytes[i + run] == value && run < typemax(UInt8)
            run += 1
        end
        push!(output, value)
        push!(output, UInt8(run))
        i += run
    end
    return output
end

function _rle_decompress(bytes::Vector{UInt8}, expected_len::Int)::Vector{UInt8}
    output = Vector{UInt8}(undef, expected_len)
    pos = 1
    i = 1
    while i <= length(bytes)
        i + 1 <= length(bytes) || throw(SerializationError("Corrupt RLE payload (truncated run)"))
        value = bytes[i]
        count = Int(bytes[i + 1])
        for _ in 1:count
            pos <= expected_len || throw(SerializationError("RLE overrun while decompressing page"))
            output[pos] = value
            pos += 1
        end
        i += 2
    end
    pos - 1 == expected_len || throw(SerializationError("Incomplete RLE payload"))
    return output
end

function serialize_page(page::Page)::Vector{UInt8}
    mask = Vector{Bool}(page.mask)
    packed_mask = _pack_bool_vector(mask)
    data = Vector{UInt8}(page.data)
    compressed_data = _rle_compress(data)
    payload = (
        page.id,
        page.epoch,
        packed_mask,
        page.size,
        compressed_data,
        Int(page.location),
        page.metadata,
    )
    io = IOBuffer()
    Serialization.serialize(io, payload)
    return take!(io)
end

"""
    deserialize_page(bytes::Vector{UInt8}) -> Page

Reconstruct a page instance from serialized bytes.
"""
function deserialize_page(bytes::Vector{UInt8})::Page
    io = IOBuffer(bytes)
    id, epoch, packed_mask, size, compressed_data, location_val, metadata =
        Serialization.deserialize(io)
    location = PageLocation(location_val)
    mask = _unpack_bool_vector(Vector{UInt8}(packed_mask), size)
    raw_data = _rle_decompress(Vector{UInt8}(compressed_data), size)
    length(raw_data) == size || throw(SerializationError("Corrupt page payload for $id"))
    if location != CPU_LOCATION && !CUDA.functional()
        location = CPU_LOCATION
    end
    page = Page(id, size, location)
    page.epoch = epoch
    if location == CPU_LOCATION
        page.mask = Vector{Bool}(mask)
        page.data = copy(raw_data)
    else
        page.mask = CuArray(Vector{Bool}(mask))
        page.data = CuArray(raw_data)
    end
    page.metadata = _clone_metadata(metadata)
    return page
end

end # module PageTypes
