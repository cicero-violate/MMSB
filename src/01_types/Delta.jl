# src/types/Delta.jl
"""
Delta - Represents a byte-level state change

Deltas capture minimal change sets for page updates.
Contains only changed bytes with mask + data.
"""
module DeltaTypes

using Serialization
using ..PageTypes: PageID

export Delta, DeltaID, delta_byte_count, compress_delta, merge_deltas,
       dense_data, is_sparse_delta, serialize_delta, deserialize_delta

const DeltaID = UInt64
const SPARSE_THRESHOLD = 0.25

"""
    Delta

Represents a byte-level change to a page.
"""
struct Delta
    id::DeltaID
    page_id::PageID
    epoch::UInt32
    mask::BitVector
    data::Vector{UInt8}
    timestamp::UInt64
    source::Symbol
    
    function Delta(id::DeltaID, page_id::PageID, epoch::UInt32, 
                   mask::AbstractVector{Bool}, data::AbstractVector{UInt8}, source::Symbol)
        @assert length(mask) == length(data) "Delta mask/data length mismatch"
        mask_vec = BitVector(mask)
        payload = _compress_payload(mask_vec, Vector{UInt8}(data))
        return new(
            id,
            page_id,
            epoch,
            mask_vec,
            payload,
            time_ns(),
            source,
        )
    end
    
    function Delta(id::DeltaID, page_id::PageID, epoch::UInt32, 
                   mask::AbstractVector{Bool}, data::AbstractVector{UInt8}, 
                   source::Symbol, timestamp::UInt64)
        @assert length(mask) == length(data) || length(data) == count(identity, mask)
        mask_vec = BitVector(mask)
        payload = Vector{UInt8}(data)
        return new(id, page_id, epoch, mask_vec, payload, timestamp, source)
    end
end

"""
Decide whether to store sparse or dense payloads and build buffer accordingly.
"""
function _compress_payload(mask::BitVector, data::Vector{UInt8})::Vector{UInt8}
    changed = count(identity, mask)
    changed == 0 && return UInt8[]
    ratio = changed / length(mask)
    if ratio <= SPARSE_THRESHOLD
        compressed = Vector{UInt8}(undef, changed)
        idx = 1
        @inbounds for i in eachindex(mask)
            if mask[i]
                compressed[idx] = data[i]
                idx += 1
            end
        end
        return compressed
    else
        return copy(data)
    end
end

"""
    delta_byte_count(delta::Delta) -> Int

Count number of changed bytes in delta.
"""
delta_byte_count(delta::Delta)::Int = count(identity, delta.mask)

"""
    is_sparse_delta(delta::Delta) -> Bool

Return true if the delta stores only changed bytes.
"""
is_sparse_delta(delta::Delta)::Bool = length(delta.data) != length(delta.mask)

"""
    dense_data(delta::Delta) -> Vector{UInt8}

Materialize a dense byte vector (length == mask length) for a delta.
"""
function dense_data(delta::Delta)::Vector{UInt8}
    len = length(delta.mask)
    if !is_sparse_delta(delta)
        return copy(delta.data)
    end
    dense = zeros(UInt8, len)
    idx = 1
    @inbounds for i in 1:len
        if delta.mask[i]
            dense[i] = delta.data[idx]
            idx += 1
        end
    end
    return dense
end

"""
    compress_delta(delta::Delta) -> Delta

Create a compressed version of delta (sparse representation).
"""
function compress_delta(delta::Delta)::Delta
    dense = dense_data(delta)
    return Delta(
        delta.id,
        delta.page_id,
        delta.epoch,
        delta.mask,
        dense,
        delta.source,
    )
end

"""
    merge_deltas(d1::Delta, d2::Delta) -> Delta

Merge two deltas targeting the same page with d2 taking precedence.
"""
function merge_deltas(d1::Delta, d2::Delta)::Delta
    @assert d1.page_id == d2.page_id "Cannot merge deltas across pages"
    merged_mask = copy(d1.mask)
    merged_data = dense_data(d1)
    d2_dense = dense_data(d2)
    @inbounds for i in eachindex(d2.mask)
        if d2.mask[i]
            merged_mask[i] = true
            merged_data[i] = d2_dense[i]
        end
    end
    new_epoch = max(d1.epoch, d2.epoch)
    return Delta(d2.id, d2.page_id, new_epoch, merged_mask, merged_data, d2.source)
end

"""
    serialize_delta(delta::Delta) -> Vector{UInt8}

Serialize a delta to a compact byte vector.
"""
function serialize_delta(delta::Delta)::Vector{UInt8}
    payload = (
        delta.id,
        delta.page_id,
        delta.epoch,
        Vector{Bool}(delta.mask),
        delta.data,
        length(delta.data) != length(delta.mask),
        delta.timestamp,
        delta.source,
    )
    io = IOBuffer()
    Serialization.serialize(io, payload)
    return take!(io)
end

"""
    deserialize_delta(bytes::Vector{UInt8}) -> Delta

Reconstruct delta from serialized bytes.
"""
function deserialize_delta(bytes::Vector{UInt8})::Delta
    io = IOBuffer(bytes)
    id, page_id, epoch, mask_vec, data, is_sparse, timestamp, source = Serialization.deserialize(io)
    mask = BitVector(mask_vec)
    if !is_sparse && length(data) != length(mask)
        resize!(data, length(mask))
    end
    return Delta(id, page_id, epoch, mask, data, source, timestamp)
end

end # module DeltaTypes
