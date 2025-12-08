module DeltaTypes

using Serialization
using ..PageTypes: PageID
using ..FFIWrapper

export Delta, DeltaID, new_delta_handle, apply_delta!, dense_data,
       serialize_delta, deserialize_delta

const DeltaID = UInt64
mutable struct Delta
    id::DeltaID
    page_id::PageID
    epoch::UInt32
    is_sparse::Bool
    mask::Vector{UInt8}
    payload::Vector{UInt8}
    timestamp::UInt64
    source::Symbol
    handle::FFIWrapper.RustDeltaHandle
    
    # Inner constructor
    function Delta(id::DeltaID, page_id::PageID, epoch::UInt32,
                   mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol=:ffi;
                   is_sparse::Bool=false)
        handle = FFIWrapper.rust_delta_new(id, page_id, epoch, mask, payload, source; is_sparse=is_sparse)
        timestamp = FFIWrapper.rust_delta_timestamp(handle)
        delta = new(id, page_id, epoch, is_sparse, copy(mask), copy(payload), timestamp, source, handle)
        finalizer(delta) do d
            FFIWrapper.rust_delta_free!(d.handle)
        end
        delta
    end

    function Delta(handle::FFIWrapper.RustDeltaHandle)
        handle.ptr == C_NULL && error("Cannot construct Delta from null handle")
        id = FFIWrapper.rust_delta_id(handle)
        page_id = PageID(FFIWrapper.rust_delta_page_id(handle))
        epoch = FFIWrapper.rust_delta_epoch(handle)
        is_sparse = FFIWrapper.rust_delta_is_sparse(handle)
        timestamp = FFIWrapper.rust_delta_timestamp(handle)
        source = Symbol(FFIWrapper.rust_delta_source(handle))
        mask = FFIWrapper.rust_delta_mask(handle)
        payload = FFIWrapper.rust_delta_payload(handle)
        delta = new(id, page_id, epoch, is_sparse, mask, payload, timestamp, source, handle)
        finalizer(delta) do d
            FFIWrapper.rust_delta_free!(d.handle)
        end
        delta
    end
end

function new_delta_handle(id::DeltaID, page_id::PageID, epoch::UInt32,
                          mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol=:ffi;
                          is_sparse::Bool=false)
    FFIWrapper.rust_delta_new(id, page_id, epoch, mask, payload, source; is_sparse=is_sparse)
end

function apply_delta!(page_handle::FFIWrapper.RustPageHandle, delta::Delta)
    FFIWrapper.rust_delta_apply!(page_handle, delta.handle)
end

function dense_data(delta::Delta)::Vector{UInt8}
    if !delta.is_sparse
        return copy(delta.payload)
    end
    dense = Vector{UInt8}(undef, length(delta.mask))
    idx = 1
    @inbounds for i in eachindex(delta.mask)
        if delta.mask[i] != 0
            dense[i] = delta.payload[idx]
            idx += 1
        else
            dense[i] = 0x00
        end
    end
    dense
end

function serialize_delta(delta::Delta)::Vector{UInt8}
    bool_mask = map(x -> x != 0, delta.mask)
    payload = (
        delta.id,
        delta.page_id,
        delta.epoch,
        bool_mask,
        delta.payload,
        delta.is_sparse,
        delta.timestamp,
        delta.source,
    )
    io = IOBuffer()
    Serialization.serialize(io, payload)
    take!(io)
end

function deserialize_delta(bytes::Vector{UInt8})::Delta
    io = IOBuffer(bytes)
    id, pid, epoch, mask_vec, payload, sparse, timestamp, source = Serialization.deserialize(io)
    mask_bytes = UInt8.(mask_vec)
    delta = Delta(id, pid, epoch, mask_bytes, Vector{UInt8}(payload), source; is_sparse=sparse)
    delta.timestamp = timestamp
    delta
end

end # module
