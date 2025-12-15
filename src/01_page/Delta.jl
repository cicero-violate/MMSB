module DeltaTypes

using Serialization
using ..PageTypes: PageID
using ..FFIWrapper

export Delta, DeltaID, new_delta_handle, apply_delta!, dense_data,
       serialize_delta, deserialize_delta, set_intent_metadata!, intent_metadata

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
    intent_metadata::Union{Nothing,String}
    handle::FFIWrapper.RustDeltaHandle
    
    # Inner constructor
    function Delta(id::DeltaID, page_id::PageID, epoch::UInt32,
                   mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol=:ffi;
                   is_sparse::Bool=false)
        handle = FFIWrapper.rust_delta_new(id, page_id, epoch, mask, payload, source; is_sparse=is_sparse)
        timestamp = FFIWrapper.rust_delta_timestamp(handle)
        delta = new(id, page_id, epoch, is_sparse, copy(mask), copy(payload), timestamp, source, nothing, handle)
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
        metadata = FFIWrapper.rust_delta_intent_metadata(handle)
        delta = new(id, page_id, epoch, is_sparse, mask, payload, timestamp, source, metadata, handle)
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
        delta.intent_metadata,
    )
    io = IOBuffer()
    Serialization.serialize(io, payload)
    take!(io)
end

function deserialize_delta(bytes::Vector{UInt8})::Delta
    io = IOBuffer(bytes)
    id, pid, epoch, mask_vec, payload, sparse, timestamp, source, intent_metadata = Serialization.deserialize(io)
    mask_bytes = UInt8.(mask_vec)
    delta = Delta(id, pid, epoch, mask_bytes, Vector{UInt8}(payload), source; is_sparse=sparse)
    delta.timestamp = timestamp
    delta.intent_metadata = intent_metadata
    delta
end

function set_intent_metadata!(delta::Delta, metadata::Union{Nothing,AbstractString,Dict{Symbol,Any}})
    if metadata === nothing
        FFIWrapper.rust_delta_set_intent_metadata!(delta.handle, nothing)
        delta.intent_metadata = nothing
        return delta
    elseif metadata isa AbstractString
        value = String(metadata)
        FFIWrapper.rust_delta_set_intent_metadata!(delta.handle, value)
        delta.intent_metadata = value
        return delta
    elseif metadata isa Dict{Symbol,Any}
        json = _encode_metadata_dict(metadata)
        FFIWrapper.rust_delta_set_intent_metadata!(delta.handle, json)
        delta.intent_metadata = json
        return delta
    else
        throw(ArgumentError("Unsupported intent metadata type: $(typeof(metadata))"))
    end
end

function intent_metadata(delta::Delta; parse::Bool=false)
    delta.intent_metadata === nothing && return nothing
    parse || return delta.intent_metadata
    _decode_metadata(delta.intent_metadata)
end

function _encode_metadata_value(value)
    if value isa Dict{Symbol,Any}
        return _encode_metadata_dict(value)
    elseif value isa AbstractDict
        return _encode_metadata_dict(Dict(Symbol(k) => v for (k, v) in value))
    elseif value isa AbstractVector
        return "[" * join((_encode_metadata_value(v) for v in value), ",") * "]"
    elseif value isa Tuple
        return "[" * join((_encode_metadata_value(v) for v in value), ",") * "]"
    elseif value isa Symbol
        return "\"" * _escape_metadata_string(String(value)) * "\""
    elseif value isa AbstractString
        return "\"" * _escape_metadata_string(String(value)) * "\""
    elseif value isa Bool
        return value ? "true" : "false"
    elseif value === nothing
        return "null"
    elseif value isa Integer || value isa AbstractFloat
        return string(value)
    else
        return "\"" * _escape_metadata_string(string(value)) * "\""
    end
end

function _encode_metadata_dict(metadata::AbstractDict)
    pairs = String[]
    for (key, value) in metadata
        key_str = "\"" * _escape_metadata_string(String(Symbol(key))) * "\""
        push!(pairs, string(key_str, ":", _encode_metadata_value(value)))
    end
    "{" * join(pairs, ",") * "}"
end

function _escape_metadata_string(str::AbstractString)
    io = IOBuffer()
    for c in str
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        else
            print(io, c)
        end
    end
    String(take!(io))
end

"""
    merge_deltas_simd!(data_a, mask_a, data_b, mask_b, out_data, out_mask)

SIMD-optimized delta merge using AVX2/AVX-512 when available.
"""
function merge_deltas_simd!(
    data_a::Vector{UInt8}, mask_a::Vector{Bool},
    data_b::Vector{UInt8}, mask_b::Vector{Bool},
    out_data::Vector{UInt8}, out_mask::Vector{Bool}
)
    len = min(length(data_a), length(data_b))
    
    if ccall((:cpu_has_avx2, LIBMMSB), Bool, ())
        ccall((:merge_dense_simd_ffi, LIBMMSB), Cvoid,
              (Ptr{UInt8}, Ptr{Bool}, Ptr{UInt8}, Ptr{Bool},
               Ptr{UInt8}, Ptr{Bool}, Csize_t),
              data_a, mask_a, data_b, mask_b, out_data, out_mask, len)
    else
        @inbounds for i in 1:len
            if mask_b[i]
                out_data[i] = data_b[i]
                out_mask[i] = true
            else
                out_data[i] = data_a[i]
                out_mask[i] = mask_a[i]
            end
        end
    end
end

mutable struct _MetadataParser
    data::String
    idx::Int
end

function _decode_metadata(json::String)
    parser = _MetadataParser(json, 1)
    value = _parse_metadata_value(parser)
    value isa Dict || error("Intent metadata must decode to a JSON object")
    Dict(Symbol(k) => v for (k, v) in value)
end

function _parse_metadata_value(parser::_MetadataParser)
    _skip_ws(parser)
    c = _peek(parser)
    c == '{' && return _parse_metadata_object(parser)
    c == '[' && return _parse_metadata_array(parser)
    c == '"' && return _parse_metadata_string(parser)
    if c == '-' || (c >= '0' && c <= '9')
        return _parse_metadata_number(parser)
    end
    if startswith(parser.data[parser.idx:end], "true")
        parser.idx += 4
        return true
    elseif startswith(parser.data[parser.idx:end], "false")
        parser.idx += 5
        return false
    elseif startswith(parser.data[parser.idx:end], "null")
        parser.idx += 4
        return nothing
    else
        error("Invalid intent metadata at position $(parser.idx)")
    end
end

function _parse_metadata_object(parser::_MetadataParser)
    _consume(parser, '{')
    result = Dict{String,Any}()
    _skip_ws(parser)
    if _peek(parser) == '}'
        parser.idx += 1
        return result
    end
    while true
        key = _parse_metadata_string(parser)
        _skip_ws(parser)
        _consume(parser, ':')
        value = _parse_metadata_value(parser)
        result[key] = value
        _skip_ws(parser)
        char = _peek(parser)
        if char == '}'
            parser.idx += 1
            break
        end
        _consume(parser, ',')
    end
    result
end

function _parse_metadata_array(parser::_MetadataParser)
    _consume(parser, '[')
    values = Any[]
    _skip_ws(parser)
    if _peek(parser) == ']'
        parser.idx += 1
        return values
    end
    while true
        push!(values, _parse_metadata_value(parser))
        _skip_ws(parser)
        char = _peek(parser)
        if char == ']'
            parser.idx += 1
            break
        end
        _consume(parser, ',')
    end
    values
end

function _parse_metadata_string(parser::_MetadataParser)
    _consume(parser, '"')
    io = IOBuffer()
    while parser.idx <= lastindex(parser.data)
        c = parser.data[parser.idx]
        parser.idx += 1
        if c == '"'
            break
        elseif c == '\\'
            c2 = parser.data[parser.idx]
            parser.idx += 1
            if c2 == '"' || c2 == '\\' || c2 == '/'
                print(io, c2)
            elseif c2 == 'n'
                print(io, '\n')
            elseif c2 == 'r'
                print(io, '\r')
            elseif c2 == 't'
                print(io, '\t')
            else
                error("Unsupported escape sequence in intent metadata")
            end
        else
            print(io, c)
        end
    end
    String(take!(io))
end

function _parse_metadata_number(parser::_MetadataParser)
    start_idx = parser.idx
    while parser.idx <= lastindex(parser.data)
        c = parser.data[parser.idx]
        if !((c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E')
            break
        end
        parser.idx += 1
    end
    token = parser.data[start_idx:parser.idx-1]
    if occursin('.' , token) || occursin('e', lowercase(token))
        return parse(Float64, token)
    else
        parsed = tryparse(Int, token)
        parsed === nothing && error("Invalid number in intent metadata")
        return parsed
    end
end

function _skip_ws(parser::_MetadataParser)
    while parser.idx <= lastindex(parser.data) && parser.data[parser.idx] in (' ', '\t', '\r', '\n')
        parser.idx += 1
    end
end

function _consume(parser::_MetadataParser, expected::Char)
    actual = _peek(parser)
    actual == expected || error("Expected $(expected) in intent metadata, found $(actual)")
    parser.idx += 1
end

function _peek(parser::_MetadataParser)
    parser.idx <= lastindex(parser.data) || error("Unexpected end of intent metadata")
    parser.data[parser.idx]
end

end # module
