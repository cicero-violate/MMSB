module PageTypes

using ..FFIWrapper

export Page, PageID, PageLocation, CPU_LOCATION, GPU_LOCATION, UNIFIED_LOCATION,
       is_gpu_page, is_cpu_page, page_size_bytes,
       read_page, initialize!, activate!, deactivate!

@enum PageLocation begin
    CPU_LOCATION = 0
    GPU_LOCATION = 1
    UNIFIED_LOCATION = 2
end

const PageID = UInt64

mutable struct Page
    handle::FFIWrapper.RustPageHandle
    id::PageID
    location::PageLocation
    size::Int
    metadata::Dict{Symbol,Any}
    state::Symbol
end

# THE ONE AND ONLY CONSTRUCTOR
function Page(handle::FFIWrapper.RustPageHandle, id::PageID, location::PageLocation, size::Int;
              metadata::Dict{Symbol,Any} = Dict{Symbol,Any}())
    @assert size > 0 "Page size must be positive, got: $size"
    @assert handle.ptr ≠ C_NULL "Cannot create Page with NULL handle"

    page = Page(handle, id, location, size, Dict{Symbol,Any}(), :allocated)

    _apply_metadata!(page, metadata)
    page
end

# Helper constructor for UInt64 IDs (internal only)
Page(handle::FFIWrapper.RustPageHandle, id::UInt64, location::PageLocation, size::Int; kwargs...) =
    Page(handle, PageID(id), location, size; kwargs...)

is_gpu_page(page::Page) = page.location in (GPU_LOCATION, UNIFIED_LOCATION)
is_cpu_page(page::Page) = page.location == CPU_LOCATION
page_size_bytes(page::Page) = page.size

function initialize!(page::Page)
    @assert page.state == :allocated "Can only initialize allocated pages"
    page.state = :initialized
    page
end

function activate!(page::Page)
    @assert page.state == :initialized "Can only activate initialized pages"
    page.state = :active
    page
end

function deactivate!(page::Page)
    @assert page.state == :active "Can only deactivate active pages"
    page.state = :initialized
    page
end

function read_page(page::Page)::Vector{UInt8}
    @assert page.state ∈ (:initialized, :active) "Cannot read page in state $(page.state)"
    buffer = Vector{UInt8}(undef, page.size)
    GC.@preserve buffer begin
        FFIWrapper.rust_page_read!(page.handle, buffer)
    end
    buffer
end

function _apply_metadata!(page::Page, metadata::Dict{Symbol,Any})
    isempty(metadata) && return page
    blob = _encode_metadata_dict(metadata)
    FFIWrapper.rust_page_metadata_import!(page.handle, blob)
    page.metadata = Dict{Symbol,Any}(metadata)
    page
end

function _encode_metadata_dict(metadata::Dict{Symbol,Any})
    count = UInt32(length(metadata))
    io = IOBuffer()
    write(io, count)
    for (key, value) in metadata
        key_bytes = codeunits(String(key))
        val_bytes = _coerce_metadata_value(value)
        write(io, UInt32(length(key_bytes)))
        write(io, key_bytes)
        write(io, UInt32(length(val_bytes)))
        write(io, val_bytes)
    end
    take!(io)
end

function _coerce_metadata_value(value::Any)
    value isa AbstractVector{UInt8} && return Vector{UInt8}(value)
    value isa AbstractString && return Vector{UInt8}(codeunits(String(value)))
    throw(ArgumentError("metadata values must be Vector{UInt8} or String"))
end

function _decode_metadata_blob(blob::Vector{UInt8})
    isempty(blob) && return Dict{Symbol,Any}()
    io = IOBuffer(blob)
    count = read(io, UInt32)
    dict = Dict{Symbol,Any}()
    for _ in 1:count
        key_len = read(io, UInt32)
        key = Symbol(String(read(io, key_len)))
        val_len = read(io, UInt32)
        value = read(io, val_len)
        dict[key] = value
    end
    dict
end

metadata_from_blob(blob::Vector{UInt8}) = _decode_metadata_blob(blob)
export metadata_from_blob

end # module
