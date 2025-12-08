module PageTypes

using ..FFIWrapper

export Page, PageID, PageLocation, CPU_LOCATION, GPU_LOCATION, UNIFIED_LOCATION,
       is_gpu_page, is_cpu_page, page_size_bytes, serialize_page, deserialize_page,
       read_page

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
    state::Symbol  # T3: Lifecycle state tracking
end

function Page(handle::FFIWrapper.RustPageHandle, id::PageID, location::PageLocation, size::Int;
              metadata::Dict{Symbol,Any} = Dict{Symbol,Any}())
    # Validate inputs
    @assert size > 0 "Page size must be positive, got: $size"
    @assert handle.ptr ≠ C_NULL "Cannot create Page with NULL handle from allocator"

    # Create the page object
    page = Page(handle, id, location, size, Dict{Symbol,Any}(), :allocated)

    # FINALIZER: Release via the global allocator, NOT rust_page_free!
    finalizer(page) do p
        try
            FFIWrapper.rust_allocator_release!(PageAllocator._allocator_handle(), UInt64(p.id))
        catch
            # Silent during shutdown — common in Julia finalizers
        end
    end

    # Apply user metadata if provided
    _apply_metadata!(page, metadata)
    page
end

# External constructor: only for internal use by PageAllocator
# Do NOT expose this publicly — it's only called after successful allocation
function Page(handle::FFIWrapper.RustPageHandle, id::PageID, location::PageLocation, size::Int;
              metadata::Dict{Symbol,Any}=Dict{Symbol,Any}())
    # T3: Validate construction parameters
    @assert size > 0 "Page size must be positive, got: $size"
    @assert handle.ptr ≠ C_NULL "Cannot create Page with NULL handle"
    
    page = Page(handle, id, location, size, Dict{Symbol,Any}(), :allocated)
    finalizer(p -> FFIWrapper.rust_page_free!(p.handle), page)
    _apply_metadata!(page, metadata)
    page
end

is_gpu_page(page::Page) = page.location in (GPU_LOCATION, UNIFIED_LOCATION)
is_cpu_page(page::Page) = page.location == CPU_LOCATION
page_size_bytes(page::Page) = page.size

# T3: State transition functions
# State transitions: :allocated → :initialized → :active

function initialize!(page::Page)
    @assert page.state == :allocated "Can only initialize allocated pages, current state: $(page.state)"
    @assert page.handle.ptr ≠ C_NULL "Cannot initialize page with NULL handle"
    page.state = :initialized
    return page
end

function activate!(page::Page)
    @assert page.state == :initialized "Can only activate initialized pages, current state: $(page.state)"
    @assert page.handle.ptr ≠ C_NULL "Cannot activate page with NULL handle"
    page.state = :active
    return page
end

function deactivate!(page::Page)
    @assert page.state == :active "Can only deactivate active pages, current state: $(page.state)"
    page.state = :initialized
    return page
end

export initialize!, activate!, deactivate!

serialize_page(::Page) = error("Use Rust checkpoint APIs for page serialization")
deserialize_page(::Vector{UInt8}) = error("Use Rust checkpoint APIs for page deserialization")

function read_page(page::Page)::Vector{UInt8}
    # T4: STATE PRECONDITION - Only read initialized or active pages
    @assert page.state ∈ (:initialized, :active) "Cannot read page in state $(page.state)"
    @assert page.handle.ptr ≠ C_NULL "Page handle is NULL"
    
    page.handle.ptr == C_NULL && error("Attempted to read freed page $(page.id)")
    buffer = Vector{UInt8}(undef, page.size)
    GC.@preserve page buffer begin
        FFIWrapper.rust_page_read!(page.handle, buffer)
    end
    return buffer
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
        key_str = String(key)
        key_bytes = Vector{UInt8}(codeunits(key_str))
        val_bytes = _coerce_metadata_value(value)
        write(io, UInt32(length(key_bytes)))
        write(io, key_bytes)
        write(io, UInt32(length(val_bytes)))
        write(io, val_bytes)
    end
    take!(io)
end

function _coerce_metadata_value(value::Any)
    if isa(value, AbstractVector{UInt8})
        return Vector{UInt8}(value)
    elseif isa(value, AbstractString)
        return Vector{UInt8}(codeunits(String(value)))
    else
        throw(ArgumentError("metadata values must be Vector{UInt8} or String"))
    end
end

function _decode_metadata_blob(blob::Vector{UInt8})
    io = IOBuffer(blob)
    count = read(io, UInt32)
    dict = Dict{Symbol,Any}()
    for _ in 1:count
        key_len = read(io, UInt32)
        key_bytes = Vector{UInt8}(undef, key_len)
        read!(io, key_bytes)
        val_len = read(io, UInt32)
        value = Vector{UInt8}(undef, val_len)
        read!(io, value)
        dict[Symbol(String(key_bytes))] = value
    end
    dict
end

metadata_from_blob(blob::Vector{UInt8}) = isempty(blob) ? Dict{Symbol,Any}() : _decode_metadata_blob(blob)

export metadata_from_blob

end # module
