# src/01_types/Errors.jl
"""
ErrorTypes - Custom MMSB error hierarchy

Defines structured exceptions for key failure modes so callers
receive actionable diagnostics instead of generic `error` messages.
"""
module ErrorTypes

export MMSBError, PageNotFoundError, InvalidDeltaError, GPUMemoryError,
       SerializationError, GraphCycleError, UnsupportedLocationError

abstract type MMSBError <: Exception end

struct PageNotFoundError <: MMSBError
    page_id::UInt64
    context::String
end

struct InvalidDeltaError <: MMSBError
    reason::String
    page_id::Union{UInt64, Nothing}
end

struct GPUMemoryError <: MMSBError
    message::String
end

struct SerializationError <: MMSBError
    message::String
end

struct GraphCycleError <: MMSBError
    parent::UInt64
    child::UInt64
end

struct UnsupportedLocationError <: MMSBError
    location::String
    context::String
end

Base.showerror(io::IO, err::PageNotFoundError) =
    print(io, "PageNotFoundError: page ", err.page_id, " not found in ", err.context)

Base.showerror(io::IO, err::InvalidDeltaError) = begin
    pid = isnothing(err.page_id) ? "unknown page" : "page $(err.page_id)"
    print(io, "InvalidDeltaError ($pid): ", err.reason)
end

Base.showerror(io::IO, err::GPUMemoryError) =
    print(io, "GPUMemoryError: ", err.message)

Base.showerror(io::IO, err::SerializationError) =
    print(io, "SerializationError: ", err.message)

Base.showerror(io::IO, err::GraphCycleError) =
    print(io, "GraphCycleError: cycle introduced between ", err.parent, " → ", err.child)

Base.showerror(io::IO, err::UnsupportedLocationError) =
    print(io, "UnsupportedLocationError: location ", err.location, " in ", err.context)

end # module ErrorTypes
