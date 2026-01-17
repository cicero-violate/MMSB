module RustErrors

using ..FFIWrapper: register_error_hook, rust_get_last_error
using ..ErrorTypes: GPUMemoryError, SerializationError

export RustFFIError, check_rust_error, translate_error, rethrow_translated,
       MMSB_OK, MMSB_ALLOC_ERROR, MMSB_IO_ERROR, MMSB_SNAPSHOT_ERROR,
       MMSB_CORRUPT_LOG, MMSB_INVALID_HANDLE

const MMSB_OK = Int32(0)
const MMSB_ALLOC_ERROR = Int32(1)
const MMSB_IO_ERROR = Int32(2)
const MMSB_SNAPSHOT_ERROR = Int32(3)
const MMSB_CORRUPT_LOG = Int32(4)
const MMSB_INVALID_HANDLE = Int32(5)

const _ERROR_NAMES = Dict(
    MMSB_OK => "Ok",
    MMSB_ALLOC_ERROR => "AllocError",
    MMSB_IO_ERROR => "IOError",
    MMSB_SNAPSHOT_ERROR => "SnapshotError",
    MMSB_CORRUPT_LOG => "CorruptLog",
    MMSB_INVALID_HANDLE => "InvalidHandle",
)

struct RustFFIError <: Exception
    code::Int32
    context::String
end

function Base.showerror(io::IO, err::RustFFIError)
    name = get(_ERROR_NAMES, err.code, "Unknown")
    print(io, "RustFFIError[$name] (code=$(err.code)) in $(err.context)")
end

function check_rust_error(context::AbstractString)
    code = rust_get_last_error()
    code == MMSB_OK && return
    throw(RustFFIError(code, String(context)))
end

function _default_message(err::RustFFIError)
    name = get(_ERROR_NAMES, err.code, "Unknown")
    "Rust FFI error ($name) in $(err.context)"
end

function translate_error(err::RustFFIError; message::Union{Nothing,String}=nothing)
    msg = something(message, _default_message(err))
    if err.code == MMSB_ALLOC_ERROR
        return GPUMemoryError(msg)
    elseif err.code == MMSB_IO_ERROR || err.code == MMSB_SNAPSHOT_ERROR || err.code == MMSB_CORRUPT_LOG
        return SerializationError(msg)
    elseif err.code == MMSB_INVALID_HANDLE
        return ArgumentError(msg)
    else
        return err
    end
end

rethrow_translated(err::RustFFIError; message::Union{Nothing,String}=nothing) =
    throw(translate_error(err; message=message))

register_error_hook(context -> check_rust_error(context))

end # module
