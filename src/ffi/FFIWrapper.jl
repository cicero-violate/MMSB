module FFIWrapper

using Base: Csize_t

const LIBMMSB = joinpath(@__DIR__, "..", "..", "target", "release", "libmmsb_core.so")
const _LIBMMSB_AVAILABLE = Ref{Union{Nothing,Bool}}(nothing)

const _ERROR_HOOK = Base.RefValue{Function}((::AbstractString) -> nothing)

function register_error_hook(f::Function)
    _ERROR_HOOK[] = f
    nothing
end

function _check_rust_error(context::AbstractString)
    _ERROR_HOOK[](context)
    nothing
end

struct RustPageHandle
    ptr::Ptr{Cvoid}
end

struct RustDeltaHandle
    ptr::Ptr{Cvoid}
end

struct RustAllocatorHandle
    ptr::Ptr{Cvoid}
end

struct RustTLogHandle
    ptr::Ptr{Cvoid}
end

struct RustTLogReaderHandle
    ptr::Ptr{Cvoid}
end

struct RustEpoch
    value::UInt32
end

struct RustTLogSummary
    total_deltas::UInt64
    total_bytes::UInt64
    last_epoch::UInt32
end

struct RustPageInfo
    page_id::UInt64
    size::Csize_t
    location::Int32
    epoch::UInt32
    metadata_ptr::Ptr{UInt8}
    metadata_len::Csize_t
end

struct RustSemiringPairF64
    add::Float64
    mul::Float64
end

struct RustSemiringPairBool
    add::UInt8
    mul::UInt8
end

function rust_artifacts_available()
    cached = _LIBMMSB_AVAILABLE[]
    cached === true && return true
    available = isfile(LIBMMSB)
    _LIBMMSB_AVAILABLE[] = available
    return available
end

function ensure_rust_artifacts()
    rust_artifacts_available() || error("Rust artifacts not built — run `cargo build --release`")
end

_base_handle(::Type{T}) where {T} = T(C_NULL)

function rust_page_read!(handle::RustPageHandle, buffer::Vector{UInt8})
    # T1: SAFETY VALIDATION - Added to prevent NULL pointer crashes
    @assert handle.ptr ≠ C_NULL "Page handle pointer is NULL - page not properly initialized"
    @assert !isempty(buffer) "Buffer must not be empty"
    
    # T5: DEBUG LOGGING - Track FFI boundary crossing
    ptr_addr = UInt(handle.ptr)
    ptr_hex = string(ptr_addr, base=16, pad=16)
    @debug "FFI page read" ptr=ptr_hex buffer_size=length(buffer)
    
    # T2: GC PROTECTION - Prevent Julia from freeing memory during Rust access
    ensure_rust_artifacts()
    GC.@preserve buffer begin
        copied = ccall((:mmsb_page_read, LIBMMSB), Csize_t,
                       (RustPageHandle, Ptr{UInt8}, Csize_t), handle, pointer(buffer), length(buffer))
    end
    _check_rust_error("rust_page_read!")
    @debug "FFI page read completed" copied=Int(copied)
    Int(copied)
end

function rust_page_epoch(handle::RustPageHandle)::UInt32
    ensure_rust_artifacts()
    value = UInt32(ccall((:mmsb_page_epoch, LIBMMSB), UInt32, (RustPageHandle,), handle))
    _check_rust_error("rust_page_epoch")
    value
end

function rust_page_metadata_blob(handle::RustPageHandle)::Vector{UInt8}
    ensure_rust_artifacts()
    len = ccall((:mmsb_page_metadata_size, LIBMMSB), Csize_t, (RustPageHandle,), handle)
    _check_rust_error("rust_page_metadata_size")
    len == 0 && return UInt8[]
    buffer = Vector{UInt8}(undef, len)
    GC.@preserve buffer begin
        copied = ccall((:mmsb_page_metadata_export, LIBMMSB), Csize_t,
                       (RustPageHandle, Ptr{UInt8}, Csize_t), handle, pointer(buffer), len)
    end
    _check_rust_error("rust_page_metadata_export")
    buffer[1:Int(copied)]
end

function rust_page_metadata_import!(handle::RustPageHandle, blob::Vector{UInt8})
    ensure_rust_artifacts()
    GC.@preserve blob begin
        ccall((:mmsb_page_metadata_import, LIBMMSB), Cint,
              (RustPageHandle, Ptr{UInt8}, Csize_t), handle, pointer(blob), length(blob))
    end
    _check_rust_error("rust_page_metadata_import!")
    nothing
end

function rust_page_write_masked!(handle::RustPageHandle, mask::Vector{UInt8}, payload::Vector{UInt8};
                                 is_sparse::Bool=false, epoch::UInt32=0)
    ensure_rust_artifacts()
    GC.@preserve mask payload begin
        ccall((:mmsb_page_write_masked, LIBMMSB), Cint,
              (RustPageHandle, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, UInt8, RustEpoch),
              handle, pointer(mask), length(mask), pointer(payload), length(payload), is_sparse ? UInt8(1) : UInt8(0), RustEpoch(epoch))
    end
    _check_rust_error("rust_page_write_masked!")
    nothing
end

function rust_delta_new(delta_id::UInt64, page_id::UInt64, epoch::UInt32,
                        mask::Vector{UInt8}, payload::Vector{UInt8}, source::Symbol; is_sparse::Bool=false)
    ensure_rust_artifacts()
    src = String(source)
    GC.@preserve mask payload src begin
        handle = ccall((:mmsb_delta_new, LIBMMSB), RustDeltaHandle,
                       (UInt64, UInt64, RustEpoch, Ptr{UInt8}, Csize_t, Ptr{UInt8}, Csize_t, UInt8, Cstring),
                       delta_id, page_id, RustEpoch(epoch), pointer(mask), length(mask), pointer(payload), length(payload), is_sparse ? UInt8(1) : UInt8(0), src)
    end
    _check_rust_error("rust_delta_new")
    handle
end

function rust_delta_free!(handle::RustDeltaHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_delta_free, LIBMMSB), Cvoid, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_free!")
    nothing
end

function rust_delta_apply!(page::RustPageHandle, delta::RustDeltaHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_delta_apply, LIBMMSB), Cint, (RustPageHandle, RustDeltaHandle), page, delta)
    _check_rust_error("rust_delta_apply!")
    nothing
end

# ──────────────────────────────────────────────────────────────
#  NEW CLEAN ALLOCATOR FFI — MATCHES CURRENT RUST CORE
# ──────────────────────────────────────────────────────────────

function rust_allocator_new()::RustAllocatorHandle
    ensure_rust_artifacts()
    handle = ccall((:mmsb_allocator_new, LIBMMSB), RustAllocatorHandle, ())
    _check_rust_error("rust_allocator_new")
    handle
end

function rust_allocator_clear!(handle::RustAllocatorHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_allocator_clear, LIBMMSB), Cvoid, (RustAllocatorHandle,), handle)
    _check_rust_error("rust_allocator_clear!")
    nothing
end

function rust_allocator_free!(handle::RustAllocatorHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_allocator_free, LIBMMSB), Cvoid, (RustAllocatorHandle,), handle)
    _check_rust_error("rust_allocator_free!")
    nothing
end

function rust_allocator_allocate(
    handle::RustAllocatorHandle,
    page_id::UInt64,
    size::Int,
    location::Int32 = 0,
)::RustPageHandle
    ensure_rust_artifacts()
    page = ccall(
        (:mmsb_allocator_allocate, LIBMMSB),
        RustPageHandle,
        (RustAllocatorHandle, UInt64, Csize_t, Int32),
        handle, page_id, size, location,
    )
    _check_rust_error("rust_allocator_allocate")
    page
end

function rust_allocator_release!(handle::RustAllocatorHandle, page_id::UInt64)
    ensure_rust_artifacts()
    ccall(
        (:mmsb_allocator_release, LIBMMSB),
        Cvoid,
        (RustAllocatorHandle, UInt64),
        handle, page_id,
    )
    _check_rust_error("rust_allocator_release!")
    nothing
end

function rust_allocator_get_page(handle::RustAllocatorHandle, page_id::UInt64)::RustPageHandle
    ensure_rust_artifacts()
    page = ccall(
        (:mmsb_allocator_get_page, LIBMMSB),
        RustPageHandle,
        (RustAllocatorHandle, UInt64),
        handle, page_id,
    )
    _check_rust_error("rust_allocator_get_page")
    page
end

function rust_tlog_new(path::AbstractString)
    ensure_rust_artifacts()
    handle = ccall((:mmsb_tlog_new, LIBMMSB), RustTLogHandle, (Cstring,), path)
    _check_rust_error("rust_tlog_new")
    handle
end

function rust_tlog_free!(handle::RustTLogHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_tlog_free, LIBMMSB), Cvoid, (RustTLogHandle,), handle)
    _check_rust_error("rust_tlog_free!")
    nothing
end

function rust_tlog_append!(handle::RustTLogHandle, delta::RustDeltaHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_tlog_append, LIBMMSB), Cint, (RustTLogHandle, RustDeltaHandle), handle, delta)
    _check_rust_error("rust_tlog_append!")
    nothing
end

function rust_tlog_clear_entries!(handle::RustTLogHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_tlog_clear_entries, LIBMMSB), Cvoid, (RustTLogHandle,), handle)
    _check_rust_error("rust_tlog_clear_entries!")
    nothing
end

function rust_tlog_reader_new(path::AbstractString)
    ensure_rust_artifacts()
    handle = ccall((:mmsb_tlog_reader_new, LIBMMSB), RustTLogReaderHandle, (Cstring,), path)
    _check_rust_error("rust_tlog_reader_new")
    handle
end

function rust_tlog_reader_free!(handle::RustTLogReaderHandle)
    ensure_rust_artifacts()
    ccall((:mmsb_tlog_reader_free, LIBMMSB), Cvoid, (RustTLogReaderHandle,), handle)
    _check_rust_error("rust_tlog_reader_free!")
    nothing
end

function rust_tlog_reader_next(handle::RustTLogReaderHandle)::RustDeltaHandle
    ensure_rust_artifacts()
    delta = ccall((:mmsb_tlog_reader_next, LIBMMSB), RustDeltaHandle, (RustTLogReaderHandle,), handle)
    _check_rust_error("rust_tlog_reader_next")
    delta
end

function rust_tlog_summary(path::AbstractString)
    ensure_rust_artifacts()
    summary = Ref{RustTLogSummary}(RustTLogSummary(0, 0, UInt32(0)))
    rc = ccall((:mmsb_tlog_summary, LIBMMSB), Cint, (Cstring, Ref{RustTLogSummary}), path, summary)
    if rc != 0
        _check_rust_error("rust_tlog_summary")
        return nothing
    end
    _check_rust_error("rust_tlog_summary")
    value = summary[]
    (total_deltas = Int(value.total_deltas),
     total_bytes = Int(value.total_bytes),
     last_epoch = UInt32(value.last_epoch))
end

function rust_delta_id(handle::RustDeltaHandle)::UInt64
    ensure_rust_artifacts()
    value = ccall((:mmsb_delta_id, LIBMMSB), UInt64, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_id")
    value
end

function rust_delta_page_id(handle::RustDeltaHandle)::UInt64
    ensure_rust_artifacts()
    value = ccall((:mmsb_delta_page_id, LIBMMSB), UInt64, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_page_id")
    value
end

function rust_delta_epoch(handle::RustDeltaHandle)::UInt32
    ensure_rust_artifacts()
    value = UInt32(ccall((:mmsb_delta_epoch, LIBMMSB), UInt32, (RustDeltaHandle,), handle))
    _check_rust_error("rust_delta_epoch")
    value
end

function rust_delta_is_sparse(handle::RustDeltaHandle)::Bool
    ensure_rust_artifacts()
    value = ccall((:mmsb_delta_is_sparse, LIBMMSB), UInt8, (RustDeltaHandle,), handle) != 0
    _check_rust_error("rust_delta_is_sparse")
    value
end

function rust_delta_timestamp(handle::RustDeltaHandle)::UInt64
    ensure_rust_artifacts()
    value = ccall((:mmsb_delta_timestamp, LIBMMSB), UInt64, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_timestamp")
    value
end

function rust_delta_source(handle::RustDeltaHandle)::String
    ensure_rust_artifacts()
    len = ccall((:mmsb_delta_source_len, LIBMMSB), Csize_t, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_source_len")
    if len == 0
        return ""
    end
    buffer = Vector{UInt8}(undef, len)
    GC.@preserve buffer begin
        copied = ccall((:mmsb_delta_copy_source, LIBMMSB), Csize_t,
                       (RustDeltaHandle, Ptr{UInt8}, Csize_t), handle, pointer(buffer), len)
    end
    _check_rust_error("rust_delta_copy_source")
    String(buffer[1:Int(copied)])
end

function rust_delta_mask(handle::RustDeltaHandle)::Vector{UInt8}
    ensure_rust_artifacts()
    len = ccall((:mmsb_delta_mask_len, LIBMMSB), Csize_t, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_mask_len")
    len == 0 && return UInt8[]
    buf = Vector{UInt8}(undef, len)
    GC.@preserve buf begin
        copied = ccall((:mmsb_delta_copy_mask, LIBMMSB), Csize_t,
                       (RustDeltaHandle, Ptr{UInt8}, Csize_t), handle, pointer(buf), len)
    end
    _check_rust_error("rust_delta_copy_mask")
    buf[1:Int(copied)]
end

function rust_delta_payload(handle::RustDeltaHandle)::Vector{UInt8}
    ensure_rust_artifacts()
    len = ccall((:mmsb_delta_payload_len, LIBMMSB), Csize_t, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_payload_len")
    len == 0 && return UInt8[]
    buf = Vector{UInt8}(undef, len)
    GC.@preserve buf begin
        copied = ccall((:mmsb_delta_copy_payload, LIBMMSB), Csize_t,
                       (RustDeltaHandle, Ptr{UInt8}, Csize_t), handle, pointer(buf), len)
    end
    _check_rust_error("rust_delta_copy_payload")
    buf[1:Int(copied)]
end

function rust_delta_set_intent_metadata!(handle::RustDeltaHandle, metadata::Union{Nothing,AbstractString})
    ensure_rust_artifacts()
    if metadata === nothing
        ccall((:mmsb_delta_set_intent_metadata, LIBMMSB), Cint,
              (RustDeltaHandle, Ptr{UInt8}, Csize_t), handle, Ptr{UInt8}(C_NULL), 0)
        _check_rust_error("rust_delta_set_intent_metadata!")
        return nothing
    end
    bytes = Vector{UInt8}(codeunits(String(metadata)))
    GC.@preserve bytes begin
        ccall((:mmsb_delta_set_intent_metadata, LIBMMSB), Cint,
              (RustDeltaHandle, Ptr{UInt8}, Csize_t), handle, pointer(bytes), length(bytes))
    end
    _check_rust_error("rust_delta_set_intent_metadata!")
    nothing
end

function rust_delta_intent_metadata(handle::RustDeltaHandle)::Union{Nothing,String}
    ensure_rust_artifacts()
    len = ccall((:mmsb_delta_intent_metadata_len, LIBMMSB), Csize_t, (RustDeltaHandle,), handle)
    _check_rust_error("rust_delta_intent_metadata_len")
    len == 0 && return nothing
    buffer = Vector{UInt8}(undef, len)
    GC.@preserve buffer begin
        copied = ccall((:mmsb_delta_copy_intent_metadata, LIBMMSB), Csize_t,
                       (RustDeltaHandle, Ptr{UInt8}, Csize_t), handle, pointer(buffer), len)
    end
    _check_rust_error("rust_delta_copy_intent_metadata")
    String(buffer[1:Int(copied)])
end

function rust_checkpoint_write!(allocator::RustAllocatorHandle, log::RustTLogHandle, path::AbstractString)
    ensure_rust_artifacts()
    ccall((:mmsb_checkpoint_write, LIBMMSB), Cint,
          (RustAllocatorHandle, RustTLogHandle, Cstring),
          allocator, log, path)
    _check_rust_error("rust_checkpoint_write!")
    nothing
end

function rust_checkpoint_load!(allocator::RustAllocatorHandle, log::RustTLogHandle, path::AbstractString)
    ensure_rust_artifacts()
    ccall((:mmsb_checkpoint_load, LIBMMSB), Cint,
          (RustAllocatorHandle, RustTLogHandle, Cstring),
          allocator, log, path)
    _check_rust_error("rust_checkpoint_load!")
    nothing
end

function rust_allocator_page_infos(handle::RustAllocatorHandle)::Vector{RustPageInfo}
    ensure_rust_artifacts()
    count = ccall((:mmsb_allocator_page_count, LIBMMSB), Csize_t, (RustAllocatorHandle,), handle)
    _check_rust_error("rust_allocator_page_count")
    count == 0 && return RustPageInfo[]
    buf = Vector{RustPageInfo}(undef, count)
    GC.@preserve buf begin
        actual = ccall((:mmsb_allocator_list_pages, LIBMMSB), Csize_t,
                       (RustAllocatorHandle, Ptr{RustPageInfo}, Csize_t),
                       handle, pointer(buf), count)
    end
    _check_rust_error("rust_allocator_list_pages")
    buf[1:Int(actual)]
end

function rust_allocator_acquire_page(handle::RustAllocatorHandle, page_id::UInt64)::RustPageHandle
    ensure_rust_artifacts()
    page = ccall((:mmsb_allocator_get_page, LIBMMSB), RustPageHandle,
                 (RustAllocatorHandle, UInt64), handle, page_id)
    _check_rust_error("rust_allocator_get_page")
    page
end

function rust_get_last_error()::Int32
    ensure_rust_artifacts()
    ccall((:mmsb_get_last_error, LIBMMSB), Int32, ())
end

function rust_semiring_tropical_fold_add(values::Vector{Float64})::Float64
    ensure_rust_artifacts()
    result = GC.@preserve values begin
        ptr = isempty(values) ? Ptr{Float64}(C_NULL) : pointer(values)
        ccall((:mmsb_semiring_tropical_fold_add, LIBMMSB), Float64,
              (Ptr{Float64}, Csize_t), ptr, length(values))
    end
    _check_rust_error("rust_semiring_tropical_fold_add")
    result
end

function rust_semiring_tropical_fold_mul(values::Vector{Float64})::Float64
    ensure_rust_artifacts()
    result = GC.@preserve values begin
        ptr = isempty(values) ? Ptr{Float64}(C_NULL) : pointer(values)
        ccall((:mmsb_semiring_tropical_fold_mul, LIBMMSB), Float64,
              (Ptr{Float64}, Csize_t), ptr, length(values))
    end
    _check_rust_error("rust_semiring_tropical_fold_mul")
    result
end

function rust_semiring_tropical_accumulate(left::Float64, right::Float64)
    ensure_rust_artifacts()
    pair = ccall((:mmsb_semiring_tropical_accumulate, LIBMMSB), RustSemiringPairF64,
                 (Float64, Float64), left, right)
    _check_rust_error("rust_semiring_tropical_accumulate")
    return pair.add, pair.mul
end

function rust_semiring_boolean_fold_add(values::Vector{UInt8})::Bool
    ensure_rust_artifacts()
    raw = GC.@preserve values begin
        ptr = isempty(values) ? Ptr{UInt8}(C_NULL) : pointer(values)
        ccall((:mmsb_semiring_boolean_fold_add, LIBMMSB), UInt8,
              (Ptr{UInt8}, Csize_t), ptr, length(values))
    end
    _check_rust_error("rust_semiring_boolean_fold_add")
    raw != 0
end

function rust_semiring_boolean_fold_mul(values::Vector{UInt8})::Bool
    ensure_rust_artifacts()
    raw = GC.@preserve values begin
        ptr = isempty(values) ? Ptr{UInt8}(C_NULL) : pointer(values)
        ccall((:mmsb_semiring_boolean_fold_mul, LIBMMSB), UInt8,
              (Ptr{UInt8}, Csize_t), ptr, length(values))
    end
    _check_rust_error("rust_semiring_boolean_fold_mul")
    raw != 0
end

function rust_semiring_boolean_accumulate(left::Bool, right::Bool)
    ensure_rust_artifacts()
    pair = ccall((:mmsb_semiring_boolean_accumulate, LIBMMSB), RustSemiringPairBool,
                 (UInt8, UInt8), left ? UInt8(1) : UInt8(0), right ? UInt8(1) : UInt8(0))
    _check_rust_error("rust_semiring_boolean_accumulate")
    return pair.add != 0, pair.mul != 0
end

isnull(handle) = handle.ptr == C_NULL

end # module
