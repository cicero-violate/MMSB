module Semiring

using ..FFIWrapper

export SemiringOps, tropical_semiring, boolean_semiring
export tropical_fold_add, tropical_fold_mul, tropical_accumulate
export boolean_fold_add, boolean_fold_mul, boolean_accumulate

struct SemiringOps{T}
    zero::T
    one::T
    add::Function
    mul::Function
end

function tropical_semiring()
    SemiringOps(
        Inf,
        0.0,
        (a, b) -> min(a, b),
        (a, b) -> a + b,
    )
end

function boolean_semiring()
    SemiringOps(
        false,
        true,
        (a, b) -> a || b,
        (a, b) -> a && b,
    )
end

_FLOAT_BUF(values::AbstractVector{<:Real}) = Vector{Float64}(values)

function _bool_buf(values::AbstractVector{Bool})
    buf = Vector{UInt8}(undef, length(values))
    @inbounds for idx in eachindex(values)
        buf[idx] = values[idx] ? UInt8(1) : UInt8(0)
    end
    buf
end

"""
    tropical_fold_add(values)

Call the Rust `fold_add` implementation for the tropical semiring using `values`.
"""
function tropical_fold_add(values::AbstractVector{<:Real})
    buf = _FLOAT_BUF(values)
    return FFIWrapper.rust_semiring_tropical_fold_add(buf)
end

"""
    tropical_fold_mul(values)
"""
function tropical_fold_mul(values::AbstractVector{<:Real})
    buf = _FLOAT_BUF(values)
    return FFIWrapper.rust_semiring_tropical_fold_mul(buf)
end

"""
    tropical_accumulate(left, right)
"""
function tropical_accumulate(left::Real, right::Real)
    return FFIWrapper.rust_semiring_tropical_accumulate(Float64(left), Float64(right))
end

"""
    boolean_fold_add(values)
"""
function boolean_fold_add(values::AbstractVector{Bool})
    buf = _bool_buf(values)
    return FFIWrapper.rust_semiring_boolean_fold_add(buf)
end

"""
    boolean_fold_mul(values)
"""
function boolean_fold_mul(values::AbstractVector{Bool})
    buf = _bool_buf(values)
    return FFIWrapper.rust_semiring_boolean_fold_mul(buf)
end

"""
    boolean_accumulate(left, right)
"""
function boolean_accumulate(left::Bool, right::Bool)
    return FFIWrapper.rust_semiring_boolean_accumulate(left, right)
end

end # module
