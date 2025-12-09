module Semiring

export SemiringOps, tropical_semiring, boolean_semiring

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

end # module
