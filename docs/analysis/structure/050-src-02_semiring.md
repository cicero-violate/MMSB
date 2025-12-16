# Structure Group: src/02_semiring

## File: MMSB/src/02_semiring/DeltaRouter.jl

- Layer(s): 02_semiring
- Language coverage: Julia (7)
- Element types: Function (6), Module (1)
- Total elements: 7

### Elements

- [Julia | Module] `DeltaRouter` (line 8, pub)
- [Julia | Function] `route_delta!` (line 29, pub)
  - Signature: `route_delta!(state::MMSBState, delta::Delta; propagate::Bool`
- [Julia | Function] `create_delta` (line 47, pub)
  - Signature: `create_delta(state::MMSBState, page_id::PageID, mask::AbstractVector{Bool}, data::AbstractVector{UInt8}; source::Symbol`
- [Julia | Function] `length` (line 58, pub)
  - Signature: `length(mask_bytes)`
  - Calls: InvalidDeltaError, length, throw
- [Julia | Function] `batch_route_deltas!` (line 71, pub)
  - Signature: `batch_route_deltas!(state::MMSBState, deltas::Vector{Delta})`
  - Calls: collect, get!, get_page, isempty, propagate_change!, push!, route_delta!, sort
- [Julia | Function] `propagate_change!` (line 89, pub)
  - Signature: `propagate_change!(state::MMSBState, changed_page_id::PageID)`
  - Calls: engine.propagate_change!, getfield, parentmodule
- [Julia | Function] `propagate_change!` (line 94, pub)
  - Signature: `propagate_change!(state::MMSBState, changed_pages::AbstractVector{PageID})`
  - Calls: engine.propagate_change!, getfield, parentmodule

## File: MMSB/src/02_semiring/Semiring.jl

- Layer(s): 02_semiring
- Language coverage: Julia (12)
- Element types: Function (10), Module (1), Struct (1)
- Total elements: 12

### Elements

- [Julia | Module] `Semiring` (line 1, pub)
- [Julia | Struct] `SemiringOps{T}` (line 9, pub)
  - Signature: `struct SemiringOps{T}`
- [Julia | Function] `tropical_semiring` (line 16, pub)
  - Signature: `tropical_semiring()`
  - Calls: SemiringOps, min
- [Julia | Function] `boolean_semiring` (line 25, pub)
  - Signature: `boolean_semiring()`
  - Calls: SemiringOps
- [Julia | Function] `_FLOAT_BUF` (line 34, pub)
  - Signature: `_FLOAT_BUF(values::AbstractVector{<:Real})`
- [Julia | Function] `_bool_buf` (line 36, pub)
  - Signature: `_bool_buf(values::AbstractVector{Bool})`
  - Calls: UInt8, eachindex, length
- [Julia | Function] `tropical_fold_add` (line 49, pub)
  - Signature: `tropical_fold_add(values::AbstractVector{<:Real})`
  - Calls: FFIWrapper.rust_semiring_tropical_fold_add, _FLOAT_BUF
- [Julia | Function] `tropical_fold_mul` (line 57, pub)
  - Signature: `tropical_fold_mul(values::AbstractVector{<:Real})`
  - Calls: FFIWrapper.rust_semiring_tropical_fold_mul, _FLOAT_BUF
- [Julia | Function] `tropical_accumulate` (line 65, pub)
  - Signature: `tropical_accumulate(left::Real, right::Real)`
  - Calls: FFIWrapper.rust_semiring_tropical_accumulate, Float64
- [Julia | Function] `boolean_fold_add` (line 72, pub)
  - Signature: `boolean_fold_add(values::AbstractVector{Bool})`
  - Calls: FFIWrapper.rust_semiring_boolean_fold_add, _bool_buf
- [Julia | Function] `boolean_fold_mul` (line 80, pub)
  - Signature: `boolean_fold_mul(values::AbstractVector{Bool})`
  - Calls: FFIWrapper.rust_semiring_boolean_fold_mul, _bool_buf
- [Julia | Function] `boolean_accumulate` (line 88, pub)
  - Signature: `boolean_accumulate(left::Bool, right::Bool)`
  - Calls: FFIWrapper.rust_semiring_boolean_accumulate

## File: MMSB/src/02_semiring/SemiringConfig.jl

- Layer(s): 02_semiring
- Language coverage: Julia (3)
- Element types: Function (1), Module (1), Struct (1)
- Total elements: 3

### Elements

- [Julia | Module] `SemiringConfig` (line 1, pub)
- [Julia | Struct] `SemiringConfigOptions` (line 5, pub)
  - Signature: `struct SemiringConfigOptions`
- [Julia | Function] `build_semiring` (line 9, pub)
  - Signature: `build_semiring(config::SemiringConfigOptions)`
  - Calls: error, return

## File: MMSB/src/02_semiring/mod.rs

- Layer(s): 02_semiring
- Language coverage: Rust (3)
- Element types: Module (3)
- Total elements: 3

### Elements

- [Rust | Module] `semiring_ops` (line 0, pub)
- [Rust | Module] `semiring_types` (line 0, pub)
- [Rust | Module] `standard_semirings` (line 0, pub)

## File: MMSB/src/02_semiring/semiring_ops.rs

- Layer(s): 02_semiring
- Language coverage: Rust (3)
- Element types: Function (3)
- Total elements: 3

### Elements

- [Rust | Function] `accumulate` (line 0, pub)
  - Signature: `# [doc = " Convenience helper for combining two elements with both operations."] pub fn accumulate < S : Semiring > (...`
  - Generics: S
  - Calls: add, mul
- [Rust | Function] `fold_add` (line 0, pub)
  - Signature: `# [doc = " Apply semiring addition across an iterator of elements."] pub fn fold_add < S : Semiring > (semiring : & S...`
  - Generics: S
  - Calls: fold, into_iter, zero, add
- [Rust | Function] `fold_mul` (line 0, pub)
  - Signature: `# [doc = " Apply semiring multiplication across an iterator of elements."] pub fn fold_mul < S : Semiring > (semiring...`
  - Generics: S
  - Calls: fold, into_iter, one, mul

## File: MMSB/src/02_semiring/semiring_types.rs

- Layer(s): 02_semiring
- Language coverage: Rust (1)
- Element types: Trait (1)
- Total elements: 1

### Elements

- [Rust | Trait] `Semiring` (line 0, pub)

## File: MMSB/src/02_semiring/standard_semirings.rs

- Layer(s): 02_semiring
- Language coverage: Rust (4)
- Element types: Impl (2), Struct (2)
- Total elements: 4

### Elements

- [Rust | Struct] `BooleanSemiring` (line 0, pub)
  - Signature: `# [derive (Clone , Copy , Debug , Default)] pub struct BooleanSemiring ;`
- [Rust | Impl] `Semiring for impl Semiring for BooleanSemiring { type Element = bool ; fn zero (& self) -> Self :: Element { false } fn one (& self) -> Self :: Element { true } fn add (& self , a : & Self :: Element , b : & Self :: Element) -> Self :: Element { * a || * b } fn mul (& self , a : & Self :: Element , b : & Self :: Element) -> Self :: Element { * a && * b } } . self_ty` (line 0, priv)
- [Rust | Impl] `Semiring for impl Semiring for TropicalSemiring { type Element = f64 ; fn zero (& self) -> Self :: Element { f64 :: INFINITY } fn one (& self) -> Self :: Element { 0.0 } fn add (& self , a : & Self :: Element , b : & Self :: Element) -> Self :: Element { a . min (* b) } fn mul (& self , a : & Self :: Element , b : & Self :: Element) -> Self :: Element { a + b } } . self_ty` (line 0, priv)
- [Rust | Struct] `TropicalSemiring` (line 0, pub)
  - Signature: `# [derive (Clone , Copy , Debug , Default)] pub struct TropicalSemiring ;`

