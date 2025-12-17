# Structure Group: src/02_semiring

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

