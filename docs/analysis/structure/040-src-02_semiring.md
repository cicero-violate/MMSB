# Structure Group: src/02_semiring

## File: MMSB/src/02_semiring/mod.rs

- Layer(s): 02_semiring
- Language coverage: Rust (4)
- Element types: Module (4)
- Total elements: 4

### Elements

- [Rust | Module] `purity_validator` (line 0, pub)
- [Rust | Module] `semiring_ops` (line 0, pub)
- [Rust | Module] `semiring_types` (line 0, pub)
- [Rust | Module] `standard_semirings` (line 0, pub)

## File: MMSB/src/02_semiring/purity_validator.rs

- Layer(s): 02_semiring
- Language coverage: Rust (9)
- Element types: Function (2), Impl (3), Module (1), Struct (3)
- Total elements: 9

### Elements

- [Rust | Impl] `Default for impl Default for PurityValidator { fn default () -> Self { PurityValidator :: new (3) } } . self_ty` (line 0, priv)
- [Rust | Struct] `PurityFailure` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PurityFailure < I , O > { pub input : I , pub expected : O , pub observed : O ,...`
  - Generics: I, O
- [Rust | Struct] `PurityReport` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct PurityReport < I , O > { pub runs : usize , pub samples : usize , pub failures ...`
  - Generics: I, O
- [Rust | Struct] `PurityValidator` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Copy)] pub struct PurityValidator { runs : usize , }`
- [Rust | Function] `detects_impure_function` (line 0, priv)
  - Signature: `# [test] fn detects_impure_function () { let validator = PurityValidator :: default () ; let inputs = vec ! [1u32 , 2...`
  - Calls: PurityValidator::default, AtomicU32::new, validate_fn, fetch_add
- [Rust | Impl] `impl < I , O > PurityReport < I , O > { pub fn is_pure (& self) -> bool { self . failures . is_empty () } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl PurityValidator { pub fn new (runs : usize) -> Self { Self { runs : runs . max (2) } } pub fn validate_fn < F , I , O > (& self , inputs : & [I] , func : F) -> PurityReport < I , O > where F : Fn (& I) -> O , I : Clone + Debug , O : Clone + PartialEq + Debug , { let baseline : Vec < O > = inputs . iter () . map (| input | func (input)) . collect () ; let mut failures = Vec :: new () ; for run in 1 .. self . runs { for (idx , input) in inputs . iter () . enumerate () { let observed = func (input) ; if observed != baseline [idx] { failures . push (PurityFailure { input : input . clone () , expected : baseline [idx] . clone () , observed , run , }) ; } } } PurityReport { runs : self . runs , samples : inputs . len () , failures , } } pub fn validate_semiring < S > (& self , semiring : & S , samples : & [Vec < S :: Element >]) -> bool where S : Semiring , S :: Element : Clone + Debug + PartialEq , { for values in samples { let baseline_add = fold_add (semiring , values . clone ()) ; let baseline_mul = fold_mul (semiring , values . clone ()) ; for _ in 1 .. self . runs { if fold_add (semiring , values . clone ()) != baseline_add { return false ; } if fold_mul (semiring , values . clone ()) != baseline_mul { return false ; } } } true } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)
- [Rust | Function] `validates_semiring_operations` (line 0, priv)
  - Signature: `# [test] fn validates_semiring_operations () { let validator = PurityValidator :: default () ; let bool_samples = vec...`
  - Calls: PurityValidator::default

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

