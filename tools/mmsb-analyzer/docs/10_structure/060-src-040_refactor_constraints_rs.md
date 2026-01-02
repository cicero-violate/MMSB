# Structure Group: src/040_refactor_constraints.rs

## File: src/040_refactor_constraints.rs

- Layer(s): 040_refactor_constraints.rs
- Language coverage: Rust (5)
- Element types: Enum (1), Function (1), Impl (2), Module (1)
- Total elements: 5

### Elements

- [Rust | Impl] `Display for impl fmt :: Display for RefactorConstraint { fn fmt (& self , f : & mut fmt :: Formatter < '_ >) -> fmt :: Result { match self { RefactorConstraint :: NoMove { target , reason , strength } => { write ! (f , "[{}] NoMove: {} (reason: {})" , strength , target , reason) } RefactorConstraint :: PreserveSignature { target , signature , strength } => { write ! (f , "[{}] PreserveSignature: {} -> {}" , strength , target , signature) } RefactorConstraint :: FixedLayer { target , layer , strength } => { write ! (f , "[{}] FixedLayer: {} @ layer {}" , strength , target , layer) } RefactorConstraint :: PreserveOrdering { target , must_come_before , strength } => { write ! (f , "[{}] PreserveOrdering: {} before {:?}" , strength , target , must_come_before) } RefactorConstraint :: MustPreserve { target , facts , strength } => { write ! (f , "[{}] MustPreserve: {} -> {} facts" , strength , target , facts . len ()) } RefactorConstraint :: NoDelete { target , dependents , strength } => { write ! (f , "[{}] NoDelete: {} (dependents: {})" , strength , target , dependents . len ()) } RefactorConstraint :: PreserveDegree { target , in_degree , out_degree , strength } => { write ! (f , "[{}] PreserveDegree: {} (in={}, out={})" , strength , target , in_degree , out_degree) } } } } . self_ty` (line 0, priv)
- [Rust | Enum] `RefactorConstraint` (line 0, pub)
- [Rust | Function] `generate_constraints` (line 0, pub)
  - Signature: `# [doc = " Generate all constraints from an invariant analysis result"] pub fn generate_constraints (analysis : & Inv...`
  - Calls: collect, filter_map, iter
- [Rust | Impl] `impl RefactorConstraint { # [doc = " Get the target of this constraint"] # [allow (dead_code)] pub fn target (& self) -> & str { match self { RefactorConstraint :: NoMove { target , .. } => target , RefactorConstraint :: PreserveSignature { target , .. } => target , RefactorConstraint :: FixedLayer { target , .. } => target , RefactorConstraint :: PreserveOrdering { target , .. } => target , RefactorConstraint :: MustPreserve { target , .. } => target , RefactorConstraint :: NoDelete { target , .. } => target , RefactorConstraint :: PreserveDegree { target , .. } => target , } } # [doc = " Get the strength of the invariant backing this constraint"] # [allow (dead_code)] pub fn strength (& self) -> & InvariantStrength { match self { RefactorConstraint :: NoMove { strength , .. } => strength , RefactorConstraint :: PreserveSignature { strength , .. } => strength , RefactorConstraint :: FixedLayer { strength , .. } => strength , RefactorConstraint :: PreserveOrdering { strength , .. } => strength , RefactorConstraint :: MustPreserve { strength , .. } => strength , RefactorConstraint :: NoDelete { strength , .. } => strength , RefactorConstraint :: PreserveDegree { strength , .. } => strength , } } # [doc = " Check if this constraint should block a refactoring"] # [allow (dead_code)] pub fn is_blocking (& self) -> bool { match self . strength () { InvariantStrength :: Proven => true , InvariantStrength :: Empirical { paths_checked } => * paths_checked >= 10 , InvariantStrength :: Heuristic => false , } } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)

