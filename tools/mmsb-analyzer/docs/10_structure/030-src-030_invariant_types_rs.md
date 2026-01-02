# Structure Group: src/030_invariant_types.rs

## File: src/030_invariant_types.rs

- Layer(s): 030_invariant_types.rs
- Language coverage: Rust (23)
- Element types: Enum (7), Impl (8), Module (1), Struct (7)
- Total elements: 23

### Elements

- [Rust | Struct] `Confidence` (line 0, pub)
  - Signature: `# [doc = " Confidence score for an invariant [0.0, 1.0]"] # [derive (Debug , Clone , Copy , PartialEq , PartialOrd , ...`
- [Rust | Impl] `Default for impl Default for InvariantAnalysisResult { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Enum] `DeltaInvariant` (line 0, pub)
- [Rust | Impl] `Display for impl fmt :: Display for InvariantKind { fn fmt (& self , f : & mut fmt :: Formatter < '_ >) -> fmt :: Result { match self { InvariantKind :: Structural (s) => write ! (f , "Structural: {:?}" , s) , InvariantKind :: Semantic (s) => write ! (f , "Semantic: {:?}" , s) , InvariantKind :: Delta (d) => write ! (f , "Delta: {:?}" , d) , InvariantKind :: PathIntersection (p) => { write ! (f , "PathIntersection: {} facts across {} paths" , p . facts . len () , p . paths_analyzed) } } } } . self_ty` (line 0, priv)
- [Rust | Impl] `Display for impl fmt :: Display for InvariantStrength { fn fmt (& self , f : & mut fmt :: Formatter < '_ >) -> fmt :: Result { match self { InvariantStrength :: Proven => write ! (f , "PROVEN") , InvariantStrength :: Empirical { paths_checked } => { write ! (f , "EMPIRICAL (paths: {})" , paths_checked) } InvariantStrength :: Heuristic => write ! (f , "HEURISTIC (low confidence)") , } } } . self_ty` (line 0, priv)
- [Rust | Impl] `Display for impl fmt :: Display for ViolationSeverity { fn fmt (& self , f : & mut fmt :: Formatter < '_ >) -> fmt :: Result { match self { ViolationSeverity :: Critical => write ! (f , "CRITICAL") , ViolationSeverity :: High => write ! (f , "HIGH") , ViolationSeverity :: Medium => write ! (f , "MEDIUM") , ViolationSeverity :: Low => write ! (f , "LOW") , } } } . self_ty` (line 0, priv)
- [Rust | Struct] `Invariant` (line 0, pub)
  - Signature: `# [doc = " A detected invariant with metadata"] # [derive (Debug , Clone , Serialize , Deserialize)] pub struct Invar...`
- [Rust | Struct] `InvariantAnalysisResult` (line 0, pub)
  - Signature: `# [doc = " Complete invariant analysis result"] # [derive (Debug , Clone , Serialize , Deserialize)] pub struct Invar...`
- [Rust | Enum] `InvariantKind` (line 0, pub)
- [Rust | Struct] `InvariantStats` (line 0, pub)
  - Signature: `# [doc = " Statistics about detected invariants"] # [derive (Debug , Clone , Default , Serialize , Deserialize)] pub ...`
- [Rust | Enum] `InvariantStrength` (line 0, pub)
- [Rust | Struct] `InvariantViolation` (line 0, pub)
  - Signature: `# [doc = " A violation of a detected invariant"] # [derive (Debug , Clone , Serialize , Deserialize)] pub struct Inva...`
- [Rust | Struct] `LayerInfo` (line 0, pub)
  - Signature: `# [doc = " Layer information inferred from call graph"] # [derive (Debug , Clone , Serialize , Deserialize)] pub stru...`
- [Rust | Enum] `MonotonicDirection` (line 0, pub)
- [Rust | Struct] `PathIntersectionInvariant` (line 0, pub)
  - Signature: `# [doc = " Path-intersection invariant - facts that hold on ALL execution paths"] # [derive (Debug , Clone , PartialE...`
- [Rust | Enum] `SemanticInvariant` (line 0, pub)
- [Rust | Enum] `StructuralInvariant` (line 0, pub)
- [Rust | Enum] `ViolationSeverity` (line 0, pub)
- [Rust | Impl] `impl Confidence { pub fn from_strength (strength : & InvariantStrength) -> Self { match strength { InvariantStrength :: Proven => Confidence (1.0) , InvariantStrength :: Empirical { paths_checked } => { if * paths_checked > 100 { Confidence (0.9) } else if * paths_checked > 10 { Confidence (0.7) } else { Confidence (0.5) } } InvariantStrength :: Heuristic => Confidence (0.3) , } } pub fn value (& self) -> f64 { self . 0 } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl Invariant { # [doc = " Create a new invariant"] pub fn new (target : String , file_path : String , kind : InvariantKind , strength : InvariantStrength , description : String ,) -> Self { let confidence = Confidence :: from_strength (& strength) ; Self { target , file_path , kind , strength , confidence , description , evidence : Vec :: new () , } } # [doc = " Add evidence supporting this invariant"] pub fn add_evidence (& mut self , evidence : String) { self . evidence . push (evidence) ; } # [doc = " Check if this invariant should block refactorings"] pub fn is_blocking (& self) -> bool { match self . strength { InvariantStrength :: Proven => true , InvariantStrength :: Empirical { paths_checked } => paths_checked >= 10 , InvariantStrength :: Heuristic => false , } } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl InvariantAnalysisResult { pub fn new () -> Self { Self { invariants : Vec :: new () , violations : Vec :: new () , layer_assignments : HashMap :: new () , stats : InvariantStats :: default () , } } # [doc = " Add an invariant to the result"] pub fn add_invariant (& mut self , invariant : Invariant) { match invariant . strength { InvariantStrength :: Proven => self . stats . proven_count += 1 , InvariantStrength :: Empirical { .. } => self . stats . empirical_count += 1 , InvariantStrength :: Heuristic => self . stats . heuristic_count += 1 , } match & invariant . kind { InvariantKind :: Structural (_) => self . stats . structural_count += 1 , InvariantKind :: Semantic (_) => self . stats . semantic_count += 1 , InvariantKind :: Delta (_) => self . stats . delta_count += 1 , InvariantKind :: PathIntersection (_) => self . stats . path_intersection_count += 1 , } self . invariants . push (invariant) ; } # [doc = " Add a violation to the result"] pub fn add_violation (& mut self , violation : InvariantViolation) { self . stats . violation_count += 1 ; self . violations . push (violation) ; } # [doc = " Get all blocking invariants for a specific target"] # [allow (dead_code)] pub fn get_blocking_invariants (& self , target : & str) -> Vec < & Invariant > { self . invariants . iter () . filter (| inv | inv . target == target && inv . is_blocking ()) . collect () } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl InvariantStats { pub fn update_totals (& mut self) { self . total_count = self . proven_count + self . empirical_count + self . heuristic_count ; } # [doc = " Get percentage of proven invariants"] pub fn proven_percentage (& self) -> f64 { if self . total_count == 0 { 0.0 } else { (self . proven_count as f64 / self . total_count as f64) * 100.0 } } # [doc = " Get percentage of heuristics"] pub fn heuristic_percentage (& self) -> f64 { if self . total_count == 0 { 0.0 } else { (self . heuristic_count as f64 / self . total_count as f64) * 100.0 } } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)

