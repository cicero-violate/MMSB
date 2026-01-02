# Structure Group: src/560_confidence_scorer.rs

## File: src/560_confidence_scorer.rs

- Layer(s): 560_confidence_scorer.rs
- Language coverage: Rust (2)
- Element types: Function (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `PredictionContext` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Default)] pub struct PredictionContext { pub has_test_coverage : bool , }`
- [Rust | Function] `compute_confidence` (line 0, pub)
  - Signature: `pub fn compute_confidence (prediction : & ViolationPrediction , context : & PredictionContext ,) -> f64 { let base : ...`
  - Calls: min

