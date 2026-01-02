# Structure Group: src/440_dead_code_confidence.rs

## File: src/440_dead_code_confidence.rs

- Layer(s): 440_dead_code_confidence.rs
- Language coverage: Rust (2)
- Element types: Function (1), Struct (1)
- Total elements: 2

### Elements

- [Rust | Struct] `Evidence` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Default)] pub struct Evidence { pub intent_tag : bool , pub test_reference : bool , pub ca...`
- [Rust | Function] `assign_confidence` (line 0, pub)
  - Signature: `pub fn assign_confidence (item : & DeadCodeItem , evidence : & Evidence) -> ConfidenceLevel { if evidence . intent_ta...`

