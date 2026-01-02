# Structure Group: src/430_dead_code_classifier.rs

## File: src/430_dead_code_classifier.rs

- Layer(s): 430_dead_code_classifier.rs
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `is_reachable` (line 0, pub)
  - Signature: `pub fn is_reachable (symbol : & str , call_graph : & CallGraph , entrypoints : & HashSet < String >) -> bool { if ent...`
  - Calls: is_empty, crate::dead_code_call_graph::compute_reachability, contains

