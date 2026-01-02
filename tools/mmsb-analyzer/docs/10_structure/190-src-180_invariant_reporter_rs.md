# Structure Group: src/180_invariant_reporter.rs

## File: src/180_invariant_reporter.rs

- Layer(s): 180_invariant_reporter.rs
- Language coverage: Rust (3)
- Element types: Function (2), Module (1)
- Total elements: 3

### Elements

- [Rust | Function] `export_constraints_json` (line 0, pub)
  - Signature: `# [doc = " Export refactoring constraints to JSON"] pub fn export_constraints_json (constraints : & [RefactorConstrai...`
  - Calls: join, fs::create_dir_all, join, map_err, serde_json::to_string_pretty, std::io::Error::new, fs::write, Ok
- [Rust | Function] `export_json` (line 0, pub)
  - Signature: `# [doc = " Export invariants to JSON for agent consumption"] pub fn export_json (result : & InvariantAnalysisResult ,...`
  - Calls: join, map_err, serde_json::to_string_pretty, std::io::Error::new, fs::write, Ok
- [Rust | Module] `tests` (line 0, priv)

