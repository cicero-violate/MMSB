# Structure Group: src/200_action_validator.rs

## File: src/200_action_validator.rs

- Layer(s): 200_action_validator.rs
- Language coverage: Rust (6)
- Element types: Enum (2), Function (2), Module (1), Struct (1)
- Total elements: 6

### Elements

- [Rust | Enum] `AgentAction` (line 0, pub)
- [Rust | Struct] `ConstraintViolation` (line 0, pub)
  - Signature: `# [doc = " Constraint violation with explanation"] # [derive (Debug , Clone , Serialize , Deserialize)] pub struct Co...`
- [Rust | Enum] `ViolationSeverity` (line 0, pub)
- [Rust | Function] `extract_layer` (line 0, priv)
  - Signature: `# [doc = " Extract layer number from file path (e.g., \"src/040_test.rs\" -> Some(40))"] fn extract_layer (path : & P...`
  - Calls: and_then, and_then, file_name, to_str, collect, split, is_empty, ok, parse
- [Rust | Module] `tests` (line 0, priv)
- [Rust | Function] `validate_action` (line 0, pub)
  - Signature: `# [doc = " Validate action against all constraints"] pub fn validate_action (action : & AgentAction , constraints : &...`
  - Calls: Vec::new, enumerate, iter, push, extract_layer, extract_layer, push, push, contains, push, is_empty, Ok, Err

