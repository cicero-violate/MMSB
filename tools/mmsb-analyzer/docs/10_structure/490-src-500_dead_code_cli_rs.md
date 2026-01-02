# Structure Group: src/500_dead_code_cli.rs

## File: src/500_dead_code_cli.rs

- Layer(s): 500_dead_code_cli.rs
- Language coverage: Rust (4)
- Element types: Function (3), Struct (1)
- Total elements: 4

### Elements

- [Rust | Struct] `DeadCodeRunConfig` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct DeadCodeRunConfig { pub root : PathBuf , pub output_dir : PathBuf , pub policy ...`
- [Rust | Function] `is_test_path` (line 0, pub(crate))
  - Signature: `pub (crate) fn is_test_path (path : & Path) -> bool { path . components () . any (| c | { let name = c . as_os_str ()...`
  - Calls: any, components, unwrap_or, to_str, as_os_str
- [Rust | Function] `merge_intent_map` (line 0, pub(crate))
  - Signature: `pub (crate) fn merge_intent_map (base : & mut HashMap < String , Vec < crate :: dead_code_types :: IntentMetadata > >...`
  - Calls: extend, or_default, entry
- [Rust | Function] `reason_for_category` (line 0, pub(crate))
  - Signature: `pub (crate) fn reason_for_category (category : DeadCodeCategory , intent_tag : bool , test_reference : bool) -> Strin...`
  - Calls: to_string, to_string, to_string, to_string, to_string, to_string

