# Structure Group: src/400_dead_code_intent.rs

## File: src/400_dead_code_intent.rs

- Layer(s): 400_dead_code_intent.rs
- Language coverage: Rust (5)
- Element types: Function (4), Struct (1)
- Total elements: 5

### Elements

- [Rust | Struct] `DeadCodePolicy` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Default)] pub struct DeadCodePolicy { pub planned_directories : Vec < PathBuf > , pub publ...`
- [Rust | Function] `check_planned_directory` (line 0, pub)
  - Signature: `pub fn check_planned_directory (path : & Path , policy : Option < & DeadCodePolicy >) -> bool { let Some (policy) = p...`
  - Calls: starts_with
- [Rust | Function] `collect_symbols` (line 0, pub(crate))
  - Signature: `pub (crate) fn collect_symbols (file : & Path) -> Vec < String > { let contents = std :: fs :: read_to_string (file) ...`
  - Calls: unwrap_or_default, std::fs::read_to_string, syn::parse_file, Vec::new, collect, filter_map, iter
- [Rust | Function] `merge_intent_sources` (line 0, pub)
  - Signature: `pub fn merge_intent_sources (attrs : IntentMap , docs : IntentMap , dir : IntentMap ,) -> IntentMap { let mut merged ...`
  - Calls: IntentMap::new, extend, or_default, entry, extend, or_default, entry, extend, or_default, entry
- [Rust | Function] `planned_directory_intent` (line 0, pub(crate))
  - Signature: `pub (crate) fn planned_directory_intent (file : & Path , policy : Option < & DeadCodePolicy >) -> IntentMap { if ! ch...`
  - Calls: check_planned_directory, IntentMap::new, HashMap::new, collect_symbols, push, or_default, entry

