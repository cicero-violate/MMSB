# Structure Group: src/420_dead_code_entrypoints.rs

## File: src/420_dead_code_entrypoints.rs

- Layer(s): 420_dead_code_entrypoints.rs
- Language coverage: Rust (5)
- Element types: Function (5)
- Total elements: 5

### Elements

- [Rust | Function] `collect_entrypoints` (line 0, pub)
  - Signature: `pub fn collect_entrypoints (elements : & [CodeElement] , policy : Option < & DeadCodePolicy > ,) -> HashSet < String ...`
  - Calls: HashSet::new, insert, clone, insert, clone, treat_public_as_entrypoint, insert, clone
- [Rust | Function] `collect_exports` (line 0, pub)
  - Signature: `pub fn collect_exports (root : & Path) -> HashSet < String > { let mut exports = HashSet :: new () ; let src_dir = ro...`
  - Calls: HashSet::new, join, filter_map, into_iter, WalkDir::new, path, is_file, and_then, extension, to_str, Some, unwrap_or_default, std::fs::read_to_string, syn::parse_file, collect_use_tree_idents, insert, to_string
- [Rust | Function] `collect_use_tree_idents` (line 0, priv)
  - Signature: `fn collect_use_tree_idents (tree : & syn :: UseTree , exports : & mut HashSet < String >) { match tree { syn :: UseTr...`
  - Calls: insert, to_string, insert, to_string, collect_use_tree_idents, collect_use_tree_idents, insert, to_string
- [Rust | Function] `is_public_api` (line 0, pub)
  - Signature: `pub fn is_public_api (symbol : & str , exports : & HashSet < String >) -> bool { exports . contains (symbol) } . sig`
  - Calls: contains
- [Rust | Function] `treat_public_as_entrypoint` (line 0, priv)
  - Signature: `fn treat_public_as_entrypoint (policy : Option < & DeadCodePolicy >) -> bool { policy . map (| p | p . treat_public_a...`
  - Calls: unwrap_or, map

