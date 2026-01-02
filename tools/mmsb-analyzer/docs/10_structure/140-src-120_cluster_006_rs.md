# Structure Group: src/120_cluster_006.rs

## File: src/120_cluster_006.rs

- Layer(s): 120_cluster_006.rs
- Language coverage: Rust (5)
- Element types: Function (5)
- Total elements: 5

### Elements

- [Rust | Function] `common_root` (line 0, pub)
  - Signature: `pub fn common_root (files : & [PathBuf]) -> Option < PathBuf > { let mut iter = files . iter () ; let first = iter . ...`
  - Calls: iter, collect, components, next, len, collect, components, len, PathBuf::new, take, into_iter, push, as_os_str, Some
- [Rust | Function] `compute_cohesion_score` (line 0, pub)
  - Signature: `pub fn compute_cohesion_score (func : & FunctionInfo , functions : & [FunctionInfo] , outgoing : & HashMap < usize , ...`
  - Calls: unwrap_or_else, cloned, get, clone, unwrap_or_else, cloned, get, clone, crate::cluster_008::layer_adheres, clamp
- [Rust | Function] `layer_prefix_value` (line 0, pub)
  - Signature: `# [doc = " Extracts numeric layer prefix from a layer string (e.g., \"060_file_ordering\" -> 60)"] pub fn layer_prefi...`
  - Calls: chars, String::new, next, is_ascii_digit, push, is_empty, ok, parse
- [Rust | Function] `order_directories` (line 0, pub)
  - Signature: `pub fn order_directories (files : & [PathBuf] , dep_map : & HashMap < PathBuf , Vec < PathBuf > > ,) -> Vec < PathBuf...`
  - Calls: common_root, HashSet::new, map, parent, starts_with, insert, clone, map, parent, collect, into_iter, sort_by, crate::cluster_008::compare_path_components, HashMap::new, enumerate, iter, insert, clone, map, parent, get, map, parent, get, insert, collect, filter_map, enumerate, iter, Some, Vec::with_capacity, len, next, iter, remove, push, clone, clone, insert, len, len, enumerate, iter, push, clone
- [Rust | Function] `strip_numeric_prefix` (line 0, pub(crate))
  - Signature: `pub (crate) fn strip_numeric_prefix (name : & str) -> & str { use once_cell :: sync :: Lazy ; use regex :: Regex ; st...`
  - Calls: Lazy::new, unwrap, Regex::new, unwrap_or, map, and_then, captures, get, as_str

