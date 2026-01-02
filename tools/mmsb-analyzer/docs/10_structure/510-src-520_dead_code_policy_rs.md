# Structure Group: src/520_dead_code_policy.rs

## File: src/520_dead_code_policy.rs

- Layer(s): 520_dead_code_policy.rs
- Language coverage: Rust (4)
- Element types: Function (4)
- Total elements: 4

### Elements

- [Rust | Function] `load_policy` (line 0, pub)
  - Signature: `pub fn load_policy (path : & Path) -> std :: io :: Result < DeadCodePolicy > { let contents = std :: fs :: read_to_st...`
  - Calls: std::fs::read_to_string, Ok, parse_policy, unwrap_or, parent
- [Rust | Function] `parse_bool` (line 0, priv)
  - Signature: `fn parse_bool (value : & str) -> Option < bool > { match value . trim () . to_ascii_lowercase () . as_str () { "true"...`
  - Calls: as_str, to_ascii_lowercase, trim, Some, Some
- [Rust | Function] `parse_list` (line 0, priv)
  - Signature: `fn parse_list (value : & str) -> Vec < String > { let mut trimmed = value . trim () . to_string () ; if let Some (str...`
  - Calls: to_string, trim, strip_prefix, to_string, strip_suffix, to_string, collect, filter, map, split, to_string, trim_matches, trim_matches, trim, is_empty
- [Rust | Function] `parse_policy` (line 0, pub)
  - Signature: `pub fn parse_policy (contents : & str , base : & Path) -> DeadCodePolicy { let mut planned_directories = Vec :: new (...`
  - Calls: Vec::new, Vec::new, Vec::new, lines, trim, is_empty, starts_with, starts_with, split_once, trim, trim, collect, map, into_iter, parse_list, join, collect, map, into_iter, parse_list, join, parse_list, unwrap_or, parse_bool

