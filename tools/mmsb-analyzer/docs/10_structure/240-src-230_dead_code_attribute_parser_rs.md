# Structure Group: src/230_dead_code_attribute_parser.rs

## File: src/230_dead_code_attribute_parser.rs

- Layer(s): 230_dead_code_attribute_parser.rs
- Language coverage: Rust (11)
- Element types: Function (9), Module (2)
- Total elements: 11

### Elements

- [Rust | Function] `collect_latent_attrs` (line 0, priv)
  - Signature: `fn collect_latent_attrs (attrs : & [Attribute]) -> Vec < IntentMetadata > { let mut markers = Vec :: new () ; for att...`
  - Calls: Vec::new, is_ident, path, parse_nested_meta, is_ident, is_ident, is_ident, is_ident, is_ident, is_ident, value, parse, Some, value, is_ident, value, parse, marker_from_str, value, Ok, parse_args, Some, value, push
- [Rust | Function] `detect_intent_signals` (line 0, pub)
  - Signature: `pub fn detect_intent_signals (file : & Path , policy : Option < & DeadCodePolicy >) -> IntentMap { let attrs = parse_...`
  - Calls: parse_mmsb_latent_attr, scan_doc_comments, merge_doc_intent, planned_directory_intent, merge_intent_sources
- [Rust | Function] `detect_test_modules` (line 0, pub)
  - Signature: `pub fn detect_test_modules (file : & Path) -> HashSet < String > { let contents = std :: fs :: read_to_string (file) ...`
  - Calls: unwrap_or_default, std::fs::read_to_string, syn::parse_file, HashSet::new, HashSet::new, is_cfg_test_item, insert, to_string
- [Rust | Function] `detect_test_symbols` (line 0, pub)
  - Signature: `pub fn detect_test_symbols (file : & Path) -> HashSet < String > { let contents = std :: fs :: read_to_string (file) ...`
  - Calls: unwrap_or_default, std::fs::read_to_string, syn::parse_file, HashSet::new, HashSet::new, has_test_attr, insert, to_string, is_cfg_test_item, insert, to_string, insert, to_string
- [Rust | Function] `extract_attribute_value` (line 0, pub)
  - Signature: `pub fn extract_attribute_value (attr : & Attribute , key : & str) -> Option < String > { let mut found = None ; let _...`
  - Calls: parse_nested_meta, is_ident, value, parse, Some, value, Ok
- [Rust | Function] `marker_from_str` (line 0, priv)
  - Signature: `fn marker_from_str (raw : & str) -> IntentMarker { match raw . to_ascii_lowercase () . as_str () { "planned" => Inten...`
  - Calls: as_str, to_ascii_lowercase
- [Rust | Module] `moved_is_cfg_test_item` (line 0, priv)
- [Rust | Module] `moved_scan_doc_comments` (line 0, priv)
- [Rust | Function] `parse_mmsb_latent_attr` (line 0, pub)
  - Signature: `pub fn parse_mmsb_latent_attr (path : & Path) -> HashMap < String , Vec < IntentMetadata > > { let contents = std :: ...`
  - Calls: unwrap_or_default, std::fs::read_to_string, syn::parse_file, HashMap::new, HashMap::new, item_name, collect_latent_attrs, item_attrs, is_empty, extend, or_default, entry
- [Rust | Function] `scan_file_attributes` (line 0, pub)
  - Signature: `pub fn scan_file_attributes (path : & Path) -> Vec < IntentTag > { let contents = std :: fs :: read_to_string (path) ...`
  - Calls: unwrap_or_default, std::fs::read_to_string, syn::parse_file, Vec::new, Vec::new, item_name, collect_latent_attrs, item_attrs, push, clone, to_path_buf, clone
- [Rust | Function] `scan_intent_tags` (line 0, pub)
  - Signature: `pub fn scan_intent_tags (file : & Path , policy : Option < & DeadCodePolicy >) -> Vec < IntentTag > { let mut tags = ...`
  - Calls: Vec::new, parse_mmsb_latent_attr, push, clone, to_path_buf, clone, scan_doc_comments, push, clone, to_path_buf, check_planned_directory, collect_symbols, push, to_path_buf

