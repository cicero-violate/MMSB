# Structure Group: src/211_dead_code_doc_comment_scanner.rs

## File: src/211_dead_code_doc_comment_scanner.rs

- Layer(s): 211_dead_code_doc_comment_scanner.rs
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `scan_doc_comments` (line 0, pub)
  - Signature: `pub fn scan_doc_comments (file : & Path) -> HashMap < String , Vec < IntentMarker > > { let contents = std :: fs :: r...`
  - Calls: unwrap_or_default, std::fs::read_to_string, syn::parse_file, HashMap::new, HashMap::new, item_name, extract_doc_markers, item_attrs, is_empty, extend, or_default, entry

