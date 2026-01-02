# Structure Group: src/380_dead_code_doc_comment_parser.rs

## File: src/380_dead_code_doc_comment_parser.rs

- Layer(s): 380_dead_code_doc_comment_parser.rs
- Language coverage: Rust (5)
- Element types: Function (5)
- Total elements: 5

### Elements

- [Rust | Function] `detect_latent_markers` (line 0, pub)
  - Signature: `pub fn detect_latent_markers (comment : & str) -> Option < IntentMarker > { IntentMarker :: from_comment (comment) } ...`
  - Calls: IntentMarker::from_comment
- [Rust | Function] `extract_doc_markers` (line 0, pub(crate))
  - Signature: `pub (crate) fn extract_doc_markers (attrs : & [Attribute]) -> Vec < IntentMarker > { let mut markers = Vec :: new () ...`
  - Calls: Vec::new, is_ident, path, detect_latent_markers, value, push
- [Rust | Function] `item_attrs` (line 0, pub(crate))
  - Signature: `pub (crate) fn item_attrs (item : & Item) -> & [Attribute] { match item { Item :: Fn (item_fn) => & item_fn . attrs ,...`
- [Rust | Function] `item_name` (line 0, pub(crate))
  - Signature: `pub (crate) fn item_name (item : & Item) -> Option < String > { match item { Item :: Fn (item_fn) => Some (item_fn . ...`
  - Calls: Some, to_string, Some, to_string, Some, to_string, Some, to_string, Some, to_string
- [Rust | Function] `merge_doc_intent` (line 0, pub)
  - Signature: `pub fn merge_doc_intent (map : HashMap < String , Vec < IntentMarker > >) -> IntentMap { let mut merged = IntentMap :...`
  - Calls: IntentMap::new, HashSet::new, insert, push, or_default, entry, clone

