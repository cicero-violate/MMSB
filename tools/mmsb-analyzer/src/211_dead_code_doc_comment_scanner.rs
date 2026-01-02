use crate::dead_code_doc_comment_parser::extract_doc_markers;
use crate::dead_code_doc_comment_parser::item_attrs;
use crate::dead_code_doc_comment_parser::item_name;
use crate::dead_code_types::IntentMarker;
use std::collections::HashMap;
use std::path::Path;
pub fn scan_doc_comments(file: &Path) -> HashMap<String, Vec<IntentMarker>> {
    let contents = std::fs::read_to_string(file).unwrap_or_default();
    let parsed = match syn::parse_file(&contents) {
        Ok(file) => file,
        Err(_) => return HashMap::new(),
    };
    let mut map: HashMap<String, Vec<IntentMarker>> = HashMap::new();
    for item in &parsed.items {
        let Some(symbol) = item_name(item) else {
            continue;
        };
        let markers = extract_doc_markers(item_attrs(item));
        if markers.is_empty() {
            continue;
        }
        map.entry(symbol).or_default().extend(markers);
    }
    map
}
