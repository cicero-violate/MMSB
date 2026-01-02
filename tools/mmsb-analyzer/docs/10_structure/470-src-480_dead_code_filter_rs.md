# Structure Group: src/480_dead_code_filter.rs

## File: src/480_dead_code_filter.rs

- Layer(s): 480_dead_code_filter.rs
- Language coverage: Rust (3)
- Element types: Function (3)
- Total elements: 3

### Elements

- [Rust | Function] `collect_excluded_symbols` (line 0, priv)
  - Signature: `fn collect_excluded_symbols (report : & DeadCodeReportWithMeta) -> HashSet < String > { report . items . iter () . fi...`
  - Calls: collect, map, filter, iter, should_exclude_from_analysis, clone
- [Rust | Function] `filter_dead_code_elements` (line 0, pub)
  - Signature: `pub fn filter_dead_code_elements (elements : & [CodeElement] , report : & DeadCodeReportWithMeta ,) -> Vec < CodeElem...`
  - Calls: collect_excluded_symbols, collect, cloned, filter, iter, contains
- [Rust | Function] `should_exclude_from_analysis` (line 0, pub)
  - Signature: `pub fn should_exclude_from_analysis (category : DeadCodeCategory) -> bool { matches ! (category , DeadCodeCategory ::...`

