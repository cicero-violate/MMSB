# Structure Group: src/540_dead_code_report_split.rs

## File: src/540_dead_code_report_split.rs

- Layer(s): 540_dead_code_report_split.rs
- Language coverage: Rust (4)
- Element types: Function (4)
- Total elements: 4

### Elements

- [Rust | Function] `plan_options` (line 0, priv)
  - Signature: `fn plan_options (item : & DeadCodeItem) -> String { let options = match item . category { DeadCodeCategory :: Unreach...`
  - Calls: push_str
- [Rust | Function] `top_items` (line 0, priv)
  - Signature: `fn top_items (items : & [DeadCodeItem] , limit : usize) -> Vec < DeadCodeItem > { let mut items = items . to_vec () ;...`
  - Calls: to_vec, sort_by_key, len, truncate
- [Rust | Function] `write_plan_markdown` (line 0, pub)
  - Signature: `pub fn write_plan_markdown (path : & Path , report : & DeadCodeReportWithMeta , limit : usize ,) -> std :: io :: Resu...`
  - Calls: String::new, push_str, push_str, push_str, push_str, top_items, push_str, is_empty, push_str, plan_options, push_str, push, std::fs::write
- [Rust | Function] `write_summary_markdown` (line 0, pub)
  - Signature: `pub fn write_summary_markdown (path : & Path , report : & DeadCodeReportWithMeta , limit : usize ,) -> std :: io :: R...`
  - Calls: String::new, push_str, push_str, push_str, push_str, top_items, push_str, is_empty, push_str, push_str, push, std::fs::write

