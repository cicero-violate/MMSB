# Structure Group: src/470_dead_code_report.rs

## File: src/470_dead_code_report.rs

- Layer(s): 470_dead_code_report.rs
- Language coverage: Rust (6)
- Element types: Function (4), Struct (2)
- Total elements: 6

### Elements

- [Rust | Struct] `DeadCodeReportMetadata` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct DeadCodeReportMetadata { pub analyzer_version : Strin...`
- [Rust | Struct] `DeadCodeReportWithMeta` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct DeadCodeReportWithMeta { pub timestamp : String , pub...`
- [Rust | Function] `build_basic_report` (line 0, pub)
  - Signature: `pub fn build_basic_report (timestamp : String , items : Vec < DeadCodeItem >) -> DeadCodeReport { let summary = DeadC...`
  - Calls: DeadCodeSummary::from_items
- [Rust | Function] `build_report` (line 0, pub)
  - Signature: `pub fn build_report (timestamp : String , items : Vec < DeadCodeItem > , metadata : DeadCodeReportMetadata ,) -> Dead...`
  - Calls: DeadCodeSummary::from_items
- [Rust | Function] `write_outputs` (line 0, pub(crate))
  - Signature: `pub (crate) fn write_outputs (report : & DeadCodeReportWithMeta , config : & DeadCodeRunConfig) -> Result < () > { le...`
  - Calls: unwrap_or_else, clone, join, parent, std::fs::create_dir_all, write_report, unwrap_or_else, clone, join, parent, std::fs::create_dir_all, write_summary_markdown, unwrap_or_else, map, parent, to_path_buf, clone, join, write_plan_markdown, Ok
- [Rust | Function] `write_report` (line 0, pub)
  - Signature: `pub fn write_report (path : & Path , report : & DeadCodeReportWithMeta) -> std :: io :: Result < () > { let json = se...`
  - Calls: serde_json::to_string_pretty, std::fs::write

