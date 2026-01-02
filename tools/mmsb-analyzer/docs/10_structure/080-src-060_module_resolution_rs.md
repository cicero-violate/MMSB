# Structure Group: src/060_module_resolution.rs

## File: src/060_module_resolution.rs

- Layer(s): 060_module_resolution.rs
- Language coverage: Rust (15)
- Element types: Function (9), Impl (2), Module (1), Struct (3)
- Total elements: 15

### Elements

- [Rust | Struct] `LayerResolver` (line 0, pub(crate))
  - Signature: `pub (crate) struct LayerResolver { aliases : HashMap < String , String > , }`
- [Rust | Struct] `ModuleRoot` (line 0, pub)
  - Signature: `# [derive (Clone)] pub struct ModuleRoot { pub layer : String , }`
- [Rust | Struct] `UseCollector` (line 0, priv)
  - Signature: `# [derive (Default)] struct UseCollector { roots : BTreeSet < String > , mods : BTreeSet < String > , }`
- [Rust | Impl] `Visit for impl < 'ast > Visit < 'ast > for UseCollector { fn visit_item_use (& mut self , node : & 'ast ItemUse) { crate :: dependency :: collect_roots (& node . tree , RootState :: Start , & mut self . roots) ; } fn visit_item_mod (& mut self , node : & 'ast syn :: ItemMod) { if node . content . is_none () { self . mods . insert (node . ident . to_string ()) ; } } } . self_ty` (line 0, priv)
- [Rust | Function] `build_dependency_map` (line 0, pub)
  - Signature: `pub fn build_dependency_map (files : & [PathBuf] , file_set : & HashSet < PathBuf > , module_map : & HashMap < String...`
  - Calls: HashMap::new, with_context, extract_dependencies, insert, clone, Ok
- [Rust | Function] `build_module_root_map` (line 0, pub)
  - Signature: `pub fn build_module_root_map (root : & Path) -> Result < HashMap < String , ModuleRoot > , std :: io :: Error > { let...`
  - Calls: join, HashMap::new, is_dir, fs::read_dir, path, contains_tools, to_string, trim_end_matches, to_string, to_string_lossy, file_name, is_dir, normalize_module_name, insert, clone, unwrap_or, map, extension, insert, clone, crate::cluster_001::detect_layer, Ok
- [Rust | Function] `contains_tools` (line 0, pub)
  - Signature: `pub fn contains_tools (path : & Path) -> bool { path . components () . any (| c | c . as_os_str () == "tools") } . sig`
  - Calls: any, components, as_os_str
- [Rust | Function] `extract_dependencies` (line 0, pub(crate))
  - Signature: `pub (crate) fn extract_dependencies (file : & Path , file_set : & HashSet < PathBuf > , module_map : & HashMap < Stri...`
  - Calls: unwrap_or, and_then, extension, to_str, extract_rust_dependencies, extract_julia_dependencies, Ok, Vec::new
- [Rust | Function] `extract_julia_dependencies` (line 0, pub)
  - Signature: `pub fn extract_julia_dependencies (file : & Path , file_set : & HashSet < PathBuf > , module_map : & HashMap < String...`
  - Calls: Lazy::new, unwrap, Regex::new, Lazy::new, unwrap, Regex::new, Lazy::new, unwrap, Regex::new, Lazy::new, unwrap, Regex::new, Lazy::new, unwrap, Regex::new, unwrap_or, next, split, resolve_module, with_context, fs::read_to_string, Vec::new, captures_iter, get, as_str, PathBuf::from, is_none, extension, set_extension, is_absolute, unwrap_or, map, parent, join, crate::cluster_011::resolve_path, push, captures_iter, get, resolve_module_name, as_str, push, captures_iter, get, filter, map, split, as_str, trim, is_empty, resolve_module_name, push, captures_iter, get, resolve_module_name, as_str, push, captures_iter, get, as_str, starts_with, resolve_module_name, push, Ok
- [Rust | Function] `extract_rust_dependencies` (line 0, pub)
  - Signature: `pub fn extract_rust_dependencies (file : & Path , file_set : & HashSet < PathBuf > , module_map : & HashMap < String ...`
  - Calls: crate::dependency::collect_roots, is_none, insert, to_string, with_context, fs::read_to_string, with_context, syn::parse_file, UseCollector::default, visit_file, Vec::new, resolve_module, push, resolve_module, push, Ok
- [Rust | Impl] `impl LayerResolver { pub (crate) fn build (root : & Path) -> Result < Self > { let mut resolver = LayerResolver { aliases : HashMap :: new () , } ; let src_dir = resolve_source_root (root) ; if src_dir . is_dir () { for entry in WalkDir :: new (& src_dir) . into_iter () . filter_map (| e | e . ok ()) { let path = entry . path () ; if contains_tools (path) { continue ; } let layer = crate :: cluster_001 :: detect_layer (path) ; if layer == "root" { continue ; } if path . is_dir () { if let Some (name) = path . file_name () . and_then (| n | n . to_str ()) { resolver . add_aliases (name , & layer) ; } } else if path . extension () . map_or (false , | ext | ext == "jl") { if let Some (stem) = path . file_stem () . and_then (| n | n . to_str ()) { resolver . add_aliases (stem , & layer) ; } } } } Ok (resolver) } fn add_aliases (& mut self , name : & str , layer : & str) { let lower = name . to_lowercase () ; self . aliases . entry (lower . clone ()) . or_insert_with (| | layer . to_string ()) ; let condensed = lower . replace ('_' , "") ; self . aliases . entry (condensed) . or_insert_with (| | layer . to_string ()) ; } pub (crate) fn resolve_module (& self , module : & str) -> Option < String > { let key = module . to_lowercase () ; if let Some (layer) = self . aliases . get (& key) { return Some (layer . clone ()) ; } let condensed = key . replace ('_' , "") ; if let Some (layer) = self . aliases . get (& condensed) { return Some (layer . clone ()) ; } self . aliases . iter () . filter (| (alias , _) | ! alias . is_empty ()) . find (| (alias , _) | key . starts_with (alias . as_str ())) . map (| (_ , layer) | layer . clone ()) } } . self_ty` (line 0, priv)
- [Rust | Module] `moved_gather_rust_files` (line 0, priv)
- [Rust | Function] `normalize_module_name` (line 0, pub)
  - Signature: `pub fn normalize_module_name (name : & str) -> String { if let Some (pos) = name . find ('_') { if name [.. pos] . ch...`
  - Calls: find, all, chars, is_ascii_digit, to_string, to_string
- [Rust | Function] `resolve_module` (line 0, pub)
  - Signature: `pub fn resolve_module (root : & str , file_set : & HashSet < PathBuf > , module_map : & HashMap < String , PathBuf > ...`
  - Calls: normalize_module_name, get, Some, clone, or_else, or_else, map, find, iter, clone, map, find, iter, starts_with, as_str, clone, crate::cluster_011::resolve_path, PathBuf::from
- [Rust | Function] `resolve_module_name` (line 0, priv)
  - Signature: `fn resolve_module_name (module : & str , file_set : & HashSet < PathBuf > , module_map : & HashMap < String , PathBuf...`
  - Calls: unwrap_or, next, split, resolve_module

