# Structure Group: src/010_layer_utilities.rs

## File: src/010_layer_utilities.rs

- Layer(s): 010_layer_utilities.rs
- Language coverage: Rust (22)
- Element types: Function (21), Struct (1)
- Total elements: 22

### Elements

- [Rust | Struct] `FunctionInfo` (line 0, pub)
  - Signature: `# [derive (Clone)] pub struct FunctionInfo { pub name : String , pub signature : String , pub file_path : String , pu...`
- [Rust | Function] `adjacency_from_edges` (line 0, priv)
  - Signature: `fn adjacency_from_edges (edges_map : & BTreeMap < (String , String) , BTreeSet < ReferenceDetail > > ,) -> HashMap < ...`
  - Calls: HashMap::new, insert, or_default, entry, clone, clone
- [Rust | Function] `build_result` (line 0, pub)
  - Signature: `pub fn build_result (files : & [PathBuf] , file_layers : HashMap < PathBuf , String > , nodes : BTreeSet < String > ,...`
  - Calls: adjacency_from_edges, topo_sort, position, iter, remove, insert, layer_rank_map, to_vec, sort_by, is_mmsb_main, is_mmsb_main, contains, contains, unwrap_or_else, cloned, get, to_string, unwrap_or_else, cloned, get, to_string, unwrap_or, cloned, get, len, unwrap_or, cloned, get, len, then_with, then_with, cmp, cmp, cmp, collect, map, into_iter, is_layer_violation, collect, into_iter, Ok
- [Rust | Function] `cluster_target_path` (line 0, pub)
  - Signature: `pub fn cluster_target_path (target : PathBuf , members : & [crate :: report :: ClusterMember] , root_path : & Path , ...`
  - Calls: is_core_module_path, unwrap_or, and_then, and_then, file_stem, to_str, layer_prefix_value, unwrap_or, and_then, first, parent, join
- [Rust | Function] `collect_cluster_plans` (line 0, pub)
  - Signature: `pub fn collect_cluster_plans (clusters : & [crate :: types :: FunctionCluster] , root_path : & Path ,) -> Vec < crate...`
  - Calls: Vec::new, enumerate, iter, parse_cluster_members, clone, first, join, unwrap_or, parent, join, cluster_target_path, collect, filter, into_iter, len, push, sort_by, then_with, then_with, unwrap_or, partial_cmp, cmp, len, len, cmp
- [Rust | Function] `compare_dir_layers` (line 0, pub)
  - Signature: `pub fn compare_dir_layers (a : & Path , b : & Path) -> Ordering { let a_name = a . file_name () . and_then (| n | n ....`
  - Calls: unwrap_or, and_then, file_name, to_str, unwrap_or, and_then, file_name, to_str, unwrap_or, layer_prefix_value, unwrap_or, layer_prefix_value, then_with, cmp, cmp
- [Rust | Function] `compare_path_components` (line 0, pub)
  - Signature: `pub fn compare_path_components (a : & Path , b : & Path) -> Ordering { let a_components : Vec < _ > = a . components ...`
  - Calls: collect, components, collect, components, min, len, len, to_string_lossy, as_os_str, to_string_lossy, as_os_str, layer_prefix_value, layer_prefix_value, cmp, cmp, cmp, len, len
- [Rust | Function] `cyclomatic_complexity` (line 0, pub)
  - Signature: `pub fn cyclomatic_complexity (cfg : & crate :: types :: FunctionCfg) -> usize { let edges = cfg . edges . len () as i...`
  - Calls: len, len
- [Rust | Function] `detect_layer_violation` (line 0, pub)
  - Signature: `pub fn detect_layer_violation (func : & FunctionInfo , functions : & [FunctionInfo] , outgoing : & HashMap < usize , ...`
  - Calls: unwrap_or_else, cloned, get, clone, layer_prefix_value, unwrap_or_else, cloned, get, clone, layer_prefix_value, Some, map
- [Rust | Function] `insert_sorted` (line 0, priv)
  - Signature: `fn insert_sorted (queue : & mut VecDeque < String > , value : String) { let mut inserted = false ; for idx in 0 .. qu...`
  - Calls: len, insert, clone, push_back
- [Rust | Function] `is_core_module_path` (line 0, pub)
  - Signature: `pub fn is_core_module_path (path : & Path) -> bool { let Some (stem) = path . file_stem () . and_then (| name | name ...`
  - Calls: and_then, file_stem, to_str, starts_with, starts_with
- [Rust | Function] `is_layer_violation` (line 0, pub)
  - Signature: `# [doc = " Checks if a dependency from one layer to another violates layer ordering"] # [doc = " Returns true if from...`
  - Calls: layer_prefix_value, layer_prefix_value
- [Rust | Function] `is_mmsb_main` (line 0, priv)
  - Signature: `fn is_mmsb_main (path : & Path) -> bool { path . file_name () . and_then (| n | n . to_str ()) . map (| n | n == "MMS...`
  - Calls: unwrap_or, map, and_then, file_name, to_str
- [Rust | Function] `layer_adheres` (line 0, pub)
  - Signature: `pub fn layer_adheres (current_layer : & str , target_layer : & str) -> bool { match (layer_prefix_value (current_laye...`
  - Calls: layer_prefix_value, layer_prefix_value
- [Rust | Function] `layer_prefix_value` (line 0, priv)
  - Signature: `# [doc = " Extracts numeric layer prefix from a layer string (e.g., \"060_file_ordering\" -> 60)"] fn layer_prefix_va...`
  - Calls: chars, String::new, next, is_ascii_digit, push, is_empty, ok, parse
- [Rust | Function] `layer_rank_map` (line 0, priv)
  - Signature: `fn layer_rank_map (order : & [String]) -> HashMap < String , usize > { let mut rank = HashMap :: new () ; for (idx , ...`
  - Calls: HashMap::new, enumerate, iter, insert, clone
- [Rust | Function] `node_style` (line 0, pub)
  - Signature: `pub fn node_style (node_type : & NodeType) -> (& str , & str , & str) { match node_type { NodeType :: Entry => ("elli...`
- [Rust | Function] `parse_cluster_members` (line 0, pub)
  - Signature: `pub fn parse_cluster_members (cluster : & crate :: types :: FunctionCluster ,) -> Vec < crate :: report :: ClusterMem...`
  - Calls: collect, filter_map, iter, rsplit_once, Some, PathBuf::from, to_string
- [Rust | Function] `sort_structural_items` (line 0, pub)
  - Signature: `pub fn sort_structural_items (items : & mut Vec < crate :: report :: PlanItem >) { use std :: collections :: HashMap ...`
  - Calls: len, len, HashMap::new, enumerate, iter, push, or_default, entry, clone, structural_layer_value, structural_layer_value, Some, Some, Some, Some, push, enumerate, iter, get, push, Vec::with_capacity, collect, filter, is_empty, sort_by, structural_cmp, remove, push, saturating_sub, push, len, sort_by, Vec::with_capacity, push, clone
- [Rust | Function] `structural_cmp` (line 0, pub)
  - Signature: `pub fn structural_cmp (a : & crate :: report :: PlanItem , b : & crate :: report :: PlanItem) -> std :: cmp :: Orderi...`
  - Calls: structural_layer_value, structural_layer_value, structural_layer_value, structural_layer_value, saturating_mul, saturating_mul, then_with, then_with, then_with, then_with, then_with, cmp, cmp, cmp, cmp, cmp, cmp
- [Rust | Function] `structural_layer_value` (line 0, pub(crate))
  - Signature: `pub (crate) fn structural_layer_value (layer : & Option < String > , default : i32) -> i32 { layer . as_ref () . and_...`
  - Calls: unwrap_or, and_then, as_ref, layer_prefix_value
- [Rust | Function] `topo_sort` (line 0, priv)
  - Signature: `fn topo_sort (nodes : & BTreeSet < String > , adjacency : & HashMap < String , BTreeSet < String > > ,) -> (Vec < Str...`
  - Calls: HashMap::new, or_insert, entry, clone, values, or_insert, entry, clone, collect, filter_map, iter, Some, clone, sort, make_contiguous, Vec::new, pop_front, push, clone, get, get_mut, insert_sorted, clone, len, len, collect, cloned, filter, iter, contains, sort, clone, extend, Vec::new

