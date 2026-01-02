# Structure Group: src/280_file_ordering.rs

## File: src/280_file_ordering.rs

- Layer(s): 280_file_ordering.rs
- Language coverage: Rust (4)
- Element types: Function (1), Impl (1), Struct (2)
- Total elements: 4

### Elements

- [Rust | Impl] `# [allow (dead_code)] impl DagCache { pub fn new (files : & [PathBuf]) -> Result < Self > { let file_set : HashSet < PathBuf > = files . iter () . cloned () . collect () ; let module_map = build_module_map (files) ; let dep_map = build_dependency_map (files , & file_set , & module_map) ? ; let (graph , node_map) = build_file_dag (files , & dep_map) ; let topo_order = topological_sort (& graph) . unwrap_or_else (| _ | { graph . node_indices () . collect :: < Vec < _ > > () }) ; let mut last_modified = HashMap :: new () ; for file in files { if let Ok (meta) = fs :: metadata (file) { if let Ok (modified) = meta . modified () { last_modified . insert (file . clone () , modified) ; } } } Ok (Self { graph , node_map , topo_order , last_modified , file_set , module_map , }) } pub fn incremental_update (& mut self , changed_files : & [PathBuf]) -> Result < () > { for file in changed_files { if ! self . file_set . contains (file) { continue ; } let Some (& node) = self . node_map . get (file) else { continue ; } ; let old_edges : Vec < _ > = self . graph . edges (node) . map (| e | e . id ()) . collect () ; for edge in old_edges { self . graph . remove_edge (edge) ; } let deps = extract_dependencies (file , & self . file_set , & self . module_map) ? ; for dep in deps { if let Some (& dep_node) = self . node_map . get (& dep) { self . graph . add_edge (dep_node , node , ()) ; } } if let Ok (meta) = fs :: metadata (file) { if let Ok (modified) = meta . modified () { self . last_modified . insert (file . clone () , modified) ; } } } if let Ok (order) = topological_sort (& self . graph) { self . topo_order = order ; } Ok (()) } pub fn topo_files (& self) -> Vec < PathBuf > { self . topo_order . iter () . map (| idx | self . graph [* idx] . clone ()) . collect () } } . self_ty` (line 0, priv)
- [Rust | Struct] `DagCache` (line 0, pub)
  - Signature: `# [allow (dead_code)] pub struct DagCache { graph : DiGraph < PathBuf , () > , node_map : HashMap < PathBuf , NodeInd...`
- [Rust | Struct] `DirectoryMove` (line 0, pub)
  - Signature: `# [derive (Clone , Debug)] pub struct DirectoryMove { pub from : PathBuf , pub to : PathBuf , }`
- [Rust | Function] `parallel_build_file_dag` (line 0, pub)
  - Signature: `# [allow (dead_code)] pub fn parallel_build_file_dag (directories : & [PathBuf]) -> Result < DiGraph < PathBuf , () >...`
  - Calls: collect, map, par_iter, crate::dependency::build_directory_dag, DiGraph::new, HashMap::new, node_indices, clone, or_insert_with, entry, clone, add_node, edge_indices, edge_endpoints, clone, clone, expect, get, expect, get, add_edge, Ok

