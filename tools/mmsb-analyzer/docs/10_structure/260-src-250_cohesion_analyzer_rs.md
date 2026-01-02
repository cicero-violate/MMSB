# Structure Group: src/250_cohesion_analyzer.rs

## File: src/250_cohesion_analyzer.rs

- Layer(s): 250_cohesion_analyzer.rs
- Language coverage: Rust (3)
- Element types: Function (1), Impl (1), Struct (1)
- Total elements: 3

### Elements

- [Rust | Struct] `FunctionCohesionAnalyzer` (line 0, pub)
  - Signature: `# [derive (Default)] pub struct FunctionCohesionAnalyzer { threshold : f64 , }`
- [Rust | Function] `extract_identifiers` (line 0, priv)
  - Signature: `fn extract_identifiers (text : & str) -> Vec < String > { let mut tokens = Vec :: new () ; let mut current = String :...`
  - Calls: Vec::new, String::new, chars, is_ascii_alphanumeric, push, is_empty, push, clone, clear, is_empty, push
- [Rust | Impl] `impl FunctionCohesionAnalyzer { pub fn new () -> Self { Self { threshold : DEFAULT_THRESHOLD , } } # [allow (dead_code)] pub fn with_threshold (threshold : f64) -> Self { Self { threshold } } pub fn analyze (& self , result : & AnalysisResult) -> Result < Vec < FunctionPlacement > > { let (functions , outgoing_counts , incoming_counts) = build_call_edges (result) ; let file_layers = build_function_layers (& functions) ; let (file_types , all_types) = build_type_maps (result) ; let mut placements = Vec :: new () ; for (idx , func) in functions . iter () . enumerate () { let call_analysis = build_call_analysis (func , & functions , & outgoing_counts [idx] , & incoming_counts [idx] , & file_types , & all_types ,) ; let cohesion_score = compute_cohesion_score (func , & functions , & outgoing_counts [idx] , & file_layers , & call_analysis ,) ; let layer_violation = detect_layer_violation (func , & functions , & outgoing_counts [idx] , & file_layers) ; let placement_status = determine_status (& call_analysis , cohesion_score , self . threshold , layer_violation) ; let suggested_file = match & placement_status { PlacementStatus :: ShouldMove { .. } => suggest_file (& call_analysis) , PlacementStatus :: Orphaned { .. } => None , PlacementStatus :: LayerViolation { .. } => None , PlacementStatus :: Correct => None , } ; placements . push (FunctionPlacement { name : func . name . clone () , signature : func . signature . clone () , current_file : PathBuf :: from (& func . file_path) , suggested_file , placement_status , cohesion_score , call_analysis , }) ; } Ok (placements) } pub fn detect_clusters (& self , result : & AnalysisResult) -> Result < Vec < FunctionCluster > > { let (functions , outgoing_counts , _) = build_call_edges (result) ; let assignments = louvain_communities (& outgoing_counts) ; let mut clusters_map : HashMap < usize , Vec < usize > > = HashMap :: new () ; for (idx , comm) in assignments . into_iter () . enumerate () { clusters_map . entry (comm) . or_default () . push (idx) ; } let mut clusters = Vec :: new () ; for component in clusters_map . values () { if component . len () < 3 { continue ; } let cohesion = compute_cluster_cohesion (component , & outgoing_counts) ; let suggested_file = suggest_cluster_file (component , & functions) ; let members = component . iter () . map (| idx | { format ! ("{}::{}" , functions [* idx] . file_path , functions [* idx] . name) }) . collect () ; clusters . push (FunctionCluster { members , cohesion , suggested_file , }) ; } Ok (clusters) } } . self_ty` (line 0, priv)

