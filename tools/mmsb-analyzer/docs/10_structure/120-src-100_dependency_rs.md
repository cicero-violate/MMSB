# Structure Group: src/100_dependency.rs

## File: src/100_dependency.rs

- Layer(s): 100_dependency.rs
- Language coverage: Rust (7)
- Element types: Enum (1), Function (1), Impl (1), Struct (4)
- Total elements: 7

### Elements

- [Rust | Impl] `Default for impl Default for LayerGraph { fn default () -> Self { LayerGraph { ordered_layers : Vec :: new () , edges : Vec :: new () , cycles : Vec :: new () , unresolved : Vec :: new () , } } } . self_ty` (line 0, priv)
- [Rust | Struct] `LayerEdge` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct LayerEdge { pub from : String , pub to : String , pub references : Vec < Refere...`
- [Rust | Struct] `LayerGraph` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct LayerGraph { pub ordered_layers : Vec < String > , pub edges : Vec < LayerEdge ...`
- [Rust | Struct] `ReferenceDetail` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Eq , PartialEq , Ord , PartialOrd)] pub struct ReferenceDetail { pub file : PathBuf , pub ...`
- [Rust | Enum] `RootState` (line 0, pub)
- [Rust | Struct] `UnresolvedDependency` (line 0, pub)
  - Signature: `# [derive (Debug , Clone)] pub struct UnresolvedDependency { pub file : PathBuf , pub reference : String , }`
- [Rust | Function] `collect_roots` (line 0, pub)
  - Signature: `pub fn collect_roots (tree : & UseTree , state : RootState , acc : & mut BTreeSet < String >) { match tree { UseTree ...`
  - Calls: to_string, collect_roots, insert, insert, collect_roots, insert, to_string, insert, to_string

