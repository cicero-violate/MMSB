# Structure Group: src/090_fixpoint_solver.rs

## File: src/090_fixpoint_solver.rs

- Layer(s): 090_fixpoint_solver.rs
- Language coverage: Rust (5)
- Element types: Impl (2), Module (1), Struct (2)
- Total elements: 5

### Elements

- [Rust | Impl] `Default for impl Default for SymbolicAbstraction { fn default () -> Self { Self :: new () } } . self_ty` (line 0, priv)
- [Rust | Struct] `FixpointResult` (line 0, pub)
  - Signature: `# [doc = " Result of fixpoint computation"] # [derive (Debug)] # [allow (dead_code)] pub struct FixpointResult { # [d...`
- [Rust | Struct] `SymbolicAbstraction` (line 0, pub)
  - Signature: `# [doc = " Symbolic abstraction of program state"] # [derive (Debug , Clone , PartialEq , Eq)] # [allow (dead_code)] ...`
- [Rust | Impl] `impl SymbolicAbstraction { pub fn new () -> Self { Self { type_sig : None , effects : HashSet :: new () , layer : None , visibility : None , properties : HashSet :: new () , } } # [doc = " Check if two abstractions are approximately equal"] pub fn approx_eq (& self , other : & Self) -> bool { self . type_sig == other . type_sig && self . effects == other . effects && self . layer == other . layer && self . visibility == other . visibility && self . properties == other . properties } # [doc = " Merge two abstractions (used during propagation)"] pub fn merge (& mut self , other : & Self) { if self . type_sig . is_none () && other . type_sig . is_some () { self . type_sig = other . type_sig . clone () ; } self . effects . extend (other . effects . clone ()) ; match (self . layer , other . layer) { (Some (l1) , Some (l2)) => self . layer = Some (l1 . max (l2)) , (None , Some (l)) => self . layer = Some (l) , _ => { } } if self . visibility . is_none () && other . visibility . is_some () { self . visibility = other . visibility . clone () ; } self . properties . extend (other . properties . clone ()) ; } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)

