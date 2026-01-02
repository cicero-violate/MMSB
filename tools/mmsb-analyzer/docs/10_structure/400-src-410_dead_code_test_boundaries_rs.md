# Structure Group: src/410_dead_code_test_boundaries.rs

## File: src/410_dead_code_test_boundaries.rs

- Layer(s): 410_dead_code_test_boundaries.rs
- Language coverage: Rust (4)
- Element types: Function (3), Struct (1)
- Total elements: 4

### Elements

- [Rust | Struct] `TestBoundaries` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Default)] pub struct TestBoundaries { pub test_modules : HashSet < String > , pub test_sym...`
- [Rust | Function] `find_test_callers` (line 0, pub)
  - Signature: `pub fn find_test_callers (symbol : & str , call_graph : & CallGraph , test_symbols : & HashSet < String > ,) -> Vec <...`
  - Calls: is_empty, Vec::new, build_reverse_call_graph, Vec::new, HashSet::new, collect, into_iter, unwrap_or_default, cloned, get, pop_front, insert, clone, contains, push, clone, get, contains, push_back, clone
- [Rust | Function] `has_test_attr` (line 0, pub(crate))
  - Signature: `pub (crate) fn has_test_attr (attrs : & [Attribute]) -> bool { attrs . iter () . any (| attr | { let path = attr . pa...`
  - Calls: any, iter, path, is_ident, map, last, to_string
- [Rust | Function] `item_attrs` (line 0, priv)
  - Signature: `fn item_attrs (item : & Item) -> & [Attribute] { match item { Item :: Fn (item_fn) => & item_fn . attrs , Item :: Str...`

