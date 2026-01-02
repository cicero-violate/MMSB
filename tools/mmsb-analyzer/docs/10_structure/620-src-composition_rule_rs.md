# Structure Group: src/composition_rule.rs

## File: src/composition_rule.rs

- Layer(s): root
- Language coverage: Rust (10)
- Element types: Enum (3), Function (4), Impl (1), Module (1), Struct (1)
- Total elements: 10

### Elements

- [Rust | Struct] `ComposedEffectState` (line 0, pub)
  - Signature: `# [doc = " Symbolic shadow state of accumulated effects"] # [doc = ""] # [doc = " This is **not** the program state. ...`
- [Rust | Enum] `CompositionResult` (line 0, pub)
- [Rust | Enum] `ConflictReason` (line 0, pub)
- [Rust | Enum] `InvariantType` (line 0, pub)
- [Rust | Function] `check_conflicts` (line 0, priv)
  - Signature: `# [doc = " Check if a signature conflicts with accumulated state"] # [doc = ""] # [doc = " Returns `Some(ConflictReas...`
  - Calls: get, Some, clone, get, Some, clone, clone, clone, get, Some, clone, clone, get, Some, clone, clone, get, Some, get, Some, collect_invariants_touched, get, is_empty, Some
- [Rust | Function] `collect_invariants_touched` (line 0, priv)
  - Signature: `# [doc = " Collect invariants that are touched by a signature"] fn collect_invariants_touched (touchpoints : & Invari...`
  - Calls: Vec::new, push, push, push, push, push
- [Rust | Function] `compose_batch` (line 0, pub)
  - Signature: `# [doc = " Compose a batch of actions in order, aborting on first conflict"] # [doc = ""] # [doc = " # Arguments"] # ...`
  - Calls: ComposedEffectState::empty, enumerate, iter, check_conflicts, clone, compose_into_state, len
- [Rust | Function] `compose_into_state` (line 0, priv)
  - Signature: `# [doc = " Compose a signature into the accumulated state"] # [doc = ""] # [doc = " This is called only after conflic...`
  - Calls: insert, clone, insert, clone, clone, clone, insert, clone, clone, insert, clone, clone, insert, insert, clone, insert, clone, collect_invariants_touched, push, or_insert_with, entry
- [Rust | Impl] `impl ComposedEffectState { # [doc = " Create empty initial state"] pub fn empty () -> Self { Self { files_written : BTreeMap :: new () , modules_written : BTreeMap :: new () , imports_written : BTreeMap :: new () , re_exports_written : BTreeMap :: new () , visibility_modifiers_written : BTreeMap :: new () , files_read : BTreeSet :: new () , symbols_read : BTreeSet :: new () , invariants_touched : BTreeMap :: new () , executor_surfaces : ExecutorSurface { requires_import_repair : false , requires_module_shim : false , requires_re_export_enforcement : false , requires_verification_gate : false , } , action_count : 0 , } } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)

