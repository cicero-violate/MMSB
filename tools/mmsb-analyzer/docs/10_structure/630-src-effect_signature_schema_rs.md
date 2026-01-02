# Structure Group: src/effect_signature_schema.rs

## File: src/effect_signature_schema.rs

- Layer(s): root
- Language coverage: Rust (19)
- Element types: Enum (3), Impl (1), Module (1), Struct (14)
- Total elements: 19

### Elements

- [Rust | Enum] `BoundaryCrossingType` (line 0, pub)
- [Rust | Struct] `EffectSignature` (line 0, pub)
  - Signature: `# [doc = " Top-level effect signature for a transformation action"] # [doc = ""] # [doc = " # Completeness Requiremen...`
- [Rust | Struct] `ExecutorSurface` (line 0, pub)
  - Signature: `# [doc = " Executor surface: infrastructure gates required for safe execution"] # [doc = ""] # [doc = " # Enforcement...`
- [Rust | Struct] `FileToModuleTransition` (line 0, pub)
  - Signature: `# [doc = " File to module transition"] # [derive (Debug , Clone , PartialEq , Eq , Serialize , Deserialize)] pub stru...`
- [Rust | Struct] `ImportWrite` (line 0, pub)
  - Signature: `# [doc = " Import write operation"] # [derive (Debug , Clone , PartialEq , Eq , PartialOrd , Ord , Serialize , Deseri...`
- [Rust | Struct] `InvariantTouchpoints` (line 0, pub)
  - Signature: `# [doc = " Invariant touchpoints: which invariants this action validates against"] # [doc = ""] # [doc = " # Exhausti...`
- [Rust | Struct] `ModuleBoundary` (line 0, pub)
  - Signature: `# [doc = " Module boundary crossing descriptor"] # [derive (Debug , Clone , PartialEq , Eq , PartialOrd , Ord , Seria...`
- [Rust | Enum] `ModuleOperation` (line 0, pub)
- [Rust | Struct] `ModuleToLayerTransition` (line 0, pub)
  - Signature: `# [doc = " Module to layer transition"] # [derive (Debug , Clone , PartialEq , Eq , Serialize , Deserialize)] pub str...`
- [Rust | Struct] `ModuleWrite` (line 0, pub)
  - Signature: `# [doc = " Module write operation"] # [derive (Debug , Clone , PartialEq , Eq , PartialOrd , Ord , Serialize , Deseri...`
- [Rust | Struct] `ReExportWrite` (line 0, pub)
  - Signature: `# [doc = " Re-export write operation"] # [derive (Debug , Clone , PartialEq , Eq , PartialOrd , Ord , Serialize , Des...`
- [Rust | Struct] `ReadEffects` (line 0, pub)
  - Signature: `# [doc = " Read effects: surfaces the action inspects"] # [doc = ""] # [doc = " # Conservatism"] # [doc = ""] # [doc ...`
- [Rust | Struct] `StructuralTransitions` (line 0, pub)
  - Signature: `# [doc = " Structural transitions: architectural shape changes"] # [doc = ""] # [doc = " These represent phase transi...`
- [Rust | Struct] `TestBoundaryCrossing` (line 0, pub)
  - Signature: `# [doc = " Test boundary crossing"] # [derive (Debug , Clone , PartialEq , Eq , Serialize , Deserialize)] pub struct ...`
- [Rust | Enum] `TestBoundaryDirection` (line 0, pub)
- [Rust | Struct] `VisibilityWrite` (line 0, pub)
  - Signature: `# [doc = " Visibility modifier write"] # [derive (Debug , Clone , PartialEq , Eq , PartialOrd , Ord , Serialize , Des...`
- [Rust | Struct] `WriteEffects` (line 0, pub)
  - Signature: `# [doc = " Write effects: surfaces the action mutates"] # [doc = ""] # [doc = " # Guard Equivalence"] # [doc = ""] # ...`
- [Rust | Impl] `impl EffectSignature { # [doc = " Validate that this signature is complete and well-formed"] # [doc = ""] # [doc = " # Returns"] # [doc = ""] # [doc = " - `Ok(())` if signature is valid"] # [doc = " - `Err(reason)` if signature is incomplete or malformed"] pub fn validate (& self) -> Result < () , String > { if self . schema_version != SCHEMA_VERSION { return Err (format ! ("Schema version mismatch: expected {}, got {}" , SCHEMA_VERSION , self . schema_version)) ; } if self . action_type . is_empty () { return Err ("action_type cannot be empty" . to_string ()) ; } if self . action_id . is_empty () { return Err ("action_id cannot be empty" . to_string ()) ; } Ok (()) } # [doc = " Check if this signature conflicts with another on write surfaces"] # [doc = ""] # [doc = " Two actions conflict if they write to overlapping surfaces in ways"] # [doc = " that cannot be proven commutative."] pub fn conflicts_with (& self , other : & EffectSignature) -> bool { if ! self . writes . files . is_disjoint (& other . writes . files) { return true ; } let self_modules : BTreeSet < _ > = self . writes . modules . iter () . map (| m | & m . module_path) . collect () ; let other_modules : BTreeSet < _ > = other . writes . modules . iter () . map (| m | & m . module_path) . collect () ; if ! self_modules . is_disjoint (& other_modules) { return true ; } let self_inv = & self . invariant_touchpoints ; let other_inv = & other . invariant_touchpoints ; if (self_inv . i1_module_coherence && other_inv . i1_module_coherence) || (self_inv . i2_dependency_direction && other_inv . i2_dependency_direction) || (self_inv . visibility_law && other_inv . visibility_law) || (self_inv . re_export_law && other_inv . re_export_law) || (self_inv . test_topology_law && other_inv . test_topology_law) { return true ; } false } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)

