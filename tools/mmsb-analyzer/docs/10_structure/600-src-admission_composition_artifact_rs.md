# Structure Group: src/admission_composition_artifact.rs

## File: src/admission_composition_artifact.rs

- Layer(s): root
- Language coverage: Rust (12)
- Element types: Enum (2), Function (6), Module (1), Struct (3)
- Total elements: 12

### Elements

- [Rust | Struct] `AdmissionCompositionArtifact` (line 0, pub)
  - Signature: `# [doc = " Top-level admission composition artifact"] # [doc = ""] # [doc = " This is the complete, durable proof obj...`
- [Rust | Enum] `CompositionResultProjection` (line 0, pub)
- [Rust | Enum] `ConflictReasonProjection` (line 0, pub)
- [Rust | Struct] `ExecutorSurfaceProjection` (line 0, pub)
  - Signature: `# [doc = " Projection of ExecutorSurface"] # [derive (Debug , Clone , PartialEq , Eq , Serialize , Deserialize)] pub ...`
- [Rust | Struct] `StateProjection` (line 0, pub)
  - Signature: `# [doc = " Projection of ComposedEffectState"] # [derive (Debug , Clone , PartialEq , Eq , Serialize , Deserialize)] ...`
- [Rust | Function] `generate_artifact` (line 0, pub)
  - Signature: `# [doc = " Generate admission composition artifact from batch and result"] # [doc = ""] # [doc = " # Pure Projection"...`
  - Calls: to_rfc3339, chrono::Utc::now, project_invariants_touched, collect, map, iter, clone, project_conflict_reason, project_state, clone, to_string, len, to_string
- [Rust | Function] `project_conflict_reason` (line 0, priv)
  - Signature: `fn project_conflict_reason (reason : & ConflictReason) -> ConflictReasonProjection { match reason { ConflictReason ::...`
  - Calls: to_string, display, clone, clone, to_string, clone
- [Rust | Function] `project_invariants_touched` (line 0, priv)
  - Signature: `fn project_invariants_touched (state : & ComposedEffectState) -> Vec < String > { use crate :: composition_rule :: In...`
  - Calls: Vec::new, is_empty, push, to_string, sort
- [Rust | Function] `project_state` (line 0, priv)
  - Signature: `fn project_state (state : & ComposedEffectState) -> StateProjection { StateProjection { action_count : state . action...`
  - Calls: len, len, project_invariants_touched
- [Rust | Function] `read_artifact` (line 0, pub)
  - Signature: `# [doc = " Read artifact from JSON file"] pub fn read_artifact (path : & Path ,) -> Result < AdmissionCompositionArti...`
  - Calls: std::fs::read_to_string, serde_json::from_str, Ok
- [Rust | Module] `tests` (line 0, priv)
- [Rust | Function] `write_artifact` (line 0, pub)
  - Signature: `# [doc = " Write artifact to JSON file"] # [doc = ""] # [doc = " # Determinism"] # [doc = ""] # [doc = " JSON is seri...`
  - Calls: serde_json::to_string_pretty, std::fs::write, Ok

