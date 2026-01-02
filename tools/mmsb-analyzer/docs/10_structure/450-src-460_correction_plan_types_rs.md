# Structure Group: src/460_correction_plan_types.rs

## File: src/460_correction_plan_types.rs

- Layer(s): 460_correction_plan_types.rs
- Language coverage: Rust (9)
- Element types: Enum (5), Impl (1), Struct (3)
- Total elements: 9

### Elements

- [Rust | Struct] `CorrectionPlan` (line 0, pub)
  - Signature: `# [derive (Clone , Debug , Serialize , Deserialize)] pub struct CorrectionPlan { pub action_id : String , pub tier : ...`
- [Rust | Enum] `CorrectionStrategy` (line 0, pub)
- [Rust | Enum] `ErrorTier` (line 0, pub)
- [Rust | Enum] `RefactorAction` (line 0, pub)
- [Rust | Enum] `Severity` (line 0, pub)
- [Rust | Struct] `ViolationPrediction` (line 0, pub)
  - Signature: `# [derive (Clone , Debug , Serialize , Deserialize)] pub struct ViolationPrediction { pub violation_type : ViolationT...`
- [Rust | Enum] `ViolationType` (line 0, pub)
- [Rust | Struct] `VisibilityPlanOption` (line 0, pub)
  - Signature: `# [derive (Clone , Debug , Serialize , Deserialize)] pub struct VisibilityPlanOption { pub policy : String , pub targ...`
- [Rust | Impl] `impl RefactorAction { pub fn action_id (& self) -> String { match self { RefactorAction :: MoveFunction { function , to , .. } => { format ! ("move_{}_to_{}" , function , to . display ()) } RefactorAction :: RenameFunction { old_name , new_name , .. } => { format ! ("rename_{}_to_{}" , old_name , new_name) } RefactorAction :: RenameFile { from , to } => { format ! ("rename_file_{}_to_{}" , from . display () , to . display ()) } RefactorAction :: CreateFile { path } => format ! ("create_file_{}" , path . display ()) , RefactorAction :: AdjustVisibility { symbol , file , .. } => { format ! ("adjust_visibility_{}_in_{}" , symbol , file . display ()) } } } } . self_ty` (line 0, priv)

