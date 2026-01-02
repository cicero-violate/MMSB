# Structure Group: src/620_correction_plan_serializer.rs

## File: src/620_correction_plan_serializer.rs

- Layer(s): 620_correction_plan_serializer.rs
- Language coverage: Rust (4)
- Element types: Function (4)
- Total elements: 4

### Elements

- [Rust | Function] `serialize_correction_plan` (line 0, pub)
  - Signature: `pub fn serialize_correction_plan (plan : & CorrectionPlan , verification : & VerificationPolicy , rollback : & Rollba...`
- [Rust | Function] `serialize_correction_plans` (line 0, pub)
  - Signature: `pub fn serialize_correction_plans (report : & CorrectionIntelligenceReport ,) -> serde_json :: Value { let items = re...`
  - Calls: collect, map, zip, zip, iter, iter, iter, serialize_correction_plan
- [Rust | Function] `write_intelligence_outputs` (line 0, pub)
  - Signature: `pub fn write_intelligence_outputs (report : & CorrectionIntelligenceReport , output_dir : & Path ,) -> std :: io :: R...`
  - Calls: write_intelligence_outputs_at
- [Rust | Function] `write_intelligence_outputs_at` (line 0, pub)
  - Signature: `pub fn write_intelligence_outputs_at (report : & CorrectionIntelligenceReport , output_dir : & Path , correction_json...`
  - Calls: std::fs::create_dir_all, unwrap_or_else, map, to_path_buf, join, parent, std::fs::create_dir_all, serialize_correction_plans, std::fs::write, serde_json::to_string_pretty, unwrap_or_else, map, to_path_buf, join, parent, std::fs::create_dir_all, emit_verification_policy, Ok

