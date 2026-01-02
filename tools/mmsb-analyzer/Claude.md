# Constraint-Induced Program Transformation (CIPT)

This repository follows the CIPT process: execute safely, harvest constraints, encode them as law, then shift to admission-first prediction.

## Operating Principles
- Execution is a confirmation tool, not a discovery tool.
- All mutations must be gated by admission intelligence.
- Constraints are extracted from observed failures and encoded as deterministic rules.
- Verification must be clean to certify admission behavior.

## Current Canonical Flow
1. Generate correction intelligence outputs.
2. Run admission preflight (action-level feasibility).
3. Run batch-level admission (`admit_batch`).
4. `admission_composition.json` is the authoritative execution gate.
5. Executor may run only if `AdmissionDecision == Admissible`.

## Admission-First Rules
- Preflight predicts per-action feasibility (necessary but not sufficient).
- Batch-level admission provides compositional proof.
- Execution must not run unless batch admission is admissible.
- `admission_composition.json` artifact is the authority for execution decisions.

## Constraint Handling
- New constraints are logged in `docs/97_correction_intelligence/constraint_harvest_log.md`.
- Encode constraints as deterministic admission or enforcement rules.
- Re-run the same probe only after encoding the constraint.

## Phase 5 Posture
- MoveToLayer semantics are frozen.
- Batch scaling is allowed only after admission intelligence confirms eligibility.
- Discovery lives in admission intelligence improvements, not forced execution.

## PHASE 6.5 Equilibrium (Architectural Stopping Point)
- **Status**: Complete, FROZEN, and HARDENED (2026-01-01)
- **Achievement**: Deterministic, compositional program transformation calculus
- **Key Properties**:
  - Predictive (not reactive)
  - Proof-driven (not heuristic)
  - Compositional (batch-level admission via symbolic fold)
  - Auditable (durable admission_composition.json artifacts)
- **Separation of Concerns**:
  - Admission = Feasibility + Proof (mmsb-analyzer)
  - Execution = Mutation only (mmsb-executor)
  - Verification = Confirmation (cargo check)
- **Hardening Complete**:
  - Modules frozen (architectural freeze comments added)
  - CI gates enforced (3/3 tests passing)
  - Execution precondition documented
  - Stopping point documented
- **No changes without explicit architectural intent**
- **PHASE 7+ is optional**: Represents autonomy escalation, not correctness requirement

## Category 1 Wiring (Complete)
- **Status**: COMPLETE (2026-01-01)
- **Implementation**: Mechanical integration of PHASE 6.5 into workflow
- **Changes**: Zero new semantics, pure enforcement wiring
- **Task 1.1**: Execution precondition enforced in `run_executor.sh` ✅
- **Task 1.2**: Batch admission wired into `run_analysis.sh` ✅
- **Result**: Self-enforcing system (execution gated by admission)

## Status Snapshot (Aligned With Cluster_progress.txt)
- Phase 4: Visibility & Re-exports: Completed.
- Phase 5: Batch Automation: In Progress.
  - Phase 2 Cluster Extraction (B1 baseline): Closed (25 actions).
  - Action-level feasibility gates operational.
  - Batch-level admission (PHASE 6.5) is the execution authority.
- Phase 6.5: Admission Intelligence Formalization: **COMPLETE & FROZEN**.
  - Static Effect Signature: Schema v0.1.0 (locked).
  - Composition Rule (Abort-First): Deterministic fold operational.
  - admission_composition.json Artifact: Durable proof serialization.
  - Batch-Level Admission Predicate: Single admission gate (admit_batch).
  - Proof Surface: Complete, auditable, CI-consumable.

## Artifact Index
- Overall phase tracker: `Cluster_progress.txt`.
- Action-level feasibility: `docs/97_correction_intelligence/admission_preflight.json`.
- Batch-level admission (authoritative): `admission_composition.json` (generated at runtime).
- Constraint log: `docs/97_correction_intelligence/constraint_harvest_log.md`.
- Batch slices: `docs/97_correction_intelligence/slice_cluster_*`.
- PHASE 6.5 artifacts:
  - Effect signature schema: `src/effect_signature_schema.rs` (FROZEN)
  - Composition rule: `src/composition_rule.rs` (FROZEN)
  - Artifact generation: `src/admission_composition_artifact.rs` (FROZEN)
  - Batch admission: `src/batch_admission.rs` (FROZEN)
  - Compositional proof: `admission_composition.json` (generated at runtime)
  - CI gate tests: `tests/ci_admission_gate.rs` (3/3 passing)
  - Stopping point: `docs/PHASE_6_5_STOPPING_POINT.md`
  - Execution precondition: `docs/EXECUTION_PRECONDITION.md`
  - Validation summary: `docs/PHASE_6_5_VALIDATION_SUMMARY.md`
- Category 1 artifacts:
  - Admission runner: `examples/run_batch_admission.rs`
  - Wiring documentation: `docs/CATEGORY_1_WIRING_COMPLETE.md`
  - Analysis workflow: `run_analysis.sh` (admission wired in)
  - Execution workflow: `run_executor.sh` (precondition enforced)
