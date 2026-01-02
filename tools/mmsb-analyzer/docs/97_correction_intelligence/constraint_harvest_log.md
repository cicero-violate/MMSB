Constraint Harvest Log
======================

Move: move_scan_doc_comments_to_src/211_dead_code_attribute_parser.rs
--------------------------------------------------------------------

Instrumentation
---------------
- Imports added/removed: none added; new target file created without required `use` items.
- Visibility changes implied but not applied: `scan_doc_comments` moved into a new module file with no `mod` declaration or re-export.
- Re-export pressure: existing call sites in `src/230_dead_code_attribute_parser.rs` still expect `scan_doc_comments` in-module.
- Test breakage vectors: executor verification failed (`cargo check`/`cargo test` exit 101) after move.
- Shim necessity: yes; preserve API via re-export or wrapper in `dead_code_attribute_parser`.

Observed Constraints (C)
------------------------
- C1: Import repair required for moved function (HashMap/Path/IntentMarker + helper fns).
- C2: Module declaration or visibility update required to expose new file.
- C3: Re-export or call-site updates required for in-module references.
- C4: Verification failure triggered by unresolved symbols/imports after move.

Move: move_action_function_to_src/640_correction_intelligence_report.rs
-----------------------------------------------------------------------

Instrumentation
---------------
- Imports added/removed: shim created in `src/570_correction_plan_generator.rs` for moved helper.
- Visibility changes implied but not applied: moved helper remained private in new module, breaking re-export.
- Re-export pressure: `pub use moved_action_function::action_function` failed due to private item.
- Test breakage vectors: `cargo check`/`cargo test` failed after move; rollback applied.
- Shim necessity: yes; shim must re-export a public item or update call sites.

Observed Constraints (C)
------------------------
- C5: MoveToLayer must preserve cross-module visibility for moved helpers; private items cannot be re-exported without adjusting visibility or call sites, and may cause type/module shadowing errors during verification.

Probe Outcome (C5 Enforcement)
------------------------------
- C5 enforcement added: moved helpers are promoted to `pub(super)` when re-export shim is required.
- Result: privacy errors cleared, but compile still fails due to module duplication.

Observed Constraints (C)
------------------------
- C6: Re-export shim using `#[path = "<target file>"] mod moved_*` creates a second module scope for the entire target file, causing type identity collisions (e.g., duplicate `CorrectionIntelligenceReport`) when the target module defines shared types.

Probe Outcome (C6 Enforcement, Identity-Preserving Shim)
--------------------------------------------------------
- C6 enforcement applied: existing module targets now use `use crate::<module_path>::item;` without `#[path]`.
- Probe still fails: duplicate type identities remain due to pre-existing `#[path]` shims that reference the same target file from other modules.

Observed Constraints (C)
------------------------
- C7: Legacy `#[path]` shims pointing at already-module files must be migrated to direct module-path re-exports before further MoveToLayer into that target file can succeed.

Probe Outcome (C7 Enforcement, Legacy Shim Migration)
-----------------------------------------------------
- C7 enforcement added: legacy `#[path]` shims targeting the moved file are rewritten to direct module-path re-exports.
- Result: probe still fails; `action_function` becomes unresolved in `src/570_correction_plan_generator.rs` after the move, indicating missing in-scope import or visibility.

Observed Constraints (C)
------------------------
- C8: MoveToLayer must ensure the source module regains in-scope access to moved helpers (direct `use` shim or call-site update) with correct visibility, or compilation fails with unresolved symbols.

Probe Outcome (C8 Enforcement, Reachability Shim)
-------------------------------------------------
- C8 enforcement added: source modules regain explicit `use` reachability to moved helpers and target visibility is widened only when needed.
- Result: probe passes with MoveToLayer + invariants intact.

Move: slice_cluster_b1 (25-item batch)
--------------------------------------

Observed Constraints (C)
------------------------
- C9: EnsureImports must avoid self-imports into the target module and must import symbols that are declared in the source module but referenced by the moved function (visibility-aware resolution).

Probe Outcome (C9 Enforcement, Batch Re-run)
--------------------------------------------
- C9 enforcement added: self-import guard + source-declared symbol import resolution.
- Result: batch re-run improved but still fails due to dependency-plan ordering and warning-based invariant failures.

Observed Constraints (C)
------------------------
- C10: Batch ordering/import hygiene must prevent warning-only failures (unused imports introduced by intermediate moves) or the zero-warning invariant will abort remaining moves.

Probe Outcome (C10 Enforcement, Batch Re-run)
---------------------------------------------
- C10 enforcement added: deterministic unused-import pruning after each MoveToLayer (touched files only, re-export shims preserved).
- Result: warning-based failures cleared; remaining failures are dependency ordering (C11).

Observed Constraints (C)
------------------------
- C11: Batch execution order must respect intra-batch helper dependencies so that helpers are moved before primary functions that call them.

Probe Outcome (C11 Enforcement, Batch Re-run)
---------------------------------------------
- C11 enforcement added: action ordering now includes identifier-based helper dependencies (MoveToLayer symbols referenced by other MoveToLayer functions).
- Result: full 25-action batch applied with zero failures under all invariants.

Move: slice_cluster_n2 (guarded N=2 batch)
------------------------------------------

Observed Constraints (C)
------------------------
- C12: MoveToLayer must be blocked unless all EnsureImports actions can resolve the moved function signature in the destination module graph after the move.

Probe Outcome (C12 Observation, N=2 Composition)
-----------------------------------------------
- Result: N=2 batch failed with EnsureImports signature resolution errors, indicating a compositional precondition gap (not a gate/invariant failure).

Probe Outcome (C12 Enforcement, Pre-flight Gate)
-----------------------------------------------
- C12 enforcement added: EnsureImports blocked when MoveToLayer is skipped by pre-flight guards, preserving admission correctness.
- Result: pending N=2 rerun to validate compositional eligibility gate.

Probe Outcome (C12 Validation, N=2 Admission Gate)
--------------------------------------------------
- Result: MoveToLayer actions blocked at admission; no diffs emitted (0 file diffs).
- Note: verification failed due to existing compile errors in src/330_report.rs (missing imports/types), not due to mutation.

Probe Outcome (C12 Validation, Clean Baseline)
---------------------------------------------
- Result: N=2 rerun completed with cargo check/test passing; no diffs emitted (0 file diffs).
- Admission gate still blocks infeasible MoveToLayer actions before mutation.
