# structural_code_editor

This tool is **pure/observational**. It does not apply patches or write to the filesystem.
It observes a filesystem change and produces authoritative **state + structure** deltas
that must be committed by the caller under judgment.

## Full Trip (end-to-end flow)

The editor produces a **full trip** when it executes the following sequence:

1) **Load snapshots (before/after)**
   - Files: `mmsb-tools/structural_code_editor/src/lib.rs`
   - Input: `RepoSnapshot` for `before` and `after` (read from disk by caller)
   - Output: in-memory buffers only

2) **Read source files (indexer input)**
   - Files: `mmsb-tools/structural_code_indexer/src/fs/reader.rs`
   - Produces `SourceFile[]` for each snapshot

3) **Scan and index snapshots**
   - Files: `mmsb-tools/structural_code_editor/src/scan.rs`
   - Uses indexer extraction:
     - `mmsb-tools/structural_code_indexer/src/extract/rust.rs`
     - `mmsb-tools/structural_code_indexer/src/graph/build.rs`
   - Builds deterministic `DependencyGraph` from extracted references

4) **Validate active DAG snapshot**
   - Files: `mmsb-tools/structural_code_editor/src/diff.rs`
   - Ensures the provided `active_dag` matches the `before` snapshot

5) **Compute repo diffs**
   - Files: `mmsb-tools/structural_code_editor/src/diff.rs`
   - Outputs `RepoDiff` with file changes and `StructuralOp[]`

6) **Map file diffs to page deltas**
   - Files: `mmsb-tools/structural_code_editor/src/map.rs`
   - Produces `Delta[]` with full payloads (no patch text)

7) **Index semantic symbols (intent source)**
   - Files: `mmsb-tools/structural_code_editor/src/propagation/index.rs`
   - Builds `PageIndex` (exports/imports/references) per page

8) **Extract edit intent**
   - Files: `mmsb-tools/structural_code_editor/src/propagation/intent.rs`
   - Produces `EditIntent[]` from symbol diffs

9) **Propagate semantic effects**
   - Files: `mmsb-tools/structural_code_editor/src/propagation/propagate.rs`
   - Traverses DAG dependents, rewrites sources, emits **derived** `Delta[]`
   - Requires `JudgmentToken` for emission
   - Judgment token source: `mmsb-judgment/src/issue.rs`

10) **Commit state deltas (authority boundary)**
    - Files: `mmsb-core/src/01_page/page_commit.rs`, `mmsb-core/src/01_page/tlog.rs`
    - `commit_delta` appends to `TransactionLog` (state authority boundary)

11) **Commit structural ops (authority boundary)**
    - Files: `mmsb-core/src/03_dag/dag_commit.rs`, `mmsb-core/src/03_dag/dag_log.rs`
    - `commit_structural_delta` persists ops before apply (structural authority boundary)

12) **Propagation over DAG snapshot (state pipeline)**
    - Files: `mmsb-core/src/04_propagation/propagation_engine.rs`,
      `mmsb-core/src/04_propagation/tick_orchestrator.rs`
    - Uses the committed DAG snapshot for downstream recomputation

## Authority & Judgment

- **Judgment token** lives in `mmsb-judgment/src/types.rs` and is issued by
  `mmsb-judgment/src/issue.rs`. It is opaque and must be provided by the caller.
- Structural authority enters at `commit_structural_delta` (core) **after** persistence.
- State authority enters at `commit_delta` (core) **after** TLog append.
- This tool never commits or mutates authority; it only produces deltas and ops.
- Propagation emits derived deltas only when a **JudgmentToken** is provided.

## Example

Run the example to see a rename intent and propagated rewrite:

```bash
cargo run -p structural_code_editor --example analyze_code_edit
```
