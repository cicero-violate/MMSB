# Bridge Architecture

## Overview

The **Bridge Layer** connects `declarative_code_editor` (declarative query/mutation DSL) to `structural_code_editor` (MMSB authority model) and the dual-pipeline commit system.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECLARATIVE CODE EDITOR                       │
│                                                                  │
│  QueryPlan → MutationPlan → UpsertEngine → EditBuffer (pure)    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BRIDGE LAYER                              │
│                                                                  │
│  ┌──────────────────┐  ┌────────────────────┐                   │
│  │  IntentBridge    │  │ StructuralClassifier│                  │
│  │                  │  │                    │                   │
│  │ Extract semantic │  │ Separate:          │                   │
│  │ intent from AST  │  │ - Structural ops   │                   │
│  │ diffs            │  │ - Page deltas      │                   │
│  └──────────────────┘  └────────────────────┘                   │
│                             │                                    │
│                             ▼                                    │
│                    ┌─────────────────┐                           │
│                    │  BridgedOutput  │                           │
│                    │                 │                           │
│                    │ - intents       │                           │
│                    │ - page_deltas   │                           │
│                    │ - structural_ops│                           │
│                    │ - route         │                           │
│                    └─────────────────┘                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│  STRUCTURAL PIPELINE     │  │   STATE PIPELINE         │
│                          │  │                          │
│  structural_ops          │  │  page_deltas             │
│       ↓                  │  │       ↓                  │
│  ShadowGraph             │  │  JudgmentToken           │
│       ↓                  │  │       ↓                  │
│  Validate                │  │  commit_delta            │
│       ↓                  │  │       ↓                  │
│  JudgmentToken           │  │  TLog append             │
│       ↓                  │  │       ↓                  │
│  commit_structural_delta │  │  Snapshot DAG (read)     │
│       ↓                  │  │       ↓                  │
│  DAG updated             │  │  PropagationEngine       │
│                          │  │       ↓                  │
│                          │  │  Derived deltas          │
└──────────────────────────┘  └──────────────────────────┘
```

## Components

### 1. IntentBridge

**Purpose**: Extract semantic intent from declarative mutations

**Input**: 
- `PlannedEdit[]` - matched AST nodes + operations
- `EditBuffer` - buffer with before/after states

**Output**:
- `EditIntent[]` - semantic intents (rename, delete, signature change, import change)

**Algorithm**:
1. For each planned edit
2. Parse before state (from buffer tree)
3. Parse after state (from edit.new_text)
4. Compare AST nodes to extract intent
5. Return all extracted intents

**Example**:
```rust
// Planned edit: rename function "foo" → "bar"
IntentBridge::extract_from_planned(&planned, &buffer)
// → [EditIntent::RenameSymbol { old: "foo", new: "bar" }]
```

### 2. StructuralClassifier

**Purpose**: Classify edits into structural vs state changes

**Input**:
- `EditIntent[]` - semantic intents
- `Edit[]` - buffer edits
- `PageID` - current page
- `PathBuf` - file path
- `&str` - source after edits

**Output**:
- `Vec<Delta>` - page deltas (STATE PIPELINE)
- `Vec<StructuralOp>` - structural ops (STRUCTURAL PIPELINE)

**Classification Rules**:

| Intent Type        | Category    | Output                |
|--------------------|-------------|-----------------------|
| RenameSymbol       | State       | Delta                 |
| DeleteSymbol       | State       | Delta                 |
| AddSymbol          | State       | Delta                 |
| SignatureChange    | State       | Delta                 |
| ImportChange       | Structural  | StructuralOp::AddEdge |
| ModuleChange       | Both        | Delta + StructuralOp  |

**Example**:
```rust
// Intent: ImportChange { path: "std::collections", added: true }
StructuralClassifier::classify(&intents, &edits, page_id, &path, source)
// → ([], [StructuralOp::AddEdge { from: page_id, to: target_id, edge_type: Data }])
```

### 3. PropagationBridge

**Purpose**: Trigger propagation using structural_code_editor engine

**Input**:
- `PageID` - root page that changed
- `EditIntent[]` - semantic intents
- `DependencyGraph` - snapshot (read-only)
- `JudgmentToken` - authority to emit derived deltas

**Output**:
- `Vec<Delta>` - propagated deltas for dependent pages

**Algorithm**:
1. Convert declarative EditIntent → structural EditIntent
2. Call `structural_code_editor::propagate_edits()`
3. Extract deltas from propagation results
4. Return derived deltas

**Example**:
```rust
// Intent: RenameSymbol { old: "foo", new: "bar" }
// DAG: page_a → page_b (depends on foo)
PropagationBridge::propagate(page_a, &intents, &graph, &judgment)
// → [Delta for page_b with foo→bar rewrite]
```

### 4. BridgeOrchestrator

**Purpose**: Orchestrate full flow from declarative edit to MMSB pipelines

**Main API**:

#### `execute_and_bridge()`

**Signature**:
```rust
pub fn execute_and_bridge(
    plan: &MutationPlan,
    buffer: &mut EditBuffer,
    page_id: PageID,
    file_path: &PathBuf,
    allow_empty: bool,
    allow_multiple: bool,
) -> Result<BridgedOutput, EditorError>
```

**Flow**:
1. Plan mutations from query
2. Extract semantic intent
3. Apply edits to buffer
4. Classify into structural/state
5. Return `BridgedOutput`

#### `execute_with_propagation()`

**Signature**:
```rust
pub fn execute_with_propagation(
    plan: &MutationPlan,
    buffer: &mut EditBuffer,
    page_id: PageID,
    file_path: &PathBuf,
    graph: &DependencyGraph,
    judgment: &JudgmentToken,
    allow_empty: bool,
    allow_multiple: bool,
) -> Result<BridgedOutputWithPropagation, EditorError>
```

**Flow**:
1. Execute and bridge (as above)
2. **IF** state changes exist:
   - Trigger propagation engine
   - Collect derived deltas
3. Return `BridgedOutputWithPropagation`

## Pipeline Routing

```rust
pub enum PipelineRoute {
    State,       // commit_delta only
    Structural,  // commit_structural_delta only
    Both,        // structural first, then state
}
```

**Routing Logic**:

```rust
fn determine_route(intents: &[EditIntent]) -> PipelineRoute {
    let has_state = intents has State or Both category
    let has_structural = intents has Structural or Both category
    
    match (has_state, has_structural) {
        (true, true)   => Both,
        (true, false)  => State,
        (false, true)  => Structural,
        (false, false) => State, // default
    }
}
```

## Usage Example

```rust
use declarative_code_editor::*;
use mmsb_core::types::PageID;
use std::path::PathBuf;

// Setup
let source = "fn old_name() { }";
let mut buffer = EditBuffer::new(source.to_string());
let page_id = PageID(123);
let file_path = PathBuf::from("src/lib.rs");

// Build query + mutation
let query = QueryPlan::new()
    .with_predicate(ItemKind::Function)
    .with_predicate(NamePredicate::new("old_name"));

let mutation = MutationPlan::new(query)
    .with_operation(ReplaceOp::new("sig.ident", "new_name"));

// Execute through bridge
let output = BridgeOrchestrator::execute_and_bridge(
    &mutation,
    &mut buffer,
    page_id,
    &file_path,
    false,
    false,
)?;

// Check routing
match output.route {
    PipelineRoute::State => {
        // commit_delta(output.page_deltas, judgment)
        // → TLog
        // → propagation
    }
    PipelineRoute::Structural => {
        // commit_structural_delta(output.structural_ops, judgment)
        // → DAG updated
    }
    PipelineRoute::Both => {
        // 1. commit_structural_delta first
        // 2. commit_delta second
    }
}
```

## Critical Invariants

### ABSOLUTE RULE (from STRUCTURAL_vs_STATE_CHANGE)

> **Propagation may NEVER change the DAG.**
> **Structural commit may NEVER trigger propagation.**

**Bridge enforcement**:
1. `StructuralClassifier` separates ops from deltas
2. `BridgedOutput` keeps structural_ops and page_deltas separate
3. `PropagationBridge` only operates on STATE intents
4. Caller controls commit order (structural → state)

### Judgment Requirements

1. **Structural ops** → require `JudgmentToken` (structural)
2. **Page deltas** → require `JudgmentToken` (state)
3. **Propagation** → requires `JudgmentToken` (for derived deltas)

**Bridge does NOT issue judgment** - caller provides tokens.

### Pure Observation

The bridge layer remains **pure/observational**:
- No filesystem writes
- No commits
- No mutations of authority structures
- Only produces `BridgedOutput` for caller to commit

## Integration with Existing MMSB

### structural_code_editor

**Used by bridge**:
- `propagation::propagate_edits()` - for DAG traversal + rewriting
- `EditIntent` type - for intent conversion
- `PageIndex` - for matching intents to dependents

**Not used**:
- `scan_repo()` - bridge operates on single buffer
- `diff_repo()` - bridge extracts intent from AST diff
- `map_edit()` - bridge classifies directly

### mmsb-core

**Used by bridge**:
- `Delta` - state changes
- `StructuralOp` - DAG changes
- `DependencyGraph` - read-only for propagation
- `PageID`, `DeltaID`, `Epoch` - identity types

**Authority boundaries** (caller responsibility):
- `commit_delta()` - state authority
- `commit_structural_delta()` - structural authority

### mmsb-judgment

**Used by bridge**:
- `JudgmentToken` - required for propagation

**Issuance** (caller responsibility):
- Bridge never issues tokens
- Caller obtains from `mmsb-judgment::issue`

## Future Extensions

1. **Multi-file support**: Extend to operate on repository snapshots
2. **Conflict resolution**: Merge bridge with `structural_code_editor::diff`
3. **Incremental intent**: Cache intent extraction results
4. **Custom propagation**: User-defined rewrite rules
5. **Dry-run propagation**: Preview derived changes without commit
