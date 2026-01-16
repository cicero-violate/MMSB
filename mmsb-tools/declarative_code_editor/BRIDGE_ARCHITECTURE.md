# Bridge Architecture

## Overview

The **Bridge Layer** connects `declarative_code_editor` (declarative query/mutation DSL) to MMSB's authority model, producing Delta and StructuralOp objects ready for commit.

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DECLARATIVE CODE EDITOR                       │
│                                                                  │
│  SourceBuffer (in-memory) → QueryPlan → MutationPlan           │
│  { path, content, ast }                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        BRIDGE LAYER                              │
│                                                                  │
│  ┌────────────────────┐                                         │
│  │ StructuralClassifier│                                        │
│  │                    │                                         │
│  │ Separate:          │                                         │
│  │ - Structural ops   │                                         │
│  │ - Page deltas      │                                         │
│  └────────────────────┘                                         │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │  BridgedOutput  │                                           │
│   │                 │                                           │
│   │ - intents       │ (TODO: extract from AST diff)            │
│   │ - page_deltas   │ ✅ Generated from source                │
│   │ - structural_ops│ (TODO: detect imports)                   │
│   │ - route         │ ✅ State/Structural/Both                │
│   └─────────────────┘                                           │
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

### 1. SourceBuffer (src/source.rs)

**Purpose**: In-memory source representation with cached AST

**Structure**:
```rust
pub struct SourceBuffer {
    pub path: PathBuf,       // File identification
    pub content: String,     // In-memory source
    ast: File,               // Cached parsed AST
}
```

**API**:
- `new(path, content)` - Parse and cache AST
- `ast()` - Get cached AST
- `source()` - Get source content
- `update(new_content)` - Update and reparse

**Aligned with**: `structural_code_indexer::SourceFile`

### 2. StructuralClassifier (src/bridge/structural_classifier.rs)

**Purpose**: Separate structural vs state changes

**Input**: 
- `EditIntent[]` - Semantic intents
- `PageID` - Current page
- `PathBuf` - File path
- `&str` - Source after mutation

**Output**:
- `Vec<Delta>` - Page deltas (STATE PIPELINE)
- `Vec<StructuralOp>` - Structural ops (STRUCTURAL PIPELINE)

**Current Implementation**:
- ✅ Builds Delta from source content
- ⚠️  StructuralOp generation is stub (TODO: detect imports)
- ⚠️  Intent-based classification not yet implemented

**Classification Rules**:

| Intent Type        | Category    | Output                |
|--------------------|-------------|-----------------------|
| RenameSymbol       | State       | Delta                 |
| DeleteSymbol       | State       | Delta                 |
| SignatureChange    | State       | Delta                 |
| ImportChange       | Structural  | StructuralOp::AddEdge |
| ModuleChange       | Both        | Delta + StructuralOp  |

### 3. BridgeOrchestrator (src/bridge/orchestrator.rs)

**Purpose**: End-to-end flow coordination

**API**:
```rust
pub fn execute_and_bridge(
    buffer: &mut SourceBuffer,
    mutation: &MutationPlan,
    page_id: PageID,
) -> Result<BridgedOutput, EditorError>
```

**Flow**:
1. Apply mutation to buffer (⚠️ TODO: actual implementation)
2. Extract semantic intent (⚠️ TODO: AST diffing)
3. Classify into structural/state
4. Build Delta from source
5. Return `BridgedOutput`

**Current Status**:
- ✅ Orchestration flow correct
- ⚠️  Step 1 (mutation) is no-op - needs span-based replacement
- ⚠️  Step 2 (intent) returns empty vec - needs AST diff

### 4. Output Types (src/bridge/output.rs)

**BridgedOutput**:
```rust
pub struct BridgedOutput {
    pub intents: Vec<EditIntent>,           // Currently empty
    pub page_deltas: Vec<Delta>,            // ✅ Generated
    pub structural_ops: Vec<StructuralOp>,  // Currently empty
    pub route: PipelineRoute,               // ✅ Working
}
```

**PipelineRoute**: `State | Structural | Both`

## Integration with MMSB

### ✅ What Works

1. **SourceBuffer** - In-memory like MMSB Pages
2. **Delta generation** - Produces valid Delta objects
3. **Pipeline routing** - Correctly identifies State vs Structural
4. **Authority boundaries** - Pure observation until commit

### ⚠️  What's TODO

1. **Intent extraction** - Currently returns empty vec
   - Need AST diffing (before/after comparison)
   - Detect renames, signature changes, imports

2. **Mutation application** - Currently no-op
   - Need span-based source transformation
   - Apply operations to SourceBuffer.content
   - Update cached AST

3. **StructuralOp generation** - Currently empty vec
   - Detect import statements
   - Detect module declarations
   - Generate AddEdge/RemoveEdge

4. **Propagation** - Removed from bridge
   - Propagation happens in MMSB core after commit
   - Not part of declarative editor

## Usage Example

```rust
use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

// Create in-memory buffer
let mut buffer = SourceBuffer::new(
    PathBuf::from("src/lib.rs"),
    source_code.to_string(),
)?;

// Build query + mutation
let query = QueryPlan::new()
    .with_predicate(KindPredicate::new(ItemKind::Function))
    .with_predicate(NamePredicate::new("old_name"));

let mutation = MutationPlan::new(query)
    .with_operation(ReplaceOp::new("sig.ident", "new_name"));

// Execute through bridge
let output = BridgeOrchestrator::execute_and_bridge(
    &mut buffer,
    &mutation,
    PageID(123),
)?;

// Route to appropriate pipeline
match output.route {
    PipelineRoute::State => {
        // STATE PIPELINE
        let judgment = obtain_state_judgment()?;
        commit_delta(&output.page_deltas[0], &judgment)?;
    }
    PipelineRoute::Structural => {
        // STRUCTURAL PIPELINE
        let judgment = obtain_structural_judgment()?;
        commit_structural_delta(&output.structural_ops, &judgment)?;
    }
    PipelineRoute::Both => {
        // Structural first, then state
        commit_structural_delta(&output.structural_ops, &s_judgment)?;
        commit_delta(&output.page_deltas[0], &p_judgment)?;
    }
}
```

## Critical Invariants

### ABSOLUTE RULE (from STRUCTURAL_vs_STATE_CHANGE)

> **Propagation may NEVER change the DAG.**
> **Structural commit may NEVER trigger propagation.**

**Bridge enforcement**:
1. ✅ StructuralClassifier separates ops from deltas
2. ✅ BridgedOutput keeps structural_ops and page_deltas separate
3. ✅ Caller controls commit order (structural → state)
4. ✅ No propagation in bridge layer

### Judgment Requirements

1. **Structural ops** → require `JudgmentToken` (structural)
2. **Page deltas** → require `JudgmentToken` (state)
3. **Propagation** → happens after commit in MMSB core

**Bridge does NOT issue judgment** - caller provides tokens.

### Pure Observation

The bridge layer remains **pure/observational**:
- ✅ No filesystem writes
- ✅ No commits
- ✅ No mutations of authority structures
- ✅ Only produces `BridgedOutput` for caller to commit

## TODO for Production

### High Priority
1. **Implement mutation application** (src/executor/mod.rs)
   - Span-based source transformation
   - Update SourceBuffer.content
   - Reparse AST

2. **Implement intent extraction** (src/intent/extraction.rs)
   - AST diff algorithm
   - Detect symbol renames
   - Detect signature changes

3. **Implement import detection** (src/bridge/structural_classifier.rs)
   - Parse use statements
   - Generate StructuralOp::AddEdge
   - Track module dependencies

### Medium Priority
4. Conflict detection
5. Error recovery
6. Performance optimization
7. Multi-file transactions

See TODO.md for complete roadmap.
