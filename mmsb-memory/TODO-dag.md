# MMSB Memory Refactor - Dependency DAG

## Variables

Let $M_{semantic}$ = truth semantics components (KEEP in mmsb-memory)
Let $E_{substrate}$ = execution substrate components (MOVE to mmsb-executor)
Let $D_{numbered}$ = numbered directories (IGNORE/DELETE)

---

## Current State Analysis

### Dependencies (from Cargo.toml)
```
mmsb-memory dependencies:
├── mmsb-proof ✅ (CORRECT - semantic layer)
├── mmsb-authenticate ✅ (CORRECT - verification)
├── mmsb-storage ✅ (CORRECT - persistence)
└── mmsb-events ✅ (CORRECT - event emission)

NO VIOLATIONS: No tokio, async runtime, OS APIs detected
```

### mmsb-executor Current State
```
MINIMAL SKELETON:
- lib.rs (207 bytes)
- module.rs (2723 bytes) - ExecutorModule stub
- Dependencies: mmsb-proof, mmsb-events only

READY FOR SUBSTRATE MERGE
```

---

## Directory Classification

### $M_{semantic}$ - KEEP in mmsb-memory

| Directory          | Reason                       | Proofs              |
|--------------------+------------------------------+---------------------|
| `admission/`       | Gate logic for D proof       | AdmissionProof      |
| `commit/`          | Commit semantics for E proof | CommitProof         |
| `outcome/`         | Outcome witness for F proof  | OutcomeProof        |
| `proofs/`          | D/E/F proof builders         | All                 |
| `truth/`           | Canonical time, epochs       | -                   |
| `epoch/`           | Epoch management             | -                   |
| `dag/`             | Logical dependency graph     | -                   |
| `delta/`           | Logical mutation deltas      | -                   |
| `page/`            | Logical page identifiers     | -                   |
| `tlog/`            | Transaction log replay       | -                   |
| `replay/`          | Deterministic replay         | -                   |
| `semiring/`        | Algebraic invariants         | -                   |
| `structural/`      | Graph validation, cycles     | -                   |
| `materialization/` | Pure view construction       | KEEP (defines WHAT) |

### $E_{substrate}$ - MOVE to mmsb-executor

| Directory       | Reason                                  | Components  |
|-----------------+-----------------------------------------+-------------|
| `physical/`     | GPU pools, NCCL, allocator stats        | Hardware    |
| `device/`       | Device registry, host-device sync, SIMD | Hardware    |
| `propagation/`  | Engines, queues, buffers, orchestrators | Runtime     |
| `optimization/` | Locality, clustering, layout            | Performance |

### $D_{numbered}$ - IGNORE/DELETE

All `__<n>_*` directories are non-semantic markers per TODO-answers.md

---

## Refactor DAG (Execution Order)

### Phase 0: Pre-Flight ✅
- [x] Read canonical docs
- [x] Read TODO.md
- [x] Read TODO-answers.md
- [x] Dependency audit
- [x] Create TODO-dag.md (this file)

### Phase 1: Semantic Contract (TODO-1) ✅
**Goal**: Lock memory's boundaries before any moves

```
T1.1: Create mmsb-memory/SEMANTIC_CONTRACT.md
├── Define: truth semantics scope
├── Define: D/E/F proof production
├── Define: NOT owned (hardware, scheduling, runtime)
└── Explicit invariants

T1.2: Document memory_engine.rs authority
├── Comment MemoryEngine as truth authority
└── Document proof production chain D→E→F
```

**Output**: Clear boundary definition
**Status**: ✅ Complete

---

### Phase 2: Rename Module to Engine (TODO-2) ✅
**Goal**: Clarify memory is not a runtime participant

```
T2.1: Rename type
├── memory_engine.rs: MemoryModule → MemoryEngine
├── Update constructor
└── Update all internal method references

T2.2: Rename file
└── module.rs → memory_engine.rs

T2.3: Update lib.rs exports
└── pub use memory_engine::MemoryEngine

T2.4: Audit external references (DEFER to later)
└── Check mmsb-service, mmsb-learning references
```

**Dependency**: None  
**Output**: MemoryEngine replaces MemoryModule
**Status**: ✅ Complete

---

### Phase 3: Tag Substrate Directories (TODO-3)
**Goal**: Mark what will move, prevent new deps

```
T3.1: Add SUBSTRATE_MARKER.md to each:
├── physical/SUBSTRATE_MARKER.md
├── device/SUBSTRATE_MARKER.md
├── propagation/SUBSTRATE_MARKER.md
└── optimization/SUBSTRATE_MARKER.md

Content: "Execution substrate - to be moved to mmsb-executor"
```

**Dependency**: None  
**Output**: Clear visual markers

---

### Phase 4: Extract Substrate to Executor (TODO-4)
**Goal**: Physical relocation without code changes

```
T4.1: Move directories verbatim
├── mmsb-memory/src/physical/ → mmsb-executor/src/physical/
├── mmsb-memory/src/device/ → mmsb-executor/src/device/
├── mmsb-memory/src/propagation/ → mmsb-executor/src/propagation/
└── mmsb-memory/src/optimization/ → mmsb-executor/src/optimization/

T4.2: Update mmsb-executor/src/lib.rs
├── pub mod physical;
├── pub mod device;
├── pub mod propagation;
└── pub mod optimization;

T4.3: Remove from mmsb-memory/src/lib.rs
└── Delete substrate module declarations

T4.4: Fix broken imports (will exist)
├── Identify files importing substrate
└── DEFER fixes to Phase 5
```

**Dependency**: Phase 3 complete  
**Output**: Substrate physically relocated

---

### Phase 5: Split What/How Interfaces (TODO-5)
**Goal**: Define semantic boundary between memory and executor

```
T5.1: Define memory interfaces (KEEP)
├── PropagationIntent (what must propagate)
├── CommitIntent (what must commit)
└── MaterializationSpec (what state becomes)

T5.2: Implement in executor (NEW)
├── GPU kernel execution
├── Buffer management
├── Queue scheduling
└── Fast paths

T5.3: Create mmsb-executor interface traits
├── ExecuteSubstrate trait
└── ExecutionPlan → ExecutionResult contract
```

**Dependency**: Phase 4 complete  
**Output**: Clean what/how separation

---

### Phase 6: Purge Allocation from Memory (TODO-6)
**Goal**: Remove hardware concerns from semantic layer

```
T6.1: Audit page/ and delta/
├── Identify: allocator.rs (MOVE if exists)
├── Identify: lockfree_allocator.rs (MOVE if exists)
├── Identify: host_device_sync.rs (ALREADY MOVED in Phase 4)
└── Identify: SIMD/buffer code (ALREADY MOVED in Phase 4)

T6.2: Keep in memory:
├── PageID (logical identifier)
├── Page metadata (semantic properties)
├── Delta (logical mutation)
└── Invariant checks

T6.3: Move to executor:
└── Any physical allocation code found
```

**Dependency**: Phase 4 complete  
**Output**: Memory is allocation-free

---

### Phase 7: Lock Replay Semantics (TODO-7)
**Goal**: Ensure replay is pure and deterministic

```
T7.1: Audit replay/ directory
├── Verify: takes sealed events as input ✓
├── Verify: produces deterministic outcomes ✓
├── Verify: does NOT emit events during replay ✓
└── Verify: does NOT touch storage directly ✓

T7.2: Add replay invariant test
└── Test: replay(memory, history) → same_state

T7.3: Document replay contract
└── Comment: "Replay is pure, no side effects"
```

**Dependency**: None (can run parallel)  
**Output**: Replay verified as pure

---

### Phase 8: Runtime Import Audit (TODO-8)
**Goal**: Verify zero runtime dependencies

```
T8.1: Check Cargo.toml
├── ✅ No tokio
├── ✅ No async runtimes
├── ✅ No threading primitives
└── ✅ No filesystem APIs

T8.2: Allowed dependencies check
├── ✅ mmsb-proof
├── ✅ mmsb-authenticate
├── ✅ mmsb-events (EventSink only)
├── ✅ mmsb-storage (interface only)
└── ✅ serde, serde_json (pure)

T8.3: Final verification
└── cargo tree | grep -E "(tokio|async|thread)" → empty
```

**Dependency**: Phase 4 complete  
**Output**: Zero runtime violations confirmed

---

### Phase 9: Create MemoryView for Learning (TODO-9)
**Goal**: Read-only interface to prevent back-edges

```
T9.1: Define MemoryView trait
pub trait MemoryView {
    fn get_outcome_proof(&self, hash: Hash) -> Option<&OutcomeProof>;
    fn get_commit_proof(&self, hash: Hash) -> Option<&CommitProof>;
    fn get_admission_proof(&self, hash: Hash) -> Option<&AdmissionProof>;
    fn get_committed_facts(&self, epoch: Epoch) -> Vec<CommittedFact>;
    fn get_replay_snapshot(&self, epoch: Epoch) -> ReplaySnapshot;
}

T9.2: Implement MemoryView for MemoryEngine
└── Read-only accessors only

T9.3: Update dependency graph
└── mmsb-learning depends on MemoryView, not MemoryEngine
```

**Dependency**: Phase 2 complete  
**Output**: Safe read-only memory access

---

### Phase 10: Final Invariant Check (TODO-10)
**Goal**: Verify memory can operate standalone

```
T10.1: Create standalone test
└── tests/memory_standalone.rs

T10.2: Test operations
├── Instantiate MemoryEngine in test
├── Replay history offline
├── Verify invariants
└── Produce proofs D/E/F

T10.3: WITHOUT dependencies on
├── ✗ mmsb-executor
├── ✗ mmsb-service
└── ✗ mmsb-storage (only interface, not impl)

T10.4: Success criteria
└── All tests pass in isolation
```

**Dependency**: All previous phases  
**Output**: Memory is semantically complete

---

## Critical Path Analysis

```
Sequential (must follow order):
Phase 1 → Phase 2 → Phase 9
Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 8

Parallel (can run alongside):
Phase 7 (anytime after Phase 1)

Final gate:
Phase 10 (requires all)
```

---

## Rollback Safety

Each phase is git-committable independently:

```
Phase 1: git commit -m "doc: define memory semantic contract"
Phase 2: git commit -m "refactor: MemoryModule → MemoryEngine"
Phase 3: git commit -m "doc: tag substrate directories for migration"
Phase 4: git commit -m "refactor: move substrate to mmsb-executor"
Phase 5: git commit -m "refactor: split what/how interfaces"
Phase 6: git commit -m "refactor: purge allocation from memory"
Phase 7: git commit -m "test: verify replay purity"
Phase 8: git commit -m "verify: zero runtime dependencies"
Phase 9: git commit -m "feat: add MemoryView for learning"
Phase 10: git commit -m "test: verify memory standalone operation"
```

---
