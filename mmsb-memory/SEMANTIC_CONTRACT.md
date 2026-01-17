# MMSB Memory - Semantic Contract

## Core Identity

**mmsb-memory = truth semantics only**

---

## Owns

### Truth Semantics
- Canonical truth authority
- Commit semantics (when mutations become fact)
- Invariant enforcement (structural + algebraic)
- Deterministic replay semantics
- Proof production: AdmissionProof (D), CommitProof (E), OutcomeProof (F)

### Logical Structures
- Page identifiers (PageID - logical only)
- Delta definitions (logical mutations)
- Epoch management (logical time)
- DAG structure (dependency graphs)
- Transaction log (tlog - replay history)
- Structural validation (cycle detection, graph integrity)

---

## Does NOT Own

### Hardware
- GPU memory allocation
- CUDA/device management
- Buffer pools
- Physical memory layout
- SIMD execution
- NCCL integration

### Runtime
- Scheduling
- Execution orchestration
- Performance optimization
- Throughput engines
- Queue management
- Fast paths

### External Systems
- Filesystem operations (delegates to mmsb-storage)
- Async runtime (no tokio, no threading)
- Network operations

---

## Proof Production Chain

```
D (AdmissionProof)
  ↓ verification of JudgmentProof (C)
  ↓ epoch validity
  ↓ replay protection

E (CommitProof)
  ↓ applied delta
  ↓ structural invariants
  ↓ state invariants
  ↓ deterministic ordering

F (OutcomeProof)
  ↓ commit success/failure
  ↓ final state witness
  ↓ knowledge derivation input
```

---

## Invariants

1. **Determinism**: `replay(memory, history) → same_state`
2. **Isolation**: Memory operates without executor/service/runtime
3. **Purity**: Proof production has no side effects
4. **Authority**: SOLE truth authority (no distributed consensus)
5. **Immutability**: Proofs are immutable once produced

---

## Dependencies (Allowed)

- `mmsb-proof` (proof structure definitions)
- `mmsb-authenticate` (proof verification)
- `mmsb-events` (EventSink for MemoryCommitted emission)
- `mmsb-storage` (sync persistence interface only)
- `serde`, `serde_json` (pure serialization)

**FORBIDDEN**: tokio, async runtimes, threading, filesystem, OS APIs

---

## Boundary with mmsb-executor

**Memory describes WHAT must happen**
- PropagationIntent (what must propagate)
- CommitIntent (what must commit)
- MaterializationSpec (what state becomes)

**Executor decides HOW it happens**
- GPU kernel execution
- Buffer management
- Queue scheduling
- Performance optimization
