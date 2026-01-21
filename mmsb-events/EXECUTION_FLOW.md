# MMSB Execution Flow

## Core Principle

**The executor NEVER mutates canonical state. Only MemoryEngine mutates.**

## Variables

- $J$ = `JudgmentProof` (authority approval witness)
- $E_r$ = `ExecutionRequest` (prepared work description)
- $E_p$ = `ExecutionProof` (execution completion witness)
- $\Delta$ = `Delta` (state change descriptor)
- $\Delta'$ = `NormalizedDelta` (validated + normalized state change)
- $P$ = `PropagationEngine` (dependency derivation engine)
- $M$ = `MemoryEngine` (canonical truth owner)
- $A$ = `AdmissionProof` (permission to mutate)
- $C$ = `CommitProof` (mutation complete witness)
- $O$ = `OutcomeProof` (validation result witness)

## Flow Equations

```
(1)  J → Executor → (E_p, Δ_proposed)
(2)  (E_p, Δ_proposed) → Propagation → Δ'
(3)  (J, Δ') → Memory.admit() → A
(4)  (A, Δ') → Memory.commit() → C  
(5)  C → Memory.record_outcome() → O
```

## Detailed Flow

### Phase 1: Execution (mmsb-executor)

**Authority**: NONE
**Location**: `mmsb-executor/src/execution_loop.rs`

```rust
impl ExecutionLoop {
    pub fn execute(&mut self, judgment: &JudgmentProof) -> ExecutionOutcome {
        // Perform side effects: file IO, device operations, compute
        // Produce:
        // - ExecutionProof (witnesses execution occurred)
        // - ProposedDelta (describes what WOULD change)
        
        // CRITICAL: No mutation of canonical state
    }
}
```

**Produces**:
- `ExecutionProof`: witnesses that execution occurred
- `ProposedDelta`: proposed state change (not yet applied)

**Does NOT**:
- Mutate MMSB canonical state
- Approve or deny work
- Validate deltas

### Phase 2: Propagation (mmsb-propagation)

**Authority**: NONE
**Location**: `mmsb-propagation/src/delta_normalizer.rs`

```rust
impl DeltaNormalizer {
    pub fn normalize(
        page_id: PageID,
        epoch: Epoch,
        payload: Vec<u8>,
    ) -> NormalizedDelta {
        // Pure transformation:
        // - Type checking
        // - Schema validation
        // - Format normalization
        
        // Derive secondary deltas from dependency graph
        
        // CRITICAL: No mutation, pure derivation
    }
}
```

**Produces**:
- `NormalizedDelta`: validated, normalized delta
- Secondary deltas from dependency propagation

**Does NOT**:
- Mutate canonical state
- Approve deltas
- Apply changes

### Phase 3: Admission (mmsb-memory)

**Authority**: Truth Verification
**Location**: `mmsb-memory/src/memory_engine.rs::admit_execution()`

```rust
impl MemoryEngine {
    pub fn admit_execution(
        &self,
        judgment_proof: &JudgmentProof,
    ) -> Result<AdmissionProof, AdmissionError> {
        // Verify JudgmentProof validity
        // Check epoch staleness
        // Replay protection
        
        // Produce AdmissionProof (permission to mutate)
        
        // CRITICAL: Still no mutation yet
    }
}
```

**Produces**:
- `AdmissionProof`: permission to mutate canonical state

**Does NOT**:
- Mutate state yet (only verifies permission)

### Phase 4: Commit (mmsb-memory) - **THE ONLY MUTATION POINT**

**Authority**: Canonical State Mutation
**Location**: `mmsb-memory/src/memory_engine.rs::commit_delta()`

```rust
impl MemoryEngine {
    pub fn commit_delta(
        &self,
        admission: &AdmissionProof,
        delta: &Delta,
    ) -> Result<CommitProof, CommitError> {
        // Advance epoch
        self.epoch.increment();
        
        // Write to transaction log
        self.tlog.write().append(admission, delta.clone())?;
        
        // THIS IS THE ONLY PLACE CANONICAL STATE CHANGES
        
        Ok(CommitProof { /* ... */ })
    }
}
```

**THIS IS THE ONLY FUNCTION THAT MUTATES CANONICAL STATE**

**Produces**:
- `CommitProof`: witnesses that mutation occurred
- Canonical state is now changed

### Phase 5: Outcome Validation (mmsb-memory)

**Authority**: Invariant Validation
**Location**: `mmsb-memory/src/memory_engine.rs::record_outcome()`

```rust
impl MemoryEngine {
    pub fn record_outcome(
        &self,
        commit: &CommitProof,
    ) -> Result<OutcomeProof, OutcomeError> {
        // Validate DAG invariants
        // Check for cycles
        // Verify structural integrity
        
        Ok(OutcomeProof { /* ... */ })
    }
}
```

**Produces**:
- `OutcomeProof`: witnesses that invariants held post-commit

## Why This Design is Optimal

### 1. Deterministic Replay

Since executor doesn't mutate, we can replay execution at any time:

```
replay(J) → same (E_p, Δ)  // Always deterministic
```

### 2. Proof Chain Integrity

Every stage produces immutable proof:

```
J → E_p → Δ' → A → C → O
```

No stage can backtrack or modify previous proofs.

### 3. Single Source of Truth

Only `MemoryEngine::commit_delta()` mutates:

```
∀ mutation m : m ∈ MemoryEngine.commit_delta()
```

### 4. Testability

Each phase is pure function (except final commit):

```rust
execute(J) → (E_p, Δ)        // Pure, testable
normalize(Δ) → Δ'             // Pure, testable
admit(J) → A                  // Pure verification
commit(A, Δ') → C             // ONLY MUTATION
validate(C) → O               // Pure validation
```

## Anti-Pattern: What NOT To Do

### ❌ WRONG: Executor Mutates State

```rust
// VIOLATION - DO NOT DO THIS
impl ExecutionLoop {
    pub fn execute(&mut self, judgment: &JudgmentProof) {
        // ❌ Direct mutation breaks everything
        self.memory.mutate_page(page_id, data);  // FORBIDDEN
    }
}
```

**Why this breaks**:
- Non-deterministic replay
- Proof chain collapses
- Multiple mutation points
- Authority leakage

### ❌ WRONG: Propagation Applies Deltas

```rust
// VIOLATION - DO NOT DO THIS
impl DeltaNormalizer {
    pub fn normalize(&mut self, delta: Delta) {
        // ❌ Propagation must not apply changes
        self.apply_to_memory(delta);  // FORBIDDEN
    }
}
```

**Why this breaks**:
- Bypasses admission
- No transaction log entry
- No proof of commit

## Correct Pattern: Separation of Concerns

```rust
// ✅ CORRECT: Executor proposes, Memory decides

// Executor: "Here's what WOULD change"
let outcome = executor.execute(judgment);

// Propagation: "Here's the normalized form"
let normalized = propagation.normalize(outcome.delta);

// Memory: "I verify, admit, commit, validate"
let admission = memory.admit(judgment)?;
let commit = memory.commit(admission, normalized)?;
let outcome = memory.record_outcome(commit)?;

// NOW state has changed, with full proof chain
```

## Summary

**One Law**: Only `MemoryEngine::commit_delta()` mutates canonical state.

**Three Phases**:
1. Execution → produces proof + proposed delta (no mutation)
2. Propagation → normalizes delta (no mutation)
3. Memory → admits, commits (MUTATION), validates

**Zero Exceptions**: No other code path may mutate canonical state.
