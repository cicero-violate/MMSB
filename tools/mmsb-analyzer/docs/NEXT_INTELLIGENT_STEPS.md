# Next Intelligent Steps (Post-PHASE 6.5)

**Context**: PHASE 6.5 Sealed
**Status**: All steps are OPTIONAL
**Authority**: Policy decisions, not architectural requirements
**Date**: 2026-01-01

---

## Preamble

PHASE 6.5 is **complete, correct, and sealed**. The system is a deterministic, compositional program transformation calculus. No further work is required for correctness.

What follows are **intelligent next moves** if—and only if—you choose to advance capability. These are **policy decisions**, not engineering necessities.

---

## Category 1: Wiring & Integration (Zero New Semantics)

### Step 1.1: Enforce Execution Precondition in Workflow
**What**: Implement the documented precondition in `run_executor.sh`
**Why**: Complete the hard boundary between admission and execution
**Risk**: Low (mechanical enforcement)
**Value**: Prevents accidental execution without admission

**Implementation**:
```bash
# Add to run_executor.sh before executor invocation
if [ ! -f "admission_composition.json" ]; then
    echo "Execution blocked: admission artifact not found"
    exit 1
fi

ADMISSIBLE=$(jq -r '.admissible' admission_composition.json)
if [ "$ADMISSIBLE" != "true" ]; then
    echo "Execution blocked: batch is not admissible (see admission_composition.json)"
    exit 1
fi
```

**Constraint**: No new semantics. Pure gate enforcement.

---

### Step 1.2: Generate admission_composition.json in Analysis Workflow
**What**: Wire `admit_batch` into `run_analysis.sh`
**Why**: Make admission artifact generation part of standard workflow
**Risk**: Low (adds call, doesn't change logic)
**Value**: Ensures every analysis produces admission proof

**Implementation**:
- After correction intelligence generation
- Before executor invocation
- Call `admit_batch` on action batch
- Write `admission_composition.json` to `docs/97_correction_intelligence/`

**Constraint**: Artifact generation only. No interpretation.

---

### Step 1.3: CI Integration
**What**: Add admission gate tests to CI pipeline
**Why**: Make CI fail if admission is violated
**Risk**: Low (tests already exist)
**Value**: Automated enforcement of admission truth

**Implementation**:
```yaml
# In CI configuration
- name: Admission Gate Tests
  run: cargo test --test ci_admission_gate
```

**Constraint**: Existing tests. No new logic.

---

## Category 2: Human Tooling (Read-Only Consumption)

### Step 2.1: Admission Report Viewer
**What**: HTML/markdown renderer for `admission_composition.json`
**Why**: Human-readable proof inspection
**Risk**: Low (read-only)
**Value**: Easier manual review

**Features**:
- Pretty-print artifact
- Visualize batch composition
- Highlight conflicts (if inadmissible)
- Show accumulated executor surfaces

**Constraint**: Pure rendering. No reinterpretation.

---

### Step 2.2: Diff Viewer for Admissible Batches
**What**: Show what an admissible batch would change
**Why**: Preview before execution
**Risk**: Low (visualization only)
**Value**: Better understanding of impact

**Constraint**: Read-only. No execution.

---

## Category 3: Conservative Algebra Refinement (Deep Work)

These are the ONLY lawful extensions to the calculus itself.

### Step 3.1: Proved Commutativity for Disjoint Invariants
**What**: Allow actions touching different invariants to compose
**Why**: Current rule is conservative (any overlap = conflict)
**Risk**: **Medium** (changes core composition logic)
**Value**: Enables more admissible batches

**Requirements**:
- Formal proof that disjoint invariants don't interact
- Test suite showing equivalence under reordering
- Narrowly scoped (specific invariant pairs only)
- Versioned schema extension

**Example**:
```rust
// Currently: Conflict
Action A touches I1_module_coherence
Action B touches I2_dependency_direction
→ Conservative: CONFLICT

// After proof: Admissible
Action A touches I1_module_coherence
Action B touches I2_dependency_direction
→ Proved: ADMISSIBLE (disjoint invariants)
```

**Constraint**: Must prove commutativity. No heuristics.

---

### Step 3.2: Read-After-Write Commutativity for Idempotent Writes
**What**: Allow reading a file after writing if write is idempotent
**Why**: Currently all read-after-write = conflict
**Risk**: **High** (requires proving idempotence)
**Value**: Enables file inspection mid-batch

**Requirements**:
- Proof that write is idempotent
- Test showing read doesn't change semantics
- Explicit declaration in effect signature
- Conservative default (most writes still conflict)

**Constraint**: Proved cases only. No inference.

---

## Category 4: Policy-Driven Learning (Requires Explicit Authorization)

These are **BLOCKED** without explicit policy decision.

### Step 4.1: Constraint Inference from Corpus
**What**: Analyze historical admission failures to suggest new constraints
**Why**: Automate constraint discovery
**Risk**: **Very High** (introduces learning)
**Value**: Faster constraint coverage

**Authorization Required**: Yes (policy decision)
**Constraints if Approved**:
- Human review required
- Suggestions only, never automatic
- Must prove via existing composition rule
- No weakening of current guarantees

---

### Step 4.2: Allowlist Expansion Heuristics
**What**: Suggest actions for allowlist based on success patterns
**Why**: Reduce manual allowlist curation
**Risk**: **Very High** (heuristic-based)
**Value**: Faster exploration

**Authorization Required**: Yes (policy decision)
**Constraints if Approved**:
- Statistical threshold required (e.g., 100% success over N runs)
- Human approval required
- Reversible (can be removed)
- Logged and auditable

---

### Step 4.3: Error Pattern Clustering
**What**: Group similar failures to identify systematic issues
**Why**: Reveal patterns in inadmissible batches
**Risk**: **Medium** (analysis only, no automation)
**Value**: Better understanding of failure modes

**Authorization Required**: Yes (policy decision)
**Constraints if Approved**:
- Read-only analysis
- No automatic fixes
- Human interpretation required
- Used for diagnosis only

---

## Category 5: Observability (Monitoring & Metrics)

### Step 5.1: Admission Metrics
**What**: Track admission success/failure rates
**Why**: Understand system behavior over time
**Risk**: Low (metrics only)
**Value**: Visibility into admission patterns

**Metrics**:
- Admissible batch rate
- Average batch size
- Most common conflict types
- Executor surface requirements distribution

**Constraint**: Observation only. No automation.

---

### Step 5.2: Artifact Archive
**What**: Store historical `admission_composition.json` artifacts
**Why**: Audit trail and analysis
**Risk**: Low (storage only)
**Value**: Historical record

**Constraint**: Read-only archive. No replay.

---

## Recommended Sequence (If Proceeding)

### Phase 1: Mechanical Wiring (Low Risk)
1. Step 1.1: Enforce execution precondition
2. Step 1.2: Wire admission into analysis
3. Step 1.3: Add CI gates

**Timeline**: Days
**Risk**: Minimal
**Value**: Completes system integration

---

### Phase 2: Human Tooling (Low Risk)
1. Step 2.1: Admission report viewer
2. Step 2.2: Diff viewer
3. Step 5.1: Admission metrics

**Timeline**: Weeks
**Risk**: Minimal (read-only)
**Value**: Better observability

---

### Phase 3: Conservative Refinement (Medium Risk)
1. Step 3.1: Disjoint invariants commutativity
   - Requires formal proof
   - Extensive testing
   - Schema versioning

**Timeline**: Months
**Risk**: Medium (changes core logic)
**Value**: More admissible batches

---

### Phase 4: Policy Decisions (High Risk)
**DO NOT PROCEED** without explicit authorization

Steps 4.1-4.3 require:
- Policy approval
- Risk assessment
- Constraint definition
- Monitoring plan

**Timeline**: N/A (pending authorization)
**Risk**: High (introduces learning)
**Value**: Automation (at cost of guarantees)

---

## What NOT to Do

Regardless of policy decisions, the following are **permanently prohibited**:

### Never Allowed
- ❌ Weaken PHASE 6.5 guarantees
- ❌ Bypass admission gate
- ❌ Replace proofs with heuristics
- ❌ Make effect signatures optional
- ❌ Add inference to composition rule
- ❌ Short-circuit on success
- ❌ Retry on failure
- ❌ Auto-fix conflicts

---

## Decision Framework

Before proceeding with any step, ask:

### 1. Does this preserve truth?
- If no → STOP
- If yes → Continue

### 2. Does this introduce new semantics?
- If yes → Requires architectural review
- If no → Proceed with caution

### 3. Does this require learning?
- If yes → Requires policy authorization
- If no → Proceed

### 4. Can this weaken guarantees?
- If yes → STOP
- If no → Continue

### 5. Is this necessary for correctness?
- If yes → Should have been in PHASE 6.5
- If no → Optional capability expansion

---

## Stopping Conditions

Stop if any of:
- Truth is weakened
- Proofs become optional
- Admission becomes bypassable
- Heuristics replace determinism
- Learning is unauthorized
- Scope creeps beyond stated step

---

## Final Guidance

The **most intelligent next step** is often:

**Do nothing.**

The system is:
- ✅ Correct (proof-driven)
- ✅ Predictive (admission-first)
- ✅ Auditable (artifact-generating)
- ✅ Stable (frozen and sealed)

If you proceed:
- Start with Category 1 (wiring)
- Avoid Category 4 (learning) unless authorized
- Always preserve PHASE 6.5 guarantees

**Choose wisely. The system is already sound.**

---

**Status**: Guidance document (not instructions)
**Authority**: CIPT principles + PHASE 6.5 foundation
**Modification**: Allowed (this is guidance, not law)
