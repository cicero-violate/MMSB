# Execution Precondition Requirement

**Status**: MANDATORY
**Authority**: PHASE 6.5 System Hardening
**Date**: 2026-01-01

---

## Invariant

**Execution MUST NOT proceed unless batch admission has succeeded.**

This is a hard requirement enforced at the system boundary between admission (mmsb-analyzer) and execution (mmsb-executor).

---

## Required Checks

Before `mmsb-executor` is invoked, the following conditions MUST be verified:

### 1. Admission Decision is Admissible
```
AdmissionDecision == Admissible
```

### 2. Artifact Exists
```
admission_composition.json exists at expected path
```

### 3. Schema Version Matches
```
artifact.schema_version == SCHEMA_VERSION (from mmsb-analyzer)
```

---

## Enforcement Point

The precondition MUST be enforced at **the point where mmsb-executor is invoked**.

### Current Workflow
```
run_analysis.sh
  ↓
generates: correction_intelligence.json
           admission_preflight.json
  ↓
run_executor.sh
  ↓
mmsb-executor (mutation + verification)
```

### Required Enforcement
```
run_analysis.sh
  ↓
generates: correction_intelligence.json
           admission_composition.json  ← PHASE 6.5 artifact
  ↓
[ENFORCEMENT POINT]
  ↓
IF admission_composition.json shows admissible == false:
    ABORT with error message
    DO NOT invoke mmsb-executor
  ↓
run_executor.sh
  ↓
mmsb-executor (mutation + verification)
```

---

## Error Handling

If any condition fails:

### Action
- **Abort immediately**
- **Do NOT invoke executor**
- **Do NOT attempt recovery**
- **Do NOT re-run admission**

### Error Message (Single Line)
```
Execution blocked: batch is not admissible (see admission_composition.json)
```

### Exit Code
- **Non-zero** (suggesting exit code 1 for "admission failure")

---

## Prohibited Actions

When precondition fails, the system MUST NOT:

- ❌ Re-run admission
- ❌ Interpret the artifact
- ❌ Attempt to fix conflicts
- ❌ Suggest corrections
- ❌ Skip failed actions
- ❌ Proceed with partial execution
- ❌ Downgrade to dry-run

**The artifact is authority. Execution is blocked. Period.**

---

## Implementation Options

### Option 1: Workflow Script Guard (Recommended)
Add check in `run_executor.sh` before invoking mmsb-executor:

```bash
# Check admission artifact exists and shows admissible
if [ ! -f "admission_composition.json" ]; then
    echo "Execution blocked: admission artifact not found"
    exit 1
fi

ADMISSIBLE=$(jq -r '.admissible' admission_composition.json)
if [ "$ADMISSIBLE" != "true" ]; then
    echo "Execution blocked: batch is not admissible (see admission_composition.json)"
    exit 1
fi

# Schema version check
ARTIFACT_VERSION=$(jq -r '.schema_version' admission_composition.json)
EXPECTED_VERSION="0.1.0"  # From SCHEMA_VERSION
if [ "$ARTIFACT_VERSION" != "$EXPECTED_VERSION" ]; then
    echo "Execution blocked: schema version mismatch"
    exit 1
fi

# Precondition satisfied - proceed with execution
cargo run -- <executor args>
```

### Option 2: Executor CLI Flag
Add `--admission-artifact <path>` flag to executor that:
- Reads artifact
- Verifies admissible == true
- Aborts if not

### Option 3: CI Gate
Make CI fail if executor runs without valid admission artifact.

---

## Verification

The precondition is correctly enforced when:

1. Running executor with inadmissible artifact → **Fails immediately**
2. Running executor without artifact → **Fails immediately**
3. Running executor with schema mismatch → **Fails immediately**
4. Running executor with admissible artifact → **Proceeds normally**

---

## Integration with PHASE 6.5

This precondition completes the separation of concerns:

```
Admission (mmsb-analyzer)
  ↓ produces
admission_composition.json (proof)
  ↓ gates
Execution (mmsb-executor)
  ↓ produces
executor_report.json (confirmation)
```

**The proof gates the power.**

---

## Status

**Current**: Documented requirement
**Next**: Implement enforcement at chosen point
**Authority**: CIPT Laws + PHASE 6.5 Completion

---

**This precondition is non-negotiable.**
