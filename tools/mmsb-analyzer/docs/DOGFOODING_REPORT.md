# PHASE 6.5 Dogfooding Report

**Date**: 2026-01-01T07:30:50.470358577+00:00
**Target**: mmsb-analyzer (self-dogfood)
**Status**: Read-only validation

## Summary

- Single-action admissibility: 1 types tested
- Sequential pairs tested: 10
- Mixed batches tested: 1
- Scaling tests: 5 batch sizes

## Single-Action Baseline

All single actions should be admissible (no self-conflict):

- ✅ `UpdateCaller`: 10 actions admissible

## Sequential Pair Analysis

- Admissible: 0
- Inadmissible: 10 (conservative conflicts)

## Conservatism Validation

Conservative admission correctly blocks overlapping actions:

- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)
- ❌ `UpdateCaller` + `UpdateCaller` → conflict (expected)

## Batch Scaling

Admission conservatism increases with batch size:

- N=1: ✅ admissible
- N=2: ❌ conflict
- N=5: ❌ conflict
- N=10: ❌ conflict
- N=20: ❌ conflict

## Findings

### Expected Behavior

- ✅ Single actions are admissible (baseline)
- ✅ Conservative composition detects conflicts
- ✅ Larger batches trigger more conflicts (conservative)
- ✅ Artifacts written for all tests

### Anomalies

- None detected (system behaving as specified)

## Conclusion

PHASE 6.5 admission system validated against real-world actions from mmsb-analyzer's own codebase. Conservatism is working as designed.

**System Status**: ✅ Sealed and operational
