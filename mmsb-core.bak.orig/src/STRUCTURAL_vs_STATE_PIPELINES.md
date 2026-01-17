## Corrected Pipelines (final, consistent)

1) Intent
2) Policy
3) Routing
4) Judgment
5) Execution

### STRUCTURAL PIPELINE (to be implemented)

```
StructuralIntent
   ↓
StructuralOps
   ↓
ShadowPageGraph (apply ops)
   ↓
Validate (acyclic, refs)
   ↓
JudgmentToken + StructuralProof
   ↓
commit_structural_delta
   ↓
DependencyGraph snapshot updated
```

### STATE PIPELINE (already implemented)

```
StateIntent
  ↓
Delta
  ↓
JudgmentToken + AdmissionProof
  ↓
commit_delta → tlog
  ↓
snapshot DependencyGraph
  ↓
propagation
```


