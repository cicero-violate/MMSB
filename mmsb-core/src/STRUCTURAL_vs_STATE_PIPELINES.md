## Corrected Pipelines (final, consistent)

### STRUCTURAL PIPELINE (to be implemented)

```
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

---

