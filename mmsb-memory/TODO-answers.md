**Ignore all `__<name>` directories — they are non-semantic markers.**

1. **Numbered Directory Structure**
   Remove numbered prefixes during refactor; canonical structure is the non-numbered semantic directories only.
2. **Existing `mmsb-executor` Crate**
   Review its current contents first, then merge substrate code into it incrementally.
3. **Dual Directory Naming**
   Non-numbered directories are canonical; numbered ones are legacy/markers and ignored.
4. **MemoryView Interface Boundaries**
   Expose read-only access to proofs D/E/F, committed facts, epochs, and replay snapshots only.
5. **Event Emission from Memory**
   Memory emits factual commit events during normal operation, but replay itself emits nothing.
6. **Proof Production vs Truth Semantics**
   Proof production D/E/F is part of truth semantics, not runtime behavior.
7. **Storage Boundary**
   Memory calls `mmsb-storage` via a synchronous, minimal persistence interface; storage owns IO details.
8. **Existing `proofs/` Directory**
   It contains implementations of D/E/F and must align with definitions in `mmsb-proof`.
9. **Module vs Engine Rename Scope**
   Rename affects type names and imports only; audit dependencies before changing external references.
10. **Materialization Location**
    Materialization stays in memory if it defines *what* state becomes, not *how* it is executed.

**Yes — proceed with the dependency audit first.**
