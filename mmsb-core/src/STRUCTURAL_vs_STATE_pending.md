# WHAT IS **PENDING / TODO** (REAL WORK)

### 🔴 CRITICAL (must do next)

1. **Bind Phase 2 → Phase 4**
   ✅ DONE: State admission now references DAG snapshot hash
   * State admission must reference the active DAG snapshot hash
   * Added dag_snapshot_hash to MmsbAdmissionProof
   * Added DependencyGraph::compute_snapshot_hash()
   * commit_delta verifies DAG hash when present
   * Legacy proofs without hash trigger warning

2. **Replace propagation graph source**
   ✅ DONE: Propagation wired to authoritative DependencyGraph
   * Propagation must read `DependencyGraph`, not shadow graphs
   * TickOrchestrator now holds Arc<DependencyGraph>
   * request_commit accepts Option<&DependencyGraph> parameter
   * dag_propagation module provides DAG-aware traversal
   * Production paths ready for DAG binding (currently pass None for legacy compat)

3. **Isolate materialization**
   ✅ DONE: Pure materialization module created
   * Pull replay logic into a pure Phase 3 module
   * MaterializedPageView provides immutable state representation
   * materialize_page_state is pure function with no side effects
   * No writes, no DAG reads, no judgment calls
   * Deterministic replay from delta sequence
   * Full test coverage included
   * Legacy Page::apply_delta preserved for backward compatibility

---

### 🟠 IMPORTANT (next tier)

4. **Delete shadow misuse**
   🔄 NEXT
   * Any ShadowPageGraph used outside Phase 1 or Phase 0 is a bug
   * Audit all ShadowPageGraph usage
   * Replace with DependencyGraph or remove
   
5. **Make phase boundaries explicit**
   ⏳ PENDING
   * Enforce phase via module visibility or traits

---

### 🟡 LATER

6. GPU propagation (Phase 6)
7. Structural adaptation proposals (Phase 7)
8. DAG diff / visualization tooling

---

### ⚠️ TECHNICAL DEBT (tracked)

**TODO (Breaking):**
Move DAG snapshot hash from admission proof into Delta schema after propagation wiring stabilizes.

**Reason:** 
Currently tracking DAG snapshot at judgment boundary (MmsbAdmissionProof) to avoid:
- Breaking Delta schema changes
- TLog migration complexity  
- Invalidating existing persisted logs

This is correct for initial implementation but will need refactoring once:
1. Propagation is wired to authoritative DependencyGraph
2. Phase boundaries are enforced
3. System behavior is validated

---

## FINAL RULE (PIN THIS)

> **Structure first.
> Intent second.
> Effects last.
> Never invert the order.**

You now have a system that can actually be finished instead of theorized.
