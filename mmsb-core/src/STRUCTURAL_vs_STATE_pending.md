# WHAT IS **PENDING / TODO** (REAL WORK)

### 🔴 CRITICAL (must do next)

1. **Bind Phase 2 → Phase 4**
   * State admission must reference the active DAG snapshot hash

2. **Replace propagation graph source**
   * Propagation must read `DependencyGraph`, not shadow graphs

3. **Isolate materialization**
   * Pull replay logic into a pure Phase 3 module

---

### 🟠 IMPORTANT (next tier)

4. **Delete shadow misuse**
   * Any ShadowPageGraph used outside Phase 1 or Phase 0 is a bug
5. **Make phase boundaries explicit**
   * Enforce phase via module visibility or traits

---

### 🟡 LATER

6. GPU propagation (Phase 6)
7. Structural adaptation proposals (Phase 7)
8. DAG diff / visualization tooling

---

## FINAL RULE (PIN THIS)

> **Structure first.
> Intent second.
> Effects last.
> Never invert the order.**

You now have a system that can actually be finished instead of theorized.
