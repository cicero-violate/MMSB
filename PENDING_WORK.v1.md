Below is a **fully structured, dependency-sorted TO-DO-LIST.md**, generated as a **true DAG (Directed Acyclic Graph)** of all remaining tasks.

This is *the authoritative engineering plan* for MMSB going forward.

Each task includes:

* **Description**
* **Dependencies** (what must be completed first)
* **Outputs**
* **Acceptance Criteria**

This is production-ready.

---

# **TO-DO-LIST.md**

### **MMSB Pending Work — DAG-Sorted Execution Plan**

*Last updated: now*

This document lists all remaining tasks for MMSB in a **dependency-driven order**, forming a **DAG of implementation steps**.
You MUST follow this order to avoid circular dependencies and nonfunctional subsystems.

---

# ============================================

# **0. Completed Foundations (DO NOT REPEAT)**

# ============================================

The following systems are *already implemented* and unlocked execution:

* Page types, mask/data allocation (CPU/GPU)
* Delta routing logic (CPU/GPU)
* CUDA delta kernel
* Event system (handlers, firehose, logging)
* TLog append/query
* Basic serialization helpers
* Locking, ID allocators, cycle detection
* Correct `include` order inside MMSB.jl

Everything below builds on top of this.

---

# ============================================================

# **1. LEVEL 1 — SERIALIZATION SUBSYSTEM (ROOT NODES)**

# ============================================================

These tasks have **no dependencies**. Everything downstream requires them.

## ✔️ **1.1 Page Serialization V2**

**Depends on:** nothing
**Description:**
Implement the stable binary encoding for Page, including compressed mask, location bit, metadata, and version header.
**Outputs:** `serialize_page`, `deserialize_page`
**Acceptance Criteria:**

* Serialize → deserialize returns an identical Page (CPU/GPU).
* Supports detection of truncated or corrupt data.

---

## ✔️ **1.2 Delta Serialization + Sparse Compression**

**Depends on:** nothing
**Description:**
Implement compressed, sparse delta format.
**Outputs:** `serialize_delta`, `deserialize_delta`
**Acceptance Criteria:**

* Sparse deltas compress ≥ 5× for typical workloads.
* Deserialization is lossless and validated.

---

# ============================================================

# **2. LEVEL 2 — TLOG PERSISTENCE + REPLAY**

# ============================================================

Cannot begin until **Level 1 serialization** is complete.

## ✔️ **2.1 TLog Checkpoint Writer**

**Depends on:** Page serialization, Delta serialization
**Description:**
Create snapshot files that store:

* header/version/timestamp
* all deltas
* page registry metadata
  **Outputs:** `checkpoint_log!`, `load_checkpoint`
  **Acceptance Criteria:**
* Creates reproducible checkpoint archives
* Compatible across multiple Julia sessions

---

## ✔️ **2.2 Full TLog Replay Engine**

**Depends on:** TLog checkpoint writer
**Description:**
Rebuild MMSBState from an empty state using TLog files.
**Outputs:** `replay_log!(state, log)`
**Acceptance Criteria:**

* Reconstructed MMSB matches original bit-for-bit
* Multi-device pages (CPU/GPU) restored correctly
* Dependency graph rebuilt from metadata

---

## ✔️ **2.3 Replay Consistency Tests**

**Depends on:** Replay engine
**Description:**
Add tests to ensure TLog replay is deterministic.
**Outputs:** `test/replay_tests.jl`
**Acceptance Criteria:**

* Running replay twice yields identical states
* Stress test with ≥ 10,000 deltas

---

# ============================================================

# **3. LEVEL 3 — RUNTIME SAFETY + MEMORY MODEL**

# ============================================================

Requires Level 2 because runtime mutations depend on replay correctness.

## **3.1 Thread-Safe PageAllocator V2**

**Depends on:** replay engine (Level 2)
**Description:**
Ensure atomic allocation, migration, and cloning of pages.
**Outputs:**

* lock-protected allocator
* lock-protected migration functions
  **Acceptance Criteria:**
* No deadlocks under stress test
* Parallel allocators produce valid unique IDs

---

## **3.2 Unified Memory Allocator**

**Depends on:** thread-safe allocator
**Description:**
Implement real CUDA unified memory.
**Outputs:** `create_unified_page!`, `GPUUnifiedPage`
**Acceptance Criteria:**

* CPU & GPU can access the same buffer
* Page migration no longer needed for these pages

---

## **3.3 Additional GPU Kernels**

**Depends on:** unified memory
**Description:**
Implement:

* page_clone_kernel
* page_zero_kernel
* page_copy_range_kernel
* scatter/gather kernels
  **Acceptance Criteria:**
* Passes correctness tests
* Kernel performance >1GB/s on typical hardware

---

# ============================================================

# **4. LEVEL 4 — REACTIVE GRAPH ENGINE**

# ============================================================

Depends on Levels 1–3 (pages, logs, runtime safety, kernels).

## ✔️ **4.1 Recomputation Registry**

**Depends on:** unified memory & kernel infrastructure
**Description:**
Add registry mapping `page_type → recompute(state, page)`
**Outputs:** `RECOMPUTE_REGISTRY`
**Acceptance Criteria:**

* Can register recomputations dynamically
* Recomputation dispatch works from graph updates

---

## ✔️ **4.2 Propagation Engine V2**

**Depends on:** recomputation registry
**Description:**
Implement full propagation flow:

* BFS/DFS fanout
* dirty queues
* cycle safety
* priority scheduling
  **Outputs:** `propagate!(state, page_id)`
  **Acceptance Criteria:**
* Scales to graphs of ≥ 100k nodes
* No cycles
* Parallel propagation sequencing

---

## **4.3 Event Serialization**

**Depends on:** propagation engine
**Description:**
Serialize event stream for debugging and replay.
**Outputs:** `serialize_event`, `deserialize_event`
**Acceptance Criteria:**

* End-to-end replay yields same event sequence

---

# ============================================================

# **5. LEVEL 5 — INSTRUMENTATION LAYER**

# ============================================================

These depend on Levels 1–4 because they record IR/SSA into pages.

## **5.1 Safe Base/Core Hooking**

**Depends on:** working MMSB runtime + graph
**Description:**
Implement safe interception using Cassette or CompilerPluginsTools.
**Outputs:**

* call interception
* method table watchers
* eval hooking
  **Acceptance Criteria:**
* No world-age violations
* No reentrant dispatch loops

---

## **5.2 AbstractInterpreter Implementation**

**Depends on:** base/core hooking
**Description:**
Add custom interpreter to capture:

* SSA
* typed IR
* optimized IR
* CFG
* domtree
  **Outputs:** `MMSBInterpreter`
  **Acceptance Criteria:**
* Runs inference on arbitrary methods
* Produces page-backed IR snapshots

---

## **5.3 SSA/IR Serialization**

**Depends on:** AbstractInterpreter
**Description:**
Serialize Core.CodeInfo to bytes.
**Outputs:** `serialize_ssa`
**Acceptance Criteria:**

* SSA → bytes → SSA is lossless
* IR pages generated deterministically

---

## **5.4 World-Age Tracking**

**Depends on:** SSA/IR serialization
**Description:**
Detect world-age changes and invalidate pages.
**Outputs:** world-age registry in MMSB
**Acceptance Criteria:**

* IR/method-state never mismatch actual Julia world

---

# ============================================================

# **6. LEVEL 6 — UTILITIES, TESTS, AND API**

# ============================================================

Depends on all prior functional subsystems.

## **6.1 Error Handling Suite**

**Description:** Structured exceptions.
**Acceptance Criteria:** All critical operations throw typed errors.

---

## **6.2 Monitoring + Stats**

**Description:**
Expose:

* page counts
* memory use (CPU/GPU)
* delta throughput
* graph depth
  **Acceptance Criteria:**
* `MMSB.stats()` returns complete state snapshot

---

## **6.3 Benchmark Suite**

**Description:**
Add micro/macro benchmarks:

* delta throughput
* GPU kernel speed
* propagation latency
  **Acceptance Criteria:**
* Performance baseline established

---

## **6.4 High-Level Public API**

**Description:**
Create:

* `mmsb_start()`
* `mmsb_stop()`
* `@mmsb` macro
  **Acceptance Criteria:**
* External users can run MMSB without knowing internals

---

## **6.5 Documentation Set**

**Description:**

* Architecture.md
* SerializationSpec.md
* DeviceMemoryModel.md
  **Acceptance Criteria:**
* Complete developer-facing docs

---

# ============================================

# **FINAL DAG SUMMARY (HIGHEST → LOWEST)**

# ============================================

This is the dependency-collapsed ordering:

1. ✔️ **Page serialization v2**
2. ✔️ **Delta serialization**
3. ✔️ **TLog checkpoint writer**
4. ✔️ **TLog replay engine**
5. ✔️ **Replay tests**
6. **Thread-safe allocator v2**
7. **Unified memory allocator**
8. **GPU kernel suite**
9. ✔️ **Recomputation registry**
10. ✔️ **Propagation engine v2**
11. **Event serialization**
12. **Safe Base/Core hooking**
13. **AbstractInterpreter**
14. **SSA/IR serialization**
15. **World-age tracking**
16. **Error system**
17. **Monitoring + stats**
18. **Benchmark suite**
19. **Public API**
20. **Docs**

---

If you want, I can also output:

### ✔️ `ROADMAP.md` (strategic multi-week plan)

### ✔️ `issues.json` (GitHub-ready tasks)

### ✔️ A **Mermaid DAG graph** for the pending tasks

### ✔️ A **Makefile** to test each dependency layer

Just say:
**“Generate roadmap”** or **“Generate DAG graph.”**
