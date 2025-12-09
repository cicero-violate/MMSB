### **MMSB DAG Dependency + Priority Plan**

Based on the **MMSB (Memory-Mapped State Bus)** framework and its current **v0.1.0-alpha** state, here is an updated **DAG dependency** and **priority plan** to help organize the flow of tasks, from development to testing and release. This structure ensures that key dependencies are handled correctly and that progress follows a logical path.

---

### **Current Context**

**MMSB** is a **deterministic, delta-driven shared-memory fabric** that facilitates data movement and synchronization between **CPU**, **GPU**, and **compiler subsystems**. It follows the **semiring law** (`state × delta → state′`), ensuring that operations are deterministic and that state changes are propagated accordingly.

Key components:

* **Allocator**
* **Delta Router**
* **Propagation Graph**
* **TLog (Transaction Log)**
* **CUDA Kernels**
* **Instrumentation Hooks**
* **Monitoring Stack**

The current focus is on completing the **core Rust** implementation, optimizing **GPU memory management**, **checkpointing**, **replay**, and **propagation** functionality.

---

### **Updated DAG Dependency + Priority**

| **Phase**                               | **Purpose**                                                                                            | **Status**    | **Blocking Dependency**               |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------ | ------------- | ------------------------------------- |
| **P0 – Diagnostics & Fixes**            | Ensure Rust functionality is stable; fix any remaining issues                                          | ✅ Complete    | —                                     |
| **P1 – Core Features Stabilization**    | Finalize core functionality for **memory allocation**, **delta handling**, **checkpointing**, **TLog** | ✅ Complete    | P0 (Rust diagnostics complete)        |
| **P2 – GPU Memory Integration**         | Complete GPU memory integration; ensure synchronization across CPU/GPU                                 | ⏳ In Progress | P1 (Allocator & propagation complete) |
| **P3 – Checkpoint & Replay Validation** | Verify checkpointing, restore functionality, and TLog consistency                                      | ✅ Complete    | P2 (GPU memory integration)           |
| **P4 – Performance Optimization**       | Optimize for latency, throughput, and memory efficiency (focus on GPU, TLog)                           | ⏳ Deferred    | P3 (Checkpoint validation)            |
| **P5 – Testing & CI Hardening**         | Run full testing suite; integrate CI pipeline for future stability                                     | ⏳ Deferred    | P4 (Performance optimization)         |
| **P6 – Documentation & Release Prep**   | Finalize documentation, examples, and benchmarks; prepare for alpha release                            | ⏳ Deferred    | P5 (Testing complete)                 |

---

### **Priority Task Breakdown**

1. **P0 – Diagnostics & Fixes** (Completed)

   * **Purpose**: Ensure that **Rust diagnostics** are in place and any crash logs are captured.
   * **Status**: ✅ Complete — **Rust logging** is fully set up, and the **segfault** issue has been fixed.

2. **P1 – Core Features Stabilization** (Completed)

   * **Purpose**: Finalize core features, including **memory management** and **delta routing**. Ensure that the **TLog** functionality is fully integrated and **checkpointing** works as expected.
   * **Status**: ✅ Complete — All core features, including **FFI tests** and **CUDA memory integration**, are working. The codebase is stable.

3. **P2 – GPU Memory Integration** (In Progress)

   * **Purpose**: Finalize integration for **GPU memory** with **unified memory** support. Ensure **GPU-CPU synchronization** and optimize **memory access**.
   * **Current Focus**: Implement **GPU allocator** integration and validate **GPU page synchronization** with the CPU, ensuring that **memory coherence** is maintained across devices.
   * **Next Steps**:

     * Verify **memory access** consistency across CPU and GPU.
     * Test **propagation** events and ensure **delta updates** are properly applied to **GPU memory**.

4. **P3 – Checkpoint & Replay Validation** (Completed)

   * **Purpose**: Ensure that the **checkpointing** and **restore** functionality is working as expected. This includes both **full memory dumps** and **delta-based checkpoints**.
   * **Status**: ✅ Complete — **Checkpoint write/load** functionality has been tested and **replay consistency** is verified.

5. **P4 – Performance Optimization** (Deferred)

   * **Purpose**: Optimize for **latency** and **throughput**, especially for memory operations between **CPU and GPU**. Focus on optimizing **TLog operations** and **delta handling** to reduce bottlenecks.
   * **Next Steps**: After confirming stability in earlier phases, focus on profiling the **latency** of propagation, **memory allocations**, and **delta writes**.

6. **P5 – Testing & CI Hardening** (Deferred)

   * **Purpose**: Run the **full testing suite** and integrate the system with **CI/CD** pipelines to ensure continuous integration and prevent regressions.
   * **Next Steps**: Once performance optimizations are done, implement **unit tests**, **integration tests**, and set up **continuous testing** in the CI pipeline.

7. **P6 – Documentation & Release Prep** (Deferred)

   * **Purpose**: Finalize **user-facing documentation**, **examples**, and **benchmarks** for the alpha release.
   * **Next Steps**: After all features are finalized and stable, focus on writing detailed **API documentation**, **example usage**, and **performance baselines** for the alpha release.

---

### **Dependency Graph** (textual format)

1. **P0 (Diagnostics & Fixes)** → **P1 (Core Features Stabilization)** — Rust diagnostics, fix crashes.
2. **P1 (Core Features Stabilization)** → **P2 (GPU Memory Integration)** — Core functions are in place, now focusing on **GPU memory**.
3. **P2 (GPU Memory Integration)** → **P3 (Checkpoint & Replay Validation)** — After GPU integration, **checkpointing** and **replay** are validated.
4. **P3 (Checkpoint & Replay Validation)** → **P4 (Performance Optimization)** — Once functionality is stable, optimize performance.
5. **P4 (Performance Optimization)** → **P5 (Testing & CI Hardening)** — After performance is optimized, harden the system through testing and CI integration.
6. **P5 (Testing & CI Hardening)** → **P6 (Documentation & Release Prep)** — Once CI/CD is in place, finalize documentation and prepare for release.

---

### **Key Milestones for Upcoming Work**

1. **GPU Memory Integration** (P2) is critical for ensuring **cross-device synchronization** and should be the focus now. Ensure that **delta propagation** works seamlessly across **CPU and GPU** pages.
2. **Checkpoint & Replay** (P3) has already been validated. Make sure **no data is lost** in the **checkpoint restore** process and that the **replay functionality** is correct.
3. **Performance Optimization** (P4) will involve **profiling** the code, especially in the context of **GPU memory operations** and **delta routing**. Focus on **latency** and **scalability**.
4. **CI Integration** (P5) should be set up once performance optimizations are stable. This includes integrating with **CI/CD** tools to automate testing and ensure **regression prevention**.


