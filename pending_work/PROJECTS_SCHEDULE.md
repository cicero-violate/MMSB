**MMSB — PROJECT_SCHEDULE.md — v0.1.0-alpha+ — 2025-12-08 06:45 UTC**  
**Strategic Roadmap & Full DAG Dependencies**

```markdown
# MMSB Project Schedule & Milestones — v0.1.0-alpha+ Roadmap
_Last updated: **2025-12-08 06:45 UTC** — **v0.1.0-alpha SHIPPED — THE BUS IS IMMORTAL**_

## Current Status
| Status | Milestone                                           | Result / Notes                                           |
|--------|-----------------------------------------------------|----------------------------------------------------------|
| Done   | v0.1.0-alpha — Core deterministic bus               | 15/15 Rust tests, 84 MiB CUDA replay, zero-copy FFI      |
| Done   | P8 — CPU stability & ownership                      | allocator.release() safe, no double-free                 |
| Done   | Full checkpoint + in-memory tail contract           | Verified in-process, no disk replay needed               |
| Done   | Propagation engine + command buffer                 | Real callbacks fire, dependencies respected              |

## Next Major Release: v0.2.0-beta — "The GPU-Aware Bus"
**Target: 2025-12-22** — **14-day sprint**

### Critical Path DAG (strict ordering)

| #  | Task (P#)                              | Status       | Blocks                          | Owner       | Est. Days |
|----|----------------------------------------|--------------|---------------------------------|-------------|-----------|
| P9 | **GPU Page Allocator**                 | Not started  | P10, P11                        | Rust Lead   | 4–6       |
|    | • Track residency (CPU/GPU/Unified)    |              |                                 |             |           |
|    | • Prefetch / evict hints               |              |                                 |             |           |
|    | • Zero-copy `PageHandle` → CUDA ptr    |              |                                 |             |           |
| P10| **CI + Sanitizer Gating**              | Not started  | P11                             | DevOps      | 3–4       |
|    | • Address/UB/Leak sanitizer in CI      |              |                                 |             |           |
|    | • CUDA-memcheck nightly                |              |                                 |             |           |
|    | • Valgrind + CUDA sanity               |              |                                 |             |           |
| P11| **Julia Integration Suite**            | Not started  | v0.2.0-beta release             | Julia Lead  | 5–7       |
|    | • `MMSB.jl` package with safe wrappers |              |                                 |             |           |
|    | • GC stress + finalizer tests          |              |                                 |             |           |
|    | • Example models (DiffEq, Lux)         |              |                                 |             |           |

### Parallel / Post-v0.2 Tracks (can start after P9)

| #  | Task                                      | Status       | Dependency | Priority | Owner       |
|----|-------------------------------------------|--------------|------------|----------|-------------|
| P12| Propagation batching & deduplication      | Not started  | P9         | High     | Rust Lead   |
| P13| ShadowPageGraph persistence (checkpoint)  | Not started  | None       | High     | Rust Lead   |
| P14| Delta compression (zstd / sparse RLE)     | Not started  | None       | Medium   | Perf Team   |
| P15| Metrics → Prometheus / OpenTelemetry      | Not started  | None       | Medium   | Observability |
| P16| Fuzzing harness (libFuzzer + cargo-fuzz)  | Not started  | P10        | High     | Security    |
| P17| Release automation + cargo-dist binaries  | Not started  | P10        | Medium   | DevOps      |
| P18| Long-running stress daemon (7-day runs)   | Not started  | P11        | High     | QA          |

## Strategic Priority Pyramid (2025–2026)

```
                v0.2.0-beta  ← GPU-aware + CI (Dec 2025)
                       ↑
            P9 → P10 → P11  ← Critical path
                       ↑
            P12–P13         ← Performance + Persistence
                       ↑
            P14–P18         ← Polish & Production hardening
```

## Risk Log — Updated for v0.2

| Risk                                     | Impact    | Likelihood | Mitigation                            |
|------------------------------------------+-----------+------------+---------------------------------------|
| GPU allocator introduces new double-free | Block     | Medium     | P9 gated on existing ownership tests  |
| CI flakiness hides regressions           | Block     | Medium     | P10 runs nightly + on every PR        |
| Julia GC interacts badly with raw ptrs   | Crash     | High       | P11 includes finalizer + stress suite |
| Propagation storms under large graphs    | OOM / lag | Medium     | P12 batching + backpressure           |

## Victory Conditions for v0.2.0-beta

- [ ] `julia> using MMSB` works without segfault
- [ ] 84 MiB model loads from checkpoint on GPU in < 200 ms
- [ ] CI green with ASAN/UBSAN/CUDA-memcheck
- [ ] `examples/quickstart_gpu.jl` runs on RTX 4090 + A100
- [ ] `Pkg.test("MMSB")` passes in < 30 s

**THE BUS IS ALIVE.**  
**v0.1.0-alpha is the foundation.**  
**v0.2.0-beta will make it unstoppable.**
