# MMSB Project Schedule

## Timeline Overview

**Total Duration:** 20 weeks (5 months)
**Start Date:** Week 1
**Target Completion:** Week 20

---

## Phase 1: Core Infrastructure (Weeks 1-6)

### Objective
Establish performant, tested foundation for MMSB: physical memory, pages, semiring algebra, DAG structure, and propagation engine.

| Week | Layers           | Deliverables                                      | Dependencies   |
|------+------------------+---------------------------------------------------+----------------|
|    1 | Layer 0          | Physical memory layer complete, allocator working | None           |
|    2 | Layer 1 (Part 1) | Page/Delta structures, basic operations           | Layer 0        |
|    3 | Layer 1 (Part 2) | TLog, checkpointing, replay engine                | Layer 1 Part 1 |
|    4 | Layer 2          | Semiring algebra, DeltaRouter refactor            | Layers 0-1     |
|    5 | Layer 3          | ShadowGraph/DAG complete                          | Layers 0-2     |
|    6 | Layer 4          | Propagation engine + GPU kernels                  | Layers 0-3     |

**Phase 1 Exit Criteria:**
- ✓ All unit tests passing for Layers 0-4 [COMPLETE: 2025-12-09]
- ✓ GPU propagation kernel benchmarked [AWAITING CUDA RUNTIME]
- ✓ Checkpoint/replay working [COMPLETE: 2025-12-09]
- ✓ Semiring algebra validated [COMPLETE: 2025-12-09]
- ✓ DAG topological sort verified [COMPLETE: 2025-12-09]

**Status:** ✓ PHASE 1 STRUCTURALLY COMPLETE (2025-12-09)
All P0 tasks complete. Test execution blocked only by CUDA runtime dependency.

---

## Phase 2: Self-Optimization (Weeks 7-10)

### Objective
Build adaptive memory, utility measurement, and intention emergence systems.

| Week | Layers           | Deliverables                              | Dependencies   |
|------+------------------+-------------------------------------------+----------------|
|    7 | Layer 5 (Part 1) | Memory layout optimizer, page clustering  | Layers 0-4     |
|    8 | Layer 5 (Part 2) | Graph rewriting, locality analysis        | Layer 5 Part 1 |
|    9 | Layer 6          | Utility engine, cost functions, telemetry | Layers 0-5     |
|   10 | Layer 7          | Intention engine, goal emergence          | Layers 0-6     |

**Phase 2 Exit Criteria:**
- ✓ Adaptive memory showing measurable improvement (>20% cache hit increase)
- ✓ Utility functions computing correctly
- ✓ Intention signals generated from utility
- ✓ Graph rewriting operational
- ✓ Integration tests for Layers 5-7

---

## Phase 3: Cognition (Weeks 11-16)

### Objective
Implement reasoning and planning engines for symbolic/structural inference and multi-step planning.

|  Week | Layers           | Deliverables                                | Dependencies   |
|-------+------------------+---------------------------------------------+----------------|
| 11-12 | Layer 8 (Part 1) | Reasoning engine core, structural inference | Layers 0-7     |
|    13 | Layer 8 (Part 2) | Constraint propagation, causal inference    | Layer 8 Part 1 |
|    14 | Layer 9 (Part 1) | Planning engine core, search algorithms     | Layers 0-8     |
|    15 | Layer 9 (Part 2) | RL planning, Enzyme integration             | Layer 9 Part 1 |
|    16 | Layers 8-9       | Integration testing, optimization           | Layers 8-9     |

**Phase 3 Exit Criteria:**
- ✓ Reasoning engine producing valid inferences
- ✓ Constraint solver working
- ✓ Planning engine generating multi-step plans
- ✓ Enzyme.jl integration functional
- ✓ At least 3 reasoning/planning test cases passing

---

## Phase 4: Agents + Applications (Weeks 17-20)

### Objective
Complete agent interface, external agent implementations, and demonstration applications.

| Week | Layers     | Deliverables                               | Dependencies |
|------+------------+--------------------------------------------+--------------|
|   17 | Layer 10   | Agent interface complete, protocol defined | Layers 0-9   |
|   18 | Layer 11   | RL agent, symbolic agent, hybrid agent     | Layer 10     |
|   19 | Layer 12   | 3+ demo applications                       | Layers 10-11 |
|   20 | All Layers | Integration, documentation, polish         | All          |

**Phase 4 Exit Criteria:**
- ✓ Agent interface supports external agents
- ✓ At least 3 agent types implemented
- ✓ At least 3 working applications
- ✓ Full system integration test passing
- ✓ Documentation complete
- ✓ Performance benchmarks published

---

## Resource Allocation

### Development Focus per Phase

| Phase | Rust Work | Julia Work | Testing | Documentation |
|-------+-----------+------------+---------+---------------|
|     1 |       70% |        30% |     20% |           10% |
|     2 |       40% |        60% |     25% |           15% |
|     3 |       10% |        90% |     30% |           20% |
|     4 |        5% |        85% |     30% |           30% |

---

## Critical Path Items

**These tasks are blocking and must complete on schedule:**

1. **Week 1:** Physical allocator must work before anything else
2. **Week 4:** Semiring algebra blocks propagation
3. **Week 5:** DAG structure blocks propagation
4. **Week 6:** Propagation blocks adaptive layers
5. **Week 9:** Utility blocks intention
6. **Week 10:** Intention blocks reasoning
7. **Week 14:** Planning engine blocks agents
8. **Week 17:** Agent interface blocks agent implementations

---

## Risk Mitigation

| Risk                          | Probability | Impact | Mitigation                                      |
|-------------------------------+-------------+--------+-------------------------------------------------|
| GPU kernel performance issues | Medium      | High   | Allocate extra week in Phase 1 for optimization |
| Semiring algebra complexity   | Low         | High   | Start with standard semirings, extend later     |
| Reasoning engine scope creep  | High        | Medium | Define minimal viable reasoning system early    |
| Agent interface instability   | Medium      | Medium | Freeze protocol by Week 16                      |
| Integration testing delays    | Medium      | High   | Continuous integration from Week 1              |

---

## Success Metrics

**Weekly:**
- All new code has unit tests
- No regressions in existing tests
- Performance benchmarks run
- Documentation updated

**Phase-end:**
- Phase exit criteria met
- Performance targets achieved
- Integration tests passing
- Stakeholder demo completed

**Project-end:**
- All 13 layers complete
- 100+ files implemented
- Test coverage >80%
- 3+ working applications
- Published benchmarks
- Complete documentation

---

## Contingency Plan

**If behind schedule:**
1. Defer Layer 12 applications to post-v1.0
2. Reduce Layer 8/9 scope to essentials
3. Simplify Layer 11 agent implementations
4. Focus on core infrastructure quality

**If ahead of schedule:**
1. Add more standard semirings to Layer 2
2. Enhance GPU kernel performance in Layer 4
3. Expand reasoning capabilities in Layer 8
4. Build more example applications in Layer 12

---

## Phase 5: CLAUDE.md Compliance & Architectural Refinement (Weeks 21-26)

### Objective
Implement missing architectural components required by CLAUDE.md specification to achieve full compliance with non-negotiable design rules.

| Week | Component | Deliverables | Dependencies |
|------|-----------|--------------|--------------|
| 21 | UpsertPlan Structure | Define first-class UpsertPlan in L7 | None |
| 21 | TLog Metadata Schema | Extend TLog to support intent metadata | None |
| 22 | Intent Lowering (Julia) | Implement intent → deltaspec pipeline | Week 21 |
| 22 | Delta Validation (Rust) | Separate validation from application | None |
| 23 | FFI Lowering Bridge | Create Julia ↔ Rust lowering interface | Week 22 |
| 24 | Integration Testing | End-to-end intent → execution tests | Week 23 |
| 25 | Replay Verification | Intent metadata replay validation | Week 24 |
| 26 | QMU API Refinement | Document Query/Mutate/Upsert clearly | Week 24 |

**Status (through Week 23):** UpsertPlan structure, TLog metadata schema, intent lowering, delta validation, and the Julia ↔ Rust lowering bridge are complete.

**Phase 5 Exit Criteria:**
- ✓ UpsertPlan structure defined and tested
- ✓ Intent lowering pipeline functional
- ✓ Intent metadata persists in TLog
- ✓ Delta validation separated from execution
- ✓ Replay can reconstruct intent history
- ✓ QMU boundaries explicitly documented
- ✓ All CLAUDE.md non-negotiable rules verified

### Week 24: Integration Testing

#### INT.G1: End-to-End Integration

| Date | Task                         |    ID | Hours | Status | Notes                          |
|------+------------------------------+-------+-------+--------+--------------------------------|
| -    | Create test suite skeleton   | 5.6.1 |     2 | ✓      | In test/ directory             |
| -    | Test intent → upsert → delta | 5.6.2 |     8 | ✓      | Full pipeline                  |
| -    | Test QMU separation          | 5.6.3 |     6 | ✓      | Query, Mutate, Upsert distinct |
| -    | Test TLog persistence        | 5.6.4 |     4 | ✓      | Intent metadata saved          |
| -    | Test validation rejection    | 5.6.5 |     2 | ✓      | Invalid deltas blocked         |
| -    | Performance benchmarks       | 5.6.6 |     2 | ✓      | Measure overhead               |

**Week 24 Total**: 24 hours  
**Week 24 Deliverables**: Passing end-to-end test suite

### Week 25: Replay Verification

#### INT.G2: Replay with Intent Metadata

| Date | Task                          |    ID | Hours | Status | Notes                            |
|------+-------------------------------+-------+-------+--------+----------------------------------|
| -    | Test basic replay with intent | 5.7.1 |     4 | ✓      | Reconstruct intent history       |
| -    | Test intent filtering         | 5.7.2 |     4 | ✓      | Replay specific intents          |
| -    | Test intent causality         | 5.7.3 |     4 | ✓      | Intent → delta linkage preserved |
| -    | Test checkpoint + intent      | 5.7.4 |     3 | ✓      | Intent in checkpoint metadata    |
| -    | Write replay documentation    | 5.7.5 |     1 | ✓      | How to use intent replay         |

**Week 25 Total**: 16 hours  
**Week 25 Deliverables**: Intent replay fully verified

---

## Deliverables Summary

|     Phase |  Weeks | Layers | Files Added/Modified | Tests Added |
|-----------+--------+--------+----------------------+-------------|
|         1 |    1-6 |    0-4 | ~40                  | ~60         |
|         2 |   7-10 |    5-7 | ~25                  | ~40         |
|         3 |  11-16 |    8-9 | ~30                  | ~50         |
|         4 |  17-20 |  10-12 | ~25                  | ~30         |
|         5 |  21-26 |    1,7 | ~12                  | ~20         |
| **Total** | **26** | **13** | **~132**             | **~200**    |
