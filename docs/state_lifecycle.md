# MMSB State Lifecycle Specification

## Overview
Formal specification for MMSBState reset, pooling, and deterministic replay.

## Mathematical Model

### Variables
- $\mathcal{S}$: MMSB state space
- $P_s = \{p_1, \ldots, p_n\}$: set of pages in state $s$
- $G_s = (V, E)$: shadow page graph (vertices $V$, edges $E$)
- $\text{pid}_s \in \mathbb{N}$: next page ID counter
- $\text{did}_s \in \mathbb{N}$: next delta ID counter
- $T_s$: transaction log entries
- $B_s$: propagation buffers
- $\Sigma_s$: cached recompute signatures

### State Reset Operation
$$\text{reset}(s) = s' \text{ where } \begin{cases}
P_{s'} = \emptyset \\
G_{s'} = (\emptyset, \emptyset) \\
\text{pid}_{s'} = 1 \\
\text{did}_{s'} = 1 \\
T_{s'} = \emptyset \\
B_{s'} = \emptyset \\
\Sigma_{s'} = \emptyset
\end{cases}$$

### Invariants
1. **ID Consistency**: $\forall p \in P_s : p.\text{id} < \text{pid}_s$
2. **Graph Consistency**: $\forall (u,v) \in E : u,v \in V \land u,v \in \{p.\text{id} : p \in P_s\}$
3. **Delta Ordering**: $T_s = [\delta_1, \ldots, \delta_m] \implies \delta_i.\text{id} < \delta_j.\text{id} \text{ for } i < j$

### Deterministic Replay
$$\text{replay}(T_s, s_0) = s \iff \forall i \in [1..m] : \text{apply}(\delta_i, s_{i-1}) = s_i$$

## Implementation Requirements

### State Reset Function
Must clear in order:
1. `pages::Dict{PageID, Page}` - empty dictionary
2. `graph::ShadowPageGraph` - replace with new empty graph
3. `next_page_id[]` - reset to `PageID(1)`
4. `next_delta_id` - atomic store to `UInt64(1)`
5. `tlog_handle` - call `FFIWrapper.rust_tlog_clear!()`
6. `PROPAGATION_BUFFERS[state]` - delete entry if exists
7. Cached signatures in page metadata - cleared via page dict clear

### State Pool
- Pre-allocated pool of `MMSBState` objects
- Capacity: 10 states (tunable)
- Reset on checkout, return on completion
- **Expected performance**: allocation 6μs → 2-3μs

### Thread Safety
- Reset must be single-threaded (caller's responsibility)
- No concurrent operations during reset
- Atomic operations for delta ID allocation preserved

## Testing Requirements

### T3.3 - State Pool Determinism Test
Verify:
1. Pool checkout → reset → computation → return → checkout produces identical state
2. Replay from checkpoint produces same final state regardless of pool reuse
3. ID sequences match across reset cycles

### Checkpoint Replay Test
1. Run transaction sequence $T_1$
2. Checkpoint at state $s_k$
3. Reset and replay $T_1[1..k]$
4. Verify $s_k' = s_k$ (structural equality)

## Performance Targets
- State allocation: < 3μs (down from 6μs)
- Reset operation: < 1μs
- Pool contention: < 5% in concurrent tests

## References
- `src/01_types/MMSBState.jl` - state structure
- `src/04_propagation/PropagationEngine.jl` - buffer management
- `project_schedule/DAG-to-DO-DETAILED.md` - task dependencies
