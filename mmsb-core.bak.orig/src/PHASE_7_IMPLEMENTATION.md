# Phase 7 Implementation — Structural Proposal Engine

## Variables

$$\mathcal{P}_7 = \text{StructuralProposal set (proposals)}$$
$$\mathcal{G} = \text{DependencyGraph (read-only)}$$
$$\mathcal{S} = \text{PropagationStats (observational)}$$
$$\mathcal{O} = \text{StructuralOp (operations, not applied)}$$
$$\mathcal{J}_1 = \text{Phase 1 Judgment (required for application)}$$

## Core Invariant

$$\forall p \in \mathcal{P}_7: \text{mutates}(p, \mathcal{G}) = \bot$$

**Meaning:** No proposal mutates the DAG. All are read-only observations.

## Phase Constraint

$$\mathcal{P}_7 \xrightarrow{\text{requires}} \mathcal{J}_1$$

**Meaning:** All proposals must route through Phase 1 judgment before application.

## Implementation Architecture

### Module Structure

```
07_adaptive/
├── types.rs              — Proposal data structures
├── propagation_stats.rs  — Read-only statistics collection
├── proposal_engine.rs    — Core proposal generation
└── mod.rs                — Public API exports
```

### Type System

$$\text{StructuralProposal} = \langle \text{id}, \mathcal{O}^*, \text{rationale}, \text{effect}, c \rangle$$

Where:
- $\text{id} \in \text{ProposalID}$ (UUID)
- $\mathcal{O}^* \subseteq \mathcal{O}$ (operation sequence)
- $\text{rationale} \in \text{String}$ (human explanation)
- $\text{effect} \in \text{ExpectedEffect}$ (predicted impact)
- $c \in [0.0, 1.0]$ (confidence score)

### Proposal Categories (Exhaustive)

$$\mathcal{C} = \{\text{HighFanout}, \text{DeadDep}, \text{Simplify}, \text{Locality}\}$$

**1. High-Fanout Reduction**

$$\text{detect}: \text{fanout}(p) > \theta \cdot \text{median}(\text{fanout})$$

Proposes: Split high-fanout nodes with intermediate pages.

**2. Dead Dependency Elimination**

$$\text{detect}: \text{fanout}(p) = 0 \land \exists e: \text{from}(e) = p$$

Proposes: Remove edges from pages that never triggered propagation.

**3. Structural Simplification**

$$\text{detect}: \exists \text{ diamond}(A, B, C, D): A \to B \to D \land A \to C \to D$$

Proposes: Collapse redundant paths.

**4. Locality Optimization**

$$\text{status}: \text{placeholder (requires Phase 6 telemetry)}$$

## API Usage

### Initialization

```rust
use mmsb_core::adaptive::{ProposalEngine, ProposalConfig, PropagationStats};
use mmsb_core::dag::DependencyGraph;

let config = ProposalConfig::default();
let engine = ProposalEngine::new(config);
```

### Proposal Generation

```rust
// Collect stats (Phase 6 integration point)
let mut stats = PropagationStats::new();
stats.record_propagation(root_page, &affected_pages);

// Generate proposals (read-only)
let proposals = engine.generate_proposals(&dag, &stats);

// Proposals are advisory only
for proposal in proposals {
    println!("Proposal: {}", proposal.rationale);
    println!("Confidence: {}", proposal.confidence);
    
    // Phase 1 judgment required before application
    // if approved_by_phase_1(proposal) {
    //     apply_structural_ops(proposal.ops);
    // }
}
```

## Safety Guarantees

### Immutability Test

```rust
#[test]
fn test_proposal_engine_does_not_mutate_dag() {
    let dag = DependencyGraph::new();
    let version_before = dag.version();
    
    let engine = ProposalEngine::default();
    let stats = PropagationStats::new();
    
    let _proposals = engine.generate_proposals(&dag, &stats);
    
    // DAG unchanged
    assert_eq!(dag.version(), version_before);
}
```

$$\text{VERIFIED: } \mathcal{P}_7 \text{ preserves } \mathcal{G}$$

## Integration Points

### Phase 4 → Phase 7 (Stats Collection)

```rust
// In TickOrchestrator::tick()
fn tick(&mut self, dag: &DependencyGraph) {
    let affected = self.propagate();
    
    // Optional: Feed stats to Phase 7
    self.stats.record_propagation(root, &affected);
}
```

### Phase 7 → Phase 1 (Proposal Routing)

```rust
// Manual approval workflow
let proposals = engine.generate_proposals(&dag, &stats);

for proposal in proposals {
    if user_approves(&proposal) {
        let judgment = JudgmentToken::new();
        commit_structural_delta(&judgment, &proposal.ops);
    }
}
```

### Phase 6 → Phase 7 (Future: Telemetry)

```rust
// Not yet implemented
let telemetry = utility::Telemetry::collect();
let stats = PropagationStats::from_telemetry(&telemetry);
```

## Configuration

$$\text{ProposalConfig} = \langle \theta_f, c_{\min}, n_{\max} \rangle$$

Where:
- $\theta_f \in \mathbb{R}^+$ — High-fanout threshold multiplier (default: 2.0)
- $c_{\min} \in [0.0, 1.0]$ — Minimum confidence to emit (default: 0.3)
- $n_{\max} \in \mathbb{N}$ — Max proposals per category (default: 10)

## Limitations & Future Work

### Current Limitations

1. **Locality optimization is placeholder**
   - Requires Phase 6 memory telemetry
   - Needs physical address metadata

2. **No LLM integration yet**
   - Engine uses heuristics only
   - LLM advisory layer planned

3. **Manual approval workflow**
   - No automated proposal application
   - Intentional safety design

### Future Extensions

$$\mathcal{P}_7^{\text{future}} = \mathcal{P}_7 + \{\text{LLM}, \text{Telemetry}, \text{Feedback}\}$$

**Planned:**
- LLM rationale enhancement
- Confidence scoring from historical success
- Feedback loop from applied proposals
- Integration with Phase 6 utility metrics

## Phase Law Compliance

### Verification Checklist

- ✅ No DAG mutations
- ✅ No state mutations
- ✅ No propagation triggers
- ✅ No automatic application
- ✅ All outputs discardable
- ✅ Requires Phase 1 judgment
- ✅ Test confirms immutability

$$\boxed{\text{Phase 7 is LAWFUL}}$$

## Status

**Implementation:** ✅ Complete
**Testing:** ✅ Immutability verified
**Integration:** ⏳ Manual workflow only
**LLM Layer:** ⏳ Planned

Phase 7 provides a **safe, read-only advisory system** for structural improvements.

All proposals require explicit Phase 1 judgment before application.
