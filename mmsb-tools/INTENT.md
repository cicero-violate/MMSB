## Variables

Let $\mathcal{Q}$ = Query DSL (declarative predicate composition)
Let $\mathcal{M}$ = Mutation DSL (declarative operation specification)
Let $\mathcal{U}$ = Upsert Engine (transactional apply with insert-on-miss)
Let $\mathcal{P}$ = Pure Observation (no filesystem writes)
Let $\mathcal{G}$ = Judgment Gate (token-based authority)
Let $\mathcal{C}_s$ = State Commit (TLog boundary)
Let $\mathcal{C}_d$ = Structural Commit (DAG boundary)
Let $\mathcal{I}$ = Intent Extraction (semantic symbol analysis)
Let $\mathcal{R}$ = Propagation (DAG traversal + rewrite)

## Conceptual Architecture

$$\text{JuliaSyntaxEdit Pattern} = \mathcal{Q} \circ \mathcal{M} \circ \mathcal{U}$$

where:
- **Query** matches AST nodes via predicates (kind, name, custom)
- **Mutation** declares operations (replace, wrap, delete, insert)
- **Upsert** applies transactionally with rollback + insert-on-miss

$$\text{MMSB Authority Flow} = \mathcal{P} \rightarrow \mathcal{I} \xrightarrow{\mathcal{G}} \mathcal{R} \rightarrow \mathcal{C}_s \times \mathcal{C}_d$$

where:
- **Pure observation** produces deltas without commits
- **Intent extraction** analyzes semantic changes (renames, deletes, signatures)
- **Judgment gate** guards propagation emission
- **Propagation** traverses DAG dependents and rewrites
- **Dual commit** persists state (TLog) and structure (DAG) at separate boundaries

## Desired Integration

$$\text{Rust Declarative Editor} = (\mathcal{Q} \circ \mathcal{M} \circ \mathcal{U}) \cap (\mathcal{P} \rightarrow \mathcal{I} \xrightarrow{\mathcal{G}} \mathcal{R} \rightarrow \mathcal{C}_s \times \mathcal{C}_d)$$

## Explanation

You want a **Rust-based structural editor** that combines:

1. **Declarative Query/Mutation/Upsert DSL** from JuliaSyntaxEdit.jl:
   - Query predicates compose declaratively (kind == Function, name == "foo", custom lambdas)
   - Mutation operations specify intent (replace, wrap, delete, insert with anchors)
   - Upsert engine handles insert-on-miss with anchor positioning (before/after :top/:bottom or specific nodes)
   - Conflict detection prevents overlapping byte ranges
   - Dry-run previews with diff generation

2. **MMSB Authority Model**:
   - Editor remains **pure/observational** - produces deltas without filesystem writes
   - Intent extraction analyzes semantic changes (rename symbol X→Y, delete symbol Z, signature change)
   - **JudgmentToken** gates propagation - derived deltas require explicit authority
   - Propagation traverses DAG dependents, rewrites affected pages
   - **Two-phase commit**: state deltas append to TLog, structural ops persist to DAG log

3. **Key Integration Points**:
   - Query/Mutation DSL operates on **Rust AST** (via syn or tree-sitter)
   - Upsert produces **Delta** objects (not filesystem writes)
   - Intent extraction bridges declarative mutations to semantic propagation
   - Judgment token controls when propagated rewrites emit deltas
   - Commit boundaries enforce authority model

The tool would let you write:
```rust
let query = query!(buffer, kind == Function && name == "process_data");
let mutation = mutate!(query, replace(:body, "{ rewritten_impl() }"));
let upsert_spec = upsert!(buffer, mutation, 
    on_missing = insert,
    anchor = after(:bottom),
    value = "fn process_data() { rewritten_impl() }"
);
```

But instead of immediately applying, it would:
1. Produce `Delta` objects
2. Extract `EditIntent` (e.g., "renamed process_data signature")
3. Require `JudgmentToken` for propagation
4. Traverse DAG to find dependents
5. Emit propagated deltas for affected pages
6. Caller commits via `commit_delta` (state) and `commit_structural_delta` (DAG)
