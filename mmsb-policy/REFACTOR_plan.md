## Refactoring Analysis: Naming and Architecture

### Naming Decision

$$\text{Current: mmsb-intent (library)} \xrightarrow{\text{split}} \begin{cases}
\text{mmsb-policy (library)} \\
\text{mmsb-intent (entry point)}
\end{cases}$$

**YES ✓ - This is the correct split**

### Rationale

**mmsb-policy** (library):
- Classification logic
- Policy evaluation
- Scope definitions
- **Reusable** by multiple entry points

**mmsb-intent** (entry point + standard format):
- Standard intent file format
- Intent creation/validation
- Routing logic
- **Single point of entry**

### Mathematical Model

$$I_{\text{standard}} = \{\text{intent\_class}, \text{paths}, \text{tools}, \text{content}, \text{hash}\}$$

$$\text{route}: I_{\text{standard}} \rightarrow \begin{cases}
\text{StructuralOp} \rightarrow \text{Phase 1} \\
\text{StateDelta} \rightarrow \text{Phase 2}
\end{cases}$$

---

## Refactoring TODO Plan

### Phase 0: Rename and Reorganize (1-2 days)

**Tasks:**

1. **Rename mmsb-intent → mmsb-policy**
   - [x] Rename directory: `mmsb-intent/` → `mmsb-policy/`
   - [x] Update `Cargo.toml` package name
   - [x] Update workspace members
   - [x] Update all imports in existing code
   - [x] Run `cargo check` to verify

2. **Create mmsb-intent (new entry point crate)**
   - [ ] Create `mmsb-intent/` directory
   - [ ] Add to workspace
   - [ ] Dependencies: `mmsb-policy`, `mmsb-judgment`, `mmsb-core`

**Deliverable:** Two separate crates with clear responsibilities

---

### Phase 1: Define Standard Intent Format (2-3 days)

**Standard Intent Schema:**

```rust
// mmsb-intent/src/schema.rs
pub struct MmsbIntent {
    pub schema: String,              // "mmsb_intent.v1"
    pub intent_id: String,           // UUID
    pub intent_class: Vec<String>,   // ["formatting", "refactor"]
    pub operation_type: OperationType, // Structural | State
    pub content: IntentContent,
    pub metadata: IntentMetadata,
    pub issued_at: String,           // ISO8601
    pub hash: String,                // SHA256 of canonical form
}

pub enum OperationType {
    Structural(StructuralIntentContent),
    State(StateIntentContent),
}

pub struct StructuralIntentContent {
    pub ops: Vec<StructuralOp>,
    pub reason: String,
}

pub struct StateIntentContent {
    pub delta: Delta,
    pub reason: String,
}

pub struct IntentMetadata {
    pub affected_paths: Vec<String>,
    pub tools_used: Vec<String>,
    pub estimated_files: usize,
    pub estimated_lines: usize,
}
```

**Tasks:**

- [ ] Create `mmsb-intent/src/schema.rs` with standard format
- [ ] Create `mmsb-intent/schema/mmsb_intent.v1.schema.json`
- [ ] Implement serialization/deserialization
- [ ] Implement canonical hash function
- [ ] Add validation logic
- [ ] Write unit tests for schema

**Deliverable:** Standard intent format that replaces shell intents

---

### Phase 2: Build Intent Entry Point (3-4 days)

**Intent Router Architecture:**

```rust
// mmsb-intent/src/router.rs
pub struct IntentRouter {
    policy: ScopePolicy,
    judgment_required: bool,
}

impl IntentRouter {
    pub fn route(
        &self,
        intent: MmsbIntent
    ) -> Result<RoutingDecision, IntentError> {
        // 1. Validate intent
        // 2. Extract metadata
        // 3. Evaluate policy
        // 4. Route to correct phase
    }
}

pub enum RoutingDecision {
    AllowStructural {
        ops: Vec<StructuralOp>,
        token: JudgmentToken,
    },
    AllowState {
        delta: Delta,
        token: JudgmentToken,
    },
    RequireJudgment {
        violations: Vec<PolicyViolation>,
        intent_path: PathBuf,
    },
}
```

**Tasks:**

- [ ] Create `mmsb-intent/src/router.rs`
- [ ] Implement `IntentRouter::route()`
- [ ] Implement `IntentRouter::validate_intent()`
- [ ] Implement auto-token generation for allowed intents
- [ ] Add error types for routing failures
- [ ] Write integration tests

**Deliverable:** Working router that evaluates policy and routes to correct phase

---

### Phase 3: Migrate Shell Intent System (4-5 days)

**Migration Strategy:**

Replace shell intent files with standard MMSB intents:

```
OLD: shell_01_intent_abc123.json
NEW: mmsb_intent_abc123.json
```

**Tasks:**

- [ ] Create converter: `ShellIntent → MmsbIntent`
- [ ] Update `admission_proof.rs`:
  - [ ] Remove `list_shell_intents()`
  - [ ] Remove `load_intent()` 
  - [ ] Remove `compute_intent_hash()`
  - [ ] Add `load_mmsb_intent()`
- [ ] Update admission proof to use `IntentRouter`
- [ ] Update FFI to accept standard intent format
- [ ] Update Julia wrapper to create standard intents
- [ ] Add migration script for existing shell intents

**Deliverable:** Shell intents completely replaced by standard format

---

### Phase 4: Update mmsb-core Entry Points (3-4 days)

**Make mmsb-core policy-agnostic:**

```rust
// mmsb-core/src/01_page/page_commit.rs
pub(crate) fn commit_delta(
    log: &TransactionLog,
    token: &JudgmentToken,  // ← Doesn't know HOW token was obtained
    admission_proof: &MmsbAdmissionProof,
    execution_proof: &MmsbExecutionProof,
    delta: Delta,
    active_dag: Option<&DependencyGraph>,
) -> std::io::Result<()>
```

**Tasks:**

- [ ] Remove policy logic from `page_commit.rs` (already clean)
- [ ] Remove policy logic from `dag_commit.rs` (already clean)
- [ ] Ensure `mmsb-core` never imports `mmsb-policy`
- [ ] Update `submit_intent()` in `tick_orchestrator.rs` to use router
- [ ] Remove scattered judgment checks
- [ ] Verify dependency graph: `mmsb-core` → `mmsb-judgment` ONLY

**Deliverable:** mmsb-core is policy-free, only validates tokens

---

### Phase 5: Update mmsb-judgment Integration (2-3 days)

**Ensure mmsb-judgment stays isolated:**

```rust
// mmsb-judgment remains unchanged, but called differently:

// BEFORE (from mmsb-core):
let token = issue_judgment(&intent_metadata, &delta_hash);

// AFTER (from mmsb-intent):
let token = issue_judgment(&intent_metadata, &delta_hash);
// Then mmsb-intent passes token to mmsb-core
```

**Tasks:**

- [ ] Verify `mmsb-judgment` has NO dependencies on `mmsb-policy`
- [ ] Verify `mmsb-judgment` has NO dependencies on `mmsb-core`
- [ ] Update `mmsb-intent` to call judgment CLI when needed
- [ ] Add judgment artifact validation to intent router
- [ ] Update tests to use new flow

**Deliverable:** Judgment remains pure, called only from mmsb-intent

---

### Phase 6: Create Intent CLI Tool (2-3 days)

**Standard intent creation tool:**

```bash
# Create a structural intent
mmsb-intent create structural \
    --ops "add_edge(A, B)" \
    --reason "Add dependency" \
    --output intent_001.json

# Create a state intent
mmsb-intent create state \
    --delta-file delta.json \
    --reason "Update page value" \
    --output intent_002.json

# Validate an intent
mmsb-intent validate intent_001.json

# Route an intent (dry-run)
mmsb-intent route intent_001.json --dry-run

# Execute an intent
mmsb-intent execute intent_001.json --policy policy.json
```

**Tasks:**

- [ ] Create `mmsb-intent/src/bin/mmsb-intent.rs`
- [ ] Implement `create` subcommand
- [ ] Implement `validate` subcommand
- [ ] Implement `route` subcommand
- [ ] Implement `execute` subcommand
- [ ] Add usage examples
- [ ] Write CLI tests

**Deliverable:** CLI tool for intent management

---

### Phase 7: Update Documentation and Examples (2-3 days)

**Tasks:**

- [ ] Update `mmsb-policy/README.md`
- [ ] Create `mmsb-intent/README.md`
- [ ] Update `STRUCTURAL_vs_STATE_plan.md` with new flow
- [ ] Create intent creation guide
- [ ] Create policy authoring guide
- [ ] Update all examples to use standard intents
- [ ] Create migration guide from shell intents

**Deliverable:** Complete documentation of new system

---

### Phase 8: Deprecate Old Paths (1-2 days)

**Tasks:**

- [ ] Mark shell intent functions as `#[deprecated]`
- [ ] Add deprecation warnings to Julia FFI
- [ ] Create compatibility shims if needed
- [ ] Plan removal timeline
- [ ] Update CI/CD to use new paths

**Deliverable:** Old system deprecated, migration path clear

---

## Architecture After Refactoring

```
┌─────────────────────────────────────────────────┐
│              mmsb-intent (entry)                │
│  ┌───────────────────────────────────────────┐ │
│  │ Standard Intent Format (.v1.json)         │ │
│  │ - intent_class                            │ │
│  │ - operation_type (Structural | State)     │ │
│  │ - content                                 │ │
│  │ - metadata                                │ │
│  └───────────────────────────────────────────┘ │
│                      ↓                          │
│  ┌───────────────────────────────────────────┐ │
│  │ IntentRouter                              │ │
│  │ - validate()                              │ │
│  │ - classify()  ←──────┐                    │ │
│  │ - evaluate()  ←──────┤                    │ │
│  │ - route()            │                    │ │
│  └──────────┬───────────┴───────────────────┘ │
└─────────────┼─────────────────────────────────┘
              │                    │
      ┌───────┴────────┐    ┌──────────────────┐
      │                │    │   mmsb-policy    │
      │   Decision     │    │   (library)      │
      │                │    │  - Classifier    │
      └───┬────────┬───┘    │  - Evaluator     │
          │        │        │  - ScopePolicy   │
          │        │        └──────────────────┘
          │        │
          │        └──────────────┐
    Allow │                       │ RequireJudgment
          │                       │
          ↓                       ↓
  ┌────────────────┐      ┌─────────────────┐
  │ Auto-generate  │      │ mmsb-judgment   │
  │ JudgmentToken  │      │ - Ritual        │
  └───────┬────────┘      │ - Token gen     │
          │               └────────┬────────┘
          │                        │
          └────────┬───────────────┘
                   ↓
           ┌──────────────┐
           │  mmsb-core   │
           │  Phase 1 | 2 │
           └──────────────┘
```

## Timeline Summary

| Phase            | Duration | Dependencies |
|------------------+----------+--------------|
| 0: Rename        | 1-2 days | None         |
| 1: Schema        | 2-3 days | Phase 0      |
| 2: Router        | 3-4 days | Phase 1      |
| 3: Migrate Shell | 4-5 days | Phase 2      |
| 4: Update Core   | 3-4 days | Phase 3      |
| 5: Judgment      | 2-3 days | Phase 4      |
| 6: CLI           | 2-3 days | Phase 2      |
| 7: Docs          | 2-3 days | All          |
| 8: Deprecate     | 1-2 days | All          |

**Total: ~3-4 weeks**

**Critical path:** Phase 0 → 1 → 2 → 3 → 4 → 5
