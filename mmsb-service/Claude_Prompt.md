# Follow-Up Prompt: Implement Reactive Service Execution Pipeline

## Context

You are continuing work on the MMSB (Multi-Modal State Bus) system - a Rust-based reactive architecture for deterministic computation with proof chains.

**Current State (COMPLETED):**
1. ✅ MemoryEngine - canonical truth owner (proofs D, E, F)
2. ✅ MemoryAdapter - implements StateBus (write) + MemoryReader (read)
3. ✅ CommitNotifier - zero-latency event notification system
4. ✅ Event-driven architecture - services subscribe to events (NO polling)
5. ✅ RuntimeContext - provides memory access + event subscriptions
6. ✅ EventListenerService - example service demonstrating event consumption

**Architecture:**
```
Services → subscribe_commits() → CommitNotifier → MemoryEngine
Services → with_state_bus()   → MemoryAdapter   → MemoryEngine
Services → memory_reader()    → MemoryAdapter   → MemoryEngine
```

**Location:** `/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/`

## Your Task

Implement the **complete reactive service execution pipeline** that processes work through the judgment → execution → commit → learning cycle.

## Implementation Goals

### Phase 1: Implement Real ExecutorService

**File:** `mmsb-service/src/services/executor_service.rs`

**Current State:** Stub implementation with unused `execute_one()` method

**What to Implement:**
```rust
impl Service for ExecutorService {
    fn run(&mut self, ctx: Arc<RuntimeContext>) -> Pin<Box<...>> {
        Box::pin(async move {
            // Subscribe to MemoryCommitted events
            let mut events = ctx.memory_reader().subscribe_commits();
            
            while let Ok(commit_event) = events.recv().await {
                // Extract AdmissionProof from commit_event
                let admission = commit_event.admission_proof;
                
                // Execute the work via ExecutionBus
                ctx.with_execution_bus(|eb| {
                    let outcome = eb.execute(admission);
                    eb.report_outcome(outcome);
                });
            }
        })
    }
}
```

**Key Points:**
- Service awaits commit events (zero polling)
- Extracts admission proof from MemoryCommitted
- Executes via ExecutionBus
- Reports outcomes back

### Phase 2: Implement Real CommitService

**File:** `mmsb-service/src/services/commit_service.rs`

**Current State:** Has `commit()` method but no event loop

**What to Implement:**
```rust
impl Service for CommitService {
    fn run(&mut self, ctx: Arc<RuntimeContext>) -> Pin<Box<...>> {
        Box::pin(async move {
            // Subscribe to execution outcomes
            // (Need ExecutionBus to emit ExecutionCompleted events!)
            
            while let Ok(outcome) = /* receive outcome */ {
                // Convert outcome to Fact
                let fact = outcome_to_fact(outcome);
                
                // Commit via StateBus
                ctx.with_state_bus(|sb| {
                    let commit_proof = sb.commit(fact).expect("commit failed");
                    // This triggers MemoryCommitted event!
                });
            }
        })
    }
}
```

**Challenge:** ExecutionBus needs to emit events! Follow the CommitNotifier pattern.

### Phase 3: Implement Real LearningService

**File:** `mmsb-service/src/services/learning_service.rs`

**Current State:** Has `learn()` method but no event loop

**What to Implement:**
```rust
impl Service for LearningService {
    fn run(&mut self, ctx: Arc<RuntimeContext>) -> Pin<Box<...>> {
        Box::pin(async move {
            // Subscribe to MemoryCommitted events
            let mut events = ctx.memory_reader().subscribe_commits();
            
            while let Ok(commit) = events.recv().await {
                // Learn from the commit
                ctx.with_learning_bus(|lb| {
                    let outcome = lb.observe_outcome(commit.commit_proof);
                    let knowledge = lb.derive_knowledge(outcome);
                    lb.report_knowledge(knowledge);
                });
            }
        })
    }
}
```

### Phase 4: Implement ExecutionBus Event Emission

**Problem:** ExecutionBus needs its own notifier (like CommitNotifier)

**Create:** `mmsb-executor/src/notifier/execution_notifier.rs` (follow CommitNotifier pattern)

**Event Type:** `ExecutionCompleted` (already defined in mmsb-events)

**Integration:**
1. Create ExecutionNotifier component
2. Inject into executor implementation
3. Emit ExecutionCompleted when execution finishes
4. CommitService subscribes to these events

### Phase 5: Wire Everything in main.rs

**Update main.rs to:**
1. Create ExecutionNotifier (if needed)
2. Register ALL services: Proposer, Judge, Executor, Commit, Learning
3. Services auto-coordinate through events
4. Remove stub implementations

## Critical Constraints

### MUST FOLLOW:
1. **Event-Driven Pattern:** Services MUST use `subscribe_commits()` - NO polling
2. **Zero Latency:** Events emitted inline with operations
3. **Dependency Injection:** Notifiers created in main.rs, injected into components
4. **Component Pattern:** Notifiers are infrastructure, NOT services
5. **Thread Safety:** Use `Arc<Mutex<>>` or `parking_lot::Mutex`
6. **Non-Blocking:** Event emission must not block on slow receivers

### DO NOT:
- ❌ Make notifiers into services (they're infrastructure!)
- ❌ Add polling loops (use async event streams!)
- ❌ Create circular dependencies
- ❌ Block on event sends (use broadcast with lagging policy)

## Architecture Pattern to Follow

**For each bus that needs events:**

```rust
// 1. Create notifier component (like CommitNotifier)
pub struct XNotifier {
    tx: broadcast::Sender<XEvent>,
}

// 2. Inject into component
pub struct XEngine {
    notifier: Arc<XNotifier>,
}

// 3. Emit events inline
impl XEngine {
    pub fn do_work(&self) -> Result<...> {
        // ... do work ...
        self.notifier.notify(event);
        Ok(result)
    }
}

// 4. Expose via trait
pub trait XReader: Send + Sync {
    fn subscribe_x(&self) -> broadcast::Receiver<XEvent>;
}

// 5. Services subscribe
let mut events = ctx.x_reader().subscribe_x();
while let Ok(event) = events.recv().await {
    // React to event
}
```

## Expected Event Flow

```
ProposerService
  ↓ creates Intent
JudgeService (subscribes to IntentCreated)
  ↓ produces JudgmentProof
Memory.admit() via StateBus
  ↓ produces AdmissionProof
  ↓ emits MemoryCommitted event
ExecutorService (subscribes to MemoryCommitted)
  ↓ executes work
  ↓ produces ExecutionOutcome
  ↓ emits ExecutionCompleted event
CommitService (subscribes to ExecutionCompleted)
  ↓ commits to StateBus
  ↓ triggers MemoryCommitted event
LearningService (subscribes to MemoryCommitted)
  ↓ derives knowledge
  ↓ updates learning state
```

## Testing Strategy

**Create:** `mmsb-service/examples/full_pipeline_test.rs`

**Test Flow:**
1. Manually inject a JudgmentProof into memory
2. Verify ExecutorService receives event
3. Verify CommitService receives outcome
4. Verify LearningService receives commit
5. Print full event trace

## Files to Modify/Create

**Modify:**
- `mmsb-service/src/services/executor_service.rs` - implement event loop
- `mmsb-service/src/services/commit_service.rs` - implement event loop
- `mmsb-service/src/services/learning_service.rs` - implement event loop
- `mmsb-service/src/main.rs` - register all services

**Create (if needed):**
- `mmsb-executor/src/notifier/execution_notifier.rs` - event emission
- `mmsb-service/examples/full_pipeline_test.rs` - integration test

**Keep Unchanged:**
- `mmsb-memory/` - already complete
- `mmsb-events/` - traits already defined
- Event notification pattern - already established

## Success Criteria

1. ✅ All services use event subscriptions (no polling)
2. ✅ Full pipeline processes work reactively
3. ✅ Events flow through all stages
4. ✅ System compiles without errors
5. ✅ Services log events as they process
6. ✅ Zero CPU usage when idle
7. ✅ Sub-microsecond latency between stages

## Questions to Ask

Before implementing, verify:
1. Do ExecutionBus implementations already exist in mmsb-executor?
2. Does ExecutionBus need event emission (like CommitNotifier)?
3. Should CommitService subscribe to ExecutionBus events or MemoryCommitted?
4. Are LearningBus implementations already in mmsb-learning?

## Documentation to Check

Look at existing code:
```bash
# See current service implementations
bat mmsb-service/src/services/*.rs

# See event definitions
bat mmsb-events/src/*.rs

# See CommitNotifier pattern
bat mmsb-memory/src/notifier/commit_notifier.rs

# See how main.rs wires things
bat mmsb-service/src/main.rs
```

## Git Commit Message Template

```
feat: implement reactive execution pipeline

Implemented full event-driven service coordination:

1. ExecutorService - subscribes to MemoryCommitted, executes work
2. CommitService - subscribes to execution outcomes, commits results
3. LearningService - subscribes to MemoryCommitted, derives knowledge
4. [If needed] ExecutionNotifier - emits ExecutionCompleted events

Event Flow:
  Memory → ExecutorService → CommitService → Memory → LearningService

Key Features:
- Zero polling (all services await events)
- Sub-microsecond latency
- Zero CPU when idle
- Complete reactive coordination

All services now operate through push-based event streams.
System ready for end-to-end proof chain execution.
```

---

**Start here:** Examine the current service implementations and decide whether ExecutionBus needs a notifier. Then implement the reactive event loops for each service following the established CommitNotifier pattern.
