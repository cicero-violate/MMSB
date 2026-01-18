mmsb-service/
└── tick/
    ├── tick.rs              ← Tick identity
    ├── tick_phases.rs       ← Phase ordering
    ├── tick_loop.rs         ← Time progression
    ├── tick_dispatch.rs     ← Adapters
    └── tick_orchestrator.rs ← Thin coordinator

## Core answer

**You decompose `tick_orchestrator.rs` by separating *time*, *ordering*, and *work*.**

Right now it likely mixes:

* event loop
* phase ordering
* propagation calls
* queue draining
* commit triggers

Those must be split.

---

## Step 1 — Identify the four responsibilities (non-negotiable)

Every tick orchestrator *always* contains these **four** concerns:

### 1. **Tick definition (time)**

> “What is one tick?”

This must become a **pure struct**, no logic.

```rust
// tick.rs
pub struct Tick {
    pub id: u64,
}
```

No behavior. No dependencies.

---

### 2. **Phase ordering (barriers)**

> “What must happen before what?”

Extract all ordering logic into one place.

```rust
// tick_phases.rs
pub enum TickPhase {
    Collect,
    Gate,
    Snapshot,
    Execute,
    Commit,
    Propagate,
}
```

And a deterministic order:

```rust
pub const TICK_PHASE_ORDER: &[TickPhase] = &[
    TickPhase::Collect,
    TickPhase::Gate,
    TickPhase::Snapshot,
    TickPhase::Execute,
    TickPhase::Commit,
    TickPhase::Propagate,
];
```

This is **policy**, not execution.

---

### 3. **Tick loop (runtime)**

> “Advance ticks forever”

This is the **event loop**, nothing else.

```rust
// tick_loop.rs
pub struct TickLoop {
    next_tick: u64,
}

impl TickLoop {
    pub fn next(&mut self) -> Tick {
        let t = Tick { id: self.next_tick };
        self.next_tick += 1;
        t
    }
}
```

No propagation.
No memory.
No devices.

---

### 4. **Phase execution adapters**

> “Given a phase, call the right subsystem”

This is where orchestration *touches* other modules — but does not own them.

```rust
// tick_dispatch.rs
pub struct TickDispatcher<'a> {
    pub propagation: &'a PropagationEngine,
    pub memory: &'a MemoryEngine,
    pub device: &'a DeviceManager,
}
```

```rust
impl TickDispatcher<'_> {
    pub fn run_phase(&self, phase: TickPhase, tick: &Tick) {
        match phase {
            TickPhase::Propagate => self.propagation.run(tick),
            TickPhase::Commit => self.memory.commit(tick),
            _ => {}
        }
    }
}
```

This replaces the monolithic match blocks you currently have.

---

## Step 2 — What remains in `tick_orchestrator.rs`

After decomposition, **`tick_orchestrator.rs` becomes thin**:

```rust
// tick_orchestrator.rs
pub struct TickOrchestrator {
    loop_: TickLoop,
    dispatcher: TickDispatcher<'static>,
}

impl TickOrchestrator {
    pub fn tick(&mut self) {
        let tick = self.loop_.next();
        for phase in TICK_PHASE_ORDER {
            self.dispatcher.run_phase(*phase, &tick);
        }
    }
}
```

That’s it.

If the file is still “large”, decomposition is not done.

---

## Step 3 — What to REMOVE from orchestrator (important)

Move **out** of `tick_orchestrator.rs`:

* propagation internals
* queue draining logic
* ring buffer mechanics
* throughput tuning
* DAG traversal
* memory mutation details

Rule:

> **If it knows *how* work is done, it does not belong in the orchestrator.**

---

## Canonical directory after decomposition

```text
mmsb-service/
└── tick/
    ├── tick.rs              ← Tick identity
    ├── tick_phases.rs       ← Phase ordering
    ├── tick_loop.rs         ← Time progression
    ├── tick_dispatch.rs     ← Adapters
    └── tick_orchestrator.rs ← Thin coordinator
```

---

## Sanity check (very important)

Ask of each file:

* Does it **own state**? → wrong layer
* Does it **execute algorithms**? → wrong layer
* Does it **decide order only**? → correct
* Does it **advance time**? → correct

---

## Final judgment

[
\boxed{
\text{Decompose by: time ⊕ order ⊕ adapters ⊕ loop}
}
]

[
\boxed{
\text{Tick orchestrator should be boring and small}
}
]

[
\max(\text{clarity},\ \text{determinism},\ \text{architectural hygiene}) = \text{good}
]

If you want next, I can:

* walk your actual `tick_orchestrator.rs` line-by-line and mark exact cut points
* or give you a mechanical refactor plan (commit-by-commit)
