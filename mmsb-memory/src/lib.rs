// mmsb-memory - Domain-separated MMSB architecture
#![allow(dead_code)]
#![allow(hidden_glob_reexports)]

// Memory engine - canonical truth owner
pub mod memory_engine;

// Foundation layers
// pub mod physical;          // ← Substrate: MOVE to mmsb-executor
// pub mod semiring;          // ← Keep if semantic invariants, but check

// Core data structures
pub mod page;
pub mod epoch;
pub mod delta;
pub mod tlog;

// Graph and propagation
pub mod dag;
pub mod structural;
// pub mod propagation; // ← Substrate: MOVE to mmsb-executor
pub mod materialization;

// Verification and optimization
pub mod proofs;
// pub mod optimization; // ← Substrate: MOVE to mmsb-executor

// System layers
// pub mod device; // ← Substrate: MOVE to mmsb-executor
pub mod commit;
pub mod replay;

// Control flow
pub mod admission;
pub mod truth;
pub mod outcome;

// Event notification
pub mod notifier;

// Bus adapters
pub mod adapter;
