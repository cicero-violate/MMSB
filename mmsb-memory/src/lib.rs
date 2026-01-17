// mmsb-memory - Domain-separated MMSB architecture
#![allow(dead_code)]
#![allow(hidden_glob_reexports)]

// Foundation layers
pub mod physical;
pub mod semiring;

// Core data structures
pub mod page;
pub mod epoch;
pub mod delta;
pub mod tlog;

// Graph and propagation
pub mod dag;
pub mod structural;
pub mod propagation;
pub mod materialization;

// Verification and optimization
pub mod proofs;
pub mod optimization;

// System layers
pub mod device;
pub mod commit;
pub mod replay;

// Control flow
pub mod admission;
pub mod truth;
pub mod outcome;
