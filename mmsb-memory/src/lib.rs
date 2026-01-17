//! MMSB Memory Module - Truth Authority
//! 
//! Owns canonical truth, time, and invariants.
//! Produces AdmissionProof (D), CommitProof (E), OutcomeProof (F).

#![allow(dead_code)]

pub mod memory_engine;
pub mod admission;
pub mod truth;
pub mod delta;
pub mod structural;
pub mod dag;
pub mod commit;
pub mod page;
pub mod epoch;
pub mod tlog;
pub mod replay;
pub mod outcome;
pub mod propagation;
pub mod materialization;
pub mod semiring;
pub mod physical;
pub mod optimization;
pub mod device;
pub mod proofs;

pub use memory_engine::MemoryEngine;
