//! Minimal, correct example for mmsb-memory
//!
//! Demonstrates:
//! - MemoryEngine construction
//! - Delta construction using real types
//! - Correct crate boundaries (no private API access)

use std::path::PathBuf;

use mmsb_memory::{
    memory_engine::{MemoryEngine, MemoryEngineConfig},
    delta::{Delta, Source},
    epoch::Epoch,
    page::types::PageLocation,
};

use mmsb_primitives::{
    DeltaID,
    PageID,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ─────────────────────────────────────────────────────────────
    // Memory engine configuration (exact fields)
    // ─────────────────────────────────────────────────────────────
    let config = MemoryEngineConfig {
        tlog_path: PathBuf::from("/tmp/mmsb-example.tlog"),
        default_location: PageLocation::Cpu,
    };

    let _engine = MemoryEngine::new(config)?;

    // ─────────────────────────────────────────────────────────────
    // Construct a valid Delta
    // ─────────────────────────────────────────────────────────────
    let delta = Delta::new_dense(
        DeltaID(1),
        PageID(1),
        Epoch(1),
        vec![1, 2, 3, 4],              // payload
        vec![true, true, true, true],  // mask
        Source("example".to_string()), // provenance
    )?;

    println!("Constructed delta with id {:?}", delta.delta_id);

    // NOTE:
    // Admission, commit, and outcome recording are intentionally
    // private to mmsb-memory. This example demonstrates only
    // what the public API guarantees.

    Ok(())
}
