// src/lib.rs
#![allow(dead_code)]
#![allow(hidden_glob_reexports)]
// mmsb-core MUST NEVER issue judgments. It only validates tokens.

// Internal modules - not part of public API
mod logging;
#[path = "01_types/mod.rs"]     pub mod types;
#[path = "00_physical/mod.rs"]  mod physical;
#[path = "01_page/mod.rs"]      mod page;
#[path = "02_semiring/mod.rs"]  mod semiring;
#[path = "03_dag/mod.rs"]       pub mod dag;
#[path = "03_materialization/mod.rs"] mod materialization;
#[path = "04_propagation/mod.rs"] mod propagation;
#[path = "05_proof/mod.rs"]     mod proof;
#[path = "06_optimization/mod.rs"] mod optimization;
#[path = "06_utility/mod.rs"]   mod utility;
#[path = "07_adaptive/mod.rs"]  pub mod adaptive;

// FFI layer - C-compatible interface
pub mod ffi;

// Public prelude - official stable API surface
pub mod prelude;

// Re-export prelude at crate root for convenience
pub use prelude::*;
