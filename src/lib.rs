// src/lib.rs
#![allow(dead_code)]

#[path = "00_physical/mod.rs"]  pub mod physical;
#[path = "01_types/mod.rs"]      pub mod types;
#[path = "02_runtime/mod.rs"]   pub mod runtime;
#[path = "05_graph/mod.rs"]      pub mod graph;

mod ffi;
pub use ffi::*;
