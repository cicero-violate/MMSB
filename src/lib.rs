// src/lib.rs
#![allow(dead_code)]

#[path = "01_types/mod.rs"]      pub mod types;
#[path = "02_runtime/mod.rs"]   pub mod runtime;
#[path = "03_device/mod.rs"]     pub mod device;
#[path = "05_graph/mod.rs"]      pub mod graph;

mod ffi;
pub use ffi::*;
