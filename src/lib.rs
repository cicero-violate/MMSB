// src/lib.rs
#![allow(dead_code)]

#[path = "00_physical/mod.rs"]  pub mod physical;
#[path = "01_page/mod.rs"]      pub mod page;
#[path = "02_semiring/mod.rs"]  pub mod semiring;
#[path = "03_dag/mod.rs"]       pub mod dag;
#[path = "04_propagation/mod.rs"] pub mod propagation;

mod ffi;
pub use ffi::*;
