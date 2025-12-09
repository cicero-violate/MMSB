// src/lib.rs
#![allow(dead_code)]

#[path = "00_physical/mod.rs"]  pub mod physical;
#[path = "01_page/mod.rs"]      pub mod page;
#[path = "02_semiring/mod.rs"]  pub mod semiring;
#[path = "03_dag/mod.rs"]       pub mod dag;
#[path = "04_propagation/mod.rs"] pub mod propagation;
#[path = "05_adaptive/mod.rs"]  pub mod adaptive;
#[path = "06_utility/mod.rs"]   pub mod utility;

pub mod ffi;
pub use ffi::*;
