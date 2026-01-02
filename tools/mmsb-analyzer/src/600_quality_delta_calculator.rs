#![allow(dead_code)]
//! Quality delta calculator.

#[allow(unused_imports)] pub use crate::correction_intelligence_report::estimate_impact;

#[derive(Clone, Debug, Default)]
pub struct Metrics {
    pub cohesion: f64,
    pub violations: usize,
    pub complexity: f64,
}





#[allow(unused_imports)] pub use crate::correction_intelligence_report::calculate_quality_delta;
