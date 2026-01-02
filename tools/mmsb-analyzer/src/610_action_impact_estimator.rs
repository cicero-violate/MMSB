#![allow(dead_code)]
//! Action impact estimator.

use crate::quality_delta_calculator::Metrics;

#[allow(unused_imports)] pub use crate::quality_delta_calculator::estimate_impact;
#[allow(unused_imports)] pub(crate) use crate::correction_intelligence_report::simulate_action;

#[derive(Clone, Debug)]
pub struct AnalysisState {
    pub metrics: Metrics,
}




