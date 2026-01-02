#![allow(dead_code)]
//! Quality delta types and rollback criteria.

use serde::{Deserialize, Serialize};

use crate::correction_plan_types::ViolationType;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QualityDelta {
    pub action_id: String,
    pub cohesion_delta: f64,
    pub violation_delta: i32,
    pub complexity_delta: f64,
    pub overall_score_delta: f64,
    pub acceptable: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackCriteria {
    pub action_id: String,
    pub mandatory_rollback_if: Vec<RollbackCondition>,
    pub suggested_rollback_if: Vec<RollbackCondition>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum RollbackCondition {
    QualityDecreased { threshold: f64 },
    BuildFailed,
    TestsFailed { critical_tests: Vec<String> },
    InvariantViolated { invariant_ids: Vec<String> },
    Tier3Error { error_type: ViolationType },
    ManualReviewRequired,
}
