use super::anchor::AnchorSpec;
use crate::mutation::MutationPlan;

/// Upsert behavior on query miss
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnMissing {
    Error,
    Insert,
    Skip,
}

/// Upsert specification - declarative config
#[derive(Debug, Clone)]
pub struct UpsertSpec {
    pub plan: MutationPlan,
    pub on_missing: OnMissing,
    pub anchor: Option<AnchorSpec>,
    pub insert_value: Option<String>,
    pub allow_multiple: bool,
    pub dry_run: bool,
}

impl UpsertSpec {
    pub fn new(plan: MutationPlan) -> Self {
        Self {
            plan,
            on_missing: OnMissing::Error,
            anchor: None,
            insert_value: None,
            allow_multiple: false,
            dry_run: false,
        }
    }

    pub fn on_missing(mut self, on_missing: OnMissing) -> Self {
        self.on_missing = on_missing;
        self
    }

    pub fn anchor(mut self, anchor: AnchorSpec) -> Self {
        self.anchor = Some(anchor);
        self
    }

    pub fn insert_value(mut self, value: impl Into<String>) -> Self {
        self.insert_value = Some(value.into());
        self
    }

    pub fn allow_multiple(mut self, allow: bool) -> Self {
        self.allow_multiple = allow;
        self
    }

    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }
}
