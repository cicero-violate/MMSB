/// Result of upsert execution
#[derive(Debug, Clone)]
pub struct UpsertResult {
    /// Whether any changes were applied
    pub applied: bool,

    /// Whether new content was inserted
    pub inserted: bool,

    /// Optional diff preview (if dry_run)
    pub diff: Option<String>,

    /// Items affected by the operation
    pub affected_items: Vec<String>,
}

impl UpsertResult {
    pub fn new(applied: bool, inserted: bool) -> Self {
        Self {
            applied,
            inserted,
            diff: None,
            affected_items: Vec::new(),
        }
    }

    pub fn with_diff(mut self, diff: String) -> Self {
        self.diff = Some(diff);
        self
    }

    pub fn with_affected(mut self, items: Vec<String>) -> Self {
        self.affected_items = items;
        self
    }
}
