/// Edit intent categories for semantic analysis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditIntent {
    /// Rename a symbol (function, struct, etc.)
    RenameSymbol { old: String, new: String },

    /// Delete a symbol
    DeleteSymbol { name: String },

    /// Add a new symbol
    AddSymbol { name: String, kind: String },

    /// Change function signature (params, return type)
    SignatureChange { name: String },

    /// Add/remove import
    ImportChange { path: String, added: bool },

    /// Modify module structure
    ModuleChange { module: String },
}

/// Intent category - determines pipeline routing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentCategory {
    /// STATE PIPELINE: Changes page content only
    State,

    /// STRUCTURAL PIPELINE: Changes DAG causality
    Structural,

    /// BOTH: Requires dual commit
    Both,
}

impl EditIntent {
    /// Categorize intent for pipeline routing
    pub fn category(&self) -> IntentCategory {
        match self {
            EditIntent::RenameSymbol { .. } => IntentCategory::State,
            EditIntent::DeleteSymbol { .. } => IntentCategory::State,
            EditIntent::AddSymbol { .. } => IntentCategory::State,
            EditIntent::SignatureChange { .. } => IntentCategory::State,
            EditIntent::ImportChange { .. } => IntentCategory::Structural,
            EditIntent::ModuleChange { .. } => IntentCategory::Both,
        }
    }
}
