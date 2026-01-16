use syn::Item;

// TODO: Advanced mutation operations
// - PartialReplaceOp (replace specific fields in struct, params in fn)
// - TransformOp (apply function to matched items)
// - ConditionalOp (apply mutation only if predicate holds)
// - BatchOp (apply multiple operations atomically)
// - InsertBeforeOp / InsertAfterOp (relative positioning)
// - MergeOp (combine multiple items)
// - ExtractOp (extract function/method from code block)
// - InlineOp (inline function calls)

/// Base trait for mutation operations
pub trait MutationOp: std::fmt::Debug + MutationOpClone {
    fn apply(&self, item: &Item) -> String;
}

/// Helper trait for cloning MutationOp trait objects
pub trait MutationOpClone {
    fn clone_box(&self) -> Box<dyn MutationOp>;
}

impl<T> MutationOpClone for T
where
    T: 'static + MutationOp + Clone,
{
    fn clone_box(&self) -> Box<dyn MutationOp> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn MutationOp> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Replace entire item with new code
#[derive(Debug, Clone)]
pub struct ReplaceOp {
    pub selector: String,
    pub replacement: String,
}

impl ReplaceOp {
    pub fn new(selector: impl Into<String>, replacement: impl Into<String>) -> Self {
        Self {
            selector: selector.into(),
            replacement: replacement.into(),
        }
    }
}

impl MutationOp for ReplaceOp {
    fn apply(&self, _item: &Item) -> String {
        self.replacement.clone()
    }
}

/// Wrap item with prefix and suffix
#[derive(Debug, Clone)]
pub struct WrapOp {
    pub prefix: String,
    pub suffix: String,
}

impl WrapOp {
    pub fn new(prefix: impl Into<String>, suffix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
            suffix: suffix.into(),
        }
    }
}

impl MutationOp for WrapOp {
    fn apply(&self, item: &Item) -> String {
        let item_str = quote::quote!(#item).to_string();
        format!("{}{}{}", self.prefix, item_str, self.suffix)
    }
}

/// Delete item
#[derive(Debug, Clone)]
pub struct DeleteOp;

impl MutationOp for DeleteOp {
    fn apply(&self, _item: &Item) -> String {
        String::new()
    }
}

/// Insert position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertPosition {
    Before,
    After,
}

/// Insert new code before or after item
#[derive(Debug, Clone)]
pub struct InsertOp {
    pub position: InsertPosition,
    pub content: String,
}

impl InsertOp {
    pub fn before(content: impl Into<String>) -> Self {
        Self {
            position: InsertPosition::Before,
            content: content.into(),
        }
    }

    pub fn after(content: impl Into<String>) -> Self {
        Self {
            position: InsertPosition::After,
            content: content.into(),
        }
    }
}

impl MutationOp for InsertOp {
    fn apply(&self, item: &Item) -> String {
        let item_str = quote::quote!(#item).to_string();
        match self.position {
            InsertPosition::Before => format!("{}\n{}", self.content, item_str),
            InsertPosition::After => format!("{}\n{}", item_str, self.content),
        }
    }
}
