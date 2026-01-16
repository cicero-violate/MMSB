use super::predicate::Predicate;
use crate::buffer::EditBuffer;
use syn::Item;

/// Query plan - declarative specification of what to find
#[derive(Debug, Clone)]
pub struct QueryPlan {
    predicates: Vec<Box<dyn Predicate>>,
}

impl QueryPlan {
    pub fn new() -> Self {
        Self {
            predicates: Vec::new(),
        }
    }

    pub fn with_predicate(mut self, predicate: impl Predicate + 'static) -> Self {
        self.predicates.push(Box::new(predicate));
        self
    }

    pub fn predicates(&self) -> &[Box<dyn Predicate>] {
        &self.predicates
    }

    /// Execute query against buffer's AST
    pub fn execute<'a>(&self, buffer: &'a EditBuffer) -> Vec<&'a Item> {
        let tree = buffer.tree();
        let mut results = Vec::new();

        for item in &tree.items {
            if self.matches(item) {
                results.push(item);
            }
        }

        results
    }

    fn matches(&self, item: &Item) -> bool {
        self.predicates.iter().all(|pred| pred.matches(item))
    }
}

impl Default for QueryPlan {
    fn default() -> Self {
        Self::new()
    }
}
