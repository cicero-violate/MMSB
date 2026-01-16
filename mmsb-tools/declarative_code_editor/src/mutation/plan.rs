use super::operations::MutationOp;
use crate::query::QueryPlan;

/// Mutation plan - declarative operations tied to a query
#[derive(Debug, Clone)]
pub struct MutationPlan {
    query: QueryPlan,
    operations: Vec<Box<dyn MutationOp>>,
}

impl MutationPlan {
    pub fn new(query: QueryPlan) -> Self {
        Self {
            query,
            operations: Vec::new(),
        }
    }

    pub fn with_operation(mut self, op: impl MutationOp + 'static) -> Self {
        self.operations.push(Box::new(op));
        self
    }

    pub fn query(&self) -> &QueryPlan {
        &self.query
    }

    pub fn operations(&self) -> &[Box<dyn MutationOp>] {
        &self.operations
    }
}
