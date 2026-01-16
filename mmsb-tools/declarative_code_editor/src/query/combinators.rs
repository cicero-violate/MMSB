//! Query combinator predicates
//!
//! Combinators allow composing predicates using logical operators:
//! - AND: All predicates must match
//! - OR: At least one predicate must match
//! - NOT: Predicate must not match

use super::predicate::Predicate;
use syn::Item;

/// Combine multiple predicates with AND logic
#[derive(Debug, Clone)]
pub struct AndPredicate {
    predicates: Vec<Box<dyn Predicate>>,
}

impl AndPredicate {
    pub fn new(predicates: Vec<Box<dyn Predicate>>) -> Self {
        Self { predicates }
    }
    
    pub fn of(p1: impl Predicate + 'static, p2: impl Predicate + 'static) -> Self {
        Self {
            predicates: vec![Box::new(p1), Box::new(p2)],
        }
    }
    
    pub fn add(mut self, predicate: impl Predicate + 'static) -> Self {
        self.predicates.push(Box::new(predicate));
        self
    }
}

impl Predicate for AndPredicate {
    fn matches(&self, item: &Item) -> bool {
        self.predicates.iter().all(|p| p.matches(item))
    }
}

/// Combine multiple predicates with OR logic
#[derive(Debug, Clone)]
pub struct OrPredicate {
    predicates: Vec<Box<dyn Predicate>>,
}

impl OrPredicate {
    pub fn new(predicates: Vec<Box<dyn Predicate>>) -> Self {
        Self { predicates }
    }
    
    pub fn of(p1: impl Predicate + 'static, p2: impl Predicate + 'static) -> Self {
        Self {
            predicates: vec![Box::new(p1), Box::new(p2)],
        }
    }
    
    pub fn add(mut self, predicate: impl Predicate + 'static) -> Self {
        self.predicates.push(Box::new(predicate));
        self
    }
}

impl Predicate for OrPredicate {
    fn matches(&self, item: &Item) -> bool {
        self.predicates.iter().any(|p| p.matches(item))
    }
}

/// Negate a predicate
#[derive(Debug, Clone)]
pub struct NotPredicate {
    predicate: Box<dyn Predicate>,
}

impl NotPredicate {
    pub fn new(predicate: impl Predicate + 'static) -> Self {
        Self {
            predicate: Box::new(predicate),
        }
    }
}

impl Predicate for NotPredicate {
    fn matches(&self, item: &Item) -> bool {
        !self.predicate.matches(item)
    }
}

/// Helper functions for ergonomic combinator usage
pub fn and(p1: impl Predicate + 'static, p2: impl Predicate + 'static) -> AndPredicate {
    AndPredicate::of(p1, p2)
}

pub fn or(p1: impl Predicate + 'static, p2: impl Predicate + 'static) -> OrPredicate {
    OrPredicate::of(p1, p2)
}

pub fn not(predicate: impl Predicate + 'static) -> NotPredicate {
    NotPredicate::new(predicate)
}
