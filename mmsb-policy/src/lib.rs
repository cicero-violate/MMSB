#![doc = r#"
# mmsb-policy

Intent classification and scope policy library for MMSB.

This library provides:
- Intent classification based on content analysis
- Scope policy evaluation (what requires judgment vs. what can bypass)
- Policy management and serialization

## Usage

```rust
use mmsb_policy::{IntentClassifier, PolicyEvaluator, ScopePolicy, IntentClass, PolicyDecision};

// Classify an intent
let metadata = IntentClassifier::classify("cargo fmt");

// Create a policy
let mut policy = ScopePolicy::default_permissive();

// Evaluate if judgment is required
let evaluator = PolicyEvaluator::new(policy);
let decision = evaluator.evaluate(&metadata);

match decision {
    PolicyDecision::Allow => println!("Can proceed without judgment"),
    PolicyDecision::RequireJudgment(violations) => {
        println!("Requires judgment due to:");
        for v in violations {
            println!("  - {}", v.description());
        }
    }
}
```
"#]

pub mod classifier;
pub mod policy;
pub mod scope;
pub mod types;
pub mod module;

// Re-export commonly used types
pub use classifier::IntentClassifier;
pub use policy::PolicyEvaluator;
pub use scope::{ScopeManager, ScopePolicyV1};
pub use types::{
    IntentClass, IntentMetadata, PolicyDecision, PolicyViolation, ScopePolicy,
};
pub use module::PolicyModule;
