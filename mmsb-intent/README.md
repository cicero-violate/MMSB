# mmsb-intent

Intent classification and scope policy library for MMSB.

## Purpose

This library provides the infrastructure to:
1. Classify intent operations based on their content
2. Evaluate whether operations require human judgment
3. Define and manage scope policies that determine what can bypass judgment

## Architecture

```
Intent Content
      ↓
IntentClassifier → IntentMetadata
      ↓
PolicyEvaluator + ScopePolicy → PolicyDecision
      ↓
  Allow | RequireJudgment
```

## Core Components

### IntentClassifier

Analyzes intent content and extracts metadata:
- Classification (formatting, refactor, structural change, etc.)
- Affected paths
- Estimated complexity (files touched, diff lines)

### ScopePolicy

Defines rules for what operations can bypass judgment:
- Allowed intent classes
- Allowed/forbidden paths
- Allowed/forbidden tools
- Size limits (max files, max diff lines)

### PolicyEvaluator

Evaluates intent metadata against scope policy:
- Returns `PolicyDecision::Allow` if all rules pass
- Returns `PolicyDecision::RequireJudgment(violations)` otherwise

## Usage Example

```rust
use mmsb_intent::{
    IntentClassifier, PolicyEvaluator, ScopePolicy, 
    IntentClass, PolicyDecision
};

// Load intent content
let intent_content = std::fs::read_to_string("my_intent.patch")?;

// Classify the intent
let mut metadata = IntentClassifier::classify(&intent_content);
let paths = IntentClassifier::extract_paths(&intent_content);
let (files, lines) = IntentClassifier::estimate_complexity(&intent_content);

metadata = metadata
    .with_paths(paths)
    .with_files_touched(files)
    .with_diff_lines(lines);

// Create or load a policy
let mut policy = ScopePolicy::new("safe_operations");
policy.allowed_classes.insert(IntentClass::formatting());
policy.allowed_classes.insert(IntentClass::lint_fix());
policy.allowed_paths.push("src/**/*.rs".to_string());
policy.forbidden_paths.push("migrations/**".to_string());
policy.max_files_touched = Some(50);

// Evaluate
let evaluator = PolicyEvaluator::new(policy);
match evaluator.evaluate(&metadata) {
    PolicyDecision::Allow => {
        println!("Operation can proceed without judgment");
    }
    PolicyDecision::RequireJudgment(violations) => {
        println!("Judgment required due to:");
        for violation in violations {
            println!("  - {}", violation.description());
        }
    }
}
```

## Integration with mmsb-judgment

The intended workflow:

```rust
use mmsb_intent::{IntentClassifier, PolicyEvaluator, PolicyDecision};
use mmsb_judgment::issue::issue_judgment;

let metadata = IntentClassifier::classify(&content);
let evaluator = PolicyEvaluator::new(policy);

match evaluator.evaluate(&metadata) {
    PolicyDecision::Allow => {
        // Bypass judgment, proceed directly
        execute_intent(&content)?;
    }
    PolicyDecision::RequireJudgment(violations) => {
        // Must go through judgment ritual
        let token = issue_judgment(&metadata_str, &delta_hash);
        execute_with_token(&content, token)?;
    }
}
```

## Schema

See `schema/intent_policy.v1.schema.json` for the JSON schema definition.

Example policies are in `schema/example_policy.json`.

## Design Principles

1. **Safe by Default**: Unknown or unclassified intents require judgment
2. **Explicit Allowlisting**: Operations must be explicitly allowed by policy
3. **Composable**: Multiple policies can be combined
4. **Auditable**: Every decision includes violation reasons
5. **Extensible**: Classification can be enhanced with ML/pattern matching

## Future Enhancements

- [ ] ML-based intent classification
- [ ] Policy composition and inheritance
- [ ] Intent templates and patterns
- [ ] Historical analysis (learn from past judgments)
- [ ] Policy validation and testing framework
