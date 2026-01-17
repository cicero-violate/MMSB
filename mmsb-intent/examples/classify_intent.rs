use mmsb_intent::{
    IntentClass, IntentClassifier, PolicyDecision, PolicyEvaluator, ScopePolicy,
};

fn main() {
    // Example 1: Simple formatting intent
    println!("=== Example 1: Formatting Intent ===");
    let formatting_intent = r#"
*** Update File: src/main.rs
-fn main(){
+fn main() {
     println!("Hello, world!");
 }
    "#;

    let mut metadata = IntentClassifier::classify(formatting_intent);
    let paths = IntentClassifier::extract_paths(formatting_intent);
    let (files, lines) = IntentClassifier::estimate_complexity(formatting_intent);

    metadata = metadata
        .with_paths(paths)
        .with_files_touched(files)
        .with_diff_lines(lines);

    println!("Classes: {:?}", metadata.classes);
    println!("Paths: {:?}", metadata.affected_paths);
    println!("Files touched: {}", metadata.files_touched);
    println!("Diff lines: {}", metadata.diff_lines);

    // Create a permissive policy
    let mut policy = ScopePolicy::new("example");
    policy.allowed_classes.insert(IntentClass::formatting());
    policy.allowed_classes.insert(IntentClass::lint_fix());
    policy.allowed_paths.push("src/".to_string());
    policy.max_files_touched = Some(10);
    policy.max_diff_lines = Some(100);

    let evaluator = PolicyEvaluator::new(policy);
    match evaluator.evaluate(&metadata) {
        PolicyDecision::Allow => println!("✓ Decision: ALLOW (no judgment required)\n"),
        PolicyDecision::RequireJudgment(violations) => {
            println!("✗ Decision: REQUIRE JUDGMENT");
            for v in violations {
                println!("  - {}", v.description());
            }
            println!();
        }
    }

    // Example 2: Structural change in forbidden path
    println!("=== Example 2: Structural Change in Forbidden Path ===");
    let structural_intent = r#"
*** Update File: migrations/001_create_users.sql
+CREATE TABLE users (
+    id SERIAL PRIMARY KEY,
+    name TEXT NOT NULL
+);
    "#;

    let mut metadata = IntentClassifier::classify(structural_intent);
    let paths = IntentClassifier::extract_paths(structural_intent);
    let (files, lines) = IntentClassifier::estimate_complexity(structural_intent);

    metadata = metadata
        .with_paths(paths)
        .with_files_touched(files)
        .with_diff_lines(lines)
        .with_class(IntentClass::structural_change());

    println!("Classes: {:?}", metadata.classes);
    println!("Paths: {:?}", metadata.affected_paths);

    let mut policy = ScopePolicy::new("example");
    policy.allowed_classes.insert(IntentClass::formatting());
    policy.allowed_paths.push("src/".to_string());
    policy.forbidden_paths.push("migrations/".to_string());

    let evaluator = PolicyEvaluator::new(policy);
    match evaluator.evaluate(&metadata) {
        PolicyDecision::Allow => println!("✓ Decision: ALLOW\n"),
        PolicyDecision::RequireJudgment(violations) => {
            println!("✗ Decision: REQUIRE JUDGMENT");
            for v in violations {
                println!("  - {}", v.description());
            }
            println!();
        }
    }

    // Example 3: Too many files
    println!("=== Example 3: Too Many Files ===");
    let mut metadata = IntentClassifier::classify("formatting changes");
    metadata = metadata
        .with_class(IntentClass::formatting())
        .with_files_touched(100)
        .with_diff_lines(50);

    println!("Files touched: {}", metadata.files_touched);

    let mut policy = ScopePolicy::new("example");
    policy.allowed_classes.insert(IntentClass::formatting());
    policy.max_files_touched = Some(50);

    let evaluator = PolicyEvaluator::new(policy);
    match evaluator.evaluate(&metadata) {
        PolicyDecision::Allow => println!("✓ Decision: ALLOW\n"),
        PolicyDecision::RequireJudgment(violations) => {
            println!("✗ Decision: REQUIRE JUDGMENT");
            for v in violations {
                println!("  - {}", v.description());
            }
            println!();
        }
    }
}
