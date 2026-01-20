//! Comprehensive MMSB Service Example
//! Demonstrates policy classification based on intent classes, paths, and tools

use mmsb_service::Runtime;
use mmsb_intent::{IntentModule, Intent};
use mmsb_policy::PolicyModule;
use mmsb_judgment::JudgmentModule;
use mmsb_events::AnyEvent;

#[tokio::main]
async fn main() {
    println!("=== MMSB Service Comprehensive Example ===\n");
    
    let runtime = Runtime::new();
    let event_bus = runtime.event_bus().clone();
    
    let mut intent_module = IntentModule::new().with_sink(event_bus.clone());
    let mut policy_module = PolicyModule::new().with_sink(event_bus.clone());
    let mut judgment_module = JudgmentModule::new().with_sink(event_bus.clone());
    
    let mut receiver = event_bus.subscribe();
    
    tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AnyEvent::IntentCreated(e) => {
                    println!("  → IntentCreated");
                    policy_module.handle_intent_created(e);
                }
                AnyEvent::PolicyEvaluated(e) => {
                    println!("  → PolicyEvaluated: {:?} (risk={:?})", 
                             e.policy_proof.category, e.policy_proof.risk_class);
                    judgment_module.handle_policy_evaluated(e);
                }
                AnyEvent::JudgmentApproved(e) => {
                    println!("  → JudgmentApproved: {}", 
                             if e.judgment_proof.approved { "✓ APPROVED" } else { "✗ DENIED" });
                }
                _ => {}
            }
        }
    });
    
    // Test 1: Safe formatting operation (would auto-approve if policy loaded)
    println!("Test 1: Formatting intent");
    intent_module.submit_intent(Intent {
        description: "Format main.rs".to_string(),
        intent_class: "formatting".to_string(),
        target_paths: vec!["src/main.rs".to_string()],
        tools_used: vec!["rustfmt".to_string()],
        files_touched: 1,
        diff_lines: 20,
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    // Test 2: Structural change - requires review
    println!("\nTest 2: Structural change");
    intent_module.submit_intent(Intent {
        description: "Refactor module structure".to_string(),
        intent_class: "structural_change".to_string(),
        target_paths: vec!["src/lib.rs".to_string(), "src/module.rs".to_string()],
        tools_used: vec!["patch_runner".to_string()],
        files_touched: 5,
        diff_lines: 200,
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    // Test 3: High-impact state change
    println!("\nTest 3: State change (high impact)");
    intent_module.submit_intent(Intent {
        description: "Modify database schema".to_string(),
        intent_class: "state_change".to_string(),
        target_paths: vec!["migrations/001_schema.sql".to_string()],
        tools_used: vec!["sql_runner".to_string()],
        files_touched: 10,
        diff_lines: 500,
    });
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    println!("\n=== Summary ===");
    println!("Policy classifies intents by:");
    println!("  • Intent class (formatting, structural_change, state_change, etc.)");
    println!("  • Target paths (allowed vs forbidden)");
    println!("  • Tools used (allowed vs forbidden)");
    println!("  • Scope (files_touched, diff_lines)");
    println!("\nJudgment exercises sole authority to approve/deny");
}
