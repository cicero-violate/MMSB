use mmsb_service::Runtime;
use mmsb_intent::IntentModule;
use mmsb_policy::PolicyModule;
use mmsb_judgment::JudgmentModule;
use mmsb_events::Intent;

#[tokio::main]
async fn main() {
    println!("=== MMSB Service Example ===\n");
    
    let runtime = Runtime::new();
    let event_bus = runtime.event_bus().clone();
    
    let mut intent_module = IntentModule::new();
    let mut policy_module = PolicyModule::new();
    let mut judgment_module = JudgmentModule::new();
    
    let mut intent_rx = event_bus.subscribe_intent();
    
    tokio::spawn(async move {
        while let Ok(event) = intent_rx.recv().await {
            let policy_event = policy_module.handle_intent_created(event);
            println!("  → PolicyEvaluated: {:?} (risk={:?})", 
                     policy_event.policy_proof.category, policy_event.policy_proof.risk_class);
            
            if let Some(_) = judgment_module.handle_policy_evaluated(policy_event) {
                println!("  → JudgmentApproved: ✓");
            }
        }
    });
    
    // Test 1: Low risk
    println!("Test 1: Formatting");
    event_bus.emit_intent(intent_module.submit_intent(Intent {
        description: "Format".to_string(),
        intent_class: "formatting".to_string(),
        target_paths: vec!["src/main.rs".to_string()],
        tools_used: vec!["rustfmt".to_string()],
        files_touched: 1,
        diff_lines: 20,
        max_duration_ms: 100,
        max_memory_bytes: 4096,
    }));
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    // Test 2: Medium risk
    println!("\nTest 2: Structural change");
    event_bus.emit_intent(intent_module.submit_intent(Intent {
        description: "Refactor".to_string(),
        intent_class: "structural_change".to_string(),
        target_paths: vec!["src/lib.rs".to_string()],
        tools_used: vec!["patch_runner".to_string()],
        files_touched: 5,
        diff_lines: 200,
        max_duration_ms: 1000,
        max_memory_bytes: 1024 * 1024,
    }));
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    
    println!("\nPipeline complete!");
}
