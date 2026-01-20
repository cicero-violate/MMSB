use mmsb_service::Runtime;
use mmsb_intent::IntentModule;
use mmsb_policy::PolicyModule;
use mmsb_judgment::JudgmentModule;
use mmsb_events::Intent;

#[tokio::main]
async fn main() {
    let runtime = Runtime::new();
    let event_bus = runtime.event_bus().clone();
    
    let mut intent_module = IntentModule::new();
    let mut policy_module = PolicyModule::new();
    let mut judgment_module = JudgmentModule::new();
    
    let mut intent_rx = event_bus.subscribe_intent();
    let mut policy_rx = event_bus.subscribe_policy();
    let mut judgment_rx = event_bus.subscribe_judgment();
    
    tokio::spawn(async move {
        while let Ok(event) = intent_rx.recv().await {
            println!("  → IntentCreated");
            let policy_event = policy_module.handle_intent_created(event);
            println!("  → PolicyEvaluated: {:?}", policy_event.policy_proof.category);
            
            if let Some(judgment_event) = judgment_module.handle_policy_evaluated(policy_event) {
                println!("  → JudgmentApproved: ✓");
            }
        }
    });
    
    let intent = Intent {
        description: "Format code".to_string(),
        intent_class: "formatting".to_string(),
        target_paths: vec!["src/main.rs".to_string()],
        tools_used: vec!["rustfmt".to_string()],
        files_touched: 1,
        diff_lines: 10,
        max_duration_ms: 100,
        max_memory_bytes: 4096,
    };
    
    println!("Submitting intent...");
    let created = intent_module.submit_intent(intent);
    event_bus.emit_intent(created);
    
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    println!("Pipeline complete!");
}
