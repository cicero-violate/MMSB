//! Full judgment pipeline example
//! Demonstrates: Intent -> Policy -> Judgment flow

use mmsb_service::Runtime;
use mmsb_intent::{IntentModule, Intent};
use mmsb_policy::PolicyModule;
use mmsb_judgment::JudgmentModule;
use mmsb_events::AnyEvent;

#[tokio::main]
async fn main() {
    let runtime = Runtime::new();
    let event_bus = runtime.event_bus().clone();
    
    // Create modules
    let mut intent_module = IntentModule::new()
        .with_sink(event_bus.clone());
    let mut policy_module = PolicyModule::new()
        .with_sink(event_bus.clone());
    let mut judgment_module = JudgmentModule::new()
        .with_sink(event_bus.clone());
    
    // Subscribe to events to observe the pipeline
    let mut receiver = event_bus.subscribe();
    
    // Spawn event listener
    tokio::spawn(async move {
        while let Ok(event) = receiver.recv().await {
            match event {
                AnyEvent::IntentCreated(e) => {
                    println!("✓ IntentCreated: hash={:?}", &e.intent_hash[..4]);
                    policy_module.handle_intent_created(e);
                }
                AnyEvent::PolicyEvaluated(e) => {
                    println!("✓ PolicyEvaluated: category={:?}", e.policy_proof.category);
                    judgment_module.handle_policy_evaluated(e);
                }
                AnyEvent::JudgmentApproved(e) => {
                    println!("✓ JudgmentApproved: approved={}", e.judgment_proof.approved);
                }
                _ => {}
            }
        }
    });
    
   // Submit intent - this triggers the pipeline
   let intent = Intent {
       description: "allocate 4KB".to_string(),
        intent_class: "formatting".to_string(),
        target_paths: vec!["src/main.rs".to_string()],
        tools_used: vec!["rustfmt".to_string()],
        files_touched: 1,
        diff_lines: 10,
   };
   
   println!("Submitting intent...");
    intent_module.submit_intent(intent);
    
    // Wait for pipeline to complete
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    println!("\nPipeline complete!");
}
