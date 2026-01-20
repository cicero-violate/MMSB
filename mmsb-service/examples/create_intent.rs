use mmsb_service::Runtime;
use mmsb_intent::IntentModule;
use mmsb_events::Intent;

fn main() {
    let runtime = Runtime::new();
    let mut intent_module = IntentModule::new();
    
    let intent = Intent {
        description: "allocate 4KB".to_string(),
        intent_class: "formatting".to_string(),
        target_paths: vec!["src/main.rs".to_string()],
        tools_used: vec!["rustfmt".to_string()],
        files_touched: 1,
        diff_lines: 10,
        max_duration_ms: 100,
        max_memory_bytes: 4096,
    };
    
    let created = intent_module.submit_intent(intent);
    runtime.event_bus().emit_intent(created.clone());
    println!("Intent submitted: {:?}", created.event_id);
}
