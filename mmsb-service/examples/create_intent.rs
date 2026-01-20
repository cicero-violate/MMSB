// In mmsb-service/examples/create_intent.rs
use mmsb_service::Runtime;
use mmsb_intent::{IntentModule, Intent};

fn main() {
    let runtime = Runtime::new();
    let mut intent_module = IntentModule::new()
        .with_sink(runtime.event_bus().clone());
    
   let intent = Intent {
       description: "allocate 4KB".to_string(),
        intent_class: "formatting".to_string(),
        target_paths: vec!["src/main.rs".to_string()],
        tools_used: vec!["rustfmt".to_string()],
        files_touched: 1,
        diff_lines: 10,
   };
   
   let proof = intent_module.submit_intent(intent);
    println!("Intent submitted with hash: {:?}", proof.intent_hash);
}
