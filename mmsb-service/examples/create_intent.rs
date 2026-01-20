// In mmsb-service/examples/create_intent.rs
use mmsb_service::Runtime;
use mmsb_intent::{IntentModule, Intent};

fn main() {
    let runtime = Runtime::new();
    let mut intent_module = IntentModule::new()
        .with_sink(runtime.event_bus().clone());
    
   let intent = Intent {
       description: "allocate 4KB".to_string(),
       metadata: "{}".to_string(),
   };
   
    let proof = intent_module.submit_intent(intent);
    println!("Intent submitted with hash: {:?}", proof.intent_hash);
}
