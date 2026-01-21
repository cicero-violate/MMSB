cargo build

cargo test -p mmsb-service --test execution_propagation_flow
cargo test -p mmsb-service --test memory_mutation_integration

cargo run --example create_intent
cargo run --example judgment_pipeline
cargo run --example mmsb_service_example

