//! Example 6: Delta → Page Flow
//!
//! Shows how Delta produced by the bridge flows into MMSB Page system.

use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

fn main() {
    println!("=== Example 6: Delta → Page Flow ===\n");

    let source = r#"
fn compute(x: i32) -> i32 {
    x * 2
}
"#;

    let mut buffer = SourceBuffer::new(
        PathBuf::from("src/compute.rs"),
        source.to_string(),
    ).unwrap();

    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function));

    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "double"));

    let page_id = PageID(500);
    
    println!("📄 Page {} (src/compute.rs)", page_id.0);
    println!("BEFORE:");
    println!("{}", buffer.source());

    match BridgeOrchestrator::execute_and_bridge(&mut buffer, &mutation, page_id) {
        Ok(output) => {
            println!("\n✅ Bridge Output:");
            println!("├─ Page Deltas: {}", output.page_deltas.len());
            println!("├─ Structural Ops: {}", output.structural_ops.len());
            println!("└─ Route: {:?}", output.route);

            if let Some(delta) = output.page_deltas.first() {
                println!("\n📦 Delta Details:");
                println!("├─ Delta ID: {:?}", delta.delta_id);
                println!("├─ Page ID: {:?}", delta.page_id);
                println!("├─ Epoch: {:?}", delta.epoch);
                println!("├─ Payload size: {} bytes", delta.payload.len());
                println!("├─ Mask size: {} bits", delta.mask.len());
                println!("├─ Is sparse: {}", delta.is_sparse);
                println!("├─ Source: {:?}", delta.source);
                println!("└─ Timestamp: {}", delta.timestamp);

                println!("\n🔄 MMSB Integration Flow:");
                println!("1. Obtain JudgmentToken (state authority)");
                println!("   let judgment = mmsb_judgment::issue_token(...);");
                println!();
                println!("2. Commit delta to TLog");
                println!("   mmsb_core::commit_delta(&delta, &judgment)?;");
                println!();
                println!("3. TLog appends delta");
                println!("   - Delta persisted to transaction log");
                println!("   - Page state marked for update");
                println!();
                println!("4. Snapshot DAG (read-only)");
                println!("   - PropagationEngine gets current DAG");
                println!("   - Finds pages that depend on page {}", page_id.0);
                println!();
                println!("5. Propagation Engine");
                println!("   - Traverses dependents in DAG");
                println!("   - Rewrites each dependent page");
                println!("   - Emits derived deltas");
                println!();
                println!("6. Commit derived deltas");
                println!("   - Each dependent gets its own delta");
                println!("   - commit_delta() for each derived delta");
                println!();
                println!("7. Page materialization");
                println!("   - On read, Page reconstructs from TLog");
                println!("   - Applies all deltas in order");
                println!("   - Returns current state");
                
                println!("\n🎯 Key Properties:");
                println!("✓ Pure observation until commit");
                println!("✓ Judgment required for authority");
                println!("✓ DAG never mutated during propagation");
                println!("✓ Deterministic replay from TLog");
            }
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
