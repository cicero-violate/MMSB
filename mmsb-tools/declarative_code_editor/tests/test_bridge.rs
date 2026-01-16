use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use mmsb_core::types::PageID;
use std::path::PathBuf;

#[test]
fn test_bridge_execution() {
    let source = "fn old_name() {}";
    let page_id = PageID(12345);
    let file_path = PathBuf::from("test.rs");
    
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function));
    
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "new_name"));
    
    let result = BridgeOrchestrator::execute_and_bridge(
        source,
        &mutation,
        page_id,
        &file_path,
    );
    
    assert!(result.is_ok());
}

#[test]
fn test_bridge_output_has_delta() {
    let source = "fn test() {}";
    let page_id = PageID(12345);
    let file_path = PathBuf::from("test.rs");
    
    let query = QueryPlan::new()
        .with_predicate(NamePredicate::new("test"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "renamed"));
    
    let output = BridgeOrchestrator::execute_and_bridge(
        source,
        &mutation,
        page_id,
        &file_path,
    ).unwrap();
    
    assert!(!output.page_deltas.is_empty());
    assert_eq!(output.page_deltas[0].page_id, page_id);
}
