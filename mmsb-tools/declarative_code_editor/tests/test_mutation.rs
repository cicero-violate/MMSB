use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;

#[test]
fn test_apply_mutation_basic() {
    let source = "fn old_name() {}";
    
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function));
    
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "new_name"));
    
    let result = apply_mutation(source, &mutation);
    assert!(result.is_ok());
}

#[test]
fn test_apply_mutation_no_matches() {
    let source = "fn test() {}";
    
    let query = QueryPlan::new()
        .with_predicate(NamePredicate::new("nonexistent"));
    
    let mutation = MutationPlan::new(query)
        .with_operation(ReplaceOp::new("sig.ident", "new_name"));
    
    let result = apply_mutation(source, &mutation);
    assert!(result.is_err());
}
