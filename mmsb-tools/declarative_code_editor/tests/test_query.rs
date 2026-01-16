use declarative_code_editor::*;
use declarative_code_editor::query::ItemKind;
use std::path::PathBuf;

#[test]
fn test_execute_query_function() {
    let source = r#"
fn test_function() {}
struct TestStruct {}
"#;
    
    let buffer = SourceBuffer::new(PathBuf::from("test.rs"), source.to_string()).unwrap();
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function));
    
    let results = execute_query(&buffer, &query);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_execute_query_name() {
    let source = r#"
fn foo() {}
fn bar() {}
"#;
    
    let buffer = SourceBuffer::new(PathBuf::from("test.rs"), source.to_string()).unwrap();
    let query = QueryPlan::new()
        .with_predicate(NamePredicate::new("foo"));
    
    let results = execute_query(&buffer, &query);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_execute_query_combined() {
    let source = r#"
fn target() {}
struct target {}
"#;
    
    let buffer = SourceBuffer::new(PathBuf::from("test.rs"), source.to_string()).unwrap();
    let query = QueryPlan::new()
        .with_predicate(KindPredicate::new(ItemKind::Function))
        .with_predicate(NamePredicate::new("target"));
    
    let results = execute_query(&buffer, &query);
    assert_eq!(results.len(), 1);
}
