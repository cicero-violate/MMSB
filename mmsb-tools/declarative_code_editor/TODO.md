# TODO: Advanced Features & Hardening

## High Priority

### 1. Intent Extraction (Currently Stub)
- [x] Implement actual AST diffing in `src/intent/extraction.rs`
- [x] Detect RenameSymbol from before/after comparison
- [x] Detect SignatureChange (params, return type)
- [x] Detect ImportChange (use statement analysis)
- [ ] Track cascading changes (struct rename → method renames) (noted in comments)

### 2. Mutation Application (Currently No-Op)
- [x] Implement actual source transformation in `src/executor/mod.rs`
- [x] Apply transformations and update SourceBuffer
- [x] Handle partial replacements (field-level identifier replacement)
- [ ] Handle param-level and body-level replacements

### 3. Propagation Integration
- [ ] Connect to structural_code_editor propagation engine
- [ ] Build PageIndex from SourceBuffer
- [ ] Implement source rewriting for dependent pages
- [ ] Test propagation with actual DAG

## Query Enhancements

### Advanced Predicates
- [ ] `VisibilityPredicate` - match pub/private/pub(crate)
- [ ] `GenericPredicate` - match items with generics
- [ ] `AttributePredicate` - match by #[derive], #[cfg], etc.
- [ ] `SignaturePredicate` - match function signatures
- [ ] `BodyPredicate` - pattern match in function bodies
- [ ] `DocPredicate` - search doc comments

### Query Combinators
- [ ] `and()` - combine predicates with AND
- [ ] `or()` - combine predicates with OR
- [ ] `not()` - negate predicate
- [ ] Parent/child navigation (find impl for struct)
- [ ] Sibling queries (all methods in impl)

### Performance
- [ ] Index-based lookups instead of linear scan
- [ ] Query result caching with invalidation
- [ ] Incremental queries (only reparse changed parts)

## Mutation Operations

### New Operations
- [ ] `PartialReplaceOp` - replace struct fields, fn params
- [ ] `TransformOp` - apply function to matched items
- [ ] `ConditionalOp` - conditional mutations
- [ ] `BatchOp` - atomic multi-operation
- [ ] `InsertBeforeOp` / `InsertAfterOp` - relative positioning
- [ ] `MergeOp` - combine multiple items
- [ ] `ExtractOp` - extract function refactoring
- [ ] `InlineOp` - inline function calls

### Safety
- [ ] Conflict detection (overlapping edits)
- [ ] Deterministic ordering strategy
- [ ] Transaction rollback on error
- [ ] Dry-run preview mode

## Structural Classification

### Import Analysis
- [ ] Parse use statements
- [ ] Track module imports
- [ ] Detect re-exports (pub use)
- [ ] Cross-crate dependencies

### Semantic Analysis
- [ ] Breaking vs non-breaking changes
- [ ] Trait implementation tracking
- [ ] Orphan rule checking
- [ ] Feature gate analysis (#[cfg])

## SourceBuffer Improvements

### Performance
- [ ] Incremental parsing (reparse only changed regions)
- [ ] Memory mapping for large files
- [ ] Compression for storage
- [ ] Concurrent access (RwLock)

### Robustness
- [ ] Syntax error recovery (partial AST)
- [ ] Comment preservation
- [ ] Formatting preservation
- [ ] Span mapping through edits
- [ ] Undo/redo history

## Error Handling

### Rich Diagnostics
- [ ] Source span in error messages
- [ ] "Did you mean?" suggestions
- [ ] Warning system (non-fatal)
- [ ] Error aggregation
- [ ] LSP-compatible diagnostics
- [ ] Error codes for docs

## Bridge Orchestrator

### Multi-File Support
- [ ] Transaction across multiple files (all-or-nothing)
- [ ] Dependency ordering (topological sort)
- [ ] Conflict resolution strategies
- [ ] Preview all changes before commit

### Observability
- [ ] Performance metrics per stage
- [ ] Audit trail logging
- [ ] Progress reporting
- [ ] Error recovery (partial success)

## Testing & Validation

### Test Coverage
- [ ] Unit tests for each predicate type
- [ ] Integration tests for multi-file scenarios
- [ ] Roundtrip tests (parse → mutate → parse)
- [ ] Performance benchmarks

### Validation
- [ ] AST validity after mutations
- [ ] Semantic preservation checks
- [ ] Formatting validation
- [ ] Type-checking after transformations

## Integration

### MMSB Integration
- [ ] Real PageIndex building from SourceBuffer
- [ ] Actual propagation engine integration
- [ ] JudgmentToken integration tests
- [ ] Full TLog commit flow test

### Tooling
- [ ] LSP server integration
- [ ] CLI tool for batch operations
- [ ] Web interface for exploration
- [ ] VS Code extension

## Documentation

### User Guide
- [ ] Comprehensive examples for each feature
- [ ] Best practices guide
- [ ] Performance tuning guide
- [ ] Migration guide from manual edits

### API Documentation
- [ ] Rustdoc for all public items
- [ ] Usage examples in docs
- [ ] Architecture diagrams
- [ ] Decision records

## Future Features

### Advanced Analysis
- [ ] Control flow analysis
- [ ] Data flow analysis
- [ ] Taint tracking
- [ ] Dead code detection

### Refactoring
- [ ] Extract module
- [ ] Move item to different file
- [ ] Rename with preview
- [ ] Safe delete (check references)

### Code Generation
- [ ] Template-based generation
- [ ] Trait implementation scaffolding
- [ ] Test generation
- [ ] Boilerplate reduction
