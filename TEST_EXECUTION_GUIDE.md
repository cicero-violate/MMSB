# Test Execution Guide

## Quick Start

### Rust Tests
```bash
# Run all integration tests
cargo test --tests

# Run specific test suite
cargo test --test week27_31_integration
cargo test --test examples_basic

# With GPU features
cargo test --tests --features cuda
```

### Julia Tests
```bash
# Run all tests
julia test/runtests.jl

# Run specific suite
julia test/week27_31_integration.jl
julia test/examples_basic.jl

# Validate implementation
julia test/validate_week27_32.jl
```

## Test Structure

### Rust (`tests/`)
- `week27_31_integration.rs` - Week 27-31 features (24 tests)
- `examples_basic.rs` - Basic usage examples (5 tests)
- `mmsb_tests.rs` - Core functionality tests
- `delta_validation.rs` - Delta operation tests

### Julia (`test/`)
- `week27_31_integration.jl` - Week 27-32 features (40+ tests)
- `examples_basic.jl` - Basic usage examples (6 tests)
- `validate_week27_32.jl` - Implementation validation
- `test_*.jl` - Layer-specific tests

## Expected Results

### Week 27: Benchmarking
- Allocator latency < 1ms for small pages
- Semiring reduction < 10ms for 1000 elements
- Graph traversal < 5ms for 10-node DAG
- Propagation < 2ms per operation

### Week 28-29: GPU
- GPU allocation functional (if CUDA available)
- Command buffer operations succeed
- Memory pool reuse confirmed

### Week 30-31: Performance
- SIMD delta merge works correctly
- Lock-free allocator handles concurrent access
- Delta compression reduces size
- Batch propagation < 20ms for 20 pages

### Week 32: Reliability
- Retry succeeds after transient failures
- GPU fallback to CPU works
- LRU eviction correctly identifies cold pages
- Checkpoint validation detects corruption

## Troubleshooting

### CUDA Tests Fail
If GPU tests fail, disable with:
```bash
cargo test --tests --no-default-features
```

### Julia Package Issues
Ensure dependencies installed:
```julia
using Pkg
Pkg.instantiate()
```

### Validation Script Timeout
Run file checks manually:
```bash
ls -la src/00_physical/{allocator,gpu_memory_pool,lockfree_allocator}.rs
ls -la src/04_propagation/{gpu_propagation.cu,propagation_queue.rs}
ls -la src/06_utility/{cpu_features.rs,ErrorRecovery.jl,MemoryPressure.jl}
```
