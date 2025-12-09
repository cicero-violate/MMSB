# AGENTS.md - Guide for AI Agents Working on MMSB

## Purpose

This document provides instructions for AI coding agents (like yourself) to work autonomously on the MMSB project using the planning documents.

---

## Core Documents

You have access to these planning documents:

1. **ARCHITECTURE.md** - Complete 13-layer system architecture
2. **PROJECT_SCHEDULE.md** - 20-week timeline with phases
3. **DAG_DEPENDENCIES.md** - Task dependency graph and checklist
4. **TASK_LOG.md** - Detailed task tracking log

---

## Agent Workflow

### Step 1: Read Architecture
```bash
cat /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/project_schedule/ARCHITECTURE.md
```

Understand:
- 13 layer structure
- Which layers use Rust vs Julia
- File locations and purposes
- Mathematical foundations

### Step 2: Check Schedule
```bash
cat /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/project_schedule/PROJECT_SCHEDULE.md
```

Know:
- Current phase and week
- What should be completed
- What's coming next
- Exit criteria

### Step 3: Find Your Task
```bash
cat /home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/project_schedule/DAG_DEPENDENCIES.md
```

Locate:
- Unblocked P0 tasks (critical path)
- Task dependencies
- What must complete before starting

### Step 4: Update Task Log
```bash
# Edit TASK_LOG.md to mark task as [⧗] in progress
```

Document:
- Task started
- Your agent name
- Current date

### Step 5: Execute Task

Follow this pattern:

**For file moves:**
```bash
# Create destination folder if needed
mkdir -p src/00_physical/

# Move file
git mv src/02_runtime/allocator.rs src/00_physical/allocator.rs

# Update imports in affected files
# Test that code still compiles
cargo test --lib
julia --project=. -e 'using MMSB'
```

**For new implementations:**
```bash
# Read existing similar code
# Implement new file following patterns
# Add tests
# Update mod.rs/module includes
# Run tests
```

**For refactoring:**
```bash
# Understand current implementation
# Plan refactor strategy
# Make changes incrementally
# Test after each change
# Ensure no regressions
```

### Step 6: Verify Work
```bash
# Run relevant tests
cargo test --lib
julia --project=. test/runtests.jl

# Check compilation
cargo check
julia --project=. -e 'using MMSB'

# Run benchmarks if performance-critical
cargo bench
```

### Step 7: Update Task Log
```bash
# Mark task as [✓] complete
# Add commit hash
# Note any issues or decisions
# Link to test results
```

### Step 8: Find Next Task
```bash
# Check DAG_DEPENDENCIES.md for newly unblocked tasks
# Prioritize P0 tasks on critical path
# Update schedule if ahead/behind
```

---

## Task Selection Strategy

### Priority Order:
1. **P0 tasks** with no blockers (critical path)
2. **P0 tasks** you can unblock by completing dependencies
3. **P1 tasks** if all P0s are blocked or complete
4. **Documentation** if code is done but docs lag

### When Blocked:
- Check if you can work on parallel task in different layer
- Consider P1 tasks that add value
- Document the blocker clearly
- Estimate when blocker will clear

---

## Code Quality Standards

### Rust Code:
- Use `rustfmt` for formatting
- Pass `clippy` with no warnings
- Add doc comments for public APIs
- Include unit tests for new functions
- Use `#[cfg(test)]` modules
- Prefer `Result<T, E>` over panics

### Julia Code:
- Follow existing style conventions
- Add docstrings with examples
- Include unit tests using `@testset`
- Prefer explicit types in public APIs
- Use `@inbounds` only when verified safe
- Profile performance-critical code

### Testing:
- Unit tests for individual functions
- Integration tests for layer interactions
- Performance benchmarks for critical paths
- Document test coverage
- Include edge cases

---

## File Organization Rules

### Moving Files:
1. Use `git mv` to preserve history
2. Update all imports in affected files
3. Check both Rust (`mod.rs`) and Julia (`include()`)
4. Update lib.rs and MMSB.jl if needed
5. Verify compilation before committing

### Creating New Files:
1. Follow naming convention (snake_case Rust, PascalCase Julia modules)
2. Add copyright header if applicable
3. Include module docstring explaining purpose
4. Add to appropriate `mod.rs` or include chain
5. Create corresponding test file

### Refactoring:
1. Preserve existing tests (they should still pass)
2. Add new tests for new functionality
3. Update documentation
4. Consider backwards compatibility
5. Note breaking changes in TASK_LOG.md

---

## Communication Protocol

### When You Complete a Task:
Update TASK_LOG.md with:
```
[✓] L0.2 - Move allocator.rs from 02_runtime/
    Owner: Agent-GPT4
    Started: 2025-12-09
    Completed: 2025-12-09
    Commit: abc123def
    Test Results: All tests pass (cargo test --lib)
    Notes: Updated lib.rs imports, no API changes
```

### When You Find Issues:
Document in TASK_LOG.md:
```
[✗] L2.3 - Implement semiring_ops.rs
    Owner: Agent-Claude
    Started: 2025-12-10
    Blockers: Need semiring trait definition from L2.2
    Notes: Cannot proceed until trait API is stable
```

### When You Make Decisions:
Add to TASK_LOG.md notes:
```
Notes: Chose HashMap over BTreeMap for page registry because
       benchmarks showed 3x faster lookups. See commit abc123.
```

---

## Common Patterns

### Pattern: Moving a Rust Module
```bash
# 1. Create destination
mkdir -p src/00_physical/

# 2. Move files
git mv src/02_runtime/allocator.rs src/00_physical/
git mv src/02_runtime/allocator_stats.rs src/00_physical/

# 3. Update mod.rs in destination
cat > src/00_physical/mod.rs << 'MODRS'
pub mod allocator;
pub mod allocator_stats;
// ...
MODRS

# 4. Update lib.rs
# Change: mod runtime;
# To: mod physical;

# 5. Update imports in other files
# Change: use crate::runtime::allocator
# To: use crate::physical::allocator

# 6. Test
cargo test --lib
```

### Pattern: Creating Julia Module
```julia
# src/02_semiring/Semiring.jl
module Semiring

export SemiringOps, ⊕, ⊗

"""
    SemiringOps

Abstract interface for semiring operations.
"""
abstract type SemiringOps end

# Implementation...

end # module
```

### Pattern: Adding Rust FFI
```rust
// In ffi.rs or layer-specific FFI file
#[no_mangle]
pub extern "C" fn rust_semiring_add(
    a: *const u8,
    b: *const u8,
    out: *mut u8,
) -> i32 {
    // Implementation
}
```

```julia
# In Julia wrapper
function semiring_add(a::Vector{UInt8}, b::Vector{UInt8})
    out = Vector{UInt8}(undef, length(a))
    GC.@preserve a b out begin
        code = ccall((:rust_semiring_add, LIBMMSB),
                    Int32, (Ptr{UInt8}, Ptr{UInt8}, Ptr{UInt8}),
                    a, b, out)
    end
    code == 0 || error("Semiring add failed")
    return out
end
```

---

## Testing Strategy

### Per Layer:
- **Layer 0-1:** Unit tests + integration tests for page lifecycle
- **Layer 2:** Algebraic property tests (associativity, identity, etc)
- **Layer 3:** Graph structure tests (cycle detection, topo sort)
- **Layer 4:** Propagation correctness + performance benchmarks
- **Layer 5-7:** Optimization verification (>20% improvement metrics)
- **Layer 8-9:** Reasoning/planning correctness on sample problems
- **Layer 10-12:** End-to-end agent interaction tests

### Test Naming:
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_allocator_basic() { }
    
    #[test]
    fn test_allocator_gpu_allocation() { }
    
    #[test]
    #[should_panic]
    fn test_allocator_invalid_size() { }
}
```

```julia
@testset "Semiring Operations" begin
    @test semiring_add([1,2], [3,4]) == [4,6]
    @test_throws ErrorException semiring_add([1], [1,2])
end
```

---

## Performance Benchmarking

### When to Benchmark:
- After completing Layer 0 (allocator)
- After completing Layer 4 (propagation)
- After completing Layer 5 (adaptive memory)
- Before Phase 1-4 exits

### How to Benchmark:
```bash
# Rust benchmarks
cargo bench

# Julia benchmarks
julia --project=. -e '
using BenchmarkTools
using MMSB
@benchmark allocate_page($state, 4096)
'
```

### Document Results:
```
Task: L4.9 - Benchmark propagation
Results:
- CPU propagation: 1.2ms for 1000-node DAG
- GPU propagation: 0.3ms for 1000-node DAG
- 4x speedup on GPU
- Memory usage: 10MB
Notes: GPU kernel limited by PCIe bandwidth
```

---

## Error Handling

### When Tests Fail:
1. Read error message carefully
2. Check if it's a real bug or test issue
3. Fix code or fix test
4. Document in TASK_LOG.md
5. Rerun tests
6. If still failing, mark task as blocked

### When Compilation Fails:
1. Check imports and module structure
2. Verify FFI boundaries
3. Look for typos in module names
4. Check Rust/Julia version compatibility
5. Document issue clearly

### When Stuck:
1. Review ARCHITECTURE.md for context
2. Look at similar existing code
3. Check git history for patterns
4. Document the issue precisely
5. Mark task as blocked with details

---

## Integration Points

### Rust ↔ Julia FFI:
- All FFI goes through `ffi/` layer
- Use `#[no_mangle]` and `extern "C"` in Rust
- Use `ccall()` with `GC.@preserve` in Julia
- Handle errors via return codes
- Document FFI contracts clearly

### Layer Dependencies:
- Never import from higher-numbered layer
- Use traits/interfaces for inversion
- Keep layers loosely coupled
- Document cross-layer contracts

---

## Completion Criteria

### Task Complete When:
- ✓ Code implemented
- ✓ Tests passing
- ✓ Documentation updated
- ✓ Benchmarks run (if applicable)
- ✓ No regressions
- ✓ TASK_LOG.md updated
- ✓ Git commit made

### Phase Complete When:
- ✓ All P0 tasks done
- ✓ Phase exit criteria met
- ✓ Integration tests passing
- ✓ Performance targets achieved
- ✓ Documentation complete

### Project Complete When:
- ✓ All 13 layers implemented
- ✓ All tests passing
- ✓ Performance benchmarks published
- ✓ 3+ demo applications working
- ✓ Documentation complete
- ✓ README updated

---

## Quick Reference

**Check current phase:**
```bash
grep -A 10 "^## Phase" PROJECT_SCHEDULE.md | head -20
```

**Find next unblocked P0 task:**
```bash
grep "| P0 |.*| ☐ |" DAG_DEPENDENCIES.md | head -5
```

**See what you're blocking:**
```bash
# Find tasks that depend on your current task
grep -A 5 "Blocking.*L0.2" DAG_DEPENDENCIES.md
```

**Update task status:**
```bash
# Change [☐] to [⧗] when starting
# Change [⧗] to [✓] when done
# Change [⧗] to [✗] when blocked
sed -i 's/\[☐\] L0.2/[⧗] L0.2/' TASK_LOG.md
```

---

## Remember

1. **Follow the DAG** - Don't start tasks with unmet dependencies
2. **Test continuously** - Run tests after every change
3. **Document decisions** - Future agents need context
4. **Maintain quality** - Don't rush for speed
5. **Update logs** - Keep TASK_LOG.md current
6. **Check architecture** - Stay aligned with design
7. **Preserve history** - Use git mv, not rm+add
8. **Think critically** - Question unclear requirements

---

## Getting Started

**First task for a new agent:**

1. Read ARCHITECTURE.md completely
2. Read PROJECT_SCHEDULE.md Phase 1
3. Read DAG_DEPENDENCIES.md Phase 1 section
4. Find first unblocked P0 task
5. Update TASK_LOG.md marking task in progress
6. Execute task following patterns above
7. Test thoroughly
8. Update TASK_LOG.md marking complete
9. Move to next task

**Good luck shipping MMSB! 🚀**

