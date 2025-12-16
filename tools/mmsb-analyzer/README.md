# MMSB Analyzer

Intelligence substrate analyzer for the MMSB project. 
It is designed to enforce the layered architecture: start from the true entry points (e.g. `src/MMSB.jl`) 
and surface every file/module/function/symbol that reaches “up” into the wrong layer so those violations can be corrected during refactors.

## Features

- **AST-based Rust parsing** using `syn` crate (not regex)
- **Julia integration** via FFI to Julia's native parser
- **Layer dependency enforcement** that models entry points and highlights cross-layer violations
- **Control flow analysis** with call graph generation
- **Module dependency tracking** (imports, exports, submodules)
- **Mermaid diagram generation** for visualization
- **Multi-language support** (Rust + Julia)

## Architecture

```
mmsb-analyzer/
├── src/
│   ├── main.rs              # CLI entry point
│   ├── types.rs             # Core data structures
│   ├── rust_parser.rs       # Rust AST analysis (syn-based)
│   ├── julia_parser.rs      # Julia FFI interface
│   ├── control_flow.rs      # Call graph building (petgraph)
│   └── report.rs            # Markdown report generation
├── julia_analyzer.jl        # Julia AST analyzer
└── Cargo.toml
```

## Usage

### Basic Analysis

```bash
cd tools/mmsb-analyzer
cargo run --release
```

### Custom Paths

```bash
cargo run --release -- \
  --root ../../ \
  --output ../../docs/analysis \
  --julia-script ./julia_analyzer.jl
```

### Verbose Output

```bash
cargo run --release -- --verbose
```

## Generated Reports

All reports are saved to `docs/analysis/`:

1. **structure.md** - Code structure grouped per file
2. **call_graph.md & cfg.md** - Call graph statistics + per-function CFGs
3. **module_dependencies.md** - Module import/export relationships
4. **function_analysis.md** - Function signatures and call lists
5. **layer_dependencies.md** - Layer ordering, cycles, violations, unresolved symbols

## Integration

### With CI/CD

Add to `.github/workflows/analysis.yml`:

```yaml
- name: Run Structure Analysis
  run: |
    cd tools/mmsb-analyzer
    cargo run --release
    git add ../../docs/analysis/
```

### With Build Script

Add to `build.rs`:

```rust
fn main() {
    // Run analyzer on significant changes
    std::process::Command::new("cargo")
        .args(&["run", "--manifest-path", "tools/mmsb-analyzer/Cargo.toml"])
        .status()
        .expect("Failed to run analyzer");
}
```

## Dependencies

### Rust
- `syn` - Rust AST parsing
- `petgraph` - Graph algorithms for call flow
- `walkdir` - Directory traversal
- `serde` / `serde_json` - Julia data interchange

### Julia
- Julia runtime (1.6+)
- Standard library only (no external packages)

## For Future AI Agents

This tool generates a comprehensive map of the MMSB codebase:

1. **Run the analyzer** before changes to understand the current layer state.
2. **Read the reports** (especially `layer_dependencies.md`) to see mis-layered symbols.
3. **Fix violations** by moving the referenced code into its correct layer.
4. **Use call graphs/CFGs** when reasoning about control flow impacts.
5. **Regenerate reports** to ensure the layered architecture remains consistent.

The reports are designed to be easily parsed by LLM-based agents for understanding project architecture.
