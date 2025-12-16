# MMSB Analyzer

Intelligence substrate analyzer for the MMSB project. Provides deep static analysis of Rust and Julia codebases.

## Features

- **AST-based Rust parsing** using `syn` crate (not regex)
- **Julia integration** via FFI to Julia's native parser
- **Control flow analysis** with call graph generation
- **Module dependency tracking**
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

1. **structure.md** - Code structure by layer and language
2. **control_flow.md** - Call graphs with Mermaid diagrams
3. **module_dependencies.md** - Module import/export relationships
4. **function_analysis.md** - Detailed function signatures and call lists

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

1. **Run the analyzer** before making changes to understand structure
2. **Read generated reports** in `docs/analysis/` for context
3. **Update reports** after significant changes
4. **Use call graphs** to trace function dependencies
5. **Check module deps** before adding imports

The reports are designed to be easily parsed by LLM-based agents for understanding project architecture.
