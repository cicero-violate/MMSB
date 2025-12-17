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
  --julia-script ./src/julia/analyzer.jl
```

### Verbose Output

```bash
cargo run --release -- --verbose
```

### Skip Per-File Julia Analysis

If you only need Rust results plus the Julia layer/project CFGs, add `--skip-julia` to
avoid per-file Julia parsing while still emitting the higher-level DOT exports:

```bash
cargo run --release -- --skip-julia 
```

### Julia Analyzer Script

The Julia helpers now live alongside the Rust sources under `tools/mmsb-analyzer/src/`. The CLI defaults to
`--julia-script ./src/analyzer.jl`. When the Julia runtime is unavailable (e.g. in
restricted environments), the analyzer automatically falls back to an internal parser so the
reports continue to include Julia modules, structs, and functions.

## Generated Reports

All reports now live under `docs/analysis/` with one directory per category. Each directory exposes an `index.md` summary plus additional numbered files so `ls` already shows the intended reading order:

1. **`structure/`** - `index.md` summarizes counts, while `0xx-*.md` files group source files by MMSB prefix (e.g. `src/00_physical`).
2. **`call_graph/`** - `index.md` retains the call graph statistics and Mermaid diagram.
3. **`cfg/`** - Directory contains per-prefix CFG breakdowns so each file stays manageable; `index.md` lists the generated chunks.
4. **`cfg/dots/`** - Per-Julia-file Graphviz `call_graph.dot` exports derived from the SSA/CFG engine; Markdown pages link to these artifacts.
5. **`module_dependencies/`** - `index.md` summarizes module stats, and numbered files split imports, exports, submodules, and the current violations placeholder.
6. **`function_analysis/`** - Functions are bucketed alphabetically (`010-functions_A-F.md`, etc.) with per-layer/language details.
7. **`layer_dependencies/`** - `index.md` holds the combined Rust/Julia layer graph summary.

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
