# Report Restructuring Checklist

Goal: keep every analyzer report easy to skim for LLM agents by splitting the current multi‑hundred line Markdown dumps into directory-based collections with predictable ordering.

1. **Create a directory per report type** under `docs/analysis/` (e.g. `structure/`, `cfg/`, `module_dependencies/`, `function_analysis/`, `layer_dependencies/`, `call_graph/`). Keep the summary page in `index.md` (or `00-summary.md`) so existing links stay simple.
2. **Split the large reports (>500 lines) into stable chunks** that reflect what the file enumerates:
   - `structure/` → one file per source tree prefix (`000-root.md`, `010-src-00_physical.md`, …) sorted lexicographically by prefix so `ls` already shows the order of traversal.
   - `cfg/` → per-prefix files (same rule as `structure/`) so each file hosts the functions belonging to that directory; this keeps each Mermaid graph close to its owning module.
   - `module_dependencies/` → publish `summary.md`, `imports.md`, `exports.md`, and `submodules.md` so readers can jump straight to the relationship they need.
   - `function_analysis/` → alphabetical buckets such as `functions_A-F.md`, `functions_G-L.md`, etc., since agents normally look up functions by name.
3. **Drop a short README (or header section) inside every directory** describing how files are ordered and where new summaries belong, so a plain `ls` communicates the structure.
4. **Update `tools/mmsb-analyzer/README.md`** to describe the new layout and call out the directory naming conventions so humans and agents know where to look.
