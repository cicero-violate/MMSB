//! Markdown report generation

use crate::control_flow::ControlFlowAnalyzer;
use crate::dependency::LayerGraph;
use crate::types::*;
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

/// Compress absolute paths to MMSB-relative format
fn compress_path(path: &str) -> String {
    // Find MMSB in the path and return everything from there
    if let Some(idx) = path.find("/MMSB/") {
        return format!("MMSB{}", &path[idx + 5..]);
    }
    // If already starts with MMSB/, return as-is
    if path.starts_with("MMSB/") {
        return path.to_string();
    }
    // Fallback: try to find src/ or other common markers
    if let Some(idx) = path.rfind("/src/") {
        return format!("MMSB/src{}", &path[idx + 4..]);
    }
    // Last resort: return original
    path.to_string()
}

pub struct ReportGenerator {
    output_dir: String,
}

impl ReportGenerator {
    pub fn new(output_dir: String) -> Self {
        Self { output_dir }
    }

    pub fn generate_all(
        &self,
        result: &AnalysisResult,
        cf_analyzer: &ControlFlowAnalyzer,
        rust_layers: &LayerGraph,
        julia_layers: &LayerGraph,
    ) -> Result<()> {
        fs::create_dir_all(&self.output_dir)?;
        self.cleanup_legacy_reports()?;

        self.generate_structure_report(result)?;
        self.generate_call_graph_report(cf_analyzer)?;
        self.generate_cfg_report(cf_analyzer)?;
        self.generate_module_dependencies(result)?;
        self.generate_function_analysis(result)?;
        self.generate_layer_dependency_report(rust_layers, julia_layers)?;

        Ok(())
    }

    fn cleanup_legacy_reports(&self) -> Result<()> {
        let legacy_files = [
            "structure.md",
            "call_graph.md",
            "cfg.md",
            "module_dependencies.md",
            "function_analysis.md",
            "layer_dependencies.md",
        ];
        for file in legacy_files {
            let path = Path::new(&self.output_dir).join(file);
            if path.exists() {
                fs::remove_file(path)?;
            }
        }
        let report_dirs = [
            "structure",
            "call_graph",
            "cfg",
            "module_dependencies",
            "function_analysis",
            "layer_dependencies",
        ];
        for dir in report_dirs {
            let path = Path::new(&self.output_dir).join(dir);
            if !path.exists() {
                continue;
            }
            for entry in fs::read_dir(&path)? {
                let entry = entry?;
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    if dir == "cfg" && entry_path.file_name().map_or(false, |n| n == "dots") {
                        continue;
                    }
                    fs::remove_dir_all(entry_path)?;
                } else {
                    fs::remove_file(entry_path)?;
                }
            }
        }
        Ok(())
    }

    fn prepare_report_dir(&self, name: &str) -> Result<PathBuf> {
        let dir = Path::new(&self.output_dir).join(name);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    fn generate_structure_report(&self, result: &AnalysisResult) -> Result<()> {
        let dir = self.prepare_report_dir("structure")?;
        let generated_at = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");

        let mut files: BTreeMap<String, Vec<&CodeElement>> = BTreeMap::new();
        for elem in &result.elements {
            files
                .entry(elem.file_path.clone())
                .or_insert_with(Vec::new)
                .push(elem);
        }

        let mut grouped: BTreeMap<String, Vec<(String, Vec<&CodeElement>)>> = BTreeMap::new();
        for (file_path, elements) in files {
            let compressed = compress_path(&file_path);
            let key = prefix_key_from_path(&compressed);
            grouped
                .entry(key)
                .or_insert_with(Vec::new)
                .push((compressed, elements));
        }

        let mut grouped: Vec<_> = grouped.into_iter().collect();
        grouped.sort_by(|a, b| group_key_cmp(&a.0, &b.0));

        let mut index = String::from("# MMSB Code Structure Overview\n\n");
        index.push_str(&format!("Generated: {}\n\n", generated_at));
        index.push_str(
            "Each numbered file groups source files by MMSB prefix so a simple `ls structure/` \
shows the traversal order.\n\n",
        );

        if grouped.is_empty() {
            index.push_str("No code elements were recorded.\n");
        } else {
            index.push_str("## Group Files\n\n");
            for (idx, (group_key, _)) in grouped.iter().enumerate() {
                let slug = slugify_key(group_key);
                let file_name = format!("{:03}-{}.md", idx * 10, slug);
                index.push_str(&format!("- `{}` → `{}`\n", group_key, file_name));
            }
        }

        for (idx, (group_key, mut entries)) in grouped.into_iter().enumerate() {
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            let slug = slugify_key(&group_key);
            let file_name = format!("{:03}-{}.md", idx * 10, slug);
            let mut content = format!("# Structure Group: {}\n\n", group_key);

            for (file_path, mut elements) in entries {
                content.push_str(&format!("## File: {}\n\n", file_path));

                let layers: BTreeSet<String> = elements.iter().map(|e| e.layer.clone()).collect();
                let layer_summary = if layers.is_empty() {
                    "root".to_string()
                } else {
                    layers.iter().cloned().collect::<Vec<_>>().join(", ")
                };

                let mut language_counts: BTreeMap<String, usize> = BTreeMap::new();
                let mut type_counts: BTreeMap<String, usize> = BTreeMap::new();
                for elem in &elements {
                    *language_counts
                        .entry(language_label(&elem.language).to_string())
                        .or_insert(0) += 1;
                    *type_counts
                        .entry(format!("{:?}", elem.element_type))
                        .or_insert(0) += 1;
                }

                let lang_summary = if language_counts.is_empty() {
                    "n/a".to_string()
                } else {
                    language_counts
                        .iter()
                        .map(|(lang, count)| format!("{} ({})", lang, count))
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                let type_summary = if type_counts.is_empty() {
                    "n/a".to_string()
                } else {
                    type_counts
                        .iter()
                        .map(|(ty, count)| format!("{} ({})", ty, count))
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                content.push_str(&format!("- Layer(s): {}\n", layer_summary));
                content.push_str(&format!("- Language coverage: {}\n", lang_summary));
                content.push_str(&format!("- Element types: {}\n", type_summary));
                content.push_str(&format!("- Total elements: {}\n\n", elements.len()));

                content.push_str("### Elements\n\n");
                elements.sort_by(|a, b| {
                    a.line_number
                        .cmp(&b.line_number)
                        .then_with(|| a.name.cmp(&b.name))
                });
                for elem in elements {
                    content.push_str(&self.format_element_entry(elem));
                }
                content.push('\n');
            }

            fs::write(dir.join(file_name), content)?;
        }

        // Summary statistics
        index.push_str("\n## Summary Statistics\n\n");
        index.push_str(&format!("- Total elements: {}\n", result.elements.len()));
        index.push_str(&format!(
            "- Rust elements: {}\n",
            result
                .elements
                .iter()
                .filter(|e| matches!(e.language, Language::Rust))
                .count()
        ));
        index.push_str(&format!(
            "- Julia elements: {}\n",
            result
                .elements
                .iter()
                .filter(|e| matches!(e.language, Language::Julia))
                .count()
        ));

        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for elem in &result.elements {
            let key = format!("{:?}_{:?}", elem.language, elem.element_type);
            *type_counts.entry(key).or_insert(0) += 1;
        }

        index.push_str("\n### Elements by Type\n\n");
        let mut sorted_types: Vec<_> = type_counts.iter().collect();
        sorted_types.sort_by_key(|(k, _)| k.as_str());
        for (type_name, count) in sorted_types {
            index.push_str(&format!("- {}: {}\n", type_name, count));
        }

        fs::write(dir.join("index.md"), index)?;
        Ok(())
    }

    fn format_element_entry(&self, elem: &CodeElement) -> String {
        let mut entry = format!(
            "- [{} | {:?}] `{}` (line {}, {})\n",
            language_label(&elem.language),
            elem.element_type,
            elem.name,
            elem.line_number,
            visibility_label(&elem.visibility),
        );

        if !elem.signature.is_empty()
            && matches!(
                elem.element_type,
                ElementType::Function | ElementType::Struct
            )
        {
            entry.push_str(&format!(
                "  - Signature: `{}`\n",
                short_signature(&elem.signature)
            ));
        }

        if !elem.generic_params.is_empty() {
            entry.push_str(&format!(
                "  - Generics: {}\n",
                elem.generic_params.join(", ")
            ));
        }

        if matches!(elem.element_type, ElementType::Function) && !elem.calls.is_empty() {
            entry.push_str(&format!("  - Calls: {}\n", elem.calls.join(", ")));
        }

        entry
    }

    fn generate_call_graph_report(&self, cf_analyzer: &ControlFlowAnalyzer) -> Result<()> {
        let dir = self.prepare_report_dir("call_graph")?;
        let path = dir.join("index.md");
        let mut content = String::from("# Call Graph Analysis\n\n");
        content.push_str("This document shows the **interprocedural call graph** - which functions call which other functions.\n\n");
        content.push_str("> **Note:** This is NOT a control flow graph (CFG). CFG shows intraprocedural control flow (branches, loops) within individual functions.\n\n");

        let stats = cf_analyzer.get_statistics();

        content.push_str("## Call Graph Statistics\n\n");
        content.push_str(&format!("- Total functions: {}\n", stats.total_functions));
        content.push_str(&format!("- Total function calls: {}\n", stats.total_calls));
        content.push_str(&format!("- Maximum call depth: {}\n", stats.max_depth));
        content.push_str(&format!(
            "- Leaf functions (no outgoing calls): {}\n\n",
            stats.leaf_functions
        ));

        content.push_str("## Call Graph Visualization\n\n");
        content.push_str(&cf_analyzer.generate_mermaid());

        fs::write(path, content)?;
        Ok(())
    }

    fn generate_cfg_report(&self, cf_analyzer: &ControlFlowAnalyzer) -> Result<()> {
        let dir = self.prepare_report_dir("cfg")?;
        let mut index = String::from("# Control Flow Graphs (CFG)\n\n");
        index.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));

        if cf_analyzer.cfgs().is_empty() {
            index.push_str("No control flow graphs were captured.\n");
            fs::write(dir.join("index.md"), index)?;
            return Ok(());
        }

        let mut grouped: BTreeMap<String, Vec<(String, &FunctionCfg)>> = BTreeMap::new();
        for cfg in cf_analyzer.cfgs() {
            let compressed = compress_path(&cfg.file_path);
            let key = prefix_key_from_path(&compressed);
            grouped
                .entry(key)
                .or_insert_with(Vec::new)
                .push((compressed, cfg));
        }

        let mut grouped: Vec<_> = grouped.into_iter().collect();
        grouped.sort_by(|a, b| group_key_cmp(&a.0, &b.0));

        index.push_str(&format!("- Total CFGs: {}\n", cf_analyzer.cfgs().len()));
        index.push_str(
            "- Files are grouped by MMSB directory prefix; numeric prefixes match lexical ordering.\n\n",
        );

        index.push_str("## Group Files\n\n");
        for (idx, (group_key, _)) in grouped.iter().enumerate() {
            let file_name = format!("{:03}-{}.md", idx * 10, slugify_key(group_key));
            index.push_str(&format!("- `{}` → `{}`\n", group_key, file_name));
        }

        for (idx, (group_key, mut entries)) in grouped.into_iter().enumerate() {
            entries.sort_by(|a, b| a.1.function.cmp(&b.1.function));
            let slug = slugify_key(&group_key);
            let file_name = format!("{:03}-{}.md", idx * 10, slug);
            let mut content = format!("# CFG Group: {}\n\n", group_key);

            for (compressed, cfg) in entries {
                content.push_str(&format!("## Function: `{}`\n\n", cfg.function));
                content.push_str(&format!(
                    "- File: {}\n- Branches: {}\n- Loops: {}\n- Nodes: {}\n- Edges: {}\n\n",
                    compressed,
                    cfg.branch_count,
                    cfg.loop_count,
                    cfg.nodes.len(),
                    cfg.edges.len(),
                ));
                if let Some(dot_rel) = self.dot_path_for(&compressed) {
                    content.push_str(&format!("- DOT call graph: `{}`\n\n", dot_rel));
                }

                content.push_str("```mermaid\nflowchart TD\n");
                let mut id_map = HashMap::new();
                let prefix = sanitize_mermaid_id(&cfg.function);
                for node in &cfg.nodes {
                    let raw_id = format!("{}_{}", prefix, node.id);
                    let safe_id = sanitize_mermaid_id(&raw_id);
                    id_map.insert(node.id, safe_id.clone());
                    content.push_str(&format!(
                        "    {}[\"{}\"]\n",
                        safe_id,
                        sanitize_mermaid_label(&node.label)
                    ));
                }
                for (from, to) in &cfg.edges {
                    if let (Some(src), Some(dst)) = (id_map.get(from), id_map.get(to)) {
                        content.push_str(&format!("    {} --> {}\n", src, dst));
                    }
                }
                content.push_str("```\n\n");
            }

            fs::write(dir.join(file_name), content)?;
        }

        fs::write(dir.join("index.md"), index)?;
        Ok(())
    }

    fn generate_layer_dependency_report(
        &self,
        rust_layers: &LayerGraph,
        julia_layers: &LayerGraph,
    ) -> Result<()> {
        let dir = self.prepare_report_dir("layer_dependencies")?;
        let path = dir.join("index.md");
        let mut content = String::from("# Layer Dependency Report\n\n");
        content.push_str(&format!(
            "Generated: {}\n\n",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
        ));

        self.write_layer_section(&mut content, "Rust", rust_layers);
        self.write_layer_section(&mut content, "Julia", julia_layers);

        fs::write(path, content)?;
        Ok(())
    }

    fn write_layer_section(&self, content: &mut String, label: &str, graph: &LayerGraph) {
        content.push_str(&format!("## {} Layer Graph\n\n", label));

        if graph.ordered_layers.is_empty() {
            content.push_str("No layers discovered.\n\n");
            return;
        }

        content.push_str("### Layer Order\n");
        for (idx, layer) in graph.ordered_layers.iter().enumerate() {
            let cycle_tag = if graph.cycles.contains(layer) {
                " (cycle)"
            } else {
                ""
            };
            content.push_str(&format!("{}. `{}`{}\n", idx + 1, layer, cycle_tag));
        }
        content.push('\n');

        if !graph.cycles.is_empty() {
            content.push_str("### Cycles Detected\n");
            for cycle in &graph.cycles {
                content.push_str(&format!("- `{}`\n", cycle));
            }
            content.push('\n');
        }

        let violations: Vec<_> = graph.edges.iter().filter(|e| e.violation).collect();
        content.push_str("### Layer Violations\n");
        if violations.is_empty() {
            content.push_str("- None detected.\n\n");
        } else {
            for edge in violations {
                content.push_str(&format!(
                    "- `{}` depends on `{}` ({} references)\n",
                    edge.to,
                    edge.from,
                    edge.references.len()
                ));
                for reference in &edge.references {
                    let compressed = compress_path(reference.file.to_string_lossy().as_ref());
                    content.push_str(&format!("  - {} :: {}\n", compressed, reference.reference));
                }
            }
            content.push('\n');
        }

        content.push_str("### Dependency Edges\n");
        if graph.edges.is_empty() {
            content.push_str("- No cross-layer dependencies recorded.\n\n");
        } else {
            for edge in &graph.edges {
                content.push_str(&format!(
                    "- `{}` → `{}` ({} references{})\n",
                    edge.from,
                    edge.to,
                    edge.references.len(),
                    if edge.violation { ", VIOLATION" } else { "" }
                ));
                for reference in &edge.references {
                    let compressed = compress_path(reference.file.to_string_lossy().as_ref());
                    content.push_str(&format!("  - {} :: {}\n", compressed, reference.reference));
                }
            }
            content.push('\n');
        }

        content.push_str("### Unresolved References\n");
        if graph.unresolved.is_empty() {
            content.push_str("- None.\n\n");
        } else {
            for unresolved in &graph.unresolved {
                let compressed = compress_path(unresolved.file.to_string_lossy().as_ref());
                content.push_str(&format!("- {} → `{}`\n", compressed, unresolved.reference));
            }
            content.push('\n');
        }
    }

    fn generate_module_dependencies(&self, result: &AnalysisResult) -> Result<()> {
        let dir = self.prepare_report_dir("module_dependencies")?;
        let index_path = dir.join("index.md");
        let mut index = String::from("# Module Dependencies\n\n");

        if result.modules.is_empty() {
            index.push_str("No module metadata captured yet.\n");
            fs::write(index_path, index)?;
            return Ok(());
        }

        let mut modules_by_file: BTreeMap<String, ModuleAggregate> = BTreeMap::new();
        for module in &result.modules {
            let layer = self.extract_layer_from_path(&module.file_path);
            let entry = modules_by_file
                .entry(module.file_path.clone())
                .or_insert_with(|| ModuleAggregate::new(module.name.clone(), layer.clone()));

            if entry.name == "unknown" && !module.name.is_empty() {
                entry.name = module.name.clone();
            }

            entry.layer = layer;
            for import in &module.imports {
                entry.imports.insert(normalize_use_stmt(import));
            }
            for export in &module.exports {
                entry.exports.insert(normalize_use_stmt(export));
            }
            for sub in &module.submodules {
                entry.submodules.insert(sub.clone());
            }
        }

        let total_imports: usize = modules_by_file.values().map(|m| m.imports.len()).sum();
        let total_exports: usize = modules_by_file.values().map(|m| m.exports.len()).sum();
        let total_submodules: usize = modules_by_file.values().map(|m| m.submodules.len()).sum();

        let mut modules: Vec<_> = modules_by_file.into_iter().collect();
        modules.sort_by(|a, b| a.0.cmp(&b.0));

        index.push_str(&format!("- Module files analyzed: {}\n", modules.len()));
        index.push_str(&format!("- Unique imports captured: {}\n", total_imports));
        index.push_str(&format!("- Unique exports captured: {}\n", total_exports));
        index.push_str(&format!(
            "- Submodule declarations captured: {}\n\n",
            total_submodules
        ));
        index.push_str("## Per-file Summary\n\n");
        for (file_path, module) in &modules {
            let compressed = compress_path(file_path);
            index.push_str(&format!(
                "- `{}` → module `{}` (layer {}, {} imports / {} exports / {} submodules)\n",
                compressed,
                module.name,
                module.layer,
                module.imports.len(),
                module.exports.len(),
                module.submodules.len()
            ));
        }
        index.push_str(
            "\n## Detailed Files\n\n- `010-imports.md` – expanded import lists\n- `020-exports.md` – export statements\n- `030-submodules.md` – nested module declarations\n- `040-violations.md` – placeholder for future per-module violations\n",
        );
        fs::write(&index_path, index)?;

        let mut imports_doc = String::from("# Module Imports\n\n");
        let mut has_imports = false;
        for (file_path, module) in &modules {
            if module.imports.is_empty() {
                continue;
            }
            has_imports = true;
            let compressed = compress_path(file_path);
            imports_doc.push_str(&format!("## {} ({})\n\n", compressed, module.layer));
            imports_doc.push_str(&format!("Module `{}`\n\n", module.name));
            for import in &module.imports {
                imports_doc.push_str(&format!("- `{}`\n", import));
            }
            imports_doc.push('\n');
        }
        if !has_imports {
            imports_doc.push_str("No imports captured across modules.\n");
        }
        fs::write(dir.join("010-imports.md"), imports_doc)?;

        let mut exports_doc = String::from("# Module Exports\n\n");
        let mut has_exports = false;
        for (file_path, module) in &modules {
            if module.exports.is_empty() {
                continue;
            }
            has_exports = true;
            let compressed = compress_path(file_path);
            exports_doc.push_str(&format!("## {} ({})\n\n", compressed, module.layer));
            exports_doc.push_str(&format!("Module `{}`\n\n", module.name));
            for export in &module.exports {
                exports_doc.push_str(&format!("- `{}`\n", export));
            }
            exports_doc.push('\n');
        }
        if !has_exports {
            exports_doc.push_str("No exports captured across modules.\n");
        }
        fs::write(dir.join("020-exports.md"), exports_doc)?;

        let mut subs_doc = String::from("# Submodules\n\n");
        let mut has_submodules = false;
        for (file_path, module) in &modules {
            if module.submodules.is_empty() {
                continue;
            }
            has_submodules = true;
            let compressed = compress_path(file_path);
            subs_doc.push_str(&format!("## {} ({})\n\n", compressed, module.layer));
            subs_doc.push_str(&format!("Module `{}`\n\n", module.name));
            for sub in &module.submodules {
                subs_doc.push_str(&format!("- `{}`\n", sub));
            }
            subs_doc.push('\n');
        }
        if !has_submodules {
            subs_doc.push_str("No nested modules recorded.\n");
        }
        fs::write(dir.join("030-submodules.md"), subs_doc)?;

        let mut violations_doc = String::from("# Module Violations\n\n");
        violations_doc.push_str(
            "Per-module import/export violations are not computed yet.\n\
Refer to `layer_dependencies/index.md` for cross-layer problems.\n",
        );
        fs::write(dir.join("040-violations.md"), violations_doc)?;

        Ok(())
    }

    fn generate_function_analysis(&self, result: &AnalysisResult) -> Result<()> {
        let dir = self.prepare_report_dir("function_analysis")?;
        let mut index = String::from("# Function Analysis\n\n");

        let functions: Vec<_> = result
            .elements
            .iter()
            .filter(|e| matches!(e.element_type, ElementType::Function))
            .collect();

        index.push_str(&format!("## Total Functions: {}\n\n", functions.len()));
        index.push_str(
            "Functions are bucketed alphabetically so `ls function_analysis/` advertises the range.\n\n",
        );

        if functions.is_empty() {
            fs::write(dir.join("index.md"), index)?;
            return Ok(());
        }

        let bucket_labels = ["A-F", "G-M", "N-S", "T-Z", "Other"];
        let mut buckets: HashMap<&'static str, Vec<&CodeElement>> = HashMap::new();
        for label in bucket_labels {
            buckets.insert(label, Vec::new());
        }

        for func in &functions {
            let label = function_bucket_label(&func.name);
            buckets.entry(label).or_insert_with(Vec::new).push(func);
        }

        index.push_str("## Bucket Files\n\n");
        for (idx, label) in bucket_labels.iter().enumerate() {
            let file_name = format!("{:03}-functions_{}.md", (idx + 1) * 10, label);
            let count = buckets.get(label).map(|v| v.len()).unwrap_or(0);
            index.push_str(&format!(
                "- `{}` → `{}` ({} functions)\n",
                label, file_name, count
            ));
        }
        fs::write(dir.join("index.md"), index)?;

        for (idx, label) in bucket_labels.iter().enumerate() {
            let mut funcs = buckets.remove(label).unwrap_or_default();
            funcs.sort_by_key(|f| (&f.layer, &f.name));
            let file_name = format!("{:03}-functions_{}.md", (idx + 1) * 10, label);
            let mut content = format!("# Functions {}\n\n", label);

            if funcs.is_empty() {
                content.push_str("No functions fell into this range.\n");
                fs::write(dir.join(file_name), content)?;
                continue;
            }

            let mut layer_map: BTreeMap<String, Vec<&CodeElement>> = BTreeMap::new();
            for func in funcs {
                layer_map
                    .entry(func.layer.clone())
                    .or_insert_with(Vec::new)
                    .push(func);
            }

            for (layer, entries) in layer_map {
                content.push_str(&format!("## Layer: {}\n\n", layer));

                let mut rust_funcs: Vec<_> = entries
                    .iter()
                    .filter(|f| matches!(f.language, Language::Rust))
                    .collect();
                let mut julia_funcs: Vec<_> = entries
                    .iter()
                    .filter(|f| matches!(f.language, Language::Julia))
                    .collect();

                rust_funcs.sort_by_key(|f| &f.name);
                julia_funcs.sort_by_key(|f| &f.name);

                if !rust_funcs.is_empty() {
                    content.push_str("### Rust Functions\n\n");
                    for func in rust_funcs {
                        content.push_str(&format!("#### `{}`\n\n", func.name));
                        let compressed = compress_path(&func.file_path);
                        content.push_str(&format!(
                            "- **File:** {}:{}\n",
                            compressed, func.line_number
                        ));
                        content.push_str(&format!("- **Visibility:** {:?}\n", func.visibility));

                        if !func.generic_params.is_empty() {
                            content.push_str(&format!(
                                "- **Generics:** {}\n",
                                func.generic_params.join(", ")
                            ));
                        }

                        if !func.calls.is_empty() {
                            content.push_str("- **Calls:**\n");
                            for call in &func.calls {
                                content.push_str(&format!("  - `{}`\n", call));
                            }
                        }
                        content.push_str("\n");
                    }
                }

                if !julia_funcs.is_empty() {
                    content.push_str("### Julia Functions\n\n");
                    for func in julia_funcs {
                        content.push_str(&format!("#### `{}`\n\n", func.name));
                        let compressed = compress_path(&func.file_path);
                        content.push_str(&format!(
                            "- **File:** {}:{}\n",
                            compressed, func.line_number
                        ));
                        content.push_str(&format!("- **Signature:** `{}`\n", func.signature));

                        if !func.calls.is_empty() {
                            content.push_str("- **Calls:**\n");
                            for call in &func.calls {
                                content.push_str(&format!("  - `{}`\n", call));
                            }
                        }
                        content.push_str("\n");
                    }
                }
            }

            fs::write(dir.join(file_name), content)?;
        }

        Ok(())
    }

    fn extract_layer_from_path(&self, path: &str) -> String {
        for component in path.split('/') {
            if component
                .chars()
                .next()
                .map_or(false, |c| c.is_ascii_digit())
            {
                if let Some(pos) = component.find('_') {
                    if component[..pos].chars().all(|c| c.is_ascii_digit()) {
                        return component.to_string();
                    }
                }
            }
        }
        "root".to_string()
    }

    fn dot_path_for(&self, compressed_path: &str) -> Option<String> {
        let slug = slugify_file_path(compressed_path);
        let rel = format!("cfg/dots/{}/call_graph.dot", slug);
        let absolute = Path::new(&self.output_dir).join(&rel);
        if absolute.exists() {
            Some(rel)
        } else {
            None
        }
    }
}

fn prefix_key_from_path(path: &str) -> String {
    let relative = path.strip_prefix("MMSB/").unwrap_or(path);
    if relative.is_empty() {
        return "root".to_string();
    }
    let parts: Vec<&str> = relative.split('/').collect();
    if parts.len() == 1 {
        return "root".to_string();
    }
    if parts[0] == "src" && parts.len() >= 2 {
        return format!("{}/{}", parts[0], parts[1]);
    }
    parts[0].to_string()
}

fn slugify_key(input: &str) -> String {
    input
        .chars()
        .map(|c| match c {
            '/' => '-',
            ' ' => '_',
            _ if c.is_ascii_alphanumeric() || c == '-' => c.to_ascii_lowercase(),
            _ => '_',
        })
        .collect()
}

fn group_key_cmp(a: &str, b: &str) -> Ordering {
    match (a == "root", b == "root") {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        _ => a.cmp(b),
    }
}

fn function_bucket_label(name: &str) -> &'static str {
    let first = name
        .chars()
        .find(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .unwrap_or('#');

    match first {
        'A'..='F' => "A-F",
        'G'..='M' => "G-M",
        'N'..='S' => "N-S",
        'T'..='Z' => "T-Z",
        _ => "Other",
    }
}

fn slugify_file_path(path: &str) -> String {
    path.trim_start_matches("MMSB/")
        .replace('/', "-")
        .replace('.', "_")
        .to_lowercase()
}

fn language_label(language: &Language) -> &'static str {
    match language {
        Language::Rust => "Rust",
        Language::Julia => "Julia",
    }
}

fn visibility_label(vis: &Visibility) -> &'static str {
    match vis {
        Visibility::Public => "pub",
        Visibility::Crate => "pub(crate)",
        Visibility::Private => "priv",
    }
}

fn short_signature(input: &str) -> String {
    let collapsed = input.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() > 120 {
        let mut truncated = collapsed.chars().take(117).collect::<String>();
        truncated.push_str("...");
        truncated
    } else {
        collapsed
    }
}

struct ModuleAggregate {
    name: String,
    layer: String,
    imports: BTreeSet<String>,
    exports: BTreeSet<String>,
    submodules: BTreeSet<String>,
}

impl ModuleAggregate {
    fn new(name: String, layer: String) -> Self {
        Self {
            name: if name.is_empty() {
                "unknown".to_string()
            } else {
                name
            },
            layer,
            imports: BTreeSet::new(),
            exports: BTreeSet::new(),
            submodules: BTreeSet::new(),
        }
    }
}

fn normalize_use_stmt(stmt: &str) -> String {
    let collapsed = stmt.replace('\n', " ");
    let mut cleaned = collapsed.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(idx) = cleaned.find(';') {
        cleaned.truncate(idx);
    }
    cleaned = cleaned.trim().to_string();
    if cleaned.starts_with("pub") {
        if let Some(pos) = cleaned.find(' ') {
            cleaned = cleaned[pos + 1..].trim().to_string();
        }
    }
    if let Some(stripped) = cleaned.strip_prefix("use ") {
        cleaned = stripped.trim().to_string();
    }
    cleaned
}

fn sanitize_mermaid_id(input: &str) -> String {
    input
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

fn sanitize_mermaid_label(label: &str) -> String {
    label.replace('"', "'").replace('`', "'")
}
