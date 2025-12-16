//! Markdown report generation

use crate::types::*;
use crate::control_flow::{ControlFlowAnalyzer, CallGraphStats};
use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Compress absolute paths to MMSB-relative format
fn compress_path(path: &str) -> String {
    if let Some(idx) = path.find("/MMSB/") {
        return format!("MMSB{}", &path[idx + 5..]);
    }
    if path.starts_with("MMSB/") {
        return path.to_string();
    }
    if let Some(idx) = path.rfind("/src/") {
        return format!("MMSB/src{}", &path[idx + 4..]);
    }
    path.to_string()
}

pub struct ReportGenerator {
    output_dir: String,
}

impl ReportGenerator {
    pub fn new(output_dir: String) -> Self {
        Self { output_dir }
    }
    
    pub fn generate_all(&self, result: &AnalysisResult, cf_analyzer: &ControlFlowAnalyzer) -> Result<()> {
        fs::create_dir_all(&self.output_dir)?;
        
        self.generate_structure_report(result)?;
        self.generate_call_graph_report(result, cf_analyzer)?;
        self.generate_module_dependencies(result)?;
        self.generate_function_analysis(result)?;
        
        Ok(())
    }
    
    fn generate_structure_report(&self, result: &AnalysisResult) -> Result<()> {
        let path = Path::new(&self.output_dir).join("structure.md");
        let mut content = String::from("# MMSB Code Structure Analysis\n\n");
        content.push_str(&format!("Generated: {}\n\n", chrono::Local::now().format("%Y-%m-%d %H:%M:%S")));
        
        // Group by layer
        let mut layers: HashMap<String, Vec<&CodeElement>> = HashMap::new();
        for elem in &result.elements {
            layers.entry(elem.layer.clone()).or_insert_with(Vec::new).push(elem);
        }
        
        let mut sorted_layers: Vec<_> = layers.keys().collect();
        sorted_layers.sort();
        
        for layer in sorted_layers {
            content.push_str(&format!("\n## Layer: {}\n\n", layer));
            
            let elements = &layers[layer];
            
            // Group by language
            let rust_elems: Vec<_> = elements.iter().filter(|e| matches!(e.language, Language::Rust)).collect();
            let julia_elems: Vec<_> = elements.iter().filter(|e| matches!(e.language, Language::Julia)).collect();
            
            if !rust_elems.is_empty() {
                content.push_str("### Rust\n\n");
                self.write_elements(&mut content, rust_elems);
            }
            
            if !julia_elems.is_empty() {
                content.push_str("\n### Julia\n\n");
                self.write_elements(&mut content, julia_elems);
            }
        }
        
        // Summary statistics
        content.push_str("\n## Summary Statistics\n\n");
        content.push_str(&format!("- Total elements: {}\n", result.elements.len()));
        content.push_str(&format!("- Rust elements: {}\n", 
            result.elements.iter().filter(|e| matches!(e.language, Language::Rust)).count()));
        content.push_str(&format!("- Julia elements: {}\n",
            result.elements.iter().filter(|e| matches!(e.language, Language::Julia)).count()));
        
        // By type
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for elem in &result.elements {
            let key = format!("{:?}_{:?}", elem.language, elem.element_type);
            *type_counts.entry(key).or_insert(0) += 1;
        }
        
        content.push_str("\n### By Type:\n\n");
        let mut sorted_types: Vec<_> = type_counts.iter().collect();
        sorted_types.sort_by_key(|(k, _)| k.as_str());
        for (type_name, count) in sorted_types {
            content.push_str(&format!("- {}: {}\n", type_name, count));
        }
        
        fs::write(path, content)?;
        Ok(())
    }
    
    fn write_elements(&self, content: &mut String, elements: Vec<&&CodeElement>) {
        let mut sorted = elements;
        sorted.sort_by_key(|e| (&e.file_path, e.line_number));
        
        for elem in sorted {
            let vis = match elem.visibility {
                Visibility::Public => "pub",
                Visibility::Crate => "pub(crate)",
                Visibility::Private => "priv",
            };
            content.push_str(&format!("- [{:?}] `{}` **{}** @ {}:{}\n", 
                elem.element_type, vis, elem.name, elem.file_path, elem.line_number));
        }
    }
    
    fn generate_control_flow_report(&self, result: &AnalysisResult, cf_analyzer: &ControlFlowAnalyzer) -> Result<()> {
        let path = Path::new(&self.output_dir).join("control_flow.md");
        let mut content = String::from("# Control Flow Analysis\n\n");
        
        let stats = cf_analyzer.get_statistics();
        
        content.push_str("## Call Graph Statistics\n\n");
        content.push_str(&format!("- Total functions: {}\n", stats.total_functions));
        content.push_str(&format!("- Total function calls: {}\n", stats.total_calls));
        content.push_str(&format!("- Maximum call depth: {}\n", stats.max_depth));
        content.push_str(&format!("- Leaf functions (no outgoing calls): {}\n\n", stats.leaf_functions));
        
        content.push_str("## Call Graph Visualization\n\n");
        content.push_str(&cf_analyzer.generate_mermaid());
        
        fs::write(path, content)?;
        Ok(())
    }
    
    fn generate_module_dependencies(&self, result: &AnalysisResult) -> Result<()> {
        let path = Path::new(&self.output_dir).join("module_dependencies.md");
        let mut content = String::from("# Module Dependencies\n\n");
        
        // Group modules by layer
        let mut layer_modules: HashMap<String, Vec<&ModuleInfo>> = HashMap::new();
        for module in &result.modules {
            layer_modules.entry(self.extract_layer_from_path(&module.file_path))
                .or_insert_with(Vec::new)
                .push(module);
        }
        
        for (layer, modules) in layer_modules {
            content.push_str(&format!("## Layer: {}\n\n", layer));
            
            for module in modules {
                content.push_str(&format!("### Module: `{}`\n\n", module.name));
                content.push_str(&format!("**File:** {}\n\n", module.file_path));
                
                if !module.imports.is_empty() {
                    content.push_str("**Imports:**\n");
                    for import in &module.imports {
                        content.push_str(&format!("- `{}`\n", import));
                    }
                    content.push_str("\n");
                }
                
                if !module.submodules.is_empty() {
                    content.push_str("**Submodules:**\n");
                    for submod in &module.submodules {
                        content.push_str(&format!("- `{}`\n", submod));
                    }
                    content.push_str("\n");
                }
            }
        }
        
        fs::write(path, content)?;
        Ok(())
    }
    
    fn generate_function_analysis(&self, result: &AnalysisResult) -> Result<()> {
        let path = Path::new(&self.output_dir).join("function_analysis.md");
        let mut content = String::from("# Function Analysis\n\n");
        
        let functions: Vec<_> = result.elements.iter()
            .filter(|e| matches!(e.element_type, ElementType::Function))
            .collect();
        
        content.push_str(&format!("## Total Functions: {}\n\n", functions.len()));
        
        // Group by layer
        let mut layer_functions: HashMap<String, Vec<&CodeElement>> = HashMap::new();
        for func in &functions {
            layer_functions.entry(func.layer.clone())
                .or_insert_with(Vec::new)
                .push(func);
        }
        
        let mut sorted_layers: Vec<_> = layer_functions.keys().collect();
        sorted_layers.sort();
        
        for layer in sorted_layers {
            content.push_str(&format!("## Layer: {}\n\n", layer));
            
            let funcs = &layer_functions[layer];
            let mut rust_funcs: Vec<_> = funcs.iter().filter(|f| matches!(f.language, Language::Rust)).collect();
            let mut julia_funcs: Vec<_> = funcs.iter().filter(|f| matches!(f.language, Language::Julia)).collect();
            
            rust_funcs.sort_by_key(|f| &f.name);
            julia_funcs.sort_by_key(|f| &f.name);
            
            if !rust_funcs.is_empty() {
                content.push_str("### Rust Functions\n\n");
                for func in rust_funcs {
                    content.push_str(&format!("#### `{}`\n\n", func.name));
                    content.push_str(&format!("- **File:** {}:{}\n", func.file_path, func.line_number));
                    content.push_str(&format!("- **Visibility:** {:?}\n", func.visibility));
                    
                    if !func.generic_params.is_empty() {
                        content.push_str(&format!("- **Generics:** {}\n", func.generic_params.join(", ")));
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
                    content.push_str(&format!("- **File:** {}:{}\n", func.file_path, func.line_number));
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
        
        fs::write(path, content)?;
        Ok(())
    }
    
    fn extract_layer_from_path(&self, path: &str) -> String {
        for component in path.split('/') {
            if component.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                if let Some(pos) = component.find('_') {
                    if component[..pos].chars().all(|c| c.is_ascii_digit()) {
                        return component.to_string();
                    }
                }
            }
        }
        "root".to_string()
    }
}
