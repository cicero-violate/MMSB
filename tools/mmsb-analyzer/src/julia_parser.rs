//! Julia file analyzer via FFI to Julia script

use crate::types::*;
use anyhow::{Context, Result};
use std::path::Path;
use std::process::{Command, Stdio};

pub struct JuliaAnalyzer {
    script_path: String,
    root_path: String,
}

impl JuliaAnalyzer {
    pub fn new(root_path: String, script_path: String) -> Self {
        Self {
            root_path,
            script_path,
        }
    }
    
    pub fn analyze_file(&self, file_path: &Path) -> Result<AnalysisResult> {
        let output = Command::new("julia")
            .arg("--startup-file=no")
            .arg(&self.script_path)
            .arg(file_path)
            .stderr(Stdio::null())
            .output()
            .with_context(|| format!("Failed to execute Julia analyzer on {:?}", file_path))?;
        
        if !output.status.success() {
            anyhow::bail!("Julia analyzer failed with status: {}", output.status);
        }
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        
        // Clean output - remove any non-JSON lines
        let json_line = stdout.lines()
            .find(|line| line.trim_start().starts_with('[') || line.trim_start().starts_with('{'))
            .unwrap_or("");
        
        let julia_elements: Vec<JuliaElement> = serde_json::from_str(&stdout)
            .or_else(|_| serde_json::from_str(json_line))
            .with_context(|| format!("Failed to parse Julia analyzer output. Raw output: {}", stdout))?;
        
        let mut result = AnalysisResult::new();
        let layer = self.extract_layer(file_path);
        
        for elem in julia_elements {
            let element_type = match elem.element_type.as_str() {
                "struct" => ElementType::Struct,
                "function" => ElementType::Function,
                "module" => ElementType::Module,
                _ => continue,
            };
            
            result.add_element(CodeElement {
                element_type,
                name: elem.name,
                file_path: elem.file_path,
                line_number: elem.line_number,
                language: Language::Julia,
                layer: layer.clone(),
                signature: elem.signature,
                calls: elem.calls,
                visibility: Visibility::Public, // Julia doesn't have private by default
                generic_params: Vec::new(),
            });
        }
        
        Ok(result)
    }
    
    fn extract_layer(&self, path: &Path) -> String {
        for component in path.components() {
            if let Some(name) = component.as_os_str().to_str() {
                if name.chars().next().map_or(false, |c| c.is_ascii_digit()) {
                    if let Some(pos) = name.find('_') {
                        if name[..pos].chars().all(|c| c.is_ascii_digit()) {
                            return name.to_string();
                        }
                    }
                }
            }
        }
        "root".to_string()
    }
}
