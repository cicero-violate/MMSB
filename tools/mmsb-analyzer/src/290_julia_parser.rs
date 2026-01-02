//! Julia file analyzer via Julia script when available, falling back to a built-in parser.

use crate::types::*;
use anyhow::{anyhow, Context, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};

static INLINE_FN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^\s*([A-Za-z_][A-Za-z0-9_!.]*)\s*(\([^=]*\))\s*=\s*(.+)$").unwrap());
static CALL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"([A-Za-z_][A-Za-z0-9_!.]*)\s*\(").unwrap());

pub struct JuliaAnalyzer {
    script_path: PathBuf,
    root_path: PathBuf,
    project_dir: PathBuf,
    dot_root: PathBuf,
    julia_bin: PathBuf,
    script_disabled: AtomicBool,
    global_cfg_generated: AtomicBool,
}

impl JuliaAnalyzer {
    pub fn new(root_path: PathBuf, script_path: PathBuf, dot_root: PathBuf) -> Self {
        let _base_dir = script_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let julia_bin = resolve_julia_binary();
        let project_dir = find_julia_project_dir(&script_path);
        Self {
            root_path,
            script_path,
            project_dir,
            dot_root,
            julia_bin,
            script_disabled: AtomicBool::new(false),
            global_cfg_generated: AtomicBool::new(false),
        }
    }

    #[allow(dead_code)]
    pub fn generate_global_cfgs(&self) -> Result<()> {
        self.ensure_global_cfgs()
    }

    pub fn analyze_file(&self, file_path: &Path) -> Result<AnalysisResult> {
        println!("    [Julia] Analyzing {:?}", file_path);

        if !self.script_disabled.load(Ordering::Relaxed) {
            self.ensure_global_cfgs()?;

            match self.run_script(file_path) {
                Ok(result) => return Ok(result),
                Err(err) => {
                    eprintln!(
                        "Warning: Julia script execution failed ({err}). Falling back to internal parser for the remaining files."
                    );
                    self.script_disabled.store(true, Ordering::Relaxed);
                }
            }
        }

        self.fallback_parse(file_path)
    }

    fn ensure_global_cfgs(&self) -> Result<()> {
        if self.global_cfg_generated.load(Ordering::Relaxed) {
            return Ok(());
        }

        let status = Command::new(&self.julia_bin)
            .arg("--startup-file=no")
            .arg(&self.script_path)
            .arg("--global-cfgs")
            .arg(&self.root_path)
            .arg(&self.dot_root)
            .env("JULIA_PROJECT", &self.project_dir)
            .stderr(Stdio::inherit())
            .stdout(Stdio::inherit())
            .status()
            .context("Failed to execute Julia analyzer for global CFG generation")?;

        if !status.success() {
            return Err(anyhow!(
                "Julia analyzer exited with {} while generating global CFGs",
                status
            ));
        }

        self.global_cfg_generated.store(true, Ordering::Relaxed);
        Ok(())
    }

    fn run_script(&self, file_path: &Path) -> Result<AnalysisResult> {
        let dot_dir = self.compute_dot_dir(file_path);
        fs::create_dir_all(&dot_dir)?;
        let output = Command::new(&self.julia_bin)
            .arg("--startup-file=no")
            .arg(&self.script_path)
            .arg(file_path)
            .arg(&dot_dir)
            .arg(&self.root_path)
            .env("JULIA_PROJECT", &self.project_dir)
            .stderr(Stdio::inherit())
            .output()
            .with_context(|| format!("Failed to execute Julia analyzer on {:?}", file_path))?;

        if !output.status.success() {
            return Err(anyhow!(
                "Julia analyzer exited with {}: {}",
                output.status,
                "check stderr output above"
            ));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let json_line = stdout
            .lines()
            .find(|line| line.trim_start().starts_with('[') || line.trim_start().starts_with('{'))
            .unwrap_or("");

        let julia_elements: Vec<JuliaElement> = serde_json::from_str(&stdout)
            .or_else(|_| serde_json::from_str(json_line))
            .with_context(|| {
                format!(
                    "Failed to parse Julia analyzer output. Raw output: {}",
                    stdout
                )
            })?;

        let mut result = AnalysisResult::new();
        let layer = self.extract_layer(file_path);

        for elem in julia_elements {
            let element_type = match elem.element_type.as_str() {
                "struct" => ElementType::Struct,
                "function" => ElementType::Function,
                "module" => ElementType::Module,
                _ => continue,
            };

            let file_path = relativize_path(Path::new(&elem.file_path), &self.root_path);

            result.add_element(CodeElement {
                element_type,
                name: elem.name,
                file_path,
                line_number: elem.line_number,
                language: Language::Julia,
                layer: layer.clone(),
                signature: elem.signature,
                calls: elem.calls,
                visibility: Visibility::Public,
                generic_params: Vec::new(),
            });
        }

        Ok(result)
    }

    fn fallback_parse(&self, file_path: &Path) -> Result<AnalysisResult> {
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read Julia file {:?}", file_path))?;
        let mut result = AnalysisResult::new();
        let layer = self.extract_layer(file_path);
        let file_path_str = relativize_path(file_path, &self.root_path);
        let lines: Vec<&str> = content.lines().collect();
        let mut idx = 0;

        while idx < lines.len() {
            let line = lines[idx];
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                idx += 1;
                continue;
            }

            if let Some(module_name) = parse_module_name(trimmed) {
                result.add_element(CodeElement {
                    element_type: ElementType::Module,
                    name: module_name,
                    file_path: file_path_str.clone(),
                    line_number: idx + 1,
                    language: Language::Julia,
                    layer: layer.clone(),
                    signature: trimmed.to_string(),
                    calls: Vec::new(),
                    visibility: Visibility::Public,
                    generic_params: Vec::new(),
                });
                idx += 1;
                continue;
            }

            if let Some(struct_name) = parse_struct_name(trimmed) {
                result.add_element(CodeElement {
                    element_type: ElementType::Struct,
                    name: struct_name,
                    file_path: file_path_str.clone(),
                    line_number: idx + 1,
                    language: Language::Julia,
                    layer: layer.clone(),
                    signature: trimmed.to_string(),
                    calls: Vec::new(),
                    visibility: Visibility::Public,
                    generic_params: Vec::new(),
                });
                idx += 1;
                continue;
            }

            if trimmed.starts_with("function ") {
                let (consumed, maybe_element) =
                    self.parse_function_block(&file_path_str, &layer, &lines, idx);
                if let Some(element) = maybe_element {
                    result.add_element(element);
                }
                idx += consumed.max(1);
                continue;
            }

            if let Some(element) =
                self.parse_inline_function(&file_path_str, &layer, lines[idx], idx + 1)
            {
                result.add_element(element);
            }

            idx += 1;
        }

        Ok(result)
    }

    fn parse_function_block(
        &self,
        file_path: &str,
        layer: &str,
        lines: &[&str],
        start_idx: usize,
    ) -> (usize, Option<CodeElement>) {
        let mut header = lines[start_idx].trim().to_string();
        let mut consumed = 1usize;
        while paren_balance(&header) > 0 && start_idx + consumed < lines.len() {
            header.push(' ');
            header.push_str(lines[start_idx + consumed].trim());
            consumed += 1;
        }

        let mut inline_body = None;
        if let Some(eq_idx) = header.find('=') {
            inline_body = Some(header[eq_idx + 1..].trim().to_string());
            header = header[..eq_idx].trim().to_string();
        }

        let mut body_lines = Vec::new();
        let mut total_consumed = consumed;
        if inline_body.is_none() {
            let mut depth = 1i32;
            let mut idx = start_idx + consumed;
            while idx < lines.len() {
                let current = lines[idx];
                let trimmed = current.trim();
                if trimmed.starts_with("function ")
                    || trimmed.starts_with("let")
                    || trimmed.starts_with("if ")
                    || trimmed == "if"
                    || trimmed.starts_with("for ")
                    || trimmed.starts_with("while ")
                    || trimmed.starts_with("begin")
                    || trimmed.starts_with("try")
                    || trimmed.starts_with("module ")
                    || trimmed.starts_with("baremodule ")
                    || trimmed.starts_with("struct ")
                    || trimmed.starts_with("mutable struct ")
                    || trimmed.starts_with("quote")
                {
                    depth += 1;
                }

                if trimmed == "end" || trimmed.starts_with("end ") {
                    depth -= 1;
                    idx += 1;
                    if depth == 0 {
                        break;
                    } else {
                        continue;
                    }
                }

                body_lines.push(current.to_string());
                idx += 1;
            }
            total_consumed = idx - start_idx;
        }

        let signature = header
            .trim()
            .trim_start_matches("function")
            .trim()
            .to_string();
        if signature.is_empty() {
            return (total_consumed, None);
        }

        let func_name = signature
            .split(|c: char| c == '(' || c.is_whitespace())
            .next()
            .unwrap_or("anonymous")
            .to_string();

        let calls = if let Some(inline) = inline_body {
            extract_calls_from_text(&inline)
        } else {
            extract_calls_from_lines(&body_lines)
        };

        let element = CodeElement {
            element_type: ElementType::Function,
            name: func_name,
            file_path: file_path.to_string(),
            line_number: start_idx + 1,
            language: Language::Julia,
            layer: layer.to_string(),
            signature,
            calls,
            visibility: Visibility::Public,
            generic_params: Vec::new(),
        };

        (total_consumed, Some(element))
    }

    fn parse_inline_function(
        &self,
        file_path: &str,
        layer: &str,
        line: &str,
        line_number: usize,
    ) -> Option<CodeElement> {
        let trimmed = line.trim();
        if trimmed.starts_with("function ") || trimmed.starts_with('#') {
            return None;
        }

        let captures = INLINE_FN_RE.captures(trimmed)?;
        let func_name = captures.get(1)?.as_str().to_string();
        let args = captures.get(2).map(|m| m.as_str()).unwrap_or("()");
        let body = captures
            .get(3)
            .map(|m| m.as_str())
            .unwrap_or("")
            .to_string();
        let signature = format!("{}{}", func_name, args);
        let calls = extract_calls_from_text(&body);

        Some(CodeElement {
            element_type: ElementType::Function,
            name: func_name,
            file_path: file_path.to_string(),
            line_number,
            language: Language::Julia,
            layer: layer.to_string(),
            signature,
            calls,
            visibility: Visibility::Public,
            generic_params: Vec::new(),
        })
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

    fn compute_dot_dir(&self, path: &Path) -> PathBuf {
        if let Ok(relative) = path.strip_prefix(&self.root_path) {
            if let Some(parent) = relative.parent() {
                return self.dot_root.join(parent);
            }
            return self.dot_root.clone();
        }

        let slug = slugify_relative(&self.root_path, path);
        self.dot_root.join(slug)
    }
}

fn slugify_relative(root: &Path, path: &Path) -> String {
    let relative = path.strip_prefix(root).unwrap_or(path);
    relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().replace('.', "_"))
        .collect::<Vec<_>>()
        .join("-")
}

fn resolve_julia_binary() -> PathBuf {
    if let Ok(custom) = env::var("JULIA_BINARY") {
        let candidate = PathBuf::from(custom);
        if candidate.exists() {
            return candidate;
        }
    }
    if let Ok(home) = env::var("HOME") {
        let juliaup_root = Path::new(&home).join(".julia/juliaup");
        if let Ok(entries) = fs::read_dir(&juliaup_root) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_dir() {
                    continue;
                }
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("julia-") {
                        let candidate = path.join("bin/julia");
                        if candidate.exists() {
                            return candidate;
                        }
                    }
                }
            }
        }
    }
    let alt = PathBuf::from("/home/cicero-arch-omen/git/julia/usr/bin/julia");
    if alt.exists() {
        return alt;
    }
    PathBuf::from("julia")
}

fn find_julia_project_dir(script_path: &Path) -> PathBuf {
    let mut current = script_path.parent();
    while let Some(dir) = current {
        if dir.join("Project.toml").exists() {
            return dir.to_path_buf();
        }
        current = dir.parent();
    }
    script_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

fn parse_module_name(line: &str) -> Option<String> {
    if line.starts_with("module ") {
        return line
            .split_whitespace()
            .nth(1)
            .map(|name| name.trim_end_matches(';').to_string());
    }
    if line.starts_with("baremodule ") {
        return line
            .split_whitespace()
            .nth(1)
            .map(|name| name.trim_end_matches(';').to_string());
    }
    None
}

fn parse_struct_name(line: &str) -> Option<String> {
    if line.starts_with("mutable struct ") || line.starts_with("struct ") {
        let offset = if line.starts_with("mutable struct ") {
            2
        } else {
            1
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        return tokens.get(offset).map(|name| {
            name.split("<:")
                .next()
                .unwrap_or(name)
                .trim_end_matches('{')
                .to_string()
        });
    }
    None
}

fn relativize_path(path: &Path, root: &Path) -> String {
    if let Ok(stripped) = path.strip_prefix(root) {
        stripped.to_string_lossy().to_string()
    } else {
        path.to_string_lossy().to_string()
    }
}

fn extract_calls_from_lines(lines: &[String]) -> Vec<String> {
    let joined = lines.join("\n");
    extract_calls_from_text(&joined)
}

fn extract_calls_from_text(text: &str) -> Vec<String> {
    let mut calls = Vec::new();
    for capture in CALL_RE.captures_iter(text) {
        if let Some(name) = capture.get(1) {
            let identifier = name.as_str();
            if is_reserved(identifier) {
                continue;
            }
            calls.push(identifier.to_string());
        }
    }
    calls.sort();
    calls.dedup();
    calls
}

fn is_reserved(name: &str) -> bool {
    matches!(
        name,
        "if" | "for"
            | "while"
            | "begin"
            | "let"
            | "struct"
            | "mutable"
            | "quote"
            | "macro"
            | "module"
            | "end"
            | "baremodule"
            | "function"
    )
}

fn paren_balance(input: &str) -> i32 {
    let mut balance = 0i32;
    for ch in input.chars() {
        if ch == '(' {
            balance += 1;
        } else if ch == ')' {
            balance -= 1;
        }
    }
    balance
}
