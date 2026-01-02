#![allow(dead_code)]
//! Dead code policy file support.

use crate::dead_code_intent::DeadCodePolicy;
use std::path::Path;

pub fn load_policy(path: &Path) -> std::io::Result<DeadCodePolicy> {
    let contents = std::fs::read_to_string(path)?;
    Ok(parse_policy(&contents, path.parent().unwrap_or(path)))
}

pub fn parse_policy(contents: &str, base: &Path) -> DeadCodePolicy {
    let mut planned_directories = Vec::new();
    let mut public_api_roots = Vec::new();
    let mut entrypoint_symbols = Vec::new();
    let mut treat_public_as_entrypoint = true;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        match key {
            "planned_directories" => {
                planned_directories = parse_list(value)
                    .into_iter()
                    .map(|p| base.join(p))
                    .collect();
            }
            "public_api_roots" => {
                public_api_roots = parse_list(value)
                    .into_iter()
                    .map(|p| base.join(p))
                    .collect();
            }
            "entrypoint_symbols" => {
                entrypoint_symbols = parse_list(value);
            }
            "treat_public_as_entrypoint" => {
                treat_public_as_entrypoint = parse_bool(value).unwrap_or(true);
            }
            _ => {}
        }
    }

    DeadCodePolicy {
        planned_directories,
        public_api_roots,
        entrypoint_symbols,
        treat_public_as_entrypoint,
    }
}

fn parse_list(value: &str) -> Vec<String> {
    let mut trimmed = value.trim().to_string();
    if let Some(stripped) = trimmed.strip_prefix('[') {
        trimmed = stripped.to_string();
    }
    if let Some(stripped) = trimmed.strip_suffix(']') {
        trimmed = stripped.to_string();
    }
    trimmed
        .split(',')
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn parse_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}
