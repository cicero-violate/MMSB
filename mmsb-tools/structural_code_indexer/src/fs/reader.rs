//! Filesystem Reader (Read-Only)
//!
//! Recursively reads source files without modification.

use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Source file content
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub content: String,
}

/// Read all source files from a directory
///
/// **READ-ONLY:** Never modifies files
pub fn read_source_files(root: &Path) -> Vec<SourceFile> {
    let mut files = Vec::new();
    
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_entry(|e| !is_ignored(e.path()))
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        
        if path.is_file() && is_rust_file(path) {
            if let Ok(content) = std::fs::read_to_string(path) {
                files.push(SourceFile {
                    path: path.to_path_buf(),
                    content,
                });
            }
        }
    }
    
    files
}

/// Check if path should be ignored
fn is_ignored(path: &Path) -> bool {
    path.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some(".git") | Some("target") | Some("node_modules") | Some(".cache")
        )
    })
}

/// Check if file is a Rust source file
fn is_rust_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e == "rs")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_is_rust_file() {
        assert!(is_rust_file(Path::new("foo.rs")));
        assert!(!is_rust_file(Path::new("foo.txt")));
    }
    
    #[test]
    fn test_ignores_git() {
        assert!(is_ignored(Path::new(".git/config")));
        assert!(is_ignored(Path::new("target/debug")));
        assert!(!is_ignored(Path::new("src/main.rs")));
    }
}
