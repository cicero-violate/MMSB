use crate::types::{IntentClass, IntentMetadata};

/// Classifies intent operations based on content and patterns
pub struct IntentClassifier;

impl IntentClassifier {
    /// Classify an intent based on its content
    pub fn classify(content: &str) -> IntentMetadata {
        let mut metadata = IntentMetadata::new();

        // Simple pattern-based classification
        // This can be extended with more sophisticated analysis
        
        let content_lower = content.to_lowercase();

        // Formatting operations
        if content_lower.contains("rustfmt")
            || content_lower.contains("cargo fmt")
            || content_lower.contains("formatting")
        {
            metadata = metadata.with_class(IntentClass::formatting());
        }

        // Lint fixes
        if content_lower.contains("clippy")
            || content_lower.contains("cargo clippy")
            || content_lower.contains("lint")
        {
            metadata = metadata.with_class(IntentClass::lint_fix());
        }

        // Documentation
        if content_lower.contains("/// ")
            || content_lower.contains("//!")
            || content_lower.contains("doc comment")
            || content_lower.contains("documentation")
        {
            metadata = metadata.with_class(IntentClass::documentation());
        }

        // Refactoring
        if content_lower.contains("refactor")
            || content_lower.contains("rename")
            || content_lower.contains("extract")
        {
            metadata = metadata.with_class(IntentClass::rust_refactor());
        }

        // Structural changes
        if content_lower.contains("struct ")
            || content_lower.contains("enum ")
            || content_lower.contains("impl ")
            || content_lower.contains("trait ")
        {
            metadata = metadata.with_class(IntentClass::structural_change());
        }

        metadata
    }

    /// Extract affected paths from intent content
    pub fn extract_paths(content: &str) -> Vec<String> {
        let mut paths = Vec::new();

        // Simple line-by-line extraction
        // Look for common path patterns
        for line in content.lines() {
            // Match lines like: "*** Update File: path/to/file.rs"
            if let Some(path) = Self::extract_file_path(line) {
                paths.push(path);
            }
        }

        paths
    }

    fn extract_file_path(line: &str) -> Option<String> {
        let line = line.trim();

        // Match patch-style file markers
        if let Some(path) = line.strip_prefix("*** Update File: ") {
            return Some(path.trim().to_string());
        }
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            return Some(path.trim().to_string());
        }
        if let Some(path) = line.strip_prefix("*** Delete File: ") {
            return Some(path.trim().to_string());
        }

        // Match diff-style markers
        if let Some(path) = line.strip_prefix("+++ ") {
            return Some(path.trim().to_string());
        }
        if let Some(path) = line.strip_prefix("--- ") {
            return Some(path.trim().to_string());
        }

        None
    }

    /// Estimate diff complexity
    pub fn estimate_complexity(content: &str) -> (usize, usize) {
        let mut files = std::collections::HashSet::new();
        let mut diff_lines = 0;

        for line in content.lines() {
            // Count file changes
            if let Some(path) = Self::extract_file_path(line) {
                files.insert(path);
            }

            // Count diff lines (lines starting with + or -)
            let trimmed = line.trim();
            if trimmed.starts_with('+') || trimmed.starts_with('-') {
                diff_lines += 1;
            }
        }

        (files.len(), diff_lines)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_formatting() {
        let content = "Run cargo fmt on all files";
        let metadata = IntentClassifier::classify(content);
        assert!(metadata
            .classes
            .iter()
            .any(|c| c == &IntentClass::formatting()));
    }

    #[test]
    fn test_extract_paths() {
        let content = r#"
*** Update File: src/main.rs
*** Add File: src/lib.rs
*** Delete File: src/old.rs
        "#;

        let paths = IntentClassifier::extract_paths(content);
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&"src/main.rs".to_string()));
        assert!(paths.contains(&"src/lib.rs".to_string()));
        assert!(paths.contains(&"src/old.rs".to_string()));
    }

    #[test]
    fn test_estimate_complexity() {
        let content = r#"
*** Update File: src/main.rs
+fn new_function() {}
-fn old_function() {}
*** Update File: src/lib.rs
+pub mod test;
        "#;

        let (files, lines) = IntentClassifier::estimate_complexity(content);
        assert_eq!(files, 2);
        assert_eq!(lines, 3); // 2 changes in main.rs, 1 in lib.rs
    }
}
