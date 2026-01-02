use crate::layer_utilities::allow_analysis_dir;
use crate::layer_utilities::resolve_source_root;
use std::path::Path;
use std::path::PathBuf;
pub fn gather_rust_files(root: &Path) -> Vec<PathBuf> {
    use walkdir::WalkDir;

    let src_root = resolve_source_root(root);
    WalkDir::new(&src_root)
        .into_iter()
        .filter_entry(|entry| {
            if entry.depth() == 0 {
                return true;
            }
            if !entry.file_type().is_dir() {
                return true;
            }
            allow_analysis_dir(&src_root, entry.path())
        })
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "rs"))
        .filter(|e| {
            let rel = e.path().strip_prefix(&src_root).unwrap_or(e.path());
            rel.components().count() == 1 || e.path().starts_with(src_root.join("src"))
        })
        .map(|entry| entry.into_path())
        .collect()
}
