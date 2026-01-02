# Structure Group: src/020_gather_rust_files.rs

## File: src/020_gather_rust_files.rs

- Layer(s): 020_gather_rust_files.rs
- Language coverage: Rust (1)
- Element types: Function (1)
- Total elements: 1

### Elements

- [Rust | Function] `gather_rust_files` (line 0, pub)
  - Signature: `pub fn gather_rust_files (root : & Path) -> Vec < PathBuf > { use walkdir :: WalkDir ; let src_root = resolve_source_...`
  - Calls: resolve_source_root, collect, map, filter, filter, filter_map, filter_entry, into_iter, WalkDir::new, depth, is_dir, file_type, allow_analysis_dir, path, ok, map_or, extension, path, unwrap_or, strip_prefix, path, path, count, components, starts_with, path, join, into_path

