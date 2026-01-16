use std::path::PathBuf;

use mmsb_core::dag::DependencyGraph;
use structural_code_editor::{
    analyze_code_edit, extract_intent, index_page, index_snapshot, propagate_edits, RepoSnapshot,
};
use mmsb_judgment::issue::issue_judgment;
use structural_code_indexer::fs::{read_source_files, SourceFile};

fn print_snapshot(label: &str, snapshot: &RepoSnapshot) {
    println!("{label} files:");
    for file in &snapshot.files {
        println!("  {} -> PageID({})", file.path.display(), stable_page_id(&file.path));
    }
}

fn stable_page_id(path: &std::path::Path) -> u64 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let bytes = path.to_string_lossy();
    hasher.update(bytes.as_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ])
}

fn normalize_snapshot(snapshot: RepoSnapshot, root: &PathBuf) -> RepoSnapshot {
    let files = snapshot
        .files
        .into_iter()
        .map(|file| SourceFile {
            path: file
                .path
                .strip_prefix(root)
                .unwrap_or(&file.path)
                .to_path_buf(),
            content: file.content,
        })
        .collect();

    RepoSnapshot { files }
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("test_project");
    let before_dir = root.join("before");
    let after_dir = root.join("after");

    let before = normalize_snapshot(
        RepoSnapshot {
            files: read_source_files(&before_dir),
        },
        &before_dir,
    );
    let after = normalize_snapshot(
        RepoSnapshot {
            files: read_source_files(&after_dir),
        },
        &after_dir,
    );

    print_snapshot("before", &before);
    print_snapshot("after", &after);

    let before_scan = structural_code_editor::scan::scan_repo(&before)
        .expect("scan failed");
    let active_dag = before_scan.graph.clone();
    let edit = analyze_code_edit(&before, &after, &active_dag)
        .expect("analysis failed");

    println!("deltas: {}", edit.deltas.len());
    println!("ops: {}", edit.ops.len());

    for op in &edit.ops {
        println!("op: {:?}", op);
    }

    for delta in &edit.deltas {
        println!(
            "delta: page_id={} payload_len={}",
            delta.page_id.0,
            delta.payload.len()
        );
    }

    let after_scan = structural_code_editor::scan::scan_repo(&after)
        .expect("scan failed");
    let (index_store, source_store) = index_snapshot(&after);

    let before_root = before
        .files
        .iter()
        .find(|file| file.path.ends_with("src/lib.rs"))
        .expect("missing before src/lib.rs");
    let after_root = after
        .files
        .iter()
        .find(|file| file.path.ends_with("src/lib.rs"))
        .expect("missing after src/lib.rs");

    let before_index = index_page(&before_root.path, &before_root.content);
    let after_index = index_page(&after_root.path, &after_root.content);
    let intents = extract_intent(&before_index, &after_index);

    println!("intents: {}", intents.len());
    for intent in &intents {
        println!("intent: {:?}", intent);
    }

    let root_page = before_index.page_id;
    let judgment = issue_judgment("example propagation", "propagate");
    let propagated = propagate_edits(
        root_page,
        &intents,
        &after_scan.graph,
        &index_store,
        &source_store,
        &judgment,
    );

    if propagated.is_empty() {
        println!("propagation: no dependent rewrites");
    } else {
        for delta in &propagated {
            println!(
                "propagation: page_id={} reason={}",
                delta.page_id.0,
                delta.reason
            );
        }
    }
}
