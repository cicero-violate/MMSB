use crate::delta::{DeltaID, Source};
use crate::epoch::Epoch;
use crate::page::{Delta, PageAllocator, PageAllocatorConfig, PageID};
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Debug, Clone)]
pub struct MmsbDelta {
    pub patch_text: String,
    pub expected_hash: String,
    pub message_id: String,
    pub epoch: u64,
}

#[derive(Debug, Clone)]
pub struct MmsbDeltaStream {
    pub conversation_id: String,
    pub target_message_id: String,
    pub deltas: Vec<MmsbDelta>,
}

#[derive(Debug, Error)]
pub enum MmsbDeltaStreamError {
    #[error("failed to read delta file: {0}")]
    ReadDelta(#[from] std::io::Error),
    #[error("failed to parse delta json: {0}")]
    ParseDelta(#[from] serde_json::Error),
    #[error("delta hash mismatch: expected {expected}, found {found}")]
    DeltaHashMismatch { expected: String, found: String },
    #[error("missing delta_id in {0}")]
    MissingDeltaId(String),
    #[error("delta has no patch content: {0}")]
    MissingPatchContent(String),
    #[error("message chain missing entry: {0}")]
    MissingMessage(String),
    #[error("invalid message json: {0}")]
    InvalidMessage(String),
    #[error("shadow page failure: {0}")]
    ShadowPage(String),
}

#[derive(Clone)]
struct MessageEntry {
    conversation_id: String,
    message_id: String,
    parent_id: Option<String>,
    dir: PathBuf,
}

#[derive(Debug, Deserialize)]
struct PatchDelta {
    delta_id: String,
    operations: Vec<PatchOperation>,
}

#[derive(Debug, Deserialize)]
struct PatchOperation {
    content: String,
}

pub fn build_delta_streams(
    base_path: &Path,
    cwd: &Path,
) -> Result<Vec<MmsbDeltaStream>, MmsbDeltaStreamError> {
    let conversations = build_index(base_path)?;
    let mut streams = Vec::new();

    for (conversation_id, messages) in conversations {
        for message_id in messages.keys() {
            let entry = messages
                .get(message_id)
                .ok_or_else(|| MmsbDeltaStreamError::MissingMessage(message_id.clone()))?;
            let chain = message_chain(&messages, message_id);
            let mut patch_meta = Vec::new();

            for chain_id in chain {
                let chain_entry = messages
                    .get(&chain_id)
                    .ok_or_else(|| MmsbDeltaStreamError::MissingMessage(chain_id.clone()))?;
                let mut patch_files = list_patch_deltas(&chain_entry.dir);
                patch_files.sort_by_key(|path| parse_patch_seq(path).unwrap_or(0));
                for path in patch_files {
                    let raw = std::fs::read_to_string(&path)?;
                    let delta_id = extract_delta_id(&raw, &path)?;
                    let patch_text = extract_patch_text(&raw, &path)?;
                    let expected_hash = hash_patch_text(&patch_text);
                    patch_meta.push(DeltaMeta {
                        path,
                        message_id: chain_entry.message_id.clone(),
                        patch_text,
                        expected_hash,
                        delta_id,
                    });
                }
            }

            let deltas = order_deltas_with_shadow_page(&patch_meta, cwd, &conversation_id)?;
            streams.push(MmsbDeltaStream {
                conversation_id: entry.conversation_id.clone(),
                target_message_id: entry.message_id.clone(),
                deltas,
            });
        }
    }

    Ok(streams)
}

struct DeltaMeta {
    path: PathBuf,
    message_id: String,
    patch_text: String,
    expected_hash: String,
    delta_id: String,
}

fn order_deltas_with_shadow_page(
    patch_meta: &[DeltaMeta],
    cwd: &Path,
    conversation_id: &str,
) -> Result<Vec<MmsbDelta>, MmsbDeltaStreamError> {
    if patch_meta.is_empty() {
        return Ok(Vec::new());
    }

    let mut payloads = Vec::with_capacity(patch_meta.len());
    let mut max_payload = 0usize;
    for meta in patch_meta {
        let payload = meta.patch_text.as_bytes().to_vec();
        max_payload = max_payload.max(payload.len());
        payloads.push(payload);
    }
    let page_size = max_payload.max(1);
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let page_id = PageID(hash_str_to_u64(conversation_id));
    let page_ptr = allocator
        .allocate_raw(page_id, page_size, None)
        .map_err(|err| MmsbDeltaStreamError::ShadowPage(err.to_string()))?;
    let page = unsafe { page_ptr.as_mut() }
        .ok_or_else(|| MmsbDeltaStreamError::ShadowPage("null page pointer".to_string()))?;

    let mut ordered = Vec::with_capacity(patch_meta.len());
    for (index, (meta, payload)) in patch_meta.iter().zip(payloads.into_iter()).enumerate() {
        let mut dense_payload = vec![0u8; page_size];
        let copy_len = payload.len().min(page_size);
        dense_payload[..copy_len].copy_from_slice(&payload[..copy_len]);
        let mask = vec![true; page_size];
        let source = Source(cwd.to_string_lossy().to_string());
        let mut delta = Delta::new_dense(
            DeltaID(hash_str_to_u64(&meta.delta_id)),
            page_id,
            Epoch(index as u32),
            dense_payload,
            mask,
            source,
        )
        .map_err(|err| MmsbDeltaStreamError::ShadowPage(err.to_string()))?;
        delta.intent_metadata = Some(meta.delta_id.clone());
        delta
            .apply_to(page)
            .map_err(|err| MmsbDeltaStreamError::ShadowPage(err.to_string()))?;
        ordered.push(MmsbDelta {
            patch_text: meta.patch_text.clone(),
            expected_hash: meta.expected_hash.clone(),
            message_id: meta.message_id.clone(),
            epoch: index as u64,
        });
    }

    Ok(ordered)
}

fn build_index(
    base_path: &Path,
) -> Result<HashMap<String, HashMap<String, MessageEntry>>, MmsbDeltaStreamError> {
    let mut conversation_map: HashMap<String, HashMap<String, MessageEntry>> = HashMap::new();

    for entry in WalkDir::new(base_path)
        .follow_links(true)
        .into_iter()
        .filter_map(|entry| entry.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        if let Ok(relative) = path.strip_prefix(base_path) {
            if relative.components().any(|c| {
                let part = c.as_os_str();
                part == "codeblock_extractor"
                    || part == "orchestrator"
                    || part == "review_tui"
                    || part == "state_reducer"
                    || part == "patch_runner"
                    || part == "target"
            }) {
                continue;
            }
        }
        let file_stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(stem) => stem,
            None => continue,
        };
        let parent_name = match path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) {
            Some(name) => name,
            None => continue,
        };
        if file_stem != parent_name {
            continue;
        }

        let contents = std::fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&contents)?;
        let conversation_id = json
            .get("conversationId")
            .and_then(Value::as_str)
            .ok_or_else(|| MmsbDeltaStreamError::InvalidMessage(path.display().to_string()))?
            .to_string();
        let message_id = json
            .get("messageId")
            .and_then(Value::as_str)
            .ok_or_else(|| MmsbDeltaStreamError::InvalidMessage(path.display().to_string()))?
            .to_string();
        let parent_id = extract_parent_id(&json);
        let dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| base_path.to_path_buf());

        conversation_map
            .entry(conversation_id.clone())
            .or_default()
            .insert(
                message_id.clone(),
                MessageEntry {
                    conversation_id,
                    message_id,
                    parent_id,
                    dir,
                },
            );
    }

    Ok(conversation_map)
}

fn extract_parent_id(json: &Value) -> Option<String> {
    let events = json.get("sseEvents")?.as_array()?;
    for event in events {
        if let Value::Object(map) = event {
            if let Some(message) = map
                .get("message")
                .or_else(|| map.get("v").and_then(|v| v.get("message")))
            {
                if let Some(parent_id) = message
                    .get("metadata")
                    .and_then(|meta| meta.get("parent_id"))
                    .and_then(Value::as_str)
                {
                    return Some(parent_id.to_string());
                }
            }
            if let Some(parent_id) = map
                .get("input_message")
                .and_then(|msg| msg.get("metadata"))
                .and_then(|meta| meta.get("parent_id"))
                .and_then(Value::as_str)
            {
                return Some(parent_id.to_string());
            }
        }
    }
    None
}

fn message_chain(messages: &HashMap<String, MessageEntry>, start: &str) -> Vec<String> {
    let mut chain = Vec::new();
    let mut current = Some(start.to_string());
    while let Some(id) = current {
        let Some(entry) = messages.get(&id) else {
            eprintln!(
                "[mmsb-delta] missing message entry for chain id {}, treating as root",
                id
            );
            break;
        };
        chain.push(id.clone());
        current = entry.parent_id.clone();
    }
    chain.reverse();
    chain
}

fn list_patch_deltas(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| name.starts_with("patch_06_delta_") && name.ends_with(".json"))
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
    }
    files
}

fn parse_patch_seq(path: &Path) -> Option<usize> {
    let file_name = path.file_name()?.to_string_lossy();
    let stem = file_name.strip_suffix(".json")?;
    let number = stem.strip_prefix("patch_06_delta_")?;
    let seq_part = number.split('-').next()?;
    seq_part.parse::<usize>().ok()
}

fn extract_delta_id(
    raw: &str,
    path: &Path,
) -> Result<String, MmsbDeltaStreamError> {
    let mut value: Value = serde_json::from_str(raw)?;
    let delta_id = value
        .get("delta_id")
        .and_then(Value::as_str)
        .ok_or_else(|| MmsbDeltaStreamError::MissingDeltaId(path.display().to_string()))?
        .to_string();
    if let Value::Object(map) = &mut value {
        map.insert(
            "delta_id".to_string(),
            Value::String("sha256:__DERIVED__".to_string()),
        );
    }
    let canonical = serde_json::to_string(&value)?;
    let hash = sha256_hex(canonical.as_bytes());
    let expected = format!("sha256:{hash}");
    if expected != delta_id {
        return Err(MmsbDeltaStreamError::DeltaHashMismatch {
            expected,
            found: delta_id,
        });
    }
    Ok(expected)
}

fn extract_patch_text(
    raw: &str,
    path: &Path,
) -> Result<String, MmsbDeltaStreamError> {
    let delta: PatchDelta = serde_json::from_str(raw)?;
    for op in delta.operations {
        if !op.content.trim().is_empty() {
            return Ok(op.content);
        }
    }
    Err(MmsbDeltaStreamError::MissingPatchContent(
        path.display().to_string(),
    ))
}

fn hash_patch_text(patch_text: &str) -> String {
    format!("sha256:{}", sha256_hex(patch_text.as_bytes()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn hash_str_to_u64(input: &str) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3],
        digest[4], digest[5], digest[6], digest[7],
    ])
}
