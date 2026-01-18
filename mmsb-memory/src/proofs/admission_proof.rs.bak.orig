use crate::delta::{DeltaID, Source};
use crate::epoch::Epoch;
use crate::page::{Delta, PageAllocator, PageAllocatorConfig};
use mmsb_primitives::PageID;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;
use walkdir::WalkDir;

pub const ADMISSION_PROOF_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct MmsbAdmissionProof {
    pub version: u32,
    pub delta_hash: String,
    pub dag_snapshot_hash: Option<String>,
    pub conversation_id: String,
    pub message_id: String,
    pub suffix: String,
    pub intent_hash: String,
    pub approved: bool,
    pub command: Vec<String>,
    pub cwd: Option<String>,
    pub env: Option<BTreeMap<String, String>>,
    pub epoch: u64,
}

#[derive(Debug, Clone)]
pub struct MmsbAdmission {
    pub command: Vec<String>,
    pub cwd: Option<String>,
}

#[derive(Debug, Clone)]
pub struct MmsbAdmissionProofStream {
    pub conversation_id: String,
    pub proofs: Vec<MmsbAdmissionProof>,
}

#[derive(Debug, Error)]
pub enum MmsbAdmissionProofError {
    #[error("failed to read file: {0}")]
    ReadFile(#[from] std::io::Error),
    #[error("failed to parse json: {0}")]
    ParseJson(#[from] serde_json::Error),
    #[error("invalid message json: {0}")]
    InvalidMessage(String),
    #[error("missing message entry: {0}")]
    MissingMessage(String),
    #[error("invalid intent: {0}")]
    InvalidIntent(String),
    #[error("intent hash mismatch: expected {expected}, found {found}")]
    IntentHashMismatch { expected: String, found: String },
    #[error("policy error: {0}")]
    PolicyError(String),
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
#[serde(deny_unknown_fields)]
struct ShellIntent {
    #[serde(default)]
    schema: Option<String>,
    #[serde(default)]
    judgment_id: Option<String>,
    delta_type: String,
    hash: String,
    command: Vec<String>,
    #[serde(default)]
    intent_class: Option<String>,
    cwd: Option<String>,
    env: Option<BTreeMap<String, String>>,
    declared_inputs: Option<Vec<String>>,
    declared_outputs: Option<Vec<String>>,
    constraints: Option<Value>,
}

#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDecision {
    Allow,
    Deny,
    RequireApproval,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellPolicy {
    policy_type: String,
    default_decision: PolicyDecision,
    rules: Vec<ShellPolicyRule>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellPolicyRule {
    id: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "match")]
    matcher: ShellPolicyMatch,
    decision: PolicyDecision,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShellPolicyMatch {
    command_prefix: Option<Vec<String>>,
    cwd_prefix: Option<String>,
}

#[derive(Debug, Error)]
pub enum PolicyError {
    #[error("failed to read policy: {0}")]
    ReadPolicy(#[from] std::io::Error),
    #[error("failed to parse policy json: {0}")]
    ParsePolicy(#[from] serde_json::Error),
    #[error("invalid policy_type: {0}")]
    InvalidPolicyType(String),
    #[error("policy file not found")]
    PolicyNotFound,
}

pub fn evaluate_admission(
    admission: &MmsbAdmission,
    policy: &ShellPolicy,
) -> Result<PolicyDecision, PolicyError> {
    if policy.policy_type != "shell.policy.v1" {
        return Err(PolicyError::InvalidPolicyType(policy.policy_type.clone()));
    }
    for rule in &policy.rules {
        if policy_match(rule, admission) {
            return Ok(rule.decision.clone());
        }
    }
    Ok(policy.default_decision.clone())
}

fn policy_match(rule: &ShellPolicyRule, admission: &MmsbAdmission) -> bool {
    if let Some(prefixes) = &rule.matcher.command_prefix {
        if let Some(cmd) = admission.command.first() {
            if !prefixes.iter().any(|prefix| prefix == cmd) {
                return false;
            }
        } else {
            return false;
        }
    }
    if let Some(prefix) = &rule.matcher.cwd_prefix {
        if let Some(cwd) = &admission.cwd {
            if !cwd.starts_with(prefix) {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

pub fn load_shell_policy(path: &Path) -> Result<ShellPolicy, PolicyError> {
    let data = std::fs::read_to_string(path)?;
    let policy: ShellPolicy = serde_json::from_str(&data)?;
    if policy.policy_type != "shell.policy.v1" {
        return Err(PolicyError::InvalidPolicyType(policy.policy_type));
    }
    Ok(policy)
}

pub fn build_admission_proof_streams(
    base_path: &Path,
) -> Result<Vec<MmsbAdmissionProofStream>, MmsbAdmissionProofError> {
    let policy_path = find_shell_policy_path().map_err(|err| {
        MmsbAdmissionProofError::PolicyError(err.to_string())
    })?;
    let policy = load_shell_policy(&policy_path)
        .map_err(|err| MmsbAdmissionProofError::PolicyError(err.to_string()))?;
    let conversations = build_index(base_path)?;
    let mut streams = Vec::new();

    let mut conversation_ids: Vec<String> = conversations.keys().cloned().collect();
    conversation_ids.sort();

    for conversation_id in conversation_ids {
        let messages = conversations
            .get(&conversation_id)
            .ok_or_else(|| MmsbAdmissionProofError::MissingMessage(conversation_id.clone()))?;
        let message_order = ordered_messages(messages)?;
        let mut proofs = Vec::new();

        for message_id in message_order {
            let entry = messages
                .get(&message_id)
                .ok_or_else(|| MmsbAdmissionProofError::MissingMessage(message_id.clone()))?;
            let mut intent_files = list_shell_intents(&entry.dir);
            intent_files.sort_by(|a, b| compare_shell_suffix(a, b));

            eprintln!(
                "[mmsb-admission] {} intents found for message {}",
                intent_files.len(),
                message_id
            );
            for intent_path in intent_files {
                let suffix = suffix_from_intent_path(&intent_path)
                    .ok_or_else(|| {
                        MmsbAdmissionProofError::InvalidIntent(intent_path.display().to_string())
                    })?;
                eprintln!(
                    "[mmsb-admission] loading intent path={} suffix={}",
                    intent_path.display(),
                    suffix
                );
                let intent = match load_intent(&intent_path)? {
                    Some(intent) => intent,
                    None => {
                        eprintln!(
                            "[mmsb-admission] skipped intent path={} (judgment artifact)",
                            intent_path.display()
                        );
                        continue;
                    }
                };
                if intent.delta_type != "shell.intent.v1" {
                    return Err(MmsbAdmissionProofError::InvalidIntent(format!(
                        "Invalid delta_type: {}",
                        intent.delta_type
                    )));
                }
                if intent.command.is_empty() {
                    return Err(MmsbAdmissionProofError::InvalidIntent(format!(
                        "Empty command: {}",
                        intent_path.display()
                    )));
                }
                let expected_hash = compute_intent_hash(&intent)?;
                if intent.hash != expected_hash {
                    return Err(MmsbAdmissionProofError::IntentHashMismatch {
                        expected: expected_hash,
                        found: intent.hash,
                    });
                }

                let admission = MmsbAdmission {
                    command: intent.command.clone(),
                    cwd: intent.cwd.clone(),
                };
                let decision = evaluate_admission(&admission, &policy)
                    .map_err(|err| MmsbAdmissionProofError::PolicyError(err.to_string()))?;
                let approved = match decision {
                    PolicyDecision::Allow => true,
                    PolicyDecision::RequireApproval => {
                        let ok = intent_has_valid_judgment(&intent_path);
                        eprintln!(
                            "[mmsb-admission] require-approval intent={} judgment_ok={}",
                            intent_path.display(),
                            ok
                        );
                        ok
                    }
                    PolicyDecision::Deny => false,
                };
                eprintln!(
                    "[mmsb-admission] decision intent={} decision={:?} approved={}",
                    intent_path.display(),
                    decision,
                    approved
                );

                proofs.push(MmsbAdmissionProof {
                    version: ADMISSION_PROOF_VERSION,
                    delta_hash: intent.hash.clone(),
                    dag_snapshot_hash: None,
                    conversation_id: entry.conversation_id.clone(),
                    message_id: entry.message_id.clone(),
                    suffix,
                    intent_hash: intent.hash,
                    approved,
                    command: intent.command,
                    cwd: intent.cwd,
                    env: intent.env,
                    epoch: 0,
                });
            }
        }

        let proofs = order_with_shadow_page(&proofs, &conversation_id)?;
        streams.push(MmsbAdmissionProofStream {
            conversation_id,
            proofs,
        });
    }

    Ok(streams)
}

fn order_with_shadow_page(
    proofs: &[MmsbAdmissionProof],
    conversation_id: &str,
) -> Result<Vec<MmsbAdmissionProof>, MmsbAdmissionProofError> {
    if proofs.is_empty() {
        return Ok(Vec::new());
    }

    let mut payloads = Vec::with_capacity(proofs.len());
    let mut max_payload = 0usize;
    for proof in proofs {
        let payload = proof.intent_hash.as_bytes().to_vec();
        max_payload = max_payload.max(payload.len());
        payloads.push(payload);
    }
    let page_size = max_payload.max(1);
    let allocator = PageAllocator::new(PageAllocatorConfig::default());
    let page_id = PageID(hash_str_to_u64(conversation_id));
    let page_ptr = allocator
        .allocate_raw(page_id, page_size, None)
        .map_err(|err| MmsbAdmissionProofError::ShadowPage(err.to_string()))?;
    let page = unsafe { page_ptr.as_mut() }
        .ok_or_else(|| MmsbAdmissionProofError::ShadowPage("null page pointer".to_string()))?;

    let mut ordered = Vec::with_capacity(proofs.len());
    for (index, (proof, payload)) in proofs.iter().zip(payloads.into_iter()).enumerate() {
        let mut dense_payload = vec![0u8; page_size];
        let copy_len = payload.len().min(page_size);
        dense_payload[..copy_len].copy_from_slice(&payload[..copy_len]);
        let mask = vec![true; page_size];
        let source = Source("shell_admission".to_string());
        let mut delta = Delta::new_dense(
            DeltaID(hash_str_to_u64(&proof.intent_hash)),
            page_id,
            Epoch(index as u32),
            dense_payload,
            mask,
            source,
        )
        .map_err(|err| MmsbAdmissionProofError::ShadowPage(err.to_string()))?;
        delta.intent_metadata = Some(proof.intent_hash.clone());
        delta
            .apply_to(page)
            .map_err(|err| MmsbAdmissionProofError::ShadowPage(err.to_string()))?;
        let mut proof = proof.clone();
        proof.epoch = index as u64;
        ordered.push(proof);
    }

    Ok(ordered)
}

fn build_index(
    base_path: &Path,
) -> Result<HashMap<String, HashMap<String, MessageEntry>>, MmsbAdmissionProofError> {
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
            .ok_or_else(|| MmsbAdmissionProofError::InvalidMessage(path.display().to_string()))?
            .to_string();
        let message_id = json
            .get("messageId")
            .and_then(Value::as_str)
            .ok_or_else(|| MmsbAdmissionProofError::InvalidMessage(path.display().to_string()))?
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

fn ordered_messages(
    messages: &HashMap<String, MessageEntry>,
) -> Result<Vec<String>, MmsbAdmissionProofError> {
    let mut cache = HashMap::new();
    let mut ids: Vec<String> = messages.keys().cloned().collect();
    ids.sort_by(|a, b| {
        let depth_a = message_depth(a, messages, &mut cache).unwrap_or(0);
        let depth_b = message_depth(b, messages, &mut cache).unwrap_or(0);
        depth_a.cmp(&depth_b).then_with(|| a.cmp(b))
    });
    Ok(ids)
}

fn message_depth(
    id: &str,
    messages: &HashMap<String, MessageEntry>,
    cache: &mut HashMap<String, usize>,
) -> Result<usize, MmsbAdmissionProofError> {
    if let Some(depth) = cache.get(id) {
        return Ok(*depth);
    }
    let entry = messages
        .get(id)
        .ok_or_else(|| MmsbAdmissionProofError::MissingMessage(id.to_string()))?;
    let depth = match entry.parent_id.as_ref() {
        Some(parent) => message_depth(parent, messages, cache)? + 1,
        None => 0,
    };
    cache.insert(id.to_string(), depth);
    Ok(depth)
}

fn list_shell_intents(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| {
                    name.starts_with("shell_01_intent_")
                        && name.ends_with(".json")
                        && !name.contains(".judgment.")
                })
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
    }
    if files.is_empty() {
        eprintln!(
            "[mmsb-admission] no shell intents found in {}",
            dir.display()
        );
    }
    files
}

fn suffix_from_intent_path(intent_path: &Path) -> Option<String> {
    let file_name = intent_path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".json")?;
    stem.strip_prefix("shell_01_intent_").map(|s| s.to_string())
}

fn compare_shell_suffix(a: &PathBuf, b: &PathBuf) -> std::cmp::Ordering {
    let suffix_a = suffix_from_intent_path(a).unwrap_or_default();
    let suffix_b = suffix_from_intent_path(b).unwrap_or_default();
    let seq_a = suffix_a.parse::<usize>().ok();
    let seq_b = suffix_b.parse::<usize>().ok();
    match (seq_a, seq_b) {
        (Some(a), Some(b)) => a.cmp(&b).then_with(|| suffix_a.cmp(&suffix_b)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => suffix_a.cmp(&suffix_b),
    }
}

fn load_intent(path: &Path) -> Result<Option<ShellIntent>, MmsbAdmissionProofError> {
    let data = std::fs::read_to_string(path)?;
    let value: Value = serde_json::from_str(&data)?;
    if let Some(obj) = value.as_object() {
        let schema = obj
            .get("schema")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        if schema.starts_with("judgment.") || obj.contains_key("judgment_id") {
            return Ok(None);
        }
    }
    let intent: ShellIntent = match serde_json::from_value(value.clone()) {
        Ok(intent) => intent,
        Err(err) => {
            if let Some(obj) = value.as_object() {
                let schema = obj
                    .get("schema")
                    .and_then(|value| value.as_str())
                    .unwrap_or("");
                let has_judgment_id = obj.contains_key("judgment_id");
                let mut keys: Vec<&str> = obj.keys().map(|key| key.as_str()).collect();
                keys.sort_unstable();
                eprintln!(
                    "[mmsb-admission] Failed to parse shell intent at {}: {}",
                    path.display(),
                    err
                );
                eprintln!(
                    "[mmsb-admission] intent schema='{}' judgment_id={} keys={}",
                    schema,
                    has_judgment_id,
                    keys.join(",")
                );
            } else {
                eprintln!(
                    "[mmsb-admission] Failed to parse shell intent at {}: {}",
                    path.display(),
                    err
                );
                eprintln!(
                    "[mmsb-admission] intent value is not a JSON object"
                );
            }
            return Err(MmsbAdmissionProofError::ParseJson(err));
        }
    };
    Ok(Some(intent))
}

fn intent_has_valid_judgment(intent_path: &Path) -> bool {
    let judgment_path = PathBuf::from(format!("{}.judgment.json", intent_path.display()));
    if !judgment_path.is_file() {
        eprintln!(
            "[mmsb-admission] judgment missing for {}",
            judgment_path.display()
        );
        return false;
    }
    let contents = match fs::read_to_string(&judgment_path) {
        Ok(contents) => contents,
        Err(err) => {
            eprintln!(
                "[mmsb-admission] Failed to read judgment {}: {err}",
                judgment_path.display()
            );
            return false;
        }
    };
    let json: Value = match serde_json::from_str(&contents) {
        Ok(json) => json,
        Err(err) => {
            eprintln!(
                "[mmsb-admission] Invalid judgment JSON {}: {err}",
                judgment_path.display()
            );
            return false;
        }
    };
    let schema = json.get("schema").and_then(|value| value.as_str());
    if schema != Some("judgment.v1") {
        eprintln!(
            "[mmsb-admission] Unexpected judgment schema {} in {}",
            schema.unwrap_or("<missing>"),
            judgment_path.display()
        );
        return false;
    }
    let acknowledged = json
        .get("acknowledged")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    if !acknowledged {
        eprintln!(
            "[mmsb-admission] Judgment not acknowledged: {}",
            judgment_path.display()
        );
        return false;
    }
    let delta_hash = json.get("delta_hash").and_then(|value| value.as_str());
    let intent_bytes = match fs::read(intent_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!(
                "[mmsb-admission] Failed to read intent for judgment check {}: {err}",
                intent_path.display()
            );
            return false;
        }
    };
    let intent_hash = format!("{:x}", Sha256::digest(&intent_bytes));
    if delta_hash != Some(intent_hash.as_str()) {
        eprintln!(
            "[mmsb-admission] Judgment hash mismatch for {} (expected {}, found {})",
            intent_path.display(),
            intent_hash,
            delta_hash.unwrap_or("<missing>")
        );
        return false;
    }
    eprintln!(
        "[mmsb-admission] judgment ok for {}",
        intent_path.display()
    );
    true
}

fn compute_intent_hash(intent: &ShellIntent) -> Result<String, MmsbAdmissionProofError> {
    let mut intent_map = BTreeMap::new();
    intent_map.insert(
        "command".to_string(),
        Value::Array(intent.command.iter().cloned().map(Value::String).collect()),
    );
    if let Some(intent_class) = intent.intent_class.as_ref() {
        intent_map.insert("intent_class".to_string(), Value::String(intent_class.clone()));
    }
    intent_map.insert(
        "constraints".to_string(),
        intent.constraints.clone().unwrap_or(Value::Null),
    );
    intent_map.insert(
        "cwd".to_string(),
        intent
            .cwd
            .as_ref()
            .map(|value| Value::String(value.clone()))
            .unwrap_or(Value::Null),
    );
    intent_map.insert(
        "declared_inputs".to_string(),
        intent
            .declared_inputs
            .as_ref()
            .map(|inputs| Value::Array(inputs.iter().cloned().map(Value::String).collect()))
            .unwrap_or(Value::Null),
    );
    intent_map.insert(
        "declared_outputs".to_string(),
        intent
            .declared_outputs
            .as_ref()
            .map(|outputs| Value::Array(outputs.iter().cloned().map(Value::String).collect()))
            .unwrap_or(Value::Null),
    );
    intent_map.insert(
        "delta_type".to_string(),
        Value::String(intent.delta_type.clone()),
    );
    intent_map.insert(
        "env".to_string(),
        intent
            .env
            .as_ref()
            .map(|env| {
                Value::Object(
                    env.iter()
                        .map(|(key, value)| (key.clone(), Value::String(value.clone())))
                        .collect(),
                )
            })
            .unwrap_or(Value::Null),
    );
    intent_map.insert(
        "hash".to_string(),
        Value::String("sha256:__DERIVED__".to_string()),
    );

    let canonical = serde_json::to_string(&intent_map)?;
    Ok(hash_bytes(canonical.as_bytes()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
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

fn find_shell_policy_path() -> Result<PathBuf, PolicyError> {
    if let Ok(path) = std::env::var("MMSB_SHELL_POLICY_PATH") {
        let path = PathBuf::from(path);
        eprintln!(
            "[mmsb-admission] MMSB_SHELL_POLICY_PATH={}",
            path.display()
        );
        if path.is_file() {
            return Ok(path);
        }
        eprintln!(
            "[mmsb-admission] MMSB_SHELL_POLICY_PATH not found at {}",
            path.display()
        );
    }

    let start = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut current = start.as_path();
    eprintln!("[mmsb-admission] policy search start={}", start.display());
    loop {
        let candidate = current.join("tools").join("shell_runner").join("shell.policy.v1.json");
        eprintln!(
            "[mmsb-admission] policy candidate={}",
            candidate.display()
        );
        if candidate.is_file() {
            return Ok(candidate);
        }
        let fallback = current.join("shell.policy.v1.json");
        eprintln!(
            "[mmsb-admission] policy fallback={}",
            fallback.display()
        );
        if fallback.is_file() {
            return Ok(fallback);
        }
        if let Some(parent) = current.parent() {
            current = parent;
        } else {
            break;
        }
    }

    Err(PolicyError::PolicyNotFound)
}
