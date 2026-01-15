use super::admission_proof::build_admission_proof_streams;
use super::delta_stream::build_delta_streams;
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const EXECUTION_PROOF_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct MmsbExecutionProof {
    pub version: u32,
    pub delta_hash: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub output: Value,
    pub epoch: u64,
}

#[derive(Debug, Clone)]
pub struct MmsbExecutionProofStream {
    pub conversation_id: String,
    pub message_id: String,
    pub proofs: Vec<MmsbExecutionProof>,
}

#[derive(Debug, Error)]
pub enum MmsbExecutionProofError {
    #[error("failed to build delta stream: {0}")]
    DeltaStream(#[from] super::delta_stream::MmsbDeltaStreamError),
    #[error("failed to build admission proofs: {0}")]
    AdmissionStream(#[from] super::admission_proof::MmsbAdmissionProofError),
    #[error("failed to read file: {0}")]
    ReadFile(#[from] std::io::Error),
    #[error("failed to parse json: {0}")]
    ParseJson(#[from] serde_json::Error),
    #[error("conversation stream not found: {0}")]
    MissingConversation(String),
}

#[derive(Debug, Deserialize)]
struct PatchApplyFile {
    stdout: Option<String>,
    stderr: Option<String>,
    errors: Option<Vec<Value>>,
    status: Option<String>,
    exit_code: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct ShellApply {
    intent_hash: Option<String>,
    exit_code: Value,
    stdout: Option<String>,
    stderr: Option<String>,
}

pub fn build_execution_proof_stream(
    messages_dir: &Path,
    conversation_id: &str,
    message_id: &str,
) -> Result<MmsbExecutionProofStream, MmsbExecutionProofError> {
    let conversation_dir = messages_dir.join(conversation_id);
    let delta_streams = build_delta_streams(&conversation_dir, messages_dir)?;
    let admission_streams = build_admission_proof_streams(&conversation_dir)?;

    let mut proofs = Vec::new();
    let mut epoch = 0u64;

    if let Some(stream) = delta_streams
        .iter()
        .find(|stream| stream.conversation_id == conversation_id && stream.target_message_id == message_id)
    {
        let apply_path = messages_dir
            .join(conversation_id)
            .join(message_id)
            .join(format!("patch_05_apply_{message_id}.json"));
        if apply_path.is_file() {
            if let Some(text) = patch_output_text_from_apply_file(&apply_path)? {
                for delta in stream.deltas.iter().filter(|delta| delta.message_id == message_id) {
                    proofs.push(MmsbExecutionProof {
                        version: EXECUTION_PROOF_VERSION,
                        delta_hash: delta.expected_hash.clone(),
                        tool_call_id: format!("call_{}_{}", message_id, delta.epoch),
                        tool_name: "apply_patch".to_string(),
                        output: json!({ "text": text }),
                        epoch,
                    });
                    epoch += 1;
                }
            }
        }
    }

    if let Some(stream) = admission_streams
        .iter()
        .find(|stream| stream.conversation_id == conversation_id)
    {
        for proof in stream.proofs.iter().filter(|proof| proof.message_id == message_id) {
            let apply_path = messages_dir
                .join(&proof.conversation_id)
                .join(&proof.message_id)
                .join(format!("shell_05_apply_{}.json", proof.suffix));
            if !apply_path.is_file() {
                continue;
            }
            let apply = load_shell_apply(&apply_path)?;
            if should_skip_shell_apply(&apply) {
                continue;
            }
            let intent_hash = apply
                .intent_hash
                .clone()
                .unwrap_or_else(|| proof.intent_hash.clone());
            proofs.push(MmsbExecutionProof {
                version: EXECUTION_PROOF_VERSION,
                delta_hash: intent_hash.clone(),
                tool_call_id: format!("shell_apply_{}", intent_hash),
                tool_name: "shell_runner".to_string(),
                output: json!({
                    "type": "shell_apply",
                    "intent_hash": intent_hash,
                    "exit_code": apply.exit_code,
                    "stdout": apply.stdout,
                    "stderr": apply.stderr,
                }),
                epoch,
            });
            epoch += 1;
        }
    }

    let message_dir = messages_dir.join(conversation_id).join(message_id);
    if message_dir.is_dir() {
        let mut plan_applies = list_plan_applies(&message_dir);
        plan_applies.sort_by(|a, b| a.cmp(b));
        for apply_path in plan_applies {
            let apply = load_plan_apply(&apply_path)?;
            let suffix = suffix_from_plan_apply_path(&apply_path).unwrap_or_else(|| "plan".to_string());
            let mut output = serde_json::Map::new();
            let steps: Vec<Value> = apply
                .plan
                .iter()
                .map(|step| {
                    json!({
                        "step": step.step,
                        "status": step.status,
                    })
                })
                .collect();
            output.insert("plan".to_string(), Value::Array(steps));
            if let Some(explanation) = apply.explanation {
                output.insert("explanation".to_string(), Value::String(explanation));
            }
            let delta_hash = format!("sha256:{}", apply.intent_hash);
            proofs.push(MmsbExecutionProof {
                version: EXECUTION_PROOF_VERSION,
                delta_hash,
                tool_call_id: format!("call_{}_plan_{}", message_id, suffix),
                tool_name: "update_plan".to_string(),
                output: Value::Object(output),
                epoch,
            });
            epoch += 1;
        }
    }

    Ok(MmsbExecutionProofStream {
        conversation_id: conversation_id.to_string(),
        message_id: message_id.to_string(),
        proofs,
    })
}

fn patch_output_text_from_apply_file(path: &PathBuf) -> Result<Option<String>, MmsbExecutionProofError> {
    let contents = std::fs::read_to_string(path)?;
    let apply: PatchApplyFile = serde_json::from_str(&contents)?;
    let mut text = String::new();
    if let Some(stdout) = apply.stdout.as_ref().map(|s| s.trim_end()).filter(|s| !s.is_empty()) {
        text.push_str(stdout);
    }
    if let Some(stderr) = apply.stderr.as_ref().map(|s| s.trim_end()).filter(|s| !s.is_empty()) {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(stderr);
    }
    if let Some(errors) = apply.errors.as_ref() {
        for err in errors {
            let err_text = err.as_str().map(|s| s.to_string()).unwrap_or_else(|| {
                serde_json::to_string(err).unwrap_or_else(|_| err.to_string())
            });
            if err_text.is_empty() {
                continue;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(&err_text);
        }
    }
    if text.is_empty() {
        if let Some(status) = apply.status.as_ref() {
            text = status.clone();
        } else if let Some(code) = apply.exit_code.as_ref() {
            text = code.to_string();
        }
    }
    if text.is_empty() {
        return Ok(None);
    }
    Ok(Some(text))
}

fn load_shell_apply(path: &Path) -> Result<ShellApply, MmsbExecutionProofError> {
    let contents = std::fs::read_to_string(path)?;
    let apply: ShellApply = serde_json::from_str(&contents)?;
    Ok(apply)
}

fn should_skip_shell_apply(apply: &ShellApply) -> bool {
    match &apply.exit_code {
        Value::String(text) if text == "n/a" => true,
        Value::Null => true,
        _ => false,
    }
}

#[derive(Debug, Deserialize)]
struct PlanApply {
    #[serde(default)]
    explanation: Option<String>,
    plan: Vec<PlanStep>,
    intent_hash: String,
}

#[derive(Debug, Deserialize)]
struct PlanStep {
    step: String,
    status: String,
}

fn list_plan_applies(message_dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(message_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|name| {
                    name.starts_with("plan_05_apply_")
                        && name.ends_with(".json")
                        && !name.ends_with(".judgment.json")
                        && !name.ends_with(".judgment_scope.json")
                })
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
    }
    files
}

fn suffix_from_plan_apply_path(path: &Path) -> Option<String> {
    let file_name = path.file_name()?.to_str()?;
    let stem = file_name.strip_suffix(".json")?;
    stem.strip_prefix("plan_05_apply_").map(|s| s.to_string())
}

fn load_plan_apply(path: &Path) -> Result<PlanApply, MmsbExecutionProofError> {
    let contents = std::fs::read_to_string(path)?;
    let apply: PlanApply = serde_json::from_str(&contents)?;
    Ok(apply)
}
