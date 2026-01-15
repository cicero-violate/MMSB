use sha2::{Digest, Sha256};
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{SystemTime, UNIX_EPOCH};

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

fn write_intent_file(dir: &std::path::Path, name: &str, contents: &[u8]) -> std::path::PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).expect("write intent");
    path
}

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{nanos}"))
}

fn find_judgment_bin() -> PathBuf {
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_mmsb-judgment") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_mmsb_judgment") {
        return PathBuf::from(path);
    }
    let exe = std::env::current_exe().expect("current exe");
    let target_dir = exe
        .parent()
        .and_then(|p| p.parent())
        .expect("target/debug");
    let candidate = target_dir.join("mmsb-judgment");
    if candidate.is_file() {
        return candidate;
    }
    panic!("missing mmsb-judgment bin");
}

#[test]
fn issue_fails_on_hash_mismatch() {
    let bin = find_judgment_bin();
    let temp_dir = unique_temp_dir("mmsb_judgment_cli_hash_mismatch");
    fs::create_dir_all(&temp_dir).expect("create temp dir");

    let intent_path = write_intent_file(&temp_dir, "intent.json", b"{\"intent\":\"test\"}\n");
    let judgment_path = std::path::PathBuf::from(format!("{}.judgment.json", intent_path.display()));

    let mut child = Command::new(bin)
        .arg("issue")
        .arg("--intent")
        .arg(&intent_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mmsb-judgment");

    {
        let stdin = child.stdin.as_mut().expect("stdin");
        stdin.write_all(b"\n").expect("enter");
        stdin
            .write_all(b"deadbeef\n")
            .expect("hash mismatch");
    }

    let output = child.wait_with_output().expect("wait");
    assert!(!output.status.success(), "expected failure status");
    assert!(
        !judgment_path.exists(),
        "judgment artifact should not be written on failure"
    );
}

#[test]
fn issue_writes_artifact_and_outputs_token() {
    let bin = find_judgment_bin();
    let temp_dir = unique_temp_dir("mmsb_judgment_cli_success");
    fs::create_dir_all(&temp_dir).expect("create temp dir");

    let intent_bytes = b"{\"intent\":\"ok\"}\n";
    let intent_path = write_intent_file(&temp_dir, "intent.json", intent_bytes);
    let judgment_path = std::path::PathBuf::from(format!("{}.judgment.json", intent_path.display()));
    let hash = sha256_hex(intent_bytes);
    let input = format!("\n{}\n1\n", hash);

    let output = Command::new(bin)
        .arg("issue")
        .arg("--intent")
        .arg(&intent_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(input.as_bytes())?;
            child.wait_with_output()
        })
        .expect("run mmsb-judgment");

    assert!(output.status.success(), "expected success status");
    let token = String::from_utf8_lossy(&output.stdout).trim().to_string();
    assert!(is_lower_hex_64(&token), "token must be 64 hex chars");
    assert!(judgment_path.exists(), "judgment artifact not written");

    let artifact = fs::read_to_string(&judgment_path).expect("read artifact");
    let json: serde_json::Value = serde_json::from_str(&artifact).expect("parse artifact");
    assert_eq!(json.get("schema").and_then(|v| v.as_str()), Some("judgment.v1"));
    assert_eq!(
        json.get("delta_hash").and_then(|v| v.as_str()),
        Some(hash.as_str())
    );
    assert_eq!(
        json.get("intent_path").and_then(|v| v.as_str()),
        Some(intent_path.to_string_lossy().as_ref())
    );
    assert_eq!(
        json.get("token").and_then(|v| v.as_str()),
        Some(token.as_str())
    );
}
