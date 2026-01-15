use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};

#[derive(Serialize)]
pub struct JudgmentArtifact {
    pub schema: String,
    pub judgment_id: String,
    pub token: String,
    pub intent_path: String,
    pub delta_hash: String,
    pub acknowledged: bool,
    pub issued_at: String,
    pub issuer: String,
    pub version: u32,
}

impl JudgmentArtifact {
    pub fn new(
        judgment_id: String,
        token: String,
        intent_path: String,
        delta_hash: String,
        issued_at: String,
    ) -> Self {
        Self {
            schema: "judgment.v1".to_string(),
            judgment_id,
            token,
            intent_path,
            delta_hash,
            acknowledged: true,
            issued_at,
            issuer: "human".to_string(),
            version: 1,
        }
    }
}

pub fn judgment_path_for_intent(intent_path: &str) -> PathBuf {
    PathBuf::from(format!("{intent_path}.judgment.json"))
}

pub fn write_artifact_atomic(path: &Path, artifact: &JudgmentArtifact) -> io::Result<()> {
    if path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "judgment artifact already exists",
        ));
    }

    let serialized = serde_json::to_string_pretty(artifact)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    let tmp_path = PathBuf::from(format!("{}.tmp", path.display()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp_path)?;
    io::Write::write_all(&mut file, serialized.as_bytes())?;
    file.sync_all()?;
    fs::rename(&tmp_path, path)?;
    Ok(())
}
