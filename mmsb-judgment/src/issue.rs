use crate::types::JudgmentToken;
use rand::RngCore;

pub fn issue_judgment(_intent_metadata: &str, _delta_hash: &str) -> JudgmentToken {
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    JudgmentToken::new(hex_encode(&bytes))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{:02x}", byte);
    }
    out
}
