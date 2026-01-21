//! Incoming message capture from Chrome/ChatGPT

use mmsb_events::ChromeMessage;
use mmsb_primitives::EventId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::{ChromeConnection, ChromeError};

#[derive(Debug, Serialize, Deserialize)]
pub struct CapturedMessage {
    #[serde(rename = "conversationId")]
    pub conversation_id: String,
    #[serde(rename = "messageId")]
    pub message_id: String,
    pub timestamp: u64,
    #[serde(rename = "responseText")]
    pub response_text: Option<String>,
}

pub struct MessageCapture {
    conn: ChromeConnection,
}

impl MessageCapture {
    pub fn new(conn: ChromeConnection) -> Self {
        Self { conn }
    }
    
    /// Set up response listener in Chrome
    pub fn setup_listener(&mut self) -> Result<(), ChromeError> {
        // Register CDP binding for JS → Rust communication
        self.conn.send_command("Runtime.addBinding", serde_json::json!({
            "name": "__chromiumMessenger"
        }))?;
        
        // Inject response capture script
        let script = include_str!("../scripts/response_capture.js");
        self.conn.send_command("Runtime.evaluate", serde_json::json!({
            "expression": script,
            "awaitPromise": false,
        }))?;
        
        Ok(())
    }
    
    /// Wait for and capture next message
    pub fn capture_next(&mut self) -> Result<Option<ChromeMessage>, ChromeError> {
        if let Some(event) = self.conn.read_event()? {
            if event.get("method").and_then(|m| m.as_str()) == Some("Runtime.bindingCalled") {
                let payload = event["params"]["payload"]
                    .as_str()
                    .ok_or_else(|| ChromeError::Protocol("missing binding payload".into()))?;
                
                let captured: CapturedMessage = serde_json::from_str(payload)
                    .map_err(|e| ChromeError::Protocol(e.to_string()))?;
                
                if let Some(text) = captured.response_text {
                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    
                    return Ok(Some(ChromeMessage {
                        event_id: Self::generate_event_id(&captured.conversation_id, &captured.message_id),
                        timestamp,
                        conversation_id: captured.conversation_id,
                        message_id: captured.message_id,
                        response_text: text,
                    }));
                }
            }
        }
        
        Ok(None)
    }
    
    fn generate_event_id(conversation_id: &str, message_id: &str) -> EventId {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(conversation_id.as_bytes());
        hasher.update(message_id.as_bytes());
        hasher.finalize().into()
    }
}
