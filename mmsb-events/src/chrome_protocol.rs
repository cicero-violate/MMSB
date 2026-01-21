//! Chrome Protocol Bus - CDP (Chrome DevTools Protocol) integration
//!
//! This bus handles communication with Chrome/Chromium via CDP.
//! It receives LLM responses and emits structured intents.

use mmsb_primitives::{EventId, Timestamp};
use serde::{Deserialize, Serialize};

/// Chrome message captured from ChatGPT/LLM interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeMessage {
    pub event_id: EventId,
    pub timestamp: Timestamp,
    pub tab_id: String,
    pub website: String,
    pub conversation_id: String,
    pub message_id: String,
    pub response_text: String,
}

/// Chrome tab information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeTab {
    pub tab_id: String,
    pub url: String,
    pub title: String,
}

/// Chrome protocol bus trait
pub trait ChromeProtocolBus: Send + Sync {
    /// Capture a message from Chrome via CDP
    fn capture_message(&mut self, message: ChromeMessage);
    
    /// Send a message to Chrome via CDP  
    fn send_to_chrome(
        &mut self,
        tab_id: &str,
        conversation_id: &str,
        content: &str,
    ) -> Result<(), String>;
    
    /// List active Chrome tabs
    fn list_tabs(&self) -> Vec<ChromeTab>;
    
    /// Get current active tab
    fn get_active_tab(&self) -> Option<ChromeTab>;
}
