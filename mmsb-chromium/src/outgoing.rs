//! Outgoing message sending to Chrome/ChatGPT

use crate::{ChromeConnection, ChromeError};
use serde_json::json;

pub struct MessageSender {
    conn: ChromeConnection,
}

impl MessageSender {
    pub fn new(conn: ChromeConnection) -> Self {
        Self { conn }
    }
    
    /// Send a message to ChatGPT via Chrome
    pub fn send_message(&mut self, conversation_id: &str, content: &str) -> Result<(), ChromeError> {
        // Inject message into ChatGPT interface
        let script = format!(
            r#"
            (function() {{
                const textarea = document.querySelector('textarea[data-id="root"]');
                if (textarea) {{
                    textarea.value = {};
                    textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    
                    setTimeout(() => {{
                        const button = document.querySelector('button[data-testid="send-button"]');
                        if (button && !button.disabled) {{
                            button.click();
                        }}
                    }}, 100);
                }}
            }})();
            "#,
            json!(content)
        );
        
        self.conn.send_command("Runtime.evaluate", json!({
            "expression": script,
            "awaitPromise": false,
        }))?;
        
        Ok(())
    }
}
