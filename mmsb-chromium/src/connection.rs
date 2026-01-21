//! Chrome DevTools Protocol Connection
//!
//! Manages WebSocket/Unix socket connection to Chrome DevTools

use serde_json::Value;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChromeError {
    #[error("Connection error: {0}")]
    Connection(String),
    
    #[error("Protocol error: {0}")]
    Protocol(String),
    
    #[error("Timeout")]
    Timeout,
}

pub struct ChromeConnection {
    stream: TcpStream,
    next_id: u64,
}

impl ChromeConnection {
    pub fn new(host: &str, port: u16) -> Result<Self, ChromeError> {
        let stream = TcpStream::connect((host, port))
            .map_err(|e| ChromeError::Connection(e.to_string()))?;
        
        stream.set_read_timeout(Some(Duration::from_millis(100)))
            .map_err(|e| ChromeError::Connection(e.to_string()))?;
        
        Ok(Self {
            stream,
            next_id: 1,
        })
    }
    
    pub fn send_command(&mut self, method: &str, params: Value) -> Result<Value, ChromeError> {
        let id = self.next_id;
        self.next_id += 1;
        
        let command = serde_json::json!({
            "id": id,
            "method": method,
            "params": params,
        });
        
        let msg = serde_json::to_string(&command)
            .map_err(|e| ChromeError::Protocol(e.to_string()))?;
        
        self.stream.write_all(msg.as_bytes())
            .map_err(|e| ChromeError::Connection(e.to_string()))?;
        
        // Read response
        self.read_response(id)
    }
    
    pub fn read_event(&mut self) -> Result<Option<Value>, ChromeError> {
        let mut buffer = vec![0u8; 8192];
        
        match self.stream.read(&mut buffer) {
            Ok(0) => Ok(None),
            Ok(n) => {
                let data = &buffer[..n];
                serde_json::from_slice(data)
                    .map(Some)
                    .map_err(|e| ChromeError::Protocol(e.to_string()))
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(ChromeError::Connection(e.to_string())),
        }
    }
    
    fn read_response(&mut self, expected_id: u64) -> Result<Value, ChromeError> {
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(5);
        
        while start.elapsed() < timeout {
            if let Some(event) = self.read_event()? {
                if event.get("id").and_then(|v| v.as_u64()) == Some(expected_id) {
                    return Ok(event);
                }
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        
        Err(ChromeError::Timeout)
    }
}
