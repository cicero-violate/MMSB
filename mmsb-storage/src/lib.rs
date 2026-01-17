//! MMSB Storage
//!
//! Durable persistence layer for MMSB artifacts.
//! ONLY mmsb-memory and mmsb-knowledge may write to storage.

use mmsb_proof::Hash;
use std::path::PathBuf;

/// Storage error types
#[derive(Debug)]
pub enum StorageError {
    IoError(std::io::Error),
    SerializationError(serde_json::Error),
    NotFound,
    PermissionDenied,
}

impl From<std::io::Error> for StorageError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<serde_json::Error> for StorageError {
    fn from(e: serde_json::Error) -> Self {
        Self::SerializationError(e)
    }
}

impl std::fmt::Display for StorageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IoError(e) => write!(f, "IO error: {}", e),
            Self::SerializationError(e) => write!(f, "Serialization error: {}", e),
            Self::NotFound => write!(f, "Item not found"),
            Self::PermissionDenied => write!(f, "Permission denied"),
        }
    }
}

impl std::error::Error for StorageError {}

/// Storage backend trait
pub trait Storage: Send + Sync {
    /// Store bytes at the given key
    fn store(&mut self, key: &Hash, data: &[u8]) -> Result<(), StorageError>;
    
    /// Retrieve bytes for the given key
    fn retrieve(&self, key: &Hash) -> Result<Vec<u8>, StorageError>;
    
    /// Check if key exists
    fn exists(&self, key: &Hash) -> bool;
    
    /// Delete data at key
    fn delete(&mut self, key: &Hash) -> Result<(), StorageError>;
}

/// File-based storage implementation
pub struct FileStorage {
    root_path: PathBuf,
}

impl FileStorage {
    pub fn new(root_path: PathBuf) -> Self {
        Self { root_path }
    }
    
    fn key_to_path(&self, key: &Hash) -> PathBuf {
        let hex = hex::encode(key);
        self.root_path.join(&hex[..2]).join(&hex[2..])
    }
}

impl Storage for FileStorage {
    fn store(&mut self, key: &Hash, data: &[u8]) -> Result<(), StorageError> {
        let path = self.key_to_path(key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, data)?;
        Ok(())
    }
    
    fn retrieve(&self, key: &Hash) -> Result<Vec<u8>, StorageError> {
        let path = self.key_to_path(key);
        if !path.exists() {
            return Err(StorageError::NotFound);
        }
        Ok(std::fs::read(path)?)
    }
    
    fn exists(&self, key: &Hash) -> bool {
        self.key_to_path(key).exists()
    }
    
    fn delete(&mut self, key: &Hash) -> Result<(), StorageError> {
        let path = self.key_to_path(key);
        if !path.exists() {
            return Err(StorageError::NotFound);
        }
        std::fs::remove_file(path)?;
        Ok(())
    }
}

/// Memory-only storage for testing
pub struct MemoryStorage {
    data: std::collections::HashMap<Hash, Vec<u8>>,
}

impl MemoryStorage {
    pub fn new() -> Self {
        Self {
            data: std::collections::HashMap::new(),
        }
    }
}

impl Default for MemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Storage for MemoryStorage {
    fn store(&mut self, key: &Hash, data: &[u8]) -> Result<(), StorageError> {
        self.data.insert(*key, data.to_vec());
        Ok(())
    }
    
    fn retrieve(&self, key: &Hash) -> Result<Vec<u8>, StorageError> {
        self.data.get(key).cloned().ok_or(StorageError::NotFound)
    }
    
    fn exists(&self, key: &Hash) -> bool {
        self.data.contains_key(key)
    }
    
    fn delete(&mut self, key: &Hash) -> Result<(), StorageError> {
        self.data.remove(key).ok_or(StorageError::NotFound)?;
        Ok(())
    }
}
