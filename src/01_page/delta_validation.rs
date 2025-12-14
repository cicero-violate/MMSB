use crate::page::{Delta, DeltaError};

/// Validate structural consistency of a delta before application.
pub fn validate_delta(delta: &Delta) -> Result<(), DeltaError> {
    if delta.mask.len() != delta.payload.len() {
        return Err(DeltaError::SizeMismatch {
            mask_len: delta.mask.len(),
            payload_len: delta.payload.len(),
        });
    }

    Ok(())
}
