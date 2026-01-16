use crate::intent::EditIntent;
use crate::error::EditorError;
use mmsb_core::prelude::Delta;
use mmsb_core::dag::{StructuralOp, EdgeType};
use mmsb_core::types::{DeltaID, Epoch, PageID, Source};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Classify edits into structural vs state changes
pub struct StructuralClassifier;

impl StructuralClassifier {
    /// Convert buffer edits to MMSB deltas and structural ops
    pub fn classify(
        intents: &[EditIntent],
        page_id: PageID,
        file_path: &PathBuf,
        source_after: &str,
    ) -> Result<(Vec<Delta>, Vec<StructuralOp>), EditorError> {
        let mut page_deltas = Vec::new();
        let mut structural_ops = Vec::new();
        
        // Build page delta from new source
        let delta = Self::build_page_delta(page_id, source_after);
        page_deltas.push(delta);
        
        // Extract structural ops from intents
        for intent in intents {
            if let Some(op) = Self::intent_to_structural_op(intent, page_id, file_path) {
                structural_ops.push(op);
            }
        }
        
        Ok((page_deltas, structural_ops))
    }
    
    /// Build a Delta from final source
    fn build_page_delta(page_id: PageID, source: &str) -> Delta {
        let payload = source.as_bytes().to_vec();
        let mask = vec![true; payload.len()];
        let delta_id = DeltaID(hash_u64(&page_id.0.to_le_bytes(), &payload));
        
        Delta {
            delta_id,
            page_id,
            epoch: Epoch(0),
            mask,
            payload,
            is_sparse: false,
            timestamp: 0,
            source: Source("declarative_editor".to_string()),
            intent_metadata: None,
        }
    }
    
    /// Convert EditIntent to StructuralOp (if structural)
    fn intent_to_structural_op(
        intent: &EditIntent,
        page_id: PageID,
        _file_path: &PathBuf,
    ) -> Option<StructuralOp> {
        use crate::intent::IntentCategory;
        
        match intent.category() {
            IntentCategory::Structural | IntentCategory::Both => {
                match intent {
                    EditIntent::ImportChange { path, added } => {
                        // Convert import path to target PageID
                        // Simplified: hash the path to PageID
                        let target_id = PageID(hash_u64(path.as_bytes(), &[]));
                        
                        if *added {
                            Some(StructuralOp::AddEdge {
                                from: page_id,
                                to: target_id,
                                edge_type: EdgeType::Data,
                            })
                        } else {
                            Some(StructuralOp::RemoveEdge {
                                from: page_id,
                                to: target_id,
                            })
                        }
                    }
                    EditIntent::ModuleChange { .. } => {
                        // Module changes might add/remove nodes
                        None // TODO: implement module node ops
                    }
                    _ => None,
                }
            }
            IntentCategory::State => None,
        }
    }
}

fn hash_u64(prefix: &[u8], payload: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(prefix);
    hasher.update(payload);
    let hash = hasher.finalize();
    u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3],
        hash[4], hash[5], hash[6], hash[7],
    ])
}
