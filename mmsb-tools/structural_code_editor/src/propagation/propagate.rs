use crate::propagation::index::PageIndex;
use crate::propagation::intent::EditIntent;
use crate::propagation::rewrite::rewrite_page;
use mmsb_core::dag::DependencyGraph;
use mmsb_core::prelude::Delta;
use mmsb_core::types::{DeltaID, Epoch, PageID, Source};
use mmsb_judgment::JudgmentToken;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone)]
pub struct PropagatedDelta {
    pub page_id: PageID,
    pub delta: Delta,
    pub reason: String,
}

pub fn propagate_edits(
    root_page: PageID,
    intents: &[EditIntent],
    graph: &DependencyGraph,
    index_store: &HashMap<PageID, PageIndex>,
    source_store: &HashMap<PageID, String>,
    judgment: &JudgmentToken,
) -> Vec<PropagatedDelta> {
    let mut results = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut epoch = 0u32;

    queue.push_back(root_page);
    visited.insert(root_page);

    while let Some(current) = queue.pop_front() {
        let Some(edges) = graph.get_adjacency().get(&current) else {
            continue;
        };

        for (neighbor, _) in edges {
            if visited.insert(*neighbor) {
                queue.push_back(*neighbor);
            }

            let Some(index) = index_store.get(neighbor) else {
                continue;
            };
            let Some(src) = source_store.get(neighbor) else {
                continue;
            };

            if !matches_intents(index, intents) {
                continue;
            }

            if let Some(new_src) = rewrite_page(src, intents) {
                let reason = format!("propagated from {}", root_page.0);
                let delta = build_delta(*neighbor, &new_src, epoch, &reason);
                epoch = epoch.wrapping_add(1);
                let _ = judgment;
                results.push(PropagatedDelta {
                    page_id: *neighbor,
                    delta,
                    reason,
                });
            }
        }
    }

    results
}

fn matches_intents(index: &PageIndex, intents: &[EditIntent]) -> bool {
    intents.iter().any(|intent| match intent {
        EditIntent::RenameSymbol { old, .. } => {
            index.imports.contains(old) || index.references.contains(old)
        }
        EditIntent::DeleteSymbol { name } => {
            index.imports.contains(name) || index.references.contains(name)
        }
        EditIntent::AddSymbol { .. } => false,
        EditIntent::SignatureChange { name } => {
            index.imports.contains(name) || index.references.contains(name)
        }
    })
}

fn build_delta(page_id: PageID, src: &str, epoch: u32, reason: &str) -> Delta {
    let payload = src.as_bytes().to_vec();
    let mask = vec![true; payload.len()];
    let delta_id = DeltaID(hash_u64(&page_id.0.to_le_bytes(), &payload));

    Delta {
        delta_id,
        page_id,
        epoch: Epoch(epoch),
        mask,
        payload,
        is_sparse: false,
        timestamp: epoch as u64,
        source: Source(reason.to_string()),
        intent_metadata: None,
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
