use crate::propagation::index::PageIndex;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditIntent {
    RenameSymbol { old: String, new: String },
    DeleteSymbol { name: String },
    AddSymbol { name: String },
    SignatureChange { name: String },
}

pub fn extract_intent(before: &PageIndex, after: &PageIndex) -> Vec<EditIntent> {
    let removed: Vec<String> = before
        .exports
        .difference(&after.exports)
        .cloned()
        .collect();
    let added: Vec<String> = after
        .exports
        .difference(&before.exports)
        .cloned()
        .collect();

    if removed.len() == 1 && added.len() == 1 {
        return vec![EditIntent::RenameSymbol {
            old: removed[0].clone(),
            new: added[0].clone(),
        }];
    }

    let mut intents = Vec::new();
    for name in removed {
        intents.push(EditIntent::DeleteSymbol { name });
    }
    for name in added {
        intents.push(EditIntent::AddSymbol { name });
    }
    intents
}
