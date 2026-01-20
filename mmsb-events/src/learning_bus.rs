//! LearningBus - Advisory derivation

use mmsb_proof::*;

pub trait LearningBus {
    fn observe_outcome(&mut self, commit_proof: CommitProof) -> OutcomeProof;
    fn derive_knowledge(&mut self, outcome_proof: OutcomeProof) -> KnowledgeProof;
    fn report_knowledge(&mut self, knowledge_proof: KnowledgeProof);
}
