//! MMSB Memory Engine - Truth Authority
//!
//! ## Authority
//! SOLE truth authority in MMSB - the only component that can establish canonical facts.
//!
//! ## Proof Production Chain
//! D (AdmissionProof) → E (CommitProof) → F (OutcomeProof)
//!
//! ## Semantic Boundaries
//! - OWNS: truth semantics, invariants, deterministic replay, proof production
//! - NOT: hardware, scheduling, performance, runtime
//!
//! See SEMANTIC_CONTRACT.md for full specification.

use mmsb_events::{EventSink, ExecutionRequested, MemoryCommitted};
use mmsb_proof::{AdmissionProof, AdmissionStage, CommitProof, CommitStage, Hash, OutcomeProof, OutcomeStage, Proof, ProduceProof};

use crate::admission::admission_gate::AdmissionGate;
use crate::proofs::{
    admission_proof::AdmissionProofBuilder, commit_proof::CommitProofBuilder,
    outcome_proof::OutcomeProofBuilder,
};
use crate::truth::canonical_time::CanonicalTime;

/// Input for AdmissionProof
pub struct AdmissionInput {
    pub judgment_proof_hash: Hash,
    pub epoch: u64,
}

/// Input for CommitProof
pub struct CommitInput {
    pub admission_proof_hash: Hash,
    pub delta_hash: Hash,
}

/// Input for OutcomeProof
pub struct OutcomeInput {
    pub commit_proof_hash: Hash,
    pub success: bool,
    pub error: Option<String>,
}

/// MemoryEngine - Truth Authority
///
/// SOLE authority over committing mutations and establishing canonical truth.
/// Produces the proof chain D → E → F in strict order:
/// AdmissionProof (D) → CommitProof (E) → OutcomeProof (F).
pub struct MemoryEngine<S: EventSink> {
    sink: Option<S>,
    time: CanonicalTime,
    admission_gate: AdmissionGate,
}

impl<S: EventSink> MemoryEngine<S> {
    pub fn new() -> Self {
        Self {
            sink: None,
            time: CanonicalTime::new(),
            admission_gate: AdmissionGate::new(),
        }
    }

    pub fn with_sink(mut self, sink: S) -> Self {
        self.sink = Some(sink);
        self
    }
}

impl<S: EventSink> Default for MemoryEngine<S> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement AdmissionStage
impl<S: EventSink> ProduceProof for MemoryEngine<S> {
    type Input = AdmissionInput;
    type Proof = AdmissionProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        AdmissionProofBuilder::new(input.judgment_proof_hash, input.epoch)
    }
}

impl<S: EventSink> AdmissionStage for MemoryEngine<S> {}

// Simplified commit/outcome implementations
pub struct CommitStageImpl;

impl ProduceProof for CommitStageImpl {
    type Input = CommitInput;
    type Proof = CommitProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        CommitProofBuilder::new(input.admission_proof_hash, input.delta_hash)
    }
}

impl CommitStage for CommitStageImpl {}

pub struct OutcomeStageImpl;

impl ProduceProof for OutcomeStageImpl {
    type Input = OutcomeInput;
    type Proof = OutcomeProof;

    fn produce_proof(input: &Self::Input) -> Self::Proof {
        if input.success {
            OutcomeProofBuilder::new(input.commit_proof_hash, true)
        } else {
            OutcomeProofBuilder::with_error(
                input.commit_proof_hash,
                input.error.clone().unwrap_or_default(),
            )
        }
    }
}

impl OutcomeStage for OutcomeStageImpl {}

impl<S: EventSink> MemoryEngine<S> {
    pub fn handle_execution_requested(&mut self, event: ExecutionRequested) {
        // TODO(Phase B): restore full commit logic
        match self.admission_gate.admit(&event.judgment_proof) {
            Ok(()) => {
                let admission_proof = Self::produce_proof(&AdmissionInput {
                    judgment_proof_hash: event.judgment_proof.hash(),
                    epoch: self.time.epoch(),
                });
                
                // Stub: full logic in Phase B
                if let Some(sink) = &self.sink {
                    let memory_event = MemoryCommitted {
                        event_id: event.event_id,
                        timestamp: self.time.next(),
                        admission_proof,
                        commit_proof: CommitProofBuilder::new([0; 32], [0; 32]),
                        outcome_proof: OutcomeProofBuilder::new([0; 32], true),
                    };
                    sink.emit(mmsb_events::AnyEvent::MemoryCommitted(memory_event));
                }
            }
            Err(_) => { /* Admission failed */ }
        }
    }
}
