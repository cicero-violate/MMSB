//! Example: Phase 7 Integration
//!
//! Demonstrates full pipeline from code indexing to LLM proposal generation.

use structural_code_indexer::index_repository;
use mmsb_core::adaptive::{LLMProposalEngine, ProposalEngine, ProposalConfig};
use std::path::Path;

fn main() {
    println!("=== Phase 7 Integration Example ===\n");
    
    // Step 1: Index repository
    let repo_path = Path::new("../../mmsb-core");
    println!("Step 1: Indexing repository...");
    let snapshot = index_repository(repo_path);
    println!("  ✓ Snapshot hash: {}", snapshot.snapshot_hash);
    println!("  ✓ Edges: {}", snapshot.dag.edges().len());
    println!("  ✓ Total propagations: {}", snapshot.stats.total_propagations);
    
    // Step 2: Generate heuristic proposals
    println!("\nStep 2: Generating heuristic proposals...");
    let config = ProposalConfig::default();
    let engine = ProposalEngine::new(config);
    let heuristic_proposals = engine.generate_proposals(&snapshot.dag, &snapshot.stats);
    println!("  ✓ Generated {} proposals", heuristic_proposals.len());
    
    for (i, proposal) in heuristic_proposals.iter().enumerate().take(3) {
        println!("\n  Proposal {}:", i + 1);
        println!("    Category: {:?}", proposal.category);
        println!("    Confidence: {:.2}", proposal.confidence);
        println!("    Operations: {}", proposal.ops.len());
        println!("    Rationale: {}", &proposal.rationale[..80.min(proposal.rationale.len())]);
    }
    
    // Step 3: Create LLM request
    println!("\nStep 3: Creating LLM request...");
    let llm_request = LLMProposalEngine::create_request(&snapshot.dag, &snapshot.stats);
    println!("  ✓ DAG summary: {} nodes, {} edges", 
        llm_request.dag_summary.total_nodes,
        llm_request.dag_summary.total_edges
    );
    println!("  ✓ High-fanout pages: {}", llm_request.stats_summary.high_fanout_pages.len());
    println!("  ✓ Zero-fanout pages: {}", llm_request.stats_summary.zero_fanout_pages.len());
    
    // Step 4: Display LLM instruction
    println!("\nStep 4: LLM Instruction Preview:");
    let instruction = LLMProposalEngine::generate_instruction();
    let lines: Vec<&str> = instruction.lines().collect();
    println!("  First 10 lines:");
    for line in lines.iter().take(10) {
        println!("  {}", line);
    }
    println!("  ... ({} total lines)", lines.len());
    
    // Step 5: Simulate LLM response validation
    println!("\nStep 5: Example LLM Response Validation:");
    println!("  (In production, Claude would generate ProposalJSON)");
    println!("  Validator checks:");
    println!("    ✓ Structural safety (acyclicity, no orphans, no self-deps)");
    println!("    ✓ Causal justification (evidence required)");
    println!("    ✓ Evidence requirements (PageIDs, fanout, snapshot hash)");
    println!("    ✓ Minimality rules (max 5 ops)");
    
    // Step 6: Integration summary
    println!("\n=== Integration Pipeline ===");
    println!("  1. Source Code → [Indexer] → DependencyGraph + PropagationStats");
    println!("  2. DAG + Stats → [Heuristic Engine] → Proposals");
    println!("  3. DAG + Stats → [LLM Request] → JSON");
    println!("  4. JSON → [Claude] → ProposalJSON");
    println!("  5. ProposalJSON → [Validator] → Approved/Rejected");
    println!("  6. Approved → [Phase 1 Judgment] → Applied");
    
    println!("\n=== Example Complete ===");
}
