//! Agent CLI Interface
//!
//! CLI for agents to query conscience and validate actions.

use crate::action_validator::AgentAction;
use crate::agent_conscience::AgentConscience;
use crate::invariant_types::Invariant;
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "mmsb-agent")]
#[command(about = "Agent interface to MMSB conscience", long_about = None)]
pub struct AgentCli {
    #[command(subcommand)]
    pub command: AgentCommand,
}

#[derive(Parser, Debug)]
pub enum AgentCommand {
    /// Check if action is allowed
    Check {
        /// JSON file with action proposal
        #[arg(short, long)]
        action: PathBuf,

        /// Conscience file (invariants.json)
        #[arg(short, long)]
        conscience: PathBuf,
    },

    /// Query allowed actions for function
    Query {
        /// Function name
        #[arg(short, long)]
        function: String,

        /// Conscience file
        #[arg(short, long)]
        conscience: PathBuf,
    },

    /// List all blocking invariants
    Invariants {
        /// Conscience file
        #[arg(short, long)]
        conscience: PathBuf,

        /// Show only blocking invariants
        #[arg(short, long)]
        blocking_only: bool,
    },

    /// Show conscience statistics
    Stats {
        /// Conscience file
        #[arg(short, long)]
        conscience: PathBuf,
    },
}

pub fn run_agent_cli() -> Result<()> {
    let cli = AgentCli::parse();

    match cli.command {
        AgentCommand::Check { action, conscience } => {
            check_action(&action, &conscience)?;
        }

        AgentCommand::Query {
            function,
            conscience,
        } => {
            query_function(&function, &conscience)?;
        }

        AgentCommand::Invariants {
            conscience,
            blocking_only,
        } => {
            list_invariants(&conscience, blocking_only)?;
        }

        AgentCommand::Stats { conscience } => {
            show_stats(&conscience)?;
        }
    }

    Ok(())
}

/// Check if an action is allowed
fn check_action(action_path: &PathBuf, conscience_path: &PathBuf) -> Result<()> {
    // Load action
    let action_json = std::fs::read_to_string(action_path)?;
    let action: AgentAction = serde_json::from_str(&action_json)?;

    // Load conscience
    let invariants = load_invariants(conscience_path)?;
    let conscience = AgentConscience::new(invariants);

    // Check action
    let result = conscience.check_action(&action);

    // Output JSON result
    let output = serde_json::to_string_pretty(&result)?;
    println!("{}", output);

    // Exit code: 0=allowed, 1=blocked
    std::process::exit(if result.allowed { 0 } else { 1 });
}

/// Query allowed actions for a function
fn query_function(function: &str, conscience_path: &PathBuf) -> Result<()> {
    let invariants = load_invariants(conscience_path)?;
    let conscience = AgentConscience::new(invariants);

    let allowed = conscience.query_allowed_actions(function);

    let output = serde_json::to_string_pretty(&allowed)?;
    println!("{}", output);

    Ok(())
}

/// List all invariants
fn list_invariants(conscience_path: &PathBuf, blocking_only: bool) -> Result<()> {
    let invariants = load_invariants(conscience_path)?;

    let filtered: Vec<_> = if blocking_only {
        invariants
            .iter()
            .filter(|i| i.is_blocking())
            .cloned()
            .collect()
    } else {
        invariants
    };

    println!("Total invariants: {}", filtered.len());

    if blocking_only {
        println!("(Showing only blocking invariants)\n");
    }

    let output = serde_json::to_string_pretty(&filtered)?;
    println!("{}", output);

    Ok(())
}

/// Show conscience statistics
fn show_stats(conscience_path: &PathBuf) -> Result<()> {
    let invariants = load_invariants(conscience_path)?;
    let conscience = AgentConscience::new(invariants);

    let stats = conscience.stats();

    println!("Conscience Statistics");
    println!("====================\n");
    println!("Total invariants:    {}", stats.total_invariants);
    println!("Blocking invariants: {}", stats.blocking_invariants);
    println!("Total constraints:   {}", stats.total_constraints);
    println!();
    println!("By strength:");
    println!("  Proven:     {}", stats.proven_count);
    println!("  Empirical:  {}", stats.empirical_count);
    println!("  Heuristic:  {}", stats.heuristic_count);

    Ok(())
}

/// Load invariants from JSON file
fn load_invariants(path: &PathBuf) -> Result<Vec<Invariant>> {
    let json = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&json)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_invariants_empty() {
        // Write test JSON to a temp location
        let temp_path = std::env::temp_dir().join("test_invariants.json");
        std::fs::write(&temp_path, "[]").unwrap();

        let result = load_invariants(&temp_path);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 0);

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }
}
