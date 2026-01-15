use std::env;
use std::fs;
use std::io::{self, Write};
use std::process;

use chrono::{SecondsFormat, Utc};
use mmsb_judgment::artifact::{judgment_path_for_intent, write_artifact_atomic, JudgmentArtifact};
use mmsb_judgment::issue::issue_judgment;
use sha2::{Digest, Sha256};
use uuid::Uuid;

// Forbidden: --yes, --force, env-based confirmation, non-interactive mode, batch issuance.
fn print_usage() {
    eprintln!("Usage:\n  mmsb-judgment issue --intent PATH [--judgment-id UUID]\n");
}

fn main() {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_usage();
        process::exit(2);
    };

    if command != "issue" {
        print_usage();
        process::exit(2);
    }

    let mut intent_path = None;
    let mut judgment_id_override = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--intent" => intent_path = args.next(),
            "--judgment-id" => judgment_id_override = args.next(),
            "--help" | "-h" => {
                print_usage();
                return;
            }
            _ => {
                eprintln!("Unknown argument: {arg}");
                print_usage();
                process::exit(2);
            }
        }
    }

    let Some(intent_path) = intent_path else {
        eprintln!("Missing --intent PATH");
        print_usage();
        process::exit(2);
    };

    let delta_bytes = match fs::read(&intent_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("Failed to read intent {intent_path}: {err}");
            process::exit(1);
        }
    };
    let delta_size = delta_bytes.len();
    let delta_hash = format!("{:x}", Sha256::digest(&delta_bytes));

    eprintln!("========================================");
    eprintln!("Judgment Ritual");
    eprintln!("========================================");
    eprintln!("intent.path={intent_path}");
    eprintln!("intent.size={delta_size}");
    eprintln!("intent.hash={delta_hash}");
    eprintln!("----------------------------------------");
    eprint!("{}", String::from_utf8_lossy(&delta_bytes));

    eprintln!();
    eprintln!("THIS ACTION IS IRREVERSIBLE.");
    eprintln!("By authorizing this judgment, you accept full responsibility");
    eprintln!("for the resulting state transition.");
    eprintln!("This authorization cannot be reused.");
    eprintln!("This authorization cannot be automated.");
    eprintln!();
    eprintln!("Press Enter to acknowledge and continue.");
    let _ = io::stdin().read_line(&mut String::new());

    eprint!("Type the full intent hash to authorize:\n> ");
    let _ = io::stderr().flush();
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        eprintln!("Failed to read confirmation input");
        process::exit(1);
    }
    let input = input.trim();
    if input != delta_hash {
        eprintln!("Confirmation failed; hash mismatch");
        process::exit(1);
    }

    eprintln!();
    eprintln!("Choose one:");
    eprintln!("  1) AUTHORIZE & EXECUTE");
    eprintln!("  2) DO NOT AUTHORIZE & DO NOT EXECUTE");
    eprint!("> ");
    let _ = io::stderr().flush();
    let mut decision = String::new();
    if io::stdin().read_line(&mut decision).is_err() {
        eprintln!("Failed to read authorization choice");
        process::exit(1);
    }
    match decision.trim() {
        "1" | "AUTHORIZE & EXECUTE" => {}
        "2" | "DO NOT AUTHORIZE & DO NOT EXECUTE" => {
            eprintln!("Judgment declined");
            process::exit(1);
        }
        _ => {
            eprintln!("Invalid choice");
            process::exit(1);
        }
    }

    let token = issue_judgment("manual judgment issuance", &delta_hash);
    let issued_at = Utc::now().to_rfc3339_opts(SecondsFormat::Secs, true);
    let judgment_id = match judgment_id_override {
        Some(value) => match Uuid::parse_str(&value) {
            Ok(parsed) => parsed.to_string(),
            Err(_) => {
                eprintln!("Invalid --judgment-id UUID");
                process::exit(2);
            }
        },
        None => Uuid::new_v4().to_string(),
    };
    let artifact = JudgmentArtifact::new(
        judgment_id,
        token.token().to_string(),
        intent_path.clone(),
        delta_hash,
        issued_at,
    );
    let judgment_path = judgment_path_for_intent(&intent_path);
    if let Err(err) = write_artifact_atomic(&judgment_path, &artifact) {
        eprintln!(
            "Failed to write judgment artifact {}: {err}",
            judgment_path.display()
        );
        process::exit(1);
    }

    println!("{}", token.token());
}
