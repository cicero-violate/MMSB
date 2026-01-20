use mmsb_policy::{PolicyModule, PolicyConfig};

fn main() {
    let path = "/home/cicero-arch-omen/ai_sandbox/codex-agent/codex_sse/server-tools/MMSB/mmsb-policy/schema/example_policy.json";
    
    match PolicyModule::load_from_file(path) {
        Ok(_) => println!("✓ Loaded policy from schema"),
        Err(e) => println!("✗ Failed: {}", e),
    }
    
    let config = PolicyConfig::load_file(path).unwrap();
    println!("\nPath validation:");
    println!("  src/main.rs: {}", config.matches_path("src/main.rs"));
    println!("  .git/config: {}", config.matches_path(".git/config"));
    println!("  migrations/001.sql: {}", config.matches_path("migrations/001.sql"));
}
