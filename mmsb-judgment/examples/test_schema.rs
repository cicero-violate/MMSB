use mmsb_judgment::{JudgmentArtifact, JudgmentScope};

fn main() {
    // Create judgment artifact
    let artifact = JudgmentArtifact::new(
        "abc123".to_string(),
        "intents/test.json".to_string(),
        "def456".to_string(),
    );
    
    println!("JudgmentArtifact:\n{}\n", artifact.to_json().unwrap());
    
    // Create judgment scope
    let scope = JudgmentScope::new(
        artifact.judgment_id.clone(),
        vec!["formatting".to_string()],
        vec!["src/**/*.rs".to_string()],
        vec![".git/**".to_string()],
        vec!["rustfmt".to_string()],
        vec!["shell_runner".to_string()],
        50,
        2000,
    );
    
    println!("JudgmentScope:\n{}", scope.to_json().unwrap());
}
