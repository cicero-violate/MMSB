# Structure Group: src/350_agent_cli.rs

## File: src/350_agent_cli.rs

- Layer(s): 350_agent_cli.rs
- Language coverage: Rust (9)
- Element types: Enum (1), Function (6), Module (1), Struct (1)
- Total elements: 9

### Elements

- [Rust | Struct] `AgentCli` (line 0, pub)
  - Signature: `# [derive (Parser , Debug)] # [command (name = "mmsb-agent")] # [command (about = "Agent interface to MMSB conscience...`
- [Rust | Enum] `AgentCommand` (line 0, pub)
- [Rust | Function] `check_action` (line 0, priv)
  - Signature: `# [doc = " Check if an action is allowed"] fn check_action (action_path : & PathBuf , conscience_path : & PathBuf) ->...`
  - Calls: std::fs::read_to_string, serde_json::from_str, load_invariants, AgentConscience::new, check_action, serde_json::to_string_pretty, std::process::exit
- [Rust | Function] `list_invariants` (line 0, priv)
  - Signature: `# [doc = " List all invariants"] fn list_invariants (conscience_path : & PathBuf , blocking_only : bool) -> Result < ...`
  - Calls: load_invariants, collect, cloned, filter, iter, is_blocking, serde_json::to_string_pretty, Ok
- [Rust | Function] `load_invariants` (line 0, priv)
  - Signature: `# [doc = " Load invariants from JSON file"] fn load_invariants (path : & PathBuf) -> Result < Vec < Invariant > > { l...`
  - Calls: std::fs::read_to_string, Ok, serde_json::from_str
- [Rust | Function] `query_function` (line 0, priv)
  - Signature: `# [doc = " Query allowed actions for a function"] fn query_function (function : & str , conscience_path : & PathBuf) ...`
  - Calls: load_invariants, AgentConscience::new, query_allowed_actions, serde_json::to_string_pretty, Ok
- [Rust | Function] `run_agent_cli` (line 0, pub)
  - Signature: `pub fn run_agent_cli () -> Result < () > { let cli = AgentCli :: parse () ; match cli . command { AgentCommand :: Che...`
  - Calls: AgentCli::parse, check_action, query_function, list_invariants, show_stats, Ok
- [Rust | Function] `show_stats` (line 0, priv)
  - Signature: `# [doc = " Show conscience statistics"] fn show_stats (conscience_path : & PathBuf) -> Result < () > { let invariants...`
  - Calls: load_invariants, AgentConscience::new, stats, Ok
- [Rust | Module] `tests` (line 0, priv)

