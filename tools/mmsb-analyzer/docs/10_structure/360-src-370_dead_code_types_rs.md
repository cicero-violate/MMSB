# Structure Group: src/370_dead_code_types.rs

## File: src/370_dead_code_types.rs

- Layer(s): 370_dead_code_types.rs
- Language coverage: Rust (12)
- Element types: Enum (5), Impl (2), Struct (5)
- Total elements: 12

### Elements

- [Rust | Enum] `ConfidenceLevel` (line 0, pub)
- [Rust | Enum] `DeadCodeCategory` (line 0, pub)
- [Rust | Struct] `DeadCodeItem` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct DeadCodeItem { pub symbol : String , pub file : PathB...`
- [Rust | Struct] `DeadCodeReport` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct DeadCodeReport { pub timestamp : String , pub summary...`
- [Rust | Struct] `DeadCodeSummary` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize , Default)] pub struct DeadCodeSummary { pub unreachable : usize ,...`
- [Rust | Enum] `IntentMarker` (line 0, pub)
- [Rust | Struct] `IntentMetadata` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct IntentMetadata { pub marker : IntentMarker , pub sour...`
- [Rust | Enum] `IntentSource` (line 0, pub)
- [Rust | Struct] `IntentTag` (line 0, pub)
  - Signature: `# [derive (Debug , Clone , Serialize , Deserialize)] pub struct IntentTag { pub symbol : String , pub file : PathBuf ...`
- [Rust | Enum] `RecommendedAction` (line 0, pub)
- [Rust | Impl] `impl DeadCodeSummary { pub fn from_items (items : & [DeadCodeItem]) -> Self { let mut summary = DeadCodeSummary :: default () ; summary . total_analyzed = items . len () ; for item in items { match item . category { DeadCodeCategory :: Unreachable => summary . unreachable += 1 , DeadCodeCategory :: ReachableUnused => summary . reachable_unused += 1 , DeadCodeCategory :: TestOnly => summary . test_only += 1 , DeadCodeCategory :: LatentPlanned => summary . latent_planned += 1 , } } summary } } . self_ty` (line 0, priv)
- [Rust | Impl] `impl IntentMarker { pub fn from_comment (comment : & str) -> Option < Self > { let lower = comment . to_ascii_lowercase () ; if lower . contains ("@deprecated-planned") { return Some (IntentMarker :: DeprecatedPlanned) ; } if lower . contains ("@planned") { return Some (IntentMarker :: Planned) ; } if lower . contains ("@future") { return Some (IntentMarker :: Future) ; } if lower . contains ("@latent") { return Some (IntentMarker :: Latent) ; } None } } . self_ty` (line 0, priv)

