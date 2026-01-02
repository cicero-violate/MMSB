# Structure Group: src/batch_admission.rs

## File: src/batch_admission.rs

- Layer(s): root
- Language coverage: Rust (3)
- Element types: Enum (1), Impl (1), Module (1)
- Total elements: 3

### Elements

- [Rust | Enum] `AdmissionDecision` (line 0, pub)
- [Rust | Impl] `impl AdmissionDecision { # [doc = " Check if batch was admissible"] pub fn is_admissible (& self) -> bool { matches ! (self , AdmissionDecision :: Admissible { .. }) } # [doc = " Get artifact path (always present)"] pub fn artifact_path (& self) -> & Path { match self { AdmissionDecision :: Admissible { artifact_path } => artifact_path , AdmissionDecision :: Inadmissible { artifact_path } => artifact_path , } } } . self_ty` (line 0, priv)
- [Rust | Module] `tests` (line 0, priv)

