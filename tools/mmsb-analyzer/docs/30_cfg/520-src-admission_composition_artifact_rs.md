# CFG Group: src/admission_composition_artifact.rs

## Function: `create_test_signature`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    create_test_signature_0["ENTRY"]
    create_test_signature_1["EffectSignature { schema_version : SCHEMA_VERSION . to_string () , action_typ..."]
    create_test_signature_2["EXIT"]
    create_test_signature_0 --> create_test_signature_1
    create_test_signature_1 --> create_test_signature_2
```

## Function: `generate_artifact`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    generate_artifact_0["ENTRY"]
    generate_artifact_1["let timestamp = chrono :: Utc :: now () . to_rfc3339 ()"]
    generate_artifact_2["let (admissible , composition_result) = match result { CompositionResult :: Admissible { action_count , final_state ,..."]
    generate_artifact_3["AdmissionCompositionArtifact { schema_version : ARTIFACT_SCHEMA_VERSION . to_..."]
    generate_artifact_4["EXIT"]
    generate_artifact_0 --> generate_artifact_1
    generate_artifact_1 --> generate_artifact_2
    generate_artifact_2 --> generate_artifact_3
    generate_artifact_3 --> generate_artifact_4
```

## Function: `project_conflict_reason`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    project_conflict_reason_0["ENTRY"]
    project_conflict_reason_1["match reason { ConflictReason :: FileWriteConflict { file , prior_action_inde..."]
    project_conflict_reason_2["EXIT"]
    project_conflict_reason_0 --> project_conflict_reason_1
    project_conflict_reason_1 --> project_conflict_reason_2
```

## Function: `project_invariants_touched`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    project_invariants_touched_0["ENTRY"]
    project_invariants_touched_1["use"]
    project_invariants_touched_2["let mut invariants = Vec :: new ()"]
    project_invariants_touched_3["for (inv_type , indices) in & state . invariants_touched { if ! indices . is_..."]
    project_invariants_touched_4["invariants . sort ()"]
    project_invariants_touched_5["invariants"]
    project_invariants_touched_6["EXIT"]
    project_invariants_touched_0 --> project_invariants_touched_1
    project_invariants_touched_1 --> project_invariants_touched_2
    project_invariants_touched_2 --> project_invariants_touched_3
    project_invariants_touched_3 --> project_invariants_touched_4
    project_invariants_touched_4 --> project_invariants_touched_5
    project_invariants_touched_5 --> project_invariants_touched_6
```

## Function: `project_state`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    project_state_0["ENTRY"]
    project_state_1["StateProjection { action_count : state . action_count , files_written_count :..."]
    project_state_2["EXIT"]
    project_state_0 --> project_state_1
    project_state_1 --> project_state_2
```

## Function: `read_artifact`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    read_artifact_0["ENTRY"]
    read_artifact_1["let json = std :: fs :: read_to_string (path) ?"]
    read_artifact_2["let artifact = serde_json :: from_str (& json) ?"]
    read_artifact_3["Ok (artifact)"]
    read_artifact_4["EXIT"]
    read_artifact_0 --> read_artifact_1
    read_artifact_1 --> read_artifact_2
    read_artifact_2 --> read_artifact_3
    read_artifact_3 --> read_artifact_4
```

## Function: `test_admissible_batch_artifact`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 10
- Edges: 9

```mermaid
flowchart TD
    test_admissible_batch_artifact_0["ENTRY"]
    test_admissible_batch_artifact_1["let batch = vec ! [create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')..."]
    test_admissible_batch_artifact_2["let result = compose_batch (& batch)"]
    test_admissible_batch_artifact_3["let artifact = generate_artifact (& batch , & result)"]
    test_admissible_batch_artifact_4["macro assert_eq"]
    test_admissible_batch_artifact_5["macro assert_eq"]
    test_admissible_batch_artifact_6["macro assert"]
    test_admissible_batch_artifact_7["macro assert_eq"]
    test_admissible_batch_artifact_8["match artifact . composition_result { CompositionResultProjection :: Admissib..."]
    test_admissible_batch_artifact_9["EXIT"]
    test_admissible_batch_artifact_0 --> test_admissible_batch_artifact_1
    test_admissible_batch_artifact_1 --> test_admissible_batch_artifact_2
    test_admissible_batch_artifact_2 --> test_admissible_batch_artifact_3
    test_admissible_batch_artifact_3 --> test_admissible_batch_artifact_4
    test_admissible_batch_artifact_4 --> test_admissible_batch_artifact_5
    test_admissible_batch_artifact_5 --> test_admissible_batch_artifact_6
    test_admissible_batch_artifact_6 --> test_admissible_batch_artifact_7
    test_admissible_batch_artifact_7 --> test_admissible_batch_artifact_8
    test_admissible_batch_artifact_8 --> test_admissible_batch_artifact_9
```

## Function: `test_determinism`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 13
- Edges: 12

```mermaid
flowchart TD
    test_determinism_0["ENTRY"]
    test_determinism_1["let batch = vec ! [create_test_signature ('test_action' , vec ! [PathBuf :: from ('test.r..."]
    test_determinism_2["let result = compose_batch (& batch)"]
    test_determinism_3["let artifact1 = generate_artifact (& batch , & result)"]
    test_determinism_4["let artifact2 = generate_artifact (& batch , & result)"]
    test_determinism_5["let mut artifact1_normalized = artifact1 . clone ()"]
    test_determinism_6["let mut artifact2_normalized = artifact2 . clone ()"]
    test_determinism_7["artifact1_normalized . timestamp = 'NORMALIZED' . to_string ()"]
    test_determinism_8["artifact2_normalized . timestamp = 'NORMALIZED' . to_string ()"]
    test_determinism_9["let json1 = serde_json :: to_string_pretty (& artifact1_normalized) . unwrap ()"]
    test_determinism_10["let json2 = serde_json :: to_string_pretty (& artifact2_normalized) . unwrap ()"]
    test_determinism_11["macro assert_eq"]
    test_determinism_12["EXIT"]
    test_determinism_0 --> test_determinism_1
    test_determinism_1 --> test_determinism_2
    test_determinism_2 --> test_determinism_3
    test_determinism_3 --> test_determinism_4
    test_determinism_4 --> test_determinism_5
    test_determinism_5 --> test_determinism_6
    test_determinism_6 --> test_determinism_7
    test_determinism_7 --> test_determinism_8
    test_determinism_8 --> test_determinism_9
    test_determinism_9 --> test_determinism_10
    test_determinism_10 --> test_determinism_11
    test_determinism_11 --> test_determinism_12
```

## Function: `test_inadmissible_batch_artifact`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 8
- Edges: 7

```mermaid
flowchart TD
    test_inadmissible_batch_artifact_0["ENTRY"]
    test_inadmissible_batch_artifact_1["let batch = vec ! [create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')..."]
    test_inadmissible_batch_artifact_2["let result = compose_batch (& batch)"]
    test_inadmissible_batch_artifact_3["let artifact = generate_artifact (& batch , & result)"]
    test_inadmissible_batch_artifact_4["macro assert_eq"]
    test_inadmissible_batch_artifact_5["macro assert"]
    test_inadmissible_batch_artifact_6["match artifact . composition_result { CompositionResultProjection :: Inadmiss..."]
    test_inadmissible_batch_artifact_7["EXIT"]
    test_inadmissible_batch_artifact_0 --> test_inadmissible_batch_artifact_1
    test_inadmissible_batch_artifact_1 --> test_inadmissible_batch_artifact_2
    test_inadmissible_batch_artifact_2 --> test_inadmissible_batch_artifact_3
    test_inadmissible_batch_artifact_3 --> test_inadmissible_batch_artifact_4
    test_inadmissible_batch_artifact_4 --> test_inadmissible_batch_artifact_5
    test_inadmissible_batch_artifact_5 --> test_inadmissible_batch_artifact_6
    test_inadmissible_batch_artifact_6 --> test_inadmissible_batch_artifact_7
```

## Function: `write_artifact`

- File: src/admission_composition_artifact.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    write_artifact_0["ENTRY"]
    write_artifact_1["let json = serde_json :: to_string_pretty (artifact) ?"]
    write_artifact_2["std :: fs :: write (path , json) ?"]
    write_artifact_3["Ok (())"]
    write_artifact_4["EXIT"]
    write_artifact_0 --> write_artifact_1
    write_artifact_1 --> write_artifact_2
    write_artifact_2 --> write_artifact_3
    write_artifact_3 --> write_artifact_4
```

