# CFG Group: src/batch_admission.rs

## Function: `admit_batch`

- File: src/batch_admission.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    admit_batch_0["ENTRY"]
    admit_batch_1["let result = compose_batch (batch)"]
    admit_batch_2["let artifact = generate_artifact (batch , & result)"]
    admit_batch_3["write_artifact (& artifact , artifact_path) ?"]
    admit_batch_4["let decision = if artifact . admissible { AdmissionDecision :: Admissible { artifact_path : ..."]
    admit_batch_5["Ok (decision)"]
    admit_batch_6["EXIT"]
    admit_batch_0 --> admit_batch_1
    admit_batch_1 --> admit_batch_2
    admit_batch_2 --> admit_batch_3
    admit_batch_3 --> admit_batch_4
    admit_batch_4 --> admit_batch_5
    admit_batch_5 --> admit_batch_6
```

## Function: `create_test_signature`

- File: src/batch_admission.rs
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

## Function: `test_admissible_batch`

- File: src/batch_admission.rs
- Branches: 0
- Loops: 0
- Nodes: 13
- Edges: 12

```mermaid
flowchart TD
    test_admissible_batch_0["ENTRY"]
    test_admissible_batch_1["let temp_dir = TempDir :: new () . unwrap ()"]
    test_admissible_batch_2["let artifact_path = temp_dir . path () . join ('admission_composition.json')"]
    test_admissible_batch_3["let batch = vec ! [create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')..."]
    test_admissible_batch_4["let decision = admit_batch (& batch , & artifact_path) . unwrap ()"]
    test_admissible_batch_5["macro assert"]
    test_admissible_batch_6["macro assert_eq"]
    test_admissible_batch_7["macro assert"]
    test_admissible_batch_8["let artifact_json = std :: fs :: read_to_string (& artifact_path) . unwrap ()"]
    test_admissible_batch_9["let artifact : serde_json :: Value = serde_json :: from_str (& artifact_json) . unwrap ()"]
    test_admissible_batch_10["macro assert_eq"]
    test_admissible_batch_11["macro assert_eq"]
    test_admissible_batch_12["EXIT"]
    test_admissible_batch_0 --> test_admissible_batch_1
    test_admissible_batch_1 --> test_admissible_batch_2
    test_admissible_batch_2 --> test_admissible_batch_3
    test_admissible_batch_3 --> test_admissible_batch_4
    test_admissible_batch_4 --> test_admissible_batch_5
    test_admissible_batch_5 --> test_admissible_batch_6
    test_admissible_batch_6 --> test_admissible_batch_7
    test_admissible_batch_7 --> test_admissible_batch_8
    test_admissible_batch_8 --> test_admissible_batch_9
    test_admissible_batch_9 --> test_admissible_batch_10
    test_admissible_batch_10 --> test_admissible_batch_11
    test_admissible_batch_11 --> test_admissible_batch_12
```

## Function: `test_inadmissible_batch`

- File: src/batch_admission.rs
- Branches: 0
- Loops: 0
- Nodes: 16
- Edges: 15

```mermaid
flowchart TD
    test_inadmissible_batch_0["ENTRY"]
    test_inadmissible_batch_1["let temp_dir = TempDir :: new () . unwrap ()"]
    test_inadmissible_batch_2["let artifact_path = temp_dir . path () . join ('admission_composition.json')"]
    test_inadmissible_batch_3["let batch = vec ! [create_test_signature ('action1' , vec ! [PathBuf :: from ('file1.rs')..."]
    test_inadmissible_batch_4["let decision = admit_batch (& batch , & artifact_path) . unwrap ()"]
    test_inadmissible_batch_5["macro assert"]
    test_inadmissible_batch_6["macro assert_eq"]
    test_inadmissible_batch_7["macro assert"]
    test_inadmissible_batch_8["let artifact_json = std :: fs :: read_to_string (& artifact_path) . unwrap ()"]
    test_inadmissible_batch_9["let artifact : serde_json :: Value = serde_json :: from_str (& artifact_json) . unwrap ()"]
    test_inadmissible_batch_10["macro assert_eq"]
    test_inadmissible_batch_11["macro assert_eq"]
    test_inadmissible_batch_12["let result = & artifact ['composition_result']"]
    test_inadmissible_batch_13["macro assert_eq"]
    test_inadmissible_batch_14["macro assert_eq"]
    test_inadmissible_batch_15["EXIT"]
    test_inadmissible_batch_0 --> test_inadmissible_batch_1
    test_inadmissible_batch_1 --> test_inadmissible_batch_2
    test_inadmissible_batch_2 --> test_inadmissible_batch_3
    test_inadmissible_batch_3 --> test_inadmissible_batch_4
    test_inadmissible_batch_4 --> test_inadmissible_batch_5
    test_inadmissible_batch_5 --> test_inadmissible_batch_6
    test_inadmissible_batch_6 --> test_inadmissible_batch_7
    test_inadmissible_batch_7 --> test_inadmissible_batch_8
    test_inadmissible_batch_8 --> test_inadmissible_batch_9
    test_inadmissible_batch_9 --> test_inadmissible_batch_10
    test_inadmissible_batch_10 --> test_inadmissible_batch_11
    test_inadmissible_batch_11 --> test_inadmissible_batch_12
    test_inadmissible_batch_12 --> test_inadmissible_batch_13
    test_inadmissible_batch_13 --> test_inadmissible_batch_14
    test_inadmissible_batch_14 --> test_inadmissible_batch_15
```

