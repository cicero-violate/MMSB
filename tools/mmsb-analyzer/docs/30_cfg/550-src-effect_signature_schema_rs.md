# CFG Group: src/effect_signature_schema.rs

## Function: `create_test_signature`

- File: src/effect_signature_schema.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    create_test_signature_0["ENTRY"]
    create_test_signature_1["let mut files = BTreeSet :: new ()"]
    create_test_signature_2["files . insert (PathBuf :: from (file))"]
    create_test_signature_3["EffectSignature { schema_version : SCHEMA_VERSION . to_string () , action_typ..."]
    create_test_signature_4["EXIT"]
    create_test_signature_0 --> create_test_signature_1
    create_test_signature_1 --> create_test_signature_2
    create_test_signature_2 --> create_test_signature_3
    create_test_signature_3 --> create_test_signature_4
```

## Function: `test_signature_conflict_detection`

- File: src/effect_signature_schema.rs
- Branches: 0
- Loops: 0
- Nodes: 7
- Edges: 6

```mermaid
flowchart TD
    test_signature_conflict_detection_0["ENTRY"]
    test_signature_conflict_detection_1["let sig1 = create_test_signature ('action1' , 'file1.rs')"]
    test_signature_conflict_detection_2["let sig2 = create_test_signature ('action2' , 'file2.rs')"]
    test_signature_conflict_detection_3["let sig3 = create_test_signature ('action3' , 'file1.rs')"]
    test_signature_conflict_detection_4["macro assert"]
    test_signature_conflict_detection_5["macro assert"]
    test_signature_conflict_detection_6["EXIT"]
    test_signature_conflict_detection_0 --> test_signature_conflict_detection_1
    test_signature_conflict_detection_1 --> test_signature_conflict_detection_2
    test_signature_conflict_detection_2 --> test_signature_conflict_detection_3
    test_signature_conflict_detection_3 --> test_signature_conflict_detection_4
    test_signature_conflict_detection_4 --> test_signature_conflict_detection_5
    test_signature_conflict_detection_5 --> test_signature_conflict_detection_6
```

## Function: `test_signature_validation`

- File: src/effect_signature_schema.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    test_signature_validation_0["ENTRY"]
    test_signature_validation_1["let sig = EffectSignature { schema_version : SCHEMA_VERSION . to_string () , action_typ..."]
    test_signature_validation_2["macro assert"]
    test_signature_validation_3["EXIT"]
    test_signature_validation_0 --> test_signature_validation_1
    test_signature_validation_1 --> test_signature_validation_2
    test_signature_validation_2 --> test_signature_validation_3
```

