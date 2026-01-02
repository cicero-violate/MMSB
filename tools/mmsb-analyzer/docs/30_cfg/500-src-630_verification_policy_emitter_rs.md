# CFG Group: src/630_verification_policy_emitter.rs

## Function: `emit_verification_policy`

- File: src/630_verification_policy_emitter.rs
- Branches: 0
- Loops: 0
- Nodes: 5
- Edges: 4

```mermaid
flowchart TD
    emit_verification_policy_0["ENTRY"]
    emit_verification_policy_1["let policy_file = json ! ({ 'version' : '1.0' , 'policies' : policies . iter () . map (| p | js..."]
    emit_verification_policy_2["std :: fs :: write (output_path , serde_json :: to_string_pretty (& policy_fi..."]
    emit_verification_policy_3["Ok (())"]
    emit_verification_policy_4["EXIT"]
    emit_verification_policy_0 --> emit_verification_policy_1
    emit_verification_policy_1 --> emit_verification_policy_2
    emit_verification_policy_2 --> emit_verification_policy_3
    emit_verification_policy_3 --> emit_verification_policy_4
```

## Function: `serialize_check`

- File: src/630_verification_policy_emitter.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    serialize_check_0["ENTRY"]
    serialize_check_1["match check { VerificationCheck :: CargoCheck => json ! ({ 'type' : 'CargoChe..."]
    serialize_check_2["EXIT"]
    serialize_check_0 --> serialize_check_1
    serialize_check_1 --> serialize_check_2
```

## Function: `serialize_scope`

- File: src/630_verification_policy_emitter.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    serialize_scope_0["ENTRY"]
    serialize_scope_1["match scope { VerificationScope :: SyntaxOnly { files } => json ! ({ 'type' :..."]
    serialize_scope_2["EXIT"]
    serialize_scope_0 --> serialize_scope_1
    serialize_scope_1 --> serialize_scope_2
```

## Function: `serialize_thresholds`

- File: src/630_verification_policy_emitter.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    serialize_thresholds_0["ENTRY"]
    serialize_thresholds_1["json ! ({ 'min_cohesion_delta' : thresholds . min_cohesion_delta , 'max_viola..."]
    serialize_thresholds_2["EXIT"]
    serialize_thresholds_0 --> serialize_thresholds_1
    serialize_thresholds_1 --> serialize_thresholds_2
```

