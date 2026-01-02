# CFG Group: src/620_correction_plan_serializer.rs

## Function: `serialize_check`

- File: src/620_correction_plan_serializer.rs
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

## Function: `serialize_correction_plan`

- File: src/620_correction_plan_serializer.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    serialize_correction_plan_0["ENTRY"]
    serialize_correction_plan_1["json ! ({ 'action_id' : plan . action_id , 'tier' : format ! ('{:?}' , plan ...."]
    serialize_correction_plan_2["EXIT"]
    serialize_correction_plan_0 --> serialize_correction_plan_1
    serialize_correction_plan_1 --> serialize_correction_plan_2
```

## Function: `serialize_correction_plans`

- File: src/620_correction_plan_serializer.rs
- Branches: 0
- Loops: 0
- Nodes: 4
- Edges: 3

```mermaid
flowchart TD
    serialize_correction_plans_0["ENTRY"]
    serialize_correction_plans_1["let items = report . correction_plans . iter () . zip (report . verification_policies . i..."]
    serialize_correction_plans_2["json ! ({ 'version' : report . version , 'timestamp' : report . timestamp , '..."]
    serialize_correction_plans_3["EXIT"]
    serialize_correction_plans_0 --> serialize_correction_plans_1
    serialize_correction_plans_1 --> serialize_correction_plans_2
    serialize_correction_plans_2 --> serialize_correction_plans_3
```

## Function: `serialize_scope`

- File: src/620_correction_plan_serializer.rs
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

## Function: `serialize_strategy`

- File: src/620_correction_plan_serializer.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    serialize_strategy_0["ENTRY"]
    serialize_strategy_1["match strategy { CorrectionStrategy :: AddImport { module_path , symbol } => ..."]
    serialize_strategy_2["EXIT"]
    serialize_strategy_0 --> serialize_strategy_1
    serialize_strategy_1 --> serialize_strategy_2
```

## Function: `serialize_thresholds`

- File: src/620_correction_plan_serializer.rs
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

## Function: `write_intelligence_outputs`

- File: src/620_correction_plan_serializer.rs
- Branches: 0
- Loops: 0
- Nodes: 3
- Edges: 2

```mermaid
flowchart TD
    write_intelligence_outputs_0["ENTRY"]
    write_intelligence_outputs_1["write_intelligence_outputs_at (report , output_dir , None , None)"]
    write_intelligence_outputs_2["EXIT"]
    write_intelligence_outputs_0 --> write_intelligence_outputs_1
    write_intelligence_outputs_1 --> write_intelligence_outputs_2
```

## Function: `write_intelligence_outputs_at`

- File: src/620_correction_plan_serializer.rs
- Branches: 2
- Loops: 0
- Nodes: 19
- Edges: 20

```mermaid
flowchart TD
    write_intelligence_outputs_at_0["ENTRY"]
    write_intelligence_outputs_at_1["std :: fs :: create_dir_all (output_dir) ?"]
    write_intelligence_outputs_at_2["let json_path = correction_json . map (| p | p . to_path_buf ()) . unwrap_or_else (| | output..."]
    write_intelligence_outputs_at_3["if let Some (parent) = json_path . parent ()"]
    write_intelligence_outputs_at_4["THEN BB"]
    write_intelligence_outputs_at_5["std :: fs :: create_dir_all (parent) ?"]
    write_intelligence_outputs_at_6["EMPTY ELSE"]
    write_intelligence_outputs_at_7["IF JOIN"]
    write_intelligence_outputs_at_8["let contract = serialize_correction_plans (report)"]
    write_intelligence_outputs_at_9["std :: fs :: write (& json_path , serde_json :: to_string_pretty (& contract)..."]
    write_intelligence_outputs_at_10["let policy_path = verification_policy_json . map (| p | p . to_path_buf ()) . unwrap_or_else (|..."]
    write_intelligence_outputs_at_11["if let Some (parent) = policy_path . parent ()"]
    write_intelligence_outputs_at_12["THEN BB"]
    write_intelligence_outputs_at_13["std :: fs :: create_dir_all (parent) ?"]
    write_intelligence_outputs_at_14["EMPTY ELSE"]
    write_intelligence_outputs_at_15["IF JOIN"]
    write_intelligence_outputs_at_16["emit_verification_policy (& report . verification_policies , & policy_path) ?"]
    write_intelligence_outputs_at_17["Ok (())"]
    write_intelligence_outputs_at_18["EXIT"]
    write_intelligence_outputs_at_0 --> write_intelligence_outputs_at_1
    write_intelligence_outputs_at_1 --> write_intelligence_outputs_at_2
    write_intelligence_outputs_at_2 --> write_intelligence_outputs_at_3
    write_intelligence_outputs_at_3 --> write_intelligence_outputs_at_4
    write_intelligence_outputs_at_4 --> write_intelligence_outputs_at_5
    write_intelligence_outputs_at_3 --> write_intelligence_outputs_at_6
    write_intelligence_outputs_at_5 --> write_intelligence_outputs_at_7
    write_intelligence_outputs_at_6 --> write_intelligence_outputs_at_7
    write_intelligence_outputs_at_7 --> write_intelligence_outputs_at_8
    write_intelligence_outputs_at_8 --> write_intelligence_outputs_at_9
    write_intelligence_outputs_at_9 --> write_intelligence_outputs_at_10
    write_intelligence_outputs_at_10 --> write_intelligence_outputs_at_11
    write_intelligence_outputs_at_11 --> write_intelligence_outputs_at_12
    write_intelligence_outputs_at_12 --> write_intelligence_outputs_at_13
    write_intelligence_outputs_at_11 --> write_intelligence_outputs_at_14
    write_intelligence_outputs_at_13 --> write_intelligence_outputs_at_15
    write_intelligence_outputs_at_14 --> write_intelligence_outputs_at_15
    write_intelligence_outputs_at_15 --> write_intelligence_outputs_at_16
    write_intelligence_outputs_at_16 --> write_intelligence_outputs_at_17
    write_intelligence_outputs_at_17 --> write_intelligence_outputs_at_18
```

